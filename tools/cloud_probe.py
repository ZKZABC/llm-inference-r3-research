#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端探测：真实模型的路由分布

回答四个只有真机能答的问题：

  1. 路由到底有没有局部性？
     —— 不是看 unique=60，而是看「覆盖曲线 vs 均匀随机基线」的偏离
  2. capacity_factor 该取多少？
     —— 看不同 token 数下，最忙 expert 相对期望值的倍数
  3. 时间局部性存在吗？
     —— 相邻 token 选同一专家的比例，对比随机基线 k/E
  4. 语义局部性存在吗？
     —— 同 prompt 内 vs 跨 prompt 的专家重叠

前置：vLLM 已用 --enable-return-routed-experts 启动
      VLLM_BATCH_INVARIANT=1（可复现性）

用法:
    python cloud_probe.py --port 8000
    python cloud_probe.py --n-gen 64 --out probe.json

输出: 控制台报告 + JSON（供后续脚本消费）
"""

import argparse
import base64
import io
import json
import sys
import urllib.request
from collections import Counter

import numpy as np


def _post(url, payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# 12 个不同领域的 prompt，让路由分布覆盖尽可能多的语义区域
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    return",
    "请解释一下量子纠缠的基本原理，用简单的语言描述",
    "In 2024, the global semiconductor market reached approximately",
    "证明：对于任意正整数 n，n^3 - n 能被 6 整除。证明过程如下：",
    "The patient presented with acute abdominal pain and fever. The differential diagnosis includes",
    "写一首关于秋天的五言绝句，要求押平声韵：",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE",
    "The transformer architecture relies on self-attention mechanisms that compute",
    "将下面这段话翻译成英文：人工智能正在改变我们的生活方式",
    "The financial statement shows revenue growth of 12% year-over-year, driven primarily by",
    "机器学习模型的泛化能力取决于训练数据的分布是否能够代表真实场景，因此",
]


def resolve_model(host, given):
    """vLLM 的 model 字段必须匹配启动时的名字，否则 404。自动探测一次。"""
    if given:
        return given
    try:
        items = _get(f"{host}/v1/models").get("data", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return "default"


def fetch_routes(host, prompt, max_tokens, model_name=None):
    """发一次 completions 请求，拿 routed_experts"""
    url = f"{host}/v1/completions"
    payload = {
        "model": model_name or "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.0,
    }
    data = _post(url, payload)
    choice = data["choices"][0]

    if "routed_experts" not in choice:
        raise RuntimeError(
            "响应里没有 routed_experts。确认启动时带了 --enable-return-routed-experts"
        )

    raw = base64.b64decode(choice["routed_experts"])
    return np.load(io.BytesIO(raw)), len(choice.get("text", ""))


def coverage_curve(routes, num_experts):
    """
    覆盖曲线：前 n 个 token 覆盖了多少 expert。
    同时给出「纯均匀随机」的理论基线。

    routes: [N, L, k]
    """
    N, L, k = routes.shape
    flat = routes.reshape(N, L * k)

    ns = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= N]
    if not ns:
        ns = [N]

    rows = []
    for n in ns:
        seg = flat[:n]
        actual = len(np.unique(seg))
        slots = n * L * k
        expect = num_experts * (1.0 - (1.0 - 1.0 / num_experts) ** slots)
        rows.append(
            {
                "n_tokens": int(n),
                "slots": int(slots),
                "actual_unique": int(actual),
                "uniform_expected": round(float(expect), 2),
                "gap": round(float(expect - actual), 2),
            }
        )
    return rows


def distribution_shape(routes, num_experts, top_n=10):
    """每个 expert 的激活次数分布：均匀的话 top-10 占 10/E"""
    flat = routes.reshape(-1)
    counts = np.bincount(flat, minlength=num_experts).astype(float)
    total = counts.sum()
    order = np.argsort(-counts)

    top_share = counts[order[:top_n]].sum() / total
    uniform_share = min(top_n, num_experts) / num_experts

    return {
        "total_activations": int(total),
        "mean_per_expert": round(float(total / num_experts), 1),
        "std_per_expert": round(float(counts.std()), 1),
        "cv": round(float(counts.std() / max(counts.mean(), 1e-9)), 3),
        "top10_share": round(float(top_share), 4),
        "uniform_top10_share": round(float(uniform_share), 4),
        "max_over_mean": round(float(counts.max() / max(counts.mean(), 1e-9)), 2),
        "counts_sorted": [int(c) for c in counts[order]],
    }


def temporal_locality(routes, num_experts):
    """
    相邻 token 在同一层的专家重叠率。
    随机基线 = k/E

    必须看中位数和零值比例：均值会被少数「连续相同段」拉高，
    而实际逐 token 可能是点状抖动（大量 0）。
    """
    N, L, k = routes.shape
    if N < 2:
        return {"note": "token 太少"}

    overlaps = []
    for layer in range(L):
        for i in range(N - 1):
            a = set(routes[i, layer].tolist())
            b = set(routes[i + 1, layer].tolist())
            overlaps.append(len(a & b) / k)

    arr = np.array(overlaps)
    mean_ov = float(arr.mean())
    baseline = k / num_experts

    return {
        "mean_overlap": round(mean_ov, 4),
        "median_overlap": round(float(np.median(arr)), 4),
        "zero_ratio": round(float((arr == 0).mean()), 4),
        "random_baseline": round(baseline, 4),
        "ratio_vs_random": round(mean_ov / baseline, 2) if baseline > 0 else None,
        "n_pairs": len(overlaps),
    }


def semantic_locality(per_prompt_routes, num_experts):
    """
    同 prompt 相邻 token vs 跨 prompt 相邻 token 的重叠率。

    两者必须是同一个量纲，否则不可比。
    早期版本用「专家集合的 Jaccard」算跨 prompt，
    但每个 prompt 都覆盖全部专家，那个恒等于 1，毫无意义。
    """
    k = per_prompt_routes[0].shape[-1]
    within, across = [], []

    for routes in per_prompt_routes:
        N, L, _ = routes.shape
        for layer in range(L):
            for i in range(N - 1):
                a = set(routes[i, layer].tolist())
                b = set(routes[i + 1, layer].tolist())
                within.append(len(a & b) / k)

    # 跨 prompt：A 的末尾 token vs B 的开头 token，同为「相邻」语义
    for p in range(len(per_prompt_routes) - 1):
        a_routes = per_prompt_routes[p]
        b_routes = per_prompt_routes[p + 1]
        L = min(a_routes.shape[1], b_routes.shape[1])
        for layer in range(L):
            sa = set(a_routes[-1, layer].tolist())
            sb = set(b_routes[0, layer].tolist())
            across.append(len(sa & sb) / k)

    w = float(np.mean(within)) if within else 0.0
    a = float(np.mean(across)) if across else 0.0

    return {
        "within_prompt_overlap": round(w, 4),
        "across_prompt_overlap": round(a, 4),
        "ratio": round(w / a, 2) if a > 0 else None,
        "note": "两者都是相邻 token 的 k 元交集比例；ratio > 1 说明同 prompt 内更集中",
    }


def _sample_batch(per_prompt, m, mode, rng):
    """
    取一个 batch 的路由切片 [m, L, k]。

    continuous —— 连续 m 个 token（同一 prompt 内），对应 prefill
    random     —— 每个 token 来自不同 prompt，对应 decode
    """
    L = per_prompt[0].shape[1]
    k = per_prompt[0].shape[2]
    dtype = per_prompt[0].dtype

    if mode == "continuous":
        for r in per_prompt:
            if r.shape[0] >= m:
                s = int(rng.integers(0, r.shape[0] - m + 1))
                return r[s : s + m]
        allr = np.concatenate(per_prompt, axis=0)
        s = int(rng.integers(0, max(1, allr.shape[0] - m + 1)))
        return allr[s : s + m]

    out = np.zeros((m, L, k), dtype=dtype)
    for i in range(m):
        p = int(rng.integers(0, len(per_prompt)))
        t = int(rng.integers(0, per_prompt[p].shape[0]))
        out[i] = per_prompt[p][t]
    return out


def _peak_counts(batch, num_experts):
    """一个 batch 内，逐层统计最忙 expert 的负载，返回最大值"""
    L = batch.shape[1]
    peak = 0
    peak_ratio = 0.0
    for layer in range(L):
        seg = batch[:, layer, :].reshape(-1)
        counts = np.bincount(seg, minlength=num_experts).astype(float)
        mean = counts.mean()
        if mean <= 0:
            continue
        if counts.max() > peak:
            peak = float(counts.max())
            peak_ratio = float(counts.max() / mean)
    return peak, peak_ratio


def capacity_stats(per_prompt, num_experts, ks, n_trials=20):
    """
    模拟 batch 推理时每个 expert 的负载。

    两种采样模式对两个真实阶段：
      continuous —— prefill（同一 prompt 的连续 token，语义相关）
      random     —— decode（不同请求各 1 个 token，语义互不相关）

    倍数会随 M 变小而爆炸（分母太小），所以同时给出峰值绝对值和建议 C。
    peak 取多次采样的最大值（真实 batch 组成是随机的，要看最坏情况）。
    """
    rng = np.random.default_rng(0)
    k = per_prompt[0].shape[2]
    out = []

    for m in ks:
        row = {"M": int(m), "expected_per_expert": round(m * k / num_experts, 2)}

        for mode in ("continuous", "random"):
            peaks, ratios = [], []
            ok = True
            for _ in range(n_trials):
                try:
                    batch = _sample_batch(per_prompt, m, mode, rng)
                except Exception:
                    ok = False
                    break
                p, r = _peak_counts(batch, num_experts)
                peaks.append(p)
                ratios.append(r)
            if not ok or not peaks:
                continue
            row[f"{mode}_peak_max"] = int(max(peaks))
            row[f"{mode}_peak_avg"] = round(float(np.mean(peaks)), 1)
            row[f"{mode}_ratio"] = round(float(np.mean(ratios)), 2)
            row[f"{mode}_C"] = int(np.ceil(max(peaks) * 1.15))

        if "continuous_peak_max" in row or "random_peak_max" in row:
            out.append(row)
    return out


def interpret(cov, dist, temp, sem, cap):
    """把数字翻译成决策，且不做过强的断言"""
    lines = []

    last = cov[-1] if cov else None
    if last:
        ratio = last["actual_unique"] / max(last["uniform_expected"], 1e-9)
        lines.append(
            f"覆盖度 {ratio:.1%} 于随机基线 —— 但 unique 是基数不是分布，"
            "这一项不能单独用于判断局部性"
        )

    if dist["cv"] < 0.1:
        lines.append(
            f"边际分布均匀：cv={dist['cv']}，top-10 {dist['top10_share']:.1%}"
            f"（均匀 {dist['uniform_top10_share']:.1%}），最忙/均值 {dist['max_over_mean']}x"
        )
    else:
        lines.append(
            f"边际分布有偏斜：cv={dist['cv']}，最忙/均值 {dist['max_over_mean']}x"
        )

    t = temp.get("ratio_vs_random")
    if t is not None:
        med = temp.get("median_overlap", 0)
        zero = temp.get("zero_ratio", 0)
        if t < 1.2:
            lines.append(f"相邻 token 重叠 {t}x 随机 → 无时间局部性")
        elif med == 0 or zero > 0.5:
            lines.append(
                f"相邻 token 重叠 {t}x 随机，但中位数 {med}、零值占 {zero:.0%}"
                " → 少数连续段拉高均值，逐 token 是点状抖动，预取收益有限"
            )
        else:
            lines.append(
                f"相邻 token 重叠 {t}x 随机，中位数 {med} → 有稳定时间局部性，预取有价值"
            )

    r = sem.get("ratio")
    if r is not None:
        if r > 1.5:
            lines.append(f"同 prompt 内比跨 prompt 高 {r}x → 存在语义局部性")
        else:
            lines.append(f"同 prompt vs 跨 prompt 比值 {r}x → 语义局部性不强")

    if cap:
        lines.append("capacity 要分两个阶段看：")
        for c in cap:
            if c["M"] in (8, 32, 128, 512):
                lines.append(
                    f"  M={c['M']:<4} prefill C={c.get('continuous_C', '-')!s:<5} "
                    f"decode C={c.get('random_C', '-')}"
                )
        lines.append("prefill 的 token 同源、语义相关 → 路由集中，C 要大")
        lines.append("decode 的 token 跨请求 → 路由分散，C 可小；统一按 prefill 取更安全")

    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=None, help="等价于 --host http://localhost:PORT")
    ap.add_argument("--n-gen", type=int, default=48)
    ap.add_argument("--num-experts", type=int, default=60)
    ap.add_argument("--model", default="", help="留空则自动从 /v1/models 探测")
    ap.add_argument("--out", default="probe_result.json")
    args = ap.parse_args()

    host = f"http://localhost:{args.port}" if args.port else args.host
    E = args.num_experts
    model_name = resolve_model(host, args.model)

    print("=" * 68)
    print("云端探测 · 真实模型路由分布")
    print("=" * 68)
    print(f"\n  host        {host}")
    print(f"  model       {model_name}")
    print(f"  prompts     {len(PROMPTS)}")
    print(f"  n_gen       {args.n_gen}")
    print(f"  experts     {E}")

    per_prompt = []
    failed = 0
    for i, p in enumerate(PROMPTS):
        try:
            routes, ntxt = fetch_routes(host, p, args.n_gen, model_name)
            per_prompt.append(routes)
            print(f"  [{i + 1}/{len(PROMPTS)}] shape={routes.shape} dtype={routes.dtype} unique={len(np.unique(routes))}")
        except Exception as e:
            failed += 1
            print(f"  [{i + 1}/{len(PROMPTS)}] 失败: {type(e).__name__}: {str(e)[:80]}")

    if not per_prompt:
        print("\n没有拿到任何路由数据，检查 vLLM 是否启动、是否带了 --enable-return-routed-experts")
        return 1

    all_routes = np.concatenate(per_prompt, axis=0)
    k = all_routes.shape[-1]
    E = max(E, int(all_routes.max()) + 1)

    print(f"\n合计 {all_routes.shape[0]} 个 token × {all_routes.shape[1]} 层 × {k} 专家")

    cov = coverage_curve(all_routes, E)
    dist = distribution_shape(all_routes, E)
    temp = temporal_locality(all_routes, E)
    sem = semantic_locality(per_prompt, E)
    cap = capacity_stats(per_prompt, E, [8, 16, 32, 64, 128, 256, 512])

    print("\n① 覆盖曲线（vs 纯均匀随机）")
    print("─" * 68)
    print(f"  {'tokens':>7} {'槽位':>9} {'实际':>6} {'随机期望':>9} {'差距':>7}")
    for r in cov:
        print(f"  {r['n_tokens']:>7} {r['slots']:>9} {r['actual_unique']:>6} {r['uniform_expected']:>9} {r['gap']:>7}")

    print("\n② 分布形状")
    print("─" * 68)
    print(f"  总激活 {dist['total_activations']}, 每专家均值 {dist['mean_per_expert']}")
    print(f"  标准差 {dist['std_per_expert']}, 变异系数 {dist['cv']}")
    print(f"  top-10 占比 {dist['top10_share']:.2%}（均匀应为 {dist['uniform_top10_share']:.2%}）")
    print(f"  最忙/均值 {dist['max_over_mean']}x")

    print("\n③ 时间局部性")
    print("─" * 68)
    if "median_overlap" in temp:
        print(f"  相邻 token 重叠  均值 {temp['mean_overlap']}  中位数 {temp['median_overlap']}")
        print(f"  零重叠占比       {temp['zero_ratio']:.1%}")
        print(f"  随机基线         {temp['random_baseline']}  → {temp['ratio_vs_random']}x")
    else:
        print(f"  {temp.get('note', '无数据')}")

    print("\n④ 语义局部性")
    print("─" * 68)
    print(f"  同 prompt 相邻   {sem['within_prompt_overlap']}")
    print(f"  跨 prompt 相邻   {sem['across_prompt_overlap']}")
    print(f"  比值             {sem['ratio']}x（>1 说明同 prompt 内更集中）")

    print("\n⑤ capacity 分布")
    print("─" * 68)
    print("  prefill = 连续 token（同一 prompt）   decode = 跨请求各取 1 个")
    print()
    print(f"  {'M':>5} {'期望':>7} | {'prefill峰值':>11} {'建议C':>6} | {'decode峰值':>10} {'建议C':>6}")
    for c in cap:
        pc = c.get("continuous_peak_max", "-")
        pcc = c.get("continuous_C", "-")
        rc = c.get("random_peak_max", "-")
        rcc = c.get("random_C", "-")
        print(f"  {c['M']:>5} {c['expected_per_expert']:>7} | {pc:>11} {pcc:>6} | {rc:>10} {rcc:>6}")

    lines = interpret(cov, dist, temp, sem, cap)
    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    for ln in lines:
        print(f"  · {ln}")

    result = {
        "meta": {
            "n_prompts": len(PROMPTS),
            "failed": failed,
            "n_tokens": int(all_routes.shape[0]),
            "n_layers": int(all_routes.shape[1]),
            "top_k": int(k),
            "num_experts": int(E),
        },
        "coverage": cov,
        "distribution": dist,
        "temporal": temp,
        "semantic": sem,
        "capacity": cap,
        "interpretation": lines,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
