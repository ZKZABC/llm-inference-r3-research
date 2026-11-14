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
    随机基线 = k/E（两个 k 元子集随机相交的期望）
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

    mean_ov = float(np.mean(overlaps))
    baseline = k / num_experts

    return {
        "mean_overlap": round(mean_ov, 4),
        "random_baseline": round(baseline, 4),
        "ratio_vs_random": round(mean_ov / baseline, 2) if baseline > 0 else None,
        "n_pairs": len(overlaps),
    }


def semantic_locality(per_prompt_routes, num_experts):
    """
    同一 prompt 内的专家重叠 vs 跨 prompt 的重叠。

    若同 prompt 明显更高 → 存在语义局部性（可以按 workload 做缓存）
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

    for p in range(len(per_prompt_routes) - 1):
        a = per_prompt_routes[p]
        b = per_prompt_routes[p + 1]
        L = min(a.shape[1], b.shape[1])
        for layer in range(L):
            sa = set(a[:, layer, :].reshape(-1).tolist())
            sb = set(b[:, layer, :].reshape(-1).tolist())
            across.append(len(sa & sb) / max(len(sa | sb), 1))

    return {
        "within_prompt_overlap": round(float(np.mean(within)), 4) if within else None,
        "across_prompt_jaccard": round(float(np.mean(across)), 4) if across else None,
        "note": "within 是相邻 token 的 k 元交集比例；across 是两个 prompt 专家集合的 Jaccard",
    }


def capacity_stats(routes, num_experts, ks):
    """
    模拟 batch 推理：取 M 个 token 一批，统计每个 expert 的负载。

    返回不同 M 下「最忙 expert / 期望值」的倍数。
    这个倍数就是 capacity_factor 至少要取到多少。
    """
    N, L, k = routes.shape
    out = []

    for m in ks:
        if m > N:
            continue
        worst_ratios = []
        worst_over_mean = 0.0
        for layer in range(L):
            seg = routes[:m, layer, :].reshape(-1)
            counts = np.bincount(seg, minlength=num_experts).astype(float)
            mean = counts.mean()
            if mean <= 0:
                continue
            ratio = counts.max() / mean
            worst_ratios.append(ratio)
            worst_over_mean = max(worst_over_mean, ratio)

        out.append(
            {
                "M": int(m),
                "expected_per_expert": round(m * k / num_experts, 2),
                "max_over_mean_layer_avg": round(float(np.mean(worst_ratios)), 2) if worst_ratios else None,
                "max_over_mean_worst": round(float(worst_over_mean), 2),
            }
        )
    return out


def interpret(cov, dist, temp, sem, cap):
    """把数字翻译成决策"""
    lines = []

    last = cov[-1] if cov else None
    if last:
        ratio = last["actual_unique"] / max(last["uniform_expected"], 1e-9)
        if ratio > 0.97:
            lines.append(f"覆盖度 {ratio:.1%} 于随机基线 → 无局部性，不要做专家缓存")
        elif ratio > 0.85:
            lines.append(f"覆盖度 {ratio:.1%} 于随机基线 → 弱局部性，只能做软分层")
        else:
            lines.append(f"覆盖度 {ratio:.1%} 于随机基线 → 存在明显局部性，可考虑分级缓存")

    if dist["cv"] < 0.1:
        lines.append(f"分布变异系数 {dist['cv']} → 接近均匀，load balancing 生效良好")
    else:
        lines.append(f"分布变异系数 {dist['cv']} → 有明显偏斜，最忙 expert 是均值的 {dist['max_over_mean']}x")

    t = temp.get("ratio_vs_random")
    if t is not None:
        if t < 1.2:
            lines.append(f"相邻 token 重叠 {t}x 随机 → 无时间局部性，预取无收益")
        else:
            lines.append(f"相邻 token 重叠 {t}x 随机 → 有时间局部性，可做权重预取")

    w = sem.get("within_prompt_overlap")
    a = sem.get("across_prompt_jaccard")
    if w is not None and a is not None and a > 0:
        lines.append(f"同 prompt 内 {w} vs 跨 prompt {a} → 比值 {w / max(a, 1e-9):.1f}x")

    if cap:
        worst = max(c["max_over_mean_worst"] for c in cap)
        lines.append(f"capacity 峰值倍数 {worst:.2f}x → capacity_factor 至少取 {worst + 0.5:.1f}")

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
    cap = capacity_stats(all_routes, E, [8, 16, 32, 64, 128, 256, 512])

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
    print(f"  相邻 token 重叠 {temp['mean_overlap']} vs 随机 {temp['random_baseline']} → {temp['ratio_vs_random']}x")

    print("\n④ 语义局部性")
    print("─" * 68)
    print(f"  同 prompt 内 {sem['within_prompt_overlap']}")
    print(f"  跨 prompt   {sem['across_prompt_jaccard']}")

    print("\n⑤ capacity 分布")
    print("─" * 68)
    print(f"  {'M':>5} {'期望/专家':>10} {'峰值/均值(层均)':>16} {'峰值/均值(最差)':>16}")
    for c in cap:
        print(f"  {c['M']:>5} {c['expected_per_expert']:>10} {c['max_over_mean_layer_avg']:>16} {c['max_over_mean_worst']:>16}")

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
