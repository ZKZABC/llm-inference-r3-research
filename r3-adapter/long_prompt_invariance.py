#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长 prompt 批不变性实验

背景：`gemm_gate.py` 在 A10 上测出 GEMM 的分界点在 M=128——
M≤64 走一条 kernel，M≥128 走另一条（35.9% 元素差 4 ulp）。

而 vLLM 的批不变性实验用的是 `"Hi"` 这种 1~4 token 的 prompt，
恰好落在安全区。真实 RL 场景的 prompt 是几百 token，正好跨分界。

这个脚本直接验证那个风险：

    A 单独跑   → prefill M = len(A)
    A + B 同批 → prefill M = len(A) + len(B)

若两者落在不同 kernel，A 的路由就会变——**加一条无关的短 prompt，
改变了另一条 prompt 的路由**。

这是 R3 在真实场景下最隐蔽的失效方式。

用法:
    python long_prompt_invariance.py --port 8000
    python long_prompt_invariance.py --port 8000 --lengths 60,120,127,128,200
    python long_prompt_invariance.py --port 8000 --budget 20

退出码: 0 = 全部一致, 1 = 存在分歧
"""

import argparse
import base64
import io
import json
import sys
import urllib.request

import numpy as np


def _post(url, payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(host, given):
    if given:
        return given
    try:
        req = urllib.request.Request(f"{host}/v1/models")
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read().decode("utf-8")).get("data", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return "default"


def token_count(host, model, text):
    """用 /tokenize 拿精确 token 数；不可用则返回 None"""
    try:
        r = _post(f"{host}/tokenize", {"model": model, "prompt": text})
        return int(r.get("count", 0)) or None
    except Exception:
        return None


# 一段中性文本，重复拼接后 token 数近似线性增长
_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the engineer "
    "carefully measures latency across every layer of the model. "
)


def build_prompt(host, model, target_tokens, tag="A", tries=12):
    """
    构造接近 target_tokens 的 prompt。

    优先用 /tokenize 做精确校准；不可用时按 15 token/句 估算，
    实际长度从返回的 routed_experts 形状反推。
    """
    lo, hi = 1, max(2, target_tokens // 8 + 40)
    best = _SENTENCE * lo

    for _ in range(tries):
        mid = (lo + hi) // 2
        cand = (_SENTENCE * mid) + f"[{tag}]"
        n = token_count(host, model, cand)
        if n is None:
            return _SENTENCE * max(1, round(target_tokens / 15)) + f"[{tag}]", target_tokens
        if n < target_tokens:
            lo = mid + 1
            best = cand
        else:
            hi = mid - 1
            if abs(n - target_tokens) <= 2:
                return cand, n
    return best, token_count(host, model, best) or target_tokens


def run_batch(host, model, prompts, max_tokens=1):
    """发一次请求，返回每个 prompt 的 routed_experts"""
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.0,
    }
    data = _post(f"{host}/v1/completions", payload)
    out = []
    for ch in data["choices"]:
        raw = ch.get("routed_experts")
        if raw is None:
            raise RuntimeError("响应里没有 routed_experts，检查 --enable-return-routed-experts")
        out.append(np.load(io.BytesIO(base64.b64decode(raw))))
    return out


def diff_rows(a, b):
    """返回 (不同元素数, 总元素数, 出现差异的行号列表)"""
    n = min(a.shape[0], b.shape[0])
    x, y = a[:n], b[:n]
    neq = x != y
    rows = np.where(neq.any(axis=tuple(range(1, neq.ndim))))[0]
    return int(neq.sum()), int(neq.size), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--lengths", default="60,100,120,126,127,128,129,130,200,512")
    ap.add_argument("--partner", type=int, default=10, help="B 的长度")
    ap.add_argument("--budget", type=int, default=0, help="最多跑几个长度（0=全跑）")
    args = ap.parse_args()

    host = f"http://localhost:{args.port}" if args.port else args.host
    model = resolve_model(host, args.model)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if args.budget:
        lengths = lengths[: args.budget]

    print("=" * 70)
    print("长 prompt 批不变性实验")
    print("=" * 70)
    print(f"\n  host       {host}")
    print(f"  model      {model}")
    print(f"  长度梯度   {lengths}")
    print(f"  B 的长度   {args.partner}")

    # 伙伴 prompt 只构造一次，保证两轮用同一条
    partner_text, partner_n = build_prompt(host, model, args.partner, tag="B")
    print(f"  B 实际     {partner_n} token")

    print("\n实验：A 单独跑 vs A 与 B 同批")
    print("─" * 70)
    print(f"  {'A长度':>6} {'同批M':>7} {'不一致行':>10} {'元素差异':>12}  结果")
    print("  " + "-" * 66)

    results = []
    n_fail = 0

    for target in lengths:
        a_text, a_n = build_prompt(host, model, target, tag=f"L{target}")
        try:
            single = run_batch(host, model, [a_text])[0]
            batched = run_batch(host, model, [a_text, partner_text])[0]
        except Exception as e:
            print(f"  {target:>6} 请求失败: {type(e).__name__}: {str(e)[:50]}")
            continue

        actual_a = single.shape[0]
        m_batch = actual_a + partner_n
        nd, tot, rows = diff_rows(single, batched)

        ok = nd == 0
        if not ok:
            n_fail += 1
        tag = "一致" if ok else f"分歧 行{list(rows[:3])}"
        print(
            f"  {actual_a:>6} {m_batch:>7} {len(rows):>10} {nd:>7}/{tot:<6} {tag}"
        )

        results.append(
            {
                "target": target,
                "actual_A": int(actual_a),
                "batch_M": int(m_batch),
                "n_diff": nd,
                "rows_with_diff": [int(r) for r in rows[:20]],
                "ok": bool(ok),
            }
        )

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if n_fail == 0:
        print("  所有长度上，加一条无关 prompt 都没有改变 A 的路由。")
        print("  → 当前配置下批不变性成立。")
    else:
        ok_rows = [r for r in results if r["ok"]]
        bad_rows = [r for r in results if not r["ok"]]
        print(f"  {n_fail}/{len(results)} 个长度出现分歧。")
        if ok_rows and bad_rows:
            max_ok = max(r["batch_M"] for r in ok_rows)
            min_bad = min(r["batch_M"] for r in bad_rows)
            print(f"  最后一次一致时的同批 M = {max_ok}")
            print(f"  最早出现分歧时的同批 M = {min_bad}")
            if min_bad > max_ok:
                print(f"  → 分界落在 M ∈ ({max_ok}, {min_bad}] 之间")
        print("\n  这意味着：加一条无关的短 prompt，会改变另一条 prompt 的路由。")
        print("  在 RL rollout 里，batch 组成是动态的——记录的路由无法稳定复现。")
        print("\n  处置：")
        print("    1. 确认 VLLM_BATCH_INVARIANT=1 已开启（本脚本不负责开）")
        print("    2. 若已开启仍分歧，说明该开关没覆盖到 M≥128 的路径")
        print("    3. 只能把 prefill 的 M 固定到单一 bucket（padding 到固定长度）")

    print()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
