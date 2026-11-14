#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨引擎路由对比：vLLM vs transformers

你报告 §7 把「跨引擎一致性」标为「高难度」。这个脚本不追求一致——
不同 kernel 不可能完全一致——而是**量化差异有多大**。

这个数决定 R3 的价值定位：

    一致率 99%  →  R3 修的是那 1%，价值有限但精确
    一致率 70%  →  R3 在重建 30% 的路由，这才是主要矛盾
    一致率 50%  →  两个引擎基本在两个世界里，"路由"这个概念本身要重新定义

对比方式：只比 prefill（max_tokens=1），避开生成阶段的对齐问题。

用法:
    python cross_engine_routes.py --port 8000
    python cross_engine_routes.py --port 8000 --prompt "The capital of France is"
    python cross_engine_routes.py --port 8000 --prompts-file prompts.txt
"""

import argparse
import base64
import io
import json
import sys
import urllib.request

import numpy as np
import torch

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "请解释一下量子纠缠的基本原理",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id",
    "The patient presented with acute abdominal pain and fever. The differential diagnosis includes",
    "证明：对于任意正整数 n，n^3 - n 能被 6 整除。证明过程如下：",
    "写一首关于秋天的五言绝句，要求押平声韵：",
    "The transformer architecture relies on self-attention mechanisms that compute",
    "将下面这段话翻译成英文：人工智能正在改变我们的生活方式",
    "Revenue recognition under ASC 606 requires the entity to identify",
    "机器学习模型的泛化能力取决于训练数据的分布是否能够代表真实场景",
    "fn main() { let mut v: Vec<i32> = Vec::new(); v.push(1); println!(\"{:?}\", v); }",
]


def _post(url, payload, timeout=300):
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


def vllm_routes(host, model, prompt):
    """只跑 prefill（max_tokens=1），返回 [prompt_len, layers, k]"""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "repetition_penalty": 1.0,
    }
    data = _post(f"{host}/v1/completions", payload)
    ch = data["choices"][0]
    raw = ch.get("routed_experts")
    if raw is None:
        raise RuntimeError("响应里没有 routed_experts")
    routes = np.load(io.BytesIO(base64.b64decode(raw)))
    return routes, ch.get("text", "")


def gate_logits(block, h):
    res = block.gate(h)
    return res[0] if isinstance(res, tuple) else res


def block_attr(block, name, default=None):
    if hasattr(block, name):
        return getattr(block, name)
    if hasattr(block.gate, name):
        return getattr(block.gate, name)
    return default


def capture_all(model, input_ids, k):
    """抓全部层的路由，拼成 [tokens, layers, k]"""
    per_layer = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make(i=i, block=layer.mlp):
            def hook(module, args):
                x = args[0]
                h = x.view(-1, x.shape[-1])
                with torch.no_grad():
                    logits = gate_logits(block, h)
                    _, ids = torch.topk(logits, k, dim=-1)
                per_layer[i] = ids.detach().cpu().numpy()
            return hook
        hooks.append(layer.mlp.register_forward_pre_hook(make()))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for hd in hooks:
            hd.remove()

    n_layer = len(per_layer)
    n_tok = per_layer[0].shape[0]
    out = np.zeros((n_tok, n_layer, k), dtype=np.int16)
    for i in range(n_layer):
        out[:, i, :] = per_layer[i]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model-dir", default="/mnt/workspace/model")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts-file", default=None)
    args = ap.parse_args()

    host = f"http://localhost:{args.port}" if args.port else args.host

    try:
        from gptqmodel import GPTQModel
    except ImportError as e:
        print(f"需要 gptqmodel: {e}")
        return 2

    prompts = [args.prompt] if args.prompt else PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    model_name = resolve_model(host, args.model_name)

    print("=" * 70)
    print("跨引擎路由对比 · vLLM vs transformers")
    print("=" * 70)
    print(f"\n  vLLM        {host}")
    print(f"  model       {model_name}")
    print(f"  prompts     {len(prompts)}")

    print(f"\n加载 transformers 侧模型 {args.model_dir}（CPU，约 1 分钟）...")
    gm = GPTQModel.load(args.model_dir, device="cpu")
    tm = gm.model.eval()
    tok = getattr(gm, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    k = int(block_attr(tm.model.layers[0].mlp, "top_k", 4))
    n_layer = len(tm.model.layers)
    print(f"  完成：{n_layer} 层，top_k={k}")

    all_v, all_t = [], []
    print("\n逐 prompt 对比")
    print("─" * 70)
    print(f"  {'tokens':>7} {'vLLM':>10} {'tf':>10} {'一致率':>9}  {'不一致槽位':>12}")
    print("  " + "-" * 66)

    for p in prompts:
        try:
            rv, _ = vllm_routes(host, model_name, p)
        except Exception as e:
            print(f"  vLLM 失败: {type(e).__name__}: {str(e)[:50]}")
            continue

        ids = tok(p, return_tensors="pt")["input_ids"]
        rt = capture_all(tm, ids, k)

        n = min(len(ids[0]) - 1, rv.shape[0], rt.shape[0])
        v = rv[:n].astype(np.int16)
        t = rt[:n]
        all_v.append(v)
        all_t.append(t)

        bad = (np.sort(v, axis=-1) != np.sort(t, axis=-1)).any(axis=-1)
        total = bad.size
        nbad = int(bad.sum())
        agree = 1.0 - nbad / max(total, 1)
        print(
            f"  {n:>7} {rv.shape[0]:>10} {rt.shape[0]:>10} "
            f"{agree:>8.2%}  {nbad:>6}/{total:<5}"
        )

    if not all_v:
        print("\n没有成功的数据。")
        return 1

    V = np.concatenate(all_v, axis=0)
    T = np.concatenate(all_t, axis=0)

    bad = (np.sort(V, axis=-1) != np.sort(T, axis=-1)).any(axis=-1)
    total = int(bad.size)
    nbad = int(bad.sum())
    agree = 1.0 - nbad / max(total, 1)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"  总槽位       {total}")
    print(f"  不一致槽位   {nbad}")
    print(f"  一致率       {agree:.4%}")
    se = (agree * (1 - agree) / max(total, 1)) ** 0.5
    print(f"  95% 置信区间 {max(0.0, agree - 1.96 * se):.2%} – {min(1.0, agree + 1.96 * se):.2%}")

    per_layer = 1.0 - bad.mean(axis=0)
    n_tok = V.shape[0]
    print(f"\n  逐层一致率（每层 {n_tok} token × {k} 专家 = {n_tok * k} 槽位，")
    print(f"  单个槽位权重 {1.0 / max(n_tok * k, 1):.2%}）：")
    for i, a in enumerate(per_layer):
        bar = "█" * int(a * 30) + "░" * (30 - int(a * 30))
        print(f"    L{i:<2d} {bar} {a:>7.2%}")

    worst = np.argsort(per_layer)[:3]
    best = np.argsort(-per_layer)[:3]
    print(f"\n  最差三层  {list(worst)}")
    print(f"  最好三层  {list(best)}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if agree > 0.99:
        print(f"  一致率 {agree:.2%} —— 两个引擎的路由几乎相同。")
        print("  → R3 修的是那不到 1% 的边界情况，价值精确但有限。")
    elif agree > 0.9:
        print(f"  一致率 {agree:.2%} —— 有稳定差异但不大。")
        print("  → R3 在修正约 10% 的路由，这些是训练-推理不一致的直接来源。")
    elif agree > 0.7:
        print(f"  一致率 {agree:.2%} —— 差异显著。")
        print("  → R3 在重建约 30% 的路由，是消除不一致的主要手段。")
    else:
        print(f"  一致率 {agree:.2%} —— 两个引擎基本在不同的路由上。")
        print("  → 这个量级下，'跨引擎路由一致' 不是可追求的目标，")
        print("    R3 的意义变成「记录哪一条就用哪一条」而非「让两边算一样」。")

    print("\n  注意：这个对比受 kernel 差异影响（vLLM 用 HUMMING 原生 int4，")
    print("  transformers 用 TorchAtenLinear），差异里混了两部分：")
    print("    ① router logits 的数值差异")
    print("    ② 不同引擎的 top-k 边界行为")
    print("  这个脚本测不出两者的占比，只能给出总差异。")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
