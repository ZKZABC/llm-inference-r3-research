#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实模型注入验证（Qwen1.5-MoE / GPTQ int4 / CPU）

inject_selftest.py 在小模型上验证了注入机制成立。
这个脚本在真实权重上再验一次——确认我们对模型结构的理解没有偏差。

为什么值得做：
  小模型是 Qwen2MoeConfig 随机初始化的，结构和真实模型同源但填充方式不同
  （3D gate_up_proj vs ModuleList）。如果结构理解有误，小模型可能测不出来。

判据同 inject_selftest：
  注入后的差异 <= 不注入时的基线（重写实现本身的 fp32 精度噪声）

用法:
    python inject_real.py
    python inject_real.py --prompt "The capital of France is" --max-tokens 8
    python inject_real.py --layers 4          # 只加载前 N 层的真实权重

注意：CPU + bf16（无 AVX512-BF16）会很慢，一次前向可能几十秒到几分钟。
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Diff:
    same: bool
    max_diff: float
    n_diff: int
    total: int


def diff_of(a, b) -> Diff:
    if torch.equal(a, b):
        return Diff(True, 0.0, 0, a.numel())
    d = (a.float() - b.float()).abs()
    return Diff(False, d.max().item(), int((a != b).sum().item()), a.numel())


def gate_logits(block, h):
    """用 gate 模块本身的输出，兼容 tuple / tensor 两种返回"""
    res = block.gate(h)
    return res[0] if isinstance(res, tuple) else res


def block_attr(block, name, default=None):
    """gptqmodel 把 top_k / num_experts / norm_topk_prob 放在 block 上，
    原生 transformers 放在 gate 上。两边都查。"""
    if hasattr(block, name):
        return getattr(block, name)
    if hasattr(block.gate, name):
        return getattr(block.gate, name)
    return default


def expert_forward(experts, x, e):
    """兼容两种布局：3D gate_up_proj / ModuleList"""
    if hasattr(experts, "gate_up_proj"):
        gu = F.linear(x, experts.gate_up_proj[e])
        gate, up = gu.chunk(2, dim=-1)
        return F.linear(experts.act_fn(gate) * up, experts.down_proj[e])
    return experts[e](x)


class InjectedMoE(nn.Module):
    """用外部给定的路由替换 top-k 选择；权重仍从本次 logits gather"""

    def __init__(self, hf_block, route_ids=None, top_k=None):
        super().__init__()
        self.gate = hf_block.gate
        self.experts = hf_block.experts
        self.shared_expert = getattr(hf_block, "shared_expert", None)
        self.shared_expert_gate = getattr(hf_block, "shared_expert_gate", None)
        self.route_ids = route_ids
        self.top_k = int(route_ids.shape[-1]) if route_ids is not None else int(top_k)
        self.num_experts = (
            int(route_ids.max().item()) + 1 if route_ids is not None else None
        )
        self.norm_topk_prob = bool(block_attr(hf_block, "norm_topk_prob", False))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        h = hidden_states.view(-1, H)

        logits = gate_logits(self, h)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)

        if self.route_ids is not None:
            if self.route_ids.shape[0] != h.shape[0]:
                raise ValueError(
                    f"路由行数 {self.route_ids.shape[0]} != token 数 {h.shape[0]}"
                )
            ids = self.route_ids
            w = probs.gather(-1, ids)
        else:
            w, ids = torch.topk(probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(logits.dtype)

        n_exp = self.num_experts or int(ids.max().item()) + 1

        final = torch.zeros_like(h)
        for e in range(n_exp):
            mask = ids == e
            if not mask.any():
                continue
            pos, slot = torch.where(mask)
            y = expert_forward(self.experts, h[pos], e)
            y = y * w[pos, slot].unsqueeze(-1)
            final.index_add_(0, pos, y.to(final.dtype))

        if self.shared_expert is not None:
            sh = self.shared_expert(h)
            sh = torch.sigmoid(self.shared_expert_gate(h)) * sh
            final = final + sh

        return final.view(B, S, H)


def capture_routes(model, input_ids, layers):
    routes = {}
    hooks = []

    for i, layer in enumerate(layers):
        def make(i=i, block=layer.mlp):
            def hook(module, args):
                x = args[0]
                h = x.view(-1, x.shape[-1])
                k = int(block_attr(block, "top_k", 4))
                with torch.no_grad():
                    logits = gate_logits(block, h)
                    _, ids = torch.topk(logits, k, dim=-1)
                routes[i] = ids.detach()
            return hook
        hooks.append(layer.mlp.register_forward_pre_hook(make()))

    try:
        with torch.no_grad():
            out = model(input_ids)
    finally:
        for hd in hooks:
            hd.remove()

    return routes, out.logits


def run_injected(model, input_ids, layers, routes):
    saved = []
    for i, layer in enumerate(layers):
        saved.append((layer, layer.mlp))
        if routes is None:
            layer.mlp = InjectedMoE(layer.mlp, None, top_k=block_attr(layer.mlp, "top_k", 4))
        else:
            layer.mlp = InjectedMoE(layer.mlp, routes[i])
    try:
        with torch.no_grad():
            out = model(input_ids)
    finally:
        for layer, orig in saved:
            layer.mlp = orig
    return out.logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/workspace/model")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--layers", type=int, default=0, help="只替换前 N 层做注入（0=全部）")
    args = ap.parse_args()

    print("=" * 68)
    print("真实模型注入验证")
    print("=" * 68)

    try:
        from gptqmodel import GPTQModel
    except ImportError:
        print("需要 gptqmodel: pip install gptqmodel")
        return 2

    t0 = time.time()
    print(f"\n加载模型 {args.model}（CPU，可能 1-2 分钟）...")
    m = GPTQModel.load(args.model, device="cpu")
    mm = m.model.eval()
    print(f"  完成，耗时 {time.time() - t0:.1f}s")

    tok = getattr(m, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"\n  prompt    {args.prompt!r}")
    print(f"  tokens    {input_ids.shape[1]}")
    print(f"  layers    {len(mm.model.layers)}")
    print(f"  top_k     {block_attr(mm.model.layers[0].mlp, 'top_k', '?')}")
    print(f"  norm_prob {block_attr(mm.model.layers[0].mlp, 'norm_topk_prob', '?')}")

    layers = list(mm.model.layers)
    if args.layers:
        layers = layers[: args.layers]
        print(f"  注入范围  前 {len(layers)} 层（其余层保持原始）")
        print("  注意：hook 只在这些层上，前向仍走全部 24 层")

    print("\n步骤 1  抓取路由（完整前向，慢）")
    t1 = time.time()
    routes, logits_ref = capture_routes(mm, input_ids, layers)
    print(f"  完成，耗时 {time.time() - t1:.1f}s")
    for i, r in routes.items():
        print(f"    layer {i}  shape={tuple(r.shape)}  unique={len(torch.unique(r))}")

    print("\n步骤 2  等价性对照（InjectedMoE 自算 topk，不注入）")
    t2 = time.time()
    logits_eq = run_injected(mm, input_ids, layers, None)
    d_eq = diff_of(logits_ref, logits_eq)
    print(f"  耗时 {time.time() - t2:.1f}s")
    if d_eq.same:
        print("  bit-identical ✓")
    else:
        print(f"  {d_eq.n_diff}/{d_eq.total} 不同 · max={d_eq.max_diff:.3e}")

    print("\n步骤 3  注入刚抓到的路由")
    t3 = time.time()
    logits_inj = run_injected(mm, input_ids, layers, routes)
    d_inj = diff_of(logits_ref, logits_inj)
    print(f"  耗时 {time.time() - t3:.1f}s")
    if d_inj.same:
        print("  bit-identical ✓")
    else:
        print(f"  {d_inj.n_diff}/{d_inj.total} 不同 · max={d_inj.max_diff:.3e}")

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)

    baseline = d_eq.max_diff
    injected = d_inj.max_diff

    if d_inj.same:
        print("  真实权重上注入后 bit-identical。")
        print("  → 注入机制在真实模型上零误差。")
    elif injected <= max(baseline * 1.05, 1e-5):
        print(f"  注入 max={injected:.3e}，基线 max={baseline:.3e}")
        print("  → 注入未引入额外误差。")
        print("  → 结构理解正确，可以用于「vLLM vs transformers」的路由对比。")
    else:
        print(f"  注入 max={injected:.3e}，基线只有 {baseline:.3e}")
        print("  → 注入引入额外误差，排查：")
        print("    1. hook 抓的 ids 与 forward 时 gate 用的 ids 是否同一次")
        print("    2. norm_topk_prob 的取值（block 上 vs gate 上可能不同）")
        print("    3. expert 布局（ModuleList vs 3D 张量）")
    print()
    return 0 if (d_inj.same or injected <= max(baseline * 1.05, 1e-5)) else 1


if __name__ == "__main__":
    sys.exit(main())
