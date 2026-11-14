#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 骨架：MoE block 与 HF 对齐

用缩小版 Qwen2Moe（零下载）验证三件事：
  1. 3D 权重布局 [E, 2*I, H] / [E, H, I] 理解正确
  2. router 的 top-k 选择与归一化正确
  3. MoE block 的 combine 顺序影响

接法：把 layer.mlp 替换成 wrapper，让模型自己跑完整 forward。
这样只有 MoE 被替换，attention / norm 全部原样。

注意 transformers 5.x 的变化：
  · experts 是 3D Parameter，不是 ModuleList
  · forward 不返回 router_logits —— 用 pre-hook 从 gate 抓

用法:
    python phase0_skeleton.py
    python phase0_skeleton.py --dtype bf16 --layers 3
"""

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
except ImportError:
    print("需要 transformers >= 4.40（Qwen2Moe 从这里开始提供）")
    sys.exit(2)


@dataclass
class Diff:
    same: bool
    max_diff: float
    n_diff: int
    total: int


def diff_of(ref, got) -> Diff:
    if torch.equal(ref, got):
        return Diff(True, 0.0, 0, ref.numel())
    d = (ref.float() - got.float()).abs()
    return Diff(False, d.max().item(), int((ref != got).sum().item()), ref.numel())


def topk_flips(logits_a, logits_b, k):
    """top-k 选择发生变化的 token 数（按集合比较，与顺序无关）"""
    ia = logits_a.topk(k, dim=-1).indices.sort(dim=-1).values
    ib = logits_b.topk(k, dim=-1).indices.sort(dim=-1).values
    changed = (ia != ib).any(dim=-1)
    return int(changed.sum()), logits_a.shape[0]


class DetMoE(nn.Module):
    """
    确定性版 SparseMoeBlock（适配 transformers 5.x 的 3D 权重布局）。

    combine="slot"   → 按 top-k slot 顺序累加
    combine="expert" → 按 expert 顺序累加（复现 HF 的遍历顺序）
    """

    def __init__(self, hf_block, combine="slot"):
        super().__init__()
        self.gate = hf_block.gate
        self.experts = hf_block.experts
        self.shared_expert = hf_block.shared_expert
        self.shared_expert_gate = hf_block.shared_expert_gate
        self.top_k = self.gate.top_k
        self.num_experts = self.experts.num_experts
        self.combine = combine
        self.last_logits = None

    def _expert(self, x, e):
        """单个 expert 的 MLP：down(act(gate) * up)"""
        gu = F.linear(x, self.experts.gate_up_proj[e])
        gate, up = gu.chunk(2, dim=-1)
        return F.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[e])

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        h = hidden_states.view(-1, H)
        k = self.top_k

        logits, top_w, top_i = self.gate(h)
        self.last_logits = logits.detach()
        top_w = top_w.to(h.dtype)

        out = torch.zeros_like(h)

        if self.combine == "slot":
            for j in range(k):
                for e in range(self.num_experts):
                    sel = top_i[:, j] == e
                    if sel.any():
                        out[sel] = out[sel] + top_w[sel, j : j + 1] * self._expert(h[sel], e)
        else:
            for e in range(self.num_experts):
                mask = top_i == e
                if not mask.any():
                    continue
                pos, slot = torch.where(mask)
                if pos.numel() == 0:
                    continue
                y = self._expert(h[pos], e)
                out[pos] = out[pos] + top_w[pos, slot].unsqueeze(-1) * y

        sh = self.shared_expert(h)
        sh = torch.sigmoid(self.shared_expert_gate(h)) * sh
        out = out + sh

        return out.view(B, S, H)


def run_hf(model, input_ids):
    """跑原始 HF，用 pre-hook 从 gate 抓 router_logits"""
    captured = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make(i=i, gate=layer.mlp.gate):
            def hook(module, args):
                x = args[0]
                x = x.view(-1, x.shape[-1])
                with torch.no_grad():
                    captured[i] = gate(x)[0].detach()
            return hook
        hooks.append(layer.mlp.register_forward_pre_hook(make()))

    try:
        with torch.no_grad():
            out = model(input_ids)
    finally:
        for hd in hooks:
            hd.remove()

    return out.logits, captured


def run_det(model, input_ids, combine):
    """替换 mlp 后跑一遍"""
    saved = []
    for layer in model.model.layers:
        saved.append((layer, layer.mlp))
        layer.mlp = DetMoE(layer.mlp, combine=combine)

    try:
        with torch.no_grad():
            out = model(input_ids)
        logits = {i: layer.mlp.last_logits for i, layer in enumerate(model.model.layers)}
    finally:
        for layer, orig in saved:
            layer.mlp = orig

    return out.logits, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print("=" * 68)
    print("Phase 0 骨架 · MoE block 与 HF 对齐")
    print("=" * 68)

    config = Qwen2MoeConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )

    print("\n配置")
    print(f"  hidden / experts      {config.hidden_size} / {config.num_experts}")
    print(f"  top_k / layers        {config.num_experts_per_tok} / {config.num_hidden_layers}")
    print(f"  q_heads / kv_heads    {config.num_attention_heads} / {config.num_key_value_heads}")
    print(f"  dtype / seq           {dtype} / {args.seq}")

    torch.manual_seed(0)
    model = Qwen2MoeForCausalLM(config).to(dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, args.seq))

    print("\n跑三轮 forward（原始 / slot 顺序 / expert 顺序）")
    logits_hf, rl_hf = run_hf(model, input_ids)
    logits_slot, rl_slot = run_det(model, input_ids, "slot")
    logits_exp, rl_exp = run_det(model, input_ids, "expert")

    print("\n逐层 router_logits 与 top-k 翻转")
    print("─" * 68)
    print(f"  {'layer':<7} {'logits':<16} {'top-k 翻转':<18} 说明")
    rl_ok = True
    for i in range(len(model.model.layers)):
        d = diff_of(rl_hf[i], rl_slot[i])
        n_flip, n_tok = topk_flips(rl_hf[i], rl_slot[i], config.num_experts_per_tok)
        if i == 0:
            rl_ok &= d.same
        note = "gate 本身正确" if i == 0 else ("误差从上游传入" if not d.same else "")
        flag = "一致" if d.same else f"{d.max_diff:.1e}"
        print(f"  {i:<7} {flag:<16} {n_flip}/{n_tok:<14} {note}")

    print("\n最终 logits")
    print("─" * 68)
    d_slot = diff_of(logits_hf, logits_slot)
    d_exp = diff_of(logits_hf, logits_exp)
    print(f"  HF vs slot     max={d_slot.max_diff:.3e}  {'一致' if d_slot.same else f'{d_slot.n_diff}/{d_slot.total} 不同'}")
    print(f"  HF vs expert   max={d_exp.max_diff:.3e}  {'一致' if d_exp.same else f'{d_exp.n_diff}/{d_exp.total} 不同'}")

    ok_hidden = d_slot.max_diff <= args.tol
    ok = rl_ok and ok_hidden

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)

    if rl_ok:
        print("  layer 0 的 router_logits 完全一致")
        print("  → gate 权重、softmax、top-k、归一化全部正确")
    else:
        print("  layer 0 的 router_logits 就不一致 → gate 路径有问题")
        print("    检查 gate_up_proj 的前后两半是不是 [gate, up] 而不是 [up, gate]")

    if ok_hidden:
        print(f"  logits 在 {args.tol:.0e} 内一致 → MoE 实现正确，骨架可进下一步")
    else:
        print("  logits 超差 → 检查 expert 内部顺序")
        print("    down(act(gate) * up)：gate_up_proj 的前半是 gate，后半是 up")

    flips = [topk_flips(rl_hf[i], rl_slot[i], config.num_experts_per_tok)[0] for i in range(len(model.model.layers))]
    if sum(flips) > 0:
        print(f"\n  路由翻转：各层 {flips}，共 {sum(flips)} 次")
        print("  → 即使 logits 只差 2e-7，top-k 边界也会翻，这就是 3/768 的机制")
    else:
        print("\n  没有路由翻转 —— 但这不代表安全，只是这次没踩到边界")

    if not d_exp.same:
        print(f"\n  按 expert 顺序累加与 HF 差 {d_exp.max_diff:.2e}")
        print("  → §2.2 的 combine 顺序敏感，在真实结构上复现了")
    else:
        print("\n  按 expert 顺序与 HF 完全一致（HF 本身就是 expert 顺序）")

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
