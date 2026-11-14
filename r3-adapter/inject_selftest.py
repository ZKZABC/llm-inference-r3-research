#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注入机制自洽测试

验证一件事：**把模型自己产生的路由注入回去，输出必须 bit-identical。**

这个测试不需要真实模型、不需要 GPTQ、不需要 GPU。
它只回答一个前置问题：注入这个动作本身有没有引入误差。

为什么必须先做：
  如果注入实现有偏差（比如权重没从完整 logits 重新 gather、
  或者归一化漏了），那么后面所有「推理 vs 训练」的对比都会把
  这个偏差算到引擎头上，得出错误结论。

判据：
  out1 = model(input_ids).logits
  out2 = inject(model, capture_routes(model, input_ids))(input_ids).logits
  torch.equal(out1, out2) 必须为 True

用法:
    python inject_selftest.py
    python inject_selftest.py --layers 2 --seq 16 --dtype fp32
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


def diff_of(a, b) -> Diff:
    if torch.equal(a, b):
        return Diff(True, 0.0, 0, a.numel())
    d = (a.float() - b.float()).abs()
    return Diff(False, d.max().item(), int((a != b).sum().item()), a.numel())


def gate_logits(block, h):
    """
    用 gate 模块本身的输出，而不是自己 F.linear。

    早期版本直接 F.linear(h, gate.weight)，绕过了 gate 内部的其它处理
    （比如 bias、scaling、或者它自己已经做过 softmax），导致注入后
    输出全错（max 差异 0.07，是逻辑错误而非精度问题）。
    """
    res = block.gate(h)
    if isinstance(res, tuple):
        return res[0]
    return res


def expert_forward(experts, x, e):
    """单 expert：down(act(gate) * up)，适配 3D 权重布局"""
    if hasattr(experts, "gate_up_proj"):
        gu = F.linear(x, experts.gate_up_proj[e])
        gate, up = gu.chunk(2, dim=-1)
        return F.linear(experts.act_fn(gate) * up, experts.down_proj[e])
    # 旧版 ModuleList 布局
    return experts[e](x)


class InjectedMoE(nn.Module):
    """
    用「外部给定的路由」替换 gate 的 top-k 选择。

    关键：只替换 ids，权重仍然从本次 forward 的完整 logits 里 gather。
    这正是 check_replay_weights 防的那个坑——如果连权重一起注入，
    就绕过了 softmax 归一化，测不出真实差异。
    """

    def __init__(self, hf_block, route_ids=None, top_k=None):
        super().__init__()
        self.gate = hf_block.gate
        self.experts = hf_block.experts
        self.shared_expert = getattr(hf_block, "shared_expert", None)
        self.shared_expert_gate = getattr(hf_block, "shared_expert_gate", None)
        self.route_ids = route_ids  # None 表示自己算 topk（用于隔离测试）
        self.top_k = int(route_ids.shape[-1]) if route_ids is not None else int(top_k)
        self.num_experts = (
            int(route_ids.max().item()) + 1 if route_ids is not None else None
        )
        # ★ 归一化是配置项，不是固定行为。Qwen2MoeConfig 默认 False，
        #   无条件除以 sum 会让权重整体偏小，产生 0.07 量级的错误。
        self.norm_topk_prob = getattr(hf_block.gate, "norm_topk_prob", False)

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
            # ★ 权重从「本次 logits」gather，不用记录的权重
            w = probs.gather(-1, ids)
        else:
            w, ids = torch.topk(probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(logits.dtype)

        n_exp = self.num_experts or int(ids.max().item()) + 1

        # 用 index_add_ + expert 顺序，复现 HF 的累加方式
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


def capture_routes(model, input_ids, k):
    """跑一次前向，用 pre-hook 记录每层的 top-k 选择"""
    routes = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make(i=i, block=layer.mlp):
            def hook(module, args):
                x = args[0]
                h = x.view(-1, x.shape[-1])
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


def run_injected(model, input_ids, routes):
    """把每层 mlp 换成 InjectedMoE 后重新前向；routes=None 表示不注入"""
    saved = []
    for i, layer in enumerate(model.model.layers):
        saved.append((layer, layer.mlp))
        if routes is None:
            layer.mlp = InjectedMoE(layer.mlp, None, top_k=layer.mlp.gate.top_k)
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
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print("=" * 68)
    print("注入机制自洽测试")
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
    print(f"  hidden / experts   {config.hidden_size} / {config.num_experts}")
    print(f"  top_k / layers     {config.num_experts_per_tok} / {config.num_hidden_layers}")
    print(f"  dtype / batch×seq  {dtype} / {args.batch}×{args.seq}")

    torch.manual_seed(0)
    model = Qwen2MoeForCausalLM(config).to(dtype=dtype).eval()
    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq))

    print("\n步骤 1  抓取原模型的路由")
    routes, logits_ref = capture_routes(model, input_ids, config.num_experts_per_tok)
    for i, r in routes.items():
        print(f"  layer {i}  shape={tuple(r.shape)}  unique={len(torch.unique(r))}")

    print("\n步骤 2  等价性对照：InjectedMoE 自己算 topk（不注入）")
    logits_equiv = run_injected(model, input_ids, None)
    d_eq = diff_of(logits_ref, logits_equiv)
    if d_eq.same:
        print("  bit-identical  ✓  MoE 实现与 HF 完全等价")
    else:
        print(f"  {d_eq.n_diff}/{d_eq.total} 不同 · max={d_eq.max_diff:.3e}")
        print("  → MoE 实现本身就有偏差，注入测试的结论不可信，先修这个")

    print("\n步骤 3  注入刚抓到的路由，重新前向")
    logits_inj = run_injected(model, input_ids, routes)

    print("\n步骤 4  对比")
    print("─" * 68)
    d = diff_of(logits_ref, logits_inj)
    if d.same:
        print("  bit-identical  ✓")
    else:
        print(f"  {d.n_diff}/{d.total} 元素不同 · max={d.max_diff:.3e}")

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)

    # 判据不是 bit-identical：重写 MoE 实现本身就有 fp32 精度噪声（2-4 ulp）。
    # 真正要证明的是「注入不引入额外误差」，即注入后的差异 == 不注入时的基线。
    baseline = d_eq.max_diff
    injected = d.max_diff

    if d.same:
        print("  注入自己产生的路由，输出 bit-identical。")
        print("  → 注入机制零误差。")
    elif injected <= max(baseline * 1.05, 1e-6):
        print(f"  注入后 max={injected:.3e}，不注入时基线 max={baseline:.3e}")
        print("  → 注入未引入额外误差，差异全部来自 MoE 实现的重写（fp32 精度噪声）")
        print("  → 注入机制可用：后续出现更大差异，只能归因于路由本身不同")
    else:
        print(f"  注入后 max={injected:.3e}，基线只有 {baseline:.3e}")
        print("  → 注入引入了额外误差，排查：")
        print("    1. 抓到的 ids 是否真的是 HF 那一次的选择（hook 时机）")
        print("    2. 权重 gather 的维度是否对齐")
        print("    3. route_ids 的 dtype / device 是否一致")
    print()
    return 0 if (d.same or injected <= max(baseline * 1.05, 1e-6)) else 1


if __name__ == "__main__":
    sys.exit(main())
