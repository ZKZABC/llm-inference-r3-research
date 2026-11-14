#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
受控跨引擎模拟：R3 到底修掉了跨引擎差异的哪一部分？

真机上「路由差异」和「kernel 差异」混在一起，分不开（报告 §6.1 的局限）。
CPU 上可以人为构造两个"引擎"，把变量拆开：

  引擎 A（rollout 侧）：HF 原生 forward，fp32，expert 顺序累加
  引擎 B（训练侧）  ：重写 MoE，可配置 compute dtype 与 combine 顺序

四个 B 变体，逐个叠加变量：

  B1 fp32 + expert 顺序   纯实现重写噪声（基线，应 ~1e-7）
  B2 fp32 + slot 顺序     + combine 顺序差异
  B3 bf16 + expert 顺序   + dtype 差异（路由开始分叉）
  B4 bf16 + slot 顺序     + 两者

每个变体测三个数：
  路由一致率   B 自己算的 top-k vs A 的（set 比较）
  gap_raw     max|logits_B - logits_A|        不修 R3
  gap_r3      max|logits_B注入A路由 - logits_A|  修 R3

预期：
  B1/B2 一致率 100%（gate 都是 fp32），gap 极小，R3 几乎全修
  B3/B4 一致率 <100%，gap_r3 明显小于 gap_raw，残差 = bf16 数值噪声

这给出一个真机给不出的分解：**R3 修掉路由诱导的部分，留下 kernel 噪声的部分。**

用法：python r3_cross_sim.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
except ImportError:
    print("需要 transformers >= 4.40")
    sys.exit(2)


def build():
    cfg = Qwen2MoeConfig(
        vocab_size=64, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        num_experts=8, num_experts_per_tok=4,
        shared_expert_intermediate_size=256,
    )
    torch.manual_seed(0)
    m = Qwen2MoeForCausalLM(cfg).eval()
    # 随机初始化的 router 边界太宽，bf16 噪声翻不动 top-4。
    # 真实训练后的 router 边界很密（真机一致率才 91.7%）。
    # 人为把第 5 名专家权重设成第 4 名的 1.0005 倍，制造一个
    # 落在 bf16 舍入误差（~0.4%）内的 4/5 边界平局，模拟训练态。
    with torch.no_grad():
        for layer in m.model.layers:
            w = layer.mlp.gate.weight
            w[4] = w[3] * 1.0005
    return m


def expert_fwd(experts, x, e, cd):
    gu = F.linear(x, experts.gate_up_proj[e].to(cd))
    g, u = gu.chunk(2, dim=-1)
    return F.linear(experts.act_fn(g) * u, experts.down_proj[e].to(cd))


def shared_fwd(mlp, x, cd):
    g = F.linear(x, mlp.gate_proj.weight.to(cd))
    u = F.linear(x, mlp.up_proj.weight.to(cd))
    return F.linear(mlp.act_fn(g) * u, mlp.down_proj.weight.to(cd))


class SimMoE(nn.Module):
    """引擎 B 的 MoE：可配置 compute dtype 与 combine 顺序，可注入路由"""

    def __init__(self, block, cd, order, routes=None):
        super().__init__()
        self.b = block
        self.cd = cd
        self.order = order
        self.routes = routes
        self.k = block.gate.top_k
        self.E = block.experts.num_experts
        self.norm = getattr(block, "norm_topk_prob", False)

    def forward(self, h):
        B, S, H = h.shape
        x = h.view(-1, H).to(self.cd)
        logits = F.linear(x, self.b.gate.weight.to(self.cd))
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        ids = self.routes if self.routes is not None else torch.topk(probs, self.k, -1).indices
        w = probs.gather(-1, ids)
        if self.norm:
            w = w / w.sum(-1, keepdim=True)
        w = w.to(self.cd)

        out = torch.zeros_like(x)
        if self.order == "expert":
            for e in range(self.E):
                pos, slot = torch.where(ids == e)
                if pos.numel() == 0:
                    continue
                y = expert_fwd(self.b.experts, x[pos], e, self.cd)
                out.index_add_(0, pos, (y * w[pos, slot].unsqueeze(-1)).to(self.cd))
        else:
            for j in range(self.k):
                for e in range(self.E):
                    sel = ids[:, j] == e
                    if sel.any():
                        y = expert_fwd(self.b.experts, x[sel], e, self.cd)
                        out[sel] = out[sel] + y * w[sel, j].unsqueeze(-1)

        sh = shared_fwd(self.b.shared_expert, x, self.cd)
        sh = torch.sigmoid(F.linear(x, self.b.shared_expert_gate.weight.to(self.cd))) * sh
        return (out + sh).view(B, S, H)


def capture(model, ids, cd):
    """抓每层 gate 在指定 dtype 下的 top-k"""
    per = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def mk(i=i, blk=layer.mlp):
            def hook(m, args):
                x = args[0].view(-1, args[0].shape[-1]).to(cd)
                lg = F.linear(x, blk.gate.weight.to(cd))
                per[i] = torch.topk(lg, blk.gate.top_k, -1).indices.detach()
            return hook
        hooks.append(layer.mlp.register_forward_pre_hook(mk()))
    with torch.no_grad():
        model(ids)
    for hk in hooks:
        hk.remove()
    return [per[i] for i in sorted(per)]


def run(model, ids, cd=None, order=None, routes=None):
    if cd is None:
        with torch.no_grad():
            return model(ids).logits.float()
    saved = []
    for i, layer in enumerate(model.model.layers):
        saved.append((layer, layer.mlp))
        layer.mlp = SimMoE(layer.mlp, cd, order, routes[i] if routes else None)
    try:
        with torch.no_grad():
            return model(ids).logits.float()
    finally:
        for layer, orig in saved:
            layer.mlp = orig


def agreement(ra, rb):
    n = ok = 0
    for a, b in zip(ra, rb):
        sa = a.sort(-1).values
        sb = b.sort(-1).values
        ok += int((sa == sb).all(-1).sum())
        n += a.shape[0]
    return ok / max(n, 1)


def main():
    model = build()
    ids = torch.randint(0, 64, (4, 32))

    print("=" * 74)
    print("受控跨引擎模拟 · R3 修复比例分解")
    print("=" * 74)

    logits_A = run(model, ids)
    routes_A = capture(model, ids, torch.float32)

    print(f"\n  {'变体':<16} {'路由一致率':>10} {'gap_raw':>11} {'gap_r3':>11} {'R3修掉':>8}")
    print("  " + "-" * 62)

    rows = []
    for name, cd, order in [
        ("B1 fp32+expert", torch.float32, "expert"),
        ("B2 fp32+slot", torch.float32, "slot"),
        ("B3 bf16+expert", torch.bfloat16, "expert"),
        ("B4 bf16+slot", torch.bfloat16, "slot"),
    ]:
        rB = capture(model, ids, cd)
        agr = agreement(routes_A, rB)
        g_raw = (run(model, ids, cd, order) - logits_A).abs().max().item()
        g_r3 = (run(model, ids, cd, order, routes_A) - logits_A).abs().max().item()
        closed = 1 - g_r3 / g_raw if g_raw > 0 else 1.0
        rows.append((name, agr, g_raw, g_r3, closed))
        print(f"  {name:<16} {agr:>9.2%} {g_raw:>11.3e} {g_r3:>11.3e} {closed:>7.1%}")

    print("\n" + "=" * 74)
    print("解读")
    print("=" * 74)
    b1, b2, b3, b4 = rows
    print(f"  B1 基线噪声 {b1[2]:.2e}（实现重写本身，应 ~1e-7）")
    print(f"  B2 只换 combine 顺序：一致率 {b2[1]:.0%}，gap {b2[2]:.2e}")
    print(f"     → 顺序差异不改变路由（gate 同 dtype），只改输出累加")
    print(f"  B3 只换 dtype：一致率 {b3[1]:.2%}")
    print(f"     → 路由分叉来自 dtype，不是 combine 顺序")
    print(f"  B4 两者叠加：gap_raw {b4[2]:.2e} → gap_r3 {b4[3]:.2e}，R3 修掉 {b4[4]:.1%}")
    print(f"     → 残差 {b4[3]:.2e} 是 bf16 数值噪声，R3 修不掉也不该修")

    ok = b1[2] < 1e-5 and b1[1] == 1.0 and b4[3] < b4[2]
    print(f"\n  sanity：基线<1e-5 且 B1 全一致 且 R3 减小 gap → {'PASS' if ok else 'FAIL'}")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
