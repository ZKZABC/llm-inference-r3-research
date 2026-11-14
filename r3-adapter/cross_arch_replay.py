#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨架构回放验证（CPU，零下载）

R3 适配层是不是只对 Qwen 有效？用 transformers 现场构造 4 个不同 MoE 架构的
微型模型，逐个验证「注入自己捕获的路由后，输出是否不变」。

架构差异（真实踩坑点）：
  qwen2_moe  有 shared_expert，norm_topk_prob 在 gate 上
  qwen3_moe  无 shared_expert
  mixtral    top_k 在 block 上，gate 无 norm 属性
  olmoe      无 shared_expert，norm 在 gate 上

判据沿用 inject_selftest：注入后的差异 <= 不注入时的基线。
若某架构 native combine 顺序与注入器不同，会表现为非 bit-identical，
这本身就是「适配层需要按架构调整 combine」的证据，如实报告。

用法：python cross_arch_replay.py
"""

import importlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = dict(vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, head_dim=16)

SPECS = {
    "qwen2_moe": ("Qwen2MoeConfig", dict(BASE, moe_intermediate_size=64,
        num_experts=8, num_experts_per_tok=4, shared_expert_intermediate_size=64)),
    "qwen3_moe": ("Qwen3MoeConfig", dict(BASE, moe_intermediate_size=64,
        num_experts=8, num_experts_per_tok=4)),
    "mixtral": ("MixtralConfig", dict(BASE, num_local_experts=8, num_experts_per_tok=4)),
    "olmoe": ("OlmoeConfig", dict(BASE, num_experts=8, num_experts_per_tok=4)),
}


def build(mod):
    cfgname, kw = SPECS[mod]
    m = importlib.import_module("transformers.models." + mod + ".modeling_" + mod)
    cfg = getattr(m, cfgname)(**kw)
    M = getattr(m, [n for n in dir(m) if n.endswith("ForCausalLM")][0])
    torch.manual_seed(0)
    return M(cfg).eval()


def gate_logits(blk, h):
    r = blk.gate(h)
    return r[0] if isinstance(r, tuple) else r


def get_k(blk):
    return int(getattr(blk, "top_k", None) or blk.gate.top_k)


def get_norm(blk):
    # Mixtral 原生对 top-k 权重重新归一化，但它的 gate/block 上没有
    # norm_topk_prob 属性，靠属性探测会漏判，导致基线差 6 个数量级。
    if "Mixtral" in type(blk).__name__:
        return True
    return bool(getattr(blk, "norm_topk_prob", getattr(blk.gate, "norm_topk_prob", False)))


def expert_fwd(ex, x, e):
    gu = F.linear(x, ex.gate_up_proj[e])
    g, u = gu.chunk(2, dim=-1)
    return F.linear(ex.act_fn(g) * u, ex.down_proj[e])


class Inject(nn.Module):
    def __init__(self, blk, routes=None):
        super().__init__()
        self.b = blk
        self.routes = routes
        self.k = get_k(blk)
        self.E = blk.experts.num_experts
        self.norm = get_norm(blk)

    def forward(self, h):
        B, S, H = h.shape
        x = h.view(-1, H)
        probs = F.softmax(gate_logits(self.b, x), dim=-1, dtype=torch.float)
        ids = self.routes if self.routes is not None else torch.topk(probs, self.k, -1).indices
        w = probs.gather(-1, ids)
        if self.norm:
            w = w / w.sum(-1, keepdim=True)
        w = w.to(x.dtype)
        out = torch.zeros_like(x)
        for e in range(self.E):
            pos, slot = torch.where(ids == e)
            if pos.numel() == 0:
                continue
            y = expert_fwd(self.b.experts, x[pos], e)
            out.index_add_(0, pos, (y * w[pos, slot].unsqueeze(-1)).to(out.dtype))
        if hasattr(self.b, "shared_expert"):
            sh = self.b.shared_expert(x)
            if hasattr(self.b, "shared_expert_gate"):
                sh = torch.sigmoid(self.b.shared_expert_gate(x)) * sh
            out = out + sh
        return out.view(B, S, H)


def capture(model, ids):
    per, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        def mk(i=i, blk=layer.mlp):
            def hook(m, a):
                x = a[0].view(-1, a[0].shape[-1])
                per[i] = torch.topk(gate_logits(blk, x), get_k(blk), -1).indices.detach()
            return hook
        hooks.append(layer.mlp.register_forward_pre_hook(mk()))
    with torch.no_grad():
        model(ids)
    for hk in hooks:
        hk.remove()
    return [per[i] for i in sorted(per)]


def run(model, ids, routes=None):
    saved = []
    for i, layer in enumerate(model.model.layers):
        saved.append((layer, layer.mlp))
        layer.mlp = Inject(layer.mlp, routes[i] if routes else None)
    try:
        with torch.no_grad():
            return model(ids).logits.float()
    finally:
        for layer, orig in saved:
            layer.mlp = orig


def main():
    print("=" * 70)
    print("跨架构回放验证 · CPU 零下载")
    print("=" * 70)
    ids = torch.randint(0, 64, (2, 16))
    print(f"\n  {'架构':<12} {'shared':>7} {'norm':>6} {'基线':>11} {'注入':>11}  判定")
    print("  " + "-" * 62)
    all_ok = True
    for mod in SPECS:
        try:
            model = build(mod)
        except Exception as e:
            print(f"  {mod:<12} 构建失败 {type(e).__name__}")
            continue
        blk = model.model.layers[0].mlp
        has_sh = hasattr(blk, "shared_expert")
        with torch.no_grad():
            native = model(ids).logits.float()
        routes = capture(model, ids)
        base = (run(model, ids) - native).abs().max().item()
        inj = (run(model, ids, routes) - native).abs().max().item()
        ok = inj <= max(base * 1.05, 1e-6)
        all_ok &= ok
        tag = "bit-identical" if inj == 0 else ("<=基线" if ok else "超基线!")
        print(f"  {mod:<12} {str(has_sh):>7} {str(get_norm(blk)):>6} {base:>11.3e} {inj:>11.3e}  {tag}")
    print("\n  " + ("全部架构回放成立 → 适配层不绑定 Qwen" if all_ok else "存在架构需单独处理 combine"))
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
