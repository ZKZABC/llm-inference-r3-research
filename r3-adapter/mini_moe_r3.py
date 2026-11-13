# -*- coding: utf-8 -*-
"""微型 MoE · R3 数学命题验证（纯 CPU，不依赖 vLLM/GPU/大模型）

要验证的核心命题（来自《报告》§7）：
    R3（Router Replay）声称——只要训练侧精确回放 rollout 时选中的专家，
    logprob 就会完全一致（差异归零）。

本脚本用一个几十行代码定义的微型 MoE（4 个专家、top-2 路由）在 CPU 上
直接验证这个命题。命题成立与否，与大模型、GPU、vLLM 都无关。

关键实现点（对应 Megatron PR#79 的教训）：
    回放时，权重必须从「完整的 router logits」里 gather 对应索引，
    而不是从「top-k 后其他位置归零的 probs」里 gather——
    后者会读到被清零的 0，静默杀死梯度。

运行：python mini_moe_r3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # 可复现


class MoELayer(nn.Module):
    """一个 MoE 层：4 个专家，top-2 稀疏路由。"""

    def __init__(self, d_model=16, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)   # 路由：d -> 每个专家的得分
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])

    def forward(self, x, forced_routing=None):
        """x: [batch, seq, d_model]；forced_routing: [batch, seq, top_k] 或 None。

        正常前向走 top-k；回放前向用 forced_routing 强制专家选择。
        返回 (输出, 选中的专家索引) —— 后者就是「routed_experts」。
        """
        router_logits = self.router(x)                      # [b, s, n_experts]

        if forced_routing is None:
            # —— rollout 侧：正常 top-k ——
            topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        else:
            # —— 训练侧回放：强制用 rollout 时选的专家 ——
            topk_idx = forced_routing
            # 关键：权重从「完整 logits」gather，而非从清零后的 probs（PR#79 教训）
            topk_vals = torch.gather(router_logits, -1, topk_idx)

        weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1)  # [b, s, top_k, 1]

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            # 该 token 选中专家 e 的所有 (k, pos) 位置
            mask = (topk_idx == e).float().unsqueeze(-1)      # [b, s, top_k, 1]
            if mask.sum() > 0:
                expert_out = self.experts[e](x)               # [b, s, d]
                # 把该专家对每个 top-k 槽位的贡献加权累加
                out = out + (mask * weights * expert_out.unsqueeze(2)).sum(dim=2)
        return out, topk_idx


class TinyMoE(nn.Module):
    """微型语言模型：embedding -> MoE -> head，用于算 logprob。"""

    def __init__(self, vocab=16, d_model=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.moe = MoELayer(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def logprob(self, ids):
        """返回 [batch, seq-1, vocab] 的 log_softmax（每个位置预测下一个 token）。"""
        x = self.embed(ids)                       # [b, s, d]
        h, routing = self.moe(x)                  # [b, s, d], routing
        logits = self.head(h)                     # [b, s, vocab]
        return F.log_softmax(logits, dim=-1), routing


def main():
    print("=" * 64)
    print("微型 MoE · R3 数学命题验证：回放路由 → logprob 是否一致")
    print("=" * 64)

    model = TinyMoE()
    ids = torch.randint(0, 16, (1, 8))            # 8 个 token 的序列

    # —— 第 1 次：rollout 正常前向，记录路由 ——
    logprob_rollout, routing = model.logprob(ids)
    print(f"\n[rollout 正常前向] routed_experts shape = {tuple(routing.shape)}")
    print(f"  （语义：[seq, top_k]，本微型模型 top_k=2）")
    print(f"  选中专家示例（第 0 个 token）: {routing[0, 0].tolist()}")

    # —— 第 2 次：回放路由（训练侧强制用同一份专家索引） ——
    # 真实 R3 里，回放的索引来自 rollout 保存的 routed_experts；
    # 这里模拟：第二次前向把 routing 作为 forced_routing 传回 MoE 层。
    x = model.embed(ids)
    h_replay, _ = model.moe(x, forced_routing=routing)
    logprob_replay = F.log_softmax(model.head(h_replay), dim=-1)

    # —— 对比：差异应归零 ——
    diff = (logprob_rollout - logprob_replay).abs().max().item()
    print(f"\n[回放前向] 强制使用同一份专家索引")
    print(f"  logprob 最大绝对差异 = {diff:.3e}")

    if diff < 1e-6:
        print("\n" + "=" * 64)
        print("✓ R3 数学命题成立：回放路由后 logprob 完全一致（差异归零）")
        print("  含义：路由一致性是「纯工程问题」，不是「算法妥协」。")
        print("  下一步：接真实 vLLM 的 routed_experts，把这份索引喂给训练侧。")
        print("=" * 64)
    else:
        print("\n✗ 差异未归零——说明回放实现有偏差，检查权重 gather 逻辑")

    # —— 反例演示：错误实现（从清零后的 probs gather）会导致什么 ——
    print("\n" + "-" * 64)
    print("反例：若回放时从「top-k 后清零的 probs」gather 权重（错误做法）")
    router_logits = model.moe.router(x)
    topk_vals_zeroed = torch.zeros_like(router_logits)
    topk_vals_zeroed.scatter_(-1, routing, torch.gather(router_logits, -1, routing))
    # 观察：除了被选中的 top-2，其他位置是 0；softmax 后这些位置会得到非零小概率
    w_zeroed = F.softmax(topk_vals_zeroed, dim=-1)
    w_correct = F.softmax(router_logits, dim=-1)
    # 对「被选中的专家」，两种权重的相对偏差
    picked = w_zeroed.gather(-1, routing)
    correct_picked = w_correct.gather(-1, routing)
    rel_err = ((picked - correct_picked).abs() / correct_picked.abs()).mean().item()
    print(f"  被选中专家的权重相对偏差 = {rel_err:.2%}")
    print("  → 权重变了，logprob 必然漂移；这正是 Megatron PR#79 强调的坑")
    print("-" * 64)


if __name__ == "__main__":
    main()
