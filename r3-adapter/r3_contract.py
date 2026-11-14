# -*- coding: utf-8 -*-
"""R3 数据契约适配层 —— 可复用校验库

把「防止别人踩 15.84% 的坑」从「靠人记着」变成「靠代码拦住」。

提供 6 个校验原语，接入方（RL 训练框架）直接 import：

  normalize_routed_experts    dtype 归一化 + 合法性校验 + unmappable 统计
  validate_shape              末 token 无路由的 shape 断言
  handle_null                 被抢占/中止请求的 None 处理
  compare_routes              ★ 对比「记录的路由」与「重算的路由」是否一致
  check_replay_weights        ★ 校验回放权重是否从完整 logits 正确 gather
  verify_replay_consistency   端到端校验回放后 logprob 是否一致（应归零）

依赖：numpy（契约处理，纯 Python 可跑）；torch 仅在用回放相关校验时需要。

用法示例：
    from r3_contract import normalize_routed_experts, check_replay_weights
    arr, unmappable = normalize_routed_experts(vllm_routed, num_experts=256)
    ok, err = check_replay_weights(router_logits, topk_idx, my_replayed_weights)
    if not ok:
        raise ValueError(f"回放权重偏离 {err:.2%}——你很可能从清零后的 probs gather 了")
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. 契约处理（纯 numpy）
# ---------------------------------------------------------------------------

def normalize_routed_experts(routed, num_experts):
    """把任意 dtype 的路由数组统一为 int32，并校验专家 ID 合法性。

    背景：vLLM 官方用 uint8（≤256 专家）/ uint16，部分 fork 文档写 int16，
    Miles 实际用 int32——同一份语义数据三种宽度，体积差 2-4 倍。

    返回 (归一化 int32 数组, unmappable 计数)。
    unmappable = 值为 -1 的条目数（哨兵，保持实时路由；0 填充会静默别名到 expert 0）。
    """
    arr = np.asarray(routed, dtype=np.int32)
    valid = ((arr >= 0) & (arr < num_experts)) | (arr == -1)
    if not valid.all():
        bad = np.unique(arr[~valid])
        raise ValueError(f"非法专家 ID {bad[:8]}：应在 [0,{num_experts}) 或 -1")
    return arr, int((arr == -1).sum())


def validate_shape(routed, prompt_len, gen_len):
    """断言 routed_experts 行数 = prompt_len + gen_len - 1。

    最后一个采样 token 没有路由（它还没通过模型），数组比 token 总数少 1。
    不处理会导致训练侧对齐整体错位一格。
    """
    expected = prompt_len + gen_len - 1
    if routed.shape[0] != expected:
        raise AssertionError(
            f"契约违背：路由 {routed.shape[0]} 行 != {expected} "
            f"(prompt {prompt_len} + gen {gen_len} - 1)"
        )
    return expected


def handle_null(routed, request_aborted):
    """被抢占/中止的请求 routed_experts 为 None，必须显式跳过。

    不能假设字段恒存在，更不能把 None 当空数组（会静默吞掉整条样本）。
    """
    if request_aborted:
        if routed is not None:
            raise ValueError("中止请求应返回 None，却收到了非空路由")
        return None
    if routed is None:
        raise ValueError("正常请求不应返回 None（可能：非 MoE / 开关未开 / 被抢占）")
    return routed


# ---------------------------------------------------------------------------
# 2. 路由一致性校验（★ 回放时路由是否真的对上了）
# ---------------------------------------------------------------------------

def compare_routes(recorded, replayed, mode="set", threshold=None, max_report=10):
    """对比推理侧记录的路由与训练侧重算的路由。

    和 verify_replay_consistency 的分工：
      · 那个校验 logprob（端到端，要跑模型）
      · 这个校验路由本身（早、便宜、能定位到具体层和 token）
    两者不能互相替代：路由一致但权重错 → logprob 会变；
    logprob 巧合一致但路由不同 → 训练侧回放的是错的专家。

    mode="set"    每行比较专家集合（top-k 内部顺序无关）。R3 关心的是
                  「选中了哪些专家」，顺序不影响按索引 gather 的权重。
    mode="strict" 逐元素比较，顺序敏感，仅用于排查 kernel 行为。

    threshold     若给出，结果里附带 passed 字段（agreement >= threshold）。

    返回 dict: agreement / n_slots / n_mismatch / positions / note

    实测参考（A10 + VLLM_BATCH_INVARIANT=1，941 个 token）：
    长 prompt 的不一致率在 0% ~ 0.12% 之间，且随输入变化。
    它不是一个能「调到 0」的量，只能监控。
    """
    a = np.asarray(recorded)
    b = np.asarray(replayed)

    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"期望 [tokens, layers, top_k]，收到 {a.shape} 与 {b.shape}")

    note = None
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        note = f"shape 不一致 {a.shape} vs {b.shape}，比较前 {n} 行"
        a, b = a[:n], b[:n]

    if mode == "strict":
        bad_slot = (a != b).any(axis=-1)
    else:
        bad_slot = (np.sort(a, axis=-1) != np.sort(b, axis=-1)).any(axis=-1)

    total = int(bad_slot.size)
    n_bad = int(bad_slot.sum())
    agreement = 1.0 - n_bad / max(total, 1)

    positions = [
        (int(t), int(l), a[t, l].tolist(), b[t, l].tolist())
        for t, l in np.argwhere(bad_slot)[:max_report]
    ]

    result = {
        "agreement": agreement,
        "n_slots": total,
        "n_mismatch": n_bad,
        "positions": positions,
        "note": note,
    }
    if threshold is not None:
        result["passed"] = agreement >= threshold
    return result


# ---------------------------------------------------------------------------
# 3. 回放权重校验（★ 核心：拦住 15.84% 的坑，需要 torch）
# ---------------------------------------------------------------------------

def check_replay_weights(router_logits, topk_idx, replayed_weights, rtol=1e-5):
    """校验「回放权重」是否从完整 logits 正确 gather。

    R3 回放时，专家权重必须 = softmax( gather(router_logits, topk_idx) )，
    即从「完整的 router logits」里取被选中索引对应的值再 softmax。

    错误做法：从「top-k 后其他位置清零的 probs」里取——softmax 后这些
    被清零位置会分到非零概率，导致选中专家的权重漂移（实测可达 ~16%），
    静默污染梯度（Megatron PR#79 强调的坑）。

    返回 (ok, max_rel_err)。ok=False 表示踩坑了，应改用完整 logits gather。
    """
    import torch
    import torch.nn.functional as F

    correct = F.softmax(torch.gather(router_logits, -1, topk_idx), dim=-1)
    err = (replayed_weights - correct).abs() / (correct.abs() + 1e-9)
    max_err = float(err.max().item())
    return bool(max_err < rtol), max_err


def make_wrong_weights(router_logits, topk_idx):
    """构造「错误做法」的回放权重（从 top-k 后清零的 probs 取），用于测试/演示。

    正确性校验器应对这个输出报 ok=False。
    """
    import torch
    import torch.nn.functional as F

    zeroed = torch.zeros_like(router_logits)
    zeroed.scatter_(-1, topk_idx, torch.gather(router_logits, -1, topk_idx))
    full_probs = F.softmax(zeroed, dim=-1)          # 错误：在清零后的分布上 softmax
    return full_probs.gather(-1, topk_idx)          # 取回选中专家位置的「漂移」权重


def verify_replay_consistency(logprob_rollout, logprob_replay, rtol=1e-6):
    """端到端校验：回放后 logprob 应与 rollout 完全一致（差异归零）。

    R3 的本质是「精确复现」而非「近似逼近」，所以正确实现下差异是精确的 0。
    """
    diff = float((logprob_rollout - logprob_replay).abs().max().item())
    return bool(diff < rtol), diff
