# -*- coding: utf-8 -*-
"""R3 最小验证脚本 —— 在 vLLM 上验证「路由契约」是否成立（需要 GPU + MoE 模型）

这是《报告》§9.4 最小验证方案的可执行实现。回答一个决定性问题：

    推理侧选中的专家，能否被训练侧原样回放？

本脚本只做「契约成立」的前两步（拿到路由 + 校验形状/合法性/确定性）；
真正的「训练侧回放后 logprob 差异归零」需要 Megatron/transformers 训练前向，
属于阶段一的后半段（见文末 TODO）。

──────────────────────────────────────────────────────────────
运行前提（当前机器是 Windows + 8GB 卡，vLLM 未装，需先准备环境）：
  1. Linux 或 WSL2（vLLM 主要支持 Linux；Windows 原生支持不完整）
  2. pip install vllm
  3. 一个 MoE 模型（dense 模型没有专家路由，无法验证 R3）
     · 最小：Qwen/Qwen1.5-MoE-A2.7B（约 5GB，8GB 卡可跑）
     · 官方示例用：Qwen/Qwen3-30B-A3B（需多卡或量化）

运行：
  MODEL=Qwen/Qwen1.5-MoE-A2.7B python r3_verify.py
  # 或显式传参：
  python r3_verify.py --model Qwen/Qwen1.5-MoE-A2.7B --tp 1
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

# 批不变性：保证同一 prompt 在不同 batch 组成下路由一致（报告 §9.4 第 1 步）。
# 缺失时 MoE 的 top-k 可能因数值差异产生不同路由，R3 回放就失去意义。
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")


def _force_pin_memory_available() -> bool:
    """坑 #7：vLLM 在 WSL 上误判 pinned memory 不可用。

    实测（WSL2 + RTX 5060）：torch.empty(pin_memory=True) 能正常分配 3 GB，
    但 vLLM 的 is_pin_memory_available() 返回 False，于是 UVA offload 静默回退
    到 device memory —— 权重根本没搬走，KV cache 预算变成负数（-0.46 GiB）。

    这里强制覆盖成 True，让 offload 真正生效。
    这是「框架的防御性降级把可用路径变成不可用路径」的典型案例。
    """
    import importlib

    candidates = [
        "vllm.utils.buffer_utils",
        "vllm.utils",
        "vllm.worker.buffer_utils",
        "vllm.v1.worker.buffer_utils",
    ]
    for name in candidates:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(m, "is_pin_memory_available"):
            m.is_pin_memory_available = lambda: True
            print(f"  [patch] 强制 {name}.is_pin_memory_available() -> True")
            return True

    print("  [patch] 未找到 is_pin_memory_available，跳过（offload 可能仍会失效）")
    return False


_force_pin_memory_available()

import argparse

import numpy as np


def run_once(model: str, tp: int, prompt: str, max_tokens: int,
             max_model_len: int, gpu_util: float, dtype: str,
             cpu_offload_gb: float, cpu_offload_params: str,
             offload_backend: str, enforce_eager: bool = False,
             offload_group_size: int = 0, offload_num_in_group: int = 1,
             max_num_batched_tokens: int = 0, max_num_seqs: int = 0):
    """跑一次推理，返回 (routed_experts, prompt_len, gen_len, num_experts)。"""
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model,
        enable_return_routed_experts=True,  # 关键开关
        tensor_parallel_size=tp,
        max_model_len=max_model_len,        # 8GB 卡建议 512~2048，省 KV cache
        gpu_memory_utilization=gpu_util,    # 8GB 卡建议 0.85~0.95，预留系统显存
        dtype=dtype,                        # 4bit 模型会自动识别，也可显式 half
        enforce_eager=enforce_eager,        # 坑 #3：Blackwell(sm_120) 上 CUDA graph 可能编译失败
    )
    # 坑 #8：profile_run 用 dummy batch 测峰值显存，MoE 上会把 offload 的专家全拉回 GPU，
    # 导致 offload 白做。压小 batch 能让峰值显存降下来。
    if max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs > 0:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    # 8GB 卡装 8GB 权重 → 用 MoE 专家 offload
    # 坑 #1（实测于 vLLM 0.33.0）：Python API 里 cpu_offload_params 是 set[str]，
    # 传字符串会 pydantic ValidationError: "Input should be a valid set"。
    # CLI 的 --cpu-offload-params experts 会自动转 set，但 LLM(...) 不会，必须自己转。
    if cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = cpu_offload_gb
        if cpu_offload_params:
            if isinstance(cpu_offload_params, str):
                cpu_offload_params = {p.strip() for p in cpu_offload_params.split(",") if p.strip()}
            llm_kwargs["cpu_offload_params"] = set(cpu_offload_params)
        if offload_backend:
            llm_kwargs["offload_backend"] = offload_backend
    # 坑 #6：prefetch 后端必须给 group_size，否则
    # "offload_num_in_group (1) must be <= offload_group_size (0)"
    if offload_group_size > 0:
        llm_kwargs["offload_group_size"] = offload_group_size
        llm_kwargs["offload_num_in_group"] = offload_num_in_group
        if offload_backend:
            llm_kwargs["offload_backend"] = offload_backend
    llm = LLM(**llm_kwargs)
    # 从模型配置读专家数（用于合法性校验）
    num_experts = getattr(llm.model_config.hf_text_config, "num_experts", 0) or \
        getattr(llm.model_config.hf_text_config, "num_local_experts", 0)
    assert num_experts > 0, "无法从模型配置读到 num_experts（确认是 MoE 模型）"

    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens))[0]
    completion = out.outputs[0]
    routed = completion.routed_experts

    # 被抢占/中止会返回 None（报告 §7.4 静默失败之二）
    assert routed is not None, "routed_experts 为 None（可能：非 MoE / 开关未开 / 请求被抢占）"

    prompt_len = len(out.prompt_token_ids)
    gen_len = len(completion.token_ids)
    return routed, prompt_len, gen_len, num_experts


def validate(routed: np.ndarray, prompt_len: int, gen_len: int, num_experts: int):
    """校验契约：shape、dtype、合法性、unmappable 统计。"""
    # 1) shape：末 token 无路由，行数 = prompt + gen - 1
    expected = prompt_len + gen_len - 1
    assert routed.shape[0] == expected, (
        f"契约违背：路由 {routed.shape[0]} 行 != {expected} (prompt {prompt_len} + gen {gen_len} - 1)"
    )
    print(f"  ✓ shape = {routed.shape}（末 token 无路由，行数正确）")
    print(f"  ✓ dtype = {routed.dtype}（vLLM 官方：≤256 专家 uint8，否则 uint16）")

    # 2) 合法性：专家 ID ∈ [0, num_experts) 或 -1 哨兵
    arr = np.asarray(routed)
    valid = ((arr >= 0) & (arr < num_experts)) | (arr == -1)
    unmappable = int((arr == -1).sum())
    assert valid.all(), f"非法专家 ID：{np.unique(arr[~valid])[:8]}"
    print(f"  ✓ 专家 ID 合法，范围 [{arr.min()}, {arr.max()}]，"
          f"unmappable(-1) = {unmappable}")


def main():
    ap = argparse.ArgumentParser(description="R3 路由契约最小验证")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen1.5-MoE-A2.7B"))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")))
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="8GB 卡建议 2048 或更短，省 KV cache")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="8GB 卡建议 0.9，预留系统显存")
    ap.add_argument("--dtype", default="auto", help="4bit 模型用 auto，bf16 可显式 half")
    ap.add_argument("--cpu-offload-gb", type=float, default=0,
                    help="CPU offload 预算 GiB（8GB 卡跑 8GB 模型建议 6~8）")
    ap.add_argument("--cpu-offload-params", default=None,
                    help="只 offload 指定参数，MoE 用 experts（只放稀疏专家，几乎不掉速）")
    ap.add_argument("--offload-backend", default="uva",
                    help="uva=按需读（MoE 用这个）| prefetch=整层预取（不适合 MoE）| auto")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="关 CUDA graph，绕过 Blackwell(sm_120) 内核编译失败；代价是慢一些")
    ap.add_argument("--offload-group-size", type=int, default=0,
                    help="prefetch 后端专用：每组层数（0=不启用 prefetch）")
    ap.add_argument("--offload-num-in-group", type=int, default=1,
                    help="prefetch 后端专用：每组中常驻 GPU 的层数（须 <= group-size）")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="限制 prefill 批次（默认 8192）。MoE 上压小可避免 profile_run 把专家全拉回")
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="最大并发序列数（默认 256）。压到 1 可进一步降 profile 峰值")
    ap.add_argument("--deterministic", action="store_true",
                    help="跑两次并对比路由是否一致（验证批不变性）")
    args = ap.parse_args()

    prompt = "Explain quantum computing in simple terms:"

    print("=" * 60)
    print(f"R3 最小验证：model={args.model}, tp={args.tp}, "
          f"max_model_len={args.max_model_len}, gpu_util={args.gpu_memory_utilization}")
    if args.cpu_offload_gb > 0:
        print(f"  CPU offload: {args.cpu_offload_gb} GiB, params={args.cpu_offload_params}, "
              f"backend={args.offload_backend}")
    print("=" * 60)

    print("\n[第 1 次推理]")
    r1, p1, g1, n_exp = run_once(args.model, args.tp, prompt, args.max_tokens,
                                 args.max_model_len, args.gpu_memory_utilization, args.dtype,
                                 args.cpu_offload_gb, args.cpu_offload_params,
                                 args.offload_backend, args.enforce_eager,
                                 args.offload_group_size, args.offload_num_in_group,
                                 args.max_num_batched_tokens, args.max_num_seqs)
    validate(r1, p1, g1, n_exp)
    print(f"  ✓ 契约成立：拿到 {r1.shape[0]} 行路由，模型共 {n_exp} 个专家")

    if args.deterministic:
        print("\n[第 2 次推理 —— 验证批不变性]")
        r2, p2, g2, _ = run_once(args.model, args.tp, prompt, args.max_tokens,
                                 args.max_model_len, args.gpu_memory_utilization, args.dtype,
                                 args.cpu_offload_gb, args.cpu_offload_params,
                                 args.offload_backend, args.enforce_eager,
                                 args.offload_group_size, args.offload_num_in_group,
                                 args.max_num_batched_tokens, args.max_num_seqs)
        same = (np.asarray(r1) == np.asarray(r2)).all()
        print(f"  {'✓ 两次路由完全一致' if same else '✗ 两次路由不一致——检查 VLLM_BATCH_INVARIANT'}")
        assert same, "批不变性未满足，R3 回放将失效"

    print("\n" + "=" * 60)
    print("✓ 契约前两步通过。")
    print("TODO（阶段一后半段）：训练侧用同一路由重建前向，对比 logprob 差异；")
    print("     归零则契约完全成立，不归零则已定位到第一个要修的差异点。")
    print("=" * 60)


if __name__ == "__main__":
    main()
