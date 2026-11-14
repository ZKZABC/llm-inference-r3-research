#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 门禁：确定性 Attention 验证

GEMM 的裂缝在 gemv/gemm 之间（M=1 vs M>=2）。
Attention 的裂缝可能在更多地方：

  1. batch 大小变化   → kernel 选择可能变
  2. 后端 dispatch    → flash / mem_efficient / math / cudnn 是四套算法
  3. KV padding 长度  → 同一请求在不同 batch 里被 pad 到不同长度
  4. 因果 mask        → prefill 的归约形状随 seq_len 变

任何一条不成立，R3 契约都会在 attention 这一段断掉。

用法:
    python phase2_attention_gate.py
    python phase2_attention_gate.py --dtype fp32
    python phase2_attention_gate.py --device cpu
    python phase2_attention_gate.py --seq 512       # 改 KV 长度

退出码: 0 = 核心门禁通过, 1 = 未通过
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

# Qwen1.5-MoE-A2.7B 的 attention 配置
NUM_HEADS = 16
HEAD_DIM = 128
NUM_KV_HEADS = 16

BASE_KV = 100


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    max_diff: float = 0.0
    n_diff: int = 0
    total: int = 0
    skipped: bool = False


def fmt_num(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def compare(ref: torch.Tensor, got: torch.Tensor) -> tuple[bool, float, int, int]:
    same = torch.equal(ref, got)
    if same:
        return True, 0.0, 0, ref.numel()
    diff = (ref.float() - got.float()).abs()
    return False, diff.max().item(), int((ref != got).sum().item()), ref.numel()


def verdict(max_diff: float, dtype: torch.dtype) -> str:
    tol = {torch.float32: 1e-6, torch.bfloat16: 1e-2, torch.float16: 1e-3}.get(dtype, 1e-5)
    if max_diff == 0:
        return "bit-identical"
    if max_diff < tol:
        return "数值等价，归约顺序不同"
    return "显著差异"


def rnd(shape, dtype, device, seed) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def sdpa(q, k, v, mask=None, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)


# ---------------------------------------------------------------- 各项检查


def check_repeat(dtype, device, kv_len, rounds=10) -> Check:
    q = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)
    ref = sdpa(q, k, v)
    for i in range(rounds):
        if not torch.equal(ref, sdpa(q, k, v)):
            return Check("基线重复性", False, f"第 {i + 1} 次与首次不一致")
    return Check("基线重复性", True, f"{rounds} 次调用完全一致")


def check_decode_batch(dtype, device, kv_len) -> list[Check]:
    """
    decode 阶段：q 只有 1 个 token，batch 是同时在跑的请求数。
    同一请求单独跑 vs 和别的请求同批，结果必须一致。
    """
    q1 = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k1 = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v1 = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)
    ref = sdpa(q1, k1, v1)

    out = []
    for b in (2, 4, 8):
        q = torch.cat([q1, rnd((b - 1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 10 + b)], dim=0)
        k = torch.cat([k1, rnd((b - 1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 20 + b)], dim=0)
        v = torch.cat([v1, rnd((b - 1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 30 + b)], dim=0)
        got = sdpa(q, k, v)[:1]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"batch=1 vs {b} (decode)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def run_backend(be, q, k, v) -> tuple[torch.Tensor, bool]:
    """
    在指定后端下跑一次，同时检测是否真的用上了该后端。

    sdpa_kernel 在后端不可用时会静默 fallback（只发 warning 不抛异常），
    所以不能只看有没有异常，必须读 warning。
    """
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        with sdpa_kernel(be):
            out = sdpa(q, k, v)
    fell_back = any("not used because" in str(x.message) for x in caught)
    return out, fell_back


def check_backends(dtype, device, kv_len) -> list[Check]:
    """
    后端一致性：同一输入，各后端算出来是否相同。

    如果不同，那「用哪个后端」就成了必须固定的设计决策，
    不能再让 PyTorch 按 shape 自动 dispatch。
    """
    if sdpa_kernel is None:
        return [Check("后端一致性", True, "torch 版本无 sdpa_kernel", skipped=True)]

    q = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)

    candidates = [
        ("FLASH", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ]

    outs = {}
    fell_back_to = {}
    for name, be in candidates:
        try:
            out, fell_back = run_backend(be, q, k, v)
            outs[name] = out
            if fell_back:
                fell_back_to[name] = True
        except Exception:
            continue

    real = [n for n in outs if n not in fell_back_to]
    if len(real) < 2:
        note = f"真正可用的后端仅 {len(real)} 个: {real or '无'}"
        return [Check("后端一致性", True, note, skipped=True)]

    base = real[0]
    out = []
    for other in real[1:]:
        same, md, nd, tot = compare(outs[base], outs[other])
        out.append(
            Check(
                f"{base} vs {other}",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )

    if fell_back_to:
        out.append(
            Check("后端 fallback", True, f"静默回退: {', '.join(fell_back_to)}", skipped=True)
        )
    return out


def check_pad_content(dtype, device, kv_len) -> list[Check]:
    """
    pad 位置填什么值，会不会影响结果？

    如果 mask 是在 softmax 之前生效的，pad 内容应该完全无关。
    如果 mask 不严格（先 softmax 后 mask），pad 内容会泄漏进分母。

    这条比「padding 长度」更根本：它决定你有没有必要清零 pad 区。
    """
    q = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)

    pad_to = kv_len + 28
    mask = torch.zeros(1, 1, 1, pad_to, dtype=torch.bool, device=device)
    mask[..., :kv_len] = True

    out = []
    base_out = None
    for tag, filler in (
        ("零", 0.0),
        ("随机", None),
        ("大值", 100.0),
    ):
        pad_n = pad_to - kv_len
        if filler is None:
            kp = rnd((1, NUM_HEADS, pad_n, HEAD_DIM), dtype, device, 77)
            vp = rnd((1, NUM_HEADS, pad_n, HEAD_DIM), dtype, device, 78)
        else:
            kp = torch.full((1, NUM_HEADS, pad_n, HEAD_DIM), filler, dtype=dtype, device=device)
            vp = torch.full((1, NUM_HEADS, pad_n, HEAD_DIM), filler, dtype=dtype, device=device)

        got = sdpa(q, torch.cat([k, kp], dim=2), torch.cat([v, vp], dim=2), mask=mask)
        if base_out is None:
            base_out = got
            out.append(Check(f"pad 内容 = {tag}", True, "基准"))
            continue

        same, md, nd, tot = compare(base_out, got)
        out.append(
            Check(
                f"pad 内容 零 vs {tag}",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_kv_padding(dtype, device, kv_len) -> list[Check]:
    """
    KV padding 不变性。

    真实场景：同一条请求在 batch=1 时 KV 不 pad，
    在 batch=8 时可能被 pad 到对齐长度。
    被 mask 掉的 pad 位置，不应该影响结果。
    """
    q = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)
    ref = sdpa(q, k, v)

    out = []
    for pad_to in (kv_len + 28, kv_len + 156):
        pad_n = pad_to - kv_len
        k_pad = torch.cat([k, torch.zeros(1, NUM_HEADS, pad_n, HEAD_DIM, dtype=dtype, device=device)], dim=2)
        v_pad = torch.cat([v, torch.zeros(1, NUM_HEADS, pad_n, HEAD_DIM, dtype=dtype, device=device)], dim=2)

        mask = torch.zeros(1, 1, 1, pad_to, dtype=torch.bool, device=device)
        mask[..., :kv_len] = True

        got = sdpa(q, k_pad, v_pad, mask=mask)
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"KV pad {kv_len}→{pad_to}",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_mask_style(dtype, device, seq_len) -> list[Check]:
    """
    分离变量 1：同样的 shape，is_causal=True vs 显式 mask 矩阵。

    如果这两者不一致，那前面所有带 mask 的测试结果都被污染了——
    因为你不知道差异来自 mask 内容还是 mask 表示方式。
    """
    q = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 2)

    ref = sdpa(q, k, v, is_causal=True)

    idx = torch.arange(seq_len, device=device)
    explicit = (idx.view(-1, 1) >= idx.view(1, -1)).view(1, 1, seq_len, seq_len)
    got = sdpa(q, k, v, mask=explicit)

    same, md, nd, tot = compare(ref, got)
    return [
        Check(
            "is_causal vs 显式 mask",
            same,
            "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
            md,
            nd,
            tot,
        )
    ]


def check_chunked_prefill(dtype, device, seq_len) -> list[Check]:
    """
    分离变量 2：真实 chunked prefill（q 长度会变）。

    基准和分块都用显式 mask，只差 q_len 这一个变量。
    这是真实引擎里会发生的事：prefill 被切成多块，第二块的 q 只有剩余长度。
    """
    q = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 2)

    idx = torch.arange(seq_len, device=device)
    full_mask = (idx.view(-1, 1) >= idx.view(1, -1)).view(1, 1, seq_len, seq_len)
    ref = sdpa(q, k, v, mask=full_mask)

    out = []
    for split in (seq_len // 2, seq_len // 4):
        tail = seq_len - split
        q2 = q[:, :, split:, :]

        iq = torch.arange(tail, device=device).view(-1, 1) + split
        ik = torch.arange(seq_len, device=device).view(1, -1)
        mask = (ik <= iq).view(1, 1, tail, seq_len)

        got = sdpa(q2, k, v, mask=mask)
        same, md, nd, tot = compare(ref[:, :, split:, :], got)
        out.append(
            Check(
                f"chunked {split}+{tail} (q_len 变)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_chunked_aligned(dtype, device, seq_len) -> list[Check]:
    """
    分离变量 3：q 长度保持不变的分块（把第二块前部 pad 成 dummy query）。

    如果「q_len 变」的版本 FAIL 而「q_len 不变」的版本 PASS，
    那问题就不在分块，而在 kernel 对 q_len 的敏感性。
    """
    q = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 2)

    idx = torch.arange(seq_len, device=device)
    full_mask = (idx.view(-1, 1) >= idx.view(1, -1)).view(1, 1, seq_len, seq_len)
    ref = sdpa(q, k, v, mask=full_mask)

    out = []
    split = seq_len // 2
    tail = seq_len - split

    dummy = torch.zeros(1, NUM_HEADS, split, HEAD_DIM, dtype=dtype, device=device)
    q_pad = torch.cat([dummy, q[:, :, split:, :]], dim=2)

    iq = torch.arange(seq_len, device=device).view(-1, 1)
    ik = torch.arange(seq_len, device=device).view(1, -1)
    mask_pad = (ik <= iq).view(1, 1, seq_len, seq_len)
    mask_pad[:, :, :split, :] = False

    got = sdpa(q_pad, k, v, mask=mask_pad)[:, :, split:, :]
    same, md, nd, tot = compare(ref[:, :, split:, :], got)
    out.append(
        Check(
            f"chunked {split}+{tail} (q_len 不变)",
            same,
            "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
            md,
            nd,
            tot,
        )
    )
    return out


def check_prefill_causal(dtype, device, seq_len) -> list[Check]:
    """
    prefill：因果 mask 下，不同 seq_len 的前缀是否一致。

    同一段文本被切成不同长度的 chunk 送进来，
    前半部分的 attention 结果不应该受后面有没有 token 的影响。
    """
    q_full = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 0)
    k_full = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 1)
    v_full = rnd((1, NUM_HEADS, seq_len, HEAD_DIM), dtype, device, 2)

    ref = sdpa(q_full, k_full, v_full, is_causal=True)

    out = []
    for cut in (seq_len // 2, seq_len // 4):
        q_c = q_full[:, :, :cut, :]
        k_c = k_full[:, :, :cut, :]
        v_c = v_full[:, :, :cut, :]
        got = sdpa(q_c, k_c, v_c, is_causal=True)
        same, md, nd, tot = compare(ref[:, :, :cut, :], got)
        out.append(
            Check(
                f"prefill 前缀 {cut}/{seq_len}",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


# ---------------------------------------------------------------- 主流程


def run_suite(dtype, device, kv_len, strict):
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    checks = []
    checks.append(check_repeat(dtype, device, kv_len))
    checks += check_decode_batch(dtype, device, kv_len)
    checks += check_backends(dtype, device, kv_len)
    checks += check_kv_padding(dtype, device, kv_len)
    checks += check_pad_content(dtype, device, kv_len)
    checks += check_mask_style(dtype, device, kv_len)
    checks += check_prefill_causal(dtype, device, kv_len)
    checks += check_chunked_prefill(dtype, device, kv_len)
    checks += check_chunked_aligned(dtype, device, kv_len)
    return checks


def print_section(checks, title):
    print(f"\n{title}")
    print("─" * 68)
    for c in checks:
        mark = "SKIP" if c.skipped else ("PASS" if c.passed else "FAIL")
        tail = f"  max={fmt_num(c.max_diff)}" if c.max_diff else ""
        print(f"  [{mark}] {c.name:<26} {c.detail}{tail}")


def detect_backends(dtype, device, kv_len):
    """
    探测当前环境实际能用哪些 SDPA 后端。

    必须用真实 shape 和 dtype 探测：
      - fp32 下 flash/cudnn 一律不可用（只吃 fp16/bf16）
      - 不同 seq_len 也会影响可用性
    """
    if sdpa_kernel is None:
        return ["(无 sdpa_kernel)"]

    q = rnd((1, NUM_HEADS, 1, HEAD_DIM), dtype, device, 0)
    k = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 1)
    v = rnd((1, NUM_HEADS, kv_len, HEAD_DIM), dtype, device, 2)

    available = []
    for name, be in [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ]:
        try:
            _, fell_back = run_backend(be, q, k, v)
            if not fell_back:
                available.append(name)
        except Exception:
            pass
    return available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seq", type=int, default=BASE_KV, help="KV 长度")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # SDPA 的后端选择警告很吵，且 run_backend 内部会局部重新捕获
    import warnings as _w

    _w.filterwarnings("ignore", message=".*kernel not used because.*")
    _w.filterwarnings("ignore", message=".*has been runtime disabled.*")
    _w.filterwarnings("ignore", message=".*Expected query, key and value.*")

    print("=" * 68)
    print("Phase 2 门禁 · 确定性 Attention")
    print("=" * 68)

    if device == "cpu":
        print("\n[提示] CPU 上通常只有 math 后端，后端一致性检查会 SKIP。")
        print("       这个脚本的价值在 GPU 上体现。")

    print("\n环境")
    print(f"  torch            {torch.__version__}")
    print(f"  device           {device}")
    if device.startswith("cuda"):
        print(f"  gpu              {torch.cuda.get_device_name(0)}")
    print(f"  dtype            {dtype}")
    print(f"  heads / dim      {NUM_HEADS} / {HEAD_DIM}")
    print(f"  kv_len           {args.seq}")
    print(f"  可用后端          {', '.join(detect_backends(dtype, device, args.seq))}")

    checks = run_suite(dtype, device, args.seq, strict=False)
    print_section(checks, "检查项")

    core = [c for c in checks if not c.skipped]
    core_ok = all(c.passed for c in core)
    n_fail = sum(1 for c in core if not c.passed)

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    if core_ok:
        print("  attention 在所有 tested 维度上一致。")
        print("  → R3 的裂缝只在 GEMM 的 gemv/gemm 边界。")
    else:
        print(f"  有 {n_fail} 项不一致。分类看：")
        print()
        if any("batch=" in c.name and not c.passed for c in checks):
            print("  · batch 维度 FAIL → decode 必须固定 batch，和 GEMM 同一类问题")
        if any("vs" in c.name and not c.passed and "batch" not in c.name for c in checks):
            print("  · 后端间 FAIL   → 必须显式指定后端，不能靠 dispatch")
            print("    处置：全局锁死一个后端")
        if any("pad" in c.name and "内容" not in c.name and not c.passed for c in checks):
            print("  · padding FAIL  → KV 张量总长度 S 必须固定（不是有效长度）")
            print("    处置：全局固定 KV 容量，所有请求 pad 到同一个 S")
        if any("前缀" in c.name and not c.passed for c in checks):
            print("  · 前缀 FAIL     → 改 S 会改变结果，截断重算不安全")
        var = [c for c in checks if "q_len 变" in c.name]
        same_len = [c for c in checks if "q_len 不变" in c.name]
        if var and any(not c.passed for c in var):
            if same_len and all(c.passed for c in same_len):
                print("  · chunked FAIL，但 q_len 不变时 PASS")
                print("    结论：问题不是「分块」，是 kernel 对 q 长度的敏感性")
                print("    处置：chunked prefill 要么禁用，要么把每个块 q pad 到固定长度")
            else:
                print("  · chunked FAIL（q_len 不变也不一致）→ chunked prefill 禁用")
        elif var:
            print("  · chunked PASS  → chunked prefill 可用")
        if any("mask" in c.name and not c.passed for c in checks):
            print("  · mask 风格 FAIL → is_causal 与显式 mask 不等价，必须统一用一种")
    print()

    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
