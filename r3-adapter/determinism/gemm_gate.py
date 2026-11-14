#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 门禁：确定性 GEMM 验证

回答一个问题：同一批输入放在不同的 M（矩阵行数）下计算，
前 N 行是否 bit-identical？

如果答案是否 —— 固定 shape bucket 不是"优化选项"，是"唯一出路"。

用法:
    python phase1_gemm_gate.py                  # 默认 bf16 + 自动选设备
    python phase1_gemm_gate.py --both           # 对比 deterministic 开 / 关
    python phase1_gemm_gate.py --dtype fp32
    python phase1_gemm_gate.py --device cpu
    python phase1_gemm_gate.py --compile        # 额外测 torch.compile 路径

退出码: 0 = 核心门禁通过, 1 = 未通过
"""

import os

# 必须在 import torch 之前设置，否则 cuBLAS 已经用默认 workspace 初始化
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Qwen1.5-MoE-A2.7B 的真实维度，让测试贴近目标场景
HIDDEN = 2048
INTERMEDIATE = 5632
EXPERT_INTERMEDIATE = 1408

BASE_M = 128


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
    """返回 (是否 bit-identical, 最大绝对差, 分歧元素数, 总元素数)"""
    same = torch.equal(ref, got)
    if same:
        return True, 0.0, 0, ref.numel()
    diff = (ref.float() - got.float()).abs()
    return False, diff.max().item(), int((ref != got).sum().item()), ref.numel()


def make_input(m: int, k: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, k, generator=g, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def make_weight(n: int, k: int, dtype: torch.dtype, device: str, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g, dtype=torch.float32) * 0.02
    return w.to(device=device, dtype=dtype)


def verdict(max_diff: float, dtype: torch.dtype) -> str:
    """把数值差异翻译成人话"""
    tol = {torch.float32: 1e-6, torch.bfloat16: 1e-2, torch.float16: 1e-3}.get(dtype, 1e-5)
    if max_diff == 0:
        return "bit-identical"
    if max_diff < tol:
        return "数值等价，归约顺序不同"
    return "显著差异，疑似 split-K / atomic"


# ---------------------------------------------------------------- 各项检查


def check_decode_scale(w, k, dtype, device) -> list[Check]:
    """
    decode 阶段 M = batch_size，典型 1~64。

    这是最容易被漏测、也最可能出问题的区间：
      M=1      → gemv（矩阵乘向量，串行累加）
      M=2~64   → 小 tile gemm
      M>=128   → 标准 tiled gemm
    三条路径的归约顺序完全不同。
    """
    x1 = make_input(1, k, dtype, device, 0)
    ref = F.linear(x1, w)

    out = []
    for m in (2, 3, 4, 5, 6, 7, 8, 16, 32, 64):
        x = torch.cat([x1, make_input(m - 1, k, dtype, device, 3)], dim=0)
        got = F.linear(x, w)[:1]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"M=1 vs M={m} (decode)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )

    # 第三段：M=2 是否已进入 tiled 路径
    # 决定 decode 的最小可用 batch：若 M=2 与 M=4 一致，最小 batch 取 2 即可
    x2 = make_input(2, k, dtype, device, 8)
    ref2 = F.linear(x2, w)
    for m in (4, 8, 16):
        x = torch.cat([x2, make_input(m - 2, k, dtype, device, 9)], dim=0)
        got = F.linear(x, w)[:2]
        same, md, nd, tot = compare(ref2, got)
        out.append(
            Check(
                f"M=2 vs M={m} (组内)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )

    # 第四段：大 batch 区间内部一致性
    x4 = make_input(4, k, dtype, device, 4)
    ref4 = F.linear(x4, w)
    for m in (32, 64):
        x = torch.cat([x4, make_input(m - 4, k, dtype, device, 6)], dim=0)
        got = F.linear(x, w)[:4]
        same, md, nd, tot = compare(ref4, got)
        out.append(
            Check(
                f"M=4 vs M={m} (组内)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_repeat(w, k, dtype, device, rounds=10) -> Check:
    x = make_input(BASE_M, k, dtype, device, 0)
    ref = F.linear(x, w)
    for i in range(rounds):
        if not torch.equal(ref, F.linear(x, w)):
            return Check("基线重复性", False, f"第 {i + 1} 次与首次不一致")
    return Check("基线重复性", True, f"{rounds} 次调用完全一致")


def check_m_invariance(w, k, dtype, device, targets=(256, 512, 1024)) -> list[Check]:
    x_base = make_input(BASE_M, k, dtype, device, 0)
    ref = F.linear(x_base, w)

    out = []
    for m in targets:
        tail = make_input(m - BASE_M, k, dtype, device, 1)
        x = torch.cat([x_base, tail], dim=0)
        got = F.linear(x, w)[:BASE_M]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"M={BASE_M} vs M={m}",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_unaligned(w, k, dtype, device, targets=(130, 200, 300)) -> list[Check]:
    """非 128 倍数的 M，模拟不做 padding 的情况"""
    x_base = make_input(BASE_M, k, dtype, device, 0)
    ref = F.linear(x_base, w)

    out = []
    for m in targets:
        tail = make_input(m - BASE_M, k, dtype, device, 2)
        x = torch.cat([x_base, tail], dim=0)
        got = F.linear(x, w)[:BASE_M]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"M={BASE_M} vs M={m} (非对齐)",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_position_invariance(w, k, dtype, device) -> list[Check]:
    """
    同一个 x 放在 batch 的不同位置，结果是否一致。

    真实场景里，同一条 prompt 可能是 batch 的第一条，也可能是第三条。
    如果 kernel 对行偏移敏感（tile 边界的对齐处理不同），M 一致也没用。
    """
    x = make_input(BASE_M, k, dtype, device, 0)
    pad = make_input(BASE_M, k, dtype, device, 9)
    ref = F.linear(x, w)

    out = []
    cases = [
        ("位置：行 128-255", torch.cat([pad, x], dim=0), slice(BASE_M, 2 * BASE_M)),
        ("位置：行 256-383", torch.cat([pad, pad, x], dim=0), slice(2 * BASE_M, 3 * BASE_M)),
    ]
    for name, xin, sl in cases:
        got = F.linear(xin, w)[sl]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                name,
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_k_sweep(w, n, dtype, device, ks=(512, 1024, 1408, 2048, 4096, 5632)) -> list[Check]:
    """
    固定 M=128，扫 K。如果 K 越大越容易出现分歧，说明触发了 split-K
    （并行度不足时 cuBLAS 会把 K 维切块并行，归约顺序由此改变）
    """
    m = BASE_M
    out = []
    for k in ks:
        wk = make_weight(n, k, dtype, device)
        x = make_input(m, k, dtype, device, 0)
        ref = F.linear(x, wk)
        got = F.linear(x.clone(), wk)
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"K={k} 重复调用",
                same,
                "一致" if same else f"{nd}/{tot} 不同 · {verdict(md, dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_compile(w, k, dtype, device) -> Check:
    """torch.compile 会生成 Triton kernel，这条路径的确定性要单独验"""
    if not hasattr(torch, "compile"):
        return Check("torch.compile 路径", True, "当前 torch 无 compile", skipped=True)

    def fn(x):
        return F.linear(x, w)

    try:
        compiled = torch.compile(fn, dynamic=False)
    except Exception as e:
        return Check("torch.compile 路径", True, f"不可用 ({type(e).__name__})", skipped=True)

    x_base = make_input(BASE_M, k, dtype, device, 0)
    try:
        ref = compiled(x_base)
        x2 = torch.cat([x_base, make_input(BASE_M, k, dtype, device, 1)], dim=0)
        _ = compiled(x2)
        got = compiled(x_base)
    except Exception as e:
        msg = str(e).split("\n")[0][:52]
        return Check("torch.compile 路径", True, f"执行失败 ({type(e).__name__}: {msg})", skipped=True)

    same, md, nd, tot = compare(ref, got)
    return Check(
        "torch.compile 重入一致性",
        same,
        "一致" if same else f"{nd}/{tot} 不同 · {verdict(md, dtype)}",
        md,
        nd,
        tot,
    )


# ---------------------------------------------------------------- 主流程


def run_suite(dtype: torch.dtype, device: str, strict: bool, use_compile: bool) -> list[Check]:
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    torch.manual_seed(0)
    w = make_weight(INTERMEDIATE, HIDDEN, dtype, device)
    k = HIDDEN

    checks: list[Check] = []
    checks.append(check_repeat(w, k, dtype, device))
    checks += check_decode_scale(w, k, dtype, device)
    checks += check_m_invariance(w, k, dtype, device)
    checks += check_position_invariance(w, k, dtype, device)
    checks += check_unaligned(w, k, dtype, device)
    checks += check_k_sweep(w, INTERMEDIATE, dtype, device)
    if use_compile:
        checks.append(check_compile(w, k, dtype, device))
    return checks


def print_env(dtype, device, strict):
    print("环境")
    print(f"  torch            {torch.__version__}")
    print(f"  cuda             {torch.version.cuda or 'N/A'}")
    print(f"  device           {device}")
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu              {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"  dtype            {dtype}")
    print(f"  CUBLAS_WORKSPACE {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print(f"  deterministic    {strict}")


def print_section(checks: list[Check], title: str) -> None:
    print(f"\n{title}")
    print("─" * 68)
    for c in checks:
        if c.skipped:
            mark = "SKIP"
        else:
            mark = "PASS" if c.passed else "FAIL"
        tail = f"  max={fmt_num(c.max_diff)}" if c.max_diff else ""
        print(f"  [{mark}] {c.name:<28} {c.detail}{tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--both", action="store_true", help="对比 deterministic 开 / 关")
    ap.add_argument("--strict", action="store_true", help="开启 use_deterministic_algorithms")
    ap.add_argument("--compile", action="store_true", help="额外测 torch.compile 路径")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("Phase 1 门禁 · 确定性 GEMM")
    print("=" * 68)

    if device == "cpu":
        print("\n[警告] 当前在 CPU 上跑，CPU 的 BLAS 行为不能代表 GPU。")
        print("       CPU 上通常全 PASS —— 这恰好证明「跨 M 一致」在数学上可行：")
        print("       每个输出元素独立计算、归约顺序固定。")
        print("       GPU 上失败的唯一原因，是并行划分随 M 变化。")
        print("       真机结论请在 CUDA 设备上重跑。")

    print_env(dtype, device, args.strict)

    if args.both:
        print("\n" + "=" * 68)
        print("Round 1 · deterministic = False（默认行为）")
        print("=" * 68)
        torch.use_deterministic_algorithms(False)
        loose = run_suite(dtype, device, strict=False, use_compile=args.compile)
        print_section(loose, "检查项")

        print("\n" + "=" * 68)
        print("Round 2 · deterministic = True（严格模式）")
        print("=" * 68)
        torch.use_deterministic_algorithms(False)
        strict_checks = run_suite(dtype, device, strict=True, use_compile=args.compile)
        print_section(strict_checks, "检查项")

        print("\n" + "=" * 68)
        print("对比")
        print("=" * 68)
        print(f"  {'检查项':<30} {'默认':>8} {'严格':>8}")
        print("  " + "-" * 48)
        for a, b in zip(loose, strict_checks):
            print(f"  {a.name:<30} {'PASS' if a.passed else 'FAIL':>8} {'PASS' if b.passed else 'FAIL':>8}")
        checks = strict_checks
    else:
        if args.strict:
            torch.use_deterministic_algorithms(False)
        checks = run_suite(dtype, device, strict=args.strict, use_compile=args.compile)
        print_section(checks, "检查项")

    core = [
        c
        for c in checks
        if ("M=128 vs M=" in c.name and "非对齐" not in c.name)
        or c.name.startswith("位置")
        or "(decode)" in c.name
    ]
    core_ok = all(c.passed for c in core)
    decode_fail = [c for c in checks if "(decode)" in c.name and not c.passed]
    prefill_fail = [
        c
        for c in checks
        if "M=128 vs M=" in c.name and "非对齐" not in c.name and not c.passed
    ]

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    if core_ok:
        print("  核心门禁通过：M 变化不改变前 N 行，decode 区间也一致。")
        print("  → GEMM 不是污染源。翻转要找 attention / MoE 分组。")
    elif decode_fail and not prefill_fail:
        print("  prefill 区间通过，decode 区间未通过。")
        print("  → 问题精确定位在小 M（1~64）。")
        print("  → 处置：decode 阶段 padding 到固定 batch，prefill 不需要。")
        print("  → 这比全局 bucket 便宜得多，只有 decode 付出冗余代价。")
    else:
        print(f"  核心门禁未通过（{len(core) - sum(c.passed for c in core)} 项）。")
        print("  → 先解决底层 GEMM，不要往上叠 attention / MoE。")
        print("\n  排查顺序：")
        print("    1. 加 --both 对比，确认 deterministic 开关是否起作用")
        print("    2. 看 K 扫描结果：K 小一致、K 大不一致 → split-K")
        print("    3. 若是 split-K：强制 padding M 到固定值，让并行度恒定")
        print("    4. 若重复调用就不一致：kernel 里有 atomic，换实现")
    print()

    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
