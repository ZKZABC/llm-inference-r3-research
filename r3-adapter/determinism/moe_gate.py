#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 门禁：确定性 MoE 验证

GEMM 的裂缝在 gemv/gemm，attention 的裂缝在后端和 mask。
MoE 的裂缝在**分组逻辑**本身 —— 它比前两者更根本，因为：

    「哪个 token 分到哪个 expert」是路由算出来的，
     不是你预先能定的。

所以要验证三个层次的顺序依赖：

  1. combine 顺序   —— 按 slot 累加 vs 按 expert 累加，浮点结果不同
  2. 分组内顺序     —— 同一 expert 内 token 的排列变化
  3. capacity 固定  —— 固定槽位能否消除 batch 依赖

注意：本脚本测的是「逻辑层」的顺序依赖（用 PyTorch 模拟）。
真实 grouped GEMM kernel 的 tile 效应，要等引擎写出来再测。

用法:
    python phase3_moe_gate.py
    python phase3_moe_gate.py --dtype bf16
    python phase3_moe_gate.py --device cpu

退出码: 0 = 逻辑层无问题, 1 = 存在顺序依赖
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

# 简化规模：真实是 60 专家 / hidden 2048
# 这里缩小是为了跑得快，逻辑等价。
# top_k 必须用真实的 4：浮点加法可交换但不可结合，
# k=2 时 combine 顺序无所谓，k>=3 才会暴露问题。
NUM_EXPERTS = 8
HIDDEN = 256
TOP_K = 4


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


class MoE:
    """
    一个可切换分组策略的简化 MoE。

    capacity=None        → 动态分组（vLLM 的做法）
    capacity=C, passes=T → 固定槽位，每个 expert 跑 T 趟
    combine_order        → "slot"（按 top-k 位置）| "expert"（按 expert id）
    """

    def __init__(self, dtype, device, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.Wg = (torch.randn(NUM_EXPERTS, HIDDEN, generator=g) * 0.05).to(device=device, dtype=dtype)
        self.We = nn.ModuleList(
            [nn.Linear(HIDDEN, HIDDEN, bias=False).to(device=device, dtype=dtype) for _ in range(NUM_EXPERTS)]
        )
        with torch.no_grad():
            for m in self.We:
                m.weight.copy_((torch.randn(HIDDEN, HIDDEN, generator=g) * 0.05).to(device=device, dtype=dtype))
        self.dtype = dtype
        self.device = device

    def __call__(self, x, capacity=None, passes=1, combine_order="slot"):
        return self.forward(x, capacity, passes, combine_order)

    def forward(self, x, capacity=None, passes=1, combine_order="slot"):
        M = x.shape[0]
        k = TOP_K

        logits = x @ self.Wg.t()
        w, ids = logits.topk(k, dim=-1)
        w = w.softmax(dim=-1)

        tok = torch.arange(M, device=self.device).repeat_interleave(k)
        exp = ids.flatten()
        wts = w.flatten()

        ys = torch.zeros(M * k, HIDDEN, dtype=self.dtype, device=self.device)

        for e in range(NUM_EXPERTS):
            sel = (exp == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue

            if capacity is None:
                ys[sel] = self.We[e](x[tok[sel]])
            else:
                for p in range(passes):
                    s = sel[p * capacity : (p + 1) * capacity]
                    if s.numel() == 0:
                        break
                    ys[s] = self.We[e](x[tok[s]])

        out = torch.zeros(M, HIDDEN, dtype=self.dtype, device=self.device)

        if combine_order == "slot":
            for j in range(k):
                out = out + ys[j::k] * wts[j::k].unsqueeze(-1)
        else:
            order = torch.argsort(exp, stable=True).tolist()
            for i in order:
                out[tok[i]] = out[tok[i]] + ys[i] * wts[i]

        return out


# ---------------------------------------------------------------- 检查项


def check_combine_order(moe, x) -> Check:
    """
    combine 的累加顺序：按 top-k 位置 vs 按 expert id。

    这是最容易被忽略的一处：为了 grouped GEMM 的效率，
    实现者会把 token 按 expert 排序，combine 时也就按这个顺序累加。
    但浮点加法不满足结合律，顺序一变，结果就变。
    """
    a = moe(x, combine_order="slot")
    b = moe(x, combine_order="expert")
    same, md, nd, tot = compare(a, b)
    return Check(
        "combine: slot vs expert",
        same,
        "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, moe.dtype)}",
        md,
        nd,
        tot,
    )


def check_row_permutation(moe, x) -> Check:
    """
    batch 内 token 的排列顺序变化。

    同一批 token，只是行顺序不同，每个 token 的输出必须跟着行走。
    如果不跟着，说明分组逻辑里有跨 token 的耦合。
    """
    ref = moe(x)

    perm = torch.randperm(x.shape[0], device=x.device)
    got = moe(x[perm])
    inv = torch.argsort(perm)
    got = got[inv]

    same, md, nd, tot = compare(ref, got)
    return Check(
        "batch 内行序重排",
        same,
        "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, moe.dtype)}",
        md,
        nd,
        tot,
    )


def check_capacity_invariance(moe, x1, capacity) -> list[Check]:
    """
    核心测试：固定 capacity 后，同一个 token 放在不同 batch 里是否一致。

    前提：capacity 必须足够大，不产生溢出。
    溢出会丢 token，那是容量问题，不是顺序问题，两者必须分开测。
    """
    ref = moe(x1, capacity=capacity)[0]

    out = []
    for extra in (1, 3, 7):
        tail = torch.randn(extra, HIDDEN, dtype=moe.dtype, device=moe.device) * 0.5
        x = torch.cat([x1, tail], dim=0)

        logits = x @ moe.Wg.t()
        _, ids = logits.topk(TOP_K, dim=-1)
        max_per_expert = int(torch.bincount(ids.flatten(), minlength=NUM_EXPERTS).max())

        if max_per_expert > capacity:
            out.append(
                Check(
                    f"capacity 跨 batch (extra={extra})",
                    True,
                    f"溢出 max={max_per_expert} > C={capacity}，跳过",
                    skipped=True,
                )
            )
            continue

        got = moe(x, capacity=capacity)[0]
        same, md, nd, tot = compare(ref, got)
        out.append(
            Check(
                f"capacity 跨 batch (extra={extra})",
                same,
                "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, moe.dtype)}",
                md,
                nd,
                tot,
            )
        )
    return out


def check_dynamic_vs_fixed(moe, x, capacity) -> Check:
    """
    动态分组 vs 固定 capacity。

    如果 capacity 足够大（不溢出），两者应该完全一样。
    不一样说明固定 capacity 的实现引入了额外差异。
    """
    a = moe(x, capacity=None)
    b = moe(x, capacity=capacity)
    same, md, nd, tot = compare(a, b)
    return Check(
        "动态 vs 固定 capacity",
        same,
        "一致" if same else f"{nd}/{tot} 元素不同 · {verdict(md, moe.dtype)}",
        md,
        nd,
        tot,
    )


def check_overflow(moe, x, capacity) -> Check:
    """
    capacity 不够时会不会丢 token。

    丢 token 是静默的正确性事故：不报错、不崩溃，输出只是少了一部分贡献。
    """
    M = x.shape[0]
    logits = x @ moe.Wg.t()
    _, ids = logits.topk(TOP_K, dim=-1)
    counts = torch.bincount(ids.flatten(), minlength=NUM_EXPERTS)
    max_per_expert = int(counts.max())

    needed = max_per_expert
    if needed <= capacity:
        return Check(
            f"capacity 溢出 (max={needed} ≤ C={capacity})",
            True,
            "未溢出",
            skipped=True,
        )

    a = moe(x, capacity=None)
    b = moe(x, capacity=capacity)
    same, md, nd, tot = compare(a, b)
    return Check(
        f"capacity 溢出 (max={needed} > C={capacity})",
        same,
        "一致（溢出被正确处理）" if same else f"{nd}/{tot} 元素缺失 · 会静默丢 token",
        md,
        nd,
        tot,
    )


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("Phase 3 门禁 · 确定性 MoE")
    print("=" * 68)

    print("\n环境")
    print(f"  torch            {torch.__version__}")
    print(f"  device           {device}")
    print(f"  dtype            {dtype}")
    print(f"  experts / top_k  {NUM_EXPERTS} / {TOP_K}")
    print(f"  hidden           {HIDDEN}")

    moe = MoE(dtype, device)
    torch.manual_seed(0)
    x = torch.randn(args.batch, HIDDEN, dtype=dtype, device=device) * 0.5

    # capacity 取期望值的 1.5 倍
    expect = args.batch * TOP_K / NUM_EXPERTS
    capacity = max(2, int(expect * 1.5) + 1)

    print(f"  capacity         {capacity}  (batch={args.batch}, 期望 {expect:.2f}/expert)")

    checks = []
    checks.append(check_combine_order(moe, x))
    checks.append(check_row_permutation(moe, x))
    checks += check_capacity_invariance(moe, x[:1], capacity)
    checks.append(check_dynamic_vs_fixed(moe, x, capacity))
    checks.append(check_overflow(moe, x, max(1, capacity // 2)))

    print("\n检查项")
    print("─" * 68)
    for c in checks:
        mark = "SKIP" if c.skipped else ("PASS" if c.passed else "FAIL")
        tail = f"  max={fmt_num(c.max_diff)}" if c.max_diff else ""
        print(f"  [{mark}] {c.name:<28} {c.detail}{tail}")

    core = [c for c in checks if not c.skipped]
    n_fail = sum(1 for c in core if not c.passed)

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    if n_fail == 0:
        print("  逻辑层无顺序依赖。")
        print("  → 固定 capacity + 统一 combine 顺序可以保证一致。")
    else:
        print(f"  {n_fail} 项存在顺序依赖。")
        if any("combine" in c.name and not c.passed for c in checks):
            print()
            print("  · combine 顺序敏感 → 这是 MoE 最隐蔽的坑")
            print("    做法：combine 必须按固定的 slot 顺序累加，")
            print("    不能按 expert id 排序后累加（grouped GEMM 的自然顺序）。")
        if any("行序" in c.name and not c.passed for c in checks):
            print()
            print("  · 行序敏感 → 分组逻辑里有跨 token 耦合，必须查")
        if any("跨 batch" in c.name and not c.passed for c in checks):
            print()
            print("  · capacity 跨 batch FAIL → 固定槽位不够，还要固定")
            print("    「槽位里放哪个 token」的映射顺序")
        if any("溢出" in c.name and not c.passed for c in checks):
            print()
            print("  · capacity 溢出丢 token → C 必须够大，或溢出时增加趟数")
            print("    注意：增加趟数会改变形状，破坏确定性，所以 C 要一次给足")
    print()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
