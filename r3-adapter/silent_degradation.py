#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静默降级检测 —— 第 3 类校验

前两类校验管的是"能不能跑"和"数据格式对不对"。
这一类管的是：**日志说做了，但其实没生效。**

这类问题的共同特征：
  · 不报错
  · 不崩溃
  · 数据看起来正常
  · 只在某个数字对不上时才暴露

四个真实案例（来自本项目实测）：

  1. offload 报告搬运 3.01 GB，但 Model loading 仍是 7.85 GiB
     → 只记账没搬运，显存预算全错

  2. 开了 VLLM_BATCH_INVARIANT=1，但日志里没有 matmul_persistent
     → 开关没生效，路由依然不可复现

  3. 量化后端回退到 EMULATION，把 int4 反量化成 BF16
     → 只打一行 WARNING，然后显存涨 3.5 倍

  4. SDPA 指定的后端不可用，静默 fallback 到 math
     → 数值和预期不同，但没有任何报错

用法:
    python silent_degradation.py --log vllm_startup.log
    docker logs vllm 2>&1 | python silent_degradation.py --stdin
    python silent_degradation.py --log x.log --json

退出码: 0 = 无 CRITICAL, 1 = 有 CRITICAL
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    code: str
    severity: str  # CRITICAL / WARN / INFO / OK
    title: str
    evidence: list = field(default_factory=list)
    impact: str = ""
    fix: str = ""


# ---------------------------------------------------------------- 日志解析


def parse_facts(text: str) -> dict:
    """从日志里抽取可用于交叉验证的事实"""
    facts = {
        "model_loading_gib": None,
        "offload_gib": None,
        "kv_cache_gib": None,
        "moe_backend": None,
        "runner": None,
        "batch_invariant": False,
        "sdpa_fallbacks": [],
        "sampling_override": False,
        "pinned_mem_fail": False,
        "flash_attn_abi": False,
        "capture_buffer": None,
        "dequant_bf16": False,
        "emulation": False,
    }

    m = re.search(r"Model loading took\s+([\d.]+)\s*GiB", text, re.I)
    if m:
        facts["model_loading_gib"] = float(m.group(1))

    m = re.search(r"offloaded\s+([\d.]+)\s*GB", text, re.I)
    if m:
        facts["offload_gib"] = float(m.group(1))

    m = re.search(r"Available KV cache[^\d]*([\d.]+)\s*GiB", text, re.I)
    if m:
        facts["kv_cache_gib"] = float(m.group(1))

    m = re.search(r"Using '?(\w+)'?\s+(?:WNA16\s+)?MoE backend", text, re.I)
    if m:
        facts["moe_backend"] = m.group(1).upper()

    if re.search(r"using the V1 model runner instead", text, re.I):
        facts["runner"] = "V1"
    elif re.search(r"Model Runner V2", text, re.I):
        facts["runner"] = "V2"

    facts["batch_invariant"] = bool(
        re.search(r"matmul_persistent|batch_invariant\.py", text)
    )

    for m in re.finditer(r"([A-Za-z ]+) kernel not used because", text):
        facts["sdpa_fallbacks"].append(m.group(1).strip())

    if re.search(r"overridden by the model's\s+`?generation_config", text, re.I):
        facts["sampling_override"] = True

    if re.search(r"Pinned memory is not available", text, re.I):
        facts["pinned_mem_fail"] = True

    if re.search(r"flash_attn.*undefined symbol", text, re.I | re.S):
        facts["flash_attn_abi"] = True

    m = re.search(
        r"allocating buffer with max_tokens=(\d+),\s*num_layers=(\d+),\s*num_experts_per_tok=(\d+)",
        text,
        re.I,
    )
    if m:
        facts["capture_buffer"] = {
            "max_tokens": int(m.group(1)),
            "num_layers": int(m.group(2)),
            "top_k": int(m.group(3)),
        }

    facts["dequant_bf16"] = bool(re.search(r"dequantized to BF16", text, re.I))
    facts["emulation"] = bool(re.search(r"\bEMULATION\b", text))

    return facts


def grab(text: str, pattern: str, n: int = 2) -> list:
    """抓证据行"""
    out = []
    for line in text.splitlines():
        if re.search(pattern, line, re.I):
            out.append(line.strip()[:180])
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------- 检测规则


def run_rules(facts: dict, text: str, model_gib: float = None) -> list:
    F = []

    # ① 量化后端回退
    if facts["emulation"] or facts["dequant_bf16"]:
        F.append(
            Finding(
                code="QUANT_EMULATION",
                severity="CRITICAL",
                title="量化后端回退到 EMULATION",
                evidence=grab(text, r"EMULATION|dequantized to BF16"),
                impact="int4 权重被反量化成 BF16，显存需求涨约 3.5 倍（实测 7.84 GiB → 28 GiB）",
                fix="升级 vLLM 到认识该量化格式的版本；GPTQ MoE 需要 HUMMING 后端（0.30+）",
            )
        )

    # ② offload 是否真的搬运了
    if facts["offload_gib"] is not None and facts["model_loading_gib"] is not None:
        off = facts["offload_gib"]
        load = facts["model_loading_gib"]

        if model_gib is not None:
            expected = model_gib - off
            if load > expected + 0.5:
                F.append(
                    Finding(
                        code="OFFLOAD_NOOP",
                        severity="CRITICAL",
                        title="offload 只记账没搬运",
                        evidence=grab(text, r"offloaded|Model loading took"),
                        impact=(
                            f"声称搬运 {off:.2f} GB，按模型 {model_gib:.2f} GiB 推算应剩 "
                            f"{expected:.2f} GiB，实际仍占 {load:.2f} GiB —— "
                            "显存预算按 offload 后计算会严重低估"
                        ),
                        fix="该组合下 offload 实现有缺陷，换硬件或换量化格式",
                    )
                )
            else:
                F.append(
                    Finding(
                        code="OFFLOAD_OK",
                        severity="OK",
                        title="offload 生效",
                        evidence=grab(text, r"offloaded|Model loading took", 2),
                    )
                )
        else:
            F.append(
                Finding(
                    code="OFFLOAD_UNVERIFIED",
                    severity="WARN",
                    title="offload 量无法核对",
                    evidence=grab(text, r"offloaded|Model loading took"),
                    impact=(
                        f"日志称搬运 {off:.2f} GB，实际加载 {load:.2f} GiB。"
                        "若 offload 生效，加载量应比模型原始大小少约这个数；"
                        "没有原始大小做基准，无法判断是否真的搬运了"
                    ),
                    fix="加 --model-gib <模型实际大小> 重跑，可精确判定",
                )
            )

    # ③ batch invariant 开关未生效
    if not facts["batch_invariant"]:
        F.append(
            Finding(
                code="BATCH_INVARIANT_MISSING",
                severity="CRITICAL",
                title="未检测到 batch invariant 生效痕迹",
                evidence=grab(text, r"batch.invariant|matmul_persistent") or ["（日志中无相关行）"],
                impact="MoE 路由在 batch 组成变化时会翻转（实测 3/768），R3 数据不可复现",
                fix="VLLM_BATCH_INVARIANT=1 启动；若已设置仍无日志，说明该版本不支持此开关",
            )
        )
    else:
        F.append(
            Finding(
                code="BATCH_INVARIANT_OK",
                severity="OK",
                title="batch invariant 已生效",
                evidence=grab(text, r"matmul_persistent|batch_invariant\.py", 1),
            )
        )

    # ④ SDPA 后端 fallback
    if facts["sdpa_fallbacks"]:
        F.append(
            Finding(
                code="SDPA_BACKEND_FALLBACK",
                severity="WARN",
                title="注意力后端发生静默回退",
                evidence=grab(text, r"kernel not used because"),
                impact=(
                    f"以下后端未被使用：{', '.join(sorted(set(facts['sdpa_fallbacks'])))}。"
                    "实测四个后端互不一致（23% 元素差异），回退会改变数值结果"
                ),
                fix="显式指定后端并校验；不要依赖自动 dispatch",
            )
        )

    # ⑤ 采样参数被模型 config 覆盖
    if facts["sampling_override"]:
        F.append(
            Finding(
                code="SAMPLING_OVERRIDE",
                severity="WARN",
                title="采样参数被模型 generation_config 覆盖",
                evidence=grab(text, r"overridden by the model"),
                impact="repetition_penalty 等会改变 logits，间接改变专家选择",
                fix='请求里显式传 {"temperature": 0, "repetition_penalty": 1.0}',
            )
        )

    # ⑥ pinned memory 误判
    if facts["pinned_mem_fail"]:
        F.append(
            Finding(
                code="PINNED_MEM_MISJUDGE",
                severity="WARN",
                title="pinned memory 被判定为不可用",
                evidence=grab(text, r"Pinned memory is not available"),
                impact="UVA offload 会静默失效；但 WSL2 实测支持 pinned memory，这是平台判断误判",
                fix="改 vllm/utils/platform_utils.py 的 is_uva_available() 返回 True（需改源码）",
            )
        )

    # ⑦ flash-attn ABI 不匹配
    if facts["flash_attn_abi"]:
        F.append(
            Finding(
                code="FLASH_ATTN_ABI",
                severity="WARN",
                title="flash-attn 与 torch 的 C++ ABI 不匹配",
                evidence=grab(text, r"undefined symbol|flash_attn"),
                impact="Python 层面的依赖能解析，C++ 层面不会自动跟随；attention 会走 native 路径",
                fix="pip uninstall flash-attn，走 native attention 或重装匹配版本",
            )
        )

    # ⑧ 路由捕获 buffer 的硬上界
    cb = facts["capture_buffer"]
    if cb and cb["max_tokens"] <= 2048:
        F.append(
            Finding(
                code="CAPTURE_BUFFER_SMALL",
                severity="INFO",
                title=f"路由捕获 buffer 上界 max_tokens={cb['max_tokens']}",
                evidence=grab(text, r"allocating buffer with max_tokens"),
                impact="超过该长度的请求，routed_experts 可能被截断或分配失败",
                fix="长序列场景需要调大，或自己管理 buffer",
            )
        )

    # ⑨ 显存余量
    if facts["kv_cache_gib"] is not None and facts["kv_cache_gib"] < 1.0:
        F.append(
            Finding(
                code="KV_CACHE_TIGHT",
                severity="WARN",
                title=f"KV cache 仅 {facts['kv_cache_gib']:.2f} GiB",
                evidence=grab(text, r"Available KV cache"),
                impact="并发能力受限，长序列容易 OOM",
                fix="降低 max-model-len 或 gpu-memory-utilization，或换更大显存",
            )
        )

    # ⑩ Model Runner 回退（不一定是问题）
    if facts["runner"] == "V1":
        F.append(
            Finding(
                code="RUNNER_V1",
                severity="INFO",
                title="Model Runner 回退到 V1",
                evidence=grab(text, r"V1 model runner"),
                impact="文档说 V2 才能捕获路由，但实测 V1 可用；不影响正确性",
                fix="无需处理，但要知道这与文档描述不符",
            )
        )

    return F


# ---------------------------------------------------------------- 输出


SEV_ORDER = {"CRITICAL": 0, "WARN": 1, "INFO": 2, "OK": 3}


def print_report(findings: list, facts: dict):
    print("\n关键事实")
    print("─" * 68)
    for k, v in facts.items():
        if isinstance(v, list):
            v = ", ".join(v) if v else "—"
        if v is None:
            v = "—"
        print(f"  {k:<20} {v}")

    print("\n检测结果")
    print("─" * 68)

    ordered = sorted(findings, key=lambda f: SEV_ORDER.get(f.severity, 9))
    for f in ordered:
        print(f"\n  [{f.severity}] {f.code}")
        print(f"      {f.title}")
        if f.evidence:
            for e in f.evidence:
                print(f"      > {e}")
        if f.impact:
            print(f"      影响：{f.impact}")
        if f.fix:
            print(f"      处置：{f.fix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="vLLM 启动日志文件")
    ap.add_argument("--stdin", action="store_true", help="从标准输入读日志")
    ap.add_argument("--model-gib", type=float, default=None, help="模型原始大小，用于核对 offload 是否生效")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if args.stdin or not args.log:
        text = sys.stdin.read()
    else:
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if not text.strip():
        print("日志为空。用 --log 指定文件，或 --stdin 从管道读。")
        return 1

    facts = parse_facts(text)
    findings = run_rules(facts, text, model_gib=args.model_gib)

    if args.json:
        print(json.dumps({"facts": facts, "findings": [asdict(f) for f in findings]}, ensure_ascii=False, indent=2))
    else:
        print("=" * 68)
        print("静默降级检测")
        print("=" * 68)
        print(f"\n  日志长度   {len(text.splitlines())} 行")
        print_report(findings, facts)

        crit = [f for f in findings if f.severity == "CRITICAL"]
        print("\n" + "=" * 68)
        print("结论")
        print("=" * 68)
        if crit:
            print(f"  {len(crit)} 项 CRITICAL，这台机器上的 R3 数据不可直接使用：")
            for f in crit:
                print(f"    · {f.title}")
        else:
            print("  无阻断性问题。但 WARN 项仍会改变数值，契约校验时要注意。")
        print()

    return 1 if any(f.severity == "CRITICAL" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
