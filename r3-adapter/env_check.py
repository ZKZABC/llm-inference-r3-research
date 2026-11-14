#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R3 环境自检 —— 在跑 r3_verify.py 之前，先确认这台机器到底能不能跑。

设计原则：**每一条检查都对应一个真实踩过的坑**，不检查"理论上该有"的东西。
来源：WSL2 + RTX 5060 8GB + vLLM 0.33 上连续撞到的 6 道门槛（见 SETUP_WSL2.md）。

用法：
    python env_check.py                                  # 只查环境
    python env_check.py --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
                                                         # 连模型一起算显存预算

退出码：0 = 可以跑 / 1 = 有 FAIL / 2 = 有 WARN 但无 FAIL
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
MARK = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}

FINDINGS: list[tuple[str, str, str, str]] = []   # (level, name, detail, fix)
_SECTION = ""


# ─────────────────────────── 基础设施 ───────────────────────────

def section(title: str) -> None:
    global _SECTION
    _SECTION = title
    print(f"\n[{title}]")


def add(name: str, status: str, detail: str, fix: str = "") -> None:
    print(f"  {MARK[status]} {name:<22} {detail}")
    if fix and status in (WARN, FAIL):
        print(f"         {'':<22} 修复：{fix}")
    FINDINGS.append((status, f"{_SECTION}/{name}", detail, fix))


def run(cmd: list[str], timeout: int = 15) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def gib(nbytes: float) -> float:
    return nbytes / (1024 ** 3)


# ─────────────────────────── 各项检查 ───────────────────────────

def check_platform() -> None:
    section("系统与框架")
    wsl = is_wsl()
    osname = f"{platform.system()} {platform.release()}"
    add("平台", OK, f"{osname}{'  (WSL2)' if wsl else ''}")

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    add("Python", OK if sys.version_info >= (3, 9) else FAIL, py)

    # vLLM —— 坑 #1/#6 的前提：offload 要 0.30+，老版本会静默忽略 --cpu-offload-gb
    try:
        v = metadata.version("vllm")
        major, minor = (int(x) for x in v.split(".")[:2])
        if (major, minor) >= (0, 30):
            add("vLLM", OK, v)
        else:
            add("vLLM", WARN, f"{v}（<0.30，--cpu-offload-gb 可能被静默忽略）",
                "升级：pip install -U vllm")
    except metadata.PackageNotFoundError:
        add("vLLM", FAIL, "未安装", "pip install vllm")

    # torch —— 坑 #3/#4 的前提：CUDA 版本决定新架构能否被支持
    try:
        import torch
        cuda = torch.version.cuda or "cpu-only"
        st = OK if torch.version.cuda else WARN
        add("torch", st, f"{torch.__version__} (cuda {cuda})",
            "" if torch.version.cuda else "这是 CPU 版 torch，GPU 推理不可用")
    except ImportError:
        add("torch", WARN, "未安装", "pip install torch（vLLM 会自带，装完 vLLM 即可）")


def check_gpu() -> dict | None:
    section("GPU 硬件")
    if not shutil.which("nvidia-smi"):
        add("nvidia-smi", FAIL, "未找到", "确认 NVIDIA 驱动已安装；WSL 里需 Windows 侧驱动")
        return None

    out = run(["nvidia-smi",
               "--query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version",
               "--format=csv,noheader,nounits"])
    if not out:
        add("nvidia-smi", FAIL, "执行失败", "在终端手动跑 nvidia-smi 看报错")
        return None

    p = [x.strip() for x in out.splitlines()[0].split(",")]
    gpu = dict(name=p[0], total_mib=float(p[1]), used_mib=float(p[2]),
               free_mib=float(p[3]), cap=p[4], driver=p[5])

    add("驱动", OK, gpu["driver"])
    add("GPU", OK, gpu["name"])
    add("总显存", OK, f"{gpu['total_mib'] / 1024:.2f} GiB")
    add("计算能力", OK, f"sm_{gpu['cap'].replace('.', '')}")

    # 坑 #2：gpu_memory_utilization 校验的是「空闲显存」，别的程序占了就到不了 0.95
    used_pct = gpu["used_mib"] / gpu["total_mib"] * 100
    if used_pct > 5:
        add("空闲显存", WARN,
            f"{gpu['free_mib'] / 1024:.2f} GiB（已有 {gpu['used_mib'] / 1024:.2f} GiB 被占用）",
            "关掉浏览器/微信/视频播放器等吃显存的程序")
    else:
        add("空闲显存", OK, f"{gpu['free_mib'] / 1024:.2f} GiB")
    return gpu


def check_cuda_toolchain() -> None:
    section("CUDA 工具链")

    # 坑 #4：libnvrtc-builtins 常在 venv 的 site-packages 里，C 层加载器找不到
    cands: list[str] = []
    for pat in ("/usr/local/cuda*/lib64/libnvrtc-builtins.so*",
                "/usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so*"):
        cands += glob.glob(pat)
    try:
        import site
        paths = list(site.getsitepackages()) + [site.getusersitepackages()]
        for sp in paths:
            if sp:
                cands += glob.glob(os.path.join(sp, "nvidia", "*", "lib", "libnvrtc-builtins.so*"))
    except Exception:
        pass

    if not cands:
        add("libnvrtc-builtins", FAIL, "未找到",
            "pip install nvidia-cuda-nvrtc-cu13（或对应 CUDA 大版本）")
    else:
        target = cands[0]
        try:
            ctypes.CDLL(target)
            add("libnvrtc-builtins", OK, f"可加载 {os.path.basename(target)}")
        except OSError as e:
            add("libnvrtc-builtins", FAIL, f"存在但加载失败：{e}",
                f"export LD_LIBRARY_PATH={os.path.dirname(target)}:$LD_LIBRARY_PATH")

        # 坑 #3 的前置：nvcc 缺失 → FlashInfer / Triton 的 JIT 走不通
        if not shutil.which("nvcc"):
            add("nvcc", WARN, "未找到（FlashInfer JIT 不可用）",
                "若报 'Could not find nvcc'，设 VLLM_USE_FLASHINFER_SAMPLER=0")
        else:
            add("nvcc", OK, (run(["nvcc", "--version"]) or "").splitlines()[-1].strip())

    # 坑 #5：Triton 要 JIT 编译 C 扩展，缺 Python.h 会 fatal error
    import sysconfig
    inc = sysconfig.get_paths().get("include", "")
    pyh = os.path.join(inc, "Python.h")
    if os.path.exists(pyh):
        add("Python.h", OK, pyh)
    else:
        add("Python.h", FAIL, f"缺失 {pyh}",
            "sudo apt install -y python3-dev python3.12-dev（版本号对应你的 Python）")


def check_memory() -> dict:
    section("内存")
    info = {"total_gib": 0.0, "avail_gib": 0.0, "pinned": SKIP}

    if os.path.exists("/proc/meminfo"):
        vals = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.partition(":")
                vals[k.strip()] = int(v.split()[0])  # kB
        info["total_gib"] = vals.get("MemTotal", 0) / 1024 / 1024
        info["avail_gib"] = vals.get("MemAvailable", 0) / 1024 / 1024
        # 注意：这里的"可用"是脚本运行时的快照。模型加载时 safetensors 会 mmap
        # 7~8 GiB 进 page cache，MemAvailable 会明显下降（今晚实测降到 0.6 GiB）。
        add("host 总内存", OK if info["total_gib"] >= 12 else WARN,
            f"{info['total_gib']:.1f} GiB"
            + ("" if info["total_gib"] >= 12 else "（WSL 默认只给一半，跑大模型偏紧）"),
            "Windows 建 %UserProfile%\\.wslconfig 设 memory=10GB / swap=8GB，再 wsl --shutdown")
        add("host 可用", OK if info["avail_gib"] >= 6 else WARN,
            f"{info['avail_gib']:.1f} GiB（模型加载后还会明显下降）")
    else:
        add("host 内存", SKIP, "非 Linux，跳过")

    # ★ 最关键的一条：坑 #6。WSL2 不支持 pinned memory，
    #   导致 UVA offload 静默回退到 device memory（等于完全没 offload）
    try:
        import torch
        if not torch.cuda.is_available():
            add("pinned memory", SKIP, "CUDA 不可用，跳过")
        else:
            try:
                t = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
                del t
                info["pinned"] = OK
                add("pinned memory", OK, "可用（offload 能真正生效）")
            except Exception as e:
                info["pinned"] = FAIL
                add("pinned memory", FAIL,
                    f"不可用：{type(e).__name__}",
                    "WSL2 的已知限制。UVA offload 会静默失效，只能用 --offload-backend prefetch"
                    "（但实测对 GPTQ MoE 只省 ~0.16GB，基本无效）")
    except ImportError:
        add("pinned memory", SKIP, "torch 未安装，跳过")

    return info


def check_model(path: str, gpu: dict | None, mem: dict, util: float, max_len: int) -> None:
    section("模型适配")

    if not os.path.isdir(path):
        add("模型目录", FAIL, f"不存在：{path}")
        return

    shards = glob.glob(os.path.join(path, "*.safetensors"))
    if not shards:
        add("权重文件", FAIL, "目录里没有 *.safetensors",
            "确认下载完整，或改用 --model 指向正确目录")
        return
    total_bytes = sum(os.path.getsize(f) for f in shards)
    model_gib = gib(total_bytes)
    add("模型大小", OK, f"{model_gib:.2f} GiB（{len(shards)} 个分片）")

    cfg: dict = {}
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers") or 0
    n_exp = cfg.get("num_experts") or cfg.get("num_local_experts") or 0
    hidden = cfg.get("hidden_size") or cfg.get("d_model") or 0
    n_heads = cfg.get("num_attention_heads") or 0
    n_kv = cfg.get("num_key_value_heads") or n_heads

    if n_exp:
        add("架构", OK, f"{layers} 层 / {n_exp} 专家 / hidden {hidden}")
    elif layers:
        add("架构", WARN, f"{layers} 层 / 非 MoE（没有 routed_experts，R3 无法验证）",
            "R3 需要 MoE 模型，换一个带专家的模型")
    else:
        add("架构", WARN, "config.json 里读不到层数/专家数")

    if not gpu:
        return

    # KV cache 估算：2(K+V) × layers × kv_heads × head_dim × 2 bytes
    head_dim = (hidden // n_heads) if (hidden and n_heads) else 128
    kv_bytes = 2 * max(layers, 1) * max(n_kv, 1) * head_dim * 2 * max_len
    kv_gib = gib(kv_bytes)

    budget = gpu["free_mib"] / 1024 * util
    overhead = 0.5   # 激活 + 临时 buffer 的经验值
    need = model_gib + kv_gib + overhead

    add("KV cache 估算", OK, f"{kv_gib * 1024:.0f} MiB @ max_model_len={max_len}")
    add("显存预算", OK if need <= budget else FAIL,
        f"需要 {need:.2f} GiB vs 可用 {budget:.2f} GiB"
        f"（{util:.0%} × 空闲），缺口 {max(need - budget, 0):.2f} GiB")

    if need <= budget:
        add("结论", OK, "显存够，无需 offload")
        return

    # ── 超预算 → 必须 offload。这里有两道预算要同时满足：host 内存 + pinned 能力 ──
    shortfall = need - budget
    add("offload 需求", WARN, f"{shortfall:.2f} GiB 权重必须驻留 host")

    # ★ 坑 #6 的真正前置：offload 的权重吃 host RAM。
    #   今晚实测基准：WSL 总内存 9.7 GiB、模型 7.84 GiB，加载后 MemAvailable 只剩 2.60 GiB
    #   ——safetensors 的 mmap 会吃掉约 90% 模型大小的物理内存。
    #   所以不能看"脚本运行时的空闲"，要看"加载完之后还剩多少"。
    total = mem.get("total_gib", 0.0)
    est_after_load = max(total - model_gib * 0.9 - 0.5, 0.0) if total else 0.0
    host_cap = min(mem.get("avail_gib", est_after_load), est_after_load) if total else 0.0

    if host_cap > 0:
        ok_host = shortfall <= host_cap
        add("host 内存预算", OK if ok_host else FAIL,
            f"offload 需 {shortfall:.2f} GiB；模型加载后 host 约剩 {host_cap:.2f} GiB"
            + ("" if ok_host else f"，缺口 {shortfall - host_cap:.2f} GiB"),
            "" if ok_host else
            "给 WSL 加内存：Windows 建 %UserProfile%\\.wslconfig 设 memory=10GB，"
            "再 wsl --shutdown；或减少 offload 量")
    else:
        add("host 内存预算", SKIP, "非 Linux，读不到 /proc/meminfo")

    # 再看 pinned 能力（决定 offload 会不会被静默降级）
    if mem.get("pinned") == FAIL:
        add("offload 可行性", FAIL,
            "pinned memory 不可用 → 两种后端都救不了", "见下方结论")
    elif mem.get("pinned") == OK:
        if host_cap and shortfall > host_cap:
            add("offload 可行性", FAIL,
                "pinned 可用，但 host 内存不够分配 → 会静默回退 device memory",
                "先把 host 内存加上去")
        else:
            add("offload 可行性", WARN, "pinned 可用，能跑但会很慢",
                f"加 --cpu-offload-gb {int(shortfall) + 1} --cpu-offload-params experts "
                f"--offload-backend uva")
    else:
        add("offload 可行性", WARN, "无法判断（缺 torch），建议先装依赖再复检")


def check_quant_stack(model_path: str | None) -> None:
    """GPTQ 加载路径的依赖兼容性

    真实事故（2026-11-14）：
      transformers 5.16 调用 optimum.gptq.GPTQQuantizer.from_dict()，
      但 optimum 2.3 已移除 QuantizeConfig，加载时抛
      NameError: name 'QuantizeConfig' is not defined

    这个检查能在加载模型之前拦住，不用等到跑起来才炸。
    """
    section("量化栈兼容性")

    if not model_path or not os.path.isdir(model_path):
        add("GPTQ 加载路径", SKIP, "未指定 --model，跳过")
        return

    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        add("量化格式", SKIP, "无 config.json")
        return

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    method = str(qc.get("quant_method", "")).lower()

    if method != "gptq":
        add("量化格式", OK, f"{method or 'dense'}，不走 GPTQ 加载")
        return
    add("量化格式", OK, "GPTQ")

    tf_ver, opt_ver = None, None
    try:
        import transformers

        tf_ver = transformers.__version__
    except Exception:
        pass
    try:
        import optimum

        opt_ver = getattr(optimum, "__version__", "?")
    except Exception:
        pass

    has_qc = False
    try:
        from optimum.gptq.quantization import QuantizeConfig  # noqa: F401

        has_qc = True
    except Exception:
        has_qc = False

    has_gptqmodel = False
    try:
        import gptqmodel  # noqa: F401

        has_gptqmodel = True
    except Exception:
        pass

    try:
        tf_major = int(str(tf_ver).split(".")[0])
    except Exception:
        tf_major = 0

    if tf_major >= 5 and not has_qc:
        if has_gptqmodel:
            add(
                "GPTQ 加载路径",
                WARN,
                f"transformers {tf_ver} 会调 optimum.gptq，但 optimum {opt_ver} 已无 QuantizeConfig",
                "用 gptqmodel.GPTQModel.load()（它不走 transformers.quantizers）",
            )
        else:
            add(
                "GPTQ 加载路径",
                FAIL,
                f"transformers {tf_ver} + optimum {opt_ver} 组合必然 NameError: QuantizeConfig",
                "pip install gptqmodel；或降到 transformers 4.x + optimum 1.x",
            )
    else:
        add(
            "GPTQ 加载路径",
            OK,
            f"transformers {tf_ver} / optimum {opt_ver} / gptqmodel={has_gptqmodel}",
        )


def verdict() -> int:
    fails = [f for f in FINDINGS if f[0] == FAIL]
    warns = [f for f in FINDINGS if f[0] == WARN]

    print("\n" + "=" * 66)
    if not fails:
        print("结论：可以跑" + ("（有警告，注意看上面 [WARN]）" if warns else ""))
        print("=" * 66)
        print("\n推荐命令：")
        print("  python r3_verify.py \\")
        print("    --model <你的模型目录> \\")
        print("    --max-model-len 512 \\")
        print("    --gpu-memory-utilization 0.85 \\")
        print("    --enforce-eager")
        return 2 if warns else 0

    print("结论：这台机器跑不了 R3 验证")
    print("=" * 66)
    print(f"\n共 {len(fails)} 项 FAIL：\n")
    for _, name, detail, fix in fails:
        print(f"  · {name}")
        print(f"      {detail}")
        if fix:
            print(f"      → {fix}")
    print("\n可行路径（按成本从低到高）：")
    print("  1. 云 GPU（4090 24GB / A100 40GB）：不用量化、不用 offload，一次绕开所有环境坑")
    print("  2. 换 16GB+ 显存的本地卡")
    print("  3. 先用 CPU 上的微型 MoE 验证数学命题：python mini_moe_r3.py")
    return 1


def main() -> int:
    if platform.system() == "Windows" and not is_wsl():
        print("提示：在 Windows 原生环境检测。vLLM GPU 推理需要 WSL2 或 Linux，")
        print("      本报告仅供参考（GPU/内存部分仍有效）。\n")

    ap = argparse.ArgumentParser(description="R3 环境自检")
    ap.add_argument("--model", default=None, help="模型目录（给出则计算显存预算）")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    print("=" * 66)
    print("R3 环境自检 —— 7 项检查对应 7 个真实踩过的坑")
    print("=" * 66)

    check_platform()
    gpu = check_gpu()
    check_cuda_toolchain()
    mem = check_memory()
    if args.model:
        check_model(args.model, gpu, mem, args.gpu_memory_utilization, args.max_model_len)
    else:
        section("模型适配")
        add("显存预算", SKIP, "未指定 --model，跳过",
            "加 --model <目录> 可计算模型能否装下")

    check_quant_stack(args.model)

    return verdict()


if __name__ == "__main__":
    sys.exit(main())
