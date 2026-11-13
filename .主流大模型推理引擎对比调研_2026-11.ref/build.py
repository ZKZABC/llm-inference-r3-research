# -*- coding: utf-8 -*-
"""生成《主流大模型推理引擎对比调研（2026-11）》工作簿"""

try:
    import openpyxl
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openpyxl>=3.1.0"])
    import openpyxl

import math
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule


def xl_color(css_hex: str) -> str:
    value = css_hex.removeprefix("#").upper()
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {css_hex}")
    return "FF" + value


# 商务蓝配色
XL_HEADER_BG = xl_color("#4472C4")
XL_HEADER_FG = xl_color("#FFFFFF")
XL_ZEBRA = xl_color("#F2F6FC")
XL_BORDER = xl_color("#BFBFBF")
XL_TEXT = xl_color("#1A1A1A")
XL_MUTED = xl_color("#595959")

# 状态色
XL_G_DARK_BG, XL_G_DARK_FG = xl_color("#C6EFCE"), xl_color("#006100")
XL_G_LIGHT_BG, XL_G_LIGHT_FG = xl_color("#E2EFDA"), xl_color("#375623")
XL_Y_BG, XL_Y_FG = xl_color("#FFEB9C"), xl_color("#9C6500")
XL_R_BG, XL_R_FG = xl_color("#FFC7CE"), xl_color("#9C0006")
XL_N_BG, XL_N_FG = xl_color("#E7E6E6"), xl_color("#595959")

FONT_NAME = "微软雅黑"

thin = Side(style="thin", color=XL_BORDER)
BORDER_ALL = Border(left=thin, right=thin, top=thin, bottom=thin)
BORDER_HEADER = Border(left=thin, right=thin, top=thin, bottom=thin)

HEADER_FONT = Font(name=FONT_NAME, size=10, bold=True, color=XL_HEADER_FG)
BODY_FONT = Font(name=FONT_NAME, size=9.5, color=XL_TEXT)
HEADER_FILL = PatternFill("solid", start_color=XL_HEADER_BG, end_color=XL_HEADER_BG)
ZEBRA_FILL = PatternFill("solid", start_color=XL_ZEBRA, end_color=XL_ZEBRA)

HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
TEXT_ALIGN = Alignment(horizontal="left", vertical="top", wrap_text=True)
NUM_ALIGN = Alignment(horizontal="right", vertical="top")


def est_row_height(values, widths, base=14.5, min_h=20, max_h=230):
    """按列宽估算自动换行后的行高"""
    lines = 1
    for value, width in zip(values, widths):
        if value is None:
            continue
        text = str(value)
        segs = text.split("\n")
        need = 0
        capacity = max(int(width * 1.05), 8)
        for seg in segs:
            units = sum(2 if ord(ch) > 127 else 1 for ch in seg)
            need += max(1, math.ceil(units / capacity))
        lines = max(lines, need)
    return min(max_h, max(min_h, lines * base + 6))


def build_table_sheet(wb, title, headers, rows, widths, numeric_cols=(), freeze="B2"):
    ws = wb.create_sheet(title)
    ws.sheet_properties.tabColor = XL_HEADER_BG

    # 表头（第 1 行）
    for idx, name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=idx, value=name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
        cell.border = BORDER_HEADER
    ws.row_dimensions[1].height = 34

    # 列宽
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width

    # 数据区
    first_data_row = 2
    for r_off, row in enumerate(rows):
        r = first_data_row + r_off
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c_idx, value=value)
            cell.font = BODY_FONT
            cell.border = BORDER_ALL
            if c_idx in numeric_cols:
                cell.alignment = NUM_ALIGN
                cell.number_format = "#,##0"
            else:
                cell.alignment = TEXT_ALIGN
            if r_off % 2 == 1:
                cell.fill = ZEBRA_FILL
        ws.row_dimensions[r].height = est_row_height(row, widths)

    last_row = first_data_row + len(rows) - 1
    ws.freeze_panes = freeze
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last_row}"
    ws.sheet_view.zoomScale = 90
    return ws, first_data_row, last_row


wb = Workbook()
wb.remove(wb.active)

# ---------------------------------------------------------------- Sheet1 服务端
H_SERVER = ["类别", "项目", "主导方/起源", "开源协议", "GitHub Stars\n(2026-11)", "核心架构范式",
            "关键机制", "主要优势", "主要缺陷/边界", "硬件 / 部署形态", "最适用场景", "不建议使用于"]
W_SERVER = [9, 13, 22, 15, 12, 20, 40, 52, 52, 26, 32, 30]

ROWS_SERVER = [
    ["服务端", "vLLM",
     "UC Berkeley 起家，现为独立开源社区项目",
     "Apache-2.0", 96857,
     "分页内存管理型服务端引擎",
     "PagedAttention 分页 KV；continuous batching；V1 多进程异步调度；APC 前缀缓存；KV Connector 插件化；多 LoRA 热切换",
     "生态与社区规模最大；新模型 day-0 支持最快；硬件覆盖最广（NVIDIA/AMD/TPU/Trainium/Gaudi/昇腾）；零编译直接起服务；原生 OpenAI 兼容 API；K8s operator 与可观测性最成熟",
     "单卡极致吞吐非第一（Python 层调度开销）；前缀复用不如 RadixAttention 激进；代码库膨胀（open issues 超 1 万），二次开发门槛高；大版本间 API 仍在快速变动，升级需回归测试",
     "NVIDIA、AMD ROCm、Google TPU、AWS Trainium/Inferentia、Intel Gaudi、昇腾",
     "通用生产默认；模型频繁更换；多模型/多 LoRA 多租；需要最快上线",
     "单卡极限吞吐且模型长期不变；需要确定性编译产物与极致尾延迟"],
    ["服务端", "SGLang",
     "LMSYS / UC Berkeley",
     "Apache-2.0", 38804,
     "程序运行时型（前缀复用优先）",
     "RadixAttention 基数树 KV 复用；SGL 前端 DSL；压缩状态机结构化输出；PD/EPD 分离；HiCache 分层缓存；Elastic EP",
     "共享前缀负载的 TTFT 与吞吐显著领先（多轮对话、Agent、RAG）；JSON Schema / 正则约束解码性能最强（语法一次编译跨请求复用）；RL rollout 生态好；对大 MoE 专家并行支持激进",
     "算子与运维生态小于 vLLM（K8s operator、分布式追踪较弱）；前缀不重复时优势消失；版本演进快、文档滞后，排障常需读源码；社区规模约为 vLLM 的 40%",
     "NVIDIA、AMD ROCm",
     "Agent、多轮对话、RAG、结构化/JSON 输出、大 MoE 高并发、RL 训练推理一体",
     "流量前缀高度离散；团队无源码阅读能力；需要最成熟运维工具链"],
    ["服务端", "TensorRT-LLM",
     "NVIDIA",
     "NVIDIA 自有许可\n（非标准 OSI）", 15075,
     "编译式（AOT）引擎",
     "模型编译为 TensorRT engine/plan；融合 CUDA kernel；FP8/FP4(NVFP4)；in-flight batching；多机 TP/PP/EP；PyTorch backend 免编译路径；KVCMv2",
     "NVIDIA 硬件上绝对吞吐与尾延迟最强（公开数据较 vLLM 高 15%-40%，视负载而定）；新硬件特性首发（NVFP4、MTP、投机解码）；与 Triton / NIM / Dynamo 深度集成",
     "锁定 NVIDIA；编译期成本高（70B FP8 约 28 分钟，形状 profile 固定，换模型或换序列长度需重编）；新模型架构支持滞后 1 周至 1 月；engine 黑盒难调试；许可证非纯 Apache，商用需审阅条款",
     "仅 NVIDIA（Hopper / Blackwell 最优）",
     "单模型长期稳定；NVIDIA 独栈；SLO 极严的在线服务；已具备 CI 编译流水线",
     "模型月更；多模型混跑；非 NVIDIA 硬件；小团队快速试验"],
    ["服务端", "LMDeploy",
     "上海人工智能实验室",
     "Apache-2.0", 8140,
     "手写 kernel 的持久化批处理引擎",
     "TurboMind C++/CUDA 引擎 + PyTorch 引擎双轨；W4A16(AWQ) 一等支持；PD 分离；Mooncake Store 集成",
     "低显存（INT4）场景吞吐优秀；国产模型（InternLM / Qwen 系）适配快；中文社区与文档友好；量化压缩工具链完整（量化 + 部署一体）",
     "模型覆盖面明显窄于 vLLM；TurboMind 与 PyTorch 双引擎功能不一致，行为分裂；国际化文档与生态弱；团队规模小，长期路线存在不确定性",
     "NVIDIA、昇腾（部分支持）",
     "显存受限的私有化部署；国产模型；量化压缩优先",
     "需要最广模型覆盖；海外多云环境"],
    ["服务端", "TGI\n(Text Generation Inference)",
     "HuggingFace",
     "Apache-2.0", 10877,
     "已归档，停止演进",
     "Rust Router + Python/CUDA kernel；continuous batching",
     "与 HF Hub 无缝集成；生产级默认观测；曾是行业标配，文档与案例存量丰富",
     "仓库已 archived（最后提交 2026-03-21），不再跟进新模型与新硬件；HF 自家 Inference Endpoints 已转向 vLLM",
     "NVIDIA、AMD",
     "仅维护存量系统",
     "任何新项目（明确不推荐）"],
]

# 服务端与端侧合并为一张总览表，在 ROWS_EDGE 定义之后统一构建

# ---------------------------------------------------------------- Sheet2 端侧
H_EDGE = ["项目", "主导方/起源", "开源协议", "GitHub Stars\n(2026-11)", "核心架构范式",
          "关键机制", "主要优势", "主要缺陷/边界", "部署形态", "最适用场景", "不建议使用于"]
W_EDGE = [13, 20, 14, 12, 20, 40, 50, 52, 26, 32, 28]

ROWS_EDGE = [
    ["端侧/轻量", "llama.cpp",
     "ggml 社区（Georgi Gerganov 发起）",
     "MIT", 134741,
     "可移植 C/C++ 解释执行",
     "GGUF 单文件自描述格式；块量化 Q2_K~Q8_0；mmap 权重映射；CPU/GPU 按层切分卸载；纯 C/C++ 零运行时依赖",
     "硬件覆盖最广（x86 / ARM / Metal / Vulkan / CUDA / 昇腾）；零依赖开箱即用；2-bit 量化也能跑；同机二次加载近瞬时（mmap）；社区与 PR 量极大",
     "不为高并发设计（batch 调度简单）；KV 缓存连续分配，显存浪费约 10%-20%；多卡并行能力有限；缺 PD 分离、多租 LoRA 调度等服务器特性；长上下文能力弱于服务端引擎",
     "CPU、Apple Silicon、消费级 GPU、嵌入式/IoT、安卓/鸿蒙",
     "本地开发、桌面工具、离线与隐私场景、嵌入式分发",
     "数据中心高并发在线服务"],
    ["端侧/轻量", "Ollama",
     "Ollama Inc.",
     "MIT", 186844,
     "模型分发与运行时管理（Go）",
     "模型库 + Modelfile + runner 生命周期管理；GGUF 为后端格式",
     "体验最好（一条命令拉起模型）；模型分发与版本管理成熟；跨平台桌面集成好；开发者心智占有率最高，教程与生态最丰富",
     "本质是发行版与进程管理器，而非高性能推理引擎；并发吞吐低；缺企业级调度、多租与细粒度观测；能力上限受底层 llama.cpp 路线约束",
     "Windows / macOS / Linux 桌面与服务端",
     "本地原型验证；个人开发；内网小流量",
     "生产高并发；多租户计费场景"],
    ["端侧/轻量", "MLC-LLM",
     "MLC AI / CMU",
     "Apache-2.0", 23420,
     "编译式跨端（TVM 栈）",
     "TVM 编译到 WebGPU / Vulkan / Metal / CUDA / ROCm；统一 runtime；编译期算子融合与内存规划；WebLLM 浏览器推理",
     "真正的跨端覆盖（浏览器、iOS/Android、Windows/Linux）；无 Python 运行时依赖；编译期优化空间大；端侧数据不出设备，隐私友好",
     "编译链复杂、上手陡峭；新模型支持滞后（需补算子与编译配置）；调优需理解 TVM；服务端高并发非设计目标；社区规模中等",
     "浏览器 WebGPU、移动端、PC、嵌入式",
     "端侧 App 内嵌；浏览器内推理；跨平台产品",
     "服务端批处理；需要 day-0 新模型"],
]
ALL_ENGINES = ROWS_SERVER + ROWS_EDGE
ws1, r1_start, r1_end = build_table_sheet(
    wb, "引擎总览", H_SERVER, ALL_ENGINES, W_SERVER, numeric_cols=(5,), freeze="C2"
)
# TGI 行整体弱化标记（第 6 行 = 表头 + 第 5 条数据）
for c in range(1, len(H_SERVER) + 1):
    ws1.cell(row=6, column=c).font = Font(name=FONT_NAME, size=9.5, color=XL_MUTED, italic=True)

# ---------------------------------------------------------------- Sheet3 编排层
H_ORCH = ["项目", "定位层级", "主导方", "开源协议", "核心机制", "价值", "局限", "与引擎的关系"]
W_ORCH = [22, 20, 20, 14, 50, 50, 46, 34]

ROWS_ORCH = [
    ["NVIDIA Dynamo 1.0\n(2026-03)", "编排层（引擎之上）", "NVIDIA", "Apache-2.0",
     "Planner 在运行时决策 aggregate / disaggregate；分布式 KV Cache Manager 四层（HBM→DRAM→SSD→对象存储）；NIXL 传输",
     "不重造引擎即可获得 PD 分离与全局 KV 复用；官方称 DeepSeek-R1 671B 达 30x 吞吐（含 EP 与长上下文复合收益），Llama 70B 纯分离收益约 2x",
     "绑定 NVIDIA 生态与 NIXL；组件多、运维复杂度高；收益强依赖负载形态",
     "以 vLLM / SGLang / TensorRT-LLM 为后端，不替代引擎"],
    ["llm-d\n(CNCF Sandbox, 2026-03)", "K8s 原生编排", "Red Hat / Google / IBM 等社区", "Apache-2.0",
     "Gateway API Inference Extension（前缀感知路由成为 K8s 原语）；LeaderWorkerSet 管理多节点副本生命周期",
     "把前缀缓存感知路由与多机副本做成 K8s 标准能力，可移植性好，不锁定单一厂商",
     "强依赖 K8s 成熟度；非 K8s 环境无意义；公开 benchmark 中 RPS 低于 vLLM 自带 Rust router 约 25%",
     "调度 vLLM / SGLang 实例"],
    ["Mooncake", "KV 缓存池化 + PD 分离架构", "Moonshot AI（Kimi）", "Apache-2.0",
     "以 KVCache 为中心的分离式架构；Transfer Engine（RDMA 零拷贝）；Store（内容寻址分布式 KV 池）；预测式早拒绝调度",
     "Kimi 真实负载下多承接 75% 请求；长上下文模拟场景吞吐提升最高 525%；已被 vLLM / SGLang / TensorRT-LLM / LMDeploy / FastDeploy 集成为 L2/L3 缓存后端",
     "需高速网络（RDMA）与独立存储层；部署门槛高；小集群性价比低",
     "作为引擎的 KV Connector / 缓存层被集成"],
    ["LMCache", "KV 缓存复用层", "LMCache 社区", "Apache-2.0",
     "跨实例 KV 缓存复用；多级存储后端；对接 vLLM KV Connector",
     "显著降低重复前缀的 prefill 成本；可独立于引擎版本升级",
     "需额外部署缓存服务；命中率高度依赖流量前缀分布",
     "vLLM 生态插件"],
    ["NIXL", "传输抽象层", "NVIDIA", "Apache-2.0",
     "统一抽象 NVLink / InfiniBand / RoCE / PCIe / SSD 的点对点传输，非阻塞",
     "让 PD 分离的 KV 传输跨异构互联可用；DistServe 实测在 25Gbps 以太网下 KV 传输仅占总延迟不到 0.1%（因放置算法让多数传输走 NVLink）",
     "底层依赖具体网卡与驱动；非 NVIDIA 环境支持有限",
     "被 Dynamo / llm-d / 各引擎调用"],
]
build_table_sheet(wb, "编排层与KV基础设施", H_ORCH, ROWS_ORCH, W_ORCH, freeze="B2")

# ---------------------------------------------------------------- Sheet3.5 零件层
H_PART = ["类别", "项目", "Stars", "核心机制", "为什么值得看", "对自研引擎的启示"]
W_PART = [16, 26, 10, 48, 54, 56]

ROWS_PART = [
    # Kernel 与算子
    ["Kernel 与算子", "FlashInfer", 6900,
     "可组合的 attention / MoE kernel 库；JIT 编译；支持 paged KV、cascade attention、RoPE、sampling 等",
     "vLLM、SGLang、TensorRT-LLM 都在用它，事实上已成为 kernel 中间层",
     "不要自己写 attention kernel。FlashInfer 已覆盖绝大多数形状，自研只在极端形状下才有收益"],
    ["Kernel 与算子", "kt-kernel\n(KTransformers)", 19700,
     "Intel AMX / AVX512 / AVX2 优化的 CPU 端 INT4/INT8 算子；NUMA 感知的 MoE 调度；热专家在 GPU、冷专家在 CPU",
     "CPU 侧算子质量的标杆，可直接集成进 SGLang；也支持昇腾 NPU",
     "若做 CPU+GPU 异构，CPU 算子库是最难也最有价值的部分，也是国产化场景的刚需"],
    ["Kernel 与算子", "Tutel\n(Microsoft)", 1000,
     "MoE 专用优化库，支持 FP8 / NVFP4 / MXFP4，覆盖 GLM / Kimi / DeepSeek / Qwen",
     "微软的 MoE 优化集，含自适应并行策略",
     "MoE 的专家并行策略（capacity factor、token drop、all-to-all 调度）本身就是独立研究课题"],
    ["Kernel 与算子", "KVarN\n(huawei-csl)", 520,
     "vLLM 原生 KV cache 量化后端；免校准，一个开关；宣称 3-5 倍上下文、吞吐高于 FP16、精度接近 FP16",
     "华为团队做的 KV 量化，工程完成度高",
     "KV 量化是降低长上下文成本最直接的杠杆，优先级高于优化 attention 计算"],
    # 投机解码
    ["投机解码", "EAGLE", 2600,
     "EAGLE-1/2/3 官方实现；特征级 draft，而非 token 级",
     "学术界主流投机解码路线，已被 vLLM、SGLang 采纳",
     "除非有独家 draft 模型或训练侧协同，否则直接用现成实现，自研算法追不上迭代速度"],
    ["投机解码", "speculators\n(vLLM 官方)", 955,
     "统一构建、评测、存储投机解码算法的库",
     "说明投机解码正在被「标准化、可插拔化」",
     "引擎应预留算法插槽，而不是把某个算法焊死在调度器里"],
    # 存储层级
    ["存储层级延伸", "inferssd", 1000,
     "从 SSD 流式加载 MoE 专家，运行比 RAM / VRAM 更大的模型；基于 llama.cpp",
     "把内存层级从 HBM→DRAM 延伸到 SSD",
     "「存储即显存」是内存墙的终极解法，但延迟管理（预取、缓存淘汰）是全部难点"],
    ["存储层级延伸", "BigMoeOnEdge", 825,
     "自适应流式优化，目标是在手机 / 笔记本上跑前沿规模 MoE；基于原生 llama.cpp",
     "端侧跑数百 B 级模型的可行性验证",
     "SSD 随机读是瓶颈，预取策略比算子优化更能决定体验"],
    ["存储层级延伸", "llama-modchip", 1200,
     "284B DeepSeek-V4-Flash 在 64GB 纯 CPU 上运行（cold prefill 41.1 tok/s，decode 9.1 tok/s，投机解码最高 25.3 tok/s）",
     "少见的「无 GPU」真实性能基线，数据完整",
     "这就是没有 GPU 时的现实上限，可作为端侧产品的可行性判断依据"],
    # Apple Silicon
    ["Apple Silicon 专用", "Rapid-MLX", 4100,
     "基于 MLX 的 OpenAI/Anthropic 兼容服务；宣称比 mlx-lm 快 4 倍；专注 coding agent 的工具调用可靠性",
     "端侧引擎开始按「Agent 工具调用正确率」而非 tok/s 做优化",
     "Agent 场景的引擎指标应该是任务成功率，而不是吞吐——这是个被主流引擎忽略的评价体系"],
    ["Apple Silicon 专用", "MTPLX", 2700,
     "Apple Silicon 原生 MTP 投机解码；M5 Max 上 125 tok/s；16GB Mac 跑 27B",
     "端侧投机解码的收益远大于服务端",
     "带宽受限场景下，投机解码几乎是免费加速，应在端侧引擎中默认开启"],
    ["Apple Silicon 专用", "HyperQwen", 2700,
     "在单张 24GB 消费级卡上用 vLLM 跑 Qwen3.8-27B：单用户 127 tok/s，64 并发约 1035 tok/s，150k-262k 上下文",
     "消费级显卡的极限调优样本，全部改动以 patch 形式公开",
     "证明「给主流引擎打补丁」也能达到专用引擎效果，是成本最低的验证路径"],
    ["Apple Silicon 专用", "splash", 2200,
     "Apple Silicon 本地引擎，围绕具体模型设计而非通用抽象",
     "「为模型设计引擎」而非「为引擎适配模型」的思路",
     "专用化优于通用化，在端侧尤其明显——通用抽象的代价在内存受限时被放大"],
    ["Apple Silicon 专用", "backburner", 1200,
     "iPhone 通过 USB-C 作为 Mac 的协处理器，共同跑 27B 模型",
     "异构端侧集群的极端实验：把手机变成算力单元",
     "端侧算力可以被「组网」，这是尚未被产品化的方向"],
    # 其他
    ["代理与整合层", "optillm", 4300,
     "推理时优化代理：MoA、MCTS、CoT 等多种推理策略，不改模型也不改引擎",
     "说明「推理质量优化」可以在引擎之上独立成层",
     "引擎与算法的边界正在模糊，proxy 层同样是创新空间，且工程量远小于造引擎"],
    ["代理与整合层", "project-maya", 2900,
     "一个引擎一个面板跑 GLM-5.3 744B / DeepSeek-V4.1-Flash 748B / MiniMax-M3.1 428B；Windows + CUDA/ROCm",
     "面向「自建 GPU 的中小团队」的整合方案",
     "打包体验 + 多模型管理是真实付费需求，性能不是唯一维度"],
    ["代理与整合层", "DeepSpeed", 43300,
     "训练与推理优化库；ZeRO、MoE 并行、异构内存管理",
     "不是推理引擎，但它是 MoE 并行策略与显存优化的源头",
     "很多推理引擎的并行策略源自 DeepSpeed，读它的设计比读引擎实现更能理解「为什么这样切」"],
]

# 零件层与细分路线两张表列结构一致，合并为一张（在 ROWS_ROUND2 定义之后统一构建）

# ---------------------------------------------------------------- Sheet3.6 语言栈与加速器
H_LANG = ["类别", "项目 / 技术", "Stars", "核心机制", "为什么重要", "对自研引擎的启示"]
W_LANG = [16, 30, 10, 48, 50, 56]

ROWS_LANG = [
    # Rust 生态
    ["Rust 生态", "mistral.rs", 7600,
     "Rust + Candle；45+ 模型架构；PagedAttention、连续批处理、FlashAttention V2/V3、ISQ 在位量化、UQFF 格式；多模态（视觉/视频/音频/语音/图像生成）；内置 agentic runtime（Web 搜索、Python 执行、shell、MCP 客户端）；OpenAI + Anthropic 双 API",
     "唯一在功能完整度上对标 vLLM 的 Rust 引擎，且是单二进制部署，v0.9.1（2026-08）仍在活跃迭代",
     "若自研，Rust 能同时拿到「无 GC 停顿」与「无 GIL」。vLLM 的公认短板正是 Python 调度开销，这是结构性差异而非优化差距"],
    ["Rust 生态", "分层：Candle / Ratchet\nBurn / kalosm", "—",
     "Candle 是张量层地基（可编译到 WASM，浏览器近原生）；Ratchet 走 wgpu 面向 WebGPU；Burn 是含 autograd 的完整 DL 框架；kalosm 是高层易用封装",
     "Rust 推理已形成从算子到应用的四层结构，不再是实验品",
     "自研应站在 Candle 之上，不要从张量层写起；浏览器场景直接选 Ratchet"],
    ["Rust 生态", "rustformers/llm、llama-rs\n（均已归档）", "—",
     "早期 Rust 推理库，现均已停止维护，社区迁向 Candle 系",
     "生态洗牌的实证：语言不是护城河，维护才是",
     "选型先看最后提交时间与 archived 标记，比看 star 更可靠（TGI 就是前车之鉴）"],
    # NPU 与嵌入式
    ["NPU 与嵌入式", "FastFlowLM\n(ROCm)", 2000,
     "专为 AMD Ryzen AI NPU 深度优化，C++，宣称分钟级上手",
     "首个把 Ryzen AI NPU 做成可用产品的推理引擎",
     "NPU 编程模型与 GPU 差异极大（无 warp 调度、片上 SRAM 极小），调度器需要重新设计而非移植"],
    ["NPU 与嵌入式", "lemonade\n(lemonade-sdk)", 6200,
     "AMD GPU / NPU 上的本地模型服务，ONNX Runtime 后端，OpenAI 兼容，附带 MCP server",
     "AMD 生态里体验最完整的本地服务层",
     "硬件厂商的 SDK 往往缺少好的服务层，这个空位值得填"],
    ["NPU 与嵌入式", "Tengine\n(阿里 OAID)", 4500,
     "轻量模块化嵌入式推理引擎；支持 ARM / MIPS / RISC-V / x86；NPU / FPGA / NVDLA 后端；ONNX / TensorFlow / PyTorch 前端",
     "硬件谱系最宽的嵌入式引擎，连 RISC-V 与 MIPS 都覆盖",
     "端侧引擎的抽象层必须做到「算子可替换」，否则无法应对硬件碎片化"],
    ["NPU 与嵌入式", "rkllama", 626,
     "Rockchip NPU（RK3588 / RK3576）上的 Ollama 替代，基于 rkllm",
     "单板计算机 NPU 生态的代表",
     "低成本硬件的推理需求真实存在，但工具链极不友好——体验层就是机会"],
    ["NPU 与嵌入式", "LiteRT-LM\n(Google)", 494,
     "Android / iOS / WebGPU 上的 Gemma 系列推理，LiteRT 为 TFLite 后继",
     "移动端官方路径，生态背书强",
     "移动端引擎要额外处理模型分片、内存压力与进程被系统回收"],
    ["NPU 与嵌入式", "MasterAgent\n(OpenSparX)", 976,
     "Qualcomm NPU 上端侧 agent，宣称 100ms 内延迟，零云依赖",
     "证明 agent 可以完全跑在 NPU 上",
     "端侧 agent 的瓶颈是延迟而非吞吐，与服务器优化目标完全相反"],
    # 编译器栈
    ["编译器栈", "mlir-aie\n(Xilinx / AMD)", 716,
     "用 MLIR 方言编程 Ryzen AI NPU（AI Engine），基于开源 MLIR 编译链",
     "算子生成正在从「手写 CUDA」转向「写 MLIR lowering」",
     "新硬件的算子不该手写，应该写 lowering——这是 2026 年与 2023 年最大的方法论差异"],
    ["编译器栈", "Allo\n(Cornell, PLDI'24)", 424,
     "加速器设计与编程框架，可生成 ASIC / FPGA 加速器",
     "学术界的「生成式硬件」路线",
     "长期看，算子与硬件会一起被生成，手写 kernel 的窗口会关闭"],
    # 技术谱系
    ["技术谱系", "Orca\n(OSDI'22) → 连续批处理", "—",
     "迭代级调度：每个 decode step 重新组批，完成的请求立即换出、新请求立即插入",
     "所有现代推理引擎的调度基础",
     "这是入场券，不是差异点。但理解它是理解一切调度的前提"],
    ["技术谱系", "PagedAttention\n(SOSP'23) → 分页 KV", "—",
     "把 KV cache 按块管理，像操作系统虚拟内存一样按需分配，消除外部碎片",
     "vLLM 的立身之本，也是显存利用率从 20% 提到 90% 的原因",
     "同上。跳过这一步自研，等于重走 2023 年"],
    ["技术谱系", "Sarathi-Serve\n(OSDI'24) → chunked prefill", "—",
     "把长 prompt 切成块，与 decode 步骤交错在同一批次，缓解 prefill 对 decode 的干扰",
     "单机即可拿到 PD 分离 30-50% 的收益，零新增基础设施",
     "优先做 chunked prefill，再考虑 PD 分离——这是性价比最高的顺序"],
    ["技术谱系", "DistServe\n(OSDI'24) → PD 分离", "—",
     "prefill 与 decode 分池部署；核心结论是放置（placement）算法比互联带宽更决定成败",
     "PD 分离的理论基础；实测在 25Gbps 以太网下 KV 传输仅占总延迟 0.1%",
     "这是算法题不是工程题：建模与仿真的价值高于调 kernel"],
    ["技术谱系", "Mooncake\n(FAST'25) → KV 池化", "—",
     "以 KVCache 为中心的分离式架构，把 GPU 集群闲置的 CPU / DRAM / SSD 组成分布式 KV 池",
     "Kimi 真实负载多承接 75% 请求；长上下文模拟场景吞吐提升最高 525%",
     "KV 池化已经是独立基础设施，不必自研，接 Mooncake / LMCache 即可"],
    ["技术谱系", "SGLang / RadixAttention\n(NeurIPS'24)", "—",
     "用基数树索引 KV cache，任意两条共享前缀的请求自动复用已算部分",
     "前缀复用的理论最优形式，Agent / RAG 场景的核心收益来源",
     "前缀复用要做就做到树级，线性缓存只能拿到部分收益"],
]

build_table_sheet(wb, "语言栈与加速器版图", H_LANG, ROWS_LANG, W_LANG, numeric_cols=(3,), freeze="B2")

# ---------------------------------------------------------------- Sheet3.7 纵向范本
H_VERT = ["类别", "项目", "Stars", "关键数据", "为什么重要", "对自研引擎的启示"]
W_VERT = [18, 26, 10, 50, 44, 56]

ROWS_VERT = [
    # DeepSeek 六件套（2026-09-30 开源，MIT）
    ["DeepSeek 六件套", "TileLang（昇腾版）", "—",
     "kernel 语言与编译器，直接生成 Ascend C 指令；DeepSeek 训练 V4 系列所用算子已有高性能实现；同一套代码可编译到 GPU 与 NPU",
     "编译器层是「一次编写、多硬件运行」的关键，也是 DeepSeek 敢同时押注两套硬件的原因",
     "自研引擎的算子不要手写。先用 TileLang 这类 DSL 描述，再 lowering 到目标硬件——这是 2026 年的正确姿势"],
    ["DeepSeek 六件套", "DeepGEMM-Ascend", "—",
     "密集 BF16 矩阵乘（4096×7168×16384）达 432 TFLOPS 硬件上限的 99.8%；FP8 达 861/865 TFLOPS（99.5%）；支持 MoE grouped GEMM",
     "接近硬件理论极限，证明「移植到国产芯片」不等于「性能降级」",
     "GEMM 是推理的算力主体，自研不可能超过 DeepGEMM——直接用它，把精力放在别处"],
    ["DeepSeek 六件套", "FlashMLA", "—",
     "稀疏注意力 kernel；prefill 410 TFLOPS（95% 峰值）、decode 360 TFLOPS（83%）；支持 Blackwell 与 Ascend 950，**彻底放弃 Hopper 支持**",
     "放弃 Hopper 是极强的平台押注信号：前沿实验室已不再把 Hopper 当作优化目标",
     "注意力 kernel 必须跟着新硬件走。若 2026 年还在为 Hopper 做优化，方向已经落后"],
    ["DeepSeek 六件套", "DeepEP-Ascend", "—",
     "MoE 专家并行的 all-to-all 通信；FP8 dispatch 373-375 GB/s、combine 345-347 GB/s（EP8），约为物理带宽上限的 90-95%（EP≤32）",
     "EP 通信是 MoE 推理的最大瓶颈，比 attention 更限制扩展性",
     "MoE 引擎的性能上限由 all-to-all 决定，不是由 attention 决定。优化顺序应先通信后计算"],
    ["DeepSeek 六件套", "TileKernels / DeepSelect", "—",
     "TileKernels 提供向量与内存算子，一套 Python 接口同时跑 GPU 与 NPU；DeepSelect 提供 top-k 选择（注意力索引与采样）",
     "补齐算子栈的最后两块，且刻意做成跨平台统一接口",
     "采样与 top-k 看似边缘，但在长上下文与投机解码里是真实热点"],
    # 硬件规格
    ["硬件规格", "昇腾 950DT", "—",
     "144GB HiZQ 2.0 自研内存、4TB/s 带宽、1 PFLOPS（FP8）、2 PFLOPS（MXFP4）",
     "内存容量与带宽已对齐 H200 / B200 级别，不再是「能跑但慢」",
     "硬件参数决定引擎设计：4TB/s 带宽意味着 KV cache 可以更大胆地留在 HBM，而不必急着下沉"],
    ["硬件规格", "华为 SuperPoD Flex", "—",
     "128 卡单层 scale-up（3.2 Tbps）；两层 scale-out 网络可扩至 256K 卡；配套 ASC-COMM 自定义通信算子库",
     "互联规模直接决定专家并行能开多大",
     "大规模 EP 的前提是 scale-up 域足够大；否则 all-to-all 会吃掉全部收益"],
    ["硬件规格", "实测基线：DeepSeek-V4.1-Flash", "—",
     "EP32 + 128K 上下文：2469 output tok/s/卡 @ 5ms/token；5102 tok/s @ 10ms（官方注明不含服务调度与框架负载均衡）",
     "国产栈少见的公开性能基线，且明确标注了测量边界",
     "「不含调度」正是关键：纯算子性能与端到端性能的差距，就是调度器做好做坏的空间"],
    # 长上下文压缩
    ["长上下文", "kvpress (NVIDIA)", 1300,
     "KV cache 压缩统一库，集成 SnapKV / H2O / TOVA 等多种方法，基于 transformers",
     "把散落在论文里的 KV 压缩方法变成可调用组件",
     "长上下文成本的第一杠杆是 KV 压缩，而不是更快的 attention。先压缩，再谈算力"],
    ["长上下文", "KVarN (华为)", 520,
     "vLLM 原生 KV 量化后端；免校准、一个开关；宣称 3-5 倍上下文、吞吐高于 FP16、精度接近 FP16",
     "量化比结构化压缩更容易落地，改动面小",
     "先上 KV 量化（一个开关的成本），再考虑结构化压缩（需要改调度）"],
    ["长上下文", "Ring Attention", 548,
     "环形注意力：把长序列切分到多卡，注意力计算沿环传递",
     "单卡放不下的上下文只能靠序列并行",
     "序列并行是超长上下文唯一的分布式解法，但通信开销随卡数线性增长"],
    ["长上下文", "LoCoMo Refined", 677,
     "长上下文记忆评测集的修正版：更严格的 LLM 判分 + 清洗后的数据",
     "长上下文能力需要可信评测，原版基准被证明存在判分宽松问题",
     "造引擎必须同时准备评测方法。没有可信评测，收益就无法被证明，也就无法被采用"],
]

build_table_sheet(wb, "纵向范本_DeepSeek与昇腾", H_VERT, ROWS_VERT, W_VERT, numeric_cols=(3,), freeze="B2")

# ---------------------------------------------------------------- Sheet3.8 RL 协同
H_RL = ["类别", "机制 / 问题", "Stars", "现状与关键数据", "对推理引擎的接口要求", "机会判断"]
W_RL = [16, 30, 10, 54, 48, 46]

ROWS_RL = [
    ["权重同步", "NCCL 裸广播", "—",
     "一次性广播全部参数，耗时 100-500ms；是多数框架的默认做法",
     "引擎需支持权重热加载而不重启进程", "已解决但慢，不构成差异点"],
    ["权重同步", "1GB 连续 buffer 打包（verl）", "—",
     "把参数打包进 1GB 连续缓冲区再广播，降到约 20ms，约 25 倍提升",
     "引擎需提供参数内存的连续布局与缓冲区接口", "工程优化，非架构问题，可复现"],
    ["权重同步", "LoRA adapter-only 同步", "—",
     "只传适配器增量（rank 32 约 50MB），不到 1ms；16 个主流 RL 库中有 8 个已实现",
     "引擎需支持适配器热切换（vLLM / SGLang 已具备）",
     "**当前最佳实践**：LoRA + 异步训练把同步开销从瓶颈变成近零"],
    ["权重同步", "RDMA P2P（Awex / inclusionAI）", "—",
     "1 万亿参数模型跨 256 张 H20 同步约 16 秒",
     "引擎需支持跨数据中心点对点传输", "多数据中心场景的解法，单集群用不上"],
    ["中断策略", "隐式继续（PipelineRL）", "—",
     "在两个 token 的生成间隙（约 1-10ms）换权重，序列不中断；代价是同一序列的相邻 token 可能来自不同策略版本",
     "引擎需支持 per-token 的策略版本标记", "最优雅也最难实现，需要 token 级溯源"],
    ["中断策略", "abort + prefix-resume（SkyRL / SLIME）", "—",
     "在同步边界中断生成，保存已生成 token 与 KV 状态，换权重后从该前缀续跑",
     "引擎必须支持 **KV cache 状态导出/导入**", "**多数引擎不支持 KV 导出**，这是真实缺口"],
    ["中断策略", "等当前 batch 完成（NeMo-RL / Tunix）", "—",
     "最保守：同步只发生在 batch 边界，in-flight 生成必须跑完；等待时间最长",
     "无特殊接口要求", "落后方案，长序列场景代价极高"],
    ["正确性缺口", "MoE 路由不一致", "—",
     "vLLM 的推理 router 与 Megatron 的训练 router，可能因浮点舍入与 kernel 差异为同一输入选中不同专家；DeepSeek-V3 报告提出「Keep Routing」，现已有 R3（arXiv:2606.02437）论文与多个工程实现",
     "引擎需回传每个 token 的专家路由路径", "已有三个独立实现（NeMo RL 的 R3、阿里 ROLL、Megatron PR #79），不是空白；但边界约定不一致，谁统一谁受益"],
    ["正确性缺口", "采样掩码不匹配", "—",
     "引擎在采样前做 top-p / top-k 掩码，而训练前向看到完整词表分布，有效动作空间不一致，破坏重要性采样恒等式",
     "引擎需回传 truncation mask", "vLLM 已有 SamplingMask 数据结构（token_ids: list[list[int]]），缺口在 API 默认暴露与训练侧消费，不在定义"],
    ["数据契约", "五元组正在成型", "—",
     "vLLM 已在 API 响应中返回 routed_experts（[gen_len, layers, top_k]、int16）与 prompt_routed_experts；SamplingMask 已有数据结构。事实标准由 vLLM 的字段主导，但跨引擎格式不统一",
     "引擎需统一字段名、数组形状与三个边界约定",
     "**不是空白而是缝隙**：格式已有一半，统一剩下那一半的人拿走生态位"],
    ["框架实例", "ROLL（阿里巴巴）", 3500,
     "实现 6 种重要性采样变体（CISPO / TIS / TOPR 等），对 off-policy 修正支持最完整",
     "对引擎的 logprob 与版本信息要求最高", "可参考其对数据契约的实际用法"],
    ["学习路径", "hands-on-modern-rl", 4900,
     "从 RL 基础到 LLM 对齐、RLVR、Agentic RL 的开源课程，含异步训练架构章节",
     "—", "造 RL-native 引擎前，先理解 off-policy 与陈旧度问题"],
]

build_table_sheet(wb, "RL推理协同与数据契约", H_RL, ROWS_RL, W_RL, numeric_cols=(3,), freeze="B2")

# ---------------------------------------------------------------- Sheet3.9 协议实现现状
H_PROTO = ["层级", "项目 / 接口", "关键数据 / 字段", "行为约定", "限制 / 缺口", "可实现的事"]
W_PROTO = [14, 26, 48, 50, 42, 46]

ROWS_PROTO = [
    ["vLLM", "--enable-return-routed-experts", 
     "响应体 routed_experts（base64 编码的 .npy）：[gen_len, num_moe_layers, top_k]，dtype 为 uint8（专家数 ≤256）或 uint16；另有 prompt_routed_experts: [prompt_len, num_moe_layers, top_k]",
     "prompt 路由挂在 RequestOutput 层（n>1 时各 choice 共享，避免重复存储）；生成路由挂在 CompletionOutput（每 choice 独立）",
     "最后一个采样 token 无路由记录（尚未通过模型），数组长度比 token 数少 1",
     "**这就是事实标准**，其他实现应向它对齐，而不是另起一套"],
    ["vLLM", "SamplingMask 数据结构",
     "token_ids: list[list[int]]，每个生成 token 对应一组「通过 top-k / top-p / min-p 后存活的词表 ID」",
     "定义在 vllm/outputs.py，字段为 CompletionOutput.sampling_mask，注释标注 if requested",
     "默认是否回传、JSON API 中的字段名尚未在文档中固定",
     "可作为契约中 mask 字段的参考形状，直接复用而不必重新设计"],
    ["vLLM (fork)", "WeightVersionSpan",
     "version: str、start: int、end: int，语义为「[start, end) 区间的输出 token 由该版本权重采样」",
     "出现在 beclab/vllm fork 的 CompletionOutput.weight_versions 字段，仅最终输出时设置",
     "上游主干是否合并未知",
     "这正是 PipelineRL 式「逐 token 策略版本」所需的最小字段，值得跟踪"],
    ["NeMo RL", "Router Replay (R3)",
     "policy.router_replay.enabled: true；开启后 vLLM 侧自动置 enable_return_routed_experts=True",
     "PR #2590（约 4059 行）；Megatron 在 prev-logprob 与 train forward 中重放专家选择；支持 Sync 与 Async RL；用 R3-on/off 配对实验验证",
     "只测了 vLLM rollout，其他推理后端未接入",
     "目前最完整的参考实现，附带 trace 校验工具与有效性验证方法"],
    ["阿里 ROLL", "R3 实现（SGLang 路线）",
     "SGLang 服务以 enable_return_routed_experts=True 启动；每个 generate 请求带 return_routed_experts=True；从 meta_info 按样本拼装",
     "要求 SGLang >= 0.5.6.post3；**不支持 vLLM rollout**；R3 与 sequence packing 暂不兼容（布局重映射未打通）",
     "与 NeMo RL 选了相反的引擎，两套实现互不兼容",
     "说明 vLLM 系与 SGLang 系在 R3 上已分叉——适配层有真实需求"],
    ["Megatron-LM", "PR #79 的边界处理",
     "router_replay_source: \"rollout\" | \"self_record\"；不可映射 token 使用哨兵值 -1",
     "关键细节：偏离路由若用 0 填充，会静默别名到 expert 0 并杀死梯度；因此 -1 表示「保持实时路由」，并导出 router_replay/unmappable_frac 指标",
     "PR 未合入主干",
     "**边界处理的范本**：哨兵值而非零填充。这类细节决定协议能否被信任"],
    ["理论", "R3 论文 (arXiv:2606.02437)",
     "Rollout Routing Replay：在每次 trainer forward 固定 rollout 时选中的专家索引",
     "从源头消除 train-inference mismatch 中的 routing 分量，而非靠 IS 修正事后补偿",
     "—",
     "术语与正确性论证的来源，写实现文档时应引用"],
    ["边界约定", "最后一个 token", "无路由记录", "vLLM 文档明确：该 token 尚未通过模型", "—",
     "适配层必须显式处理，否则数组与 token 序列长度差 1，训练侧对齐会整体错位"],
    ["边界约定", "中止请求", "routed_experts 返回 null", "vLLM 行为", "—",
     "训练侧消费方必须能容忍 null，不能假设字段恒存在"],
    ["边界约定", "不可映射 token", "哨兵 -1，保持实时路由", "Megatron PR #79 约定", "—",
     "零填充是错误做法；这是最容易踩且最难发现的坑"],
    # ---- 启动约束
    ["启动约束", "四项前置条件", "必须启用 Model Runner V2（否则设 VLLM_USE_V2_MODEL_RUNNER=1）；runner 为 generate；模型必须是 MoE；**prefix caching 必须开启**",
     "路由按 KV block hash 存储并与 prefix 缓存共享，这是强制依赖 prefix caching 的原因",
     "旧版文档建议关掉 prefix caching，是错的", "部署前先核对这四项，否则启动直接 ValueError"],
    ["启动约束", "被拒绝的并行/连接器配置",
     "adaptive speculative verification、PP > 1、DCP/PCP > 1，以及 Nixl / NixlPull / NixlPush / MoRIIO / Mooncake 等 KV connector 均被拒绝；常规投机解码可用",
     "在启动阶段检查并报错，不会静默降级",
     "**与 PD 分离常用的 KV connector 不兼容**，这限制了大规模部署",
     "若必须 PD 分离 + R3，需要自行做适配，现成方案没有"],
    # ---- 开销
    ["开销", "吞吐", "约 2%（random data，ISL=1024，OSL=1024）",
     "开销主要来自 forward 中每层的 .copy_()；async D2H 拷贝完全在后台运行",
     "这是短序列下的数字，长序列需重测", "2% 说明这是可选功能而非免费功能，要按场景开"],
    ["开销", "显存", "int16 buffer、40 层、8192 token、top-22，约 14 MB / GPU",
     "所有 rank 都必须分配真实 buffer（原因见下一条 CUDA graph 陷阱）", "—", "显存代价可忽略"],
    ["开销", "传输体积（真实 RL 场景）",
     "每条 routing 张量 = (num_tokens−1) × layers × top_k × 字节数；Miles 实测 32K token / 60 层 / top_k=8 ≈ 60 MB / 条轨迹",
     "Miles 用 int32，比 vLLM 的 uint8/uint16 大 2-4 倍",
     "**dtype 不统一会直接放大传输成本**", "这 60MB 才是真正要付的钱，不是那 2% 吞吐"],
    ["开销", "精度影响", "GSM8K pass@1 与不开启时完全一致",
     "NVIDIA 的配对验证结果", "—", "零精度损失，说明这是纯工程问题，不是算法妥协"],
    # ---- 陷阱
    ["陷阱", "CUDA graph 与 NCCL 死锁",
     "若只有 rank 0 持有 device buffer，FusedMoE 在 rank 0 与其他 rank 会走不同代码路径，导致 CUDA graph 结构不对称，collective 操作中可能 NCCL 死锁",
     "解法：所有 rank 都分配真实 buffer，保证图结构对称；仅 rank 0 做 D2H 与 host 缓存管理",
     "最容易踩且最难定位的坑，表现为偶发 hang", "实现时必须先保证所有 rank 代码路径一致，再谈性能"],
    ["陷阱", "被抢占的请求静默丢路由",
     "请求被调度器抢占（后续 re-prefill 恢复）时，worker host cache 中已累积的路由被丢弃且不发出，消费方只看到 routed_experts=None，**没有任何其他信号**",
     "官方已知限制", "对 partial rollout / 异步 RL 是致命的：你以为拿到了，其实丢了且不知道",
     "要么禁用抢占（给足 KV headroom），要么在恢复的 prefill 上重建路由"],
    ["陷阱", "异步调度不兼容",
     "异步调度下 worker 的 stop 判断读到 placeholder -1，遇到 EOS / stop token 结束会静默丢路由",
     "config 阶段直接拒绝", "—", "使用同步调度（开启该 flag 时默认即为同步）"],
    ["陷阱", "流式不可用",
     "路由数据只在请求结束时才可得，不能随流式响应增量返回", "官方限制", "—",
     "Agent 长会话要特别注意：必须等整个请求结束才能拿到路由"],
    # ---- 分歧
    ["实现分歧", "dtype 三种写法",
     "vLLM 官方：≤256 专家用 uint8，否则 uint16；部分 fork 文档写 int16；Miles 实际用 int32",
     "同一份语义数据，三种宽度", "**这是协议不统一最硬的证据**",
     "适配层必须做 dtype 归一化，否则体积估算与解析都会错"],
    ["实现分歧", "per-request 开关",
     "vLLM：**没有** per-request 开关，只有服务级；SGLang-JAX：请求级可传列表，如 return_routed_experts=[True, True, False]",
     "SGLang 支持按样本粒度控制，vLLM 只能全局开",
     "vLLM 的粒度更粗", "批量里混有不需要路由的样本时，SGLang 方案更省带宽"],
    ["实现分歧", "序列化格式",
     "HTTP：base64 编码的 .npy 字节流，用 pybase64（SIMD 实现，比标准库快约 3 倍）；offline API：直接返回 ndarray",
     "JSON 无法承载原始字节，只能编码", "**Rust HTTP 前端永不返回该字段**，必须用 Python API server",
     "选型时确认你用的是 Python server，否则字段凭空消失"],
    # ---- 实现参考
    ["实现参考", "Miles 的双游标缓冲",
     "pop_forward() 与 pop_backward() 两个独立游标，同一批路由被前向与反向各消费一次",
     "用于应对激活重算：backward 会重跑 router，必须复放同一份索引",
     "—", "这是训练侧实现的必备结构；用单游标会在重算时错位"],
    ["实现参考", "SGLang-JAX 的 Capturer",
     "RoutedExpertsCapturer 管理 host_buffer 并处理 D2H；extract_routed_experts_from_meta_info 负责解码",
     "旁路捕获管线，与正常推理并行运行；支持 enable_deterministic_sampling",
     "—", "SGLang 路线的参考实现，接口形态与 vLLM 不同"],
    # ---- Sampling Mask（第二个偏差源）
    ["sampling mask", "--return-sampling-mask",
     "启动：vllm serve --return-sampling-mask --logprobs-mode processed_logprobs",
     "实现 DeepSeek-V3.2 技术报告 3.3 节的 Keep Sampling Mask 策略",
     "**引擎级开关**：全局禁用 FlashInfer fused sampler，所有请求都改走 PyTorch 采样路径",
     "开启即全服降速，必须评估后再开"],
    ["sampling mask", "四项使用前提",
     "temperature > 0；top_k > 0；Model Runner V2；--logprobs-mode processed_logprobs",
     "贪婪采样没有截断分布；纯 top-p 会产生词表大小的掩码；processed_logprobs 让返回的 logprob 在 nucleus 上归一化",
     "版本间对投机解码的支持描述不一致（有的文档列为拒绝项）",
     "四项缺一，掩码要么无意义、要么体积爆炸"],
    ["sampling mask", "生成原理",
     "先施加全部 logit processors（bias / penalty / bad words / temperature / min-p），再做 top-k/top-p 把排除项置 -inf；采样后取 torch.isfinite(processed_logits) 即为幸存集合",
     "实际 kernel 为 _compact_sampling_mask_kernel（vllm/v1/worker/gpu/sample/output.py），用 MAX_COMPACT_SUPPORT = 2048 加 bitmask 兜底控制数据量",
     "—", "掩码不是额外计算，而是复用采样时的中间结果，几乎免费"],
    ["sampling mask", "训练侧必须配对使用",
     "keep = zeros(vocab); keep[mask_ids] = True; masked_logits = logits.masked_fill(~keep, -inf); log_prob = log_softmax(masked_logits)[token]",
     "训练侧也必须在同一个支撑集上归一化，否则重要性比率依然有偏",
     "只取 mask 却不设 processed_logprobs，等于白开",
     "**mask 与 processed_logprobs 是一对**，拆开用就是错的"],
    ["sampling mask", "第三个偏差源：score centering",
     "服务器与训练器的 logprob 永不严格相等（kernel 不同、精度不同、策略落后几步）；∇θ log πθ 在采样器下不平均为零，仅靠重要性权重修不掉",
     "训练侧需减去该期望（score_centering=True），并丢弃比率超出 clip 范围的 token",
     "这是 TRL 的实现细节，非 vLLM 内置能力", "routing、mask 之外的第三个独立偏差，要单独修"],
    ["sampling mask", "吞吐代价的实测",
     "num_sampler_logprobs=8 且只有单个 API server 时，生成吞吐减半；改用 --api-server-count 8（vLLM 0.32+）后，top-8 logprob 的开销低于 3%",
     "序列化 top logprobs 是 API server 侧的纯 CPU 工作，不占 GPU",
     "默认配置下代价极大，容易被误判为「这功能不能用」",
     "**最容易低估的成本不是显存，是 CPU 序列化**；加 API server 就能摊掉"],
    ["sampling mask", "可用端点与流式",
     "sampling_mask 在 /inference/v1/generate 返回；stream=true 时每个 chunk 携带其发出 token 的掩码",
     "与 routed_experts「只能在请求结束时返回」的行为不同",
     "OpenAI 兼容端点 /v1/* 不返回该字段",
     "两个字段的可用端点与流式行为都不一样，适配层要分别处理"],
    # ---- 总览
    ["总览", "RL 训练的三个独立偏差源",
     "① 路由偏差（MoE 选错专家）→ R3；② 动作空间偏差（top-p 截断）→ sampling mask；③ 打分偏差（logprob 不一致）→ score centering",
     "三者互相独立，各有对应机制，缺一不可",
     "社区常见只修一个就以为解决了，另外两个仍在污染梯度",
     "**这是本轮最重要的结构性认知：不是一个问题，是三个**"],
]

build_table_sheet(wb, "协议实现现状_路由与掩码", H_PROTO, ROWS_PROTO, W_PROTO, freeze="B2")

# ---------------------------------------------------------------- Sheet4 能力矩阵
H_MATRIX = ["能力维度", "vLLM", "SGLang", "TensorRT-LLM", "LMDeploy", "llama.cpp", "关键说明"]
W_MATRIX = [22, 15, 15, 17, 15, 15, 62]

ROWS_MATRIX = [
    ["分页 KV 内存管理", "极强", "强", "强", "强", "弱", "vLLM 首创 PagedAttention；llama.cpp 连续分配，显存浪费约 10%-20%"],
    ["连续批处理调度", "极强", "强", "强", "强", "弱", "TRT-LLM 为 in-flight batching；llama.cpp 仅有限支持"],
    ["前缀缓存复用", "强", "极强", "强", "强", "弱", "SGLang 的 RadixAttention 用基数树索引，前缀重复率 >60% 时优势明显"],
    ["约束/结构化解码", "强", "极强", "强", "强", "中", "SGLang 用压缩状态机、语法一次编译复用，Agent 场景差异最大"],
    ["投机解码", "强", "强", "极强", "强", "中", "TRT-LLM 首发新算法（如 MTP）；vLLM 支持 EAGLE3 等"],
    ["多 LoRA 多租服务", "极强", "强", "强", "弱", "中", "vLLM 支持适配器热切换，多租成本最低"],
    ["量化能力", "强", "强", "极强", "极强", "强", "TRT-LLM 支持 NVFP4（Blackwell）；LMDeploy 的 W4A16 最成熟；llama.cpp 可到 2-bit"],
    ["PD 分离", "强", "极强", "强", "强", "无", "SGLang 支持 PD 与 EPD（多模态编码器分离）；需 >1000 RPS 或长上下文才划算"],
    ["并行策略广度", "强", "极强", "强", "弱", "弱", "SGLang 在大规模专家并行（EP）上最激进；llama.cpp 多卡能力有限"],
    ["硬件覆盖广度", "极强", "中", "弱", "中", "极强", "vLLM 覆盖 TPU/Trainium/Gaudi；llama.cpp 几乎覆盖所有能跑 C 的平台"],
    ["新模型支持速度", "极强", "强", "中", "中", "中", "vLLM 常为 day-0；TRT-LLM 需等算子与编译支持，滞后 1 周至 1 月"],
    ["部署运维简易度", "极强", "强", "弱", "强", "极强", "TRT-LLM 需编译流水线与 .engine 版本管理；Ollama 类体验最佳"],
    ["二次开发友好度", "中", "强", "弱", "中", "强", "SGLang 代码结构清晰、易改；TRT-LLM 为编译产物黑盒"],
]

ws4, m_start, m_end = build_table_sheet(wb, "能力矩阵", H_MATRIX, ROWS_MATRIX, W_MATRIX, freeze="B2")

# 条件格式：分级染色（C2:F{last}，说明列不染）
matrix_range = f"B{m_start}:F{m_end}"
for label, bg, fg in [
    ("极强", XL_G_DARK_BG, XL_G_DARK_FG),
    ("强", XL_G_LIGHT_BG, XL_G_LIGHT_FG),
    ("中", XL_Y_BG, XL_Y_FG),
    ("弱", XL_R_BG, XL_R_FG),
    ("无", XL_N_BG, XL_N_FG),
]:
    ws4.conditional_formatting.add(
        matrix_range,
        CellIsRule(operator="equal", formula=[f'"{label}"'],
                   fill=PatternFill("solid", start_color=bg, end_color=bg),
                   font=Font(name=FONT_NAME, size=9.5, color=fg, bold=(label == "极强"))),
    )

# ---------------------------------------------------------------- Sheet5 机会分析
H_OPP = ["类别", "事项", "判断", "理由与建议"]
W_OPP = [20, 30, 16, 88]

ROWS_OPP = [
    ["已被填满的坑", "PagedAttention 式 KV 内存管理", "不要重复造", "2023 年已被解决，现为所有引擎标配；自研只会得到更慢更不稳定的版本"],
    ["已被填满的坑", "Continuous Batching / 调度基础", "不要重复造", "这是入场券而非差异点，vLLM 与 SGLang 已高度成熟"],
    ["已被填满的坑", "OpenAI 兼容 API 协议层", "不要重复造", "直接复用协议，用户不会为协议付费"],
    ["已被填满的坑", "基础量化（AWQ/GPTQ/FP8）", "不要重复造", "算子库已开源，自研无收益"],
    ["已被填满的坑", "单点投机解码算法", "谨慎", "算法迭代极快，除非有独家 draft 模型或训练侧协同，否则追不上"],
    ["真实空白", "Agent 原生调度", "高价值", "现有引擎以「请求」为调度单位，而 Agent 是含工具调用与分支的 DAG；跨请求状态复用、抢占粒度、失败重试语义均不匹配，这是最大的结构性错位"],
    ["真实空白", "PD 分离的放置（placement）算法", "高价值", "DistServe 的核心结论：放置算法比原始带宽更决定成败。这是调度与建模问题，不是 kernel 问题，门槛在仿真与预测"],
    ["真实空白", "异构与国产硬件协同", "高价值", "昇腾 / 寒武纪 / 摩尔线程的算子库与调度协同仍碎片化，vLLM 之外缺少一等公民支持"],
    ["真实空白", "编译式与解释式的融合", "中高价值", "TRT-LLM 的编译收益与模型迭代速度直接冲突；在线 JIT + 编译缓存复用的中间路线尚未成型"],
    ["真实空白", "以 goodput 为目标函数的调度器", "中高价值", "现有引擎优化 throughput 或 TTFT，缺少在 SLO 约束下显式最大化有效吞吐的目标函数"],
    ["真实空白", "端云协同 KV 迁移", "中价值", "端侧与云侧 KV 缓存跨设备迁移涉及隐私与成本，尚无标准方案"],
    ["真实空白", "大规模 EP 的弹性与容错", "中高价值", "1T 级 MoE 下专家并行的故障恢复代价极高，Mooncake EP / Elastic EP 才刚起步"],
    ["真实空白", "多元算力适配层", "高价值", "Chitu 赤兔证明了国产芯片适配本身就是一条产品线；昇腾 / 沐曦 / 海光 / 摩尔线程 / 燧原的算子与调度协同仍碎片化。若目标市场是国内私有化部署，这可能是比性能更硬的护城河"],
    ["真实空白", "极限内存与端侧 MoE", "中高价值", "kimi-k3-in-c（2.78T 参数跑在 8GB RAM）与 turbo-fieldfare（26B-A4B 跑在 2GB RAM）证明内存墙可被量化 + 稀疏激活 + 线性注意力绕开；端侧引擎的第一性原理是减少内存读取次数，而非提高 FLOPS"],
    ["真实空白", "存储层级延伸（SSD 当显存）", "中高价值", "inferssd / BigMoeOnEdge / llama-modchip 证明把 SSD 纳入内存层级，能让消费级硬件跑 744B 级 MoE；瓶颈从算力变成 SSD 随机读延迟与预取策略，这是典型的系统问题而非 kernel 问题"],
    ["策略建议", "组件复用清单", "建议", "kernel 用 FlashInfer / kt-kernel，投机解码用 EAGLE / speculators，KV 量化用 KVarN，结构化输出用 XGrammar，分布式编排用 Ray / llm-d。自研的应该只有调度器与状态管理——那是唯一能形成差异的地方"],
    ["真实空白", "RL-native 推理引擎", "高价值", "16 个开源 RL 库各自重复实现权重同步与部分 rollout 处理；vLLM 已补上 routed_experts 与（fork 中的）weight_versions，但 KV 状态导出与跨引擎一致性仍是缺口——引擎在补齐，只是补齐得不统一"],
    ["正在成型", "推理与训练的数据契约", "高价值（非空白）", "vLLM 已返回 routed_experts 与 SamplingMask，NeMo RL / 阿里 ROLL / Megatron 各有 R3 实现；缺口不在「有没有」，而在跨引擎格式不统一：vLLM 用 3D int16 数组、SGLang 走 meta_info、ROLL 按样本拼成 jagged"],
    ["正在成型", "跨引擎协议适配层", "高价值", "vLLM 系与 SGLang 系在 R3 上已经分叉（ROLL 明确不支持 vLLM rollout）；做一层统一的 routed_experts / sampling_mask 适配与一致性校验，工程量小、被采用的概率高"],
    ["策略建议", "协议实现路径", "建议", "不要从零定义协议：vLLM 的 routed_experts / SamplingMask 已是事实标准。可做的是适配层——统一 vLLM 的 3D uint8/uint16 数组与 SGLang 的 meta_info 格式，并正确处理三个边界（末 token 无路由、中止请求为 null、不可映射 token 用 -1 哨兵）"],
    ["策略建议", "R3 的启用判断", "建议", "R3 不是无脑开：吞吐约 -2%、显存约 14MB/GPU 都可接受，但传输体积按 (tokens × layers × top_k × 字节) 增长，32K token 约 60MB/条轨迹。dense 模型不需要；已用 TIS 屏蔽掉 off-policy 项的也不需要"],
    ["策略建议", "最小验证", "建议", "50 行代码即可验证契约：开 --enable-return-routed-experts 取一条 routed_experts，训练侧把同一批 token 的专家强制成该索引，对比 logprob 差异。归零则契约成立；不归零则已定位到第一个要修的点"],
    ["策略建议", "三类偏差分别修", "建议", "RL 训练不稳定有三个独立来源：路由偏差（R3）、动作空间偏差（sampling mask）、打分偏差（score centering）。vLLM 侧两个开关 + 训练侧一个期望校正，缺一不可；只修一个会误以为解决了"],
    ["真实空白", "算子栈的国产替代生态", "已被验证但未饱和", "DeepSeek 于 2026-09-30 开源昇腾六件套，证明这条路可行且能接近硬件极限；但 EP>32 仍在优化、生态才刚起步，工具链与调试体验仍是明显空白"],
    ["策略建议", "护城河判断", "建议", "引擎的护城河在 kernel 栈而非调度器：continuous batching 三天能写出来，DeepGEMM 的 99.8% 硬件利用率三年难复现。想做长期投入就下沉到 kernel，只想快速验证就留在调度层"],
    ["策略建议", "技术栈选型", "建议", "若要自研，Rust 优于 Python：vLLM 的公认短板正是 Python 调度开销与 GIL，而 mistral.rs 已证明 Rust 能到 llama.cpp 同级速度且多出 PagedAttention / 连续批处理 / ISQ。算子层站在 Candle 或 MLIR 上，不要手写 CUDA"],
    ["策略建议", "顺序建议", "建议", "先做 chunked prefill（单机、零基础设施、拿 PD 分离 30-50% 的收益），再评估是否上 PD 分离。反过来做是最常见的浪费"],
    ["策略建议", "起步路径", "建议", "不要从零写 kernel。先以 vLLM / SGLang 为后端，在编排层或调度层做差异化，用真实流量证明收益后再下沉到算子层"],
    ["策略建议", "验证方式", "建议", "用自有真实流量做 bake-off：统一量化格式、真实长度分布、真实前缀结构、看 goodput 而非峰值吞吐、剔除 warmup 请求"],
    ["策略建议", "风险提示", "建议", "能力差距在快速收敛（分页 KV、前缀缓存、投机解码已在互相借鉴），选型或立项应基于设计中心而非当前 benchmark"],
]

ws5, o_start, o_end = build_table_sheet(wb, "新引擎机会分析", H_OPP, ROWS_OPP, W_OPP, freeze="B2")

opp_range = f"C{o_start}:C{o_end}"
for label, bg, fg in [
    ("高价值", XL_G_DARK_BG, XL_G_DARK_FG),
    ("中高价值", XL_G_LIGHT_BG, XL_G_LIGHT_FG),
    ("中价值", XL_Y_BG, XL_Y_FG),
    ("不要重复造", XL_N_BG, XL_N_FG),
    ("谨慎", XL_Y_BG, XL_Y_FG),
    ("建议", xl_color("#D9E2F3"), xl_color("#2F5597")),
]:
    ws5.conditional_formatting.add(
        opp_range,
        CellIsRule(operator="equal", formula=[f'"{label}"'],
                   fill=PatternFill("solid", start_color=bg, end_color=bg),
                   font=Font(name=FONT_NAME, size=9.5, color=fg, bold=label in ("高价值", "建议"))),
    )

# ---------------------------------------------------------------- Sheet5.5 第二轮补充
H_ROUND2 = ["类别", "项目", "Stars", "核心机制", "为什么值得看", "对自研引擎的启示"]
W_ROUND2 = [17, 26, 10, 46, 56, 56]

ROWS_ROUND2 = [
    # 多元算力路线
    ["多元算力路线", "Chitu 赤兔\n(thu-pacman/chitu)", 3000,
     "面向国产芯片深度优化：昇腾 910B / 沐曦 / 海光 / 摩尔线程 / 燧原 + NVIDIA；CPU+GPU 异构混合推理，单卡可跑 DeepSeek-R1 671B；FP4 在线转 FP8/BF16；PD 分离；chitu.run 单文件部署",
     "目前「多元算力适配」做得最彻底的开源引擎之一，清华 PACMAN 出品，Apache-2.0；v0.3.9 首发支持昇腾 910B 跑 GLM-4.5 MoE，v0.5.1 适配摩尔线程",
     "硬件适配层本身就是一条独立产品线。若自研，算子抽象层的设计比调度器更决定生死"],
    ["多元算力路线", "xLLM\n(xLLM-AI/xllm)", 1600,
     "C++；面向多样 AI 加速器；同一引擎同时支持 LLM / VLM / DiT（扩散模型）",
     "开放原子基金会托管，国产中立路线；把 DiT 纳入同一引擎调度是少见设计",
     "多模态 + 生成式（图像/视频）统一调度，是主流引擎尚未覆盖的空白"],
    ["多元算力路线", "trnrt\n(RunanywhereAI)", 1600,
     "AWS Trainium 专用；NKI 手写 megakernel；支持 1M token 上下文；OpenAI 兼容 API",
     "证明非 NVIDIA 路线可以做到接近可用，不必依赖 CUDA 生态",
     "专用加速器的 megakernel 路线，可能比通用 kernel 库更适合固定形状负载"],
    ["多元算力路线", "vllm-ascend", 3100,
     "vLLM 的昇腾硬件插件，社区维护",
     "说明主流引擎的扩展方式是「硬件插件化」而非 fork 分支",
     "自研引擎若想被生态接纳，应提供插件接口，而不是要求用户换整条栈"],
    ["多元算力路线", "FastDeploy\n(PaddlePaddle)", 3700,
     "飞桨系 LLM/VLM 推理部署工具链，ERNIE 系列一等公民",
     "国产全栈闭环（训练框架 + 引擎 + 模型）的完整样本",
     "引擎与训练框架耦合是双刃剑：省适配成本，但锁死生态"],
    ["多元算力路线", "ZhiLight\n(知乎)", 906,
     "Llama 系高优化推理加速引擎，C++/CUDA",
     "互联网公司自研引擎的典型形态：只服务自家模型，做极致优化",
     "垂直场景专用引擎可以比通用引擎更快，但迁移成本极高"],
    # 极限内存路线
    ["极限内存路线", "kimi-k3-in-c", 11900,
     "2.78T 参数 Kimi K3 在单 CPU、8.24GB 内存上推理；纯 C99，无 BLAS / 框架 / GPU；MXFP4 量化 + 线性注意力",
     "极端工程示范：把「模型比可用内存大 300 倍」变成可运行",
     "量化 + 稀疏 + 线性注意力的组合能绕开内存墙，这是端侧引擎的核心课题"],
    ["极限内存路线", "turbo-fieldfare", 7300,
     "Gemma 4 26B-A4B 在约 2GB RAM 的 M 系列 Mac 上运行；Swift / Metal",
     "说明 MoE 的稀疏激活在端侧是巨大红利（26B 总参数，仅 4B 激活）",
     "MoE 端侧推理的瓶颈是专家加载而非算力，专家缓存策略比 kernel 优化更重要"],
    ["极限内存路线", "MTPLX", 2700,
     "Apple Silicon 原生 MTP 投机解码；M5 Max 上 125 tok/s；16GB Mac 可跑 27B；OpenAI/Anthropic 双兼容",
     "投机解码在端侧的收益远大于服务端，因为端侧是带宽受限而非算力受限",
     "端侧引擎的第一性原理是减少内存读取次数，不是提高 FLOPS"],
    ["极限内存路线", "Lucebox", 3000,
     "异构硬件与消费级 GPU；megakernel + speculative prefetch；支持 ROCm（R9700 / Strix Halo）",
     "消费级 GPU 路线的代表，绕开数据中心卡",
     "消费级硬件显存带宽是硬约束，prefetch 比增大 batch 更有效"],
    ["极限内存路线", "PowerInfer\n(Tiiny-AI)", 9900,
     "基于神经元稀疏激活：热/冷神经元分离，热权重放显存，冷权重留 CPU 内存",
     "稀疏激活的经典实现，影响了大量后续端侧工作",
     "利用激活稀疏性，可以把大部分权重放在 CPU 内存，显存只留热路径"],
    # KV 与内存虚拟化
    ["KV 与内存虚拟化", "kvcached\n(ovg-project)", 2000,
     "虚拟化弹性 KV Cache；动态 GPU 共享；在线/离线任务混部；支持 vLLM / Ollama / SGLang",
     "把 KV cache 当作可虚拟化的操作系统资源，而不是引擎内部实现细节",
     "这是 serverless 推理的关键拼图；「KV 即内存」的抽象可能重构引擎分层"],
    # 多租 LoRA
    ["多租 LoRA", "lorax\n(predibase)", 3800,
     "多 LoRA 推理服务，可扩展到数千个微调模型",
     "多租户微调服务的极致形态",
     "LoRA 调度（批内合并、适配器缓存淘汰）是独立于主模型调度的子问题"],
    # 服务编排
    ["服务编排", "Ray\n(ray-project)", 44300,
     "分布式计算框架，Ray Serve 承载推理服务",
     "事实标准的多机编排层，生态最广",
     "自研引擎不必自己解决分布式，交给 Ray / llm-d 更划算"],
    ["服务编排", "KServe", 6200,
     "K8s 上的标准化推理平台，支持 vLLM 等作为后端",
     "CNCF 生态的标准入口",
     "与 K8s 原语对齐，比自造部署格式更容易被采纳"],
    ["服务编排", "OpenLLM\n(BentoML)", 12600,
     "一条命令把开源模型变成 OpenAI 兼容端点",
     "开发者体验路线的代表",
     "体验层的价值可以独立于性能存在，用户愿为省心付费"],
    ["服务编排", "Parallax\n(GradientHQ)", 1400,
     "去中心化分布式服务，支持自建 AI 集群",
     "边缘 / 去中心化推理的探索方向",
     "去中心化场景下，调度目标从吞吐变成「可用性优先」"],
    # 学习路径
    ["学习路径", "zero-to-sglang\n(Datawhale × SGLang)", 2100,
     "官方课程：先理解推理原理，再从零实现 mini-sglang，最后读真实源码并提交第一个 PR；中英双语",
     "目前最系统的「从零写推理引擎」路径，有官方背书",
     "建议路径：跑通 mini-sglang → 对照源码改一个调度策略 → 向真实项目提交 PR"],
    ["学习路径", "KuiperInfer", 3500,
     "从零实现高性能深度学习推理库，支持 llama2 / UNet / YOLOv5 / ResNet",
     "国内经典教学项目，讲透算子、计算图、内存池",
     "适合补「算子与计算图」基础，但偏 CV，LLM 部分需配合其他材料"],
    ["学习路径", "ai-infra-book\n(李博杰)", 7500,
     "《深入理解 AI Infra：量化分析与系统设计》开源书稿，从硬件约束出发定量推导推理系统设计",
     "少见的「从硬件约束做定量推导」的中文材料，而非经验之谈",
     "造引擎前先建立 roofline / 带宽 / 显存的量化直觉，比读代码更重要"],
]

build_table_sheet(wb, "专项路线与组件全集", H_PART, ROWS_PART + ROWS_ROUND2, W_PART, numeric_cols=(3,), freeze="B2")

# ---------------------------------------------------------------- Sheet6 说明
ws6 = wb.create_sheet("说明")
ws6.sheet_properties.tabColor = XL_MUTED
ws6.column_dimensions["A"].width = 22
ws6.column_dimensions["B"].width = 110

ws6["A1"] = "口径与使用说明"
ws6["A1"].font = Font(name=FONT_NAME, size=12, bold=True, color=xl_color("#2F5597"))
ws6.merge_cells("A1:B1")
ws6["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws6.row_dimensions[1].height = 28

NOTES = [
    ("数据采集口径", "GitHub Stars / Forks / Open Issues / 最后提交时间 / 许可证状态，均通过 GitHub API 于 2026-11-13 采集，非估算值。"),
    ("性能数据口径", "百分比与倍数来自多方公开 benchmark（Spheron、LeetLLM、DistServe、Mooncake 技术报告等），仅代表量级区间。不同硬件、量化格式、序列长度分布下结论可能反转，甚至完全相反。"),
    ("特别提示", "HuggingFace TGI 仓库已归档（archived，最后提交 2026-03-21），不再演进；新项目不应选用。这一事实说明本领域格局仍在变动。"),
    ("选型纪律", "必须用自有真实流量做 bake-off：统一量化格式、使用真实长度分布、保留真实前缀结构、以 goodput 而非峰值吞吐为目标、剔除 warmup 请求。跨硬件或跨负载的 benchmark 不可直接采信。"),
    ("常见误判", "把「吞吐最高」等同于「最适合」是最常见的错误。峰值吞吐的工作点往往已有大量请求超时；SLO 约束下的有效吞吐（goodput）才决定成本。"),
    ("重要边界", "DeepEP-Ascend 的带宽数据来自 DeepSeek 提供的 PoC HDK（非公开发售版本），其 README 明确提示早期 PoC 硬件可能更低；官方建议以华为 2026-10-15 前后发布的商用品为基线。引用这些数字时必须注明该边界。"),
    ("适用范围", "本工作簿用于技术选型与新引擎立项参考，不构成性能承诺。推理引擎迭代极快，建议每季度复核一次。"),
]
r = 3
for label, content in NOTES:
    c1 = ws6.cell(row=r, column=1, value=label)
    c1.font = Font(name=FONT_NAME, size=10, bold=True, color=XL_TEXT)
    c1.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    c1.fill = PatternFill("solid", start_color=xl_color("#D9E2F3"), end_color=xl_color("#D9E2F3"))
    c1.border = BORDER_ALL

    c2 = ws6.cell(row=r, column=2, value=content)
    c2.font = Font(name=FONT_NAME, size=10, color=XL_TEXT)
    c2.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    c2.border = BORDER_ALL

    ws6.row_dimensions[r].height = est_row_height([content], [110], base=15)
    r += 1

ws6.sheet_view.showGridLines = False

wb.properties.title = "主流大模型推理引擎对比调研（2026-11）"
wb.properties.creator = "WorkBuddy"

OUT = r"C:/Users/lenovo/WorkBuddy/2026-11-13-16-48-36/主流大模型推理引擎对比调研_2026-11.xlsx"
import os, time
base, ext = os.path.splitext(OUT)
stamp = time.strftime("%Y%m%d_%H%M%S")
saved = None
for candidate in [OUT, f"{base}_v3{ext}", f"{base}_v3_{stamp}{ext}"]:
    try:
        wb.save(candidate)
        saved = candidate
        break
    except PermissionError:
        continue
if saved is None:
    raise RuntimeError("所有候选路径均被占用，无法保存")
OUT = saved
print("saved:", OUT)
print("sheets:", wb.sheetnames)
