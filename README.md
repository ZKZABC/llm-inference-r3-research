# 大模型推理引擎调研 & R3 路由契约适配层

> 一次从"想自研推理引擎"出发、最终收敛到"先做 RL 数据契约适配层"的完整调研 + 工程实践记录。

## 一句话结论

**通用推理引擎赛道已经没有空白，真正的缝隙在"RL 训练侧与推理侧的数据契约"上。**
与其造第 N 个引擎，不如做那个让所有引擎都不踩坑的**可复用校验库**。

## 这个仓库里有什么

```
.
├── r3-adapter/                     # R3 路由契约适配层（核心）
│   ├── env_check.py                # ★ 环境自检（跑之前先查这台机器行不行）
│   ├── r3_contract.py              # 可复用校验库（5 个校验原语）
│   ├── test_r3_contract.py         # unittest 测试套件（11 项）
│   ├── contract_check.py           # 纯逻辑契约测试（无需 GPU）
│   ├── mini_moe_r3.py              # 微型 MoE 在 CPU 上验证 R3 数学命题
│   ├── r3_verify.py                # 真实 vLLM 环境验证脚本（需 GPU）
│   └── SETUP_WSL2.md               # WSL2 环境搭建 + 真实踩坑记录
├── output/                         # 调研报告（Word / HTML / Markdown）
│   └── rpt_20261113_172650/
├── 主流大模型推理引擎对比调研_v9.xlsx   # 九轮调研数据（10 表 167 行）
└── .主流大模型推理引擎对比调研_2026-11.ref/  # 建表脚本
```

## R3 是什么

**R3 = Routed Experts**。在 MoE（混合专家）模型里，每个 token 只会激活少数几个专家。
推理引擎（如 vLLM）可以返回「哪个 token 选了哪些专家」——这就是 `routed_experts`。

**契约命题**：推理侧选中的专家，能否被训练侧原样回放，使 logprob 差异归零？

如果成立 → RL 训练可以精确复现推理行为，消除「训练-推理不一致」。
如果不成立 → 说明存在隐藏的状态/精度/调度差异，这本身就是要修的 bug。

### 三个独立的偏差源

| 偏差 | 来源 | 应对 |
|---|---|---|
| 路由偏差 | 推理与训练的专家选择不同 | R3 记录 + Router Replay |
| 动作空间偏差 | 采样 mask 不一致 | sampling mask 对齐 |
| 打分偏差 | score centering 不同 | 归一化对齐 |

## 快速开始

### 0. 先自检（强烈建议，30 秒）

```bash
cd r3-adapter
python env_check.py --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
```

**这一步能省下 2 小时。** `env_check.py` 的每一条检查都对应下面「踩坑记录」里的一个真实坑：
vLLM 版本、CUDA 架构、空闲显存、`libnvrtc-builtins`、`Python.h`、pinned memory、模型显存预算。

输出示例（真实结果）：

```
[GPU 硬件]
  [ OK ] GPU          NVIDIA GeForce RTX 5060 Laptop GPU
  [ OK ] 总显存        7.96 GiB
  [ OK ] 计算能力      sm_120

[内存]
  [FAIL] pinned memory  不可用：RuntimeError
                        修复：WSL2 的已知限制。UVA offload 会静默失效

[模型适配]
  [FAIL] 显存预算        需要 8.34 GiB vs 可用 6.31 GiB，缺口 2.03 GiB

结论：这台机器跑不了 R3 验证
```

### 1. 纯逻辑验证（无需 GPU，10 秒）

```bash
cd r3-adapter
python contract_check.py        # 6 项契约检查
python -m unittest test_r3_contract -v   # 11 项单测
```

### 2. CPU 数学验证（无需 GPU，1 分钟）

```bash
python mini_moe_r3.py
```

在一个 4 专家 top-2 的微型 MoE 上验证：**回放路由后 logprob 差异精确归零（0.000e+00）**。
同时演示一个反例：错误地从 top-k 清零后的 probs 取权重，会产生 **15.84% 的权重偏差**。

### 3. 真实环境验证（需 GPU + MoE 模型）

见 `r3-adapter/SETUP_WSL2.md`。核心命令：

```bash
python r3_verify.py \
  --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

## 真实环境踩坑记录（最有价值的部分）

在 **Windows + WSL2 + RTX 5060 8GB + vLLM 0.33.0 + Qwen1.5-MoE-A2.7B-GPTQ-Int4** 上实测，
连续撞上 6 道环境门槛。**没有一个是"用户操作错误"**，全是框架假设了理想环境：

| # | 坑 | 根因 | 解法 |
|---|---|---|---|
| 1 | `pydantic ValidationError: cpu_offload_params Input should be a valid set` | Python API 要 `set[str]`，CLI 接受 `str` | 传 `{'experts'}` 而非 `'experts'` |
| 2 | `Free memory ... less than desired GPU memory utilization` | `gpu_memory_utilization` 按**空闲显存**校验，不是总显存 | 降到 0.85，关掉吃显存的程序 |
| 3 | `NVRTCCompiler failed` | Blackwell sm_120 + CUDA graph 编译失败 | `--enforce-eager` |
| 4 | `failed to open libnvrtc-builtins.so.13.0` | 库在 venv 的 `site-packages/nvidia/cu13/lib/`，但 NVRTC 是 C 层加载器，不走 Python 路径 | 设 `LD_LIBRARY_PATH` |
| 5 | `driver.c: fatal error: Python.h` | Triton 要 JIT 编译 C 扩展，缺 Python 开发头文件 | `apt install python3-dev` |
| 6 | `Offloader set to UVAOffloader` + `KV cache memory: -0.46 GiB` | **WSL2 不支持 pinned memory**，UVA offload 静默回退到 device memory，等于没 offload | 换 `prefetch` 后端——但实测只省 0.16 GB，仍不够 |

### 第 6 个坑的最终结论

```
模型实占:        7.69 ~ 7.85 GiB
GPU 总显存:      7.96 GiB
Windows 占用:    1.11 GiB
实际可用:        6.85 GiB
缺口:            ≈ 1 GiB
```

**UVA 与 prefetch 两种 offload 后端都补不上这 1 GiB**，因为 GPTQ 量化的 MoE 专家权重在 vLLM 里是融合 tensor，offloader 无法按层/按参数拆分搬运。

> **结论：8GB 消费级卡 + WSL2 + vLLM 跑 14B MoE，在当前软件栈下是硬件天花板。**
> 这本身就是一个有价值的结论——它说明"最小验证方案"必须在文档里明确标注硬件前提。

## 设计哲学：校验的应该是环境，不只是数据

最初以为适配层的核心是校验 `routed_experts` 的 shape/dtype/合法性。
但真实环境跑下来发现：**真正让人卡住 2 小时的，从来不是数据格式，而是环境前提。**

所以这个库真正该做的三件事：

1. **启动前检查**：CUDA 版本 / 架构 / 空闲显存 / host 内存 / 必要动态库，不满足就明确报错
2. **契约校验**：shape、dtype（uint8 vs uint16）、`-1` 哨兵、越界 ID、末 token 无路由
3. **静默降级检测**：`falling back to device memory`、`offloaded 0.16GB` 这类"不报错但失效"的情况

## 调研数据

`主流大模型推理引擎对比调研_v9.xlsx` 含 10 张表、167 行数据，覆盖九轮调研：

- 引擎总览（vLLM / SGLang / TensorRT-LLM / llama.cpp …）
- 编排层（Dynamo / llm-d）
- KV 层（Mooncake / LMCache / NIXL）
- 零件层（kernel / 量化 / 投机解码）
- 语言栈与加速器（Rust / NPU / 编译器）
- DeepSeek 昇腾六件套
- RL 与推理的数据契约（R3、sampling mask、score centering）
- 协议实现现状与机会分析

## License

Apache-2.0
