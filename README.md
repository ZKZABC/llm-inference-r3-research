# 大模型推理引擎调研 & R3 路由契约适配层

> 一次从"想自研推理引擎"出发、最终收敛到"先做 RL 数据契约适配层"的完整调研 + 工程实践记录。

## 一句话结论

**通用推理引擎赛道已经没有空白，真正的缝隙在"RL 训练侧与推理侧的数据契约"上。**
与其造第 N 个引擎，不如做那个让所有引擎都不踩坑的**可复用校验库**。

## 这个仓库里有什么

```
.
├── r3-adapter/                     # R3 路由契约适配层（核心）
│   ├── env_check.py                # ★ 第 1 类：环境自检（跑之前先查这台机器行不行）
│   ├── r3_contract.py              # ★ 第 2 类：可复用校验库（5 个校验原语）
│   ├── silent_degradation.py       # ★ 第 3 类：静默降级检测（日志说做了但没生效）
│   ├── determinism/                # ★ 第 4 类：可复现性门禁（跨 batch / 跨路径）
│   │   ├── gemm_gate.py            #   gemv/gemm 路径分叉
│   │   ├── attention_gate.py       #   后端 / mask / KV 长度
│   │   ├── moe_gate.py             #   combine 顺序 / capacity 溢出
│   │   └── skeleton_moe.py         #   MoE 实现与 HF 对齐的参考实现
│   ├── test_r3_contract.py         # unittest 测试套件（11 项）
│   ├── contract_check.py           # 纯逻辑契约测试（无需 GPU）
│   ├── mini_moe_r3.py              # 微型 MoE 在 CPU 上验证 R3 数学命题
│   ├── r3_verify.py                # 真实 vLLM 环境验证脚本（需 GPU）
│   ├── FIRST_REAL_DATA.md          # ★ 第一条真实 routed_experts 实测记录
│   └── SETUP_WSL2.md               # WSL2 环境搭建 + 真实踩坑记录
├── tools/
│   └── cloud_probe.py              # routed_experts 分布离线分析
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

## 第一条真实数据（2026-11-14 实测）

在 ModelScope 的 NVIDIA GPU 实例（**24 GB 显存 / CUDA 13.0**）上跑通：

```python
shape  = (8, 24, 4)     # 8 行 = 1 prompt + 8 gen − 1（末 token 无路由）
dtype  = uint8
range  = 0 ~ 59         # 60 个专家，无越界、无 -1
unique = 60             # 全部 60 个专家都被激活
token0-layer0 = [33, 52, 7, 54]
```

**契约四条全部成立。** 详细记录见 [`r3-adapter/FIRST_REAL_DATA.md`](r3-adapter/FIRST_REAL_DATA.md)。

#### 一个必须修正的结论：路由不是"无局部性"

初版根据 `unique = 60` 断定"路由无局部性"，**这个推论是错的**——
`unique` 是基数，对分布形状零分辨力（60 专家 × 768 次抽样，纯随机也有 99.98% 概率得到 60）。

换正确指标重测（8 token / 768 槽位）：

| 指标 | 实测 | 随机基线 | 结论 |
|---|---|---|---|
| counts 标准差 | **5.29** | 3.58（泊松） | 过散 2.19 倍，**有偏好** |
| top-10 占比 | **27.3%** | 16.7% | 中度偏斜，非长尾 |
| 相邻 token Jaccard | **0.1395** | 0.034 | **4.1 倍随机，存在时序局部性** |

**修正后的判断**：边际分布接近均匀，但**时间维度有弱局部性**（Jaccard 2.7 倍随机）。

#### 长序列复测推翻了这个"弱局部性"的乐观解读

438 token、42,048 槽位复测后：

| 维度 | Jaccard | vs 随机 | 结论 |
|---|---|---|---|
| 时间（相邻 token）| 0.0917 | 2.7× | 但**中位数为 0**，逐 token 点状抖动，无稳定段 |
| 层间（相邻层）| 0.0388 | **1.14×** | **纯随机** |

**专家访问在任何可预取维度上都接近随机。** 专家级缓存的收益上限只有 10~15%，撑不起复杂度。

#### 再复测（818 token / 78,528 槽位）：连"过散"也是小样本噪声

| 指标 | 8 token | 818 token |
|---|---|---|
| counts 标准差 | 5.29（过散 2.19×） | **cv 0.052** |
| top-10 占比 | 27.3% | **17.86%**（均匀 16.7%） |
| 最忙 / 均值 | — | **1.08×** |

8 token 时那个「过散 2.19 倍」是**泊松噪声**。样本量上来后，边际分布几乎完美均匀——
load balancing 的效果比小样本看起来好得多。

时间局部性这次有了完整定性：

| 指标 | 值 | 含义 |
|---|---|---|
| 均值重叠 | 0.1515（2.27× 随机） | 看起来有 |
| **中位数** | **0.0** | 半数以上相邻对**完全没有**共享专家 |
| **零重叠占比** | **56.9%** | 逐 token 点状抖动，无稳定段 |

均值被少数连续段拉高，中位数是 0——与 438 token 时的判断一致，预取收益有限。
（对照：纯随机路由的零重叠占比是 76.9%，56.9% 说明确实有「成段」现象，只是不普遍。）

但换一个维度，有个干净信号：

| 对比 | 相邻 token 重叠 | 相对 |
|---|---|---|
| 同 prompt 内 | 0.1527 | **2.64×** |
| 跨 prompt | 0.0578 | 基准 |

**专家选择由 context 决定。** 微观证据：`batch=8` 时某个专家被 8 个 token **全部选中**
（32 个选择里占 8 个，均匀期望只有 0.53）——因为那 8 个 token 来自同一条 prompt。

但要克制解读：0.15 的绝对重叠率意味着 4 个专家里平均只重叠 0.6 个，
**能否转化成调度收益需要单独验证，不能直接说「可以做亲和调度」**。

所以结论要分层：

- **边际分布**：均匀 → 冷热分层不可行
- **时间维度**：点状抖动 → 预取收益有限
- **语义维度**：2.64× 的相对优势 → 有信号，但绝对强度不足以支撑缓存

capacity 的实测建议值（M=8..512）见 [`r3-adapter/determinism/README.md`](r3-adapter/determinism/README.md)。

#### 对照：KV 层有 99% 命中

| 请求 | queries | hits | 命中率 | 耗时 |
|---|---|---|---|---|
| 冷 | 840 | 0 | 0% | 0.091 s |
| 热（同 prompt）| 840 | 832 | **99.05%** | **0.030 s** |

`832 = 52 × 16` —— block_size 对齐，最后不足一个块的 8 个 token 不缓存。

**结论：可复用性来自「输入结构」（context），不是「计算结构」（路由决策）。**
引擎该优化的是「复用算过的 KV」，不是「复用选过的专家」——后者不存在。

### 但 R3 有一个隐藏前提：批不变性

同一条 prompt 在 `batch=1` 和 `batch=2` 下的路由对比：

| 配置 | 结果 |
|---|---|
| 默认 | `diff: 3 / 768` ❌ 路由翻转 |
| `VLLM_BATCH_INVARIANT=1` | `diff: 0 / 768` ✅ 完全一致 |

**结论：R3 可行，但必须显式开启 `VLLM_BATCH_INVARIANT=1`。**

这条不在官方 `routed_experts.md` 的 Requirements 里——少了它，
shape/dtype/合法性全部通过，但数据在真实 RL 场景下不可复现。

代价：`torch.compile` 让引擎初始化从 10 秒变成 **130 秒**（一次性，有缓存）。

### 跑通 R3 的三个必备条件

```bash
VLLM_BATCH_INVARIANT=1 vllm serve <model> \
  --enable-return-routed-experts \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9
```

```json
{"temperature": 0, "repetition_penalty": 1.0}
```

| # | 条件 | 不满足会怎样 |
|---|---|---|
| ① | `--enable-return-routed-experts` | `routed_experts` 为 `None` |
| ② | `VLLM_BATCH_INVARIANT=1` | batch 组成一变，路由就翻转 |
| ③ | `repetition_penalty=1.0` | 模型自带的 1.05 改变 logits，间接影响专家选择 |

## 快速开始

### 0. 先自检（强烈建议，30 秒）

```bash
cd r3-adapter
python env_check.py --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
```

**这一步能省下 2 小时。** `env_check.py` 的每一条检查都对应下面「踩坑记录」里的一个真实坑：
vLLM 版本、CUDA 架构、空闲显存、`libnvrtc-builtins`、`Python.h`、pinned memory、
**模型显存预算** 和 **host 内存预算**。

两条预算的计算方式是这套脚本的核心：

```
显存预算：  模型 + KV cache + 激活    vs   空闲显存 × gpu_memory_utilization
host 预算： 需要 offload 的量         vs   总内存 − 模型 × 0.9 − 0.5
                                            ↑ 用 0.9 是因为实测 safetensors 的 mmap
                                              会吃掉约 90% 模型大小的物理内存
```

实测校验（三次真实运行的数字）：

| 场景 | 需 offload | host 加载后剩 | 判定 | 日志实测 |
|---|---|---|---|---|
| WSL 7.4 GiB | 2.61 GiB | 0.00 GiB | FAIL 缺口 2.61 | `Available RAM: 0.62 GiB` |
| WSL 9.7 GiB | 2.61 GiB | 2.14 GiB | FAIL 缺口 0.46 | `Available RAM: 2.60 GiB` |
| 4090 24GB + 64G RAM | — | — | 显存够，无需 offload | — |

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

### 4. 静默降级检测（拿到启动日志就能跑，30 秒）

```bash
python silent_degradation.py --log vllm_startup.log --model-gib 7.84
# 或
docker logs vllm 2>&1 | python silent_degradation.py --stdin
```

10 条规则，专抓「日志说做了但没生效」：

```
[CRITICAL] QUANT_EMULATION        量化后端回退，显存涨 3.5 倍
[CRITICAL] OFFLOAD_NOOP           offload 只记账不搬运
[CRITICAL] BATCH_INVARIANT_MISSING 开关未生效，R3 数据不可复现
[WARN]     SDPA_BACKEND_FALLBACK   注意力后端静默回退
[WARN]     SAMPLING_OVERRIDE       采样参数被 generation_config 覆盖
```

`--model-gib` 是可选的：不给就只能提示「需要核对」，给了才能精确判定 offload 是否真的搬运了。

### 5. 可复现性门禁（第 4 类校验）

```bash
cd determinism
python gemm_gate.py --both --dtype bf16        # GEMM 路径分叉
python attention_gate.py --dtype bf16          # 后端 / mask / KV 长度
python moe_gate.py --dtype bf16                # combine 顺序 / capacity
python skeleton_moe.py                         # MoE 与 HF 对齐（零下载）
```

前三类全绿不代表数据可用。**这一层专门抓「格式全对但换 batch 就变」的情况。**

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
| 6 | `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | vLLM 选了 FlashInfer 做采样，它要 nvcc 现场编译 | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 7 | `Pinned memory is not available on this platform` | **WSL2 其实支持 pinned memory（实测可分配 3GB），是 vLLM 平台判断误判**；`is_uva_available()` 返回 False → 用 `NonUvaBuffer` | 改 `vllm/utils/platform_utils.py:57` → `return True`。**必须改源码：spawn 子进程不继承 monkey patch** |
| 8 | offload 报告搬运 3.01 GB，但 `Model loading took` 仍是 7.85 GiB | **vLLM 0.33 的 UVA offloader 对 GPTQ MoE 只记账不搬运**：多复制一份到 CPU（加载变慢），不释放 GPU 原件 | 无解。`--max-num-batched-tokens 512` 只省激活（-0.46→-0.11 GiB），补不上 7.85 GiB 模型本体 |

### 第 8 个坑的最终结论

```
模型实占:        7.85 GiB（offload 前后一模一样）
GPU 总显存:      7.96 GiB
Windows 占用:    1.11 GiB
实际可用:        6.85 GiB
缺口:            ≈ 1 GiB
```

**UVA 与 prefetch 两种 offload 后端都无效**：UVA 因为误判走不到 pinned 路径（改源码后走通了但仍不搬运），prefetch 只省 0.16 GB（接管 6 个小模块，没碰专家权重）。

> **结论：8GB 消费级卡 + WSL2 + vLLM 0.33 跑 14B MoE，在当前软件栈下是硬件天花板。**
> 这不是"参数没调对"，而是 vLLM 的 CPU offload 在「WSL + Blackwell sm_120 + GPTQ 量化 MoE」这个四重叠加场景下存在实现缺陷——这种组合不在它的测试矩阵里。

## 设计哲学：四类校验，层层递进

最初以为适配层的核心是校验 `routed_experts` 的 shape/dtype/合法性。
但真实环境跑下来发现：**真正让人卡住 2 小时的，从来不是数据格式，而是环境前提。**

而格式校验还有一个更深的盲区：**它对"数据不可复现"完全无感**。
`3/768` 那种翻转，shape 对、dtype 对、ID 全合法。

所以这个库真正该做的四件事：

| # | 校验类 | 文件 | 回答什么问题 | 典型盲区 |
|---|---|---|---|---|
| 1 | 环境 | `env_check.py` | 这台机器跑得起来吗 | — |
| 2 | 格式 | `r3_contract.py` | 数据结构符合契约吗 | 格式对但数据不可复现 |
| 3 | 静默降级 | `silent_degradation.py` | 日志说做了，真做了吗 | 只记账不搬运、开关未生效 |
| 4 | 可复现性 | `determinism/` | 换个 batch 还成立吗 | — |

**第 4 类是最容易被跳过、也最致命的一类。** 前三类全绿，数据依然可能不可用。

### 第 4 类的实测结论

三层门禁在 RTX 5060 Laptop 8GB / bf16 下的结果：

| 层 | 裂缝 | 影响面 | 约束 |
|---|---|---|---|
| GEMM | `M=1` vs `M≥2`（gemv / gemm） | 0.1% | decode `M ≥ 2` |
| Attention | 四个 SDPA 后端互不一致 | 23% | 启动时锁定一个 |
| Attention | `is_causal` vs 显式 mask | 6.6% | 统一写法 |
| Attention | KV 总长度 `S` 变化 | 23% | 全局固定 `S` |
| MoE | combine 顺序（slot vs expert） | 17% | 按 slot 累加 |
| MoE | capacity 溢出 | 静默丢 token | `C` ≥ 2.5× 期望 |

**根因只有一个：同一段数学存在多套代码路径，路径之间的浮点归约顺序不同。**

所以不变量不是「固定 shape」，是**「固定代码路径」**——
把运行时 dispatch 变成启动时配置。

细节见 [`r3-adapter/determinism/README.md`](r3-adapter/determinism/README.md)。

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
