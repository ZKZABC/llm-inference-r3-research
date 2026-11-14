# R3 路由契约真实环境验证报告

> **副标题**：从 8 个环境坑到批不变性 —— 一次从消费级 8GB 显卡到云端 GPU 的完整验证记录
>
> **日期**：2026-11-14
> **验证对象**：vLLM `--enable-return-routed-experts`（R3 / Routed Experts Return）
> **一句话结论**：**契约在真实环境成立，但它有一个文档里没写的隐藏前提——批不变性必须显式开启。**

---

## 摘要

R3（Routed Experts Return）是 vLLM 为 RL 训练提供的"路由回放"能力：把推理时 MoE 每层选中的专家 ID 返回给调用方，让训练侧能原样重放这次前向。

本次验证回答三个问题：

| 问题 | 答案 | 证据 |
|---|---|---|
| 拿到的数据格式对吗？ | ✅ 对 | `shape (8,24,4)`，`uint8`，`range 0~59` |
| 同一个请求重复调用一致吗？ | ✅ 一致 | `diff: 0 / 768` |
| **换个 batch 组成还一致吗？** | ⚠️ **默认不一致，需显式开启** | 默认 `3/768`，开启后 `0/768` |

**第三个问题是本次最重要的发现。** 它不在官方文档的 Requirements 列表里，但少了它，数据在真实 RL rollout 场景下**不可复现**。

---

## 1. 背景：为什么要验证 R3

RL 训练（尤其是 MoE 模型的 RLHF/GRPO）存在一个隐蔽的不一致问题：

```
推理（rollout）时：MoE router 选中专家 A、B、C、D
训练（update）时：  MoE router 可能选中 A、B、C、E
```

如果训练侧无法复现推理侧的路由，那么：

- 训练用的 forward 和产生 reward 的 forward 是**两个不同的函数**
- 梯度方向与真实策略偏离
- 表现为训练不稳定、reward 震荡

R3 的解法是：**推理时把路由记下来，训练时强制回放**。

本次验证的目标不是"跑通一个 demo"，而是**找出这条路径上有多少隐藏前提**。

---

## 2. 验证环境

### 2.1 最终成功环境

| 项目 | 值 |
|---|---|
| 平台 | ModelScope Notebook（魔搭免费 GPU 实例，36 小时额度） |
| 硬件 | NVIDIA GPU，24 GB 显存，8 核 / 32 GB 内存 |
| 计算栈 | CUDA 13.0，torch 2.14.0+cu130 |
| 推理引擎 | **vLLM 0.33.0**（PyPI 官方 wheel） |
| 模型 | `qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`（7.84 GiB，60 专家 top-4，24 层） |
| 启动方式 | `vllm serve`（HTTP API） |
| 文件系统 | NFS（`/mnt/workspace`，跨实例持久） |

### 2.2 失败的起点：本地 8GB 显卡

最初尝试在 **Windows + WSL2 + RTX 5060 Laptop 8GB** 上运行，连续撞上 8 个环境门槛，
最终确认是硬件天花板（详见 §5.1）。这段失败经历本身就是本次报告的重要素材。

---

## 3. 实验结果

### 3.1 契约几何 —— 格式全部正确

```python
shape  = (8, 24, 4)
dtype  = uint8
range  = 0 ~ 59
unique = 60      # ⚠️ 这是基数，不代表分布形状，详见 §4.1
token0-layer0 = [33, 52, 7, 54]
```

npy 文件头（base64 解码后）：

```python
{'descr': '|u1', 'fortran_order': False, 'shape': (8, 24, 4)}
```

逐项对照契约定义：

| 契约条款 | 实测 | 结论 |
|---|---|---|
| `shape = [num_tokens - 1, num_moe_layers, top_k]` | `(8, 24, 4)`，8 = 1(prompt) + 8(gen) − 1 | ✅ 末 token 无路由成立 |
| dtype：≤256 专家用 uint8 | `\|u1` = uint8，60 专家 | ✅ |
| 专家 ID ∈ [0, num_experts) 或 −1 | `0~59`，无 −1 | ✅ 无 unmappable |
| 行序 = `prompt_token_ids + output_token_ids` 的位置 | 逐行对齐 | ✅ |

### 3.2 确定性 —— 重复调用完全一致

同一请求（`prompt="Hi"`, `max_tokens=8`, `temperature=0`）连发两次：

```
identical: True
diff: 0 / 768
```

**结论**：推理侧的 top-k 选择在相同输入下是确定性的。

### 3.3 批不变性 —— 本次核心发现

#### 实验设计

对比同一 prompt 在两种 batch 组成下的路由：

- **A**：单独跑 `["Hi"]`
- **B**：`["Hi"]` 与 `["Hello there"]` 同批（batch=2），取第一条

#### 结果

| 配置 | identical | diff |
|---|---|---|
| **默认**（未开 batch invariant） | False | **3 / 768** |
| **`VLLM_BATCH_INVARIANT=1`** | True | **0 / 768** |

**768 个位置里 3 个翻转，翻转率 0.39%。**

从"稳定性"看，99.6% 一致似乎可以接受。但从 R3 的角度看**这是致命的**——
因为 RL rollout 的 batch 是动态拼的，同一条 prompt 可能单独跑、也可能和另外 7 条一起跑。
一旦 batch 组成变化，记录的路由就失效了。

#### 为什么默认会翻转

MoE 的 top-4 是比较 logits 大小选出来的。当第 4 名和第 5 名的 logits 相差极小时
（例如 `2.3412` vs `2.3409`），batch 变化导致底层 kernel 的并行度、浮点归约顺序改变，
这个微小的数值差异就足以让 top-k 翻转。

这是**浮点归约顺序的不确定性**，不是 bug，是所有 GPU 推理引擎的共性。

#### 硬证据：开关真的生效了

开启 `VLLM_BATCH_INVARIANT=1` 后，启动日志出现：

```
Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'batch_invariant.py:293'
This call originates from:
  File "vllm/model_executor/determinism/batch_invariant.py", line 293, in matmul_persistent
    NUM_SMS = num_compute_units(a.device.index)
```

vLLM 确实加载了 `determinism/batch_invariant.py` 并启用 `matmul_persistent`
（确定性矩阵乘实现）。这说明该开关**覆盖到了 MoE 的路由计算路径**，不只是 attention。

#### 性能代价

| 阶段 | 默认 | 开 batch invariant |
|---|---|---|
| 模型加载 | 10.4 s | 10.4 s |
| `torch.compile` | 不触发 | **32.1 s** |
| **引擎初始化总计** | **~10 s** | **130.1 s** |

一次性成本（编译结果有缓存），不影响单次推理延迟。但频繁重启调参时会累加。

---

## 4. 六个关键发现

### 4.1 `unique = 60` 是零信息量 —— 路由不是"无局部性"，是"弱到中度局部性"

初版报告根据 `unique = 60` 断定"路由无局部性"，**这个推论是错的**，此处修正。

#### 为什么 `unique` 证明不了局部性

`unique` 是**基数（cardinality）**，局部性是**结构（structure）**。基数对结构不敏感。

纯随机假设下的计算：60 个专家、768 次独立抽样，

```
P(某专家 0 次) = (59/60)^768 = e^(-12.907) ≈ 2.47e-6
E[未覆盖数]    = 60 × 2.47e-6 ≈ 1.5e-4
```

**即路由完全随机，也有 99.98% 概率测出 `unique=60`。** 这个观测的似然比接近 1。

更根本的：`unique` 对不同偏斜度完全不敏感。

| top-10 占比 | 期望 unique |
|---|---|
| 0%（完全均匀） | 60 |
| 50% | ≈ 60 |
| 70% | ≈ 59.5 |
| 90%（极端塌缩） | ≈ 49 |

**它只能排除 90% 级的极端塌缩，对 0~70% 的偏斜零分辨力。**

#### 换三个正确指标重测（8 token / 768 槽位）

| 指标 | 实测 | 随机基线 | 结论 |
|---|---|---|---|
| counts 标准差 | **5.29** | 3.58（泊松 √12.8） | 方差是均值的 **2.19 倍**，显著过散 |
| top-10 占比 | **27.3%** | 16.7%（均匀） | 中度偏斜，**非长尾塌缩** |
| 相邻 token Jaccard | **0.1395** | 0.034 | **4.1 倍于随机，存在时序局部性** |

top-15 计数：`[29, 26, 24, 21, 20, 18, 18, 18, 18, 18, 17, 17, 16, 16, 16]`
—— 一条平滑的鼓包，没有长尾。

换算成绝对值：相邻 token 平均约 **1 个专家重叠**（4 个中），随机期望仅 **0.27 个**。

#### 修正后的结论

| 初版断言 | 修正 |
|---|---|
| 路由无局部性 | **边际接近均匀，但时间维度有显著局部性** |
| 缓存分层不可行 | **硬稀疏假设不行，软分层 / 预取仍值得** |
| CPU offload 收益有限（因访问均匀） | **冷启动确实省不下来，稳态下约 25% 命中率** |

**冷启动 vs 稳态必须分开看**：

- 冷启动：60 个专家全部必须可访问，这是硬约束
- 稳态：相邻 token 有约 1/4 的专家复用，预取/缓存有明确收益

#### 这次修正的教训

一个"看起来很有说服力"的数字（`unique=60`，正好等于专家总数），
可能对真正要回答的问题**零信息量**。

判断一个指标是否有效，要问：**「如果假设错了，这个指标会变吗？」**
纯随机路由和高度偏斜路由都会给出 `unique=60`——所以它不能用来区分两者。

#### 局限

仅 **8 token、单 prompt、768 槽位**，且相邻 token 的 hidden state 高度相关，
**768 次抽样不独立**，有效样本量远小于 768。

- `jaccard 4.1x` 倍数很大，**趋势可信**
- `jaccard = 0.1395` 的**绝对值**可能只是这个 prompt 的特性
- 待 **500 token 长序列**（47,904 槽位）复测确认

### 4.2 批不变性是隐藏前提，不在官方 Requirements 里

官方文档 `docs/training/routed_experts.md` 的 Requirements 只写了：

> - Model Runner V2 被使用
> - runner 是 generate 且模型是 MoE
> - prefix caching 开启

**没有提 batch invariant。** 但实测证明，少了它：

```
契约的 shape/dtype/合法性 全部通过
但数据在真实 RL 场景下不可复现
```

这是最难查的一类 bug：**不报错、不崩溃、数据看起来完全正常，但训练和推理悄悄不一致。**

### 4.3 文档说"必须 V2 runner"，实测 V1 也能工作

文档写：

> Startup fails with a `ValueError` **unless Model Runner V2 is used**

但实测启动日志是：

```
WARNING: Model Runner V2 does not yet support routed experts capture;
using the V1 model runner instead.
```

vLLM 自己回退到 V1，`routed_experts` 正常返回，数据完整且符合契约。

**教训**：文档的"必须"和代码的实际行为，是两件需要分别验证的事。
只信文档会得出"跑不了"，只信警告会得出"功能缺失"，**只有实际跑一遍才知道两个都不对**。

### 4.4 `repetition_penalty` 会间接污染路由

模型自带的 `generation_config.json` 会把 `repetition_penalty` 设为 **1.05**：

```
WARNING: Default vLLM sampling parameters have been overridden by the model's
`generation_config.json`: {'repetition_penalty': 1.05, 'temperature': 0.7, ...}
```

它会改变 logits → 改变 router 的输入 → **改变专家选择**。

所以验证路由时必须显式设 `"repetition_penalty": 1.0`，否则测量的是"带惩罚的路由"。

### 4.5 vLLM 版本差 6 个小版本，显存需求差 3.5 倍

| vLLM 版本 | MoE 后端 | 权重精度 | 显存需求 | 24GB 卡 |
|---|---|---|---|---|
| **0.27.1** | `EMULATION` | 反量化成 **BF16** | **~28 GiB** | ❌ OOM |
| **0.33.0** | `HUMMING` | 原生 int4 | 7.84 GiB | ✅ 轻松 |

旧版遇到不认识的量化格式会静默回退到 EMULATION，把 int4 权重全反量化成 BF16。
**只打一行 WARNING，然后开始把 7.84 GiB 膨胀成 28 GiB。**

如果没注意版本号，会误判为"卡不够"。

### 4.6 局部性在 KV 层，不在专家层（差 10~30 倍）

这是本次最有架构价值的一组对照。

#### 三组实测

| 维度 | 指标 | 数值 | vs 随机 |
|---|---|---|---|
| 专家访问（层间，同 token 相邻层）| Jaccard | 0.0388 | **1.14×** |
| 专家访问（时间，相邻 token 同层）| Jaccard | 0.0917 | 2.7×（中位数 0，无稳定段）|
| **KV prefix cache** | 命中率 | **99.05%** | — |

#### 专家层：两个维度都接近随机

- **层间**：0.0388 vs 随机基线 0.034，只高 14%——噪声级别
- **时间**：0.0917 看似 2.7 倍，但中位数是 0，且逐 token 序列显示**点状抖动、无稳定段**：
  ```
  [0.05, 0.21, 0.06, 0.08, 0.03, 0.11, 0.04, 0.09, 0.02, 0.05, ...]
         ↑ 0.21 之后立刻掉到 0.06
  ```
  只有 11.4% 的步长 > 0.2，且孤立不成块。

**原因**：每层 router 的输入是该层自己的 hidden state，层间快速变化；时间维度只有输入层相近，24 层传递后已衰减。**router 本质是内容哈希，输入变化快，输出就随机。**

#### KV 层：99% 命中

实验：构造 840 token 的重复文本 prompt，`max_tokens=1`（让 prefill 成为主要开销）：

| 请求 | queries | hits | 命中率 | 耗时 |
|---|---|---|---|---|
| 冷（第一次）| 840 | 0 | 0% | 0.091 s |
| 热（第二次，同 prompt）| 840 | 832 | **99.05%** | **0.030 s** |

**延迟快 3 倍。**

还有个漂亮的细节：`832 = 52 × 16`，`840 − 832 = 8`。
**说明 block_size = 16，prefix cache 按 block 对齐，最后不足一个 block 的 8 个 token 不缓存。**

#### 为什么差这么多：本质区别

```
KV cache   缓存的是「输入的历史」（context）        → 天然可复用
专家缓存   缓存的是「计算的中间结果」（路由决策）    → 依赖当前 hidden state，不可复用
```

**可复用性来自「输入结构」，不是「计算结构」。**

LLM 的真实负载里，前缀重复是常态（system prompt、多轮对话、RAG 固定 context、批量评测模板）；而专家路由是内容哈希，没有任何可复用的东西。

#### 对引擎设计的直接含义

**可以彻底划掉的方向**（均有实测依据）：

| 方向 | 实测收益 |
|---|---|
| 专家级 LRU / LFU 缓存 | 上限 10~15%，且命中点状不可预测 |
| 按热度分层（top-10 常驻 GPU）| 只覆盖 19.2% |
| 时间维度预取 | 无稳定段 |
| 层间流水线预取 | 1.14× 随机 |

**应该成为核心的方向**：

| 方向 | 依据 |
|---|---|
| **prefix cache** | 99% 命中，最高 ROI |
| KV block 管理（16-token 对齐、eviction）| 实测对齐行为 |
| PD 分离 + KV 传输（跨实例复用）| 同上 |
| offload 按「确定性带宽预算」而非命中率 | 专家访问随机，但流量可精确计算 |

**一句话**：引擎该优化的是「怎么复用已经算过的 KV」，不是「怎么复用已经选过的专家」——后者根本不存在。

#### 一个可复用的方法论教训

本节和 4.1 是同一类错误的两个变体：

| 假设 | 数据 | 结果 |
|---|---|---|
| `unique = 60` → 无局部性 | 三个正确指标 | ❌ 错（有过散 + 弱时序相关）|
| p90 = 0.33 → 存在稳定段 | token 级序列 | ❌ 错（点状，非块状）|
| 层间可能有相关性 | 0.0388 vs 0.034 | ❌ 错（纯随机）|

**三次假设，三次被数据修正，全部源于「从聚合指标推断结构」。**

> **聚合量（基数、均值、分位数）不能恢复时间结构。**
> 要看结构，必须看序列本身或联合分布，而不是边际分布。



---

## 5. 环境门槛清单：11 个坑

### 5.1 本地 WSL2 + 8GB 显卡阶段（坑 1-8，最终判定为硬件天花板）

| # | 症状 | 根因 | 解法 |
|---|---|---|---|
| 1 | `ValidationError: cpu_offload_params Input should be a valid set` | Python API 要 `set[str]`，CLI 接受 `str` | 传 `{'experts'}` 而非 `'experts'` |
| 2 | `Free memory ... less than desired GPU memory utilization` | `gpu_memory_utilization` 按**空闲显存**校验，不是总显存 | 降到 0.85，关掉吃显存的程序 |
| 3 | `NVRTCCompiler failed` | Blackwell sm_120 + CUDA graph 编译失败 | `--enforce-eager` |
| 4 | `failed to open libnvrtc-builtins.so.13.0` | 库在 venv 的 `site-packages/nvidia/cu13/lib/`，但 NVRTC 是 C 层加载器，不走 Python 路径 | 设 `LD_LIBRARY_PATH` |
| 5 | `driver.c: fatal error: Python.h` | Triton 要 JIT 编译 C 扩展，缺 Python 开发头文件 | `apt install python3-dev` |
| 6 | `Could not find nvcc ... '/usr/local/cuda' doesn't exist` | vLLM 选了 FlashInfer 做采样，它要 nvcc | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 7 | `Pinned memory is not available on this platform` | **WSL2 实测支持 pinned memory（可分配 3GB），是 vLLM 平台判断误判** | 改 `vllm/utils/platform_utils.py:57` → `return True`（**必须改源码，spawn 子进程不继承 monkey patch**） |
| 8 | offload 报告搬运 3.01 GB，但 `Model loading took` 仍是 7.84 GiB | **vLLM 0.33 的 UVA offloader 对 GPTQ MoE 只"记账"不"搬运"**：多复制一份到 CPU（加载变慢），不释放 GPU 原件 | 无解。`--max-num-batched-tokens 512` 只省激活（KV 预算 -0.46→-0.11 GiB），补不上 7.85 GiB 模型本体 |

**第 8 个坑的最终数字**：

```
模型实占      7.85 GiB
GPU 可用      6.85 GiB（Windows 占 1.11）
缺口          ≈ 1 GiB
```

**结论：8GB 消费级卡 + WSL2 + vLLM 跑 14B MoE，在当前软件栈下是硬件天花板。**
不是参数没调对，是 offload 实现在「WSL + Blackwell sm_120 + GPTQ 量化 MoE」
这个四重叠加场景下存在缺陷——这种组合不在它的测试矩阵里。

### 5.2 云端 GPU 阶段（坑 9-11）

| # | 症状 | 根因 | 解法 |
|---|---|---|---|
| 9 | `Int4 weights are dequantized to BF16 at load time` + OOM | 实例预装 vLLM **0.27.1**，不认识 GPTQ MoE，回退 `EMULATION` 后端，7.84 GiB → 28 GiB | 升级到 `vllm==0.33.0` |
| 10 | `flash_attn_2_cuda...so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` | flash-attn 2.8.3 是针对 torch 2.13 编译的，装 vLLM 时 torch 升到 2.14，**C++ ABI 不匹配** | `pip uninstall -y flash-attn`，走 native attention |
| 11 | batch 组成一变，路由翻转 3/768 | `VLLM_BATCH_INVARIANT` 默认关闭 | `VLLM_BATCH_INVARIANT=1`（代价：初始化 10s→130s） |

**坑 10 的本质**和坑 4 同一类：**Python 层面的依赖能解析，C++ 层面的 ABI 不会自动跟随。**

---

## 6. 工程结论：适配层到底该做什么

本次验证最大的收获，不是"跑通了"，而是**发现"跑通"和"可复现"是两件事**。

### 6.1 三层校验：前两层是"自证"，只有第三层能"证伪"

```
第一层：格式校验 —— 自证
  shape / dtype / 专家 ID 范围 / -1 哨兵
  规范来源是"对当前实现的观测"，不是第一性原理
  → 只能证明"和这个版本一致"，实现一变契约就失效

第二层：环境校验 —— 自证
  vLLM 版本（影响 MoE 后端、显存需求 3.5 倍）
  batch invariant 是否开启
  repetition_penalty 是否被模型 config 覆盖
  → 只能证明"配置看起来对"，不能证明"行为真的对"

第三层：可复现性校验 —— 证伪
  两条独立路径是否收敛
  → 不依赖规范，不需要知道"应该是什么"
  → 唯一的证伪手段
```

#### 为什么前两层不够

**符合性测试（conformance）** 验证"输出符合规范"。但规范本身可能是错的——
`shape = (num_tokens-1, layers, top_k)` 是从 vLLM 输出反推的**描述**，不是**规范**。
实现一变，契约跟着变，而校验器毫无察觉。

**差分测试（differential）** 验证"两条路径等价"。它不需要规范：

```
断言 A == B
若不等 → 至少一条错了（不需要知道哪条）
```

#### 四条差分路径，没有一条需要契约文档

| 路径 A | 路径 B | 能发现什么 |
|---|---|---|
| batch=1 | batch=N | 批不变性（本次发现 3/768 翻转就是这条查出来的）|
| 确定性 kernel | 参考实现（慢但正确）| kernel 数值偏差 |
| 单卡 | 多卡 TP | 并行归约顺序差异 |
| 冷启动 | 热启动（有 prefix cache）| prefix caching 污染路由 |

**新引擎的测试体系应该以这四条为骨架，而不是以 schema 校验为核心。**

### 6.2 一个可直接落地的检查清单

```python
# 跑 R3 之前必须确认的三件事
assert vllm.__version__ >= "0.30", "旧版会静默回退 EMULATION，显存需求涨 3.5 倍"
assert os.environ.get("VLLM_BATCH_INVARIANT") == "1", "不开则 batch 一变路由就翻转"
assert sampling_params.repetition_penalty == 1.0, "模型 config 的 1.05 会污染路由"

# 拿到数据后必须做的两项校验
assert routed.dtype in (np.uint8, np.uint16)
assert routed.shape[0] == prompt_len + gen_len - 1     # 末 token 无路由

# 以及最重要的一项：跨 batch 一致性
assert np.array_equal(route_single, route_batch2), "批不变性不成立，R3 不可用"
```

### 6.3 商业价值在哪

`VLLM_BATCH_INVARIANT=1` 为什么默认关闭？不是忘了开，而是它要付性能代价
（32 秒编译 + 更慢的 kernel），而**绝大多数推理场景根本不在乎路由是否可复现**——
用户只要正确的文本。

只有 RL 训练这个场景，才把"可复现"看得比"快"更重要。

**所以这层适配的本质是：用推理性能换训练可复现性。**
这个交换，只有懂 RL 的人才会主动做。

---

## 7. 尚未验证的部分

本次只完成了"拿到路由 + 确认可复现"。真正的 R3 闭环还差最后一环：

| 待验证 | 说明 | 难度 |
|---|---|---|
| 训练侧回放 | 用这份 `routed_experts` 在 transformers/Megatron 前向里强制指定专家 | 中 |
| logprob 归零 | 回放后与原推理的 logprob 差异应为 0 | 中 |
| 跨引擎一致性 | vLLM vs SGLang 的路由是否一致 | 高 |
| dtype 统一 | vLLM uint8/uint16、部分 fork int16、Miles int32，体积差 2-4 倍 | 中 |

第 1、2 步在 CPU 上就能做。`mini_moe_r3.py` 已在微型 MoE（4 专家 top-2）上验证数学命题，
回放后 logprob 差异为 `0.000e+00`；反例（从 top-k 清零后的 probs 取权重）会产生
**15.84% 的权重偏差**。缺的是"真模型 + 真路由"的组合验证。

---

## 附录 A：完整复现命令

### A.1 启动服务

```bash
# 必须带 VLLM_BATCH_INVARIANT=1
VLLM_BATCH_INVARIANT=1 vllm serve /mnt/workspace/model \
  --enable-return-routed-experts \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

启动日志必须包含：

```
version 0.33.0
Using 'HUMMING' WNA16 MoE backend          ← 不是 EMULATION
Using HummingIndexedExperts
RoutedExpertsCapturer: allocating buffer with max_tokens=2048, num_layers=24, num_experts_per_tok=4
Uvicorn running on http://0.0.0.0:8000
```

### A.2 下载模型（ModelScope 内网，约 1 分钟）

```bash
python -c "
from modelscope import snapshot_download
print(snapshot_download('qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4', local_dir='/mnt/workspace/model'))
"
```

### A.3 请求与解析

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/workspace/model","prompt":"Hi","max_tokens":8,"temperature":0,"repetition_penalty":1.0}' \
  -o resp.json
```

```python
import json, base64, io, numpy as np

r = json.load(open("resp.json"))
b = r["choices"][0]["routed_experts"]        # base64 编码的 .npy
a = np.load(io.BytesIO(base64.b64decode(b)))

print(a.shape, a.dtype, a.min(), a.max())
```

### A.4 批不变性对照

```bash
# A：单独跑
curl -s ... -d '{"prompt":"Hi", ...}' -o single.json

# B：和另一条一起跑
curl -s ... -d '{"prompt":["Hi","Hello there"], ...}' -o batch2.json
```

```python
import json, base64, io, numpy as np

def load(path, i=0):
    d = json.load(open(path))
    return np.load(io.BytesIO(base64.b64decode(d["choices"][i]["routed_experts"])))

a, b = load("single.json"), load("batch2.json")
print("identical:", np.array_equal(a, b))
print("diff:", int((a != b).sum()), "/", a.size)
```

---

## 附录 B：原始数据

### B.1 单次请求（prompt="Hi", max_tokens=8）

```
shape  (8, 24, 4)
dtype  uint8
range  0 ~ 59
unique 60
token0-layer0  [33, 52, 7, 54]
npy header  {'descr': '|u1', 'fortran_order': False, 'shape': (8, 24, 4)}
```

### B.2 批不变性

| 配置 | identical | diff |
|---|---|---|
| 默认 | False | 3 / 768 |
| `VLLM_BATCH_INVARIANT=1` | True | 0 / 768 |

### B.3 显存与 KV cache（24GB 卡）

```
Model loading took      7.84 GiB
Available KV cache      10.61 GiB
GPU KV cache size       57,936 tokens
Max concurrency (512)   113.16x
CUDA graph memory       0.35 GiB
Peak activation         1.08 GiB
```

---

## 附录 C：版本差异对照

| 配置项 | 失败配置 | 成功配置 |
|---|---|---|
| 硬件 | RTX 5060 Laptop 8GB | NVIDIA 24GB |
| 系统 | Windows + WSL2 | Linux（云端） |
| vLLM | 0.33.0 / 0.27.1 | **0.33.0** |
| MoE 后端 | `EMULATION`（反量化 BF16） | **`HUMMING`（原生 int4）** |
| flash-attn | 2.8.3（与 torch 2.14 ABI 不匹配） | 已卸载，走 native |
| 显存占用 | 28 GiB（OOM） | 7.84 GiB |
| batch invariant | 未开（3/768 翻转） | **开启（0/768）** |

---

*报告生成于 2026-11-14，数据来自实测，非文档摘录。*
