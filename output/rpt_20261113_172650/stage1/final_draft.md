# 大模型推理引擎技术调研与自研路线决策报告

> 调研日期：2026-11-13　|　数据来源：GitHub API 实时采集 + 官方文档 + 论文　|　性质：立项决策参考，非性能承诺

## 一、摘要

本报告基于九轮 GitHub 深度调研，覆盖 60 余个推理引擎与相关基础设施项目，回答一个问题：**在 2026 年，一个没有大厂资源的团队，自研大模型推理引擎的正确切入点是哪里？**

核心结论是：**不自研通用推理引擎，先做 RL 训练与推理引擎之间的"数据契约适配层"。**

理由有三。其一，通用引擎已被 vLLM（96,857★）、SGLang（38,804★）与 TensorRT-LLM 的生态壁垒锁死，自研无差异。其二，Kernel 层的极限已被 DeepSeek 的 DeepGEMM（BF16 达硬件极限 99.8%）探明，难以超越。其三，推理与 RL 训练之间至今存在一条**字段级不统一的缝隙**——同一份 MoE 路由数据，vLLM、SGLang、Megatron 用了三种不同 dtype 与三种不同传输格式，边界约定互不一致。这条缝隙有真实需求、有可对齐的事实标准、且工程量可控。

本报告给出四阶段实施路线、50 行代码即可验证的最小闭环，以及明确的验收标准。

---

## 二、调研方法与范围

### 2.1 数据口径

- **采集方式**：GitHub API 于 2026-11-13 实时采集 stars / forks / open issues / 最后提交时间 / archived 标记 / 许可证；辅以项目官方文档、README、以及 NeurIPS / SOSP / OSDI / FAST 等会议的原始论文。
- **性能数据口径**：吞吐、延迟等数字来自多方公开 benchmark 与官方技术报告，**仅代表量级区间**。不同硬件、量化格式、序列长度分布下，结论可能反转甚至完全相反。

### 2.2 调研范围

九轮调研按四个维度展开，覆盖以下层次：

| 维度 | 覆盖内容 |
|------|---------|
| 引擎本体 | 服务端（vLLM / SGLang / TensorRT-LLM / LMDeploy / TGI）、端侧（llama.cpp / Ollama / MLC-LLM）、多元算力（Chitu / xLLM / trnrt / vllm-ascend）、极限内存（kimi-k3-in-c / turbo-fieldfare） |
| 零件层 | kernel（FlashInfer / kt-kernel）、投机解码（EAGLE / speculators）、KV 量化（KVarN）、MoE 库（Tutel）、SSD 流式（inferssd / llama-modchip） |
| 语言与硬件 | Rust（mistral.rs / Candle）、NPU（FastFlowLM / Tengine / lemonade）、编译器（mlir-aie / TileLang） |
| RL 协同 | 权重同步、部分 rollout、R3 路由回放、sampling mask、数据契约 |

### 2.3 局限声明

以下内容为**判断与建议**，与"已验证事实"分开标注：性能对比、机会评估、实施顺序。以下内容存在**不确定性**：vLLM 字段仍在演进（`weight_versions` 仅存在于 fork）、上游协议统一进度不明。

---

## 三、市场格局

### 3.1 三层分化

2026 年的推理基础设施已经分化为三层，任何新进入者都必须先明确自己属于哪一层：

| 层次 | 代表项目 | 定位 |
|------|---------|------|
| 引擎层 | vLLM、SGLang、TensorRT-LLM | 承载模型权重、调度请求、管理 KV cache |
| 编排层 | NVIDIA Dynamo 1.0、llm-d（CNCF Sandbox） | 在引擎之上做 PD 分离决策、多副本调度 |
| KV 层 | Mooncake、LMCache、NIXL | 把 KV cache 从引擎解耦成独立基础设施 |

**关键含义**：编排层与 KV 层都不自己写引擎，而是在引擎之上/之下做差异化。这是"不重造引擎也能生存"的实证。

### 3.2 主流引擎对比

| 引擎 | Stars | 架构范式 | 核心优势 | 主要缺陷 |
|------|-------|---------|---------|---------|
| vLLM | 96,857 | 分页内存管理 | 生态最大、模型 day-0、硬件最广、零编译 | 单卡吞吐非第一、代码膨胀（1 万+ open issues） |
| SGLang | 38,804 | 前缀复用优先 | RadixAttention 使 Agent/RAG 领先、约束解码最强 | 运维生态小、前缀不重复时优势消失 |
| TensorRT-LLM | 15,075 | 编译式 AOT | NVIDIA 上吞吐/尾延迟最强（+15~40%） | 锁 NVIDIA、编译约 28 分钟、许可非标准 OSI |
| LMDeploy | 8,140 | 手写 kernel 持久批处理 | INT4 显存受限优秀、国产模型适配快 | 模型覆盖窄、双引擎分裂 |
| llama.cpp | 134,741 | 可移植 C/C++ | 硬件覆盖无敌、2-bit 可跑、零依赖 | 不为并发设计、KV 浪费 10-20% |
| Ollama | 186,844 | 分发层 + 运行时管理 | 体验最好、开发者心智第一 | 本质是发行版非高性能引擎 |

### 3.3 两个格局信号

1. **HuggingFace TGI 已归档**（`archived: true`，最后提交 2026-03-21），HF 自家 Inference Endpoints 转向 vLLM。曾经的四强之一退场，说明**这个领域没有"稳定胜利者"**。
2. **DeepSeek 的 FlashMLA 放弃 Hopper 支持**，只保留 Blackwell 与 Ascend 950。前沿实验室已不再把上一代 NVIDIA 芯片当作优化目标——**平台押注的窗口正在移动**。

---

## 四、技术路线版图（简述）

以下路线不展开，仅记录其存在与代表项目，作为"为什么不自研这些方向"的背景：

| 路线 | 代表项目 | Stars | 一句话定位 |
|------|---------|-------|-----------|
| 多元算力 | Chitu 赤兔（清华 PACMAN） | 3,000 | 昇腾/沐曦/海光/摩尔线程/燧原 + NVIDIA，单卡跑 DeepSeek-R1 671B |
| 极限内存 | kimi-k3-in-c | 11,900 | 2.78T 参数跑在单 CPU 8.24GB 内存，纯 C99 |
| 存储层级延伸 | inferssd / llama-modchip | 1,000 / 1,200 | 从 SSD 流式加载专家；284B 跑在 64GB 纯 CPU |
| Rust 引擎 | mistral.rs | 7,600 | 45+ 架构、PagedAttention、单二进制、内置 agentic runtime |
| NPU | FastFlowLM / Tengine | 2,000 / 4,500 | AMD Ryzen AI NPU；阿里 Tengine 连 RISC-V 都支持 |

这些路线各有真实市场，但共性是需要**算子功底或厂商关系**，非本方案的切入点。

---

## 五、可复用零件层

**造引擎已经不需要从零写。** 2026 年的现实是"组装 + 在一层做差异"：

| 零件 | 项目 | Stars | 定位 |
|------|------|-------|------|
| Attention/MoE kernel | FlashInfer | 6,900 | 事实上的 kernel 中间层，vLLM/SGLang/TRT-LLM 都在用 |
| CPU 端算子 | kt-kernel / KTransformers | 19,700 | 热专家在 GPU、冷专家在 CPU，NUMA 感知 |
| 投机解码 | EAGLE / speculators | 2,600 / 955 | 算法可插拔化是趋势 |
| KV 量化 | KVarN（华为） | 520 | 免校准、一个开关、3-5 倍上下文 |
| MoE 库 | Tutel（微软） | 1,000 | FP8/NVFP4/MXFP4 全覆盖 |

**结论**：自研的应该只有**调度器与状态管理**——那是唯一能形成差异的地方。

---

## 六、纵向范本：DeepSeek 昇腾六件套

2026-09-30，DeepSeek 开源了整套昇腾软件栈（MIT），六个组件，是"自研引擎"最完整的范本：

| 组件 | 关键数据 |
|------|---------|
| TileLang（昇腾版） | kernel 语言，直接生成 Ascend C，同一套代码编译到 GPU 与 NPU |
| DeepGEMM-Ascend | BF16 矩阵乘达硬件上限 99.8%、FP8 达 99.5% |
| FlashMLA | prefill 95% 峰值、decode 83%；弃 Hopper，留 Blackwell + Ascend 950 |
| DeepEP-Ascend | EP8 下 dispatch 373-375 GB/s，约 90-95% 物理带宽上限 |
| TileKernels / DeepSelect | 向量/内存算子；top-k 选择 |

配套硬件：昇腾 950DT（144GB 自研内存、4TB/s、1 PFLOPS FP8）；SuperPoD Flex（128 卡单层 3.2Tbps，可扩 256K 卡）。

**重要边界（已验证事实）**：DeepEP-Ascend 的带宽数据来自 DeepSeek 提供的 **PoC HDK（非公开发售版本）**，其 README 明确提示早期 PoC 硬件可能更低，官方基线应以华为商用品为准。

---

## 七、被忽略的战场：RL 与推理的接口（本报告重点）

### 7.1 三个独立偏差源

RL（强化学习）训练不稳定，有三个**互相独立**的来源，各有一个对应机制，缺一不可：

| 偏差源 | 表现 | 修法 |
|--------|------|------|
| 路由偏差 | MoE 推理与训练为同一 token 选不同专家 | R3（Router Replay） |
| 动作空间偏差 | top-p 截断使采样分布 ≠ 训练全词表 | sampling mask |
| 打分偏差 | 推理与训练 logprob 因 kernel/精度/策略滞后不相等 | score centering |

**社区最常见的错误是只修一个就宣布解决，另外两个仍在悄悄污染梯度。**

### 7.2 R3 实现现状

| 实现方 | 状态 |
|--------|------|
| vLLM | `--enable-return-routed-experts` 已上线，返回 `routed_experts`（[gen_len, layers, top_k]，dtype uint8/uint16）与 `prompt_routed_experts` |
| NeMo RL v0.7 | PR #2590 实现 R3（vLLM 路线），约 4059 行，支持 Sync/Async RL |
| 阿里 ROLL | 也实现 R3，但走 SGLang 路线，**明确不支持 vLLM rollout** |
| Megatron-LM | PR #79 给出最细边界处理，未合入主干 |
| 论文 | R3 = arXiv:2606.02437 |

### 7.3 字段级分歧——协议不统一的最硬证据

同一份语义数据，三种 dtype、三种传输格式：

| 实现 | dtype | 体积（同一数据） |
|------|-------|-----------------|
| vLLM 官方 | uint8（≤256 专家）/ uint16 | 1× |
| 部分 fork 文档 | int16 | 2× |
| Miles 实测 | int32 | 2-4× |

**32K token、60 层、top_k=8 的一条 rollout 轨迹，路由数据就约 60MB（int32）。** 统一 dtype 后同样数据可降到 15MB。

### 7.4 四类静默失败（不报错，只是数据不对）

这是实现协议时最危险、最该先处理的坑：

1. **CUDA graph 与 NCCL 死锁**：只有 rank 0 持有 device buffer 会导致各 rank 代码路径不同、图结构不对称，collective 中偶发 hang。解法是所有 rank 都分配真实 buffer。
2. **被抢占的请求静默丢路由**：host cache 中已累积的路由被丢弃且不发出，消费方只看到 `None`，**无任何信号**。
3. **异步调度下 EOS 静默丢路由**：stop 判断读到 placeholder `-1`。
4. **流式不可用 / Rust HTTP 前端永不返回该字段**。

### 7.5 成本真相

| 指标 | 数值 |
|------|------|
| R3 吞吐开销 | 约 -2%（ISL/OSL=1024） |
| R3 显存开销 | 约 14MB/GPU（40 层/8192 token/top-22） |
| R3 精度影响 | 零差异（GSM8K pass@1 完全一致） |
| 传输体积 | 约 60MB/条轨迹（32K token） |
| sampling mask 吞吐代价 | num_sampler_logprobs=8 且单 API server 时减半；加 8 个 API server 后 <3% |

**关键认知**：R3 的那 2% 吞吐不是成本，60MB 传输才是；mask 的成本不在显存，而在 **CPU 序列化**。

### 7.6 sampling mask 的配对要求

`--return-sampling-mask` 必须与 `--logprobs-mode processed_logprobs` **同时使用**。只开 mask 不改 logprobs 模式，返回的 logprob 仍在全词表归一化，而 mask 定义的是核内支撑集——分子分母不在同一空间，比率照样有偏。训练侧还需：

```python
keep = torch.zeros(vocab_size, dtype=torch.bool); keep[mask_ids] = True
log_prob = log_softmax(logits.masked_fill(~keep, -inf))[sampled_token_id]
```

---

## 八、机会评估

### 8.1 已被填满的坑（不要碰）

PagedAttention 式 KV 管理、连续批处理、OpenAI 兼容协议、基础量化（AWQ/GPTQ/FP8）、单点投机解码——这些是**入场券，不是差异点**。

### 8.2 三条备选路线的取舍（判断）

| 路线 | 为什么不选 |
|------|-----------|
| 约束型引擎（国产芯片/端侧） | 需算子功底与厂商关系，Chitu 已占位 |
| 调度层创新（Agent DAG） | 工程量小但缺场景验证，且尚无明确需求方 |
| 做零件（kernel/量化） | DeepGEMM 的 99.8% 无法超越 |

### 8.3 选中的缝隙

**跨引擎的 RL 数据契约适配层**。理由：需求真实（16 个开源 RL 库各自造轮子）、有可对齐的事实标准（vLLM 的字段）、工程量可控、且两派已分叉（NeMo RL 走 vLLM，ROLL 走 SGLang 且互不兼容）。

---

## 九、最终方案

### 9.1 决策：先实现协议，不定义协议

**理由**：定义协议需要生态号召力，实现协议可在实践中积累话语权；且 vLLM 的 `routed_experts` / `SamplingMask` 已是事实标准，与它对齐的成本远低于另起一套。

### 9.2 定位

跨引擎的 RL 数据契约适配层——统一 `routed_experts` 与 `sampling_mask` 的 dtype、形状、边界约定，并提供一致性校验。

### 9.3 四阶段实施路线

| 阶段 | 周期 | 目标 |
|------|------|------|
| 阶段一 | 约 1 周 | 在 vLLM 上跑通 R3 + sampling mask 的最小验证 |
| 阶段二 | 1-2 周 | dtype 归一化 + 三类边界处理（末 token 无路由 / 中止为 null / 不可映射 -1 哨兵） |
| 阶段三 | 2-3 周 | vLLM / SGLang 双引擎适配 |
| 阶段四 | 持续 | 一致性校验工具与指标 |

### 9.4 最小验证方案（50 行代码可验证）

1. 启动：`vllm serve <moe-model> --enable-return-routed-experts`，并设 `VLLM_BATCH_INVARIANT=1` 保证批不变性。
2. 取一条 `routed_experts`，断言 `shape[0] == prompt_len + gen_len - 1`（末 token 无路由）。
3. 训练侧把同一批 token 的专家强制成该索引。
4. 对比 logprob 差异：**归零则契约成立；不归零则已定位到第一个要修的点。**

### 9.5 验收标准与指标

- logprob 差异归零或低于阈值。
- `unmappable_frac` 可观测（不可映射 token 比例）。
- 抢占场景能被**显式检测**，而非静默丢数据。

### 9.6 风险与边界

- vLLM 字段仍在演进（`weight_versions` 仅存在于 fork），上游合并状态不确定。
- PD 分离与 R3 目前不兼容（Nixl/Mooncake 等 KV connector 被拒）。
- 若上游统一协议，适配层价值下降——**应对**：把适配层做成可被上游吸收的形态。

---

## 十、结论与下一步

九轮调研把问题收敛到一句话：**在约束条件里做引擎，比在无约束条件里做引擎容易十倍；而当前最真实的约束条件是"推理与训练之间没有统一的契约"。**

未来 30 天的三个动作：

1. 用 50 行最小验证脚本在 vLLM 上跑通 R3，确认 logprob 差异能否归零。
2. 写 dtype 归一化层，覆盖 uint8/uint16/int32 三种输入。
3. 处理三类静默失败，重点是被抢占请求的显式检测。

---

## 附录 A：术语表

| 术语 | 含义 |
|------|------|
| R3（Router Replay） | 训练前向重放 rollout 时选中的 MoE 专家索引 |
| PD 分离 | 把 prefill 与 decode 拆到不同 GPU 池 |
| EP（Expert Parallel） | 专家并行，MoE 的分布式策略 |
| goodput | SLO 约束下的有效吞吐，区别于峰值吞吐 |
| RadixAttention | SGLang 的基数树前缀复用 |
| sampling mask | 每个生成 token 通过 top-k/top-p 后存活的词表 ID 集合 |
| score centering | 训练侧减去采样器下 logprob 的期望 |

## 附录 B：数据来源

- GitHub API（2026-11-13）：各项目 stars / forks / open issues / archived / license
- vLLM 官方文档：`docs/training/routed_experts.md`、`docs/training/sampling_mask.md`、`examples/rl/routed_experts_e2e.py`
- NeMo RL v0.7 Release Notes（Router Replay / R3）
- 阿里 ROLL 文档：`User Guides > Advanced Features > router_replay`
- 论文：R3（arXiv:2606.02437）、PagedAttention（SOSP'23）、Sarathi-Serve（OSDI'24）、DistServe（OSDI'24）、Mooncake（FAST'25）
