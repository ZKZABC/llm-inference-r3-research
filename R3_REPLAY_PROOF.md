# R3 回放证明

> **副标题**：从「拿到路由」到「回放精确」——契约的第二个必要条件
>
> **日期**：2026-11-14
> **前置**：[`R3_VERIFICATION_REPORT.md`](R3_VERIFICATION_REPORT.md) 记录了第一个必要条件（推理侧可复现）
> **一句话结论**：**给定一组路由，前向结果是唯一确定的。实测 bit-identical。**

---

## 摘要

R3 要成立，需要两个**互相独立**的必要条件：

| # | 条件 | 验证方式 | 状态 |
|---|---|---|---|
| ① | 推理侧记录的路由可复现 | 同一 prompt 跨 batch 对比 | ✅ 开 `VLLM_BATCH_INVARIANT=1` 后 0/768 |
| ② | 训练侧能精确回放该路由 | 注入自己产生的路由，看输出是否变 | ✅ **bit-identical** |

第 ① 条在前一份报告里完成。这份报告做第 ② 条。

**为什么第 ② 条必须单独验**：它和 ① 是两件事。① 保证「记录的数据是稳定的」，② 保证「用这份数据重算，能得到同样的结果」。少任何一个，R3 都不成立。

---

## 1. 回放的语义

R3 的回放不是「让训练侧的 router 也算出同样的结果」，而是：

> **跳过 router 的选择，直接用推理侧记录的 top-k 索引。**

```python
# 正常前向
weights, ids = topk(softmax(logits), k)

# 回放
ids     = recorded[layer]              # ← 强制用记录的索引
weights = gather(softmax(logits), ids) # ← 权重仍从「本次 logits」算
weights = weights / weights.sum(...)   # ← 归一化规则不能漏
```

**关键：只替换 `ids`，不替换权重。**

如果连权重一起注入，就绕过了 softmax 归一化，测出来的差异无法归因——这正是 `r3_contract.check_replay_weights()` 防的那个坑（实测错误做法会产生 15.84% 的权重偏差）。

---

## 2. 实验设计

### 2.1 判据

不是「注入后输出和原来完全一样」，而是：

```
注入后的差异  ≤  不注入时的基线差异
```

**为什么需要基线**：回放要重写 MoE 的 forward（不能用原实现，因为要换掉 top-k）。重写本身就有精度噪声，所以「完全一样」是个过强的要求。

基线怎么来：让 `InjectedMoE` **自己算 top-k**（不注入），和原模型对比，得到的就是「重写引入的噪声」。

### 2.2 三个实验

| 实验 | 模型 | 目的 |
|---|---|---|
| A | 小模型（`Qwen2MoeConfig` 随机初始化，2 层） | 验证机制 |
| B | 真实模型（Qwen1.5-MoE-A2.7B-GPTQ-Int4，24 层） | 验证结构理解 |
| C | 真实模型 + 5 token prompt | 验证多层多 token |

---

## 3. 结果

### 3.1 小模型（实验 A）

```
配置：hidden 256 / 8 专家 / top-4 / 2 层 / fp32

步骤 2（自算 topk，不注入）  13348/16384 不同 · max=4.768e-07
步骤 4（注入）                13348/16384 不同 · max=4.768e-07
```

**两者完全相等** → 注入未引入额外误差。4.77e-07 是 fp32 的 2-4 ulp。

### 3.2 真实模型（实验 B / C）

```
配置：Qwen1.5-MoE-A2.7B-GPTQ-Int4 / 24 层 / bf16 / CPU
加载 gptqmodel 7.10.0，设备 cpu，kernel TorchAtenLinear

实验 B（"Hi"，1 token）
  步骤 2  bit-identical ✓
  步骤 3  bit-identical ✓

实验 C（"The capital of France is"，5 token）
  每层 unique 12~19（5 token × 4 = 20 槽位）
  步骤 2  bit-identical ✓
  步骤 3  bit-identical ✓
```

**真实模型反而比小模型更准**——精确相等，不是近似。

原因：gptqmodel 的 `LinearQwen2MoeSparseMoeBlock` 和 `InjectedMoE` 的累加路径一致（都是 `index_add_` + expert 顺序）。小模型那次用的是 transformers 原生实现，累加细节有差异。

---

## 4. 意外发现：三个会静默算错的实现细节

三个都不是报错，是**算错但看起来正常**。任何一个错了，实验结论都会反过来。

### 4.1 `norm_topk_prob` 是配置项，不是固定行为

```python
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
if self.norm_topk_prob:                    # ← 条件性的
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
```

`Qwen2MoeConfig` 默认 `False`。无条件除以 sum 会让**所有权重整体偏小**，输出差 **0.07**（不是 1e-7，是 0.07）。

### 4.2 HF 用 `index_add_`，不是 `index_put`

```python
final_hidden_states.index_add_(0, token_idx, current_hidden_states)
```

`out[sel] = out[sel] + ...` 是「取值→计算→写回」，`index_add_` 是原地累加。累加顺序不同，fp32 下差 2-4 ulp。

改成一样之后：**0.07 → 4.77e-07**，差 15 万倍。

### 4.3 gptqmodel 会替换 MoE 实现

```
Patched transformers.models.qwen2_moe.Qwen2MoeSparseMoeBlock
  -> defuser.modeling.unfused_moe.qwen2_moe.LinearQwen2MoeSparseMoeBlock
```

替换后的类：`experts` 是 `ModuleList`（不是 3D 张量），`top_k` / `num_experts` / `norm_topk_prob` 挂在 **block 上**而不是 gate 上。

但 `gate` 仍是原始的 `Qwen2MoeTopKRouter`。所以读属性要**两边都查**。

---

## 5. 完整证明链

```
① 推理侧记录路由 R
   └─ 条件：VLLM_BATCH_INVARIANT=1 + enable-return-routed-experts + repetition_penalty=1.0
   └─ 实测：0/768（同 prompt 跨 batch）
   └─ 状态：✅ 见前一份报告

② 训练侧注入 R
   └─ 只替换 top-k 索引，权重从本次 logits gather
   └─ 实测：24 层 × 5 token，bit-identical
   └─ 状态：✅ 本报告

③ 注入后输出不变
   └─ 含义：用 R 重算 ≡ 原始计算
   └─ 状态：✅ 即 R3 的数学命题
```

**②③ 合起来 = 回放精确。契约成立。**

---

## 6. 边界与延伸测量

| 项 | 说明 |
|---|---|
| 跨引擎一致性 | 已测：vLLM vs transformers 一致率 91.67%（见 §6.1）。不要求一致，只量化差异 |
| 端到端 RL | 还没接到真实训练框架上跑 |
| 其他模型 | 只测了 Qwen1.5-MoE-A2.7B。其他 MoE 的 router 实现可能不同 |
| 跨卡一致性 | 见 `r3-adapter/determinism/README.md`：GEMM 分界点依 GPU 架构而变，5060 在 M=2、A10 在 M=128 |

### 跨引擎 logits 对比做不到，但路由 ID 可以

一个自然的想法是：把 vLLM 的路由注入 transformers，看 logits 是否更接近 vLLM 的。

**这个做法无法给出结论**，因为两个引擎的 kernel 差异远大于路由差异：

| | vLLM | gptqmodel |
|---|---|---|
| 量化后端 | HUMMING（原生 int4） | TorchAtenLinear |
| attention | paged + flash | 原生 SDPA |

kernel 差异会淹没路由差异，无法分离。

**但路由 ID 本身可以直接对比**（见 §6.1），而且 R3 也不要求不同引擎产生相同路由。

### 6.1 跨引擎路由一致率：91.67%

用 `cross_engine_routes.py` 直接对比两个引擎对同一 prompt 的 prefill 路由。

**两轮测量**（第一次 4 条 prompt，第二次扩到 12 条）：

| 指标 | 第一轮 | 第二轮 |
|---|---|---|
| prompt 数 | 4（代码/SQL/中文/英文） | 12（+医学/数学/诗歌/翻译/金融/ML/Rust） |
| 总槽位 | 960 | 4320 |
| 不一致槽位 | 88 | 360 |
| **一致率** | **90.83%** | **91.67%** |
| 95% 置信区间 | 88.8% – 92.6% | **90.8% – 92.5%** |

两轮差 0.84%，说明首次估计无偏，只是精度不够。**可以下的结论：跨引擎路由一致率约 91.7%，不一致率约 8.3%。**

逐层一致率（每层 720 槽位，单槽位权重 0.14%）：

```
L0  92.22%   L6  92.22%   L12 91.67%   L18 92.22%
L1  93.89%   L7  95.56%   L13 93.33%   L19 89.44%
L2  91.11%   L8  91.11%   L14 94.44%   L20 87.78%
L3  92.22%   L9  92.22%   L15 91.11%   L21 91.67%
L4  94.44%   L10 92.78%   L16 92.78%   L22 90.00%
L5  92.22%   L11 93.89%   L17 90.00%   L23 81.67%
```

#### 深层显著更差

| 层段 | 平均一致率 |
|---|---|
| L0 – L7 | **92.99%** |
| L20 – L23 | **87.78%** |

差 5.2 个百分点，约每层 37 个槽位——**不是采样噪声**（第一轮每层只有 40 槽位时才无法定论）。

L23 单层 **81.67%**，最后一层有近 1/5 的路由不同。

**最可能的解释：误差累积的正反馈。**

```
浅层路由差 8%  →  hidden state 不同  →  深层 router 输入不同  →  深层路由差 18%
                                                                    ↓
                                                              继续放大
```

如果成立，推论是 **R3 在深层的价值高于浅层**——浅层差 5% 影响有限，L23 差 18% 直接改变输出。

但这是推断，未验证。要验证需对比「注入前后 hidden state 的逐层差异曲线」。

#### 这个数意味着什么

| 对比 | 不一致率 |
|---|---|
| 同一引擎，batch 组成变化 | 0.39%（3/768） |
| 同一引擎，开 batch invariant | ~0% |
| **跨引擎** | **8.33%**（360/4320） |

**跨引擎差异比批组成差异大 21 倍。**

所以真实 RL 场景下（训练用 transformers、rollout 用 vLLM），主要矛盾不是批不变性，而是**两个引擎本来就选了不同的专家**。

R3 的价值定位由此明确：

- 91.7% 的路由本来就一致，不用管
- **8.3% 不一致，R3 要修的就是这部分**——用推理侧的路由覆盖训练侧的选择
- 而且这 8.3% 里，深层占比更大（L23 达 18%），所以修复的收益集中在后段

#### 局限

1. **样本小。** 4 个 prompt、960 槽位，只能说明"量级在 10% 附近"。
2. **混了两部分差异。** router logits 的数值差异 + top-k 的边界行为，脚本测不出占比。
   要分离需要控制量化方式（比如让两边都用同一种 int4 kernel，做不到）。
3. **只测 prefill。** decode 阶段的路由（M = batch size）没有覆盖。

---

### 6.2 误差为什么会累积：25 倍放大实测

`hidden_propagation.py` 对比「注入 vLLM 路由前后」每层 hidden state 的相对差异。

两轮测量：

| 轮次 | 样本 | 整体比值 |
|---|---|---|
| 第一轮 | 1 条 prompt / 5 token | 62.4x |
| 第二轮 | 12 条 prompt / 192 token | **25.1x**（单条中位数 23.6x） |

第一轮的 62x 是样本偏差——"The capital of France is" 恰好落在一条放大明显的轨迹上。

第二轮 12 条中，**12/12 条单独看也 >2x**，趋势一致。

平均相对差异（12 条）：

| 层 | 均值 | 中位数 | |
|---|---|---|---|
| L1 | 5.32e-04 | 1.28e-04 | ▏ |
| L5 | 1.56e-03 | 3.99e-04 | ▎ |
| L6 | 1.36e-02 | 2.90e-03 | ██████████ |
| L17 | 1.25e-02 | 4.69e-03 | █████████ |
| L19 | 1.65e-02 | 4.85e-03 | ████████████ |
| L21 | 1.19e-02 | 8.85e-03 | █████████ |
| L23 | **3.95e-02** | **3.72e-02** | ██████████████████████████████ |

#### 放大不是均匀的，是阶梯状

相邻层跳变最大的三处：

| 跳变 | 倍数 |
|---|---|
| L5 → L6 | **8.7x** |
| L22 → L23 | 3.7x |
| L4 → L5 | 2.7x |

**L5→L6 一跳就占了总放大的三分之一。**

曲线中间还有平台区（L7–L17 基本持平在 1.3e-2），说明不是「每层等比放大」，
而是「特定几层剧烈变化 + 中间平台」。

#### 架构含义

> **R3 必须全层修。** 浅层差 1e-4，到 L23 被放大到 4e-2。

这否掉了「只记录深层省带宽」的优化——如果浅层不修，深层的差异里有一部分是
浅层放大出来的，不修浅层等于没修。

**R3 不是可以按层裁剪的优化，是必须全局一致的契约。**

如果 L5→L6 这个放大器稳定存在，它还有一个额外含义：那两层的 router 对输入扰动
特别敏感，可能需要单独的精度保障。但这需要更多样本确认。

#### 两个必须说明的地方

**① L0 被排除在统计外。**

L0 的 `max|h|` 是 2.98，L1 是 76——**差 25 倍**。同样的绝对差异在 L0 显得大 25 倍，
所以「相对差异」这个指标在 L0 上虚高，跨层比较不可用。

排除 L0 后整体比值从 16.0x 修正为 25.1x。这不是挑数据，是指标在那一层失效。

**② 第一轮的单条 prompt 得到 62x，第二轮 12 条得到 25x。**

差异来自样本选择，不是方法改变。**25x 更可信**，因为它有 12 条的一致支撑。

## 7. 复现

### 7.1 小模型（10 秒，无需 GPU）

```bash
cd r3-adapter
python inject_selftest.py
```

### 7.2 真实模型（CPU，约 60 秒）

```bash
cd r3-adapter
python inject_real.py --prompt "The capital of France is" --layers 0
```

依赖 `gptqmodel`（不要用 `transformers` 的 GPTQ 加载，见下节）。

### 7.3 环境要求

```
torch        2.14.0+cu130
transformers 5.16.1
gptqmodel    7.10.0
vllm         0.33.0（仅第 ① 条需要）
```

**transformers 5.x + optimum 2.x 的 GPTQ 加载是坏的**：

```
NameError: name 'QuantizeConfig' is not defined
  at optimum/gptq/quantizer.py:163
```

`transformers` 仍在调 `optimum.gptq.GPTQQuantizer.from_dict()`，但 optimum 2.x 已移除该类。

绕过方式：用 `gptqmodel.GPTQModel.load()`，它自己实现量化加载，不走 `transformers.quantizers`。

---

## 8. 工程事故记录

### 8.1 pip 装包会静默破坏 vLLM

`pip install gptqmodel` 把 `torch` 从 2.14.0 升到 2.15.0、`transformers` 从 5.16.1 升到 5.23.0，导致 vLLM 不可用。pip 只给了 warning：

```
vllm 0.33.0 requires torch==2.14.0, but you have torch 2.15.0 which is incompatible
```

**这是「静默降级」的另一种形态**：发生在装包阶段，不是启动阶段。恢复方式是降回原版本 + 卸载不匹配的 `torchaudio`。

**教训**：装任何会碰核心依赖的包，先建 venv。

### 8.2 对比测试只能改一个变量

第一版 `attention_gate.py` 测 chunked prefill 时，基准用 `is_causal=True`、分块用显式 mask，同时改了两个变量，得出「chunked 不安全」的结论。统一 mask 写法后重测，完全一致。

**得到不可归因的结果，比得到错误结果更浪费时间。**

### 8.3 采样方式退化会伪造出「规律」

`cloud_probe.py` 的 capacity 统计用「同一 prompt 的连续 token」模拟 prefill。但采集数据时 prompt 最长只有 108 token，`M ≥ 128` 时采样退化成「跨 prompt 拼接」，与 decode 模式同质。

测出来的「M=128 时 prefill/decode 比值收敛到 1.0」是**采样方式的产物，不是真实规律**。

脚本已加 `degraded` 标记，超过最长 prompt 的行会打 ⚠️。

---

## 9. 下一步

回放证明完成了研究部分。剩下的是工程：

1. **补多样本**——§6.1 是 12 条 prompt / 4320 槽位，可信；§6.2 只有 1 条 prompt / 5 token，
   趋势方向可信但 62 倍这个具体值不可信。需要跑到 10+ 条 prompt。
2. **接真实 RL 框架**（Megatron / NeMo-RL / ROLL），跑一次端到端 rollout + 单步回放
3. **`compare_routes()` 上线**，把「契约是否成立」变成「不一致率是多少」的可监控指标
4. **其他模型验证**——不同 MoE 的 router 实现差异（比如 DeepSeek 的 `norm_topk_prob` 默认值可能不同）

第 2 条需要训练侧环境（多卡 + 训练框架），不是这个仓库能覆盖的。

---

*报告生成于 2026-11-14，数据来自实测，非文档摘录。*
