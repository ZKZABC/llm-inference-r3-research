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

## 6. 这个证明不能覆盖什么

| 项 | 说明 |
|---|---|
| 跨引擎一致性 | 已测：vLLM vs transformers 一致率 90.83%（见 §6.1）。不要求一致，只量化差异 |
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

### 6.1 跨引擎路由一致率：90.83%

用 `cross_engine_routes.py` 直接对比两个引擎对同一 prompt 的 prefill 路由：

| 指标 | 值 |
|---|---|
| prompt 数 | 4（中英混合，代码/SQL/自然语言） |
| 总槽位 | 960（token × 24 层 × 4 专家） |
| 不一致槽位 | 88 |
| **一致率** | **90.83%** |
| 95% 置信区间 | 约 88.8% – 92.6% |

逐层一致率（样本每层 40 槽位，波动较大）：

```
L0  95.0%   L6  95.0%   L12 90.0%   L18 92.5%
L1  95.0%   L7  95.0%   L13 90.0%   L19 90.0%
L2  87.5%   L8  85.0%   L14 95.0%   L20 85.0%
L3  85.0%   L9  95.0%   L15 92.5%   L21 80.0%
L4  90.0%   L10 95.0%   L16 100.0%  L22 90.0%
L5  92.5%   L11 92.5%   L17 92.5%   L23 80.0%
```

**深层似乎略差**：L20–L23 平均 83.75%，L0–L7 平均 92.2%。
但每层只有 40 个样本，2–3 个槽位就能造成 5% 的波动，**这个趋势还不能定论**。

#### 这个数意味着什么

| 对比 | 不一致率 |
|---|---|
| 同一引擎，batch 组成变化 | 0.39%（3/768） |
| 同一引擎，开 batch invariant | ~0% |
| **跨引擎** | **9.17%** |

**跨引擎差异比批组成差异大 23 倍。**

所以真实 RL 场景下（训练用 transformers、rollout 用 vLLM），主要矛盾不是批不变性，而是**两个引擎本来就选了不同的专家**。

R3 的价值定位由此明确：

- 91% 的路由本来就一致，不用管
- **9% 不一致，R3 要修的就是这部分**——用推理侧的路由覆盖训练侧的选择

#### 局限

1. **样本小。** 4 个 prompt、960 槽位，只能说明"量级在 10% 附近"。
2. **混了两部分差异。** router logits 的数值差异 + top-k 的边界行为，脚本测不出占比。
   要分离需要控制量化方式（比如让两边都用同一种 int4 kernel，做不到）。
3. **只测 prefill。** decode 阶段的路由（M = batch size）没有覆盖。

---

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

1. **接真实 RL 框架**（Megatron / NeMo-RL / ROLL），跑一次端到端 rollout + 单步回放
2. **`compare_routes()` 上线**，把「契约是否成立」变成「不一致率是多少」的可监控指标
3. **其他模型验证**——不同 MoE 的 router 实现差异（比如 DeepSeek 的 `norm_topk_prob` 默认值可能不同）

第 1 条需要训练侧环境（多卡 + 训练框架），不是这个仓库能覆盖的。

---

*报告生成于 2026-11-14，数据来自实测，非文档摘录。*
