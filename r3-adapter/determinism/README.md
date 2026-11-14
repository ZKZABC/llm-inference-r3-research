# determinism/ —— 第 4 类校验：可复现性

前三类校验回答「能不能跑」「格式对不对」「有没有静默降级」。
这一类回答一个问题：**拿到的路由数据，换一个 batch 组成还成立吗？**

## 为什么需要单独一类

`r3_contract.py` 校验 shape / dtype / ID 范围 / 末 token 无路由。
这些检查对 `3/768` 那种翻转**完全无感**——shape 对、dtype 对、ID 合法，
但数据在真实 rollout 场景下不可复现。

这三层的差别不是精度问题，是**代码路径**问题。

## 四个脚本

| 脚本 | 测什么 | 需要 GPU |
|---|---|---|
| `gemm_gate.py` | GEMM 的算法路径分叉（gemv / 小 tile / 大 tile） | 建议用，CPU 结论不可直接搬 |
| `attention_gate.py` | SDPA 后端、mask 写法、KV 总长度 | 是（CPU 上后端检查会 SKIP） |
| `moe_gate.py` | combine 顺序、capacity 溢出 | 否（纯 PyTorch 逻辑模拟） |
| `skeleton_moe.py` | MoE 实现与 HF 逐层对齐 | 否（零下载，现场构造小模型） |

## 实测结论（bf16，两张卡对照）

| 层 | 裂缝 | 影响面 | 约束 |
|---|---|---|---|
| GEMM | 路径分界点**依架构而变** | 0.1% ~ 36% | 见下方对照表 |
| Attention | 四个后端互不一致 | 23%（475/2048） | 启动时锁定一个 |
| Attention | `is_causal` vs 显式 mask | 6.6% | 统一写法 |
| Attention | KV 总长度 `S` 变化 | 23% | 全局固定 `S` |
| MoE | combine 顺序（slot vs expert） | 17%（174/1024） | 按 slot 累加 |
| MoE | capacity 溢出 | 静默丢 token | `C` 按 M 定，见下 |

### GEMM 的分界点不可跨卡迁移

同一份 `gemm_gate.py`，两张卡的结果：

| 检查 | RTX 5060 Laptop（Blackwell sm_120） | A10（Ampere sm_86） |
|---|---|---|
| `M=1 vs M=2` | **FAIL**（6/5632） | **PASS** |
| `M=1 vs M=64` | FAIL | **PASS** |
| `M=128 vs M=256` | PASS | **FAIL**（35.9%，max=0.0312） |
| `M=128 vs M=130` | PASS | PASS |

**分界点从 `M=2` 挪到 `M=128`。**

A10 上 `M≥128` 的表现：

| 对照 | 元素差异 | max | 判断 |
|---|---|---|---|
| `M=128 vs M=256` | 35.9% | 0.0312（4 ulp） | 不同 kernel |
| `M=128 vs M=200` | 26.8% | 0.0312 | 不同 kernel |
| `M=128 vs M=512` | 35.9% | 0.0312 | 不同 kernel |
| `M=128 vs M=300` | 0.1% | 0.0156（2 ulp） | 同 kernel 边界效应 |

至少 4 条路径：`M≤64` / `M=128` / `M=200` / `M≥256`。

**推论**：

1. **不能跨卡搬结论。** 5060 上“decode 需要 `M≥2`”，A10 上 M≤64 全部一致，这条约束不存在。
2. **A10 上危险的是 prefill 而非 decode。** decode 的 M = batch size（通常 ≤64，安全）；
   prefill 的 M = token 数（长 prompt 动辄几百，跨分界）。
3. **短 prompt 的实验会漏掉真正的问题。** `"Hi"` + `"Hello there"` 的 M 在 1~4，
   恰好落在 A10 的安全区——这时候测出的“批不变性成立”是假象。

每台新机器都必须跑一次 `gemm_gate.py`，不能假设分界点。

**反直觉的一点**：GEMM 不一定最轻。在 A10 上它的影响面（36%）超过了 attention（23%）。

### capacity 的实测值（Qwen1.5-MoE-A2.7B / 60 专家 top-4 / 818 token）

prefill 与 decode 的采样方式不同：prefill 是同一 prompt 的连续 token（语义相关），
decode 是 M 个请求各出 1 个 token（语义无关）。

| M | 期望/专家 | prefill C | decode C | prefill 相对冗余 |
|---|---|---|---|---|
| 8 | 0.53 | **10** | 6 | 19x |
| 16 | 1.07 | **15** | 7 | 14x |
| 32 | 2.13 | **21** | 13 | 9.9x |
| 64 | 4.27 | 23 | 23 | 5.4x |
| 128 | 8.53 | ⚠️ 49 | 34 | 5.7x |
| 256 | 17.07 | ⚠️ 72 | 63 | 4.2x |
| 512 | 34.13 | ⚠️ 118 | 118 | 3.5x |

**⚠️ 标记的行不可信。** 采集时 prompt 最长只有 108 token，`M ≥ 128` 时
「同一 prompt 的连续 token」已经退化成了「跨 prompt 拼接」，与 decode 模式同质。
M=128 起 prefill/decode 比值收敛到 1.0 是采样方式的产物，不是真实规律。

可信区间是 **M ≤ 64**，要拿大 batch 的 prefill 数据需要生成更长的序列（`--n-gen 256`）。

**倍数不能当 factor 用。** M=8 时峰值/均值 = 15x，但那是分母只有 0.53 造成的；
按绝对值定 `C` 才可靠。

**`C` 必须随 M 变**，不能用一个常数倍数：
`C = ceil(2.5 × M × top_k / E)` 在 M=8 时需要 1.3，实测要 10；
M=512 时需要 85，实测 118。两头都不对。

### 小 batch 的效率问题

| M | C | 期望 | 浪费比例 |
|---|---|---|---|
| 8 | 10 | 0.53 | **95%** |
| 32 | 21 | 2.13 | 90% |
| 512 | 118 | 34.13 | 71% |

小 batch 时 capacity 的相对效率极低（95% 的槽位是 padding）。
但绝对浪费小（10 个槽位 vs 118 个），所以**代价可承受、效率不可接受**。

这解释了为什么 capacity + 固定趟数的方案适合 RL rollout（batch 大），
不适合低延迟 serving（batch 小）——后者应该用别的策略。

## 统一原则

> 不变量不是「固定 shape」，是「固定代码路径」。

同一段数学存在多套实现时，实现之间的浮点归约顺序不同：

- GEMM：gemv / 小 tile / 大 tile
- Attention：flash / efficient / math / cudnn
- Mask：`is_causal` / 显式矩阵
- MoE：slot 顺序 / expert 顺序

而**「哪几条路径存在、分界在哪」由 GPU 架构决定**（cuBLAS 按 SM 数、tile 尺寸、
寄存器压力挑 kernel）。所以约束必须实测，不能从别的机器抄。

处置手段只有一个：**把运行时 dispatch 变成启动时配置。**

## 用法

```bash
# GEMM：跨 M 与跨位置的 bit 级一致性
python gemm_gate.py --both --dtype bf16

# Attention：后端 / mask / padding
python attention_gate.py --dtype bf16 --seq 100

# MoE：combine 顺序与 capacity
python moe_gate.py --dtype bf16 --batch 4

# 骨架：MoE 实现与 HF 对齐（需 transformers）
python skeleton_moe.py --layers 2
```

## 已知局限

1. **结论是 dtype-specific 的。** bf16 下 `M≥2` 全通，fp32 下连 `M=128 vs M=256` 都 FAIL（96% 元素不同）。换精度必须重测。
2. **`moe_gate.py` 测的是逻辑层。** 真实 grouped GEMM kernel 的 tile 效应需要另外验证。
3. **`gemm_gate.py` 的分界点依平台而变。** cuBLAS 在 `M=1|2` 分叉，MKL 在 `M=3|4` 分叉，结构一样但数值不同。

## 一个反复踩的坑

对比测试**只能改一个变量**。

第一版 `attention_gate.py` 测 chunked prefill 时，基准用 `is_causal=True`、
分块用显式 mask，同时改了两个变量，得出「chunked 不安全」的错误结论。
统一 mask 写法后重测，chunked 完全一致。

**得到不可归因的结果，比得到错误结果更浪费时间。**
