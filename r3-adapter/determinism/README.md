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

## 实测结论（RTX 5060 Laptop 8GB / torch 2.14.0+cu130 / bf16）

| 层 | 裂缝 | 影响面 | 约束 |
|---|---|---|---|
| GEMM | `M=1` vs `M≥2`（gemv / gemm） | 0.1%（6/5632） | decode `M ≥ 2` |
| Attention | 四个后端互不一致 | 23%（475/2048） | 启动时锁定一个 |
| Attention | `is_causal` vs 显式 mask | 6.6% | 统一写法 |
| Attention | KV 总长度 `S` 变化 | 23% | 全局固定 `S` |
| MoE | combine 顺序（slot vs expert） | 17%（174/1024） | 按 slot 累加 |
| MoE | capacity 溢出 | 静默丢 token | `C` 给足（≥2.5× 期望） |

**反直觉的一点**：GEMM 最轻，attention 和 MoE 才是主战场。
分叉越多，裂缝越多。

## 统一原则

> 不变量不是「固定 shape」，是「固定代码路径」。

同一段数学存在多套实现时，实现之间的浮点归约顺序不同：

- GEMM：gemv / 小 tile / 大 tile
- Attention：flash / efficient / math / cudnn
- Mask：`is_causal` / 显式矩阵
- MoE：slot 顺序 / expert 顺序

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
