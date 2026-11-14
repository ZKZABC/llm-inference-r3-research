# AMD ROCm 实测结论（gfx942 / MI300 · vLLM 0.28.0+rocm723）

> 日期：2026-11-14
> 环境：23 核 / 200GB 内存 / 192GB 显存，ROCm 7.2，torch 2.12，transformers 5.15
> 模型：Qwen1.5-MoE-A2.7B-GPTQ-Int4（NFS 共享，int4 在 ROCm 正常加载，TRITON MoE，无 EMULATION）

## 三个结论

### ① R3 capture 跨厂商可用 ✅

`--enable-return-routed-experts` 在 0.28 ROCm 上正常返回 routed_experts
（shape `[tokens,24,4]`、uint8、unique=60）。路由统计与 NVIDIA A10 高度一致：

| 指标 | NVIDIA A10 | AMD gfx942 |
|---|---|---|
| 变异系数 cv | 0.052 | 0.056 |
| top-10 占比 | 17.9% | 18.0% |
| 时间局部性 | 2.27x | 2.27x |
| 语义局部性 | 2.64x | 1.59x |

→ 适配层的测量不绑定 GPU 厂商。

### ② batch_invariant 崩溃根因 = LDS 超限（可调，非缺实现）❌→🔧

开 `VLLM_BATCH_INVARIANT=1` 时 engine init 崩溃，根因：

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536.
Reducing block sizes or `num_stages` may help.
```

路径：`linear_batch_invariant → matmul_persistent → matmul_kernel_persistent`。
持久化 matmul kernel 按 NVIDIA 的 shared memory 调块（要 96KB），
gfx942 每 workgroup 只有 64KB LDS → 超 1.5 倍。

**这是 kernel 配置问题，不是 ROCm 缺实现。** 减小 BLOCK / num_stages 使其 ≤64KB 即可。

### ③ 无 invariant 时批方差 = 偶发边界翻转（与 NVIDIA 默认同量级）

关掉开关后 `long_prompt_invariance.py`：4/10 长度分歧，0.004%–0.649%，
属偶发 top-k 边界翻转，非系统性 kernel 分叉（对照裸 GEMM 分叉 35.9%）。
→ 条件①（可复现）依赖 invariant kernel；该 kernel 在 AMD 需先修 LDS。

## 判定表落地

| 项 | 结果 |
|---|---|
| int4 加载 ROCm | ✅ TRITON，无 EMULATION |
| R3 capture | ✅ |
| batch_invariant（默认配置） | ❌ LDS OOM |
| batch_invariant（修 LDS 后） | 待验证 |
| 无 invariant 批方差 | 0.004–0.65%（偶发） |

## 下一步（可选）

修 `batch_invariant.py` 的 `matmul_kernel_persistent`：在 HIP 下减小
`num_stages`/`BLOCK_K` 使 shared ≤64KB，再开 `VLLM_BATCH_INVARIANT=1`
复跑 `long_prompt_invariance.py`，若全 PASS 则条件①在 AMD 也成立。

检查 kernel 配置：
```bash
grep -n "num_stages\|BLOCK_SIZE\|def matmul_persistent" \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/batch_invariant.py
```