# AMD ROCm 实测结论（gfx942 / MI300 · vLLM 0.28.0+rocm723）

> 日期：2026-11-14
> 环境：23 核 / 200GB 内存 / 192GB 显存，ROCm 7.2，torch 2.12，transformers 5.15
> 模型：Qwen1.5-MoE-A2.7B-GPTQ-Int4（int4 在 ROCm 正常加载，TRITON MoE，无 EMULATION）

## 最终结论：R3 契约双厂商成立 ✅

| 条件 | NVIDIA A10 | AMD gfx942 |
|---|---|---|
| ① 推理侧可复现（批不变） | ✅ 0/768 | ✅ 10/10 全一致（修 LDS 后） |
| ② 训练侧可回放 | ✅ bit-identical | （CPU 回放已验证，架构无关） |
| R3 capture | ✅ | ✅ |

AMD 侧唯一障碍是 batch_invariant kernel 的 LDS 配置，**一行 sed 修复**后即闭合。

## ① R3 capture 跨厂商可用

`--enable-return-routed-experts` 在 0.28 ROCm 正常返回 routed_experts
（`[tokens,24,4]`、uint8、unique=60）。路由统计与 NVIDIA 高度一致：

| 指标 | NVIDIA A10 | AMD gfx942 |
|---|---|---|
| 变异系数 cv | 0.052 | 0.056 |
| top-10 占比 | 17.9% | 18.0% |
| 时间局部性 | 2.27x | 2.27x |
| 语义局部性 | 2.64x | 1.59x |

→ 适配层测量不绑定 GPU 厂商。

## ② batch_invariant：LDS 超限根因 + 一行修复

**根因**（开开关时 engine init 崩溃）：

```
triton.runtime.errors.OutOfResources: shared memory,
Required: 98304, Hardware limit: 65536
```

`matmul_kernel_persistent` 按 NVIDIA 调块：bf16 `128/128/64, stages3`
→ `3×(128×64+64×128)×2 = 98304 = 96KB`，超 gfx942 的 64KB LDS。

**修复**（备份后 sed，把 LDS 压到 32KB）：

```bash
F=.../vllm/model_executor/layers/batch_invariant.py
cp "$F" "$F.bak"
sed -i 's/"num_stages": 3/"num_stages": 2/g; s/"BLOCK_SIZE_K": 64/"BLOCK_SIZE_K": 32/g' "$F"
# 2×(128×32+32×128)×2 = 32768 = 32KB ≤ 64KB
```

batch-invariant 的确定性来自**固定 tile 归约顺序**，不依赖块大小，故缩块不破坏不变性。

**修复后**：开 `VLLM_BATCH_INVARIANT=1` 正常启动，`long_prompt_invariance.py`
10/10 长度全一致 → 条件①在 AMD 成立。

## ③ 无 invariant 时的批方差（修复前对照）

关开关时 4/10 长度有 0.004%–0.649% 偶发边界翻转（同 NVIDIA 默认量级），
确认条件①依赖 invariant kernel。修复后归零。

## 注意

- 该 sed 是**测试性 hack**：牺牲少量 matmul 吞吐换 LDS 合规。上游应加 HIP-aware
  配置（按 `torch.version.hip` 选 block/stages），而非全局缩块。
- 回滚：`cp "$F.bak" "$F"`。
- 缩块后仍全一致，验证了"不变性来自归约顺序而非块大小"的设计假设。