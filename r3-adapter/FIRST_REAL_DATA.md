# 第一个真实样本 —— R3 契约在真实环境成立

> 2026-11-14 15:00，ModelScope 免费 AMD GPU 实例。
> 这份文档记录的是**第一条从真实 vLLM 服务拿到的 `routed_experts` 数据**，
> 以及它可以推翻 / 证实哪些纸面假设。

## 环境

| 项目 | 值 |
|---|---|
| 平台 | ModelScope Notebook（魔搭免费实例） |
| 硬件 | AMD GPU，**192 GB 显存**，23 核 / 200 GB 内存 |
| 计算栈 | **ROCm 7.2.3**（不是 CUDA） |
| vLLM | 0.33.0（ROCm wheel） |
| 模型 | `qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`（7.84 GiB） |
| 启动方式 | `vllm serve`（HTTP API，非 Python SDK） |

## 复现命令

```bash
# 终端 A：起服务
vllm serve /mnt/workspace/model \
  --enable-return-routed-experts \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# 终端 B：请求
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/workspace/model","prompt":"Hi","max_tokens":8,"temperature":0}' \
  -o resp.json
```

## 实测结果

```python
shape  = (8, 24, 4)
dtype  = uint8
range  = 0 ~ 59
unique = 60
token0-layer0 = [33, 52, 7, 54]
```

npy 头部（base64 解码后）：

```python
{'descr': '|u1', 'fortran_order': False, 'shape': (8, 24, 4)}
```

## 逐项对契约

| 契约条款 | 实测 | 结论 |
|---|---|---|
| `shape = [num_tokens - 1, num_moe_layers, top_k]` | `(8, 24, 4)`，8 = 1(prompt) + 8(gen) − 1 | ✅ 末 token 无路由成立 |
| dtype：≤256 专家用 uint8 | `\|u1` = uint8，60 专家 | ✅ |
| 专家 ID ∈ [0, num_experts) 或 −1 | range `0~59`，60 专家，无 −1 | ✅ 无 unmappable |
| 行序 = `prompt_token_ids + output_token_ids` 的位置 | 逐行对齐 | ✅ |

## 三个被这次实测修正的判断

### 1. `unique = 60` —— 路由是分散的，不是集中的

8 token × 24 层 × top-4 = **768 个专家实例**，覆盖全部 **60** 个专家，平均 12.8 次/专家。

**这意味着：**
- Qwen1.5-MoE 没有"专家塌缩"，负载均衡良好
- **Router Replay 不能依赖"热门/冷门专家"做缓存分层**——全部 60 个都必须可用
- 也解释了为什么 CPU offload 在 MoE 上收益有限：**访问模式均匀，没有局部性**

### 2. ROCm 上也能拿到 `routed_experts`（推翻了此前预判）

启动日志里有这一行：

```
WARNING: Model Runner V2 does not yet support routed experts capture;
using the V1 model runner instead.
```

**读起来像"功能缺失"，实际含义相反**：V2 还没接这个功能，所以回退到 **V1——而 V1 是功能完整的实现**。

> **教训：警告文案 ≠ 实际行为。** "using V1 instead" 是切换到可用实现的兜底，不是降级。

### 3. `spawn` 模式下不能用 `python - << EOF`

```
FileNotFoundError: '/mnt/workspace/<stdin>'
```

子进程启动时要 `runpy.run_path(主模块路径)` 重新执行，`<stdin>` 不是真实路径。
**必须写成文件，并且加 `if __name__ == "__main__":` 守卫。**

顺带一个坑：通过终端 heredoc 粘贴 Python 代码时，**零宽空格（`&#8203;`）和缩进丢失**会静默破坏文件。
更可靠的方式是：JupyterLab 图形界面建文件，或 `base64 -d`。

## 还不能证明什么

这份数据只验证了**"拿到路由"**这一步。真正的 R3 闭环还差：

1. **训练侧回放**：用这份 `routed_experts` 在 transformers/Megatron 前向里强制指定专家
2. **logprob 归零**：回放后与原推理的 logprob 差异应为 0

第 1、2 步在 CPU 上就能做（`mini_moe_r3.py` 已在微型 MoE 上验证数学命题，差异 `0.000e+00`）。
缺的是"真模型 + 真路由"的组合验证。
