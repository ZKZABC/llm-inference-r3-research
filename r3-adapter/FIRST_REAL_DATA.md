# 第一个真实样本 —— R3 契约在真实环境成立

> 2026-11-14 15:00，ModelScope 免费 AMD GPU 实例。
> 这份文档记录的是**第一条从真实 vLLM 服务拿到的 `routed_experts` 数据**，
> 以及它可以推翻 / 证实哪些纸面假设。

## 环境

| 项目 | 值 |
|---|---|
| 平台 | ModelScope Notebook（魔搭免费 GPU 实例，**36 小时额度**） |
| 硬件 | NVIDIA GPU，**24 GB 显存**，8 核 / 32 GB 内存 |
| 计算栈 | **CUDA 13.0**（torch 2.14.0+cu130） |
| vLLM | 0.33.0（PyPI 官方 wheel，非 ROCm 版） |
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

### 2. 官方文档说"必须 V2 runner"，实测 V1 也能工作

启动日志里有这一行：

```
WARNING: Model Runner V2 does not yet support routed experts capture;
using the V1 model runner instead.
```

而 vLLM 官方文档 `docs/training/routed_experts.md` 写的是：

> Startup fails with a `ValueError` **unless Model Runner V2 is used**

**实测结果与文档相反**：vLLM 自己回退到 V1，`routed_experts` 正常返回，数据完整且符合契约。

可能的解释：

- 文档描述的是 V2 完成后的目标状态，当前 V1 是唯一可用路径
- 或者 `VLLM_USE_V2_MODEL_RUNNER=1` 的强制校验只在某些分支/版本生效
- 或者两者并存：V2 未接入时静默回退，而不是报错

> **教训：文档里的"必须"和代码的实际行为，是两件需要分别验证的事。**
> 这个案例里，如果只信文档，会得出"我的环境跑不了"的错误结论；
> 如果只信日志警告，会得出"功能缺失"的错误结论。**只有实际跑一遍才知道。**

### 3. `spawn` 模式下不能用 `python - << EOF`

```
FileNotFoundError: '/mnt/workspace/<stdin>'
```

子进程启动时要 `runpy.run_path(主模块路径)` 重新执行，`<stdin>` 不是真实路径。
**必须写成文件，并且加 `if __name__ == "__main__":` 守卫。**

顺带一个坑：通过终端 heredoc 粘贴 Python 代码时，**零宽空格（`&#8203;`）和缩进丢失**会静默破坏文件。
更可靠的方式是：JupyterLab 图形界面建文件，或 `base64 -d`。

## 批不变性实验（本次最重要的发现）

R3 的全部价值建立在「记录一次，就能精确重放」上。但 RL rollout 的 batch 是**动态拼的**——
同一条 prompt 可能单独跑，也可能和另外 7 条一起跑。如果 batch 组成一变、路由就变，
那么"记录的路由"和"训练时要回放的路由"天然对不上。

### 实验设计

对比同一条 prompt（`"Hi"`）在两种 batch 组成下的路由：

- **A**：单独跑 `["Hi"]`
- **B**：`["Hi"]` 和 `["Hello there"]` 一起跑（batch=2），取第一条的 `routed_experts`

```bash
# A
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"/mnt/workspace/model","prompt":"Hi","max_tokens":8,"temperature":0,"repetition_penalty":1.0}' \
  -o bi_single.json

# B
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"/mnt/workspace/model","prompt":["Hi","Hello there"],"max_tokens":8,"temperature":0,"repetition_penalty":1.0}' \
  -o bi_batch2.json
```

### 结果

| 配置 | identical | diff |
|---|---|---|
| **默认**（未开 batch invariant） | False | **3 / 768** |
| **`VLLM_BATCH_INVARIANT=1`** | True | **0 / 768** |

**结论：R3 可行，但必须在 rollout 侧强制开启 `VLLM_BATCH_INVARIANT=1`，否则记录的路由无法复现。**

### 硬证据：开关真的生效了

开启后启动日志出现：

```
Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'batch_invariant.py:293'
This call originates from:
  File "vllm/model_executor/determinism/batch_invariant.py", line 293, in matmul_persistent
    NUM_SMS = num_compute_units(a.device.index)
```

vLLM 确实加载了 `determinism/batch_invariant.py` 并启用 `matmul_persistent`
（确定性矩阵乘实现），说明这个开关**覆盖到了 MoE 的路由计算路径**，不只是 attention。

### 代价：初始化慢 13 倍

| 阶段 | 默认 | 开 batch invariant |
|---|---|---|
| 模型加载 | 10.4 s | 10.4 s |
| torch.compile | 不触发 | **32.1 s** |
| **引擎初始化总计** | **~10 s** | **130.1 s** |

这是**一次性的启动成本**（编译结果有缓存），不影响单次推理延迟。
但如果你要频繁重启服务（比如调参），这个税会累积。

### 三个必须显式设置的前提

跑通 R3 需要的完整前置条件，缺一不可：

```bash
VLLM_BATCH_INVARIANT=1 vllm serve <model> \
  --enable-return-routed-experts      # ① 打开路由捕获
  --max-model-len 512 \
  --gpu-memory-utilization 0.9
```

```json
// ② 请求里显式指定，排除模型自带 generation_config 的干扰
{"temperature": 0, "repetition_penalty": 1.0}
```

| # | 条件 | 不满足会怎样 |
|---|---|---|
| ① | `--enable-return-routed-experts` | `routed_experts` 为 `None` |
| ② | `VLLM_BATCH_INVARIANT=1` | **batch 组成一变，路由就翻转（实测 3/768）** |
| ③ | `repetition_penalty=1.0` | 模型自带的 1.05 会改变 logits，间接影响专家选择 |

**第 ② 条是文档里最不显眼、但最致命的一条。** 它不在 `routed_experts.md` 的"Requirements"里，
只出现在 batch invariant 相关的文档角落。少了它，契约的 shape/dtype/合法性全部通过，
但数据在真实场景下不可复现。



## 还不能证明什么

这份数据只验证了**"拿到路由"**这一步。真正的 R3 闭环还差：

1. **训练侧回放**：用这份 `routed_experts` 在 transformers/Megatron 前向里强制指定专家
2. **logprob 归零**：回放后与原推理的 logprob 差异应为 0

第 1、2 步在 CPU 上就能做（`mini_moe_r3.py` 已在微型 MoE 上验证数学命题，差异 `0.000e+00`）。
缺的是"真模型 + 真路由"的组合验证。
