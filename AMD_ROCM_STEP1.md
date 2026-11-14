# AMD ROCm 第一步验证：R3 的两个前提

> **这不是 RL 训练，是「值不值得往下做」的判定。**
>
> 目标：确认 `--enable-return-routed-experts` 和 `VLLM_BATCH_INVARIANT=1`
> 在 AMD ROCm 上是否成立。这两项是今天所有结论的前提，只在 NVIDIA A10 上验证过。
>
> 预计 30–90 分钟。通过后再考虑装 verl 跑端到端。

---

## 为什么必须先做这一步

今天在 NVIDIA 上建立的完整链条：

```
条件① 可复现      0/768（依赖 VLLM_BATCH_INVARIANT=1）
条件② 可回放      bit-identical
跨引擎差异        91.67%
误差传播          25 倍
```

**ROCm 是另一套 kernel**（hip / AITER / Triton），`matmul_persistent` 是 CUDA 的
确定性实现，ROCm 上是否有等价物未知。`routed_experts` 的捕获逻辑在 hip 路径上
是否走得通，也未知。

如果这两项不成立，装 verl 跑 RL 也拿不到可用的 R3 数据——**等于裸跑**。

---

## 阶段 0：环境体检（不装任何东西）

先看清镜像里已经有什么，**不要一上来就 pip install**。

```bash
# 硬件
rocminfo | grep -E "Name:|Marketing Name" | head -8
rocm-smi --showmeminfo vram 2>/dev/null || rocm-smi

# 已装的关键包
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
pip list 2>/dev/null | grep -Ei "^(vllm|transformers|optimum|gptqmodel|torch|triton|numpy) "

# 模型在不在（新实例可能不共享 /mnt/workspace）
ls -lh /mnt/workspace/model 2>/dev/null | head -3 || echo "模型目录不存在，需要重新下载"
```

**记下这几个值**，后面判断要用：

| 项 | 用途 |
|---|---|
| GPU 型号 / gfx 架构 | verl 的 ROCm 支持主要验证 gfx942 (MI300) / gfx950 (MI350) |
| `torch.version.hip` | 确认是 ROCm 版 torch，不是 CUDA 版 |
| vLLM 是否预装、版本 | 镜像名末尾的 `1.40.1` 含义不明，实测确认 |

### 如果模型不在

```bash
# ModelScope 下载（约 8 GB，通常几分钟）
pip install -q modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4', cache_dir='/mnt/workspace/model')
"
```

下载完确认 `config.json` 在目录里，且 `quant_method` 是 `gptq`。

---

## 阶段 1：装包前检查（今天的事故教训）

**任何 `pip install` 之前先跑这个。** 今天 `pip install gptqmodel` 把 torch 2.14→2.15、
transformers 5.16→5.23，vLLM 直接不可用，而 pip 只给 warning。

```bash
cd /mnt/workspace/llm-inference-r3-research/r3-adapter

# 用 --dry-run 看它会动什么，再用 --pip-plan 拦截
pip install --dry-run vllm 2>&1 | tee /tmp/pip_dry.txt
python silent_degradation.py --pip-plan /tmp/pip_dry.txt
```

**如果输出 `CRITICAL: 这次安装会破坏 torch/transformers/numpy`** → 不要装，改用
隔离 venv 或官方 ROCm Docker 镜像。

同理适用于后面装任何包（包括 `modelscope`、`gptqmodel`）。

---

## 阶段 2：装 / 确认 vLLM（ROCm 版）

### 情况 A：镜像里已预装 vLLM

```bash
python -c "import vllm; print('vllm', vllm.__version__)"
vllm --version
```

如果已有且能 import，跳到阶段 3。

### 情况 B：需要安装

**优先用官方 ROCm 镜像**（避免源码编译，ROCm 上编译 vLLM 很慢且容易失败）：

```bash
docker pull vllm/vllm-openai-rocm:latest
```

如果实例不支持 Docker，再从源码/PyPI 装，但**务必先在 venv 里**：

```bash
python -m venv ~/rocm-vllm && source ~/rocm-vllm/bin/activate
# 先 dry-run 检查，再装
pip install --dry-run vllm 2>&1 | tee /tmp/pip_dry.txt
python /mnt/workspace/llm-inference-r3-research/r3-adapter/silent_degradation.py --pip-plan /tmp/pip_dry.txt
pip install vllm
```

> ⚠️ 不要用 `pip install -U` 升级预装包。DSW 镜像是调好版本的组合，
> 动 torch / transformers / numpy 任一都可能让整个栈崩掉。

---

## 阶段 3：起服务，日志落盘

**两个开关 + tee，缺一不可**：

```bash
mkdir -p /mnt/workspace && cd /mnt/workspace

VLLM_BATCH_INVARIANT=1 nohup vllm serve /mnt/workspace/model \
  --enable-return-routed-experts \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --port 8000 > /mnt/workspace/vllm_rocm.log 2>&1 &

# 等就绪（ROCm 首次编译可能 3–10 分钟，别 Ctrl-C）
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "ready"
```

`> vllm_rocm.log 2>&1` 必须在 `&` 前面，否则 stderr 的 WARNING 会漏掉——
**静默降级恰恰全是 WARNING**。

---

## 阶段 4：三项验证

```bash
cd /mnt/workspace/llm-inference-r3-research/r3-adapter
```

### 4.1 静默降级检测（最关键）

```bash
python silent_degradation.py --log /mnt/workspace/vllm_rocm.log --model-gib 7.84
```

**必看两行**：

| 输出 | 通过条件 |
|---|---|
| `BATCH_INVARIANT_OK` | 必须为 OK；若 `BATCH_INVARIANT_MISSING` → **AMD 上开关不生效，第一步失败** |
| `QUANT_EMULATION` | 不能出现；出现说明量化后端回退，显存会炸 |

同时看 `关键事实` 里的 `moe_backend` —— ROCm 上可能不是 HUMMING，
而是什么 AITER / Triton 后端，**记下来，和 NVIDIA 的对照**。

### 4.2 路由能不能拿到

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/workspace/model","prompt":"Hi","max_tokens":8,"temperature":0,"repetition_penalty":1.0}' \
  -o /tmp/resp.json

python -c "
import json, base64, io, numpy as np
d = json.load(open('/tmp/resp.json'))
r = d['choices'][0].get('routed_experts')
print('routed_experts:', 'None' if r is None else np.load(io.BytesIO(base64.b64decode(r))).shape)
"
```

**通过**：返回 `(8, 24, 4)` 或类似形状。
**失败**：`None` → 检查 `--enable-return-routed-experts` 是否被 ROCm 分支忽略。

### 4.3 批不变性在 AMD 上是否成立

```bash
cd /mnt/workspace/llm-inference-r3-research/r3-adapter
python long_prompt_invariance.py --port 8000
```

**判据**：

| 结果 | 含义 |
|---|---|
| 全 PASS | AMD 上批不变性成立，可以继续 |
| 出现分歧 | 需要先定位是 `matmul_persistent` 没生效，还是 ROCm 自身的数值路径不同 |

如果这里 FAIL 但 4.1 显示 `BATCH_INVARIANT_OK`，那说明**开关生效了却没覆盖 ROCm 的全部路径**——这本身就是一个值得记录的发现。

### 4.4 （可选）分布对照

```bash
cd /mnt/workspace/llm-inference-r3-research/tools
python cloud_probe.py --port 8000 --n-gen 64
```

把 AMD 上的一致率、分布形状和今天 NVIDIA 的数据对照。不要求相同，只看量级。

---

## 判定表

| 4.1 BATCH_INVARIANT | 4.2 routed_experts | 4.3 批不变性 | 结论 |
|---|---|---|---|
| OK | 有数据 | 全 PASS | ✅ **可以进第二步（装 verl）** |
| OK | 有数据 | FAIL | ⚠️ 开关生效但覆盖不全，需要先修 |
| MISSING | 有数据 | — | ❌ 确定性前提不成立，R3 数据不可用 |
| — | None | — | ❌ ROCm 分支不支持路由捕获，停下来 |

---

## 如果全部通过，第二步做什么

1. 用 verl 官方 ROCm 镜像（`docker/rocm/Dockerfile.rocm` 或 `amdagi/verl-dev:*-rocm*`）
2. 跑最小 GRPO，用 `Qwen1.5-MoE-A2.7B` 或更小的 MoE
3. 用仓库里的 `r3_contract.compare_routes()` 把「契约是否成立」变成可监控的不一致率

**注意**：verl 的 ROCm 支持主要在 gfx942 / gfx950 上验证过。阶段 0 查到的架构
不在这两个里的话，预期会踩坑。

---

## 一个必须记住的提醒

**今天测的 91.67% 是 NVIDIA 的数字。** AMD 的 kernel 是另一套实现，
数值行为会不同——这是「路径分叉」问题从「跨卡」变成「跨厂商」的重演。

所以：

- 不要预设 AMD 上一致率还是 91.67%
- 不要预设 `matmul_persistent` 存在
- **前三项验证跑完之前，什么都别假设**

---

*清单生成于 2026-11-14。所有判据来自 NVIDIA A10 上的实测，AMD 侧待验证。*
