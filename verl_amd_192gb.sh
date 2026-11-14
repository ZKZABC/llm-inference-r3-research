#!/usr/bin/env bash
# verl 端到端 · AMD gfx942 192GB 显存 / 200GB 内存 · OLMoE-1B-7B
#
# 为什么 OLMoE：真 MoE（R3 适用）、cross_arch_replay 已验证、bf16 仅 14GB。
# 192GB 显存下两种模式都装得下：
#   LoRA（默认）   : 14GB 权重 + 适配器/激活 ≈ 20GB
#   FULL=1 全参    : 7B×16字节 = 112GB + 激活 < 192GB，无需 offload
#
# 前提：R3 两前提已在本机验证（capture 可用；batch_invariant 需 LDS 补丁，
#       见 AMD_ROCM_FINDINGS.md）。verl 端到端本身不依赖 R3，R3-in-loop 是后续扩展。
#
# 用法：bash verl_amd_192gb.sh          # LoRA 快验
#       FULL=1 bash verl_amd_192gb.sh   # 全参
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/workspace/olmoe}
MODEL_ID=${MODEL_ID:-allenai/OLMoE-1B-7B-0924}
FULL=${FULL:-0}

echo "== 预检 =="
rocminfo | grep -m1 "gfx" || true
python -c "import torch; assert torch.version.hip, '需要 ROCm torch'; print('hip', torch.version.hip)"
nvidia-smi 2>/dev/null || rocm-smi --showmeminfo vram | head -5
free -g | awk '/Mem/{print "RAM free:", $7, "GB"}'

# 1) 下载 OLMoE bf16（仅当不存在）
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "== 下载 OLMoE (~14GB) =="
  python - "$MODEL_DIR" "$MODEL_ID" <<'PY'
import sys
dst, mid = sys.argv[1], sys.argv[2]
try:
    from modelscope import snapshot_download; snapshot_download(mid, cache_dir=dst)
except Exception as e:
    print("modelscope 失败回退 HF:", e)
    from huggingface_hub import snapshot_download as hf; hf(mid, local_dir=dst)
PY
fi

# 2) 装 verl 前必须做依赖破坏检查（ROCm torch/vllm 很容易被 pip 升级弄坏）
echo "== 检查 verl 安装计划 =="
pip install --dry-run verl 2>&1 | tee /tmp/verl_amd_pip.txt || true
python r3-adapter/silent_degradation.py --pip-plan /tmp/verl_amd_pip.txt || {
  echo "verl 会破坏 ROCm torch/vllm，停止。请用 verl 官方 ROCm docker 或隔离 venv。"; exit 1; }

# 3) ROCm 单卡集合通信保险
export RCCL_MSCCL_ENABLE=0 RCCL_MSCCLPP_ENABLE=0

LORA_ARGS=""
if [ "$FULL" != "1" ]; then
  LORA_ARGS="actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 actor_rollout_ref.model.target_modules=all-linear"
fi

echo "== GRPO (FULL=$FULL, 1 epoch, batch 8) =="
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${DATA:-data/gsm8k/train.parquet} \
  data.val_files=${DATA:-data/gsm8k/test.parquet} \
  data.train_batch_size=8 \
  data.max_prompt_length=256 data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  $LORA_ARGS \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  trainer.n_gpus_per_node=1 trainer.total_epochs=1 \
  trainer.save_freq=-1 trainer.test_freq=-1