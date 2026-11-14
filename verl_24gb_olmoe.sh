#!/usr/bin/env bash
# verl 端到端最小验证 · 24GB 显存 / 32GB 内存 单卡 · OLMoE-1B-7B + LoRA
#
# 为什么换 OLMoE：现有 Qwen1.5-MoE int4 只能推理，不能做 FSDP LoRA 训练
# （bf16 base 28GB 超 24GB）。OLMoE bf16 ~14GB，24GB 可装，且是真 MoE（R3 适用），
# cross_arch_replay.py 已验证 olmoe 回放成立。
#
# 内存布局（分时复用，不叠加）：
#   rollout 阶段：vLLM 占显存（权重13.8 + KV），util=0.6
#   training 阶段：vLLM sleep 释放；param/optimizer offload 把训练态放 32GB 内存
#
# 用法：bash verl_24gb_olmoe.sh
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/workspace/olmoe}
MODEL_ID=${MODEL_ID:-allenai/OLMoE-1B-7B-0924}

echo "== 预检 =="
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader
free -g | awk '/Mem/{print "RAM free:", $7, "GB"}'

# 1) 下载（仅当不存在）。ModelScope 镜像优先，失败回退 HF。
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "== 下载 OLMoE (~14GB) =="
  python - "$MODEL_DIR" "$MODEL_ID" <<'PY'
import sys
dst, mid = sys.argv[1], sys.argv[2]
try:
    from modelscope import snapshot_download
    snapshot_download(mid, cache_dir=dst)
except Exception as e:
    print("modelscope 失败，回退 HF:", e)
    from huggingface_hub import snapshot_download as hf
    hf(mid, local_dir=dst)
PY
fi

# 2) 装 verl 前必须先做依赖破坏检查（今天 pip 弄坏过 vLLM）。
#    若报 CRITICAL，改用隔离 venv 或 verl 官方 docker，不要直接装。
echo "== 检查 verl 安装计划 =="
pip install --dry-run verl 2>&1 | tee /tmp/verl_pip.txt || true
python r3-adapter/silent_degradation.py --pip-plan /tmp/verl_pip.txt || {
  echo "verl 会破坏关键依赖，停止。请用隔离环境安装。"; exit 1; }

echo "== 跑 GRPO（1 epoch, batch 8, 仅验证流程）=="
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${DATA:-data/gsm8k/train.parquet} \
  data.val_files=${DATA:-data/gsm8k/test.parquet} \
  data.train_batch_size=8 \
  data.max_prompt_length=256 data.max_response_length=256 \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  trainer.n_gpus_per_node=1 trainer.total_epochs=1 \
  trainer.save_freq=-1 trainer.test_freq=-1
