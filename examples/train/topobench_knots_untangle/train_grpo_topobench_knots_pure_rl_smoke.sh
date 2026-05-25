#!/bin/bash

set -euo pipefail
set -x

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="topobench_knots_untangle_grpo_pure_rl_smoke"

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "${SCRIPTDIR}/../../.." && pwd)"
TOPOBENCH_ROOT="$(cd "${VAGEN_ROOT}/.." && pwd)"

# Make `python -m vagen...` and `import verl...` work.
# Also include TopoBench root so `import environments...` resolves in wrappers.
export PYTHONPATH="${VAGEN_ROOT}:${VAGEN_ROOT}/verl:${TOPOBENCH_ROOT}:${PYTHONPATH:-}"

EXPERIMENT_DIR="${SCRIPTDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}"
SAVE_CHECKPOINT_DIR="${EXPERIMENT_DIR}/verl_checkpoints"
DATASET_TRAIN="${SCRIPTDIR}/train_topobench_knots_untangle_one_step_smoke.yaml"
DATASET_VAL="${SCRIPTDIR}/val_topobench_knots_untangle_one_step_smoke.yaml"
CONFIG_DIR="${VAGEN_ROOT}/vagen/configs"
agent_loop_config_path="${CONFIG_DIR}/agent.yaml"
CUSTOM_DATASET_CLS_PATH="${VAGEN_ROOT}/vagen/gym_agent_dataset.py"
REF_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
mkdir -p "${EXPERIMENT_DIR}"

# Conservative defaults for a smoke run.
TRAIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=1
ROLLOUT_N=2
MAX_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.20
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=256
ROLLOUT_DTYPE=float16
ROLLOUT_MAX_NUM_SEQS=8
ACTOR_TORCH_COMPILE=False
ACTOR_STRATEGY=fsdp2
FSDP2_OFFLOAD_POLICY=True

TRAINER_LOGGER="['console', 'tensorboard']"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export SGLANG_LOG_LEVEL=info

if ! python3 -c "import hydra" >/dev/null 2>&1; then
    echo "ERROR: python3 cannot import hydra. Activate your vagen env first." >&2
    exit 1
fi

python3 - <<'PY'
import os
import sys

print("[preflight] python:", sys.executable, flush=True)
print("[preflight] PYTHONPATH:", os.environ.get("PYTHONPATH", ""), flush=True)

import torch
print("[preflight] torch:", torch.__version__, flush=True)
print("[preflight] cuda_available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("[preflight] gpu:", torch.cuda.get_device_name(0), flush=True)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[preflight] gpu_mem_gb: {total_gb:.2f}", flush=True)

import ray
import sglang
import transformers
import vagen

print("[preflight] ray:", ray.__version__, flush=True)
print("[preflight] sglang:", sglang.__version__, flush=True)
print("[preflight] transformers:", transformers.__version__, flush=True)
print("[preflight] vagen:", vagen.__file__, flush=True)

from environments.knots_untangle.gym.env import KnotsUntangleEnv
print("[preflight] topobench env import: OK", KnotsUntangleEnv, flush=True)
PY

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name='vagen_multiturn' \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.custom_cls.path="${CUSTOM_DATASET_CLS_PATH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${FSDP2_OFFLOAD_POLICY}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    actor_rollout_ref.ref.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.use_torch_compile="${ACTOR_TORCH_COMPILE}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${agent_loop_config_path}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.critic_warmup=0 \
    "trainer.logger=${TRAINER_LOGGER}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
    trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    trainer.log_val_generations=0 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path="${REF_MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.enable=False \
    critic.model.fsdp_config.model_dtype=fp16 \
    critic.model.fsdp_config.dtype=float16 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.total_training_steps=1 \
    |& tee "${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"
