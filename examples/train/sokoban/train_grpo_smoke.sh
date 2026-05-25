#!/bin/bash

set -euo pipefail
set -x

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="sokoban_grpo_qwen3vl4b"

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR="${SCRIPTDIR}"
VAGEN_ROOT="$(cd "${SCRIPTDIR}/../../.." && pwd)"

# Make `python -m vagen...` and `import verl...` work when running from this examples directory.
export PYTHONPATH="${VAGEN_ROOT}:${VAGEN_ROOT}/verl:${PYTHONPATH:-}"

EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_free_wm.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_free_wm.yaml
CONFIG_DIR="${VAGEN_ROOT}/vagen/configs"
agent_loop_config_path="${CONFIG_DIR}/agent.yaml"
CUSTOM_DATASET_CLS_PATH="${VAGEN_ROOT}/vagen/gym_agent_dataset.py"
REF_MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
mkdir -p ${EXPERIMENT_DIR}

GPU_MEM_MIB="$(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -n1 \
        | tr -d '[:space:]' || true
)"

# RTX 3060 Laptop GPU is typically 6144 MiB; keep this smoke run conservative.
if [[ -n "${GPU_MEM_MIB}" && "${GPU_MEM_MIB}" -le 8192 ]]; then
    TRAIN_BATCH_SIZE=1
    PPO_MINI_BATCH_SIZE=1
    ROLLOUT_N=2
    MAX_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.25
    MAX_PROMPT_LEN=512
    MAX_RESPONSE_LEN=1024
    ROLLOUT_DTYPE=float16
    ROLLOUT_MAX_NUM_SEQS=8
    ACTOR_TORCH_COMPILE=False
    ACTOR_STRATEGY=fsdp2
    FSDP2_OFFLOAD_POLICY=True
else
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=32
    ROLLOUT_N=8
    MAX_BATCHED_TOKENS=10000
    GPU_MEMORY_UTILIZATION=0.6
    MAX_PROMPT_LEN=1000
    MAX_RESPONSE_LEN=4000
    ROLLOUT_DTYPE=bfloat16
    ROLLOUT_MAX_NUM_SEQS=1024
    ACTOR_TORCH_COMPILE=True
    ACTOR_STRATEGY=fsdp
    FSDP2_OFFLOAD_POLICY=False
fi

TRAINER_LOGGER="['console']"

# Reduce CUDA memory fragmentation on small VRAM GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# This run uses large CPU-side states (offload, HF cache, Ray). Avoid Ray killing workers too aggressively.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

if ! python3 -c "import hydra" >/dev/null 2>&1; then
    echo "ERROR: python3 cannot import hydra. Did you forget to activate the vagen environment (e.g. \`conda activate vagen\`)?" >&2
    echo "       Current python3: $(command -v python3)" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path="${CONFIG_DIR}" \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.custom_cls.path="${CUSTOM_DATASET_CLS_PATH}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${FSDP2_OFFLOAD_POLICY} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.use_torch_compile=${ACTOR_TORCH_COMPILE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
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
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
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
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=32 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    critic.optim.lr=1e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.total_training_steps=1 \
    |& tee "${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log" "${BASEDIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"
# actor_rollout_ref.model.lora_rank=8 \
#     actor_rollout_ref.model.lora_alpha=16 \
#     actor_rollout_ref.rollout.load_format="safetensors" \
#     actor_rollout_ref.model.target_modules="all-linear" \
