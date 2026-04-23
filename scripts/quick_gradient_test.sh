#!/bin/bash
# Quick Gradient Test Script - Tests if actor gradients update correctly
# Uses minimal data: batch_size=1, 3 turns, 3 QA pairs

set -xeuo pipefail

# Quick test parameters
export HMEMS_MAX_TEST_TURNS=3
export HMEMS_MAX_TEST_QAS=3

# Project configuration
project_name='HMEMS-QuickTest'
exp_name="qwen3-1.7b-quick-gradient-test"

# Paths
BASE_MODEL='/root/autodl-tmp/models/Qwen3-1.7B'
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${WORKING_DIR}/data/hmems_session_based"
TRAIN_DATA_DIR="${DATA_DIR}"
TEST_DATA_DIR="${DATA_DIR}"

# Algorithm
adv_estimator=grpo
use_kl_loss=false
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# Length limits
max_prompt_length=8192
max_response_length=2048
max_start_length=4096
max_obs_length=512

# Minimal batch for quick test
train_batch_size=1
val_batch_size=1
customized_grpo_rollout_n=4

# Model
MODEL_PATH=${BASE_MODEL}
CKPTS_DIR="${WORKING_DIR}/checkpoints/${exp_name}"
mkdir -p "${CKPTS_DIR}"

# Generation
temperature=0.3
top_p=1.0
top_k=-1

# Memory optimization
offload=true
actor_param_offload=true
actor_optimizer_offload=true
ref_param_offload=true

# vLLM memory
gpu_memory_utilization=0.5
tensor_model_parallel_size=1

# Micro batch
ppo_mini_batch_size=1
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1

# Custom dataset path
CUSTOM_DATASET_PATH="${WORKING_DIR}/src/session_based_dataset.py"

# Ray configuration
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
RAY_ADDRESS="http://${MASTER_ADDR}:8265"

# Environment
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Server URLs
respond_url="http://127.0.0.1:5005/batch_process"
consolidate_url="http://127.0.0.1:5005/consolidate"

# Per-turn training mode
use_per_turn_mode=true
evidence_reward_mode=1

echo "=============================================="
echo "HMEMS Quick Gradient Test"
echo "=============================================="
echo "  Max turns: ${HMEMS_MAX_TEST_TURNS}"
echo "  Max QAs: ${HMEMS_MAX_TEST_QAS}"
echo "  Model: ${MODEL_PATH}"
echo "=============================================="

# Check data exists
if [ ! -f "${TRAIN_DATA_DIR}/train.jsonl" ]; then
    echo "Error: Training data not found"
    exit 1
fi

# Check if Ray is running, if not start it
if ! ray status 2>&1 | grep -q "Ray"; then
    echo "Starting Ray..."
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
    sleep 10
fi

# Check if memory server is running
if ! curl -s "${respond_url}" > /dev/null 2>&1; then
    echo "Starting memory server..."
    cd "${WORKING_DIR}"
    nohup bash -c 'export $(grep -v "^#" .env | xargs) && python3 consolidation_server.py --port 5005' > memory_server.log 2>&1 &
    sleep 3
fi

# Submit quick training job
echo "Submitting quick training job..."
ray job submit --no-wait \
    --working-dir "${WORKING_DIR}" \
    --runtime-env "${WORKING_DIR}/verl/trainer/runtime_env.yaml" \
    -- python3 run_hmems_session_training.py \
    data.train_files="${TRAIN_DATA_DIR}/train.jsonl" \
    data.val_files="${TEST_DATA_DIR}/validation.jsonl" \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.respond_url="${respond_url}" \
    +data.consolidate_url="${consolidate_url}" \
    +data.use_per_turn_mode=${use_per_turn_mode} \
    +data.evidence_reward_mode=${evidence_reward_mode} \
    data.custom_cls.path="${CUSTOM_DATASET_PATH}" \
    data.custom_cls.name="SessionBasedHMEMSDataset" \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload} \
    reward_model.enable=True \
    trainer.logger="[console, tensorboard]" \
    trainer.rollout_data_dir="${WORKING_DIR}/rollout_data/${exp_name}" \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

echo "=============================================="
echo "Quick test job submitted!"
echo "Monitor at: http://localhost:8265"
echo "=============================================="