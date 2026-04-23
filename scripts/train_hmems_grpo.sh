#!/bin/bash
# HMEMS Consolidation Agent GRPO Training Script
# Based on Mem-alpha training script structure

set -xeuo pipefail

# Training parameters
compression_ratio_weight=${1:-0.05}
sub_sample_question_ratio=1.0

# Project configuration
project_name='HMEMS-Consolidation'
exp_name="qwen3-1.7b-compression${compression_ratio_weight}"

# Paths
BASE_MODEL='Qwen/Qwen3-1.7B'
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${WORKING_DIR}/data/hmems_consolidation"
TRAIN_DATA_DIR="${DATA_DIR}"
TEST_DATA_DIR="${DATA_DIR}"
QA_LOOKUP="${DATA_DIR}/qa_lookup.json"

# Algorithm
adv_estimator=grpo
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# Length limits
max_prompt_length=2048
max_response_length=1024
max_start_length=1024
max_obs_length=256

# Training
train_batch_size=32
val_batch_size=32
customized_grpo_rollout_n=8

# Model
MODEL_PATH=${BASE_MODEL}
CKPTS_DIR="${WORKING_DIR}/checkpoints/${exp_name}"
mkdir -p "${CKPTS_DIR}"

# Generation
temperature=0.3
top_p=1.0
top_k=-1
offload=true

# Ray configuration
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

# Environment
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Server URLs
respond_url="http://127.0.0.1:5005/batch_process"
consolidate_url="http://127.0.0.1:5005/consolidate"

echo "=============================================="
echo "HMEMS Consolidation Agent Training"
echo "=============================================="
echo "  Project: ${project_name}"
echo "  Experiment: ${exp_name}"
echo "  Model: ${MODEL_PATH}"
echo "  Nodes: ${NNODES}"
echo "  Compression ratio weight: ${compression_ratio_weight}"
echo "=============================================="

# Check data exists
if [ ! -f "${TRAIN_DATA_DIR}/train.jsonl" ]; then
    echo "Error: Training data not found. Run: python scripts/preprocess_locomo.py"
    exit 1
fi

if [ ! -f "${QA_LOOKUP}" ]; then
    echo "Error: QA lookup not found at ${QA_LOOKUP}"
    exit 1
fi

if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head..."
    ray stop || true
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
    sleep 10

    # Submit training job
    ray job submit --no-wait \
        --working-dir "${WORKING_DIR}" \
        -- python3 run_hmems_training.py \
        data.train_files="${TRAIN_DATA_DIR}/train.jsonl" \
        data.val_files="${TEST_DATA_DIR}/validation.jsonl" \
        data.train_batch_size=${train_batch_size} \
        data.val_batch_size=${val_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.max_start_length=${max_start_length} \
        data.max_obs_length=${max_obs_length} \
        data.shuffle_train_dataloader=True \
        data.custom_cls.path="src.hmems_dataset" \
        data.custom_cls.name="HMEMSConsolidationDataset" \
        data.qa_lookup_path="${QA_LOOKUP}" \
        algorithm.adv_estimator=${adv_estimator} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        reward_model.enable=True \
        reward_model.strategy=fsdp \
        reward_model.reward_manager=hmems_naive \
        reward_model.compression_ratio_weight=${compression_ratio_weight} \
        reward_model.qa_weight=1.0 \
        trainer.logger=['console'] \
        trainer.rollout_data_dir="${WORKING_DIR}/rollout_data/${exp_name}" \
        trainer.val_only=false \
        trainer.val_before_train=false \
        trainer.n_gpus_per_node=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l || echo 1) \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=1 \
        trainer.test_freq=50 \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.total_epochs=5 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        customized_grpo_rollout_n=${customized_grpo_rollout_n} \
        respond_url="${respond_url}" \
        consolidate_url="${consolidate_url}"

    echo "Job submitted. Keeping Ray head alive..."
    while true; do
        echo "$(date): Master node is alive"
        sleep 60
    done
else
    echo "Starting Ray worker on node rank $NODE_RANK"
    ray stop || true
    ray start --address="${MASTER_ADDR}:6379"
    echo "Ray worker connected"
    while true; do
        sleep 60
    done
fi
