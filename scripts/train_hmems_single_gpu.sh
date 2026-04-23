#!/bin/bash
# HMEMS Consolidation Agent Single GPU Training Script
# Optimized for single GPU training with limited VRAM

set -xeuo pipefail

# Training parameters
compression_ratio_weight=${1:-0.05}
sub_sample_question_ratio=1.0

# Project configuration
project_name='HMEMS-Consolidation'
exp_name="qwen3-1.7b-single-gpu-compression${compression_ratio_weight}"

# Paths
BASE_MODEL='/root/autodl-tmp/models/Qwen3-1.7B'
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${WORKING_DIR}/data/hmems_session_based"
TRAIN_DATA_DIR="${DATA_DIR}"
TEST_DATA_DIR="${DATA_DIR}"
export TENSORBOARD_DIR="${WORKING_DIR}/tensorboard_dir/${project_name}/${exp_name}"

# Algorithm
adv_estimator=grpo
use_kl_loss=false
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# Length limits (session-based needs longer context)
max_prompt_length=8192
max_response_length=2048
max_start_length=4096
max_obs_length=512

# Single GPU training - reduced batch sizes
train_batch_size=1
val_batch_size=1
customized_grpo_rollout_n=4

# Model
MODEL_PATH=${BASE_MODEL}
CKPTS_DIR="${WORKING_DIR}/checkpoints/${exp_name}"
mkdir -p "${CKPTS_DIR}"

# Generation (reduced memory usage)
temperature=0.3
top_p=1.0
top_k=-1

# Single GPU memory optimization - MUST enable offload
offload=true
actor_param_offload=true
actor_optimizer_offload=true
ref_param_offload=true

# vLLM memory (reduced for single GPU)
gpu_memory_utilization=0.5
tensor_model_parallel_size=1

# Single GPU micro batch
ppo_mini_batch_size=1
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1

# Required: path to custom dataset class (MUST be absolute path)
CUSTOM_DATASET_PATH="${WORKING_DIR}/src/session_based_dataset.py"

# Ray configuration - single node
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
use_per_turn_mode=true  # Set to true for per-turn training
evidence_reward_mode=1  # 1, 2, or 3

echo "=============================================="
echo "HMEMS Consolidation Agent Single GPU Training"
echo "=============================================="
echo "  Project: ${project_name}"
echo "  Experiment: ${exp_name}"
echo "  Model: ${MODEL_PATH}"
echo "  Nodes: ${NNODES}"
echo "  Compression ratio weight: ${compression_ratio_weight}"
echo "  Batch sizes: train=${train_batch_size}, val=${val_batch_size}"
echo "  Memory optimization: offload=${offload}"
echo "=============================================="

# Check data exists
if [ ! -f "${TRAIN_DATA_DIR}/train.jsonl" ]; then
    echo "Error: Training data not found. Run: python scripts/preprocess_locomo.py"
    exit 1
fi


# Determine GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "Detected ${GPU_COUNT} GPU(s)"

# Load .env file for API keys
if [ -f "${WORKING_DIR}/.env" ]; then
    echo "Loading .env file..."
    set -a
    source "${WORKING_DIR}/.env"
    set +a
    export $(grep -v '^#' "${WORKING_DIR}/.env" | xargs 2>/dev/null) || true
fi

# Stop any existing Ray and memory server
ray stop || true
pkill -f "mock_memory_server.py" || true
pkill -f "consolidation_server.py" || true
sleep 2

# Clean up old Ray temp files
rm -rf /root/autodl-tmp/ray_temp 2>/dev/null || true

# Start memory server (real consolidation_server) in background with .env loaded
echo "Starting memory server on port 5005..."
cd "${WORKING_DIR}"
nohup bash -c 'export $(grep -v "^#" .env | xargs) && python3 consolidation_server.py --port 5005' > memory_server.log 2>&1 &
sleep 3
echo "Memory server started (PID: $!)"

# Ray temp directory (use /root/autodl-tmp to avoid /tmp full)
export RAY_TMP_DIR="/root/autodl-tmp/ray_temp"
mkdir -p "${RAY_TMP_DIR}"

# Start Ray head for single node
echo "Starting Ray head..."
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir="${RAY_TMP_DIR}"
sleep 10

# Submit training job
echo "Note: Ensure memory server is running at ${respond_url}"
echo "Or use a remote memory server URL if available."
echo ""
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
    trainer.n_gpus_per_node=${GPU_COUNT} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    customized_grpo_rollout_n=${customized_grpo_rollout_n}

echo "=============================================="
echo "Training job submitted successfully!"
echo "Monitor at: http://localhost:8265"
echo "Checkpoints saved to: ${CKPTS_DIR}"
echo "=============================================="

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${WORKING_DIR}/logs/training_job_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

# Monitor training progress
echo "Starting training monitor..."
echo "Monitor at: http://localhost:8265"
echo "Press Ctrl+C to stop monitoring (training will continue in background)"
echo ""

monitor_interval=10  # seconds between status checks
last_line_count=0

# Get job ID - extract just the job ID string (e.g., raysubmit_xxx)
get_job_id() {
    ray job list 2>/dev/null | grep -E "RUNNING|PENDING" | head -1 | grep -oE "raysubmit_[a-zA-Z0-9]+" | head -1
}

while true; do
    # Get running job ID
    JOB_ID=$(get_job_id)

    if [ -n "${JOB_ID}" ]; then
        echo ""
        echo "=============================================="
        echo "Training Status - $(date)"
        echo "=============================================="

        # Get training output
        job_output=$(ray job logs "${JOB_ID}" 2>/dev/null)
        total_lines=$(echo "$job_output" | wc -l)
        new_lines=$((total_lines - last_line_count))

        if [ "$new_lines" -gt 0 ]; then
            if [ "$last_line_count" -eq 0 ]; then
                # First time - show last 10 lines as context
                echo ">>> Training output (last 10 lines) <<<"
                echo "$job_output" | tail -10
            else
                # Show only new lines
                echo ">>> New training output (+${new_lines} lines) <<<"
                echo "$job_output" | tail -${new_lines}
            fi
        else
            echo ">>> No new training output yet <<<"
        fi
        last_line_count=$total_lines

        echo ""
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=index,gpu_util,mem_util,mem_used --format=csv,noheader 2>/dev/null || echo "  Failed to query GPU"

        echo ""
        echo "--- Memory Server (last line) ---"
        tail -1 "${WORKING_DIR}/memory_server.log" 2>/dev/null | head -c 200 || echo "  No log"

    else
        echo ""
        echo "=============================================="
        echo "Status - $(date)"
        echo "=============================================="
        echo "No RUNNING/PENDING job found."

        # Check if job finished
        if [ -d "${CKPTS_DIR}" ]; then
            ckpt_count=$(ls "${CKPTS_DIR}" 2>/dev/null | wc -l)
            echo "Checkpoints: ${ckpt_count} files"
        fi

        echo ""
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=index,gpu_util,mem_util,mem_used,mem_total --format=csv,noheader 2>/dev/null || echo "  Failed to query GPU"

        echo ""
        echo "Training completed! Full logs saved to ${LOG_FILE}"
        break  # Exit the monitor loop
    fi

    echo ""
    echo "Next update in ${monitor_interval}s... (Ctrl+C to stop)"
    sleep ${monitor_interval}
done
