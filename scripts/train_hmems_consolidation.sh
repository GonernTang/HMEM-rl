#!/bin/bash
# HMEMS Consolidation Agent Training Script
#
# Usage:
#   bash scripts/train_hmems_consolidation.sh
#
# Requirements:
#   - Ray cluster must be started
#   - Training data must exist at data/hmems_session_based/train.jsonl

set -e  # Exit on error

# Configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
CONFIG_FILE="config/hmems_consolidation.yaml"
TRAIN_DATA="${TRAIN_DATA:-data/hmems_session_based/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/hmems_session_based/validation.jsonl}"

echo "=========================================="
echo "HMEMS Consolidation Agent Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "=========================================="

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Please run scripts/build_session_dataset.py first"
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VAL_DATA" ]; then
    echo "WARNING: Validation data not found at $VAL_DATA"
    echo "Will use training data for validation"
    VAL_DATA=$TRAIN_DATA
fi

# Check if Ray is initialized
echo "Checking Ray status..."
ray status > /dev/null 2>&1 || {
    echo "ERROR: Ray is not initialized"
    echo "Please start Ray first: ray start --head"
    exit 1
}

# Set Python path to include src and parent directory (for Mem-alpha import)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/../Mem-alpha"

# Launch training using verl main_ppo
echo "Starting training..."
python -m verl.trainer.main_ppo \
    --config-path config \
    --config-name hmems_consolidation \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.model.path="$MODEL_PATH"
