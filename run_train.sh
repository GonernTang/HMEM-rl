#!/bin/bash
export PATH="/root/miniconda3/envs/verl-agent/bin:$PATH"
cd /root/autodl-tmp/memoryrl/HMEMS
echo "Starting training script..."
mkdir -p logs
bash scripts/train_hmems_single_gpu.sh