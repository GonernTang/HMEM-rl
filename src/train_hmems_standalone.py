"""
HMEMS Consolidation Agent Standalone Training Script

This script provides a simplified training interface that:
1. Loads the HMEMS dataset
2. Initializes the consolidation manager
3. Runs GRPO training using verl

For full distributed training, use: bash scripts/train_hmems_grpo.sh
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add verl to path
VERL_PATH = Path(__file__).parent.parent / "verl"
if VERL_PATH.exists():
    sys.path.insert(0, str(VERL_PATH))

from src.hmems_dataset import HMEMSConsolidationDataset, hmems_collate_fn
from src.hmems_generation import (
    HMEMSGenerationManager,
    ConsolidationGenerationConfig,
    create_hmems_generation_manager,
)
from src.hmems_reward_manager import HMEMSConsolidationRewardManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMEMS Consolidation Agent")

    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_name", type=str, default="qwen3-1.7b")

    # Data
    parser.add_argument("--train_data", type=str, default="data/hmems_consolidation/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/hmems_consolidation/validation.jsonl")
    parser.add_argument("--qa_lookup", type=str, default="data/hmems_consolidation/qa_lookup.json")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--customized_grpo_rollout_n", type=int, default=16)

    # Length limits
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--max_start_length", type=int, default=1024)
    parser.add_argument("--max_obs_length", type=int, default=256)

    # Reward weights
    parser.add_argument("--compression_ratio_weight", type=float, default=0.05)

    # Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    # Server URLs
    parser.add_argument("--respond_url", type=str, default="http://localhost:5005/batch_process")
    parser.add_argument("--consolidate_url", type=str, default="http://localhost:5005/consolidate")

    # Other
    parser.add_argument("--enable_thinking", type=bool, default=False)
    parser.add_argument("--offload", type=bool, default=True)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1)

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    return parser.parse_args()


def build_config(args) -> Dict[str, Any]:
    """Build verl-compatible config from args."""
    return {
        "data": {
            "train_files": [args.train_data],
            "val_files": [args.val_data],
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_start_length": args.max_start_length,
            "max_obs_length": args.max_obs_length,
            "shuffle_train_dataloader": True,
            "custom_cls": {
                "path": "src.hmems_dataset",
                "name": "HMEMSConsolidationDataset",
            },
            "qa_lookup_path": args.qa_lookup,
        },
        "algorithm": {
            "adv_estimator": "grpo",
        },
        "actor_rollout_ref": {
            "model": {
                "path": args.model_path,
                "enable_gradient_checkpointing": True,
                "use_remove_padding": True,
            },
            "actor": {
                "optim": {
                    "lr": args.lr,
                    "lr_warmup_steps_ratio": 0.05,
                },
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": 2,
                "fsdp_config": {
                    "param_offload": args.offload,
                    "optimizer_offload": args.offload,
                },
            },
            "rollout": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "tensor_model_parallel_size": 1,
                "name": "vllm",
                "gpu_memory_utilization": 0.8,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {
                    "param_offload": args.offload,
                },
            },
        },
        "reward_model": {
            "enable": True,
            "strategy": "fsdp",
            "reward_manager": "hmems_naive",
            "compression_ratio_weight": args.compression_ratio_weight,
            "qa_weight": 1.0,
        },
        "trainer": {
            "logger": ["console"],
            "rollout_data_dir": f"./rollout_data/{args.model_name}",
            "val_only": False,
            "val_before_train": False,
            "n_gpus_per_node": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "nnodes": 1,
            "save_freq": args.save_freq,
            "test_freq": args.val_freq,
            "total_epochs": args.total_epochs,
            "default_local_dir": args.checkpoint_dir,
            "resume_mode": "auto",
        },
        "customized_grpo_rollout_n": args.customized_grpo_rollout_n,
        "max_turns": 5,
        "enable_thinking": args.enable_thinking,
        "respond_url": args.respond_url,
        "consolidate_url": args.consolidate_url,
    }


def print_training_info(args, config):
    """Print training configuration."""
    print("=" * 60)
    print("HMEMS Consolidation Agent Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"QA lookup: {args.qa_lookup}")
    print()
    print("Hyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Val batch size: {args.val_batch_size}")
    print(f"  Total epochs: {args.total_epochs}")
    print(f"  Compression ratio weight: {args.compression_ratio_weight}")
    print()
    print("Length limits:")
    print(f"  Max prompt length: {args.max_prompt_length}")
    print(f"  Max response length: {args.max_response_length}")
    print()
    print("Server URLs:")
    print(f"  Respond (QA): {args.respond_url}")
    print(f"  Consolidate: {args.consolidate_url}")
    print("=" * 60)


def main():
    args = parse_args()

    # Validate data paths
    if not os.path.exists(args.train_data):
        print(f"Error: Training data not found at {args.train_data}")
        print("Please run: python scripts/preprocess_locomo.py")
        sys.exit(1)

    if not os.path.exists(args.qa_lookup):
        print(f"Error: QA lookup not found at {args.qa_lookup}")
        sys.exit(1)

    # Build config
    config = build_config(args)
    print_training_info(args, config)

    # Save config for reproducibility
    config_path = Path(args.checkpoint_dir) / f"{args.model_name}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nConfig saved to {config_path}")

    print("\n" + "=" * 60)
    print("IMPORTANT: For full distributed training with Ray cluster,")
    print("use the following command:")
    print()
    print(f"  bash scripts/train_hmems_grpo.sh {args.compression_ratio_weight}")
    print()
    print("This script is for local testing only.")
    print("=" * 60)

    # For local testing, we would need to:
    # 1. Initialize verl's actor/rollout/ref workers
    # 2. Set up the data loader
    # 3. Run the GRPO training loop
    #
    # See verl/trainer/main_ppo.py for the full implementation

    print("\nFor local testing, start the consolidation server first:")
    print("  python consolidation_server.py --port 5005")


if __name__ == "__main__":
    main()
