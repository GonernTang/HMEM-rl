"""
HMEMS Session-Based Training Entry Point

This script is the main entry point for training the HMEMS session-based model.
It generates a configuration JSON and then launches the actual training.

Usage:
    # Generate config and train locally (for testing):
    python train_session.py --train_data data/hmems_session_based/train.jsonl ...

    # For full distributed training with Ray:
    bash scripts/train_session.sh

Note:
    The actual training is performed by run_hmems_session_training.py which uses verl.
    This script only generates the configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add verl to path
VERL_PATH = Path(__file__).parent / "Mem-alpha" / "verl"
if VERL_PATH.exists():
    sys.path.insert(0, str(VERL_PATH))


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMEMS Session-Based Model")

    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model_name", type=str, default="qwen3-1.7b")

    # Session-based Data
    parser.add_argument("--train_data", type=str, default="data/hmems_session_based/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/hmems_session_based/validation.jsonl")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--customized_grpo_rollout_n", type=int, default=16)

    # Length limits (session-based data is longer)
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_response_length", type=int, default=2048)
    parser.add_argument("--max_start_length", type=int, default=4096)
    parser.add_argument("--max_obs_length", type=int, default=512)

    # Reward weights
    parser.add_argument("--qa_weight", type=float, default=1.0)
    parser.add_argument("--operation_reward_weight", type=float, default=0.05)

    # Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    # Memory server
    parser.add_argument("--memory_server_url", type=str, default="http://localhost:5005/batch_process")

    # Other
    parser.add_argument("--enable_thinking", type=bool, default=False)
    parser.add_argument("--offload", type=bool, default=True)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1)

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    return parser.parse_args()


def build_config(args) -> dict:
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
                "path": "src.session_based_dataset",
                "name": "SessionBasedHMEMSDataset",
            },
        },
        "algorithm": {
            "adv_estimator": "session_grpo",
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
            "reward_manager": "session_based",
            "qa_weight": args.qa_weight,
            "operation_reward_weight": args.operation_reward_weight,
        },
        "trainer": {
            "logger": ["console"],
            "rollout_data_dir": f"./rollout_data/{args.model_name}_session",
            "val_only": False,
            "val_before_train": False,
            "n_gpus_per_node": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "nnodes": 1,
            "save_freq": args.save_freq,
            "test_freq": args.val_freq,
            "total_epochs": args.total_epochs,
            "default_local_dir": f"{args.checkpoint_dir}/{args.model_name}_session",
            "resume_mode": "auto",
        },
        "customized_grpo_rollout_n": args.customized_grpo_rollout_n,
        "max_turns": 10,
        "enable_thinking": args.enable_thinking,
        "respond_url": args.memory_server_url,
    }


def print_training_info(args):
    """Print training configuration."""
    print("=" * 60)
    print("HMEMS Session-Based Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print()
    print("Hyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Val batch size: {args.val_batch_size}")
    print(f"  Total epochs: {args.total_epochs}")
    print(f"  QA weight: {args.qa_weight}")
    print(f"  Operation reward weight: {args.operation_reward_weight}")
    print()
    print("Length limits:")
    print(f"  Max prompt length: {args.max_prompt_length}")
    print(f"  Max response length: {args.max_response_length}")
    print()
    print("Server URLs:")
    print(f"  Memory server: {args.memory_server_url}")
    print("=" * 60)


def main():
    args = parse_args()

    # Validate data paths
    if not os.path.exists(args.train_data):
        print(f"Error: Training data not found at {args.train_data}")
        print("Please run: python scripts/build_session_dataset.py")
        sys.exit(1)

    # Build config
    config = build_config(args)
    print_training_info(args)

    # Save config for reproducibility
    config_path = Path(args.checkpoint_dir) / f"{args.model_name}_session_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nConfig saved to {config_path}")

    print("\n" + "=" * 60)
    print("To run training:")
    print()
    print("  1. Start memory server:")
    print("     python consolidation_server.py --port 5005")
    print()
    print("  2. Run training (single GPU):")
    print("     bash scripts/train_session_single_gpu.sh")
    print()
    print("  3. Or run distributed training:")
    print("     bash scripts/train_session.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
