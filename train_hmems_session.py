"""
HMEMS Session-Based GRPO Training Script

This script trains the HMEMS agent using session-level granularity.
Each session is processed as a single step, with QA reward computed
only for evidence sessions.

Key features:
- Session-level granularity instead of turn-level
- Only evidence sessions compute policy gradient (advantage != 0)
- Non-evidence sessions have advantage = 0
"""

import os
import sys
from pathlib import Path

# Add verl to path (HMEMS uses verl from Mem-alpha)
sys.path.insert(0, str(Path(__file__).parent.parent / "Mem-alpha" / "verl"))

import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train HMEMS with Session-Based Data")

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

    return parser.parse_args()


class SessionBasedDataset:
    """Simple dataset wrapper for session-based training."""

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} session steps from {data_path}")

        # Statistics
        evidence_count = sum(1 for d in self.data if d.get('is_evidence_session', False))
        non_evidence_count = len(self.data) - evidence_count
        total_qas = sum(len(d.get('qa_pairs', [])) for d in self.data)
        print(f"  - Evidence sessions: {evidence_count}")
        print(f"  - Non-evidence sessions: {non_evidence_count}")
        print(f"  - Total QA pairs: {total_qas}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(dataset: SessionBasedDataset, batch_size: int, shuffle: bool = True):
    """Create DataLoader for the session-based dataset."""

    def collate_fn(batch):
        return {
            "conv_id": [item["conv_id"] for item in batch],
            "session_name": [item["session_name"] for item in batch],
            "session_idx": [item["session_idx"] for item in batch],
            "session_dialogue": [item["session_dialogue"] for item in batch],
            "qa_pairs": [item["qa_pairs"] for item in batch],
            "is_evidence_session": [item["is_evidence_session"] for item in batch],
            "n_turns": [item["n_turns"] for item in batch],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def main():
    args = parse_args()

    print("=" * 60)
    print("HMEMS Session-Based GRPO Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"QA weight: {args.qa_weight}")
    print(f"Operation reward weight: {args.operation_reward_weight}")
    print("=" * 60)

    if not os.path.exists(args.train_data):
        print(f"Error: Training data not found at {args.train_data}")
        print("Please run: python scripts/build_session_dataset.py")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = hf_tokenizer(args.model_path)

    print("Loading datasets...")
    train_dataset = SessionBasedDataset(args.train_data)
    val_dataset = SessionBasedDataset(args.val_data)

    train_loader = create_dataloader(train_dataset, args.train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.val_batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Statistics
    train_evidence = sum(1 for d in train_dataset.data if d.get('is_evidence_session', False))
    val_evidence = sum(1 for d in val_dataset.data if d.get('is_evidence_session', False))
    print(f"Train evidence sessions: {train_evidence} / {len(train_dataset)}")
    print(f"Val evidence sessions: {val_evidence} / {len(val_dataset)}")

    config = {
        "data": {
            "train_files": args.train_data,
            "val_files": args.val_data,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_start_length": args.max_start_length,
            "max_obs_length": args.max_obs_length,
            "shuffle_train_dataloader": True,
        },
        "algorithm": {"adv_estimator": "grpo"},
        "actor_rollout_ref": {
            "model": {
                "path": args.model_path,
                "enable_gradient_checkpointing": True,
                "use_remove_padding": True,
            },
            "actor": {
                "optim": {"lr": args.lr, "lr_warmup_steps_ratio": 0.05},
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "ppo_mini_batch_size": args.train_batch_size,
                "ppo_micro_batch_size_per_gpu": 2,
                "fsdp_config": {"param_offload": args.offload, "optimizer_offload": args.offload},
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
                "fsdp_config": {"param_offload": args.offload},
            },
        },
        "reward_model": {
            "qa_weight": args.qa_weight,
            "operation_reward_weight": args.operation_reward_weight,
            "reward_manager": "session_based",
            "reward_fn_key": "data_source",
        },
        "trainer": {
            "logger": ["console"],
            "rollout_data_dir": "./rollout_data",
            "val_only": False,
            "val_before_train": False,
            "n_gpus_per_node": torch.cuda.device_count(),
            "nnodes": 1,
            "save_freq": args.save_freq,
            "test_freq": args.val_freq,
            "total_epochs": args.total_epochs,
            "default_local_dir": f"./checkpoints/{args.model_name}_session",
            "resume_mode": "auto",
        },
        "customized_grpo_rollout_n": args.customized_grpo_rollout_n,
        "max_turns": 10,
        "enable_thinking": args.enable_thinking,
        "respond_url": args.memory_server_url,
    }

    config_path = f"./configs/{args.model_name}_session_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nConfig saved to {config_path}")
    print("\nTo run full distributed training, use verl.trainer.main_ppo with Ray cluster")
    print("See scripts/train_hmems_session.sh for the full launch script")


if __name__ == "__main__":
    main()