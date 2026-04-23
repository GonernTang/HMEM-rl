"""
HMEMS Consolidation Agent Training Script

This script trains the consolidation agent using GRPO algorithm via verl.
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

    # Memory server
    parser.add_argument("--memory_server_url", type=str, default="http://localhost:5005/batch_process")

    # Other
    parser.add_argument("--enable_thinking", type=bool, default=False)
    parser.add_argument("--offload", type=bool, default=True)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1)

    return parser.parse_args()


def build_prompt_template() -> str:
    """
    Build the prompt template for consolidation agent.
    """
    return """You are a memory consolidation agent. Your task is to decide how to handle new conversation memories.

## Memory System
The system has two types of memories:
1. Episodic Memory: High-level summaries of related conversation events
2. Vector Memory: Raw conversation turns stored for retrieval

## Input Format
You will receive:
- New conversation turn to store
- Relevant episodic memories (if any)
- Relevant vector memories (if any)

## Your Task
Decide which action to take:

1. **merge**: If the new memory is related to an existing episodic memory, merge them
2. **augment**: If multiple vector memories are relevant, create a new episodic memory
3. **none**: If no relevant memories found, store as raw vector memory

## Output Format
Respond with JSON:
{
    "action": "merge" | "augment" | "none",
    "reasoning": "why you chose this action",
    "merged_content": "<for merge: combined content>",
    "augmented_content": "<for augment: new episodic content>"
}

Now process the following memory:

New Conversation:
{new_memory}

Previous Context:
{prev_context}

Relevant Episodic Memories:
{episodic_memories}

Relevant Vector Memories:
{vec_memories}

Your decision:"""


class ConsolidationDataset:
    """Dataset for consolidation agent training."""

    def __init__(self, data_path: str, qa_lookup_path: str):
        import pandas as pd

        self.data = pd.read_parquet(data_path)
        with open(qa_lookup_path, "r") as f:
            self.qa_lookup = json.load(f)

        self.prompt_template = build_prompt_template()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row["sample_id"]
        new_memory = row["new_memory"]
        prev_context = row["prev_context"]
        qa_pairs = self.qa_lookup.get(sample_id, [])

        episodic_memories = "None found"
        vec_memories = "None found"

        prompt = self.prompt_template.format(
            new_memory=new_memory,
            prev_context=prev_context if prev_context else "(no previous context)",
            episodic_memories=episodic_memories,
            vec_memories=vec_memories,
        )

        return {
            "prompt": prompt,
            "new_memory": new_memory,
            "sample_id": sample_id,
            "qa_pairs": qa_pairs,
            "n_turns": row["n_turns"],
        }


def create_dataloader(dataset: ConsolidationDataset, batch_size: int, shuffle: bool = True):
    """Create DataLoader for the dataset."""

    def collate_fn(batch):
        return {
            "prompts": [item["prompt"] for item in batch],
            "new_memory": [item["new_memory"] for item in batch],
            "sample_id": [item["sample_id"] for item in batch],
            "qa_pairs": [item["qa_pairs"] for item in batch],
            "n_turns": [item["n_turns"] for item in batch],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def main():
    args = parse_args()

    print("=" * 60)
    print("HMEMS Consolidation Agent Training")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Compression ratio weight: {args.compression_ratio_weight}")
    print("=" * 60)

    if not os.path.exists(args.train_data):
        print(f"Error: Training data not found at {args.train_data}")
        print("Please run: python scripts/preprocess_locomo.py")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = hf_tokenizer(args.model_path)

    print("Loading datasets...")
    train_dataset = ConsolidationDataset(args.train_data, args.qa_lookup)
    val_dataset = ConsolidationDataset(args.val_data, args.qa_lookup)

    train_loader = create_dataloader(train_dataset, args.train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.val_batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

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
            "compression_ratio_weight": args.compression_ratio_weight,
            "reward_manager": "naive",
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
            "default_local_dir": f"./checkpoints/{args.model_name}",
            "resume_mode": "auto",
        },
        "customized_grpo_rollout_n": args.customized_grpo_rollout_n,
        "max_turns": 5,
        "enable_thinking": args.enable_thinking,
        "respond_url": args.memory_server_url,
    }

    config_path = f"./configs/{args.model_name}_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\nConfig saved to {config_path}")
    print("\nTo run full distributed training, use verl.trainer.main_ppo with Ray cluster")
    print("See scripts/train_hmems_grpo.sh for the full launch script")


if __name__ == "__main__":
    main()
