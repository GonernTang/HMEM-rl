"""
HMEMS Session-Based GRPO Training Entry Point

This script registers HMEMS session-based custom components (dataset, reward_manager, GRPO)
before launching the HMEMS standalone trainer.

Key difference from Mem-alpha training:
- Uses HMEMS's own RayHMEMSTrainer (not Mem-alpha's RayPPOTrainer)
- Uses HMEMSGenerationManagerWrapper (not MemoryGenerationManager)
- No external POST to consolidation server for memory operations

Usage:
    python run_hmems_session_training.py data.train_files=/path/to/train.jsonl ...
    # Or via Ray:
    ray job submit -- python run_hmems_session_training.py ...
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set PYTHONPATH BEFORE any other imports to ensure correct verl is used
# This is inherited by Ray workers
HMEMS_ROOT = Path(__file__).parent
MEMALPHA_ROOT = HMEMS_ROOT.parent / "Mem-alpha"

# Set PYTHONPATH to include Mem-alpha first (for verl)
os.environ['PYTHONPATH'] = f"{MEMALPHA_ROOT}:{MEMALPHA_ROOT}/verl:{HMEMS_ROOT}"

# Filter out conflicting verl from site-packages
sys.path = [p for p in sys.path if '/root/verl-agent/verl' not in p and 'site-packages/verl' not in p]

# Prepend Mem-alpha paths to ensure they are found first
sys.path.insert(0, str(MEMALPHA_ROOT))
sys.path.insert(0, str(MEMALPHA_ROOT / "verl"))
sys.path.insert(0, str(HMEMS_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(HMEMS_ROOT / ".env")

# Import and register HMEMS components BEFORE importing verl trainer
import hydra
from omegaconf import OmegaConf

# 1. Register custom GRPO algorithms (this also registers them globally)
from src.session_based_core_algos import register_session_based_grpo
register_session_based_grpo()
print("Registered HMEMS custom GRPO: session_grpo, session_grpo_passk")

# 2. Import HMEMS standalone trainer components
from src.standalone_trainer import RayHMEMSTrainer
from src.session_based_dataset import SessionBasedHMEMSDataset, session_based_collate_fn


def create_config_from_args(cfg):
    """Create OmegaConf config from Hydra config."""
    return cfg


@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(cfg):
    """
    HMEMS training using Hydra config.

    The config is loaded from verl/trainer/config/ppo_trainer.yaml
    and can be overridden via command line.
    """
    import json
    import ray
    import torch
    from omegaconf import OmegaConf

    # Resolve config
    OmegaConf.resolve(cfg)

    print("=" * 60)
    print("HMEMS Session-Based GRPO Training (Standalone)")
    print("=" * 60)

    # Print key config
    model_path = cfg.actor_rollout_ref.model.path
    print(f"Model: {model_path}")
    print(f"Train files: {cfg.data.train_files}")
    print(f"Val files: {cfg.data.val_files}")
    print(f"Batch sizes: train={cfg.data.train_batch_size}, val={cfg.data.val_batch_size}")
    print(f"Adv estimator (before override): {cfg.algorithm.adv_estimator}")

    # Override algorithm to use session_grpo BEFORE creating trainer
    if "session" not in cfg.algorithm.adv_estimator:
        print(f"Warning: adv_estimator is '{cfg.algorithm.adv_estimator}', overriding to 'session_grpo'")
        cfg.algorithm.adv_estimator = "session_grpo"
    print(f"Adv estimator (after override): {cfg.algorithm.adv_estimator}")
    print("=" * 60)

    # Initialize Ray (don't specify num_cpus when connecting to existing cluster)
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}},
        )

    # Import verl utilities
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager
    from src.standalone_trainer.ray_hmems_trainer import Role

    # Define worker classes
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    # Define resource pool
    global_pool_id = "global_pool"
    n_gpus_per_node = cfg.trainer.n_gpus_per_node
    nnodes = cfg.trainer.get("nnodes", 1)
    n_gpus = n_gpus_per_node * nnodes
    print(f"[DEBUG] trainer.n_gpus_per_node={n_gpus_per_node}, nnodes={nnodes}, total_n_gpus={n_gpus}")
    resource_pool_spec = {
        global_pool_id: [n_gpus],
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    # Get tokenizer
    tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path)

    # Create datasets
    train_dataset = SessionBasedHMEMSDataset(
        data_files=cfg.data.train_files if isinstance(cfg.data.train_files, list) else [cfg.data.train_files],
        tokenizer=tokenizer,
        config={
            "max_prompt_length": cfg.data.max_prompt_length,
        },
    )
    val_dataset = SessionBasedHMEMSDataset(
        data_files=cfg.data.val_files if isinstance(cfg.data.val_files, list) else [cfg.data.val_files],
        tokenizer=tokenizer,
        config={
            "max_prompt_length": cfg.data.max_prompt_length,
        },
    )

    # Create trainer
    trainer = RayHMEMSTrainer(
        config=cfg,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=session_based_collate_fn,
        device_name=cfg.trainer.get("device", "cuda"),
    )

    # Initialize workers and start training
    trainer.init_workers()
    trainer.fit()

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
