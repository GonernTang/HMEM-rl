"""
HMEMS GRPO Training Entry Point

This script registers HMEMS custom components (dataset, reward_manager)
before launching the verl trainer.

Usage:
    python run_hmems_training.py --help
    # Or via Ray:
    ray job submit -- python run_hmems_training.py ...
"""

import sys
from pathlib import Path

# Add verl from Mem-alpha to path (MUST be before any verl imports)
sys.path.insert(0, str(Path(__file__).parent / ".." / "Mem-alpha"))

# Import and register HMEMS reward manager BEFORE importing verl.trainer
from src.hmems_reward_manager import HMEMSConsolidationRewardManager
from verl.workers.reward_manager import register

# Register HMEMS reward manager
register("hmems_naive")(HMEMSConsolidationRewardManager)
print("Registered HMEMS reward manager: hmems_naive")

# Now import and run verl trainer
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
