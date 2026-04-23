"""
HMEMS GRPO Trainer using Ray backend.

This trainer replaces the MemoryGenerationManager with HMEMSGenerationManager
and uses session-based dataset and reward manager.

Key differences from RayPPOTrainer:
1. Uses HMEMSGenerationManagerWrapper instead of MemoryGenerationManager
2. Uses SessionBasedDataset and session_based_collate_fn
3. Uses SessionBasedRewardManager
4. Uses compute_session_grpo_advantage for selective learning (non-evidence sessions get advantage=0)
"""

import json
import os
import socket
import uuid
import requests
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking

# Import HMEMS components
from .generation_manager import HMEMSGenerationManagerWrapper, HMEMSGenerationConfig
from src.session_based_dataset import SessionBasedHMEMSDataset, session_based_collate_fn
from src.session_based_reward_manager import SessionBasedRewardManager
from src.session_based_core_algos import (
    compute_session_grpo_advantage,
    compute_copy_aware_grpo_advantage,
    register_session_based_grpo,
)


class Role(Enum):
    """Role definitions for workers."""
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """Define a resource pool specification."""
    resource_pool_spec: dict
    mapping: dict
    resource_pool_dict: dict = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values()
                    for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values()
                                   for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")


class RayHMEMSTrainer:
    """
    HMEMS GRPO Trainer with Ray backend.

    A simplified trainer for HMEMS that uses:
    - HMEMSGenerationManagerWrapper for generation (no external POST)
    - SessionBasedRewardManager for rewards
    - compute_session_grpo_advantage for selective learning
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name: str = "cuda",
    ):
        """
        Initialize HMEMS trainer.

        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            ray_worker_group_cls: Class for Ray worker groups
            processor: Optional data processor (for multimodal)
            reward_fn: Function for computing training rewards
            val_reward_fn: Function for computing validation rewards
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Function to collate data samples
            train_sampler: Sampler for training dataset
            device_name: Device for training
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.device_name = device_name
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping

        # Register custom session-based GRPO
        register_session_based_grpo()

        # Determine if using critic
        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.SESSION_GRPO,
            AdvantageEstimator.SESSION_GRPO_PASSK,
        ]:
            self.use_critic = False
        else:
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Initialize generation and reward managers
        self.generation_manager = None
        self.hmems_reward_manager = None

    def _validate_config(self):
        """Validate configuration."""
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
            minimal_bsz = n_gpus
            assert real_train_batch_size % minimal_bsz == 0, \
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal batch size ({minimal_bsz})"

        print("[HMEMS Config] Configuration validation passed!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Create train and validation dataloaders."""
        from torch.utils.data import RandomSampler, SequentialSampler
        from src.session_based_dataset import SessionBasedHMEMSDataset

        if train_dataset is None:
            train_dataset = SessionBasedHMEMSDataset(
                data_files=self.config.data.train_files if isinstance(self.config.data.train_files, list) else [self.config.data.train_files],
                tokenizer=self.tokenizer,
                config={
                    "max_prompt_length": self.config.data.max_prompt_length,
                },
            )
        if val_dataset is None:
            val_dataset = SessionBasedHMEMSDataset(
                data_files=self.config.data.val_files if isinstance(self.config.data.val_files, list) else [self.config.data.val_files],
                tokenizer=self.tokenizer,
                config={
                    "max_prompt_length": self.config.data.max_prompt_length,
                },
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            # Use simple random sampler with seed
            seed = self.config.data.get("seed", 1) or 1
            generator = torch.Generator()
            generator.manual_seed(seed)
            train_sampler = RandomSampler(
                data_source=self.train_dataset,
                generator=generator,
            )
        if collate_fn is None:
            collate_fn = session_based_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 4),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 4),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        print(f"HMEMS Trainer: train dataloader size = {len(self.train_dataloader)}, val dataloader size = {len(self.val_dataloader)}")

    def init_workers(self):
        """Initialize distributed workers using Ray backend."""
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()

        # Create actor_rollout worker class
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )

        # Create colocated worker class dict
        class_dict = {"actor_rollout": actor_rollout_cls}
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

        # Get resource pool
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)

        # Create worker group with the class dict
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name=self.device_name,
        )

        # Spawn workers
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        self.actor_rollout_wg = spawn_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        print("HMEMS Trainer: Workers initialized successfully")

    def _create_hmems_generation_manager(self) -> HMEMSGenerationManagerWrapper:
        """Create HMEMS generation manager."""
        use_per_turn_mode = self.config.data.get("use_per_turn_mode", False)
        customized_grpo_rollout_n = self.config.data.get("customized_grpo_rollout_n", 4)

        gen_config = HMEMSGenerationConfig(
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_start_length=self.config.data.get("max_start_length", 4096),
            max_obs_length=self.config.data.get("max_obs_length", 512),
            num_gpus=self.resource_pool_manager.get_n_gpus(),
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            top_p=self.config.actor_rollout_ref.rollout.top_p,
            top_k=self.config.actor_rollout_ref.rollout.top_k,
            compression_ratio_weight=self.config.reward_model.get("compression_ratio_weight", 0.05),
            consolidate_url=self.config.data.get("consolidate_url", "http://127.0.0.1:5005/consolidate"),
            respond_url=self.config.data.get("respond_url", "http://127.0.0.1:5005/batch_process"),
            use_per_turn_mode=use_per_turn_mode,
            customized_grpo_rollout_n=customized_grpo_rollout_n,
        )

        return HMEMSGenerationManagerWrapper(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=False,  # Training mode needs per-turn processing
        )

    def _create_hmems_reward_manager(self) -> SessionBasedRewardManager:
        """Create HMEMS reward manager."""
        return SessionBasedRewardManager(
            tokenizer=self.tokenizer,
            qa_weight=self.config.reward_model.get("qa_weight", 1.0),
            operation_reward_weight=self.config.reward_model.get("operation_reward_weight", 0.05),
            threshold=self.config.reward_model.get("threshold", None),
        )

    def fit(self):
        """
        Main training loop for HMEMS GRPO with full verl metrics.

        Flow (matching verl standard trainer):
        1. HMEMS generation (no external POST)
        2. compute_log_prob (for entropy)
        3. HMEMS reward computation
        4. Selective GRPO advantage computation
        5. update_actor (actual policy gradient update)
        6. Log all metrics (entropy, advantages, returns, timing, throughput)

        Note: KL penalty (use_kl_in_reward) requires reference policy setup which is not
        enabled in HMEMS by default. The use_kl_loss is handled inside update_actor.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.get("project_name", "hmems"),
            experiment_name=self.config.trainer.get("experiment_name", "hmems_session"),
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        timing_raw = {}  # For timing metrics

        # Create HMEMS generation and reward managers
        self.generation_manager = self._create_hmems_generation_manager()
        self.hmems_reward_manager = self._create_hmems_reward_manager()

        total_epochs = self.config.trainer.total_epochs
        total_batches = len(self.train_dataloader)
        n_gpus = self.resource_pool_manager.get_n_gpus()

        print(f"Starting HMEMS training: {total_epochs} epochs, {total_batches} batches per epoch")
        print(f"Using {n_gpus} GPU(s)")

        for epoch in range(total_epochs):
            # Reset memory store at the start of each new epoch (except epoch 0)
            if epoch > 0:
                try:
                    reset_url = self.config.data.get("consolidate_url", "http://127.0.0.1:5005").replace("/consolidate", "/reset")
                    resp = requests.post(reset_url, timeout=10)
                    if resp.status_code == 200:
                        print(f"[Epoch {epoch}] Memory store reset for new conversation")
                    else:
                        print(f"[Epoch {epoch}] Warning: Memory store reset returned {resp.status_code}")
                except Exception as e:
                    print(f"[Epoch {epoch}] Warning: Failed to reset memory store: {e}")

            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}

                # Separate tensors and non-tensors from batch_dict
                tensors = {}
                non_tensors = {}
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        tensors[key] = val
                    elif isinstance(val, (list, str, int, float, bool)):
                        non_tensors[key] = val
                    else:
                        non_tensors[key] = val

                # Convert batch to DataProto with separate tensors and non-tensors
                batch: DataProto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

                # Pop generation keys
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask'])
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': True,  # Need log_probs for entropy
                    'do_sample': True,
                    'temperature': self.config.actor_rollout_ref.rollout.temperature,
                }

                # Save non_tensor_batch for later
                non_tensor_batch = batch.non_tensor_batch.copy()

                # Extract session-level data from non_tensor_batch
                session_dialogue_list = batch.non_tensor_batch.get('session_dialogue', [])
                is_evidence_session_list = batch.non_tensor_batch.get('is_evidence_session', [])
                qa_pairs_list = batch.non_tensor_batch.get('qa_pairs', [])
                sample_ids = [batch.non_tensor_batch['conv_id'][i]
                             for i in range(len(batch))]

                # Run HMEMS generation
                with marked_timer('generation', timing_raw, color="yellow"):
                    gen_output = self.generation_manager.run_generation(
                        gen_batch=gen_batch,
                        session_dialogue_list=session_dialogue_list,
                        is_evidence_session_list=is_evidence_session_list,
                        qa_pairs_list=qa_pairs_list,
                        sample_ids=sample_ids,
                    )

                # Restore non_tensor_batch
                # For per-turn mode, generation_manager already sets non_tensor_batch correctly
                # Skip replication to avoid overwriting with incorrectly-replicated data
                if not self.generation_manager.config.use_per_turn_mode:
                    gen_output.non_tensor_batch = non_tensor_batch.copy()

                # Check if per-turn mode is enabled
                use_per_turn_mode = self.config.data.get("use_per_turn_mode", False)

                # ========== Full verl metrics collection ( Steps 2-7 ) ==========

                # Compute response mask (needed for entropy and metrics)
                responses = gen_output.batch.get('responses')
                if responses is not None:
                    response_length = responses.size(1)
                    attention_mask = gen_output.batch['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    gen_output.batch['response_mask'] = response_mask
                else:
                    response_mask = gen_output.batch['attention_mask']

                # Step 2: Compute log_prob and entropy
                with marked_timer('old_log_prob', timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_output)
                    if "entropys" in old_log_prob.batch:
                        entropys = old_log_prob.batch["entropys"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.get("loss_agg_mode", "mean")
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        old_log_prob.batch.pop("entropys")
                    gen_output = gen_output.union(old_log_prob)

                # Step 3: Compute reference log_prob for KL penalty (if using reference policy)
                if self.use_reference_policy:
                    with marked_timer('ref', timing_raw, color="olive"):
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(gen_output)
                        gen_output = gen_output.union(ref_log_prob)

                # Step 4: Compute rewards via HMEMS reward manager
                with marked_timer('reward', timing_raw, color="green"):
                    if use_per_turn_mode and 'turn_rewards_list' in gen_output.meta_info:
                        # Per-turn mode: use pre-computed rewards from generation
                        turn_rewards_list = gen_output.meta_info['turn_rewards_list']
                        # Build reward tensor from turn_rewards_list
                        # Each copy has rewards for each turn, we use the first copy's rewards for simplicity
                        # TODO: properly aggregate across copies
                        customized_grpo_rollout_n = gen_output.meta_info.get('customized_grpo_rollout_n', 4)
                        num_turns = gen_output.meta_info.get('num_turns', 1)

                        # Flatten: use first copy's rewards as the reward signal
                        # Shape: (customized_grpo_rollout_n, num_turns) -> we use mean across copies
                        reward_tensor = torch.tensor([
                            np.mean([turn_rewards_list[c].get(did, 0.0)
                                    for c in range(customized_grpo_rollout_n)])
                            for did in gen_output.meta_info.get('all_dia_ids', [])
                        ]).unsqueeze(0).expand(customized_grpo_rollout_n, -1)

                        reward_extra_info = {'turn_rewards_list': turn_rewards_list}
                    else:
                        # Session-level mode: compute rewards via HMEMS reward manager
                        reward_result = self.hmems_reward_manager(gen_output, return_dict=True)
                        reward_tensor = reward_result['reward_tensor']
                        reward_extra_info = reward_result.get('reward_extra_info', {})

                # Assign token-level scores
                gen_output.batch['rm_scores'] = reward_tensor
                gen_output.batch['token_level_scores'] = reward_tensor

                # Add uid for GRPO grouping
                uids = np.array([str(uuid.uuid4()) for _ in range(len(gen_output))], dtype=object)
                gen_output.non_tensor_batch['uid'] = uids

                # Set token_level_rewards (use_kl_in_reward would apply KL penalty here if enabled)
                gen_output.batch['token_level_rewards'] = reward_tensor

                # Step 5: Compute advantage using session-based GRPO
                with marked_timer('adv', timing_raw, color="brown"):
                    indices = np.arange(len(gen_output))
                    if use_per_turn_mode and 'turn_rewards_list' in gen_output.meta_info:
                        """
                        Per-turn mode advantage computation:

                        Data structure:
                        - customized_grpo_rollout_n copies of the same session (each with independent memory state)
                        - Each copy processes num_turns sequentially
                        - Each copy produces a reward based on QA answering

                        Reward broadcast flow:
                        1. Each copy[i] gets reward[i] based on its memory state answering QA
                        2. Each copy's reward is broadcast to ALL its turns (turn_rewards_list)
                        3. For GRPO: average reward across copies is computed per turn
                           This measures how well this turn's actions led to good final rewards

                        The turn_rewards_list structure:
                        - turn_rewards_list[copy_idx][dia_id] = reward for that copy's turn

                        GRPO advantage computation:
                        - For each turn position, average reward across copies
                        - Then expand to all copies (each copy's same turn gets same advantage)
                        - This measures relative performance of different turn decisions
                        """
                        customized_grpo_rollout_n = gen_output.meta_info.get('customized_grpo_rollout_n', 4)
                        num_turns = gen_output.meta_info.get('num_turns', 1)
                        all_dia_ids = gen_output.meta_info.get('all_dia_ids', [])
                        all_responses = gen_output.meta_info.get('all_responses', [])

                        # Compute GRPO advantage per turn across copies
                        # Each copy gets its own reward for each turn
                        # GRPO advantage = (individual_reward - mean_across_copies) / (std_across_copies + epsilon)
                        turn_rewards_list = gen_output.meta_info['turn_rewards_list']

                        # Build per-copy reward tensor: shape (customized_grpo_rollout_n, num_turns)
                        per_copy_rewards = torch.tensor([
                            [turn_rewards_list[c].get(dia_id, 0.0)
                             for dia_id in all_dia_ids]
                            for c in range(customized_grpo_rollout_n)
                        ]).float()

                        # GRPO normalization across copies for each turn
                        # mean and std across copies for each turn
                        mean_per_turn = per_copy_rewards.mean(dim=0)  # shape (num_turns,)
                        std_per_turn = per_copy_rewards.std(dim=0) + 1e-6  # shape (num_turns,)

                        # Compute advantage per copy per turn: (reward - mean) / std
                        # turn_advantages[copy, turn] = (per_copy_rewards[copy, turn] - mean_per_turn[turn]) / std_per_turn[turn]
                        turn_advantages = (per_copy_rewards - mean_per_turn.unsqueeze(0)) / std_per_turn.unsqueeze(0)
                        turn_returns = turn_advantages.clone()

                        # Expand per-turn advantages to token-level
                        # We need to expand to the ORIGINAL response shape (from generation)
                        # so that log_probs and advantages have compatible shapes for update_actor
                        if all_responses and len(all_responses) > 0 and len(all_responses[0]) == num_turns:
                            # Get the original response shape from generation
                            original_responses = gen_output.batch['responses']
                            orig_shape = original_responses.shape  # (customized_grpo_rollout_n, max_response_len)

                            # Get per-turn response lengths from first copy
                            turn_response_lens = [len(all_responses[0][turn_idx]['gen_output'])
                                                 for turn_idx in range(num_turns)]

                            # Create expanded advantages matching original response shape
                            expanded_advantages = torch.zeros(orig_shape, dtype=torch.float32)
                            expanded_returns = torch.zeros(orig_shape, dtype=torch.float32)
                            expanded_response_mask = torch.zeros(orig_shape, dtype=torch.float32)

                            # Expand each turn's advantage to its tokens
                            # Use the same expansion for all copies (assuming similar response lengths)
                            current_pos = 0
                            for turn_idx, turn_len in enumerate(turn_response_lens):
                                end_pos = current_pos + turn_len
                                if end_pos <= orig_shape[1]:
                                    expanded_advantages[:, current_pos:end_pos] = turn_advantages[:, turn_idx:turn_idx + 1].expand(-1, turn_len)
                                    expanded_returns[:, current_pos:end_pos] = turn_returns[:, turn_idx:turn_idx + 1].expand(-1, turn_len)
                                    expanded_response_mask[:, current_pos:end_pos] = 1.0
                                else:
                                    # This turn extends beyond original response length
                                    # Fill what we can and stop
                                    remaining = orig_shape[1] - current_pos
                                    if remaining > 0:
                                        expanded_advantages[:, current_pos:current_pos + remaining] = turn_advantages[:, turn_idx:turn_idx + 1].expand(-1, remaining)
                                        expanded_returns[:, current_pos:current_pos + remaining] = turn_returns[:, turn_idx:turn_idx + 1].expand(-1, remaining)
                                        expanded_response_mask[:, current_pos:current_pos + remaining] = 1.0
                                    break
                                current_pos = end_pos

                            # For positions beyond the expanded length (if any), advantage stays 0
                            # These correspond to padding or tokens beyond our turn tracking

                            advantages = expanded_advantages
                            returns = expanded_returns
                            response_mask = expanded_response_mask
                        else:
                            # Fallback: use turn-level advantages as-is (will have shape mismatch with update_actor)
                            print(f"[WARNING] Per-turn mode: all_responses not available, using turn-level advantages")
                            advantages = turn_advantages
                            returns = turn_returns
                            response_mask = torch.ones(customized_grpo_rollout_n, num_turns)
                    else:
                        # Session-level mode
                        advantages, returns = compute_session_grpo_advantage(
                            token_level_rewards=reward_tensor,
                            response_mask=response_mask,
                            index=indices,
                            is_evidence_session=np.array(is_evidence_session_list, dtype=bool),
                            config=self.config.algorithm,
                        )

                    gen_output.batch['advantages'] = advantages
                    gen_output.batch['returns'] = returns

                # Step 7: Compute data metrics (advantages, returns, response_length, etc.)
                with marked_timer('compute_metrics', metrics):
                    batch_size = len(gen_output)

                    if use_per_turn_mode:
                        # Per-turn mode metrics
                        # Use raw rewards (mean across copies) for monitoring, not normalized advantages
                        raw_reward_scores = reward_tensor.cpu().numpy()  # Mean reward per turn
                        metrics['train/mean_reward'] = np.mean(raw_reward_scores)
                        metrics['train/max_reward'] = np.max(raw_reward_scores)
                        metrics['train/min_reward'] = np.min(raw_reward_scores)
                        # Also track normalized advantages for debugging
                        advantage_scores = advantages.cpu().numpy()
                        metrics['train/mean_advantage'] = np.mean(advantage_scores)
                        metrics['train/num_turns'] = gen_output.meta_info.get('num_turns', 0)
                        metrics['train/customized_grpo_rollout_n'] = gen_output.meta_info.get('customized_grpo_rollout_n', 0)

                        # Compute response_length metrics manually for per-turn mode
                        # In per-turn mode, responses are concatenated across turns
                        responses = gen_output.batch.get('responses')
                        if responses is not None:
                            # Response length is the actual length of generated responses
                            # responses shape: (customized_grpo_rollout_n, max_response_len)
                            # We compute mean/max/min across the batch
                            actual_response_lens = []
                            for i in range(responses.shape[0]):
                                # Find the last non-padding token
                                response_row = responses[i]
                                non_padding = (response_row != self.tokenizer.pad_token_id).sum().item()
                                actual_response_lens.append(non_padding)

                            response_lens_array = np.array(actual_response_lens, dtype=np.float32)
                            metrics['response_length/mean'] = np.mean(response_lens_array)
                            metrics['response_length/max'] = np.max(response_lens_array)
                            metrics['response_length/min'] = np.min(response_lens_array)
                            metrics['response_length/clip_ratio'] = 0.0  # No clipping in per-turn

                            # Also compute prompt_length metrics
                            attention_mask = gen_output.batch.get('attention_mask')
                            if attention_mask is not None:
                                prompt_lens = attention_mask.sum(-1).float() - torch.from_numpy(response_lens_array).float().to(attention_mask.device)
                                prompt_lens_np = prompt_lens.cpu().numpy()
                                metrics['prompt_length/mean'] = np.mean(prompt_lens_np)
                                metrics['prompt_length/max'] = np.max(prompt_lens_np)
                                metrics['prompt_length/min'] = np.min(prompt_lens_np)
                                metrics['prompt_length/clip_ratio'] = 0.0
                    else:
                        # Session-level mode metrics
                        reward_scores = reward_tensor.sum(-1).cpu().numpy()
                        metrics['train/mean_reward'] = np.mean(reward_scores)
                        metrics['train/max_reward'] = np.max(reward_scores)
                        metrics['train/min_reward'] = np.min(reward_scores)
                        metrics['train/evidence_ratio'] = np.mean(is_evidence_session_list)

                    # Add verl standard data metrics (skip for per-turn mode - incompatible shapes)
                    # For per-turn mode, we already computed basic metrics above
                    if not use_per_turn_mode:
                        data_metrics = compute_data_metrics(gen_output, use_critic=False)
                        metrics.update(data_metrics)

                    # Add timing metrics
                    timing_metrics = compute_timing_metrics(gen_output, timing_raw)
                    metrics.update(timing_metrics)

                    # Add throughput metrics (optional - skip if timing not available)
                    gen_output.meta_info['global_token_num'] = [
                        int(gen_output.batch['attention_mask'].sum()) for _ in range(len(gen_output))
                    ]
                    if 'step' in timing_raw and timing_raw['step'] > 0:
                        throughput_metrics = compute_throughout_metrics(gen_output, timing_raw, n_gpus)
                        metrics.update(throughput_metrics)
                    else:
                        # Fallback throughput calculation without timing
                        metrics["perf/total_num_tokens"] = int(gen_output.batch['attention_mask'].sum())
                        metrics["perf/time_per_step"] = 0.0
                        metrics["perf/throughput"] = 0.0

                # Step 8: Update actor (actual policy gradient update)
                # Per-turn mode now supported if advantages were expanded to token-level
                skip_update = False
                if use_per_turn_mode:
                    # Check if we successfully expanded advantages to token-level
                    all_responses = gen_output.meta_info.get('all_responses', [])
                    if not all_responses or len(all_responses) == 0:
                        print(f"[PER-TURN] Skipping update_actor (all_responses not available)")
                        skip_update = True
                    elif len(all_responses[0]) != gen_output.meta_info.get('num_turns', 0):
                        print(f"[PER-TURN] Skipping update_actor (turn count mismatch)")
                        skip_update = True

                if not skip_update:
                    with marked_timer('update_actor', timing_raw, color="red"):
                        gen_output.meta_info['multi_turn'] = self.config.actor_rollout_ref.rollout.get('multi_turn', {}).get('enable', False)
                        actor_output = self.actor_rollout_wg.update_actor(gen_output)
                        if actor_output and hasattr(actor_output, 'meta_info') and 'metrics' in actor_output.meta_info:
                            actor_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                            metrics.update(actor_metrics)
                else:
                    if use_per_turn_mode:
                        print(f"[PER-TURN] Skipping update_actor (batch shape incompatible with verl)")

                # ========== End full metrics collection ==========

                # Logging
                logger.log(data=metrics, step=self.global_steps)

                if batch_idx % self.config.trainer.get("test_freq", 1) == 0:
                    # Print comprehensive metrics
                    entropy_str = f"Entropy: {metrics.get('actor/entropy', 0):.4f}"
                    # For per-turn mode, show our computed GRPO advantage; for other modes show critic advantage
                    if use_per_turn_mode:
                        adv_mean_str = f"GRPO-Adv: {metrics.get('train/mean_advantage', 0):.4f}"
                    else:
                        adv_mean_str = f"Adv: {metrics.get('critic/advantages/mean', 0):.4f}"
                    ret_mean_str = f"Ret: {metrics.get('critic/returns/mean', 0):.4f}"
                    resp_len_str = f"RespLen: {metrics.get('response_length/mean', 0):.1f}"
                    throughput_str = f"Throughput: {metrics.get('perf/throughput', 0):.1f} tok/s/GPU"

                    print(f"Epoch {epoch}/{total_epochs}, Batch {batch_idx}/{total_batches}, "
                          f"Step {self.global_steps}, Mean Reward: {metrics['train/mean_reward']:.4f}, "
                          f"{entropy_str} {adv_mean_str} {ret_mean_str} {resp_len_str} {throughput_str}")

                self.global_steps += 1

            # End of epoch
            print(f"Completed epoch {epoch + 1}/{total_epochs}")

        print("HMEMS training completed!")

    def _validate(self):
        """
        Validation loop for HMEMS.

        Runs HMEMS generation on validation data and computes rewards.
        """
        reward_tensor_lst = []
        qa_reward_scores_lst = []
        operation_reward_scores_lst = []
        is_evidence_session_lst = []

        total_batch = len(self.val_dataloader)

        print(f"Starting HMEMS validation with {total_batch} batches")

        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            print(f"Validation Batch {batch_idx}/{total_batch}...")

            # Convert batch to DataProto
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # Pop generation keys
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,  # Validation uses greedy decoding
            }

            # Extract session-level data
            session_dialogue_list = batch.non_tensor_batch.get('session_dialogue', [])
            is_evidence_session_list = batch.non_tensor_batch.get('is_evidence_session', [])
            qa_pairs_list = batch.non_tensor_batch.get('qa_pairs', [])
            sample_ids = [f"{batch.non_tensor_batch['conv_id'][i]}_{batch.non_tensor_batch['session_name'][i]}"
                          for i in range(len(batch))]

            # Run HMEMS generation
            gen_output = self.generation_manager.run_generation(
                gen_batch=gen_batch,
                session_dialogue_list=session_dialogue_list,
                is_evidence_session_list=is_evidence_session_list,
                qa_pairs_list=qa_pairs_list,
                sample_ids=sample_ids,
            )

            # Restore non_tensor_batch
            # For per-turn mode with replicated sessions, replicate non_tensor_batch fields to match customized_grpo_rollout_n
            if self.generation_manager.config.use_per_turn_mode:
                replicated_non_tensor_batch = {}
                customized_grpo_rollout_n = self.generation_manager.config.customized_grpo_rollout_n
                for key, value in batch.non_tensor_batch.items():
                    if isinstance(value, (list, tuple)):
                        if len(value) == 1 and customized_grpo_rollout_n > 1:
                            replicated_non_tensor_batch[key] = value * customized_grpo_rollout_n
                        else:
                            replicated_non_tensor_batch[key] = value
                    else:
                        replicated_non_tensor_batch[key] = value
                gen_output.non_tensor_batch = replicated_non_tensor_batch
            else:
                gen_output.non_tensor_batch = batch.non_tensor_batch.copy()

            # Compute rewards
            reward_result = self.hmems_reward_manager(gen_output, return_dict=True)
            reward_tensor = reward_result['reward_tensor']
            reward_extra_info = reward_result.get('reward_extra_info', {})

            # Collect metrics
            reward_tensor_lst.append(reward_tensor)
            qa_reward_scores_lst.extend(reward_extra_info.get('qa_reward_scores', []))
            operation_reward_scores_lst.extend(reward_extra_info.get('operation_reward_scores', []))
            is_evidence_session_lst.extend(is_evidence_session_list)

        # Aggregate metrics
        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        qa_reward_scores = np.array(qa_reward_scores_lst)
        operation_reward_scores = np.array(operation_reward_scores_lst)
        is_evidence_session_arr = np.array(is_evidence_session_lst)

        # Compute mean rewards for evidence vs non-evidence sessions
        evidence_mask = is_evidence_session_arr
        non_evidence_mask = ~evidence_mask

        metrics = {}
        metrics['val/mean_reward'] = np.mean(reward_tensor.numpy())
        metrics['val/mean_reward_evidence'] = np.mean(reward_tensor[evidence_mask].numpy()) if evidence_mask.any() else 0.0
        metrics['val/mean_reward_non_evidence'] = np.mean(reward_tensor[non_evidence_mask].numpy()) if non_evidence_mask.any() else 0.0
        metrics['val/mean_qa_reward'] = np.mean(qa_reward_scores) if len(qa_reward_scores) > 0 else 0.0
        metrics['val/mean_operation_reward'] = np.mean(operation_reward_scores) if len(operation_reward_scores) > 0 else 0.0
        metrics['val/evidence_ratio'] = np.mean(evidence_mask)

        print(f"Validation Results: Mean Reward = {metrics['val/mean_reward']:.4f}, "
              f"Evidence Mean = {metrics['val/mean_reward_evidence']:.4f}, "
              f"QA Reward = {metrics['val/mean_qa_reward']:.4f}")

        return metrics
