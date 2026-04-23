"""
HMEMS Generation Manager Wrapper

Wraps HMEMSGenerationManager to provide a verl-compatible interface.
This avoids using Mem-alpha's MemoryGenerationManager which sends wrong format to consolidation server.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from verl import DataProto

from src.hmems_generation import HMEMSGenerationManager, ConsolidationGenerationConfig
from src.reward_function import RewardComputer


@dataclass
class HMEMSGenerationConfig:
    """Configuration for HMEMS generation."""
    max_prompt_length: int = 2048
    max_response_length: int = 1024
    max_start_length: int = 4096
    max_obs_length: int = 512
    num_gpus: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    # HMEMS-specific
    compression_ratio_weight: float = 0.05
    # Memory server URLs
    consolidate_url: str = "http://127.0.0.1:5005/consolidate"
    respond_url: str = "http://127.0.0.1:5005/batch_process"
    # Per-turn training mode
    use_per_turn_mode: bool = False  # If True, use run_per_turn_loop instead of run_consolidation_loop
    customized_grpo_rollout_n: int = 4  # Number of session copies for per-turn training (GRPO group size)


class HMEMSGenerationManagerWrapper:
    """
    Wrapper around HMEMSGenerationManager that provides a verl-compatible interface.

    Key differences from Mem-alpha's MemoryGenerationManager:
    - Uses HMEMSGenerationManager.run_consolidation_loop() directly
    - No POST to external consolidation server for memory operations
    - Produces correct meta_info fields for SessionBasedRewardManager
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: HMEMSGenerationConfig,
        is_validation: bool = False,
    ):
        """
        Initialize HMEMS generation manager wrapper.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            actor_rollout_wg: Ray worker group for actor/rollout
            config: HMEMS generation config
            is_validation: Whether this is for validation
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # Convert to HMEMSGenerationManager config format
        hmems_config = ConsolidationGenerationConfig(
            max_turns=1,  # Single step decision for HMEMS
            max_prompt_length=config.max_prompt_length,
            max_response_length=config.max_response_length,
            num_gpus=config.num_gpus,
            temperature=config.temperature,
            consolidate_url=getattr(config, 'consolidate_url', 'http://127.0.0.1:5005/consolidate'),
            respond_url=getattr(config, 'respond_url', 'http://127.0.0.1:5005/batch_process'),
        )

        # Create reward computer
        reward_computer = RewardComputer(
            compression_ratio_weight=config.compression_ratio_weight
        )

        # Create inner HMEMSGenerationManager
        # qa_lookup is not needed - qa_pairs are passed per-session in run_generation
        self.inner = HMEMSGenerationManager(
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            config=hmems_config,
            qa_lookup=None,  # QA pairs passed directly in run_generation
            reward_computer=reward_computer,
            is_validation=is_validation,
        )

        print(f"Initialized HMEMSGenerationManagerWrapper (validation={is_validation})")

    def run_generation(
        self,
        gen_batch: DataProto,
        session_dialogue_list: List[List[Dict]],
        is_evidence_session_list: List[bool],
        qa_pairs_list: List[List[Dict]],
        sample_ids: List[str],
    ) -> DataProto:
        """
        Run HMEMS consolidation generation for a batch of sessions.

        Supports two modes:
        - use_per_turn_mode=False (default): Session-level processing for validation/test
        - use_per_turn_mode=True: Per-turn processing for training

        Args:
            gen_batch: DataProto with prompt token IDs
            session_dialogue_list: List of session dialogues
            is_evidence_session_list: List of bool (True for evidence sessions)
            qa_pairs_list: List of QA pairs per session
            sample_ids: List of sample IDs for QA lookup

        Returns:
            DataProto with consolidation results and rewards in meta_info
        """
        if self.config.use_per_turn_mode:
            return self._run_per_turn_generation(
                gen_batch=gen_batch,
                session_dialogue_list=session_dialogue_list,
                is_evidence_session_list=is_evidence_session_list,
                qa_pairs_list=qa_pairs_list,
                sample_ids=sample_ids,
            )
        else:
            return self._run_session_level_generation(
                gen_batch=gen_batch,
                session_dialogue_list=session_dialogue_list,
                is_evidence_session_list=is_evidence_session_list,
                qa_pairs_list=qa_pairs_list,
                sample_ids=sample_ids,
            )

    def _run_session_level_generation(
        self,
        gen_batch: DataProto,
        session_dialogue_list: List[List[Dict]],
        is_evidence_session_list: List[bool],
        qa_pairs_list: List[List[Dict]],
        sample_ids: List[str],
    ) -> DataProto:
        """
        Run session-level consolidation generation (for validation/test).
        """
        batch_size = len(session_dialogue_list)

        # Format session dialogues into prompts and contexts
        prompts = []
        prev_contexts = []
        new_memories = []

        for dialogue in session_dialogue_list:
            # Format dialogue as text
            formatted_dialogue = self._format_dialogue(dialogue)
            prompts.append(formatted_dialogue)
            prev_contexts.append("")  # HMEMS consolidation uses full session
            new_memories.append(formatted_dialogue)

        # Run consolidation loop via inner manager
        output = self.inner.run_consolidation_loop(
            gen_batch=gen_batch,
            new_memories=new_memories,
            prev_contexts=prev_contexts,
            sample_ids=sample_ids,
            qa_pairs_list=qa_pairs_list,
            num_gpus=self.config.num_gpus,
        )

        # Add session-level info to meta_info
        output.meta_info['is_evidence_session'] = is_evidence_session_list
        output.meta_info['qa_pairs'] = qa_pairs_list

        return output

    def _run_per_turn_generation(
        self,
        gen_batch: DataProto,
        session_dialogue_list: List[List[Dict]],
        is_evidence_session_list: List[bool],
        qa_pairs_list: List[List[Dict]],
        sample_ids: List[str],
    ) -> DataProto:
        """
        Run per-turn consolidation generation (for training).

        In per-turn mode:
        - batch_size is actually customized_grpo_rollout_n (same session repeated)
        - Each copy has independent memory state
        - We process all turns sequentially, then compute rewards
        """
        # In per-turn mode, session_dialogue_list should have customized_grpo_rollout_n identical sessions
        # We take the first one as the source and replicate it
        customized_grpo_rollout_n = self.config.customized_grpo_rollout_n

        if len(session_dialogue_list) != customized_grpo_rollout_n:
            print(f"[WARNING] Per-turn mode expects {customized_grpo_rollout_n} copies, got {len(session_dialogue_list)}")
            print(f"[INFO] Replicating single session {customized_grpo_rollout_n} times for per-turn processing")
            # Replicate the single session to create customized_grpo_rollout_n copies
            if len(session_dialogue_list) == 1:
                session_dialogue_list = [session_dialogue_list[0]] * customized_grpo_rollout_n
                is_evidence_session_list = [is_evidence_session_list[0]] * customized_grpo_rollout_n
                qa_pairs_list = [qa_pairs_list[0]] * customized_grpo_rollout_n if len(qa_pairs_list) > 0 else []
                sample_ids = [sample_ids[0]] * customized_grpo_rollout_n if len(sample_ids) > 0 else []
            else:
                # Fall back to session-level if batch size mismatch and can't replicate
                print(f"[ERROR] Cannot replicate {len(session_dialogue_list)} sessions to {customized_grpo_rollout_n}")
                return self._run_session_level_generation(
                    gen_batch, session_dialogue_list, is_evidence_session_list,
                    qa_pairs_list, sample_ids
                )

        # Use the first session's dialogue and QA pairs (all copies are identical)
        session_dialogue = session_dialogue_list[0]
        # Handle numpy arrays or other sequences that may have ambiguous truth value
        if hasattr(qa_pairs_list, '__len__') and len(qa_pairs_list) > 0:
            first_item = qa_pairs_list[0]
            # Debug: log what first_item is
            print(f"[DEBUG generation_manager] qa_pairs_list type: {type(qa_pairs_list)}, len: {len(qa_pairs_list)}, shape: {getattr(qa_pairs_list, 'shape', 'N/A')}")
            print(f"[DEBUG generation_manager] first_item type: {type(first_item)}, is_list_or_dict: {isinstance(first_item, (list, dict))}")
            try:
                print(f"[DEBUG generation_manager] first_item content: {first_item}")
            except:
                print(f"[DEBUG generation_manager] first_item: <cannot print>")
            # Handle numpy arrays - convert to list
            if isinstance(first_item, (list, dict)):
                qa_pairs = first_item
            elif hasattr(first_item, 'tolist'):  # numpy array
                qa_pairs = first_item.tolist()
            else:
                qa_pairs = []
            print(f"[DEBUG generation_manager] qa_pairs after extraction: {len(qa_pairs)} pairs")
        else:
            qa_pairs = []

        # Run per-turn loop
        output = self.inner.run_per_turn_loop(
            gen_batch=gen_batch,
            session_dialogue=session_dialogue,
            qa_pairs=qa_pairs,
            customized_grpo_rollout_n=customized_grpo_rollout_n,
        )

        # Add session-level info to meta_info
        output.meta_info['is_evidence_session'] = is_evidence_session_list
        output.meta_info['qa_pairs'] = qa_pairs_list

        # Replicate non_tensor_batch fields to match customized_grpo_rollout_n
        # This fixes the shape mismatch when compute_log_prob checks non_tensor_batch lengths
        replicated_non_tensor_batch = {}
        for key, value in gen_batch.non_tensor_batch.items():
            if isinstance(value, (list, tuple)):
                if len(value) == 1 and customized_grpo_rollout_n > 1:
                    # Single element list - replicate to customized_grpo_rollout_n
                    replicated_non_tensor_batch[key] = value * customized_grpo_rollout_n
                elif len(value) != customized_grpo_rollout_n and customized_grpo_rollout_n > 1:
                    # List with different length than expected - tile to customized_grpo_rollout_n
                    replicated_non_tensor_batch[key] = value * customized_grpo_rollout_n
                else:
                    replicated_non_tensor_batch[key] = value
            elif isinstance(value, np.ndarray):
                if len(value) != customized_grpo_rollout_n and customized_grpo_rollout_n > 1:
                    replicated_non_tensor_batch[key] = np.tile(value, (customized_grpo_rollout_n,))
                else:
                    replicated_non_tensor_batch[key] = value
            else:
                # Single value (string, int, float, etc.) - replicate to customized_grpo_rollout_n
                if customized_grpo_rollout_n > 1:
                    replicated_non_tensor_batch[key] = [value] * customized_grpo_rollout_n
                else:
                    replicated_non_tensor_batch[key] = value
        output.non_tensor_batch = replicated_non_tensor_batch

        return output

    def _format_dialogue(self, dialogue: List[Dict]) -> str:
        """Format dialogue turns as text."""
        lines = []
        for turn in dialogue:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            dia_id = turn.get('dia_id', '')
            if dia_id:
                lines.append(f"[{dia_id}] {speaker}: {text}")
            else:
                lines.append(f"{speaker}: {text}")
        return '\n'.join(lines)
