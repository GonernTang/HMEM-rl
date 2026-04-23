"""
HMEMS Consolidation Agent Reward Manager

Custom reward manager for HMEMS consolidation agent RL training,
compatible with verl's reward manager interface.

NOTE: Import verl from Mem-alpha by running via run_hmems_training.py
"""

import json
import os
import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class HMEMSConsolidationRewardManager:
    """
    Reward manager for HMEMS consolidation agent.

    Computes rewards based on:
    1. QA accuracy: Whether the model can answer questions correctly using consolidated memories
    2. Compression reward: Whether memories were effectively compressed (1 - l_m/l_c)

    The total reward is:
        reward = qa_accuracy + compression_ratio_weight * compression_reward
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        return_separate_scores: bool = False,
        compression_ratio_weight: float = 0.05,
        qa_weight: float = 1.0,
        threshold: float = None,
        **kwargs
    ):
        """
        Initialize HMEMS reward manager.

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of batches to print for debugging
            compute_score: Not used (for interface compatibility)
            reward_fn_key: Key to access data source in batch
            return_separate_scores: Whether to return separate reward scores
            compression_ratio_weight: Weight for compression reward
            qa_weight: Weight for QA reward
            threshold: Threshold for binary reward (optional)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.return_separate_scores = return_separate_scores
        self.compression_ratio_weight = compression_ratio_weight
        self.qa_weight = qa_weight
        self.threshold = threshold

    def _parse_consolidation_action(self, response_str: str) -> Dict:
        """Parse consolidation action from model response."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response_str, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                if 'action' in action_dict and action_dict['action'] in ['merge', 'augment', 'none']:
                    return action_dict
        except json.JSONDecodeError:
            pass
        return {'action': 'none'}

    def _check_answer_match(
        self,
        predicted_answer: str,
        gold_answer: str,
    ) -> float:
        """Check if predicted answer matches gold answer."""
        if not gold_answer:
            return 0.0

        pred_lower = predicted_answer.lower().strip()
        gold_lower = gold_answer.lower().strip()

        # Exact match
        if pred_lower == gold_lower:
            return 1.0

        # For numeric answers
        if isinstance(gold_answer, (int, float)):
            try:
                pred_num = float(re.search(r'\d+\.?\d*', pred_lower).group())
                gold_num = float(gold_answer)
                return 1.0 if abs(pred_num - gold_num) < 0.01 else 0.0
            except:
                pass

        # Keyword match for multi-part answers
        if ";" in gold_answer:
            keywords = [k.strip() for k in gold_answer.split(";")]
            hits = sum(1 for k in keywords if k in pred_lower)
            return hits / len(keywords)

        # Substring match
        if gold_lower in pred_lower:
            return 1.0

        # Partial match for date-like answers
        gold_words = set(gold_lower.split())
        pred_words = set(pred_lower.split())
        if gold_words and len(gold_words & pred_words) / len(gold_words) > 0.7:
            return 0.8

        return 0.0

    def _compute_qa_reward(
        self,
        predicted_answers: List[str],
        gold_answers: List[str],
    ) -> float:
        """Compute QA reward as average accuracy."""
        if len(predicted_answers) != len(gold_answers) or len(predicted_answers) == 0:
            return 0.0

        scores = [
            self._check_answer_match(pred, gold)
            for pred, gold in zip(predicted_answers, gold_answers)
        ]
        return sum(scores) / len(scores)

    def _compute_compression_reward(
        self,
        memory_content_length: int,
        original_content_length: int,
    ) -> float:
        """Compute compression reward: r = 1 - l_m / l_c"""
        if original_content_length == 0:
            return 0.0
        return 1 - memory_content_length / original_content_length

    def __call__(self, data, data_sources: list = None, return_dict: bool = False):
        """
        Compute rewards for a batch of data.

        Args:
            data: DataProto containing batch data and responses
            data_sources: List of data sources (for compatibility)
            return_dict: Whether to return rewards as dict

        Returns:
            reward_tensor or dict with rewards
        """
        # Import DataProto here to avoid circular import issues
        from verl.protocol import DataProto

        # Initialize reward tensor
        batch_size = data.batch['responses'].shape[0]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Extract data from meta_info
        questions_list = data.meta_info.get('questions_list', [])
        ground_truth_answers_list = data.meta_info.get('ground_truth_answers_list', [])
        predicted_answers_list = data.meta_info.get('predicted_answers_list', [])
        total_chunk_length = data.meta_info.get('total_chunk_length', [])
        total_memory_length = data.meta_info.get('total_memory_length', [])
        all_function_call_rewards = data.meta_info.get('all_function_call_rewards', [])

        # Compute compression rewards
        compression_ratio_reward_scores = [
            self._compute_compression_reward(mem_len, chunk_len)
            for mem_len, chunk_len in zip(total_memory_length, total_chunk_length)
        ]

        # Compute QA rewards
        qa_reward_scores = []
        for pred_list, gold_list in zip(predicted_answers_list, ground_truth_answers_list):
            qa_reward = self._compute_qa_reward(pred_list, gold_list)
            qa_reward_scores.append(qa_reward)

        # Combine rewards
        reward_scores = []
        for i in range(len(qa_reward_scores)):
            qa_r = qa_reward_scores[i]
            comp_r = compression_ratio_reward_scores[i]

            if self.threshold is not None:
                qa_r = 0.0 if qa_r < self.threshold else 1.0

            combined = self.qa_weight * qa_r + self.compression_ratio_weight * comp_r
            reward_scores.append(combined)

        # Expand rewards to match response length
        indices_in_batch = data.meta_info.get('indices_in_batch', list(range(batch_size)))

        all_reward_scores = []
        for i in indices_in_batch:
            all_reward_scores.append(reward_scores[i])

        # Fill reward tensor at the last position of each sequence
        for i in range(len(all_reward_scores)):
            response_length = data.batch['attention_mask'][i].sum().item()
            reward_tensor[i, response_length - 1] = all_reward_scores[i]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "qa_reward_scores": qa_reward_scores,
                    "compression_ratio_reward_scores": compression_ratio_reward_scores,
                    "all_function_call_rewards": all_function_call_rewards,
                }
            }
        else:
            return reward_tensor


def create_hmems_reward_manager(
    config,
    tokenizer,
    num_examine: int = 0,
    **kwargs
) -> HMEMSConsolidationRewardManager:
    """Factory function to create HMEMS reward manager."""
    return HMEMSConsolidationRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compression_ratio_weight=config.reward_model.get("compression_ratio_weight", 0.05),
        qa_weight=config.reward_model.get("qa_weight", 1.0),
        threshold=config.reward_model.get("threshold", None),
        **kwargs
    )
