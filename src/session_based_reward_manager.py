"""
HMEMS Session-Based Reward Manager

Custom reward manager for HMEMS session-level RL training.

Key differences from consolidation reward manager:
1. Rewards are computed at session level, not turn level
2. Non-evidence sessions get advantage = 0 (no policy gradient)
3. QA reward is computed from session's qa_pairs
4. Memory operations reward based on consolidation decisions

In RL training:
- For evidence sessions: compute reward based on QA accuracy
- For non-evidence sessions: advantage = 0 (no policy gradient)
"""

import json
import os
import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class SessionBasedRewardManager:
    """
    Reward manager for HMEMS session-based training.

    Computes rewards based on:
    1. QA accuracy: Whether the model can answer questions correctly
    2. Memory operation reward: Whether consolidation decisions are appropriate

    Key design:
    - Non-evidence sessions: advantage = 0 (no policy gradient)
    - Evidence sessions: full reward computation
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        return_separate_scores: bool = False,
        qa_weight: float = 1.0,
        operation_reward_weight: float = 0.05,
        threshold: float = None,
        **kwargs
    ):
        """
        Initialize session-based reward manager.

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of batches to print for debugging
            compute_score: Not used (for interface compatibility)
            reward_fn_key: Key to access data source in batch
            return_separate_scores: Whether to return separate reward scores
            qa_weight: Weight for QA reward
            operation_reward_weight: Weight for memory operation reward
            threshold: Threshold for binary reward (optional)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.return_separate_scores = return_separate_scores
        self.qa_weight = qa_weight
        self.operation_reward_weight = operation_reward_weight
        self.threshold = threshold

    def _parse_memory_operations(self, response_str: str) -> List[Dict]:
        """Parse memory operations from model response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_str, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                if 'memory_operations' in action_dict:
                    return action_dict['memory_operations']
                elif 'action' in action_dict:
                    return [action_dict]
        except (json.JSONDecodeError, re.error):
            pass
        return []

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
                match = re.search(r'\d+\.?\d*', pred_lower)
                if match:
                    pred_num = float(match.group())
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

    def _compute_operation_reward(
        self,
        operations: List[Dict],
        session_has_qa: bool,
    ) -> float:
        """
        Compute memory operation reward.

        For sessions with QA: reward appropriate consolidation actions
        For sessions without QA: reward is 0 (no policy gradient)
        """
        if not operations:
            return 0.0

        # Check if operations are valid
        valid_actions = {'merge', 'augment', 'none'}
        valid_count = sum(1 for op in operations if op.get('action', '') in valid_actions)

        if valid_count == 0:
            return 0.0

        # Base reward for having valid operations
        base_reward = valid_count / len(operations)

        return base_reward

    def __call__(self, data, data_sources: list = None, return_dict: bool = False):
        """
        Compute rewards for a batch of session data.

        Flow:
        1. Use memory_server_rewards if available (from LLM-as-judge)
        2. Otherwise compute QA reward from predicted_answers and ground_truth

        Args:
            data: DataProto containing batch data and responses
            data_sources: List of data sources (for compatibility)
            return_dict: Whether to return rewards as dict

        Returns:
            reward_tensor or dict with rewards
        """
        from verl.protocol import DataProto

        # Initialize reward tensor
        batch_size = data.batch['responses'].shape[0]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Extract metadata
        is_evidence_session_list = data.meta_info.get('is_evidence_session', [False] * batch_size)
        qa_pairs_list = data.meta_info.get('qa_pairs', [[] for _ in range(batch_size)])
        ground_truth_answers_list = data.meta_info.get('ground_truth_answers_list', [[] for _ in range(batch_size)])
        predicted_answers_list = data.meta_info.get('predicted_answers_list', [[] for _ in range(batch_size)])
        memory_server_rewards = data.meta_info.get('memory_server_rewards', None)

        # Debug output
        print(f"[DEBUG Reward Manager] batch_size={batch_size}")
        print(f"[DEBUG Reward Manager] meta_info keys={list(data.meta_info.keys())}")
        print(f"[DEBUG Reward Manager] is_evidence_session_list={is_evidence_session_list}")
        print(f"[DEBUG Reward Manager] predicted_answers_list lengths={[len(x) for x in predicted_answers_list]}")
        print(f"[DEBUG Reward Manager] ground_truth_answers_list lengths={[len(x) for x in ground_truth_answers_list]}")
        print(f"[DEBUG Reward Manager] memory_server_rewards type={type(memory_server_rewards)}")
        if memory_server_rewards is not None:
            print(f"[DEBUG Reward Manager] memory_server_rewards lengths={[len(x) for x in memory_server_rewards]}")

        # Compute rewards
        qa_reward_scores = []
        total_reward_scores = []

        for i in range(batch_size):
            is_evidence = is_evidence_session_list[i] if i < len(is_evidence_session_list) else False
            gold_answers = ground_truth_answers_list[i] if i < len(ground_truth_answers_list) else []
            pred_answers = predicted_answers_list[i] if i < len(predicted_answers_list) else []

            # Priority 1: Use memory_server_rewards if available (from LLM-as-judge)
            if memory_server_rewards is not None and i < len(memory_server_rewards):
                server_rewards = memory_server_rewards[i]
                if isinstance(server_rewards, list) and len(server_rewards) > 0:
                    # Average the per-QA rewards from LLM judge
                    qa_reward = sum(server_rewards) / len(server_rewards)
                else:
                    qa_reward = float(server_rewards) if server_rewards else 0.0
            # Priority 2: Compute QA reward locally
            elif is_evidence and gold_answers and pred_answers:
                qa_reward = self._compute_qa_reward(pred_answers, gold_answers)
            else:
                qa_reward = 0.0

            # Binary threshold reward if configured
            if self.threshold is not None and is_evidence:
                qa_reward = 0.0 if qa_reward < self.threshold else 1.0

            # Non-evidence sessions get advantage = 0 (no policy gradient)
            if not is_evidence:
                qa_reward = 0.0

            # Only QA reward
            total_reward = self.qa_weight * qa_reward

            qa_reward_scores.append(qa_reward)
            total_reward_scores.append(total_reward)

        # Fill reward tensor at the last position of each sequence
        indices_in_batch = data.meta_info.get('indices_in_batch', list(range(batch_size)))

        for i, idx in enumerate(indices_in_batch):
            if idx < len(total_reward_scores):
                # Use response length, not full sequence length
                response_length = data.batch['responses'].shape[1]
                reward_tensor[i, response_length - 1] = total_reward_scores[idx]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "qa_reward_scores": qa_reward_scores,
                    "is_evidence_session": is_evidence_session_list,
                    "total_reward_scores": total_reward_scores,
                }
            }
        else:
            return reward_tensor


def create_session_based_reward_manager(
    config,
    tokenizer,
    num_examine: int = 0,
    **kwargs
) -> SessionBasedRewardManager:
    """Factory function to create session-based reward manager."""
    return SessionBasedRewardManager(
        tokenizer=tokenizer,
        num_examine=num_examine,
        qa_weight=config.reward_model.get("qa_weight", 1.0),
        operation_reward_weight=config.reward_model.get("operation_reward_weight", 0.05),
        threshold=config.reward_model.get("threshold", None),
        **kwargs
    )