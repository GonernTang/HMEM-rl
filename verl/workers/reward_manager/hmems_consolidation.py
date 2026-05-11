# Copyright 2024 HMEMS Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HMEMS Consolidation Reward Manager

Reward manager for HMEMS consolidation agent training.
Computes binary QA reward (1=correct, 0=wrong) for evidence sessions.
Non-evidence sessions get reward=0 (advantage=0 in GRPO).

Key features:
1. Binary QA accuracy reward (correct=1, wrong=0)
2. Only evidence sessions receive non-zero reward
3. Consolidation decision is parsed but not directly rewarded (future extension)
"""

import json
import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .registry import register


def _extract_answer_from_response(response: str) -> str:
    """Extract the answer from model response."""
    # Remove thinking tags if present
    if "<think>" in response and "</think>" in response:
        response = response.split("</think>")[1].strip()
    if "<think>" in response:
        response = "Empty"

    # Try to find JSON first
    try:
        json_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
        if json_match:
            # Check if it's a complete JSON object
            json_str = json_match.group()
            json.loads(json_str)  # Validate
            return json_str
    except (json.JSONDecodeError, re.error):
        pass

    # Fallback: return the response as-is
    return response.strip()


def _parse_consolidation_response(response_str: str) -> Tuple[Optional[str], Optional[Dict], Optional[List[str]]]:
    """
    Parse consolidation decision and QA answers from model response.

    Returns:
        (action, decision_dict, predicted_answers)
        - action: "merge", "augment", or "none" (or None if not found)
        - decision_dict: full parsed JSON dict (or None)
        - predicted_answers: list of predicted answer strings (or None)
    """
    try:
        # Try to find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response_str, re.DOTALL)
        if json_match:
            action_dict = json.loads(json_match.group())

            # Extract action
            action = action_dict.get('action', None)

            # Extract predicted answers (if present in response)
            predicted_answers = action_dict.get('predicted_answers', None)
            if predicted_answers is None:
                # Try alternative key names
                predicted_answers = action_dict.get('answers', None)
            if predicted_answers is None:
                predicted_answers = action_dict.get('qa_answers', None)

            return action, action_dict, predicted_answers

    except (json.JSONDecodeError, re.error) as e:
        print(f"[WARNING] Failed to parse JSON from response: {e}")

    return None, None, None


def _check_answer_match(predicted_answer: str, gold_answer: str) -> float:
    """
    Check if predicted answer matches gold answer.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
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

    # Keyword match for multi-part answers (separated by ";")
    if ";" in gold_answer:
        keywords = [k.strip() for k in gold_answer.split(";")]
        hits = sum(1 for k in keywords if k.lower() in pred_lower)
        return 1.0 if hits == len(keywords) else 0.0

    # Substring match
    if gold_lower in pred_lower:
        return 1.0

    # Partial word overlap for short answers
    gold_words = set(gold_lower.split())
    pred_words = set(pred_lower.split())
    if gold_words and len(gold_words & pred_words) / len(gold_words) > 0.8:
        return 1.0

    return 0.0


@register("hmems_consolidation")
class HMEMSConsolidationRewardManager:
    """
    Reward manager for HMEMS consolidation agent training.

    Reward design:
    - Binary QA accuracy: correct = 1, incorrect = 0
    - Only evidence sessions receive non-zero reward
    - Non-evidence sessions: reward = 0 (advantage = 0 in GRPO)

    The model outputs a JSON with:
    {
        "action": "merge" | "augment" | "none",
        "reasoning": "...",
        "target_episodic_id": int (for merge),
        "relevant_vec_ids": list (for augment),
        "merged_content": str (for merge),
        "augmented_content": str (for augment),
        "predicted_answers": [...] (optional, for QA reward)
    }
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        return_separate_scores: bool = False,
        qa_weight: float = 1.0,
        threshold: float = None,
        **kwargs
    ):
        """
        Initialize HMEMS Consolidation Reward Manager.

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of batches to print for debugging
            compute_score: Not used (for interface compatibility)
            reward_fn_key: Key to access data source in batch
            return_separate_scores: Whether to return separate reward scores
            qa_weight: Weight for QA reward (default 1.0)
            threshold: Binary threshold (if set, reward is 0 or 1)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.return_separate_scores = return_separate_scores
        self.qa_weight = qa_weight
        self.threshold = threshold

    def __call__(self, data, data_sources: list = None, return_dict: bool = False):
        """
        Compute rewards for a batch of consolidation decisions.

        Args:
            data: DataProto containing batch data and responses
            data_sources: List of data sources (for compatibility)
            return_dict: Whether to return rewards as dict

        Returns:
            reward_tensor or dict with rewards
        """
        batch_size = data.batch['responses'].shape[0]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Extract metadata
        is_evidence_session_list = data.meta_info.get('is_evidence_session', [False] * batch_size)
        ground_truth_answers_list = data.meta_info.get('ground_truth_answers_list', [[] for _ in range(batch_size)])

        # Decode all responses
        responses = self.tokenizer.batch_decode(
            data.batch['responses'],
            skip_special_tokens=True
        )

        # Compute rewards
        qa_reward_scores = []
        action_list = []
        valid_action_count = 0

        for i in range(batch_size):
            is_evidence = is_evidence_session_list[i] if i < len(is_evidence_session_list) else False
            gold_answers = ground_truth_answers_list[i] if i < len(ground_truth_answers_list) else []

            # Parse response
            action, decision_dict, predicted_answers = _parse_consolidation_response(responses[i])

            # Track action validity
            if action in ['merge', 'augment', 'none']:
                valid_action_count += 1
            action_list.append(action)

            # Compute QA reward
            if is_evidence and gold_answers:
                if predicted_answers and len(predicted_answers) == len(gold_answers):
                    # Compare each predicted answer with gold answer
                    scores = [
                        _check_answer_match(pred, gold)
                        for pred, gold in zip(predicted_answers, gold_answers)
                    ]
                    qa_reward = sum(scores) / len(scores)
                elif predicted_answers and len(predicted_answers) > 0:
                    # Partial match - compare available predictions
                    scores = [
                        _check_answer_match(pred, gold)
                        for pred, gold in zip(predicted_answers[:len(gold_answers)], gold_answers)
                    ]
                    qa_reward = sum(scores) / max(len(gold_answers), 1)
                else:
                    # No predicted answers found - reward = 0
                    qa_reward = 0.0
            else:
                qa_reward = 0.0

            # Apply binary threshold if set
            if self.threshold is not None:
                qa_reward = 0.0 if qa_reward < self.threshold else 1.0

            # Non-evidence sessions get 0 reward
            if not is_evidence:
                qa_reward = 0.0

            # Apply QA weight
            total_reward = self.qa_weight * qa_reward
            qa_reward_scores.append(qa_reward)

        # Print debug info for first few batches
        if self.num_examine > 0 and int(qa_reward_scores[0] * 100) < self.num_examine:
            for i in range(min(3, batch_size)):
                print(f"\n[DEBUG Reward Manager] Sample {i}:")
                print(f"  is_evidence: {is_evidence_session_list[i] if i < len(is_evidence_session_list) else 'N/A'}")
                print(f"  action: {action_list[i]}")
                print(f"  qa_reward: {qa_reward_scores[i]}")
                print(f"  response (first 200 chars): {responses[i][:200]}...")

        # Fill reward tensor at the last position of each sequence
        response_length = data.batch['responses'].shape[1]
        for i in range(batch_size):
            reward_tensor[i, response_length - 1] = qa_reward_scores[i]

        # For HMEMS, we only have qa_reward_scores (acc_reward_scores)
        # compression_ratio and function_call rewards are not applicable
        reward_len = len(qa_reward_scores)
        zeros = [0.0] * reward_len

        if return_dict:
            # Return via compute_reward path (used in training)
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "acc_reward_scores": qa_reward_scores,
                    "compression_ratio_reward_scores": zeros,
                    "all_function_call_rewards": zeros,
                    "all_function_call_content_rewards": zeros,
                    "is_evidence_session": is_evidence_session_list,
                    "qa_reward_scores": qa_reward_scores,
                }
            }
        elif self.return_separate_scores:
            # Validation path - return 4 values for compatibility with _validate
            # Order must match: (reward_tensor, compression_ratio_scores, function_call_scores, function_call_content_scores)
            # For HMEMS: compression_ratio_scores = qa_reward_scores (used as acc), others = zeros
            return reward_tensor, qa_reward_scores, zeros, zeros
        else:
            return reward_tensor
