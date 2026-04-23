"""
HMEMS Consolidation Agent Reward Function

This module implements the reward computation for RL training:
1. QA accuracy reward: Based on downstream question answering performance
2. Compression reward: Based on memory compression ratio
"""

from typing import Dict, List, Any, Optional
import re
import json


class RewardComputer:
    """Computes rewards for consolidation agent training."""

    def __init__(
        self,
        compression_ratio_weight: float = 0.05,
        qa_weight: float = 1.0,
    ):
        """
        Initialize reward computer.

        Args:
            compression_ratio_weight: Weight for compression reward
            qa_weight: Weight for QA accuracy reward
        """
        self.compression_ratio_weight = compression_ratio_weight
        self.qa_weight = qa_weight

    def compute_compression_reward(
        self,
        memory_content_length: int,
        original_content_length: int,
    ) -> float:
        """
        Compute compression reward: r = 1 - l_m / l_c

        Args:
            memory_content_length: Length of consolidated memory
            original_content_length: Length of original content

        Returns:
            Compression reward in range (-inf, 1]
        """
        if original_content_length == 0:
            return 0.0
        return 1 - memory_content_length / original_content_length

    def check_answer_match(
        self,
        predicted_answer: str,
        gold_answer: str,
        data_source: str = "default",
    ) -> float:
        """
        Check if predicted answer matches gold answer.

        Supports multiple matching strategies based on data source.
        """
        if not gold_answer:
            return 0.0

        pred_lower = predicted_answer.lower().strip()

        # For numeric answers - check before string operations
        if isinstance(gold_answer, (int, float)):
            try:
                pred_num = float(re.search(r'\d+\.?\d*', pred_lower).group())
                gold_num = float(gold_answer)
                return 1.0 if abs(pred_num - gold_num) < 0.01 else 0.0
            except:
                pass

        gold_lower = gold_answer.lower().strip()

        # Exact match
        if pred_lower == gold_lower:
            return 1.0

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

    def compute_qa_reward(
        self,
        predicted_answers: List[str],
        gold_answers: List[str],
        questions: List[str],
    ) -> float:
        """
        Compute QA reward as average accuracy across questions.

        Args:
            predicted_answers: List of model predicted answers
            gold_answers: List of gold standard answers
            questions: List of questions (for context)

        Returns:
            Average accuracy score
        """
        if len(predicted_answers) != len(gold_answers):
            return 0.0

        scores = [
            self.check_answer_match(pred, gold)
            for pred, gold in zip(predicted_answers, gold_answers)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def compute_total_reward(
        self,
        qa_scores: float,
        memory_length: int,
        original_length: int,
    ) -> float:
        """
        Compute total reward combining QA and compression.

        Args:
            qa_scores: QA accuracy score
            memory_length: Consolidated memory length
            original_length: Original content length

        Returns:
            Total weighted reward
        """
        compression_reward = self.compute_compression_reward(memory_length, original_length)
        total = self.qa_weight * qa_scores + self.compression_ratio_weight * compression_reward
        return total

    def __call__(
        self,
        predicted_answers: List[str],
        gold_answers: List[str],
        questions: List[str],
        memory_content: str,
        original_content: str,
    ) -> Dict[str, float]:
        """
        Compute all rewards.

        Returns:
            Dict with qa_reward, compression_reward, total_reward
        """
        qa_reward = self.compute_qa_reward(predicted_answers, gold_answers, questions)
        compression_reward = self.compute_compression_reward(
            len(memory_content),
            len(original_content) if original_content else 1
        )
        total_reward = self.compute_total_reward(
            qa_reward,
            len(memory_content),
            len(original_content) if original_content else 1
        )

        return {
            "qa_reward": qa_reward,
            "compression_reward": compression_reward,
            "total_reward": total_reward,
        }
