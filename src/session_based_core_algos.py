"""
Custom GRPO implementation for Session-Based HMEMS training.

This module provides a selective GRPO implementation that supports:
- is_evidence_session mask: non-evidence sessions get advantage = 0
- unique_uid per sample: each session is its own group

Key difference from standard GRPO:
- Non-evidence sessions have advantage = 0 (no policy gradient)
- This is achieved by setting their reward to 0 AND their advantage to 0
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Mem-alpha" / "verl"))

from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator, get_adv_estimator_fn


def register_session_based_grpo():
    """Register custom session-based GRPO estimators."""
    # session_grpo: GRPO with selective advantage for non-evidence sessions
    register_adv_est("session_grpo")(compute_session_grpo_advantage)
    # session_grpo_passk: Pass@k variant
    register_adv_est("session_grpo_passk")(compute_session_grpo_passk_advantage)
    print("Registered session_grpo and session_grpo_passk advantage estimators")


def compute_session_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    is_evidence_session: np.ndarray = None,
    config=None,
):
    """
    Compute advantage for Session-Based GRPO.

    This is a modified GRPO that sets advantage = 0 for non-evidence sessions.
    The is_evidence_session array is passed from reward_extra_info via verl's compute_advantage.

    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        index: group index per sample (uid)
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std
        is_evidence_session: array of bool, True for evidence sessions (passed from reward manager)
        config: algorithm config

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Determine which samples are evidence sessions
    # is_evidence_session is passed from reward manager via verl's compute_advantage
    if is_evidence_session is not None:
        # Convert to numpy array if needed
        if isinstance(is_evidence_session, list):
            is_evidence_session = np.array(is_evidence_session, dtype=bool)
        elif not isinstance(is_evidence_session, np.ndarray):
            is_evidence_session = np.array(is_evidence_session)
        # Ensure it's the right shape
        if is_evidence_session.ndim > 1:
            is_evidence_session = is_evidence_session.squeeze()
    else:
        # Fallback: assume all are evidence sessions
        is_evidence_session = np.ones(scores.shape[0], dtype=bool)

    # Build group statistics only for evidence sessions
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]

        # Group scores by uid, but only for evidence sessions
        for i in range(bsz):
            if is_evidence_session[i]:  # Only group evidence sessions
                id2score[index[i]].append(scores[i])
                id2indices[index[i]].append(i)

        # Compute mean and std for each evidence group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # Compute advantage for each sample
        advantages = torch.zeros_like(scores)
        returns = torch.zeros_like(scores)

        for i in range(bsz):
            if is_evidence_session[i]:
                # Evidence session: compute normal GRPO advantage
                idx = index[i]
                if norm_adv_by_std_in_grpo:
                    advantage = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
                else:
                    advantage = scores[i] - id2mean[idx]
                advantages[i] = advantage
                returns[i] = advantage
            else:
                # Non-evidence session: advantage = 0 (no policy gradient)
                advantages[i] = torch.tensor(0.0)
                returns[i] = torch.tensor(0.0)

        advantages = advantages.unsqueeze(-1) * response_mask
        returns = returns.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_session_grpo_passk_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    is_evidence_session: np.ndarray = None,
    config=None,
):
    """
    Compute Pass@k advantage for Session-Based GRPO.

    Only the best response per group gets non-zero advantage.
    Non-evidence sessions get advantage = 0.
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Determine which samples are evidence sessions
    if is_evidence_session is not None:
        if isinstance(is_evidence_session, list):
            is_evidence_session = np.array(is_evidence_session, dtype=bool)
        elif not isinstance(is_evidence_session, np.ndarray):
            is_evidence_session = np.array(is_evidence_session)
        if is_evidence_session.ndim > 1:
            is_evidence_session = is_evidence_session.squeeze()
    else:
        is_evidence_session = np.ones(scores.shape[0], dtype=bool)

    advantages = torch.zeros_like(scores)
    returns = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]

        # Group evidence sessions only
        for i in range(bsz):
            if is_evidence_session[i]:
                id2scores[index[i]].append(scores[i])
                id2indices[index[i]].append(i)

        for idx in id2scores:
            rewards = torch.tensor(id2scores[idx])
            indices = id2indices[idx]

            if len(rewards) == 1:
                advantage = torch.tensor(0.0)
            else:
                topk, topk_idx = torch.topk(rewards, 2)
                r_max, r_second_max = topk[0], topk[1]
                i_max = indices[topk_idx[0].item()]
                advantage = r_max - r_second_max
                if norm_adv_by_std_in_grpo:
                    std = torch.std(rewards)
                    advantage = advantage / (std + epsilon)

            # Assign advantage only to the max sample in the group
            for i, global_idx in enumerate(indices):
                if global_idx == indices[topk_idx[0].item()] if len(rewards) > 1 else False:
                    advantages[global_idx] = advantage
                else:
                    advantages[global_idx] = torch.tensor(0.0)
                returns[global_idx] = advantages[global_idx]

        # Non-evidence sessions: advantage = 0
        for i in range(bsz):
            if not is_evidence_session[i]:
                advantages[i] = torch.tensor(0.0)
                returns[i] = torch.tensor(0.0)

    advantages = advantages.unsqueeze(-1) * response_mask
    returns = returns.unsqueeze(-1) * response_mask
    return advantages, returns


def compute_copy_aware_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    customized_grpo_rollout_n: int = 4,
    num_turns: int = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    is_evidence_session: np.ndarray = None,
    config=None,
):
    """
    Compute GRPO advantage considering copy structure.

    This is for per-turn training where:
    - customized_grpo_rollout_n copies of the same session are processed
    - Each copy has its own memory state and produces independent rewards
    - The GRPO advantage is computed across the copies

    Structure: (bs, seq_len) where bs = customized_grpo_rollout_n
    Each copy has num_turns tokens (approximate, since response lengths vary)

    Args:
        token_level_rewards: shape (bs, seq_len)
        response_mask: shape (bs, seq_len)
        index: group index per sample
        customized_grpo_rollout_n: number of session copies (default 4)
        num_turns: number of turns per copy (if None, inferred from shape)
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std
        is_evidence_session: array of bool (not used in this implementation)
        config: algorithm config

    Returns:
        advantages: shape (bs, seq_len)
        returns: shape (bs, seq_len)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    seq_len = token_level_rewards.shape[1]

    # Infer num_turns from data if not provided
    if num_turns is None:
        # Approximate: assume roughly equal token lengths per turn
        # This is an approximation since actual response lengths vary
        num_turns = 1  # fallback

    # Infer turn length per copy (approximate)
    # In practice, each copy has variable tokens, but we approximate per-copy reward
    # by dividing total score by expected turns
    tokens_per_turn_approx = seq_len // max(customized_grpo_rollout_n, 1)

    # Per-copy reward approximation
    # For exact computation, we would need turn boundaries
    # Here we use average reward across the sequence
    per_copy_scores = scores / max(tokens_per_turn_approx, 1)  # (bs,)

    # GRPO across copies: customized_grpo_rollout_n copies form one group
    mean_score = per_copy_scores.mean()
    std_score = per_copy_scores.std() + epsilon

    advantages = torch.zeros_like(scores)

    with torch.no_grad():
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                adv = (per_copy_scores[i] - mean_score) / std_score
            else:
                adv = per_copy_scores[i] - mean_score
            advantages[i] = adv

        # Expand to token level
        advantages = advantages.unsqueeze(-1) * response_mask
        returns = advantages.clone()

    return advantages, returns


# Auto-register when module is imported
register_session_based_grpo()