"""Reward shaping rules for the MaskGuard RL environment."""

from __future__ import annotations

from typing import Iterable

CORRECT_MASK_REWARD = 2
MISSED_ENTITY_PENALTY = -2
OVERMASK_PENALTY = -1
COMPLIANCE_SUCCESS_REWARD = 5
INVALID_MASKING_PENALTY = -3


def calculate_reward(
    *,
    correct_masks: int = 0,
    missed_entities: int = 0,
    overmasks: int = 0,
    invalid_masks: int = 0,
    compliance_success: bool = False,
    extra_penalties: Iterable[float] | None = None,
) -> float:
    reward = 0.0
    reward += correct_masks * CORRECT_MASK_REWARD
    reward += missed_entities * MISSED_ENTITY_PENALTY
    reward += overmasks * OVERMASK_PENALTY
    reward += invalid_masks * INVALID_MASKING_PENALTY
    if compliance_success:
        reward += COMPLIANCE_SUCCESS_REWARD
    if extra_penalties:
        reward += sum(extra_penalties)
    return reward
