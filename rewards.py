# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward helpers for the Maskguard Openenv environment."""

CORRECT_MASK_REWARD = 2.0
MISSED_ENTITY_PENALTY = -2.0
OVERMASK_PENALTY = -1.0
COMPLIANCE_SUCCESS_REWARD = 5.0
INVALID_MASKING_PENALTY = -3.0


def calculate_raw_reward(
    *,
    correct_masks: int = 0,
    missed_entities: int = 0,
    overmasks: int = 0,
    invalid_masks: int = 0,
    compliance_success: bool = False,
) -> float:
    """Calculate shaped raw reward for the current transition."""

    reward = 0.0
    reward += correct_masks * CORRECT_MASK_REWARD
    reward += missed_entities * MISSED_ENTITY_PENALTY
    reward += overmasks * OVERMASK_PENALTY
    reward += invalid_masks * INVALID_MASKING_PENALTY
    if compliance_success:
        reward += COMPLIANCE_SUCCESS_REWARD
    return reward


def normalize_reward(raw_reward: float, min_reward: float, max_reward: float) -> float:
    """Normalize a raw reward into the required [0.0, 1.0] range."""

    if max_reward <= min_reward:
        return 0.0
    normalized = (raw_reward - min_reward) / (max_reward - min_reward)
    return max(0.0, min(1.0, normalized))


def calculate_reward(
    *,
    correct_masks: int = 0,
    missed_entities: int = 0,
    overmasks: int = 0,
    invalid_masks: int = 0,
    compliance_success: bool = False,
    min_reward: float = -10.0,
    max_reward: float = 10.0,
) -> float:
    """Calculate normalized reward in the required [0.0, 1.0] range."""

    raw_reward = calculate_raw_reward(
        correct_masks=correct_masks,
        missed_entities=missed_entities,
        overmasks=overmasks,
        invalid_masks=invalid_masks,
        compliance_success=compliance_success,
    )
    return normalize_reward(raw_reward, min_reward=min_reward, max_reward=max_reward)
