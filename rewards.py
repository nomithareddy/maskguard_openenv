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


def calculate_reward(
    *,
    correct_masks: int = 0,
    missed_entities: int = 0,
    overmasks: int = 0,
    invalid_masks: int = 0,
    compliance_success: bool = False,
) -> float:
    """Calculate shaped reward for the current transition."""

    reward = 0.0
    reward += correct_masks * CORRECT_MASK_REWARD
    reward += missed_entities * MISSED_ENTITY_PENALTY
    reward += overmasks * OVERMASK_PENALTY
    reward += invalid_masks * INVALID_MASKING_PENALTY
    if compliance_success:
        reward += COMPLIANCE_SUCCESS_REWARD
    return reward
