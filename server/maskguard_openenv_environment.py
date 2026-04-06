# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maskguard Openenv Environment Implementation.

A task-specific RL environment for policy-aware PII masking.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..env import MaskGuardEnv
    from ..models import MaskguardOpenenvAction, MaskguardOpenenvObservation
except ImportError:
    from env import MaskGuardEnv
    from models import MaskguardOpenenvAction, MaskguardOpenenvObservation


class MaskguardOpenenvEnvironment(Environment):
    """
    An RL masking environment that iteratively detects, masks, validates, and submits.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the maskguard_openenv environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = MaskGuardEnv()

    def reset(self) -> MaskguardOpenenvObservation:
        """Reset the environment and return the initial observation."""
        observation = self._env.reset()
        self._state = State(episode_id=self._env.episode_id, step_count=0)
        return MaskguardOpenenvObservation(
            text=observation["text"],
            detected_entities=observation["detected_entities"],
            masked_entities=observation["masked_entities"],
            remaining_entities=observation["remaining_entities"],
            policy_mode=observation["policy_mode"],
            step_count=observation["step_count"],
            done=False,
            reward=0.0,
            metadata={"status": "reset"},
        )

    def step(self, action: MaskguardOpenenvAction) -> MaskguardOpenenvObservation:  # type: ignore[override]
        """Execute one masking action in the environment."""
        payload = action.model_dump(exclude_none=True)
        if payload.get("text") or payload.get("policy_mode"):
            observation = self._env.reset(
                text=payload.get("text"),
                policy_mode=payload.get("policy_mode"),
                target_entities=[],
            )
            return MaskguardOpenenvObservation(
                text=observation["text"],
                detected_entities=observation["detected_entities"],
                masked_entities=observation["masked_entities"],
                remaining_entities=observation["remaining_entities"],
                policy_mode=observation["policy_mode"],
                step_count=observation["step_count"],
                done=False,
                reward=0.0,
                metadata={"status": "reset_via_step"},
            )

        observation, reward, done, info = self._env.step(payload)
        self._state.step_count = self._env.step_count
        return MaskguardOpenenvObservation(
            text=observation["text"],
            detected_entities=observation["detected_entities"],
            masked_entities=observation["masked_entities"],
            remaining_entities=observation["remaining_entities"],
            policy_mode=observation["policy_mode"],
            step_count=observation["step_count"],
            done=done,
            reward=reward,
            metadata=info,
        )

    def validate(self):
        """Validate the current masking state."""
        return self._env.validate()

    def submit(self):
        """Submit the current masking result."""
        return self._env.submit()

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
