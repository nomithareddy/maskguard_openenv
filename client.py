# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maskguard Openenv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MaskguardOpenenvAction, MaskguardOpenenvObservation


class MaskguardOpenenvEnv(
    EnvClient[MaskguardOpenenvAction, MaskguardOpenenvObservation, State]
):
    """Client for the Maskguard Openenv Environment."""

    def _step_payload(self, action: MaskguardOpenenvAction) -> Dict:
        """Convert MaskguardOpenenvAction to JSON payload for step message."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[MaskguardOpenenvObservation]:
        """Parse server response into StepResult[MaskguardOpenenvObservation]."""
        obs_data = payload.get("observation", {})
        observation = MaskguardOpenenvObservation(
            text=obs_data.get("text", ""),
            detected_entities=obs_data.get("detected_entities", []),
            masked_entities=obs_data.get("masked_entities", []),
            remaining_entities=obs_data.get("remaining_entities", []),
            policy_mode=obs_data.get("policy_mode", "GDPR"),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("info", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
