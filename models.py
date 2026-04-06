# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Maskguard Openenv Environment.

The maskguard_openenv environment is a policy-aware RL environment for iterative PII masking.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MaskguardOpenenvAction(Action):
    """Action for the Maskguard Openenv environment."""

    action_type: str = Field(..., description="Action name such as detect_entity or mask_entity")
    entity_id: Optional[str] = Field(default=None, description="Unique entity identifier")
    entity_type: Optional[str] = Field(default=None, description="Entity type to target")
    entity_value: Optional[str] = Field(default=None, description="Literal entity value to target")
    text: Optional[str] = Field(default=None, description="Optional replacement input text")
    policy_mode: Optional[str] = Field(default=None, description="Optional policy mode override")


class MaskguardOpenenvObservation(Observation):
    """Observation from the Maskguard Openenv environment."""

    text: str = Field(default="", description="Current document text")
    detected_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities detected in the source text")
    masked_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities already masked")
    remaining_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities still requiring masking")
    policy_mode: str = Field(default="GDPR", description="Active masking policy mode")
    step_count: int = Field(default=0, description="Current environment step count")
