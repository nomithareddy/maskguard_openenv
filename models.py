"""OpenEnv action and observation models for MaskGuardEnv."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MaskguardOpenenvAction(Action):
    action_type: str = Field(..., description="Action name such as detect_entity or mask_entity")
    entity_id: Optional[str] = Field(default=None, description="Optional entity identifier to act on")
    entity_type: Optional[str] = Field(default=None, description="Optional entity type filter")
    entity_value: Optional[str] = Field(default=None, description="Optional raw entity value to act on")
    text: Optional[str] = Field(default=None, description="Optional new episode text for reset-like flows")
    policy_mode: Optional[str] = Field(default=None, description="Optional policy mode override")


class MaskguardOpenenvObservation(Observation):
    text: str = Field(default="", description="Current masked or unmasked working text")
    detected_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities detected from the source text")
    masked_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities already masked")
    remaining_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entities still requiring masking")
    policy_mode: str = Field(default="GDPR", description="Active policy mode")
    step_count: int = Field(default=0, description="Current environment step count")
