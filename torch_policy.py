"""
Optional PyTorch policy for the MaskGuard environment.

This file is intentionally dependency-optional: if `torch` is not installed,
`inference.py` will fall back to the deterministic policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TorchPolicy:
    """
    Tiny policy stub to demonstrate a PyTorch-compatible agent surface.

    The environment state is symbolic/structured; a real RL solution would
    embed observations (text + entity features) and train a policy/value model.
    """

    device: str = "cpu"
    _torch: Any = None

    def __post_init__(self) -> None:
        import torch  # optional dependency

        self._torch = torch

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        torch = self._torch

        # Demonstration-only: use a tensor op so the agent is concretely tied to torch.
        remaining = int(len(observation.get("remaining_entities") or []))
        _ = torch.tensor([remaining], device=self.device).float()

        if observation.get("step_count", 0) == 0:
            return {"action_type": "detect_entity"}
        if remaining > 0:
            first = (observation.get("remaining_entities") or [])[0]
            return {"action_type": "mask_entity", "entity_id": first.get("id")}
        return {"action_type": "validate_document"}


def try_create_torch_policy(device: Optional[str] = None) -> Optional[TorchPolicy]:
    try:
        return TorchPolicy(device=device or "cpu")
    except Exception:
        return None

