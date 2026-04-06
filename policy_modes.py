"""Policy modes and validation requirements for MaskGuard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class PolicyMode:
    name: str
    description: str
    required_entities: Set[str]
    min_compliance_score: float = 1.0


POLICY_MODES: Dict[str, PolicyMode] = {
    "GDPR": PolicyMode(
        name="GDPR",
        description="Mask personal contact and identity information.",
        required_entities={"EMAIL", "PHONE", "PERSON"},
    ),
    "HIPAA": PolicyMode(
        name="HIPAA",
        description="Mask protected health identifiers including names, contact details, and IDs.",
        required_entities={"EMAIL", "PHONE", "PERSON", "ID"},
    ),
    "FINANCE": PolicyMode(
        name="FINANCE",
        description="Mask financial identifiers and contact channels.",
        required_entities={"EMAIL", "PHONE", "CARD", "ACCOUNT"},
    ),
}


def get_policy_mode(name: str) -> PolicyMode:
    return POLICY_MODES.get(name.upper(), POLICY_MODES["GDPR"])


def list_policy_modes() -> List[str]:
    return list(POLICY_MODES)
