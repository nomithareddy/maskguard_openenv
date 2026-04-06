# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Policy mode definitions for the Maskguard Openenv environment."""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class PolicyMode:
    """Policy mode configuration for required masking behavior."""

    name: str
    description: str
    required_entity_types: Set[str]


POLICY_MODES: Dict[str, PolicyMode] = {
    "GDPR": PolicyMode(
        name="GDPR",
        description="Mask personal contact and identity information.",
        required_entity_types={"EMAIL", "PHONE", "PERSON"},
    ),
    "HIPAA": PolicyMode(
        name="HIPAA",
        description="Mask protected health information and direct identifiers.",
        required_entity_types={"EMAIL", "PHONE", "PERSON", "ID"},
    ),
    "FINANCE": PolicyMode(
        name="FINANCE",
        description="Mask financial account, card, and contact identifiers.",
        required_entity_types={"EMAIL", "PHONE", "ACCOUNT", "CARD"},
    ),
}


def get_policy_mode(name: str) -> PolicyMode:
    """Return the configured policy mode, defaulting to GDPR."""

    return POLICY_MODES.get(name.upper(), POLICY_MODES["GDPR"])


def list_policy_modes() -> List[str]:
    """Return all supported policy mode names."""

    return list(POLICY_MODES.keys())
