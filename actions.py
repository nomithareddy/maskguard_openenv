# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Action definitions for the Maskguard Openenv environment."""

from enum import Enum


class MaskGuardActionType(str, Enum):
    """Supported action types for the MaskGuardEnv environment."""

    DETECT_ENTITY = "detect_entity"
    MASK_ENTITY = "mask_entity"
    SKIP_ENTITY = "skip_entity"
    VALIDATE_DOCUMENT = "validate_document"
    RECHECK_ENTITIES = "recheck_entities"
    SUBMIT_RESULT = "submit_result"


ACTION_SPACE = [action.value for action in MaskGuardActionType]
