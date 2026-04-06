"""Action definitions for the MaskGuard RL environment."""

from enum import Enum


class ActionType(str, Enum):
    DETECT_ENTITY = "detect_entity"
    MASK_ENTITY = "mask_entity"
    SKIP_ENTITY = "skip_entity"
    VALIDATE_DOCUMENT = "validate_document"
    RECHECK_ENTITIES = "recheck_entities"
    SUBMIT_RESULT = "submit_result"


ACTION_SPACE = [action.value for action in ActionType]
