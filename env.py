"""
Core MaskGuardEnv implementation for policy-aware PII masking.
Validator-safe version for OpenEnv Phase-2 compliance.
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from actions import MaskGuardActionType
from evaluator import MaskGuardEvaluator
from policy_modes import get_policy_mode
from rewards import calculate_raw_reward, normalize_reward


DEFAULT_TEXT = "My email is john@gmail.com and my phone number is 9876543210."
MAX_STEPS = 16
DEFAULT_TASK_NAME = "contact_masking"


TASK_LIBRARY = {
    "contact_masking": {
        "difficulty": "easy",
        "text": "My email is john@gmail.com and call me at 9876543210.",
        "policy_mode": "GDPR",
        "target_entities": ["EMAIL", "PHONE"],
    },
    "healthcare_note": {
        "difficulty": "medium",
        "text": "Patient is John Doe with ID MRN-7788, phone 9876543210, and email john.doe@hospital.org.",
        "policy_mode": "HIPAA",
        "target_entities": ["PERSON", "ID", "PHONE", "EMAIL"],
    },
    "finance_record": {
        "difficulty": "hard",
        "text": (
            "Finance escalation: account number 1234567890 belongs to john@gmail.com, "
            "backup contact is fin.ops@corp.com, primary card 4111 1111 1111 1111, "
            "secondary card 5555 4444 3333 2222, and callback line is 9876543210. "
            "Reference ticket INV-2026-APR is public and should remain visible."
        ),
        "policy_mode": "FINANCE",
        "target_entities": ["ACCOUNT", "EMAIL", "CARD", "PHONE"],
    },
}


class MaskGuardEnv:
    """Task-specific RL environment for iterative document masking."""

    def __init__(
        self,
        text: str = DEFAULT_TEXT,
        policy_mode: str = "GDPR",
        target_entities: Optional[List[str]] = None,
        task_name: str = DEFAULT_TASK_NAME,
    ):
        self.reset(text, policy_mode, target_entities, task_name)

    # ======================
    # RESET
    # ======================

    def reset(
        self,
        text: Optional[str] = None,
        policy_mode: Optional[str] = None,
        target_entities: Optional[List[str]] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        task_name = task_name or DEFAULT_TASK_NAME
        task_config = TASK_LIBRARY.get(task_name, TASK_LIBRARY[DEFAULT_TASK_NAME])

        self.episode_id = str(uuid4())
        self.task_name = task_name
        self.difficulty = task_config["difficulty"]

        self.original_text = text or task_config["text"]
        self.current_text = self.original_text

        self.policy_mode = (policy_mode or task_config["policy_mode"]).upper()
        self.policy = get_policy_mode(self.policy_mode)

        self.target_entities = target_entities or task_config["target_entities"]

        self.detected_entities = []
        self.masked_entities = []
        self.remaining_entities = []

        self.step_count = 0
        self.invalid_mask_count = 0
        self.done = False
        self.submitted = False

        self._refresh_entity_views()

        return self._build_observation()

    # ======================
    # STEP
    # ======================

    def step(self, action: Dict[str, Any]):

        if self.done:
            return self._build_observation(), 0.0, True, {"status": "already_done"}

        self.step_count += 1
        action_type = action.get("action_type")

        raw_reward = 0.0
        info = {"task_name": self.task_name, "difficulty": self.difficulty}

        if action_type == MaskGuardActionType.DETECT_ENTITY.value:

            self._refresh_entity_views()

            info["status"] = "entities_detected"

        elif action_type == MaskGuardActionType.MASK_ENTITY.value:

            raw_reward, info = self._apply_mask(action)

        elif action_type == MaskGuardActionType.SKIP_ENTITY.value:

            raw_reward = calculate_raw_reward(missed_entities=1)
            info["status"] = "entity_skipped"

        elif action_type == MaskGuardActionType.VALIDATE_DOCUMENT.value:

            validation = self.validate()
            raw_reward = validation["raw_reward"]
            info["validation"] = validation

        elif action_type == MaskGuardActionType.RECHECK_ENTITIES.value:

            self._refresh_entity_views()

        elif action_type == MaskGuardActionType.SUBMIT_RESULT.value:

            submission = self.submit()
            raw_reward = submission["raw_reward"]
            info["submission"] = submission

        else:

            self.invalid_mask_count += 1
            raw_reward = calculate_raw_reward(invalid_masks=1)
            info["status"] = "invalid_action"

        if self.step_count >= MAX_STEPS:
            self.done = True

        reward = normalize_reward(raw_reward, -10, 10)

        info["grader"] = self._build_grader_result(self._progress_score())

        return self._build_observation(), reward, self.done, info

    # ======================
    # VALIDATE
    # ======================

    def validate(self):

        self._refresh_entity_views()

        remaining = len(self.remaining_entities)
        masked = len(self.masked_entities)

        metrics = MaskGuardEvaluator.evaluate(
            true_positives=masked,
            false_positives=0,
            false_negatives=remaining,
            masked_required=masked,
            total_required=masked + remaining,
            invalid_masks=self.invalid_mask_count,
        )

        compliant = remaining == 0 and metrics["compliance_score"] >= 0.95

        raw_reward = calculate_raw_reward(
            missed_entities=remaining,
            compliance_success=compliant,
        )

        grader = self._build_grader_result(metrics["score"], remaining)

        return {
            "metrics": metrics,
            "compliant": compliant,
            "raw_reward": raw_reward,
            "grader": grader,
            "score": grader["score"],
        }

    # ======================
    # SUBMIT
    # ======================

    def submit(self):

        validation = self.validate()

        accepted = validation["compliant"]
        raw_reward = validation["raw_reward"]

        if accepted:
            self.done = True
            self.submitted = True
        else:
            raw_reward += calculate_raw_reward(invalid_masks=1)

        reward = normalize_reward(raw_reward, -10, 10)

        return {
            "accepted": accepted,
            "raw_reward": raw_reward,
            "reward": reward,
            "final_text": self.current_text if accepted else None,
            "validation": validation,
            "score": validation["score"],
            "grader": validation["grader"],  # REQUIRED FOR VALIDATOR
        }

    # ======================
    # ENTITY DETECTION
    # ======================

    def _refresh_entity_views(self):

        self.detected_entities = self._detect_entities(self.original_text)

        required_types = set(self.target_entities)

        self.remaining_entities = [
            e for e in self.detected_entities
            if e["type"] in required_types and e["value"] in self.current_text
        ]

    def _detect_entities(self, text):

        patterns = {
            "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
            "PHONE": r"\d{10}",
            "CARD": r"(?:\d[ -]*?){13,16}",
            "ACCOUNT": r"\b\d{8,12}\b",
            "ID": r"[A-Z]{2,}-\d+",
            "PERSON": r"(?:Patient is|my name is)\s+([A-Z][a-z]+)",
        }

        entities = []

        for entity_type, pattern in patterns.items():
            for i, match in enumerate(re.finditer(pattern, text)):
                entities.append(
                    {
                        "id": f"{entity_type}-{i}",
                        "type": entity_type,
                        "value": match.group(0),
                    }
                )

        return entities

    # ======================
    # MASK APPLY
    # ======================

    def _apply_mask(self, action):

        entity = self.remaining_entities[0] if self.remaining_entities else None

        if not entity:
            return calculate_raw_reward(invalid_masks=1), {"status": "invalid_mask"}

        masked_value = f"[{entity['type']}_MASKED]"

        self.current_text = self.current_text.replace(entity["value"], masked_value, 1)

        self.masked_entities.append(entity)

        self._refresh_entity_views()

        return calculate_raw_reward(correct_masks=1), {"status": "entity_masked"}

    # ======================
    # OBSERVATION
    # ======================

    def _build_observation(self):

        score = self._progress_score()

        return {
            "text": self.current_text,
            "detected_entities": copy.deepcopy(self.detected_entities),
            "masked_entities": copy.deepcopy(self.masked_entities),
            "remaining_entities": copy.deepcopy(self.remaining_entities),
            "policy_mode": self.policy_mode,
            "step_count": self.step_count,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "score": score,
            "grader": self._build_grader_result(score),  # REQUIRED
        }

    # ======================
    # PROGRESS SCORE
    # ======================

    def _progress_score(self):

        total = max(1, len(self.target_entities))

        masked = len([
            e for e in self.masked_entities
            if e["type"] in self.target_entities
        ])

        return masked / total

    # ======================
    # GRADER
    # ======================

    def _build_grader_result(self, score, remaining_count=None):

        return MaskGuardEvaluator.grade_task(
            task_name=self.task_name,
            difficulty=self.difficulty,
            metrics={"compliance_score": score},
            remaining_entities=len(self.remaining_entities)
            if remaining_count is None else remaining_count,
        )