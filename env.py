"""MaskGuardEnv core RL environment."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from actions import ActionType
from evaluator import Evaluator
from policy_modes import get_policy_mode
from rewards import calculate_reward

DEFAULT_TEXT = "Contact John Doe at john@gmail.com or 9876543210 for account 1234567890."
MAX_STEPS = 16


class MaskGuardEnv:
    """Policy-aware RL environment for iterative PII masking."""

    def __init__(self, text: str | None = None, policy_mode: str = "GDPR", expected_entities: Optional[List[str]] = None):
        self.default_text = text or DEFAULT_TEXT
        self.default_policy_mode = policy_mode
        self.default_expected_entities = expected_entities or []
        self.reset(text=text, policy_mode=policy_mode, expected_entities=expected_entities)

    def reset(
        self,
        text: str | None = None,
        policy_mode: str | None = None,
        expected_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.episode_id = str(uuid4())
        self.original_text = self._normalize_text(text or self.default_text)
        self.current_text = self.original_text
        self.policy_mode = (policy_mode or self.default_policy_mode).upper()
        self.policy = get_policy_mode(self.policy_mode)
        self.expected_entities = list(expected_entities or self.default_expected_entities)
        self.detected_entities: List[Dict[str, Any]] = []
        self.masked_entities: List[Dict[str, Any]] = []
        self.remaining_entities: List[Dict[str, Any]] = []
        self.skipped_entities: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.invalid_actions = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        self.submitted = False
        self.last_info: Dict[str, Any] = {"status": "reset"}
        self._refresh_entities(use_current_text=False)
        return self._observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            info = {"status": "already_done", "message": "Episode has already been submitted."}
            return self._observation(), 0.0, True, info

        action_type = action.get("action_type")
        info: Dict[str, Any] = {"action_type": action_type}
        reward = 0.0
        self.step_count += 1

        if action_type == ActionType.DETECT_ENTITY.value:
            self._refresh_entities(use_current_text=False)
            info.update(
                {
                    "status": "entities_detected",
                    "detected_count": len(self.detected_entities),
                    "remaining_count": len(self.remaining_entities),
                }
            )
        elif action_type == ActionType.MASK_ENTITY.value:
            reward, info = self._mask_entity(action)
        elif action_type == ActionType.SKIP_ENTITY.value:
            info = self._skip_entity(action)
        elif action_type == ActionType.VALIDATE_DOCUMENT.value:
            validation = self.validate()
            info = {"status": "validated", "validation": validation}
            reward = validation["reward"]
        elif action_type == ActionType.RECHECK_ENTITIES.value:
            self._refresh_entities(use_current_text=True)
            info.update(
                {
                    "status": "rechecked",
                    "remaining_count": len(self.remaining_entities),
                    "remaining_entities": copy.deepcopy(self.remaining_entities),
                }
            )
            if self.remaining_entities:
                reward = calculate_reward(missed_entities=len(self.remaining_entities))
        elif action_type == ActionType.SUBMIT_RESULT.value:
            submission = self.submit()
            info = {"status": "submitted" if submission["accepted"] else "submission_rejected", "submission": submission}
            reward = submission["reward"]
        else:
            self.invalid_actions += 1
            reward = calculate_reward(invalid_masks=1)
            info = {"status": "invalid_action", "message": f"Unsupported action: {action_type}"}

        if self.step_count >= MAX_STEPS and not self.done:
            timeout_penalty = calculate_reward(missed_entities=max(1, len(self.remaining_entities)))
            reward += timeout_penalty
            self.done = True
            info["terminated_reason"] = "max_steps"

        self.total_reward += reward
        self.last_info = info
        return self._observation(), reward, self.done, info

    def validate(self) -> Dict[str, Any]:
        self._refresh_entities(use_current_text=True)
        required_types = self._required_entity_types()
        remaining_required = [entity for entity in self.remaining_entities if entity["type"] in required_types]
        false_positives = self._count_overmasks()
        true_positives = len([entity for entity in self.masked_entities if entity["type"] in required_types])
        false_negatives = len(remaining_required)
        metrics = Evaluator.evaluate(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            masked_required=true_positives,
            total_required=true_positives + false_negatives,
            invalid_actions=self.invalid_actions,
        )
        compliant = metrics["compliance_score"] >= self.policy.min_compliance_score and false_negatives == 0
        reward = calculate_reward(
            missed_entities=false_negatives,
            overmasks=false_positives,
            compliance_success=compliant,
        )
        validation = {
            "compliant": compliant,
            "remaining_required_entities": copy.deepcopy(remaining_required),
            "masked_entities": copy.deepcopy(self.masked_entities),
            "metrics": metrics,
            "reward": reward,
        }
        self.validation_history.append(validation)
        return validation

    def submit(self) -> Dict[str, Any]:
        validation = self.validate()
        accepted = validation["compliant"]
        reward = validation["reward"] if accepted else validation["reward"] + calculate_reward(invalid_masks=1)
        if accepted:
            self.done = True
            self.submitted = True
        return {
            "accepted": accepted,
            "reward": reward,
            "final_text": self.current_text if accepted else None,
            "validation": validation,
        }

    def _mask_entity(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        self._refresh_entities(use_current_text=False)
        entity = self._select_entity(action)
        if entity is None:
            self.invalid_actions += 1
            return calculate_reward(invalid_masks=1), {"status": "invalid_mask", "message": "Entity not found for masking."}

        placeholder = f"[{entity['type']}_MASKED]"
        if placeholder in self.current_text and entity["value"] not in self.current_text:
            self.invalid_actions += 1
            return calculate_reward(invalid_masks=1), {"status": "invalid_mask", "message": "Entity appears to be already masked."}

        if entity["value"] not in self.current_text:
            self.invalid_actions += 1
            return calculate_reward(invalid_masks=1), {"status": "invalid_mask", "message": "Entity value not present in current text."}

        self.current_text = self.current_text.replace(entity["value"], placeholder, 1)
        masked_record = copy.deepcopy(entity)
        masked_record["masked_value"] = placeholder
        if not any(existing["id"] == masked_record["id"] for existing in self.masked_entities):
            self.masked_entities.append(masked_record)
        self._refresh_entities(use_current_text=True)
        reward = calculate_reward(correct_masks=1)
        info = {
            "status": "entity_masked",
            "masked_entity": masked_record,
            "remaining_entities": copy.deepcopy(self.remaining_entities),
        }
        if self.remaining_entities:
            info["next_recommended_action"] = ActionType.RECHECK_ENTITIES.value
        return reward, info

    def _skip_entity(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._refresh_entities(use_current_text=False)
        entity = self._select_entity(action)
        if entity is not None and not any(existing["id"] == entity["id"] for existing in self.skipped_entities):
            self.skipped_entities.append(copy.deepcopy(entity))
        return {
            "status": "entity_skipped",
            "skipped_entity": copy.deepcopy(entity) if entity else None,
            "remaining_entities": copy.deepcopy(self.remaining_entities),
        }

    def _select_entity(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entity_id = action.get("entity_id")
        entity_type = action.get("entity_type")
        entity_value = action.get("entity_value")
        candidates = copy.deepcopy(self.remaining_entities or self.detected_entities)
        if entity_id:
            for entity in candidates:
                if entity["id"] == entity_id:
                    return entity
        if entity_value:
            for entity in candidates:
                if entity["value"] == entity_value:
                    return entity
        if entity_type:
            for entity in candidates:
                if entity["type"] == entity_type:
                    return entity
        return candidates[0] if candidates else None

    def _refresh_entities(self, *, use_current_text: bool) -> None:
        detection_source = self.original_text
        current_entities = self._detect_entities(detection_source)
        self.detected_entities = current_entities
        self.remaining_entities = [
            entity
            for entity in current_entities
            if entity["value"] in self.current_text and entity["type"] in self._required_entity_types()
        ]

    def _required_entity_types(self) -> set[str]:
        return set(self.policy.required_entities).union(self.expected_entities)

    def _count_overmasks(self) -> int:
        required_types = self._required_entity_types()
        return len([entity for entity in self.masked_entities if entity["type"] not in required_types])

    def _observation(self) -> Dict[str, Any]:
        return {
            "text": self.current_text,
            "detected_entities": copy.deepcopy(self.detected_entities),
            "masked_entities": copy.deepcopy(self.masked_entities),
            "remaining_entities": copy.deepcopy(self.remaining_entities),
            "policy_mode": self.policy_mode,
            "step_count": self.step_count,
        }

    def _detect_entities(self, text: str) -> List[Dict[str, Any]]:
        patterns = {
            "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
            "PHONE": r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)",
            "CARD": r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)",
            "ACCOUNT": r"(?i)(?:account|acct)\s*(?:number|no\.?|#)?\s*[:=-]?\s*([A-Z0-9]{6,})",
            "ID": r"(?i)(?:id|mrn|ssn)\s*[:=-]?\s*([A-Z0-9-]{4,})",
            "PERSON": r"(?:(?:my name is|patient is|contact)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        }
        entities: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for entity_type, pattern in patterns.items():
            for idx, match in enumerate(re.finditer(pattern, text)):
                value = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)
                key = (entity_type, value, start)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    {
                        "id": f"{entity_type}-{start}-{idx}",
                        "type": entity_type,
                        "value": value,
                        "start": start,
                        "end": end,
                    }
                )
        entities.sort(key=lambda item: item["start"])
        return entities

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\[(.*?)\]\(mailto:(.*?)\)", r"\1", text)
