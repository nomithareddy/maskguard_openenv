# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core MaskGuardEnv implementation for policy-aware PII masking."""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from actions import MaskGuardActionType
from evaluator import MaskGuardEvaluator
from policy_modes import get_policy_mode
from rewards import calculate_reward

DEFAULT_TEXT = "My email is john@gmail.com and my phone number is 9876543210."
MAX_STEPS = 16


class MaskGuardEnv:
    """Task-specific RL environment for iterative document masking."""

    def __init__(
        self,
        text: str = DEFAULT_TEXT,
        policy_mode: str = "GDPR",
        target_entities: Optional[List[str]] = None,
    ):
        self._default_text = text
        self._default_policy_mode = policy_mode
        self._default_target_entities = target_entities or []
        self.reset(text=text, policy_mode=policy_mode, target_entities=target_entities)

    def reset(
        self,
        text: Optional[str] = None,
        policy_mode: Optional[str] = None,
        target_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.episode_id = str(uuid4())
        self.original_text = self._normalize_text(text or self._default_text)
        self.current_text = self.original_text
        self.policy_mode = (policy_mode or self._default_policy_mode).upper()
        self.policy = get_policy_mode(self.policy_mode)
        self.target_entities = list(target_entities or self._default_target_entities)
        self.detected_entities: List[Dict[str, Any]] = []
        self.masked_entities: List[Dict[str, Any]] = []
        self.remaining_entities: List[Dict[str, Any]] = []
        self.skipped_entities: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.submitted = False
        self.invalid_mask_count = 0
        self._refresh_entity_views()
        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._build_observation(), 0.0, True, {"status": "already_done"}

        self.step_count += 1
        action_type = action.get("action_type")
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action_type}

        if action_type == MaskGuardActionType.DETECT_ENTITY.value:
            self._refresh_entity_views()
            info.update(
                {
                    "status": "entities_detected",
                    "detected_entities": copy.deepcopy(self.detected_entities),
                    "remaining_entities": copy.deepcopy(self.remaining_entities),
                }
            )
        elif action_type == MaskGuardActionType.MASK_ENTITY.value:
            reward, info = self._apply_mask(action)
        elif action_type == MaskGuardActionType.SKIP_ENTITY.value:
            info = self._skip_entity(action)
        elif action_type == MaskGuardActionType.VALIDATE_DOCUMENT.value:
            validation_result = self.validate()
            reward = validation_result["reward"]
            info = {"status": "validated", "validation": validation_result}
        elif action_type == MaskGuardActionType.RECHECK_ENTITIES.value:
            self._refresh_entity_views()
            reward = calculate_reward(missed_entities=len(self.remaining_entities)) if self.remaining_entities else 0.0
            info = {
                "status": "rechecked",
                "remaining_entities": copy.deepcopy(self.remaining_entities),
            }
        elif action_type == MaskGuardActionType.SUBMIT_RESULT.value:
            submission_result = self.submit()
            reward = submission_result["reward"]
            info = {"status": "submitted" if submission_result["accepted"] else "submission_rejected", "submission": submission_result}
        else:
            self.invalid_mask_count += 1
            reward = calculate_reward(invalid_masks=1)
            info = {"status": "invalid_action", "message": f"Unsupported action: {action_type}"}

        if self.step_count >= MAX_STEPS and not self.done:
            self.done = True
            reward += calculate_reward(missed_entities=max(1, len(self.remaining_entities)))
            info["termination_reason"] = "max_steps"

        self.total_reward += reward
        return self._build_observation(), reward, self.done, info

    def validate(self) -> Dict[str, Any]:
        self._refresh_entity_views()
        required_types = self._required_entity_types()
        remaining_required_entities = [
            entity for entity in self.remaining_entities if entity["type"] in required_types
        ]
        true_positives = len([entity for entity in self.masked_entities if entity["type"] in required_types])
        false_positives = len([entity for entity in self.masked_entities if entity["type"] not in required_types])
        false_negatives = len(remaining_required_entities)
        metrics = MaskGuardEvaluator.evaluate(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            masked_required=true_positives,
            total_required=true_positives + false_negatives,
            invalid_masks=self.invalid_mask_count,
        )
        compliant = false_negatives == 0 and metrics["compliance_score"] >= 1.0
        reward = calculate_reward(
            missed_entities=false_negatives,
            overmasks=false_positives,
            compliance_success=compliant,
        )
        result = {
            "compliant": compliant,
            "remaining_required_entities": copy.deepcopy(remaining_required_entities),
            "metrics": metrics,
            "reward": reward,
        }
        self.validation_results.append(result)
        return result

    def submit(self) -> Dict[str, Any]:
        validation_result = self.validate()
        accepted = validation_result["compliant"]
        reward = validation_result["reward"]
        if not accepted:
            reward += calculate_reward(invalid_masks=1)
        else:
            self.done = True
            self.submitted = True
        return {
            "accepted": accepted,
            "reward": reward,
            "final_text": self.current_text if accepted else None,
            "validation": validation_result,
        }

    def _apply_mask(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        target_entity = self._select_entity(action)
        if target_entity is None:
            self.invalid_mask_count += 1
            return calculate_reward(invalid_masks=1), {"status": "invalid_mask", "message": "No matching entity found."}

        if target_entity["value"] not in self.current_text:
            self.invalid_mask_count += 1
            return calculate_reward(invalid_masks=1), {"status": "invalid_mask", "message": "Entity is already masked or missing."}

        masked_value = f"[{target_entity['type']}_MASKED]"
        self.current_text = self.current_text.replace(target_entity["value"], masked_value, 1)
        masked_entity = copy.deepcopy(target_entity)
        masked_entity["masked_value"] = masked_value
        if not any(existing["id"] == masked_entity["id"] for existing in self.masked_entities):
            self.masked_entities.append(masked_entity)
        self._refresh_entity_views()
        info = {
            "status": "entity_masked",
            "masked_entity": masked_entity,
            "remaining_entities": copy.deepcopy(self.remaining_entities),
        }
        if self.remaining_entities:
            info["next_action"] = MaskGuardActionType.RECHECK_ENTITIES.value
        return calculate_reward(correct_masks=1), info

    def _skip_entity(self, action: Dict[str, Any]) -> Dict[str, Any]:
        target_entity = self._select_entity(action)
        if target_entity and not any(existing["id"] == target_entity["id"] for existing in self.skipped_entities):
            self.skipped_entities.append(copy.deepcopy(target_entity))
        return {
            "status": "entity_skipped",
            "skipped_entity": copy.deepcopy(target_entity) if target_entity else None,
            "remaining_entities": copy.deepcopy(self.remaining_entities),
        }

    def _select_entity(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entity_id = action.get("entity_id")
        entity_type = action.get("entity_type")
        entity_value = action.get("entity_value")
        candidate_entities = self.remaining_entities or self.detected_entities

        if entity_id:
            for entity in candidate_entities:
                if entity["id"] == entity_id:
                    return copy.deepcopy(entity)
        if entity_value:
            for entity in candidate_entities:
                if entity["value"] == entity_value:
                    return copy.deepcopy(entity)
        if entity_type:
            for entity in candidate_entities:
                if entity["type"] == entity_type:
                    return copy.deepcopy(entity)
        return copy.deepcopy(candidate_entities[0]) if candidate_entities else None

    def _required_entity_types(self) -> set:
        return set(self.policy.required_entity_types).union(self.target_entities)

    def _refresh_entity_views(self) -> None:
        self.detected_entities = self._detect_entities(self.original_text)
        required_types = self._required_entity_types()
        self.remaining_entities = [
            entity
            for entity in self.detected_entities
            if entity["type"] in required_types and entity["value"] in self.current_text
        ]

    def _build_observation(self) -> Dict[str, Any]:
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
        seen = set()
        for entity_type, pattern in patterns.items():
            for index, match in enumerate(re.finditer(pattern, text)):
                value = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start(0)
                end = match.end(1) if match.lastindex else match.end(0)
                key = (entity_type, value, start)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    {
                        "id": f"{entity_type}-{start}-{index}",
                        "type": entity_type,
                        "value": value,
                        "start": start,
                        "end": end,
                    }
                )
        entities.sort(key=lambda entity: entity["start"])
        return entities

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\[(.*?)\]\(mailto:(.*?)\)", r"\1", text)
