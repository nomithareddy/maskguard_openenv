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
from rewards import calculate_raw_reward, normalize_reward

DEFAULT_TEXT = "My email is john@gmail.com and my phone number is 9876543210."
MAX_STEPS = 16
DEFAULT_TASK_NAME = "contact_masking"

TASK_LIBRARY = {
    "contact_masking": {
        "difficulty": "easy",
        "text": "My email is [john@gmail.com](mailto:john@gmail.com) and call me at 9876543210.",
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
        task_config = self._resolve_task(
            task_name=task_name,
            text=text,
            policy_mode=policy_mode,
            target_entities=target_entities,
            prefer_task_defaults=True,
        )
        self._default_text = task_config["text"]
        self._default_policy_mode = task_config["policy_mode"]
        self._default_target_entities = task_config["target_entities"]
        self._default_task_name = task_config["task_name"]
        self.reset(
            text=task_config["text"],
            policy_mode=task_config["policy_mode"],
            target_entities=task_config["target_entities"],
            task_name=task_config["task_name"],
        )

    def reset(
        self,
        text: Optional[str] = None,
        policy_mode: Optional[str] = None,
        target_entities: Optional[List[str]] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_config = self._resolve_task(
            task_name=task_name or self._default_task_name,
            text=text if text is not None else self._default_text,
            policy_mode=policy_mode if policy_mode is not None else self._default_policy_mode,
            target_entities=target_entities if target_entities is not None else self._default_target_entities,
            prefer_task_defaults=False,
        )
        self.episode_id = str(uuid4())
        self.task_name = task_config["task_name"]
        self.difficulty = task_config["difficulty"]
        self.original_text = self._normalize_text(task_config["text"])
        self.current_text = self.original_text
        self.policy_mode = task_config["policy_mode"].upper()
        self.policy = get_policy_mode(self.policy_mode)
        self.target_entities = list(task_config["target_entities"])
        self.detected_entities: List[Dict[str, Any]] = []
        self.masked_entities: List[Dict[str, Any]] = []
        self.remaining_entities: List[Dict[str, Any]] = []
        self.skipped_entities: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        self.step_count = 0
        self.total_reward = 0.0
        self.raw_total_reward = 0.0
        self.done = False
        self.submitted = False
        self.invalid_mask_count = 0
        self._reward_bounds = self._compute_reward_bounds()
        self._refresh_entity_views()
        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._build_observation(), 0.0, True, {"status": "already_done"}

        self.step_count += 1
        action_type = action.get("action_type")
        raw_reward = 0.0
        info: Dict[str, Any] = {
            "action_type": action_type,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
        }

        if action_type == MaskGuardActionType.DETECT_ENTITY.value:
            self._refresh_entity_views()
            progress_score = self._progress_score()
            info.update(
                {
                    "status": "entities_detected",
                    "detected_entities": copy.deepcopy(self.detected_entities),
                    "remaining_entities": copy.deepcopy(self.remaining_entities),
                    "grader": self._build_grader_result(progress_score),
                }
            )
        elif action_type == MaskGuardActionType.MASK_ENTITY.value:
            raw_reward, info = self._apply_mask(action)
        elif action_type == MaskGuardActionType.SKIP_ENTITY.value:
            info = self._skip_entity(action)
            raw_reward = calculate_raw_reward(missed_entities=1)
            info["grader"] = self._build_grader_result(self._progress_score())
        elif action_type == MaskGuardActionType.VALIDATE_DOCUMENT.value:
            validation_result = self.validate()
            raw_reward = validation_result["raw_reward"]
            info = {"status": "validated", "validation": validation_result}
        elif action_type == MaskGuardActionType.RECHECK_ENTITIES.value:
            self._refresh_entity_views()
            raw_reward = calculate_raw_reward(missed_entities=len(self.remaining_entities)) if self.remaining_entities else 0.0
            info = {
                "status": "rechecked",
                "remaining_entities": copy.deepcopy(self.remaining_entities),
                "grader": self._build_grader_result(self._progress_score()),
            }
        elif action_type == MaskGuardActionType.SUBMIT_RESULT.value:
            submission_result = self.submit()
            raw_reward = submission_result["raw_reward"]
            info = {
                "status": "submitted" if submission_result["accepted"] else "submission_rejected",
                "submission": submission_result,
            }
        else:
            self.invalid_mask_count += 1
            raw_reward = calculate_raw_reward(invalid_masks=1)
            info = {"status": "invalid_action", "message": f"Unsupported action: {action_type}"}

        if self.step_count >= MAX_STEPS and not self.done:
            self.done = True
            raw_reward += calculate_raw_reward(missed_entities=max(1, len(self.remaining_entities)))
            info["termination_reason"] = "max_steps"

        reward = self._normalize_reward(raw_reward)
        self.raw_total_reward += raw_reward
        self.total_reward = self._normalize_reward(self.raw_total_reward)
        info["raw_reward"] = raw_reward
        info["normalized_reward"] = reward
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
        exploit_penalty = self._exploit_penalty()
        if exploit_penalty:
            metrics["compliance_score"] = max(0.0, metrics["compliance_score"] - exploit_penalty)
            metrics["score"] = metrics["compliance_score"]
        compliant = false_negatives == 0 and metrics["compliance_score"] >= 1.0
        raw_reward = calculate_raw_reward(
            missed_entities=false_negatives,
            overmasks=false_positives,
            compliance_success=compliant,
        )
        if exploit_penalty:
            raw_reward += calculate_raw_reward(overmasks=1)
        reward = self._normalize_reward(raw_reward)
        grader = self._build_grader_result(metrics["score"], remaining_count=false_negatives)
        result = {
            "compliant": compliant,
            "remaining_required_entities": copy.deepcopy(remaining_required_entities),
            "metrics": metrics,
            "grader": grader,
            "raw_reward": raw_reward,
            "reward": reward,
            "score": grader["score"],
            "exploit_penalty": exploit_penalty,
        }
        self.validation_results.append(result)
        return result

    def submit(self) -> Dict[str, Any]:
        validation_result = self.validate()
        accepted = validation_result["compliant"]
        raw_reward = validation_result["raw_reward"]
        if not accepted:
            raw_reward += calculate_raw_reward(invalid_masks=1)
        else:
            self.done = True
            self.submitted = True
        reward = self._normalize_reward(raw_reward)
        return {
            "accepted": accepted,
            "raw_reward": raw_reward,
            "reward": reward,
            "final_text": self.current_text if accepted else None,
            "validation": validation_result,
            "score": validation_result["score"],
        }

    def state(self) -> Dict[str, Any]:
        """Return the current task and episode state."""
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "policy_mode": self.policy_mode,
            "step_count": self.step_count,
            "done": self.done,
            "submitted": self.submitted,
            "score": self._progress_score(),
            "total_reward": self.total_reward,
            "available_tasks": {
                name: {
                    "difficulty": config["difficulty"],
                    "policy_mode": config["policy_mode"],
                }
                for name, config in TASK_LIBRARY.items()
            },
        }

    def _apply_mask(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        target_entity = self._select_entity(action)
        if target_entity is None:
            self.invalid_mask_count += 1
            return calculate_raw_reward(invalid_masks=1), {"status": "invalid_mask", "message": "No matching entity found."}

        if target_entity["value"] not in self.current_text:
            self.invalid_mask_count += 1
            return calculate_raw_reward(invalid_masks=1), {"status": "invalid_mask", "message": "Entity is already masked or missing."}

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
            "grader": self._build_grader_result(self._progress_score()),
        }
        if self.remaining_entities:
            info["next_action"] = MaskGuardActionType.RECHECK_ENTITIES.value
        return calculate_raw_reward(correct_masks=1), info

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
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "score": self._progress_score(),
        }

    def _detect_entities(self, text: str) -> List[Dict[str, Any]]:
        patterns = {
            "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
            "PHONE": r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)",
            "CARD": r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)",
            "ACCOUNT": r"(?i)(?:account|acct)\s*(?:number|no\.?|#)?\s*[:=-]?\s*([A-Z0-9]{6,})",
            "ID": r"(?i)(?:id|mrn|ssn)\s*[:=-]?\s*([A-Z0-9-]{4,})",
            "PERSON": r"(?:(?:my name is|patient is)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
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

    def _compute_reward_bounds(self) -> Tuple[float, float]:
        entity_count = max(1, len(self.target_entities))
        max_reward = (entity_count * 2.0) + 5.0
        min_reward = -((entity_count * 3.0) + 4.0)
        return min_reward, max_reward

    def _normalize_reward(self, raw_reward: float) -> float:
        return normalize_reward(raw_reward, self._reward_bounds[0], self._reward_bounds[1])

    def _progress_score(self) -> float:
        required_total = max(1, len(self._required_entity_types()))
        masked_required = len([entity for entity in self.masked_entities if entity["type"] in self._required_entity_types()])
        return max(0.0, min(1.0, masked_required / required_total))

    def _build_grader_result(self, score: float, remaining_count: Optional[int] = None) -> Dict[str, Any]:
        return MaskGuardEvaluator.grade_task(
            task_name=self.task_name,
            difficulty=self.difficulty,
            metrics={"compliance_score": score},
            remaining_entities=len(self.remaining_entities) if remaining_count is None else remaining_count,
        )

    def _exploit_penalty(self) -> float:
        """
        Penalize exploit-like behavior where repeated placeholder insertion or
        blanket masking patterns could score well without faithful compliance.
        """
        placeholder_count = self.current_text.count("_MASKED")
        expected_mask_count = len(self.masked_entities)
        excessive_placeholders = max(0, placeholder_count - expected_mask_count)
        if self.task_name == "finance_record":
            public_reference_present = "INV-2026-APR" in self.current_text
            if not public_reference_present:
                return 0.20
            if excessive_placeholders > 0:
                return min(0.25, 0.05 * excessive_placeholders)
        return 0.0

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\[(.*?)\]\(mailto:(.*?)\)", r"\1", text)

    def _resolve_task(
        self,
        task_name: str,
        text: str,
        policy_mode: str,
        target_entities: Optional[List[str]],
        prefer_task_defaults: bool,
    ) -> Dict[str, Any]:
        task_key = (task_name or DEFAULT_TASK_NAME).lower()
        if task_key in TASK_LIBRARY:
            task_config = TASK_LIBRARY[task_key]
            if prefer_task_defaults:
                resolved_text = task_config["text"]
                resolved_policy = task_config["policy_mode"]
                resolved_targets = list(task_config["target_entities"])
            else:
                resolved_text = text
                resolved_policy = policy_mode
                resolved_targets = list(target_entities or task_config["target_entities"])
            return {
                "task_name": task_key,
                "difficulty": task_config["difficulty"],
                "text": resolved_text,
                "policy_mode": resolved_policy,
                "target_entities": resolved_targets,
            }
        return {
            "task_name": task_key,
            "difficulty": "custom",
            "text": text,
            "policy_mode": policy_mode,
            "target_entities": list(target_entities or []),
        }
