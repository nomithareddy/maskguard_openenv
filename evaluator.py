# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation helpers for the Maskguard Openenv environment."""

from typing import Dict


class MaskGuardEvaluator:
    """Compute structured evaluation metrics for masking performance."""

    @staticmethod
    def precision(true_positives: int, false_positives: int) -> float:
        denominator = true_positives + false_positives
        return true_positives / denominator if denominator else 1.0

    @staticmethod
    def recall(true_positives: int, false_negatives: int) -> float:
        denominator = true_positives + false_negatives
        return true_positives / denominator if denominator else 1.0

    @staticmethod
    def f1_score(precision_value: float, recall_value: float) -> float:
        denominator = precision_value + recall_value
        return 2 * precision_value * recall_value / denominator if denominator else 0.0

    @staticmethod
    def compliance_score(masked_required: int, total_required: int, invalid_masks: int = 0) -> float:
        if total_required == 0:
            return max(0.0, 1.0 - (0.1 * invalid_masks))
        raw_score = masked_required / total_required
        penalty = min(0.5, 0.1 * invalid_masks)
        return max(0.0, raw_score - penalty)

    @classmethod
    def evaluate(
        cls,
        *,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        masked_required: int,
        total_required: int,
        invalid_masks: int = 0,
    ) -> Dict[str, float]:
        precision_value = cls.precision(true_positives, false_positives)
        recall_value = cls.recall(true_positives, false_negatives)
        compliance = cls.compliance_score(masked_required, total_required, invalid_masks)
        return {
            "precision": precision_value,
            "recall": recall_value,
            "f1_score": cls.f1_score(precision_value, recall_value),
            "compliance_score": compliance,
            "score": compliance,
        }

    @staticmethod
    def grade_task(*, task_name: str, difficulty: str, metrics: Dict[str, float], remaining_entities: int) -> Dict[str, float | str]:
        """Return a task-specific grader result in the required 0.0-1.0 range."""

        difficulty_bonus = {
            "easy": 0.00,
            "medium": 0.01,
            "hard": 0.00,
        }.get(difficulty, 0.0)
        remaining_penalty = {
            "easy": 0.05,
            "medium": 0.08,
            "hard": 0.12,
        }.get(difficulty, 0.05)
        base_score = max(0.0, metrics["compliance_score"] - (remaining_penalty * remaining_entities))
        grader_score = max(0.0, min(1.0, base_score + difficulty_bonus))
        return {
            "grader_name": f"{task_name}_grader",
            "difficulty": difficulty,
            "score": grader_score,
        }
