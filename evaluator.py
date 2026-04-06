"""Evaluation metrics for MaskGuard environment runs."""

from __future__ import annotations

from typing import Dict


class Evaluator:
    @staticmethod
    def precision(true_positives: int, false_positives: int) -> float:
        denom = true_positives + false_positives
        return true_positives / denom if denom else 1.0

    @staticmethod
    def recall(true_positives: int, false_negatives: int) -> float:
        denom = true_positives + false_negatives
        return true_positives / denom if denom else 1.0

    @staticmethod
    def f1_score(precision_value: float, recall_value: float) -> float:
        denom = precision_value + recall_value
        return 2 * precision_value * recall_value / denom if denom else 0.0

    @staticmethod
    def compliance_score(masked_required: int, total_required: int, invalid_actions: int = 0) -> float:
        if total_required == 0:
            return 1.0 if invalid_actions == 0 else max(0.0, 1.0 - 0.1 * invalid_actions)
        raw = masked_required / total_required
        penalty = min(0.5, invalid_actions * 0.1)
        return max(0.0, raw - penalty)

    @classmethod
    def evaluate(cls, *, true_positives: int, false_positives: int, false_negatives: int, masked_required: int, total_required: int, invalid_actions: int = 0) -> Dict[str, float]:
        precision_value = cls.precision(true_positives, false_positives)
        recall_value = cls.recall(true_positives, false_negatives)
        return {
            "precision": precision_value,
            "recall": recall_value,
            "f1_score": cls.f1_score(precision_value, recall_value),
            "compliance_score": cls.compliance_score(masked_required, total_required, invalid_actions),
        }
