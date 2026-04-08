from __future__ import annotations
from typing import Dict


class MaskGuardEvaluator:
    """Compute structured evaluation metrics for masking performance."""

    # Task grader scores must be strictly in (0, 1). Values near endpoints break when
    # formatters use 2 decimal places (0.001 -> "0.00", 0.999 -> "1.00").
    GRADER_SCORE_MIN = 0.01
    GRADER_SCORE_MAX = 0.99

    @staticmethod
    def precision(true_positives: int, false_positives: int) -> float:
        denom = true_positives + false_positives
        return true_positives / denom if denom else 0.5

    @staticmethod
    def recall(true_positives: int, false_negatives: int) -> float:
        denom = true_positives + false_negatives
        return true_positives / denom if denom else 0.5

    @staticmethod
    def f1_score(p: float, r: float) -> float:
        denom = p + r
        return 2 * p * r / denom if denom else 0.5

    @staticmethod
    def compliance_score(masked_required: int, total_required: int, invalid_masks: int = 0) -> float:
        if total_required == 0:
            return 0.8

        score = masked_required / total_required
        penalty = min(0.4, invalid_masks * 0.1)

        return max(0.0, score - penalty)

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

        compliance = cls.compliance_score(
            masked_required,
            total_required,
            invalid_masks,
        )

        clamped_score = cls.clamp_grader_score(compliance)

        return {
            "precision": precision_value,
            "recall": recall_value,
            "f1_score": cls.f1_score(precision_value, recall_value),
            "compliance_score": clamped_score,
            "score": clamped_score,
        }

    @classmethod
    def grade_task(
        cls,
        *,
        task_name: str,
        difficulty: str,
        metrics: Dict[str, float],
        remaining_entities: int,
    ) -> Dict[str, float | str]:

        difficulty_bonus = {
            "easy": 0.02,
            "medium": 0.03,
            "hard": 0.04,
        }.get(difficulty, 0.02)

        penalty_per_remaining = {
            "easy": 0.05,
            "medium": 0.08,
            "hard": 0.12,
        }.get(difficulty, 0.05)

        base_score = metrics["compliance_score"] - (
            penalty_per_remaining * remaining_entities
        )

        raw_score = max(0.0, min(1.0, base_score + difficulty_bonus))

        return {
            "grader_name": f"{task_name}_grader",
            "difficulty": difficulty,
            "score": cls.clamp_grader_score(raw_score),
        }

    @classmethod
    def clamp_grader_score(cls, value: float) -> float:
        """Map a raw score into (0, 1) with room for two-decimal formatting."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return cls.GRADER_SCORE_MIN
        if v != v:  # NaN
            return cls.GRADER_SCORE_MIN
        return max(cls.GRADER_SCORE_MIN, min(cls.GRADER_SCORE_MAX, v))