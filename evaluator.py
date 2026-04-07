from typing import Dict


class MaskGuardEvaluator:
    """Compute structured evaluation metrics for masking performance."""

    SCORE_EPSILON = 0.001

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

        return {
            "precision": precision_value,
            "recall": recall_value,
            "f1_score": cls.f1_score(precision_value, recall_value),
            "compliance_score": compliance,
            "score": compliance,
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
            "score": cls._strict_unit_interval(raw_score),
        }

    @classmethod
    def _strict_unit_interval(cls, value: float) -> float:
        return max(cls.SCORE_EPSILON, min(1.0 - cls.SCORE_EPSILON, value))