"""Dataset runner for MaskGuardEnv."""

from __future__ import annotations

import json
from pathlib import Path

from env import MaskGuardEnv
from evaluator import Evaluator

DATASET_PATH = Path(__file__).parent / "datasets" / "sample_inputs.json"


def infer_policy(expected_entities: list[str]) -> str:
    if any(entity in {"ID", "PERSON"} for entity in expected_entities):
        return "HIPAA"
    if any(entity in {"CARD", "ACCOUNT"} for entity in expected_entities):
        return "FINANCE"
    return "GDPR"


def run_sample(sample: dict) -> tuple[dict, float]:
    policy_mode = infer_policy(sample.get("entities", []))
    env = MaskGuardEnv(text=sample["text"], policy_mode=policy_mode, expected_entities=sample.get("entities", []))
    observation = env.reset(text=sample["text"], policy_mode=policy_mode, expected_entities=sample.get("entities", []))
    total_reward = 0.0

    observation, reward, _, _ = env.step({"action_type": "detect_entity"})
    total_reward += reward

    while observation["remaining_entities"]:
        observation, reward, _, _ = env.step({"action_type": "mask_entity", "entity_id": observation["remaining_entities"][0]["id"]})
        total_reward += reward
        observation, reward, _, _ = env.step({"action_type": "recheck_entities"})
        total_reward += reward

    _, reward, _, _ = env.step({"action_type": "validate_document"})
    total_reward += reward
    _, reward, _, _ = env.step({"action_type": "submit_result"})
    total_reward += reward

    validation = env.validate()
    return validation["metrics"], total_reward


def main() -> None:
    data = json.loads(DATASET_PATH.read_text())
    metrics_list = []
    rewards = []

    for sample in data["samples"]:
        metrics, reward = run_sample(sample)
        metrics_list.append(metrics)
        rewards.append(reward)

    precision = sum(item["precision"] for item in metrics_list) / len(metrics_list)
    recall = sum(item["recall"] for item in metrics_list) / len(metrics_list)
    f1_score = Evaluator.f1_score(precision, recall)
    average_reward = sum(rewards) / len(rewards)

    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1 score: {f1_score:.3f}")
    print(f"average reward: {average_reward:.3f}")


if __name__ == "__main__":
    main()
