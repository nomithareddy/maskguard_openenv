# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset runner for the Maskguard Openenv environment."""

import json
from pathlib import Path

from env import MaskGuardEnv
from evaluator import MaskGuardEvaluator

DATASET_PATH = Path(__file__).parent / "datasets" / "sample_inputs.json"


def infer_policy_mode(entity_types):
    """Infer a policy mode from the target entity types."""
    if any(entity_type in {"ACCOUNT", "CARD"} for entity_type in entity_types):
        return "FINANCE"
    if any(entity_type in {"ID", "PERSON"} for entity_type in entity_types):
        return "HIPAA"
    return "GDPR"


def run_sample(sample):
    """Run one sample through the masking environment."""
    policy_mode = infer_policy_mode(sample["entities"])
    env = MaskGuardEnv(text=sample["text"], policy_mode=policy_mode, target_entities=sample["entities"])
    observation = env.reset(text=sample["text"], policy_mode=policy_mode, target_entities=sample["entities"])
    total_reward = 0.0

    observation, reward, _, _ = env.step({"action_type": "detect_entity"})
    total_reward += reward

    while observation["remaining_entities"]:
        observation, reward, _, _ = env.step(
            {"action_type": "mask_entity", "entity_id": observation["remaining_entities"][0]["id"]}
        )
        total_reward += reward
        observation, reward, _, _ = env.step({"action_type": "recheck_entities"})
        total_reward += reward

    _, reward, _, _ = env.step({"action_type": "validate_document"})
    total_reward += reward
    _, reward, _, _ = env.step({"action_type": "submit_result"})
    total_reward += reward

    validation_result = env.validate()
    return validation_result["metrics"], total_reward


def main() -> None:
    """Run the environment across the sample dataset and print aggregate metrics."""
    samples = json.loads(DATASET_PATH.read_text())["samples"]
    metrics_list = []
    rewards = []

    for sample in samples:
        metrics, reward = run_sample(sample)
        metrics_list.append(metrics)
        rewards.append(reward)

    precision = sum(metric["precision"] for metric in metrics_list) / len(metrics_list)
    recall = sum(metric["recall"] for metric in metrics_list) / len(metrics_list)
    f1_score = MaskGuardEvaluator.f1_score(precision, recall)
    average_reward = sum(rewards) / len(rewards)

    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1 score: {f1_score:.3f}")
    print(f"average reward: {average_reward:.3f}")


if __name__ == "__main__":
    main()
