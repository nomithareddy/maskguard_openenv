# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset runner for the Maskguard Openenv environment."""

import json
from pathlib import Path

from env import MaskGuardEnv, TASK_LIBRARY
from evaluator import MaskGuardEvaluator

DATASET_PATH = Path(__file__).parent / "datasets" / "sample_inputs.json"


def infer_policy_mode(entity_types):
    """Infer a policy mode from the target entity types."""
    if any(entity_type in {"ACCOUNT", "CARD"} for entity_type in entity_types):
        return "FINANCE"
    if any(entity_type in {"ID", "PERSON"} for entity_type in entity_types):
        return "HIPAA"
    return "GDPR"


def run_episode(env: MaskGuardEnv):
    """Run one deterministic episode and return validation metrics and reward."""
    observation, reward, _, _ = env.step({"action_type": "detect_entity"})
    total_reward = reward

    while observation["remaining_entities"]:
        observation, reward, _, _ = env.step(
            {"action_type": "mask_entity", "entity_id": observation["remaining_entities"][0]["id"]}
        )
        total_reward += reward
        observation, reward, _, _ = env.step({"action_type": "recheck_entities"})
        total_reward += reward

    _, reward, _, validation_info = env.step({"action_type": "validate_document"})
    total_reward += reward
    _, reward, _, _ = env.step({"action_type": "submit_result"})
    total_reward += reward

    return validation_info["validation"], total_reward


def run_sample(sample):
    """Run one dataset sample through the masking environment."""
    policy_mode = infer_policy_mode(sample["entities"])
    env = MaskGuardEnv(text=sample["text"], policy_mode=policy_mode, target_entities=sample["entities"], task_name="contact_masking")
    env.reset(text=sample["text"], policy_mode=policy_mode, target_entities=sample["entities"], task_name="contact_masking")
    validation_result, total_reward = run_episode(env)
    return validation_result["metrics"], validation_result["grader"], total_reward


def run_builtin_tasks():
    """Run the easy, medium, and hard built-in tasks."""
    results = []
    for task_name in ["contact_masking", "healthcare_note", "finance_record"]:
        env = MaskGuardEnv(task_name=task_name)
        env.reset(task_name=task_name)
        validation_result, total_reward = run_episode(env)
        results.append(
            {
                "task_name": task_name,
                "difficulty": TASK_LIBRARY[task_name]["difficulty"],
                "score": validation_result["score"],
                "grader": validation_result["grader"],
                "reward": total_reward,
            }
        )
    return results


def main() -> None:
    """Run the environment across the sample dataset and built-in tasks."""
    samples = json.loads(DATASET_PATH.read_text())["samples"]
    metrics_list = []
    rewards = []

    for sample in samples:
        metrics, _, reward = run_sample(sample)
        metrics_list.append(metrics)
        rewards.append(reward)

    task_results = run_builtin_tasks()
    precision = sum(metric["precision"] for metric in metrics_list) / len(metrics_list)
    recall = sum(metric["recall"] for metric in metrics_list) / len(metrics_list)
    f1_score = MaskGuardEvaluator.f1_score(precision, recall)
    average_reward = sum(rewards) / len(rewards)
    average_task_score = sum(result["score"] for result in task_results) / len(task_results)

    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"F1 score: {f1_score:.3f}")
    print(f"average reward: {average_reward:.3f}")
    print(f"average task score: {average_task_score:.3f}")
    for result in task_results:
        print(
            f"task={result['task_name']} difficulty={result['difficulty']} score={result['score']:.3f} reward={result['reward']:.3f} grader={result['grader']['grader_name']}"
        )


if __name__ == "__main__":
    main()
