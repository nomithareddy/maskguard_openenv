# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset runner for the Maskguard Openenv environment."""

import json
import os
import sys
from pathlib import Path

# Explicitly add the project root to sys.path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env import MaskGuardEnv, TASK_LIBRARY
from evaluator import MaskGuardEvaluator

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

DATASET_PATH = Path(__file__).parent / "datasets" / "sample_inputs.json"


def infer_policy_mode(entity_types):
    """Infer a policy mode from the target entity types."""
    if any(entity_type in {"ACCOUNT", "CARD"} for entity_type in entity_types):
        return "FINANCE"
    if any(entity_type in {"ID", "PERSON"} for entity_type in entity_types):
        return "HIPAA"
    return "GDPR"


def run_episode(env: MaskGuardEnv, task_name: str):
    """Run one deterministic episode and return validation metrics and reward."""
    log_start(task=task_name, env="maskguard_openenv", model="deterministic_baseline")
    rewards = []
    step_count = 0
    
    # Step 1: Detect
    action = {"action_type": "detect_entity"}
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    step_count += 1
    log_step(step=step_count, action=json.dumps(action, separators=(",", ":")), reward=reward, done=done)

    # Step N: Mask
    while obs["remaining_entities"]:
        action = {"action_type": "mask_entity", "entity_id": obs["remaining_entities"][0]["id"]}
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step_count += 1
        log_step(step=step_count, action=json.dumps(action, separators=(",", ":")), reward=reward, done=done)

    # Step Validation
    action = {"action_type": "validate_document"}
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    step_count += 1
    log_step(step=step_count, action=json.dumps(action, separators=(",", ":")), reward=reward, done=done)
    
    validation_info = info.get("validation", {})
    
    # Step Submit
    action = {"action_type": "submit_result"}
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    step_count += 1
    log_step(step=step_count, action=json.dumps(action, separators=(",", ":")), reward=reward, done=done)

    success = bool(validation_info.get("compliant", False))
    score = validation_info.get("score", 0.0)
    log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return validation_info, sum(rewards)


def run_sample(sample):
    """Run one dataset sample through the masking environment."""
    policy_mode = infer_policy_mode(sample["entities"])
    task_name = "contact_masking" # sample tasks use the base task
    env = MaskGuardEnv(text=sample["text"], policy_mode=policy_mode, target_entities=sample["entities"], task_name=task_name)
    validation_info, total_reward = run_episode(env, task_name)
    return validation_info["metrics"], validation_info["grader"], total_reward


def run_builtin_tasks():
    """Run the easy, medium, and hard built-in tasks."""
    results = []
    for task_name in ["contact_masking", "healthcare_note", "finance_record", "education_record"]:
        env = MaskGuardEnv(task_name=task_name)
        validation_info, total_reward = run_episode(env, task_name)
        results.append(
            {
                "task_name": task_name,
                "difficulty": TASK_LIBRARY[task_name]["difficulty"],
                "score": validation_info["score"],
                "grader": validation_info["grader"],
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
    
    print("\n# Final Summary Statistics", flush=True)
    precision = sum(metric["precision"] for metric in metrics_list) / len(metrics_list) if metrics_list else 1.0
    recall = sum(metric["recall"] for metric in metrics_list) / len(metrics_list) if metrics_list else 1.0
    f1_score = MaskGuardEvaluator.f1_score(precision, recall)
    average_reward = sum(rewards) / len(rewards) if rewards else sum(r["reward"] for r in task_results) / len(task_results)
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
