"""
Inference Script Example
===================================
This local runner exercises the MaskGuardEnv environment and prints the
required step-by-step reward trace for validation.
"""

from typing import List

from env import MaskGuardEnv

TASK_NAME = "maskguard_masking"
BENCHMARK = "maskguard_openenv"
MODEL_NAME = "rule-based-masker"
MAX_STEPS = 12


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def choose_action(observation: dict) -> dict:
    if observation["step_count"] == 0:
        return {"action_type": "detect_entity"}
    if observation["remaining_entities"]:
        return {
            "action_type": "mask_entity",
            "entity_id": observation["remaining_entities"][0]["id"],
        }
    if observation["masked_entities"]:
        return {"action_type": "validate_document"}
    return {"action_type": "submit_result"}


def main() -> None:
    env = MaskGuardEnv(
        text="My email is [john@gmail.com](mailto:john@gmail.com) and call me at 9876543210.",
        policy_mode="GDPR",
        target_entities=["EMAIL", "PHONE"],
    )

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    observation = env.reset(
        text="My email is [john@gmail.com](mailto:john@gmail.com) and call me at 9876543210.",
        policy_mode="GDPR",
        target_entities=["EMAIL", "PHONE"],
    )

    for step in range(1, MAX_STEPS + 1):
        action = choose_action(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action["action_type"], reward=reward, done=done, error=info.get("message"))

        if action["action_type"] == "validate_document" and info["validation"]["compliant"]:
            observation, reward, done, info = env.step({"action_type": "submit_result"})
            rewards.append(reward)
            steps_taken += 1
            log_step(
                step=steps_taken,
                action="submit_result",
                reward=reward,
                done=done,
                error=info.get("message"),
            )
            break

        if done:
            break

    validation_result = env.validate()
    score = validation_result["metrics"]["compliance_score"]
    success = validation_result["compliant"]
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
