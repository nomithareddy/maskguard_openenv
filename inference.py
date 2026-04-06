"""Local inference runner for the MaskGuard RL environment."""

from __future__ import annotations

from env import MaskGuardEnv
from evaluator import Evaluator


def choose_next_action(observation: dict) -> dict:
    remaining = observation["remaining_entities"]
    if observation["step_count"] == 0:
        return {"action_type": "detect_entity"}
    if remaining:
        return {"action_type": "mask_entity", "entity_id": remaining[0]["id"]}
    if observation["masked_entities"]:
        return {"action_type": "validate_document"}
    return {"action_type": "submit_result"}


def run_episode() -> dict:
    text = "My email is john@gmail.com and my phone number is 9876543210."
    env = MaskGuardEnv(text=text, policy_mode="GDPR", expected_entities=["EMAIL", "PHONE"])
    observation = env.reset(text=text, policy_mode="GDPR", expected_entities=["EMAIL", "PHONE"])
    rewards: list[float] = []

    print("[START] env=MaskGuardEnv policy=GDPR")

    done = False
    while not done:
        action = choose_next_action(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        print(f"[STEP] step={observation['step_count']} action={action['action_type']} reward={reward:.2f} done={str(done).lower()} info={info['status']}")
        if action["action_type"] == "validate_document" and info["validation"]["compliant"]:
            observation, reward, done, info = env.step({"action_type": "submit_result"})
            rewards.append(reward)
            print(f"[STEP] step={observation['step_count']} action=submit_result reward={reward:.2f} done={str(done).lower()} info={info['status']}")

    validation = env.validate()
    metrics = validation["metrics"]
    print(
        "[END] success={} total_reward={:.2f} precision={:.2f} recall={:.2f} f1={:.2f} compliance={:.2f}".format(
            str(validation["compliant"]).lower(),
            sum(rewards),
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["compliance_score"],
        )
    )
    return {"rewards": rewards, "metrics": metrics}


if __name__ == "__main__":
    run_episode()
