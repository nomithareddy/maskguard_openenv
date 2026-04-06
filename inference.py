"""
Inference Script Example
===================================
This inference runner uses the OpenAI client to choose actions for MaskGuardEnv
while preserving the required OpenEnv stdout contract.
"""

import json
import os
from typing import List, Optional

from openai import OpenAI

from env import MaskGuardEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "local-dev"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME = os.getenv("MASKGUARD_TASK", "contact_masking")
BENCHMARK = os.getenv("MASKGUARD_BENCHMARK", "maskguard_openenv")
MAX_STEPS = 12


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
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


def build_prompt(observation: dict) -> str:
    return (
        "You are controlling a policy-aware masking environment.\n"
        "Choose exactly one next action as JSON with keys action_type and optional entity_id.\n"
        "Allowed actions: detect_entity, mask_entity, skip_entity, validate_document, recheck_entities, submit_result.\n"
        f"Observation: {json.dumps(observation)}\n"
        "Rules:\n"
        "- If step_count is 0, choose detect_entity.\n"
        "- If remaining_entities is non-empty, choose mask_entity for the first remaining entity id.\n"
        "- If no remaining_entities and masked_entities exists, choose validate_document.\n"
        "- If validated and compliant, choose submit_result.\n"
        "Return JSON only."
    )


def choose_action(client: OpenAI, observation: dict) -> dict:
    prompt = build_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You output valid compact JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        content = (completion.choices[0].message.content or "").strip()
        action = json.loads(content)
        if "action_type" in action:
            return action
    except Exception:
        pass

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
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MaskGuardEnv(task_name=TASK_NAME)
    observation = env.reset(task_name=TASK_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        action = choose_action(client, observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        steps_taken = step
        log_step(
            step=step,
            action=json.dumps(action, separators=(",", ":")),
            reward=reward,
            done=done,
            error=info.get("message"),
        )

        if action["action_type"] == "validate_document" and info["validation"]["compliant"]:
            action = {"action_type": "submit_result"}
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            steps_taken += 1
            log_step(
                step=steps_taken,
                action=json.dumps(action, separators=(",", ":")),
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
