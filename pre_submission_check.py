# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight pre-submission validator for the Maskguard Openenv environment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from env import MaskGuardEnv, TASK_LIBRARY

ROOT = Path(__file__).parent


def run_command(command: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_tasks() -> None:
    required_tasks = ["contact_masking", "healthcare_note", "finance_record"]
    for task_name in required_tasks:
        env = MaskGuardEnv(task_name=task_name)
        env.reset(task_name=task_name)
        observation, reward, _, _ = env.step({"action_type": "detect_entity"})
        assert 0.0 <= reward <= 1.0, f"detect reward out of range for {task_name}"
        while observation["remaining_entities"]:
            observation, reward, _, _ = env.step(
                {"action_type": "mask_entity", "entity_id": observation["remaining_entities"][0]["id"]}
            )
            assert 0.0 <= reward <= 1.0, f"mask reward out of range for {task_name}"
            observation, reward, _, _ = env.step({"action_type": "recheck_entities"})
            assert 0.0 <= reward <= 1.0, f"recheck reward out of range for {task_name}"
        _, reward, _, info = env.step({"action_type": "validate_document"})
        assert 0.0 <= reward <= 1.0, f"validate reward out of range for {task_name}"
        assert 0.0 <= info["validation"]["score"] <= 1.0, f"score out of range for {task_name}"
        assert info["validation"]["grader"]["grader_name"] == f"{task_name}_grader"
        assert TASK_LIBRARY[task_name]["difficulty"] == info["validation"]["grader"]["difficulty"]


def check_inference() -> None:
    code, stdout, stderr = run_command([sys.executable, "inference.py"])
    assert code == 0, f"inference.py failed: {stderr}"
    lines = [line for line in stdout.strip().splitlines() if line]
    assert lines[0].startswith("[START] "), "missing [START]"
    assert lines[-1].startswith("[END] "), "missing [END]"
    assert any(line.startswith("[STEP] ") for line in lines), "missing [STEP]"


def check_dataset_runner() -> None:
    code, stdout, stderr = run_command([sys.executable, "dataset_runner.py"])
    assert code == 0, f"dataset_runner.py failed: {stderr}"
    assert "average task score:" in stdout, "dataset runner missing task score output"


def check_openenv_yaml() -> None:
    content = (ROOT / "openenv.yaml").read_text()
    assert "name: maskguard_openenv" in content
    assert "tasks:" in content
    assert "reward_function:" in content


def main() -> None:
    check_openenv_yaml()
    check_tasks()
    check_inference()
    check_dataset_runner()
    print(json.dumps({"status": "ok", "checks": ["openenv", "tasks", "inference", "dataset_runner"]}))


if __name__ == "__main__":
    main()
