# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Maskguard Openenv Environment.

This module creates an HTTP server that exposes the MaskguardOpenenvEnvironment
with explicit reset, step, and submit endpoints for policy-aware masking.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from ..server.maskguard_openenv_environment import MaskguardOpenenvEnvironment
except ModuleNotFoundError:
    from server.maskguard_openenv_environment import MaskguardOpenenvEnvironment

app = FastAPI(title="Maskguard Openenv Environment", version="0.1.0")
environment = MaskguardOpenenvEnvironment()


class ResetRequest(BaseModel):
    text: Optional[str] = None
    policy_mode: str = Field(default="GDPR")
    target_entities: List[str] = Field(default_factory=list)


class StepRequest(BaseModel):
    action_type: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity_value: Optional[str] = None


@app.post("/reset")
def reset_environment(request: ResetRequest) -> Dict[str, Any]:
    observation = environment._env.reset(
        text=request.text,
        policy_mode=request.policy_mode,
        target_entities=request.target_entities,
    )
    environment._state.episode_id = environment._env.episode_id
    environment._state.step_count = environment._env.step_count
    return {
        "observation": observation,
        "reward": 0.0,
        "done": False,
        "info": {"status": "reset"},
    }


@app.post("/step")
def step_environment(request: StepRequest) -> Dict[str, Any]:
    observation = environment.step(request)
    return {
        "observation": observation.model_dump(exclude={"reward", "done", "metadata"}),
        "reward": observation.reward,
        "done": observation.done,
        "info": observation.metadata,
    }


@app.post("/submit")
def submit_environment() -> Dict[str, Any]:
    submission_result = environment.submit()
    return {
        "observation": environment._env._build_observation(),
        "reward": submission_result["reward"],
        "done": submission_result["accepted"],
        "info": submission_result,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
