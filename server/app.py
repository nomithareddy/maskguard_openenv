"""FastAPI app exposing the MaskGuardEnv endpoints required for OpenEnv use."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from ..env import MaskGuardEnv
except ImportError:
    from env import MaskGuardEnv

app = FastAPI(title="MaskGuardEnv API", version="0.1.0")
ENV = MaskGuardEnv()


class ResetRequest(BaseModel):
    text: Optional[str] = None
    policy_mode: str = Field(default="GDPR")
    expected_entities: list[str] = Field(default_factory=list)


class StepRequest(BaseModel):
    action_type: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity_value: Optional[str] = None


@app.get("/")
def root() -> Dict[str, Any]:
    return {"environment": "MaskGuardEnv", "status": "ready"}


@app.post("/reset")
def reset_environment(request: ResetRequest) -> Dict[str, Any]:
    observation = ENV.reset(
        text=request.text,
        policy_mode=request.policy_mode,
        expected_entities=request.expected_entities,
    )
    return {"observation": observation, "reward": 0.0, "done": False, "info": {"status": "reset"}}


@app.post("/step")
def step_environment(request: StepRequest) -> Dict[str, Any]:
    observation, reward, done, info = ENV.step(request.model_dump(exclude_none=True))
    return {"observation": observation, "reward": reward, "done": done, "info": info}


@app.post("/submit")
def submit_environment() -> Dict[str, Any]:
    submission = ENV.submit()
    return {
        "observation": ENV._observation(),
        "reward": submission["reward"],
        "done": submission["accepted"],
        "info": submission,
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
