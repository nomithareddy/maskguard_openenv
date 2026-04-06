"""OpenEnv-compatible wrapper around MaskGuardEnv."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..env import MaskGuardEnv
    from ..models import MaskguardOpenenvAction, MaskguardOpenenvObservation
except ImportError:
    from env import MaskGuardEnv
    from models import MaskguardOpenenvAction, MaskguardOpenenvObservation


class MaskguardOpenenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._env = MaskGuardEnv()

    def reset(self) -> MaskguardOpenenvObservation:
        observation = self._env.reset()
        self._state = State(episode_id=self._env.episode_id, step_count=0)
        return MaskguardOpenenvObservation(**observation, reward=0.0, done=False, metadata={"status": "reset"})

    def step(self, action: MaskguardOpenenvAction) -> MaskguardOpenenvObservation:  # type: ignore[override]
        payload = action.model_dump(exclude_none=True)
        observation, reward, done, info = self._env.step(payload)
        self._state.step_count = self._env.step_count
        return MaskguardOpenenvObservation(**observation, reward=reward, done=done, metadata=info)

    @property
    def state(self) -> State:
        return self._state
