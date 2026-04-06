"""Simple HTTP client for the MaskGuardEnv server endpoints."""

from __future__ import annotations

from typing import Any, Dict

import requests


class MaskguardOpenenvEnv:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, text: str | None = None, policy_mode: str = "GDPR", expected_entities: list[str] | None = None) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"text": text, "policy_mode": policy_mode, "expected_entities": expected_entities or []},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        response.raise_for_status()
        return response.json()

    def submit(self) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/submit", timeout=30)
        response.raise_for_status()
        return response.json()
