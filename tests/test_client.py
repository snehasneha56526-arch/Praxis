"""
tests/test_client.py - PraxisEnv HTTP client contract tests.
"""

import pytest

from praxis_env import PraxisAction
from praxis_env.client import PraxisEnv


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.last_step_headers: dict[str, str] | None = None
        self.last_state_headers: dict[str, str] | None = None
        self.closed = False

    async def post(
        self, path: str, json: dict | None = None, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        if path == "/reset":
            return _FakeResponse(
                {
                    "session_id": "session-123",
                    "observation": {
                        "alert_summary": "alert",
                        "system_status": {"auth": "critical"},
                        "investigation_result": "",
                        "available_commands": [
                            "query_logs service=<name> timerange=<N>m"
                        ],
                        "time_elapsed_minutes": 0.0,
                        "severity": "P2",
                        "services_affected": ["auth"],
                        "step_number": 0,
                        "memory_active": False,
                        "saved_findings_count": 0,
                    },
                }
            )
        if path == "/step":
            self.last_step_headers = headers
            return _FakeResponse(
                {
                    "observation": {
                        "alert_summary": "alert",
                        "system_status": {"auth": "degraded"},
                        "investigation_result": "ok",
                        "available_commands": [
                            "query_logs service=<name> timerange=<N>m"
                        ],
                        "time_elapsed_minutes": 2.5,
                        "severity": "P2",
                        "services_affected": ["auth"],
                        "step_number": 1,
                        "memory_active": False,
                        "saved_findings_count": 0,
                    },
                    "reward": 0.11,
                    "done": False,
                    "info": {},
                }
            )
        raise AssertionError(f"Unexpected POST path: {path}")

    async def get(
        self, path: str, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        if path == "/state":
            self.last_state_headers = headers
            return _FakeResponse(
                {
                    "episode_id": "single-service-alert_1",
                    "step_count": 1,
                    "task_name": "single-service-alert",
                    "incident_resolved": False,
                    "root_cause_identified": False,
                    "cumulative_reward": 0.12,
                    "session_id": "session-123",
                    "memory_active": False,
                }
            )
        raise AssertionError(f"Unexpected GET path: {path}")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_uses_session_header_for_step_and_state(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeAsyncClient()
    monkeypatch.setattr("praxis_env.client.httpx.AsyncClient", lambda **_: fake)

    env = await PraxisEnv.from_url("http://127.0.0.1:7860")
    _ = await env.reset(task_name="single-service-alert")
    _ = await env.step(PraxisAction(command="query_logs service=auth timerange=5m"))
    _ = await env.get_state()

    assert fake.last_step_headers == {"x-session-id": "session-123"}
    assert fake.last_state_headers == {"x-session-id": "session-123"}

    await env.close()
    assert fake.closed is True
