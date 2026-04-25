"""
praxis_env.client — PraxisEnv client for connecting to the running environment server.

Usage from inference.py:
    from praxis_env import PraxisEnv, PraxisAction

    # Connect to locally running server
    env = await PraxisEnv.from_url("http://localhost:7860")

    # Connect to HuggingFace Space
    env = await PraxisEnv.from_url("https://your-space.hf.space")

    # Run an episode
    obs = await env.reset(task_name="single-service-alert")
    result = await env.step(PraxisAction(command="query_logs service=auth timerange=5m"))
    await env.close()

Note: PraxisEnv wraps the HTTP API. The server runs in the container.
      The client is what inference.py uses to interact with the server.
"""

from __future__ import annotations

import httpx

from praxis_env.models import PraxisAction, PraxisObservation, PraxisState


class StepResult:
    """
    Result from a single env.step() call.

    Mirrors the OpenEnv StepResult contract:
        observation:  The new PraxisObservation after the action
        reward:       Per-step reward in [0.01, 0.99]
        done:         True if the episode has ended
        info:         Optional dict with extra debugging information
    """

    def __init__(
        self,
        observation: PraxisObservation,
        reward: float,
        done: bool,
        info: dict | None = None,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info: dict = info or {}

    def __repr__(self) -> str:
        return (
            f"StepResult(reward={self.reward:.3f}, done={self.done}, "
            f"step={self.observation.step_number})"
        )


class PraxisEnv:
    """
    HTTP client for the Praxis environment server.

    Wraps the FastAPI server running in Docker / HF Spaces.
    All methods are async to work naturally with the OpenAI async client.

    Example:
        async with PraxisEnv("http://localhost:7860") as env:
            obs = await env.reset("single-service-alert")
            result = await env.step(PraxisAction(command="query_logs service=auth timerange=5m"))
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._session_id: str = ""

    @classmethod
    async def from_url(cls, url: str, timeout: float = 30.0) -> "PraxisEnv":
        """Create and initialise a PraxisEnv client."""
        env = cls(url, timeout)
        await env._init()
        return env

    async def _init(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    async def reset(self, task_name: str = "single-service-alert") -> PraxisObservation:
        """
        Start a new episode.

        Args:
            task_name: One of "single-service-alert", "cascading-failure",
                       "ambiguous-incident"

        Returns:
            Initial PraxisObservation with the incident alert and system status.
        """
        assert self._client is not None, "Call from_url() before using the client"
        resp = await self._client.post("/reset", json={"task_name": task_name})
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data.get("session_id", "")
        payload = data.get("observation", data)
        return _parse_observation(payload)

    async def step(self, action: PraxisAction) -> StepResult:
        """
        Execute one action in the environment.

        Args:
            action: PraxisAction with a command string

        Returns:
            StepResult with observation, reward, done, and info.
        """
        assert self._client is not None, "Call from_url() before using the client"
        resp = await self._client.post(
            "/step",
            json={"command": action.command},
            headers=self._session_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=_parse_observation(data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            info=data.get("info", {}),
        )

    async def get_state(self) -> PraxisState:
        """
        Get the current episode state (lightweight metadata).

        Returns:
            PraxisState with episode_id, step_count, task_name, etc.
        """
        assert self._client is not None, "Call from_url() before using the client"
        resp = await self._client.get("/state", headers=self._session_headers())
        resp.raise_for_status()
        data = resp.json()
        return PraxisState(
            episode_id=data["episode_id"],
            step_count=data["step_count"],
            task_name=data["task_name"],
            incident_resolved=data.get("incident_resolved", False),
            root_cause_identified=data.get("root_cause_identified", False),
            cumulative_reward=data.get("cumulative_reward", 0.01),
            session_id=data.get("session_id", ""),
            memory_active=data.get("memory_active", False),
        )

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._session_id = ""

    async def __aenter__(self) -> "PraxisEnv":
        await self._init()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _session_headers(self) -> dict[str, str]:
        """Attach session affinity header for stateful server endpoints."""
        if not self._session_id:
            return {}
        return {"x-session-id": self._session_id}


def _parse_observation(data: dict) -> PraxisObservation:
    """Parse a raw API response dict into a PraxisObservation."""
    return PraxisObservation(
        alert_summary=data["alert_summary"],
        system_status=data["system_status"],
        investigation_result=data.get("investigation_result", ""),
        available_commands=data["available_commands"],
        time_elapsed_minutes=float(data["time_elapsed_minutes"]),
        severity=data["severity"],
        services_affected=data["services_affected"],
        step_number=int(data["step_number"]),
    )
