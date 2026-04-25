"""
server.praxis_environment — Core environment logic.

This module implements the Environment interface expected by openenv-core.
It is loaded by server/app.py and mounted into the FastAPI application.

Command parsing is delegated to server.command_parser — this module never
does raw string parsing itself.

Flow:
    POST /reset  → praxis_environment.reset() → PraxisObservation
    POST /step   → praxis_environment.step()  → StepResult
    GET  /state  → praxis_environment.state() → PraxisState

The environment delegates all scenario-specific logic to the active
BaseScenario instance (loaded from the scenario registry on reset).

Design choices:
    - The environment is stateful: one active scenario per server instance
    - Thread safety: single-threaded sequential steps (not concurrent)
    - Episode ID: generated as "{task_name}_{count}" for deterministic tracing
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from praxis_env.models import (
    PraxisAction,
    PraxisObservation,
    PraxisState,
    StepOutcome,
    ensure_ascii_text,
)
from praxis_env.memory import PraxisMemory
from praxis_env.scenarios import get_scenario, list_tasks
from praxis_env.scenarios.base import BaseScenario
from server.command_parser import parse_command
from server.reward import MAX_REWARD, MIN_REWARD, RewardEngine

logger = logging.getLogger(__name__)


TASK_NAME_ALIASES: dict[str, str] = {
    "easy": "single-service-alert",
    "medium": "ambiguous-incident",
    "hard": "cascading-failure",
}
PROCEDURAL_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


class PraxisEnvironment:
    """
    Stateful environment controller.

    Manages one active scenario at a time. The scenario tracks all
    episode state; this class is responsible for:
      - Parsing commands from PraxisAction
      - Routing commands to the active scenario
      - Serialising scenario outcomes back to OpenEnv types

    Usage (in server/app.py):
        env = PraxisEnvironment()
        obs = env.reset(task_name="single-service-alert")
        result = env.step(PraxisAction(command="query_logs service=auth timerange=5m"))
        state = env.state()
    """

    # mirrors openenv.core.env_server.interfaces.Environment.SUPPORTS_CONCURRENT_SESSIONS (S14)
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    REQUIRES_SINGLE_THREAD_EXECUTOR: bool = False

    def __init__(self) -> None:
        self._scenario: BaseScenario | None = None
        self._episode_count: int = 0
        self._last_reset_metadata: dict[str, Any] = {}
        self._memory = PraxisMemory()
        self._reward_engine = RewardEngine()
        self._investigation_history: list[str] = []
        self._session_id: str = ""

    @staticmethod
    def resolve_task_name(task_name: str) -> str:
        """Resolve user-facing aliases to canonical scenario task names."""
        normalized = (task_name or "single-service-alert").strip().lower()
        if not normalized:
            normalized = "single-service-alert"
        return TASK_NAME_ALIASES.get(normalized, normalized)

    @staticmethod
    def resolve_task_request(task_name: str) -> tuple[str, str | None]:
        """Resolve canonical task name and optional procedural difficulty."""
        normalized = (task_name or "single-service-alert").strip().lower()
        if not normalized:
            normalized = "single-service-alert"

        if normalized.startswith("procedural-incident"):
            if normalized == "procedural-incident":
                return "procedural-incident", "medium"
            if ":" in normalized:
                _, _, suffix = normalized.partition(":")
                if suffix in PROCEDURAL_DIFFICULTIES:
                    return "procedural-incident", suffix
            if normalized.count("-") >= 2:
                suffix = normalized.rsplit("-", 1)[-1]
                if suffix in PROCEDURAL_DIFFICULTIES:
                    return "procedural-incident", suffix

        return TASK_NAME_ALIASES.get(normalized, normalized), None

    @property
    def last_reset_metadata(self) -> dict[str, Any]:
        return dict(self._last_reset_metadata)

    # ── Public API (called by FastAPI routes) ─────────────────────────────────

    def reset(
        self,
        task_name: str = "single-service-alert",
        seed: int | None = None,
        session_id: str = "",
    ) -> PraxisObservation:
        """
        Start a new episode with the named scenario.

        Args:
            task_name: Scenario to load (see list_tasks() for options).
            seed: Optional deterministic seed for procedural scenarios.

        Returns:
            Initial PraxisObservation with the incident alert and system status.

        Raises:
            ValueError: if task_name is not registered.
        """
        canonical_task_name, procedural_difficulty = self.resolve_task_request(
            task_name
        )
        resolved_seed = seed

        self._episode_count += 1
        episode_id = f"{canonical_task_name}_{self._episode_count}"

        logger.info(
            "reset() -> episode_id=%s task=%s requested_task=%s seed=%s",
            episode_id,
            canonical_task_name,
            task_name,
            resolved_seed,
        )

        if canonical_task_name == "procedural-incident":
            if resolved_seed is None:
                resolved_seed = random.randint(0, 2**31 - 1)
            self._scenario = get_scenario(
                canonical_task_name,
                seed=resolved_seed,
                difficulty=procedural_difficulty,
            )
        else:
            self._scenario = get_scenario(canonical_task_name)

        self._last_reset_metadata = {
            "seed": resolved_seed,
            "difficulty": (
                procedural_difficulty
                if canonical_task_name == "procedural-incident"
                else None
            ),
            "task_name": canonical_task_name,
        }
        cutoff = getattr(
            self._scenario,
            "MEMORY_CUTOFF_OVERRIDE",
            PraxisMemory.CONTEXT_CUTOFF_STEP,
        )
        self._memory.CONTEXT_CUTOFF_STEP = int(cutoff)
        self._memory.reset()
        self._investigation_history = []
        self._session_id = session_id
        self._scenario.reset(episode_id=episode_id)

        obs = self._scenario.get_observation()
        # Override investigation_result with scenario's initial text
        obs.investigation_result = ensure_ascii_text(
            self._scenario.get_initial_observation_text()
        )
        obs.memory_active = self._memory.is_active(self._scenario._step_count)
        obs.saved_findings_count = len(self._memory.saved_findings)
        return obs

    def step(self, action: PraxisAction) -> dict[str, Any]:
        """
        Execute one action and return the raw result dict.

        Args:
            action: PraxisAction with a command string.

        Returns:
            Dict with keys: observation, reward, done, info
            (serialisable to JSON for the FastAPI response).

        Raises:
            RuntimeError: if called before reset().
        """
        if self._scenario is None:
            raise RuntimeError("step() called before reset(). Call reset() first.")

        if self._scenario.clamp_reward(self._scenario._cumulative_reward) >= MAX_REWARD:
            obs = self._scenario.get_observation()
            obs.step_number = self._scenario._step_count
            return {
                "observation": self._obs_to_dict(obs),
                "reward": MIN_REWARD,
                "done": True,
                "info": {
                    "error": "episode_score_cap_reached",
                    "score_cap_reached": True,
                },
            }

        logger.debug("step() → command=%r", action.command)

        # The environment owns step_count and cumulative reward bookkeeping.
        # Scenarios return domain outcomes without mutating those counters.
        # Parse command string -> structured ParsedCommand
        parsed = parse_command(action.command)
        current_step = self._scenario._step_count
        reward_event: str | None = None

        if parsed.action_type in {"save_finding", "recall_memory"}:
            outcome, reward_event = self._handle_memory_action(
                parsed.action_type, parsed.params
            )
        elif (
            parsed.action_type in {"query_logs", "check_logs"}
            and current_step >= self._memory.CONTEXT_CUTOFF_STEP
        ):
            reward_event = "memory.illegal_log_after_cutoff"
            result_text = (
                "Log access is disabled after context cutoff.\n"
                "[CONTEXT LIMIT] Use save_finding and recall_memory instead."
            )
            reward = self._score_memory_event(reward_event)
            outcome = StepOutcome(
                investigation_result=result_text,
                reward=reward,
                done=self._scenario.is_done(),
                incident_resolved=self._scenario._incident_resolved,
                root_cause_identified=self._scenario._root_cause_identified,
                info={"event": reward_event},
            )
        else:
            # Delegate to active scenario
            outcome = self._scenario.step(parsed)
        raw_step_reward = self._scenario.clamp_reward(outcome.reward)
        current_cumulative_reward = self._scenario.clamp_reward(
            self._scenario._cumulative_reward
        )

        # The environment is the final authority for emitted rewards: once the
        # episode budget is nearly exhausted, trim the outgoing reward so the
        # cumulative task score exposed to validators remains strictly < 1.0.
        remaining_budget = max(0.0, MAX_REWARD - current_cumulative_reward)
        step_reward = min(raw_step_reward, remaining_budget)
        # Guard against rounding-to-zero edge cases: if there's still some
        # remaining budget, always emit a strictly-positive reward.
        if 0.0 < remaining_budget and step_reward <= 0.0:
            step_reward = remaining_budget
        # Ensure emitted reward is strictly within (0, 1) even after any
        # downstream float formatting/serialization.
        if step_reward <= 0.0:
            step_reward = math.nextafter(0.0, 1.0)
        elif step_reward >= 1.0:
            step_reward = math.nextafter(1.0, 0.0)
        next_cumulative_reward = self._scenario.clamp_reward(
            current_cumulative_reward + step_reward
        )
        score_cap_reached = next_cumulative_reward >= MAX_REWARD

        # Update scenario's investigation result for next observation
        self._scenario._last_investigation_result = outcome.investigation_result
        self._investigation_history.append(outcome.investigation_result)
        self._scenario._step_count += 1
        self._scenario._cumulative_reward = next_cumulative_reward

        # Build the next observation
        obs = self._scenario.get_observation()
        # Override step_number to reflect the step just taken
        obs.step_number = self._scenario._step_count
        memory_active = self._memory.is_active(self._scenario._step_count)
        if memory_active:
            obs.investigation_result = self._memory.get_observation_context(
                self._investigation_history,
                self._scenario._step_count,
            )
        obs.memory_active = memory_active
        obs.saved_findings_count = len(self._memory.saved_findings)

        info = dict(outcome.info or {})
        if reward_event is not None:
            info["event"] = reward_event
        if score_cap_reached:
            info["score_cap_reached"] = True

        logger.debug(
            "step() → reward=%.3f done=%s step=%d",
            step_reward,
            outcome.done or self._scenario.is_done() or score_cap_reached,
            obs.step_number,
        )

        result = {
            "observation": self._obs_to_dict(obs),
            "reward": step_reward,
            "done": outcome.done or self._scenario.is_done() or score_cap_reached,
            "info": info,
        }
        return result

    def state(self) -> PraxisState:
        """
        Return current episode metadata without the full observation.

        Returns:
            PraxisState snapshot.

        Raises:
            RuntimeError: if called before reset().
        """
        if self._scenario is None:
            raise RuntimeError("state() called before reset(). Call reset() first.")
        state = self._scenario.get_state()
        state.memory_active = self._memory.is_active(self._scenario._step_count)
        state.session_id = self._session_id
        return state

    def list_tasks(self) -> list[str]:
        """Return all available task names."""
        return list_tasks()

    @staticmethod
    def _obs_to_dict(obs: PraxisObservation) -> dict[str, Any]:
        """Convert PraxisObservation to a JSON-serialisable dict."""
        return {
            "alert_summary": obs.alert_summary,
            "system_status": obs.system_status,
            "investigation_result": obs.investigation_result,
            "available_commands": obs.available_commands,
            "time_elapsed_minutes": obs.time_elapsed_minutes,
            "severity": obs.severity,
            "services_affected": obs.services_affected,
            "step_number": obs.step_number,
            "memory_active": obs.memory_active,
            "saved_findings_count": obs.saved_findings_count,
        }

    def _score_memory_event(self, event: str) -> float:
        """Score memory event tags through the shared reward engine."""
        if self._scenario is None:
            raise RuntimeError("Cannot score memory event before reset().")
        result = self._reward_engine.score(
            task_name=self._scenario.NAME,
            event=event,
            step_number=self._scenario._step_count + 1,
            max_steps=self._scenario.MAX_STEPS,
            root_cause_identified=self._scenario._root_cause_identified,
        )
        return result.reward

    def _handle_memory_action(
        self,
        action_type: str,
        params: dict[str, str],
    ) -> tuple[StepOutcome, str]:
        """Execute a memory command and return deterministic outcome + event tag."""
        if self._scenario is None:
            raise RuntimeError("Cannot handle memory action before reset().")

        cutoff_state = (
            "after_cutoff"
            if self._memory.is_active(self._scenario._step_count)
            else "before_cutoff"
        )

        if action_type == "save_finding":
            key = params.get("key", "").strip()
            value = params.get("value", "").strip()
            if not key or not value:
                event = "invalid_input"
                result_text = (
                    "Invalid save_finding command.\n"
                    "Expected: save_finding key=<key> value=<finding>"
                )
                reward = self._score_memory_event(event)
            else:
                event = f"memory.save_finding.{cutoff_state}"
                result_text = self._memory.save_finding(key, value)
                reward = self._score_memory_event(event)
        else:
            key = params.get("key")
            has_findings = bool(self._memory.saved_findings)
            if cutoff_state == "after_cutoff" and not has_findings:
                event = "memory.empty_recall_after_cutoff"
            else:
                event = f"memory.recall_memory.{cutoff_state}"
            result_text = self._memory.recall_memory(key=key)
            reward = self._score_memory_event(event)

        outcome = StepOutcome(
            investigation_result=result_text,
            reward=reward,
            done=self._scenario.is_done(),
            incident_resolved=self._scenario._incident_resolved,
            root_cause_identified=self._scenario._root_cause_identified,
            info={"event": event},
        )
        return outcome, event
