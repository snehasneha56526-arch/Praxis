"""
praxis_env.scenarios.base — Base class for all incident scenarios.

Every scenario is a pure state machine with NO randomness — same actions always
produce the same observations and rewards (determinism guarantee required by
the judging system).

Design:
  - Scenarios own all their data (logs, metrics, topology) as class attributes
  - All data is generated at class definition time, not at runtime
  - Episode state is instance state, reset cleanly on each reset() call
  - The environment server calls scenario methods; scenarios never call back

Extending Praxis with new scenarios:
  1. Create a new file in praxis_env/scenarios/
  2. Subclass BaseScenario
  3. Implement all abstract methods
  4. Register in praxis_env/scenarios/__init__.py
  See docs/developer/adding-scenarios.md for the full guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

from praxis_env.models import (
    AVAILABLE_COMMANDS,
    ParsedCommand,
    PraxisObservation,
    PraxisState,
    StepOutcome,
)
from server.reward import RewardEngine, RewardResult


class BaseScenario(ABC):
    """
    Abstract base for all Praxis incident scenarios.

    Lifecycle:
        scenario = SomeScenario()         # Instantiate
        scenario.reset()                   # Initialise clean episode state
        outcome = scenario.step(cmd)       # Process each action
        obs = scenario.get_observation()   # Get current observation

    Subclass contract:
        - Implement all @abstractmethod methods
        - Set class attributes: NAME, SEVERITY, MAX_STEPS, ALERT_SUMMARY,
          INITIAL_SYSTEM_STATUS, INITIAL_AFFECTED_SERVICES
        - Keep ALL state in instance variables set in reset()
        - No randomness — deterministic state machine only
    """

    # ── Class-level constants (set by each subclass) ─────────────────────────
    NAME: str = "unnamed-scenario"
    SEVERITY: str = "P2"
    MAX_STEPS: int = 15
    ALERT_SUMMARY: str = "No alert defined"
    INITIAL_SYSTEM_STATUS: dict[str, str] = {}
    INITIAL_AFFECTED_SERVICES: list[str] = []

    def __init__(self) -> None:
        # Episode instance state — reset each time
        self._step_count: int = 0
        self._cumulative_reward: float = 0.01
        self._incident_resolved: bool = False
        self._root_cause_identified: bool = False
        self._investigation_history: list[str] = []
        self._last_investigation_result: str = ""
        self._current_system_status: dict[str, str] = {}
        self._episode_id: str = ""
        self._reward_engine = RewardEngine()

    def reset(self, episode_id: str = "") -> None:
        """
        Reset to a fresh episode. Called by the environment server on reset().

        Args:
            episode_id: Unique identifier for this episode (provided by server)
        """
        self._step_count = 0
        self._cumulative_reward = 0.01
        self._incident_resolved = False
        self._root_cause_identified = False
        self._investigation_history = []
        self._last_investigation_result = ""
        self._current_system_status = dict(self.INITIAL_SYSTEM_STATUS)
        self._episode_id = episode_id
        self._reset_scenario_state()

    @abstractmethod
    def _reset_scenario_state(self) -> None:
        """
        Subclass hook: reset any scenario-specific instance state.
        Called at end of reset(). Set your scenario-specific variables here.
        """
        ...

    @abstractmethod
    def step(self, command: ParsedCommand) -> StepOutcome:
        """
        Process one parsed command and return the outcome.

        Args:
            command: Parsed action with action_type and params

        Returns:
            StepOutcome with result text, reward, done flag, and state flags

        Implementation rules:
            - NEVER raise exceptions — return an error StepOutcome instead
            - NEVER use randomness — pure function over deterministic state
            - ALWAYS clamp reward to [0.01, 0.99] before returning
            - Do not mutate self._step_count or cumulative reward here;
              PraxisEnvironment applies that bookkeeping after the outcome
              is returned.
        """
        ...

    @abstractmethod
    def get_initial_observation_text(self) -> str:
        """
        Returns the investigation_result text shown on the FIRST observation
        (before any agent action). Typically empty string or a brief intro.
        """
        ...

    # ── Shared helpers (available to all subclasses) ─────────────────────────

    def get_observation(self) -> PraxisObservation:
        """Build the current PraxisObservation from scenario state."""
        current_severity = self.SEVERITY
        # Escalate severity if unresolved and past 70% of max steps
        if not self._incident_resolved and self._step_count >= self.MAX_STEPS * 0.7:
            if current_severity == "P3":
                current_severity = "P2"
            elif current_severity == "P2":
                current_severity = "P1"
            elif current_severity == "P1":
                current_severity = "P0"

        return PraxisObservation(
            alert_summary=self.ALERT_SUMMARY,
            system_status=dict(self._current_system_status),
            investigation_result=self._last_investigation_result,
            available_commands=list(AVAILABLE_COMMANDS),
            time_elapsed_minutes=float(self._step_count * 2.5),  # 2.5 min per step
            severity=current_severity,
            services_affected=[
                service
                for service in self.INITIAL_AFFECTED_SERVICES
                if self._current_system_status.get(service) != "healthy"
            ],
            step_number=self._step_count,
            memory_active=False,
            saved_findings_count=0,
        )

    def get_state(self) -> PraxisState:
        """Build the current PraxisState."""
        return PraxisState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self.NAME,
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
            cumulative_reward=self.clamp_reward(self._cumulative_reward),
            session_id="",
            memory_active=False,
        )

    def is_done(self) -> bool:
        """True if episode has ended (resolved, escalated, or max steps reached)."""
        return self._incident_resolved or self._step_count >= self.MAX_STEPS

    @staticmethod
    def clamp_reward(reward: float) -> float:
        """Clamp reward to the judge-safe open interval (0, 1)."""
        value = max(0.01, min(0.99, float(reward)))
        # Defensive: ensure floating-point serialization cannot round to boundaries.
        if value <= 0.0:
            return math.nextafter(0.0, 1.0)
        if value >= 1.0:
            return math.nextafter(1.0, 0.0)
        return value

    def _score_event(
        self,
        event: str,
        *,
        duplicate: bool = False,
        premature: bool = False,
        destructive: bool = False,
        resolved: bool = False,
    ) -> RewardResult:
        """Delegate reward computation to the centralized reward engine."""
        return self._reward_engine.score(
            task_name=self.NAME,
            event=event,
            duplicate=duplicate,
            premature=premature,
            destructive=destructive,
            resolved=resolved,
            step_number=self._step_count + 1,
            max_steps=self.MAX_STEPS,
        )

    def _handle_unknown_command(self, raw_command: str) -> StepOutcome:
        """
        Standard response for unrecognised commands.
        Returns zero reward but does NOT crash.
        """
        available = "\n".join(f"  {cmd}" for cmd in AVAILABLE_COMMANDS)
        score = self._score_event("unknown_command")
        return StepOutcome(
            investigation_result=(
                f"Unknown command: '{raw_command}'\n\nAvailable commands:\n{available}"
            ),
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
            info={"error": "unknown_command", "raw": raw_command},
        )


# ── Param extraction helpers (used by all scenario step() handlers) ───────────
# Live here so scenarios import from praxis_env.scenarios.base,
# not from server/ (which would create a circular import).


def get_service_param(params: dict[str, str], default: str = "") -> str:
    """Extract and return the 'service' param, lowercased."""
    return params.get("service", default).lower().strip()


def get_metric_param(params: dict[str, str], default: str = "") -> str:
    """Extract and return the 'metric' param, lowercased."""
    return params.get("metric", default).lower().strip()


def get_timerange_minutes(params: dict[str, str], default: int = 5) -> int:
    """
    Parse the timerange param (e.g. '5m', '15m') into an integer number of minutes.
    Returns default if param is missing or unparseable.
    """
    raw_tr = params.get("timerange", "")
    if not raw_tr:
        return default
    try:
        return int(raw_tr.rstrip("m").strip())
    except ValueError:
        return default
