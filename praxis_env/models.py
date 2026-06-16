"""
praxis_env.models — Data models for Praxis environment.

Uses Pydantic BaseModel for full spec compliance with the OpenEnv typed model requirement.
All models are fully typed with Python 3.11+ type hints.

Design Decisions:
  - Pydantic BaseModel: provides robust validation and serialization
  - Field(default_factory=...) for mutable defaults (list, dict)
  - No Optional where possible — every field should have a defined value
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any
import unicodedata


_TEXT_PAYLOAD_TRANSLATIONS = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2248": "approx",
        "\u26a0": "[WARN]",
        "\u274c": "[FAIL]",
        "\u2705": "[OK]",
        "\U0001f6a8": "[ALERT]",
        "\ufe0f": "",
    }
)


def ensure_ascii_text(text: str) -> str:
    """
    Normalise human-facing payloads to ASCII for Windows console safety.

    The scenario content was authored with symbols such as emoji, arrows,
    and typographic punctuation. Those render poorly in some local Windows
    consoles, so the environment normalises outbound payloads before they
    are surfaced to tests, the HTTP layer, or local debugging output.
    """
    if not text:
        return text

    translated = text.translate(_TEXT_PAYLOAD_TRANSLATIONS)
    normalized = unicodedata.normalize("NFKD", translated)
    return normalized.encode("ascii", "ignore").decode("ascii")


class ParsedCommand(BaseModel):
    """Structured representation of a parsed action command string."""

    action_type: str
    params: dict[str, str] = Field(default_factory=dict)
    raw: str = ""


class StepOutcome(BaseModel):
    """
    Internal result from a scenario processing a command.
    Converted to HTTP response by the environment server.
    """

    investigation_result: str
    reward: float
    done: bool
    incident_resolved: bool
    root_cause_identified: bool
    info: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_text(self) -> "StepOutcome":
        self.investigation_result = ensure_ascii_text(self.investigation_result)
        return self


class PraxisAction(BaseModel):
    """
    What the agent sends to the environment each step.

    The action is a single command string that the environment parses
    into a structured operation. This design is LLM-friendly: the model
    generates natural text commands rather than structured JSON.

    Command grammar:
        query_logs service=<name> timerange=<N>m
        check_metrics service=<name> metric=<type>
        check_deps service=<name>
        check_config service=<name>
        diagnose root_cause=<cause>
        restart_service service=<name>
        rollback_deploy service=<name>
        scale_resource service=<name> resource=<type> [value=<N>]
        kill_query service=<name> query_id=<id>
        escalate reason=<text>
        save_finding key=<key> value=<finding>
        recall_memory
        recall_memory key=<key>

    Example:
        PraxisAction(command="query_logs service=auth timerange=5m")
        PraxisAction(command="diagnose root_cause=bad_config")
        PraxisAction(command="rollback_deploy service=auth")
    """

    command: str


class PraxisObservation(BaseModel):
    """
    What the agent receives after each action (or from reset()).

    Provides rich, text-based information that an LLM can reason about
    to decide the next action. Every field is always populated — no Nones.

    Fields:
        alert_summary:          Current incident description (markdown-formatted)
        system_status:          Service name → health status map
                                Status values: "healthy" | "degraded" | "critical" | "down"
        investigation_result:   Output of the last command (logs, metrics, dep graph, etc.)
                                Empty string on reset() (no previous command)
        available_commands:     List of valid command templates the agent can use
        time_elapsed_minutes:   Float minutes since incident started (increases each step)
        severity:               Incident severity: "P0" | "P1" | "P2" | "P3"
        services_affected:      List of services currently showing issues
        step_number:            Current step count (0 on reset, increments with each step)
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    alert_summary: str
    system_status: dict[str, str]
    investigation_result: str
    available_commands: list[str]
    time_elapsed_minutes: float
    severity: str
    services_affected: list[str]
    step_number: int
    memory_active: bool = False
    saved_findings_count: int = 0

    @model_validator(mode="after")
    def normalize_text(self) -> "PraxisObservation":
        # Use object.__setattr__ to avoid validate_assignment re-entry loops.
        object.__setattr__(self, "alert_summary", ensure_ascii_text(self.alert_summary))
        object.__setattr__(
            self,
            "investigation_result",
            ensure_ascii_text(self.investigation_result),
        )
        return self


class PraxisState(BaseModel):
    """
    Episode metadata exposed via the state() endpoint.

    Provides a lightweight snapshot of what's happening without the
    full observation payload. Useful for monitoring and debugging.

    Fields:
        episode_id:             Unique identifier for this episode (task_name + counter)
        step_count:             How many steps have been taken in this episode
        task_name:              The active scenario name
        incident_resolved:      True when the incident has been correctly remediated
        root_cause_identified:  True when agent has issued a correct diagnose command
        cumulative_reward:      Sum of rewards so far (for monitoring only)
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    step_count: int
    task_name: str
    incident_resolved: bool = False
    root_cause_identified: bool = False
    cumulative_reward: float = 0.01
    session_id: str = ""
    memory_active: bool = False


# ────────────────────────────────────────────────────────────────────────────
# Available commands advertised to the agent
# ────────────────────────────────────────────────────────────────────────────

AVAILABLE_COMMANDS: list[str] = [
    "query_logs service=<name> timerange=<N>m",
    "check_metrics service=<name> metric=<type>",
    "check_deps service=<name>",
    "check_config service=<name>",
    "check_runbook service=<name>",
    "diagnose root_cause=<cause>",
    "restart_service service=<name>",
    "rollback_deploy service=<name>",
    "scale_resource service=<name> resource=<type>",
    "kill_query service=<name> query_id=<id>",
    "escalate reason=<text>",
    "save_finding key=<key> value=<finding>",
    "recall_memory",
    "recall_memory key=<key>",
]

VALID_METRICS: frozenset[str] = frozenset(
    {
        "error_rate",
        "latency_p95",
        "throughput",
        "connections",
        "memory",
        "cpu",
        "resolution_failures",
        "cache_hit_rate",
    }
)

VALID_SEVERITIES: frozenset[str] = frozenset({"P0", "P1", "P2", "P3"})

VALID_SERVICE_STATUSES: frozenset[str] = frozenset(
    {
        "healthy",
        "degraded",
        "critical",
        "down",
    }
)
