"""
server.reward - Centralized reward engine for Praxis scenarios.

Phase 6 introduces a single scoring module so scenario classes only emit
semantic reward events (investigation, diagnosis, remediation, escalation)
instead of carrying duplicated numeric constants.

All returned rewards are clamped to a judge-safe open interval [0.01, 0.99].

Calibration rationale (updated for difficulty-curve fix):
  - Easy (single-service-alert):
      Target optimal: ~0.63. Investigation rewards are generous so even
      a 4-step agent scores well.  This is the "on-ramp" task.
  - Medium (ambiguous-incident):
      Target optimal: ~0.71 deterministic path. The agent must correlate
      signals across app and infra before diagnosis is rewarded.
  - Hard (cascading-failure, memory-leak):
      Target optimal: ~0.46 to ~0.48. Investigation is still rewarded,
      but stronger step pressure and lower remediation margins penalize
      wandering and reward disciplined triage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


MIN_REWARD = 0.01
MAX_REWARD = 0.99


def clamp_reward(value: float) -> float:
    """Clamp score to the judge-safe open interval [0.01, 0.99]."""
    return max(MIN_REWARD, min(MAX_REWARD, float(value)))


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component score contribution for one agent action."""

    investigation_reward: float = 0.0
    redundancy_penalty: float = 0.0
    diagnosis_reward: float = 0.0
    diagnosis_penalty: float = 0.0
    remediation_reward: float = 0.0
    destructive_penalty: float = 0.0
    efficiency_bonus: float = 0.0
    escalation_reward: float = 0.0
    premature_penalty: float = 0.0
    time_pressure_cost: float = 0.0
    total_unclamped: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "investigation_reward": self.investigation_reward,
            "redundancy_penalty": self.redundancy_penalty,
            "diagnosis_reward": self.diagnosis_reward,
            "diagnosis_penalty": self.diagnosis_penalty,
            "remediation_reward": self.remediation_reward,
            "destructive_penalty": self.destructive_penalty,
            "efficiency_bonus": self.efficiency_bonus,
            "escalation_reward": self.escalation_reward,
            "premature_penalty": self.premature_penalty,
            "time_pressure_cost": self.time_pressure_cost,
            "total_unclamped": self.total_unclamped,
        }


@dataclass(frozen=True)
class RewardResult:
    """Final clamped reward and detailed component accounting."""

    reward: float
    breakdown: RewardBreakdown


@dataclass(frozen=True)
class RewardPolicy:
    """Task-level reward policy and tuning knobs."""

    event_values: Mapping[str, float]
    redundancy_penalty: float = -0.02
    premature_penalty: float = -0.05
    destructive_penalty: float = -0.15
    efficiency_bonus_max: float = 0.0
    time_pressure_cost_per_step: float = 0.0


_MEMORY_EVENTS: Mapping[str, float] = {
    "memory.save_finding.before_cutoff": 0.05,
    "memory.save_finding.after_cutoff": 0.02,
    "memory.recall_memory.before_cutoff": 0.01,
    "memory.recall_memory.after_cutoff": 0.08,
    "memory.illegal_log_after_cutoff": -0.05,
    "memory.empty_recall_after_cutoff": -0.02,
}


def _with_memory_events(events: Mapping[str, float]) -> dict[str, float]:
    """Return a new event-value mapping with cross-task memory event rows."""
    merged = dict(events)
    merged.update(_MEMORY_EVENTS)
    return merged


# ── Reward Calibration ──────────────────────────────────────────────────────
#
# Each value is chosen to produce a target difficulty curve:
#   easy  >  medium  >  hard   (when scored by a frontier model)
#
# Key design decisions:
#   - Investigation rewards scale with diagnostic value: checking the
#     service closest to the root cause gives more signal.
#   - Diagnosis is worth 0.15-0.20 depending on difficulty — enough to
#     matter but not so much that a lucky guess dominates.
#   - Remediation is the largest single reward to incentivize taking
#     correct action, not just identifying the problem.
#   - check_runbook rewards institutional-knowledge usage (real on-call
#     engineers consult runbooks before guessing).
#   - Step cost is 0 for easy (no pressure), mild for medium/hard
#     (discourages aimless exploration).
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_REWARD_POLICIES: dict[str, RewardPolicy] = {
    # ── EASY: single-service-alert ──────────────────────────────────────
    # Target optimal path: ~0.63 in 4 steps.
    # Generous investigation rewards so partial investigation is well-rewarded.
    # No step cost — the easy task is forgiving by design.
    "single-service-alert": RewardPolicy(
        event_values=_with_memory_events(
            {
                # Investigation — generous for the easy task
                "investigation.query_logs.auth": 0.08,  # key service
                "investigation.query_logs.default": 0.05,  # exploring is okay
                "investigation.check_metrics.connections": 0.08,
                "investigation.check_metrics.default": 0.05,
                "investigation.check_deps.default": 0.05,
                "investigation.check_config.auth": 0.10,  # high-value: reveals the config typo
                "investigation.check_config.default": 0.03,
                "investigation.check_runbook.default": 0.05,  # consulting runbook is rewarded
                # Diagnosis
                "diagnosis.correct": 0.20,
                "diagnosis.wrong": 0.0,
                # Remediation
                "remediation.rollback_deploy.auth": 0.25,  # correct fix = highest single reward
                "remediation.wrong": 0.0,
                # Escalation
                "escalation.with_evidence": 0.15,
                "escalation.no_evidence": 0.0,
                # Error handling
                "unknown_command": 0.0,
                "invalid_input": 0.0,
            }
        ),
        # Easy task: no step cost, mild penalties
        time_pressure_cost_per_step=0.0,
    ),
    # ── HARD: cascading-failure ─────────────────────────────────────────
    # Target optimal path: ~0.46 in 7 steps.
    # Investigation rewards are lower — must follow the dependency chain.
    # Stronger step cost penalizes wandering through red herrings.
    "cascading-failure": RewardPolicy(
        event_values=_with_memory_events(
            {
                # Investigation — lower rewards, must multi-hop to find root cause
                "investigation.query_logs.api": 0.03,  # symptom service, low value
                "investigation.query_logs.database": 0.05,  # closer to root cause
                "investigation.query_logs.analytics": 0.05,  # reveals the runaway query
                "investigation.query_logs.default": 0.02,  # exploring other services
                "investigation.check_metrics.database.connections": 0.08,  # key metric
                "investigation.check_metrics.default": 0.02,
                "investigation.check_deps.core": 0.03,  # reveals db dependency
                "investigation.check_deps.default": 0.02,
                "investigation.check_config.database": 0.03,
                "investigation.check_config.analytics": 0.03,
                "investigation.check_config.default": 0.02,
                "investigation.check_runbook.default": 0.03,
                # Diagnosis — moderate, must earn it through investigation
                "diagnosis.correct": 0.14,
                "diagnosis.wrong": 0.0,
                # Remediation — both needed for full resolution
                "remediation.kill_query.database": 0.09,  # stop the bleeding
                "remediation.scale_resource.database.connection_pool": 0.08,  # prevent recurrence
                "remediation.wrong": 0.0,
                # Escalation
                "escalation.with_evidence": 0.09,
                "escalation.no_evidence": 0.0,
                # Error handling
                "unknown_command": 0.0,
                "invalid_input": 0.0,
            }
        ),
        # Hard task: stronger step cost discourages aimless exploration
        time_pressure_cost_per_step=0.006,
    ),
    # ── MEDIUM: ambiguous-incident ──────────────────────────────────────
    # Target optimal path: ~0.71 deterministic, requiring cross-service
    # evidence correlation but with less harsh remediation pressure than
    # the hard tasks.
    "ambiguous-incident": RewardPolicy(
        event_values=_with_memory_events(
            {
                # Investigation — must check 3+ app services + infra
                "investigation.query_logs.app": 0.048,
                "investigation.query_logs.dns-resolver": 0.095,
                "investigation.query_logs.default": 0.028,
                "investigation.check_metrics.dns-resolver.resolution_failures": 0.095,
                "investigation.check_metrics.app": 0.028,
                "investigation.check_metrics.load-balancer": 0.018,
                "investigation.check_metrics.default": 0.018,
                "investigation.check_deps.default": 0.028,
                "investigation.check_config.dns-resolver": 0.045,
                "investigation.check_config.app": 0.018,
                "investigation.check_config.default": 0.01,
                "investigation.check_runbook.default": 0.028,
                # Diagnosis
                "diagnosis.correct": 0.19,
                "diagnosis.wrong": 0.0,
                # Remediation
                "remediation.restart_service.dns-resolver": 0.14,
                "remediation.wrong": 0.0,
                # Escalation
                "escalation.with_evidence": 0.14,
                "escalation.no_evidence": 0.0,
                # Error handling
                "unknown_command": 0.0,
                "invalid_input": 0.0,
            }
        ),
        # Medium task: mild step cost
        time_pressure_cost_per_step=0.003,
    ),
    # ── HARD: memory-leak ───────────────────────────────────────────────
    # Requires checking memory metrics and config to find the OOM cause.
    # Target optimal path: ~0.48 in 5 steps.
    "memory-leak": RewardPolicy(
        event_values=_with_memory_events(
            {
                # Investigation
                "investigation.query_logs.worker": 0.04,
                "investigation.query_logs.default": 0.02,
                "investigation.check_metrics.worker.memory": 0.09,
                "investigation.check_metrics.default": 0.02,
                "investigation.check_deps.default": 0.02,
                "investigation.check_config.worker": 0.04,
                "investigation.check_config.default": 0.02,
                "investigation.check_runbook.default": 0.03,
                # Diagnosis
                "diagnosis.correct": 0.14,
                "diagnosis.wrong": 0.0,
                # Remediation
                "remediation.rollback_deploy.worker": 0.19,
                "remediation.scale_resource.worker.memory": 0.19,
                "remediation.wrong": 0.0,
                # Escalation
                "escalation.with_evidence": 0.10,
                "escalation.no_evidence": 0.0,
                # Error handling
                "unknown_command": 0.0,
                "invalid_input": 0.0,
            }
        ),
        time_pressure_cost_per_step=0.005,
    ),
}


class RewardEngine:
    """Deterministic event-based reward calculator shared by all scenarios."""

    def __init__(self, policies: Mapping[str, RewardPolicy] | None = None) -> None:
        self._policies = dict(policies or DEFAULT_REWARD_POLICIES)

    def score(
        self,
        *,
        task_name: str,
        event: str,
        duplicate: bool = False,
        premature: bool = False,
        destructive: bool = False,
        root_cause_identified: bool = True,
        resolved: bool = False,
        step_number: int = 1,
        max_steps: int = 1,
    ) -> RewardResult:
        """
        Score a single scenario event.

        Args:
            task_name: Scenario/task identifier.
            event: Canonical event key, e.g. "diagnosis.correct".
            duplicate: True if this action repeats already-seen evidence.
            premature: True if action happened before evidence threshold.
            destructive: True for materially harmful wrong remediations.
            root_cause_identified: Whether root cause was diagnosed already.
            resolved: True when this action resolves or ends the incident.
            step_number: 1-based action index for optional timing bonuses.
            max_steps: Episode step limit.
        """
        policy = self._policies.get(task_name)
        if policy is None:
            available = ", ".join(sorted(self._policies.keys()))
            raise ValueError(
                f"Unknown reward policy for task '{task_name}'. "
                f"Available policies: [{available}]"
            )

        if event.startswith("remediation.") and not root_cause_identified:
            breakdown = RewardBreakdown(total_unclamped=0.0)
            return RewardResult(
                reward=clamp_reward(breakdown.total_unclamped),
                breakdown=breakdown,
            )

        event_value = policy.event_values.get(event, 0.0)
        effective_value = 0.0 if duplicate else event_value

        investigation_reward = 0.0
        diagnosis_reward = 0.0
        diagnosis_penalty = 0.0
        remediation_reward = 0.0
        escalation_reward = 0.0

        if event.startswith("investigation."):
            investigation_reward = effective_value
        elif event.startswith("memory."):
            investigation_reward = effective_value
        elif event == "diagnosis.correct":
            diagnosis_reward = effective_value
        elif event.startswith("diagnosis."):
            diagnosis_penalty = effective_value
        elif event.startswith("remediation."):
            remediation_reward = effective_value
        elif event.startswith("escalation."):
            escalation_reward = effective_value

        redundancy_penalty = policy.redundancy_penalty if duplicate else 0.0
        premature_penalty = policy.premature_penalty if premature else 0.0
        destructive_penalty = policy.destructive_penalty if destructive else 0.0

        # Step cost: mild per-step penalty that discourages aimless exploration
        time_pressure_cost = (
            -policy.time_pressure_cost_per_step
            if policy.time_pressure_cost_per_step > 0.0
            else 0.0
        )

        efficiency_bonus = 0.0
        if resolved and policy.efficiency_bonus_max > 0.0 and max_steps > 0:
            bounded_step = min(max(step_number, 1), max_steps)
            progress = bounded_step / max_steps
            efficiency_bonus = policy.efficiency_bonus_max * (1.0 - progress)

        total_unclamped = (
            investigation_reward
            + redundancy_penalty
            + diagnosis_reward
            + diagnosis_penalty
            + remediation_reward
            + destructive_penalty
            + efficiency_bonus
            + escalation_reward
            + premature_penalty
            + time_pressure_cost
        )

        breakdown = RewardBreakdown(
            investigation_reward=investigation_reward,
            redundancy_penalty=redundancy_penalty,
            diagnosis_reward=diagnosis_reward,
            diagnosis_penalty=diagnosis_penalty,
            remediation_reward=remediation_reward,
            destructive_penalty=destructive_penalty,
            efficiency_bonus=efficiency_bonus,
            escalation_reward=escalation_reward,
            premature_penalty=premature_penalty,
            time_pressure_cost=time_pressure_cost,
            total_unclamped=total_unclamped,
        )

        return RewardResult(
            reward=clamp_reward(total_unclamped),
            breakdown=breakdown,
        )
