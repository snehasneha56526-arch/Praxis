"""
praxis_env.scenarios.procedural_incident - Task 6: Seeded procedural incident.

Generates deterministic scenario content from (seed, difficulty), enabling
infinite reproducible incidents without global mutable state.
"""

from __future__ import annotations

import random

from praxis_env.scenarios.base import (
    BaseScenario,
    ParsedCommand,
    StepOutcome,
    get_metric_param,
    get_service_param,
)
from server.reward import RewardPolicy


class ProceduralIncidentScenario(BaseScenario):
    """Seeded incident generator with difficulty-aware reward calibration."""

    NAME = "procedural-incident"
    SEVERITY = "P2"
    MAX_STEPS = 25
    MEMORY_CUTOFF_OVERRIDE = 15

    ROOT_CAUSE_POOL = [
        "db_connection",
        "session_drift",
        "dns_config",
        "memory_leak",
        "network_partition",
        "cpu_exhaustion",
        "disk_full",
        "cert_expiry",
    ]
    RED_HERRING_POOL = [
        "cache_miss_spike",
        "queue_backlog",
        "scheduled_maintenance",
        "log_rotation",
        "metric_collector_lag",
        "cdn_rebalance",
    ]
    SERVICE_POOL = [
        "api",
        "auth",
        "database",
        "cache",
        "worker",
        "queue",
        "cdn",
        "dns",
        "search",
        "frontend",
    ]

    DIFFICULTY_CONFIG = {
        "easy": {
            "n_services": 1,
            "n_red_herrings": 0,
            "max_steps": 15,
            "memory_cutoff": 8,
        },
        "medium": {
            "n_services": 3,
            "n_red_herrings": 2,
            "max_steps": 25,
            "memory_cutoff": 15,
        },
        "hard": {
            "n_services": 5,
            "n_red_herrings": 4,
            "max_steps": 50,
            "memory_cutoff": 25,
        },
    }

    REMEDIATION_COMMAND_BY_ROOT_CAUSE: dict[str, str] = {
        "db_connection": "restart_service database",
        "session_drift": "rollback_deploy api",
        "dns_config": "restart_service dns",
        "memory_leak": "scale_resource worker",
        "network_partition": "restart_service api",
        "cpu_exhaustion": "scale_resource api",
        "disk_full": "scale_resource database",
        "cert_expiry": "rollback_deploy cdn",
    }

    def __init__(self, seed: int | None = None, difficulty: str = "medium"):
        super().__init__()
        resolved_difficulty = (difficulty or "medium").strip().lower()
        if resolved_difficulty not in self.DIFFICULTY_CONFIG:
            raise ValueError(
                f"Unknown procedural difficulty '{difficulty}'. "
                "Use one of: easy, medium, hard."
            )
        self._seed = int(seed) if seed is not None else None
        self._rng = random.Random(seed)
        self._difficulty = resolved_difficulty

    def _build_procedural_policy(self, difficulty: str) -> RewardPolicy:
        profile = {
            "easy": {"diagnosis": 0.20, "remediation": 0.25, "step_cost": 0.0},
            "medium": {"diagnosis": 0.15, "remediation": 0.20, "step_cost": 0.003},
            "hard": {"diagnosis": 0.12, "remediation": 0.15, "step_cost": 0.005},
        }[difficulty]

        event_values = {
            "investigation.query_logs.root": 0.08 if difficulty == "easy" else 0.05,
            "investigation.query_logs.default": 0.04 if difficulty == "easy" else 0.025,
            "investigation.check_metrics.root": 0.08 if difficulty == "easy" else 0.05,
            "investigation.check_metrics.default": 0.04
            if difficulty == "easy"
            else 0.02,
            "investigation.check_deps.default": 0.04 if difficulty == "easy" else 0.025,
            "investigation.check_config.root": 0.09 if difficulty == "easy" else 0.045,
            "investigation.check_config.default": 0.03
            if difficulty == "easy"
            else 0.02,
            "investigation.check_runbook.default": 0.04
            if difficulty == "easy"
            else 0.025,
            "diagnosis.correct": profile["diagnosis"],
            "diagnosis.wrong": 0.0,
            self._correct_remediation_event: profile["remediation"],
            "remediation.wrong": 0.0,
            "escalation.with_evidence": 0.15,
            "escalation.no_evidence": 0.0,
            "unknown_command": 0.0,
            "invalid_input": 0.0,
        }

        return RewardPolicy(
            event_values=event_values,
            time_pressure_cost_per_step=profile["step_cost"],
            destructive_penalty=-0.10,
        )

    def _reset_scenario_state(self) -> None:
        cfg = self.DIFFICULTY_CONFIG[self._difficulty]
        self.MAX_STEPS = int(cfg["max_steps"])
        self.MEMORY_CUTOFF_OVERRIDE = int(cfg["memory_cutoff"])

        self._done_investigations: set[str] = set()
        self._chosen_root_cause = self._rng.choice(self.ROOT_CAUSE_POOL)
        self._root_service = self._rng.choice(self.SERVICE_POOL)

        n_services = int(cfg["n_services"])
        n_red_herrings = int(cfg["n_red_herrings"])

        other_services = [s for s in self.SERVICE_POOL if s != self._root_service]
        sampled_services = self._rng.sample(other_services, k=max(0, n_services - 1))
        self._scenario_services = [self._root_service, *sampled_services]

        red_herrings = list(self.RED_HERRING_POOL)
        self._chosen_red_herrings = (
            self._rng.sample(red_herrings, k=min(n_red_herrings, len(red_herrings)))
            if n_red_herrings > 0
            else []
        )

        remediation_command = self.REMEDIATION_COMMAND_BY_ROOT_CAUSE[
            self._chosen_root_cause
        ]
        remediation_action, remediation_service = remediation_command.split(" ", 1)
        self._correct_remediation_action = remediation_action
        self._correct_remediation_service = remediation_service
        self._correct_remediation_event = f"remediation.{self._correct_remediation_action}.{self._correct_remediation_service}"

        self._reward_engine.register_policy(
            self.NAME, self._build_procedural_policy(self._difficulty)
        )

        self.INITIAL_SYSTEM_STATUS = {
            service: ("critical" if service == self._root_service else "degraded")
            for service in self._scenario_services
        }
        self.INITIAL_AFFECTED_SERVICES = list(self._scenario_services)
        self._current_system_status = dict(self.INITIAL_SYSTEM_STATUS)

        self._logs = {}
        self._configs = {}
        self._deps = {}
        self._metrics = {}
        for service in self._scenario_services:
            is_root = service == self._root_service
            root_tag = self._chosen_root_cause.replace("_", "-")
            self._logs[service] = (
                f"19:42:11 [ERROR] {service}: incident markers correlated with {root_tag}\n"
                if is_root
                else f"19:42:11 [WARN] {service}: downstream retries observed\n"
            )
            self._configs[service] = (
                f"Recent config for {service}: root-cause candidate {self._chosen_root_cause}\n"
                if is_root
                else f"Recent config for {service}: no direct regression found\n"
            )
            self._deps[service] = (
                f"{service} dependencies: root path includes {self._root_service}\n"
            )
            self._metrics[(service, "error_rate")] = (
                "error_rate: 19.5% (root service anomaly)\n"
                if is_root
                else "error_rate: 4.2% (secondary impact)\n"
            )
            self._metrics[(service, "latency_p95")] = (
                "latency_p95: 4200ms\n" if is_root else "latency_p95: 850ms\n"
            )

        self.ALERT_SUMMARY = (
            "## PROCEDURAL INCIDENT\n\n"
            f"Difficulty: {self._difficulty}\n"
            f"Seed: {self._seed}\n"
            f"Affected services: {', '.join(self._scenario_services)}\n"
            f"Noise signals: {', '.join(self._chosen_red_herrings) or 'none'}\n\n"
            "Investigate, diagnose, and remediate the incident."
        )

    def get_initial_observation_text(self) -> str:
        return ""

    def step(self, command: ParsedCommand) -> StepOutcome:
        action = command.action_type
        if action == "query_logs":
            return self._handle_query_logs(command)
        if action == "check_metrics":
            return self._handle_check_metrics(command)
        if action == "check_deps":
            return self._handle_check_deps(command)
        if action == "check_config":
            return self._handle_check_config(command)
        if action == "check_runbook":
            return self._handle_check_runbook(command)
        if action == "diagnose":
            return self._handle_diagnose(command)
        if action in (
            "restart_service",
            "rollback_deploy",
            "scale_resource",
            "kill_query",
        ):
            return self._handle_remediation(command)
        if action == "escalate":
            return self._handle_escalate(command)
        return self._handle_unknown_command(command.raw)

    def _handle_query_logs(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default=self._root_service)
        logs = self._logs.get(service)
        if logs is None:
            return self._invalid_input(f"No log data for service '{service}'.")

        duplicate = self._is_duplicate(f"logs:{service}")
        event = (
            "investigation.query_logs.root"
            if service == self._root_service
            else "investigation.query_logs.default"
        )
        score = self._score_event(event, duplicate=duplicate)
        return self._outcome(logs, score.reward)

    def _handle_check_metrics(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default=self._root_service)
        metric = get_metric_param(command.params, default="error_rate")
        metric_text = self._metrics.get((service, metric))
        if metric_text is None:
            return self._invalid_input(f"No metric '{metric}' for '{service}'.")

        duplicate = self._is_duplicate(f"metric:{service}:{metric}")
        event = (
            "investigation.check_metrics.root"
            if service == self._root_service
            else "investigation.check_metrics.default"
        )
        score = self._score_event(event, duplicate=duplicate)
        return self._outcome(metric_text, score.reward)

    def _handle_check_deps(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default=self._root_service)
        deps = self._deps.get(service)
        if deps is None:
            return self._invalid_input(f"No dependency data for service '{service}'.")

        duplicate = self._is_duplicate(f"deps:{service}")
        score = self._score_event(
            "investigation.check_deps.default", duplicate=duplicate
        )
        return self._outcome(deps, score.reward)

    def _handle_check_config(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default=self._root_service)
        config = self._configs.get(service)
        if config is None:
            return self._invalid_input(f"No config data for service '{service}'.")

        duplicate = self._is_duplicate(f"config:{service}")
        event = (
            "investigation.check_config.root"
            if service == self._root_service
            else "investigation.check_config.default"
        )
        score = self._score_event(event, duplicate=duplicate)
        return self._outcome(config, score.reward)

    def _handle_check_runbook(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default=self._root_service)
        runbook = (
            f"RUNBOOK {service}:\n"
            "1. Correlate logs and metrics.\n"
            "2. Validate config and dependencies.\n"
            "3. Diagnose before remediation.\n"
        )
        duplicate = self._is_duplicate(f"runbook:{service}")
        score = self._score_event(
            "investigation.check_runbook.default",
            duplicate=duplicate,
        )
        return self._outcome(runbook, score.reward)

    def _handle_diagnose(self, command: ParsedCommand) -> StepOutcome:
        raw = command.params.get("root_cause", "").strip().lower()
        normalized = raw.replace("-", "_").replace(" ", "_")
        if normalized == self._chosen_root_cause:
            self._root_cause_identified = True
            score = self._score_event("diagnosis.correct")
            return self._outcome(
                (
                    "Correct diagnosis.\n\n"
                    f"Root cause confirmed: {self._chosen_root_cause}\n"
                    f"Recommended remediation: "
                    f"{self._correct_remediation_action} service={self._correct_remediation_service}"
                ),
                score.reward,
                done=self.is_done(),
                root_cause_identified=True,
            )

        score = self._score_event("diagnosis.wrong", premature=True)
        return self._outcome(
            f"Incorrect diagnosis: '{raw}'. Continue investigating.",
            score.reward,
            done=self.is_done(),
            root_cause_identified=False,
        )

    def _handle_remediation(self, command: ParsedCommand) -> StepOutcome:
        action = command.action_type
        service = get_service_param(command.params, default=self._root_service)

        if not self._root_cause_identified:
            # Explicit evidence gate (ADR-13): no positive remediation score before diagnosis.
            return self._outcome(
                (
                    "Remediation blocked: diagnose the root cause before "
                    "applying remediation."
                ),
                0.0,
                done=self.is_done(),
                incident_resolved=False,
                root_cause_identified=False,
            )

        if (
            action == self._correct_remediation_action
            and service == self._correct_remediation_service
        ):
            self._incident_resolved = True
            self._current_system_status = {
                service_name: "healthy" for service_name in self._current_system_status
            }
            score = self._score_event(self._correct_remediation_event, resolved=True)
            return self._outcome(
                "Remediation succeeded. Incident resolved.",
                score.reward,
                done=True,
                incident_resolved=True,
                root_cause_identified=True,
            )

        score = self._score_event("remediation.wrong", destructive=True)
        return self._outcome(
            f"Action '{action}' on '{service}' did not resolve the incident.",
            score.reward,
            done=self.is_done(),
            incident_resolved=False,
            root_cause_identified=True,
        )

    def _handle_escalate(self, command: ParsedCommand) -> StepOutcome:
        investigations = len(self._done_investigations)
        reason = command.params.get("reason", "")
        if investigations >= 3:
            self._incident_resolved = True
            score = self._score_event("escalation.with_evidence", resolved=True)
            return self._outcome(
                (
                    "Escalation accepted with evidence.\n"
                    f"Investigations: {investigations}\n"
                    f"Reason: {reason}"
                ),
                score.reward,
                done=True,
                incident_resolved=True,
                root_cause_identified=self._root_cause_identified,
            )

        score = self._score_event("escalation.no_evidence", premature=True)
        return self._outcome(
            (
                "Escalation filed without sufficient evidence. "
                f"Investigations completed: {investigations}."
            ),
            score.reward,
            done=True,
            incident_resolved=False,
            root_cause_identified=self._root_cause_identified,
        )

    def _is_duplicate(self, key: str) -> bool:
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)
        return duplicate

    def _invalid_input(self, text: str) -> StepOutcome:
        score = self._score_event("invalid_input")
        return self._outcome(text, score.reward)

    def _outcome(
        self,
        investigation_result: str,
        reward: float,
        *,
        done: bool | None = None,
        incident_resolved: bool | None = None,
        root_cause_identified: bool | None = None,
    ) -> StepOutcome:
        return StepOutcome(
            investigation_result=investigation_result,
            reward=reward,
            done=self.is_done() if done is None else done,
            incident_resolved=(
                self._incident_resolved
                if incident_resolved is None
                else incident_resolved
            ),
            root_cause_identified=(
                self._root_cause_identified
                if root_cause_identified is None
                else root_cause_identified
            ),
        )
