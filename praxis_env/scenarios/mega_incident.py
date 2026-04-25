"""
praxis_env.scenarios.mega_incident — Task 5: Mega incident (120-step horizon).

This scenario is intentionally long-horizon and noisy:
  - 8 affected services
  - 3 simultaneous root causes
  - sparse, deterministic evidence
  - memory cutoff override at step 30
"""

from __future__ import annotations

from praxis_env.scenarios.base import (
    BaseScenario,
    ParsedCommand,
    StepOutcome,
    get_metric_param,
    get_service_param,
)


class MegaIncidentScenario(BaseScenario):
    """Task 5: Cascading platform failure with three concurrent root causes."""

    NAME = "cascading-platform-failure"
    SEVERITY = "P1"
    MAX_STEPS = 120
    MEMORY_CUTOFF_OVERRIDE = 30

    ALERT_SUMMARY = """\
## MEGA INCIDENT: CASCADING PLATFORM FAILURE

**Alert ID**: MEGA-007
**Severity**: P1 -- broad customer impact
**Triggered**: 02:05 UTC
**Scope**: 8 core services unhealthy

**Active Alerts**:
- api:      HTTP 5xx error rate 31%
- auth:     login timeout rate 26%
- database: connection corruption and pool lockups
- cache:    stale object surge and eviction storms
- worker:   crash-loop and memory saturation
- queue:    backlog growth > 240k jobs
- cdn:      TLS handshake failures across edge POPs
- dns:      elevated retries and timeout amplification

Investigate all plausible causes and drive resolution. Evidence quality matters.
"""

    INITIAL_SYSTEM_STATUS = {
        "api": "critical",
        "auth": "critical",
        "database": "critical",
        "cache": "degraded",
        "worker": "critical",
        "queue": "critical",
        "cdn": "critical",
        "dns": "degraded",
    }

    INITIAL_AFFECTED_SERVICES = [
        "api",
        "auth",
        "database",
        "cache",
        "worker",
        "queue",
        "cdn",
        "dns",
    ]

    _LOGS = {
        "api": """\
02:03:10 [ERROR] upstream auth timeout after 5.0s
02:03:12 [ERROR] edge request failed: TLS handshake error at cdn-edge-17
02:03:15 [ERROR] DB session init failed: pool state corrupted
02:03:17 [WARN] fallback retries exhausted for 29% of requests
""",
        "auth": """\
02:03:11 [ERROR] token validation timeout waiting on database
02:03:14 [ERROR] callback to api failed: edge TLS failure
02:03:18 [WARN] queue dispatch delayed > 11s
""",
        "database": """\
02:01:40 [WARN] pool consistency check failed: checksum mismatch
02:02:10 [ERROR] allocator flagged corrupted connection slot block
02:02:45 [ERROR] lock manager stalled while recycling pool entries
02:03:05 [ERROR] client checkout failures rising: 0 -> 412/min
""",
        "cache": """\
02:02:20 [WARN] cache misses climbing due to API retries
02:02:45 [WARN] stale object invalidation lag increased to 48s
02:03:10 [WARN] evictions elevated from queue replay traffic
""",
        "worker": """\
02:01:55 [WARN] memory usage crossed 92%
02:02:15 [WARN] GC pause 3.4s
02:02:30 [ERROR] process terminated: OOMKilled
02:02:31 [INFO] pod restart initiated
02:02:55 [WARN] memory climbed to 95% immediately after restart
""",
        "queue": """\
02:02:00 [WARN] pending jobs: 120k
02:02:30 [WARN] pending jobs: 188k
02:03:00 [ERROR] pending jobs: 243k (critical threshold breached)
""",
        "cdn": """\
02:01:10 [WARN] cert expiry monitor: edge cert expires in 0 minutes
02:01:15 [ERROR] TLS handshake failure: certificate expired
02:02:00 [ERROR] edge POP eu-west-3 handshake failure ratio 41%
02:02:40 [ERROR] origin fetch aborted: invalid certificate chain
""",
        "dns": """\
02:02:05 [WARN] resolver retries elevated due to upstream timeouts
02:02:35 [WARN] p95 lookup latency increased from 12ms to 64ms
02:03:05 [WARN] NXDOMAIN stable at baseline (not primary fault)
""",
    }

    _METRICS = {
        ("database", "connections"): """\
connections (database)
  pool slots: 100/100 allocated
  corrupted slots: 37
  successful checkout rate: 21%
  failed checkout rate: 79%
""",
        ("database", "integrity"): """\
integrity (database pool manager)
  pool_checksum: INVALID
  allocator_consistency: FAILED
  lock_manager_queue: 84 blocked workers
""",
        ("worker", "memory"): """\
memory (worker)
  current: 97%
  restart count (1h): 23
  pattern: linear growth to OOM within ~45s
""",
        ("worker", "cpu"): """\
cpu (worker)
  current: 88%
  spikes to 100% during GC pauses
""",
        ("cdn", "tls_handshake_failures"): """\
tls_handshake_failures (cdn)
  current: 38%
  1h avg: 0.2%
  cert_not_after: 2026-04-25T02:01:00Z
""",
        ("cdn", "error_rate"): """\
error_rate (cdn)
  current: 34%
  1h avg: 1.1%
  correlated with TLS handshake failures across regions
""",
        ("queue", "backlog"): """\
backlog (queue)
  current: 243k jobs
  growth: +6.2k jobs/min
""",
        ("api", "error_rate"): """\
error_rate (api)
  current: 31%
  dominant failure signatures: auth timeout, cdn TLS, db checkout failure
""",
    }

    _DEPS = {
        "api": "api -> auth, database, cache, cdn",
        "auth": "auth -> database, queue",
        "database": "database -> none (shared dependency root)",
        "cache": "cache -> database (write-through), dns",
        "worker": "worker -> queue, database, cache",
        "queue": "queue -> database, cache",
        "cdn": "cdn -> dns, origin-api",
        "dns": "dns -> none (infrastructure resolver)",
    }

    _CONFIGS = {
        "database": """\
Recent database changes:
  - Pool maintenance patch applied at 01:55 UTC
  - allocator.compaction=true (new)
  - pool_integrity_guard=disabled (temporary hotfix)
""",
        "worker": """\
Recent worker changes:
  - Deploy v5.9.0 at 01:40 UTC
  - BATCH_SIZE: 200 -> 5000
  - HEAP_LIMIT_MB unchanged at 2048
""",
        "cdn": """\
Recent CDN changes:
  - edge cert rotation job FAILED at 01:50 UTC
  - fallback certificate chain not promoted
  - renewal automation retries exhausted
""",
        "api": "No direct config changes in the last 24h.",
        "auth": "No direct config changes in the last 24h.",
        "cache": "No direct config changes in the last 24h.",
        "queue": "No direct config changes in the last 24h.",
        "dns": "No direct config changes in the last 24h.",
    }

    _RUNBOOKS = {
        "database": """\
RUNBOOK database (SRE-DB-091)
If pool integrity is invalid, remediate by restoring pool guard and scaling pool.
Avoid service restarts during corruption until diagnosis is confirmed.
""",
        "worker": """\
RUNBOOK worker (SRE-WORK-033)
Memory crash loops after batch-size changes usually require rollback_deploy.
""",
        "cdn": """\
RUNBOOK cdn (SRE-CDN-044)
TLS handshake failures with expired certs require cert refresh or rollback_deploy.
""",
        "api": "RUNBOOK api: verify upstream dependencies first.",
    }

    ROOT_CAUSE_ALIASES = {
        "db_pool_corrupted": "db_pool_corrupted",
        "database_pool_corrupted": "db_pool_corrupted",
        "pool_corrupted": "db_pool_corrupted",
        "cdn_tls_expired": "cdn_tls_expired",
        "tls_expired": "cdn_tls_expired",
        "expired_certificate": "cdn_tls_expired",
        "worker_memory_leak": "worker_memory_leak",
        "memory_leak_worker": "worker_memory_leak",
        "worker_oom": "worker_memory_leak",
    }

    RED_HERRING_CAUSES = frozenset(
        {
            "dns_outage",
            "cache_failure",
            "api_deploy",
            "auth_regression",
            "queue_bug",
            "network_partition",
        }
    )

    _REMEDIATION_EVENTS = {
        "db_pool_corrupted": "remediation.scale_resource.database.connection_pool",
        "cdn_tls_expired": "remediation.rollback_deploy.cdn",
        "worker_memory_leak": "remediation.rollback_deploy.worker",
    }

    def _reset_scenario_state(self) -> None:
        self._done_investigations: set[str] = set()
        self._diagnosed_root_causes: set[str] = set()
        self._applied_remediations: set[str] = set()

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
        if action == "scale_resource":
            return self._handle_scale_resource(command)
        if action == "rollback_deploy":
            return self._handle_rollback_deploy(command)
        if action in {"restart_service", "kill_query"}:
            return self._handle_wrong_remediation(command)
        if action == "escalate":
            return self._handle_escalate(command)
        return self._handle_unknown_command(command.raw)

    def get_initial_observation_text(self) -> str:
        return ""

    def _score_remediation_event(
        self,
        event: str,
        *,
        duplicate: bool = False,
        destructive: bool = False,
        resolved: bool = False,
    ) -> float:
        """
        Score remediation events with explicit diagnosis gating for this scenario.
        """
        result = self._reward_engine.score(
            task_name=self.NAME,
            event=event,
            duplicate=duplicate,
            destructive=destructive,
            resolved=resolved,
            root_cause_identified=bool(self._diagnosed_root_causes),
            step_number=self._step_count + 1,
            max_steps=self.MAX_STEPS,
        )
        return result.reward

    def _handle_query_logs(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="api")
        data = self._LOGS.get(service)
        if data is None:
            score = self._score_event("invalid_input")
            return StepOutcome(
                investigation_result=f"No log data for service '{service}'.",
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        key = f"logs:{service}"
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)

        if service in {"database", "cdn", "worker"}:
            event = f"investigation.query_logs.{service}"
        else:
            event = "investigation.query_logs.default"
        score = self._score_event(event, duplicate=duplicate)

        return StepOutcome(
            investigation_result=data,
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_check_metrics(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="database")
        metric = get_metric_param(command.params, default="connections")
        data = self._METRICS.get((service, metric))
        if data is None:
            score = self._score_event("invalid_input")
            return StepOutcome(
                investigation_result=f"No metric '{metric}' for service '{service}'.",
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        key = f"metric:{service}:{metric}"
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)

        if service == "database" and metric == "connections":
            event = "investigation.check_metrics.database.connections"
        elif service == "cdn" and metric == "tls_handshake_failures":
            event = "investigation.check_metrics.cdn.tls_handshake_failures"
        elif service == "worker" and metric == "memory":
            event = "investigation.check_metrics.worker.memory"
        else:
            event = "investigation.check_metrics.default"
        score = self._score_event(event, duplicate=duplicate)

        return StepOutcome(
            investigation_result=data,
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_check_deps(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="api")
        data = self._DEPS.get(service)
        if data is None:
            score = self._score_event("invalid_input")
            return StepOutcome(
                investigation_result=f"No dependency map for service '{service}'.",
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        key = f"deps:{service}"
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)

        score = self._score_event(
            "investigation.check_deps.default", duplicate=duplicate
        )
        return StepOutcome(
            investigation_result=data,
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_check_config(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="database")
        data = self._CONFIGS.get(service)
        if data is None:
            score = self._score_event("invalid_input")
            return StepOutcome(
                investigation_result=f"No config history for service '{service}'.",
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        key = f"config:{service}"
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)

        if service in {"database", "cdn", "worker"}:
            event = f"investigation.check_config.{service}"
        else:
            event = "investigation.check_config.default"
        score = self._score_event(event, duplicate=duplicate)

        return StepOutcome(
            investigation_result=data,
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_check_runbook(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="database")
        data = self._RUNBOOKS.get(service)
        if data is None:
            score = self._score_event("invalid_input")
            return StepOutcome(
                investigation_result=f"No runbook available for service '{service}'.",
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        key = f"runbook:{service}"
        duplicate = key in self._done_investigations
        if not duplicate:
            self._done_investigations.add(key)

        score = self._score_event(
            "investigation.check_runbook.default",
            duplicate=duplicate,
        )
        return StepOutcome(
            investigation_result=data,
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=self._incident_resolved,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_diagnose(self, command: ParsedCommand) -> StepOutcome:
        raw = command.params.get("root_cause", "").lower().strip()
        normalized = raw.replace("-", "_").replace(" ", "_")
        canonical = self.ROOT_CAUSE_ALIASES.get(normalized)

        if canonical is not None:
            duplicate = canonical in self._diagnosed_root_causes
            if not duplicate:
                self._diagnosed_root_causes.add(canonical)
            self._root_cause_identified = bool(self._diagnosed_root_causes)
            score = self._score_event("diagnosis.correct", duplicate=duplicate)

            return StepOutcome(
                investigation_result=(
                    f"Diagnosis accepted: {canonical}.\n"
                    f"Diagnosed root causes: {len(self._diagnosed_root_causes)}/3."
                ),
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        if normalized in self.RED_HERRING_CAUSES:
            score = self._score_event("diagnosis.wrong", premature=True)
            return StepOutcome(
                investigation_result=(
                    f"Incorrect diagnosis '{raw}'. This is a red herring in this incident."
                ),
                reward=score.reward,
                done=self.is_done(),
                incident_resolved=False,
                root_cause_identified=self._root_cause_identified,
            )

        score = self._score_event("diagnosis.wrong", premature=True)
        return StepOutcome(
            investigation_result=f"Incorrect diagnosis: '{raw}'.",
            reward=score.reward,
            done=self.is_done(),
            incident_resolved=False,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_scale_resource(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params, default="database")
        resource = command.params.get("resource", "").lower().strip()

        if service == "database" and resource in {
            "connection_pool",
            "connections",
            "pool",
            "db_connections",
        }:
            return self._apply_correct_remediation("db_pool_corrupted")

        return self._handle_wrong_remediation(command)

    def _handle_rollback_deploy(self, command: ParsedCommand) -> StepOutcome:
        service = get_service_param(command.params)
        if service == "cdn":
            return self._apply_correct_remediation("cdn_tls_expired")
        if service == "worker":
            return self._apply_correct_remediation("worker_memory_leak")
        return self._handle_wrong_remediation(command)

    def _apply_correct_remediation(self, cause: str) -> StepOutcome:
        duplicate = cause in self._applied_remediations
        if not duplicate:
            self._applied_remediations.add(cause)

        resolved_now = self._all_causes_and_remediations_complete()
        if resolved_now:
            self._incident_resolved = True
            self._current_system_status = {
                service: "healthy" for service in self._current_system_status
            }
        else:
            self._apply_partial_recovery(cause)

        event = self._REMEDIATION_EVENTS[cause]
        reward = self._score_remediation_event(
            event,
            duplicate=duplicate,
            resolved=resolved_now,
        )

        return StepOutcome(
            investigation_result=(
                f"Remediation applied for {cause}.\n"
                f"Remediations completed: {len(self._applied_remediations)}/3.\n"
                f"Diagnoses completed: {len(self._diagnosed_root_causes)}/3."
            ),
            reward=reward,
            done=resolved_now,
            incident_resolved=resolved_now,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_wrong_remediation(self, command: ParsedCommand) -> StepOutcome:
        reward = self._score_remediation_event("remediation.wrong", destructive=True)

        if not self._diagnosed_root_causes:
            message = (
                "Remediation blocked: diagnose at least one root cause before applying "
                "remediation actions."
            )
        else:
            message = f"Action '{command.action_type}' did not remediate any active root cause."

        return StepOutcome(
            investigation_result=message,
            reward=reward,
            done=self.is_done(),
            incident_resolved=False,
            root_cause_identified=self._root_cause_identified,
        )

    def _handle_escalate(self, command: ParsedCommand) -> StepOutcome:
        reason = command.params.get("reason", "")
        investigations = len(self._done_investigations)
        has_diagnosis = bool(self._diagnosed_root_causes)

        if has_diagnosis and investigations >= 6:
            self._incident_resolved = True
            score = self._score_event("escalation.with_evidence", resolved=True)
            return StepOutcome(
                investigation_result=(
                    "Escalation accepted with evidence package.\n"
                    f"Reason: {reason}\n"
                    f"Diagnosed root causes: {len(self._diagnosed_root_causes)}\n"
                    f"Unique investigations: {investigations}"
                ),
                reward=score.reward,
                done=True,
                incident_resolved=True,
                root_cause_identified=self._root_cause_identified,
                info={"resolution": "escalated_with_evidence"},
            )

        score = self._score_event("escalation.no_evidence", premature=True)
        return StepOutcome(
            investigation_result=(
                "Escalation rejected: insufficient evidence for handoff.\n"
                f"Diagnoses: {len(self._diagnosed_root_causes)} (need >=1)\n"
                f"Unique investigations: {investigations} (need >=6)"
            ),
            reward=score.reward,
            done=True,
            incident_resolved=False,
            root_cause_identified=self._root_cause_identified,
        )

    def _all_causes_and_remediations_complete(self) -> bool:
        return (
            len(self._diagnosed_root_causes) == 3
            and len(self._applied_remediations) == 3
        )

    def _apply_partial_recovery(self, cause: str) -> None:
        if cause == "db_pool_corrupted":
            self._current_system_status["database"] = "degraded"
            self._current_system_status["api"] = "degraded"
            self._current_system_status["auth"] = "degraded"
        elif cause == "cdn_tls_expired":
            self._current_system_status["cdn"] = "degraded"
            self._current_system_status["api"] = "degraded"
        elif cause == "worker_memory_leak":
            self._current_system_status["worker"] = "degraded"
            self._current_system_status["queue"] = "degraded"
