"""
tests/test_command_parser.py — Phase 2: command parser unit tests.

Tests every command type including edge cases (empty, unknown,
escalate free text, mixed quoting).
"""

from server.command_parser import is_known_action, parse_command
from praxis_env.scenarios.base import (
    get_metric_param,
    get_service_param,
    get_timerange_minutes,
)


class TestParseCommand:
    # ── Basic parsing ─────────────────────────────────────────────────────────

    def test_query_logs_basic(self):
        cmd = parse_command("query_logs service=auth timerange=5m")
        assert cmd.action_type == "query_logs"
        assert cmd.params["service"] == "auth"
        assert cmd.params["timerange"] == "5m"
        assert cmd.raw == "query_logs service=auth timerange=5m"

    def test_check_metrics(self):
        cmd = parse_command("check_metrics service=database metric=connections")
        assert cmd.action_type == "check_metrics"
        assert cmd.params["service"] == "database"
        assert cmd.params["metric"] == "connections"

    def test_check_deps(self):
        cmd = parse_command("check_deps service=api")
        assert cmd.action_type == "check_deps"
        assert cmd.params["service"] == "api"

    def test_check_config(self):
        cmd = parse_command("check_config service=auth")
        assert cmd.action_type == "check_config"
        assert cmd.params["service"] == "auth"

    def test_diagnose(self):
        cmd = parse_command("diagnose root_cause=db_connection_pool_exhausted")
        assert cmd.action_type == "diagnose"
        assert cmd.params["root_cause"] == "db_connection_pool_exhausted"

    def test_restart_service(self):
        cmd = parse_command("restart_service service=payment")
        assert cmd.action_type == "restart_service"
        assert cmd.params["service"] == "payment"

    def test_rollback_deploy(self):
        cmd = parse_command("rollback_deploy service=auth")
        assert cmd.action_type == "rollback_deploy"
        assert cmd.params["service"] == "auth"

    def test_scale_resource(self):
        cmd = parse_command("scale_resource service=database resource=connection_pool")
        assert cmd.action_type == "scale_resource"
        assert cmd.params["service"] == "database"
        assert cmd.params["resource"] == "connection_pool"

    def test_kill_query(self):
        cmd = parse_command("kill_query service=database query_id=abc123")
        assert cmd.action_type == "kill_query"
        assert cmd.params["service"] == "database"
        assert cmd.params["query_id"] == "abc123"

    # ── Escalate — free text after reason= ────────────────────────────────────

    def test_escalate_with_reason(self):
        cmd = parse_command("escalate reason=DNS is broken and services are down")
        assert cmd.action_type == "escalate"
        assert cmd.params["reason"] == "DNS is broken and services are down"

    def test_escalate_reason_with_other_params_before(self):
        """reason= captures everything after it, even if there were earlier k=v tokens."""
        cmd = parse_command("escalate reason=network partition affecting 3 zones")
        assert "reason" in cmd.params
        assert "network partition" in cmd.params["reason"]

    def test_escalate_no_reason_key(self):
        """If no reason= key, whole remainder becomes reason value."""
        cmd = parse_command("escalate something went very wrong")
        assert cmd.action_type == "escalate"
        assert cmd.params.get("reason") == "something went very wrong"

    # ── Memory actions ────────────────────────────────────────────────────────

    def test_save_finding_free_text_value(self):
        cmd = parse_command("save_finding key=db_pool value=exhausted at step 4")
        assert cmd.action_type == "save_finding"
        assert cmd.params == {"key": "db_pool", "value": "exhausted at step 4"}

    def test_save_finding_bogus_without_value(self):
        cmd = parse_command("save_finding key=db_pool exhausted at step 4")
        assert cmd.action_type == "save_finding"
        assert cmd.params == {}

    def test_recall_memory_no_args(self):
        cmd = parse_command("recall_memory")
        assert cmd.action_type == "recall_memory"
        assert cmd.params == {}

    def test_recall_memory_with_key(self):
        cmd = parse_command("recall_memory key=db_pool")
        assert cmd.action_type == "recall_memory"
        assert cmd.params == {"key": "db_pool"}

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_string(self):
        cmd = parse_command("")
        assert cmd.action_type == ""
        assert cmd.params == {}

    def test_whitespace_only(self):
        cmd = parse_command("   ")
        assert cmd.action_type == ""
        assert cmd.params == {}

    def test_action_only_no_params(self):
        cmd = parse_command("check_deps")
        assert cmd.action_type == "check_deps"
        assert cmd.params == {}

    def test_unknown_action(self):
        cmd = parse_command("explode everything now")
        assert cmd.action_type == "explode"
        # remaining tokens without = are ignored
        assert cmd.params == {}

    def test_action_type_lowercased(self):
        cmd = parse_command("QUERY_LOGS service=auth timerange=5m")
        assert cmd.action_type == "query_logs"

    def test_quoted_values_stripped(self):
        cmd = parse_command("query_logs service='auth' timerange=\"5m\"")
        assert cmd.params["service"] == "auth"
        assert cmd.params["timerange"] == "5m"

    def test_extra_whitespace_between_tokens(self):
        cmd = parse_command("check_metrics  service=auth  metric=error_rate")
        assert cmd.action_type == "check_metrics"
        assert cmd.params["service"] == "auth"
        assert cmd.params["metric"] == "error_rate"

    def test_raw_preserved(self):
        raw = "query_logs service=auth timerange=5m"
        cmd = parse_command(raw)
        assert cmd.raw == raw


class TestIsKnownAction:
    def test_known_actions(self):
        known = [
            "query_logs",
            "check_metrics",
            "check_deps",
            "check_config",
            "diagnose",
            "restart_service",
            "rollback_deploy",
            "scale_resource",
            "kill_query",
            "escalate",
            "save_finding",
            "recall_memory",
        ]
        for action in known:
            assert is_known_action(action), f"{action} should be known"

    def test_unknown_action(self):
        assert not is_known_action("explode")
        assert not is_known_action("")
        assert not is_known_action("reboot")

    def test_case_insensitive(self):
        assert is_known_action("QUERY_LOGS")
        assert is_known_action("Diagnose")


class TestParamHelpers:
    def test_get_service_param(self):
        params = {"service": "Auth"}
        assert get_service_param(params) == "auth"

    def test_get_service_param_missing(self):
        assert get_service_param({}) == ""
        assert get_service_param({}, default="unknown") == "unknown"

    def test_get_metric_param(self):
        params = {"metric": "ErrorRate"}
        assert get_metric_param(params) == "errorrate"

    def test_get_timerange_minutes_basic(self):
        assert get_timerange_minutes({"timerange": "5m"}) == 5
        assert get_timerange_minutes({"timerange": "15m"}) == 15
        assert get_timerange_minutes({"timerange": "30m"}) == 30

    def test_get_timerange_minutes_missing(self):
        assert get_timerange_minutes({}) == 5  # default
        assert get_timerange_minutes({}, default=10) == 10

    def test_get_timerange_minutes_bad_value(self):
        assert get_timerange_minutes({"timerange": "bad"}) == 5  # fallback to default
