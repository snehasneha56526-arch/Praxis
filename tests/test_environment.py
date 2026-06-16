"""
tests/test_environment.py — Phase 2: environment lifecycle tests.

Tests the PraxisEnvironment's reset/step/state contract independent
of any specific scenario details, plus shared observation contract
regressions that apply across registered scenarios.

Full round-trip tests (with scenarios) land in tests/test_scenarios.py
in Phases 3-5.
"""

import pytest
from server.praxis_environment import PraxisEnvironment
from praxis_env.models import PraxisAction


class TestPraxisEnvironmentInit:
    def test_instantiates(self):
        env = PraxisEnvironment()
        assert env is not None

    def test_list_tasks_returns_registered_scenarios(self):
        env = PraxisEnvironment()
        tasks = env.list_tasks()
        assert isinstance(tasks, list)
        assert "single-service-alert" in tasks
        assert "cascading-failure" in tasks
        assert "ambiguous-incident" in tasks

    def test_step_before_reset_raises_runtime_error(self):
        env = PraxisEnvironment()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(PraxisAction(command="query_logs service=auth timerange=5m"))

    def test_state_before_reset_raises_runtime_error(self):
        env = PraxisEnvironment()
        with pytest.raises(RuntimeError, match="reset"):
            env.state()


class TestResetErrors:
    def test_reset_unknown_task_raises_value_error(self):
        env = PraxisEnvironment()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset(task_name="no-such-task")

    def test_reset_error_message_includes_task_name(self):
        env = PraxisEnvironment()
        with pytest.raises(ValueError) as exc_info:
            env.reset(task_name="imaginary-scenario")
        assert "imaginary-scenario" in str(exc_info.value)


class TestTaskAliases:
    def test_reset_accepts_easy_alias(self):
        env = PraxisEnvironment()
        obs = env.reset(task_name="easy")
        assert obs.severity == "P2"
        assert env.state().task_name == "single-service-alert"

    def test_reset_accepts_medium_alias(self):
        env = PraxisEnvironment()
        obs = env.reset(task_name="medium")
        assert obs.severity == "P2"
        assert env.state().task_name == "ambiguous-incident"

    def test_reset_accepts_hard_alias(self):
        env = PraxisEnvironment()
        obs = env.reset(task_name="hard")
        assert obs.severity == "P1"
        assert env.state().task_name == "cascading-failure"


class TestCommandParserIntegration:
    """
    Test that parse_command integrates correctly with step().
    Uses a minimal stub scenario that records what command it got.
    """

    def test_parse_command_standalone(self):
        """Direct smoke test of parse_command from the server module."""
        from server.command_parser import parse_command

        cmd = parse_command("query_logs service=auth timerange=5m")
        assert cmd.action_type == "query_logs"
        assert cmd.params == {"service": "auth", "timerange": "5m"}

    def test_parse_empty_command(self):
        from server.command_parser import parse_command

        cmd = parse_command("")
        assert cmd.action_type == ""
        assert cmd.params == {}

    def test_parse_escalate_freetext(self):
        from server.command_parser import parse_command

        cmd = parse_command("escalate reason=everything is on fire please help")
        assert cmd.action_type == "escalate"
        assert cmd.params["reason"] == "everything is on fire please help"


class TestObsToDict:
    """Test the observation serialisation helper."""

    def test_obs_to_dict_has_all_keys(self):
        from server.praxis_environment import PraxisEnvironment
        from praxis_env.models import PraxisObservation

        env = PraxisEnvironment()
        obs = PraxisObservation(
            alert_summary="test alert",
            system_status={"auth": "critical"},
            investigation_result="some logs",
            available_commands=["query_logs"],
            time_elapsed_minutes=5.0,
            severity="P2",
            services_affected=["auth"],
            step_number=1,
        )
        d = env._obs_to_dict(obs)
        required_keys = {
            "alert_summary",
            "system_status",
            "investigation_result",
            "available_commands",
            "time_elapsed_minutes",
            "severity",
            "services_affected",
            "step_number",
            "memory_active",
            "saved_findings_count",
        }
        assert required_keys == set(d.keys())

    def test_obs_to_dict_values_are_json_safe(self):
        import json
        from server.praxis_environment import PraxisEnvironment
        from praxis_env.models import PraxisObservation

        env = PraxisEnvironment()
        obs = PraxisObservation(
            alert_summary="test",
            system_status={"auth": "critical"},
            investigation_result="",
            available_commands=["cmd1"],
            time_elapsed_minutes=2.5,
            severity="P1",
            services_affected=["auth"],
            step_number=0,
        )
        d = env._obs_to_dict(obs)
        # This will raise if anything is not JSON serialisable
        serialised = json.dumps(d)
        assert "test" in serialised


class TestScenarioObservationContracts:
    def test_cascading_failure_reset_contract(self):
        env = PraxisEnvironment()
        obs = env.reset(task_name="cascading-failure")
        assert obs.severity == "P1"
        assert obs.step_number == 0
        assert obs.services_affected == [
            "api",
            "auth",
            "payment",
            "database",
            "notification",
            "cache",
        ]
        assert "cascading-failure" in env.list_tasks()

    def test_ambiguous_incident_reset_contract(self):
        env = PraxisEnvironment()
        obs = env.reset(task_name="ambiguous-incident")
        assert obs.severity == "P2"
        assert obs.step_number == 0
        assert obs.services_affected == [
            "frontend",
            "api",
            "auth",
            "search",
            "dns-resolver",
        ]
        assert "ambiguous-incident" in env.list_tasks()

    @pytest.mark.parametrize(
        ("task_name", "command"),
        [
            ("single-service-alert", "query_logs service=auth timerange=5m"),
            ("cascading-failure", "query_logs service=api timerange=10m"),
            ("ambiguous-incident", "query_logs service=frontend timerange=10m"),
        ],
    )
    def test_reset_and_step_payloads_are_ascii_safe(self, task_name, command):
        env = PraxisEnvironment()
        obs = env.reset(task_name=task_name)
        assert obs.alert_summary.isascii()
        assert obs.investigation_result.isascii()

        result = env.step(PraxisAction(command=command))
        observation = result["observation"]
        assert observation["alert_summary"].isascii()
        assert observation["investigation_result"].isascii()


class TestEpisodeScoreBudget:
    HIGH_REWARD_TRAJECTORIES = [
        (
            "single-service-alert",
            [
                "query_logs service=auth timerange=5m",
                "query_logs service=api timerange=15m",
                "query_logs service=database timerange=30m",
                "check_metrics service=auth metric=connections",
                "check_metrics service=database metric=connections",
                "check_deps service=auth",
                "check_deps service=api",
                "check_config service=auth",
                "check_config service=api",
                "check_runbook service=auth",
                "diagnose root_cause=bad_config",
                "rollback_deploy service=auth",
            ],
            True,
        ),
        (
            "ambiguous-incident",
            [
                "query_logs service=frontend timerange=10m",
                "query_logs service=api timerange=10m",
                "query_logs service=auth timerange=10m",
                "query_logs service=search timerange=10m",
                "query_logs service=dns-resolver timerange=30m",
                "check_metrics service=dns-resolver metric=resolution_failures",
                "check_metrics service=api metric=error_rate",
                "check_metrics service=load-balancer metric=latency_p95",
                "check_deps service=frontend",
                "check_config service=dns-resolver",
                "check_runbook service=dns-resolver",
                "diagnose root_cause=dns_misconfiguration",
                "restart_service service=dns-resolver",
            ],
            False,
        ),
        (
            "memory-leak",
            [
                "query_logs service=worker timerange=10m",
                "query_logs service=api timerange=10m",
                "check_metrics service=worker metric=memory",
                "check_metrics service=api metric=cpu",
                "check_deps service=worker",
                "check_config service=worker",
                "check_runbook service=worker",
                "diagnose root_cause=large_batch_size_oom",
                "rollback_deploy service=worker",
            ],
            False,
        ),
        (
            "cascading-failure",
            [
                "query_logs service=api timerange=10m",
                "query_logs service=database timerange=15m",
                "query_logs service=analytics timerange=15m",
                "check_metrics service=database metric=connections",
                "check_deps service=api",
                "check_config service=database",
                "check_config service=analytics",
                "check_runbook service=database",
                "diagnose root_cause=db_connection_pool_exhausted",
                "kill_query service=database query_id=runaway_analytics",
                "scale_resource service=database resource=connection_pool",
            ],
            False,
        ),
    ]

    OPTIMAL_PATHS = [
        (
            "single-service-alert",
            [
                "query_logs service=auth timerange=5m",
                "check_config service=auth",
                "diagnose root_cause=bad_config",
                "rollback_deploy service=auth",
            ],
            0.63,
        ),
        (
            "ambiguous-incident",
            [
                "query_logs service=frontend timerange=10m",
                "query_logs service=api timerange=10m",
                "query_logs service=auth timerange=10m",
                "check_deps service=frontend",
                "check_metrics service=dns-resolver metric=resolution_failures",
                "query_logs service=dns-resolver timerange=30m",
                "check_config service=dns-resolver",
                "diagnose root_cause=dns_misconfiguration",
                "restart_service service=dns-resolver",
            ],
            0.71,
        ),
        (
            "memory-leak",
            [
                "query_logs service=worker timerange=10m",
                "check_metrics service=worker metric=memory",
                "check_config service=worker",
                "diagnose root_cause=large_batch_size_oom",
                "rollback_deploy service=worker",
            ],
            0.475,
        ),
        (
            "cascading-failure",
            [
                "query_logs service=api timerange=10m",
                "check_deps service=api",
                "check_metrics service=database metric=connections",
                "query_logs service=database timerange=15m",
                "diagnose root_cause=db_connection_pool_exhausted",
                "kill_query service=database query_id=runaway_analytics",
                "scale_resource service=database resource=connection_pool",
            ],
            0.458,
        ),
    ]

    @staticmethod
    def _run_commands(task_name, commands):
        env = PraxisEnvironment()
        env.reset(task_name=task_name)
        rewards = []
        last_result = None

        for command in commands:
            last_result = env.step(PraxisAction(command=command))
            rewards.append(last_result["reward"])
            if last_result["done"]:
                break

        return env, rewards, last_result

    @pytest.mark.parametrize(
        ("task_name", "commands", "expect_score_cap"),
        HIGH_REWARD_TRAJECTORIES,
    )
    def test_high_reward_trajectories_keep_emitted_total_below_one(
        self,
        task_name,
        commands,
        expect_score_cap,
    ):
        env, rewards, last_result = self._run_commands(task_name, commands)

        assert rewards
        assert all(0.0 < reward < 1.0 for reward in rewards)
        assert sum(rewards) < 1.0

        expected_cumulative = min(0.99, 0.01 + sum(rewards))
        assert env.state().cumulative_reward == pytest.approx(expected_cumulative)

        if expect_score_cap:
            assert last_result["done"] is True
            assert last_result["info"].get("score_cap_reached") is True

    @pytest.mark.parametrize(("task_name", "commands", "expected_total"), OPTIMAL_PATHS)
    def test_optimal_paths_keep_existing_environment_totals(
        self,
        task_name,
        commands,
        expected_total,
    ):
        _, rewards, last_result = self._run_commands(task_name, commands)

        assert sum(rewards) == pytest.approx(expected_total, abs=1e-6)
        assert last_result["info"].get("score_cap_reached") is not True

    def test_step_after_score_cap_short_circuits_with_floor_reward(self):
        task_name = "single-service-alert"
        commands = [
            "query_logs service=auth timerange=5m",
            "query_logs service=api timerange=15m",
            "query_logs service=database timerange=30m",
            "check_metrics service=auth metric=connections",
            "check_metrics service=database metric=connections",
            "check_deps service=auth",
            "check_deps service=api",
            "check_config service=auth",
            "check_config service=api",
            "check_runbook service=auth",
            "diagnose root_cause=bad_config",
            "rollback_deploy service=auth",
        ]

        env, _, last_result = self._run_commands(task_name, commands)
        assert last_result["info"].get("score_cap_reached") is True
        assert env.state().cumulative_reward == pytest.approx(0.99)

        repeated = env.step(
            PraxisAction(command="query_logs service=auth timerange=5m")
        )

        assert repeated["reward"] == pytest.approx(0.01)
        assert repeated["done"] is True
        assert repeated["info"]["error"] == "episode_score_cap_reached"
        assert repeated["info"]["score_cap_reached"] is True
        assert env.state().cumulative_reward == pytest.approx(0.99)
