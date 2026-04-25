"""
tests/test_scenarios.py - Phase 6 cross-scenario quality gates.

Checks shared behavior across all tasks:
  - deterministic rewards for fixed command paths
  - better paths score higher than naive paths
  - environment respects reward bounds and max-step termination
"""

import pytest

from praxis_env.models import PraxisAction
from praxis_env.scenarios.ambiguous_incident import AmbiguousIncidentScenario
from praxis_env.scenarios.cascading_failure import CascadingFailureScenario
from praxis_env.scenarios.mega_incident import MegaIncidentScenario
from praxis_env.scenarios.memory_leak_scenario import MemoryLeakScenario
from praxis_env.scenarios.single_service_alert import SingleServiceAlertScenario
from server.command_parser import parse_command
from server.praxis_environment import PraxisEnvironment


def run_scenario_path(scenario_cls, commands: list[str]) -> list[float]:
    scenario = scenario_cls()
    scenario.reset(episode_id="phase6-test")
    rewards: list[float] = []
    for cmd in commands:
        outcome = scenario.step(parse_command(cmd))
        rewards.append(outcome.reward)
        if outcome.done:
            break
    return rewards


def run_environment_path(task_name: str, commands: list[str]) -> list[float]:
    env = PraxisEnvironment()
    env.reset(task_name=task_name)
    rewards: list[float] = []
    for cmd in commands:
        result = env.step(PraxisAction(command=cmd))
        rewards.append(result["reward"])
        if result["done"]:
            break
    return rewards


DETERMINISTIC_CASES = [
    (
        SingleServiceAlertScenario,
        [
            "query_logs service=auth timerange=5m",
            "check_config service=auth",
            "diagnose root_cause=bad_config",
            "rollback_deploy service=auth",
        ],
    ),
    (
        CascadingFailureScenario,
        [
            "query_logs service=api timerange=10m",
            "check_deps service=api",
            "check_metrics service=database metric=connections",
            "query_logs service=database timerange=15m",
            "diagnose root_cause=db_connection_pool_exhausted",
            "kill_query service=database query_id=runaway_analytics",
            "scale_resource service=database resource=connection_pool",
        ],
    ),
    (
        AmbiguousIncidentScenario,
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
    ),
    (
        MemoryLeakScenario,
        [
            "query_logs service=worker timerange=10m",
            "check_metrics service=worker metric=memory",
            "check_config service=worker",
            "diagnose root_cause=large_batch_size_oom",
            "rollback_deploy service=worker",
        ],
    ),
    (
        MegaIncidentScenario,
        [
            "query_logs service=database timerange=15m",
            "check_metrics service=database metric=connections",
            "query_logs service=cdn timerange=15m",
            "check_metrics service=cdn metric=tls_handshake_failures",
            "query_logs service=worker timerange=15m",
            "check_metrics service=worker metric=memory",
            "diagnose root_cause=db_pool_corrupted",
            "diagnose root_cause=cdn_tls_expired",
            "diagnose root_cause=worker_memory_leak",
            "scale_resource service=database resource=connection_pool",
            "rollback_deploy service=cdn",
            "rollback_deploy service=worker",
        ],
    ),
]


@pytest.mark.parametrize(("scenario_cls", "commands"), DETERMINISTIC_CASES)
def test_rewards_are_deterministic_for_all_scenarios(scenario_cls, commands):
    run_1 = run_scenario_path(scenario_cls, commands)
    run_2 = run_scenario_path(scenario_cls, commands)
    run_3 = run_scenario_path(scenario_cls, commands)
    assert run_1 == run_2 == run_3


QUALITY_CASES = [
    (
        SingleServiceAlertScenario,
        [
            "query_logs service=auth timerange=5m",
            "check_config service=auth",
            "diagnose root_cause=bad_config",
            "rollback_deploy service=auth",
        ],
        [
            "query_logs service=api timerange=5m",
            "diagnose root_cause=network_partition",
            "restart_service service=auth",
            "escalate reason=unclear",
        ],
        0.30,
    ),
    (
        CascadingFailureScenario,
        [
            "query_logs service=api timerange=10m",
            "check_deps service=api",
            "check_metrics service=database metric=connections",
            "query_logs service=database timerange=15m",
            "diagnose root_cause=db_connection_pool_exhausted",
            "kill_query service=database query_id=runaway_analytics",
            "scale_resource service=database resource=connection_pool",
        ],
        [
            "query_logs service=api timerange=10m",
            "check_config service=api",
            "diagnose root_cause=api_deployment",
            "rollback_deploy service=api",
            "escalate reason=need help",
        ],
        0.38,
    ),
    (
        AmbiguousIncidentScenario,
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
        [
            "query_logs service=frontend timerange=10m",
            "diagnose root_cause=api_deployment",
            "restart_service service=api",
            "escalate reason=need help",
        ],
        0.40,
    ),
    (
        MemoryLeakScenario,
        [
            "query_logs service=worker timerange=10m",
            "check_metrics service=worker metric=memory",
            "check_config service=worker",
            "diagnose root_cause=large_batch_size_oom",
            "rollback_deploy service=worker",
        ],
        [
            "query_logs service=api timerange=10m",
            "diagnose root_cause=db_timeout",
            "restart_service service=worker",
            "escalate reason=need help",
        ],
        0.35,
    ),
    (
        MegaIncidentScenario,
        [
            "query_logs service=database timerange=15m",
            "check_metrics service=database metric=connections",
            "query_logs service=cdn timerange=15m",
            "check_metrics service=cdn metric=tls_handshake_failures",
            "query_logs service=worker timerange=15m",
            "check_metrics service=worker metric=memory",
            "diagnose root_cause=db_pool_corrupted",
            "diagnose root_cause=cdn_tls_expired",
            "diagnose root_cause=worker_memory_leak",
            "scale_resource service=database resource=connection_pool",
            "rollback_deploy service=cdn",
            "rollback_deploy service=worker",
        ],
        [
            "query_logs service=api timerange=10m",
            "diagnose root_cause=api_deploy",
            "restart_service service=auth",
            "escalate reason=unclear",
        ],
        0.45,
    ),
]


@pytest.mark.parametrize(("scenario_cls", "optimal", "naive", "min_gap"), QUALITY_CASES)
def test_better_path_scores_higher_than_naive_path(
    scenario_cls, optimal, naive, min_gap
):
    optimal_total = sum(run_scenario_path(scenario_cls, optimal))
    naive_total = sum(run_scenario_path(scenario_cls, naive))
    assert optimal_total > naive_total
    assert (optimal_total - naive_total) >= min_gap


@pytest.mark.parametrize(
    ("task_name", "commands"),
    [
        (
            "single-service-alert",
            [
                "query_logs service=auth timerange=5m",
                "check_metrics service=auth metric=error_rate",
                "diagnose root_cause=bad_config",
                "rollback_deploy service=auth",
            ],
        ),
        (
            "cascading-failure",
            [
                "query_logs service=api timerange=10m",
                "check_metrics service=database metric=connections",
                "diagnose root_cause=db_connection_pool_exhausted",
                "kill_query service=database query_id=runaway_analytics",
                "scale_resource service=database resource=connection_pool",
            ],
        ),
        (
            "ambiguous-incident",
            [
                "query_logs service=frontend timerange=10m",
                "check_metrics service=dns-resolver metric=resolution_failures",
                "diagnose root_cause=dns_misconfiguration",
                "escalate reason=dns evidence gathered",
            ],
        ),
        (
            "memory-leak",
            [
                "query_logs service=worker timerange=10m",
                "check_metrics service=worker metric=memory",
                "diagnose root_cause=large_batch_size_oom",
                "rollback_deploy service=worker",
            ],
        ),
        (
            "cascading-platform-failure",
            [
                "query_logs service=database timerange=15m",
                "check_metrics service=database metric=connections",
                "query_logs service=cdn timerange=15m",
                "diagnose root_cause=db_pool_corrupted",
            ],
        ),
    ],
)
def test_environment_path_rewards_stay_in_bounds(task_name, commands):
    rewards = run_environment_path(task_name, commands)
    assert rewards, "Expected at least one reward value"
    for reward in rewards:
        assert 0.01 <= reward <= 0.99


@pytest.mark.parametrize(
    ("task_name", "max_steps"),
    [
        ("single-service-alert", SingleServiceAlertScenario.MAX_STEPS),
        ("cascading-failure", CascadingFailureScenario.MAX_STEPS),
        ("ambiguous-incident", AmbiguousIncidentScenario.MAX_STEPS),
        ("memory-leak", MemoryLeakScenario.MAX_STEPS),
    ],
)
def test_environment_terminates_when_max_steps_reached(task_name, max_steps):
    env = PraxisEnvironment()
    env.reset(task_name=task_name)

    result = None
    for _ in range(max_steps):
        result = env.step(PraxisAction(command="unknown command"))

    assert result is not None
    assert result["done"] is True
    assert 0.01 <= result["reward"] <= 0.99

    state = env.state()
    assert state.step_count == max_steps
