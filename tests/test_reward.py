"""
tests/test_reward.py - Phase 6 reward engine unit tests.

These tests validate centralized reward scoring semantics independently from
scenario business logic.
"""

import random

import pytest

from server.reward import (
    DEFAULT_REWARD_POLICIES,
    _MEMORY_EVENTS,
    RewardEngine,
    RewardPolicy,
    clamp_reward,
)


def test_clamp_reward_bounds():
    assert clamp_reward(-0.5) == pytest.approx(0.01)
    assert clamp_reward(0.25) == pytest.approx(0.25)
    assert clamp_reward(1.5) == pytest.approx(0.99)


def test_known_event_value_for_single_service_diagnosis():
    engine = RewardEngine()
    result = engine.score(
        task_name="single-service-alert",
        event="diagnosis.correct",
    )
    assert result.reward == pytest.approx(0.20)
    assert result.breakdown.diagnosis_reward == pytest.approx(0.20)


def test_duplicate_investigation_applies_redundancy_penalty():
    engine = RewardEngine()
    result = engine.score(
        task_name="single-service-alert",
        event="investigation.query_logs.auth",
        duplicate=True,
    )
    assert result.reward == pytest.approx(0.01)
    assert result.breakdown.investigation_reward == pytest.approx(0.0)
    assert result.breakdown.redundancy_penalty == pytest.approx(-0.02)


def test_premature_penalty_is_applied_and_clamped():
    engine = RewardEngine()
    result = engine.score(
        task_name="single-service-alert",
        event="escalation.no_evidence",
        premature=True,
    )
    assert result.reward == pytest.approx(0.01)
    assert result.breakdown.premature_penalty == pytest.approx(-0.05)


def test_destructive_penalty_is_applied_and_clamped():
    engine = RewardEngine()
    result = engine.score(
        task_name="single-service-alert",
        event="remediation.wrong",
        destructive=True,
    )
    assert result.reward == pytest.approx(0.01)
    assert result.breakdown.destructive_penalty == pytest.approx(-0.15)


def test_unknown_task_policy_raises_value_error():
    engine = RewardEngine()
    with pytest.raises(ValueError, match="Unknown reward policy"):
        engine.score(task_name="not-a-task", event="diagnosis.correct")


def test_efficiency_bonus_applies_for_custom_policy():
    policy = RewardPolicy(
        event_values={"remediation.correct": 0.25},
        redundancy_penalty=0.0,
        premature_penalty=0.0,
        destructive_penalty=0.0,
        efficiency_bonus_max=0.10,
        time_pressure_cost_per_step=0.0,
    )
    engine = RewardEngine(policies={"custom": policy})

    result = engine.score(
        task_name="custom",
        event="remediation.correct",
        resolved=True,
        step_number=2,
        max_steps=10,
    )
    assert result.breakdown.efficiency_bonus == pytest.approx(0.08, abs=1e-6)
    assert result.reward == pytest.approx(0.33, abs=1e-6)


def test_time_pressure_cost_applies_for_custom_policy():
    policy = RewardPolicy(
        event_values={"investigation.probe": 0.05},
        redundancy_penalty=0.0,
        premature_penalty=0.0,
        destructive_penalty=0.0,
        efficiency_bonus_max=0.0,
        time_pressure_cost_per_step=0.01,
    )
    engine = RewardEngine(policies={"custom": policy})

    result = engine.score(
        task_name="custom",
        event="investigation.probe",
        step_number=1,
        max_steps=10,
    )
    assert result.breakdown.time_pressure_cost == pytest.approx(-0.01, abs=1e-6)
    assert result.reward == pytest.approx(0.04, abs=1e-6)


def test_clamp_upper_bound_with_large_event_reward():
    policy = RewardPolicy(
        event_values={"diagnosis.correct": 1.25},
        redundancy_penalty=0.0,
        premature_penalty=0.0,
        destructive_penalty=0.0,
    )
    engine = RewardEngine(policies={"custom": policy})

    result = engine.score(task_name="custom", event="diagnosis.correct")
    assert result.reward == pytest.approx(0.99)
    assert result.breakdown.total_unclamped == pytest.approx(1.25)


def test_all_default_policies_include_memory_event_rows():
    expected_keys = set(_MEMORY_EVENTS.keys())
    for task_name, policy in DEFAULT_REWARD_POLICIES.items():
        assert expected_keys.issubset(policy.event_values.keys()), task_name


def test_memory_recall_after_cutoff_value_is_used_pre_clamp():
    engine = RewardEngine()
    result = engine.score(
        task_name="single-service-alert",
        event="memory.recall_memory.after_cutoff",
    )
    assert result.breakdown.total_unclamped == pytest.approx(0.08)
    assert result.reward == pytest.approx(0.08)


@pytest.mark.parametrize("task_name", sorted(DEFAULT_REWARD_POLICIES.keys()))
def test_remediation_requires_diagnosis(task_name: str):
    engine = RewardEngine()
    result = engine.score(
        task_name=task_name,
        event="remediation.wrong",
        root_cause_identified=False,
    )
    assert result.breakdown.remediation_reward == pytest.approx(0.0)
    assert result.breakdown.total_unclamped == pytest.approx(0.0)
    assert result.reward == pytest.approx(0.01)


def test_random_sequences_remain_clamped():
    engine = RewardEngine()
    rng = random.Random(42)
    policies = list(DEFAULT_REWARD_POLICIES.items())
    for _ in range(1000):
        task_name, policy = rng.choice(policies)
        event = rng.choice(tuple(policy.event_values.keys()))
        result = engine.score(
            task_name=task_name,
            event=event,
            duplicate=rng.choice((True, False)),
            premature=rng.choice((True, False)),
            destructive=rng.choice((True, False)),
            root_cause_identified=rng.choice((True, False)),
            resolved=rng.choice((True, False)),
            step_number=rng.randint(1, 120),
            max_steps=rng.randint(1, 120),
        )
        assert 0.01 <= result.reward <= 0.99
