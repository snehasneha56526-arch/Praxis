"""
tests/test_task6_procedural.py - Procedural scenario contract tests.
"""

import pytest
from fastapi.testclient import TestClient

from praxis_env.models import PraxisAction
from praxis_env.scenarios.procedural_incident import ProceduralIncidentScenario
from server.app import app
from server.praxis_environment import PraxisEnvironment


client = TestClient(app)


@pytest.mark.parametrize(
    ("difficulty", "expected_steps", "expected_cutoff"),
    [("easy", 15, 8), ("medium", 25, 15), ("hard", 50, 25)],
)
def test_procedural_reset_stamps_difficulty_runtime_config(
    difficulty, expected_steps, expected_cutoff
):
    scenario = ProceduralIncidentScenario(seed=7, difficulty=difficulty)
    scenario.reset(episode_id="procedural-runtime-config")

    assert scenario.MAX_STEPS == expected_steps
    assert scenario.MEMORY_CUTOFF_OVERRIDE == expected_cutoff
    obs = scenario.get_observation()
    assert obs.alert_summary
    assert obs.system_status


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_same_seed_and_difficulty_are_deterministic_for_three_runs(difficulty):
    commands = [
        "query_logs",
        "check_metrics metric=error_rate",
        "check_config",
        "diagnose root_cause=definitely_wrong",
        "rollback_deploy service=api",
    ]

    def _run_once():
        env = PraxisEnvironment()
        initial = env.reset(task_name=f"procedural-incident:{difficulty}", seed=99)
        trace = [initial.model_dump(mode="json")]
        for command in commands:
            result = env.step(PraxisAction(command=command))
            trace.append(result)
            if result["done"]:
                break
        return trace

    run_1 = _run_once()
    run_2 = _run_once()
    run_3 = _run_once()
    assert run_1 == run_2 == run_3


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_remediation_before_diagnosis_scores_zero(difficulty):
    env = PraxisEnvironment()
    env.reset(task_name=f"procedural-incident:{difficulty}", seed=123)

    result = env.step(PraxisAction(command="rollback_deploy service=api"))
    assert result["reward"] == pytest.approx(0.01)
    assert result["done"] is False


def test_reset_assigns_default_seed_and_returns_metadata():
    response = client.post(
        "/reset", json={"task_name": "procedural-incident", "seed": None}
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert isinstance(metadata["seed"], int)
    assert metadata["difficulty"] == "medium"
    assert metadata["task_name"] == "procedural-incident"


def test_reset_preserves_explicit_seed_in_metadata():
    response = client.post(
        "/reset", json={"task_name": "procedural-incident:hard", "seed": 42}
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["seed"] == 42
    assert metadata["difficulty"] == "hard"
    assert metadata["task_name"] == "procedural-incident"


def test_reset_rejects_non_integer_seed():
    response = client.post(
        "/reset",
        json={"task_name": "procedural-incident", "seed": "not-an-int"},
    )
    assert response.status_code == 400
    assert "Invalid seed" in response.json()["detail"]
