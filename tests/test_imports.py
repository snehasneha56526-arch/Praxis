"""
tests/test_imports.py — Phase 1 import validation.

The single most important test: can everything be imported without errors?
If this fails, the environment will NEVER pass openenv validate.
"""


def test_import_praxis_env_package():
    """The public API must be importable."""
    from praxis_env import (
        PraxisAction,
        PraxisEnv,
        PraxisMemory,
        PraxisObservation,
        PraxisState,
    )

    assert PraxisAction is not None
    assert PraxisObservation is not None
    assert PraxisState is not None
    assert PraxisEnv is not None
    assert PraxisMemory is not None


def test_import_models_directly():
    """Direct model imports must work."""
    from praxis_env.models import (
        PraxisAction,
    )

    assert PraxisAction is not None


def test_import_scenarios_package():
    """Scenario registry must be importable."""
    from praxis_env.scenarios import get_scenario, list_tasks

    assert callable(get_scenario)
    assert callable(list_tasks)


def test_import_base_scenario():
    """Base scenario must be importable."""
    from praxis_env.scenarios.base import BaseScenario, ParsedCommand, StepOutcome

    assert BaseScenario is not None
    assert ParsedCommand is not None
    assert StepOutcome is not None


def test_import_server_environment():
    """Server environment must be importable."""
    from server.praxis_environment import PraxisEnvironment

    assert PraxisEnvironment is not None


def test_import_server_reward_engine():
    """Centralized reward engine must be importable."""
    from server.reward import RewardEngine, RewardPolicy, RewardResult

    assert RewardEngine is not None
    assert RewardPolicy is not None
    assert RewardResult is not None


def test_import_server_app():
    """FastAPI app must be importable."""
    from server.app import app, create_app, main

    assert app is not None
    assert callable(create_app)
    assert callable(main)


def test_app_has_routes():
    """FastAPI app must have the required routes."""
    from server.app import app

    paths = {route.path for route in app.routes}
    assert "/reset" in paths, "POST /reset route missing"
    assert "/step" in paths, "POST /step route missing"
    assert "/state" in paths, "GET /state route missing"
    assert "/health" in paths, "GET /health route missing"
