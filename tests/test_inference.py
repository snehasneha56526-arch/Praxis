"""
tests/test_inference.py - Phase 7 output contract tests.

Focuses on pure formatting and command-selection helpers so tests are
deterministic and do not require network access.
"""

from __future__ import annotations

import re

import pytest

import inference


def test_render_start_line_contract():
    line = inference.render_start_line(
        task="single-service-alert",
        env_name="praxis",
        model_name="Qwen/Qwen2.5-72B-Instruct",
    )
    assert (
        line
        == "[START] task=single-service-alert env=praxis model=Qwen/Qwen2.5-72B-Instruct"
    )


def test_render_step_line_contract_two_decimal_and_lowercase_bool():
    line = inference.render_step_line(
        step=2,
        action="query_logs service=auth timerange=5m",
        reward=0.5,
        done=False,
        error=None,
    )
    assert (
        line == "[STEP] step=2 action=query_logs service=auth timerange=5m "
        "reward=0.50 done=false error=null"
    )


def test_render_step_line_normalizes_multiline_action_and_error():
    line = inference.render_step_line(
        step=1,
        action="query_logs service=auth\n timerange=5m",
        reward=0.0,
        done=True,
        error="unknown\ncommand",
    )
    assert "\n" not in line
    assert "done=true" in line
    assert "error=unknown command" in line


def test_render_end_line_contract_and_rewards_csv():
    line = inference.render_end_line(
        success=True,
        steps=3,
        score=0.083333,
        rewards=[0.01, 0.05, 0.2],
    )
    assert line == "[END] success=true steps=3 score=0.083 rewards=0.01,0.05,0.20"


def test_render_end_line_allows_empty_rewards_list():
    line = inference.render_end_line(success=False, steps=0, score=0.001, rewards=[])
    assert line == "[END] success=false steps=0 score=0.001 rewards="


def test_clamp_output_reward_uses_printable_safe_bounds():
    assert inference.clamp_output_reward(-1.0) == pytest.approx(0.01)
    assert inference.clamp_output_reward(0.001) == pytest.approx(0.01)
    assert inference.clamp_output_reward(0.55) == pytest.approx(0.55)
    assert inference.clamp_output_reward(0.999) == pytest.approx(0.99)
    assert inference.clamp_output_reward(2.0) == pytest.approx(0.99)


def test_clamp_output_reward_exact_open_interval_edges():
    assert inference.clamp_output_reward(0.0) == pytest.approx(0.01)
    assert inference.clamp_output_reward(1.0) == pytest.approx(0.99)


def test_clamp_output_score_uses_strict_open_interval_bounds():
    assert inference.clamp_output_score(-1.0) == pytest.approx(0.001)
    assert inference.clamp_output_score(0.0) == pytest.approx(0.001)
    assert inference.clamp_output_score(0.42) == pytest.approx(0.42)
    assert inference.clamp_output_score(1.0) == pytest.approx(0.999)
    assert inference.clamp_output_score(5.0) == pytest.approx(0.999)


def test_compute_task_score_uses_mean_and_is_strictly_inside_open_interval():
    assert inference.compute_task_score([]) == pytest.approx(0.001)
    assert inference.compute_task_score([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert inference.compute_task_score([0.99, 0.99]) == pytest.approx(0.99)


def test_format_rewards_csv_printable_safe_bounds():
    assert inference.format_rewards_csv([0.01, 0.99]) == "0.01,0.99"


def test_parse_task_list_uses_defaults_for_empty_input():
    tasks = inference.parse_task_list(None)
    assert tasks == [
        "single-service-alert",
        "cascading-failure",
        "ambiguous-incident",
        "memory-leak",
        "cascading-platform-failure",
    ]


def test_parse_task_list_filters_invalid_tasks():
    tasks = inference.parse_task_list(
        "single-service-alert,not-a-task,cascading-failure"
    )
    assert tasks == ["single-service-alert", "cascading-failure"]


def test_parse_task_list_falls_back_when_all_invalid():
    tasks = inference.parse_task_list("foo,bar")
    assert tasks == [
        "single-service-alert",
        "cascading-failure",
        "ambiguous-incident",
        "memory-leak",
        "cascading-platform-failure",
    ]


def test_fallback_command_sequences_start_correctly():
    assert (
        inference.fallback_command("single-service-alert", 1)
        == "query_logs service=auth timerange=5m"
    )
    assert (
        inference.fallback_command("cascading-failure", 1)
        == "query_logs service=api timerange=10m"
    )
    assert (
        inference.fallback_command("ambiguous-incident", 1)
        == "query_logs service=frontend timerange=10m"
    )


def test_fallback_command_clamps_to_last_sequence_item():
    cmd = inference.fallback_command("single-service-alert", 999)
    assert cmd == "rollback_deploy service=auth"


def test_model_output_normalization_keeps_single_command():
    text = "command: query_logs service=auth timerange=5m\nextra text"
    normalized = inference._normalize_model_output(text)
    assert normalized == "query_logs service=auth timerange=5m"


def test_step_line_matches_contract_regex():
    line = inference.render_step_line(
        step=4,
        action="diagnose root_cause=bad_config",
        reward=0.2,
        done=False,
        error=None,
    )
    pattern = re.compile(
        r"^\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=(.+)$"
    )
    assert pattern.match(line)


def test_emit_step_line_once_emits_on_first_use(capsys):
    emitted: set[int] = set()
    did_emit = inference.emit_step_line_once(
        emitted,
        step=2,
        action="check_deps service=api",
        reward=0.02,
        done=False,
        error=None,
    )

    captured = capsys.readouterr()
    assert did_emit is True
    assert (
        "[STEP] step=2 action=check_deps service=api reward=0.02 done=false error=null"
        in captured.out
    )
    assert emitted == {2}


def test_emit_step_line_once_skips_duplicate_step(capsys):
    emitted: set[int] = set()
    first = inference.emit_step_line_once(
        emitted,
        step=2,
        action="check_deps service=api",
        reward=0.02,
        done=False,
        error=None,
    )
    second = inference.emit_step_line_once(
        emitted,
        step=2,
        action="check_deps service=api",
        reward=0.02,
        done=False,
        error=None,
    )

    captured = capsys.readouterr()
    assert first is True
    assert second is False
    assert captured.out.count("[STEP] step=2") == 1


class _FakeStepResult:
    def __init__(self, observation, reward, done, info=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


def _make_fake_observation(step_number: int = 0):
    return inference.PraxisObservation(
        alert_summary="Synthetic test alert",
        system_status={"auth": "critical"},
        investigation_result="Synthetic investigation result",
        available_commands=["query_logs service=<name> timerange=<N>m"],
        time_elapsed_minutes=float(step_number * 2.5),
        severity="P2",
        services_affected=["auth"],
        step_number=step_number,
    )


class _FakeEnv:
    def __init__(self, *, reward: float | None = None, raise_on_step: bool = False):
        self._reward = reward
        self._raise_on_step = raise_on_step
        self._step_count = 0

    async def reset(self, task_name: str = "single-service-alert"):
        self._step_count = 0
        return _make_fake_observation(step_number=0)

    async def step(self, action):
        if self._raise_on_step:
            raise RuntimeError("simulated_step_failure")

        self._step_count += 1
        return _FakeStepResult(
            observation=_make_fake_observation(step_number=self._step_count),
            reward=float(self._reward if self._reward is not None else 0.5),
            done=True,
            info={},
        )

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_run_episode_clamps_raw_zero_reward_to_output_floor(monkeypatch, capsys):
    fake_env = _FakeEnv(reward=0.0)

    async def _fake_from_url(cls, url: str, timeout: float = 30.0):
        return fake_env

    monkeypatch.setattr(inference.PraxisEnv, "from_url", classmethod(_fake_from_url))

    result = await inference.run_episode("single-service-alert", client=None)
    out = capsys.readouterr().out

    assert result.rewards == [pytest.approx(0.01)]
    assert result.score == pytest.approx(0.01)
    assert "reward=0.01" in out
    assert "score=0.010" in out
    assert "rewards=0.01" in out


@pytest.mark.asyncio
async def test_run_episode_clamps_raw_one_reward_to_output_ceiling(monkeypatch, capsys):
    fake_env = _FakeEnv(reward=1.0)

    async def _fake_from_url(cls, url: str, timeout: float = 30.0):
        return fake_env

    monkeypatch.setattr(inference.PraxisEnv, "from_url", classmethod(_fake_from_url))

    result = await inference.run_episode("single-service-alert", client=None)
    out = capsys.readouterr().out

    assert result.rewards == [pytest.approx(0.99)]
    assert result.score == pytest.approx(0.99)
    assert "reward=0.99" in out
    assert "score=0.990" in out
    assert "rewards=0.99" in out


@pytest.mark.asyncio
async def test_run_episode_step_exception_uses_output_floor(monkeypatch, capsys):
    fake_env = _FakeEnv(raise_on_step=True)

    async def _fake_from_url(cls, url: str, timeout: float = 30.0):
        return fake_env

    monkeypatch.setattr(inference.PraxisEnv, "from_url", classmethod(_fake_from_url))

    result = await inference.run_episode("single-service-alert", client=None)
    out = capsys.readouterr().out

    assert result.rewards == [pytest.approx(0.01)]
    assert result.score == pytest.approx(0.01)
    assert "reward=0.01" in out
    assert "score=0.010" in out
    assert "simulated_step_failure" in out
    assert "rewards=0.01" in out
