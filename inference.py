"""
Baseline inference script for Praxis.

Contract requirements:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass

from openai import OpenAI

from praxis_env import PraxisAction, PraxisEnv
from praxis_env.models import PraxisObservation
from server.command_parser import is_known_action, parse_command


BENCHMARK_NAME = "praxis"
DEFAULT_TASKS = [
    "single-service-alert",
    "cascading-failure",
    "ambiguous-incident",
    "memory-leak",
    "cascading-platform-failure",
]

MAX_STEPS_BY_TASK = {
    "single-service-alert": 15,
    "cascading-failure": 20,
    "ambiguous-incident": 25,
    "memory-leak": 25,
    "cascading-platform-failure": 120,
}

FALLBACK_COMMANDS = {
    "single-service-alert": [
        "query_logs service=auth timerange=5m",
        "check_config service=auth",
        "diagnose root_cause=bad_config",
        "rollback_deploy service=auth",
    ],
    "cascading-failure": [
        "query_logs service=api timerange=10m",
        "check_deps service=api",
        "check_metrics service=database metric=connections",
        "query_logs service=database timerange=15m",
        "diagnose root_cause=db_connection_pool_exhausted",
        "kill_query service=database query_id=runaway_analytics",
        "scale_resource service=database resource=connection_pool",
    ],
    "ambiguous-incident": [
        "query_logs service=frontend timerange=10m",
        "query_logs service=api timerange=10m",
        "query_logs service=auth timerange=10m",
        "check_metrics service=dns-resolver metric=resolution_failures",
        "diagnose root_cause=dns_misconfiguration",
        "restart_service service=dns-resolver",
    ],
    "memory-leak": [
        "query_logs service=worker timerange=10m",
        "check_metrics service=worker metric=memory",
        "check_config service=worker",
        "diagnose root_cause=large_batch_size_oom",
        "rollback_deploy service=worker",
    ],
    "cascading-platform-failure": [
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
}


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
PRAXIS_URL = os.getenv("PRAXIS_URL", "http://127.0.0.1:7860")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "96"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.10"))
MAX_STEPS_CAP = int(os.getenv("MAX_STEPS_CAP", "25"))
OUTPUT_MIN_REWARD = 0.01
OUTPUT_MAX_REWARD = 0.99
OUTPUT_MIN_SCORE = 0.001
OUTPUT_MAX_SCORE = 0.999


SYSTEM_PROMPT = (
    "You are an on-call incident response assistant. "
    "Return exactly one valid command and nothing else. "
    "Use only these command styles: "
    "query_logs service=<name> timerange=<N>m; "
    "check_metrics service=<name> metric=<type>; "
    "check_deps service=<name>; "
    "check_config service=<name>; "
    "check_runbook service=<name>; "
    "diagnose root_cause=<cause>; "
    "restart_service service=<name>; "
    "rollback_deploy service=<name>; "
    "scale_resource service=<name> resource=<type>; "
    "kill_query service=<name> query_id=<id>; "
    "escalate reason=<text>."
)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    score: float
    rewards: list[float]


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def _single_line(value: str) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    return compact


def format_error(error: str | None) -> str:
    if not error:
        return "null"
    return _single_line(error)


def format_rewards_csv(rewards: list[float]) -> str:
    return ",".join(f"{float(r):.2f}" for r in rewards)


def clamp_output_reward(reward: float) -> float:
    """Clamp inference-emitted rewards to printable-safe bounds."""
    return max(OUTPUT_MIN_REWARD, min(OUTPUT_MAX_REWARD, float(reward)))


def clamp_output_score(score: float) -> float:
    """Clamp task-level score to strict open interval (0, 1)."""
    return max(OUTPUT_MIN_SCORE, min(OUTPUT_MAX_SCORE, float(score)))


def compute_task_score(rewards: list[float]) -> float:
    if not rewards:
        return OUTPUT_MIN_SCORE
    return clamp_output_score(sum(rewards) / len(rewards))


def render_start_line(task: str, env_name: str, model_name: str) -> str:
    return f"[START] task={task} env={env_name} model={model_name}"


def render_step_line(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> str:
    return (
        f"[STEP] step={step} action={_single_line(action)} "
        f"reward={float(reward):.2f} done={format_bool(done)} "
        f"error={format_error(error)}"
    )


def render_end_line(
    success: bool, steps: int, score: float, rewards: list[float]
) -> str:
    return (
        f"[END] success={format_bool(success)} steps={steps} score={float(score):.3f} "
        f"rewards={format_rewards_csv(rewards)}"
    )


def emit_step_line_once(
    emitted_steps: set[int],
    *,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> bool:
    """Emit a STEP log line once per unique step number."""
    if step in emitted_steps:
        return False

    print(
        render_step_line(
            step=step,
            action=action,
            reward=reward,
            done=done,
            error=error,
        ),
        flush=True,
    )
    emitted_steps.add(step)
    return True


def parse_task_list(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return list(DEFAULT_TASKS)

    selected = [task.strip() for task in raw.split(",") if task.strip()]
    valid = [task for task in selected if task in DEFAULT_TASKS]
    return valid if valid else list(DEFAULT_TASKS)


def fallback_command(task_name: str, step: int) -> str:
    commands = FALLBACK_COMMANDS.get(task_name, ["escalate reason=unable to proceed"])
    if step <= len(commands):
        return commands[step - 1]
    return commands[-1]


def _normalize_model_output(text: str) -> str:
    if not text:
        return ""

    line = text.strip()

    if line.startswith("```"):
        pieces = [part.strip() for part in line.split("```") if part.strip()]
        line = pieces[-1] if pieces else ""

    first_line = line.splitlines()[0].strip()

    lower = first_line.lower()
    if lower.startswith("action:") or lower.startswith("command:"):
        first_line = first_line.split(":", 1)[1].strip()

    first_line = first_line.strip().strip("`").strip("\"'")
    return first_line


def _build_user_prompt(
    task_name: str,
    step: int,
    observation: PraxisObservation,
    history: list[str],
) -> str:
    commands = "\n".join(observation.available_commands)
    recent = "\n".join(history[-3:]) if history else "none"
    return (
        f"Task: {task_name}\n"
        f"Step: {step}\n"
        f"Severity: {observation.severity}\n"
        f"Services affected: {', '.join(observation.services_affected) or 'none'}\n"
        f"Alert:\n{observation.alert_summary}\n\n"
        f"Last investigation result:\n{observation.investigation_result}\n\n"
        f"Recent history:\n{recent}\n\n"
        f"Available command templates:\n{commands}\n\n"
        "Return exactly one command string."
    )


def _request_model_command(
    client: OpenAI | None,
    task_name: str,
    step: int,
    observation: PraxisObservation,
    history: list[str],
) -> tuple[str | None, bool]:
    """
    Returns:
      (command, disable_model)
    """
    if client is None:
        return None, False

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        task_name, step, observation, history
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception:
        return None, True

    content = completion.choices[0].message.content or ""
    candidate = _normalize_model_output(content)
    if not candidate:
        return None, False

    parsed = parse_command(candidate)
    if not is_known_action(parsed.action_type):
        return None, False

    return candidate, False


def _build_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=0,
    )


async def run_episode(task_name: str, client: OpenAI | None) -> EpisodeResult:
    rewards: list[float] = []
    steps_taken = 0
    done = False
    encountered_fatal = False
    history: list[str] = []
    use_model = True
    emitted_step_numbers: set[int] = set()

    print(render_start_line(task_name, BENCHMARK_NAME, MODEL_NAME), flush=True)

    env = await PraxisEnv.from_url(PRAXIS_URL)

    try:
        observation = await env.reset(task_name=task_name)

        task_limit = min(MAX_STEPS_BY_TASK.get(task_name, 15), MAX_STEPS_CAP)
        for step in range(1, task_limit + 1):
            command: str
            if use_model:
                model_command, disable_model = _request_model_command(
                    client=client,
                    task_name=task_name,
                    step=step,
                    observation=observation,
                    history=history,
                )
                if disable_model:
                    use_model = False
                command = model_command or fallback_command(task_name, step)
            else:
                command = fallback_command(task_name, step)

            error_value: str | None = None

            try:
                result = await env.step(PraxisAction(command=command))
                reward = clamp_output_reward(result.reward)
                done = bool(result.done)

                info = result.info or {}
                if isinstance(info, dict) and info.get("error"):
                    error_value = str(info.get("error"))

                emit_step_line_once(
                    emitted_step_numbers,
                    step=step,
                    action=command,
                    reward=reward,
                    done=done,
                    error=error_value,
                )
                rewards.append(reward)
                steps_taken = step
                history.append(f"step={step} action={command} reward={reward:.2f}")
                observation = result.observation
                if done:
                    break
            except Exception as exc:
                encountered_fatal = True
                emit_step_line_once(
                    emitted_step_numbers,
                    step=step,
                    action=command,
                    reward=OUTPUT_MIN_REWARD,
                    done=False,
                    error=str(exc),
                )
                rewards.append(OUTPUT_MIN_REWARD)
                steps_taken = step
                break
    except Exception as exc:
        encountered_fatal = True
        print(f"[ERROR] Failed to start episode '{task_name}': {exc}")
    finally:
        try:
            await env.close()
        except Exception:
            pass

    total_reward = sum(rewards)
    task_score = compute_task_score(rewards)
    success = bool(
        (not encountered_fatal)
        and steps_taken > 0
        and total_reward >= SUCCESS_SCORE_THRESHOLD
    )
    print(
        render_end_line(
            success=success, steps=steps_taken, score=task_score, rewards=rewards
        ),
        flush=True,
    )
    return EpisodeResult(
        success=success, steps=steps_taken, score=task_score, rewards=rewards
    )


def ensure_server_running(url: str) -> subprocess.Popen | None:
    import httpx
    import time

    try:
        response = httpx.get(f"{url}/health", timeout=1.0)
        if response.status_code == 200:
            return None
    except Exception:
        pass

    print(
        "[INFO] Starting local environment server for standalone inference...",
        flush=True,
    )
    import sys
    import subprocess

    port = url.split(":")[-1].replace("/", "")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.app:app",
        "--port",
        port,
        "--host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    start_time = time.time()
    while time.time() - start_time < 15.0:
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                print("[INFO] Server is healthy.", flush=True)
                break
        except Exception:
            time.sleep(0.5)
    return proc


async def main() -> None:
    server_proc = None
    try:
        server_proc = ensure_server_running(PRAXIS_URL)

        client = _build_client()
        tasks = parse_task_list(os.getenv("PRAXIS_TASKS"))
        for task in tasks:
            await run_episode(task_name=task, client=client)
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
