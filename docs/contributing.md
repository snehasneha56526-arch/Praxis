# Contributing to Praxis

Contributions are welcome: new scenarios, bug fixes, test improvements, and
documentation cleanup all help.

---

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## Running tests

```bash
pytest tests/ -v --tb=short
```

Targeted checks for the current implemented tasks:

```bash
pytest tests/test_task1_single_service_alert.py tests/test_task2_cascading_failure.py -v
```

---

## Current project structure

```text
praxis_env/
  models.py
  client.py
  scenarios/
    __init__.py
    base.py
    single_service_alert.py
    cascading_failure.py

server/
  app.py
  command_parser.py
  praxis_environment.py
  requirements.txt

tests/
  test_imports.py
  test_models.py
  test_command_parser.py
  test_environment.py
  test_task1_single_service_alert.py
  test_task2_cascading_failure.py

docs/
  README.md
  getting-started.md
  action-space.md
  observation-space.md
  tasks.md
  api-reference.md
  configuration.md
  contributing.md
```

Key implementation note:

- scenario classes own domain logic and return `StepOutcome`
- `PraxisEnvironment.step()` owns `step_count` and `cumulative_reward` bookkeeping

---

## Adding a new scenario

1. Create `praxis_env/scenarios/<your_scenario>.py`.
2. Subclass `BaseScenario`.
3. Implement `_reset_scenario_state()`, `step()`, and `get_initial_observation_text()`.
4. Register the new scenario in `praxis_env/scenarios/__init__.py`.
5. Add a dedicated scenario test file under `tests/`.
6. Update shared environment-contract tests if the public surface changes.

Minimal template:

```python
from praxis_env.scenarios.base import BaseScenario, ParsedCommand, StepOutcome


class YourScenario(BaseScenario):
    NAME = "your-scenario"
    SEVERITY = "P2"
    MAX_STEPS = 15
    ALERT_SUMMARY = "## INCIDENT ALERT"
    INITIAL_SYSTEM_STATUS = {"service-a": "critical"}
    INITIAL_AFFECTED_SERVICES = ["service-a"]

    def _reset_scenario_state(self) -> None:
        self._seen: set[str] = set()

    def step(self, command: ParsedCommand) -> StepOutcome:
        if command.action_type == "query_logs":
            return StepOutcome(
                investigation_result="Example log line",
                reward=self.clamp_reward(0.05),
                done=self.is_done(),
                incident_resolved=self._incident_resolved,
                root_cause_identified=self._root_cause_identified,
            )

        return self._handle_unknown_command(command.raw)

    def get_initial_observation_text(self) -> str:
        return ""
```

---

## Code standards

- Keep scenarios deterministic.
- Handle bad input without raising exceptions from `step()`.
- Clamp rewards with `self.clamp_reward(...)`.
- Return service names in `services_affected`; let the base observation builder
  derive that list from the current status map.
- Do not mutate `_step_count` or `_cumulative_reward` inside scenario code.
- Keep scenario text ASCII-friendly when practical.
- Run `pre-commit run --all-files` before pushing.

---

## Pull request checklist

- [ ] Scenario registry updated if you added a task
- [ ] Scenario-specific tests added or updated
- [ ] `pytest tests/ -v --tb=short` passes
- [ ] No broken links introduced in `docs/`
- [ ] No hardcoded secrets added
