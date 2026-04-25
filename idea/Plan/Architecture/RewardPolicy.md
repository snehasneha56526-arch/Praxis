# Reward Policy — Per-Task Event Tables + Memory Bonuses

> Single canonical source for `server/reward.py` calibration. All values pre-clamp.
> Reward engine clamps to `[0.01, 0.99]` (judge-safe open interval). Memory bonuses
> stack inside that band.
>
> Read alongside [`MemoryModel.md`](./MemoryModel.md) and
> [`ScenarioCatalog.md`](./ScenarioCatalog.md). Implementation lives at
> [`server/reward.py`](../../../server/reward.py).

---

## 1. Cross-task memory event row (added to **every** policy)

| Event tag                            | Value | Why                                                         |
| ------------------------------------ | ----- | ----------------------------------------------------------- |
| `memory.save_finding.before_cutoff`  | +0.05 | Reward foresight — saving when the agent doesn't _need_ to. |
| `memory.save_finding.after_cutoff`   | +0.02 | Late saves are fine; smaller signal.                        |
| `memory.recall_memory.before_cutoff` | +0.01 | Discourages habitual recalls.                               |
| `memory.recall_memory.after_cutoff`  | +0.08 | Strong reward — proves a planned memory strategy.           |
| `memory.illegal_log_after_cutoff`    | −0.05 | Tried to query logs after the cutoff banner appeared.       |
| `memory.empty_recall_after_cutoff`   | −0.02 | Recalled when nothing was saved — wasted a turn.            |

These rows are appended to each task's `event_values` map by `server/reward.py` via a `_with_memory_events()` helper (Issue #5).

---

## 2. Existing tasks (calibration kept; memory rows added)

### `single-service-alert` (easy, target ~0.63)

Target optimal in 4 steps. Investigation generous; no step cost.

```python
"investigation.query_logs.auth": 0.08,
"investigation.query_logs.default": 0.05,
"investigation.check_metrics.connections": 0.08,
"investigation.check_metrics.default": 0.05,
"investigation.check_deps.default": 0.05,
"investigation.check_config.default": 0.10,
"investigation.check_runbook.default": 0.05,
"diagnosis.bad_config": 0.20,
"diagnosis.wrong": 0.0,
"remediation.rollback": 0.25,
"remediation.wrong": 0.0,
"escalation.with_evidence": 0.15,
"escalation.no_evidence": 0.0,
# + cross-task memory rows
```

### `ambiguous-incident` (medium, target ~0.71)

Evidence threshold (≥3 services + ≥1 infra) before diagnosis credit.

```python
"investigation.dns": 0.10,
"investigation.lb": 0.05,
"investigation.app_service": 0.04,
"diagnosis.dns_misconfiguration": 0.20,
"escalation.with_evidence": 0.15,
"time_pressure_cost_per_step": 0.003,
# + cross-task memory rows
```

### `cascading-failure` (hard, target ~0.46)

Strong step pressure; remediation requires kill_query AND scale_resource.

```python
"investigation.query_logs.database": 0.05,
"investigation.check_metrics.database.connections": 0.10,
"diagnosis.db_connection_pool_exhausted": 0.15,
"remediation.kill_query": 0.15,
"remediation.scale_resource": 0.10,
"time_pressure_cost_per_step": 0.005,
"destructive_penalty": -0.15,
# + cross-task memory rows
```

### `memory-leak` (hard, target ~0.48)

First task with **active** memory pressure (cutoff = 20 of 25).

```python
"investigation.check_metrics.worker.memory": 0.08,
"investigation.query_logs.worker": 0.05,
"diagnosis.memory_leak": 0.15,
"remediation.restart_worker": 0.10,
"remediation.fix_session_cache": 0.15,
"time_pressure_cost_per_step": 0.004,
# + cross-task memory rows  (these matter — agent must save heap-growth evidence)
```

---

## 3. NEW `cascading-platform-failure` (mega-incident)

Sparse milestone rewards by design. See [`ScenarioCatalog.md`](./ScenarioCatalog.md) §3.

```python
"cascading-platform-failure": RewardPolicy(
    event_values={
        # Investigation — sparse signal
        "investigation.query_logs.database": 0.03,
        "investigation.query_logs.cdn": 0.03,
        "investigation.query_logs.worker": 0.03,
        "investigation.query_logs.default": 0.01,
        "investigation.check_metrics.database.connections": 0.03,
        "investigation.check_metrics.worker.memory": 0.03,
        "investigation.check_metrics.cdn.tls": 0.03,
        "investigation.check_metrics.default": 0.01,
        "investigation.check_deps.default": 0.02,
        "investigation.check_config.database": 0.02,
        "investigation.check_config.worker": 0.02,
        "investigation.check_config.cdn": 0.02,
        "investigation.check_config.default": 0.01,
        "investigation.check_runbook.default": 0.02,

        # Milestones (sparse — the whole point)
        "diagnosis.first_correct": 0.12,
        "diagnosis.all_correct":   0.15,
        "diagnosis.wrong":         0.0,

        # Remediation
        "remediation.complete": 0.20,
        "remediation.partial":  0.05,
        "remediation.wrong":    0.0,

        # Escalation
        "escalation.with_evidence": 0.10,
        "escalation.no_evidence":   0.0,

        # Errors
        "unknown_command": 0.0,
        "invalid_input":   0.0,
    },
    redundancy_penalty=-0.02,
    premature_penalty=-0.05,
    destructive_penalty=-0.10,
    time_pressure_cost_per_step=0.002,
)
# + cross-task memory rows merged in via _with_memory_events()
```

---

## 4. `procedural-incident` (seeded, shipped in Issue #8)

Reward map is **built at scenario init** by `_build_procedural_policy(difficulty)`. Skeleton:

```python
def _build_procedural_policy(difficulty: str) -> RewardPolicy:
    base = {
        "easy":   {"diagnosis": 0.20, "remediation": 0.25, "step_cost": 0.0},
        "medium": {"diagnosis": 0.15, "remediation": 0.20, "step_cost": 0.003},
        "hard":   {"diagnosis": 0.12, "remediation": 0.15, "step_cost": 0.005},
    }[difficulty]

    return RewardPolicy(
        event_values={
            f"diagnosis.<chosen_root_cause>": base["diagnosis"],
            "diagnosis.wrong": 0.0,
            f"remediation.<chosen_remediation>": base["remediation"],
            "remediation.wrong": 0.0,
            "escalation.with_evidence": 0.15,
            # investigation table reused from a difficulty-aware default
            **DEFAULT_INVESTIGATION_TABLE[difficulty],
        },
        time_pressure_cost_per_step=base["step_cost"],
        destructive_penalty=-0.10,
    )
```

The chosen root cause and remediation are picked from the pools in [`ScenarioCatalog.md`](./ScenarioCatalog.md) §4.2 using `random.Random(seed)`.

---

## 5. Helper: `_with_memory_events`

```python
_MEMORY_EVENTS: Mapping[str, float] = {
    "memory.save_finding.before_cutoff":  0.05,
    "memory.save_finding.after_cutoff":   0.02,
    "memory.recall_memory.before_cutoff": 0.01,
    "memory.recall_memory.after_cutoff":  0.08,
    "memory.illegal_log_after_cutoff":   -0.05,
    "memory.empty_recall_after_cutoff":  -0.02,
}

def _with_memory_events(events: Mapping[str, float]) -> Mapping[str, float]:
    """Return a new mapping merging memory event tags into `events`."""
    merged = dict(events)
    merged.update(_MEMORY_EVENTS)
    return merged
```

`DEFAULT_REWARD_POLICIES` is built by wrapping every existing `event_values` through `_with_memory_events`. Issue #5 enforces this in tests by asserting every policy has all 6 memory keys.

---

## 6. Clamp + composition reminder

- `RewardEngine.score()` returns a `RewardResult(reward, breakdown)` where `reward = clamp(sum(components), 0.01, 0.99)`.
- Memory bonuses are added in the `investigation_reward` slot (positive) or `redundancy_penalty` slot (negative); they do **not** double-clamp before the final sum.
- Per ADR-13, any `remediation.*` event is forced to zero pre-clamp when `root_cause_identified=False` so remediation cannot score before diagnosis.
- Issue #5 acceptance criteria require: the test suite proves no policy exceeds `[0.01, 0.99]` over 1000 random sequences.

---

## 7. Audit checklist (every PR that touches reward)

- [ ] Each task in `DEFAULT_REWARD_POLICIES` has the 6 memory event tags.
- [ ] No event value is outside `[-0.20, +0.30]` pre-clamp (sanity bound).
- [ ] `time_pressure_cost_per_step` is set per task as in §2/§3.
- [ ] `clamp_reward` is applied exactly once at the boundary.
- [ ] `remediation.*` events are zeroed when `root_cause_identified=False` (ADR-13 evidence gate).
- [ ] `pytest -q tests/test_reward.py` and `tests/test_memory.py` pass.
