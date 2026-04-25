# API Contract — Praxis Theme #2

> Source of truth for HTTP endpoints, request/response shapes, action/observation/state schemas, and the new session header.
>
> Evidence: `server/app.py`, `praxis_env/models.py`, `OpenEnv/src/openenv/core/env_server/types.py`,
> `idea/PROBLEM STATEMENT/How Judging works.md`, `idea/Task.md`.

---

## 1. Endpoints

| Method | Path         | Purpose                                                                                                                                                  | Session header required |
| ------ | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `GET`  | `/health`    | Liveness probe — must return `{"status": "healthy"}` (judging gate).                                                                                     | No                      |
| `GET`  | `/metadata`  | Environment metadata (name, version, tasks, endpoints).                                                                                                  | No                      |
| `GET`  | `/schema`    | JSON Schema for action / observation / state (OpenEnv validator).                                                                                        | No                      |
| `GET`  | `/tasks`     | List of registered task names.                                                                                                                           | No                      |
| `POST` | `/reset`     | Allocate a session, load a scenario, return initial observation.                                                                                         | No (issues one)         |
| `POST` | `/step`      | Execute one action against an existing session.                                                                                                          | **Yes**                 |
| `GET`  | `/state`     | Current `PraxisState` for a session.                                                                                                                     | **Yes**                 |
| `POST` | `/mcp`       | Minimal JSON-RPC stub for OpenEnv runtime validators.                                                                                                    | No                      |
| `GET`  | `/benchmark` | Benchmark surface (Issue #21, ADR-14) — reads `docs/baseline_scores.md` and returns model-vs-mean-score rows for the "designed-for-benchmarking" signal. | No                      |
| `GET`  | `/`          | Friendly redirect/info page for judges browsing the HF Space.                                                                                            | No                      |

The session header is `X-Session-Id: <uuid4>`. See [`ConcurrencyModel.md`](./ConcurrencyModel.md) and [`SessionLifecycle.md`](./SessionLifecycle.md) for lifecycle and eviction.

---

## 2. Action / Observation / State (Pydantic v2)

Schemas mirror the OpenEnv pattern from `OpenEnv/src/openenv/core/env_server/types.py`: every model uses `model_config = ConfigDict(extra="forbid", validate_assignment=True)` so unknown fields get rejected and the JSON schema is judge-stable.

### `PraxisAction`

Single text command parsed by [`server.command_parser`](../../../server/command_parser.py). LLM-friendly grammar — agent emits natural strings.

```python
class PraxisAction(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    command: str  # e.g. "query_logs service=auth timerange=5m"
```

### `PraxisObservation`

Memory-aware observation. After `PraxisMemory.CONTEXT_CUTOFF_STEP` (default 30) the `investigation_result` field flips to a memory-summary view (see [`MemoryModel.md`](./MemoryModel.md)).

```python
class PraxisObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    alert_summary: str
    system_status: dict[str, str]
    investigation_result: str           # memory-aware after cutoff
    available_commands: list[str]       # includes save_finding / recall_memory
    time_elapsed_minutes: float
    severity: str                       # "P0" | "P1" | "P2" | "P3"
    services_affected: list[str]
    step_number: int
    memory_active: bool                 # NEW — true when cutoff reached
    saved_findings_count: int           # NEW — len(memory.saved_findings)
```

### `PraxisState`

```python
class PraxisState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    episode_id: str
    step_count: int
    task_name: str
    incident_resolved: bool = False
    root_cause_identified: bool = False
    cumulative_reward: float = 0.01
    session_id: str = ""                # NEW — links state to session
    memory_active: bool = False         # NEW — mirrors observation
```

---

## 3. Wire formats

### `POST /reset`

Request:

```json
{ "task_name": "cascading-platform-failure", "seed": 42 }
```

- `task_name` optional (default `single-service-alert`); resolved through `TASK_NAME_ALIASES` in `server/praxis_environment.py`.
- `seed` optional; consumed only by `procedural-incident` (deterministic per seed).

Response (200):

```json
{
  "session_id": "8c1c5b2c-3f3a-4d8b-9b41-1cb3d1a44f9e",
  "observation": { ... PraxisObservation ... },
  "alert_summary": "...",
  "system_status": { ... },
  "step_number": 0,
  "memory_active": false,
  "saved_findings_count": 0
}
```

The flat fields are kept for backwards compatibility with judges that look for them at the top level (current behaviour at `server/app.py:194`).

### `POST /step`

Request: `X-Session-Id: <uuid>` header **required**.

```json
{ "command": "save_finding key=root_cause value=db_pool_exhausted" }
```

Response (200):

```json
{
  "observation": { ... PraxisObservation ... },
  "reward": 0.05,
  "done": false,
  "info": { "event": "memory.save_finding.before_cutoff" }
}
```

Errors:

- `400` — missing/unknown `X-Session-Id`, no active session, malformed body.
- `500` — environment crash (must be rare; scenarios return error outcomes, not exceptions).

### `GET /state`

Header: `X-Session-Id: <uuid>` required. Returns the `PraxisState` model above.

### `GET /health`

```json
{ "status": "healthy", "environment": "praxis-env", "version": "1.0.0", "available_tasks": [ ... ] }
```

### `GET /metadata`

Adds the new tasks and the concurrency flag:

```json
{
  "name": "praxis-env",
  "version": "1.0.0",
  "supports_concurrent_sessions": true,
  "themes": ["long-horizon-planning"],
  "tasks": [
    { "name": "single-service-alert", "difficulty": "easy", "max_steps": 15 },
    { "name": "ambiguous-incident", "difficulty": "medium", "max_steps": 25 },
    { "name": "cascading-failure", "difficulty": "hard", "max_steps": 20 },
    { "name": "memory-leak", "difficulty": "hard", "max_steps": 25 },
    {
      "name": "cascading-platform-failure",
      "difficulty": "hard",
      "max_steps": 120
    },
    { "name": "procedural-incident", "difficulty": "medium", "max_steps": 25 }
  ]
}
```

### `GET /benchmark`

Issue #21 (ADR-14). Read-only adapter over `docs/baseline_scores.md`. No session header required. Pydantic response uses `extra="forbid"` (S15).

```json
{
  "environment": "praxis-env",
  "model_scores": [
    {
      "model": "random_baseline",
      "mean_score": 0.021,
      "run_date": "2026-04-25",
      "behaviour": "Random commands; step costs accumulate"
    },
    {
      "model": "Qwen2.5-72B-Instruct (no system prompt)",
      "mean_score": 0.062,
      "run_date": "2026-04-25",
      "behaviour": "Some investigation; weak diagnosis"
    },
    {
      "model": "Qwen2.5-72B-Instruct (SRE system prompt)",
      "mean_score": 0.3,
      "run_date": "2026-04-25",
      "behaviour": "Systematic investigation; correct diagnosis"
    }
  ],
  "note": "Higher scores require multi-hop investigation before diagnosis",
  "source": "docs/baseline_scores.md"
}
```

Missing-file fallback (200, not 404):

```json
{
  "environment": "praxis-env",
  "model_scores": [],
  "note": "Benchmark scores not yet recorded; see docs/baseline_scores.md once Issue #10 ships.",
  "source": "docs/baseline_scores.md"
}
```

---

## 4. Inference stdout contract (unchanged — judges parse this)

```
[START] task=<task_name> env=praxis model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Rules: 2-decimal rewards, lowercase booleans, `error=null` when none. Source: [`idea/PROBLEM STATEMENT/How Judging works.md`](../../PROBLEM%20STATEMENT/How%20Judging%20works.md).

---

## 5. Field-parity rules (every issue cites this)

When a PR touches the wire, its `Acceptance criteria` must include the line **`APIContract: field parity with <schema>`** referencing this file. Field parity means:

- All fields listed here exist on the response with matching names and types.
- No extra fields leak (`extra="forbid"` on the model).
- Memory-aware fields (`memory_active`, `saved_findings_count`) are populated even when the cutoff has not been reached.
- `session_id` is returned exactly once, on `/reset`, as a top-level UUID4 string.

---

## 6. Backwards-compat guarantees

- The flat-field shape on `/reset` (current `**obs_dict` spread at `server/app.py:194`) is preserved; judges that read `alert_summary` / `system_status` from the top level still pass.
- Existing 4 task names continue to work. `POST /step` and `GET /state` now strictly require `X-Session-Id`; missing headers return `400 {"detail":"Missing X-Session-Id header"}`.
- Reward stays in `[0.01, 0.99]` (judge-safe open interval) per `server/reward.py` `clamp_reward()`.
