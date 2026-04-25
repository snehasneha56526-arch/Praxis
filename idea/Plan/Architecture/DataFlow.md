# Data Flow — End-to-End Praxis Trajectory

> Single-source diagrams that every issue can link to. Mirrors OpenEnv's
> `Environment / Action / Observation / State` contract from
> `OpenEnv/src/openenv/core/env_server/interfaces.py`.

---

## 1. Component map

```mermaid
flowchart LR
    subgraph Client[Client side]
        T[TRL GRPOTrainer<br/>environment_factory]
        I[inference.py]
        H[curl / judge harness]
    end

    subgraph API[FastAPI server: server/app.py]
        R[/reset/]
        S[/step/]
        ST[/state/]
        HE[/health, /metadata, /schema/]
    end

    subgraph Core[Per-session core]
        SM[SessionManager<br/>sessions: dict<br/>+ threading.Lock]
        PE[PraxisEnvironment]
        CP[command_parser]
        MEM[PraxisMemory]
        SC[BaseScenario subclass]
        RE[RewardEngine]
    end

    T --> R
    I --> R
    H --> R
    T --> S
    I --> S
    H --> S
    T --> ST
    H --> HE

    R --> SM
    S --> SM
    ST --> SM

    SM --> PE
    PE --> CP
    PE --> MEM
    PE --> SC
    PE --> RE
    SC --> RE
    MEM --> RE
```

---

## 2. Reset flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as POST /reset
    participant SM as SessionManager
    participant PE as PraxisEnvironment
    participant SC as Scenario
    participant M as PraxisMemory

    C->>API: {task_name, seed?}
    API->>SM: allocate_session(task_name, seed)
    SM->>SM: uuid4() + LRU evict if > 128
    SM->>PE: PraxisEnvironment()
    PE->>SC: get_scenario(task_name)(seed=seed)
    PE->>M: PraxisMemory()
    PE->>SC: reset(episode_id="task_N")
    SC-->>PE: initial state
    PE-->>SM: PraxisObservation(memory_active=false, ...)
    SM-->>API: (session_id, obs)
    API-->>C: {session_id, observation, ...flat}
```

Key invariants:

- `session_id` is a fresh UUID4 returned in the JSON body (not in a header).
- `seed` is forwarded only to scenarios that opt in (`procedural-incident`).
- Memory always resets in lockstep with the scenario.

---

## 3. Step flow (with memory branch)

Status: shipped in Issue #6 (`server/praxis_environment.py` + `server/session_manager.py`).

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as POST /step
    participant SM as SessionManager
    participant PE as PraxisEnvironment
    participant CP as command_parser
    participant M as PraxisMemory
    participant SC as Scenario
    participant RE as RewardEngine

    C->>API: X-Session-Id, {command}
    API->>SM: get(session_id)
    alt missing session
        API-->>C: 400 No active session
    end
    API->>PE: step(PraxisAction(command))
    PE->>CP: parse_command(command)
    CP-->>PE: ParsedCommand(action_type, params)

    alt action_type in {save_finding, recall_memory}
        PE->>M: save / recall
        PE->>RE: score(memory.save_finding.<cutoff_state>) or<br/>score(memory.recall_memory.<cutoff_state>)
        M-->>PE: result_text
    else action_type in {query_logs, check_logs} and step >= cutoff
        PE->>RE: score(memory.illegal_log_after_cutoff)
        PE-->>PE: append [CONTEXT LIMIT] marker to result text
    else scenario action
        PE->>SC: step(parsed)
        SC->>RE: _score_event(event)
        SC-->>PE: StepOutcome
    end

    PE->>M: get_observation_context(step)
    PE-->>API: {observation, reward, done, info}
    API-->>C: 200 JSON
```

Observation rewrite rules (executed by `PraxisEnvironment.step`):

- If `step >= memory.CONTEXT_CUTOFF_STEP`, replace `observation.investigation_result` with `memory.get_observation_context(...)`.
- Always set `observation.memory_active = (step >= cutoff)`.
- Always set `observation.saved_findings_count = len(memory.saved_findings)`.

---

## 4. Reward composition

```mermaid
flowchart TD
    Event[Event tag] --> Lookup[event_values lookup<br/>per task in DEFAULT_REWARD_POLICIES]
    Lookup --> Comp[RewardBreakdown:<br/>investigation/diagnosis/<br/>remediation/escalation +<br/>memory bonus]
    Comp --> Step[time_pressure_cost_per_step *<br/>(step / max_steps)]
    Step --> Sum[Sum components]
    Sum --> Clamp["clamp_reward → [0.01, 0.99]"]
    Clamp --> Out[Returned to scenario / env]
```

Memory events feed into the same engine through new event tags (see [`RewardPolicy.md`](./RewardPolicy.md)):

- `memory.save_finding.before_cutoff` → +0.05
- `memory.save_finding.after_cutoff` → +0.02
- `memory.recall_memory.before_cutoff` → +0.01
- `memory.recall_memory.after_cutoff` → +0.08
- `memory.illegal_log_after_cutoff` → −0.05 (log queries after cutoff)

---

## 5. Episode terminator decision tree

```mermaid
flowchart TD
    Start[Step returns] --> Resolved{incident_resolved?}
    Resolved -- yes --> Done1[done = true]
    Resolved -- no --> Max{step >= MAX_STEPS?}
    Max -- yes --> Done2[done = true, max-steps timeout]
    Max -- no --> Esc{escalation with evidence?}
    Esc -- yes --> Done3[done = true, escalated]
    Esc -- no --> Cont[done = false, continue]
```

Driven by `BaseScenario.is_done()` plus the memory-aware `PraxisEnvironment` wrapper.

---

## 6. Cross-doc links

- Endpoints / schemas: [`APIContract.md`](./APIContract.md)
- Sessions + locking: [`ConcurrencyModel.md`](./ConcurrencyModel.md)
- Memory tool design: [`MemoryModel.md`](./MemoryModel.md)
- Per-task reward maps: [`RewardPolicy.md`](./RewardPolicy.md)
- Scenario inventory: [`ScenarioCatalog.md`](./ScenarioCatalog.md)
- Reset/step/done lifecycle: [`SessionLifecycle.md`](./SessionLifecycle.md)
