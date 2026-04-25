# Decision Log — Architecture Decision Records (ADRs)

> Append-only. Each ADR captures one decision and its alternatives at the moment
> it was made. Never edit a past ADR — supersede it with a new one.

---

## ADR-01 — Pydantic models, not dataclasses (kept)

**Date**: pre-2026-04-25 · **Status**: Accepted · **Source**: `praxis_env/models.py` (S10), OpenEnv `types.py` (S15).

We use Pydantic `BaseModel` for `PraxisAction`, `PraxisObservation`, `PraxisState` because:

- OpenEnv core (`Action`, `Observation` in S15) is Pydantic.
- `model_config = ConfigDict(extra="forbid")` rejects unknown fields, which judges hammer.
- Pydantic generates JSON Schema for the `/schema` endpoint for free.

Alternative considered: dataclasses (cited in legacy `idea/Architecture/implementation_plan.md` ADR-01). Rejected because OpenEnv evolved to Pydantic since the original Praxis scaffold, and `extra="forbid"` is hard to mimic on dataclasses.

---

## ADR-02 — Single text-command action (kept)

**Date**: pre-2026-04-25 · **Status**: Accepted · **Source**: `praxis_env/models.py` (S10), `server/command_parser.py` (S13).

Action is a single `command: str` parsed into structured form. LLMs emit text natively; structured JSON-action would require either function-calling glue or post-hoc parsing anyway.

---

## ADR-03 — Reward clamped to open `[0.01, 0.99]` interval (kept)

**Date**: pre-2026-04-25 · **Status**: Accepted · **Source**: `server/reward.py` `MIN_REWARD/MAX_REWARD` (S12), `BaseScenario.clamp_reward()` (S11).

Closed `[0.0, 1.0]` would let floating-point serialisation round to `0.0` or `1.0` exactly, which our judges check is "not always the same value" (S3 DQ rule). Open interval avoids that boundary while staying numerically equivalent.

This decision is the only deliberate divergence from the spec's `0.0–1.0` (S4); we surface it in [`APIContract.md`](../Architecture/APIContract.md) §6.

---

## ADR-04 — Session-based environment, no module-level singleton (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #1 · **Source**: S9, S14, S16.

Replace `env = PraxisEnvironment()` at `server/app.py:46` with a `SessionManager` (LRU, `max_sessions=128`, `threading.Lock`).

Forces:

- TRL parallel rollouts (`num_generations=8`, S17) work without state corruption.
- OpenEnv runtime validator (S14) accepts `SUPPORTS_CONCURRENT_SESSIONS=true`.
- Multiple judges hitting the HF Space simultaneously don't clobber each other.

Alternatives considered:

- **WebSockets per OpenEnv `WSReset/Step/State`** — rejected: more transport surface to ship by the deadline; HTTP + session header gets the parity we need.
- **Process-per-session via uvicorn `--workers`** — rejected: HF Spaces vCPU=2 budget; process-level isolation eats RAM we need for Qwen2.5-7B inference.

---

## ADR-05 — Memory as explicit tool, not passive summarisation (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #3 · **Source**: S26, S28, S29 (and S22 as the rejected alternative).

Add `save_finding` and `recall_memory` as agent-callable commands. After step 30 the full investigation log is removed from observation; only saved findings remain.

This is the **moat**. The frontier paper (S29) says passive summarisation by the framework fails because the agent never had a chance to express what it considered important. AgeMem (S28) shows GRPO trains naturally on memory-as-tool. Praxis ships exactly this.

Implementation cross-link: `praxis_env/memory.py` (Issue #3). Cutoff behavior details live in [`MemoryModel.md`](../Architecture/MemoryModel.md) §3.

Alternatives considered:

- **SUPO-style automatic summarisation (S22)** — rejected: same failure mode as S29; also would dilute the moat ("nobody else has memory-as-tool").
- **Sliding window observation only** — rejected: doesn't reward foresight; the only signal is "did you survive", not "did you plan".

---

## ADR-06 — Memory cutoff varies per scenario (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issues**: #3, #6 · **Source**: S27, [`MemoryModel.md`](../Architecture/MemoryModel.md) §8.

Default `CONTEXT_CUTOFF_STEP=30`. Scenarios may override via `MEMORY_CUTOFF_OVERRIDE`. Concrete values in [`MemoryModel.md`](../Architecture/MemoryModel.md) §8.

Why per-scenario: short tasks (`single-service-alert`, MAX_STEPS=15) shouldn't ever hit the cutoff; the mega-incident (MAX_STEPS=120) wants the cutoff to fire at ~25%.

---

## ADR-07 — Procedural generator, not scattered-instructions (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #8 · **Source**: `idea/Task.md` Decision 1, S27.

We add `procedural-incident` (seeded, infinite scenarios) and **drop the originally-planned scattered-instructions task (T08)** because:

- T08 was 4 hours alone, novel design — high risk of demo-breaking bugs.
- Procedural generator gives ∞ width AND is a proven pattern.
- Memory adds the depth that scattered-instructions would have added breadth — see `idea/Task.md` line 129.

---

## ADR-08 — Inference-tier evidence first, GRPO curve if credits arrive (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issues**: #10, #12 · **Source**: `idea/Task.md` Decision 2 (lines 132–146), S30.

Build the SRE-prompt vs. random-baseline 3-row score gap table immediately (Issue #10, ~30 min, no GPU). If HF compute credits arrive, run 50 GRPO steps on Qwen2.5-7B for `single-service-alert` and capture the curve (Issue #12, ~45 min on A10G).

The inference table covers ~70% of the 20% rewards judging weight per S30. We prefer the curve, but the table is non-blocking.

---

## ADR-09 — Pitch: lead memory, close process-aware reward (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #20 · **Source**: `idea/Task.md` Decision 3 (lines 147–154).

Pitch order: problem → memory tools → cutoff demo → process-aware reward → "Anthropic could train on this". Memory leads because it's the room's hottest 2026 topic; process-aware reward closes because it's our existing moat.

Alternative (memory-only) — rejected: leaves out 30% of our existing investment in `server/reward.py` shaping.

---

## ADR-10 — Single stacked `github_issues.md`, full Architecture/ tree (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **User decision**: §1 of the executing plan · **Source**: this thread.

20 issues stacked in one file (`idea/Plan/github_issues.md`), and a full architecture set (7 docs) under `idea/Plan/Architecture/`. Per-file issues were considered (and may be re-extracted later via `gh issue create -F …` if the team wants).

---

## ADR-11 — Review policy with auto-PR-review on SDE PRs (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **User decision**: §4 of the plan.

- Architect (Guna) PRs → reviewed by TechLead (Gokul).
- TechLead PRs → reviewed by Architect.
- SDE (Sneha) PRs → reviewed by **both leads** + auto PR review.
- SDE never appears as a reviewer.

Quality > parallelism. Reviewer can hold a lane indefinitely.

---

## ADR-12 — All evidence + plots committed to `docs/` (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issues**: #10, #11, #12, #19 · **Source**: S2 ("Make your plots readable").

Score table → `docs/baseline_scores.md`. Reward curve → `docs/reward_curve.png`. Determinism receipt → `docs/determinism_receipt.txt`. Demo trajectory → `docs/demo_trajectory.txt`. All linked from README.

Reason: judges spend seconds on plots. They must be in the repo, not only in a Colab cell or a deleted Wandb run (per S2 "What makes a submission stand out").

---

## ADR-13 — Evidence gating made explicit across all scenarios (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issues**: #5, #7, #8, #14 · **Source**: critique-thread item #7, S11, S12, [`Architecture/RewardPolicy.md`](../Architecture/RewardPolicy.md).

Every reward policy must return 0 for any `remediation.*` event when the scenario's `_root_cause_identified` flag is False. The legacy 4 scenarios already enforce this implicitly via `BaseScenario`'s `premature_penalty`, but `MegaIncidentScenario` and `ProceduralIncidentScenario` ship with **sparse-reward** policies (per [`Architecture/RewardPolicy.md`](../Architecture/RewardPolicy.md) §3 and §4) where the implicit penalty is too weak. Without this gate, an agent can guess a remediation and bank `remediation.partial = 0.05` (mega) or `remediation.<x> = 0.15-0.25` (procedural) without ever diagnosing.

Why now: with sparse rewards added in this hackathon push, "diagnose-before-remediate" becomes the agent's most-trainable inductive bias for long-horizon planning (the exact Theme #2 capability gap). Codifying it kills a class of reward-hacking trajectories before judges run agentic eval (S3 Phase 2).

Implementation:

- `server/reward.py` (Issue #5): `RewardEngine.score(...)` checks `event.startswith("remediation.")` and returns `RewardResult(reward=clamp_reward(0.0), ...)` when `root_cause_identified=False`. New unit test row per task in `tests/test_reward.py::test_remediation_requires_diagnosis`.
- `tests/test_reward.py` (Issue #5): verifies all default policies include the 6 cross-task memory event tags and includes a 1000-sequence clamp sweep to keep scores within `[0.01, 0.99]`.
- `MegaIncidentScenario.step` (Issue #7) and `ProceduralIncidentScenario.step` (Issue #8): same guard in the scenario layer for defense-in-depth.
- Tests (Issue #14): `test_remediation_before_diagnosis_scores_zero` runs against both new scenarios + 3 procedural difficulties.

Alternatives considered:

- **Rely on `premature_penalty` only** — rejected: penalty is `-0.05`, the new sparse remediation rewards range `+0.05` to `+0.25`, so the net is still positive without the gate.
- **Move the gate into `BaseScenario.step` as a hard pre-check** — rejected for now: legacy scenarios have nuanced semantics (e.g., `cascading-failure` resolves via `kill_query + scale_resource` _and_ implicit `db_connection_pool_exhausted` diagnosis) that we don't want to refactor under hackathon pressure.

---

## ADR-14 — `GET /benchmark` endpoint as a benchmarking signal (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #21 · **Source**: critique-thread item #5, S2 ("What makes a submission stand out"), [`Architecture/APIContract.md`](../Architecture/APIContract.md) §1+§3.

Add a thin read-only `GET /benchmark` endpoint that parses `docs/baseline_scores.md` (produced by Issue #10) and returns `[{model, mean_score, run_date, behaviour}]` plus a fixed explanatory `note`. The response model uses Pydantic `extra="forbid"` per S15.

Why now: SF-winner envs shipped a benchmark surface; Meta engineers browsing the API discover it in the first 30 seconds and immediately understand "this env was designed to be trained against, not just run once". Cost: <50 LOC + 3 tests + 30 minutes. Risk: zero — pure reader, no new state, no concurrency surface.

Implementation:

- `server/app.py` (or new `server/benchmark.py`) reads `docs/baseline_scores.md` once at startup, caches the parsed list for the process lifetime.
- Mounts `GET /benchmark` returning the schema in [`Architecture/APIContract.md`](../Architecture/APIContract.md) §3.
- Missing-file fallback returns 200 with `model_scores: []` and a helpful `note`. We deliberately do not 404; judges should never see a non-200 from this endpoint just because Issue #10's evidence file isn't yet in the repo.

Alternatives considered:

- **Skip the endpoint; rely on `/metadata` + `docs/baseline_scores.md` in the repo** — rejected: judges don't open random `docs/*.md`; an HTTP route is the discovery primitive.
- **Make `/benchmark` write-able (let teams POST scores to it)** — rejected: turns it into stateful infra; not in scope for the hackathon.
- **Embed the scores into `/metadata`** — rejected: pollutes the metadata surface that the OpenEnv runtime validator depends on (S14, S18).

Trade-off accepted: small duplication with `/metadata` (which lists tasks) and `docs/baseline_scores.md` (which is the source of truth). Worth it for the discoverability win.

---

## ADR-15 — Procedural generator shipped with diagnosis gate (NEW)

**Date**: 2026-04-25 · **Status**: Accepted · **Issue**: #8 · **Source**: ADR-07, ADR-13, `tests/test_task6_procedural.py`.

Issue #8 ships the `procedural-incident` scenario from ADR-07 as a seeded deterministic generator with difficulty tiers (`easy`/`medium`/`hard`) and runtime-stamped `MAX_STEPS` + `MEMORY_CUTOFF_OVERRIDE`.

Cross-link to ADR-13: the new scenario enforces a hard evidence gate where remediation attempts before diagnosis score zero (clamped at the shared floor in emitted rewards), with explicit tests across all three procedural difficulties.

---

## How to add a new ADR

1. Append at the bottom with the next number.
2. Include: Date, Status (`Accepted` / `Rejected` / `Superseded by ADR-NN`), Issue/PR, Source IDs from [`EvidenceIndex.md`](./EvidenceIndex.md).
3. Capture **decision**, **why now**, **alternatives**, and the **trade-off accepted**.
4. Never edit a past ADR — supersede.
