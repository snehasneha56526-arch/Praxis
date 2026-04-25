# Scenario Catalog

## 3. Mega Incident (Theme 2)

Status: shipped in `praxis_env/scenarios/mega_incident.py`.

- Scenario name: `cascading-platform-failure`
- Severity: `P1` (escalates to `P0` after 70% max-step threshold via `BaseScenario`)
- Max steps: `120`
- Memory cutoff override: `30`
- Initially affected services:
  - `api`
  - `auth`
  - `database`
  - `cache`
  - `worker`
  - `queue`
  - `cdn`
  - `dns`

Root causes modeled in deterministic evidence maps:

1. `db_pool_corrupted`
2. `cdn_tls_expired`
3. `worker_memory_leak`

Resolution criteria:

- Incident resolves when all 3 root causes are diagnosed and all 3 corresponding
  remediations are applied.
- Alternative terminal resolution is evidence-backed escalation, which requires
  at least one correct diagnosis and at least 6 unique investigations.
