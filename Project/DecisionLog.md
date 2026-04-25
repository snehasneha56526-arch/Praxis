# Decision Log

## ADR-13: Remediation Gate for Mega Incident

Date: 2026-04-25

Context:

- The mega incident includes three simultaneous root causes and sparse evidence.
- Un-gated remediation commands let agents collect reward before building
  grounded diagnosis, reducing the intended reasoning difficulty.

Decision:

- In `MegaIncidentScenario`, remediation events are scored through the reward
  engine with explicit `root_cause_identified` state.
- When no root cause has been diagnosed yet, remediation events score zero
  before clamp (emitted reward floor remains `0.01`).

Consequences:

- Agent behavior is forced toward investigation and diagnosis before acting.
- The acceptance test
  `tests/test_task5_mega_incident.py::test_remediation_before_diagnosis_scores_zero`
  verifies the gate.
- This ADR is specific to the mega scenario implementation and keeps existing
  scenario behavior unchanged.
