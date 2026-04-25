# Reward Policy

## 3. Mega Incident Reward Calibration

Task: `cascading-platform-failure`

Reward rows are defined in `server/reward.py` under
`DEFAULT_REWARD_POLICIES["cascading-platform-failure"]` and merged through
`_with_memory_events(...)` to include cross-task memory events.

Policy intent:

- Encourage broad and relevant triage across three simultaneous fault domains
  (`database`, `cdn`, `worker`).
- Require explicit diagnosis progress across multiple root causes.
- Reward execution of all required remediations.
- Keep random or red-herring actions low-value.
- Apply per-step time pressure for long-horizon discipline.

Issue acceptance alignment:

- Optimal deterministic trajectory scores above `0.55`.
- Wrong-diagnosis path remains at or below `0.10`.
- Remediation events are gated by diagnosis state in scenario logic (ADR-13).
