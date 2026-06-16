# Plan 0007 - Learning Control Grounding - status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** Planned. Authored from the 2026-06-15 `xagent.db` investigation and
code audit; no implementation tasks have landed yet.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Make the live GPU runtime evaluate the genes persisted in
  `xagent.db`, add behavior evidence that can prove or falsify food chasing and
  danger avoidance, slow the perception-action loop through a learning
  curriculum, repair multiplier-only klinotaxis, and require red-green probes
  before interpreting a long evolution run.
- **Measured baseline:** `xagent.db` has 1 run, 251 nodes, 2500 agent results,
  250 recordings; status distribution is 249 failed descendants, 1 exhausted
  root, 1 active node. Root score remains best (`0.0042712856`). Late
  generations 225-249 average `0.0014031432` fitness, 262.096 food, 324.42
  deaths, and 174.592 cells. Decoded recordings show mean straightness
  `0.007105`, mean absolute turn `0.375534`, turn-bias ratio `0.851702`, and
  turn-sign persistence `0.940761`; mean absolute turn correlates with deaths
  (`r=0.9107`) and against fitness (`r=-0.6596`).
- **Outcome:** Planned; no code outcome yet.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Runtime genome authority | `effective-agent-config-upload`, `mutation-provenance-for-effective-genes` | Planned |
| 0002 | Behavioral evidence telemetry | `recording-format-v2`, `navigation-telemetry-slots`, `behavior-metric-table` | Planned |
| 0003 | Control-rate curriculum | `learning-curriculum-preset`, `movement-speed-range-revisit` | Planned |
| 0004 | Turn-attractor and klinotaxis repair | `authoritative-turn-persistence`, `sign-breaking-klinotaxis` | Planned |
| 0005 | Food/danger emergence gates | `food-closure-probe`, `danger-exit-probe`, `short-evolution-gate` | Planned |

## Verification

Documentation-only authoring so far. Code gates have not been run for this plan
yet.
