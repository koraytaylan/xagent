# Plan 0021 — Effort-Danger-Lever-Graduation — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete.
_Last updated: 2026-06-27, against `develop`._

- **Goal:** Three gated levers have recorded flip-or-retire decisions with
  measured 95% CI evidence and documented unlock conditions: effort-fitness (run
  speed-decoupling A/B, decide flip/retire/defer), danger-percept (run
  intent-impact A/B, decide flip/retire/defer), innate-instincts (mark decision
  terminal or gated-on-credit-path). No defaults flipped; flags remain
  default-off. All decisions recorded in decision docs or STATUS updates.
- **Root cause:** The 0009/0010/0013/0014 plans built and hardened three gated
  levers, but left them in limbo without recorded graduation decisions or
  explicit unlock conditions. The harnesses exist and are proven; this plan runs
  them on production seeds and records the verdicts so the board reflects the true
  status (KEEP OFF, FLIP, or DEFER WITH GATE).
- **Approach:** Measurement-first: probe and baseline tasks run production A/Bs
  with bootstrap 95% CI, then decision docs record the evidence and verdict. No
  code integration or default flips; pure measurement + documentation.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Effort-Fitness-Production-A-B | `effort-fitness-production-a-b` | ✅ Done |
| 0002 | Danger-Percept-Production-A-B | `danger-percept-production-a-b` | ✅ Done |
| 0003 | Innate-Instincts-Graduation-Status | `innate-instincts-terminal-or-gated-mark` | ✅ Done |
