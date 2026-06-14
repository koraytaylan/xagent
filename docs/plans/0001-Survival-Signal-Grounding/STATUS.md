# Plan 0001 — Survival Signal Grounding · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete · 10/10 tasks · all four workstreams landed on `develop` (per-task commits `e123556` … `43f55cf`).
_Last updated: 2026-06-14, against `develop`._

- **Goal:** Ground the survival reward in space so steering can be learned (diagnosed root cause: spatially-blind reward).
- **Outcome:** Fitness reworked to a multiplicative survival gate (Variant B), cutting deaths-per-food 4.02 → 2.23 (−45%); the three-arm lag sweep confirmed `lag100` as the default (higher strides cost more ticks/sec than the learning gain is worth); hazard grounding re-measured and recorded.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Hazard observability and baseline | `sensory-tail-telemetry`, `hazard-probe-baseline`, `terminal-death-update`, `hazard-edge-touch` | ✅ |
| 0002 | Survival-signal grounding | `same-cycle-interoception`, `hazard-grounding-remeasure` | ✅ |
| 0003 | Learning visibility and lag economics | `quarter-learning-metric`, `stride-lag-sweep` | ✅ |
| 0004 | Gated follow-ups | `fitness-rework` (GATED), `predictor-forward-objective` (GATED) | ✅ both gates passed |
