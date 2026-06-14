# Plan 0002 — CPU/GPU Runtime Decoupling · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete · 12/12 tasks · squash-merged to `develop` as `945b9e7` (`feat(sandbox): decouple simulation from redraw via a GPU worker thread`).
_Last updated: 2026-06-14, against `develop`._

- **Goal:** Move GPU dispatch/readback off the redraw path so the simulation advances on its own cadence and rendering reads published snapshots.
- **Outcome:** A simulation worker thread owns `GpuKernel`; physics state publishes at ≤ 60 Hz and heavy selected-agent telemetry is throttled; ticks/sec is no longer pinned to the frame rate. The 0005 shared-GPU render-path spike was prototyped and **rejected** — see [`0005-SHARED-DEVICE-DECISION.md`](0005-SHARED-DEVICE-DECISION.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Baseline and tick-accounting cleanup | `runtime-decoupling-baseline`, `governor-advance-ticks`, `generation-budget-clamp` | ✅ |
| 0002 | Split compute from publication | `split-dispatch-readback-api` | ✅ |
| 0003 | Rate-limited publication in the current loop | `state-snapshot-rate-limit`, `selected-telemetry-rate-limit`, `main-loop-publication-baseline` | ✅ |
| 0004 | Simulation worker ownership | `sim-runtime-protocol`, `sim-worker-owns-kernel`, `worker-generation-handoff`, `worker-runtime-baseline` | ✅ |
| 0005 | Gated shared-GPU render path | `shared-device-render-spike` | ✅ resolved → rejected (decision doc) |
