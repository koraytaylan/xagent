# Plan 0003 — Simulation Throughput Ceiling · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned · 0/8 tasks · authored against `develop` @ `2a751a8`.
_Last updated: 2026-06-14, against `develop`._

- **Goal:** Lift the high-speed-multiplier tps ceiling (≈20 k tps at 1000× instead of the proportional ≈60 k) by measuring the true per-batch limiter, then fusing all full kernel-batches in a `dispatch_ticks` call into one command encoder + one submit — bit-identical to today.
- **Outcome:** _Pending._ Expected: a measured verdict on submit vs `global`-pass dominance, a fused single-submit dispatch path with a quantified tps gain, and either a `global`-pass parallelization follow-up or a recorded negative.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Per-batch cost instrumentation | `dispatch-cost-instrumentation` | 📋 |
| 0002 | Fuse kernel-batches into one submit | `kernel-start-tick-push-constant`, `fuse-dispatch-ticks-submits`, `widen-worker-dispatch-cap`, `fused-throughput-remeasure` | 📋 |
| 0003 | Per-iteration fixed-cost cleanup | `conditional-readback-polls`, `world-config-scratch-buffer` | 📋 |
| 0004 | Gated `global`-pass parallelization | `parallelize-global-pass-spike` (GATED) | 📋 gated on `0001` measurement |
