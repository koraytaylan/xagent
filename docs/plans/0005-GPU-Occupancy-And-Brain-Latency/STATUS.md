# Plan 0005 — GPU Occupancy & Brain-Pass Latency · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned · 0/6 tasks · authored against `develop` @ `3af717b`.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Bank the measured ~10× evolution-throughput win by running at the GPU occupancy knee (default `population_size` 10 → ≈200), produce a definitive on-target per-cooperative-pass cost profile of the fused brain pass, then — gated strictly on that profile — apply one bit-identical latency reduction to the dominant pass that ships a measured speedup or records a measured negative.
- **Outcome:** _Pending._ Expected: a shipped occupancy-sized population (~10× agent-ticks/sec, fixed-seed-validated), a recorded subgroup-path fact + per-pass brain cost table, and either a shipped brain-pass speedup or a recorded measured negative — a concrete result in every branch.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Occupancy throughput (guaranteed win) | `occupancy-sweep-harness`, `population-default-to-occupancy-knee`, `governor-capacity-allocation-validation` | 📋 |
| 0002 | Brain-pass cost profile (guaranteed artifact) | `subgroup-topk-verification`, `per-cooperative-pass-limit-probe` | 📋 |
| 0003 | Brain-pass latency reduction (gated, ship-or-record) | `dominant-pass-latency-reduction` (GATED) | 📋 gated on `0002` profile |

**Design contract (no trial-and-error):** `0001` and `0002` deliver unconditional, measured artifacts (the occupancy win + the per-pass profile). `0003` is bounded and measurement-gated — it ships a bit-identical speedup or records a measured negative in `0003-BRAIN-LATENCY-DECISION.md`. The plan cannot end without a concrete shipped result.
