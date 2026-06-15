# Plan 0005 — GPU Occupancy & Brain-Pass Latency · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 In progress · 5/6 code-complete, on-target numbers + gated `0003` pending the reference GPU · against `develop`.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Bank the measured ~10× evolution-throughput win by running at the GPU occupancy knee (default `population_size` 10 → 192), produce a definitive on-target per-cooperative-pass cost profile of the fused brain pass, then — gated strictly on that profile — apply one bit-identical latency reduction to the dominant pass that ships a measured speedup or records a measured negative.
- **Outcome:** _Code/wiring merged; on-target measurements pending._ The `0001`+`0002` instrumentation and the occupancy-sized population default (192) are landed and byte-identical/green (`fmt`/`clippy -D warnings`/`test`). The on-target tables (sweep, subgroup-path fact, per-pass profile, fixed-seed evolution) are **pending a run on the macOS/Metal reference GPU** — this branch was developed in a Linux container with no GPU adapter (`GpuKernel::is_available()` == false), the split the plan anticipates ("harness wiring verified on lavapipe; throughput numbers require the target GPU"). `0003` is correctly **gated/open** on that profile (no speculative change). Repro commands + placeholder tables: Plan 0005 subsection of `docs/superpowers/specs/2026-06-10-learning-baseline.md`; gate state: [`0003-BRAIN-LATENCY-DECISION.md`](0003-BRAIN-LATENCY-DECISION.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Occupancy throughput (guaranteed win) | `occupancy-sweep-harness`, `population-default-to-occupancy-knee`, `governor-capacity-allocation-validation` | 🚧 code merged; sweep table + fixed-seed comparison pending GPU |
| 0002 | Brain-pass cost profile (guaranteed artifact) | `subgroup-topk-verification`, `per-cooperative-pass-limit-probe` | 🚧 code merged (byte-identical when unset); on-target subgroup fact + per-pass table pending GPU |
| 0003 | Brain-pass latency reduction (gated, ship-or-record) | `dominant-pass-latency-reduction` (GATED) | ⛔ gated/open on `0002` on-target profile |

**Design contract (no trial-and-error):** `0001` and `0002` deliver unconditional, measured artifacts (the occupancy win + the per-pass profile). `0003` is bounded and measurement-gated — it ships a bit-identical speedup or records a measured negative in `0003-BRAIN-LATENCY-DECISION.md`. The plan cannot end without a concrete shipped result.
