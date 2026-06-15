# Plan 0005 — GPU Occupancy & Brain-Pass Latency · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 In progress · `0002` profile recorded + `0003` resolved (measured negative); `0001` on-target sweep/fixed-seed tables still pending the reference GPU · against `develop`.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Bank the measured ~10× evolution-throughput win by running at the GPU occupancy knee (default `population_size` 10 → 192), produce a definitive on-target per-cooperative-pass cost profile of the fused brain pass, then — gated strictly on that profile — apply one bit-identical latency reduction to the dominant pass that ships a measured speedup or records a measured negative.
- **Outcome:** The `0001`+`0002` instrumentation and the occupancy-sized population default (192) are merged and byte-identical/green. The **`0002` per-pass profile is recorded on target** (macOS/Metal): top-K path inactive, and the dominant passes are the dense NN passes (`learn_and_store` > `predict_and_act` > `encode`) — **not** the bitonic top-K that the plan assumed. That falsified `0003`'s candidate #1; the dominant-pass reductions are float-order-locked, so `0003` is **resolved as a measured negative** (no bit-identical micro-fix ships) with the real lever — a cooperative restructure of `predict_and_act` — handed to a follow-up. Still pending an on-target run: `0001`'s occupancy-sweep table and the fixed-seed N=10-vs-192 comparison. Profile + reading: Plan 0005 subsection of `docs/superpowers/specs/2026-06-10-learning-baseline.md`; decision: [`0003-BRAIN-LATENCY-DECISION.md`](0003-BRAIN-LATENCY-DECISION.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Occupancy throughput (guaranteed win) | `occupancy-sweep-harness`, `population-default-to-occupancy-knee`, `governor-capacity-allocation-validation` | 🚧 code merged; sweep table + fixed-seed comparison pending GPU |
| 0002 | Brain-pass cost profile (guaranteed artifact) | `subgroup-topk-verification`, `per-cooperative-pass-limit-probe` | ✅ code merged + on-target subgroup fact & per-pass profile recorded |
| 0003 | Brain-pass latency reduction (gated, ship-or-record) | `dominant-pass-latency-reduction` (GATED) | ✅ resolved — measured negative; cooperative-restructure lever deferred to follow-up |

**Design contract (no trial-and-error):** `0001` and `0002` deliver unconditional, measured artifacts (the occupancy win + the per-pass profile). `0003` is bounded and measurement-gated — it ships a bit-identical speedup or records a measured negative in `0003-BRAIN-LATENCY-DECISION.md`. The plan cannot end without a concrete shipped result.
