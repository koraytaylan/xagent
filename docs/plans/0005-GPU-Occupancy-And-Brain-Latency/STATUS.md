# Plan 0005 — GPU Occupancy & Brain-Pass Latency · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 In progress · all three workstreams measured on target; the headline occupancy→evolution win did not hold (recorded), `0002`/`0003` done · against `develop`.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Bank the measured ~10× evolution-throughput win by running at the GPU occupancy knee, produce a definitive on-target per-cooperative-pass cost profile of the fused brain pass, then — gated strictly on that profile — apply one bit-identical latency reduction to the dominant pass that ships a measured speedup or records a measured negative.
- **Outcome:** The `0001`+`0002` instrumentation is merged and byte-identical/green. **The occupancy sweep is recorded** (knee N=200, 2.40 M agent-ticks/sec, ~10× the GPU throughput of N=10) — but the fixed-seed evolution validation **falsified the assumption that this converts to an evolution win**: raising the population in the shared world regresses hard (food pinned at the supply cap, deaths-per-food ~11× worse); enlarging the world (area ∝ N) removes the competition (per-capita metrics match N=10) but yields **no fitness gain** because the limiter is learner strength, not search breadth. So the default `population_size` is **kept at 10** (occupancy is banked as GPU-throughput infrastructure, not a population change). The **`0002` per-pass profile is recorded**: top-K path inactive, the cost is the dense NN passes (`learn_and_store` > `predict_and_act` > `encode`), **not** the bitonic top-K the plan assumed — which falsified `0003`'s candidate #1; the dominant-pass reductions are float-order-locked, so `0003` is **resolved as a measured negative**, the cooperative-restructure lever handed to a follow-up. Full readings: Plan 0005 subsection of `docs/superpowers/specs/2026-06-10-learning-baseline.md`; decision: [`0003-BRAIN-LATENCY-DECISION.md`](0003-BRAIN-LATENCY-DECISION.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Occupancy throughput | `occupancy-sweep-harness`, `population-default-to-occupancy-knee`, `governor-capacity-allocation-validation` | ⚠️ sweep recorded (knee N=200); validation found population-raise regresses evolution (shared-world competition; scaling fixes competition but no benefit — learner-limited) → **default kept at 10** |
| 0002 | Brain-pass cost profile (guaranteed artifact) | `subgroup-topk-verification`, `per-cooperative-pass-limit-probe` | ✅ code merged + on-target subgroup fact & per-pass profile recorded |
| 0003 | Brain-pass latency reduction (gated, ship-or-record) | `dominant-pass-latency-reduction` (GATED) | ✅ resolved — measured negative; cooperative-restructure lever deferred to follow-up |

**Design contract (no trial-and-error):** `0001` and `0002` deliver unconditional, measured artifacts (the occupancy win + the per-pass profile). `0003` is bounded and measurement-gated — it ships a bit-identical speedup or records a measured negative in `0003-BRAIN-LATENCY-DECISION.md`. The plan cannot end without a concrete shipped result.
