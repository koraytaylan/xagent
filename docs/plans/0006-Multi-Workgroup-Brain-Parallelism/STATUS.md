# Plan 0006 - Multi-Workgroup Brain Parallelism - status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned · 0/10 tasks · authored against `develop` @ `069ff8a`.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Reach the 60 k raw-tps target by first proving the N=10 no-brain floor, then applying same-dispatch dense tiling, then using split-cycle multi-workgroup brain phases, action-tail reductions, memory reinforcement tiling, and food-grid floor recovery as needed.
- **Outcome:** Planned. The plan cannot stop at "right attack path": it ships `ParallelTiled` as the default only if it reaches >=60,000 tps at N=10 with correctness and learning gates green; otherwise `0006-60K-CLOSURE.md` must name the single remaining measured owner after every known heavy loop has been addressed.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Throughput budget and no-go map | `n10-throughput-budget-baseline` | 📋 |
| 0002 | Same-dispatch cooperative tiling | `same-dispatch-dense-tiling` | 📋 |
| 0003 | Split-cycle execution scaffold | `split-serial-cycle-scaffold` | 📋 |
| 0004 | Multi-workgroup dense brain phases | `scratch-buffer-and-feature-phase`, `multi-workgroup-encode-and-credit`, `multi-workgroup-predictor-and-action`, `parallel-reduce-action-tail`, `multi-workgroup-memory-reinforcement` | 📋 |
| 0005 | Non-brain floor recovery | `fused-food-grid-detect-floor-recovery` | 📋 |
| 0006 | 60 k closure | `sixty-k-throughput-closure` | 📋 |

**Design contract:** Default fused behavior stays unchanged until the final gate. Every branch either ships measured speed or records a measured negative; no open-ended optimization work is left without a decision.
