# Plan status

Roll-up board for `docs/plans/` — **one row per plan, no per-task detail**. Task-level status lives in each plan's own `STATUS.md` (linked in the last column). This board exists to answer "what is done, in flight, or blocked?" across all plans at a glance, and stays small as plan count grows.

**It is only useful if it is accurate.** Update the relevant row in the *same change* that moves a plan, and keep it in sync with that plan's `STATUS.md` (see [README → STATUS.md](README.md#statusmd)).

_Last updated: 2026-06-15, against `develop`._

| Plan | Title | Status | Tasks | Outcome | Detail |
|---|---|---|---|---|---|
| 0001 | Survival Signal Grounding | ✅ Complete | 10/10 | Deaths-per-food 4.02 → 2.23 (−45%); `lag100` confirmed default. | [status](0001-Survival-Signal-Grounding/STATUS.md) |
| 0002 | CPU/GPU Runtime Decoupling | ✅ Complete | 12/12 | Sim decoupled from redraw onto a GPU worker thread; shared-GPU render path rejected. | [status](0002-CPU-GPU-Runtime-Decoupling/STATUS.md) |
| 0003 | Simulation Throughput Ceiling | ✅ Complete | 8/8 | Fused single-submit dispatch landed & **bit-identical**; submit tax removed (10 000 batches → 417 submits). On-target A/B then **adjudicated the ceiling**: fusion engaged but tps ≈23 k unchanged, `global` pass only ≈3% → 0004 **rejected with evidence**; the real limiter is the fused **kernel/brain pass** (≥93%, agent-count-independent) → new plan. | [status](0003-Simulation-Throughput-Ceiling/STATUS.md) |
| 0004 | Approach Reward Shaping | 📋 Planned | 0/14 | Add potential-based approach shaping to `raw_gradient` so steering becomes learnable (the spatially-blind-reward root cause); restore the selection signal; reactive/heritable/vision follow-ups gated on the remeasure that proves the unlock. | [status](0004-Approach-Reward-Shaping/STATUS.md) |
| 0005 | GPU Occupancy & Brain-Pass Latency | 🚧 In progress | `0002`/`0003` done, `0001` numbers pending | `0001`+`0002` wiring merged & byte-identical/green: `--bench-agent-sweep`, occupancy-sized population default (10 → **192**), `GpuKernel::has_subgroup()` + top-K-path log, `XAGENT_KERNEL_PASS_LIMIT` probe. **On-target `0002` profile recorded**: top-K path inactive and the cost is the dense NN passes (`learn_and_store` > `predict_and_act` > `encode`), **not** the bitonic top-K the plan assumed — so `0003` is **resolved as a measured negative** (reductions are float-order-locked; cooperative-restructure lever deferred to a follow-up). Still pending the reference GPU: `0001`'s occupancy-sweep + fixed-seed N=10-vs-192 tables. | [status](0005-GPU-Occupancy-And-Brain-Latency/STATUS.md) |

**Status legend:** 📋 Planned (authored, not started) · 🚧 In progress (some tasks merged) · ✅ Complete (all tasks merged to `develop`) · ⛔ Blocked (waiting on a gate or upstream task) · 🗄️ Superseded (replaced or abandoned).
