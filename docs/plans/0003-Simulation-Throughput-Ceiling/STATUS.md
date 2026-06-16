# Plan 0003 — Simulation Throughput Ceiling · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete · 8/8 · merged to `develop`; 0004 rejected with on-target evidence.
_Last updated: 2026-06-15, against `develop`._

- **Goal:** Lift the high-speed-multiplier tps ceiling (≈20 k tps at 1000× instead of the proportional ≈60 k) by measuring the true per-batch limiter, then fusing all full kernel-batches in a `dispatch_ticks` call into one command encoder + one submit — bit-identical to today.
- **Outcome:** Fused single-submit path shipped and **bit-identical** (`deterministic_across_batch_sizes` + `fused_dispatch_matches_split` green); per-batch submit tax removed (10 000 batches → 417 submits on target). **But the on-target `--bench-phase-ab` run adjudicated the ceiling and ruled out both 0002's and 0004's targets:** fusion is engaged yet baseline tps ≈23 k still sits at the original ceiling (CPU-submit attribution falsified), and the `global` pass is only ≈3% (skip-global +3%, skip-global+vision +7%). **≥93% of per-batch time is the fused kernel/brain pass** — agent-count-independent → latency-bound, GPU-under-occupied serial brain chain. Plan 0003's deltas are all merged; the remaining throughput work (kernel/brain-pass latency/occupancy) is a **new plan**, out of scope here. Per-iteration cleanups (conditional polls, world-config scratch) landed. Instrumentation (probe + independent `XAGENT_SKIP_GLOBAL`/`XAGENT_SKIP_VISION` knobs + `--bench-phase-ab` harness) shipped.

| WS | Workstream | Task | State |
|---|---|---|---|
| 0001 | Per-batch cost instrumentation | `dispatch-cost-instrumentation` | ✅ probe wired (`[SIM-PROBE]` / `[BENCH-PROBE]`); independent `XAGENT_SKIP_GLOBAL`/`XAGENT_SKIP_VISION` knobs + `--bench-phase-ab`; on-target table recorded in baseline spec |
| 0002 | Fuse kernel-batches into one submit | `kernel-start-tick-push-constant` | ✅ kernel reads `start_tick` from push constant; determinism green |
| 0002 | | `fuse-dispatch-ticks-submits` | ✅ one submit per ≤24 full batches; `fused_dispatch_matches_split` green |
| 0002 | | `widen-worker-dispatch-cap` | ✅ cap = `kernel_batch_size × MAX_FUSED_BATCHES`; `worker_runs_generation_budget_handoff` green |
| 0002 | | `fused-throughput-remeasure` | ✅ on-target: fusion engaged (10 000 batches → 417 submits) but tps ≈23 k unchanged → submit overhead was **not** the limiter |
| 0003 | Per-iteration fixed-cost cleanup | `conditional-readback-polls` | ✅ both hot-path polls skipped when staging+telemetry idle; tests green |
| 0003 | | `world-config-scratch-buffer` | ✅ `fill_world_config` reuses a `[f32; 24]` scratch; no per-batch heap alloc; determinism green |
| 0004 | Gated `global`-pass parallelization | `parallelize-global-pass-spike` (GATED) | ✅ resolved — **REJECTED with on-target evidence** (skip-global +3%; global pass ≈3% of cost) (`0004-GLOBAL-PASS-DECISION.md`) |

**Quality gate:** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` (verified on CI's Rust 1.96.0), and `cargo test -p xagent-sandbox` (83 lib + 13 bin + 62 integration) all green; `cargo test -p xagent-brain` (50) green.

**Follow-up surfaced (new plan):** the throughput ceiling is the fused **kernel/brain pass** (≥93% of per-batch time, agent-count-independent). Not addressable by submit fusion or `global`-pass parallelization — needs a brain-pass latency/occupancy plan.
