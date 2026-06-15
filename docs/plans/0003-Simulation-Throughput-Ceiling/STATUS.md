# Plan 0003 — Simulation Throughput Ceiling · status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 🚧 In progress · 8/8 implemented (bit-identical-verified) · authored against `develop` @ `2a751a8`.
_Last updated: 2026-06-14, against branch `claude/determined-lamport-qtmx6y`._

- **Goal:** Lift the high-speed-multiplier tps ceiling (≈20 k tps at 1000× instead of the proportional ≈60 k) by measuring the true per-batch limiter, then fusing all full kernel-batches in a `dispatch_ticks` call into one command encoder + one submit — bit-identical to today.
- **Outcome:** The fused single-submit path is implemented and **bit-identical** — `deterministic_across_batch_sizes` and the new `fused_dispatch_matches_split` (1037-tick non-multiple, both remainder paths) pass on Mesa lavapipe. The per-batch submit tax is removed: one `dispatch_ticks(0, 24000)` now records **10 submits** for **240** kernel-batches (⌈240/24⌉) instead of 240. Conditional readback polls + reused world-config scratch landed. **Pending:** the authoritative three-arm 1000× throughput table on macOS/Metal (this environment has only lavapipe, a CPU rasterizer — non-representative; repro + verdict recorded in the baseline spec), and merge to `develop`. The `global`-pass rewrite (0004) is **closed, not opened** — gate pending the on-target measurement (`0004-GLOBAL-PASS-DECISION.md`).

| WS | Workstream | Task | State |
|---|---|---|---|
| 0001 | Per-batch cost instrumentation | `dispatch-cost-instrumentation` | ✅ code + probe wired (`[SIM-PROBE]` / `[BENCH-PROBE]`); lavapipe three-arm table + verdict recorded; **on-target table pending** (repro in baseline spec) |
| 0002 | Fuse kernel-batches into one submit | `kernel-start-tick-push-constant` | ✅ kernel reads `start_tick` from push constant; determinism green |
| 0002 | | `fuse-dispatch-ticks-submits` | ✅ one submit per ≤24 full batches; `fused_dispatch_matches_split` green (240 batches → 10 submits) |
| 0002 | | `widen-worker-dispatch-cap` | ✅ cap = `kernel_batch_size × MAX_FUSED_BATCHES`; `worker_runs_generation_budget_handoff` green (budget clamp stays upstream) |
| 0002 | | `fused-throughput-remeasure` | ✅ lavapipe before/after fusion ratio recorded; **on-target before/after pending** |
| 0003 | Per-iteration fixed-cost cleanup | `conditional-readback-polls` | ✅ both hot-path polls skipped when staging+telemetry idle; tests green |
| 0003 | | `world-config-scratch-buffer` | ✅ `fill_world_config` reuses a `[f32; 24]` scratch; no per-batch heap alloc; determinism green |
| 0004 | Gated `global`-pass parallelization | `parallelize-global-pass-spike` (GATED) | ✅ resolved — **closed, not opened**; gate unevaluated on target HW (`0004-GLOBAL-PASS-DECISION.md`) |

**Quality gate (Mesa lavapipe):** `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test -p xagent-sandbox` (83 lib + 13 bin + 62 integration) all green; `cargo test -p xagent-brain` (50) green.
