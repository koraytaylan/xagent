# XAgent Plan 0003 — Simulation Throughput Ceiling

Lift the high-speed-multiplier tps ceiling: first instrument the per-batch path
to measure whether the limiter is CPU submit, Metal back-pressure, or the
single-workgroup `global` pass; then move `start_tick` to a kernel push constant
and fuse all full kernel-batches in a `dispatch_ticks` call into one command
encoder + one submit; widen the worker dispatch cap so a backlog actually fuses;
trim the two unconditional readback polls and the per-batch world-config
allocation; and, only if the measurement fingers it, gate a `global`-pass
parallelization spike.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md)
for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only (`—` means none).
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green")
  unless the task explicitly says it is documentation-only.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs
  Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Per-batch cost instrumentation

### dispatch-cost-instrumentation — Measure the true per-batch limiter at 1000×

The ≈20 k tps ceiling's cause is unmeasured: the `2026-06-14-grok-43.md` review
blames CPU command-recording, but the single-workgroup `global` pass
(`pass.dispatch_workgroups(1, 1, 1)`, `gpu_kernel.rs:1416-1424`) and Metal submit
back-pressure are at least as plausible, and ≈5 ms/batch is implausible for
recording 4 tiny passes. This task adds default-off probes that isolate
submit-return wall time, GPU-complete wall time, and the `global`+`vision` pass
cost, then records a 1000× baseline. It changes no default behavior.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, add a `DispatchProbe` struct and a
   `GpuKernel` field, read once at construction from the environment (both
   default off):

   ```rust
   /// Default-off per-batch timing / A-B knobs for the throughput-ceiling probe.
   /// Reads happen once at construction so the steady-state path stays branch-light.
   struct DispatchProbe {
       /// `XAGENT_PROBE_GPU_WAIT=1`: add one `Maintain::Wait` after the last
       /// submit to measure GPU-complete wall time vs submit-return wall time.
       wait_for_gpu: bool,
       /// `XAGENT_SKIP_GLOBAL_VISION=1`: skip recording the global+vision passes
       /// to measure residual cost. Produces wrong results; measurement only.
       skip_global_vision: bool,
   }
   ```

2. In `dispatch_ticks` (`gpu_kernel.rs:1354-1460`), bracket the per-batch submit
   loop with `std::time::Instant` and accumulate submit-return nanoseconds into a
   new counter on the kernel; when `self.probe.wait_for_gpu`, issue one
   `self.device.poll(wgpu::Maintain::Wait)` after the final submit and accumulate
   GPU-complete nanoseconds into a second counter. When
   `self.probe.skip_global_vision`, skip recording the `global` and `vision`
   passes (`gpu_kernel.rs:1416-1432`) only.
3. Surface the two counters (totals + per-batch averages) in
   `Worker::maybe_log_counters` (`crates/xagent-sandbox/src/sim_runtime.rs:659-689`)
   beside the existing `sim_ticks` / `dispatch_calls` line.
4. Build a release binary, run at `1000x` for ≥30 s in each arm —
   (a) default, (b) `XAGENT_PROBE_GPU_WAIT=1`, (c) `XAGENT_SKIP_GLOBAL_VISION=1`
   — with `RUST_LOG=debug`, and record submit-return ns/batch, GPU-complete
   ns/batch, and achieved tps per arm in the PR body and in a dated subsection of
   `docs/superpowers/specs/2026-06-10-learning-baseline.md`.
5. State the verdict in that subsection: if tps jumps in arm (c), the `global`
   pass dominates (opens `0004`); if arm (b) shows submit-return ≪ GPU-complete,
   Metal back-pressure dominates; if both are sub-millisecond yet tps stays
   ≈200 batches/sec, CPU recording dominates.

- **Depends on:** —
- **Done when:** the three-arm table and the written verdict exist in the
  baseline spec and PR body; default behavior is unchanged (both env flags off);
  cargo fmt/clippy/test green.

---

## 0002 — Fuse kernel-batches into one submit

### kernel-start-tick-push-constant — Feed the kernel `start_tick` via push constant

The kernel reads its per-batch `start_tick` from the shared world-config uniform
(`let start_tick = wc_u32(WC_TICK);`, `kernel_tick.wgsl:490`), which is the sole
reason `dispatch_ticks` must submit once per batch. Move it to an 8-byte push
constant — exactly how `physics_pipeline` and the `global` pass already take
their tick — so the uniform becomes constant across full batches.

**Steps:**

1. In `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`, add the
   push-constant input near the module globals and replace the `WC_TICK` read at
   line 490:

   ```wgsl
   struct KernelPushConstants { start_tick: u32, _pad: u32, }
   var<push_constant> kpc: KernelPushConstants;
   ```

   then `let start_tick = kpc.start_tick;`. Leave the `WC_VISION_STRIDE` (`:488`)
   and `WC_BRAIN_TICK_STRIDE` (`:489`) reads and the `cycle` loop unchanged.
2. In `crates/xagent-brain/src/gpu_kernel.rs`, build a dedicated kernel pipeline
   layout with a compute push-constant range (clone the `global_layout` shape at
   `gpu_kernel.rs:883-890`) and use it for `kernel_pipeline` instead of
   `brain_layout` (`gpu_kernel.rs:899`):

   ```rust
   push_constant_ranges: &[wgpu::PushConstantRange {
       stages: wgpu::ShaderStages::COMPUTE,
       range: 0..8,
   }],
   ```

   No `max_push_constant_size` bump — it is already 8 and `PUSH_CONSTANTS` is
   already required (`gpu_kernel.rs:463-466`, `:474`).
3. In `dispatch_ticks`, set the kernel push constant on the kernel pass
   (`gpu_kernel.rs:1408-1413`) with `pass.set_push_constants(0,
   bytemuck::cast_slice(&[tick_cursor as u32, 0u32]))`, keeping the per-batch
   uniform write for now (the single-submit fusion lands in the next task). The
   kernel no longer reads `WC_TICK`.

- **Depends on:** `dispatch-cost-instrumentation`
- **Done when:** the kernel reads `start_tick` from the push constant and
  `deterministic_across_batch_sizes` (`integration.rs:629-699`) stays green
  (equivalence preserved — the change is bit-identical for all ticks ≤ 2²⁴);
  cargo fmt/clippy/test green.

### fuse-dispatch-ticks-submits — One encoder + one submit per chunk of full batches

With `start_tick` off the uniform, every full batch in a `dispatch_ticks` call
shares one uniform write, so they can be recorded into a single command encoder
and submitted once — collapsing the per-100-tick submit tax.

**Steps:**

1. In `dispatch_ticks` (`gpu_kernel.rs:1354-1460`), add the fusion bound:

   ```rust
   /// Max full kernel-batches fused into one command buffer. 4 passes/batch keeps
   /// the fused pass count at/under the Metal command-buffer deadlock bound that
   /// CYCLES_PER_CHUNK=100 enforces elsewhere; beyond this, start a new submit.
   const MAX_FUSED_BATCHES: u32 = 24;
   ```

2. Hoist `upload_world_config_with_cycles` (`gpu_kernel.rs:1376`) and
   `create_command_encoder` (`:1383`) above the full-batch loop; write the
   uniform once with `cycles = vision_stride`. Record each full batch's kernel
   (push-constant `start_tick`) + global (existing `gpc` push constant,
   `:1416-1424`) + vision passes into the shared encoder, advancing `tick_cursor`
   per batch; emit one `queue.submit` per chunk of `MAX_FUSED_BATCHES`. `prepare`
   may be hoisted to once per chunk (its `dispatch_args` depend only on
   `agent_count`).
3. Keep the `remainder_cycles` batch (`gpu_kernel.rs:1367-1372`) as its own
   uniform write + encoder + submit (it changes `WC_VISION_STRIDE`), and leave the
   `physics_remainder` block (`:1438-1456`) unchanged. Keep the
   `active_config_index` flip at end of call (`:1458`).
4. Add the equivalence test to `crates/xagent-sandbox/tests/integration.rs`,
   embedding the self-skip guard verbatim, over a non-multiple total that
   exercises both remainder paths:

   ```rust
   #[test]
   fn fused_dispatch_matches_split() {
       if !xagent_brain::GpuKernel::is_available() {
           eprintln!("Skipping: no GPU/fallback adapter available");
           return;
       }
       // One fused dispatch_ticks(0, 1037) must equal the same 1037 ticks run as
       // many small dispatch_ticks calls — asserts byte-equal final physics state.
       // 1037 = 10 full batches (1000) + 3 remainder cycles' ticks + physics rem.
   }
   ```

- **Depends on:** `kernel-start-tick-push-constant`
- **Done when:** full batches in one `dispatch_ticks` call produce one submit per
  ≤24 batches (verify by reading the code path / a submit counter), the new
  `fused_dispatch_matches_split` test passes, and
  `deterministic_across_batch_sizes` stays green; cargo fmt/clippy/test green.

### widen-worker-dispatch-cap — Hand `dispatch_ticks` many batches at high speed

`advance_compute` caps each dispatch at one kernel-batch
(`dispatch_cap = self.kernel.kernel_batch_size().max(min_dispatch)`,
`sim_runtime.rs:530`), so the fusion above never sees more than one batch. Widen
the cap and the accumulator so a real backlog fuses, keeping the budget clamp
upstream.

**Steps:**

1. In `advance_compute` (`crates/xagent-sandbox/src/sim_runtime.rs:518-557`),
   change the cap to allow up to `MAX_FUSED_BATCHES` batches:

   ```rust
   let dispatch_cap = self
       .kernel
       .kernel_batch_size()
       .saturating_mul(MAX_FUSED_BATCHES)
       .max(min_dispatch);
   ```

   Reference the same `MAX_FUSED_BATCHES` value as the kernel (re-export it from
   `xagent_brain` or mirror it with a doc-comment cross-reference; do not
   hardcode a second magic number).
2. Raise `max_accumulator` (`sim_runtime.rs:523-525`) in lockstep so the
   accumulator can fill the wider cap at high multipliers; leave the low-speed
   path (one batch when due) unchanged.
3. Confirm the generation-budget clamp (`clamp_ticks_to_generation_budget`,
   `sim_runtime.rs:538-543`) stays upstream of `dispatch_ticks` (`:549`); do not
   move it.

- **Depends on:** `fuse-dispatch-ticks-submits`
- **Done when:** at 1000× a single `advance_compute` dispatches up to
  `MAX_FUSED_BATCHES` kernel-batches in one submit, and
  `worker_runs_generation_budget_handoff` plus the budget-clamp unit tests
  (`sim_runtime.rs` tests) stay green proving a fused batch still stops exactly at
  `tick_budget`; cargo fmt/clippy/test green.

### fused-throughput-remeasure — Quantify the gain

**Steps:**

1. Re-run the three-arm 1000× measurement from `dispatch-cost-instrumentation`
   on the fused build.
2. Record before/after submit-return ns/batch, GPU-complete ns/batch, submits per
   second, and achieved tps in the PR body and the
   `docs/superpowers/specs/2026-06-10-learning-baseline.md` subsection.
3. State whether the residual ceiling is now the `global` pass (opens `0004`) or
   something else, citing the arm-(c) delta.

- **Depends on:** `widen-worker-dispatch-cap`
- **Done when:** the before/after table and the residual-ceiling verdict are
  recorded; cargo fmt/clippy/test green. (Measurement task — no new behavioral
  unit beyond the green gate.)

---

## 0003 — Per-iteration fixed-cost cleanup

### conditional-readback-polls — Skip the two polls when nothing is in flight

`Worker::step` calls `try_collect_state_snapshot` and `try_collect_telemetry`
every iteration, each issuing `device.poll(wgpu::Maintain::Poll)`
(`gpu_kernel.rs:1544`, `:2136`) even when no readback is outstanding. Skip the
poll when both readback kinds are idle.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, add the predicates:

   ```rust
   /// True while any staging slot has an outstanding readback.
   pub fn has_staging_in_flight(&self) -> bool { self.staging_in_flight.iter().any(|&b| b) }
   /// True while a selected-agent telemetry readback is pending.
   pub fn has_pending_telemetry(&self) -> bool { self.pending_telemetry.is_some() }
   ```

2. In `try_collect_state_snapshot` (`gpu_kernel.rs:1543`) and
   `try_collect_telemetry` (`:2135`), early-return `false`/`None` *before* the
   `device.poll(Poll)` when `!(self.has_staging_in_flight() ||
   self.has_pending_telemetry())` — both polls service the whole device, so gate
   on the union and skip only when both are empty.
3. Leave the unmap-on-all-paths logic in `try_collect_staging`
   (`gpu_kernel.rs:1283-1318`) and the blocking `Maintain::Wait` cold paths
   untouched.

- **Depends on:** —
- **Done when:** the two hot-path polls are skipped on iterations with no readback
  in flight, snapshots/telemetry still collect when in flight (existing tests
  green), and no staging slot is left mapped; cargo fmt/clippy/test green.

### world-config-scratch-buffer — Reuse a 24-float scratch instead of per-batch `Vec`

`build_world_config` allocates `vec![0.0f32; 24]` on every batch
(`crates/xagent-brain/src/buffers.rs:410-447`), called once per batch from
`upload_world_config_*` inside the dispatch loop.

**Steps:**

1. In `buffers.rs`, add `pub fn fill_world_config(out: &mut [f32; WORLD_CONFIG_SIZE], ...same args...)`
   that writes the 24 slots in place; keep `build_world_config` as a thin wrapper
   that fills a fresh `Vec` for any remaining external/test callers.
2. Add a `world_config_scratch: [f32; WORLD_CONFIG_SIZE]` field to `GpuKernel`
   (init near the kernel constructor) and have `upload_world_config_masked`
   (`gpu_kernel.rs:1126`) and `upload_world_config_with_cycles` (`:1146`) — both
   reachable under `&mut self` via `dispatch_ticks` — fill and write the scratch
   instead of allocating. Grep call sites of the `pub` upload methods and adjust
   any to the `&mut self` form.

- **Depends on:** —
- **Done when:** no per-batch heap allocation remains in the `dispatch_ticks`
  uniform path; `deterministic_across_batch_sizes` stays green (identical bytes
  written); cargo fmt/clippy/test green.

---

## 0004 — Gated `global`-pass parallelization

### parallelize-global-pass-spike — Decide whether to parallelize the single-workgroup `global` pass (GATED)

**Gate:** start only after `dispatch-cost-instrumentation` and
`fused-throughput-remeasure` show the single-workgroup `global` pass
(`pass.dispatch_workgroups(1, 1, 1)`, `gpu_kernel.rs:1416-1424`) is the dominant
residual per-batch cost — i.e. tps jumps materially with
`XAGENT_SKIP_GLOBAL_VISION=1`. If the measurement does not finger the `global`
pass, do not start this task; record the negative and close the workstream.

The `global` pass runs `phase_clear` → `phase_food_grid` → `phase_food_respawn`
→ `phase_agent_grid` → 3× collision on one 256-thread workgroup
(`global_tick.wgsl`). The grid-clear and food/agent grid build are embarrassingly
parallel (dispatch `ceil(grid_cells / 256)` workgroups), but the collision passes
carry cross-cell read/write dependencies and atomics that make multi-workgroup
correctness non-trivial.

**Steps:**

1. Prototype a multi-workgroup grid-clear + grid-build only, keeping collision
   single-workgroup, and measure the per-batch GPU-complete delta against the
   `0001`/`0002` baselines.
2. If the grid-build parallelization alone closes most of the residual ceiling,
   write up the construction sketch and a follow-up plan for the collision
   rework; if it does not, or the collision dependencies make safe
   parallelization uneconomical, record that.
3. Resolve the outcome into `0004-GLOBAL-PASS-DECISION.md` (numbered to this
   workstream): the decision up front, the candidate paths with cost/risk,
   measured evidence, and the gate that would reopen it.

- **Depends on:** `dispatch-cost-instrumentation`, `fused-throughput-remeasure`
- **Done when:** either `0004-GLOBAL-PASS-DECISION.md` rejects the path with
  measured evidence (negative result recorded), or a measured prototype justifies
  a follow-up implementation plan; any prototype code left behind keeps cargo
  fmt/clippy/test green or is reverted wholesale.

---

**End of plan 0003 TASKS.** When every "Done when" bullet is green (and `0004` is
resolved one way or the other), the plan's end state is reached.
