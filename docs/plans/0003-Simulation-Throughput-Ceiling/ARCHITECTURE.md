# Architecture — Plan 0003 (deltas)

> Edits center on `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-sandbox/src/sim_runtime.rs`, and integration tests in
> `crates/xagent-sandbox/tests/integration.rs`. The gated workstream 0004 may
> additionally touch `crates/xagent-brain/src/shaders/kernel/global_tick.wgsl`
> and its sub-passes. Line numbers are hints; locate by symbol.

## 0001 — Per-batch cost instrumentation

Today `dispatch_ticks` (`gpu_kernel.rs:1354-1460`) records and submits one
command buffer per kernel-batch with no timing, and the worker only logs
coarse counters in `maybe_log_counters` (`sim_runtime.rs:659-689`). There is no
way to tell whether the ≈200 submits/sec ceiling is CPU submit/recording, Metal
command-buffer back-pressure, or `global`-pass GPU execution. This workstream
adds two default-off probes that disambiguate, then records a baseline.

Edits:

- **Submit-vs-complete wall timer** (`gpu_kernel.rs`, inside `dispatch_ticks`):
  bracket the per-batch submit loop with `std::time::Instant` and accumulate the
  nanoseconds *between recording start and `queue.submit` return* into a new
  counter. Separately, when an env flag is set, issue one
  `device.poll(wgpu::Maintain::Wait)` after the final submit and record the
  *GPU-complete* wall time. The gap between "submit returned" and "GPU complete"
  distinguishes CPU/recording cost from queue back-pressure.

- **`global`+`vision` A/B skip** (`gpu_kernel.rs`): a construction-time flag read
  once from the environment, defaulting off, that conditionally skips *recording*
  the `global` and `vision` passes in `dispatch_ticks`. Running 1000× with it on
  vs off isolates the `global`-pass GPU cost. It is a measurement-only knob — it
  produces incorrect simulation while set and is never on in tests or release.

```rust
/// Default-off per-batch timing/A-B knobs for the throughput-ceiling probe.
/// All fields default to the non-instrumented behavior; reads happen once at
/// construction so the steady-state path stays branch-light.
struct DispatchProbe {
    /// `XAGENT_PROBE_GPU_WAIT=1` adds one `Maintain::Wait` after the last submit
    /// to measure GPU-complete wall time vs submit-return wall time.
    wait_for_gpu: bool,
    /// `XAGENT_SKIP_GLOBAL_VISION=1` skips recording the global+vision passes so
    /// the residual submit/kernel cost can be measured. Produces wrong results;
    /// measurement only.
    skip_global_vision: bool,
}
```

- **Counter surfacing** (`sim_runtime.rs:659-689`): extend `maybe_log_counters`
  to print accumulated submit-return nanos, GPU-complete nanos (when enabled),
  and the per-batch averages alongside the existing `sim_ticks` / `dispatch_calls`.

Properties that make this safe:
- Both knobs default off; the default dispatch path is byte-for-byte unchanged
  and all existing tests pass untouched.
- The wall timer is `Instant`-only (no GPU feature, no `TIMESTAMP_QUERY`), so it
  compiles and runs on Mesa lavapipe in CI without an adapter-specific feature.
- `skip_global_vision` is documented as result-corrupting and gated behind an env
  flag, never a default or a test path.

## 0002 — Fuse kernel-batches into one submit

Today `dispatch_ticks` (`gpu_kernel.rs:1354-1460`) decomposes `ticks_to_run`
into `kernel_batches` full batches + a `remainder_cycles` batch +
a `physics_remainder`, and emits one encoder + one `queue.submit` **per full
batch** (`gpu_kernel.rs:1366-1434`) because each batch writes a distinct
`start_tick` into the shared `world_config` uniform and the GPU must consume the
old `WC_TICK` before the next `write_buffer`. The worker never even hands it more
than one batch: `dispatch_cap = kernel.kernel_batch_size()` (`sim_runtime.rs:530`).

The fix removes the only per-batch-varying uniform read so all full batches share
one uniform write and one submit, then widens the cap so a real backlog fuses.

### Move `start_tick` to a kernel push constant

Today the fused kernel reads `let start_tick = wc_u32(WC_TICK);`
(`kernel_tick.wgsl:490`) and `base_tick = start_tick + cycle * stride`
(`kernel_tick.wgsl:493`) — the value feeds only `agent_physics` and
`agent_death_respawn` RNG seeds and `P_LAST_DEATH_TICK`. `WC_VISION_STRIDE`
(`:488`) and `WC_BRAIN_TICK_STRIDE` (`:489`) are constant across full batches.
The `global` pass already takes its tick via the `gpc` push constant
(`gpu_kernel.rs:1416-1424`, `global_tick.wgsl`), and `physics_pipeline` already
uses push constants — so the kernel is the only pass still tied to the uniform.

Edits:

- **WGSL push-constant input** (`kernel_tick.wgsl`, near the other module
  globals; replace the `WC_TICK` read at `:490`):

```wgsl
// start_tick arrives per-batch via push constant so multiple kernel-batches
// can share ONE world_config uniform write and ONE submit. Exact u32 — strictly
// more precise than the former WC_TICK = (tick as f32) round-trip.
struct KernelPushConstants { start_tick: u32, _pad: u32, }
var<push_constant> kpc: KernelPushConstants;
```

- **Dedicated kernel pipeline layout** (`gpu_kernel.rs:897-904`): the kernel
  pipeline currently reuses `brain_layout` (no push constants, `:899`). Give it
  its own layout with an 8-byte compute push-constant range, cloning the existing
  `global_layout` shape (`gpu_kernel.rs:883-890`). No `max_push_constant_size`
  bump — `PUSH_CONSTANTS` is already required and the limit is already 8
  (`gpu_kernel.rs:463-466`, `:474`).

```rust
// Mirrors physics_layout/global_layout: COMPUTE push constants, range 0..8.
push_constant_ranges: &[wgpu::PushConstantRange {
    stages: wgpu::ShaderStages::COMPUTE,
    range: 0..8,
}],
```

### Single-encoder fused dispatch

Edits (`dispatch_ticks`, `gpu_kernel.rs:1354-1460`):

- **Upload the world-config uniform once** before the full-batch loop
  (`vision_stride` / `brain_tick_stride` / phase identical for every full batch);
  the kernel no longer reads `WC_TICK`.
- **Set `start_tick` per batch via push constant** on the kernel pass
  (`gpu_kernel.rs:1408-1413`); keep the existing per-batch `gpc` push constant on
  the `global` pass (`:1416-1424`).
- **One encoder + one `queue.submit` per chunk** of full batches, bounded by the
  same Metal command-buffer guard the masked path already uses
  (`CYCLES_PER_CHUNK = 100`, `gpu_kernel.rs:1186`). Each full batch records
  kernel + global + vision; `prepare` may be hoisted to once-per-chunk because
  its `dispatch_args` depend only on `agent_count`.

```rust
/// Max full kernel-batches fused into one command buffer. 4 passes/batch keeps
/// the fused pass count at/under the Metal command-buffer deadlock bound that
/// CYCLES_PER_CHUNK=100 enforces elsewhere; beyond this, start a new submit.
const MAX_FUSED_BATCHES: u32 = 24;
```

- **The `remainder_cycles` batch keeps its own uniform write + submit.** It sets
  `cycles_this_batch = remainder_cycles`, which changes `WC_VISION_STRIDE` in the
  uniform, so it cannot share the full-batch group's single write
  (`gpu_kernel.rs:1367-1372`). The `physics_remainder` block
  (`gpu_kernel.rs:1438-1456`) is already its own phase/submit and is unchanged.
  At the default 100/1000-tick multiples `remainder_cycles == 0`, so neither
  fires.

### Widen the worker dispatch cap

Edits (`advance_compute`, `sim_runtime.rs:518-557`):

```rust
// Hand dispatch_ticks up to MAX_FUSED_BATCHES kernel-batches so they fuse into
// one submit. The generation-budget clamp (sim_runtime.rs:538-543) stays
// UPSTREAM of dispatch_ticks, so a fused batch still stops exactly at tick_budget.
let dispatch_cap = self
    .kernel
    .kernel_batch_size()
    .saturating_mul(MAX_FUSED_BATCHES)
    .max(min_dispatch);
```

Raise `max_accumulator` (`sim_runtime.rs:523-525`) in lockstep so the accumulator
can actually fill the wider cap at high multipliers; at low speed `raw_ticks`
stays small and a single batch still dispatches, so interactive cadence is
unchanged.

Properties that make this safe:
- **Cross-batch ordering holds inside one submit.** wgpu-core inserts a storage
  barrier between every read-write-storage dispatch even within a single command
  buffer (storage usage is `EXCLUSIVE`, not `ORDERED`), so the physics/grid/
  sensory writes of batch *i* are visible to batch *i+1* exactly as when each
  batch was its own submit. Fusing changes only the encoder/submit *grouping*.
- **Decomposition is unchanged.** `kernel_batches` / `remainder_cycles` /
  `physics_remainder` arithmetic (`gpu_kernel.rs:1355-1357`, `:1439`) is
  byte-identical; the new code is another valid grouping of the same unit
  sequence that `deterministic_across_batch_sizes` already pins.
- **Barrier-uniformity invariant untouched.** Dispatch shape `(agent_count,1,1)`,
  the 256-thread workgroup, the `cycle` loop, and the `s_alive` broadcast
  (`kernel_tick.wgsl:508`, `:523`) are unchanged; only the *source* of
  `start_tick` changes (uniform read → push-constant read), outside any barrier.
- **Budget exactness untouched.** The clamp stays upstream of `dispatch_ticks`;
  the `saturating_sub` / `u32::try_from(...).unwrap_or(u32::MAX)` guards are not
  touched.
- **Precision is strictly better.** `start_tick` as exact `u32` round-trips
  identically to the former `tick as f32 → wc_u32` for every tick ≤ 2²⁴, and
  stays exact above it; the only place a difference could appear is a
  hypothetical >16.7 M-tick determinism run, which does not exist.

## 0003 — Per-iteration fixed-cost cleanup

Today `Worker::step` (`sim_runtime.rs:463-504`) calls
`try_collect_state_snapshot` and `try_collect_telemetry` every iteration, each of
which issues a `device.poll(wgpu::Maintain::Poll)` (`gpu_kernel.rs:1544`,
`:2136`) even when nothing is in flight, and `build_world_config`
(`buffers.rs:410-447`) allocates `vec![0.0f32; 24]` on every batch.

Edits:

- **Conditional polls** (`gpu_kernel.rs`): add cheap predicates and early-return
  before the poll when no readback can be ready. `device.poll(Poll)` services
  *all* device-wide `map_async` callbacks, so drive the skip off the *union* of
  both in-flight states — skipping only when both are empty is observationally
  identical (there is nothing to service).

```rust
/// True while any staging slot has an outstanding readback.
pub fn has_staging_in_flight(&self) -> bool { self.staging_in_flight.iter().any(|&b| b) }
/// True while a selected-agent telemetry readback is pending.
pub fn has_pending_telemetry(&self) -> bool { self.pending_telemetry.is_some() }
```

- **Reused world-config scratch** (`buffers.rs` + `gpu_kernel.rs`): add
  `fill_world_config(out: &mut [f32; WORLD_CONFIG_SIZE], ...)` that writes the 24
  slots in place; keep `build_world_config` as a thin wrapper over a fresh `Vec`
  for any remaining callers. Give `GpuKernel` a `world_config_scratch:
  [f32; WORLD_CONFIG_SIZE]` field and have the `upload_world_config_*` methods
  (`gpu_kernel.rs:1120-1168`, already under `&mut self` via `dispatch_ticks`)
  fill and write the scratch instead of allocating.

Properties that make this safe:
- The poll gate keeps the unmap-on-all-paths async-readback contract
  (`try_collect_staging` `gpu_kernel.rs:1283-1318`); the staging ring naturally
  idles once both predicates are false, so no slot is stranded mapped.
- `bytemuck::cast_slice` over `[f32; 24]` and a length-24 `Vec<f32>` yields
  identical bytes; the same `write_buffer` call runs with the same data.

## 0004 — Gated `global`-pass parallelization

Today the `global` pass dispatches as `pass.dispatch_workgroups(1, 1, 1)`
(`gpu_kernel.rs:1416-1424`) — a single 256-thread workgroup in `global_tick.wgsl`
doing `phase_clear` → `phase_food_grid` → `phase_food_respawn` →
`phase_agent_grid` → 3× (`collision_accumulate` → `collision_apply`), ≈42 k
serial atomic grid stores with 9 barrier pairs, once per batch. On a discrete GPU
this pins all work to one core while the rest idle.

**This workstream is gated** — see SCOPE (locked decisions). It is opened only if
`0001` shows the `global` pass is the dominant per-batch cost (tps jumps when
`XAGENT_SKIP_GLOBAL_VISION=1`). It is framed as a spike: the grid-clear and food
grid are embarrassingly parallel (dispatch `ceil(grid_cells / 256)` workgroups),
but the collision passes carry cross-cell read/write dependencies and atomics
that make multi-workgroup correctness non-trivial. The spike prototypes the
parallel grid-clear/build, measures, and decides whether the collision rework is
worth it. Decision rule in SCOPE (locked decisions); the outcome is written up in
`0004-GLOBAL-PASS-DECISION.md`.

## Test strategy

- **`deterministic_across_batch_sizes`** (`integration.rs:629-699`) is the
  primary regression gate for `0002`/`0003`: identical final positions for
  1×1000 == 2×500 == 10×100 must still hold after fusing.
- **New `fused_dispatch_matches_split` test** (`integration.rs`): run a
  *non-multiple* tick total (e.g. 1037, exercising both `remainder_cycles > 0`
  and `physics_remainder > 0`) as one fused `dispatch_ticks` call and assert
  byte-equality of final physics state vs the same total run as many small
  `dispatch_ticks` calls. The existing test only uses exact multiples and never
  touches the remainder paths. Embeds the `GpuKernel::is_available()` self-skip
  guard.
- **`worker_runs_generation_budget_handoff`** (`sim_runtime.rs` tests) must stay
  green after widening `dispatch_cap`, proving the budget clamp still stops a
  fused batch exactly at `tick_budget`.
- CI gate that must stay green:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction with prior work

- **Honors Plan 0002 (CPU/GPU Runtime Decoupling).** The simulation worker keeps
  owning the kernel; dispatch stays decoupled from the 60 Hz/30 Hz publication
  and the 6-slot latest-wins staging ring. This plan only changes how compute
  submits are grouped, not the worker boundary.
- **Honors Plan 0001 (Survival Signal Grounding).** `lag100` and the stride
  defaults are untouched; throughput is bought with fewer submits, not stride
  changes, so no `TD_DISCOUNT` recalibration is triggered.
- **Honors the 2026-04-13 tick-accumulator-backpressure design.** The accumulator
  cap and the kernel-batch decomposition that keep vision/global cadence
  deterministic are preserved; only the per-batch submit boundary moves.
- **Adjudicates `2026-06-14-grok-43.md`.** Confirms its mechanism (per-batch
  submit tax) and its fuse-submits direction, but rejects its CPU-recording
  attribution as unmeasured (see SCOPE rejected-claims table) and adds the
  measurement-first `0001` workstream plus the `global`-pass `0004` gate the
  review did not consider.
