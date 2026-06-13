# XAgent Plan 0002 - CPU/GPU Runtime Decoupling

Make simulation progress independent from redraw cadence: batch the remaining
CPU tick accounting, split GPU compute dispatch from CPU-visible snapshot
publication, throttle state and telemetry publication, then move the kernel
into a simulation worker so the render loop consumes latest snapshots instead
of driving ticks.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md)
for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists direct prerequisites only.
- **Done when** is the verifiable acceptance check. Every task must keep
  `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test -p xagent-sandbox` green unless the task explicitly says it is
  documentation-only.
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI
  runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 - Baseline and tick-accounting cleanup

### runtime-decoupling-baseline - Record current loop behavior

The runtime already batches GPU work and uses async readback, so this plan
needs numbers before changing scheduling.

**Steps:**

1. Add a small internal counter struct near the sandbox runtime loop:
   `sim_ticks`, `dispatch_calls`, `state_snapshot_requests`,
   `state_snapshots_collected`, `telemetry_requests`,
   `telemetry_snapshots_collected`, `frames_rendered`, and
   `generation_clamps`.
2. Increment counters at the existing call sites in
   `gpu_orchestration.rs`, `main.rs`, and telemetry request/collect code.
3. Expose the counters only through a debug log or bench output; do not add
   new UI.
4. Run a fixed-speed baseline at `100x` and `1000x` with 3D on and off, and
   record the numbers in a short subsection under
   `docs/superpowers/specs/` or in the PR body.

- **Depends on:** -
- **Done when:** baseline numbers show dispatches, readback requests,
  collected snapshots, telemetry requests, and rendered frames separately;
  no behavioral code changes are included in this task.

### governor-advance-ticks - Replace per-tick governor advancement

`dispatch_sim_ticks` currently advances the governor by looping once per
simulated tick even though `Governor::tick()` only increments a counter.

**Steps:**

1. In `crates/xagent-sandbox/src/governor.rs`, add
   `pub fn advance_ticks(&mut self, ticks: u64)` using saturating addition.
2. Reimplement `tick()` as `self.advance_ticks(1)`.
3. Replace the `for _ in 0..ticks_to_run { gov.tick(); }` loop in
   `crates/xagent-sandbox/src/gpu_orchestration.rs` with a single
   `gov.advance_ticks(u64::from(ticks_to_run))`.
4. Add unit tests proving repeated `tick()` and one `advance_ticks(n)` produce
   the same `gen_tick`.

- **Depends on:** -
- **Done when:** no runtime path loops over `ticks_to_run` just to advance the
  governor; tests cover equivalence.

### generation-budget-clamp - Stop batches at the generation boundary

High multipliers can choose a batch larger than the remaining generation
budget. The generation should end exactly at its configured tick budget.

**Steps:**

1. In `dispatch_sim_ticks`, after computing the raw tick batch and before
   dispatch, clamp to `governor.config.tick_budget - governor.gen_tick` when a
   governor exists.
2. If the clamp produces zero ticks, skip dispatch and let the existing
   generation-complete path run.
3. Subtract only the clamped tick count from `sim_accumulator`.
4. Increment the baseline clamp counter when clamping occurs.
5. Add a focused test for a small tick budget where accumulated ticks exceed
   the remaining generation budget.

- **Depends on:** `governor-advance-ticks`
- **Done when:** the final dispatch of a generation cannot overshoot
  `tick_budget`, and the accumulator/tick counters reflect the actual
  dispatched count.

---

## 0002 - Split compute from publication

### split-dispatch-readback-api - Separate GPU compute from snapshot request

`GpuKernel::dispatch_batch` advances compute and then opportunistically starts
a staging readback. Runtime decoupling needs those operations independently
schedulable.

**Steps:**

1. In `crates/xagent-brain/src/gpu_kernel.rs`, extract the compute portion of
   `dispatch_batch` into `dispatch_ticks(start_tick, ticks_to_run)`.
2. Extract the staging-copy/free-slot scan into
   `request_state_snapshot() -> bool`.
3. Rename or alias `try_collect_state()` to
   `try_collect_state_snapshot()` while preserving existing call sites during
   migration.
4. Keep `dispatch_batch` as a compatibility wrapper that calls
   `dispatch_ticks` and then `request_state_snapshot`.
5. Update `crates/xagent-brain/README.md` and relevant module docs so
   "dispatch" and "publish/readback request" are distinct.
6. Add or update integration tests:
   - dispatch without requesting a snapshot still advances state visible via a
     blocking debug read,
   - requesting a snapshot after dispatch eventually updates `cached_state`,
   - the old `dispatch_batch` wrapper still behaves as before.

- **Depends on:** `generation-budget-clamp`
- **Done when:** new code can advance GPU simulation without asking for CPU
  state publication, and old tests using `dispatch_batch` remain green.

---

## 0003 - Rate-limited publication in the current loop

### state-snapshot-rate-limit - Publish physics state at max 60 Hz

Prove the decoupled scheduler while everything still runs on the main thread.

**Steps:**

1. Add named constants for state snapshot max frequency and minimum interval.
2. Track `last_state_snapshot_request` in `App`.
3. Change `dispatch_sim_ticks` to call `kernel.dispatch_ticks(...)` for
   compute, and call `kernel.request_state_snapshot()` only when the cadence
   is due or when a force flag is set.
4. Keep `collect_state_readback` non-blocking and callable every frame.
5. Move history/heatmap/trail/replay sampling so it runs only when a new state
   snapshot is collected and applied, not on frames that reuse the old state.
6. Document that CPU histories and replay sample the latest published state,
   not every simulated tick.

- **Depends on:** `split-dispatch-readback-api`
- **Done when:** at 120 Hz rendering, state snapshot requests are capped near
  60 Hz while simulation ticks continue to advance at the requested speed.

### selected-telemetry-rate-limit - Throttle heavy selected-agent telemetry

Selected-agent telemetry copies sensory, decision, brain, and physics slices.
It should not run at unbounded redraw cadence.

**Steps:**

1. Add a telemetry max frequency constant, initially 30 Hz.
2. Track `last_telemetry_request` and the last selected agent index.
3. Request telemetry immediately when selection changes; otherwise request
   only when the telemetry cadence is due.
4. Continue polling collection every frame.
5. Keep the existing authority split: physics snapshot owns position, vitals,
   motor, gradient, urgency, fatigue, prediction error, and exploration;
   telemetry owns selected-agent vision color and derived brain fields.

- **Depends on:** `state-snapshot-rate-limit`
- **Done when:** telemetry request rate is bounded independently from render
  FPS and selected-agent changes still update promptly.

### main-loop-publication-baseline - Re-measure after rate limits

**Steps:**

1. Repeat the `runtime-decoupling-baseline` measurements at the same speeds.
2. Record before/after dispatch rate, snapshot request rate, telemetry request
   rate, collected snapshots, rendered frames, and achieved TPS.
3. Decide whether the worker migration should keep the same 60/30 Hz defaults
   or adjust them before workstream 0004.

- **Depends on:** `selected-telemetry-rate-limit`
- **Done when:** the current-loop scheduler has measured request-rate caps and
  no loss of requested simulation TPS relative to baseline beyond documented
  noise.

---

## 0004 - Simulation worker ownership

### sim-runtime-protocol - Define commands, events, and owned snapshots

The worker boundary should be explicit before moving `GpuKernel`.

**Steps:**

1. Add a new sandbox module, for example
   `crates/xagent-sandbox/src/sim_runtime.rs`.
2. Define `SimCommand`, `SimEvent`, `StateSnapshot`, and `SimRuntime` handle
   types matching the architecture document.
3. Use bounded channels for commands/events. Snapshot events must be latest-wins
   in the main loop: drain all pending snapshot events and apply only the last.
4. Ensure `StateSnapshot` owns its vectors; it must not borrow from kernel
   caches.
5. Add unit tests for latest-wins event draining and command-channel shutdown
   behavior without requiring a GPU.

- **Depends on:** `main-loop-publication-baseline`
- **Done when:** the protocol compiles and is tested before it owns real GPU
  state.

### sim-worker-owns-kernel - Move dispatch scheduling out of redraw

**Steps:**

1. Create the worker thread from `SimRuntime::start(...)`.
2. Move `GpuKernel::new`, world/agent upload, accumulator, tick counter,
   speed multiplier, pause state, snapshot scheduler, and telemetry scheduler
   into the worker.
3. The worker calls `dispatch_ticks`, `request_state_snapshot`,
   `try_collect_state_snapshot`, `request_agent_telemetry`, and
   `try_collect_telemetry`.
4. The main redraw path drains events and renders cached state. It does not
   call `GpuKernel` methods or `kernel.device().poll()`.
5. Implement deterministic shutdown: send `Shutdown`, close the channel, join
   the thread in `Drop`, and log any panic.
6. Preserve the background kernel-creation behavior as either part of the
   worker startup or as a `KernelReady` event.

- **Depends on:** `sim-runtime-protocol`
- **Done when:** simulation ticks advance while redraw only consumes snapshot
  events; searching the redraw path shows no direct `GpuKernel` dispatch,
  readback, or device-poll calls.

### worker-generation-handoff - Migrate generation transitions to commands

Generation transitions are the main correctness risk because they need final
physics state and champion brain state while the worker owns the kernel.

**Steps:**

1. Give the worker a generation tick budget via `ConfigureGeneration`.
2. Clamp worker dispatches to the remaining budget and emit
   `GenerationBudgetReached(StateSnapshot)` exactly once per generation.
3. On that event, main applies the snapshot, evaluates with the existing
   governor, and sends `RequestAgentState` for the chosen champion.
4. Worker replies with `AgentState { request_id, state }` while paused.
5. Main prepares the next population and sends `ResetPopulation(...)`.
6. Worker uploads world/agent rows, applies inherited/mutated brain states
   using existing kernel write APIs, resets generation tick counters, and waits
   for resume/unpause.
7. Remove or retire the old `App.gpu_kernel` generation-transition path.

- **Depends on:** `sim-worker-owns-kernel`
- **Done when:** generation advance, champion inheritance, and reset work
  through worker commands only; no transition code directly accesses
  `App.gpu_kernel`.

### worker-runtime-baseline - Measure worker decoupling

**Steps:**

1. Repeat the baseline matrix from workstreams 0001 and 0003.
2. Add a redraw-stall check: artificially skip or delay rendering for a short
   interval and verify the worker continues advancing ticks until paused or
   generation budget is reached.
3. Record worker event rates, dropped/replaced snapshot counts, achieved TPS,
   and UI frame rate.

- **Depends on:** `worker-generation-handoff`
- **Done when:** simulation progress no longer depends on redraw cadence, and
  snapshot publication remains bounded.

---

## 0005 - Gated shared-GPU render path

### shared-device-render-spike - Decide whether direct GPU instance publishing is worth it

The worker path still updates CPU-side agent caches and uploads render
instances from CPU. Direct GPU render buffers are a separate, higher-risk
optimization.

**Steps:**

1. Write a short design note comparing:
   - worker-only path,
   - shared renderer/compute device,
   - compute-written render instance buffer.
2. Prototype only if worker measurements show CPU agent-cache or instance
   upload cost is material.
3. If prototyped, use one shared `wgpu::Device`/`Queue` for renderer and
   compute, and add a compute pass that writes render-ready agent instances.
4. Verify desktop/mobile-sized viewports still render correctly and no blank
   frame appears while compute is running.

- **Depends on:** `worker-runtime-baseline`
- **Done when:** either the note rejects the shared-device path with measured
  evidence, or a measured prototype shows enough benefit to justify a follow-up
  implementation plan.
