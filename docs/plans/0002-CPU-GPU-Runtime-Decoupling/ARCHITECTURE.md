# Architecture - Plan 0002 (deltas)

> Edits center on `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-sandbox/src/gpu_orchestration.rs`,
> `crates/xagent-sandbox/src/app.rs`, `crates/xagent-sandbox/src/main.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-sandbox/src/replay_coord.rs`,
> `crates/xagent-sandbox/src/snapshot.rs`,
> `crates/xagent-sandbox/src/render_pipeline.rs`, a new sandbox runtime module
> such as `crates/xagent-sandbox/src/sim_runtime.rs`, and integration tests in
> `crates/xagent-sandbox/tests/integration.rs`. Line numbers are hints; locate
> by symbol.

## 0001 - Baseline and tick-accounting cleanup

### Runtime counters

Add lightweight counters around the live loop before changing ownership:

- simulated ticks advanced,
- dispatch calls,
- kernel batches submitted when cheaply available,
- state snapshot requests,
- state snapshots collected,
- telemetry requests,
- telemetry snapshots collected,
- frames rendered,
- generation-boundary clamp count.

The counters can live in `App` initially and be printed in the existing
diagnostic path or a focused bench. They are not a new UI feature; they are the
before/after evidence for this plan.

### Batched governor advancement

Today `Governor::tick()` increments one tick. Add:

```rust
/// Advance the generation tick counter by `ticks`.
///
/// Equivalent to calling `tick()` `ticks` times, but constant-time so the
/// sandbox loop does no CPU work proportional to simulated ticks.
pub fn advance_ticks(&mut self, ticks: u64) {
    self.gen_tick = self.gen_tick.saturating_add(ticks);
}
```

Keep `tick()` as a one-tick convenience wrapper:

```rust
pub fn tick(&mut self) {
    self.advance_ticks(1);
}
```

Unit tests assert equivalence with repeated `tick()` and saturating behavior.

### Generation-budget clamping

Before calling into the GPU, clamp the selected `ticks_to_run` to the
remaining generation budget when a governor exists:

```rust
let ticks_to_run = if let Some(governor) = &self.governor {
    let remaining = governor.config.tick_budget.saturating_sub(governor.gen_tick);
    ticks_to_run.min(remaining.min(u64::from(u32::MAX)) as u32)
} else {
    ticks_to_run
};
```

The exact code should avoid lossy casts per repository rules, but the behavior
is: no dispatch after the generation budget is exhausted, and the last dispatch
of a generation stops exactly at the budget.

## 0002 - Split compute from publication

`GpuKernel::dispatch_batch` currently performs both compute submission and an
opportunistic staging-copy request. Split that into three concepts:

```rust
pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool;
pub fn request_state_snapshot(&mut self) -> bool;
pub fn try_collect_state_snapshot(&mut self) -> bool;
```

`dispatch_ticks` is the existing compute path through full kernel batches,
remainder kernel batches, and physics-only remainder. It does not copy
`agent_phys_buffer` or `food_state_buffer` into staging buffers.

`request_state_snapshot` performs only the free-slot scan, staging copy,
`map_async` installation, and staging-ring index update. It returns `false`
when every staging slot is in flight.

`try_collect_state_snapshot` is the current non-blocking collection path. The
existing `try_collect_state` name may remain as a compatibility alias if that
keeps call sites smaller.

Keep `dispatch_batch` temporarily as:

```rust
pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {
    let dispatched = self.dispatch_ticks(start_tick, ticks_to_run);
    let _snapshot_requested = self.request_state_snapshot();
    dispatched
}
```

That preserves old tests while new runtime code switches to the separated API.
Documentation must state that compute dispatch is independent from CPU-visible
publication.

## 0003 - Rate-limited CPU publication in the current loop

Before introducing a worker thread, make the existing main-thread loop obey the
target scheduling model.

Add named cadence constants:

```rust
/// Maximum CPU-visible physics snapshot publication rate. Display smoothness
/// does not require more than one authoritative state sample per 60 Hz frame.
const STATE_SNAPSHOT_MAX_HZ: f64 = 60.0;

/// Selected-agent telemetry is heavier than physics state and drives charts,
/// not body placement, so it can publish below display rate.
const TELEMETRY_MAX_HZ: f64 = 30.0;
```

`dispatch_sim_ticks` should advance compute when ticks are available, but only
call `request_state_snapshot` when the state cadence is due or a force flag is
set. The force cases are startup, selected reset/generation transition, and
any test/debug path that needs an immediate blocking read.

`collect_state_readback` should remain non-blocking, but CPU-side agent bodies,
HUD dirtiness, history recording, heatmap/trail recording, replay recording,
and snapshot rebuild dirtiness should update only when a new state snapshot was
actually collected. Frames without a new snapshot render the previous cache.

Telemetry gets its own scheduler:

- request immediately when selected agent changes,
- otherwise request at `TELEMETRY_MAX_HZ`,
- collect opportunistically every frame,
- do not overwrite fields that the physics snapshot owns.

This stage proves the decoupled semantics without the complexity of cross-thread
ownership.

## 0004 - Simulation worker ownership

Introduce a runtime boundary that owns `GpuKernel` and all simulation-cadence
state. Exact names can change, but the shape should be close to:

```rust
pub struct SimRuntime {
    command_tx: SyncSender<SimCommand>,
    event_rx: Receiver<SimEvent>,
    join_handle: Option<JoinHandle<()>>,
}

pub enum SimCommand {
    SetPaused(bool),
    SetSpeed(u32),
    SelectAgent(usize),
    ConfigureGeneration { tick_budget: u64 },
    RequestAgentState { agent_index: u32, request_id: u64 },
    ResetPopulation(PendingUpload),
    Shutdown,
}

pub enum SimEvent {
    KernelReady { agent_count: u32 },
    Snapshot(StateSnapshot),
    Telemetry { agent_index: u32, telemetry: AgentTelemetry },
    GenerationBudgetReached(StateSnapshot),
    AgentState { request_id: u64, state: Option<AgentBrainState> },
    Log(String),
    Error(String),
}
```

`StateSnapshot` owns vectors for physics and food state plus `tick` and
`generation_tick`. It must not borrow from `GpuKernel::cached_state()`.

The worker loop:

1. Drains commands without blocking simulation indefinitely.
2. Accumulates wall time when not paused.
3. Computes a tick budget from speed, elapsed time, GPU budget, kernel stride,
   and remaining generation ticks.
4. Dispatches compute with `dispatch_ticks`.
5. Requests snapshots and telemetry only when their cadences are due.
6. Polls readbacks and publishes only the latest complete snapshot.
7. Pauses itself and emits `GenerationBudgetReached` when the generation budget
   reaches zero.
8. Exits on `Shutdown` or command-channel disconnect and joins in `Drop`.

The main thread:

- sends commands for speed, pause, selected agent, reset, and shutdown,
- drains events each frame,
- applies the newest `StateSnapshot` to CPU-side `Agent` caches,
- renders from those caches,
- stops calling `GpuKernel` from redraw handling.

### Generation transition protocol

Keep SQLite/evaluation policy in the main thread for this plan. The worker
only owns GPU state and tick-budget enforcement.

At generation budget reached:

1. Worker emits `GenerationBudgetReached(snapshot)` and pauses dispatch.
2. Main applies the snapshot to CPU agents and evaluates with the existing
   governor code.
3. Main asks the worker for the champion `AgentBrainState` with
   `RequestAgentState`.
4. Worker performs the async or blocking GPU readback while paused and replies
   with `AgentState`.
5. Main prepares the next population/world upload and sends `ResetPopulation`.
6. Worker uploads world/agents, applies inherited/mutated brain states through
   existing kernel write APIs, resets its generation tick counter, and resumes
   only after an explicit unpause/resume command.

No generation transition code may reach into `App.gpu_kernel` after this stage.

## 0005 - Gated shared-GPU render path

This is the stronger endpoint and should land only after workstream 0004 has
measurements.

Current rendering still rebuilds `InstanceData` on the CPU and writes it into
the renderer's instance buffer. To remove that copy, renderer and compute must
share one `wgpu::Device`/`Queue`, and the compute path must write render-ready
instance data or a compact render-state buffer.

Possible direction:

- make renderer/device creation the owner of `Arc<wgpu::Device>` and
  `Arc<wgpu::Queue>`,
- add a `GpuKernel::new_with_device(...)` constructor,
- allocate an agent instance storage/vertex buffer visible to both compute and
  render passes,
- add a lightweight compute pass that converts `physics_state` plus agent
  palette/alive state into render instances,
- render directly from that buffer.

This changes GPU ownership substantially and must be measured against the
worker-only path. It is not required for the first practical decoupling win.
