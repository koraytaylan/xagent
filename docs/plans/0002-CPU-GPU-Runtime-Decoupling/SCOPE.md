# Scope - Plan 0002

> Decouple simulation progress from redraw cadence: the render/UI loop should
> consume the latest available snapshot, while the GPU simulation advances on a
> simulation-owned cadence, publishes CPU-visible state at bounded rates, and
> never performs CPU work proportional to `ticks_to_run`.

## Why this plan

The live runtime already moved the expensive per-agent work into
`xagent_brain::GpuKernel`. The old "one brain tick, one CPU/GPU sync" model is
not the active path anymore: `GpuKernel::dispatch_batch` batches physics,
vision, death/respawn, and brain work on the GPU, and state readback uses
non-blocking staging buffers.

The remaining coupling is at the sandbox runtime boundary:

1. **Redraw drives simulation progress.** The window redraw path calls
   `ensure_gpu_kernel`, then `step_simulation(dt)`, then rendering. If redraws
   stall, simulation stalls; if redraws run at 120 Hz, simulation scheduling
   and readback polling run at 120 Hz.
2. **Dispatch and state publication share one high-level API.**
   `dispatch_batch` always advances compute, then opportunistically requests a
   staging copy when a slot is free. That is already non-blocking, but it still
   ties "compute happened" to "try to publish CPU state".
3. **CPU snapshots are frame-rate coupled.** `collect_state_readback` polls and
   copies the latest physics state into CPU-side agents from the render loop.
   Selected-agent telemetry is also requested from the same path.
4. **One CPU loop still scales with tick count.** The governor advances with
   `for _ in 0..ticks_to_run { gov.tick(); }`, even though `tick()` is only
   `gen_tick += 1`.
5. **Generation transitions assume `App` owns the kernel.** Moving the kernel
   into a simulation worker requires an explicit request/reply protocol for
   generation-end state, champion brain readback, reset, and resumed dispatch.
6. **Rendering still consumes CPU-side positions.** The 3D renderer uploads
   instance data from CPU agent bodies. Stronger decoupling would have compute
   write render instance buffers directly, but that requires shared
   renderer/compute device ownership and is intentionally gated.

This plan is therefore not about changing `brain_tick_stride` or
`vision_stride`. Those are simulation semantics: they define brain-cycle
frequency, global/vision cadence, and sensory lag. Runtime decoupling must
change publication and ownership, not agent perception.

## In scope

Work items in [TASKS.md](TASKS.md) (workstreams 0001-0005):

- **0001 - Baseline and tick-accounting cleanup.** Measure the current
  runtime loop, batch the governor tick counter, and clamp dispatches to the
  remaining generation budget before they hit the GPU.
- **0002 - Split compute from publication.** Refactor `GpuKernel` so compute
  dispatch and state snapshot request are separate operations; preserve the
  old wrapper only as compatibility.
- **0003 - Rate-limited CPU publication in the current loop.** Before moving
  threads, prove the scheduler shape in place: request state snapshots at max
  60 Hz, selected-agent telemetry at a lower bounded rate, and update CPU/UI
  caches only when a new snapshot arrives.
- **0004 - Simulation worker ownership.** Move `GpuKernel` and simulation
  scheduling into a deterministic worker thread with command/event channels.
  The render loop becomes a latest-snapshot consumer, not the simulation
  driver.
- **0005 - Gated shared-GPU render path.** Only after the worker path is
  measured, evaluate a shared-device renderer/compute path where the GPU
  simulation writes render instance data directly.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Governor loop scales with `ticks_to_run` | `0001` |
| High-speed dispatch can overshoot generation budget | `0001` |
| `dispatch_batch` combines compute and staging request | `0002` |
| CPU-visible state and telemetry are frame-rate coupled | `0003` |
| Redraw path owns simulation progress | `0004` |
| Generation transition protocol assumes main-thread kernel ownership | `0004` |
| Renderer still needs CPU-side positions | `0005` |

## Locked decisions

- **Decouple publication, not perception.** Do not raise
  `brain_tick_stride`, `vision_stride`, or the sensory-lag bound as a UI
  optimization. Those knobs alter learning and credit assignment.
- **Latest snapshot wins.** State snapshots and telemetry are sampled
  observations. The UI wants the newest completed state, not a queue of every
  intermediate state. Bounded channels must drop or replace stale snapshots.
- **Generation boundaries are hard stops.** Before dispatching, clamp
  `ticks_to_run` to the remaining generation budget. A high-speed batch must
  not advance past the generation boundary and then attribute the overshoot to
  the completed generation.
- **No CPU work proportional to `ticks_to_run`.** Counters advance by deltas,
  replay/history sampling happens at publication cadence, and no main-thread
  loop iterates once per simulated tick.
- **Worker shutdown is explicit.** The worker exits by command-channel close or
  `Shutdown`, drains/abandons pending readbacks safely, joins in `Drop`, and
  logs panics. This follows the repository concurrency rule for background
  threads.
- **The worker owns the kernel.** Once workstream 0004 lands, the main thread
  does not call `GpuKernel::dispatch_*`, `try_collect_state`, or
  `device().poll()` from the redraw path.
- **SQLite/evolution policy stays on the main side initially.** The worker
  owns GPU simulation state and generation tick-budget enforcement; the
  existing governor/evaluation code remains outside the worker and interacts
  through explicit generation-transition commands.
- **Shared-device rendering is gated.** Direct GPU instance-buffer publishing
  is the stronger end-state, but it is not a prerequisite for practical
  decoupling. It lands only after the worker path has stable measurements.

## Out of scope

- Changing brain, vision, physics, learning, TD, or evolution semantics to
  hide runtime coupling.
- Full telemetry readback for every agent every frame.
- Cross-process simulation, remote workers, or persistent background services.
- Rewriting the renderer before the worker path is measured.
- Reworking replay file format beyond sampling the latest published state at a
  documented cadence.
- Moving the SQLite-backed `Governor` wholesale into the simulation worker in
  the first pass.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete deltas.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
