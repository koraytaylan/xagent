---
id: worker-telemetry
title: "Worker publishes telemetry on the bounded event channel"
workstream: "0500"
kind: task
role: developer
depends_on:
  - run-recording
gated: false
status: done
merged_as: "99693e99d6b8c07c22b5954b52d9961f61729dca"
touches:
  - crates/xagent-sandbox/src/sim_runtime.rs
  - crates/xagent-sandbox/src/gpu_orchestration.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent sim_runtime::tests::"
    timeout_ms: 3600000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
---

# Worker publishes telemetry on the bounded event channel

## Outcome

The supervisor path of the simulation worker publishes one telemetry frame per
supervisor sample on the existing bounded event channel, latest-wins and never
blocking, and publishes each boundary record as an ordered control event ahead
of that sample's `SupervisorSample`. The main thread ignores both new variants.
No GPU dispatch or readback is added; the food given to the recorder is the
cached food state. The binding rules are in ARCHITECTURE.md "Worker".

## Decisions

- `SimEvent` in `sim_runtime.rs` gains two variants, each with a doc comment
  and `#[allow(dead_code)]`:
  - `SupervisorTelemetry(Box<TelemetryFrame>)`: sent with `try_send` only.
  - `SupervisorBoundary(Box<BoundaryRecord>)`: control event, sent with the
    blocking `send` loop.
  Both types come from `crate::supervisor_telemetry` (prerequisite
  `run-recording`), which derives `Clone` on both; this task does not add it.
- `Worker` gains two private fields, both `None` in `Worker::new`:
  - `pending_telemetry: Option<Box<TelemetryFrame>>` (latest replaces older).
  - `supervisor_sample_clock: Option<(Instant, u64)>`: wall-clock instant and
    tick of the previous supervisor sample, for ticks per second.
- `on_supervisor_sample`, in order:
  1. Keep the existing `read_full_state_blocking().to_vec()` as the only
     readback.
  2. `let food = self.kernel.cached_food_state().map(<[f32]>::to_vec);` (owned,
     so the kernel borrow ends before `&mut self.kernel` is passed).
  3. When `supervisor_sample_clock` is `Some((at, tick0))` and the elapsed time
     is positive, call `run.note_ticks_per_second((self.tick - tick0) as f64 /
     elapsed)`. Then set the clock to `Some((Instant::now(), self.tick))`.
  4. Call `run.on_sample(&mut self.kernel, self.tick, &physics,
     food.as_deref())` in place of the current `None`. Food is used only for
     telemetry and measurement.
  5. On `Ok(report)`, keep birth-state and finish handling unchanged, then call
     `run.take_frame()`. If the frame has a boundary, push
     `SimEvent::SupervisorBoundary(Box::new(record.clone()))` before the
     existing `SupervisorSample` push, so the boundary precedes its sample.
     Store `self.pending_telemetry = Some(Box::new(frame))`, replacing any older
     frame.
- `step()`: keep the `supervisor_events` drain with blocking `send`. Right after
  it and before the paused check, `take()` `pending_telemetry` and `try_send`
  it as `SimEvent::SupervisorTelemetry`; on `Full` or `Disconnected` drop it.
  The call never blocks.
- `partition_events` is unchanged; the new variants fall into its `other =>`
  arm and come back as control events in arrival order.
- `gpu_orchestration.rs`: in the exhaustive `SimEvent` match near
  `SimEvent::SupervisorSample(report)`, add `SimEvent::SupervisorTelemetry(_) =>
  {}` and `SimEvent::SupervisorBoundary(_) => {}` with one short comment that
  the observatory consumes them. Nothing else in that file changes.
- Test helpers go inside `mod tests` of `sim_runtime.rs`, reusing
  `supervisor_gpu_or_skip`, `supervisor_init`, `supervisor_test_config` and
  `supervisor_dt`. `supervisor_run_to` stays unchanged.
- No plan numbers, task slugs or workstream words in source comments or test
  names.

## Non-goals

- No change to generation mode, `SupervisorRun::on_sample` decisions,
  `SampleReport` / `BoundaryReport`, `partition_events` logic or
  `EVENT_CAPACITY`.
- No new GPU dispatch, readback, WGSL or kernel change.
- No UI consumption (0008), no store writes from the worker, no checkpoint or
  replay.
- No edit outside the two touches.

**Steps:**

1. In `sim_runtime.rs`, add the two `SimEvent` variants with doc comments,
   `#[allow(dead_code)]` and the `supervisor_telemetry` imports. In
   `gpu_orchestration.rs`, add the two ignoring match arms with the one-line
   comment.
2. In `sim_runtime.rs`, add `pending_telemetry` and `supervisor_sample_clock`
   to `Worker`, initialised to `None` in `Worker::new`; leave other constructor
   paths unchanged.
3. In `Worker::on_supervisor_sample`, implement the decision steps 2 to 5 above:
   food copy, ticks per second, food passed to `on_sample`, `take_frame()`,
   boundary push before the sample push, and storing `pending_telemetry`.
   Keep the birth-state, finish and error branches unchanged.
4. In `Worker::step`, after the control drain and before the paused check,
   `take()` `pending_telemetry` and `try_send` it as
   `SimEvent::SupervisorTelemetry`, dropping it on `Full` or `Disconnected`.
5. Add these tests inside `mod tests` in `sim_runtime.rs`:
   - `supervisor_telemetry_partition_keeps_new_variants_in_order` (no GPU):
     `partition_events` on `[SupervisorBoundary, Log, SupervisorTelemetry]`
     returns the boundary and the log as control events, in that order.
   - `supervisor_telemetry_boundary_control_events_follow_samples` (GPU guard):
     `supervisor_init` at speed 1000, `dt = 0.05`; drive `advance_compute`
     until tick 3000 or pause, collecting `supervisor_events` in order without
     `step`. Assert at least two boundary reports; each `SupervisorSample` with
     a boundary is immediately preceded by `SupervisorBoundary` with the same
     tick; the `SupervisorBoundary` count equals the boundary-report count.
   - `supervisor_telemetry_frame_dropped_when_channel_full` (GPU guard):
     `sync_channel::<SimEvent>(1)` filled with `SimEvent::Log`, never drained;
     loop `worker.step(&event_tx)` until tick 900 (before the first boundary).
     Assert the loop returns, `dispatch_calls` advanced, and `try_recv()`
     yields the `Log` and then `Err`.
   - `supervisor_telemetry_dispatch_schedule_unchanged_when_frames_discarded`
     (GPU guard): two workers from identical `supervisor_init` at speed 1000
     with the same `dt` sequence. Worker A runs normally; worker B sets
     `pending_telemetry = None` and removes `SupervisorBoundary` events from
     `supervisor_events` after each `advance_compute`. Run both to tick 3000.
     Assert equal `tick`, `counters.dispatch_calls`,
     `kernel.probe_kernel_batches()` and `SampleReport` vectors; also assert
     A's `probe_kernel_batches() == tick / batch`.
6. Leave `supervisor_worker_matches_headless_driver` and the other existing
   supervisor worker tests unmodified; they must stay green, showing the
   worker's `SampleReport`s still equal the headless driver's.

- **Done when:** the focused-proof check (`sim_runtime::tests::` under `--bin xagent` with `XAGENT_REQUIRE_GPU=1`) passes, including the new `supervisor_telemetry_` tests and the unchanged `supervisor_worker_matches_headless_driver`, proving INV-9.
