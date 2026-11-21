---
id: worker-tick-window
title: "Simulation worker capture and persistence"
workstream: "0200"
kind: task
role: developer
depends_on: [tick-window-core]
gated: false
touches: [crates/xagent-sandbox/src/sim_runtime.rs]
status: done
merged_as: "a140080b1ac0a501f22778e22635d979beae9539"
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent sim_runtime::"
    timeout_ms: 3600000
---

# Simulation worker capture and persistence

## Outcome

In supervisor mode with `SupervisorWorkerInit.db_path` set, the simulation worker opens a `TickWindowStore` for the latest supervisor run right after `start_supervisor` succeeds and keeps it in a new `Option` field. Each snapshot the worker builds for publication is captured as a `WindowFrame` and recorded before the `try_send`, so capture does not depend on channel back-pressure. Each supervisor sample that replaced a slot (not an idle boundary) or flipped the season records a pin. A store error is logged once, the store is dropped, and the run continues. Generation mode and supervisor mode without a database have no store and no behaviour change.

## Scope

- `sim_runtime.rs` only, including its existing `tests` module.
- Import `xagent_sandbox::tick_window::{frame_from_snapshot, Pin, PinKind}`, `xagent_sandbox::tick_window::TickWindowConfig` and `xagent_sandbox::tick_window_store::TickWindowStore` from `tick-window-core`. Use them exactly as the ARCHITECTURE.md signatures define them.
- Add the field `tick_window_store: Option<TickWindowStore>` to `struct Worker`. Set it in `Worker::new`'s supervisor `Ok(run)` arm: when `sup_init.db_path` is `Some`, call `TickWindowStore::open_for_latest_supervisor_run(path, TickWindowConfig::default())`. Otherwise, and on error, leave it `None`. An open error pushes one `SimEvent::Log` onto `supervisor_events` and leaves the run going.
- Change `fn publish_snapshot(&self, ...)` to `fn publish_snapshot(&mut self, ...)`. Its only caller is in `step`, which already takes `&mut self`.
- In `publish_snapshot`, build the snapshot as today. Before `try_send`, when a store is present, call `frame_from_snapshot(self.tick, &snapshot.physics, &snapshot.food)` and `record_frame`. The `try_send` and its latest-wins drop remain unchanged and run after capture.
- In `on_supervisor_sample`, in the `Ok(report)` arm, after the birth-state handling and before the event is pushed, record pins: `PinKind::Replacement { slot }` when `report.boundary` has a `slot`, and `PinKind::SeasonFlip` when `report.season` is `Some`. Each pin uses `tick: self.tick`, which equals the sample tick.
- A shared private method `fn disable_tick_window(&mut self, reason: String)` pushes one `SimEvent::Log` line onto `supervisor_events` and sets the field to `None`. Frame, pin and frame-build errors all call it.
- Tests in the existing `tests` module of `sim_runtime.rs`, using `supervisor_gpu_or_skip`, `supervisor_test_config`, `supervisor_init` and `supervisor_dt`.

## Non-goals

- No GPU dispatch, readback, kernel call or WGSL change. Frames come only from `StateSnapshot.physics` and `StateSnapshot.food` already built by `build_snapshot`. No new `GpuKernel` method is called from any new code path.
- No change to `SupervisorWorkerInit`, `SimInit`, `SimEvent`, `SampleReport`, `supervisor_run.rs`, `supervisor_store.rs` or any supervisor decision.
- No change to `headless.rs`, `main.rs`, the GUI or `replay.rs`.
- No CLI flag and no config default change.

## Decisions

- **Frame tick**: the frame tick is `self.tick`, the worker tick at publication. The worker keeps no per-snapshot tick, so a snapshot published after a readback can carry the tick of the newest dispatch. Every stored tick is still `<= self.tick`, which is the bound the tests assert.
- **Failure handling**: any `TickWindowStoreError` or `frame_from_snapshot` error goes to `disable_tick_window`. The worker does not retry, does not pause and does not return an error to the GUI.
- **Test driver**: the existing `supervisor_run_to` advances compute without publishing. Tests therefore collect a snapshot with `kernel.request_state_snapshot()` and a spin on `kernel.try_collect_state_snapshot()`, then call `worker.publish_snapshot(&event_tx)` on each step. Both kernel calls are test-only and sit outside the capture path.
- **Failure test**: the store failure is produced by dropping `supervisor_replay_frame` through a separate `rusqlite` connection to the same tempfile database after the worker has opened its store. The `DROP TABLE` runs while the store's connection is idle, which is the normal state between writes.
- **Generation and no-database cases**: `Worker::new` with `supervisor: None`, and supervisor init with `db_path: None`, both leave `tick_window_store` as `None`. The test asserts this field directly. No replay table is created, because no database is opened on either path.

## Steps

**Steps:**
1. In `sim_runtime.rs`, add the `tick_window_store: Option<TickWindowStore>` field to `struct Worker`, its imports, and the `disable_tick_window` method. In `Worker::new`, open the store in the supervisor `Ok(run)` arm when `sup_init.db_path` is `Some`, log and leave it `None` on error, and set the field to `None` in every other branch.
2. In `sim_runtime.rs`, change `publish_snapshot` to take `&mut self`. Inside it, build the snapshot, and when a store is present, call `frame_from_snapshot(self.tick, &snapshot.physics, &snapshot.food)` and `record_frame` before the `try_send`. On any error call `disable_tick_window`. Keep the `try_send` drop behaviour unchanged.
3. In `sim_runtime.rs`, in `on_supervisor_sample`'s `Ok(report)` arm, record `PinKind::Replacement { slot }` when the report names a replaced slot and `PinKind::SeasonFlip` when the season is `Some`, each with `tick: self.tick` through `record_pin`. Call `disable_tick_window` on any error. Keep the existing event push and the `finished` handling unchanged.
4. In the `tests` module of `sim_runtime.rs`, add `tick_window_store_persists_frames_in_bounds`. Use `supervisor_gpu_or_skip`, a `tempfile` database path and `supervisor_init(&config, Some(path))`. Drive the worker with `supervisor_run_to`, collect and publish a snapshot after each advance, then drop the worker. Open the file with `rusqlite` and assert at least one `supervisor_replay_frame` row, every row's tick `<= worker.tick` and `>= 0`, and that the run is not paused by the capture.
5. In the same module, add `tick_window_absent_without_database` and `tick_window_store_failure_logs_and_continues`. The first asserts that `tick_window_store` is `None` for `supervisor_init(&config, None)` and for `test_init` in generation mode. The second opens the worker with a database, drops `supervisor_replay_frame` through a separate connection, publishes a snapshot and asserts exactly one `SimEvent::Log` in `supervisor_events`, a `None` store, `worker.paused == false` and tick progress after another `supervisor_run_to`.
6. Keep `sim_runtime.rs` rustfmt-clean by hand, since the plan-wide `fmt` check judges it, and make sure the existing tests in `sim_runtime::` still pass unchanged. Do not edit any assertion of an existing test. Add no comment that names a plan, task or packet.

- **Done when:** the check `focused-proof` passes with `tick_window_store_persists_frames_in_bounds`, `tick_window_absent_without_database` and `tick_window_store_failure_logs_and_continues` reported ok and every existing `sim_runtime::` test still ok. The capture path reads only the snapshot slices built by `build_snapshot` and names no new kernel call (INV-2). The store failure produces exactly one `SimEvent::Log`, drops the store and leaves the worker running (INV-11).
