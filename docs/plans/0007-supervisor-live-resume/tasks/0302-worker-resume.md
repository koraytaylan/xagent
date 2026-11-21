---
id: worker-resume
title: Simulation worker start from checkpoint
workstream: "0300"
kind: task
role: developer
depends_on:
  - supervisor-run-resume
gated: false
touches:
  - crates/xagent-sandbox/src/sim_runtime.rs
  - crates/xagent-sandbox/src/gpu_orchestration.rs
status: done
merged_as: "1a0bc7951d236bb8be4a30ce01b3c4a6bc1ee06a"
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --bin xagent sim_runtime::"
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
---

# Simulation worker start from checkpoint

## Outcome

`SupervisorWorkerInit` gains `pub resume: bool`. With `resume: false` the worker
behaves exactly as today. With `resume: true` the supervisor path of
`start_supervisor` restores the run from the checkpoint in `db_path` and
continues the same boundary sequence as an uninterrupted worker (INV-9). Due
boundary ticks take their physics from `read_checkpoint_blocking` and write the
checkpoint before `on_sample`, with no extra dispatch, submit or readback (INV-4).
Only the two `touches` files change.

## Prerequisites used (not reimplemented)

From `supervisor-run-resume` (transitively `kernel-checkpoint`,
`checkpoint-envelope`, `checkpoint-store`, `supervisor-core-state`):

- `crate::supervisor_run::load_resume_point(db_path) -> Result<ResumePoint,
  CheckpointError>` with `ResumePoint { run_id, tick, config: FullConfig, state:
  SupervisorRunState, kernel: KernelCheckpoint, physics: Vec<f32>, food:
  Option<Vec<f32>> }`.
- `SupervisorRun::resume(kernel, config, store, state, &KernelCheckpoint) ->
  Result<Self, SupervisorRunError>`, `SupervisorRun::checkpoint_due(tick) -> bool`,
  `SupervisorRun::write_checkpoint(&mut self, &KernelCheckpoint)`.
- `SupervisorStore::reopen_for_resume(db_path, run_id, tick)`.
- `GpuKernel::read_checkpoint_blocking() -> KernelCheckpoint`;
  `KernelCheckpoint::physics() -> &[f32]` (packet 0101);
  `GpuKernel::queue_submit_count() -> u64` (public, read-only; the brain crate is
  not edited here).
- `CheckpointError` and its `Display`.
- `SupervisorRun::on_sample(kernel, tick, physics, food: Option<&[f32]>)`. The
  worker never enables the `RunSummaryAccumulator`, so `None` food matches the
  uninterrupted worker exactly.

## Decisions (explicit)

- **Resume branch.** `start_supervisor` takes the resume path only when
  `init.resume` is true. The non-resume path keeps `SupervisorStore::create`
  and `SupervisorRun::start` unchanged.
- **Order of refusals.** Checked in this order, before any store write: (1)
  `db_path` is `None` -> error `"checkpoint resume needs a database path"`; (2)
  `load_resume_point(path)` error -> its `Display`; (3) `serde_json::to_string`
  of the checkpoint `config` differs from that of `init.config` -> the
  `CheckpointError::ConfigMismatch` `Display`; (4) `init.agent_count !=
  config.governor.population_size` or `init.food_count` differs from
  `WorldState::new(config.world.clone()).food_items.len()` -> error
  `"checkpoint kernel counts differ from worker init"`. Only then
  `reopen_for_resume`, `SupervisorRun::resume` and the replay.
- **Replay.** After `SupervisorRun::resume`, the worker sets its `tick` to the
  checkpoint tick `T` and calls `on_sample(kernel, T, &physics, None)` with the
  checkpoint physics. The resulting `SampleReport` is queued as
  `SimEvent::SupervisorSample` like any other report. `birth_states` come from
  `supervisor_birth_states(&run, kernel.agent_count())`, as in the fresh path.
- **Failure.** Every resume-path error becomes one `SimEvent::Error(text)`
  (`text` is its `Display`) via the existing fresh-start error path: `supervisor
  = None`, `paused = true`, `birth_states` empty, no dispatch.
- **Due ticks.** In `on_supervisor_sample`, when `run.checkpoint_due(self.tick)`
  holds, call `self.kernel.read_checkpoint_blocking()`, then
  `run.write_checkpoint(&checkpoint)`, then take physics from
  `checkpoint.physics().to_vec()` and call `on_sample` with `None` food. A
  `write_checkpoint` error takes the existing `supervisor sample failed: {error}`
  path (one Error, paused, run dropped). Non-due ticks keep
  `read_full_state_blocking`.
- **Counters.** The worker tests compare `counters.dispatch_calls`,
  `counters.sim_ticks` and `self.kernel.queue_submit_count()` (through the
  `Worker`'s `kernel` field) with checkpointing on and off. The one-submit
  property of `read_checkpoint_blocking` is proved by `kernel-checkpoint`.
- **Config parity.** The checkpoint config must equal `init.config` exactly; the
  worker never overrides it.

## Non-goals

- No GUI entry point and no `load_checkpoint_config` call (plan 0008). Only
  `resume: false` is set in `gpu_orchestration.rs`.
- No change to any file outside `touches`, no new dependency, no WGSL, no extra
  compute pass or readback, no checkpoint history.

## Steps

**Steps:**

1. In `sim_runtime.rs`, add `pub resume: bool` to `SupervisorWorkerInit` with a
   doc comment. Set `resume: false` in the literal in `gpu_orchestration.rs`
   (around line 45) and in the test helper `supervisor_init` in the
   `sim_runtime.rs` test module.
2. In `sim_runtime.rs`, split `start_supervisor` into the unchanged create path
   and a resume path implementing the refusal order and replay in Decisions. The
   resume path returns the run, its tick, and the replayed `SampleReport`. Have
   `Worker::new` set `tick` from it and queue the replayed report. Keep the
   existing error path for any `Err`, so each failure yields one `SimEvent::Error`,
   paused, no dispatch. Add the needed imports.
3. In `sim_runtime.rs`, change `on_supervisor_sample` to the due-tick path in
   Decisions (`read_checkpoint_blocking`, `write_checkpoint`, physics from
   `checkpoint.physics()`), and keep the non-due path on `read_full_state_blocking`.
4. In the `sim_runtime.rs` test module, add
   `worker_resume_matches_uninterrupted`, using `supervisor_gpu_or_skip()`,
   `supervisor_init`, `supervisor_run_to` and the same `dt` as
   `supervisor_worker_matches_headless_driver`. Run worker A on a tempdir database
   to tick `X` (at least two boundaries past the first checkpoint), drop it,
   copy the database to a second path, and read `T` with `load_resume_point`. Start
   worker B on the copy with `resume: true` and run it to `X`. Assert `T > 0`, that
   worker B reports at least one sample, and that the `SampleReport`s with
   `tick > T` from A and B are equal.
5. In the same test module, add `worker_resume_invalid_checkpoint_errors_and_pauses`
   (run a database to a checkpoint, drop the worker, flip one payload byte of the
   `supervisor_checkpoint` row with `rusqlite`, then start with `resume: true` and
   assert exactly one `SimEvent::Error` containing `checkpoint`, `paused`,
   `supervisor` is `None`, `dispatch_calls == 0`, `tick == 0`, and the payload
   still equals the corrupted bytes). Add a second worker with `db_path: None` and
   `resume: true` asserting one Error and paused. Also add
   `worker_checkpoint_adds_no_submit`: run the same config to `X` once with a
   database and once with `db_path: None`, and assert equal
   `counters.dispatch_calls`, `counters.sim_ticks`,
   `kernel.queue_submit_count()` and count of `SupervisorSample` events.
6. Request `focused-proof`, `sandbox-clippy` and `sandbox-fmt` through `verify`.
   If `sandbox-fmt` fails, apply its fix and request all three again. Edit nothing
   outside `touches`.

- **Done when:** `focused-proof`, `sandbox-clippy` and `sandbox-fmt` pass, including `worker_resume_matches_uninterrupted`, `worker_resume_invalid_checkpoint_errors_and_pauses` and `worker_checkpoint_adds_no_submit`, proving INV-9 (a resumed worker reports the same boundary sequence after T as the uninterrupted run, and an invalid checkpoint yields one Error, paused, no dispatch) and INV-4 (checkpointing adds no queue submit, kernel batch or sample dispatch to the worker).
