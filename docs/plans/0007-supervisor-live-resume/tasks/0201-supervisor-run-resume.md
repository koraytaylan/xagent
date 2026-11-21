---
id: supervisor-run-resume
title: Supervisor run checkpoint and resume
workstream: "0200"
kind: task
role: developer
depends_on:
  - kernel-checkpoint
  - supervisor-core-state
  - checkpoint-envelope
  - checkpoint-store
gated: false
touches:
  - crates/xagent-sandbox/src/supervisor_run.rs
  - crates/xagent-sandbox/src/run_summary.rs
  - crates/xagent-sandbox/tests/supervisor_run_resume.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_run_resume
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
---

# Supervisor run checkpoint and resume

## Outcome

A `SupervisorRun` writes its complete pre-sample state as one checkpoint and is
rebuilt from it on a freshly built kernel. The resumed run produces the same
`SampleReport` sequence and store rows as the uninterrupted run (INV-7). The
kernel capture, core capture, envelope and store primitives come from the four
`depends_on` tasks and are used, not reimplemented. No other file changes.

## Decisions (explicit)

- **Spawn counter.** `spawn_draws: u64` on `SupervisorRun`. Every
  `safe_spawn_position` call in `supervisor_run.rs` (sites in `start` and
  `on_sample`) goes through one private helper that increments it first.
  `world/mod.rs` is untouched; resume replays the count on a fresh
  `WorldState::new(config.world)`.
- **`config: FullConfig`** is a new run field set by `start` and `resume`. It is
  not in the state struct; the checkpoint `config` section carries it.
- **Tick rule.** `tick()` is the last sample handled, so the captured
  `SupervisorRunState.tick` is the previous sample tick. The envelope tick `T`
  equals `next_sample_tick()` at write time. Resume requires `state.tick < T`
  and that the next sample tick derived from the restored state equals `T`.
  Do not set the run tick equal to `T`.
- **`checkpoint_due(&self, tick) -> bool`** is `self.store.is_some() && tick ==
  self.core.next_boundary_tick()`.
- **`SupervisorRunState`** derives `Serialize, Deserialize` and holds `core`
  (`SupervisorState`), `season` (the `TickSchedule` state from
  `supervisor-core-state`), `accumulator: Option<RunSummaryAccumulator>`,
  `population`, `kinds_on`, `nourishing: Option<FoodKind>`, `tick`, `budget`,
  `next_occupant`, `spawn_draws`, and every field 0006 added (names in
  `docs/plans/0006-supervisor-decision-telemetry/ARCHITECTURE.md`). `f32` values
  use `crate::supervisor_checkpoint::f32_bits`.
- **`capture_state(&self)`** destructures `SupervisorRun` with every field named
  and no `..`. Unowned fields (`world`, `store`, `config`) are bound and
  discarded explicitly, so a new field is a compile error.
- **`write_checkpoint(&mut self, tick, &KernelCheckpoint)`** returns
  `TickMismatch` unless `tick == next_sample_tick()`. Without a store it writes
  nothing and returns `Ok`. Otherwise it builds `Checkpoint { tick, sections }`
  with `config` (serde_json of `FullConfig`), `run` (serde_json of the state),
  one `kernel.<name>` per `KernelSection` bytes, and `kernel.cpu` equal to
  `KernelCpuState::to_bytes()` of `KernelCheckpoint.cpu`. It encodes and calls
  `SupervisorStore::write_checkpoint`. Serde failures map to
  `CheckpointError::Malformed`.
- **Errors.** Add `Checkpoint(CheckpointError)` and `KernelRestore(KernelRestoreError)`
  to `SupervisorRunError`, with `Display` arms.
- **`ResumePoint` and loaders.** `ResumePoint { run_id, tick, config, state,
  kernel, physics, food }`. `load_resume_point` reads via `read_checkpoint` (`None`
  gives `Missing`), decodes the envelope, and compares the `config` section with
  `run_configs()` (mismatch gives `ConfigMismatch`). It decodes `kernel.cpu` with
  `KernelCpuState::from_bytes` (error gives `Malformed`) into `KernelCheckpoint.cpu`
  and keeps the other `kernel.<name>` sections as `KernelSection`s. It checks that
  `T` is a multiple of `SUPERVISOR_SAMPLE_TICKS`, the tick rule above, and that the
  slot count equals `population_size`. `physics = KernelCheckpoint::physics(&self)
  -> &[f32]` copied to `Vec<f32>`; `food = KernelCheckpoint::food(&self) ->
  Option<&[f32]>` copied. Food must be `Some` exactly when `kinds_on`, else
  `Malformed`. `load_checkpoint_config` runs only the config part.
- **`resume(kernel, config, store, state, &KernelCheckpoint)`** runs `validate`,
  builds `WorldState::new`, stores `config`, replays `spawn_draws` calls, calls
  `upload_world` as `start` does, then `restore_checkpoint`, rebuilds the core with
  `Supervisor::restore`, restores `season` and `accumulator`, and rewrites every
  `supervisor_slot` row through `start`'s store call. It never calls
  `reset_agents_seeded`, `upload_agents` or `replace_agent_slot`.
- **Test file.** Each GPU test starts with the fail-closed adapter pattern.
  Config constants come from `tests/supervisor_headless.rs` (measurement on). The
  driver mirrors `supervisor_step`: at due ticks it uses `dispatch_ticks`, then
  `read_checkpoint_blocking`, then `write_checkpoint`, then `on_sample` with the
  checkpoint physics and food. Other ticks use `dispatch_batch`.

## Non-goals

- No edits to `world/mod.rs`, `supervisor.rs`, `supervisor_checkpoint.rs`,
  `supervisor_store.rs`, `gpu_kernel*.rs`, `headless.rs`, `main.rs` or
  `sim_runtime.rs`.
- No dependency or version change.
- No change to `start` behaviour or existing tests.
- No headless or worker resume entry point.

## Steps

**Steps:**

1. In `run_summary.rs`, derive `Serialize, Deserialize` on `RunSummaryAccumulator`
   and its parts. Use `f32_bits` for `f32`, and a private `f64_bits` module (stores
   `to_bits()` as `u64`) for `f64`. Change no existing method or test.
2. In `supervisor_run.rs`, add the `spawn_draws` and `config` fields, the spawn
   helper used at every call site, `SupervisorRunState`, `capture_state`,
   `checkpoint_due`, `write_checkpoint`, and the two `SupervisorRunError` variants
   with `Display` arms.
3. In `supervisor_run.rs`, add `ResumePoint`, `load_resume_point` and
   `load_checkpoint_config` with the checks in Decisions.
4. In `supervisor_run.rs`, add `SupervisorRun::resume`. Make `start` count its
   initial spawn draws through the helper.
5. Create `tests/supervisor_run_resume.rs` with
   `supervisor_run_resume_matches_uninterrupted` (two runs, the interrupted one
   stopped after its second checkpoint write and before `on_sample` at that tick,
   then `SupervisorStore::reopen_for_resume`; compare all `SampleReport`s from T
   on and every table row by rowid, including `supervisor_checkpoint`),
   `supervisor_run_resume_refuses_config_mismatch` (edit the run row's
   `brain_config` with rusqlite; expect `ConfigMismatch`), and
   `run_summary_accumulator_serde_round_trip`.
6. Make the `focused-proof`, `sandbox-clippy` and `sandbox-fmt` checks pass by
   editing only `touches`. When `sandbox-fmt` reports drift, apply its fix
   command, then re-run all three.

- **Done when:** the `focused-proof` check passes tests `supervisor_run_resume_matches_uninterrupted`, `supervisor_run_resume_refuses_config_mismatch` and `run_summary_accumulator_serde_round_trip`, and `sandbox-clippy` and `sandbox-fmt` pass, proving INV-7 (resumed run equals uninterrupted run in reports and store rows, and `capture_state` names every run field).
