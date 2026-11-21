---
id: headless-resume
title: Headless and CLI supervisor resume
workstream: "0300"
kind: task
role: developer
depends_on:
  - supervisor-run-resume
gated: false
touches:
  - crates/xagent-sandbox/src/headless.rs
  - crates/xagent-sandbox/src/main.rs
  - crates/xagent-sandbox/tests/supervisor_headless.rs
  - crates/xagent-sandbox/tests/evolution_cli_exit.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_headless
    timeout_ms: 3600000
  - name: cli-exit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test evolution_cli_exit
    timeout_ms: 3600000
  - name: cli-unit
    command: cargo test -p xagent-sandbox --bin xagent evolution_cli
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
---

# Headless and CLI supervisor resume

## Outcome

`run_headless_with_options` with `resume = true` in supervisor mode resumes the
checkpointed run from its stored `supervisor_checkpoint` row and continues the
same loop to completion (INV-10). With no checkpoint it fails with
`HeadlessError::Checkpoint(CheckpointError::Missing)` and leaves the database
untouched. The CLI no longer refuses `--resume` in supervisor mode. Generation-mode
resume is unchanged. Only the four `touches` files change.

## Prerequisites used (not reimplemented)

From `supervisor-run-resume` (and its transitive tasks `kernel-checkpoint`,
`checkpoint-envelope`, `checkpoint-store`, `supervisor-core-state`):

- `crate::supervisor_run::load_resume_point(db_path) -> Result<ResumePoint,
  CheckpointError>` with `ResumePoint { run_id, tick, config: FullConfig, state:
  SupervisorRunState, kernel: KernelCheckpoint, physics: Vec<f32>, food:
  Option<Vec<f32>> }`.
- `SupervisorRun::resume(kernel, config, store, state, &KernelCheckpoint) ->
  Result<Self, SupervisorRunError>`, `SupervisorRun::checkpoint_due(tick) -> bool`,
  `SupervisorRun::write_checkpoint(tick, &KernelCheckpoint)`.
- `SupervisorStore::reopen_for_resume(db_path, run_id, tick)`.
- `GpuKernel::read_checkpoint_blocking() -> KernelCheckpoint` (returns physics
  and food rows) and `GpuKernel::new` from a `FullConfig`.
- `crate::supervisor_checkpoint::CheckpointError`. Its `Display` for `Missing`
  contains the word "checkpoint", and every variant's `Display` names the database
  path's run and the reason (this is the envelope task's contract).

## Decisions (explicit)

- **Absent file is Missing, before any open.** In the supervisor resume branch,
  if the database path does not exist (`!Path::new(db).exists()`), return
  `HeadlessError::Checkpoint(CheckpointError::Missing)` immediately. The file must
  not be created. This keeps the existing "no database was touched" property.
- **Order.** Load and decode the resume point before building `GpuKernel`, so a
  missing or refused checkpoint never dispatches or creates a kernel.
- **Config.** The checkpoint `config` is used. If the supplied config differs
  from `resume.config` (compare `serde_json::to_string` of both `FullConfig`s),
  print exactly one `eprintln!` line saying the checkpoint config is used. Nothing
  else changes.
- **Stop option.** `HeadlessOptions.stop_after_checkpoint_at_or_after:
  Option<u64>`, default `None`. When `Some(v)`, the loop returns `Ok(())` right
  after `write_checkpoint` for the first checkpoint whose tick is `>= v`. It does
  not call `on_sample` for that tick and does not call `finish` or write the
  summary. The checkpoint is pre-sample, so a resume replays `on_sample(T)`.
- **Due ticks.** At a due tick `supervisor_step` calls `kernel.dispatch_ticks`
  (not `dispatch_batch`), then `read_checkpoint_blocking`, then
  `run.write_checkpoint`, then `run.on_sample` with the checkpoint physics and food.
  Other ticks are unchanged. No extra submit or compute pass (INV-4).
- **Error mapping.** `HeadlessError` gets `Checkpoint(CheckpointError)` with a
  `From<CheckpointError>` impl. `Display` delegates to the wrapped error. The
  `ResumeInSupervisorMode` variant and its `Display` arm are deleted.
- **CLI exit.** `main.rs` already exits 1 with `Error: {e}` on any `Err` from
  `run_headless_with_options` (`main.rs` around line 805). Nothing else changes in
  `main.rs` exit handling.

## Non-goals

- No change to `sim_runtime.rs`, `gpu_orchestration.rs`, `supervisor_run.rs`,
  `supervisor_checkpoint.rs`, `supervisor_store.rs` or any `gpu_kernel` file.
- No change to generation-mode resume (`Governor::resume` and its refusal test
  stay; plan 0010 deletes them).
- No checkpoint history, no fallback to an older state, no new dependency.
- No GUI entry point.

## Steps

**Steps:**

1. In `headless.rs`, delete `HeadlessError::ResumeInSupervisorMode` and its
   `Display` arm. Add `Checkpoint(CheckpointError)` with a `From` impl and a
   `Display` that delegates. Add `stop_after_checkpoint_at_or_after: Option<u64>`
   to `HeadlessOptions`; `HeadlessOptions::default()` must still build.
2. In `headless.rs`, replace the supervisor refusal branch of
   `run_headless_with_options` with the resume path from Decisions: absent-file
   check, `load_resume_point`, the one-line config notice, `GpuKernel::new` from
   the checkpoint config, `SupervisorStore::reopen_for_resume`,
   `SupervisorRun::resume`, `on_sample(T)` with the checkpoint physics and food,
   then the existing loop. Generation-mode code paths stay as they are.
3. In `headless.rs`, change `supervisor_step` (pass `options` in if needed) to the
   due-tick sequence in Decisions, including the stop check right after
   `write_checkpoint`. Other ticks stay the same.
4. In `main.rs`, delete the `supervisor && cli.resume` refusal in
   `apply_evolution_cli`. Rename the unit test
   `evolution_cli_resume_with_supervisor_is_error` to
   `evolution_cli_resume_with_supervisor_resolves` and make it assert `is_ok()`.
   Keep every other check in that function. Keep the `supervisor` variable in use.
5. In `tests/supervisor_headless.rs`, replace `supervisor_headless_refuses_resume`
   with `supervisor_headless_resume_without_checkpoint_is_missing` (nonexistent
   path in a tempdir, `resume = true`; expect
   `Err(HeadlessError::Checkpoint(CheckpointError::Missing))` and the path still
   does not exist). Add `supervisor_headless_stop_and_resume_finishes`: run with
   `stop_after_checkpoint_at_or_after: Some(first checkpoint tick)` and
   `resume = false` and expect `Ok` with no summary file; then run with
   `resume = true` and default options and expect `Ok` and a parsing
   `RunSummary` with mode Supervisor. Both GPU tests use the same `gpu_or_skip()`
   guard as the other tests in that file. In `tests/evolution_cli_exit.rs`, update
   `evolution_cli_exit_resume_supervisor` to expect exit code 1, stderr containing
   `Error:` and the lowercase word `checkpoint`, and no database file created.
6. Request `focused-proof`, `cli-exit`, `cli-unit`, `sandbox-clippy` and
   `sandbox-fmt` through `verify`. If `sandbox-fmt` fails, apply its fix and
   request all five again. Edit nothing outside `touches`.

- **Done when:** the `focused-proof`, `cli-exit` and `cli-unit` checks pass, including tests `supervisor_headless_resume_without_checkpoint_is_missing`, `supervisor_headless_stop_and_resume_finishes`, `evolution_cli_exit_resume_supervisor` and `evolution_cli_resume_with_supervisor_resolves`, proving INV-10 (supervisor `--resume` resumes from the stored checkpoint, and with no checkpoint fails with `CheckpointError::Missing` without modifying the database).
