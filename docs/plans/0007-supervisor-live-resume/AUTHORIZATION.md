# Plan amendment authorization record

## A1 — kernel-checkpoint: map_async failure under the infallible readback signature

- **Superseded requirement:** ARCHITECTURE.md (GPU readback) and
  tasks/0101-kernel-checkpoint.md step 4 froze
  `GpuKernel::read_checkpoint_blocking(&mut self) -> KernelCheckpoint` without
  saying what a terminal `map_async` failure does. The review read
  CONTRIBUTING.md (Concurrency, Async Readback) as requiring a Result/Option
  return, which the frozen signature rules out.
- **Contradiction:** the infallible signature, which 0201, 0301 and 0302 consume,
  leaves no non-panicking outcome for a terminal map failure. A retry can fail
  again, and returning cached or partial state would break INV-5.
- **Source-backed correction:** keep the frozen signature and every consumer
  record unchanged. Record an explicit exception: the function captures the map
  result, and on failure logs it with `log::error!` and then panics. This matches
  the existing `read_full_state_blocking` path
  (crates/xagent-brain/src/gpu_kernel.rs, `map_async` followed by
  `poll(...).panic_on_timeout()`). Retrying, returning cached or partial data,
  and substituting values stay forbidden. Step 4 and ARCHITECTURE.md now say so.
  No other requirement, check or task changes.
- **Executable witness:** the `focused-proof` check
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test kernel_checkpoint`)
  plus `brain-clippy` and `brain-fmt`, unchanged. The failure arm is checked by
  review against the step 4 wording (logged error, then panic, no substitution).

## A2 — supervisor-run-resume: season state, slot rowids and the config read

- **Superseded requirements (tasks/0201-supervisor-run-resume.md):** (a) Decisions
  say `SupervisorRunState` holds `season` (the `TickSchedule` state from
  supervisor-core-state) while Non-goals forbid any edit to `supervisor.rs`;
  (b) step 5 says to compare every table row by rowid, while Decisions say resume
  rewrites every `supervisor_slot` row through `start`'s store call; (c) Decisions
  say `load_resume_point` compares the config section with `run_configs()`.
- **Contradictions:** (a) `TickScheduleState` and its `capture`/`rebuild`
  (crates/xagent-sandbox/src/supervisor.rs:1265-1292) are private and
  `TickSchedule`'s fields are private, so the required state cannot be named
  outside `supervisor.rs`; the review rightly rejected a replayed advance
  counter. (b) `upsert_slot` writes with `INSERT OR REPLACE`
  (supervisor_store.rs), which assigns fresh rowids on every rewrite, so resumed
  slot rowids cannot equal the uninterrupted run's (observed 12 vs 22).
  (c) `run_configs(&self)` needs a `SupervisorStore`, whose only constructors
  `create` and `reopen_for_resume` write to the database, which the frozen
  `load_resume_point(db_path)` loader must not do.
- **Source-backed correction:** (a) add `supervisor.rs` to the task's touches,
  limited to making `TickScheduleState`, `capture` and `rebuild` `pub(crate)`;
  `season: TickScheduleState` is now spelled out and a counter replay is stated
  as insufficient. (b) `supervisor_slot` is compared ordered by its
  `(run_id, slot)` key with every non-rowid column equal; every other table
  stays compared by rowid, and the only other mask is the `run` row's
  `started_at`. The store is unchanged. (c) the loader compares against the same
  three run-row columns `run_configs()` returns, read through a read-only
  `rusqlite` connection with `busy_timeout` set; it must not open a store or
  mutate the database. The review finding demanding `run_configs()` itself is
  the superseded reading. Requirements to capture every 0006 field (including
  `last_frame`) are unchanged. No task id, dependency, gated flag or check changes.
- **Executable witness:** the unchanged `focused-proof` check
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_run_resume`),
  plus `sandbox-clippy` and `sandbox-fmt`, which already cover `supervisor.rs`
  as part of package `xagent-sandbox`.

## A3 — headless-resume: the generation_run_summary.rs struct literal

- **Superseded requirement:** tasks/0301-headless-resume.md listed four touches
  and said "Only the four `touches` files change" and "Edit nothing outside
  `touches`"; the review read this as forbidding the edit to
  `tests/generation_run_summary.rs`.
- **Contradiction:** step 1 requires adding `stop_after_checkpoint_at_or_after`
  to the public struct `HeadlessOptions`. The integration test
  crates/xagent-sandbox/tests/generation_run_summary.rs:55-57 builds
  `HeadlessOptions { run_summary: Some(..), }` with no rest pattern, so any new
  field fails with E0063, and `sandbox-clippy` (`--all-targets`) compiles that
  target. No design inside the four touches satisfies both.
- **Source-backed correction:** add `tests/generation_run_summary.rs` to the
  task's touches, limited to adding `..HeadlessOptions::default()` to that one
  literal (no behaviour or assertion change). The Outcome and step 1 say so.
  No task id, dependency, gated flag, check, decision or other requirement changes.
- **Executable witness:** the unchanged `sandbox-clippy` check
  (`cargo clippy -p xagent-sandbox --all-targets -- -D warnings`), which compiles
  that test target, plus `sandbox-fmt`, `focused-proof`, `cli-exit` and `cli-unit`.

## A4 — resume-digest-gate: food-row presence check in load_resume_point

- **Superseded requirement:** tasks/0201-supervisor-run-resume.md said
  `load_resume_point`'s food rows "must be `Some` exactly when `kinds_on`, else
  `Malformed`", and tasks/0401-resume-digest-gate.md forbade any product change.
- **Contradiction:** `KernelCheckpoint::food()` is `Some` whenever the kernel has
  food (`captured_buffers` keeps `food_state` when `food_count > 0`,
  crates/xagent-brain/src/gpu_kernel/checkpoint.rs:321), as ARCHITECTURE.md
  (lines 15, 26, 47-48) requires; the uninterrupted headless run already passes
  those Some rows to `on_sample` with kinds off (headless.rs:347). With 0006's
  required seeded config (food present, `food_kinds_enabled` default false,
  crates/xagent-shared/src/config.rs:1164) every checkpoint is refused at
  supervisor_run.rs:515 with `Malformed("food rows do not match the food kinds
  setting")`, so INV-1's stop-and-resume equality cannot pass with only the test file.
- **Source-backed correction:** food must be `Some` exactly when the config's
  world has food items (`WorldState::new(config.world.clone()).food_items.len() > 0`),
  else `Malformed`; `kinds_on` plays no part. 0201's wording is corrected;
  0401 adds `crates/xagent-sandbox/src/supervisor_run.rs` to its touches, limited to
  that one check (step 1), and its `sandbox-fmt` check now also formats that
  file (baseline `rustfmt --edition 2021 --check` on it passes; it declares only
  inline modules). Config, golden, digest, refusal tests, task ids, dependencies,
  gated flags and all other requirements are unchanged.
- **Executable witness:** `focused-proof`
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_resume_digest`)
  with `resume_digest_matches_uninterrupted` passing, plus `sandbox-clippy` and the
  widened `sandbox-fmt`.
