---
id: session-wiring
title: Supervisor session startup, browse and resume wiring and live event fold
workstream: "0400"
kind: task
role: developer
depends_on:
  - observatory-panel
  - observatory-store-read
gated: false
touches:
  - crates/xagent-sandbox/src/app.rs
  - crates/xagent-sandbox/src/evolution.rs
  - crates/xagent-sandbox/src/gpu_orchestration.rs
status: done
merged_as: "8b67336c7b33f74a4f4bde4a8c9e9b860821952f"
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --bin xagent -- observatory_session_
    timeout_ms: 3600000
  - name: kernel-isolation
    command: cargo test -p xagent-sandbox --bin xagent -- gui_supervisor_
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --check --edition 2021 crates/xagent-sandbox/src/app.rs crates/xagent-sandbox/src/evolution.rs crates/xagent-sandbox/src/gpu_orchestration.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/src/app.rs crates/xagent-sandbox/src/evolution.rs crates/xagent-sandbox/src/gpu_orchestration.rs
  - name: sandbox-guard
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
---

# Supervisor session startup, browse and resume wiring and live event fold

## Outcome

The bin crate (`main.rs` declares `app`, `evolution`, `gpu_orchestration`) detects a
supervisor run before any read-write open, offers Resume live or Browse read-only
through the session state, folds live plan 0006 telemetry into
`evo_snapshot.observatory`, and keeps database loads out of the event drain.

## Decisions (binding)

- **Plan-level prerequisites.** Binding as written in ARCHITECTURE.md "Session wiring decisions" (Plan-level prerequisites).
- **Edit only the three touched files.** No new dependency. No planning, task or INV
  identifier, `TODO`, `FIXME` or emoji in edited files.
- **Paths.** The bin uses `xagent_sandbox::observatory::store_read::{detect_supervisor_run,
  load_run, BrowseError}` and `xagent_sandbox::observatory::{ModelSource,
  ObservatoryModel, ObservatoryRecord}`, as `app.rs` already uses `xagent_sandbox::governor`.
- **`enum StartupSession { Supervisor(SupervisorRunSummary),
  Generation((u32, GovernorConfig, BrainConfig)), None }`** in app.rs. `Generation`
  carries `check_existing_session`'s tuple, so that function is called once.
- **`fn startup_session(db_path: &str) -> StartupSession`** (free fn in app.rs):
  1. If `Path::new(db_path).exists()` is false, return `None` with no open.
     `check_existing_session` calls `Connection::open`, which would create the file.
  2. `detect_supervisor_run(db_path)`: `Ok(Some(s))` gives `Supervisor(s)`; `Err(e)`
     logs `e` with `println!` and gives `None` without calling
     `check_existing_session`.
  3. `Ok(None)`: `check_existing_session` gives `Generation(t)` or `None`.
- **`App::new`** matches on it. `Supervisor` sets state `SupervisorSession` and
  `supervisor_session = Some(s)`, and skips `Governor::resume`. `Generation` runs the
  existing `HasSession` block unchanged. `None` sets `Idle`.
- **Resume input.** `App` gains an `Option` field typed with the plan 0007 resume
  input, `None` in `App::new`. `SessionChoice::ResumeLive` sets it. `start_sim_worker`
  (gpu_orchestration.rs) takes it with `Option::take` and passes it to
  `supervisor_worker_init`, which sets the plan 0007 field. With no input the
  `SimInit` is unchanged.
- **Actions (evolution.rs).** No new `EvolutionAction` variant exists. At the
  top of `handle_evolution_action` (called every frame from `main.rs`), take
  `evo_snapshot.session_choice` with `std::mem::take` and act on it:
  - `SessionChoice::Browse`: `load_run` for the run id in `supervisor_session`, then
    `apply_browse_supervisor(&mut evo_snapshot, model)`. That helper sets
    `observatory = Some(model)`, state `BrowsingSupervisor`, and clears `supervisor`,
    `supervisor_session` and `supervisor_resume_error`. No worker starts. On `Err(e)`,
    set `supervisor_resume_error` to `e.to_string()` truncated to 256 bytes and stay
    in `SupervisorSession`.
  - `SessionChoice::ResumeLive`: `load_run` as above, set `model.source = ModelSource::Live`,
    set `observatory`, set `supervisor` as the Start arm does, clear
    `supervisor_session`, set the App resume field, set state `Running`, call
    `start_sim_worker`. A synchronous refusal (error from plan 0007's start path or
    from `start_sim_worker`) calls `apply_resume_refusal(&mut evo_snapshot, msg,
    summary)`: it logs `msg`, clears `supervisor` and `observatory`, sets
    `supervisor_resume_error` to `msg` truncated to 256 bytes, restores
    `supervisor_session = Some(summary)` and state `SupervisorSession`. A
    `SimEvent::Error` before the first `SupervisorSample` runs the same helper and drops
    the runtime as Reset does.
  - Supervisor Start: `observatory = Some(m)` with `m = ObservatoryModel::default()`
    and `m.source = ModelSource::Live`.
- **Live fold.** Binding as written in ARCHITECTURE.md "Session wiring decisions" (Live fold).
- **Needles.** The kernel needle test stays unchanged.
- **Helpers.** `apply_browse_supervisor` and `apply_resume_refusal` are free fns in
  evolution.rs, testable without a worker.

## Tests

All names are prefixed `observatory_session_`, in the file under test.

- app.rs `mod tests`:
  - `observatory_session_startup_supervisor_db_read_only`: tempfile DB built by the
    plan 0006 run writer, writer connection dropped, main-file bytes read, then
    `startup_session` returns `Supervisor`. Bytes equal afterwards. The DB must not be
    in WAL mode, since only the main file is compared.
  - `observatory_session_startup_missing_db_not_created`: `startup_session` on a
    missing tempdir path gives `None`, and the path still does not exist.
  - `observatory_session_startup_generation_db_unchanged_behaviour`: generation DB
    built with the existing `Governor::create` fixture gives `Generation` matching
    `check_existing_session`.
- evolution.rs `mod tests`:
  - `observatory_session_browse_sets_state_without_worker`: after
    `apply_browse_supervisor`, state `BrowsingSupervisor`, `supervisor` and
    `supervisor_session` None, `observatory` Some with source Browse.
  - `observatory_session_resume_refusal_keeps_dialog`: after `apply_resume_refusal` with
    a message over 256 bytes, state `SupervisorSession`, `supervisor` and
    `observatory` None, error at most 256 bytes, summary restored.
- gpu_orchestration.rs `mod tests`:
  - `observatory_session_orchestration_never_loads_store`: `include_str!` of this file
    must not contain `concat!("store_read", "::")`. The needle appears only through
    `concat!`, never as a literal in a scanned file.

**Steps:**

1. Read the landed plan 0006 and 0007 names in Decisions and the `SupervisorRunSummary`
   fields from `observatory/store_read.rs`; report any absent name as a blocker. Then
   edit app.rs: `StartupSession`, `startup_session`, the `App::new` match, and the
   resume field.
2. Edit evolution.rs: the `session_choice` handling for Browse and ResumeLive, the two
   helpers, and the Supervisor Start `observatory` initialisation.
3. Edit gpu_orchestration.rs: `start_sim_worker` passes the resume input into
   `supervisor_worker_init`, and a pre-sample `SimEvent::Error` takes the refusal path.
4. Edit gpu_orchestration.rs: add `observatory_record` and the event-drain fold, keeping
   the `SupervisorSample` arm.
5. Add the tests in Tests to app.rs, evolution.rs and gpu_orchestration.rs, with the
   names as written.
6. Remove unused imports. Check edited files for planning, task or INV identifiers,
   `TODO`, `FIXME` and emoji, and that the store needle appears only inside `concat!`.

- **Done when:** `focused-proof` passes with `observatory_session_startup_supervisor_db_read_only`, `observatory_session_startup_missing_db_not_created`, `observatory_session_startup_generation_db_unchanged_behaviour`, `observatory_session_browse_sets_state_without_worker`, `observatory_session_resume_refusal_keeps_dialog` and `observatory_session_orchestration_never_loads_store` reported ok, and `kernel-isolation` passes with `gui_supervisor_main_thread_never_touches_kernel` reported ok. Together these prove INV-3 (startup part), INV-4 (main-thread part) and INV-12 (startup choice, browse starts no worker, refusal keeps the dialog).
