---
id: worker-session-supervisor-only
title: "Worker and main-thread session run only the supervisor"
workstream: "0200"
kind: task
depends_on:
  - integration-tests
  - store-legacy-guard
gated: false
role: developer
touches:
  - crates/xagent-sandbox/src/sim_runtime.rs
  - crates/xagent-sandbox/src/gpu_orchestration.rs
  - crates/xagent-sandbox/src/app.rs
  - crates/xagent-sandbox/src/evolution.rs
  - crates/xagent-sandbox/src/snapshot.rs
status: done
merged_as: "0cdbe14612c9fc8616dea9cda7f141f86dcc33b3"
checks:
  - name: bin-unit
    command: "cargo test -p xagent-sandbox --bin xagent"
    timeout_ms: 3600000
  - name: session-generation-free
    command: "sh -c 'cd crates/xagent-sandbox/src && ! grep -nE \"GenerationBudgetReached|EvolutionMode|evolution_mode|pending_generation|PendingGeneration|generation_epoch|crate::governor|xagent_sandbox::governor|reset_database|check_existing_session|ResetPopulation|InheritedBrain|max_generations|elitism_count|eval_repeats|num_islands|migration_interval\" app.rs evolution.rs gpu_orchestration.rs sim_runtime.rs snapshot.rs'"
  - name: all-targets-compile
    command: cargo check --workspace --all-targets
    timeout_ms: 3600000
---

# 0201 Worker and main-thread session run only the supervisor

## Outcome

The worker and the main-thread session carry no generation handoff. Start
always launches a supervisor session. A legacy generations database is
classified read-only before any open and refused with the store's exact
message; the GUI offers only "Quit" or "Start in a new file". Reset needs a
two-step confirmation. Plan 0007 resume and plan 0008 observatory wiring stay.

## Decisions

- **Scope of this task.** Only the five touched files change. `ui.rs`,
  `ui_chrome.rs`, `render_pipeline.rs`, `replay_coord.rs` and the
  `App.recording`/`App.last_recording` fields belong to `ui-replay-supervisor-only`;
  `main.rs` belongs to `cli-supervisor-only`. An `EvolutionAction` or
  `EvolutionState` variant that `ui.rs` still declares but that served only
  generations keeps a match arm here that logs one line and does nothing; the
  UI task deletes the variant and the arm. `governor.rs` is not edited.
- **Removed-config fields.** Stop every read and write of the eight
  `GovernorConfig` fields `tick_budget`, `elitism_count`, `max_generations`,
  `patience`, `eval_repeats`, `num_islands`, `migration_interval`,
  `evolution_mode` and of `EvolutionMode` in these files; `config-generation-fields`
  deletes them later.
- **Worker (`sim_runtime.rs`, `gpu_orchestration.rs`).** Delete
  `SimEvent::GenerationBudgetReached`, `clamp_ticks_to_generation_budget` and its
  tests, `generation_clamps`, the generation epoch and once-per-generation
  fields, `ResetRequest`, `SimCommand::ResetPopulation`, `InheritedBrain`,
  `inherited_brain_fits`, the champion read-back command and its handling, and
  every `SimInit` field read only by generations. Keep the supervisor path,
  `SUPERVISOR_SAMPLE_TICKS`, `clamp_ticks_to_supervisor_sample`, the
  `sync_channel` capacities and latest-wins `partition_events` without its epoch
  argument (it keeps only the newest `SimEvent::Snapshot`). Delete
  `partition_events_preserves_control_order_and_budget_snapshot` and
  `partition_events_drops_snapshots_from_other_epochs`; keep
  `partition_events_keeps_only_newest_snapshot` and
  `partition_events_handles_no_snapshot` (INV-9). `supervisor_worker_init` drops
  the `evolution_mode` check and its generation test case and still returns
  `None` unless `session_active`.
- **Session (`app.rs`, `evolution.rs`, `snapshot.rs`).** Delete
  `PendingGeneration`, `App.pending_generation`, `App.generation_epoch`,
  `App.governor`, the generation Resume, Reset and handoff paths and the
  `check_existing_session` offer. Start always builds the supervisor session.
- **Legacy database.** Before any open of the database, on App start and before
  every Start or Resume, call
  `xagent_sandbox::supervisor_store::inspect_database(path)`. Pure helper
  `fn session_offer(kind: DatabaseKind) -> SessionOffer` with `enum SessionOffer
  { Open, LegacyChoice, Refuse }` in `evolution.rs`: `Missing`, `Empty` and
  `Supervisor` give `Open`; `LegacyGenerations { .. }` gives `LegacyChoice`; an
  inspection `Err` gives `Refuse`. `LegacyChoice` stores the body
  `StoreError::LegacyGenerations { .. }.to_string()` exactly in the snapshot for
  the modal with "Quit" and "Start in a new file"; "Start in a new file" sets the
  database path to `next_fresh_db_path(path, |p| p.exists())` and starts there
  through `SupervisorStore::create`. `pub fn next_fresh_db_path(path: &Path,
  exists: impl Fn(&Path) -> bool) -> PathBuf` returns `<stem>.supervisor.db`,
  then `<stem>.supervisor-2.db`, `<stem>.supervisor-3.db` and so on in the same
  directory, the first for which `exists` is false. The legacy file is never
  opened writable, renamed or deleted.
- **Reset.** Only for supervisor databases, unavailable for `LegacyChoice` and
  `Refuse`. The first request sets `ResetState::Armed(path)`; only a second
  request with the same path deletes the file with `std::fs::remove_file`. Pure
  helper `fn reset_confirmed(state: &ResetState, path: &Path) -> bool` returns
  true only for `Armed` with the same path. No call to `governor::reset_database`.
- **Unit tests (named, must pass):** `evolution::tests::session_offer_legacy_is_quit_or_fresh_only`
  (INV-2), `evolution::tests::next_fresh_db_path_skips_existing_siblings`,
  `evolution::tests::reset_needs_two_confirmations_on_same_path` (INV-3) and
  `sim_runtime::tests::partition_events_keeps_only_newest_snapshot` (INV-9).

**Steps:**

1. In `sim_runtime.rs` and `gpu_orchestration.rs` delete the worker members
   listed in Decisions, drop the epoch argument of `partition_events`, remove the
   generation-only unit tests and make `supervisor_worker_init` ignore
   `evolution_mode`.
2. In `app.rs`, `evolution.rs` and `snapshot.rs` delete `PendingGeneration`,
   `pending_generation`, `generation_epoch`, `App.governor`, the generation
   Resume, Reset and handoff code and every `governor` import; make Start launch
   only the supervisor session.
3. In `evolution.rs` and `app.rs` add `session_offer`, `SessionOffer`,
   `next_fresh_db_path`, `ResetState` and `reset_confirmed`, call
   `inspect_database` before every open and implement the legacy choice and the
   two-step Reset as in Decisions.
4. Turn the arms for UI variants that only served generations into logged
   no-ops, and add the three named `evolution.rs` tests.
5. Remove unused imports, add no `#[allow(dead_code)]` and no planning
   reference in source.

- **Done when:** `bin-unit` passes with `session_offer_legacy_is_quit_or_fresh_only`, `next_fresh_db_path_skips_existing_siblings`, `reset_needs_two_confirmations_on_same_path` and `partition_events_keeps_only_newest_snapshot` reported ok, and `session-generation-free` and `all-targets-compile` pass, proving the worker and session carry no generation handoff (INV-1), refuse a legacy database untouched (INV-2), delete a database only after two confirmations (INV-3) and keep the bounded latest-wins channels (INV-9).
