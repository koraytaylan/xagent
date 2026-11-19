---
id: gui-supervisor-wiring
title: "GUI start, events and snapshot for supervisor mode"
workstream: "0600"
kind: task
role: developer
depends_on: [worker-supervisor]
gated: false
touches: [crates/xagent-sandbox/src/ui.rs, crates/xagent-sandbox/src/evolution.rs, crates/xagent-sandbox/src/gpu_orchestration.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --lib ui::tests::supervisor_"}
  - {name: gui-bin-tests, command: "cargo test -p xagent-sandbox --bin xagent -- gui_supervisor_"}
  - {name: snapshot-default, command: "cargo test -p xagent-sandbox --test integration -- --exact evolution_snapshot_default_tree_pane_fraction"}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# GUI start, events and snapshot for supervisor mode

## Outcome

GUI Start picks the supervisor driver from `evolution_mode`. The main thread only
receives `SimEvent::SupervisorSample` and updates `EvolutionSnapshot.supervisor`;
it never touches the kernel. Generation mode is unchanged.

## Scope

- `ui.rs`:
  - `pub struct SupervisorSnapshot { pub tick: u64, pub total_ticks: Option<u64>,
    pub boundaries: u64, pub replacements: u64, pub last_replacement: Option<String>,
    pub archive_len: usize, pub best_outcome: Option<f32>, pub nourishing_kind:
    Option<FoodKind>, pub finished: bool }`, deriving `Default, Clone, Debug`.
  - `pub fn apply(&mut self, report: &crate::supervisor_run::SampleReport)`: sets
    `tick` and `finished`. A report with `boundary` Some counts one boundary and
    sets `archive_len` and `best_outcome`. A boundary with `slot` Some also counts
    a replacement and sets `last_replacement = Some(format!("slot {slot}: {reason:?}"))`.
    A `season` Some sets `nourishing_kind`. `apply` never changes `total_ticks`.
  - `EvolutionSnapshot` gains `pub supervisor: Option<SupervisorSnapshot>`, `None` in `Default`.
  - `render_running_dashboard`: when `Some`, a compact supervisor section (tick over
    total, replacements, last replacement, archive size and best outcome, nourishing
    kind) replaces the generation progress.
  - `render_config_editor`: radio buttons Generations and Supervisor bound to
    `evo.edit_governor.evolution_mode`.
  - Add a new `#[cfg(test)] mod tests`; keep the existing `vision_cap_tests`.
- `evolution.rs`, `EvolutionAction::Start`: with `edit_governor.evolution_mode ==
  Supervisor`, skip `Governor::new_seeded`. Set `governor_config`, `brain_config`,
  `evo_snapshot.state = Running` and `evo_snapshot.supervisor = Some(SupervisorSnapshot
  { total_ticks, ..Default::default() })`, where `total_ticks` is
  `Some(budget / population)` when `supervisor.agent_tick_budget > 0` and population
  is positive (both as `u64`), else `None`. Keep the clock and pause resets, then
  `stop_sim_worker`, `spawn_evolution_population` (governor `None`) and
  `start_sim_worker`. Generation Start is unchanged. Reset needs no edit: its
  `EvolutionSnapshot::default()` clears `supervisor`.
- `gpu_orchestration.rs`:
  - Reword the module doc on line 9, which names `GpuKernel`. The isolation test
    below would otherwise match it.
  - `fn supervisor_worker_init(governor: &GovernorConfig, brain: &BrainConfig,
    world: &WorldConfig, db_path: &str, session_active: bool) ->
    Option<SupervisorWorkerInit>`: `None` unless `governor.evolution_mode ==
    Supervisor && session_active`. Otherwise `Some` with `config: FullConfig { brain,
    world, governor }` (clones) and `db_path: Some(db_path.to_string())`.
  - `start_sim_worker`: `session_active = self.evo_snapshot.supervisor.is_some()`.
    In supervisor mode, pass `tick_budget: 0`, `search_steering_genome: false` and
    `supervisor: Some(..)`. Otherwise keep the current values and pass `supervisor: None`.
  - Replace the empty `SimEvent::SupervisorSample(_)` arm. Apply the report to
    `evo_snapshot.supervisor` when it is `Some`. Log `[SUPERVISOR]` with the slot when
    the report has a boundary slot. Set `snap_dirty = true`.
  - Add a `#[cfg(test)] mod tests` with the two tests below.

## Non-goals

- No `sim_runtime.rs` change, renderer change, generation-mode behaviour change or
  new persistence. No new `App` field: the session flag is
  `evo_snapshot.supervisor.is_some()`.
- Do not write the forbidden needles in any comment in the four included files.

## Prerequisites

- `worker-supervisor`: `SupervisorWorkerInit`, `SimInit.supervisor` and
  `SimEvent::SupervisorSample` exist. It left `supervisor: None` and an empty arm in
  `gpu_orchestration.rs`, which this task replaces.
- `supervisor-run`: `SampleReport` and `BoundaryReport` are reachable as
  `crate::supervisor_run` from the library, with `ReplaceReason` implementing `Debug`.
- Plan 0004: `xagent_shared::FoodKind`.

## Decisions

- `SupervisorSnapshot::apply` counts exactly as the Scope states. The replacement
  text is the only string format.
- Store-creation failures surface through the worker's existing `SimEvent::Error`
  path, not in this task.
- `ui.rs` tests and the bin tests use the names in Done when, and nothing else.

## Steps

**Steps:**
1. In `ui.rs`, add `SupervisorSnapshot` and `apply`, the `EvolutionSnapshot` field
   and its `None` default. In a new `mod tests`, add `supervisor_snapshot_default_is_none`
   and `supervisor_snapshot_apply_counts_reports`. The latter feeds a sequence of
   reports: a non-boundary report, a season report, a replacement with archive
   length and best outcome, and a finished report. It asserts each field.
2. In `ui.rs`, add the supervisor dashboard section and the mode radio selector
   with existing palette and layout constants.
3. In `evolution.rs`, add the supervisor branch to `EvolutionAction::Start` as
   Scope states.
4. In `gpu_orchestration.rs`, reword line 9, add `supervisor_worker_init`, and set
   the supervisor fields in `start_sim_worker`.
5. In `gpu_orchestration.rs`, replace the `SupervisorSample` arm as Scope states.
6. In `gpu_orchestration.rs`, add `mod tests` with `gui_supervisor_worker_init_only_in_supervisor_mode`
   (Generations gives `None`; Supervisor with the session inactive gives `None`;
   Supervisor with it active gives `Some` with the mode and db path) and
   `gui_supervisor_main_thread_never_touches_kernel`. The latter `include_str!`s
   `gpu_orchestration.rs`, `evolution.rs`, `snapshot.rs` and `app.rs`, and asserts
   each contains none of `concat!("GpuKernel")`, `concat!("SupervisorRun", "::")`,
   `concat!("read_full_state", "_blocking")` and `concat!("on_", "sample(")`.

- **Done when:** check focused-proof reports ok for `ui::tests::supervisor_snapshot_default_is_none` and `ui::tests::supervisor_snapshot_apply_counts_reports`; check gui-bin-tests reports ok for `gui_supervisor_worker_init_only_in_supervisor_mode` and `gui_supervisor_main_thread_never_touches_kernel`; check snapshot-default passes; together they prove INV-12 (GUI Start picks the driver by mode, and `EvolutionSnapshot::default().supervisor` is `None`) and INV-13 (the four main-thread files never name the kernel, supervisor run or readback, and only `SupervisorSample` events reach the snapshot).

## Evidence and risks

- The worker has no shell. The host-run checks are the proof.
- If `SampleReport`, `BoundaryReport` or `ReplaceReason` differ from Prerequisites
  (visibility, `Debug`, field names), report the file and line rather than
  renaming landed items.
- If `include_str!` of a listed file fails to compile or the isolation test matches
  any needle, report the file and line. Do not weaken the needle list.
