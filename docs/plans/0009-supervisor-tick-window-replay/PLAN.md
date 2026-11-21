---
objective: >-
  Give supervisor runs a tick-window replay that replaces per-generation replay.
  Scope: capture a bounded rolling window of cadence-sampled per-agent state
  (the kind of record replay.rs TickRecord already holds: positions, vitals,
  motor output, phase, vision frame where recorded) from the snapshots the
  worker already publishes, so capture adds no GPU dispatch or readback and
  cannot alter a seeded run's decisions; the plan 1 supervisor digest must be
  identical with capture on and off. Retention is bounded in memory and on disk
  by an explicit, named tick window; windows around replacement boundaries and
  season flips can be pinned so the events shown in the boundary inspector and
  timeline can be replayed. Playback view in the GUI: scrub and play a recorded
  window, select an agent, jump to a window from a timeline event or boundary,
  for both a live run and a database browsed read-only. Build it as new
  supervisor replay code alongside the existing per-generation replay
  (replay.rs, replay_coord.rs, the generation_recording table), reusing shared
  record types where that does not tie it to generations; the per-generation
  replay itself is deleted by generation-mode-removal.
depends_on:
  - 0006-supervisor-decision-telemetry
  - 0008-supervisor-observatory-ui
status: completed
toolchains:
  - cargo
  - rustc
  - rustup
  - cargo-fmt
  - rustfmt
  - cargo-clippy
  - clippy-driver
checks:
  - name: fmt
    command: cargo fmt --all -- --check
    fix: cargo fmt --all
  - name: clippy
    command: cargo clippy --workspace --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: workspace-tests
    command: env XAGENT_REQUIRE_GPU=1 cargo test --workspace
    timeout_ms: 7200000
  - name: integration-tick-window-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_tick_window_digest
    timeout_ms: 3600000
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: pf-tick-window-core
    command: |-
      sh -c 'grep -qF "pub struct TickRecord {" crates/xagent-sandbox/src/replay.rs && grep -qF "pub vision_color: Option<Vec<f32>>," crates/xagent-sandbox/src/replay.rs && grep -qF "pub const PHYS_STRIDE: usize = 54;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_RAW_GRADIENT_OUT: usize = 44;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const FOOD_STATE_STRIDE: usize = 4;" crates/xagent-brain/src/buffers.rs && grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/tick_window.rs && test ! -e crates/xagent-sandbox/src/tick_window_store.rs'
  - name: pf-headless-tick-window
    command: |-
      sh -c 'grep -qF "fn run_supervisor(" crates/xagent-sandbox/src/headless.rs && grep -qF "let physics = kernel.cached_state().to_vec();" crates/xagent-sandbox/src/headless.rs && grep -qF "let food = kernel.cached_food_state().map(<[f32]>::to_vec);" crates/xagent-sandbox/src/headless.rs && grep -qF "pub boundary: Option<BoundaryReport>," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub season: Option<FoodKind>," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "fn supervisor_headless_store_is_reproducible() {" crates/xagent-sandbox/tests/supervisor_headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs'
  - name: pf-worker-tick-window
    command: |-
      sh -c 'grep -qF "fn publish_snapshot(&self, event_tx: &SyncSender<SimEvent>) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn start_supervisor(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SupervisorSample(SampleReport)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "pub db_path: Option<String>," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn supervisor_init(config: &FullConfig, db_path: Option<String>) -> SimInit {" crates/xagent-sandbox/src/sim_runtime.rs'
  - name: pf-tick-window-view
    command: |-
      sh -c 'grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && grep -qF "mod sim_runtime;" crates/xagent-sandbox/src/main.rs && grep -qF "egui" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/tick_window_view.rs'
  - name: pf-gui-tick-window-wiring
    command: |-
      sh -c 'grep -qF "fn apply_state_snapshot(&mut self, snapshot: &StateSnapshot) {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub(crate) fn new(" crates/xagent-sandbox/src/app.rs && grep -qF "pub(crate) db_path: String," crates/xagent-sandbox/src/app.rs && grep -qF "pub(crate) evo_snapshot: EvolutionSnapshot," crates/xagent-sandbox/src/app.rs && grep -qF "Browse { run_id: i64 }" docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md && grep -qF "BrowsingSupervisor" docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md && ! grep -qF "GpuKernel" crates/xagent-sandbox/src/app.rs && ! grep -qF "GpuKernel" crates/xagent-sandbox/src/gpu_orchestration.rs'
  - name: pf-replay-jump-request
    command: |-
      sh -c 'A=docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md; grep -qF "filtered(model," $A && grep -qF "boundary_lines(&BoundaryRecord)" $A && grep -qF "show_boundaries(ui," $A && grep -qF "show_timeline" $A && grep -qF "observatory_view: ObservatoryViewState" $A && grep -qF "render_observatory(ui, model, view)" $A && grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - tick-window-core
      - headless-tick-window
      - worker-tick-window
      - tick-window-view
      - replay-jump-request
      - gui-tick-window-wiring
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: tick-window-core
        name: record-type-layout-and-schema
        sources:
          - crates/xagent-sandbox/src/replay.rs
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-tick-window-core
            command: |-
              sh -c 'grep -qF "pub struct TickRecord {" crates/xagent-sandbox/src/replay.rs && grep -qF "pub vision_color: Option<Vec<f32>>," crates/xagent-sandbox/src/replay.rs && grep -qF "pub const PHYS_STRIDE: usize = 54;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_RAW_GRADIENT_OUT: usize = 44;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const FOOD_STATE_STRIDE: usize = 4;" crates/xagent-brain/src/buffers.rs && grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/tick_window.rs && test ! -e crates/xagent-sandbox/src/tick_window_store.rs'
      - task: headless-tick-window
        name: headless-sample-copies-and-reports
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/supervisor_run.rs
          - crates/xagent-sandbox/tests/supervisor_headless.rs
        checks:
          - name: pf-headless-tick-window
            command: |-
              sh -c 'grep -qF "fn run_supervisor(" crates/xagent-sandbox/src/headless.rs && grep -qF "let physics = kernel.cached_state().to_vec();" crates/xagent-sandbox/src/headless.rs && grep -qF "let food = kernel.cached_food_state().map(<[f32]>::to_vec);" crates/xagent-sandbox/src/headless.rs && grep -qF "pub boundary: Option<BoundaryReport>," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "pub season: Option<FoodKind>," crates/xagent-sandbox/src/supervisor_run.rs && grep -qF "fn supervisor_headless_store_is_reproducible() {" crates/xagent-sandbox/tests/supervisor_headless.rs && test ! -e crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: worker-tick-window
        name: worker-publish-and-store-path
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
        checks:
          - name: pf-worker-tick-window
            command: |-
              sh -c 'grep -qF "fn publish_snapshot(&self, event_tx: &SyncSender<SimEvent>) {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn start_supervisor(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SupervisorSample(SampleReport)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "pub db_path: Option<String>," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn supervisor_init(config: &FullConfig, db_path: Option<String>) -> SimInit {" crates/xagent-sandbox/src/sim_runtime.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: tick-window-view
        name: bin-module-list-and-egui
        sources:
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-tick-window-view
            command: |-
              sh -c 'grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && grep -qF "mod sim_runtime;" crates/xagent-sandbox/src/main.rs && grep -qF "egui" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/tick_window_view.rs'
      - task: gui-tick-window-wiring
        name: main-thread-snapshot-and-sample-sites
        sources:
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/app.rs
          - docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md
        checks:
          - name: pf-gui-tick-window-wiring
            command: |-
              sh -c 'grep -qF "fn apply_state_snapshot(&mut self, snapshot: &StateSnapshot) {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub(crate) fn new(" crates/xagent-sandbox/src/app.rs && grep -qF "pub(crate) db_path: String," crates/xagent-sandbox/src/app.rs && grep -qF "pub(crate) evo_snapshot: EvolutionSnapshot," crates/xagent-sandbox/src/app.rs && grep -qF "Browse { run_id: i64 }" docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md && grep -qF "BrowsingSupervisor" docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md && ! grep -qF "GpuKernel" crates/xagent-sandbox/src/app.rs && ! grep -qF "GpuKernel" crates/xagent-sandbox/src/gpu_orchestration.rs'
      - task: replay-jump-request
        name: observatory-row-render-sites
        sources:
          - docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-replay-jump-request
            command: |-
              sh -c 'A=docs/plans/0008-supervisor-observatory-ui/ARCHITECTURE.md; grep -qF "filtered(model," $A && grep -qF "boundary_lines(&BoundaryRecord)" $A && grep -qF "show_boundaries(ui," $A && grep -qF "show_timeline" $A && grep -qF "observatory_view: ObservatoryViewState" $A && grep -qF "render_observatory(ui, model, view)" $A && grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml'
  - name: integration-route
    tasks:
      - "$integration"
    role: architect
    operations:
      - edit
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
    constraints:
      - task: "$integration"
        name: repository-gates-and-guard
        sources:
          - crates/xagent-sandbox/tests/contributing_guard.rs
          - CONTRIBUTING.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-integration-guard
            command: |-
              sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
---

# 0009 Supervisor tick-window replay

## Scope

A replay for supervisor runs built on ticks, not generations. Task ids are the
packet slugs in `tasks/`; ARCHITECTURE.md is the binding specification.

1. **Core and store (`tick-window-core`).** A pure library module
   `crates/xagent-sandbox/src/tick_window.rs`: `TickWindowConfig` (named
   bounds), `WindowFrame` (tick, one `replay::TickRecord` per agent built from
   the published physics rows, food rows), the rolling `TickWindow` with pins
   around replacement boundaries and season flips, and a versioned frame codec.
   `crates/xagent-sandbox/src/tick_window_store.rs`: additive idempotent tables,
   a bounded writer that prunes on every insert, and a read-only loader.
2. **Headless capture (`headless-tick-window`).** The headless supervisor loop
   optionally captures every sample from the physics and food copies it already
   makes, pins on boundaries and season flips, and persists. A seeded GPU gate
   proves the digest and every non-replay table are identical with capture on
   and off.
3. **Worker capture (`worker-tick-window`).** The simulation worker, in
   supervisor mode with a database, captures each snapshot it publishes and
   persists it; pins come from the supervisor sample reports it already gets.
4. **Playback view (`tick-window-view`).** A bin-side egui module with the
   playback state (cursor, play/pause, speed, selected agent, jump requests)
   and its render function, unit-tested without a GPU.
5. **Jump request (`replay-jump-request`).** The 0008 observatory row
   renderers `show_timeline` (`observatory/history.rs`) and `show_boundaries`
   (`observatory/tables.rs`) gain a "Replay" button on every timeline event row
   and every boundary record row that records the row's tick in their view
   state; `ui.rs` `render_observatory` moves it into
   `ObservatoryViewState.replay_jump_request`, the state it already receives.
   Proven by simulated egui clicks.
6. **GUI wiring (`gui-tick-window-wiring`).** The main thread keeps an
   in-memory window for the live run fed from applied snapshots and sample
   reports, loads a browsed database's window read-only exactly once when the
   0008 `BrowsingSupervisor` state is entered (db path `App.db_path`, run id
   from `ModelSource::Browse { run_id }` of `evo_snapshot.observatory`), clears
   it on leaving, and forwards `observatory_view.replay_jump_request` to the
   playback view each frame.

## Prerequisite interfaces relied on

- From 0006-supervisor-decision-telemetry: the supervisor run row in the `run`
  table (`mode = 'supervisor'`), `SupervisorStore`, `SampleReport` with
  `boundary` and `season`, and the seeded headless supervisor digest helper it
  exports for its digest gate (the gate test calls that helper by the name 0006
  publishes; if 0006 publishes only a test-local helper, the gate test
  recomputes the same digest from the same inputs).
- From 0008-supervisor-observatory-ui: `EvolutionState::BrowsingSupervisor`,
  `EvolutionSnapshot.observatory: Option<ObservatoryModel>` with `source:
  ModelSource::Browse { run_id }`, `EvolutionSnapshot.observatory_view:
  ObservatoryViewState` and `ui.rs` `render_observatory(ui, model, view)`
  (0008 ARCHITECTURE Interfaces), and the row renderers
  `history::show_timeline` and `tables::show_boundaries(ui, &ObservatoryModel,
  &mut BoundaryViewState)` (0008 ARCHITECTURE Interfaces; the timeline state
  type is `TimelineViewState` per the 0008 `history-view` packet). Only
  `replay-jump-request` edits `ui.rs`, and only `ObservatoryViewState` and
  `render_observatory`.
- From the baseline: `replay::TickRecord`, the `P_*` physics slot constants,
  `PHYS_STRIDE`, `FOOD_STATE_STRIDE`, `StateSnapshot`, `publish_snapshot`.

## Non-goals

- No change, removal or deletion of the per-generation replay (`replay.rs`,
  `replay_coord.rs`, `generation_recording`); generation-mode-removal deletes
  it and must keep or move `TickRecord`.
- No GPU dispatch, readback, kernel or WGSL change; fields the published
  snapshot does not carry (phase, vision, credit magnitude, recalled patterns)
  are recorded as "not recorded", never fetched.
- No change to any supervisor decision, schedule, random stream, config
  default or `SupervisorConfig` field; no CLI flag.
- No database writes while browsing read-only.
- No resume (owned by supervisor-live-resume).
