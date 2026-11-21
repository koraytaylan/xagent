---
objective: >-
  Answer each of the user's observability questions in the egui GUI, both live
  and when browsing a finished run read-only from a database, replacing the
  six-label supervisor panel. Views: a slot table (occupant, age, alive or
  dead, meals, deaths, window score against peer median, parent); a boundary
  inspector that shows for each replacement which slots were eligible and why
  with the actual numbers, which was picked, which parent and from where, and
  the newborn's config difference, and for idle boundaries why nothing
  happened; time-series charts over ticks (food and deaths per agent-tick,
  window-score spread, retired-occupant lifespan, archive best and median) with
  replacements and season flips marked on the axis; per-occupant intake against
  age and the live late-window learning measures, so within-life learning can
  be told apart from selection; a lineage view of who descends from whom, and
  an archive browser with outcome, age and config; a season strip (current
  nourishing kind, time to next flip) and a run-progress header (tick, budget,
  ticks per second, time to next boundary); a persisted, browsable event
  timeline. Session handling: on opening a database that holds a supervisor
  run, offer to resume it live (using supervisor-live-resume) or browse it
  read-only; browsing never writes to the database. Reads only the snapshots,
  series and event tables from supervisor-decision-telemetry; the redraw path
  makes no GPU dispatch or readback. These are new views: the generation
  fitness and hazard charts, evolution tree and node detail are left in place
  for generation-mode-removal to delete. Follow the egui notes in CLAUDE.md and
  docs/REVIEW-CHECKLIST.md.
depends_on:
  - 0006-supervisor-decision-telemetry
  - 0007-supervisor-live-resume
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
    command: env XAGENT_REQUIRE_GPU=1 cargo test --workspace -- --skip cortex_throughput_profile_baseline --skip cortex_throughput_meets_budget_on_software_adapter
    timeout_ms: 7200000
  - name: integration-observatory-browse
    command: cargo test -p xagent-sandbox --test observatory_browse
    timeout_ms: 3600000
  - name: integration-observatory-purity
    command: cargo test -p xagent-sandbox --test observatory_purity
    timeout_ms: 3600000
preflight_checks:
  - name: pf-observatory-model
    command: |-
      sh -c 'grep -qF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub mod supervisor_run;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct SupervisorSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "pub nourishing_kind: Option<xagent_shared::FoodKind>," crates/xagent-sandbox/src/ui.rs && grep -qF "serde = " crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/observatory'
  - name: pf-observatory-store-read
    command: |-
      sh -c 'grep -qF "let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/tests/observatory_browse.rs'
  - name: pf-observatory-views
    command: |-
      sh -c 'grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml && grep -qF "ui.allocate_exact_size(egui::vec2(avail, chart_height), egui::Sense::hover());" crates/xagent-sandbox/src/ui.rs && grep -qF "let painter = ui.painter_at(rect);" crates/xagent-sandbox/src/ui.rs && grep -qF "ui.painter().vline(" crates/xagent-sandbox/src/ui.rs && grep -qF "egui::Grid::new" crates/xagent-sandbox/src/ui.rs && grep -qF "is INFINITY inside" CLAUDE.md && test -f docs/REVIEW-CHECKLIST.md'
  - name: pf-observatory-panel
    command: |-
      sh -c 'grep -qF "fn render_running_dashboard(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {" crates/xagent-sandbox/src/ui.rs && grep -qF "if let Some(sup) = &evo.supervisor {" crates/xagent-sandbox/src/ui.rs && grep -qF "HasSession { generation: u32 }," crates/xagent-sandbox/src/ui.rs && grep -qF "EvolutionState::HasSession { generation } => {" crates/xagent-sandbox/src/ui_chrome.rs && grep -qF "fn render_fitness_chart(" crates/xagent-sandbox/src/ui.rs && grep -qF "fn render_tree(" crates/xagent-sandbox/src/ui.rs && grep -qF "fn evolution_snapshot_default_tree_pane_fraction() {" crates/xagent-sandbox/tests/integration.rs && test ! -e crates/xagent-sandbox/tests/observatory_purity.rs'
  - name: pf-session-wiring
    command: |-
      sh -c 'grep -qF "match check_existing_session(db_path) {" crates/xagent-sandbox/src/app.rs && grep -qF "EvolutionAction::Resume => match Governor::resume(&self.db_path) {" crates/xagent-sandbox/src/evolution.rs && grep -qF "self.evo_snapshot.supervisor = Some(SupervisorSnapshot {" crates/xagent-sandbox/src/evolution.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "fn gui_supervisor_main_thread_never_touches_kernel() {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub fn check_existing_session(db_path: &str) -> Option<(u32, GovernorConfig, BrainConfig)> {" crates/xagent-sandbox/src/governor.rs'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - observatory-model
      - observatory-store-read
      - tables-view
      - plots-view
      - history-view
      - observatory-panel
      - session-wiring
    role: developer
    operations:
      - edit
    command: cargo test -p xagent-sandbox --test contributing_guard
    timeout_ms: 3600000
    constraints:
      - task: observatory-model
        name: supervisor-modules-and-snapshot
        sources:
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-observatory-model
            command: |-
              sh -c 'grep -qF "pub mod supervisor_store;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub mod supervisor_run;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct SupervisorSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "pub nourishing_kind: Option<xagent_shared::FoodKind>," crates/xagent-sandbox/src/ui.rs && grep -qF "serde = " crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/observatory'
      - task: observatory-store-read
        name: read-only-sqlite-route
        sources:
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-observatory-store-read
            command: |-
              sh -c 'grep -qF "let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn ensure_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/tests/observatory_browse.rs'
      - task: tables-view
        name: egui-grid-and-layout-notes
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/src/ui.rs
          - CLAUDE.md
          - docs/REVIEW-CHECKLIST.md
        checks:
          - name: pf-observatory-views
            command: |-
              sh -c 'grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml && grep -qF "ui.allocate_exact_size(egui::vec2(avail, chart_height), egui::Sense::hover());" crates/xagent-sandbox/src/ui.rs && grep -qF "let painter = ui.painter_at(rect);" crates/xagent-sandbox/src/ui.rs && grep -qF "ui.painter().vline(" crates/xagent-sandbox/src/ui.rs && grep -qF "egui::Grid::new" crates/xagent-sandbox/src/ui.rs && grep -qF "is INFINITY inside" CLAUDE.md && test -f docs/REVIEW-CHECKLIST.md'
      - task: plots-view
        name: painter-chart-and-axis-markers
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/src/ui.rs
          - CLAUDE.md
          - docs/REVIEW-CHECKLIST.md
        checks:
          - name: pf-observatory-views
            command: |-
              sh -c 'grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml && grep -qF "ui.allocate_exact_size(egui::vec2(avail, chart_height), egui::Sense::hover());" crates/xagent-sandbox/src/ui.rs && grep -qF "let painter = ui.painter_at(rect);" crates/xagent-sandbox/src/ui.rs && grep -qF "ui.painter().vline(" crates/xagent-sandbox/src/ui.rs && grep -qF "egui::Grid::new" crates/xagent-sandbox/src/ui.rs && grep -qF "is INFINITY inside" CLAUDE.md && test -f docs/REVIEW-CHECKLIST.md'
      - task: history-view
        name: tree-and-grid-rendering
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/src/ui.rs
          - CLAUDE.md
          - docs/REVIEW-CHECKLIST.md
        checks:
          - name: pf-observatory-views
            command: |-
              sh -c 'grep -qF "egui = \"0.31\"" crates/xagent-sandbox/Cargo.toml && grep -qF "ui.allocate_exact_size(egui::vec2(avail, chart_height), egui::Sense::hover());" crates/xagent-sandbox/src/ui.rs && grep -qF "let painter = ui.painter_at(rect);" crates/xagent-sandbox/src/ui.rs && grep -qF "ui.painter().vline(" crates/xagent-sandbox/src/ui.rs && grep -qF "egui::Grid::new" crates/xagent-sandbox/src/ui.rs && grep -qF "is INFINITY inside" CLAUDE.md && test -f docs/REVIEW-CHECKLIST.md'
      - task: observatory-panel
        name: dashboard-state-and-generation-views
        sources:
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/ui_chrome.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-observatory-panel
            command: |-
              sh -c 'grep -qF "fn render_running_dashboard(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {" crates/xagent-sandbox/src/ui.rs && grep -qF "if let Some(sup) = &evo.supervisor {" crates/xagent-sandbox/src/ui.rs && grep -qF "HasSession { generation: u32 }," crates/xagent-sandbox/src/ui.rs && grep -qF "EvolutionState::HasSession { generation } => {" crates/xagent-sandbox/src/ui_chrome.rs && grep -qF "fn render_fitness_chart(" crates/xagent-sandbox/src/ui.rs && grep -qF "fn render_tree(" crates/xagent-sandbox/src/ui.rs && grep -qF "fn evolution_snapshot_default_tree_pane_fraction() {" crates/xagent-sandbox/tests/integration.rs && test ! -e crates/xagent-sandbox/tests/observatory_purity.rs'
      - task: session-wiring
        name: startup-session-actions-and-event-fold
        sources:
          - crates/xagent-sandbox/src/app.rs
          - crates/xagent-sandbox/src/evolution.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/governor.rs
        checks:
          - name: pf-session-wiring
            command: |-
              sh -c 'grep -qF "match check_existing_session(db_path) {" crates/xagent-sandbox/src/app.rs && grep -qF "EvolutionAction::Resume => match Governor::resume(&self.db_path) {" crates/xagent-sandbox/src/evolution.rs && grep -qF "self.evo_snapshot.supervisor = Some(SupervisorSnapshot {" crates/xagent-sandbox/src/evolution.rs && grep -qF "SimEvent::SupervisorSample(report) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "fn gui_supervisor_main_thread_never_touches_kernel() {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "pub fn check_existing_session(db_path: &str) -> Option<(u32, GovernorConfig, BrainConfig)> {" crates/xagent-sandbox/src/governor.rs'
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
        checks:
          - name: pf-integration-guard
            command: |-
              sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
---

# 0008 Supervisor observatory UI

## Scope

New egui views that answer the supervisor observability questions, fed from one
bounded main-thread model, `ObservatoryModel`, which is filled either live from
the telemetry events plan 0006 publishes on the existing `SimEvent` channels, or
once, read-only, from the plan 0006 snapshot, series and event tables. Task ids
are the packet slugs in `tasks/`; ARCHITECTURE.md is binding.

1. **Model (`observatory-model`).** `crates/xagent-sandbox/src/observatory/mod.rs`
   with the pure, bounded `ObservatoryModel` and its fold from plan 0006 records;
   it creates the four submodule files as doc-only stubs that one later task
   each owns.
2. **Read-only browse (`observatory-store-read`).** Detect a supervisor run in a
   database and load it into the model through a read-only connection; a test
   proves the file is byte-identical afterwards and that writes are refused.
3. **Views (`tables-view`, `plots-view`, `history-view`).** `tables.rs`: slot
   table, boundary inspector, run-progress header and season strip. `plots.rs`:
   the four time-series charts with replacement and season-flip markers, and the
   learning view (intake against age, late-window measures). `history.rs`:
   lineage view, archive browser and the event timeline. Every render function
   takes `&ObservatoryModel`; what it shows comes from pure, unit-tested helpers.
   Charts are painter-based like the existing fitness chart; no new dependency.
4. **Panel (`observatory-panel`).** `ui.rs` composes the views in
   `render_observatory`, replaces the six-label supervisor section of
   `render_running_dashboard`, adds the supervisor session dialog and browse
   states and `SessionChoice`, and a source-scan test that the redraw path names
   no GPU, readback or database API.
5. **Session wiring (`session-wiring`).** At startup a database holding a
   supervisor run is detected read-only before any generation-mode open; the
   user's choice runs Resume live (the plan 0007 worker start-from-checkpoint
   path) or Browse read-only; live events fold into the model.

## Prerequisite interfaces relied on

- Plan 0006 (`supervisor-decision-telemetry`): the per-slot snapshot, boundary,
  series, season/progress and event record types it defines in
  `supervisor_run.rs` / `supervisor_store.rs`, the `SimEvent` variants that carry
  them to the main thread, and its run-keyed snapshot, series and event tables
  with a read API or documented schema. Archive admissions are among its
  recorded events with entry id, outcome, birth and retire tick and config.
  Plan 0006 ARCHITECTURE names the live carriers
  `SimEvent::SupervisorTelemetry(Box<TelemetryFrame>)` (latest-wins) and
  `SimEvent::SupervisorBoundary(Box<BoundaryRecord>)` (control).
- Plan 0007 (`supervisor-live-resume`): the worker start-from-checkpoint path,
  `SupervisorWorkerInit.resume: bool` (plan 0007 ARCHITECTURE), and its refusal:
  one `SimEvent::Error` for a missing, corrupt or mismatched checkpoint, after
  which the worker stays paused.
- Exact names are read from the landed code at implementation time; when a
  prerequisite lacks something listed here, the task reports the file and line
  as a blocker instead of inventing it.

## Non-goals

- No change to generation mode or its views: fitness and hazard charts,
  evolution tree, node detail, Previous Session Found and generation Resume stay.
- No GPU dispatch, readback or database access in the redraw path; no new
  telemetry, table, column or checkpoint format (plans 0006 and 0007 own them).
- No write to a browsed database: no schema migration, no `ensure_schema`, no
  Reset offered in browse mode.
- No tick-window replay (plan 0009), no new crate dependency, no WGSL change.
- No file picker: the database is the app's existing `--db` path.
