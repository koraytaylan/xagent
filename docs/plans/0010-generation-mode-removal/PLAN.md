---
objective: >-
  Remove generation mode entirely and make the supervisor the only mode. Delete:
  the Governor generation state machine and its SQLite generation persistence,
  the tick-budget handoff between worker and main thread
  (GenerationBudgetReached, pending generation, generation epoch),
  generation-only GovernorConfig fields, the EvolutionMode switch in config,
  CLI, UI and the run table, the generation UI (fitness and hazard charts,
  evolution tree, node detail, Previous Session Found and generation Resume),
  generation CLI flags, per-generation replay, and their tests. Compatibility:
  old config JSON that still carries generation-era keys loads, with the dead
  keys ignored and logged once; removed CLI flags fail with a clear message; a
  legacy generations database (the user's xagent.db is one, at generation 3670)
  is detected and refused with a plain message that this version cannot run it,
  and is never written to, overwritten or deleted without the user's explicit
  confirmation (today's Reset deletes the file outright). Dependents: start with
  a written inventory and settle each one as delete (served only generations) or
  port (guards something that still matters) with no generation mode kept alive
  in test-only form: tests/integration.rs, seeded_headless_digest.rs (removed
  only once the plan 1 supervisor digest gate is in the plan's checks),
  generation_run_summary.rs, evolution_cli_exit.rs, sense_block_parity.rs,
  food_kind_free_life.rs, archive_free_life.rs, the validation A/B flags, the
  free-life champion-db source (the archive source replaces it), and the study
  binaries food_kind_verdict, pathway_route_gate, pathway_return_fit and
  supervisor_verdict plus the 0005 measurement code that only compared against
  generations. Docs: update README.md, CLAUDE.md (architecture notes and test
  counts), CONTRIBUTING.md and EVOLUTION_JOURNEY.md to describe the
  supervisor-only system.
depends_on:
  - 0006-supervisor-decision-telemetry
  - 0007-supervisor-live-resume
  - 0008-supervisor-observatory-ui
  - 0009-supervisor-tick-window-replay
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
  - name: supervisor-seeded-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_digest
    timeout_ms: 3600000
  - name: sense-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
    timeout_ms: 5400000
  - name: generation-surface-absent
    command: |-
      sh -c '! grep -rnE --include=*.rs "GenerationBudgetReached|EvolutionMode|pending_generation|generation_epoch|crate::governor|xagent_sandbox::governor|run_generations|generation_recording" crates/ && test ! -e crates/xagent-sandbox/src/governor.rs'
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: pf-study-binaries
    command: |-
      sh -c 'test -f crates/xagent-sandbox/src/bin/food_kind_verdict.rs && test -f crates/xagent-sandbox/src/bin/pathway_route_gate.rs && test -f crates/xagent-sandbox/src/bin/pathway_return_fit.rs && test -f crates/xagent-sandbox/src/bin/supervisor_verdict.rs && grep -qF "default-run = \"xagent\"" crates/xagent-sandbox/Cargo.toml && ! grep -rqE "CARGO_BIN_EXE_(food_kind_verdict|pathway_route_gate|pathway_return_fit|supervisor_verdict)" crates'
  - name: pf-integration-tests
    command: |-
      sh -c 'grep -qF "fn gpu_adapter_present_when_required() {" crates/xagent-sandbox/tests/integration.rs && grep -qF "let gov_config = GovernorConfig {" crates/xagent-sandbox/tests/integration.rs && grep -qF "Governor::new(" crates/xagent-sandbox/tests/integration.rs && grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs'
  - name: pf-generation-test-files
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/seeded_headless_digest.rs && test -f crates/xagent-sandbox/tests/generation_run_summary.rs && test -f crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json && ! grep -rqF "seeded_headless_digest.json" crates/xagent-sandbox/src'
  - name: pf-sense-parity-supervisor
    command: |-
      sh -c 'grep -qF "use xagent_sandbox::governor::Governor;" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "tests/fixtures/sense_block_champion.db" crates/xagent-sandbox/tests/sense_block_parity.rs && test -f crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden && test -f crates/xagent-sandbox/tests/fixtures/sense_block_champion.db && grep -qF "pub fn load_archive_entry_read_only(" crates/xagent-sandbox/src/supervisor_store.rs'
  - name: pf-store-legacy-guard
    command: |-
      sh -c 'grep -qF "use crate::governor::{decode_brain_state, encode_brain_state, init_schema};" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "let mut db = Connection::open(db_path)?;" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "use crate::governor::{composite_fitness, exploration_grid_cells};" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn decode_raw_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "CREATE TABLE IF NOT EXISTS node (" crates/xagent-sandbox/src/governor.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml'
  - name: pf-gui-cli-supervisor-only
    command: |-
      sh -c 'grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "use crate::app::{App, PendingGeneration};" crates/xagent-sandbox/src/evolution.rs && grep -qF "Previous Session Found" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn reset_database(db_path: &str) -> std::io::Result<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "generations: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "dump_tree: bool," crates/xagent-sandbox/src/main.rs && grep -qF "free_life_champion_db: Option<String>," crates/xagent-sandbox/src/main.rs && grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && test -f crates/xagent-sandbox/tests/evolution_cli_exit.rs'
  - name: pf-headless-supervisor-only
    command: |-
      sh -c 'grep -qF "fn run_generations(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn dump_tree(db_path: &str) {" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn validate_danger_percept(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub enum BrainSource {" crates/xagent-sandbox/src/headless.rs && grep -qF "pub mode: EvolutionMode," crates/xagent-sandbox/src/run_summary.rs && grep -qF "fn generation_mode_is_the_default_driver() {" crates/xagent-sandbox/tests/supervisor_headless.rs && grep -qF "fn food_kind_free_life_refuses_champion_brain() {" crates/xagent-sandbox/tests/food_kind_free_life.rs && grep -qF "fn free_life_champion_source_reports_node_and_generation() {" crates/xagent-sandbox/tests/free_life.rs'
  - name: pf-headless-champion-source
    command: |-
      sh -c 'grep -qF "pub enum BrainSource {" crates/xagent-sandbox/src/headless.rs && grep -qF "fn food_kind_free_life_refuses_champion_brain() {" crates/xagent-sandbox/tests/food_kind_free_life.rs && grep -qF "fn free_life_champion_source_reports_node_and_generation() {" crates/xagent-sandbox/tests/free_life.rs && test -f crates/xagent-sandbox/tests/archive_free_life.rs'
  - name: pf-governor-removal
    command: |-
      sh -c 'grep -qF "pub mod governor;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct Governor {" crates/xagent-sandbox/src/governor.rs && grep -qF "use xagent_sandbox::governor::Governor;" crates/xagent-sandbox/tests/pathway_route_state.rs && test -f crates/xagent-sandbox/src/replay.rs'
  - name: pf-config-generation-fields
    command: |-
      sh -c 'grep -qF "pub enum EvolutionMode {" crates/xagent-shared/src/config.rs && grep -qF "pub evolution_mode: EvolutionMode," crates/xagent-shared/src/config.rs && grep -qF "pub migration_interval: u32," crates/xagent-shared/src/config.rs && grep -qF "pub momentum_decay: f32," crates/xagent-shared/src/config.rs && grep -qF "pub supervisor: SupervisorConfig," crates/xagent-shared/src/config.rs && grep -qF "EvolutionMode" crates/xagent-shared/src/lib.rs && ! grep -qF "deny_unknown_fields" crates/xagent-shared/src/config.rs'
  - name: pf-sandbox-mode-test-users
    command: |-
      sh -c 'cd crates/xagent-sandbox/tests && test -f sense_block_parity.rs && for f in supervisor_digest.rs supervisor_resume_digest.rs supervisor_run_resume.rs supervisor_sense_parity.rs supervisor_tick_window_digest.rs; do grep -qF "config.governor.evolution_mode = EvolutionMode::Supervisor;" $f || exit 1; done'
  - name: pf-sandbox-governor-default-users
    command: |-
      sh -c 'grep -qF "tick_budget: xagent_shared::GovernorConfig::default().tick_budget," crates/xagent-sandbox/src/ui.rs && grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs'
  - name: pf-docs-supervisor-only
    command: |-
      sh -c 'grep -qF "evolution state machine, SQLite persistence" CLAUDE.md && grep -qF "runs 221 total tests" CLAUDE.md && grep -qF "governor.rs" CONTRIBUTING.md && test -f EVOLUTION_JOURNEY.md && test -f README.md && test -f crates/xagent-sandbox/README.md && grep -qF "(\"crates/xagent-sandbox/src/governor.rs\", 1)," crates/xagent-sandbox/tests/contributing_guard.rs'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "fn planning_references_do_not_exceed_baseline() {" crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - integration-tests
      - store-legacy-guard
      - worker-session-supervisor-only
      - ui-replay-supervisor-only
      - headless-supervisor-only
      - headless-champion-source
      - config-generation-fields
      - sandbox-mode-test-users
      - sandbox-governor-default-users
      - guard-baseline
      - docs-supervisor-only
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: integration-tests
        name: integration-test-generation-surface
        sources:
          - crates/xagent-sandbox/tests/integration.rs
          - crates/xagent-sandbox/src/ui.rs
        checks:
          - name: pf-integration-tests
            command: |-
              sh -c 'grep -qF "fn gpu_adapter_present_when_required() {" crates/xagent-sandbox/tests/integration.rs && grep -qF "let gov_config = GovernorConfig {" crates/xagent-sandbox/tests/integration.rs && grep -qF "Governor::new(" crates/xagent-sandbox/tests/integration.rs && grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: store-legacy-guard
        name: store-codec-schema-and-fitness-sources
        sources:
          - crates/xagent-sandbox/src/supervisor_store.rs
          - crates/xagent-sandbox/src/supervisor.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-store-legacy-guard
            command: |-
              sh -c 'grep -qF "use crate::governor::{decode_brain_state, encode_brain_state, init_schema};" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "pub fn create(db_path: &str, config: &FullConfig) -> Result<Self, StoreError> {" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "let mut db = Connection::open(db_path)?;" crates/xagent-sandbox/src/supervisor_store.rs && grep -qF "use crate::governor::{composite_fitness, exploration_grid_cells};" crates/xagent-sandbox/src/supervisor.rs && grep -qF "fn decode_raw_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "CREATE TABLE IF NOT EXISTS node (" crates/xagent-sandbox/src/governor.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml'
      - task: headless-supervisor-only
        name: headless-generation-driver-and-tests
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/run_summary.rs
          - crates/xagent-sandbox/tests/supervisor_headless.rs
          - crates/xagent-sandbox/tests/food_kind_free_life.rs
          - crates/xagent-sandbox/tests/free_life.rs
        checks:
          - name: pf-headless-supervisor-only
            command: |-
              sh -c 'grep -qF "fn run_generations(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn dump_tree(db_path: &str) {" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn validate_danger_percept(" crates/xagent-sandbox/src/headless.rs && grep -qF "pub enum BrainSource {" crates/xagent-sandbox/src/headless.rs && grep -qF "pub mode: EvolutionMode," crates/xagent-sandbox/src/run_summary.rs && grep -qF "fn generation_mode_is_the_default_driver() {" crates/xagent-sandbox/tests/supervisor_headless.rs && grep -qF "fn food_kind_free_life_refuses_champion_brain() {" crates/xagent-sandbox/tests/food_kind_free_life.rs && grep -qF "fn free_life_champion_source_reports_node_and_generation() {" crates/xagent-sandbox/tests/free_life.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: worker-session-supervisor-only
        name: worker-handoff-and-session-sources
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/evolution.rs
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/tests/evolution_cli_exit.rs
          - crates/xagent-sandbox/src/app.rs
          - crates/xagent-sandbox/src/snapshot.rs
        checks:
          - name: pf-gui-cli-supervisor-only
            command: |-
              sh -c 'grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "use crate::app::{App, PendingGeneration};" crates/xagent-sandbox/src/evolution.rs && grep -qF "Previous Session Found" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn reset_database(db_path: &str) -> std::io::Result<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "generations: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "dump_tree: bool," crates/xagent-sandbox/src/main.rs && grep -qF "free_life_champion_db: Option<String>," crates/xagent-sandbox/src/main.rs && grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && test -f crates/xagent-sandbox/tests/evolution_cli_exit.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: ui-replay-supervisor-only
        name: generation-ui-and-replay-wiring
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/evolution.rs
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/tests/evolution_cli_exit.rs
          - crates/xagent-sandbox/src/ui_chrome.rs
          - crates/xagent-sandbox/src/render_pipeline.rs
          - crates/xagent-sandbox/src/replay_coord.rs
          - crates/xagent-sandbox/src/app.rs
        checks:
          - name: pf-gui-cli-supervisor-only
            command: |-
              sh -c 'grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "use crate::app::{App, PendingGeneration};" crates/xagent-sandbox/src/evolution.rs && grep -qF "Previous Session Found" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn reset_database(db_path: &str) -> std::io::Result<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "generations: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "dump_tree: bool," crates/xagent-sandbox/src/main.rs && grep -qF "free_life_champion_db: Option<String>," crates/xagent-sandbox/src/main.rs && grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && test -f crates/xagent-sandbox/tests/evolution_cli_exit.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: headless-champion-source
        name: champion-source-and-tests
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/tests/food_kind_free_life.rs
          - crates/xagent-sandbox/tests/free_life.rs
          - crates/xagent-sandbox/tests/archive_free_life.rs
        checks:
          - name: pf-headless-champion-source
            command: |-
              sh -c 'grep -qF "pub enum BrainSource {" crates/xagent-sandbox/src/headless.rs && grep -qF "fn food_kind_free_life_refuses_champion_brain() {" crates/xagent-sandbox/tests/food_kind_free_life.rs && grep -qF "fn free_life_champion_source_reports_node_and_generation() {" crates/xagent-sandbox/tests/free_life.rs && test -f crates/xagent-sandbox/tests/archive_free_life.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: config-generation-fields
        name: governor-config-fields-and-exports
        sources:
          - crates/xagent-shared/src/config.rs
          - crates/xagent-shared/src/lib.rs
        checks:
          - name: pf-config-generation-fields
            command: |-
              sh -c 'grep -qF "pub enum EvolutionMode {" crates/xagent-shared/src/config.rs && grep -qF "pub evolution_mode: EvolutionMode," crates/xagent-shared/src/config.rs && grep -qF "pub migration_interval: u32," crates/xagent-shared/src/config.rs && grep -qF "pub momentum_decay: f32," crates/xagent-shared/src/config.rs && grep -qF "pub supervisor: SupervisorConfig," crates/xagent-shared/src/config.rs && grep -qF "EvolutionMode" crates/xagent-shared/src/lib.rs && ! grep -qF "deny_unknown_fields" crates/xagent-shared/src/config.rs'
      - task: sandbox-mode-test-users
        name: sandbox-mode-test-users-lines
        sources:
          - crates/xagent-sandbox/tests/sense_block_parity.rs
          - crates/xagent-sandbox/tests/supervisor_digest.rs
          - crates/xagent-sandbox/tests/supervisor_resume_digest.rs
          - crates/xagent-sandbox/tests/supervisor_run_resume.rs
          - crates/xagent-sandbox/tests/supervisor_sense_parity.rs
          - crates/xagent-sandbox/tests/supervisor_tick_window_digest.rs
        checks:
          - name: pf-sandbox-mode-test-users
            command: |-
              sh -c 'cd crates/xagent-sandbox/tests && test -f sense_block_parity.rs && for f in supervisor_digest.rs supervisor_resume_digest.rs supervisor_run_resume.rs supervisor_sense_parity.rs supervisor_tick_window_digest.rs; do grep -qF "config.governor.evolution_mode = EvolutionMode::Supervisor;" $f || exit 1; done'
      - task: sandbox-governor-default-users
        name: sandbox-governor-default-lines
        sources:
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/agent/mod.rs
        checks:
          - name: pf-sandbox-governor-default-users
            command: |-
              sh -c 'grep -qF "tick_budget: xagent_shared::GovernorConfig::default().tick_budget," crates/xagent-sandbox/src/ui.rs && grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs'
      - task: guard-baseline
        name: guard-baseline-row
        sources:
          - crates/xagent-sandbox/tests/contributing_guard.rs
        checks:
          - name: pf-docs-supervisor-only
            command: |-
              sh -c 'grep -qF "evolution state machine, SQLite persistence" CLAUDE.md && grep -qF "runs 221 total tests" CLAUDE.md && grep -qF "governor.rs" CONTRIBUTING.md && test -f EVOLUTION_JOURNEY.md && test -f README.md && test -f crates/xagent-sandbox/README.md && grep -qF "(\"crates/xagent-sandbox/src/governor.rs\", 1)," crates/xagent-sandbox/tests/contributing_guard.rs'
      - task: docs-supervisor-only
        name: docs-and-guard-baseline
        sources:
          - CLAUDE.md
          - CONTRIBUTING.md
          - EVOLUTION_JOURNEY.md
          - README.md
          - crates/xagent-sandbox/README.md
          - crates/xagent-sandbox/tests/contributing_guard.rs
        checks:
          - name: pf-docs-supervisor-only
            command: |-
              sh -c 'grep -qF "evolution state machine, SQLite persistence" CLAUDE.md && grep -qF "runs 221 total tests" CLAUDE.md && grep -qF "governor.rs" CONTRIBUTING.md && test -f EVOLUTION_JOURNEY.md && test -f README.md && test -f crates/xagent-sandbox/README.md && grep -qF "(\"crates/xagent-sandbox/src/governor.rs\", 1)," crates/xagent-sandbox/tests/contributing_guard.rs'
  - name: architect-route
    tasks:
      - study-binaries
      - generation-test-files
      - sense-parity-supervisor
      - cli-supervisor-only
      - governor-removal
    role: architect
    operations:
      - edit
      - shell
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    actions:
      - name: file-deletion-probe
        command: |-
          sh -c 'd=$(mktemp -d) && : > "$d/probe" && rm "$d/probe" && rmdir "$d"'
    constraints:
      - task: study-binaries
        name: study-binary-files
        sources:
          - crates/xagent-sandbox/src/bin/food_kind_verdict.rs
          - crates/xagent-sandbox/src/bin/pathway_route_gate.rs
          - crates/xagent-sandbox/src/bin/pathway_return_fit.rs
          - crates/xagent-sandbox/src/bin/supervisor_verdict.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-study-binaries
            command: |-
              sh -c 'test -f crates/xagent-sandbox/src/bin/food_kind_verdict.rs && test -f crates/xagent-sandbox/src/bin/pathway_route_gate.rs && test -f crates/xagent-sandbox/src/bin/pathway_return_fit.rs && test -f crates/xagent-sandbox/src/bin/supervisor_verdict.rs && grep -qF "default-run = \"xagent\"" crates/xagent-sandbox/Cargo.toml && ! grep -rqE "CARGO_BIN_EXE_(food_kind_verdict|pathway_route_gate|pathway_return_fit|supervisor_verdict)" crates'
      - task: generation-test-files
        name: generation-only-test-files
        sources:
          - crates/xagent-sandbox/tests/seeded_headless_digest.rs
          - crates/xagent-sandbox/tests/generation_run_summary.rs
          - crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json
        checks:
          - name: pf-generation-test-files
            command: |-
              sh -c 'test -f crates/xagent-sandbox/tests/seeded_headless_digest.rs && test -f crates/xagent-sandbox/tests/generation_run_summary.rs && test -f crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json && ! grep -rqF "seeded_headless_digest.json" crates/xagent-sandbox/src'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: sense-parity-supervisor
        name: sense-parity-fixture-and-archive-loader
        sources:
          - crates/xagent-sandbox/tests/sense_block_parity.rs
          - crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden
          - crates/xagent-sandbox/src/supervisor_store.rs
        checks:
          - name: pf-sense-parity-supervisor
            command: |-
              sh -c 'grep -qF "use xagent_sandbox::governor::Governor;" crates/xagent-sandbox/tests/sense_block_parity.rs && grep -qF "tests/fixtures/sense_block_champion.db" crates/xagent-sandbox/tests/sense_block_parity.rs && test -f crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden && test -f crates/xagent-sandbox/tests/fixtures/sense_block_champion.db && grep -qF "pub fn load_archive_entry_read_only(" crates/xagent-sandbox/src/supervisor_store.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: cli-supervisor-only
        name: cli-flags-and-exit-test
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/evolution.rs
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/tests/evolution_cli_exit.rs
          - crates/xagent-sandbox/src/replay_coord.rs
        checks:
          - name: pf-gui-cli-supervisor-only
            command: |-
              sh -c 'grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "use crate::app::{App, PendingGeneration};" crates/xagent-sandbox/src/evolution.rs && grep -qF "Previous Session Found" crates/xagent-sandbox/src/ui.rs && grep -qF "pub fn reset_database(db_path: &str) -> std::io::Result<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "generations: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "dump_tree: bool," crates/xagent-sandbox/src/main.rs && grep -qF "free_life_champion_db: Option<String>," crates/xagent-sandbox/src/main.rs && grep -qF "mod replay_coord;" crates/xagent-sandbox/src/main.rs && test -f crates/xagent-sandbox/tests/evolution_cli_exit.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: governor-removal
        name: module-tree-and-governor-test
        sources:
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/replay.rs
          - crates/xagent-sandbox/tests/pathway_route_state.rs
        checks:
          - name: pf-governor-removal
            command: |-
              sh -c 'grep -qF "pub mod governor;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && grep -qF "pub struct Governor {" crates/xagent-sandbox/src/governor.rs && grep -qF "use xagent_sandbox::governor::Governor;" crates/xagent-sandbox/tests/pathway_route_state.rs && test -f crates/xagent-sandbox/src/replay.rs'
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
              sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "fn planning_references_do_not_exceed_baseline() {" crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
---

# 0010 Generation mode removal

## Scope

The supervisor becomes the only evolution mode. ARCHITECTURE.md is the binding
specification: its dependents inventory settles every dependent as delete or
port, and its removed-surface list is the shared contract between tasks. Task
ids are the packet slugs in `tasks/`.

Order is callers first, callees last, so every task leaves all targets
compiling: test files and study binaries (0100), the legacy-database guard in
the store (0100), then the GUI/CLI and the headless driver (0200), then the
Governor module and the generation config fields (0300), then the docs and the
contributing-guard baseline (0400).

1. **Leaves (`study-binaries`, `integration-tests`, `generation-test-files`,
   `sense-parity-supervisor`).** Delete the four study binaries and the
   generation-only test files; strip generation tests from `integration.rs`;
   port the sense-parity gate to supervisor runs.
2. **Store guard (`store-legacy-guard`).** `supervisor_store.rs` gains the
   read-only `inspect_database` and refuses a legacy generations database before
   any write; it takes over the brain-blob codec and the run schema, and
   `supervisor.rs` takes over the fitness helpers, so neither imports
   `governor`.
3. **GUI and CLI (`worker-session-supervisor-only`, `ui-replay-supervisor-only`,
   `cli-supervisor-only`).** The worker loses the generation handoff and the
   session refuses a legacy database and never deletes a database unconfirmed;
   then `ui.rs` and the per-generation replay wiring lose every generation
   member; then the CLI loses every generation flag, removed flags fail
   clearly and dead config keys are logged once.
4. **Headless (`headless-supervisor-only`, `headless-champion-source`).** `run_headless` runs only the
   supervisor, refuses legacy databases, and the generation driver, tree dump,
   validation A/B code, champion source and generation measurement go.
5. **Removal (`governor-removal`, `sandbox-mode-test-users`,
   `sandbox-governor-default-users`, `config-generation-fields`).** Delete
   `governor.rs` (and `replay.rs` unless 0009 still imports from it), drop the
   last sandbox users of `EvolutionMode` and `GovernorConfig::tick_budget`, then
   the generation-only `GovernorConfig` fields and `EvolutionMode`.
6. **Guard and docs (`guard-baseline`, `docs-supervisor-only`).** README.md, CLAUDE.md (architecture notes
   and test counts), CONTRIBUTING.md, EVOLUTION_JOURNEY.md and the sandbox
   README describe the supervisor-only system; the contributing-guard baseline
   is tightened to the new tree.

## GPU proofs fail closed

Every check that runs a GPU-backed test sets `XAGENT_REQUIRE_GPU=1`.

## Non-goals

- No change to supervisor behaviour, defaults or the seeded supervisor digest.
- No migration or conversion of a legacy generations database; it is only
  detected and refused.
- No edit of history records under `docs/decisions/`, `docs/measurements/` or
  other plans.
- No dependency or version change.
- No new learning rule, reward term or WGSL edit.
