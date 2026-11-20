---
objective: >-
  Add a flag, off by default, that switches evolution from synchronised
  generations to a continuous supervisor, and end with a recorded decision on
  whether to adopt it. One long-lived world keeps the same 10 agents; one slot
  is refilled at a time, only at seeded tick boundaries, when its agent died or
  scored worse than its peers over the same window; newborns are Darwinian
  (a mutated birth brain from a bounded archive of the best birth brains); the
  plan 0004 food-kind assignment flips on a seeded season timer. A
  learned-weight copy exists only as a comparison arm. A fail-closed verdict
  tool, written before any run, decides adopt, keep as an option, or reject;
  the flag stays off whatever the result.
depends_on:
  - 0004-food-contingency-learning
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
  - name: integration-seeded-digest
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest
    timeout_ms: 3600000
  - name: integration-sense-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
    timeout_ms: 5400000
  - name: integration-food-kind-default
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_draw
    timeout_ms: 3600000
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: pf-evolution-mode-config
    command: |-
      sh -c 'grep -qF "pub struct GovernorConfig {" crates/xagent-shared/src/config.rs && grep -qF "impl Default for GovernorConfig {" crates/xagent-shared/src/config.rs && grep -qF "pub governor: GovernorConfig," crates/xagent-shared/src/config.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs && grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "fn encode_brain_state(state: &AgentBrainState) -> Option<Vec<u8>> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn init_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "ALTER TABLE run ADD COLUMN simulation_ticks INTEGER DEFAULT 0;" crates/xagent-sandbox/src/governor.rs && grep -qF "let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;" crates/xagent-sandbox/src/governor.rs && grep -qF "let config: GovernorConfig = serde_json::from_str(&governor_json).unwrap_or_default();" crates/xagent-sandbox/src/governor.rs && grep -qF "let gov_config = GovernorConfig {" crates/xagent-sandbox/tests/integration.rs && grep -qF "XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
  - name: pf-run-mode-column
    command: |-
      sh -c 'grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn init_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "ALTER TABLE run ADD COLUMN simulation_ticks INTEGER DEFAULT 0;" crates/xagent-sandbox/src/governor.rs && grep -qF "let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;" crates/xagent-sandbox/src/governor.rs && grep -qF "FROM run ORDER BY id DESC LIMIT 1" crates/xagent-sandbox/src/governor.rs && grep -qF "pub fn resume(db_path: &str) -> SqlResult<Self> {" crates/xagent-sandbox/src/governor.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && grep -qF "XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
  - name: pf-run-summary
    command: |-
      sh -c 'grep -qF "pub fn nearest_food_of_kind(" docs/plans/0004-food-contingency-learning/tasks/0301-food-kind-metrics.md && grep -qF "pub struct NearestFood { pub distance: f32, pub bearing: f32 }" docs/plans/0004-food-contingency-learning/tasks/0301-food-kind-metrics.md && grep -qF "P_HARMFUL_MEAL_COUNT" docs/plans/0004-food-contingency-learning/ARCHITECTURE.md && grep -qF "SCENT_DECAY_LENGTH, SCENT_RANGE," crates/xagent-shared/src/lib.rs && grep -qF "pub const P_FOOD_COUNT: usize = 18;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_DEATH_COUNT: usize = 23;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && test ! -e crates/xagent-sandbox/src/run_summary.rs'
  - name: pf-supervisor-core
    command: |-
      sh -c 'grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "pub const HEATMAP_RES: usize = 64;" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_steering_weights(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_sensory_genes(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn new(decay: f32) -> Self {" crates/xagent-sandbox/src/momentum.rs && grep -qF "rand = { workspace = true }" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/supervisor.rs'
  - name: pf-supervisor-store
    command: |-
      sh -c 'grep -qF "fn encode_brain_state(state: &AgentBrainState) -> Option<Vec<u8>> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "CREATE TABLE IF NOT EXISTS node_brain (" crates/xagent-sandbox/src/governor.rs && grep -qF "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;" crates/xagent-sandbox/src/governor.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && grep -qF "pub fn load_champion_read_only(db_path: &str, node_id: Option<i64>) ->" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && test ! -e crates/xagent-sandbox/src/supervisor_store.rs'
  - name: pf-kernel-supervisor-hooks
    command: |-
      sh -c 'grep -qF "pub fn write_agent_state(&self, index: u32, state: &AgentBrainState) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn write_agent_heritable_config(&self, index: u32, config: &BrainConfig) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn write_agent_physics_fields(&self, index: u32, fields: &[(usize, f32)]) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn upload_agents(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "data[base + P_PREV_INTEGRITY] = max_i;" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn write_world_config_scratch(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn set_freeze_steering_weights(&mut self, freeze: bool) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn is_available() -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn nourishing_food_kind(&self) -> Option<FoodKind>" docs/plans/0004-food-contingency-learning/tasks/0105-kernel-food-kind-draw.md && grep -qF "food_kinds_slot_value" docs/plans/0004-food-contingency-learning/tasks/0105-kernel-food-kind-draw.md'
  - name: pf-supervisor-run
    command: |-
      sh -c 'grep -qF "pub fn kernel_batch_size(&self) -> u32 {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn try_collect_state_snapshot(&mut self) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn cached_state(&self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn probe_kernel_batches(&self) -> u64 {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn safe_spawn_position(&self) -> glam::Vec3 {" crates/xagent-sandbox/src/world/mod.rs && grep -qF "const HEATMAP_INTERVAL: u32 = 100;" crates/xagent-sandbox/src/headless.rs'
  - name: pf-headless-supervisor-driver
    command: |-
      sh -c 'grep -qF "pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {" crates/xagent-sandbox/src/headless.rs && grep -qF "let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);" crates/xagent-sandbox/src/headless.rs && grep -qF "while !kernel.try_collect_state_snapshot() {" crates/xagent-sandbox/src/headless.rs && grep -qF "run_free_life(config: &FullConfig, options:" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
  - name: pf-generation-run-summary
    command: |-
      sh -c 'grep -qF "governor.record_within_life_sample(cumulative_food, cumulative_alive);" crates/xagent-sandbox/src/headless.rs && grep -qF "let fitness = governor.evaluate(&agents);" crates/xagent-sandbox/src/headless.rs && grep -qF "agents[i].death_count = state[base + P_DEATH_COUNT] as u32;" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn evolution_complete(&self) -> bool {" crates/xagent-sandbox/src/governor.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs'
  - name: pf-archive-brain-source
    command: |-
      sh -c 'grep -qF "pub enum BrainSource { Fresh, Champion { db_path: String," docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub struct FreeLifeReport { pub agent_count: usize, pub ticks: u64," docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub fn load_champion_read_only(db_path: &str, node_id: Option<i64>) ->" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {" crates/xagent-sandbox/src/headless.rs'
  - name: pf-evolution-cli
    command: |-
      sh -c 'grep -qF "seed: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "headless::run_headless(config, &cli.db, cli.resume, true);" crates/xagent-sandbox/src/main.rs && grep -qF "fn resolve_config(cli: &Cli) -> FullConfig {" crates/xagent-sandbox/src/main.rs && grep -qF "#[arg(long)]" crates/xagent-sandbox/src/main.rs && grep -qF "eprintln!(\"xagent v0.1.0" crates/xagent-sandbox/src/main.rs && grep -qF "[--free-life-champion-db PATH [--free-life-champion-node ID]]" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
  - name: pf-worker-supervisor
    command: |-
      sh -c 'grep -qF "pub struct SimInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn advance_compute(&mut self, dt: f64) -> bool {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn test_init(tick_budget: u64, paused: bool, speed_multiplier: u32) -> SimInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "pub fn read_full_state_blocking(&mut self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "let init = SimInit {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::Error(message) => self.log_msg(format!(\"[GPU] {message}\"))," crates/xagent-sandbox/src/gpu_orchestration.rs'
  - name: pf-gui-supervisor-wiring
    command: |-
      sh -c 'grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "impl Default for EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "fn render_running_dashboard(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {" crates/xagent-sandbox/src/ui.rs && grep -qF "let init = SimInit {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "EvolutionAction::Start => {" crates/xagent-sandbox/src/evolution.rs && grep -qF "fn evolution_snapshot_default_tree_pane_fraction() {" crates/xagent-sandbox/tests/integration.rs'
  - name: pf-supervisor-verdict
    command: |-
      sh -c 'grep -qF "[[bin]]" crates/xagent-sandbox/Cargo.toml && grep -qF "clap = { version = " crates/xagent-sandbox/Cargo.toml && grep -qF "default-run = \"xagent\"" docs/plans/0004-food-contingency-learning/ARCHITECTURE.md && grep -qF "src/bin/food_kind_verdict.rs" docs/plans/0004-food-contingency-learning/PLAN.md && grep -qF "sighting_meal_share, turn_toward_food" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && test ! -e crates/xagent-sandbox/src/bin/supervisor_verdict.rs'
  - name: pf-supervisor-decision
    command: |-
      sh -c 'grep -qF "pub fn run_bench(" crates/xagent-sandbox/src/bench.rs && grep -qF "bench: bool," crates/xagent-sandbox/src/main.rs && grep -qF "bench_ticks: u64," crates/xagent-sandbox/src/main.rs && grep -qF "bench_agents: usize," crates/xagent-sandbox/src/main.rs && grep -qF "dump_config: bool," crates/xagent-sandbox/src/main.rs && grep -qxF "/target" .gitignore && test ! -e docs/measurements/continuous-supervisor'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
execution_checks:
  - name: developer-route
    tasks:
      - evolution-mode-config
      - run-mode-column
      - run-summary
      - supervisor-core
      - supervisor-store
      - kernel-supervisor-hooks
      - supervisor-run
      - headless-supervisor-driver
      - generation-run-summary
      - archive-brain-source
      - evolution-cli
      - worker-supervisor
      - gui-supervisor-wiring
      - supervisor-verdict
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: evolution-mode-config
        name: governor-config-and-run-schema
        sources:
          - crates/xagent-shared/src/config.rs
          - crates/xagent-shared/src/lib.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/tests/integration.rs
          - docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md
        checks:
          - name: pf-evolution-mode-config
            command: |-
              sh -c 'grep -qF "pub struct GovernorConfig {" crates/xagent-shared/src/config.rs && grep -qF "impl Default for GovernorConfig {" crates/xagent-shared/src/config.rs && grep -qF "pub governor: GovernorConfig," crates/xagent-shared/src/config.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs && grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "fn encode_brain_state(state: &AgentBrainState) -> Option<Vec<u8>> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn init_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "ALTER TABLE run ADD COLUMN simulation_ticks INTEGER DEFAULT 0;" crates/xagent-sandbox/src/governor.rs && grep -qF "let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;" crates/xagent-sandbox/src/governor.rs && grep -qF "let config: GovernorConfig = serde_json::from_str(&governor_json).unwrap_or_default();" crates/xagent-sandbox/src/governor.rs && grep -qF "let gov_config = GovernorConfig {" crates/xagent-sandbox/tests/integration.rs && grep -qF "XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: run-mode-column
        name: run-schema-resume-and-fitness-hooks
        sources:
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/Cargo.toml
          - docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-run-mode-column
            command: |-
              sh -c 'grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn init_schema(db: &Connection) -> SqlResult<()> {" crates/xagent-sandbox/src/governor.rs && grep -qF "ALTER TABLE run ADD COLUMN simulation_ticks INTEGER DEFAULT 0;" crates/xagent-sandbox/src/governor.rs && grep -qF "let total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32;" crates/xagent-sandbox/src/governor.rs && grep -qF "FROM run ORDER BY id DESC LIMIT 1" crates/xagent-sandbox/src/governor.rs && grep -qF "pub fn resume(db_path: &str) -> SqlResult<Self> {" crates/xagent-sandbox/src/governor.rs && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && grep -qF "XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: run-summary
        name: food-kind-metric-helpers-and-physics-slots
        sources:
          - docs/plans/0004-food-contingency-learning/tasks/0301-food-kind-metrics.md
          - docs/plans/0004-food-contingency-learning/ARCHITECTURE.md
          - crates/xagent-shared/src/lib.rs
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-sandbox/src/lib.rs
        checks:
          - name: pf-run-summary
            command: |-
              sh -c 'grep -qF "pub fn nearest_food_of_kind(" docs/plans/0004-food-contingency-learning/tasks/0301-food-kind-metrics.md && grep -qF "pub struct NearestFood { pub distance: f32, pub bearing: f32 }" docs/plans/0004-food-contingency-learning/tasks/0301-food-kind-metrics.md && grep -qF "P_HARMFUL_MEAL_COUNT" docs/plans/0004-food-contingency-learning/ARCHITECTURE.md && grep -qF "SCENT_DECAY_LENGTH, SCENT_RANGE," crates/xagent-shared/src/lib.rs && grep -qF "pub const P_FOOD_COUNT: usize = 18;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_DEATH_COUNT: usize = 23;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub mod replay;" crates/xagent-sandbox/src/lib.rs && test ! -e crates/xagent-sandbox/src/run_summary.rs'
      - task: supervisor-core
        name: fitness-heatmap-and-mutation-apis
        sources:
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/agent/mod.rs
          - crates/xagent-sandbox/src/momentum.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-supervisor-core
            command: |-
              sh -c 'grep -qF "fn composite_fitness(" crates/xagent-sandbox/src/governor.rs && grep -qF "pub const HEATMAP_RES: usize = 64;" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_steering_weights(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_sensory_genes(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn new(decay: f32) -> Self {" crates/xagent-sandbox/src/momentum.rs && grep -qF "rand = { workspace = true }" crates/xagent-sandbox/Cargo.toml && test ! -e crates/xagent-sandbox/src/supervisor.rs'
      - task: supervisor-store
        name: blob-codec-schema-and-sqlite
        sources:
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/Cargo.toml
          - docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md
        checks:
          - name: pf-supervisor-store
            command: |-
              sh -c 'grep -qF "fn encode_brain_state(state: &AgentBrainState) -> Option<Vec<u8>> {" crates/xagent-sandbox/src/governor.rs && grep -qF "fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {" crates/xagent-sandbox/src/governor.rs && grep -qF "CREATE TABLE IF NOT EXISTS node_brain (" crates/xagent-sandbox/src/governor.rs && grep -qF "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;" crates/xagent-sandbox/src/governor.rs && grep -qF "rusqlite = { version = \"0.34\", features = [\"bundled\"] }" crates/xagent-sandbox/Cargo.toml && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml && grep -qF "pub fn load_champion_read_only(db_path: &str, node_id: Option<i64>) ->" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && test ! -e crates/xagent-sandbox/src/supervisor_store.rs'
      - task: kernel-supervisor-hooks
        name: single-slot-writes-and-world-config-upload
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - docs/plans/0004-food-contingency-learning/tasks/0105-kernel-food-kind-draw.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-kernel-supervisor-hooks
            command: |-
              sh -c 'grep -qF "pub fn write_agent_state(&self, index: u32, state: &AgentBrainState) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn write_agent_heritable_config(&self, index: u32, config: &BrainConfig) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn write_agent_physics_fields(&self, index: u32, fields: &[(usize, f32)]) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn upload_agents(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "data[base + P_PREV_INTEGRITY] = max_i;" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn write_world_config_scratch(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn set_freeze_steering_weights(&mut self, freeze: bool) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn is_available() -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn nourishing_food_kind(&self) -> Option<FoodKind>" docs/plans/0004-food-contingency-learning/tasks/0105-kernel-food-kind-draw.md && grep -qF "food_kinds_slot_value" docs/plans/0004-food-contingency-learning/tasks/0105-kernel-food-kind-draw.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: supervisor-run
        name: dispatch-readback-and-spawn-apis
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/src/world/mod.rs
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-supervisor-run
            command: |-
              sh -c 'grep -qF "pub fn kernel_batch_size(&self) -> u32 {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn try_collect_state_snapshot(&mut self) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn cached_state(&self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn probe_kernel_batches(&self) -> u64 {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn safe_spawn_position(&self) -> glam::Vec3 {" crates/xagent-sandbox/src/world/mod.rs && grep -qF "const HEATMAP_INTERVAL: u32 = 100;" crates/xagent-sandbox/src/headless.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: headless-supervisor-driver
        name: headless-entry-and-free-life-interface
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-headless-supervisor-driver
            command: |-
              sh -c 'grep -qF "pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {" crates/xagent-sandbox/src/headless.rs && grep -qF "let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);" crates/xagent-sandbox/src/headless.rs && grep -qF "while !kernel.try_collect_state_snapshot() {" crates/xagent-sandbox/src/headless.rs && grep -qF "run_free_life(config: &FullConfig, options:" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: generation-run-summary
        name: generation-loop-measurement-sites
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-generation-run-summary
            command: |-
              sh -c 'grep -qF "governor.record_within_life_sample(cumulative_food, cumulative_alive);" crates/xagent-sandbox/src/headless.rs && grep -qF "let fitness = governor.evaluate(&agents);" crates/xagent-sandbox/src/headless.rs && grep -qF "agents[i].death_count = state[base + P_DEATH_COUNT] as u32;" crates/xagent-sandbox/src/headless.rs && grep -qF "pub fn evolution_complete(&self) -> bool {" crates/xagent-sandbox/src/governor.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: archive-brain-source
        name: free-life-brain-source-interface
        sources:
          - docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-archive-brain-source
            command: |-
              sh -c 'grep -qF "pub enum BrainSource { Fresh, Champion { db_path: String," docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub struct FreeLifeReport { pub agent_count: usize, pub ticks: u64," docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub fn load_champion_read_only(db_path: &str, node_id: Option<i64>) ->" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && grep -qF "pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {" crates/xagent-sandbox/src/headless.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: evolution-cli
        name: clap-cli-and-headless-dispatch
        sources:
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/Cargo.toml
          - docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md
        checks:
          - name: pf-evolution-cli
            command: |-
              sh -c 'grep -qF "seed: Option<u64>," crates/xagent-sandbox/src/main.rs && grep -qF "headless::run_headless(config, &cli.db, cli.resume, true);" crates/xagent-sandbox/src/main.rs && grep -qF "fn resolve_config(cli: &Cli) -> FullConfig {" crates/xagent-sandbox/src/main.rs && grep -qF "#[arg(long)]" crates/xagent-sandbox/src/main.rs && grep -qF "eprintln!(\"xagent v0.1.0" crates/xagent-sandbox/src/main.rs && grep -qF "[--free-life-champion-db PATH [--free-life-champion-node ID]]" docs/plans/0001-free-life-harness-and-trunk-lesions/PLAN.md && grep -qF "tempfile = \"3\"" crates/xagent-sandbox/Cargo.toml'
      - task: worker-supervisor
        name: worker-cadence-and-blocking-readback
        sources:
          - crates/xagent-sandbox/src/sim_runtime.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-worker-supervisor
            command: |-
              sh -c 'grep -qF "pub struct SimInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn clamp_ticks_to_generation_budget(" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "GenerationBudgetReached(StateSnapshot)," crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn advance_compute(&mut self, dt: f64) -> bool {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "fn test_init(tick_budget: u64, paused: bool, speed_multiplier: u32) -> SimInit {" crates/xagent-sandbox/src/sim_runtime.rs && grep -qF "pub fn read_full_state_blocking(&mut self) -> &[f32] {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "let init = SimInit {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::Error(message) => self.log_msg(format!(\"[GPU] {message}\"))," crates/xagent-sandbox/src/gpu_orchestration.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: gui-supervisor-wiring
        name: evolution-snapshot-and-main-thread-orchestration
        sources:
          - crates/xagent-sandbox/src/ui.rs
          - crates/xagent-sandbox/src/gpu_orchestration.rs
          - crates/xagent-sandbox/src/evolution.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-gui-supervisor-wiring
            command: |-
              sh -c 'grep -qF "pub struct EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "impl Default for EvolutionSnapshot {" crates/xagent-sandbox/src/ui.rs && grep -qF "fn render_running_dashboard(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {" crates/xagent-sandbox/src/ui.rs && grep -qF "let init = SimInit {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "SimEvent::GenerationBudgetReached(snapshot) => {" crates/xagent-sandbox/src/gpu_orchestration.rs && grep -qF "EvolutionAction::Start => {" crates/xagent-sandbox/src/evolution.rs && grep -qF "fn evolution_snapshot_default_tree_pane_fraction() {" crates/xagent-sandbox/tests/integration.rs'
      - task: supervisor-verdict
        name: second-binary-slot-and-report-shapes
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - docs/plans/0004-food-contingency-learning/ARCHITECTURE.md
          - docs/plans/0004-food-contingency-learning/PLAN.md
          - docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md
        checks:
          - name: pf-supervisor-verdict
            command: |-
              sh -c 'grep -qF "[[bin]]" crates/xagent-sandbox/Cargo.toml && grep -qF "clap = { version = " crates/xagent-sandbox/Cargo.toml && grep -qF "default-run = \"xagent\"" docs/plans/0004-food-contingency-learning/ARCHITECTURE.md && grep -qF "src/bin/food_kind_verdict.rs" docs/plans/0004-food-contingency-learning/PLAN.md && grep -qF "sighting_meal_share, turn_toward_food" docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md && test ! -e crates/xagent-sandbox/src/bin/supervisor_verdict.rs'
  - name: decision-route
    tasks:
      - supervisor-decision
    role: architect
    operations:
      - edit
      - shell
    command: cargo run --release -p xagent-sandbox -- --bench --bench-ticks 2000 --bench-agents 10
    timeout_ms: 3600000
    actions:
      - name: release-build
        command: cargo build --release -p xagent-sandbox
        timeout_ms: 3600000
    constraints:
      - task: supervisor-decision
        name: release-gpu-run-route
        sources:
          - crates/xagent-sandbox/src/bench.rs
          - crates/xagent-sandbox/src/main.rs
          - .gitignore
        checks:
          - name: pf-supervisor-decision
            command: |-
              sh -c 'grep -qF "pub fn run_bench(" crates/xagent-sandbox/src/bench.rs && grep -qF "bench: bool," crates/xagent-sandbox/src/main.rs && grep -qF "bench_ticks: u64," crates/xagent-sandbox/src/main.rs && grep -qF "bench_agents: usize," crates/xagent-sandbox/src/main.rs && grep -qF "dump_config: bool," crates/xagent-sandbox/src/main.rs && grep -qxF "/target" .gitignore && test ! -e docs/measurements/continuous-supervisor'
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

# 0005 Continuous life supervisor

## Scope

An opt-in evolution mode, off by default, in which one long-lived world keeps
the same 10 agents and a supervisor refills one slot at a time, plus the
measurements and the fail-closed verdict that decide whether to adopt it. Task
ids below are the packet slugs in `tasks/`; ARCHITECTURE.md is the binding
specification.

1. **Config (`evolution-mode-config`) and run record (`run-mode-column`).**
   `GovernorConfig.evolution_mode: EvolutionMode` (`Generations` is the serde
   default) and `GovernorConfig.supervisor: SupervisorConfig` with serde
   defaults for every field: `boundary_interval_ticks`, `min_age_ticks`,
   `score_window_ticks`, `archive_capacity`, `season_period_ticks`,
   `inheritance` (`Darwinian` default, `WeightCopy` comparison arm) and
   `agent_tick_budget` (0 = unlimited). Supervisor settings live only in
   `GovernorConfig`; they are never part of `BrainConfig` and never mutated or
   inherited; the config task also completes every existing full
   `GovernorConfig` literal. In a second task, `governor.rs` adds the `mode` column on `run` with the existing
   idempotent `ALTER TABLE` pattern, refuses to resume a supervisor run, and
   makes `composite_fitness`, the brain-blob codec, `init_schema` and the
   exploration grid size crate-visible. A test proves databases written by
   generation mode keep loading.
2. **In-run measurement (`run-summary`).** A pure module with the serialisable
   per-run summary both drivers write (meals, harmful meals, deaths and
   agent-ticks; with food kinds on, turn and meal-share accumulators judged
   against whichever kind is nourishing, with early and late windows taken
   inside each season).
3. **Supervisor core (`supervisor-core`).** A pure, deterministic module,
   `crates/xagent-sandbox/src/supervisor.rs`, unit-tested without a GPU:
   boundary and season schedules on separate seeded streams, the per-slot
   ledger, eligibility, the seeded tie-break, the bounded archive ranked by each
   archived birth brain's own life outcome, the parent draw and the mutation.
   The window score reuses `composite_fitness` restricted to the window and is
   used only for selection.
4. **Persistence (`supervisor-store`).** New tables created idempotently
   (`supervisor_slot`, `supervisor_archive` with birth brains as blobs through
   `encode_brain_state`, `supervisor_event`), a supervisor run row with
   `mode = 'supervisor'`, and a read-only archive loader.
5. **Kernel hooks (`kernel-supervisor-hooks`).** `GpuKernel` gains a
   nourishing-kind setter that updates the world-config uniform without
   resetting any agent or consuming a draw, and a single-slot replacement built
   only from the existing single-slot writes that resets the slot's physics row
   to its spawn values.
6. **Shared driver (`supervisor-run`).** The kernel-facing step driver both the
   headless loop and the GUI simulation worker use, so both produce the same
   events for the same seed.
7. **Headless drivers (`headless-supervisor-driver`, `generation-run-summary`,
   `archive-brain-source`).** `run_headless` picks the driver by mode; the
   generation driver can write the same run summary as a measurement only; the
   0001 free-life harness gains a `BrainSource::Archive` source.
8. **CLI and GUI (`evolution-cli`, `worker-supervisor`,
   `gui-supervisor-wiring`).** `--evolution-mode`, `--agent-tick-budget`,
   `--supervisor-inheritance`, `--run-summary` and the free-life archive flags;
   the simulation worker owns the supervisor and its boundary readback; the main
   thread only receives events; `EvolutionSnapshot` gains one optional field.
9. **Decision rule and record (`supervisor-verdict`, `supervisor-decision`).**
   The rule below is encoded in the `supervisor_verdict` binary before any run.
   The architect then runs the evidence and records it in
   `docs/measurements/continuous-supervisor/DECISION.md` and `results.json`;
   run databases stay in ignored `target/continuous-supervisor/`.

## Decision rule (fixed before any run; encoded in `supervisor_verdict`)

Runs use seeds 5, 6, 7 and 8, 10 agents and exactly 32,000,000 agent-ticks per
run (80 generations of 40,000 ticks in generation mode; 3,200,000 ticks in
supervisor mode). The base config is the `--dump-config` default with
`governor.tick_budget = 40000`; every supervisor setting stays at its default.

Run families, per seed:

- `generations`, `supervisor` (Darwinian), `weight-copy` (supervisor with
  `WeightCopy` inheritance): food kinds off.
- `generations-kinds`, `supervisor-kinds` (Darwinian): food kinds and the
  second odour sense on.
- `free-life-generations` (0001 free life, champion source: the generation
  database's default champion), `free-life-supervisor` and
  `free-life-weight-copy` (archive source, rank 0): 10 agents, 300,000 ticks,
  food kinds off.

Metrics (supervisor `S` against generations `G`):

- Per seed: food per agent-tick `F = nourishing meals / agent-ticks` and deaths
  per agent-tick `D = deaths / agent-ticks`, from the food-kinds-off runs.
- Within-life learning, pooled over the four seeds, late-30% windows:
  - from the free lives: `L1 = Σ late.meals / Σ late.alive_samples`,
    `L2 = mean late.turn_toward_food`, `L3 = mean late.sighting_meal_share`
    (means over all 40 agents);
  - from the food-kinds runs, late window = last 30% of each season (a
    generation is a season in generation mode): `K1` = pooled mean turn toward
    the nourishing kind, `K2` = pooled mean turn away from the harmful kind,
    `K3 = Σ late nourishing meals / Σ late meals`.

Verdict:

- `ADOPT` when `F_S ≥ F_G` and `D_S ≤ D_G` together hold on at least 3 of the 4
  seeds, and `L1–L3` and `K1–K3` of `S` are each `≥` those of `G`.
- `OPTION` when not `ADOPT`, but nothing gets worse: pooled `F_S ≥ F_G`, pooled
  `D_S ≤ D_G` (pooled = sums over the four seeds) and every `L` and `K` of `S`
  is `≥` that of `G`.
- `REJECT` otherwise.

The `weight-copy` arm is measured and reported with the same metrics but never
enters the verdict and is never adopted. The verdict fails closed: a missing,
duplicate, malformed, mismatched or undefined input is an error, not a verdict.
`OPTION` and `REJECT` leave the flag off; even `ADOPT` changes no default here
(removing generation mode is a later plan).

## Prerequisite interfaces relied on

- From 0001-free-life-harness-and-trunk-lesions: the seeded headless digest test
  `crates/xagent-sandbox/tests/seeded_headless_digest.rs` (strict mode under
  `XAGENT_REQUIRE_SEEDED_DIGEST=1`) as the bit-identity proof; `run_free_life`
  with `FreeLifeOptions`, `BrainSource` and `FreeLifeReport` in `headless.rs`;
  `free_life::metrics::{AgentMetrics, WindowMetrics}`; the `--free-life*` CLI
  flags in `main.rs`; `governor::load_champion_read_only`.
- From 0004-food-contingency-learning: `WorldConfig.food_kinds_enabled`,
  `BrainConfig.second_odour_sense_enabled`, `xagent_shared::FoodKind`,
  `WorldConfig::nourishing_food_kind`, the kernel's nourishing-kind draw
  (`GpuKernel::nourishing_food_kind`, `WC_FOOD_KINDS`, `food_kinds_slot_value`),
  `P_HARMFUL_MEAL_COUNT`, the public `food_kind_metrics::nearest_food_of_kind`
  and `food_kind_metrics::NearestFood` (only its `bearing` field is read; the
  approach sign rule `-signum(bearing)` with `signum(0) = 0` is re-stated as a
  crate-local helper in `run_summary.rs`, because 0004 does not export its
  own), the `food_kind_verdict` binary layout (`default-run =
  "xagent"`), the parity test `sense_block_parity` and the default check
  `food_kind_draw`.

## GPU proofs fail closed

Every check that runs a GPU-backed test sets `XAGENT_REQUIRE_GPU=1`, and every
new GPU test starts with the fail-closed helper pattern of
`gpu_adapter_present_when_required`: it returns early only when no adapter
exists and the variable is unset, and panics when it is set.

## Non-goals

- No change to any default: `evolution_mode` stays `Generations`, food kinds
  stay off, and no preset turns the supervisor on.
- No removal or behaviour change of generation mode; with the flag off the 0001
  digest, the 0004 parity golden and the 0004 default check are unchanged.
- No new learning rule, reward term, reward shaping, hand-coded reflex,
  action-conditioned model or structural prior; no WGSL edit. The only learning
  signal stays the agent's own homeostatic change. The window score is a
  selection measure only.
- No extra agents, worlds, kernels or GPU dispatches in supervisor mode.
- No resume of supervisor runs (refused with an error).
- No adoption of the weight-copy arm, whatever its numbers.
