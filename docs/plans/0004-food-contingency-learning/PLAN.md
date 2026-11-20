---
objective: >-
  Prove the seed on a new sense plus a new world condition: two food kinds that
  differ in colour and odour, one nourishing and one harmful, with nothing
  telling the agent which is which; a fresh brain must learn within one life to
  approach one and avoid the other. The kind assignment is redrawn per
  generation and per seeded free-life run; the second nostril pair enters only
  as a declared sense whose weights are learned-only and zero at birth. Gate:
  within one 300,000-tick life on seeds 5-8, fresh brains come to turn toward
  the nourishing kind and away from the harmful one, and eat a rising share of
  the nourishing kind, against a frozen-policy control on the same seeds. Ends
  with a decision record; a negative result names which link failed (readout,
  credit, or weight change).
depends_on:
  - 0003-uniform-sense-motor-route
status: todo
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
  - name: sense-parity
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
    timeout_ms: 5400000
preflight_checks:
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: pf-config-serde-defaults
    command: |-
      sh -c 'grep -qF "pub struct WorldConfig {" crates/xagent-shared/src/config.rs && grep -qF "pub struct BrainConfig {" crates/xagent-shared/src/config.rs && grep -qF "fn default_seed() -> u64 {" crates/xagent-shared/src/config.rs && grep -qF "pub fn crossover_config(a: &BrainConfig, b: &BrainConfig) -> BrainConfig {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs'
  - name: pf-sense-interface
    command: |-
      sh -c 'grep -qF "pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "They may declare only" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "with_senses(config, senses:" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "pub const SCENT_FEATURE_OFFSET: usize = 25;" crates/xagent-brain/src/senses.rs && grep -qF "pub const SCENT_CHANNELS: usize = 2;" crates/xagent-brain/src/buffers.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs'
  - name: pf-food-kind-layout
    command: |-
      sh -c 'grep -qF "pub const WC_DANGER_PERCEPT_BLINDED: usize = 26;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const WORLD_CONFIG_SIZE: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "array<vec4<f32>, 7>;" crates/xagent-brain/src/shaders/kernel/common.wgsl && test "$(grep -c "WC_[A-Z_]*: usize = 27;" crates/xagent-brain/src/buffers.rs)" = 0 && grep -qF "pub const PHYS_STRIDE: usize = 53;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_FOOD_CLAIM: usize = 52;" crates/xagent-brain/src/buffers.rs && grep -qF "const PHYS_STRIDE: u32 = 53u;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "const ENERGY_WEIGHT: f32 = 0.6;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "const INTEGRITY_WEIGHT: f32 = 0.4;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "P_FOOD_COUNT" crates/xagent-brain/src/gpu_kernel/rounding_validation.rs'
  - name: pf-parity-physics-hash
    command: |-
      sh -c 'grep -qF "Hash the full physics state" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md && grep -qF "pub const PHYS_STRIDE: usize = 53;" crates/xagent-brain/src/buffers.rs'
  - name: pf-extra-sensory-layout
    command: |-
      sh -c 'grep -qF "override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u + NON_VISUAL_FEATURE_CHANNELS;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "override NON_VISUAL_FEATURE_COUNT: u32 = 25u + NON_VISUAL_FEATURE_CHANNELS + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "override DANGER_PERCEPT_FEATURES_ACTIVE: u32 = 0u;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "let danger_feature_width = if danger_percept_enabled { 2 } else { 0 };" crates/xagent-brain/src/buffers.rs && grep -qF "pub fn from_config(config: &BrainConfig) -> Self {" crates/xagent-brain/src/buffers.rs && grep -qF "pub const NON_VISUAL_COUNT: usize =" crates/xagent-brain/src/buffers.rs && grep -qF "Hash the full physics state" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md'
  - name: pf-kernel-draw-hooks
    command: |-
      sh -c 'grep -qF "pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn reset_agents(&mut self, brain_config: &BrainConfig) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn write_world_config_scratch(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "self.world_config_scratch[WC_PHASE_MASK] = phase_mask as f32;" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn is_available() -> bool {" crates/xagent-brain/src/gpu_kernel.rs'
  - name: pf-second-odour-kernel-hooks
    command: |-
      sh -c 'grep -qF "fn vision_override_constants(layout: &BrainLayout) -> HashMap<String, f64> {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "\"DANGER_PERCEPT_FEATURES_ACTIVE\".to_string()," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn vision_override_map_covers_root_constants() {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn new_with_senses(agent_count: u32, food_count: usize, brain_config:" docs/plans/0003-uniform-sense-motor-route/tasks/0204-sense-override-constants.md'
  - name: pf-food-claim-site
    command: |-
      sh -c 'grep -qF "physics_state[b + P_ENERGY] += wc_f32(WC_FOOD_ENERGY);" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF "physics_state[b + P_FOOD_COUNT] += 1.0;" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF "resolve_food_claim(agent_id);" crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl && grep -qF "resolve_food_claim" crates/xagent-brain/src/shaders/kernel/physics_tick.wgsl && grep -qF "resolve_food_claim" crates/xagent-brain/src/shaders/kernel/persistent_tick.wgsl && grep -qF "P_FOOD_COUNT" crates/xagent-brain/src/shaders/kernel/phase_death.wgsl'
  - name: pf-vision-food-colour
    command: |-
      sh -c 'for f in phase_vision.wgsl phase_vision_parallel.wgsl phase_vision_object_queries.wgsl; do test "$(grep -cF "hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);" crates/xagent-brain/src/shaders/kernel/$f)" = 1 || exit 1; done && test -f crates/xagent-brain/src/gpu_kernel/vision_validation.rs'
  - name: pf-vision-scent-sites
    command: |-
      sh -c 'test "$(grep -cF "let scent = sense_scent(pos, facing, smell_strength);" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl)" = 1 && grep -qF "sensory_buffer[scent_base + 1u] = scent.y;" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl && grep -qF "fn vision_prepare_scent(agent_id: u32, tid: u32, group_threads: u32) {" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF "if alive && food_flags[food] == 0u {" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF "fn with_prepared_scent(source: &str) -> String {" crates/xagent-brain/src/gpu_kernel.rs'
  - name: pf-feature-extract-sites
    command: |-
      sh -c 'grep -qF "s_features[fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF "if (DANGER_PERCEPT_FEATURES_ACTIVE != 0u) {" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF "brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;" crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_feature_phase_for_test(&mut self) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_brain_scratch_blocking(&mut self) -> Vec<f32> {" crates/xagent-brain/src/gpu_kernel.rs'
  - name: pf-inheritance-apis
    command: |-
      sh -c 'grep -qF "pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_steering_weights(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_brain_state_seeded(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "rand = { workspace = true }" crates/xagent-sandbox/Cargo.toml && grep -qF "mutate_steering_weights_for" docs/plans/0003-uniform-sense-motor-route/tasks/0203-descriptor-driven-mutation.md'
  - name: pf-metrics-sources
    command: |-
      sh -c 'grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const FOOD_STATE_STRIDE: usize = 4;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_NEAREST_FOOD_BEARING: usize = 34;" crates/xagent-brain/src/buffers.rs && grep -qF "pub mod headless;" crates/xagent-sandbox/src/lib.rs && grep -qF "serde_json = { workspace = true }" crates/xagent-sandbox/Cargo.toml'
  - name: pf-verdict-binary-slot
    command: |-
      sh -c 'grep -qF "[[bin]]" crates/xagent-sandbox/Cargo.toml && grep -qF "path = " crates/xagent-sandbox/Cargo.toml && test "$(grep -c "default-run" crates/xagent-sandbox/Cargo.toml)" = 0 && test ! -e crates/xagent-sandbox/src/bin/food_kind_verdict.rs && grep -qF "clap = { version = " crates/xagent-sandbox/Cargo.toml'
  - name: pf-readout-crate
    command: |-
      sh -c 'grep -qF "pub mod buffers;" crates/xagent-brain/src/lib.rs && test ! -e crates/xagent-brain/src/kind_readout.rs'
  - name: pf-harness-kernel-api
    command: |-
      sh -c 'grep -qF "pub fn set_freeze_steering_weights(&mut self, freeze: bool) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "#[arg(long)]" crates/xagent-sandbox/src/main.rs && grep -qF "src/main.rs" crates/xagent-sandbox/Cargo.toml'
  - name: pf-decision-run-route
    command: |-
      sh -c 'grep -qF "pub fn run_bench(" crates/xagent-sandbox/src/bench.rs && grep -qF "bench: bool," crates/xagent-sandbox/src/main.rs && grep -qF "bench_ticks: u64," crates/xagent-sandbox/src/main.rs && grep -qF "bench_agents: usize," crates/xagent-sandbox/src/main.rs'
  - name: pf-integration-guard
    command: |-
      sh -c 'test -f crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF "cargo test -p xagent-sandbox" CONTRIBUTING.md && grep -qF "cargo clippy --workspace --all-targets -- -D warnings" CONTRIBUTING.md'
  - name: pf-task-food-kind-layout
    command: "sh -c 'grep -qF \"pub const WC_DANGER_PERCEPT_BLINDED: usize = 26;\" crates/xagent-brain/src/buffers.rs && grep -qF \"pub const WORLD_CONFIG_SIZE: usize = 28;\" crates/xagent-brain/src/buffers.rs && grep -qF \"array<vec4<f32>, 7>;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && test \"$(grep -c \"WC_[A-Z_]*: usize = 27;\" crates/xagent-brain/src/buffers.rs)\" = 0 && grep -qF \"pub const PHYS_STRIDE: usize = 53;\" crates/xagent-brain/src/buffers.rs && grep -qF \"pub const P_FOOD_CLAIM: usize = 52;\" crates/xagent-brain/src/buffers.rs && grep -qF \"const PHYS_STRIDE: u32 = 53u;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"const ENERGY_WEIGHT: f32 = 0.6;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"const INTEGRITY_WEIGHT: f32 = 0.4;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"P_FOOD_COUNT\" crates/xagent-brain/src/gpu_kernel/rounding_validation.rs && grep -qF \"Hash the full physics state\" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md && grep -qF \"pub const PHYS_STRIDE: usize = 53;\" crates/xagent-brain/src/buffers.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-extra-sensory-layout
    command: "sh -c 'grep -qF \"override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u + NON_VISUAL_FEATURE_CHANNELS;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"override NON_VISUAL_FEATURE_COUNT: u32 = 25u + NON_VISUAL_FEATURE_CHANNELS + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"override DANGER_PERCEPT_FEATURES_ACTIVE: u32 = 0u;\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF \"let danger_feature_width = if danger_percept_enabled { 2 } else { 0 };\" crates/xagent-brain/src/buffers.rs && grep -qF \"pub fn from_config(config: &BrainConfig) -> Self {\" crates/xagent-brain/src/buffers.rs && grep -qF \"pub const NON_VISUAL_COUNT: usize =\" crates/xagent-brain/src/buffers.rs && grep -qF \"Hash the full physics state\" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md && grep -qF \"pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];\" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF \"NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }\" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF \"They may declare only\" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF \"with_senses(config, senses:\" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF \"pub const SCENT_FEATURE_OFFSET: usize = 25;\" crates/xagent-brain/src/senses.rs && grep -qF \"pub const SCENT_CHANNELS: usize = 2;\" crates/xagent-brain/src/buffers.rs && grep -qF \"pub use config::{\" crates/xagent-shared/src/lib.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-kernel-food-kind-draw
    command: "sh -c 'grep -qF \"pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn reset_agents(&mut self, brain_config: &BrainConfig) {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"fn write_world_config_scratch(\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"self.world_config_scratch[WC_PHASE_MASK] = phase_mask as f32;\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn is_available() -> bool {\" crates/xagent-brain/src/gpu_kernel.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-second-odour-kernel
    command: "sh -c 'grep -qF \"fn vision_override_constants(layout: &BrainLayout) -> HashMap<String, f64> {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"\\\"DANGER_PERCEPT_FEATURES_ACTIVE\\\".to_string(),\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"fn vision_override_map_covers_root_constants() {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn new_with_senses(agent_count: u32, food_count: usize, brain_config:\" docs/plans/0003-uniform-sense-motor-route/tasks/0204-sense-override-constants.md && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-kernel-food-kind-meals
    command: "sh -c 'grep -qF \"physics_state[b + P_ENERGY] += wc_f32(WC_FOOD_ENERGY);\" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF \"physics_state[b + P_FOOD_COUNT] += 1.0;\" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF \"resolve_food_claim(agent_id);\" crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl && grep -qF \"resolve_food_claim\" crates/xagent-brain/src/shaders/kernel/physics_tick.wgsl && grep -qF \"resolve_food_claim\" crates/xagent-brain/src/shaders/kernel/persistent_tick.wgsl && grep -qF \"P_FOOD_COUNT\" crates/xagent-brain/src/shaders/kernel/phase_death.wgsl && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-food-kind-colour
    command: "sh -c 'for f in phase_vision.wgsl phase_vision_parallel.wgsl phase_vision_object_queries.wgsl; do test \"$(grep -cF \"hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);\" crates/xagent-brain/src/shaders/kernel/$f)\" = 1 || exit 1; done && test -f crates/xagent-brain/src/gpu_kernel/vision_validation.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-food-kind-senses
    command: "sh -c 'test \"$(grep -cF \"let scent = sense_scent(pos, facing, smell_strength);\" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl)\" = 1 && grep -qF \"sensory_buffer[scent_base + 1u] = scent.y;\" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl && grep -qF \"fn vision_prepare_scent(agent_id: u32, tid: u32, group_threads: u32) {\" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF \"if alive && food_flags[food] == 0u {\" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF \"fn with_prepared_scent(source: &str) -> String {\" crates/xagent-brain/src/gpu_kernel.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-second-odour-features
    command: "sh -c 'grep -qF \"s_features[fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF \"if (DANGER_PERCEPT_FEATURES_ACTIVE != 0u) {\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF \"brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;\" crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl && grep -qF \"pub fn read_agent_state(&self, index: u32) -> AgentBrainState {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn dispatch_feature_phase_for_test(&mut self) {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn read_brain_scratch_blocking(&mut self) -> Vec<f32> {\" crates/xagent-brain/src/gpu_kernel.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-food-kind-free-life
    command: "sh -c 'grep -qF \"pub fn set_freeze_steering_weights(&mut self, freeze: bool) {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"pub fn cached_food_state(&self) -> Option<&[f32]> {\" crates/xagent-brain/src/gpu_kernel.rs && grep -qF \"#[arg(long)]\" crates/xagent-sandbox/src/main.rs && grep -qF \"src/main.rs\" crates/xagent-sandbox/Cargo.toml && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
execution_checks:
  - name: developer-route
    tasks:
      - food-kind-config
      - second-odour-sense
      - food-kind-layout
      - extra-sensory-layout
      - kernel-food-kind-draw
      - second-odour-kernel
      - kernel-food-kind-meals
      - food-kind-colour
      - food-kind-senses
      - second-odour-features
      - second-odour-inheritance
      - food-kind-metrics
      - food-kind-verdict
      - kind-readout
      - food-kind-free-life
    role: developer
    operations:
      - edit
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
    constraints:
      - task: food-kind-config
        name: config-serde-defaults
        sources:
          - crates/xagent-shared/src/config.rs
          - crates/xagent-shared/src/lib.rs
          - crates/xagent-sandbox/src/agent/mod.rs
        checks:
          - name: pf-config-serde-defaults
            command: |-
              sh -c 'grep -qF "pub struct WorldConfig {" crates/xagent-shared/src/config.rs && grep -qF "pub struct BrainConfig {" crates/xagent-shared/src/config.rs && grep -qF "fn default_seed() -> u64 {" crates/xagent-shared/src/config.rs && grep -qF "pub fn crossover_config(a: &BrainConfig, b: &BrainConfig) -> BrainConfig {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs'
      - task: second-odour-sense
        name: sense-table-interface
        sources:
          - docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md
          - docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-shared/src/lib.rs
        checks:
          - name: pf-sense-interface
            command: |-
              sh -c 'grep -qF "pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "They may declare only" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "with_senses(config, senses:" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "pub const SCENT_FEATURE_OFFSET: usize = 25;" crates/xagent-brain/src/senses.rs && grep -qF "pub const SCENT_CHANNELS: usize = 2;" crates/xagent-brain/src/buffers.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs'
      - task: food-kind-layout
        name: world-config-slot-physics-stride-and-parity-prefix
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/src/gpu_kernel/rounding_validation.rs
          - docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-food-kind-layout
            command: |-
              sh -c 'grep -qF "pub const WC_DANGER_PERCEPT_BLINDED: usize = 26;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const WORLD_CONFIG_SIZE: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "array<vec4<f32>, 7>;" crates/xagent-brain/src/shaders/kernel/common.wgsl && test "$(grep -c "WC_[A-Z_]*: usize = 27;" crates/xagent-brain/src/buffers.rs)" = 0 && grep -qF "pub const PHYS_STRIDE: usize = 53;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_FOOD_CLAIM: usize = 52;" crates/xagent-brain/src/buffers.rs && grep -qF "const PHYS_STRIDE: u32 = 53u;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "const ENERGY_WEIGHT: f32 = 0.6;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "const INTEGRITY_WEIGHT: f32 = 0.4;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "P_FOOD_COUNT" crates/xagent-brain/src/gpu_kernel/rounding_validation.rs'
          - name: pf-parity-physics-hash
            command: |-
              sh -c 'grep -qF "Hash the full physics state" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md && grep -qF "pub const PHYS_STRIDE: usize = 53;" crates/xagent-brain/src/buffers.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: extra-sensory-layout
        name: extra-channel-overrides-and-layout-builder
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md
          - docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-extra-sensory-layout
            command: |-
              sh -c 'grep -qF "override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u + NON_VISUAL_FEATURE_CHANNELS;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "override NON_VISUAL_FEATURE_COUNT: u32 = 25u + NON_VISUAL_FEATURE_CHANNELS + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "override DANGER_PERCEPT_FEATURES_ACTIVE: u32 = 0u;" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF "let danger_feature_width = if danger_percept_enabled { 2 } else { 0 };" crates/xagent-brain/src/buffers.rs && grep -qF "pub fn from_config(config: &BrainConfig) -> Self {" crates/xagent-brain/src/buffers.rs && grep -qF "pub const NON_VISUAL_COUNT: usize =" crates/xagent-brain/src/buffers.rs && grep -qF "Hash the full physics state" docs/plans/0003-uniform-sense-motor-route/tasks/0101-parity-golden.md'
          - name: pf-sense-interface
            command: |-
              sh -c 'grep -qF "pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }" docs/plans/0003-uniform-sense-motor-route/tasks/0201-sense-descriptor.md && grep -qF "They may declare only" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "with_senses(config, senses:" docs/plans/0003-uniform-sense-motor-route/tasks/0202-derived-sense-layout.md && grep -qF "pub const SCENT_FEATURE_OFFSET: usize = 25;" crates/xagent-brain/src/senses.rs && grep -qF "pub const SCENT_CHANNELS: usize = 2;" crates/xagent-brain/src/buffers.rs && grep -qF "pub use config::{" crates/xagent-shared/src/lib.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: kernel-food-kind-draw
        name: kernel-reset-hooks-and-config-scratch
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-kernel-draw-hooks
            command: |-
              sh -c 'grep -qF "pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn reset_agents(&mut self, brain_config: &BrainConfig) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn write_world_config_scratch(" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "self.world_config_scratch[WC_PHASE_MASK] = phase_mask as f32;" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn is_available() -> bool {" crates/xagent-brain/src/gpu_kernel.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: second-odour-kernel
        name: kernel-sense-table-and-override-generator
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - docs/plans/0003-uniform-sense-motor-route/tasks/0204-sense-override-constants.md
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-second-odour-kernel-hooks
            command: |-
              sh -c 'grep -qF "fn vision_override_constants(layout: &BrainLayout) -> HashMap<String, f64> {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "\"DANGER_PERCEPT_FEATURES_ACTIVE\".to_string()," crates/xagent-brain/src/gpu_kernel.rs && grep -qF "fn vision_override_map_covers_root_constants() {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_ticks(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn new_with_senses(agent_count: u32, food_count: usize, brain_config:" docs/plans/0003-uniform-sense-motor-route/tasks/0204-sense-override-constants.md'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: kernel-food-kind-meals
        name: food-claim-callers-and-death-preservation
        sources:
          - crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl
          - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
          - crates/xagent-brain/src/shaders/kernel/physics_tick.wgsl
          - crates/xagent-brain/src/shaders/kernel/persistent_tick.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-food-claim-site
            command: |-
              sh -c 'grep -qF "physics_state[b + P_ENERGY] += wc_f32(WC_FOOD_ENERGY);" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF "physics_state[b + P_FOOD_COUNT] += 1.0;" crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl && grep -qF "resolve_food_claim(agent_id);" crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl && grep -qF "resolve_food_claim" crates/xagent-brain/src/shaders/kernel/physics_tick.wgsl && grep -qF "resolve_food_claim" crates/xagent-brain/src/shaders/kernel/persistent_tick.wgsl && grep -qF "P_FOOD_COUNT" crates/xagent-brain/src/shaders/kernel/phase_death.wgsl'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: food-kind-colour
        name: vision-colour-sites
        sources:
          - crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_vision_parallel.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_vision_object_queries.wgsl
          - crates/xagent-brain/src/gpu_kernel/vision_validation.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-vision-food-colour
            command: |-
              sh -c 'for f in phase_vision.wgsl phase_vision_parallel.wgsl phase_vision_object_queries.wgsl; do test "$(grep -cF "hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);" crates/xagent-brain/src/shaders/kernel/$f)" = 1 || exit 1; done && test -f crates/xagent-brain/src/gpu_kernel/vision_validation.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: food-kind-senses
        name: scent-sites-and-redirect
        sources:
          - crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-vision-scent-sites
            command: |-
              sh -c 'test "$(grep -cF "let scent = sense_scent(pos, facing, smell_strength);" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl)" = 1 && grep -qF "sensory_buffer[scent_base + 1u] = scent.y;" crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl && grep -qF "fn vision_prepare_scent(agent_id: u32, tid: u32, group_threads: u32) {" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF "if alive && food_flags[food] == 0u {" crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl && grep -qF "fn with_prepared_scent(source: &str) -> String {" crates/xagent-brain/src/gpu_kernel.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: second-odour-features
        name: feature-extract-sites-and-readback
        sources:
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-feature-extract-sites
            command: |-
              sh -c 'grep -qF "s_features[fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF "if (DANGER_PERCEPT_FEATURES_ACTIVE != 0u) {" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF "brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;" crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl && grep -qF "pub fn read_agent_state(&self, index: u32) -> AgentBrainState {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn dispatch_feature_phase_for_test(&mut self) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn read_brain_scratch_blocking(&mut self) -> Vec<f32> {" crates/xagent-brain/src/gpu_kernel.rs'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: second-odour-inheritance
        name: birth-and-mutation-apis
        sources:
          - crates/xagent-sandbox/src/agent/mod.rs
          - crates/xagent-sandbox/Cargo.toml
          - docs/plans/0003-uniform-sense-motor-route/tasks/0203-descriptor-driven-mutation.md
        checks:
          - name: pf-inheritance-apis
            command: |-
              sh -c 'grep -qF "pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_steering_weights(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "pub fn mutate_brain_state_seeded(" crates/xagent-sandbox/src/agent/mod.rs && grep -qF "rand = { workspace = true }" crates/xagent-sandbox/Cargo.toml && grep -qF "mutate_steering_weights_for" docs/plans/0003-uniform-sense-motor-route/tasks/0203-descriptor-driven-mutation.md'
      - task: food-kind-metrics
        name: metrics-inputs
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-sandbox/src/lib.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-metrics-sources
            command: |-
              sh -c 'grep -qF "pub const P_MOTOR_TURN_OUT: usize = 28;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const FOOD_STATE_STRIDE: usize = 4;" crates/xagent-brain/src/buffers.rs && grep -qF "pub const P_NEAREST_FOOD_BEARING: usize = 34;" crates/xagent-brain/src/buffers.rs && grep -qF "pub mod headless;" crates/xagent-sandbox/src/lib.rs && grep -qF "serde_json = { workspace = true }" crates/xagent-sandbox/Cargo.toml'
      - task: food-kind-verdict
        name: second-binary-target-slot
        sources:
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-verdict-binary-slot
            command: |-
              sh -c 'grep -qF "[[bin]]" crates/xagent-sandbox/Cargo.toml && grep -qF "path = " crates/xagent-sandbox/Cargo.toml && test "$(grep -c "default-run" crates/xagent-sandbox/Cargo.toml)" = 0 && test ! -e crates/xagent-sandbox/src/bin/food_kind_verdict.rs && grep -qF "clap = { version = " crates/xagent-sandbox/Cargo.toml'
      - task: kind-readout
        name: readout-module-slot
        sources:
          - crates/xagent-brain/src/lib.rs
        checks:
          - name: pf-readout-crate
            command: |-
              sh -c 'grep -qF "pub mod buffers;" crates/xagent-brain/src/lib.rs && test ! -e crates/xagent-brain/src/kind_readout.rs'
      - task: food-kind-free-life
        name: harness-kernel-api
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: pf-harness-kernel-api
            command: |-
              sh -c 'grep -qF "pub fn set_freeze_steering_weights(&mut self, freeze: bool) {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "pub fn cached_food_state(&self) -> Option<&[f32]> {" crates/xagent-brain/src/gpu_kernel.rs && grep -qF "#[arg(long)]" crates/xagent-sandbox/src/main.rs && grep -qF "src/main.rs" crates/xagent-sandbox/Cargo.toml'
          - name: gpu-adapter-present
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
  - name: decision-route
    tasks:
      - food-contingency-decision
    role: architect
    operations:
      - edit
      - shell
    command: cargo run --release -p xagent-sandbox --bin xagent -- --bench --bench-ticks 2000 --bench-agents 2
    timeout_ms: 3600000
    actions:
      - name: release-gpu-bench
        command: cargo build --release -p xagent-sandbox
        timeout_ms: 3600000
    constraints:
      - task: food-contingency-decision
        name: release-gpu-run-route
        sources:
          - crates/xagent-sandbox/src/bench.rs
          - crates/xagent-sandbox/src/main.rs
        checks:
          - name: pf-decision-run-route
            command: |-
              sh -c 'grep -qF "pub fn run_bench(" crates/xagent-sandbox/src/bench.rs && grep -qF "bench: bool," crates/xagent-sandbox/src/main.rs && grep -qF "bench_ticks: u64," crates/xagent-sandbox/src/main.rs && grep -qF "bench_agents: usize," crates/xagent-sandbox/src/main.rs'
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

# 0004 Food contingency learning

## Scope

An opt-in food-kind scenario, off by default, and the measurements that judge
whether a fresh brain learns, within one life, which of two food kinds to
approach and which to avoid.

- **World condition (opt-in).** `WorldConfig.food_kinds_enabled` (serde default
  `false`). When on, every food item has a kind fixed by its index parity
  (`First` = even index, `Second` = odd index). `Second` items render a distinct
  colour. Each kind has its own odour. Which kind is nourishing is drawn by a
  deterministic hash of the world seed and a draw index. The kernel redraws it
  when it is built and on every agent reset, which covers every generation and
  every seeded free-life run. Nothing the agent senses says which kind is
  which.
- **Harm acts only through the body.** A harmful meal gives no energy. It
  removes integrity equal in raw-gradient size to the energy a nourishing meal
  gives: `food_energy / max_energy × ENERGY_WEIGHT / INTEGRITY_WEIGHT ×
  max_integrity`. The loss is clamped, with death marked in the same scope, and
  counted in a new physics slot `P_HARMFUL_MEAL_COUNT`. `P_FOOD_COUNT` counts
  only nourishing meals. There is no reward term, label or sensory flag.
- **New sense.** `BrainConfig.second_odour_sense_enabled` (serde default
  `false`, not heritable) declares a second nostril pair, `second_odour`, as a
  sense descriptor of the 0003 sense table (`crates/xagent-brain/src/senses.rs`).
  Its descriptor: 2 channels, input `SenseInput::NonVisualFeatures { offset:
  SCENT_FEATURE_OFFSET + SCENT_CHANNELS }` (27), the smell adaptation rate and
  `ClosedForm2` whitening with the smell initial variance, `LearningRate::Actor`,
  and exactly one route, `Turn` with `Heritability::LearnedOnly`. Its block
  weights are learned-only and zero at birth. When food kinds are on, the first
  pair smells only `First` items and the second pair smells only `Second` items,
  under the same odour law.
- **Measurement probes that land first.** These are pure, unit-tested code:
  - per-kind free-life metrics, link accumulators and the per-run summary
    types (`crates/xagent-sandbox/src/food_kind_metrics.rs`);
  - the gate verdict as its own binary, `food_kind_verdict`
    (`crates/xagent-sandbox/src/bin/food_kind_verdict.rs`). It takes a
    directory of the 8 per-run summaries or a bundle file, plus
    `--write-bundle <file>` and `--expect-verdict <text>`. It needs no GPU;
  - a supervised kind readout, used for measurement only: held-out
    balanced accuracy of an L2 logistic regression, k-fold, with a
    shuffled-label control (`crates/xagent-brain/src/kind_readout.rs`).
- **Harness integration.** This extends the free-life harness of
  0001-free-life-harness-and-trunk-lesions with `--food-kinds`,
  `--freeze-policy` (the frozen-policy control) and
  `--food-kind-summary <file>`.
- **Decision record.** The record is `DECISION.md` in this plan directory, plus
  the validated bundle of the eight per-run summaries in `results.json`.

## Owner decisions on the 0003 interface

These are fixed here, not left to workers.

1. **Route set: Turn only, learned-only.** 0003's layout rule
   (`derived-sense-layout`) lets a sense appended after the base senses declare
   only `Turn`, and panics otherwise. Smell's route slice may carry `Forward`
   and `Critic` when plan 0002 adopted them. So `SECOND_ODOUR` declares its own
   one-entry slice `[RouteDescriptor { route: MotorRoute::Turn, heritability:
   Heritability::LearnedOnly }]` and never reuses smell's routes. This is the
   route the gate measures.
2. **Input kind: no new kind.** 0003's only non-visual input kind is
   `NonVisualFeatures { offset }`, which reads the brain's non-visual feature
   tail. Adding a kind that reads the sensory buffer directly would need new
   0003 block WGSL and generator code. This plan does not do that. The second
   odour enters as two extra non-visual feature channels, using the same
   pattern as the existing flag-gated danger percept:
   - one generic WGSL override `EXTRA_SENSORY_CHANNELS: u32` (default `0u`) in
     `common.wgsl`. It grows `SENSORY_STRIDE` and `NON_VISUAL_FEATURE_COUNT` by
     its value;
   - the extra sensory channels sit right after the two scent channels in the
     sensory buffer. Both feature-extraction paths copy them right after the
     scent features (before the danger percept);
   - `BrainLayout` grows `sensory_stride` and `feature_count` by the same
     amount, which is 2 when `second_odour_sense_enabled` is set and 0
     otherwise.

   As a result, the two channels are also encoder inputs, exactly as smell's
   channels are. Their encoder columns follow the encoder's existing
   initialisation rule. The weights that are learned-only and zero at birth
   are the sense block's turn weights. With the flag off,
   `EXTRA_SENSORY_CHANNELS` is 0 and every stride, offset and trajectory is
   unchanged.
3. **Config-selected sense table.** `senses.rs` gains
   `SENSES_WITH_SECOND_ODOUR: &[SenseDescriptor] = &[SMELL, VISION,
   SECOND_ODOUR]` and `pub fn senses_for_config(config: &BrainConfig) ->
   &'static [SenseDescriptor]`. That function returns `DECLARED_SENSES` when
   the flag is off. There are two consumers:
   - `BrainLayout::from_config` (in `buffers.rs`, owned by extra-sensory-layout)
     calls `with_senses(config, senses_for_config(config))`;
   - `GpuKernel::new` (in `gpu_kernel.rs`, owned by second-odour-kernel)
     delegates to `new_with_senses(..., senses_for_config(brain_config))`, and
     `vision_override_constants` emits `EXTRA_SENSORY_CHANNELS` from the
     layout.

   Every caller, including the harness, fresh brains and mutation, therefore
   gets the matching table from the flag alone.
4. **Physics parity.** Appending `P_HARMFUL_MEAL_COUNT` makes `PHYS_STRIDE` 54.
   The 0003 parity test hashes the full physics state. So food-kind-layout
   changes that test to hash, per agent, only the physics slots `[0,
   P_HARMFUL_MEAL_COUNT)`, which is the byte stream it hashed before. The test
   also asserts that the new slot is `0.0` for every agent. The committed
   golden is not re-recorded.

## Gate (fixed before any run; encoded in the `food_kind_verdict` binary)

The runs use seeds 5, 6, 7 and 8. Each run has 10 agents and lasts 300,000
ticks, starting from fresh birth brains with food kinds on and the second odour
sense on. There are two arms per seed: learning, and frozen policy
(`freeze_steering_weights = true`). Both arms of a seed share the same fresh
brains and the same nourishing kind. "Late" means the last 30% of ticks. Turn
metrics are size-weighted: the mean executed turn in the direction toward (or
away from) the nearest food of that kind within `SCENT_RANGE`. The toward
direction uses the kernel's approach-counter convention.

The behaviour gate passes only if all three conditions hold:

1. **B1:** in the learning arm, the late turn toward the nourishing kind is
   above 0 when pooled over agents, and above the frozen arm's on at least 3
   of 4 seeds.
2. **B2:** in the learning arm, the late turn away from the harmful kind is
   above 0 when pooled, and above the frozen arm's on at least 3 of 4 seeds.
3. **B3:** in the learning arm, the nourishing share of meals is higher in the
   last quarter than in the first on at least 3 of 4 seeds. The learning arm's
   last-quarter share is also above the frozen arm's on at least 3 of 4 seeds.

Links are checked in this order. The first one that fails is named:

- **readout:** on every seed, the learning arm's mean held-out balanced
  accuracy of the nearest in-range food's kind, read from the whitened sense
  inputs, is at least 0.70. Its shuffled-label control is at most 0.55. An
  absent score fails.
- **credit:** for each odour pair separately, the signed conditional turn
  credit along that pair's whitened right − left input is positive in at least
  24 of the 40 learning agents. The sign is + for the nourishing kind's pair
  and − for the harmful kind's pair. The credit is TD error × fresh turn-noise
  innovation × that input.
- **weight change:** for each odour pair separately, the signed right − left
  turn-weight difference at the end of life is positive in at least 24 of the
  40 learning agents, using the same signs.

The verdict is one of the following:

- `PASS` when B1–B3 and all three links pass.
- `FAIL readout`, `FAIL credit` or `FAIL weight-change` for the first failing
  link.
- `FAIL weight-change` when every link passes but behaviour fails. In that case
  the weights moved the right way but too little to change behaviour.

The verdict fails closed: a missing, duplicate, malformed or mismatched summary
is an error, not a verdict.

## GPU proofs fail closed

Every check that runs a GPU-backed test is prefixed `env XAGENT_REQUIRE_GPU=1`.
Every new GPU test starts with a helper that returns early only if no adapter
exists and `XAGENT_REQUIRE_GPU` is unset. When the variable is set, the helper
panics instead. This follows `gpu_adapter_present_when_required` in
`crates/xagent-sandbox/tests/integration.rs`. Under these checks, a host
without an adapter therefore fails rather than passing on skipped tests. The
`gpu-adapter-present` preflight and witness show that an adapter exists on
the route.

## Prerequisite interfaces relied on

- From 0001-free-life-harness-and-trunk-lesions:
  - the free-life runner `run_free_life` with `FreeLifeOptions` and
    `BrainSource` in `crates/xagent-sandbox/src/headless.rs`, the measurement
    types in `crates/xagent-sandbox/src/free_life/` (`metrics.rs`,
    `tick_log.rs`), and the `--free-life` CLI flags in
    `crates/xagent-sandbox/src/main.rs`;
  - the seeded fresh-brain run and the per-agent metrics;
  - the optional per-brain-tick log: reward, value, executed exploration noise,
    whitened pathway inputs, and ground-truth bearings.
- From 0003-uniform-sense-motor-route:
  - `crates/xagent-brain/src/senses.rs`: `SenseDescriptor`, `SenseInput`,
    `Whitening`, `LearningRate`, `MotorRoute`, `Heritability`,
    `RouteDescriptor`, `SMELL`, `VISION`, `DECLARED_SENSES`, `SCENT_FEATURE_OFFSET`
    and `SCENT_WHITENING_RATE`, `MAX_SENSE_SLOTS`;
  - `BrainLayout::with_senses(config, senses)` and `sense_blocks()`
    (`SenseBlockOffsets` with `whitened_input` and `turn_weights`), including
    the rule that an appended sense declares only `Turn`;
  - `GpuKernel::new_with_senses` and the generated `SENSE{k}_*` overrides, so a
    third declared sense needs no block WGSL edit;
  - the parity test `crates/xagent-sandbox/tests/sense_block_parity.rs` and its
    golden;
  - the rule that a newly declared sense's weights are learned-only and zero at
    birth.

## Non-goals

- No change to default behaviour, fitness, the evolution reference or the
  heritability of smell and visual weights. The scenario is never turned on by
  any default config or preset.
- No new `SenseInput` kind, no edit to the 0003 block code, and no route other
  than `Turn` for the second odour.
- No CPU mirror of kind colour or kind odour in
  `crates/xagent-sandbox/src/agent/senses.rs`, and no GUI food-mesh
  recolouring. The GPU pass is the authority, and the scenario runs headless.
- No new learning rule, reward shaping or hand-built pathway for the second
  odour.
- No stored-champion runs in the scenario: their layout lacks the new sense, so
  the harness refuses them.
- No port to evolution defaults, whatever the result.
