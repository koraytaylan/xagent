---
objective: >-
  Commit a reusable, seeded free-life measurement harness, then lesion the
  128-d trunk and record what it is for. A headless free-life mode in
  xagent-sandbox runs N agents (default 10) for a tick count on a seed, from a
  fresh birth brain or from a stored champion's birth brain loaded read-only
  from a governor database; it reports per-agent metrics (meals, deaths and
  deaths on hazard ground, sightings and the share ending in a meal,
  size-weighted turn toward food and away from hazard, correlation of each
  pathway's output with the food side, turn and forward credit per whitened
  pathway input, the critic's in-view minus not-in-view and centred minus
  off-axis value gaps) and optionally writes a per-brain-tick log (encoding,
  the ten whitened pathway inputs, reward, value, executed exploration noise,
  ground-truth food and hazard bearings). Four measurement-only, default-off,
  non-heritable lesion switches (128-d turn weights, 128-d forward weights,
  memory motor blend, critic held at its bias) are run singly and together on
  fresh brains and the seed 5-8 reference champions (300,000 ticks, seeds 5-8,
  10 agents), ending in a decision record of which trunk parts the behaviour
  and the within-life learning need. Default behaviour stays bit-identical,
  checked by a seeded headless run against a digest captured at the baseline.
depends_on: []
status: completed
toolchains:
  - cargo
  - rustc
  - rustfmt
  - cargo-fmt
  - cargo-clippy
  - clippy-driver
checks:
  - name: workspace-fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: workspace-clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: brain-crate-tests
    command: "cargo test -p xagent-brain"
    timeout_ms: 5400000
  - name: sandbox-tests
    command: "cargo test -p xagent-sandbox"
    timeout_ms: 3600000
  - name: seeded-digest-strict
    command: "env XAGENT_REQUIRE_GPU=1 XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest"
    timeout_ms: 3600000
preflight_checks:
  - name: pf-brain-config-size
    command: "grep -qF 'pub const CONFIG_SIZE: usize = 16;' crates/xagent-brain/src/buffers.rs"
  - name: pf-brain-config-last-slot
    command: "grep -qF 'pub const CFG_FREEZE_STEERING_WEIGHTS: usize = 15;' crates/xagent-brain/src/buffers.rs"
  - name: pf-wgsl-brain-config-array
    command: "grep -qF 'brain_config:      array<vec4<f32>, 4>;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: pf-decision-stride
    command: "grep -qF 'pub const DECISION_STRIDE: usize = DECISION_MOTOR + 4;' crates/xagent-brain/src/buffers.rs"
  - name: pf-wgsl-decision-stride
    command: "grep -qF 'const DECISION_STRIDE: u32 = DECISION_MOTOR + 4u;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: pf-turn-policy-site
    command: "grep -qF 'var turn: f32 = s_turn_dot + scent_dot + vision_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-forward-policy-site
    command: "grep -qF 'var forward: f32 = brain_state[brain_base + O_ACT_BIASES] + s_forward_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-memory-blend-site
    command: "grep -qF 'let mix = strength * MEMORY_BLEND_STRENGTH;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-critic-value-site
    command: "grep -qF 'var value: f32 = brain_state[brain_base + O_VALUE_BIAS] + s_value;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-replay-value-site
    command: "grep -qF 'let replay_value = brain_state[brain_base + O_VALUE_BIAS] + s_dense_partials[0];' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-executed-innovation-site
    command: "grep -qF 's_explore[2u] = TURN_NOISE_INNOVATION * turn_draw * exploration_rate' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: pf-scent-whitened-slot
    command: "grep -qF 'override O_SCENT_WHITENED: u32 = O_SCENT_COVARIANCE + 3u;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: pf-vision-input-slot
    command: "grep -qF 'override O_VISION_PATHWAY_INPUT: u32 =' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: pf-freeze-setter
    command: "grep -qF 'pub fn set_freeze_steering_weights(&mut self, freeze: bool) {' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-batch-size-api
    command: "grep -qF 'pub fn kernel_batch_size(&self) -> u32 {' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-full-state-read
    command: "grep -qF 'pub fn read_full_state_blocking(&mut self) -> &[f32] {' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-gpu-availability-api
    command: "grep -qF 'pub fn is_available' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-food-bearing-slot
    command: "grep -qF 'pub const P_NEAREST_FOOD_BEARING: usize = 34;' crates/xagent-brain/src/buffers.rs"
  - name: pf-hazard-bearing-slot
    command: "grep -qF 'pub const P_NEAREST_DANGER_BEARING: usize = 40;' crates/xagent-brain/src/buffers.rs"
  - name: pf-hazard-ground-slot
    command: "grep -qF 'pub const P_IN_DANGER_BIOME: usize = 35;' crates/xagent-brain/src/buffers.rs"
  - name: pf-champion-blob-decoder
    command: "grep -qF 'fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {' crates/xagent-sandbox/src/governor.rs"
  - name: pf-node-brain-table
    command: "grep -qF 'CREATE TABLE IF NOT EXISTS node_brain (' crates/xagent-sandbox/src/governor.rs"
  - name: pf-rusqlite-dependency
    command: "grep -qF 'rusqlite = { version = \"0.34\", features = [\"bundled\"] }' crates/xagent-sandbox/Cargo.toml"
  - name: pf-headless-entry
    command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs"
  - name: pf-fresh-birth-brain
    command: "grep -qF 'pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {' crates/xagent-sandbox/src/agent/mod.rs"
  - name: pf-sandbox-lib-modules
    command: "grep -qF 'pub mod headless;' crates/xagent-sandbox/src/lib.rs"
  - name: pf-cli-seed-flag
    command: "grep -qF 'seed: Option<u64>,' crates/xagent-sandbox/src/main.rs"
  - name: pf-xagent-bin-target
    command: "grep -A2 -F '[[bin]]' crates/xagent-sandbox/Cargo.toml | grep -qxF 'name = \"xagent\"' && grep -A3 -F '[dev-dependencies]' crates/xagent-sandbox/Cargo.toml | grep -qF 'tempfile'"
  - name: pf-guard-slug-pattern
    command: "grep -qF 'const PREFIX: &[u8] = b\"000\";' crates/xagent-sandbox/tests/contributing_guard.rs"
  - name: pf-banner-on-stderr
    command: "grep -qF 'eprintln!(\"xagent v0.1.0' crates/xagent-sandbox/src/main.rs"
  - name: pf-db-files-ignored
    command: "grep -qxF '*.db' .gitignore"
  - name: pf-task-seeded-headless-digest
    command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs && grep -qF 'CREATE TABLE IF NOT EXISTS node_brain (' crates/xagent-sandbox/src/governor.rs && grep -qF 'pub fn is_available' crates/xagent-brain/src/gpu_kernel.rs && grep -qxF '*.db' .gitignore"
  - name: pf-task-kernel-measurement-layout
    command: "grep -qF 'pub const CONFIG_SIZE: usize = 16;' crates/xagent-brain/src/buffers.rs && grep -qF 'pub const CFG_FREEZE_STEERING_WEIGHTS: usize = 15;' crates/xagent-brain/src/buffers.rs && grep -qF 'brain_config:      array<vec4<f32>, 4>;' crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF 'pub const DECISION_STRIDE: usize = DECISION_MOTOR + 4;' crates/xagent-brain/src/buffers.rs && grep -qF 'const DECISION_STRIDE: u32 = DECISION_MOTOR + 4u;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: pf-task-kernel-lesions-and-probe
    command: "grep -qF 'var turn: f32 = s_turn_dot + scent_dot + vision_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 'var forward: f32 = brain_state[brain_base + O_ACT_BIASES] + s_forward_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 'let mix = strength * MEMORY_BLEND_STRENGTH;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 'var value: f32 = brain_state[brain_base + O_VALUE_BIAS] + s_value;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 'let replay_value = brain_state[brain_base + O_VALUE_BIAS] + s_dense_partials[0];' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 's_explore[2u] = TURN_NOISE_INNOVATION * turn_draw * exploration_rate' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -qF 'override O_SCENT_WHITENED: u32 = O_SCENT_COVARIANCE + 3u;' crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF 'override O_VISION_PATHWAY_INPUT: u32 =' crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -qF 'pub fn set_freeze_steering_weights(&mut self, freeze: bool) {' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-task-read-only-champion-loader
    command: "grep -qF 'fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {' crates/xagent-sandbox/src/governor.rs && grep -qF 'CREATE TABLE IF NOT EXISTS node_brain (' crates/xagent-sandbox/src/governor.rs && grep -qF 'rusqlite = { version = \"0.34\", features = [\"bundled\"] }' crates/xagent-sandbox/Cargo.toml"
  - name: pf-task-free-life-metrics-and-log
    command: "grep -qF 'pub const P_NEAREST_FOOD_BEARING: usize = 34;' crates/xagent-brain/src/buffers.rs && grep -qF 'pub const P_NEAREST_DANGER_BEARING: usize = 40;' crates/xagent-brain/src/buffers.rs && grep -qF 'pub const P_IN_DANGER_BIOME: usize = 35;' crates/xagent-brain/src/buffers.rs && grep -qF 'pub mod headless;' crates/xagent-sandbox/src/lib.rs"
  - name: pf-task-free-life-runner
    command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs && grep -qF 'pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {' crates/xagent-sandbox/src/agent/mod.rs && grep -qF 'pub fn kernel_batch_size(&self) -> u32 {' crates/xagent-brain/src/gpu_kernel.rs && grep -qF 'pub fn read_full_state_blocking(&mut self) -> &[f32] {' crates/xagent-brain/src/gpu_kernel.rs && grep -qF 'pub fn is_available' crates/xagent-brain/src/gpu_kernel.rs"
  - name: pf-task-trunk-lesion-matrix
    command: "grep -qF 'const PREFIX: &[u8] = b\"000\";' crates/xagent-sandbox/tests/contributing_guard.rs && grep -qF 'seed: Option<u64>,' crates/xagent-sandbox/src/main.rs && grep -qxF '*.db' .gitignore"
execution_checks:
  - name: developer-route
    tasks:
      - kernel-measurement-layout
      - kernel-lesions-and-probe
      - read-only-champion-loader
      - free-life-metrics-and-log
      - free-life-runner
      - free-life-cli
      - free-life-cli-exit-test
    role: developer
    operations: [edit]
    command: "sh .makina/lifecycle/probe"
    timeout_ms: 5400000
    fixtures:
      - source: docs/plans/0001-free-life-harness-and-trunk-lesions/execution/developer-probe.sh
        target: .makina/lifecycle/probe
    constraints:
      - task: kernel-measurement-layout
        name: uniform-slots-and-decision-stride
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: pf-brain-config-size
            command: "grep -qF 'pub const CONFIG_SIZE: usize = 16;' crates/xagent-brain/src/buffers.rs"
          - name: pf-brain-config-last-slot
            command: "grep -qF 'pub const CFG_FREEZE_STEERING_WEIGHTS: usize = 15;' crates/xagent-brain/src/buffers.rs"
          - name: pf-wgsl-brain-config-array
            command: "grep -qF 'brain_config:      array<vec4<f32>, 4>;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
          - name: pf-decision-stride
            command: "grep -qF 'pub const DECISION_STRIDE: usize = DECISION_MOTOR + 4;' crates/xagent-brain/src/buffers.rs"
          - name: pf-wgsl-decision-stride
            command: "grep -qF 'const DECISION_STRIDE: u32 = DECISION_MOTOR + 4u;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
      - task: kernel-lesions-and-probe
        name: policy-sites-and-setter-pattern
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
        checks:
          - name: pf-turn-policy-site
            command: "grep -qF 'var turn: f32 = s_turn_dot + scent_dot + vision_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-forward-policy-site
            command: "grep -qF 'var forward: f32 = brain_state[brain_base + O_ACT_BIASES] + s_forward_dot;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-memory-blend-site
            command: "grep -qF 'let mix = strength * MEMORY_BLEND_STRENGTH;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-critic-value-site
            command: "grep -qF 'var value: f32 = brain_state[brain_base + O_VALUE_BIAS] + s_value;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-replay-value-site
            command: "grep -qF 'let replay_value = brain_state[brain_base + O_VALUE_BIAS] + s_dense_partials[0];' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-executed-innovation-site
            command: "grep -qF 's_explore[2u] = TURN_NOISE_INNOVATION * turn_draw * exploration_rate' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: pf-scent-whitened-slot
            command: "grep -qF 'override O_SCENT_WHITENED: u32 = O_SCENT_COVARIANCE + 3u;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
          - name: pf-vision-input-slot
            command: "grep -qF 'override O_VISION_PATHWAY_INPUT: u32 =' crates/xagent-brain/src/shaders/kernel/common.wgsl"
          - name: pf-freeze-setter
            command: "grep -qF 'pub fn set_freeze_steering_weights(&mut self, freeze: bool) {' crates/xagent-brain/src/gpu_kernel.rs"
      - task: read-only-champion-loader
        name: champion-blob-and-table
        sources:
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/Cargo.toml
        checks:
          - name: pf-champion-blob-decoder
            command: "grep -qF 'fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState> {' crates/xagent-sandbox/src/governor.rs"
          - name: pf-node-brain-table
            command: "grep -qF 'CREATE TABLE IF NOT EXISTS node_brain (' crates/xagent-sandbox/src/governor.rs"
          - name: pf-rusqlite-dependency
            command: "grep -qF 'rusqlite = { version = \"0.34\", features = [\"bundled\"] }' crates/xagent-sandbox/Cargo.toml"
      - task: free-life-metrics-and-log
        name: ground-truth-slots-and-module-root
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-sandbox/src/lib.rs
        checks:
          - name: pf-food-bearing-slot
            command: "grep -qF 'pub const P_NEAREST_FOOD_BEARING: usize = 34;' crates/xagent-brain/src/buffers.rs"
          - name: pf-hazard-bearing-slot
            command: "grep -qF 'pub const P_NEAREST_DANGER_BEARING: usize = 40;' crates/xagent-brain/src/buffers.rs"
          - name: pf-hazard-ground-slot
            command: "grep -qF 'pub const P_IN_DANGER_BIOME: usize = 35;' crates/xagent-brain/src/buffers.rs"
          - name: pf-sandbox-lib-modules
            command: "grep -qF 'pub mod headless;' crates/xagent-sandbox/src/lib.rs"
      - task: free-life-runner
        name: kernel-run-and-birth-brain-apis
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/agent/mod.rs
          - crates/xagent-brain/src/gpu_kernel.rs
        checks:
          - name: pf-headless-entry
            command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs"
          - name: pf-fresh-birth-brain
            command: "grep -qF 'pub fn fresh_brain_state(config: &BrainConfig, rng: &mut impl Rng) -> AgentBrainState {' crates/xagent-sandbox/src/agent/mod.rs"
          - name: pf-batch-size-api
            command: "grep -qF 'pub fn kernel_batch_size(&self) -> u32 {' crates/xagent-brain/src/gpu_kernel.rs"
          - name: pf-full-state-read
            command: "grep -qF 'pub fn read_full_state_blocking(&mut self) -> &[f32] {' crates/xagent-brain/src/gpu_kernel.rs"
          - name: pf-gpu-availability-api
            command: "grep -qF 'pub fn is_available' crates/xagent-brain/src/gpu_kernel.rs"
      - task: free-life-cli
        name: cli-parser
        sources:
          - crates/xagent-sandbox/src/main.rs
        checks:
          - name: pf-cli-seed-flag
            command: "grep -qF 'seed: Option<u64>,' crates/xagent-sandbox/src/main.rs"
      - task: free-life-cli-exit-test
        name: bin-target-and-stderr-banner
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/src/main.rs
        checks:
          - name: pf-xagent-bin-target
            command: "grep -A2 -F '[[bin]]' crates/xagent-sandbox/Cargo.toml | grep -qxF 'name = \"xagent\"' && grep -A3 -F '[dev-dependencies]' crates/xagent-sandbox/Cargo.toml | grep -qF 'tempfile'"
          - name: pf-banner-on-stderr
            command: "grep -qF 'eprintln!(\"xagent v0.1.0' crates/xagent-sandbox/src/main.rs"
  - name: architect-route
    tasks:
      - seeded-headless-digest
      - trunk-lesion-matrix
    role: architect
    operations: [edit, shell]
    command: "sh .makina/lifecycle/probe"
    timeout_ms: 5400000
    fixtures:
      - source: docs/plans/0001-free-life-harness-and-trunk-lesions/execution/architect-probe.sh
        target: .makina/lifecycle/probe
    actions:
      - name: release-bench-run
        command: "cargo run --release -p xagent-sandbox -- --bench --bench-ticks 10000 --bench-agents 10"
        timeout_ms: 3600000
    constraints:
      - task: seeded-headless-digest
        name: headless-entry-and-ignored-databases
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-brain/src/gpu_kernel.rs
          - .gitignore
        checks:
          - name: pf-headless-entry
            command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs"
          - name: pf-node-brain-table
            command: "grep -qF 'CREATE TABLE IF NOT EXISTS node_brain (' crates/xagent-sandbox/src/governor.rs"
          - name: pf-gpu-availability-api
            command: "grep -qF 'pub fn is_available' crates/xagent-brain/src/gpu_kernel.rs"
          - name: pf-db-files-ignored
            command: "grep -qxF '*.db' .gitignore"
      - task: trunk-lesion-matrix
        name: guard-safe-record-paths-and-cli
        sources:
          - crates/xagent-sandbox/tests/contributing_guard.rs
          - crates/xagent-sandbox/src/main.rs
          - .gitignore
        checks:
          - name: pf-guard-slug-pattern
            command: "grep -qF 'const PREFIX: &[u8] = b\"000\";' crates/xagent-sandbox/tests/contributing_guard.rs"
          - name: pf-cli-seed-flag
            command: "grep -qF 'seed: Option<u64>,' crates/xagent-sandbox/src/main.rs"
          - name: pf-db-files-ignored
            command: "grep -qxF '*.db' .gitignore"
  - name: integration-route
    tasks:
      - $integration
    role: architect
    operations: [edit]
    command: "sh .makina/lifecycle/probe"
    timeout_ms: 5400000
    fixtures:
      - source: docs/plans/0001-free-life-harness-and-trunk-lesions/execution/integration-probe.sh
        target: .makina/lifecycle/probe
    constraints:
      - task: $integration
        name: mandated-group-commands-and-guard
        sources:
          - CONTRIBUTING.md
          - Cargo.toml
          - crates/xagent-sandbox/tests/contributing_guard.rs
          - crates/xagent-sandbox/src/headless.rs
        checks:
          - name: pf-guard-slug-pattern
            command: "grep -qF 'const PREFIX: &[u8] = b\"000\";' crates/xagent-sandbox/tests/contributing_guard.rs"
          - name: pf-headless-entry
            command: "grep -qF 'pub fn run_headless(config: FullConfig, db_path: &str, resume: bool, _has_gpu: bool) {' crates/xagent-sandbox/src/headless.rs"
          - name: pf-mandated-tooling
            command: "grep -qF 'cargo clippy --workspace --all-targets -- -D warnings' CONTRIBUTING.md"
---

# Free-life harness and trunk lesions

## Why

`crates/xagent-sandbox/src/headless.rs` only runs the evolution loop. Every
free-life number in `docs/reviews/2026-09-25-learner-foraging-investigation.md`
came from scratch code that was never committed, so no later change can be
measured against it. This plan commits the harness, then uses it to find out
what the 128-d trunk (the encoding's turn weights, its forward weights, the
memory motor blend and the critic) contributes now that the whitened smell and
visual pathways carry steering.

## Scope

Task ids below are the packet slugs in `tasks/`; ARCHITECTURE.md is the
binding specification for each.

1. **Baseline digest (`seeded-headless-digest`, architect).** Before any
   kernel change, a seeded headless evolution run (seed 5, 2 generations, tick
   budget 2,000, default config otherwise) is reduced to a digest over every
   node row, agent result and stored champion blob, captured at baseline
   revision `0976448` into a committed fixture. A test reproduces the run and,
   under `XAGENT_REQUIRE_SEEDED_DIGEST=1`, requires the fixture's digest;
   without it the test only checks that two identical runs match (so CI on
   another adapter stays green).
2. **Kernel measurement layout (`kernel-measurement-layout`).** The new
   uniform slots, `TrunkLesions`, packers, the probe block layout,
   `BrainTickProbe` and its decoder, in Rust and WGSL, with no behaviour
   change.
3. **Lesion branches and probe (`kernel-lesions-and-probe`).** The four
   lesion branches and the probe writes in the fused brain pass, and the
   `GpuKernel` setters and readback, default off, never stored in
   `BrainConfig`, `AgentBrainState` or the governor database.
4. **Read-only champion loader (`read-only-champion-loader`).** Opens a
   governor database with SQLite's read-only flag and returns a stored
   champion's birth brain and node config without any schema or write
   statement.
5. **Free-life measurement types (`free-life-metrics-and-log`).** The
   per-brain-tick sample, the per-agent metrics over the whole life and its
   last 30 %, and the binary tick log with its reader.
6. **Free-life runner (`free-life-runner`).** `run_free_life` in
   `headless.rs`: builds the world and kernel, writes the chosen birth brain to
   every agent, applies the lesions, steps one brain tick per sample and
   produces a serialisable report.
7. **CLI (`free-life-cli`).** `--free-life` and its flags in `main.rs`, with
   bin unit tests.
8. **Binary exit test (`free-life-cli-exit-test`).** An integration test runs
   the built binary with `--free-life --free-life-agents 0` and asserts exit
   status 1 and the agent-range message on stderr.
9. **Lesion matrix and decision record (`trunk-lesion-matrix`, architect).**
   The seed 5-8 reference lineages, 48 free lives joined into one committed
   `lesion-matrix.json`, and the decision record with verdicts recomputed by a
   test.

## Non-goals

- No change to learning rules, rewards, the world, the governor's selection,
  fitness, heritability, or any default value. The lesions and the probe are
  measurement instruments only.
- No route of the whitened pathway inputs to the critic or the forward channel
  (that is plan 0002), no generic sense block (plan 0003), no food kinds (plan
  0004).
- No change to `run_headless`, the GUI, `--bench*`, or the `--validate-*`
  harnesses beyond what compiling against the new kernel constants requires.
- No migration of stored databases; the loader never writes.
- The reference champion databases are regenerated at the current code with
  the documented 80-generation baseline command; databases that may exist
  outside the checkout are neither read nor required.

## Interfaces provided to later plans

- `xagent_sandbox::headless::run_free_life(config: &FullConfig, options:
  &FreeLifeOptions) -> Result<FreeLifeReport, FreeLifeError>` with
  `FreeLifeOptions { agent_count, ticks, seed, brain_source, lesions,
  log_path }` and `BrainSource::{Fresh, Champion { db_path, node_id }}`.
- CLI: `xagent --free-life [--free-life-agents N] [--free-life-ticks N]
  [--seed S] [--free-life-champion-db PATH [--free-life-champion-node ID]]
  [--free-life-lesion LIST] [--free-life-log PATH] [--free-life-out PATH]`,
  with `LIST` a comma list of `turn-weights`, `forward-weights`,
  `memory-blend`, `critic`, or `all`/`none`.
- `xagent_sandbox::free_life::metrics::{FreeLifeSample, AgentMetrics,
  WindowMetrics}` and `xagent_sandbox::free_life::tick_log::{TickLogWriter,
  read_tick_log}`; the log is little-endian `f32` rows with a JSON header
  naming every column.
- `xagent_brain::buffers::{TrunkLesions, BrainTickProbe}`,
  `GpuKernel::{set_trunk_lesions, set_brain_tick_probe,
  read_brain_tick_probes_blocking}`.
- `xagent_sandbox::governor::load_champion_read_only(db_path, node_id) ->
  Result<StoredChampion, ChampionLoadError>`.
- `docs/measurements/trunk-lesions/DECISION.md` and the 48 run reports as one
  JSON array in `docs/measurements/trunk-lesions/lesion-matrix.json`.

## Decisions

- Free-life mode requires `vision_stride == 1` (the default) and dispatches
  exactly `kernel_batch_size()` ticks per sample, so every sample is exactly
  one brain tick; any other stride is rejected with an error.
- All agents of a free life share one birth brain: one `fresh_brain_state`
  draw from the run's seeded stream, or the champion's stored birth brain
  unmutated with its node's `BrainConfig`. The world comes from the CLI config
  and `--seed`.
- The default champion is the `node_brain` row whose node has the greatest
  `(generation, id)`; `--free-life-champion-node` selects another.
- The lesion and probe state lives in new brain-config uniform slots (the
  uniform grows from 4 to 6 `vec4`s); the probe block is appended to the
  decision buffer after the motor slots and is written only while the probe
  flag is set.
- The decision record and the joined run reports live under
  `docs/measurements/` so the record test does not name a plan in source
  (`contributing_guard.rs`); per-run reports and lineage databases stay in
  ignored `target/trunk-lesions/`.
