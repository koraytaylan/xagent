---
objective: >-
  Replace the two hand-built pathways (smell, vision) with one generic per-sense
  block (adapt, whiten, plastic weights to each motor route whose gate passed,
  plus the critic if its gate passed), so a newly declared sense needs no new
  code path; provide a declarative sense descriptor, brain-state offsets derived
  from BrainLayout, unchanged loading and replay of stored champion databases,
  and a parity test plus a byte-identical seeded headless run proving smell and
  vision behave exactly as before. Also run two separately gated investigations:
  learned pooling against the left/right hemifields (ported only if it reaches
  the hemifield score), and removal of the klinotaxis turn-gain and fatigue
  modulators, each behind its own flag, landed or reverted with the result
  recorded.
depends_on:
  - 0002-pathway-critic-and-forward-drive
status: todo
toolchains:
  - cargo
  - rustc
  - rustup
  - rustfmt
  - cargo-fmt
  - cargo-clippy
  - clippy-driver
checks:
  - name: sandbox-suite
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox
    timeout_ms: 5400000
  - name: shared-suite
    command: cargo test -p xagent-shared
    timeout_ms: 1800000
  - name: brain-lib-suite
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib
    timeout_ms: 5400000
  - name: gpu-parity-suites
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity --test modulator_flags -- --include-ignored
    timeout_ms: 5400000
preflight_checks:
  - name: pathway-source-anchors
    command: >-
      sh -c 'grep -q "fn vision_pathway_step(" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "whiten_scent(" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "O_TRACE_VISION" crates/xagent-brain/src/shaders/kernel/common.wgsl
      && grep -q "override O_SCENT_MEAN: u32" crates/xagent-brain/src/shaders/kernel/common.wgsl
      && grep -q "pub const O_SCENT_MEAN: usize" crates/xagent-brain/src/buffers.rs
      && grep -q "pub const O_SENSORY_MEAN: usize" crates/xagent-brain/src/buffers.rs
      && grep -q "fn vision_override_constants(layout: &BrainLayout)" crates/xagent-brain/src/gpu_kernel.rs'
  - name: heritability-source-anchors
    command: >-
      sh -c 'grep -q "offsets.scent_turn_weights" crates/xagent-sandbox/src/agent/mod.rs
      && grep -q "offsets.vision_turn_weights" crates/xagent-sandbox/src/agent/mod.rs
      && grep -q "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs
      && grep -q "pub fn champion_brain(&self, node_id: i64)" crates/xagent-sandbox/src/governor.rs
      && grep -q "pub fn run_headless(" crates/xagent-sandbox/src/headless.rs'
  - name: modulator-source-anchors
    command: >-
      sh -c 'grep -q "KLINOTAXIS_SENSITIVITY" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "turn \*= klinotaxis_factor;" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "brain_state\[brain_base + O_FATIGUE_FACTOR\] = fatigue_factor;" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "pub const CONFIG_SIZE: usize" crates/xagent-brain/src/buffers.rs
      && grep -q "pub danger_percept_enabled: bool," crates/xagent-shared/src/config.rs'
  - name: pooling-source-anchors
    command: >-
      sh -c 'grep -q "pub fn read_brain_scratch_blocking" crates/xagent-brain/src/gpu_kernel.rs
      && grep -q "fn vision_hemifields() -> array<f32, 8>" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
      && grep -q "columns, whitened, learned | +0.16" docs/reviews/2026-09-25-learner-foraging-investigation.md
      && grep -q "hemifields, whitened, learned\*\* | \*\*+0.33" docs/reviews/2026-09-25-learner-foraging-investigation.md'
  - name: gpu-adapter-present
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    timeout_ms: 3600000
  - name: brain-layout-baseline
    command: cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent buffers::tests::config_indices_within_bounds
    timeout_ms: 3600000
  - name: whitening-validations-baseline
    command: >-
      env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact
      gpu_kernel::visual_pathway_validation::visual_pathway_composition_preserves_other_passes
      gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_complete_state
      gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_cortex_skip
      gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state
      gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state
      gpu_kernel::cooperative_jacobi_validation::jacobi_composition_keeps_other_helpers_and_scratch
      gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_complete_state
      gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state
    timeout_ms: 5400000
  - name: pf-task-parity-golden
    command: "sh -c 'grep -q \"offsets.scent_turn_weights\" crates/xagent-sandbox/src/agent/mod.rs && grep -q \"offsets.vision_turn_weights\" crates/xagent-sandbox/src/agent/mod.rs && grep -q \"pub fn steering_population(\" crates/xagent-sandbox/src/agent/mod.rs && grep -q \"pub fn champion_brain(&self, node_id: i64)\" crates/xagent-sandbox/src/governor.rs && grep -q \"pub fn run_headless(\" crates/xagent-sandbox/src/headless.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-sense-descriptor
    command: "sh -c 'grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs && cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent buffers::tests::config_indices_within_bounds'"
    timeout_ms: 3600000
  - name: pf-task-derived-sense-layout
    command: "sh -c 'grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs && cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent buffers::tests::config_indices_within_bounds'"
    timeout_ms: 3600000
  - name: pf-task-sense-override-constants
    command: "sh -c 'grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs && cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent buffers::tests::config_indices_within_bounds'"
    timeout_ms: 3600000
  - name: pf-task-generic-sense-helpers
    command: "sh -c 'grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact gpu_kernel::visual_pathway_validation::visual_pathway_composition_preserves_other_passes gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_complete_state gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_cortex_skip gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state gpu_kernel::cooperative_jacobi_validation::jacobi_composition_keeps_other_helpers_and_scratch gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_complete_state gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state'"
    timeout_ms: 5400000
  - name: pf-task-generic-sense-block-sites
    command: "sh -c 'grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact gpu_kernel::visual_pathway_validation::visual_pathway_composition_preserves_other_passes gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_complete_state gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_cortex_skip gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state gpu_kernel::cooperative_jacobi_validation::jacobi_composition_keeps_other_helpers_and_scratch gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_complete_state gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state'"
    timeout_ms: 5400000
  - name: pf-task-declared-sense-extension-test
    command: "sh -c 'env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required && grep -q \"fn vision_pathway_step(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"whiten_scent(\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"O_TRACE_VISION\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"override O_SCENT_MEAN: u32\" crates/xagent-brain/src/shaders/kernel/common.wgsl && grep -q \"pub const O_SCENT_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub const O_SENSORY_MEAN: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"fn vision_override_constants(layout: &BrainLayout)\" crates/xagent-brain/src/gpu_kernel.rs'"
    timeout_ms: 3600000
  - name: pf-task-modulator-flags
    command: "sh -c 'grep -q \"KLINOTAXIS_SENSITIVITY\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"turn \\*= klinotaxis_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"brain_state\\[brain_base + O_FATIGUE_FACTOR\\] = fatigue_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"pub const CONFIG_SIZE: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub danger_percept_enabled: bool,\" crates/xagent-shared/src/config.rs && cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent buffers::tests::config_indices_within_bounds'"
    timeout_ms: 3600000
  - name: pf-task-modulator-study
    command: "sh -c 'grep -q \"KLINOTAXIS_SENSITIVITY\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"turn \\*= klinotaxis_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"brain_state\\[brain_base + O_FATIGUE_FACTOR\\] = fatigue_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"pub const CONFIG_SIZE: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub danger_percept_enabled: bool,\" crates/xagent-shared/src/config.rs && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-pooling-study
    command: "sh -c 'grep -q \"pub fn read_brain_scratch_blocking\" crates/xagent-brain/src/gpu_kernel.rs && grep -q \"fn vision_hemifields() -> array<f32, 8>\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"columns, whitened, learned | +0.16\" docs/reviews/2026-09-25-learner-foraging-investigation.md && grep -q \"hemifields, whitened, learned\\*\\* | \\*\\*+0.33\" docs/reviews/2026-09-25-learner-foraging-investigation.md && env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required'"
    timeout_ms: 3600000
  - name: pf-task-learned-pooling-port
    command: "sh -c 'grep -q \"pub fn read_brain_scratch_blocking\" crates/xagent-brain/src/gpu_kernel.rs && grep -q \"fn vision_hemifields() -> array<f32, 8>\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"columns, whitened, learned | +0.16\" docs/reviews/2026-09-25-learner-foraging-investigation.md && grep -q \"hemifields, whitened, learned\\*\\* | \\*\\*+0.33\" docs/reviews/2026-09-25-learner-foraging-investigation.md && grep -q \"KLINOTAXIS_SENSITIVITY\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"turn \\*= klinotaxis_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"brain_state\\[brain_base + O_FATIGUE_FACTOR\\] = fatigue_factor;\" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl && grep -q \"pub const CONFIG_SIZE: usize\" crates/xagent-brain/src/buffers.rs && grep -q \"pub danger_percept_enabled: bool,\" crates/xagent-shared/src/config.rs'"
execution_checks:
  - name: parity-reproducibility-witness
    tasks:
      - parity-golden
    role: architect
    operations:
      - edit
      - shell
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
    fixtures:
      - source: docs/plans/0003-uniform-sense-motor-route/execution/parity_reproducibility_probe.rs
        target: crates/xagent-sandbox/tests/sense_block_parity.rs
    actions:
      - name: seeded-gpu-run
        command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored seeded_gpu_trajectory_is_reproducible
        timeout_ms: 3600000
    constraints:
      - task: parity-golden
        name: seeded-runs-reproduce
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/governor.rs
          - crates/xagent-sandbox/src/agent/mod.rs
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: headless-and-champion-api
            command: >-
              sh -c 'grep -q "pub fn run_headless(" crates/xagent-sandbox/src/headless.rs
              && grep -q "pub fn champion_brain(&self, node_id: i64)" crates/xagent-sandbox/src/governor.rs
              && grep -q "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs
              && grep -q "fn gpu_adapter_present_when_required()" crates/xagent-sandbox/tests/integration.rs'
          - name: gpu-adapter
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
  - name: study-route-witness
    tasks:
      - modulator-study
      - pooling-study
    role: architect
    operations:
      - edit
      - shell
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
    actions:
      - name: headless-cli-help
        command: cargo run -p xagent-sandbox --bin xagent -- --help
        timeout_ms: 3600000
    constraints:
      - task: modulator-study
        name: modulator-study-route
        sources:
          - crates/xagent-sandbox/src/headless.rs
          - crates/xagent-sandbox/src/main.rs
          - docs/reviews/2026-09-25-learner-foraging-investigation.md
        checks:
          - name: headless-cli-and-references
            command: >-
              sh -c 'grep -q "no_render: bool," crates/xagent-sandbox/src/main.rs
              && grep -q "generations: Option<u64>," crates/xagent-sandbox/src/main.rs
              && grep -q "config: Option<String>," crates/xagent-sandbox/src/main.rs
              && grep -q "pub fn run_headless(" crates/xagent-sandbox/src/headless.rs
              && grep -q "## Eighty generations with recombined steering" docs/reviews/2026-09-25-learner-foraging-investigation.md'
          - name: gpu-adapter
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: pooling-study
        name: pooling-study-route
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - docs/reviews/2026-09-25-learner-foraging-investigation.md
        checks:
          - name: pooling-references
            command: >-
              sh -c 'grep -q "pub fn read_brain_scratch_blocking" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "fn vision_hemifields() -> array<f32, 8>" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
              && grep -q "columns, whitened, learned | +0.16" docs/reviews/2026-09-25-learner-foraging-investigation.md'
          - name: gpu-adapter
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
  - name: developer-route-witness
    tasks:
      - sense-descriptor
      - derived-sense-layout
      - descriptor-driven-mutation
      - sense-override-constants
      - generic-sense-helpers
      - generic-sense-block-sites
      - generic-schedule-candidates
      - remove-sense-specific-wgsl
      - declared-sense-extension-test
      - modulator-flags
      - learned-pooling-port
    role: developer
    operations:
      - edit
    command: cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::config_indices_within_bounds
    constraints:
      - task: sense-descriptor
        name: table-constants-known
        sources:
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/lib.rs
        checks:
          - name: table-constants
            command: >-
              sh -c 'grep -q "const SCENT_FEATURE_OFFSET: u32 = 25u;" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "const SCENT_WHITENING_RATE: f32 = 0.01;" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "const VISION_WHITENING_REFRESH: u32 = 20u;" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "pub const SCENT_INITIAL_VARIANCE: f32" crates/xagent-brain/src/buffers.rs
              && grep -q "pub const O_VISION_PLASTICITY: usize" crates/xagent-brain/src/buffers.rs
              && grep -q "pub mod buffers;" crates/xagent-brain/src/lib.rs'
      - task: sense-override-constants
        name: override-map-extensible
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: override-map
            command: >-
              sh -c 'grep -q "fn vision_override_constants(layout: &BrainLayout)" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "map.insert(\"VISION_W\"" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "^override " crates/xagent-brain/src/shaders/kernel/common.wgsl'
      - task: generic-sense-helpers
        name: helper-callers-known
        sources:
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
          - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
        checks:
          - name: helper-callers
            command: >-
              sh -c 'grep -q "fn whiten_scent(" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "fn refresh_vision_whitening(" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -Eq "settle_(turn_pathways|senses)_at_death\(brain_base\)" crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
              && grep -Eq "settle_(turn_pathways|senses)_at_death\(brain_base\)" crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl'
      - task: generic-sense-block-sites
        name: site-anchors-and-validations
        sources:
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/src/gpu_kernel/context_gather.rs
          - crates/xagent-brain/src/gpu_kernel/brain_sections.rs
          - crates/xagent-brain/src/gpu_kernel/whitening_validation.rs
          - crates/xagent-brain/src/gpu_kernel/whitening_storage_validation.rs
        checks:
          - name: site-anchors
            command: >-
              sh -c 'grep -q "fn vision_pathway_step(" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
              && grep -q "const REFRESH_BLOCK: &str" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "cooperative_refresh_vision_whitening" crates/xagent-brain/src/gpu_kernel/context_gather.rs
              && grep -q "cooperative_refresh_vision_whitening" crates/xagent-brain/src/gpu_kernel/brain_sections.rs
              && grep -q "REFRESH_BLOCK" crates/xagent-brain/src/gpu_kernel/whitening_validation.rs
              && grep -q "refresh_vision_whitening" crates/xagent-brain/src/gpu_kernel/whitening_storage_validation.rs'
          - name: pathway-validations
            command: >-
              env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact
              gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state
              gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state
              gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state
            timeout_ms: 5400000
      - task: generic-schedule-candidates
        name: candidate-fragments-known
        sources:
          - crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs
          - crates/xagent-brain/src/gpu_kernel/cooperative_jacobi_validation.rs
          - crates/xagent-brain/src/shaders/kernel/brain_cooperative_visual_pathway.wgsl
          - crates/xagent-brain/src/shaders/kernel/brain_cooperative_jacobi.wgsl
          - crates/xagent-brain/src/shaders/kernel/whitening_refresh.wgsl
          - crates/xagent-brain/src/shaders/kernel/brain_cooperative_whitening.wgsl
        checks:
          - name: candidate-fragments
            command: >-
              sh -c 'grep -q "brain_cooperative_visual_pathway.wgsl" crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs
              && grep -q "brain_cooperative_jacobi.wgsl" crates/xagent-brain/src/gpu_kernel/cooperative_jacobi_validation.rs
              && grep -q "VISION_WHITENING_REFRESH" crates/xagent-brain/src/shaders/kernel/whitening_refresh.wgsl
              && grep -q "VISION_WHITENING_REFRESH" crates/xagent-brain/src/shaders/kernel/brain_cooperative_whitening.wgsl
              && grep -q "O_VISION_PATHWAY_MEAN" crates/xagent-brain/src/shaders/kernel/brain_cooperative_visual_pathway.wgsl
              && grep -q "O_VISION_PATHWAY_COVARIANCE" crates/xagent-brain/src/shaders/kernel/brain_cooperative_jacobi.wgsl'
          - name: composition-tests
            command: >-
              sh -c 'grep -q "fn visual_pathway_composition_preserves_other_passes" crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs
              && cargo test -p xagent-brain --lib -- --exact gpu_kernel::cooperative_jacobi_validation::jacobi_composition_keeps_other_helpers_and_scratch'
            timeout_ms: 3600000
      - task: remove-sense-specific-wgsl
        name: cascade-and-readme
        sources:
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-brain/README.md
        checks:
          - name: cascade-present
            command: >-
              sh -c 'grep -q "override O_VISION_PATHWAY_MEAN: u32" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "O_SCENT_MEAN" crates/xagent-brain/README.md'
      - task: learned-pooling-port
        name: port-surfaces-known
        sources:
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-shared/src/config.rs
          - docs/reviews/2026-09-25-learner-foraging-investigation.md
        checks:
          - name: port-surfaces
            command: >-
              sh -c 'grep -q "fn vision_hemifields()" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
              && grep -q "pub visual_cortex_enabled: bool," crates/xagent-shared/src/config.rs
              && grep -q "columns, whitened, learned" docs/reviews/2026-09-25-learner-foraging-investigation.md'
      - task: derived-sense-layout
        name: offsets-derivable
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: offset-cascade
            command: >-
              sh -c 'grep -q "pub const O_SCENT_MEAN: usize" crates/xagent-brain/src/buffers.rs
              && grep -q "pub const O_TRACE_VISION: usize" crates/xagent-brain/src/buffers.rs
              && grep -q "pub const O_SENSORY_MEAN: usize" crates/xagent-brain/src/buffers.rs
              && grep -q "override O_SCENT_MEAN: u32" crates/xagent-brain/src/shaders/kernel/common.wgsl
              && grep -q "override O_SENSORY_MEAN: u32" crates/xagent-brain/src/shaders/kernel/common.wgsl'
          - name: layout-tests
            command: cargo test -p xagent-brain --lib -- --exact buffers::tests::brain_stride_is_consistent buffers::tests::brain_layout_dynamic_is_consistent
            timeout_ms: 3600000
      - task: descriptor-driven-mutation
        name: mutation-order-known
        sources:
          - crates/xagent-sandbox/src/agent/mod.rs
        checks:
          - name: mutation-order
            command: >-
              sh -c 'grep -q "offsets.scent_turn_weights" crates/xagent-sandbox/src/agent/mod.rs
              && grep -q "offsets.vision_turn_weights" crates/xagent-sandbox/src/agent/mod.rs
              && grep -q "STEERING_MUTATION_FRACTION" crates/xagent-sandbox/src/agent/mod.rs'
      - task: declared-sense-extension-test
        name: kernel-api-available
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/tests/vision_pathway.rs
        checks:
          - name: kernel-constructor
            command: >-
              sh -c 'grep -q "pub fn reset_agents_seeded(&mut self, brain_config: &BrainConfig, seed: u64)" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "pub fn read_agent_state(&self, index: u32) -> AgentBrainState" crates/xagent-brain/src/gpu_kernel.rs
              && grep -q "O_VISION_TURN_WEIGHTS" crates/xagent-brain/tests/vision_pathway.rs'
          - name: gpu-adapter
            command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration -- --exact gpu_adapter_present_when_required
            timeout_ms: 3600000
      - task: modulator-flags
        name: modulators-and-config-slots
        sources:
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-shared/src/config.rs
        checks:
          - name: modulator-anchors
            command: >-
              sh -c 'grep -q "KLINOTAXIS_SENSITIVITY" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
              && grep -q "pub const CONFIG_SIZE: usize" crates/xagent-brain/src/buffers.rs
              && grep -q "pub danger_percept_enabled: bool," crates/xagent-shared/src/config.rs'
          - name: config-slot-tests
            command: cargo test -p xagent-brain --lib -- --exact buffers::tests::config_indices_within_bounds buffers::tests::config_size_fits_vec4_alignment
            timeout_ms: 3600000
  - name: integration-witness
    tasks:
      - $integration
    role: architect
    command: >-
      sh -c 'grep -q "pub struct BrainLayout" crates/xagent-brain/src/buffers.rs
      && grep -q "pub const O_SENSORY_MEAN: usize" crates/xagent-brain/src/buffers.rs
      && grep -q "override O_SENSORY_MEAN: u32" crates/xagent-brain/src/shaders/kernel/common.wgsl
      && grep -q "pub fn steering_population(" crates/xagent-sandbox/src/agent/mod.rs'
    constraints:
      - task: $integration
        name: joint-route
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-sandbox/src/agent/mod.rs
          - crates/xagent-shared/src/config.rs
        checks:
          - name: joint-anchors
            command: >-
              sh -c 'grep -q "pub fn from_config(config: &BrainConfig) -> Self" crates/xagent-brain/src/buffers.rs
              && grep -q "fn coop_learn_and_store(" crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
              && grep -q "pub fn mutate_steering_weights(" crates/xagent-sandbox/src/agent/mod.rs
              && grep -q "pub fatigue_floor: f32," crates/xagent-shared/src/config.rs'
---

# 0003 — Uniform sense-motor route

## Scope

- A declarative **sense descriptor** (`crates/xagent-brain/src/senses.rs`):
  input channels and their source, adaptation (whitening-statistics) rate,
  whitening method, learning-rate source, and the motor routes the sense feeds
  (turn always; forward and critic only if their gate passed in plan 0002),
  each route with a declared heritability.
- **Brain-state offsets derived from `BrainLayout`**: every sense slot (mean,
  covariance, whitening matrix, whitened input, per-route weights and traces)
  is computed from the descriptor table. The smell and vision slots, and any
  route slots plan 0002 added, land at exactly their pre-plan offsets, so
  `BRAIN_STRIDE` does not change and stored champion databases still load and
  replay identically.
- **One generic WGSL block** (adapt → whiten → route dot products → traces →
  TD update → terminal settle at death) that loops over the declared senses
  using pipeline override constants generated from `BrainLayout`. The WGSL
  holds no smell- or vision-specific code.
- **Parity proof**: a golden captured from the pre-refactor tree (seeded
  multi-agent GPU trajectories, a seeded headless evolution run, a stored
  champion database replay, and a CPU steering-mutation draw) that every later
  task keeps byte-identical.
- **Heritability**: the smell and visual turn weights stay mutated exactly as
  today (same RNG draw order); every other sense or route weight, including any
  newly declared sense, is learned-only and zero at birth.
- **Investigation 1 (task 0301)**: offline test, on logged free lives, of
  pooling learned within a life against the hand-chosen left/right hemifields
  (reference: 32 whitened columns learned +0.16 against +0.33 for 8
  hemifields), ported behind a default-off flag only if it reaches the
  hemifield score on the same logs; decision recorded either way.
- **Investigation 2 (tasks 0401, 0402)**: the klinotaxis turn-gain and the
  motor-fatigue modulators each behind its own flag (default on), judged on
  within-life metrics from the plan-0001 free-life harness and on the per-seed
  80-generation reference, each landed or reverted with the result recorded.

## Non-goals

- No change to which routes exist: routes plan 0002 rejected are not wired into
  the block, and their flagged code (if any) is left exactly as plan 0002 left
  it.
- No new world condition, food kind or new sense in production; the extension
  proof (task 0204) declares an extra sense only inside a test.
- No change to the 128-d trunk, the encoder, the critic's 128-d input, memory,
  exploration noise or fitness, beyond what the two investigations land.
- No change to sensory adaptation (`O_SENSORY_MEAN`), the free-life harness
  (plan 0001's files are read and called, not edited), or the governor schema.
- No GPU-schedule (performance) work; the existing alternate schedules
  (cooperative whitening, cooperative Jacobi, visual-pathway parallelization)
  are re-expressed through the block and must stay state-identical.

## Task order

`parity-golden` captures the golden before anything changes. Then
`sense-descriptor` → `derived-sense-layout`, after which
`descriptor-driven-mutation` and the WGSL chain run in parallel. The WGSL chain
is staged so every step stays byte-identical within six files:
`sense-override-constants` → `generic-sense-helpers` →
`generic-sense-block-sites` → `generic-schedule-candidates` →
`remove-sense-specific-wgsl` → `declared-sense-extension-test`.
`modulator-flags` follows the chain and the mutation task (shared shader,
layout, config and agent files); `modulator-study` lands or reverts each
modulator; `pooling-study` then measures pooling on the final learner and
`learned-pooling-port` applies its verdict.

## Decision records (authored by tasks)

- `docs/decisions/sense-route-wiring.md`
- `docs/decisions/modulator-study.md`
- `docs/decisions/pooling-study.md`
