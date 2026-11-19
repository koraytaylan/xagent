---
objective: >-
  Find out whether the critic and the forward channel learn better from the ten
  whitened pathway inputs (two nostrils, eight visual hemifield means) than from
  the 128-d encoding, and adopt each route only if its own gate passes. First an
  offline fit of logged returns on the ten inputs, on the 128-d encoding and on
  both, scored by held-out R2 and by the in-view and centred value gaps, with a
  decision record. Then two routes, each behind its own default-off flag and its
  own gate: (a) the critic reads the winning input set; (b) the forward channel
  gets plastic, learned-only, zero-at-birth weights on the whitened smell and
  visual inputs under the existing pathway rule. The forward gate uses the
  conditional credit (forward-noise credit times each whitened pathway input),
  never unconditional forward credit. A negative result is recorded and the
  route stays off.
depends_on:
  - 0001-free-life-harness-and-trunk-lesions
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
  - name: workspace-fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: workspace-clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: workspace-check
    command: "cargo check --workspace"
    timeout_ms: 3600000
  - name: sandbox-tests
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox"
    timeout_ms: 7200000
  - name: brain-lib-tests
    command: "cargo test -p xagent-brain --lib"
    timeout_ms: 3600000
  - name: brain-vision-pathway
    command: "cargo test -p xagent-brain --test vision_pathway"
    timeout_ms: 3600000
  - name: shared-tests
    command: "cargo test -p xagent-shared"
    timeout_ms: 3600000
  - name: critic-inputs-record
    command: "cargo run -p xagent-sandbox --bin pathway_return_fit -- verify docs/decisions/pathway-critic-inputs.json"
    timeout_ms: 3600000
  - name: critic-route-record-integrated
    command: "cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-critic-route.json"
    timeout_ms: 3600000
  - name: forward-route-record-integrated
    command: "cargo run -p xagent-sandbox --bin pathway_route_gate -- verify docs/decisions/pathway-forward-route.json"
    timeout_ms: 3600000
preflight_checks:
  - name: witness-cli-config
    command: "grep -q -F 'fn resolve_config(cli: &Cli) -> FullConfig' crates/xagent-sandbox/src/main.rs"
  - name: witness-no-deny-unknown-fields
    command: "awk 'index($0, \"deny_unknown_fields\") { found = 1 } END { exit found }' crates/xagent-shared/src/config.rs"
  - name: witness-config-size
    command: "grep -q -F 'pub const CONFIG_SIZE: usize = 24;' crates/xagent-brain/src/buffers.rs"
  - name: witness-sensory-mean-follows-vision-trace
    command: "grep -q -F 'pub const O_SENSORY_MEAN: usize = O_TRACE_VISION + VISION_PATHWAY_INPUTS;' crates/xagent-brain/src/buffers.rs"
  - name: witness-wgsl-sensory-mean
    command: "grep -q -F 'override O_SENSORY_MEAN: u32 = O_TRACE_VISION + VISION_PATHWAY_INPUTS;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: witness-slot-block-unambiguous
    command: "awk 'BEGIN { exit ((10 + 10 + 10 + 10 + 1 + 128 * 10) % (128 + 1) == 0) }'"
  - name: witness-decode-brain-state
    command: "grep -q -F 'fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState>' crates/xagent-sandbox/src/governor.rs"
  - name: witness-value-before-whitening
    command: "awk 'index($0, \"if (tid == 0u) { s_value = s_dense_partials[0]; }\") && !v { v = NR } index($0, \"brain_state[brain_base + O_SCENT_WHITENED] = scent_whitened.x;\") { w = NR } END { exit !(v && w && v < w) }' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: witness-forward-noise-fresh
    command: "grep -q -F 's_explore[0u] = noise_forward * exploration_rate * fatigue_factor;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: witness-pathway-trace-rule
    command: "grep -q -F '+ s_explore[2u] * input / (1.0 + vision_sq);' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: witness-death-settle
    command: "grep -q -F 'fn settle_turn_pathways_at_death(brain_base: u32)' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: witness-composition-markers
    command: "awk 'FILENAME ~ /gpu_kernel.rs$/ && index($0, \"fn compose_brain_passes(cooperative_whitening: bool) -> String {\") { c = 1 } FILENAME ~ /visual_pathway_validation.rs$/ && index($0, \"const ORIGINAL_CALL: &str = \") && index($0, \"vision_pathway_step(\") { o = 1 } FILENAME ~ /brain_passes.wgsl$/ && index($0, \"// ── Thread 0: exploration, noise, motor, telemetry\") { e++ } END { exit !(c && o && e == 1) }' crates/xagent-brain/src/gpu_kernel.rs crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
  - name: witness-adapter-log
    command: "grep -q -F 'log::info!(\"[GpuKernel] Adapter: {:?}\", adapter.get_info());' crates/xagent-brain/src/gpu_kernel.rs"
  - name: witness-bin-autodiscovery
    command: "awk 'index($0, \"autobins\") { found = 1 } END { exit found }' crates/xagent-sandbox/Cargo.toml"
  - name: witness-rusqlite-dependency
    command: "grep -q -F 'rusqlite = { version = \"0.34\"' crates/xagent-sandbox/Cargo.toml"
  - name: witness-generation-fitness-schema
    command: "awk '/generation INTEGER NOT NULL/ { a = 1 } /composite_fitness REAL/ { b = 1 } END { exit !(a && b) }' crates/xagent-sandbox/src/governor.rs"
  - name: witness-td-discount
    command: "grep -q -F 'const TD_DISCOUNT: f32 = 0.97;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: witness-require-gpu-convention
    command: "grep -q -F 'XAGENT_REQUIRE_GPU' crates/xagent-sandbox/tests/integration.rs"
  - name: witness-ci-commands
    command: "grep -q -F 'cargo clippy --workspace --all-targets -- -D warnings' .github/workflows/ci.yml"
  - name: witness-80-generation-reference
    command: "grep -q -F -- '--generations 80' docs/reviews/2026-09-25-learner-foraging-investigation.md"
  - name: witness-full-config-literals
    command: "awk 'index($0, \"cortex_stage_limit: parent.cortex_stage_limit,\") { a = 1 } index($0, \"cortex_stage_limit: 0,\") { b = 1 } END { exit !(a && b) }' crates/xagent-sandbox/src/agent/mod.rs"
  - name: preflight-default-fingerprint
    command: "awk 'FILENAME ~ /main.rs$/ && index($0, \"fn resolve_config(cli: &Cli) -> FullConfig\") { a = 1 } FILENAME ~ /config.rs$/ && index($0, \"deny_unknown_fields\") { d = 1 } END { exit !(a && !d) }' crates/xagent-sandbox/src/main.rs crates/xagent-shared/src/config.rs"
  - name: preflight-route-flags
    command: "awk 'FILENAME ~ /config.rs$/ && index($0, \"deny_unknown_fields\") { d = 1 } FILENAME ~ /mod.rs$/ && index($0, \"cortex_stage_limit: parent.cortex_stage_limit,\") { a = 1 } FILENAME ~ /mod.rs$/ && index($0, \"cortex_stage_limit: 0,\") { b = 1 } END { exit !(a && b && !d) }' crates/xagent-shared/src/config.rs crates/xagent-sandbox/src/agent/mod.rs"
  - name: preflight-brain-slot-layout
    command: "awk 'FILENAME ~ /buffers.rs$/ && index($0, \"pub const CONFIG_SIZE: usize = 24;\") { c = 1 } FILENAME ~ /buffers.rs$/ && index($0, \"pub const O_SENSORY_MEAN: usize = O_TRACE_VISION + VISION_PATHWAY_INPUTS;\") { s = 1 } FILENAME ~ /common.wgsl$/ && index($0, \"override O_SENSORY_MEAN: u32 = O_TRACE_VISION + VISION_PATHWAY_INPUTS;\") { w = 1 } END { u = ((10 + 10 + 10 + 10 + 1 + 128 * 10) % (128 + 1) != 0); exit !(c && s && w && u) }' crates/xagent-brain/src/buffers.rs crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: preflight-kernel-routes
    command: "awk 'FILENAME ~ /brain_passes.wgsl$/ && index($0, \"if (tid == 0u) { s_value = s_dense_partials[0]; }\") && !v { v = FNR } FILENAME ~ /brain_passes.wgsl$/ && index($0, \"brain_state[brain_base + O_SCENT_WHITENED] = scent_whitened.x;\") { w = FNR } FILENAME ~ /brain_passes.wgsl$/ && index($0, \"s_explore[0u] = noise_forward * exploration_rate * fatigue_factor;\") { n = 1 } FILENAME ~ /brain_passes.wgsl$/ && index($0, \"+ s_explore[2u] * input / (1.0 + vision_sq);\") { t = 1 } FILENAME ~ /common.wgsl$/ && index($0, \"fn settle_turn_pathways_at_death(brain_base: u32)\") { d = 1 } FILENAME ~ /integration.rs$/ && index($0, \"XAGENT_REQUIRE_GPU\") { g = 1 } END { exit !(v && w && v < w && n && t && d && g) }' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl crates/xagent-brain/src/shaders/kernel/common.wgsl crates/xagent-sandbox/tests/integration.rs"
  - name: preflight-return-fit-tool
    command: "awk 'FILENAME ~ /Cargo.toml$/ && index($0, \"autobins\") { b = 1 } FILENAME ~ /common.wgsl$/ && index($0, \"const TD_DISCOUNT: f32 = 0.97;\") { t = 1 } END { exit !(t && !b) }' crates/xagent-sandbox/Cargo.toml crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: preflight-route-gate-tool
    command: "awk 'FILENAME ~ /Cargo.toml$/ && index($0, \"autobins\") { b = 1 } FILENAME ~ /Cargo.toml$/ && index($0, \"rusqlite = { version = \") && index($0, \"0.34\") { r = 1 } FILENAME ~ /governor.rs$/ && /generation INTEGER NOT NULL/ { g = 1 } FILENAME ~ /governor.rs$/ && /composite_fitness REAL/ { f = 1 } END { exit !(r && g && f && !b) }' crates/xagent-sandbox/Cargo.toml crates/xagent-sandbox/src/governor.rs"
  - name: preflight-critic-input-decision
    command: "awk 'FILENAME ~ /main.rs$/ && index($0, \"fn resolve_config(cli: &Cli) -> FullConfig\") { a = 1 } FILENAME ~ /common.wgsl$/ && index($0, \"const TD_DISCOUNT: f32 = 0.97;\") { t = 1 } END { exit !(a && t) }' crates/xagent-sandbox/src/main.rs crates/xagent-brain/src/shaders/kernel/common.wgsl"
  - name: preflight-route-gate-decisions
    command: "awk 'FILENAME ~ /main.rs$/ && index($0, \"fn resolve_config(cli: &Cli) -> FullConfig\") { a = 1 } FILENAME ~ /governor.rs$/ && /generation INTEGER NOT NULL/ { g = 1 } FILENAME ~ /governor.rs$/ && /composite_fitness REAL/ { f = 1 } FILENAME ~ /investigation.md$/ && index($0, \"--generations 80\") { r = 1 } END { exit !(a && g && f && r) }' crates/xagent-sandbox/src/main.rs crates/xagent-sandbox/src/governor.rs docs/reviews/2026-09-25-learner-foraging-investigation.md"
execution_checks:
  - name: developer-implementation-route
    tasks:
      - "route-flags"
      - "brain-slot-layout"
      - "legacy-decode-and-birth-state"
      - "kernel-routes"
      - "return-fit-tool"
      - "route-gate-tool"
    role: developer
    operations:
      - edit
    command: "cargo test -p xagent-sandbox --no-run"
    timeout_ms: 3600000
    constraints:
      - task: "route-flags"
        name: serde-defaulted-flag-fields
        sources:
          - crates/xagent-shared/src/config.rs
          - crates/xagent-sandbox/src/agent/mod.rs
        checks:
          - name: witness-no-deny-unknown-fields
            command: "awk 'index($0, \"deny_unknown_fields\") { found = 1 } END { exit found }' crates/xagent-shared/src/config.rs"
          - name: witness-full-config-literals
            command: "awk 'index($0, \"cortex_stage_limit: parent.cortex_stage_limit,\") { a = 1 } index($0, \"cortex_stage_limit: 0,\") { b = 1 } END { exit !(a && b) }' crates/xagent-sandbox/src/agent/mod.rs"
      - task: "legacy-decode-and-birth-state"
        name: single-blob-decoder
        sources:
          - crates/xagent-sandbox/src/governor.rs
        checks:
          - name: witness-decode-brain-state
            command: "grep -q -F 'fn decode_brain_state(bytes: &[u8]) -> Option<AgentBrainState>' crates/xagent-sandbox/src/governor.rs"
      - task: "brain-slot-layout"
        name: layout-and-flag-slots
        sources:
          - crates/xagent-brain/src/buffers.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: witness-config-size
            command: "grep -q -F 'pub const CONFIG_SIZE: usize = 24;' crates/xagent-brain/src/buffers.rs"
          - name: witness-sensory-mean-follows-vision-trace
            command: "grep -q -F 'pub const O_SENSORY_MEAN: usize = O_TRACE_VISION + VISION_PATHWAY_INPUTS;' crates/xagent-brain/src/buffers.rs"
          - name: witness-wgsl-sensory-mean
            command: "grep -q -F 'override O_SENSORY_MEAN: u32 = O_TRACE_VISION + VISION_PATHWAY_INPUTS;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
          - name: witness-slot-block-unambiguous
            command: "awk 'BEGIN { exit ((10 + 10 + 10 + 10 + 1 + 128 * 10) % (128 + 1) == 0) }'"
      - task: "kernel-routes"
        name: kernel-route-insertion-points
        sources:
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: witness-value-before-whitening
            command: "awk 'index($0, \"if (tid == 0u) { s_value = s_dense_partials[0]; }\") && !v { v = NR } index($0, \"brain_state[brain_base + O_SCENT_WHITENED] = scent_whitened.x;\") { w = NR } END { exit !(v && w && v < w) }' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: witness-forward-noise-fresh
            command: "grep -q -F 's_explore[0u] = noise_forward * exploration_rate * fatigue_factor;' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: witness-pathway-trace-rule
            command: "grep -q -F '+ s_explore[2u] * input / (1.0 + vision_sq);' crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: witness-death-settle
            command: "grep -q -F 'fn settle_turn_pathways_at_death(brain_base: u32)' crates/xagent-brain/src/shaders/kernel/common.wgsl"
          - name: witness-require-gpu-convention
            command: "grep -q -F 'XAGENT_REQUIRE_GPU' crates/xagent-sandbox/tests/integration.rs"
      - task: "kernel-routes"
        name: shader-text-composition-preserved
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
          - crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs
          - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
        checks:
          - name: witness-composition-markers
            command: "awk 'FILENAME ~ /gpu_kernel.rs$/ && index($0, \"fn compose_brain_passes(cooperative_whitening: bool) -> String {\") { c = 1 } FILENAME ~ /visual_pathway_validation.rs$/ && index($0, \"const ORIGINAL_CALL: &str = \") && index($0, \"vision_pathway_step(\") { o = 1 } FILENAME ~ /brain_passes.wgsl$/ && index($0, \"// ── Thread 0: exploration, noise, motor, telemetry\") { e++ } END { exit !(c && o && e == 1) }' crates/xagent-brain/src/gpu_kernel.rs crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl"
          - name: brain-lib-baseline
            command: "cargo test -p xagent-brain --lib"
            timeout_ms: 3600000
      - task: "return-fit-tool"
        name: return-fit-binary-target
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: witness-bin-autodiscovery
            command: "awk 'index($0, \"autobins\") { found = 1 } END { exit found }' crates/xagent-sandbox/Cargo.toml"
          - name: witness-td-discount
            command: "grep -q -F 'const TD_DISCOUNT: f32 = 0.97;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
      - task: "route-gate-tool"
        name: route-gate-binary-target
        sources:
          - crates/xagent-sandbox/Cargo.toml
          - crates/xagent-sandbox/src/governor.rs
        checks:
          - name: witness-bin-autodiscovery
            command: "awk 'index($0, \"autobins\") { found = 1 } END { exit found }' crates/xagent-sandbox/Cargo.toml"
          - name: witness-rusqlite-dependency
            command: "grep -q -F 'rusqlite = { version = \"0.34\"' crates/xagent-sandbox/Cargo.toml"
          - name: witness-generation-fitness-schema
            command: "awk '/generation INTEGER NOT NULL/ { a = 1 } /composite_fitness REAL/ { b = 1 } END { exit !(a && b) }' crates/xagent-sandbox/src/governor.rs"
  - name: architect-measurement-route
    tasks:
      - "default-fingerprint"
      - "critic-input-decision"
      - "route-gate-decisions"
    role: architect
    operations:
      - shell
      - edit
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration gpu_adapter_present_when_required"
    timeout_ms: 7200000
    actions:
      - name: dump-default-config
        command: "cargo run --release -p xagent-sandbox --bin xagent -- --dump-config"
        timeout_ms: 3600000
    constraints:
      - task: "default-fingerprint"
        name: pinned-config-through-cli
        sources:
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-shared/src/config.rs
        checks:
          - name: witness-cli-config
            command: "grep -q -F 'fn resolve_config(cli: &Cli) -> FullConfig' crates/xagent-sandbox/src/main.rs"
          - name: witness-no-deny-unknown-fields
            command: "awk 'index($0, \"deny_unknown_fields\") { found = 1 } END { exit found }' crates/xagent-shared/src/config.rs"
      - task: "default-fingerprint"
        name: adapter-name-from-info-log
        sources:
          - crates/xagent-brain/src/gpu_kernel.rs
        checks:
          - name: witness-adapter-log
            command: "grep -q -F 'log::info!(\"[GpuKernel] Adapter: {:?}\", adapter.get_info());' crates/xagent-brain/src/gpu_kernel.rs"
      - task: "critic-input-decision"
        name: logged-free-lives-through-cli
        sources:
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-brain/src/shaders/kernel/common.wgsl
        checks:
          - name: witness-cli-config
            command: "grep -q -F 'fn resolve_config(cli: &Cli) -> FullConfig' crates/xagent-sandbox/src/main.rs"
          - name: witness-td-discount
            command: "grep -q -F 'const TD_DISCOUNT: f32 = 0.97;' crates/xagent-brain/src/shaders/kernel/common.wgsl"
      - task: "route-gate-decisions"
        name: paired-lives-and-80-generation-runs
        sources:
          - crates/xagent-sandbox/src/main.rs
          - crates/xagent-sandbox/src/governor.rs
          - docs/reviews/2026-09-25-learner-foraging-investigation.md
        checks:
          - name: witness-cli-config
            command: "grep -q -F 'fn resolve_config(cli: &Cli) -> FullConfig' crates/xagent-sandbox/src/main.rs"
          - name: witness-generation-fitness-schema
            command: "awk '/generation INTEGER NOT NULL/ { a = 1 } /composite_fitness REAL/ { b = 1 } END { exit !(a && b) }' crates/xagent-sandbox/src/governor.rs"
          - name: witness-80-generation-reference
            command: "grep -q -F -- '--generations 80' docs/reviews/2026-09-25-learner-foraging-investigation.md"
  - name: integration-route
    tasks:
      - "$integration"
    role: architect
    operations:
      - edit
    command: "cargo fmt --all -- --check"
    timeout_ms: 600000
    constraints:
      - task: "$integration"
        name: repository-gates
        sources:
          - .github/workflows/ci.yml
          - crates/xagent-sandbox/tests/integration.rs
        checks:
          - name: witness-ci-commands
            command: "grep -q -F 'cargo clippy --workspace --all-targets -- -D warnings' .github/workflows/ci.yml"
          - name: witness-require-gpu-convention
            command: "grep -q -F 'XAGENT_REQUIRE_GPU' crates/xagent-sandbox/tests/integration.rs"
---

# Pathway critic and forward drive

## Scope

This plan answers one question twice: does a learner channel learn better from
the ten whitened pathway inputs (whitened left and right nostril, then the
eight whitened visual hemifield means, in that order) than from the 128-d
encoding? It asks it for the critic and for the forward channel, and adopts
each answer only through that route's own gate.

In scope:

- A routes-off fingerprint of a seeded free life, recorded before any change,
  proving that both routes switched off leave behaviour bit-identical
  (default-fingerprint).
- The route flags `critic_input` and `forward_pathway_enabled` in
  `BrainConfig`, locked per batch, never heritable (route-flags).
- Their config-uniform slots and the new brain-state slot block (forward
  pathway weights and traces, critic pathway weights and traces, the previous
  pathway-input norm, the ring's pathway keys) with a legacy-length upgrade
  function (brain-slot-layout).
- The upgrade applied in the blob decoder so champion blobs stored at the plan
  baseline still load, and birth/mutation proofs that the block stays zero and
  the flags are inherited unchanged (legacy-decode-and-birth-state).
- The kernel side of both routes (kernel-routes): the critic input selection
  `encoding` / `pathway` / `encoding_and_pathway`, and the forward pathway
  under the existing pathway rule (fresh-innovation credit as executed,
  normalised by 1 + squared input size per sense). No existing kernel text
  moves (ARCHITECTURE D5): the critic reads the whitened inputs stored by the
  previous brain tick, the forward pathway reads this tick's, and the
  production cooperative-whitening composition and its text-validation tests
  stay valid (INV-13, `cargo test -p xagent-brain --lib`).
- An offline return-fit tool (return-fit-tool) and a route-gate tool
  (route-gate-tool).
- The offline fit decision record (critic-input-decision) and the route gate
  decision records with any adoption (route-gate-decisions).

Non-goals:

- No change to the 128-d turn policy, the smell or visual turn pathways, their
  heritability, the encoder, or the reward.
- No heritability for any new weight: every new weight is learned-only and
  zero at birth; mutation, recombination and inheritance never write it.
- No change to the free-life harness, its CLI flags, its metrics, its per-tick
  log format or the lesion switches owned by
  `0001-free-life-harness-and-trunk-lesions`; this plan only consumes them.
- No generic per-sense block or removal of hand-coded modulators (plan 0003),
  no food kinds (plan 0004).
- No unconditional forward credit in any gate.

## Prerequisites

`0001-free-life-harness-and-trunk-lesions` must be integrated: its free-life
mode (agent count, tick count, seed, fresh or stored-champion birth brain,
config from the shared `--config` resolution in `main.rs`), its per-agent
metrics, its per-brain-tick log (encoding, the ten whitened pathway inputs,
reward, value, executed exploration noise, ground-truth food and hazard
bearings), its lesion switches and its lesion decision record.

Plan 0001's strict seeded evolution digest (its fixture hashes `node_brain`
blobs and `config_json`) is expected to change once route-flags and
brain-slot-layout grow `BrainConfig` and the brain layout; that is a recorded consequence, not a regression.
Its default (non-strict) mode, which `cargo test -p xagent-sandbox` runs, only
checks run-to-run determinism and stays green. This plan's routes-off proof is
its own free-life fingerprint (INV-1), which hashes only behaviour (per-agent
metrics and tick-log rows), not layout or config text. The fingerprint belongs
to the GPU adapter it was recorded on; on another adapter (for example CI's
lavapipe, if the recording host differs) the replay prints a skip naming both
adapters instead of failing, while its digest-comparison unit tests run
everywhere.

All runs, checks and witnesses need a GPU adapter on the host (real or Mesa
lavapipe), as the repository's CI does; `XAGENT_REQUIRE_GPU=1` makes the GPU
tests fail instead of skipping when none is present.

## Decision records

All records live in `docs/decisions/` with a machine-readable JSON beside each
Markdown record, re-checked by the tools' `verify` subcommands:

- `docs/decisions/pathway-critic-inputs.md` / `.json` (critic-input-decision)
- `docs/decisions/pathway-critic-route.md` / `.json` (route-gate-decisions)
- `docs/decisions/pathway-forward-route.md` / `.json` (route-gate-decisions)

## Tasks

| File | Task id | Outcome | Role | Depends on |
|---|---|---|---|---|
| 0101 | default-fingerprint | Routes-off fingerprint fixture and its replay test | architect | — |
| 0102 | route-flags | `critic_input` / `forward_pathway_enabled` config fields | developer | — |
| 0103 | brain-slot-layout | Config slots, brain-state slot block, legacy upgrade function | developer | default-fingerprint, route-flags |
| 0104 | legacy-decode-and-birth-state | Upgrade in the blob decoder; zero-at-birth and inheritance proofs | developer | brain-slot-layout |
| 0201 | kernel-routes | Kernel critic input selection and forward pathway | developer | brain-slot-layout |
| 0301 | return-fit-tool | Offline return-fit tool | developer | — |
| 0302 | route-gate-tool | Route-gate tool (conditional credit, paired gates, verify) | developer | route-flags |
| 0401 | critic-input-decision | Offline fit run and critic-input decision record | architect | return-fit-tool |
| 0402 | route-gate-decisions | Paired gate runs, route decision records, adoption | architect | kernel-routes, legacy-decode-and-birth-state, route-gate-tool, critic-input-decision |

`route-gate-decisions` is the only task besides `route-flags` that edits
`crates/xagent-shared/src/config.rs`, and only to flip a `DEFAULT_*` constant
of a route whose gates all passed; it runs strictly after `route-flags`
through its dependency chain.
