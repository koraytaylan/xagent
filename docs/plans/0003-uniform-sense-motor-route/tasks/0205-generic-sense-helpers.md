---
id: generic-sense-helpers
title: "Sense-indexed WGSL helpers behind the existing names"
workstream: "0200"
kind: task
role: developer
depends_on:
  - sense-override-constants
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
  - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
  - crates/xagent-brain/src/gpu_kernel.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test scent_pathway --test vision_pathway --test executed_action_credit"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
  - name: alternate-schedules
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_complete_state gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_complete_state gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state"
    timeout_ms: 7200000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# 0205 — Sense-indexed WGSL helpers behind the existing names

## Outcome

`common.wgsl` gains a complete set of named, sense-indexed accessors over the
`SENSE{k}_*` overrides added by `sense-override-constants`, and sense-indexed
helpers built only on those accessors. The old named functions remain as thin
wrappers, so every Rust text anchor and every unchanged caller still matches.
The two death callers call `settle_senses_at_death`. Smell is sense 0 and
vision is sense 1, the declared order of `DECLARED_SENSES`. Float operations
and their order are unchanged, so seeded trajectories stay byte-identical.

## Decisions

- **Accessors (fixed set, in `common.wgsl` before `fn refresh_vision_whitening`).**
  Each is implemented once by `switch sense { case 0u: ... case 3u: ... default: ... }`
  over `0u..MAX_SENSE_SLOTS` (4), returning the override of the matching
  suffix; `default` returns the inert value. No accessor reads a buffer, so every
  value is uniform across the workgroup.
  - `fn sense_count() -> u32` takes no argument and returns `SENSE_COUNT`.
  - `fn sense_channels(sense: u32) -> u32` returns `SENSE{k}_CHANNELS`.
  - `fn sense_input_kind(sense: u32) -> u32` returns `SENSE{k}_INPUT_KIND`.
  - `fn sense_input_offset(sense: u32) -> u32` returns `SENSE{k}_INPUT_OFFSET`.
  - `fn sense_whitening_kind(sense: u32) -> u32` returns `SENSE{k}_WHITENING_KIND`.
  - `fn sense_refresh_ticks(sense: u32) -> u32` returns `SENSE{k}_REFRESH_TICKS`.
  - `fn sense_adaptation_rate(sense: u32) -> f32` returns `SENSE{k}_ADAPTATION_RATE`
    (inert `0.0`).
  - `fn sense_learning_rate_gene(sense: u32) -> u32` returns `SENSE{k}_LEARNING_RATE_GENE`.
  - `fn sense_route_mask(sense: u32) -> u32` returns `SENSE{k}_ROUTE_MASK`.
  - Slot offsets, `fn name(sense: u32) -> u32`, per-agent `brain_state` indices
    that callers add to `brain_base`: `sense_mean` (`_MEAN`),
    `sense_covariance` (`_COVARIANCE`), `sense_whitening` (`_WHITENING`),
    `sense_input` (`_INPUT`, the whitened input slot), `sense_turn_weights`
    (`_TURN_WEIGHTS`), `sense_turn_trace` (`_TURN_TRACE`),
    `sense_forward_weights` (`_FORWARD_WEIGHTS`), `sense_forward_trace`
    (`_FORWARD_TRACE`), `sense_critic_weights` (`_CRITIC_WEIGHTS`),
    `sense_critic_trace` (`_CRITIC_TRACE`).
  - Add `const MAX_SENSE_CHANNELS: u32 = 8u;` beside them (the name of the
    Rust `senses::MAX_SENSE_CHANNELS`; a `SENSE_` prefix would collide with the
    `sense_override_wgsl_mirrors_rust_codes` code-mirror scan).
- Helpers use these accessors only: no unindexed `SENSE_CHANNELS`, no smell or
  vision identifiers. Only the wrappers pass `0u` or `1u`.
- **Closed form.** `sense_whiten_closed_form(sense, centred, covariance)` keeps
  the `whiten_scent` arithmetic verbatim; `sense` is accepted and unused in the
  math. `whiten_scent` becomes a one-line wrapper calling it with `0u`.
- **Learning rate.** `sense_learning_rate(brain_base, sense)` returns
  `ACTION_WEIGHT_LEARNING_RATE` when `sense_learning_rate_gene(sense)` is `0u`;
  otherwise `ACTION_WEIGHT_LEARNING_RATE * clamp(brain_state[brain_base + gene],
  VISION_PLASTICITY_MIN, VISION_PLASTICITY_MAX)`. `vision_pathway_learning_rate`
  wraps it with `1u`.
- **Layout-relative gene offset (Rust).** `insert_sense_overrides` in
  `gpu_kernel.rs` emits `SENSE{k}_LEARNING_RATE_GENE` as the gene's offset in
  the live `BrainLayout`, not the static `buffers.rs` constant: for a nonzero
  `gene_offset_or_zero()`, `gene - O_PREDICTOR_CONTEXT_WEIGHT` plus the
  layout's predictor-context-weight offset
  (`feature_count * ENCODED_DIMENSION + ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION`);
  `0` stays `0`. This matches the WGSL `O_VISION_PLASTICITY` override, which
  moves with `FEATURE_COUNT` when the visual cortex is on. Add one new
  `gpu_kernel.rs` unit test asserting that, under a visual-cortex-enabled
  layout, `SENSE1_LEARNING_RATE_GENE` equals the layout-relative
  `O_VISION_PLASTICITY` offset and differs from the static constant. The same
  layout-relative shift applies to every nonzero `SENSE{k}_*` block slot offset
  `insert_sense_overrides` emits (mean, covariance, whitening, input, turn,
  forward and critic weights and traces), since `derive_sense_layout` places
  them from the static `SENSE_BASE` while the WGSL `O_SCENT_*` and
  `O_VISION_*` overrides move with `FEATURE_COUNT`; `0` stays `0`. The new test
  also asserts `SENSE1_TURN_WEIGHTS` equals the layout-relative
  `O_VISION_TURN_WEIGHTS` under the cortex layout. No other Rust change;
  existing test bodies are unchanged.
- **Route settle.** `sense_settle_routes_at_death(brain_base, sense)` steps only
  the Turn route, gated by `steering_weights_learn()` and
  `sense_route_mask(sense) & SENSE_ROUTE_TURN`, zeroing weights then trace per
  channel. No forward or critic settle is added. `settle_senses_at_death(brain_base)`
  loops `0u..sense_count()`. `settle_turn_pathways_at_death(brain_base)` becomes a
  wrapper calling the route settle for `0u` and `1u`.
- **Jacobi.** `sense_refresh_jacobi_whitening(brain_base, sense)` is the
  `refresh_vision_whitening` body with `n = sense_channels(sense)`, matrix indices
  `i * n + j`, `f32(n)` in the floor, and covariance and whitening offsets from
  `sense_covariance` and `sense_whitening`. `refresh_vision_whitening(brain_base)`
  becomes a wrapper calling it with `1u`.
- **Placement (accepted constraint).** The slice from `fn refresh_vision_whitening(brain_base: u32) {`
  to `fn settle_recent_moments_at_death` contains only the wrapper followed by
  `sense_refresh_jacobi_whitening`; the accessors sit above the wrapper. The
  source slice MUST keep the private `var a`/`var v` declaration and all its
  unrenamed `a[`/`v[` indexing (`whitening_storage_validation.rs:50-51`); that
  indexing is required, not a defect. Only the test's in-memory rewritten copy
  has no `a[`/`v[` (lines 52-56).

**Steps:**

1. **Accessors.** Add `MAX_SENSE_CHANNELS`, `sense_count()` and the 18
   `switch` accessors listed above, before `fn refresh_vision_whitening` in `common.wgsl`.
2. **Closed form.** Add `sense_whiten_closed_form(sense, centred, covariance)`
   with the `whiten_scent` body unchanged; make `whiten_scent` a one-line wrapper calling it with `0u`.
3. **Learning rate and turn settle.** Add `sense_learning_rate(brain_base, sense)`,
   `sense_settle_routes_at_death(brain_base, sense)` and
   `settle_senses_at_death(brain_base)` as described above; make
   `vision_pathway_learning_rate` and `settle_turn_pathways_at_death` the wrappers.
4. **Jacobi.** Add `sense_refresh_jacobi_whitening(brain_base, sense)` directly
   after the `refresh_vision_whitening` wrapper and before `fn settle_recent_moments_at_death`;
   the wrapper calls it with `1u`.
5. **Death callers.** In `phase_death.wgsl` and `kernel_tick.wgsl`, replace
   `settle_turn_pathways_at_death(brain_base)` with `settle_senses_at_death(brain_base)`.
   Change nothing else in those files.
6. **Hygiene.** No `TODO`, plan, task or decision names in comments; no `as`
   casts in Rust; `brain_passes.wgsl` is not edited.

- **Done when:** INV-01 holds: the focused-proof, brain-lib and alternate-schedules
  checks pass unskipped and unedited, the five named validations included, the
  parity check reports the golden byte-identical, and the fmt, clippy and guard
  checks pass, proving INV-01 (seeded trajectories and the headless run match the
  pre-refactor golden; the naga composition in brain-lib accepts the accessors).

## Non-goals

- No edit to `brain_passes.wgsl`, any cooperative fragment, or any Rust file
  other than the `gpu_kernel.rs` layout-relative offset fix and its new test above.
- No new override, no new sense or route, and no forward or critic settle code.
- No change to a float expression's operand order, a constant's value, or the
  steering freeze.
- No change to the golden or to any test body.
