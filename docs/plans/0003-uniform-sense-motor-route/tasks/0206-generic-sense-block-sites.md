---
id: generic-sense-block-sites
title: "Brain passes loop over the declared senses"
workstream: "0200"
kind: task
role: developer
depends_on:
  - generic-sense-helpers
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/src/gpu_kernel/context_gather.rs
  - crates/xagent-brain/src/gpu_kernel/brain_sections.rs
  - crates/xagent-brain/src/gpu_kernel/whitening_validation.rs
  - crates/xagent-brain/src/gpu_kernel/whitening_storage_validation.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --skip gpu_kernel::visual_pathway_validation::visual_pathway_composition_preserves_other_passes --skip gpu_kernel::cooperative_jacobi_validation::jacobi_composition_keeps_other_helpers_and_scratch"
    timeout_ms: 5400000
  - name: alternate-schedules
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state gpu_kernel::whitening_validation::isolated_whitening_with_interleaved_recall_matches_complete_serial_state gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state gpu_kernel::cooperative_whitening_validation::cooperative_brain_preserves_cortex_and_odd_vision_layouts"
    timeout_ms: 7200000
  - name: pathway-tests
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test scent_pathway --test vision_pathway --test executed_action_credit"
    timeout_ms: 3600000
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

# 0206 — Brain passes loop over the declared senses

## Outcome

In `brain_passes.wgsl` the hand-written smell and vision code at every site is
replaced by loops over `0..SENSE_COUNT` in declared order (smell is sense 0,
vision sense 1), built on the sense-indexed helpers and `SENSE{k}_*` overrides
from `generic-sense-helpers`. Float operation order is unchanged, so seeded
trajectories stay byte-identical. Afterwards `brain_passes.wgsl` calls none of
the four transitional wrapper names (`whiten_scent`, `refresh_vision_whitening`,
`vision_pathway_learning_rate`, `settle_turn_pathways_at_death`).

## Decisions

- Exploration block, per sense `k`, in this order: gather the input
  (`NonVisualFeatures` at its layout offset; `VisualHemifields` through the
  existing hemifield pooling, silent when the visual cortex is on), centre,
  whiten (closed form, or the Jacobi matrix with its refresh cadence), store
  the whitened input, update statistics. This sits at plan 0002's position
  before the value reduction.
- TD update, per declared route: weights move at the sense's rate and are
  clamped to the sense's per-sense L2 ball; the whole update is gated by
  `steering_weights_learn()`.
- Route dot products stay where they are today, after the weight update. The
  turn sum is `s_turn_dot` then sense 0 then sense 1. Forward is added at the
  forward command only if declared; critic follows plan 0002's D6 only if
  declared.
- Traces: each route's credited innovation (`s_explore[2]`) is normalised by
  `1 + |input|^2`, per sense, as today.
- Plan-0002 routes NOT declared keep their flag-gated code. Their 10 inputs are
  gathered through the sense loop and the layout offsets, not by new literals.
- Prerequisite accessors: the loops need per-sense `switch` accessors for input
  kind, input offset, whitening kind, refresh ticks, adaptation rate and the
  whitened-input slot. `generic-sense-helpers` must add them (names
  `sense_input_kind`, `sense_input_offset`, `sense_whitening_kind`,
  `sense_refresh_ticks`, `sense_adaptation_rate`, `sense_input`). This task does
  not edit `common.wgsl`.
- Refresh site: the old `REFRESH_BLOCK` text is replaced in place by exactly
  `    for (var sense = 0u; sense < SENSE_COUNT; sense++) {\n        if (sense_whitening_kind(sense) == SENSE_WHITENING_KIND_JACOBI\n            && !(visual_cortex_on && sense_input_kind(sense) == SENSE_INPUT_KIND_VISUAL_HEMIFIELDS)\n            && u32(brain_state[brain_base + O_TICK_COUNT]) % sense_refresh_ticks(sense) == 0u) {\n            sense_refresh_jacobi_whitening(brain_base, sense);\n        }\n    }\n`.
  The cortex term keeps today's early return: a `VisualHemifields` sense is not
  refreshed while the visual cortex is on.
- Accepted naming exception: `cooperative_refresh_vision_whitening(brain_base, tid);`
  keeps its name and call text. `gpu_kernel.rs`, `context_gather.rs` and
  `brain_sections.rs` keep searching that call; only the surrounding site text
  they search moves to the new block.
- Anchor moves (wrapper names leave `brain_passes.wgsl`):
  - `whitening_validation.rs`: its `REFRESH_BLOCK` const is the pinned text
    above; `passes.matches(REFRESH_BLOCK).count() == 1` is unchanged; the
    `refresh_vision_whitening(` count 1 and its `separated` check become
    `sense_refresh_jacobi_whitening(`, same count and strength.
  - `whitening_storage_validation.rs`: the slice begin anchor becomes
    `fn sense_refresh_jacobi_whitening(` and keeps its end anchor
    `fn settle_recent_moments_at_death`, with the rules from `generic-sense-helpers`
    (one private-matrix declaration, no residual `a[` or `v[` after rewriting).
    The `refresh_vision_whitening(` count 1 becomes `sense_refresh_jacobi_whitening(`
    count 1. The `vision_pathway_step(` count 2 becomes an exact count equal to
    the number of `for (var sense = 0u; sense < SENSE_COUNT; sense++) {` headers
    written by steps 1 to 4. Add exact-zero assertions in this file for
    `whiten_scent(`, `refresh_vision_whitening(`, `vision_pathway_learning_rate(`
    and `settle_turn_pathways_at_death(` in the passes source.
- The `focused-proof` check skips exactly two composition tests,
  `visual_pathway_composition_preserves_other_passes` and
  `jacobi_composition_keeps_other_helpers_and_scratch`. This skip is
  transitional: `generic-schedule-candidates` rewrites their fragments and the
  plan's `brain-lib-suite` runs them unskipped.

**Steps:**

1. **Input and whitening.** In `brain_passes.wgsl`, replace the vision hemifield
   gather, the smell gather and `vision_pathway_step` with one per-sense loop that
   gathers, centres, whitens via the `generic-sense-helpers` functions, stores the
   whitened input and updates statistics, before the value reduction.
2. **Route dot products.** Replace the hand-written `whiten_scent` and
   `vision_pathway_step` dot-product call sites with the per-sense dot product
   loop, keeping the summation order stated above.
3. **TD update.** Replace the smell and vision weight-update blocks with a loop
   over senses and declared Turn routes using each sense's rate and L2 ball,
   inside the existing `steering_weights_learn()` gates. Replace the
   `vision_pathway_learning_rate` call with `sense_learning_rate`.
4. **Traces and refresh.** Replace the smell and vision trace updates with a
   per-sense loop normalising the credited innovation by `1 + |input|^2`. Write
   the pinned refresh block in place of the old refresh call.
5. **Anchors.** Update the anchors listed in Decisions (`gpu_kernel.rs`,
   `context_gather.rs`, `brain_sections.rs`, both validation files) so each
   search matches once and every count keeps its strength.
6. **Hygiene.** No smell- or vision-specific identifiers in loop bodies; no
   plan, task or decision names in comments; no `as` casts in Rust; the two
   transitional composition tests keep their bodies.

- **Done when:** INV-01 and INV-09 hold: the focused-proof check (with only its
  two transitional skips) and the alternate-schedules and pathway-tests checks
  pass unskipped and unedited, the zero-count wrapper assertions prove
  `brain_passes.wgsl` no longer calls the four wrapper names, and the parity
  check reports the golden byte-identical, proving smell and vision match the
  pre-refactor trajectories for the production and whitening schedules.

## Non-goals

- No edit to `common.wgsl`, any cooperative fragment, `phase_death.wgsl`,
  `kernel_tick.wgsl`, or any file outside the six touches.
- No new sense, route, override, accessor, or config flag.
- No change to a float expression's operand order, a constant's value, or the
  steering freeze.
- No change to the golden or to any test body beyond its text anchors and the
  zero-count wrapper assertions.
