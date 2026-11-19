---
id: remove-sense-specific-wgsl
title: "Remove the sense-specific WGSL names"
workstream: "0200"
kind: task
role: developer
depends_on:
  - sense-descriptor
  - sense-override-constants
  - generic-schedule-candidates
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/src/senses.rs
  - crates/xagent-brain/README.md
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- sense_specific"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
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

# 0208 — Remove the sense-specific WGSL names

## Outcome

`common.wgsl` and `brain_passes.wgsl` contain none of the removed identifiers
below (whole-identifier match, comments included). Per-sense parameters come
only from the generated `SENSE{k}_*` overrides and the sense-indexed helpers.
Default behaviour, the parity golden and the pathway tests are unchanged.

## Decisions

- Removed WGSL constants (no shader reads them after the refactor):
  `SCENT_FEATURE_OFFSET`, `SCENT_WHITENING_RATE`, `SCENT_INITIAL_VARIANCE`,
  `VISION_PATHWAY_RATE`, `VISION_WHITENING_REFRESH`,
  `VISION_PATHWAY_INITIAL_VARIANCE`.
- Removed functions and overrides: `whiten_scent`, `refresh_vision_whitening`,
  `vision_pathway_learning_rate`, `settle_turn_pathways_at_death`,
  `vision_pathway_step`; and the identifiers `O_SCENT_MEAN`, `O_SCENT_COVARIANCE`,
  `O_SCENT_WHITENED`, `O_SCENT_TURN_WEIGHTS`, `O_TRACE_SCENT`,
  `O_VISION_PATHWAY_MEAN`, `O_VISION_PATHWAY_COVARIANCE`,
  `O_VISION_PATHWAY_WHITENING`, `O_VISION_PATHWAY_INPUT`,
  `O_VISION_TURN_WEIGHTS`, `O_TRACE_VISION`.
- Still-read constants renamed to kind names; each is read only inside the two
  touched WGSL files (confirmed in step 1): `SCENT_CHANNELS` becomes
  `NON_VISUAL_FEATURE_CHANNELS`; `SCENT_EIGEN_FLOOR` becomes
  `CLOSED_FORM2_EIGEN_FLOOR`; `VISION_PLASTICITY_MIN` and `_MAX` become
  `PLASTICITY_GENE_MIN` and `_MAX`; `VISION_PATHWAY_CHANNELS` becomes
  `HEMIFIELD_CHANNELS`.
- Kept, not removed (read by files outside these touches, or a Rust-public
  slot name): `VISION_PATHWAY_INPUTS` (read by `brain_cooperative_whitening.wgsl`,
  `brain_cooperative_jacobi.wgsl`, `brain_cooperative_visual_pathway.wgsl`);
  `SCENT_DECAY_LENGTH` and `SCENT_RANGE` (read by `phase_vision.wgsl` and
  `phase_vision_scent_parallel.wgsl`); `O_VISION_PLASTICITY` and
  `O_SMELL_STRENGTH` (Rust-public slots); `vision_hemifields` (anchor of the
  pooling checks); the sensory-grid `VISION_*` overrides. The source test
  forbids only the removed list above.
- `vision_pathway_step` is renamed `visual_hemifields_input_step` (the input
  kind `SenseInput::VisualHemifields`); body and float expression order are
  unchanged.
- `cooperative_refresh_vision_whitening` is the accepted naming exception. It
  does not match the removed name `refresh_vision_whitening` because matching is
  whole-identifier.
- Stop rule: a removed name, or a constant renamed away, still referenced outside
  the five touches (for example a Rust splice anchor in `gpu_kernel/*.rs` or a
  WGSL fragment) gets no edit there. Stop and report the file and line.

**Steps:**

1. **Inventory.** With search tools, list every whole-identifier hit of the
   removed names and renamed constants under `crates/xagent-brain/src`. Hits
   outside the touches, other than `buffers.rs` public constants and comments,
   trigger the stop rule. Confirm each renamed constant's readers are only in
   `common.wgsl` and `brain_passes.wgsl`; if not, stop and report.
2. **common.wgsl.** Delete any remaining copy of the four wrapper functions,
   the removed constants and the hand-written `override O_SCENT_*`,
   `O_VISION_PATHWAY_*`, `O_VISION_TURN_WEIGHTS`, `O_TRACE_SCENT` and
   `O_TRACE_VISION` declarations, plus comments naming them. Apply the renames
   above, including comments. Declare `override O_SENSORY_MEAN: u32;` with no
   initializer, so its value comes only from the generated override map.
3. **brain_passes.wgsl.** Make sure no smell or vision call site remains (the
   sites already loop over senses from `generic-sense-block-sites`). Replace any
   remaining removed-name reference with the generic helper of the same sense
   index. Rename `vision_pathway_step` to `visual_hemifields_input_step` and
   `VISION_PATHWAY_CHANNELS` to `HEMIFIELD_CHANNELS`. Leave no removed name in
   the file.
4. **gpu_kernel.rs tests.** Add `sense_specific_wgsl_has_no_sense_names`: for each
   removed name (the constants, functions and identifiers above, plus the
   renamed-away `SCENT_CHANNELS`, `SCENT_EIGEN_FLOOR`, `VISION_PLASTICITY_MIN`,
   `VISION_PLASTICITY_MAX`, `VISION_PATHWAY_CHANNELS`), assert that
   `BRAIN_PASSES_SRC` and `COMMON_SRC` have no occurrence whose preceding and
   following characters are not in `[A-Za-z0-9_]`. Also assert the new names are
   present. Add `sense_specific_sensory_mean_matches_layout`: for the default
   layout and the extra-sense layout used by the `sense-override-constants`
   tests, the override map's `O_SENSORY_MEAN` equals the layout's
   `O_SENSORY_MEAN`, and `COMMON_SRC` contains `override O_SENSORY_MEAN: u32;`.
   Use no `as` casts.
5. **senses.rs mirror test.** Replace the body of the existing mirror test from
   `sense-descriptor` with a single-source check that parses `common.wgsl` via
   `include_str!("shaders/kernel/common.wgsl")`. For each index `k` of
   `DECLARED_SENSES`, every `override SENSE{k}_<FIELD>: <ty> = <literal>;`
   default must equal the matching field of `DECLARED_SENSES[k]`. The test fails
   if a declared sense has no override or a literal differs. Name it
   `sense_specific_constants_single_source` (rename if the old name differs), so
   the focused check selects it. Rewrite doc comments that said "mirrors
   common.wgsl" so `senses.rs` is the source and the WGSL defaults mirror it. If
   the `SENSE{k}_<FIELD>` names from
   `sense-override-constants` do not match the descriptor fields, stop and
   report.
6. **README.** In `crates/xagent-brain/README.md`, rewrite the `Brain State
   Buffer` heading and bullets and the TD-credit section `6.6.3`. Describe the
   descriptor-driven block: slots from `DECLARED_SENSES` through `BrainLayout`,
   and one generic credit and settle path. Remove any plan or task reference.
   Keep Rust constant names.

- **Done when:** INV-05 and INV-01 hold: the focused-proof check passes its
  `sense_specific` tests (re-adding any removed name fails the first one, and a
  changed `SENSE{k}_*` default fails the senses.rs test), the brain-lib and
  pathway-tests checks pass, the parity check reports the golden byte-identical,
  and the fmt, clippy and guard checks pass.

## Non-goals

- No new sense, route, input kind, config flag or override value; no change to
  a constant's value or to `BRAIN_STRIDE`.
- No edit to `buffers.rs`, the validation modules, `phase_vision*.wgsl` or the
  other `.wgsl` fragments.
- No change to the golden, to existing test bodies (except the senses.rs mirror
  test named above), or to test names.
- No TODO, plan or task names in comments; no `as` casts in Rust.
