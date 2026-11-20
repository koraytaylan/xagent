---
id: generic-schedule-candidates
title: "Alternate brain schedules over every Jacobi sense"
workstream: "0200"
kind: task
role: developer
depends_on:
  - generic-sense-block-sites
gated: false
status: done
merged_as: "c3e8a130a96a847398e6d337d54e1d394e1cf89b"
touches:
  - crates/xagent-brain/src/gpu_kernel/visual_pathway_validation.rs
  - crates/xagent-brain/src/gpu_kernel/cooperative_jacobi_validation.rs
  - crates/xagent-brain/src/shaders/kernel/brain_cooperative_visual_pathway.wgsl
  - crates/xagent-brain/src/shaders/kernel/brain_cooperative_jacobi.wgsl
  - crates/xagent-brain/src/shaders/kernel/whitening_refresh.wgsl
  - crates/xagent-brain/src/shaders/kernel/brain_cooperative_whitening.wgsl
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
  - name: alternate-schedules
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib -- --include-ignored --exact gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_complete_state gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_phase_ordering gpu_kernel::visual_pathway_validation::cooperative_visual_pathway_preserves_cortex_skip gpu_kernel::whitening_validation::isolated_whitening_matches_complete_serial_state gpu_kernel::whitening_validation::isolated_whitening_with_interleaved_recall_matches_complete_serial_state gpu_kernel::whitening_storage_validation::workgroup_whitening_matches_complete_serial_state gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_complete_state gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_covariance_edges_and_skipped_refresh gpu_kernel::cooperative_jacobi_validation::cooperative_jacobi_preserves_cortex_skip gpu_kernel::cooperative_whitening_validation::cooperative_whitening_matches_complete_serial_state gpu_kernel::cooperative_whitening_validation::cooperative_brain_preserves_cortex_and_odd_vision_layouts"
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

# 0207 — Alternate brain schedules over every Jacobi sense

## Outcome

The four alternate-schedule fragments (cooperative whitening, the production
fragment composed by `compose_brain_passes` when `XAGENT_BRAIN_COOPERATIVE_WHITENING=1`;
cooperative Jacobi; visual-pathway parallelization; isolated `whitening_refresh`)
refresh or prepare every `Jacobi` sense through the sense-indexed helpers and
`SENSE{k}_*` overrides from `generic-sense-helpers` and `sense-override-constants`,
not through vision slot constants, and none of them references the hand-written
names listed below. The two composition tests that `generic-sense-block-sites`
skipped run unskipped. Test names and assertion counts are unchanged.

## Decisions

- Keep the name and call text `cooperative_refresh_vision_whitening(brain_base, tid)`
  and `whitening_refresh` (accepted naming exception in ARCHITECTURE.md). Only
  bodies change; the body is generic over `Jacobi` senses.
- Refresh loop: `for (var k = 0u; k < SENSE_COUNT; k++)` in declared order. Skip any
  sense whose `sense_whitening_kind(k)` is not `SENSE_WHITENING_KIND_JACOBI`. Senses
  run one after another, each closed by its barriers, so shared workgroup scratch
  is reused serially.
- Accessors: use only names that `generic-sense-block-sites` and
  `generic-sense-helpers` landed in `common.wgsl`; add none. The names fixed by
  `generic-sense-block-sites` are `sense_input_kind`, `sense_input_offset`,
  `sense_whitening_kind`, `sense_refresh_ticks`, `sense_adaptation_rate`,
  `sense_input` and `sense_refresh_jacobi_whitening(brain_base, sense)`. For channel
  count, mean, covariance, whitening matrix, turn weights, turn trace and
  learning-rate gene, use the sense-indexed accessor that `generic-sense-helpers`
  declares for that field (read `common.wgsl` at dispatch).
- `sense_input_offset(k)` is the non-visual feature offset and is `0` for hemifield
  input. Cooperative candidates support only Jacobi senses whose input kind is
  `SENSE_INPUT_KIND_VISUAL_HEMIFIELDS`. `cooperative_jacobi_validation.rs` asserts
  this over `DECLARED_SENSES`. The cooperative cortex gate stays as today.
- Matrix stride is the sense's channel count `n`. The eigenvector base stays at 64
  and the Jacobi flags at 128 onward, so vision (n = 8) keeps today's indices.
- Cadence: refresh when `brain_state[brain_base + O_TICK_COUNT]` (as `u32`)
  `% sense_refresh_ticks(k) == 0u`. A `VisualHemifields` sense is not refreshed while
  the visual cortex is on, as the vision gate does today.
- Visual-pathway prepare runs for each Jacobi hemifield sense. Its gather, in the
  fragment and in the test's probe prepare, is a direct `vision_hemifields()`
  call (`let raw = vision_hemifields();`). `sense_hemifield_input()` is the
  serial block's wrapper in `brain_passes.wgsl` that returns `vision_hemifields()`;
  its serial-block call stays as is and is not a finding against this task.
- Arithmetic: every float expression keeps its operand order, because the
  validations compare full state bit for bit. Keep the expression strings the
  composition tests search for, changing only operand names to accessors.
  `CANDIDATE.matches("workgroupUniformLoad(")` stays at 2.
- After this task, the four fragment files (`brain_cooperative_visual_pathway.wgsl`,
  `brain_cooperative_jacobi.wgsl`, `whitening_refresh.wgsl`,
  `brain_cooperative_whitening.wgsl`) contain none of: `O_SCENT_`, `O_VISION_PATHWAY_`,
  `O_VISION_TURN_WEIGHTS`, `O_TRACE_SCENT`, `O_TRACE_VISION`, `whiten_scent`,
  `refresh_vision_whitening`, `vision_pathway_learning_rate`,
  `settle_turn_pathways_at_death`, `VISION_WHITENING_REFRESH`, `VISION_PATHWAY_RATE`.

**Steps:**

1. **Production whitening.** In `brain_cooperative_whitening.wgsl`, rewrite the body
   of `cooperative_refresh_vision_whitening` as the per-sense loop above. Remove
   `VISION_WHITENING_REFRESH`, `VISION_PATHWAY_INPUTS` and `O_VISION_PATHWAY_*` uses
   from the file. Keep the helper index functions and the workgroup scratch use.
2. **Isolated refresh.** In `whitening_refresh.wgsl`, the `whitening_refresh` entry
   keeps its alive guard and calls `sense_refresh_jacobi_whitening(brain_base, k)`
   for each Jacobi sense, gated by the cadence and cortex rules, instead of the
   vision wrapper.
3. **Cooperative Jacobi.** In `brain_cooperative_jacobi.wgsl`, apply the same loop to
   the rotation code. Keep both `workgroupUniformLoad(` calls, the flag indices, the
   two-barrier rotation structure and the six rotation expressions verbatim. Keep
   every constant and helper inside this file, because the test splices the whole
   file over the refresh function.
4. **Visual-pathway fragment.** In `brain_cooperative_visual_pathway.wgsl`, replace
   `cooperative_prepare_vision_pathway` with `cooperative_prepare_jacobi_senses(brain_base, tid)`,
   which runs centre, whiten, store-input and mean/covariance update per Jacobi
   hemifield sense through the accessors. Replace `prepared_vision_pathway_turn` with
   `prepared_sense_turn(brain_base, sense, cortex_on)`, summing channels in ascending
   order as today.
5. **Validation anchors and assertions.** In `visual_pathway_validation.rs` and
   `cooperative_jacobi_validation.rs`, update `REFRESH_CALL`, `ORIGINAL_CALL`,
   `PROBE_ENTRY`, `probe_entry`, the expression lists and the renamed helper calls to
   the exact text that `generic-sense-block-sites` left in `brain_passes.wgsl` and
   `gpu_kernel.rs`, read at dispatch. Keep every existing assertion count, test name
   and body. Add two non-ignored tests: `candidate_fragments_drop_sense_specific_names`
   in `visual_pathway_validation.rs` loads the four fragment files as the module
   already loads fragments and asserts zero occurrences of each forbidden token
   listed in Decisions; `candidate_jacobi_senses_are_hemifield_only` in
   `cooperative_jacobi_validation.rs` asserts every `Jacobi` entry of
   `DECLARED_SENSES` has `VisualHemifields` input.
6. **Hygiene.** No TODO, plan or task names in comments; no `as` cast in Rust; no new
   `var<workgroup>` declaration in any fragment; no edit outside the six touches.

- **Done when:** INV-09 and INV-01 hold: the focused-proof check (whole xagent-brain
  lib, no skipped test, including the two new source tests) and the
  alternate-schedules check pass with unchanged assertion counts, the parity check
  reports the golden byte-identical, and the fmt, clippy and guard checks pass,
  proving every alternate schedule stays complete-state identical to the serial
  block for each Jacobi sense (INV-09) and default seeded trajectories are unchanged
  (INV-01).

## Non-goals

- No edit to `brain_passes.wgsl`, `common.wgsl`, `gpu_kernel.rs`, the context gather
  or brain-section modules, or any other validation module.
- No rename, new accessor, sense, route, override or flag; no constant value change.
- No change to the golden or to a test body beyond text anchors.
