---
id: derived-sense-layout
title: "Brain-state sense offsets derived from the descriptor"
workstream: "0200"
kind: task
role: developer
depends_on:
  - sense-descriptor
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/buffers.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- buffers::tests::"
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

# 0202 — Brain-state sense offsets derived from the descriptor

## Outcome

In `crates/xagent-brain/src/buffers.rs`, every per-sense brain-state slot offset
is computed by one `const fn` rule over `crate::senses::DECLARED_SENSES` (the
table from `sense-descriptor`) and exposed through `BrainLayout`. Values do not
change: default `BRAIN_STRIDE`, `FIXED_TAIL_SIZE`, every public offset constant
and every stored blob stay as they are. No WGSL, config, `senses.rs` or
`gpu_kernel.rs` edit.

## Baseline facts

- Cascade from `O_SCENT_MEAN = O_RECENT_NORMS + RECENT_CAP`: smell 11 slots
  (mean 2, covariance 3, whitened 2, turn 2, trace 2); vision 160 slots (mean 8,
  covariance 64, whitening 64, input 8, turn 8, trace 8); then `O_SENSORY_MEAN`.
- Plan 0002's route block sits between `O_TRACE_VISION + VISION_PATHWAY_INPUTS`
  and `O_SENSORY_MEAN`: `PATHWAY_INPUT_COUNT` (10), the six `O_*PATHWAY*`
  constants and `PATHWAY_ROUTE_SLOT_COUNT` (1321). Its joint inputs are smell
  (0..2) then vision (2..10).
- `init_brain_state_for` writes scent covariance diagonal (packed indices 0 and
  2), vision covariance diagonal (`VISION_PATHWAY_INITIAL_VARIANCE`) and vision
  whitening diagonal (`1/sqrt` of that). It draws no RNG for senses.
- `upgrade_legacy_brain_state` (plan 0002) is in `buffers.rs`, so `buffers::tests`
  reaches it. If it is absent at task start, stop as BLOCKED.

## Decisions

- Rule: a `const fn` over a `&[SenseDescriptor]`. The cursor starts at
  `SENSE_BASE = O_RECENT_NORMS + RECENT_CAP`. Per sense, in declared order:
  mean (`channels`), covariance (3 for `ClosedForm2`, `channels²` for `Jacobi`),
  whitening (`channels²`, `Jacobi` only), whitened input (`channels`), turn
  weights (`channels`), turn trace (`channels`). All offsets are absolute indices
  into `brain_state`.
- The first `DECLARED_SENSES.len()` entries are base senses. After them the
  route block starts at the cursor and keeps plan 0002's constant values. A
  sense's forward/critic weights and traces are `route base + input_start +
  k` for its channel slice, where `input_start` is the summed channels of earlier
  base senses. They are `Some` only for routes the descriptor declares.
- Entries after the base senses append after the route block in declared order.
  They may declare only `Turn` (assert otherwise), so their forward/critic
  fields are `None`. The sense region ends at `sensory_mean_base`.
- `SenseBlockOffsets` derives `Clone, Copy, Debug, PartialEq` with fields
  `channels`, `mean`, `covariance`, `whitening: Option<usize>`, `whitened_input`,
  `turn_weights`, `turn_trace`, `forward_weights`, `forward_trace`,
  `critic_weights`, `critic_trace` (the four route fields `Option<usize>`).
- Storage: `BrainLayout` holds `[SenseBlockOffsets; MAX_SENSE_SLOTS]`, a count,
  `sensory_mean_base` and the `&'static [SenseDescriptor]` table it was built
  from, with no heap allocation. `sense_blocks()` returns the filled prefix;
  `senses()` returns the table in declared order. `brain_stride` adds the
  extra-sense region (`sensory_mean_base − (route base + PATHWAY_ROUTE_SLOT_COUNT)`), which is 0 for
  the declared table. `with_senses(config, senses: &'static [SenseDescriptor])`
  is the builder; `from_config` passes `DECLARED_SENSES`; `with_retina` and
  `with_retina_flagged` keep their signatures.
- Public names stay, and each value is read from the default table computed by
  the same `const fn`: `O_SCENT_*`, `O_VISION_PATHWAY_*`, `O_VISION_TURN_WEIGHTS`,
  `O_TRACE_SCENT`, `O_TRACE_VISION`, the route constants, `O_SENSORY_MEAN`,
  `BRAIN_STRIDE`, `FIXED_TAIL_SIZE`. `PATHWAY_INPUT_COUNT` and
  `PATHWAY_ROUTE_SLOT_COUNT` keep their values. `feature_count_for_brain_stride`
  and `fixed_tail_base` keep their code and values.
- `init_brain_state_for` loops `layout.sense_blocks()`: `ClosedForm2` writes
  variance at covariance indices 0 and 2; `Jacobi` writes variance on the
  covariance diagonal and `1/sqrt` on the whitening diagonal. Same values, slots,
  no RNG use.
- Doc comments name no plan, task or decision document and contain no `TODO` or
  `FIXME` (guard check). Every new item has a `///` comment.

**Steps:**

1. **Offset rule.** Add `SENSE_BASE`, `SenseBlockOffsets`, and a `const fn` that
   takes `&[SenseDescriptor]` and returns the filled block array, its count, the
   route base and `sensory_mean_base`. Add a `const` assertion that the count is at
   most `MAX_SENSE_SLOTS`, and a `panic!` if an extra sense declares a route other
   than `Turn`.
2. **Default table and public constants.** Compute `DEFAULT_SENSE_LAYOUT` from
   `DECLARED_SENSES`. Redefine every listed public offset, `BRAIN_STRIDE` and
   `FIXED_TAIL_SIZE` as reads of that table, keeping names and values. Delete the
   hand-written cascade lines.
3. **BrainLayout.** Add the table fields, `sense_blocks()`, `senses()`,
   `with_senses`, and the extra-region term in the `brain_stride` formula. Route `from_config`,
   `with_retina` and `with_retina_flagged` through the builder with
   `DECLARED_SENSES`.
4. **Initial variances.** Replace the scent and vision variance writes in
   `init_brain_state_for` with the loop over `sense_blocks()`, as in the decisions.
5. **Tests.** In `buffers::tests`, add `derived_offsets_match_pre_plan_cascade`
   (literal `+2,+3,+2,+2,+2,+8,+64,+64,+8,+8,+8` sums and route constants),
   `default_stride_and_tail_match_pre_plan_formula` (`BRAIN_STRIDE` is
   `O_RECENT_NORMS + RECENT_CAP + 171 + PATHWAY_ROUTE_SLOT_COUNT + FEATURE_COUNT`,
   `FIXED_TAIL_SIZE` is `171 + PATHWAY_ROUTE_SLOT_COUNT`),
   `extra_sense_grows_stride_by_eleven` (declared senses plus a test-only 2-channel
   `ClosedForm2` `Turn`/`LearnedOnly` sense: stride +11, no base or route offset
   moves), `route_slices_split_joint_inputs` (test-only table with all three routes
   on both senses, so verdicts do not matter: smell slices at 0, vision at 2),
   `baseline_length_legacy_blob_upgrades` (length `BRAIN_STRIDE −
   PATHWAY_ROUTE_SLOT_COUNT` upgrades to `BRAIN_STRIDE`, route slots zero, old
   values kept), `init_writes_default_sense_slots` (same variance and whitening
   diagonals at the same indices), `default_layout_senses_match_declared_table`
   (`BrainLayout::default().senses()` equals `DECLARED_SENSES` by name and order).
6. **Hygiene.** Confirm `buffers.rs` has no `TODO`, `FIXME`, plan, task or decision
   name in comments, that every new item has a `///` comment, and that no `as`
   cast or magic number lacks a named constant.

- **Done when:** INV-02 and INV-01 — the focused-proof check passes with the seven
  tests named in step 5 beside the existing `buffers::tests` (no test changed or
  skipped), and the parity check passes with the existing golden unchanged and no
  re-record, proving INV-02 (derived offsets, stride and tail unchanged, a
  declared extra sense moves no slot) and INV-01 (default seeded trajectories
  byte-identical).

## Non-goals

- No WGSL, `common.wgsl`, `gpu_kernel.rs`, `senses.rs`, `config.rs` or
  `governor.rs` edit; the override generator and blob decode keep their code.
- No new route, sense or input kind in production; the extra sense exists only
  in a test table.
- No change to `feature_count_for_brain_stride` or `fixed_tail_base` behaviour for
  the default layout, and no change to RNG draw order or the variance values.
