---
id: extra-sensory-layout
title: "Extra sensory channels and the second-odour block in the brain layout"
workstream: "0200"
kind: task
role: developer
depends_on:
  - food-kind-layout
  - second-odour-sense
gated: false
touches:
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib buffers"
    timeout_ms: 3600000
  - name: gradient-audit
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test gradient_parity_audit"
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Extra sensory channels and the second-odour block in the brain layout

## Outcome

A generic WGSL override `EXTRA_SENSORY_CHANNELS` (default `0u`) grows the
sensory and non-visual widths, and `BrainLayout` mirrors it. With
`second_odour_sense_enabled` on, the layout gains the two extra channels and
the third (`second_odour`) sense block; with it off every layout value equals
today's. See PLAN.md "Owner decisions on the 0003 interface" items 2 and 3,
and ARCHITECTURE.md INV-01, INV-05 and the `with_senses` row of "Untrusted
inputs".

## Scope

### `crates/xagent-brain/src/buffers.rs`

- Add `pub extra_sensory_channels: usize` to `BrainLayout`, doc: mirrors the WGSL
  `EXTRA_SENSORY_CHANNELS` override; 0 unless the second odour sense is declared.
- `sensory_stride` and `feature_count` both grow by `extra_sensory_channels`.
  `brain_stride` and `brain_scratch_stride` follow from `feature_count` by the
  existing formulas; do not change those formulas.
- Move the body of `with_retina_flagged` into a private helper that takes
  `extra_sensory_channels: usize` and adds it to `feature_count` and
  `sensory_stride`. `with_retina_flagged` and `with_retina` keep their
  signatures and pass `0`.
- `from_config(config)` becomes `Self::with_senses(config,
  crate::senses::senses_for_config(config))`, still reading `vision_width`,
  `vision_height`, `retina_width`, `retina_height`, `visual_cortex_enabled` and
  `danger_percept_enabled`.
- `with_senses(config, senses)` (the 0003 builder) sets
  `extra_sensory_channels = crate::senses::extra_sensory_channels(config)` and
  panics with a clear message when `config.second_odour_sense_enabled` differs
  from whether `senses` contains `crate::senses::SECOND_ODOUR` (compare by name).
- New Rust uses `u32::try_from(...)` or `checked_*` with `expect`; no new lossy
  `as` casts.

### `crates/xagent-brain/src/shaders/kernel/common.wgsl`

- Declare, before its first use, with a `///` comment saying the channels sit
  right after the two scent channels and are copied into features right after
  the scent features: `override EXTRA_SENSORY_CHANNELS: u32 = 0u;`.
- Change exactly two lines:
  `override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u + SCENT_CHANNELS + EXTRA_SENSORY_CHANNELS;`
  and
  `override NON_VISUAL_FEATURE_COUNT: u32 = 25u + SCENT_CHANNELS + EXTRA_SENSORY_CHANNELS + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;`.
- Generic name only: no smell- or vision-specific identifier and no
  `second_odour` text in WGSL.

### Tests in `buffers::tests` (names fixed)

- `extra_sensory_wgsl_mirror_matches_rust`: `common.wgsl` has the exact override
  line, its default parses to `0`, and the lines starting with
  `override SENSORY_STRIDE: u32 =` and `override NON_VISUAL_FEATURE_COUNT: u32 =`
  each contain `EXTRA_SENSORY_CHANNELS` (exact substring checks).
- `second_odour_off_layout_is_unchanged`: the default config and the config with
  `second_odour_sense_enabled = false` give equal `format!("{:?}", layout)` text
  (`BrainLayout` derives only `Clone, Debug`), which includes `sense_blocks()`,
  and `extra_sensory_channels` is 0.
- `second_odour_on_grows_sensory_and_feature_width_by_two`: with the flag on,
  `sensory_stride` and `feature_count` each grow by exactly 2 and `brain_stride`
  by exactly `2 * (ENCODED_DIMENSION + 1)` (derived from the formula in source).
  `sense_blocks().len()` is 3; the third block has `forward` and `critic` `None`
  and a footprint of mean 2 + covariance 3 + whitened 2 + turn 2 + trace 2 = 11
  slots, all inside `[end of the 0003 route block, sensory_mean_base)`. For the
  two base senses every field differs from the flag-off layout by the same
  constant shift, so no base slot moves relative to the tail.
- `with_senses_rejects_mismatched_table`: `#[should_panic]`; flag on, table
  `DECLARED_SENSES`.
- `second_odour_initial_state_is_zero_weights`: flag on, `init_brain_state_for`;
  the third block's turn weights and trace are `0.0` and its covariance diagonal
  equals the initial variance `SMELL` gets.

If the landed 0003 `sense_blocks()` field names differ, use the landed names and
report it; if they are missing, stop and report the file and line.

## Non-goals

- No change with the flag off; no edit to `gpu_kernel.rs` (the override value is
  emitted by `second-odour-kernel`), `senses.rs`, any shader other than
  `common.wgsl`, the parity test or its golden.
- No planning vocabulary or `TODO` in source (`contributing_guard.rs`).

Prerequisites: `senses.rs` with `SECOND_ODOUR`, `senses_for_config`,
`extra_sensory_channels` and `BrainConfig.second_odour_sense_enabled`
(second-odour-sense); `with_senses` and `sense_blocks()` from 0003; the
food-kind constants from food-kind-layout, which edits the same two files first.

## Steps

**Steps:**

1. In `buffers.rs`, add `extra_sensory_channels` and the private width helper,
   keeping `with_retina_flagged` and `with_retina` signatures.
2. Route `from_config` through `with_senses` and add the mismatch panic.
3. In `common.wgsl`, add the override and change the two override lines exactly.
4. Add the five `buffers::tests` tests above; keep every existing test unchanged.

- **Done when:** the checks `focused-proof`, `gradient-audit` and `parity` all pass on the host: `focused-proof` runs the five tests above green with every existing `buffers` test unchanged, `gradient-audit` stays green, and `parity` matches the unchanged golden. This proves INV-05 (with the flag on the layout adds exactly 2 to sensory and feature width, appends one Turn-only block after the 0003 route block without moving base slots relative to the tail, starts it with zero turn weights and trace, and rejects a mismatched table) and INV-01 (with the flag off `EXTRA_SENSORY_CHANNELS` defaults to 0 and every `BrainLayout` value equals the layout without the new sense).
