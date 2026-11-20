---
id: second-odour-features
title: "Extra sensory channels copied into the brain features"
workstream: "0200"
kind: task
role: developer
depends_on:
  - food-kind-senses
gated: false
touches:
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl
  - crates/xagent-brain/tests/second_odour_features.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test second_odour_features"
    timeout_ms: 3600000
  - name: scent-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test scent_pathway"
    timeout_ms: 3600000
  - name: gradient-audit
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test gradient_parity_audit"
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Extra sensory channels copied into the brain features

## Outcome

Both feature extractors copy the `EXTRA_SENSORY_CHANNELS` sensory values that
follow the scent channels into the features directly after the two scent
features and before the danger percept. With the second odour sense on, the
`second_odour` block (input `NonVisualFeatures { offset: 27 }`) therefore reads
the second odour; with it off (`EXTRA_SENSORY_CHANNELS = 0`) the loop runs zero
times and every feature is unchanged (ARCHITECTURE.md INV-05, INV-01).

## Scope

- `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, in
  `coop_feature_extract`: directly after
  `s_features[fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;`
  add
  `for (var k: u32 = 0u; k < EXTRA_SENSORY_CHANNELS; k = k + 1u) { s_features[fi] = sensory_buffer[s_base + scent_offset + NON_VISUAL_FEATURE_CHANNELS + k]; fi = fi + 1u; }`.
  `NON_VISUAL_FEATURE_CHANNELS` (`common.wgsl`, `2u`) is 0003's rename of the
  WGSL `SCENT_CHANNELS`; do not write or define `SCENT_CHANNELS` in WGSL (0003's
  `SENSE_SPECIFIC_NAMES` bans it). The Rust `buffers::SCENT_CHANNELS` stays
  usable in Rust tests.
- `crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl`: the same
  loop directly after
  `brain_scratch[agent_base + SCRATCH_FEATURES + fi] = sensory_buffer[s_base + scent_offset + 1u]; fi = fi + 1u;`,
  writing `brain_scratch[agent_base + SCRATCH_FEATURES + fi]`.
- The danger block that follows keeps its text; it now starts at the shifted
  `fi`, which matches `NON_VISUAL_FEATURE_COUNT` (extra-sensory-layout).
- `crates/xagent-brain/tests/second_odour_features.rs` (new): the tests under
  Steps, with the fail-closed `gpu_or_skip` helper of `tests/food_kind_senses.rs`
  (copied; test files share no module).

If 0003 landed different text at these two sites, insert the loop directly
after the landed scent-feature writes in each extractor and report it; if a
third feature extractor exists, stop and report its file and line.

## Decisions

- **Generic only.** The loop names `EXTRA_SENSORY_CHANNELS` and `NON_VISUAL_FEATURE_CHANNELS`
  (no literal scent width);
  no `second_odour` text, no new binding, no sense-specific block code.
- **Readback.** Only existing APIs: `dispatch_feature_phase_for_test`,
  `read_brain_scratch_blocking`, `read_agent_telemetry_blocking`,
  `read_agent_state` and `brain_layout()` (second-odour-kernel). Add no helper to
  `gpu_kernel.rs`.

## Non-goals

- No edit to `gpu_kernel.rs`, `common.wgsl`, `buffers.rs`, `senses.rs`, the vision
  shaders or the 0003 block code.
- No planning vocabulary or `TODO` in source (`contributing_guard.rs`).

## Steps

**Steps:**

1. Add the copy loop in `coop_feature_extract` (`brain_passes.wgsl`).
2. Add the same loop in `phase_brain_features.wgsl`.
3. Create `tests/second_odour_features.rs` with `gpu_or_skip` and
   `second_odour_scratch_features_follow_scent` (the `phase_brain_features.wgsl`
   route): kinds and the second sense on, kernel from `GpuKernel::new`, a `Second`
   item in range; after a tick, `dispatch_feature_phase_for_test()` and
   `read_brain_scratch_blocking()` give, per agent at `SCRATCH_FEATURES`, features
   27 and 28 bit-equal to the two extra sensory channels and 25 and 26 bit-equal to
   the scent channels (sensory from `read_agent_telemetry_blocking`).
4. Add `second_odour_block_reads_fused_features` (the `coop_feature_extract`
   route): same setup run for a few brain ticks; the third sense block's
   `whitened_input` slots (`brain_layout().sense_blocks()[2]`, read from
   `read_agent_state`) are nonzero. With every in-range item `First` (no second
   odour), they stay exactly `0.0`.
5. Add `second_odour_features_off_leave_features_unchanged`: flag off, the scratch
   feature vector after the same ticks has length
   `BrainLayout::default().feature_count` and is bit-identical across two kernels
   of the same seed and config.

- **Done when:** the focused-proof check passes with `second_odour_scratch_features_follow_scent`, `second_odour_block_reads_fused_features` and `second_odour_features_off_leave_features_unchanged` ok, and scent-regression, gradient-audit and parity pass unchanged, proving INV-05 (both feature extractors copy the two extra channels to features 27 and 28) and INV-01 (with the flag off the features and the parity golden are unchanged), with INV-10 held by the fail-closed helper under `XAGENT_REQUIRE_GPU=1`.
