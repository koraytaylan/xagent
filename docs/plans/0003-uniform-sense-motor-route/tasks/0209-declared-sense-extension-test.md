---
id: declared-sense-extension-test
title: "An extra declared sense needs no new code path"
workstream: "0200"
kind: task
role: developer
depends_on:
  - remove-sense-specific-wgsl
gated: false
status: done
merged_as: "6377a86dcb28845fe5fc803d8b9e7c77d286b5e8"
touches:
  - crates/xagent-brain/tests/sense_block_extension.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test sense_block_extension -- --include-ignored"
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

# 0209 — An extra declared sense needs no new code path

## Outcome

`crates/xagent-brain/tests/sense_block_extension.rs` declares one extra sense,
PROBE, only inside the test and proves the generic block carries it with no
WGSL or Rust edit outside the test. The production `DECLARED_SENSES` table
stays `[SMELL, VISION]`; EXTENDED is the test-local set `[SMELL, VISION, PROBE]`.

## Decisions

- PROBE: `SenseInput::NonVisualFeatures { offset: ENERGY_DELTA_FEATURE }` over
  2 contiguous channels. No `buffers.rs` constant names these indices, and this
  task adds none; the test defines them locally with doc comments giving the
  documented non-visual feature order (`NON_VISUAL_FEATURE_COUNT` = 27 in
  `buffers.rs`, the `common.wgsl` comment): speed (1) at 0, facing (3) at 1..=3,
  angular velocity (1) at 4, energy (1) at 5, integrity (1) at 6, energy delta
  (1) at 7, integrity delta (1) at 8, touch (16) at 9..=24, scent (2) at 25..=26.
  The local constants are `ENERGY_DELTA_FEATURE = 7`, `INTEGRITY_DELTA_FEATURE = 8`
  and `SCENT_FEATURE = 25` (equal to 1+3+1+1+1+2+16). The test asserts
  `INTEGRITY_DELTA_FEATURE == ENERGY_DELTA_FEATURE + 1`, and that the smell
  descriptor's `NonVisualFeatures` offset equals `SCENT_FEATURE`, so a layout
  change fails the test.
- PROBE uses `Whitening::ClosedForm2`, `LearningRate::Actor`, one
  `MotorRoute::Turn` route with `Heritability::LearnedOnly`.
- The extended layout is built with `BrainLayout::with_senses` and
  `sense_blocks()` from `derived-sense-layout`; the kernel is built with
  `GpuKernel::new_with_senses` from `sense-override-constants`. Descriptor
  types come from `sense-descriptor`. Report any missing API as a defect.
- CPU layout test: the EXTENDED stride is the default stride plus exactly 11.
  No smell, vision or route-block offset moves. The probe's first slot is
  `O_TRACE_VISION + 8 + PATHWAY_ROUTE_SLOT_COUNT`, and `O_SENSORY_MEAN` moves
  by exactly 11.
- GPU test (`#[ignore]`, fails without an adapter): 4 agents, built with
  `new_with_senses`. At birth the probe's weights and traces are exactly `0.0`.
  After a run where agents eat and take hazard damage (a locally written small
  max energy is allowed so deltas vary), the probe's whitened inputs are finite
  and not all zero, its turn weights are finite, inside `MAX_WEIGHT_NORM` and not
  all zero, and the smell and vision whitened inputs are non-zero.
- Never weaken an assertion. Do not edit WGSL, `gpu_kernel.rs`, `buffers.rs` or
  any other file.

**Steps:**

1. **Layout test.** Define PROBE and EXTENDED, and the test-local
   `ENERGY_DELTA_FEATURE`, `INTEGRITY_DELTA_FEATURE` and `SCENT_FEATURE` constants
   with the doc comments in Decisions, in the test file. Add
   `probe_layout_offsets_append_without_moving_existing_slots`, asserting the
   stride, `O_SENSORY_MEAN`, probe offset values and the feature-index relations
   listed in Decisions.
2. **Birth test.** Add `probe_sense_starts_at_zero_on_gpu` (`#[ignore]`):
   build the 4-agent kernel with `new_with_senses`, read each agent's state with
   `read_agent_state`, and assert probe weights and traces are exactly `0.0`.
3. **Run helper.** In the same file, add a helper that resets seeded agents with
   `reset_agents_seeded`, steps the kernel so agents eat and take hazard damage,
   and returns the states. Use no other helper or file.
4. **Learning assertions.** Add `probe_sense_learns_through_generic_block`
   (`#[ignore]`) asserting the finite, non-zero, in-ball and non-all-zero
   conditions listed in Decisions, and the smell and vision whitened non-zero
   conditions.
5. **Sanity.** Confirm the test file uses only the public APIs named above and
   that no assertion is removed or loosened.
6. **Hygiene.** No `as` casts in Rust, no TODO or plan/task names in comments,
   no edit outside the one touch.

- **Done when:** INV-02, INV-03 and INV-05 hold: `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test sense_block_extension -- --include-ignored` passes its `probe_layout_offsets_append_without_moving_existing_slots`, `probe_sense_starts_at_zero_on_gpu` and `probe_sense_learns_through_generic_block` tests, proving a newly declared sense appends, starts at zero and learns with no WGSL or Rust edit.

## Non-goals

- No change to `DECLARED_SENSES`, production configs, presets or the golden.
- No edit to `senses.rs`, `buffers.rs`, `gpu_kernel.rs`, WGSL files or the
  existing `vision_pathway.rs` test.
- No new world condition, food kind or production sense.
