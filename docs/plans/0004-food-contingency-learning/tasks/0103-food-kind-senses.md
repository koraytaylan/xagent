---
id: food-kind-senses
title: "Kind odour in the senses pass and the extra sensory channels"
workstream: "0100"
kind: task
role: developer
depends_on:
  - food-kind-colour
  - second-odour-kernel
gated: false
touches:
  - crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl
  - crates/xagent-brain/tests/food_kind_senses.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_senses"
    timeout_ms: 3600000
  - name: scent-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test scent_pathway"
    timeout_ms: 3600000
  - name: sandbox-sensory-regression
    command: "cargo test -p xagent-sandbox --test sensory_genome"
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Kind odour in the senses pass and the extra sensory channels

## Outcome

With food kinds on, nostril pair 1 smells
only `First` items. The extra sensory channels that directly follow the scent
channels smell only `Second` items under the same odour law, and are written
whenever `EXTRA_SENSORY_CHANNELS` is at least `SCENT_CHANNELS`. Serial and
parallel pair-1 scent agree bit for bit. With kinds off and the second sense off,
scent and the sensory stride equal the baseline. Kind colour is task
`food-kind-colour`, which edits `phase_vision.wgsl` first.

## Scope

- `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl`:
  - `sense_scent` (lines 423-450): after the existing
    `if (food_flags[f] != 0u) { continue; }` line, add
    `if (food_kinds_enabled() && food_item_kind(f) != FOOD_KIND_FIRST) { continue; }`.
    Accumulation and the return expression are unchanged.
  - New `fn sense_second_odour(pos: vec3<f32>, facing: vec3<f32>, strength: f32) -> vec2<f32>`
    directly after `sense_scent`: the same nostril geometry, per-nostril
    contribution `exp(-sqrt(d_sq) / SCENT_DECAY_LENGTH)` inside `SCENT_RANGE` and
    law `vec2<f32>(1.0, 1.0) - exp(-strength * concentration)`, skipping eaten items
    and items whose `food_item_kind(f) != FOOD_KIND_SECOND`; it returns
    `vec2<f32>(0.0, 0.0)` before the loop when `food_kinds_enabled()` is false.
  - Smell block in `phase_vision_senses` (lines 402-413). Keep the line
    `let scent = sense_scent(pos, facing, smell_strength);` byte-identical and unique,
    and keep `sensory_buffer[scent_base] = scent.x;` and
    `sensory_buffer[scent_base + 1u] = scent.y;`. Immediately after the second of
    these, add:
    `if (EXTRA_SENSORY_CHANNELS >= SCENT_CHANNELS) { let second = sense_second_odour(pos, facing, smell_strength); sensory_buffer[scent_base + SCENT_CHANNELS] = second.x; sensory_buffer[scent_base + SCENT_CHANNELS + 1u] = second.y; }`
    Use one integer type in the comparison (`u32(SCENT_CHANNELS)` if needed) and no
    literal offset.
- `crates/xagent-brain/src/shaders/kernel/phase_vision_scent_parallel.wgsl`: in
  `vision_prepare_scent`, change line 56 from `if alive && food_flags[food] == 0u {`
  to `if alive && food_flags[food] == 0u && !(food_kinds_enabled() && food_item_kind(food) != FOOD_KIND_FIRST) {`.
- `crates/xagent-brain/tests/food_kind_senses.rs` (new file): three GPU tests, listed
  under Steps.

WGSL names used: `food_kinds_enabled()`, `food_item_kind(i)`, `FOOD_KIND_FIRST`,
`FOOD_KIND_SECOND`, `SCENT_CHANNELS` and `EXTRA_SENSORY_CHANNELS`.

## Non-goals

- No edit to `gpu_kernel.rs` (including `with_prepared_scent`), `common.wgsl`,
  `buffers.rs` or `agent/senses.rs`; a missing name is a stop-and-report. The scalar
  call text stays.
- No `second_odour` text or `senses.rs` use in WGSL: pair 2 sits at
  `scent_base + SCENT_CHANNELS` via `EXTRA_SENSORY_CHANNELS` (PLAN.md "Owner
  decisions on the 0003 interface", item 2).
- No read of `food_item_is_nourishing`; no channel encodes which kind nourishes.
- No CPU mirror, GUI or reward change; no planning vocabulary or `TODO` in source.

## Prerequisites

Helpers (food-kind-layout), `NON_VISUAL_COUNT` (baseline `buffers.rs`), the override
(extra-sensory-layout), `nourishing_food_kind()` (kernel-food-kind-draw), the
flag-built kernel (second-odour-kernel), the 0003 parity test (plan dependency).

## Decisions

- **Pair 1 filter sits inside the existing loops.** No new pass or buffer.
- **Pair 2 is serial and separate.** It is written from `phase_vision_senses`, not from
  the scalar-call redirect, so the redirect does not touch it.
- **Where pair 2 is written (checked in source).** Serial vision
  (`vision_tick.wgsl` and `cached_brain_vision_tick.wgsl`) calls
  `phase_vision_senses`, which holds the only scent write. Under
  `XAGENT_VISION_PARALLEL_SCENT=1`, `vision_prepare_scent` only prepares a value and
  `phase_vision_senses` still runs with its scalar call redirected, so pair 2 is
  written in both modes.
  - `phase_vision_parallel.wgsl` and `phase_vision_object_queries.wgsl` contain no scent
    write (confirm by search). If `cached_brain_vision_tick.wgsl` does not reach
    `phase_vision_senses`, stop and report; do not edit it.
- **Redirect interaction.** One vision variant rewrites `sensory_buffer[` to
  `sensory_next[` and asserts none remains (`gpu_kernel.rs` near line 1709); the
  pair-2 writes use `sensory_buffer[` too, so they are redirected alike.
- **Fail-closed GPU tests.** The helper `fn gpu_or_skip() -> bool` in the new test file
  returns `true` when `GpuKernel::is_available()`. When it is false and
  `std::env::var_os("XAGENT_REQUIRE_GPU").is_some()`, it panics with a message that
  names the missing adapter. Otherwise it prints a skip line and returns `false`. Every
  test starts with `if !gpu_or_skip() { return; }`.
- **Readback.** Sensory values come from `read_agent_telemetry_blocking(index)`
  (`gpu_kernel.rs` line 4005), which reads `sensory_buffer` at `layout.sensory_stride`.
  Physics state comes from `read_full_state_blocking()`. Add no readback helper. The
  expected odour is computed in the test from the Rust mirrors of
  `NOSTRIL_FORWARD_OFFSET`, `NOSTRIL_SIDE_OFFSET`, `SCENT_RANGE` and
  `SCENT_DECAY_LENGTH` (verify each name by search) under the shader's law.

## Steps

**Steps:**

1. In `phase_vision.wgsl` `sense_scent`, add the First-only `continue` after the
   food-flag check. In `phase_vision_scent_parallel.wgsl`, extend the per-slot guard
   in `vision_prepare_scent` as specified above. Do not change the scalar call text.
2. In `phase_vision.wgsl`, add `sense_second_odour` after `sense_scent` and,
   immediately after `sensory_buffer[scent_base + 1u] = scent.y;`, the guarded
   pair-2 write described under Scope. Use no literal offset.
3. Create `crates/xagent-brain/tests/food_kind_senses.rs` with the `gpu_or_skip`
   helper and `food_kind_senses_default_off_matches_baseline`: kinds off, second
   sense off; pair-1 scent equals the baseline odour law over all uneaten items
   within `1e-6` relative, and the sensory stride equals `BrainLayout::default()`'s.
4. Add `food_kind_senses_odour_split_by_kind`: kinds and the second sense on,
   kernel built through `GpuKernel::new`. Pair 1 equals the law over `First` items
   only; the two extra channels at sensory index `vision_count +
   crate::buffers::NON_VISUAL_COUNT` and that index + 1 equal the law over `Second`
   items only (`1e-6` relative). Then `reset_agents_seeded` with a different seed,
   same positions and tick count (re-upload positions if the reset moves them):
   `nourishing_food_kind()` changed and every sensory value is unchanged. Do not
   change the assertion to make it pass.
5. Add `food_kind_senses_parallel_scent_matches_serial`: kinds on, living agents
   only. Read the serial kernel first, then set `XAGENT_VISION_PARALLEL_SCENT=1`
   only around the parallel kernel's construction and restore the previous value.
   Pair-1 values are bit-identical (`to_bits()`) and the extra channels equal.

- **Done when:** the focused-proof check (the three `food_kind_senses` tests), the scent-regression, sandbox-sensory-regression and parity checks all exit 0 on the host, proving INV-04 (pair 1 smells only `First` and the extra channels right after scent smell only `Second` under the baseline law, and serial and parallel pair-1 values are bit-identical), INV-02 (the sensory values are unchanged by the redraw), INV-01 (with kinds and the second sense off, odour and stride equal the baseline and the parity golden is unchanged), and INV-10 (every new GPU test calls `gpu_or_skip` and panics without an adapter when `XAGENT_REQUIRE_GPU` is set).

## Risks

- Serial `sense_scent` has no alive guard while `vision_prepare_scent` zeroes dead
  agents; the parallel test uses living agents only and this stays unchanged.
- The env-var window is unlocked; asserted values are identical on either route.
