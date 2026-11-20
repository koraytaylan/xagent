---
id: second-odour-kernel
title: "Build the kernel's sense table and extra sensory override from the second-odour flag"
workstream: "0200"
kind: task
role: developer
depends_on:
  - extra-sensory-layout
  - kernel-food-kind-draw
gated: false
touches:
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/tests/second_odour_kernel.rs
status: done
merged_as: "e83c8089068feba8cbf4e5ce53b8908bf9abd390"
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test second_odour_kernel"
    timeout_ms: 3600000
  - name: override-unit
    command: "cargo test -p xagent-brain --lib -- extra_sensory"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Build the kernel's sense table and extra sensory override from the second-odour flag

## Outcome

`GpuKernel::new` picks its sense table from the config flag alone: with
`second_odour_sense_enabled` set it builds with the three-sense table and emits
`EXTRA_SENSORY_CHANNELS = 2`; with it off it builds exactly as today. Callers
need no extra argument (PLAN.md "Owner decisions on the 0003 interface" item 3).

## Scope

- `crates/xagent-brain/src/gpu_kernel.rs`:
  - `GpuKernel::new` delegates to `new_with_senses(agent_count, food_count,
    brain_config, world_config, crate::senses::senses_for_config(brain_config))`
    instead of `DECLARED_SENSES`. `new_with_senses` keeps its signature.
  - `pub fn brain_layout(&self) -> &BrainLayout`, beside
    `nourishing_food_kind`, documented as returning the layout the pipelines
    were built from. It is added because no existing public accessor returns
    the layout and the GPU test needs the sense-block count and sensory stride.
  - `vision_override_constants(layout)`: insert `"EXTRA_SENSORY_CHANNELS"` with
    `layout.extra_sensory_channels` converted by `u32::try_from(..).expect(..)`
    and then `f64::from`, as the retina conversions do, next to the
    `DANGER_PERCEPT_FEATURES_ACTIVE` insert. Keep the signature and the
    `map.insert("VISION_W", ...)` line unchanged. Do not set `FEATURE_COUNT` or
    `SENSORY_STRIDE` from Rust (`vision_override_map_covers_root_constants`
    forbids derived values); if that test enumerates expected keys, add
    `EXTRA_SENSORY_CHANNELS` to its list.
  - Unit tests in `mod tests`, CPU only, each with the `extra_sensory` prefix:
    `extra_sensory_override_is_zero_by_default` (layout from
    `BrainConfig::default()` gives `0.0`),
    `extra_sensory_override_follows_second_odour_flag` (flag on gives `2.0`),
    `extra_sensory_override_is_declared_in_common` (`COMMON_SRC` has the exact
    line `override EXTRA_SENSORY_CHANNELS: u32 = 0u;`).
- `crates/xagent-brain/tests/second_odour_kernel.rs` (new), with the same
  fail-closed `gpu_guard_decision` / `gpu_or_skip` helper as
  `tests/food_kind_draw.rs` (copy it; test files share no module):
  - `second_odour_kernel_builds_with_extended_table`: a kernel built through
    `GpuKernel::new` with the flag on has `brain_layout().sense_blocks().len()
    == 3` and `sensory_stride` equal to
    `BrainLayout::from_config(&config).sensory_stride`; `dispatch_ticks(0, 4)`
    returns without panicking.
  - `second_odour_kernel_off_matches_default_layout`: with the flag off,
    `format!("{:?}", kernel.brain_layout())` equals the `Debug` text of
    `BrainLayout::from_config(&BrainConfig::default())` (`BrainLayout` derives
    only `Clone, Debug`).

## Non-goals

- No WGSL, `buffers.rs` or `senses.rs` change; no change to the food-kind draw,
  meals, vision or the harness.
- No planning vocabulary or `TODO` in source or test comments.

Prerequisites: `senses_for_config` (second-odour-sense, through
extra-sensory-layout), `BrainLayout::extra_sensory_channels` and the WGSL
override (extra-sensory-layout), `new_with_senses` (0003 task 0204), and the
draw changes to the same file (kernel-food-kind-draw).

## Steps

**Steps:**

1. Change `GpuKernel::new` to pass `senses_for_config(brain_config)`.
2. Add `brain_layout()` with its doc comment.
3. Add the `EXTRA_SENSORY_CHANNELS` insert to `vision_override_constants` and keep
   `vision_override_map_covers_root_constants` green.
4. Add the three `extra_sensory_*` unit tests.
5. Create `tests/second_odour_kernel.rs` with the helper and the two GPU tests,
   each starting with `if !gpu_or_skip() { return; }`.

- **Done when:** the check focused-proof passes with `second_odour_kernel_builds_with_extended_table` and `second_odour_kernel_off_matches_default_layout` ok, override-unit passes its three `extra_sensory` tests, brain-lib passes and parity matches the unchanged golden, proving INV-05 (a kernel built from the flag alone uses the three-sense table and emits `EXTRA_SENSORY_CHANNELS = 2`), INV-01 (with the flag off the override is 0, the layout is the default one and the golden is unchanged) and INV-10 (the GPU tests start with the fail-closed helper and the GPU checks set `XAGENT_REQUIRE_GPU=1`).
