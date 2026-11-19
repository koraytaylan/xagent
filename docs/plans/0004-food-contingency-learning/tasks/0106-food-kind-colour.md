---
id: food-kind-colour
title: "Kind colour in every vision path"
workstream: "0100"
kind: task
role: developer
depends_on:
  - food-kind-layout
  - kernel-food-kind-draw
gated: false
touches:
  - crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_vision_parallel.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_vision_object_queries.wgsl
  - crates/xagent-brain/tests/food_kind_colour.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_colour"
    timeout_ms: 3600000
  - name: vision-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib vision_validation -- --include-ignored"
    timeout_ms: 5400000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Kind colour in every vision path

## Outcome

With food kinds on, a `Second` item renders `FOOD_SECOND_KIND_COLOR` in every
vision path and every other item renders the base colour. With kinds off every
item renders the baseline `(0.7, 0.95, 0.2, 1.0)`. No colour encodes which kind
is nourishing (ARCHITECTURE.md INV-04, INV-02, INV-01).

## Scope

- `phase_vision.wgsl` colour site (line 103, inside the loop that defines `fidx`
  at line 89): replace `hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);` with
  `hit_color = food_item_color(fidx);`.
- `phase_vision_parallel.wgsl` (line 191): the same replacement, using `fidx`
  (in scope from line 73).
- `phase_vision_object_queries.wgsl` (line 196): the same replacement. Read
  lines 185-200 and use the food-index variable in scope there; invent no index.
- `crates/xagent-brain/tests/food_kind_colour.rs` (new): the tests under Steps.

`food_item_color(i)` (from food-kind-layout in `common.wgsl`) returns
`FOOD_BASE_COLOR` when kinds are off, which is the baseline literal, and
`FOOD_SECOND_KIND_COLOR` (`(0.85, 0.3, 0.75, 1.0)`) for a `Second` item when kinds
are on. The kernel packs `WC_FOOD_KINDS` (kernel-food-kind-draw).

## Decisions

- **One helper, no new pass.** Only the three literals change; no scent code,
  buffer or binding changes here (task `food-kind-senses`).
- **No value label.** No vision code reads `food_item_is_nourishing`.
- **Fail-closed GPU tests.** The test file defines `fn gpu_or_skip() -> bool`:
  `true` when `GpuKernel::is_available()`; when false and
  `std::env::var_os("XAGENT_REQUIRE_GPU").is_some()`, panic naming the missing
  adapter; otherwise print a skip line and return `false`. Every test starts
  with `if !gpu_or_skip() { return; }`.
- **Readback.** Pixels come from the existing vision readback the
  `vision_validation` tests use; find the accessor by search and add no
  readback helper to `gpu_kernel.rs`.

## Non-goals

- No edit to `gpu_kernel.rs`, `common.wgsl`, `buffers.rs`,
  `gpu_kernel/vision_validation.rs` (its `FOOD_COLOR` constant and `#[ignore]`
  tests stay), any CPU mirror or the GUI. If a needed name is missing, stop and
  report it.
- No planning vocabulary or `TODO` in source (`contributing_guard.rs`).

## Steps

**Steps:**

1. Replace the colour literal in the three shaders with
   `food_item_color(<item index in scope>)`, and confirm by search that
   `0.7, 0.95, 0.2` no longer appears in any of them.
2. Create `tests/food_kind_colour.rs` with `gpu_or_skip` and
   `food_kind_colour_default_off_is_baseline`: kinds off, an odd-index item in
   view renders `(0.7, 0.95, 0.2, 1.0)` in its hit pixels.
3. Add `food_kind_colour_by_kind`: kinds on, a `Second` item renders
   `FOOD_SECOND_KIND_COLOR` and a `First` item the base colour in their hit
   pixels.
4. Add `food_kind_colour_ignores_assignment`: kinds on, the same scene rendered
   after `reset_agents_seeded` with a seed that changes
   `nourishing_food_kind()` gives bit-identical hit pixels (re-upload the same
   positions first if the reset moves them).

- **Done when:** the focused-proof check passes with `food_kind_colour_default_off_is_baseline`, `food_kind_colour_by_kind` and `food_kind_colour_ignores_assignment` ok, and the vision-regression and parity checks pass unchanged, proving INV-04 (`Second` items render `FOOD_SECOND_KIND_COLOR` in every vision path), INV-02 (the colour does not change when the nourishing kind is redrawn), INV-01 (with kinds off every item renders the baseline colour and the parity golden is unchanged) and INV-10 (every test starts with `gpu_or_skip`, which panics without an adapter when `XAGENT_REQUIRE_GPU` is set).
