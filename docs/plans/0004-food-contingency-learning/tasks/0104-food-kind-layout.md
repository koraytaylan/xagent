---
id: food-kind-layout
title: "Food-kind slots, harm constants and WGSL kind helpers"
workstream: "0100"
kind: task
role: developer
depends_on:
  - food-kind-config
gated: false
touches:
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/gpu_kernel/rounding_validation.rs
  - crates/xagent-sandbox/tests/sense_block_parity.rs
status: done
merged_as: "ea1b2260ac6e98a7e7fe03b5d85cb57cd33466c9"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib buffers"
    timeout_ms: 3600000
  - name: rounding-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib rounding_validation -- --include-ignored"
    timeout_ms: 3600000
  - name: gradient-audit
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test gradient_parity_audit"
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Food-kind slots, harm constants and WGSL kind helpers

## Outcome

The Rust/WGSL food-kind layout exists, with no behaviour change while the
flags are off:

- the world-config kind slot (`WC_FOOD_KINDS`, slot 27) and its Rust helper;
- the harmful-meal physics slot (`P_HARMFUL_MEAL_COUNT` = 53, `PHYS_STRIDE` 54),
  the harm ratio constant with its Rust mirror, and the harm loss helper;
- the kind colour constants and the WGSL kind helpers, which nothing calls yet;
- the 0003 parity test hashes only the pre-scenario physics slots, so its
  committed golden still matches, and it asserts that the new slot is `0.0`.

Behaviour is unchanged. See PLAN.md "Owner decisions on the 0003 interface"
item 4, and ARCHITECTURE.md INV-01, INV-03 and INV-10. The extra sensory
channels and the second-odour layout are task `extra-sensory-layout`.

## Scope

### Step 1: `crates/xagent-brain/src/buffers.rs` (food-kind constants)

- `pub const WC_FOOD_KINDS: usize = 27;`, doc: `0.0` off, `1.0` `First`
  nourishing, `2.0` `Second` nourishing. It uses the padding slot that is unused
  at baseline, so `WORLD_CONFIG_SIZE` stays 28.
- `pub fn food_kinds_slot_value(nourishing: Option<FoodKind>) -> f32`: `None`
  gives `0.0`, `Some(First)` gives `1.0`, `Some(Second)` gives `2.0`. Import
  `FoodKind` from `xagent_shared` (provided by food-kind-config, used unchanged).
- `fill_world_config` and `build_world_config` keep their signatures and output.
  Slot 27 stays `0.0` there. The kernel packs it in a later task.
- `pub const P_HARMFUL_MEAL_COUNT: usize = 53;`, doc: generation-cumulative count
  of harmful meals, preserved across respawn like `P_FOOD_COUNT`.
  `pub const PHYS_STRIDE: usize = 54;` (replaces 53 at line 414).
- `pub const HARMFUL_MEAL_INTEGRITY_PER_ENERGY: f32 = 0.6 / 0.4;`, doc: Rust
  mirror of the WGSL `ENERGY_WEIGHT / INTEGRITY_WEIGHT`, so a harmful meal's raw
  gradient is the negative of a nourishing meal's.
- `const HARMFUL_MEAL_MAX_ENERGY_FLOOR: f32 = 1e-6;`, documented as the divisor
  guard.
- `pub fn harmful_meal_integrity_loss(food_energy: f32, max_energy: f32,
  max_integrity: f32) -> f32` returns `food_energy / max_energy.max(FLOOR) *
  HARMFUL_MEAL_INTEGRITY_PER_ENERGY * max_integrity`. The clamp stays in WGSL.
- Parity map (about line 1752): add `WC_FOOD_KINDS` and `P_HARMFUL_MEAL_COUNT`
  beside `P_FOOD_COUNT`. Add `P_HARMFUL_MEAL_COUNT` to the field list in
  `phys_stride_covers_all_fields` (about line 2036), beside `P_FOOD_COUNT`.

### Step 2: `crates/xagent-brain/src/shaders/kernel/common.wgsl`

- `const PHYS_STRIDE: u32 = 54u;` replaces `53u` (line 426). Add
  `const P_HARMFUL_MEAL_COUNT: u32 = 53u;` and `const WC_FOOD_KINDS: u32 = 27u;`.
- Beside `INTEGRITY_WEIGHT` (line 754), add exactly:
  `const HARMFUL_MEAL_INTEGRITY_PER_ENERGY: f32 = ENERGY_WEIGHT / INTEGRITY_WEIGHT;`.
- Add `const FOOD_KIND_FIRST: u32 = 0u;`, `const FOOD_KIND_SECOND: u32 = 1u;`,
  `const FOOD_BASE_COLOR: vec4<f32> = vec4<f32>(0.7, 0.95, 0.2, 1.0);` and
  `const FOOD_SECOND_KIND_COLOR: vec4<f32> = vec4<f32>(0.85, 0.3, 0.75, 1.0);`.
- After `wc_f32` and `wc_u32` (line 1060 onward), add these helpers. None of them
  reads `food_flags`, because vision declares it non-atomic:
  - `fn food_kinds_enabled() -> bool`: `wc_f32(WC_FOOD_KINDS) > 0.5`.
  - `fn food_item_kind(item: u32) -> u32`: `item % 2u`.
  - `fn food_item_is_nourishing(item: u32) -> bool`: `true` when kinds are off,
    else `food_item_kind(item) + 1u == u32(wc_f32(WC_FOOD_KINDS))`.
  - `fn food_item_color(item: u32) -> vec4<f32>`: `FOOD_SECOND_KIND_COLOR` when
    kinds are on and the kind is `FOOD_KIND_SECOND`, else `FOOD_BASE_COLOR`.
### Step 3: `crates/xagent-brain/src/gpu_kernel/rounding_validation.rs`

- Import `P_HARMFUL_MEAL_COUNT` and add it beside `P_FOOD_COUNT` in both PHYSICS
  discrete-float lists (line 179 and line 507).

### Step 4: `crates/xagent-sandbox/tests/sense_block_parity.rs` (physics prefix)

Created by 0003 task 0101 (`parity-golden`), with its committed golden.

- Add documented `PRE_SCENARIO_PHYSICS_SLOTS: usize = P_HARMFUL_MEAL_COUNT;` (the
  slots the golden was recorded from; import it and `PHYS_STRIDE` from
  `xagent_brain::buffers`).
- Hash, per agent, only slots `[0, PRE_SCENARIO_PHYSICS_SLOTS)`, in the same order
  and encoding as before, reading the same contiguous buffer at stride `PHYS_STRIDE`.
- Assert slot `P_HARMFUL_MEAL_COUNT` is `0.0` for every agent in every GPU
  scenario, with a failing assertion, never a skip.
- Do not touch or re-record the golden, and change no scenario, tick, seed or other
  hashed buffer. If the landed test does not hash physics as one contiguous
  per-agent buffer, stop and report the file and line.

### Step 5: tests in `buffers::tests` (names are fixed)

Add to `buffers::tests` (it holds `parse_wgsl_u32_constants` and
`parse_wgsl_f32_constant`), so `--lib buffers` selects them:

- `food_kinds_slot_value_maps_each_choice`: `None` 0.0, `First` 1.0, `Second` 2.0.
- `harmful_meal_loss_matches_ratio_formula`: `(30.0, 100.0, 100.0)` gives 45.0
  within 1e-4.
- `harmful_meal_loss_guards_zero_max_energy`: `max_energy` 0.0 gives a finite
  result.
- `food_kind_wgsl_mirror_matches_rust`: `PHYS_STRIDE` is 54 and `WC_FOOD_KINDS`
  27 (via `parse_wgsl_u32_constants`), `P_HARMFUL_MEAL_COUNT` is 53. The WGSL
  `ENERGY_WEIGHT / INTEGRITY_WEIGHT` (via `parse_wgsl_f32_constant`) equals the
  Rust constant. `common.wgsl` contains the exact line
  `const HARMFUL_MEAL_INTEGRITY_PER_ENERGY: f32 = ENERGY_WEIGHT / INTEGRITY_WEIGHT;`.
Existing tests stay green; an assertion in `touches` pinning `PHYS_STRIDE` to 53
becomes 54. A slot-count assumption outside `touches` is a stop-and-report.

## Non-goals

- No change to `fill_world_config`/`build_world_config` signatures or output,
  `WORLD_CONFIG_SIZE` (28) or `WC_DANGER_PERCEPT_BLINDED` (26).
- No `BrainLayout` field, `with_senses` change or `EXTRA_SENSORY_CHANNELS`
  (task `extra-sensory-layout`); no edit to `gpu_kernel.rs`, `senses.rs`,
  `agent/mod.rs`, any other shader, the golden or tests outside `touches`; no
  caller of the new WGSL helpers.
- No planning vocabulary or `TODO` in source (`contributing_guard.rs`).

Prerequisites: `xagent_shared::FoodKind` and `WorldConfig.food_kinds_enabled`
(food-kind-config); the parity test from 0003 task 0101 (plan dependency).
Baseline: `P_FOOD_COUNT` (18), `P_FOOD_CLAIM` (52), `PHYS_STRIDE` 53,
`ENERGY_WEIGHT`/`INTEGRITY_WEIGHT`, `wc_f32`/`wc_u32`; rounding validation lists
`P_FOOD_COUNT` twice.

## Steps

**Steps:**
1. Scope Step 1 in `buffers.rs`, each new item with a `///` doc comment.
2. Scope Step 2 in `common.wgsl`.
3. Scope Step 3 in `rounding_validation.rs`.
4. Scope Step 4 in `sense_block_parity.rs`, applying its stop rule.
5. Scope Step 5: the four `buffers::tests` tests.

- **Done when:** the checks `focused-proof`, `rounding-regression`, `gradient-audit` and `parity` all pass on the host: `focused-proof` runs the four Step 5 tests green; `rounding-regression` runs the rounding validations with the new counter in the discrete list; `gradient-audit` stays green unchanged; and `parity` reports that the unchanged golden matched with every agent's `P_HARMFUL_MEAL_COUNT` slot `0.0`. Together they prove INV-01 (`fill_world_config` and `build_world_config` output is unchanged, slot 27 is `0.0` there, `WORLD_CONFIG_SIZE` stays 28, and the golden is unchanged), INV-03 (the harm ratio constant equals `ENERGY_WEIGHT / INTEGRITY_WEIGHT` in Rust and WGSL, `P_HARMFUL_MEAL_COUNT` is at index 53 and `PHYS_STRIDE` is 54) and INV-10 (the GPU checks run with `XAGENT_REQUIRE_GPU=1`).
