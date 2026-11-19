---
id: kernel-food-kind-meals
title: "Harmful meals act only through integrity, counted separately"
workstream: "0100"
kind: task
role: developer
depends_on:
  - food-kind-layout
  - kernel-food-kind-draw
gated: false
touches:
  - crates/xagent-brain/src/shaders/kernel/phase_food_claim.wgsl
  - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
  - crates/xagent-brain/src/shaders/kernel/physics_tick.wgsl
  - crates/xagent-brain/src/shaders/kernel/persistent_tick.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
  - crates/xagent-brain/tests/food_kind_meals.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_meals"
    timeout_ms: 3600000
  - name: food-claims-regression
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_claims"
    timeout_ms: 3600000
  - name: gradient-audit
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test gradient_parity_audit"
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Harmful meals act only through integrity, counted separately

## Outcome

`resolve_food_claim` handles harmful items: no energy, an integrity loss, a
same-scope death mark, and a separate `P_HARMFUL_MEAL_COUNT`. Nourishing meals
and default-off meals behave as before. The constants, helpers and kind draw
come from food-kind-layout and kernel-food-kind-draw; this task consumes them
unchanged.

## Scope

- `phase_food_claim.wgsl`: `fn resolve_food_claim(agent: u32, tick: u32)`. The
  claim test, the clear of `P_FOOD_CLAIM` and the `atomicStore` that consumes the
  item run for both kinds.
  - Nourishing (`food_item_is_nourishing`): exactly the existing two statements,
    energy then `P_FOOD_COUNT`.
  - Harmful: no energy and no `P_FOOD_COUNT`. Then, in one scope:
    `integrity = clamp(integrity - food_energy / max(P_MAX_ENERGY, EPSILON) *
    HARMFUL_MEAL_INTEGRITY_PER_ENERGY * P_MAX_INTEGRITY, 0.0, P_MAX_INTEGRITY)`.
    When the result is `<= 0.0`, set `P_ALIVE = 0.0`, `P_DIED_FLAG = 1.0` and
    `P_LAST_DEATH_TICK = f32(tick)`. Then `P_HARMFUL_MEAL_COUNT += 1.0`.
  - Do not touch `raw_gradient`, any reward term, or `ENERGY_WEIGHT` /
    `INTEGRITY_WEIGHT`.
- Callers pass the tick already in scope:
  - `kernel_tick.wgsl`: `resolve_food_claim(agent_id, base_tick);`
  - `persistent_tick.wgsl`: `resolve_food_claim(tid, cycle_tick);`
  - `physics_tick.wgsl`: `resolve_food_claim(tid, tick);`
- Respawn: in `kernel_tick.wgsl` `agent_death_respawn`, save
  `P_HARMFUL_MEAL_COUNT` beside `saved_food_count` (about line 623) and restore
  it beside the `P_FOOD_COUNT` restore (about line 663). Do the same in
  `phase_death.wgsl` `phase_death_respawn` (about lines 39 and 85).

## Non-goals

- No edit to Rust files, `common.wgsl`, vision shaders, or the `food_claims` and
  `gradient_parity_audit` tests. Both must stay green unchanged.
- No reward, label or sensory flag. No planning vocabulary in comments or
  strings, and no `TODO`.

## Decisions

- The harmful item is consumed by the same `atomicStore` as a nourishing item.
- Kernel path (`kernel_tick.wgsl`, `persistent_tick.wgsl`): the claim runs before
  `agent_death_respawn`, so a claim death is respawned in that cycle.
- `physics_tick.wgsl` runs `phase_death_respawn` before the claim, so a claim
  death there is respawned on the next physics tick. This ordering is accepted.
  The lethal test uses the kernel path.
- The harmful test keeps energy at 40% of max, below the 50% gate, so integrity
  regeneration cannot run.
- Named tolerance: `const LOSS_TOLERANCE: f32 = 1e-3;` (f32 rounding).

## Steps

**Steps:**

1. In `phase_food_claim.wgsl`, change the signature to take `tick` and add the
   harmful branch from Scope. Keep the nourishing statements unchanged.
2. Update the three callers in Scope to pass the tick already in scope. Save and
   restore `P_HARMFUL_MEAL_COUNT` in `kernel_tick.wgsl` and `phase_death.wgsl`.
3. Create `crates/xagent-brain/tests/food_kind_meals.rs` with the GPU helpers of
   `food_claims.rs` (`movement_speed: 0.0`, `upload_world`, `upload_agents`),
   but not its silent-return guard. Add `fn gpu_or_skip() -> bool`: it returns
   true when `GpuKernel::is_available()`; otherwise it panics with a message
   naming the missing GPU adapter when `std::env::var_os("XAGENT_REQUIRE_GPU")`
   is set, and else prints a skip line and returns false. Every GPU test starts
   with `if !gpu_or_skip() { return; }`. Add
   `food_kind_meals_default_off_every_item_nourishes`: with
   kinds off, items 0 and 1 each give exactly `food_energy_value`,
   `P_FOOD_COUNT` 1, integrity at max unchanged, and `P_HARMFUL_MEAL_COUNT` 0.
4. Add `food_kind_meals_nourishing_and_harmful`: kinds on, one agent per item,
   energy at 40% of max. Pick the nourishing index with
   `nourishing_food_kind()`. The nourishing agent gains `food_energy_value` and
   `P_FOOD_COUNT` 1. The harmful agent gains no energy, loses
   `buffers::harmful_meal_integrity_loss(...)` within `LOSS_TOLERANCE`,
   `P_HARMFUL_MEAL_COUNT` 1 and `P_FOOD_COUNT` 0.
5. Add `food_kind_meals_lethal_harmful_meal_marks_death`: integrity set below the
   loss, one kernel dispatch. Read the death handling in `kernel_tick.wgsl`
   first, then assert `P_DEATH_COUNT` 1 and `P_LAST_DEATH_TICK` equal to the
   dispatch tick. Removing the same-scope mark must fail this test.
6. Add `food_kind_meals_harmful_counter_survives_respawn`: after a lethal harmful
   meal and respawn, `P_HARMFUL_MEAL_COUNT` is 1 and `P_FOOD_COUNT` is 0.

- **Done when:** the checks focused-proof, food-claims-regression, gradient-audit and parity all pass on the host, with focused-proof green for the four food_kind_meals tests run with XAGENT_REQUIRE_GPU=1 (so a missing adapter fails rather than skips), the other two regressions passing unchanged, and parity reporting the committed 0003 golden unchanged, proving INV-03 (a harmful meal adds no energy, subtracts the clamped loss, marks death at zero integrity in the same scope, increments only P_HARMFUL_MEAL_COUNT, and the counter survives respawn), INV-01 (kinds off: each meal adds exactly food_energy_value and increments only P_FOOD_COUNT, and the parity golden is byte-identical), and INV-10 (every new GPU test panics without an adapter when XAGENT_REQUIRE_GPU is set).
