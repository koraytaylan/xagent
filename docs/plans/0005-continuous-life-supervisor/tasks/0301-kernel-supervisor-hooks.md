---
id: kernel-supervisor-hooks
title: "Kernel nourishing-kind setter and single-slot replacement"
workstream: "0300"
kind: task
role: developer
depends_on: []
gated: false
touches: [crates/xagent-brain/src/gpu_kernel.rs, crates/xagent-brain/tests/supervisor_kernel.rs]
status: todo
merged_as: ""
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test supervisor_kernel", timeout_ms: 3600000}
  - {name: brain-lib, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib", timeout_ms: 5400000}
  - {name: food-kind-default, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_draw", timeout_ms: 3600000}
  - {name: sense-parity, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored", timeout_ms: 5400000}
  - {name: seeded-digest-determinism, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest", timeout_ms: 3600000}
  - {name: brain-clippy, command: "cargo clippy -p xagent-brain --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: brain-fmt, command: "cargo fmt -p xagent-brain -- --check", fix: "cargo fmt -p xagent-brain"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Kernel nourishing-kind setter and single-slot replacement

## Outcome

`xagent_brain::gpu_kernel` gains `SlotSpawn`, `slot_spawn_physics_row`,
`replace_agent_slot` and `set_nourishing_food_kind`, and `upload_agents` uses
the row builder with identical bytes.

## Scope

- `crates/xagent-brain/src/gpu_kernel.rs`:
  - `#[derive(Clone, Copy, Debug, PartialEq)] pub struct SlotSpawn { pub
    position: glam::Vec3, pub max_energy: f32, pub max_integrity: f32, pub
    memory_capacity: usize, pub processing_slots: usize }`, placed before
    `pub struct GpuKernel`.
  - `pub fn slot_spawn_physics_row(spawn: &SlotSpawn) -> Vec<(usize, f32)>`, a
    free function beside it. It starts from `vec![0.0f32; PHYS_STRIDE]`, sets
    the named slots and returns all `PHYS_STRIDE` pairs `(k, row[k])` in slot
    order. Named slots: `P_POS_X/Y/Z` from `position`; `P_FACING_Z` 1.0;
    `P_ENERGY`, `P_MAX_ENERGY`, `P_PREV_ENERGY` from `max_energy`;
    `P_INTEGRITY`, `P_MAX_INTEGRITY`, `P_PREV_INTEGRITY` from `max_integrity`;
    `P_ALIVE` 1.0; `P_MEMORY_CAP` and `P_PROCESSING_SLOTS` from the two counts
    as `f32`. Every other slot is 0.0.
  - `upload_agents` keeps its signature and its single `write_buffer`. Its
    per-agent loop copies each pair of `slot_spawn_physics_row` into
    `data[i * PHYS_STRIDE + slot]`.
  - `#[derive(Debug)] pub enum SlotReplaceError { IndexOutOfRange { index: u32,
    agent_count: u32 }, BrainStateLength { expected: usize, actual: usize },
    PatternLength { expected: usize, actual: usize } }`.
  - `pub fn replace_agent_slot(&self, index: u32, birth: &AgentBrainState, config:
    &BrainConfig, spawn: &SlotSpawn) -> Result<(), SlotReplaceError>`, placed
    after `write_agent_physics_fields`. Before any write it checks, in order,
    `index < self.agent_count`, `birth.brain_state.len() ==
    self.layout.brain_stride` and `birth.patterns.len() == PATTERN_STRIDE`. Then
    it calls, in this order and nothing else, `write_agent_state(index, birth)`,
    `write_agent_heritable_config(index, config)` and
    `write_agent_physics_fields(index, &slot_spawn_physics_row(spawn))`. The doc
    comment states that it is a single-slot write used between dispatches and
    touches no other agent.
  - `#[derive(Debug)] pub enum FoodKindSetError { FoodKindsDisabled }`.
  - `pub fn set_nourishing_food_kind(&mut self, kind: FoodKind) -> Result<(),
    FoodKindSetError>`, placed beside `set_freeze_steering_weights`. With
    `food_kinds_enabled` false it returns `Err(FoodKindsDisabled)` and changes
    nothing. Otherwise it sets `nourishing_food_kind = Some(kind)` and
    `world_config_scratch[WC_FOOD_KINDS] = food_kinds_slot_value(Some(kind))`.
    Its doc comment says the kernel's owning thread is the only caller.
  - Both error enums implement `Display` and `std::error::Error`.
- `crates/xagent-brain/tests/supervisor_kernel.rs` (new): the seven tests in
  Steps 4 and 5.

## Non-goals

- No WGSL, `buffers.rs`, `senses.rs` or sandbox edit.
- No `lib.rs` edit. `gpu_kernel` is already `pub mod`, so the items are reached
  as `xagent_brain::gpu_kernel::{...}`; a crate-root re-export would need
  `lib.rs`, which is outside `touches`.
- No change to draw rules, reset paths, dispatch or any existing signature; no
  accessor beyond the items above. No plan, task or `TODO` wording in code.

## Prerequisites

Plan 0004 must be landed; none of its symbols exist in `crates/` at baseline,
and the plan-level `depends_on` carries that. Used: `FoodKind { First, Second }`
(0101); `food_kinds_enabled`, `nourishing_food_kind`, `food_kind_draws` and
`world_config_scratch` (0105); `WC_FOOD_KINDS`, `food_kinds_slot_value` (0104);
`nourishing_food_kind()` (0105); `P_HARMFUL_MEAL_COUNT` and the placement in
`tests/food_kind_meals.rs` (0102). If a landed name differs, report the file and
line; do not rename.

## Decisions

- **One row builder** shared by `upload_agents` and `replace_agent_slot`.
  Unnamed slots are 0.0, so a replaced slot's counters start at zero.
- **Validation before any write.** **Setter does not draw.**
- **Fail-closed helper.** The test file copies 0004's `gpu_guard_decision` and
  `gpu_or_skip`; each GPU test starts with `if !gpu_or_skip() { return; }`.
- **Named constants:** `AGENT_COUNT = 3`, `TARGET_SLOT: u32 = 1`,
  `WARMUP_BATCHES: u64 = 3`, `MEAL_ITEM: usize = 0` (index 0 is `First`),
  `BIRTH_SEED: u64 = 11`, `RESET_SEED: u64 = 8`; the batch length is
  `kernel_batch_size()`.

## Steps

**Steps:**
1. In `gpu_kernel.rs`, add `SlotSpawn` and `slot_spawn_physics_row` as in
   Scope, and rewrite the per-agent loop of `upload_agents` to copy that row's
   pairs into `data`. Keep the signature and the single `write_buffer`.
2. In `gpu_kernel.rs`, add `SlotReplaceError` with `Display` and `Error`, and
   `replace_agent_slot` after `write_agent_physics_fields`, with the checks and
   exactly the three writes of Scope.
3. In `gpu_kernel.rs`, add `FoodKindSetError` with `Display` and `Error`, and
   `set_nourishing_food_kind` beside `set_freeze_steering_weights`.
4. Create `crates/xagent-brain/tests/supervisor_kernel.rs` with the two helpers
   and the tests `gpu_guard_fails_closed_when_required` (CPU: `(false, true)` is
   `Err`, `(false, false)` is `Ok(false)`, `(true, _)` is `Ok(true)`),
   `spawn_row_matches_upload_agents` (three agents at distinct positions; each
   agent's `PHYS_STRIDE` slice from `read_full_state_blocking` equals its spawn
   row), `replace_slot_resets_row_and_writes_brain` (dispatch `WARMUP_BATCHES`
   batches of `kernel_batch_size()` ticks; record every physics row and
   `read_agent_state` brain; replace `TARGET_SLOT` with the birth brain read from
   a second kernel built and reset with `BIRTH_SEED`; the target's row equals its
   spawn row and its `brain_state` and `patterns` equal the birth brain; every
   other agent's row and brain equal the recorded values) and
   `replace_slot_rejects_bad_index_and_lengths` (index `AGENT_COUNT as u32`, a
   short `brain_state` and a short `patterns` return the matching variant, and
   the target's row and brain are unchanged).
5. In the same file add `set_kind_refused_when_kinds_off` (`Err(FoodKindsDisabled)`,
   kind stays `None`), `set_kind_keeps_agents_and_draw_counter` (kinds on: record
   brains and physics, call `set_nourishing_food_kind(FoodKind::Second)`, re-read
   and compare; then `reset_agents_seeded(.., RESET_SEED)` and the kind equals
   `world.nourishing_food_kind(1)`, so the counter did not move) and
   `set_kind_changes_meal_outcome` (0102 placement: one item at the origin at
   index `MEAL_ITEM`, one agent on it, one brain cycle, a fresh kernel per kind,
   the setter after `reset_agents_seeded`; `First` set gives `P_FOOD_COUNT` 1,
   `Second` set gives `P_HARMFUL_MEAL_COUNT` 1).

- **Done when:** the checks focused-proof, brain-lib, food-kind-default, sense-parity and seeded-digest-determinism all pass on the host, with focused-proof reporting ok for gpu_guard_fails_closed_when_required, spawn_row_matches_upload_agents, replace_slot_resets_row_and_writes_brain, replace_slot_rejects_bad_index_and_lengths, set_kind_refused_when_kinds_off, set_kind_keeps_agents_and_draw_counter and set_kind_changes_meal_outcome; together they prove INV-09 (the replaced slot's row equals its spawn row and its brain the birth brain, every other agent is unchanged, and a bad index or length errors with no write), INV-10 (the setter changes the kind the next upload packs, resets no agent, leaves the draw counter unchanged, and is refused when kinds are off), INV-01 (the sense-parity golden, the food-kind-default check pass unchanged after the upload_agents refactor, spawn_row_matches_upload_agents shows the refactored rows equal the builder rows, and the seeded headless run gives the same digest twice; this task does not require the digest to equal the recorded fixture, whose value is adapter-specific and was recorded on a different adapter at 0976448), and INV-15 (the GPU guard panics without an adapter when required).

## Evidence and risks

The worker has no shell; host-run checks are the proof. The uniform effect is
proven through the meal outcome at dispatch time. Brain equality compares the
`brain_state` and `patterns` vectors.
