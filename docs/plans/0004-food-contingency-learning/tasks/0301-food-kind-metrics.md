---
id: food-kind-metrics
title: "Food-kind free-life metrics, link accumulators and run summaries"
workstream: "0300"
kind: task
role: developer
depends_on:
  - food-kind-config
gated: false
touches:
  - crates/xagent-sandbox/src/lib.rs
  - crates/xagent-sandbox/src/food_kind_metrics.rs
status: done
merged_as: "b9c75c76cc0610c099ad7c29c44b0fcd435248af"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib food_kind_metrics"
---

# Food-kind free-life metrics, link accumulators and run summaries

## Outcome

A pure module, `crates/xagent-sandbox/src/food_kind_metrics.rs`, with per-kind
measurements and serde run-summary types. It holds no GPU handle and writes nothing.

## Scope

- `crates/xagent-sandbox/src/lib.rs`: add `pub mod food_kind_metrics;` after `pub mod bench;`.
- `crates/xagent-sandbox/src/food_kind_metrics.rs` (new), in this order:
  1. `//!` doc. Constants with `///` docs giving each value's reason:
     `LATE_FRACTION_NUMERATOR: u64 = 3`, `LATE_FRACTION_DENOMINATOR: u64 = 10`
     (last 30% of ticks, integer so the boundary is exact), `QUARTER_COUNT: usize = 4`.
  2. Serde types, all `pub`, deriving `Debug, Clone, PartialEq`:
     - `Arm { Learning, Frozen }`, also `Copy, Eq, Hash, Serialize, Deserialize`,
       `#[serde(rename_all = "lowercase")]`.
     - `AgentFoodKindSummary`, fields in this order: `turn_toward_nourishing_whole: Option<f32>`,
       `turn_toward_nourishing_late: Option<f32>`, `turn_away_harmful_whole: Option<f32>`,
       `turn_away_harmful_late: Option<f32>`, `nourishing_meals_by_quarter: [u32; 4]`,
       `harmful_meals_by_quarter: [u32; 4]`, `deaths: u32`,
       `readout_balanced_accuracy: Option<f32>`, `readout_shuffled_accuracy: Option<f32>`,
       `signed_credit_pair_first: f32`, `signed_credit_pair_second: f32`,
       `signed_weight_difference_pair_first: f32`, `signed_weight_difference_pair_second: f32`.
     - `RunFoodKindSummary`, fields in this order: `seed: u64`, `arm: Arm`,
       `ticks: u64`, `agent_count: usize`, `fresh_brains: bool`,
       `nourishing_kind: FoodKind`, `agents: Vec<AgentFoodKindSummary>`. Derives `Serialize, Deserialize`.
  3. The helpers in Decisions.
- Every `pub` item has a `///` doc and `#[must_use]` where clippy pedantic requires it.
- No verdict code, thresholds or `EXPECTED_*` constants. Those belong to food-kind-verdict.

## Non-goals

- No kernel, GPU, buffer, WGSL or config change, no `GpuKernel` read, and no mutation
  of any state (INV-06). No file I/O, readout, harness or CLI code.
- No physics constant import; meal counts arrive as `u32`.
- No mention of any plan, task, workstream, decision or spec in source, doc comments
  or test names (`contributing_guard.rs` enforces this).

## Decisions

- **Imports.** `use xagent_shared::FoodKind;` (from food-kind-config; do not redefine).
  Import `FOOD_STATE_STRIDE`, `FOOD_POSITION_X`, `FOOD_POSITION_Z` and
  `FOOD_RESPAWN_TIMER` from `xagent_brain::buffers` by name, never the literals 3 or 4.
  The range is a parameter.
- **Nearest food.** `pub struct NearestFood { pub distance: f32, pub bearing: f32 }` and
  `pub fn nearest_food_of_kind(position: [f32; 3], facing: [f32; 2], food_state: &[f32], kind: FoodKind, range: f32) -> Option<NearestFood>`.
  - Item `i` counts only if `FoodKind::of_item(i) == kind`, and is available iff
    `food_state[i * FOOD_STATE_STRIDE + FOOD_RESPAWN_TIMER] == 0.0`.
  - `to = (x − position[0], z − position[2])` from `FOOD_POSITION_X` and `FOOD_POSITION_Z`.
    In range iff `to.x² + to.z² < range²`. Compare squared values, then one `sqrt`.
  - Nearest wins; ties go to the lowest index (strict `<`).
  - `bearing = atan2(facing[0] * to.z − facing[1] * to.x, facing[0] * to.x + facing[1] * to.z)`,
    the kernel formula behind `P_NEAREST_FOOD_BEARING` in `kernel_tick.wgsl`. Do not reinterpret the sign.
- **Approach sign.** `approach_sign(bearing: f32) -> f32 = -signum(bearing)`, with
  `signum(0.0) = 0.0`. Positive means toward the food, matching the kernel's approach
  test (`executed_turn × bearing < 0`, near line 307). Do not edit the conflicting comments in `buffers.rs` or the shader.
- **Turn accumulator.** `KindTurnAccumulator::new(total_ticks: u64)` and
  `record(&mut self, tick: u64, executed_turn: f32, nourishing_bearing: Option<f32>, harmful_bearing: Option<f32>)`.
  - `Some` nourishing bearing adds `executed_turn × approach_sign(b)` to toward-nourishing.
    `Some` harmful bearing adds `-executed_turn × approach_sign(b)` to away-from-harmful.
  - Late iff `tick >= late_start`, with
    `late_start = total − total.saturating_mul(LATE_FRACTION_NUMERATOR) / LATE_FRACTION_DENOMINATOR`
    (210,000 for 300,000 ticks).
  - `finish(&self) -> KindTurnMeans` has four `Option<f32>` fields, named
    `turn_toward_nourishing_whole`, `turn_toward_nourishing_late`,
    `turn_away_harmful_whole` and `turn_away_harmful_late`. Each is the mean of its
    samples, or `None` if none. Sums are `f64`, counts go through `f64::from`, and one
    documented helper narrows to `f32` with `#[allow(clippy::cast_possible_truncation)]`.
- **Meal quarters.** `pub struct KindMealQuarters { pub nourishing_by_quarter: [u32; 4], pub harmful_by_quarter: [u32; 4] }`.
  `from_cumulative(cumulative: [(u32, u32); 4]) -> Option<Self>` takes cumulative
  counts at each boundary, takes differences (the first from zero), and returns `None`
  if any count decreases. `nourishing_share(&self, quarter: usize) -> Option<f32>` is
  `n / (n + h)`, or `None` for zero meals or `quarter >= QUARTER_COUNT`.
- **Credit.** `PairCreditAccumulator` derives `Default, Debug, Clone, PartialEq`.
  `record(&mut self, pair_kind: FoodKind, td_error: f32, turn_noise_innovation: f32, whitened_left: f32, whitened_right: f32)`
  adds `td_error × turn_noise_innovation × (whitened_right − whitened_left)` to that
  pair's `f64` sum. `credit(&self, pair_kind: FoodKind) -> f32`.
- **Signing.** `signed_for_nourishing(value: f32, pair_kind: FoodKind, nourishing_kind: FoodKind) -> f32`
  returns `value` when the kinds match, else `-value`.
  `signed_weight_difference(left_weight: f32, right_weight: f32, pair_kind: FoodKind, nourishing_kind: FoodKind) -> f32`
  is `signed_for_nourishing(right_weight − left_weight, pair_kind, nourishing_kind)`.

## Steps

**Steps:**
1. Add the `lib.rs` line. Create `food_kind_metrics.rs` with the `//!` doc, the
   constants, and the serde types with the exact fields and order above.
2. Implement `NearestFood` and `nearest_food_of_kind`. Add
   `nearest_food_kind_filter_and_bearing_sign` (the other kind is closer; distance and
   bearing belong to the requested kind), `nearest_food_skips_unavailable_items` and
   `nearest_food_out_of_range_is_none`.
3. Implement `approach_sign`, `KindTurnAccumulator` and `KindTurnMeans`. Add
   `turn_accumulator_late_window_and_signs` (tick 209,999 excluded and 210,000
   included for 300,000 ticks; turn `+1.0` with bearing `-0.5` adds `+1.0` to toward)
   and `turn_accumulator_none_without_samples`.
4. Implement `KindMealQuarters`, `from_cumulative` and `nourishing_share`. Add
   `meal_quarters_shares_with_zero_meal_quarter` and `meal_quarters_rejects_decreasing_counts`.
5. Implement `PairCreditAccumulator`, `signed_for_nourishing` and
   `signed_weight_difference`. Add `pair_credit_sum_and_signing` (a two-tick sum matches
   the hand-computed value; the harmful pair flips sign) and
   `signed_weight_difference_sign`.
6. Add `run_food_kind_summary_serde_round_trip`: one agent with both `Some` and `None`
   fields, serialised with `serde_json`, deserialised, and asserted equal.

- **Done when:** the focused-proof check passes on the host with every test named in Steps reported ok and zero failed, proving INV-06 because no function constructs a kernel or writes brain, physics, food or config state.
