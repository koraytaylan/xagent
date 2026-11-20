---
id: food-kind-free-life
title: "Food-kind scenario and frozen-policy control in the free-life harness"
workstream: "0400"
kind: task
role: developer
depends_on:
  - kernel-food-kind-meals
  - kernel-food-kind-draw
  - second-odour-features
  - food-kind-metrics
  - kind-readout
  - second-odour-inheritance
  - flag-on-tail-base
gated: false
touches:
  - crates/xagent-sandbox/src/headless.rs
  - crates/xagent-sandbox/src/main.rs
  - crates/xagent-sandbox/tests/food_kind_free_life.rs
status: done
merged_as: "948bc4e3e51e8c85b899ada9ce68e272cd07e50e"
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test food_kind_free_life"
    timeout_ms: 3600000
---

# Food-kind scenario and frozen-policy control in the free-life harness

## Outcome

The free-life harness of 0001 gains three opt-in CLI surfaces: `--food-kinds`
(per-kind measurement, no state written), `--freeze-policy` (turn weights frozen) and `--food-kind-summary <file>` (one run's
`RunFoodKindSummary` JSON). Without them the harness is unchanged. The verdict CLI is
task food-kind-verdict.

## Scope

- `crates/xagent-sandbox/src/headless.rs`: extend the landed 0001 free-life runner
  (`run_free_life(config: &FullConfig, options: &FreeLifeOptions) ->
  Result<FreeLifeReport, FreeLifeError>`, `FreeLifeOptions`, `BrainSource`) with the
  food-kind run-config struct, the champion guard, the config and freeze wiring, the
  per-brain-tick measurement, per-agent accumulation, end-of-run readout and weight
  read, and `RunFoodKindSummary` construction. Do not restructure the run loop or
  per-agent metrics.
- `crates/xagent-sandbox/src/main.rs`: add clap `#[arg(long)]` fields
  `food_kinds: bool`, `freeze_policy: bool` and `food_kind_summary: Option<PathBuf>`,
  and wire them into the `--free-life` branch (`free_life_options_from_cli` and
  `run_free_life_cli`, landed by 0001).
- `crates/xagent-sandbox/tests/food_kind_free_life.rs` (new): the three tests named
  under Steps. GPU tests start with a `gpu_or_skip()` helper that panics without an
  adapter when `XAGENT_REQUIRE_GPU` is set (INV-10), like
  `gpu_adapter_present_when_required`.

## Non-goals

- No change to kernel, WGSL, buffers, config defaults, evolution, fitness or
  heritability. No new learning rule, reward term or encoder input.
- No write to brain, physics, food or config state from any measurement path
  (INV-06). The only setter called is the freeze flag.
- No stored-champion food-kind run (INV-09).
- No verdict CLI, `results/` file, `DECISION.md` or long run (tests use a few
  thousand ticks); no planning names in source; no file outside the three touches.

## Prerequisites

- 0001 landed the runner in `headless.rs` (`BrainSource::{Fresh, Champion}`,
  `FreeLifeError`), `free_life/` (`metrics::FreeLifeSample` with `td_error` and
  `turn_innovation`, `AgentMetrics`, the tick log) and the `--free-life` CLI. This task
  does not edit `free_life/` or `lib.rs`.
- Via depends_on (or transitively): `GpuKernel::nourishing_food_kind()`
  and `brain_layout()` (kernel-food-kind-draw, second-odour-kernel);
  `P_HARMFUL_MEAL_COUNT` (food-kind-layout via kernel-food-kind-meals); both config
  flags (food-kind-config); `second_odour` senses and features (second-odour-features);
  flag-on stride fix (flag-on-tail-base);
  `food_kind_metrics` items; `kind_readout::fit_kind_readout`.
- Baseline APIs: `set_freeze_steering_weights(bool)`, `cached_food_state()`, physics
  snapshot; sense offsets come from `BrainLayout`, never hard-coded. TD error
  and turn innovation come from the 0001 probe sample, so no critic constant is
  re-derived.

## Decisions

- **Food-kind config.** `--food-kinds` sets `world.food_kinds_enabled = true` and
  `brain.second_odour_sense_enabled = true` before kernel construction. A stored
  champion brain source returns an error before any GPU work, naming the
  `--food-kinds` flag (INV-09).
- **Frozen control.** `--freeze-policy` calls `kernel.set_freeze_steering_weights(true)`
  before the first dispatch. Nothing else differs (INV-07).
- **Per-brain-tick measurement** (only with `--food-kinds`; snapshot and log reads only,
  INV-06). Whitened inputs exist only in brain state, so a read-only per-agent copy via
  `GpuKernel::read_agent_state` each brain tick is a permitted snapshot; the probe stays
  as is. Per agent:
  - nearest in-range nourishing and harmful items via `nearest_food_of_kind` at
    `SCENT_RANGE`; nourishing kind is `kernel.nourishing_food_kind()`, harmful is its
    opposite. Feed `KindTurnAccumulator::record` with the executed turn and both bearings.
  - A `ReadoutSample` when an item is in range. Features are all declared senses'
    whitened inputs through BrainLayout offsets (ten smell and visual inputs plus two
    `second_odour` inputs). Label is true when the nearest in-range item is nourishing.
  - `PairCreditAccumulator::record` once per odour pair with the brain's own TD
    error and fresh turn-noise innovation from that tick's 0001 probe sample
    (`FreeLifeSample.td_error`, `FreeLifeSample.turn_innovation`), plus that pair's
    whitened right and left inputs read through `BrainLayout` offsets.
  - Cumulative `P_FOOD_COUNT` and `P_HARMFUL_MEAL_COUNT` at quarter boundaries, into
    `KindMealQuarters::from_cumulative`; deaths per agent.
- **End of run.** Each pair's final right and left turn weights form
  `signed_weight_difference`. One readout per agent via
  `fit_kind_readout(samples, shuffle_seed)`, where the seed comes from a named helper
  of run seed and agent index. Store `None` for an absent score, never 0.5.
- **Summary file.** `--food-kind-summary <file>` writes `RunFoodKindSummary` JSON with
  `seed`, `arm` (`learning`/`frozen`), `ticks`, `agent_count`, `fresh_brains`,
  `nourishing_kind`, `agents`, and prints one summary line. I/O and serde
  errors are returned; no `.expect()` or `.unwrap()` on these paths.
- **Default behaviour.** Without the three flags, code path, output and readbacks are
  unchanged.

## Steps

**Steps:**

1. Add the three flags to `main.rs`, the run-config struct and the CPU champion guard
   (an error before kernel construction) to `headless.rs`. Add test
   `food_kind_free_life_refuses_champion_brain`: a champion source gives an `Err`
   naming the food-kind flag. No adapter needed.
2. In the run path, set both config flags before kernel construction when
   `--food-kinds` is set, and call `set_freeze_steering_weights(true)` before the first
   dispatch when `--freeze-policy` is set. Add
   `food_kind_free_life_frozen_arm_keeps_turn_weights`: 2 agents, a few thousand ticks,
   frozen arm. Assert every `second_odour` turn weight is exactly `0.0` at the end and
   every smell and visual turn weight equals its birth value bit for bit. Run the
   learning arm with the same seed and assert the same `nourishing_kind`. Starts with
   `gpu_or_skip()`.
3. Implement the per-brain-tick measurement and per-agent accumulation as Decisions
   specifies, gated on `--food-kinds`; the default path does no extra readback.
4. Build `RunFoodKindSummary` and implement `--food-kind-summary <file>` with the JSON
   write and one-line print. Add `food_kind_free_life_summary_is_complete`: 2 agents, a
   few thousand ticks, `--food-kinds`; read the file back with serde and assert it
   parses, every float is finite, `agent_count` is 2, `fresh_brains` is true, and
   `nourishing_kind` equals the kernel's `nourishing_food_kind()` draw. Starts with
   `gpu_or_skip()`.

- **Done when:** the focused-proof check passes on the host with output reporting food_kind_free_life_refuses_champion_brain, food_kind_free_life_frozen_arm_keeps_turn_weights and food_kind_free_life_summary_is_complete as ok and zero failed, and INV-06 holds (measurement reads only snapshots and logs, and the only setter called is the freeze flag), INV-07 holds (the frozen arm ends with exactly zero second_odour turn weights and smell and visual turn weights bit-equal to birth values, with the same nourishing kind as the learning arm), INV-09 holds (a stored champion brain source returns an Err naming the food-kind flag before any GPU dispatch), INV-02 holds (the summary nourishing_kind equals the kernel nourishing_food_kind draw), and INV-10 holds (the check sets XAGENT_REQUIRE_GPU=1 and both GPU tests start with gpu_or_skip, which panics without an adapter).
