# Architecture — Plan 0010 (deltas)

> Edits in `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/src/agent/mod.rs`,
> `crates/xagent-sandbox/src/main.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_physics.wgsl`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-sandbox/tests/integration.rs`, and a new decision doc
> `docs/plans/0010-Intent-Aware-Fitness-Hardening/0010-FITNESS-RECALIBRATION-DECISION.md`.
> Line numbers are hints; locate by symbol.

## 0001 — Effort-fitness scale-invariance & recalibration

Today `composite_fitness` (`governor.rs` ~115-155) computes, under the
`effort_rebased_fitness` branch (`:129`), `foraging = ((food_consumed / energy) /
FORAGING_ENERGY_TARGET).min(1.0)` with `energy = energy_spent.max(ENERGY_FLOOR)`
(`:132-133`) and `exploration = coverage.min(cells_per_dist)` with `cells_per_dist
= (cells_explored / (dist / EXPLORATION_DISTANCE_BUDGET)).min(1.0)` (`:135-139`).
`FORAGING_ENERGY_TARGET = 0.5` (`:52`) and `EXPLORATION_DISTANCE_BUDGET = 16.0`
(`:62`) were calibrated against synthetic per-life profiles
(`fitness_calibration_replay_profiles`, `:2351-2384`) whose `energy`/`distance`
(180/600, 800/4000, 60/1) are 2–3 orders of magnitude smaller than the real
respawn-preserved per-generation accumulators. The calibration test asserts only
ordering (`:2503-2516`) and uses `grid = 1000.0` (`:2352`) where production uses
`total_grid_cells = (HEATMAP_RES * HEATMAP_RES / 4) as f32 = 1024` (`:656`).

### Brain-drain energy accounting

Today `kernel_tick.wgsl` (~186-189) records `physics_state[b + P_ENERGY_SPENT] +=
depletion_drain + movement_drain;` then later subtracts the brain metabolic drain
from `energy` without accumulating it (`:228`); `phase_physics.wgsl` mirrors this
(`:140-143`, `:178`).

Edits:

- **Accumulate the full per-tick energy delta** (`kernel_tick.wgsl` ~186-228,
  `phase_physics.wgsl` ~140-178): add the brain metabolic drain term to
  `P_ENERGY_SPENT` at the same site it is subtracted from `energy`, identically in
  both shaders, so the slot equals the true total energy burned.

```wgsl
// Per-tick energy burned == per-tick energy delta. The brain metabolic drain is
// part of the cost an effort-denominated foraging score must see; omitting it
// makes the score scale with brain size instead of skill.
let brain_drain = (METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST
    + proc_slots * METABOLIC_PROCESSING_COST) * metabolic_rate;
energy -= brain_drain;
physics_state[b + P_ENERGY_SPENT] += brain_drain;
```

Properties that make this safe:
- The edit is byte-identical between the two shaders, so
  `split_serial_matches_fused_serial` (full 44-slot vector) keeps passing.
- `metabolic_rate` still cancels in the `food/energy` ratio, so the *direction* of
  the foraging axis is unchanged; only the denominator's magnitude and its
  brain-size dependence change — which is the point.
- The accumulator stays respawn-preserved (whitelist untouched).

### Recorded-generation recalibration

Today the replay machinery (`store_recording` `:1375`, `load_recording` `:1472`,
`generation_recording` table `:1801-1808`, format v2) exists but the calibration
never uses it.

Edits:

- **Real-telemetry recalibration** (`governor.rs` constants `:52,:62`; new
  decision doc): run a recorded generation through old-vs-new `composite_fitness`,
  read the real `energy_spent`/`distance_traveled` distributions, and either
  (Variant A) re-pick `FORAGING_ENERGY_TARGET`/`EXPLORATION_DISTANCE_BUDGET`
  against those distributions, or (Variant B, preferred if A cannot reach the
  production-scale assertion) make the axes scale-invariant by denominating on a
  per-tick rate, e.g. `food_consumed / (energy_spent / ticks_alive)` and
  `cells_explored / (distance_traveled / ticks_alive)`. The chosen variant and the
  re-derived constants are recorded in `0010-FITNESS-RECALIBRATION-DECISION.md`.

```rust
/// Food-per-energy ratio that earns full foraging credit, re-derived from real
/// recorded cumulative telemetry (not synthetic per-life profiles) so a competent
/// forager saturates the axis at production tick budgets.
const FORAGING_ENERGY_TARGET: f32 = /* set by 0010-FITNESS-RECALIBRATION-DECISION.md */;
```

Properties that make this safe:
- Dormant behind `effort_rebased_fitness = false` until graduated, so no shipped
  default changes.
- A production-scale assertion (a competent forager on real telemetry reaches
  foraging ≈ 1.0) is added so a future scale regression fails `cargo test`.

### Calibration test falsifiability

Edits:

- **Pin magnitudes and the production grid** (`fitness_calibration_replay_profiles`
  asserts `:2503-2516`, grid `:2352`): replace `grid = 1000.0` with
  `(HEATMAP_RES * HEATMAP_RES / 4) as f32` and add tolerance asserts on the
  re-derived composites/deltas so a math regression that preserves ordering still
  fails. (Numbers come from the recalibration decision doc, not the stale 0009
  values.)

## 0002 — Decision-machinery hardening

Today `format_validation_markdown` (`headless.rs` ~946-1133) computes
`speed_decoupled = on_stats.speed_fitness_correlation.abs() < 0.3` (`:947`),
`ticks_alive_ok = on_stats.mean_ticks_alive > baseline.mean_ticks_alive * 80 / 100`
(`:948`), `avoidance_retained = on_stats.mean_avoidance_intent_fraction >= 0.0`
(`:950`), and `gate_passed = speed_decoupled && ticks_alive_ok && danger_retained`
(`:951`). `corr_delta`/`corr_direction` (`:969-970`) feed only prose. The two arms
are independent `run_headless_with_flags` calls (`:423,:427`); none of the three
RNG sites (`agent/mod.rs:378,500`, `gpu_kernel.rs:523`) is seeded, though
`reset_agents_seeded` exists (`gpu_kernel.rs:516-520`). `tick_budget` defaults to
`1_000_000` (`config.rs:429`). None of `compute_correlation`,
`compute_regression`, or the gate has a unit test.

Edits:

- **Gate predicate** (`headless.rs` ~947-951): require an exploitable baseline and
  strict improvement, and make the danger-retention conjunct meaningful and
  *included*.

```rust
/// The gate only certifies decoupling when the baseline actually had a speed
/// exploit to remove (strongly-positive correlation) AND the ON arm strictly
/// reduces it below threshold. A baseline already below threshold is reported
/// "inconclusive", never PASS.
const BASELINE_CORR_MIN: f32 = 0.3;          // baseline must be exploitable
const DECOUPLE_CORR_MAX: f32 = 0.3;          // ON must fall below this
const DECOUPLE_MARGIN: f32 = 0.02;           // and strictly below baseline by a margin
let speed_decoupled = baseline.speed_fitness_correlation.abs() >= BASELINE_CORR_MIN
    && on_stats.speed_fitness_correlation.abs() < DECOUPLE_CORR_MAX
    && on_stats.speed_fitness_correlation.abs()
        <= baseline.speed_fitness_correlation.abs() - DECOUPLE_MARGIN;
```

- **Viability conjunct** (`headless.rs` ~948): replace the budget-saturated
  `ticks_alive_ok` with an uncapped metric — mean death-count per generation (or
  mean longest-life), which is not pinned to `tick_budget`.
- **Avoidance conjunct** (`headless.rs` ~950-951): after the `0003` sign fix,
  replace the tautology with `avoidance_above_chance = on_stats.mean_avoidance_intent_fraction
  >= AVOIDANCE_FLOOR` and add it to `gate_passed`.
- **Seeded paired A/B** (`headless.rs` ~423-587, `agent/mod.rs:378,500`,
  `gpu_kernel.rs:523`): thread a fixed seed derived from `config.world.seed`
  through the config mutation, brain-state mutation, and `reset_agents_seeded` so
  both arms draw identical randomness and only the flags differ. Add seeded
  variants of `mutate_config`/`mutate_brain_state` (or pass an `&mut impl Rng`).
- **Decision-doc metadata** (`headless.rs` ~946-1133, `main.rs:125`): emit
  `num_generations`, the world seed, and population N into the markdown header, and
  remove the `-wal`/`-shm` sidecars alongside the base temp DB (`headless.rs:796`).

Properties that make this safe:
- All of `0002` is offline advisory tooling behind `--validate-speed-decoupling`;
  it never alters runtime/selection behavior. The risk it removes is a misleading
  decision record, not a shipped regression.
- The seeded path reuses the existing `reset_agents_seeded`; the world
  (terrain/food/spawn) is already seeded, so this only makes genome/brain draws
  paired.

## 0003 — Danger-percept correctness

Today the avoidance counter increments on `(motor_turn * danger_bearing) < 0.0`
(`kernel_tick.wgsl:258-260`, `phase_physics.wgsl:261-263`), where `danger_bearing
= atan2(facing_x*to_danger.z - facing_z*to_danger.x, facing·to_danger)`
(`kernel_tick.wgsl:319-321`). Positive `motor_turn` turns right
(`kernel_tick.wgsl:86`, `TURN_SPEED=3.0` `common.wgsl:375`) but right-side danger
yields a **negative** bearing, so the product is negative when the agent turns
*toward* danger — inverted. In the fused path `agent_physics` (counter increment)
runs before `agent_danger_detect` (`:803-805`), reading previous-cycle danger. The
scan runs unconditionally (no `WC_DANGER_PERCEPT_ENABLED` guard at `:803-805`), and
the `atan2` has no `dist > EPSILON` guard.

Edits:

- **Sign + timing** (`kernel_tick.wgsl` ~254-260,790-805; `phase_physics.wgsl`
  ~257-263): correct the convention so a genuine turn-away increments the counter,
  and move the fused counter accumulation to after `agent_danger_detect` so it
  reads same-cycle danger. Fix the backwards `:256` comment. Keep both paths
  consistent.

```wgsl
// danger_bearing is the signed facing-relative angle to the nearest danger:
// NEGATIVE = danger to the right (positive motor_turn turns right), POSITIVE =
// danger to the left. A genuine turn-AWAY rotates against the bearing, so the
// product (motor_turn * danger_bearing) is POSITIVE for an avoidance turn.
let turn_away = (motor_turn * danger_bearing) > 0.0;
```

- **Flag-gate the scan** (`kernel_tick.wgsl:803-805`, `phase_physics.wgsl:197-247`):
  wrap the entire `agent_danger_detect` scan (and the counter update that depends
  on it) in `if (WC_DANGER_PERCEPT_ENABLED != 0u) { ... }` so the default build is
  a true compute no-op.
- **`atan2` guard** (`kernel_tick.wgsl:310-321`, `phase_physics.wgsl:224-235`): add
  `dist > EPSILON` to the nearest-danger acceptance test before computing the
  bearing, so an agent standing on a danger-cell center never reaches `atan2(0,0)`.

Properties that make this safe:
- The default-off encoded state stays byte-identical (the flag now gates COMPUTE
  too), verified by the existing flag-off determinism test plus the `0004`
  golden-or-rename hardening.
- Sign/timing/guard edits are applied identically in both shaders, preserving the
  full-vector parity test.
- The counter remains observability-only; selection output is unchanged.

## 0004 — Test, layout & migration integrity

Today `nearest_danger_bearing_points_at_danger` asserts only distance for the near
agent and bearing only for the far agent (`integration.rs:6470,6474,6497`);
`recorded_telemetry_persists_in_agent_fitness` asserts copied GPU slots
(`:6369,6378`); `danger_percept_byte_identical_when_flag_off` compares two flag-off
runs (`:4010,4014`). `shader_config_constants_match_rust` stops at
`CFG_VISUAL_CORTEX_ENABLED` (`buffers.rs:1257-1270`) and the WC in-bounds block
stops at `WC_SPEED_COST_EXPONENT` (`:1496-1520`). The `behavior_metric` `ALTER`
precedes its `CREATE` (`governor.rs:1827` before `:1830`). `integration.rs:3732`
says `PHYS_STRIDE=39` (real 44); `buffers.rs:638` says "24 world-config slots"
(`WORLD_CONFIG_SIZE=28`). The WGSL literals `255`/`1.414`/`20.0` are un-named.

Edits:

- **Assert the titular properties** (`integration.rs`): for
  `nearest_danger_bearing_points_at_danger`, compute the analytic facing-relative
  bearing for the near agent and assert it falls in the expected angular window
  (and a new away-turn counter test from `0003`); for
  `recorded_telemetry_persists_in_agent_fitness`, drive `governor.evaluate` and
  assert the `AgentFitness` fields (or a DB round-trip), or rename to reflect it
  only checks GPU-slot population; for `danger_percept_byte_identical_when_flag_off`,
  pin a golden encoded-state vector captured from a flag-off build and assert
  against it, or rename to "determinism".
- **`WC_*`/`CFG_*` parity** (`buffers.rs` ~1257-1520): add
  `CFG_DANGER_PERCEPT_ENABLED` to `shader_config_constants_match_rust`, add a
  `shader_wc_constants_match_rust` test mirroring the P_* one, and add the missing
  `assert!(WC_DANGER_PERCEPT_ENABLED < WORLD_CONFIG_SIZE)`.
- **Serde default-off** (`config.rs` ~712-757): extend the legacy-blob test to
  assert `!d.danger_percept_enabled` and `!d.effort_rebased_fitness`.
- **Migration order** (`governor.rs` ~1825-1843): move the `behavior_metric` ALTER
  after its `CREATE`, matching `agent_result`/`node`.
- **Stale layout fixes** (`integration.rs:3732`, `buffers.rs:638`): correct the
  `PHYS_STRIDE=39 → 44` comment and the "24 → 28 world-config slots" doc (prefer
  phrasing counts via the constants).
- **Avoidance-counter persistence decision** (`governor.rs` ~674-680,1768-1823):
  either add `avoidance_sense_range_ticks`/`avoidance_turns_opposing` to
  `agent_result` (CREATE + idempotent ALTER + INSERT + a round-trip assert) or
  document in-code why only the derived `avoidance_intent_fraction` is persisted.
- **Name the WGSL magic numbers** (`common.wgsl`, `kernel_tick.wgsl`,
  `phase_physics.wgsl` + Rust mirror in `buffers.rs` where applicable): introduce
  `BIOME_GRID_MAX_INDEX = 255u`, `SQRT_2 = 1.41421356237`, and a single
  `DEFAULT_MOVE_SPEED = 20.0` const, replacing the duplicated bare literals.

```wgsl
/// Biome grid is 256×256; the last valid index is 255. Naming it removes the
/// silent coupling between the clamp literal and the grid resolution.
const BIOME_GRID_MAX_INDEX: u32 = 255u;
/// Baseline locomotion speed; movement energy and the path-length-hazard
/// reference step are both normalized by it. One name, one source of truth.
const DEFAULT_MOVE_SPEED: f32 = 20.0;
```

Properties that make this safe:
- All of `0004` is test/doc/migration/constant-naming; the magic-number
  substitutions are exact-value replacements verified by the existing
  `shader_*_constants_match_rust` and physics-equivalence tests, so behavior is
  byte-identical.
- The migration reorder is net-idempotent either way (the `let _ =` swallows a
  fresh-DB ALTER); the change only restores the project's CREATE-then-ALTER
  convention.

## Test strategy

Named tests, each keeping `cargo fmt`/`clippy`/`cargo test -p xagent-sandbox`
green (GPU tests self-skip without an adapter via `GpuKernel::is_available()`):

- `energy_spent_includes_brain_drain` (GPU): two agents with `BrainConfig::default()`
  vs `BrainConfig::large()`, identical food/movement, must record `P_ENERGY_SPENT`
  differing by the analytic brain-drain delta; fused == split.
- `competent_forager_saturates_foraging_on_real_scale` (CPU): a recorded/replayed
  telemetry profile at production magnitudes yields foraging ≈ 1.0 under the
  re-derived constants — fails before recalibration, passes after.
- `fitness_calibration_pins_magnitudes` (CPU): tolerance asserts on the re-derived
  composites/deltas; uses grid `1024`.
- `gate_rejects_rising_correlation`, `gate_rejects_weak_baseline`,
  `gate_rejects_tick_collapse`, `compute_correlation_extremes` (CPU, GPU-free):
  the gate booleans and correlation math.
- `seeded_ab_arms_are_paired` (CPU/GPU): two ON-flag runs with the same seed
  produce identical genomes/brain draws.
- `avoidance_counter_increments_only_on_turn_away` (GPU): known `motor_turn` vs a
  known left/right danger placement increments only for the away-turn; fused ==
  split.
- `nearest_danger_bearing_points_at_danger` (repaired): asserts the near agent's
  bearing window.
- `danger_scan_is_compute_noop_when_flag_off` (GPU): flag-off encoded state matches
  a pinned golden.
- `shader_wc_constants_match_rust`, `legacy_blob_defaults_new_flags_false`.

CI gate (every task): `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p xagent-sandbox`.

## Interaction with prior work

- **Honors Plan `0009`'s locked decisions.** Danger stays graded and
  observability-only; both physics paths and both respawn whitelists keep moving
  together; defaults stay byte-identical no-ops; the default-flip stays GATED.
- **Completes the measurement Plan `0009` deferred.** `0001` runs the
  recorded-generation replay `0009`'s `fitness-calibration-replay` task spec
  required but substituted with synthetic profiles, and `0002` corrects the
  speed-decoupling gate that `0009`'s `speed-decoupling-validation` produced.
- **Respects the `2026-06-15` latency ceiling.** `0003`'s scan gating removes a
  permanent per-tick cost from the default build, which the brain-pass-latency
  review identified as the throughput owner; no dispatch/workgroup change is made.
- **Defers the wholesale planning-reference scrub to Plan `0011`** (disjoint file
  set), keeping the `contributing_guard.rs` baseline contention-free.
