# XAgent Plan 0010 — Intent-Aware Fitness Hardening

Make Plan 0009's flag-gated machinery safe to graduate: add the brain metabolic
drain to the energy accumulator, re-derive the effort-fitness calibration from
real recorded telemetry so the foraging/exploration axes stop collapsing to ≈0 at
production scale, pin the calibration magnitudes and the production grid
denominator, rewrite the speed-decoupling gate to require a strongly-positive
baseline and strict improvement with a seeded paired A/B and real unit tests, fix
the avoidance-intent sign inversion and fused stale-telemetry ordering, gate the
danger scan and guard its `atan2`, and make the acceptance-named tests assert
their titular property while closing the parity/serde/migration/layout/magic-number
gaps — flipping no default.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none). This plan is
  authored for maximum parallelism: most tasks have no dependency and may branch
  concurrently; the only edges are where a test needs a code fix first or two
  tasks edit the same shader region and must merge in order.
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).
- **On-touch cleanup:** if a task edits a file region containing planning
  references (`plan 000N`, `Layer A/B/C/D`, `speed-decoupling`, task/workstream
  ids), strip them in the same commit and lower that file's row in
  `crates/xagent-sandbox/tests/contributing_guard.rs::PLANNING_REFERENCE_BASELINE`
  (the guard fails on both an over-count and an un-lowered under-count). Introduce
  zero new planning references in new code.

---

## 0001 — Effort-fitness scale-invariance & recalibration

### brain-drain-energy-accounting — Add Brain Metabolic Drain to `P_ENERGY_SPENT`

The energy accumulator that feeds the effort-rebased foraging axis records only
depletion + movement drain: `physics_state[b + P_ENERGY_SPENT] += depletion_drain
+ movement_drain;` in `agent_physics` (`kernel_tick.wgsl` ~186-189), while the
per-tick brain metabolic drain is subtracted from `energy` ~40 lines later
(`kernel_tick.wgsl:228`) without being accumulated. `phase_physics.wgsl` mirrors
the same omission (`:140-143`, `:178`). At `BrainConfig::large()` the omitted term
is ≈62% of depletion drain and scales with brain size, so the foraging score
ranks two equally-skilled foragers by brain size — a confound that breaks the
"effort-rebased = brain-agnostic" intent and must be fixed before recalibration.

**Steps:**
1. In `kernel_tick.wgsl` `agent_physics`, at the site the brain metabolic drain
   `(METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots *
   METABOLIC_PROCESSING_COST) * metabolic_rate` is subtracted from `energy`
   (`:228`), bind it to a `let brain_drain = ...;`, subtract that, and immediately
   `physics_state[b + P_ENERGY_SPENT] += brain_drain;`.
2. Apply the byte-identical edit in `phase_physics.wgsl` (`:178`).
3. Add the GPU test below to `crates/xagent-sandbox/tests/integration.rs`.

```rust
#[test]
fn energy_spent_includes_brain_drain() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Two agents, identical food/movement, BrainConfig::default() vs large():
    // P_ENERGY_SPENT must differ by the analytic brain-drain delta, and the
    // fused and split paths must record byte-identical accumulators.
    // (Drive a fixed-tick run; assert delta == expected within f32 tolerance.)
}
```

- **Depends on:** —
- **Done when:** `energy_spent_includes_brain_drain` fails before the change and
  passes after; `split_serial_matches_fused_serial` stays green (both paths edited
  identically); cargo fmt/clippy/test green.

### recorded-generation-calibration-replay — Replay Real Telemetry Through Old-vs-New Fitness

Plan 0009's `fitness-calibration-replay` task spec required replaying a recorded
generation's telemetry through old-vs-new `composite_fitness`, but the landed test
`fitness_calibration_replay_profiles` (`governor.rs:2351-2384`) substituted three
synthetic per-life profiles whose energy/distance are 2–3 orders of magnitude
smaller than the real respawn-preserved per-generation accumulators. The recording
machinery (`store_recording` `:1375`, `load_recording` `:1472`,
`generation_recording` table `:1801-1808`, format v2) exists and is unused. This
task runs the real replay and records the production-scale distributions that the
recalibration needs.

**Steps:**
1. Add a replay path (test or `--`-gated harness) that loads a recorded generation
   via `load_recording` and computes both legacy and effort-rebased
   `composite_fitness` over the real per-agent `energy_spent`/`distance_traveled`/
   `food_consumed`/`cells_explored`/`ticks_alive`.
2. Record the real distributions (min/mean/max of `energy_spent`,
   `distance_traveled`, and the resulting `food/energy`, `cells_per_dist`) and the
   per-axis foraging/exploration values under the current constants.
3. Write `docs/plans/0010-Intent-Aware-Fitness-Hardening/0010-FITNESS-RECALIBRATION-DECISION.md`
   capturing those numbers, confirming the ≈0.02 axis collapse under the current
   `FORAGING_ENERGY_TARGET=0.5`/`EXPLORATION_DISTANCE_BUDGET=16.0`, and selecting
   Variant A (re-pick constants) or Variant B (per-tick-rate scale-invariant axes,
   preferred if A cannot reach the production-scale assertion) per
   [ARCHITECTURE.md](ARCHITECTURE.md) §0001.

- **Depends on:** brain-drain-energy-accounting
- **Done when:** the decision doc records the real production-scale distributions
  and a chosen variant with its re-derived constants/formulas; the replay path is
  reproducible by a documented command; cargo fmt/clippy/test green. (This task is
  a measurement + decision doc; the constant change lands in the next task.)

### effort-axes-recalibration — Apply the Re-derived Scale-Invariant Calibration

The foraging axis `min((food_consumed / energy) / FORAGING_ENERGY_TARGET, 1.0)`
(`governor.rs:133`, weight 0.85) and exploration `cells_per_dist` axis
(`governor.rs:136-139`, budget 16.0) collapse to ≈0.02 across the whole population
at production tick budgets. This task applies the variant chosen in the decision
doc so a competent forager saturates the axis on real telemetry.

**Steps:**
1. Apply the chosen variant from `0010-FITNESS-RECALIBRATION-DECISION.md`: either
   set `FORAGING_ENERGY_TARGET`/`EXPLORATION_DISTANCE_BUDGET` (`governor.rs:52,62`)
   to the re-derived values, or change the axis formulas in `composite_fitness`
   (`governor.rs` ~132-139) to the per-tick-rate scale-invariant form, with
   doc-comments stating the production-scale rationale.
2. Add the CPU test below.

```rust
#[test]
fn competent_forager_saturates_foraging_on_real_scale() {
    // A production-magnitude telemetry profile (energy ~1.5e4, distance ~6e5,
    // food/cells from the recorded competent forager) must yield foraging ~= 1.0
    // (>= 0.95) and exploration not pinned to ~0 under the re-derived calibration.
    // Fails under the old constants (axis ~= 0.02), passes after recalibration.
}
```

- **Depends on:** recorded-generation-calibration-replay
- **Done when:** `competent_forager_saturates_foraging_on_real_scale` fails before
  the change and passes after; defaults unchanged (`effort_rebased_fitness=false`
  still reproduces legacy scores — existing no-op test stays green); cargo
  fmt/clippy/test green.

### calibration-test-falsifiability — Pin Magnitudes and the Production Grid Denominator

`fitness_calibration_replay_profiles` asserts only `effort_competent >
effort_aimless` and `effort_camper < 1.0` (`governor.rs:2503-2516`); the documented
composites/deltas live only in `eprintln!` (`:2415-2498`), and the test uses `grid
= 1000.0` (`:2352`) where production `evaluate` uses `total_grid_cells =
(HEATMAP_RES * HEATMAP_RES / 4) as f32 = 1024` (`:656`). A math regression that
preserves ordering passes today.

**Steps:**
1. Replace `let grid = 1000.0_f32;` (`governor.rs:2352`) with
   `let grid = (HEATMAP_RES * HEATMAP_RES / 4) as f32;`.
2. Add tolerance asserts (e.g. `(effort_aimless - <doc>).abs() < 1e-3`, and the
   competent–aimless gap) against the magnitudes recorded in
   `0010-FITNESS-RECALIBRATION-DECISION.md` (the re-derived numbers, not the stale
   0009 values).
3. Update `0009-FITNESS-CALIBRATION.md`'s `600/1000` reference to `600/1024` for
   consistency, or note the rounding (docs-only edit alongside the test).

- **Depends on:** effort-axes-recalibration
- **Done when:** the test pins the documented composites/deltas to tolerance and
  uses grid `1024`; a deltas-preserving-but-magnitude-shifting regression fails;
  cargo fmt/clippy/test green.

---

## 0002 — Decision-machinery hardening

### harden-gate-predicate — Require Strong Baseline, Strict Improvement, and an Uncapped Viability Metric

The gate is `speed_decoupled = on_stats.speed_fitness_correlation.abs() < 0.3`
with `gate_passed = speed_decoupled && ticks_alive_ok && danger_retained`
(`headless.rs:947,951`). It never requires the baseline to be strongly positive or
ON to improve, so it printed PASS while the committed doc's correlation **rose**
0.2238 → 0.2569. `ticks_alive_ok = on_mean > baseline_mean * 80 / 100`
(`headless.rs:948`) is uninformative because both arms saturate `tick_budget =
1_000_000` (`config.rs:429`; `P_TICKS_ALIVE` preserved across respawn).

**Steps:**
1. In `format_validation_markdown` (`headless.rs` ~947-951), replace the gate with
   the baseline-and-improvement predicate from [ARCHITECTURE.md](ARCHITECTURE.md)
   §0002 (named consts `BASELINE_CORR_MIN`, `DECOUPLE_CORR_MAX`, `DECOUPLE_MARGIN`),
   and report "inconclusive" (not PASS) when the baseline is below `BASELINE_CORR_MIN`.
2. Replace `ticks_alive_ok` with an uncapped viability metric — mean death-count
   per generation (or mean longest-life) — that is not pinned to `tick_budget`;
   wire it into `gate_passed`.
3. Make the criterion-1 prose derive from `corr_direction`, so it cannot print
   "falls … PASS" when the correlation rose.

- **Depends on:** —
- **Done when:** the gate requires a strongly-positive baseline AND strict
  improvement AND an uncapped viability metric; the criterion text matches
  `corr_direction`; verified by `gate-machinery-unit-tests`; cargo fmt/clippy/test green.

### seeded-paired-ab-harness — Make the Baseline vs ON A/B a Seeded Paired Comparison

Baseline and ON are independent `run_headless_with_flags` calls
(`headless.rs:423,427`); the three population/brain RNG sites are unseeded —
`mutate_config_with_strength` (`agent/mod.rs:378`), `mutate_brain_state`
(`agent/mod.rs:500`), and `reset_agents` (`gpu_kernel.rs:523`) — so the arms get
different genomes and brain weights and cross-arm deltas cannot be attributed to
the flags. A deterministic `reset_agents_seeded` exists (`gpu_kernel.rs:516-520`)
but the harness calls plain `reset_agents` (`headless.rs:587`).

**Steps:**
1. Add seeded entry points (or `&mut impl Rng` params) for `mutate_config` and
   `mutate_brain_state` in `agent/mod.rs`, deriving the seed from `config.world.seed`.
2. In the harness, switch `reset_agents` → `reset_agents_seeded` and thread the
   same per-arm seed through both arms so they draw identical randomness.
3. Add the test below.

```rust
#[test]
fn seeded_ab_arms_are_paired() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Two ON-flag runs with the same world seed produce identical initial genomes
    // and brain-state draws (assert byte-equal population config + brain_state).
}
```

- **Depends on:** —
- **Done when:** both arms share identical random draws given a fixed seed
  (`seeded_ab_arms_are_paired` passes); the harness uses `reset_agents_seeded`;
  cargo fmt/clippy/test green.

### gate-machinery-unit-tests — Unit-Test the Correlation, Regression, and Gate Math

`compute_correlation`, `compute_regression`, `format_validation_markdown`, and the
gate booleans are reachable only via `--validate-speed-decoupling` and have zero
unit tests (no `#[test]`/`#[cfg(test)]` references them). A regression in the
correlation denominator guard or the boolean conjunction would not be caught.

**Steps:**
1. Add GPU-free unit tests in `headless.rs` (a `#[cfg(test)] mod tests`):
   - `compute_correlation_extremes`: returns +1 / −1 on perfectly (anti)correlated
     vectors, and reports insufficient-variance (not a silent 0 that could pass a
     gate) on constant x.
   - `compute_regression_slope`: known slope on a linear series.
   - `gate_rejects_rising_correlation`, `gate_rejects_weak_baseline`,
     `gate_rejects_tick_collapse`, `gate_rejects_danger_zero`: the gate returns
     NOT-PASSED for each failure mode and PASSED only when all conjuncts hold.

- **Depends on:** harden-gate-predicate
- **Done when:** the unit tests exercise the correlation/regression/gate math with
  no GPU dependency and fail if any conjunct or the correlation guard regresses;
  cargo fmt/clippy/test green.

### gate-includes-avoidance-floor — Replace the Tautological Avoidance Conjunct With a Real, Included Floor

`avoidance_retained = on_stats.mean_avoidance_intent_fraction >= 0.0`
(`headless.rs:950`) is always true and is **excluded** from `gate_passed`
(`:951`). After the `0003` sign fix the metric finally measures genuine avoidance,
so the gate can require it to clear a meaningful floor.

**Steps:**
1. In `format_validation_markdown`, replace `avoidance_retained` with
   `avoidance_above_chance = on_stats.mean_avoidance_intent_fraction >=
   AVOIDANCE_FLOOR` (named const with a doc-comment justifying the floor), and add
   it to the `gate_passed` conjunction.
2. Remove the vacuous "avoidance intent non-negative: PASS" line from the rendered
   doc; render the real threshold check instead.

- **Depends on:** harden-gate-predicate, avoidance-intent-sign-timing
- **Done when:** the avoidance conjunct uses the corrected metric against a real
  floor and is part of `gate_passed`; covered by a `gate_rejects_*` unit test;
  cargo fmt/clippy/test green.

### share-danger-reduction — Unify the Danger/Avoidance Reductions Across Production and Harness

Production persists `danger_dwell_fraction = sum(danger_path)/sum(distance)` and
`avoidance_intent_fraction = sum(turns_opposing)/sum(sense_ticks)` (population
sum/sum, `governor.rs:832-844`), while the harness averages per-agent clamped
ratios (`headless.rs:768-793`). These are different statistics that diverge under
skewed distributions; the headless comment claiming it mirrors the governor is
inaccurate.

**Steps:**
1. Extract the sum/sum reduction (the more defensible population statistic) into a
   shared helper and call it from both `persist_behavior_metrics` (`governor.rs`)
   and the harness (`headless.rs`), mirroring how `quarter_food_rates` is already
   shared.
2. Add a CPU test asserting both call sites produce the identical value on a fixed
   per-agent fixture.

- **Depends on:** —
- **Done when:** both paths compute the danger/avoidance fractions through one
  shared reducer (verified equal on a fixture); cargo fmt/clippy/test green.

### decision-doc-metadata-and-temp-cleanup — Self-Describing Decision Doc and Complete Temp-DB Cleanup

`format_validation_markdown` (`headless.rs` ~946-1133) never emits `num_generations`,
the world seed, or population N, so the "canonical baseline" record is not
self-describing (the CLI default `validation_generations` is 10 at `main.rs:125`
but the committed doc shows 5). The temp-DB cleanup removes only the base file
(`headless.rs:796`), leaking `-wal`/`-shm` sidecars.

**Steps:**
1. Add `num_generations`, the world seed, and population N (ideally per-arm speed
   variance) to the markdown header in `format_validation_markdown`.
2. In the cleanup (`headless.rs:796`), also `remove_file` the `-wal` and `-shm`
   sidecars for each temp DB.

- **Depends on:** —
- **Done when:** the generated markdown header records generations/seed/N and the
  cleanup removes the sidecars; cargo fmt/clippy/test green. (Documentation/tooling
  task; no behavioral test required beyond fmt/clippy/test staying green.)

---

## 0003 — Danger-percept correctness

### avoidance-intent-sign-timing — Fix the Avoidance-Intent Sign Inversion and Fused Stale-Telemetry Ordering

The counter increments on `(motor_turn * danger_bearing) < 0.0`
(`kernel_tick.wgsl:258-260`), but `danger_bearing = atan2(facing_x*to_danger.z -
facing_z*to_danger.x, facing·to_danger)` (`:319-321`) is **negative** for
right-side danger while positive `motor_turn` turns right (`:86`, `TURN_SPEED=3.0`
`common.wgsl:375`), so a genuine turn-away gives a positive product and is not
counted — the metric counts turning *toward* danger. The `:256` comment is
backwards. In the fused path `agent_physics` (counter) runs before
`agent_danger_detect` (`:803-805`), so the counter reads previous-cycle danger;
the split `phase_physics.wgsl` scans first then increments (`:197-247`, `:261-263`).
The counter is observability-only — it never reaches `composite_fitness`
(`governor.rs:115-124`) — so this corrects a metric, not selection.

**Steps:**
1. In `kernel_tick.wgsl`, change the increment condition to `let turn_away =
   (motor_turn * danger_bearing) > 0.0;` and replace the backwards `:256` comment
   with the corrected convention from [ARCHITECTURE.md](ARCHITECTURE.md) §0003.
2. Move the fused avoidance-counter accumulation so it runs **after**
   `agent_danger_detect` writes `P_NEAREST_DANGER_*` that cycle (read same-cycle
   telemetry), keeping behavior consistent with the split path.
3. Apply the byte-identical sign fix in `phase_physics.wgsl` (`:261-263`); the
   split path's ordering already reads fresh telemetry, so only the sign changes
   there.

- **Depends on:** —
- **Done when:** the counter increments on a genuine turn-away (proven by
  `danger-percept-gpu-tests`); `split_serial_matches_fused_serial` stays green;
  cargo fmt/clippy/test green.

### danger-percept-gpu-tests — Assert the Avoidance Counter and the Near-Agent Bearing

The away-turn semantics need a direct GPU test, and the existing
`nearest_danger_bearing_points_at_danger` (`integration.rs:6403`) only `eprintln!`s
the near agent's bearing (`:6470`) and asserts bearing solely for the far agent
(`:6497`) — a wrong/garbage/NaN near-agent bearing passes today.

**Steps:**
1. Add `avoidance_counter_increments_only_on_turn_away` (GPU): place danger on a
   known side, drive a known `motor_turn`, and assert `P_AVOIDANCE_TURNS_OPPOSING`
   increments only for the away-turn and not for the toward-turn; assert fused ==
   split.
2. Repair `nearest_danger_bearing_points_at_danger`: compute the analytic
   facing-relative bearing for the near agent (seed-42 facing, agent at (0,y,2.0),
   danger at origin) and assert it falls within the expected angular window (at
   minimum finite and non-sentinel), in addition to the existing distance assert.

```rust
#[test]
fn avoidance_counter_increments_only_on_turn_away() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Danger on the right (+x), facing +z. A left turn (motor_turn<0) is an
    // avoidance turn -> counter increments. A right turn (motor_turn>0) steers
    // into danger -> counter does NOT increment. Fused and split agree.
}
```

- **Depends on:** avoidance-intent-sign-timing
- **Done when:** both tests fail against the pre-fix sign/assertion and pass after;
  cargo fmt/clippy/test green.

### danger-scan-gate-and-atan2-guard — Gate the Ring-Scan Behind the Flag and Guard `atan2(0,0)`

`agent_danger_detect` runs an O((DANGER_SENSE_RADIUS/cell)²) biome-grid scan +
`atan2` every cycle and is called with no flag guard (`kernel_tick.wgsl:803-805`;
mirrored `phase_physics.wgsl:197-247`); `danger_percept_enabled` gates only feature
packing (`brain_passes.wgsl:239`), so the default build pays a permanent per-tick
cost — byte-identical in OUTPUT but not COMPUTE. The bearing `atan2(cross_y,
dot_val)` (`:319-321`) has no `dist > EPSILON` guard, so an agent on a danger-cell
center hits WGSL-indeterminate `atan2(0,0)`.

**Steps:**
1. Wrap the entire `agent_danger_detect` scan and its dependent counter update in
   `if (WC_DANGER_PERCEPT_ENABLED != 0u) { ... }` in `kernel_tick.wgsl` (call site
   `:803-805`) and `phase_physics.wgsl` (`:197-247`), so the default build is a
   true compute no-op.
2. Add `dist > EPSILON` to the nearest-danger acceptance test (`kernel_tick.wgsl:310`,
   `phase_physics.wgsl:224`) before the `atan2`, identically in both shaders.

- **Depends on:** avoidance-intent-sign-timing
- **Done when:** with `danger_percept_enabled=false` the scan body is not executed
  (flag-off determinism preserved, verified by `flag-off-byte-identity-golden`) and
  no `atan2(0,0)` is reachable; `split_serial_matches_fused_serial` and the
  percept-on tests stay green; cargo fmt/clippy/test green.

---

## 0004 — Test, layout & migration integrity

### recorded-telemetry-persistence-test — Make the Persistence Test Exercise the Real Path

`recorded_telemetry_persists_in_agent_fitness` (`integration.rs:6296`) copies GPU
slots into agent-struct fields (`:6369`) then asserts the copies `> 0.0`
(`:6378,6384`) — tautological; it never constructs an `AgentFitness`, calls
`governor.evaluate`, or touches the DB.

**Steps:**
1. Either drive the real transfer — call `governor.evaluate` and assert the
   resulting `AgentFitness` fields (or a DB round-trip via the `agent_result`
   query) carry the telemetry — or rename the test to
   `gpu_slots_populate_during_generation` to reflect that it only checks slot
   population. Prefer the real transfer.

- **Depends on:** —
- **Done when:** the test exercises `evaluate`→`AgentFitness` (or a DB round-trip),
  or is renamed to its true scope; cargo fmt/clippy/test green.

### flag-off-byte-identity-golden — Pin a Flag-Off Golden Instead of Self-Comparison

`danger_percept_byte_identical_when_flag_off` (`integration.rs:3938`) asserts only
`off_phys_a == off_phys_b` and `off_brain_a == off_brain_b` (`:4010,4014`) —
run-to-run determinism, not byte-identity to a pre-percept build, despite the
docstring's claim. A uniform flag-off regression passes.

**Steps:**
1. Capture a golden encoded-state vector from a flag-off run and assert flag-off
   matches *it* (as `visual_cortex_passthrough_*` does for the encoder), or correct
   the docstring to state it proves determinism only. Prefer the golden.

- **Depends on:** danger-scan-gate-and-atan2-guard
- **Done when:** the test pins a flag-off golden (or its docstring is corrected to
  determinism-only); cargo fmt/clippy/test green.

### wc-cfg-constant-parity — Close the `WC_*`/`CFG_*` Rust↔WGSL Parity Gaps

`shader_config_constants_match_rust` stops at `CFG_VISUAL_CORTEX_ENABLED`
(`buffers.rs:1257-1270`) and never asserts `CFG_DANGER_PERCEPT_ENABLED`; the WC
in-bounds block stops at `WC_SPEED_COST_EXPONENT` (`:1496-1520`) and omits
`WC_DANGER_PERCEPT_ENABLED`; there is no `shader_wc_constants_match_rust` parity
test. A one-file renumber to a padding slot would read 0.0 and every test would
still pass.

**Steps:**
1. Add `CFG_DANGER_PERCEPT_ENABLED` to `shader_config_constants_match_rust`.
2. Add `assert!(WC_DANGER_PERCEPT_ENABLED < WORLD_CONFIG_SIZE)` to the WC in-bounds
   block.
3. Add `shader_wc_constants_match_rust` mirroring `shader_phys_constants_match_rust`,
   asserting Rust↔WGSL index parity for the WC family including
   `WC_SPEED_COST_EXPONENT`/`WC_DANGER_PERCEPT_ENABLED`.

- **Depends on:** —
- **Done when:** the new asserts/test fail under a deliberately renumbered WC/CFG
  constant and pass as shipped; cargo fmt/clippy/test green.

### serde-default-off-coverage — Assert New Flags Deserialize to `false`

No test asserts `danger_percept_enabled`/`effort_rebased_fitness` default to
`false` from a legacy/missing-field blob; the existing
`legacy_config_without_visual_cortex_fields_still_loads` (`config.rs:712-757`)
covers only the visual fields. Both rely on bare `#[serde(default)]`.

**Steps:**
1. Extend the legacy-blob deserialization test (or add a sibling) to assert
   `!d.danger_percept_enabled` and `!d.effort_rebased_fitness` from a config JSON
   missing those keys.

- **Depends on:** —
- **Done when:** the test fails if either `#[serde(default)]` is dropped and passes
  as shipped; cargo fmt/clippy/test green.

### behavior-metric-migration-order — Order the `behavior_metric` Migration CREATE-then-ALTER

The `behavior_metric` `ALTER TABLE ... ADD COLUMN avoidance_intent_fraction`
(`governor.rs:1827`) runs **before** its `CREATE TABLE IF NOT EXISTS behavior_metric`
(`:1830-1843`) — the only migration whose ALTER precedes its CREATE, contrary to
the `agent_result`/`node` pattern. It is net-idempotent (the fresh-DB ALTER fails
and is swallowed) but inconsistent.

**Steps:**
1. Move the `behavior_metric` ALTER to after its CREATE, matching the
   CREATE-then-ALTER order used for the other tables.

- **Depends on:** —
- **Done when:** the ALTER follows the CREATE; the existing
  `behavior_metric_table_persists_danger_metrics` and migration tests stay green;
  cargo fmt/clippy/test green.

### stale-layout-doc-fixes — Correct the `PHYS_STRIDE` and World-Config Slot-Count Comments

`integration.rs:3732` says "offsets 36, 37, 38 within PHYS_STRIDE=39" (real value
`PHYS_STRIDE = 44`, `buffers.rs:224`); `buffers.rs:638` says `fill_world_config`
"Writes the 24 world-config slots" while it writes through
`WC_DANGER_PERCEPT_ENABLED=25` within `WORLD_CONFIG_SIZE = 28` (`:472`).

**Steps:**
1. Fix `integration.rs:3732` to `PHYS_STRIDE=44` (or phrase via the constant).
2. Fix the `fill_world_config` doc to the real slot count (prefer phrasing via
   `WORLD_CONFIG_SIZE`).

- **Depends on:** —
- **Done when:** both comments match the landed 44-wide layout / `WORLD_CONFIG_SIZE=28`;
  cargo fmt/clippy/test green. (Documentation-only.)

### avoidance-counter-persistence — Decide and Document Raw Avoidance-Counter Storage

The raw avoidance counters (`avoidance_sense_range_ticks`,
`avoidance_turns_opposing`) are persisted to neither `agent_result` (INSERT lists
only the three effort columns, `governor.rs:674-680`) nor `behavior_metric` (only
the derived `avoidance_intent_fraction`). Either is defensible; the gap should be a
decision, not an accident.

**Steps:**
1. Either add the two counters to `agent_result` (CREATE + idempotent ALTER +
   INSERT column + placeholder, preserving INSERT/VALUES order) with a round-trip
   test extension, or add an in-code comment at the INSERT site documenting why
   only the derived ratio is persisted.

- **Depends on:** —
- **Done when:** the persistence choice is explicit (columns + round-trip test, or
  a documented rationale at the INSERT); cargo fmt/clippy/test green.

### shader-magic-number-naming — Name the WGSL `255`, `1.414`, and `20.0` Literals

The biome index clamp `255` (`common.wgsl:640`, `kernel_tick.wgsl:301`,
`phase_physics.wgsl:215`), the √2 motor-magnitude clamp `1.414`
(`kernel_tick.wgsl:182`, `phase_physics.wgsl:136`), and the default move speed
`20.0` (duplicated across `kernel_tick.wgsl:208`, `phase_physics.wgsl:158`, and
aliased by `GRAVITY=20.0` `common.wgsl:374`) are un-named, violating the
named-constant rule and creating a silent grid-resolution / speed coupling.

**Steps:**
1. Add `const BIOME_GRID_MAX_INDEX: u32 = 255u;`, `const SQRT_2: f32 =
   1.41421356237;`, and a single `const DEFAULT_MOVE_SPEED: f32 = 20.0;` to
   `common.wgsl` (with doc-comments), and replace the bare literals at every site
   in `kernel_tick.wgsl`/`phase_physics.wgsl`/`common.wgsl`. Mirror in `buffers.rs`
   only if a Rust-side parity assert references the value.

- **Depends on:** avoidance-intent-sign-timing, danger-scan-gate-and-atan2-guard
- **Done when:** the named constants replace every bare `255`/`1.414`/`20.0` move-
  speed literal; the physics-equivalence and `shader_*_constants_match_rust` tests
  stay green (byte-identical values); cargo fmt/clippy/test green.

---

**End of plan 0010 TASKS.** When every "Done when" bullet is green, the effort
calibration is real and scale-invariant, the speed-decoupling gate is trustworthy
and unit-tested, the danger percept measures genuine avoidance, and the
acceptance-named tests assert their titular property — making `0009`'s
`default-flip-gate` decidable on evidence.
