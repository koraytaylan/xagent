# XAgent Plan 0009 — Intent-Aware Fitness

This plan stops evolution from winning by raw speed. It adds effort/exposure
telemetry, makes hazard damage proportional to path length through danger
(speed-invariant, never lethal), makes locomotion energy super-linear in speed
above baseline (single-peaked speed→fitness), re-bases foraging on energy spent
and exploration on distance traveled (so brute coverage stops paying), and gives
the agent a dedicated danger percept with a symmetric avoidance-shaping potential
and a sensed-then-turned intent metric — so deliberate foraging and deliberate
avoidance become the only ways to score, and avoidance is learnable and
measurable. All behind flags defaulting to a byte-identical no-op, graduated only
on a measured speed-decoupling.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/0009-intent-aware-fitness--{id}/`).
- **Depends on** lists *direct* prerequisites only (`—` means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (state as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter:

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- **Both physics paths move together:** every kernel mechanism edit lands in
  `kernel_tick.wgsl` AND `phase_physics.wgsl`; every new generation-cumulative
  slot is added to the respawn whitelist in `kernel_tick.wgsl` AND
  `phase_death.wgsl`. WGSL is `include_str!`'d — `cargo clean -p xagent-brain`
  after shader edits.
- All buffer offsets derive from `BrainLayout` / kernel constants — never
  hardcode a stride. Shared Rust↔WGSL constants have a single canonical source.
- Numeric safety: `try_into()` for lossy casts, `checked_mul` before sizing,
  `max(denominator, EPSILON)` before every division (Rust and WGSL).

---

## 0001 — Effort & exposure telemetry

### phys-accumulator-slots — Add The Three Cumulative Slots

Today `PHYS_STRIDE = 36` with the last slot `P_IN_DANGER_BIOME = 35`
(`buffers.rs:201-202`, `common.wgsl:327`). Add the effort/exposure accumulators.

**Steps:**
1. Add `P_DISTANCE_TRAVELED = 36`, `P_ENERGY_SPENT = 37`,
   `P_DANGER_PATH_LENGTH = 38` to `buffers.rs`; bump `PHYS_STRIDE` to `39`.
2. Mirror the three consts in `common.wgsl` and bump its `PHYS_STRIDE` to `39u`.
3. Extend the parity test `shader_phys_constants_match_rust` (`buffers.rs:1310-1356`):
   add the three offsets to the highest-offset list so the `PHYS_STRIDE ==
   max_offset + 1` assertion covers them.

- **Depends on:** —
- **Done when:** the parity test passes with `PHYS_STRIDE == 39`; cargo
  fmt/clippy/test green.

### effort-telemetry-fused — Accumulate Distance/Energy/Danger-Path (Fused)

Accumulate the three quantities in the fused kernel and preserve them across
respawn.

**Steps:**
1. In `kernel_tick.wgsl` physics tick, after the final position write
   (~`:156-158`), compute `step_len = length(vec2(pos.x-last_pos.x,
   pos.z-last_pos.z))` and `physics_state[b + P_DISTANCE_TRAVELED] += step_len;`.
2. After the energy depletion (~`:160-166`), accumulate the exact drain into
   `P_ENERGY_SPENT` (depletion + movement terms, as in ARCHITECTURE 0001).
3. Inside the existing `if in_danger { … }` branch (~`:176-181`),
   `physics_state[b + P_DANGER_PATH_LENGTH] += step_len;`.
4. Add all three to the respawn save/restore whitelist (`kernel_tick.wgsl:408-448`):
   save before the `for i in 0..PHYS_STRIDE` zero-loop, restore after.
5. Add GPU test `effort_accumulators_survive_respawn`: run a fixed-seed world,
   force a death, assert distance/energy read back non-zero and are *preserved*
   (not reset) across the respawn.

- **Depends on:** phys-accumulator-slots
- **Done when:** the survive-respawn test passes (and fails if a slot is omitted
  from the whitelist); cargo fmt/clippy/test green.

### effort-telemetry-split — Mirror Accumulation In The Split Path

The split passes (`phase_physics.wgsl` + `phase_death.wgsl`) are still compiled
(`gpu_kernel.rs`) and must match the fused path or they diverge.

**Steps:**
1. Apply the identical step-length / energy-spent / danger-path accumulation to
   the physics region of `phase_physics.wgsl`.
2. Add the identical save/restore of the three slots to the respawn whitelist in
   `phase_death.wgsl:38-76`.
3. Add GPU test `split_matches_fused_effort_telemetry`: after N ticks the split
   path's three accumulators match the fused path's (mirror the comparison in
   `split_serial_matches_fused_serial`).

- **Depends on:** phys-accumulator-slots
- **Done when:** the split/fused equivalence test passes; cargo fmt/clippy/test green.

### effort-telemetry-readback — Surface Telemetry To AgentFitness

**Steps:**
1. Add `distance_traveled: f32`, `energy_spent: f32`, `danger_path_length: f32`
   to `AgentFitness` (`governor.rs:30`) and to the `agent_result` schema via the
   idempotent `ALTER TABLE … ADD COLUMN` pattern.
2. Populate them in `gpu_orchestration.rs:249` and `headless.rs:225`, next to the
   `P_FOOD_COUNT` / `P_TICKS_ALIVE` reads.
3. Add a CPU test asserting an evaluated generation carries non-zero
   `distance_traveled` / `energy_spent` for active agents.

- **Depends on:** effort-telemetry-fused
- **Done when:** `AgentFitness` carries the telemetry and it persists to the DB;
  cargo fmt/clippy/test green.

### populate-danger-dwell-metric — Fill behavior_metric.danger_dwell_fraction

The column exists (`governor.rs:1725`) but is never populated.

**Steps:**
1. When persisting a generation's `behavior_metric` row, write
   `danger_dwell_fraction = danger_path_length / max(distance_traveled, EPSILON)`
   (population aggregate or per-champion, matching how the table is keyed).
2. Extend `behavior_metric_table_persists_danger_metrics` (`governor.rs:3425`) or
   add a sibling test asserting a real generation persists a non-null,
   in-`[0,1]` `danger_dwell_fraction`.

- **Depends on:** effort-telemetry-readback
- **Done when:** a generation persists a non-null `danger_dwell_fraction`; cargo
  fmt/clippy/test green.

---

## 0002 — Dwell-invariant hazard (Layer B)

### path-length-hazard-fused — Damage Proportional To Path Through Danger (Fused)

Replace the per-tick hazard (`kernel_tick.wgsl:176-178`) with a path-length dose.

**Steps:**
1. Compute `reference_step = 20.0 * wc_f32(WC_DT)` (default speed × dt; reuse the
   `20.0` anchor already at `:162-163`, or lift it to a shared `common.wgsl`
   constant).
2. Replace the hazard subtraction with
   `integrity -= WC_HAZARD_DAMAGE * integrity_scale * (step_len / max(reference_step, EPSILON))`.
   Keep `P_IN_DANGER_BIOME` published exactly as today.
3. Add GPU test `default_speed_crossing_damage_unchanged`: a default-speed agent
   crossing a fixed danger band loses (within tolerance) the same total integrity
   as under the per-tick model; a 2× `movement_speed` agent loses the same total
   per crossing (asserting speed-invariance), while a per-tick model would show
   the 2× agent losing half.

- **Depends on:** effort-telemetry-fused
- **Done when:** the damage-invariance test passes (default-speed neutral, 2×
  speed same total); cargo fmt/clippy/test green.

### path-length-hazard-split — Mirror Path-Length Hazard In The Split Path

**Steps:**
1. Apply the identical `step_len / reference_step` hazard change to
   `phase_physics.wgsl`.
2. Extend `split_matches_fused_effort_telemetry` (or a sibling) to assert the
   integrity trajectory through a danger crossing matches between the two paths.

- **Depends on:** path-length-hazard-fused, effort-telemetry-split
- **Done when:** split/fused integrity trajectories match through a danger
  crossing; cargo fmt/clippy/test green.

---

## 0003 — Super-linear locomotor energetics (Layer A)

### speed-cost-exponent-config — Add The Drag Knob (Default No-Op)

**Steps:**
1. Add `speed_cost_exponent: f32` to `BrainConfig` (`config.rs`) with
   `#[serde(default = "default_speed_cost_exponent")]` returning `1.0`; add to
   `Default`, `tiny`, `large` presets.
2. Add `WC_SPEED_COST_EXPONENT = 24` (`buffers.rs` + `common.wgsl` mirror); grow
   `WORLD_CONFIG_SIZE` `24 → 28` (7 × vec4); write the value in `fill_world_config`.
3. Confirm `config_size_fits_vec4_alignment` (`buffers.rs:995`) and
   `config_indices_within_bounds` (`buffers.rs:1004`) pass at the new size; add an
   assertion for the new index if needed.
4. Add a CPU test `speed_cost_exponent_round_trips` (config serialize/deserialize
   keeps the value; default is `1.0`).

- **Depends on:** —
- **Done when:** the config round-trips, the world-config size/alignment tests
  pass at 28, and the default is `1.0`; cargo fmt/clippy/test green.

### super-linear-drag-fused — Above-Baseline Drag In The Energy Drain (Fused)

**Steps:**
1. In `kernel_tick.wgsl:160-166`, gate the drag on the exponent so `k == 1.0` is
   the *exact* current expression: when `k == 1.0` use `move_speed / 20.0`; else
   use `pow(max(move_speed / 20.0, 1.0), wc_f32(WC_SPEED_COST_EXPONENT))`. Apply
   it as the `movement_mag` multiplier (ARCHITECTURE 0003).
2. Add GPU test `speed_cost_exponent_default_is_noop`: with `k = 1.0` the encoded
   brain state after N fixed-seed ticks is bit-identical to the pre-task build;
   with `k = 2.0`, an agent at `move_speed = 40` shows measurably higher energy
   drain (and earlier starvation) than at `move_speed = 20`, while a sub-baseline
   `move_speed = 10` agent's drain is unchanged from `k = 1.0` (above-baseline-only
   check — no torpor gradient).

- **Depends on:** speed-cost-exponent-config, path-length-hazard-fused
- **Done when:** `k=1.0` is bit-identical and `k>1.0` raises drag only above
  baseline; cargo fmt/clippy/test green.

### super-linear-drag-split — Mirror The Drag In The Split Path

**Steps:**
1. Apply the identical gated-drag change to `phase_physics.wgsl`.
2. Extend the split/fused equivalence test to cover `k = 2.0`.

- **Depends on:** super-linear-drag-fused, path-length-hazard-split
- **Done when:** split matches fused at `k = 1.0` and `k = 2.0`; cargo
  fmt/clippy/test green.

---

## 0004 — Effort-rebased fitness (Layer C)

### composite-fitness-effort-rebase — Food-Per-Energy + Cells-Per-Distance

Re-base the two coverage axes onto effort, behind a formula flag.

**Steps:**
1. Add the new consts (`FORAGING_ENERGY_TARGET`, `ENERGY_FLOOR`, `DISTANCE_FLOOR`,
   `EXPLORATION_DISTANCE_BUDGET`) near `governor.rs:43`, with provisional values
   documented as calibrated in `fitness-calibration-replay`. Set
   `EXPLORATION_DISTANCE_BUDGET` well above one cell width (world 256 / HEATMAP_RES
   64 = 4.0) so the cap binds.
2. Rewrite `composite_fitness` (`governor.rs:84-102`) per ARCHITECTURE 0004:
   keep the `ticks_alive` arg and the survival term; add `distance_traveled` /
   `energy_spent` args; foraging = `food/energy`, exploration =
   `min(coverage, cells_per_distance)`. Update the call site (`governor.rs:601`).
3. Gate the new formula behind a config flag (e.g. `effort_rebased_fitness`,
   default `false`) so the default build keeps the time-denominated formula until
   the `0006` gate.
4. Add CPU test `composite_fitness_rewards_efficiency`: synthetic telemetry where
   a fast-aimless agent (high food, high distance, high energy) scores lower than
   a slow-deliberate agent (same food, low distance/energy), and a camper (high
   food, ~0 distance, nonzero energy) does not max foraging. Update existing
   `composite_fitness` unit tests for the new signature.

- **Depends on:** effort-telemetry-readback
- **Done when:** the efficiency test passes and the flag-off path reproduces the
  old scores; cargo fmt/clippy/test green.

### fitness-calibration-replay — Calibrate The New Targets

**Steps:**
1. Replay a recorded generation's telemetry through the old vs new
   `composite_fitness` (reuse the `generation_recording` data); pick
   `FORAGING_ENERGY_TARGET` / `EXPLORATION_DISTANCE_BUDGET` so a competent
   forager's foraging/exploration max near 1.0 and a fast-aimless agent drops.
2. Record the calibrated values and the before/after score deltas in a decision
   doc `0009-FITNESS-CALIBRATION.md` in this folder.

- **Depends on:** composite-fitness-effort-rebase
- **Done when:** the decision doc records calibrated consts with replayed
  before/after numbers; cargo fmt/clippy/test green.

---

## 0005 — Danger percept, avoidance learning, intent metric (Layer D)

### nearest-danger-telemetry — Compute Nearest-Danger Bearing/Distance (Both Paths)

The danger analogue of the nearest-food reduction, over the static biome grid.

**Steps:**
1. Add `P_NEAREST_DANGER_DISTANCE = 39`, `P_NEAREST_DANGER_BEARING = 40`
   (`buffers.rs` + `common.wgsl`); bump `PHYS_STRIDE` `39 → 41`; extend the parity
   test. Add a `DANGER_SENSE_RADIUS` constant.
2. In `kernel_tick.wgsl` (near the food scan ~`:296-356`) and `phase_physics.wgsl`,
   scan biome-grid cells within `DANGER_SENSE_RADIUS` (bounded
   `(radius/biome_cell)^2` samples); for the nearest `sample_biome == BIOME_DANGER`
   cell store distance and the signed facing-relative bearing (reuse the cross/dot
   bearing math at `:342-349`). Sentinel `= DANGER_SENSE_RADIUS` / `0.0` when none.
3. Reset both slots to the sentinel in both respawn whitelists alongside
   `P_NEAREST_FOOD_BEARING` (`kernel_tick.wgsl:447`, `phase_death.wgsl`).
4. Add GPU test `nearest_danger_bearing_points_at_danger`: an agent placed near a
   known danger patch reads a finite distance and a bearing pointing at it; an
   agent far from any danger reads the sentinel.

- **Depends on:** effort-telemetry-fused, effort-telemetry-split
- **Done when:** the bearing/distance test passes in both paths; parity test green
  at `PHYS_STRIDE = 41`; cargo fmt/clippy/test green.

### danger-percept-sense — Feed Danger Bearing/Distance As Senses (Flagged)

**Steps:**
1. Add `danger_percept_enabled: bool` to `BrainConfig` (`#[serde(default)]` →
   `false`) and a world-config bit, documented as the `0006` gate flag.
2. When enabled, grow `NON_VISUAL_FEATURE_COUNT` `25 → 27` and `SENSORY_STRIDE`
   accordingly (`common.wgsl`); pack nearest-danger bearing + distance into the
   non-visual tail in `coop_feature_extract` (`brain_passes.wgsl`). `FEATURE_COUNT`
   and `O_ENC_WEIGHTS` resize from the canonical constants.
3. Add GPU test `danger_percept_byte_identical_when_flag_off`: with the flag off,
   `FEATURE_COUNT` and the encoded state match the current build; with it on,
   `FEATURE_COUNT == VISION_INPUT + 27` and the encoder runs without a wgpu
   validation error.

- **Depends on:** nearest-danger-telemetry
- **Done when:** flag-off byte-identical, flag-on accepts the new width; cargo
  fmt/clippy/test green.

### danger-avoidance-potential — Symmetric Potential Shaping (Flagged)

Mirror the food approach potential for danger avoidance.

**Steps:**
1. In `brain_passes.wgsl`, define the avoidance potential
   `Φ_d = -(1 - nearest_danger_distance / DANGER_SENSE_RADIUS)` and add
   `γ·Φ_d(s') - Φ_d(s)` into the same shaping path the food approach potential
   uses (`:778-787`), behind `danger_percept_enabled`. Reuse `P_PREV_POTENTIAL`'s
   pattern (store a previous danger potential slot, reset on respawn).
2. Add GPU test `avoidance_potential_sign`: a fixed-seed agent stepping toward a
   danger patch receives a negative shaping increment; stepping away, positive;
   with the flag off, zero.

- **Depends on:** nearest-danger-telemetry
- **Done when:** the shaping-sign test passes and is zero when flagged off; cargo
  fmt/clippy/test green.

### avoidance-intent-metric — Sensed-Then-Turned Observability

**Steps:**
1. Accumulate, per agent, the fraction of in-sense-range ticks where the motor
   turn opposed the danger bearing (deliberate turn-away). Surface it via readback
   into a new `behavior_metric` column (idempotent `ALTER TABLE … ADD COLUMN`,
   e.g. `avoidance_intent_fraction`).
2. Add a test asserting the column persists and that a scripted turn-away agent
   scores higher avoidance-intent than a straight-through agent.

- **Depends on:** nearest-danger-telemetry
- **Done when:** the avoidance-intent column persists and discriminates
  turn-away from straight-through; cargo fmt/clippy/test green.

---

## 0006 — Validation & default-flip gate

### speed-decoupling-validation — Measure The Decoupling (Flags On vs Off)

**Steps:**
1. Extend the headless harness (`headless.rs`) to run N generations with all 0009
   flags off (baseline) then on, reporting per ARCHITECTURE 0006:
   correlation(`movement_speed`, `composite_fitness`); the population-mean
   `movement_speed` trajectory; regression(`death_count`, `movement_speed`);
   food-per-energy vs `movement_speed` slope; `danger_dwell_fraction` and
   `avoidance_intent_fraction` retention; mean `ticks_alive`.
2. Record the baseline-vs-on numbers in `0009-SPEED-DECOUPLING.md` in this folder.

- **Depends on:** path-length-hazard-split, super-linear-drag-split,
  composite-fitness-effort-rebase, danger-percept-sense
- **Done when:** the decoupling doc records the full baseline-vs-on metric set;
  cargo fmt/clippy/test green.

### default-flip-gate — Flip Or Hold The Defaults (GATED)

**Gate:** speed↔fitness correlation falls from strongly-positive to ≈0 /
single-peaked (peak well below the `[1,100]` clamp), AND mean `ticks_alive` does
not collapse vs baseline, AND `danger_dwell_fraction` / `avoidance_intent_fraction`
stay non-zero (danger-decision data still collected).

**Steps:**
1. If the gate holds, set `speed_cost_exponent` default `> 1.0` (the calibrated
   value), `effort_rebased_fitness` default `true`, and `danger_percept_enabled`
   default `true`; record the decision and the retained numbers in
   `0009-SPEED-DECOUPLING.md`.
2. If the gate fails (speed still dominates, or population collapses, or danger
   data vanishes), leave the defaults off, record why, and file the smallest
   follow-up (re-tune `k` / the fitness targets / the danger-sense radius) — do
   not flip.

- **Depends on:** speed-decoupling-validation
- **Done when:** the decision doc resolves to flip-or-hold with the measured
  numbers; if flipped, the default-on path keeps cargo fmt/clippy/test green.

---

**End of plan 0009 TASKS.** When every "Done when" bullet is green, raw speed no
longer buys score on any axis, deliberate foraging and avoidance are the only
gradients that pay, and the agent perceives danger ahead well enough to make —
and to be measured making — the avoidance decision, all proven by the
speed-decoupling gate rather than assumed.
