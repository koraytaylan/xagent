# Architecture - Plan 0009 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_physics.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-sandbox/src/gpu_orchestration.rs`,
> `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/src/agent/mod.rs`.
> Line numbers are hints; locate by symbol. WGSL is `include_str!`'d into the
> brain crate — run `cargo clean -p xagent-brain` after shader edits to avoid a
> stale rlib.

The exploit is killed in three reinforcing layers plus a percept upgrade.
Layers **A** (`0003`, super-linear energetics) and **B** (`0002`, path-length
hazard) are pure kernel-mechanism changes that neutralize raw coverage, so the
exploit cannot migrate between axes. Layer **C** (`0004`, effort-rebased fitness)
removes the residual coverage gradient from scoring. Layer **D** (`0005`, danger
percept) makes deliberate avoidance learnable and measurable. `0001` lays the
telemetry substrate all of them read; `0006` measures the result and gates the
default-flip.

```
                          ┌─ 0002 path-length hazard ──┐ (survival axis)
0001 effort/exposure ─────┼─ 0003 super-linear drag ───┤ (all axes; keystone)
     telemetry (no-op)    ├─ 0004 food/energy fitness ─┘ (foraging + exploration)
                          └─ 0005 danger percept + avoidance potential + intent metric
                                                          (makes avoidance learnable/measurable)
                                          │
                                          ▼
                          0006 headless A/B speed-decoupling gate → flip defaults
```

## 0001 — Effort & exposure telemetry

Today the per-agent physics record is `PHYS_STRIDE = 36` slots, the last being
`P_IN_DANGER_BIOME = 35` (`buffers.rs:201-202`, `common.wgsl:327`); a parity test
asserts `PHYS_STRIDE == max_offset + 1` (`buffers.rs:1310-1356`). Fitness reads
only `P_FOOD_COUNT`, `P_TICKS_ALIVE`, `P_DEATH_COUNT`
(`gpu_orchestration.rs:249-251`, `headless.rs:225-227`). None of the quantities
speed cannot inflate — distance, energy spent, danger path length — is recorded.

### New cumulative accumulators

Three new `f32` slots, `PHYS_STRIDE 36 → 39`. All are *generation-cumulative*
(like `P_FOOD_COUNT` / `P_TICKS_ALIVE`), so they must survive respawn.

```rust
// buffers.rs, after P_IN_DANGER_BIOME (= 35)
pub const P_DISTANCE_TRAVELED: usize = 36; // Σ planar |Δpos| over the generation
pub const P_ENERGY_SPENT: usize = 37;      // Σ movement+metabolic drain this generation
pub const P_DANGER_PATH_LENGTH: usize = 38; // Σ |Δpos| accumulated while in danger
pub const PHYS_STRIDE: usize = 39;
```

```wgsl
// common.wgsl — mirror, and bump PHYS_STRIDE to 39u
const P_DISTANCE_TRAVELED: u32 = 36u;
const P_ENERGY_SPENT: u32 = 37u;
const P_DANGER_PATH_LENGTH: u32 = 38u;
```

### Accumulation in the physics tick (BOTH paths)

In `kernel_tick.wgsl` (after the final position write ~`:156-158`, alongside the
energy depletion ~`:160-166` and the `in_danger` branch ~`:175-181`) and in the
identical region of `phase_physics.wgsl`:

```wgsl
// step length this tick (last_pos is loaded at the top of the physics tick)
let step_len = length(vec2<f32>(pos.x - last_pos.x, pos.z - last_pos.z));
physics_state[b + P_DISTANCE_TRAVELED] += step_len;

// energy spent = exactly the drain subtracted this tick (depletion + movement)
let drain_this_tick = wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate
                    + movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;
physics_state[b + P_ENERGY_SPENT] += drain_this_tick;

// inside the existing `if in_danger { … }` branch:
physics_state[b + P_DANGER_PATH_LENGTH] += step_len;
```

### Respawn whitelist (BOTH paths)

The three slots join the save/restore block in `kernel_tick.wgsl:408-448` AND
`phase_death.wgsl:38-76`:

```wgsl
let saved_distance     = physics_state[base + P_DISTANCE_TRAVELED];
let saved_energy_spent = physics_state[base + P_ENERGY_SPENT];
let saved_danger_path  = physics_state[base + P_DANGER_PATH_LENGTH];
// … after the `for i in 0..PHYS_STRIDE { = 0.0 }` zero-loop …
physics_state[base + P_DISTANCE_TRAVELED]  = saved_distance;
physics_state[base + P_ENERGY_SPENT]       = saved_energy_spent;
physics_state[base + P_DANGER_PATH_LENGTH] = saved_danger_path;
```

### Readback + behavior_metric

```rust
// AgentFitness (governor.rs:30) gains:
pub distance_traveled: f32,
pub energy_spent: f32,
pub danger_path_length: f32,
// populated at gpu_orchestration.rs:249 and headless.rs:225, next to P_FOOD_COUNT:
a.distance_traveled = state[base + P_DISTANCE_TRAVELED];
a.energy_spent      = state[base + P_ENERGY_SPENT];
a.danger_path_length = state[base + P_DANGER_PATH_LENGTH];
```

`behavior_metric.danger_dwell_fraction` (declared at `governor.rs:1725`, never
populated) is written as `danger_path_length / max(distance_traveled, EPSILON)`
— a speed-invariant fraction of the agent's path spent in danger. This is the
only consumer of `P_DANGER_PATH_LENGTH`; it is observability, not selection.

Properties that make this safe:
- The accumulators reuse values already in registers (`pos`, `last_pos`,
  `movement_mag`, `in_danger`) — ~1 `length()` + a few adds per tick, no new
  buffer, no extra biome/food lookup. The +12 bytes/agent widens the
  layout-derived staging buffers automatically (everything sizes from
  `PHYS_STRIDE`).
- Adding `f32` fields to `AgentFitness` does not break the `agent_result` DB
  schema (idempotent `ALTER TABLE … ADD COLUMN` pattern, CLAUDE.md); the new
  columns default to NULL on old rows.
- The parity test `shader_phys_constants_match_rust` (`buffers.rs:1310-1356`)
  fails loudly if `PHYS_STRIDE` and the offsets drift between Rust and WGSL.

## 0002 — Dwell-invariant hazard (Layer B)

Today hazard is per-tick: `integrity -= WC_HAZARD_DAMAGE * integrity_scale` while
`in_danger` (`kernel_tick.wgsl:176-178`, mirrored in `phase_physics.wgsl`). A 2×
agent halves dwell ticks → halves total damage.

Replace the per-tick subtraction with a path-length dose. `reference_step` is the
distance a default-speed agent travels in one tick, *derived* from config (no
magic literal):

```wgsl
// reference_step = default_movement_speed * dt = 20.0 * WC_DT
let reference_step = 20.0 * wc_f32(WC_DT);
if in_danger {
    physics_state[b + P_INTEGRITY] -=
        wc_f32(WC_HAZARD_DAMAGE) * integrity_scale * (step_len / max(reference_step, EPSILON));
    physics_state[b + P_DANGER_PATH_LENGTH] += step_len; // from 0001
    physics_state[b + P_IN_DANGER_BIOME] = 1.0;
} else {
    physics_state[b + P_IN_DANGER_BIOME] = 0.0;
}
```

Properties that make this safe:
- **Default-speed-neutral by construction.** A default-speed agent moves
  `reference_step` per tick, so `step_len / reference_step ≈ 1.0` and the
  per-tick damage is byte-identical to today — zero curriculum disruption
  (`hazard_damage_rate` calibration stays valid). A 2× agent loses 2× per tick
  over half the ticks = the same total per crossing.
- **Graded, not lethal** — the user's hard constraint. `P_IN_DANGER_BIOME` is
  still published every tick, so `danger_exit_probe` and the `behavior_metric`
  danger consumers are untouched; agents still enter danger and generate decision
  data, but speed no longer dodges the cost, so their true avoidance preference is
  revealed instead of masked.
- **`20.0` is the documented default-speed constant** already used as the
  normalizer at `kernel_tick.wgsl:162-163`; this reuses the same anchor rather
  than introducing a new literal. (If a single-source constant for default speed
  is preferred, lift it into `common.wgsl` and share it with line 163.)
- Biologically this is dose-response: harm scales with exposure integrated over
  the path through the hazard, not with wall-clock time — a fast sprint and a
  slow walk across the same band absorb the same dose.

## 0003 — Super-linear locomotor energetics (Layer A, the keystone)

Today movement energy is linear in speed (`kernel_tick.wgsl:160-166`):
`movement_mag = min(|fwd|+|strafe|, 1.414) * (move_speed / 20.0)`, so
cost-per-distance is flat. Add a drag exponent so cost-per-distance rises with
speed, applied **above baseline only**.

### Config + world-config slot

`speed_cost_exponent` is a global tuning knob (not heritable), so it lives in the
world-config uniform alongside `hazard_damage_rate`:

```rust
// config.rs — BrainConfig (carries the world knob, like energy_depletion_rate)
#[serde(default = "default_speed_cost_exponent")]
pub speed_cost_exponent: f32,           // default 1.0 → exact no-op
fn default_speed_cost_exponent() -> f32 { 1.0 }
```

```rust
// buffers.rs — new WC index; WORLD_CONFIG_SIZE 24 → 28 (next 6→7 × vec4)
pub const WC_SPEED_COST_EXPONENT: usize = 24;
pub const WORLD_CONFIG_SIZE: usize = 28; // padded to 7 × vec4
// fill_world_config: out[WC_SPEED_COST_EXPONENT] = config.speed_cost_exponent;
```

```wgsl
// common.wgsl — mirror the index
const WC_SPEED_COST_EXPONENT: u32 = 24u;
```

### Drag in the energy drain (BOTH paths)

```wgsl
// kernel_tick.wgsl:163 and the phase_physics.wgsl mirror
let speed_ratio = move_speed / 20.0;
let drag = pow(max(speed_ratio, 1.0), wc_f32(WC_SPEED_COST_EXPONENT)); // 1.0 when k=1
let movement_mag = min(abs(motor_forward) + abs(motor_strafe), 1.414) * drag;
```

Properties that make this safe:
- **`k = 1.0` is a bit-exact no-op:** `pow(max(r,1.0), 1.0)` over `r ≥ 1` equals
  `r`, and for `r < 1` the old code already used `r` while the new code uses
  `max(r,1.0) = 1.0` — but at `k = 1.0` we keep the *exact* old expression by
  guarding the whole drag behind the flag (`if k == 1.0 use r else use
  pow(max(r,1.0),k)`), so the default path is byte-identical. (The
  `max(r,1.0)` clamp only engages once `k ≠ 1.0`.)
- **Above-baseline only** kills the torpor-drift failure mode: there is no
  `drag < 1` region below default speed, so selection gets no downward gradient
  that would ratchet `move_speed` to its floor. The speed→fitness curve becomes
  single-peaked.
- **Disciplines both reward layers for free.** Selection feels it via more
  starvation deaths at high speed; the in-life TD learner feels it via
  `energy_delta` (energy is a brain interoception input,
  `phase_vision.wgsl:206-209`) flowing into `raw_gradient`
  (`brain_passes.wgsl:778-787`) — no explicit reward term is added (an explicit
  speed-penalty term is rejected; it would poison the klinotaxis-escape EMA
  baseline that drives the danger-escape reflex).
- `WORLD_CONFIG_SIZE` growth is validated by `config_size_fits_vec4_alignment`
  (`buffers.rs:995`) and `config_indices_within_bounds` (`buffers.rs:1004`); the
  uniform stays 16-byte aligned at 28 floats (7 × vec4).
- Biologically: locomotor power scales super-linearly with speed (drag/work
  ∝ v²–v³); sprinting is expensive, so an animal that sweeps at max speed pays
  for it metabolically.

## 0004 — Effort-rebased fitness (Layer C)

`composite_fitness` (`governor.rs:84-102`) keeps its weights and survival term;
the foraging and exploration *denominators* change from time to effort. The
signature retains `ticks_alive` (dropping it touches every caller and the
unit tests; keep it threaded) and gains the new telemetry.

```rust
// new consts near FORAGING_RATE_TARGET (governor.rs:43)
const FORAGING_ENERGY_TARGET: f32 = /* food per energy unit for a competent
    forager; calibrated in 0004 replay, replaces FORAGING_RATE_TARGET */;
const ENERGY_FLOOR: f32 = /* min energy before the rate counts; anti-div0 + anti-camp */;
const DISTANCE_FLOOR: f32 = /* min meters before the per-distance rate counts */;
const EXPLORATION_DISTANCE_BUDGET: f32 = /* meters that buy one full-credit cell;
    set well above one cell width (world 256 / HEATMAP_RES 64 = 4.0) so the cap binds */;

fn composite_fitness(
    death_count: u32, food_consumed: u32, cells_explored: u32,
    ticks_alive: u64,               // retained (within-life guard / callers)
    distance_traveled: f32, energy_spent: f32,
    total_grid_cells: f32,
) -> f32 {
    // FORAGING: food per ENERGY spent — speed-invariant and camp-proof (a parked
    // agent still burns unavoidable metabolic + brain energy).
    let energy = energy_spent.max(ENERGY_FLOOR);
    let foraging = ((food_consumed as f32 / energy) / FORAGING_ENERGY_TARGET).min(1.0);

    // EXPLORATION: cells per distance, capped by true coverage. A fast aimless
    // sweep and a slow aimless walk get the same cells-per-meter.
    let dist = distance_traveled.max(DISTANCE_FLOOR);
    let coverage = (cells_explored as f32 / total_grid_cells).min(1.0);
    let cells_per_dist = (cells_explored as f32 / (dist / EXPLORATION_DISTANCE_BUDGET)).min(1.0);
    let exploration = coverage.min(cells_per_dist);

    // SURVIVAL: unchanged — death_count is now speed-invariant via 0002.
    let survival = SURVIVAL_FLOOR + (1.0 - SURVIVAL_FLOOR) / (1.0 + death_count as f32 * DEATH_PENALTY);
    survival * (foraging * FORAGING_WEIGHT + exploration * EXPLORATION_WEIGHT)
}
```

The call site (`governor.rs:601`) passes the new `AgentFitness` fields; the
display-only second reference (~`:1233`) reads stored fields and needs no change.
The new formula is selected by a config flag so the default build keeps the old
time-denominated formula until `0006` flips it.

Properties that make this safe:
- **Food-per-energy defeats camping** where food-per-distance fails: energy is
  spent even while stationary (`kernel_tick.wgsl:165` metabolic + `:194`
  brain-drain), so a parked-on-respawning-food agent cannot drive its
  denominator to zero.
- **`min(coverage, cells_per_dist)` is both efficient and bounded** — a
  straight-line sweeper banks high cells-per-meter but is capped by the fraction
  of the world it actually covered, and its foraging term (0.85 weight) stays at
  baseline because it isn't *seeking*.
- The `WithinLifeTracker` quarter-food-rate learning signal
  (`governor.rs:110-151`) is independent of `composite_fitness` and is preserved.
- `reduce_fitness` and the significance guard average `composite_fitness`
  unchanged; only the per-agent value's derivation changed.

## 0005 — Danger percept, avoidance learning, and intent metric (Layer D)

Today the only danger percept is the entangled red terrain color in vision and
the post-contact `TOUCH_HAZARD` flag. Foraging, by contrast, is taught by a
nearest-food bearing/distance feeding a potential-based approach shaping term
(`kernel_tick.wgsl:296-356`, `brain_passes.wgsl:778-781`). Layer D builds the
exact mirror image for danger.

### Nearest-danger telemetry (BOTH paths)

Two new physics slots (`PHYS_STRIDE 39 → 41`), computed by a bounded scan of the
static biome grid in a ring around the agent — the danger analogue of the
nearest-food reduction, but over biome cells rather than a food list:

```rust
pub const P_NEAREST_DANGER_DISTANCE: usize = 39; // sentinel = DANGER_SENSE_RADIUS when none
pub const P_NEAREST_DANGER_BEARING: usize = 40;  // signed bearing from facing, like food
pub const PHYS_STRIDE: usize = 41;
```

```wgsl
// scan biome-grid cells within DANGER_SENSE_RADIUS of the agent; for the nearest
// cell with sample_biome == BIOME_DANGER, store distance and signed bearing
// (cross/dot of facing × to_danger, exactly as P_NEAREST_FOOD_BEARING at
// kernel_tick.wgsl:342-349). Bounded: (DANGER_SENSE_RADIUS / biome_cell)^2 samples.
```

Both slots are non-cumulative (recomputed each tick), so they are reset to the
sentinel in both respawn whitelists alongside `P_NEAREST_FOOD_BEARING`
(`kernel_tick.wgsl:447`, `phase_death.wgsl`).

### Dedicated danger sense (behind `danger_percept_enabled`)

Feed nearest-danger bearing + distance as two new sensory inputs, growing the
non-visual tail by 2 (the same pattern `0008` used to redefine `FEATURE_COUNT`):

```wgsl
// common.wgsl
const NON_VISUAL_FEATURE_COUNT: u32 = 27u; // was 25; +bearing +distance
override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 29u; // was +27
```

`coop_feature_extract` (`brain_passes.wgsl`) packs the two danger features into
the non-visual tail; `FEATURE_COUNT` (`common.wgsl:172-175`) and the encoder
weight region `O_ENC_WEIGHTS` resize from the canonical constant automatically.
Behind the flag (default off) the tail stays 25 and the build is byte-identical.

### Avoidance potential (behind the flag)

A potential-based shaping term mirroring the food approach potential
(`brain_passes.wgsl:778-781`). Define an avoidance potential
`Φ_d(s) = -(1 - nearest_danger_distance / DANGER_SENSE_RADIUS)` (more negative as
danger nears); the shaping reward adds `γ·Φ_d(s') - Φ_d(s)`, so moving away from
danger yields a positive increment and moving toward it a negative one. Because
it is a potential difference, it is optimal-policy-invariant (Ng et al. 1999),
consistent with the existing approach-shaping framework (`common.wgsl:388`).

### Avoidance-intent metric

Populate a `behavior_metric` avoidance-intent signal — the fraction of ticks
where danger was within sense range *and* the agent turned away from it (motor
turn opposed the danger bearing). Derived from `P_NEAREST_DANGER_BEARING` and the
motor turn output already in `physics_state`. This is the measurable
"deliberate avoidance" the user wants; observability only, never selection.

Properties that make this safe:
- **Symmetric to a proven mechanism.** Bearing/distance computation, the
  potential-shaping reward, and the readback all reuse the food-approach code
  paths; risk is in the biome-ring scan cost, bounded by `DANGER_SENSE_RADIUS`.
- **Flag-gated and byte-identical when off.** With `danger_percept_enabled =
  false`, the sense is not packed (`NON_VISUAL_FEATURE_COUNT` stays 25), the
  potential term is skipped, and only the telemetry slots are written (used by
  the intent metric); the encoded brain state is unchanged.
- **Policy-invariant reward.** The avoidance potential is a difference of a
  state-only potential, so it cannot change the optimal policy — it only shapes
  the gradient toward it, the same guarantee the food approach potential relies
  on.
- The biome is static per generation, so the scan is deterministic and adds no
  cross-agent dependency.

## 0006 — Validation & default-flip gate

The plan's success is a measured speed-decoupling, not a green build. A headless
A/B harness (extending the existing headless run, `headless.rs`) reports, with
all flags off (baseline) then on:

- **correlation(`movement_speed`, `composite_fitness`)** across the population —
  baseline strongly positive; target ≈0 or single-peaked with the peak well
  below the `[1,100]` clamp.
- **population-mean `movement_speed` trajectory** across generations — must stop
  ratcheting toward 100 without collapsing to 1.0 (torpor check).
- **regression(`death_count`, `movement_speed`)** — baseline negative
  (fast = fewer deaths); target ≈0 (0002 made per-crossing damage
  speed-invariant).
- **food-per-energy vs `movement_speed` slope** — flat/negative confirms skill,
  not speed, drives foraging.
- **`behavior_metric.danger_dwell_fraction` and the avoidance-intent signal stay
  non-zero** — proof the population still enters danger and generates the
  avoidance-decision data the user requires.
- **mean `ticks_alive` does not collapse** vs baseline (energy-balance sanity).

The gate (a conjunction, like `0008`'s) flips the defaults — `speed_cost_exponent
> 1.0`, the effort-rebased fitness, `danger_percept_enabled = true` — only if the
speed-fitness correlation falls, the population does not collapse, and danger
data is retained. Otherwise it holds and records why in a decision doc in this
folder.

## Test Strategy

Each layer ships a falsifiable test; GPU tests self-skip without an adapter.

- `phys_stride_parity` — `shader_phys_constants_match_rust` extended for the new
  slots (`PHYS_STRIDE == max_offset + 1` in Rust and WGSL).
- `effort_accumulators_survive_respawn` (GPU) — distance/energy/danger-path read
  back non-zero and are *preserved* across a forced death in both the fused and
  split paths.
- `split_matches_fused_effort_telemetry` (GPU) — the split path's accumulators
  match the fused path's after N ticks (mirror `split_serial_matches_fused_serial`).
- `default_speed_crossing_damage_unchanged` (GPU) — a default-speed agent
  crossing a fixed danger band loses the same total integrity under the
  path-length model as under the per-tick model; a 2× agent loses the same total.
- `speed_cost_exponent_default_is_noop` (GPU) — with `k = 1.0` the encoded state
  after N ticks is bit-identical to the pre-task build; with `k = 2.0` energy
  drain at `move_speed = 40` is measurably higher than at 20.
- `composite_fitness_rewards_efficiency` (CPU) — synthetic telemetry: a
  fast-aimless agent (high food, high distance, high energy) scores *lower* than
  a slow-deliberate agent (same food, low distance, low energy); a camper (high
  food, ~0 distance, nonzero energy) does not max foraging.
- `world_config_size_alignment` — `config_size_fits_vec4_alignment` /
  `config_indices_within_bounds` pass at `WORLD_CONFIG_SIZE = 28`.
- `nearest_danger_bearing_points_at_danger` (GPU) — an agent placed near a danger
  patch reads a finite distance and a bearing pointing at the patch; sentinel
  when none is in range.
- `danger_percept_byte_identical_when_flag_off` (GPU) — with
  `danger_percept_enabled = false`, `FEATURE_COUNT` and the encoded state match
  the current build.
- `avoidance_potential_sign` (GPU) — a fixed-seed agent stepping toward danger
  receives a negative shaping increment, away a positive one.

Canonical CI gate for every task:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction With Prior Work

- **Plan 0007 (Learning Control Grounding) — built upon.** 0007 added the nav
  telemetry slots (`P_NEAREST_FOOD_BEARING`, `P_IN_DANGER_BIOME`), the
  `behavior_metric` table, the danger-exit probe, and runtime genome authority
  (`patch_agent_configs`). 0009 extends the same telemetry/metric machinery and
  reuses the (correctly non-inverted) `P_IN_DANGER_BIOME` flag. The 0007 finding
  that the credit path — not the encoder — is the learning bottleneck is why
  Layer D adds a *potential shaping* term (gradient help) rather than a new
  encoder.
- **Plan 0004 (Approach Reward Shaping) — mirrored.** The food approach potential
  (`F = γΦ(s′) − Φ(s)` into `raw_gradient`) is the exact template for the Layer D
  avoidance potential; the policy-invariance guarantee carries over.
- **Plan 0008 (Hubel-Wiesel) — pattern reused.** The `FEATURE_COUNT` redefinition
  for the danger sense follows 0008's flag-gated, byte-identical-when-off
  encoder-input change; the default-flip gate mirrors 0008's conjunction gate and
  decision-doc discipline. New work stays inside the `FusedSerial` default.
- **Plan 0006 / 0005 (throughput) — respected.** Layer D's biome-ring scan and
  the new accumulators add per-agent work; their cost is tracked against the
  0006/0005 throughput baselines and is part of the `0006` gate (the danger scan
  is bounded; the accumulators are a few ops).
