# Architecture — Plan 0004 (deltas)

> Edits center on `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-shared/src/config.rs`,
> `crates/xagent-sandbox/src/headless.rs`, and the probe tests in
> `crates/xagent-sandbox/tests/integration.rs`. The gated workstreams `0004`/
> `0005` additionally touch `crates/xagent-sandbox/src/agent.rs` (mutation) and
> `crates/xagent-brain/src/shaders/kernel/phase_vision.wgsl`.
> Line numbers are hints; locate by symbol.

## 0001 — Approach reward shaping

Today the homeostasis pass `coop_habituate_homeo` (`brain_passes.wgsl:144-196`)
forms `raw_gradient = energy_delta·ENERGY_WEIGHT + integrity_delta·INTEGRITY_WEIGHT`
(`:170`) from purely interoceptive deltas, blends it into three EMAs
(`:171-176`), amplifies it to `s_homeo[1u] = raw_gradient·(1+urgency)` (`:188-190`),
and that scalar is consumed verbatim as the TD reward in
`coop_credit_assignment` (`let reward = s_homeo[1u];`, `:423`). Nothing in the
chain references the agent's distance or bearing to food. The food-detect pass
`agent_food_detect` (`kernel_tick.wgsl:201-277`) computes per-food `dx/dz`
(`:229-231`) but reduces only the nearest food *within `eat_radius`* (`:232`),
for eating — there is no "nearest food in navigational range" signal anywhere.

### Surface nearest-food-within-shaping-radius

Today `agent_food_detect` discards every food outside `eat_radius`. We add a
second, parallel reduction over a wider `SHAPING_RADIUS` and publish it to a new
physics-state slot the brain pass can read in the same kernel cycle (food-detect
runs before the brain in the cycle, `kernel_tick.wgsl:512-523`). `physics_state`
is per-agent live state, regenerated each run and not inherited across
generations, so growing `PHYS_STRIDE` carries no serialization/inheritance cost.

Edits:

- **Two new physics-state slots** (`buffers.rs:121-159`): append after
  `P_LAST_DEATH_TICK` and bump `PHYS_STRIDE`. The slot-map guard test
  (`buffers.rs:997-1039`) and the WGSL/Rust mirror test (`:824`) must be updated
  in lockstep.

```rust
/// Distance (world units) to the nearest food within SHAPING_RADIUS, written by
/// `agent_food_detect`; SHAPING_RADIUS (= no food in range) when none found.
/// Read by the homeostasis pass to form the approach potential Φ. Telemetry-
/// visible for the alignment/foraging probes.
pub const P_NEAREST_FOOD_DIST: usize = 32;
/// Previous brain tick's approach potential Φ(s), written by the homeostasis
/// pass after forming F = γΦ(s′) − Φ(s); reset on respawn so the food teleport
/// cannot inject a spurious shaping reward across a death.
pub const P_PREV_POTENTIAL: usize = 33;
pub const PHYS_STRIDE: usize = 34;
```

- **Shaping-radius reduction** (`agent_food_detect`, `kernel_tick.wgsl:216-277`):
  carry a second per-thread minimum `local_best_shaping_dist_sq` over
  `SHAPING_RADIUS²` (independent of the `eat_radius` gate), reduce it through the
  same two-phase shared-memory pattern already used for the eat candidate
  (`:239-266`), and have thread 0 write `sqrt(best)` (or `SHAPING_RADIUS` when
  none) to `physics_state[b + P_NEAREST_FOOD_DIST]`. The eat path
  (`:267-275`) is unchanged.

- **Reset on respawn** (`agent_death_respawn`, `kernel_tick.wgsl:283-348`): set
  `physics_state[base + P_PREV_POTENTIAL] = 0.0` and
  `physics_state[base + P_NEAREST_FOOD_DIST] = SHAPING_RADIUS` in the same
  thread-0-only restore block that preserves `saved_food_count` etc. (`:318-346`).
  This contains no barriers, so early-return uniformity is untouched.

```wgsl
// world-units radius within which food contributes to the approach potential.
// = VISION_MAX_DIST: Φ is a proxy for "nearest visible food"; PBRS invariance
// holds for any state potential, so the radius proxy is sound (SCOPE: locked).
const SHAPING_RADIUS: f32 = 30.0;
```

Properties that make this safe:
- The new reduction reuses the existing `s_similarities`/`shared_sort_indices`
  barrier structure and runs unconditionally for all 256 threads, so workgroup-
  barrier uniformity (the top-of-file SAFETY INVARIANT) is preserved.
- `P_NEAREST_FOOD_DIST`/`P_PREV_POTENTIAL` are written only by thread 0 inside an
  already-thread-0-gated region; no new cross-thread write hazard.
- Growing `PHYS_STRIDE` is a buffer-sizing change only; `physics_state` is never
  serialized or inherited, so no migration is required.

### Potential-based approach shaping

Today `raw_gradient` (`brain_passes.wgsl:170`) is the sole reward source and
feeds the EMAs, the amplified reward `s_homeo[1u]`, and (via `s_homeo[1u]` read
back at `:739`) the memory valence. We inject the shaping term at that one site
so it propagates to all three.

Edits (`coop_habituate_homeo`, thread-0 block `brain_passes.wgsl:160-195`):

- **Form `Φ` and `F`, add `F` to `raw_gradient` before the EMAs** (`:170`):

```wgsl
// Approach shaping (potential-based, policy-invariant — SCOPE: locked).
// Φ(s) = −APPROACH_SHAPING_GAIN · clamp(nearest_food_dist / SHAPING_RADIUS, 0, 1)
// F     = TD_DISCOUNT · Φ(s′) − Φ(s)   (telescopes to a constant; cannot corrupt
//                                        the eat objective, only speeds credit)
let d_norm = clamp(physics_state[phys_base_homeo + P_NEAREST_FOOD_DIST]
                   / SHAPING_RADIUS, 0.0, 1.0);
let potential = -APPROACH_SHAPING_GAIN * d_norm;
let prev_potential = physics_state[phys_base_homeo + P_PREV_POTENTIAL];
let shaping = TD_DISCOUNT * potential - prev_potential;
physics_state[phys_base_homeo + P_PREV_POTENTIAL] = potential;
let raw_gradient = energy_delta * ENERGY_WEIGHT
                 + integrity_delta * INTEGRITY_WEIGHT
                 + shaping;
```

```wgsl
// common.wgsl: initial gain (SCOPE: locked); promoted to a gene in plan 0004
// workstream 0004 if the 0001 remeasure shows the unlock lands.
const APPROACH_SHAPING_GAIN: f32 = 0.05;
```

Properties that make this safe:
- `phys_base_homeo` is already in scope in this block (`brain_passes.wgsl:161`);
  the read/write is thread-0-only, no barrier crossed.
- `F` is bounded: `|Φ| ≤ APPROACH_SHAPING_GAIN`, so `|shaping| ≤ (1+TD_DISCOUNT)·
  APPROACH_SHAPING_GAIN ≈ 0.099`, and the downstream `raw_gradient` flows through
  the existing EMA blends and the `MAX_TD_ERROR` clamp at `:425-428`; no new
  unbounded path.
- Optimal-policy invariance for the TD objective holds by construction (PBRS):
  the only behavioral effect is faster credit toward the unchanged eat objective.

### Split the actor's vector scale from the critic's

Today the forward/turn/critic weight updates all multiply by the single
`TD_VECTOR_SCALE = 1/ENCODED_DIMENSION = 1/128` (`common.wgsl:306`;
`brain_passes.wgsl:450-455`). That `1/128` is a critic-stability scale; applied
to the actor it throttles the policy step to `0.10/128 ≈ 8e-4`.

Edits:

- **New `ACTOR_VECTOR_SCALE` constant** (`common.wgsl`, beside `TD_VECTOR_SCALE`):

```wgsl
// Actor (forward/turn) weight-step scale, separate from the critic's
// TD_VECTOR_SCALE = 1/128 stability scale. Larger so the actor latches onto the
// now-sign-correct δ at a usable rate; bounded by the MAX_WEIGHT_NORM L2 ball.
const ACTOR_VECTOR_SCALE: f32 = 1.0 / 16.0;
```

- **Use it in the actor weight steps only** (`brain_passes.wgsl:452-455`): change
  the `O_ACTION_FORWARD_WEIGHTS` / `O_ACTION_TURN_WEIGHTS` updates to multiply by
  `ACTOR_VECTOR_SCALE`; leave the `O_VALUE_WEIGHTS` (critic) update at
  `TD_VECTOR_SCALE` (`:450-451`). The bias updates at `:435-440` are unchanged.

Properties that make this safe:
- The forward/turn weights are already L2-ball-clamped to `MAX_WEIGHT_NORM` every
  tick (`brain_passes.wgsl:476-497`), so a larger step cannot grow the weight
  norm unbounded; it only changes how fast the ball is traversed.
- The critic scale is untouched, so the critic's bootstrapping stability argument
  is preserved.

## 0002 — Reactive & valence layers on the external gradient (gated)

Today klinotaxis reads `gradient_deviation = s_homeo[3u] − s_homeo[4u]`
(`brain_passes.wgsl:653`) — `gradient_fast − gradient_medium`, two EMAs of the
interoceptive `raw_gradient`. After `0001` those EMAs already carry the shaping
term, but they mix it with the self-energy signal at EMA timescales. Memory
valence is written from `raw_gradient` (`:787,:813`); after `0001` a food-in-view
approach state accrues *positive* valence, which should already flip the
"negate approach → escape" pathology (`:527`) — `0002` confirms it and cleans up
only if recall still injects net-negative valence for food-in-view states.

**This workstream is gated** — see SCOPE (locked decisions); it opens only if the
`0001` remeasure moves alignment above the `0.62` band.

Edits (gated):

- **Klinotaxis on the external gradient** (`brain_passes.wgsl:648-655`): replace
  the deviation source with the *change in the approach potential* `Φ`
  (an external concentration compared across time), e.g. carry a fast/slow EMA of
  `d_norm` in two free `s_homeo`/`brain_state` slots and form
  `gradient_deviation` from their difference, leaving `KLINOTAXIS_SENSITIVITY` and
  the `clamp(…, 0.3, 3.0)` envelope as-is. Variant chosen at implementation time
  against the `0001` remeasure data; the construction is a delta on the existing
  two-EMA pattern, not a new module.
- **Memory valence confirmation** (`brain_passes.wgsl:780-789,:813`): no code
  change if the `0001` raw-gradient shaping already yields positive valence for
  food-in-view recall (verify via the recall-valence sign in a probe). If recall
  still escapes food, separate the approach component into the stored valence so
  `sim·valence` blends *toward* food.

Decision rule and gate in SCOPE (locked decisions).

## 0003 — Selection signal restoration

Today `composite_fitness` (`governor.rs:54-68`) multiplies a foraging+exploration
term by the survival gate `1/(1+death_count·0.5)`; the breeder couples unique
configs to repeats (`unique_count = (population_size/eval_repeats).max(1)`,
`governor.rs:944`, each repeated `:997-998`); the accept rule is a bare
`gen_avg ≥ parent_fitness` (`governor.rs:627`); and the q1→q4 within-life metric
lives only in `run_headless` (`headless.rs:173-265`), not on the `Governor`
production path.

Edits:

- **Foraging-primary fitness** (`composite_fitness`, `governor.rs:54-68`): make
  `food_per_1k_alive_ticks` the primary objective with survival as a bounded
  multiplier rather than the dominant gate, restoring dynamic range where the
  agents actually live so the new foraging variance is visible to selection. The
  exact form is pinned in TASKS; `food_target`/`total_grid_cells` denominators
  (`:58-59`) and the `ticks_alive` source (`P_TICKS_ALIVE`) already exist.

- **Decoupled experiment resolution** (`config.rs` + `governor.rs:941-1001`): add
  an `eval_unique_configs` knob independent of `population_size`/`eval_repeats`
  so the grid is `N_unique × M_repeats` (target ≈8–16 unique × 4–6 repeats),
  default `num_islands → 1` (`config.rs:250`), and ensure the per-repeat world
  seed actually varies (derive `seed + repeat_index` where the chunked evaluation
  groups by `agent_index/eval_repeats`, `governor.rs:559-567`). Backward-compatible
  via `#[serde(default)]`.

- **Significance guard** (`Governor::advance`, `governor.rs:609-655`): replace
  `if gen_avg >= parent_fitness` (`:627`) with
  `if gen_avg - parent_fitness > k · pooled_stderr`, where `pooled_stderr` is
  computed from the per-config repeat variance already available in `evaluate`
  (`governor.rs:559-567` groups repeats; surface their spread). `k` pinned in
  TASKS. This stops the spawn-bar from ratcheting on noise.

- **Within-life metric on the governor path** (`governor.rs` + `headless.rs`):
  lift the `quarter_rates` computation (`headless.rs:311-325`) onto the
  `Governor` so `evaluate`/`log_generation` (`governor.rs:482,:267`) persist a
  per-generation q1→q4 within-life food-rate, making within-life learning visible
  in any run (not just the headless side-tool). Reuse the existing
  `mutation_outcomes`-style DB column pattern; migrations are idempotent
  (`let _ = db.execute_batch("ALTER TABLE … ADD COLUMN …");`).

Properties that make this safe:
- `composite_fitness` is a pure function with unit tests; the new form is pinned
  with its own assertions and the existing fitness tests are updated in lockstep.
- New config fields are `#[serde(default)]`, so existing saved configs load
  unchanged; `num_islands` default change does not affect explicitly-set configs.

## 0004 — Heritable learning dynamics (gated)

Today `ACTION_WEIGHT_LEARNING_RATE` (`common.wgsl:277`), `KLINOTAXIS_SENSITIVITY`
(`:281`), `TD_DISCOUNT` (`:294`), `TD_LAMBDA` (`:298`), `CRITIC_LEARNING_RATE`
(`:301`), and the new `ACTOR_VECTOR_SCALE` are `const`. `BrainConfig`
(`config.rs:24-109`) packs its genes into the GPU config via `build_config_for`
(`buffers.rs:545-564`) at indices `CFG_*` (`buffers.rs:292-294`,
`common.wgsl:110-114`); evolution mutates genes in `mutate_config_with_strength`
(`agent.rs`, called from `governor.rs:899`).

**This workstream is gated** — see SCOPE; it opens only after the `0001`
remeasure shows the unlock lands (tuning steering dynamics before steering exists
is pointless and risks destabilizing the just-validated default).

Edits (gated):

- **New `BrainConfig` fields + `CFG_*` indices** for the six constants, each
  `#[serde(default = …)]` to the current `const` value, with heritable
  clamp ranges (e.g. `action_weight_learning_rate ∈ [0.01, 0.5]`,
  `td_discount ∈ [0.9, 0.995]`, `actor_vector_scale ∈ [1/128, 1/4]`). Bump
  `CONFIG_SIZE` and the `CFG_*` mirror test (`buffers.rs:710-712,:858`).
- **Read them in the shader** via `bc_f32(CFG_…)` (`common.wgsl:355-356`) at the
  use sites in `brain_passes.wgsl`, replacing the `const` reads.
- **Mutate + persist** in `mutate_config_with_strength` and the
  `mutation_outcomes` logging (`governor.rs:1514-1541`).

Note the `TD_DISCOUNT`↔sensory-lag coupling (Plan 0001 recalibrates `TD_DISCOUNT`
with stride): making it a gene interacts with `0005`; the clamp range keeps the
real-time horizon sane across the lag values `0005` may adopt.

## 0005 — Sensory lag & vision geometry (gated)

Today the `lag100` default (`config.rs:163-169`) and the 8×6 ray grid over a 90°
FOV (`common.wgsl:13-14,:203-204`; `phase_vision.wgsl:31-37`) were locked by Plan
0001 *under the spatially-blind reward*. With `0001` landed there is now steering
to act on, so the lag verdict is re-opened and the vision-geometry deferral
("until the learner can act on directional vision", `config.rs:73-83`) is
satisfied.

**This workstream is gated** — see SCOPE; opens only after the `0001` remeasure.

Edits (gated):

- **Heritable `vision_stride`** (`config.rs`, `agent.rs` mutation): make
  `vision_stride` mutable within `[1, MAX_VISION_STRIDE]`; re-run the three-arm
  stride/lag sweep from Plan 0001 *post-shaping*, adopting a smaller default only
  if the re-measured foraging gain beats the tps cost (Plan 0001's budget rule),
  recalibrating `TD_DISCOUNT` in the same change to preserve the real-time horizon.
- **Vision-row geometry** (`phase_vision.wgsl:31-37` ray pitch; default
  `vision_height`): give the grid an odd height so a ray row grazes the horizon
  (the `learning_probe` vision-acuity check at `integration.rs:2083+` already
  documents the 17×13 odd-grid rationale), so navigational-range ground food is
  no longer invisible between ray rows.

Decision rule and budget gate in SCOPE (locked decisions).

## Test strategy

- **New deterministic mechanism tests** gate the `0001` code tasks independent of
  the stochastic probes: an integration test that an agent closer to in-range
  food this brain tick than last produces a positive `shaping` contribution
  (read back via `P_NEAREST_FOOD_DIST` / `P_GRADIENT_OUT`), red before the
  shaping edit and green after; and an actor-step test that the forward/turn
  weight delta per unit `δ` scales with `ACTOR_VECTOR_SCALE`. Both embed the
  `GpuKernel::is_available()` self-skip guard.
- **`learning_probe_baseline_turn_alignment_is_chance`** (`integration.rs:1992`)
  and **`learning_probe_free_run_foraging_baseline`** (`:2031`) stay green through
  the `0001` code tasks (their 60-tick / 3000-tick horizons are too short for the
  throttled-then-modestly-raised actor to flip the band). The `0001` remeasure
  adds **`learning_probe_shaped_turn_alignment_beats_chance`** — a several-thousand-
  tick warm-up in the same fixed-bearing arena asserting alignment `> 0.62` — as
  the red→green learning evidence.
- **`composite_fitness` unit tests** are updated in lockstep with the `0003`
  fitness rework; `deterministic_across_batch_sizes` and the slot-map guard tests
  (`buffers.rs`) must stay green after the `PHYS_STRIDE` growth.
- CI gate that must stay green:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

## Interaction with prior work

- **Delivers Plan 0001's stated goal.** Plan 0001 ("ground the survival reward in
  space") landed the hazard/survival side — terminal-death TD update
  (`TERMINAL_DEATH_TD_ERROR`, `common.wgsl:317`), hazard touch, same-cycle
  interoception, the multiplicative survival gate, and the stride/lag sweep — but
  never added an approach term to `raw_gradient`. This plan adds exactly that term;
  it honors Plan 0001's terminal-death and interoception work unchanged.
- **Re-opens Plan 0001's `lag100` verdict, on purpose and gated.** That verdict
  was measured pre-shaping and is valid only there; `0005` re-measures it under
  the new reward and keeps the same tps-vs-gain budget rule.
- **Reworks the survival gate Plan 0001 adopted.** Plan 0001's Variant B
  multiplicative gate solved kamikaze foraging but compressed the live range;
  `0003` keeps the anti-kamikaze property while making `food_per_1k_alive_ticks`
  primary so selection regains dynamic range.
- **Honors Plan 0002/0003 (runtime).** No change to the worker boundary, dispatch
  fusion, or publication cadence; all edits are inside the per-agent kernel passes
  and the governor's CPU-side fitness/breeding.
- **Compiles `2026-06-14-claude-opus-48.md`.** Adopts its ranked interventions
  (#1 shaping → `0001`; #2 foraging-primary fitness, #3 resolution → `0003`; #4
  heritable constants → `0004`; #5 lag/vision → `0005`) and its secondaries
  (klinotaxis, memory → `0002`), with the three verification refinements recorded
  in SCOPE.
