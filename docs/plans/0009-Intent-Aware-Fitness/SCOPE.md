# Scope - Plan 0009

> Make evolutionary fitness reward *deliberate* foraging and *deliberate* danger
> avoidance, not the accidental by-products of raw speed. Attack the exploit at
> its mechanism — re-base every time-denominated score axis onto effort, make
> hazard exposure speed-invariant, make locomotion pay for itself — and give the
> agent a dedicated danger percept so avoidance becomes a learnable, measurable
> decision rather than an entangled guess. All behind flags, default no-op,
> graduated only on a measured speed-decoupling.

## Why this plan

Evolution found a degenerate strategy: raise the heritable `movement_speed`
gene and score rises on every axis at once — more food (a wider sweep clips more
food per lifetime), more exploration (more cells touched per lifetime), and
fewer deaths (a fast crossing spends fewer ticks inside a danger biome, so takes
less cumulative damage). None of this is deliberate behavior; it is brute
coverage. The user's intent is the opposite: more food should count only when it
is *sought*, and surviving danger should count only when it is *avoided*. The
load-bearing observations:

1. **Every fitness axis is denominated in TIME, and speed is the only gene that
   buys time.** `composite_fitness` (`governor.rs:84-102`) computes
   `food_per_1k = food_consumed / (ticks_alive/1000)`, `exploration =
   cells_explored / total_grid_cells`, and a `survival` multiplier from
   `death_count`. `movement_speed` converts directly to "more distance per unit
   time," so it inflates the foraging *rate*, the absolute cell coverage, and
   (via fewer-ticks-in-danger) the survival multiplier simultaneously. This is
   one dimensional flaw with three symptoms, not three bugs.
2. **Speed is metabolically free per unit distance.** Movement energy is exactly
   linear in speed (`kernel_tick.wgsl:163`:
   `movement_mag = … * (move_speed / 20.0)`) while distance covered is also
   linear in speed, so cost-per-distance is speed-invariant. There is no
   super-linear drag term to make speed pay for itself, so selection sees only
   the upside.
3. **Hazard damage is dwell-TIME based, so speed is partial immunity.** Hazard
   subtracts `WC_HAZARD_DAMAGE * integrity_scale` PER TICK while
   `biome == BIOME_DANGER` (`kernel_tick.wgsl:176-178`). Crossing a danger band
   at 2× speed halves the ticks inside and therefore halves the total damage —
   "sprint = immunity," with zero avoidance skill.
4. **Instant-death danger zones are rejected by the user and would destroy the
   data we want.** Making danger lethal would remove the agent before it can
   demonstrate a decision; dead agents generate no avoidance-decision telemetry.
   The hard constraint is that danger stays *graded and survivable* so the
   avoidance decision remains observable.
5. **The agent can already SEE danger, but only as an entangled color.** The
   vision raycast paints danger terrain a distinct dark-red
   (`phase_vision.wgsl:149-151`, `vec4(0.6, 0.2, 0.1, 1.0)`) versus green normal
   terrain — so danger ahead *is* in the visual field. But there is no dedicated
   danger percept: the brain never samples the biome itself, and the only
   proximal danger signal is `TOUCH_HAZARD` *after* contact
   (`phase_vision.wgsl:236-240`). Deliberate avoidance is therefore learnable in
   principle but weak — the credit path must learn "reddish-ahead = bad" purely
   from color, the project's known learning bottleneck (the credit path, not the
   encoder).
6. **The substrate for a symmetric danger percept already exists.** Foraging is
   taught by a *potential-based approach shaping* term: nearest-food bearing and
   distance (`P_NEAREST_FOOD_BEARING`, `P_NEAREST_FOOD_DISTANCE`,
   `kernel_tick.wgsl:296-356`) feed an approach potential into the reward
   (`brain_passes.wgsl:778-781`), policy-invariant by construction
   (`common.wgsl:388`). A danger *avoidance* potential is the exact mirror image
   and reuses the same machinery.
7. **`behavior_metric` already has danger columns waiting for data.** The table
   declares `danger_dwell_fraction` and `danger_exit_latency_ticks`
   (`governor.rs:1717-1726`) with a persistence test
   (`behavior_metric_table_persists_danger_metrics`), but production never
   populates them from real telemetry. Intent observability has a home already.
8. **Two live physics paths must move together.** The fused kernel
   (`kernel_tick.wgsl`) is the default, but the split passes
   (`phase_physics.wgsl` + `phase_death.wgsl`) are still compiled
   (`gpu_kernel.rs`). Both implement the per-tick hazard and the energy drain,
   and both carry a respawn save/restore whitelist (`kernel_tick.wgsl:408-448`
   and `phase_death.wgsl:38-76`). Every mechanism edit and every new
   generation-cumulative accumulator must land in BOTH or the split path
   silently diverges and new slots zero on first death.

Together: the agents win by sweeping, not seeking; the fix is to change the
*denominators* and the *hazard model* so brute coverage stops paying, and to
upgrade the danger percept so deliberate avoidance is both learnable and
measurable — without ever making danger lethal.

## In scope

- **0001 — Effort & exposure telemetry.** Add per-agent cumulative accumulators
  (`P_DISTANCE_TRAVELED`, `P_ENERGY_SPENT`, `P_DANGER_PATH_LENGTH`) in both
  physics paths, whitelisted across respawn, read back into `AgentFitness`, and
  populate `behavior_metric.danger_dwell_fraction`. No behavior or fitness change.
- **0002 — Dwell-invariant hazard (the survival-axis kill).** Replace per-tick
  hazard with damage proportional to path length through danger
  (`damage * step_len / reference_step`), `reference_step` derived from default
  speed so a default-speed crossing is byte-identical to today and a fast
  crossing integrates to the same total. Graded, never lethal.
- **0003 — Super-linear locomotor energetics (the keystone).** A drag exponent
  `speed_cost_exponent` makes per-tick movement energy scale as
  `pow(max(speed/20, 1.0), k)` — speed stops being free, the speed→fitness
  gradient becomes single-peaked, and both reward layers feel it. Default `1.0`
  (exact no-op).
- **0004 — Effort-rebased fitness (the foraging + exploration kill).** Re-base
  foraging on **energy spent** (`food / energy`, defeats camping where
  `food / distance` does not) and exploration on **distance** (cells per meter,
  capped by true coverage). Survival input (`death_count`) is unchanged — 0002
  already made it speed-invariant.
- **0005 — Danger percept, avoidance learning, and intent metric.** Compute a
  nearest-danger bearing/distance (bounded biome-grid scan), feed it as a
  dedicated sensory input, add a symmetric potential-based avoidance shaping
  term so steering away is learnable, and populate a `behavior_metric`
  sensed-then-turned avoidance-intent signal so the decision is measurable.
- **0006 — Validation and default-flip gate.** A headless A/B harness that
  measures the speed-decoupling and danger-data retention with flags off vs on,
  and a gated decision to flip the defaults only when the exploit is demonstrably
  gone and the population has not collapsed.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Time-denominated axes inflated by speed (1) | `0002`, `0003`, `0004` |
| Speed metabolically free per distance (2) | `0003` |
| Dwell-time hazard = sprint immunity (3) | `0002` |
| Instant death rejected; keep observability (4) | `0002`, `0005` |
| Danger visible only as entangled color (5) | `0005` |
| Approach-potential substrate exists (6) | `0005` |
| `behavior_metric` danger columns empty (7) | `0001`, `0005` |
| Two live physics paths + two whitelists (8) | `0001`, `0002`, `0003` |

## Locked decisions

- **Attack the mechanism, not the symptom.** The exploit succeeds because raw
  coverage pays on every axis. Patching one axis only re-routes the speed
  gradient through another. So the core fixes (`0002`, `0003`) are *kernel
  mechanism* changes that neutralize coverage itself; the fitness re-basing
  (`0004`) then cleans up the residual coverage gradient in scoring.
- **Move denominators from TIME to EFFORT — but foraging re-bases on ENERGY, not
  distance.** Food-per-distance is gameable by a near-stationary agent parked on
  respawning food (distance ≈ 0 → rate → ∞). Energy cannot be dodged that way: a
  parked agent still pays unavoidable metabolic + brain-drain energy
  (`kernel_tick.wgsl:165,194`), so food-per-energy does not saturate for a
  camper. Exploration (weight 0.15) re-bases on distance, capped by true
  coverage; the smaller residual is accepted rather than over-engineered.
- **Hazard is a graded path-length dose, never instant death.** This is the
  user's hard constraint. A default-speed crossing loses exactly the integrity
  it loses today; a fast crossing loses the same total. Danger stays survivable,
  `P_IN_DANGER_BIOME` is still published every tick, and agents still enter
  danger and generate decision data.
- **Danger is NOT a subtractive fitness penalty.** A penalty multiplier on
  danger dwell/entries collapses the population to zero crossings, which destroys
  the avoidance-decision data the user wants. The survival axis is fixed entirely
  by `0002`'s mechanism change (which keeps `death_count` as the input but makes
  it speed-invariant). `P_DANGER_PATH_LENGTH` and the avoidance-intent signal are
  observability telemetry and a `0005` learning signal — never selection terms.
- **Super-linear speed cost applies ABOVE baseline only.** `pow(max(speed/20,
  1.0), k)` — no sub-baseline discount, so there is no downward energy gradient
  that would ratchet speed to the floor (the "torpor drift" failure mode). The
  curve is single-peaked, not monotonic-down.
- **`movement_speed` bounds stay `[1.0, 100.0]`.** The single-peaked gradient
  from `0003` makes the clamp non-binding; mutation/crossover are untouched.
- **Intent is rewarded emergently first, perceived explicitly second.** Once
  speed is no longer free (`0003`), no longer dodges hazard (`0002`), and
  coverage is effort-denominated (`0004`), the only remaining gradient on every
  axis is finding food per unit energy (deliberate steering) and routing around
  danger (deliberate avoidance). `0005` then strengthens the danger percept so
  that avoidance is directly learnable and measurable, rather than left to an
  entangled red-color inference.
- **Everything ships behind flags, default a bit-exact no-op.**
  `speed_cost_exponent` defaults `1.0`; `danger_percept_enabled` defaults
  `false`; the effort-rebased fitness is gated by a formula switch. With all
  flags at their defaults the build is byte-identical to today (mirrors the
  `visual_cortex_enabled` precedent, `config.rs:439`).
- **Both physics paths and both respawn whitelists move together.** Every
  mechanism edit lands in `kernel_tick.wgsl` AND `phase_physics.wgsl`; every new
  cumulative slot is added to the respawn whitelist in `kernel_tick.wgsl` AND
  `phase_death.wgsl`. A parity/equivalence test locks them.
- **Success is a measured speed-decoupling, not a green build.** The claim is
  that the correlation between evolved `movement_speed` and fitness falls from
  strongly-positive to ≈0 / single-peaked (peak well below the clamp) while mean
  `ticks_alive` does not collapse and danger-decision data is still collected.
  The default-flip is gated on those numbers (`0006`), exactly as `0008` gated
  the encoder default on throughput.

## Out of scope

- **Instant-death (or near-instant) danger zones.** Explicitly rejected by the
  user: lethality removes the agent before it demonstrates a decision and
  destroys the avoidance-decision data. Danger stays graded.
- **Changing `movement_speed` bounds, mutation, or crossover.** The single-peaked
  gradient makes the clamp non-binding (Locked decisions). No gene-range change.
- **Reworking the encoder, memory, or TD(λ) credit path.** Plan 0007 owns the
  learner; plan 0004 adjudicated the credit path as the bottleneck. This plan
  changes *selection* and *percept*, not the learner architecture — the only
  brain-pass change is the additive, policy-invariant avoidance potential
  (`0005`), behind its flag.
- **Per-biome-conditional speed.** `move_speed` is a single global gene, not
  biome-conditional, so "slow down only inside danger" is not an available
  policy; no design effort is spent closing it.
- **A counterfactual / advantage-over-null-policy fitness.** Considered and
  rejected for v1: the speed-matched random-walker baseline is mis-specified
  (food only spawns in the food-rich biome, not world-uniform) and needs
  empirical per-speed calibration. The mechanism fixes achieve the same
  speed-invariance more cheaply.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
