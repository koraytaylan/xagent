# Learning Baseline: Phase-0 Probe Measurements

**Date:** 2026-06-10
**Plan:** [`docs/superpowers/plans/2026-06-10-emergent-learning-pathway.md`](../plans/2026-06-10-emergent-learning-pathway.md)
**Status:** Baseline recorded. Phases 1–3 must beat these numbers to merge.

## Purpose

Pin the current learner's measurable behavior before any learning-pathway
change lands. Every number below comes from a deterministic, seeded probe
(`crates/xagent-sandbox/tests/integration.rs`, "Learning Probe Tests"
section), so before/after comparisons are reproducible.

## Probe arena

Flat terrain, uniform food-rich biome (no hazards), 16 agents on a 4×4 grid
spaced 64 units apart (beyond the 30-unit vision range — fully independent
arenas). Each agent has exactly one food item at bearing ±atan(3/7)
(≈ ±23.2°, exactly on a vision ray column, alternating left/right) at
horizontal distance 5 — inside the band where the lowest below-horizon ray
row intersects ground-level food, outside both touch range (3.0) and
consume radius (2.0).

A geometry note that fell out of building this: on the default 8×6 grid the
**vertical** ray layout is the binding constraint on food visibility, not
horizontal acuity. The rows closest to horizontal sit at ≈ ±10.4°, so a ray
aimed below the horizon both passes the food's height band and strikes flat
ground at ≈ 5.5 units out. Ground-level food much beyond that distance falls
between ray rows and is invisible regardless of bearing. This sharpens the
Phase 3 vision work: row count / a horizon-grazing row matters at least as
much as column count.

## Baseline numbers

Measured on Mesa lavapipe (software Vulkan, `llvmpipe`), debug build,
seeds fixed in the tests. Rates are statistics over hundreds of samples, so
adapter-level float differences do not move them materially; re-record on
real hardware when convenient.

| Metric | Test | Value |
|---|---|---|
| Turn/bearing alignment (food visible, stationary agent) | `learning_probe_baseline_turn_alignment_is_chance` | **325/653 = 0.498** (chance = 0.5; asserted band 0.38–0.62) |
| Free-run foraging, default config, 3000 ticks | `learning_probe_free_run_foraging_baseline` | **food = 2 total (16 agents), 0.042 food/agent/1k-ticks, deaths = 0** |
| Food visibility at probe geometry | `learning_probe_food_is_visible` | **16/16 agents** see ≥ 1 food pixel |

## Reading the numbers

- **0.498 alignment** is the headline: with food *visible* and the agent
  free to turn, the turn direction carries zero information about where the
  food is. This is the direct, quantified form of the failure that keeps
  the parent issue open. A working spatial learner must push this decisively
  above 0.62 (the test's upper band edge documents the re-pin point).
- **0.042 food/agent/1k-ticks** with food spawned 5 units away and visible
  means essentially all eating is accidental contact. Klinotaxis alone does
  not convert visibility into approach in this arena.
- **16/16 visibility** confirms the probe measures the learner, not the
  information path.

## Per-generation metrics (headless)

`run_headless` now prints per generation:
`Food | Deaths | Food/1k-ticks | w_fwd | w_turn` — the behavioral signal
(food per 1k alive-ticks; see the metric note in the evolution-scale
section) plus the best agent's policy weight norms. For evolution-level A/B
runs, use a fixed `--db` seed config and compare these lines across the same
generation counts.

## Merge gates (from the plan)

- Phase 1 (TD credit): alignment probe above the 0.62 band edge with the
  band re-pinned, foraging rate above baseline over a fixed-seed
  20-generation headless run, TPS within 10% of pre-change.
- Phase 2 (encoder self-supervision): encoded-state separability test
  passes; alignment improves over Phase 1; TPS cost < 15%.
- Phase 3 (vision acuity): food visible at range in the probe; foraging rate
  (food per 1k alive-ticks) trends upward across generations.

## Phase 1 results (TD(λ) actor-critic)

Same lavapipe environment, same probe seeds. The windowed-REINFORCE credit
path (history ring, deadzone, tonic fallback, pain amplifier) was replaced
by a linear TD(λ) value head with per-dimension eligibility traces; the TD
error δ is the sole credit signal for critic, both policy channels, and the
encoder. New constants: `TD_DISCOUNT=0.97`, `TD_LAMBDA=0.9`,
`CRITIC_LEARNING_RATE=0.01`, `TD_VECTOR_SCALE=1/ENCODED_DIMENSION`,
`MAX_TD_ERROR=1.0`. Per-tick action-weight decay removed.

| Metric | Test | Baseline | Phase 1 |
|---|---|---|---|
| Trained turn/bearing alignment (confounded — see correction) | `learning_probe_mirrored_steering_is_chance` (renamed) | 0.498 (chance, untrained) | 0.643 (confounded); **0.52 confound-free** |
| Episode food (first half → second half of training) | same test | — | **588 → 730** of 960/half (rising) |
| Critic value under constant drain | `td_critic_tracks_metabolic_drain` | n/a | **−0.002** (correctly negative, finite, δ within clamp) |
| Trace bound across deaths | `td_traces_bounded_across_deaths` | n/a | bounded after 16 deaths (no cross-life leak) |
| Untrained stationary alignment | `learning_probe_baseline_turn_alignment_is_chance` | 0.498 | 0.50 (still chance — nothing to learn from with no reward events) |
| Free-run foraging (debug adapter, single run) | `learning_probe_free_run_foraging_baseline` | 2 food | 7 food (noisy single-seed; not a gate) |

The directional probe reported **0.643** (vs 0.498 chance) after 120 episodes
of food-reaching practice, and TPS was unchanged (the serial history-ring
credit loop is gone; the TD path is fully parallel across the 128 trace
dimensions).

> **Correction (Phase 3).** The 0.643 figure was **confounded** and overstated
> directional learning. In that protocol each agent always saw food on the
> same side in *both* training and evaluation, so a per-agent constant turn
> bias scored above chance without any vision-conditional steering. The
> confound-free protocol (`learning_probe_mirrored_steering_is_chance`, which
> mirrors the food side every training episode) lands at **0.52 — chance**.
> See the Phase 3 section. What Phase 1 genuinely delivered is intact: the
> TD(λ) critic learns (value tracks drain, traces stay episodic), foraging and
> fitness rise at evolution scale, and the brittle deadzone/tonic/pain credit
> pile is gone. What it did *not* deliver is genuine vision-conditional
> steering — that remains open.

## Phase 1 at evolution scale (validation)

The probe measures within-lifetime learning in isolation; the real question
is whether it compounds across generations through inheritance. A 24-generation
headless run (`--config` with `tick_budget=120000`, `population_size=12`,
`patience` disabled, world seed 42) on lavapipe:

| Signal | Gen 0 | Gen ~22 | Trend |
|---|---|---|---|
| Champion composite fitness (`survival·0.4 + foraging·0.3 + exploration·0.3`) | 0.239 | 0.386 | **+62%**, rising then plateauing |
| Foraging rate (food per 1k alive-ticks, population) | 0.164 | 0.285 | **+74%**, monotone-ish |
| Total food consumed (population) | ~240 | ~400 | **+60%** |
| Inherited policy weight norms (`w_fwd` / `w_turn`) | ~0.005 | ~0.29 | monotone growth, bounded |

The monotone growth of the inherited policy weight norms is the direct
evidence that learned weights persist and **compound across generations** via
the inherit/mutate path (not just within a lifetime). Champion fitness rising
+62% while the survival term is weighted 0.4 confirms the best lineage is
genuinely improving, not just trading deaths for food.

**Metric note.** An earlier population-aggregate *food-per-life* proxy
(`food / (agents + deaths)`) *fell* over the same run because its denominator
is dominated by the aggregate death count across all 12 agents (active
foragers and exploratory mutants die often). It was replaced by
**food-per-1k-alive-ticks**, which normalizes by accrued lifetime and is
robust to death count — that is the metric `run_headless` now prints, and it
rises as expected. Death rate climbing alongside foraging flags
danger-avoidance / energy economics as a live tension (the next plausibly
binding constraint), consistent with the journey's energy-economics notes.

## Phase 2 (encoder self-supervision) — negative result, not merged

Tied-weight vision reconstruction was implemented and measured against the
Phase-1 gate (same training seed): alignment **0.603** at rate 0.001 and
**0.613** at 0.0003, versus **0.643** for Phase 1 with no reconstruction.
Reconstruction was neutral-to-slightly-negative at every rate and never
cleared the "improves over Phase 1" bar, so it was reverted (see the plan's
"Phase 2 outcome" section for the mechanism analysis).

> **Caveat (Phase 3) and resolution.** This section's "TD(λ) already
> extracts the food-direction signal" conclusion rested on the
> **confounded** 0.643 directional number, which briefly re-opened the
> encoder question under the honest probe. Two follow-up measurements then
> settled it: the `encoder_food_side_separability_diagnostic` probe shows
> the random encoder *preserves* food-left/right separability (≈ 4× margin
> over within-class nuisance in angular distance — see the Phase 3 section
> for the numbers), and re-testing this reconstruction objective
> against the honest mirrored gate landed at exactly chance (0.500). The
> revert stands, and the encoder is confirmed not to be the binding
> constraint — see the Phase 3 section for the measurements.

## Phase 3 (vision acuity) — range-visibility fixed; directional confound found

Two outcomes, one expected and one not.

**1. Vision acuity (the planned work).** The **17×13** odd grid was
implemented and measured, but the **default stays 8×6** (see the decision
below). The odd row/column counts put one ray row exactly on the horizon
(passing a constant 0.65 below eye level — inside the 1.0 food hit radius)
and one column straight ahead. `vision_horizon_row_sees_food_at_range`
proves the payoff with explicit configs: at 17×13 ground-level food is
visible at distances {5,10,15,20,25}; at 8×6 only at distance 5 (the lowest
below-horizon ray strikes flat ground ≈ 5.5 units out, so distal food falls
between rows). This is a real information defect the odd grid fixes.

At evolution scale (16 generations, seed 42) the foraging-rate trend at
17×13 (0.192 → 0.254) is **comparable to 8×6** (0.164 → 0.285) — no clear
behavioral win from the extra visibility. Consistent with outcome 2: the
information is now available, but the learner cannot act on directional
vision, so it is not cashed in.

**Decision: default reverted to 8×6.** 17×13 costs 4.3× more features
(265 → 1130) with no measured behavioral win, and the directional learner
cannot use the added information yet — so shipping it as the default would
be an unmeasured-benefit cost increase (the same discipline that reverted
Phase 2). The odd-grid capability and its range-visibility test are kept;
17×13 becomes the default once directional steering works and can exploit
it. The information path is *proven fixable*, which is the prerequisite the
journey's rule #1 ("verify the information path before optimizing the
algorithm") asks for.

**2. The directional-steering confound (unplanned, more important).**
Validating Phase 1 at the new resolution surfaced that the 0.643 directional
result was an artifact. The probe trained each agent with food always on one
side and evaluated on the same side, so a per-agent constant turn bias scored
above chance. The confound-free protocol mirrors the food side every training
episode (`learning_probe_mirrored_steering_is_chance`):

| Protocol | Resolution | Trained alignment |
|---|---|---|
| Unmirrored (confounded) | 8×6 | 0.643 |
| Unmirrored (confounded) | 17×13 | 0.586 |
| **Mirrored (honest)** | **17×13** | **0.52 (chance)** |

More training episodes do not move the mirrored number (120 → 0.523,
240 → 0.516). **Genuine vision-conditional steering — "turn toward the side
where food is seen" — is not being learned**, at either resolution. The
hypothesis this initially suggested — the one issue #13 and both reviews
named — was that a random-projection encoder does not make "food-left" and
"food-right" linearly separable for the policy's readout, so a constant bias
is learnable but a conditional response is not. Two follow-up measurements
**refuted** that hypothesis: the
`encoder_food_side_separability_diagnostic` probe shows the random encoder
*preserves* the food-side direction — comparing **angular distances**
(acos of the cosine similarity), right-vs-left scenes are ≈ 0.085 rad apart
(cosine 0.9964) while two same-side scenes at slightly different distances
are ≈ 0.020 rad apart (cosine 0.9998), an ≈ 4× between/within margin — and
re-testing tied-weight reconstruction against the honest mirrored gate
landed at exactly chance (0.500), so a "better" encoder does not produce
steering either. The open
bottleneck is therefore the **credit/learning dynamics under movement
nuisance** (the bearing the credit should explain changes as the agent
moves, while the reward stays sparse and delayed), not representability.

**Net:** Phase 3 delivers the information substrate (food visible at range)
and, more valuably, a confound-free directional probe that correctly reports
the open problem. The TD(λ) critic and the evolution-scale foraging/fitness
/survival gains from Phase 1 stand; the specific claim of learned directional
steering does not.

## Hazard probe baseline

Half-plane danger arena (biome 2 for x < 0, food-rich for x ≥ 0) on the
standard probe geometry (flat, 16 agents). Agents spawned at x =
−HAZARD_PROBE_START_DEPTH (10 units inside danger), y=1, z-spread ±60 step 8,
facing +Z (parallel to the x=0 boundary). Episode: 600 physics ticks
(HAZARD_PROBE_EPISODE_TICKS), sampled every 5 ticks. 3 episodes (48 trials
total).

Hazard-exit latency = first sample tick where x ≥ 0 and still alive (death
count unchanged from start of episode). Death = death count increased before
any exit. Timeouts (neither) possible but did not occur in baseline.

**Recorded numbers (2026-06-12, macOS Metal/wgpu adapter, seeds fixed in test):**

| Metric | Value | Raw count |
|---|---|---|
| exit_fraction | 0.104 | 5/48 |
| mean_exit_latency (of exits) | 145.0 | — |
| death_fraction | 0.896 | 43/48 |

**Pinned bands (in `hazard_probe_exit_latency_baseline`):** ±50% relative
around the post-grounding re-pinned values (see "Hazard probe re-measure
after grounding" below for before/after):
exit_fraction [0.094, 0.282], mean_exit_latency [68.6, 205.8],
death_fraction [0.406, 1.218].

**Gate:** workstream 0002 must move mean exit latency or death fraction
outside the pinned bands to claim a behavioral win. (Exit fraction is
derivative; the primary economics signals are latency to escape and death
rate under hazard exposure.) The same re-pin discipline as the steering
probes: when the numbers move, update the bands + this spec + the inline
comment.

## Hazard probe re-measure after grounding (2026-06-12)

Re-ran the probe (`cargo test -p xagent-sandbox --test integration
hazard_probe_exit_latency_baseline -- --nocapture`) with the three grounding
changes (`terminal-death-update`, `hazard-edge-touch`,
`same-cycle-interoception`) in place on the same macOS Metal adapter.

**Printed result:** `hazard probe baseline: trials=48 exit_fraction=0.188 mean_exit_latency=137.2 death_fraction=0.812`

| Metric | Before (pre-grounding) | After (post-grounding) | Raw count (after) |
|---|---|---|---|
| exit_fraction | 0.104 | **0.188** | 9/48 |
| mean_exit_latency (of exits) | 145.0 | 137.2 | — |
| death_fraction | 0.896 | 0.812 | 39/48 |

**Verdict: improved** beyond the original pinned bands (exit_fraction 0.188 > 0.156 upper edge; death fraction moved inward from 0.896). Grounding produced a clear behavioral shift on the direct probe: nearly double the escape rate, fewer deaths. The danger pathway now supplies a usable teaching signal (terminal TD on death + timely touch/interoception), satisfying the prerequisite. (Per the plan's mirrored-probe precedent, the signal existing does not guarantee that the policy will exploit it at evolution scale.)

**Pinned bands re-pinned** (in test + this spec) to ±50% relative around the post-grounding values (now the baseline for subsequent work):

exit_fraction [0.094, 0.282], mean_exit_latency [68.6, 205.8],
death_fraction [0.406, 1.218].

(The test asserts were updated to these bands during the grounding sub-tasks and confirmed on final re-measure.)

## Evolution-scale check (post-grounding control)

Generated `experiments/control.json` (edited per task: tick_budget=120000,
population_size=12, max_generations=16, seed=42) and ran:

```
cargo run --release -p xagent-sandbox -- --no-render --config experiments/control.json --db experiments/control.db
```

(15 generations completed; wall ~87s on macOS Metal, ~20900 ticks/sec.)

Key lines (`Food | Deaths | Food/1k-ticks`):

- Gen 0: Food: 236 | Deaths: 516 | Food/1k-ticks: 0.164
- Gen 1: 287 | 767 | 0.200
- Gen 2: 273 | 843 | 0.190
- Gen 3: 274 | 684 | 0.191
- Gen 4: 316 | 997 | 0.220
- Gen 5: 337 | 1201 | 0.235
- Gen 6: 357 | 1268 | 0.249
- Gen 7: 367 | 1951 | 0.257
- Gen 8: 306 | 965 | 0.213
- Gen 9: 360 | 1855 | 0.252
- Gen 10: 375 | 2028 | 0.262
- Gen 11: 338 | 1431 | 0.236
- Gen 12: 379 | 2164 | 0.265
- Gen 13: 396 | 2372 | 0.278
- Gen 14: 397 | 2323 | 0.278

**Comparison to Phase-1 numbers in this spec** (pre-grounding 24-gen reference, same config params, seed 42):

- Gen 0 matches exactly (food ~240, rate 0.164).
- Foraging rate rose to 0.278 by gen 14 (comparable to Phase-1's 0.285 at gen~22); total food ~397 vs Phase-1 ~400.
- **Deaths did not trend down relative to food.** Deaths climbed (516 → 2323) while food rose modestly; deaths-per-food worsened from ~2.2 to ~5.8. The grounding/terminal lesson improved the within-episode probe (escape behavior), but at population/evolution scale the dynamics still show high mortality alongside foraging gains (consistent with the "live tension" noted in the Phase-1 section). The signal is present; selection pressure to exploit it for lower death may need more generations, larger pop, or adjusted fitness weights.

The post-grounding control run is recorded here for the learning baseline.

## Predictor forward objective (GATED, 2026-06-12)

**Implementation:** In `brain_passes.wgsl`:
- `coop_predict_and_act`: replaced predictor matmul with train-then-predict on `s_encoded` (threads 0..PREDICTOR_DIMENSION) using `O_PREV_PREDICTION` vs current encoded, grad via `O_PREV_ENCODED`; new prediction also from `s_encoded`.
- Pass 6 thread 0 novelty error loop now targets `s_encoded[d]` (was `s_habituated`); deleted the end-of-block recompute of `s_pred_error` (keeps the true forward error).
- Pass 7: deleted block 7a (old this-tick identity train on habituated); context-weight adaptation now uses the shared `s_pred_error`.
- `coop_habituate_homeo`: deleted the `O_PREV_ENCODED` write (read stays); appended 7g at very end of `coop_learn_and_store` to publish `s_encoded` for next tick's habituation delta + predictor input.

Predictor now learns forward dynamics in encoded space; habituation remains downstream attention.

**Gate execution (macOS Metal adapter):**
- Full relevant suite: `td_critic_tracks_metabolic_drain` (mean value −0.00204, passes), `learning_probe_mirrored_steering_is_chance` (0.524 alignment, within chance band ~0.5), hazard probe, etc.
- Re-ran `hazard_probe_exit_latency_baseline`: trials=48 exit_fraction=0.188 mean_exit_latency=130.6 death_fraction=0.812 (statistically matches prior post-grounding 0.188/137.2/0.812; bands hold, no regression).
- Re-ran reduced fixed-seed headless control (seed=42, pop=4, 20k-tick budget, 3 gens, --no-render): gen 0 rate 0.451 deaths 53; gen 1 rate 0.639 deaths 32 — foraging up, deaths down; metrics consistent with no behavioral regression (as expected for this objective-consistency change).
- All cargo fmt/clippy/test requirements satisfied for the change (probes exercised the paths).

**prediction_error telemetry ranges** (sampled from `td_critic_tracks_metabolic_drain` probe: 300 brain ticks in constant-drain arena, 16 agents; "early" at tick 10, "late" at tick 299; values are the per-tick `P_PREDICTION_ERROR`):

Before (pass 7a identity-autoencoder objective training *this* prediction vs *this* habituated; novelty compared last vs habituated):
- early: mean 0.7558 range [0.5826, 1.0000]
- late:  mean 0.0981 range [0.0324, 0.2204]
  (error drops but plateaus at state-change / habituation magnitude in stretches)

After (forward model trains last prediction vs *arrived* `s_encoded`; single error from prev vs encoded drives both novelty and context weight; `O_PREV_ENCODED` timing fixed):
- early: mean 0.0860 range [0.0403, 0.1741]
- late:  mean 0.0269 range [0.0047, 0.0514]
  (forward error now markedly lower overall and continues to decrease over lifetime in quiet stretches, approaching near-zero for stable dynamics — objective now matches the forward prediction semantics used by novelty.)

**Decision:** All probe bands held (mirrored steering, td_critic, hazard); fixed-seed control showed no regression. Change kept (objective consistency achieved; small behavioral side-effect as predicted).

(The full 16-gen control re-run with identical params as the post-grounding section would be expected to produce nearly identical per-gen Food/Deaths/rate sequences within seed noise; the distinguishing signal is the telemetry ranges above.)

## Stride/lag sweep — three-arm A/B at evolution scale (2026-06-13)

Three headless arms, identical except for the stride pair (which together set
the sensory lag = `vision_stride × brain_tick_stride`), run on the same
release binary, macOS Metal adapter. Shared params: `world.seed = 42`,
`governor.tick_budget = 120000`, `population_size = 12`,
`max_generations = 16`, `patience = 5`. All three terminated identically at
**15 generations** (patience, reproducible at seed 42), so the comparison is
gen-for-gen fair. Configs: `experiments/{lag100-control,lag10,lag2}.json`.

| Arm | `brain_tick_stride` | `vision_stride` | sensory lag | ticks/sec (mean) | cost vs control | Food/1k-ticks (gen0 → gen14, peak) | Deaths (gen0 → gen14) | deaths/food @ gen14 |
|---|---|---|---|---|---|---|---|---|
| `lag100-control` | 10 | 10 | 100 | **~21,500** | — | 0.187 → 0.257 (peak 0.276) | 599 → 1625 | **4.41** |
| `lag10` | 2 | 5 | 10 | **~4,220** | **5.1× slower (~80%)** | 0.468 → 2.076 (peak 2.076) | 494 → 2410 | **0.81** |
| `lag2` | 1 | 2 | 2 | **~1,710** | **12.6× slower (~92%)** | 0.536 → 3.712 (peak 5.443) | 619 → 1072 | **0.20** |

**Two findings, both recorded:**

1. **Cost: both lower-lag arms blow the TPS budget by an order of
   magnitude.** The locked decision rule (SCOPE) adopts a lower-lag default
   only if its ticks/sec cost versus control is **under ~30%** *and* it beats
   control on the fixed seed. `lag10` runs at ~80% cost (5.1× slower), `lag2`
   at ~92% cost (12.6× slower) — both far over the ~30% gate. The cost scales
   as expected: lower `brain_tick_stride` multiplies brain ticks per physics
   tick (×5 / ×10), lower lag multiplies vision passes (×10 / ×50).

2. **Behavior: lower lag dramatically improves open-world foraging.** `lag10`
   reaches ~8× and `lag2` ~14× the control's final food/1k-ticks, and the
   kamikaze pattern (deaths ≫ food) collapses — deaths-per-food falls from
   **4.41** (control) to **0.81** (`lag10`) to **0.20** (`lag2`). The effect
   is large and consistent across generations (control 0.16–0.28 vs `lag2`
   0.46–5.44, non-overlapping for most gens), not seed noise.

**Decision (locked rule applied): no new default — lag 100 stays.** Neither
lower-lag arm satisfies the hard `<30%` cost gate, so despite the strong
behavioral win the default stride pair is unchanged. The corrected
`TD_DISCOUNT` comment (real-time horizon formula) is the only code change;
no `config.rs` default change, no test re-pin, no γ recalibration.

**This is the answer to the reviews' lag experiment.** Timeliness of action
on fresh sensory state is a *large* lever for open-world foraging efficiency
(the agent stops blowing past food during the 100-tick blind window and acts
on present-moment energy/touch). But this is a **separate axis** from the
directional-steering deficit the mirrored probe isolates — that probe trains
at lag 1 and still lands at chance, and nothing here changes that. So the
reviews' dominant hypothesis ("remove the lag and learning appears") is
half-right in a way the probe alone could not show: lag removal does not
manufacture vision-conditional steering, but it does unlock a big chunk of
foraging the lagged frame was leaving on the table. The TPS budget — not the
behavior — is what keeps it out of the default. **Flag for future
performance work:** if the kernel gets ~5× faster, `lag10` lands inside the
budget and becomes a clear adopt; revisit then.

## Fitness rework — Variant B (multiplicative survival), 2026-06-13

The post-grounding data confirmed **both** fitness pathologies the additive
`survival·0.4 + foraging·0.3 + exploration·0.3` formula allowed:

- **Kamikaze foraging.** The post-grounding control's deaths-per-food
  *worsened* across the run (deaths climbed far faster than food), and the
  stride sweep's control reproduced it (deaths-per-food 4.41 at gen 14).
- **Within-lifetime learning invisible.** The new `Learn q1→q4` metric is
  flat-to-declining every generation in the control (e.g. 0.251 → 0.164,
  0.291 → 0.252) while cross-generation foraging rises — improvement is not
  happening *within* a lifetime.

Per the locked rule ("both confirmed → multiplicative first, improvement
second, separately measured"), this task ships **Variant B** only; the
improvement term (Variant A) is the separate gated follow-up.

**Change.** `composite_fitness` (extracted as a pure, unit-tested helper in
`governor.rs`) becomes:

```rust
survival * (foraging * 0.5 + exploration * 0.5)
```

Survival now gates the whole score multiplicatively instead of contributing
an additive 0.4; the foraging/exploration split keeps their prior 1:1 ratio.
`composite_fitness_gates_score_multiplicatively_on_survival` asserts the
formula exactly (careful forager 0.375 vs same-foraging-but-4-deaths 0.125;
foraging cap; zero-on-no-forage).

**Fixed-seed comparison (seed 42, `tick_budget` 120000, pop 12, 15 gens, the
same `experiments/lag100-control.json` config as the stride-sweep control,
macOS Metal):**

| Metric | Additive (before) | Variant B multiplicative (after) | Δ |
|---|---|---|---|
| Total deaths (Σ gens) | 20,336 | 8,546 | **−58%** |
| Total food (Σ gens) | 5,056 | 3,836 | −24% |
| **Deaths-per-food (run total)** | **4.02** | **2.23** | **−45%** |
| Deaths gen0 → gen14 | 599 → 1625 | 555 → 738 | trend flattened |
| Peak generation deaths | 2,036 (gen7) | 1,041 (gen13) | −49% |
| Food/1k-ticks gen0 → gen14 | 0.187 → 0.257 | 0.180 → 0.196 | foraging slightly lower |

**Verdict: Variant B does exactly what it is for.** Multiplicative survival
nearly halves deaths-per-food and flattens the death trend; the population
stops being rewarded for foraging by dying. Total food drops modestly
(−24%) while deaths drop more than twice as much (−58%) — the intended
trade. Inherited policy weight norms still grow (`w_fwd`/`w_turn`
0.006 → 0.308), so the learning machinery is unaffected; the `Learn q1→q4`
signal stays flat-to-declining, which is expected — Variant B targets the
kamikaze economics, not within-lifetime learning. That remaining flatness is
the standing case for the Variant A improvement term as the next separate
measurement.

## Simulation throughput ceiling — submit fusion (Plan 0003, 2026-06-14)

Plan 0003 instrumented the per-batch dispatch path (`DispatchProbe`, default-off
`XAGENT_PROBE_GPU_WAIT` / `XAGENT_SKIP_GLOBAL_VISION` knobs, submit-return vs
GPU-complete wall counters surfaced in `Worker::maybe_log_counters` as
`[SIM-PROBE]` and in `bench::run_bench` as `[BENCH-PROBE]`), then moved the
kernel's per-batch `start_tick` to a push constant so every full kernel-batch in
a `dispatch_ticks` call shares one `world_config` uniform write and fuses into
one command encoder + one submit per `MAX_FUSED_BATCHES` (= 24) batches. The
worker dispatch cap was widened to `kernel_batch_size × MAX_FUSED_BATCHES` so a
high-speed backlog actually fuses.

**Bit-identical gate (passes):** `deterministic_across_batch_sizes` and the new
`fused_dispatch_matches_split` (one fused `dispatch_ticks(0, 1037)` == ten
one-batch calls + a final 37-tick call, exercising both the remainder-cycles and
physics-remainder paths) assert byte-equal final physics state. Verified on Mesa
lavapipe (`llvmpipe`).

**Fusion ratio (hardware-independent):** `--bench --bench-ticks 24000
--bench-agents 1` issues one `dispatch_ticks(0, 24000)` = 240 kernel-batches.
After fusion it records **10 submits** (= ⌈240 / 24⌉) instead of the former
**240** — the one-submit-per-100-tick tax is removed. This ratio does not depend
on the GPU.

**Three-arm wall-clock — Mesa lavapipe (`llvmpipe`, CPU software rasterizer),
NOT representative of the target Metal/discrete GPU.** Recorded only to validate
that the probe knobs are wired correctly end-to-end; absolute tps and the
submit-vs-complete split on a CPU rasterizer say nothing about the ≈20 k-tps /
1000× Metal ceiling this plan targets. `--bench --bench-ticks 24000
--bench-agents 1`, one run each:

| Arm | Env | tps | submits | submit-return ns/batch | gpu-complete ns/batch |
|---|---|---|---|---|---|
| (a) default | — | 10,754 | 10 | 8,794,589 | — (not measured) |
| (b) gpu-wait | `XAGENT_PROBE_GPU_WAIT=1` | 19,524 | 10 | 4,614,971 | 5,119,094 |
| (c) skip g+v | `XAGENT_SKIP_GLOBAL_VISION=1` | 22,138 | 10 | 4,059,447 | — (not measured) |

These lavapipe numbers do not adjudicate the target ceiling (CPU rasterizer);
they only validate the probe wiring. The adjudicating run was done on target.

**On-target result — macOS/Metal, `--bench-phase-ab --bench-ticks 1000000
--bench-agents 10` (THE adjudicating run, 2026-06-15):**

| Arm | tps | Δ vs baseline | batches | submits |
|---|---|---|---|---|
| full (baseline) | 22,989 | — | 10,000 | 417 |
| skip global | 23,595 | **+3%** | 10,000 | 417 |
| skip vision | 24,422 | +6% | 10,000 | 417 |
| skip global+vision | 24,710 | +7% | 10,000 | 417 |

**Verdict — both 0002 and 0004 targets are ruled out; the kernel/brain pass is
the ceiling.** Three facts from this one table:

1. **Fusion is engaged** (417 submits for 10 000 batches ≈ ⌈10000/24⌉) yet
   baseline tps ≈23 k still sits at the originally observed ≈20 k ceiling — so
   collapsing submits (0002) did not move it. The limiter was never CPU submit
   overhead (the `2026-06-14-grok-43.md` review's attribution is falsified).
2. **The `global` pass is ≈3%** (skip-global +3%) and **global+vision together
   ≈7%** — so the single-workgroup `global` pass is *not* the residual floor.
   Workstream **0004 is REJECTED with on-target evidence** (see
   `docs/plans/0003-Simulation-Throughput-Ceiling/0004-GLOBAL-PASS-DECISION.md`);
   parallelizing it could recover ≤3%, not worth the determinism risk.
3. **≥93% of per-batch wall time is the `prepare`+`kernel` (fused brain)
   dispatch.** Combined with the long-standing observation that agent count
   10→4 changes nothing, this points at a **latency-bound, GPU-under-occupied
   serial brain chain** at low agent counts — a new plan's subject, outside 0003.

Repro (release binary):

```
cargo build --release -p xagent-sandbox
# One-command pass-isolation A/B (full / skip global / skip vision / skip both):
./target/release/xagent --bench-phase-ab --bench-ticks 1000000 --bench-agents 10
# Equivalent env knobs for arm-by-arm runs:
#   XAGENT_SKIP_GLOBAL=1 / XAGENT_SKIP_VISION=1 / XAGENT_PROBE_GPU_WAIT=1
# GUI at 1000× with RUST_LOG=debug → read [SIM-PROBE] (submits ≪ kernel_batches).
```

## GPU occupancy & brain-pass profile (Plan 0005, 2026-06-15)

Plan 0005 lands the measurement wiring for two levers the 0003 verdict above
isolated (occupancy + per-workgroup barrier depth). All knobs default to a
byte-identical dispatch; the determinism gate
(`deterministic_across_batch_sizes` + `fused_dispatch_matches_split`) stays
green with them unset, and `cargo fmt`/`clippy -D warnings`/`test` are green.

**Wiring landed (this branch):**

- `--bench-agent-sweep` (`bench::run_agent_sweep`): for each N in
  `[1, 4, 10, 50, 100, 200, 400, 1000]` runs a fixed `--bench-ticks` through one
  fused `dispatch_batch(0, ticks)` and prints tps + agent-ticks/sec (tps × N),
  then flags the N that maximizes agent-ticks/sec as the occupancy knee.
- `GovernorConfig::population_size` default stays at **10** — raising it to the
  occupancy knee regresses evolution (shared-world food competition; see (4)).
  The knee (≈200) is documented in `default_population_size` but not used for the
  population until evaluation is arena-isolated.
- `GpuKernel::has_subgroup()` accessor + an explicit `[GpuKernel] top-K recall
  path: …` `log::info!` at construction (read off `RUST_LOG=info`).
- `XAGENT_KERNEL_PASS_LIMIT=k` (default 7): runs only the first `k` of the seven
  cooperative passes in `brain_tick_inner`, carried via the kernel push
  constant's second word (`KernelPushConstants.pass_limit`). Default 7 ⇒
  byte-identical; `0..6` deliberately corrupt results for timing only.

**Repro (release binary, target macOS/Metal GPU required for numbers):**

```
cargo build --release -p xagent-sandbox
# (1) Occupancy sweep — locate the knee:
./target/release/xagent --bench-agent-sweep --bench-ticks 200000
# (2) Subgroup top-K path fact:
RUST_LOG=info ./target/release/xagent --bench --bench-ticks 1000 --bench-agents 10 2>&1 \
  | grep 'top-K recall path'
# (3) Per-cooperative-pass cumulative cost — sweep the limit 0..7:
for k in 0 1 2 3 4 5 6 7; do \
  echo "limit=$k"; XAGENT_KERNEL_PASS_LIMIT=$k \
  ./target/release/xagent --bench --bench-ticks 200000 --bench-agents 200; done
# (4) Fixed-seed evolution comparison (no-regression check), N=10 vs N=200,
#     same --seed / tick_budget / generations, headless:
./target/release/xagent --no-render --seed 42 --generations 8 --config <pop10.json>
./target/release/xagent --no-render --seed 42 --generations 8 --config <pop192.json>
```

> **On-target status (2026-06-15):** (2) subgroup fact and (3) per-pass profile
> are recorded below from a macOS/Metal run. (1) the occupancy sweep and (4) the
> fixed-seed evolution comparison are still **pending** an on-target run — the
> harness wiring was developed in a Linux container with no GPU adapter
> (`GpuKernel::is_available()` == false). Run the (1)/(4) commands above on the
> reference machine and paste results into those tables.

**(1) Occupancy sweep — `--bench-agent-sweep --bench-ticks 200000`, on target
(macOS/Metal, 2026-06-15):**

| N | tps | agent-ticks/sec |
|---|---|---|
| 1 | 23,253 | 23,253 |
| 4 | 23,028 | 92,112 |
| 10 | 22,935 | 229,348 |
| 50 | 20,888 | 1,044,394 |
| 100 | 17,218 | 1,721,793 |
| **200** | **11,990** | **2,397,914 ← knee** |
| 400 | 5,191 | 2,076,200 |
| 1000 | 2,387 | 2,387,226 |

tps is flat from N=1 to ~N=10 (latency-bound, GPU idle), then useful throughput
(agent-ticks/sec) climbs to a peak at **N=200 (2.40 M)** and falls off beyond it
(N=400 → 2.08 M). N=200 is the *smallest* N at peak useful throughput, so it also
minimizes per-generation wall time (N=1000 matches its agent-ticks/sec but at ~5×
the wall time). This is a ~10× useful-throughput gain over the old default of 10
(229 k → 2.40 M). **Knee → shipped default = 200** (multiple of `eval_repeats` →
100 unique genomes/gen). Safe max 1000 (runs; N=5000 did not complete).

**(2) Subgroup top-K path on target (2026-06-15, macOS/Metal):**
`workgroup-memory bitonic fallback (barrier-dense)` — the subgroup-accelerated
top-K path is **inactive** on this device. (Irrelevant in light of (3): the
top-K pass is not the bottleneck.)

**(3) Per-cooperative-pass cumulative cost — `XAGENT_KERNEL_PASS_LIMIT` sweep,
on target (macOS/Metal, `--bench-ticks 200000 --bench-agents 200`, 2026-06-15):**

| limit | pass added | wall | tps | Δ wall = pass cost |
|---|---|---|---|---|
| 0 | — (physics+food+death+vision, no brain) | 1.66s | 120,165 | floor |
| 1 | feature_extract | 1.66s | 120,501 | ~0.00s |
| 2 | **encode** | 3.61s | 55,415 | **+1.95s** |
| 3 | habituate_homeo | 3.60s | 55,537 | ~0.00s |
| 4 | recall_score | 3.87s | 51,672 | +0.27s |
| 5 | recall_topk | 4.14s | 48,334 | +0.27s |
| 6 | **predict_and_act** | 9.59s | 20,853 | **+5.45s** (confounded) |
| 7 | **learn_and_store** (= full) | 16.73s | 11,957 | **+7.14s** (confounded) |

**Dominant passes: `learn_and_store`, `predict_and_act`, then `encode`** — the
dense neural-net passes. The prime suspect `recall_topk` (the bitonic sort) is
~1.6% — **not** the bottleneck; the subgroup-path candidate is ruled out.

**Survival confound (limits 6–7).** `predict_and_act` (pass 5) is the first pass
that emits motor output, so limits 0–5 run a non-acting brain → identical
foraging/survival → the deltas through `recall_topk` are clean (`encode`≈2s and
`recall`≈0.5s are solid). Enabling passes 5–6 lets agents actually survive/forage,
raising how many agents are alive and doing work, so the `+5.45s`/`+7.14s` are
**upper bounds** inflated by survival, not pure pass compute. Re-run with
death/respawn churn suppressed to separate predict vs learn cleanly.

**(4) Fixed-seed evolution — population sweep (seed 42, tick_budget 120000,
16 generations, on target 2026-06-15):**

| Arm | world | genomes/gen | food/1k | deaths-per-food | best fitness (mean / peak) | wall/gen |
|---|---|---|---|---|---|---|
| N=10 | 256 | 5 | 0.185 | 2.05 | 0.0201 / 0.0250 | 5.7s |
| N=200 (unscaled) | 256 | 100 | 0.017 | 22.7 | 0.0175 / 0.0211 | 10.0s |
| N=200 (scaled, area ∝ N) | 1145 | 100 | 0.200 | 1.89 | 0.0159 / 0.0198* | 10.4s |

\* fitness deflated by a measurement artifact, not behavior — see verdict.

**Verdict — population default reverted to 10; scaling is premature.**

1. **Naively raising the population regresses evolution.** At N=200 in the
   unchanged world, `Food` pins at the world's supply cap (416 every generation),
   per-capita foraging collapses 10× (food/1k 0.185 → 0.017) and deaths-per-food
   blows up ~11× (2.05 → 22.7): the agents share one world and compete for finite
   food, destroying the foraging selection signal.
2. **Enlarging the world (area ∝ N, constant food density) removes the
   competition.** `Food` varies again (3.5k–6k), and the world-size-invariant
   per-capita metrics match N=10 — deaths-per-food **1.89** (≈ N=10's 2.05) and
   food/1k **0.200** (≈ 0.185). The lower composite fitness is a measurement
   artifact: the heatmap is a fixed `HEATMAP_RES²` grid spanning the world, so in
   the 4.5×-bigger world each cell is coarser and the same physical travel covers
   fewer cells (`cells` 400–700 → 120–500), deflating the exploration fitness
   term. Within a run all agents share that scale, so selection is unaffected.
3. **The population lever yields no benefit.** Champion fitness does not improve
   over 15 generations in *any* arm — it wanders 0.01–0.02 and ends where it
   began. Evaluating 100 genomes/gen instead of 5 buys nothing because the
   bottleneck is learner strength, not search breadth; and N=200 costs ~1.8×
   wall-time/gen (plus a 4.5× world when scaled).

The occupancy sweep/profile remain valuable infrastructure, but the default
`population_size` stays at **10**. Revisit population scaling only after the
learner improves — and then with a world-size-invariant exploration metric and
ideally independent per-genome arenas (so population becomes true parallel
evaluation rather than shared-world competition).

## 2026-06-15 — Plan 0004 approach-shaping pre-change baseline

Pinned "before" numbers for the Plan 0004 approach-reward-shaping unlock,
captured on `develop` @ `069ff8a` (Metal adapter, macOS), single-threaded:

| Probe | Result |
|---|---|
| `learning_probe_baseline_turn_alignment_is_chance` | 285/573 = **0.497** (chance band `[0.38, 0.62]`) |
| `learning_probe_free_run_foraging_baseline` | food=7, deaths=0, **food/agent/1k-ticks = 0.146** |
| `learning_probe_mirrored_steering_is_chance` | food=1033, alignment 385/735 = **0.524** (chance) |
| `encoder_food_side_separability_diagnostic` | within(right,right') cos=0.9998 (dist 0.0002); between(right,left) cos=0.9964 (dist **0.0036**) |

The chance-band edge the shaped remeasure must beat is **0.62**
(`learning_probe_baseline_turn_alignment_is_chance`).

The separability diagnostic is the load-bearing context: the between-side encoded
distance (0.0036) is ≈ 18× the within-side noise (0.0002), so the food side **is**
linearly represented in `s_encoded`. That places the bottleneck on the
credit/temporal path (the spatially-blind reward), not the encoder — the
condition under which approach shaping is expected to unlock steering.

## 2026-06-15 — Plan 0004 approach-shaping remeasure: prove-or-kill verdict

Approach shaping (`Φ(s) = −0.05·d_norm`, `F = γΦ(s′) − Φ(s)` folded into
`raw_gradient`) and the split actor learning rate (`ACTOR_VECTOR_SCALE = 1/16`,
separate from the critic's `1/128`) are landed and mechanism-tested
(`shaped_reward_rewards_approach`, `actor_step_scales_with_actor_vector_scale`).
The falsification test is the mirrored-steering regime (only vision-conditional
turning pays) trained with movement, then scored stationary.

**Result — negative. The unlock did not land.** Stationary turn/bearing
alignment held at chance across a generous shaped warm-up (default
`APPROACH_SHAPING_GAIN` / `ACTOR_VECTOR_SCALE`, well past the implied warm-up):

| Episodes trained (×100 ticks) | alignment |
|---|---|
| 100 | 417/878 = 0.475 |
| 300 | 229/477 = 0.480 |
| 600 | 155/322 = 0.481 |
| 900 | 50/138 = 0.362 |
| 1200 | 51/134 = 0.381 |

No upward trend toward the 0.62 edge at any budget; the full sweep stayed inside
`[0.36, 0.50]`. Free-run foraging was flat within single-seed noise
(`food/agent/1k-ticks` ≈ 0.10–0.13 vs the 0.146 baseline — not a rise).

**Refined next suspect (measurement over the review's guess).** The 2026-06-14
review named the encoder as the next suspect on a negative. The baseline
separability margin (between 0.0036 ≫ within 0.0002) contradicts that: the food
side is represented adequately. The actual bottleneck is the credit/temporal
path. The shaped reward rewards *closing distance*, which is dominated by forward
motion; the turn channel's contribution to approach is a weak second-order
effect, so the TD(λ) eligibility-trace credit cannot isolate "turn toward the
seen food" into the turn weights. Closing the gap needs a turn-credit signal
(e.g. a bearing-aligned reward term or an action-conditioned advantage), not an
encoder change.

**Gate decision.** Per the locked prove-or-kill rule, the gated workstreams
`0002` (klinotaxis/valence), `0004` (heritable policy constants), and `0005`
(lag/vision geometry) **do not open** — tuning or extending steering machinery is
pointless until a change moves this probe. The PBRS-invariant shaping and the
actor-scale split are kept (they cannot corrupt the eat objective and are the
substrate for the next credit-path attempt). Workstream `0003`
(selection-signal restoration) is unconditional and proceeds regardless.

`learning_probe_mirrored_steering_is_chance` remains the falsifiable pin: it
stays green at chance and will trip the day a credit-path change finally
produces directional steering.

## 2026-06-15 — Plan 0006 N=10 budget (multi-workgroup brain parallelism)

Task `n10-throughput-budget-baseline`. Target machine: Apple M3 Max, Metal 4.
Release binary, `--bench --bench-ticks 1000000 --bench-agents 10`, sweeping
`XAGENT_KERNEL_PASS_LIMIT`. The pass order (from the Plan 0005 N=200 table above)
is: `feature_extract, encode, habituate_homeo, recall_score, recall_topk,
predict_and_act, learn_and_store`.

| mode | tps | batches | submits | submit_ns/batch | wall |
|---|---|---|---|---|---|
| full (all 7 passes) | **23,230** | 10000 | 417 | 4,161,716 | 43.05s |
| pass-limit 0 (no brain) | **159,634** | 10000 | 417 | 605,375 | 6.26s |
| pass-limit 2 (≤ encode) | 109,913 | 10000 | 417 | 879,213 | 9.10s |
| pass-limit 5 (≤ recall_topk) | 90,440 | 10000 | 417 | 1,068,703 | 11.06s |
| pass-limit 7 (full) | 23,374 | 10000 | 417 | 4,135,482 | 42.78s |

(`read` = async readback, not measured under `--bench`; `gpu_complete_ns` is 0 by
design in the bench probe.)

**Brain-cost decomposition (1M ticks):** floor 6.26s, full 43.05s → brain = 36.79s.
- passes 1–2 (`feature_extract`+`encode`): 6.26→9.10s = **+2.84s** (encode owns ~all)
- passes 3–5 (`habituate_homeo`+`recall_score`+`recall_topk`): 9.10→11.06s = **+1.96s**
- passes 6–7 (`predict_and_act`+`learn_and_store`): 11.06→42.78s = **+31.72s (86% of brain)**

`encode` + `predict_and_act` + `learn_and_store` ≈ 34.56s = **94% of brain cost**,
confirming the plan's breakthrough targets (tiled encode, predictor train+predict,
encoder-credit learning, memory reinforcement). To hit 60k tps = 16.67s/1M, the
brain budget is 16.67−6.26 = 10.41s, i.e. the 36.79s brain must shrink **3.53×**.

**Decision (locked rule for this task):** pass-limit-0 floor = **159,634 tps ≥
90,000** → **CONTINUE**: the non-brain floor comfortably supports 60k.
`fused-food-grid-detect-floor-recovery` is **not** forced mandatory by this gate
(it remains a closure dependency and is implemented either way, but its ≥20%
floor-recovery bar does not apply since the floor is already 159,634 tps).

### Plan 0006 task results (running)

On-target N=10 / N=200, FusedSerial path unless noted. Baseline = pre-0002.

| after task | N=10 tps | Δ vs baseline | step gain |
|---|---:|---:|---:|
| baseline (pre-0002) | 23,230 | — | — |
| `same-dispatch-dense-tiling` (0002) | 34,001 | +46.4% | +46.4% |
| `parallel-reduce-action-tail` (0004d) | 41,200 | +77.4% | +21.3% |
| `multi-workgroup-memory-reinforcement` (0004e) | 44,680 | +92.3% | +8.4% |
| 7c reinforcement dot (memory reinf complete) | **45,506** | **+95.9%** | +1.8% |

**FINAL (closure): TARGET MISSED — 45,506 tps = 75.8% of 60k.** Fused N=200 =
4,633,798 agent-ticks/sec (+93% vs baseline). The split multi-workgroup
`ParallelTiled` path is a **measured negative** (N=10 42,780 = −6% vs fused;
N=200 3,321,576 = −28%) — dispatch overhead exceeds the parallelism gain, so
default stays `FusedSerial`. Owner of the 60k gap: the **10-workgroup occupancy
ceiling at N=10** (irreducible matrix-vector FLOPs + barriers at ~25% GPU
utilization; more workgroups needs the split, which the overhead blocks). Floor
(pass-limit-0) = 157,910 tps → food scan is a measured non-owner. Learning
unchanged (fused vs ParallelTiled Food/1k ≈ 0.27, best fitness ≈ 0.085–0.092,
within single-seed noise; learning probes hold the chance baseline). Full
analysis: `docs/plans/0006-Multi-Workgroup-Brain-Parallelism/0006-60K-CLOSURE.md`.

0002 gate (≥25% N=10 AND N=200 not worse than −5%): **PASS** decisively. The
same-dispatch 4-lane tiling of encode/predictor/encoder-credit (using all 256
lanes instead of 128, cutting each dense MAD loop ~4×) already moves N=10 from
23,230 to 34,001 tps; the split multi-workgroup path (0004) still has the larger
remaining gap to 60k (need 16.67s/1M; currently 29.41s).

**0003 split-serial overhead gate (M3 Max, N=10, 3 trials each, variance <0.2%):**

| mode | N=10 tps | wall/1M | submits |
|---|---:|---:|---:|
| fused-serial | 34,000 | 29.41s | 417 |
| split-serial (one kernel dispatch/cycle) | 28,761 | 34.78s | 1,667 |

Overhead = **15.4% slower** → **JUST over the locked 15% gate.** SplitSerial is
byte-identical (physics + brain_state + pattern_buffer) so the scaffold is
correct; the tax is the 10× extra kernel dispatches/submits at N=10's tiny work
size (dispatch/submit-bound, not compute-bound). Per the 0003 Done-when at >15%,
the plan **prioritizes the same-dispatch (in-workgroup) optimizations
[`parallel-reduce-action-tail` fused part, `fused-food-grid-detect-floor-recovery`]
before any further split dispatches.** The finer-grained ParallelTiled phase
split would incur strictly MORE dispatch overhead than this minimal per-cycle
split, so the split multi-workgroup path is on the back foot for the 60k target.

## Plan 0007 — Learning Control Grounding (landed `c40dcb7`, 2026-06-16)

Continuity append (not a replacement). Plan 0007 made the live runtime truthful
and behavior measurable before asking evolution to amplify behavior. What landed:

- **Runtime genome authority.** The interactive worker now applies per-agent
  heritable brain-state genes after upload *and* after generation inheritance
  (`patch_agent_configs` in `sim_runtime.rs`, parallel to the headless path), and
  `record_mutations` now records `movement_speed` provenance.
- **Behavior telemetry.** Recording format v2 (legacy 15-float blobs still load),
  new all-agent physics slots `P_NEAREST_FOOD_BEARING` / `P_IN_DANGER_BIOME` /
  turn-persistence, and a per-node `behavior_metric` table.
- **Control-rate curriculum.** `BrainConfig::learning_curriculum()`
  (`movement_speed=8.0`, `brain_tick_stride=2`, `vision_stride=5`) bounds
  full-forward sensory-lag travel to ~2.7 world units (vs ~66.7 at the old
  defaults); breeding speed clamp lowered `[20.0, 100.0] → [4.0, 30.0]`.
- **Klinotaxis repair.** Multiplier-only turn scaling replaced with a
  worsening-gradient + turn-persistence-gated sign-breaking escape path; the
  noise-based eligibility-trace invariant is preserved.
- **Emergence probes.** Food-closure, danger-exit, and anti-circle GPU probes are
  red-green control/telemetry gates under the curriculum.

**Post-merge correction (caught by the repo-root full-workspace gate the
per-task gates skip):** `P_IN_DANGER_BIOME` was published **inverted** (`0.0`
while *in* a danger biome), which let `danger_exit_probe` pass vacuously (agent
appeared never in danger). Fixed to `1.0` = in danger; the danger telemetry tests
now assert the flag against the biome at the agent's actual readback position
(CPU `biome_at` and GPU `sample_biome` index the same 256×256 grid identically),
so they are deterministic and red-green for the inversion. With the corrected
flag, `danger_exit_probe` measures a real danger-dwell fraction of **0.7**
(previously a vacuous ~0.3). A separate first full-workspace run had also reported
spurious failures from a stale `xagent-brain` rlib (its `include_str!`'d shaders
were baked from an intermediate merge state); a clean rebuild compiled the
squashed source correctly — no source defect there.

**Gate (repo root, post-fix):** `cargo fmt`/`clippy` clean; `cargo test
--workspace --no-fail-fast` = 232 passed / 0 failed (51 brain + 87 sandbox-lib +
13 bin + 75 integration + 6 shared); GPU tests ran against a real adapter.

The 20-generation fixed-seed evolution comparison against this DB's late baseline
remains an **offline** run, deferred via
`docs/plans/0007-Learning-Control-Grounding/DECISION-short-evolution-gate.md`
(must use a build at/after the danger-flag fix, since pre-fix danger-dwell is
inverted and not comparable to the baseline here).
