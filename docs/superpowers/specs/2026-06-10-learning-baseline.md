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

> **Caveat (Phase 3).** This section's "TD(λ) already extracts the
> food-direction signal" conclusion rested on the **confounded** 0.643
> directional number. The confound-free probe shows directional steering at
> chance, so the encoder/representation question is *not* settled — it is
> re-opened under correct measurement. Reconstruction-as-implemented was
> still neutral-to-negative, so reverting it remains correct; but "encoder is
> not the binding constraint" no longer follows.

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
likely cause is the one issue #13 and both reviews named: a random-projection
encoder does not make "food-left" and "food-right" linearly separable for the
policy's readout, so a constant bias is learnable but a conditional response
is not. This re-opens the encoder/representation problem under a correct
measurement (Phase 2 had dismissed it using the confounded metric).

**Net:** Phase 3 delivers the information substrate (food visible at range)
and, more valuably, a confound-free directional probe that correctly reports
the open problem. The TD(λ) critic and the evolution-scale foraging/fitness
/survival gains from Phase 1 stand; the specific claim of learned directional
steering does not.
