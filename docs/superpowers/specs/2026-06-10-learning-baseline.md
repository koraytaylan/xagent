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
`Food | Deaths | Food/life | w_fwd | w_turn` — the behavioral signal plus
the best agent's policy weight norms. For evolution-level A/B runs, use a
fixed `--db` seed config and compare these lines across the same generation
counts.

## Merge gates (from the plan)

- Phase 1 (TD credit): alignment probe above the 0.62 band edge with the
  band re-pinned, foraging rate above baseline over a fixed-seed
  20-generation headless run, TPS within 10% of pre-change.
- Phase 2 (encoder self-supervision): encoded-state separability test
  passes; alignment improves over Phase 1; TPS cost < 15%.
- Phase 3 (vision acuity): food visible at range in the probe; food/life
  trends upward across generations.

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
| Trained turn/bearing alignment | `learning_probe_td_learns_turn_alignment` | 0.498 (chance, untrained) | **0.643** after 120 episodes of food-reaching practice |
| Episode food (first half → second half of training) | same test | — | **588 → 730** of 960/half (rising) |
| Critic value under constant drain | `td_critic_tracks_metabolic_drain` | n/a | **−0.002** (correctly negative, finite, δ within clamp) |
| Trace bound across deaths | `td_traces_bounded_across_deaths` | n/a | bounded after 16 deaths (no cross-life leak) |
| Untrained stationary alignment | `learning_probe_baseline_turn_alignment_is_chance` | 0.498 | 0.50 (still chance — nothing to learn from with no reward events) |
| Free-run foraging (debug adapter, single run) | `learning_probe_free_run_foraging_baseline` | 2 food | 7 food (noisy single-seed; not a gate) |

The directional gate is the load-bearing result: with food visible and the
TD reward arriving on contact, **alignment rises from chance (0.498) to
0.643** — clearing the baseline chance band's upper edge (0.62). This is the
first time the policy's turn direction has carried information about food
bearing. The untrained stationary probe stays at chance, confirming the
gain comes from reward-driven learning, not a directional bias artifact.

TPS unchanged within noise (the serial history-ring credit loop is gone;
the TD path is fully parallel across the 128 trace dimensions).

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
"Phase 2 outcome" section for the mechanism analysis). The headline: TD(λ)
already extracts the food-direction signal from the random-projection
encoder at 8×6, so encoder representation is not the binding constraint and
reshaping it only adds a moving-target cost. Work proceeds to Phase 3
(vision acuity).
