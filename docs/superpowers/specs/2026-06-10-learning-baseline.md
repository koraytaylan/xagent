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
