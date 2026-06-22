# Decision — Plan 0014 / Workstream 0001: Effort-Fitness Camper Inversion

## Context

The four 2026-06-19 due-diligence reviews flagged that effort-rebased fitness
(`effort_rebased_fitness`, default-off) ranked an idle **camper** as the fittest
archetype: the synthetic 100k-tick calibration replay pinned
`Camper effort 0.8646 > Competent 0.3209`, and the test asserted only
`competent > aimless` / `camper < 1.0` — never `competent > camper`.

## What landed

1. **`fix-foraging-duration-leak`** made both effort axes duration-independent
   cumulative ratios — foraging = `food / energy`, exploration =
   `min(coverage, cells / distance / TARGET)` — removing the `ticks_alive`
   factor that previously inflated long-lived agents. This raised the competent
   forager from **0.3209 → 0.7034**.

2. **Finding (this plan):** removing `ticks` alone did **not** invert the
   ranking. The original camper fixture (`food=250`, `energy=60`) has
   `food/energy = 0.51`, a *strictly higher* foraging ratio than the competent
   forager (`food=180 / energy=180 = 0.28`). Both therefore cap the foraging axis
   at 1.0, and the camper wins on having zero deaths (survival 1.0 vs 0.75). No
   choice of target separates them — that fixture actually modeled a
   hyper-efficient respawn *exploiter*, not the idle agent the review meant.

## Decision

**Redefine the calibration camper as a true idle agent** (near-zero food,
~0 movement), matching the archetype the review and `add-anti-camper-assertion`
describe ("high ticks_alive, near-zero food_consumed"). The live composite
formula and weights are unchanged; only the test fixture was corrected to model
the archetype it claims to. The high-food respawn-exploiter is a separate,
legitimately-effective archetype and is not treated as a pathology.

New camper fixture: `food=5, energy=100, distance=2, cells=40, deaths=0` →
foraging ≈ 0.17, composite **0.1504**, well below the competent forager's 0.7034.

## Guards added (`add-anti-camper-assertion`)

- `competent > camper` — the missing inversion guard.
- Foraging negative control: the idle camper scores `< 0.5` on the foraging axis.
- `effort_fitness_is_duration_invariant`: identical efficiency ratios with
  different lifetimes score equal in effort mode, while legacy mode (the control)
  differs — proving the `ticks` leak is gone and the test would catch its return.

## Why not a structural formula change

Making exploration multiplicatively required (so any non-exploring agent scores
low regardless of food) was considered. It would penalize the high-food exploiter
too and aligns with the project's exploration ethos, but it changes selection
semantics under the (default-off) effort flag and would need re-calibrating every
archetype. Deferred as a possible future iteration; not needed to close the
review finding, which was specifically about the *idle* camper.
