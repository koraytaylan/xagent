# Plan 0016 — Intent-As-Meta-Score Evolved-Policy Ablation — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned.
_Last updated: 2026-06-23, against `feat/danger-percept-ablation-measurement`._

- **Goal:** Danger-percept evolved-policy A/B harness built and measured; paired
  delta reported with 95% CI, A/A noise floor, and verdict
  (DELIBERATE/NEGLIGIBLE-BUT-REAL/INERT/INDETERMINATE); intent-as-meta-score
  framework documented explicitly so future evolvable-parameter ablations can
  follow the identical pattern; per-generation approach-intent trajectory
  collected and reported for visualization; per-arm lifespan and death-count
  statistics captured and analyzed for denominator normalization. All
  measurement-only (no threshold assertion, no learning-signal change);
  homeostasis-only constraint intact.
- **Root cause:** Plan 0015 shipped intent measurement infrastructure (counters,
  population fractions, baseline distribution) but left open the causality
  question: are within-life avoidance fractions DELIBERATE (brain-caused) or
  INCIDENTAL (geometry-coincidence)? The within-life ablation shows sighted vs
  blinded delta ≈ +0.0036 (negligible). Whether generational selection amplifies
  the causal signal is unknown. The intent-as-meta-score framework is also
  implicit (not documented as a pattern for future ablations).
- **Approach:** Extend `run_headless_with_flags` with a `danger_percept_blinded`
  flag and run paired seeded evolved-policy A/B (identical seed, genomes, world;
  only the blinding mask differs) across G generations. Collect ValidationStats
  per arm with per-generation intent trajectories and aggregate intent fractions.
  Compute paired delta + 95% CI, measure A/A noise floor (two sighted runs, same
  seed), report per-arm lifespan and death counts (for denominator
  normalization), apply verdict rules, and record measured numbers (run-then-paste
  idiom, mirroring the within-life harness). Document the framework explicitly in
  READMEs so future evolvable-parameter ablations can replicate the pattern.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Danger-Percept Evolved-Policy A/B Harness | `wire-danger-percept-blinded-flag`, `extend-validation-stats-approach-trajectory`, `implement-danger-percept-evolved-ab-harness` | 📋 Planned |
| 0002 | Intent-As-Meta-Score Framework Documentation | `document-intent-as-meta-score-framework` | 📋 Planned |
