# Scope — Plan 0016

> Operationalize intent fractions as an evolutionary selection criterion across
> generations via the seeded-paired A/B harness, measuring danger-percept
> evolved-policy intent delta with 95% CI and baseline noise floor, then document
> intent-as-meta-score as an explicit framework for future ablations.

## Why this plan

Plan 0015 shipped a complete intent measurement infrastructure — approach and
avoidance counters, per-agent telemetry, population fractions, and a baseline
distribution under homeostasis-only learning (`crates/xagent-brain/tests/intent_baseline_measurement.rs`,
baseline approach ≈ 0.498, avoidance ≈ 0.421). However, those are *within-life*
baselines (random init + TD only, no evolution across generations). The baseline
numbers raise a critical question the framework explicitly defers: are those
fractions *deliberate* (the brain's learned steering causally depends on
perceiving the target) or *incidental* (agents turn by chance geometry-gated
coincidence)?

Two load-bearing facts motivate this plan:

1. **Within-life intent alone cannot adjudicate causality.** A within-life run
   measures whether agents turn when the target is in range, but does not isolate
   whether the turn is *caused by* sensing the target or is incidental motion that
   the counter happens to score. Proof: in `crates/xagent-brain/tests/danger_percept_ablation_ab.rs`
   (lines 1–72, the lived seeded A/B harness), the within-life avoidance fraction
   is ≈0.42 under both sighted (danger bearing visible) and blinded (danger bearing
   masked) runs — the delta is +0.0036, real but negligible (CI excludes 0, noise
   floor ≈ 0). The brain *is* wired to see the bearing (proof: delta ≠ 0) but
   contributes only +0.36 pp of extra avoidance, ~14× below practical significance
   (±0.05). That measurement was within-life only; an evolved-policy ablation is
   needed to adjudicate whether generational selection (fitness and evolution)
   amplifies the causal signal.

2. **Evolved populations may show stronger intent-perception coupling than
   within-life learners.** The within-life harness uses random network
   initialization + 100-tick TD learning (the default short regime). Over
   generations, selection might build populations where danger-avoidance becomes
   more clearly causal to steering, or it might not — the outcome is empirical, not
   assumed. This plan runs the danger-percept blinded/sighted A/B through the full
   generational loop to measure the evolved delta, establishing the ground truth
   for danger-avoidance intent under the project's selection regime.

3. **Intent-as-meta-score is a new framework for future ablations, and
   danger-percept is the pilot.** Plan 0015 measured the baseline; this plan
   operationalizes intent fractions as an **evolutionary selection criterion** —
   paired seeded runs, population aggregation, CI gating, A/A noise floor, explicit
   verdict — so a future plan can pick another evolvable parameter (e.g.,
   `food_percept_enabled`, a hypothetical approach-intent analogue) and apply the
   exact pattern. Documenting the framework explicitly (validation harness idiom,
   delta interpretation, threshold vs noise floor, per-arm lifespan accounting)
   ensures the pattern is reproducible.

Findings:

1. **Within-life danger-percept ablation shows percept wiring but negligible
   steering contribution.** The seeded within-life A/B in
   `crates/xagent-brain/tests/danger_percept_ablation_ab.rs:1-72` measures mean avoidance delta = +0.0036
   (sighted 0.427, blinded 0.424) with CI [+0.0008, +0.0065], above the A/A noise
   floor (0.0, deterministic kernel) but far below practical significance (±0.05).
   This rules out broken wiring but leaves open whether selection amplifies the
   signal.
2. **Generational selection regime is untested for intent-causality.** The baseline
   distribution (`crates/xagent-brain/tests/intent_baseline_measurement.rs`) is within-life only (random init
   + TD, no evolution). Whether evolved populations show stronger danger-avoidance
   intent delta under generational selection is unknown.
3. **Paired seeded A/B harness for evolved policies exists and is proven.** The
   `run_innate_instinct_ab` pattern in `headless.rs:492-556` uses
   `run_headless_with_flags` to run paired seeded generations, collecting
   `ValidationStats` with per-arm mean intent fraction
   (`mean_avoidance_intent_fraction`), lifespan, death counts, and other
   aggregates. This is the harness template this plan replicates for danger-percept
   ablation.
4. **Intent-as-meta-score framework is not yet explicit.** Threshold setting, delta
   interpretation, noise-floor gating, and lifespan-denominator accounting are
   implicit in the within-life test (`crates/xagent-brain/tests/danger_percept_ablation_ab.rs:27-31`); they
   must be documented as an explicit framework so future ablations can follow the
   pattern (`docs/plans/0015-Intent-Awareness-Measurement-Framework/SCOPE.md:99-104`).

## In scope

- **0001 — Danger-Percept Evolved-Policy A/B Harness.** Extend
  `run_headless_with_flags` to accept a `danger_percept_blinded` flag; run paired
  seeded generational evolution (danger_percept_blinded = false vs true, identical
  initial genomes, same seed, same world) across G generations; collect
  ValidationStats per arm with population approach and avoidance intent fractions;
  compute paired mean delta, 95% CI, per-arm death and lifespan statistics; measure
  A/A noise floor (two sighted runs, same seed, deterministic kernel); apply
  decision rules to produce a verdict (DELIBERATE if delta ≥ +0.05 and CI clears
  noise floor; NEGLIGIBLE-BUT-REAL if |delta| < 0.05 but CI excludes noise; INERT
  if within noise; INDETERMINATE if inconclusive); report all per-generation
  numbers and the aggregate summary to establish the ground truth for
  evolved-policy danger-avoidance intent. See [TASKS.md](TASKS.md).
- **0002 — Intent-As-Meta-Score Framework Documentation.** Document the
  intent-as-meta-score validation framework explicitly: the paired-seeded A/B
  pattern (why seed identity isolates causality, how both arms share world and
  genomes, how blinding works), the CI computation and noise-floor gating (how to
  interpret a delta, when to claim DELIBERATE vs NEGLIGIBLE, how A/A proves
  determinism), the per-arm lifespan and death-count accounting (why
  generation-cumulative denominators must be normalized), and how a future plan
  replicates the pattern on another evolvable parameter. Clarify why
  homeostasis-only learning is untouched (intent measurement is selection-only,
  never a learning term) and why within-life and evolved-policy measurements
  address different questions. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Within-life danger-percept ablation shows percept wiring but negligible steering contribution (1) | `0001` |
| Generational selection regime is untested for intent-causality (2) | `0001` |
| Paired seeded A/B harness for evolved policies exists and is proven (3) | `0001` |
| Intent-as-meta-score framework is not yet explicit (4) | `0002` |

## Locked decisions

- **Intent measurement is selection-only, never a learning term.** Intent
  fractions (approach/avoidance) are computed post-hoc from telemetry and used only
  for evolutionary selection and measurement. They never flow back to the kernel's
  TD learning signal, reward computation, or homeostatic credit path. This plan
  respects the homeostasis-only constraint (Plan 0012) absolutely. Consequence:
  evolving higher intent does not mean agents learn a new credit path; it means
  selection picks genomes that already exhibit higher intent under the existing
  learning regime. If a future plan wants to make intent a learning signal (e.g., a
  bonus reward term), that is a new plan with its own justification, not a change to
  this one.
- **Paired seeded A/B isolation: both arms share seed, world, and initial
  genomes.** To isolate the causal contribution of a percept (or flag) to steering,
  both the sighted (flag=off) and blinded (flag=on) arms must start with identical
  initial genomes and world, seeded by the same world seed. The kernel is
  bit-deterministic (A/A noise floor ≈ 0), so the paired delta is the causal effect
  of the mask. Consequence: both arms must be run in the exact same configuration
  except for the single flag being ablated. Any difference in world seed, population
  initialization, or other flags breaks the pairing.
- **Verdict thresholds are calibrated from danger-percept pilot; subject to
  revision.** The decision rules (DELIBERATE iff Δ ≥ +0.05 and CI > noise;
  NEGLIGIBLE-BUT-REAL iff |Δ| < 0.05 and CI excludes noise, etc.) are derived from
  the within-life danger-percept ablation results and the evolved-policy pilot
  measurement. These thresholds represent practical significance in the context of
  intent fractions (which scale [0, 1]). If future measurements on other evolvable
  parameters (e.g., food-percept) show systematically different effect sizes, the
  thresholds may be revised. The framework is the pattern, not the exact numbers.
- **Per-arm lifespan and death-count normalization for generation-cumulative
  denominators.** The population intent fraction is `sum(turns) / sum(sense_range_ticks)`,
  aggregated across all agents and all generations in a run. As generations
  accumulate, both the numerator and denominator grow. A difference in per-arm mean
  lifespan (e.g., sighted agents live longer) directly affects the denominator and
  can confound the delta interpretation. Reporting mean lifespan and mean death
  count per arm allows reviewers to account for this and verify that the intent
  delta is not an artifact of lifespan difference. Consequence: any significant
  delta in lifespan between arms must be noted in the verdict.

## Out of scope

- **Changing the learning signal or reward computation based on intent.** Intent
  measurement is selection-only. If a future measurement shows intent is strong and
  causal, a *new* plan with its own justification may propose an intent-based
  learning term; this plan does not. Homeostasis-only learning (Plan 0012) is
  locked.
- **Modifying the danger-percept or food-percept physics or detection mechanics.**
  The percepts are already built (danger-percept by Plan 0009, food-percept by Plan
  0001). This plan measures via blinding mask only, not by changing how detection
  runs. The physics and sensing remain unchanged.
- **Within-life danger-percept re-measurement or refinement.** The within-life
  harness exists (`crates/xagent-brain/tests/danger_percept_ablation_ab.rs`) and is mature. This plan measures
  evolved policy only, not within-life. If the within-life numbers need refinement,
  that is a separate task.
- **Threshold-based gating of evolution on intent.** This plan measures and
  documents the framework. Gating evolution on intent thresholds (e.g., "only select
  agents with avoidance > 0.5") is a future plan decision, after the framework and
  baseline are established.
- **Reworking the 95% CI computation or statistical methodology.** The normal
  approximation (CI = 1.96 * sem) is standard and is used in the within-life test.
  If a future plan needs a different statistical approach (e.g., bootstrap, Bayesian
  credible intervals), that is a justified change with evidence; this plan uses the
  proven method.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
