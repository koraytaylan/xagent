# Scope — Plan 0012

> Restore within-life learning to pure homeostatic deltas by removing both
> potential-based reward shaping terms (Plan 0004's approach PBRS and Plan 0009's
> avoidance PBRS) from the `raw_gradient` computation and their supporting
> machinery, aligning code with the stated README vision of homeostasis-only
> evaluation.

## Why this plan

The project README (§1, §3, §10) and LOCKED decisions (see Plan 0009's status
and the 2026-06-18 review consensus) commit to homeostasis-only evaluation: "the
only evaluative signal in the entire system is homeostatic stability" — no reward
functions, no shaping. Yet the code contradicts this: `brain_passes.wgsl`
(lines 791–805 / 807–822) actively computes and injects two potential-based
reward shaping (PBRS) terms into the learning signal.

1. **Approach-PBRS term is active in production.** The `shaping` variable
   (`brain_passes.wgsl:804`) reads `APPROACH_SHAPING_GAIN = 0.05` (`common.wgsl:500`)
   and computes `Φ(s) = −gain·d_norm` (`brain_passes.wgsl:802`) using the
   nearest-food distance, yielding `F = γΦ(s′) − Φ(s)` that is folded into
   `raw_gradient` (`brain_passes.wgsl:823-826`). The machinery
   (`P_PREV_POTENTIAL` storage, food-distance reads) is wired into production
   (`kernel_tick.wgsl`, `phase_physics.wgsl`) despite Plan 0004's own remeasure
   (Plan 0004 STATUS.md) falsifying the unlock: "mirrored-steering alignment held
   at chance across 1200 shaped episodes."
2. **Avoidance-PBRS term is active behind a flag (default false).** The
   `danger_shaping` variable (`brain_passes.wgsl:820`) is gated on
   `CFG_DANGER_PERCEPT_ENABLED` and computes `Φ_d(s) = −(1 − danger_d_norm)`,
   yielding the same PBRS increment into `raw_gradient`
   (`brain_passes.wgsl:823-826`). The Plan 0009 decision doc (Intent-Aware
   Fitness) intended this as a "sensory hint," but it is mechanically
   indistinguishable from the approach shaping — both are hand-engineered reward
   kernels, contradicting the stated vision.
3. **The README is aspirational but the code is not.** README §1 ("There are no
   reward signals, no utility functions, no goal hierarchies. The only evaluative
   signal in the entire system is homeostatic stability") is written as fact, not
   as a future goal (`README.md:10`). §10 (Why Homeostasis-Only Evaluation?)
   rationalizes it philosophically (`README.md:515`). But
   `brain_passes.wgsl:823-826` computes `raw_gradient = energy_delta*ENERGY_WEIGHT
   + integrity_delta*INTEGRITY_WEIGHT + shaping + danger_shaping`, where `shaping`
   and `danger_shaping` are both designed reward kernels. This is the
   highest-order philosophical contradiction the 2026-06-18 review identified:
   Plans 0010 & 0011 resolved 95% of those findings with on-touch debt cleanup,
   but they did not address the core mismatch. This plan is the step back to
   restore vision-code alignment.

**Provenance.** Verified against `develop`. The always-on approach shaping in the
default brain was independently confirmed by the `2026-06-19-claude-opus-48.md`
review (finding **M1**: *"the default build already violates 'no reward functions'…
an always-on, unflagged potential-based food-approach reward is folded
unconditionally into the brain's TD reward"*), which ranks it the deepest purity
breach on the floor and notes Plan 0004's own remeasure already falsified that
shaping as ineffective (none of the three peer 06-19 reviews caught it; two even
state — incorrectly — that the brain "sees only homeostatic deltas"). The finding
stands in code: both PBRS terms are present in the live `raw_gradient` assembly
(`brain_passes.wgsl:823-826`), the approach term unconditionally active, the
avoidance term gated only by a default-false flag, not removed.

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| The avoidance-shaping flag default (false) makes it inert and not worth removing | 2026-06-18 review (mitigating note) | The flag short-circuits at runtime but the engineered reward kernel remains in source; it is mechanically identical to the approach term and contradicts the stated vision whether or not it currently executes. Removal is for code-vision coherence, not just runtime behavior. |
| The README must be rewritten to describe the prior shaping mechanism | 2026-06-18 review (doc note) | The README wording already states homeostasis-only correctly; it was aspirational and becomes true by construction once the terms are removed. No rewrite is needed — only verification that no doc references the shaping as active. |

## In scope

- **0001 — Measurement-Baseline.** Wire a CPU-readable `raw_gradient` handle
  (`P_RAW_GRADIENT_OUT` physics slot, surfaced as `AgentTelemetry::raw_gradient`)
  and capture the pre-removal `raw_gradient` distribution at default config. Authored
  first and depended on by both removal workstreams so the baseline is measured
  against the BEFORE state. See [TASKS.md](TASKS.md).
- **0002 — Approach-Shaping-Removal.** Remove the approach-PBRS term
  (potential-based reward shaping for food seeking) from the `raw_gradient`
  computation in `coop_habituate_homeo` (stop writing the `P_PREV_POTENTIAL`
  slot; the slot itself is retained as reserved), and rewrite the test that
  verified the shaping (`shaped_reward_rewards_approach`) to assert the
  post-removal homeostatic-only behavior. See [TASKS.md](TASKS.md).
- **0003 — Avoidance-Shaping-Removal.** Remove the avoidance-PBRS term
  (danger-percept potential shaping) from the `raw_gradient` computation (stop
  writing the `P_PREV_DANGER_POTENTIAL` slot; the slot is retained as reserved),
  and rewrite the test that verified it (`avoidance_potential_sign`). See
  [TASKS.md](TASKS.md).
- **0004 — Dead-Code-Cleanup.** Delete the genuinely-orphaned
  `APPROACH_SHAPING_GAIN`; rename the still-live `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS`
  (it bounds the food-detect scan, independent of shaping); and mark the
  now-unwritten `P_PREV_POTENTIAL`/`P_PREV_DANGER_POTENTIAL` slots reserved
  (consts kept — layout-parity and integration tests reference them; no
  `PHYS_STRIDE` shift). See [TASKS.md](TASKS.md).
- **0005 — Parity-And-Tests.** Ensure fused (`kernel_tick.wgsl`) and split
  (`phase_physics.wgsl`) brain-gradient paths compute identically after shaping
  removal; add a falsifiable test asserting `raw_gradient` contains only the two
  homeostatic terms at default config. See [TASKS.md](TASKS.md).
- **0006 — Documentation-Update.** Update README sections that claimed "no reward
  function" and "homeostasis-only" to reflect the restored alignment between
  design and code; update crate READMEs and contributing guardrails if needed.
  See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Approach-PBRS term active in `raw_gradient` assembly (1) | `0002` |
| Avoidance-PBRS term active in `raw_gradient` assembly behind flag (2) | `0003` |
| README contradicts code: states homeostasis-only but code has reward shaping (3) | `0006` |
| Pre-removal `raw_gradient` baseline + CPU-readable handle, measured before any removal (derived, measurement-first) | `0001` |
| Shaping-const cleanup: delete orphaned `APPROACH_SHAPING_GAIN`, rename live `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS`, reserve `P_PREV_*` slots (derived) | `0004` |
| Fused/split gradient-path parity and falsifiable homeostasis-only gate (derived) | `0005` |

## Locked decisions

- **Removal is unconditional, not gated.** Once the shaping terms are removed and
  tests pass, there is no conditional revert. The design commits to
  homeostasis-only learning. If a future experiment determines that shaping is
  necessary (unlikely given Plan 0004's own falsification), that would be a new
  plan with new evidence, not a gate to this plan.
- **Danger-percept sensory features remain available for natural discovery.**
  Removing the danger-shaping reward term does NOT remove the danger-percept
  sensory channel (distance and bearing features written to the feature vector in
  `coop_feature_extract`). Agents can still discover the danger signal through
  prediction error and homeostatic correlation if the encoder/predictor find it
  useful. The shaping term merely short-circuited that discovery process;
  removing it restores learning-from-experience.
- **Buffer layout does not shift; the prev-potential slots become reserved.** The
  `P_PREV_POTENTIAL` (slot 33) and `P_PREV_DANGER_POTENTIAL` (slot 41) consts are
  **retained**, not deleted — they are referenced by the `buffers.rs` PHYS_STRIDE
  layout-parity test and by `integration.rs`, so deleting them would break the
  build/tests and force a stride shift. After the shaping removal nothing *writes*
  them; their doc-comments are updated to mark them reserved. The respawn zeroing
  of those slots (`phase_death.wgsl`, `kernel_tick.wgsl`) is left in place — it is
  harmless and keeps the flag-off integration assertion (`== 0.0`) valid. No
  layout shift. Separately, `SHAPING_RADIUS` is **not** removed: it bounds the
  food-detect scan in `kernel_tick.wgsl`, so it is renamed to `FOOD_SENSE_RADIUS`;
  only `APPROACH_SHAPING_GAIN` is genuinely orphaned and deleted.

## Out of scope

- **Evolution, fitness metrics, or agent behavior changes.** Plan 0012 removes an
  engineered reward component from the learning signal. This is a design
  simplification to code-vision alignment, not a fitness optimization. If the
  removal causes measurable behavior regression, that is a separate concern (a
  future Plan 0013+ would measure and respond). This plan is about correctness
  and philosophical coherence, not empirical outcome.
- **Danger-percept sensory removal or architecture changes.** The danger-percept
  feature channel (distance and bearing) remains active and available to the
  encoder. Only the hand-engineered shaping reward term is removed. Agents can
  still learn to avoid danger through prediction error; the path is just longer
  and natural rather than short-circuited by a reward hint.
- **Encoder or predictor retrain or re-initialization.** The change is a
  signal-level edit (removing terms from the gradient computation), not a
  buffer-layout change. Existing trained weights remain valid; the gradient
  signals they receive are simply different (purer homeostatic signal). A fresh
  run may show different learning trajectories, but there is no need to reset or
  retrain weights explicitly.
- **Plan 0009's danger-percept senses configuration or telemetry.** The
  danger-percept sensory features (distance, bearing) are still written and
  available. Only the potential-based reward shaping term is removed. The sensory
  configuration and telemetry remain unchanged; the signal pathway is restored to
  pure learning-from-experience rather than reward-injection.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
