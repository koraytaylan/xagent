# Scope — Plan 0017

> Fix the credit-path (TD actor-critic) bottleneck so vision→action steering
> becomes learnable, then A/B the resulting capability against an emergent
> self-organizing encoder versus the hand-coded Hubel-Wiesel cortex.

## Why this plan

Current state: Plan 0008 (Hubel-Wiesel cortex) is stranded behind a flag with
mechanical correctness proven (6.97× vertical-tuning selectivity, phase
invariance ~1.6e-7) but throughput at 0.24% of budget (~420× slower). Plan 0013
(seeded instincts) was rejected with a diagnosis: the credit path is the limiter,
not the encoder or prior seeding. The `2026-06-19` due-diligence reviews (Claude
Opus, GPT-5-codex, Grok 4.3, Gemini 3.1 Pro) independently confirmed this,
ranking it M1 priority. This plan's four workstreams (0001→0002→0003→0004) execute
the project's founding principle — emergence from constraints, not imported design —
by first unlocking steering learnability here, then comparing emergent vs imported
encoders on fair footing.

1. **Vision→action steering alignment is at chance despite encoder separability
   55×.** The `learning_probe_mirrored_steering_is_chance()` test
   (`integration.rs:2716-2792`) trains agents for 120 episodes on alternating
   left/right food, then pins movement (`movement_speed=0`) and scores
   turn-alignment: 120 training episodes on dense TD (`brain_tick_stride=1,
   vision_stride=1`) yields alignment of 0.38–0.62 (chance band), while
   `encoder_food_side_separability_diagnostic()` (`integration.rs:2810-2887`)
   shows within-class cosine ~0.998 (diff 0.002) and between-class cosine ~0.036
   (diff 0.964) — a 55× margin. The encoder learns to separate food-left from
   food-right; the policy cannot learn to turn toward it.

2. **The credit-path structure (TD actor-critic in `brain_passes.wgsl`) is the
   suspect, not the encoder or prior seeding.** `raw_gradient`
   (`brain_passes.wgsl:811-814`) is purely homeostatic (`energy_delta *
   ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT + shaping + danger_shaping`;
   shaping and danger terms are zero post-0012). It is amplified by urgency
   (`raw_gradient_amplified = raw_gradient * (1.0 + urgency)`) and fed into TD as
   the reward signal (`s_homeo[1u] = raw_gradient_amplified`, then `reward =
   s_homeo[1u]` at `kernel_tick.wgsl:1118`). The TD error `δ = reward + γ·V(s′) −
   V(s)` (clamped `[-MAX_TD_ERROR, MAX_TD_ERROR]` = `[-1, 1]`) updates the critic
   at `CRITIC_LEARNING_RATE=0.01` × `TD_VECTOR_SCALE=1/128` and both policy
   channels at `ACTION_WEIGHT_LEARNING_RATE=0.1` × `ACTOR_VECTOR_SCALE=1/16`. The
   decay schedules (`TD_DISCOUNT=0.97`, `TD_LAMBDA` implied ≈0.90, per-step trace
   decay not present in the current audit) and trace clipping (none observed) are
   candidates for the stall: high-variance δ, insufficient trace longevity, or
   premature decay could flatten the vision→action credit gradient.

3. **Cortex throughput (0.24% of budget) is a separate gate from credit
   learnability, and the hand-coded Gabor bank conflicts with the emergence
   principle.** The fused-baseline cortex runs at ~81 tps vs ~34,000 tps on N=10
   (`coop_visual_cortex()`, implemented in 0008), so it cannot ride the default
   path; optimizing it independently allows a fair comparison arm. The Gabor bank
   proves the mechanical pipeline (orientation tuning, phase invariance) but seeds
   filter structure by hand — the founding principle demands structure that
   self-organizes from input statistics. The plan order is dependency-locked by
   measurement gates, not file conflicts: workstream 0001 fixes steering learnability
   (proof: alignment moves above 0.62 in the new probe); workstream 0002 optimizes
   throughput independently (disjoint WGSL — 0001 touches credit-path constants/decay
   in `common.wgsl` and trace logic in `brain_passes.wgsl:1082-1200`, 0002 touches
   convolution kernels/pooling/radii); workstream 0003 authors the emergent encoder
   as a new `coop_visual_encoding` pass; workstream 0004 A/Bs both encoders once
   workstreams 0001+0002+0003 are green. The workstreams honor the README's
   homeostasis-only contract (no shaped rewards in the credit path) and the
   CONTRIBUTING rule (no planning references in source).

**Provenance.** Every finding re-verified against current source: the steering
probe at `integration.rs:2716` (current main branch), encoder separability at
`integration.rs:2810`, `raw_gradient` at `brain_passes.wgsl:811-814` (no
approach/avoidance shaping post-0012), the TD reward assignment at
`kernel_tick.wgsl:1118`, and the credit scales at `common.wgsl:517-564`.

## In scope

- **0001 — Credit-Path Diagnosis and Learning Unlock.** Diagnose why vision→action
  steering is at chance despite encoder separability; implement credit-path fixes
  (TD decay schedule, trace longevity, urgency scaling, or gradient structuring) so
  steering alignment rises measurably above chance (≥0.70 target); replace the
  mirrored-steering probe's chance baseline with a new one reflecting the fix. See
  [TASKS.md](TASKS.md).
- **0002 — Cortex Throughput Optimization.** Optimize visual-cortex throughput to
  bring it within the ≥50% budget (currently 0.24%) without breaking existing
  orientation/invariance probes; keep it flag-gated; do NOT flip the default until
  the workstream 0004 A/B verdict. Options: separable DoG kernels, shared-memory
  precomputation, smaller retina, reduced feature count, or pooling-radius tuning.
  See [TASKS.md](TASKS.md).
- **0003 — Emergent Self-Organizing Encoder.** Replace the hand-built Gabor bank
  with a learning objective (sparse coding or predictive coding) that
  self-organizes receptive-field structure from input statistics; validate via an
  orientation-selectivity probe (no hardcoded filters); ensure the learned
  structure is heritable and evolves across generations. See [TASKS.md](TASKS.md).
- **0004 — A/B Comparison and Winner Promotion.** Build a seeded-paired A/B harness
  comparing the emergent encoder (workstream 0003) vs the imported Gabor cortex
  (workstream 0002 + 0008 optimized) on fitness/intent metrics; measure with an A/A
  noise floor and 95% CI
  verdict; promote the measured winner to default; measure and document fitness
  delta, lifespan, approach/avoidance intent trajectories, and generalization
  across seeds. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Vision→action steering alignment is at chance (0.38–0.62) despite encoder separability 55× (1) | `0001` |
| Credit-path TD parameters (decay schedule, trace longevity, clipping, urgency scaling) are the suspect, not the encoder or prior seeding (2) | `0001` |
| Cortex throughput (0.24% of budget) is a separate gate from credit learnability; optimizing it independently allows fair comparison (3) | `0002` |
| Hand-coded Gabor bank proves the mechanical pipeline but conflicts with the emergence principle; self-organizing coding is the vision (3) | `0003` |
| A/B verdict requires seeded-paired determinism, 95% CI, and per-generation fitness/intent trajectories to adjudicate emergence vs import (3) | `0004` |

## Locked decisions

- **Emergence first: learned encoders before (or instead of) imported cortex.**
  The project's founding principle commits to structure arising from learning and
  constraints, not imported design. Workstream 0003 (self-organizing encoder) is
  co-equal with workstream 0002 (optimized cortex) in scope and must ship complete
  before the workstream 0004 A/B.
  If the learned encoder matches Gabor performance it becomes the default; if Gabor
  wins it is promoted, but the learned-encoder code ships complete behind a flag so
  future experiments can revisit. Decision gates: (a) the learned encoder must
  separate food-left/right with cosine-diff >0.5 (matching the Gabor baseline), (b)
  orientation selectivity must be emergent from statistics (tuning >3× on at least
  20% of the learned codes, demonstrating unsupervised structure), (c) the A/B must
  include intent fractions (approach/avoidance) to measure whether the winner also
  enables intent learning. Revisit: if a later experiment shows the learned encoder
  reaches 0.7+ alignment (vs current Gabor 0.24%), flip the default immediately
  without waiting for the full A/B.

- **Homeostasis-only credit path: no shaped rewards in the TD learning signal.**
  The fixes to the credit path (workstream 0001) operate only on the magnitude and
  timing of the homeostatic gradient (`raw_gradient = energy_delta +
  integrity_delta`). No approach/avoidance shaping terms (0012 removed both), no
  auxiliary losses, and no goal signals flow into the TD learning. Urgency
  amplification or decay constants are fair game (they tune the magnitude), but no
  new signal sources are introduced. This locks the solve to constraints: improved
  credit routing from existing homeostatic signals, not signal design. Gate:
  `raw_gradient` must remain derived only from `energy_delta` and `integrity_delta`;
  any addition (approach loss, distance-to-food incentive, etc.) fails the gate and
  is deferred to a separate plan with explicit goal-introduction discussion.

- **The four workstreams are strictly ordered: 0001 → 0002 → 0003 → 0004, not
  rearrangeable.** Workstream 0001 (credit unlock) must complete before 0002/0003
  begin, because steering learnability is the gate for fair encoder comparison.
  Workstreams 0002 (cortex optimization) and 0003 (emergent encoder) run in parallel
  (disjoint files), but both must land before the 0004 A/B. The A/B (0004) requires
  both encoders fully developed and credit unlocked, so the comparison measures
  learning, not throughput or incomplete features. Gate: workstream 0001's steering
  alignment must be ≥0.70 (above chance) before 0002 starts; 0002 and 0003 must both
  pass their respective probes (orientation/phase/position for cortex, emergent
  selectivity for the learned encoder) before 0004. Do not try to parallelize 0001
  and 0002; credit is the prerequisite.

- **A/B verdict is binding and irreversible within this plan.** The A/B
  comparison (workstream 0004) measures fitness, lifespan, and intent fractions over 16
  generations on seeded-paired worlds. The winner is promoted to default (flipping
  the flag); the loser ships behind an off-by-default flag. This verdict holds for
  the next five plans (0021–0025); it cannot be revisited until a new measurement
  (new seed batch, longer generations, intent-gated evolution) is run. If the result
  is inconclusive (CI crosses zero), both flags ship off-by-default and the decision
  is deferred to a follow-on plan with a larger sample. If one encoder regresses in
  a later plan (e.g. orientation selectivity is lost), the decision is revisited and
  documented then, but the workstream 0004 verdict remains the historical record.

## Out of scope

- **Improving the legacy raycast encoder (direct 8×6 raycasting path).** The legacy
  path is the baseline; this series assumes it remains constant. Future plans may
  optimize raycasting (fewer rays, better culling), but it is outside this plan's
  scope, which compares two vision alternatives (cortex vs emergent) on top of
  legacy as the fallback.

- **Graduated lever defaults for effort-rebased fitness, danger percept, or
  speed-cost exponent.** Plans 0009, 0010, 0012, 0013 left these flags off;
  this plan preserves that state. A later plan (0021+) may graduate the fitness
  levers after the workstream 0004 A/B anchors the learning-signal quality; that is
  out of scope here.

- **Multi-agent competition or environmental-pressure fitness shaping.** The 0014
  review surfaced this as a long-term alternative to hand-shaped fitness; it is a
  research direction, not a tactical fix. Defer to a follow-on series after the
  emergence A/B in this plan is complete.

- **GPU kernel workgroup restructuring or dispatch optimization.** Workstream 0002's
  cortex optimization stays within the existing `coop_visual_cortex()` function;
  restructuring the kernel (e.g. splitting into sub-passes, multi-workgroup
  parallelism) is out of scope (0005 and 0006 already explored this). Only
  intra-pass optimizations (separable convolutions, shared memory, etc.) are in
  scope.

- **Intent-as-meta-score evolved-policy changes for the A/B arms.** Plan 0016 (in
  progress) builds the intent measurement framework. Workstream 0004's A/B uses that
  framework as a measurement target but does not change the evolution algorithm, mutation
  rates, or selection pressure. The A/B is purely an observational comparison, not a
  lever flip.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
