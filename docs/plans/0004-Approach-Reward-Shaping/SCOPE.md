# Scope — Plan 0004

> Give the within-lifetime reward a spatial approach gradient — today
> `raw_gradient` is purely interoceptive (energy/integrity deltas), so the
> learner is rewarded by how it *feels*, never by its *relation to food*, and
> no amount of knob-tuning downstream can teach vision-conditional steering —
> then restore the selection signal and gate the reactive/heritable/perceptual
> follow-ups behind the one measurement that proves the unlock.

## Why this plan

This plan compiles the 2026-06-14 learning-and-evolution root-cause review
(`docs/reviews/2026-06-14-claude-opus-48.md`), which measured a ~24 h / 1834-
generation production run (`xagent.db`) and diagnosed a single architectural
error: **the learning reward is spatially blind.** It is the genuine delivery
of Plan 0001's stated goal — "ground the survival reward in space so steering
can be learned" — which Plan 0001 named but never implemented (Plan 0001
landed hazard-touch, the terminal-death TD update, same-cycle interoception,
the multiplicative survival gate, and the stride/lag sweep, but never added an
approach term to `raw_gradient`).

**Code claims verified against `develop` @ `1f1b421` by direct reading; the run
metrics are the review's measurements of `xagent.db` (1834 generations,
`best_score` 0.00208, alignment 0.498 at chance, ~0.14 food/1k-ticks vs ~1.5
break-even).** Claims that needed restatement during verification are recorded
at the bottom of this section; none were rejected outright. The diagnosed,
verified findings:

1. **The reward carries no approach gradient.**
   `raw_gradient = energy_delta·ENERGY_WEIGHT + integrity_delta·INTEGRITY_WEIGHT`
   (`brain_passes.wgsl:170`, `ENERGY_WEIGHT=0.6`/`INTEGRITY_WEIGHT=0.4` at
   `common.wgsl:264-265`), amplified to `raw_gradient·(1+urgency)`
   (`brain_passes.wgsl:188-190`) and consumed verbatim as the TD reward
   (`let reward = s_homeo[1u];`, `brain_passes.wgsl:423`). No distance/bearing/
   approach term enters the reward path. During the entire approach to *visible*
   food the reward is a near-constant metabolic drain — identical whether
   steering toward food or away — until the single contact-eat tick.
2. **The actor's REINFORCE gradient has nothing bearing-correlated to latch
   onto, and its rate is throttled.** The turn trace is
   `turn_trace[d] ← γλ·turn_trace[d] + noise_turn·s_encoded[d]`
   (`brain_passes.wgsl:711-712`) and the weight step is
   `ACTION_WEIGHT_LEARNING_RATE · TD_VECTOR_SCALE · δ · turn_trace[d]`
   (`brain_passes.wgsl:454-455`). Because `δ` is built from the blind reward,
   `E[δ·noise_turn·s_encoded]` carries no sign-correct, state-conditional
   information; under a mirrored-food probe the conditional component averages
   to zero. Separately, `TD_VECTOR_SCALE = 1/ENCODED_DIMENSION = 1/128`
   (`common.wgsl:306`) is a *critic*-stability scale that also crushes the
   *actor* step to `0.10/128 ≈ 8e-4`, so even a sign-correct `δ` would latch
   slowly.
3. **Klinotaxis is driven by self-energy, not an external gradient.**
   `gradient_deviation = s_homeo[3u] − s_homeo[4u]` (`brain_passes.wgsl:653`) is
   `gradient_fast − gradient_medium` — two EMAs of the *same* interoceptive
   `raw_gradient` — so both relax equally during a flat-reward approach and
   `klinotaxis_factor → 1.0` (inert). Real chemotaxis compares an *external*
   concentration across time; the reactive layer is wired to the wrong input.
4. **Memory is a second spatially-blind learner that can teach avoidance.**
   Pattern valence is written from `raw_gradient`
   (`brain_passes.wgsl:787,:813`) and recall blends stored motor by
   `sim·valence` (`brain_passes.wgsl:536-539`); a food-in-view state accrues
   *negative* valence during the draining approach, so recall injects a
   negative-valence blend that, per the code's own comment
   (`brain_passes.wgsl:527`), "negates approach → escape."
5. **The policy/critic learning constants are hardcoded; the evolvable genes
   cannot move steering dynamics.** `ACTION_WEIGHT_LEARNING_RATE` (`:277`),
   `KLINOTAXIS_SENSITIVITY` (`:281`), `TD_DISCOUNT` (`:294`), `TD_LAMBDA`
   (`:298`), and `CRITIC_LEARNING_RATE` (`:301`) are all `const` in
   `common.wgsl`. The evolvable `CFG_LEARNING_RATE`/`CFG_DECAY_RATE` genes drive
   only the peripheral predictor (`brain_passes.wgsl:320`) and pattern-memory
   reinforcement/valence (`brain_passes.wgsl:736,:781,:784,:787`) — never the
   actor/critic policy weights. Evolution searches a subspace disconnected from
   the policy's learning dynamics.
6. **Fitness has no foraging-primary objective; the survival gate compresses
   the live range.** `composite_fitness` is
   `survival·(foraging·0.5 + exploration·0.5)` with
   `survival = 1/(1+death_count·0.5)` (`governor.rs:62-67`). At ~276 deaths/gen
   the gate collapses the composite into the ~0.0014 tail, below the eval-noise
   floor — the foraging variance selection must see is crushed.
7. **The experiment cannot resolve learning.** Unique configs are coupled to
   repeats — `unique_count = (population_size / eval_repeats).max(1)`
   (`governor.rs:944`), each repeated to fill the population
   (`governor.rs:997-998`) — giving only ~2–5 unique configs/gen; the accept
   rule is a bare `gen_avg ≥ parent_fitness` (`governor.rs:627`) with no
   significance guard, and `num_islands` defaults to 3. Even a working learner's
   ~8 % edge is statistically invisible at this resolution.
8. **No within-lifetime learning metric on the production governor path.** The
   q1→q4 within-life food-rate exists only in the headless side-tool
   (`run_headless`/`quarter_rates`, `headless.rs:173-265,:311-325`);
   `Governor::evaluate`/`log_generation` (`governor.rs:482,:267`) record only
   end-of-life aggregates, so the production run could not detect within-life
   learning even if it occurred.
9. **Sensory lag and vision-row geometry blind the agent at navigational
   range.** The `lag100` default (`vision_stride·brain_tick_stride = 10·10`,
   `config.rs:163-169`) freezes the encoded frame for 100 physics ticks while
   the agent moves; the 8×6 ray grid over a 90° FOV
   (`VISION_W=8`/`VISION_H=6`, `VISION_FOV_HALF=PI/4`, `VISION_MAX_DIST=30`,
   `common.wgsl:13-14,:203-204`; `phase_vision.wgsl:31-37`) puts half the rows
   above the horizon, so ground food beyond ~5 units falls between ray rows.
   Both throttle a *fair* evaluation of finding 1.

The causal chain is deterministic: no approach gradient (1) ⟹ the actor cannot
acquire state-conditional steering (2) ⟹ diffusive foraging at ~0.14 food/1k
vs ~1.5 break-even ⟹ chronic starvation crushes the survival-gated fitness (6)
into the noise floor ⟹ selection random-walks (7) and finds only the metabolic-
cost proxies (`processing_slots` 16→1, `memory_capacity` 128→1, both mutated
metabolic proxies per `config.rs:25-36`). Fix any downstream costume and you get
another flat 1834-generation run; fix the reward and the rest becomes a tuning
problem.

**Claims refined during verification** (recorded so the refinements are not
re-litigated; none of the findings were rejected):

| Claim as written in the review | Status | Refinement |
|---|---|---|
| "The data already exists in the food-detect pass (`kernel_tick.wgsl:229-235`)" — implying `Φ` can read nearest-visible-food for free | Refined | The per-food `dx/dz` is computed there, but `agent_food_detect` reduces only the nearest food *within `eat_radius`* (`kernel_tick.wgsl:232`). Nearest-food-within-shaping-radius must be added as a new reduction + telemetry slot (see ARCHITECTURE §0001). |
| F4: "`CFG_LEARNING_RATE` feeds *only* the peripheral predictor (`:320`)" | Refined | It also drives pattern-memory reinforcement and valence learning rate (`learning_rate = brain_config[1].x`, `brain_passes.wgsl:736`, used at `:781,:784,:787`). The substantive claim holds: it never touches the actor/critic policy weights, which use the `const` rates. |
| Run config "`population_size=5`" | Refined (provenance) | That was the *run's* config; the `develop` default is now `population_size: 10` (`config.rs:265`). The structural defect — `unique_count = population/eval_repeats` (`governor.rs:944`) — is independent of the exact default and is what finding 7 addresses. |

## In scope

Work items in [TASKS.md](TASKS.md) (workstreams `0001`–`0005`):

- **0001 — Approach reward shaping (the unlock).** Pin the pre-change alignment
  and foraging baselines; surface nearest-food-within-shaping-radius from the
  food-detect pass; add potential-based approach shaping `F = γΦ(s′) − Φ(s)`
  with `Φ(s) = −APPROACH_SHAPING_GAIN·d_norm` into `raw_gradient`; split the
  actor's vector scale from the critic's `1/128`; re-measure the alignment and
  foraging probes against a hard decision rule.
- **0002 — Reactive & valence layers on the external gradient (GATED on 0001).**
  Drive klinotaxis from the external food-distance gradient instead of the
  self-energy EMA difference, and confirm/clean memory valence so a food-in-view
  state recalls *toward* food. Opened only once 0001 moves the probe.
- **0003 — Selection signal restoration.** Make `food_per_1k_alive_ticks` the
  primary fitness objective; decouple the unique-config count from `eval_repeats`
  with independent per-repeat seeds and `num_islands → 1`; add a
  `child − parent > k·pooled_stderr` significance guard; surface the q1→q4
  within-life food-rate on the production governor path.
- **0004 — Heritable learning dynamics (GATED on 0001).** Promote the policy/
  critic learning constants (`ACTION_WEIGHT_LEARNING_RATE`,
  `CRITIC_LEARNING_RATE`, `KLINOTAXIS_SENSITIVITY`, `TD_DISCOUNT`, `TD_LAMBDA`,
  and the new actor vector scale) into `BrainConfig` genes so evolution can tune
  the dynamics that produce steering.
- **0005 — Sensory lag & vision geometry (GATED on 0001; re-opens Plan 0001's
  lag verdict).** Re-measure the stride/lag sweep now that steering exists and
  make `vision_stride` heritable if the foraging gain beats the tps cost; fix the
  vision-row geometry so navigational-range ground food is visible.

## Origin → workstream mapping

| Finding | Addressed by |
|---|---|
| Reward carries no approach gradient (1) | `0001` |
| REINFORCE gradient blind + actor rate throttled (2) | `0001` |
| Klinotaxis driven by self-energy (3) | `0002` (gated) |
| Memory is a second spatially-blind learner (4) | `0002` (gated) |
| Policy/critic constants hardcoded; genes inert (5) | `0004` (gated) |
| No foraging-primary objective; survival gate compresses range (6) | `0003` |
| Experiment cannot resolve learning (7) | `0003` |
| No within-life metric on the governor path (8) | `0003` |
| Sensory lag + vision-row geometry blind distal food (9) | `0005` (gated) |

## Locked decisions

- **Potential-based shaping, added to `raw_gradient` before the EMAs.** Define
  `Φ(s) = −APPROACH_SHAPING_GAIN · d_norm`, `d_norm = clamp(nearest_food_dist /
  SHAPING_RADIUS, 0, 1)`, and add `F = TD_DISCOUNT·Φ(s′) − Φ(s)` to `raw_gradient`
  at `brain_passes.wgsl:170`. Adding it *there* (not only to the TD reward at
  `:423`) is deliberate: the same edit feeds the reward, the homeostatic EMAs
  that drive klinotaxis, and the memory valence — exactly the propagation the
  review prescribes. Potential-based shaping is provably optimal-policy-invariant
  for the TD objective (Ng et al. 1999): the added return telescopes to
  `−Φ(s₀) + γᵀΦ(s_T)`, so it cannot corrupt the eat objective, only accelerate
  credit toward it.
- **`Φ` is over nearest food within `SHAPING_RADIUS`, a robust proxy for
  "visible."** PBRS invariance holds for *any* state potential, so a radius proxy
  (set to `VISION_MAX_DIST = 30`) is sound and avoids an FOV-visibility test in
  the food-detect pass. The perception gap (food not actually visible at range)
  is closed separately and gated in `0005`; `Φ` correlating with what the policy
  can see via `s_encoded` is what makes the shaped `δ` *learnable*, and that is
  what the `0001` remeasure tests.
- **Initial `APPROACH_SHAPING_GAIN = 0.05`, validated then made heritable.**
  Sized so the per-brain-tick `F` (~`GAIN·0.05`–`GAIN·0.1` ≈ `2.5e-3`–`5e-3`)
  dominates the ~`2e-4` metabolic drain (making approach the dominant steering
  signal) while staying well below the ~`0.12` contact-eat spike (so the eat
  objective still anchors). It is a starting value the `0001` remeasure
  validates; `0004` promotes it to a gene if the unlock lands.
- **The actor gets its own vector scale, separate from the critic's `1/128`.**
  `TD_VECTOR_SCALE = 1/128` is a critic-stability argument; the actor inherits it
  only by accident. Introduce `ACTOR_VECTOR_SCALE` (initial `1/16`) for the
  forward/turn weight steps so the actor can latch onto the now-sign-correct `δ`
  at a usable rate. Bounded by the existing `MAX_WEIGHT_NORM` L2 ball — no new
  divergence path. Validated by the `0001` remeasure, promoted to a gene in
  `0004`.
- **The `0001` remeasure is the gate for everything else (prove-or-kill).** The
  diagnosis predicts that after shaping + actor scale, turn/bearing alignment
  moves decisively above the existing `0.62` chance-band edge
  (`learning_probe_baseline_turn_alignment_is_chance`, `integration.rs:2019-2023`)
  and free-run foraging rises over a fixed-seed run. If alignment stays at chance,
  the encoder/representation is the next suspect and the negative is recorded —
  `0002`/`0004`/`0005` do not open. Code tasks gate on *deterministic* mechanism
  tests; the stochastic behavioral probes are owned by the remeasure.
- **`0003` is unconditional; `0002`/`0004`/`0005` are gated.** The selection-
  signal work improves the experiment regardless of whether shaping lands (it is
  what lets selection *see* any learning), so it is not gated. The reactive,
  heritable, and perceptual follow-ups are land-or-record-the-negative gated on
  the `0001` remeasure, so the plan cannot sprawl ahead of its one load-bearing
  measurement.
- **`0005` re-opens Plan 0001's lag verdict on purpose.** Plan 0001 concluded
  `lag100` is the default because higher strides cost more tps than the learning
  gain — but that was measured under the spatially-blind regime, where there was
  no steering to gain from fresher state. The verdict is only valid pre-shaping;
  `0005` re-measures it post-shaping and adopts a smaller lag only if the
  re-measured foraging gain beats the tps cost (same budget rule as Plan 0001).
- **Measurement first, always.** Nothing in `0001` past the baseline merges
  without before/after probe numbers (alignment band, foraging rate), fixed
  seeds, and green `fmt`/`clippy`/`test` in the PR body — the discipline of
  Plans 0001/0003.

## Out of scope

- **Encoder / representation changes.** Two measured negatives (the 2026-06-10
  Phase-2 reconstruction revert; the separability diagnostic) put encoder work
  behind a probe regression that implicates representation. The `0001` remeasure
  is exactly that probe: encoder changes unlock only if alignment stays at chance
  *after* shaping lands.
- **Removing the food-teleport-on-eat discontinuity.** Food respawning elsewhere
  on each eat injects a one-tick `Φ` blip, but PBRS invariance absorbs it; a
  smoother respawn is a separate world-economics change, deferred until the
  remeasure shows the blip materially hurts credit.
- **Clamping `F` out of the urgency amplification.** `s_homeo[1] =
  raw_gradient·(1+urgency)` now amplifies `F` too, which can over-weight late-
  approach credit as energy drops (the review's urgency-asymmetry secondary). It
  is left in for `0001` (the review prescribes adding to `raw_gradient` at `:170`)
  and revisited only if the remeasure shows urgency inverting late-approach
  credit.
- **Within-batch encoded-state staleness** (the encoded frame frozen for a full
  vision-stride while physics advances). Folded into the `0005` lag re-measure,
  not addressed independently.
- **Respawn energy economics, memory store-gating, exploration-floor decay,
  efference-copy predictor input.** All Plan 0001 out-of-scope items remain out
  of scope here; none is on the approach-gradient critical path.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
