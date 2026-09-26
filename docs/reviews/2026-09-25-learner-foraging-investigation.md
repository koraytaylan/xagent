# Investigation: why the learner does not learn foraging

**Date:** 2026-09-25
**Base:** `0152212` on `develop` (experiments ran in a scratch worktree whose tree is identical to that commit; the working-tree homeostatic-predictor changes were not present and the predictor flag stays off)
**Trigger:** a 43-generation GUI run under the new defaults (effort-rebased fitness, quadratic drag) showed no score improvement: approach intent 0.485 (chance 0.5), within-life food rate falling from 0.45 to 0.30 per 1 000 alive ticks, and executed forward drive falling from 0.175 to 0.129 during each life.

## Method

All numbers below come from GPU runs on the AMD Raphael iGPU (RADV), `--test-threads=1`. The experiments were throwaway tests plus throwaway shader switches in the scratch worktree; none of that code is committed. The switches reused config slots that are inert while `homeo_predictive_credit_enabled = false`:

- **Encoding centering.** The actor, and optionally the critic, reads `encoded − EMA(encoded)` (rate 0.01) instead of `encoded`. The EMA lives in the write-only prediction-error ring.
- **Actor step boost.** The actor bias and weight updates are multiplied by 10 or 100.
- **Diagnostic oracle.** Not homeostatic, diagnostic only. It adds `|bearing_prev| − |bearing_now|` of the nearest food to `raw_gradient`: a dense reward that depends directly on turning.

Three probe geometries were used. Each has 16 agents with food alternating left/right, is mirrored every episode, and is scored with `score_turn_alignment`:

| Probe | Vision | Food distance / bearing | Walking straight eats it? |
|---|---|---|---|
| Standard (`learning_probe_mirrored_steering_is_chance`) | 8×6 | 5 / ±0.405 rad → lateral 1.97 | **yes** (eat radius 2.0) |
| Steering-required | 17×13 | 10 / ±0.405 rad → lateral 3.94 | no |
| Steering-required, blind control | 8×6 | 10 / ±0.405 rad | no (food invisible) |

The **decomposition** reads each trained agent's turn pre-activation `b + w·φ` for the food-right and food-left scenes. It reports `|common|`, the side-independent part, and `|diff|`, the part that flips with the food side. `diff_correct` counts agents whose `diff` has the steering sign.

Free runs used 10 agents for 100 000 ticks in the default world with the default brain (8×6, strides 10/10).

## Findings

### F1 — The default eye cannot see food beyond ~5.5 units

`phase_vision.wgsl` places row *r* at `v = r/(H−1)·2 − 1`. For H = 6 that gives v ∈ {±0.2, ±0.6, ±1.0}, with no row near the horizon. From eye height 1.0, the shallowest downward row (slope 0.2) meets flat ground about 5 units out. The upward rows intersect the food sphere (center 0.35, radius 1) only within about 1.75 units.

At the default density of 0.005, the mean nearest-food distance is about 7 units, so food is rarely in view. When it is, it is usually close enough that walking straight eats it. The existing test `vision_horizon_row_sees_food_at_range` already pins this: the 8×6 grid cannot see food at 20, while 17×13 sees every distance.

### F2 — The standard steering probe cannot detect steering learning

The probe food is 1.97 units off-axis with an eat radius of 2.0, so an agent walking straight eats it. Turning carries almost no reward. Every arm ate the same amount (1029–1043 of 1920 agent-episodes) whatever it learned about turning. Every credit-path verdict in plans 0017–0023 was measured on this probe, and so was the 2026-06-11 "17×13 does not help" revert. Those results cannot distinguish "the learner can't steer" from "the probe gives no reason to steer".

### F3 — The learned turn output ignores which side the food is on

Standard probe, 120 episodes, seeds 17/23/29:

| Arm | alignment | \|common\| | \|diff\| | diff_correct |
|---|---|---|---|---|
| baseline | 0.395 / 0.415 / 0.439 | 0.156–0.195 | 0.0006–0.0008 | 9, 7, 8 of 16 |
| actor-centered | 0.503 / 0.494 / 0.499 | 0.037–0.044 | 0.0002–0.0004 | 8, 10, 12 of 16 |
| actor+critic-centered | 0.501 / 0.509 / 0.520 | 0.038–0.046 | 0.0002–0.0004 | 9, 11, 11 of 16 |

The side-independent part is about 250× the side-dependent part, and the side-dependent sign is random. Centering removes most of the common mode, which brings alignment from below chance back to 0.50, but it creates no side dependence. The encodings of the two sides stay 98.2–98.6% similar (cosine) after training.

### F4 — Where steering is required, exploration never finds the reward

Steering-required probe, 240 episodes. Both 17×13 arms, baseline and actor+critic-centered, stay at alignment 0.465–0.499. Success is about 2% in the first sixth of training and at most 9–10% in the last sixth (baseline, seeds 17/23). The 8×6 blind control behaves the same (0.481–0.488). With homeostatic reward only, random turning almost never reaches off-axis food, so there are almost no reward events to learn from.

### F5 — The actor update cannot extract a steering direction even when the reward depends directly on turning

Steering-required probe, 17×13, 240 episodes:

| Arm | seed | success first → last | \|common\| | \|diff\| | diff_correct |
|---|---|---|---|---|---|
| oracle ×1 | 17 / 23 | 0.052→0.152 / 0.041→0.131 | 8.72 / 5.87 | 0.010 / 0.000 | 2 / 1 of 16 |
| oracle ×10 | 17 / 23 | 0.306→0.297 / 0.305→0.275 | 9.22 / 10.78 | 0.002 / 0.364 | 2 / 3 of 16 |
| oracle ×100 | 17 / 23 | 0.387→0.387 / 0.358→0.406 | 15.64 / 17.75 | 0.032 / 0.001 | 1 / 1 of 16 |
| homeostatic ×10 | 17 / 23 | 0.134→0.134 / 0.102→0.142 | 1.07 / 0.61 | 0.0015 / 0.0010 | 11 / 10 of 16 |
| homeostatic ×100 | 17 | 0.064→0.064 | 3.21 | 0.009 | 11 of 16 |
| centered, homeostatic ×100 | 17 | 0.139→0.039 | 1.06 | 0.003 | 8 of 16 |
| centered, oracle ×10 | 17 | 0.220→0.180 | 0.91 | 0.003 | 10 of 16 |
| centered, oracle ×100 | 17 | 0.328→0.327 | 1.04 | 0.504 | 9 of 16 |
| centered, standard probe, homeostatic ×100 | 17 | 0.272→0.114 | 1.13 | 0.008 | 11 of 16 |

- Even with a dense, turn-contingent reward, learning drives the turn channel into a saturated constant rotation rather than side-dependent steering.
- Larger steps enlarge the common mode, not the difference. When the difference does grow (0.36, 0.50), its sign is random.
- In the homeostatic arms, training lowers success.

The oracle also produced reward spikes at episode resets and when food left sense range, so it supports this finding rather than settling it. The data are consistent with the likelihood-ratio actor update (action-blind δ × exploration-noise × feature traces) having far too little signal per step for the direction that matters. The old claim that "magnitude is not the bottleneck" came from a CPU-only normalization that never changed GPU weights, so it does not contradict this.

### F6 — Within each life, the hazard dose teaches the agent to stop moving

Free runs (per agent, 100 000 ticks):

| Arm | food | deaths | distance | exec forward | forward bias Q1→Q4 |
|---|---|---|---|---|---|
| default | 19.4 | 18.8 | 8 898 | 0.075 | 0.243→0.142 |
| movement energy cost 0 | 16.5 | 21.0 | 9 574 | 0.039 | 0.233→0.142 |
| hazard damage 0 | **25.0** | **12.9** | **12 915** | **0.192** | **0.346→0.384** |
| hazard 0 and movement cost 0 | 23.8 | 12.6 | 11 653 | 0.154 | 0.305→0.258 |

Integrity loss in danger is proportional to distance moved (`kernel_tick.wgsl:199-212`). At full speed that is about 0.02 of reward per brain tick, roughly 20× the per-tick energy drain and over 100× the movement energy cost. A stationary agent takes no dose. So the only dense, action-contingent signal the actor receives says "don't move", and inside danger it rewards freezing over escaping.

Removing the hazard dose stops the decay of forward drive and raises food by 29%. Removing the movement cost alone does not. The GUI run's falling within-life food rate matches this.

### F7 — Fatigue collapses motor output through its own feedback

Expected travel is built from the pre-fatigue forward command (`brain_passes.wgsl:1547-1587`). Actual displacement already includes the fatigue multiplier (`brain_passes.wgsl:1601`, then `kernel_tick.wgsl:96-103`). With a path efficiency *k* below 1 (curvature, backward noise), fatigue settles at `floor / (1 − k·(1 − floor))`. That is about 0.36 at *k* = 0.8 and 0.53 at *k* = 0.9, versus 1.0 at *k* = 1.

Measured mean fatigue is 0.61 in the free run and about 0.4 in the GUI recording. Disabling fatigue (`fatigue_floor = 1.0`) gives +41% distance and +17% food.

## What would have to change

These are listed by expected leverage. None are implemented.

1. **Make the probe able to measure steering.** Place food off-axis beyond the eat radius and make it visible, as in the steering-required geometry above. Otherwise no credit-path change can be validated.
2. **Remove the "freeze" incentive.** Make hazard damage accrue per tick in danger rather than per distance, or make escaping cheaper than staying. With the current dose, TD correctly learns not to move.
3. **Fix the fatigue feedback.** Compare displacement with the executed (post-fatigue) command, so fatigue detects obstruction only.
4. **Let the eye see food at range.** An odd grid (17×13) puts a row on the horizon. It costs 4.3× the features (265 → 1130).
5. **Replace or restructure the actor credit estimator.** Centering the actor features is necessary to stop common-mode drift but not sufficient. Candidates within the homeostatic-only contract, none tried yet: normalized advantages on the GPU, an action-conditioned forward model `(s, a) → s′` whose predicted homeostatic change credits the chosen turn, and n-step or episodic returns.

## Outcome of implementing the recommendations

Measured the same day on the same machine. Each option was implemented, A/B-tested against the unmodified learner, and kept only if it helped. Three kinds of measurement were used:

- **Free runs:** 10 agents, 100 000 ticks, default world, seeds 5 and 6.
- **Probes:** the steering-required and standard probes above.
- **Headless evolution:** release builds, population 10, 2 evaluation repeats, 40 000 ticks per generation, 19–20 generations.

Mean agent composite fitness and food per agent below are for the first and last five generations.

| Option | What was built | Kept? |
|---|---|---|
| 1. Steering-required probe | `learning_probe_steering_required_is_chance`. Food 10 units away, 3.94 off-axis, 17×13 eye. Baseline alignment 454/892 = 0.509, late success 0.031. | **yes** |
| 2. Remove the freeze incentive | Hazard dose floored at one default step per tick; also a 0.15-step floor variant | no, reverted |
| 3. Fatigue feedback | Staleness measured against the executed command, as the absolute-value sum; also a signed-sum variant | no, reverted |
| 4. See food at range | 17×13 eye (existing config) | no, default stays 8×6 |
| 5. Actor credit estimator | Actor feature centering and GPU advantage normalization, as runtime switches (not committed) | no |

### Evolution

| Arm | Fitness, first → last | Food/agent | Deaths/agent |
|---|---|---|---|
| **original learner** | **0.0330 → 0.0352** | 9.0 → 10.4 | 6.0 → 7.6 |
| floor 1.0 + abs fatigue | 0.0171 → 0.0176 | 6.4 → 6.4 | 11.3 → 10.1 |
| floor 1.0 + abs fatigue + 17×13 | 0.0194 → 0.0208 | 7.8 → 8.8 | 14.8 → 14.8 |
| floor 1.0 + abs fatigue + actor centering | 0.0208 → 0.0214 | 8.0 → 7.7 | 14.0 → 11.8 |
| floor 0.15 + abs fatigue | 0.0283 → 0.0257 | 9.0 → 8.1 | 8.5 → 7.6 |
| abs fatigue only | 0.0218 → 0.0232 | 6.1 → 6.2 | 6.5 → 5.8 |
| signed fatigue only | 0.0292 → 0.0308 | 8.7 → 9.3 | 7.3 → 7.2 |

No arm beat the original learner. Every arm stayed at chance approach intent (0.48–0.50).

### Why the mechanism fixes did not become fitness

**Hazard floor.** It did what it was meant to: in free runs the forward bias stopped collapsing within a life. But the learner cannot learn to avoid danger, so a lethal dwell dose mostly adds deaths: 2.2–2.7× in free runs, about 1.8× in evolution. With the full floor, agents also explored half as many cells (136 vs 263). A floor sized to exploration speed (0.15) kept food level but still raised deaths, and the survival multiplier turned that into lower fitness.

**Fatigue.** The absolute-value accumulator counts forward/backward exploration jitter that cancels in real displacement. Fatigue then punished noise, and distance fell 27%. In free runs with the absolute-value accumulator, fatigue still sat at 0.47–0.50 (original 0.48–0.61), so in hilly terrain path curvature, not the feedback loop, sets most of it. The signed version, which avoids counting jitter, was evaluated only in evolution: food stayed level with the original (8.7 → 9.3 vs 9.0 → 10.4) while early deaths rose (7.3 vs 6.0), so the extra movement mostly bought hazard exposure.

**17×13 eye.** It saw food at range but nothing used it: approach intent stayed at chance. It cost 28–32% throughput, and fitness gains were within noise.

**Actor centering and normalization.** In free runs with the dose floor and absolute-value fatigue fix in place:

| Arm | Food/agent (seeds 5 / 6) | Deaths/agent | Forward bias |
|---|---|---|---|
| control | 17.0 / 23.5 | 38 / 48 | stable |
| centered | 18.6 / 25.3 | 40 / 44 | stable |
| normalized | 22.3 / 24.6 | 64 / 54 | oscillates −0.90 … +0.66 |
| centered + normalized | 24.4 / 25.0 | 67 / 59 | oscillates |

Approach intent stayed at 0.487–0.501 everywhere. Normalization makes the policy unstable and adds deaths; centering shows no steering and no significant food gain. Neither was committed.

### Where this leaves the learner

The tested levers are mechanical: movement costs, fatigue, eye range, and actor step statistics. None of them creates the missing capability, which is turning toward food that is seen. Fitness here rewards food per unit of energy and survival, so more movement without steering only buys energy cost and hazard exposure.

The remaining candidates from finding F5 need a structurally different credit signal for the turn channel:

- an action-conditioned forward model `(s, a) → s′` whose predicted homeostatic change credits the chosen turn;
- n-step or episodic returns.

The steering-required probe is now the instrument to judge them. The mirrored probe cannot see steering, so it can't.

## Episodic memory with homeostatic salience

Implemented after the above as the episodic-return candidate. The agent is not told what to remember. Every brain tick stores a moment with no valence. A tick is *salient* when its homeostatic change (the energy/integrity `raw_gradient`) lies at least 3 standard deviations from the agent's own running normal. A salient tick credits its signed, normalized label back to the moments stored over the preceding 8 brain ticks, discounted by γλ per tick. Death adds no label of its own. Fully valued moments do not fade with time, and eviction keeps moments that are still awaiting their outcome.

Memory keys are now the encoded state minus its running mean. Raw encodings of different scenes are 98–99% alike by cosine, so uncentered recall could not tell scenes apart.

The work also fixed an eviction bug: inactive slots scored 999 as a keep score, so every store overwrote the same slot. Before the fix, memory never held more than one pattern.

**Free runs.** Setup: default 8×6 eye, 10 agents, 100k ticks, seeds 5–10. Each arm's food per agent is paired against the original learner (19.88) on the same seeds:

| Arm | Valued memories/agent | Food/agent | Paired difference |
|---|---|---|---|
| original learner (one-pattern memory) | 0 | 19.88 | — |
| salience credit, death labeled −1, valued memories fade (seeds 5/6 only) | 3–10 | 19.4 | −0.3 |
| valued memories retained, death labeled −1 | 118–119 | 18.57 | −1.32 (t = −2.4) |
| valued memories retained, no death label (**committed**) | 95–113 | 18.75 | −1.13 (t = −1.9) |
| committed, memory motor replay off | 97–116 | 19.12 | −0.77 (t = −0.9) |

Approach intent stayed at 0.484–0.501 in every arm. With the 17×13 eye (seeds 5/6), the committed arm ate 18.7 / 19.8 against 19.5 / 18.4.

**Probes.** Seeds 17 / 23. The committed arm against the original learner:

| Probe | Metric | Original learner | Committed arm |
|---|---|---|---|
| Standard | success | 0.542 → 0.545 / 0.544 → 0.538 | 0.533 → 0.530 / 0.541 → 0.533 |
| Standard | alignment | 0.466 / 0.488 | 0.398 / 0.462 |
| Steering-required | success | 0.020 → 0.097 / 0.022 → 0.094 | 0.023 → 0.131 / 0.023 → 0.106 |
| Steering-required | alignment | 0.465 / 0.487 | 0.453 / 0.477 |

The mechanism works as designed: the tests in `crates/xagent-brain/tests/episodic_memory.rs` pin salience, credit, retention and eviction, and memory fills with valued episodes. What the brain does with those episodes does not steer.

Recall blends in the stored motor commands of similar valued moments. That blend neither raises approach intent nor aligns turns with the food's side. Food per agent falls by about 1 unit (5–7%). Switching the replay off recovers about a third of that loss. The rest is within noise of the original learner.

A valued memory therefore needs a use other than replaying the motor command it was stored with. The salience label is a candidate reward event for an action-conditioned model or an n-step return.

## Does the encoded state carry the food's side?

This test only measured and changed nothing in the agent. It fit a linear readout on the encoded state, the 128 values the turn head reads.

**Brains tested:**

- 10 fresh brains, with random encoders;
- 20 brains trained by 100k-tick free runs, seeds 5 and 6.

**Setup.** Each brain was copied to 16 agents, pinned in place on flat ground. One food item was placed per agent:

- side: left or right at random;
- bearing: 0.1–0.6 rad;
- distance: 2.5–5 units, inside the default 8×6 eye's range.

That gave 640 scenes per brain. Scoring used an L2 logistic regression with 5-fold held-out accuracy, taking the best of three regularization strengths.

| Brains | Encoded state | Input features | Encoded, shuffled labels | Sign of the policy's own turn output |
|---|---|---|---|---|
| fresh | 0.994 ± 0.002 | 0.996 ± 0.002 | 0.506 | 0.499 |
| trained, seed 5 | 0.995 ± 0.002 | 0.997 ± 0.003 | 0.497 | 0.505 |
| trained, seed 6 | 0.997 ± 0.003 | 0.998 ± 0.002 | 0.498 | 0.499 |

No brain scored below 0.991.

The food's side is linearly present in exactly the space the turn head reads, and the encoder keeps it whether trained or not. The learned turn weights still point at chance, so the missing capability is not perception or representation. It is the learning of the turn weights from the homeostatic signal.

## Does the executed turn erase the exploration noise the actor credits?

The turn trace credits the raw noise kick, `noise_turn × exploration_rate`. The executed turn is different: `clamp(tanh(policy) + kick)`, scaled by fatigue and klinotaxis, then clamped to ±1 again by physics. A scratch build counted, on every brain tick, how much of the recorded kick survived into the executed turn:

| Run | Kick erased | Kick partly clipped | Fraction surviving the clamps | \|policy turn\| | Fatigue × klinotaxis below 0.5 | corr(recorded, expressed) |
|---|---|---|---|---|---|---|
| free run, 100k ticks, seed 5 | 0.0% | 0.7% | 0.998 | 0.135 | 49% | 0.78 |
| free run, 100k ticks, seed 6 | 0.1% | 2.3% | 0.991 | 0.188 | 46% | 0.78 |
| standard probe, 240 episodes | 0.3% | 2.0% | 0.990 | 0.147 | 23% | 0.85 |
| steering-required probe, 240 episodes | 0.0% | 0.1% | 1.000 | 0.030 | 11% | 0.95 |

Clamping almost never erases the kick, because the learned turn output stays far from saturation. The mismatch that does exist is multiplicative: fatigue halves the turn on about half of the free-run ticks, and klinotaxis rescales it by 0.3–3. As a result, the credited noise explains only about 60% of the variance of the noise the agent actually expressed (correlation 0.78).

That weakens the actor's signal but cannot explain chance-level steering. Meanwhile the turn output grows during learning (free-run |policy turn| 0.08 → 0.23) while alignment stays at chance: the learned turn is side-blind, not suppressed.
