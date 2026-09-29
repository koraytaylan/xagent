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

## Does the critic's value change with where the food is?

Between meals, the only TD signal that could reward turning toward food is the critic's value rising as food gets nearer and more centered in view. This test only measured and changed nothing in the agent.

**Setup.** Same pinned-agent arena as the readout test. Food was placed in view (2.5–5 units, 0–0.7 rad off-center) in 70% of scenes and directly behind the agent (out of view) in the rest. Each scene was evaluated with each brain's own critic weights, `V = value_bias + value_weights · encoded`, 640 scenes per brain. A meal is worth about 0.12 reward. As a ceiling, a ridge regression tested whether food distance is linearly present in the encoded state (held-out R²).

| Brains | Mean V | SD of V across scenes | V(food near, centered) − V(food behind) | corr(V, distance) | Distance R² from encoded |
|---|---|---|---|---|---|
| fresh (critic weights start at 0) | 0 | 0 | 0 | — | 0.86 |
| free run 100k ticks, seed 5 | −0.151 | 0.0004 | −0.0004 meals | +0.15 | 0.85 |
| free run 100k ticks, seed 6 | −0.167 | 0.0005 | −0.0021 meals | +0.30 | 0.85 |
| standard probe, 240 episodes | −0.002 | 0.0003 | +0.0028 meals | −0.49 | 0.82 |

No brain valued nearby, centered food more than 0.007 of a meal above food behind it. The free-run critics even lean the wrong way: value rises with distance.

The mean value is the resting-drain baseline and nothing else. Food distance is linearly available in the encoded state (R² ≈ 0.85), so a linear critic could represent it, but the trained critic is flat. Between meals, then, δ carries no information about approaching food, and the turn weights only ever receive the sparse meal-time credit. The critic's vector step is `CRITIC_LEARNING_RATE × TD_VECTOR_SCALE` = 0.01 / 128 per dimension, and a free-running agent eats about once every 5k ticks. That combination leaves the value weights almost untouched by food within a lifetime.

## Normalized critic step

The critic now takes a normalized-LMS step, `CRITIC_LEARNING_RATE / (1 + ‖x‖²)`, where x is the encoded state whose value δ corrects. It used to take a fixed `CRITIC_LEARNING_RATE / 128` per dimension. At the measured encodings (‖x‖² ≈ 2–35), that makes the critic about 4–45× faster.

**Critic value over food position.** Same test as above:

| Brains | SD of V across scenes | V(food near, centered) − V(food behind) | corr(V, distance) |
|---|---|---|---|
| free run, seed 5 | 0.0180 (was 0.0004) | +0.026 meals (was −0.0004) | +0.15 (was +0.15) |
| free run, seed 6 | 0.0090 (was 0.0005) | −0.009 meals (was −0.0021) | +0.23 (was +0.30) |
| standard probe, 240 episodes | 0.0007 (was 0.0003) | +0.007 meals (was +0.0028) | −0.49 (was −0.49) |

**Free runs.** Default 8×6 eye, 10 agents, 100k ticks, seeds 5–10, paired against the previous critic:

- Food/agent: 19.68 vs 18.75 (+0.93, paired t = 1.9).
- Deaths/agent: 17.1 vs 17.6.
- Approach intent: 0.490 vs 0.496.

**Probes.** Seeds 17 / 23:

| Probe | Success | Turn alignment |
|---|---|---|
| Standard | 0.530 → 0.541 / 0.536 → 0.542 | 0.462 / 0.494 (was 0.398 / 0.462) |
| Steering-required | 0.023 → 0.092 / 0.023 → 0.067 (was 0.131 / 0.106) | 0.486 / 0.478 |

The critic now moves 25–50× more across scenes, but not along the food. Nearby, centered food is still worth at most a few percent of a meal more than food behind the agent, and in free-run brains value still rises with distance.

Food per agent recovers to the level before the episodic-memory change. Steering does not appear: approach intent and alignment stay at chance.

A faster critic is therefore not enough. Raw encodings of different scenes are 98–99% alike by cosine, so each update moves the value of nearly every scene together. The part of an update that separates "food near" from "food behind" is a few percent of it.

## Centered critic input

The critic now reads the centered encoding, `encoded − O_ENCODED_MEAN`, for its value, its trace and its normalized step. The bias carries the baseline, and the weights can only learn what separates scenes.

**Critic value over food position.** Same test as above. Each cell shows the centered critic, then the uncentered normalized critic, then the original critic:

| Brains | V(food near, centered) − V(food behind) | Brains with that difference > 0 | corr(V, distance) |
|---|---|---|---|
| standard probe, 240 episodes | +0.044 / +0.007 / +0.003 meals | 14 of 16 | −0.40 / −0.49 / −0.49 (all 16 negative) |
| free run, seed 5 | −0.005 / +0.026 / −0.0004 meals | 5 of 10 | +0.11 / +0.15 / +0.15 |
| free run, seed 6 | +0.043 / −0.009 / −0.002 meals | 6 of 10 | −0.05 / +0.23 / +0.30 |

**Free runs** (seeds 5–10):

- Food/agent: 19.38, against 19.68 for the uncentered normalized critic.
- Deaths/agent: 17.7, against 17.1.
- Approach intent: 0.490.

With food as the only thing that varies (the probe), centering raises the value of nearby, centered food about sixfold, to about 4% of a meal. In free runs the per-brain differences grow (−0.17 to +0.15 meals) but their sign is a coin flip. The free-run critic still does not consistently value approaching food: its value spreads over scenes along something other than the food.

## Can the turn learner learn steering from a clean signal?

This was a CPU replay of the brain's turn learner, run in scratch code outside the repo. It uses the exact per-brain-tick order and constants from `coop_predict_and_act`:

- the centered, normalized TD critic;
- trace-weighted updates, `AWLR · ACTOR_VECTOR_SCALE · δ · z` for the weights and `AWLR · δ · z` for the bias;
- the L2 ball;
- a tanh policy with uniform exploration noise;
- the traces updated with `noise · encoded`;
- the running mean updated as in pass 7g.

The inputs were real encodings of food at 29 bearings × 3 distances, dumped from 12 brains: 4 fresh, 4 trained by a 100k-tick free run, and 4 trained by the standard probe. Each brain started from its own weights and learned for 20k brain ticks on two tasks:

- **Bandit**: each tick shows a random in-view scene. The next tick's reward is `0.12 × executed turn × side of the food`. This is the easiest possible steering problem: immediate, dense, and linearly separable.
- **Steering**: turning rotates the food's bearing (0.1 rad per tick at full turn). The only reward is 0.12 on centering the food (|bearing| < 0.05); an episode times out after 30 ticks.

Scores below are the alignment of the policy's own turn sign with the food's side over all scenes.

| Arm | Bandit alignment (fresh / free-run / probe brains) | Steering alignment | Steering success (first → last fifth) |
|---|---|---|---|
| exact rule | 0.51 / 0.57 / 0.56 | 0.49 / 0.49 / 0.43 | 0.00→0.15 / 0.38→0.48 / 0.50→0.51 |
| actor input centered | 0.51 / 0.65 / 0.57 | 0.51 / 0.49 / 0.55 | 0.01→0.00 / 0.41→0.49 / 0.14→0.36 |
| `ACTOR_VECTOR_SCALE` 1 (was 1/16) | 0.54 / 0.72 / 0.56 | 0.49 / 0.49 / 0.55 | 0.04→0.26 / 0.49→0.50 / 0.49→0.50 |
| **centered + scale 1** | **0.98 / 0.99 / 0.66** | 0.51 / 0.49 / 0.55 | 0.00→0.01 / 0.38→0.51 / 0.16→0.30 |
| reward × 8.3 (1.0 per unit) | 0.51 / 0.49 / 0.56 | 0.49 / 0.49 / 0.43 | 0.08→0.25 / 0.48→0.52 / 0.51→0.50 |
| no L2 ball | 0.62 / 0.60 / 0.57 | 0.50 / 0.49 / 0.43 | 0.02→0.26 / 0.42→0.49 / 0.48→0.50 |

**The exact rule cannot learn even the bandit.** The side signal in the encodings is small next to what every scene shares: the left/right half-difference has norm 0.23–0.28, against a scene mean of norm 2.5–4.2 (0.04–0.11 for probe-trained encoders, whose shared component grew to 10–11). An uncentered update is dominated by that shared component and by the turn bias. The bias steps 16× faster than the per-dimension weights, so learning goes into a side-blind constant turn. The per-dimension step, `0.10 / 16`, then moves the side direction too slowly to matter within a lifetime.

**Two changes together make the bandit learnable** (alignment 0.98–0.99 for fresh and free-run encoders): centering the actor's input as the critic's already is, and an actor step 16× larger. Neither is enough alone, and a larger reward or removing the L2 ball does not help. Probe-trained encoders reach only 0.66: there the side signal is under 1% of the shared component, and centering does not recover it.

**No arm learns the sequential task in 20k ticks.** With the reward only on centering, exploration noise almost never centers the food by chance. Fresh brains succeed in 0–1% of episodes. The one strategy that pays off is a constant rotation, which the bias learns and which sweeps through the food in about half the episodes. That is the "saturated constant rotation" of finding F5, now reproduced without any reset artifacts.

Fixing the update rule is therefore necessary but not sufficient. The turn learner also needs a between-meals signal that grows as the food gets centered, which is the critic's job — and in free runs the critic does not yet learn that.

## Centered, normalized turn channel

The turn policy and its trace now read the centered encoding, and its weights and bias take the critic's normalized step at the actor's rate, `ACTION_WEIGHT_LEARNING_RATE / (1 + ‖x‖²)`. In the replay above, this variant reached bandit alignment 0.99 / 1.00 / 0.69 for fresh / free-run / probe-trained encoders, against 0.98 / 0.99 / 0.66 with a fixed 16× step.

**Free runs** (seeds 5–10), against the centered-critic code:

- Food/agent: 19.88 vs 19.38 (paired +0.5; the per-seed differences range from −2.1 to +3.1).
- Approach intent: 0.493–0.503.

**Probes** (seeds 17 / 23):

| Probe | Success, fix | Success, before | Alignment, fix | Alignment, before |
|---|---|---|---|---|
| Standard | 0.538 → 0.523 / 0.522 → 0.542 | 0.527 → 0.525 / 0.534 → 0.480 | 0.463 / 0.458 | 0.448 / 0.431 |
| Steering-required | 0.020 → 0.038 / 0.022 → 0.030 | 0.023 → 0.130 / 0.023 → 0.159 | 0.498 / 0.496 | 0.457 / 0.476 |

**Headless evolution.** Release builds, `evo_default` config, 20 generations, seeds 5 and 6. Lifetime turn learning is frozen during evolution, so only the centered turn input takes part.

| Run | Mean fitness | Food/agent | Fitness, first 5 → last 5 generations |
|---|---|---|---|
| before, seed 5 | 0.0315 | 8.1 | 0.0348 → 0.0333 |
| fix, seed 5 | 0.0268 | 7.3 | 0.0293 → 0.0258 |
| before, seed 6 | 0.0244 | 6.7 | 0.0257 → 0.0235 |
| fix, seed 6 | 0.0293 | 8.0 | 0.0303 → 0.0294 |

The fix is neutral in free runs and in evolution, and steering still does not appear. Steering-required success falls because the turn weights can no longer absorb the constant spin that used to sweep through the food there.

This is the replay's prediction for the sequential task: once the rule is able to learn, the missing piece is a signal between meals that grows as the food gets centered. Exploration alone almost never finds the reward, so a lifetime yields too few meals to learn steering from.

## Faster critic or persistent exploration in the sequential task

Both candidates for a between-meals signal were tested in the replay's sequential task (reward only on centering the food), starting from the landed rule: centered turn input, normalized turn step. Each run was 20k brain ticks. The persistent noise is an AR(1) process with the same marginal variance as the independent uniform draw.

| Arm | Steering alignment (fresh / free-run / probe brains) | Side component | \|Common turn\| | Success, first → last fifth | V(centered) − V(off-center) |
|---|---|---|---|---|---|
| landed rule | 0.49 / 0.49 / 0.55 | +0.00 / −0.01 / +0.21 | 0.01 / 0.54 / 0.56 | 0.00→0.00 / 0.34→0.50 / 0.14→0.24 | +0.00 / +0.71 / +0.06 meals |
| `CRITIC_LEARNING_RATE` 0.1 | 0.49 / 0.49 / 0.55 | −0.00 / +0.00 / +0.19 | 0.00 / 0.90 / 0.88 | 0.00→0.00 / 0.45→0.51 / 0.24→0.51 | +0.00 / +0.55 / +0.30 |
| noise persistence ρ = 0.8 | 0.51 / 0.49 / 0.55 | +0.03 / −0.01 / +0.21 | 0.99 / 1.27 / 1.28 | 0.34→0.48 / 0.46→0.52 / 0.41→0.50 | +0.63 / +0.70 / +0.17 |
| noise persistence ρ = 0.9 | 0.50 / 0.49 / 0.55 | +0.02 / −0.01 / +0.22 | 1.34 / 1.51 / 1.39 | 0.45→0.49 / 0.48→0.52 / 0.44→0.47 | +0.63 / +0.68 / +0.17 |
| both | 0.51 / 0.49 / 0.56 | +0.06 / +0.02 / +0.22 | 1.46 / 1.60 / 1.61 | 0.46→0.49 / 0.49→0.50 / 0.48→0.49 | +0.46 / +0.57 / +0.10 |

Neither change makes the task learnable, so neither was ported.

- **Persistent noise lets fresh brains find the reward** (0.34–0.46 success from the start, where independent noise gets 0), and their critic learns that centered food is worth about two thirds of a meal more than off-center food.
- **A faster critic** strengthens that value gradient for probe-trained brains (+0.30 meals).
- **Every arm still converges to the constant spin.** The common turn grows to 1–1.6, the side component stays near zero, and success settles at the spin's ceiling of about 0.5.

So even with a value gradient between meals, the turn learner falls into the spin attractor. A saturated constant turn clips the exploration kicks that push further along the spin while still crediting them, and the turn bias carries the spin directly. The next candidates are inside the actor: credit the kick that was actually expressed after clipping, and stop the turn bias learning a spin.

## Bias-free turn channel with persistent exploration

The replay tested the two actor-internal candidates, crediting the kick that was actually expressed and stopping the turn bias, each with persistent noise (ρ = 0.9). The table shows sequential-task alignment over 20k brain ticks:

| Arm | Fresh | Free-run | Probe-trained | Success, first → last fifth |
|---|---|---|---|---|
| persistent noise (the previous table) | 0.50 | 0.49 | 0.55 | ≈ 0.45 → 0.50 (spin) |
| + expressed kick | 0.50 | 0.49 | 0.55 | ≈ 0.46 → 0.51 (spin) |
| + turn bias frozen at its starting value | 0.95 | 0.66 | 0.55 | 0.12–0.27 → 0.14–0.36 |
| **+ turn bias fixed at 0** | **0.96** | **0.95** | **0.81** | 0.07–0.13 → 0.09–0.15 |
| + expressed kick + bias 0 | 0.95 | 0.97 | 0.84 | 0.09–0.13 → 0.08–0.21 |
| bias 0, independent noise | 0.61 | 0.57 | 0.55 | 0.00 → 0.00 |

- **The turn bias is what carries the spin.** A frozen bias keeps whatever spin a trained brain already had. Without persistent noise the reward is never found.
- **The ported version:** the turn channel has no bias. The policy ignores the slot, TD never updates it, and evolution no longer perturbs it. The turn noise is an AR(1) process, ρ = 0.9, with the variance of one uniform draw.
- **What the replay's success rests on:** its task rewards centering the food directly. That is a test-bench stand-in for checking whether the rule can learn, not a signal the agent has.

**In the simulator.** Free runs (seeds 5–10):

- Food/agent: 20.48 vs 19.88 (paired +0.6).
- Deaths/agent: 17.9 vs 18.2.

Headless evolution (20 generations, seeds 5 and 6): mean fitness 0.0263 / 0.0264, against 0.0268 / 0.0293 for the previous turn channel. Fitness is flat across generations. The probes' executed-turn alignment is no longer interpretable. With persistent noise, an agent rotating in place spends more scored ticks past the food than approaching it, which alone pushes the score below 0.5 (0.41 measured). Steering-required success of 0.21–0.24 from the first episode is noise finding food, not learning.

The learned policy itself (sign of `w · (encoded − mean)` over the scene grid) stays at chance:

| Brains | Before the turn fix | Centered, normalized turn | Bias-free, persistent noise |
|---|---|---|---|
| free run, 100k ticks | 0.490 | 0.499 | 0.494 |
| standard probe, 240 episodes | 0.428 | 0.461 | 0.466 (per brain 0.32–0.73) |

In the simulator the only signal is the agent's own energy gain when it eats, and the free-run critic has not learned that food ahead predicts one. The rule can now learn steering, but the agent's experience does not yet teach it.

## Does the agent's own experience contain the lesson?

This test only measured, using ground truth from the physics state (nearest food distance and bearing each sample). It ran on the current code: free runs of 10 agents × 100k ticks, seeds 5–7, with the default perception (a new visual frame every `vision_stride × brain_tick_stride` = 100 physics ticks = 10 brain ticks) and with a fresh frame every brain tick (`vision_stride` 1).

Categories:

- **near ahead**: food < 3 units away, |bearing| < 0.4;
- **ahead**: food 3–5.5 units away, |bearing| < 0.4;
- **in view off-axis**: food < 5.5 units away, 0.4 ≤ |bearing| < π/4;
- **not in view**: everything else.

"Value" is the discounted next meal, E[γ^(brain ticks to next meal)] with the critic's γ = 0.97, in meals.

| Regime (seeds 5 / 6 / 7) | Meals per agent | Share of ticks with food in view | P(meal within 10 brain ticks): ahead / off-axis / not in view | Value, meals: ahead / off-axis / not in view |
|---|---|---|---|---|
| default perception | 19.5 / 17.0 / 15.2 | 1.2–1.5% | 0.46–0.59 / 0.23–0.36 / 0.011–0.013 | 0.45–0.61 / 0.29–0.41 / 0.04–0.05 |
| frame every brain tick | 35.8 / 37.8 / 28.9 | 2.3–3.1% | 0.54–0.74 / 0.35–0.41 / 0.016–0.020 | 0.55–0.72 / 0.40–0.46 / 0.07–0.09 |

(Near-ahead food behaves like ahead: 0.43–0.74 meals.)

- **The lesson is in the experience, and it is large.** Food ahead is worth about half a meal more than no food in view, and 0.15–0.3 of a meal more than the same food off-axis — exactly the gradient a turn toward food would climb. The trained critics measured above value nearby, centered food at most 0.04 of a meal above food behind. They have learned less than a tenth of what the agent's own experience teaches.
- **The lesson is rare.** Food is in view on only 1–3% of ticks.
- **Perception rate matters by itself.** With a fresh frame every brain tick, the same learner eats 1.9× as many meals with no other change. Under the default strides the agent walks about 9 units (more than the ~5.5-unit food visibility range) between frames. Only 27–37% of meals are preceded by food in view on the previous frame, against 75–79% with fresh frames. Headless evolution and free runs use the default strides.

## Fresh perception by default

`vision_stride` now defaults to 1, so the agent gets a fresh frame every brain tick. The same knob also runs the other global passes (grid rebuild, food respawn, collisions) every brain tick.

**Throughput** (10 agents, `--bench`, 30k ticks):

| `vision_stride` | Ticks/s | Change |
|---|---|---|
| 10 | 6458 | — |
| 2 | 5666 | −12% |
| 1 | 5021 | −22% |

**Headless evolution.** Release builds, `evo_default` config with only `vision_stride` changed, 20 generations; the same code at `vision_stride` 10 is the baseline.

| Run | Mean fitness | Food/agent | Deaths/agent | Best | Fitness, first 5 → last 5 generations |
|---|---|---|---|---|---|
| stride 10, seed 5 | 0.0263 | 6.9 | 6.2 | 0.0342 | 0.0267 → 0.0265 |
| **stride 1, seed 5** | **0.0390** | **11.5** | 5.3 | 0.0531 | 0.0417 → 0.0400 |
| stride 10, seed 6 | 0.0264 | 7.3 | 6.3 | 0.0334 | 0.0272 → 0.0259 |
| **stride 1, seed 6** | **0.0540** | **22.7** | 8.9 | 0.0642 | 0.0566 → 0.0549 |

Mean fitness rises 76% (0.0465 vs 0.0264) and food per agent 1.6–3.1×. The gain is immediate rather than evolved: fitness is flat across generations in every run. This matches the free-run measurement above (meals 1.9× with fresh frames), where the learned turn policy stays at chance. The better foraging comes from acting on current rather than 3-second-old perception, not from learned steering.

## Critic replay of remembered outcomes

Re-measured under fresh perception, the critic valued nearby, centered food only 0.05 / 0.07 of a meal above food behind (free-run seeds 5 / 6). The agent's experience holds about half a meal, and the learned turn policy stayed at chance (0.49).

Each remembered moment now keeps a return: the reward the critic sees (`raw_gradient × (1 + urgency)`) from salient changes that followed within the credit window, discounted by γ per brain tick. Every brain tick one settled memory slot, chosen by hash, steps the critic's weights toward that return, `w += 0.1 / (1 + ‖key‖²) · (return − w·key) · key`. The bias keeps carrying the baseline. Only the agent's own homeostatic outcomes enter; nothing refers to food.

**Critic value over food position** (the same test as above):

| Brains | V(near, centered) − V(behind), before replay | With replay | Brains positive |
|---|---|---|---|
| free run, seed 5 | +0.050 meals | +0.321 meals | 5 of 10 |
| free run, seed 6 | +0.072 | +0.216 | 7 of 10 |
| standard probe | +0.034 | +0.172 | 14 of 16 |

**Learned turn policy** (sign over the scene grid): 0.470 free-run and 0.481 probe-trained, against 0.492 / 0.466 before — still chance.

**Free runs.** Default config (fresh perception), 10 agents, 100k ticks, seeds 5–10:

| | Before | With replay |
|---|---|---|
| Food/agent | 36.4 | 59.9 (paired +23.6, t = 14.1) |
| Distance/agent | 7.1–7.5k | 12.7–15.2k |
| Deaths/agent | 15.8 | 27.6 |
| Food per quarter of the run | falls, q1 112–156 → q4 50–87 | holds or rises, q1 113–185 → q4 124–202 |
| Approach intent | 0.495 | 0.495 |

**Headless evolution** (`evo_default` with fresh perception, 20 generations):

| Seed | Mean fitness, before → replay | Food/agent | Deaths/agent |
|---|---|---|---|
| 5 | 0.0390 → 0.0561 | 11.5 → 30.8 | 5.3 → 18.6 |
| 6 | 0.0540 → 0.0443 | 22.7 → 15.8 | 8.9 → 7.0 |

Mean fitness is 0.0502 against 0.0465.

Replay makes the critic learn a large share of the food-ahead lesson (0.2–0.3 of the ~0.5-meal gap on average), though brain-to-brain spread is wide. It also changes behavior. Agents keep moving instead of slowing over a life, and foraging no longer declines within a lifetime. In the free runs they cover about twice the ground, eat 65% more, and die 75% more often. Steering still does not appear: approach intent and the learned turn policy stay at chance.

## Does the critic's value rise as food moves toward the centre of view?

Steering needs the value to rise as the food's bearing narrows at the same distance, not only when food is present. From the same scene dumps (29 bearings × 3 distances, 10 brains per group, each brain's own critic), in meals:

| Brains | V(centred, \|b\| < 0.15) − V(off-axis, \|b\| > 0.45), same distance | Brains positive | dV/d\|bearing\| | \|V(food left) − V(food right)\| |
|---|---|---|---|---|
| free run, before replay | +0.05 | 8/10 | −0.10 per rad | 0.09 |
| free run, with replay | +0.03 (per brain −3.87 … +1.64) | 5/10 | −0.11 per rad | 2.30 |
| probe, before replay | −0.03 | 0/10 | +0.06 per rad | 0.02 |
| probe, with replay | −0.14 | 2/10 | +0.18 per rad | 0.09 |

The steering gradient is not there. With replay, free-run critics value "food present" (near vs behind: +0.2–0.3 meals). But their value of the same food at the same distance swings with which side it is on, by 2.3 meals on average and by up to ~4 meals in single brains, while centred and off-axis food come out even on average.

Those magnitudes are far beyond any real return (a meal is 1), so the replayed weights extrapolate wildly on these flat-ground probe scenes. Replay of a few remembered moments at rate 0.1 fits the directions those moments happen to share, including a side, rather than learning centring. A turn learner climbing such a value would learn a side preference, not steering.

## The critic's value of food in the world itself

The flat-ground probe scenes are unlike anything a free-run brain has seen. So this test read each free-running agent's own critic value (`O_PREV_VALUE`) on the ticks where food was in view, paired with the true bearing and distance of the food in the frame it perceived. Each value is compared with the same agent's value when no food was in view, in the same 10k-tick window.

Setup: default config (fresh perception), 10 agents, 100k ticks, seeds 5–7. Units are meals.

| Arm, half of run | V(in view) − V(not in view) | V(centred) − V(off-axis), same distance bins | \|V(left) − V(right)\| |
|---|---|---|---|
| before replay, first half | +0.10 / +0.59 / +0.41 | −0.03 / +0.27 / +0.22 | 0.10 / 0.16 / 0.13 |
| before replay, second half | +0.12 / +0.38 / +0.46 | −0.35 / +0.43 / +0.10 | 0.19 / 0.03 / 0.27 |
| with replay, first half | +0.97 / +1.33 / +0.55 | +0.92 / −0.02 / +0.02 | 0.63 / 0.38 / 0.31 |
| with replay, second half | −0.00 / +0.44 / +0.19 | −0.22 / +0.38 / +0.15 | 0.25 / 0.20 / 1.44 |

- **In the world, the critic already values food in view** (+0.1 to +0.6 meals before replay), well above what the probe scenes showed. Replay raises this further early in a life.
- **Whether it prefers centred food cannot be resolved at this sample size.** Only 135–275 in-view samples fell in each half-run, so the distance bins hold 8–48 samples and swing by ±0.5 meals. The sign is inconsistent in both arms.
- **Replay's left/right swings appear in the world too:** 0.20–1.44 meals, against 0.03–0.27 before.

### Ten seeds, and a gentler replay

The same in-world measurement over seeds 5–14 (whole runs, about 470–545 in-view samples per seed). A third arm replays at `CRITIC_REPLAY_RATE` 0.02 instead of 0.1. Values are in meals, shown as mean ± standard error over seeds:

| Arm | V(in view) − V(not in view) | V(centred) − V(off-axis), same distance | \|V(left) − V(right)\| |
|---|---|---|---|
| no replay | +0.27 ± 0.05 | −0.03 ± 0.04 (4/10 positive) | 0.14 ± 0.02 |
| replay 0.1 (on develop) | +0.26 ± 0.28 | +0.14 ± 0.19 (6/10) | 0.70 ± 0.15 |
| replay 0.02 | +0.23 ± 0.48 | −0.60 ± 0.51 (3/10) | 1.24 ± 0.32 |

Free runs, seeds 5–10:

| Arm | Food/agent | Deaths/agent | Distance/agent |
|---|---|---|---|
| no replay | 36.4 | 15.8 | 7.3k |
| replay 0.1 | 59.9 | 27.6 | 13.9k |
| replay 0.02 | 57.6 | 23.8 | 11.9k |

- **Without replay, the in-world critic already values food in view** (+0.27 ± 0.05 meals), but not centred food over off-axis food (−0.03 ± 0.04). The centring gradient that steering needs is absent.
- **Replay at either rate leaves that gradient absent** while making the critic's value 6–10× noisier across seeds and inflating left/right differences 5–9×. The gentler rate does not reduce the swings.
- **Replay's foraging gain works through movement, not a better critic.** Agents keep moving (1.6–1.9× the distance), eat more, and die more.

## Is it data starvation? One life ten times longer

The same in-world measurement over a single life of 1M ticks: 10 agents, default config, seeds 5–7, with and without the critic's replay. It reports per 100k-tick window, and also the in-world turn policy: how often the policy's own turn before noise, `w_turn · (encoded − mean)`, points toward food in view with |bearing| > 0.1. Values are means over the three seeds.

| Window (100k ticks) | No replay: V(in view) − V(not) | centred − off | \|left − right\| | policy toward food | Replay: V(in view) − V(not) | centred − off | \|left − right\| | policy toward food |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.39 | +0.12 | 0.26 | 0.531 | +0.32 | +0.98 | 0.91 | 0.524 |
| 1 | +0.28 | −0.20 | 0.18 | 0.518 | −2.46 | +1.65 | 3.71 | 0.495 |
| 2 | +0.10 | +0.47 | 0.55 | 0.489 | +0.39 | −3.01 | 1.98 | 0.501 |
| 3 | +0.79 | +0.33 | 4.43 | 0.493 | +0.59 | +1.45 | 1.42 | 0.499 |
| 4 | −0.04 | −2.90 | 0.85 | 0.510 | −0.12 | +0.42 | 0.67 | 0.491 |
| 5 | +0.78 | +0.02 | 0.68 | 0.503 | −0.59 | +0.48 | 2.54 | 0.501 |
| 6 | −0.29 | +0.04 | 0.60 | 0.503 | −0.08 | +1.56 | 1.16 | 0.495 |
| 7 | +0.31 | −0.46 | 0.40 | 0.512 | −1.12 | −1.31 | 1.09 | 0.513 |
| 8 | +0.44 | +0.26 | 1.16 | 0.505 | +0.46 | +0.26 | 0.73 | 0.500 |
| 9 | +0.20 | −0.66 | 1.19 | 0.494 | +1.02 | −0.26 | 0.49 | 0.497 |

**Ten times the experience does not help.**

- The turn policy points toward food 49–53% of the time in every window of both arms. Its best value, 0.52–0.53, came in the first window and then settled at chance.
- The critic never settles on a centring preference; its sign flips window to window.
- Without replay, the critic's value grows noisier as the life goes on. Left/right differences reach 0.4–4.4 meals after the first 200k ticks, against 0.18–0.26 before, and the value of food in view even turns negative in some windows.

So this is not data starvation. The critic gets less reliable with more experience, which points at something underneath it drifting. A likely suspect is its input: the encoder keeps learning while the critic fits it. That remains a hypothesis to test.

## Does encoder drift make the critic noisy?

A scratch build (not committed) repeated the 1M-tick life on the current code without replay, seeds 5–7. At tick 200k it could freeze the encoder's weights, the only weights the encoder credit step changes, alone or together with the running mean the critic's input is centred by. Drift is the relative change of each over a 100k-tick window (‖Δ‖/‖·‖).

| Arm | Encoder change per window (from 200k on) | V(centred) − V(off-axis), windows 2–9 | \|V(left) − V(right)\|, windows 2–9 | Policy toward food |
|---|---|---|---|---|
| no freeze | 1.44 → 0.44–0.70 | −0.70 … +3.52 | 0.69 – 3.36 | 0.48 – 0.51 |
| encoder frozen | 0 | −0.50 … +2.11 | 0.59 – 2.04 | 0.49 – 0.53 |
| encoder + mean frozen | 0 (mean change 0) | −9.12 … +7.45 | 0.27 – 6.73 | 0.48 – 0.53 |

- **The encoder does rewrite itself heavily.** Its weights change by 124–146% per 100k ticks in the first 400k ticks and by 44–70% per window afterwards. The earlier readout shows the food's side survives that.
- **But freezing it does not calm the critic.** With the encoder and the mean both frozen from tick 200k, its value of food in view still swings by up to ±9 meals between windows. It still shows no stable centring preference, and the turn policy stays at chance.

So drift is not the cause. What is left is the critic itself. Scenes with food in view make up only about 1–3% of its data and sit far from the average scene it learns from. Their values are barely constrained, so they swing with every update fitted to common scenes. Meanwhile the weight norm grows from 0.25 to 1.0–1.3 over the life, which amplifies the swings. The lesson is in the experience, but this linear critic cannot hold it for rare scenes.

## Would an episodic value hold what the linear critic cannot?

This was an offline check in the world, not a shader change. On the same in-view and baseline ticks as the in-world measurement (current code, seeds 5–14, 100k ticks each), it computed the similarity-weighted remembered return of the 16 settled memories most similar to the agent's current memory key. Moments still inside the credit window were excluded. Values are in meals, shown as mean ± standard error over seeds:

| Value | V(in view) − V(not in view) | V(centred) − V(off-axis), same distance | \|V(left) − V(right)\| |
|---|---|---|---|
| linear critic, whole run | +0.48 ± 0.26 | +0.13 ± 0.15 (6/10 positive) | 0.58 ± 0.10 |
| linear critic, second half | +0.76 ± 0.35 | −0.24 ± 0.62 (5/10) | 1.49 ± 0.51 |
| episodic value, whole run | +0.22 ± 0.05 | +0.05 ± 0.05 (5/10) | 0.22 ± 0.04 |
| episodic value, second half | +0.14 ± 0.05 | +0.01 ± 0.06 (4/10) | 0.32 ± 0.07 |

- **The episodic value is what the linear critic is not: stable.** It values food in view consistently (+0.22 ± 0.05 meals, positive in every seed) with a fifth to a sixth of the linear critic's seed-to-seed error, and its left/right differences stay small.
- **It carries no centring gradient either:** +0.05 ± 0.05, half the seeds each way. Recall by whole-scene similarity pools centred and off-axis moments together. Each moment's return is also cut at the 8-tick credit window, so the difference between "reached food in 3 ticks" and "in 10" is mostly lost.

Porting it would give the agent a reliable sense that food is in view, but still no gradient for turning toward it.

## Is it rarity? Food five and ten times as common

The same 1M-tick in-world measurement on current code (with the critic's replay), seeds 5–7, with `food_density` multiplied by 5 and by 10. In-view reads were subsampled 1-in-5 and 1-in-10 to keep the run cheap.

| Window (100k ticks) | 5× food: meals/agent | food in view | V(in view) − V(not) | centred − off | policy toward food | 10× food: meals/agent | food in view | V(in view) − V(not) | centred − off | policy toward food |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 293 | 8.0% | −0.40 | −1.95 | 0.49 | 535 | 11.0% | +0.01 | −1.55 | 0.47 |
| 2 | 341 | 7.4% | −0.80 | −0.11 | 0.47 | 692 | 10.6% | −0.40 | −1.67 | 0.48 |
| 4 | 372 | 7.8% | −2.36 | −0.41 | 0.50 | 692 | 10.7% | +0.18 | −0.20 | 0.49 |
| 6 | 370 | 7.7% | −0.22 | +0.99 | 0.50 | 693 | 10.3% | −0.03 | +0.10 | 0.49 |
| 8 | 377 | 7.6% | −15.23 | −6.12 | 0.49 | 723 | 10.4% | −0.56 | −0.30 | 0.49 |
| 9 | 376 | 7.3% | −7.63 | −10.25 | 0.50 | 703 | 10.1% | +0.15 | +0.55 | 0.50 |

With 10–20× the reward events (meals) of the default world, and food in view on 7–11% of ticks instead of 1–3%:

- **The turn policy stays at chance** (0.46–0.54 per seed) in every window of both worlds.
- **The critic still prefers neither centred nor off-axis food.** At 5× food it even destabilizes: late in the life it values food in view 8–15 meals *below* no food.
- **Meals do rise 28–31% over the life,** but without any steering.

So rarity is not the explanation. Even with plenty of the right data, this TD actor-critic does not learn to turn toward food it can see, and a food-rich early world would not by itself teach steering. The ceiling is the learner.

## Can a linear critic even express "centred is better"?

The side and distance of food were already known to be linearly present in the encoding. A centring gradient needs something else: a V-shaped function of bearing, |bearing|. On the scene dumps (29 bearings × 3 distances per brain, current code), a held-out ridge regression (5-fold, best of three ridge strengths) read each property from the encoding:

| Encoders | \|bearing\| R² | bearing R² | side R² | distance R² |
|---|---|---|---|---|
| fresh (10) | 0.91 (min 0.91) | 0.92 | 0.81 | 0.80 |
| free run, 100k ticks (10) | 0.88 (min 0.78) | 0.89 | 0.76 | 0.75 |
| standard probe (10) | 0.78 (min 0.27) | 0.79 | 0.68 | 0.67 |

How centred the food is can be read linearly from the encoding about as well as its signed bearing. A linear value function can therefore express "centred food is worth more" for fresh and free-run encoders. The function class is not what stops the critic, though a few probe-trained encoders have degraded (R² down to 0.27).

## Implementation or algorithm? A shadow critic on logged experience

A scratch build (not committed) logged, for every agent and every brain tick of a 200k-tick life, the encoding, the running mean, the GPU critic's value and the exact reward, `raw_gradient × (1 + urgency) + β × predicted_gradient`. This was on the pre-replay code, pure TD, seeds 5–7, 30 agents. Offline, from the same logs:

- **Shadow critic:** the GPU critic's TD(λ) update replayed on the CPU from the same starting weights. It uses the centred, normalized step, the traces, the δ clamp, the L2 ball, and death's terminal lesson and reset.
- **Returns:** the actual discounted return after each tick, Σ γᵏ r, cut at death with the −1 terminal.
- **Least-squares fit:** a ridge regression of those returns on the same centred encoding, fitted on the first 70% of each life.

All four were scored on the held-out last 30% of each life (in meals, mean ± standard error over 30 agents):

| Value | V(in view) − V(not) | V(centred) − V(off-axis), same distance | Agents positive |
|---|---|---|---|
| actual returns (ground truth) | +1.41 ± 0.16 | **+0.76 ± 0.35** | 23/30 |
| least-squares fit of returns | +0.54 ± 0.11 | −0.87 ± 0.47 | 11/30 |
| shadow TD critic | +0.53 ± 0.40 | −0.44 ± 0.82 | 10/30 |
| GPU critic | +0.53 ± 0.40 | −0.44 ± 0.82 | 10/30 |

The shadow reproduces the GPU critic exactly: correlation 1.0000, mean difference ≤ 0.0006 meals. **The implementation is faithful.**

The lesson is in the returns. Centred food is followed by 0.76 meals more discounted return than off-axis food at the same distance, in 23 of 30 agents.

But even a direct least-squares fit of the same linear value to those very returns does not carry it to the rest of the life: −0.87, 11 of 30 positive. Its held-out R² for the returns is negative on average (−1.28, best 0.16). So TD is not what loses the lesson. A single fixed linear value over this input cannot hold it across a life, for two likely reasons:

- the fit spends its capacity on the 97–99% of ticks with no food in view;
- the encoder rewrites itself by 50–150% per 100k ticks, so later encodings mean something else.

### Drift or capacity? Two more fits

The same logging on seeds 5–9 (50 agents, 200k ticks). Each fit regresses the actual discounted returns on the centred encoding, and is then scored on how it values centred against off-axis food on held-out ticks, next to what the returns themselves show on those ticks. Two splits:

- **time-ordered:** the first 70% of the life is fitted, the last 30% held out;
- **mixed:** interleaved 500-tick blocks, 70% fitted, all from the same period. 40 agents had enough held-out in-view ticks.

Two fitting sets: all ticks, or only ticks with food in view (about 300–360 per agent), at two ridge strengths. Values in meals:

| Fit | Fitted V(centred) − V(off-axis) | Agents positive | Actual returns on the same held-out ticks |
|---|---|---|---|
| time-ordered, all ticks | −0.08 ± 0.29 | 32/50 | +0.44 ± 0.12 (34/50) |
| mixed, all ticks | +0.00 ± 0.07 | 20/40 | +0.52 ± 0.10 (36/40) |
| time-ordered, in view only, ridge 10 | +0.11 ± 0.03 | 33/50 | +0.44 ± 0.12 |
| mixed, in view only, ridge 10 | +0.02 ± 0.04 | 24/40 | +0.52 ± 0.10 |
| time-ordered, in view only, ridge 100 | −0.00 ± 0.02 | 28/50 | +0.44 ± 0.12 |
| mixed, in view only, ridge 100 | −0.01 ± 0.02 | 22/40 | +0.52 ± 0.10 |

- **Drift is not the cause.** A fit from the same period (mixed split) does no better than one from earlier in the life.
- **Capacity spent on common scenes is not the cause either.** Fitting only in-view ticks recovers at most a fifth of the gap (+0.11 of about +0.5), and nothing at a stronger ridge.

The centring lesson is robust in the returns: 36 of 40 agents on the mixed split. Yet no linear fit on the agent's own experience recovers it, even though centring is linearly readable from the encoding (R² 0.88 above). What differs from that readout is the target. Individual returns are dominated by whether a meal happens to follow, and each agent has only ~300 in-view ticks to find a 0.5-meal signal in 128 dimensions. **The limit is statistical: the signal in a single life's noisy returns is too small to locate in a 128-dimensional linear value.**

## Turn learning unfrozen in evolution

Evolution no longer freezes lifetime turn learning. The steering-genome search stays: repeat-groups still share one perturbation of the champion's turn weights, via the renamed `search_steering_genome` flag. Each agent now keeps learning its turn weights through its life, and the champion read back after a generation carries that learning into the next one. The kernel's freeze switch remains for tests.

**Headless evolution.** Release builds, `evo_default` config with fresh perception, 20 generations, against current develop (frozen):

| Seed | Mean fitness, frozen → unfrozen | Last 5 generations | Food/agent, first 5 → last 5 generations | Deaths/agent |
|---|---|---|---|---|
| 5 | 0.0561 → 0.0595 | 0.0585 → 0.0630 | frozen 27.0 → 31.8; unfrozen 28.2 → 37.8 | 18.6 → 17.1 |
| 6 | 0.0443 → 0.0445 | 0.0441 → 0.0430 | frozen 16.4 → 15.0; unfrozen 22.4 → 21.3 | 7.0 → 13.1 |

On seed 5, food per agent rises by a third across the 20 generations (28 → 38), against 27 → 32 frozen. Seed 6 is flat in both arms.

The stored champions show no steering. A generation's brain is stored only when its node is accepted, so there are only 1–3 per run. On the probe scene grid, their policy points toward the food's side 0.49–0.52 of the time (seed 5, generations 0, 14 and 15). Seed 6's generation-0 champion scores 0.26, turning away with a large turn command. The frozen champions score 0.51–0.66, with near-zero turn output.

Accumulating turn learning along the lineage raises foraging on one of two seeds, but it has not produced steering in 20 generations of 40k ticks.

## Would a smaller input let a linear value keep the lesson?

This tests the statistical diagnosis on the same logging (seeds 5–9, 50 agents, 200k ticks). The actual returns were regressed on the top k principal components of the centred encoding (unsupervised, fitted on the same ticks as the value fit) instead of all 128 dimensions. Values in meals:

| Fit | Time-ordered split: V(centred) − V(off-axis) | Agents positive | Mixed split | Agents positive |
|---|---|---|---|---|
| actual returns (held out) | +0.60 ± 0.12 | 38/50 | +0.54 ± 0.10 | 32/40 |
| top 4 components | +0.06 ± 0.03 | 32/50 | +0.04 ± 0.04 | 24/40 |
| top 8 | +0.06 ± 0.04 | 25/50 | +0.05 ± 0.03 | 24/40 |
| top 16 | +0.14 ± 0.07 | 30/50 | +0.05 ± 0.07 | 24/40 |
| top 32 | +0.30 ± 0.08 | 32/50 | −0.00 ± 0.09 | 26/40 |

- **Unsupervised compression does not isolate the lesson.** The top 4–16 components carry almost none of it (+0.04 to +0.14 of about +0.55). The 32-component result is inconsistent: half the gap on the time-ordered split, none on the mixed split from the same period.
- **This is what rarity predicts.** Food is in view on 1–3% of ticks, so the directions of greatest variance in the encoding describe common scenes (terrain, motion), not where the food is.

A compression that could help would have to be weighted toward the moments that mattered homeostatically, not toward overall variance.

### Compression built from salient moments

This is the same test, except the principal components were computed only over the moments the episodic memory would credit: ticks in the 8-brain-tick window before a salient homeostatic change of either sign (|reward| > 0.05; a meal is about 0.12). That is about 600–750 moments per agent. The returns were still fitted over all ticks. Values in meals:

| Components | Time-ordered: over all ticks | Time-ordered: over salient moments | Mixed: over all ticks | Mixed: over salient moments |
|---|---|---|---|---|
| actual returns (held out) | +0.40 ± 0.13 | +0.40 ± 0.13 | +0.56 ± 0.11 | +0.56 ± 0.11 |
| top 4 | −0.00 ± 0.03 | −0.03 ± 0.03 | +0.03 ± 0.02 | +0.07 ± 0.02 |
| top 8 | +0.01 ± 0.04 | +0.03 ± 0.04 | +0.05 ± 0.03 | +0.04 ± 0.03 |
| top 16 | −0.02 ± 0.07 | −0.03 ± 0.07 | −0.01 ± 0.04 | +0.01 ± 0.05 |
| top 32 | +0.01 ± 0.08 | +0.05 ± 0.09 | +0.14 ± 0.04 | +0.12 ± 0.05 |

- **Weighting the compression toward salient moments does not help.** Fits recover at most about a quarter of the gap (+0.12–0.14 of +0.56) and usually nothing.
- **The earlier +0.30 is not reproduced.** The all-ticks, time-ordered, 32-component fit came out at +0.01 here. The simulator's trajectories differ from run to run on the same seeds, so single-run differences of that size are noise.

The directions along which remembered salient moments vary most are not the direction that separates centred from off-axis food.

## Why the learner does not benefit from more data

A scratch harness (not committed) logged whole free-running lives in three worlds: default food over 1M ticks, 3× food over 1M ticks, and 10× food over 300k ticks. Each world had seeds 5 and 6, 10 agents each, on the pre-replay code. On the same logs it compared:

- the lesson in the actual returns;
- an ideal least-squares linear value, fitted on 70% of 500-tick blocks spread over the whole life, with a variant using a tenth of that data;
- online TD variants replayed from the same starting weights: the exact GPU rule (it matched the GPU to 0.0000 meals), 10× slower and 10× faster steps, TD(0) and λ = 1.

The mixed block split came out degenerate for seed 6 (no held-out blocks), so its mixed-split numbers are excluded. Values are in meals, mean ± standard error:

| World | Food in view | Returns: in view − not | Returns: centred − off | Least squares: in view − not | Least squares: centred − off | Least squares on a tenth of the data: centred − off |
|---|---|---|---|---|---|---|
| default food, 1M | 2.1% | +1.49 ± 0.09 | +0.55 ± 0.11 | +0.10 ± 0.03 | +0.04 ± 0.05 | −0.07 ± 0.04 |
| 3× food, 1M | 5.4% | +1.83 ± 0.07 | +0.57 ± 0.11 | +0.10 ± 0.02 | +0.02 ± 0.02 | −0.05 ± 0.06 |
| 10× food, 300k | 10.5% | +3.04 ± 0.24 | +1.22 ± 0.32 | +0.26 ± 0.07 | −0.00 ± 0.04 | +0.03 ± 0.15 |

- **The lesson grows with food; it does not fade.**
- **The best possible linear value captures under a tenth of even the food-in-view difference, and none of the centring difference. More data does not change that:** ten times the fitting data gives the same result.
- **The online variants are either inert (TD(0), slow steps) or noisy without a consistent centring sign** (the exact rule, fast steps, λ = 1).

So the limit is not sample size, as an earlier section concluded. A follow-up on seed 5 (300k ticks) asked whether food is linearly readable from the critic's input in the world itself, rather than on the flat probe ground (held-out R², fitted on the same spread of blocks):

| World | Food in view | \|Bearing\|, among in-view ticks | Distance, among in-view ticks |
|---|---|---|---|
| probe arena (flat ground, section above) | — | 0.78–0.91 | 0.67–0.80 |
| default food, in the world | 0.06 (−0.09 … 0.28) | 0.24 (0.08 … 0.50) | 0.07 (−0.25 … 0.24) |
| 10× food, in the world | 0.02 | −0.01 | 0.00 |

**In the world, the critic's input hardly encodes where food is.** The in-view label is a physics proxy that terrain can occlude, which costs some R², but not the gap from about 0.9 to about 0.1. Two likely reasons:

- on terrain, the rest of the scene (ground, slope, sky) dominates the encoding;
- the encoder keeps rewriting itself, by 50–150% per 100k ticks, so no single linear map reads food from it across a life.

That is why more time, more food or more agents cannot help. There is nothing linear for the value or the turn policy to latch onto.

## Sensory adaptation: dimming what is always there

In a scratch build (not committed), each vision and depth feature is replaced by its deviation from a slow running mean before the encoder sees it. The non-visual senses are left alone. A steady sky and ground fade, while anything new stands out. Nothing refers to food.

The rate was set per run: off, 0.01 or 0.1 per brain tick (a time constant of about 30 s or about 3 s). Setup: 300k-tick free runs, seeds 5 and 6, 20 agents per arm, pre-replay code. This run also replaced the block-split hash with splitmix64; the old one gave seed 6 no held-out blocks. Readability is held-out R² of a linear readout from the critic's input. Values are in meals.

| Arm | R²: food in view | R²: \|bearing\| (in view) | R²: distance (in view) | Returns: centred − off | Least squares: centred − off (agents positive) | Online TD: centred − off |
|---|---|---|---|---|---|---|
| no adaptation | 0.04 ± 0.04 | 0.27 ± 0.04 | 0.05 ± 0.03 | +0.71 ± 0.18 | +0.11 ± 0.11 (12/20) | +0.06 ± 0.37 |
| adaptation 0.01 | 0.15 ± 0.03 | 0.46 ± 0.02 | 0.11 ± 0.03 | +0.60 ± 0.13 | **+0.27 ± 0.10 (18/20)** | −0.51 ± 1.02 |
| adaptation 0.1 | 0.08 ± 0.02 | 0.37 ± 0.03 | 0.05 ± 0.04 | +0.39 ± 0.12 | +0.26 ± 0.15 (13/20) | +0.29 ± 0.26 |

- **Slow adaptation (0.01) makes food markedly more readable in the world:** its presence 4×, its bearing 1.7×, its distance 2×.
- **With it, the ideal linear value captures about half of the centring lesson with a consistent sign:** +0.27 of +0.60, positive in 18 of 20 agents, against 12 of 20 without.
- **Fast adaptation (0.1) helps less.** It fades food too, and the lesson in the returns shrinks.
- **The online TD critic still does not use it.** Its centring estimate stays noisy, with standard errors of 0.3–1.0 meals.

The representation problem is partly fixable in a natural way. What remains is the critic's noise.

## Sensory adaptation in the learner

Sensory adaptation at 0.01 per brain tick is now part of the learner (`coop_sensory_adapt`). Each visual feature reaches the encoder relative to its own running mean; the mean survives death. Headless evolution, `evo_vs1` config, 20 generations, seeds 5–8. Both arms run the same code apart from adaptation. Generation 0 compares the same fresh brains in the same world, so it measures lifetime learning alone.

| Seed | Gen 0 fitness (off → on) | Gen 0 meals | Mean fitness, all generations | Mean meals | Best score |
|---|---|---|---|---|---|
| 5 | 0.052 → 0.061 | 19 → 24 | 0.0595 → 0.0638 | 32.7 → 34.1 | 0.0658 → 0.0702 |
| 6 | 0.056 → 0.061 | 22 → 25 | 0.0445 → **0.0307** | 21.8 → **10.4** | 0.0564 → 0.0614 |
| 7 | 0.049 → 0.052 | 17 → 19 | 0.0484 → 0.0570 | 23.5 → 29.5 | 0.0540 → 0.0609 |
| 8 | 0.053 → 0.058 | 20 → 23 | 0.0516 → 0.0550 | 25.8 → 22.7 | 0.0584 → 0.0648 |
| mean | 0.0525 → 0.0580 | 19.5 → 22.8 | 0.0510 → 0.0516 | 26.0 → 24.2 | 0.0587 → 0.0643 |

- **Within a lifetime, adaptation helps in every seed:** generation-0 fitness rises about 10% and each agent eats about three more meals.
- **The best score rises in every seed, and mean fitness in three of four.**
- **Seed 6's lineage collapsed after its first champion.** Its descendants moved half as far (about 2,900 against 6,900 units per agent) and ate 9–12 meals instead of about 22. The baseline shows the same drop after inheritance on that seed, only smaller (0.056 → 0.045). Averaged over the four seeds, evolution is a wash.

Adaptation makes the world easier to learn from within one life. Whether a learned brain passes on well depends on the lineage.
