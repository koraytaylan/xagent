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

Sensory adaptation at 0.01 per brain tick is now part of the learner (`coop_sensory_adapt`). Each visual feature reaches the encoder relative to its own running mean; the mean survives death. Headless evolution, `evo_vs1` config, 20 generations, seeds 5–8. Both arms run the same code apart from adaptation. Generation 0 starts from fresh brains in the same world, so it measures lifetime learning alone. The fresh brains are drawn unseeded, so they differ between arms.

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

## Inheriting the birth brain

Until now a champion passed on its end-of-life brain: everything it had learned, its memories included. Its fitness, though, was earned with the brain it was born with. Offspring now inherit that birth brain, mutated as before, and learn again from birth. Headless evolution, same config, 20 generations, seeds 5–8. Both arms have sensory adaptation.

| Seed | Mean fitness (learned → birth brain) | Mean meals | Birth brain: first 5 → last 5 generations | Best score |
|---|---|---|---|---|
| 5 | 0.0638 → 0.0554 | 34.1 → 21.1 | 0.0567 → 0.0557 | 0.0702 → 0.0626 |
| 6 | 0.0307 → **0.0581** | 10.4 → 23.3 | 0.0576 → 0.0601 | 0.0614 → 0.0690 |
| 7 | 0.0570 → 0.0523 | 29.5 → 18.6 | 0.0516 → 0.0528 | 0.0609 → 0.0595 |
| 8 | 0.0550 → 0.0575 | 22.7 → 22.7 | 0.0555 → 0.0574 | 0.0648 → 0.0628 |
| mean | 0.0516 → 0.0558 | 24.2 → 21.4 | 0.0554 → 0.0565 | 0.0643 → 0.0635 |

- **No lineage collapses any more.** Seed 6 goes from 0.031 to 0.058, and every generation stays near the first.
- **Scores stop falling across generations, but they do not rise either.** Evolution now selects only the birth turn weights, and those are still at chance.
- **Passing on learned brains had helped two lineages.** Seeds 5 and 7 ate about ten more meals: those learned brains carried more foraging skill than a new life gains within one generation. The same mechanism passed on the seed-6 collapse, and it is not how nature inherits.

## Evolving the whole birth brain

A scratch build (not committed) let mutation vary the whole birth brain, not just the turn weights. It perturbs:

- the turn weights, as before;
- the forward and value weights, on the same scale as the turn weights;
- the encoder weights and biases, and the predictor weights, as in `mutate_brain_state`.

Headless evolution: same config, 20 generations, seeds 5–8, with 10 or 40 agents per generation (two evaluations per genome).

| Arm | Mean fitness | First 5 → last 5 generations | Mean meals | Best score | Accepted nodes |
|---|---|---|---|---|---|
| turn weights only, 10 agents | 0.0558 | 0.0553 → 0.0565 | 21.4 | 0.0635 | 3.0 |
| whole birth brain, 10 agents | 0.0542 | 0.0542 → 0.0554 | 21.0 | 0.0614 | 3.0 |
| turn weights only, 40 agents | 0.0444 | 0.0445 → 0.0448 | 14.6 | 0.0463 | 1.8 |
| whole birth brain, 40 agents | 0.0434 | 0.0438 → 0.0433 | 15.0 | 0.0450 | 1.0 |

- **Varying the whole birth brain made no difference:** no arm improves across generations.
- **More agents lowers every score.** All agents share one world, so four times the agents means about a third fewer meals each.
- **Evolution is selecting on noise.** The differences between genomes are no larger than the noise of evaluating one genome:

| Arm | SD of one evaluation | SD between genome means | SD expected from noise alone |
|---|---|---|---|
| turn weights only, 10 agents | 0.0102 | 0.0067 | 0.0072 |
| whole birth brain, 10 agents | 0.0117 | 0.0070 | 0.0082 |
| turn weights only, 40 agents | 0.0085 | 0.0060 | 0.0060 |
| whole birth brain, 40 agents | 0.0084 | 0.0069 | 0.0060 |

So the birth brain's heritable differences do not show in fitness: whatever a genome starts with, the life it then lives decides the score.

## An encoder critical period

The encoder keeps rewriting itself through life, so two questions follow. Does that erase the differences between birth brains, and does it keep the critic's input moving under it?

A scratch build (not committed) adds a critical period: encoder plasticity falls as T / (T + age), with age in brain ticks since birth. The setup: 300k-tick free runs, seeds 5–8, 40 agents per arm, current code (sensory adaptation, episodic value replay). The critic is scored on its own GPU values over the last 30% of each life. The earlier CPU replay of the TD rule predates value replay and no longer matches the GPU critic. Values are in meals.

| Arm | Encoder drift per 100k ticks | R²: food in view | R²: \|bearing\| | R²: distance | Returns: in view − not | Least squares: in view − not | Least squares: centred − off (agents positive) | GPU critic, late: in view − not | GPU critic, late: centred − off (agents positive) | Meals |
|---|---|---|---|---|---|---|---|---|---|---|
| no critical period | 0.77–0.89 | 0.13 | 0.29 | 0.06 | +1.6 | +0.49 | +0.29 (30/40) | −1.8 ± 1.6 | −1.4 ± 1.1 (21/40) | 166 ± 13 |
| T = 3000 | 0.10–0.21 | 0.19 | 0.50 | 0.15 | +1.5 | +0.64 | +0.33 (32/40) | +1.2 ± 0.7 | −0.1 ± 0.2 (20/40) | 152 ± 16 |
| T = 300 | 0.01–0.03 | 0.22 | 0.52 | 0.18 | +1.4 | +0.76 | +0.36 (31/40) | **−4.0 ± 1.1** | **−2.7 ± 0.8** (15/40) | 157 ± 15 |

- **A critical period does steady the encoder.** At T = 300 it changes by 1–3% per 100k ticks instead of 77–89%.
- **A steady encoder makes food more readable:** its presence by 1.7×, its bearing by 1.8×, its distance by 3×. The ideal linear value then captures more of the in-view lesson (+0.76 of about +1.4 meals).
- **The online critic still does not use it. With the steadiest encoder it is confidently wrong.** At T = 300 it values food in view 2.9–6.1 meals *below* the rest in every seed, while the actual returns there are 1.1–1.7 meals higher.
- **Evolution is unchanged** (headless, T = 300, seeds 5–8, 10 agents):

| Arm | Mean fitness | First 5 → last 5 generations | SD between genome means | SD expected from noise alone |
|---|---|---|---|---|
| turn weights only, no critical period | 0.0558 | 0.0553 → 0.0565 | 0.0067 | 0.0072 |
| turn weights only, T = 300 | 0.0560 | 0.0557 → 0.0561 | 0.0066 | 0.0070 |
| whole birth brain, no critical period | 0.0542 | 0.0542 → 0.0554 | 0.0070 | 0.0082 |
| whole birth brain, T = 300 | 0.0558 | 0.0561 → 0.0550 | 0.0075 | 0.0076 |

So the encoder's drift is not what keeps the critic from learning. Given a steady input on which food is linearly readable, the online critic learns the wrong sign. The fault lies in the critic's own update, not in its input.

## Which part of the critic's update learns the wrong sign

The scratch build switches off one part of the critic's update at a time, with the steadiest encoder (critical period T = 300) as the test bench. Setup: 300k-tick free runs, seeds 5–8, 40 agents per arm. The actual returns are computed with each arm's own reward, so they are the target that arm's critic should learn. Values are in meals; the critic is scored on its GPU values over the last 30% of each life.

| Arm | Returns: in view − not | Returns: centred − off | GPU critic: in view − not | GPU critic: centred − off (agents positive) | Meals |
|---|---|---|---|---|---|
| as shipped | +1.6 | +0.57 | **−3.5 ± 0.9** | **−2.6 ± 0.6** (14/40) | 166 |
| no episodic value replay | +1.3 | +0.66 | **+0.24 ± 0.04** | +0.07 ± 0.05 (23/40) | 83 |
| no −1 at death | +1.3 | +0.66 | −5.0 ± 1.1 | −4.1 ± 1.1 (12/40) | 177 |
| no urgency weighting of the reward | +0.8 | +0.34 | −1.8 ± 0.4 | −1.3 ± 0.4 (13/40) | 153 |
| none of the three | +0.6 | +0.31 | +0.16 ± 0.01 | +0.04 ± 0.01 (30/40) | 134 |

- **The episodic value replay is what makes the critic learn the wrong sign.** Without it, the critic values food in view above the rest in every seed (+0.17 to +0.32 meals), and does so from early in life. With it, the in-view value is 1.2–4.3 meals below the rest in every seed.
- **Neither the death penalty nor the urgency weighting causes the flip.** Removing the death penalty makes it worse. Removing the urgency weighting only shrinks everything, the true lesson included.
- **Replay still roughly doubles how much the agents eat (166 against 83 meals),** even while it teaches the critic the wrong sign. Its benefit to behaviour comes from something other than a correct value of food.
- **Even without replay the critic captures only a sixth of the in-view lesson** (+0.24 of +1.3) and almost none of the centring lesson (+0.07 of +0.66).

## Why replay teaches the wrong sign

A scratch harness logged whole lives with the steadiest encoder (T = 300; 300k ticks, seeds 5–8, 40 agents). It recorded:

- every salient tick, classified by cause;
- memory snapshots every 30k ticks;
- replay's target for every moment: the discounted salient rewards in the eight brain ticks that follow it.

Memory keys matched the log exactly (cosine 1.000), and the stored returns equalled the recomputed targets. An offline replica of the critic's update ran on each logged life. It tracks the GPU critic: a late in-view gap of −2.4 ± 0.7 meals, against −2.6 ± 0.7 on the GPU. Without the critical period the encoder moves under the logged keys and the replica drifts, so this section uses the T = 300 arm.

**The target is not wrong.**
- Every positive salient tick is a meal (154 per agent). The only negative ones are hazard damage (6 per agent). No sudden energy losses occur near food.
- Memory keeps almost only moments before a meal: 48% of it is food in view (2.4% of lived moments), 92% has a positive return and none a negative one.
- Replay's target values food in view above the rest: +0.83 meals over all moments and +0.16 among remembered ones. The true returns give +1.60 and +0.43.
- A value fitted to replay's target on memory alone would rank food in view correctly on every lived moment (+1.27 meals; the ideal TD value gives +0.89).

**The flip comes from how replay and TD share the critic.** Offline replica, late in-view gap in meals (agents positive):

| Update | In view − not |
|---|---|
| TD only | +0.19 ± 0.17 (33/40) |
| TD + replay, as shipped | −2.37 ± 0.66 (16/40) |
| TD + replay at a tenth of the rate | −2.36 ± 0.68 (19/40) |
| TD + replay that also moves the bias | −0.06 ± 0.35 (29/40) |

Replay moves only the weights, never the bias. It teaches them that nearly every remembered moment is worth about two meals, because its target counts the coming meal but not the steady drain, and memory holds almost nothing else. TD shares those weights and must pull lived moments back to their true value. The tug of war leaves food in view valued below the rest. Slowing replay tenfold changes nothing, so this is where the two updates settle, not a matter of step size. Letting replay move the bias as well removes most of the flip, but it is still no better than TD alone: replay's target still leaves out the drain that TD's target includes.

## Giving replay TD's target

The same offline replica tested replay targets consistent with TD's. The full return is every reward in the eight brain ticks after the moment, the steady drain included, with the −1 at death. The bootstrapped variant adds the critic's value at the window's end, as it was then. Setup: 300k-tick logged lives, steady encoder (T = 300), seeds 5–8, 40 agents; late-life values in meals (agents positive). R² is how much of the variance in the true return the value explains late in life.

| Replay | In view − not | Centred − off | R² vs true return |
|---|---|---|---|
| none (TD only) | **+0.36 ± 0.14** (34/40) | **+0.07 ± 0.12** (24/40) | 0.07 |
| as shipped (salient-only target, weights only) | −2.08 ± 0.89 (23/40) | −2.10 ± 0.61 (12/40) | 0.01 |
| salient-only target, bias too | −0.12 ± 0.41 (30/40) | −0.39 ± 0.29 (18/40) | 0.02 |
| full return, bias too | −0.08 ± 0.40 (30/40) | −0.54 ± 0.28 (17/40) | 0.02 |
| full return + bootstrap, bias too | −0.12 ± 0.41 (29/40) | −0.62 ± 0.28 (16/40) | 0.02 |
| full return + bootstrap, weights only | +0.41 ± 0.19 (33/40) | −0.21 ± 0.24 (20/40) | 0.02 |
| ideal least squares (reference) | +0.82 ± 0.05 (40/40) | +0.41 ± 0.09 (30/40) | 0.26 |

- **A consistent target removes the flip, but every replay variant still leaves the critic worse than TD alone,** most of all on centring.
- **The remaining fault is what memory keeps, not the target.** Eviction keeps moments by their outcome, and 92% of remembered moments preceded a meal. Replay therefore learns what a moment is worth *given that a meal followed*. An off-axis sighting that the agent then turned into a meal looks as good as a centred one, and the in-view advantage shrinks as well. Among remembered moments the true in-view advantage is +0.43 meals; among lived moments it is +1.60. Replaying that sample more faithfully cannot recover a difference the sample no longer contains.

## What replay does to behaviour

A lean behaviour harness compared replay on and off (scratch switch), with and without the critical period. Setup: 300k-tick free runs, seeds 5–8, 40 agents per arm. A sighting is a run of brain ticks with food in view; it converts when a meal follows within 30 brain ticks of its start. "Toward" is the share of in-view ticks on which the turn command points at the food. Values are in meals where marked.

| | Replay | No replay | Replay, T = 300 | No replay, T = 300 |
|---|---|---|---|---|
| Meals | **173 ± 12** | 83 ± 4 | 154 ± 14 | 83 ± 5 |
| Distance travelled | 42,700 | 20,600 | 35,900 | 19,700 |
| Mean speed | 4.26 | 2.06 | 3.59 | 1.97 |
| Mean forward command | 0.126 | 0.034 | 0.117 | 0.041 |
| Forward bias at the end | +0.48 | −0.04 | +0.36 | +0.01 |
| Sightings | 303 | 181 | 265 | 175 |
| Sightings that end in a meal | 45% | 35% | 46% | 37% |
| Brain ticks from sighting to meal | 5.4 | 7.9 | 6.1 | 7.7 |
| Turning toward visible food | **49%** | **51%** | **51%** | **52%** |
| Deaths | 89 | 49 | 75 | 46 |
| Mean energy | 0.81 | 0.63 | 0.78 | 0.63 |
| Critic's mean value (meals) | −4.0 | −1.7 | −3.2 | −1.6 |
| Mean TD error per brain tick (meals) | +0.088 | +0.019 | +0.067 | +0.016 |

- **Replay's gain is movement, not steering.** Agents with replay drive forward about four times as hard. They cover twice the ground, come across food 1.7 times as often, and reach it sooner once seen. They turn toward visible food no more often than chance, as without replay.
- **The mechanism is a lasting positive TD error.** Replay holds the critic's values below what TD would settle on, so the TD error stays positive, 4–5 times larger than without replay. Forward exploration that runs into moments resembling remembered pre-meal ones then keeps being reinforced, and the forward bias grows. In effect it works like an optimism about places that looked like food before.
- **The price is more deaths** (89 against 49), although meals double and mean energy rises. This harness does not record the cause of death.

## Replaying recent experience instead

The same offline replica tried replay drawn without regard to outcome. The source was either the last 128 or 1024 settled moments, or an even sample of the whole life so far (reservoir of 128). The target was TD's: the full eight-tick return with the drain, plus the critic's value at the window's end as it was then, and replay moved the bias as well. Setup: 300k-tick logged lives, steady encoder (T = 300), seeds 5–8, 40 agents; late-life values in meals (agents positive).

| Replay | In view − not | Centred − off | R² vs true return |
|---|---|---|---|
| none (TD only) | +0.29 ± 0.13 (32/40) | −0.01 ± 0.08 (18/40) | 0.05 |
| as shipped (outcome-filtered memory) | −2.48 ± 0.70 (17/40) | −2.31 ± 0.54 (10/40) | 0.01 |
| outcome-filtered memory, TD's target, bias too | −0.36 ± 0.33 (24/40) | −0.67 ± 0.27 (14/40) | 0.01 |
| **last 128 moments, TD's target, bias too** | **+0.64 ± 0.06 (38/40)** | **+0.25 ± 0.09 (26/40)** | 0.08 |
| last 128 moments, no bootstrap, bias too | +0.48 ± 0.05 (36/40) | +0.24 ± 0.06 (33/40) | 0.05 |
| last 1024 moments, TD's target, bias too | +0.54 ± 0.05 (40/40) | +0.13 ± 0.06 (25/40) | 0.16 |
| even sample of the life (128), TD's target, bias too | +0.35 ± 0.10 (34/40) | +0.19 ± 0.18 (23/40) | 0.08 |
| ideal least squares (reference) | +0.96 ± 0.05 (40/40) | +0.27 ± 0.08 (29/40) | 0.25 |

- **Replaying recent experience makes the critic clearly better than TD alone.** With the last 128 moments, the in-view lesson doubles (+0.64 against +0.29, of an ideal +0.96) and holds in 38 of 40 agents. The centring lesson appears at almost its ideal size (+0.25 against +0.27), where TD alone learns none.
- **What mattered was not filtering by outcome.** The same target on the outcome-filtered memory stays below TD alone.
- **A longer window (1024) predicts returns best (R² 0.16) but centres less.** An even sample of the whole life is noisier than a recent one.

These are offline results on the behaviour of today's replay agents. The GPU critic, the steering and the meals have yet to be measured with this replay in place.

## Recent-experience replay in the learner

The critic's value replay now draws on a ring of the last 128 brain ticks, kept whatever followed them. Each moment gathers TD's own eight-tick return, completed with the critic's value, or −1 if the life ends first, and replay moves the bias with the weights. Pattern memory no longer carries a return.

**Free runs** (300k ticks, seeds 5–8, 40 agents, shipped config without a critical period; GPU values, in meals):

| | Memory replay (before) | Recent replay (now) | No replay (scratch, for reference) |
|---|---|---|---|
| Critic, early life: in view − not | +0.33 ± 0.16 | +0.67 ± 0.05 | — |
| Critic, late life: in view − not | −1.85 ± 1.59 | **+0.51 ± 0.08** | — |
| Critic, late life: centred − off (agents positive) | −1.43 ± 1.08 (21/40) | +0.13 ± 0.13 (23/40) | — |
| Returns: centred − off | +0.64 | +0.78 | — |
| R²: \|bearing\| from the critic's input | 0.29 | 0.48 | — |
| Meals | 173 ± 12 | **88 ± 4** | 83 ± 4 |
| Distance travelled | 42,700 | 19,900 | 20,600 |
| Forward bias at the end | +0.48 | +0.05 | −0.04 |
| Sightings that end in a meal | 45% | 39% | 35% |
| Turning toward visible food | 49% | 50% | 51% |
| Deaths | 89 | 47 | 49 |

**Headless evolution** (`evo_vs1`, 20 generations, seeds 5–8, 10 agents):

| | Memory replay (before) | Recent replay (now) |
|---|---|---|
| Mean fitness | 0.0558 | 0.0554 |
| First generation | 0.0560 | 0.0576 |
| First 5 → last 5 generations | 0.0553 → 0.0565 | 0.0562 → 0.0552 |
| Meals per agent per generation | 21.4 | 19.9 |
| SD between genome means / from noise alone | 0.0067 / 0.0072 | 0.0069 / 0.0074 |

- **The critic now has the right sign, consistently.** Food in view is worth more than the rest early and late in life, with a small spread across agents. The centring lesson is positive but not yet significant.
- **Long lives eat half as much, as expected.** The doubling of meals came from the old replay's accidental optimism, which drove forward movement. Without it, agents move and eat like agents with no replay at all.
- **Steering is unchanged:** turning toward visible food stays at chance.
- **Evolution is unchanged.** A 40k-tick generation is too short for the old optimism to have built up much movement.

## Do the actors get a learning signal now?

The setup: a scratch harness on the current code (recent-experience replay), 300k-tick free runs, seeds 5–8, 40 agents. It logged every brain tick and measured the credit the eligibility traces give each tick's exploration noise: later TD errors, decayed by γλ per tick and cut at death. For steering, the credit is multiplied by the turn noise signed toward the food, on in-view ticks. For moving, it is multiplied by the forward noise, on all ticks. The same products with the true advantage (return minus the critic's value) give the ceiling a perfect critic would provide.

| Signal | Mean | Agents positive | Agents with t > 2 / t < −2 |
|---|---|---|---|
| Steering: credit from the traces | +1.6e-3 ± 0.3e-3 | 34/40 | 16 / 1 |
| Steering: true advantage (ceiling) | +1.6e-3 ± 0.4e-3 | 31/40 | 12 / 0 |
| Steering: credit through the encoding's side readout | +1.1e-3 ± 0.2e-3 | 31/40 | 12 / 0 |
| Moving: credit from the traces | −7.1e-5 ± 0.7e-5 | 3/40 | 0 / 2 |
| Moving: true advantage | −3.3e-5 ± 2.2e-5 | 14/40 | 1 / 1 |

Other measures:

- The food's side is readable from the critic's input on in-view ticks (R² 0.61).
- Over a life, the learned change in the turn weights aligns with that readout at cos +0.045 ± 0.006.
- Turning toward visible food stays at 49%.
- Fatigue averages 0.55, and on 49% of ticks it cuts the executed action below half.

- **Steering now has a right-signed signal, as strong as a perfect critic would give.** It also reaches the turn weights through the encoding.
- **But it barely moves the weights.** Food is in view on about 2.5% of ticks, and on the other 97.5% the trace updates are noise times encoding, a random walk in the same weights. What the turn weights learn over a life points only faintly at the food's side (cos 0.045).
- **Moving forward gets no positive signal, and the ceiling offers none either.** A one-tick forward kick costs energy for certain, while the food it might lead to comes too late and too rarely for the traces. Even the true advantage of a forward kick is about zero. The earlier doubling of meals came from optimism, not from a lesson available in the data.
- **Fatigue damps the executed action on half the ticks.** It does not damp the noise the traces credit, so the actors are credited for actions the body only partly carried out.

## Persistent forward exploration and crediting the executed action

A scratch build (not committed) tried two switches on the current code:

- **Persistent forward noise:** forward exploration becomes AR(1) with the turn noise's persistence (0.9), so each exploratory run lasts several ticks instead of one.
- **Credit for the executed action:** the actor traces carry the exploration noise as the body actually executed it, scaled by fatigue (and by the klinotaxis factor for turning).

Setup: 300k-tick free runs, seeds 5–8. The behaviour figures cover 40 agents per arm; the signal figures cover 30, because the fourth seed ran out of time under eight-way GPU sharing.

| | Current code | Persistent forward noise | Executed-action credit | Both |
|---|---|---|---|---|
| Meals | 86 ± 5 | **143 ± 9** | 107 ± 6 | **138 ± 8** |
| Distance travelled | 19,600 | 41,600 | 20,700 | 35,800 |
| Mean forward command | 0.053 | 0.007 | 0.088 | 0.098 |
| Forward bias at the end | +0.05 | −0.14 | +0.10 | +0.12 |
| Sightings | 191 | 255 | 208 | 253 |
| Sightings that end in a meal | 36% | 36% | **45%** | 42% |
| Turning toward visible food | 50% | 48% | 50% | 49% |
| Deaths | 47 | 84 | 45 | 72 |
| Mean energy | 0.63 | 0.76 | 0.65 | 0.76 |
| Forward credit (agents positive) | −7.8e-5 (1/30) | −2.2e-4 (5/30) | −6.8e-5 (1/30) | −3.7e-5 (15/30) |
| Forward true advantage (agents positive) | −1.4e-5 (10/30) | −5.2e-5 (14/30) | −5.2e-5 (8/30) | +1.7e-4 ± 1.4e-4 (20/30) |
| Steering credit (agents positive) | +1.7e-3 (29/30) | +1.3e-3 (22/30) | **+2.1e-3 (30/30)** | +1.1e-3 (24/30) |
| Turn-weight change vs food side (cos) | +0.047 | +0.019 | +0.056 | +0.006 |

- **Neither switch gives moving forward a positive learning signal.** The forward credit stays negative, and it only reaches about zero with both switches. Even the true advantage becomes positive only with both, and then not significantly.
- **Persistent forward noise raises meals by two thirds, but through exploration, not learning.** The noise's own longer runs cover twice the ground. The learned forward command stays at about zero, and the bias turns negative. Deaths nearly double, and mean energy rises; the harness does not record the cause of death.
- **Crediting the executed action helps modestly, and it is a correctness fix.** Meals rise by a quarter at the same distance, more sightings end in a meal (45% against 36%), and the steering credit becomes positive in every agent.
- **Turning toward visible food stays at chance in every arm.** More movement adds more no-food updates to the turn weights, which dilutes the steering signal (cos 0.019 and 0.006 against 0.047).

## Executed-action credit in the learner

The actors' eligibility traces now carry the exploration noise as the body executed it, scaled by fatigue and, for turning, by the klinotaxis factor.

**Free runs** (300k ticks, seeds 5–8, 40 agents):

| | Before | Now |
|---|---|---|
| Meals | 86 ± 5 | **107 ± 6** |
| Distance travelled | 19,600 | 21,000 |
| Sightings | 191 | 207 |
| Sightings that end in a meal | 36% | **45%** |
| Mean forward command | 0.053 | 0.080 |
| Deaths | 47 | 46 |
| Turning toward visible food | 50% | 50% |
| Steering credit (agents positive) | +1.7e-3 (29/30) | **+2.6e-3 (39/40)** |
| Turn-weight change vs food side (cos) | +0.047 | +0.059 |
| Forward credit (agents positive) | −7.8e-5 (1/30) | −6.7e-5 (1/40) |

**Headless evolution** (`evo_vs1`, 20 generations, seeds 5–8, 10 agents): mean fitness 0.0555 against 0.0554, with 19.9 meals per agent per generation in both. The SD between genome means is 0.0066, against 0.0071 from noise alone.

- **The port reproduces the scratch result.** Long lives eat a quarter more at about the same distance, because more sightings end in a meal. The steering credit is positive in 39 of 40 agents.
- **Neither steering nor moving is learned yet.** Turning toward visible food stays at chance, and the forward credit stays negative.
- **Evolution is unchanged.**

## Hunger arousal

Urgency (the agent's own energy and integrity distress) currently scales the reward by (1 + urgency). It also *lowers* exploration as hunger grows, by up to 0.5, reaching the 10% floor when the agent is starving. A scratch build (not committed) tried three switches of its own, applied cumulatively, all driven only by urgency:

- **Exploration:** rises with hunger by the same amount instead of falling.
- **Vigour:** the executed command, and the noise the traces credit, are scaled by up to ×1.5.
- **Plasticity:** the actors' learning rate is scaled by up to ×2.

Setup: 300k-tick free runs, seeds 5–8, 40 agents per arm.

| | Current code | Exploration rises | + vigour | + plasticity |
|---|---|---|---|---|
| Meals | 105 ± 6 | 84 ± 5 | **156 ± 11** | 139 ± 13 |
| Distance travelled | 20,700 | 26,000 | 56,700 | 50,400 |
| Sightings | 210 | 211 | 322 | 301 |
| Sightings that end in a meal | 43% | 30% | 33% | 32% |
| Deaths | 46 | 55 | **114** | 101 |
| Mean energy | 0.66 | 0.65 | 0.78 | 0.76 |
| Mean exploration rate | 0.41 | 0.82 | 0.80 | 0.80 |
| Forward bias at the end | +0.06 | −0.02 | −0.36 | −0.17 |
| Turning toward visible food | 50% | 49% | 49% | 49% |

- **More exploration alone hurts.** Urgency is near its cap most of the time, so exploration is pinned high. Behaviour becomes more random, fewer sightings end in a meal, and meals fall by a fifth.
- **Vigour raises meals by half, but only by amplifying motion.** Agents cover 2.7 times the ground while the learned forward bias turns negative. Deaths more than double even though mean energy rises, so the extra deaths are not starvation; the harness does not record their cause.
- **Stronger learning when hungry adds nothing,** and turning toward visible food stays at chance in every arm.

## Energy-limited processing capacity

The idea is a capacity limit rather than a drive, like a motor whose torque falls with its supply voltage, or the narrowing of attention under depletion. A scratch build (not committed) let only the k visual inputs that deviate most from their adapted level reach the encoder. k was the whole block (240) while the lower of energy and integrity stayed at or above a threshold, and shrank in proportion below it, down to 10%. A control held k at 10% always. Setup: 300k-tick free runs, seeds 5–8, 40 agents per arm.

| | Current code | Threshold 0.5 | Threshold 0.8 | Always 10% |
|---|---|---|---|---|
| Meals | 107 ± 6 | 104 ± 6 | 95 ± 5 | 103 ± 5 |
| Distance travelled | 21,200 | 21,000 | 20,100 | 20,900 |
| Sightings that end in a meal | 44% | 45% | 40% | 45% |
| Deaths | 47 | 47 | 47 | 46 |
| Turning toward visible food | 50% | 50% | 50% | 49% |
| R²: food side from the critic's input (in view) | 0.61 | 0.62 | 0.62 | 0.61 |
| Steering credit (agents positive) | +2.3e-3 (39/40) | +2.1e-3 (38/40) | +1.9e-3 (37/40) | +2.1e-3 (39/40) |
| Turn-weight change vs food side (cos) | +0.056 | +0.054 | +0.053 | +0.055 |
| Forward credit (agents positive) | −6.4e-5 (2/40) | −7.3e-5 (0/40) | −8.0e-5 (0/40) | −7.2e-5 (4/40) |

- **Capacity changes nothing measurable,** in behaviour or in learning, even with only a tenth of the visual inputs always passing.
- **Sensory adaptation already does the narrowing.** After adaptation the strongest deviations are what is new in view, food included, while the steady background is already near zero. Keeping only the strongest leaves the food's side exactly as readable (R² 0.61).
- **The limit is still in the actors.** The steering credit is right-signed in nearly every agent, but the turn weights barely follow it (cos 0.055), and moving forward is still credited negatively.

## Heritable angles of view and a sense of smell

Three new heritable sensory genes:

- **Horizontal and vertical angle of view** (seed 90° × 90°): the vision rays span them per agent.
- **Smell strength** (seed 1): sensitivity of two nostrils, 0.5 units ahead of the body and 1 unit to either side.

Each uneaten food item within 30 units adds exp(−d / 10) to a nostril's concentration C, and the nostril perceives 1 − exp(−strength · C). Breeding now gives every repeat group after the first its own mutation of the three genes; until now no config gene evolved, because every slot copied the parent's config.

**Free runs** (300k ticks, seeds 5–8, 40 agents per arm). "Near" ticks are those with food within smelling range (30 units). "Toward" is the share of those ticks on which the turn command points at the nearest food.

| | 90° view, no smell | 60° view | 150° view | Smell strength 1 | Smell strength 3 |
|---|---|---|---|---|---|
| Meals | 107 ± 6 | 103 ± 4 | **121 ± 5** | 100 ± 6 | 108 ± 5 |
| Sightings | 210 | 154 | 301 | 201 | 210 |
| Sightings that end in a meal | 45% | 55% | 36% | 43% | 45% |
| Mean energy | 0.66 | 0.64 | 0.68 | 0.65 | 0.65 |
| Share of ticks with food in smelling range | 90% | 90% | 88% | 89% | 89% |
| Toward the nearest food (in smelling range) | 49.7% | 49.5% | 49.8% | 49.6% | 49.6% |

| Learning signal (40 agents) | 90° view, no smell | 150° view | Smell strength 3 |
|---|---|---|---|
| Steering credit, food in view (agents positive) | +1.9e-3 (40/40) | +1.9e-3 (40/40) | +1.8e-3 (36/40) |
| Steering credit, food in smelling range (agents positive) | +5.6e-4 (40/40) | +5.4e-4 (40/40) | +5.6e-4 (40/40) |
| Turn-weight change vs food side (cos) | +0.046 | +0.059 | +0.053 |
| Forward credit (agents positive) | −5.5e-5 (6/40) | −5.9e-5 (2/40) | −7.8e-5 (0/40) |

**Headless evolution** (`evo_vs1`, 20 generations, seeds 5–8, 10 agents):

| | Before (no sensory genes) | Genes present, not varied | Genes evolving |
|---|---|---|---|
| Mean fitness | 0.0555 | 0.0563 | 0.0555 |
| First 5 → last 5 generations | 0.0562 → 0.0541 | 0.0559 → 0.0578 | 0.0566 → 0.0572 |
| Best score | 0.0601 | 0.0615 | 0.0630 |
| SD between genome means / from noise alone | 0.0066 / 0.0071 | 0.0060 / 0.0066 | 0.0067 / 0.0071 |

The accepted champions' genes wander without a common direction. Seeds sampled horizontal views of 45–138° and smell strengths of 0.6–1.5. Accepted champions ranged from 64° to 105° and from 0.86 to 1.22.

- **A wider view is the one sense that helps.** At 150°, agents come across food 43% more often and eat 13% more, although fewer sightings end in a meal. A 60° view loses as many sightings as it gains in conversion.
- **Smell does not change behaviour yet.** Food is within smelling range on about 90% of ticks, but turning toward it stays at chance at either strength.
- **The credit for turning toward food in smelling range is positive in every agent even without a nose.** The nose does not strengthen it. The turn policy, which reads the nostrils only through the learned encoding, does not pick up the side.
- **Evolution cannot yet tell the genes apart.** Differences between genomes stay within evaluation noise, so the champions' angles of view and smell strengths drift at random rather than climbing.

## Why smell does not help steering

Smell gives the cleanest test of the steering problem, because its cue is just the difference between two inputs: right nostril minus left. A scratch harness followed that cue link by link through whole lives. Setup: 300k-tick free runs, seeds 5–8, 40 agents per arm, current code. Ticks counted are those with food within smelling range (about 89% of all ticks). The learning signal is the credit the traces give each tick's executed turn noise, multiplied by the cue in question; t is per agent, over a life.

| Link | No nose | Smell 1 | Smell 3 |
|---|---|---|---|
| Nostrils: corr(right − left, food side) | — | **0.64** | 0.52 |
| Nostrils: R² of food side from the raw pair | — | 0.41 | 0.27 |
| Encoding: R² of food side from the turn policy's input | 0.09 | **0.19** | 0.18 |
| Encoding: R² of (right − left) from the turn policy's input | — | 0.37 | 0.47 |
| Credit for turns toward the food: mean t (agents t > 2) | +5.6 (36/40) | +6.0 (35/40) | +6.3 (38/40) |
| Learning signal along raw right − left: mean t (agents t > 2) | — | +6.1 (37/40) | +4.1 (34/40) |
| Learning signal along the encoding's side readout: mean t (agents t > 2) | +5.5 (35/40) | +5.2 (31/40) | +5.2 (34/40) |
| Turn-weight change along the side readout | +0.020 | **+0.010** | +0.011 |
| Turn-weight change in all other directions | 0.41 | **0.42** | 0.45 |
| Final policy: corr(turn output, food side) | +0.002 | +0.007 | +0.006 |

For smell strength 1, the learning rule was also rebuilt offline from the logged lives:

- The actual turn-weight change follows it (cos 0.72).
- Its first and second halves of life point the same way (cos +0.34; above 0.3 in 24 of 40 agents).
- A policy pointing along its whole-life direction does not steer either (corr with the food side +0.014).
- The large change is not the agent learning from its own rotation: the change along the encoding's readout of the body's angular velocity is +0.002.

**Every link works except the last.** The nostrils tell the food's side. The encoding keeps part of it, doubling what the turn policy's input says about the side over vision alone. Turns toward food are credited. The update drifts significantly along the food's side. But the turn weights change about 40 times more in other directions than along the side. That change is partly consistent through a life, so the rule is also learning other things, and partly noise. The policy's turning reflects those other directions and never the food's side.

This is the same failure as with vision: the side of food in view is readable (R² 0.6) and correctly credited, yet the turn weights barely align with it (cos 0.05). The bottleneck is not the senses, the encoding or the credit. It is the turn actor: one linear readout of a 128-dimensional encoding, trained by an update in which the food-side component is a few percent of the whole.

## Fixing the turn actor offline

The same logged lives were used to rebuild the turn weights offline under variants of the actor's rule. The baseline is the GPU's rule: rate 0.1, step normalised by 1 + |x|², credit from the traces, noise as executed. Each variant was scored on its late-life policy, by the correlation of its turn output with the food's side on ticks with food in smelling range. "Echo" is the correlation of the turn output with the agent's own current turn noise. Setup: smell strength 1, 300k-tick lives, seeds 5–8, 40 agents.

| Actor rule | Turns toward food: corr (agents positive) | Echoes its own turn noise: corr (agents positive) |
|---|---|---|
| as on the GPU | +0.015 ± 0.005 (28/40) | **+0.137 ± 0.021 (34/40)** |
| credit only the fresh noise innovation | +0.019 ± 0.004 (26/40) | −0.021 ± 0.025 (18/40) |
| credit centred on its mean | +0.015 ± 0.004 (28/40) | +0.139 ± 0.021 (35/40) |
| both | +0.019 ± 0.004 (24/40) | −0.022 ± 0.025 (18/40) |
| both + weight decay 1e-4 per tick | +0.009 ± 0.006 (24/40) | — |
| both + weight decay 1e-3 per tick | +0.001 ± 0.006 (21/40) | — |
| both + only the 16 strongest inputs active | +0.025 ± 0.006 (30/40) | — |
| both + sparse 16 + decay 1e-4 | +0.009 ± 0.006 (21/40) | — |

The real GPU policy, for reference: +0.007 ± 0.005 (23/40).

- **The consistent unrelated change is an artefact: the policy learns to echo its own exploration.** The turn noise persists from tick to tick (AR(1)), and the encoding shows the effects of earlier turns. The trace term noise × input therefore has a non-zero mean, and the rule drifts toward "keep turning the way you are". Crediting only each tick's fresh innovation, the part independent of the state, removes the echo in full. Centring the credit does nothing, because its mean is already about zero.
- **Removing the artefact does not uncover steering.** No variant gets the policy to turn toward food beyond a correlation of 0.025. Weight decay makes it worse, and a sparse input helps only marginally.
- **The remaining limit is dimensionality.** The same credit, applied to a one-parameter policy reading the raw nostril difference, points the right way in 37 of 40 agents (t ≈ 6, previous section). Spread over 128 dense inputs, the food-side drift is a few percent of the update and stays buried in the others' noise for the whole life.

## A turn policy reading the nostrils directly

The same offline rebuild tried a turn policy that reads the two nostrils directly, as separate left and right inputs with two learned weights, so no symmetry is built in. It was credited with the fresh noise innovation, the cleanest rule from the previous section. Inputs were tried raw, adapted (each nostril minus its running mean, rate 0.01 per brain tick), and adapted alongside the 128-dimensional encoding. Setup: smell strength 1, 300k-tick lives, seeds 5–8, 40 agents.

| Turn policy input | Learned right − left weight (agents positive) | Turns toward food: corr (agents positive) |
|---|---|---|
| raw nostrils | +3.5e-3 ± 0.6e-3 (**33/40**) | −0.002 ± 0.010 (21/40) |
| adapted nostrils | +4.4e-3 ± 0.6e-3 (**35/40**) | +0.048 ± 0.012 (28/40) |
| encoding + adapted nostrils | — | +0.014 ± 0.005 (24/40) |
| encoding only (the GPU's rule, for reference) | — | +0.017 ± 0.004 (31/40) |

- **With two inputs the rule learns the right direction.** The right nostril's weight ends above the left's in 33 of 40 agents raw and 35 of 40 adapted.
- **The policy still barely steers, because what the nostrils share drowns what tells them apart.** Overall odour strength varies about thirty times more than the right − left difference. The rule's step along each direction scales with that direction's variation, so the shared weight random-walks thirty times faster than the difference weight drifts. Turning then follows how strong the smell is, not which side it comes from. Adapting each nostril to its running mean helps a little; feeding the nostrils in beside the encoding dilutes them again.

So the limit is not only how many inputs there are but how unequal their variation is. The informative directions — the nostrils' difference, food's side in view — have small variance next to what they share with the rest of the input, and a learning rule whose step scales with input size learns along them slowest.

## Normalising and whitening the turn policy's input

The same offline rebuild tried transforms that equalise how much each input direction varies, so that the rule's step no longer favours the shared component. Setup: smell strength 1, 300k-tick lives, seeds 5–8, 40 agents. All variants are credited with the fresh noise innovation.

- **Divisive normalisation:** each nostril divided by the pooled odour, plus a semi-saturation constant of 0.01; optionally adapted to its running mean.
- **Online whitening:** a running mean and covariance of the two nostrils (rate 0.01 per brain tick), with the input transformed by C^(−1/2).
- **Preconditioned (the best case for whitening):** the rule's summed update over the first 70% of the life, multiplied by the inverse input covariance (a natural-gradient step), scored on the last 30%.

| Turn policy input | Turns toward food: corr | Agents positive | Agents above 0.2 |
|---|---|---|---|
| encoding, the GPU's rule (reference) | +0.008 ± 0.003 | 24/40 | 0 |
| raw nostrils | −0.013 ± 0.009 | 15/40 | 0 |
| adapted nostrils | +0.021 ± 0.010 | 26/40 | 1 |
| nostrils, divisive normalisation | +0.138 ± 0.025 | 33/40 | 14 |
| nostrils, divisive normalisation, adapted | +0.137 ± 0.022 | 33/40 | 12 |
| **nostrils, whitened online** | **+0.414 ± 0.034** | **39/40** | **36** |
| nostrils, preconditioned | +0.538 ± 0.028 | 39/40 | 38 |
| encoding, preconditioned | +0.048 ± 0.008 | 34/40 | 0 |
| encoding + nostrils, preconditioned | +0.086 ± 0.010 | 38/40 | 3 |

- **Whitening the nostrils makes the turn policy steer by smell within one life,** for the first time in this investigation: correlation +0.41 with the food's side, positive in 39 of 40 agents. The online version comes close to the best case (+0.54).
- **Divisive normalisation helps, but a third as much.** Dividing by the pooled odour removes overall strength, but the left–right balance it leaves is still small next to the remaining variation.
- **Whitening does not rescue the 128-dimensional encoding.** Even preconditioned, the encoding route stays near zero (+0.05), and adding the nostrils to it gives only +0.09. Estimating one good direction among 128 from credit this noisy needs far more than one life.

So steering by smell is learnable from the agent's own homeostatic signal. It takes a low-dimensional input in which the left–right difference varies as much as the shared part. The encoding the turn policy reads today is neither.

## The whitened smell pathway in the learner

The turn policy now also reads the two nostrils directly through two learned weights, zero at birth. Each brain tick the nostrils are centred and whitened by their running mean and covariance (rate 0.01). The pathway's trace gathers the turn noise's fresh innovation, as executed, times the whitened scent over 1 + |whitened|².

**Free runs** (300k ticks, seeds 5–8, 40 agents per arm, current code):

| | No smell | Smell strength 1 (default) | Smell strength 3 |
|---|---|---|---|
| Meals | 111 ± 5 | **160 ± 12** | 131 ± 7 |
| Sightings | 215 | 274 | 243 |
| Sightings that end in a meal | 46% | 51% | 48% |
| Deaths | 45 | 38 | 41 |
| Mean energy | 0.67 | 0.78 | 0.68 |
| Toward the nearest food in smelling range, whole life | 49.6% | 52.7% | 51.6% |
| Toward the nearest food in smelling range, last 30% | 49.7% | **53.6%** | 52.0% |
| Toward visible food | 49.2% | 52.0% | 51.0% |
| Pathway output vs food side, last 30% (agents positive; above 0.2) | — | **+0.41 ± 0.03 (38/40; 34)** | +0.31 ± 0.03 (36/40; 32) |

**Headless evolution** (`evo_vs1`, 20 generations, seeds 5–8, 10 agents):

| | Before | With the smell pathway |
|---|---|---|
| Mean fitness | 0.0555 | **0.0589** (higher in all 4 seeds) |
| First 5 → last 5 generations | 0.0566 → 0.0572 | 0.0598 → 0.0584 |
| Meals per agent per generation | 19.9 | 21.7 |
| Best score | 0.0630 | 0.0653 |
| Accepted generations | 2.8 | 3.8 |
| SD between genome means / from noise alone | 0.0067 / 0.0071 | 0.0072 / 0.0075 |

- **For the first time in this investigation, agents learn to steer toward food within one life,** from their own homeostatic signal alone. The pathway's turning tracks the food's side in 38 of 40 agents, as strongly as the offline rebuild predicted (+0.41). Over the last 30% of life, turning toward food in smelling range rises from chance to 53.6%.
- **That small turning bias is worth a lot.** Long lives eat 44% more and die less. Agents come across food more often (274 sightings against 215) and more sightings end in a meal.
- **Evolution benefits too, though a 40k-tick generation leaves little time to learn.** Mean fitness rises 6% and meals 9%. Genome differences still sit at the noise level, and the sensory genes still drift at random.
- **A stronger nose (3) steers less than the seed nose (1),** because saturation narrows the left–right difference.

## A whitened visual pathway, offline

The smell recipe applied to vision: turn policies rebuilt offline that read a small visual input instead of the 128-dimensional encoding. They use the smell pathway's rule: the fresh turn-noise innovation as eligibility, credit from the traces, and inputs normalised by 1 + |input|². The lives were logged with the current learner, smell pathway on (183 meals per life on average). Setup: 300k ticks, seeds 5–8, 40 agents.

The inputs:

- **Hemifields:** the left and right halves of the 8×6 field, averaged into red, green, blue and depth (8 inputs).
- **Columns:** each of the 8 columns averaged the same way (32 inputs).

Every value is adapted to its own running mean (rate 0.01). "Learned" means learned online through the life, whitened online where marked. "Preconditioned" is the best case: the summed update over the first 70% of life, multiplied by the inverse input covariance, scored on the last 30%.

Scores are the correlation of the policy's turn output with the food's side over the last 30% of life (agents positive; agents above 0.2). "Food in view" is within 5.5 units inside the 90° field; "food visible" is within 30 units inside it.

| Turn policy input | Food in view | Food visible |
|---|---|---|
| hemifields, raw, learned | +0.13 ± 0.03 (31/40; 10) | +0.05 ± 0.01 (32/40; 0) |
| **hemifields, whitened, learned** | **+0.33 ± 0.04 (37/40; 29)** | **+0.18 ± 0.02 (34/40; 21)** |
| columns, whitened, learned | +0.16 ± 0.03 (32/40; 17) | +0.09 ± 0.02 (34/40; 6) |
| hemifields, preconditioned | +0.52 ± 0.03 (38/40; 37) | +0.28 ± 0.02 (40/40; 33) |
| columns, preconditioned | +0.34 ± 0.03 (37/40; 31) | +0.16 ± 0.02 (37/40; 16) |
| raw vision (240), preconditioned | +0.15 ± 0.03 (36/40; 16) | +0.09 ± 0.01 (37/40; 2) |
| encoding (128), preconditioned | +0.09 ± 0.02 (33/40; 6) | +0.05 ± 0.01 (35/40; 0) |

- **Sight works the same way as smell.** A turn policy that reads the two hemifields, whitened online, learns within a life to turn toward food in view, in 37 of 40 agents (+0.33).
- **Fewer, balanced inputs learn better, at every step:** 8 hemifield inputs > 32 columns > 240 raw values > the 128-dimensional encoding. Whitening roughly doubles to triples what each input set learns.
- **The pattern matches smell exactly.** Steering is learnable from the agent's own homeostatic signal when the steering cue reaches the turn policy through a few inputs whose differences vary as much as what they share.

## Heritable pathway weights, the visual pathway, and the danger measurement

Three changes, measured in headless evolution (`evo_vs1.json`, seeds 5–8, 20 generations, 40k ticks each):

- **The visual pathway, ported.** The turn policy reads the whitened hemifields (left and right means of red, green, blue and depth) through eight learned weights. They use the smell pathway's credit. The whitening matrix is C^(−1/2) of the running 8×8 covariance, recomputed by Jacobi every 20 brain ticks.
- **Heritable pathway weights.** Evolution now perturbs the smell and visual pathway weights along with the turn-policy weights. Under birth-brain inheritance, both pathways had been born at zero in every generation, so selection had nothing in them to act on.
- **A fixed danger measurement.** The kernel measured the nearest danger only when the brain was given the danger percept, which is off by default. The avoidance-intent counters then read a stale distance of 0 with a bearing of 0: every tick counted as in range, none as a turn away, and avoidance intent read 0 whatever the agents did. The measurement now always runs, as the observer's. The brain still never sees it, so hazard ground can only be learned from homeostasis.

| | Smell pathway | + visual pathway | + heritable pathway weights (2 × 4 runs) |
|---|---|---|---|
| Mean fitness | 0.0581 | 0.0600 | **0.112 / 0.120** |
| First 5 → last 5 generations | 0.0578 → 0.0574 | 0.0615 → 0.0583 | **0.075 → 0.138 / 0.098 → 0.140** |
| Runs whose last 5 beat their first 5 | 2 / 4 | 0 / 4 | **8 / 8** |
| Accepted generations | 3.3 | 2.0 | 7.5 / 6.0 |
| Best score | 0.0646 | 0.0653 | 0.185 / 0.183 |
| Meals per agent per generation | 21.5 | 22.3 | 30.9 / 34.0 |
| Deaths per agent per generation | 6.94 | 6.67 | 3.11 / 3.12 |
| Share of distance travelled on hazard ground | 26.5% | 26.1% | 12.9% / 14.6% |
| Avoidance intent (turns away from danger within 30 units) | 0.514 | — | 0.519 |
| Approach intent (turns toward food within 30 units) | 0.497 | 0.497 | 0.513 / 0.524 |

The smell-pathway column and the second heritable batch carry the fixed danger measurement. The other runs predate it, so they have no avoidance intent. GPU runs are not bit-reproducible, so the two heritable batches differ by run-to-run variation.

- **Evolution finally climbs.** With the pathway weights heritable, the last five generations beat the first five in all 8 runs. Fitness doubles, meals rise by half, and deaths halve. Without heritability, the visual pathway adds nothing (0.0600 against 0.0581), and the smell pathway had added only 6%. What one life teaches is not passed on, so every generation starts again from zero.
- **What evolved is steering toward food, by smell and by sight.** In 3 of the 4 seeds of the second batch, the champion's smell weights are negative on the left nostril and positive on the right: turn toward the stronger side. The fourth (seed 5) has only the right one positive. The visual weights share no sign pattern across seeds, but they act on each agent's whitened coordinates, so their signs alone say little. Measured by behaviour (next section), the evolved visual weights steer toward food in view in every agent.
- **Agents spend half as long on hazard ground, but they do not turn away from it.** Avoidance intent stays at chance (0.519). Food grows only on food-rich ground, the opposite end of the biome noise from hazard. Agents that follow scent therefore stay near food and away from hazards without ever reacting to them. Learned danger avoidance, where an agent turns away from red ground because its integrity fell there, is still missing.

## Why sight does not learn to avoid hazards

Free lives under the evolution config (`evo_vs1.json`), 200k ticks, 20 agents, seeds 5–8. One arm starts from fresh brains (pathway weights zero). The other starts every agent from the last stored champion of the matching seed's second heritable batch (its birth brain, unmutated). Hazard ground covers 33% of the world.

Scores are correlations over the last 30% of life. Each one compares a turn output with the side to turn: away from the nearest hazard cell or toward the nearest food. "Hazard visible" means a hazard cell within 30 units inside the field of view, with the agent not on hazard ground; "near" means within 10 units; "food in view" means within 5.5 units. The figure in parentheses counts agents scoring above 0. Rows:

- **Readout:** the best linear map from the whitened hemifields (as the GPU pathway saw them) to the hazard side. It is trained supervised on the first 70% of life and is a measurement only.
- **Offline rule:** the pathway's own learning rule, rebuilt from the logged homeostatic credit.
- **GPU pathway:** what the pathway actually learned, or inherited and then learned.

| | Fresh brains | Evolved champions |
|---|---|---|
| Deaths per agent | 26.2 | 12.2 |
| Deaths with integrity below energy | 60% | 69% |
| Deaths on hazard ground | 68% | 73% |
| Time on hazard ground | 19.4% | 11.5% |
| Meals per agent | 99 | 177 |
| TD error on stepping onto hazard ground | −0.010 | −0.012 |
| Readout of the hazard side, visible / near | +0.33 / +0.38 (80/80) | +0.34 / +0.41 (80/80) |
| Offline rule, away from hazard, visible / near | +0.19 / +0.22 (76/80) | +0.12 / +0.15 (66/80) |
| GPU pathway, away from hazard, visible / near | +0.18 / +0.22 (75/80) | +0.06 / +0.08 (60/80) |
| GPU pathway, toward food in view | +0.32 (68/80) | **+0.48 (80/80)** |
| Size of the visual turn weights at the end | 0.026 | 0.19 |

- **Hazards are what kill, even for evolved agents.** Two thirds of deaths happen on hazard ground with integrity as the failing meter. Fitness counts every death through its survival multiplier, 0.25 + 0.75 / (1 + 0.5 · deaths), so going from three deaths to one raises the score by 36%. The selection pressure to avoid hazards is there.
- **The hazard side is in the input, and the homeostatic credit carries it.** A linear readout of the eight whitened hemifield values finds the hazard side in every agent. The critic's TD error turns negative on stepping onto hazard ground. Fresh brains' visual pathways learn, from that credit alone, to turn away from hazard in 75 of 80 agents, about half as strongly as the supervised readout.
- **But what one life learns is too weak, and none of it is inherited.** After 200k ticks, five generations' worth, the learned visual weights are only 0.026 in size. Their turn is a small fraction of the exploration noise. A generation lasts 40k ticks, and the next one starts from the champion's birth weights.
- **Evolution has spent the visual weights on food instead.** The champions' inherited visual weights are seven times larger and steer toward food in every agent (+0.48), more than a whole fresh life learns (+0.32). In those agents, lifetime learning barely turns them away from hazard (+0.06). So the earlier reading that steering by sight had not been selected was wrong. Sight was selected, for food, which is worth more per generation than the deaths avoided so far.
- **So the bottleneck is neither signal nor credit, but rate and inheritance.** The cue is there and the credit points the right way. Within a 40k-tick life, the pathway learns avoidance too slowly to matter, and evolution has not yet found that direction among its mutations.

## A faster visual pathway

The visual pathway's weight step was scaled by 3 and by 10 (scratch build only), in fresh-brain free lives: evolution config, 200k ticks, 20 agents, seeds 5–8. Each row gives the mean per agent ± standard error over 80 agents. "Generation window" covers ticks 28k–40k, the last 30% of a generation-length life. "Late" covers the last 30% of the 200k-tick life.

| Visual pathway rate | ×1 | ×3 | ×10 |
|---|---|---|---|
| Deaths per life | 26.6 ± 0.6 | 23.1 ± 0.5 | **20.8 ± 0.6** |
| Deaths on hazard ground per life | 18.7 ± 0.7 | 14.8 ± 0.6 | **11.2 ± 0.5** |
| Time on hazard ground | 19.6% | 16.8% | **14.1%** |
| Meals per life | 103 ± 5 | 110 ± 5 | 109 ± 4 |
| Generation window: deaths | 2.15 ± 0.08 | 1.84 ± 0.09 | **1.40 ± 0.09** |
| Generation window: time on hazard ground | 24.7% | 22.4% | **17.4%** |
| Generation window: size of the visual turn weights | 0.013 | 0.040 | 0.111 |
| Late: deaths on hazard ground | 3.4 ± 0.4 | 2.2 ± 0.2 | **1.6 ± 0.2** |
| Late: time on hazard ground | 15.7% | 12.4% | **10.4%** |
| GPU pathway, away from visible hazard, late | +0.19 (77/80) | +0.17 (73/80) | +0.15 (75/80) |
| GPU pathway, toward food in view, generation window | +0.22 | +0.23 | +0.13 |
| Avoidance intent, late | 0.530 | 0.528 | 0.523 |

- **A faster visual pathway avoids hazards within a single generation.** At ten times the rate, deaths on hazard ground fall by 40% over a life. Within the generation window, deaths fall by 35% and time on hazard ground by 30%. All four seeds move the same way at each step. Meals do not suffer.
- **The direction was already right; the size was missing.** The turn's correlation with the side away from hazard barely changes with rate. The weights grow eightfold, and by the end of a generation-length life they reach the size evolution gave the champions' food-steering weights (0.11 against 0.19).
- **There is a cost on the food side.** At ×10, the pathway's turn toward food in view within the generation window drops from +0.22 to +0.13: a faster rate makes the weights noisier. Meals hold, because smell still carries food steering.
- **Avoidance intent does not register any of this.** It counts every tick with any hazard cell within 30 units, including hazard cells behind the agent and ticks spent standing on hazard ground, so it stays near chance. Deaths and time on hazard ground are the measures that move.
- The trend has not levelled off at ×10.

## A heritable plasticity gene for the visual pathway

The visual pathway's turn weights now learn at the actors' rate times a heritable gene, `vision_plasticity` (seed 1, bounds 0–30). Like the other sensory genes, each later repeat group shares one momentum-biased mutation of it, at most ±10% per generation. Headless evolution, `evo_vs1.json`, seeds 5–8, 20 generations. One arm seeds the gene at 1 (the default). The other seeds it at 10 through the run config. The comparison is the last batch without the gene.

| | Without the gene | Gene seeded at 1 | Gene seeded at 10 |
|---|---|---|---|
| Mean fitness | 0.120 | 0.133 | 0.113 |
| First 5 → last 5 generations | 0.098 → 0.140 | 0.084 → 0.166 | 0.082 → 0.138 |
| Meals per agent per generation | 34.0 | 37.3 | 33.9 |
| Deaths per agent per generation | 3.12 | 2.58 | 3.54 |
| Share of distance travelled on hazard ground | 14.6% | 12.8% | 15.6% |
| Gene at the last generation, per seed | — | 1.00, 1.20, 0.88, 1.49 | 9.1, 8.8, 10.4, 11.4 |

Two batches of the same code without the gene gave mean fitness 0.112 and 0.120, so differences of about 0.01 are run-to-run variation.

- **Seeding plasticity at 10 does not help an evolving lineage.** Fitness, deaths and time on hazard ground are no better than at 1, and one seed (7) stalled at 0.069. Free lives showed fast learning cutting hazard deaths when the pathway starts from zero weights. An evolved lineage is born with large food-steering weights, and fast learning mostly adds noise to them: the free-life runs already showed food steering dropping from +0.22 to +0.13 at ×10.
- **Selection does not push the gene either way within 20 generations.** In each arm one run drifted up and another down, with no shared direction. The gene stays heritable at seed 1, so a lineage can tune it, but the speed of visual learning is not what limits hazard avoidance in evolution today.

## Why evolved agents barely learn avoidance in their own lives

The same free lives as before: evolution config, 200k ticks, 20 agents, seeds 5–8, fresh brains against the matching seed's stored champion. The new measures separate three explanations: smaller exploration noise, a saturated turn, and food rewards drowning out the hazard credit.

- **Learned change:** the visual turn weights now minus at birth, scored on its own against the side away from hazard. This isolates what the life learned from what it inherited.
- **Saturation:** the share of brain ticks on which `tanh(policy) + noise` hits the ±1 clamp. The trace credits the noise even where the clamp swallowed it.
- **Credit split:** the pathway's rule rebuilt offline from two disjoint sets of ticks: those with hazard visible or underfoot, and all the rest. Each is scored on late hazard ticks.

| | Fresh brains | Evolved champions |
|---|---|---|
| Size of the visual weights at birth | 0 | 0.194 |
| Size of what the life changed them by | 0.0275 | 0.0152 |
| Learned change, away from visible hazard / near hazard | +0.16 / +0.20 (73/80) | +0.09 / +0.12 (62/80) |
| Whole turn output, away from visible hazard | +0.16 | +0.05 |
| Whole turn output, toward food in view | +0.27 | +0.47 (80/80) |
| Exploration rate (hazard visible) | 0.389 (0.407) | 0.351 (0.360) |
| Mean size of the credited turn innovation | 0.0301 | 0.0241 |
| Saturated turns (hazard visible) | 0.18% (0.17%) | 0.23% (0.27%) |
| Policy's own turn, mean size | 0.067 | 0.181 |
| Time on hazard ground | 19.4% | 11.0% |
| Offline rule from hazard ticks only, away from hazard | +0.11 (size 0.021) | +0.08 (size 0.010) |
| Offline rule from all other ticks only, away from hazard | +0.14 (size 0.016) | +0.08 (size 0.011) |
| Mean TD error size on meals / otherwise | 0.35 / 0.021 | 0.24 / 0.019 |

- **Evolved agents do learn to turn away from hazards; the inherited weights hide it.** What a champion's life adds points away from hazard (+0.09 / +0.12), at about 60% of a fresh brain's strength. But it is 13 times smaller than the inherited weights, which steer toward food and barely react to hazard, so the whole turn barely shows it (+0.05). The lesson is then lost, because the next generation inherits the champion's birth weights.
- **Saturation is not the cause.** Fewer than 0.3% of turns hit the clamp.
- **Food credit does not drown the hazard lesson.** Rebuilt from non-hazard ticks alone, the rule still points away from hazard, as strongly as from the hazard ticks themselves.
- **Champions learn less for two smaller reasons.** Their stronger policy turn (0.18 against 0.07) lowers the adaptive exploration rate, so the credited innovations are 20% smaller. They also spend 11% of their time on hazard ground against 19%, which halves the credit gathered there.

## Removing the three suspected causes

Each suspected cause of weak lifetime avoidance in evolved agents was removed in a scratch build, alone and all together. Free lives started from the stored champions: evolution config, 200k ticks, 20 agents, seeds 5–8. Values are means ± standard error over 80 agents.

- **Exploration:** the policy-confidence cut to the exploration rate is removed, so a strong policy no longer explores less.
- **Clamp:** only the part of the turn noise the ±1 clamp let through is credited.
- **TD clip:** the TD error the visual pathway learns from is clipped at ±0.05, so meal-sized errors (about 0.24) cannot dominate its steps.

| | Champions as they are | Exploration | Clamp | TD clip | All three |
|---|---|---|---|---|---|
| Learned change, away from visible hazard | +0.090 ± 0.014 | +0.098 | +0.097 | +0.106 | **+0.121 ± 0.013** |
| Learned change, away from near hazard | +0.123 ± 0.019 | +0.126 | +0.131 | +0.133 | **+0.147 ± 0.017** |
| Size of the learned change | 0.015 | 0.018 | 0.015 | 0.011 | 0.012 |
| Whole turn, away from visible hazard | +0.052 | +0.044 | +0.049 | +0.056 | +0.048 |
| Whole turn, toward food in view | +0.466 | +0.480 | +0.467 | +0.469 | +0.464 |
| Exploration rate | 0.351 | 0.397 | 0.357 | 0.354 | 0.402 |
| Deaths per life | 11.7 ± 0.6 | 11.9 | 12.1 | 11.9 | 11.7 ± 0.5 |
| Deaths on hazard ground per life | 8.3 ± 0.7 | 8.3 | 8.9 | 8.2 | 8.0 ± 0.6 |
| Time on hazard ground | 11.0% | 11.0% | 11.4% | 11.0% | 11.2% |
| Meals per life | 178 ± 4 | 173 | 179 | 175 | 175 ± 4 |

- **Each fix sharpens the direction of what a life learns a little, and together they add about a third.** The learned change points away from visible hazard at +0.12 against +0.09, closer to a fresh brain's +0.16. Removing the exploration cut brings the exploration rate and innovation size back to a fresh brain's level. The TD clip points the lesson better but makes it smaller.
- **Behaviour does not change at all.** Deaths, deaths on hazard ground, time on hazard ground and meals are the same in every arm, within one standard error. The whole turn still reacts to hazard at about +0.05.
- **So these three were not what limits avoidance.** The learned change stays 13–16 times smaller than the inherited food-steering weights, and the next generation does not inherit it. It is the size of the lifetime lesson relative to the inherited steering, and its loss at every generation, that keeps evolved agents walking onto hazard ground.

## Is avoidance worth anything to evolution?

Every agent started from a stored champion's birth brain, with a vector added to its eight visual turn weights. The vector is either the champion's own lifetime lesson or a random direction of the same size. The lesson is the mean change those weights made over twenty 200k-tick lives from that champion: it points away from hazard. Lives ran for one generation (40k ticks) under the evolution config, 20 agents, seeds 5–8, 3 repeats each. They were scored with the governor's effort-rebased composite fitness, with exploration from the same 64×64 heatmap sampled every 100 ticks. Spawn positions and exploration noise are fixed per seed and repeat, so every arm is paired with the unmodified champion. Differences are means ± standard error over the 12 seed × repeat runs. The inherited visual weights are about 0.2 in size.

| Added to the visual weights | Fitness | Deaths | Deaths on hazard ground | Meals |
|---|---|---|---|---|
| Nothing (champion as is) | 0.123 | 2.35 | 1.78 | 35.6 |
| Lesson direction, size 0.05 | +0.020 ± 0.002 | −0.68 ± 0.08 | −0.55 ± 0.10 | +2.3 ± 0.4 |
| Lesson direction, size 0.10 | +0.029 ± 0.004 | −1.02 ± 0.19 | −0.93 ± 0.19 | +3.1 ± 0.6 |
| Lesson direction, size 0.20 | **+0.040 ± 0.007** | **−1.50 ± 0.31** | **−1.45 ± 0.33** | +2.2 ± 0.6 |
| Random direction, size 0.10 | −0.015 ± 0.008 | +0.32 ± 0.34 | +0.24 ± 0.37 | −2.5 ± 1.5 |
| Random direction, size 0.20 | −0.038 ± 0.011 | +1.35 ± 0.71 | +0.72 ± 0.80 | −8.7 ± 2.4 |

- **Avoidance is worth a lot to fitness.** The lesson direction at the inherited weights' size raises fitness by a third (0.123 → 0.163). Deaths on hazard ground fall by 81% and meals rise. Every seed gains at every size, from +0.010 to +0.071.
- **The direction matters, not the size.** Random vectors of the same size lower fitness and cost up to a quarter of the meals, because they disturb the inherited food steering.
- **So the fitness function rewards avoidance; evolution's search cannot find it.** A single life finds the avoidance direction and selection would reward it. But the steering mutations are random kicks to a quarter of the weights, and random moves of a useful size mostly hurt. A lineage of a few repeat groups per generation is unlikely to stumble on the narrow direction that helps, and the next generation never inherits what each life learned.

## Recombining the champion's steering from the fitter groups

An accepted node's champion brain is now the best agent's birth brain with its steering weights (turn policy, smell and visual pathways) recombined from the fitter half of the repeat groups. The recombined weights are the unperturbed template plus the log-rank-weighted mean of those groups' perturbations; with five groups the weights are 0.64, 0.28 and 0.08. Headless evolution, `evo_vs1.json`, 20 generations, seeds 5–12. The control is develop without recombination, run alongside on the same seeds.

| | Champion only (control) | Recombined | Seed-paired difference |
|---|---|---|---|
| Mean fitness | 0.126 | **0.157** | +0.032 ± 0.014 (5/8 seeds higher) |
| First 5 → last 5 generations | 0.093 → 0.162 | 0.103 → 0.196 | last 5: +0.034 ± 0.019 (6/8) |
| Meals per agent per generation | 36.3 | **44.5** | +8.2 ± 3.7 (5/8) |
| Deaths per agent per generation | 2.89 | 2.49 | −0.40 ± 0.41 (5/8 lower) |
| Share of distance travelled on hazard ground | 13.8% | 12.9% | −0.9 ± 1.5 points |
| Avoidance intent / approach intent | 0.524 / 0.529 | 0.532 / 0.549 | |

- **Recombination lets evolution climb faster.** Mean fitness rises by a quarter, and the last five generations end higher, mostly through more meals: the fitter groups' perturbations agree on better food steering, and averaging keeps that.
- **Hazard avoidance still does not emerge.** Deaths and time on hazard ground do not change beyond noise. Over 20 generations, the direction that turns away from hazard is still not among what the fitter groups share. Food steering pays off sooner and more reliably, so it wins the rank-weighted average first.

## Eighty generations with recombined steering

The current develop learner, with recombined steering, ran headless for 80 generations (`evo_vs1.json`, seeds 5–8). There was no control lineage, so the trend cannot be attributed to recombination alone. Values pool the four runs within each block of ten generations.

| Generations | Fitness | Meals per agent | Deaths per agent | Share of distance on hazard ground |
|---|---|---|---|---|
| 0–9 | 0.127 | 41.1 | 3.74 | 17.3% |
| 10–19 | 0.174 | 49.3 | 1.96 | 11.2% |
| 20–29 | 0.181 | 51.2 | 1.90 | 10.7% |
| 30–39 | 0.193 | 52.2 | 1.49 | 9.9% |
| 40–49 | 0.190 | 51.3 | 1.56 | 9.9% |
| 50–59 | 0.200 | 51.8 | 1.28 | 9.1% |
| 60–69 | 0.192 | 49.2 | 1.47 | 9.8% |
| 70–79 | 0.199 | 51.5 | 1.32 | 9.1% |

- **Food steering is settled by generation 20.** Meals rise from 41 to about 51 per agent in the first twenty generations and stay there.
- **After that, deaths keep falling slowly while meals stay flat.** From generations 10–19 to 70–79, deaths fall by a third (1.96 → 1.32) and time on hazard ground by a fifth (11.2% → 9.1%). Fitness creeps from 0.17 to about 0.20. Three of the four seeds die less (seed 5: 2.65 → 1.56; seed 7: 1.18 → 0.80; seed 8: 2.74 → 1.19). Seed 6 dies slightly more (1.28 → 1.74).
- **So avoidance arrives by slow accumulation, not as a step.** Once food steering stops improving, survival is what still improves. At about 1.3 deaths per generation, agents are still far from what a single life's avoidance lesson was worth when added directly: 81% fewer hazard deaths. (The control lineage in the next section shows that recombination slows this rather than causing it.)

## Eighty generations: the control, and twice the population

Two more 80-generation lineages on seeds 5–8, to set beside the recombined one above:

- **Champion only:** develop before recombination, where the stored brain is the best group's birth brain unchanged.
- **Recombined, population 20:** recombination with 20 agents in 10 repeat groups instead of 10 agents in 5. This doubles the agents competing for the same food, so meals are not comparable with the other arms.

The population-20 runs reached generation 78 within the time limit. The table pools the four seeds per block of ten generations.

| Generations | Champion only: deaths / hazard / fitness | Recombined, population 10 | Recombined, population 20 |
|---|---|---|---|
| 0–9 | 3.37 / 15.6% / 0.123 | 3.74 / 17.3% / 0.127 | 3.47 / 15.5% / 0.114 |
| 10–19 | 1.83 / 10.2% / 0.171 | 1.96 / 11.2% / 0.174 | 1.93 / 10.2% / 0.148 |
| 30–39 | 1.25 / 8.6% / 0.194 | 1.49 / 9.9% / 0.193 | 1.31 / 8.2% / 0.172 |
| 50–59 | 0.92 / 7.5% / 0.219 | 1.28 / 9.1% / 0.200 | 1.06 / 7.4% / 0.188 |
| 70–79 | **0.57 / 6.0% / 0.246** | 1.32 / 9.1% / 0.199 | 0.93 / 6.9% / 0.197 |

Generations 60–78, per seed (5, 6, 7, 8):

| | Deaths per agent | Fitness |
|---|---|---|
| Champion only | 1.06, 0.87, 0.42, 0.48 | 0.226, 0.223, 0.230, 0.259 |
| Recombined, population 10 | 1.58, 1.81, 0.94, 1.28 | 0.168, 0.190, 0.204, 0.219 |
| Recombined, population 20 | 1.63, 0.67, 1.08, 0.68 | 0.170, 0.206, 0.177, 0.214 |

- **Given enough generations, the champion-only lineage evolves avoidance on its own.** Deaths fall by 83% (3.37 → 0.57), time on hazard ground from 15.6% to 6.0%, and fitness keeps rising to 0.246. Meals keep rising too (38 → 57).
- **Recombination slows this down.** Over the first twenty generations the two lineages are indistinguishable, and the +25% seen earlier over 20 generations was early-phase variation. From generation 30 on, the champion-only lineage pulls ahead in all four seeds, on fitness, deaths and time on hazard ground. A recombined brain is stored without ever having been evaluated, and averaging pulls in the unperturbed group and weaker perturbations, so each accepted step is smaller and less sure than the best group's own.
- **More groups help recombination, but not enough.** With ten groups the recombined lineage dies and wanders onto hazard ground less than with five (0.93 against 1.32 deaths). It still trails the champion-only lineage, and it has less food per agent to work with.
- **What a lineage needed was time, not a better search.** Twenty generations were too few to see avoidance evolve. Eighty are enough for the original champion-only search, while recombination costs it ground.

## Avoidance intent restricted to hazard ahead

The avoidance-intent counters now count only ticks where the nearest hazard cell is within 30 units and inside the agent's own horizontal field of view, with the agent off hazard ground. Hazard behind or beside the agent, and ticks spent standing on it, no longer count.

To check the corrected metric, the champion-only lineage's stored champions were replayed. Each run starts every agent from the champion the lineage held at a given generation, or from a fresh brain, and lives one generation (40k ticks) under the evolution config: 20 agents, seeds 5–8, one repeat. "Hazard deaths" are deaths while on hazard ground.

| Champion at generation | Avoidance intent (seeds 5, 6, 7, 8) | Approach intent | Deaths | Hazard deaths | Meals | Fitness |
|---|---|---|---|---|---|---|
| Fresh brain | 0.518 (0.528, 0.511, 0.515, 0.518) | 0.506 | 6.90 | 6.45 | 19.7 | 0.056 |
| 0 | 0.535 (0.543, 0.522, 0.542, 0.532) | 0.576 | 4.97 | 4.33 | 29.6 | 0.081 |
| 10 | 0.535 | 0.568 | 2.34 | 1.88 | 38.6 | 0.134 |
| 20 | 0.535 | 0.556 | 1.99 | 1.49 | 39.5 | 0.141 |
| 40 | 0.533 | 0.566 | 1.85 | 1.52 | 42.1 | 0.152 |
| 60 | 0.533 | 0.565 | 0.97 | 0.75 | 46.0 | 0.185 |
| 78 | 0.535 (0.549, 0.527, 0.516, 0.540) | 0.563 | 0.64 | 0.53 | 47.6 | 0.201 |

- **The replay confirms the lineage evolves avoidance.** From generation 0 to 78, deaths on hazard ground fall by 88% (4.33 → 0.53) while meals rise by 60%.
- **Even restricted to hazard ahead, avoidance intent does not show it.** It stays at about 0.535 from generation 0 on. A per-tick count of which way the motor turn points cannot see this avoidance: whatever the evolved agents do differently, it is not a consistent turn away on the ticks when hazard is ahead. What they do instead was not measured here.
- **Deaths on hazard ground and time spent on it are the measures to watch** for avoidance. The `behavior_metric` table already records the latter as `danger_dwell_fraction`.

## How evolved agents avoid hazard ground

The champion-only lineage's champions were replayed for one generation (40k ticks, evolution config, 20 agents, seeds 5–8), with every brain tick logged. Columns are a fresh brain and the champions held at generations 0, 40 and 78. Values are means ± standard error over 80 agents. "Hazard ahead" means a hazard cell within 10 units inside the agent's view, with the agent off hazard ground; distances are in world units and visit lengths in brain ticks (10 physics ticks each).

| | Fresh brain | Generation 0 | Generation 40 | Generation 78 |
|---|---|---|---|---|
| Deaths on hazard ground per life | 6.56 ± 0.16 | 4.21 ± 0.18 | 1.40 ± 0.13 | **0.41 ± 0.06** |
| Steps onto hazard ground per life | 55.6 | 47.7 | 29.8 | **22.4** |
| Length of a visit on hazard ground | 18.7 | 16.0 | 13.1 | **11.5** |
| Integrity on stepping onto it | 60.5 | 60.3 | 67.2 | 72.6 |
| Time on hazard ground | 25.8% | 18.9% | 10.0% | 6.3% |
| Time on food-rich ground | 38.3% | 47.6% | 58.6% | **61.8%** |
| Speed | 3.02 | 2.98 | 2.58 | 2.49 |
| Turn size | 0.074 | 0.088 | 0.136 | **0.165** |
| Hazard ahead: turn away, signed and weighted by size | +0.004 | +0.013 | +0.030 | **+0.041** |
| Hazard ahead: distance change over the next 5 brain ticks | −3.10 | −3.12 | −2.65 | −2.45 |
| Hazard ahead: onto hazard ground within 5 brain ticks | 38.1% | 34.6% | 23.0% | **18.6%** |
| Meals per life | 20.4 | 29.9 | 41.0 | 47.2 |

- **Evolved agents avoid hazards in four ways at once.**
  - They stay on food-rich ground, 62% of their time against 48% at generation 0, which lies at the far end of the biome noise from hazard.
  - They move more slowly and turn about twice as much, a winding search within food patches.
  - With hazard ahead, they step onto it half as often (18.6% against 34.6% within five brain ticks), approaching more slowly and turning away.
  - When they do step on, they leave sooner (11.5 against 16.0 brain ticks) and arrive with more integrity.
  Together they step onto hazard ground half as often and die there 90% less.
- **The turn away is real but small next to their turning overall.** Weighted by size, the turn with hazard ahead leans away from it ten times more than a fresh brain's (+0.041 against +0.004). That lean is a quarter of the turn's typical size (0.17), so the sign of a single tick's turn points away only slightly more than half the time. That is why the sign-count avoidance intent stays at about 0.535.
- **Measures that show the avoidance:**
  - the size-weighted turn away with hazard ahead;
  - how often hazard ahead is followed by stepping onto it;
  - steps onto hazard ground per life, and the length of each visit;
  - time on hazard ground and on food-rich ground.

## Hazard avoidance in the evolution panel

Two generation-cumulative physics counters, preserved across respawn, now record what the mechanism study found to move with an evolved lineage:

- **Size-weighted turn away** (`P_AVOIDANCE_TURN_AWAY`): `motor_turn · sign(danger_bearing)` summed over the avoidance-counted ticks. Per generation it is stored as `behavior_metric.avoidance_turn_away`, the mean turn away, with 0 at chance.
- **Steps onto hazard ground** (`P_HAZARD_ENTRIES`): stored per generation as `behavior_metric.hazard_entries_per_agent`.

The evolution panel charts steps onto hazard ground per agent and the share of the path on it, below the fitness chart, with the latest turn away in the legend.

A three-generation headless check (seed 5) recorded, for generations 0 and 1:

| | Generation 0 | Generation 1 |
|---|---|---|
| Steps onto hazard ground per agent | 61.0 | 46.6 |
| Share of the path on hazard ground | 23.9% | 17.4% |
| Mean turn away from hazard ahead | +0.004 | +0.011 |

These are in line with the replayed lineage at the same stage.

## A reproducible eighty-generation baseline

Seeded headless runs now reproduce exactly: the kernel settles contested food by agent index and keeps grid cells in index order, and the governor draws every random choice of a run from its seed. Two runs of seed 5 match in every node, agent result and stored brain. Seed 6 rerun alone for ten generations matches the baseline below in every config, fitness and agent result; only node statuses differ, where the baseline later abandoned a champion. A later learner change can therefore be compared with this baseline run for run, on the same seeds, instead of through averages over noisy lineages.

The baseline is `develop` at `3d505b3`, headless, seeds 5–8, `--generations 80` (generations 0–78). The config is the `--dump-config` default with `governor.tick_budget` set to 40,000: 10 agents in 5 repeat groups, `vision_stride` 1. To reproduce a seed:

```bash
xagent --dump-config > base.json   # then set governor.tick_budget to 40000
xagent --config base.json --seed 5 --no-render --db base_s5.db --generations 80
```

`XAGENT_BRAIN_BESIDE_VISION=0` gives the same results and ran faster on the iGPU (four runs side by side, about 23 s per generation, 30 minutes per run). The table pools the four seeds per block of ten generations.

| Generations | Fitness | Meals per agent | Deaths per agent | Share of distance on hazard ground | Steps onto hazard ground per agent | Turn away from hazard ahead |
|---|---|---|---|---|---|---|
| 0–9 | 0.137 | 41.6 | 3.31 | 16.0% | 45.8 | +0.006 |
| 10–19 | 0.169 | 48.5 | 2.01 | 11.9% | 39.4 | +0.008 |
| 20–29 | 0.155 | 45.1 | 2.25 | 12.8% | 41.6 | +0.008 |
| 30–39 | 0.188 | 48.8 | 1.54 | 9.7% | 33.8 | +0.008 |
| 40–49 | 0.183 | 48.2 | 1.55 | 9.5% | 33.8 | +0.009 |
| 50–59 | 0.187 | 52.0 | 1.80 | 10.8% | 36.2 | +0.008 |
| 60–69 | 0.201 | 52.8 | 1.60 | 10.6% | 34.7 | +0.009 |
| 70–78 | 0.196 | 52.0 | 1.49 | 9.9% | 34.3 | +0.009 |

Per seed (5, 6, 7, 8):

| | Generations 0–9 | Generations 60–78 |
|---|---|---|
| Deaths per agent | 3.31, 3.07, 3.91, 2.95 | 0.74, 2.90, 1.92, 0.65 |
| Fitness | 0.125, 0.162, 0.096, 0.166 | 0.245, 0.157, 0.158, 0.234 |
| Meals per agent | 36.7, 50.8, 29.5, 49.3 | 58.8, 53.1, 43.5, 54.4 |
| Best score at the end | | 0.281, 0.244, 0.226, 0.292 |

Each seed accepted 15–19 generations, but most were later abandoned: after five failed children in a row, a champion is marked exhausted and the search backtracks to its parent. The final champion's line (seeds 5, 6, 7, 8):

- Seed 5: generations 0–3, 24, 43, 62, 76 (11 more accepted and abandoned)
- Seed 6: generations 0, 2, 3, 35 (11 more abandoned)
- Seed 7: generations 0–3, 16, 43, 75, 77 (11 more abandoned)
- Seed 8: generations 0–3, 28, 29, 73, 74 (11 more abandoned)

- **After generation 3, a lasting champion arrives only every 15–40 generations.** Most generations fail against their parent (60–64 of 79 per seed), and most accepted ones lead nowhere: their children all fail, so the step was probably a lucky evaluation.
- **Seeds 5 and 8 evolve avoidance as the champion-only lineage did before:** deaths fall by about 75% and the share of the path on hazard ground halves. Seed 7 gets there more slowly. Seed 6's line gained one lasting champion after generation 3 (at 35), and its deaths barely fall.
- **The pooled numbers trail the earlier champion-only lineage** (0.57 deaths per agent and 0.246 fitness by generations 70–79). That run used other random streams and an earlier kernel, and its seed 6 did well where this one stalls. Single lineages differ this much, so changes should be judged against this baseline seed by seed, not against earlier pooled numbers.

## Judging a generation against its champion scored alongside it

In the baseline, a generation was accepted when its mean fitness beat the spawn parent's stored mean by one pooled standard error. That stored mean was a single noisy score. After a lucky one, the parent's own children could not reach it: five failures in a row marked the parent exhausted, and the search backtracked to the grandparent and discarded the parent's champion. Of the 15–19 generations each seed accepted, 11 ended that way.

Every generation already evaluates the champion unmutated in its first repeat group, in the same world as its mutants. The governor now accepts a generation when its best mutated group beats that champion group by one standard error of a group difference (from the pooled within-group variance). A root's first evaluation is still accepted, and a generation with no mutated group falls back to the parent's stored score.

The variant measured alongside it also kept the champion when patience ran out, instead of backtracking. It made no difference by the end: pooled fitness over generations 60–78 was 0.252 against 0.254, two seeds each way. Only the comparison against the concurrent champion was landed.

Same config and seeds as the baseline (80 generations, generations 0–78). Pooled per block of ten generations, fitness / deaths per agent / share of distance on hazard ground:

| Generations | Baseline | Concurrent champion | Concurrent champion, keep on exhaustion |
|---|---|---|---|
| 0–9 | 0.137 / 3.31 / 16.0% | 0.137 / 3.29 / 16.5% | 0.137 / 3.29 / 16.5% |
| 10–19 | 0.169 / 2.01 / 11.9% | 0.192 / 1.31 / 9.8% | 0.204 / 1.19 / 9.5% |
| 20–29 | 0.155 / 2.25 / 12.8% | 0.179 / 1.44 / 10.0% | 0.225 / 0.81 / 8.0% |
| 30–39 | 0.188 / 1.54 / 9.7% | 0.229 / 0.68 / 7.0% | 0.253 / 0.55 / 6.6% |
| 40–49 | 0.183 / 1.55 / 9.5% | 0.232 / 0.51 / 6.0% | 0.255 / 0.48 / 6.3% |
| 50–59 | 0.187 / 1.80 / 10.8% | 0.252 / 0.51 / 7.1% | 0.262 / 0.45 / 6.7% |
| 60–69 | 0.201 / 1.60 / 10.6% | 0.262 / 0.42 / 6.7% | 0.250 / 0.60 / 7.3% |
| 70–78 | 0.196 / 1.49 / 9.9% | 0.244 / 0.53 / 7.1% | 0.254 / 0.45 / 6.2% |

Generations 60–78 per seed (5, 6, 7, 8):

| | Baseline | Concurrent champion |
|---|---|---|
| Fitness | 0.245, 0.157, 0.158, 0.234 | 0.273, 0.249, 0.224, 0.269 |
| Deaths per agent | 0.74, 2.90, 1.92, 0.65 | 0.45, 0.68, 0.56, 0.22 |
| Meals per agent | 58.8, 53.1, 43.5, 54.4 | 64.1, 61.4, 51.2, 59.7 |
| Share of distance on hazard ground | 7.2%, 15.6%, 11.2%, 7.1% | 7.6%, 8.3%, 6.0%, 5.6% |
| Accepted generations (abandoned later) | 19 (11), 15 (11), 19 (11), 19 (11) | 29 (7), 40 (3), 25 (5), 27 (5) |
| Best score | 0.281, 0.244, 0.226, 0.292 | 0.336, 0.301, 0.269, 0.307 |

- **Every seed ends fitter and dies less.** Fitness over generations 60–78 rises by 11–59% and deaths fall by 39–77%. Seed 6, which stalled in the baseline, now does as well as the others.
- **The gain comes early and holds.** By generations 10–19 the lineages are ahead (1.31 against 2.01 deaths per agent). They stay ahead in every later block.
- **Accepted steps now stick.** Each seed accepts 25–40 generations and abandons 3–7 of them, against 11 of 15–19 before. The final champion's line is two and a half to nine times as long.
- **Hazard avoidance keeps improving through evolution.** Deaths per agent reach about 0.5 by generation 40, which the baseline never reached in 80 generations.

## How much of the evolved avoidance is learned within a life

Each reference lineage's final champion (selection against the concurrent champion, generation 78) was replayed for one generation: 40,000 ticks, 10 agents all starting from the champion's birth brain and config, on its home world and on six worlds it never saw (seeds 101–106). Each replay ran in three arms. **Learning on** is the champion as evolved. **Steering frozen** holds the turn policy and the smell and visual pathways (the existing switch). **Policy frozen** also holds the forward weights, the forward bias and the encoder (a scratch switch, measurement only). In every arm the value head, predictor, memory and habituation keep adapting, but those reach the motor only through the frozen policy. A fresh birth brain, with learning on and with the policy frozen, ran on the same ten worlds for reference. Seeded runs repeat exactly, so the arms differ only in what learns.

Champions, means per agent (home: 4 lineage-world pairs; unseen: 24):

| | Learning on | Steering frozen | Policy frozen |
|---|---|---|---|
| Fitness, home / unseen | 0.244 / 0.257 | 0.239 / 0.256 | 0.250 / 0.253 |
| Deaths, home / unseen | 0.57 / 0.50 | 0.53 / 0.48 | 0.48 / 0.47 |
| Meals, home / unseen | 58.4 / 61.0 | 57.1 / 59.9 | 60.0 / 59.4 |
| Share of distance on hazard ground, home / unseen | 7.4% / 7.1% | 7.2% / 6.7% | 6.8% / 6.9% |
| Steps onto hazard ground, first → last quarter of the life | 5.7 → 6.3 | 5.6 → 5.8 | 5.7 → 5.5 |

Paired over all 28 lineage-world pairs, learning on minus policy frozen: fitness +0.003 ± 0.005, deaths +0.04 ± 0.05, meals +1.1 ± 0.9, hazard share +0.24 ± 0.15 points.

Fresh brain, 10 worlds:

| | Learning on | Policy frozen |
|---|---|---|
| Fitness | 0.063 | 0.060 |
| Deaths per agent | 6.68 | 7.86 |
| Share of distance on hazard ground | 25.4% | 27.7% |
| Steps onto hazard ground, first → last quarter | 17.6 → 15.3 | 17.3 → 17.6 |
| Deaths, first → last quarter | 1.56 → 1.53 | 1.58 → 2.05 |

Paired, learning on minus policy frozen: deaths −1.18 ± 0.15 (fewer in all 10 worlds), hazard share −2.3 ± 0.5 points (lower in all 10).

- **The evolved avoidance is innate.** Freezing the whole policy for a life changes nothing measurable: deaths, meals, fitness and time on hazard ground stay within noise. If anything, learning lets a champion drift slightly onto hazard ground over its life (5.7 → 6.3 steps per quarter, against 5.7 → 5.5 frozen).
- **It is not a memorised map.** The champions avoid hazard ground as well on six unseen worlds as at home (0.50 against 0.57 deaths per agent). Evolution always used one world per seed, yet what it found is general.
- **The lifetime learner does learn avoidance from homeostatic decline, from a naive start.** A fresh brain with learning on dies 15% less than the same brain frozen, in every world. Its visits to hazard ground fall over the life (17.6 → 15.3 per quarter) while the frozen brain's stay flat and its deaths climb.
- **But a life's lesson is small against what evolution stores.** Learning takes a fresh brain from 7.9 to 6.7 deaths per agent in one life; eighty generations reach 0.5. Once the inherited policy avoids hazards, the lifetime learner adds nothing more.
