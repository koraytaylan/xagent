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
