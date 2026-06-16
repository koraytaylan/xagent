# Learning-Loop Review: Why Danger Avoidance and Food Chasing Aren't Emerging

**Scope:** the full within-lifetime learning stack — the 7 brain passes
(`brain_passes.wgsl`), the fused kernel (`kernel_tick.wgsl`), vision/senses
(`phase_vision.wgsl`), constants (`common.wgsl`), brain init (`buffers.rs`),
config defaults (`config.rs`), and the governor's fitness function
(`governor.rs`).

**Question under review:** after many iterations there is still no consistent
danger avoidance or food chasing. Behavior is clearly not random, but learning
is not observable at a useful rate. Is this an intelligence problem? A failure
to define intelligence properly? A missing survival instinct?

---

## TL;DR

The learning method itself — a TD(λ) actor-critic over urgency-amplified
homeostatic reward — is conceptually sound and is the right "let them figure
out food=good, danger=bad on their own" framing. The failure is **not** the
definition of intelligence. It is that the world, as the brain experiences it,
is not learnable:

1. Agents act on senses that are up to **100 physics ticks stale** and frozen
   across 10 consecutive decisions — closed-loop pursuit is structurally
   impossible.
2. The reward for food pays only on contact, and the bridge across that gap is
   thin.
3. Death — the one event that should teach "danger=bad" — carries **zero
   learning signal and zero cost**.

The survival-instinct hunch is correct in a precise, fixable way (see Finding
2). No learner of any size can compensate for a severed sensorimotor loop, so
making the learner "more intelligent" will not help until these are fixed.

---

## Finding 1 (dominant): the perception-action loop is open-loop

Defaults are `brain_tick_stride = 10`, `vision_stride = 10`
(`config.rs:163-169`), so the entire sensory frame refreshes once per
100 physics ticks. With `dt = 1/30` (`buffers.rs:424`) and
`movement_speed = 20`, an agent travels **~67 units between vision updates —
more than 2× the entire 30-unit vision range** (`VISION_MAX_DIST`,
`common.wgsl:204`).

Worse, all 10 brain decisions within a batch see byte-identical `s_features`:
not just vision but velocity, facing, touch, and even the interoceptive
energy/integrity features are packed only by the vision pass
(`phase_vision.wgsl:173-222`), which runs once per batch
(`gpu_kernel.rs:1423-1429`).

Chasing food is a closed-loop behavior: turn, see the bearing error shrink,
correct. That feedback channel does not exist here — by the time a food
sighting reaches the policy, the agent is typically past it. This is why
behavior looks "definitely not random but never converging": the TD machinery
is genuinely correlating noise with reward, but the state it conditions on is
decorrelated from the consequence it is blamed for.

The codebase already half-knows this: `kernel_tick.wgsl:515-523` flags the lag
as the failure mode behind the circling investigation (issue #115), and the
`vision_width` doc comment in `config.rs:73-77` defers better vision "until
the learner can act on directional vision."

Evolution makes this worse, not better: `movement_speed` is heritable up to
100, and fitness rewards exploration coverage, so selection actively pushes
toward faster (= blinder) agents.

## Finding 2: there is no survival instinct because death is free and invisible

Mechanically:

- On death, `agent_death_respawn` zeroes the eligibility traces and
  `O_PREV_VALUE` (`kernel_tick.wgsl:383-391`) before the next brain tick. The
  transition *into* death is therefore never evaluated — no negative TD error
  ever propagates from dying. The comment says "credit must never leak across
  the death boundary," which is correct for the *new* episode, but as written
  it also discards the terminal lesson of the *old* one.
- Respawn happens in the same kernel cycle with **full energy restored**. From
  the within-lifetime learner's perspective, a starving agent that walks into
  a hazard and dies executes the best homeostatic move available — instant
  +95% energy at no felt cost.
- The only anti-death pressure is the governor's
  `survival = 1/(1 + 0.5·deaths)` term (`governor.rs:473`), which acts on
  hyperparameter evolution, not on policy weights within a lifetime.

The fix that preserves the design philosophy is not an innate fear module —
it is making death *felt*: apply one terminal update with
δ = −`MAX_TD_ERROR` through the existing traces in `agent_death_respawn`
**before** zeroing them. That single change gives "danger=bad" a learnable
gradient, because the integrity-drain reward alone (≈ −0.02 per brain tick in
hazard at defaults) is weak and, per Finding 1, gets blamed on stale states.

## Finding 3: agents cannot feel present danger at decision time

`TOUCH_HAZARD` and `TOUCH_TERRAIN_EDGE` are defined (`common.wgsl:213-214`)
but never written — the touch packer only emits food and agent contacts. And
since the interoceptive features (energy, integrity, and their deltas) ride
the once-per-batch vision pass, the *current-cycle* integrity drain enters the
brain only through the reward and urgency scalars, never through the state.
The agent literally does not know it is standing in a hazard right now; it
knows it was hurting up to 100 ticks ago, somewhere else.

A cheap, high-leverage fix: in `coop_feature_extract`, read energy/integrity
directly from `physics_state`. The homeostasis pass
(`brain_passes.wgsl:148-183`) already does same-cycle reads of exactly these
fields, so the precedent and barrier-safety pattern exist.

## Finding 4: the food gradient is invisible until contact, and the bridge is thin

Reward is purely the homeostatic delta — approaching food pays exactly zero
until the eat event (+0.2 normalized, one tick). The critic must bridge that
gap by bootstrapping, but its effective per-dimension step is
`CRITIC_LEARNING_RATE × TD_VECTOR_SCALE` = 0.01/128 ≈ 8e-5, and the actor
learns only from rank-1 correlations between zero-mean noise kicks and δ
(REINFORCE-style traces, `brain_passes.wgsl:675-700`) through a frozen-ish
random Xavier encoder. That is workable but slow under clean conditions — and
conditions are not clean (Findings 1–3).

Minor: the `TD_DISCOUNT` comment (`common.wgsl:280-282`) claims the
33-brain-tick horizon "matches travel time from the edge of vision range to
food" — actual travel time is ~4.5 brain ticks (30 units at 20 u/s = 45
physics ticks). Harmless, but the calibration rationale is off ~7×.

## Finding 5: the outer loop's definition of intelligence is satisfiable without learning

Fitness is `0.4·survival + 0.3·foraging + 0.3·exploration`
(`governor.rs:471-480`). The initial forward bias (`buffers.rs:34`) +
exploration noise + klinotaxis already produce a competent random-walker that
covers cells, stumbles into food, and keeps its accumulated
`food_count`/`ticks_alive` across cheap deaths. That is a strong local optimum
scoring roughly 0.4–0.6 with zero learning, so generation-over-generation
selection — including the Lamarckian champion-weight inheritance in
`evolution.rs:438-460` — mostly propagates *good random-walk
hyperparameters*, not learned policies.

If the governor should select for intelligence in the intended sense, measure
*improvement within a lifetime* — e.g., food rate or time-to-food in the last
quarter of a generation versus the first — or make survival multiplicative
rather than additive so kamikaze foraging stops paying.

## Smaller mechanical issues (worth a pass later, not root causes)

- The recorded eligibility (`s_explore`, `brain_passes.wgsl:633-634`) is
  captured before fatigue and klinotaxis scale the executed action (×0.1–1.0
  and ×0.3–3.0), so credit magnitude mismatches what was actually done; sign
  is preserved.
- Memory stores a pattern *every* brain tick with valence = the instantaneous
  gradient (≈ −0.001 metabolic drain for almost all ticks), so the
  valence-weighted motor blend is mostly inert-to-slightly-contrarian; only
  eating-tick memories carry meaningful valence, and their state keys are
  stale per Finding 1.
- The vision alpha channel is constant 1.0 for every outcome including sky —
  25% of color features are dead weight.
- Urgency-amplified reward makes the reward function non-stationary with
  respect to internal state the critic cannot currently observe (ties back to
  Finding 3).

---

## Recommended experiments, in order

1. **Lag isolation**: `vision_stride = 1`, `brain_tick_stride = 2` (lag 2
   ticks), default speed, small population. If approach behavior emerges
   within a generation, Finding 1 is confirmed as dominant; then walk the
   strides back up to find the throughput/learnability frontier. Costs only a
   config change.
2. Same-cycle interoception in `coop_feature_extract` (Finding 3, small WGSL
   change).
3. Terminal death δ before trace zeroing (Finding 2, small WGSL change in
   `agent_death_respawn`).
4. Wire `TOUCH_HAZARD`/`TOUCH_TERRAIN_EDGE` contacts.
5. Rework fitness toward measured improvement (Finding 5).

## Direct answers to the framing questions

- **Is it an intelligence problem? A definition problem?** No. Homeostatic TD
  learning is a legitimate minimal definition of "figure the environment out
  on your own." It is an *observability and contiguity* problem: the agents
  act on a 3.3-second-old frozen snapshot of the world, so cues and
  consequences are decorrelated beyond what any learner of this class can
  recover.
- **Is it a missing survival instinct?** Yes — but not as an innate module
  that would have to be hand-wired (which would betray the design goal). It is
  that death must be felt by the learner (terminal negative TD) and must be
  costly (respawn should not be a free full-energy reset for the same
  continuing brain). With those in place, "danger=bad" is learnable from
  experience — which is the point.
