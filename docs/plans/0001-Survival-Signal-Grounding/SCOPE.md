# Scope — Plan 0001

> Make "danger = bad" learnable by the within-lifetime TD(λ) learner — today
> the death event carries zero learning signal, hazard is invisible to touch,
> and interoception arrives one vision batch late — and extend the measurement
> harness so danger avoidance and within-lifetime improvement become pinned,
> falsifiable numbers like the foraging probes already are.

## Why this plan

This plan synthesizes the four 2026-06-12 learning-loop reviews
(`docs/reviews/2026-06-12-{claude-fable-5,gemini-31-pro,gpt-5-codex,grok-43}.md`).
The reviews were written against different snapshots, so every load-bearing
claim was re-verified against `develop` @ `0260f7e` before being adopted.
Claims that did not survive verification are recorded at the bottom of this
section. The question under review: agents show no consistent danger
avoidance or food chasing after many iterations — is it an intelligence
problem, a definition problem, or a missing survival instinct?

The verified answer: the learner is sound in principle, but the danger side
of the world is unlearnable as experienced. Specifically:

1. **Death teaches nothing.** On death, `agent_death_respawn` zeroes the
   eligibility traces and `O_PREV_VALUE` with no terminal update
   (`kernel_tick.wgsl:383-391`), restores full energy in the same kernel
   cycle (`kernel_tick.wgsl:337`), and runs *before* the brain tick
   (`kernel_tick.wgsl:474-507`) — so the transition into death is never
   evaluated. A starving agent that walks into a hazard and dies experiences
   it as a free heal. The only anti-death pressure is the governor's
   fitness term, which acts on evolution, not on policy weights within a
   lifetime.
2. **Hazard has no touch grounding.** `TOUCH_HAZARD` and
   `TOUCH_TERRAIN_EDGE` are defined (`common.wgsl:213-214`) but the live GPU
   sensory path emits only food and agent contacts
   (`phase_vision.wgsl:224-311`). The CPU reference path
   (`agent/senses.rs::detect_touch`) emits both missing tags. Food has three
   timely grounding paths (lime visual cue, touch contact, +energy spike);
   danger has one lagged visual cue plus a slow integrity drain.
3. **Interoception is stale at decision time.** Energy, integrity, and their
   deltas enter the feature vector only via the once-per-batch vision pass
   (`phase_vision.wgsl:206-222`), so the state the policy/critic conditions
   on is up to `vision_stride × brain_tick_stride` = 100 physics ticks old —
   while the reward/urgency scalars read `physics_state` same-cycle
   (`brain_passes.wgsl:148-183`). The agent literally does not know it is
   standing in a hazard right now; it knows it was hurting up to 100 ticks
   ago, somewhere else.
4. **The sensory frame is frozen and the agent outruns it.** All
   `vision_stride` brain decisions in a batch see byte-identical features,
   stale by one full batch (`kernel_tick.wgsl:509-523`; the lag-100 default
   is pinned by a test at `config.rs:497`). At `movement_speed = 20` an
   agent travels ~67 units per vision refresh versus a 30-unit vision range
   (`common.wgsl:204`), and speed is heritable only upward
   (`config.rs:105-108`).
5. **The outer loop is satisfiable without learning.** Fitness is
   `survival·0.4 + foraging·0.3 + exploration·0.3` with
   `survival = 1/(1+0.5·deaths)` (`governor.rs:473-479`), while
   `food_count`/`ticks_alive` persist across free respawns
   (`kernel_tick.wgsl:318-348`) — kamikaze foraging pays, and a competent
   random walker with klinotaxis scores respectably with zero cue-conditioned
   learning.
6. **The predictor trains the wrong objective.** Pass 7a regresses *this
   tick's* prediction against *this tick's* input — an identity-autoencoder
   objective (`brain_passes.wgsl:719-730`) — while the novelty error that
   drives exploration compares *last tick's* prediction against the current
   state (`brain_passes.wgsl:323-330`). The trained quantity is not the
   measured quantity.
7. **No danger-side measurement exists.** The probe harness
   (`tests/integration.rs`, Learning Probe Tests) pins food-approach and
   foraging baselines in hazard-free arenas; nothing measures hazard-exit
   latency or death economics, so none of the above can land behind a gate
   today.

One critical piece of context the reviews lacked: the confound-free steering
probe (`learning_probe_mirrored_steering_is_chance`) trains at
`brain_tick_stride = 1, vision_stride = 1` — sensory lag of one tick — and
still lands at chance (0.52), and encoder separability was measured twice as
non-binding (baseline spec, Phase 2 revert + separability diagnostic). So
"remove the lag and learning appears" — the dominant hypothesis shared by
all four reviews — is already partially falsified. Lag removal is necessary
for the open world but not sufficient for steering; finding 4 is therefore
treated as a budgeted experiment, not an expected fix.

**Review claims rejected during verification** (recorded so they are not
re-litigated):

| Claim | Source | Why rejected |
|---|---|---|
| Architecture is REINFORCE over a 64-slot history ring with `CREDIT_DECAY`/`PAIN_AMP` | gemini-31-pro (and the original grok-43, since re-issued) | Replaced by the TD(λ) actor-critic (`brain_passes.wgsl:368-430`, "no history ring") |
| The encoder already uses an Oja/PCA rule | gpt-5-codex | Not on `develop`; that review examined a worktree with Gemini's patch applied |
| Fix the encoder with Oja's rule | gemini-31-pro | Encoder self-supervision was implemented and reverted as a measured negative (plan 2026-06-10, Phase 2); `encoder_food_side_separability_diagnostic` shows the random encoder preserves food-side separability at a ≈4× margin |
| Depth pixels are dropped from the feature vector | grok-43 (re-issued) | False: `coop_feature_extract` copies color and depth 1:1 into `s_features` (`brain_passes.wgsl:73-80`) |
| Lag isolation will confirm sensorimotor lag as the dominant cause | claude-fable-5, gpt-5-codex | The mirrored probe already trains at lag 1 and stays at chance — see above |

## In scope

Work items in [TASKS.md](TASKS.md) (workstreams 0001–0004):

- **0001 — Hazard observability and baseline.** Expose the non-visual
  sensory tail (touch contacts, interoception) in `AgentTelemetry`; build a
  half-plane danger arena probe measuring hazard-exit latency and deaths;
  pin the untrained baseline with falsifiable bands, house-style.
- **0002 — Survival-signal grounding.** Three small WGSL changes, each
  independently testable: a terminal TD update through the dying episode's
  eligibility traces before they are zeroed; `TOUCH_HAZARD` /
  `TOUCH_TERRAIN_EDGE` contacts emitted by the live GPU sensory path
  (ported from the CPU reference); energy/integrity features read
  same-cycle from `physics_state`. Then re-measure the hazard probe and
  record the verdict.
- **0003 — Learning visibility and lag economics.** Quarter-split
  within-lifetime food-rate metric (`Learn q1→q4`) in headless runs; a
  three-arm stride/lag sweep at evolution scale with a TPS budget and a
  decision rule; the mis-calibrated `TD_DISCOUNT` comment corrected.
- **0004 — Gated follow-ups.** Fitness rework (two fully specified variants;
  the quarter-metric and hazard data pick which) and the predictor
  forward-model objective fix (verified bug, honestly small expected effect,
  lands Phase-2-style: keep only if nothing regresses).

## Origin → workstream mapping

| Finding | Addressed by |
|---|---|
| No danger-side measurement (7) | `0001` |
| Death teaches nothing (1) | `0002` |
| Hazard has no touch grounding (2) | `0002` |
| Interoception stale at decision time (3) | `0002` |
| Frozen/lagged sensory frame (4) | `0003` (budgeted sweep) |
| Fitness satisfiable without learning (5) | `0003` (metric), `0004` (formula) |
| Predictor objective mismatch (6) | `0004` |

## Locked decisions

- **Death becomes a lesson, not a module.** The terminal update applies
  `δ = −MAX_TD_ERROR` through the *existing* traces in `agent_death_respawn`
  before zeroing them — no innate fear circuitry, no new drive. The
  magnitude is bounded by the same per-transition clamp that protects
  against artifacts. The free full-energy respawn itself (world economics)
  is explicitly untouched; if the terminal lesson proves insufficient,
  respawn economics is a separate, gated follow-up.
- **Hazard touch mirrors the CPU reference and takes slot priority.** Zero
  planar direction (the hazard is the ground underfoot), fixed intensity
  (`TOUCH_HAZARD_INTENSITY = 0.5`), tag `3/4` — exactly `detect_touch`'s
  semantics. It is emitted *first* so present-moment damage can never be
  evicted when the four contact slots fill. Edge contacts point inward with
  closeness intensity, matching the CPU path.
- **Same-cycle interoception changes the delta semantics deliberately.**
  The packed deltas were "change across one batch, one batch late"; they
  become "change across the last physics sub-tick, now" — which is the
  sub-tick that contains any eat event or hazard damage from the current
  cycle. The `sensory_buffer` slots keep being packed for CPU readback;
  only the brain stops consuming them.
- **The stride sweep is a budgeted A/B, not a confirmation exercise.**
  Adopt the smallest-lag configuration only if its ticks/sec cost versus
  control is under ~30% *and* it beats control on the fixed seed. Any
  default-stride change recalibrates `TD_DISCOUNT` in the same commit to
  preserve the real-time horizon. A negative result is recorded as the
  answer to the reviews' lag experiment.
- **Fitness changes are data-gated, one change per measurement.** Kamikaze
  confirmed (deaths rise with foraging despite the terminal lesson) →
  multiplicative survival. Learning invisible (`q1→q4` flat while
  cross-generation foraging rises) → improvement term. Both confirmed →
  multiplicative first, improvement second, separately measured.
- **The predictor fix claims consistency, not fireworks.** Without an
  efference copy the best forward model in a mostly-static world is close
  to identity anyway; the fix aligns the trained objective with the
  measured one and is kept only if no probe regresses (the same discipline
  that reverted Phase 2 reconstruction).
- **The encoder is not touched.** Two measured negatives (Phase-2 revert,
  separability diagnostic) put encoder changes behind a future probe
  regression that implicates representation — not behind review opinion.
- **Measurement first, always.** Nothing in workstream 0002+ merges without
  before/after probe numbers (hazard probe, mirrored steering band,
  foraging baselines) in the PR body, fixed seeds, and green
  `fmt`/`clippy`/`test` — the discipline established by the 2026-06-10 plan.

## Out of scope

- **Efference copy for the predictor** (motor as predictor input) — the
  principled completion of the forward model; layout change that invalidates
  inheritance blobs. Justified only by Task `predictor-forward-objective`
  data showing prediction error dominated by self-motion.
- **Memory store gating** (store only on |valence| above the metabolic noise
  floor or on eat/damage events) — today every brain tick stores a pattern
  with valence ≈ −0.001, churning the 128 slots in ~42 s. Needs a
  recall-valence separation diagnostic first.
- **Eligibility magnitude alignment** (record noise after fatigue/klinotaxis
  scaling) — sign is preserved today, only magnitude mismatches; revisit
  once anything moves the mirrored probe off chance.
- **Exploration floor decay** — masks learned policy if one exists;
  currently nothing to mask per the mirrored probe.
- **Vision alpha channel** (constant 1.0, 48 dead features of 192) — bundle
  with the next vision-layout change (17×13 default flip), itself gated on
  directional steering existing.
- **Respawn energy economics** (partial-energy respawn) — see locked
  decisions; separate follow-up if the terminal lesson under-delivers.
- **Encoder changes of any kind** — see locked decisions.
- **17×13 vision default** — already measured (no win until steering works);
  capability retained.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
