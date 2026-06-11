# Emergent Learning Pathway: TD Credit, Encoder Self-Supervision, Vision Acuity

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give agents a credit pathway that can actually assign a sparse, delayed, interoceptive reward (eating) to the distal navigational action that earned it (turning toward food 30 brain ticks earlier), and an encoder that becomes spatially discriminative without needing a working policy first. This is the mechanism gap that keeps issue #13 open after all suppression bugs (#14–#17, #87–#100) were fixed.

**Issue:** #13

**Reference:** `EVOLUTION_JOURNEY.md` §19–20, `docs/superpowers/specs/2026-04-21-credit-assignment-audit.md`, `docs/reviews/2026-04-15-gemini-31-pro.md`, `docs/reviews/2026-04-15-gpt-54-high.md`.

---

## Evaluation of the four candidate mechanisms

The investigation on #13 surfaced four missing mechanisms. Verdicts:

### 1. Long-horizon credit assignment — ADOPT (core of this plan)

The current learner is REINFORCE with an `exp(-age * 0.3)` eligibility window
(~7 meaningful brain ticks) against an action→reward delay of ~30 brain ticks
(food visible at `VISION_MAX_DIST = 30`, ~1 unit of travel per brain tick at
defaults). The decisive turn is outside the window when the energy spike
fires. PR #102's own conclusion: "The REINFORCE credit signal has too low SNR
for spatial learning."

Two options were considered:

- **Longer decaying trace over the existing history ring.** Rejected: REINFORCE
  variance grows with horizon, the ring loop is the serial thread-0 bottleneck
  that caused the #95 perf regression, and the #13 comment's arithmetic
  (~64M food encounters to break even) gets *worse* with more random terms.
- **TD(λ) critic + advantage-based actor updates.** Adopted. A linear value
  head over `s_encoded` learns "states with food in view are worth more";
  the TD error δ then propagates the terminal eating reward backwards along
  approach trajectories *across episodes*, and per-dimension eligibility
  traces carry credit to actions ~1/(1−γ) ≈ 33 brain ticks back at O(128)
  cost per tick — fully parallel, no history ring, no serial loop.

This preserves the project philosophy: the reward is still only the
homeostatic gradient. The value function is learned, not hardcoded.

### 2. Proximal approach signal — SUBSUMED by the critic; anticipation deferred

Approaching food currently yields zero gradient; only eating does. The TD
error δ fixes this for *credit* purposes: once the critic learns that
food-in-view states have higher value, any action that increases proximity
produces positive δ immediately. Re-enabling prospective motor blending
(removed in PR #100 for the `s_habituated`/`s_encoded` scale mismatch, dead
constant deleted in #140) is a separate *motor-side* feature. Deferred to a
follow-up issue, gated on Phase 1+2 results showing predictor quality is the
remaining bottleneck.

### 3. Self-supervised encoder — ADOPT (tied-weight reconstruction)

The encoder is Xavier-random at birth and trained only by action credit
(`ENCODER_CREDIT_SCALE`), which is the same degenerate signal it is supposed
to fix — the chicken-and-egg from `EVOLUTION_JOURNEY.md` §20B. Tuning the
scale was tried three times (0.001 → 0.01 → 0.1) without spatial features
emerging. A reconstruction objective (decode `s_encoded` back to `s_features`
through the transposed encoder weights) is dense from tick 0, needs no
working policy, and guarantees information preservation: food-left and
food-right *must* encode differently to reconstruct differently.
Alternatives rejected: pure predictability objectives collapse to constants;
contrastive losses need negative sampling that does not fit the one-agent
workgroup model.

### 4. Vision acuity — ADOPT at 16×12; 32×24 and stride changes measured separately

At 8×6 over 90° FOV, ray spacing at distance 20 is ~3.9 units against a food
diameter of 2 — distant food is usually invisible (issue #14: "Depth won't
fix this"). At 16×12 (192 rays) spacing at distance 20 is ~2.0 units —
comparable to the food diameter, so at least one ray reliably hits.
16×12 stays under the 256-thread `vision_tick` workgroup (no strided-loop fix
needed) and `s_features` grows to ~985 floats ≈ 3.9 KB of workgroup memory —
well inside the 16 KB default limit. 32×24 (768 rays, ~15.5 KB shared memory)
is deferred: it needs the `vision_tick` strided loop from the dynamic-vision
plan and a workgroup-storage budget audit. `vision_stride` (default 10 —
only ~3 distinct visual frames per food approach) is a measured sweep in
Phase 3, not a blind change.

### What stays untouched

- **Klinotaxis stays.** It is the reactive substrate that generates correlated
  approach/escape trajectories for the critic to learn values along.
- **Pattern memory, valence learning, memory blend stay** as-is in this plan.
- **The reward definition stays interoceptive-only** (`raw_gradient`).

---

## Architecture

```
                    s_features (per brain tick, one vision frame per batch)
                        │
                  encoder (trainable: reconstruction + value-aligned credit)
                        │
                    s_encoded ──────────────┬─────────────────┐
                        │                   │                 │
                  policy (actor)      value head (critic)   pattern memory
                  fwd/turn weights    v = w·s + b           (unchanged)
                        │                   │
                  motor + noise       δ = r + γ·v' − v   ← r = raw homeostatic gradient
                        │                   │
                  actor traces  ←──── δ ────┴──→ critic trace
                  z_fwd, z_turn       (replaces improvement/deadzone/tonic/PAIN_AMP)
```

Per brain tick:

```
z_critic = γλ·z_critic + s_encoded            (128 floats)
z_fwd    = γλ·z_fwd    + noise_fwd·s_encoded  (128 floats)
z_turn   = γλ·z_turn   + noise_turn·s_encoded (128 floats)
δ        = clamp(r + γ·v(s') − v(s), ±MAX_TD_ERROR)
Δv_w     = CRITIC_LR · δ · z_critic
Δw_fwd   = ACTOR_LR  · δ · z_fwd
Δw_turn  = ACTOR_LR  · δ · z_turn
```

All trace and weight updates are per-dimension → threads 0..127 in parallel.
The 64-entry history ring, its thread-0 serial credit loop, and the
`DEADZONE` / `TONIC_CREDIT_SCALE` / `PAIN_AMP` / `CREDIT_DECAY` branch are
removed. δ replaces `improvement` as the single credit signal.

**Constants (named per CONTRIBUTING.md, each with a why-comment):**

| Constant | Value | Why |
|---|---|---|
| `TD_DISCOUNT` | 0.97 | horizon 1/(1−γ) ≈ 33 brain ticks ≈ travel time from `VISION_MAX_DIST` at default speed |
| `TD_LAMBDA` | 0.9 | trace decay; effective credit span γλ ≈ 0.87 per tick reaches the full approach |
| `CRITIC_LEARNING_RATE` | 0.01 | 10× slower than actor — the critic must be stabler than the policy it evaluates |
| `MAX_TD_ERROR` | 1.0 | bounds δ against respawn/clamp artifacts (mirrors `MAX_HOMEOSTATIC_DELTA` intent) |
| `MAX_VALUE_WEIGHT_NORM` | 2.0 | same budget as action weights |
| `ENCODER_RECON_RATE` | 0.001 | reconstruction is dense (every tick, every dim) so the per-step rate must be small |

**Known limitation (documented, accepted):** vision runs once per kernel
batch, so `s_encoded` is constant within a batch (`vision_stride` brain
ticks). TD steps within a batch see a static state; state transitions are
batch-granular. δ still spikes on the eating tick because `r` reads
physics-fresh energy. This is the existing one-batch sensory lag invariant;
Phase 3 measures whether tightening `vision_stride` pays for its cost.

**Risks and mitigations:**

| Risk | Mitigation |
|---|---|
| Critic diverges (linear TD with function approximation) | clamp δ, L2-ball value weights, slow critic LR; Phase-0 probe catches it |
| Critic can't see food on random encoder | random projections of 240 vision floats keep "food present" linearly decodable; Phase 2 reconstruction sharpens it; probe measures both orders |
| Layout change breaks inheritance | `BRAIN_STRIDE` changes; length-mismatch import guards must log and skip (journey lesson 15), evolution DB weight blobs invalidated by documented version bump |
| Repeat of #94/#97/#99 oscillation | every phase lands with before/after probe numbers in the PR body; no constant tuning without a measurement |

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/xagent-brain/src/shaders/kernel/common.wgsl` | Modify | new offsets (`O_VALUE_WEIGHTS`, `O_VALUE_BIAS`, `O_PREV_VALUE`, `O_TRACE_CRITIC`, `O_TRACE_FWD`, `O_TRACE_TURN`), new constants, delete `DEADZONE`/`TONIC_CREDIT_SCALE`/`PAIN_AMP`/`CREDIT_DECAY`/history offsets |
| `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` | Modify | pass 6 credit block → TD update; pass 7b encoder reconstruction; delete history-ring writes |
| `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl` | Modify | zero traces + `O_PREV_VALUE` on death (life-boundary discontinuity) |
| `crates/xagent-brain/src/buffers.rs` | Modify | mirror offsets, `BRAIN_STRIDE`, init (value head small-random), drop history init |
| `crates/xagent-brain/src/gpu_kernel.rs` | Modify | buffer sizing, telemetry (expose `value`, `td_error`), remove history buffer plumbing |
| `crates/xagent-sandbox/tests/integration.rs` | Modify | Phase-0 probes + per-phase learning tests |
| `crates/xagent-sandbox/src/headless.rs` | Modify | A/B metric logging (food-per-life, weight norms, δ stats) |
| `crates/xagent-shared/src/config.rs` | Modify (Phase 3) | default `vision_width`/`vision_height` 8×6 → 16×12 |

---

## Phase 0: Measurement harness (prerequisite — nothing merges without it)

The audit's core lesson: every constant change was individually plausible and
collectively wrong, and the two changes that fully froze learning passed
review. So the gates come first.

- [x] **Step 1: Directional learning probe (integration test).** Built as a
  16-agent flat-world arena (4×4 grid, 64-unit spacing — beyond vision
  range, so independent trials), one food item per agent at bearing
  ±atan(3/7) (exactly on a ray column) at distance 5 — the distance the
  default 8×6 ray rows can actually see ground food (the vertical ray
  layout, not horizontal acuity, binds visibility; see the baseline note).
  Single-tick strides + zero movement speed pin the geometry; turn-sign
  correctness is scored against the per-tick bearing. Baseline asserted to
  the chance band [0.38, 0.62]. Includes a separate information-path test
  asserting all agents see a food pixel.
- [x] **Step 2: Foraging-rate metric.** Free-running probe arena with the
  default config: food-per-agent-per-1k-ticks and deaths over 3000 ticks,
  with liveness accounting asserted.
- [x] **Step 3: Headless A/B logging.** `run_headless` now prints per
  generation: food, deaths, food-per-life, and the best agent's policy
  weight norms (`w_fwd`, `w_turn`).
- [x] **Step 4: Record baseline numbers** —
  [`docs/superpowers/specs/2026-06-10-learning-baseline.md`](../specs/2026-06-10-learning-baseline.md):
  alignment 0.498 (chance), foraging 0.042 food/agent/1k-ticks, visibility
  16/16.

## Phase 1: TD(λ) critic and trace-based actor credit — DONE

- [x] **Step 1: Layout.** Added the value head + trace regions to
  `common.wgsl` and `buffers.rs` (single canonical derivation). Removed the
  history layout and the entire `history_buffer` binding/plumbing in
  `gpu_kernel.rs` (binding 13 retired; layout intentionally non-contiguous).
  `AgentBrainState` dropped its `history` vector.
- [x] **Step 2: Critic forward + TD error.** Pass 6 reduces
  `v = dot(value_weights, s_encoded) + bias` across `ENCODED_DIMENSION`
  threads; thread 0 forms `δ = clamp(r + TD_DISCOUNT·v − prev_value,
  ±MAX_TD_ERROR)` with `r = s_homeo[1]` (urgency-amplified raw gradient),
  stores `v` into `O_PREV_VALUE`, publishes δ via `s_td_error`.
- [x] **Step 3: Trace + weight updates.** Per-dimension threads decay traces
  by `TD_DISCOUNT·TD_LAMBDA`, accumulate critic (`+s_encoded`) and actor
  (`+noise·s_encoded`) terms, and apply `Δ = lr·TD_VECTOR_SCALE·δ·z`. The
  motor block publishes its exploration noise to `s_explore` so the trace
  update (end of pass) sees this tick's action. `s_credit[d] = δ·(z_fwd+z_turn)`
  feeds the encoder. Value head gets the same L2-ball clamp as the actor.
- [x] **Step 4: Delete the old branch.** Removed `DEADZONE`,
  `TONIC_CREDIT_SCALE`, `PAIN_AMP`, `CREDIT_DECAY`, `ACTION_HISTORY_LEN`,
  `ACTION_WEIGHT_DECAY`, the history-ring offsets, and the serial credit
  loop. READMEs and CLAUDE.md updated; no dangling references remain.
- [x] **Step 5: Death boundary.** Traces and `O_PREV_VALUE` zeroed in both
  `phase_death.wgsl` and the fused `kernel_tick.wgsl` respawn path.
- [x] **Step 6: Inheritance.** Value weights live in `brain_state` and ride
  the existing inherit/mutate path; `mutate_brain_state`'s `FIXED_TAIL_SIZE`
  math absorbs the larger tail unchanged.
- [x] **Step 7: Tests.** `td_critic_tracks_metabolic_drain` (value goes
  negative under drain, δ within clamp), `td_traces_bounded_across_deaths`,
  and the gate `learning_probe_td_learns_turn_alignment`. `fmt`, `clippy
  -D warnings`, 128 tests green.

**Gate — PASSED.** Trained turn/bearing alignment **0.643** (was 0.498 at
chance), clearing the 0.62 band edge; training food rises 588→730 across
halves; TPS unchanged (serial loop gone). Numbers recorded in the baseline
spec. One adjustment vs the plan: per-dimension trace updates needed a
`TD_VECTOR_SCALE = 1/ENCODED_DIMENSION` factor to keep the bootstrapped
critic inside the linear-TD stability limit (the aggregate step is a sum of
128 trace×feature products), and per-tick weight decay was dropped (it bled
away the learned policy and the initial forward bias; δ being
surprise-driven plus the L2 ball already bound magnitude).

## Phase 2: Encoder self-supervision (tied-weight reconstruction)

- [ ] **Step 1: Reconstruction pass.** In pass 7b, all 256 threads
  cooperatively compute `x̂ = Wᵀ·s_encoded` over `FEATURE_COUNT` (strided
  loop), then update `Δw[j][d] = ENCODER_RECON_RATE · (x[j] − x̂[j]) ·
  s_encoded[d]`, clamped to the existing encoder weight budget. Keep the
  δ-driven encoder credit path (now secondary).
- [ ] **Step 2: Separability test.** Build two synthetic sensory frames
  (food-pixels left vs right), tick the brain T times on alternating
  frames, assert encoded-state cosine similarity *decreases* from its
  random-init value (falsifiable: frozen encoder keeps it constant).
- [ ] **Step 3: Probe re-run + cost check.** Reconstruction adds one
  encoder-sized matmul per brain tick; measure TPS. Commit with numbers.

**Gate:** separability test passes; directional probe improves over Phase 1;
TPS cost < 15%.

### Phase 2 outcome — NOT MERGED (documented negative result)

Tied-weight vision reconstruction was implemented (vision-only target, decode
through the transposed encoder weights, `Δw = rate·(x−x̂)·encoded`, vision
features only) and measured against the Phase-1 directional gate on the
fixed training seed:

| Reconstruction rate | Trained alignment | Phase-1 reference |
|---|---|---|
| 0.001 | 0.603 | 0.643 |
| 0.0003 | 0.613 | 0.643 |
| 0 (Phase 1) | 0.643 | 0.643 |

Reconstruction was **neutral-to-slightly-negative** at every rate tried and
never cleared the gate's "improves over Phase 1" bar; lowering the rate only
walked the metric back toward the Phase-1 value. (Training-time foraging rose
slightly — 588→730 vs 609→770 — but the post-training behavioral readout did
not.) Two compounding reasons:

1. **The encoder is not the currently-binding constraint.** Phase 1 already
   reaches 0.643 ≫ chance with the random-projection encoder, so the
   8×6 random projection preserves enough food-direction signal for the
   policy. There was little representational headroom for reconstruction to
   add.
2. **Moving-target cost.** Reshaping the encoder while the policy reads it
   makes the policy chase a shifting code, which slightly slowed policy
   convergence — visible as the rate-dependent dip (0.001 worse than 0.0003).

Per the plan's own discipline ("No phase merges on 'should work'"), the
reconstruction code was reverted; `develop`/this branch keep the validated
Phase-1 learner. The trainable-encoder idea is not refuted in general — it is
not worthwhile *at 8×6 with TD(λ) already extracting the signal*. It should be
revisited only if a later phase makes the encoder the binding constraint (e.g.
much higher vision resolution where a random projection dilutes a small food
signal across many pixels), and then with an objective that emphasizes the
*varying* part of the input rather than dominant background variance.

**Re-scope:** proceed directly to Phase 3 (vision acuity), the next plausibly
binding constraint — can the agent see food at range at all.

## Phase 3: Vision acuity — DONE (+ a more important discovery)

- [x] **Step 1: Default 17×13** (better than the planned 16×12: odd counts
  put a ray row on the horizon and a column straight ahead). Verified the
  layout-aware paths end-to-end; pinned-count tests rebased onto the live
  layout so default and reference (8×6) are both covered. Feature vector
  grows 265 → 1130.
- [x] **Step 2: Range-visibility test.** `vision_horizon_row_sees_food_at_range`
  places food at {5,10,15,20,25}: 17×13 sees all five, 8×6 sees only the
  nearest — proving the horizon row fixes the distal-food blindness.
- [x] **Step 3: Evolution-scale measurement.** 16-generation run at 17×13:
  foraging rate 0.192 → 0.254, comparable to 8×6 (no clear win — the learner
  can't yet use directional vision, so the added information isn't cashed in).

**Gate result.** Food visible at range: **met**. "Rising food-per-life
trend": already validated at evolution scale in Phase 1 (foraging rate +74%,
fitness +62%) and reconfirmed at 17×13.

**The unplanned discovery (more important than the acuity work).** Validating
the directional probe at the new resolution exposed that the Phase-1 "0.643
learned to turn toward food" was **confounded**: each agent saw food on one
fixed side in both training and eval, so a per-agent constant turn bias scored
above chance without any vision-conditional steering. The confound-free probe
(`learning_probe_mirrored_steering_is_chance`, food side mirrored every
training episode) sits at **0.52 — chance**, and more episodes don't move it.
**Genuine vision-conditional steering is not being learned at any resolution.**
This re-opens the encoder/representation question (Phase 2 had dismissed it
using the confounded metric): a random-projection encoder likely does not make
"food-left" vs "food-right" linearly separable for the policy readout, so a
constant bias is learnable but a conditional response is not. The TD(λ) critic
and the evolution-scale foraging/fitness/survival gains stand; the directional
claim does not. See the baseline spec's Phase 3 section.

## Phase 4 (deferred — separate issues, not in this plan's scope)

- Prospective/anticipation motor blending retrained in `s_encoded` space.
- 32×24 vision (needs `vision_tick` strided loop + workgroup-storage audit).
- Adaptive normalization of δ (the audit's "adaptive thresholds" ask, now
  reduced to one signal instead of two constants).
- Neuroevolution of action/value weights (journey "Path B").

---

## Validation discipline

Every phase is its own PR with: baseline → after numbers from the Phase-0
probes in the PR body, fixed seeds, and CI green (`fmt`, `clippy
-D warnings`, full suite). No phase merges on "should work" — journey rule 7.
