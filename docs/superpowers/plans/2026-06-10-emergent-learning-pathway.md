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

- [ ] **Step 1: Directional learning probe (integration test).** Construct a
  `GpuKernel` with one agent and one food item placed at a fixed bearing
  (±30°, distance 15). Run N brain ticks, read motor telemetry, score
  turn-sign correctness against the food bearing over M seeded trials.
  Assert the *baseline* (current code) scores ≈ chance — this makes the
  probe falsifiable and gives the number future phases must beat. Gate
  behind the existing GPU-test mechanism.
- [ ] **Step 2: Approach-rate metric.** Same scaffold, free-running: count
  food-visible→distance-decreased transitions per 1k brain ticks, and
  food-per-life from `P_FOOD_COUNT`/`P_DEATH_COUNT`.
- [ ] **Step 3: Headless A/B logging.** Extend `run_headless` to log per
  generation: food-per-life, policy weight norms, exploration rate mean.
  Fixed seed → reproducible before/after comparison.
- [ ] **Step 4: Record baseline numbers** in the PR body and in a short
  `docs/superpowers/specs/` baseline note. Commit.

## Phase 1: TD(λ) critic and trace-based actor credit

- [ ] **Step 1: Layout.** Add the six new regions to `common.wgsl` and
  `buffers.rs` (single canonical derivation, no hardcoded strides). Remove
  `O_MOTOR_RING`/`O_STATE_RING` history layout and the `history_buffer`
  plumbing in `gpu_kernel.rs`. Net per-agent memory: −8514 floats (ring)
  +516 floats (head + traces).
- [ ] **Step 2: Critic forward + TD error.** In pass 6, threads 0..127
  compute `v = dot(value_weights, s_encoded) + bias` via the existing
  parallel-reduction idiom; thread 0 forms
  `δ = clamp(r + TD_DISCOUNT·v − prev_value, ±MAX_TD_ERROR)` with
  `r = s_homeo[1]` (raw amplified gradient — immediate signal per journey
  rule 3), stores `v` into `O_PREV_VALUE`, publishes δ via shared memory.
- [ ] **Step 3: Trace + weight updates.** Threads 0..127: decay traces by
  `TD_DISCOUNT * TD_LAMBDA`, accumulate (`z_critic += s_encoded[d]`,
  `z_fwd += noise_forward·s_encoded[d]`, `z_turn += noise_turn·s_encoded[d]`),
  apply `Δ = lr·δ·z` to value/forward/turn weights. Keep existing decay +
  L2-ball normalization for actor weights; add the same for value weights.
  `s_credit[d] = δ·(z_fwd[d]+z_turn[d])` keeps the encoder-credit interface.
- [ ] **Step 4: Delete the old branch.** Remove `DEADZONE`,
  `TONIC_CREDIT_SCALE`, `PAIN_AMP`, `CREDIT_DECAY`, `ACTION_HISTORY_LEN`,
  the phase-1/phase-2 credit loop, and the undocumented `* 0.1` bias
  multiplier (biases now update from δ·trace like weights). Grep all
  shaders and docs for dangling references per CONTRIBUTING.md.
- [ ] **Step 5: Death boundary.** Zero `O_TRACE_*` and `O_PREV_VALUE` in
  `phase_death.wgsl` — credit must not leak across lives (journey rule 4).
- [ ] **Step 6: Inheritance.** Include value weights in exported brain state;
  length-mismatch on import logs a warning and starts fresh (no silent skip).
- [ ] **Step 7: Unit + probe tests.** Critic converges to `r/(1−γ)` under
  constant reward; δ > 0 on an unexpected energy gain; traces zeroed on
  death; directional probe beats Phase-0 baseline with statistical margin;
  TPS within 10% of baseline (expected: improvement — the serial loop is
  gone). `cargo fmt`, `clippy -D warnings`, full test suite. Commit.

**Gate:** directional probe above chance, food-per-life ≥ baseline over a
fixed-seed 20-generation headless run, no TPS regression > 10%.

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

## Phase 3: Vision acuity

- [ ] **Step 1: Default 16×12.** Change config defaults; verify the
  layout-aware paths (`BrainLayout`, overrides, readback) end-to-end —
  this is config + tests, the WGSL already derives from `VISION_W/H`.
- [ ] **Step 2: Ray-hit instrumentation test.** Place food at distances
  {5, 10, 15, 20, 25}; assert hit-rate at d=20 materially exceeds the 8×6
  baseline.
- [ ] **Step 3: Measured `vision_stride` sweep.** Headless A/B at stride
  {10, 5} × resolution {8×6, 16×12}; record food-per-life vs TPS. Adopt
  the best point that keeps TPS acceptable; document the choice.

**Gate:** food visible at range in the probe; combined Phases 1–3 show a
rising food-per-life trend across generations — the first time this metric
has ever trended.

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
