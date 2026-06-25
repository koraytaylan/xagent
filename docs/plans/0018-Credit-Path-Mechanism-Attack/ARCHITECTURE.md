# Architecture — Plan 0018 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-shared/src/config.rs`, and
> `crates/xagent-sandbox/tests/integration.rs`.
> Line numbers are hints; locate by symbol (grep for `coop_predict_and_act`,
> `raw_gradient`, `TD_DISCOUNT`, `TD_LAMBDA`, `learning_probe_mirrored_steering_is_chance`,
> `encoder_food_side_separability_diagnostic`).

## 0001 — Auxiliary Steering Objective

Today the policy learns only through the homeostatic TD(λ) path: `raw_gradient =
energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT`
(`brain_passes.wgsl:913-915`) → TD error → eligibility-trace updates
(`brain_passes.wgsl:1247-1252`). No direct vision-to-action supervision exists; the
policy must infer from sparse food rewards propagated backward across ~10-tick
sensory latency. The steering probe `learning_probe_mirrored_steering_is_chance()`
(`integration.rs:2848-2929`) trains on alternating left/right food, then pins
movement and scores turn-alignment; alignment stays at chance (0.489 baseline,
0.38–0.62 band) despite encoder separability 18–24×.

Edits (one of the following, determined by prototype; the landed recipe is recorded
in `0001-AUXILIARY-LOSS-DECISION.md`, decision rule in SCOPE):

- **Direct steering supervision** (`brain_passes.wgsl`, new function
  `coop_steer_auxiliary_loss()` or integrated into `coop_predict_and_act()`): after
  encoding the retina, compute the policy's forward/turn action logits (or soft
  action probabilities), and compare them to a direct target inferred from the
  encoded food bearing. The target is the food bearing from the retina (e.g.,
  cosine/sine of the bearing angle). Apply a small auxiliary loss (e.g., L2 distance
  between action logits and target, or cross-entropy on turn direction) to update
  the encoder and action weights toward vision-conditioned steering. The learning
  rate is small (e.g., 1/10th of the main TD rate) so the homeostatic signal
  dominates but steering supervision shapes the policy.

```wgsl
/// Auxiliary steering-supervision loss: compares the policy's turn direction
/// (from turn_logit) to the true food bearing (from encoded features) with a
/// small learning rate, directly bypassing the slow TD bootstrap across sensory
/// latency. Applied only when vision is fresh (brain_tick_stride boundary).
const AUXILIARY_STEER_LEARNING_RATE: f32 = 0.01;
const AUXILIARY_STEER_WEIGHT: f32 = 0.1;  // Scales the auxiliary loss relative to TD
```

- **Contrastive encoding** (alternative): add a secondary loss to the encoder that
  increases the cosine distance between food-left and food-right encodings. For
  example, during training on alternating food, compute `loss = max(0.0, margin −
  cosine_dist(enc_left, enc_right))` where margin=0.5. This pushes the encoder to
  amplify food-side separation without shaping rewards. Applied during the encoding
  stage only, preserving homeostasis-only credit.

```wgsl
/// Contrastive encoding loss: widens the cosine gap between food-left and
/// food-right encodings up to a margin, sharpening side-separability in the
/// encoding stage only — never in the TD credit path.
const CONTRASTIVE_ENCODING_LOSS_WEIGHT: f32 = 0.01;
const CONTRASTIVE_MARGIN: f32 = 0.5;
```

- **Predictive steering target** (alternative): after encoding, predict the next
  food bearing (1-step lookahead) from the current encoding and the action taken.
  Train a small 128→2 linear head to predict (cos_bearing, sin_bearing) at the next
  step, and use L2 loss. This teaches the encoder to carry bearing information
  forward, indirectly improving steering.

Properties that make this safe:
- The auxiliary loss is applied to the encoder and action weights *only* during the
  encoding/action stage, never to the TD critic or homeostatic gradient, so the
  homeostasis-only contract holds.
- The learning rate is small (e.g., 1/10th of TD), so the homeostatic signal is
  dominant and the policy still prioritizes energy/integrity over steering.
- The auxiliary target (bearing from encoded vision, or contrastive margin) is
  derived purely from sensory input, not from external goals or shaped rewards,
  preserving constraint integrity.
- Reversibility: if the auxiliary loss regresses food-consumption or other probes,
  it can be disabled via a flag and reverted without affecting the core TD path.

## 0002 — Credit Horizon and Trace Restructuring

Today eligibility traces decay via `trace_decay = TD_DISCOUNT * TD_LAMBDA = 0.873`
per brain tick (`common.wgsl:560`, `564`), giving (0.873)^10 ≈ 6.7% retention per
vision-frame cycle (10 brain ticks at default strides). The raw gradient is sampled
once per `vision_stride` ticks (default 10), so fresh credit from a new vision frame
arrives ~10 ticks after the previous frame's credit has largely decayed. This
creates a timing mismatch: the trace has forgotten by the time fresh signal arrives.

Edits (determined by prototype; recipe in `0002-TRACE-HORIZON-DECISION.md`, decision
rule in SCOPE):

- **N-step TD returns** (`brain_passes.wgsl`, TD error computation at
  `brain_passes.wgsl:1220-1227`): replace single-step `δ = reward + γ·V(s′) − V(s)`
  with n-step lookahead: `δ_n = Σ(γ^i · reward_i for i=0..n−1) + γ^n·V(s_n) − V(s)`.
  Store a ring buffer of past rewards and values (n=5–20 steps), and accumulate them
  on every TD update. This gives the critic a longer temporal window to propagate
  credit, allowing traces to accumulate across multiple vision frames before being
  applied.

```wgsl
/// N-step TD return depth: accumulate rewards over the next n steps before
/// bootstrapping to V(s_n). n=10 spans roughly one vision cycle (vision_stride=10).
const TD_NSTEP: u32 = 10u;
```

- **Frame-synchronized trace decay** (`brain_passes.wgsl`, trace update block at
  ~`1588-1603`): apply aggressive decay *only* at vision-frame boundaries (when
  brain_tick matches a vision-frame boundary), allowing traces to persist unchanged
  within a frame cycle. This decouples the trace timescale from the brain-tick
  timescale and explicitly aligns trace longevity with sensory cadence.

```wgsl
/// Trace decay applied only at vision-frame boundaries (brain_tick % vision_stride == 0),
/// preserving traces within a frame. Decay constant targets ~10% retention per frame,
/// allowing credit to accumulate across frames without decay attenuation.
const TRACE_DECAY_AT_FRAME_BOUNDARY: f32 = 0.9;  // 10% decay per vision frame
```

- **Decoupled trace-decay constant** (alternative, `common.wgsl`): separate the
  trace decay from `TD_DISCOUNT × TD_LAMBDA`, introducing an independent
  `TRACE_DECAY_SCALE` that is tuned to the vision-stride cadence. For dense strides
  (vision_stride=1), use decay=0.99; for sparse strides (vision_stride=10), use
  decay=0.95. This allows a single tuned decay constant to work across stride
  configurations.

```wgsl
/// Per-brain-tick trace decay, independent of TD_DISCOUNT. Tuned to match the
/// vision-stride cadence: (TRACE_DECAY_SCALE)^vision_stride ≈ 50% retention per
/// vision cycle, allowing credit to bridge sensory latency.
const TRACE_DECAY_SCALE: f32 = 0.95;  // Empirically tuned; replaces TD_DISCOUNT*TD_LAMBDA
```

Properties that make this safe:
- N-step TD is a standard RL technique; the edit only extends the temporal window,
  not the learning rule itself.
- Frame-synchronized decay explicitly aligns the trace timescale with the sensory
  cadence, making the coupling transparent and tunable.
- Decoupled decay preserves the existing TD critic stability (traces are bounded,
  updates are scaled by ENCODED_DIMENSION), only changing the decay schedule.
- All three variants can be toggled via a gate or flag so a regression triggers
  reversion.

## 0003 — Gradient Variance and Signal Shaping

Today the learning signal during foraging is dominated by tiny homeostatic deltas:
mean|δ| = 8.7×10⁻⁵, std = 6.5×10⁻⁵ (from 0017's variance test at
`0001-CREDIT-PATH-DECISION.md`). The encoder-to-action weight updates
(`brain_passes.wgsl:1250-1252`) scale by `ACTION_WEIGHT_LEARNING_RATE ·
ACTOR_VECTOR_SCALE · δ = 0.10 · (1/16) · (8.7e-5) ≈ 5.4e-7` per dimension per tick —
effectively zero compared to the weight initialization scale (0.1). Over a 100-tick
episode the cumulative update is still ~5e-5, too small to move the policy
perceptibly.

Edits (determined by prototype; recipe in `0003-GRADIENT-SHAPING-DECISION.md`,
decision rule in SCOPE):

- **TD-error normalization** (`brain_passes.wgsl`, after TD error computation at
  `1227`): normalize the TD error by the running standard deviation of recent
  errors. Maintain a moving average of |δ| and divide each δ by max(std(δ), ε) to
  restore a signal magnitude of order ~0.1–1.0. This makes the learning rate
  effective regardless of the raw-gradient variance.

```wgsl
/// Running estimate of TD-error magnitude (EMA of |δ|). Used to normalize δ
/// so learning rates stay effective during low-signal episodes.
const TD_ERROR_EMA_ALPHA: f32 = 0.01;  // EMA alpha for |δ| tracking
```

- **Gradient clipping and shaping** (`brain_passes.wgsl`, weight updates at
  `1247-1252`): apply a soft normalizing function to the TD error before scaling by
  the weight updates. For example, replace `δ` with `tanh(δ·k)` where k is a scaling
  factor (e.g., k=10) that maps the tiny δ ≈ 1e-4 to a usable range like 0.1. This
  preserves sign (direction of credit) but amplifies magnitude.

```wgsl
/// Scaling factor applied to TD error before weight updates, mapping tiny
/// homeostatic deltas (~1e-4) to usable learning-rate ranges (~0.1).
const TD_ERROR_SCALE_FACTOR: f32 = 100.0;
```

- **Auxiliary gradient from vision-action alignment** (alternative, integrated with
  0001): add a small direct-supervision loss from the auxiliary steering objective
  (0001) that targets turn-direction alignment with high gradients (~0.1 per tick).
  This auxiliary signal amplifies the weak homeostatic signal, allowing the policy
  to learn steering while remaining grounded in homeostasis.

Properties that make this safe:
- TD-error normalization preserves sign and relative ordering (larger errors still
  propagate more credit); it only amplifies magnitude to match the learning-rate
  regime.
- Gradient shaping via tanh or scaling factors keeps gradients bounded and
  reversible: if the scaling factor is too large, the policy oscillates and fitness
  drops, triggering reversion.
- Auxiliary supervision (0001) adds a second signal source but only in the
  encoder/action stage, preserving the homeostasis-only credit path.
- The variance probe from 0017 (mean|δ|, std|δ|) can be re-run to confirm the
  shaping actually increases signal magnitude.
