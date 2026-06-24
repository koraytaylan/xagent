# Architecture — Plan 0017 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-sandbox/src/evolution.rs`,
> `crates/xagent-sandbox/tests/integration.rs`, and
> `crates/xagent-shared/src/config.rs`.
> Line numbers are hints; locate by symbol (grep for `coop_predict_and_act`,
> `raw_gradient`, `TD_DISCOUNT`, `coop_visual_cortex`, `coop_encode`,
> `emergent_encoder_enabled`).

## 0001 — Credit-Path Diagnosis and Learning Unlock

Today `coop_predict_and_act()` (`brain_passes.wgsl:947-1250`) applies TD(λ) credit
assignment each frame. The reward fed to TD is the urgency-amplified homeostatic
gradient: `raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta *
INTEGRITY_WEIGHT + shaping + danger_shaping` (`brain_passes.wgsl:811-814`, where
`shaping` and `danger_shaping` are zero post-0012), then `raw_gradient_amplified =
raw_gradient * (1.0 + urgency)` (`brain_passes.wgsl:831-835`) is written to
`s_homeo[1u]`. The kernel reads `reward = s_homeo[1u]` (`kernel_tick.wgsl:1118`)
and computes `δ = clamp(reward + TD_DISCOUNT·V(s′) − V(s), [-MAX_TD_ERROR,
MAX_TD_ERROR])` = `[-1, 1]` (`kernel_tick.wgsl:1120-1123`). The critic updates via
`V_weights += CRITIC_LEARNING_RATE·TD_VECTOR_SCALE·δ·trace_critic`
(`brain_passes.wgsl:1146`) and both policy channels via `action_weights +=
ACTION_WEIGHT_LEARNING_RATE·ACTOR_VECTOR_SCALE·δ·trace_*`
(`brain_passes.wgsl:1148-1150`), with `TD_DISCOUNT=0.97` (`common.wgsl:517-564`)
and traces that are reset on respawn but never explicitly per-step decayed. The
mirrored-steering probe `learning_probe_mirrored_steering_is_chance()`
(`integration.rs:2716-2792`) trains 120 episodes on dense TD (`brain_tick_stride=1,
vision_stride=1`) with alternating left/right food, pins movement
(`movement_speed=0`), and scores turn-alignment at 0.38–0.62 (chance) — while
`encoder_food_side_separability_diagnostic()` (`integration.rs:2810-2887`) shows
within-class cosine ~0.998 and between-class cosine ~0.036 (a 55× separability
margin). The encoder separates food-left from food-right; the policy cannot learn
to turn toward it.

Edits (one of the following, or a combination determined by the spike tasks; the
landed recipe is recorded in `0001-CREDIT-PATH-DECISION.md`, decision rule in
SCOPE):

- **TD decay-schedule hardening** (`common.wgsl`, after `TD_DISCOUNT`): introduce
  an explicit per-step eligibility-trace decay so recency-weighted credit
  accumulates across vision-frame windows rather than single-tick transitions.
  Vision is sampled once per `vision_stride` ticks, so credit must bridge that
  latency.

```wgsl
/// Per-step eligibility-trace decay applied before each TD update. 0.99 retains
/// ~86% over 10 steps (one default vision cycle), bridging the vision-frame
/// latency between raw_gradient samples. Raise (e.g. 0.995) for denser strides.
const TRACE_DECAY_PER_STEP: f32 = 0.99;
```

- **Decay-first trace update** (`brain_passes.wgsl:1140-1155`, inside the
  per-encoded-dimension block, before the TD update): attenuate the loaded trace
  by `TRACE_DECAY_PER_STEP`, then apply this tick's gradient, so old credit decays
  toward zero over multiple vision cycles and new credit can dominate.

```wgsl
// Decay first, then apply this tick's TD gradient, so traces attenuate over
// multiple vision cycles and credit from the current frame can dominate.
let decayed_critic_trace = brain_state[brain_base + O_TRACE_CRITIC + tid] * TRACE_DECAY_PER_STEP;
```

- **Urgency-scaling isolation** (`brain_passes.wgsl:832`): separate the urgency
  flag (used for the homeostatic monitoring loop and exploration control) from the
  TD learning signal, since `(1.0 + urgency)` can spike δ to the clamp boundary
  near death and collapse steady-state credit variance — letting the policy learn
  only at death/respawn, not during foraging.

```wgsl
// Learning signal is the bare homeostatic gradient; urgency still scales the
// monitoring loop below, but no longer saturates δ at the [-1, 1] clamp.
let raw_gradient_amplified = raw_gradient;
```

- **Trace clipping** (`brain_passes.wgsl:1150-1154`): add a per-trace magnitude
  clamp to prevent spike-and-flatline pathologies when a single high-variance δ
  event locks the traces.

```wgsl
// Bound each eligibility trace so one high-variance δ cannot lock it; the clamp
// can only tighten the credit signal, never create a gradient.
let clamped_trace = clamp(trace, -0.1, 0.1);
```

- **Gradient structuring** (`brain_passes.wgsl:1119`, test-harness first):
  investigate whether the reward should carry a small approach-gradient leakage,
  or whether an auxiliary homeo-only loss could supervise the encoder's food
  representation. Prototyped in isolation; only promoted if it holds the
  homeostasis-only gate (SCOPE).

Properties that make this safe:
- The homeostasis-only contract holds: every edit operates on the magnitude or
  timing of `raw_gradient` (energy/integrity deltas only) and its scaling — no
  shaped rewards, goal signals, or new signal sources enter the TD path.
- Eligibility traces are a standard, already-present RL mechanism, not a new
  construct; this plan tunes their decay and bounds.
- Per-step decay and the trace clamp are conservative: they can only tighten the
  credit signal, never synthesize an unwanted gradient.
- The mirrored-steering probe's assertion band is re-pinned only when the fix
  lands (e.g. `(0.38..=0.62)` → `(0.70..=0.85)`), so the baseline stays falsifiable
  and the encoder-separability diagnostic must still show the 55× margin after the
  fix, proving the change moved the policy, not the encoder.

## 0002 — Cortex Throughput Optimization

Today `coop_visual_cortex()` (`brain_passes.wgsl`, from 0008) reads a dense 32×32
luminance retina and applies: (1) a DoG center-surround (σ_surround:σ_center ≈
1.6), (2) a 4×2×2 oriented Gabor bank (8 orientations over 2 scales), (3)
quadrature-energy complex cells plus 4×4 MAX pooling → 128 output features.
Runtime is ~420× the fused baseline (~81 tps vs ~34,000 tps on N=10), consuming
the entire GPU cycle budget at 0.24%. Probes confirm mechanical correctness:
vertical-bar tuning 6.97× orthogonal, phase invariance ~1.6e-7, position
tolerance ~7.8%.

Edits (candidates, ranked by cost-reduction potential; combination and final
recipe set by per-change profile runs):

- **Separable DoG** (`brain_passes.wgsl`, DoG convolution): replace the full 2D
  DoG kernel with the outer product of 1D Gaussian row/column passes, cutting
  ~25 multiplies/pixel to ~10. Verify the 1D outer product reconstructs the 2D
  kernel within 1e-6, then re-check center-surround ratio and phase invariance.

```wgsl
// 1D normalized Gaussian (σ=1); the separable row/column pass reconstructs the
// 2D DoG center kernel via outer product to within 1e-6 of the dense form.
const DOG_CENTER_KERNEL_1D: array<f32, 5u> = array<f32, 5u>(0.05, 0.244, 0.401, 0.244, 0.05);
```

- **Shared-memory precomputation** (`brain_passes.wgsl`, cooperative tiling): load
  a 34×34 retina tile into workgroup shared memory once behind a single barrier,
  then have each thread read its Gabor neighborhood from shared memory instead of
  repeated global loads — a ~2–4× gain if memory bandwidth dominates.

- **Smaller retina** (`common.wgsl`, `VISUAL_RETINA_WIDTH` / `VISUAL_RETINA_HEIGHT`
  defaults 32×32): reduce to 24×24 (input pixels shrink ~1.78×). Re-run the
  orientation and phase probes on the smaller retina; the selectivity floor must
  hold (tuning ≥3×, phase variance <10%, position variance <15%).

```wgsl
// Retina shrunk 32→24 to cut input pixels ~1.78x; orientation/phase/position
// probes must still pass (tuning ≥3×, phase <10%, position <15%) after the change.
const VISUAL_RETINA_WIDTH: u32 = 24u;
const VISUAL_RETINA_HEIGHT: u32 = 24u;
```

- **Reduced feature count** (`common.wgsl`, `VISUAL_FEATURES` or equivalent): halve
  128 → 64 complex cells by dropping to 4 orientations or one scale. The
  food-separability diagnostic must still hold (cosine-diff >0.5); revert if it
  drops below.

- **Pooling-radius tuning** (`brain_passes.wgsl`, complex-cell pool): reduce MAX
  pooling from 4×4 to 3×3 (or 2×2). Position-invariance probe must stay <15%.

Combination strategy: stack (separable DoG + smaller retina + pooling 3×3),
keeping enough feature/scale dimensions to preserve probe margins, until
throughput lands ≥17,000 tps (≥50% of the 34k baseline) at N=10, verified via
`--bench-agent-sweep`.

Properties that make this safe:
- All edits stay within `coop_visual_cortex()` and the Gabor-bank constants; no
  cross-stage changes to encoding, credit, or physics.
- Every edit re-runs the 0008 probes — orientation tuning, phase invariance,
  position tolerance, food separability — and reverts on regression.
- The cortex is flag-gated and default-off, so the baseline stays byte-identical
  for workstream 0001's steering validation and only flips after the workstream 0004
  A/B verdict.
- Throughput is profiled per change and must reach ≥50% of the fused baseline
  before the suite is considered complete.

## 0003 — Emergent Self-Organizing Encoder

Today the Gabor bank (0008) is a fixed biological scaffold: wavelength, aspect
ratio, `dog_surround_ratio`, and orientation offset are heritable global-bank
genes, seeded at birth and never learned — they evolve only by mutating those four
genes. The founding principle demands structure that arises from learning and
input statistics, not imported design.

Edits:

- **Sparse / predictive coding objective** (`brain_passes.wgsl`, in `coop_encode()`
  or a `coop_visual_encode_learned()` variant): replace the fixed Gabor filters
  with a learned dictionary trained online. Each brain tick, after reading the raw
  retina, encode to 128 dims via learned `encode_weights`, decode via
  `decode_weights`, and apply a small gradient step on `loss = ||x − x̂||² +
  λ·||code||₁` (or a one-frame-lag predictive variant). The learning rate is small
  so the weights drift slowly and evolution acts on top.

```wgsl
// Online sparse-coding loss on the learned encoder: reconstruction MSE plus an
// L1 penalty on the code. λ keeps codes sparse; gradient updates encode/decode
// weights each tick. Auxiliary objective in the encoding stage only — never the
// credit path — so the homeostasis-only contract is preserved.
const SPARSE_CODE_L1_LAMBDA: f32 = 0.05;
const ENCODER_LEARNING_RATE: f32 = 1e-4;
```

- **Heritable weight matrices** (`buffers.rs`, `config.rs`): seed `encode_weights`
  and `decode_weights` from a Gaussian (mean 0, std 0.1) at birth, pass them
  through the inheritance/mutation path like other heritable brain weights, and
  update them during the agent's lifetime via the learning objective. Keep the
  per-agent footprint within budget (reduce dimensionality if the two matrices
  exceed the resident state limit).

```rust
/// Learned visual front-end: two heritable matrices seeded N(0, 0.1) at birth,
/// inherited with Gaussian mutation, and refined each tick by the sparse/
/// predictive objective. Mirrors the existing heritable predictor/recall weights.
const ENCODER_WEIGHT_INIT_STD: f32 = 0.1;
const ENCODER_WEIGHT_MUTATION_STD: f32 = 0.01;
```

- **Emergent-encoder flag** (`config.rs`): add `emergent_encoder_enabled`
  (default off), independent of `visual_cortex_enabled`, so the learned encoder
  toggles separately and both stay off by default until the workstream 0004 A/B flips one.

- **Orientation-selectivity probe** (new test in `integration.rs`): after training
  on a naturalistic world, extract the learned `encode_weights`, present isolated
  vertical/horizontal bars (the Gabor-baseline stimulus), and compute per-code
  tuning ratios. Document the distribution — most codes isotropic, some showing
  >3× tuning *emergent* from statistics, not hardcoded.

- **Generation-heritability probe** (new evolution test in `integration.rs`): run
  a 5-generation seeded evolution with the learned encoder, recording
  `encode_weights` per generation, and assert orientation-selectivity margins
  persist across generations (inherited structure is retained, mutants re-learn or
  diverge measurably).

Properties that make this safe:
- The sparse/predictive loss is an auxiliary encoding-stage objective, not a shaped
  reward; it never touches the TD credit path, so the homeostasis-only contract
  holds.
- Heritable weight matrices already exist for the predictor and recall patterns;
  this extends the same mechanism to a learned visual front-end.
- The orientation-selectivity probe measures emergence directly: learned tuning
  must clear the 3× threshold *without* hardcoded filters, proving
  self-organization rather than asserting it.
- Reversibility: if learned codes diverge or selectivity drops,
  `emergent_encoder_enabled` (default off) keeps the Gabor bank available.

## 0004 — A/B Comparison and Winner Promotion

Today the brain carries two independently-evolving visual paths: legacy direct
raycasting (default) and the flag-gated Gabor cortex (0008). With workstream 0003's
learned encoder there is a third (flag-gated). This workstream compares the visual
paths on equal footing once workstreams 0001 (credit), 0002 (cortex), and 0003
(emergent encoder) are green.

Edits:

- **Seeded-paired A/B harness** (`evolution.rs` or a new
  `integration_ab_encoder_comparison.rs`): add a function that runs a fixed-seed
  evolution and returns per-generation population means, so both arms see identical
  food placement, physics, and RNG and differ only in the encoder.

```rust
/// Runs one fixed-seed evolution arm and returns per-generation population means
/// (mean fitness, lifespan, approach/avoidance intent fraction, food consumed,
/// death count). Seeding the world and population RNG from `seed` makes the two
/// arms differ only in the encoder, so the comparison is fair.
fn run_paired_evolution_ab(arm: EncoderChoice, seed: u64, generations: u16, population: usize) -> EvolutionResult;
```

- **Paired A/B test** (`integration.rs`): run Arm A
  (`emergent_encoder_enabled=false, visual_cortex_enabled=false`, legacy) against
  Arm B (`emergent_encoder_enabled=true, visual_cortex_enabled=false`, learned) on
  identical seeds for 16 generations, population 10; record per-generation fitness,
  lifespan, intent fractions, food, and deaths to
  `ab_paired_results.csv`; compute the B−A delta and a 95% CI (bootstrap or paired
  t-test). Separately measure the A/A noise floor (Arm A twice on distinct seeds);
  if inter-run variance >2%, re-run with larger population or longer generations.

- **Statistical verdict** (`0004-ENCODER-CHOICE-DECISION.md`): apply the verdict
  rule — `EMERGENT_WINS` if the fitness CI excludes zero and the point estimate is
  ≥+5%; `INCONCLUSIVE_KEEP_BOTH` if the CI crosses zero or the estimate is <+5%;
  `GABOR_WINS` if the CI is negative and ≤−5%. Record intent trajectories over the
  16 generations and per-seed generalization (winning arm on 3 distinct seeds).

- **Promotion logic** (`config.rs`): if the emergent encoder wins, flip
  `emergent_encoder_enabled=true` as the new default and archive
  `visual_cortex_enabled=false`; if Gabor wins or the result is inconclusive, keep
  both flags off-by-default with a design note recording the decision. Both code
  paths ship complete so a later finding can revisit the default.

Properties that make this safe:
- Seeded-paired determinism guarantees a fair comparison: both arms see identical
  food placement, physics, and random-number sequences; only the encoder differs.
- The 95% CI plus A/A noise floor guards against overstating a marginal
  difference; within-noise results resolve to "inconclusive, defer" rather than
  promoting a false winner.
- The comparison runs only after workstream 0001 unlocks credit and workstreams
  0002/0003 fully develop both encoders, so it measures learning capability, not
  throughput artifacts or incomplete features.
- Reversibility: both flags remain in the code, so a later measurement can flip the
  default again on better evidence.

## Test strategy

- **0001:** `baseline-mirrored-steering-probe` pins the chance band
  (`integration.rs:2716-2792`) and adds
  `baseline_encoder_separability_vs_steering_gap()` reporting the ~24× margin
  between separability (cosine-diff 0.964) and steering (~0.40);
  `credit-path-fix-lands` re-pins the assertion to the new band (e.g.
  `(0.70..=0.85)`) and proves the encoder-separability and food-visibility probes
  still pass. The spikes (`td-decay-schedule-audit`, `urgency-scaling-isolation`)
  prototype in worktrees and record pass/fail in `0001-CREDIT-PATH-DECISION.md`.
- **0002:** `cortex-throughput-profile-baseline` records the 0.24% per-component
  profile; `separable-dog-optimization` and `cortex-throughput-optimization-suite`
  re-run the 0008 orientation/phase/position probes on every edit and must land
  ≥17,000 tps (≥50% of baseline) with all probes green.
- **0003:** `sparse_encoder_converges_on_fixed_input()` asserts reconstruction MSE
  decays and code magnitude stays bounded; the orientation-selectivity probe
  asserts emergent tuning >3× without hardcoded filters; the food-separability
  diagnostic asserts cosine-diff >0.5 on the learned encoder.
- **0004:** the paired A/B test produces `ab_paired_results.csv`, the A/A noise
  floor confirms reproducibility, and `0004-ENCODER-CHOICE-DECISION.md` records the
  95% CI verdict.
- CI gate (every task): `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p xagent-sandbox` (and `-p xagent-brain` for shader-touching tasks).
  GPU tests self-skip without an adapter locally; CI runs Mesa lavapipe.

## Interaction with prior work

- **Resolves the 0013 diagnosis.** 0013 (seeded instincts) was rejected with the
  finding that the credit path — not the encoder or prior seeding — is the limiter;
  the `2026-06-19` due-diligence reviews independently ranked this M1. Workstream
  0001 acts on that verdict by hardening the TD timescale instead of re-seeding.
- **Honors the 0012 homeostasis-only contract.** 0012 removed both approach and
  danger shaping; this plan keeps `raw_gradient` derived only from `energy_delta`
  and `integrity_delta` and tunes only magnitude/timing, never introducing new
  signal sources.
- **Builds on 0008's stranded cortex.** 0008 proved mechanical correctness (6.97×
  tuning, phase invariance ~1.6e-7) but at 0.24% of budget; workstream 0002 optimizes
  it as a fair control arm, and workstream 0003 offers the emergent alternative the
  founding principle demands. The workstream 0004 A/B adjudicates between them.
- **Defers the fitness levers.** The effort-rebased fitness, danger percept, and
  speed-cost levers (0009/0010/0012/0013) stay off; graduating them waits for a
  later plan once the workstream 0004 A/B anchors the learning-signal quality.
- **Consumes, does not change, the 0016 intent framework.** The A/B reports
  approach/avoidance intent fractions as measurement targets; it does not alter the
  evolution algorithm, mutation rates, or selection pressure.
