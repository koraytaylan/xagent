# XAgent Plan 0017 — Credit Path Fix and Emergent Visual Learning A/B — Four-Phase Measurement-Driven Roadmap

This plan's four workstreams execute the project's founding principle: emergence from constraints, not imported design. Workstream 0001 diagnoses and fixes the credit-path bottleneck (TD decay, decay schedule, trace clipping, or gradient structuring) that leaves vision→action steering at chance despite encoder separability (55×) — unlocking steering learnability. Workstream 0002 independently optimizes cortex throughput as a fair control arm. Workstream 0003 replaces the hand-coded Gabor bank with self-organizing sparse/predictive coding, tested against orientation selectivity. Workstream 0004 runs a seeded-paired A/B (16 gens, pop 10) comparing emergent vs imported on fitness/intent metrics with 95% CI verdict and measured winner promoted to default.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Credit-Path Diagnosis and Learning Unlock

### baseline-mirrored-steering-probe — Verify and Document Mirrored-Steering Baseline (Chance Band)

The steering probe `learning_probe_mirrored_steering_is_chance()` (integration.rs:2716–2792) trains agents for 120 episodes on alternating left/right food with dense TD strides (`brain_tick_stride=1, vision_stride=1`), then pins movement speed (`movement_speed=0`) and evaluates turn-alignment (correct/scored ratio). Current result: 0.38–0.62 (chance band). This baseline will be re-measured after each credit-path edit to confirm the fix moves alignment above 0.62.

**Steps:**
1. Run the existing `learning_probe_mirrored_steering_is_chance()` test locally on your machine (GPU/Metal adapter) and record the measured alignment rate (correct/scored ratio).
2. Document the current baseline in a comment above the assertion at integration.rs:2787–2791, including date, machine/adapter, and the measured rate (e.g. '2026-06-23 local Metal: aligned=463/1204=0.384').
3. Add a companion test `baseline_encoder_separability_vs_steering_gap()` that runs the encoder diagnostic and steering probe back-to-back and reports the margin (separability cosine-diff 0.964 vs steering rate 0.40, ratio ~24:1), documenting the diagnosis: encoder is 24× better at separating food-left/right than the policy is at turning toward it.

- **Depends on:** —
- **Done when:** The baseline test runs, passes on current code with alignment in [0.38, 0.62], and the margin diagnostic computes and reports the 24× gap; cargo fmt/clippy/test green. This baseline will be re-pinned by task credit-path-fix-lands (credit-path-hardening) once the fix is merged.

---

### td-decay-schedule-audit — Audit and Prototype TD Decay Schedule for Trace Longevity

The TD(λ) credit path (brain_passes.wgsl:1082–1200) updates traces and accumulates credit each frame. The eligibility traces are never explicitly decayed in the current code audit (trace reset only on respawn); they are implicitly attenuated by the policy's changing outputs. Vision is sampled once per `vision_stride` (default 10) ticks, so a food-item bearing change takes ~10 ticks to propagate through the retina → encoding → credit path. If traces decay too fast or the TD error is clipped too tightly, credit from the `vision_stride` boundary never reaches the encoder. This spike investigates: (a) whether explicit per-step trace decay is already in the shader, (b) if not, what decay constant would extend trace life to match the sensory lag (≥100 ticks for dense settings, ≤10 ticks for sparse), and (c) whether the urgency amplification (1.0 + urgency, up to ~5x near death) saturates δ and kills credit gradient variance.

**Steps:**
1. Grep the WGSL code for trace decay: search `brain_passes.wgsl`, `kernel_tick.wgsl`, and `common.wgsl` for `trace.*=.*trace` patterns and comment blocks mentioning decay, λ (lambda), or eligibility. Document every found reference with file:line.
2. If no explicit per-step decay exists: prototype a new constant `TRACE_DECAY_PER_STEP: f32 = 0.99;` in `common.wgsl` (after `TD_DISCOUNT` and before the MAX_TD constants) with a comment: '// Per-step eligibility trace decay; default 0.99 = 86% retention over 10 steps (one vision cycle). Empirically tuned to bridge vision-frame latency (raw_gradient sampled ~10 ticks apart on default strides). Increase (e.g. 0.995) for denser strides (faster credit propagation needs longer traces).'
3. In `brain_passes.wgsl` at the trace-update sites (≈1140–1155, inside the `if tid < ENCODED_DIMENSION` block after TD error is computed), apply decay **before** the TD update: `let decayed_critic_trace = brain_state[brain_base + O_TRACE_CRITIC + tid] * TRACE_DECAY_PER_STEP;` then use `decayed_critic_trace` in place of the bare `brain_state[...]` load in the update equation. Repeat for forward and turn traces. Add a comment: '// Decay first, then apply this tick's TD gradient, so traces naturally attenuate toward zero over multiple vision cycles, allowing new credit to dominate.'
4. Prototype a test `trace_decay_enables_cross_vision_cycle_credit()` in gpu_kernel.rs or integration.rs: (a) create a test agent with a seeded brain, (b) manually set all traces to 1.0 at `t=0`, (c) compute one frame of no-movement, no-food (δ ≈ 0 or small negative), and record the traces at `t=1` (should be ~0.99 if decay is applied, 1.0 if not), (d) assert `trace[t=1] < 0.999` to confirm decay is active.
5. Run the baseline steering probe again with the prototype `TRACE_DECAY_PER_STEP=0.99` enabled; if alignment moves above 0.62, the decay is a candidate fix. If it stays in chance band, revert the prototype and move to the next hypothesis (urgency scaling or gradient structuring).

- **Depends on:** baseline-mirrored-steering-probe
- **Done when:** The audit reports all trace references (file:line) with decay status (found or not found); a prototype decay constant and trace-update edits are drafted (in a worktree, not yet merged); a test confirms the decay is active (trace[t=1] < 0.999 if decay=0.99); and the steering probe is re-run with the prototype. If alignment moves above 0.62, the decision document (0001-CREDIT-PATH-DECISION.md) records this as the suspected fix; if not, note the failure and the spike continues to urgency scaling (next task). Cargo fmt/clippy/test green on any committed edits (the prototype may be reverted if inconclusive).

---

### urgency-scaling-isolation — Prototype Urgency-Scaling Isolation for Credit Variance

The TD reward is computed as `raw_gradient_amplified = raw_gradient * (1.0 + urgency)` (brain_passes.wgsl:832–835), where urgency ranges 0 to ~5 (quadratic distress curve near death). This amplification can spike the TD error to `±MAX_TD_ERROR * (1 + urgency)`, which is then clamped to `[-1, 1]` (kernel_tick.wgsl:1120–1123). If the agent is in low-urgency steady state (foraging at mid-energy), `urgency ≈ 0.1` and the reward signal is tiny (~0.01 food-energy-delta). If the agent is near death, urgency ≈ 5 and the signal spikes, then gets clamped, then the agent respawns and urgency resets. This could cause the policy to learn only during death/respawn events, not during exploration. This spike isolates urgency from the learning signal to test whether a gentler or separate urgency flag improves credit variance.

**Steps:**
1. In `brain_passes.wgsl:832`, comment out or remove the urgency amplification on the learning signal: change `let raw_gradient_amplified = raw_gradient * (1.0 + urgency);` to `let raw_gradient_amplified = raw_gradient;` (remove the urgency factor, keeping raw_gradient as-is).
2. Keep `let gradient = blended_gradient * (1.0 + urgency);` (brain_passes.wgsl:831) unchanged so urgency still affects the homeostatic monitoring loop and exploration control, but not the TD learning signal.
3. Prototype a test `urgency_isolation_preserves_learning_signal()`: measure the variance of δ across a 1000-tick episode with and without urgency amplification, sampling δ values at steady-state (mid-energy, no death). With amplification, variance should be much lower (most values clamped); without, variance should reflect the true energy-delta variability.
4. Run the baseline steering probe again with urgency isolation enabled; record alignment and compare to the decay-prototype result (if decay alone did not fix it). If alignment moves above 0.62, urgency isolation is a candidate; if both decay and urgency isolation are needed, document the combination.

- **Depends on:** td-decay-schedule-audit
- **Done when:** The urgency amplification is isolated in a worktree (raw_gradient_amplified = raw_gradient, no (1.0 + urgency) factor); a variance test is drafted; the steering probe is re-run. If alignment moves above 0.62, note it in the decision doc. The decision doc (0001-CREDIT-PATH-DECISION.md) records which hypotheses passed/failed and which is the best candidate fix. Cargo fmt/clippy/test green on any committed edits (prototypes may be reverted pending the decision).

---

### credit-path-fix-lands — Land the Credit-Path Fix (Decay, Urgency, Traces, or Combination) (GATED)

**Gate:** Both `td-decay-schedule-audit` and `urgency-scaling-isolation` have completed and the decision document (0001-CREDIT-PATH-DECISION.md) identifies the fix (or combination of fixes). The fix must move steering alignment from chance (0.38–0.62) to above 0.62 (ideally ≥0.70) on the mirrored-steering probe. This task lands the fix into the main codebase with updated baseline assertions.

**Steps:**
1. Based on the decision document's recommendation, apply the fix (or combination) to the source: (a) if trace decay, add TRACE_DECAY_PER_STEP to common.wgsl and update brain_passes.wgsl; (b) if urgency isolation, remove the (1.0 + urgency) factor from raw_gradient_amplified; (c) if gradient structuring, add the auxiliary loss (as a separate test harness first, then integrated if promising).
2. Update the mirrored-steering probe's baseline assertion (integration.rs:2787–2791) with the new expected range. If the fix moves alignment to 0.75, update the assertion from `(0.38..=0.62)` to `(0.70..=0.85)` (new pass band reflecting the fix).
3. Run `cargo test -p xagent-sandbox -- --test-threads=1` locally and confirm: (a) the steering probe now passes with alignment ≥0.70, (b) the baseline-gap diagnostic still computes, (c) the encoder separability diagnostic (encoder_food_side_separability_diagnostic) still runs and shows the 55× margin, (d) no other probes regress (food visibility, free-run foraging, baseline turn alignment should all remain green).
4. Document the fix in a code comment at brain_passes.wgsl:810–835 (raw_gradient computation) and common.wgsl (any new constants): explain why the change was needed (credit variance under vision-frame latency, chance-level steering), what it does, and the gate (steering alignment above 0.62).
5. Ensure cargo fmt, clippy, and test green; the fix is now committed and ready for downstream workstreams 0002–0004 to build on top.

- **Depends on:** urgency-scaling-isolation
- **Done when:** Binary outcome — *either* the decision-recommended fix is merged into develop with steering alignment holding ≥0.70 on the mirrored-steering probe, the updated steering-alignment assertion passing, and the encoder-separability and food-visibility probes all still green (`0001-CREDIT-PATH-DECISION.md` in the plan folder records the landed recipe); *or* no hypothesis cleared the 0.62 gate and the change is reverted wholesale, with the negative result recorded in `0001-CREDIT-PATH-DECISION.md` and the chance-band assertion left intact. Cargo fmt/clippy/test green on whichever branch lands. The next workstream (0002, cortex optimization) proceeds either way.

---

## 0002 — Cortex Throughput Optimization

### cortex-throughput-profile-baseline — Profile Cortex Throughput (Baseline 0.24%, All Sub-components)

The visual-cortex throughput is currently 0.24% of the fused baseline (~81 tps vs ~34,000 tps on N=10). The cortex is 420× slower due to the dense Gabor convolutions (8 orientations × 2 scales × 32×32 input = ~200k operations per-frame). This task profiles each sub-component (DoG, Gabor bank, quadrature-energy, pooling) to identify the highest-cost targets for optimization in the next tasks.

**Steps:**
1. In `gpu_kernel.rs` or a new profiling test, add per-pass timing: after `coop_visual_cortex()` in the kernel, record a GPU timestamp and compute the cycle count. Run a fixed-duration benchmark (100 frames, N=10 agents) with cortex enabled, then disabled, and compute the per-frame cortex cost in microseconds and as a % of the total kernel cycle.
2. Add intermediate markers inside `coop_visual_cortex()` (if the WGSL compiler preserves comments/markers; alternatively, split the pass into sub-functions) to measure: (a) DoG retina convolution cost, (b) Gabor filter bank cost, (c) quadrature-energy and MAX-pooling cost. Document which sub-component dominates.
3. Run `cargo build -p xagent-brain --release` then execute the benchmark via `cargo run --release -- --bench-agent-sweep` (or a local variant) and record the throughput with cortex enabled and disabled, at N=10 and N=100.
4. Document the baseline profile in a comment in `gpu_kernel.rs` or a new file `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/CORTEX-PROFILE-BASELINE.txt` with columns: component, cost (µs), % of cortex total, % of kernel total.

- **Depends on:** td-decay-schedule-audit, urgency-scaling-isolation
- **Done when:** The profiling test runs and reports per-frame and per-component cortex cost; baseline is documented; cargo fmt/clippy/test green. This profile will guide the optimization priorities in the next task (separable-dog-optimization).

---

### separable-dog-optimization — Optimize DoG Convolution via Separable Kernels (1D Gaussians)

The DoG (Difference-of-Gaussians) center-surround is currently implemented as a full 2D kernel applied to each pixel in the 32×32 retina. A 2D Gaussian requires ~25 multiplies per-pixel (5×5 kernel). Separating the kernel into two 1D Gaussians (one row, one column) reduces this to ~10 multiplies, a ~2.5× speedup. This task implements the separable DoG in the shader and confirms it does not regress the orientation/invariance probes.

**Steps:**
1. Locate the DoG implementation in `brain_passes.wgsl` (search for `dog` or `center_surround` or the actual Gaussian kernel values). Extract the current 2D kernel matrix (likely a constant array of floats).
2. Derive the 1D row and column kernels: if the 2D kernel is σ_center ≈ 1, σ_surround ≈ 1.6, factor each into 1D Gaussians. For example, the row kernel is `[0.05, 0.244, 0.401, 0.244, 0.05]` (normalized 1D Gaussian σ=1), and surround is a wider 1D kernel. Compute the outer product to verify it matches the 2D kernel within 1e-6.
3. Rewrite the DoG pass to apply 1D kernels sequentially: (a) load a 32×height row slice into shared memory, apply the 1D row filter to produce a 32-element intermediate, (b) apply the 1D column filter to produce the final DoG output. Use cooperat threading to tile the work (all 256 workgroup threads contribute to the shared-memory row/column reductions).
4. Run the orientation-selectivity and phase-invariance probes (`visual_encoder_orientation_selectivity_probe`, `visual_encoder_phase_invariance_probe` from 0008) and confirm: vertical-bar tuning ratio ≥3× (baseline 6.97×), phase invariance <10% (baseline ~1.6e-7), position invariance <15% (baseline ~7.8%). If any probe regresses, revert and debug.
5. Profile the throughput again (cortex-throughput-profile-baseline) and record the new per-frame cost. Target: ~2–4× DoG speedup (e.g. from 50µs to 20µs), contributing to the overall cortex speedup.

- **Depends on:** cortex-throughput-profile-baseline, baseline-mirrored-steering-probe
- **Done when:** Separable DoG is implemented in brain_passes.wgsl; orientation, phase, and position probes pass without regression; throughput profile shows ≥2× DoG speedup; cargo fmt/clippy/test green.

---

### cortex-throughput-optimization-suite — Stack Remaining Optimizations (Pooling, Retina Size, Feature Count) to Hit ≥50% Budget

After separable DoG, the cortex is faster but likely still below the 50% budget. This task stacks additional optimizations: reduce pooling from 4×4 to 3×3, reduce retina from 32×32 to 24×24, and/or reduce the Gabor bank from 128 to 64 features. Each optimization is measured for throughput gain and probe regression. The combination lands when the cortex reaches ≥50% of baseline throughput (≥17,000 tps on N=10) with all probes passing.

**Steps:**
1. In `common.wgsl`, reduce the pooling radius: change the MAX-pool kernel from 4×4 to 3×3 (or 2×2 if needed). Re-run position-invariance probe; target <15%. Expected gain: ~1.2× throughput.
2. Reduce the retina resolution: change `VISUAL_RETINA_WIDTH` and `VISUAL_RETINA_HEIGHT` from 32×32 to 24×24 (or 20×20). The encoder input size shrinks by a factor of (32/24)² ≈ 1.78. Re-run orientation and phase probes; margins must stay >3× and <10%. Expected gain: ~1.5–2× throughput.
3. If still below 50% budget, reduce the Gabor bank: halve from 128 to 64 features by reducing the number of scales or orientations (e.g. 4 orientations × 2 scales = 8 vs 8 × 2 = 16). Re-run the food-separability diagnostic (encoder must still separate left/right with cosine-diff >0.5); if it drops below 0.5, revert and keep 128. Expected gain: ~1.4× throughput (fewer features = fewer complex-cell reductions).
4. Run the full throughput profile after each change; record the cumulative gain. Stop when throughput reaches ≥17,000 tps (≥50% of 34k baseline).
5. Confirm that all probes pass: orientation tuning ≥3×, phase invariance <10%, position invariance <15%, food separability >0.5 (cosine-diff).

- **Depends on:** separable-dog-optimization, baseline-mirrored-steering-probe
- **Done when:** Cortex throughput reaches ≥17,000 tps (≥50% of baseline) at N=10; all orientation, phase, position, and separability probes pass; changes are documented in comments explaining the trade-offs; cargo fmt/clippy/test green. The cortex now has a path to default if the workstream 0004 A/B verdict favors it (though the workstream 0003 emergent encoder is the focus).

---

## 0003 — Emergent Self-Organizing Encoder

### emergent-encoder-spike — Prototype Sparse/Predictive Self-Organizing Encoder

The hand-coded Gabor bank (0008) is biologically realistic but conflicts with the project's emergence principle. This spike prototypes a learned alternative: sparse coding (MSE + L1 regularization on the code) or predictive coding (predict next frame). The encoder learns a 128×128 dictionary of receptive-field prototypes from the statistics of the 32×32 retina, heritable and mutable via evolution. This spike determines: (a) whether the learning objective is tractable on the GPU (convergence in <100 ticks per frame), (b) whether learned structure shows orientation selectivity emergent from statistics (no hardcoded Gabor), (c) whether the learned encoder separates food-left/right as well as or better than the Gabor bank.

**Steps:**
1. Design the sparse-coding loss: after encoding the raw retina to 128 dims (via learned `encode_weights: 128×768`), decode it back via learned `decode_weights: 768×128` (transposed for simplicity) and compute `loss = ||retina - decode(encode(retina))||² + λ·||encode_result||₁`. Update both `encode_weights` and `decode_weights` via gradient descent at a small learning rate (e.g. 1e-4). Alternatively, replace the decode step with a next-frame prediction: `loss = ||future_retina - predictor(encode(current_retina))||²` where `predictor` is a small 128→128 linear layer. Document the choice in a comment.
2. Add the two matrices to the heritable config: `encode_weights` (128×768 = 98k floats, ~390KB per agent at fp32; feasible, as the 64KB per-agent limit applies to resident GPU state, not the full brain_state buffer). Alternatively, reduce to 64×768 if needed. Ensure mutation/inheritance paths in `config.rs` and `buffers.rs` pass the matrices through.
3. Implement the learning objective in a new WGSL function `coop_visual_encode_learned()` (or modify `coop_encode()`): each tick, after reading the sensory buffer, compute the sparse-coding or predictive loss and apply a small gradient step to update the weights. This is test-only; the actual agent learning still uses the main encoder, so the sparse-coding weights are a side-band training signal. Alternatively, integrate the loss into the main `coop_encode()` path so the learned encoder *replaces* the Gabor bank.
4. Write a test `sparse_encoder_converges_on_fixed_input()`: present the same retina frame for 1000 ticks and measure the reconstruction MSE (should decay toward zero), and the average code magnitude (should stay bounded; L1 regularization prevents blow-up).
5. Run the food-separability diagnostic on the learned encoder: present food-left and food-right to the trained encoder and measure the encoded cosine similarity (must be >0.5 difference, matching the Gabor baseline). If separability is poor, investigate whether the encoding capacity is too low (increase to 256 dims) or the learning rate is too high (causes oscillation).
6. Document the outcome in the workstream decision note `0003-EMERGENT-ENCODER-DECISION.md` (in this plan folder): if learned encoder matches Gabor separability, record ACCEPT — it is a candidate for 0004 A/B and unlocks `emergent-encoder-integration`. If it diverges (orientation selectivity visible but separability poor), record REJECT (or DEFER) with the measured issue and whether it is fixable via tuning.

- **Depends on:** baseline-mirrored-steering-probe, td-decay-schedule-audit, urgency-scaling-isolation, cortex-throughput-profile-baseline, separable-dog-optimization, cortex-throughput-optimization-suite
- **Done when:** A sparse or predictive coding objective is designed and prototyped in a test harness or side-band module; convergence is verified (MSE decays, code magnitude bounded); the food-separability diagnostic is run on the learned encoder; and the spike resolves into `0003-EMERGENT-ENCODER-DECISION.md` in the plan folder, recording ACCEPT/REJECT/DEFER with measured evidence on whether the learned encoder matches Gabor performance or shows potential (e.g. 'orientation selectivity emergent, but separability 0.35 < target 0.5; needs higher capacity or longer training'). Cargo fmt/clippy/test green.

---

### emergent-encoder-integration — Integrate Learned Encoder as Agent Lifetime Learning (Heritable, Evolving) (GATED)

**Gate:** The `emergent-encoder-spike` has landed and its `0003-EMERGENT-ENCODER-DECISION.md` records an ACCEPT verdict — sparse/predictive coding learns food separability comparable to or better than the Gabor bank. This task integrates the learned encoder into the main agent pipeline: the encoder weights are heritable (seeded at birth from a Gaussian, evolve via mutation), and they are updated on every brain tick via the sparse/predictive loss.

**Steps:**
1. Move the learned encoder from the test harness into the main `coop_encode()` function in `brain_passes.wgsl`. Replace the hardcoded Gabor bank logic with a call to the learned-encoder function. Keep the raw-retina → encoded-state transformation identical in shape (768→128 dims).
2. Add `encode_weights` and `decode_weights` (or just `encode_weights` if using a predictive loss) to the heritable genome in `BrainConfig` and `buffers.rs`. They are initialized as Gaussian (mean 0, std 0.1) at agent birth and inherited with mutation (e.g. Gaussian noise stddev 0.01).
3. Ensure the sparse-coding or predictive loss is applied on every brain tick: compute `loss = reconstruction_mse + λ·l1_penalty` and apply gradient descent to update the weights. Set the learning rate small (e.g. 1e-4) so the weights drift slowly (consistent with evolution acting on top).
4. Confirm orientation-selectivity emergence: after training an agent on a naturalistic world for 1 generation, extract the learned `encode_weights`, apply the vertical/horizontal-bar probe, and measure tuning ratios. Document the distribution: most codes should be isotropic; some should show >3× tuning emergent from the statistics, not hardcoded.
5. Verify food-separability is maintained: run the encoder-separability diagnostic on the learned encoder after 1 generation of training and confirm cosine-diff >0.5.
6. Add a flag `emergent_encoder_enabled` (default off) to the config so the learned encoder can be toggled independently of the Gabor bank (which remains behind `visual_cortex_enabled`). Both are off by default; the workstream 0004 A/B will flip one to true.

- **Depends on:** emergent-encoder-spike
- **Done when:** Binary outcome — *either* the spike's `0003-EMERGENT-ENCODER-DECISION.md` accepts the learned encoder and it is merged: the learned encoder replaces the Gabor bank in `coop_encode()`, `encode_weights`/`decode_weights` are heritable and evolve via mutation, the orientation-selectivity probe shows emergent tuning without hardcoding, the food-separability diagnostic passes (cosine-diff >0.5), and the `emergent_encoder_enabled` flag is added default-off; *or* the decision doc rejects the approach (separability or convergence insufficient) and the integration is reverted wholesale, with the negative result and its measured cause recorded in `0003-EMERGENT-ENCODER-DECISION.md` and the Gabor bank left as the encoder behind `visual_cortex_enabled`. Cargo fmt/clippy/test green on whichever branch lands. On the merge branch the learned encoder is a first-class citizen alongside the legacy raycast and Gabor bank, ready for 0004 A/B comparison.

---

## 0004 — A/B Comparison and Winner Promotion

### ab-harness-design — Design and Implement Seeded-Paired A/B Harness for Encoder Comparison

To fairly compare the learned encoder (workstream 0003) and the optimized Gabor cortex (workstream 0002) on equal footing, we need a seeded-paired harness: both arms run 16 generations of evolution on identical seed-deterministic worlds, differing only in the encoder. The harness records per-generation population mean fitness, lifespan, approach/avoidance intent fractions, food consumed, and death counts. A/A noise floor (two runs of the baseline arm) verifies the reproducibility.

**Steps:**
1. In `sandbox/evolution.rs` or a new test file `integration_ab_encoder_comparison.rs`, add a function `run_paired_evolution_ab(arm: EncoderChoice, seed: u64, generations: u16, population: usize) -> EvolutionResult` that executes a fixed-seed evolution and returns per-generation metrics (mean fitness, mean lifespan, approach_intent_fraction, avoidance_intent_fraction, food_consumed, death_count, variance/std for each).
2. Implement deterministic seeding: use `seed` to seed the world RNG and the population RNG so that food placement, agent mutations, and random exploration are identical across the two arms (only the encoder differs).
3. Add flags to `BrainConfig` to control encoder selection: `emergent_encoder_enabled` (workstream 0003) vs `visual_cortex_enabled` (workstream 0002 Gabor), both initially false (legacy baseline).
4. Create a test `ab_encoder_comparison_paired()` that (a) runs Arm A (emergent=false, cortex=false, legacy), (b) runs Arm B (emergent=true, cortex=false), on identical seeds for 16 generations, population 10, (c) compares mean fitness per generation (B - A), (d) computes 95% CI using bootstrap or paired t-test, (e) records a verdict: EMERGENT_WINS if CI does not cross zero and point estimate ≥+5%, INCONCLUSIVE otherwise. Save the per-generation data to a CSV file in `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/ab_paired_results.csv`.
5. Separately run A/A noise floor: execute Arm A twice on different seed-deterministic worlds (e.g. seed 1 and seed 2) and measure the inter-run variance (fitness, lifespan). If inter-run variance >2%, the signal is noisy; re-run with larger population or longer generations.

- **Depends on:** emergent-encoder-integration, cortex-throughput-optimization-suite
- **Done when:** The paired A/B harness is implemented; a test runs Arm A (legacy) vs Arm B (learned encoder) on identical 16-generation seeds, population 10; per-generation metrics are collected and saved; A/A noise floor is computed and reported; cargo fmt/clippy/test green. The harness is ready for analysis and decision in the next task.

---

### ab-analysis-and-verdict — Analyze A/B Results and Record Winner Promotion Decision

The A/B harness has run and produced per-generation metrics. This task analyzes the data: compute 95% CI on the fitness delta (B − A), check A/A noise floor, measure per-generation lifespan and intent-fraction trajectories, and apply the verdict rule (emergent wins if CI excludes zero and point estimate ≥+5%). The outcome is a decision document (0004-ENCODER-CHOICE-DECISION.md) with the verdict and logic for promoting the winner.

**Steps:**
1. Load the paired A/B CSV (`ab_paired_results.csv` with per-generation fitness, lifespan, intent for both arms) into a Python script or R notebook.
2. Compute the delta: `delta_per_gen = armB_fitness - armA_fitness`. Compute the mean delta across all 16 generations and the 95% CI using bootstrap (5000 resamples) or paired t-test. Report the point estimate and CI lower/upper bounds.
3. Interpret the verdict rule: if CI does not cross zero and point estimate ≥+5% (5% absolute fitness increase), the verdict is EMERGENT_WINS. If CI crosses zero or point estimate <+5%, the verdict is INCONCLUSIVE_KEEP_BOTH. If CI is negative and point estimate ≤−5%, the verdict is GABOR_WINS (unlikely, but possible if learned encoder regresses).
4. Analyze intent trajectories: plot approach_intent_fraction and avoidance_intent_fraction over the 16 generations for both arms. Do they diverge? Does one arm learn intents faster?
5. Check generalization: if time permits, run a third arm on a different seed-deterministic world (seed 3) and compare the fitness (should be similar to seeds 1–2 if the signal is robust).
6. Draft the decision document (0004-ENCODER-CHOICE-DECISION.md) with: (a) verdict (EMERGENT/GABOR/INCONCLUSIVE), (b) 95% CI table, (c) per-generation fitness plot, (d) intent trajectories, (e) A/A noise floor report, (f) recommendation for promotion logic.

- **Depends on:** ab-harness-design
- **Done when:** A/B results are analyzed; 95% CI is computed and reported; verdict is decided (EMERGENT_WINS, GABOR_WINS, or INCONCLUSIVE); intent trajectories are documented; A/A noise floor is reported; the decision document is drafted in `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/0004-ENCODER-CHOICE-DECISION.md`. The document records the promotion decision (promoting the winner to default or keeping both gated).

---

**End of plan 0017 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
