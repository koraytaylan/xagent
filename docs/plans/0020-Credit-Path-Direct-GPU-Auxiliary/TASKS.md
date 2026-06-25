# XAgent Plan 0020 — Credit-Path-Direct-GPU-Auxiliary

Plan 0020 implements the GPU-integrated auxiliary self-supervision test that Plan 0018 explicitly did not run, falsifying or accepting the credit-alignment hypothesis under actual weight-update conditions. The plan has three workstreams: (0001) implement GPU-side auxiliary loss injection into `O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS` in `brain_passes.wgsl`, keyed to bearing targets computed on GPU each tick; (0002) measure steering alignment and control probes (encoder separability, food visibility) against the mirrored-steering baseline with auxiliary loss active; (0003) record the binary land-or-revert decision: if alignment clears ≥0.70, integrate the loss as a gated feature and update the steering baseline; if it stays ≤0.62, reject auxiliary supervision as a mechanism and document structural rethink requirements. The GPU test runs under the existing `GpuKernel::is_available()` self-skip guard, CI runs it after 0014's lavapipe installation.

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

## 0001 — GPU-Auxiliary-Loss-Implementation

### implement-gpu-auxiliary-loss — Implement GPU-Integrated Bearing-Aligned Auxiliary Loss

The 0018 spike tested auxiliary loss as a CPU-side measurement overlay that computed loss on `motor_turn` outputs but injected zero GPU weight updates; steering stayed at 0.509 (chance). The true test of auxiliary supervision requires computing bearing targets on the GPU and applying weight updates to `O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS` in the same tick. This task implements that mechanism.

**Steps:**
1. Open `crates/xagent-shared/src/config.rs`, locate the `BrainConfig` struct (≈24–222), and add a new field:
```rust
/// Enable direct-supervision auxiliary loss on turn/forward action channels.
/// When true, bearing-aligned gradients are injected into policy weights
/// in addition to TD(λ) credit. Zero-cost when false.
pub auxiliary_steering_loss_enabled: bool = false,
```
2. In `crates/xagent-brain/src/gpu_kernel.rs`, locate the `write_agent_heritable_config()` function and the uniform-buffer write path (≈400–450). Confirm that `BrainConfig` fields are serialized into the wconfig uniform buffer. Add a line to pack `auxiliary_steering_loss_enabled` as a u32 flag (0 or 1) into the buffer at a documented offset (e.g., `cfg_auxiliary_loss_enabled` at offset 28). Document the offset and size in a comment.
3. In `crates/xagent-brain/src/shaders/kernel/common.wgsl` (the shared constants header), add the flag constant:
```wgsl
/// Offset into the wconfig uniform buffer for the auxiliary-loss-enabled flag.
/// Set by BrainConfig::auxiliary_steering_loss_enabled.
const CFG_AUXILIARY_LOSS_ENABLED_OFFSET: u32 = 28u;
```
Confirm with the Rust side that the offset matches.
4. In `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, locate the `coop_predict_and_act()` function (≈1100–1400) and the weight-update section (≈1240–1260, where TD-error is applied). Inside the "Threads 0..ENCODED_DIMENSION: apply δ through traces" block (lines 1242–1256), immediately after the three TD weight updates (O_VALUE_WEIGHTS, O_ACTION_FORWARD_WEIGHTS, O_ACTION_TURN_WEIGHTS), add the auxiliary loss:
```wgsl
// Auxiliary bearing-alignment loss: direct supervision of turn output
// (only if enabled via BrainConfig flag).
let cfg_flags = u32(wc[CFG_AUXILIARY_LOSS_ENABLED_OFFSET / 4u]);
if ((cfg_flags & 1u) != 0u) {
    // Compute bearing to food: atan2(food_dx, food_dz) - agent_yaw.
    // Agent yaw is in s_agent_yaw (loaded from physics state earlier in the function).
    // Food position is in s_food_pos (loaded or passed from parent scope).
    let food_dx = s_food_pos.x - s_agent_pos.x;
    let food_dz = s_food_pos.z - s_agent_pos.z;
    let mut bearing = atan2(food_dx, food_dz) - s_agent_yaw;
    // Normalize bearing to [-π, π]
    while (bearing > 3.14159) { bearing -= 6.28318; }
    while (bearing < -3.14159) { bearing += 6.28318; }
    let bearing_target = clamp(bearing / 3.14159, -1.0, 1.0);
    
    // Get turn output (from the action output that was computed in this tick).
    // s_turn_output is the turn action, typically s_motor[S_TURN] after softmax/scaling.
    let turn_output = s_motor[S_TURN];
    
    // L2-based gradient: if turn_output is off from bearing_target, apply corrective update.
    let bearing_error = turn_output - bearing_target;
    let aux_learning_rate = 0.01;  // 1/10th of typical ACTION_WEIGHT_LEARNING_RATE (~0.1).
    
    // Update turn weights to reduce bearing_error.
    // Gradient: -aux_lr * (turn_output - bearing_target) * encoded_feature.
    brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] -=
        aux_learning_rate * bearing_error * s_encoded[tid];
}
```
If `s_food_pos` and `s_agent_yaw` are not in scope at this line, move the computation to where they are available (e.g., earlier in the tick, in shared memory).
5. Verify that `CFG_AUXILIARY_LOSS_ENABLED_OFFSET`, `O_ACTION_TURN_WEIGHTS`, `S_TURN`, and `s_encoded` are all in scope and match the buffer layout. Run `cargo build -p xagent-brain` to confirm no shader compilation errors.
6. Write a simple GPU-gated unit test (sketch, to verify the changes compile and run):
```rust
#[test]
fn auxiliary_loss_flag_passes_to_gpu() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut brain_config = BrainConfig::default();
    brain_config.auxiliary_steering_loss_enabled = true;
    // Verify the flag serializes without panic.
    let mut arena = build_probe_arena(&brain_config, 17);
    arena.kernel.dispatch_batch(0, 1);
    // If this runs without crashing, the flag passed through.
}
```
Add this to `crates/xagent-sandbox/tests/integration.rs`.

- **Depends on:** —
- **Done when:** The auxiliary loss mechanism is implemented and compiles: (1) `BrainConfig::auxiliary_steering_loss_enabled` flag defaults to false; (2) bearing target is computed on GPU from agent yaw and food position; (3) weight updates to `O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS` are applied when the flag is true, using gradient descent on the bearing error; (4) the test `auxiliary_loss_flag_passes_to_gpu` runs without crashing; (5) existing tests still pass; cargo fmt/clippy/test green.

---

## 0002 — GPU-Auxiliary-Steering-Probe

### gpu-auxiliary-steering-probe — Measure Steering Alignment with GPU-Integrated Auxiliary Loss Active

With the auxiliary loss implemented, the measurement probe runs the same mirrored-steering evaluation as 0018-0001 but with GPU weight updates active. The CPU-overlay test proved the mechanism does not work without GPU injection (0.509, chance band); this probe tests whether GPU injection unblocks it. The probe is a paired A/B: baseline (0018 result, 0.489, chance) vs auxiliary-enabled measurement.

**Steps:**
1. In `crates/xagent-sandbox/tests/integration.rs`, add a GPU-gated test `gpu_auxiliary_steering_alignment_probe` (≈9000–9200) that:
1. Embeds the self-skip guard verbatim:
```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```
2. Creates a training arena with auxiliary loss enabled:
```rust
let train_brain = BrainConfig {
    brain_tick_stride: 1,
    vision_stride: 1,
    auxiliary_steering_loss_enabled: true,  // NEW: enable GPU auxiliary loss
    ..Default::default()
};
let mut arena = build_probe_arena(&train_brain, 17);
arena.reset_bodies();
```
3. Runs 100 training ticks (dispatching the GPU kernel with auxiliary loss active):
```rust
const TRAIN_TICKS: usize = 100;
for t in 0..TRAIN_TICKS {
    arena.kernel.dispatch_batch(t as u64, 1);
    // Read telemetry and log diagnostics (optional, for offline inspection).
}
```
4. Switches to evaluation config, resets bodies, and runs steering evaluation for 60 ticks:
```rust
let eval_brain = probe_brain_config();
for a in 0..PROBE_AGENT_COUNT {
    arena.kernel.write_agent_heritable_config(a as u32, &eval_brain);
}
arena.reset_bodies();
const EVAL_TICKS: usize = 60;
let tick_offset = TRAIN_TICKS as u64;
let (correct, scored_eval) = score_turn_alignment(&mut arena, tick_offset, EVAL_TICKS);
```
5. Computes the alignment rate and 95% Clopper–Pearson CI:
```rust
let rate = correct as f64 / scored_eval.max(1) as f64;
let ci_lower = clopper_pearson_ci_lower(correct as u64, scored_eval as u64, 0.95);
let ci_upper = clopper_pearson_ci_upper(correct as u64, scored_eval as u64, 0.95);
eprintln!(
    "gpu_auxiliary_steering_alignment: {} / {} = {:.3} (95% CI [{:.3}, {:.3}])",
    correct, scored_eval, rate, ci_lower, ci_upper
);
```
6. Record (print to stderr, and optionally a file) the measured alignment rate and 95% CI WITHOUT asserting the steering verdict — the test must stay green on a REJECT (a chance-band outcome is a valid scientific result and must not fail the build). The only asserts in this test are the control probes in step 4; the ACCEPT/REJECT verdict is interpreted by the downstream `auxiliary-loss-decision-and-integration-gate` task.
2. Helper function for Clopper–Pearson CI (add to integration.rs if not already present):
```rust
fn clopper_pearson_ci_lower(successes: u64, trials: u64, confidence: f64) -> f64 {
    if successes == 0 { return 0.0; }
    // Simplified: use beta distribution quantile (or lookup table).
    // For exact calculation, use the relationship with the F-distribution:
    // Lower CI = 1 / (1 + (trials - successes + 1) / (successes * F_{2s, 2(n-s+1)}^{1-α/2}))
    // For a rough approximation (sufficient for this purpose):
    let p = successes as f64 / trials as f64;
    let z = 1.96;  // 95% confidence
    let margin = z * ((p * (1.0 - p)) / trials as f64).sqrt();
    (p - margin).max(0.0)
}

fn clopper_pearson_ci_upper(successes: u64, trials: u64, confidence: f64) -> f64 {
    if successes == trials { return 1.0; }
    let p = successes as f64 / trials as f64;
    let z = 1.96;  // 95% confidence
    let margin = z * ((p * (1.0 - p)) / trials as f64).sqrt();
    (p + margin).min(1.0)
}
```
Or, if the crate has a stats library, use it.
3. Run the test locally to record the measured alignment: `cargo test -p xagent-sandbox -- gpu_auxiliary_steering_alignment_probe --nocapture`. Capture the output alignment rate and CI.
4. Add control-probe asserts to rule out regression in encoder or arena (same as 0018):
```rust
// Encoder separability must hold (else arena or encoder regressed).
assert!(
    encoder_separability > 0.9,  // cosine-diff should remain ~0.964
    "encoder separability dropped to {}, possible regression",
    encoder_separability
);
// Food must be consumed (arena must be functional).
assert!(
    food_consumed > 0.0,
    "no food consumed during training, arena broken"
);
```

- **Depends on:** implement-gpu-auxiliary-loss
- **Done when:** The GPU-auxiliary-steering-alignment probe runs and records a measured alignment rate with 95% CI. The rate and CI are printed to stderr. The verdict is mechanical: (1) if CI lower bound >= 0.70, ACCEPT (steering clears gate); (2) if CI upper bound <= 0.62, REJECT (chance band, no improvement); (3) if CI straddles [0.62, 0.70], inconclusive (repeat with larger sample or seed variation). Control probes (encoder separability, food consumption) are the ONLY assertions — the test does NOT assert the steering verdict, so a REJECT (CI upper <= 0.62) keeps the build green. The measured rate + CI are recorded to stderr; the ACCEPT/REJECT verdict is rendered by the downstream `auxiliary-loss-decision-and-integration-gate` task using the mechanical rule above. Cargo fmt/clippy/test green.

---

### auxiliary-loss-decision-and-integration-gate — Auxiliary Loss Binary Decision: ACCEPT (Integrate) or REJECT (Carry Forward)

**Gate:** Depends on `gpu-auxiliary-steering-probe` task completing and recording a measured alignment rate with 95% CI. This task uses that measurement to render a binary decision: if CI lower >= 0.70, ACCEPT and trigger the integration task; if CI upper <= 0.62, REJECT and record the decision doc; if inconclusive, flag for re-run with larger sample.

**Steps:**
1. After `gpu-auxiliary-steering-probe` completes, examine the recorded alignment rate and 95% CI from the test output.
2. Render the decision using the mechanical rule:
- **ACCEPT** if `CI_lower >= 0.70`: The GPU-integrated auxiliary loss successfully raised steering alignment above chance. Proceed to `auxiliary-loss-integrate-and-update-baseline` task.
- **REJECT** if `CI_upper <= 0.62`: The GPU-integrated auxiliary loss failed to clear chance band. Proceed to `structural-rethink-fallback` task. Record decision doc `0001-GPU-AUXILIARY-LOSS-DECISION.md` (see below).
- **Inconclusive** if `0.62 < CI_lower < 0.70` or `CI_upper > 0.70`: The measurement is ambiguous. Recommend re-run with larger sample (more eval ticks or repeated seeds) or defer to 0021.
3. If REJECT, author `docs/plans/0020-Credit-Path-Direct-GPU-Auxiliary/0001-GPU-AUXILIARY-LOSS-DECISION.md`:
```markdown
# Decision: GPU-Integrated Auxiliary Steering Loss

**Date:** [date of decision]
**Status:** REJECT (or ACCEPT)
**Design:** GPU-side bearing-aligned auxiliary loss injecting gradients into `O_ACTION_TURN_WEIGHTS` and `O_ACTION_FORWARD_WEIGHTS`.

## Summary

[Measured alignment rate: X / Y = R (95% CI [L, U]). Verdict: CI upper bound U <= 0.62 (chance band). GPU-integrated auxiliary loss does not raise steering above chance.]

## Measurements

#### Steering Alignment After GPU-Auxiliary-Loss Training (Test: `gpu_auxiliary_steering_alignment_probe`)
- **Date/Adapter:** [date] [GPU adapter]
- **Training:** 100 dense-stride ticks with GPU-integrated auxiliary loss enabled
- **Evaluation:** pinned movement, 60 eval ticks
- **Measured alignment:** R correct / Y total = X (95% CI [L, U])
- **Chance band:** [0.38, 0.62]
- **Result:** [REJECT — CI upper U <= 0.62] or [ACCEPT — CI lower L >= 0.70]

#### Supporting Probes
- **Encoder separability:** [value] (unchanged, not regressed)
- **Food consumption:** [value] > 0 (arena functional)

## Why Reject (if applicable)

[Interpret the measurement. If REJECT: The GPU-integrated auxiliary loss with bearing targets and direct weight updates still leaves steering in the chance band. This falsifies auxiliary supervision (at this configuration) as the mechanism to unblock credit alignment. The gradient-shaping result (0018-0003) proved magnitude is not the bottleneck; the auxiliary test now shows that direct supervision of turn toward bearing, when applied via GPU weight update, is also insufficient. The problem is either: (1) the bearing target is misaligned with the actual credit signal (timing, scope), (2) TD(λ) with eligibility traces is structurally inadequate over 10-tick latency and requires n-step returns or per-timestep decay, or (3) the encoder cannot separate the action from the feature routing path under the homeostasis-only contract. Recommend structural rethink (Plan 0021).]]

## Measured Cause

[If REJECT: Primary cause is that GPU-integrated auxiliary loss with bearing supervision is not sufficient to raise steering above chance. This rules out auxiliary loss as a salvageable mechanism (at this scope) and implicates credit timing/structure.]

## Next Step

[If ACCEPT: Proceed to integration task (`auxiliary-loss-integrate-and-update-baseline`), which copies the spike code into the main branch, sets the default-off flag, and updates the steering baseline.]

[If REJECT: Record this decision doc. The credit-path problem carries forward to Plan 0021 with the structural-rethink hypothesis (`0003-STRUCTURAL-RETHINK-DECISION.md`). No integration task runs. The gated integration task (`auxiliary-steering-integration` from 0018) remains unstarted.]

**Authored by:** Plan 0020 Workstream 0002
**Gate:** Steering alignment must clear >= 0.70. Measured: [R]. Verdict: [ACCEPT / REJECT].
```

- **Depends on:** gpu-auxiliary-steering-probe
- **Done when:** A binary ACCEPT or REJECT decision is rendered based on the measured alignment rate and 95% CI from `gpu-auxiliary-steering-probe`. The decision doc (if REJECT) or integration task trigger (if ACCEPT) is recorded. No code changes in this task; it is a measurement interpretation task. Cargo fmt/clippy/test green (N/A, no code change).

---

### auxiliary-loss-integrate-and-update-baseline — Integrate GPU Auxiliary Loss and Update Steering Baseline (GATED)

**Gate:** This task runs only if `auxiliary-loss-decision-and-integration-gate` task records an ACCEPT verdict (CI lower >= 0.70). If REJECT, this task is not run and the plan moves to the structural-rethink fallback.

**Steps:**
1. If the decision doc records ACCEPT: the code in `implement-gpu-auxiliary-loss` already implements the mechanism. This task formalizes it as part of the default configuration and updates the baseline tests to reflect the new steering alignment band.
2. Update the steering baseline assertion in `auxiliary_steering_probe_baseline` (0018-0001 test, now inherited): change the expected band from [0.38, 0.62] (chance) to the measured ACCEPT band (e.g., [0.70, 0.90] if the acceptance rate is >0.80). Or, keep a separate `auxiliary_steering_probe_baseline_gpu_enabled` test if they should be tracked separately.
3. Update `docs/plans/STATUS.md` (root roll-up row for plan 0020) to record the ACCEPT verdict and the new steering alignment baseline.
4. Ensure all tests still pass: `cargo test -p xagent-sandbox --lib && cargo test -p xagent-sandbox --test contributing_guard && cargo test -p xagent-sandbox --test integration -- --include-ignored` (if any tests are gated on the flag).

- **Depends on:** auxiliary-loss-decision-and-integration-gate
- **Done when:** If ACCEPT is recorded: the auxiliary loss mechanism is confirmed working, the steering baseline assertion is updated to reflect the new alignment band (>= 0.70), and all tests pass. If REJECT: this task does not run. Cargo fmt/clippy/test green.

---

## 0003 — Structural-Rethink-Fallback-Decision

### structural-rethink-fallback — Structural-Rethink Fallback Decision Doc (GATED)

**Gate:** This task runs only if `auxiliary-loss-decision-and-integration-gate` records a REJECT verdict (CI upper <= 0.62). If ACCEPT, this task is not run. When REJECT, this task documents the structural hypothesis for the next plan.

**Steps:**
1. If the decision doc records REJECT: GPU-integrated auxiliary loss with bearing targets and direct weight updates does not raise steering above chance. The auxiliary mechanism is falsified. The gradient-shaping result (0018-0003) proved magnitude is not the bottleneck. The remaining hypotheses are structural: TD(λ) with eligibility traces over 10-tick sensory latency may be inadequate for credit alignment when the signal is sparse and misaligned in time.
2. Author `docs/plans/0020-Credit-Path-Direct-GPU-Auxiliary/0003-STRUCTURAL-RETHINK-DECISION.md`:
```markdown
# Decision: Credit-Path Structural Rethink Required

**Date:** [date]
**Status:** REJECT (auxiliary loss) → Structural rethink recommended
**Context:** GPU-integrated auxiliary loss failed to raise steering above chance (0.509 → [same band]). Gradient shaping proved magnitude not the bottleneck. Credit alignment, not magnitude, is the problem.

## Summary

[GPU auxiliary loss (Plan 0020 WS0001-0002) tested direct supervision of turn output toward bearing on the GPU with full weight updates. Measured steering alignment: [rate] (95% CI [L, U]). Result: REJECT (CI upper U <= 0.62). This falsifies auxiliary supervision as a sufficient mechanism and implicates credit **timing and alignment**, not learning-rate magnitude.]

## Root Cause

The credit signal suffers from two problems:
1. **Magnitude too small?** No (0018-0003 proved 400× amplification changes steering by zero).
2. **Timing / alignment misaligned?** Yes (this is the consistent failure mode across 0018-0001, 0018-0002, and now 0020).

The encoder separates food-bearing (cosine-diff 0.964 ~55×), but the policy cannot route it to the turn channel under any of: TD-only, TD+trace-decay tuning, TD+auxiliary-bearing-loss, TD+frame-sync traces, or TD+normalized-error. The common failure: the turn channel receives a sparse credit signal (mean|δ| ~9e-5 per tick) that accumulates over traces decaying at (0.873)^10 ≈ 6.7% per vision cycle.

## Structural Candidates

The next plan should prototype one of the following with the same measurement discipline:

| Candidate | What it is | Why it might work | Structural change |
|---|---|---|---|
| **n-step returns** | Replace λ-weighted TD with n-step bootstrapping (e.g., n=3–5 steps, or n=10 to span sensory latency) | Explicitly bridges the 10-tick sensory latency without relying on trace decay. Credit accumulates in lookahead, not in eligibility. | Modify `coop_predict_and_act()` in `brain_passes.wgsl` to compute n-step returns instead of TD-error scaling. |
| **Eligibility-decay-per-timestep rework** | Decay traces by a different schedule (e.g., per step vs per-tick, or reset at sensory boundaries). | Current traces decay as (0.873)^tick; at 10 ticks, they are ~6.7% of initial. Perhaps the decay is too aggressive or misaligned with the sensory rhythm. | Modify trace accumulation and decay in vision-tick and brain-tick phases. |
| **Eligibility-reset at vision boundary** | Reset (or sharply reduce) traces each time vision updates, so credit from one sensory cycle does not bleed into the next. | Vision latency is the dominant time constant; traces spanning multiple vision cycles may accumulate misaligned credit. | Add trace reset logic at vision update boundary. |
| **Auxiliary-head supervision with shared encoder** | Add a dedicated auxiliary head (separate from turn/forward) that predicts bearing directly, and backprop through the shared encoder. | The current auxiliary loss only updates action weights; a shared encoder supervision might unblock feature routing. | Add an auxiliary prediction head in `phase_brain_tail_from_scratch.wgsl` or `brain_passes.wgsl`. |

## Gating Conditions for Revisit

The structural rethink is locked until Plan 0021. Revisit if:
- A new encoder architecture (e.g., recurrent visual cortex, temporal convolution) provides a better feature separability signal.
- Multi-agent or environmental dynamics (food depletion, seasons) break the current credit deadlock.
- A measurement shows that n-step returns or trace-reset actually improves credit correlation (before measuring steering).

## Recommendation for Plan 0021

Pick **one** structural candidate from the table above. Prototype it under the same measurement discipline (GPU-gated, steering probe, decision doc). Do not mix candidates in a single plan. The candidates are not mutually exclusive (e.g., both n-step returns and eligibility-decay rework could be right), but the spike discipline requires testing them one at a time.

**Authored by:** Plan 0020 Workstream 0003
**Decision:** GPU auxiliary loss rejected (REJECT, CI upper <= 0.62). Next plan required with structural hypothesis.
```
3. Commit the decision doc (no code changes).

- **Depends on:** auxiliary-loss-decision-and-integration-gate
- **Done when:** If REJECT is recorded: a decision doc `0003-STRUCTURAL-RETHINK-DECISION.md` is authored and committed, documenting the root cause (credit alignment), the structural candidates, and the gating conditions for Plan 0021. If ACCEPT: this task does not run. Documentation-only task; cargo fmt/clippy/test green (N/A).

---

**End of plan 0020 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
