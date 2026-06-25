# XAgent Plan 0018 — Credit-Path Mechanism Attack: Auxiliary Loss, Trace Horizon, and Gradient Shaping

This plan executes three parallel workstreams attacking the credit-path bottleneck that plan 0017 falsified. Workstream 0001 designs and validates an auxiliary self-supervised steering loss that directly supervises vision-to-action alignment without shaped rewards, measuring improvement against the mirrored-steering probe. Workstream 0002 restructures eligibility traces and credit horizon via n-step returns and frame-synchronized updates to bridge the ~10-tick sensory latency between raw-gradient samples and action. Workstream 0003 diagnoses and fixes the credit-variance collapse (0017 measured mean|δ| ~9e-5 during foraging), applying gradient normalization or TD-error shaping to restore non-degenerate learning signal. Each workstream gates on steering alignment ≥0.70 while encoder-separability and food-visibility probes hold constant. Reuse 0017's baseline mirrored-steering probe and measurement-gated decision-doc pattern; respect homeostasis-only learning and disallow approach/danger reward shaping.

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

## 0001 — Auxiliary Steering Objective

### baseline-steering-and-variance-probe — Verify Baseline Steering Alignment and Measure TD-Error Variance

The steering probe `learning_probe_mirrored_steering_is_chance()` (integration.rs:2848–2929) is the gate for all three workstreams — steering alignment must move from the current chance band (0.38–0.62, baseline 0.489) above 0.62 (target ≥0.70). Additionally, 0017's variance test (`urgency_isolation_preserves_learning_signal()`, referenced in 0001-CREDIT-PATH-DECISION.md) measured the TD-error magnitude during steady-state foraging: mean|δ| ≈ 8.7e-5, std ≈ 6.5e-5. This baseline probe task re-measures both on current code (post-0017 reversion) to confirm the baseline is stable and document it as the gate for all three workstreams.

**Steps:**
1. Run the existing `learning_probe_mirrored_steering_is_chance()` test locally (GPU/Metal adapter) and record the measured alignment rate (correct/scored ratio). Document the result in a comment in integration.rs at the test location, including date, machine/adapter, and the measured rate (e.g. '2026-06-25 local Metal: aligned=229/468=0.489').
2. Add a companion test `baseline_td_error_variance_during_foraging()` to `integration.rs`. The kernel writes the per-tick TD error (`s_pred_td[S_TD_ERROR]`, brain_passes.wgsl:1227) into the decision buffer at `DECISION_MOTOR + 3` (brain_passes.wgsl:1552), and the CPU-side reader `read_agent_telemetry_blocking()` surfaces that slot as `AgentTelemetry.td_error` (gpu_kernel.rs:2974, field declared gpu_kernel.rs:356) — collect from that field, do **not** read the WGSL buffer directly. Use the existing probe harness (`build_probe_arena`, `probe_brain_config`, `PROBE_AGENT_COUNT`). Run dense-stride foraging (no movement pinning during the run; `probe_brain_config()` already sets `brain_tick_stride=1, vision_stride=1` but pins movement, so build the arena with a movement-enabled config) for 1000 ticks, collect every `td_error` sample, compute mean|δ|, std|δ|, min, max (five statistics), and assert mean|δ| is within `[5e-5, 1e-4]` (the expected range from 0017). Embed the full test verbatim:

   ```rust
   /// Re-measures the TD-error magnitude during steady-state foraging on
   /// current (post-0017-reversion) code to confirm the credit-variance
   /// baseline is stable. 0017 measured mean|δ| ≈ 8.7e-5, std ≈ 6.5e-5 at
   /// mid-energy steady state; this pins that as the gate for plan 0018.
   /// MEASURED <date> <machine/adapter>: mean|δ|=<…>, std=<…>, min=<…>, max=<…>.
   #[test]
   fn baseline_td_error_variance_during_foraging() {
       if !xagent_brain::GpuKernel::is_available() {
           eprintln!("Skipping: no GPU/fallback adapter available");
           return;
       }

       // Dense strides (fresh vision + brain decision every tick) with normal
       // movement enabled so agents actually forage — `probe_brain_config()`
       // pins movement, so build a movement-enabled config here instead.
       let forage_brain = BrainConfig {
           brain_tick_stride: 1,
           vision_stride: 1,
           ..Default::default()
       };
       let mut arena = build_probe_arena(&forage_brain, 17);
       arena.reset_bodies();

       const TICKS: u64 = 1000;
       let mut samples: Vec<f32> = Vec::with_capacity(TICKS as usize * PROBE_AGENT_COUNT);
       for t in 0..TICKS {
           arena.kernel.dispatch_batch(t, 1);
           for a in 0..PROBE_AGENT_COUNT {
               let td = arena.kernel.read_agent_telemetry_blocking(a as u32).td_error;
               samples.push(td.abs());
           }
       }

       assert!(!samples.is_empty(), "no TD-error samples collected");
       let n = samples.len() as f32;
       let mean = samples.iter().sum::<f32>() / n;
       let var = samples.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / n;
       let std = var.sqrt();
       let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
       let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
       eprintln!(
           "baseline TD-error variance: mean|δ|={mean:.3e}, std={std:.3e}, \
            min={min:.3e}, max={max:.3e}, n={}",
           samples.len()
       );

       // Expected range from 0017's variance probe (mean|δ| ≈ 8.7e-5).
       assert!(
           (5e-5..=1e-4).contains(&mean),
           "mean|δ| {mean:.3e} left the expected baseline band [5e-5, 1e-4] — \
            the credit-variance baseline shifted; re-pin before building on it"
       );
   }
   ```

   After the first local run, paste the measured five statistics (date, machine/adapter, mean|δ|, std, min, max) into the doc-comment `MEASURED` line.
3. Add a third test `encoder_food_side_separability_baseline()` that re-runs the encoder diagnostic from 0017: present food on the left and right, measure the cosine similarity of the left/right encodings, and report within-class cosine (same side, ~1.0) and between-class cosine (left vs right, ~0.0–0.2). Assert the cosine-diff (between − within ≈ −0.964 for Gabor, or ~0.0036 for raycast) holds to confirm the encoder is unchanged by this plan.

- **Depends on:** —
- **Done when:** The baseline steering probe runs and passes with alignment in [0.38, 0.62]; the TD-error variance test computes and reports mean|δ|, std, min, max; the encoder-separability test runs and reports cosine-diff within the expected margin; all three tests pass; cargo fmt/clippy/test green. This baseline is the measurement gate for all downstream tasks in all three workstreams.

---

### auxiliary-steering-spike — Prototype Auxiliary Self-Supervised Steering Objective

The mirrored-steering probe trains agents for 120 episodes on alternating left/right food, then pins movement and measures turn-alignment: baseline 0.489 (chance band 0.38–0.62). The hypothesis is that a direct auxiliary loss supervising vision-to-action alignment (turning direction toward the food bearing) could bypass the slow TD bootstrap across ~10-tick sensory latency and unlock steering. This spike prototypes one of three designs: direct-supervision (comparing turn logits to food bearing), contrastive encoding (maximizing cosine-diff between left/right), or predictive (predicting next-step bearing). The spike determines which design is tractable on the GPU, learns without regressing food-consumption or other probes, and produces measurable improvement in steering alignment.

**Steps:**
1. Select one design from ARCHITECTURE 0001 (direct supervision recommended for simplicity): compute the policy's turn logit/action output, extract the food bearing from the encoded features (or a secondary bearing-prediction head), and define a small loss comparing them (e.g. L2 distance if both are angles, or cross-entropy if the turn action is discrete). Set the auxiliary loss weight small (0.01–0.1 of TD updates) and learning rate small (1/10th of TD).
2. Implement the auxiliary loss in a test harness in `integration.rs` and add a test `auxiliary_steering_loss_converges_on_bearing()` that asserts the loss decays. Compute the per-tick auxiliary loss as the squared mismatch between the policy's turn output and the food bearing — both already available without new plumbing: read the turn output from `read_agent_telemetry_blocking().motor_turn` (gpu_kernel.rs:337-338) and compute the bearing from the arena's food/agent positions and the agent's yaw exactly as `score_turn_alignment` does (integration.rs:2212-2221, using `P_YAW` from `read_full_state_blocking`). Window the loss into an early third and a late third of the run and assert the late-window mean is below the early-window mean (the loss decays as the bearing→action mapping is learned). Embed the full test verbatim:

   ```rust
   /// Spike harness: a direct-supervision auxiliary steering loss is the
   /// squared mismatch between the policy turn output and the food bearing.
   /// As dense-stride foraging proceeds the bearing→turn mapping should
   /// tighten, so the late-window mean loss must fall below the early-window
   /// mean. This proves the auxiliary objective is learnable before it is
   /// promoted into `coop_predict_and_act()` (see auxiliary-steering-integration).
   /// MEASURED <date> <machine/adapter>: early=<…>, late=<…>.
   #[test]
   fn auxiliary_steering_loss_converges_on_bearing() {
       use std::f32::consts::{PI, TAU};
       use xagent_brain::buffers::{PHYS_STRIDE, P_YAW};

       if !xagent_brain::GpuKernel::is_available() {
           eprintln!("Skipping: no GPU/fallback adapter available");
           return;
       }

       // Dense strides so a fresh bearing→turn pair is produced every tick.
       let brain = BrainConfig {
           brain_tick_stride: 1,
           vision_stride: 1,
           ..Default::default()
       };
       let mut arena = build_probe_arena(&brain, 17);
       arena.reset_bodies();

       const TICKS: usize = 100;
       let mut losses: Vec<f32> = Vec::with_capacity(TICKS);
       for t in 0..TICKS {
           arena.kernel.dispatch_batch(t as u64, 1);
           let state = arena.kernel.read_full_state_blocking();
           let mut tick_loss = 0.0_f32;
           let mut scored = 0_usize;
           for a in 0..PROBE_AGENT_COUNT {
               let base = a * PHYS_STRIDE;
               let yaw = state[base + P_YAW];
               let turn = arena.kernel.read_agent_telemetry_blocking(a as u32).motor_turn;
               let dx = arena.food_pos[a].0 - arena.agent_pos[a].x;
               let dz = arena.food_pos[a].2 - arena.agent_pos[a].z;
               let mut bearing = dx.atan2(dz) - yaw;
               while bearing > PI {
                   bearing -= TAU;
               }
               while bearing < -PI {
                   bearing += TAU;
               }
               // Direct supervision: turn output should match the (sign of the)
               // bearing. L2 between the turn output and the normalized bearing.
               let target = (bearing / PI).clamp(-1.0, 1.0);
               let diff = turn - target;
               tick_loss += diff * diff;
               scored += 1;
           }
           losses.push(tick_loss / scored.max(1) as f32);
       }

       let third = TICKS / 3;
       let early: f32 = losses[..third].iter().sum::<f32>() / third.max(1) as f32;
       let late: f32 = losses[TICKS - third..].iter().sum::<f32>() / third.max(1) as f32;
       eprintln!("auxiliary steering loss: early={early:.4}, late={late:.4}");
       assert!(
           late < early,
           "auxiliary loss did not decay: early={early:.4}, late={late:.4} — \
            the bearing→action mapping is not being learned"
       );
   }
   ```

   After the first local run, paste the measured early/late means (date, machine/adapter) into the doc-comment `MEASURED` line.
3. Run the baseline steering probe with the auxiliary loss enabled and record alignment. If alignment moves above 0.62 (ideally ≥0.70), the spike is a candidate fix. If it stays in chance band or regresses other probes (food-visibility, food-consumption), note the failure and move to the next design or hypothesis.
4. Document the outcome in a decision note `0001-AUXILIARY-LOSS-DECISION.md` in the plan folder: if the auxiliary loss clears 0.70, record ACCEPT with the measured improvement (e.g. 'steering 0.489 → 0.72, +47%'); if it fails or regresses, record REJECT with the specific issue (e.g. 'loss converges but steering stays at 0.51, no improvement despite decoder convergence' or 'food-consumption regressed 15%, encoding disrupted').

- **Depends on:** baseline-steering-and-variance-probe
- **Done when:** A single auxiliary-loss design (direct supervision, contrastive, or predictive) is prototyped in a test harness; the loss convergence test passes (loss decays); the steering probe is re-run with the loss enabled; the decision document `0001-AUXILIARY-LOSS-DECISION.md` records ACCEPT (clears ≥0.70 gate with specifics on design and improvement) or REJECT (stayed in chance band, with measured cause); all tests pass; cargo fmt/clippy/test green.

---

### auxiliary-steering-integration — Integrate Auxiliary Steering Objective into Main Brain (GATED)

**Gate:** The `auxiliary-steering-spike` has landed and its `0001-AUXILIARY-LOSS-DECISION.md` records an ACCEPT verdict — the auxiliary loss moves steering alignment to ≥0.70 on the mirrored-steering probe while maintaining food-visibility and food-consumption at baseline. This task integrates the auxiliary loss into the main `coop_predict_and_act()` function, applies it on every brain tick (or at vision-frame boundaries), and adds a flag to enable/disable it independently.

**Steps:**
1. Move the auxiliary loss from the test harness into `brain_passes.wgsl`, integrated into `coop_predict_and_act()`. Add the loss computation in a new function (e.g. `compute_auxiliary_steering_loss()` or inline) that runs after the turn/forward action logits are computed and the encoding is fresh. Apply the loss update (gradient descent on the action weights and/or encoder weights) using a small learning rate (1/10th of the main TD rate, e.g. 0.01).
2. Add a flag `auxiliary_steering_loss_enabled` (default off) to `BrainConfig` (crates/xagent-shared/src/config.rs) so the loss can be toggled independently. Both the main TD path and the auxiliary loss ship in the code; the flag controls which signal dominates during training.
3. Run the mirrored-steering probe, encoder-separability diagnostic, and a food-consumption test to confirm: (a) steering alignment stays ≥0.70 (the fix), (b) encoder separability is unchanged, (c) food-per-episode is unchanged (no regression from the auxiliary update).
4. Document the auxiliary loss in a code comment at the loss-computation site, explaining: the design (direct supervision / contrastive / predictive), the learning rate rationale (1/10th of TD to preserve homeostasis-only dominance), and the gate (steering alignment ≥0.70 with encoder and food probes green). Ensure cargo fmt, clippy, and test green.

- **Depends on:** auxiliary-steering-spike
- **Done when:** Binary outcome — *either* the spike's `0001-AUXILIARY-LOSS-DECISION.md` accepts the auxiliary loss and it is merged: the loss is integrated into `coop_predict_and_act()`, a flag `auxiliary_steering_loss_enabled` (default off) is added, the mirrored-steering probe passes with alignment ≥0.70, and encoder-separability and food-visibility probes hold; *or* the decision doc rejects the approach (alignment did not improve, or other probes regressed) and no code lands — the main TD path stays unchanged. Cargo fmt/clippy/test green on whichever branch. If the auxiliary loss lands, it becomes the first candidate fix for plan 0018; if it fails, workstreams 0002 and 0003 proceed in parallel.

---

### credit-path-fix-lands — Land the Credit-Path Fix (One or More of 0001/0002/0003) or Record Negative Result

Three parallel workstreams (0001, 0002, 0003) have each run a spike-prototype-decide cycle. Each may have produced an ACCEPT or REJECT verdict in its decision document. At least one (ideally all three) must accept and land for the plan to succeed. This task consolidates the landed fixes into develop, updates the mirrored-steering probe baseline assertion to reflect the new ≥0.70 band, and ensures all three measurement gates (steering, encoder separability, food visibility) pass. If all three spikes rejected (no fix clears ≥0.70), this task records the negative result and keeps the chance-band baseline assertion intact.

**Steps:**
1. Review the three decision documents (`0001-AUXILIARY-LOSS-DECISION.md`, `0002-TRACE-HORIZON-DECISION.md`, `0003-GRADIENT-SHAPING-DECISION.md`) and identify which spikes accepted. For each accepted spike, ensure its integration task has landed (the fix is in brain_passes.wgsl, common.wgsl, config.rs, and enabled by a flag).
2. Update the mirrored-steering probe's baseline assertion in integration.rs (~2924–2928) to the new band reflecting the fix(es). If at least one fix landed and cleared ≥0.70, change the assertion from `(0.38..=0.62)` to `(0.70..=0.85)`. If all spikes rejected, leave the assertion at `(0.38..=0.62)` and document the rejection in the plan's STATUS.md.
3. Run the full integration-test suite locally and confirm: (a) the steering probe passes with the new expected alignment, (b) `encoder_food_side_separability_diagnostic()` runs and reports separability unchanged, (c) `learning_probe_baseline_turn_alignment_is_chance()` or equivalent food-visibility test passes, (d) no regressions in food-consumption, baseline-turn, or other existing probes.
4. Document the landed fix(es) in a summary comment in the relevant WGSL files, explaining which workstream(s) succeeded, the improvement delta (e.g. 'steering 0.489 → 0.74, +51%'), and the gate conditions (encoder, food visibility held). Ensure all three measurement gates pass: steering ≥0.70, encoder separability unchanged, food visibility unchanged.
5. Update the plan's STATUS.md (in docs/plans/0018-Credit-Path-Mechanism-Attack/) to record the outcome: either 'Landed: auxiliary loss + trace-horizon + gradient shaping; steering 0.489 → 0.75 (+53%); all probes green' (if all three landed), or 'Landed: auxiliary loss only; steering 0.489 → 0.72 (+47%); remaining two approaches (0002/0003) did not clear gate' (if partial), or 'All spikes rejected; no fix cleared 0.70 gate; steering remains at 0.489 (chance band); credit-path bottleneck carries to plan 0019' (if negative).

- **Depends on:** auxiliary-steering-integration, trace-horizon-integration, gradient-shaping-integration
- **Done when:** At least one of the three workstreams' fixes has landed (decision doc says ACCEPT and the integration task merged). The mirrored-steering probe assertion is updated to reflect the new baseline (≥0.70 if at least one fix landed, or stays [0.38..=0.62] if all rejected). All three measurement gates pass: steering alignment in the new band, encoder separability unchanged (cosine-diff holds), food-visibility and baseline-turn tests pass. Cargo fmt/clippy/test green. The plan's STATUS.md is updated with outcome. If all three spikes rejected, this is the final task; the negative result is recorded, and the credit-path bottleneck is marked for a future plan.

---

## 0002 — Credit Horizon and Trace Restructuring

### trace-horizon-spike — Prototype N-Step TD Returns or Frame-Synchronized Trace Decay

Eligibility traces decay via `TD_DISCOUNT × TD_LAMBDA = 0.873` per brain tick (common.wgsl:560, 564), giving (0.873)^10 ≈ 6.7% retention per vision-frame cycle (10 ticks at default strides). Fresh credit from a new vision frame arrives ~10 ticks after the previous frame, by which time old traces have largely decayed. This spike prototypes one of three approaches: n-step TD (accumulate rewards over the next n steps before bootstrapping), frame-synchronized decay (decay only at vision-frame boundaries), or decoupled trace-decay constant (independent of `TD_DISCOUNT × TD_LAMBDA`). The spike determines which design is feasible on the GPU, maintains TD stability, and produces measurable improvement in steering alignment (≥0.70 on the mirrored-steering probe).

**Steps:**
1. Select one design from ARCHITECTURE 0002 (frame-synchronized decay recommended for minimal code changes): modify the trace-update logic in brain_passes.wgsl (~1588–1603) to apply decay *only* when the current brain-tick matches a vision-frame boundary (brain_tick % vision_stride == 0). Within a frame, traces persist unchanged. Implement a test to verify decay is applied only at frame boundaries: zero all weights, seed traces to 1.0, run 5 ticks (assume vision_stride=10), and assert traces are still 1.0 at tick 4 and decayed at tick 10.
2. Alternatively, if selecting n-step TD: add a ring buffer of past rewards and values (length=10–20 steps, stored in `O_*` buffer slots). On every TD update, accumulate the sum of γ^i · reward_i for i=0..n-1, plus γ^n · V(s_n) − V(s), to compute the n-step TD error. Implement a test to verify n-step accumulation: manually set rewards for 10 steps, verify the accumulated sum matches the manual calculation.
3. Run the mirrored-steering probe with the prototype enabled and record alignment. If alignment moves above 0.62 (ideally ≥0.70), the trace-horizon fix is a candidate. If it stays in chance band, document the failure and consider the next design or hypothesis.
4. Document the outcome in `0002-TRACE-HORIZON-DECISION.md`: if the prototype clears ≥0.70, record ACCEPT with the design, decay/step parameters, and improvement percentage. If it fails, record REJECT with the measured misalignment and whether other probes (encoder, food visibility) regressed or stayed stable.

- **Depends on:** baseline-steering-and-variance-probe, auxiliary-steering-spike
- **Done when:** One trace-horizon design (n-step, frame-synchronized, or decoupled decay) is prototyped; a unit test confirms the mechanism is active (decay/accumulation working as intended); the steering probe is re-run; the decision document `0002-TRACE-HORIZON-DECISION.md` records ACCEPT (≥0.70 with details) or REJECT (chance band, with measured cause); all tests pass; cargo fmt/clippy/test green.

---

### trace-horizon-integration — Integrate Trace-Horizon Fix into Main Brain (GATED)

**Gate:** The `trace-horizon-spike` has landed and its `0002-TRACE-HORIZON-DECISION.md` records an ACCEPT verdict — the trace-horizon fix moves steering alignment to ≥0.70 on the mirrored-steering probe while maintaining encoder-separability and food-visibility at baseline. This task integrates the n-step returns, frame-synchronized decay, or decoupled trace constant into the main TD code, adds a flag to enable/disable it, and validates all downstream probes.

**Steps:**
1. Apply the trace-horizon fix to the main TD code in brain_passes.wgsl (trace update, ~1588–1603, and TD error computation, ~1220–1227). If frame-synchronized decay: modify the decay condition to check `brain_tick % vision_stride == 0` before applying decay. If n-step TD: integrate the ring-buffer accumulation into the TD error computation. If decoupled decay: replace `TD_DISCOUNT * TD_LAMBDA` with the new constant in the trace update.
2. Add a flag (if not already present) to enable/disable the fix independently. Frame-synchronized decay and n-step TD can be separate flags, or combined under a single `credit_horizon_fix_enabled` flag (default off) in BrainConfig.
3. Run the steering probe, encoder-separability diagnostic, food-visibility, and baseline-turn-alignment tests to confirm: (a) steering alignment ≥0.70, (b) encoder separability unchanged, (c) food-visibility and free-run foraging probes green.
4. Document the fix in code comments, explaining the design, the gate (steering ≥0.70 with probes green), and any tuning parameters (decay constant, n-step depth). Ensure cargo fmt, clippy, and test green.

- **Depends on:** trace-horizon-spike
- **Done when:** Binary outcome — *either* the spike's `0002-TRACE-HORIZON-DECISION.md` accepts the trace-horizon fix and it is merged: the fix is integrated into brain_passes.wgsl and/or common.wgsl, a flag is added (default off), the steering probe passes ≥0.70, and encoder/food-visibility/baseline probes hold; *or* the decision doc rejects the approach and no code lands. Cargo fmt/clippy/test green. If the fix lands, it is the second candidate for plan 0018; if it fails, workstream 0003 (gradient variance) proceeds in parallel.

---

## 0003 — Gradient Variance and Signal Shaping

### gradient-variance-diagnosis — Diagnose Gradient-Variance Collapse and Design Shaping Strategy

Plan 0017's variance probe measured TD-error magnitude during steady-state foraging: mean|δ| ≈ 8.7e-5, std ≈ 6.5e-5. The resulting weight updates (ACTION_WEIGHT_LEARNING_RATE · ACTOR_VECTOR_SCALE · δ ≈ 0.10 · 1/16 · 8.7e-5 ≈ 5.4e-7 per dimension per tick) are too small to move the policy on a single episode. This spike diagnoses the cause (degenerate reward signal during foraging vs mismatched learning-rate scales) and designs a shaping strategy (TD-error normalization, gradient scaling, auxiliary supervision, or bias-term amplification) that restores non-degenerate learning signal without violating the homeostasis-only contract.

**Steps:**
1. Extend the baseline TD-error variance probe (baseline-steering-and-variance-probe) to collect per-context statistics: separate mean|δ| and std|δ| when the agent is at low energy (mid-foraging) vs high energy (post-food, before decay). Also measure the variance of raw_gradient (energy_delta + integrity_delta) to confirm it is indeed small during foraging. Document all statistics.
2. Prototype a shaping strategy (select one from ARCHITECTURE 0003): (a) TD-error normalization: maintain a running EMA of |δ| and divide each δ by max(std(δ), ε); (b) gradient scaling: apply a soft scaling function (e.g. tanh(δ·k)) to amplify small δ before weight updates; (c) auxiliary-bias learning: add a small constant bias term to the action/value updates that learns independently of δ. Implement the shaping as a conditional compilation flag or a togglable function in brain_passes.wgsl.
3. Run the steering probe and variance test with the shaping enabled. Measure: (a) new mean|δ| after shaping (should be ~0.1–1.0 if the shaping is successful), (b) steering alignment (should move ≥0.62, ideally ≥0.70), (c) encoder separability and food-consumption (should remain stable). If alignment improves and other probes green, the shaping is a candidate fix.
4. Document the diagnosis and outcome in `0003-GRADIENT-SHAPING-DECISION.md`: if the shaping clears ≥0.70, record ACCEPT with the chosen strategy, the new mean|δ|, and the improvement percentage. If it fails or regresses, record REJECT with the measured issue and whether the gradient magnitude actually increased or stayed degenerate.

- **Depends on:** baseline-steering-and-variance-probe, auxiliary-steering-spike, trace-horizon-spike
- **Done when:** TD-error variance is re-measured with per-context breakdowns; one shaping strategy (normalization, scaling, or auxiliary bias) is prototyped; the steering probe is re-run; the decision document `0003-GRADIENT-SHAPING-DECISION.md` records ACCEPT (≥0.70 with new mean|δ| and choice) or REJECT (chance band, with measured variance post-shaping); all tests pass; cargo fmt/clippy/test green.

---

### gradient-shaping-integration — Integrate Gradient-Shaping Fix into Main Brain (GATED)

**Gate:** The `gradient-variance-diagnosis` has landed and its `0003-GRADIENT-SHAPING-DECISION.md` records an ACCEPT verdict — the gradient-shaping fix moves steering alignment to ≥0.70 on the mirrored-steering probe while maintaining encoder-separability and food-visibility at baseline. This task integrates the chosen shaping strategy (normalization, scaling, or auxiliary bias) into the main TD weight-update code, adds a flag to enable/disable it, and validates stability.

**Steps:**
1. Apply the gradient-shaping fix to the weight-update block in brain_passes.wgsl (~1247–1252 for actor weights, ~1232–1237 for critic weights). If TD-error normalization: insert `let norm_td_error = td_error / max(td_error_std_ema, epsilon)` before the weight updates. If gradient scaling: insert `let scaled_td_error = tanh(td_error * scale_factor)` before weight updates. If auxiliary bias: add a small constant term to the bias updates independent of δ.
2. Add the necessary state tracking if required (e.g. running EMA of |δ| for normalization). Store this in the brain_state buffer at an unused offset (e.g. `O_TD_ERROR_EMA_MAGNITUDE`).
3. Add a flag `gradient_shaping_fix_enabled` (default off) to BrainConfig to enable/disable the fix. Both the original TD path and the shaped path ship in the code; the flag controls which is used.
4. Run the full test suite: steering probe (≥0.70), encoder-separability diagnostic, food-visibility, baseline-turn-alignment, and a new stability test that measures weight-update magnitudes before/after shaping (to confirm updates are non-degenerate). Confirm all probes pass.
5. Document the shaping strategy in code comments, explaining the rationale (credit-variance collapse during foraging, mean|δ| ~1e-4 before shaping), the mechanism (chosen normalization/scaling/bias method), and the expected behavior (mean|δ| increases, steering improves). Ensure cargo fmt, clippy, and test green.

- **Depends on:** gradient-variance-diagnosis
- **Done when:** Binary outcome — *either* the spike's `0003-GRADIENT-SHAPING-DECISION.md` accepts the gradient-shaping fix and it is merged: the fix is integrated into brain_passes.wgsl (weight updates, and any state tracking in O_* offsets), a flag is added (default off), the steering probe passes ≥0.70, the encoder/food-visibility probes hold, and a weight-update magnitude test confirms updates are non-degenerate; *or* the decision doc rejects the approach (steering stayed in chance band, or shaping regressed stability) and no code lands. Cargo fmt/clippy/test green. If the fix lands, it is the third candidate for plan 0018; if it fails, the plan concludes with decision docs recording which fixes landed (if any).

---

**End of plan 0018 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
