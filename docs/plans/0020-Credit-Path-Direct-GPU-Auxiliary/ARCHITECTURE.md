# Architecture — Plan 0020 (deltas)

> Edits in `crates/xagent-shared/src/config.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-sandbox/tests/integration.rs`,
> `docs/plans/STATUS.md`,
> `docs/plans/0020-Credit-Path-Direct-GPU-Auxiliary/0001-GPU-AUXILIARY-LOSS-DECISION.md`, and
> `docs/plans/0020-Credit-Path-Direct-GPU-Auxiliary/0003-STRUCTURAL-RETHINK-DECISION.md`.
> Line numbers are hints; locate by symbol (grep for `auxiliary_steering_loss_enabled`,
> `coop_predict_and_act`, `O_ACTION_TURN_WEIGHTS`, `O_ACTION_FORWARD_WEIGHTS`,
> `score_turn_alignment`, `is_available`).

## 0001 — GPU-Auxiliary-Loss-Implementation

Today `coop_predict_and_act()` (`brain_passes.wgsl:1049`, the weight-update body around
`brain_passes.wgsl:1242-1256`) applies TD-error-scaled weight updates to the action channels
through eligibility traces. Inside the "Threads 0..ENCODED_DIMENSION: apply δ through the traces"
block the actor weights are updated as
`brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * td_error * turn_trace`
(`brain_passes.wgsl:1251-1252`), and the forward channel mirrors it (`:1249-1250`). No auxiliary
objective modulates these weights; the turn channel receives only the TD-bootstrap credit that has
triple-falsified at chance. The 0018 spike measured an auxiliary loss on the GPU's `motor_turn`
outputs but injected zero weight updates, so the mechanism itself was never built.

Edits:

- **Add the auxiliary-loss-enabled flag to `BrainConfig`** (`crates/xagent-shared/src/config.rs`, the
  `BrainConfig` struct, ≈24-222). The field is the only knob that activates the new path and is
  default-off, so the
  shipped configuration and every existing test are byte-identical when it is unset:

```rust
/// Enable direct-supervision auxiliary loss on the turn/forward action channels.
/// When true, bearing-aligned gradients are injected into the policy weights in
/// addition to the TD(λ) credit; the targets derive only from agent state and
/// food geometry, not from any reward term. Zero-cost and inert when false.
pub auxiliary_steering_loss_enabled: bool,
```

- **Bind the flag into the heritable-config uniform** (`gpu_kernel.rs`, the
  `write_agent_heritable_config()` uniform-write path, ≈400-450): pack
  `auxiliary_steering_loss_enabled` as a `u32` (0 or 1) into the existing wconfig uniform buffer at a
  documented offset, mirroring how the other `BrainConfig` fields are serialized. The offset and size
  are documented in a comment so the WGSL side reads the same word.

- **Declare the matching offset constant in the shared header** (`common.wgsl`), so both sides agree
  on the wconfig layout by a single named constant rather than a magic index:

```wgsl
/// Word offset into the wconfig uniform buffer for the auxiliary-loss-enabled flag.
/// Mirrors BrainConfig::auxiliary_steering_loss_enabled (0 = off, 1 = on). The
/// Rust-side packing in write_agent_heritable_config() must write the same word.
const CFG_AUXILIARY_LOSS_ENABLED_OFFSET: u32 = 28u;
```

- **Inject the bearing-aligned auxiliary gradient** (`brain_passes.wgsl`, immediately after the three
  TD weight updates inside the `tid < ENCODED_DIMENSION` block, ≈1252). The target is the geometric
  bearing to food relative to the agent's yaw, normalized to [−1, 1]; the update is plain L2 gradient
  descent of the turn output toward that target, gated entirely on the flag:

```wgsl
// Auxiliary bearing-alignment loss: direct supervision of the turn output toward
// the geometric bearing-to-food. Active only when BrainConfig enables it; the
// target is self-supervised (agent yaw + food position), never a reward term.
let cfg_flags = u32(wc[CFG_AUXILIARY_LOSS_ENABLED_OFFSET / 4u]);
if ((cfg_flags & 1u) != 0u) {
    var bearing = atan2(food_dx, food_dz) - agent_yaw;
    // Wrap to [−π, π] so the target stays single-valued, then normalize to [−1, 1].
    while (bearing > 3.14159) { bearing -= 6.28318; }
    while (bearing < -3.14159) { bearing += 6.28318; }
    let bearing_target = clamp(bearing / 3.14159, -1.0, 1.0);
    let bearing_error = turn_output - bearing_target;
    // 1/10th of the TD actor rate (~0.1) so auxiliary updates stay subordinate.
    let aux_learning_rate = 0.01;
    brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] -=
        aux_learning_rate * bearing_error * s_encoded[tid];
}
```

  The agent yaw, food offset (`food_dx`/`food_dz`), and turn output are read from the shared-memory
  state already loaded earlier in the tick; if any are not in scope at the injection site the
  computation moves to where they are available rather than re-reading from storage.

Properties that make this safe:
- The auxiliary loss is an additional self-supervised objective, not a reward-shaping term: its
  target derives only from agent state (turn output, yaw) and geometry (food position), so it honors
  the homeostasis-only contract — decision rule in SCOPE (locked decisions).
- The update reuses `s_encoded` — the same features already flowing through both actor and critic
  paths — so no new data dependency or buffer is introduced; the write touches only the existing
  `O_ACTION_TURN_WEIGHTS` slot.
- The flag is default-off, so production and every existing test run the untouched TD-only path;
  enabling it is a measurement-only choice confined to the new probe.
- The bearing target is recomputed each tick and clamped to [−1, 1], so `bearing_error` and the
  resulting gradient magnitude stay bounded regardless of geometry.
- The auxiliary learning rate (0.01) is 1/10th of the TD actor rate (~0.1), so auxiliary updates
  scale down relative to the primary learning signal and cannot dominate the credit path.

## 0002 — GPU-Auxiliary-Steering-Probe

Today the mirrored-steering baseline (`auxiliary_steering_probe_baseline`, recorded in 0018's
`STATUS.md`) runs agents in pinned movement (`movement_speed = 0`) with food placed at a fixed
bearing and scores the turn-alignment rate under the default TD-only learning signal via
`score_turn_alignment()`. The result was 0.489 (446/911), inside the chance band [0.38, 0.62], with
encoder separability confirmed (cosine-diff 0.964). The probe measured CPU-side and never exercised
GPU weight updates, so it falsified the CPU overlay, not the mechanism.

Edits:

- **Add a GPU-gated alignment probe** (`integration.rs`, ≈9000-9200) named
  `gpu_auxiliary_steering_alignment_probe`, embedding the standard self-skip guard verbatim so it
  does not run without an adapter:

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

  The probe (1) builds a training arena with `auxiliary_steering_loss_enabled: true` and dense
  strides (`brain_tick_stride = 1`, `vision_stride = 1`); (2) runs 100 training ticks so the GPU
  kernel updates the turn/forward weights under both TD and auxiliary loss; (3) switches to
  `probe_brain_config()`, resets bodies, and runs the 60-tick steering evaluation at
  `movement_speed = 0`; (4) scores the turn-alignment rate with the same `score_turn_alignment()` as
  the baseline so the comparison is like-for-like; (5) computes a 95% Clopper–Pearson CI on the
  success count and renders the mechanical verdict.

- **Render the verdict by the CI bounds, not the point estimate.** The band is fixed and binary so
  there is no interpretation gap — decision rule in SCOPE (locked decisions):

```rust
// ACCEPT iff the 95% CI clears the gate; REJECT iff it is entirely in the chance
// band; otherwise inconclusive — re-run with a larger sample or a fresh seed.
let verdict = if ci_lower >= 0.70 {
    Verdict::Accept
} else if ci_upper <= 0.62 {
    Verdict::Reject
} else {
    Verdict::Inconclusive
};
```

- **Keep the control probes unchanged** (encoder separability and food consumption): assert
  separability stays ≈0.964 and food consumed > 0 during training, identical to the 0018 baseline,
  so a regression in the encoder or the arena is caught before the alignment number is trusted.

Properties that make this safe:
- The probe is GPU-gated with the standard self-skip guard, so it never runs without an adapter and
  matches the project's CI/local model — decision rule in SCOPE (locked decisions).
- The measurement is structurally identical to the 0018-0001 baseline — same strides, same
  evaluation geometry, same `score_turn_alignment()` — so the only changed variable is that the
  auxiliary loss now injects GPU weight updates instead of measuring CPU-side.
- The verdict is mechanical (95% Clopper–Pearson CI) and binary (ACCEPT ≥ 0.70 or REJECT ≤ 0.62),
  leaving no judgment to the executor.
- No new assumption is carried: the baseline encoder and arena are already proven, so the probe tests
  only the one hypothesis — whether GPU-injected auxiliary loss unblocks the credit path.

## 0003 — Structural-Rethink-Fallback-Decision

Today the credit path is a single TD(λ) actor-critic with eligibility traces that decay per tick and
accumulate over ~10 ticks of sensory latency (`coop_predict_and_act()`, `brain_passes.wgsl:1242-1256`).
When the GPU-injected auxiliary loss (0001) is measured (0002) and still fails to clear 0.70, the
bottleneck is neither magnitude (0018-0003 amplified mean|δ| ~400× to zero effect) nor an untested
mechanism (auxiliary loss will have been built and run). The remaining hypothesis is structural: the
TD(λ) trace architecture may be inadequate over a 10-tick latency with a sparse, time-misaligned
credit signal.

Edits:

- **Author the structural-rethink decision doc, conditional on the 0002 verdict.** This task produces
  only `0003-STRUCTURAL-RETHINK-DECISION.md` and no code — decision rule in SCOPE (locked decisions):
  - If the probe ACCEPTs, this doc is not written; the plan ends with the integration task.
  - If the probe REJECTs, the doc records the **measured evidence** (steering alignment from the
    auxiliary-loss probe, encoder separability, TD-error variance over the 10-tick window), the
    **structural candidates**, and the **gating conditions** for revisiting.
- **Frame the structural candidates as sketches for Plan 0021, not commitments**, each tied to where
  the prototype would land:
  1. **n-step returns** — replace the λ-weighted TD with n-step bootstrapping (n≈3-5, or n=10 to span
     the latency) so credit accumulates in lookahead rather than in eligibility decay; prototyped in
     `coop_predict_and_act()`.
  2. **Eligibility-decay-per-timestep rework** — decay traces on a different schedule (per step rather
     than per tick, or reset at sensory boundaries) so the trace lifetime matches the vision rhythm.
  3. **Auxiliary-head supervision with a shared encoder** — add a dedicated bearing-prediction head,
     separate from turn/forward, that backprops into the shared encoder, so the supervision can
     unblock feature routing rather than only the action weights.
- The doc closes with the revisit gate ("Unlocked by Plan 0021 if a structural candidate is chosen;
  revisit if a new measurement breaks the credit-alignment deadlock") and points at the relevant
  ARCHITECTURE sections that would describe the chosen prototype.

Properties that make this safe:
- The task is conditional (gated on the 0002 verdict) and produces only a decision doc — it alters no
  running code and introduces no new assumption.
- The reasoning is measurement-backed: it uses the actual probe results rather than speculation, so it
  grounds the next plan's design in observed failure.
- The structural candidates are framed as one-at-a-time hypotheses to prototype under a new plan, not
  a multi-change commitment, preserving the falsify-or-kill spike discipline.
