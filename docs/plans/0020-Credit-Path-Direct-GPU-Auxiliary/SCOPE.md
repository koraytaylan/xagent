# Scope — Plan 0020

> Implement and test GPU-integrated auxiliary self-supervision loss to inject
> bearing-aligned gradients directly into policy weights, falsifying or clearing
> the credit-alignment bottleneck that has triple-rejected under CPU overlay.

## Why this plan

The credit-path bottleneck has now triple-falsified under CPU measurement and requires the GPU-injected test to avoid ritual non-progress.

1. **CPU-overlay auxiliary loss was never actually tested as a mechanism.**
   `0018-0001` built a CPU-side measurement of the existing TD path
   (`auxiliary_steering_loss_converges_on_bearing`, decision doc line 19: "loss
   convergence is an artifact of the GPU kernel's existing TD(λ) credit path …
   does not cause it"). The test computed loss on GPU outputs but injected zero
   GPU weight updates. Steering stayed at `0.509` (chance band `[0.38, 0.62]`),
   but this result is a falsification of "CPU overlay" not "auxiliary loss as a
   mechanism" — the mechanism itself was never built. GLM-5.2 review 2026-06-25
   F5 states: "The actual mechanism (an auxiliary self-supervised loss that
   **does** inject a vision→action gradient into the GPU weights) was never built
   or tested." (`docs/plans/0018-Credit-Path-Mechanism-Attack/0001-AUXILIARY-LOSS-DECISION.md:19-41`,
   `docs/reviews/2026-06-25-glm-52.md:F5`)

2. **The credit-path learning signal is separated but misaligned.** The 0018-0003
   gradient-shaping result (decision doc `0003-GRADIENT-SHAPING-DECISION.md`)
   normalizes TD error to amplify mean|δ| by ~400×, yet steering stays at `0.498`
   (chance). This falsifies magnitude as the bottleneck and proves the problem is
   **credit timing and alignment**, not signal strength. The encoder separates
   food-bearing (cosine-diff 0.964 ~55×, Gabor 18–24×) but the policy cannot
   route that separable signal to the turn channel under TD bootstrap alone.
   Direct supervision of the turn output toward bearing is the untried mechanism.
   (`docs/plans/0018-Credit-Path-Mechanism-Attack/STATUS.md:32-33`,
   `docs/reviews/2026-06-25-glm-52.md:F3:174`)

3. **The review explicitly identifies GPU-injected auxiliary loss as the
   unmeasured candidate.** GLM-5.2 F5 recommended path: "If auxiliary
   self-supervision is in scope for 0019, the GPU-integrated version must be built
   before any REJECT is recorded against the mechanism." Plan 0020 (this plan)
   runs that test. The 0017-WS0001 carry-forward (trace decay failed) and 0018's
   three rejects (auxiliary CPU-only, frame-sync disabled-at-probe-stride,
   gradient-shaping proved magnitude not bottleneck) create a critical gate:
   without GPU-integrated auxiliary loss, the next plan has no new hypothesis
   class and the process becomes ritual. (`docs/reviews/2026-06-25-glm-52.md:F3:162-191`,
   `docs/plans/0018-Credit-Path-Mechanism-Attack/STATUS.md:35-39`)

4. **Homeostasis-only contract allows direct supervision of behavior.** The plan
   does not introduce a new reward term or goal signal; auxiliary loss is a
   **self-supervised auxiliary objective** derived from agent state (its own turn
   output) and geometry (bearing to food), not from external reward. The agent
   still learns under homeostatic TD pressure; auxiliary loss is an additional
   self-critic that injects bearing-aligned gradients into action channels. This
   is in-scope under 0012's "no reward shaping" gate because it is not shaping the
   primary TD credit, only adding a direct-supervision term to the action weights.
   (`docs/plans/0012-Homeostatic-Only-Learning-Restoration/SCOPE.md`,
   `docs/reviews/2026-06-25-glm-52.md:F3:177-178`)

**Provenance.** Every load-bearing claim verified against source and review:
the 0018-0001 decision doc (`docs/plans/0018-Credit-Path-Mechanism-Attack/0001-AUXILIARY-LOSS-DECISION.md`:19, 36)
confirms CPU-side measurement only, no GPU weight updates; the 0018 STATUS
(`docs/plans/0018-Credit-Path-Mechanism-Attack/STATUS.md`:23) records "auxiliary
loss … CPU-side measurement overlay … injects no GPU weight updates"; the GLM-5.2
review (`docs/reviews/2026-06-25-glm-52.md` F3:171, F5:232–241) confirms
triple-falsified under CPU and calls the GPU-integrated test the path forward;
`brain_passes.wgsl` (`crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl:1249-1252`)
shows the existing TD-inject pattern `brain_state[brain_base + O_ACTION_TURN_WEIGHTS + tid] += ACTION_WEIGHT_LEARNING_RATE * ACTOR_VECTOR_SCALE * td_error * turn_trace`;
encoder separability is proven in the 0017 cortex ablation tests and 0018
baseline probes (embedded in the test suite).

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| The 0018-0001 spike already tested auxiliary self-supervision and rejected it (0.509, chance). | `docs/plans/0018-Credit-Path-Mechanism-Attack/0001-AUXILIARY-LOSS-DECISION.md`:19 | The spike was a CPU-side measurement overlay that computed loss on GPU outputs but injected **zero** GPU weight updates. It falsifies the measurement method, not the mechanism — the gradient-injecting version this plan builds was never run. |
| Magnitude of the TD credit signal is the bottleneck; amplifying it should restore steering. | `docs/plans/0018-Credit-Path-Mechanism-Attack/STATUS.md`:32-33 (0018-0003) | Gradient shaping amplified mean|δ| by ~400× and steering stayed at `0.498` (chance). Magnitude is falsified; the residual problem is credit **timing and alignment**. |

## In scope

- **0001 — GPU-Auxiliary-Loss-Implementation.** Implement auxiliary
  self-supervision loss that computes bearing targets on GPU and injects
  bearing-aligned gradients into `O_ACTION_TURN_WEIGHTS` and
  `O_ACTION_FORWARD_WEIGHTS` in `brain_passes.wgsl`, integrated alongside the
  existing TD(λ) credit path under an auxiliary-loss-enabled flag. See [TASKS.md](TASKS.md).
- **0002 — GPU-Auxiliary-Steering-Probe.** Measure steering alignment under
  GPU-integrated auxiliary loss active; paired A/B against the homeostatic
  baseline (0018-0001 CPU overlay); record the verdict (ACCEPT/REJECT) with 95%
  CI and a structured decision doc if steering clears ≥0.70 or rejects ≤0.62. See [TASKS.md](TASKS.md).
- **0003 — Structural-Rethink-Fallback-Decision.** If GPU auxiliary loss still
  fails (steering ≤0.62), document whether TD(λ) actor-critic with per-tick
  eligibility traces is structurally adequate over 10-tick sensory latency, or
  whether credit-alignment architecture redesign (n-step returns,
  eligibility-decay-per-timestep, auxiliary-head supervision) is necessary;
  record a decision doc with measurement-backed reasoning and gating conditions
  for revisiting. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| CPU-side auxiliary-loss measurement stays at chance (0.509) but never injected GPU weight updates, so the mechanism itself was never tested (1) | `0001` |
| Gradient-shaping proves magnitude is not the bottleneck (400× amplification, zero steering improvement); timing and alignment are the problem (2) | `0001` |
| Steering alignment triple-falsified (0017, 0018×3) and stayed in chance band [0.38, 0.62]; next plan requires a new hypothesis class or the process is ritual (3) | `0002` |
| Encoder food-bearing separability is high (cosine-diff 0.964 ~55×) but the policy cannot route it to the turn channel under TD alone (4) | `0002` |
| Homeostasis-only contract allows self-supervised auxiliary objectives; GPU-injected bearing-aligned gradients do not violate 0012's purity gate (5) | `0001` |

## Locked decisions

- **Auxiliary loss is self-supervised, not reward-shaping; homeostasis-only
  contract holds.** The auxiliary loss derives targets from agent state (turn
  output, yaw) and geometry (food position), not from external reward. It is a
  direct-supervision objective on the action channel, not a goal signal or reward
  term. This respects the homeostasis-only contract (Plan 0012), which prohibits
  reward shaping but allows self-supervised losses that steer internal feature
  routing. The mechanism is either sufficient (ACCEPT, integrate) or insufficient
  (REJECT, carry to structural rethink); it does not violate the purity gate
  either way.
- **GPU test runs with the standard self-skip guard; CI executes after 0014
  lavapipe installation.** The auxiliary loss probe is GPU-gated
  (`GpuKernel::is_available()` self-skip guard). Local runs without an adapter
  self-skip gracefully. CI runs after Plan 0014 lands (which installs Mesa
  lavapipe and sets `XAGENT_REQUIRE_GPU` to ensure GPU tests execute). This is the
  project's standard GPU test model.
- **Binary ACCEPT/REJECT decision based on 95% CI; no gradual gates.** The verdict
  is mechanical and binary: if the 95% Clopper–Pearson CI lower bound ≥0.70,
  ACCEPT (integrate); if the upper bound ≤0.62, REJECT (carry to structural
  rethink); if inconclusive (CI straddles [0.62, 0.70]), recommend a re-run with a
  larger sample. This avoids an interpretation gap and grounds the next plan's
  design decision.
- **No code integration unless ACCEPT clears 0.70; REJECT triggers structural
  rethink, not retry.** If the auxiliary loss measurement rejects (CI upper
  ≤0.62), the plan does not spawn a new spike or variant. Instead, the decision
  doc `0003-STRUCTURAL-RETHINK-DECISION.md` documents structural candidates
  (n-step returns, trace-decay rework, encoder supervision) for Plan 0021. This is
  the commitment to the "falsify-or-kill" discipline: if the mechanism fails at
  the GPU level, the next plan must try a structurally different hypothesis, not a
  tuning variant.

## Out of scope

- **Encoder architecture redesign (e.g., recurrent visual cortex, temporal
  convolution).** Encoder separability is already high (cosine-diff 0.964, ~55×).
  The bottleneck is downstream (credit alignment, not feature extraction). Encoder
  redesign is a separate research direction and is deferred to Plan 0021+ only if
  the structural rethink (Plan 0021) identifies shared-encoder supervision or
  feature-routing changes as the path forward.
- **Multi-agent dynamics or environmental pressure (food depletion, seasons).**
  The current setup (fixed food, single-agent, homeostatic credit) is the
  controlled measurement regime. Environmental pressure would add complexity and
  is a separate plan (out of scope for the credit-path spike discipline). Revisit
  if the structural rethink (Plan 0021) suggests that environmental dynamics are
  necessary to ground credit.
- **Trace-decay or TD-parameter re-tuning (Plan 0017 follow-ups).** Plan 0017
  falsified simple parameter tuning (trace decay 0.99, urgency isolation). The
  gradient-shaping result (0018-0003) proved magnitude is not the bottleneck.
  Tuning variants are not new hypotheses; the next plan (0021) should try
  structural changes (n-step returns, per-timestep decay) rather than more
  parameter searches.
- **Cortex integration or vision-stride variation.** Cortex is orthogonal to
  credit alignment and is throughput-blocked on real GPU (0017 F2). Vision-stride
  variation would confound the auxiliary-loss measurement (the baseline probe runs
  at vision_stride=1, so auxiliary training must also use dense strides). Cortex
  and stride tuning are separate plans.
- **Graduation decision for the `auxiliary_steering_loss_enabled` flag (if
  ACCEPT).** If the probe accepts (clears ≥0.70 steering), the flag lands as
  default-off (following the 0014 pattern for `effort_rebased_fitness`). A separate
  plan (or decision doc) will record the graduation decision (flip to default-on or
  retire). This plan measures the mechanism only.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
