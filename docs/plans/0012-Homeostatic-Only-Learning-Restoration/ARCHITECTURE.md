# Architecture — Plan 0012 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/tests/raw_gradient_homeostatic_only.rs`,
> `crates/xagent-brain/tests/learning_signal_baseline.rs`,
> `README.md`, and `crates/xagent-brain/README.md`.
> Line numbers are hints; locate by symbol (grep for `shaping`,
> `APPROACH_SHAPING_GAIN`, `P_PREV_POTENTIAL`, `raw_gradient`,
> `P_RAW_GRADIENT_OUT`, `CFG_DANGER_PERCEPT_ENABLED`).

## 0001 — Measurement-Baseline

Today the homeostasis pass `coop_habituate_homeo` in `brain_passes.wgsl`
(`brain_passes.wgsl:823`) computes the learning signal `raw_gradient` as a local
and publishes only its *amplified* form to the workgroup-shared `s_homeo[1u]`
(`brain_passes.wgsl:846`, `raw_gradient_amplified = raw_gradient * (1 + urgency)`).
The CPU-readable physics slot `P_GRADIENT_OUT` (`buffers.rs:174`, index 29) carries
the *blended/smoothed* `gradient` (`brain_passes.wgsl:1470`), not the raw
pre-amplification signal — so `AgentTelemetry::gradient` (`gpu_kernel.rs:344`) cannot
observe `raw_gradient`, and there is no test-visible handle on it. Before removing
the shaping terms we add that handle and capture the BEFORE distribution, so both
removal phases (0002, 0003) gate on this baseline and the post-removal guard test
(0005) reads the same field.

Edits:

- **Add a CPU-readable `raw_gradient` debug slot** following the existing `P_*_OUT`
  idiom. In `buffers.rs`, append a new physics-stride slot immediately before
  `PHYS_STRIDE` and bump the stride from 44 to 45 (mirror the same const and stride
  in `common.wgsl`, where `P_GRADIENT_OUT: u32 = 29u` lives and `PHYS_STRIDE` is the
  matching WGSL const). Update the `buffers.rs` PHYS_STRIDE layout-parity test
  (≈line 1488) to include `P_RAW_GRADIENT_OUT`, so the `stride == max offset + 1`
  assertion holds at 45.

```rust
/// Pre-amplification homeostatic learning signal `raw_gradient`
/// (energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT + shaping terms),
/// written by `coop_habituate_homeo` for CPU readback. Per-agent live state,
/// never serialized.
pub const P_RAW_GRADIENT_OUT: usize = 44;
pub const PHYS_STRIDE: usize = 45;
```

- **Publish `raw_gradient` through workgroup memory** in `brain_passes.wgsl`. The
  local is in scope only at the assembly site (`brain_passes.wgsl:823`), not at the
  telemetry write-out, so widen `s_homeo` from `array<f32, 6>` to `array<f32, 7>`
  (`brain_passes.wgsl:31`), set `s_homeo[6u] = raw_gradient;` right after
  `s_homeo[1u] = raw_gradient_amplified;` (`brain_passes.wgsl:846`), and write
  `physics_state[phys_base + P_RAW_GRADIENT_OUT] = s_homeo[6u];` in the thread-0
  telemetry block beside `physics_state[phys_base + P_GRADIENT_OUT] = gradient;`
  (`brain_passes.wgsl:1470`).

- **Surface the slot on `AgentTelemetry`** (`gpu_kernel.rs:330`): add a
  `pub raw_gradient: f32` field beside `pub gradient: f32` (`gpu_kernel.rs:344`),
  read it from the physics buffer in both readback paths —
  `read_agent_telemetry_blocking` (`gpu_kernel.rs:2944`) and `try_collect_telemetry`
  (`gpu_kernel.rs:3153`) — and include it in each struct literal.

- **Add the measurement probe** `tests/learning_signal_baseline.rs`: a single-agent
  default-config run capturing the `raw_gradient` distribution (mean/std/min/max over
  100 ticks) into the test's doc comment via
  `read_agent_telemetry_blocking(0).raw_gradient`. It embeds the standard GPU
  self-skip guard and asserts nothing — a measurement, not a gate.

Properties that make this safe:
- `s_homeo[6u]` is written by thread 0 (inside the existing `if (tid == 0u)` block at
  `brain_passes.wgsl:845-850`) and read by thread 0 only (the telemetry write-out at
  `brain_passes.wgsl:1470`), inheriting the same barrier discipline the existing
  `s_homeo[0u..5u]` slots already rely on across passes; no new cross-thread read is
  introduced, so barrier uniformity is untouched.
- The new physics slot is appended at the end of the stride (index 44); all existing
  `P_*` indices are unchanged, so no buffer-layout shift affects prior readers. The
  slot is live state only (never serialized), so save/layout compatibility is
  unaffected.
- The probe asserts nothing, so it cannot itself regress; it only records the
  pre-removal reference the guard test (0005) embeds.

## 0002 — Approach-Shaping-Removal

Today `coop_habituate_homeo` in `brain_passes.wgsl` (lines 791–805) computes a
potential-based shaping reward for food seeking: it reads the nearest-food
distance (`physics_state[phys_base_homeo + P_NEAREST_FOOD_DISTANCE]`), normalizes
it by `SHAPING_RADIUS` (`common.wgsl:427`, value 30.0), applies the gain
(`APPROACH_SHAPING_GAIN`, `common.wgsl:500`, value 0.05) to form a potential
Φ(s) = −gain·d_norm, stores the previous state in
`physics_state[…P_PREV_POTENTIAL]` (`buffers.rs:193`), and computes the shaping
increment `shaping = TD_DISCOUNT * potential − prev_potential` (line 804). That
increment is then folded into `raw_gradient` at line 825. The machinery
(`P_PREV_POTENTIAL` storage, food-distance reads) is wired into production despite
the approach-shaping unlock having been falsified by remeasure
(mirrored-steering alignment held at chance across 1200 shaped episodes), and it
contradicts the README's stated homeostasis-only design.

Edits:

- **Remove the approach-shaping block** (`brain_passes.wgsl:791–805`): delete the
  `let d_norm = …`, `let potential = …`, `let prev_potential = …`,
  `let shaping = …`, and `physics_state[… P_PREV_POTENTIAL] = …` statements, and
  replace them with a single zero constant so the variable still exists for the
  `raw_gradient` assembly below.

```wgsl
// Shaping term removed: pure homeostatic learning only.
let shaping: f32 = 0.0;
```

- **Rewrite the approach-shaping test.** `shaped_reward_rewards_approach`
  (`integration.rs:2267`) exists to verify this term; rewrite it (and the
  distance-closing credit assertions near `:2369`/`:2703`) to assert the
  post-removal homeostatic-only behavior — approaching and receding agents now
  receive an equal `raw_gradient`. (Const cleanup — deleting `APPROACH_SHAPING_GAIN`,
  renaming `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS` — happens in workstream 0004, not
  here, because `SHAPING_RADIUS` is still live in the food-detect scan.)

Properties that make this safe:
- `shaping` is consumed only in the `raw_gradient` assembly (line 825); pinning it
  to a zero constant is behavior-preserving for all downstream credit, homeostatic
  EMAs, and memory valence that feed on `raw_gradient`.
- The `P_NEAREST_FOOD_DISTANCE` read (line 799) is removed, but the slot itself is
  retained — it is still written by the food-detect pass and consumed by agent
  telemetry. The write to `P_PREV_POTENTIAL` (line 805) is removed; the slot is no
  longer touched, respawn does not wipe it, and no code reads it, so
  initialization order is unaffected.
- No shader bindings, uniforms, override constants, or memory layout change — only
  the compute kernel's logic.

## 0003 — Avoidance-Shaping-Removal

Today `coop_habituate_homeo` (lines 807–822) computes a danger-avoidance shaping
reward behind the `CFG_DANGER_PERCEPT_ENABLED` flag: it reads the nearest-danger
distance from `physics_state`, normalizes it by `DANGER_SENSE_RADIUS`, computes
Φ_d(s) = −(1 − danger_d_norm), stores the previous state in
`physics_state[…P_PREV_DANGER_POTENTIAL]` (`buffers.rs:217`), and yields the
shaping increment `danger_shaping = TD_DISCOUNT * danger_potential −
prev_danger_potential` (line 820). That increment is folded into `raw_gradient` at
line 826. Despite the flag (default false), the code is an engineered reward
kernel — mechanically indistinguishable from the approach shaping — and
contradicts the homeostasis-only design.

Edits:

- **Remove the avoidance-shaping block** (`brain_passes.wgsl:807–822`): delete the
  `var danger_shaping: f32 = 0.0;` declaration and the entire
  `if (bc_f32(CFG_DANGER_PERCEPT_ENABLED) != 0.0) { … }` conditional, and replace
  the block with a single unconditional zero constant.

```wgsl
// Avoidance shaping term removed: danger percept remains available to the
// encoder/predictor for natural discovery.
let danger_shaping: f32 = 0.0;
```

- **Rewrite the avoidance-shaping test.** `avoidance_potential_sign`
  (`integration.rs:6902`, ≈`:6894-7058`) exists to verify the danger-potential
  telescoping; rewrite the "on" cases to assert the shaping no longer fires (the
  slot stays zero), keeping the flag-off `== 0.0` case. The `P_PREV_DANGER_POTENTIAL`
  const/slot are **retained** as reserved — they are referenced by this test and
  the `buffers.rs` parity test; the reserved-slot doc-comment update happens in
  workstream 0004. No stride shift.

Properties that make this safe:
- `danger_shaping` is consumed only in the `raw_gradient` assembly (line 826);
  pinning it to a zero constant is behavior-preserving.
- The danger-percept sensory channel (distance/bearing features in
  `coop_feature_extract`) is untouched — it is still written and fed to the
  encoder, so agents can discover the danger signal through learning if the
  encoder/predictor find it useful. The shaping term merely short-circuited that
  discovery; removing it restores pure homeostasis-based learning.
- The `P_NEAREST_DANGER_DISTANCE` slot is retained (still written by the
  physics/food-detect passes, consumed by telemetry), so no downstream
  buffer-layout shifts.

## 0004 — Dead-Code-Cleanup

Removing the two shaping terms (workstreams 0002, 0003) leaves the supporting
machinery in **three different states** — verified by grep, each handled
correctly rather than blanket-deleted.

Edits:

- **Delete `APPROACH_SHAPING_GAIN`** (`common.wgsl:500`) and its doc-comment — it
  is genuinely orphaned after the approach block is gone (no source or test
  reference remains; only historical `docs/plans/0004-*` mentions, which are fine).
- **Rename `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS`** (`common.wgsl:427`), updating
  every reference in `kernel_tick.wgsl` (≈361, 373, 387, 427–444, 588) and the
  doc-comments in `buffers.rs` (≈183–184, 231). This const is NOT orphaned: it
  bounds the nearest-food scan in `agent_food_detect` that writes
  `P_NEAREST_FOOD_DISTANCE`, independent of shaping; the rename just makes the name
  match its real role. The comment must be purely technical (no plan/process
  language — `contributing_guard.rs` enforces this).
- **Reserve `P_PREV_POTENTIAL` (slot 33) and `P_PREV_DANGER_POTENTIAL` (slot 41)**:
  keep both consts and slots, updating their doc-comments to note they are no
  longer written. Do NOT delete them — the `buffers.rs` PHYS_STRIDE layout-parity
  test (≈1231, 1249–1250, 1488, 1496) and `integration.rs` (≈6903–7058) reference
  them by name, and deleting would force a stride shift. Leave the respawn zeroing
  (`phase_death.wgsl`, `kernel_tick.wgsl`) in place.

Properties that make this safe:
- The only deletion is `APPROACH_SHAPING_GAIN`, which grep confirms is referenced
  nowhere in source after the approach block is removed; the compiler surfaces any
  missed reference immediately ("cannot find … in this scope").
- The rename is mechanical and total (definition + all call sites), so the
  food-detect scan keeps working unchanged; only the symbol name changes.
- No buffer-layout shift: the prev-potential slots stay at indices 33/41, so all
  downstream `P_*` offsets and the PHYS_STRIDE parity test are unaffected; the
  slots simply go unused (reserved).

## 0005 — Parity-And-Tests

Today the brain's learning signal (`raw_gradient`) is computed in the fused-kernel
path — `kernel_tick.wgsl` driving `coop_habituate_homeo` in `brain_passes.wgsl`
(lines 823–826) — and may also be assembled in a split-kernel path
(`phase_physics.wgsl` or a sibling phase shader). After shaping removal both paths
must compute `raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta *
INTEGRITY_WEIGHT` with no additional terms. A parity check keeps them in sync, and
a falsifiable unit test guards against any future reintroduction of shaping. The
guard reads `raw_gradient` through the `AgentTelemetry::raw_gradient` field wired in
workstream 0001 (the `P_RAW_GRADIENT_OUT` physics slot); no new readback plumbing is
introduced here.

Edits:

- **Audit every gradient path** by grepping the kernel shaders for `raw_gradient`
  and the homeostatic-term expressions, then confirm each computes the same
  expression post-removal. If a split path exists, apply the same shaping-removal
  edits there so the assemblies are byte-identical.

- **Add a regression-guard test** in
  `crates/xagent-brain/tests/raw_gradient_homeostatic_only.rs` that instantiates a
  single-agent kernel at default config, ticks it with known deltas, and asserts
  `read_agent_telemetry_blocking(0).raw_gradient` is exactly the homeostatic weighted
  sum — zero deltas yield zero, and non-zero deltas yield the weighted sum with no
  shaping component. The test embeds the standard GPU self-skip guard and the
  pre-removal baseline (from workstream 0001's probe) in its doc comment.

```rust
/// Verifies raw_gradient contains only homeostatic deltas (energy + integrity),
/// with both approach-PBRS and avoidance-PBRS terms removed. Regression guard:
/// reintroducing any shaping term fails this assertion.
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

Properties that make this safe:
- The guard test is a logical consequence of the design: correct code passes;
  reintroduced shaping fails loudly.
- The `raw_gradient` readback path is established in workstream 0001 and reused
  verbatim here, so this test adds no kernel/buffer surface of its own.
- Beyond the shaping removal in workstreams 0002–0003 (and any split-path mirror),
  the tests are purely additive; no production behavior changes.

## 0006 — Documentation-Update

Today the README states the homeostasis-only design as fact — §1 ("There are no
reward signals, no utility functions, no goal hierarchies. The only evaluative
signal in the entire system is homeostatic stability", `README.md:10`) and §10
("Why Homeostasis-Only Evaluation?", around `README.md:515`) — while
`brain_passes.wgsl:823–826` injected two engineered reward kernels (`shaping`,
`danger_shaping`) into `raw_gradient`. After the removals the prose is true by
construction; the documentation should confirm the alignment without overstating
the change.

Edits:

- **README.md**: verify §1 and §10 already describe the homeostasis-only vision
  correctly (they do); no rewrite is needed. Reconcile any lingering mention that
  would imply approach- or avoidance-shaping is active.
- **`crates/xagent-brain/README.md`**: if it describes the learning-signal
  composition or the gradient assembly, update it to state that `raw_gradient` is
  purely homeostatic
  (`energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT`), framing
  the prior approach/avoidance shaping as historical context that this plan
  removed.
- **Optional shader doc-comment** at the head of `coop_habituate_homeo`:
  "computes pure homeostatic gradients (energy + integrity deltas, no reward
  shaping)".

Properties that make this safe:
- Documentation edits are non-functional; they clarify the existing design without
  altering execution.
- The wording moves from aspirational to descriptive only after the code matches,
  so the docs cannot misrepresent live behavior.

## Test strategy

The falsifiable gate is `test_raw_gradient_is_homeostatic_only` in
`crates/xagent-brain/tests/raw_gradient_homeostatic_only.rs`: it constructs a
brain state with controlled deltas, ticks the kernel, and asserts the resulting
`raw_gradient` (read via `AgentTelemetry::raw_gradient`, the `P_RAW_GRADIENT_OUT`
slot wired in workstream 0001) equals the
homeostatic weighted sum — zero deltas → 0.0, energy-only → `0.1 * ENERGY_WEIGHT`,
integrity-only → `0.05 * INTEGRITY_WEIGHT` — with no shaping component. The test
is a permanent regression guard: any future edit that reintroduces a shaping term
fails it. It embeds the `GpuKernel::is_available()` self-skip guard, so it
self-skips without an adapter and runs under Mesa lavapipe in CI.

A measurement probe, `learning_signal_baseline.rs`, captures the pre-removal
`raw_gradient` distribution (mean/std/min/max over 100 ticks at default config)
into its doc comment, supplying the "before" reference embedded in the guard test;
the probe asserts nothing.

CI gate (every task): `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p xagent-sandbox`.

## Interaction with prior work

- **Resolves the consensus #1 philosophical contradiction.** The 2026-06-18
  reviews (Grok 4.3, Gemini 3.1 Pro) flagged the README-versus-code mismatch as
  the top pre-merge finding once Plans 0010 and 0011 had cleared the
  planning-reference leakage; this plan removes the engineered reward kernels that
  caused it, restoring vision-code alignment.
- **Honors Plan 0004's own falsification.** Plan 0004's remeasure showed
  mirrored-steering alignment held at chance across 1200 shaped episodes; the
  approach-shaping mechanism nonetheless persisted in production. This plan retires
  it on that recorded evidence rather than re-litigating the unlock.
- **Preserves Plan 0009's danger percept as a sensory channel.** Only the
  avoidance-shaping reward term is removed; the danger-percept distance/bearing
  features remain written and available to the encoder/predictor, so the sensory
  pathway Plan 0009 added survives — the change restores learning-from-experience,
  it does not delete the sense.
