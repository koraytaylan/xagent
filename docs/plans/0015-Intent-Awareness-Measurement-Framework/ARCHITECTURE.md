# Architecture — Plan 0015 (deltas)

> Edits in `crates/xagent-brain/src/buffers.rs`,
> `crates/xagent-brain/src/gpu_kernel.rs`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_death.wgsl`,
> `crates/xagent-brain/tests/intent_baseline_measurement.rs`,
> `crates/xagent-sandbox/src/agent/mod.rs` (the `Agent` cache struct, ≈line 134),
> `crates/xagent-sandbox/src/gpu_orchestration.rs`,
> `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-brain/README.md`, and `README.md`.
> Line numbers are hints; locate by symbol (grep for `P_AVOIDANCE_SENSE_RANGE_TICKS`,
> `agent_avoidance_accumulate`, `compute_avoidance_intent_fraction`,
> `agent_death_respawn`, `PHYS_STRIDE`, `decision_buffer`).

## Guiding principle — mirror the existing avoidance machinery

The project already ships a complete **avoidance-intent** measurement path. Plan 0015 does
**not** invent a new mechanism; it adds the symmetric **approach-intent** path for food by
copying the avoidance structure at every layer, and then measures the baseline. The
existing avoidance path, layer by layer, is the template:

| Layer | Avoidance (exists) | Approach (this plan adds) |
|---|---|---|
| Physics slots | `P_AVOIDANCE_SENSE_RANGE_TICKS` (42), `P_AVOIDANCE_TURNS_OPPOSING` (43) | `P_APPROACH_SENSE_RANGE_TICKS` (45), `P_APPROACH_TURNS_TOWARD` (46) |
| Kernel accumulation | `agent_avoidance_accumulate` (`kernel_tick.wgsl:257-274`) | `agent_approach_accumulate` (new, same shape) |
| Call site | thread-0 block after `agent_danger_detect` (`kernel_tick.wgsl:~812`) | same block, same `motor_turn` |
| Respawn semantics | generation-cumulative: saved/restored in both death paths | identical save/restore |
| CPU cache | state buffer → `Agent` (3 sites: `gpu_orchestration.rs:256-257`, `headless.rs:235-236` and `:823-824`) | same 3 copy sites |
| Fitness | `AgentFitness` from `Agent` in `evaluate()` (`governor.rs:676-677`) | same |
| Population metric | `compute_avoidance_intent_fraction` (`governor.rs:203-207`) | `compute_approach_intent_fraction` |
| Persistence | `avoidance_intent_fraction` column in `behavior_metric` | `approach_intent_fraction` column |

The signal is **count-based** (not a cosine or angle): each tick the target is in sense
range, increment a `sense_range_ticks` counter; if the motor turn rotates the correct way,
increment a `turns_*` counter. The intent fraction is `turns / sense_range_ticks`.

## 0001 — Approach-Intent Telemetry Pipeline

`agent_avoidance_accumulate` (`kernel_tick.wgsl:257-274`) is the exact template:

```wgsl
fn agent_avoidance_accumulate(agent_id: u32, motor_turn: f32) {
    let b = agent_id * PHYS_STRIDE;
    let danger_distance = physics_state[b + P_NEAREST_DANGER_DISTANCE];
    let danger_bearing  = physics_state[b + P_NEAREST_DANGER_BEARING];
    if danger_distance < DANGER_SENSE_RADIUS {
        physics_state[b + P_AVOIDANCE_SENSE_RANGE_TICKS] += 1.0;
        // danger_bearing<0 = danger to the right; a turn AWAY rotates against the
        // bearing, so motor_turn * danger_bearing > 0 is an avoidance turn.
        let turn_away = (motor_turn * danger_bearing) > 0.0;
        if turn_away { physics_state[b + P_AVOIDANCE_TURNS_OPPOSING] += 1.0; }
    }
}
```

Edits:

- **Append two physics slots** in `buffers.rs` after `P_RAW_GRADIENT_OUT` (44):
  `P_APPROACH_SENSE_RANGE_TICKS = 45`, `P_APPROACH_TURNS_TOWARD = 46`, bump `PHYS_STRIDE`
  to 47; mirror both in `common.wgsl` (and its `PHYS_STRIDE = 47u`); extend the
  `phys_stride_covers_all_fields` parity test (max offset 46) and the WGSL-mirror
  assertions.

- **Add `agent_approach_accumulate`** in `kernel_tick.wgsl`, the food sign-mirror of the
  avoidance function. Turning *toward* the food bearing is the opposite sign of turning
  away from danger, so the predicate flips from `> 0.0` to `< 0.0`:

```wgsl
fn agent_approach_accumulate(agent_id: u32, motor_turn: f32) {
    let b = agent_id * PHYS_STRIDE;
    let food_distance = physics_state[b + P_NEAREST_FOOD_DISTANCE];
    let food_bearing  = physics_state[b + P_NEAREST_FOOD_BEARING];
    if food_distance < FOOD_SENSE_RADIUS {
        physics_state[b + P_APPROACH_SENSE_RANGE_TICKS] += 1.0;
        // food_bearing<0 = food to the right; a turn TOWARD rotates with the bearing,
        // so motor_turn * food_bearing < 0 is an approach turn (opposite sign to avoidance).
        let turn_toward = (motor_turn * food_bearing) < 0.0;
        if turn_toward { physics_state[b + P_APPROACH_TURNS_TOWARD] += 1.0; }
    }
}
```

- **Call it from the existing thread-0 block** in the cycle loop (`kernel_tick.wgsl:~812`),
  immediately after `agent_avoidance_accumulate`, passing the same
  `motor_turn = decision_buffer[decision_base + DECISION_MOTOR + 1u]`. No new guard or
  barrier — reuse the avoidance one. The block already runs after `agent_food_detect` (so
  `P_NEAREST_FOOD_*` is current this cycle) and after `agent_danger_detect`.

- **Generation-cumulative across respawn in BOTH death paths.** The avoidance counters are
  saved before the full-stride zero loop and restored after, in both
  `agent_death_respawn` (`kernel_tick.wgsl:553-554` save) — the **fused** production path —
  and `phase_death.wgsl` (`:54-55` save, `:89-90` restore) — the **split/remainder** path.
  Add the identical save/restore for `P_APPROACH_*` in **both**. Editing only one diverges
  the two death paths on the new slots (both zero the full stride), violating the
  death-path-parity invariant that a regression test guards — so both must change together.

- **Surface all four counters on `AgentTelemetry`** (`gpu_kernel.rs:330`) so the baseline
  probe can read them: `approach_sense_range_ticks`, `approach_turns_toward`,
  `avoidance_sense_range_ticks`, `avoidance_turns_opposing`, read from their physics slots
  in both `read_agent_telemetry_blocking` (literal ≈2984) and `try_collect_telemetry`
  (literal ≈3197), using the `phys[CONST]` idiom `gradient` already uses.

Properties that make this safe:
- The accumulation is thread-0-only inside the existing avoidance block, inheriting its
  barrier discipline; no new cross-thread read or barrier is introduced.
- The counters are append-only physics slots at the end of the stride (45-46); all existing
  `P_*` offsets are unchanged, so no prior reader shifts.
- The save/restore mirrors the avoidance counters exactly, so the fused and split death
  paths stay byte-identical on the new slots (death-path parity preserved).

## 0002 — Homeostasis-Only Intent Metrics

The avoidance fields are already populated end-to-end (NOT "always zero"): the state buffer
is copied onto the `Agent` cache (`gpu_orchestration.rs:256-257`, `headless.rs:235-236`,
`:823-824`), `evaluate()` builds `AgentFitness` from the `Agent`
(`governor.rs:676-677`), and `compute_avoidance_intent_fraction` (`governor.rs:203-207`)
aggregates them:

```rust
pub(crate) fn compute_avoidance_intent_fraction(fitness: &[AgentFitness]) -> f32 {
    let total_sense_range_ticks: f32 = fitness.iter().map(|f| f.avoidance_sense_range_ticks).sum();
    let total_turns_opposing: f32 = fitness.iter().map(|f| f.avoidance_turns_opposing).sum();
    total_turns_opposing / total_sense_range_ticks.max(EPSILON)
}
```

Edits (each mirrors an existing avoidance line; the avoidance plumbing is untouched):

- **`Agent` struct** (`crate::agent`): add `approach_sense_range_ticks: f32` /
  `approach_turns_toward: f32` beside the avoidance pair; default to `0.0`.
- **State→cache copy**: at `gpu_orchestration.rs:256-257`, `headless.rs:235-236`, and
  `:823-824`, copy `P_APPROACH_SENSE_RANGE_TICKS` / `P_APPROACH_TURNS_TOWARD` onto the
  `Agent` next to the avoidance copy.
- **`AgentFitness`** (`governor.rs:42`): add the two approach fields; populate them in
  `evaluate()` from the `Agent` (`governor.rs:676-677`); update every other
  `AgentFitness { .. }` literal in the file (test fixtures) to init them to `0.0`.
- **`compute_approach_intent_fraction`** mirroring the avoidance aggregator (sum of
  `approach_turns_toward` over sum of `approach_sense_range_ticks`, `.max(EPSILON)`).
- **Persist** `approach_intent_fraction` in `persist_behavior_metrics` (`governor.rs:861+`)
  alongside `avoidance_intent_fraction`, with an `approach_intent_fraction REAL` column in
  the table init (≈`governor.rs:1864`) and an idempotent `ALTER TABLE … ADD COLUMN`
  migration (≈`governor.rs:1870`).
- **Discrimination test**: mirror `avoidance_intent_fraction_discriminates_turn_away`
  (`governor.rs:4763`) — a turn-toward population must outscore a straight-through one.

Properties that make this safe:
- Every field is a read-only summary of accumulated GPU telemetry; nothing flows back to
  the kernel, so the homeostasis-only learning constraint (Plan 0012) is preserved.
- The computation is deterministic from telemetry and runs only during readback
  aggregation — zero runtime cost during learning.
- The `behavior_metric` schema change is append-only and idempotent; existing rows and
  readers are unaffected.

## 0003 — Baseline Measurement & Documentation

There is no reference distribution for the intent fractions under pure homeostatic learning
(post-Plan-0012, README §10). This workstream captures it and documents the shipped
telemetry.

Edits:

- **Measurement probe** `crates/xagent-brain/tests/intent_baseline_measurement.rs`,
  mirroring the kernel-driving idiom of `learning_signal_baseline.rs` (the snapshot/zeros
  trap is reading telemetry off a kernel that was never dispatched). Concretely:
  `BrainConfig::default()` / `WorldConfig::default()` (from `xagent_shared`),
  `GpuKernel::new(16, food_count, …)`, `reset_agents_seeded(&brain_config, 12345)`,
  `upload_world(…)`, `upload_agents(…)`, then a loop of
  `dispatch_batch(start_tick, kernel_batch_size())` (each batch ≈100 ticks; sample per
  **batch**), and finally read each of the 16 agents' telemetry and compute its approach /
  avoidance fraction.

- **Record the distribution incl. percentiles.** Compute mean/std/min/max **and
  p25/p50/p75** of each axis across the 16 agents and write them into the test doc comment
  (run-then-paste). The percentiles are the reference the deferred follow-up validation
  harness will consume; this probe asserts nothing.

```rust
/// Baseline intent fractions under pure homeostatic learning (post-Plan-0012, no PBRS
/// shaping). Records mean/std/min/max + p25/p50/p75 per axis as the reference distribution
/// a follow-up validation-harness plan will use to set deliberate-vs-incidental thresholds.
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- **Document the shipped framework.** In `crates/xagent-brain/README.md`, add an
  "Intent & Awareness Telemetry" subsection (the two count-based intent fractions — toward
  food / away from danger — how they are computed, generation-cumulative, exposed on
  `AgentTelemetry`, aggregated into `behavior_metric`; how to read the baseline
  percentiles). In `README.md`, note in §3 / §10 that the intent fractions are an
  observational, measurement-only lens on emergence (zero impact on learning), and add a
  short "Measuring Intentional Behavior" section summarizing the telemetry counters and the
  two `behavior_metric` fraction columns, pointing forward to the deferred validation harness.

## Deferred to a follow-up plan — Intent Verification & Deliberate-Behavior Harness

The seeded A/B validation harness — which would classify agents deliberate-vs-incidental
against the baseline percentiles this plan records, build a seeded null random-walk
control, and assert intent correlates with path-coherence / encounter statistics — is
intentionally **not** authored here. Its thresholds can only be set defensibly from the
*real* baseline numbers workstream 0003 measures, and (per the project's own
learning-at-chance history) the baseline may show intent at chance, in which case the next
plan should target the credit path rather than a classifier. It is therefore deferred to a
follow-up plan authored against this plan's recorded baseline. The infrastructure that
follow-up consumes — the approach/avoidance counters on `AgentTelemetry`, the population
fractions, and the baseline distribution — is exactly what this plan delivers.

## Test strategy

The functional gates for this plan are the existing suite plus: the `PHYS_STRIDE`
layout-parity test (`buffers.rs`, updated for the two new slots), the new
`approach_intent_fraction_discriminates_turn_toward` test (mirroring the avoidance
discrimination test at `governor.rs:4763`), and the death-path-parity regression test
(which passes only if both death functions save/restore the new counters identically). The
baseline probe `intent_baseline_measurement.rs` is measurement-only (asserts nothing) and
embeds the `GpuKernel::is_available()` self-skip guard, so it self-skips without an adapter
and runs under Mesa lavapipe in CI. The falsifiable deliberate-vs-incidental gate lives in
the deferred follow-up harness, authored once the baseline numbers exist.

CI gate (every task): `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p xagent-sandbox`.

## Interaction with prior work

- **Completes Plan 0009's intent measurement symmetrically.** Plan 0009 added the danger
  percept and the avoidance-intent counters; this plan adds the food-side approach-intent
  counters using the identical mechanism, so both deliberate behaviors (approach food,
  avoid danger) are measured the same way.
- **Builds on Plan 0007's behavioral-telemetry infrastructure.** The `behavior_metric`
  table and its existing `avoidance_intent_fraction` column are reused; this plan adds the
  `approach_intent_fraction` column beside it. The deferred follow-up harness will correlate
  intent against the Plan 0007 straightness telemetry.
- **Respects Plan 0012's homeostasis-only constraint absolutely.** All counters and
  fractions are computed post-hoc from telemetry with zero impact on the kernel's learning
  or fitness. The 7-stage pipeline and `raw_gradient` assembly are untouched; if a future
  experiment shows an intent-based learning signal is beneficial, that is a new plan with
  new evidence, not a change to this one.
