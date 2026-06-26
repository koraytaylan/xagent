# XAgent Plan 0015 — Intent & Awareness Measurement Framework

Add the **approach-intent** counterpart to the project's existing **avoidance-intent**
machinery: the GPU already counts, per agent, how many in-sense-range ticks the agent
turned *away* from danger (`agent_avoidance_accumulate`, the `P_AVOIDANCE_*` physics
slots, `compute_avoidance_intent_fraction`). This plan mirrors that exactly for food —
counting how many in-range ticks the agent turned *toward* food — exposes both signals
on `AgentTelemetry`, routes the approach counters into `AgentFitness` and a population
`approach_intent_fraction`, and captures the baseline intent distribution under
homeostasis-only learning. This is **measurement infrastructure only** — observational
counters with zero impact on learning or fitness selection. The seeded A/B validation
harness that classifies agents deliberate-vs-incidental is **deferred to a follow-up
plan**, authored against the real baseline this plan measures.

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
- **Mirror, don't invent.** Every edit in workstreams 0001–0002 has an existing
  avoidance-side analogue (`P_AVOIDANCE_SENSE_RANGE_TICKS` / `P_AVOIDANCE_TURNS_OPPOSING`,
  `agent_avoidance_accumulate`, `compute_avoidance_intent_fraction`). Find that analogue
  first and copy its structure; do not introduce a different data-flow.

---

## 0001 — Approach-Intent Telemetry Pipeline

### wire-approach-intent-counters — Wire Approach-Intent Counters (mirror of avoidance)

The GPU already maintains avoidance-intent counters per agent: `agent_avoidance_accumulate`
(`crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl:257-274`) increments
`P_AVOIDANCE_SENSE_RANGE_TICKS` (`buffers.rs:219`, offset 42) every tick danger is in
range and `P_AVOIDANCE_TURNS_OPPOSING` (`buffers.rs:222`, offset 43) every tick the motor
turn rotates away from the danger bearing. It is called from the cycle's thread-0 block
(`kernel_tick.wgsl:~812`, just after `agent_danger_detect`), reading the motor turn from
`decision_buffer[decision_base + DECISION_MOTOR + 1u]`. The counters are
generation-cumulative: both death/respawn functions save them before the full-stride zero
loop and restore them after (`kernel_tick.wgsl:553-554`/restore, and
`phase_death.wgsl:54-55`/`89-90`). This task adds the symmetric **approach** counters for
food and exposes both intent signals on `AgentTelemetry`.

**Steps:**
1. In `crates/xagent-brain/src/buffers.rs`, append two new physics-slot constants
   immediately after `P_RAW_GRADIENT_OUT` (offset 44, currently the highest):
   `pub const P_APPROACH_SENSE_RANGE_TICKS: usize = 45;` and
   `pub const P_APPROACH_TURNS_TOWARD: usize = 46;`. Doc-comment them as the food-side
   mirror of `P_AVOIDANCE_SENSE_RANGE_TICKS` / `P_AVOIDANCE_TURNS_OPPOSING`:
   'Generation-cumulative count of in-sense-range ticks / ticks the motor turn rotated
   toward the nearest food bearing. Population approach-intent = turns_toward /
   sense_range_ticks. Live state only, never serialized.' Bump `PHYS_STRIDE` to `47`.
2. Mirror both consts in `crates/xagent-brain/src/shaders/kernel/common.wgsl` next to the
   existing `P_AVOIDANCE_*` WGSL consts: `const P_APPROACH_SENSE_RANGE_TICKS: u32 = 45u;`,
   `const P_APPROACH_TURNS_TOWARD: u32 = 46u;`, and bump the WGSL `PHYS_STRIDE` const to `47u`.
3. In `crates/xagent-brain/src/buffers.rs`, update the `phys_stride_covers_all_fields`
   parity test (≈line 1504): add the two new constants to the enumerated `P_*` array so
   the max offset is 46 and stride is 47. If the WGSL-mirror test (≈line 1251, which
   asserts `wgsl["P_AVOIDANCE_*"] == P_AVOIDANCE_* as u32`) enumerates the avoidance
   consts, add the matching `P_APPROACH_*` assertions there too.
4. In `kernel_tick.wgsl`, add `fn agent_approach_accumulate(agent_id: u32, motor_turn: f32)`
   directly mirroring `agent_avoidance_accumulate` but for food. Read
   `P_NEAREST_FOOD_DISTANCE` (`buffers.rs:187`, offset 32) and `P_NEAREST_FOOD_BEARING`
   (`buffers.rs:196`, offset 34); gate on `food_distance < FOOD_SENSE_RADIUS` (the same
   sentinel the food-detect scan uses — grep `FOOD_SENSE_RADIUS` in the kernel WGSL).
   Inside the gate: `physics_state[b + P_APPROACH_SENSE_RANGE_TICKS] += 1.0;` and a
   turn-*toward* test that is the sign-mirror of avoidance's turn-away. Avoidance uses
   `turn_away = (motor_turn * danger_bearing) > 0.0` (rotating against the bearing);
   turning *toward* the food bearing is the opposite sign, so
   `let turn_toward = (motor_turn * food_bearing) < 0.0;` and
   `if (turn_toward) { physics_state[b + P_APPROACH_TURNS_TOWARD] += 1.0; }`. Copy the
   sign-convention comment from `agent_avoidance_accumulate` and adapt it for food.
5. In `kernel_tick.wgsl`, call the new function from the **same thread-0 block** that calls
   `agent_avoidance_accumulate` (the `if (tid == 0u) { ... agent_avoidance_accumulate(...) }`
   block at ≈line 812, after `agent_danger_detect`). Pass the same
   `motor_turn = decision_buffer[decision_base + DECISION_MOTOR + 1u]` value:
   `agent_approach_accumulate(agent_id, motor_turn);`. Do **not** add a new thread-0 guard
   or barrier — reuse the existing one, exactly as avoidance does.
6. Make the new counters generation-cumulative in **both** death/respawn paths, mirroring
   the avoidance save/restore (do not edit only one — the fused and split death paths must
   stay byte-identical, a tested invariant):
   - In `kernel_tick.wgsl` `agent_death_respawn` (the fused path; the avoidance save is at
     `kernel_tick.wgsl:553-554`), add `let saved_approach_sense_range = physics_state[base + P_APPROACH_SENSE_RANGE_TICKS];`
     and `let saved_approach_turns_toward = physics_state[base + P_APPROACH_TURNS_TOWARD];`
     before the `for i < PHYS_STRIDE` zero loop, and restore both after it (next to the
     avoidance restore).
   - In `phase_death.wgsl` (the split/remainder path; avoidance save at `:54-55`, restore
     at `:89-90`), add the identical save-before-zero / restore-after-zero for the two new
     approach slots.
7. In `crates/xagent-brain/src/gpu_kernel.rs`, expose all four intent counters on
   `AgentTelemetry` (struct at ≈line 330, beside `pub gradient: f32` ≈line 344) so the
   baseline probe (workstream 0003) can read them: add
   `pub approach_sense_range_ticks: f32,`, `pub approach_turns_toward: f32,`,
   `pub avoidance_sense_range_ticks: f32,`, `pub avoidance_turns_opposing: f32,`. In both
   readback paths — `read_agent_telemetry_blocking` (struct literal ≈line 2984) and
   `try_collect_telemetry` (struct literal ≈line 3197) — read each from its physics slot
   (`phys[P_APPROACH_SENSE_RANGE_TICKS]`, etc., the same `phys[CONST]` idiom `gradient`
   uses) and include all four in the returned struct literal.
8. Run `cargo fmt --all` and `cargo test -p xagent-sandbox`; the PHYS_STRIDE parity test
   fails loudly if the offsets/stride are wrong — fix any mismatch.

- **Depends on:** —
- **Done when:** `P_APPROACH_SENSE_RANGE_TICKS` / `P_APPROACH_TURNS_TOWARD` exist in
  `buffers.rs` and `common.wgsl` at stride 47 with the parity test passing;
  `agent_approach_accumulate` is called from the existing thread-0 avoidance block and
  increments the counters using the `(motor_turn * food_bearing) < 0.0` turn-toward test;
  both death/respawn paths save+restore the new counters (fused and split identical); all
  four intent counters are exposed on `AgentTelemetry` and read in both readback paths;
  `cargo fmt/clippy/test green`.

---

## 0002 — Homeostasis-Only Intent Metrics

### populate-approach-intent-fraction — Populate Approach Intent Fields & Population Fraction

The avoidance intent path is already complete end-to-end: the `P_AVOIDANCE_*` slots flow
from the raw GPU state buffer onto the `Agent` cache
(`gpu_orchestration.rs:256-257`, `headless.rs:235-236` and `:823-824`), `evaluate()`
builds `AgentFitness` from the `Agent` (`governor.rs:676-677`,
`a.avoidance_sense_range_ticks` / `a.avoidance_turns_opposing`), and
`compute_avoidance_intent_fraction` (`governor.rs:203-207`) aggregates them into the
`behavior_metric` table. This task mirrors that path for the new **approach** counters —
it does *not* touch the avoidance plumbing, which already works.

**Steps:**
1. Add two approach fields to the `Agent` struct (in `crate::agent`; grep `pub struct Agent`
   in `crates/xagent-sandbox/src` — it is the struct holding `avoidance_sense_range_ticks`),
   mirroring the avoidance pair: `pub approach_sense_range_ticks: f32,` and
   `pub approach_turns_toward: f32,`. Ensure any `Default`/constructor initializes them to
   `0.0` like the avoidance fields.
2. Copy the new slots from the GPU state buffer onto the `Agent` cache at every site that
   already copies the avoidance slots — `gpu_orchestration.rs:256-257`, `headless.rs:235-236`,
   and `headless.rs:823-824` — adding, beside each avoidance copy:
   `a.approach_sense_range_ticks = state[base + P_APPROACH_SENSE_RANGE_TICKS];` and
   `a.approach_turns_toward = state[base + P_APPROACH_TURNS_TOWARD];` (import the two new
   consts from `xagent_brain`/`buffers` alongside the existing `P_AVOIDANCE_*` imports).
3. In `crates/xagent-sandbox/src/governor.rs`, add the two approach fields to `AgentFitness`
   (after `avoidance_turns_opposing`, ≈line 42): `pub approach_sense_range_ticks: f32,` and
   `pub approach_turns_toward: f32,`. Populate them in `evaluate()` directly from the
   `Agent` next to the avoidance population (`governor.rs:676-677`):
   `approach_sense_range_ticks: a.approach_sense_range_ticks,` and
   `approach_turns_toward: a.approach_turns_toward,`. Then update **every other**
   `AgentFitness { .. }` literal in the file to initialize the two new fields to `0.0` — do
   not trust a line list: grep `AgentFitness {` (there are ~15+ literals, mostly test
   fixtures) and fix each one, or the build fails. `cargo build` / `cargo test` (a Done-when
   gate) errors loudly on any missed literal, so the compiler is the checklist.
4. Add `compute_approach_intent_fraction(fitness: &[AgentFitness]) -> f32` directly
   mirroring `compute_avoidance_intent_fraction` (`governor.rs:203-207`):
   `let total_sense_range_ticks: f32 = fitness.iter().map(|f| f.approach_sense_range_ticks).sum();`
   `let total_turns_toward: f32 = fitness.iter().map(|f| f.approach_turns_toward).sum();`
   `total_turns_toward / total_sense_range_ticks.max(EPSILON)` (reuse the existing
   `const EPSILON: f32 = 1e-6;` at `governor.rs:98`; do not redeclare). Document it as the
   population fraction of in-range ticks the agents steered toward food.
5. In `persist_behavior_metrics` (`governor.rs:861+`), compute and persist
   `approach_intent_fraction` alongside the existing `avoidance_intent_fraction`: add it to
   the INSERT column list and bound parameters. Add an `approach_intent_fraction REAL`
   column to the `behavior_metric` table init (≈`governor.rs:1864`) and an idempotent
   `ALTER TABLE behavior_metric ADD COLUMN approach_intent_fraction REAL;` migration next
   to the 2026-06-18 `avoidance_intent_fraction` migration (≈`governor.rs:1870`) so old
   databases auto-upgrade. (Scope note: the avoidance counters are *also* persisted
   per-agent to the `agent_result` table; the approach fraction is intentionally routed to
   `behavior_metric` only — the population fraction is the measurement target, and per-agent
   `agent_result` persistence of approach is left to the deferred harness plan that would
   consume it. Do not add `agent_result` columns here.)
6. Mirror the existing avoidance discrimination test: copy
   `avoidance_intent_fraction_discriminates_turn_away` (`governor.rs:4763`) to an
   `approach_intent_fraction_discriminates_turn_toward` test that builds a turn-toward-food
   population and a straight-through population and asserts the toward population scores a
   strictly higher `approach_intent_fraction`. Run `cargo test -p xagent-sandbox`.

- **Depends on:** wire-approach-intent-counters
- **Done when:** `Agent` and `AgentFitness` carry `approach_sense_range_ticks` /
  `approach_turns_toward`, populated from the GPU state buffer through the same cache path
  the avoidance counters use; `compute_approach_intent_fraction` mirrors
  `compute_avoidance_intent_fraction` and returns a non-zero population statistic on a
  turn-toward population; the `behavior_metric` table gains an `approach_intent_fraction`
  column (with idempotent migration); a discrimination test proves toward-food populations
  outscore straight-through ones; `cargo fmt/clippy/test green`.

---

## 0003 — Baseline Measurement & Documentation

### measure-baseline-intent-distribution — Measure Baseline Intent Distribution at Default Config

Pure homeostatic learning (post-Plan-0012 removal of PBRS shaping) has an unknown intent
baseline: does steering correlate with sensed food/danger at all, or is it at chance? This
probe captures the across-agent distribution of the approach- and avoidance-intent
fractions under default learning params — in a world set up so both axes are actually
exercised (food in sense range; danger percept enabled with danger in range) — so a
follow-up plan can set deliberate-vs-incidental thresholds against real numbers (and so we
learn whether a signal even exists before building a classifier on top of it).

**Steps:**
1. Create `crates/xagent-brain/tests/intent_baseline_measurement.rs`. Embed the standard
   GPU self-skip guard first: `if !xagent_brain::GpuKernel::is_available() { eprintln!("Skipping: no GPU/fallback adapter available"); return; }`.
2. Mirror the kernel-driving idiom of `crates/xagent-brain/tests/learning_signal_baseline.rs`
   (`use xagent_shared::{BrainConfig, WorldConfig};`, `GpuKernel::new`, `reset_agents_seeded`,
   `upload_world`, `upload_agents`, `dispatch_batch`) — but **do not copy its world layout**:
   that test deliberately places "one food far away" at `(50,0,50)` with the agent at the
   origin (~70 units apart, well outside `FOOD_SENSE_RADIUS = 30.0`, `common.wgsl:430`), and
   uses no danger biome. Copied verbatim, neither intent counter ever increments and the
   baseline is all-zeros and meaningless. Instead, build a world that **exercises both
   signals**:
   - **Population & spawn:** `agent_count = 16`, each agent spawned on a small grid near the
     origin (e.g. a 4×4 lattice, ~12 units apart) so agents do not all overlap.
   - **Food in sense range:** one food item per agent placed within `FOOD_SENSE_RADIUS`
     (e.g. ≈10 units) of that agent's spawn, so `agent_food_detect` reports food in range
     and the approach counter is exercised from tick 0. Size `food_pos` / `food_consumed` /
     `food_timers` to `agent_count`.
   - **Danger in sense range + percept on:** `agent_food_detect` always runs, but
     `agent_danger_detect` only runs when the danger percept is enabled
     (`kernel_tick.wgsl:~810` gates on `WC_DANGER_PERCEPT_ENABLED`), so at the shipped
     default (`BrainConfig::danger_percept_enabled = false`, `config.rs:141`) the avoidance
     counter is *structurally* zero. To make avoidance measurable, **set
     `brain_config.danger_percept_enabled = true`** before constructing the kernel (it flows
     to `WC_DANGER_PERCEPT_ENABLED`, `buffers.rs:475`) and mark biome cells within
     `DANGER_SENSE_RADIUS` of the agents as the danger biome (grep `agent_danger_detect` for
     the biome id it treats as danger). Document in a comment that the baseline runs with
     `danger_percept_enabled = true` specifically so the avoidance axis is defined — it is
     off in the shipped default.
3. Advance the sim with `let batch_size = kernel.kernel_batch_size();` and a loop of
   `kernel.dispatch_batch(start_tick, batch_size);` (each batch ≈ 100 physics ticks; the
   brain runs once per batch — so sample per **batch**, not per physics tick). Run enough
   batches to let the counters accumulate (e.g. 8 batches), advancing `start_tick` by
   `batch_size` each iteration.
4. After the final batch, read every agent's telemetry —
   `for i in 0..agent_count { let t = kernel.read_agent_telemetry_blocking(i); }` — and
   compute each agent's intent fractions:
   `approach = t.approach_turns_toward / t.approach_sense_range_ticks.max(1.0)` and
   `avoidance = t.avoidance_turns_opposing / t.avoidance_sense_range_ticks.max(1.0)`
   (guard the zero-denominator case where an agent never sensed the target). **Validity
   guard:** sum each axis's `sense_range_ticks` across agents; if either sum is zero,
   `eprintln!` a clear warning that the world layout failed to exercise that axis (the
   baseline would be degenerate) so a silent all-zero run is visible rather than recorded as
   "no intent". This yields a 16-sample across-agent distribution per axis.
5. Compute, for each axis, mean / std / min / max **and the 25th / 50th / 75th percentiles**
   (sort the per-agent vector, index at `(0.25 * n) as usize` / `0.5 * n` / `0.75 * n`).
   The percentiles are the reference a follow-up validation-harness plan will turn into
   deliberate-vs-incidental thresholds.
6. Do **not** assert any threshold — this is measurement-only. Record the computed
   statistics in a doc comment on the test (run it, paste the real numbers back, matching
   the `crates/xagent-brain/tests/learning_signal_baseline.rs` record-then-paste idiom), and add a high-level comment:
   'Baseline intent fractions under pure homeostatic learning (no PBRS shaping,
   post-Plan-0012); the reference distribution a follow-up validation-harness plan will use
   to set deliberate-vs-incidental thresholds.'
7. Run the test (`cargo test -p xagent-brain` / `-p xagent-sandbox`) and confirm it
   compiles, runs (or self-skips without a GPU), and records the statistics.

- **Depends on:** wire-approach-intent-counters
- **Done when:** A measurement probe drives a 16-agent kernel via the
  `crates/xagent-brain/tests/learning_signal_baseline.rs` upload/dispatch idiom in a world that exercises both intent
  axes (food within `FOOD_SENSE_RADIUS` of each agent; `danger_percept_enabled = true` with
  danger biome in range), reads each agent's approach- and avoidance-intent fractions, and
  records mean/std/min/max + p25/p50/p75 per axis in its doc comment. A validity guard
  `eprintln!`s if either axis's summed `sense_range_ticks` is zero (degenerate world). The
  test asserts nothing (measurement-only), embeds the GPU self-skip guard, and runs green;
  `cargo fmt/clippy/test green`.

---

### document-intent-measurement-framework — Document the Intent & Awareness Measurement Framework

Document the measurement infrastructure this plan ships — the approach/avoidance intent
counters, their population fractions, and the baseline distribution — so a researcher can
read and interpret them. Documentation only; it points forward to the deferred validation
harness rather than describing it as shipped.

**Steps:**
1. In `crates/xagent-brain/README.md`, near where the brain's homeostatic learning signal
   is described, add an '## Intent & Awareness Telemetry' subsection explaining: (1) the two
   intent signals — approach-intent = fraction of in-sense-range ticks the agent's motor
   turn rotated *toward* the nearest food bearing; avoidance-intent = the existing fraction
   of in-range ticks it turned *away* from danger; (2) how they are computed (per-tick
   counters in `agent_approach_accumulate` / `agent_avoidance_accumulate`,
   generation-cumulative, exposed on `AgentTelemetry` and aggregated by
   `compute_*_intent_fraction` into the `behavior_metric` table); (3) what they measure —
   whether steering correlates with sensed state (deliberate) or not (incidental); (4) how
   to read the baseline distribution that `crates/xagent-brain/tests/intent_baseline_measurement.rs` records
   (mean/std/min/max + p25/p50/p75), noting a follow-up plan will turn the percentiles into
   deliberate (≥ p75) / incidental (≤ p25) classification thresholds via a validation harness.
2. In `README.md`, add a note in §3 (The Cognitive Architecture) or §10 (Why
   Homeostasis-Only Evaluation?) that the intent fractions are an observational lens on
   emergence — when steering consistently aligns with sensed food/danger, intent is high;
   when uncorrelated, motion is incidental — and stress that they are measurement-only with
   zero impact on learning, fitness, or selection.
3. Add a short `README.md` section (§11 or a new §12) 'Measuring Intentional Behavior'
   summarizing the `AgentTelemetry` counter fields and the `behavior_metric`
   `approach_intent_fraction` / `avoidance_intent_fraction` columns, and stating that the
   seeded A/B validation harness that classifies agents against the baseline is a deferred
   follow-up plan.
4. Run `cargo fmt --all` to ensure the markdown edits introduce no formatting issues.

- **Depends on:** measure-baseline-intent-distribution
- **Done when:** The brain-crate README explains the approach/avoidance intent counters,
  how they are computed, and how to read them against the baseline distribution; the main
  README acknowledges the measurement-only intent lens and points forward to the deferred
  validation harness. Documentation-only; `cargo fmt` clean.

---

**End of plan 0015 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
