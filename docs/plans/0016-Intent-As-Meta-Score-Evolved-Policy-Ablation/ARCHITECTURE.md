# Architecture — Plan 0016 (deltas)

> Edits in `crates/xagent-sandbox/src/headless.rs`,
> `crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs` (new),
> `README.md`, and `crates/xagent-brain/README.md`.
> Line numbers are hints; locate by symbol (grep for `run_headless_with_flags`,
> `ValidationStats`, `speed_trajectory_per_gen`, `compute_approach_intent_fraction`,
> `danger_percept_blinded`).

## 0001 — Danger-Percept Evolved-Policy A/B Harness

Today `run_headless_with_flags` (`headless.rs:634-987`) accepts three flags —
`effort_rebased_fitness`, `danger_percept_enabled`, `innate_instincts_enabled`
(`headless.rs:637-639`) — threads each onto `config.brain.*` (`headless.rs:646-648`), runs
`num_generations` of evolution, collects per-generation `AgentFitness` slices, and
aggregates them into a single `ValidationStats` struct (`headless.rs:561-580`) with
population-level metrics: `mean_fitness`, `mean_movement_speed`, `mean_ticks_alive`,
`mean_death_count`, `mean_food_consumed`, `mean_danger_dwell_fraction`,
`mean_avoidance_intent_fraction` (the last computed from all generations at
`headless.rs:963`). The danger percept is gated by `danger_percept_enabled`, but there is
no mask to *blind* the encoder after detection runs, so both arms of a paired A/B would be
byte-identical. The kernel-side blinding mask already exists — `WC_DANGER_PERCEPT_BLINDED`
in `buffers.rs:489`, packed into the world-config by `write_agent_heritable_config`
(`buffers.rs:674-737`) — and `config.brain.danger_percept_blinded` already defaults to
`false`; only the harness plumbing is missing. The current harness also records only the
*aggregate* `mean_avoidance_intent_fraction`, while it already tracks a per-generation
`speed_trajectory_per_gen: Vec<f32>` (`headless.rs:579`, pushed at `headless.rs:845`) — but
no per-generation approach-intent trajectory, which paired-A/B delta analysis and
cross-generation visualization need.

Edits:

- **Add `danger_percept_blinded` to the `run_headless_with_flags` signature**
  (`headless.rs:634`): a fourth bool parameter after `innate_instincts_enabled`, threaded to
  the `config` struct beside the existing instinct flag (`headless.rs:648`). The flag feeds
  the encoder the "no danger" sentinel (distance `1.0`, bearing `0.0`) while the physics
  counter stays alive — it is a mask-only knob, orthogonal to learning, fitness, and
  selection.

```rust
/// `danger_percept_blinded`: masks the danger bearing/distance fed to the brain encoder
/// (sentinel: distance 1.0, bearing 0.0) while the avoidance counter keeps accumulating.
/// Mask-only — never touches learning, fitness, or selection. The sighted (false) vs
/// blinded (true) pair is the A/B that isolates the causal danger→steering path.
fn run_headless_with_flags(
    config: FullConfig,
    num_generations: u64,
    effort_rebased_fitness: bool,
    danger_percept_enabled: bool,
    innate_instincts_enabled: bool,
    danger_percept_blinded: bool,
) -> ValidationStats {
```

- **Wire the flag to the GPU kernel each generation** (`headless.rs:756`): ensure the
  per-generation `reset_agents_seeded` / `write_agent_heritable_config` reflects the per-arm
  blinding state, so both arms differ on exactly the one bit. `write_agent_heritable_config`
  already packs `danger_percept_blinded` into `WC_DANGER_PERCEPT_BLINDED`
  (`buffers.rs:705`); the edit is to source it from `config.brain.danger_percept_blinded`
  rather than a hard-coded `false`.

- **Add a per-generation approach-intent trajectory to `ValidationStats`**
  (`headless.rs:561-580`), mirroring `speed_trajectory_per_gen` (`headless.rs:579`):

```rust
/// Population-mean approach-intent fraction per generation (chronological order). Used to
/// assess whether approach intent strengthens or weakens across generations in paired A/B
/// runs — the food-side companion to `speed_trajectory_per_gen`.
pub approach_intent_trajectory_per_gen: Vec<f32>,
```

- **Populate it inside the generation loop** (`headless.rs:836-848`), next to the speed-
  trajectory push (`headless.rs:845`), via the production reducer
  `compute_approach_intent_fraction` (`governor.rs:216`) — the same aggregator selection
  reads, so the per-generation number is identical and trustworthy:

```rust
// Companion to the speed-trajectory push: the population approach-intent fraction for
// this generation, computed with the same reducer production aggregation uses.
let gen_mean_approach_intent =
    if fitness.is_empty() { 0.0 } else { compute_approach_intent_fraction(&fitness) };
approach_intent_trajectory_per_gen.push(gen_mean_approach_intent);
```

- **A new measurement test** `crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs`,
  mirroring the within-life harness `danger_percept_ablation_ab.rs:1-72` but calling the
  *evolved* harness. Per seed it runs a sighted arm (`danger_percept_blinded = false`) and a
  blinded arm (`= true`) across `NUM_GENERATIONS`, both with `danger_percept_enabled = true`
  and the identical world seed, so genomes and world match and only the mask differs. It
  collects per-arm avoidance/approach intent trajectories, mean avoidance fraction, mean
  lifespan, mean death count, and mean food consumed; computes the paired mean delta
  (sighted − blinded), its sample std, and a 95% CI by the normal approximation
  `ci95 = 1.96 * std_delta / sqrt(num_seeds)` (the formula the within-life test uses at
  `danger_percept_ablation_ab.rs:247`); and measures an A/A noise floor by running the same
  seed sighted twice and taking the absolute fraction difference. It embeds the GPU
  self-skip guard verbatim and follows the run-then-paste idiom — measured numbers pasted
  into the file's doc comment (mirroring `danger_percept_ablation_ab.rs:33-71`):

```rust
/// Evolved-policy danger-percept A/B: paired seeded evolution, sighted vs blinded encoder,
/// identical genomes/world/seed. Records per-arm mean avoidance/approach intent, lifespan,
/// death counts; paired delta + 95% CI vs A/A noise floor; verdict per SCOPE decision rules.
/// Measurement-only: asserts ONLY that the A/A noise floor < 0.02 (kernel determinism), so
/// the A/B delta is interpretable. The delta itself is recorded, not asserted.
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

  Verdict rules (DELIBERATE / NEGLIGIBLE-BUT-REAL / INERT / INDETERMINATE) and the
  per-arm lifespan/death-count accounting that normalizes generation-cumulative denominators
  are the decision policy — stated in SCOPE (locked decisions), reported but not asserted here.

Properties that make this safe:
- `danger_percept_blinded` is a mask-only parameter, orthogonal to learning, fitness, and
  selection; both the physics counter and the world-config slot already exist
  (`buffers.rs:489`, `buffers.rs:705`), so the edit only threads the bit through harness
  plumbing — run semantics are unchanged when the flag is `false` (the default both prior
  call sites pass).
- The per-generation approach intent aggregates via the same `compute_approach_intent_fraction`
  reducer (`governor.rs:216`) production uses, so the trajectory is identical to what
  selection sees — no parallel implementation to drift.
- Adding `approach_intent_trajectory_per_gen` is a post-hoc measurement vector that changes
  neither run semantics nor performance; the compiler enforces that every `ValidationStats { … }`
  literal (including the test fixtures from `headless.rs:1432`) initializes the new field.
- The paired-A/B pairing holds because both arms share seed, world, and initial genomes and
  differ only in the single mask bit; the kernel is bit-deterministic (A/A noise floor ≈ 0),
  so the measured delta is the causal effect of blinding, and the test asserts that
  determinism (A/A < 0.02) rather than any threshold on the delta.

## 0002 — Intent-As-Meta-Score Framework Documentation

Today the danger-percept within-life ablation (`danger_percept_ablation_ab.rs`) documents
the measurement idiom — paired runs, seed identity, delta/CI, A/A noise floor, verdict logic
— only inline as comments and measured numbers in the test file
(`danger_percept_ablation_ab.rs:27-31`, `:33-71`, `:279-290`). The framework is implicit: a
future plan that wants to measure another evolvable parameter (a hypothetical
`food_percept_blinded`, `danger_cost_enabled`, `slow_damage_enabled`) must reverse-engineer
the pattern from this one test. There is no explicit statement of the intent-as-meta-score
principle, the threshold logic, or the lifespan-accounting considerations. Plan 0015's
deferred-validation note (`docs/plans/0015-Intent-Awareness-Measurement-Framework/SCOPE.md:99-104`)
flagged exactly this gap — that thresholds can only be set from real baseline numbers — but
left the framework itself unwritten.

Edits (documentation-only; no code or behavior changes):

- **Add an "Intent-Based Selection Framework" section to `README.md`** (§12 or later, after
  the current intent discussion). State the intent-as-meta-score principle: intent fractions
  (approach / avoidance) are evolutionary *selection criteria*, measured as population
  aggregates across generations, compared between conditions via paired seeded runs with 95%
  CI gating, validated against an A/A noise floor, with per-arm lifespan/death-count
  accounting normalizing the generation-cumulative denominators. State that the framework is
  measurement-only (never a learning term; respects the homeostasis-only constraint), and
  that the first instance — danger-percept blinded/sighted A/B on evolved policy — establishes
  the pattern that subsequent evolvable parameters follow identically. Include three
  subsections: `### Paired-Seeded A/B Pattern` (seed identity ⇒ identical genomes and world;
  the blinding mask keeps counters alive; delta = sighted − blinded isolates the causal
  percept→steering path; A/A noise floor separates real effects from deterministic jitter),
  `### Verdict Thresholds` (the decision rules, marked domain-calibrated from the pilot and
  subject to revision), and `### Recipe for Future Evolvable-Parameter Ablations` (the
  eight-step recipe: extend `run_headless_with_flags` with a new flag; run paired seeded
  evolution; collect per-arm intent trajectories and statistics via `ValidationStats`;
  compute mean delta + 95% CI; measure A/A noise floor; apply verdict rules; record and
  interpret the measured delta; update this framework doc), linking the danger-percept
  harness (`crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs`) as the template.

- **Add an "Evolved-Policy Intent Validation" subsection to `crates/xagent-brain/README.md`**
  under "Intent & Awareness Telemetry" (added by Plan 0015). Explain the two intent regimes:
  **within-life** (random init + TD, e.g. `intent_baseline_measurement.rs`) measures whether
  counters increment when the target is in range — the baseline Plan 0015 established; and
  **evolved-policy** (generational selection, e.g. `danger_percept_evolved_ab.rs`) measures
  whether evolution amplifies or diminishes the causal brain→steering coupling. Document the
  paired-seeded A/B pattern and why it isolates causality (seed identity, blinding mask,
  delta interpretation), and note that the within-life pilot showed a +0.0036 delta
  (negligible) against the evolved delta the harness records. Add a
  `#### Intent-Based Threshold Calibration` subsection explaining why thresholds (e.g.
  ≥ +0.05 for DELIBERATE) can only be set after a measured baseline exists, pointing at the
  framework doc in the main README and the pilot results as the calibration data.

- **Document why homeostasis-only learning is untouched** in both READMEs (main README's
  "Why Homeostasis-Only Evaluation?" section and the brain README's new subsection):
  emphasize that intent fractions are computed post-hoc from telemetry and never flow back to
  the kernel's learning signal — no new reward term, no fitness penalty. Selection (via
  `run_headless_with_flags` and the evolution loop) is the only place intent fractions
  matter. This respect for the homeostasis-only constraint is a Locked Decision carried from
  Plan 0012 and Plan 0015 and is preserved here.

Properties that make this safe:
- Documentation-only edits; no code or behavior changes, so no functional gate can regress.
- The framework documents the already-lived within-life ablation extended to evolved policy
  — empirical practice, not speculation.
- Threshold values (±0.05, CI gating, noise-floor logic) are derived from the measured
  danger-percept results and are explicitly marked domain-calibrated, subject to revision if
  future measurements on other parameters show different effect sizes; the framework is the
  pattern, not the exact numbers.
