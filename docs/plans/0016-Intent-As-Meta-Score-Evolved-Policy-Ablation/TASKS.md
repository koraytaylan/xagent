# XAgent Plan 0016 — Intent-As-Meta-Score Evolved-Policy Ablation

Add a `danger_percept_blinded` flag pair to the generational A/B harness in `run_headless_with_flags`; run paired seeded evolution (identical genomes, same world, same seed) with `danger_percept_blinded = false` (sighted) vs `true` (blinded) across G generations; collect all-generations population intent fractions and compute paired mean deltas with 95% CI; measure per-arm death and lifespan statistics to account for generation-cumulative denominators; compare the delta to an A/A noise floor (two sighted runs, same seed) to isolate the causal contribution of danger perception to avoidance steering; report the verdict (DELIBERATE, NEGLIGIBLE-BUT-REAL, INERT, or INDETERMINATE) and surface the measured baseline for threshold calibration; document intent-as-meta-score framework explicitly so future plans can replicate the pattern on another evolvable parameter.

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

## 0001 — Danger-Percept Evolved-Policy A/B Harness

### wire-danger-percept-blinded-flag — Wire danger_percept_blinded Flag Through run_headless_with_flags

Today `run_headless_with_flags` (`headless.rs:634–987`) accepts three boolean flags — `effort_rebased_fitness`, `danger_percept_enabled`, `innate_instincts_enabled` — and runs G generations of evolution, collecting per-generation `AgentFitness` slices and aggregating them into a single `ValidationStats` struct (lines 561–580). The struct already carries `mean_avoidance_intent_fraction` (line 576, computed via `compute_avoidance_intent_fraction(&all_agents_fitness)` at line 966), and the kernel state carries `danger_percept_blinded` in `BrainConfig` (`config.rs:154`, defaults to false). However, `run_headless_with_flags` does not accept or thread the blinding flag, so both arms of a paired A/B run would be identical. This task adds the parameter and ensures it flows through to the GPU kernel and the `config` struct each generation.

**Steps:**
1. In `crates/xagent-sandbox/src/headless.rs` at the `run_headless_with_flags` signature (line 634), add a fourth bool parameter `danger_percept_blinded: bool` after `innate_instincts_enabled`.
2. Inside `run_headless_with_flags`, after line 648 where `config.brain.innate_instincts_enabled = innate_instincts_enabled;`, add `config.brain.danger_percept_blinded = danger_percept_blinded;` to thread the blinding flag to the `config` struct.
3. At line 756 where `kernel.reset_agents_seeded(&current_configs[0], config.world.seed);` is called each generation, ensure the kernel receives the blinded state via `config.brain.danger_percept_blinded`. Verify that `GpuKernel::reset_agents_seeded` uses the current config passed to it (or equivalently, that `kernel.write_agent_heritable_config` applies the config including `danger_percept_blinded` at line 777); if the kernel caches config from initialization, add a mechanism to update `danger_percept_blinded` per-generation (grep `write_heritable_config` or equivalent setter).
4. Add a unit test `test_run_headless_blinded_flag_accepted` in `headless.rs` or the integration tests that calls `run_headless_with_flags(…, 1, false, true, false)` with a sighted (blinded=false) call and `run_headless_with_flags(…, 1, false, true, true)` with a blinded call on the same seed, verifies both return `ValidationStats` without panic, and asserts that the avoidance-intent fraction is non-NaN in both cases (the intent counter must stay alive regardless of blinding; blinding only masks the encoder input).
5. Run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` to ensure the signature edit passes lint.

- **Depends on:** —
- **Done when:** `run_headless_with_flags` accepts a `danger_percept_blinded: bool` parameter and threads it to `config.brain.danger_percept_blinded` each generation; the test verifies that a sighted and blinded run both complete without panic and both report non-NaN avoidance-intent; cargo fmt/clippy/test green.

---

### extend-validation-stats-approach-trajectory — Extend ValidationStats with Per-Generation Approach-Intent Trajectory

The `ValidationStats` struct (line 561) carries per-generation speed trajectory (`speed_trajectory_per_gen: Vec<f32>`, line 579) so the speed ratchet can be visualized. For paired A/B validation, per-generation approach-intent is equally important: it shows whether the signal strengthens or weakens as selection progresses. Today only the aggregate `mean_avoidance_intent_fraction` (computed from all generations, line 963) is stored. This task adds `approach_intent_trajectory_per_gen` to `ValidationStats` and populates it alongside the speed trajectory each generation.

**Steps:**
1. In `crates/xagent-sandbox/src/headless.rs` at the `ValidationStats` struct definition (line 561), add a new field after `speed_trajectory_per_gen: Vec<f32>` (line 579): `pub approach_intent_trajectory_per_gen: Vec<f32>,` with a doc comment: `/// Population-mean approach-intent fraction per generation (chronological order). Used to assess whether approach intent evolves across generations in paired A/B runs.`
2. Initialize the accumulator **before** the generation loop, mirroring `speed_trajectory_per_gen`: immediately after the `let mut speed_trajectory_per_gen: Vec<f32> = Vec::new();` line (line 710, before the `for _ in 0..num_generations` loop at line 712), add `let mut approach_intent_trajectory_per_gen: Vec<f32> = Vec::new();`. Then, inside the loop, after line 845 where `speed_trajectory_per_gen.push(gen_mean_speed);` is called, populate it: `let gen_mean_approach_intent = if fitness.is_empty() { 0.0 } else { compute_approach_intent_fraction(&fitness) }; approach_intent_trajectory_per_gen.push(gen_mean_approach_intent);` (with the necessary import of `compute_approach_intent_fraction` from `governor` if not already present).
3. At the end of `run_headless_with_flags` (line 974), update the `ValidationStats` struct literal to include `approach_intent_trajectory_per_gen,` (must match the new field added to the struct definition).
4. Verify that the field order is consistent: check that all creation sites of `ValidationStats` (there are test fixtures starting around line 1422) are updated. Search for `ValidationStats {` in the file and update each literal to include `approach_intent_trajectory_per_gen: vec![],` or the appropriate value for that test.
5. Run `cargo build -p xagent-sandbox` to ensure the struct is complete; the compiler will error on any missed literal.

- **Depends on:** wire-danger-percept-blinded-flag
- **Done when:** `ValidationStats` carries a `approach_intent_trajectory_per_gen: Vec<f32>` field, populated per-generation inside the loop (line 845 region); every `ValidationStats { … }` literal in the file initializes the field (compiler enforces this); the field is returned in the final `ValidationStats` struct at the end of `run_headless_with_flags`; cargo fmt/clippy/test green.

---

### implement-danger-percept-evolved-ab-harness — Implement Danger-Percept Evolved-Policy A/B Harness (GATED)

**Gate:** Upstream tasks `wire-danger-percept-blinded-flag` and `extend-validation-stats-approach-trajectory` must land first.

The within-life danger-percept ablation (`danger_percept_ablation_ab.rs:1–72`) runs paired seeded runs (sighted vs blinded) within a single kernel lifetime (random init + 100 ticks of within-life learning, no evolution). It measures the within-life delta: avoidance fraction sighted = 0.427, blinded = 0.424, Δ = +0.0036, CI [+0.0008, +0.0065], above noise floor but negligible (|Δ| < 0.05). Now this task extends the harness to the **evolved-policy regime** — run paired seeded evolution across G generations, collecting per-arm population approach/avoidance intent fractions, computing paired deltas with 95% CI, measuring A/A noise floor (two sighted runs, same seed), and reporting a verdict. The harness is the template for future intent-based ablations.

**Steps:**
1. Create a new test function `danger_percept_evolved_ab()` in a new file `crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs` (mirroring the structure of `danger_percept_ablation_ab.rs` but calling the evolved harness instead of the within-life one). Embed the GPU self-skip guard at the top:

   ```rust
   if !xagent_brain::GpuKernel::is_available() {
       eprintln!("Skipping: no GPU/fallback adapter available");
       return;
   }
   ```
2. Inside the test, define constants for the evolved A/B run: `const NUM_GENERATIONS: u64 = 8;` (same as or larger than the within-life run to show evolution signal; balance against CI runtime), `const SEEDS: [u64; 8] = [30250, 30251, 30252, 30253, 30254, 30255, 30256, 30257];` (distinct from within-life seeds to avoid cross-test contamination), `const DANGER_PERCEPT_ENABLED: bool = true;` (both arms keep danger detection on; only the blinding mask differs).
3. Implement a `run_evolved_ab_arm(seed: u64, blinded: bool) -> (Vec<f32>, Vec<f32>, f32, u64, f32, f32)` function that calls `run_headless_with_flags(FullConfig::default(), NUM_GENERATIONS, false, DANGER_PERCEPT_ENABLED, false, blinded)` (with the new `danger_percept_blinded` parameter from the upstream task) and returns a tuple of: (avoidance_intent_trajectory, approach_intent_trajectory, mean_avoidance_intent, mean_lifespan_ticks, mean_death_count, mean_food_consumed). Compute the per-generation deltas inside the function and store them for later reporting.
4. In the main test function, iterate over `SEEDS`. For each seed, call `run_evolved_ab_arm(seed, false)` to get the sighted arm's results and `run_evolved_ab_arm(seed, true)` for the blinded arm. Compute the per-seed avoidance-intent delta (mean sighted − mean blinded) and store it in a `deltas: Vec<f32>` vector. Similarly, collect mean lifespan and death count per arm per seed.
5. After the seed loop, compute the aggregate statistics: mean delta, std delta, 95% CI using the formula `ci95 = 1.96 * (std_delta / sqrt(num_seeds))` (normal approximation, as in the within-life test line 247). Also compute an A/A noise floor: call `run_evolved_ab_arm(SEEDS[0], false)` twice (both sighted, same seed) and compute the absolute delta between the two runs' mean avoidance fractions.
6. Apply the verdict rules (lines 279–290 of `danger_percept_ablation_ab.rs`): output lines clearly labeling the mean avoidance fractions (sighted, blinded), the delta with CI, the noise floor, and the verdict string (DELIBERATE if Δ ≥ +0.05 and CI lower bound > noise; NEGLIGIBLE-BUT-REAL if |Δ| < 0.05 but CI excludes noise; INERT if within noise; else INDETERMINATE).
7. Embed record-then-paste measurements: run the test locally, capture the mean/std/CI/noise-floor numbers, and paste them into a doc comment at the top of the test file (mirroring `danger_percept_ablation_ab.rs:33–71`). Document the hardware, the measured lifespan and death counts per arm, and the approach-intent (food-blind) trajectory as well, for context.
8. Assert only the determinism check: assert that the A/A noise floor is < 0.02 (same as the within-life test, line 296), proving the kernel is bit-deterministic so the A/B delta is interpretable. Do NOT assert a threshold on the delta; this is measurement-only.
9. Place this test at `crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs` — it must live in **xagent-sandbox**, not xagent-brain, because it calls `run_headless_with_flags` which lives in xagent-sandbox, and xagent-brain does not (and must not) depend on xagent-sandbox. Make the harness items importable: `run_headless_with_flags` and `ValidationStats` are currently **private** in `headless.rs` (the `fn run_headless_with_flags(…)` definition at line 634 has no `pub` keyword, and the `struct ValidationStats {…}` definition at line 561 has no `pub` keyword), so the test cannot import them. Change line 634 to `pub fn run_headless_with_flags(…)` and the line 561 struct definition to `pub struct ValidationStats {…}`. Then add the import in the test: `use xagent_sandbox::{run_headless_with_flags, ValidationStats};` plus `use xagent_shared::FullConfig;` (`FullConfig` is already public via `xagent_shared`). Confirm by grepping `pub fn run_headless_with_flags` and `pub struct ValidationStats` in `headless.rs` after the edit.

- **Depends on:** wire-danger-percept-blinded-flag, extend-validation-stats-approach-trajectory
- **Done when:** A `danger_percept_evolved_ab()` test exists, runs paired seeded evolved-policy A/B (sighted vs blinded) for G generations, collects and reports all statistics per the run-then-paste idiom (per-arm mean avoidance/approach intent trajectories, death counts, lifespans, paired delta, 95% CI, A/A noise floor) in the doc comment, applies verdict logic (DELIBERATE/NEGLIGIBLE-BUT-REAL/INERT/INDETERMINATE), and **asserts only that the A/A noise floor is < 0.02 (determinism check)** — the delta is recorded but not asserted (measurement-only). The test embeds the GPU self-skip guard and runs green; cargo fmt/clippy/test green. Land-or-revert: if the A/A noise floor assertion fails (noise floor ≥ 0.02, kernel non-deterministic), revert the test and record the measured noise floor in STATUS.md as a determinism-failure finding.

---

## 0002 — Intent-As-Meta-Score Framework Documentation

### document-intent-as-meta-score-framework — Document Intent-As-Meta-Score Framework and Future-Ablation Recipe

The danger-percept within-life ablation (`danger_percept_ablation_ab.rs`) and now the evolved-policy harness (from the upstream gated task) establish a measurement pattern: paired seeded A/B runs, CI gating, A/A noise-floor validation, verdict logic. However, the framework is implicit — embedded in test comments and decision rules. A future plan that wants to measure another evolvable parameter (e.g., a hypothetical `food_percept_enabled`, or `danger_cost_enabled`, or `slow_damage_enabled`) must reverse-engineer the pattern from the danger-percept pilot. This task documents the framework explicitly in the main README and the brain README, so the pattern is discoverable, and includes a recipe for future evolvable-parameter ablations.

**Steps:**
1. In `README.md`, add a new section (§12 or §13, after the current intent discussion) titled `## Intent-Based Selection Framework`. Open with a paragraph explaining that intent fractions (approach/avoidance) are evolutionary selection criteria, measured at the population level across generations, and compared between conditions (e.g., sighted vs blinded encoders) via paired seeded runs. State that the framework is measurement-only (intent never flows back to the learning signal, respecting homeostasis-only constraint), and that the first instance (danger-percept blinded/sighted A/B on evolved policy) establishes the pattern.
2. In the same section, add a subsection `### Paired-Seeded A/B Pattern`. Explain: seed identity ensures both arms start with identical initial genomes and world; `danger_percept_blinded` (or the flag for a future parameter) masks the encoder input while counters stay alive; delta = sighted − blinded isolates the causal contribution of the percept to steering; 95% CI gating with A/A noise floor (two sighted runs, same seed) separates real effects from deterministic jitter. Document that per-arm lifespan and death counts must be reported to account for generation-cumulative denominators (avoidance = turns_opposing / sense_range_ticks, aggregated per-generation then per-arm).
3. In the same section, add a subsection `### Verdict Thresholds` (or inline as a table). List the decision rules explicitly: DELIBERATE iff Δ ≥ +0.05 and CI lower bound > noise floor; NEGLIGIBLE-BUT-REAL iff |Δ| < 0.05 and CI excludes noise; INERT iff |Δ| < 0.05 and within noise; else INDETERMINATE. Note that these thresholds are calibrated from the danger-percept pilot and may be revised if future measurements suggest different resolution.
4. In the same section, add a subsection `### Recipe for Future Evolvable-Parameter Ablations`. List the steps: (1) extend `run_headless_with_flags` with a new flag (e.g., `food_percept_blinded`); (2) run paired seeded evolution (flag off vs on, identical seed, same world); (3) collect per-arm population intent trajectories and per-generation statistics via `ValidationStats`; (4) compute mean delta + 95% CI; (5) measure A/A noise floor; (6) apply verdict rules; (7) record measured delta and interpretation; (8) update this framework doc with new findings. Link to the danger-percept harness as the template (`crates/xagent-sandbox/tests/danger_percept_evolved_ab.rs`).
5. In `crates/xagent-brain/README.md`, locate the "Intent & Awareness Telemetry" section (added by Plan 0015). After the subsection explaining approach/avoidance counters, add a new subsection `#### Evolved-Policy Intent Validation` explaining the two intent measurement regimes: **Within-life** (random init + TD, e.g., `intent_baseline_measurement.rs`) measures whether counters increment when the target is in range; **Evolved-policy** (generational selection, e.g., `danger_percept_evolved_ab.rs`) measures whether evolution amplifies or diminishes the causal brain→steering coupling. Document the paired-seeded A/B pattern and why it isolates causality (seed identity, blinding mask, delta interpretation). Note that the danger-percept pilot showed +0.0036 within-life delta (negligible) and [measured evolved delta to be inserted when task lands], offering evidence of how the two regimes differ.
6. In the same brain README section, add a subsection `#### Intent-Based Threshold Calibration` explaining why thresholds (e.g., ≥ +0.05 for DELIBERATE) can only be set after the measured baseline exists. Point forward to the framework doc in the main README and the danger-percept pilot results as the calibration data.
7. In `README.md`, in the "Why Homeostasis-Only Evaluation?" section (§10 or equivalent), add a note clarifying that intent fractions are computed post-hoc from telemetry and never flow back to the kernel's learning signal. Restate that this respects the homeostasis-only constraint (Plan 0012, Locked Decision). Evolution (via `run_headless_with_flags` and selection) is the only place intent fractions matter.
8. Run `cargo fmt --all` to ensure the markdown edits are clean.

- **Depends on:** implement-danger-percept-evolved-ab-harness
- **Done when:** `README.md` contains a new §12 (or later) titled "Intent-Based Selection Framework" with subsections on the paired-seeded pattern, verdict thresholds, and a recipe for future evolvable-parameter ablations. `crates/xagent-brain/README.md` contains an "Evolved-Policy Intent Validation" subsection and a threshold-calibration note. Both READMEs clarify that intent is selection-only and respects homeostasis-only learning. Documentation-only; `cargo fmt` clean. (No behavioral changes or tests.)

---

**End of plan 0016 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
