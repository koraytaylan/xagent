# XAgent Plan 0013 — Innate Survival Instincts

This plan seeds the pattern memory with evolved biosimilar instinct signatures at agent initialization—a danger context (high negative valence) paired with avoidance motor priors, and an energy-gain context (high positive valence) paired with approach priors—as one-time injected priors subject to normal reinforcement/decay/eviction like learned patterns, rather than standing external reward shaping. Add heritable instinct-strength genes to BrainConfig (instinct_danger_strength, instinct_food_strength), thread them through init and evolution, and gate all seeding behind innate_instincts_enabled (default off, byte-identical when off). Author a headless A/B benchmark (seeded-instinct agents vs blank-slate baseline on identical worlds, seeded mutations) measuring population mean survival ticks, steering-alignment (mirrored-steering intent vs danger bearing), and deaths-per-food consumed. Define explicit pass thresholds (e.g. survival +10%, alignment ≥0.4, food-per-death ≥2.0), mark the result in a decision doc, and lock the flag to off until the gate passes with a documented follow-up condition.

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

## 0001 — Innate-Pattern-Seeding

### add-instinct-config-fields — Add Heritable Instinct-Strength Fields to BrainConfig

BrainConfig currently carries four heritable visual-genome scalar genes (gabor_wavelength, gabor_aspect_ratio, dog_surround_ratio, orientation_offset) defined in `crates/xagent-shared/src/config.rs:150–178`. Two new heritable f32 fields are needed: instinct_danger_strength and instinct_food_strength, each controlling the valence magnitude of the seeded danger and food instinct patterns. They must be serializable (serde), carry default values, and include doc-comments explaining their role.

**Steps:**
1. Open `crates/xagent-shared/src/config.rs` and locate the BrainConfig struct definition around line 23.
2. After the `orientation_offset` field (line 178), add two new fields with doc-comments and serde defaults:
```rust
    /// **Heritable (innate instinct gene).** Multiplier for the danger instinct
    /// pattern's negative valence, controlling how strongly aversive the seeded danger
    /// prior is. Seed 0.8; mutated during breeding, clamped to
    /// `[INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_DANGER_STRENGTH_MAX]` = `[0.1, 1.0]`.
    #[serde(default = "default_instinct_danger_strength")]
    pub instinct_danger_strength: f32,
    /// **Heritable (innate instinct gene).** Multiplier for the food instinct
    /// pattern's positive valence, controlling how strongly appetitive the seeded
    /// food/energy-gain prior is. Seed 0.8; mutated during breeding, clamped to
    /// `[INSTINCT_FOOD_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX]` = `[0.1, 1.0]`.
    #[serde(default = "default_instinct_food_strength")]
    pub instinct_food_strength: f32,
```
3. After the `ORIENTATION_OFFSET_PERIOD` constant (line 186), add the new clamp constant pairs:
```rust
/// Inclusive clamp bounds for the heritable instinct-strength scalar genes (plan
/// 0013). These control the magnitude of the seeded danger and food instinct
/// patterns; they are re-imposed during breeding and in the shader after seeding.
pub const INSTINCT_DANGER_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_DANGER_STRENGTH_MAX: f32 = 1.0;
pub const INSTINCT_FOOD_STRENGTH_MIN: f32 = 0.1;
pub const INSTINCT_FOOD_STRENGTH_MAX: f32 = 1.0;
```
4. Add the two default functions at the end of the file (after the existing `default_*` functions, around line 270+):
```rust
fn default_instinct_danger_strength() -> f32 {
    0.8
}

fn default_instinct_food_strength() -> f32 {
    0.8
}
```

5. Add the two new fields to **every exhaustive `BrainConfig` literal** so the workspace compiles after this task alone: the `tiny`/`default`/`large` presets in `crates/xagent-shared/src/config.rs` (seed each to `0.8`); any `BrainConfig { .. }` literals in tests; AND the literals in `mutate_config_with_strength_rng` and `crossover_config` (`crates/xagent-sandbox/src/agent/mod.rs`) — for now as simple passthroughs (`instinct_danger_strength: parent.instinct_danger_strength,` and `instinct_food_strength: parent.instinct_food_strength,` in the mutation literal; `a.instinct_danger_strength`/`a.instinct_food_strength` in crossover). `add-instinct-mutation-to-breeding` then replaces those passthroughs with the real drift/inheritance logic. (Adding a struct field makes all exhaustive literals incomplete until updated — the compiler lists each one.)

- **Depends on:** —
- **Done when:** BrainConfig has the two new f32 fields with serde defaults and doc-comments (no plan/process language in the comments — `contributing_guard.rs` enforces this); the four clamp constants are defined; every `BrainConfig` preset/test literal is updated so the workspace compiles; cargo fmt/clippy/test green.

---

### implement-seed-instinct-patterns — Implement seed_instinct_patterns() Helper Function

The pattern memory buffer layout (`buffers.rs:130–141`) includes encoded states (O_PAT_STATES), norms (O_PAT_NORMS), reinforcement (O_PAT_REINF), motor vectors with valence (O_PAT_MOTOR), metadata, and active flags. A new helper function must populate two instinct pattern slots (danger and food) with fixed encoded signatures, unit norms, unit reinforcement, pre-filled motor vectors (avoidance for danger, approach for food), and valence magnitudes scaled by the heritable strength genes. The function must return a zero-filled PATTERN_STRIDE-length vector with just the two instinct slots populated.

**Steps:**
1. Open `crates/xagent-brain/src/buffers.rs` and locate `init_pattern_memory()` at line 803.
2. After `init_pattern_memory()`, add the new function:
```rust
/// Seed two innate instinct patterns into the pattern buffer for one agent.
/// Instinct 0 (slot 0): danger context — high negative valence paired with
/// avoidance motor priors (backward, evasive turn).
/// Instinct 1 (slot 1): energy-gain context — high positive valence paired with
/// approach motor priors (forward, no turn bias).
/// Both are subject to normal recall, decay, and reinforcement like learned patterns.
/// Returns a vector of length PATTERN_STRIDE with the two priors pre-filled and
/// remaining slots zero-initialized.
pub fn seed_instinct_patterns(danger_strength: f32, food_strength: f32) -> Vec<f32> {
    let mut patterns = vec![0.0_f32; PATTERN_STRIDE];
    
    // ─ Instinct 0: danger (slot 0) ─
    // Encoded state: fixed signature for "danger detected" (e.g., all -0.5).
    for d in 0..ENCODED_DIMENSION {
        patterns[O_PAT_STATES + d * MEMORY_CAP + 0] = -0.5;
    }
    patterns[O_PAT_NORMS + 0] = 1.0;
    patterns[O_PAT_REINF + 0] = 1.0;
    // Motor: [forward, turn, valence]
    patterns[O_PAT_MOTOR + 0 * 3 + 0] = -0.7;  // backward (escape behavior).
    patterns[O_PAT_MOTOR + 0 * 3 + 1] = 0.5;   // evasive turn.
    patterns[O_PAT_MOTOR + 0 * 3 + 2] = -danger_strength; // strong negative valence.
    // Active and metadata
    patterns[O_PAT_ACTIVE + 0] = 1.0;
    patterns[O_ACTIVE_COUNT] = 1.0;
    patterns[O_LAST_STORED_IDX] = 1.0;
    
    // ─ Instinct 1: energy gain / food (slot 1) ─
    // Encoded state: fixed signature for "food/energy detected" (e.g., all +0.5).
    for d in 0..ENCODED_DIMENSION {
        patterns[O_PAT_STATES + d * MEMORY_CAP + 1] = 0.5;
    }
    patterns[O_PAT_NORMS + 1] = 1.0;
    patterns[O_PAT_REINF + 1] = 1.0;
    // Motor: [forward, turn, valence]
    patterns[O_PAT_MOTOR + 1 * 3 + 0] = 0.7;   // forward (approach behavior).
    patterns[O_PAT_MOTOR + 1 * 3 + 1] = 0.0;   // no turn bias.
    patterns[O_PAT_MOTOR + 1 * 3 + 2] = food_strength; // strong positive valence.
    // Active and metadata
    patterns[O_PAT_ACTIVE + 1] = 1.0;
    patterns[O_ACTIVE_COUNT] = 2.0;
    patterns[O_LAST_STORED_IDX] = 2.0;
    
    patterns
}
```
3. Verify the function compiles by running `cargo check -p xagent-brain`. The function uses only public constants (O_PAT_STATES, O_PAT_NORMS, O_PAT_REINF, O_PAT_MOTOR, O_PAT_ACTIVE, O_ACTIVE_COUNT, O_LAST_STORED_IDX, ENCODED_DIMENSION, MEMORY_CAP, PATTERN_STRIDE) already defined in the module.

- **Depends on:** add-instinct-config-fields, add-instinct-mutation-to-breeding
- **Done when:** The function seed_instinct_patterns(danger_strength, food_strength) is defined in buffers.rs, accepts two f32 parameters (instinct strength genes), returns a PATTERN_STRIDE-length vector with slots 0 and 1 populated (danger and food instincts) and the rest zero-filled, and cargo fmt/clippy/test green.

---

## 0002 — Heritable-Instinct-Config

### add-instinct-mutation-to-breeding — Thread Instinct Genes Through Mutation + Crossover

The exhaustive `BrainConfig { .. }` literals that set heritable genes live in two functions in `crates/xagent-sandbox/src/agent/mod.rs`: `mutate_config_with_strength_rng` (≈line 408 — the visual genes are set ≈line 479 via `momentum.biased_perturb_f(rng, parent.<gene>, "<gene>", strength)` then clamp) and `crossover_config` (≈line 587 — visual genes inherited ≈line 657 via 50/50 `if rng.random::<f32>() < 0.5 { a.<gene> } else { b.<gene> }`). Both new instinct genes must be threaded through BOTH, using those exact existing idioms. (A raw `rng.random()` drift would bypass the directed-mutation momentum — do not use it. The wrappers `mutate_config`/`mutate_config_seeded` just call `mutate_config_with_strength_rng`, so they are NOT the edit sites.)

**Steps:**
1. In `mutate_config_with_strength_rng` (≈408), replace the placeholder passthrough entries (added by `add-instinct-config-fields`) for the two genes with momentum-biased drift + clamp, mirroring the visual-genome block:
```rust
instinct_danger_strength: momentum
    .biased_perturb_f(rng, parent.instinct_danger_strength, "instinct_danger_strength", strength)
    .clamp(INSTINCT_DANGER_STRENGTH_MIN, INSTINCT_DANGER_STRENGTH_MAX),
instinct_food_strength: momentum
    .biased_perturb_f(rng, parent.instinct_food_strength, "instinct_food_strength", strength)
    .clamp(INSTINCT_FOOD_STRENGTH_MIN, INSTINCT_FOOD_STRENGTH_MAX),
```
   Import the clamp consts from `xagent_shared`; follow whatever exact clamp idiom `gabor_wavelength` uses.
2. In `crossover_config` (≈587), replace the passthrough entries with the 50/50 parent-inheritance pattern the visual genes use:
```rust
instinct_danger_strength: if rng.random::<f32>() < 0.5 { a.instinct_danger_strength } else { b.instinct_danger_strength },
instinct_food_strength:   if rng.random::<f32>() < 0.5 { a.instinct_food_strength }   else { b.instinct_food_strength },
```
3. The wrappers `mutate_config` (≈367) and `mutate_config_seeded` (≈399) call `mutate_config_with_strength_rng`, so they need no direct edit — confirm they still compile.
4. Extend the gene-bounds test (mirror `mutate_config_respects_visual_gene_bounds`, ≈mod.rs:925) to assert both instinct genes stay within `[0.1, 1.0]` after mutating from out-of-range parents.

- **Depends on:** add-instinct-config-fields
- **Done when:** Both `mutate_config_with_strength_rng` and `crossover_config` thread `instinct_danger_strength`/`instinct_food_strength` (momentum-biased drift + clamp for mutation; 50/50 inheritance for crossover); a bounds test covers them; cargo fmt/clippy/test green.

---

## 0003 — Default-Off-Gating

### add-instinct-gate-flag — Add innate_instincts_enabled Flag to BrainConfig

BrainConfig already carries three gate flags (visual_cortex_enabled, danger_percept_enabled, effort_rebased_fitness) defined around `config.rs:127–149`. These are locked per batch (not heritable), control entire shader passes or metrics recomputations, and default to false. A new innate_instincts_enabled flag must follow the same pattern: default false (byte-identical to blank-slate behavior), enabled only after the prove-or-kill A/B gate passes.

**Steps:**
1. Open `crates/xagent-shared/src/config.rs` and locate the effort_rebased_fitness field (line 142–149).
2. After effort_rebased_fitness, add the new gate flag:
```rust
    /// Gate flag for seeded innate instinct priors. When `false`,
    /// pattern memory initializes to all zeros (blank slate, byte-identical to
    /// pre-instinct behavior). When `true`, danger and food instinct patterns are
    /// seeded at initialization via seed_instinct_patterns(). Locked per batch,
    /// not heritable. Default `false` until the prove-or-kill A/B gate passes.
    #[serde(default)]
    pub innate_instincts_enabled: bool,
```

- **Depends on:** add-instinct-config-fields, add-instinct-mutation-to-breeding
- **Done when:** BrainConfig struct has the new bool field innate_instincts_enabled with doc-comment and serde default (false); cargo fmt/clippy/test green.

---

### integrate-seeding-into-reset-path — Route Every Fresh-Agent Init Through One Gated Helper

`init_pattern_memory()` is called for fresh agents at THREE sites in `gpu_kernel.rs`: `reset_agents_with_rng` (≈557), `GpuKernel::new` (≈855), and the agent-grow/add path (≈2820). All must route through a single gated helper so seeding is consistent — gating only the reset path would leave generation-0 agents created by `GpuKernel::new` unseeded, and no test would catch it. When the flag is off, every site calls `init_pattern_memory()`, so the path is byte-identical to the pre-instinct codebase.

**Steps:**
1. In `crates/xagent-brain/src/gpu_kernel.rs`, add a small private helper:
```rust
fn pattern_init_for(config: &BrainConfig) -> Vec<f32> {
    if config.innate_instincts_enabled {
        seed_instinct_patterns(config.instinct_danger_strength, config.instinct_food_strength)
    } else {
        init_pattern_memory()
    }
}
```
2. Replace the per-agent `init_pattern_memory()` call with `pattern_init_for(brain_config)` at ALL fresh-agent fill sites: `reset_agents_with_rng` (≈557), `GpuKernel::new` (≈855), and the agent-grow/add path (≈2820). Grep `init_pattern_memory(` in `gpu_kernel.rs` to confirm every fresh-agent site is covered.
3. Do NOT alter the saved-state restore path (≈2774) that uploads persisted `s.patterns` — that restores saved memory and must stay as-is.
4. `cargo check -p xagent-brain`.

- **Depends on:** implement-seed-instinct-patterns, add-instinct-gate-flag
- **Done when:** A single `pattern_init_for()` helper routes every fresh-agent pattern fill (reset, new, grow) through the `innate_instincts_enabled` gate; the saved-state restore path is unchanged; with the flag off all sites still call `init_pattern_memory()` (byte-identical). A test asserts `pattern_init_for` equals `init_pattern_memory()` output when the flag is off; cargo fmt/clippy/test green.

---

## 0004 — Prove-Or-Kill-Gate

### implement-ab-validation-harness — Implement Innate-Instinct A/B Validation Harness

The headless module (`headless.rs:508–580`) already contains the post-0010 **seeded paired** A/B framework for speed-decoupling (`run_headless_with_flags`, `ValidationStats`, the governor `evaluate` → `AgentFitness` path, explicit gates), where both arms draw identical seeded populations/worlds and differ only in flags. REUSE it — do not write a fresh ad-hoc loop. The new harness runs baseline (flag off) and ON (flag on) on the same `pop_init_seed`/world seed, collects population means from the `AgentFitness` vector (`total_ticks_alive`, `death_count`, `food_consumed`) plus `compute_avoidance_intent_fraction(&[AgentFitness])`, and applies explicit pass/fail gates. **NOTE: the skeleton below is illustrative scaffolding only** — the real implementation must use the seeded-paired runner and the real `AgentFitness` aggregates; food-per-death uses real `food_consumed`, never `mean_fitness` (a composite); add a `mean_food_consumed` field to `ValidationStats` if it is absent.

**Steps:**
1. Open `crates/xagent-sandbox/src/headless.rs` and locate the ValidationStats struct (line 458–474) and the run_headless_with_flags function (line 508).
2. Add three new gate constants after ON_SPEED_COST_EXPONENT (around line 481):
```rust
/// Gate: ON (instincts seeded) must improve survival over baseline by at least this margin.
/// Tuned at 0.10 (10%) to require meaningful benefit while tolerating natural variance.
const INSTINCT_SURVIVAL_MARGIN: f32 = 0.10;

/// Gate: mean avoidance-intent fraction (turns opposing danger bearing) in ON run
/// must exceed this floor. Tuned at 0.4 to require substantial steering-alignment.
const INSTINCT_ALIGNMENT_FLOOR: f32 = 0.4;

/// Gate: ON run's food-per-death ratio (mean food count / mean death count) must
/// exceed this threshold. Tuned at 2.0 to require at least 2 food consumed per death.
const INSTINCT_FOOD_PER_DEATH_MIN: f32 = 2.0;
```
3. Add a new validation function after run_headless_with_flags (around line 700+):
```rust
/// Run innate-instinct prove-or-kill A/B benchmark.
/// Baseline: innate_instincts_enabled=false (blank slate, learning from scratch).
/// ON: innate_instincts_enabled=true (seeded instinct patterns + learning).
/// Both arms use identical seeded populations and worlds (deterministic mutations from pop_init_seed).
/// Returns (baseline_stats, on_stats, passed: bool).
/// Passed iff ON's survival > baseline + margin AND avoidance-intent >= floor AND food-per-death >= threshold.
fn run_innate_instinct_ab(
    mut config: FullConfig,
    num_generations: u64,
) -> (ValidationStats, ValidationStats, bool) {
    // Baseline: flag off (blank slate).
    config.brain.innate_instincts_enabled = false;
    println!("\n=== Baseline (innate_instincts_enabled=false) ===");
    let baseline_stats = run_headless_instinct_validation(config.clone(), num_generations);
    
    // ON: flag on (seeded instincts).
    config.brain.innate_instincts_enabled = true;
    println!("\n=== ON (innate_instincts_enabled=true) ===");
    let on_stats = run_headless_instinct_validation(config.clone(), num_generations);
    
    // Apply gates.
    println!("\n=== Gate Evaluation ===");
    
    let survival_gate = on_stats.mean_ticks_alive >= baseline_stats.mean_ticks_alive * (1.0 + INSTINCT_SURVIVAL_MARGIN);
    println!(
        "Survival gate (ON >= baseline * {:.1}%): baseline={:.1}, ON={:.1} → {}",
        INSTINCT_SURVIVAL_MARGIN * 100.0,
        baseline_stats.mean_ticks_alive,
        on_stats.mean_ticks_alive,
        if survival_gate { "✓ PASS" } else { "✗ FAIL" }
    );
    
    let alignment_gate = on_stats.mean_avoidance_intent_fraction >= INSTINCT_ALIGNMENT_FLOOR;
    println!(
        "Alignment gate (avoidance-intent >= {:.1}): ON={:.3} → {}",
        INSTINCT_ALIGNMENT_FLOOR,
        on_stats.mean_avoidance_intent_fraction,
        if alignment_gate { "✓ PASS" } else { "✗ FAIL" }
    );
    
    // REAL food consumed (population mean), never mean_fitness (a composite).
    // Requires a mean_food_consumed field on ValidationStats — add it if absent,
    // summing AgentFitness::food_consumed / agent count.
    let food_per_death_on = if on_stats.mean_death_count > 1e-4 {
        on_stats.mean_food_consumed / on_stats.mean_death_count
    } else {
        f32::INFINITY
    };
    let food_per_death_gate = food_per_death_on >= INSTINCT_FOOD_PER_DEATH_MIN;
    println!(
        "Food-per-death gate (ratio >= {:.1}): ON={:.2} → {}",
        INSTINCT_FOOD_PER_DEATH_MIN,
        food_per_death_on,
        if food_per_death_gate { "✓ PASS" } else { "✗ FAIL" }
    );
    
    let passed = survival_gate && alignment_gate && food_per_death_gate;
    println!("\nOverall: {}", if passed { "✓✓✓ ALL GATES PASS" } else { "✗✗✗ GATE FAILURE" });
    
    (baseline_stats, on_stats, passed)
}

/// Helper: run headless evolution and collect instinct-specific validation stats.
/// Reuses the existing per-generation loop and fitness calculation.
fn run_headless_instinct_validation(
    config: FullConfig,
    num_generations: u64,
) -> ValidationStats {
    // (Implement by adapting run_headless_with_flags logic:
    // - Create Governor and seeded population using config.world.seed for determinism.
    // - Loop num_generations, calling kernel.dispatch_batch() and collecting:
    //   * mean_ticks_alive (from agent physics state P_TICKS_ALIVE).
    //   * mean_death_count (from agent physics state P_DEATH_COUNT).
    //   * mean_avoidance_intent_fraction (via compute_avoidance_intent_fraction).
    //   * mean_fitness (via governor.current_fitness or local calculation).
    // - Return ValidationStats with collected means.)
    // Skeleton implementation:
    let mut governor = Governor::new(
        &format!("xagent-instinct-{}.db", std::process::id()),
        config.governor.clone(),
        &config.brain,
        "",
    ).expect("Failed to initialize validation governor");
    
    let seed_config = config.brain.clone();
    let pop_init_seed = config.world.seed;
    let mut current_configs: Vec<BrainConfig> = {
        let repeats = governor.config.eval_repeats.max(1);
        let unique_count = (governor.config.population_size / repeats).max(1);
        let mut unique_configs = vec![seed_config.clone()];
        for i in 1..unique_count {
            let mutation_seed = pop_init_seed.wrapping_add(i as u64);
            unique_configs.push(mutate_config_seeded(&seed_config, mutation_seed));
        }
        let mut configs = Vec::with_capacity(governor.config.population_size);
        for uc in &unique_configs {
            for _ in 0..repeats {
                if configs.len() >= governor.config.population_size { break; }
                configs.push(uc.clone());
            }
        }
        configs
    };
    
    let mut ticks_alive_sum = 0.0;
    let mut death_count_sum = 0.0;
    let mut avoidance_intent_sum = 0.0;
    let mut fitness_sum = 0.0;
    let mut sample_count = 0u32;
    
    let pop_size = governor.config.population_size;
    let world_base = WorldState::new(config.world.clone());
    let food_count = world_base.food_items.len();
    let mut kernel = GpuKernel::new(pop_size as u32, food_count, &seed_config, &config.world);
    
    for _ in 0..num_generations {
        if governor.evolution_complete() { break; }
        
        let world = WorldState::new(config.world.clone());
        let agents: Vec<Agent> = current_configs.iter().enumerate()
            .map(|(i, cfg)| Agent::new(i as u32, world.safe_spawn_position(), i as u32, cfg.clone(), 0))
            .collect();
        
        let biomes = world.biome_map.grid_as_u32();
        let food_pos: Vec<(f32, f32, f32)> = world.food_items.iter()
            .map(|f| (f.position.x, f.position.y, f.position.z)).collect();
        let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
        let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
        kernel.upload_world(&world.terrain.heights, &biomes, &food_pos, &food_consumed, &food_timers);
        
        let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agents.iter()
            .map(|a| (a.body.body.position, a.body.body.internal.max_energy, a.body.body.internal.max_integrity, 
                      a.brain_config.memory_capacity, a.brain_config.processing_slots))
            .collect();
        kernel.upload_agents(&agent_data);
        kernel.reset_agents(&current_configs[0]);
        
        // Run ticks and readback
        kernel.dispatch_batch(governor.config.tick_budget);
        
        if let Ok(readback) = kernel.readback_physics_state() {
            for (agent_id, phys_vals) in readback.iter().enumerate() {
                if agent_id >= agents.len() { break; }
                // Extract P_TICKS_ALIVE, P_DEATH_COUNT, etc.
                // (Indices from buffers.rs: P_TICKS_ALIVE=19, P_DEATH_COUNT=23)
                if phys_vals.len() > 23 {
                    ticks_alive_sum += phys_vals[19]; // P_TICKS_ALIVE
                    death_count_sum += phys_vals[23]; // P_DEATH_COUNT
                    sample_count += 1;
                }
            }
        }
        
        // Avoidance intent comes from the governor's per-agent fitness vector:
        // compute_avoidance_intent_fraction takes &[AgentFitness] (NOT &kernel).
        // Build the AgentFitness vec via governor.evaluate(...) for this generation:
        //   let fitness = governor.evaluate(...);
        //   avoidance_intent_sum += compute_avoidance_intent_fraction(&fitness);
        //   food_sum += fitness.iter().map(|f| f.food_consumed as f32).sum::<f32>();
    }
    
    let n_gens = (avoidance_intent_sum / INSTINCT_ALIGNMENT_FLOOR).max(1.0) as u32; // rough; refine
    ValidationStats {
        mean_fitness: 0.0, // TODO: populate from governor.current_fitness()
        mean_movement_speed: 0.0, // TODO: if needed
        mean_ticks_alive: if sample_count > 0 { ticks_alive_sum / (sample_count as f32) } else { 0.0 },
        mean_death_count: if sample_count > 0 { death_count_sum / (sample_count as f32) } else { 0.0 },
        speed_fitness_correlation: 0.0, // not used for instinct gate
        death_speed_regression: 0.0, // not used
        food_per_energy_vs_speed_slope: 0.0, // not used
        mean_danger_dwell_fraction: 0.0, // not used
        mean_avoidance_intent_fraction: if n_gens > 0 { avoidance_intent_sum / (n_gens as f32) } else { 0.0 },
        speed_trajectory_per_gen: vec![],
    }
}
```

- **Depends on:** integrate-seeding-into-reset-path
- **Done when:** `run_innate_instinct_ab()` runs the seeded-paired baseline-vs-ON benchmark off identical seeds (the flag is the only difference), aggregates REAL `food_consumed`/`death_count`/`total_ticks_alive` and `compute_avoidance_intent_fraction(&[AgentFitness])` from the governor's fitness vector, and evaluates the three gates — survival (+10%), alignment (≥0.4), food-per-death (≥2.0, real food/deaths) — with GPU-free unit tests for the gate predicate; a `--validate-innate-instincts` CLI arm runs it; cargo fmt/clippy/test green.

---

### author-ab-gate-decision-doc — Run A/B Gate and Author Decision Document (GATED)

**Gate:** All prior tasks (seeding, gating, validation harness) must be merged first. This task runs the headless A/B benchmark, records the measured results, and authors the 0013-INNATE-INSTINCT-DECISION.md document capturing the decision (PASS → flag-default-true follow-up planned; FAIL → rejection with revisit conditions).

**Steps:**
1. After all prior tasks land, run the innate-instinct A/B benchmark in headless mode (command-line invocation or new test harness). Capture the output: baseline vs ON statistics (mean_ticks_alive, mean_death_count, mean_avoidance_intent_fraction) and pass/fail status for each gate.
2. Record the measured numbers in a temporary file or log.
3. Create the decision document `docs/plans/0013-Innate-Survival-Instincts/0013-INNATE-INSTINCT-DECISION.md` with the following structure:
```markdown
# Decision — Plan 0013: Innate Survival Instincts

> This task runs the headless A/B benchmark measuring seeded-instinct agents
> (innate_instincts_enabled=true, heritable danger/food strength genes) vs
> blank-slate baseline (innate_instincts_enabled=false) on identical seed-deterministic
> worlds. Done when: pass/fail gates are evaluated and recorded, and the decision
> (land seeded-instinct default-true follow-up or reject with revisit conditions)
> is documented.

## Decision

**Result:** [PASS / FAIL]

[If PASS:] The innate instinct seeding mechanism passed all three gates:
- Survival improved by [X]% (≥10% threshold: ✓).
- Steering-alignment (avoidance-intent) reached [X] fraction (≥0.4 threshold: ✓).
- Food-per-death ratio [X] (≥2.0 threshold: ✓).

Recommendation: Proceed with a follow-up plan to flip innate_instincts_enabled=true by default and measure long-term learning impact.

[If FAIL:] [Reason — which gate(s) failed and by how much.] The instinct mechanism did not achieve the survival or alignment threshold required to demonstrate clear benefit. Recommendation: reject this iteration and document revisit conditions below.

## The three paths

| Path | What it is | Benefit | Risk / cost |
|---|---|---|---|
| A: Blank slate | Current: learn from zero, no priors | Clean, interpretable learning | Fragile in early generations; slow credit assignment |
| B: Seeded instincts (this plan) | Seeded danger + food priors, heritable strength | Faster early survival; aligns with biology | Risk: instincts could suppress learning if mistuned |
| C: Seeded + default-ON | If B passes, flip the default | Stable long-term learner | Must verify no regression in later generations |

## Measured evidence

A/B run on [X] generations, [Y] population size, seed [Z]:

### Baseline (innate_instincts_enabled=false)
- Mean ticks-alive: [num]
- Mean death count: [num]
- Mean avoidance-intent fraction: [num]
- Mean food count: [num]

### ON (innate_instincts_enabled=true)
- Mean ticks-alive: [num]
- Mean death count: [num]
- Mean avoidance-intent fraction: [num]
- Mean food count: [num]

### Gate results
- **Survival gate** (ON ≥ baseline × 1.10): [baseline * 1.10 = X], ON = [Y] → [PASS/FAIL]
- **Alignment gate** (avoidance-intent ≥ 0.4): ON = [num] → [PASS/FAIL]
- **Food-per-death gate** (food / death ≥ 2.0): ON = [num] → [PASS/FAIL]

## Why [PASS / REJECT] now

[PASS narrative:] The seeded instincts provide immediate benefit in all three metrics: agents live longer, engage with danger-avoidance more actively, and consume food more efficiently per death. The effect is statistically meaningful and not offset by any observed learning suppression (compared with baseline). Heritable strength genes mean the instincts can evolve toward the population's needs.

[FAIL narrative:] [Diagnosis of which gate failed and why.] The measured benefit was marginal (failure margin: [X]%). Without a clear survival or alignment advantage, seeding the patterns introduces additional complexity without demonstrated payoff. Further iteration required.

## When to revisit

[PASS follow-up:] Flip innate_instincts_enabled=true by default in BrainConfig (new plan) and measure long-term learning under full seeding (e.g., comparison of late-generation genomes between runs, learning rate convergence, final fitness distribution). Revert if later measurements show learning suppression or fitness regression.

[FAIL conditions for reopening:] Revisit this plan only if offline prototyping shows:
1. Alternative instinct signatures (e.g., danger → freeze rather than backward; food → turn-toward) that pass the existing gates, or
2. Different instinct-strength initialization (e.g., weaker default seeds, 0.4 instead of 0.8) that maintain survival/alignment without overconstraining learning, or
3. New measurements of learning speed (time-to-first-food, generations-to-stable-policy) showing the current signatures accelerate learning meaningfully in later stages despite early-stage marginal benefit.
```
4. After writing the decision doc, run the benchmark one final time to confirm the recorded numbers match the code output.

- **Depends on:** implement-ab-validation-harness
- **Done when:** 0013-INNATE-INSTINCT-DECISION.md is written with measured baseline and ON statistics, all three gate evaluations (survival, alignment, food-per-death), a decision (PASS/FAIL), and explicit conditions for revisiting if rejected; the decision is accurate to the measured A/B output and conforms to the template from Plan 0004/0005/0006 precedent; cargo fmt/clippy/test green.

---

**End of plan 0013 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
