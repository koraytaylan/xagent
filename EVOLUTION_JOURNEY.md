# Evolution Journey: What We Learned

A reference document capturing every wrong turn, root cause, and insight from debugging the evolution system. Read this before making changes to evolution, credit assignment, or the brain architecture.

---

## The Core Philosophy

Agents discover everything through experience. No behavior is hardcoded. The only evaluative signal is the homeostatic gradient — whether the agent's internal state (energy, integrity) is improving or worsening. Evolution selects for brain configurations that produce better learners.

## Timeline of Issues Found

### 1. Evolution Never Improved (Gen 0 Mismatch)

**Symptom:** Fitness stuck at gen 0's level across hundreds of generations.

**Root cause:** `spawn_evolution_population()` created `pop_size` unique configs, but `reduce_fitness()` grouped agents by `eval_repeats`, assuming paired configs. Gen 0's mixed groups inflated the best score, creating an unbeatable bar.

**Fix:** Create `unique_count` configs repeated `eval_repeats` times, matching `breed_next_generation()` structure.

**Lesson:** When two functions operate on the same data, they must agree on its structure.

### 2. Winner's Curse in Beat-the-Parent

**Symptom:** Each successful generation set an unreproducible bar. 5% success rate.

**Root cause:** The bar (`best_fitness`) was the max of N noisy group averages — systematically inflated by selection bias. Next generation couldn't match a lucky peak.

**Fix:** Use population average for both comparison and bar (same metric, no bias).

**Lesson:** Don't compare a statistic against its own previous maximum. Use a robust estimator.

### 3. Credit Assignment Punished Correct Actions

**Symptom:** Agents turned away from danger but then arced back in.

**Root cause:** `assign_credit` used the *current* homeostatic gradient to judge *past* actions. Due to EMA lag (~25 ticks), turning away from danger was punished because the gradient still reflected past damage.

**Fix:** Record gradient at action time. Credit = improvement since then: `gradient_now - gradient_at_action`.

**Lesson:** Credit for an action must reflect the *change* it caused, not the *state* when credit is computed.

### 4. Death/Respawn Reward Inversion

**Symptom:** Agents deliberately headed toward death zones with low prediction error.

**Root cause:** After death, the agent respawned healthy. The gradient spiked positive. Old actions (from the previous life) were still in the history ring — they received massive *positive* credit from the respawn, canceling ~60% of the death penalty.

**Fix:** Clear action history and reset homeostatic monitor on death.

**Lesson:** Life boundaries are discontinuities. The history from one life must not leak credit into the next.

### 5. Agent Couldn't See Food (The Duplication Bug)

**Symptom:** 600+ generations, no food-seeking behavior. Agents acted completely randomly after 10+ hours of evolution. Flat fitness graph.

**Root cause:** Two separate ray-march functions existed: `march_ray` (checked food, agents, terrain) and `march_ray_positions` (checked agents and terrain only — no food). Both `main.rs` and `headless.rs` used `extract_senses_with_positions`, which called the broken function. Agents were completely blind to food during all actual simulation runs. Credit assignment reinforced noise because there was no distinguishable visual state before eating food.

**Fix:** Unified both functions into `march_ray_unified` with an `AgentSlice` enum that dispatches agent-hit checks. Food detection (via spatial grid lookup, lime green `[0.70, 0.95, 0.20]`) is now in the single code path. Added integration tests that verify both extraction paths see food.

**Lesson:** Code duplication is a bug factory. When two functions do the same thing, they will eventually diverge. The fix isn't adding the missing code to both — it's eliminating the duplication. Also: before optimizing a learning algorithm, verify the agent has access to the information it needs to learn from.

### 6. Encoder Destroyed Spatial Information

**Symptom:** Agent saw food but couldn't learn directional approach.

**Root cause:** Encoder pooled all 48 pixels into color-averaged bins, losing which column contained food. "Food left" and "food right" produced identical encodings.

**Fix:** Per-pixel RGBD encoding preserving spatial order (row-major).

**Lesson:** Compression that serves memory/prediction (generalization) may destroy specificity needed for action selection.

### 7. Inherited Weights Decayed to Zero

**Symptom:** Weight inheritance didn't accumulate across generations.

**Root cause:** L2 regularization on encoder: 92% loss per generation. Weight decay on actions: 39% loss per generation. Global action values: 100% loss (EMA retention 0.995^50000 ≈ 0).

**Fix:** Deterministic seeded encoder (no adaptation needed). Removed weight decay from actions (MAX_WEIGHT_NORM caps suffice).

**Lesson:** Mechanisms that prevent explosion during learning can destroy inherited signal. Inheritance requires different decay assumptions than within-lifetime learning.

### 8. Learning Signal Too Weak

**Symptom:** 600+ generations, weights barely changed.

**Root cause:** A food event (+20 energy) produced raw gradient 0.12. After EMA smoothing: 0.04 (67% lost). With WEIGHT_LR 0.02: Δw ≈ 0.0003/dim. Needed ~133 generations to build a preference of 1.0.

**Fix:** Pass raw gradient (not EMA composite) to credit assignment. Increase WEIGHT_LR to 0.10. EMA composite still used for exploration rate and urgency (where smoothing is appropriate).

**Lesson:** The homeostatic EMAs serve urgency/exploration. Credit assignment needs the immediate per-tick signal.

### 9. Agent Couldn't Feel Hunger

**Symptom:** Agents avoided danger but never sought food.

**Root cause:** Energy depletion (-0.006/tick raw gradient) was below credit deadzone (0.01) at ALL energy levels. The agent had no concept of hunger.

**Fix:** Amplify raw gradient by urgency: `raw_gradient *= (1 + urgency)`. At 50% energy, gradient becomes -0.021 (above deadzone). Eating at 20% energy produces +0.888 relief.

**Lesson:** A constant slow drain isn't pain. Pain must intensify nonlinearly as the situation worsens.

### 10. 32-Dim Encoding Bottleneck

**Symptom:** Agent saw food, felt hunger, but couldn't learn directional approach.

**Root cause:** 201→32 compression via random projection. Cosine similarity between "food-left" and "food-right" was 0.98. Credit spread uniformly — couldn't distinguish directions.

**Fix:** Action selector moved to raw 201-dim features. Memory/prediction kept 32-dim encoding.

**Lesson (partial):** This was the right diagnosis but wrong fix — see "The Disconnect" below.

### 11. Mutants Could Never Compete

**Symptom:** Evolution useless — champion always won because mutants started from scratch.

**Fix:** All agents inherit action weights (raw policy is config-independent).

**Lesson:** If one agent has accumulated knowledge and another starts blank, selection can't discover anything.

### 12. Brain Parameters Irrelevant

**Symptom:** memory_capacity 24→512, learning_rate 0.01→0.1 — identical fitness.

**Root cause:** The 5 evolved parameters (memory_capacity, processing_slots, representation_dim, learning_rate, decay_rate) control the encoder/memory/predictor stack. The action selector bypasses all of these, operating on raw features with its own fixed WEIGHT_LR. Evolution was tuning disconnected knobs.

**Lesson:** Evolution must mutate things that affect fitness. If the genome is disconnected from the phenotype, selection has no signal.

### 13. Agents Spawned in Death Zones

**Symptom:** 30-40 deaths per generation, mostly from hazards.

**Root cause:** Random spawn placed ~25% of agents in danger biomes. Respawn at 70% integrity gave only 70 ticks to escape. Death spiral: die → respawn in danger → die.

**Fix:** Safe spawn positions (reject danger biome), full integrity on respawn.

**Lesson:** Don't spawn agents in immediately fatal situations. Organisms don't gestate in lethal environments.

### 14. Mutations Were Random Walks

**Symptom:** Evolution progressed slowly. Each mutation was a coin flip — no learning from what worked before.

**Root cause:** `mutate_config_with_strength` applied uniform random perturbation (±strength%) to every parameter independently. The mutation tracking data (`mutation_log` table) was collected but never used. Evolution was a random walk filtered by selection, with no directional intelligence.

**Fix:** Per-parameter momentum vectors (one set per island). After each generation, offspring that beat their parent contribute their mutation deltas to an exponentially-decaying momentum. Future perturbations are biased toward winning directions. This provides: (A) directional bias — parameters trend toward values that improve fitness, (B) emergent correlated mutations — parameters that consistently move together develop aligned momentum, (C) selective focus — parameters with strong signal get larger perturbations while stagnant ones stay near random noise.

**Lesson:** Evolution with memory is faster than evolution without. The same principle that makes gradient descent faster than random search applies to neuroevolution — you don't need exact gradients, just a noisy directional signal accumulated over time.

### 15. `representation_dim` Mutation Destroyed Weight Inheritance

**Symptom:** Offspring frequently started from random initialization despite weight inheritance code being present.

**Root cause:** `representation_dim` was evolvable via both mutation and crossover. When a child's `representation_dim` differed from the parent's, all three weight imports — encoder (`repr_dim × feature_count`), action selector (`repr_dim × 2 + 2`), and predictor (`repr_dim × repr_dim`) — were silently skipped because the dimension check in `import_weights` failed. The child started from a fresh random brain every time this happened.

**Fix:** Locked `representation_dim` — removed it from both `mutate_config_with_strength` and `crossover_config`. The field remains in `BrainConfig` for population-level configuration but never changes across generations.

**Lesson:** Silent failures are the worst kind. The `import_weights` guard that skips mismatched dimensions is correct safety behavior, but the upstream code that freely mutated the dimension was silently defeating the entire inheritance mechanism. When a safety guard fires, it should log — or better, the condition should be prevented upstream.

### 16. Touch Contacts Computed But Never Encoded

**Symptom:** Agents had no tactile sense despite touch infrastructure being fully wired on the sensing side.

**Root cause:** `detect_touch()` and `detect_touch_positions()` populated `frame.touch_contacts` every tick (food proximity, terrain edges, hazard zones, other agents). But `extract_features_into()` in the encoder never read `touch_contacts` — the feature vector contained only visual and proprioceptive/interoceptive features (9 non-visual features). Touch data was computed and discarded.

**Fix:** Encoder now encodes the top 4 touch contacts (by intensity) into 16 additional features: direction.x, direction.z, intensity, and normalized surface_tag for each. `NON_VISUAL_FEATURES` increased from 9 to 25.

**Lesson:** An untested data path is a broken data path. The sensing side was correct; the encoding side never consumed it. Integration tests that verify end-to-end data flow (sensor → encoder → feature vector) catch this class of bug.

### Issue #17: Evolution Tab UI Overhaul

**What changed:** Five improvements to the evolution tab:
1. Fitness chart now shows per-island lines with distinct colors. Each island's evolutionary trajectory is visible independently.
2. Evolution tree is now a left-side pane (file explorer style). Clicking a generation node displays its details (fitness, mutations, config) in the main panel.
3. Dead branches (failed/exhausted) are collapsed by default but can be manually expanded to inspect what happened.
4. Generation progress bar moved to the top toolbar for persistent visibility.
5. Removed the "Pop | Elite | Patience" status line (redundant information).

**Root cause:** The previous single-line fitness chart made it impossible to distinguish island performance. The tree was embedded in a vertical scroll with no way to inspect node details. Dead branches were permanently locked, preventing post-mortem analysis.

**Technical details:**
- Added `island_id INTEGER` column to the `node` table with backwards-compatible migration
- `breed_next_generation` now stores `active_island` on each new node
- New `fitness_history_by_island()` query groups data by island
- `EvolutionSnapshot.fitness_history` changed from `Vec<(u32, f32, f32)>` to `HashMap<i64, Vec<(u32, f32, f32)>>`
- `TreeNode` gained `island_id: Option<i64>` field
- Tree nodes are now clickable (`egui::Sense::click()`) with a `selected_node_id` state
- `render_running_dashboard` uses `ui.columns(2, ...)` for the horizontal split layout
- Force-collapse (`header.open(Some(false))`) removed from dead-end branches
- Progress bar renders in `TopBottomPanel::top("top_bar")` using captured `gen_tick`/`tick_budget`

### Issue #18: GPU Mega-Kernel — 100× Throughput

**What changed:** Replaced the per-tick 7-pass dispatch loop with a fused mega-kernel. One dispatch per `vision_stride` cycles (default 10), 256 threads per agent workgroup, all per-agent computation (physics, food detection, death/respawn, and all 7 brain passes) in a single shader.

**Why it matters for evolution:** At ~600 ticks/second (old dispatch model), a 50,000-tick generation takes ~83 seconds wall time. At 60,000+ ticks/second (mega-kernel), the same generation takes < 1 second. This makes multi-island evolution with `eval_repeats > 1` practical — running 3 islands × 10 agents × 2 repeats previously took hours; now it takes minutes.

**Technical details:**
- `mega_tick.wgsl`: Fused per-agent kernel with cycle loop. Per cycle: physics (thread 0), brute-force food detection (all 256 threads, two-phase shared-memory reduction into 128-element arrays), death/respawn (thread 0), full brain pipeline (all 256 threads)
- `global_tick.wgsl`: Grid rebuild + collision (dispatched as 1,1,1), runs once per vision_stride cycles
- `gpu_mega_kernel.rs`: Shader composition from common + brain function bodies (entry point stripped) + mega_tick. Subgroup intrinsic support with marker-based substitution
- Async double-buffered state readback with backpressure throttle
- Per-selected-agent telemetry readback (`read_agent_telemetry`) for UI: vision, motor, habituation, curiosity, fatigue, urgency, gradient

**Key bugs encountered:**
- `queue.write_buffer` race: Multiple writes to the same uniform buffer in one encoder — only the last survives at submit. Fixed by one encoder+submit per mega-batch.
- Subgroup marker mismatch: Brain's `/* SUBGROUP_TOPK_PARAMS */` vs mega's `/* MEGA_SUBGROUP_TOPK_PARAMS */` — both need replacement when composing the fused shader.
- 256-thread arrays with 128-element shared memory: Two-phase reduction (first 128 write directly, barrier, second 128 merge) resolves the bounds mismatch.

**Lesson:** CPU↔GPU coordination overhead dominates at high tick rates. The simulation math was already fast — the bottleneck was 11–19 dispatches per tick, per-tick uniform writes, and blocking readback. Fusing passes and batching cycles eliminated the coordination cost. Also: when composing shaders from fragments, marker-based string replacement is fragile — every code path that uses the markers must be tested with subgroups both enabled and disabled.

---

## The Disconnect (Current State)

After all fixes, the architecture has a fundamental structural problem:

```
EVOLUTION mutates → BrainConfig (memory, slots, dim, lr, decay)
                        ↓
                  Encoder / Memory / Predictor (32-dim compressed space)
                        ↓
                  Prospective evaluation (weak bridge)

ACTION SELECTOR uses → Raw features (201-dim)
                        ↓
                  Credit assignment (raw gradient × raw features)
                        ↓
                  BEHAVIOR (the only thing fitness measures)
```

The action selector — which determines ALL behavior — is disconnected from the brain's encoding/memory/prediction stack. Evolution tunes the stack; behavior comes from the selector. These are two separate systems that happen to live in the same struct.

## The Path Forward (A, B, C)

### A. Reunify the Brain

The action selector must work in the SAME representational space as memory and prediction. This means the encoder must be TRAINABLE — learning to produce representations that are useful for action selection, not just a frozen random projection.

When credit flows to the action selector ("turn-left was good in this state"), it should also update the encoder ("the features that led to choosing turn-left should be emphasized"). This connects the entire brain into one coherent learning system.

With a trainable encoder: food-left and food-right project to different encoded states. The predictor predicts meaningful transitions. Memory stores behaviorally relevant patterns. Evolution can influence behavior by tuning how fast the encoder adapts (learning_rate), how many patterns it can store (memory_capacity), and how rich the representation is (representation_dim).

### B. Neuroevolution of Action Weights

Evolution should directly explore the behavioral space by mutating the inherited action weights. Mutants get the parent's weights plus small perturbations (e.g., 10% of weights perturbed by ±5%). This lets evolution discover behavioral variations that within-lifetime learning might miss.

### C. Continuous Motor Output

Replace 8 discrete actions with continuous motor output: `forward = tanh(dot(fw_weights, state))`, `turn = tanh(dot(turn_weights, state))`. Exploration becomes additive Gaussian noise, not full-action replacement. Reduces policy from 1,608 weights to ~64-100 weights (2 outputs × repr_dim). Smoother movement, easier to learn, more natural behavior.

---

## Rules for Future Changes

1. **Verify the information path before optimizing the algorithm.** Can the agent see/sense what it needs? (Issue #5)
2. **Check that evolved parameters actually affect fitness.** Run with min and max configs. If fitness is identical, evolution is disconnected. (Issue #12)
3. **Credit assignment must use immediate signals, not smoothed ones.** EMAs are for urgency, not for credit. (Issue #3, #8)
4. **Life boundaries are hard discontinuities.** Never let credit/gradient leak across death/respawn. (Issue #4)
5. **Inheritance and decay are enemies.** If you add decay, verify inherited weights survive it. (Issue #7)
6. **The encoding, prediction, and action spaces must be the same.** Disconnected spaces create disconnected systems. (Issue #10, #12)
7. **Test with actual data, not intuition.** Run headless, query the DB, compute the math. Every "should work" assumption in this journey was wrong until measured. (All issues)
