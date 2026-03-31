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

### 5. Agent Couldn't See Food

**Symptom:** 600+ generations, no food-seeking behavior.

**Root cause:** Ray marching checked terrain biome colors and other agents, but never food items. Inside a green biome, every direction looked identical. Zero directional signal.

**Fix:** Added food item detection to ray marching via spatial grid lookup.

**Lesson:** Before optimizing a learning algorithm, verify the agent has access to the information it needs to learn from.

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
