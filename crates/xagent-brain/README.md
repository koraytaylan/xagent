# xagent-brain

A general-purpose cognitive architecture based on **predictive processing**.

The brain crate is the decision-making core of each xagent. It has no hardcoded behaviors — no "hunger module", no "fear module", no goal system. Everything the agent does emerges from a single loop and a single principle:

> **Prediction error drives everything.**

```
sense → encode → recall → predict → learn → act → (repeat)
```

The brain receives a `SensoryFrame` and emits a `MotorCommand` (movement + discrete actions). Between those two endpoints, the encoder flattens all sensory data — regardless of modality — into an opaque numerical vector (`EncodedState`). The brain then recalls relevant past experiences, predicts what will happen next, measures how wrong it was, and uses that error signal to update every component. It never sees field names, struct types, or sensory modality labels. Capacity constraints force it to prioritize, giving rise to attention. Homeostatic feedback (numerical signals the brain must learn to interpret) is the only evaluative signal, giving rise to self-preserving behavior. There are no rewards, no utility functions, no explicit goals.

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Component Deep Dive](#4-component-deep-dive)
   - [4.1 Sensory Encoder](#41-sensory-encoder-encoderrs)
   - [4.2 Pattern Memory](#42-pattern-memory-memoryrs)
   - [4.3 Prediction Engine](#43-prediction-engine-predictorrs)
   - [4.4 Action Selector](#44-action-selector-actionrs)
   - [4.5 Homeostatic Monitor](#45-homeostatic-monitor-homeostasisrs)
   - [4.6 Sensory Habituation](#46-sensory-habituation-habituationrs)
   - [4.7 Motor Fatigue](#47-motor-fatigue-fatiguers)
   - [4.8 Capacity Manager](#48-capacity-manager-capacityrs)
   - [4.9 Brain Orchestrator](#49-brain-orchestrator-brainrs)
5. [Emergent Phenomena](#5-emergent-phenomena)
6. [Configuration (BrainConfig)](#6-configuration-brainconfig)
7. [Testing](#7-testing)
8. [Design Decisions](#8-design-decisions)
9. [CPU Performance Optimizations](#9-cpu-performance-optimizations)
10. [Known Limitations & Future Work](#10-known-limitations--future-work)

---

## The Brain Has No Eyes

The most important thing to understand about this architecture: **the brain has zero semantic knowledge of its inputs**.

The `SensoryFrame` that arrives from the sandbox has named fields — `vision`, `touch_contacts`, `energy_signal`. But the brain never sees those names. The `SensoryEncoder` flattens *everything* into a single `Vec<f32>` (the `EncodedState`), and from that point on, the brain operates on an opaque numerical vector. It has no concept of "vision," no awareness that it has "eyes," no understanding that index 47 was once an RGBA pixel and index 73 was once an energy level.

```
World → SensoryFrame → Encoder → [0.31, 0.72, 0.08, ...] → Brain
              ↑                          ↑
     Named fields like           Brain sees only a
     "vision", "energy"          flat Vec<f32>
```

Consider what happens when another agent — say, a magenta-colored one — enters the visual field. The brain doesn't receive "agent detected" or "entity of type Agent at bearing 30°." It experiences indices 12–15 shifting from `[0.3, 0.6, 0.2, 1.0]` to `[0.9, 0.2, 0.6, 1.0]`. Simultaneously, a touch contact might add nonzero values at indices 78–80 (direction) and 81 (intensity with `surface_tag=4`). The brain has no legend for any of this. It doesn't know that `surface_tag=4` means "agent." It doesn't know that the shifted values represent magenta. Over hundreds of ticks, if this pattern of input correlates with energy dropping (food competition), the brain discovers — through prediction error and homeostatic gradient alone — that "those numerical patterns are bad for me." The concept of "that's a competitor" *emerges* from experience, not from labels.

This is the fundamental difference from traditional AI systems. There are no reward functions hand-crafted by engineers. No labeled feature vectors telling the model "this is vision, this is hunger." No hardcoded categories like "food," "hazard," or "friend." The `SensoryFrame` struct with its named fields is engineering scaffolding — it's the "body's" wiring that collects data from the simulated world. The encoder strips all that structure away. What remains is prediction + homeostatic gradient + experience, and from these three ingredients, all meaning is discovered.

---

## 1. Theoretical Foundation

### Predictive Processing & Active Inference

The brain crate implements a simplified version of the **predictive processing** framework from computational neuroscience. The core idea, developed by Karl Friston (free energy principle) and echoed in Jeff Hawkins' work on hierarchical temporal memory, is that brains are fundamentally *prediction machines*:

- The brain constantly generates predictions about what sensory input it will receive next.
- When reality differs from the prediction, the resulting **prediction error** is the signal that drives all learning and adaptation.
- The brain's overarching goal is to minimize prediction error — either by updating its internal model (learning) or by acting on the world to make the prediction come true (active inference).

### Prediction Error as Universal Currency

In this crate, prediction error is not just one signal among many — it is the *only* learning signal. It:

- **Modulates learning rates**: higher error → faster weight updates in encoder, predictor, and memory
- **Drives exploration**: high error signals novelty → the action selector increases exploration rate
- **Allocates capacity**: the capacity manager gives more recall budget when error is high
- **Reinforces memory**: patterns that co-occur with low prediction error get strengthened

There is no separate reward signal. There is no loss function designed by a human. The agent learns because its predictions are wrong, and prediction error is metabolically expensive.

### Homeostatic Feedback as the Only Evaluative Signal

The brain has no concept of "good" or "bad" built in. Instead, the `HomeostaticMonitor` tracks whether internal variables (energy, physical integrity) are trending toward or away from stability. This gradient — positive means improving, negative means worsening — modulates:

- **Learning rate**: the brain learns faster when homeostatic state is changing rapidly (either direction)
- **Action selection**: negative gradient gives credit/blame to recent actions; positive gradient reinforces them
- **Urgency**: when energy or integrity drops critically low, urgency suppresses exploration in favor of exploitation

This is analogous to how biological organisms don't have explicit goals — they have homeostatic set points, and deviations from those set points drive behavior.

### Capacity Constraints → Emergent Cognition

The brain has finite resources:
- A fixed-size memory (`memory_capacity` patterns)
- A per-tick processing budget (`processing_slots` recall operations)
- A fixed-dimension representation space (`representation_dim`)

These constraints aren't limitations to be engineered around — they are **generative**. Because the brain can't attend to everything, it must select. Because memory is finite, it must forget. Because the representation is compressed, it must abstract. These constraints give rise to attention, habit formation, chunking, and other cognitive phenomena without any of them being explicitly programmed.

---

## 2. Architecture Overview

```
Brain
├── SensoryEncoder    — Compresses raw sensory input into fixed-size representation
├── PatternMemory     — Stores, recalls, and associates encoded patterns
├── Predictor         — Predicts next state; generates prediction error
├── ActionSelector    — Chooses motor commands based on context and outcomes
├── HomeostaticMonitor — Tracks internal stability gradients and urgency
└── CapacityManager   — Allocates processing budgets adaptively
```

All components are orchestrated by `Brain::tick()`, which runs once per simulation step. Components communicate through the encoded state representation (`EncodedState`) and scalar signals (prediction error, homeostatic gradient, urgency).

---

## 3. Data Flow Diagram

```
                         ┌──────────────────────────────────────────────────────────────┐
                         │                        Brain::tick()                         │
                         └──────────────────────────────────────────────────────────────┘

  SensoryFrame                                                               MotorCommand
  (structured data                                                           (forward, strafe,
   from the body —                                                            turn, action)
   brain never sees                                                                ▲
   these field names)                                                              │
       ▼                                                                           │
  ┌─────────────┐    EncodedState     ┌──────────────┐                    ┌────────────────┐
  │   Sensory   │───────────────────▶│   Pattern     │◀──recall_budget────│   Capacity     │
  │   Encoder   │        │           │   Memory      │                    │   Manager      │
  └─────────────┘        │           └──────┬────────┘                    └───────┬────────┘
       ▲                 │                  │                                     ▲
       │                 │          recalled patterns                             │
  adapt(error, lr)       │                  │                              prediction_error
       │                 │                  ▼                                     │
       │                 │           ┌─────────────┐                              │
       │                 └──────────▶│  Predictor  │────prediction_error──────────┘
       │                             └──────┬──────┘
       │                                    │
       │                         prediction + error
       │                                    │
       │                                    ▼
       │                            ┌──────────────┐     ┌────────────────────┐
       │                            │   Action     │◀────│   Homeostatic      │
       │                            │   Selector   │     │   Monitor          │
       │                            └──────────────┘     └────────────────────┘
       │                                    │                     ▲
       │                             MotorCommand                 │
       │                                                   energy_signal,
       │                                                   integrity_signal
       └──────────────────────────────────────────────────────────┘
                          Learning signals flow back

  Legend:
  ────▶  Data flow (per tick)
  ◀────  Budget / modulation signal

  Note: Between Encoder and downstream consumers (Memory, Predictor, ActionSelector),
  SensoryHabituation attenuates the encoded state and produces a curiosity_bonus.
  After ActionSelector produces a MotorCommand, MotorFatigue dampens the output
  based on recent motor variance.
```

### Tick Execution Order (14 steps)

```
 1.  Encode sensory input           → EncodedState (opaque Vec<f32>)
 1a. Apply sensory habituation      → habituated state, curiosity_bonus
 2.  Update homeostasis             → HomeostaticState (gradient, urgency)
 3.  Compute prediction error       → scalar_error, error_vec
     ├─ Learn: memory reinforcement
     ├─ Learn: predictor weights (gradient descent)
     └─ Learn: encoder weights (L2 regularization)
 4.  Allocate recall budget         → (recall_budget, surprise_budget)
 5.  Recall similar patterns        → Vec<EncodedState>
 6.  Predict next state             → prediction
 7.  Store current pattern in memory
 8.  Decay old patterns
 9.  Select action (using habituated state and curiosity_bonus) → MotorCommand
 9a. Apply motor fatigue            → dampened MotorCommand
10.  Record prediction for next tick
11.  Compute behavior quality metrics
12.  Update telemetry snapshot
```

---

## 4. Component Deep Dive

### 4.1 Sensory Encoder (`encoder.rs`)

**What it does**: Compresses a raw `SensoryFrame` (variable-size visual data + proprioceptive + interoceptive signals) into a fixed-size `EncodedState` vector of `representation_dim` floats. This is the **semantic firewall** — everything upstream has named fields and modality structure; everything downstream is an opaque `Vec<f32>`. The brain never sees the original field names, modality boundaries, or data types.

**Why it exists**: The brain needs a uniform representation to work with. Raw sensory data is high-dimensional (a 64×48 visual field = 12,288 RGBA values alone). The encoder acts as the bottleneck that forces information compression — what gets through this bottleneck is what the brain "pays attention to".

**How it works**:

1. **Feature extraction** (`extract_features`):
   - **Visual features**: Per-channel spatial pooling divides the visual field into `visual_encoding_size` spatial bins. Each bin produces 3 features (R, G, B averages), preserving color information. This is critical — danger zones (reddish) and food zones (greenish) are visually distinct, and the brain needs separate color channels to distinguish them. Alpha is dropped (carries no information).
   - **Proprioceptive features** (5): velocity magnitude, facing direction (x, y, z), angular velocity
   - **Interoceptive features** (4): energy level, integrity level, energy delta, integrity delta
   - Total input dimensionality = `visual_encoding_size × 3 + 9`

2. **Projection**: The feature vector is multiplied by a weight matrix `[feature_count × representation_dim]` with bias terms, then passed through `tanh()` to produce the output:

   ```rust
   for i in 0..self.representation_dim {
       let mut sum = self.biases[i];
       for (j, &feat) in features.iter().enumerate() {
           sum += feat * self.weights[j * self.representation_dim + i];
       }
       data[i] = sum.tanh();
   }
   ```

3. **Weight initialization**: Xavier/Glorot uniform — `uniform(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`. This prevents saturation of tanh activations at initialization.

4. **Adaptation** (`adapt`): Uses L2 regularization to keep weights bounded. This is a simple but effective approach — rather than backpropagating through the predictor (which would be too indirect given the architecture), weights are gently pulled toward zero:

   ```rust
   *w *= 1.0 - (learning_rate * L2_REGULARIZATION_FACTOR);  // 0.001
   *w = w.clamp(-WEIGHT_CLAMP_RANGE, WEIGHT_CLAMP_RANGE);   // ±2.0
   ```

**Key constants**:

| Constant | Value | Effect |
|----------|-------|--------|
| `NON_VISUAL_FEATURES` | 9 | Fixed proprioceptive + interoceptive feature count |
| `WEIGHT_CLAMP_RANGE` | 2.0 | Prevents weight explosion |
| `L2_REGULARIZATION_FACTOR` | 0.001 | Gentle regularization strength |

**Emergent properties**: The encoder creates a **selectivity bottleneck**. With `visual_encoding_size=64` and `representation_dim=32`, a 16×12 RGBA image (768 values) becomes 192 per-channel features + 9 non-visual = 201 inputs, compressed to 32 floats. The per-channel pooling ensures the brain can learn color-based associations: "red ahead → danger" vs "green ahead → food."

**Performance**: The encoder uses a pre-allocated **scratch buffer** for feature extraction — no heap allocations occur per tick. The `tanh()` activation is replaced with a **fast Padé approximant** (`fast_tanh`) that avoids the cost of the standard library's transcendental function while maintaining sufficient accuracy for the brain's purposes.

---

### 4.2 Pattern Memory (`memory.rs`)

**What it does**: Stores encoded states as `Pattern` objects in a fixed-capacity bank. Patterns can be recalled by similarity, associated with each other through co-occurrence, linked temporally (predecessor/successor), and decay over time unless reinforced.

**Why it exists**: This is the agent's experience store — analogous to episodic memory. It allows the agent to recognize familiar situations, anticipate what comes next (via temporal links), and build associations between co-occurring experiences. The fixed capacity means old, unused memories are eventually overwritten by new ones, implementing a form of forgetting.

**How it works**:

#### Storage

When `store()` is called:
1. A slot is found — either an empty one, or the one with the lowest reinforcement (weakest memory gets overwritten).
2. If overwriting, temporal links pointing to the old pattern are cleaned up (`unlink_temporal`).
3. The new pattern is created with `reinforcement=1.0`, `activation_count=1`, and the current tick.
4. Temporal links are established: the previous pattern's `successor` points here, and this pattern's `predecessor` points back. Both forward (0.5 strength) and backward (0.3 strength) associations are created.

#### Recall by Similarity

`recall()` computes cosine similarity between the query and all stored patterns, sorts by similarity, and returns the top-N (capped by `budget`). Each recalled pattern gets its `last_accessed` tick updated and `activation_count` incremented.

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < 1e-8 || mag_b < 1e-8 { return 0.0; }
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}
```

There is also `recall_weighted()` which returns `(EncodedState, similarity_score)` pairs for use by the predictor's context-weighted blending.

#### Association Links with Generation Validation

Each pattern can have associations to other patterns. An `AssociationLink` contains:
- `target_idx`: the slot index of the associated pattern
- `target_generation`: the generation of the target when the link was created
- `strength`: how strong the association is (capped at `MAX_ASSOCIATION_STRENGTH = 5.0`)

Generation validation solves the **stale link problem**: when a pattern at slot `i` is overwritten, its generation counter increments. Any association links pointing to slot `i` with the old generation are recognized as stale and ignored during `retrieve_associated()`. This avoids the need for expensive cleanup of all links across the memory bank every time a slot is overwritten.

#### Temporal Sequence Tracking

Each pattern stores `predecessor` and `successor` indices, forming a linked list of temporal experience. When pattern B is stored after pattern A, A's `successor` becomes B and B's `predecessor` becomes A. This allows the brain to "replay" sequences and predict what comes next.

#### Association Chain Retrieval

`retrieve_associated()` performs a BFS traversal through association links, prioritizing strongest links first. At each hop, strength decays by 0.5×, so distant associations contribute less. Only generation-validated links are followed.

#### Smart Decay

The `decay()` method doesn't remove patterns at a flat rate. Instead, each pattern's effective decay rate is modulated by:
- **Frequency**: frequently accessed patterns decay slower — `frequency_factor = 1 / (1 + activation_count * 0.2)`
- **Recency**: recently accessed patterns decay slower — `recency_factor = min(recency / 100, 3.0)`
- **Combined**: `effective_rate = base_rate × frequency_factor × (0.2 + recency_factor)`

Patterns whose reinforcement drops to zero are removed (set to `None`).

#### Trauma

The `trauma(severity)` method models the cognitive cost of catastrophic events (e.g., death). It applies a bulk reinforcement decay across all stored patterns:

```rust
pub fn trauma(&mut self, severity: f32) {
    // severity typically 0.2 = 20% reinforcement decay
    for pattern in &mut self.patterns {
        if let Some(p) = pattern {
            p.reinforcement *= 1.0 - severity;
            // Patterns that drop below threshold are removed
        }
    }
}
```

With a severity of 0.2 (used on agent death), the weakest memories are wiped while the strongest survive — a single trauma event removes approximately the bottom 20% of memories by reinforcement. This models the cognitive cost of catastrophic discontinuity: death damages memory, but deeply learned patterns persist.

**Associations are capped** during recall to `.take(8)` for performance — only the 8 strongest associations per pattern are followed during retrieval.

#### Learning

`learn()` does two things:
1. **Reinforcement**: Patterns similar to the current state (cosine similarity > 0.3) are reinforced proportional to `similarity × learning_rate × (1 - error)`. Low prediction error strengthens matching patterns more.
2. **Co-occurrence association**: Patterns that are *both* highly similar to the current state (similarity > 0.5) are associated with each other via bidirectional links with strength `learning_rate × 0.1`.

**Key constants**:

| Constant | Value | Effect |
|----------|-------|--------|
| `MAX_ASSOCIATION_STRENGTH` | 5.0 | Cap on association link strength |
| `MAX_REINFORCEMENT` | 10.0 | Cap on pattern reinforcement |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum similarity for reinforcement |
| `HIGH_SIMILARITY_THRESHOLD` | 0.5 | Minimum similarity for co-occurrence association |
| `FORWARD_ASSOCIATION_DEFAULT` | 0.5 | Initial temporal link strength (pred → succ) |
| `BACKWARD_ASSOCIATION_DEFAULT` | 0.3 | Initial temporal link strength (succ → pred) |

**Emergent properties**: Fixed capacity creates **memory competition** — important memories (frequently accessed, recently relevant) survive while unimportant ones are forgotten. Temporal links create **expectation chains** — the agent "knows" what typically comes after a given experience. Co-occurrence associations create **contextual binding** — experiences that happen together become linked, forming proto-concepts.

---

### 4.3 Prediction Engine (`predictor.rs`)

**What it does**: Predicts the next encoded state from the current state and recalled context patterns. Computes the prediction error that drives the entire learning system.

**Why it exists**: Prediction error is the universal currency. Without the predictor, there is no error signal, and therefore no learning, no exploration modulation, no capacity allocation — nothing works. The predictor is the engine that converts experience into expectation.

**How it works**:

#### Prediction

The predictor applies a learned linear transform to the current state, then blends in context from recalled patterns:

```rust
// 1. Linear transform: predicted = current × weights
for i in 0..self.dim {
    let mut sum = 0.0;
    for j in 0..self.dim {
        sum += current.data[j] * self.weights[j * self.dim + i];
    }
    predicted[i] = sum;
}

// 2. Blend recalled context (similarity-weighted)
if !recalled.is_empty() {
    let total_sim: f32 = recalled.iter().map(|(_, s)| s.max(0.0)).sum();
    if total_sim > 1e-8 {
        for (state, sim) in recalled {
            let w = self.context_weight * sim.max(0.0) / total_sim;
            for i in 0..self.dim {
                predicted[i] += state.data[i] * w;
            }
        }
    }
}

// 3. Nonlinearity
for val in &mut predicted {
    *val = val.tanh();
}
```

The `context_weight` parameter controls how much influence recalled patterns have on the prediction. It is itself learned — increasing when prediction error is high (recalled context might help) and decreasing when error is low (the linear transform alone is sufficient).

#### Weight Initialization

The weight matrix starts as a **near-identity** with small off-diagonal noise:
- Diagonal: 0.9 (predict next state ≈ current state)
- Off-diagonal: uniform random in [-0.01, 0.01]

This is a strong inductive bias: the default prediction is "things stay roughly the same". The predictor then learns deviations from this assumption.

#### Online Gradient Descent

After each tick, the predictor updates its weights using the error vector (predicted − actual):

```rust
for i in 0..dim {
    let tanh_deriv = 1.0 - predicted[i].powi(2);  // tanh'(x) = 1 - tanh(x)²
    for j in 0..dim {
        let grad = (error[i] * tanh_deriv * input[j])
            .clamp(-GRADIENT_CLAMP, GRADIENT_CLAMP);     // ±1.0
        self.weights[j * dim + i] -= learning_rate * grad;
        self.weights[j * dim + i] = self.weights[j * dim + i]
            .clamp(-WEIGHT_CLAMP_RANGE, WEIGHT_CLAMP_RANGE); // ±3.0
    }
}
```

The tanh derivative is applied correctly — since the output passes through `tanh()`, the gradient must account for this nonlinearity. Gradient clipping at ±1.0 prevents instability.

The context weight is also adapted:
```rust
self.context_weight += learning_rate * CONTEXT_WEIGHT_LR * (error_mag - 0.5);
```
If error magnitude > 0.5, context weight increases (the predictor needs more help from memory). If < 0.5, it decreases.

#### Error Metrics

- **Scalar error**: RMSE of the prediction error vector — `sqrt(mean((predicted - actual)²))`
- **Error history**: A 128-entry ring buffer stores recent scalar errors for computing moving averages via `recent_avg_error(window)`.

**Key constants**:

| Constant | Value | Effect |
|----------|-------|--------|
| `ERROR_HISTORY_LEN` | 128 | Ring buffer size for error tracking |
| `GRADIENT_CLAMP` | 1.0 | Maximum gradient magnitude per update |
| `WEIGHT_CLAMP_RANGE` | 3.0 | Maximum absolute weight value |
| `CONTEXT_WEIGHT_LR` | 0.01 | Learning rate for context weight adaptation |
| `CONTEXT_WEIGHT_MIN` | 0.05 | Minimum context blending weight |
| `CONTEXT_WEIGHT_MAX` | 0.5 | Maximum context blending weight |

**Emergent properties**: The predictor creates the **surprise signal** that powers the entire system. When the agent enters a novel environment, prediction errors spike, causing increased exploration, faster learning, and expanded recall. As the agent learns the regularities of its environment, errors decrease, exploration drops, and behavior stabilizes — the agent has *adapted*.

**Performance**: The predictor uses a pre-allocated **scratch buffer** for prediction computation — no heap allocations per tick. The `tanh()` activation uses the same `fast_tanh` Padé approximant as the encoder.

---

### 4.4 Action Selector (`action.rs`)

**What it does**: Chooses one of 8 discrete motor commands each tick, balancing exploration (trying new things) with exploitation (repeating what worked).

**Why it exists**: The brain needs to act, and those actions need to be informed by experience. The action selector learns a **linear policy** that maps encoded sensory features directly to action preferences, letting it discover associations like "when green pixels are ahead, forward is good."

**How it works**:

#### The 8-Action Discrete Space

| Index | Action | MotorCommand |
|-------|--------|--------------|
| 0 | Move forward | `forward: 1.0` |
| 1 | Move backward | `forward: -1.0` |
| 2 | Turn left | `turn: -1.0` |
| 3 | Turn right | `turn: 1.0` |
| 4 | Forward-right curve | `forward: 0.5, turn: 0.3` |
| 5 | Forward-left curve | `forward: 0.5, turn: -0.3` |
| 6 | Jump | `action: Jump` |
| 7 | Forage | `forward: 0.15, action: Consume` |

#### Linear Policy (replaces context-hash table)

Each action has a weight vector of size `repr_dim` (default 32). The preference for an action is computed as the dot product of the weight vector with the current encoded state, plus a global bias:

```
preference[a] = dot(action_weights[a], encoded_state) + global_bias[a]
```

This is fundamentally superior to the old context-hash approach because:
- It uses **all dimensions** of the encoded state, not just 8
- It's **continuous** — similar states produce similar preferences (no hash collisions)
- Each action learns **which sensory features** predict positive outcomes
- The weights learn relationships like "when visual feature X is high, forward is good"
- Memory: `32 × 8 = 256` floats (smaller than the old `128 × 8 = 1024` hash table)

#### Temporal Credit Assignment

When the homeostatic gradient arrives (positive = things improved, negative = things worsened), each recent action's **state snapshot** (the encoded state at the time of action) is used to update the policy weights:

```
Δweights[action] += WEIGHT_LR × credit × state_similarity × state_snapshot_at_action_time
```

The effective credit signal combines two components:

1. **Pain-amplified gradient**: Negative gradients are multiplied by `PAIN_AMPLIFIER` (3.0×), reflecting the biological reality that amygdala neurons respond 2-3× more strongly to aversive stimuli. This is NOT hard-coded avoidance; the brain must learn WHAT to do about the amplified signal.

2. **Death signal**: When the agent dies, `death_signal()` fires a calibrated one-time credit event (`DEATH_CREDIT = -0.5`, which after PAIN_AMPLIFIER becomes -1.5 effective). This retroactively updates **state-dependent weights only** — global action biases are NOT modified. This is critical: without this guard, the death signal's accumulated credit across 64 recent actions creates a catastrophic shift in global bias per death, which punishes whichever action the agent was using regardless of context. By restricting the death signal to state-dependent weights, the agent learns "forward-in-danger is bad" without learning "forward is always bad."

**Credit deadzone** (`CREDIT_DEADZONE = 0.01`): In `assign_credit()`, if the absolute gradient is below 0.01, credit assignment is skipped entirely. This silences the constant metabolic noise — energy depletion produces a gradient of ~0.006/tick, which is below the deadzone threshold. Meaningful signals like food (+0.03), damage (-0.02), and death (-0.5) exceed the threshold and pass through. Without the deadzone, constant metabolic noise triggers credit updates every tick, eroding learned associations with random noise.

**State-conditioned credit**: Credit is modulated by the **cosine similarity** between the current state (when the event occurs) and each recorded state in the action history ring buffer. This means death in a danger zone only penalizes actions taken in danger-similar states; actions taken in safe-state contexts are spared. This is analogous to biological synaptic plasticity being context-dependent — a burn from a stove doesn't make you afraid of chairs.

**Weight normalization** (`MAX_WEIGHT_NORM = 2.0`): After each credit assignment pass, per-action weight vectors are clipped to L2 norm ≤ 2.0. This prevents unbounded weight accumulation from repeated reinforcement and acts as synaptic homeostasis — ensuring no single action's weight vector can dominate the preference space.

The credit decays exponentially with action age. The key insight: because each action stores the state snapshot from when it was chosen (not the current state), the weights learn to associate the **correct sensory context** with the action.

Global action biases are updated via EMA:

```
global_bias[a] = global_bias[a] × 0.995 + accumulated_credit[a] × 0.08
```

The last 64 actions are tracked with zero per-tick heap allocation (pre-allocated ring buffer for state snapshots). The most recent action gets full credit; an action from 30 ticks ago gets `exp(-30 × 0.04) ≈ 30%` of the credit.

#### Adaptive Exploration Rate

The exploration rate is computed dynamically each tick:

```rust
let stability = recalled.len() as f32 / 16.0;
let novelty_bonus = (prediction_error * 2.0).min(0.4);
let urgency_penalty = (urgency * 0.4).min(0.5);
exploration_rate = (0.5 - stability * 0.15 + novelty_bonus + curiosity_bonus - urgency_penalty)
    .clamp(0.10, 0.85);
```

- **Base rate is 0.5** — agent exploits ~50% of the time by default
- **More recalled patterns** (stable environment) → less exploration (stability weight 0.15)
- **Higher prediction error** (novel situation) → more exploration
- **Higher curiosity bonus** (sensory monotony) → more exploration
- **Higher urgency** (danger) → less exploration (exploit known-good actions)
- **Exploration floor is 10%** — even at maximum urgency, agents explore at least 10% of the time, preventing total exploitation lock-in

#### Uniform Random Exploration

When exploring, the agent picks an action uniformly at random (`rng.random_range(0..NUM_ACTIONS)`), giving each action a 12.5% chance. The previous softmax-biased exploration was effectively deterministic: in 32 dimensions with norm-2 weight vectors, dot product differences between actions were large enough that the best action received 99%+ of the softmax probability mass, defeating the purpose of exploration entirely. Uniform random guarantees genuine coverage of the action space.

#### Prospective Evaluation (Model-Based Reasoning)

The action selector applies its learned weights not just to the **current** state but also to the **predicted future** state. Critically, this prediction is not a single-tick lookahead but a **multi-step rollout**: the predictor is iteratively applied up to 10 times to mentally simulate the agent's trajectory ~1/3 second into the future.

The prospection signal is **delta-based**: it measures the *change* in preferences between the current state and the predicted trajectory, not the absolute predicted value. This eliminates a fixed-point convergence problem where long rollouts contract to a constant state that overrides the reactive policy.

```
// In brain.rs — multi-step rollout for prospection
look_ahead = confidence × MAX_LOOK_AHEAD     // scale by prediction accuracy
far_prediction = predictor.rollout(prediction, look_ahead)

// In action.rs — delta-based evaluation
current_prefs[a] = dot(action_weights[a], current_state)
future_prefs[a] = dot(action_weights[a], far_prediction)
delta = future_prefs[a] - current_prefs[a]       // "is my trajectory improving?"
preferences[a] += confidence × ANTICIPATION_WEIGHT × delta
```

**Why delta-based?** Iterative prediction contracts toward a fixed point (only 3% of state info survives 30 steps with near-identity weights). Evaluating the fixed point directly produces a constant bias that overrides the reactive policy — causing behavior to get *worse* over time. The delta approach eliminates this: it measures the *change* along the trajectory, which always depends on the current state and provides a directional "getting better/worse" signal.

The key properties:
- **No hardcoded avoidance**: The same learned weights (trained via credit assignment) are reused for both present and future evaluation. The agent doesn't know what's "dangerous" — it just knows which features correlated with negative outcomes.
- **Directional signal**: Heading toward danger → delta < 0 → avoidance. Heading toward food → delta > 0 → approach. Stable trajectory → delta ≈ 0 → no effect.
- **Self-calibrating horizon**: Look-ahead steps scale with prediction confidence. Low accuracy → short horizon. High accuracy → full foresight.
- **Fixed-point immune**: The delta always depends on the current state, even if the rollout converges to a fixed point.

| Constant | Value | Effect |
|----------|-------|--------|
| `ANTICIPATION_WEIGHT` | 0.5 | How much the predicted trajectory change influences action preferences |
| `MAX_LOOK_AHEAD` | 10 | Maximum rollout steps (~1/3 second of foresight at 30 ticks/sec) |

#### Exploitation

When exploiting, the agent picks the action with the highest preference (dot-product score + global bias). **Random argmax tie-breaking**: `best_action()` collects all actions sharing the maximum preference and picks one randomly. Previously, Rust's `max_by` returned the last tied index (action 7), creating a deterministic bias toward one action when all weights were zero-initialized. With random tie-breaking, a fresh agent distributes equally across all actions.

**Key constants**:

| Constant | Value | Effect |
|----------|-------|--------|
| `NUM_ACTIONS` | 8 | Size of discrete action space |
| `ACTION_HISTORY_LEN` | 64 | Credit assignment lookback window |
| `CREDIT_DECAY_RATE` | 0.04 | How fast credit decays with action age |
| `WEIGHT_LR` | 0.02 | Learning rate for policy weight updates |
| `WEIGHT_DECAY` | 0.00001 | L2 regularization per tick (tuned for ~4% loss per 4000-tick life) |
| `PAIN_AMPLIFIER` | 3.0 | Negative gradient multiplier (pain teaches faster) |
| `DEATH_CREDIT` | -0.5 | One-time retroactive signal on death (state-dependent weights only) |
| `CREDIT_DEADZONE` | 0.01 | Minimum |gradient| to trigger credit assignment (filters metabolic noise) |
| `MAX_WEIGHT_NORM` | 2.0 | L2 norm cap per action weight vector (synaptic homeostasis) |
| `GLOBAL_LR` | 0.08 | EMA update factor for global biases |
| `GLOBAL_RETAIN` | 0.995 | EMA retention for global biases |
| `ANTICIPATION_WEIGHT` | 0.5 | How much the predicted trajectory change influences preferences |

**Emergent properties**: The linear policy creates **continuous, state-dependent action preferences** — the agent learns to do different things in different situations through feature-level associations. The adaptive exploration rate creates a natural **curiosity → exploitation** transition: novel environments are explored, familiar environments are exploited. Urgency-driven exploitation creates **panic behavior** — when things are going badly, the agent falls back on what has worked before. The exploration floor (10%) ensures agents never stop trying new things, even under maximum urgency.

**Performance**: The action selector uses **stack arrays** for action value computation and a **pre-allocated ring buffer** for state snapshots, eliminating all per-tick heap allocations.

---

### 4.5 Homeostatic Monitor (`homeostasis.rs`)

**What it does**: Tracks the agent's internal physiological signals (energy, integrity) across three timescales and computes a gradient (improving vs. worsening) plus a non-linear urgency signal.

**Why it exists**: The homeostatic monitor provides the **only evaluative signal** in the entire brain. It is deliberately *not* a reward function — it doesn't say "eating is good" or "damage is bad". It simply reports whether internal variables are trending toward or away from their stable points. All behavior that appears goal-directed emerges from the brain learning to keep this gradient positive.

**How it works**:

#### Three-Timescale Gradient Tracking

Each tick, the raw gradient is computed from the change in energy and integrity:

```rust
let raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;
// ENERGY_WEIGHT = 0.6, INTEGRITY_WEIGHT = 0.4
```

Note that integrity is weighted lower than energy (0.4 vs 0.6). This raw gradient is then tracked at three timescales via exponential moving averages:

| Timescale | EMA α | Effective window | Purpose |
|-----------|-------|------------------|---------|
| Fast | 0.4 | ~5 ticks | Immediate reactions (flinch, grab) |
| Medium | 0.04 | ~50 ticks | Short-term strategy (approach food, avoid threats) |
| Slow | 0.004 | ~500 ticks | Long-term trends (is this environment safe?) |

The composite gradient blends all three:
```rust
let base_gradient = gradient_fast * 0.50
                  + gradient_medium * 0.35
                  + gradient_slow * 0.15;
let gradient = base_gradient * (1.0 + urgency);
```

The urgency amplifier means the gradient signal is stronger when the agent is in danger — a small improvement matters more when you're nearly dead.

#### Bounded Distress Curve

Urgency is computed from a non-linear distress function that maps health levels [0, 1] to distress [0, 10]:

```rust
fn distress_curve(level: f32, exponent: f32) -> f32 {
    let clamped = level.clamp(0.01, 1.0);
    (1.0 - clamped).powf(exponent) * DISTRESS_SCALE  // DISTRESS_SCALE = 10.0
}
```

The exponent is configurable via `BrainConfig::distress_exponent` (default 2.0, range [1.5, 5.0], heritable). Lower exponents make the agent react sooner to moderate drops; higher exponents keep the agent calm longer but produce sharper panic at critical levels.

Example values (with default exponent 2.0):
| Level | Distress | Interpretation |
|-------|----------|----------------|
| 1.0 | 0.0 | Perfect health, no urgency |
| 0.8 | 0.4 | Mildly concerned |
| 0.5 | 2.5 | Moderately urgent |
| 0.2 | 6.4 | Highly urgent |
| 0.1 | 8.1 | Critical |
| 0.01 | 9.8 | Near death |

The shape means urgency increases slowly at first (the agent is tolerant of mild drops) but accelerates rapidly as levels approach zero. The exponent controls the steepness of this curve.

Final urgency is the average distress of energy and integrity:
```rust
self.urgency = (energy_distress + integrity_distress) * 0.5;
```

#### HomeostaticState

The `HomeostaticState` struct returned from `update()` contains:
- `gradient`: composite gradient (positive = improving)
- `gradient_fast`, `gradient_medium`, `gradient_slow`: per-timescale gradients
- `urgency`: non-linear distress signal [0, ∞)

**Key constants**:

| Constant | Value | Effect |
|----------|-------|--------|
| `ENERGY_WEIGHT` | 0.6 | Energy contribution to raw gradient |
| `INTEGRITY_WEIGHT` | 0.4 | Integrity contribution to raw gradient |
| `FAST_EMA_ALPHA` | 0.6 | Fast timescale responsiveness |
| `MEDIUM_EMA_ALPHA` | 0.04 | Medium timescale responsiveness |
| `SLOW_EMA_ALPHA` | 0.004 | Slow timescale responsiveness |
| `GRADIENT_BLEND_FAST` | 0.5 | Fast timescale weight in composite |
| `GRADIENT_BLEND_MEDIUM` | 0.35 | Medium timescale weight in composite |
| `GRADIENT_BLEND_SLOW` | 0.15 | Slow timescale weight in composite |
| `DISTRESS_SCALE` | 10.0 | Maximum distress value |
| `MAX_DISTRESS` | 10.0 | Hard cap on distress |

**Emergent properties**: The three timescales create **temporal context** — the agent can react to immediate threats (fast gradient) while also considering whether its overall strategy is working (slow gradient). The urgency signal creates **survival pressure** — low health makes the agent conservative and reactive. The gradient amplification by urgency creates **desperation** — small improvements in critical situations feel much more significant.

---

### 4.6 Sensory Habituation (`habituation.rs`)

**What it does**: A post-encoder filter that attenuates repetitive sensory dimensions and produces a `curiosity_bonus` that feeds into the exploration rate. When the encoded state stops changing, habituation grows and curiosity rises, pushing the agent to break out of monotonous loops.

**How it works**:

For each dimension of the encoded state, an exponential moving average (EMA) tracks the magnitude of change between ticks. Dimensions with low change magnitude are attenuated (dampened toward zero), producing a **habituated state** that downstream consumers (Memory, Predictor, ActionSelector) use instead of the raw encoded state.

1. **Per-dimension change EMA**: `change_ema[i] = alpha * |current[i] - prev[i]| + (1 - alpha) * change_ema[i]`
2. **Attenuation**: `attenuation[i] = max(ATTENUATION_FLOOR, change_ema[i] / sensitivity)`
3. **Habituated state**: `habituated[i] = encoded[i] * attenuation[i]`
4. **Curiosity bonus**: `curiosity_bonus = (1.0 - mean_attenuation) * max_curiosity_bonus`

When all dimensions are changing rapidly, mean attenuation is high and curiosity bonus is near zero. When the agent is stuck in a loop seeing the same thing, attenuation drops and curiosity bonus rises, increasing exploration.

**Key constants**:

| Constant / Parameter | Value | Configurable? |
|----------------------|-------|---------------|
| `HABITUATION_EMA_ALPHA` | 0.15 | No (hardcoded) |
| `ATTENUATION_FLOOR` | 0.05 | No (hardcoded) |
| `habituation_sensitivity` | 20.0 (default) | Yes (`BrainConfig`) |
| `max_curiosity_bonus` | 0.6 (default) | Yes (`BrainConfig`) |

---

### 4.7 Motor Fatigue (`fatigue.rs`)

**What it does**: Tracks a ring buffer of recent motor outputs and dampens the MotorCommand when motor variance is low. This prevents the agent from mechanically repeating the same motor pattern indefinitely. Recovery is immediate -- as soon as the agent produces varied output, fatigue lifts.

**How it works**:

A fixed-size ring buffer stores recent `(forward, turn)` pairs. The variance of each component is computed over the window. Low variance means repetitive output, which triggers fatigue dampening:

1. **Motor variance**: `variance = var(forward_history) + var(turn_history)`
2. **Fatigue factor**: `factor = max(fatigue_floor, 1.0 - exp(-variance * recovery_sensitivity))`
3. **Dampened output**: `motor.forward *= factor; motor.turn *= factor;`

When variance is high (diverse motor output), the fatigue factor is near 1.0 and output is unaffected. When variance is near zero (repetitive output), the factor drops toward `fatigue_floor`, weakening the command and giving other action candidates a chance to win on subsequent ticks.

**Key constants / parameters**:

| Constant / Parameter | Value | Configurable? |
|----------------------|-------|---------------|
| `FATIGUE_WINDOW` | 32 | No (hardcoded) |
| `fatigue_recovery_sensitivity` | 8.0 (default) | Yes (`BrainConfig`) |
| `fatigue_floor` | 0.1 (default) | Yes (`BrainConfig`) |

---

### 4.8 Capacity Manager (`capacity.rs`)

**What it does**: Manages the brain's per-tick processing budget, dynamically allocating recall slots and reserving capacity for surprise (novel patterns).

**Why it exists**: Without capacity limits, the brain would recall all patterns every tick — there would be no selectivity, no attention, no prioritization. The capacity manager enforces scarcity, forcing the brain to be strategic about what it recalls. The adaptive allocation means the brain "thinks harder" about novel situations and "runs on autopilot" in familiar ones.

**How it works**:

#### Adaptive Recall Budget

The recall budget scales linearly with prediction error:

```rust
let error_scale = (0.5 + avg_prediction_error * 2.0).clamp(0.5, 1.0);
let base_budget = (max_recall_budget as f32 * error_scale).round() as usize;
```

- Low error (familiar situation): use 50% of maximum budget
- High error (novel situation): use up to 100% of maximum budget

#### Surprise Budget

A fraction of the total capacity is reserved for novel/unexpected patterns. This fraction increases when a surprise spike is detected:

```rust
if prediction_error > avg_prediction_error * 2.0 && prediction_error > 0.05 {
    self.surprise_fraction = (self.surprise_fraction + 0.05).min(0.4);
    self.surprise_active_ticks = 0;
} else {
    // Decay surprise fraction after 20 ticks of no surprises
    if self.surprise_active_ticks > 20 {
        self.surprise_fraction = (self.surprise_fraction - 0.01).max(0.05);
    }
}
```

The surprise budget is subtracted from the recall budget:
```rust
let recall_budget = base_budget.min(max_recall_budget - surprise_budget);
```

#### Cognitive Load Tracking

The `CognitiveLoad` struct tracks:
- `recall_budget`: current allocation
- `surprise_budget`: reserved for novelty
- `avg_utilization`: EMA of actual slots used vs. available
- `avg_prediction_error`: EMA of prediction error

**Emergent properties**: The adaptive budget creates **attentional focus** — novel situations get more processing resources. The surprise budget creates **novelty bias** — capacity is reserved so the brain can respond to unexpected events even when fully loaded. The overall constraint creates **cognitive load** — the brain can be "overwhelmed" in highly novel, rapidly changing environments.

---

### 4.9 Brain Orchestrator (`brain.rs`)

**What it does**: Owns all components, orchestrates the tick loop, and produces telemetry.

**Why it exists**: Individual components need to be coordinated in a specific order (you can't compute prediction error before you have the prediction and the actual state). The brain struct is the integration point that wires everything together.

#### The tick() Method

The 14-step orchestration documented in the [Data Flow Diagram](#3-data-flow-diagram) section. Key design choice: learning happens *before* recall and prediction, using the *previous* tick's prediction error. This means the brain is always learning from one tick ago, which avoids circular dependencies.

#### The trauma() Method

`Brain::trauma(severity)` delegates to `PatternMemory::trauma()`, applying a bulk reinforcement decay across all stored patterns. Called by the sandbox on agent death (with severity 0.2) when brain persistence is enabled. This models the cognitive cost of catastrophic discontinuity — death damages memory, but deeply learned patterns persist.

#### The death_signal() Method

`Brain::death_signal()` delegates to `ActionSelector::death_signal()`, which fires a calibrated negative credit event (`DEATH_CREDIT = -0.5`, urgency = 0.0) with `update_global = false`. After pain amplification (3.0×), the effective gradient is -1.5. This retroactively penalizes all recent actions in the 64-tick history buffer, with exponential decay so the most recent actions receive the strongest penalty. It also resets `MotorFatigue`, clearing the ring buffer so the respawned agent starts with a fresh fatigue factor of 1.0 — without this, the dying agent's constant motor output saturates the buffer and locks fatigue at maximum on respawn.

**Global biases are intentionally excluded** from the death signal. Without this guard, the accumulated credit across 64 recent actions (typically all the same action) creates a ~-0.68 shift in global bias per death — catastrophically destroying whichever action the agent happened to be using. This leads to the agent cycling through punishing every action until all global biases are equally negative, causing "learned helplessness" (straight-line walking). By restricting death credit to state-dependent weights only, the agent learns context-specific avoidance ("forward when I see red terrain is bad") without context-independent punishment ("forward is always bad").

#### BrainTelemetry

All tracked metrics per tick:

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `u64` | Current tick number |
| `prediction_error` | `f32` | This tick's prediction error (0 = perfect) |
| `memory_utilization` | `f32` | Fraction of memory slots in use [0, 1] |
| `memory_active_count` | `usize` | Number of active patterns |
| `action_entropy` | `f32` | Shannon entropy of action distribution |
| `exploration_rate` | `f32` | Current exploration probability |
| `homeostatic_gradient` | `f32` | Composite gradient (+ improving, − worsening) |
| `homeostatic_urgency` | `f32` | Non-linear distress level |
| `recall_budget` | `usize` | Recall slots allocated this tick |
| `avg_prediction_error` | `f32` | Moving average error (window=32) |
| `exploitation_ratio` | `f32` | Fraction of actions that were exploitative [0, 1] |
| `decision_quality` | `f32` | Composite quality score [0, 1] |

#### Decision Quality Score

```rust
let decision_quality = (1.0 - scalar_error.clamp(0.0, 1.0))
    * (1.0 - exploration_rate)
    * (1.0 + homeo_state.gradient).clamp(0.0, 2.0)
    / 2.0;
```

This is a composite metric that is high when:
- Prediction error is low (the agent understands its environment)
- Exploration rate is low (the agent is exploiting learned knowledge)
- Homeostatic gradient is positive (things are going well)

Divided by 2.0 to normalize to [0, 1].

#### Behavior Phase Classification

Based on a composite behavioral score that accounts for exploitation, prediction accuracy, and homeostatic state:

```
score = exploitation_ratio × (1 − prediction_error) × (1 − homeostatic_urgency)
```

This means an agent that avoids danger but starves (high urgency) cannot reach ADAPTED.

| Phase | Composite Score | Interpretation |
|-------|-------------------|----------------|
| `RANDOM` | < 2% | Brain is mostly exploring randomly |
| `EXPLORING` | 2–8% | Starting to learn, still exploring heavily |
| `LEARNING` | 8–20% | Learning is working, composite score increasing |
| `ADAPTED` | ≥ 20% | Brain has adapted to its environment |

#### DecisionSnapshot

A per-tick snapshot of the brain's decision state, captured at the end of `tick_inner()` and stored in `Brain::last_decision`. This enables the UI to show a real-time "decision stream" — a scrollable log of what the brain decided, why, and what happened.

```rust
pub struct DecisionSnapshot {
    pub tick: u64,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub exploration_rate: f32,
    pub gradient: f32,          // composite homeostatic gradient
    pub raw_gradient: f32,      // avg prediction error (raw)
    pub urgency: f32,           // homeostatic urgency
    pub prediction_error: f32,
    pub patterns_recalled: usize,
    pub credit_magnitude: f32,  // abs sum of credit assigned this tick
    pub energy: f32,            // energy signal [0, 1]
    pub integrity: f32,         // integrity signal [0, 1]
    pub phase: &'static str,    // behavior phase label
    pub alive: bool,
}
```

| Field | Description |
|-------|-------------|
| `motor_forward` / `motor_turn` | The continuous motor outputs chosen this tick |
| `exploration_rate` | Current exploration probability (higher = more random) |
| `gradient` | Composite homeostatic gradient (positive = improving) |
| `raw_gradient` | Rolling average prediction error |
| `urgency` | Non-linear distress level (suppresses exploration) |
| `prediction_error` | This tick's prediction error |
| `patterns_recalled` | How many memory patterns were recalled this tick |
| `credit_magnitude` | Total absolute credit assigned to actions this tick (higher = stronger learning signal) |
| `energy` / `integrity` | Normalized physiological signals at decision time |
| `phase` | Current behavior phase label (RANDOM/EXPLORING/LEARNING/ADAPTED) |

The sandbox's agent detail tab renders this as a color-coded scrollable log, with credit magnitude highlighted in green (positive) or red (negative gradient) to show at a glance whether the agent's recent decisions are being reinforced or penalized.

---

## 5. Emergent Phenomena

None of these behaviors are explicitly programmed. They arise from the interaction of components and constraints:

| Phenomenon | How It Emerges | Contributing Components |
|------------|---------------|------------------------|
| **Attention** | Capacity constraints force selective recall; encoder bottleneck compresses information | Encoder, Capacity Manager |
| **Fear / Avoidance** | Negative homeostatic gradient from damage → pain amplifier (3×) makes damage signal loud → credit assignment blames recent actions via state snapshots → policy weights learn to avoid danger-associated sensory features → **prospective evaluation** applies these learned associations to the predicted future, so the agent anticipates danger before entering it | Memory, Action Selector, Predictor, Homeostasis |
| **Curiosity** | High prediction error in safe (low urgency) situations → exploration rate increases; additionally, sensory habituation produces a curiosity_bonus when input is monotonous, further boosting exploration even when prediction error is low | Predictor, Action Selector, Capacity Manager, Sensory Habituation |
| **Habit Formation** | Repeated successful actions build strong context-action values → exploitation ratio increases → behavior becomes automatic | Memory, Action Selector |
| **Startle / Surprise** | Sudden prediction error spike → surprise budget increases → recall budget shifts → exploration spikes | Predictor, Capacity Manager, Action Selector |
| **Adaptation** | Prediction error decreases over time in stable environments → exploration drops → behavior stabilizes | Predictor, Action Selector |
| **Panic** | Low energy/integrity → high urgency → exploration suppressed → agent falls back on best-known actions | Homeostasis, Action Selector |
| **Forgetting** | Patterns that are not recalled or reinforced decay below threshold and are removed | Memory (smart decay) |
| **Chunking** | Co-occurring patterns get associated → recalling one activates the chain → complex sequences are treated as units | Memory (associations) |
| **Desensitization** | The slow EMA timescale integrates gradual changes; constant mild negative gradient eventually stops triggering strong reactions | Homeostasis (multi-timescale) |
| **Contextual Memory** | Linear policy weights map encoded state features to action preferences — different percepts trigger different behaviors | Action Selector, Encoder |
| **Boredom / Loop Breaking** | Monotonous sensory input → sensory habituation attenuates repetitive dimensions → curiosity_bonus rises → exploration increases; simultaneously, low motor variance → motor fatigue dampens output → agent's repeated action weakens, giving other actions a chance | Sensory Habituation, Motor Fatigue, Action Selector |
| **Cognitive Overload** | Novel, rapidly changing environments exhaust recall budget → brain can't keep up → behavior degrades | Capacity Manager, Memory |

---

## 6. Configuration (BrainConfig)

```rust
pub struct BrainConfig {
    pub memory_capacity: usize,            // Maximum stored patterns
    pub processing_slots: usize,           // Max recall operations per tick
    pub visual_encoding_size: usize,       // Visual downsampling resolution
    pub representation_dim: usize,         // Internal representation vector length
    pub learning_rate: f32,                // Base learning rate
    pub decay_rate: f32,                   // Pattern decay per tick
    pub distress_exponent: f32,            // Distress curve exponent (default 2.0)
    pub habituation_sensitivity: f32,      // Boredom speed (default 20.0)
    pub max_curiosity_bonus: f32,          // Max exploration from monotony (default 0.6)
    pub fatigue_recovery_sensitivity: f32, // Fatigue relief speed (default 8.0)
    pub fatigue_floor: f32,                // Min motor output under fatigue (default 0.1)
}
```

### Parameter Effects

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| `memory_capacity` | Rapid forgetting, lives in the moment, fast adaptation but no long-term memory | Rich memory, can recognize situations seen long ago, but slower slot search |
| `processing_slots` | Narrow attention, only recalls best match, focused but brittle | Broad attention, considers many patterns, flexible but slower per tick |
| `visual_encoding_size` | Coarse vision, can't distinguish similar scenes, but small weight matrix | Fine vision, better visual discrimination, but more parameters to learn |
| `representation_dim` | Compressed representation, fast but loses information | Rich representation, captures more nuance but harder to learn |
| `learning_rate` | Slow adaptation, stable but takes longer to respond to change | Fast adaptation, responsive but risks oscillation |
| `decay_rate` | Long memory retention, accumulates patterns, can fill memory with stale data | Aggressive forgetting, only keeps very recent/frequent patterns |
| `distress_exponent` | Reacts sooner to moderate health drops, more cautious overall | Stays calm longer, but panics harder at critical levels |
| `habituation_sensitivity` | Slow to bore, tolerates repetitive input longer | Bores quickly, curiosity bonus rises fast in monotonous situations |
| `max_curiosity_bonus` | Weak exploration boost from monotony, loops persist longer | Strong exploration boost from monotony, breaks loops aggressively |
| `fatigue_recovery_sensitivity` | Slow fatigue recovery, motor dampening lingers after variance returns | Fast fatigue recovery, dampening lifts immediately with diverse output |
| `fatigue_floor` | Motor output can be nearly zeroed by fatigue, strong loop-breaking | Motor output stays substantial even under full fatigue, gentler loop-breaking |

### Presets

| Preset | Capacity | Slots | Visual | Repr Dim | LR | Decay | Distress Exp | Hab. Sens. | Max Curiosity | Fatigue Recov. | Fatigue Floor | Character |
|--------|----------|-------|--------|----------|------|-------|-------------|-----------|---------------|---------------|--------------|-----------|
| `tiny()` | 24 | 8 | 32 | 16 | 0.08 | 0.002 | 2.0 | 25.0 | 0.6 | 10.0 | 0.15 | Reactive, impulsive, forgetful. Bores fast, breaks loops quickly. |
| `default()` | 128 | 16 | 64 | 32 | 0.05 | 0.001 | 2.0 | 20.0 | 0.6 | 8.0 | 0.1 | Balanced. Good starting point for all anti-loop parameters. |
| `large()` | 512 | 32 | 128 | 64 | 0.03 | 0.0005 | 2.5 | 15.0 | 0.4 | 6.0 | 0.1 | Thoughtful, patient. Slower to bore, calmer under moderate stress. |

### Tuning Guide

| Problem | Likely Cause | Try |
|---------|-------------|-----|
| Agent is too random / never settles | Learning rate too low, or memory too small to build stable patterns | Increase `learning_rate` to 0.08–0.1, or increase `memory_capacity` |
| Agent gets stuck doing one thing | Learning rate too high (overfit to first success), or decay too low (stale patterns dominate) | Decrease `learning_rate`, increase `decay_rate`, or decrease `memory_capacity` |
| Agent ignores visual changes | `visual_encoding_size` too small, or `representation_dim` too small to capture visual information | Increase `visual_encoding_size` and `representation_dim` together |
| Agent seems "blind" to threats | Homeostatic gradient is too slow to react (this is in homeostasis constants, not config) | Increase `learning_rate` so credit assignment is stronger |
| Agent panics too early | Urgency distress curve kicks in at moderate levels | Increase `distress_exponent` (e.g., 3.0–4.0) — agent stays calmer longer but panics harder at critical levels |
| Agent stuck in loops | Repetitive sensory input and motor output without breaking free | Increase `habituation_sensitivity` and `max_curiosity_bonus`, decrease `fatigue_floor` |
| Memory fills up too fast | `decay_rate` too low or `memory_capacity` too small | Increase `decay_rate` or increase `memory_capacity` |

---

## 7. Testing

### Philosophy

Tests verify **behavioral properties**, not implementation details. They check things like:
- "prediction error should decrease with repeated input" (the system learns)
- "similar inputs should produce similar encodings" (the encoder preserves similarity)
- "positive gradient should increase action value" (credit assignment works)
- "the brain should not produce NaN with extreme inputs" (numerical stability)

### Running Tests

```bash
cargo test -p xagent-brain --lib
```

### Test Categories

| Module | Tests | What They Verify |
|--------|-------|-----------------|
| `encoder` | 5 | Similarity preservation, dimension correctness, determinism, adaptation modifies weights, different inputs ≠ same encoding |
| `memory` | 8 | Store/recall, smart decay, frequency-based retention, temporal sequence tracking, association chains, capacity limits, generation-based staleness detection, co-occurrence strengthening |
| `predictor` | 5 | Zero error for identical states, error decreases with learning, context influences prediction, error history tracking, gradient descent convergence |
| `action` | 7 | Exploration increases with prediction error, urgency decreases exploration, initial entropy is maximal, positive gradient increases action values, exploitation ratio starts at zero, stability reduces exploration, linear policy learns state-dependent preferences |
| `homeostasis` | 8 | Stable signals → zero gradient, improving → positive gradient, worsening → negative gradient, critical levels → high urgency, multi-timescale separation, distress curve bounds, energy drop/gain response, low-level urgency |
| `capacity` | 3 | High error increases recall budget, surprise spikes increase surprise fraction, cognitive load reporting |
| `brain` | 5 | Produces motor command, telemetry updates each tick, runs 100 ticks without panic, prediction error decreases with repeated input, handles extreme inputs (all-zero, all-one) |

---

## 8. Design Decisions

### Why Explicit Data Structures Over Neural Networks

The brain uses weight matrices and hash tables instead of deep neural networks. This is intentional:
- **Inspectability**: Every policy weight, every association link, every pattern reinforcement can be read and understood. You can answer "why did the agent turn left?" by inspecting the action weight vectors and seeing which sensory features drove the preference.
- **Deterministic debugging**: Given the same state, you can trace exactly which patterns were recalled, what prediction was made, and why a particular action was chosen.
- **Minimal dependencies**: No ML framework needed. Just `f32` arrays and standard math.
- **Interpretable emergence**: When interesting behavior appears, you can trace it to specific data structure interactions rather than opaque deep weight matrices.

### Why Cosine Similarity for Pattern Matching

Cosine similarity measures angle between vectors, ignoring magnitude. This is correct for encoded states because:
- The encoder uses `tanh()`, so all values are in [-1, 1] — magnitude carries less information than direction.
- Patterns with similar perceptual meaning should be similar regardless of activation strength.
- It's cheap to compute (dot product + two norms) and well-understood.

### Why Generation Counters for Association Integrity

When a memory slot is overwritten, all association links pointing to that slot become stale. We could scan all patterns and remove stale links (O(n × k) where k = average associations per pattern), or we can use generation counters and validate lazily at retrieval time (O(1) per link check). The generation approach is both simpler and faster.

### Why Three Timescales in Homeostasis

Biological nervous systems track changes at multiple timescales — immediate reflexes (milliseconds), emotional responses (seconds-minutes), and mood/disposition (hours-days). Three timescales (fast ≈5 ticks, medium ≈50 ticks, slow ≈500 ticks) capture this hierarchy:
- **Fast**: "I just got hit" — immediate reaction
- **Medium**: "This area has been bad for me" — tactical adjustment
- **Slow**: "My overall strategy isn't working" — strategic shift

### Why Linear Policy Instead of Context Hashing

The original context-hash approach used 128 buckets to discretize the encoded state. This had three critical flaws:
1. **Credit assigned to wrong context**: All actions in history were credited in the *current* bucket, not the bucket where each action was originally chosen.
2. **Hash collisions**: 128 buckets from hashing 8 quantized dimensions → similar scenes mapped to different buckets, requiring 128× more experience.
3. **Random walks give equal credit**: All 8 action types received roughly equal credit from random exploration, producing no net learning.

The linear policy replaces the hash table with a weight matrix (`repr_dim × NUM_ACTIONS`). Each action's preference is `dot(weights[a], state)`. Credit assignment uses the stored state snapshot from when each action was chosen:

```
Δweights[a] += lr × credit × state_similarity × state_snapshot_at_action_time
```

This means the weights learn **which features of the encoded state predict good outcomes for each action**. The linear policy is continuous (similar states → similar preferences), uses all encoded dimensions, and correctly attributes credit to the sensory context where each action was taken.

### Why Credit Deadzone Filters Metabolic Noise

Every tick, energy depletion produces a small negative homeostatic gradient (~0.006). Without a deadzone, this constant signal triggers credit assignment on every tick, treating normal metabolism as a negative outcome. The `CREDIT_DEADZONE = 0.01` threshold ensures only meaningful events — food consumption (+0.03), damage (-0.02), death (-0.5) — produce weight updates. This is analogous to sensory gating in biological systems, where constant background stimuli are filtered out to preserve signal clarity.

### Why State-Conditioned Credit Assignment

Naive credit assignment applies the same credit to all recent actions regardless of the state in which they were taken. This means dying in a danger zone penalizes actions taken in safe states equally — the agent learns "forward is bad everywhere" instead of "forward-in-danger is bad." State-conditioned credit modulates each action's update by the cosine similarity between the current state and the state when that action was chosen. High similarity → full credit. Low similarity → minimal credit. This is analogous to context-dependent synaptic plasticity in biological systems.

### Why Weight Normalization (Synaptic Homeostasis)

Without a norm cap, repeated reinforcement in one context causes unbounded weight growth, which creates two problems: (1) dot-product preferences become so large that exploration-phase softmax collapses to a deterministic choice, and (2) future credit updates in different contexts are overwhelmed by the accumulated magnitude. Clipping to `MAX_WEIGHT_NORM = 2.0` after each credit pass provides synaptic homeostasis — keeping weights in a range where learning remains effective and exploration remains meaningful.

### Why Uniform Random Exploration Instead of Softmax

Softmax exploration was effectively deterministic with the linear policy. In 32 dimensions with norm-2 weight vectors, the dot product differences between the best and worst actions are large enough that the softmax assigns 99%+ probability to the best action. This makes "exploration" functionally identical to exploitation. Uniform random (`rng.random_range(0..NUM_ACTIONS)`) guarantees 12.5% probability per action during exploration ticks, ensuring genuine behavioral diversity.

### Why Random Argmax Tie-Breaking

With zero-initialized weights, all actions have identical preferences. Rust's `max_by` deterministically returns the last tied element (action 7), creating a fixed behavioral bias from tick zero. Random tie-breaking ensures a fresh agent distributes equally across all actions until learning creates genuine preference differences.

### Why "Forage" Replaced "Do Nothing" (Action 7)

The original action 7 was "stay still" (zero motor output). This created a freeze trap: agents with no learned preferences defaulted to action 7 (due to `max_by` tie-breaking) and never moved, making it impossible to encounter food or threats and bootstrap learning. Replacing it with "forage" (`forward: 0.15, action: Consume`) ensures every action produces some movement, guaranteeing environmental interaction even with untrained weights.

### Why Learning Rate is Modulated by Homeostatic Gradient

```rust
let modulated_lr = self.config.learning_rate * (1.0 + homeo_state.gradient.abs());
```

When internal state is changing rapidly (either improving or worsening), the brain should learn faster — these are the moments that matter. When things are stable, there's less to learn. This mimics the role of neuromodulators (dopamine, norepinephrine) in biological brains, which modulate synaptic plasticity based on salience.

---

## 9. CPU Performance Optimizations

The brain is designed for **zero per-tick heap allocations**. Key optimizations:

| Component | Optimization | Impact |
|-----------|-------------|--------|
| **Encoder** | Pre-allocated scratch buffer for feature extraction | No `Vec` allocation per tick |
| **Encoder / Predictor** | `fast_tanh` Padé approximant replaces `f32::tanh()` | Avoids expensive transcendental function |
| **Memory** | Capped associations (`.take(8)`) during retrieval | Bounds work per recall operation |
| **Predictor** | Pre-allocated scratch buffer for prediction computation | No `Vec` allocation per tick |
| **Action Selector** | Stack arrays for action value computation | No heap allocation for selection |

The `fast_tanh` approximant uses a rational Padé form that is accurate to within ~0.001 for inputs in [-3, 3] (which covers the clamped weight range), while being significantly cheaper than the standard library implementation.

---

## 10. Known Limitations & Future Work

### Current Limitations

- **Encoder adaptation is minimal**: Only L2 regularization is applied. The encoder weights are never truly trained to optimize prediction — they are initialized randomly and gently constrained. A future version could use prediction-error-driven weight updates that flow through the predictor.

- **Action space is discrete and fixed**: The 8 actions are hardcoded. A continuous action space with learned parameterization would allow more nuanced behavior (e.g., moving at different speeds, turning at specific angles).

- **No hierarchical pattern abstraction**: All patterns are stored at the same level of abstraction. There is no mechanism for forming higher-order patterns ("I'm in a corridor" from a sequence of wall-patterns) or chunking temporal sequences into reusable units.

- **Memory capacity is fixed**: The brain cannot grow its memory. A dynamic capacity that expands in rich environments and contracts in simple ones would better match biological memory allocation.

- **No inter-agent brain communication**: Each brain is entirely isolated. There is no mechanism for one agent to share learned patterns or action values with another. Social learning and cultural transmission would require some form of brain-to-brain communication channel.

- **Flat recall search**: Recall iterates over all patterns (O(n)) every tick. For large memory capacities, this becomes expensive. A spatial index (e.g., locality-sensitive hashing) would enable sub-linear recall.

- **No sleep/consolidation**: Biological brains consolidate memories during sleep, replaying and strengthening important patterns. The current system has no offline consolidation phase — all learning happens online during ticks.

- **Single sensory modality fusion**: All sensory inputs are concatenated into a single feature vector. There is no modality-specific processing (e.g., separate visual and proprioceptive streams that are later integrated).

### Future Directions

- **Hierarchical temporal memory**: Stack multiple levels of pattern memory, each operating at a different temporal granularity.
- **Continuous action space**: Replace the 8-action table with a continuous policy parameterized by the encoded state.
- **Curiosity-driven exploration**: Use prediction error as an intrinsic reward for exploration, beyond just modulating the exploration rate.
- **Multi-agent pattern sharing**: Allow agents to "teach" each other by sharing association strengths or pattern representations.
- **Dreaming/replay**: Periodically replay stored pattern sequences during idle time to consolidate important memories and prune irrelevant ones.
