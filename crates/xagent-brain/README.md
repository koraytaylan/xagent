# xagent-brain

A general-purpose cognitive architecture based on **predictive processing**, running entirely on GPU.

The brain crate is the decision-making core of each xagent. It has no hardcoded behaviors -- no "hunger module", no "fear module", no goal system. Everything the agent does emerges from a single loop and a single principle:

> **Prediction error drives everything.**

```
sense --> extract --> encode --> habituate/homeo --> recall --> predict+act --> learn+store
  |         |           |              |                |           |              |
  |    feature_extract  |     habituate_homeo    recall_score  predict_and_act  learn_and_store
  |      (pass 1)   encode      (pass 3)        recall_topk     (pass 6)        (pass 7)
  |                 (pass 2)                    (pass 4 & 5)
  |                                                                              |
CPU uploads                                                              CPU reads back
sensory frames                                                          motor commands
(~52KB / 50 agents)                                                    (~800B / 50 agents)
```

The brain receives packed `SensoryFrame` data and emits `MotorCommand` values. Between those two endpoints, 7 WGSL compute shaders run on GPU in a single `queue.submit()`. All persistent brain state lives permanently in GPU storage buffers -- there is no CPU-side brain state, no per-tick marshaling bottleneck. The only CPU<-->GPU transfers per tick are sensory input upload and motor command readback.

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [GPU-Resident Design](#4-gpu-resident-design)
5. [Buffer Layout](#5-buffer-layout)
6. [Component Deep Dive: The 7-Pass Pipeline](#6-component-deep-dive-the-7-pass-pipeline)
   - [6.1 Feature Extraction (Pass 1)](#61-feature-extraction-pass-1--feature_extractwgsl)
   - [6.2 Encoding (Pass 2)](#62-encoding-pass-2--encodewgsl)
   - [6.3 Habituation + Homeostasis (Pass 3)](#63-habituation--homeostasis-pass-3--habituate_homeowgsl)
   - [6.4 Recall Scoring (Pass 4)](#64-recall-scoring-pass-4--recall_scorewgsl)
   - [6.5 Recall Top-K Selection (Pass 5)](#65-recall-top-k-selection-pass-5--recall_topkwgsl)
   - [6.6 Prediction + Action Selection (Pass 6)](#66-prediction--action-selection-pass-6--predict_and_actwgsl)
   - [6.7 Learning + Memory Storage (Pass 7)](#67-learning--memory-storage-pass-7--learn_and_storewgsl)
7. [Emergent Phenomena](#7-emergent-phenomena)
8. [Host API (gpu_brain.rs)](#8-host-api-gpu_brainrs)
9. [Configuration (BrainConfig)](#9-configuration-brainconfig)
10. [Testing](#10-testing)
11. [Design Decisions](#11-design-decisions)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## The Brain Has No Eyes

The most important thing to understand about this architecture: **the brain has zero semantic knowledge of its inputs**.

The `SensoryFrame` that arrives from the sandbox has named fields -- `vision`, `touch_contacts`, `energy_signal`. But the brain never sees those names. `buffers::pack_sensory_frame()` flattens *everything* into a single `[f32; 267]` array, and `feature_extract.wgsl` further compresses it to 217 features. From that point on, the brain operates on opaque numerical vectors. It has no concept of "vision," no awareness that it has "eyes," no understanding that index 47 was once an RGBA pixel and index 73 was once an energy level.

```
World --> SensoryFrame --> pack_sensory_frame() --> [267 f32] --> feature_extract.wgsl --> [217 f32]
               |                                                        |
      Named fields like                                        Brain sees only a
      "vision", "energy"                                       flat array<f32>
```

Consider what happens when another agent -- say, a magenta-colored one -- enters the visual field. The brain doesn't receive "agent detected" or "entity of type Agent at bearing 30 degrees." It experiences indices 12--15 shifting from `[0.3, 0.6, 0.2, 1.0]` to `[0.9, 0.2, 0.6, 1.0]`. Simultaneously, a touch contact might add nonzero values at indices 199--202 (direction, intensity, tag). The brain has no legend for any of this. It doesn't know that `surface_tag=4` means "agent." It doesn't know that the shifted values represent magenta. Over hundreds of ticks, if this pattern of input correlates with energy dropping (food competition), the brain discovers -- through prediction error and homeostatic gradient alone -- that "those numerical patterns are bad for me." The concept of "that's a competitor" *emerges* from experience, not from labels.

This is the fundamental difference from traditional AI systems. There are no reward functions hand-crafted by engineers. No labeled feature vectors telling the model "this is vision, this is hunger." No hardcoded categories like "food," "hazard," or "friend." The `SensoryFrame` struct with its named fields is engineering scaffolding -- it's the "body's" wiring that collects data from the simulated world. The packing function strips all that structure away. What remains is prediction + homeostatic gradient + experience, and from these three ingredients, all meaning is discovered.

---

## 1. Theoretical Foundation

### Predictive Processing & Active Inference

The brain crate implements a simplified version of the **predictive processing** framework from computational neuroscience. The core idea, developed by Karl Friston (free energy principle) and echoed in Jeff Hawkins' work on hierarchical temporal memory, is that brains are fundamentally *prediction machines*:

- The brain constantly generates predictions about what sensory input it will receive next.
- When reality differs from the prediction, the resulting **prediction error** is the signal that drives all learning and adaptation.
- The brain's overarching goal is to minimize prediction error -- either by updating its internal model (learning) or by acting on the world to make the prediction come true (active inference).

### Prediction Error as Universal Currency

In this crate, prediction error is not just one signal among many -- it is the *only* learning signal. It:

- **Modulates learning rates**: higher error --> faster weight updates in predictor and encoder
- **Drives exploration**: high error signals novelty --> the action policy increases exploration noise
- **Reinforces memory**: patterns that co-occur with low prediction error get strengthened
- **Guides prospection**: prediction confidence (inverse of error) controls how much weight the predicted future carries in action selection

There is no separate reward signal. There is no loss function designed by a human. The agent learns because its predictions are wrong, and prediction error is metabolically expensive.

### Homeostatic Feedback as the Only Evaluative Signal

The brain has no concept of "good" or "bad" built in. Instead, `habituate_homeo.wgsl` tracks whether internal variables (energy, physical integrity) are trending toward or away from stability. This gradient -- positive means improving, negative means worsening -- modulates:

- **Credit assignment**: the `predict_and_act.wgsl` pass uses the homeostatic gradient to assign credit/blame to recent actions in the 64-tick history ring
- **Urgency**: when energy or integrity drops critically low, urgency suppresses exploration in favor of exploitation

This is analogous to how biological organisms don't have explicit goals -- they have homeostatic set points, and deviations from those set points drive behavior.

### Capacity Constraints --> Emergent Cognition

The brain has finite resources:
- A fixed-size memory (`MEMORY_CAP = 128` patterns per agent)
- A per-tick recall budget (`RECALL_K = 16` top patterns)
- A fixed-dimension representation space (`DIM = 32`)

These constraints aren't limitations to be engineered around -- they are **generative**. Because the brain can't attend to everything, it must select. Because memory is finite, it must forget. Because the representation is compressed, it must abstract. These constraints give rise to attention, habit formation, chunking, and other cognitive phenomena without any of them being explicitly programmed.

---

## 2. Architecture Overview

```
xagent-brain/src/
  lib.rs          -- Re-exports, fast_tanh utility, BrainTelemetry stub
  gpu_brain.rs    -- GpuBrain struct, wgpu device/queue, 12 GPU buffers,
                     7 compute pipelines, tick/submit/collect API,
                     state read/write, death_signal, resize
  buffers.rs      -- All buffer layout constants, sensory packing,
                     initialization functions, AgentBrainState type

xagent-brain/src/shaders/
  feature_extract.wgsl   -- Pass 1: raw sensory --> features
  encode.wgsl            -- Pass 2: features --> encoded state
  habituate_homeo.wgsl   -- Pass 3: habituation + homeostasis
  recall_score.wgsl      -- Pass 4: cosine similarity scoring
  recall_topk.wgsl       -- Pass 5: top-K pattern selection
  predict_and_act.wgsl   -- Pass 6: prediction, credit, policy, fatigue
  learn_and_store.wgsl   -- Pass 7: weight updates, memory store, decay
```

All components are orchestrated by `GpuBrain`, which encodes all 7 shader dispatches into a single command encoder and submits them to the GPU in one `queue.submit()` call. The entire cognitive loop -- from raw sensory input to motor command output -- runs on GPU with no intermediate CPU synchronization.

---

## 3. Data Flow Diagram

```
                     ┌──────────────────────────────────────────────────────────────────┐
                     │                    GPU: 7 compute passes                         │
                     └──────────────────────────────────────────────────────────────────┘

SensoryFrame                                                                  MotorCommand
(packed on CPU                                                                (forward, turn)
 267 f32/agent)                                                                     |
      |                                                                             |
      v                                                                             |
┌─────────────┐  features   ┌──────────┐  encoded   ┌──────────────────┐            |
│   Pass 1:   │  (217 f32)  │  Pass 2: │  (32 f32)  │     Pass 3:      │            |
│   Feature   │────────────>│  Encode  │───────────>│  Habituate +     │            |
│   Extract   │             │          │            │  Homeostasis     │            |
└─────────────┘             └──────────┘            └────────┬─────────┘            |
                                                  habituated |  homeo_out           |
                                                  (32 f32)   |  (6 f32)             |
                                                             v                      |
                                                    ┌─────────────────┐             |
                                                    │    Pass 4:      │             |
                                                    │  Recall Score   │             |
                                                    │ (cosine sim x   │             |
                                                    │  128 patterns)  │             |
                                                    └────────┬────────┘             |
                                                   sims(128) |                      |
                                                             v                      |
                                                    ┌─────────────────┐             |
                                                    │    Pass 5:      │             |
                                                    │  Recall Top-K   │             |
                                                    │ (best 16 of 128)│             |
                                                    └────────┬────────┘             |
                                              recall_idx(17) |                      |
                                                             v                      |
                                                    ┌─────────────────┐             |
                                                    │    Pass 6:      │     decision|
                                                    │  Predict + Act  │─────────────┘
                                                    │ (predict, credit│     (motor +
                                                    │  policy, fatigue│      prediction +
                                                    │  exploration)   │      credit signal)
                                                    └────────┬────────┘
                                                   decision  |
                                                             v
                                                    ┌─────────────────┐
                                                    │    Pass 7:      │
                                                    │  Learn + Store  │
                                                    │ (grad descent,  │
                                                    │  Hebbian credit,│
                                                    │  memory reinf., │
                                                    │  pattern store, │
                                                    │  decay)         │
                                                    └─────────────────┘

  Persistent GPU buffers (live across ticks):
  ─── brain_state_buf ───  8,468 f32/agent  (encoder weights, predictor, habituation, homeo, action, fatigue)
  ─── pattern_buf ────────  5,251 f32/agent  (128 patterns: states, norms, reinforcement, motor, meta, active)
  ─── history_buf ────────  2,370 f32/agent  (64-entry action history ring: motor+state snapshots)

  Transient GPU buffers (overwritten each tick):
  sensory, features, encoded, habituated, homeo_out, similarities, recall_buf, decision
```

### Tick Execution Order

```
1. CPU packs SensoryFrames into flat f32 arrays (pack_sensory_frame)
2. CPU uploads sensory buffer to GPU                           (~52KB for 50 agents)
3. GPU Pass 1: feature_extract   (267 f32 --> 217 f32)        one thread per agent
4. GPU Pass 2: encode            (217 f32 --> 32 f32)         one thread per agent
5. GPU Pass 3: habituate_homeo   (habituation EMA + homeostatic gradient/urgency)
6. GPU Pass 4: recall_score      (cosine sim vs 128 patterns)
7. GPU Pass 5: recall_topk       (top-16 selection, metadata update)
8. GPU Pass 6: predict_and_act   (prediction error, credit assignment, policy eval,
                                  prospection, memory blend, exploration noise,
                                  motor fatigue, history recording)
9. GPU Pass 7: learn_and_store   (predictor gradient descent, encoder Hebbian credit,
                                  memory reinforcement, pattern storage, decay)
10. CPU reads back motor commands from decision buffer         (~800B for 50 agents)
```

---

## 4. GPU-Resident Design

The previous CPU architecture allocated brain state as heap objects. The GPU rewrite moves **all** brain state permanently onto GPU storage buffers. This eliminates the CPU<-->GPU marshaling bottleneck that would otherwise dominate per-tick cost.

### Per-Tick I/O Budget

| Direction | What | Size (50 agents) |
|-----------|------|-------------------|
| CPU --> GPU | Sensory frames | ~52 KB (`50 * 267 * 4 bytes`) |
| GPU --> CPU | Motor commands | ~800 B (`50 * 4 * 4 bytes`) |
| GPU only | Brain state, patterns, history | 0 bytes transferred |

The asymmetry is extreme and intentional: sensory data is the only thing the GPU doesn't already have, and motor commands are the only thing the CPU needs back. Everything else -- the 8,468 f32s of brain state per agent, the 5,251 f32s of pattern memory, the 2,370 f32s of action history -- stays on GPU permanently.

### Double-Buffered Readback

`GpuBrain` supports two usage patterns:

1. **Synchronous** (`tick(&frames) -> Vec<MotorCommand>`): Upload, dispatch all 7 passes, read back motor commands. Simple but blocks CPU until GPU finishes.

2. **Asynchronous** (`submit(&frames)` / `collect() -> Vec<MotorCommand>`): Pipeline the motor readback with the next frame's sensory packing. Two staging buffers alternate: while one is being filled by the GPU, the other is mapped for CPU reading. This hides readback latency behind computation.

### Buffer Allocation

All buffers are created at initialization (`GpuBrain::new`) with sizes proportional to agent count. The 4 persistent buffers use `STORAGE | COPY_SRC | COPY_DST` flags (allowing shader access plus CPU read/write for state I/O). The 8 transient buffers use `STORAGE` only.

---

## 5. Buffer Layout

All buffer offsets and stride constants are defined once in `buffers.rs` and auto-generated into WGSL via `wgsl_constants()`. This function emits a constants header that is prepended to every shader at pipeline creation time. The constants include all offsets, strides, and utility functions (`fast_tanh`, `pcg_hash`, `rand_f32`, `rand_normal`). Because both Rust and WGSL code derive from the same source of truth, offset mismatch bugs are impossible.

### Core Dimensions

| Constant | Value | Description |
|----------|-------|-------------|
| `DIM` | 32 | Internal representation dimensionality |
| `FEATURE_COUNT` | 217 | Extracted feature count (192 RGBA + 25 non-visual) |
| `MEMORY_CAP` | 128 | Maximum patterns per agent |
| `RECALL_K` | 16 | Top-K recalled patterns per tick |
| `ACTION_HISTORY_LEN` | 64 | Credit assignment lookback window |
| `ERROR_HISTORY_LEN` | 128 | Prediction error ring buffer size |

### Sensory Input Layout (CPU --> GPU)

```
[  192 RGBA vision  |  48 depth  |  vel(3)  fac(3)  ang(1)  e(1)  i(1)  ed(1)  id(1)  touch(16)  ]
 ^                   ^            ^                                                                ^
 0                   192          240                                                              267
```

Total: `SENSORY_STRIDE = 267` f32 per agent. `pack_sensory_frame()` handles the CPU-side packing, including sorting touch contacts by intensity and zero-padding.

### Brain State Buffer (per agent: `BRAIN_STRIDE = 8,468` f32)

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Encoder weights | `O_ENC_WEIGHTS = 0` | 6,944 | `FEATURE_COUNT * DIM` weight matrix |
| Encoder biases | `O_ENC_BIASES = 6944` | 32 | Per-dimension bias |
| Predictor weights | `O_PRED_WEIGHTS = 6976` | 1,024 | `DIM * DIM` prediction matrix |
| Predictor context weight | `O_PRED_CTX_WT = 8000` | 1 | Recall blending strength |
| Prediction error ring | `O_PRED_ERR_RING = 8001` | 128 | Error history |
| Habituation EMA | `O_HAB_EMA = 8131` | 32 | Per-dim change tracking |
| Habituation attenuation | `O_HAB_ATTEN = 8163` | 32 | Per-dim dampening factor |
| Previous encoded state | `O_PREV_ENCODED = 8195` | 32 | For habituation delta |
| Homeostasis state | `O_HOMEO = 8227` | 6 | `[grad_fast, grad_med, grad_slow, urgency, prev_energy, prev_integrity]` |
| Action forward weights | `O_ACT_FWD_WTS = 8233` | 32 | Policy weights for forward |
| Action turn weights | `O_ACT_TURN_WTS = 8265` | 32 | Policy weights for turn |
| Action biases | `O_ACT_BIASES = 8297` | 2 | `[fwd_bias, turn_bias]` |
| Exploration rate | `O_EXPLORATION_RATE = 8299` | 1 | Current exploration level |
| Fatigue rings | `O_FATIGUE_FWD_RING = 8300` | 128 | `[fwd(64), turn(64)]` |
| Fatigue state | `O_FATIGUE_CURSOR = 8428` | 3 | `[cursor, factor, length]` |
| Previous prediction | `O_PREV_PREDICTION = 8431` | 32 | For next-tick error |
| Tick count | `O_TICK_COUNT = 8463` | 1 | Agent-local tick counter |
| Heritable config | `O_HAB_SENSITIVITY = 8464` | 4 | `[hab_sens, max_curiosity, fatigue_recovery, fatigue_floor]` |

### Pattern Memory Buffer (per agent: `PATTERN_STRIDE = 5,251` f32)

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Pattern states | `O_PAT_STATES = 0` | 4,096 | `128 * 32` encoded state per pattern |
| Norms | `O_PAT_NORMS = 4096` | 128 | Cached L2 norm per pattern |
| Reinforcement | `O_PAT_REINF = 4224` | 128 | Strength (decays over time) |
| Motor context | `O_PAT_MOTOR = 4352` | 384 | `[forward, turn, outcome_valence] * 128` |
| Metadata | `O_PAT_META = 4736` | 384 | `[created_at, last_accessed, activation_count] * 128` |
| Active flags | `O_PAT_ACTIVE = 5120` | 128 | `1.0` = active, `0.0` = empty |
| Bookkeeping | `O_ACTIVE_COUNT = 5248` | 3 | `[active_count, min_reinf_idx, last_stored_idx]` |

### Action History Buffer (per agent: `HISTORY_STRIDE = 2,370` f32)

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Motor ring | `O_MOTOR_RING = 0` | 320 | `[forward, turn, tick, gradient, _pad] * 64` |
| State ring | `O_STATE_RING = 320` | 2,048 | `[encoded_state(32)] * 64` snapshots |
| Bookkeeping | `O_HIST_CURSOR = 2368` | 2 | `[cursor, length]` |

### Integer Storage Convention

Integer values (cursors, counts, tick counters) are stored as `f32` in GPU buffers and cast via `u32()` in WGSL. This is safe for exact integers up to 2^24 = 16,777,216, which is far beyond any practical tick count or buffer index.

---

## 6. Component Deep Dive: The 7-Pass Pipeline

### 6.1 Feature Extraction (Pass 1) -- `feature_extract.wgsl`

**What it does**: Transforms raw sensory input (267 f32 per agent) into a feature vector (217 f32 per agent). This is the first stage of the semantic firewall -- structured sensory data becomes a flat feature array.

**How it works**:

1. **Vision RGBA**: Direct copy of 192 values (8x6 grid, 4 channels each). No spatial pooling -- the full color+alpha grid is preserved. This gives the brain per-pixel color access, critical for learning that "red ahead = danger zone" and "green ahead = food zone."

2. **Velocity magnitude**: Computes `sqrt(vx^2 + vy^2 + vz^2)` from the 3-component velocity vector, collapsing direction into a single speed scalar.

3. **Proprioception**: Copies facing direction (3), angular velocity (1).

4. **Interoception**: Copies energy (1), integrity (1), energy delta (1), integrity delta (1).

5. **Touch contacts**: Copies 4 contact slots x 4 features = 16 values `[dir_x, dir_z, intensity, surface_tag/4]`.

**Feature layout**: `[192 RGBA | 1 speed | 3 facing | 1 angular | 1 energy | 1 integrity | 1 e_delta | 1 i_delta | 16 touch] = 217`

**Why depth is skipped**: The 48 depth values from the sensory frame are present in the upload buffer but not extracted as features. Vision RGBA already encodes biome-specific color, which is the primary discriminant. Depth would add dimensionality without proportional information gain for the current world complexity.

---

### 6.2 Encoding (Pass 2) -- `encode.wgsl`

**What it does**: Projects the 217-dimensional feature vector into a 32-dimensional encoded representation via a learned weight matrix and tanh nonlinearity. This is the **information bottleneck** -- 217 inputs compressed to 32 outputs, forcing the brain to learn what matters.

**How it works**:

```
encoded[d] = fast_tanh( sum_f( features[f] * weights[f * DIM + d] ) + biases[d] )
```

For each of the 32 output dimensions, the shader computes a weighted sum across all 217 features (column-major weight layout: `weights[f * DIM + d]`) plus a per-dimension bias, then squashes through `fast_tanh`.

**Weight initialization** (in `buffers::init_brain_state`): Xavier/Glorot uniform -- `uniform(-scale, scale)` where `scale = 1/sqrt(FEATURE_COUNT)`. This prevents tanh saturation at initialization.

**Weight layout**: The encoder weight matrix is stored column-major (`[FEATURE_COUNT x DIM]`, indexed as `[f * DIM + d]`). This layout means each output dimension's weights are scattered across memory at stride `DIM` -- not cache-optimal on CPU, but irrelevant on GPU where each invocation computes one agent's full encoding.

**Emergent property**: The encoder creates a **selectivity bottleneck**. 192 RGBA values + 25 non-visual features = 217 inputs compressed to 32 floats. What gets through this bottleneck is what the brain "pays attention to." The tanh squashing bounds all encoded values to [-1, 1], making cosine similarity a natural distance metric for downstream recall.

---

### 6.3 Habituation + Homeostasis (Pass 3) -- `habituate_homeo.wgsl`

Two independent subsystems combined into a single pass to reduce GPU dispatch count.

#### Habituation

**What it does**: Attenuates repetitive encoded dimensions and produces a habituated state used by all downstream passes.

**How it works**:

For each dimension of the encoded state:
1. **Per-dimension change EMA**: `ema[d] = (1 - alpha) * old_ema[d] + alpha * |encoded[d] - prev_encoded[d]|`
2. **Attenuation**: `atten[d] = clamp(ema[d] * sensitivity, ATTEN_FLOOR, 1.0)`
3. **Habituated state**: `habituated[d] = encoded[d] * atten[d]`

The curiosity bonus is not stored separately -- `predict_and_act.wgsl` computes it on-the-fly from the attenuation values stored in brain state: `curiosity = (1 - mean_atten) * max_curiosity_bonus`.

When all dimensions are changing rapidly, mean attenuation is high and curiosity is near zero. When the agent is stuck in a loop seeing the same thing, attenuation drops and curiosity rises, increasing exploration noise.

| Constant | Value | Source |
|----------|-------|--------|
| `HAB_EMA_ALPHA` | 0.02 | Hardcoded in shader |
| `ATTEN_FLOOR` | 0.1 | Hardcoded in shader |
| `habituation_sensitivity` | 20.0 (default) | Per-agent in brain_state, heritable |
| `max_curiosity_bonus` | 0.6 (default) | Per-agent in brain_state, heritable |

#### Homeostasis

**What it does**: Tracks the agent's internal physiological signals across three timescales and computes a composite gradient + non-linear urgency signal.

**How it works**:

1. **Raw gradient**: `raw = energy_delta * 0.6 + integrity_delta * 0.4`

2. **Three-timescale EMA tracking**:

   | Timescale | Alpha | Effective window | Purpose |
   |-----------|-------|------------------|---------|
   | Fast | 0.6 | ~5 ticks | Immediate reactions (flinch, grab) |
   | Medium | 0.04 | ~50 ticks | Short-term strategy (approach food, avoid threats) |
   | Slow | 0.004 | ~500 ticks | Long-term trends (is this environment safe?) |

3. **Composite gradient**: `base = fast * 0.50 + med * 0.35 + slow * 0.15`, then amplified by urgency: `gradient = base * (1 + urgency)`

4. **Distress curve**: `distress(level, exp) = min(pow(1 - clamp(level, 0.01, 1.0), exp) * 10.0, 10.0)`. The exponent is heritable via `BrainConfig::distress_exponent`.

5. **Urgency**: `(energy_distress + integrity_distress) * 0.5`

**Output** (`homeo_out` buffer, 6 f32 per agent): `[gradient, raw_gradient_amplified, urgency, grad_fast, grad_med, grad_slow]`

---

### 6.4 Recall Scoring (Pass 4) -- `recall_score.wgsl`

**What it does**: Computes cosine similarity between the habituated state and all 128 memory patterns. Inactive slots receive a sentinel score of `-2.0`.

**How it works**:

```
For each pattern j in [0, MEMORY_CAP):
    if not active: sim[j] = -2.0
    else: sim[j] = clamp(dot(habituated, pattern[j]) / (||habituated|| * ||pattern[j]||), -1.0, 1.0)
```

Pattern norms are pre-cached in `O_PAT_NORMS` (written during pattern storage in pass 7), avoiding redundant norm computation. The query norm is computed once per agent at the start of the pass.

**Why cosine similarity**: The encoder uses `tanh()`, so all values are in [-1, 1] -- magnitude carries less information than direction. Patterns with similar perceptual meaning should be similar regardless of activation strength.

---

### 6.5 Recall Top-K Selection (Pass 5) -- `recall_topk.wgsl`

**What it does**: Selects the best K=16 patterns from the 128 similarity scores. Each selected pattern is marked with `-3.0` in the similarities buffer to exclude it from subsequent iterations.

**How it works**:

A simple iterative argmax loop runs K times:
1. Find the slot with the highest similarity score.
2. If the best score is <= -1.5 (meaning only inactive/already-selected slots remain), stop early.
3. Record the slot index in the recall buffer, increment the count.
4. Update the pattern's `last_accessed` tick and `activation_count` metadata.
5. Write `-3.0` to the selected slot's similarity score to exclude it.

**Output** (`recall_buf`, 17 f32 per agent): `[idx_0, idx_1, ..., idx_15, count]`. The count is stored at position `RECALL_K` (index 16). Unused slots are zeroed.

---

### 6.6 Prediction + Action Selection (Pass 6) -- `predict_and_act.wgsl`

This is the largest and most complex pass (~360 lines). It combines what were previously 5 separate CPU components into a single GPU dispatch: prediction error computation, predictor matrix multiply, credit assignment, policy evaluation with prospection and memory blend, exploration noise, and motor fatigue.

#### 6.6.1 Prediction Error

Computes RMSE between the previous tick's prediction (stored in `O_PREV_PREDICTION`) and the current habituated state:

```
pred_error = sqrt( mean( (prev_prediction[d] - habituated[d])^2 ) )
```

The scalar error is recorded into the 128-entry error ring buffer for moving average computation.

#### 6.6.2 Predictor

Predicts the next encoded state from the current habituated state and recalled context:

1. **Linear transform**: `prediction[i] = sum_j( habituated[j] * pred_weights[i * DIM + j] )`
2. **Context blend**: If recalled patterns exist, blend them in weighted by similarity: `prediction[d] += context_weight * sim * pattern[d] / total_sim`
3. **Nonlinearity**: `prediction[d] = fast_tanh(prediction[d])`

The `context_weight` parameter (stored at `O_PRED_CTX_WT`, initialized to 0.15) controls how much recalled patterns influence the prediction. It is itself adapted in pass 7.

#### 6.6.3 Credit Assignment

The homeostatic gradient is used to assign credit/blame to recent actions in the 64-entry history ring buffer. For each recorded action:

1. **Temporal decay**: `temporal = exp(-age * CREDIT_DECAY)` where `CREDIT_DECAY = 0.04`. Actions older than ~60 ticks contribute negligibly.
2. **Improvement signal**: `improvement = current_gradient - recorded_gradient`. If `|improvement| < DEADZONE (0.01)`, skip (filters metabolic noise).
3. **Pain amplification**: Negative improvements are multiplied by `PAIN_AMP = 3.0`, reflecting the biological reality that aversive stimuli produce stronger learning signals.
4. **State-conditioned weight update**: `weights[d] += WEIGHT_LR * credit * recorded_motor * recorded_state[d]`. The recorded state snapshot from when the action was taken ensures credit is attributed to the correct sensory context.

**Weight normalization**: After credit assignment, forward and turn weight vectors are clipped to L2 norm <= `MAX_WEIGHT_NORM = 2.0` (synaptic homeostasis).

#### 6.6.4 Policy Evaluation

Continuous motor output is computed as a dot product of learned weights with the habituated state:

```
fwd = dot(fwd_weights, habituated) + fwd_bias
trn = dot(trn_weights, habituated) + trn_bias
```

This is a continuous-output linear policy -- no discrete action table. The agent learns which features predict beneficial forward motion and which predict beneficial turning.

#### 6.6.5 Prospective Evaluation

The policy weights are applied not just to the current state but also to the predicted future state:

```
fwd += confidence * ANTICIPATION_WEIGHT * (fwd_future - fwd)
trn += confidence * ANTICIPATION_WEIGHT * (trn_future - trn)
```

This is delta-based: it measures the *change* between current and predicted preferences, not the absolute predicted value. This eliminates fixed-point convergence problems. `confidence = 1 - clamp(pred_error, 0, 1)` -- low accuracy means short effective horizon.

#### 6.6.6 Memory-Informed Motor Blend

Recalled patterns contribute their stored motor commands weighted by similarity and outcome valence:

```
mem_fwd = sum( sim * valence * stored_forward ) / sum( |sim * valence| )
mix = clamp(mean_|sim*valence|, 0, 1) * 0.4
fwd = fwd * (1 - mix) + mem_fwd * mix
```

- **Positive valence**: "do what I did before" -- reinforces the recalled motor command
- **Negative valence**: "do the opposite" -- the sign flip steers away from past mistakes
- Memory contributes up to 40% of motor signal

#### 6.6.7 Exploration Noise

Exploration rate is computed dynamically:

```
novelty_bonus = min(pred_error * 2.0, 0.4)
urgency_penalty = min(urgency * 0.4, 0.5)
policy_confidence = clamp((|fwd| + |trn|) / 2.0, 0.0, 1.0)
exploration_rate = clamp(0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity - urgency_penalty, 0.10, 0.85)
```

Gaussian noise scaled by `exploration_rate` is added to the motor output, using `pcg_hash` GPU RNG seeded by `(agent * 1000 + tick)`. The motor output is clamped to [-1, 1].

#### 6.6.8 Motor Fatigue

A ring buffer of recent motor outputs (forward and turn separately) tracks motor variance. Low variance means repetitive output, which triggers dampening:

```
motor_variety = sqrt(var_fwd + var_trn) * recovery_sensitivity
fatigue_factor = clamp(floor + (1 - floor) * clamp(motor_variety, 0, 1), floor, 1.0)
fwd *= fatigue_factor
trn *= fatigue_factor
```

When motor output is varied, fatigue factor is near 1.0 (no dampening). When output is repetitive, the factor drops toward `fatigue_floor`, weakening the command and giving other motor patterns a chance.

| Constant | Value | Source |
|----------|-------|--------|
| `fatigue_recovery_sensitivity` | 8.0 (default) | Per-agent, heritable |
| `fatigue_floor` | 0.1 (default) | Per-agent, heritable |

#### 6.6.9 Output Recording

After computing final motor output:
1. Records `[fwd, trn, tick, gradient, _pad]` to the action history ring at `O_MOTOR_RING`.
2. Records the habituated state snapshot at `O_STATE_RING` for future credit assignment.
3. Saves the prediction to `O_PREV_PREDICTION` for next tick's error computation.
4. Increments `O_TICK_COUNT`.
5. Writes `[prediction(32), credit_signal(32), fwd, trn, strafe, _pad]` to the decision buffer for pass 7 and CPU readback.

---

### 6.7 Learning + Memory Storage (Pass 7) -- `learn_and_store.wgsl`

Five learning operations packed into a single pass, using the prediction and credit signal from the decision buffer written by pass 6.

#### 6.7.1 Predictor Gradient Descent

Online gradient descent on prediction error:

```
error_vec[d] = prediction[d] - habituated[d]
for each (i, j):
    grad = clamp(error_vec[i] * (1 - prediction[i]^2) * habituated[j], -1.0, 1.0)
    weights[i * DIM + j] -= learning_rate * grad
    weights[i * DIM + j] = clamp(weights[i * DIM + j], -3.0, 3.0)
```

The `(1 - prediction^2)` term is the tanh derivative, correctly accounting for the nonlinearity. Gradient clipping at +/-1.0 prevents instability.

The context weight is also adapted: `ctx_wt += learning_rate * 0.01 * (error_mag - 0.5)`. If error > 0.5, context weight increases (the predictor needs more help from memory). If < 0.5, it decreases.

#### 6.7.2 Encoder Hebbian Credit Adaptation

The encoder weights are adapted based on the credit signal from pass 6:

```
for each (i, j) where |credit_signal[i]| > 1e-6:
    weights[j * DIM + i] += learning_rate * credit_signal[i] * 0.001 * features[j]
    weights[j * DIM + i] = clamp(weights, -2.0, 2.0)
```

This is a Hebbian-style update: features that co-occur with strong credit signals have their encoder weights strengthened. The 0.001 scale factor makes encoder adaptation much slower than predictor learning, reflecting the intuition that the perceptual representation should change gradually while the prediction model adapts quickly.

#### 6.7.3 Memory Reinforcement

Active patterns with cosine similarity > 0.3 to the current habituated state are reinforced:

```
reinforcement[j] += sim * learning_rate * (1 - pred_error)
```

Low prediction error strengthens matching patterns more -- successful prediction means the memory is accurate.

**Retroactive valence update**: Similar patterns have their `outcome_valence` nudged toward the current homeostatic gradient via an EMA: `valence += sim * (learning_rate * 0.3) * (gradient - valence)`. This lets the agent update its assessment of past situations.

#### 6.7.4 Pattern Storage

Each tick, the current habituated state is stored to the weakest memory slot (the one with the lowest reinforcement, tracked at `O_MIN_REINF_IDX`):

- State vector and cached norm are written to the pattern slot.
- Motor context `[fwd, trn, raw_gradient]` is stored.
- Metadata `[created_at, last_accessed, activation_count]` is initialized.
- Active flag is set to 1.0.
- Active count is incremented if the slot was previously empty.

#### 6.7.5 Memory Decay

For each active pattern, an effective decay rate is computed that is modulated by:
- **Frequency**: `freq_factor = 1 / (1 + activation_count * 0.2)` -- frequently accessed patterns decay slower
- **Recency**: `recency_factor = min((tick - last_accessed) / 100, 3.0)` -- recently accessed patterns decay slower
- **Combined**: `effective_rate = base_decay * freq_factor * (0.2 + recency_factor)`

Patterns whose reinforcement drops to zero are deactivated. After decay, the slot with the minimum reinforcement is identified and cached in `O_MIN_REINF_IDX` for next tick's storage target.

---

## 7. Emergent Phenomena

None of these behaviors are explicitly programmed. They arise from the interaction of the 7 shader passes and their shared constraints:

| Phenomenon | How It Emerges | Contributing Passes |
|------------|---------------|---------------------|
| **Attention** | Memory capacity (128) forces selective recall; encoder bottleneck (217 --> 32) compresses information | Pass 2, Pass 4-5 |
| **Fear / Avoidance** | Negative homeostatic gradient --> pain amplifier (3x) makes damage signal loud --> credit assignment blames recent actions via state snapshots --> policy weights learn to avoid danger-associated features --> prospective evaluation applies these weights to the predicted future, anticipating danger before entering it | Pass 3, 6 (credit + prospection) |
| **Curiosity** | High prediction error in safe situations --> exploration noise increases; habituation produces a curiosity bonus when input is monotonous, further boosting exploration | Pass 3 (habituation), Pass 6 (exploration) |
| **Habit Formation** | Repeated successful actions build strong policy weights --> exploitation ratio increases --> behavior becomes automatic | Pass 6 (credit), Pass 7 (reinforcement) |
| **Startle / Surprise** | Sudden prediction error spike --> novelty bonus increases --> exploration spikes | Pass 6 (error + exploration) |
| **Adaptation** | Prediction error decreases in stable environments --> exploration drops --> behavior stabilizes | Pass 6, Pass 7 (predictor learning) |
| **Panic** | Low energy/integrity --> high urgency --> exploration suppressed --> agent falls back on policy weights | Pass 3 (urgency), Pass 6 (exploration) |
| **Forgetting** | Patterns not recalled or reinforced decay below zero and are deactivated | Pass 7 (decay) |
| **Boredom / Loop Breaking** | Monotonous input --> habituation attenuates repetitive dimensions --> curiosity rises --> exploration increases; simultaneously, low motor variance --> fatigue dampens output --> repeated action weakens | Pass 3 (habituation), Pass 6 (fatigue + exploration) |
| **Contextual Memory** | Policy weights map encoded features to motor preferences -- different percepts trigger different behaviors; recalled patterns blend motor advice | Pass 6 (policy + memory blend) |
| **Desensitization** | The slow EMA timescale integrates gradual changes; constant mild negative gradient eventually stops triggering strong reactions | Pass 3 (homeostasis) |

---

## 8. Host API (gpu_brain.rs)

### GpuBrain

```rust
pub struct GpuBrain {
    // wgpu device + queue
    // 4 persistent buffers (brain_state, pattern, history, config)
    // 8 transient buffers (sensory, features, encoded, habituated, homeo_out, similarities, recall, decision)
    // 2 staging buffers for double-buffered motor readback
    // 7 compute pipelines + bind groups
}
```

### Public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_available` | `() -> bool` | Static probe: returns true if any wgpu adapter (real GPU or software fallback) exists. Used by tests to skip GPU work on headless CI |
| `new` | `(agent_count: u32, config: &BrainConfig) -> Self` | Creates wgpu device (tries real GPU first, falls back to CPU/software adapter for headless CI), allocates all buffers, compiles all 7 shaders with auto-generated constants header |
| `tick` | `(&mut self, frames: &[SensoryFrame]) -> Vec<MotorCommand>` | Synchronous: upload, dispatch all 7 passes, readback |
| `submit` | `(&mut self, frames: &[SensoryFrame])` | Async step 1: upload sensory + dispatch all passes + copy motor to staging |
| `collect` | `(&mut self) -> Vec<MotorCommand>` | Async step 2: map staging buffer, read motor commands |
| `read_agent_state` | `(&mut self, index: u32) -> AgentBrainState` | Download one agent's full brain state from GPU for inspection/inheritance |
| `write_agent_state` | `(&mut self, index: u32, state: &AgentBrainState)` | Upload one agent's brain state to GPU (for offspring initialization) |
| `death_signal` | `(&mut self, index: u32)` | Halves all pattern reinforcements, resets homeostasis + exploration |
| `resize` | `(&mut self, agent_count: u32)` | Rebuilds all buffers and pipelines for a new agent count |

### AgentBrainState

```rust
pub struct AgentBrainState {
    pub brain_state: Vec<f32>,   // BRAIN_STRIDE = 8,468 f32
    pub patterns: Vec<f32>,      // PATTERN_STRIDE = 5,251 f32
    pub history: Vec<f32>,       // HISTORY_STRIDE = 2,370 f32
}
```

Used for cross-generation inheritance (the governor reads parent state, mutates it, writes to offspring), mutation, and DB persistence. The three vectors are the exact GPU buffer contents for one agent slice.

### death_signal Behavior

`death_signal(index)` performs three operations:

1. **Trauma**: Halves all pattern reinforcements. Patterns with reinforcement below 0.5 are deactivated. The weakest memories are wiped while the strongest survive.
2. **Homeostasis reset**: Zeros all gradient EMAs (`grad_fast`, `grad_med`, `grad_slow`), urgency, and previous energy/integrity. The respawned agent starts with no homeostatic memory.
3. **Exploration reset**: Sets exploration rate to 0.5 (balanced start).

### BrainTelemetry

Stub telemetry struct for UI/recording compatibility. Populated with zero/default values since per-tick GPU readback of all telemetry fields would negate the performance benefits of GPU residence. Key fields: `prediction_error`, `memory_utilization`, `exploration_rate`, `homeostatic_gradient`, `homeostatic_urgency`, `fatigue_factor`, etc.

The `behavior_phase()` method classifies agent state:

| Phase | Composite Score | Interpretation |
|-------|----------------|----------------|
| `RANDOM` | < 2% | Brain is mostly exploring randomly |
| `EXPLORING` | 2-8% | Starting to learn, still exploring heavily |
| `LEARNING` | 8-20% | Learning is working, composite score increasing |
| `ADAPTED` | >= 20% | Brain has adapted to its environment |

---

## 9. Configuration (BrainConfig)

The `BrainConfig` struct (defined in `xagent-shared`) provides heritable parameters. Fixed dimensions (`DIM`, `FEATURE_COUNT`, `MEMORY_CAP`, `RECALL_K`) are constants in `buffers.rs`. Tunable parameters are stored per-agent in the brain state buffer and passed to shaders via the config uniform:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `learning_rate` | 0.05 | Base rate for predictor gradient descent, memory reinforcement, encoder credit |
| `decay_rate` | 0.001 | Pattern reinforcement decay per tick |
| `distress_exponent` | 2.0 | Urgency curve steepness (heritable, range [1.5, 5.0]) |
| `habituation_sensitivity` | 20.0 | How fast attenuation responds to change (heritable) |
| `max_curiosity_bonus` | 0.6 | Maximum exploration boost from sensory monotony (heritable) |
| `fatigue_recovery_sensitivity` | 8.0 | How fast fatigue lifts when motor output diversifies (heritable) |
| `fatigue_floor` | 0.1 | Minimum motor output under full fatigue (heritable) |

### Parameter Effects

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| `learning_rate` | Slow adaptation, stable but takes longer to respond | Fast adaptation, responsive but risks oscillation |
| `decay_rate` | Long memory retention, can fill memory with stale data | Aggressive forgetting, only keeps recent/frequent patterns |
| `distress_exponent` | Reacts sooner to moderate health drops, more cautious | Stays calm longer, but panics harder at critical levels |
| `habituation_sensitivity` | Slow to bore, tolerates repetitive input longer | Bores quickly, curiosity bonus rises fast |
| `max_curiosity_bonus` | Weak exploration boost from monotony, loops persist | Strong exploration boost, breaks loops aggressively |
| `fatigue_recovery_sensitivity` | Slow fatigue recovery, dampening lingers | Fast recovery, dampening lifts immediately with diverse output |
| `fatigue_floor` | Motor output nearly zeroed by fatigue, strong loop-breaking | Motor output stays substantial under fatigue, gentler |

### Tuning Guide

| Problem | Likely Cause | Try |
|---------|-------------|-----|
| Agent is too random / never settles | Learning rate too low | Increase `learning_rate` to 0.08-0.1 |
| Agent gets stuck doing one thing | Learning rate too high or decay too low | Decrease `learning_rate`, increase `decay_rate` |
| Agent ignores visual changes | Encoder weights adapting too slowly | Increase `learning_rate` (credit signal scales with it) |
| Agent panics too early | Urgency kicks in at moderate levels | Increase `distress_exponent` (e.g., 3.0-4.0) |
| Agent stuck in loops | Repetitive sensory + motor without breaking free | Increase `habituation_sensitivity` and `max_curiosity_bonus`, decrease `fatigue_floor` |
| Memory fills with stale data | Decay too slow | Increase `decay_rate` |

---

## 10. Testing

### Philosophy

Tests verify **behavioral properties**, not implementation details. They check things like:
- "the encode pass produces tanh-bounded output" (the shader computes correctly)
- "habituation attenuates repeated input" (the system detects monotony)
- "learning changes weights" (gradient descent actually runs)
- "agents produce varied motor output" (the system isn't degenerate)

### Running Tests

```bash
cargo test -p xagent-brain --lib
```

### Test Categories (21 tests)

| Module | Tests | What They Verify |
|--------|-------|-----------------|
| `buffers` | 6 | Sensory stride matches feature count, brain/pattern/history stride consistency, pack_sensory_frame fills buffer with finite values, init_brain_state produces correct-length output |
| `gpu_brain` (buffer/init) | 2 | GPU brain initializes without panic, read/write state roundtrip preserves data |
| `gpu_brain` (per-shader) | 7 | Feature extraction produces correct output, encode produces tanh-bounded output, habituation attenuates repeated input, recall_score computes cosine similarity, recall_topk selects best patterns, predict_and_act produces valid motor, learn_and_store modifies weights and stores patterns |
| `gpu_brain` (integration) | 4 | Full tick produces valid motor commands, learning changes weights over 50 ticks, memory fills over time, resize changes agent count |
| `gpu_brain` (behavioral) | 2 | Multi-agent variance (50 agents produce non-degenerate output), death_signal halves reinforcement |

---

## 11. Design Decisions

### Why GPU-Resident Over CPU

The previous CPU implementation had one `Brain` struct per agent with heap-allocated weight matrices, pattern vectors, and ring buffers. At 50+ agents, the per-tick cost was dominated by:
- 50 independent matrix multiplies (encoder: 217x32, predictor: 32x32)
- 50 * 128 cosine similarity computations (recall scoring)
- Scattered memory access patterns (each agent's data in different heap locations)

Moving to GPU makes all 50 agents' matrix multiplies a single dispatch. More importantly, all brain state lives in contiguous GPU buffers with computed strides, eliminating pointer chasing entirely. The CPU's only job is packing sensory frames and reading motor commands.

### Why 7 Passes Instead of 1

A single monolithic compute shader would be simpler but would require all intermediate data (features, encoded state, similarities, recall indices) to live in per-thread registers or shared memory. With 7 passes, each shader reads from and writes to global storage buffers, keeping the working set per invocation small. The pipeline barriers between passes are cheap (all dispatches go into a single command encoder), and the data locality benefits of specialized shaders outweigh the synchronization cost.

### Why Flat array<f32> Instead of Structured Buffers

WGSL's structured buffer support requires compile-time-known layouts. With per-agent strides computed from constants (e.g., `agent * BRAIN_STRIDE + O_ENC_WEIGHTS`), a flat `array<f32>` with computed offsets is simpler and more flexible than nested structs. The offset constants are auto-generated from Rust, so the "indexing math" is actually just named constants that read like field accesses.

### Why wgsl_constants() Auto-Generation

Every shader needs the same set of 50+ offset constants, utility functions (`fast_tanh`, `pcg_hash`), and dimension values. Manually keeping these in sync between Rust and WGSL would be a maintenance nightmare. `wgsl_constants()` generates the shared header from Rust constants, which is prepended to each shader source at pipeline creation time. A single source of truth, zero chance of offset mismatch.

### Why Cosine Similarity for Pattern Matching

Cosine similarity measures angle between vectors, ignoring magnitude. This is correct for encoded states because:
- The encoder uses `tanh()`, so all values are in [-1, 1] -- magnitude carries less information than direction.
- Patterns with similar perceptual meaning should be similar regardless of activation strength.
- It's cheap to compute (dot product + two norms) and well-understood.

### Why Three Timescales in Homeostasis

Biological nervous systems track changes at multiple timescales -- immediate reflexes (milliseconds), emotional responses (seconds-minutes), and mood/disposition (hours-days). Three timescales (fast ~5 ticks, medium ~50 ticks, slow ~500 ticks) capture this hierarchy:
- **Fast**: "I just got hit" -- immediate reaction
- **Medium**: "This area has been bad for me" -- tactical adjustment
- **Slow**: "My overall strategy isn't working" -- strategic shift

### Why Continuous Motor Output Instead of Discrete Actions

The old CPU architecture used a discrete 8-action space with a learned preference table. The GPU rewrite replaces this with continuous forward/turn output computed as `dot(weights, habituated) + bias`. This is both simpler (no action table, no softmax, no argmax tie-breaking) and more expressive (the agent can move at any speed and turn at any angle). Credit assignment updates the weight vectors directly via state-conditioned gradient, which is more natural for continuous outputs.

### Why Credit Deadzone Filters Metabolic Noise

Every tick, energy depletion produces a small negative homeostatic gradient (~0.006). Without a deadzone, this constant signal triggers credit assignment on every tick, treating normal metabolism as a negative outcome. The `DEADZONE = 0.01` threshold ensures only meaningful events -- food consumption (+0.03), damage (-0.02), death (-0.5) -- produce weight updates. This is analogous to sensory gating in biological systems, where constant background stimuli are filtered out to preserve signal clarity.

### Why State-Conditioned Credit Assignment

Naive credit assignment applies the same credit to all recent actions regardless of the state in which they were taken. This means dying in a danger zone penalizes actions taken in safe states equally -- the agent learns "forward is bad everywhere" instead of "forward-in-danger is bad." State-conditioned credit uses the recorded state snapshot from when each action was chosen, so the weight update is proportional to `recorded_state[d]`. High feature activation at action time --> full credit. Low activation --> minimal credit.

### Why Pain Amplification (3x)

Negative homeostatic gradients receive a 3x multiplier in credit assignment. This reflects the biological reality that amygdala neurons respond 2-3x more strongly to aversive stimuli. This is NOT hardcoded avoidance -- the brain must learn WHAT to do about the amplified signal. The amplification just ensures that negative outcomes produce louder learning signals than positive ones, which is necessary because damage is typically more catastrophic than the benefit of a single meal.

### Why Encoder Adaptation is Hebbian, Not Backpropagated

Backpropagating prediction error through the predictor and into the encoder weights would be the "correct" gradient. But on GPU, the encoder and predictor run in separate passes (2 and 7), and computing the full chain would require storing intermediate Jacobians. Instead, the encoder weights are adapted via a Hebbian credit signal from pass 6: features that co-occur with strong credit (positive or negative) have their encoder weights nudged. This is biologically plausible, computationally cheap, and empirically sufficient -- the encoder's main job is dimensionality reduction, and Xavier-initialized random projections already do a decent job of that.

---

## 12. Known Limitations & Future Work

### Current Limitations

- **Single-threaded workgroups**: Each shader dispatches one thread per agent (`@workgroup_size(1)`). For the encode pass (217x32 matrix multiply) and recall scoring (128 cosine similarities), parallelizing across dimensions or pattern slots within a workgroup would yield significant speedups.

- **No hierarchical pattern abstraction**: All 128 patterns are stored at the same level of abstraction. There is no mechanism for forming higher-order patterns ("I'm in a corridor" from a sequence of wall-patterns) or chunking temporal sequences into reusable units.

- **Fixed memory capacity**: The brain cannot grow its memory. A dynamic capacity that expands in rich environments and contracts in simple ones would better match biological memory allocation.

- **No inter-agent brain communication**: Each brain is entirely isolated. There is no mechanism for one agent to share learned patterns or action values with another. Social learning and cultural transmission would require some form of brain-to-brain communication channel.

- **No sleep/consolidation**: Biological brains consolidate memories during sleep, replaying and strengthening important patterns. The current system has no offline consolidation phase -- all learning happens online during ticks.

- **Telemetry is stub-only**: Because brain state lives on GPU, per-tick telemetry readback is not implemented (it would negate the performance benefit). The `BrainTelemetry` struct returns zero/default values. A future sampling approach could read telemetry for a subset of agents at reduced frequency.

- **Depth features unused**: The 48 depth values from the sensory frame are uploaded but not extracted as features. Adding depth as a feature channel would improve spatial reasoning at the cost of a larger encoder weight matrix.

### Future Directions

- **Workgroup parallelism**: Use `@workgroup_size(32)` or similar for the encode and recall passes, with workgroup-level reductions for dot products and argmax operations.
- **Hierarchical temporal memory**: Stack multiple levels of pattern memory, each operating at a different temporal granularity.
- **Multi-agent pattern sharing**: Allow agents to "teach" each other by sharing association strengths or pattern representations.
- **Dreaming/replay**: Periodically replay stored pattern sequences during idle time to consolidate important memories and prune irrelevant ones.
- **Telemetry sampling**: Read full brain state for 1-2 agents per tick to populate telemetry without bottlenecking the pipeline.
- **Depth feature integration**: Extract depth as a separate feature channel for improved spatial awareness.
