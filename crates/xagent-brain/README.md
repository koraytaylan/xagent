# xagent-brain

A general-purpose cognitive architecture based on **predictive processing**, running entirely on GPU inside a fused compute kernel.

The brain crate is the decision-making core of each xagent. It has no hardcoded goals or domain modules -- no "hunger module", no "fear module", no reward function. The one exception is a single reactive substrate: `brain_passes.wgsl` applies a hardcoded klinotaxis turn-gain modulator (`klinotaxis_factor`, scaled by `KLINOTAXIS_SENSITIVITY` in `common.wgsl`) that lets prediction-error gradient nudge turn direction before policy learning has anything to say. Beyond that scaffolding, everything the agent does emerges from a single loop and a single principle:

> **Prediction error drives everything.**

```
sense --> extract --> encode --> habituate/homeo --> recall --> predict+act --> learn+store
  |         |           |              |                |           |              |
  |    coop_feature_    |    coop_habituate_      coop_recall_  coop_predict_  coop_learn_
  |      extract     coop_     homeo              score+topk      and_act      and_store
  |                  encode
  |
  | All seven coop_* stages run inside `kernel_tick.wgsl` (or `brain_tick.wgsl` for
  | brain-only entry points). One queue.submit() runs N fused ticks per batch.
```

`GpuKernel` is the sole runtime. There is no CPU-side brain object, no `Brain::tick(frame) -> MotorCommand` API, and no swappable cognitive-architecture trait. At startup the CPU uploads world geometry (terrain heightmap, biome grid, food positions/timers/consumed flags) and the initial per-agent physics rows; thereafter the only thing the CPU writes per dispatch is the world-config uniform (`start_tick`, `ticks_to_run`, `vision_stride`, `brain_tick_stride`, phase mask). `kernel.dispatch_batch(start_tick, ticks_to_run)` splits the requested ticks into full kernel-batches of `vision_stride * brain_tick_stride` ticks, plus a shorter remainder kernel-batch (`remainder_cycles * brain_tick_stride` ticks) when `brain_cycles % vision_stride != 0`, plus an optional physics-only remainder of `ticks_to_run % brain_tick_stride` ticks for the trailing fragment that does not fill a brain cycle. Each kernel-batch submits one command buffer running the four passes `prepare → kernel → global → vision`; the physics-only remainder submits a single physics pass with the brain/vision phases masked off. The brain stage in step 2 reads its sensory inputs from the `sensory_buffer` that the *previous* batch's vision pass wrote — a one-batch sensory lag that lets the costly global+vision work amortize over `vision_stride` brain cycles. All persistent brain state lives permanently in GPU storage buffers; the only data that crosses the bus while the simulation is running is whatever the CPU side asks the kernel to read back (typically position/vitals for the UI and a per-frame telemetry snapshot for the selected agent).

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [GPU-Resident Design](#4-gpu-resident-design)
5. [Buffer Layout](#5-buffer-layout)
6. [Component Deep Dive: The 7 Brain Stages](#6-component-deep-dive-the-7-brain-stages)
   - [6.1 Feature Extraction](#61-feature-extraction--coop_feature_extract)
   - [6.2 Encoding](#62-encoding--coop_encode)
   - [6.3 Habituation + Homeostasis](#63-habituation--homeostasis--coop_habituate_homeo)
   - [6.4 Recall Scoring](#64-recall-scoring--coop_recall_score)
   - [6.5 Recall Top-K Selection](#65-recall-top-k-selection--coop_recall_topk)
   - [6.6 Prediction + Action Selection](#66-prediction--action-selection--coop_predict_and_act)
   - [6.7 Learning + Memory Storage](#67-learning--memory-storage--coop_learn_and_store)
7. [Emergent Phenomena](#7-emergent-phenomena)
8. [Host API (gpu_kernel.rs)](#8-host-api-gpu_kernelrs)
9. [Configuration (BrainConfig)](#9-configuration-brainconfig)
10. [Testing](#10-testing)
11. [Design Decisions](#11-design-decisions)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## The Brain Has No Eyes

The most important thing to understand about this architecture: **the brain has zero semantic knowledge of its inputs**.

In the live runtime sensory features are produced on-GPU: the vision pass raycasts into `sensory_buffer` (default 8×6: `SENSORY_STRIDE = 267` f32 = 192 RGBA + 48 depth + 27 non-visual), and the brain's first stage `coop_feature_extract` in `src/shaders/kernel/brain_passes.wgsl` projects it into the feature vector (`BrainLayout::feature_count = VISION_RAYS * 5 + 25`, = 265 f32 for the default 8×6). The CPU-side `buffers::pack_sensory_frame()` is the one place that still touches named fields -- and even there the flattening is not free of inductive bias: the function fixes the modality layout (vision color, depth, proprioception, interoception, touch -- in a known positional order), keeps the top 4 touch contacts by intensity, and preserves `TouchContact::surface_tag` as a scalar category channel (`surface_tag as f32 / 4.0`, with concrete tags `TOUCH_FOOD`, `TOUCH_TERRAIN_EDGE`, `TOUCH_HAZARD`, `TOUCH_AGENT`). The shader boundary strips the struct labels; the packer chose the layout. From `coop_feature_extract` onward, the brain operates on opaque numerical vectors -- no concept of "vision," no awareness of "eyes," no understanding that index 47 was once an RGBA pixel and index 73 was once an energy level.

```
World --> SensoryFrame --> pack_sensory_frame() --> [267 f32] --> coop_feature_extract --> [265 f32]
               |                                                        |
      Named fields like                                        Brain sees only a
      "vision", "energy"                                       flat array<f32>
```

Consider what happens when another agent -- say, a magenta-colored one -- enters the visual field. The brain doesn't receive "agent detected" or "entity of type Agent at bearing 30 degrees." It experiences indices 12--15 shifting from `[0.3, 0.6, 0.2, 1.0]` to `[0.9, 0.2, 0.6, 1.0]`. Simultaneously, a touch contact might add nonzero values at indices 199--202 (direction, intensity, tag). The brain has no legend for any of this. It doesn't know that `surface_tag=4` means "agent." It doesn't know that the shifted values represent magenta. Over hundreds of ticks, if this pattern of input correlates with energy dropping (food competition), the brain discovers -- through prediction error and homeostatic gradient alone -- that "those numerical patterns are bad for me." The concept of "that's a competitor" *emerges* from experience, not from labels.

This is the fundamental difference from traditional AI systems. There are no reward functions hand-crafted by engineers. No labeled feature vectors telling the model "this is vision, this is hunger." But the picture is not bias-free either: the packer's fixed modality layout and the preserved `surface_tag` channel are hand-chosen priors that ride along with the otherwise opaque vector. The honest summary is that `pack_sensory_frame()` strips struct labels but keeps positional structure; what the brain then sees is a numerically flattened interface, not a pristine raw signal. What remains downstream is prediction + homeostatic gradient + experience, and from these ingredients combined with that bounded prior, all meaning is discovered.

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
- A fixed-dimension representation space (`ENCODED_DIMENSION = 128`)

These constraints aren't limitations to be engineered around -- they are **generative**. Because the brain can't attend to everything, it must select. Because memory is finite, it must forget. Because the representation is compressed, it must abstract. These constraints give rise to attention, habit formation, chunking, and other cognitive phenomena without any of them being explicitly programmed.

---

## 2. Architecture Overview

```
xagent-brain/src/
  lib.rs              -- Re-exports: GpuKernel, AgentTelemetry, AgentBrainState,
                         BrainLayout, fast_tanh, BrainTelemetry (stub for UI)
  gpu_kernel.rs       -- GpuKernel: fused dispatch, shader composition,
                         async state/telemetry readback, state read/write
  async_readback.rs   -- ReadbackTracker state machine (non-blocking collection
                         of mapped staging buffers)
  buffers.rs          -- Buffer layout constants, sensory packing,
                         initialization functions, AgentBrainState, BrainLayout

xagent-brain/src/shaders/kernel/   -- All shader fragments live here.
  common.wgsl                   -- Shared constants, override cascade
                                   (VISION_W/VISION_H), utility functions
  brain_passes.wgsl             -- Cooperative brain-stage functions:
                                   coop_feature_extract, coop_encode,
                                   coop_habituate_homeo, coop_recall_score,
                                   coop_recall_topk, coop_predict_and_act,
                                   coop_learn_and_store
  brain_tick.wgsl               -- Brain-only entry point (one tick, no physics).
                                   Composed with brain_passes for the `brain`
                                   pipeline used by unit tests.
  kernel_tick.wgsl              -- Per-agent fused kernel entry. Composed with
                                   brain_passes for the `kernel` pipeline. Loops
                                   over vision_stride cycles internally.
                                   Dispatch: (agent_count, 1, 1), 256 threads/workgroup.
  global_tick.wgsl              -- Spatial-grid rebuild (food + agent), food
                                   respawn/timer updates, and pairwise agent
                                   collision resolution.
                                   Dispatch: (1, 1, 1).
  physics_tick.wgsl             -- Physics-only stride entry.
  vision_tick.wgsl              -- Vision-only stride entry.
  phase_*.wgsl                  -- Reusable phase fragments concatenated into
                                   the entry shaders at composition time
                                   (clear, food_grid, physics, death,
                                   food_detect, food_respawn, agent_grid,
                                   collision, vision, prepare_dispatch).
  bitonic_sort_subgroup.wgsl    -- Subgroup-accelerated top-K sort, spliced in
                                   when `wgpu::Features::SUBGROUP` is supported.
```

`GpuKernel` is the only runtime. There is no alternative `GpuBrain` mode and no per-pass dispatch path — the seven cooperative brain functions in `brain_passes.wgsl` are inlined into a single fused entry point at pipeline creation. A single `dispatch_batch(start_tick, ticks_to_run)` call splits the work into full kernel-batches of `vision_stride * brain_tick_stride` ticks, plus a shorter remainder kernel-batch of `remainder_cycles * brain_tick_stride` ticks when `brain_cycles % vision_stride != 0`, plus an optional physics-only remainder for the trailing `ticks_to_run % brain_tick_stride` ticks that do not fill a brain cycle. Each kernel-batch encodes its own command buffer with `prepare → kernel → global → vision` and submits it (`queue.submit()` once per batch); the physics-only remainder is a separate submit that masks brain/vision off. A separate opportunistic staging copy is appended. Per-agent state never leaves GPU memory unless the CPU explicitly requests a readback.

---

## 3. Data Flow Diagram

```
                     ┌──────────────────────────────────────────────────────────────────┐
                     │  GpuKernel — one kernel-batch (queue.submit() x 1)               │
                     └──────────────────────────────────────────────────────────────────┘

                                  uniform: WorldConfig (CPU → GPU, once per kernel-batch)
                                          │
                                          v
                              ┌───────────────────────────┐
                              │  prepare_dispatch.wgsl    │   dispatch: (1, 1, 1)
                              │  (indirect-dispatch args) │
                              └─────────────┬─────────────┘
                                            v
                              ┌───────────────────────────┐
                              │  kernel_tick.wgsl         │   dispatch: (agent_count, 1, 1)
                              │  per-agent fused pass     │   256 threads per workgroup
                              │  ─ loops vision_stride    │
                              │    cycles ─────────────▶  │
                              │      ┌── per cycle ──┐   │
                              │      │  physics      │   │     reads sensory_buf written
                              │      │  food_detect  │   │     by the *previous* batch
                              │      │  death/respawn│   │     (one-batch sensory lag)
                              │      │  brain (7-stg)│   │
                              │      └───────────────┘   │
                              └─────────────┬─────────────┘
                                            v
                              ┌───────────────────────────┐
                              │  global_tick.wgsl         │   dispatch: (1, 1, 1)
                              │  grid rebuild (food+agent)│
                              │  food respawn/timers      │
                              │  pairwise collisions      │
                              └─────────────┬─────────────┘
                                            v
                              ┌───────────────────────────┐
                              │  vision_tick.wgsl         │   dispatch: (agent_count, 1, 1)
                              │  raycasts terrain/food/   │
                              │  agents → sensory_buf     │
                              └───────────────────────────┘

  Persistent GPU buffers (live across ticks; sizes shown for default 8×6 vision, `ENCODED_DIMENSION = 128`):
  ─── brain_state_buf ────  `BrainLayout::brain_stride` (51,381 f32/agent)  (encoder weights, predictor, habituation, homeo, action, fatigue)
  ─── pattern_buf ────────  `PATTERN_STRIDE` (17,539 f32/agent)            (128 patterns: states, norms, reinforcement, motor, meta, active)
  ─── history_buf ────────  `HISTORY_STRIDE` (8,514 f32/agent)             (64-entry action history ring: motor+state snapshots)
  ─── physics_state_buf ──     per-agent     (position, velocity, vitals, motor telemetry echoes)
  ─── food_state_buf ─────     per-food      (position, consumed flag, respawn timer)
  ─── sensory_buf ────────     per-agent     (raw vision + non-visual features written by vision pass)
  ─── decision_buf ───────     per-agent     (motor + prediction error + credit, written by brain stage 6)

  Transient GPU buffers (overwritten each cycle):
  features, encoded, habituated, homeo_out, similarities, recall_buf
```

### Brain Stages Inside the Fused Kernel

The `kernel_tick.wgsl` per-agent pass runs the seven cooperative brain functions back-to-back, all on the same workgroup and same agent slot. They are no longer separate dispatches:

```
brain cycle (executed vision_stride times per kernel-batch):
  1. coop_feature_extract   sensory_buf (`SENSORY_STRIDE`, 267 f32 for 8×6) → features (`BrainLayout::feature_count` = `VISION_RAYS * 5 + 25`, 265 f32 for 8×6)
  2. coop_encode            features → encoded (`ENCODED_DIMENSION` = 128 f32)
  3. coop_habituate_homeo   habituation EMA + homeostatic gradient/urgency
  4. coop_recall_score      cosine similarity vs 128 patterns
  5. coop_recall_topk       top-16 selection (subgroup or workgroup bitonic sort)
  6. coop_predict_and_act   prediction error, credit, policy, fatigue, exploration
                              → writes motor into decision_buf (consumed by the
                                physics step in the same kernel cycle)
  7. coop_learn_and_store   predictor gradient, Hebbian credit, memory reinforcement,
                              pattern storage, decay
```

---

## 4. GPU-Resident Design

`GpuKernel` keeps **all** simulation state permanently on the GPU. The world geometry, food state, and per-agent physics rows live in GPU storage buffers; the CPU populates them through `upload_world` / `upload_agents` and then leaves them alone while batches run. This eliminates the CPU↔GPU marshalling bottleneck that would otherwise dominate per-tick cost.

### Per-Batch I/O Budget

A `dispatch_batch(start_tick, ticks_to_run)` call splits the work into one or more kernel-batches, each its own command buffer + `queue.submit()`. The CPU side only crosses the bus when:

| Direction | When | What |
|-----------|------|------|
| CPU → GPU | At init / when terrain or food layout changes | `upload_world`: terrain heightmap, biome grid, food positions/consumed/timers |
| CPU → GPU | At spawn / on evolution offspring | `upload_agents`: initial physics rows |
| CPU → GPU | Once per kernel-batch | `upload_world_config`: world-config uniform with `start_tick`, `ticks_to_run`, `vision_stride`, `brain_tick_stride`, phase mask |
| CPU → GPU | On heritable-config edit | `write_agent_state` / `write_agent_heritable_config` |
| GPU → CPU | When `try_collect_state` is called | Position, vitals, motor cache, exploration/fatigue, death counts |
| GPU → CPU | When `try_collect_telemetry` is called | One agent's vision + decision snapshot (selected agent only) |
| GPU → CPU | When `try_collect_agent_state` is called | One agent's full `AgentBrainState` (for inheritance, debugging) |
| GPU only | Per-tick simulation | Brain state, pattern memory, action history, physics state, food state, **sensory features** (written by the vision pass into `sensory_buf`) |

The asymmetry is intentional. Sensory frames are not packed and uploaded per tick: the vision pass raycasts on the GPU from terrain/biome/food/agent buffers and writes the feature layout directly into `sensory_buf`. The brain stage in the next kernel-batch reads its inputs from that same `sensory_buf` — a one-batch sensory lag that amortizes the cost of vision + grid rebuild over `vision_stride` brain cycles.

### Non-Blocking Readback

`GpuKernel` never blocks for state. Three independent staging-buffer tracks (`state_readback`, `telemetry_readback`, `agent_state_readback`), each driven by an `async_readback::ReadbackTracker`, fence pending readbacks against GPU completion. The CPU calls `try_collect_*` each frame; if the buffer hasn't been mapped yet, the call returns without producing data and the UI uses the previous frame's cached snapshot. This decouples render-loop pacing from GPU completion time.

`read_full_state_blocking` and `read_agent_telemetry_blocking` exist for tests and one-shot debugging — production code uses the non-blocking path.

### Buffer Allocation

All buffers are created at `GpuKernel::new` with sizes proportional to `agent_count` and the vision dimensions encoded in `BrainLayout`. Persistent buffers (`brain_state`, `pattern_buffer`, `history_buffer`, `physics_state`, `food_state`, world-config, sensory) use `STORAGE | COPY_SRC | COPY_DST`; transient working buffers use `STORAGE` only. Staging buffers for readback use `MAP_READ | COPY_DST` and are sized for the worst-case message (full state for `state_readback`, one agent's slice for the others).

---

## 5. Buffer Layout

All buffer offsets and stride constants are defined once in `buffers.rs` and auto-generated into WGSL via `wgsl_constants()`. This function emits a constants header that is prepended to every shader at pipeline creation time. The constants include all offsets, strides, and utility functions (`fast_tanh`, `pcg_hash`, `rand_f32`, `rand_normal`). Because both Rust and WGSL code derive from the same source of truth, offset mismatch bugs are impossible.

### Core Dimensions

| Constant | Value (default 8×6) | Description |
|----------|--------------------:|-------------|
| `ENCODED_DIMENSION` | 128 | Internal encoded state dimensionality (`crates/xagent-brain/src/buffers.rs`) |
| `BrainLayout::feature_count` | 265 = `VISION_RAYS * 5 + 25` | Feature vector size (192 RGBA + 48 depth + 25 derived non-visual; scales with `VISION_W`/`VISION_H`) |
| `MEMORY_CAPACITY` | 128 | Maximum patterns per agent |
| `RECALL_TOPK` | 16 | Top-K recalled patterns per tick |
| `ACTION_HISTORY_LEN` | 64 | Credit-assignment lookback window |
| `ERROR_HISTORY_LEN` | 128 | Prediction-error ring-buffer size |

The feature/encoded sizes and all derived per-agent strides scale with the configured vision dimensions. `BrainLayout::new(vision_width, vision_height)` is the single source of truth — see `crates/xagent-brain/src/buffers.rs`. Concrete strides for the default 8×6 layout (`ENCODED_DIMENSION = 128`, `feature_count = 265`) come out to `brain_stride = 51,381` f32, `PATTERN_STRIDE = 17,539` f32, `HISTORY_STRIDE = 8,514` f32. The `O_*` offset constants and per-region sizes are auto-generated alongside `BrainLayout` and surfaced to WGSL via the `wgsl_constants()` helper.

### Sensory Input Layout (CPU --> GPU)

```
[  192 RGBA vision  |  48 depth  |  vel(3)  fac(3)  ang(1)  e(1)  i(1)  ed(1)  id(1)  touch(16)  ]
 ^                   ^            ^                                                                ^
 0                   192          240                                                              267
```

Total: `SENSORY_STRIDE = 267` f32 per agent. `pack_sensory_frame()` handles the CPU-side packing, including sorting touch contacts by intensity and zero-padding.

### Brain State Buffer (per agent: `BrainLayout::brain_stride`, 51,381 f32 for the default 8×6 layout)

Regions (in offset order; concrete offsets are dimension-dependent and emitted by `BrainLayout` — see `crates/xagent-brain/src/buffers.rs`):

- `O_ENCODER_WEIGHTS` — `feature_count * ENCODED_DIMENSION` (= 33,920 for 8×6) encoder weight matrix.
- `O_ENCODER_BIASES` — `ENCODED_DIMENSION` (128) per-dimension bias.
- `O_PREDICTOR_WEIGHTS` — `PREDICTOR_DIMENSION * ENCODED_DIMENSION` predictor matrix (operates in encoded space).
- `O_PREDICTOR_CONTEXT_WEIGHT` and the rest of the fixed-size tail (`FIXED_TAIL_SIZE`): predictor error ring, habituation EMA + attenuation, previous-encoded snapshot, homeostasis state, action/turn policy weights + biases, exploration rate, motor-fatigue ring + cursor + factor + length, previous prediction, tick counter, heritable config, and per-agent `movement_speed`.

### Pattern Memory Buffer (per agent: `PATTERN_STRIDE`, 17,539 f32 for the default 8×6 layout)

Stores `MEMORY_CAPACITY` (= 128) patterns. Regions: `O_PAT_STATES` (`MEMORY_CAPACITY * ENCODED_DIMENSION` encoded-space states), `O_PAT_NORMS` (cached L2 norms), `O_PAT_REINF` (per-pattern reinforcement that decays over time), `O_PAT_MOTOR` (`[forward, turn, outcome_valence] * MEMORY_CAPACITY`), `O_PAT_META` (`[created_at, last_accessed, activation_count] * MEMORY_CAPACITY`), `O_PAT_ACTIVE` (active flag — recall is gated here, not on `O_PAT_REINF`), and `O_ACTIVE_COUNT` bookkeeping. Exact offsets are derived from `BrainLayout` and emitted alongside the buffer; see `crates/xagent-brain/src/buffers.rs`.

### Action History Buffer (per agent: `HISTORY_STRIDE`, 8,514 f32 for the default 8×6 layout)

A 64-entry ring of motor commands plus per-entry encoded-state snapshots. Regions: `O_MOTOR_RING` (`[forward, turn, tick, gradient, _pad] * ACTION_HISTORY_LEN`), `O_STATE_RING` (`[encoded_state(ENCODED_DIMENSION)] * ACTION_HISTORY_LEN` snapshots — `ENCODED_DIMENSION * ACTION_HISTORY_LEN` f32 in total), and `O_HIST_CURSOR` bookkeeping. Exact offsets are dimension-dependent; see `crates/xagent-brain/src/buffers.rs`.

### Integer Storage Convention

Integer values (cursors, counts, tick counters) are stored as `f32` in GPU buffers and cast via `u32()` in WGSL. This is safe for exact integers up to 2^24 = 16,777,216, which is far beyond any practical tick count or buffer index.

---

## 6. Component Deep Dive: The 7 Brain Stages

> **Note:** the seven stages described below are the conceptual pipeline. They live in `src/shaders/kernel/brain_passes.wgsl` as cooperative functions (`coop_feature_extract`, `coop_encode`, `coop_habituate_homeo`, `coop_recall_score`, `coop_recall_topk`, `coop_predict_and_act`, `coop_learn_and_store`) and are inlined into `kernel_tick.wgsl` and `brain_tick.wgsl` at composition time. There are no per-stage shader files. The canonical dimensions and offsets are emitted by `BrainLayout` in `crates/xagent-brain/src/buffers.rs` and surfaced to WGSL via `common.wgsl`; for the default 8×6 vision they evaluate to `ENCODED_DIMENSION = 128` and `BrainLayout::feature_count = 265`. Any older `DIM = 32` / `FEATURE_COUNT = 217` literals in the §§6.1–6.7 prose are legacy — defer to `buffers.rs` and `common.wgsl` whenever the numbers disagree.

### 6.1 Feature Extraction -- `coop_feature_extract`

**What it does**: Transforms raw sensory input (`SENSORY_STRIDE = 267` f32: 192 color + 48 depth + 27 non-visual) into the brain feature vector (`FEATURE_COUNT = 265` f32: 192 color + 48 depth + 25 derived non-visual). This is the first stage of the semantic firewall -- structured sensory data becomes a flat feature array.

**How it works**:

1. **Vision RGBA**: Direct copy of 192 values (8x6 grid, 4 channels each). No spatial pooling -- the full color+alpha grid is preserved. This gives the brain per-pixel color access, critical for learning that "red ahead = danger zone" and "green ahead = food zone."

2. **Vision depth**: Direct copy of 48 depth values (one per pixel). The fused kernel includes depth in the feature buffer — not skipped.

3. **Velocity magnitude**: Computes `sqrt(vx^2 + vy^2 + vz^2)` from the 3-component velocity vector, collapsing direction into a single speed scalar.

4. **Proprioception**: Copies facing direction (3), angular velocity (1).

5. **Interoception**: Copies energy (1), integrity (1), energy delta (1), integrity delta (1).

6. **Touch contacts**: Copies 4 contact slots x 4 features = 16 values `[dir_x, dir_z, intensity, surface_tag/4]`.

**Feature layout**: `[192 RGBA | 48 depth | 1 speed | 3 facing | 1 angular | 1 energy | 1 integrity | 1 e_delta | 1 i_delta | 16 touch] = 265`

---

### 6.2 Encoding -- `coop_encode`

**What it does**: Projects the 265-dimensional feature vector into a 128-dimensional encoded representation (`ENCODED_DIMENSION`) via a learned weight matrix and tanh nonlinearity. This is the **information bottleneck** -- 265 inputs compressed to 128 outputs, forcing the brain to learn what matters.

**How it works**:

```
encoded[d] = fast_tanh( sum_f( features[f] * weights[f * ENCODED_DIMENSION + d] ) + biases[d] )
```

For each of the 128 output dimensions, the shader computes a weighted sum across all 265 features (column-major weight layout: `weights[f * ENCODED_DIMENSION + d]`) plus a per-dimension bias, then squashes through `fast_tanh`.

**Weight initialization** (in `buffers::init_brain_state`): Xavier/Glorot uniform -- `uniform(-scale, scale)` where `scale = 1/sqrt(FEATURE_COUNT)`. This prevents tanh saturation at initialization.

**Weight layout**: The encoder weight matrix is stored column-major (`[FEATURE_COUNT x ENCODED_DIMENSION]`, indexed as `[f * ENCODED_DIMENSION + d]`). This layout means each output dimension's weights are scattered across memory at stride `ENCODED_DIMENSION` -- not cache-optimal on CPU, but irrelevant on GPU where each invocation computes one agent's full encoding.

**Emergent property**: The encoder creates a **selectivity bottleneck**. 192 RGBA + 48 depth + 25 non-visual features = 265 inputs compressed to 128 floats. What gets through this bottleneck is what the brain "pays attention to." The tanh squashing bounds all encoded values to [-1, 1], making cosine similarity a natural distance metric for downstream recall.

---

### 6.3 Habituation + Homeostasis -- `coop_habituate_homeo`

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

### 6.4 Recall Scoring -- `coop_recall_score`

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

### 6.5 Recall Top-K Selection -- `coop_recall_topk`

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

### 6.6 Prediction + Action Selection -- `coop_predict_and_act`

This is the largest and most complex stage. It combines what were previously 5 separate CPU components into a single cooperative function: prediction error computation, predictor matrix multiply, credit assignment, policy evaluation with memory blend, exploration noise, and motor fatigue.

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

1. **Temporal decay**: `temporal = exp(-age * CREDIT_DECAY)` where `CREDIT_DECAY = 0.3`. The loop skips entries with `temporal < 0.01`, so actions older than ~15 ticks contribute negligibly.
2. **Improvement signal**: `improvement = current_gradient - recorded_gradient`. If `|improvement| < DEADZONE (0.005)`, the improvement is replaced by a tonic fallback `gradient * urgency * TONIC_CREDIT_SCALE (0.5)` instead of being skipped.
3. **Pain amplification**: Negative improvements are multiplied by `PAIN_AMP = 3.0`, reflecting the biological reality that aversive stimuli produce stronger learning signals.
4. **State-conditioned weight update**: `weights[d] += WEIGHT_LR * credit * recorded_motor * recorded_state[d]`. The recorded state snapshot from when the action was taken ensures credit is attributed to the correct sensory context.

**Weight normalization**: After credit assignment, forward and turn weight vectors are clipped to L2 norm <= `MAX_WEIGHT_NORM = 2.0` (synaptic homeostasis).

#### 6.6.4 Policy Evaluation

Continuous motor output is computed as a dot product of learned weights with the encoded state (pre-habituation), matching the features that credit assignment trains against:

```
fwd = dot(fwd_weights, encoded) + fwd_bias
trn = dot(trn_weights, encoded) + trn_bias
```

This is a continuous-output linear policy -- no discrete action table. The agent learns which features predict beneficial forward motion and which predict beneficial turning.

#### 6.6.5 Memory-Informed Motor Blend

Recalled patterns contribute their stored motor commands weighted by similarity and outcome valence:

```
mem_fwd = sum( sim * valence * stored_forward ) / sum( |sim * valence| )
mix = clamp(mean_|sim*valence|, 0, 1) * 0.4
fwd = fwd * (1 - mix) + mem_fwd * mix
```

- **Positive valence**: "do what I did before" -- reinforces the recalled motor command
- **Negative valence**: "do the opposite" -- the sign flip steers away from past mistakes
- Memory contributes up to 40% of motor signal

#### 6.6.6 Exploration Noise

Exploration rate is computed dynamically:

```
novelty_bonus = min(pred_error * 2.0, 0.4)
urgency_penalty = min(urgency * 0.4, 0.5)
policy_confidence = clamp((|fwd| + |trn|) / 2.0, 0.0, 1.0)
exploration_rate = clamp(0.5 - policy_confidence * 0.25 + novelty_bonus + curiosity - urgency_penalty, 0.10, 0.85)
```

Gaussian noise scaled by `exploration_rate` is added to the motor output, using `pcg_hash` GPU RNG with the exploration seed derived from `agent_id ^ (tick_u * 747796405u)` before hashing. The motor output is clamped to [-1, 1].

#### 6.6.7 Motor Fatigue

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

#### 6.6.8 Output Recording

After computing final motor output:
1. Records `[noise_fwd * exploration_rate, noise_trn * exploration_rate, tick, gradient, 0.0]` to the action history ring at `O_MOTOR_RING`. Storing the exploration noise (not the full motor) is what makes credit assignment a proper REINFORCE gradient -- see PR #97.
2. Records the encoded (pre-habituation) state snapshot at `O_STATE_RING` for future credit assignment -- matches the features the policy and credit updates train against.
3. Saves the prediction to `O_PREV_PREDICTION` for next tick's error computation.
4. Increments `O_TICK_COUNT`.
5. Writes `[prediction(32), credit_signal(32), fwd, trn, strafe, _pad]` to the decision buffer for pass 7 and CPU readback.

---

### 6.7 Learning + Memory Storage -- `coop_learn_and_store`

Five learning operations packed into a single stage, using the prediction and credit signal produced by `coop_predict_and_act`.

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

None of these behaviors are explicitly programmed. They arise from the interaction of the seven cooperative brain stages and their shared constraints (referenced by the `coop_*` function inlined into `kernel_tick.wgsl`):

| Phenomenon | How It Emerges | Contributing Stages |
|------------|---------------|---------------------|
| **Attention** | Memory capacity (128) forces selective recall; encoder bottleneck (`feature_count` → `ENCODED_DIMENSION`, e.g. 265 → 128 for 8×6) compresses information | `coop_encode`, `coop_recall_score` + `coop_recall_topk` |
| **Fear / Avoidance** | Negative homeostatic gradient --> pain amplifier (3x) makes damage signal loud --> credit assignment blames recent actions via state snapshots --> policy weights learn to avoid danger-associated features --> prospective evaluation applies these weights to the predicted future, anticipating danger before entering it | `coop_habituate_homeo`, `coop_predict_and_act` (credit + prospection) |
| **Curiosity** | High prediction error in safe situations --> exploration noise increases; habituation produces a curiosity bonus when input is monotonous, further boosting exploration | `coop_habituate_homeo`, `coop_predict_and_act` (exploration) |
| **Habit Formation** | Repeated successful actions build strong policy weights --> exploitation ratio increases --> behavior becomes automatic | `coop_predict_and_act` (credit), `coop_learn_and_store` (reinforcement) |
| **Startle / Surprise** | Sudden prediction error spike --> novelty bonus increases --> exploration spikes | `coop_predict_and_act` (error + exploration) |
| **Adaptation** | Prediction error decreases in stable environments --> exploration drops --> behavior stabilizes | `coop_predict_and_act`, `coop_learn_and_store` (predictor learning) |
| **Panic** | Low energy/integrity --> high urgency --> exploration suppressed --> agent falls back on policy weights | `coop_habituate_homeo` (urgency), `coop_predict_and_act` (exploration) |
| **Forgetting** | Patterns not recalled or reinforced decay below zero and are deactivated | `coop_learn_and_store` (decay) |
| **Boredom / Loop Breaking** | Monotonous input --> habituation attenuates repetitive dimensions --> curiosity rises --> exploration increases; simultaneously, low motor variance --> fatigue dampens output --> repeated action weakens | `coop_habituate_homeo`, `coop_predict_and_act` (fatigue + exploration) |
| **Contextual Memory** | Policy weights map encoded features to motor preferences -- different percepts trigger different behaviors; recalled patterns blend motor advice | `coop_predict_and_act` (policy + memory blend) |
| **Desensitization** | The slow EMA timescale integrates gradual changes; constant mild negative gradient eventually stops triggering strong reactions | `coop_habituate_homeo` (homeostasis) |

---

## 8. Host API (gpu_kernel.rs)

### GpuKernel

```rust
pub struct GpuKernel {
    // wgpu device + queue
    // Persistent buffers: brain_state, pattern_buffer, history_buffer,
    //                     physics_state, food_state, world_config,
    //                     sensory + working buffers
    // Compute pipelines: physics, vision, brain, kernel, global, prepare
    // Three staging-buffer tracks (state, telemetry, agent_state) each
    // driven by an async_readback::ReadbackTracker
}
```

### Public API

The method list below is the contract used by `xagent-sandbox`. See `gpu_kernel.rs` for the full set of helpers; everything per-tick that the sandbox needs is here.

| Method | Description |
|--------|-------------|
| `is_available() -> bool` | Static probe: returns true if any wgpu adapter (real GPU or software fallback) exists. Used by tests to skip GPU work on headless CI. |
| `new(agent_count, food_count, brain_config, world_config)` | Creates the wgpu device (requesting `PUSH_CONSTANTS`, optionally `SUBGROUP`), allocates all buffers and pipelines, and composes the fused shaders with subgroup markers and the `VISION_W`/`VISION_H` override cascade. |
| `upload_world(terrain_heights, biome_grid, food_positions, food_consumed, food_timers)` | Writes the terrain heightmap, biome grid, and the food spatial buffers (positions, consumed flags, respawn timers) into GPU storage. Static-ish world data — call when the world is built or food layout changes. |
| `upload_agents(&[(Vec3, f32, f32, usize, usize)])` | Writes initial physics rows `(position, max_energy, max_integrity, memory_capacity, processing_slots)` for the listed agents into `agent_phys` storage. Used when spawning a new generation. |
| `upload_world_config(start_tick, ticks_to_run)` | Writes the per-batch world-config uniform consumed by all four passes inside a kernel-batch (start tick, batch size, stride parameters). Called internally by `dispatch_batch`. |
| `dispatch_batch(start_tick, ticks_to_run)` | Splits the work into full kernel-batches of `vision_stride * brain_tick_stride` ticks, plus a shorter remainder kernel-batch of `remainder_cycles * brain_tick_stride` ticks when `brain_cycles % vision_stride != 0`, plus an optional physics-only remainder for the trailing `ticks_to_run % brain_tick_stride` ticks that do not fill a brain cycle. Each kernel-batch is one command-buffer + `queue.submit()` running `prepare → kernel → global → vision`; the physics-only remainder is a separate submit that masks brain/vision off. An opportunistic copy into a staging slot is appended. Always returns `true` (kept for API compatibility — staging copies may be skipped when all slots are in flight, but compute is decoupled from readback). |
| `dispatch_batch_masked(start_tick, ticks_to_run, phase_mask)` | Variant for tests and benchmarks that gates which phases run per cycle (bit 0 = physics, bit 1 = vision, bit 2 = brain). Iterates cycles in chunks of 100 (Metal command-buffer deadlock workaround), appends a remainder physics-only pass, copies `agent_phys` to the active staging slot, then blocks on `device.poll(Wait)` for GPU completion. Does **not** run any global pass (no grid rebuild, food respawn, or collisions) and does **not** update `cached_state` — call sites read GPU buffers directly. |
| `try_collect_state() -> bool` | Non-blocking: polls all in-flight state-readback slots; if any is ready, copies the most recent into `cached_state` (and `cached_food_state` when food exists). Returns `true` on update. |
| `cached_state() -> &[f32]` | Latest physics-state snapshot (positions, vitals, motor, death counts, telemetry slots). |
| `cached_food_state() -> Option<&[f32]>` | Latest food-state snapshot when available. |
| `read_full_state_blocking() -> &[f32]` | Test/debug only: blocks until the next state readback completes. |
| `request_agent_telemetry(index)` | Schedules a one-agent telemetry readback (vision + decision snapshot). |
| `try_collect_telemetry() -> Option<AgentTelemetry>` | Non-blocking: returns telemetry when the staged copy is mapped. |
| `cached_telemetry() -> Option<&AgentTelemetry>` | Most recent telemetry snapshot. |
| `request_agent_state(index) -> bool` | Schedules a full `AgentBrainState` readback for the named agent. |
| `try_collect_agent_state() -> Option<Option<AgentBrainState>>` | Non-blocking collection of the scheduled `AgentBrainState`. |
| `read_agent_state(index) -> AgentBrainState` | Blocking readback for tests / one-shot inspection. |
| `write_agent_state(index, &AgentBrainState)` | Uploads one agent's full brain slice (used by evolution to seed offspring). |
| `write_agent_heritable_config(index, &BrainConfig)` | Overwrites only the heritable config tail of one agent's brain slice. |
| `batch_write_agent_states(count, F)` | Bulk variant of `write_agent_state` driven by a closure (one allocation, one submit). |
| `try_reset_agents(&BrainConfig) -> bool` | Re-initializes every agent's brain + physics row from the given config. |
| `reset_agents(&BrainConfig)` / `reset_agents_seeded(&BrainConfig, seed)` | Blocking variants used at startup. |
| `brain_tick_stride() -> u32` | Returns `brain_config.brain_tick_stride` — the number of physics ticks per brain cycle. Sandbox uses it as the minimum batch size that contains a full brain cycle. |
| `kernel_batch_size() -> u32` | Returns `vision_stride * brain_tick_stride` — the number of physics ticks in one full kernel-batch; dispatching in exact multiples guarantees deterministic global- and vision-pass cadence. |

### AgentBrainState

```rust
pub struct AgentBrainState {
    pub brain_state: Vec<f32>,    // BRAIN_STRIDE, vision-dependent
    pub patterns: Vec<f32>,       // PATTERN_STRIDE
    pub history: Vec<f32>,        // HISTORY_STRIDE
}
```

Used for cross-generation inheritance (the governor reads parent state, mutates it, writes to offspring), mutation, and DB persistence. The three vectors are the exact GPU buffer contents for one agent slice. Stride values are reported by `BrainLayout` and depend on the vision dimensions; the legacy `8,468 / 5,251 / 2,370` figures from the pre-fused era are no longer accurate for arbitrary configurations.

### Death / Respawn on the GPU

Death detection and respawn live entirely in WGSL (`phase_death.wgsl`, invoked from `kernel_tick.wgsl` after the physics step). There is no `death_signal` Rust call. When the kernel decides an agent has died (energy ≤ 0 or integrity ≤ 0):

1. **Spawn search**: tries up to 50 GPU-RNG samples for a non-Danger biome position; if all 50 attempts land in Danger biomes, falls back to one fresh random position within world bounds without the biome check (see the `!found` branch in `phase_death.wgsl` / `kernel_tick.wgsl::agent_death_respawn`).
2. **Physics reset**: full energy, full integrity, zero velocity, facing +Z; death count incremented; fitness counters (`food_count`, `ticks_alive`, `last_death_tick`) preserved.
3. **Memory trauma**: all `O_PAT_REINF` entries are multiplied by `0.5`. The death pass leaves `O_PAT_ACTIVE` untouched, so recall (which gates on `O_PAT_ACTIVE` in `brain_passes.wgsl`, not on reinforcement) is not cut off by this step. Halved reinforcement only makes subsequent decay reach the `<= 0.0` deactivation point sooner for the weakest patterns; the strongest memories survive.
4. **Brain reset**: homeostasis EMAs zeroed, exploration rate set to `0.5`, habituation EMAs zeroed and attenuation reset to `1.0`, fatigue factor reset to `1.0`, position-ring staleness state cleared, action history zeroed.

The CPU only learns about a death by reading `physics_state[base + P_DEATH_COUNT]` on the next state readback.

### AgentTelemetry

`GpuKernel::request_agent_telemetry(index)` queues a one-shot copy of one agent's slice across **four** staging buffers — sensory, decision, brain_state, and phys — into the `telemetry_staging` group. The next `try_collect_telemetry()` that finds all four mappings ready returns an `AgentTelemetry` (see `gpu_kernel.rs::AgentTelemetry`) with: `vision_color` (`VISION_RAYS` RGBA floats, from sensory), `motor_fwd` and `motor_turn` (from decision), `mean_attenuation` and `curiosity_bonus` and `fatigue_factor` and `staleness` (derived from brain_state), and `urgency`, `gradient`, `prediction_error`, `exploration_rate` (from phys). There is no `motor_variance` field. The sandbox calls this once per frame for the selected agent — issuing it per-tick for all agents would negate the performance gains.

### BrainTelemetry

`BrainTelemetry` (in `lib.rs`) is a stub struct retained for UI/recording compatibility. It is not populated by the kernel directly — the sandbox builds it from `AgentTelemetry` + cached state when it needs the old field shape for replay or sparkline charts.

`BrainTelemetry::behavior_phase()` classifies an agent's composite score:

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
| `vision_rays` | 48 | Number of vision rays (W×H). Affects sensory buffer size |
| `brain_tick_stride` | 10 | Physics ticks per brain+vision cycle. Higher → faster, less responsive |
| `vision_stride` | 10 | Brain cycles between global passes (grid, collisions, vision). Higher → more throughput |
| `metabolic_rate` | 1.0 | Multiplier for all energy costs. Lower → agents survive longer |
| `integrity_scale` | 1.0 | Multiplier for integrity damage/regen. Higher → deadlier hazards |

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

### Test Modules

Tests live next to the code they cover. The lib test target compiles to a single binary with three inline modules:

| Module | Scope |
|--------|-------|
| `buffers::tests` | Buffer layout invariants — sensory stride matches feature count, `BrainLayout` derives consistent offsets across vision sizes, `pack_sensory_frame` fills buffers with finite values, `init_brain_state` produces correctly-sized output, SoA/AoS pattern indexing covers identical regions. |
| `gpu_kernel::tests` | Shader composition contract — subgroup markers present in every entry shader, vision dimensions feed the WGSL override cascade, the subgroup and non-subgroup paths each leave the composed source self-consistent (every `_PARAMS` placeholder has a matching `_ARGS` call site). Runs without requiring a GPU device. |
| `async_readback::tests` | `ReadbackTracker` state-machine invariants — request/collect transitions, idempotent polling, mapped-buffer reuse. |

GPU-dependent integration tests (full kernel-tick behavioral checks, deterministic benchmarks) live in the sandbox crate's integration tests so they can share the world setup and skip themselves cleanly on headless CI via the `GpuKernel::is_available` probe.

---

## 11. Design Decisions

### Why GPU-Resident Over CPU

The previous CPU implementation had one `Brain` struct per agent with heap-allocated weight matrices, pattern vectors, and ring buffers. At 50+ agents, the per-tick cost was dominated by:
- 50 independent matrix multiplies (encoder: `feature_count x ENCODED_DIMENSION`, predictor: `ENCODED_DIMENSION x ENCODED_DIMENSION` — for the default 8×6 vision: 265x128 and 128x128)
- 50 * 128 cosine similarity computations (recall scoring)
- Scattered memory access patterns (each agent's data in different heap locations)

Moving to GPU makes all agents' matrix multiplies a single dispatch. More importantly, all brain state lives in contiguous GPU buffers with computed strides, eliminating pointer chasing entirely. Sensory features are now produced on-GPU by the vision pass (writing `sensory_buffer`) and motor commands are consumed on-GPU by the physics step (reading `decision_buffer`); the CPU does not pack sensory frames per tick and does not read motor commands per tick. The only routine bus traffic during the simulation is the per-batch world-config uniform write and asynchronous readbacks for telemetry and UI snapshots.

### Why a Fused Kernel

The cognitive pipeline is conceptually 7 stages, but they all run inside a single fused compute kernel (`kernel_tick.wgsl`). An earlier design dispatched each stage as its own shader, but CPU↔GPU coordination overhead dominated at high tick rates: 11–19 dispatches per tick plus per-tick `write_buffer` calls and blocking motor-command readback capped throughput around 600 ticks/second.

The fused kernel runs N simulated ticks per dispatch with one workgroup per agent (256 threads). Workgroup barriers separate the stages, shared workgroup memory is used for cooperative reductions (similarity max, top-K, food detection), and physics + food detection + death/respawn + the seven brain stages all execute inline. The CPU encodes a single command buffer per batch regardless of `ticks_to_run`, lifting per-agent throughput past 60,000 ticks/second. The seven brain stages still exist as discrete cooperative functions in `brain_passes.wgsl` — they're just inlined into one shader at composition time rather than dispatched as separate passes.

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

- **No hierarchical pattern abstraction**: All 128 patterns are stored at the same level of abstraction. There is no mechanism for forming higher-order patterns ("I'm in a corridor" from a sequence of wall-patterns) or chunking temporal sequences into reusable units.

- **Fixed memory capacity**: The brain cannot grow its memory. A dynamic capacity that expands in rich environments and contracts in simple ones would better match biological memory allocation.

- **No inter-agent brain communication**: Each brain is entirely isolated. There is no mechanism for one agent to share learned patterns or action values with another. Social learning and cultural transmission would require some form of brain-to-brain communication channel.

- **No sleep/consolidation**: Biological brains consolidate memories during sleep, replaying and strengthening important patterns. The current system has no offline consolidation phase -- all learning happens online during ticks.

- **Telemetry is per-selected-agent only**: `GpuKernel::read_agent_telemetry()` reads vision, motor, and brain-state data for one agent per frame. Full telemetry for all agents simultaneously would still negate performance benefits.

- **Depth features unused**: The 48 depth values from the sensory frame are uploaded but not extracted as features. Adding depth as a feature channel would improve spatial reasoning at the cost of a larger encoder weight matrix.

### Future Directions

- **Hierarchical temporal memory**: Stack multiple levels of pattern memory, each operating at a different temporal granularity.
- **Multi-agent pattern sharing**: Allow agents to "teach" each other by sharing association strengths or pattern representations.
- **Dreaming/replay**: Periodically replay stored pattern sequences during idle time to consolidate important memories and prune irrelevant ones.
- **Depth feature integration**: Extract depth as a separate feature channel for improved spatial awareness.
- **Broadcast telemetry**: Extend the per-selected-agent telemetry to sample multiple agents without blocking the render loop.
