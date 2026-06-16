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

`GpuKernel` is the sole runtime. There is no CPU-side brain object, no `Brain::tick(frame) -> MotorCommand` API, and no swappable cognitive-architecture trait. At startup the CPU uploads world geometry (terrain heightmap, biome grid, food positions/timers/consumed flags) and the initial per-agent physics rows; thereafter the only thing the CPU writes per dispatch is the world-config uniform (`start_tick`, `ticks_to_run`, `vision_stride`, `brain_tick_stride`, phase mask). `kernel.dispatch_batch(start_tick, ticks_to_run)` splits the requested ticks into full kernel-batches of `vision_stride * brain_tick_stride` ticks, plus a shorter remainder kernel-batch (`remainder_cycles * brain_tick_stride` ticks) when `brain_cycles % vision_stride != 0`, plus an optional physics-only remainder of `ticks_to_run % brain_tick_stride` ticks for the trailing fragment that does not fill a brain cycle. Each kernel-batch submits one command buffer running the four passes `prepare → kernel → global → vision`; the physics-only remainder submits a single physics pass with the brain/vision phases masked off. The brain stage in step 2 reads its sensory inputs from the `sensory_buffer` that the *previous* batch's vision pass wrote — a one-batch sensory lag that lets the costly global+vision work amortize over `vision_stride` brain cycles. All persistent brain state lives permanently in GPU storage buffers; the only data that crosses the bus while the simulation is running is whatever the CPU side asks the kernel to read back (typically position/vitals for the UI and a sampled telemetry snapshot for the selected agent).

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [GPU-Resident Design](#4-gpu-resident-design)
5. [Buffer Layout](#5-buffer-layout)
6. [Component Deep Dive: The 7 Brain Stages](#6-component-deep-dive-the-7-brain-stages)
   - [6.1 Feature Extraction](#61-feature-extraction----coop_feature_extract)
   - [6.2 Encoding](#62-encoding----coop_encode)
   - [6.3 Habituation + Homeostasis](#63-habituation--homeostasis----coop_habituate_homeo)
   - [6.4 Recall Scoring](#64-recall-scoring----coop_recall_score)
   - [6.5 Recall Top-K Selection](#65-recall-top-k-selection----coop_recall_topk)
   - [6.6 Prediction + Action Selection](#66-prediction--action-selection----coop_predict_and_act)
   - [6.7 Learning + Memory Storage](#67-learning--memory-storage----coop_learn_and_store)
7. [Emergent Phenomena](#7-emergent-phenomena)
8. [Host API (gpu_kernel.rs)](#8-host-api-gpu_kernelrs)
9. [Configuration (BrainConfig)](#9-configuration-brainconfig)
10. [Testing](#10-testing)
11. [Design Decisions](#11-design-decisions)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## The Brain Has No Eyes

The most important thing to understand about this architecture: **the brain has zero semantic knowledge of its inputs**.

In the live runtime sensory features are produced on-GPU: the vision pass raycasts colors and depths into `sensory_buffer`, and the same vision pipeline's `phase_vision_senses` (`src/shaders/kernel/phase_vision.wgsl`) appends per-agent proprioception, interoception, energy/integrity deltas, and touch contacts in a fixed positional layout — touch slots are filled in 3×3-cell discovery order (food cells first, then agent cells) up to `MAX_TOUCH_CONTACTS`, with each contact encoded as `(direction_x, direction_z, normalized_proximity, surface_tag / 4.0)`. The CPU-side `buffers::pack_sensory_frame()` is only exercised by `buffers` tests; it does not feed the live brain. The brain's first stage `coop_feature_extract` in `src/shaders/kernel/brain_passes.wgsl` then projects `sensory_buffer` (default 8×6: `SENSORY_STRIDE = 267` f32 = 192 RGBA + 48 depth + 27 non-visual) into the feature vector (`BrainLayout::feature_count = VISION_RAYS * 5 + 25`, = 265 f32 for the default 8×6). The packing is not free of inductive bias — the modality layout, contact-cap, and `surface_tag` category channel are all hand-chosen priors — but they live entirely in shader/packer code, not as named fields the brain reads. (The `surface_tag` enum reserves `TOUCH_FOOD`, `TOUCH_TERRAIN_EDGE`, `TOUCH_HAZARD`, and `TOUCH_AGENT`, but `phase_vision_senses` currently emits only `TOUCH_FOOD` and `TOUCH_AGENT` contacts.) From `coop_feature_extract` onward, the brain operates on opaque numerical vectors: no concept of "vision," no awareness of "eyes," no understanding that index 47 was once an RGBA pixel and index 73 was once an energy level.

```
World --> GPU vision pass --> sensory_buffer [267 f32] --> coop_feature_extract --> [265 f32]
              |                                                  |
     phase_vision_raycast +                            Brain sees only a
     phase_vision_senses (GPU)                         flat array<f32>

(the test-only pack_sensory_frame() mirrors the same [267 f32] layout — not in the live path)
```

Consider what happens when another agent -- say, a magenta-colored one -- enters the visual field. The brain doesn't receive "agent detected" or "entity of type Agent at bearing 30 degrees." It experiences indices 12--15 shifting from `[0.3, 0.6, 0.2, 1.0]` to `[0.9, 0.2, 0.6, 1.0]`. Simultaneously, a touch contact might add nonzero values at indices 199--202 (direction, intensity, tag). The brain has no legend for any of this. It doesn't know that `surface_tag=4` means "agent." It doesn't know that the shifted values represent magenta. Over hundreds of ticks, if this pattern of input correlates with energy dropping (food competition), the brain discovers -- through prediction error and homeostatic gradient alone -- that "those numerical patterns are bad for me." The concept of "that's a competitor" *emerges* from experience, not from labels.

This is the fundamental difference from traditional AI systems. There are no reward functions hand-crafted by engineers. No labeled feature vectors telling the model "this is vision, this is hunger." But the picture is not bias-free either: the fixed modality layout and the preserved `surface_tag` channel are hand-chosen priors that ride along with the otherwise opaque vector. The honest summary is that the shader boundary — the GPU vision pass, with the test-only `pack_sensory_frame()` mirroring its layout — strips struct labels but keeps positional structure; what the brain then sees is a numerically flattened interface, not a pristine raw signal. What remains downstream is prediction + homeostatic gradient + experience, and from these ingredients combined with that bounded prior, all meaning is discovered.

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
- **Tunes its own influence**: the predictor's recalled-context weight adapts toward whatever blend of memory and projection keeps the error low

There is no separate reward signal. There is no loss function designed by a human. The agent learns because its predictions are wrong, and prediction error is metabolically expensive.

### Homeostatic Feedback as the Only Evaluative Signal

The brain has no concept of "good" or "bad" built in. Instead, `habituate_homeo.wgsl` tracks whether internal variables (energy, physical integrity) are trending toward or away from stability. This gradient -- positive means improving, negative means worsening -- modulates:

- **Credit assignment**: the urgency-amplified per-tick homeostatic delta is the reward in a TD(λ) actor-critic — a value head learns the discounted return, and its TD error credits recent actions through per-dimension eligibility traces
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
  ─── brain_state_buf ────  `BrainLayout::brain_stride` (51,898 f32/agent)  (encoder weights, predictor, habituation, homeo, action, fatigue, TD value head + traces)
  ─── pattern_buf ────────  `PATTERN_STRIDE` (17,539 f32/agent)            (128 patterns: states, norms, reinforcement, motor, meta, active)
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
                                physics step in the next kernel cycle)
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
| GPU → CPU | When `request_state_snapshot` has staged data and `try_collect_state_snapshot` is called | Position, vitals, motor cache, exploration/fatigue, death counts |
| GPU → CPU | When `try_collect_telemetry` is called | One agent's vision + decision snapshot (selected agent only) |
| GPU → CPU | When `try_collect_agent_state` is called | One agent's full `AgentBrainState` (for inheritance, debugging) |
| GPU only | Per-tick simulation | Brain state, pattern memory, physics state, food state, **sensory features** (written by the vision pass into `sensory_buf`) |

The asymmetry is intentional. Sensory frames are not packed and uploaded per tick: the vision pass raycasts on the GPU from terrain/biome/food/agent buffers and writes the feature layout directly into `sensory_buf`. The brain stage in the next kernel-batch reads its inputs from that same `sensory_buf` — a one-batch sensory lag that amortizes the cost of vision + grid rebuild over `vision_stride` brain cycles.

### Non-Blocking Readback

`GpuKernel` never blocks for state. Three independent staging-buffer tracks (`state_readback`, `telemetry_readback`, `agent_state_readback`), each driven by an `async_readback::ReadbackTracker`, fence pending readbacks against GPU completion. The CPU calls `try_collect_*` each frame; if the buffer hasn't been mapped yet, the call returns without producing data and the UI uses the previous frame's cached snapshot. This decouples render-loop pacing from GPU completion time.

`read_full_state_blocking` and `read_agent_telemetry_blocking` exist for tests and one-shot debugging — production code uses the non-blocking path.

### Buffer Allocation

All buffers are created at `GpuKernel::new` with sizes proportional to `agent_count` and the vision dimensions encoded in `BrainLayout`. Persistent storage buffers (`brain_state`, `pattern_buffer`, `physics_state`, `food_state`, sensory) use `STORAGE | COPY_SRC | COPY_DST`, and transient working buffers use `STORAGE` only; the double-buffered world-config and the heritable brain-config are uniform buffers (`UNIFORM | COPY_DST`). Staging buffers for readback use `MAP_READ | COPY_DST` and are sized for the worst-case message (full state for `state_readback`, one agent's slice for the others).

---

## 5. Buffer Layout

Buffer offsets, strides, and dimension constants live in two coordinated source-of-truth slots: the Rust side in `crates/xagent-brain/src/buffers.rs` (`BrainLayout`, `PHYS_STRIDE`, `PATTERN_STRIDE`, the `O_*` offset constants, `ENCODED_DIMENSION`, `MEMORY_CAP`, …), and the WGSL side in `crates/xagent-brain/src/shaders/kernel/common.wgsl` as `override` constants (`override VISION_W: u32 = 8u; override SENSORY_STRIDE: u32 = …; override O_ENC_BIASES: u32 = FEATURE_COUNT * ENCODED_DIMENSION; …`). At pipeline creation time `gpu_kernel.rs` concatenates `common.wgsl` with the relevant phase fragments via `include_str!` and sets the matching `override` values on the `ComputePipelineDescriptor`, so the WGSL constants resolve to whatever the live `BrainLayout` produced from the configured vision dimensions.

### Core Dimensions

| Constant | Value (default 8×6) | Description |
|----------|--------------------:|-------------|
| `ENCODED_DIMENSION` | 128 | Internal encoded state dimensionality (`crates/xagent-brain/src/buffers.rs`) |
| `BrainLayout::feature_count` | 265 = `VISION_RAYS * 5 + 25` | Feature vector size (192 RGBA + 48 depth + 25 derived non-visual; scales with `VISION_W`/`VISION_H`) |
| `MEMORY_CAP` | 128 | Maximum patterns per agent |
| `RECALL_K` | 16 | Top-K recalled patterns per tick |
| `ERROR_HISTORY_LEN` | 128 | Prediction-error ring-buffer size |
| `TD_DISCOUNT` / `TD_LAMBDA` | 0.97 / 0.9 | TD(λ) credit horizon and trace decay |

The feature/encoded sizes and `BrainLayout::brain_stride` scale with the configured vision dimensions (via `feature_count`). `PATTERN_STRIDE` does **not** — it is a fixed `pub const` derived from `MEMORY_CAP` and `ENCODED_DIMENSION`. `BrainLayout::new(vision_width, vision_height)` is the single source of truth for the vision-dependent values — see `crates/xagent-brain/src/buffers.rs`. For the default 8×6 layout (`ENCODED_DIMENSION = 128`, `feature_count = 265`): `brain_stride = 51,898` f32, with the fixed `PATTERN_STRIDE = 17,539` f32. The matching `O_*` offsets and per-region sizes are surfaced to WGSL via the `override` constants in `common.wgsl`, with the values supplied at pipeline creation by `gpu_kernel.rs`.

### Sensory Buffer Layout (GPU-produced)

```
[  192 RGBA vision  |  48 depth  |  vel(3)  fac(3)  ang(1)  e(1)  i(1)  ed(1)  id(1)  touch(16)  ]
 ^                   ^            ^                                                                ^
 0                   192          240                                                              267
```

Total for the default 8×6 vision: `SENSORY_STRIDE = 267` f32 per agent. In the live `GpuKernel` runtime this layout is written directly into `sensory_buffer` by the vision pass — RGBA + depth come from `phase_vision_raycast`, and the non-visual tail (velocity, facing, angular velocity, normalized energy/integrity, energy/integrity deltas, and up to `MAX_TOUCH_CONTACTS` × 4-channel touch contacts) is written by `phase_vision_senses`. Touch contacts are filled in 3×3-cell discovery order, food cells before agent cells, and stop at `MAX_TOUCH_CONTACTS`; unused slots stay zeroed. The CPU-side `buffers::pack_sensory_frame()` mirrors this layout but is only used by `buffers` tests — it is not in the per-tick data path.

### Brain State Buffer (per agent: `BrainLayout::brain_stride`, 51,898 f32 for the default 8×6 layout)

Regions (in offset order; concrete offsets are dimension-dependent and emitted by `BrainLayout` — see `crates/xagent-brain/src/buffers.rs`):

- `O_ENCODER_WEIGHTS` — `feature_count * ENCODED_DIMENSION` (= 33,920 for 8×6) encoder weight matrix.
- `O_ENCODER_BIASES` — `ENCODED_DIMENSION` (128) per-dimension bias.
- `O_PREDICTOR_WEIGHTS` — `PREDICTOR_DIMENSION * ENCODED_DIMENSION` predictor matrix (operates in encoded space).
- `O_PREDICTOR_CONTEXT_WEIGHT` and the rest of the fixed-size tail (`FIXED_TAIL_SIZE`): predictor error ring, habituation EMA + attenuation, previous-encoded snapshot, homeostasis state, action/turn policy weights + biases, exploration rate, motor-fatigue ring + cursor + factor + length, previous prediction, tick counter, heritable config, per-agent `movement_speed`, and the TD critic state — value weights (`O_VALUE_WEIGHTS`, `ENCODED_DIMENSION`) + bias + previous value, the three eligibility-trace vectors (`O_TRACE_CRITIC` / `O_TRACE_FWD` / `O_TRACE_TURN`, `ENCODED_DIMENSION` each), and the three scalar trace biases (`O_TRACE_BIASES`).

### Pattern Memory Buffer (per agent: `PATTERN_STRIDE` = 17,539 f32, a fixed constant)

Stores `MEMORY_CAP` (= 128) patterns. Regions: `O_PAT_STATES` (`MEMORY_CAP * ENCODED_DIMENSION` encoded-space states), `O_PAT_NORMS` (cached L2 norms), `O_PAT_REINF` (per-pattern reinforcement that decays over time), `O_PAT_MOTOR` (`[forward, turn, outcome_valence] * MEMORY_CAP`), `O_PAT_META` (`[created_at, last_accessed, activation_count] * MEMORY_CAP`), `O_PAT_ACTIVE` (active flag — recall is gated here, not on `O_PAT_REINF`), and `O_ACTIVE_COUNT` bookkeeping. The `O_PAT_*` offsets are fixed constants (derived from `MEMORY_CAP` and `ENCODED_DIMENSION`, independent of vision dimensions); see `crates/xagent-brain/src/buffers.rs`.

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

**What it does**: Attenuates repetitive encoded dimensions and produces a habituated state consumed by the predictor path (prediction error, predictor matmul, predictor training). Memory recall, the policy, TD credit, and pattern storage operate on the raw encoded (pre-habituation) state so attenuation can never silence them.

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

**What it does**: Computes cosine similarity between the encoded (pre-habituation) state and all 128 memory patterns. Inactive slots receive a sentinel score of `-2.0`.

**How it works**:

```
For each pattern j in [0, MEMORY_CAP):
    if not active: sim[j] = -2.0
    else: sim[j] = clamp(dot(encoded, pattern[j]) / (||encoded|| * ||pattern[j]||), -1.0, 1.0)
```

Pattern norms are pre-cached in `O_PAT_NORMS` (written during pattern storage in pass 7), avoiding redundant norm computation. The query norm is computed once per agent at the start of the pass. Querying with the pre-habituation state keeps recall alive during sustained stimuli — attenuation would otherwise mute the query exactly when remembering matters most.

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

#### 6.6.3 Credit Assignment — TD(λ) actor-critic

A linear value head over the encoded state estimates the discounted
homeostatic return; its TD error is the single credit signal for the
critic, both policy channels, and the encoder. Per brain tick:

1. **Value estimate**: `v = dot(value_weights, encoded) + value_bias` (threads compute partial products in parallel; thread 0 reduces).
2. **TD error**: `δ = clamp(reward + TD_DISCOUNT·v − prev_value, ±MAX_TD_ERROR)` where `reward` is the urgency-amplified homeostatic delta since the previous brain tick and `TD_DISCOUNT = 0.97` gives a ~33-brain-tick horizon (the travel time from the edge of vision range at default speed).
3. **Weight updates through eligibility traces**: `w += lr · TD_VECTOR_SCALE · δ · z` for the critic (`CRITIC_LEARNING_RATE = 0.01`) and both actor channels (`ACTION_WEIGHT_LEARNING_RATE = 0.10`). `TD_VECTOR_SCALE = 1/ENCODED_DIMENSION` keeps the aggregate step inside the linear-TD stability limit regardless of dimensionality. Scalar biases update from scalar traces without the vector scale.
4. **Trace update** (end of the pass, after the motor block produces this tick's exploration noise): `z ← TD_DISCOUNT·TD_LAMBDA·z + term`, where the critic trace accumulates `encoded[d]` and the actor traces accumulate `noise·encoded[d]` — the likelihood-ratio direction of the action actually taken. `TD_LAMBDA = 0.9`.

There is no deadzone, no tonic fallback, no pain amplifier, and no history
ring: credit reaches past actions through the traces, and the critic's
bootstrapping propagates reward backwards across repeated experiences
beyond the raw trace span. Traces and `prev_value` are episodic — zeroed on
death — while the value weights are learned knowledge and survive respawn
and inheritance.

**Weight normalization**: forward, turn, and value weight vectors are
clipped to L2 norm <= `MAX_WEIGHT_NORM = 2.0` (synaptic homeostasis). There
is no per-tick weight decay: TD updates are surprise-driven and stop when
δ calibrates to zero, so decay would only erase accumulated policy
knowledge (including the initial forward bias that provides exploration
mobility).

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
1. Publishes `[noise_fwd * exploration_rate, noise_trn * exploration_rate]` to shared memory for the parallel eligibility-trace update at the end of the pass. The traces carry the exploration noise (not the full motor) so only noise directions that correlate with TD errors get reinforced.
2. Saves the prediction to `O_PREV_PREDICTION` for next tick's error computation.
3. Increments `O_TICK_COUNT`.
4. Writes `[prediction(ENCODED_DIMENSION), credit_signal(ENCODED_DIMENSION), fwd, trn, strafe, td_error]` to the decision buffer for pass 7 and CPU readback.

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

The encoder weights are adapted based on the credit signal from pass 6
(`credit_signal[i] = td_error * (forward_trace[i] + turn_trace[i])` — which
encoded dimensions carried the policy's eligibility when the outcome
arrived):

```
for each (i, j) where |credit_signal[i]| > 1e-6:
    weights[j * DIM + i] += learning_rate * credit_signal[i] * ENCODER_CREDIT_SCALE * features[j]
    weights[j * DIM + i] = clamp(weights, -2.0, 2.0)
```

This is a Hebbian-style update: features that co-occur with strong credit signals have their encoder weights strengthened. The `ENCODER_CREDIT_SCALE = 0.1` factor makes encoder adaptation slower than action learning, reflecting the intuition that the perceptual representation should change gradually while the policy adapts quickly.

#### 6.7.3 Memory Reinforcement

Active patterns with cosine similarity > 0.3 to the current encoded (pre-habituation) state are reinforced:

```
reinforcement[j] += sim * learning_rate * (1 - pred_error)
```

Low prediction error strengthens matching patterns more -- successful prediction means the memory is accurate.

**Retroactive valence update**: Similar patterns have their `outcome_valence` nudged toward the current homeostatic gradient via an EMA: `valence += sim * (learning_rate * 0.3) * (gradient - valence)`. This lets the agent update its assessment of past situations.

#### 6.7.4 Pattern Storage

Each tick, the current encoded (pre-habituation) state is stored to the weakest memory slot (the one with the lowest reinforcement, tracked at `O_MIN_REINF_IDX`), keeping memory keys in the same space as the recall queries:

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
| **Fear / Avoidance** | Damage produces a negative per-tick homeostatic delta (urgency-amplified — the TD reward) --> negative TD error --> eligibility traces blame the recently active state-action directions --> policy weights learn to avoid danger-associated features, while the value head marks danger-correlated states as low-value so later TD errors penalize approaching them | `coop_habituate_homeo`, `coop_predict_and_act` (TD credit) |
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
    // Persistent buffers: brain_state, pattern_buffer,
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
| `upload_world_config(start_tick, ticks_to_run)` | Writes the per-batch world-config uniform consumed by all four passes inside a kernel-batch (start tick, batch size, stride parameters). Called internally by `dispatch_ticks`. |
| `dispatch_ticks(start_tick, ticks_to_run) -> bool` | **Compute only.** Splits the work into full kernel-batches of `vision_stride * brain_tick_stride` ticks, plus a shorter remainder kernel-batch of `remainder_cycles * brain_tick_stride` ticks when `brain_cycles % vision_stride != 0`, plus an optional physics-only remainder for the trailing `ticks_to_run % brain_tick_stride` ticks that do not fill a brain cycle. Each kernel-batch is one command-buffer + `queue.submit()` running `prepare → kernel → global → vision`; the physics-only remainder is a separate submit that masks brain/vision off. Does **not** copy any state into staging — CPU-visible publication is `request_state_snapshot`'s job. Always returns `true`. |
| `request_state_snapshot() -> bool` | **Publication only.** Scans the staging ring for a free slot and, when one exists, copies `agent_phys` (and `food_state` when food exists) into it and installs the async mapping that `try_collect_state_snapshot` later collects. Returns `false` when every slot is in flight — the request is dropped, never queued, so stale state is never preferred over advancing compute. Independent of `dispatch_ticks`: compute may advance many times between snapshot requests. |
| `dispatch_batch(start_tick, ticks_to_run) -> bool` | Compatibility wrapper: calls `dispatch_ticks` then `request_state_snapshot`, preserving the original "advance compute and opportunistically stage a readback" behavior. New runtime code schedules the two separately so dispatch cadence and publication cadence are independent. Returns `dispatch_ticks`'s result (always `true`). |
| `dispatch_batch_masked(start_tick, ticks_to_run, phase_mask)` | Variant for tests and benchmarks that gates which phases run per cycle (bit 0 = physics, bit 1 = vision, bit 2 = brain). Iterates cycles in chunks of 100 (Metal command-buffer deadlock workaround), appends a remainder physics-only pass, copies `agent_phys` to the active staging slot, then blocks on `device.poll(Wait)` for GPU completion. Does **not** run any global pass (no grid rebuild, food respawn, or collisions) and does **not** update `cached_state` — call sites read GPU buffers directly. |
| `try_collect_state_snapshot() -> bool` | Non-blocking: polls all in-flight state-readback slots; if any is ready, copies the most recent into `cached_state` (and `cached_food_state` when food exists). Returns `true` on update. Collects whatever `request_state_snapshot` (or the `dispatch_batch` wrapper) previously staged. |
| `cached_state() -> &[f32]` | Latest physics-state snapshot (positions, vitals, motor, death counts, telemetry slots). |
| `cached_food_state() -> Option<&[f32]>` | Latest food-state snapshot when available. |
| `read_full_state_blocking() -> &[f32]` | Test/debug only: blocks until the next state readback completes. |
| `request_agent_telemetry(index)` | Schedules a one-agent telemetry readback (vision + decision snapshot). |
| `try_collect_telemetry() -> Option<(u32, AgentTelemetry)>` | Non-blocking: returns `(agent_index, telemetry)` when the staged copy is mapped, where `agent_index` is the agent the readback was requested for (so callers can reject telemetry that completed after a selection change). |
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
}
```

Used for cross-generation inheritance (the governor reads parent state, mutates it, writes to offspring) and mutation. The two vectors are the exact GPU buffer contents for one agent slice. Only the `brain_state` slice length is vision-dependent (`BrainLayout::brain_stride`, via `feature_count`); the `patterns` slice uses the fixed `PATTERN_STRIDE` constant. The learned policy and the TD value head live in `brain_state` and so are inherited; the episodic eligibility traces also live there but are zeroed on the first post-respawn tick of a new life.

### Death / Respawn on the GPU

Death detection and respawn live entirely in WGSL (`phase_death.wgsl`, invoked from `kernel_tick.wgsl` after the physics step). There is no `death_signal` Rust call. When the kernel decides an agent has died (energy ≤ 0 or integrity ≤ 0):

1. **Spawn search**: tries up to 50 GPU-RNG samples for a non-Danger biome position; if all 50 attempts land in Danger biomes, the `!found` branch reuses the attempt-0 sample (RNG seed `tick * 256 + agent_id`, the same draw as attempt 0) and spawns there without re-checking the biome — so a fully Danger-blocked agent can land back in a Danger cell (see `phase_death.wgsl` / `kernel_tick.wgsl::agent_death_respawn`).
2. **Physics reset**: full energy, full integrity, zero velocity, facing +Z; death count incremented; fitness counters (`food_count`, `ticks_alive`, `last_death_tick`) preserved.
3. **Memory trauma**: all `O_PAT_REINF` entries are multiplied by `0.5`. The death pass leaves `O_PAT_ACTIVE` untouched, so recall (which gates on `O_PAT_ACTIVE` in `brain_passes.wgsl`, not on reinforcement) is not cut off by this step. Halved reinforcement only makes subsequent decay reach the `<= 0.0` deactivation point sooner for the weakest patterns; the strongest memories survive.
4. **Brain reset**: homeostasis EMAs zeroed, exploration rate set to `0.5`, habituation EMAs zeroed and attenuation reset to `1.0`, fatigue factor reset to `1.0`, position-ring staleness state cleared, TD eligibility traces and previous-state value zeroed (the value weights survive — they are learned knowledge, not episodic state).

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

### Why Shared `override` Constants in `common.wgsl`

Every shader in the kernel pipeline needs the same set of 50+ offset constants and dimension values (`SENSORY_STRIDE`, `FEATURE_COUNT`, `ENCODED_DIMENSION`, the `O_*` offsets, etc.). Manually keeping these in sync between Rust and WGSL would be a maintenance nightmare. The chosen mechanism: `common.wgsl` declares each constant as a WGSL `override` derived from `VISION_W`/`VISION_H` (`override SENSORY_STRIDE: u32 = VISION_COLOR_COUNT + VISION_DEPTH_COUNT + 27u;`), `gpu_kernel.rs` concatenates `common.wgsl` with each phase fragment via `include_str!`, and the matching `BrainLayout`-derived values are bound to the pipeline as override values at pipeline creation time. Utility functions (`fast_tanh`, `pcg_hash`, `rand_f32`, `rand_normal`) live as plain WGSL functions inside `common.wgsl` and are inlined into every concatenated shader. A single source of truth (`BrainLayout` on the Rust side, the `override` declarations on the WGSL side), zero chance of offset mismatch as long as the override values are forwarded at pipeline creation.

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

The old CPU architecture used a discrete 8-action space with a learned preference table. The GPU rewrite replaces this with continuous forward/turn output computed as `dot(weights, encoded) + bias`. This is both simpler (no action table, no softmax, no argmax tie-breaking) and more expressive (the agent can move at any speed and turn at any angle). Credit assignment updates the weight vectors directly through eligibility traces, which is natural for continuous outputs.

### Why TD(λ) Instead of Windowed REINFORCE

The earlier learner credited a fixed window of recorded actions with the *change* in homeostatic gradient since each action, gated by a deadzone and a tonic fallback. Food, however, is a sparse reward that arrives ~30 brain ticks after the navigational turn that earned it — far outside any practical exponential-decay window — so the decisive turn was never credited (the core finding of issue #13). TD(λ) closes that gap two ways: a learned value head bootstraps, propagating the terminal food reward backwards across repeated experiences, and eligibility traces give every state-action a decaying claim on future TD errors. The deadzone, tonic fallback, and pain amplifier are gone — δ is a single signed signal that calibrates to zero when the critic is accurate, so steady metabolic drain stops producing spurious updates without any thresholding.

### Why No Per-Tick Weight Decay

TD updates are surprise-driven: they vanish when δ calibrates to zero, so weights settle rather than diverge, and the L2-ball clamps bound magnitude. Per-tick decay would instead bleed away accumulated policy knowledge every tick — including the initial forward bias that gives agents the mobility to explore — so it is omitted. The value head likewise keeps its learned landscape across the lifetime; only the episodic eligibility traces and previous-value scalar reset on death.

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
