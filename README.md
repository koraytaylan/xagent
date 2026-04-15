# xagent — Emergent Cognitive Agent Platform

> **No hardcoded behaviors. No reward functions. No explicit goals.**
> Fear, curiosity, attention, and habit all emerge from capacity constraints, sensory experience, and homeostatic pressure.

## 1. Project Vision

**xagent** is an emergent cognitive agent platform that explores a radical hypothesis: complex, intelligent-looking behavior doesn't need to be designed — it can *emerge* from a handful of simple principles operating under resource constraints. Each agent in the simulation has a brain built on **predictive processing**: it constantly predicts what will happen next, compares that prediction to reality, and uses the resulting *prediction error* to update everything — its internal model, its memories, and its actions.

There are no reward signals, no utility functions, no goal hierarchies. The only evaluative signal in the entire system is **homeostatic stability** — whether the agent's internal physiological variables (energy, physical integrity) are trending toward or away from equilibrium. This is inspired by the **free energy principle** from computational neuroscience (Karl Friston): organisms don't optimize for reward; they minimize surprise relative to their expectations of continued existence.

The result is a platform for observing genuinely emergent cognition. An agent placed in a hostile environment doesn't "know" that food is good or lava is bad. It must discover this through experience. It must learn that certain visual patterns predict energy gains, that certain actions in certain contexts improve its homeostatic state. Memory is finite, so it must forget. Processing is bounded, so it must *attend*. From these constraints alone, behaviors reminiscent of curiosity, caution, habit formation, and adaptation arise — none of them explicitly programmed.

---

## 2. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        Cargo Workspace                            │
│                                                                   │
│  ┌───────────────────┐                                            │
│  │  xagent-shared    │◄────────────────────────────────────────┐  │
│  │  (interface)      │    Types, traits, config                │  │
│  └────────┬──────────┘                                         │  │
│           │                                                    │  │
│           │  depends on                      depends on        │  │
│           ▼                                                    │  │
│  ┌───────────────────┐     SensoryFrame     ┌────────────────┐ │  │
│  │  xagent-brain     │◄─────────────────────│                │ │  │
│  │  (cognition)      │                      │ xagent-sandbox │ │  │
│  │                   │─────────────────────►│ (world + app)  │ │  │
│  │  Encode → Recall  │     MotorCommand     │                │ │  │
│  │  Predict → Learn  │                      │ 3D World       │ │  │
│  │  Act              │                      │ Physics        │ │  │
│  └───────────────────┘                      │ Renderer       │─┘  │
│                                             └────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

### Crate Responsibilities

| Crate | Role |
|-------|------|
| **`xagent-shared`** | Interface contract. Defines `SensoryFrame`, `MotorCommand`, `BodyState`, `BrainConfig`, `WorldConfig`, and the `CognitiveArchitecture` trait. No logic — just types. |
| **`xagent-brain`** | Cognitive architecture. Implements predictive processing: sensory encoding, pattern memory, state prediction, homeostatic monitoring, capacity management, and action selection. |
| **`xagent-sandbox`** | 3D world simulation + application. Procedural terrain with biomes, food/hazard systems, physics, multi-agent support with evolution, wgpu-based rendering, egui IDE-like UI (sortable sidebar, agent detail tabs with vision display, mini-map, decision stream, and replay controls, console), per-generation replay recording/playback, CSV logging, and the main event loop. |

### Communication Flow

```
Sandbox                          Brain
  │                                │
  │  1. Build SensoryFrame         │
  │     (vision, touch, energy,    │
  │      integrity, velocity)      │
  │ ──────────────────────────────►│
  │                                │  2. Encode → Recall → Predict
  │                                │     → Compare → Learn → Act
  │  3. Apply MotorCommand         │
  │     (forward, strafe, turn,    │
  │      consume/push/jump)        │
  │◄────────────────────────────── │
  │                                │
  │  4. Step physics, update world │
  │                                │
  └────────────────────────────────┘
```

> **Core Principle: Semantic Opacity** — The brain never sees the named fields of `SensoryFrame`. The encoder flattens vision, touch, energy, and all other modalities into a single opaque `Vec<f32>`. The brain has no concept of "eyes," "hunger," or "other agents" — only numerical patterns. All meaning is *discovered* through prediction error and homeostatic correlation, not provided through labels. This is the architectural foundation that makes emergence genuine rather than engineered. See the [brain crate README](crates/xagent-brain/README.md#the-brain-has-no-eyes) for the full explanation.

---

## 3. The Cognitive Architecture

Each agent brain is a 7-stage predictive processing pipeline running entirely on GPU:

```
feature_extract → encode → habituate_homeo → recall_score → recall_topk → predict_and_act → learn_and_store
```

### Dispatch Modes

Two dispatch modes are available:

- **GpuBrain (7-pass)**: One compute dispatch per pass per tick, one thread per agent (`@workgroup_size(1)`). Simple, inspectable, used for small-scale experiments.
- **GpuKernel (fused)**: All per-agent computation (physics, food detection, death/respawn, and all 7 brain passes) fused into a single dispatch per `vision_stride` cycles. Each agent gets a 256-thread workgroup. A separate global pass handles grid rebuild, collisions, and vision raycasting. This achieves 60,000+ brain ticks/second at 10 agents — a 100× improvement over per-tick dispatch.

The `vision_stride` parameter (default 10) controls how many brain+physics cycles run between global passes (grid rebuild, collision, vision). Higher values mean more brain throughput but less frequent sensory updates.

### Step-by-Step

Each pass runs as a WGSL compute shader dispatched over all agents in parallel.

1. **Feature Extract** — Extracts 217 features from the packed sensory input (192 RGBA vision + 48 depth + 27 non-visual). This is the semantic firewall: the frame's named fields (vision, energy, touch) are flattened into an opaque feature vector, and from this point on the brain operates without any knowledge of what the numbers originally represented.

2. **Encode** — Projects features through a learned weight matrix and `fast_tanh` into a 32-dimensional encoded state. This fixed-size representation is the common currency of all downstream passes.

3. **Habituate & Homeostasis** — Attenuates encoded dimensions that haven't changed recently (habituation EMA), producing a habituated state that suppresses monotonous input. Simultaneously computes multi-timescale homeostatic gradients (fast ≈ 5 ticks, medium ≈ 50 ticks, slow ≈ 500 ticks) and urgency from energy and integrity signals.

4. **Recall Score** — Computes cosine similarity between the habituated state and all 128 stored memory patterns, producing a score vector that identifies the most contextually relevant past experiences.

5. **Recall Top-K** — Selects the 16 most similar patterns from the score vector and updates their recall metadata (timestamps, access counts).

6. **Predict & Act** — Computes prediction error from the previous tick's prediction against the current encoded state. Performs credit assignment over recent action history weighted by homeostatic gradient. Evaluates the linear policy with prospection blending (predicted-future state + top-recalled-memory blend), applies exploration noise (adaptive rate 10–85%) and motor fatigue dampening (repetitive commands are attenuated, forcing loop-breaking).

7. **Learn & Store** — Predictor gradient descent step, encoder Hebbian weight adaptation, memory reinforcement for patterns co-occurring with low error, pattern storage to the weakest slot, and per-pattern decay.

### Homeostatic Feedback

The `HomeostaticMonitor` tracks energy and integrity signals across three timescales (fast ≈ 5 ticks, medium ≈ 50 ticks, slow ≈ 500 ticks) using exponential moving averages. It produces:

- **Gradient** — positive = improving, negative = worsening. Modulates learning rate and action credit.
- **Urgency** — non-linear distress curve (configurable exponent, default quadratic). Near-critical levels suppress exploration.

This is the **only** evaluative signal. There is no reward function.

### Capacity Constraints

| Constraint | Effect |
|---|---|
| `memory_capacity` | Finite pattern storage → forced forgetting → what survives = what matters |
| `processing_slots` | Limited recall per tick → forced prioritization → attention-like behavior |
| `representation_dimension` | Fixed encoding size → forced compression → abstraction |

See the [brain crate README](crates/xagent-brain/README.md) for a deep dive into each component.

---

## 4. The Sandbox World

The sandbox is a real-time 3D environment rendered with **wgpu** (WebGPU/Vulkan/Metal/DX12 backend), wrapped in an **IDE-like UI** built with **egui 0.31 + egui_dock 0.16**. The UI provides: a **top bar** (FPS, agent count, evolution state, wall time, ticks/sec, pause/resume/reset controls), a sortable **left sidebar** with the agent list, a **main dock area** with the 3D viewport and agent detail tabs (featuring vision display, top-down mini-map, decision stream, and replay playback controls), and a **bottom console** for evolution event logs.

### Terrain & Biomes

- **Procedural terrain** generated with multi-octave Perlin noise (height range ±5 units)
- **256×256 unit** world (configurable) with bilinear height interpolation
- Three biome types determined by noise thresholds:

| Biome | Color | Effect |
|-------|-------|--------|
| **FoodRich** | Green | Food items spawn here |
| **Barren** | Tan/brown | No food, no hazards |
| **Danger** | Red | Damages agent integrity each tick |

### Agent Bodies

- Physical simulation: gravity, locomotion, collision with terrain
- Internal state: energy (depletes over time and with movement) and integrity (damaged by hazards, regenerates when energy > 50%)
- Sensory apparatus: 8×6 raycast vision (ray step 1.0, detects terrain, food, and other agents), touch contacts (food, terrain edges, hazards, other agents — top 4 encoded into brain), proprioception, interoception
- **Agent vision**: ray marching detects terrain (biome-colored), food items (lime green `[0.70, 0.95, 0.20, 1.0]`), and other agents (magenta `[0.9, 0.2, 0.6, 1.0]`) in the visual field
- **Agent-agent collision**: physical collision resolution pushes overlapping agents apart (2-unit minimum separation)
- Death occurs when energy or integrity reaches zero → respawn with optional brain persistence

### Brain Persistence & Death Guardrails

When brain persistence is enabled (default), three guardrails prevent suicide loops:

1. **Partial respawn energy** — 50% energy, 70% integrity (not full health). Dying is not a "free heal."
2. **Random respawn position** — agent reappears at an unpredictable location.
3. **Memory trauma** — `brain.trauma(0.2)` applies 20% reinforcement decay on death. Weakest memories are wiped; strongest survive. Models the cognitive cost of catastrophic discontinuity.

Suicide prevention is emergent: death is maximally unpredictable (massive prediction error), and the brain's core drive is minimizing prediction error.

### Agent Palette Colors

Each agent is assigned a **static palette color** at spawn. The same color is used in the 3D viewport and the sidebar agent list (with an sRGB→linear conversion for correct GPU rendering). Dead agents render as dark gray `[0.25, 0.25, 0.25]`. This makes it easy to track individual agents across the sidebar and the 3D world at a glance.

### Agent Trails

All alive agents render their movement history simultaneously as **linear ribbon trails**, each in the agent's own palette color. Positions are distance-sampled (new point every ≥ 3 units moved, up to 4000 points per life). Trail meshes are rebuilt only when data changes (dirty flag), keeping overhead negligible at high speed multipliers. Trails reset on death/respawn.

### Multi-Agent & Food Scarcity

- Up to 100 concurrent agents (raised from 20)
- **Food respawn**: 10-second timer, relocates to a new random food-rich position — prevents camping, forces foraging
- **Reproduction**: currently disabled to focus on individual learning rather than evolution
- **Directed mutations**: Per-parameter momentum vectors (one per island) bias mutations toward directions that previously improved fitness. Momentum decays each generation (configurable via `momentum_decay`) so stale signals fade. This provides directional bias, emergent correlated mutations, and selective mutation focus — all without hardcoded parameter relationships.

See the sandbox crate source for full implementation details.

---

## 5. Quick Start

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (edition 2021)
- A GPU with Vulkan, Metal, or DX12 support (for rendering)

### Build & Run

```bash
# Build
cargo build --release

# Run with default settings
cargo run --release

# Run with presets
cargo run --release -- --brain-preset tiny --world-preset hard

# Run headless (simulation + CSV logging, no window)
cargo run --release -- --no-render

# Set a specific random seed
cargo run --release -- --seed 1234

# Override tick rate
cargo run --release -- --tick-rate 60

# Export current config as JSON
cargo run --release -- --dump-config > experiment.json

# Run from a saved config
cargo run --release -- --config experiment.json
```

---

## 6. Controls Reference

### Keyboard & Mouse

| Key | Action |
|-----|--------|
| `W` / `A` / `S` / `D` | Move camera forward / left / backward / right |
| `E` | Move camera up |
| `Shift` | Move camera down |
| Mouse drag (on viewport) | Rotate camera (yaw/pitch) |
| Scroll wheel (on viewport) | Zoom camera in/out |
| `P` / `Space` | Pause / resume simulation |
| `1` | Speed 1× (1 tick/frame) |
| `2` | Speed 2× (2 ticks/frame) |
| `3` | Speed 5× (5 ticks/frame) |
| `4` | Speed 10× (10 ticks/frame) |
| `5` | Speed 100× (100 ticks/frame) |
| `6` | Speed 1000× (1000 ticks/frame) |
| `R` | Toggle brain persistence on death (persist / reset) |
| `N` | Spawn new agent (default config) |
| `M` | Spawn mutated agent (±10% parameter variation) |

### egui UI Interaction

| Action | Effect |
|--------|--------|
| Sort dropdown (sidebar top) | Sort agent list by ID, Energy, Integrity, Deaths, etc. |
| Click agent in sidebar | Select / focus that agent |
| Double-click agent in sidebar | Open an agent detail tab in the dock area |
| Drag / scroll on viewport pane | Camera rotation / zoom (only when hovering the viewport) |
| Pause / Resume button (top bar) | Pause or resume the simulation (visible during Running/Paused) |
| Reset button (top bar) | Reset the evolution run (visible during Running/Paused) |
| "Replay Gen N" button (detail tab) | Enter replay mode for the last completed generation |
| Timeline scrubber (replay mode) | Scrub through recorded ticks |
| Close detail tab | Click the × on the tab header |

Camera controls (drag, scroll) are routed to the 3D viewport only when the pointer is hovering over it; interacting with the sidebar or detail tabs does not move the camera.

---

## 7. Configuration

### Brain Presets

| Preset | `memory_capacity` | `processing_slots` | `visual_encoding_size` | `representation_dimension` | `learning_rate` | `decay_rate` | `distress_exponent` | `habituation_sensitivity` | `max_curiosity_bonus` | `fatigue_recovery_sensitivity` | `fatigue_floor` |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **tiny** | 24 | 8 | 32 | 16 | 0.08 | 0.002 | 2.0 | 20.0 | 0.6 | 8.0 | 0.1 |
| **default** | 128 | 16 | 64 | 32 | 0.05 | 0.001 | 2.0 | 20.0 | 0.6 | 8.0 | 0.1 |
| **large** | 512 | 32 | 128 | 64 | 0.03 | 0.0005 | 2.0 | 20.0 | 0.6 | 8.0 | 0.1 |

**Parameter effects:**

| Parameter | What it controls |
|-----------|-----------------|
| `memory_capacity` | Max stored patterns. Smaller → faster forgetting, stronger capacity pressure. |
| `processing_slots` | Max recall operations per tick. Smaller → narrower attention. |
| `visual_encoding_size` | Resolution of visual encoding (average-pooled bins). Smaller → coarser visual perception. |
| `representation_dimension` | Internal representation vector length. Fixed across generations (not evolved) to preserve weight inheritance. Smaller → more compression, more abstraction. |
| `learning_rate` | Base rate for weight updates (encoder, predictor, memory). Higher → faster adaptation but less stability. |
| `decay_rate` | Rate of memory decay per tick. Higher → more aggressive forgetting, favoring recent experience. |
| `distress_exponent` | Distress curve shape (default 2.0). Higher → calm longer, panic harder at critical levels. Heritable. |
| `habituation_sensitivity` | How fast boredom builds (default 20.0). Higher → faster sensory attenuation. Heritable. |
| `max_curiosity_bonus` | Exploration boost ceiling from monotony (default 0.6). Higher → stronger curiosity drive. Heritable. |
| `fatigue_recovery_sensitivity` | How easily motor fatigue lifts (default 8.0). Higher → faster recovery. Heritable. |
| `fatigue_floor` | Minimum motor output under fatigue (default 0.1). Lower → harsher dampening. Heritable. |
| `vision_rays` | Number of vision rays, W×H (default 48 = 8×6). Affects sensory buffer size. |
| `brain_tick_stride` | Physics ticks per brain+vision cycle (default 4). Higher → faster but less responsive. |
| `vision_stride` | Brain cycles between global passes — grid rebuild, collisions, vision (default 10). Higher → more brain throughput, less frequent vision updates. |
| `metabolic_rate` | Multiplier for all energy costs (default 0.5). Lower → agents survive longer. |
| `integrity_scale` | Multiplier for integrity damage and regen (default 0.5). Higher → deadlier hazards. |

### World Presets

| Preset | `energy_depletion` | `movement_cost` | `hazard_damage` | `food_density` | `food_value` |
|--------|---:|---:|---:|---:|---:|
| **easy** | 0.005 | 0.002 | 0.05 | 0.005 | 30.0 |
| **normal** | 0.01 | 0.005 | 0.1 | 0.002 | 20.0 |
| **hard** | 0.02 | 0.01 | 0.2 | 0.001 | 15.0 |

Additional world parameters: `world_size` (default 256), `integrity_regen_rate` (0.005), `tick_rate` (30 Hz), `seed` (42).

### JSON Config Format

```json
{
  "brain": {
    "memory_capacity": 128,
    "processing_slots": 16,
    "visual_encoding_size": 64,
    "representation_dimension": 128,
    "learning_rate": 0.05,
    "decay_rate": 0.001,
    "distress_exponent": 2.0,
    "habituation_sensitivity": 20.0,
    "max_curiosity_bonus": 0.6,
    "fatigue_recovery_sensitivity": 8.0,
    "fatigue_floor": 0.1,
    "vision_rays": 48,
    "brain_tick_stride": 4,
    "vision_stride": 10,
    "metabolic_rate": 0.5,
    "integrity_scale": 0.5
  },
  "world": {
    "world_size": 256.0,
    "energy_depletion_rate": 0.03,
    "movement_energy_cost": 0.005,
    "hazard_damage_rate": 1.0,
    "integrity_regen_rate": 0.005,
    "food_energy_value": 20.0,
    "food_density": 0.005,
    "tick_rate": 30.0,
    "seed": 42
  }
}
```

---

## 8. Observing Emergence

### Behavior Phases

The agent progresses through observable phases based on a **composite behavioral score**:

```
score = exploitation_ratio × (1 − prediction_error) × (1 − homeostatic_urgency)
```

This means an agent that avoids danger but starves (high urgency) cannot reach ADAPTED.

| Phase | Composite Score | What You'll See |
|-------|---:|---|
| **RANDOM** | < 2% | Aimless movement, no pattern to actions |
| **EXPLORING** | 2–8% | Starts showing directional preferences, occasionally repeating actions |
| **LEARNING** | 8–20% | Visibly seeking food, avoiding hazards, developing routes |
| **ADAPTED** | ≥ 20% | Efficient foraging, consistent hazard avoidance, deliberate behavior |

### Key Metrics to Watch

These metrics are visible in the **sidebar** (compact vitals, phase label, sortable by any metric) and in **agent detail tabs** (vitals/motor display, vision display showing what the agent sees, mini-map showing the agent's position in the world, scrollable history charts, and a real-time decision stream showing per-tick brain reasoning). The CSV log files also record all metrics per tick. Completed generations can be replayed via the built-in replay system.

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| `prediction_error` (avg) | Decreasing over time | Stuck high or oscillating |
| `exploitation_ratio` | Increasing over hundreds of ticks | Stuck near 0 |
| `memory_utilization` | Climbing toward capacity | Staying near 0 |
| `homeostatic_gradient` | Trending positive | Consistently negative |
| `exploration_rate` | Gradually decreasing | Stuck at max |
| `mean_attenuation` | Near 1.0 (diverse input) | Near 0.1 (habituated, monotonous) |
| `curiosity_bonus` | Low (varied sensory experience) | High (boredom driving exploration) |
| `fatigue_factor` | Near 1.0 (diverse motor output) | Near floor (repetitive, dampened) |

### CSV Log Files

Each run produces a timestamped CSV file (e.g., `xagent_log_2026-03-23_21-06-03.csv`) with per-tick metrics:

| Column | Description |
|--------|-------------|
| `agent_id` | Agent identifier |
| `tick` | Simulation tick number |
| `prediction_error` | Instantaneous prediction error (RMSE) |
| `avg_prediction_error` | Rolling average over last 32 ticks |
| `memory_utilization` | Fraction of memory capacity in use [0.0, 1.0] |
| `memory_capacity` | Total memory slots |
| `exploration_rate` | Current exploration probability [0.05, 0.95] |
| `homeostatic_gradient` | Composite stability gradient (+ = improving) |
| `energy` / `max_energy` | Current and maximum energy |
| `integrity` / `max_integrity` | Current and maximum integrity |
| `position_x/y/z` | World-space position |
| `facing_x/z` | Forward direction vector |
| `biome` | Current biome (FoodRich, Barren, Danger) |
| `action_forward/strafe/turn` | Continuous motor output values |
| `action_discrete` | Discrete action (None, Consume, Push, Jump) |
| `alive` | Whether the agent is currently alive |
| `exploitation_ratio` | Fraction of informed (non-random) actions |
| `decision_quality` | Composite quality score [0.0, 1.0] |
| `behavior_phase` | Current phase label (RANDOM/EXPLORING/LEARNING/ADAPTED) |
| `death_count` | Cumulative deaths |
| `life_ticks` | Ticks alive in current life |
| `generation` | Life iteration — incremented on each death/respawn |

### Experiment Tips

- **Capacity pressure**: `--brain-preset tiny --world-preset hard` — small brain, hostile world. Watch the agent struggle with limited memory and high metabolic demands.
- **Rich environment**: `--brain-preset large --world-preset easy` — lots of capacity, abundant resources. Slower emergence but richer eventual behavior.
- **Seed comparison**: Run the same config with different `--seed` values to see how initial conditions affect learning trajectories.
- **Brain persistence**: Press `R` to toggle whether brains reset on death. With persistence, agents accumulate learning across lives. Without it, each life starts fresh.
- **Multi-agent dynamics**: Press `N` multiple times to spawn several agents, then `M` to add mutated variants. Watch which configurations survive longest.

---

## 9. Project Structure

```
xagent/
├── Cargo.toml                  # Workspace manifest
├── README.md                   # This file
├── rust-toolchain.toml         # Pins Rust stable for CI + local dev
├── cliff.toml                  # git-cliff changelog config
├── .github/
│   └── workflows/
│       ├── ci.yml              # Check + test on push/PR to develop
│       └── release.yml         # Tag-triggered release pipeline
├── crates/
│   ├── xagent-shared/          # Interface contract
│   │   └── src/
│   │       ├── lib.rs          # Re-exports
│   │       ├── body.rs         # BodyState, InternalState
│   │       ├── config.rs       # BrainConfig, WorldConfig, presets
│   │       ├── motor.rs        # MotorCommand, MotorAction
│   │       ├── sensory.rs      # SensoryFrame, VisualField, TouchContact
│   │       └── traits.rs       # CognitiveArchitecture trait
│   │
│   ├── xagent-brain/           # GPU-resident cognitive architecture
│   │   ├── README.md           # Deep dive into brain internals
│   │   └── src/
│   │       ├── lib.rs          # Re-exports, fast_tanh, BrainTelemetry, AgentTelemetry
│   │       ├── gpu_brain.rs    # GpuBrain — 7-pass pipeline, state I/O, resize
│   │       ├── gpu_kernel.rs  # GpuKernel — fused dispatch, telemetry readback
│   │       ├── buffers.rs      # Buffer layout constants, sensory packing, AgentBrainState
│   │       └── shaders/
│   │           ├── feature_extract.wgsl  # Pass 1: sensory → 217 features
│   │           ├── encode.wgsl           # Pass 2: features × weights → 32-dim encoded
│   │           ├── habituate_homeo.wgsl  # Pass 3: habituation EMA + homeostasis
│   │           ├── recall_score.wgsl     # Pass 4: cosine similarity scoring
│   │           ├── recall_topk.wgsl      # Pass 5: top-16 selection
│   │           ├── predict_and_act.wgsl  # Pass 6: prediction, credit, policy, motor output
│   │           ├── learn_and_store.wgsl  # Pass 7: weight updates, memory store/decay
│   │           └── kernel/
│   │               ├── common.wgsl       # Shared constants for fused kernel shaders
│   │               ├── kernel_tick.wgsl    # Fused per-agent kernel (physics+food+death+brain)
│   │               └── global_tick.wgsl  # Grid rebuild + collision pass (1,1,1)
│   │
│   └── xagent-sandbox/         # World simulation + application
│       ├── src/
│       │   ├── main.rs         # CLI, event loop, rendering pipeline
│       │   ├── lib.rs          # Module re-exports
│       │   ├── agent/
│       │   │   ├── mod.rs      # AgentBody, Agent, reproduction, mesh generation
│       │   │   └── senses.rs   # Sensory extraction (raycast vision, touch, interoception)
│       │   ├── world/
│       │   │   ├── mod.rs      # WorldState, mesh building
│       │   │   ├── terrain.rs  # Perlin noise terrain generation
│       │   │   ├── biome.rs    # BiomeMap (FoodRich, Barren, Danger)
│       │   │   └── entity.rs   # FoodItem spawning + respawn
│       │   ├── physics/
│       │   │   └── mod.rs      # Movement, gravity, collision, consumption
│       │   ├── renderer/
│       │   │   ├── mod.rs      # wgpu pipeline, shaders, mesh upload
│       │   │   ├── camera.rs   # Free-fly camera with mouse look
│       │   │   ├── hud.rs      # HUD bar overlay (energy, integrity, etc.)
│       │   │   └── font.rs     # Bitmap font atlas + text rendering
│       │   ├── ui.rs            # egui integration (EguiIntegration, TabViewer, AgentSnapshot, WorldSnapshot, ReplayState)
│       │   ├── replay.rs       # Per-generation replay recording & playback (TickRecord, GenerationRecording)
│       │   └── recording.rs    # CSV metrics logger
│       └── tests/
│           └── integration.rs  # 32 integration tests
└── xagent_log_*.csv            # Generated metric logs (gitignored)
```

---

## 10. Architecture Decisions

### Why Rust?

Real-time simulation with per-tick brain computation demands consistent, low-latency performance. Rust's zero-cost abstractions and lack of garbage collection pauses make it ideal. The ownership model also prevents data races in multi-agent updates — agents can't accidentally alias each other's memory.

### Why wgpu over raw Vulkan?

Pragmatism. wgpu provides a safe, cross-platform GPU abstraction (Vulkan on Linux, Metal on macOS, DX12 on Windows) without the 2000-line boilerplate of raw Vulkan. The rendering needs here are modest — terrain, food cubes, agent cubes, HUD bars — so the thin abstraction cost is negligible.

### Why Predictive Processing?

A single principle (minimize prediction error) gives rise to a remarkably rich set of behaviors without requiring separate modules for each one. Traditional agent architectures have explicit subsystems for perception, planning, motivation, and learning. Here, all of these emerge from the prediction loop operating under constraints. This makes the system both simpler to implement and more scientifically interesting.

### Why No Neural Networks?

Inspectability. Every weight, every pattern, every association in xagent's brain can be examined, traced, and understood. You can watch a specific memory form, see its reinforcement value change, follow its association chain, and understand exactly why the agent made a particular decision. Neural networks, while powerful, are opaque. For a platform designed to study *emergence*, transparency is paramount.

### Why Homeostasis-Only Evaluation?

Minimal assumptions. A reward function encodes the designer's notion of what's "good" — find food, avoid obstacles, reach goals. Homeostasis makes no such assumptions. The agent's only imperative is to keep its internal variables stable. Whether it achieves this by foraging, hiding, or any other strategy is entirely up to the emergent dynamics. This is closer to how biological organisms actually work — they don't optimize for externally defined rewards; they maintain internal equilibrium.

---

## 11. Performance Invariants

The simulation's throughput depends on keeping per-tick work on the GPU. These rules are non-negotiable:

- **Per-tick simulation logic belongs in WGSL shaders, never in Rust.** Physics, brain passes, food detection, death/respawn -- all of it runs in compute shaders. Adding per-tick logic on the CPU side defeats the fused-kernel architecture.
- **The CPU main loop submits GPU dispatches (batched) and collects async readback results (non-blocking).** The Rust side orchestrates dispatches, maps readback buffers, and feeds the UI. It never steps the simulation itself.
- **Recording, telemetry, and history functions run once per frame, sampling the latest state.** CSV logging, replay recording, and UI snapshot updates happen at frame cadence, not tick cadence.
- **No CPU-side simulation work should scale with `ticks_to_run` beyond trivial bookkeeping.** The GPU tick budget is capped at 64,000 ticks per frame, and execution may be split across multiple batched dispatches. Rust may still do lightweight per-tick accounting (for example, counters or governance bookkeeping), but any per-tick simulation, physics, sensing, or brain computation on the CPU violates this invariant.

---

## 12. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full set of code style, naming, GPU/buffer safety, concurrency, performance, testing, and related rules.

---

## 13. Testing

```bash
# Run all tests across the workspace
cargo test --workspace

# Run only brain tests
cargo test -p xagent-brain

# Run only sandbox integration tests
cargo test -p xagent-sandbox

# Run a specific test
cargo test -p xagent-brain -- brain_prediction_error_decreases_with_repeated_input
```

### Test Coverage Summary

**134 tests total** across the workspace:

| Crate | Tests | Scope |
|-------|------:|-------|
| **xagent-brain** | 38 | GPU buffer layout (sensory packing, init, config alignment), per-shader unit tests (feature extraction, encoding, habituation, cosine similarity scoring, top-K selection, motor output validation, weight learning, pattern storage), full pipeline integration (tick produces valid motors), state read/write roundtrip, death signal, resize, multi-agent variance, learning convergence, memory filling, fused kernel tick loop, deterministic bench |
| **xagent-shared** | 1 | Config defaults (vision_stride) |
| **xagent-sandbox** | 95 | Physics (movement, rotation, gravity, NaN sanitization, metabolic brain drain, parallel step_pure correctness), agent lifecycle (energy depletion, death, food consumption, deferred consumption dedup), terrain (determinism, interpolation smoothness, input validation), sensory extraction (vision dimensions, interoception accuracy, GPU/CPU vision parity), spatial grids (FoodGrid query/remove/insert/rebuild, AgentGrid query/rebuild), evolution (config mutation/crossover, fitness evaluation), compute backend probe, benchmark determinism |

---

## 14. Future Directions

- **Multi-agent communication** — Agents could emit and perceive signals (sound, visual markers), enabling emergent social behaviors, cooperation, or competition.
- **Dynamic memory growth** — Allow memory capacity to expand based on environmental complexity, simulating neuroplasticity.
- **Richer sensory modalities** — Auditory input, olfactory gradients, or proprioceptive limb awareness to increase the dimensionality of the agent's experience.
- **Neural network encoder option** — A pluggable CNN-based encoder for richer visual representations, with the tradeoff of reduced inspectability.
- **Hierarchical prediction** — Multiple levels of temporal abstraction, predicting at different timescales (next tick vs. next 100 ticks).
- **Distributed simulation** — Run thousands of agents across multiple machines to study population-level emergent phenomena.
- **Persistent evolution** — Save and reload agent lineages across simulation runs, building long-term evolutionary histories.

---

## License

MIT
