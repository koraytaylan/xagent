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
| **`xagent-sandbox`** | 3D world simulation + application. Procedural terrain with biomes, food/hazard systems, physics, multi-agent support with evolution, wgpu-based rendering, egui IDE-like UI (docked sidebar, agent detail tabs, console), CSV logging, and the main event loop. |

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

Each tick, the brain executes a single loop:

```
encode → recall → predict → compare → learn → act
```

### Step-by-Step

1. **Encode** — The `SensoryEncoder` compresses the raw `SensoryFrame` into a fixed-size numerical vector (`EncodedState`). This is the semantic firewall: the frame's named fields (vision, energy, touch) are flattened into an opaque `Vec<f32>`, and from this point on the brain operates without any knowledge of what the numbers originally represented.

2. **Recall** — `PatternMemory` retrieves the most similar past experiences within a capacity budget set by the `CapacityManager`. The budget adapts: high prediction error → more recall slots to gather context.

3. **Predict** — The `Predictor` generates an expected next state by applying a learned linear transform to the current encoding, blended with recalled context patterns weighted by similarity.

4. **Compare** — The prediction from the *previous* tick is compared against the *current* encoding. The RMSE between them is the **prediction error** — the universal learning signal.

5. **Learn** — Prediction error drives all adaptation:
   - **Predictor weights** update via online gradient descent
   - **Memory reinforcement** strengthens patterns that co-occur with low error
   - **Encoder weights** receive L2 regularization to stay bounded
   - **Learning rate** is modulated by homeostatic gradient magnitude

6. **Act** — The `ActionSelector` chooses a motor command via a learned linear policy — a weight matrix mapping encoded state to action preferences. Credit assignment updates weights using the homeostatic gradient, modulated by state similarity (cosine) so learning is context-specific. Exploration uses uniform random action selection (10–85% of ticks, adaptive).

### Homeostatic Feedback

The `HomeostaticMonitor` tracks energy and integrity signals across three timescales (fast ≈ 5 ticks, medium ≈ 50 ticks, slow ≈ 500 ticks) using exponential moving averages. It produces:

- **Gradient** — positive = improving, negative = worsening. Modulates learning rate and action credit.
- **Urgency** — non-linear distress curve. Near-critical levels suppress exploration.

This is the **only** evaluative signal. There is no reward function.

### Capacity Constraints

| Constraint | Effect |
|---|---|
| `memory_capacity` | Finite pattern storage → forced forgetting → what survives = what matters |
| `processing_slots` | Limited recall per tick → forced prioritization → attention-like behavior |
| `representation_dim` | Fixed encoding size → forced compression → abstraction |

See the [brain crate README](crates/xagent-brain/README.md) for a deep dive into each component.

---

## 4. The Sandbox World

The sandbox is a real-time 3D environment rendered with **wgpu** (WebGPU/Vulkan/Metal/DX12 backend), wrapped in an **IDE-like UI** built with **egui 0.31 + egui_dock 0.16**. The UI provides a docked tab layout: a left sidebar with the agent list, a main area with the 3D viewport and agent detail tabs, and a bottom console for event logs.

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
- Sensory apparatus: 8×6 raycast vision (ray step 1.0), touch contacts (food, terrain edges, hazards, other agents), proprioception, interoception
- **Agent vision**: ray marching detects terrain, food, and other agents (rendered as magenta `[0.9, 0.2, 0.6, 1.0]` in the visual field)
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

### Multi-Agent & Food Scarcity

- Up to 100 concurrent agents (raised from 20)
- **Food respawn**: 10-second timer, relocates to a new random food-rich position — prevents camping, forces foraging
- **Reproduction**: currently disabled to focus on individual learning rather than evolution

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
| `Esc` | Print session summary and exit |

### egui UI Interaction

| Action | Effect |
|--------|--------|
| Click agent in sidebar | Select / focus that agent |
| Double-click agent in sidebar | Open an agent detail tab in the dock area |
| Drag / scroll on viewport pane | Camera rotation / zoom (only when hovering the viewport) |
| Close detail tab | Click the × on the tab header |

Camera controls (drag, scroll) are routed to the 3D viewport only when the pointer is hovering over it; interacting with the sidebar or detail tabs does not move the camera.

---

## 7. Configuration

### Brain Presets

| Preset | `memory_capacity` | `processing_slots` | `visual_encoding_size` | `representation_dim` | `learning_rate` | `decay_rate` |
|--------|---:|---:|---:|---:|---:|---:|
| **tiny** | 24 | 8 | 32 | 16 | 0.08 | 0.002 |
| **default** | 128 | 16 | 64 | 32 | 0.05 | 0.001 |
| **large** | 512 | 32 | 128 | 64 | 0.03 | 0.0005 |

**Parameter effects:**

| Parameter | What it controls |
|-----------|-----------------|
| `memory_capacity` | Max stored patterns. Smaller → faster forgetting, stronger capacity pressure. |
| `processing_slots` | Max recall operations per tick. Smaller → narrower attention. |
| `visual_encoding_size` | Resolution of visual encoding (average-pooled bins). Smaller → coarser visual perception. |
| `representation_dim` | Internal representation vector length. Smaller → more compression, more abstraction. |
| `learning_rate` | Base rate for weight updates (encoder, predictor, memory). Higher → faster adaptation but less stability. |
| `decay_rate` | Rate of memory decay per tick. Higher → more aggressive forgetting, favoring recent experience. |

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
    "representation_dim": 32,
    "learning_rate": 0.05,
    "decay_rate": 0.001
  },
  "world": {
    "world_size": 256.0,
    "energy_depletion_rate": 0.01,
    "movement_energy_cost": 0.005,
    "hazard_damage_rate": 0.1,
    "integrity_regen_rate": 0.005,
    "food_energy_value": 20.0,
    "food_density": 0.002,
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

These metrics are visible in the **sidebar** (compact vitals and sparkline charts) and in **agent detail tabs** (full vitals grid, brain info, and scrollable history charts). The CSV log files also record all metrics per tick.

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| `prediction_error` (avg) | Decreasing over time | Stuck high or oscillating |
| `exploitation_ratio` | Increasing over hundreds of ticks | Stuck near 0 |
| `memory_utilization` | Climbing toward capacity | Staying near 0 |
| `homeostatic_gradient` | Trending positive | Consistently negative |
| `exploration_rate` | Gradually decreasing | Stuck at max |

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
│   ├── xagent-brain/           # Cognitive architecture
│   │   ├── README.md           # Deep dive into brain internals
│   │   └── src/
│   │       ├── lib.rs          # Re-exports
│   │       ├── brain.rs        # Brain orchestrator + BrainTelemetry
│   │       ├── encoder.rs      # SensoryEncoder (feature extraction + projection)
│   │       ├── memory.rs       # PatternMemory (store, recall, associate, decay)
│   │       ├── predictor.rs    # Predictor (state prediction + gradient descent)
│   │       ├── action.rs       # ActionSelector (linear policy, credit assignment)
│   │       ├── homeostasis.rs  # HomeostaticMonitor (multi-timescale gradients)
│   │       └── capacity.rs     # CapacityManager (adaptive recall budgets)
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
│       │   ├── ui.rs            # egui integration (EguiIntegration, TabViewer, AgentSnapshot)
│       │   └── recording.rs    # CSV metrics logger
│       └── tests/
│           └── integration.rs  # 14 integration tests
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

## 11. Testing

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

**58 tests total** across the workspace:

| Crate | Tests | Scope |
|-------|------:|-------|
| **xagent-brain** | 44 | Encoder similarity/determinism, memory store/recall/decay/associations/temporal sequences, predictor convergence/gradient descent, homeostasis multi-timescale gradients/urgency/distress curves, action selector exploration/exploitation/credit assignment, capacity manager adaptive budgets, brain integration (100-tick stability, extreme inputs, prediction error convergence) |
| **xagent-sandbox** | 14 | Physics (movement, rotation, gravity, NaN sanitization), agent lifecycle (energy depletion, death, food consumption), terrain (determinism, interpolation smoothness, input validation), sensory extraction (vision dimensions, interoception accuracy) |

---

## 12. Future Directions

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
