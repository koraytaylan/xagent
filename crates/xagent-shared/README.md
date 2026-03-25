# xagent-shared

Shared types, traits, and configuration for the xagent cognitive agent platform.

This crate is the **interface contract** between the brain and sandbox crates. It contains no logic — only data structures, the `CognitiveArchitecture` trait, and configuration with presets. Every type that crosses the brain↔sandbox boundary lives here, which keeps the dependency graph clean and makes it possible to swap cognitive architectures without touching the simulation.

```
                ┌───────────────────────┐
                │    xagent-shared      │
                │                       │
                │  SensoryFrame         │
                │  MotorCommand         │
                │  BodyState            │
                │  BrainConfig          │
                │  WorldConfig          │
                │  CognitiveArchitecture│
                └──────────┬────────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
              ▼                         ▼
     ┌────────────────┐       ┌──────────────────┐
     │  xagent-brain  │       │ xagent-sandbox   │
     │                │       │                  │
     │  implements    │       │  uses types to   │
     │  Cognitive-    │       │  build frames,   │
     │  Architecture  │       │  interpret motor │
     └────────────────┘       │  commands, track │
                              │  body state      │
                              └──────────────────┘
```

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Module Reference](#2-module-reference)
   - [2.1 sensory.rs — Sensory Input](#21-sensoryrs--sensory-input)
   - [2.2 motor.rs — Motor Output](#22-motorrs--motor-output)
   - [2.3 body.rs — Physical Body State](#23-bodyrs--physical-body-state)
   - [2.4 config.rs — Configuration & Presets](#24-configrs--configuration--presets)
   - [2.5 traits.rs — CognitiveArchitecture Trait](#25-traitsrs--cogitivearchitecture-trait)
3. [Type Relationships & Data Flow](#3-type-relationships--data-flow)
4. [Configuration Guide](#4-configuration-guide)
5. [Adding New Types](#5-adding-new-types)

---

## 1. Design Philosophy

### Why a shared crate?

In many game/simulation architectures the "brain" and the "world" are tightly coupled — the agent directly reads world state and calls world methods. xagent deliberately splits these concerns:

| Principle | How xagent-shared achieves it |
|---|---|
| **Decoupling** | Brain and sandbox never import each other. They only import shared types. |
| **Swappable brains** | Any struct implementing `CognitiveArchitecture` can serve as the agent's brain. You can drop in a neural network, a behavior tree, or a random-action generator without changing the sandbox. |
| **Clean dependency graph** | `xagent-shared` has zero workspace dependencies. Both `xagent-brain` and `xagent-sandbox` depend on it, but never on each other. This prevents circular dependencies and keeps compile times predictable. |
| **Serialization boundary** | Every type derives `Serialize`/`Deserialize`. Configuration can be loaded from JSON files, sensory frames can be logged to disk, and motor commands can be replayed — all because the interface types are data-only. |
| **No hidden logic** | Shared contains no algorithms, no update loops, no mutable global state. It is a vocabulary crate: it defines the words that brain and sandbox use to talk to each other. |

### What belongs here vs. elsewhere

Put a type in `xagent-shared` if **both** brain and sandbox need it. If only the brain needs it (e.g., internal memory patterns), it belongs in `xagent-brain`. If only the sandbox needs it (e.g., terrain mesh data), it belongs in `xagent-sandbox`.

---

## 2. Module Reference

### 2.1 `sensory.rs` — Sensory Input

This module defines everything the agent can perceive. The sandbox constructs a `SensoryFrame` each tick and hands it to the brain. The brain has no built-in understanding of what these values mean — it must discover their significance through experience and prediction error.

> **Body–Brain Interface**: The structured types in this module (`VisualField`, `TouchContact`, `SensoryFrame`) are the **body's** wiring — engineering scaffolding that collects data from the simulated world. The named fields (`vision`, `energy_signal`, `touch_contacts`) exist for engineering clarity and for the sandbox to populate correctly. The brain's `SensoryEncoder` converts all of this into a single opaque `Vec<f32>` (`EncodedState`). From the brain's perspective, there are no modalities, no field names, no semantic labels — only a flat numerical vector whose meaning must be discovered through prediction error and homeostatic correlation.

#### `VisualField`

```rust
pub struct VisualField {
    pub width: u32,
    pub height: u32,
    pub color: Vec<f32>,   // RGBA, row-major, length = width * height * 4
    pub depth: Vec<f32>,   // per-pixel depth, length = width * height
}
```

The agent's vision is a **low-resolution grid** of color + depth samples rendered from its viewpoint. This is not a symbolic description of what the agent sees ("there is food ahead") — it is raw pixel data that the brain must learn to interpret. The resolution is configurable via `AgentDescriptor::visual_resolution` (default 8×6).

- **`color`**: Flattened RGBA float values in row-major order. Each component is in `[0.0, 1.0]`. The color encodes surface appearance — terrain type, food items, hazard zones, and other agents all have distinct colors, but the brain has no legend. It must learn that "green means food" through experience.
- **`depth`**: One float per pixel, range `[0.0, 1.0]`. Near plane = `0.0`, far plane = `1.0`. Depth provides distance information without requiring stereo vision. Objects closer to the agent have lower depth values.

`VisualField::new(width, height)` creates a blank field: colors initialize to black (`0.0`), depths to far plane (`1.0`).

#### `TouchContact`

```rust
pub struct TouchContact {
    pub direction: Vec3,    // direction of contact relative to agent body
    pub intensity: f32,     // 0.0 = none, 1.0 = hard impact
    pub surface_tag: u32,   // terrain, food, hazard, agent, etc.
}
```

Touch contacts are generated when the agent collides with or is adjacent to objects. Each contact carries:

- **`direction`**: A `Vec3` pointing from the agent's center toward the contact point. This tells the brain *where* on its body the contact occurred (front, left, behind, etc.) without requiring a labeled body map.
- **`intensity`**: How hard the contact is. Walking into a wall produces moderate intensity; being hit by a hazard produces high intensity. The brain receives this as a raw scalar and must learn to correlate intensity with danger or opportunity.
- **`surface_tag`**: An integer identifying the surface material. The sandbox assigns tags (0 = terrain, 1 = food, 2 = hazard, 3 = agent, etc.), but the brain receives them as opaque integers. It must learn that "tag 1 contacts are near food" through trial and error.

#### `SensoryFrame`

```rust
pub struct SensoryFrame {
    pub vision: VisualField,

    // Proprioception
    pub velocity: Vec3,
    pub facing: Vec3,
    pub angular_velocity: f32,

    // Interoception
    pub energy_signal: f32,
    pub integrity_signal: f32,
    pub energy_delta: f32,
    pub integrity_delta: f32,

    // Touch
    pub touch_contacts: Vec<TouchContact>,

    pub tick: u64,
}
```

The complete sensory input for one simulation tick. Organized into four sensory modalities:

**Proprioception (body awareness)**:
- `velocity` — The agent's velocity in world space. Lets the brain know if it's moving and in which direction, but only if it learns to attend to this signal.
- `facing` — A unit vector pointing in the agent's forward direction. Combined with velocity, the brain can learn to distinguish moving forward from being pushed sideways.
- `angular_velocity` — How fast the agent is turning (radians/sec). Nonzero when the agent is actively turning or being spun by a collision.

**Interoception (internal physiological signals)**:
- `energy_signal` — Energy level normalized to `[0.0, 1.0]`. This is the brain's only indication that it needs to find food. There is no "hunger" label — just a number that decreases over time.
- `integrity_signal` — Physical integrity normalized to `[0.0, 1.0]`. Drops when the agent takes damage from hazards.
- `energy_delta` — Rate of energy change. Positive when the agent consumes food, negative otherwise. This derivative signal lets the brain detect *changes* in energy without needing to remember the previous value.
- `integrity_delta` — Rate of integrity change. Spikes negative when taking damage, slightly positive when regenerating.

**Touch**: A vector of `TouchContact` structs. Empty when the agent is in open space, populated when colliding with terrain, food, hazards, or other agents.

**`tick`**: The current simulation tick number. Monotonically increasing. Can be used for temporal reasoning if the brain learns to track it.

### 2.2 `motor.rs` — Motor Output

This module defines how the brain expresses its intentions to the world.

#### `MotorCommand`

```rust
pub struct MotorCommand {
    pub forward: f32,       // [-1.0, 1.0], positive = forward
    pub strafe: f32,        // [-1.0, 1.0], positive = right
    pub turn: f32,          // [-1.0, 1.0], negative = left, positive = right
    pub action: Option<MotorAction>,
}
```

The motor command combines **continuous locomotion** with an **optional discrete action**:

- **`forward`**: Forward/backward thrust. `1.0` = full speed forward, `-1.0` = full reverse, `0.0` = no thrust. The physics engine scales this by the agent's max speed and applies acceleration.
- **`strafe`**: Lateral movement. `1.0` = strafe right, `-1.0` = strafe left. Combined with forward and turn, this gives the agent full 2D ground movement plus rotation.
- **`turn`**: Turn rate. `-1.0` = turn left at max rate, `1.0` = turn right at max rate. The physics engine converts this to angular velocity.
- **`action`**: An optional discrete action. Unlike the continuous values, this is fire-and-forget: either the agent performs an action this tick or it doesn't. The action system exists because some interactions (eating, pushing) are inherently discrete — you either eat or you don't.

**Range convention**: All continuous values are clamped to `[-1.0, 1.0]` by the physics engine. Values outside this range are silently clamped. The brain is free to output any float, but only the clamped value takes effect.

`MotorCommand::idle()` returns a command with all zeros and no action — the agent stands still and does nothing. This is the default, and it is what the brain should fall back to when it has no better plan.

#### `MotorAction`

```rust
pub enum MotorAction {
    Consume,    // eat whatever is in front
    Push,       // push/interact with object in front
    Jump,       // jump
}
```

Discrete actions the agent can perform:

- **`Consume`**: Attempt to eat whatever is directly in front of the agent. If food is present, it's consumed and the agent gains energy (`food_energy_value`). If nothing edible is in front, this is a no-op — energy is still spent on the attempt.
- **`Push`**: Push or interact with an object in front. Currently pushes movable objects and other agents.
- **`Jump`**: Jump upward. Useful for traversing terrain height changes.

### 2.3 `body.rs` — Physical Body State

This module defines the physical state of an agent body in the simulation. The body is the interface between the cognitive architecture and the physical world.

#### `InternalState`

```rust
pub struct InternalState {
    pub energy: f32,
    pub max_energy: f32,
    pub integrity: f32,
    pub max_integrity: f32,
}
```

The agent's physiological variables. These are the only evaluative signals in the system — there is no separate reward function, no utility score, no fitness measure. The agent lives or dies based on these two numbers:

- **Energy** (`[0.0, max_energy]`): Depletes continuously from base metabolism (`energy_depletion_rate` per tick) and from movement (`movement_energy_cost` per unit moved). Replenished by consuming food. When energy hits zero, the agent dies. Energy is the pressure that forces the agent to explore and find food — without it, the agent could stand still forever.
- **Integrity** (`[0.0, max_integrity]`): Damaged by hazard zones (`hazard_damage_rate` per tick in hazard areas). Slowly regenerates when energy is above 50% (`integrity_regen_rate`). When integrity hits zero, the agent dies. Integrity is the pressure that teaches the agent to avoid hazards — damage is the only negative experience in the system.

**Normalization methods**:
- `energy_signal() → f32`: Returns `energy / max_energy`, normalized to `[0.0, 1.0]`. This is what gets placed into `SensoryFrame::energy_signal`.
- `integrity_signal() → f32`: Returns `integrity / max_integrity`, normalized to `[0.0, 1.0]`.

**Death condition**: `is_dead()` returns `true` when either energy or integrity reaches zero. The sandbox checks this each tick and handles death/respawn.

`InternalState::new(max_energy, max_integrity)` creates a new state at full health — both energy and integrity start at their maximum values.

#### `BodyState`

```rust
pub struct BodyState {
    pub position: Vec3,     // world-space position
    pub facing: Vec3,       // forward direction (unit vector)
    pub velocity: Vec3,     // current velocity
    pub internal: InternalState,
    pub alive: bool,
}
```

The complete physical state of an agent. Maintained by the sandbox physics engine each tick:

- **`position`**: World-space coordinates. Updated by the physics engine based on velocity and motor commands.
- **`facing`**: A unit vector pointing in the agent's forward direction. Updated by the turn component of `MotorCommand`. Determines what the agent sees (vision is rendered from this direction) and what "in front" means for `Consume`/`Push` actions.
- **`velocity`**: Current velocity vector. Updated by the physics engine from motor thrust, gravity, and collisions. Fed into `SensoryFrame::velocity` for proprioception.
- **`internal`**: The `InternalState` physiological variables.
- **`alive`**: Whether the agent is currently alive. Set to `false` when `internal.is_dead()` returns `true`.

`BodyState::new(position, internal)` creates a new body at the given position, facing +Z (forward in the default coordinate system), at rest (zero velocity), alive.

### 2.4 `config.rs` — Configuration & Presets

This module defines all tunable parameters that affect the cognitive architecture's capacity constraints or the world's difficulty.

#### `BrainConfig`

```rust
pub struct BrainConfig {
    pub memory_capacity: usize,
    pub processing_slots: usize,
    pub visual_encoding_size: usize,
    pub representation_dim: usize,
    pub learning_rate: f32,
    pub decay_rate: f32,
}
```

Brain capacity parameters. These are not implementation details — they are the **generative constraints** that produce emergent cognition. Every parameter here creates a bottleneck, and bottlenecks create behavior:

| Parameter | Default | Effect on Emergence |
|---|---|---|
| `memory_capacity` | 128 | Maximum number of patterns the memory can hold. Smaller values force the brain to forget more aggressively, leading to stronger habit formation and more stereotyped behavior. Larger values allow richer memory but slower convergence. The increased default (128) allows learning multiple survival skills while still maintaining capacity pressure — only the most reinforced patterns survive decay, which *is* emergent attention. |
| `processing_slots` | 16 | Maximum number of patterns that can be recalled/compared per tick. This is the agent's "attention span" — with fewer slots, it makes faster but cruder decisions. More slots means better pattern matching but higher computational cost per tick. |
| `visual_encoding_size` | 64 | Resolution of the visual encoder output (downsampled from raw vision). Smaller values force more aggressive compression, which means the brain sees less detail but processes faster. Larger values preserve more visual information. |
| `representation_dim` | 32 | Length of the internal representation vector. This is the dimensionality of the space in which the brain thinks. Smaller values force more abstraction — the brain must compress its experience into fewer numbers, leading to coarser but more generalizable representations. |
| `learning_rate` | 0.05 | Base learning rate for association updates. Higher rates mean faster adaptation but more instability (catastrophic forgetting). Lower rates mean more stable memory but slower learning. |
| `decay_rate` | 0.001 | Decay rate for unreinforced patterns per tick. Patterns that aren't recalled or reinforced gradually lose strength. Higher decay means more aggressive forgetting — the brain only retains frequently-used patterns. |

**Presets**:

- **`BrainConfig::default()`** — Balanced defaults. Good starting point for most experiments. 128 memory slots, 16 processing slots, moderate learning and decay. Enough capacity to learn multiple survival skills while still maintaining prioritization pressure.
- **`BrainConfig::tiny()`** — Minimal capacity (24 memory, 8 processing, 32 visual, 16 representation). Use this to observe how severe constraints shape behavior: the agent forgets quickly, attends narrowly, and develops strong habits. Interesting for studying capacity-driven cognition.
- **`BrainConfig::large()`** — More capacity (512 memory, 32 processing, 128 visual, 64 representation). Slower emergence but richer eventual behavior. The agent can maintain more patterns, recall more context, and form finer-grained representations. Use this when you want to see what the architecture can do without tight constraints.

#### `WorldConfig`

```rust
pub struct WorldConfig {
    pub world_size: f32,
    pub energy_depletion_rate: f32,
    pub movement_energy_cost: f32,
    pub hazard_damage_rate: f32,
    pub integrity_regen_rate: f32,
    pub food_energy_value: f32,
    pub food_density: f32,
    pub tick_rate: f32,
    pub seed: u64,
}
```

World simulation parameters:

| Parameter | Default | Description |
|---|---|---|
| `world_size` | 256.0 | Side length of the square terrain in world units. Larger worlds require more exploration. |
| `energy_depletion_rate` | 0.01 | Energy drained per tick from base metabolism. This is the constant pressure that forces the agent to eat. |
| `movement_energy_cost` | 0.005 | Additional energy cost per unit of movement. Moving is expensive — the agent must learn to balance exploration (costly) with exploitation (staying near known food). |
| `hazard_damage_rate` | 0.1 | Integrity damage per tick while in a hazard zone. Higher values make hazards more lethal, increasing the selection pressure to learn avoidance. |
| `integrity_regen_rate` | 0.005 | Integrity recovered per tick when energy is above 50%. Recovery is slow relative to damage — the agent must avoid hazards rather than tanking through them. |
| `food_energy_value` | 20.0 | Energy restored per food item consumed. Higher values make each food item more impactful, reducing the frequency of foraging needed. |
| `food_density` | 0.002 | Density of food items in food-rich biomes (items per unit²). Higher density means food is easier to find, lower density increases starvation pressure. |
| `tick_rate` | 30.0 | Simulation ticks per second. Affects the real-time speed of the simulation. |
| `seed` | 42 | Random seed for world generation. Same seed = same terrain, biome layout, and initial food placement. |

**Presets**:

- **`WorldConfig::default()`** — Normal difficulty. Balanced energy economy where survival is possible but requires active foraging and hazard avoidance.
- **`WorldConfig::easy()`** — Abundant food (2.5× density, 1.5× value), slow energy drain (0.5× rate), mild hazards (0.5× damage). Use this for initial experiments or when you want to observe cognitive development without strong survival pressure.
- **`WorldConfig::hard()`** — Scarce food (0.5× density, 0.75× value), fast energy drain (2× rate), deadly hazards (2× damage). Use this to test whether the brain can develop survival strategies under extreme pressure. Most agents die quickly; those that survive exhibit stronger foraging and avoidance behaviors.

#### `AgentDescriptor`

```rust
pub struct AgentDescriptor {
    pub name: String,
    pub brain: BrainConfig,
    pub max_energy: f32,
    pub max_integrity: f32,
    pub visual_resolution: (u32, u32),
    pub fov_degrees: f32,
}
```

Describes an agent to be spawned into the world. Combines brain configuration with physical parameters:

- `name` — Human-readable identifier (default: "Agent-0")
- `brain` — The `BrainConfig` for this agent's cognitive architecture
- `max_energy` / `max_integrity` — Maximum physiological values (default: 100.0 each)
- `visual_resolution` — Width × height of the visual field (default: 8×6)
- `fov_degrees` — Field of view in degrees (default: 90°)

#### `FullConfig`

```rust
pub struct FullConfig {
    pub brain: BrainConfig,
    pub world: WorldConfig,
}
```

Combined configuration for JSON serialization. Both fields have `#[serde(default)]` so you can provide partial JSON and get defaults for the rest.

### 2.5 `traits.rs` — CognitiveArchitecture Trait

```rust
pub trait CognitiveArchitecture {
    fn tick(&mut self, frame: &SensoryFrame) -> MotorCommand;
}
```

The single trait that makes the entire architecture swappable. Any struct that implements `CognitiveArchitecture` can serve as an agent's brain.

**Why it exists**: The sandbox doesn't know or care what happens inside the brain. It only knows that each tick it provides a `SensoryFrame` and gets back a `MotorCommand`. This contract is what enables:

- **Multiple brain implementations**: The default `Brain` in `xagent-brain` uses predictive processing, but you could implement a rule-based system, a neural network, a random agent, or a human-controlled agent — all using the same interface.
- **A/B testing**: Run different cognitive architectures in the same world and compare their survival outcomes.
- **Incremental development**: Build a simple reactive brain first, then gradually add prediction, memory, and learning without changing the sandbox.

**How to implement a custom brain**:

```rust
use xagent_shared::{CognitiveArchitecture, SensoryFrame, MotorCommand};

struct MyBrain {
    // your internal state
}

impl CognitiveArchitecture for MyBrain {
    fn tick(&mut self, frame: &SensoryFrame) -> MotorCommand {
        // Process frame.vision, frame.energy_signal, etc.
        // Return a MotorCommand with forward/strafe/turn/action
        MotorCommand::idle()
    }
}
```

The `tick` method receives an immutable reference to the frame (the brain can read but not modify the world) and returns a `MotorCommand` by value. The brain can maintain any internal state it needs via `&mut self`.

---

## 3. Type Relationships & Data Flow

The types in this crate form a clean data-flow pipeline:

```
  Sandbox                     Shared Types                     Brain
  ───────                     ────────────                     ─────

  Physics engine     ──►   BodyState          ──►   (internal tracking)
  Sensory extraction ──►   SensoryFrame       ──►   brain.tick(frame)
                           MotorCommand       ◄──   return value
  Physics engine     ◄──   MotorCommand
```

**Per-tick flow**:

1. **Sandbox builds `SensoryFrame`**: The sandbox reads the agent's `BodyState`, renders low-res vision from the agent's viewpoint, gathers touch contacts, normalizes physiological signals, and packages everything into a `SensoryFrame`.

2. **Brain processes `SensoryFrame`**: The brain's `tick()` method receives the frame. It encodes the sensory data, recalls relevant patterns from memory, predicts what will happen next, measures prediction error, updates its internal model, and selects an action.

3. **Brain returns `MotorCommand`**: The brain's output is a `MotorCommand` specifying how the agent wants to move and what (if any) discrete action to perform.

4. **Sandbox executes `MotorCommand`**: The physics engine applies the motor command to the agent's `BodyState` — updating position, velocity, and facing based on the thrust/strafe/turn values. If a discrete action is specified, it's executed (e.g., consuming food in front).

5. **Sandbox updates `BodyState`**: Energy is depleted, integrity is damaged or regenerated, collisions are resolved, and the alive flag is checked. The updated `BodyState` feeds into the next tick's `SensoryFrame`.

**What flows where**:

| Type | Direction | Description |
|---|---|---|
| `SensoryFrame` | sandbox → brain | Complete sensory snapshot for one tick |
| `MotorCommand` | brain → sandbox | The brain's intended actions |
| `BodyState` | sandbox-internal | Physical state; sandbox reads/writes, brain never sees directly |
| `InternalState` | sandbox-internal | Physiological variables; normalized copies appear in `SensoryFrame` |
| `BrainConfig` | config → brain | Capacity constraints, set at startup |
| `WorldConfig` | config → sandbox | World parameters, set at startup |

Note that the brain **never** receives `BodyState` directly — it only gets the normalized, noisy signals in `SensoryFrame`. The brain doesn't know its exact energy level, only its `energy_signal` (a float between 0 and 1). This information asymmetry is deliberate: it forces the brain to build internal models of its own state rather than having perfect self-knowledge.

---

## 4. Configuration Guide

### Loading from JSON

`FullConfig` supports JSON serialization with defaults:

```json
{
  "brain": {
    "memory_capacity": 300,
    "processing_slots": 24,
    "learning_rate": 0.04
  },
  "world": {
    "world_size": 512.0,
    "food_density": 0.003
  }
}
```

Any fields not specified use `Default::default()`. You can provide a minimal JSON with just the parameters you want to change.

### Using presets

Presets can be used directly in code:

```rust
use xagent_shared::{BrainConfig, WorldConfig};

// Tiny brain in a hard world — maximum survival pressure
let brain = BrainConfig::tiny();
let world = WorldConfig::hard();

// Large brain in an easy world — focus on cognitive development
let brain = BrainConfig::large();
let world = WorldConfig::easy();
```

### Parameter tuning tips

**Brain parameters**:
- Start with defaults. Only reduce capacity if you specifically want to study constraint-driven behavior.
- `memory_capacity` below 20 produces agents that forget almost everything — interesting but hard to train.
- `processing_slots` below 8 creates agents that are essentially reactive (no time to recall context).
- `learning_rate` above 0.1 causes catastrophic forgetting; below 0.01 means very slow adaptation.
- `decay_rate` above 0.005 means patterns vanish within a few hundred ticks if not reinforced.

**World parameters**:
- `energy_depletion_rate` and `food_density` together determine survival difficulty. Increase one or decrease the other to make survival harder.
- `hazard_damage_rate` above 0.3 makes hazards nearly instantly lethal — useful for strong avoidance learning but agents die often.
- `food_energy_value` below 10 means agents need to eat very frequently, increasing foraging pressure.
- `world_size` above 512 creates large worlds where agents must travel far to find food — tests navigation and spatial memory.

---

## 5. Adding New Types

When extending the shared types, follow these guidelines:

### Backward compatibility

- **Adding fields**: Always provide a `#[serde(default)]` annotation or a `Default` implementation so that existing JSON configs and serialized data continue to work.
- **Removing fields**: Never remove a field in a minor version. Deprecate it first (mark with `#[deprecated]` and document the replacement), then remove in the next major version.
- **Renaming fields**: Use `#[serde(alias = "old_name")]` to accept both old and new names during a transition period.

### Serialization

- All public types must derive `Serialize` and `Deserialize` from serde. This is a hard requirement — the types cross process boundaries (brain↔sandbox) and must be serializable for logging, replay, and configuration.
- Use `f32` for continuous values, `u32` or `u64` for discrete identifiers, `Vec3` (from glam) for spatial data.
- Avoid complex nested generics in shared types — they make serialization and cross-crate usage painful.

### New sensory modalities

If you add a new sense (e.g., hearing):

1. Define the data structure in `sensory.rs` (e.g., `AudioSample`).
2. Add a field to `SensoryFrame` with `#[serde(default)]` so existing code that doesn't produce audio still works.
3. Update the sandbox's sensory extraction to populate the new field.
4. The brain can choose to ignore or process the new modality — no changes required in the brain unless you want it to use the new sense.

### New motor actions

To add a new discrete action:

1. Add a variant to `MotorAction` in `motor.rs`.
2. Handle it in the sandbox's physics/action processing.
3. The brain's action selector will need to learn about the new action — this may require changes in `xagent-brain`.

### New configuration parameters

1. Add the field to the appropriate config struct (`BrainConfig` or `WorldConfig`).
2. Add a default value in the `Default` implementation.
3. Update all presets (`tiny`, `large`, `easy`, `hard`) to set appropriate values.
4. Document the parameter in this README.

---

## Crate Metadata

| Key | Value |
|---|---|
| **Dependencies** | `glam` (vec math), `serde` (serialization) |
| **Dependents** | `xagent-brain`, `xagent-sandbox` |
| **Logic** | None — data structures and trait definitions only |
| **Unsafe** | None |
