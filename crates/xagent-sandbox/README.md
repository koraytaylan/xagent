# xagent-sandbox

The 3D simulation environment where cognitive agents live. If the brain crate is the
**mind**, the sandbox is the **body** — it provides terrain to walk on, food to eat,
hazards to avoid, eyes to see through, and a physics engine to obey.

## Responsibilities

| Concern | Owner |
|---|---|
| Procedural world generation (terrain, biomes, food) | `world/` |
| Rigid-body-lite physics (gravity, collision, locomotion) | `physics/` |
| Sensory extraction (vision, touch, proprioception, interoception) | `agent/senses.rs` |
| Agent lifecycle (spawn, death, respawn, reproduction) | `agent/mod.rs` + `main.rs` |
| Multi-agent management | `main.rs` |
| wgpu-based rendering (Vulkan / Metal) | `renderer/` |
| egui IDE-like UI (sidebar, docked tabs, console) | `ui.rs` |
| HUD overlay & bitmap font text | `renderer/hud.rs`, `renderer/font.rs` |
| CSV telemetry recording | `recording.rs` |
| Per-generation replay recording & playback | `replay.rs` |
| Event loop & orchestration | `main.rs` |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  main.rs  (winit ApplicationHandler – event loop & orchestration)         │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ renderer │  │  world   │  │ physics  │  │  agent   │  │ recording  │  │
│  │          │  │          │  │          │  │          │  │            │  │
│  │ mod.rs   │  │ mod.rs   │  │ mod.rs   │  │ mod.rs   │  │recording.rs│  │
│  │ camera.rs│  │terrain.rs│  │          │  │ senses.rs│  │ replay.rs  │  │
│  │ hud.rs   │  │ biome.rs │  │          │  │          │  │            │  │
│  │ font.rs  │  │ entity.rs│  │          │  │          │  │            │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │             │             │             │              │          │
│  ┌────┴─────┐                                                            │
│  │  ui.rs   │  egui integration (EguiIntegration, TabViewer, dock)       │
│  └──────────┘                                                            │
│       │             │             │             │              │          │
│       │     │ WorldState   │◄─────┤             │              │          │
│       │     │  .terrain    │      │ step()      │              │          │
│       │     │  .biome_map  │      │  reads &    │              │          │
│       │     │  .food_items │      │  mutates    │              │          │
│       │     │  .config     │      │             │              │          │
│       │     └──────────────┘      │             │              │          │
│       │                           │             │              │          │
│       │  ┌──────────────────────────────────────┤              │          │
│       │  │         Agent                        │              │          │
│       │  │  .body: AgentBody  (position,yaw,…)  │              │          │
│       │  │  .brain: Brain     (xagent-brain)    │              │          │
│       │  │  .color, .generation, .death_count   │              │          │
│       │  └──────────────────────────────────────┘              │          │
│       │                                                        │          │
│  ┌────┴──────────────────────────────────────────────┐         │          │
│  │  Per-frame pipeline                               │         │          │
│  │  1. Input events → camera update                  │         │          │
│  │  2. For each tick:                                │         │          │
│  │     a. extract_senses() → SensoryFrame            │         │          │
│  │     b. brain.tick(frame) → MotorCommand            │         │          │
│  │     c. physics::step(agent, motor, world)         │         │          │
│  │     d. Record tick data + food events to replay   │         │          │
│  │     e. death/respawn/reproduction checks          │         │          │
│  │  3. Advance replay playback (if active)           │         │          │
│  │  4. Rebuild meshes (agents, food)                 │         │          │
│  │  5. Build HUD bars                                │         │          │
│  │  6. render_with_hud(meshes, vp, bars, panels, …)  │─────────┘          │
│  │  7. Log telemetry to CSV                          │                    │
│  └───────────────────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────────────────┘

External crates:
  xagent-shared   BodyState, InternalState, SensoryFrame, MotorCommand,
                  WorldConfig, BrainConfig, FullConfig, CognitiveArchitecture
  xagent-brain    Brain (implements CognitiveArchitecture)
```

---

## Module Deep Dives

### 3.1 Renderer (`renderer/`)

#### wgpu Initialization (`renderer/mod.rs`)

`Renderer::new()` performs the full GPU setup:

1. Creates a `wgpu::Instance` with `Backends::all()` (Vulkan on Linux, Metal on macOS).
2. Requests a **HighPerformance** adapter and a default device.
3. Picks an sRGB surface format (falls back to the first available).
4. Configures the swap chain with `PresentMode::AutoVsync`.
5. Creates **three render pipelines**:
   - **3D scene pipeline** — depth-tested, back-face culled, uniform-buffered `view_proj` matrix.
   - **HUD pipeline** — no depth, no culling, alpha-blended at 0.85 opacity, no uniforms (pure NDC).
   - **Text pipeline** — texture-sampled bitmap font, alpha-blended, separate bind group.

#### Vertex Format

```rust
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],  // world-space (3D) or NDC (HUD)
    pub color:    [f32; 3],  // RGB
}
```

24 bytes per vertex. Shared by the 3D and HUD pipelines.

#### WGSL Shaders

**3D shader** — Vertex stage multiplies position by a `view_proj` uniform (`mat4x4<f32>`),
fragment stage outputs `vec4(color, 1.0)` (opaque).

**HUD shader** — Vertex stage passes NDC position through unchanged,
fragment stage outputs `vec4(color, 0.85)` (semi-transparent overlay).

**Text shader** — Vertex stage passes NDC position and UV through,
fragment stage samples an R8 font atlas texture and modulates vertex color alpha
by the glyph coverage value.

#### Camera (`renderer/camera.rs`)

Free-fly camera with 6-DOF movement:

| Property | Default |
|---|---|
| Position | `(0, 10, 20)` |
| FOV (vertical) | 45° |
| Z-near / Z-far | 0.1 / 500.0 |
| Movement speed | 15.0 units/s |
| Mouse sensitivity | 0.003 rad/pixel |

- `front()` = `(cos(yaw)·cos(pitch), sin(pitch), sin(yaw)·cos(pitch))`
- Pitch is clamped to ±(π/2 − 0.01) to prevent gimbal lock.
- Scroll wheel zooms along the view direction.
- View matrix: `Mat4::look_at_rh`.
- Projection: `Mat4::perspective_rh`.

#### HUD Overlay (`renderer/hud.rs`)

> **Note:** The legacy HUD pipeline still exists and renders status bars as a wgpu overlay. However, the primary UI is now the **egui** layer (see `ui.rs` below), which provides the sidebar, docked agent detail tabs, and bottom console. The HUD bars remain as a fallback rendering path.

`HudBar` represents a single status bar in NDC space (−1..1):

```rust
pub struct HudBar {
    pub x: f32,         // left edge
    pub y: f32,         // top edge
    pub width: f32,
    pub height: f32,
    pub fill: f32,      // 0.0–1.0
    pub color: [f32; 3],
    pub bg_color: [f32; 3],
}
```

Each bar is rendered as three quads (12 vertices, 18 indices):
1. **Border** — slightly expanded (3 px NDC), gray `[0.6, 0.6, 0.6]`
2. **Background** — dark fill
3. **Foreground** — colored, width scaled by `fill`

`build_panel_vertices()` produces a dark background panel (`[0.05, 0.05, 0.08]`).

Four bars are shown for the selected agent:

| Bar | Color | Signal |
|---|---|---|
| Energy | Green | `energy_signal()` |
| Integrity | Blue | `integrity_signal()` |
| Prediction error | Red | `prediction_error` (clamped 0–1) |
| Exploration rate | Yellow | `exploration_rate` (clamped 0–1) |

#### Bitmap Font (`renderer/font.rs`)

Fully procedural — no external font files.

- **Glyph size**: 8 × 12 pixels.
- **Character set**: `0-9 A-Z a-z % . : / - + # ( )` — 71 characters.
- **Atlas layout**: 16 glyphs per row → 128 × 60 pixel R8 texture.
- Each glyph is defined as 12 bytes of bitmask data (`GlyphBits`).
- `TextVertex` has `position [f32; 3]`, `uv [f32; 2]`, `color [f32; 4]` (36 bytes).
- Text is rendered by emitting one textured quad (4 verts, 6 indices) per character.
- The text pipeline uses `ALPHA_BLENDING` and `FilterMode::Nearest`.

### 3.1b egui UI Layer (`ui.rs`)

The IDE-like UI is built with **egui 0.31** and **egui_dock 0.16**, rendered as an overlay on top of the wgpu 3D scene.

#### Key Types

| Type | Role |
|---|---|
| `EguiIntegration` | Owns the egui context, winit integration state, wgpu renderer, and the offscreen texture that embeds the 3D viewport inside an egui panel. |
| `Tab` | Enum — `Sandbox` (3D viewport, always open), `Evolution` (evolution dashboard), or `AgentDetail(u32)` (per-agent detail view). |
| `SortMode` | Enum — 7 sorting options for the sidebar agent list: Id, Energy, Integrity, Deaths, LongestLife, PredictionError, Fitness. |
| `AgentSnapshot` | Per-frame copy of an agent's vitals, brain metrics, motor output, decision log, vision data, position, and history buffers — decouples the UI from the simulation's mutable state. |
| `WorldSnapshot` | Per-frame world state for the mini-map: world size, food positions, and a cached 256x256 biome texture. |
| `ReplayState` | Playback state for the replay system: active flag, current tick, playing/paused, speed multiplier, total ticks, selected agent index. |
| `TabContext` | Transient context passed to `TabViewer` each frame: viewport texture ID, pixel-per-point scale, desired viewport size, hover flag, chart zoom level, agent snapshots, world snapshot, replay state, and last generation recording. |

#### Layout

- **Top bar** — App title, FPS, agent count, evolution state indicator, wall time and ticks/sec (when running/paused), pause/resume and reset buttons (right-aligned, when running/paused).
- **Left sidebar** — Sort dropdown (by ID, Energy, Integrity, Deaths, Longest Life, Prediction Error, Fitness), agent list with colored dots, compact energy/integrity bars, phase label, death count, food consumed, and best life duration.
- **Main dock area** — Tabbed region. The **Sandbox** tab displays the 3D viewport; the **Evolution** tab shows the evolution dashboard; double-clicking an agent in the sidebar opens an **Agent Detail** tab.
- **Agent detail tabs** — Color dot + heading + phase label, two-column vitals/motor display (energy/integrity bars, continuous forward/turn motor gauges with weight norms), side-by-side **Agent Vision** (8x6 upscaled color grid) and **Mini-Map** (top-down biome texture with food dots, agent markers, and facing direction arrow), **History** chart (4-line: E/I/P/X with legend, scroll-to-zoom 30–10k ticks), **Decision Stream** (scrollable per-tick log with color-coded credit, motor output, gradient, urgency, and patterns recalled), and **Replay Controls** (timeline scrubber, play/pause, speed selector 0.5x–8x, live/replay toggle).
- **Bottom console** — Scrollable log of evolution events.

#### Evolution Tab

The Evolution tab is a two-column layout (rendered via `ui.columns(2, ...)`):

- **Left pane — Generation tree** (file explorer style): each node is a generation entry. Clicking a node selects it and populates the right panel with its details (status, island ID, fitness, mutations applied, full `BrainConfig`). Dead-end branches (failed/exhausted generations) are **collapsed by default** but can be manually expanded for post-mortem inspection.
- **Right panel — Generation detail**: shows the selected node's fields when a tree node is clicked; otherwise shows aggregate stats for the current run.
- **Fitness chart**: one colored line per island (up to 8 distinct colors), so each island's evolutionary trajectory is independently visible. Data comes from `fitness_history_by_island()`, which groups `EvolutionSnapshot.fitness_history` (`HashMap<i64, Vec<(u32, f32, f32)>>`) by island ID.
- **Generation progress bar**: rendered in `TopBottomPanel::top("top_bar")` — always visible regardless of which tab is active. Displays ticks elapsed vs. tick budget for the current generation.

#### Viewport Integration

The 3D scene is rendered to an **offscreen wgpu texture**, which is then displayed as an `egui::Image` inside the Sandbox tab. Camera drag and scroll events are forwarded to the 3D camera only when the pointer is hovering over the viewport pane (tracked via `viewport_hovered`). This prevents UI interactions in the sidebar or detail tabs from moving the camera.

---

### 3.2 World Simulation (`world/`)

#### Procedural Terrain (`world/terrain.rs`)

```rust
pub struct TerrainData {
    pub size: f32,          // side length (default 256.0)
    pub subdivisions: u32,  // grid resolution (default 128)
    pub heights: Vec<f32>,  // flat array [z * vps + x]
}
```

Generation uses **Perlin noise** with three octaves:

| Octave | Frequency | Amplitude |
|---|---|---|
| 1 | 0.02 | ±4.0 |
| 2 | 0.05 | ±2.0 |
| 3 | 0.10 | ±0.5 |

Total height range is approximately ±6.5 units. The seed is taken directly from
`WorldConfig.seed`.

**`height_at(x, z)`** — Bilinear interpolation over the grid. Used by physics
for ground collision and by food spawning for vertical placement.

**`build_mesh()`** — Generates an indexed triangle mesh (`129 × 129 = 16,641`
vertices at default resolution). Vertex colors are blended from biome color
and height-based tinting.

#### Biome System (`world/biome.rs`)

```rust
pub enum BiomeType {
    FoodRich,  // green, food spawns here
    Barren,    // tan/brown, no food, no hazard
    Danger,    // red, deals damage per tick
}
```

Driven by a separate Perlin noise field (seed = `config.seed + 95`, frequency = 0.03):

| Noise value | Biome |
|---|---|
| > 0.2 | `FoodRich` |
| < −0.2 | `Danger` |
| −0.2 to 0.2 | `Barren` |

Color mapping blends a height-normalized factor `t = (height + 5) / 10` into
per-biome RGB ranges:

| Biome | R | G | B |
|---|---|---|---|
| FoodRich | 0.12–0.22 | 0.40–0.45 | 0.08–0.12 |
| Barren | 0.40–0.60 | 0.32–0.37 | 0.15–0.20 |
| Danger | 0.45–0.65 | 0.15–0.20 | 0.10–0.13 |

#### Food Entities (`world/entity.rs`)

```rust
pub struct FoodItem {
    pub position: Vec3,
    pub consumed: bool,
    pub respawn_timer: f32,  // seconds remaining
}
```

- **Spawning**: Grid scan at 4-unit spacing. Only spawns in `FoodRich` biomes.
  Probability per cell = `food_density × 16` (area 4×4). Randomized within cell.
  Placed 0.35 units above terrain.
- **Consumption**: Sets `consumed = true`, `respawn_timer = 10.0` seconds.
- **Respawn**: Timer decrements each `world.update(dt)`. When ≤ 0, the item
  **relocates to a new random position** in a `FoodRich` biome rather than
  reappearing at its original spot. Up to 64 random positions are sampled to
  find a valid cell inside a food-rich biome; if none is found, the item falls
  back to its original position. This prevents agents from camping a single
  food spot indefinitely — the total food supply stays constant, but positions
  change on every respawn cycle.
- **Auto-consume**: Food is consumed automatically when an agent is within
  `FOOD_CONSUME_RADIUS` (2.5 units). No specific action is required — the brain's
  job is to move the body toward food, not to "decide" to eat. This mirrors
  biology: organisms absorb nutrients on contact.
- **Rendering**: Each visible food item is a 0.6-unit green cube `[0.1, 0.8, 0.2]`
  with 6-face shading (shade factors 1.0, 0.9, 0.85, 0.7, 0.8, 0.75).

---

### 3.3 Physics (`physics/mod.rs`)

#### Constants

| Constant | Value | Purpose |
|---|---|---|
| `GRAVITY` | 20.0 | Downward acceleration (units/s²) |
| `MOVE_SPEED` | 8.0 | Maximum horizontal speed (units/s) |
| `TURN_SPEED` | 3.0 | Maximum turn rate (rad/s) |
| `AGENT_HALF_HEIGHT` | 1.0 | Half-height for ground offset |
| `FOOD_CONSUME_RADIUS` | 2.5 | Must be within this distance to eat |
| `AGENT_MIN_SEPARATION` | 2.0 | Minimum distance between agent centers |

#### `step()` — One Simulation Tick

1. **Early exit** if agent is dead.
2. **Sanitize motor** — NaN/Infinity → 0.0, clamp to [−1, 1].
3. **Save recovery state** — snapshot position + velocity for NaN fallback.
4. **Snapshot internals** — record energy/integrity for delta computation.
5. **Turning** — `yaw += motor.turn × TURN_SPEED × dt`, update `facing` unit vector,
   compute `angular_velocity`.
6. **Locomotion** — `desired = facing × forward + right × strafe`, normalize if > 1.
   Set `velocity.xz = desired × MOVE_SPEED`.
7. **Gravity** — `velocity.y -= GRAVITY × dt`.
8. **Integrate position** — `position += velocity × dt`.
9. **Clamp to world bounds** — ±(`world_size / 2 − 1`).
10. **Ground collision** — if `position.y < ground + AGENT_HALF_HEIGHT`, snap up,
    zero vertical velocity.
11. **NaN recovery** — if any component is non-finite, revert to saved state.
12. **Jump** — if `MotorAction::Jump` and on ground (within 0.1 units), `velocity.y = 8.0`.
13. **Energy depletion** — `energy -= depletion_rate + |forward + strafe| × move_cost`.
14. **Biome effects** — in `Danger`: `integrity -= hazard_damage_rate`.
15. **Integrity regen** — if `energy_signal > 0.5` and not full, `integrity += regen_rate`.
16. **Auto-consume food** — find nearest non-consumed food within 2.5 units regardless
    of action. Consume it: `energy += food_energy_value` (capped at max). Returns
    `Some(food_index)` with the consumed item's index (used by the replay recording
    system to track food events), or `None` if no food was consumed.
17. **Death check** — if `energy ≤ 0` or `integrity ≤ 0`, set `alive = false`.

#### Agent-Agent Collision

Physical collision resolution pushes overlapping agents apart to maintain a 2-unit
minimum separation. This is an O(n²) pairwise check but remains fast with simple
distance math. Agents cannot overlap — they are treated as solid cubes.

---

### 3.4 Agent System (`agent/`)

#### AgentBody (`agent/mod.rs`)

```rust
pub struct AgentBody {
    pub body: BodyState,       // from xagent-shared (position, facing, velocity, internal, alive)
    pub yaw: f32,              // heading angle (radians)
    pub angular_velocity: f32, // turn rate this tick
    prev_energy: f32,          // for delta computation
    prev_integrity: f32,
}
```

Initialized with `energy = integrity = 100.0`, `facing = Vec3::Z`, `alive = true`.

#### Agent (`agent/mod.rs`)

```rust
pub struct Agent {
    pub id: u32,
    pub body: AgentBody,
    pub brain: Brain,
    pub color: [f32; 3],       // static palette color (matches sidebar)
    pub birth_tick: u64,
    pub death_count: u32,
    pub generation: u32,       // life iteration (incremented on each death/respawn)
    pub life_start_tick: u64,  // reset on respawn
    pub longest_life: u64,
    pub respawn_cooldown: u32, // frames to wait before respawning
    pub persist_brain: bool,   // if true, brain survives death
    pub has_reproduced: bool,  // once per life
    pub cached_motor: MotorCommand, // cached motor command for brain-decimated ticks
    pub decision_log: VecDeque<DecisionSnapshot>, // last 256 brain decisions (ring buffer)
    pub food_consumed: u32,    // cumulative food consumed
    pub total_ticks_alive: u64, // cumulative ticks alive across all lives
}
```

**Agent palette colors** — Each agent is assigned a static palette color at spawn. The same color is used in the 3D viewport (with an sRGB→linear conversion for correct GPU rendering) and in the egui sidebar. Dead agents render as dark gray `[0.25, 0.25, 0.25]`.

**Agent mesh** — 2.0-unit cube with 6-face shading (each face darkened by factors
1.0, 0.9, 0.8, 0.7, 0.85, 0.75). Combined into a single vertex buffer for all agents.

**`MAX_AGENTS`** = 100. Hard cap — spawn requests are rejected beyond this.

**`REPRODUCTION_THRESHOLD`** = 5000 ticks. Minimum age before an agent can reproduce.
Reproduction is currently disabled (commented out) to focus on individual learning
rather than evolution.

#### Sensory Extraction (`agent/senses.rs`)

Converts world state into a `SensoryFrame` — the **body's** data collection step. This is the agent's sensory apparatus: its "eyes" (raycasting), its "skin" (touch contacts), its "gut feeling" (interoception). The `SensoryFrame` with its named fields and structured types is engineering scaffolding that belongs to the body, not the brain. Once the brain's encoder flattens this into an opaque `Vec<f32>`, all field names, modality boundaries, and semantic labels are stripped away. The brain never processes "visual data" or "touch data" — it receives only a flat numerical vector and must discover what every index means through experience.

##### Vision

```
Resolution:  8 × 6 (48 pixels)
FOV:         90° horizontal
Max depth:   50.0 units
Ray step:    1.0 units
```

For each pixel in the grid, a ray is cast from the agent's position into the world
in the direction determined by the pixel's grid position within the FOV. The ray
marches in 1.0-unit steps until it hits terrain (point below the heightmap surface),
another agent, or reaches max distance.

**Hit color** is determined by what the ray strikes first (checked in this order):

| Target | RGBA | Priority |
|---|---|---|
| Food item | `[0.70, 0.95, 0.20, 1.0]` (lime green) | 1st — spatial grid lookup |
| Other agent | `[0.90, 0.20, 0.60, 1.0]` (magenta) | 2nd |
| FoodRich terrain | `[0.15, 0.50, 0.10, 1.0]` | 3rd |
| Barren terrain | `[0.50, 0.40, 0.20, 1.0]` | 3rd |
| Danger terrain | `[0.60, 0.20, 0.10, 1.0]` | 3rd |
| Sky (miss) | `[0.53, 0.81, 0.92, 1.0]` (early-out) | fallback |

A single unified `march_ray_unified` function handles all targets via an `AgentSlice` enum dispatch, ensuring both the `extract_senses` and `extract_senses_with_positions` paths see food identically.

Depth is normalized to `[0, 1]` by dividing by `max_dist` (50.0).

##### Proprioception

| Field | Source |
|---|---|
| `velocity` | `body.velocity` (world-space Vec3) |
| `facing` | `body.facing` (unit vector) |
| `angular_velocity` | `agent.angular_velocity` (scalar, rad/s) |

##### Interoception

| Field | Source |
|---|---|
| `energy_signal` | `energy / max_energy` (0.0–1.0) |
| `integrity_signal` | `integrity / max_integrity` (0.0–1.0) |
| `energy_delta` | Change since last snapshot (can be negative) |
| `integrity_delta` | Change since last snapshot |

##### Touch

Four types of touch contacts, identified by `surface_tag`:

| Tag | Value | Range | Intensity |
|---|---|---|---|
| `TOUCH_FOOD` | 1 | 3.0 units | `1 − dist/3.0` |
| `TOUCH_TERRAIN_EDGE` | 2 | 3.0 units from edge | `1 − dist_to_edge/3.0` |
| `TOUCH_HAZARD` | 3 | In `Danger` biome | 0.5 (constant) |
| `TOUCH_AGENT` | 4 | 5.0 units | `1 − dist/5.0` |

Each contact carries a `direction` (normalized Vec3 from agent to object) and an
`intensity` (0.0–1.0). Terrain edges are detected for all four sides independently.

##### Inter-Agent Perception

`extract_senses_with_others()` accepts an `&[OtherAgent]` slice. Living agents
within 5 units appear as `TOUCH_AGENT` contacts. Dead agents are ignored.

Additionally, **agents are visible in vision**: ray marching now detects other agent
cubes using a squared-distance check for performance. Other agents appear as magenta
`[0.9, 0.2, 0.6, 1.0]` in the visual field, giving the brain a visual signal for
nearby agents in addition to touch contacts.

---

### 3.5 Recording, Telemetry & Replay (`recording.rs`, `replay.rs`)

#### CSV Format

File name: `xagent_log_YYYY-MM-DD_HH-MM-SS.csv` (UTC, no chrono dependency).

Columns (29 total):

```
agent_id, tick, prediction_error, avg_prediction_error, memory_utilization,
memory_capacity, exploration_rate, homeostatic_gradient,
energy, max_energy, integrity, max_integrity,
position_x, position_y, position_z, facing_x, facing_z,
biome, action_forward, action_strafe, action_turn, action_discrete, alive,
exploitation_ratio, decision_quality, behavior_phase, death_count, life_ticks,
generation
```

#### Flush Strategy

- Writes are buffered via `BufWriter<File>`.
- **Flushed every 100 ticks** for crash safety.
- Final flush on session exit (`print_session_summary()`).

#### Per-Generation Replay Recording (`replay.rs`)

The replay system captures per-tick agent state during evolution runs, enabling
post-hoc playback of any completed generation.

**Data Structures:**

| Struct | Purpose |
|--------|---------|
| `TickRecord` | Per-tick per-agent snapshot: position, yaw, alive, energy, integrity, motor output, exploration rate, prediction error, gradient, urgency, credit magnitude, patterns recalled, phase, and optional vision keyframe |
| `FoodEvent` | Sparse food state change: tick, food index, consumed flag, new position (on respawn) |
| `GenerationRecording` | Complete generation recording: agent info, flat-array tick records (`tick * agent_count + agent_idx`), sparse food events, initial food positions |

**Storage Strategy:**
- Tick records are stored in a flat `Vec<TickRecord>` indexed as `tick * agent_count + agent_idx`
- Vision data (192 floats per agent) is only stored at keyframes (every 30 ticks) to save memory
- Food events are sparse — only recorded when food is consumed or respawns
- Initial food positions are stored once for reconstruction via `food_at_tick()`

**Integration Points:**
- Recording starts when a new generation spawns (`spawn_evolution_population` / `spawn_population_from_configs`)
- Per-tick data is captured after physics and collision resolution
- Food consumption events are captured when `physics::step()` returns `Some(food_index)`
- Food respawn events are captured from `WorldState::last_respawned_indices()`
- When a generation completes, the recording moves to `last_recording` for playback

**Playback:**
- The agent detail tab shows a "Replay Gen N" button when a recording is available
- Timeline scrubber, play/pause, and speed controls (0.5x–8x)
- Builds temporary `AgentSnapshot` and `WorldSnapshot` from recorded data at the current tick
- Food positions are reconstructed by replaying food events up to the current tick

#### Console Telemetry

Printed every 100 ticks for the selected agent:

```
[Tick   500] Agent 0 (gen 0, color: 0.8,0.2,0.2) | Agents: 3 | Age: 500 | Deaths: 0
  Energy: 85.2% | Integrity: 100.0% | Biome: FoodRich
  Brain: PredErr=0.42 (avg=0.51) | Memory: 45/128 | Explore: 0.68
  Homeo: gradient=+0.012 | urgency=0.15
  Behavior: 32% informed | Quality: 0.44 | Phase: exploring
  Life: 500 ticks (best: 500) | Deaths: 0
  Position: (12.3, 2.1, -8.5) | Facing: (0.71, 0.00, 0.71)
```

Window title is static ("xagent") — runtime stats (FPS, agent count, evolution state, wall time, ticks/sec) are displayed in the egui top bar, with selected-agent info on the HUD overlay.

---

### 3.6 Main Loop (`main.rs`)

#### Application Structure

Uses **winit's `ApplicationHandler`** trait with an `App` struct that owns all state:
renderer, camera, world, agents, logger, and simulation metadata.

#### Event Loop Flow

```
resumed()           → Create window, GPU, world, terrain mesh, spawn first agent
window_event()      → Dispatch keyboard/mouse/redraw events
about_to_wait()     → Request continuous redraw
```

#### Simulation Speed Controls

| Key | `ticks_per_frame` | Label |
|---|---|---|
| `1` | 1 | 1x |
| `2` | 2 | 2x |
| `3` | 5 | 5x |
| `4` | 10 | 10x |
| `5` | 100 | 100x |
| `6` | 1000 | 1000x |

Max ticks per frame is capped at `speed × 2` (up to 2000).

##### Brain Tick Decimation

At high speed multipliers the brain is the bottleneck, but full-fidelity brain
evaluation on every tick isn't needed when the simulation is fast-forwarding.
**Brain tick decimation** reduces brain calls while keeping physics accurate:

- The brain only runs every `brain_stride` ticks, where
  `brain_stride = sqrt(speed_multiplier)` (floored, minimum 1).
- Physics (`physics::step`) still runs **every** tick, so positions and
  collisions stay accurate.
- When the brain is skipped, the last `MotorCommand` is reused from
  `agent.cached_motor`.
- Brain ticks are **staggered** across agents:
  `(tick + agent_index) % brain_stride == 0` determines whether an agent's
  brain fires on a given tick. This spreads CPU load evenly across frames
  instead of spiking on a single tick.

| Speed | `brain_stride` | Brain calls / tick |
|---|---|---|
| 1× | 1 | every tick (no change) |
| 10× | 3 | ~1 in 3 |
| 100× | 10 | ~1 in 10 |
| 1000× | 31 | ~1 in 31 (~31× fewer brain calls) |

#### Agent Spawning

- **N key** — spawns a clone of the base `BrainConfig` (generation 0).
- **M key** — spawns a mutated variant (±10% per parameter, generation 0).

#### Evolution: Reproduction (Currently Disabled)

Reproduction is currently disabled (commented out) to focus on individual learning.
When enabled, `agent.can_reproduce(tick)` is true (alive, age ≥ 5000 ticks) and
`has_reproduced` is false and total agents < `MAX_AGENTS`:

1. Parent's `has_reproduced` is set to `true`.
2. `mutate_config()` perturbs the parent's `BrainConfig`:
   - Floats (`learning_rate`, `decay_rate`): multiplied by uniform random in [0.9, 1.1],
     clamped to ≥ 0.0001.
   - Integers (`memory_capacity`, `processing_slots`):
     multiplied by [0.9, 1.1], rounded, clamped to ≥ 1.
   - `visual_encoding_size` and `representation_dim` are **not mutated** (visual_encoding_size must match the sensory pipeline; representation_dim is locked to prevent weight inheritance breakage across generations).
3. Child spawns near parent (±5 units offset), generation = parent generation + 1.
4. Child gets a fresh brain with the mutated config.

#### Death & Respawn

1. On death: **death signal** fired first — `brain.death_signal()` sends a calibrated
   negative credit event (effective gradient ≈ -0.36) to the action selector, retroactively
   penalizing recent actions in the 64-tick history buffer. Calibrated to ~30× a single
   damage tick — enough to learn from death without obliterating positive learned behaviors.
2. Record `longest_life`, increment `death_count`, increment `generation`,
   log cause ("energy depletion" or "integrity failure"), set `respawn_cooldown = 60` frames.
3. Cooldown decrements each tick. At 0:
   - New `AgentBody` at random position.
   - **Partial health**: 50% energy, 70% integrity (not full health — no "free heal").
   - `life_start_tick` reset, `has_reproduced` cleared.
   - If `persist_brain` is true, the `Brain` is kept (learning survives death),
     and `brain.trauma(0.2)` is applied — 20% reinforcement decay that wipes the
     weakest memories while preserving the strongest.
   - If false, a fresh `Brain` is created from the same config.

#### Headless Mode

`--no-render` skips all window/GPU code and runs a tight loop with a single agent:
sleep `1/tick_rate` seconds between ticks, print telemetry every 100 ticks.
On death the agent is respawned with a **fresh brain** (no persistence in headless).

---

## Configuration

### WorldConfig

| Field | Default | Easy | Hard | Description |
|---|---|---|---|---|
| `world_size` | 256.0 | 256.0 | 256.0 | Square terrain side length (units) |
| `energy_depletion_rate` | 0.01 | 0.005 | 0.02 | Base metabolic cost per tick |
| `movement_energy_cost` | 0.005 | 0.002 | 0.01 | Energy cost per unit of movement magnitude |
| `hazard_damage_rate` | 1.0 | 0.5 | 2.0 | Integrity damage per tick in Danger biomes |
| `integrity_regen_rate` | 0.005 | 0.005 | 0.005 | Integrity recovery per tick (when energy > 50%) |
| `food_energy_value` | 20.0 | 30.0 | 15.0 | Energy restored per food item consumed |
| `food_density` | 0.005 | 0.005 | 0.001 | Food items per unit² in FoodRich biomes |
| `tick_rate` | 30.0 | 30.0 | 30.0 | Simulation ticks per second |
| `seed` | 42 | 42 | 42 | Random seed for terrain + biome generation |

**Presets:**

- **Easy** — Abundant food, slow drain, mild hazards. Good for observing long-term
  behavior emergence.
- **Normal** — Balanced defaults.
- **Hard** — Scarce food, fast drain, deadly hazards. Tests survival pressure.

### BrainConfig

| Field | Tiny | Default | Large | Description |
|---|---|---|---|---|
| `memory_capacity` | 24 | 128 | 512 | Max stored patterns |
| `processing_slots` | 8 | 16 | 32 | Max patterns recalled per tick |
| `visual_encoding_size` | 32 | 64 | 128 | Downsampled visual vector size |
| `representation_dim` | 16 | 32 | 64 | Internal representation length |
| `learning_rate` | 0.08 | 0.05 | 0.03 | Association update rate |
| `decay_rate` | 0.002 | 0.001 | 0.0005 | Unreinforced pattern decay per tick |

### CLI Flags

```
USAGE:
    xagent [OPTIONS]

OPTIONS:
    --brain-preset <PRESET>   Brain preset: tiny, default, large [default: default]
    --world-preset <PRESET>   World preset: easy, normal, hard   [default: normal]
    --config <PATH>           Load full config from a JSON file
    --seed <SEED>             Override random seed for world generation
    --tick-rate <RATE>        Override simulation ticks per second
    --no-render               Run headless (no window, just simulation + logging)
    --dump-config             Print the resolved config as JSON and exit
```

**Examples:**

```bash
# Default run
cargo run -p xagent-sandbox

# Hard world, tiny brain
cargo run -p xagent-sandbox -- --world-preset hard --brain-preset tiny

# Specific seed, fast tick rate
cargo run -p xagent-sandbox -- --seed 12345 --tick-rate 60

# Headless batch run
cargo run -p xagent-sandbox -- --no-render --world-preset easy

# Export config for reproducibility
cargo run -p xagent-sandbox -- --dump-config > my_config.json
cargo run -p xagent-sandbox -- --config my_config.json
```

---

## Controls Reference

### Camera

| Key / Input | Action |
|---|---|
| `W` / `S` | Move camera forward / backward |
| `A` / `D` | Move camera left / right |
| `E` | Move camera up |
| `Left Shift` | Move camera down |
| Left-click drag | Rotate camera (yaw + pitch) |
| Scroll wheel | Zoom in / out along view direction |

### Simulation

| Key | Action |
|---|---|
| `P` or `Space` | Toggle pause / resume |
| `1` | Speed 1x (1 tick/frame) |
| `2` | Speed 2x (2 ticks/frame) |
| `3` | Speed 5x (5 ticks/frame) |
| `4` | Speed 10x (10 ticks/frame) |
| `5` | Speed 100x (100 ticks/frame) |
| `6` | Speed 1000x (1000 ticks/frame) |
| Right-click | Select/focus nearest agent (0.05 NDC pick threshold) |
| `R` | Toggle brain persistence on death (persist ↔ reset) |
| `N` | Spawn a new agent (default config) |
| `M` | Spawn a new agent (mutated config) |
| `H` | Toggle heatmap overlay for selected agent |
| `Escape` | Print session summary and quit |

### egui UI Interaction

| Action | Effect |
|---|---|
| Click agent in sidebar | Select / focus that agent |
| Double-click agent in sidebar | Open an agent detail tab |
| Drag / scroll on viewport pane | Camera rotation / zoom (only when hovering the viewport) |
| Close detail tab | Click the × on the tab header |

### Visual Cues

| Indicator | Description |
|---|---|
| Yellow diamond | Floating marker above the currently selected agent |
| Colored ribbon trail | Linear ribbon of selected agent's full life path (distance-sampled, up to 4000 points, dirty-flag rebuild) |
| Palette-colored agent | Each agent has a unique static color matching its sidebar dot |
| Heatmap overlay (`H`) | Blue→yellow→red cells showing where the selected agent spent time |

---

## The Simulation Loop (Detailed)

Each frame, when the window requests a redraw:

```
1. Compute dt = min(elapsed since last frame, 0.05s)

2. Update camera position from held keys (WASD/E/Shift)

3. If not paused, for each tick (1–1000 per frame based on speed, capped at speed × 2 up to 2000):
   a. Collect all agent positions into a snapshot (Vec<(Vec3, bool)>)

   b. Compute brain_stride = floor(sqrt(speed_multiplier)), min 1

   c. For each living agent i:
      ├─ Build OtherAgent list (all agents except i)
      ├─ If (tick + i) % brain_stride == 0:        ← brain fires this tick
      │   ├─ extract_senses_with_others(agent.body, world, tick, others)
      │   │   → produces SensoryFrame
      │   ├─ brain.tick(&frame)
      │   │   → produces MotorCommand
      │   └─ agent.cached_motor = motor             ← cache for skipped ticks
      ├─ Else:
      │   └─ motor = agent.cached_motor              ← reuse last command
      ├─ physics::step(&mut agent.body, &motor, &mut world, dt)
      │   → updates position, velocity, energy, integrity, alive (every tick)
      └─ If selected agent: log to CSV, accumulate prediction error

   d. world.update(dt) — decrement food respawn timers, relocate respawned food

   e. Death/respawn processing:
      ├─ Dead agent with cooldown == 0 → log death, set cooldown = 60
      └─ Dead agent with cooldown > 0 → decrement; if 0 → respawn

   f. Increment global tick counter, mark food mesh as dirty

   g. Reproduction check:
      ├─ Find agents where can_reproduce() && !has_reproduced
      └─ For each: set has_reproduced, spawn_child (mutated config)

   h. Every 100 ticks: print telemetry to console

4. Fix selected_agent_idx if agents were added/removed

5. Rebuild dynamic GPU meshes:
   ├─ Food mesh (if dirty)
   └─ Agent mesh (combined, rebuilt every frame)

6. Build HUD bars for selected agent

7. Render:
   ├─ 3D pass: terrain + food + agents (depth-tested)
   ├─ HUD pass: background panels + status bars (alpha-blended)
   └─ Text pass: bitmap font labels (alpha-blended)
```

---

## Sensory Pipeline (Detailed)

### How Raw World State Becomes a SensoryFrame

```
┌────────────────────────┐
│  WorldState            │     ┌───────────────────────┐
│  ├─ terrain (heights)  │────►│ march_ray_unified()   │
│  ├─ biome_map          │     │  8×6 grid, 90° FOV    │──► vision.color [192 floats]
│  └─ food_items         │     │  50.0 max depth       │──► vision.depth [48 floats]
│                        │     │  1.0 ray step         │
│                        │     │  food + agents + terrain│
│                        │     └───────────────────────┘
│                        │
│  AgentBody             │     ┌───────────────────────┐
│  ├─ body.velocity   ───┼────►│ Proprioception        │──► velocity, facing,
│  ├─ body.facing      ──┼────►│                       │    angular_velocity
│  └─ angular_velocity ──┼────►│                       │
│                        │     └───────────────────────┘
│                        │
│  InternalState         │     ┌───────────────────────┐
│  ├─ energy/max_energy──┼────►│ Interoception         │──► energy_signal (0–1)
│  ├─ integrity/max    ──┼────►│                       │    integrity_signal (0–1)
│  ├─ prev_energy      ──┼────►│                       │    energy_delta
│  └─ prev_integrity   ──┼────►│                       │    integrity_delta
│                        │     └───────────────────────┘
│                        │
│  food_items + edges +  │     ┌───────────────────────┐
│  biome + other agents ─┼────►│ detect_touch()        │──► Vec<TouchContact>
│                        │     │  direction + intensity │    {direction, intensity,
│                        │     │  + surface_tag         │     surface_tag}
└────────────────────────┘     └───────────────────────┘
```

### What the Brain "Sees" vs What the Renderer Shows

The brain receives an **8 × 6 colored depth image** from simplified raycasting.
The renderer shows the actual **textured terrain mesh** with height-blended vertex
colors. These are related but not identical:

- The brain's vision uses flat biome colors (3 terrain options + food lime-green + agent magenta + sky).
- The renderer shows smooth height-based color gradients per biome with 3D food/agent cubes.
- The brain **can see food items** (lime green `[0.70, 0.95, 0.20, 1.0]`) and **other agents** (magenta `[0.9, 0.2, 0.6, 1.0]`) in its visual field.
- The brain also receives **touch contacts** — the top 4 contacts by intensity are encoded into 16 features (direction, intensity, surface tag).

### Touch Contact Generation

| What | Detection | Range | Direction | Intensity |
|---|---|---|---|---|
| Food | Distance check each non-consumed item | 3.0 | Agent → food | Linear falloff |
| Terrain edge | Distance from each of 4 world boundaries | 3.0 | Toward boundary | Linear falloff |
| Hazard | Biome check at agent position | 0 (zone) | `NEG_Y` (down) | Fixed 0.5 |
| Other agent | Distance check each living agent | 5.0 | Agent → other | Linear falloff |

---

## Multi-Agent & Evolution

### Agent Management

Agents are stored in a `Vec<Agent>`. The main loop iterates by index so it can
build the "other agents" list for inter-agent perception while maintaining mutable
access to the current agent.

### Agent Palette Colors

Each agent is assigned a **static palette color** at spawn. The same color appears in the 3D viewport (converted from sRGB to linear for correct GPU rendering) and in the egui sidebar's colored dot. Dead agents render as dark gray `[0.25, 0.25, 0.25]`.

### Selection Marker

The currently focused agent (selected via the sidebar or right-click) is highlighted with a bright
yellow diamond marker floating above it. The diamond has 8 triangular faces arranged
as an octahedron shape, making the selected agent easy to spot even in crowded scenes.

### Agent Trail

Each agent records its position using distance-based sampling — a new control point
is stored whenever the agent moves ≥ 3 units from the last sample (up to 4000 points
per life). The selected agent's trail is rendered as a **linear ribbon** using the
agent's own color. The mesh is only rebuilt when the trail data changes (dirty flag),
keeping CPU cost negligible even at high time multipliers.

The trail covers the agent's **entire current life** (birth to present). Only the
oldest 20% fades slightly; the rest is uniformly visible. The trail resets on
death/respawn.

### Reproduction

- **Currently disabled** (commented out) to focus on individual learning over evolution.
- When enabled: agent must survive 5000 ticks continuously.
- **Once per life**: `has_reproduced` flag prevents repeated spawning.
- **Population cap**: `MAX_AGENTS = 100`.
- **Mutation**: Each `BrainConfig` parameter is perturbed using momentum-biased perturbation. Each island maintains a per-parameter momentum vector that learns which mutation directions improve fitness. The perturbation combines random noise (±strength%) with a directional nudge from momentum. Parameters with strong momentum are pushed toward winning values; parameters with weak momentum get mostly random exploration. `visual_encoding_size` and `representation_dim` are preserved (visual_encoding_size must match the sensory pipeline; representation_dim is locked to prevent weight inheritance breakage across generations).
- **Generation tracking**: Generation increments on each death/respawn, tracking how many lives the agent has lived.

### Telemetry Focus

Clicking an agent in the **sidebar** selects it; double-clicking opens a dedicated
detail tab in the dock area. The selected agent is indicated by a yellow diamond
marker floating above it in the 3D viewport. Its data drives:
- **Sidebar**: colored dot, compact energy/integrity bars, phase label, death count, food consumed, best life. Sortable by ID, Energy, Integrity, Deaths, Longest Life, Prediction Error, or Fitness via dropdown.
- **Agent detail tab**: header with phase label, two-column vitals/motor display (energy/integrity bars + continuous forward/turn motor gauges with weight norms), side-by-side **Agent Vision** (8x6 upscaled color grid) and **Mini-Map** (top-down biome texture + food dots + agent markers with facing arrows), scrollable History chart (energy/integrity/prediction error/exploration, 30–10k ticks), **Decision Stream** (per-tick log with color-coded credit, motor output, gradient, urgency, patterns recalled), and **Replay Controls** (timeline scrubber, play/pause, 0.5x–8x speed, live/replay toggle).
- **Bottom console**: scrollable log of evolution events
- Trail ribbon showing full life path (up to 4000 distance-sampled points, dirty-flag rebuild)
- Heatmap overlay (when enabled with `H`)
- CSV logging

### Death & Respawn Guardrails

When brain persistence is enabled (default), death triggers four mechanisms:

1. **Death signal** — `brain.death_signal()` fires a calibrated negative credit event
   (effective gradient ≈ -0.36, ~30× a single damage tick). Updates state-dependent
   weights only, NOT global action biases — prevents catastrophic global bias
   destruction that causes "learned helplessness" straight-line walking.
2. **Partial respawn energy** — 50% energy, 70% integrity. No "free heal" from dying.
3. **Random respawn position** — unpredictable location.
4. **Memory trauma** — `brain.trauma(0.2)` applies 20% reinforcement decay. Weakest memories are wiped; strongest survive. Models the cognitive cost of catastrophic discontinuity.

The credit chain during danger encounters is:

```
damage onset (gradient spike, 3× pain amplified) → death event (calibrated retroactive punishment)
```

Suicide prevention is emergent: death is maximally unpredictable (massive prediction error), delivers the strongest negative learning signal, and the brain's core drive is minimizing prediction error.

---

## CPU Performance Optimizations

Several optimizations keep the simulation fast at scale:

- **Vision resolution**: reduced from 16×12 to 8×6 (48 pixels vs 192)
- **Ray march step**: increased from 0.5 to 1.0 units
- **Sky ray early-out**: rays that miss terrain exit immediately
- **Reused `others_buf`**: pre-allocated buffer for inter-agent data, reused each sim loop tick
- **Agent-agent collision**: O(n²) pairwise check using simple squared-distance math
- **Brain tick decimation**: at high speed multipliers, brain evaluations are reduced
  by a factor of `sqrt(speed_multiplier)` while physics still runs every tick.
  Staggered across agents so CPU load is spread evenly (see *Simulation Speed Controls*).
- **MAX_AGENTS**: raised from 20 → 100

---

## Testing

### Running Tests

```bash
cargo test -p xagent-sandbox
```

### Test Inventory (14 tests)

| Test | Category | Verifies |
|---|---|---|
| `motor_forward_moves_agent` | Physics | Forward motor command displaces agent in facing direction |
| `motor_turn_rotates_agent` | Physics | Turn motor command changes agent facing direction |
| `gravity_keeps_agent_on_terrain` | Physics | Agent dropped from height settles on terrain surface |
| `nan_motor_command_is_sanitized` | Physics | NaN/Infinity motor inputs don't corrupt position |
| `energy_depletes_over_time` | Agent | Idle agent loses energy from base metabolic cost |
| `agent_dies_at_zero_energy` | Agent | Agent with near-zero energy dies after movement |
| `consume_near_food_restores_energy` | Agent | Agent near food auto-consumes and increases energy |
| `terrain_height_is_deterministic` | World | Same seed produces identical heightmap |
| `biome_query_returns_valid_type` | World | `biome_at()` always returns a valid BiomeType |
| `terrain_height_interpolation_is_smooth` | World | Adjacent height samples don't jump wildly |
| `terrain_rejects_zero_subdivisions` | World | `#[should_panic]` — 0 subdivisions is rejected |
| `terrain_rejects_one_subdivision` | World | `#[should_panic]` — 1 subdivision is rejected |
| `sensory_frame_has_correct_dimensions` | Senses | Visual field is 8×6, buffers correctly sized |
| `interoception_matches_body_state` | Senses | Energy/integrity signals match InternalState |

---

## Design Decisions

### Why wgpu?

Cross-platform GPU abstraction that targets Vulkan (Linux/Windows), Metal (macOS),
and DX12. Safer than raw Ash/Vulkan, doesn't require external C dependencies like
OpenGL, and has first-class Rust support. `pollster` is used for blocking async
adapter/device creation.

### Why Procedural Terrain?

Seed-based generation means experiments are **exactly reproducible**. No asset files
needed. Multi-octave Perlin noise produces naturalistic terrain without hand-authoring.

### Why Simplified Vision (Not Render-to-Texture)?

Rendering the full 3D scene from each agent's viewpoint would require N extra render
passes per frame. For emergence experiments, biome-colored raycast sampling is
sufficient — agents need to distinguish regions, not read text on signs. The 16×12
resolution was chosen to be small enough for the brain's encoder but large enough to
carry spatial information.

### Why Discrete Food Items?

Explicit food items create a **clear reward signal** for homeostasis. The agent must:
1. Detect food visually (green cubes in the visual field)
2. Navigate to it (move within 2.5 units)

Food is auto-consumed on contact — the brain's job is to **move toward food**, not
to "decide to eat." This mirrors biology: organisms absorb nutrients automatically.
The learning challenge is purely navigational.

### Why Three Biome Types?

`FoodRich / Barren / Danger` creates a minimal decision landscape:
- **Approach** food-rich areas (positive reinforcement via energy gain).
- **Avoid** danger zones (negative reinforcement via integrity damage).
- **Traverse** barren zones (neutral, costs energy with no benefit).

This is the simplest set that produces interesting exploration/exploitation tradeoffs.

### Why Dynamic Vertex Buffers?

Agent and food meshes are rebuilt every frame from scratch. This is intentionally
simple — no transform matrices, no instancing, no scene graph. At 20 agents ×
24 vertices = 480 vertices, the cost is negligible. Simplicity beats performance
optimization at this scale.

---

## Performance Considerations

### What Scales Linearly with Agent Count

- **Sensory extraction**: O(N²) for inter-agent touch (each agent checks all others).
  With N ≤ 20 this is at most 380 distance checks per tick.
- **Brain processing**: O(N) ticks per frame, each involving memory search, prediction,
  and motor generation. This is the **dominant cost**.
- **Agent mesh rebuild**: O(N × 24 vertices) per frame — trivial.

### Vertex Buffer Rebuild Cost

- Terrain mesh: built **once** at startup (16,641 vertices). Never rebuilt.
- Food mesh: rebuilt when `food_dirty` flag is set (any tick that runs). At default
  density ≈ 200–400 food items × 24 vertices = ~5K–10K vertices.
- Agent mesh: rebuilt every frame. ≤ 20 agents × 24 vertices = ≤ 480 vertices.

### When to Worry

- **>20 agents**: The inter-agent perception cost becomes O(N²) and brain ticks
  multiply. The hard cap prevents this.
- **High ticks_per_frame (1000x)**: 1000 brain ticks per frame at 60 fps = 60,000 brain
  ticks/second per agent. With 20 agents = 1,200,000 brain ticks/second. Max ticks per frame
  cap scales with speed (`speed × 2`, capped at 2000).
- **Large BrainConfig**: `memory_capacity=1000` with `processing_slots=32` means
  searching 32 patterns per tick, each compared against a 64-dim vector.

---

## Known Limitations & Future Work

| Limitation | Detail |
|---|---|
| Simplified vision | Agents see biome-colored raycasts, not the actual rendered scene. Food items and other agents are visible as distinct colors (lime green and magenta respectively). |
| No audio channel | No sound-based sensory input — agents are effectively deaf. |
| Flat-ish terrain | Multi-octave noise produces rolling hills (±6.5 units) but no caves, overhangs, or vertical features. |
| No object manipulation | Agents cannot carry, build, or reshape the world. Push action has no physics effect. |
| No weather or lighting | No day/night cycle, rain, temperature, or lighting changes that would affect senses. |
| Single food type | All food items are identical — no nutritional variation or toxicity. |
| No inter-agent communication | Agents perceive each other via touch only — no signaling, vocalizations, or shared state. |
| Headless mode is single-agent | The `--no-render` mode runs only one agent with no persistence. |
| Food only in FoodRich biomes | Food never spawns in Barren or Danger zones, limiting foraging strategies. |
