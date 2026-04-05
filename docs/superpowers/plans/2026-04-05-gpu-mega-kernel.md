# GPU Mega-Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU-driven per-tick dispatch loop with a single GPU mega-kernel that runs N ticks autonomously, targeting 100× throughput improvement.

**Architecture:** Single WGSL mega-kernel (`@workgroup_size(256)`, `dispatch(1,1,1)`) with internal tick loop. All physics, vision, and brain phases run inside the kernel with `storageBarrier()`+`workgroupBarrier()` between phases. CPU becomes an observer that dispatches batches and reads back state asynchronously.

**Tech Stack:** Rust, wgpu 24, WGSL compute shaders, egui 0.31

**Spec:** `docs/superpowers/specs/2026-04-05-gpu-mega-kernel-design.md`

---

## Parallelization Map

```
Task 1: Foundation (common.wgsl + buffers.rs)
   ↓
┌──────────────────────────────────────────────┐
│  Task 2: Physics phases  (PARALLEL)          │
│  Task 3: Death phase     (PARALLEL)          │
│  Task 4: Vision phase    (PARALLEL)          │
│  Task 5: Brain phase     (PARALLEL)          │
└──────────────────────────────────────────────┘
   ↓
Task 6: mega_tick.wgsl entry point + gpu_mega_kernel.rs Rust module
   ↓
Task 7: Integration (main.rs + bench.rs)
   ↓
Task 8: Validation + cleanup
```

Tasks 2, 3, 4, 5 are **fully independent** and should be dispatched in parallel.

---

## Critical Porting Note: Atomic Buffer Types

In the existing shaders, some buffers are declared `array<u32>` (plain) in read-only contexts and `array<atomic<u32>>` in atomic-write contexts. In the mega-kernel, each buffer has ONE declaration shared across all phases. The following buffers are declared atomic:

- `food_flags: array<atomic<u32>>` — atomic CAS in food_detect, atomic store in food_respawn
- `food_grid: array<atomic<u32>>` — atomic add in food_grid_build and food_respawn
- `agent_grid: array<atomic<u32>>` — atomic add in agent_grid_build
- `collision_scratch: array<atomic<i32>>` — atomic add in collision_accumulate

**When porting phases that only READ these buffers** (e.g., vision reads food_grid and agent_grid, food_detect reads food_grid), you MUST use `atomicLoad(&buffer[i])` instead of `buffer[i]`. Plain indexing on `atomic<T>` types is a WGSL compilation error.

Similarly, for writes: use `atomicStore(&buffer[i], value)` instead of `buffer[i] = value` where the original shader used non-atomic writes.

---

## File Structure

### New Files
| File | Purpose |
|---|---|
| `crates/xagent-brain/src/shaders/mega/common.wgsl` | Buffer declarations, constants, shared helpers (RNG, grid, heightmap) |
| `crates/xagent-brain/src/shaders/mega/phase_clear.wgsl` | Phase 0: cooperative grid clearing |
| `crates/xagent-brain/src/shaders/mega/phase_food_grid.wgsl` | Phase 1: food spatial grid build |
| `crates/xagent-brain/src/shaders/mega/phase_physics.wgsl` | Phase 2: agent physics integration |
| `crates/xagent-brain/src/shaders/mega/phase_death.wgsl` | Phase 3: death detection + respawn + brain reset |
| `crates/xagent-brain/src/shaders/mega/phase_food_detect.wgsl` | Phase 4: food consumption |
| `crates/xagent-brain/src/shaders/mega/phase_food_respawn.wgsl` | Phase 5: food timer + respawn |
| `crates/xagent-brain/src/shaders/mega/phase_agent_grid.wgsl` | Phase 6: agent spatial grid build |
| `crates/xagent-brain/src/shaders/mega/phase_collision.wgsl` | Phases 7-9: collision accumulate + apply |
| `crates/xagent-brain/src/shaders/mega/phase_vision.wgsl` | Phase 10a: 48-ray vision (serialized per thread) |
| `crates/xagent-brain/src/shaders/mega/phase_brain.wgsl` | Phase 10b: all 7 brain passes merged |
| `crates/xagent-brain/src/shaders/mega/mega_tick.wgsl` | Entry point: tick loop calling all phases |
| `crates/xagent-brain/src/gpu_mega_kernel.rs` | Rust struct: pipeline, bind group, dispatch, readback |

### Modified Files
| File | Change |
|---|---|
| `crates/xagent-brain/src/buffers.rs` | Add `WC_TICKS_TO_RUN` constant, update `build_world_config` |
| `crates/xagent-brain/src/lib.rs` | Add `pub mod gpu_mega_kernel;` |
| `crates/xagent-sandbox/src/main.rs` | Replace tick loop with mega-kernel dispatch |
| `crates/xagent-sandbox/src/bench.rs` | Use mega-kernel for benchmark |

---

### Task 1: Foundation — common.wgsl + buffer constants

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/common.wgsl`
- Modify: `crates/xagent-brain/src/buffers.rs:156-161`

This task creates the shared WGSL foundation that all phase fragments depend on. Every buffer declaration, constant, and helper function lives here.

- [ ] **Step 1: Add WC_TICKS_TO_RUN constant to buffers.rs**

In `crates/xagent-brain/src/buffers.rs`, after line 161 (`pub const WC_GRID_OFFSET: usize = 19;`), add:

```rust
pub const WC_TICKS_TO_RUN: usize = 20;
```

Update `build_world_config` at line 497 to accept `ticks_to_run`:

```rust
pub fn build_world_config(
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
) -> Vec<f32> {
```

And after line 521 (`wc[WC_GRID_OFFSET] = go as f32;`), add:

```rust
    wc[WC_TICKS_TO_RUN] = ticks_to_run as f32;
```

Fix all existing callers of `build_world_config` to pass `ticks_to_run: 1` (preserving current behavior):
- `crates/xagent-brain/src/gpu_physics.rs` — search for `build_world_config(` and add `, 1` as the last arg.

- [ ] **Step 2: Create common.wgsl with all buffer declarations and helpers**

Create `crates/xagent-brain/src/shaders/mega/common.wgsl`:

```wgsl
// ── Constants ──────────────────────────────────────────────
const DIM: u32 = 32u;
const MEMORY_CAP: u32 = 128u;
const RECALL_K: u32 = 16u;
const SENSORY_STRIDE: u32 = 267u;
const BRAIN_STRIDE: u32 = 8468u;
const PATTERN_STRIDE: u32 = 5251u;
const HISTORY_STRIDE: u32 = 2370u;
const DECISION_STRIDE: u32 = 68u;
const PHYS_STRIDE: u32 = 24u;
const FEATURE_COUNT: u32 = 217u;
const MOTOR_RING_ENTRY: u32 = 5u;
const HISTORY_DEPTH: u32 = 64u;

// Vision layout
const VISION_W: u32 = 8u;
const VISION_H: u32 = 6u;
const VISION_COLOR_COUNT: u32 = 192u; // W*H*4
const VISION_DEPTH_COUNT: u32 = 48u;  // W*H
const NON_VISUAL_COUNT: u32 = 27u;

// Physics field offsets (P_*)
const P_POS_X: u32 = 0u;
const P_POS_Y: u32 = 1u;
const P_POS_Z: u32 = 2u;
const P_VEL_X: u32 = 3u;
const P_VEL_Y: u32 = 4u;
const P_VEL_Z: u32 = 5u;
const P_FACING_X: u32 = 6u;
const P_FACING_Y: u32 = 7u;
const P_FACING_Z: u32 = 8u;
const P_YAW: u32 = 9u;
const P_ANGULAR_VEL: u32 = 10u;
const P_ENERGY: u32 = 11u;
const P_MAX_ENERGY: u32 = 12u;
const P_INTEGRITY: u32 = 13u;
const P_MAX_INTEGRITY: u32 = 14u;
const P_PREV_ENERGY: u32 = 15u;
const P_PREV_INTEGRITY: u32 = 16u;
const P_ALIVE: u32 = 17u;
const P_FOOD_COUNT: u32 = 18u;
const P_TICKS_ALIVE: u32 = 19u;
const P_DIED_FLAG: u32 = 20u;
const P_MEMORY_CAP: u32 = 21u;
const P_PROCESSING_SLOTS: u32 = 22u;

// Brain state offsets (O_*)
const O_ENC_WEIGHTS: u32 = 0u;
const O_ENC_BIASES: u32 = 6944u;
const O_PRED_WEIGHTS: u32 = 6976u;
const O_PRED_CTX_WT: u32 = 8000u;
const O_PRED_ERR_RING: u32 = 8001u;
const O_PRED_ERR_CURSOR: u32 = 8129u;
const O_PRED_ERR_COUNT: u32 = 8130u;
const O_HAB_EMA: u32 = 8131u;
const O_HAB_ATTEN: u32 = 8163u;
const O_PREV_ENCODED: u32 = 8195u;
const O_HOMEO: u32 = 8227u;
const O_ACT_FWD_WTS: u32 = 8233u;
const O_ACT_TURN_WTS: u32 = 8265u;
const O_ACT_BIASES: u32 = 8297u;
const O_EXPLORATION_RATE: u32 = 8299u;
const O_FATIGUE_FWD_RING: u32 = 8300u;
const O_FATIGUE_TURN_RING: u32 = 8364u;
const O_FATIGUE_CURSOR: u32 = 8428u;
const O_FATIGUE_FACTOR: u32 = 8429u;
const O_FATIGUE_LEN: u32 = 8430u;
const O_PREV_PREDICTION: u32 = 8431u;
const O_TICK_COUNT: u32 = 8463u;
const O_HAB_SENSITIVITY: u32 = 8464u;
const O_HAB_MAX_CURIOSITY: u32 = 8465u;
const O_FATIGUE_RECOVERY: u32 = 8466u;
const O_FATIGUE_FLOOR: u32 = 8467u;

// Pattern buffer offsets
const O_PAT_STATES: u32 = 0u;
const O_PAT_NORMS: u32 = 4096u;
const O_PAT_REINF: u32 = 4224u;
const O_PAT_MOTOR: u32 = 4352u;
const O_PAT_META: u32 = 4736u;
const O_PAT_ACTIVE: u32 = 5120u;
const O_ACTIVE_COUNT: u32 = 5248u;
const O_MIN_REINF_IDX: u32 = 5249u;
const O_LAST_STORED_IDX: u32 = 5250u;

// History buffer offsets
const O_MOTOR_RING: u32 = 0u;
const O_STATE_RING: u32 = 320u;
const O_HIST_CURSOR: u32 = 2368u;
const O_HIST_LEN: u32 = 2369u;

// World config offsets (index into wconfig uniform)
const WC_WORLD_SIZE: u32 = 0u;
const WC_DT: u32 = 1u;
const WC_ENERGY_DEPLETION: u32 = 2u;
const WC_MOVEMENT_COST: u32 = 3u;
const WC_HAZARD_DAMAGE: u32 = 4u;
const WC_INTEGRITY_REGEN: u32 = 5u;
const WC_FOOD_ENERGY: u32 = 6u;
const WC_FOOD_RADIUS: u32 = 7u;
const WC_TERRAIN_VPS: u32 = 8u;
const WC_TERRAIN_INV_STEP: u32 = 9u;
const WC_TERRAIN_HALF: u32 = 10u;
const WC_BIOME_INV_CELL: u32 = 11u;
const WC_FOOD_COUNT: u32 = 12u;
const WC_AGENT_COUNT: u32 = 13u;
const WC_TICK: u32 = 14u;        // start_tick for this batch
const WC_RNG_SEED: u32 = 15u;
const WC_WORLD_HALF_BOUND: u32 = 16u;
const WC_BIOME_GRID_RES: u32 = 17u;
const WC_GRID_WIDTH: u32 = 18u;
const WC_GRID_OFFSET: u32 = 19u;
const WC_TICKS_TO_RUN: u32 = 20u;

// Grid constants
const CELL_SIZE: f32 = 8.0;
const FOOD_MAX_PER_CELL: u32 = 16u;
const FOOD_CELL_STRIDE: u32 = 17u;   // 1 count + 16 indices
const AGENT_MAX_PER_CELL: u32 = 32u;
const AGENT_CELL_STRIDE: u32 = 33u;  // 1 count + 32 indices
const COLLISION_MIN_DIST: f32 = 2.0;
const COLLISION_FIXED_SCALE: f32 = 1000.0;

// Vision constants
const VISION_FOV_H: f32 = 2.094;  // 120 degrees
const VISION_FOV_V: f32 = 1.047;  // 60 degrees
const VISION_MAX_DIST: f32 = 50.0;
const VISION_STEP: f32 = 0.5;
const TOUCH_FOOD_RANGE: f32 = 3.0;
const TOUCH_AGENT_RANGE: f32 = 5.0;

// Biome types
const BIOME_FOOD_RICH: u32 = 0u;
const BIOME_NEUTRAL: u32 = 1u;
const BIOME_DANGER: u32 = 2u;

// ── Buffer Bindings ────────────────────────────────────────
// All 17 bindings for the mega-kernel in one bind group.
@group(0) @binding(0) var<storage, read_write> agent_phys: array<f32>;
@group(0) @binding(1) var<storage, read_write> decision_buf: array<f32>;
@group(0) @binding(2) var<storage, read> heightmap: array<f32>;
@group(0) @binding(3) var<storage, read> biome_grid: array<u32>;
@group(0) @binding(4) var<uniform> wconfig: array<vec4<f32>, 6>;
@group(0) @binding(5) var<storage, read_write> food_state: array<f32>;
@group(0) @binding(6) var<storage, read_write> food_flags: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> food_grid: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> agent_grid: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> collision_scratch: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> sensory_buf: array<f32>;
@group(0) @binding(11) var<storage, read_write> brain_state: array<f32>;
@group(0) @binding(12) var<storage, read_write> pattern_buf: array<f32>;
@group(0) @binding(13) var<storage, read_write> history_buf: array<f32>;
@group(0) @binding(14) var<uniform> brain_config: array<vec4<f32>, 2>;

// ── World Config Accessors ─────────────────────────────────
fn wc_f32(idx: u32) -> f32 {
    return wconfig[idx / 4u][idx % 4u];
}

fn wc_u32(idx: u32) -> u32 {
    return u32(wc_f32(idx));
}

// ── RNG ────────────────────────────────────────────────────
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_to_float(h: u32) -> f32 {
    return f32(h) / 4294967295.0;
}

// ── Grid Helpers ───────────────────────────────────────────
fn cell_coord(x: f32, z: f32) -> vec2<i32> {
    let gw = i32(wc_f32(WC_GRID_WIDTH));
    let go = i32(wc_f32(WC_GRID_OFFSET));
    return vec2<i32>(
        clamp(i32(floor(x / CELL_SIZE)) + go, 0, gw - 1),
        clamp(i32(floor(z / CELL_SIZE)) + go, 0, gw - 1),
    );
}

fn cell_index(cx: i32, cz: i32) -> u32 {
    let gw = u32(wc_f32(WC_GRID_WIDTH));
    return u32(cz) * gw + u32(cx);
}

// ── Terrain Sampling ───────────────────────────────────────
fn sample_height(x: f32, z: f32) -> f32 {
    let vps = wc_f32(WC_TERRAIN_VPS);
    let inv_step = wc_f32(WC_TERRAIN_INV_STEP);
    let half = wc_f32(WC_TERRAIN_HALF);
    let fx = clamp((x + half) * inv_step, 0.0, vps - 2.0);
    let fz = clamp((z + half) * inv_step, 0.0, vps - 2.0);
    let ix = u32(floor(fx));
    let iz = u32(floor(fz));
    let dx = fx - floor(fx);
    let dz = fz - floor(fz);
    let w = u32(vps);
    let h00 = heightmap[iz * w + ix];
    let h10 = heightmap[iz * w + ix + 1u];
    let h01 = heightmap[(iz + 1u) * w + ix];
    let h11 = heightmap[(iz + 1u) * w + ix + 1u];
    return mix(mix(h00, h10, dx), mix(h01, h11, dx), dz);
}

// ── Biome Sampling ─────────────────────────────────────────
fn sample_biome(x: f32, z: f32) -> u32 {
    let inv_cell = wc_f32(WC_BIOME_INV_CELL);
    let res = u32(wc_f32(WC_BIOME_GRID_RES));
    let half = wc_f32(WC_TERRAIN_HALF);
    let bx = clamp(u32(floor((x + half) * inv_cell)), 0u, res - 1u);
    let bz = clamp(u32(floor((z + half) * inv_cell)), 0u, res - 1u);
    return biome_grid[bz * res + bx];
}

// ── Fast Tanh ──────────────────────────────────────────────
fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}
```

**NOTE:** The exact constants for VISION_FOV_H, VISION_FOV_V, VISION_MAX_DIST, VISION_STEP, TOUCH_FOOD_RANGE, TOUCH_AGENT_RANGE, COLLISION_MIN_DIST, FOOD_MAX_PER_CELL, AGENT_MAX_PER_CELL, and COLLISION_FIXED_SCALE must be verified against the existing shader files. Read the existing `shaders/vision.wgsl`, `shaders/food_grid_build.wgsl`, `shaders/collision_accumulate.wgsl`, etc. to get the exact values. Update common.wgsl to match.

- [ ] **Step 3: Verify constants match existing shaders**

Read each existing shader in `crates/xagent-brain/src/shaders/` and verify every constant in common.wgsl matches. Fix any discrepancies. Pay special attention to:
- Grid cell sizes and max-per-cell counts
- Vision FOV, step size, max distance
- Collision min distance and fixed-point scale
- Touch ranges
- Biome type enum values

- [ ] **Step 4: Compile check**

Run: `cargo check -p xagent-brain`

The `build_world_config` signature change will cause errors in callers. Fix each by adding `, 1` as the last argument (for `ticks_to_run`). Typical locations:
- `gpu_physics.rs` — `upload_world_config` method
- `bench.rs` — any direct call

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/buffers.rs crates/xagent-brain/src/shaders/mega/
git commit -m "feat: add mega-kernel foundation — common.wgsl + WC_TICKS_TO_RUN"
```

---

### Task 2: Physics Phase Fragments (PARALLEL with Tasks 3, 4, 5)

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/phase_clear.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_food_grid.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_physics.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_food_detect.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_food_respawn.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_agent_grid.wgsl`
- Create: `crates/xagent-brain/src/shaders/mega/phase_collision.wgsl`
- Read (reference): `crates/xagent-brain/src/shaders/physics.wgsl`, `food_grid_build.wgsl`, `food_detect.wgsl`, `food_respawn.wgsl`, `agent_grid_build.wgsl`, `collision_accumulate.wgsl`, `collision_apply.wgsl`

Each phase fragment defines functions (no entry point). They rely on buffer variables and helpers from `common.wgsl`.

**Porting pattern for every shader:** The existing shader has `@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>)` which uses `gid.x` as the entity index. In the mega-kernel, the entity index comes from a `tid` parameter. The port transforms: `gid.x` → function parameter `tid` (or a loop index for food operations).

- [ ] **Step 1: Create phase_clear.wgsl**

This is NEW code — no existing shader equivalent. All 256 threads cooperate to zero the three grid/scratch buffers.

Read the existing `food_grid_build.wgsl`, `agent_grid_build.wgsl` to understand the grid buffer sizes. The food grid has `grid_width * grid_width * FOOD_CELL_STRIDE` u32 entries. The agent grid has `grid_width * grid_width * AGENT_CELL_STRIDE` u32 entries. The collision scratch has `agent_count * 3` i32 entries.

Write `crates/xagent-brain/src/shaders/mega/phase_clear.wgsl`:

```wgsl
// Phase 0: Cooperative grid clearing.
// All 256 threads divide the work of zeroing food_grid, agent_grid, collision_scratch.
fn phase_clear(tid: u32) {
    let gw = wc_u32(WC_GRID_WIDTH);
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let food_grid_size = gw * gw * FOOD_CELL_STRIDE;
    let agent_grid_size = gw * gw * AGENT_CELL_STRIDE;
    let scratch_size = agent_count * 3u;

    // Clear food grid
    for (var i = tid; i < food_grid_size; i += 256u) {
        atomicStore(&food_grid[i], 0u);
    }
    // Clear agent grid
    for (var i = tid; i < agent_grid_size; i += 256u) {
        atomicStore(&agent_grid[i], 0u);
    }
    // Clear collision scratch
    for (var i = tid; i < scratch_size; i += 256u) {
        atomicStore(&collision_scratch[i], 0i);
    }
}
```

- [ ] **Step 2: Create phase_food_grid.wgsl**

Read `crates/xagent-brain/src/shaders/food_grid_build.wgsl` (37 lines). Port the `main` function body to a function `phase_food_grid_build(tid: u32)`. Key changes:
- Replace `gid.x` with loop: `for (var i = tid; i < food_count; i += 256u) { ... }`
- The existing shader skips consumed food (`food_flags[i] != 0`). Preserve this.
- Uses `atomicAdd` on `food_grid` cells to insert food indices. Preserve this.

```wgsl
fn phase_food_grid_build(tid: u32) {
    let food_count = wc_u32(WC_FOOD_COUNT);
    for (var i = tid; i < food_count; i += 256u) {
        // Port body of food_grid_build.wgsl main() here,
        // replacing gid.x with i
    }
}
```

- [ ] **Step 3: Create phase_physics.wgsl**

Read `crates/xagent-brain/src/shaders/physics.wgsl` (160 lines). Port to `fn phase_physics(tid: u32, tick: u32)`. Key changes:
- Replace `gid.x` with `tid`
- Replace `wconfig[3][2]` tick reads with the `tick` parameter
- All buffer accesses remain the same (agent_phys, decision_buf, heightmap, biome_grid, wconfig)
- Death check at the end sets `agent_phys[base + P_DIED_FLAG] = 1.0` — preserve exactly

```wgsl
fn phase_physics(tid: u32, tick: u32) {
    // Port body of physics.wgsl main() here
    // gid.x → tid, tick from parameter not wconfig
}
```

- [ ] **Step 4: Create phase_food_detect.wgsl**

Read `crates/xagent-brain/src/shaders/food_detect.wgsl` (80 lines). Port to `fn phase_food_detect(tid: u32)`.
- Replace `gid.x` with `tid`
- Preserve atomic compare-exchange for food claiming
- Preserve 3×3 grid neighborhood search

- [ ] **Step 5: Create phase_food_respawn.wgsl**

Read `crates/xagent-brain/src/shaders/food_respawn.wgsl` (109 lines). Port to `fn phase_food_respawn(tid: u32, tick: u32)`.
- Use loop: `for (var i = tid; i < food_count; i += 256u)`
- Replace `gid.x` with loop variable `i`
- The tick parameter feeds into `pcg_hash` for RNG seeding — make sure to pass it

- [ ] **Step 6: Create phase_agent_grid.wgsl**

Read `crates/xagent-brain/src/shaders/agent_grid_build.wgsl` (37 lines). Port to `fn phase_agent_grid_build(tid: u32)`.
- Replace `gid.x` with `tid`

- [ ] **Step 7: Create phase_collision.wgsl**

Read `crates/xagent-brain/src/shaders/collision_accumulate.wgsl` (84 lines) and `collision_apply.wgsl` (23 lines). Port to two functions:

```wgsl
fn phase_collision_accumulate(tid: u32) {
    // Port collision_accumulate.wgsl main(), gid.x → tid
}

fn phase_collision_apply(tid: u32) {
    // Port collision_apply.wgsl main(), gid.x → tid
}
```

- [ ] **Step 8: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/phase_clear.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_food_grid.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_physics.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_food_detect.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_food_respawn.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_agent_grid.wgsl \
        crates/xagent-brain/src/shaders/mega/phase_collision.wgsl
git commit -m "feat: add mega-kernel physics phase WGSL fragments"
```

---

### Task 3: Death Phase Fragment (PARALLEL with Tasks 2, 4, 5)

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/phase_death.wgsl`
- Read (reference): `crates/xagent-brain/src/gpu_brain.rs:565-633` (flush_death_signals logic)

This is entirely NEW code. No existing shader equivalent. The logic is ported from the Rust `flush_death_signals` method.

- [ ] **Step 1: Create phase_death.wgsl**

```wgsl
// Phase 3: Death detection + respawn + brain reset.
// Runs after physics. If an agent died this tick, immediately respawn it
// and reset its brain state. This replaces the CPU-side death readback loop.

fn phase_death_respawn(tid: u32, tick: u32) {
    let base = tid * PHYS_STRIDE;
    if (agent_phys[base + P_DIED_FLAG] < 0.5) {
        return;
    }

    // ── 1. Pick spawn position via GPU RNG + biome check ──
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);
    var seed = pcg_hash(tick * 256u + tid);
    var spawn_x = 0.0;
    var spawn_z = 0.0;
    var spawn_y = 0.0;
    for (var attempt = 0u; attempt < 50u; attempt++) {
        seed = pcg_hash(seed);
        let rx = hash_to_float(seed) * 2.0 * world_half - world_half;
        seed = pcg_hash(seed);
        let rz = hash_to_float(seed) * 2.0 * world_half - world_half;
        if (sample_biome(rx, rz) != BIOME_DANGER) {
            spawn_x = rx;
            spawn_z = rz;
            spawn_y = sample_height(rx, rz);
            break;
        }
    }

    // ── 2. Reset physics state ──
    let max_e = agent_phys[base + P_MAX_ENERGY];
    let max_i = agent_phys[base + P_MAX_INTEGRITY];
    agent_phys[base + P_POS_X] = spawn_x;
    agent_phys[base + P_POS_Y] = spawn_y;
    agent_phys[base + P_POS_Z] = spawn_z;
    agent_phys[base + P_VEL_X] = 0.0;
    agent_phys[base + P_VEL_Y] = 0.0;
    agent_phys[base + P_VEL_Z] = 0.0;
    agent_phys[base + P_FACING_X] = 0.0;
    agent_phys[base + P_FACING_Y] = 0.0;
    agent_phys[base + P_FACING_Z] = 1.0;
    agent_phys[base + P_YAW] = 0.0;
    agent_phys[base + P_ANGULAR_VEL] = 0.0;
    agent_phys[base + P_ENERGY] = max_e;
    agent_phys[base + P_INTEGRITY] = max_i;
    agent_phys[base + P_PREV_ENERGY] = max_e;
    agent_phys[base + P_PREV_INTEGRITY] = max_i;
    agent_phys[base + P_ALIVE] = 1.0;
    agent_phys[base + P_DIED_FLAG] = 0.0;
    // food_count and ticks_alive intentionally preserved for fitness tracking

    // ── 3. Reset brain state ──
    // In the mega-kernel, brain_idx == tid (1:1 mapping)
    let brain_base = tid * BRAIN_STRIDE;
    let pat_base = tid * PATTERN_STRIDE;
    let hist_base = tid * HISTORY_STRIDE;

    // Halve reinforcement values (preserves learned memories partially)
    for (var i = 0u; i < MEMORY_CAP; i++) {
        let idx = pat_base + O_PAT_REINF + i;
        pattern_buf[idx] *= 0.5;
    }

    // Zero homeostasis EMAs (6 values)
    for (var i = 0u; i < 6u; i++) {
        brain_state[brain_base + O_HOMEO + i] = 0.0;
    }

    // Reset exploration rate to 0.5
    brain_state[brain_base + O_EXPLORATION_RATE] = 0.5;

    // Zero action history ring buffer
    for (var i = 0u; i < HISTORY_STRIDE; i++) {
        history_buf[hist_base + i] = 0.0;
    }
}
```

**NOTE:** Verify that the current `flush_death_signals` in `gpu_brain.rs:565-633` doesn't do anything beyond what's listed above. Specifically check:
- Does it reset `O_PREV_ENCODED`? If so, add that.
- Does it reset fatigue rings? If so, add that.
- Does it reset prediction error ring? Check and match.

- [ ] **Step 2: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/phase_death.wgsl
git commit -m "feat: add mega-kernel death/respawn phase — GPU-side brain reset"
```

---

### Task 4: Vision Phase Fragment (PARALLEL with Tasks 2, 3, 5)

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/phase_vision.wgsl`
- Read (reference): `crates/xagent-brain/src/shaders/vision.wgsl` (362 lines)

Major rewrite: original uses `@workgroup_size(48)` with 48 threads per agent (1 per ray). The mega-kernel version serializes all 48 rays in a single thread using a loop.

- [ ] **Step 1: Create phase_vision.wgsl**

Read `crates/xagent-brain/src/shaders/vision.wgsl` thoroughly. Understand:
- How `local_invocation_id.x` maps to ray index (0..47)
- How thread 0 fills non-visual sensory data (proprioception, touch)
- The ray marching loop per ray

Port to `fn phase_vision(tid: u32)`:

```wgsl
fn phase_vision(tid: u32) {
    // Read agent state
    let base = tid * PHYS_STRIDE;
    if (agent_phys[base + P_ALIVE] < 0.5) { return; }

    let pos = vec3f(agent_phys[base + P_POS_X], agent_phys[base + P_POS_Y], agent_phys[base + P_POS_Z]);
    let facing = vec3f(agent_phys[base + P_FACING_X], agent_phys[base + P_FACING_Y], agent_phys[base + P_FACING_Z]);
    let yaw = agent_phys[base + P_YAW];
    let s_base = tid * SENSORY_STRIDE;

    // ── Vision rays: serialize the 48-ray grid ──
    // Original: 48 threads each handle 1 ray
    // Now: 1 thread loops over all 48 rays
    for (var ray_idx = 0u; ray_idx < VISION_W * VISION_H; ray_idx++) {
        // Port the per-ray logic from vision.wgsl here.
        // Replace local_invocation_id.x with ray_idx.
        // Compute ray direction from ray_idx, yaw, FOV.
        // March the ray, write color (4 f32) and depth (1 f32) to sensory_buf.
        // Color: sensory_buf[s_base + ray_idx * 4 + 0..3]
        // Depth: sensory_buf[s_base + VISION_COLOR_COUNT + ray_idx]
    }

    // ── Non-visual sensory data ──
    // Port the thread-0 block from vision.wgsl that fills:
    // - velocity (3 f32)
    // - facing (3 f32)
    // - angular velocity (1 f32)
    // - energy/integrity signals (2 f32)
    // - energy/integrity deltas (2 f32)
    // - touch contacts (16 f32: 4 contacts × 4 values)
    // In the original, only thread 0 writes these. Now every agent thread writes them.
    // Port the logic, writing to sensory_buf[s_base + VISION_COLOR_COUNT + VISION_DEPTH_COUNT + ...]
}
```

The ray marching loop and non-visual data packing are the bulk of vision.wgsl. Port them faithfully — the logic is identical, only the parallelism model changes (loop of 48 instead of 48 threads).

- [ ] **Step 2: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/phase_vision.wgsl
git commit -m "feat: add mega-kernel vision phase — serialized 48-ray marching"
```

---

### Task 5: Brain Phase Fragment (PARALLEL with Tasks 2, 3, 4)

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/phase_brain.wgsl`
- Read (reference): All 7 brain shaders in `crates/xagent-brain/src/shaders/`: `feature_extract.wgsl` (76 lines), `encode.wgsl` (22 lines), `habituate_homeo.wgsl` (106 lines), `recall_score.wgsl` (44 lines), `recall_topk.wgsl` (53 lines), `predict_and_act.wgsl` (362 lines), `learn_and_store.wgsl` (185 lines)

This merges all 7 brain passes into a single function. Key change: intermediate results (features, encoded, habituated, homeo_out, similarities, recall) become local arrays instead of global buffers.

- [ ] **Step 1: Create phase_brain.wgsl with all 7 passes**

The structure:

```wgsl
fn phase_brain(tid: u32) {
    if (agent_phys[tid * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    // ── Local arrays for intermediates ──
    // These replace the global features_buf, encoded_buf, etc.
    var features: array<f32, 217>;    // was features_buf
    var encoded: array<f32, 32>;      // was encoded_buf
    var habituated: array<f32, 32>;   // was habituated_buf
    var homeo_out: array<f32, 6>;     // was homeo_out_buf
    var similarities: array<f32, 128>; // was similarities_buf
    var recall: array<f32, 17>;       // was recall_buf (16 indices + count)

    // ── Pass 1: feature_extract ──
    brain_feature_extract(tid, &features);

    // ── Pass 2: encode ──
    brain_encode(tid, &features, &encoded);

    // ── Pass 3: habituate_homeo ──
    brain_habituate_homeo(tid, &encoded, &habituated, &homeo_out);

    // ── Pass 4: recall_score ──
    brain_recall_score(tid, &habituated, &similarities);

    // ── Pass 5: recall_topk ──
    brain_recall_topk(tid, &similarities, &recall);

    // ── Pass 6: predict_and_act ──
    brain_predict_and_act(tid, &habituated, &homeo_out, &recall);

    // ── Pass 7: learn_and_store ──
    brain_learn_and_store(tid, &features, &habituated, &homeo_out);
}
```

Each sub-function is a port of the corresponding shader. The key transformation for each:

**brain_feature_extract**: Read `feature_extract.wgsl`. Port `main()` body.
- Reads from `sensory_buf[tid * SENSORY_STRIDE + ...]`
- Writes to `(*out_features)[i]` instead of `features[tid * FEATURE_COUNT + i]`

```wgsl
fn brain_feature_extract(tid: u32, out_features: ptr<function, array<f32, 217>>) {
    let s_base = tid * SENSORY_STRIDE;
    // Port feature_extract.wgsl body here.
    // Replace: features[base + i] → (*out_features)[i]
    // Replace: sensory[s_base + i] stays the same (reads from sensory_buf global)
}
```

**brain_encode**: Read `encode.wgsl`. Port `main()` body.
- Reads `(*in_features)[f]` and `brain_state[tid * BRAIN_STRIDE + O_ENC_WEIGHTS + ...]`
- Writes to `(*out_encoded)[d]`

```wgsl
fn brain_encode(
    tid: u32,
    in_features: ptr<function, array<f32, 217>>,
    out_encoded: ptr<function, array<f32, 32>>,
) {
    let b = tid * BRAIN_STRIDE;
    for (var d = 0u; d < DIM; d++) {
        var sum = brain_state[b + O_ENC_BIASES + d];
        for (var f = 0u; f < FEATURE_COUNT; f++) {
            sum += (*in_features)[f] * brain_state[b + O_ENC_WEIGHTS + f * DIM + d];
        }
        (*out_encoded)[d] = fast_tanh(sum);
    }
}
```

**brain_habituate_homeo**: Read `habituate_homeo.wgsl` (106 lines). Port `main()` body.
- Reads `(*in_encoded)[d]` and `sensory_buf` for energy/integrity signals
- Reads/writes `brain_state` for O_HAB_EMA, O_HAB_ATTEN, O_HOMEO, O_PREV_ENCODED
- Writes `(*out_habituated)[d]` and `(*out_homeo_out)[i]`

```wgsl
fn brain_habituate_homeo(
    tid: u32,
    in_encoded: ptr<function, array<f32, 32>>,
    out_habituated: ptr<function, array<f32, 32>>,
    out_homeo: ptr<function, array<f32, 6>>,
) {
    // Port habituate_homeo.wgsl body.
    // encoded[base + d] → (*in_encoded)[d]
    // habituated[base + d] → (*out_habituated)[d]
    // homeo_out[h_base + i] → (*out_homeo)[i]
    // brain_state reads/writes stay global
}
```

**brain_recall_score**: Read `recall_score.wgsl` (44 lines).

```wgsl
fn brain_recall_score(
    tid: u32,
    in_habituated: ptr<function, array<f32, 32>>,
    out_similarities: ptr<function, array<f32, 128>>,
) {
    // Port recall_score.wgsl body.
    // habituated[base + d] → (*in_habituated)[d]
    // similarities[s_base + i] → (*out_similarities)[i]
    // pattern_buf reads stay global
}
```

**brain_recall_topk**: Read `recall_topk.wgsl` (53 lines).

```wgsl
fn brain_recall_topk(
    tid: u32,
    in_out_similarities: ptr<function, array<f32, 128>>,
    out_recall: ptr<function, array<f32, 17>>,
) {
    // Port recall_topk.wgsl body.
    // similarities[s_base + i] → (*in_out_similarities)[i]  (reads AND modifies: marks -3.0)
    // recall_buf[r_base + i] → (*out_recall)[i]
    // pattern_buf metadata updates stay global
}
```

**brain_predict_and_act**: Read `predict_and_act.wgsl` (362 lines — the largest).

```wgsl
fn brain_predict_and_act(
    tid: u32,
    in_habituated: ptr<function, array<f32, 32>>,
    in_homeo: ptr<function, array<f32, 6>>,
    in_recall: ptr<function, array<f32, 17>>,
) {
    // Port predict_and_act.wgsl body.
    // habituated[base + d] → (*in_habituated)[d]
    // homeo_out[h_base + i] → (*in_homeo)[i]
    // recall_buf[r_base + i] → (*in_recall)[i]
    // Writes to decision_buf and brain_state stay global
    // Writes to history_buf stay global
}
```

**brain_learn_and_store**: Read `learn_and_store.wgsl` (185 lines).

```wgsl
fn brain_learn_and_store(
    tid: u32,
    in_features: ptr<function, array<f32, 217>>,
    in_habituated: ptr<function, array<f32, 32>>,
    in_homeo: ptr<function, array<f32, 6>>,
) {
    // Port learn_and_store.wgsl body.
    // features[base + f] → (*in_features)[f]
    // habituated[base + d] → (*in_habituated)[d]
    // homeo_out[h_base + i] → (*in_homeo)[i]
    // Reads from decision_buf stay global
    // Writes to brain_state, pattern_buf stay global
}
```

**CRITICAL**: When porting each shader, carefully map every buffer read/write:
- If the original reads from a transient buffer (features_buf, encoded_buf, habituated_buf, homeo_out_buf, similarities_buf, recall_buf), replace with the corresponding `ptr<function, ...>` parameter.
- If the original reads/writes to a persistent buffer (brain_state, pattern_buf, history_buf, decision_buf, sensory_buf), keep as global buffer access.
- The `brain_config` uniform reads stay the same.

- [ ] **Step 2: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/phase_brain.wgsl
git commit -m "feat: add mega-kernel brain phase — 7 passes merged with local arrays"
```

---

### Task 6: Entry Point + Rust Module

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/mega_tick.wgsl`
- Create: `crates/xagent-brain/src/gpu_mega_kernel.rs`
- Modify: `crates/xagent-brain/src/lib.rs`

**Depends on:** Tasks 1-5 (all shader fragments must exist)

- [ ] **Step 1: Create mega_tick.wgsl entry point**

```wgsl
// GPU Mega-Kernel Entry Point
// Runs N simulation ticks in a single dispatch.
// All physics, vision, and brain phases execute inside the tick loop
// with barrier synchronization between phases.

@compute @workgroup_size(256)
fn mega_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);
    let start_tick = wc_u32(WC_TICK);
    let ticks_to_run = wc_u32(WC_TICKS_TO_RUN);

    for (var t = 0u; t < ticks_to_run; t++) {
        let tick = start_tick + t;

        // Phase 0: Clear grids
        phase_clear(tid);
        storageBarrier(); workgroupBarrier();

        // Phase 1: Food grid build
        phase_food_grid_build(tid);
        storageBarrier(); workgroupBarrier();

        // Phase 2: Physics
        if (tid < agent_count) { phase_physics(tid, tick); }
        storageBarrier(); workgroupBarrier();

        // Phase 3: Death/Respawn
        if (tid < agent_count) { phase_death_respawn(tid, tick); }
        storageBarrier(); workgroupBarrier();

        // Phase 4: Food detect
        if (tid < agent_count) { phase_food_detect(tid); }
        storageBarrier(); workgroupBarrier();

        // Phase 5: Food respawn
        phase_food_respawn(tid, tick);
        storageBarrier(); workgroupBarrier();

        // Phase 6: Agent grid build
        if (tid < agent_count) { phase_agent_grid_build(tid); }
        storageBarrier(); workgroupBarrier();

        // Phases 7-9: Collision x3
        for (var c = 0u; c < 3u; c++) {
            if (tid < agent_count) { phase_collision_accumulate(tid); }
            storageBarrier(); workgroupBarrier();
            if (tid < agent_count) { phase_collision_apply(tid); }
            storageBarrier(); workgroupBarrier();
        }

        // Phase 10: Vision + Brain (every 4th tick)
        if (tick % 4u == 0u && tid < agent_count) {
            phase_vision(tid);
            phase_brain(tid);
        }
        storageBarrier(); workgroupBarrier();
    }
}
```

- [ ] **Step 2: Create gpu_mega_kernel.rs**

Read `crates/xagent-brain/src/gpu_brain.rs` and `crates/xagent-brain/src/gpu_physics.rs` to understand how they create buffers, bind groups, and pipelines. The mega-kernel reuses ALL existing buffers from both structs and adds a single pipeline + bind group.

```rust
//! GPU Mega-Kernel: single-dispatch simulation engine.
//!
//! Runs N simulation ticks in one `dispatch_workgroups(1,1,1)` call.
//! Replaces the per-tick CPU encoding loop with a GPU-internal tick loop.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use xagent_shared::{BrainConfig, WorldConfig};

use crate::buffers::*;
use crate::gpu_brain::GpuBrain;
use crate::gpu_physics::GpuPhysics;

/// Composes the mega-kernel WGSL source from fragments.
fn mega_kernel_source() -> String {
    [
        include_str!("shaders/mega/common.wgsl"),
        include_str!("shaders/mega/phase_clear.wgsl"),
        include_str!("shaders/mega/phase_food_grid.wgsl"),
        include_str!("shaders/mega/phase_physics.wgsl"),
        include_str!("shaders/mega/phase_death.wgsl"),
        include_str!("shaders/mega/phase_food_detect.wgsl"),
        include_str!("shaders/mega/phase_food_respawn.wgsl"),
        include_str!("shaders/mega/phase_agent_grid.wgsl"),
        include_str!("shaders/mega/phase_collision.wgsl"),
        include_str!("shaders/mega/phase_vision.wgsl"),
        include_str!("shaders/mega/phase_brain.wgsl"),
        include_str!("shaders/mega/mega_tick.wgsl"),
    ]
    .join("\n")
}

pub struct GpuMegaKernel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    agent_count: u32,
    food_count: usize,

    // Shared buffers (owned, all phases access these)
    agent_phys_buf: wgpu::Buffer,
    decision_buf: wgpu::Buffer,
    heightmap_buf: wgpu::Buffer,
    biome_buf: wgpu::Buffer,
    world_config_buf: wgpu::Buffer,
    food_state_buf: wgpu::Buffer,
    food_flags_buf: wgpu::Buffer,
    food_grid_buf: wgpu::Buffer,
    agent_grid_buf: wgpu::Buffer,
    collision_scratch_buf: wgpu::Buffer,
    sensory_buf: wgpu::Buffer,
    brain_state_buf: wgpu::Buffer,
    pattern_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    brain_config_buf: wgpu::Buffer,

    // Pipeline
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Async state readback (double-buffered)
    state_staging: [wgpu::Buffer; 2],
    staging_idx: usize,
    state_submit_seq: u64,
    state_mapped_seq: Arc<AtomicU64>,
    state_cache: Vec<f32>,

    // Config for world_config uploads
    world_config: WorldConfig,
}
```

The `new()` constructor creates ALL buffers (same sizes/usages as current GpuBrain + GpuPhysics), creates the pipeline from `mega_kernel_source()`, and builds the bind group with all 15 storage + 2 uniform bindings.

**How to build it:** Study `GpuBrain::new()` (gpu_brain.rs ~line 72-430) and `GpuPhysics::new()` (gpu_physics.rs ~line 90-430). The mega-kernel `new()` combines both:
1. Create device/queue (same as GpuBrain::new does via wgpu adapter request)
2. Create all physics buffers (same as GpuPhysics::new)
3. Create all brain buffers (same as GpuBrain::new, but skip transient buffers: features_buf, encoded_buf, habituated_buf, homeo_out_buf, similarities_buf, recall_buf)
4. Create the mega-kernel pipeline from composed WGSL source
5. Create one bind group with all 15+2 bindings
6. Create state_staging double buffers for async readback

**Public methods to implement:**

```rust
impl GpuMegaKernel {
    pub fn new(agent_count: u32, food_count: usize, brain_config: &BrainConfig, world_config: &WorldConfig) -> Self { ... }
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    /// Upload terrain, biome, food data. Call once at world init.
    pub fn upload_world(&self, heights: &[f32], biomes: &[u32],
        food_pos: &[(f32,f32,f32)], food_consumed: &[bool], food_timers: &[f32]) { ... }

    /// Upload agent initial positions. Call at generation start.
    pub fn upload_agents(&self, agents: &[(glam::Vec3, f32, f32, usize, usize)]) { ... }

    /// Upload world config uniform. Call once per batch.
    pub fn upload_world_config(&self, start_tick: u64, ticks_to_run: u32) { ... }

    /// Dispatch N ticks + encode async state readback. Non-blocking.
    pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) {
        self.upload_world_config(start_tick, ticks_to_run);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Encode async state readback
        let phys_size = (self.agent_count as u64) * (PHYS_STRIDE as u64) * 4;
        let widx = self.staging_idx;
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[widx], 0, phys_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        // Map staging buffer
        let seq = self.state_submit_seq + 1;
        self.state_submit_seq = seq;
        let flag = self.state_mapped_seq.clone();
        self.state_staging[widx].slice(..phys_size).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() { flag.store(seq, Ordering::Release); }
        });
        self.staging_idx = 1 - self.staging_idx;
    }

    /// Non-blocking: try to collect readback state. Returns true if new data available.
    pub fn try_collect_state(&mut self) -> bool {
        self.device.poll(wgpu::Maintain::Poll);
        let ridx = 1 - self.staging_idx;
        if self.state_mapped_seq.load(Ordering::Acquire) < self.state_submit_seq { return false; }
        let n = self.agent_count as usize;
        let phys_size = (n * PHYS_STRIDE * 4) as u64;
        let slice = self.state_staging[ridx].slice(..phys_size);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        self.state_cache.clear();
        self.state_cache.extend_from_slice(floats);
        drop(data);
        self.state_staging[ridx].unmap();
        true
    }

    /// Access the cached physics state (from last successful try_collect_state).
    pub fn cached_state(&self) -> &[f32] { &self.state_cache }

    /// Blocking: read full state. Use at generation boundaries.
    pub fn read_full_state_blocking(&mut self) -> &[f32] {
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        self.try_collect_state();
        &self.state_cache
    }

    /// Read one agent's brain state (for evolution inheritance).
    /// Ported from GpuBrain::read_agent_state.
    pub fn read_agent_state(&self, agent_idx: u32) -> AgentBrainState { ... }

    /// Write one agent's brain state (for evolution inheritance).
    /// Ported from GpuBrain::write_agent_state.
    pub fn write_agent_state(&self, agent_idx: u32, state: &AgentBrainState) { ... }
}
```

Port `read_agent_state` and `write_agent_state` from `GpuBrain` (gpu_brain.rs ~lines 760-900). These do blocking readback/write of brain_state_buf, pattern_buf, and history_buf for one agent. Same logic, just accessing `self.brain_state_buf` etc. directly.

- [ ] **Step 3: Add module to lib.rs**

In `crates/xagent-brain/src/lib.rs`, after `pub mod gpu_physics;` (line 10), add:

```rust
pub mod gpu_mega_kernel;
```

And add the public re-export:

```rust
pub use gpu_mega_kernel::GpuMegaKernel;
```

- [ ] **Step 4: Compile check**

Run: `cargo check -p xagent-brain`

Fix any compilation errors. Common issues:
- Missing buffer size calculations (check GpuBrain::new and GpuPhysics::new for exact formulas)
- Bind group layout mismatches (binding indices must match common.wgsl declarations exactly)
- WGSL compilation errors from the composed shader (naga validation failures)

If WGSL fails to compile, check for:
- Duplicate function or constant names across fragments
- Missing helper functions referenced by phase code
- Type mismatches in `ptr<function, ...>` parameters

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/mega_tick.wgsl \
        crates/xagent-brain/src/gpu_mega_kernel.rs \
        crates/xagent-brain/src/lib.rs
git commit -m "feat: add GpuMegaKernel — single-dispatch simulation engine"
```

---

### Task 7: Integration — main.rs + bench.rs

**Files:**
- Modify: `crates/xagent-sandbox/src/main.rs`
- Modify: `crates/xagent-sandbox/src/bench.rs`

**Depends on:** Task 6

- [ ] **Step 1: Add GpuMegaKernel field to App struct**

In `crates/xagent-sandbox/src/main.rs`, in the App struct (around line 307-310), add:

```rust
gpu_mega_kernel: Option<xagent_brain::GpuMegaKernel>,
```

Initialize to `None` in `App::new()` (around line 436).

- [ ] **Step 2: Create ensure_mega_kernel method**

Model this on the existing `ensure_gpu_brain` (lines 468-512). The new method:

```rust
fn ensure_mega_kernel(&mut self) {
    if self.gpu_mega_kernel.is_some() { return; }
    let world = self.world.as_ref().expect("world must exist");
    let agent_count = self.agents.len();
    let brain_config = &self.agents[0].brain_config;
    let food_count = world.food_items.len();

    let mut mk = xagent_brain::GpuMegaKernel::new(
        agent_count as u32, food_count, brain_config, &world.config,
    );

    // Upload world data
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32,f32,f32)> = world.food_items.iter()
        .map(|f| (f.position.x, f.position.y, f.position.z)).collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    mk.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agents
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = self.agents.iter()
        .map(|a| (a.body.body.position, a.body.body.internal.max_energy,
                   a.body.body.internal.max_integrity,
                   a.brain_config.memory_capacity, a.brain_config.processing_slots))
        .collect();
    mk.upload_agents(&agent_data);

    self.gpu_mega_kernel = Some(mk);
}
```

- [ ] **Step 3: Replace the tick loop**

Replace the simulation tick region (`main.rs:1262-1430`) with the mega-kernel batch dispatch pattern:

```rust
// ── simulation ticks (mega-kernel batched) ──────────
if !self.paused {
    self.sim_accumulator += dt * self.speed_multiplier as f32;
    let max_ticks = max_ticks_per_frame(self.speed_multiplier, self.render_3d);

    self.sim_accumulator = self.sim_accumulator.min(SIM_DT * max_ticks as f32);
    let ticks_to_run = (self.sim_accumulator / SIM_DT) as u32;

    if ticks_to_run > 0 {
        self.ensure_mega_kernel();

        if let Some(ref mut mk) = self.gpu_mega_kernel {
            // Dispatch the batch
            mk.dispatch_batch(self.tick, ticks_to_run);

            // Update bookkeeping
            self.tick += ticks_to_run as u64;
            self.tps_tick_count += ticks_to_run as u64;
            self.sim_accumulator -= SIM_DT * ticks_to_run as f32;

            if let Some(gov) = &mut self.governor {
                for _ in 0..ticks_to_run {
                    gov.tick();
                }
            }

            // Try to collect state from previous batch (non-blocking)
            if mk.try_collect_state() {
                let state = mk.cached_state();
                for i in 0..self.agents.len() {
                    let base = i * PHYS_STRIDE;
                    if base + P_FACING_Z >= state.len() { break; }
                    let a = &mut self.agents[i];
                    a.body.body.position = glam::Vec3::new(
                        state[base + P_POS_X], state[base + P_POS_Y], state[base + P_POS_Z]);
                    a.body.body.alive = state[base + P_ALIVE] > 0.5;
                    a.body.yaw = state[base + P_YAW];
                    a.body.body.internal.energy = state[base + P_ENERGY];
                    a.body.body.internal.integrity = state[base + P_INTEGRITY];
                    a.body.body.internal.max_energy = state[base + P_MAX_ENERGY];
                    a.body.body.internal.max_integrity = state[base + P_MAX_INTEGRITY];
                    a.body.body.velocity = glam::Vec3::new(
                        state[base + P_VEL_X], state[base + P_VEL_Y], state[base + P_VEL_Z]);
                    a.food_consumed = state[base + P_FOOD_COUNT] as u32;
                    a.total_ticks_alive = state[base + P_TICKS_ALIVE] as u64;
                    a.body.body.facing = glam::Vec3::new(
                        state[base + P_FACING_X], state[base + P_FACING_Y], state[base + P_FACING_Z]);
                }
            }

            // Heatmap + trail recording
            if let Some(world) = &self.world {
                for agent in &mut self.agents {
                    if agent.body.body.alive {
                        agent.record_heatmap(world.config.world_size);
                        agent.record_trail();
                    }
                }
            }

            self.food_dirty = true;
        }
    }
}
```

- [ ] **Step 4: Update evolution boundary**

In the generation boundary code (around line 700-758), replace `gpu_brain` usage with `gpu_mega_kernel`:

- Line 716: `self.gpu_brain.as_mut().map(|gb| gb.read_agent_state(a.brain_idx))` → `self.gpu_mega_kernel.as_ref().map(|mk| mk.read_agent_state(a.brain_idx as u32))`
- Lines 731-732: `self.gpu_brain = None; self.gpu_physics = None;` → `self.gpu_mega_kernel = None;`
- Line 733: `self.ensure_gpu_brain();` → `self.ensure_mega_kernel();`
- Lines 741, 745: `gpu_brain.write_agent_state(...)` → `gpu_mega_kernel.write_agent_state(...)`

Note: In the mega-kernel, `brain_idx == agent_index` (1:1 mapping, no indirection). Verify `a.brain_idx` is equal to the agent's index. If it's a separate field, the mega-kernel needs to maintain this mapping or the field should be set to the agent index.

- [ ] **Step 5: Update bench.rs**

Rewrite `crates/xagent-sandbox/src/bench.rs` to use `GpuMegaKernel`:

```rust
use xagent_brain::GpuMegaKernel;

pub fn run_bench(
    brain: BrainConfig,
    world_config: WorldConfig,
    agent_count: usize,
    total_ticks: u64,
) -> BenchResult {
    println!("[bench] Using GpuMegaKernel ({} agents)", agent_count);

    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let mut mk = GpuMegaKernel::new(
        agent_count as u32, food_count, &brain, &world_config,
    );

    // Upload world data
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32,f32,f32)> = world.food_items.iter()
        .map(|f| (f.position.x, f.position.y, f.position.z)).collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    mk.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agents
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            let pos = world.safe_spawn_position();
            (pos, 100.0, 100.0, brain.memory_capacity, brain.processing_slots)
        })
        .collect();
    mk.upload_agents(&agent_data);

    let start = std::time::Instant::now();

    // Single dispatch for all ticks
    mk.dispatch_batch(0, total_ticks as u32);
    let state = mk.read_full_state_blocking();

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let ticks_per_sec = total_ticks as f64 / elapsed_secs;

    let final_positions: Vec<[f32; 3]> = (0..agent_count)
        .map(|i| {
            let base = i * PHYS_STRIDE;
            [state[base + P_POS_X], state[base + P_POS_Y], state[base + P_POS_Z]]
        })
        .collect();

    BenchResult { total_ticks, agent_count, elapsed_secs, ticks_per_sec, final_positions }
}
```

- [ ] **Step 6: Compile check**

Run: `cargo check -p xagent-sandbox`

Fix compilation errors. Key areas:
- Import paths for `GpuMegaKernel`
- Field access patterns (the App struct may reference `gpu_brain` or `gpu_physics` in other places — search for all usages and update)
- The `brain_idx` vs agent index mapping

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs crates/xagent-sandbox/src/bench.rs
git commit -m "feat: wire GpuMegaKernel into main loop and benchmark"
```

---

### Task 8: Validation + Cleanup

**Files:**
- All files from previous tasks
- Old shader files to eventually remove

**Depends on:** Task 7

- [ ] **Step 1: Run the benchmark**

```bash
cargo run -p xagent-sandbox --release -- --bench --agents 10 --ticks 10000
```

Check:
- Does it complete without panics?
- What TPS does it report? (Target: >>600, ideally 10K+)
- Do the final positions look reasonable? (Not all zeros, not NaN)

If it panics in WGSL (validation error), the error message will indicate which line/expression failed. Fix the WGSL and re-run.

- [ ] **Step 2: Run the full app**

```bash
cargo run -p xagent-sandbox --release
```

Verify:
- Agents move and eat food
- UI renders updated positions
- Speed multiplier works (agents fast-forward at higher speeds)
- Deaths trigger respawn (agents don't disappear permanently)
- Evolution advances between generations

- [ ] **Step 3: Run existing tests**

```bash
cargo test -p xagent-sandbox
```

Fix any failing tests. Tests that directly use `GpuBrain` or `GpuPhysics` will need updates if those types changed. The mega-kernel doesn't replace them entirely — they may still be used by tests.

- [ ] **Step 4: Clean up dead code**

After validating everything works, remove old dispatch code that's no longer used:

In `main.rs`:
- Remove `ensure_gpu_brain` method (replaced by `ensure_mega_kernel`)
- Remove the old `gpu_brain` and `gpu_physics` fields from App struct if nothing else uses them
- Remove old tick loop code (now replaced)
- Remove old death handling code

Do NOT remove `GpuBrain` or `GpuPhysics` structs yet — they may be used by tests or as reference. Mark them with a `#[deprecated]` attribute if desired.

Do NOT remove old shader files yet — keep them as reference until the mega-kernel is validated in production.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "refactor: remove old per-tick dispatch loop, wire mega-kernel throughout"
```

- [ ] **Step 6: Benchmark comparison**

Run the benchmark with different agent counts and tick counts:

```bash
cargo run -p xagent-sandbox --release -- --bench --agents 10 --ticks 50000
cargo run -p xagent-sandbox --release -- --bench --agents 50 --ticks 10000
cargo run -p xagent-sandbox --release -- --bench --agents 100 --ticks 5000
```

Record TPS for each. Compare against the ~600 TPS baseline for 10 agents.
