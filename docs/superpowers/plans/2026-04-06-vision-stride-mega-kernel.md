# Vision-Stride Fused Mega-Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 3-passes-per-brain-cycle dispatch with a fused mega-kernel that runs N brain cycles in a single GPU dispatch, achieving 60k+ brain TPS at 10 agents.

**Architecture:** Single mega-kernel dispatch (agent_count workgroups × 256 threads) loops over `vision_stride` brain cycles internally — per-agent physics, brute-force food detection, death/respawn, and 7-pass brain. A separate global pass (grid rebuild + collisions + vision raycasting) runs once per vision_stride cycles as 2–3 dispatches. CPU encodes a fixed number of passes regardless of brain cycle count.

**Tech Stack:** Rust, wgpu 24, WGSL compute shaders

---

### Task 1: Add `vision_stride` to config and world-config uniform

**Files:**
- Modify: `crates/xagent-shared/src/config.rs:11-60` (BrainConfig struct)
- Modify: `crates/xagent-brain/src/buffers.rs:203-225` (WC_* constants)
- Modify: `crates/xagent-brain/src/buffers.rs:561-594` (build_world_config)
- Modify: `crates/xagent-brain/src/shaders/mega/common.wgsl:199-222` (WGSL WC_* constants)
- Test: `crates/xagent-brain/src/buffers.rs` (existing test module)

- [ ] **Step 1: Write test for vision_stride in BrainConfig**

In `crates/xagent-shared/src/config.rs`, add a test (at the bottom, inside the existing `#[cfg(test)]` module if one exists, or create one):

```rust
#[test]
fn vision_stride_defaults_to_10() {
    let config: BrainConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(config.vision_stride, 10);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-shared vision_stride_defaults`
Expected: FAIL — `vision_stride` field doesn't exist yet.

- [ ] **Step 3: Add vision_stride field to BrainConfig**

In `crates/xagent-shared/src/config.rs`, add to `BrainConfig` struct after the `brain_tick_stride` field (line ~51):

```rust
    /// Brain cycles between global passes (grid rebuild, collisions, vision).
    /// Higher = more brain throughput, less frequent vision updates.
    /// Default 10.
    #[serde(default = "default_vision_stride")]
    pub vision_stride: u32,
```

Add the default function alongside the other defaults:

```rust
fn default_vision_stride() -> u32 { 10 }
```

Also add `vision_stride: 10,` to any preset constructors (e.g. `BrainConfig::default()` impl or `new()` methods) that exist in this file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xagent-shared vision_stride_defaults`
Expected: PASS

- [ ] **Step 5: Add WC_VISION_STRIDE constant to buffers.rs**

In `crates/xagent-brain/src/buffers.rs`, after `WC_PHASE_MASK` (line 224):

```rust
pub const WC_VISION_STRIDE: usize = 22;
```

`WORLD_CONFIG_SIZE` is already 24 (padded to 6 × vec4), so slot 22 fits without changing the size.

- [ ] **Step 6: Populate WC_VISION_STRIDE in build_world_config**

The function `build_world_config` at line 561 doesn't take `BrainConfig`. We need to add vision_stride as a parameter. Change the signature:

```rust
pub fn build_world_config(
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
    vision_stride: u32,
) -> Vec<f32> {
```

Add before the return:

```rust
    wc[WC_VISION_STRIDE] = vision_stride as f32;
```

Then update all call sites of `build_world_config` to pass vision_stride. Search with `grep -rn "build_world_config"` in the crate. The main call site is in `gpu_mega_kernel.rs` inside `upload_world_config_masked` (line ~733). Pass `self.vision_stride` there (which we'll add to the struct in Task 6).

For now, to keep things compiling, pass `10` as a literal at existing call sites. Task 6 will replace these with the actual config value.

- [ ] **Step 7: Add WC_VISION_STRIDE to common.wgsl**

In `crates/xagent-brain/src/shaders/mega/common.wgsl`, after `WC_PHASE_MASK` (line 222):

```wgsl
const WC_VISION_STRIDE: u32 = 22u;
```

- [ ] **Step 8: Run full test suite**

Run: `cargo test -p xagent-brain`
Expected: All existing tests pass. The new constant is unused in shaders for now.

- [ ] **Step 9: Commit**

```bash
git add crates/xagent-shared/src/config.rs crates/xagent-brain/src/buffers.rs crates/xagent-brain/src/shaders/mega/common.wgsl
git commit -m "feat: add vision_stride config and WC_VISION_STRIDE uniform slot"
```

---

### Task 2: Create `mega_tick.wgsl` — fused per-agent kernel shader

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/mega_tick.wgsl`
- Reference (read-only): `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl`
- Reference (read-only): `crates/xagent-brain/src/shaders/mega/phase_physics.wgsl`
- Reference (read-only): `crates/xagent-brain/src/shaders/mega/phase_death.wgsl`
- Reference (read-only): `crates/xagent-brain/src/shaders/mega/phase_food_detect.wgsl`

This is the core shader. It composes per-agent physics + brute-force food detection + death/respawn + brain into one entry point with a vision_stride cycle loop. The shader file does NOT include `common.wgsl` or the brain functions — those are concatenated at compile time in Rust (Task 5).

- [ ] **Step 1: Create mega_tick.wgsl with shared memory declarations**

Create `crates/xagent-brain/src/shaders/mega/mega_tick.wgsl`:

```wgsl
// ── Fused mega-kernel: per-agent physics + food + death + brain ─────────────
// dispatch(agent_count, 1, 1) — one workgroup per agent, 256 threads each.
// Loops over vision_stride brain cycles internally.
// Requires: common.wgsl, brain_tick.wgsl functions (concatenated by Rust).

// ── Additional shared memory for food detection ───────────────────────────
var<workgroup> s_best_food_idx: u32;
var<workgroup> s_best_food_dist: f32;

// ── Eat radius constant (sqrt of FOOD_CONSUME_RADIUS_SQ) ─────────────────
const EAT_RADIUS: f32 = 2.5;  // sqrt(6.25)

// ══════════════════════════════════════════════════════════════════════════
// Per-agent physics (extracted from phase_physics.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_physics(agent_id: u32, tick: u32) {
    let b = agent_id * PHYS_STRIDE;

    let alive = agent_phys[b + P_ALIVE];
    if alive < 0.5 { return; }

    let dt = wc_f32(WC_DT);
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);

    // Snapshot prev energy/integrity
    agent_phys[b + P_PREV_ENERGY] = agent_phys[b + P_ENERGY];
    agent_phys[b + P_PREV_INTEGRITY] = agent_phys[b + P_INTEGRITY];

    // Save last-good position/velocity for NaN recovery
    let last_pos = vec3<f32>(agent_phys[b + P_POS_X], agent_phys[b + P_POS_Y], agent_phys[b + P_POS_Z]);
    let last_vel = vec3<f32>(agent_phys[b + P_VEL_X], agent_phys[b + P_VEL_Y], agent_phys[b + P_VEL_Z]);

    // Read motor commands from decision_buf
    let dec_base = agent_id * DECISION_STRIDE;
    let motor_offset = dec_base + DIM + DIM;
    var motor_fwd = decision_buf[motor_offset];
    var motor_turn = decision_buf[motor_offset + 1u];
    var motor_strafe = decision_buf[motor_offset + 2u];

    // Sanitize motor: clamp [-1,1], NaN -> 0
    if !is_finite(motor_fwd) { motor_fwd = 0.0; }
    if !is_finite(motor_turn) { motor_turn = 0.0; }
    if !is_finite(motor_strafe) { motor_strafe = 0.0; }
    motor_fwd = clamp(motor_fwd, -1.0, 1.0);
    motor_turn = clamp(motor_turn, -1.0, 1.0);
    motor_strafe = clamp(motor_strafe, -1.0, 1.0);

    // Turning
    var yaw = agent_phys[b + P_YAW];
    let prev_yaw = yaw;
    yaw += motor_turn * TURN_SPEED * dt;
    agent_phys[b + P_YAW] = yaw;
    agent_phys[b + P_ANGULAR_VEL] = (yaw - prev_yaw) / max(dt, 1e-6);
    let facing = normalize(vec3<f32>(sin(yaw), 0.0, cos(yaw)));
    agent_phys[b + P_FACING_X] = facing.x;
    agent_phys[b + P_FACING_Y] = 0.0;
    agent_phys[b + P_FACING_Z] = facing.z;

    // Locomotion
    let right = vec3<f32>(facing.z, 0.0, -facing.x);
    var desired = facing * motor_fwd + right * motor_strafe;
    let desired_sq = dot(desired, desired);
    if desired_sq > 1.0 {
        desired = desired / sqrt(desired_sq);
    }
    agent_phys[b + P_VEL_X] = desired.x * MOVE_SPEED;
    agent_phys[b + P_VEL_Z] = desired.z * MOVE_SPEED;

    // Gravity
    agent_phys[b + P_VEL_Y] = agent_phys[b + P_VEL_Y] - GRAVITY * dt;

    // Integrate position
    var pos = vec3<f32>(
        agent_phys[b + P_POS_X] + agent_phys[b + P_VEL_X] * dt,
        agent_phys[b + P_POS_Y] + agent_phys[b + P_VEL_Y] * dt,
        agent_phys[b + P_POS_Z] + agent_phys[b + P_VEL_Z] * dt,
    );

    // Clamp to world bounds
    pos.x = clamp(pos.x, -world_half, world_half);
    pos.z = clamp(pos.z, -world_half, world_half);

    // Ground collision
    let ground = sample_height(pos.x, pos.z);
    if pos.y < ground + AGENT_HALF_HEIGHT {
        pos.y = ground + AGENT_HALF_HEIGHT;
        agent_phys[b + P_VEL_Y] = 0.0;
    }

    // NaN recovery
    if !is_finite(pos.x) || !is_finite(pos.y) || !is_finite(pos.z) {
        pos = last_pos;
        agent_phys[b + P_VEL_X] = last_vel.x;
        agent_phys[b + P_VEL_Y] = last_vel.y;
        agent_phys[b + P_VEL_Z] = last_vel.z;
    }

    agent_phys[b + P_POS_X] = pos.x;
    agent_phys[b + P_POS_Y] = pos.y;
    agent_phys[b + P_POS_Z] = pos.z;

    // Energy depletion
    let metabolic_rate = bc_f32(CFG_METABOLIC_RATE);
    let movement_mag = min(abs(motor_fwd) + abs(motor_strafe), 1.414);
    var energy = agent_phys[b + P_ENERGY];
    energy -= wc_f32(WC_ENERGY_DEPLETION) * metabolic_rate;
    energy -= movement_mag * wc_f32(WC_MOVEMENT_COST) * metabolic_rate;

    // Biome damage
    let integrity_scale = bc_f32(CFG_INTEGRITY_SCALE);
    let biome_type = sample_biome(pos.x, pos.z);
    if biome_type == BIOME_DANGER {
        agent_phys[b + P_INTEGRITY] = agent_phys[b + P_INTEGRITY] - wc_f32(WC_HAZARD_DAMAGE) * integrity_scale;
    }

    // Integrity regen when energy > 50%
    let max_e = agent_phys[b + P_MAX_ENERGY];
    var integrity = agent_phys[b + P_INTEGRITY];
    let max_i = agent_phys[b + P_MAX_INTEGRITY];
    if energy / max_e > 0.5 && integrity < max_i {
        integrity = min(integrity + wc_f32(WC_INTEGRITY_REGEN) * integrity_scale, max_i);
    }

    // Metabolic brain drain
    let mem_cap = agent_phys[b + P_MEMORY_CAP];
    let proc_slots = agent_phys[b + P_PROCESSING_SLOTS];
    energy -= (METABOLIC_BASE_COST + mem_cap * METABOLIC_MEMORY_COST + proc_slots * METABOLIC_PROCESSING_COST) * metabolic_rate;

    // Clamp and death check
    energy = max(energy, 0.0);
    integrity = max(integrity, 0.0);
    agent_phys[b + P_ENERGY] = energy;
    agent_phys[b + P_INTEGRITY] = integrity;

    if energy <= 0.0 || integrity <= 0.0 {
        agent_phys[b + P_ALIVE] = 0.0;
        agent_phys[b + P_DIED_FLAG] = 1.0;
    } else {
        agent_phys[b + P_TICKS_ALIVE] = agent_phys[b + P_TICKS_ALIVE] + 1.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brute-force food detection (replaces grid-based phase_food_detect)
// ══════════════════════════════════════════════════════════════════════════

fn agent_food_detect(agent_id: u32, tid: u32) {
    let b = agent_id * PHYS_STRIDE;

    // Skip dead agents — all threads must agree (uniform control flow)
    if agent_phys[b + P_ALIVE] < 0.5 { return; }

    let pos = vec3f(
        agent_phys[b + P_POS_X],
        agent_phys[b + P_POS_Y],
        agent_phys[b + P_POS_Z]);
    let food_count = wc_u32(WC_FOOD_COUNT);

    // Each thread scans a slice of food_state
    var local_best_idx = 0xFFFFFFFFu;
    var local_best_dist = 1e12;
    for (var f = tid; f < food_count; f += 256u) {
        if (atomicLoad(&food_flags[f]) != 0u) { continue; } // already consumed
        let fbase = f * FOOD_STATE_STRIDE;
        let fp = vec3f(
            food_state[fbase + F_POS_X],
            food_state[fbase + F_POS_Y],
            food_state[fbase + F_POS_Z]);
        let d = distance(pos, fp);
        if (d < EAT_RADIUS && d < local_best_dist) {
            local_best_dist = d;
            local_best_idx = f;
        }
    }

    // Two-phase shared-memory reduction (s_similarities/s_sort_idx are 128 elements)
    // Phase 1: first 128 threads write directly
    if (tid < 128u) {
        s_similarities[tid] = local_best_dist;
        s_sort_idx[tid] = local_best_idx;
    }
    workgroupBarrier();

    // Phase 2: second 128 threads merge into first 128 slots
    if (tid >= 128u) {
        let slot = tid - 128u;
        if (local_best_dist < s_similarities[slot]) {
            s_similarities[slot] = local_best_dist;
            s_sort_idx[slot] = local_best_idx;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist = 1e12;
        for (var i = 0u; i < 128u; i++) {
            if (s_similarities[i] < best_dist) {
                best_dist = s_similarities[i];
                best_idx = s_sort_idx[i];
            }
        }
        if (best_idx != 0xFFFFFFFFu) {
            // Atomic: claim food (prevents double-eating across workgroups)
            let result = atomicCompareExchangeWeak(&food_flags[best_idx], 0u, 1u);
            if (result.exchanged) {
                let food_energy = wc_f32(WC_FOOD_ENERGY);
                agent_phys[b + P_ENERGY] += food_energy;
                agent_phys[b + P_FOOD_COUNT] += 1.0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Death/respawn (extracted from phase_death.wgsl, single-agent)
// ══════════════════════════════════════════════════════════════════════════

fn agent_death_respawn(agent_id: u32, tick: u32) {
    let base = agent_id * PHYS_STRIDE;
    if (agent_phys[base + P_DIED_FLAG] < 0.5) { return; }

    // 1. Pick a safe spawn position
    let world_half = wc_f32(WC_WORLD_HALF_BOUND);
    var spawn_x = 0.0;
    var spawn_z = 0.0;
    var found = false;
    for (var attempt = 0u; attempt < 50u; attempt++) {
        let h = pcg_hash(tick * 256u + agent_id + attempt);
        let h2 = pcg_hash(h);
        let rx = hash_to_float(h) * 2.0 - 1.0;
        let rz = hash_to_float(h2) * 2.0 - 1.0;
        let cx = rx * world_half;
        let cz = rz * world_half;
        if (sample_biome(cx, cz) != BIOME_DANGER) {
            spawn_x = cx;
            spawn_z = cz;
            found = true;
            break;
        }
    }
    if (!found) {
        let h = pcg_hash(tick * 256u + agent_id);
        let h2 = pcg_hash(h);
        spawn_x = (hash_to_float(h) * 2.0 - 1.0) * world_half;
        spawn_z = (hash_to_float(h2) * 2.0 - 1.0) * world_half;
    }
    let spawn_y = sample_height(spawn_x, spawn_z) + AGENT_HALF_HEIGHT;

    // 2. Preserve fitness fields
    let saved_food_count   = agent_phys[base + P_FOOD_COUNT];
    let saved_ticks_alive  = agent_phys[base + P_TICKS_ALIVE];
    let saved_death_count  = agent_phys[base + P_DEATH_COUNT] + 1.0;
    let max_energy         = agent_phys[base + P_MAX_ENERGY];
    let max_integrity      = agent_phys[base + P_MAX_INTEGRITY];
    let memory_cap         = agent_phys[base + P_MEMORY_CAP];
    let processing_slots   = agent_phys[base + P_PROCESSING_SLOTS];

    // 3. Reset physics state
    for (var i = 0u; i < PHYS_STRIDE; i++) {
        agent_phys[base + i] = 0.0;
    }
    agent_phys[base + P_POS_X]           = spawn_x;
    agent_phys[base + P_POS_Y]           = spawn_y;
    agent_phys[base + P_POS_Z]           = spawn_z;
    agent_phys[base + P_FACING_Z]        = 1.0;
    agent_phys[base + P_ENERGY]          = max_energy;
    agent_phys[base + P_MAX_ENERGY]      = max_energy;
    agent_phys[base + P_INTEGRITY]       = max_integrity;
    agent_phys[base + P_MAX_INTEGRITY]   = max_integrity;
    agent_phys[base + P_PREV_ENERGY]     = max_energy;
    agent_phys[base + P_PREV_INTEGRITY]  = max_integrity;
    agent_phys[base + P_ALIVE]           = 1.0;
    agent_phys[base + P_MEMORY_CAP]      = memory_cap;
    agent_phys[base + P_PROCESSING_SLOTS] = processing_slots;
    agent_phys[base + P_FOOD_COUNT]      = saved_food_count;
    agent_phys[base + P_TICKS_ALIVE]     = saved_ticks_alive;
    agent_phys[base + P_DEATH_COUNT]     = saved_death_count;

    // 4. Reset brain state
    let brain_base = agent_id * BRAIN_STRIDE;

    let pat_base = agent_id * PATTERN_STRIDE;
    for (var i = 0u; i < MEMORY_CAP; i++) {
        pattern_buf[pat_base + O_PAT_REINF + i] *= 0.5;
    }

    for (var i = 0u; i < 6u; i++) {
        brain_state[brain_base + O_HOMEO + i] = 0.0;
    }

    brain_state[brain_base + O_EXPLORATION_RATE] = 0.5;

    for (var i = 0u; i < ACTION_HISTORY_LEN; i++) {
        brain_state[brain_base + O_FATIGUE_FWD_RING + i] = 0.0;
        brain_state[brain_base + O_FATIGUE_TURN_RING + i] = 0.0;
    }
    brain_state[brain_base + O_FATIGUE_CURSOR] = 0.0;
    brain_state[brain_base + O_FATIGUE_LEN] = 0.0;
    brain_state[brain_base + O_FATIGUE_FACTOR] = 1.0;

    for (var i = 0u; i < DIM; i++) {
        brain_state[brain_base + O_HAB_EMA + i] = 0.0;
        brain_state[brain_base + O_HAB_ATTEN + i] = 1.0;
        brain_state[brain_base + O_PREV_ENCODED + i] = 0.0;
    }

    let hist_base = agent_id * HISTORY_STRIDE;
    for (var i = 0u; i < HISTORY_STRIDE; i++) {
        history_buf[hist_base + i] = 0.0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Brain tick inner — delegates to the 7 cooperative passes from brain_tick.wgsl
// ══════════════════════════════════════════════════════════════════════════

fn brain_tick_inner(agent_id: u32, tid: u32 /* MEGA_SUBGROUP_TOPK_PARAMS */) {
    // Skip dead agents (uniform control flow — all threads agree)
    if (agent_phys[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5) { return; }

    coop_feature_extract(agent_id, tid);
    workgroupBarrier();

    coop_encode(agent_id, tid);
    workgroupBarrier();

    coop_habituate_homeo(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_recall_score(agent_id, tid);
    workgroupBarrier();

    coop_recall_topk(agent_id, tid /* MEGA_SUBGROUP_TOPK_ARGS */);
    storageBarrier(); workgroupBarrier();

    coop_predict_and_act(agent_id, tid);
    storageBarrier(); workgroupBarrier();

    coop_learn_and_store(agent_id, tid);
}

// ══════════════════════════════════════════════════════════════════════════
// Entry point
// ══════════════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn mega_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // MEGA_SUBGROUP_ENTRY_PARAMS
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    let vision_stride = wc_u32(WC_VISION_STRIDE);
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
    let start_tick = wc_u32(WC_TICK);

    for (var cycle = 0u; cycle < vision_stride; cycle++) {
        let base_tick = start_tick + cycle * stride;

        // Per-agent physics: thread 0 loops over brain_tick_stride sub-ticks
        if (tid == 0u) {
            for (var t = 0u; t < stride; t++) {
                agent_physics(agent_id, base_tick + t);
            }
        }
        workgroupBarrier();

        // Brute-force food detection: all 256 threads cooperate
        agent_food_detect(agent_id, tid);
        workgroupBarrier();

        // Death/respawn: thread 0
        if (tid == 0u) {
            agent_death_respawn(agent_id, base_tick);
        }
        workgroupBarrier();

        // Brain: all 256 threads, 7 cooperative passes
        brain_tick_inner(agent_id, tid /* MEGA_SUBGROUP_TOPK_INNER_ARGS */);
        workgroupBarrier();
    }
}
```

- [ ] **Step 2: Verify the shader file is syntactically valid by checking it compiles**

This file will be compiled as part of Task 5 (shader composition). For now, verify it's well-formed by checking no obvious syntax errors.

Run: `wc -l crates/xagent-brain/src/shaders/mega/mega_tick.wgsl`
Expected: ~300 lines, file exists.

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/mega_tick.wgsl
git commit -m "feat: add mega_tick.wgsl — fused per-agent physics+food+death+brain shader"
```

---

### Task 3: Create `global_tick.wgsl` — grid rebuild + collision shader

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/global_tick.wgsl`
- Reference (read-only): `crates/xagent-brain/src/shaders/mega/physics_tick.wgsl`

The global pass contains only the phases that need cross-agent data: grid clearing, food grid build, food respawn, agent grid build, and collision (3× accumulate + apply). Dispatched as `(1, 1, 1)` — single workgroup, same as current physics.

- [ ] **Step 1: Create global_tick.wgsl**

Create `crates/xagent-brain/src/shaders/mega/global_tick.wgsl`:

```wgsl
// ── Global pass: grid rebuild, food respawn, collisions ─────────────────────
// dispatch(1, 1, 1) — single workgroup of 256 threads.
// Runs once per vision_stride mega-kernel cycles.
// Requires: common.wgsl + phase_clear + phase_food_grid + phase_food_respawn
//           + phase_agent_grid + phase_collision (concatenated by Rust).

struct GlobalPushConstants {
    tick: u32,
    _pad: u32,
}
var<push_constant> gpc: GlobalPushConstants;

@compute @workgroup_size(256)
fn global_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);

    phase_clear(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_grid(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_respawn(tid, gpc.tick);
    storageBarrier(); workgroupBarrier();

    if (tid < agent_count) { phase_agent_grid(tid); }
    storageBarrier(); workgroupBarrier();

    for (var c = 0u; c < 3u; c++) {
        if (tid < agent_count) { phase_collision_accumulate(tid); }
        storageBarrier(); workgroupBarrier();
        if (tid < agent_count) { phase_collision_apply(tid); }
        storageBarrier(); workgroupBarrier();
    }
}
```

- [ ] **Step 2: Verify file exists**

Run: `wc -l crates/xagent-brain/src/shaders/mega/global_tick.wgsl`
Expected: ~35 lines, file exists.

- [ ] **Step 3: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/global_tick.wgsl
git commit -m "feat: add global_tick.wgsl — grid rebuild and collision pass"
```

---

### Task 4: Add `WC_BRAIN_TICK_STRIDE` to world config uniform

**Files:**
- Modify: `crates/xagent-brain/src/buffers.rs:203-225`
- Modify: `crates/xagent-brain/src/buffers.rs:561-594`
- Modify: `crates/xagent-brain/src/shaders/mega/common.wgsl:199-222`

The mega-kernel shader reads `brain_tick_stride` from the world config uniform via `wc_u32(WC_BRAIN_TICK_STRIDE)`. Currently this value only lives on the Rust side. We need a uniform slot for it.

- [ ] **Step 1: Add WC_BRAIN_TICK_STRIDE constant to buffers.rs**

In `crates/xagent-brain/src/buffers.rs`, after `WC_VISION_STRIDE` (added in Task 1):

```rust
pub const WC_BRAIN_TICK_STRIDE: usize = 23;
```

This uses the last slot in the 24-element (6 × vec4) uniform. No size change needed.

- [ ] **Step 2: Populate WC_BRAIN_TICK_STRIDE in build_world_config**

Add a `brain_tick_stride: u32` parameter to `build_world_config`:

```rust
pub fn build_world_config(
    config: &xagent_shared::WorldConfig,
    food_count: usize,
    agent_count: usize,
    tick: u64,
    ticks_to_run: u32,
    vision_stride: u32,
    brain_tick_stride: u32,
) -> Vec<f32> {
```

Add before the return:

```rust
    wc[WC_BRAIN_TICK_STRIDE] = brain_tick_stride as f32;
```

Update all call sites to pass the brain_tick_stride value (same as Task 1 for vision_stride — pass `self.brain_tick_stride`). If Task 1 used a literal, update it here too.

- [ ] **Step 3: Add WC_BRAIN_TICK_STRIDE to common.wgsl**

In `crates/xagent-brain/src/shaders/mega/common.wgsl`, after `WC_VISION_STRIDE`:

```wgsl
const WC_BRAIN_TICK_STRIDE: u32 = 23u;
```

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p xagent-brain`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-brain/src/buffers.rs crates/xagent-brain/src/shaders/mega/common.wgsl
git commit -m "feat: add WC_BRAIN_TICK_STRIDE uniform slot for mega-kernel shader"
```

---

### Task 5: Wire up mega and global pipelines in `gpu_mega_kernel.rs`

**Files:**
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs:20-66` (struct fields)
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs:340-604` (shader composition + pipeline creation)
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs:725-742` (upload_world_config_masked)

This task adds the `mega_pipeline` and `global_pipeline` to the struct, composes their shaders, creates their pipeline objects, and stores `vision_stride`. The old `physics_pipeline` and `brain_pipeline` are **kept** for now (Task 7 removes them). The dispatch loop is updated in Task 6.

- [ ] **Step 1: Add new fields to GpuMegaKernel struct**

In the struct definition (line ~20), add after `brain_pipeline`:

```rust
    mega_pipeline: wgpu::ComputePipeline,
    global_pipeline: wgpu::ComputePipeline,
    vision_stride: u32,
```

- [ ] **Step 2: Compose mega-kernel shader source**

In the `new()` function, after the brain shader composition block (after line ~478), add:

```rust
        // ── Compose mega-kernel shader ──
        // mega_tick.wgsl calls brain functions from brain_tick.wgsl.
        // brain_tick.wgsl has its own entry point `brain_tick` which we strip
        // by only including the function bodies (everything before the entry point).
        let brain_functions_src = include_str!("shaders/mega/brain_tick.wgsl");
        // Strip the entry point — take everything up to (but not including) the
        // `@compute @workgroup_size(256)` line for `fn brain_tick`.
        let brain_fn_end = brain_functions_src
            .find("@compute @workgroup_size(256)\nfn brain_tick(")
            .unwrap_or(brain_functions_src.len());
        let brain_fns_only = &brain_functions_src[..brain_fn_end];

        let mut mega_source = format!(
            "{}\n{}\n{}",
            &common_src,
            brain_fns_only,
            include_str!("shaders/mega/mega_tick.wgsl"),
        );

        // Apply subgroup intrinsics to mega shader (same markers, MEGA_ prefix)
        if has_subgroup {
            mega_source = mega_source.replace(
                "// MEGA_SUBGROUP_ENTRY_PARAMS",
                "@builtin(subgroup_invocation_id) sgid: u32,",
            );
            mega_source = mega_source.replace(
                "/* MEGA_SUBGROUP_TOPK_PARAMS */",
                ", sgid: u32",
            );
            mega_source = mega_source.replace(
                "/* MEGA_SUBGROUP_TOPK_ARGS */",
                ", sgid",
            );
            mega_source = mega_source.replace(
                "/* MEGA_SUBGROUP_TOPK_INNER_ARGS */",
                ", sgid",
            );

            // Apply same bitonic sort replacement as brain shader
            let begin_marker = "// BEGIN_BITONIC_SORT";
            let end_marker = "// END_BITONIC_SORT";
            if let (Some(begin_pos), Some(end_pos)) = (mega_source.find(begin_marker), mega_source.find(end_marker)) {
                let end_pos = end_pos + end_marker.len();
                // Reuse the same subgroup_sort string from the brain shader block
                mega_source.replace_range(begin_pos..end_pos, subgroup_sort);
            } else {
                log::error!("[GpuMegaKernel] Subgroup sort markers not found in mega shader");
            }
        } else {
            mega_source = mega_source.replace("// MEGA_SUBGROUP_ENTRY_PARAMS\n", "");
            mega_source = mega_source.replace(" /* MEGA_SUBGROUP_TOPK_PARAMS */", "");
            mega_source = mega_source.replace(" /* MEGA_SUBGROUP_TOPK_ARGS */", "");
            mega_source = mega_source.replace(" /* MEGA_SUBGROUP_TOPK_INNER_ARGS */", "");
        }
```

**Important:** The `subgroup_sort` variable used for the brain shader (line ~395-462) must be extracted into a local `let` binding BEFORE the brain shader block so it can be reused here. Move the `let subgroup_sort = r#"..."#;` declaration to before `if has_subgroup {` for the brain shader, so both blocks can reference it.

- [ ] **Step 3: Compose global shader source**

After the mega shader composition:

```rust
        // ── Compose global-pass shader ──
        let global_source = [
            &common_src,
            include_str!("shaders/mega/phase_clear.wgsl"),
            include_str!("shaders/mega/phase_food_grid.wgsl"),
            include_str!("shaders/mega/phase_food_respawn.wgsl"),
            include_str!("shaders/mega/phase_agent_grid.wgsl"),
            include_str!("shaders/mega/phase_collision.wgsl"),
            include_str!("shaders/mega/global_tick.wgsl"),
        ]
        .join("\n");
```

- [ ] **Step 4: Create mega and global pipelines**

After the existing prepare_pipeline creation (line ~604):

```rust
        // ── Global pipeline layout (has push constants for tick) ──
        let global_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("global_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        // ── Create mega-kernel pipeline (no push constants — reads from uniform) ──
        let mega_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mega_tick"),
            source: wgpu::ShaderSource::Wgsl(mega_source.into()),
        });
        let mega_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mega_tick"),
            layout: Some(&brain_layout), // no push constants
            module: &mega_module,
            entry_point: Some("mega_tick"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Create global pipeline ──
        let global_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("global_tick"),
            source: wgpu::ShaderSource::Wgsl(global_source.into()),
        });
        let global_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global_tick"),
            layout: Some(&global_layout),
            module: &global_module,
            entry_point: Some("global_tick"),
            compilation_options: Default::default(),
            cache: None,
        });
```

- [ ] **Step 5: Store new fields in the struct constructor return**

In the `GpuMegaKernel { ... }` return at the end of `new()` (around line 670), add:

```rust
            mega_pipeline,
            global_pipeline,
            vision_stride: brain_config.vision_stride,
```

- [ ] **Step 6: Update upload_world_config_masked to pass vision_stride and brain_tick_stride**

In `upload_world_config_masked` (line ~732), update the `build_world_config` call:

```rust
        let mut wc = build_world_config(
            &self.world_config,
            self.food_count,
            self.agent_count as usize,
            start_tick,
            ticks_to_run,
            self.vision_stride,
            self.brain_tick_stride,
        );
```

- [ ] **Step 7: Run compile check**

Run: `cargo check -p xagent-brain`
Expected: Compiles without errors. The new pipelines exist but aren't dispatched yet.

- [ ] **Step 8: Commit**

```bash
git add crates/xagent-brain/src/gpu_mega_kernel.rs
git commit -m "feat: wire mega_pipeline and global_pipeline in GpuMegaKernel"
```

---

### Task 6: Replace dispatch loop with mega + global dispatches

**Files:**
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs:850-948` (dispatch_batch)

Replace the per-cycle physics→vision→brain loop with mega-kernel + global pass dispatches.

- [ ] **Step 1: Rewrite dispatch_batch**

Replace the body of `dispatch_batch` (lines 850-948) with:

```rust
    pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {
        // Check if the write-target staging buffer is free.
        let widx = self.staging_idx;
        if self.staging_in_flight[widx] {
            return false;
        }

        let n = self.agent_count as usize;
        let buf_size = (n * PHYS_STRIDE * 4) as u64;

        let brain_cycles = ticks_to_run / self.brain_tick_stride;
        let mega_batches = brain_cycles / self.vision_stride;
        let remainder_cycles = brain_cycles % self.vision_stride;

        let ticks_per_mega = self.vision_stride * self.brain_tick_stride;
        let mut tick_cursor = start_tick;

        // Chunk mega-batches to avoid Metal command buffer limits
        const MEGAS_PER_CHUNK: u32 = 50;
        let total_megas = mega_batches + if remainder_cycles > 0 { 1 } else { 0 };
        let mut mega_idx = 0u32;

        while mega_idx < total_megas {
            let chunk_end = (mega_idx + MEGAS_PER_CHUNK).min(total_megas);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch_mega"),
            });

            // Prepare indirect dispatch args (once per chunk, for vision)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.prepare_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            for m in mega_idx..chunk_end {
                let is_remainder = m == mega_batches;
                let cycles_this_batch = if is_remainder { remainder_cycles } else { self.vision_stride };
                let ticks_this_batch = cycles_this_batch * self.brain_tick_stride;

                // Upload world config with current tick cursor and vision_stride for this batch
                self.upload_world_config_masked(tick_cursor, ticks_this_batch, 0x7);

                // Mega-kernel: 1 pass, cycles_this_batch brain cycles
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.mega_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
                    pass.dispatch_workgroups(self.agent_count, 1, 1);
                }

                // Global pass: grid rebuild + collisions
                {
                    let tick_for_global = tick_cursor + ticks_this_batch as u64;
                    let gpc: [u32; 2] = [tick_for_global as u32, 0];
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.global_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
                    pass.set_push_constants(0, bytemuck::cast_slice(&gpc));
                    pass.dispatch_workgroups(1, 1, 1);
                }

                // Vision pass: raycasting
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.vision_pipeline);
                    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
                    pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
                }

                tick_cursor += ticks_this_batch as u64;
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            mega_idx = chunk_end;
        }

        // Handle physics-only remainder (brain cycles that don't fill a single mega batch
        // are already handled above via remainder_cycles)
        let physics_remainder = ticks_to_run % self.brain_tick_stride;

        // Final submit: physics remainder + async state readback
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_mega_final"),
        });
        if physics_remainder > 0 {
            self.upload_world_config_masked(tick_cursor, physics_remainder, 0x1);
            let pc: [u32; 2] = [tick_cursor as u32, physics_remainder];
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.physics_pipeline);
                pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&pc));
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        // Async state readback into staging[widx]
        encoder.copy_buffer_to_buffer(&self.agent_phys_buf, 0, &self.state_staging[widx], 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        self.staging_ready[widx].store(false, Ordering::Release);
        let flag = self.staging_ready[widx].clone();
        self.state_staging[widx]
            .slice(..buf_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        self.staging_in_flight[widx] = true;
        self.staging_idx = 1 - self.staging_idx;
        self.active_config_idx = 1 - self.active_config_idx;
        true
    }
```

**Key changes from old dispatch_batch:**
- Instead of 3 passes per brain cycle (physics → vision → brain), we now do 3 passes per `vision_stride` brain cycles (mega → global → vision).
- The mega-kernel runs `vision_stride` brain cycles in a single dispatch.
- `upload_world_config_masked` is called per mega-batch (not once) since the tick changes.
- Physics remainder (ticks that don't fill a full brain cycle) still uses the old physics_pipeline.

- [ ] **Step 2: Handle the vision_stride in the world config upload**

The mega-kernel reads `WC_VISION_STRIDE` from the uniform, but we might be running a remainder batch with fewer cycles. We need to temporarily override vision_stride for remainder batches.

Add a helper method:

```rust
    fn upload_world_config_with_cycles(&self, start_tick: u64, ticks_to_run: u32, phase_mask: u32, vision_stride_override: u32) {
        let mut wc = build_world_config(
            &self.world_config,
            self.food_count,
            self.agent_count as usize,
            start_tick,
            ticks_to_run,
            vision_stride_override,
            self.brain_tick_stride,
        );
        wc[WC_PHASE_MASK] = phase_mask as f32;
        self.queue.write_buffer(&self.world_config_bufs[self.active_config_idx], 0, bytemuck::cast_slice(&wc));
    }
```

Then in dispatch_batch, replace the `self.upload_world_config_masked(tick_cursor, ticks_this_batch, 0x7)` call with:

```rust
                self.upload_world_config_with_cycles(tick_cursor, ticks_this_batch, 0x7, cycles_this_batch);
```

This ensures the GPU shader loops exactly `cycles_this_batch` times (which equals `vision_stride` for full batches and `remainder_cycles` for the last batch).

- [ ] **Step 3: Run compile check**

Run: `cargo check -p xagent-brain`
Expected: Compiles. The old dispatch code is replaced.

- [ ] **Step 4: Run tests**

Run: `cargo test -p xagent-brain`
Expected: All existing tests pass.

- [ ] **Step 5: Run the sandbox and verify it works**

Run: `cargo run -p xagent-sandbox --release`

At 1x speed: agents should move, eat, die, and respawn normally.
At 1000x speed: TPS should be significantly higher than 5.5k (target: 20k-60k depending on GPU).

Check the diagnostic log output for:
- Dispatches happening (not 100% skipped)
- Brain TPS in the thousands
- No GPU errors or panics

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-brain/src/gpu_mega_kernel.rs
git commit -m "feat: replace per-cycle dispatch with mega+global dispatch loop"
```

---

### Task 7: Clean up old dispatch code and diagnostic logging

**Files:**
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs` (remove dispatch_batch_masked if unused)
- Modify: `crates/xagent-sandbox/src/main.rs` (remove diagnostic logging fields/code)

- [ ] **Step 1: Remove diagnostic logging from main.rs**

Remove the diagnostic fields added during performance investigation (lines ~297-301):
- `diag_dispatch_count`
- `diag_skip_count`
- `diag_wall_sum_ms`
- `diag_budget_sum`
- `diag_last_print`

Remove the diagnostic logging block (lines ~1331-1354) that prints every 2 seconds.

- [ ] **Step 2: Evaluate dispatch_batch_masked**

Check if `dispatch_batch_masked` is still called anywhere. If only used for profiling and not called in production code, leave it but mark it `#[allow(dead_code)]`. If it's actively used, keep it and update it to use the new mega dispatch pattern too.

Run: `grep -rn "dispatch_batch_masked" crates/`

- [ ] **Step 3: Run tests**

Run: `cargo test -p xagent-sandbox`
Expected: All tests pass.

- [ ] **Step 4: Run the sandbox end-to-end**

Run: `cargo run -p xagent-sandbox --release`
Verify: normal operation at 1x and 1000x speed, no panics.

- [ ] **Step 5: Commit**

```bash
git add crates/xagent-sandbox/src/main.rs crates/xagent-brain/src/gpu_mega_kernel.rs
git commit -m "fix: remove diagnostic logging and clean up old dispatch code"
```
