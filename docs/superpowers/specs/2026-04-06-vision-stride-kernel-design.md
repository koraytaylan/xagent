# Vision-Stride Fused Kernel Design

## Goal

Achieve 60k+ brain ticks/sec at 10 agents by eliminating GPU compute pass boundaries. Replace the current 3-passes-per-cycle dispatch loop with a fused kernel that runs N brain cycles in a single GPU dispatch, combined with a less-frequent global pass for inter-agent physics and vision.

## Problem

The current architecture dispatches 3 separate compute passes per brain cycle (physics → vision → brain). Each pass boundary costs ~242µs of Metal GPU overhead (kernel launch + barrier). At 10 agents, shader compute is trivial — the overhead dominates. This caps throughput at ~5,500 TPS regardless of shader optimizations.

The CPU must encode each pass via wgpu API calls (~16µs each). At high budgets this becomes a secondary bottleneck: 3000 passes = 48ms of CPU encoding time.

## Architecture

Two-phase dispatch system:

**Phase 1 — Fused kernel** (1 dispatch, `agent_count` workgroups, 256 threads each):
Loops over `vision_stride` brain cycles internally. Each iteration runs per-agent physics (movement, energy, heightmap clamping), brute-force food detection, death/respawn, and the full 7-pass brain. All synchronization via `workgroupBarrier()` within each workgroup.

**Phase 2 — Global pass** (2-3 dispatches, runs once per `vision_stride` fused kernel calls):
Rebuilds spatial grids, handles collisions and food respawn, then runs vision raycasting. Writes fresh `sensory_buf` for the next fused kernel batch.

Per `vision_stride` brain cycles: 3 GPU passes total (1 kernel + 2 global/vision). CPU encodes a fixed number of passes regardless of brain cycle count.

## Tech Stack

- wgpu 24, WGSL compute shaders
- Rust (gpu_kernel.rs, buffers.rs)
- Existing shader phases recomposed into new pipelines

---

## Phase Split: Per-Agent vs Global

Current physics has 8 phases. They split cleanly:

### Per-Agent (move to fused kernel)

| Phase | What it does | Why per-agent |
|-------|-------------|---------------|
| `phase_physics` | Position, velocity, energy, heightmap clamping | Only reads/writes own agent's `agent_phys` |
| `phase_death_respawn` | Detect death, reset brain/pattern/history, respawn | Only reads/writes own agent's state |
| `phase_food_detect` | Find and consume nearest food | Converted to brute-force scan (see below) |

### Global (stay in global pass)

| Phase | What it does | Why global |
|-------|-------------|-----------|
| `phase_clear` | Zero grids and collision scratch | Covers all grid cells |
| `phase_food_grid` | Build food spatial index | Iterates all food items |
| `phase_food_respawn` | Respawn consumed food | Iterates all food items |
| `phase_agent_grid` | Build agent spatial index | Needs all agent positions |
| `phase_collision` (3× accumulate + apply) | Pair collision detection and resolution | Reads other agents via grid |

### Food Detection: Grid → Brute-Force

`phase_food_detect` currently reads `food_grid` (a global spatial index). In the fused kernel, `food_grid` is stale between global passes. Replace with a brute-force cooperative scan:

- 256 threads per workgroup scan `food_state` (327 items at food_density=0.005, world_size=256)
- Each thread checks ~2 items for proximity to its agent
- Shared-memory reduction finds the closest food within eat radius
- Atomic `food_flags[idx] = 1` prevents double-eating across workgroups

Cost: ~327 reads per agent per tick. Negligible on GPU with 256 threads.

### Consequences of Deferred Global Pass

Between global passes (up to `vision_stride` brain cycles):

- **Collisions deferred**: Agents may briefly overlap. At 10 agents on a 256-size world, overlaps are rare and resolve at the next global pass.
- **Vision stale**: Brain reacts to cached sensory data. At `vision_stride=10` and tick_rate=30, vision updates every ~0.33 sec. Biologically plausible — visual processing is slower than motor reflexes.
- **Food respawn deferred**: Consumed food stays gone for up to `vision_stride` brain cycles before respawning. Food timers are already slow (regen=0.005), so this is negligible.

---

## Fused Kernel Shader

### Entry Point

```wgsl
@compute @workgroup_size(256)
fn kernel_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // SUBGROUP_ENTRY_PARAMS (for subgroup intrinsics, if available)
) {
    let agent_id = wgid.x;
    let tid = lid.x;
    let vision_stride = wc_u32(WC_VISION_STRIDE);
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);

    for (var cycle = 0u; cycle < vision_stride; cycle++) {
        let base_tick = start_tick + cycle * stride;

        // Per-agent physics: thread 0, loops over brain_tick_stride sub-ticks
        if (tid == 0u) {
            for (var t = 0u; t < stride; t++) {
                agent_physics(agent_id, base_tick + t);
            }
        }

        // Brute-force food detect: all 256 threads cooperate
        agent_food_detect(agent_id, tid);
        workgroupBarrier();

        // Death/respawn: thread 0
        if (tid == 0u) {
            agent_death_respawn(agent_id, base_tick);
        }
        workgroupBarrier();

        // Brain: all 256 threads, 7 cooperative passes
        // (identical to current brain_tick internals)
        brain_tick_inner(agent_id, tid);
        workgroupBarrier();
    }
}
```

### Shared Memory

Reuses brain's existing shared memory (~2.5 KB). Per-agent physics uses only registers (one agent, trivial state). Brute-force food detection needs one shared variable for the reduction result:

```wgsl
var<workgroup> s_best_food_idx: u32;   // +4 bytes
var<workgroup> s_best_food_dist: f32;  // +4 bytes
```

Total shared memory: ~2.5 KB + 8 bytes. Well within the 16 KB limit.

### Per-Agent Physics

`agent_physics` is extracted from the current `phase_physics.wgsl`. It handles one agent:

- Read `decision_buf[agent_id]` for motor commands (forward, turn, strafe)
- Update velocity from motor commands and drag
- Integrate position: `pos += vel * dt`
- Clamp to heightmap
- Decay energy by move cost
- Update facing direction from turn command

No inter-agent reads. No grid reads. No shared memory.

### Brute-Force Food Detection

`agent_food_detect` replaces the grid-based `phase_food_detect`:

```wgsl
fn agent_food_detect(agent_id: u32, tid: u32) {
    let pos = vec3f(
        agent_phys[agent_id * PHYS_STRIDE + P_POS_X],
        agent_phys[agent_id * PHYS_STRIDE + P_POS_Y],
        agent_phys[agent_id * PHYS_STRIDE + P_POS_Z]);
    let food_count = wc_u32(WC_FOOD_COUNT);

    // Each thread scans a slice of food_state
    var local_best_idx = 0xFFFFFFFFu;
    var local_best_dist = 1e12;
    for (var f = tid; f < food_count; f += 256u) {
        if (food_flags[f] != 0u) { continue; } // already consumed
        let fp = vec3f(
            food_state[f * FOOD_STATE_STRIDE + 0],
            food_state[f * FOOD_STATE_STRIDE + 1],
            food_state[f * FOOD_STATE_STRIDE + 2]);
        let d = distance(pos, fp);
        if (d < EAT_RADIUS && d < local_best_dist) {
            local_best_dist = d;
            local_best_idx = f;
        }
    }

    // Shared-memory reduction (thread 0 collects)
    s_similarities[tid] = local_best_dist;  // reuse existing shared array
    s_sort_idx[tid] = local_best_idx;       // reuse existing shared array
    workgroupBarrier();

    if (tid == 0u) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist = 1e12;
        for (var i = 0u; i < 256u; i++) {
            if (s_similarities[i] < best_dist) {
                best_dist = s_similarities[i];
                best_idx = s_sort_idx[i];
            }
        }
        if (best_idx != 0xFFFFFFFFu) {
            // Atomic: mark food consumed (prevents double-eating)
            food_flags[best_idx] = 1u;
            // Credit agent
            let food_val = wc_f32(WC_FOOD_VALUE);
            agent_phys[agent_id * PHYS_STRIDE + P_ENERGY] += food_val;
            agent_phys[agent_id * PHYS_STRIDE + P_FOOD_COUNT] += 1.0;
        }
    }
}
```

Reuses `s_similarities` and `s_sort_idx` (not in use during food detection phase). No additional shared memory needed.

### Death/Respawn

`agent_death_respawn` is extracted from `phase_death_respawn.wgsl`. Handles one agent:

- Check if energy <= 0 or integrity <= 0
- If dead: increment death count, reset position (random via RNG), reset energy/integrity to max, zero brain_state/pattern_buf/history_buf
- Uses same RNG seeding as current implementation (tick-based)

### Brain Tick Inner

`brain_tick_inner` is the current `brain_tick.wgsl` internals (passes 1-7) extracted into a callable function. No changes to brain logic — same feature extraction, encoding, habituation, recall, prediction, learning.

---

## Global Pass Shader

### Structure

A stripped-down version of `physics_tick.wgsl` containing only the global phases:

```wgsl
@compute @workgroup_size(256)
fn global_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let agent_count = wc_u32(WC_AGENT_COUNT);

    phase_clear(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_grid(tid);
    storageBarrier(); workgroupBarrier();

    phase_food_respawn(tid, tick);
    storageBarrier(); workgroupBarrier();

    phase_agent_grid(tid, agent_count);
    storageBarrier(); workgroupBarrier();

    for (var i = 0u; i < 3u; i++) {
        phase_collision_accumulate(tid, agent_count);
        storageBarrier(); workgroupBarrier();
        phase_collision_apply(tid, agent_count);
        storageBarrier(); workgroupBarrier();
    }
}
```

Dispatched as `dispatch_workgroups(1, 1, 1)` — same as current physics.

### Vision

`vision_tick.wgsl` is unchanged. Dispatched as `dispatch_workgroups(agent_count, 1, 1)` immediately after `global_tick`. Reads updated grids and agent positions, writes `sensory_buf`.

---

## Rust-Side Dispatch

### New Dispatch Loop

```rust
pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {
    if self.staging_in_flight[self.staging_idx] { return false; }

    let brain_cycles = ticks_to_run / self.brain_tick_stride;
    let kernel_batches = brain_cycles / self.vision_stride;
    let remainder = brain_cycles % self.vision_stride;

    let mut encoder = self.device.create_command_encoder(&Default::default());

    let mut tick_cursor = start_tick;
    let ticks_per_batch = self.vision_stride * self.brain_tick_stride;

    for _ in 0..kernel_batches {
        // Fused kernel: 1 pass, vision_stride brain cycles
        self.upload_kernel_tick(tick_cursor, self.vision_stride);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernel_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        tick_cursor += ticks_per_batch as u64;

        // Global + vision: 2 passes
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.global_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.vision_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
        }
    }

    // Remainder: partial fused kernel + global + vision
    if remainder > 0 {
        self.upload_kernel_tick(tick_cursor, remainder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernel_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.global_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.vision_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
        }
    }

    // Async state readback (same as current)
    // ...

    self.queue.submit(std::iter::once(encoder.finish()));
    true
}
```

### Pipeline Creation

Three pipelines instead of four:

| Pipeline | Shader | Workgroups | Push Constants |
|----------|--------|-----------|----------------|
| `kernel_pipeline` | common + kernel_tick.wgsl | agent_count | start_tick (via world_config) |
| `global_pipeline` | common + global_tick.wgsl | 1 | tick (via push constants) |
| `vision_pipeline` | common + vision_tick.wgsl (unchanged) | agent_count (indirect) | none |

The current `physics_pipeline` and `brain_pipeline` are replaced by `kernel_pipeline` and `global_pipeline`. The old `physics_tick.wgsl` and `brain_tick.wgsl` entry points are no longer dispatched in the main loop but are retained for standalone test harnesses. The `prepare_pipeline` (indirect dispatch) is retained for vision.

### Budget Adapter

Keeps the backpressure-based approach: budget grows aggressively, staging double-buffer is the sole throttle, shrink only if CPU encoding > 50ms. No changes from current.

---

## Configuration

### New Field

`vision_stride: u32` added to `BrainConfig` in `crates/xagent-shared/src/config.rs`:

```rust
#[serde(default = "default_vision_stride")]
pub vision_stride: u32,
```

Default: 10. Runtime-configurable.

### World Config Uniform

Add `WC_VISION_STRIDE` to the world config buffer. The fused kernel reads it via `wc_u32(WC_VISION_STRIDE)`. Added alongside existing `WC_BRAIN_TICK_STRIDE`.

---

## Performance Estimate

At 10 agents, `brain_tick_stride=4`, `vision_stride=10`:

Per kernel batch (10 brain cycles = 40 physics ticks):
- Fused kernel: 1 pass, ~880µs GPU time (10 × 88µs per brain cycle at 10 parallel workgroups)
- Global pass: 1 pass, ~300µs GPU time (grid rebuild + collisions)
- Vision pass: 1 pass, ~300µs GPU time (raycasting)
- Total: 3 passes, ~1,480µs per 10 brain cycles

Brain cycles/sec: 10 / 1.48ms = 6,756 brain cycles/sec
TPS (physics ticks): 6,756 × 4 = 27,024

With double-buffered staging overlap (~1.5x): ~40k TPS.

CPU encoding: 3 passes per kernel batch. At budget=4000, stride=4, vision_stride=10: 4000/40 = 100 kernel batches × 3 = 300 passes. Encoding: 300 × 16µs = 4.8ms. Well under 50ms threshold.

To reach 60k TPS: increase `vision_stride` to 15-20 (reduces global pass frequency) or the brain shader optimizations (bitonic sort, SoA, subgroups) reduce the 88µs per brain cycle further.

---

## Files Changed

### New Files
- `crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` — fused per-agent physics + brain shader
- `crates/xagent-brain/src/shaders/kernel/global_tick.wgsl` — grid rebuild + collision shader

### Modified Files
- `crates/xagent-brain/src/gpu_kernel.rs` — new pipelines, new dispatch loop, vision_stride config
- `crates/xagent-brain/src/buffers.rs` — WC_VISION_STRIDE constant, food brute-force constants
- `crates/xagent-brain/src/shaders/kernel/common.wgsl` — WC_VISION_STRIDE slot
- `crates/xagent-shared/src/config.rs` — vision_stride field on BrainConfig
- `crates/xagent-sandbox/src/main.rs` — pass vision_stride to kernel, UI control

### Unchanged Files
- `crates/xagent-brain/src/shaders/kernel/vision_tick.wgsl` — no changes
- `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl` — logic extracted into kernel_tick, original kept for standalone tests

---

## Testing Strategy

- **Brute-force food detection correctness**: Unit test comparing brute-force scan results with grid-based results for known food/agent configurations.
- **Fused kernel state equivalence**: Run N ticks with old dispatch (3 passes/cycle) and new dispatch (kernel + global), compare `agent_phys` output. Allow small float divergence from food detection order differences.
- **Vision stride behavior**: Verify `sensory_buf` only updates every `vision_stride` brain cycles. Check that brain reads cached data between updates.
- **Double-eating prevention**: Test that two adjacent agents competing for the same food item results in exactly one consumption (atomic flag).
- **Death/respawn correctness**: Verify brain state resets on death within fused kernel, identical to current behavior.
- **Performance regression**: Benchmark script comparing TPS before/after at 10 agents, 1000x speed.

---

## Dependency Order

```
1. Extract per-agent physics functions (standalone)
2. Implement brute-force food detection (standalone)
3. Create kernel_tick.wgsl shader (depends on 1, 2)
4. Create global_tick.wgsl shader (standalone)
5. Add vision_stride config (standalone)
6. New dispatch loop in gpu_kernel.rs (depends on 3, 4, 5)
7. Remove diagnostic logging, clean up old dispatch code
```

1-2 and 4-5 are independent. 3 depends on 1+2. 6 depends on 3+4+5.
