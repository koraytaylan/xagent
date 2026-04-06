# GPU Mega-Kernel Design Spec

## Problem

The simulation achieves ~600 ticks/second for 10 agents despite all computation being on GPU. The bottleneck is CPU↔GPU coordination overhead: 11-19 compute dispatches per tick, per-tick `write_buffer` calls, and a blocking `device.poll(Wait)` every 50 ticks for death readback. The GPU sits idle most of the time.

Target: 60,000+ ticks/second — a 100× improvement.

## Solution

Replace the CPU-driven per-tick dispatch loop with a single GPU mega-kernel that runs N ticks autonomously. One dispatch, one workgroup of 256 threads, internal tick loop with barrier synchronization between phases. CPU becomes an observer/supervisor.

## Constraints

- **Determinism**: Identical simulation results regardless of speed multiplier or batch size. Same device + same seed = same outcome.
- **Agent count**: ≤256 (single-workgroup constraint for barrier-based synchronization).
- **Food count**: ≤ practical limit of ~1000 (threads handle ceil(food_count/256) items via loop).

## Architecture

### Dispatch Pattern

```
CPU                              GPU
 │                                │
 ├─ write world_config ──────────►│  (start_tick, ticks_to_run, 96 bytes)
 ├─ dispatch(1,1,1) ────────────►│  mega_tick kernel
 │                                │    256 threads, 1 workgroup
 │   (CPU free: event loop,      │    tick loop runs N ticks internally
 │    rendering, UI)              │    all physics + vision + brain
 │                                │    death/respawn handled in-shader
 │                                │
 ├─ encode state copy ──────────►│  (after dispatch completes)
 ├─ async readback ◄─────────────┤  (non-blocking, for UI)
 ├─ dispatch next batch ────────►│  ...repeat
 │                                │
 ╰─ generation boundary:          │
    poll(Wait), read fitness,     │  (only CPU↔GPU sync in the system)
    evolve, upload new brains     │
```

Batch size: configurable, default ~2000 ticks. At 60K+ tps each batch takes ~33ms. CPU gets async readback between batches for UI updates (~30 fps of simulation state).

### Tick Loop Phases

```wgsl
@compute @workgroup_size(256)
fn mega_tick(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;

    for (var t = 0u; t < ticks_to_run; t++) {
        let tick = start_tick + t;

        // Phase 0: Clear grids
        clear_grids(tid);
        storageBarrier(); workgroupBarrier();

        // Phase 1: Food grid build
        build_food_grid(tid);
        storageBarrier(); workgroupBarrier();

        // Phase 2: Physics
        if (tid < agent_count) { physics_step(tid, tick); }
        storageBarrier(); workgroupBarrier();

        // Phase 3: Death/Respawn
        if (tid < agent_count) { handle_death_respawn(tid, tick); }
        storageBarrier(); workgroupBarrier();

        // Phase 4: Food detect
        if (tid < agent_count) { food_detect(tid); }
        storageBarrier(); workgroupBarrier();

        // Phase 5: Food respawn
        food_respawn(tid, tick);
        storageBarrier(); workgroupBarrier();

        // Phase 6: Agent grid build
        if (tid < agent_count) { build_agent_grid(tid); }
        storageBarrier(); workgroupBarrier();

        // Phase 7-9: Collision x3
        for (var c = 0u; c < 3u; c++) {
            if (tid < agent_count) { collision_accumulate(tid); }
            storageBarrier(); workgroupBarrier();
            if (tid < agent_count) { collision_apply(tid); }
            storageBarrier(); workgroupBarrier();
        }

        // Phase 10: Vision + Brain (every 4th tick)
        if (tick % 4u == 0u && tid < agent_count) {
            vision_sense(tid);   // 48 rays serialized per thread
            brain_tick(tid);     // 7 passes sequential, local arrays
        }
        storageBarrier(); workgroupBarrier();
    }
}
```

13 barrier pairs per tick (15 on brain ticks).

### Phase Details

**Phase 0 — Clear grids**: All 256 threads cooperate. Each thread zeroes a slice of food_grid, agent_grid, and collision_scratch. Grid sizes divided evenly across threads.

**Phase 1 — Food grid build**: Each thread handles `ceil(food_count/256)` food items in a loop (`for i in tid..food_count step 256`). Unconsumed food items inserted into spatial grid cells via atomicAdd.

**Phase 2 — Physics**: Threads 0..agent_count. Reads motor commands from decision_buf, integrates position/velocity, applies terrain collision, energy depletion, biome hazards, integrity regeneration, death check. Sets `alive=0, died_flag=1` on death.

**Phase 3 — Death/Respawn**: Threads 0..agent_count. If `died_flag > 0.5`:
1. Pick spawn position via GPU RNG (`pcg_hash(tick * 256 + tid + attempt)`) with biome check (reject Danger biome, up to 50 attempts).
2. Reset physics state: position, zero velocity, restore energy/integrity to max, `alive=1, died_flag=0`.
3. Reset brain state: halve reinforcement values (128 entries), zero homeostasis EMAs (6 values), reset exploration_rate to 0.5, zero action history ring buffer.

**Phase 4 — Food detect**: Threads 0..agent_count. Searches 3x3 grid neighborhood for nearest food within consume_radius. Atomic compare-exchange to claim food (race-safe). Awards energy on success.

**Phase 5 — Food respawn**: Same thread distribution as Phase 1. Decrements respawn timers for consumed food, respawns expired items at random FoodRich positions via pcg_hash RNG.

**Phase 6 — Agent grid build**: Threads 0..agent_count. Inserts alive agents into spatial grid cells via atomicAdd.

**Phases 7-9 — Collision (x3 iterations)**: Threads 0..agent_count. Accumulate phase: search 3x3 grid neighborhood, compute pairwise repulsion, write to scratch via atomicAdd (fixed-point). Apply phase: read + zero scratch, apply displacement to position.

**Phase 10 — Vision + Brain (every 4th tick)**: Threads 0..agent_count.
- **Vision**: 48 rays marched sequentially per thread (was 48 parallel threads per agent). Writes color/depth/proprioception/touch to sensory_buf.
- **Brain**: All 7 passes run sequentially. Intermediates stored in local arrays (no global buffer traffic):
  1. feature_extract: sensory_buf → local features[217]
  2. encode: features × weights → local encoded[32]
  3. habituate_homeo: encoded → local habituated[32] + local homeo_out[6]
  4. recall_score: habituated vs patterns → local similarities[128]
  5. recall_topk: similarities → local recall[17]
  6. predict_and_act: habituated + recall + homeo → decision_buf (motor commands)
  7. learn_and_store: update weights in brain_state_buf, store patterns in pattern_buf

## Buffer Layout

### Retained Buffers (unchanged layout)

| Buffer | Type | Per-Agent Size | Purpose |
|---|---|---|---|
| agent_phys_buf | storage RW | 24 f32 | Position, velocity, energy, integrity, flags |
| brain_state_buf | storage RW | 8,468 f32 | Encoder/predictor weights, habituation, homeostasis |
| pattern_buf | storage RW | 5,251 f32 | Memory patterns, reinforcement, metadata |
| history_buf | storage RW | 2,370 f32 | Action history ring buffer |
| decision_buf | storage RW | 68 f32 | Prediction + credit + motor output |
| sensory_buf | storage RW | 267 f32 | Vision + proprioception + touch |
| food_state_buf | storage RW | 4 f32/item | Food positions, respawn timers |
| food_flags_buf | storage RW | 1 u32/item | Food consumed flags |
| food_grid_buf | storage RW | grid cells | Food spatial hash grid |
| agent_grid_buf | storage RW | grid cells | Agent spatial hash grid |
| collision_scratch_buf | storage RW | 3 i32/agent | Accumulated collision forces |
| heightmap_buf | storage R | 129x129 f32 | Terrain heights |
| biome_buf | storage R | grid u32 | Biome type grid |
| brain_config_buf | uniform | 8 f32 | Distress exponent, learning rate, decay rate |

### Modified Buffer

**world_config_buf** (uniform, 24 f32): Existing fields unchanged. Two fields added in existing padding:

| Offset | Field | Change |
|---|---|---|
| 0-21 | Existing fields | Unchanged |
| 22 | ticks_to_run | NEW (was padding) |
| 23 | start_tick | NEW (was padding) |

### Eliminated Buffers

| Former Buffer | Size/Agent | Reason |
|---|---|---|
| features_buf | 217 f32 | Local array in brain_tick |
| encoded_buf | 32 f32 | Local array in brain_tick |
| habituated_buf | 32 f32 | Local array in brain_tick |
| homeo_out_buf | 6 f32 | Local array in brain_tick |
| similarities_buf | 128 f32 | Local array in brain_tick |
| recall_buf | 17 f32 | Local array in brain_tick |
| motor_staging[2] | 4 f32/agent | Motor stays on GPU (decision_buf → physics) |
| death_staging[2] | 1 f32/agent | Deaths handled in-shader |

Total local storage per agent thread: ~432 f32 = 1.7 KB. Only agent threads (≤256) allocate this. Non-agent threads skip the brain path entirely.

### Binding Summary

15 storage buffers + 2 uniforms = 17 total bindings. Well within Metal's 31-buffer limit.

## CPU Interface

### Hot Path (per batch)

```rust
fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) {
    // 1. Update world config uniform (1 write_buffer, 96 bytes)
    self.upload_world_config_with_ticks(start_tick, ticks_to_run);

    // 2. Single dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.mega_tick_pipeline);
        pass.set_bind_group(0, &self.mega_tick_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 3. Encode async state readback (for UI)
    encoder.copy_buffer_to_buffer(
        &self.agent_phys_buf, 0,
        &self.state_staging[self.staging_idx], 0,
        self.agent_count * PHYS_STRIDE * 4,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // 4. Map staging buffer (non-blocking)
    self.state_staging[self.staging_idx].map_async(Read, callback);
    self.staging_idx = 1 - self.staging_idx;
}
```

### UI Readback (per render frame)

```rust
fn try_update_ui(&mut self) {
    device.poll(Poll);  // non-blocking nudge
    if readback_ready {
        // Copy cached state to agent structs for rendering
        update_agent_positions();
    }
    // If not ready, UI shows previous data — no stall
}
```

### Evolution (per generation boundary)

```rust
fn evolve_generation(&mut self) {
    device.poll(Wait);  // ONLY sync point in the system
    let state = read_full_state_blocking();
    let fitness = extract_fitness(&state);
    let new_brains = selection_and_mutation(fitness);
    upload_brain_states(&new_brains);
    upload_agents(&initial_positions);
}
```

### Deleted CPU Code

- Per-tick encoding loop (`main.rs:1301-1365`)
- `device.poll(Wait)` inside tick loop (`main.rs:1345`)
- `flush_death_signals` + death readback + death staging buffers
- Motor staging buffers + motor readback (`try_collect`, `collect_blocking`)
- Per-tick `update_tick` write_buffer (`gpu_physics.rs:449-452`)
- `encode_brain_passes` / `submit` / sensory packing pipeline
- `encode_tick`, `encode_vision`, `encode_death_readback` per-tick calls

## Shader Source Organization

The mega-kernel is composed from WGSL fragments via Rust-side `include_str!` concatenation. Each phase remains a separate source file for editability:

```
shaders/
  mega_tick.wgsl          # entry point, tick loop, phase dispatch
  common.wgsl             # pcg_hash, grid helpers, constants, buffer declarations
  phase_clear.wgsl        # Phase 0: grid clears
  phase_food_grid.wgsl    # Phase 1: food grid build
  phase_physics.wgsl      # Phase 2: physics integration
  phase_death.wgsl        # Phase 3: death detection + respawn + brain reset
  phase_food_detect.wgsl  # Phase 4: food consumption
  phase_food_respawn.wgsl # Phase 5: food timer + respawn
  phase_agent_grid.wgsl   # Phase 6: agent grid build
  phase_collision.wgsl    # Phases 7-9: accumulate + apply
  phase_vision.wgsl       # Phase 10a: 48-ray vision (serialized)
  phase_brain.wgsl        # Phase 10b: all 7 brain passes
```

Composed at shader creation time:

```rust
let source = [
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/phase_clear.wgsl"),
    include_str!("shaders/phase_food_grid.wgsl"),
    include_str!("shaders/phase_physics.wgsl"),
    include_str!("shaders/phase_death.wgsl"),
    include_str!("shaders/phase_food_detect.wgsl"),
    include_str!("shaders/phase_food_respawn.wgsl"),
    include_str!("shaders/phase_agent_grid.wgsl"),
    include_str!("shaders/phase_collision.wgsl"),
    include_str!("shaders/phase_vision.wgsl"),
    include_str!("shaders/phase_brain.wgsl"),
    include_str!("shaders/mega_tick.wgsl"),
].join("\n");
```

The old per-phase shader files are retained for reference during migration, then deleted after validation.

## Determinism

**Guaranteed** (same device, same seed, same generation length):
- Fixed timestep (SIM_DT = 1/60) — no wall-clock dependency.
- Single workgroup — all threads execute in lockstep after each barrier. No inter-workgroup scheduling variance.
- Tick counter is `start_tick + loop_index` — computed, not stateful.
- RNG is `pcg_hash(tick * 256 + tid)` — deterministic from tick and agent ID.
- Food consumption atomics — in a single workgroup, atomic execution order between threads after a barrier is deterministic.
- Batch size and speed multiplier only affect how many ticks per dispatch. 1× dispatching 60 batches of 1000 ticks = 100× dispatching 1 batch of 60000 ticks = identical simulation results.

**Not guaranteed**:
- Cross-device: different GPU vendors may differ in FP precision for transcendentals (tanh, sin, cos).
- If agent_count or food_count exceeds 256 in the future (would need multi-workgroup, breaking single-workgroup atomic determinism guarantee).

## Behavioral Changes From Current Implementation

| Aspect | Current | New |
|---|---|---|
| Death detection latency | 50 ticks (DEATH_CHECK_INTERVAL) | 0 ticks (same-tick respawn) |
| Brain death reset | Often skipped (flush_death_signals never called in batched path) | Always runs (in-shader, immediate) |
| Vision parallelism | 48 threads per agent (parallel rays) | 48 rays serialized per thread (loop) |
| Motor command latency | 1-frame async readback | Zero (decision_buf → physics, same dispatch) |
| UI state freshness | Per-frame readback | Per-batch readback (~30 fps at default batch size) |
| Brain intermediates | Global storage buffers | Thread-local arrays (no global memory traffic) |

## Performance Estimate

| Metric | Current | Expected |
|---|---|---|
| Dispatches per tick | 11-19 | 0 (amortized across N-tick batch) |
| queue.submit per tick | 1 per 50 ticks | 1 per batch (2000 ticks) |
| device.poll(Wait) per tick | 1 per 50 ticks | 0 (only at generation boundary) |
| CPU→GPU writes per tick | 1 (write_buffer, 4 bytes) | 0 |
| Estimated tps (10 agents) | ~600 | 60,000-100,000+ |
