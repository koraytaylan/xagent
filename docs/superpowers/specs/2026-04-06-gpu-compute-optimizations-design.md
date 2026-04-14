# GPU Compute Optimizations Design

## Goal

Reduce per-frame GPU latency for the brain compute pipeline through five targeted optimizations: parallel top-K selection, SoA pattern buffer layout, async compute overlap, indirect dispatch, and subgroup intrinsics. The primary bottleneck is per-workgroup serial time in the brain shader at low agent counts (~10).

## Architecture

In the current main-loop code path, each cycle dispatches the per-agent fused kernel together with global and vision work. Standalone physics/brain pipeline dispatches are used mainly for remainder handling rather than the primary per-cycle path. At 10 agents, GPU occupancy is low — the critical path is how fast a single brain workgroup completes.

The brain shader has 7 cooperative passes. Two are serial bottlenecks:
- **Pass 5 (top-K)**: 2,048 serial comparisons on thread 0 (16 rounds × 128 candidates)
- **Pass 6 (credit assignment)**: ~3,200 ops on thread 0 with sequential history loop

This spec addresses Pass 5 directly (optimizations 1, 5) and improves memory throughput for Pass 4 (optimization 2). Optimizations 3 and 4 reduce dispatch overhead and prepare the architecture for scaling.

## Tech Stack

- wgpu 24, WGSL compute shaders
- Rust (gpu_kernel.rs, buffers.rs)
- Apple Silicon / NVIDIA / AMD GPU targets
- New wgpu feature: `Features::SUBGROUP` (optimization 5)

---

## Optimization 1: Parallel Top-K via Bitonic Sort

### Current State

`coop_recall_topk` in `brain_tick.wgsl` uses a greedy serial scan:

```wgsl
for (var k: u32 = 0u; k < RECALL_K; k = k + 1u) {       // 16 iterations
    var best_idx: u32 = 0u;
    var best_sim: f32 = -3.0;
    for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {  // 128 iterations
        let sim = s_similarities[j];
        if (sim > best_sim) { best_sim = sim; best_idx = j; }
    }
    if (best_sim <= -1.5) { break; }
    s_recall[k] = f32(best_idx);
    // ... mark pattern as accessed, set sim to -3.0
}
```

Thread 0 does all 2,048 comparisons. Threads 1–255 wait at the barrier.

### Design

Replace with a full bitonic sort of 128 elements in shared memory, then read the top 16 from the sorted tail.

**New shared memory**:
```wgsl
var<workgroup> s_sort_idx: array<u32, 128>;  // tracks pattern index through sort
```

`s_similarities` is reused as the sort key array (already workgroup-scoped, 128 elements).

**Algorithm**:
1. Threads 0–127 initialize: `s_sort_idx[tid] = tid`
2. Bitonic sort: 7 stages, 28 total passes. Each pass:
   - 64 threads compute partner index from `(stage, pass, tid)`
   - Compare-and-swap `(s_similarities[i], s_sort_idx[i])` with `(s_similarities[j], s_sort_idx[j])`
   - `workgroupBarrier()` after each pass
3. Sort direction: descending (largest at index 0)
4. Thread 0 reads `s_sort_idx[0..RECALL_K]` and writes to `s_recall`, performs early termination check (`s_similarities[k] <= -1.5`), updates pattern metadata (last-accessed tick, activation count)

**Bitonic index math** (per pass):
```wgsl
let block_size = 1u << (stage + 1u - pass);
let half = block_size >> 1u;
let group = tid / half;
let local = tid % half;
let i = group * block_size + local;
let j = i + half;
let descending = ((i >> (stage + 1u)) & 1u) == 0u;
```

**Shared memory cost**: +512 bytes (`s_sort_idx` = 128 × u32). Total brain workgroup shared memory: ~2.5 KB (well within 16 KB limit).

**Barrier count**: 28 passes = 28 `workgroupBarrier()` calls (reduced to 13 with subgroup intrinsics — see optimization 5).

### Files Changed

- `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl` — rewrite `coop_recall_topk`

---

## Optimization 2: SoA Pattern Buffer for Cache Coherence

### Current State

Pattern states are AoS within each agent's pattern block:
```
pattern_buf[agent * 5251 + 0 + pattern_idx * 32 + dim]
                             ^O_PAT_STATES
```

In Pass 4 (recall_score), 128 threads read simultaneously:
```wgsl
pattern_buf[p_base + O_PAT_STATES + tid * DIM + d]  // tid=0..127, d loops 0..31
```

For a given `d` iteration, thread N reads offset `N*32 + d`. Adjacent threads are 32 floats (128 bytes) apart — every read misses the cache line loaded by the adjacent thread.

### Design

Transpose the O_PAT_STATES region from `[pattern][dim]` to `[dim][pattern]`:

```
// Before (AoS): pattern_buf[p_base + pattern * 32 + dim]
// After  (SoA): pattern_buf[p_base + dim * 128 + pattern]
```

The O_PAT_STATES region occupies offsets 0–4095 (128 × 32 = 4096 floats). Total size unchanged.

**Reads in Pass 4 become coalesced**:
```wgsl
pattern_buf[p_base + d * MEMORY_CAP + tid]  // adjacent threads read adjacent memory
```

128 threads × 4 bytes = 512 bytes per `d` iteration = 4 cache lines (was 128 cache lines).

**Writes in learn_and_store (thread 0, stores one pattern)**:
```wgsl
// Before: pattern_buf[p_base + min_idx * DIM + d] = h;
// After:  pattern_buf[p_base + d * MEMORY_CAP + min_idx] = h;
```

Write pattern changes from contiguous (stride 1) to strided (stride 128). This is worse for the single-thread write, but writes happen once per tick on thread 0 while reads happen 128-wide every tick. The read improvement dominates.

**Norm precomputation** (`O_PAT_NORMS` at offset 4096): Stays AoS. 128 threads each read `pattern_buf[p_base + O_PAT_NORMS + tid]` — already coalesced (stride 1).

**Other pattern regions unchanged** (reinforcement, motor, meta, active flags — all accessed by thread 0 or with stride 1).

### Files Changed

- `crates/xagent-brain/src/buffers.rs` — `init_pattern_memory()` uses transposed layout
- `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl` — Pass 4 (`coop_recall_score`) and Pass 7 (`coop_learn_and_store`) index transposed
- `crates/xagent-brain/src/shaders/kernel/common.wgsl` — add comment documenting SoA layout for O_PAT_STATES
- `crates/xagent-brain/src/gpu_kernel.rs` — `read_agent_state()` transposes on CPU read for state inheritance

---

## Optimization 3: Async Compute Overlap

### Current State

`dispatch_batch` in `gpu_kernel.rs`:
1. `device.poll(Maintain::Poll)` + `try_collect_staging()`
2. `upload_world_config_masked()` — writes to `world_config_buf`
3. Encode cycles (chunked by 100): physics + vision + brain per cycle
4. `queue.submit(encoder)` per chunk
5. Final submit: remainder + `copy_buffer_to_buffer` to staging + `map_async`

**Problem**: Step 2 writes to `world_config_buf` while the GPU may still be reading it from a previous submit. wgpu's `write_buffer` is a staging write applied at the next `submit()`, so the GPU-side read from the previous submit is safe. But consecutive dispatches within the same generation reuse the same buffer — no write hazard exists for the uniform buffer itself.

The actual inefficiency is that `try_collect_staging` is coupled to `dispatch_batch`. Collection and dispatch are independent operations that should be decoupled.

### Design

**A. Decouple staging collection from dispatch**:

Extract `try_collect_staging` and `device.poll` into a separate public method `poll_and_collect() -> bool`. The main loop calls it independently of dispatch:

```rust
// Main loop (main.rs):
let collected = mk.poll_and_collect();  // non-blocking
if collected { /* read state into agents */ }
let (dispatched, _) = mk.dispatch_batch(tick, ticks_to_run);  // no internal poll
```

This eliminates the poll-before-dispatch coupling. The main loop can collect results from the previous frame's dispatch while immediately submitting the next one.

**B. Double-buffer world config**:

Two `world_config_buf` instances and two corresponding bind groups. Alternate which is written/active each batch:

```rust
struct GpuKernel {
    world_config_bufs: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    active_bind_group: usize,  // 0 or 1
    // ...
}
```

`dispatch_batch` writes to `world_config_bufs[1 - active]`, then uses `bind_groups[1 - active]` for dispatches, then flips `active_bind_group`. This guarantees the GPU never reads a buffer being written to, even if submits overlap.

### Files Changed

- `crates/xagent-brain/src/gpu_kernel.rs` — split poll/collect, double-buffer world config + bind groups
- `crates/xagent-sandbox/src/main.rs` — call `poll_and_collect()` before `dispatch_batch()`

---

## Optimization 4: Indirect Dispatch

### Current State

Vision and brain dispatch sizes are set by the CPU:
```rust
pass.dispatch_workgroups(self.agent_count, 1, 1);  // vision
pass.dispatch_workgroups(self.agent_count, 1, 1);  // brain
```

`agent_count` is fixed for the lifetime of a `GpuKernel` instance.

### Design

A small compute shader writes dispatch arguments to an indirect buffer. Vision and brain use `dispatch_workgroups_indirect`.

**New buffer**:
```rust
let dispatch_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("kernel_dispatch_args"),
    size: 6 * 4,  // 2 × (x, y, z) u32 triplets: [vision_x, 1, 1, brain_x, 1, 1]
    usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
    mapped_at_creation: false,
});
```

**New shader** `phase_prepare_dispatch.wgsl` (~15 lines):
```wgsl
@group(0) @binding(X) var<storage, read_write> dispatch_args: array<u32, 6>;

@compute @workgroup_size(1)
fn prepare_dispatch() {
    let agent_count = wc_u32(WC_AGENT_COUNT);
    // Vision: 1 workgroup per agent
    dispatch_args[0] = agent_count;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
    // Brain: 1 workgroup per agent
    dispatch_args[3] = agent_count;
    dispatch_args[4] = 1u;
    dispatch_args[5] = 1u;
}
```

**Dispatch sequence becomes**:
```rust
// 1. Prepare (1 thread writes dispatch sizes)
pass.set_pipeline(&self.prepare_pipeline);
pass.dispatch_workgroups(1, 1, 1);
// barrier (implicit between compute passes)

// 2. Vision (indirect)
pass.set_pipeline(&self.vision_pipeline);
pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);

// 3. Brain (indirect)
pass.set_pipeline(&self.brain_pipeline);
pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 12);  // offset to second triplet
```

**New binding**: `dispatch_args_buf` needs a slot in the bind group layout. Add as binding 15 (storage read_write). The prepare shader reads `wconfig` (binding 4) for agent count and writes `dispatch_args` (binding 15).

**wgpu feature**: `dispatch_workgroups_indirect` requires no additional features — it's core wgpu functionality.

### Files Changed

- `crates/xagent-brain/src/shaders/kernel/phase_prepare_dispatch.wgsl` — new file
- `crates/xagent-brain/src/shaders/kernel/common.wgsl` — add dispatch_args binding 15
- `crates/xagent-brain/src/gpu_kernel.rs` — add buffer, pipeline, binding; change dispatch calls to indirect
- `crates/xagent-brain/src/buffers.rs` — document dispatch_args layout

---

## Optimization 5: Subgroup/Wave-Level Intrinsics

### Current State

wgpu device requests only `Features::PUSH_CONSTANTS`. The brain shader uses `workgroupBarrier()` for all cross-thread synchronization, even within 32-thread subgroups that could use hardware-native SIMD operations.

### Design

Enable `Features::SUBGROUP` on the wgpu device. Use WGSL subgroup operations to accelerate the first 5 stages of the bitonic sort (optimization 1), eliminating 15 of 28 barrier rounds.

**Device feature request**:
```rust
required_features: wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SUBGROUP,
```

**Shader directive**:
```wgsl
enable subgroups;
```

**Application to bitonic sort**:

Bitonic sort of 128 elements has 7 stages. Within each stage, subarrays grow from 2 to 128:
- Stages 1–5: subarrays of size 2, 4, 8, 16, 32 — all fit within a 32-wide subgroup
- Stages 6–7: subarrays of size 64, 128 — span multiple subgroups, need shared memory + barriers

For stages 1–5, replace shared memory compare-and-swap with `subgroupShuffle`:
```wgsl
// Instead of:
//   let partner_val = s_similarities[j];  // shared memory read
//   workgroupBarrier();
//   if (should_swap) { s_similarities[i] = partner_val; }
//   workgroupBarrier();

// Use:
let partner_val = subgroupShuffle(my_val, partner_lane);
if (should_swap) { my_val = partner_val; }
// No barrier needed — subgroup ops are synchronous within the subgroup
```

Stages 1–5 contain 1+2+3+4+5 = 15 passes. These become barrier-free.
Stages 6–7 contain 6+7 = 13 passes. These still use shared memory + barriers.

**Total barriers**: 28 → 13 (54% reduction).

**Subgroup size assumption**: 32 (Apple Silicon, NVIDIA). On AMD with 64-wide wavefronts, stages 1–6 could be subgroup-accelerated (21 barrier-free passes, only 7 remaining). The code works correctly regardless of subgroup size — it only uses subgroup ops for passes where the partner distance < subgroup_size.

**Fallback**: If `Features::SUBGROUP` is not available (older GPU/driver), the adapter request falls back to `PUSH_CONSTANTS` only, and the shader uses the shared-memory-only bitonic sort. The shader conditionally enables subgroups:

```wgsl
// Two shader variants composed at kernel creation time:
// Variant A (subgroup): enable subgroups; + subgroupShuffle in stages 1-5
// Variant B (fallback): shared memory + barriers for all 28 passes
```

The Rust side checks `adapter.features().contains(Features::SUBGROUP)` and composes the appropriate shader source.

### Files Changed

- `crates/xagent-brain/src/gpu_kernel.rs` — request SUBGROUP feature (with fallback), compose shader variant
- `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl` — bitonic sort uses subgroup ops for stages 1–5

---

## Dependency Order

```
1. Parallel Top-K (standalone — changes coop_recall_topk)
2. SoA Pattern Buffer (standalone — changes recall_score + learn_and_store indexing)
3. Async Compute Overlap (standalone — changes dispatch/poll structure)
4. Indirect Dispatch (standalone — adds new shader + buffer + binding)
5. Subgroup Intrinsics (depends on #1 — modifies the bitonic sort from #1)
```

1–4 are independent and can be implemented in any order. 5 must come after 1.

## Testing Strategy

- **Bitonic sort correctness**: Unit test that sorts known arrays of 128 floats, verifies top-16 match brute-force selection. Run on GPU via existing test harness (`gpu_brain::tests`).
- **SoA layout**: Existing `init_pattern_memory` test verifies buffer size. Add test that writes known pattern, reads back via transposed index, verifies values match.
- **Shader↔Rust sync**: Extend existing `shader_*_constants_match_rust` tests for new constants (O_PAT_STATES SoA comment, dispatch_args binding 15).
- **Async overlap**: Existing `dispatch_batch` integration tests verify state readback correctness — no new tests needed, just verify existing tests still pass with the decoupled poll/collect.
- **Indirect dispatch**: Verify dispatch produces identical state readback as direct dispatch. Run same tick sequence both ways, compare agent_phys buffers.
- **Subgroup fallback**: Test both shader variants (subgroup and shared-memory-only) produce identical top-K results.
- **Performance regression**: Benchmark script that runs N ticks and reports wall time. Compare before/after for each optimization.
