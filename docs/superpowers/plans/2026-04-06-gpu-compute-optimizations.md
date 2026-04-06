# GPU Compute Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-frame GPU latency for the brain compute pipeline through five targeted optimizations.

**Architecture:** Replace the serial top-K scan with a parallel bitonic sort, transpose pattern memory to SoA layout for cache coherence, decouple GPU poll/collect from dispatch with double-buffered world config, add GPU-driven indirect dispatch, and accelerate the bitonic sort with subgroup intrinsics where available.

**Tech Stack:** Rust, wgpu 24, WGSL compute shaders, Apple Silicon / NVIDIA / AMD GPU targets.

**Spec:** `docs/superpowers/specs/2026-04-06-gpu-compute-optimizations-design.md`

**Dependency order:** Tasks 1–4 are independent. Task 5 depends on Task 1.

**Execution order:** Execute tasks in listed order (1→2→3→4→5). While tasks 1–4 are logically independent, they share `gpu_mega_kernel.rs` — later tasks assume earlier changes are in place (e.g., Task 4 references `bind_groups[idx]` from Task 3's double-buffer).

---

## File Structure

**Create:**
- `crates/xagent-brain/src/shaders/mega/phase_prepare_dispatch.wgsl` — indirect dispatch argument writer (Task 4)

**Modify:**
- `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl` — bitonic sort (Task 1), SoA indexing (Task 2), subgroup ops (Task 5)
- `crates/xagent-brain/src/shaders/mega/common.wgsl` — dispatch_args binding 15 (Task 4), SoA layout comment (Task 2)
- `crates/xagent-brain/src/gpu_mega_kernel.rs` — async overlap + double-buffer (Task 3), indirect dispatch (Task 4), subgroup feature (Task 5), SoA transpose in read_agent_state (Task 2)
- `crates/xagent-brain/src/buffers.rs` — SoA documentation (Task 2)
- `crates/xagent-sandbox/src/main.rs` — decouple poll_and_collect from dispatch (Task 3)
- `crates/xagent-brain/src/gpu_brain.rs` — update test that indexes into O_PAT_STATES with SoA layout (Task 2)

---

### Task 1: Parallel Top-K via Bitonic Sort

Replace the serial greedy scan in `coop_recall_topk` (2,048 comparisons on thread 0) with a full bitonic sort of 128 elements using 64 threads. All 256 threads participate in barriers.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl:1-13` (shared memory), `brain_tick.wgsl:189-211` (coop_recall_topk), `brain_tick.wgsl:676-677` (entry point call)
- Modify: `crates/xagent-brain/src/buffers.rs:684` (tests)

- [ ] **Step 1: Write CPU reference test for bitonic sort correctness**

Add a test in `crates/xagent-brain/src/buffers.rs` that implements bitonic sort on CPU and verifies top-K matches a greedy scan:

```rust
#[test]
fn bitonic_sort_top_k_matches_greedy_scan() {
    // Reference greedy scan (current GPU algorithm)
    fn greedy_top_k(sims: &[f32], k: usize) -> Vec<usize> {
        let mut s = sims.to_vec();
        let mut result = Vec::new();
        for _ in 0..k {
            let mut best_idx = 0;
            let mut best_val = -3.0_f32;
            for (j, &v) in s.iter().enumerate() {
                if v > best_val { best_val = v; best_idx = j; }
            }
            if best_val <= -1.5 { break; }
            result.push(best_idx);
            s[best_idx] = -3.0;
        }
        result
    }

    // Bitonic sort (same algorithm as the new GPU code)
    fn bitonic_top_k(sims: &[f32], k: usize) -> Vec<usize> {
        let n = sims.len(); // 128
        let mut vals = sims.to_vec();
        let mut idxs: Vec<u32> = (0..n as u32).collect();

        for stage in 0..7u32 {
            for step in 0..=stage {
                let block_size = 1u32 << (stage + 1 - step);
                let half = block_size >> 1;
                for tid in 0..(n as u32 / 2) {
                    let group = tid / half;
                    let local = tid % half;
                    let i = (group * block_size + local) as usize;
                    let j = i + half as usize;
                    let descending = ((i >> (stage as usize + 1)) & 1) == 0;
                    let should_swap = (descending && vals[i] < vals[j])
                        || (!descending && vals[i] > vals[j]);
                    if should_swap {
                        vals.swap(i, j);
                        idxs.swap(i, j);
                    }
                }
            }
        }

        let mut result = Vec::new();
        for i in 0..k {
            if vals[i] <= -1.5 { break; }
            result.push(idxs[i] as usize);
        }
        result
    }

    // Test with known data
    let mut sims = vec![-2.0_f32; 128];
    // Activate 20 patterns with known similarities
    sims[5] = 0.95;
    sims[10] = 0.80;
    sims[0] = 0.75;
    sims[127] = 0.70;
    sims[64] = 0.65;
    sims[33] = 0.60;
    sims[99] = 0.55;
    sims[17] = 0.50;
    sims[42] = 0.45;
    sims[88] = 0.40;
    sims[3] = 0.35;
    sims[111] = 0.30;
    sims[50] = 0.25;
    sims[77] = 0.20;
    sims[22] = 0.15;
    sims[61] = 0.10;
    sims[100] = 0.05;

    let greedy = greedy_top_k(&sims, 16);
    let bitonic = bitonic_top_k(&sims, 16);

    assert_eq!(greedy.len(), bitonic.len(),
        "top-K count mismatch: greedy={}, bitonic={}", greedy.len(), bitonic.len());
    // Both should return the same set of indices (order may differ within equal sims)
    for idx in &greedy {
        assert!(bitonic.contains(idx),
            "greedy selected index {} (sim={}) but bitonic did not. bitonic={:?}",
            idx, sims[*idx], bitonic);
    }
}
```

- [ ] **Step 2: Run test to verify it passes (algorithm validation)**

Run: `cargo test -p xagent-brain bitonic_sort_top_k_matches_greedy_scan -- --nocapture`
Expected: PASS (both algorithms produce the same top-K set)

- [ ] **Step 3: Add s_sort_idx shared memory and rewrite coop_recall_topk**

In `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl`, add shared memory after line 11 (`s_similarities`):

```wgsl
var<workgroup> s_sort_idx: array<u32, 128>;   // tracks pattern index through bitonic sort
```

Replace the entire `coop_recall_topk` function (lines 189–211) with:

```wgsl
fn coop_recall_topk(agent_id: u32, tid: u32) {
    let p_base = agent_id * PATTERN_STRIDE;
    let b_base = agent_id * BRAIN_STRIDE;
    let tick = brain_state[b_base + O_TICK_COUNT];

    // Initialize sort index: threads 0..127
    if (tid < MEMORY_CAP) {
        s_sort_idx[tid] = tid;
    }
    workgroupBarrier();

    // Bitonic sort: 7 stages, 28 total barrier passes
    // Sort s_similarities descending (largest at index 0)
    // BEGIN_BITONIC_SORT
    for (var stage: u32 = 0u; stage < 7u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < 64u) {
                let block_size = 1u << (stage + 1u - step);
                let half = block_size >> 1u;
                let group = tid / half;
                let local_id = tid % half;
                let i = group * block_size + local_id;
                let j = i + half;
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;

                let val_i = s_similarities[i];
                let val_j = s_similarities[j];
                let idx_i = s_sort_idx[i];
                let idx_j = s_sort_idx[j];

                let should_swap = (descending && val_i < val_j) || (!descending && val_i > val_j);
                if (should_swap) {
                    s_similarities[i] = val_j;
                    s_similarities[j] = val_i;
                    s_sort_idx[i] = idx_j;
                    s_sort_idx[j] = idx_i;
                }
            }
            workgroupBarrier();
        }
    }
    // END_BITONIC_SORT

    // Thread 0: extract top-K from sorted array (index 0 = largest)
    if (tid == 0u) {
        var count: u32 = 0u;
        for (var k: u32 = 0u; k < RECALL_K; k = k + 1u) {
            if (s_similarities[k] <= -1.5) { break; }
            let idx = s_sort_idx[k];
            s_recall[k] = f32(idx);
            count = count + 1u;
            // Update pattern metadata: last-accessed tick, activation count
            pattern_buf[p_base + O_PAT_META + idx * 3u + 1u] = tick;
            pattern_buf[p_base + O_PAT_META + idx * 3u + 2u] += 1.0;
        }
        for (var k: u32 = count; k < RECALL_K; k = k + 1u) { s_recall[k] = 0.0; }
        s_recall[RECALL_K] = f32(count);
    }
}
```

- [ ] **Step 4: Update entry point to pass tid to coop_recall_topk**

In the entry point section of `brain_tick.wgsl` (around line 676), change:

```wgsl
    if (tid == 0u) { coop_recall_topk(agent_id); }
```

to:

```wgsl
    coop_recall_topk(agent_id, tid);
```

All 256 threads must enter the function to participate in barriers. The `if (tid < 64u)` guard inside the function limits compare-and-swap to 64 threads while all threads hit the barriers.

- [ ] **Step 5: Verify shader compiles**

Run: `cargo check -p xagent-brain`
Expected: Compiles without errors.

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p xagent-brain`
Expected: All tests pass (including the new bitonic sort test).

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/brain_tick.wgsl crates/xagent-brain/src/buffers.rs
git commit -m "perf: parallel top-K via bitonic sort in brain shader

Replace serial greedy scan (2048 ops on thread 0) with bitonic sort
using 64 threads and 28 barrier passes. Threads 1-255 no longer idle
during top-K selection."
```

---

### Task 2: SoA Pattern Buffer for Cache Coherence

Transpose the O_PAT_STATES region in pattern memory from AoS `[pattern][dim]` to SoA `[dim][pattern]`. This makes 128-thread parallel reads in Pass 4 (recall_score) coalesced: adjacent threads read adjacent memory addresses.

**Before:** `pattern_buf[p_base + O_PAT_STATES + pattern_idx * DIM + d]` — stride 32 between adjacent threads.
**After:** `pattern_buf[p_base + d * MEMORY_CAP + pattern_idx]` — stride 1 between adjacent threads.

Note: `O_PAT_STATES = 0`, so the offset is implicit. Region size unchanged: 128 × 32 = 4096 floats.

**Files:**
- Modify: `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl` — 5 access sites
- Modify: `crates/xagent-brain/src/shaders/mega/common.wgsl` — add layout documentation comment
- Modify: `crates/xagent-brain/src/gpu_brain.rs:1079-1081` — update test indexing
- Modify: `crates/xagent-brain/src/buffers.rs` — add SoA documentation

- [ ] **Step 1: Write test verifying SoA indexing math**

Add to `crates/xagent-brain/src/buffers.rs` tests:

```rust
#[test]
fn soa_pattern_index_covers_same_region_as_aos() {
    // AoS: pattern_idx * DIM + d  (old layout)
    // SoA: d * MEMORY_CAP + pattern_idx  (new layout)
    // Both must cover exactly offsets 0..4095 within O_PAT_STATES region
    let mut aos_offsets = std::collections::HashSet::new();
    let mut soa_offsets = std::collections::HashSet::new();
    for pat in 0..MEMORY_CAP {
        for d in 0..DIM {
            aos_offsets.insert(O_PAT_STATES + pat * DIM + d);
            soa_offsets.insert(O_PAT_STATES + d * MEMORY_CAP + pat);
        }
    }
    assert_eq!(aos_offsets.len(), MEMORY_CAP * DIM);
    assert_eq!(soa_offsets.len(), MEMORY_CAP * DIM);
    assert_eq!(aos_offsets, soa_offsets, "SoA and AoS must cover identical offsets");
    // Both should span 0..4095
    assert_eq!(*aos_offsets.iter().min().unwrap(), 0);
    assert_eq!(*aos_offsets.iter().max().unwrap(), MEMORY_CAP * DIM - 1);
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test -p xagent-brain soa_pattern_index -- --nocapture`
Expected: PASS

- [ ] **Step 3: Update all O_PAT_STATES accesses in brain_tick.wgsl**

There are 5 access sites. Change each from AoS to SoA indexing:

**Site 1 — `cosine_sim_pat_s` helper (around line 30):**
```wgsl
// Before:
let p = pattern_buf[p_base + O_PAT_STATES + idx * DIM + d];
// After:
let p = pattern_buf[p_base + d * MEMORY_CAP + idx];
```

**Site 2 — `coop_recall_score` Pass 4 (around line 173):**
```wgsl
// Before:
dot += s_habituated[d] * pattern_buf[p_base + O_PAT_STATES + tid * DIM + d];
// After:
dot += s_habituated[d] * pattern_buf[p_base + d * MEMORY_CAP + tid];
```

**Site 3 — `coop_predict_and_act` Pass 6, context blend (around line 272):**
```wgsl
// Before:
s_prediction[d] += pattern_buf[p_base + O_PAT_STATES + idx * DIM + d] * w;
// After:
s_prediction[d] += pattern_buf[p_base + d * MEMORY_CAP + idx] * w;
```

**Site 4 — `coop_learn_and_store` Pass 7c, reinforcement read (around line 564):**
```wgsl
// Before:
dot_val += h * pattern_buf[p_base + O_PAT_STATES + tid * DIM + d];
// After:
dot_val += h * pattern_buf[p_base + d * MEMORY_CAP + tid];
```

**Site 5 — `coop_learn_and_store` Pass 7d, pattern store write (around line 593):**
```wgsl
// Before:
pattern_buf[p_base + O_PAT_STATES + min_idx * DIM + d] = h;
// After:
pattern_buf[p_base + d * MEMORY_CAP + min_idx] = h;
```

- [ ] **Step 4: Add SoA layout comment in common.wgsl**

In `crates/xagent-brain/src/shaders/mega/common.wgsl`, update the pattern memory offsets comment block (around line 67):

```wgsl
// ── Pattern memory offsets ──────────────────────────────────────────────
// O_PAT_STATES uses SoA (Structure-of-Arrays) layout: [dim][pattern]
// Index as: p_base + d * MEMORY_CAP + pattern_idx
// This gives coalesced reads when 128 threads each read one pattern.
// Other regions (norms, reinf, motor, meta, active) remain AoS.
```

- [ ] **Step 5: Update gpu_brain.rs test**

In `crates/xagent-brain/src/gpu_brain.rs` around line 1079, the test writes pattern state with AoS indexing. Update to SoA:

```rust
// Before:
for d in 0..DIM {
    state.patterns[O_PAT_STATES + d] = 1.0;
}
// After (SoA: dim * MEMORY_CAP + pattern_idx, slot 0):
for d in 0..DIM {
    state.patterns[O_PAT_STATES + d * MEMORY_CAP] = 1.0;
}
```

- [ ] **Step 6: Verify compilation and run tests**

Run: `cargo check -p xagent-brain && cargo test -p xagent-brain`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/brain_tick.wgsl crates/xagent-brain/src/shaders/mega/common.wgsl crates/xagent-brain/src/buffers.rs crates/xagent-brain/src/gpu_brain.rs
git commit -m "perf: SoA pattern buffer layout for cache-coalesced reads

Transpose O_PAT_STATES region from [pattern][dim] to [dim][pattern].
128-thread recall_score reads now hit 4 cache lines per dimension
iteration instead of 128."
```

---

### Task 3: Async Compute Overlap

Decouple GPU poll/collect from dispatch so the main loop can overlap result collection with the next dispatch. Double-buffer the world config uniform to prevent write-before-read hazards across overlapping submits.

**Files:**
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs` — refactor dispatch_batch, add double-buffer
- Modify: `crates/xagent-sandbox/src/main.rs:1314-1342` — call pattern change

- [ ] **Step 1: Refactor dispatch_batch — remove internal poll/collect**

In `crates/xagent-brain/src/gpu_mega_kernel.rs`, modify `dispatch_batch` (around line 678). Remove the poll and collect calls from the top of the method and change the return type:

```rust
/// Dispatch all ticks. Fully non-blocking: skips dispatch if the
/// write-target staging buffer is still in flight (GPU backpressure).
/// Returns true if dispatch was submitted.
///
/// Call `try_collect_state()` separately before this to collect
/// results from previous frames.
pub fn dispatch_batch(&mut self, start_tick: u64, ticks_to_run: u32) -> bool {
    // Check if the write-target staging buffer is free (no poll — caller does that).
    let widx = self.staging_idx;
    if self.staging_in_flight[widx] {
        return false;
    }

    // ... rest of method unchanged from "let n = ..." onward ...
    // ... but remove the final `(true, collected)` return and return just `true`
```

The body from `let n = self.agent_count as usize;` through the staging map_async and staging_idx flip stays the same. The final line changes from `(true, collected)` to `true`.

- [ ] **Step 2: Double-buffer world_config_buf and bind_group**

In `gpu_mega_kernel.rs`, change the struct fields:

```rust
// Replace:
world_config_buf: wgpu::Buffer,
// ...
bind_group: wgpu::BindGroup,

// With:
world_config_bufs: [wgpu::Buffer; 2],
// ...
bind_groups: [wgpu::BindGroup; 2],
active_config_idx: usize,
```

In `new()`, create two world_config buffers:

```rust
let world_config_bufs = [
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mega_world_config_0"),
        size: (WORLD_CONFIG_SIZE * 4) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }),
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mega_world_config_1"),
        size: (WORLD_CONFIG_SIZE * 4) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }),
];
```

Create two bind groups, identical except for the world_config buffer at binding 4:

```rust
let make_bind_group = |wc_buf: &wgpu::Buffer, label: &str| {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0,  resource: agent_phys_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1,  resource: decision_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2,  resource: heightmap_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3,  resource: biome_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4,  resource: wc_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5,  resource: food_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6,  resource: food_flags_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7,  resource: food_grid_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8,  resource: agent_grid_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9,  resource: collision_scratch_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: sensory_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: brain_state_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: pattern_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: history_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: brain_config_buf.as_entire_binding() },
        ],
    })
};
let bind_groups = [
    make_bind_group(&world_config_bufs[0], "mega_tick_bg_0"),
    make_bind_group(&world_config_bufs[1], "mega_tick_bg_1"),
];
```

Update `upload_world_config_masked` to write to the active buffer:

```rust
pub fn upload_world_config_masked(&self, start_tick: u64, ticks_to_run: u32, phase_mask: u32) {
    let mut wc = build_world_config(
        &self.world_config, self.food_count, self.agent_count as usize, start_tick, ticks_to_run,
    );
    wc[WC_PHASE_MASK] = phase_mask as f32;
    self.queue.write_buffer(&self.world_config_bufs[self.active_config_idx], 0, bytemuck::cast_slice(&wc));
}
```

Update all `&self.bind_group` references in dispatch methods to `&self.bind_groups[self.active_config_idx]`.

At the end of `dispatch_batch`, after the staging flip, also flip the config index:

```rust
self.active_config_idx = 1 - self.active_config_idx;
```

Update the Self constructor to set `active_config_idx: 0`.

- [ ] **Step 3: Update all dispatch methods for double-buffer**

Update `dispatch_batch_masked` the same way: use `&self.bind_groups[self.active_config_idx]` for all `set_bind_group` calls, and flip `active_config_idx` at the end.

Update `upload_world_config` (calls `upload_world_config_masked` — no change needed).

- [ ] **Step 4: Update main.rs dispatch call pattern**

In `crates/xagent-sandbox/src/main.rs` around line 1314, change from:

```rust
let (dispatched, state_updated) = mk.dispatch_batch(self.tick, ticks_to_run);
// ...
if state_updated || mk.try_collect_state() {
```

to:

```rust
// Collect results from previous frame (non-blocking)
let state_updated = mk.try_collect_state();
let dispatched = mk.dispatch_batch(self.tick, ticks_to_run);
// ...
if state_updated || mk.try_collect_state() {
```

The first `try_collect_state()` call collects any ready staging buffer from the previous frame before submitting new work. The second call after dispatch catches fast completions.

- [ ] **Step 5: Verify compilation and run tests**

Run: `cargo check -p xagent-sandbox && cargo test -p xagent-sandbox`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-brain/src/gpu_mega_kernel.rs crates/xagent-sandbox/src/main.rs
git commit -m "perf: async compute overlap with double-buffered world config

Decouple GPU poll/collect from dispatch so result collection can overlap
with the next submit. Double-buffer world_config uniform and bind groups
to prevent write-before-read hazards across overlapping batches."
```

---

### Task 4: Indirect Dispatch

Replace CPU-set `dispatch_workgroups(agent_count, 1, 1)` for vision and brain with GPU-driven `dispatch_workgroups_indirect`. A tiny 1-thread shader reads agent_count from the world config uniform and writes dispatch arguments to a storage buffer.

**Files:**
- Create: `crates/xagent-brain/src/shaders/mega/phase_prepare_dispatch.wgsl`
- Modify: `crates/xagent-brain/src/shaders/mega/common.wgsl` — add binding 15
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs` — add buffer, pipeline, binding, indirect dispatch
- Modify: `crates/xagent-brain/src/buffers.rs` — test for binding count

- [ ] **Step 1: Write test for binding count**

Add to `crates/xagent-brain/src/buffers.rs` tests:

```rust
#[test]
fn shader_has_16_bindings() {
    let src = include_str!("shaders/mega/common.wgsl");
    let binding_count = src.lines()
        .filter(|l| l.trim().starts_with("@group(0) @binding("))
        .count();
    assert_eq!(binding_count, 16,
        "Expected 16 bindings (0-14 existing + 15 dispatch_args), found {}", binding_count);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xagent-brain shader_has_16_bindings -- --nocapture`
Expected: FAIL — currently 15 bindings (0–14).

- [ ] **Step 3: Add dispatch_args binding to common.wgsl**

In `crates/xagent-brain/src/shaders/mega/common.wgsl`, after binding 14 (brain_config, around line 262), add:

```wgsl
@group(0) @binding(15) var<storage, read_write> dispatch_args:     array<u32, 6>;
```

- [ ] **Step 4: Create phase_prepare_dispatch.wgsl**

Create `crates/xagent-brain/src/shaders/mega/phase_prepare_dispatch.wgsl`:

```wgsl
// Phase: prepare indirect dispatch arguments.
// Reads agent_count from world config, writes dispatch triplets.
// dispatch_args layout: [vision_x, 1, 1, brain_x, 1, 1]

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

- [ ] **Step 5: Add buffer, pipeline, and binding in gpu_mega_kernel.rs**

In the struct definition, add:

```rust
dispatch_args_buf: wgpu::Buffer,
prepare_pipeline: wgpu::ComputePipeline,
```

In `new()`, create the buffer (after other buffer definitions):

```rust
let dispatch_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("mega_dispatch_args"),
    size: 6 * 4, // 2 × (x, y, z) u32 triplets
    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

Add binding 15 to the bind group layout entries array:

```rust
storage_rw_entry(15),  // dispatch_args
```

Add the dispatch_args buffer to both bind groups (binding 15):

```rust
wgpu::BindGroupEntry { binding: 15, resource: dispatch_args_buf.as_entire_binding() },
```

Compose and create the prepare pipeline:

```rust
// ── Compose prepare-dispatch shader ──
let prepare_source = [
    &common_src,
    include_str!("shaders/mega/phase_prepare_dispatch.wgsl"),
]
.join("\n");

let prepare_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("prepare_dispatch"),
    source: wgpu::ShaderSource::Wgsl(prepare_source.into()),
});
let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("prepare_dispatch"),
    layout: Some(&brain_layout), // no push constants needed
    module: &prepare_module,
    entry_point: Some("prepare_dispatch"),
    compilation_options: Default::default(),
    cache: None,
});
```

- [ ] **Step 6: Change dispatch_batch to use indirect dispatch**

In `dispatch_batch`, before the cycle loop, add the prepare dispatch (once per encoder):

```rust
// Inside the cycle loop, before the first cycle's passes:
// Prepare indirect dispatch args (once per chunk encoder)
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&self.prepare_pipeline);
    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
    pass.dispatch_workgroups(1, 1, 1);
}
```

Change vision and brain dispatches from direct to indirect:

```rust
// Vision (indirect)
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&self.vision_pipeline);
    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
    pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 0);
}

// Brain (indirect)
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&self.brain_pipeline);
    pass.set_bind_group(0, &self.bind_groups[self.active_config_idx], &[]);
    pass.dispatch_workgroups_indirect(&self.dispatch_args_buf, 12); // offset to second triplet
}
```

Apply the same changes to `dispatch_batch_masked`: add prepare dispatch, change vision/brain to indirect.

- [ ] **Step 7: Run test to verify binding count passes**

Run: `cargo test -p xagent-brain shader_has_16_bindings -- --nocapture`
Expected: PASS

- [ ] **Step 8: Verify full compilation and tests**

Run: `cargo check -p xagent-sandbox && cargo test -p xagent-brain`
Expected: All pass.

- [ ] **Step 9: Commit**

```bash
git add crates/xagent-brain/src/shaders/mega/phase_prepare_dispatch.wgsl crates/xagent-brain/src/shaders/mega/common.wgsl crates/xagent-brain/src/gpu_mega_kernel.rs crates/xagent-brain/src/buffers.rs
git commit -m "perf: GPU indirect dispatch for vision and brain pipelines

Add phase_prepare_dispatch shader that writes dispatch args from
world config. Vision and brain use dispatch_workgroups_indirect,
removing CPU-GPU sync for workgroup counts."
```

---

### Task 5: Subgroup/Wave-Level Intrinsics (depends on Task 1)

Accelerate the first 5 stages of the bitonic sort (from Task 1) using WGSL subgroup operations. Within a 32-wide subgroup, threads can exchange values via `subgroupShuffle` without shared memory barriers. This eliminates 15 of 28 barrier passes.

**Stages 0–4** (partner distance 1–16, fits in 32-wide subgroup): use `subgroupShuffle`, no barriers.
**Stages 5–6** (partner distance 32–64, spans subgroups): use shared memory + barriers (unchanged).

**Fallback:** If the adapter doesn't support `Features::SUBGROUP`, the shared-memory-only bitonic sort from Task 1 is used unchanged.

**Files:**
- Modify: `crates/xagent-brain/src/gpu_mega_kernel.rs` — request SUBGROUP feature, compose shader variant
- Modify: `crates/xagent-brain/src/shaders/mega/brain_tick.wgsl` — subgroup sort stages via string markers

- [ ] **Step 1: Add SUBGROUP feature detection in gpu_mega_kernel.rs**

In `new()`, after adapter creation (around line 152), detect subgroup support:

```rust
let has_subgroup = adapter.features().contains(wgpu::Features::SUBGROUP);
if has_subgroup {
    log::info!("[GpuMegaKernel] Subgroup support detected — enabling subgroup intrinsics for brain shader");
} else {
    log::info!("[GpuMegaKernel] No subgroup support — using shared-memory-only bitonic sort");
}
```

Change the device request to conditionally enable SUBGROUP:

```rust
let required_features = if has_subgroup {
    wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SUBGROUP
} else {
    wgpu::Features::PUSH_CONSTANTS
};

let (device, queue) = pollster::block_on(adapter.request_device(
    &wgpu::DeviceDescriptor {
        label: Some("gpu-mega-kernel"),
        required_features,
        required_limits: {
            required_limits.max_push_constant_size = 8;
            required_limits
        },
        memory_hints: wgpu::MemoryHints::default(),
    },
    None,
))
.expect("Failed to create GPU device");
```

Store the flag in the struct:

```rust
has_subgroup: bool,
```

- [ ] **Step 2: Add BEGIN/END markers to brain_tick.wgsl bitonic sort**

The bitonic sort code from Task 1 already has `// BEGIN_BITONIC_SORT` and `// END_BITONIC_SORT` markers. Verify they exist and frame the entire sort loop (stages 0-6).

Also add a `// SUBGROUP_ENTRY_PARAMS` marker in the entry point for parameter injection:

In brain_tick.wgsl, change the entry point:

```wgsl
@compute @workgroup_size(256)
fn brain_tick(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // SUBGROUP_ENTRY_PARAMS
) {
```

And change the function signature:

```wgsl
fn coop_recall_topk(agent_id: u32, tid: u32 /* SUBGROUP_TOPK_PARAMS */) {
```

And the call site:

```wgsl
    coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */);
```

- [ ] **Step 3: Compose subgroup shader variant in gpu_mega_kernel.rs**

In `new()`, when composing the brain shader (around line 342), add subgroup string replacements:

```rust
let mut brain_source = format!("{}\n{}", &common_src, include_str!("shaders/mega/brain_tick.wgsl"));

if has_subgroup {
    // Prepend enable directive
    brain_source = format!("enable subgroups;\n{}", brain_source);

    // Add subgroup builtins to entry point
    brain_source = brain_source.replace(
        "// SUBGROUP_ENTRY_PARAMS",
        "@builtin(subgroup_invocation_id) sgid: u32,",
    );

    // Add sgid to coop_recall_topk signature and call
    brain_source = brain_source.replace(
        "/* SUBGROUP_TOPK_PARAMS */",
        ", sgid: u32",
    );
    brain_source = brain_source.replace(
        "/* SUBGROUP_TOPK_ARGS */",
        ", sgid",
    );

    // Replace bitonic sort with subgroup-accelerated version
    let subgroup_sort = r#"
    // ── Subgroup-accelerated stages 0–4 (15 barrier-free passes) ──
    // Each of 128 threads holds one element in registers.
    // subgroupShuffle exchanges values within 32-wide subgroups.
    var my_val: f32 = -3.0;
    var my_idx: u32 = 0u;
    if (tid < MEMORY_CAP) {
        my_val = s_similarities[tid];
        my_idx = s_sort_idx[tid];
    }

    for (var stage: u32 = 0u; stage < 5u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < MEMORY_CAP) {
                let half = 1u << (stage - step);
                let partner_tid = tid ^ half;
                let partner_val = subgroupShuffle(my_val, sgid ^ half);
                let partner_idx = subgroupShuffle(my_idx, sgid ^ half);

                let i = min(tid, partner_tid);
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;
                let i_am_low = tid < partner_tid;
                let want_max = i_am_low == descending;
                if ((my_val < partner_val) == want_max) {
                    my_val = partner_val;
                    my_idx = partner_idx;
                }
            }
            // No barrier needed — subgroup ops are synchronous within subgroup
        }
    }

    // Write back to shared memory for stages 5–6
    if (tid < MEMORY_CAP) {
        s_similarities[tid] = my_val;
        s_sort_idx[tid] = my_idx;
    }
    workgroupBarrier();

    // ── Shared-memory stages 5–6 (13 passes with barriers) ──
    for (var stage: u32 = 5u; stage < 7u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < 64u) {
                let block_size = 1u << (stage + 1u - step);
                let half = block_size >> 1u;
                let group = tid / half;
                let local_id = tid % half;
                let i = group * block_size + local_id;
                let j = i + half;
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;

                let val_i = s_similarities[i];
                let val_j = s_similarities[j];
                let idx_i = s_sort_idx[i];
                let idx_j = s_sort_idx[j];

                let should_swap = (descending && val_i < val_j) || (!descending && val_i > val_j);
                if (should_swap) {
                    s_similarities[i] = val_j;
                    s_similarities[j] = val_i;
                    s_sort_idx[i] = idx_j;
                    s_sort_idx[j] = idx_i;
                }
            }
            workgroupBarrier();
        }
    }
"#;

    // Find and replace the bitonic sort block
    let begin_marker = "// BEGIN_BITONIC_SORT";
    let end_marker = "// END_BITONIC_SORT";
    if let (Some(begin_pos), Some(end_pos)) = (brain_source.find(begin_marker), brain_source.find(end_marker)) {
        let end_pos = end_pos + end_marker.len();
        brain_source.replace_range(begin_pos..end_pos, subgroup_sort);
    }
} else {
    // Remove placeholder comments for non-subgroup path
    brain_source = brain_source.replace("// SUBGROUP_ENTRY_PARAMS\n", "");
    brain_source = brain_source.replace(" /* SUBGROUP_TOPK_PARAMS */", "");
    brain_source = brain_source.replace(" /* SUBGROUP_TOPK_ARGS */", "");
}
```

- [ ] **Step 4: Verify shader compiles on current hardware**

Run: `cargo check -p xagent-brain`
Expected: Compiles. The shader variant selection happens at runtime, but `include_str!` and string composition happen at compile time.

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p xagent-brain && cargo test -p xagent-sandbox`
Expected: All tests pass. On hardware without SUBGROUP support, the fallback path is used.

- [ ] **Step 6: Commit**

```bash
git add crates/xagent-brain/src/gpu_mega_kernel.rs crates/xagent-brain/src/shaders/mega/brain_tick.wgsl
git commit -m "perf: subgroup intrinsics for bitonic sort stages 0-4

Use subgroupShuffle for the first 5 stages (15 passes) of the bitonic
sort, eliminating shared memory barriers. Falls back to shared-memory-
only sort when Features::SUBGROUP is not available. 28 → 13 barriers."
```
