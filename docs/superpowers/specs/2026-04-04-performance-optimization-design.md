# Performance Optimization: 10-100× Tick Throughput

**Date:** 2026-04-04
**Goal:** Increase simulation throughput from ~4K ticks/sec to 40K+ (10×), with a path to 400K (100×), without altering physics or brain behavior at any speed multiplier.

**Hard constraint:** Identical inputs must produce identical outputs regardless of speed. No reduced fidelity, no skipped brain ticks, no approximations. The optimization is purely mechanical — same work, faster execution.

---

## Current Bottleneck Analysis

At 10 agents, ~4K ticks/sec on current hardware:

| System | Per-Tick Cost | % of Frame Time |
|--------|--------------|-----------------|
| Vision raycast (48 rays × 50 steps × O(n) agent checks) | ~240K distance checks/tick | ~40-50% |
| Brain eval (encode + recall + predict + action) | O(recall_budget × dim) per agent | ~20-30% |
| Physics + collision | O(n) physics + O(n²) collision | ~10-15% |
| Trail/heatmap/history recording | Per-tick writes to visualization buffers | ~5-10% |
| Position snapshot rebuild | Vec clear + extend every tick | ~2-3% |

The vision raycast dominates because `march_ray_unified()` checks all agents (O(n)) at every ray step. With 10 agents, 48 rays, 50 steps: 24,000 per-agent distance checks per agent per tick.

GPU brain compute exists but is disabled at speed >16× (exactly when throughput matters most).

---

## Phase 1: CPU Hot Path Optimizations

Target: 4K → 20-40K ticks/sec (5-10×)

### 1.1 Spatial Grid for Agents

Same grid approach used for food (`world/spatial.rs`, cell size 8.0). Insert agent positions into a spatial grid each tick. Ray marching queries only the 3×3 neighborhood cells instead of iterating all agents.

- Drops vision agent-check cost from O(n_agents) per ray step to O(1) amortized
- Grid rebuild is O(n) per tick — trivial at n=10-100
- Applies to `march_ray_unified()` agent loop (senses.rs lines 168-191)

### 1.2 SIMD Ray Marching

Process 4 rays simultaneously using packed `f32x4`. Use `std::arch` intrinsics on stable Rust (`_mm_*` on x86, `vld1q_f32` on ARM NEON) with a scalar fallback behind `#[cfg(target_arch)]`. Each step does 4 terrain height lookups, 4 food checks, 4 agent checks in a single pass.

- 48 rays / 4 lanes = 12 SIMD iterations instead of 48 scalar iterations
- Terrain `height_at()` is 4 bilinear interpolations — pack the 4 grid lookups
- ~4× throughput on the hottest inner loop
- Falls back to scalar on architectures without SIMD (unlikely on modern x86/ARM)

### 1.3 Parallelize Physics

`physics::step()` per agent is independent — reads world state, writes only to the agent's own body. Move to rayon `par_iter_mut`, same pattern as brain ticks.

- Collision resolution (Phase 4) stays sequential — O(n²) but at n=10 it's 45 pair checks, negligible
- World mutation (food consumption) requires collecting consumed indices and applying after the parallel pass
- Physics is ~10-15% of frame time; parallelizing it removes it from the critical path

### 1.4 Hot/Cold Data Split at High Speed

At speed >1×, skip visualization-only work:

- `record_heatmap()`: writes 4096-cell grid every tick — skip at high speed
- `record_trail()`: pushes to `Vec<[f32; 3]>` every tick — skip at high speed
- History deques (5 × VecDeque pushes per tick): downsample to 1 push per frame at high speed
- When user drops back to 1×, recording resumes immediately

### 1.5 In-Place Position Buffer

Replace per-tick `all_positions.clear() + extend()` with a persistent buffer updated in-place after physics. Eliminates allocation churn in the tick hot path.

---

## Phase 2: GPU Vision Raycast

Target: 20-40K → 100-200K ticks/sec (additional 3-5×)

### 2.1 Terrain as GPU Texture

Upload the terrain heightmap (`Vec<f32>`, typically 65×65 to 129×129) as a 2D `r32float` texture. Ray marching in the compute shader uses hardware-accelerated texture sampling with bilinear interpolation — identical to the CPU `height_at()` but massively parallel.

### 2.2 Vision Raycast Compute Shader

Each GPU thread marches one ray for one agent:
- Workgroup: `(48, 1, 1)` — 48 rays per agent
- Dispatch: `(1, n_agents, 1)` — one workgroup row per agent
- Inputs: terrain texture, food position buffer, agent position buffer, ray parameters
- Output: RGBA color + depth per ray (identical to CPU `sample_vision_positions` output)
- Same step size (1.0), same max distance (50.0), same collision radii

480 work items for 10 agents — lightweight dispatch but eliminates the entire CPU vision bottleneck.

### 2.3 Unified GPU Pipeline

Chain vision → encode → recall in a single command buffer submission:
1. Vision raycast shader produces sensory features
2. Encode shader transforms features to representation (already exists)
3. Recall shader computes cosine similarities (already exists)
4. Only final encoded state + similarities read back to CPU

Intermediate results stay on GPU — no CPU roundtrip between stages. This eliminates 2 buffer readbacks per tick.

---

## Phase 3: GPU Brain at All Speeds

Target: 100-200K → 300-400K ticks/sec (additional 1.5-2×)

### 3.1 Batched Multi-Tick GPU Dispatch

Instead of one GPU dispatch per tick, batch N ticks into a single submission:
- At 100× speed: submit 100 ticks' worth of brain evaluations in one dispatch
- GPU processes the batch while CPU prepares the next batch
- Requires: physics results from tick T feed into sensory input for tick T+1, so the batch must include physics on GPU or interleave CPU physics with GPU brain in a pipelined fashion

Practical approach: pipeline CPU physics (tick T) with GPU brain (tick T+1). The 1-tick latency already exists in the current double-buffered design. Extend it to N-tick pipelining where the GPU stays saturated.

### 3.2 Remove Speed Cutoff

Delete the `expected_ticks <= 1` gate (main.rs line 1278). Replace with adaptive throughput measurement: if GPU-batched throughput exceeds CPU throughput for the current tick volume, use GPU. Runtime decision, not static threshold.

---

## Phase 4: Runtime Architecture & Auto-Detection

### 4.1 Capability Probe

At `SessionRunner::new()`, call `wgpu::Instance::request_adapter(power_preference: HighPerformance)`. If a compute-capable adapter is found, initialize device + queue. No CLI flag needed.

Log the selected backend to stdout:
```
[xagent] Compute backend: GpuAccelerated (Apple M2 Max)
[xagent] Compute backend: CpuOptimized (no GPU detected)
```

### 4.2 Backend Tiers

```
ComputeBackend::CpuBaseline     — scalar, sequential (fallback)
ComputeBackend::CpuOptimized    — rayon + SIMD + spatial grid
ComputeBackend::GpuAccelerated  — unified GPU pipeline
```

Session runner selects highest available. Can downshift at runtime if GPU dispatch latency exceeds CPU time for the current workload.

### 4.3 Correctness Contract

Both CPU and GPU backends implement identical tick logic. Integration tests run both on the same seed and assert identical agent states after N ticks within f32 epsilon. This is the guarantee that speed multiplier and backend selection don't affect evolutionary outcomes.

---

## Phase 5: Measurability & Success Criteria

### 5.1 Benchmark Mode

`--bench` CLI flag: headless, fixed seed, fixed agent count (10), fixed world config, runs 10,000 ticks, prints ticks/sec. No UI, no database.

### 5.2 Per-Phase Targets

| Phase | Target ticks/sec | Improvement |
|-------|-----------------|-------------|
| Baseline (current) | ~4,000 | 1× |
| Phase 1 (CPU optimizations) | 20,000-40,000 | 5-10× |
| Phase 2 (GPU vision) | 100,000-200,000 | 25-50× |
| Phase 3 (GPU brain batching) | 300,000-400,000 | 75-100× |

### 5.3 Regression Guard

Benchmark runs in CI, logs ticks/sec per commit. Soft threshold — no hard gate (hardware varies) but regressions are visible in the log.

### 5.4 Determinism Validation

Benchmark includes a determinism check: same seed → same final agent states. If CPU and GPU paths diverge, the benchmark fails hard.

---

## What This Design Does NOT Change

- Ray count (48), step size (1.0), max distance (50.0)
- Brain evaluation frequency (every tick)
- Memory recall algorithm (cosine similarity, full budget)
- Physics timestep (1/60 sec)
- Collision detection behavior
- Evolutionary selection, reproduction, mutation
- Any agent-observable quantity

The simulation is identical. Only the execution speed changes.
