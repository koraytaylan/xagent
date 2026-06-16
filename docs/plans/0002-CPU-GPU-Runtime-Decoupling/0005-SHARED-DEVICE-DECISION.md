# 0005 - Shared-GPU render path: decision

> Task `shared-device-render-spike`. **Done when:** either this note rejects the
> shared-device path with measured evidence, or a measured prototype shows
> enough benefit to justify a follow-up plan.

## Decision

**Reject the shared-device / compute-written-instance prototype for now.** The
measured bottleneck is GPU **compute** throughput (per-tick kernel cost), not
CPU↔render data movement. Paths B and C below optimize data movement, which the
measurements show is immaterial. Revisit only if the gating conditions at the
end of this note are met.

This satisfies the workstream's "Done when" via the reject-with-evidence branch.

## The three paths

| Path | What it is | Cost it removes | Risk / cost to build |
|---|---|---|---|
| **A. Worker-only (current)** | Worker owns a compute `wgpu::Device`; the renderer owns a separate render `Device`. State crosses as an owned `StateSnapshot` (CPU `Vec`), is applied to `Agent` caches, then `update_agent_instances` builds `InstanceData` and uploads it to the render device. | — | none (shipped) |
| **B. Shared renderer/compute device** | One `Arc<wgpu::Device>` / `Arc<wgpu::Queue>` shared by renderer and `GpuKernel` (`GpuKernel::new_with_device`). | Two-device contention on the one physical GPU. | Couples the worker's kernel lifecycle to the renderer's device (renderer is created in `resumed()`, the worker starts on evolution *Start*); cross-thread queue submission ordering; non-trivial refactor of kernel construction. |
| **C. Compute-written instance buffer** | Path B **plus** a compute pass that writes render-ready `InstanceData` into a buffer the render pass reads directly. | The per-frame CPU `InstanceData` rebuild + upload, and the GPU→CPU→GPU round-trip *for rendering positions*. | Highest. New compute pass + shared vertex/storage buffer; the step-4 hazard (blank frames while compute runs) is real. The CPU still needs agent state for UI panels, selection, evolution evaluation, and replay, so the snapshot path **cannot** be removed — only the render-instance sub-path, which is the cheapest part. |

## Measured evidence (from live runs, `RUST_LOG=debug`)

Default stride (`brain_tick_stride=10`, `kernel_batch=100`), one representative
1-second window:

```
[RENDER]     frames=+120/s   snapshots_applied=+57/s   generation_boundaries advancing
[SIM-WORKER] sim_ticks=+300/s (5x)   dispatch_calls=+30/s   telemetry_collected=+29/s
```

- **Render is decoupled and fast.** ~120 fps while the simulation ran at 5×
  (300 ticks/s). Render cadence and sim cadence are independent — the worker
  migration's goal is met. A redraw stall does not stall the sim (the worker is
  a separate thread paced by its own wall clock).
- **The throughput ceiling is GPU compute, not data movement.** At
  `stride=1` (full global+vision+brain every tick) throughput is GPU-bound at
  ~200 ticks/s; at `stride=10` it scales with the multiplier up to the GPU's
  per-tick compute ceiling. Neither limit is CPU-cache or upload related.
- **CPU agent-cache update is negligible.** `apply_state_snapshot` runs at the
  publication cadence (≤60 Hz) over ≤`MAX_AGENTS`=100 agents (~25 field reads
  each).
- **Instance upload is negligible.** `update_agent_instances` rebuilds only when
  `hud_dirty` (≤60 Hz, *not* per redraw) and uploads ≤100 × `InstanceData`
  (32 bytes) ≈ **3.2 KB**. This is not a measurable fraction of a 120 fps frame.
- **The jag we fixed was cadence, not cost.** The earlier choppiness was
  snapshot *publication frequency* (≤1 Hz when requests were tied to dispatch),
  fixed by decoupling publication to 60 Hz and capping dispatch to one
  kernel-batch — not by reducing CPU↔render data volume.

## Why reject now

1. Paths B/C target CPU↔render data movement; the data shows that path costs
   ≈3.2 KB at ≤60 Hz plus a ≤100-element cache loop — immaterial against a
   120 fps render that is already decoupled.
2. The real lever for higher simulation speed is **per-tick GPU compute cost**
   (dominated by `vision_stride`/`brain_tick_stride` and population size), which
   is a perception/semantics knob explicitly out of scope for this plan, not a
   data-movement problem a shared device would solve.
3. The CPU still needs agent state for UI, selection, evolution, and replay, so
   the snapshot path stays regardless. Path C only removes the *render-instance*
   sub-path — the smallest cost on the board — for a large jump in GPU-ownership
   complexity and the blank-frame hazard.

## When to revisit

Reopen the shared-device path if a future measurement shows data movement (not
compute) is the bottleneck — concretely:

- **`MAX_AGENTS` raised into the thousands**, where the per-frame instance
  rebuild/upload or the per-snapshot CPU cache loop becomes a material fraction
  of frame time (profile it; don't assume).
- **Observed physical-GPU contention**: render fps drops when the sim speed is
  pushed to the GPU's compute ceiling, indicating the compute device is starving
  the render device on the shared physical GPU. The current data shows render
  holding ~120 fps under load, so no contention is observed yet. If it appears,
  **Path B** (shared device, single ordered queue) is the minimal fix and is a
  prerequisite for Path C.

If reopened, ARCHITECTURE.md §0005 has the construction sketch
(`new_with_device`, shared instance storage buffer, a compute pass converting
`physics_state` + palette/alive into render instances). It warrants its own
implementation plan, not an inline addition to this one.
