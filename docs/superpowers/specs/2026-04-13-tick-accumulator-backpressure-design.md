# Fix: Tick Accumulator Debt During GPU Backpressure

**Date:** 2026-04-13
**Branch:** `fix/tick-accumulator-backpressure`

## Problem

At 10x speed multiplier with `PresentMode::Immediate`, the UI freezes and updates become erratic. 1x/2x/5x work smoothly. Observed with a single agent, so hardware throughput is not the bottleneck.

## Root Cause

`PresentMode::Immediate` drives the frame rate to 200-400+ FPS. The simulation accumulator adds `dt * speed_multiplier` per frame and dispatches `floor(accumulator / SIM_DT)` ticks to the GPU. At 10x, every frame crosses the `SIM_DT` threshold and triggers a `dispatch_batch` call, which starts an async staging-buffer readback.

The readback pipeline is double-buffered: only 2 staging slots can be in-flight. When both are occupied, `dispatch_batch` returns `false` (GPU backpressure). Currently, the accumulator is **not drained** on backpressure — it keeps growing each frame. When a staging slot frees up, the next dispatch fires a burst of accumulated ticks, causing an irregular position jump. The cycle repeats: dispatch, dispatch, skip, skip, burst-dispatch, skip...

At 5x, `dt * 5` doesn't always cross `SIM_DT` at high FPS, so many frames skip dispatch naturally, giving staging buffers breathing room.

The accumulator cap (`SIM_DT * gpu_tick_budget * 2`) makes this worse: the budget grows aggressively (25%/frame, cap 64k), allowing the accumulator cap to reach ~2100 seconds of sim-time — far beyond what any speed multiplier needs per frame.

## Fix

Two changes in `main.rs`, both in the tick loop:

### 1. Drain accumulator on backpressure

When `dispatch_batch` returns `false`, cap the accumulator to 1 frame's worth of ticks so the next successful dispatch sends a normal-sized batch:

```rust
if dispatched {
    // ... existing budget/tick/accumulator logic
} else {
    // GPU backpressure — drop ticks rather than accumulating debt
    // that causes burst dispatches when staging frees up.
    self.sim_accumulator = self.sim_accumulator.min(SIM_DT * self.speed_multiplier as f32);
}
```

### 2. Tighten accumulator cap

Replace the budget-based cap with a speed-aware cap:

```rust
// Before:
let max_acc = SIM_DT * self.gpu_tick_budget as f32 * 2.0;

// After:
let max_acc = SIM_DT * self.speed_multiplier as f32 * 2.0;
```

This limits debt to 2 frames' worth regardless of budget. The budget still controls GPU batch size for high-speed modes (100x+), but the accumulator can't build up disproportionate debt.

## Tradeoff

At 10x + very high FPS, the simulation may run slightly below 600 TPS (dropped ticks during backpressure). Smooth visual updates over exact TPS is the right tradeoff — the user chose 10x for faster visual progression, not for precise tick counting. At 100x+ the user explicitly trades smoothness for throughput, and the larger `ticks_to_run` naturally gives staging more time between dispatches.

## Files Changed

- `crates/xagent-sandbox/src/main.rs` — tick loop accumulator logic (~lines 1582-1620)
