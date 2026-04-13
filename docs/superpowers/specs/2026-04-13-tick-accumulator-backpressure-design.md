# Fix: Tick Accumulator Debt During GPU Backpressure

**Date:** 2026-04-13
**Branch:** `fix/tick-accumulator-backpressure`

## Problem

At 10x speed multiplier with `PresentMode::Immediate`, the UI freezes and updates become erratic. 1x/2x/5x work smoothly. Observed with a single agent, so hardware throughput is not the bottleneck.

## Root Cause

`PresentMode::Immediate` drives the frame rate to 200-400+ FPS. The simulation accumulator adds `dt * speed_multiplier` per frame and dispatches `floor(accumulator / SIM_DT)` ticks to the GPU. At 10x, every frame crosses the `SIM_DT` threshold and triggers a `dispatch_batch` call, which starts an async staging-buffer readback.

The readback pipeline was double-buffered: only 2 staging slots could be in-flight. When both were occupied, `dispatch_batch` returned `false` (GPU backpressure). The accumulator was **not drained** on backpressure — it kept growing each frame. When a staging slot freed up, the next dispatch fired a burst of accumulated ticks, causing an irregular position jump. The cycle repeated: dispatch, dispatch, skip, skip, burst-dispatch, skip...

At 5x, `dt * 5` doesn't always cross `SIM_DT` at high FPS, so many frames skip dispatch naturally, giving staging buffers breathing room.

## Rejected Approaches

Several accumulator-side fixes were attempted and rejected:

1. **Drain accumulator on backpressure** — capping to 1 frame's worth halved effective TPS at 10x (50% backpressure rate → ~300 TPS).
2. **`speed_tick_cap`** — capping `ticks_to_run` to `speed_multiplier * 2` throttled throughput to ~300 TPS at all speeds above 5x.
3. **Speed-based accumulator cap** — `max_acc = SIM_DT * speed * 2.0` lost fractional precision at low FPS, causing ~45 TPS at 1x/30fps.
4. **Larger backpressure drain (2× speed)** — dispatching 20 ticks in one batch vs 10+10 changes vision/global pass frequency, breaking simulation determinism.

All of these either throttle throughput or sacrifice determinism. The root cause is in the staging pipeline, not the accumulator.

## Fix

Triple-buffer the staging pipeline in `gpu_kernel.rs`: 3 slots instead of 2. This gives readback 3 frames to complete before backpressure stalls dispatch, eliminating the bottleneck at moderate speeds without changing batch sizes or accumulator behavior.

Changes in `gpu_kernel.rs`:

- Add `const STAGING_SLOTS: usize = 3;` — single source of truth for slot count
- Buffer creation uses `std::array::from_fn` to create `STAGING_SLOTS` buffers
- Struct initialization uses `[false; STAGING_SLOTS]` and `std::array::from_fn` for Arc arrays
- All `for i in 0..2` loops changed to `for i in 0..STAGING_SLOTS` (3 locations: `reset_agents`, `try_collect_staging`, `try_reset_agents`)
- Index flip changed from `1 - self.staging_index` to `(self.staging_index + 1) % STAGING_SLOTS`
- `active_config_index` flip unchanged (separate double-buffered concern for world config bind groups)

The accumulator logic in `main.rs` is unchanged — no drain, no speed cap.

## Tradeoff

One additional staging buffer per kernel (~few KB). No throughput penalty — dispatch is no longer blocked at moderate speeds. At extreme speeds (100x+), the per-batch `queue.submit()` overhead becomes the bottleneck (tracked separately in issue #79), not staging.

## Files Changed

- `crates/xagent-brain/src/gpu_kernel.rs` — triple-buffered staging pipeline
