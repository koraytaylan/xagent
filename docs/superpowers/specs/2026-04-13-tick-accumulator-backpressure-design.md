# Fix: Tick Accumulator Backpressure & State Readback Decoupling

**Date:** 2026-04-13
**Branch:** `fix/tick-accumulator-backpressure`

## Problem

At 10x speed multiplier with `PresentMode::Immediate`, the UI freezes and updates become erratic. At 1x–10x, agents appear frozen despite ticks advancing.

## Root Causes

### 1. Staging pipeline bottleneck (10x freezes)

`PresentMode::Immediate` drives the frame rate to 200-400+ FPS. The simulation accumulator adds `dt * speed_multiplier` per frame and dispatches `floor(accumulator / SIM_DT)` ticks to the GPU. At 10x, every frame triggers a `dispatch_batch` call with an async staging-buffer readback.

The readback pipeline was double-buffered: only 2 staging slots could be in-flight. When both were occupied, `dispatch_batch` returned `false` (GPU backpressure). The accumulator was **not drained** on backpressure — it kept growing each frame. When a staging slot freed up, the next dispatch fired a burst of accumulated ticks, causing position jumps.

### 2. State readback gated on dispatch (1x–10x frozen agents)

The GPU state readback and agent position copy was nested inside `if ticks_to_run > 0`. At 1x, dispatches only happen every ~20 frames (accumulator needs ~20 render frames at 120fps to accumulate enough for `brain_tick_stride` ticks). On the other ~19 frames, the position update was skipped entirely.

Additionally, both the pre-dispatch `try_collect_state()` and post-render `try_collect_state()` consumed staging data without applying it to agent positions, so even when the every-frame block ran, it found nothing to collect.

## Rejected Approaches

Several accumulator-side fixes were attempted and rejected:

1. **Drain accumulator on backpressure** — capping to 1 fixed-timestep's worth halved effective TPS at 10x (50% backpressure rate → ~300 TPS).
2. **`speed_tick_cap`** — capping `ticks_to_run` to `speed_multiplier * 2` throttled throughput to ~300 TPS at all speeds above 5x.
3. **Speed-based accumulator cap** — `max_acc = SIM_DT * speed * 2.0` lost fractional precision at low FPS, causing ~45 TPS at 1x/30fps.
4. **Larger backpressure drain (2× speed)** — dispatching 20 ticks in one batch vs 10+10 changes vision/global pass frequency, breaking simulation determinism.
5. **Batch alignment** — rounding `ticks_to_run` to `kernel_batch_size` multiples rounded to 0 at low speeds.

All of these either throttle throughput or sacrifice determinism.

## Fix

### Staging pipeline: 6-slot ring buffer

Increase `STAGING_SLOTS` from 2 to 6 in `gpu_kernel.rs`. This gives readback ample time to complete before backpressure stalls dispatch, eliminating the bottleneck at moderate speeds.

Compute dispatch is fully decoupled from staging: `dispatch_batch` always submits GPU work, and only attempts a staging copy when a slot is free.

### State readback: every-frame collection

Move the state readback + agent position update to an every-frame block outside the `if ticks_to_run > 0` gate. Remove the pre-dispatch `try_collect_state()` (the every-frame block handles all collection). Replace the post-render `try_collect_state()` with a bare `device.poll()` so map_async callbacks fire without consuming staging data.

### Minimum dispatch size

Every dispatch includes at least `brain_tick_stride` ticks so the brain always produces motor commands. At low speeds this borrows sim-time from the future (accumulator goes negative), repaid over the next few frames.

### PresentMode: Mailbox

Switch from `Immediate` to `Mailbox` to cap render at the display refresh rate (~120fps), reducing unnecessary frame pressure.

## Files Changed

- `crates/xagent-brain/src/gpu_kernel.rs` — 6-slot staging ring buffer, `kernel_batch_size()`/`brain_tick_stride()` accessors, `reset_agents_seeded()`, telemetry unmap guard
- `crates/xagent-sandbox/src/main.rs` — every-frame state readback, min-batch dispatch, remove redundant collect calls
- `crates/xagent-sandbox/src/bench.rs` — `run_tick_loop_bench()` headless accumulator test
- `crates/xagent-sandbox/src/renderer/mod.rs` — PresentMode::Mailbox
- `crates/xagent-sandbox/tests/integration.rs` — `deterministic_across_batch_sizes` test
