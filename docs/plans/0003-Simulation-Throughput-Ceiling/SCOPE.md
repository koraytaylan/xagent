# Scope — Plan 0003

> Lift the high-speed-multiplier simulation throughput ceiling by measuring the
> true per-batch limiter first, then collapsing the one-`queue.submit`-per-100-tick
> tax in `dispatch_ticks` into a single fused submit — without changing a single
> byte of simulation result.

## Why this plan

At 1× the runtime sustains the expected 60 tps and scales linearly through 100×
(≈6 k tps). At 1000× the observed ceiling is ≈20 k tps instead of the
proportional 60 k tps, and dropping the population from 10 agents to 4 changes
nothing. The `2026-06-14-grok-43.md` review located the mechanism precisely but
left its root-cause attribution unverified. This plan adjudicates that review.

**Verified against `develop` @ `2a751a8` by direct code reading plus a 14-agent
verification/design pass.**

1. **Per-kernel-batch submit tax.** `GpuKernel::dispatch_ticks`
   (`gpu_kernel.rs:1354-1460`) records exactly one `create_command_encoder` +
   4 compute passes + one `queue.submit` per kernel-batch
   (`vision_stride * brain_tick_stride` = 100 ticks at defaults,
   `gpu_kernel.rs:1366-1434`). The worker caps each `advance_compute` at one
   kernel-batch (`dispatch_cap = kernel.kernel_batch_size()`,
   `sim_runtime.rs:530`) and spins without sleeping while a backlog drains
   (`sim_runtime.rs:736-743`). So `throughput ≈ submits_per_sec × 100`, and at
   ≈200 submits/sec the ceiling is ≈20 k tps — independent of agent count.
2. **A uniform write-after-write hazard forces the per-batch submit.** Every full
   batch in one `dispatch_ticks` call writes a fresh `start_tick` into the *same*
   `world_config_bufs[active_config_index]` uniform (the double-buffer index
   flips only once, at end of call, `gpu_kernel.rs:1458`), so a submit must drain
   the GPU before the next `write_buffer` clobbers `WC_TICK`
   (`gpu_kernel.rs:1361-1363`). `start_tick` is the **only** per-batch-varying
   value the kernel reads (`kernel_tick.wgsl:488-490`); `vision_stride` /
   `brain_tick_stride` are constant across full batches.
3. **The dominant per-batch cost is unmeasured.** The review attributes the
   ceiling to CPU command-recording overhead, but ≈5 ms per batch is implausible
   for recording 4 tiny passes, and that attribution rests on static inspection
   only. The true limiter — CPU submit/recording vs Metal command-buffer
   back-pressure vs GPU pass execution — has never been measured.
4. **The `global` pass is a single-workgroup dispatch.** The grid-rebuild +
   collision pass runs as `pass.dispatch_workgroups(1, 1, 1)`
   (`gpu_kernel.rs:1416-1424`, `global_tick.wgsl`): one 256-thread workgroup
   serially clearing ≈42 k grid entries plus a 3× collision loop with 9 barrier
   pairs, once per batch. Its cost is dominated by grid size and is
   **agent-count-independent**, which explains the observed "no change from N=10
   to N=4" better than CPU-recording does. It is a prime suspect for the real
   floor and is the one fixed cost fusing submits cannot remove.
5. **Per-iteration fixed costs.** `Worker::step` issues two unconditional
   `device.poll(wgpu::Maintain::Poll)` per loop pass (`gpu_kernel.rs:1544`,
   `gpu_kernel.rs:2136`, via `try_collect_state_snapshot` /
   `try_collect_telemetry`) even when nothing is in flight, and
   `build_world_config` allocates a fresh `vec![0.0f32; 24]` on every batch
   (`buffers.rs:422`).

This plan does **not** touch `brain_tick_stride`, `vision_stride`, or the
sensory-lag bound. Those are simulation semantics locked by Plan 0001 (`lag100`
default); throughput here comes from *fewer submits wrapping the identical pass
sequence*, never from coarser perception.

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| "The ≈200 batches/sec ceiling is dominated by per-batch CPU submit/command-recording overhead." | `2026-06-14-grok-43.md` Executive Summary & §Per-Batch Overhead | ≈5 ms/batch is implausible for recording 4 tiny compute passes; the single-workgroup `global` pass (`dispatch(1,1,1)`, ≈42 k serial grid stores, `gpu_kernel.rs:1416-1424`) is agent-count-independent and better explains the N=10→4 no-change. Attribution is unmeasured — workstream `0001` measures it before any fix is sized. |
| The `global` pass is "effectively constant time" and therefore negligible. | `2026-06-14-grok-43.md` §Why Agent Count Did Not Matter | Being constant-time is precisely why it is the leading suspect: a single-workgroup serial reduction that may *be* the dominant fixed cost, not a negligible one. Tracked as finding 4 → workstream `0004`. |

## In scope

Work items in [TASKS.md](TASKS.md) (workstreams `0001`–`0004`):

- **0001 — Per-batch cost instrumentation.** Add feature-light, default-off
  probes that isolate CPU submit-return time vs GPU-complete time and A/B the
  `global`+`vision` passes, then record a 1000× baseline. This measurement gates
  the fuse-factor sizing in `0002` and the `0004` gate.
- **0002 — Fuse kernel-batches into one submit.** Move `start_tick` to a kernel
  push constant, write the world-config uniform once per `dispatch_ticks` call,
  and record all full batches into a single command encoder + single submit
  (bounded by the existing Metal pass-count guard), then widen the worker
  dispatch cap so `dispatch_ticks` is actually handed many batches. Bit-identical.
- **0003 — Per-iteration fixed-cost cleanup.** Make the two readback polls
  conditional on work being in flight and replace the per-batch world-config
  `Vec` allocation with a reused scratch array. Cheap hygiene, bit-identical.
- **0004 — Gated `global`-pass parallelization.** Only if `0001` shows the
  single-workgroup `global` pass is the dominant per-batch cost, prototype
  parallelizing its grid-clear/collision across workgroups; resolve into a
  decision document.

## Origin → workstream mapping

| Finding | Addressed by |
|---|---|
| Per-kernel-batch submit tax (1) | `0002` |
| Uniform `WC_TICK` write-after-write forces the per-batch submit (2) | `0002` |
| Dominant per-batch cost is unmeasured (3) | `0001` |
| `global` pass is a single-workgroup dispatch, a candidate floor (4) | `0004` |
| Per-iteration fixed costs: double poll, per-batch `Vec` alloc (5) | `0003` |

## Locked decisions

- **Measure before sizing.** Do not commit `MAX_FUSED_BATCHES` or open the
  `0004` `global`-pass rewrite until `0001` quantifies, at 1000×, submit-return
  wall time vs GPU-complete wall time and tps with the `global`+`vision` passes
  on vs off. Gate: the `0004` task stays unstarted until that table exists.
- **Fuse, never widen strides.** `vision_stride`, `brain_tick_stride`, and the
  sensory-lag-100 default stay fixed (Plan 0001's budgeted decision). Throughput
  comes from fewer submits, not coarser perception. No `TD_DISCOUNT`
  recalibration is implied.
- **Bit-identical or it does not ship.** Every code change must keep
  `deterministic_across_batch_sizes` (`integration.rs:629-699`) green and add a
  new fused-vs-split equivalence test over a non-multiple tick total. The
  `start_tick` push constant carries an exact `u32`, strictly *more* precise than
  today's `tick as f32 → u32` round-trip; it can only diverge from today above
  2²⁴ ticks, and there toward correct.
- **Generation budget clamp stays upstream.** `clamp_ticks_to_generation_budget`
  (`sim_runtime.rs:538-543`) stays upstream of `dispatch_ticks`; widening
  `dispatch_cap` only enlarges the pre-clamp candidate, so a fused dispatch still
  stops exactly at `tick_budget`.
- **`global`-pass rewrite is gated.** Parallelizing the `global` pass lands only
  if `0001` shows it is the dominant per-batch cost; otherwise the negative
  result is recorded in `0004-GLOBAL-PASS-DECISION.md` and the path is closed.

## Out of scope

- **Changing strides / sensory lag as a throughput lever.** Forbidden — it is a
  learning-semantics change gated by Plan 0001, not a runtime knob.
- **A headless fast-forward mode** that suspends 60 Hz/30 Hz publication
  copy-submits at extreme speed. Complementary and display-fidelity-trading; its
  measurement value is subsumed by `0001`. Deferred to a possible follow-up
  plan, not built here.
- **Re-coupling readback to dispatch.** The decoupled 60 Hz/30 Hz publication and
  the 6-slot latest-wins staging ring from Plan 0002 are preserved unchanged.
- **Multi-queue submission or multiple command buffers per submit.** A larger
  GPU-ownership change; not required to collapse the per-batch tax.
- **Touching `hud_dirty` / render-thread cost.** That is redraw FPS, decoupled
  from worker tps by Plan 0002; it cannot move the tps ceiling.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
