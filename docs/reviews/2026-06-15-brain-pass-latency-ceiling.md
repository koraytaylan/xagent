# Review: The Real Simulation-Throughput Ceiling Is the Fused Brain/Kernel Pass, and the System Is GPU-Under-Occupied at Small Populations

**Date:** 2026-06-15
**Reviewer:** Claude (Claude Code)
**Scope:** the fused per-agent kernel pass and its scaling behaviour:
`crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl` (the
`@workgroup_size(256)` entry, the `vision_stride` cycle loop, the thread-0-only
`agent_physics` / `agent_death_respawn` phases, the `brain_tick_inner` cooperative
chain), `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` (the seven
cooperative passes, the bitonic top-K sort, the per-pass barriers),
`crates/xagent-brain/src/gpu_kernel.rs` (`dispatch_ticks` pass structure,
`dispatch_workgroups(agent_count, 1, 1)`, the `DispatchProbe` pass-skip knobs,
`apply_subgroup_markers` / SUBGROUP feature gating), and the Plan 0003
instrumentation (`--bench-phase-ab`, `XAGENT_SKIP_GLOBAL` / `XAGENT_SKIP_VISION` /
`XAGENT_PROBE_GPU_WAIT`, `[SIM-PROBE]` / `[BENCH-PROBE]`). This review **supersedes
the root-cause conclusion of `2026-06-14-grok-43.md`** with on-target
measurements; it is the findings doc for a prospective brain-pass-latency plan.

**Question under review:** After Plan 0003 fused the per-batch submits
(bit-identically), high-speed-multiplier ticks/sec did not improve. Where is the
ceiling actually, and what — if anything — can move it?

---

## Executive summary

Plan 0003's submit fusion works as designed (10 000 kernel-batches collapse to
417 `queue.submit` calls) yet ticks/sec is unchanged at ≈23 k, and an on-target
pass-isolation A/B shows that **removing the entire `global` pass recovers only
+3% and removing `global`+`vision` together only +7%.** Therefore **≥93% of
per-batch wall time is the `prepare`+`kernel` dispatch — overwhelmingly the fused
brain `kernel` pass** (`prepare` is a `dispatch(1,1,1)` that writes six `u32`s).
This **falsifies the `2026-06-14-grok-43.md` conclusion** that CPU
command-submission overhead is the limiter: submits were collapsed ~24× with no
tps movement.

An agent-count sweep then shows the system is **latency-bound and GPU-
under-occupied at small populations**: ticks/sec is *flat* from N=1 to N≈50 and
only falls past an occupancy knee at **N≈200**, where useful throughput
(agent-ticks/sec) saturates at ≈2.4 M. The default population of **10**
(`config.rs:265`, `governor.rs:1579`) therefore leaves the GPU ~90% idle and
delivers ~1/10th of the evolutionary throughput the same hardware can sustain.

Two distinct, independent levers follow, addressing two *different* goals:

1. **Occupancy (cheap, measured ~10×, for evolution throughput).** Run the
   population near the knee (~150–250). No shader change; the GPU is idle. This
   is the single biggest, lowest-risk win and is config-level.
2. **Per-workgroup serial latency (hard, for single-stream sim-tps).** The brain
   `kernel` pass is one 256-thread workgroup *per agent* executing a long,
   barrier-dense serial chain (thread-0-only physics/respawn + seven cooperative
   passes incl. a 7-stage bitonic sort, looped `vision_stride` times). Reducing
   that chain's depth is the only thing that raises single-stream ticks/sec and
   shifts the occupancy knee — but it is invasive and risks the determinism gate.

---

## Measured evidence

All numbers are on the target macOS/Metal machine where the ceiling was reported
(release build), gathered with the Plan 0003 probe tooling.

**Pass isolation — `--bench-phase-ab --bench-ticks 1000000 --bench-agents 10`:**

| Arm | tps | Δ vs baseline | batches | submits |
|---|---|---|---|---|
| full (baseline) | 22,989 | — | 10,000 | 417 |
| skip global | 23,595 | **+3%** | 10,000 | 417 |
| skip vision | 24,422 | +6% | 10,000 | 417 |
| skip global+vision | 24,710 | +7% | 10,000 | 417 |

Fusion is fully engaged (417 ≈ ⌈10000 / 24⌉ submits) and baseline tps ≈23 k
still equals the originally reported ≈20 k ceiling. The `global` pass is ≈3%;
both auxiliary passes together ≈7%; the residual ≥93% is the brain `kernel`
dispatch.

**Agent-count sweep — `--bench --bench-ticks 200000 --bench-agents N`,
agent-ticks/sec = tps × N (the metric that matters for evolution):**

| N | tps | agent-ticks/s | s/individual (120 k-tick gen) |
|---|---|---|---|
| 1 | 23,269 | 23 k | — |
| 4 | 23,037 | 92 k | — |
| 10 | 22,952 | 230 k | 0.52 |
| 50 | 20,896 | 1.04 M | 0.115 |
| 200 | 11,942 | **2.39 M** | **0.050** |
| 1000 | 2,365 | 2.37 M | 0.051 |
| 5000 | (did not complete) | — | — |

tps is flat to N≈50 (−9% across a 5× population increase), the occupancy knee is
**N≈200**, and beyond it ticks/sec falls proportionally (agent-ticks/sec
plateaus). An individual is simulated ~10× cheaper at N=200 (0.050 s) than at
N=10 (0.52 s).

---

## Why the prior CPU-submission review (`2026-06-14-grok-43.md`) was wrong

That review concluded "CPU-Side Command Submission Overhead Dominates for Small
Populations," estimating ~200 batches/sec × 100 ticks = ~20 k tps and attributing
the floor to per-batch encoder/recording/submit cost. Plan 0003 acted on that
mechanism (fusing all full batches into one encoder + one submit, push-constant
`start_tick`, conditional polls, scratch uniform) — all bit-identical and
merged. The on-target result refutes the attribution: **submits dropped ~24×
(10 000 → 417) and tps did not move.** The review's own caveat that "GPU
execution time per batch is << submission overhead" is exactly inverted on the
target hardware; GPU execution of the brain pass *is* the wall. The review was a
careful static analysis with no execution measurement — which is precisely why
Plan 0003's SCOPE flagged the attribution as unverified and led with a
measurement workstream. The submit-fusion work is not wasted (it is correct,
bit-identical, and removes a real if non-dominant cost, and its instrumentation
produced this verdict), but it was the wrong target for throughput.

---

## The structural root cause: a barrier-dense, single-workgroup-per-agent serial chain

The fused kernel is dispatched as `dispatch_workgroups(agent_count, 1, 1)` with
`@workgroup_size(256)` (`kernel_tick.wgsl:492`): **one 256-thread workgroup per
agent.** Inside, it loops `vision_stride` (default 10) cycles
(`kernel_tick.wgsl:504`), and **each cycle is a long serial dependency chain of
barrier-separated phases**:

1. **Thread-0-only physics.** `agent_physics` runs `brain_tick_stride` sub-ticks
   inside `if (tid == 0u)` (`kernel_tick.wgsl:516-519`) — **255 of 256 threads
   idle** — then `storageBarrier(); workgroupBarrier();` (`:522`).
2. **Cooperative food detect** (256 threads) with internal barriers
   (`kernel_tick.wgsl:525`, plus `brain_passes`-style barriers inside).
3. **Thread-0-only death/respawn** `agent_death_respawn` (`kernel_tick.wgsl:533-
   534`) — again **255 threads idle** — then `storageBarrier(); workgroupBarrier();`
   (`:537`).
4. **`brain_tick_inner`** (`kernel_tick.wgsl:416-445`): seven cooperative passes,
   each gated by a barrier —
   `coop_feature_extract` → barrier → `coop_encode` → barrier →
   `coop_habituate_homeo` → storage+workgroup barrier → `coop_recall_score` →
   barrier → `coop_recall_topk` → storage+workgroup barrier →
   `coop_predict_and_act` → storage+workgroup barrier → `coop_learn_and_store`,
   then a final barrier (`:555`).

The top-K recall (`coop_recall_topk`, `brain_passes.wgsl:237`) is itself a
**7-stage bitonic sort** with a `workgroupBarrier()` inside every stage/step loop
(`brain_passes.wgsl:250-275`) — on the order of ~28 barrier sub-steps in the
workgroup-memory fallback path alone. `coop_predict_and_act` and
`coop_learn_and_store` add several more storage/workgroup barriers each
(`brain_passes.wgsl:346,396,442,461,792,820,842`).

Net: **tens of full-workgroup synchronizations per cycle × `vision_stride` cycles
≈ a few hundred serialized barrier points per kernel dispatch.** A workgroup
cannot retire faster than the latency of this chain, and the chain depth is
independent of how many *other* workgroups (agents) are resident. That is the
mechanism behind the flat-tps-then-knee sweep: until N exceeds the GPU's
workgroup occupancy (~200 here), wall time per batch ≈ one workgroup's chain
latency, regardless of N — so adding agents up to the knee is nearly free, and
below the knee the machine is mostly idle.

Two amplifiers worth singling out:

- **Thread-0 serialization.** Physics and death/respawn run on a single thread of
  a 256-wide workgroup (`kernel_tick.wgsl:516-519`, `:533-534`). For those
  phases the workgroup is at 1/256 utilization, lengthening the serial chain.
- **Top-K barrier density.** The bitonic fallback dominates the per-cycle barrier
  count. A subgroup-accelerated path exists (`apply_subgroup_markers`,
  `bitonic_sort_subgroup.wgsl`, gated on `wgpu::Features::SUBGROUP`); **whether it
  is active on the target Metal device is unverified** and is the cheapest single
  thing to check (a barrier-heavy fallback running on Metal would directly inflate
  this chain).

---

## Two levers, two goals — do not conflate them

The "no speedup" symptom and the "10× free throughput" finding are about
*different* axes. A future plan must pick its goal explicitly.

### Lever A — Occupancy / population (cheap; raises evolution throughput ~10×)

The GPU is idle at N=10. Running near the knee (~150–250) yields ≈2.4 M
agent-ticks/sec vs ≈230 k — a ~10× increase in individuals-evaluated per wall
second — **with no shader change**, only a population-size change
(`population_size`, `config.rs:215/265`, consumed at `governor.rs:942`).

Caveats that make this a *plan*, not a one-line bump:

- **It is not free for evolution *semantics*.** Bigger populations change
  selection pressure and diversity, and the governor multiplexes
  `population_size` into unique configs × `eval_repeats`
  (`governor.rs:942-999`) — raising the size could mean more unique genomes or
  more repeats; the intended split must be decided.
- **Per-stream sim-tps drops** (each generation takes longer in wall-clock even
  as per-individual cost falls), so this helps headless evolution, not the
  interactive single-agent feel.
- **Auto-sizing to the knee** is attractive: probe occupancy once at startup and
  default the population to the knee for the detected GPU, rather than hardcoding
  200.
- Memory/scaling must be confirmed (N=1000 ran; N=5000 did not complete — find
  and document the practical ceiling).

### Lever B — Per-workgroup serial latency (hard; raises single-stream sim-tps)

This is the only lever that raises ticks/sec for a *single* agent stream and
shifts the occupancy knee. Candidate directions, each measurement-gated and
determinism-fenced:

- **Confirm and force the subgroup top-K path** on Metal; if the bitonic fallback
  is active, that is likely the largest single barrier-count reduction available.
- **Parallelize the thread-0-only physics/respawn** across the workgroup (or
  overlap them), removing the 1/256-utilization stretches.
- **Collapse barriers between cooperative passes** where the data dependency does
  not actually require a full storage barrier (audit each
  `storageBarrier(); workgroupBarrier();` in `brain_tick_inner` against the real
  read/write set).
- **Pack multiple agents per workgroup** to raise occupancy density at fixed N —
  a redesign, since each agent currently consumes all 256 threads cooperatively.

**Hard constraints on Lever B:** `vision_stride` / `brain_tick_stride` and the
sensory-lag-100 default are locked by Plan 0001 (not a throughput knob); every
change must keep `deterministic_across_batch_sizes` and
`fused_dispatch_matches_split` green and be bit-identical or explicitly
re-baselined. The barrier-uniformity / `s_alive` broadcast invariant
(`kernel_tick.wgsl` SAFETY block) must be preserved.

---

## Recommendation

Measure-before-build, as in Plan 0003:

1. **First, the cheap occupancy win (Lever A).** It is measured, ~10×, and
   shader-free. Author it as a plan that (a) picks/auto-sizes the population to
   the occupancy knee, (b) resolves the unique-vs-repeats governor question, and
   (c) re-validates evolution dynamics at the larger size. This recovers most of
   the *evolutionary* throughput immediately.
2. **Then, if single-stream sim-tps still matters, Lever B.** Gate it on a
   per-pass GPU profile of `brain_tick_inner` — add a `XAGENT_SKIP_KERNEL` (or
   per-cooperative-pass skip) probe in the same style as the existing
   `XAGENT_SKIP_GLOBAL` / `XAGENT_SKIP_VISION` knobs so the dominant cooperative
   pass (top-K sort is the prime suspect) is identified before any rewrite.
   Confirm the subgroup top-K path is active on the target first; it may be the
   cheapest large win on its own.

The throughput ceiling is real and now precisely located: it is the fused brain
pass's serial-barrier latency, exposed by running far below GPU occupancy. Submit
fusion (0003) and `global`-pass parallelization (0004, rejected) were both off
this path; occupancy and per-workgroup barrier depth are on it.
