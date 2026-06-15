# Scope — Plan 0005

> Bank the throughput win that is *measured and guaranteed* — running the
> evolution at the GPU's occupancy knee for ~10× more agent-ticks/sec — then,
> only against a per-cooperative-pass cost profile (never a guess), reduce the
> fused brain pass's serial-barrier latency with bit-identical changes that each
> ship a measured speedup or record a measured negative. No open-ended spikes:
> the plan cannot terminate without a concrete, shipped result.

## Why this plan

Plan 0003 fused the per-batch submits (bit-identically) but ticks/sec did not
move, and the on-target measurement adjudicated why. The findings below are all
measured on the target macOS/Metal machine with the Plan 0003 probe tooling and
written up in `docs/reviews/2026-06-15-brain-pass-latency-ceiling.md` and
`docs/superpowers/specs/2026-06-10-learning-baseline.md` (*Simulation throughput
ceiling*).

1. **The ceiling is the fused brain/kernel pass, not submits or the `global`
   pass.** With fusion engaged (10 000 batches → 417 submits) tps is unchanged at
   ≈23 k; `--bench-phase-ab` shows skipping `global` recovers **+3%** and
   `global`+`vision` together **+7%**, so **≥93% of per-batch wall time is the
   `prepare`+`kernel` (fused brain) dispatch** (`prepare` is a `dispatch(1,1,1)`
   writing six `u32`s — negligible). This falsifies the
   `2026-06-14-grok-43.md` CPU-submission attribution.
2. **The simulation is GPU-under-occupied at small populations.** An
   agent-count sweep is flat from N=1 to N≈50 (−9% across a 5× population rise)
   and only falls past an occupancy knee at **N≈200**, where useful throughput
   (agent-ticks/sec) saturates at ≈2.4 M. The default `population_size` is **10**
   (`config.rs:265`, `governor.rs:1579`), which leaves the GPU ~90% idle and
   delivers ≈230 k agent-ticks/sec versus the ≈2.39 M available — a ~10× gap.
3. **The brain pass is a barrier-dense, single-workgroup-per-agent serial
   chain.** It dispatches as `dispatch_workgroups(agent_count, 1, 1)` with
   `@workgroup_size(256)` (`kernel_tick.wgsl:492`): one 256-thread workgroup per
   agent, looping `vision_stride` cycles (`:504`), each cycle a serial chain of
   thread-0-only physics/respawn (`:516-519`, `:533-534` — 255 idle threads) and
   seven cooperative passes separated by ~15–40 barriers
   (`brain_tick_inner`, `:416-445`). Wall time per batch ≈ one workgroup's chain
   latency, which is independent of agent count below occupancy — exactly the
   flat-then-knee sweep and the long-standing "N=10→4 changes nothing".
4. **The top-K recall is the barrier-density hotspot, and its fast path may be
   inactive.** `coop_recall_topk` (`brain_passes.wgsl:237`) is a 7-stage bitonic
   sort with a `workgroupBarrier()` inside every stage/step (`:250-275`) —
   ~28 barrier sub-steps in the workgroup-memory fallback. A subgroup-accelerated
   path exists (`apply_subgroup_markers`, `bitonic_sort_subgroup.wgsl`, gated on
   `wgpu::Features::SUBGROUP`, `gpu_kernel.rs:463-466`); **whether it is active on
   the target Metal device is unverified**.
5. **The governor under-uses any extra population.** With `population_size = 10`
   and `eval_repeats = 2` (`config.rs:228/246-248`), only `pop_size / repeats = 5`
   unique genomes are evaluated per generation (`governor.rs:942-944`). Scaling
   the population could buy more unique genomes (exploration) or more repeats
   (noise reduction) — an unmade decision that must be settled when the
   population grows.

## In scope

Work items in [TASKS.md](TASKS.md) (workstreams `0001`–`0003`):

- **0001 — Occupancy throughput (the guaranteed win).** Add an agent-count sweep
  harness that finds the GPU's occupancy knee, raise the evolution population to
  it, and settle how the governor spends the extra capacity — validated on a
  fixed seed. Ships the measured ~10× agent-ticks/sec gain unconditionally.
- **0002 — Brain-pass cost profile (the guaranteed artifact).** Record whether
  the subgroup top-K path is active on target, and add a measurement-only
  per-cooperative-pass limit knob to produce a definitive on-target cost
  breakdown of `brain_tick_inner`. Ships a diagnostic table unconditionally.
- **0003 — Brain-pass latency reductions (gated, ship-or-record).** Against the
  `0002` profile only, apply specific, correctness-justified, bit-identical
  transformations to the dominant cooperative pass; each ships a measured
  speedup or records a measured negative in `0003-BRAIN-LATENCY-DECISION.md`.

## Origin → workstream mapping

| Finding | Addressed by |
|---|---|
| Brain/kernel pass is ≥93% of per-batch cost (1) | `0002`, `0003` |
| GPU under-occupied at small populations (2) | `0001` |
| Barrier-dense single-workgroup serial chain (3) | `0003` |
| Top-K bitonic barrier hotspot / subgroup path unverified (4) | `0002`, `0003` |
| Governor under-uses extra population (5) | `0001` |

## Locked decisions

- **Guaranteed-concrete spine.** `0001` and `0002` ship unconditional, measured
  deliverables — a ~10× agent-ticks/sec evolution configuration and a definitive
  per-cooperative-pass cost table — *independent of any `0003` outcome*. `0003`
  is bounded and measurement-gated: every task is one specific,
  correctness-justified, bit-identical transformation that **either ships a
  measured speedup or records a measured negative** in
  `0003-BRAIN-LATENCY-DECISION.md`. The plan has no open-ended spike and cannot
  end without a concrete shipped result.
- **No speculative optimization.** `0003` touches only the cooperative pass that
  `0002`'s profile identifies as dominant. No guess-and-check: each change
  carries a written correctness argument (cross-thread read/write set, or
  subgroup/bitonic top-K equivalence) and is fenced by the bit-identical gate.
- **Bit-identical or re-baselined.** Every `0003` shader change keeps
  `deterministic_across_batch_sizes` and `fused_dispatch_matches_split`
  (`integration.rs`) green. A change that legitimately reorders floating-point
  reductions (e.g. a subgroup top-K with a different tie-break) is allowed only
  with an explicit, reviewed re-baseline of those tests and a recorded rationale.
- **Occupancy default is measured, not guessed.** The shipped `population_size`
  default is the knee reported by the `0001` sweep harness on the reference GPU
  (≈200), clamped to a documented safe maximum — not a hardcoded number invented
  here.
- **Strides and sensory lag stay locked.** `vision_stride`, `brain_tick_stride`,
  and the sensory-lag-100 default are Plan 0001's budgeted decision and are **not**
  a throughput knob here; latency comes from fewer/cheaper barriers and better
  occupancy, never coarser perception.

## Out of scope

- **Changing `vision_stride` / sensory lag for throughput.** Forbidden — a
  learning-semantics change gated by Plan 0001.
- **Re-opening `global`-pass parallelization (Plan 0003 `0004`).** Rejected with
  on-target evidence (≈3% cost); not revisited here.
- **Re-coupling readback to dispatch or changing the worker boundary** (Plan
  0002). Preserved unchanged.
- **A multi-agent-per-workgroup brain redesign.** A large architectural change to
  the cooperative-brain threading model; noted as a candidate in
  `ARCHITECTURE.md` §0003 but deferred to a possible follow-up plan, not built
  here unless `0002`/`0003` measurements specifically justify it.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
