# Scope - Plan 0006

> Turn the measured 20-23 k-tps brain-pass ceiling into a 60 k-tps target by
> exposing more independent GPU work per low-population tick: prove the target
> budget, exhaust every dominant dense brain loop, recover any non-brain floor
> exposed by the measurements, then ship only when the target run clears 60 k.

## Why this plan

Verified against `develop` @ `069ff8a`. Plans 0003 and 0005 already ruled out
the cheap answers; this plan starts from their measured negatives and attacks
the remaining GPU-internal parallelism.

1. **The raw-tps limiter is inside the fused kernel/brain pass.** On the target
   macOS/Metal run, submit fusion was engaged (10,000 kernel-batches -> 417
   submits) but baseline stayed at 22,989 tps; skipping `global` recovered only
   +3% and skipping `global`+`vision` only +7%, so at least 93% of batch wall
   time is `prepare`+`kernel`, effectively the fused brain dispatch
   (`docs/superpowers/specs/2026-06-10-learning-baseline.md:508-533`,
   `gpu_kernel.rs:1531-1587`).
2. **Population occupancy is useful work, not the requested raw-tps win.** The
   sweep peaks at N=200 with 2.40 M agent-ticks/sec, but raw tps falls from
   22,935 at N=10 to 11,990 at N=200; the 60 k goal is raw ticks/sec at the
   existing small-population run, so just adding agents does not solve it
   (`docs/superpowers/specs/2026-06-10-learning-baseline.md:600-617`).
3. **The current dispatch shape exposes too little work at N=10.** The host
   dispatches one workgroup per agent (`pass.dispatch_workgroups(self.agent_count,
   1, 1)`, `gpu_kernel.rs:1568`) and the shader is `@workgroup_size(256)`
   (`kernel_tick.wgsl:515`). With N=10 that is only 10 workgroups for the long
   serial brain chain, matching the measured flat N=1..10 tps.
4. **The dominant passes are dense matrix/vector work, not the old top-K
   suspect.** The pass-limit profile shows `encode`, `predict_and_act`, and
   `learn_and_store` dominate; `recall_topk` is about 1.6% and the subgroup path
   is inactive but irrelevant (`docs/superpowers/specs/2026-06-10-learning-baseline.md:624-648`).
5. **Dense work is parallel only across output rows today, leaving long serial
   inner loops.** `coop_encode` assigns one encoded dimension per thread and each
   thread loops across every feature (`brain_passes.wgsl:129-136`); the predictor
   trains and predicts one row per thread, each looping across all 128 encoded
   inputs twice (`brain_passes.wgsl:316-332`); encoder-credit learning repeats
   the feature loop once per encoded dimension (`brain_passes.wgsl:753-761`).
6. **The real breakthrough requires crossing the workgroup boundary.** WGSL has
   no cross-workgroup barrier inside one dispatch, and current intermediates live
   in `var<workgroup>` scratch (`brain_passes.wgsl:22-36`). Any per-agent
   multi-workgroup design must move selected intermediates into storage buffers
   and sequence them with separate dispatches from `dispatch_ticks`.
7. **Time-axis parallelism is not a valid shortcut.** The kernel loops each
   batch's brain cycles in order (`kernel_tick.wgsl:527-578`), and each cycle's
   motor output feeds later physics through `decision_buffer`; parallelizing
   future cycles would change the simulation semantics rather than accelerate
   the same workload.

## In scope

- **0001 - Throughput budget and no-go map.** Re-measure the N=10 target budget
  with existing probes, record the pass-0 no-brain floor, and lock the decision
  rules for continuing or stopping.
- **0002 - Same-dispatch cooperative tiling.** Use all 256 lanes inside the
  existing one-workgroup-per-agent brain pass for dense dot products and exact
  thread-0 copy/context loops before paying for split dispatches.
- **0003 - Split-cycle execution scaffold.** Add an opt-in execution mode that
  runs one brain cycle per dispatch sequence and proves the dispatch/barrier
  overhead is low enough to justify multi-workgroup phases.
- **0004 - Multi-workgroup dense brain phases.** Move feature/encoded/prediction
  intermediates into storage scratch and split `encode`, encoder-credit learning,
  predictor train+predict, action-tail reductions, and memory reinforcement
  across more lanes/workgroups.
- **0005 - Non-brain floor recovery.** If the N=10 no-brain floor is too low, or
  if dense-brain parallelism lands but 60 k is still missed, replace the fused
  kernel's brute-force food scan with the existing food-grid neighborhood logic
  and remeasure the floor.
- **0006 - 60 k closure.** Ship the parallel mode as the default only after it
  reaches >=60,000 tps at N=10 and keeps learning/evolution gates green.

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Kernel/brain is the raw-tps limiter (1) | `0001`, `0005` |
| Population occupancy is not raw tps (2) | `0001`, `0005` |
| Too few workgroups at N=10 (3) | `0002`, `0003`, `0004` |
| Dominant dense passes, not top-K (4) | `0001`, `0002`, `0004` |
| Serial inner loops in dense passes (5) | `0002`, `0004` |
| Workgroup-local intermediates block multi-workgroup parallelism (6) | `0003`, `0004` |
| Time-axis dependency forbids future-cycle parallelism (7) | `0001`, `0003` |
| Brute-force per-cycle food scan can dominate the non-brain floor (1, 7) | `0005` |

## Locked decisions

- **The target metric is raw N=10 ticks/sec.** The ship gate is
  `--bench --bench-ticks 1000000 --bench-agents 10` on the reference
  macOS/Metal machine reaching >=60,000 tps. Agent-ticks/sec at N=200 is tracked
  as a regression guard, not as a substitute.
- **No early no-go while known parallel work remains.** If
  `XAGENT_KERNEL_PASS_LIMIT=0 --bench-agents 10` is below 90,000 tps on target,
  `0005` becomes mandatory before the final gate; the plan does not stop until
  the brute-force food scan and every dominant dense brain loop in `0004` have
  either shipped speed or been replaced by a faster measured equivalent.
- **Default stays fused until the final gate.** New execution modes are opt-in
  (`XAGENT_BRAIN_EXECUTION_MODE=...`) until `0005` passes. Existing tests and
  default behavior must remain green throughout.
- **Split dispatches must earn their overhead.** If the split-serial scaffold is
  more than 15% slower than fused serial at N=10 before any multi-workgroup math
  lands, multi-workgroup work stops and the decision doc records dispatch
  sequencing overhead as the blocker.
- **Float-order changes are allowed only behind deterministic gates.** Dense
  tiling changes reduction order. The gate is deterministic within the new mode,
  bounded drift against fused serial on fixed seeds, and no fixed-seed learning
  regression. The plan does not pretend these changes are byte-identical.
- **The work list is exhaustive for the measured hotspots.** The final gate may
  fail only after the plan has addressed: `encode`, predictor train+predict,
  encoder-credit learning, action-tail reductions/copies, memory reinforcement,
  and the fused kernel's brute-force food scan. A final miss must name which
  remaining code path owns the gap, not merely say "parallelism was tried."
- **No population-default change.** Plan 0005 already measured that larger shared
  populations regress or fail to improve evolution
  (`docs/superpowers/specs/2026-06-10-learning-baseline.md:650-686`). This plan
  uses parallel workgroups per agent instead.

## Out of scope

- **Parallelizing `global` or `vision` as first-order target work.** Already
  measured at <=7% combined; not enough for a 3x raw-tps goal. They may be
  revisited only if the final decision doc shows the post-brain/post-food gap is
  under 7%.
- **Raising `population_size` as the win.** Useful for occupancy studies, but it
  lowers raw tps and failed the evolution validation.
- **Changing `vision_stride`, `brain_tick_stride`, or sensory lag.** Those are
  learning-semantics knobs locked by prior plans, not raw compute
  parallelization.
- **Parallelizing future simulation time.** Brain/physics cycles are sequentially
  dependent through motor state and homeostasis.
- **f16, quantization, or model-size reductions.** Potential later speed paths,
  but not GPU parallelization of the current workload.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
