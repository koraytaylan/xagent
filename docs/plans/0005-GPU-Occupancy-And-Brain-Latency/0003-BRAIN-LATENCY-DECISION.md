# 0003 — Brain-pass latency reduction decision

> **Task:** `dominant-pass-latency-reduction` (GATED). **Gate:** start only
> after `subgroup-topk-verification` and `per-cooperative-pass-limit-probe` have
> recorded, *on the target macOS/Metal GPU*, (a) which cooperative pass dominates
> `brain_tick_inner` and (b) whether the subgroup top-K path is active. Apply the
> *one* transformation that matches the profile; do not guess. **Done when:**
> either a bit-identical (or justified-re-baselined) transformation ships with a
> recorded ≥10%-at-N=10 on-target improvement and no knee regression, **or** this
> doc records the measured negative and the working tree is reverted clean.

## Decision

**PENDING — gate not yet satisfied. No transformation applied.** The gate
requires the 0002 per-cooperative-pass cost profile and the subgroup-path fact
*measured on the target macOS/Metal discrete GPU*. The 0002 wiring is landed
(`XAGENT_KERNEL_PASS_LIMIT`, `GpuKernel::has_subgroup()` + the `top-K recall
path:` log) and verified byte-identical/compiling, but it was developed in a
Linux container with **no GPU adapter** (`GpuKernel::is_available()` == false),
so no on-target profile exists yet. The plan forbids speculative optimization
("`0003` touches only the cooperative pass that `0002`'s profile identifies as
dominant. No guess-and-check"), so the correct state is to record the gate as
open and ship nothing here.

The plan's concrete value is already banked by `0001`/`0002` (the
occupancy-sized population + the profiling instrumentation), so this open gate
is a recorded result, not a dead end — exactly the ship-or-record contract.

## How to close this gate

Run the 0002 measurements on the reference GPU (commands in the Plan 0005
subsection of `docs/superpowers/specs/2026-06-10-learning-baseline.md`), fill in
that subsection's tables, then return here and pick the single matching
candidate below by the profile.

## Candidate transformations (the profile decides which, if any, executes)

| If the profile shows… | Apply | Correctness argument | Bit-identical gate |
|---|---|---|---|
| `recall_topk` dominant **and** bitonic fallback active **and** `SUBGROUP` available on target | Force the subgroup top-K path (`apply_subgroup_markers`, `gpu_kernel.rs`) | Same K elements selected ⇒ identical top-K output | Keep `deterministic_across_batch_sizes` / `fused_dispatch_matches_split` green; **or** re-baseline them with a recorded rationale if the subgroup sort tie-breaks differently |
| A specific `storageBarrier()` is provably redundant (no thread reads another thread's storage-buffer write across it; e.g. a workgroup-memory-only handoff) | Downgrade that `storageBarrier()` to a bare `workgroupBarrier()` (`kernel_tick.wgsl:434/440/443`, `brain_passes.wgsl:461/792/820`) | Justify from the actual cross-thread read/write set at that point | Bit-identical (no result change; only removes an unneeded storage fence) |
| Thread-0 physics/respawn serialization dominates (`kernel_tick.wgsl` physics sub-tick loop + death/respawn) | Implement a bit-identical multi-thread form **if one exists**, else record the structural reason it is inherently serial (sequential sub-tick dependency) | Physics sub-ticks are sequentially dependent → likely a structural negative | N/A if negative recorded |

## Ship-or-record rule

1. Read the 0002 profile; pick the single matching transformation above.
2. Implement it; keep the two determinism tests green (or re-baseline with a
   recorded rationale for a justified reduction-order change only).
3. Measure on target: tps at `--bench-agents 10` (latency-bound regime) and at
   the occupancy knee. **Ship** if tps improves ≥10% at N=10 without regressing
   the knee; **otherwise revert** and record the measured negative here.
4. Update this doc with the decision up front, the transformation tried, the
   on-target before/after, and — if reverted — the reason and the gate that
   would reopen it.

## Status

- 2026-06-15 — **Open / pending on-target 0002 profile.** Wiring for the gate
  (pass-limit probe + subgroup accessor/log) is merged and byte-identical when
  unset. No latency transformation applied; working tree clean of any 0003
  shader change.
