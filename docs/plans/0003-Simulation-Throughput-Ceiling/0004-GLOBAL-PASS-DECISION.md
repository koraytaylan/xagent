# 0004 — `global`-pass parallelization decision

> **Spike task:** `parallelize-global-pass-spike` (GATED). **Gate:** start only
> after `dispatch-cost-instrumentation` and `fused-throughput-remeasure` show the
> single-workgroup `global` pass (`pass.dispatch_workgroups(1, 1, 1)`,
> `gpu_kernel.rs`) is the dominant residual per-batch cost — i.e. tps jumps
> materially with `XAGENT_SKIP_GLOBAL_VISION=1` on the hardware where the ceiling
> was observed. **Done when:** either this doc rejects the path with measured
> evidence (negative recorded), or a measured prototype justifies a follow-up
> implementation plan.

## Decision

**REJECTED with on-target evidence. The `global` pass is NOT the residual
ceiling — no rewrite ships.** On the macOS/Metal machine where the ≈20 k-tps /
1000× ceiling was observed, the `--bench-phase-ab` isolation run shows skipping
the `global` pass alone recovers only **+3%**, and skipping *both* `global` and
`vision` recovers only **+7%** — so ≥93% of per-batch wall time is the
`prepare`+`kernel` (fused brain) dispatch, which parallelizing `global` cannot
touch. This satisfies the "rejects the path with measured evidence (negative
recorded)" branch of the spike's `Done when`.

Fusion (0002) is engaged in this run (10 000 kernel-batches → 417 submits) yet
baseline tps is still ≈23 k — confirming the limiter is neither CPU submit
overhead (0002's target) nor the `global` pass (0004's target). The real ceiling
is the fused **kernel/brain pass** itself; addressing it is a separate plan, not
this workstream.

## The three paths

| Path | What it is | Cost it removes | Risk / cost to build |
|---|---|---|---|
| A — leave single-workgroup | Status quo: `global_tick` runs on one 256-thread workgroup, `dispatch(1,1,1)`. | Nothing. | None. **← chosen.** |
| B — parallelize grid-clear + grid-build only | Dispatch `ceil(grid_cells / 256)` workgroups for `phase_clear` / `phase_food_grid` / `phase_agent_grid`; keep collision single-workgroup. | The ≈42 k serial grid stores per batch — measured at **≤3%** of per-batch cost, so at most ~3% even if fully removed. | Medium. Not worth ≤3%. |
| C — fully parallelize, incl. collision | B plus multi-workgroup collision accumulate/apply. | All `global`-pass serial cost — measured **≤3%**. | High (cross-cell atomics, determinism risk). Not worth ≤3%. |

The spike is not opened: even the *upper bound* of what B or C could recover
(the entire `global` pass) is +3% on target, below any reasonable
risk/reward bar. See `ARCHITECTURE.md` §0004 for the construction sketch, kept
for the record.

## Measured evidence

**On-target — macOS/Metal, `--bench-phase-ab --bench-ticks 1000000
--bench-agents 10` (THE adjudicating run):**

| Arm | tps | Δ vs baseline | batches | submits |
|---|---|---|---|---|
| full (baseline) | 22,989 | — | 10,000 | 417 |
| skip global | 23,595 | **+3%** | 10,000 | 417 |
| skip vision | 24,422 | +6% | 10,000 | 417 |
| skip global+vision | 24,710 | +7% | 10,000 | 417 |

The `global` pass is ≈3% of per-batch cost; vision ≈3–6%; the two together ≈7%.
The residual ≈93% is the `prepare`+`kernel` dispatch (the fused brain). Fusion is
fully engaged (417 submits for 10 000 batches ≈ ⌈10000/24⌉), and baseline tps
≈23 k matches the originally observed ceiling — so removing submits did not move
it either. Both 0002's and 0004's targets are ruled out by this single table.

**Mesa lavapipe (`llvmpipe`, CPU software rasterizer) — NON-ADJUDICATING,**
retained for context. `--bench-phase-ab --bench-ticks 24000 --bench-agents 1`:
skip global +7%, skip vision +8%, skip both +13% — likewise no single pass
dominates, but CPU-rasterizer numbers do not transfer to the discrete GPU.

## Why reject

1. **The gate explicitly failed.** The reopen condition was "`skip global` tps
   jumps *materially*." +3% on target is not material; the entire `global` pass
   is a ~3% cost, not the dominant floor.
2. **The bottleneck is elsewhere.** ≥93% of per-batch time is the fused kernel
   pass; no amount of `global`-pass parallelization addresses it. Building B/C
   would risk the determinism gate for ≤3%.
3. **Measure-before-build paid off.** This is precisely the speculative GPU
   rewrite the plan gated; the on-target table shows it was the wrong target.

## When to revisit

## When to revisit

Effectively never, on current evidence — the `global` pass is a ~3% cost on the
target GPU. Only reopen if the world/grid model changes such that the `global`
pass's serial cost grows materially (e.g. a much larger collision grid or far
higher agent density), in which case re-run:

```
./target/release/xagent --bench-phase-ab --bench-ticks 1000000 --bench-agents <N>
```

and reopen only if the **`skip global`** arm then jumps materially over baseline
while `skip vision` does not. The construction sketch is preserved in
`ARCHITECTURE.md` §0004.

The actual lead this run surfaced is the **fused kernel/brain pass** (≥93% of
per-batch time, agent-count-independent → under-occupied / latency-bound serial
brain chain at low agent counts). That is a new plan's subject, not this one.
