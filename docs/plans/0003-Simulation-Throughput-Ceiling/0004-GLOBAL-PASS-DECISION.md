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

**Closed — not opened. The gate is unevaluated on target hardware and therefore
not met; no `global`-pass rewrite ships under Plan 0003.** This satisfies the
"record the negative and close the workstream" branch of the spike's `Done when`,
with the explicit caveat that the negative is *"gate not demonstrated on target
hardware,"* not *"global pass proven cheap on target hardware."* The path is
reversible by the concrete condition in [When to revisit](#when-to-revisit).

The fused-submit work (0002) already removed the per-batch submit tax — the
demonstrable, hardware-independent win (240 kernel-batches → 10 submits at
`--bench-ticks 24000`). The `global`-pass rewrite is a *separate, larger* GPU
change (multi-workgroup grid-clear/build, plus cross-cell-dependent collision
atomics) and is only worth its risk if a target-hardware measurement fingers the
`global` pass as the residual floor. That measurement does not yet exist.

## The three paths

| Path | What it is | Cost it removes | Risk / cost to build |
|---|---|---|---|
| A — leave single-workgroup | Status quo: `global_tick` runs on one 256-thread workgroup, `dispatch(1,1,1)`. | Nothing. | None. |
| B — parallelize grid-clear + grid-build only | Dispatch `ceil(grid_cells / 256)` workgroups for `phase_clear` / `phase_food_grid` / `phase_agent_grid`; keep collision single-workgroup. | The ≈42 k serial grid stores per batch, if they dominate. | Medium: the grid build is embarrassingly parallel, but it must be split out of the fused `global` pass into its own dispatch(es) with correct barriers; collision still serial. |
| C — fully parallelize, incl. collision | B plus multi-workgroup collision accumulate/apply. | All `global`-pass serial cost. | High: collision carries cross-cell read/write dependencies and atomics; multi-workgroup correctness is non-trivial and easy to get subtly wrong (determinism gate). |

The spike, if opened, prototypes **B** (grid-build only), measures the per-batch
GPU-complete delta against the 0001/0002 baselines, and decides whether **C** is
worth a follow-up plan. See `ARCHITECTURE.md` §0004 for the construction sketch.

## Measured evidence

**On-target (macOS/Metal or discrete GPU): none yet.** This is the gap that
keeps the gate unevaluated.

**Mesa lavapipe (`llvmpipe`, CPU software rasterizer) — NON-ADJUDICATING.**
`--bench --bench-ticks 24000 --bench-agents 1`, one run each (full table in
`docs/superpowers/specs/2026-06-10-learning-baseline.md` → *Simulation throughput
ceiling*):

| Arm | Env | tps | submits | submit-return ns/batch |
|---|---|---|---|---|
| (a) default | — | 10,754 | 10 | 8,794,589 |
| (c) skip global+vision | `XAGENT_SKIP_GLOBAL_VISION=1` | 22,138 | 10 | 4,059,447 |

Arm (c)'s ~2× tps jump on lavapipe is **not** usable as the gate signal: lavapipe
executes every shader on the CPU, so skipping global+vision removes CPU compute
work, which is a different quantity from a discrete GPU's submit/back-pressure or
single-workgroup-occupancy cost. The knob also skips *both* the global and vision
passes together, so even a representative jump would not isolate the `global`
pass alone without a vision-only control. lavapipe is used here only to prove the
probe is wired correctly, not to decide this gate.

## Why reject (not-open) now

1. **The gate is target-specific and unmeasured.** The ≈20 k-tps / 1000× ceiling
   was observed on macOS/Metal; only a Metal (or other discrete-GPU) three-arm
   run can show whether the `global` pass dominates the residual. This
   environment has only lavapipe.
2. **0002 already banked the safe, large win.** Submit fusion is bit-identical
   and removed the per-batch submit tax (the mechanism the review located). The
   `global`-pass rewrite is strictly higher-risk and should not be undertaken on
   speculation.
3. **Reversible.** Opening the spike later costs nothing that is lost by waiting;
   shipping a speculative multi-workgroup collision rewrite risks the determinism
   gate (`deterministic_across_batch_sizes`) for an unquantified gain.

## When to revisit

Reopen `parallelize-global-pass-spike` when **both** hold on the target hardware
(macOS/Metal or a discrete GPU), measured per the repro in the baseline spec:

- Arm (c) `XAGENT_SKIP_GLOBAL_VISION=1` tps jumps **materially** over arm (a)
  default at 1000× (the `global`+`vision` passes are the residual floor), **and**
- A vision-only control (skip vision but keep global, or vice-versa) attributes
  the bulk of that jump to the **`global`** pass specifically rather than vision.

If, instead, arm (b) `XAGENT_PROBE_GPU_WAIT=1` shows submit-return ≪ gpu-complete
with the global pass *not* dominating, the residual is Metal back-pressure / GPU
execution elsewhere and this workstream stays closed. Construction sketch for the
reopen: `ARCHITECTURE.md` §0004.
