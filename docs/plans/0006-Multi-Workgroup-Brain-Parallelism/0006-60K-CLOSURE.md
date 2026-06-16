# Plan 0006 — 60 k closure decision

**Decision: TARGET MISSED — remaining owner is the N=10 GPU occupancy ceiling (10 workgroups).**

Measured on Apple M3 Max / Metal 4. The 60 k raw-tps target at N=10 was not
reached; the fused default plateaus at **45,506 tps (75.8% of 60 k)**. The plan's
core hypothesis — that crossing the workgroup boundary (multi-workgroup split per
agent) unlocks 60 k — was **falsified by measurement**: the `ParallelTiled` split
path is *slower* than fused at every population. The work done in pursuit of it,
however, nearly **doubled the fused default throughput (+96%)** by exhausting the
same-dispatch (in-workgroup) parallelism of every dominant brain loop.

## Final target matrix (M3 Max, `--bench --bench-ticks 1000000 --bench-agents 10`; sweep `--bench-ticks 200000`)

| run | N=10 tps | N=200 agent-ticks/sec |
|---|---:|---:|
| **FusedSerial (default, post-plan)** | **45,506** | **4,633,798** |
| ParallelTiled (multi-workgroup split) | 42,780 (−6.0%) | 3,321,576 (−28.3%) |
| no-brain floor (`XAGENT_KERNEL_PASS_LIMIT=0`) | 157,910 | — |
| pre-plan fused baseline | 23,230 | 2,397,914 |

`ParallelTiled` regresses vs fused at both N=10 (−6%) and N=200 (−28%), so the
default **stays `FusedSerial`**; `XAGENT_BRAIN_EXECUTION_MODE=parallel-tiled`
remains available as an opt-in/diagnostic mode.

## The same-dispatch trajectory that delivered +96% on the default

Each step tiled a dense loop or a thread-0 serial chokepoint inside the single
fused workgroup (no dispatch overhead). The gains collapse as the work runs out
— the signature of an occupancy ceiling, not a missing optimization:

| after | N=10 tps | gain | what it tiled |
|---|---:|---:|---|
| pre-0002 baseline | 23,230 | — | — |
| `same-dispatch-dense-tiling` (0002) | 34,001 | +46.4% | encode, predictor, encoder-credit dot loops (128→256 lanes, 4-lane reduce) |
| `parallel-reduce-action-tail` (0004d) | 41,200 | +21.3% | thread-0 action-tail reductions → tree reductions |
| `multi-workgroup-memory-reinforcement` (0004e) | 44,680 | +8.4% | learn_and_store: e_norm-once, parallel store + argmin |
| 7c reinforcement dot (memory reinf complete) | 45,506 | +1.8% | last half-occupancy per-pattern dot |

## Split-dispatch overhead (why multi-workgroup loses)

The `split-serial-cycle-scaffold` (0003) measured the *pure* dispatch tax of
running the same serial brain one cycle per dispatch (10 dispatches/batch, no
tiling): **15.4% slower** than fused — already over the locked 15% gate. The
operator chose to build the multi-workgroup path anyway to obtain a
non-speculative answer. `ParallelTiled` issues ~6 dispatches/cycle (≈60/batch,
10,000 submits vs fused's 417). The multi-workgroup tiling *recovers* most of
that tax (encode+predictor split tracked fused to within −1.3% at one point),
but never beats it: at N=10 the GPU is dispatch/occupancy-bound, not throughput
-bound, so spreading work across more workgroups costs more in per-dispatch
overhead than it saves. The same parallelism, applied *in the same dispatch*
(no boundary), is strictly better — hence the +96% came entirely from
same-dispatch tiling.

## Correctness / drift

- `ParallelTiled` is **deterministic within mode** (byte-equal physics +
  brain_state + pattern_buffer across batch decompositions —
  `parallel_tiled_deterministic_across_batch_sizes`).
- **Bounded-drift vs fused**: finite state, motor outputs in [−1, 1], identical
  alive/death counts over the fixed-seed smoke
  (`parallel_tiled_bounded_drift_vs_fused`).
- The same-dispatch tilings change reduction order (tree vs serial), so they are
  bounded-drift, not byte-identical, against the *pre-plan* fused path. The
  argmin/active-count tiling uses a lexicographic tie-break that is byte-EXACT
  vs the serial scan. `SplitSerial` stays byte-identical to `FusedSerial`
  (`split_serial_matches_fused_serial`), and `deterministic_across_batch_sizes`
  + `fused_dispatch_matches_split` stay green.

## Fixed-seed learning comparison (seed 42, population 10, 3 generations)

The bounded-drift mode change does not regress learning:

| arm | Food/1k (gen 0/1/2) | best fitness (gen 0/1/2) | deaths/food (gen 0) |
|---|---|---|---|
| FusedSerial | 0.271 / 0.271 / 0.290 | 0.0871 / 0.0922 / 0.0864 | 1.33 |
| ParallelTiled | 0.253 / 0.284 / 0.274 | 0.0848 / 0.0852 / 0.0864 | 1.24 |

Differences are within single-seed noise. The falsifiable learning pins
(`learning_probe_mirrored_steering_is_chance` et al.) stay green at the chance
baseline for the fused default — the tilings preserved learning behavior.

## The remaining owner of the gap (non-speculative)

To hit 60 k (16.67 s / 1M ticks) the brain must shrink to ~10.4 s (floor 6.33 s).
Post-plan the brain is ~15.6 s (21.97 s − 6.33 s). Every dominant dense loop is
already tiled across all 256 lanes; the residual is **irreducible matrix-vector
work** (encode 265×128, predictor 128×128, recall 128×128, reinforce 128×128,
the recall top-K bitonic sort) plus the per-pass `workgroupBarrier()` chain,
running in **only ~10 workgroups at N=10 ≈ 25% GPU utilization**. The only lever
left is more *workgroups*, which requires the multi-workgroup split — and the
measured split-dispatch overhead (§ above) makes that net-negative at this tiny
work size. **The owner of the 60 k gap is GPU occupancy at N=10: 10 agents cannot
saturate the M3 Max, and the per-agent brain cannot be split across more
workgroups without paying dispatch overhead that exceeds the parallelism gain.**

## Owner-by-owner accounting (exhaustive work list)

| Owner | Status | Evidence |
|---|---|---|
| encode | ✅ tiled (0002, 0004b) | part of +46% |
| predictor train+predict | ✅ tiled (0002, 0004c) | part of +46% |
| encoder-credit learning | ✅ tiled (0002, 0004b) | part of +46% |
| action-tail reductions/copies | ✅ tiled (0004d) | **+21.3%** |
| memory reinforcement | ✅ tiled (0004e + 7c dot) | **+8.4% / +1.8%** |
| brute-force food scan | ◻️ measured non-owner | in the 157,910-tps floor (25× the target); the brain is the bottleneck, not the floor — gridifying it cannot help 60 k and (needing a 9×9 neighborhood for the 30-unit shaping radius) would add per-batch grid-staleness behaviour change to the eat/shaping learning signal for zero throughput gain. Addressed as a measured negative grounded in the 0001 floor. |

## When to revisit

- 60 k at N=10 is reachable only by reducing per-agent brain compute, not by
  parallelizing it further: e.g. f16/quantization, smaller `ENCODED_DIMENSION`,
  eliminating the redundant reinforce-vs-recall similarity recompute, or a
  cheaper recall (drop/approximate the bitonic top-K). All are out of this
  plan's scope (GPU parallelization of the *current* workload).
- The multi-workgroup split would become attractive only if a future workload
  has far fewer, far heavier per-agent dispatches (amortizing the per-dispatch
  tax), or on hardware with much lower compute-pass overhead.
