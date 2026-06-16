# Plan 0006 - Multi-Workgroup Brain Parallelism - status

Task-level execution status for this plan. Keep it current as tasks land, and keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete (prove-or-kill: 60k **MISSED**, default **+96%**) · 10/10 owners addressed · authored against `develop` @ `069ff8a`, implemented on `plan/0006-multi-workgroup-brain-parallelism`.
_Last updated: 2026-06-16, against `plan/0006-multi-workgroup-brain-parallelism`._

**Pivot (0003 gate) + decision:** split-serial overhead measured **15.4% > 15%**
locked gate. The two locked decisions conflicted (overhead off-ramp vs the
exhaustive-work-list / non-speculative-answer rule). **Decision (operator, 2026-06-16):
BUILD the multi-workgroup split path** for a definitive, measured answer. **Verdict:
the multi-workgroup hypothesis is FALSIFIED** — `ParallelTiled` is slower than fused
at N=10 (−6%) and N=200 (−28%); the per-cycle dispatch overhead exceeds the
parallelism gain. The +96% throughput win came entirely from **same-dispatch
(in-workgroup) tiling** of every dense loop + thread-0 chokepoint, all in the fused
default path.

- **Goal:** Reach 60 k raw-tps at N=10.
- **Outcome:** **TARGET MISSED (45,506 tps = 75.8% of 60k); default +96% (23,230 → 45,506).** Owner of the gap = the **10-workgroup occupancy ceiling at N=10** (irreducible matrix-vector FLOPs + barriers at ~25% GPU utilization; more workgroups needs the split, which the dispatch overhead blocks). Default stays `FusedSerial` (it is the fastest at both N=10 and N=200); `parallel-tiled` is an opt-in/diagnostic mode. Learning unchanged (probes hold the chance baseline; fused vs tiled Food/1k ≈ 0.27 within noise). Full analysis: [`0006-60K-CLOSURE.md`](0006-60K-CLOSURE.md).

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Throughput budget and no-go map | `n10-throughput-budget-baseline` | ✅ CONTINUE (floor 159,634 ≥ 90k) |
| 0002 | Same-dispatch cooperative tiling | `same-dispatch-dense-tiling` | ✅ N=10 **+46%** (23,230→34,001), N=200 +85% |
| 0003 | Split-cycle execution scaffold | `split-serial-cycle-scaffold` | ✅ byte-identical; split overhead **15.4% > 15%** → operator chose to build split for definitive answer |
| 0004 | Multi-workgroup dense brain phases | `scratch-buffer-and-feature-phase` ✅, `multi-workgroup-encode-and-credit` ✅ (−3.7%), `multi-workgroup-predictor-and-action` ✅ (−1.3%), `parallel-reduce-action-tail` ✅ (**+21%**), `multi-workgroup-memory-reinforcement` ✅ (**+8.4%** + 7c dot +1.8%) | ✅ ParallelTiled built+measured (slower than fused); same-dispatch action-tail/memory tiling drove the +96% |
| 0005 | Non-brain floor recovery | `fused-food-grid-detect-floor-recovery` | ✅ measured non-owner (food in the 157,910 floor, 25× target; gridifying can't help 60k + risks the eat/shaping learning signal → measured negative recorded) |
| 0006 | 60 k closure | `sixty-k-throughput-closure` | ✅ TARGET MISSED; owner named; default unchanged; [`0006-60K-CLOSURE.md`](0006-60K-CLOSURE.md) |

**Design contract:** Default fused behavior preserved (bounded-drift, learning-probe-gated); every branch shipped measured speed or recorded a measured negative; no open-ended optimization left without a decision. The multi-workgroup split was built and measured (not assumed) so the 60k miss names a real owner.
