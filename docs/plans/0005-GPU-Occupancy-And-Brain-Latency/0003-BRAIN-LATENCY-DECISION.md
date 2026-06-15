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

**Micro-fix ruled out by the on-target profile — measured negative recorded; no
bit-identical transformation shipped. The real lever is a cooperative restructure
of `predict_and_act`, deferred to a follow-up.** The gate (the 0002 per-pass
profile + subgroup fact) was satisfied on macOS/Metal on 2026-06-15, and it
falsified the plan's premise:

- **Candidate #1 (force subgroup top-K): dead.** `recall_topk` costs ~0.27s
  (~1.6% of the brain). The subgroup path is inactive on target, but forcing it
  would recover ~1%. The whole "bitonic sort is the hotspot" hypothesis is wrong.
- **Candidate #2 (prune barriers / parallelize element-wise): marginal + risky.**
  The expensive work in the dominant passes is *reductions* (`fwd/trn/val_norm_sq`,
  policy eval, `atten_sum` in `predict_and_act`), which are float-order-locked —
  a parallel reduction changes accumulation order and breaks
  `deterministic_across_batch_sizes`. The genuinely element-wise loops are a
  minority and would need new barriers splitting the credit-assignment function.
- **Candidate #3 (thread-0 serialization): partly applies, but not micro.**
  `predict_and_act`'s thread-0 block does its reductions one global element at a
  time with 255 threads idle (no latency hiding). Converting them to the
  cooperative-partial pattern the value head already uses (parallel partials →
  barrier → thread-0 sum *in the same order* = bit-identical) would hide that
  latency — but it is a multi-barrier restructure of the most correctness-critical
  pass, not the single bounded transformation 0003 was scoped to.
- **`learn_and_store` is already cooperative dense compute** (encoder-credit
  update, memory cosine sims, decay across 128 threads). No serialization to
  remove; the cost is genuine arithmetic.

This satisfies the ship-or-record rule via the "record the measured negative"
branch: the bounded micro-fix has no safe, meaningful win. The dense-compute cost
the profile exposes is the deferred multi-agent/cooperative redesign's subject
(SCOPE, Out of scope), now justified by measurement — it should be authored as a
follow-up plan, not forced into this workstream.

## Measured profile (on target — macOS/Metal, 2026-06-15)

`XAGENT_KERNEL_PASS_LIMIT` sweep, `--bench-ticks 200000 --bench-agents 200`:

| limit | pass added | wall | tps | Δ wall = pass cost |
|---|---|---|---|---|
| 0 | — (no brain) | 1.66s | 120,165 | floor |
| 1 | feature_extract | 1.66s | 120,501 | ~0.00s |
| 2 | **encode** | 3.61s | 55,415 | **+1.95s** |
| 3 | habituate_homeo | 3.60s | 55,537 | ~0.00s |
| 4 | recall_score | 3.87s | 51,672 | +0.27s |
| 5 | recall_topk | 4.14s | 48,334 | +0.27s |
| 6 | **predict_and_act** | 9.59s | 20,853 | **+5.45s** (survival-confounded) |
| 7 | **learn_and_store** | 16.73s | 11,957 | **+7.14s** (survival-confounded) |

Subgroup top-K path: **inactive** (`workgroup-memory bitonic fallback`).

Confound: passes 5–6 let agents act/survive, so more agents stay alive and do
work — the +5.45s/+7.14s are upper bounds, not pure pass compute. Deltas through
`recall_topk` are clean (same non-acting-brain regime), so `encode`≈2s and
`recall`≈0.5s are solid. Full table + reading in the Plan 0005 subsection of
`docs/superpowers/specs/2026-06-10-learning-baseline.md`.

## Reopen / follow-up

A follow-up plan should: (1) re-run the pass sweep with death/respawn churn
suppressed to separate `predict_and_act` from `learn_and_store` cleanly, then
(2) convert `predict_and_act`'s thread-0 serial reductions to the cooperative
partial-product pattern (bit-identical, same accumulation order), measuring tps
at N=10 (latency regime) and at the knee. Multi-agent-per-workgroup packing is
the larger structural option if cooperative reduction alone is insufficient.

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

- 2026-06-15 (wiring) — gate instrumentation (pass-limit probe + subgroup
  accessor/log) merged, byte-identical when unset.
- 2026-06-15 (resolved) — **Closed as measured negative.** On-target profile
  obtained; it falsified candidate #1 (top-K is ~1.6%, not the hotspot) and
  showed the cost is the dense NN passes (`learn_and_store` > `predict_and_act` >
  `encode`), whose reductions are float-order-locked. No bit-identical micro-fix
  ships. The real lever — a cooperative restructure of `predict_and_act`'s
  thread-0 reductions — exceeds this workstream's "one bounded transformation"
  scope and is handed to a follow-up plan. Working tree clean of any shader
  change.
