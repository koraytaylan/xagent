# Architecture — Plan 0022 (deltas)

> Edits in `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/common.wgsl`,
> `crates/xagent-sandbox/src/governor.rs`,
> `crates/xagent-sandbox/tests/integration.rs`,
> `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/STATUS.md`,
> `docs/plans/STATUS.md`, and the in-folder decision/baseline docs
> `CORTEX-OCCUPANCY-BASELINE.txt`, `CORTEX-THROUGHPUT-AFTER-RESTRUCTURING.txt`,
> `0022-CORTEX-ENCODER-PATH-DECISION.md`, `0022-CORTEX-PERMANENCE-DECISION.md`.
> Line numbers are hints; locate by symbol (grep for `coop_visual_cortex`,
> `BRAIN_WORKGROUP_SIZE`, `VISUAL_FEATURE_COUNT`, `GABOR_ORIENTATIONS`,
> `POOL_ROWS`, `cortex_throughput_meets_budget`, `is_software_adapter`).

## 0001 — Workgroup-Occupancy-Restructuring

Today `coop_visual_cortex` (`brain_passes.wgsl:548+`) dispatches one workgroup per
agent with `BRAIN_WORKGROUP_SIZE=256` threads (`brain_passes.wgsl:18`). The
Gabor/pooling stages iterate over the `VISUAL_FEATURE_COUNT=18` outputs
(`GABOR_ORIENTATIONS × GABOR_SCALES × POOL_ROWS × POOL_COLS = 2 × 1 × 3 × 3`,
`common.wgsl:75-76, 133-134, 145-146`), leaving 238 lanes idle. The per-stage loop
structure is `for (var i = tid; i < RETINA_PIXEL_COUNT; i += BRAIN_WORKGROUP_SIZE)`
(e.g. `brain_passes.wgsl:590`), so all 256 threads participate in retina fill and
DoG passes (both data-parallel), but only thread indices `[0..18)` produce outputs
in the Gabor stages — yielding ~7% occupancy on hardware with 256-lane warps and
~1.1% of fused-baseline throughput
(`0017-Credit-Path-And-Emergent-Visual-Evolution/CORTEX-PROFILE-BASELINE.txt:21`).
The budget test `cortex_throughput_meets_budget` (`integration.rs:714-720`) skips
this on real GPU with the rationale at `gpu_kernel.rs:492-503`, so the structural
limit is unverified on target hardware.

This workstream is **measurement-gated and not yet locked to a single variant**.
Decision rule in SCOPE (locked decisions): a variant ships only if it reaches
≥50% occupancy on real GPU hardware while preserving filter quality; otherwise the
fallback (0003) is triggered. The three explored variants:

Edits:

- **Variant A — reduce `VISUAL_FEATURE_COUNT` toward a lane-width-aligned count**
  (`common.wgsl:75-76, 133-134`): lower `GABOR_ORIENTATIONS` from `2u` to `1u`
  (features `18 → 9`), or lower `POOL_ROWS`/`POOL_COLS` from `3u` to `2u` (features
  `18 → 8` at one orientation, `→ 8`–`12` rebalanced). The trade-off is that fewer
  filters narrow the feature space; the gain is measured occupancy.

```wgsl
// Variant A: one orientation halves the output feature count, raising the active
// fraction of the 256-lane workgroup. Filter quality (DC balance, orientation
// selectivity, phase tolerance) must be re-proven by the existing probes; this
// changes feature WIDTH only, not the Gabor kernel's correctness (pure function).
const GABOR_ORIENTATIONS: u32 = 1u; // was 2u — VISUAL_FEATURE_COUNT 18 → 9
```

- **Variant B — cooperative multi-agent workgroup** (`brain_passes.wgsl:40-50`,
  shared-memory `var<workgroup>` arrays; `brain_tick.wgsl` cortex call site):
  subdivide the 256 lanes across multiple agents (e.g. 4 agents × 64 lanes), each
  sub-group computing one agent's cortex. 64 lanes give `18/64 ≈ 28%` occupancy,
  far above 7%. Requires per-agent shared-memory offsets, per-sub-group barrier
  scoping, and agent-indexing arithmetic.

```wgsl
// Variant B: stage AGENTS_PER_WORKGROUP agents' scratch in one workgroup so the
// idle-lane tax is amortized across agents. Footprint scales linearly with the
// agent count and MUST stay under the per-workgroup shared-memory limit
// (~48 KB Metal, ~96 KB modern Vulkan) — see "Properties" below.
const AGENTS_PER_WORKGROUP: u32 = 4u; // 4 × 64 lanes; 18/64 ≈ 28% occupancy
var<workgroup> retina_scratch: array<f32, BRAIN_WORKGROUP_SIZE>; // → × AGENTS_PER_WORKGROUP
```

- **Variant C — shared-memory staging and lane-balancing** (new scratch buffers in
  `brain_passes.wgsl`): pre-stage Gabor responses cooperatively with all 256 threads
  loading in parallel, then pool in a second phase where fewer lanes are active.
  Breaks the low-occupancy monopoly by amortizing the cooperative load first;
  occupancy rises only during the pool stage. Requires new synchronization.

Properties that make restructuring safe:
- The cortex is **flag-gated** (`visual_cortex_enabled`, default-off), so the
  shipped default encoder path is untouched; every variant applies only behind the
  flag and cannot regress the production simulation.
- Filter properties (DC balance, phase invariance, orientation selectivity) are
  invariant-checked by the probes `gabor_kernels_are_dc_balanced`,
  `complex_cell_selectivity_*`, `complex_cell_phase_tolerance_*`
  (`integration.rs:2880+`). The Gabor kernels are pure functions of position and
  frequency; changing the output count alters feature *width*, not correctness, so
  the probes either pass unchanged or pin the measured regression.
- Any feature-count change must mirror `VISUAL_FEATURE_COUNT` on the Rust side
  (`governor.rs` and the buffer-layout / feature-count validation) so the
  Rust↔WGSL parity check holds; a mismatch is caught before the kernel runs.
- Shared-memory footprint grows with Variant B's agent multiplier and must stay
  under the per-workgroup limit (~48 KB Metal, ~96 KB modern Vulkan); the
  multi-agent prototype reports the compiler's footprint before any commit.

## 0002 — Occupancy-Improvement-Validation

Today the budget test `cortex_throughput_meets_budget` (`integration.rs:703+`)
asserts ≥50% of fused baseline only on the lavapipe software adapter and skips on
real GPU via `is_software_adapter()` (`integration.rs:714-720`,
`gpu_kernel.rs:492-503`). The 0017 STATUS row therefore reports "Done / ≥50%
budget" on a software adapter while the target-hardware figure is ~1.1%
(`0017-Credit-Path-And-Emergent-Visual-Evolution/STATUS.md:42-43`) — the
gate-integrity regression flagged as P1 in `2026-06-25-glm-52.md` F2.

Edits:

- **Run the budget assertion on real GPU after 0001 lands**: modify
  `cortex_throughput_meets_budget` (`integration.rs:714-720`) so the
  `is_software_adapter()` branch no longer unconditionally skips. When occupancy is
  ≥50%, the test asserts on hardware; when below, it stays gated-skip but records
  the measured percentage for the fallback. Keep the no-adapter skip for local runs
  with no GPU.

```rust
// After restructuring, the ≥50% budget must hold on REAL GPU, not just lavapipe.
// fraction_of_baseline = cortex_tps / fused_baseline_tps, measured per backend.
assert!(
    fraction_of_baseline >= 0.50,
    "Cortex throughput {:.2}% of fused baseline fails the >=50% budget",
    fraction_of_baseline * 100.0,
);
```

- **Record the per-backend measurement** in
  `CORTEX-THROUGHPUT-AFTER-RESTRUCTURING.txt` (Metal minimum, RDNA / NVIDIA if the
  test infrastructure permits), so the budget claim is backend-aware and auditable.
- **Update `0017 STATUS.md` WS0002** to state the measured occupancy and throughput
  and whether the budget is met on hardware (replacing the lavapipe-only claim), and
  whether the encoder is cleared for A/B consideration.
- **Record the encoder-path decision** in `0022-CORTEX-ENCODER-PATH-DECISION.md`:
  the measured occupancy and throughput, an exploratory cortex-ON vs cortex-OFF
  fitness/steering signal, and whether to recommend flipping `visual_cortex_enabled`
  default or keep it gated pending the emergence A/B.

Properties that make this safe:
- The assertion is the only behavioral change in this workstream; it is a stricter
  gate, not a new code path, and falls back to a documented skip (never a silent
  pass) when occupancy is below target.
- Backends are evaluated independently: the budget is "met" only if it holds on at
  least one real backend; a per-backend failure (e.g. NVIDIA passes, Metal fails)
  is deemed not-met and routes to the fallback (0003), per the locked decision.
- The exploratory fitness/steering run is measurement-only on a fixed config — it
  recommends a default flip or a follow-on plan, it does not flip the flag here.

## 0003 — Cortex-Permanence-Fallback-Decision

Today the cortex is already flag-gated (`visual_cortex_enabled`, default-off), so
the fallback case needs **no runtime code change** — it is a planning-direction and
documentation-clarity outcome. The root board and the 0017 STATUS still imply the
cortex is on-track for a default path
(`docs/plans/STATUS.md` row 0017, `2026-06-25-glm-52.md` F2), which is the
misleading state this workstream closes when restructuring does not reach ≥50%.

Edits (documentation only):

- **Author `0022-CORTEX-PERMANENCE-DECISION.md`** (numbered to this workstream):
  the measured occupancy ceiling, the structural reason it cannot reach 50% (feature
  count 18 vs workgroup size 256 is not addressable by dispatch restructuring
  alone), the rationale for keeping the cortex permanently gated, the fallback
  encoder choice (legacy raycast vs. simpler learned baseline), and the revisit gate
  (e.g. native WGSL subgroup operations that allow sub-256-lane dispatch, or a
  future backend with higher occupancy).
- **Update `docs/plans/STATUS.md` row 0017** and the plan's own
  `0017-Credit-Path-And-Emergent-Visual-Evolution/STATUS.md` to the final outcome:
  cortex optimization complete but occupancy-limited on GPU hardware, decision
  resolves to permanent GPU-gating, emergence line pivots to the legacy encoder,
  A/B deferred pending credit-path unlock and fallback-encoder selection.

Properties that make this safe:
- No code, shader, or buffer-layout change — the cortex implementation (Gabor
  kernels, DoG, pooling) stays intact and probe-proven behind its flag, available
  for off-default research; only documentation and planning direction change.
- The decision is reversible by an explicit gate, not abandoned: the revisit
  condition is recorded so a future structural change reopens a *separate* plan
  rather than re-litigating this one.
- Both STATUS tiers are updated in the same change, so the root roll-up and the
  per-plan detail cannot disagree about the cortex's terminal state.
