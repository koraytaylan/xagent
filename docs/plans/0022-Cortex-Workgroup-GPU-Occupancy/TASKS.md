# XAgent Plan 0022 — Cortex-Workgroup-GPU-Occupancy

This plan either restructures the cortex dispatch from a 256-lane workgroup with 18 active threads to a layout that increases lane occupancy (e.g., smaller per-thread filters, alternate memory layout, cooperative loading), re-measures throughput on real GPU hardware, and asserts ≥50% of fused baseline, or records an explicit decision to keep visual_cortex_enabled permanently default-off, ship the cortex as a research artifact behind its flag, and pivot the emergence line back to the legacy raycast encoder (demoting Hubel-Wiesel 0008 from default pathway). The decision follows a measurement gate that determines occupancy gain is not achievable by configuration alone and requires architectural restructuring.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Workgroup-Occupancy-Restructuring

### occupancy-baseline-measurement — Profile Current Occupancy on Real GPU Hardware

Before attempting restructuring, establish the current occupancy profile as a baseline. The cortex's 256-lane workgroup with 18 active threads is measured on CPU (lavapipe is all-lanes-active, hiding the occupancy problem). Real GPU hardware (Metal, RDNA, NVIDIA) will show the actual lane utilization during the Gabor/pooling stages. This measurement gates all restructuring options: if occupancy can be improved by config changes alone (e.g. increasing feature count via more Gabor scales), cheaper variants are tried first. If occupancy is locked by the dispatch structure, more invasive restructuring (multi-agent workgroup) becomes necessary.

**Steps:**
1. Write a GPU-profiling test in `crates/xagent-sandbox/tests/integration.rs` that runs the cortex on a fixed agent for N frames and collects GPU performance counters (occupancy %, active lanes, stall reasons if the backend supports it). Use `GpuKernel::is_available()` to gate the test and `GpuKernel::is_software_adapter()` to skip on lavapipe (only profile on real GPU).
2. Implement occupancy readback using wgpu's performance-query API (if available on the test backend) or fall back to wall-clock measurement + instruction-count estimation. Document the methodology (e.g. 'measured via GPU profiler on Metal M3 Max; instruction cache behavior dominates variance; 5-frame warm-up before capture').
3. Run the test locally on Metal (required), and if CI/test infrastructure permits, on AMD RDNA and NVIDIA hardware. Record the measured occupancy percentage (active lanes / 256) for each stage of the cortex (retina fill, DoG passes, Gabor/pooling).
4. Assert that the measured occupancy during the Gabor/pooling stages matches the theoretical 18/256 ≈ 7%, confirming the structure is the bottleneck, not a measurement artifact. Document the per-stage breakdown in a baseline text file (e.g. `CORTEX-OCCUPANCY-BASELINE.txt`, parallel to 0017's throughput profile).

- **Depends on:** —
- **Done when:** The occupancy profile is measured on real GPU hardware (Metal minimum, others if available); the baseline shows Gabor/pooling at ≈7% occupancy; per-stage results documented in a `CORTEX-OCCUPANCY-BASELINE.txt` file in the plan folder; the test is GPU-gated and skips on software adapters; cargo fmt/clippy/test green.

---

### variant-a-feature-count-reduction — Explore Variant A: Reduce VISUAL_FEATURE_COUNT and Re-Validate Filters

Variant A reduces the Gabor/pooling output feature count from 18 (2 orientations × 1 scale × 3 rows × 3 cols) to a count closer to hardware lane-width (e.g. 16, 12, 8). Options: lower `GABOR_ORIENTATIONS` from 2 to 1 (cuts features to 9), lower `POOL_ROWS`/`POOL_COLS` from 3×3 to 2×2 (cuts features to 4 if GABOR_ORIENTATIONS stays 2, or 2 if reduced to 1). This directly increases occupancy: 16/256 = 6.25% (still poor), 32/256 = 12.5%, 64/256 = 25% (approaching viability). However, reducing filter diversity or pooling granularity may degrade the learned feature space. This task implements one specific reduction and validates that the filter properties (DC balance, orientation selectivity, phase invariance) are preserved or acceptable. If occupancy improves but features degrade significantly, the fallback (WS0003) is more likely.

**Steps:**
1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, locate the constants defining VISUAL_FEATURE_COUNT and its components: `GABOR_ORIENTATIONS` (line 75), `GABOR_SCALES` (line 76), `POOL_ROWS` (line 133), `POOL_COLS` (line 134).
2. For Variant A, implement one of: (a) reduce `GABOR_ORIENTATIONS` from 2u to 1u (features from 18 → 9), OR (b) reduce both `POOL_ROWS` and `POOL_COLS` from 3u to 2u and keep orientations at 2u (features from 18 → 8). Update the const blocks and the doc-comments explaining the choice and its trade-off (e.g. 'Reduced to 8 features: occupancy improves from 7% to 3% (still inadequate); orientation selectivity expected to decline slightly due to fewer filters but remains >3× threshold').
3. Update `crates/xagent-sandbox/src/governor.rs` and any Rust validation that mirrors the feature count (search for `VISUAL_FEATURE_COUNT` in Rust; verify `feature_count` in the complex module and buffer-layout validation match the new value).
4. Re-run the orientation-selectivity, phase-invariance, and position-tolerance probes (`gabor_kernels_are_dc_balanced`, `complex_cell_selectivity_horizontal_bar`, `complex_cell_phase_tolerance_*` in `integration.rs:2880–2920` approximate range) to verify that filters still meet the thresholds: orientation selectivity ≥3.0× (threshold), phase invariance <10%, position tolerance <15%.
5. Measure the new occupancy (re-run `occupancy-baseline-measurement` if the constant change affects dispatch structure, or estimate via lane count ratio). Document the measured occupancy improvement (e.g. 7% → 9% if only pool granularity changed, or 7% → 25% if orientations are halved and pool is tightened).

- **Depends on:** occupancy-baseline-measurement
- **Done when:** One specific feature reduction is implemented and tested; the filter property probes pass (selectivity, phase invariance, DC balance); occupancy is re-measured and documented (e.g. 'Reduced POOL_ROWS/COLS from 3×3 to 2×2: features 18 → 8, occupancy 7% → 12.5%'); new feature count is validated in Rust and Rust-WGSL parity checks pass; the updated constants are doc-commented with the trade-off rationale; cargo fmt/clippy/test green.

---

### variant-b-multi-agent-workgroup — Prototype Variant B: Multi-Agent Workgroup Structure

Variant B reweights the dispatch to run multiple agents within a single 256-lane workgroup, subdividing lanes across (e.g.) 4 agents × 64 lanes each. Each sub-workgroup processes one agent's cortex independently, with 64 lanes providing adequate occupancy for the 18-feature Gabor bank (18/64 ≈ 28%, still low but significantly better than 7%). The trade-off is structural: requires rewriting shared-memory layout, agent-indexing arithmetic, and barrier management. This task prototypes the structure: implement a toy version (2 agents × 128 lanes, or 4 agents × 64 lanes), validate that occupancy is measured higher, and assess whether the code complexity and per-agent latency are acceptable. If occupancy and code maintainability are both acceptable, this becomes a full implementation (move to a change task). If complexity is too high or occupancy still <50%, the fallback (WS0003) becomes more likely.

**Steps:**
1. Sketch the proposed multi-agent workgroup structure in a doc comment in `brain_passes.wgsl`, describing: (a) how the 256 lanes are subdivided (e.g. 4 sub-groups of 64 lanes), (b) which shared-memory buffers are per-agent vs shared, (c) how barriers are placed (e.g. barriers per sub-group + global barrier), and (d) the agent-indexing formula for thread_id_in_sub_group.
2. Implement a prototype: modify `coop_visual_cortex` to accept an additional parameter `agents_per_workgroup` (hardcoded to 4 for the prototype, or 2) and adjust the agent loop to dispatch multiple agents per call. Rewrite the shared-memory arrays (`var<workgroup>` declarations, lines 40–50) to account for multiple agents' data in a single workgroup (e.g. `array<f32, BRAIN_WORKGROUP_SIZE>` → `array<f32, BRAIN_WORKGROUP_SIZE * 4>` for 4 agents, with indexing offset per agent).
3. Implement the per-sub-group barrier logic: use a manual loop-counting or subgroup-width calculation to ensure all agents' barriers align. Validate that the prototype compiles with the existing crate's WGSL version (no unsupported features).
4. In `brain_tick.wgsl`, modify the cortex call to invoke the multi-agent version: `coop_visual_cortex_multi(agent_id, tid, num_agents_this_dispatch)` (stub signature for now; the prototype is measurement-only).
5. Measure occupancy on GPU using the same profiling from `occupancy-baseline-measurement`. Document the measured occupancy with the new structure (expected ≈28–30% with 4 agents × 64 lanes, depending on how WGSL's barrier scoping allows vectorization).
6. Assess code complexity: count lines of new code, number of shared-memory barriers, register pressure (compiler reports), and local latency (wall-clock time per agent vs single-agent baseline). Write a summary (e.g. 'Multi-agent occupancy improves to 28%; shared-memory rewrite adds ~150 lines; per-agent latency increases by 12% due to barrier overhead; trade-off is moderate — recommend full implementation if occupancy-vs-latency ratio meets ≥50% throughput budget when applied to the full pipeline').

- **Depends on:** occupancy-baseline-measurement, variant-a-feature-count-reduction
- **Done when:** Prototype multi-agent workgroup is implemented and compiles; occupancy is measured on GPU and documented (e.g. 'Measured 28% occupancy with 4-agent structure'); per-agent latency and code-complexity trade-off summary is written; decision is made to proceed with full implementation or escalate to fallback (the task itself does not flip any flag or commit the prototype to develop; it is measurement + design review only); cargo fmt/clippy/test green (prototype code is localized and does not ship).

---

### occupancy-restructuring-selection — Select Occupancy Restructuring Path or Escalate to Fallback (GATED)

**Gate:** both probe tasks (`variant-a-feature-count-reduction`, `variant-b-multi-agent-workgroup`) must report their measured occupancy and trade-offs. The decision rule is: (a) if any variant achieves ≥50% occupancy on GPU hardware AND preserves filter quality (selectivity ≥3×, phase tolerance <10%), proceed to full implementation (move to WS0002 validation). (b) if no variant reaches ≥50% occupancy, or if all variants introduce unacceptable complexity or latency trade-offs, escalate to WS0003 (fallback decision: keep cortex GPU-gated, pivot emergence line).

After probing occupancy variants (Variant A: feature reduction, Variant B: multi-agent workgroup), assess which path is viable. This is a decision-record-only task: no changes to develop are committed; it activates exactly one of the two downstream gates (0002 or 0003).

**Steps:**
1. Collect the measured occupancy and performance/quality metrics from `variant-a-feature-count-reduction` and `variant-b-multi-agent-workgroup`. Populate a decision table: Variant A occupancy %, Variant B occupancy %, per-agent latency overhead, code-complexity (lines added, barriers, shared-memory footprint), filter-property regressions (selectivity, phase invariance).
2. Evaluate against the ≥50% budget gate: is any variant ≥50% occupancy on real GPU? If yes, proceed. If no, document the measured ceiling (e.g. 'Variant A bottoms out at 12%; Variant B at 28%; neither variant reaches 50% occupancy; root cause is the feature count (18) vs workgroup size (256) mismatch is not addressable by dispatch restructuring alone').
3. Evaluate code-complexity trade-off: if a variant ≥50% but adds >500 lines, increases register pressure significantly, or increases per-agent latency >20%, assess whether the complexity is acceptable for a flag-gated research artifact or whether the gain doesn't justify the maintenance burden.
4. Write a decision doc (1–2 paragraphs): state which variant is selected, why (occupancy target met + acceptable trade-off), or escalate to WS0003 with the rationale (occupancy ceiling does not reach 50%; fallback decision is to keep cortex GPU-gated and pivot emergence encoder).
5. If selecting a variant: create a new task card for WS0002 validation (below). If escalating: skip WS0001's remaining tasks and move directly to WS0003.

- **Depends on:** variant-a-feature-count-reduction, variant-b-multi-agent-workgroup
- **Done when:** Decision documented: either (A) Variant X selected for full implementation (occupancy ≥50%, trade-off acceptable, move to WS0002), with rationale; OR (B) escalated to WS0003 (occupancy ceiling <50%, no viable variant, fallback decision triggered). The decision is binary and clearly states which WS gate (0002 or 0003) is activated next. No changes to develop are committed; this is a decision-record-only task.

---

### variant-full-implementation — Full Implementation of Selected Occupancy Variant (GATED)

**Gate:** `occupancy-restructuring-selection` must determine that a variant meets the ≥50% occupancy target and merits full implementation. This task polishes and ships the selected variant (e.g. Variant A if feature reduction is selected, or Variant B if multi-agent structure is selected). All filter properties are re-validated, buffer layout is fully reconciled between Rust and WGSL, and the new structure is tested end-to-end.

**Steps:**
1. If Variant A selected: finalize the feature-count reduction in common.wgsl and governor.rs, update all feature-count validation checks, re-run all filter-property probes, and confirm orientation selectivity, phase invariance, and position tolerance still meet thresholds. Add a Rust-WGSL parity check (feature-count override must match common.wgsl const).
2. If Variant B selected: finalize the multi-agent workgroup structure, fully rewrite shared-memory layout and barrier logic, update agent-indexing in brain_tick.wgsl and brain_passes.wgsl, measure occupancy on GPU one final time, and confirm per-agent latency and register pressure are acceptable. Add a test that verifies multi-agent coherence (e.g. run two agents in a shared workgroup and confirm their cortex outputs match a single-agent run).
3. For either variant: run the full integration test suite (including all existing cortex probes) to ensure no regressions.
4. Document the change in the ARCHITECTURE.md for this plan, explaining the restructured dispatch structure, the measured occupancy improvement, and the trade-offs accepted.

- **Depends on:** occupancy-restructuring-selection
- **Done when:** Selected variant is fully implemented and integrated; filter properties and occupancy are re-validated on GPU; Rust-WGSL parity checks pass; all existing cortex probes pass; cargo fmt/clippy/test green.

---

## 0002 — Occupancy-Improvement-Validation

### cortex-throughput-validation — Re-Measure Cortex Throughput on Real GPU and Assert ≥50% Budget (GATED)

**Gate:** `variant-full-implementation` must complete and occupancy must be measured ≥50%. This task re-runs the throughput budget assertion (`cortex_throughput_meets_budget`, `integration.rs:703+`) on real GPU hardware (Metal, RDNA, NVIDIA if available) with the restructured cortex, asserts that ≥50% of fused baseline is achieved, and updates the 0017 STATUS to reflect the final outcome. If any backend fails the ≥50% gate, the fallback (WS0003) is triggered instead.

**Steps:**
1. Modify `cortex_throughput_meets_budget` to run on real GPU hardware: remove or modify the `is_software_adapter()` skip condition to allow the test to run on Metal/RDNA/NVIDIA. Keep a fallback skip only if no adapter is present (preserve graceful degradation for local runs without GPU).
2. Run `cortex_throughput_meets_budget` locally on Metal (required), and if CI infrastructure permits, on AMD RDNA and NVIDIA backends. Measure the throughput (ticks per second) and compare against the fused baseline.
3. Assert `assert!(fraction_of_baseline >= 0.50, "Cortex throughput {:.2}% of baseline fails ≥50% budget", ...);` on each backend tested.
4. Document the measured throughput in a new baseline file (e.g. `CORTEX-THROUGHPUT-AFTER-RESTRUCTURING.txt`) showing per-backend results.
5. If the assertion fails on any backend, do not commit the change; instead, escalate to WS0003 (occupancy is still below 50%, fallback decision is triggered).
6. If the assertion passes on all tested backends (minimum Metal, all if available): update the 0017 STATUS.md WS0002 row to state the measured occupancy and throughput, confirming the budget is met on GPU hardware.

- **Depends on:** variant-full-implementation
- **Done when:** On GPU hardware (Metal minimum, all backends if available), `cortex_throughput_meets_budget` asserts ≥50% of fused baseline throughput; the measured throughput is documented in a baseline file; the 0017 STATUS.md WS0002 row is updated to reflect occupancy achieved and budget met on hardware (not just lavapipe); cargo fmt/clippy/test green. If the assertion fails, escalate to WS0003 instead (do not commit restructuring to develop; document the measured ceiling as the reason for fallback).

---

### cortex-encoder-path-decision — Decide Cortex Default Status and Emergence Encoder A/B Path (GATED)

**Gate:** `cortex-throughput-validation` must pass (≥50% budget confirmed on GPU hardware). Given that the cortex occupancy is now ≥50% and meets the throughput budget, the question is whether to flip `visual_cortex_enabled` default to true (making it the default encoder) or keep it flag-gated and proceed with the emergence encoder A/B (0017 WS0004). This is a decision task: measure the encoder's fitness/intent impact on a few production seeds, and record whether the cortex should be promoted to default or kept gated pending the emergence encoder comparison.

**Steps:**
1. With the restructured cortex meeting ≥50% budget, run a small-scale exploratory A/A (1–2 seeds, 10 generations, pop 10, fixed config) comparing cortex ON vs cortex OFF. Measure: fitness (foraging + exploration), steering alignment (mirrored-steering probe), intent fractions (approach, avoidance). The goal is not a definitive A/B gate (full paired harness comes later), but a signal: does the cortex improve fitness/steering, or is it neutral/regressive at the feature level?
2. Document the exploratory results: fitness delta, steering delta, intent trajectory.
3. Record the decision: (a) if cortex improves fitness/steering, recommend flipping `visual_cortex_enabled` default to true (requires a follow-on plan or decision approval from the project leads), OR (b) keep default off and proceed with emergence encoder A/B, using the cortex as one comparison arm (0017 WS0004 unfolds).

- **Depends on:** cortex-throughput-validation
- **Done when:** Exploratory fitness/steering measurement is run and documented; decision is recorded (promote to default, or keep gated and A/B against emergence encoder); the decision doc states the measured deltas and the rationale for the next step; cargo fmt/clippy/test green (this is a measurement + decision record, not a code change).

---

## 0003 — Cortex-Permanence-Fallback-Decision

### cortex-permanence-decision-fallback — Record Cortex Permanence Decision (GPU-Gated, Fallback) (GATED)

**Gate:** `occupancy-restructuring-selection` must escalate to fallback (no variant achieves ≥50% occupancy), OR `cortex-throughput-validation` must fail (restructuring is complete but ≥50% not met on hardware). This task records the explicit decision: cortex remains GPU-gated (default off), and the emergence line pivots to a fallback encoder (legacy raycast or alternate emergent baseline). This is documentation-only; no code changes.

**Steps:**
1. Author `0022-CORTEX-PERMANENCE-DECISION.md` in the plan folder, numbered to WS0003. Document: (a) the occupancy achieved (e.g. 'restructuring attempts reached max X% occupancy; structural limit prevents reaching 50%'), (b) the reasoning for permanence (e.g. 'lane occupancy is hardware-determined by workgroup size and feature count; increasing feature count to 256+ would degrade filter quality; multi-agent workgroup adds unacceptable complexity; fallback is mandatory'), (c) the impact on the emergence encoder line ('the emergence A/B (0017 WS0004) is replaced by a comparison against the legacy raycast encoder or a simpler learned baseline; the cortex remains a complete research artifact available behind visual_cortex_enabled=false'), (d) the revisit gate ('occupancy decision is locked unless: (i) WGSL gains native subgroup operations that allow sub-256-lane dispatch, or (ii) a future GPU backend achieves higher occupancy via alternative dispatch structures').
2. Update the root `docs/plans/STATUS.md` row for 0017 to reflect the final outcome: "Cortex optimization complete; occupancy-limited on GPU hardware; decision: permanently GPU-gated, emergence line pivots to legacy encoder; [decision doc link]."
3. Update 0017's own `STATUS.md` to close the cortex workstream with the permanence verdict and the emergence pivot rationale.

- **Depends on:** occupancy-restructuring-selection
- **Done when:** Decision doc `0022-CORTEX-PERMANENCE-DECISION.md` is authored and clearly states: cortex occupancy ceiling, reason for permanence, impact on emergence roadmap, and revisit gate; root STATUS.md and 0017 STATUS.md are updated to reflect the final outcome (cortex permanently GPU-gated, emergence pivots); documentation-only, no code changes; no test gate required.

---

**End of plan 0022 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
