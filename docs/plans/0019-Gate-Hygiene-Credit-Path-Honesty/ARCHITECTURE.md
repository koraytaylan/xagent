# Architecture — Plan 0019 (deltas)

> Edits in `crates/xagent-sandbox/src/ui.rs`,
> `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/STATUS.md`,
> `docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/CORTEX-PROFILE-BASELINE.txt` (read-only reference),
> `docs/plans/STATUS.md`,
> `crates/xagent-sandbox/tests/integration.rs`,
> `docs/plans/0018-Credit-Path-Mechanism-Attack/0001-AUXILIARY-LOSS-DECISION.md`,
> `docs/plans/0018-Credit-Path-Mechanism-Attack/0002-TRACE-HORIZON-DECISION.md`,
> `docs/plans/0015-Intent-Awareness-Measurement-Framework/STATUS.md`, and
> `docs/plans/0016-Intent-As-Meta-Score-Evolved-Policy-Ablation/STATUS.md`.
> Line numbers are hints; locate by symbol (grep for `tex_key`,
> `cortex_throughput_profile_baseline`, `run_innate_instinct_ab`,
> `XAGENT_VERBOSE_PROBES`).

## 0001 — Fmt-Gate-Restore

Today the agent-vision texture-key block (`ui.rs:1142-1145`) carries two
assignments broken across lines that `rustfmt` wants collapsed onto single
lines:

```rust
let tex_key =
    egui::Id::new(("agent_vision", effective_snap.id, vw, vh));
let existing =
    ctx.data(|data| data.get_temp::<egui::TextureHandle>(tex_key));
```

Commit `1da6e0f5` landed on `develop` claiming the format gate was green, but
`cargo fmt --all -- --check` is red at these lines — the pre-push gate was
skipped or CI did not run. CONTRIBUTING.md lists `cargo fmt` as the first CI
gate.

Edits:

- **Collapse the two assignments to single lines** per `rustfmt` style by
  running `cargo fmt --all`; the formatter rewrites the block to the
  canonical layout:

```rust
// rustfmt-canonical single-line form; these expressions fit the line width,
// so the manual breaks above were a pure formatting defect, not a wrap.
let tex_key = egui::Id::new(("agent_vision", effective_snap.id, vw, vh));
let existing = ctx.data(|data| data.get_temp::<egui::TextureHandle>(tex_key));
```

Properties that make this safe:
- The change is pure formatting: the token stream is identical, so the compiled
  behavior of `ui.rs` is byte-for-byte unchanged.
- It restores gate integrity per CONTRIBUTING.md — `cargo fmt --all -- --check`
  passes on `develop` — with zero logic impact and no other file touched.

## 0002 — Cortex-Throughput-Honesty

Today the 0017 STATUS marks the cortex throughput workstream (WS0002)
`Complete` with the 50% throughput budget asserted, but that assertion holds on
the lavapipe software adapter only. `CORTEX-PROFILE-BASELINE.txt` records the
real-GPU (Metal) numbers — baseline ≈ 7,625 tps, cortex ON ≈ 83 tps, i.e. 1.1%
of baseline (≈ 93× slower) — and `cortex_throughput_profile_baseline`
(`integration.rs:658`) self-skips on a real adapter, documenting the 18/256 lane
occupancy as the cause and asserting the budget only under lavapipe. A reader of
the STATUS roll-up concludes cortex is production-viable at 50% throughput when
real hardware is far below target.

Edits:

- **Correct the 0017 STATUS WS0002 row** (`docs/plans/0017-…/STATUS.md`) and the
  matching root roll-up row (`docs/plans/STATUS.md`) to state the real-GPU 1.1%
  budget miss and that the 50% assertion is lavapipe-only scope; mark the row
  `Blocked` rather than `Complete`. The change is prose in the STATUS tables plus
  a bumped `Last updated` line in both files — no code edit.
- **Rename the budget assertion to reflect its software-adapter scope** so the
  test name no longer reads as a real-hardware claim:

```rust
/// Asserts the cortex throughput budget on the lavapipe software adapter only;
/// real GPU (Metal) sits at ~1.1% of baseline (18/256 lane occupancy) and is
/// recorded as a known miss in CORTEX-PROFILE-BASELINE.txt and 0017 STATUS.
fn cortex_throughput_meets_budget_on_software_adapter() { /* unchanged body */ }
```

Properties that make this safe:
- Cortex remains default-off; this is honest accounting only — the shipped
  signal and the budget assertion's behavior are unchanged.
- The rename is a symbol-only edit with no change to the assertion or its
  threshold; the STATUS corrections are documentation, so no runtime, shader, or
  buffer-layout path is touched.

## 0003 — Innate-Instinct-Harness-Regression-Test

Today `run_innate_instinct_ab` (`headless.rs:492`) is the 0013 A/B gate harness:
it runs a baseline-vs-on comparison and returns
`(ValidationStats, ValidationStats, bool)`. It was run once on local Metal and
produced a FAILED verdict recorded in the 0013 decision doc, but no integration
test invokes it. The harness is orphaned — if the learning signal changes, the
gate cannot be re-run mechanically to confirm the failure still holds.

Edits:

- **Add a regression test in `integration.rs`** that invokes the harness with a
  tiny generation count and asserts the returned `ValidationStats` are
  well-formed, guarding the harness logic (not re-asserting the gate verdict):

```rust
/// Guards harness logic, not the 0013 gate verdict: with 2 generations and a
/// tiny population, every ValidationStats field must be finite and in a valid
/// range (no NaN/inf, counts non-negative). Catches harness rot if the learning
/// path changes shape, without pinning a pass/fail learning result.
#[test]
fn run_innate_instinct_ab_produces_valid_stats() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // 2 gens, N=5 — exercises the harness end-to-end at negligible runtime.
    let (baseline, on, _passed) = run_innate_instinct_ab(tiny_config(5), 2);
    for stats in [&baseline, &on] {
        // assert each field is finite and non-negative (well-formed), not a verdict.
    }
}
```

Properties that make this safe:
- The test guards logic, not the gate result: it asserts only that
  `ValidationStats` are well-formed, so it stays green regardless of whether the
  A/B verdict is PASS or FAILED.
- Runtime is negligible (2 generations, N=5) and the test is GPU-gated with the
  standard `GpuKernel::is_available()` self-skip, so it adds no CI cost on
  adapterless runners and exercises the harness on lavapipe.

## 0004 — Auxiliary-Loss-Mechanism-Honesty

Today `0001-AUXILIARY-LOSS-DECISION.md` (in the 0018 folder) correctly notes
that the auxiliary-steering-loss probe measures a CPU-side overlay with no GPU
gradient injection — `auxiliary_steering_loss_converges_on_bearing`
(`integration.rs:8806`) reports 0.509 alignment (chance) — and records a REJECT
verdict, correct for that CPU-overlay scope. But the 0018 roll-up's 3/3 REJECT
framing can mislead a reader into citing this as a falsification of the
auxiliary-loss mechanism, when the truthful statement is that a CPU overlay (not
the GPU-integrated mechanism) was tested.

Edits:

- **Re-frame the decision doc** so the REJECT is scoped to the CPU-side
  measurement overlay, not the mechanism: update the Summary to say so
  explicitly, add a `Scope-of-This-Rejection` section stating the mechanism was
  not tested at the GPU-integrated level, and point the Next Step at GPU
  integration as future work. If SCOPE.md lists this workstream, change its
  language from REJECT to DEFERRED-pending-GPU-integration.

Properties that make this safe:
- Documentation only — no code change, no behavior change, no default flip.
- The re-framing is strictly more honest: it preserves the measured CPU-overlay
  result while making clear the mechanism is not-tested-at-relevant-level rather
  than falsified, and it sets the scope gate a future GPU-integrated attempt
  (e.g. in 0020) would reference.

## 0005 — Frame-Sync-Spike-Clarity

Today `0002-TRACE-HORIZON-DECISION.md` (in the 0018 folder) correctly notes that
the steering probe trained at `vision_stride=1` (dense), where the frame-sync
mechanism is disabled, while the unit test confirms the mechanism is active at
`vision_stride=10` (sparse) with a +54% trace effect
(`0002-TRACE-HORIZON-DECISION.md:45`). Yet the 3/3 REJECT framing suggests the
mechanism was falsified when in fact it was never exercised under its activation
condition.

Edits:

- **Re-frame the decision doc** so the REJECT is scoped to the dense-stride
  probe, not the frame-sync mechanism: update the Result section to say the
  rejection applies to the `vision_stride=1` probe, add a `Scope-of-This-Result`
  section stating the mechanism was not tested under its activation condition,
  and add a `When-to-Revisit` section for a sparse-stride (`vision_stride=10`)
  evaluation.

Properties that make this safe:
- Documentation only — no code change, no behavior change.
- The truthful framing distinguishes "probe never exercised the mechanism" from
  "mechanism falsified," preserving the dense-stride measurement while leaving an
  explicit sparse-stride revisit gate for future work.

## 0006 — Diagnostic-Table-Silencing

Today several probes print large diagnostic tables under `--nocapture`:
`cortex_throughput_profile_baseline` (`integration.rs:658`),
`baseline_td_error_variance_during_foraging` (`integration.rs:8705`),
`gradient_variance_per_context_breakdown` (`integration.rs:9012`), and
`auxiliary_steering_loss_converges_on_bearing` (`integration.rs:8806`). The
2026-06-19 grok-43 review recommended gating these behind an env var so tests
are silent (assertion-only) by default, as was already done for the fitness
probes; the 0017/0018 probes were not yet converted.

Edits:

- **Wrap each diagnostic `eprintln!` table behind `XAGENT_VERBOSE_PROBES`** so
  the tables print only on demand while the assertions are untouched:

```rust
// Diagnostic table is opt-in: silent in CI (assertion-only), printed when a
// local developer sets XAGENT_VERBOSE_PROBES=1 to inspect the breakdown.
if std::env::var("XAGENT_VERBOSE_PROBES").is_ok() {
    eprintln!("…diagnostic table…");
}
```

Properties that make this safe:
- The env var defaults unset, so CI runs silent with assertions only; the tables
  remain available on demand for local debugging.
- Only the `eprintln!` diagnostics are wrapped — every assertion is unchanged, so
  the tests stay exactly as falsifiable as before. This aligns with the prior
  fitness-probe cleanup and reduces CI noise.

## 0007 — Test-File-Path-Traceability

Today plan STATUS (and ARCHITECTURE) docs reference brain-crate test
deliverables — `intent_baseline_measurement.rs`, `danger_percept_ablation_ab.rs`
— without a crate prefix (e.g. in `docs/plans/0015-…/STATUS.md` and
`docs/plans/0016-…/STATUS.md`). These tests live under
`crates/xagent-brain/tests/`, but a bare filename is easily misread as
`xagent-sandbox/tests/…`, sending a searcher to the wrong crate now that the
project has two test crates.

Edits:

- **Update every test-file reference to a full crate-relative path** in the 0015
  and 0016 STATUS docs (and the matching ARCHITECTURE docs), e.g.
  `crates/xagent-brain/tests/intent_baseline_measurement.rs` and
  `crates/xagent-brain/tests/danger_percept_ablation_ab.rs`; review other plans
  for the same bare-filename pattern and apply consistently.

Properties that make this safe:
- Documentation only — no code change; the deliverables themselves are
  unchanged.
- Explicit crate scoping removes the wrong-crate ambiguity, so traceability holds
  without affecting any build, test, or runtime path.
