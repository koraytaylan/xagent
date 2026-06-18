# Architecture — Plan 0011 (deltas)

> Edits in `crates/xagent-brain/src/gabor.rs`, `crates/xagent-brain/src/dog.rs`,
> `crates/xagent-brain/src/complex.rs`,
> `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_brain_encode_tiled.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_brain_encoder_credit_tiled.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_brain_predictor_tiled.wgsl`,
> `crates/xagent-brain/src/shaders/kernel/phase_brain_tail_from_scratch.wgsl`,
> `crates/xagent-sandbox/src/ui.rs`, `crates/xagent-sandbox/src/sim_runtime.rs`,
> `crates/xagent-sandbox/src/bench.rs`, and the baseline table in
> `crates/xagent-sandbox/tests/contributing_guard.rs`.
> Line numbers are hints; locate by symbol (grep for `plan 0`, `Layer`,
> `speed-decoupling`, `0008`, `0006`).

## 0001 — Vision-encoder & brain-shader reference scrub

Today the `xagent-brain` encoder modules and the brain-pass shaders carry
planning-process references frozen in `PLANNING_REFERENCE_BASELINE`
(`contributing_guard.rs:36-71`): module-intro doc-comments like "(plan 0008)" in
`gabor.rs` (5), `dog.rs` (3), `complex.rs` (1); section banners "(plan 0006)" /
"(plan 0008)" / "(plan 0009 …)" throughout `brain_passes.wgsl` (20); and single
intro/banner references in `brain_tick.wgsl` (1),
`phase_brain_encode_tiled.wgsl` (1), `phase_brain_encoder_credit_tiled.wgsl` (1),
`phase_brain_features.wgsl` (3), `phase_brain_predictor_tiled.wgsl` (1), and
`phase_brain_tail_from_scratch.wgsl` (1). None affect compiled behavior.

Edits:

- **Replace each reference with its technical reason** at the same site. Example
  transform in a shader banner:

```wgsl
// before:  // ===== feature_extract (plan 0008 visual cortex + plan 0009 danger percept) =====
// after:   // ===== feature_extract: visual-cortex features + danger percept =====
```

- **Lower the baseline rows** in `PLANNING_REFERENCE_BASELINE` for each touched
  file to its new count in the same commit; remove the row when the file reaches
  zero. The guard prints a ready-to-paste replacement table on any drift
  (`contributing_guard.rs:121-129`).

Properties that make this safe:
- Comments/doc-comments only; no WGSL statement, binding, override constant, or
  Rust signature changes — the shader bytecode and all `shader_*_constants_match_rust`
  / physics-equivalence tests are unaffected.
- The guard's `reductions` check (`contributing_guard.rs:102-129`) makes the
  baseline edit mandatory and mechanically verifies the new count matches reality.

## 0002 — Sandbox binary reference scrub

Today the `xagent-sandbox` binary modules Plan 0010 does not touch carry frozen
references (`contributing_guard.rs:73,76,77`): `ui.rs` (3, e.g. tab/telemetry
labels referencing a plan), `sim_runtime.rs` (1, "plan 0008, task 0004"), and
`bench.rs` (4, "plan 0008", "0006-fused baseline", "0008-VISUAL-CORTEX-BASELINE.md").

Edits:

- **Replace each reference with its technical reason** (e.g. `bench.rs`'s
  "0006-fused baseline" → "the fused-kernel throughput baseline"), and lower the
  three baseline rows (removing any that reach zero) in the same commit.

Properties that make this safe:
- Runtime strings and comments only; UI rendering, the bench harness, and the sim
  worker behavior are unchanged. `bench.rs`'s doc-path reference is replaced with
  prose, not a live path used by code.

## Test strategy

The falsifiable gate for every task is the existing
`planning_references_do_not_exceed_baseline` test
(`contributing_guard.rs:89-130`): after a scrub, the file's actual count must equal
its (lowered or removed) baseline row, or the test fails on either an over-count or
an un-lowered reduction. No new test is added; the guard *is* the acceptance gate.
Tasks additionally grep their touched files for the uncaught vocabulary
(`Layer `, `speed-decoupling`, bare `0008`/`0006`) and remove it, verified by the
reviewer-pass checklist (`docs/REVIEW-CHECKLIST.md`).

CI gate (every task): `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test -p xagent-sandbox`.

## Interaction with prior work

- **Completes the consensus #1 review finding.** Grok 4.3 and Gemini 3.1 Pro High
  both flagged the planning-reference leakage; this plan pays the portion of that
  debt outside Plan 0010's touched files, leaving the guard baseline strictly
  smaller.
- **Disjoint from Plan 0010.** The two plans edit non-overlapping files and
  non-overlapping `PLANNING_REFERENCE_BASELINE` rows, so they merge into `develop`
  in either order without contention.
- **Honors the project's incremental-cleanup model.** The guard exists precisely
  because `CONTRIBUTING.md` chose ratchet-on-touch over a big-bang rewrite; this
  plan accelerates the ratchet on a bounded, disjoint file set rather than
  reverting that decision.
