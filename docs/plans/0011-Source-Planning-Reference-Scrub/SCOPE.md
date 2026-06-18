# Scope — Plan 0011

> Pay down the CONTRIBUTING "source must be agnostic of the planning process"
> debt the four 2026-06-18 reviews flagged, by ratcheting the
> `contributing_guard.rs` planning-reference baseline down to zero on the files
> Plan 0010 does not touch — so the two plans run in parallel without contending
> for the guard baseline.

## Why this plan

`CONTRIBUTING.md` forbids naming an internal plan, task, workstream, decision
doc, or spec section anywhere in source, and mandates clearing the pre-existing
debt *file-by-file as code is touched*. Three reviews flagged the leakage:

1. **Pervasive planning references remain in source.** Grok 4.3 named this its #1
   pre-merge blocker and Gemini 3.1 Pro High raised it as a High rule violation;
   both are accurate in magnitude. The mechanical guard
   (`crates/xagent-sandbox/tests/contributing_guard.rs`) freezes the current debt
   at **128 reference-lines across 23 files** in `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:35-80`) and ratchets it: a file may fall below its
   frozen count but never rise, and a reduction that is not reflected in the table
   fails the test (`:102-129`). The debt is contained and green, but not yet paid.
2. **The guard catches only some of the planning vocabulary.** It matches the
   substrings `workstream`/`status.md`/`tasks.md`/`docs/superpowers`
   (`contributing_guard.rs:83-84`), the regex `plan 0[0-9]{3}` (`:187-205`), and
   the slug `000[0-9]-[a-z]` (`:209-219`). It does **not** catch `Layer A/B/C/D`,
   `speed-decoupling gate`, or a bare `0008` without the `plan ` prefix — so a file
   can be guard-green yet still carry planning language the reviewers (and the
   reviewer-pass checklist) want gone.

This plan owns the scrub for the files **Plan 0010 does not edit** — a disjoint
set, so 0010's on-touch cleanup and this plan's wholesale scrub never contend for
the same file or the same baseline row. Plan 0010 strips references in the regions
it edits and lowers those files' rows itself (see `0010/SCOPE.md` locked
decisions); the WGSL magic-number naming the reviews flagged
(`255`/`1.414`/`20.0`) lives in shaders Plan 0010 already edits and is owned there,
so this plan is pure reference-scrub with no magic-number work.

**Provenance.** Verified against branch `claude/funny-cray-84end5` @ `5ddc976`. The
per-file counts in `PLANNING_REFERENCE_BASELINE` were re-derived by re-running the
guard's exact matching logic over the current tree and match the frozen table
exactly (zero drift); the owned-file list below is the subset of that table that
Plan 0010's task list does not edit.

**Files owned by this plan (disjoint from Plan 0010), with current frozen counts:**

| File | Refs |
|---|---|
| `crates/xagent-brain/src/gabor.rs` | 5 |
| `crates/xagent-brain/src/dog.rs` | 3 |
| `crates/xagent-brain/src/complex.rs` | 1 |
| `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` | 20 |
| `crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl` | 1 |
| `crates/xagent-brain/src/shaders/kernel/phase_brain_encode_tiled.wgsl` | 1 |
| `crates/xagent-brain/src/shaders/kernel/phase_brain_encoder_credit_tiled.wgsl` | 1 |
| `crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl` | 3 |
| `crates/xagent-brain/src/shaders/kernel/phase_brain_predictor_tiled.wgsl` | 1 |
| `crates/xagent-brain/src/shaders/kernel/phase_brain_tail_from_scratch.wgsl` | 1 |
| `crates/xagent-sandbox/src/ui.rs` | 3 |
| `crates/xagent-sandbox/src/sim_runtime.rs` | 1 |
| `crates/xagent-sandbox/src/bench.rs` | 4 |

Files Plan 0010 edits (`buffers.rs`, `config.rs`, `gpu_kernel.rs`, `governor.rs`,
`agent/mod.rs`, `main.rs`, `kernel_tick.wgsl`, `phase_physics.wgsl`, `common.wgsl`,
`integration.rs`; `headless.rs` is already at 0) are **out of scope here** — their
debt is paid by 0010's on-touch obligation.

## In scope

- **0001 — Vision-encoder & brain-shader reference scrub.** Strip the
  planning-process references (and the uncaught `Layer …`/`speed-decoupling`/bare
  `0008` vocabulary) from the `xagent-brain` modules and shaders Plan 0010 does not
  touch, replacing each with a pure technical rationale, and lower each file's
  `PLANNING_REFERENCE_BASELINE` row to its new count (removing the row at zero).
- **0002 — Sandbox binary reference scrub.** The same, for the `xagent-sandbox`
  binary modules Plan 0010 does not touch (`ui.rs`, `sim_runtime.rs`, `bench.rs`).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Planning references in `xagent-brain` source (1) | `0001` |
| Planning references in `xagent-sandbox` binaries (1) | `0002` |
| Uncaught `Layer …`/`speed-decoupling`/bare `0008` vocabulary (2) | `0001`, `0002` |

## Locked decisions

- **Ratchet down, never sideways.** Every task lowers the touched files'
  `PLANNING_REFERENCE_BASELINE` rows in the same commit it removes references; the
  guard fails on an un-lowered reduction (`contributing_guard.rs:102-129`), so the
  table edit is mandatory, not optional. A file taken to zero has its row removed.
  No number is ever raised.
- **Remove the planning language, keep the technical content.** Each reference is
  replaced with the in-place technical reason for the code (e.g. "Locked per batch,
  not heritable; default false preserves the pre-percept encoder input width"), not
  deleted wholesale — the comment's information survives, only its planning framing
  goes. The planning rationale belongs in the commit message and the plan docs.
- **Scrub the uncaught vocabulary too.** Tasks also remove `Layer A/B/C/D`,
  `speed-decoupling gate`, and bare `0008`/`0006` plan tags that the guard does not
  match, per the reviewer-pass checklist — the guard is the floor, not the ceiling.
- **Disjoint from Plan 0010.** This plan touches only the files listed above; it
  never edits a file or a baseline row Plan 0010 edits, so the two are
  fully parallel.

## Out of scope

- **The files Plan 0010 edits.** Their references are paid by 0010's on-touch
  cleanup (`buffers.rs`, `config.rs`, `gpu_kernel.rs`, `governor.rs`,
  `agent/mod.rs`, `main.rs`, `kernel_tick.wgsl`, `phase_physics.wgsl`,
  `common.wgsl`, `integration.rs`).
- **Magic-number naming (`255`/`1.414`/`20.0`).** Lives in shaders Plan 0010
  edits; owned by 0010's `shader-magic-number-naming` task.
- **Any behavioral or functional change.** This is comment/string hygiene only;
  no runtime behavior, buffer layout, or test outcome changes.
- **Raising or freezing new debt.** The guard already prevents growth; this plan
  only shrinks it.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
