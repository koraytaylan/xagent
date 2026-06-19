# Plan 0011 — Source Planning-Reference Scrub — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete. All 5 tasks landed and squash-merged into
`claude/hungry-pare-b37669` as `ed47c7c119692f5ceb4a055447c3093c34857b27`.
_Last updated: 2026-06-19, against `claude/hungry-pare-b37669`._

- **Goal:** Ratchet the `contributing_guard.rs::PLANNING_REFERENCE_BASELINE` down
  to zero on the files Plan 0010 does not edit, replacing each planning-process
  reference with its technical rationale and removing the uncaught
  `Layer …`/`speed-decoupling`/bare `0008`/`0006` vocabulary the guard does not
  match — comment/string hygiene only, no behavioral change.
- **Measured baseline (current code):** the guard is green at 128 reference-lines
  across 23 files; this plan owns 44 of those lines across 13 files
  (`gabor.rs` 5, `dog.rs` 3, `complex.rs` 1, `brain_passes.wgsl` 20,
  `brain_tick.wgsl` 1, `phase_brain_encode_tiled.wgsl` 1,
  `phase_brain_encoder_credit_tiled.wgsl` 1, `phase_brain_features.wgsl` 3,
  `phase_brain_predictor_tiled.wgsl` 1, `phase_brain_tail_from_scratch.wgsl` 1,
  `ui.rs` 3, `sim_runtime.rs` 1, `bench.rs` 4).
- **Root cause:** Plan 0009 (and earlier 0006/0008 work) introduced and left
  planning-process language in source comments/doc-comments/strings; the guard
  froze and ratchets it but the debt is not yet paid on these files.
- **Approach:** two parallel workstreams, one per crate area, each with per-file
  (or small-group) scrub tasks that strip references and lower the matching
  baseline rows in the same commit — all disjoint, all `Depends on: —`, fully
  parallel with each other and with Plan 0010.
- **Outcome:** All 5 tasks completed; planning-process references stripped from the
  13 owned files across both crate areas; `contributing_guard.rs` baseline ratcheted
  down accordingly — comment/string hygiene only, no behavioral change.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Vision-encoder & brain-shader reference scrub | `scrub-vision-encoder-modules`, `scrub-brain-passes-shader`, `scrub-tiled-brain-shaders` | ✅ Done |
| 0002 | Sandbox binary reference scrub | `scrub-ui-and-sim-runtime`, `scrub-bench` | ✅ Done |

## Verification

All tasks landed green. The `planning_references_do_not_exceed_baseline` guard
remains green with this plan's 13 files removed from (or lowered in)
`PLANNING_REFERENCE_BASELINE`, and no `plan 0…`/`Layer …`/`speed-decoupling`/bare
`0008`/`0006` vocabulary remains in the owned files. No runtime behavior, buffer
layout, or other test outcome changes.
