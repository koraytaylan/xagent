# XAgent Plan 0011 — Source Planning-Reference Scrub

Ratchet the `contributing_guard.rs` planning-reference baseline down to zero on
the files Plan 0010 does not touch: strip every planning-process reference (and the
uncaught `Layer …`/`speed-decoupling`/bare `0008`/`0006` vocabulary) from the
`xagent-brain` encoder modules and brain-pass shaders and the `xagent-sandbox`
binary modules, replacing each with its technical rationale, and lower (or remove)
each file's baseline row in the same commit.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none). Every task
  here edits a disjoint file set and disjoint `PLANNING_REFERENCE_BASELINE` rows,
  so all tasks have no dependencies and branch fully in parallel.
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
  The decisive gate is `planning_references_do_not_exceed_baseline` in
  `crates/xagent-sandbox/tests/contributing_guard.rs`.
- Line numbers are hints; locate every reference by grep (`plan 0`, `Layer `,
  `speed-decoupling`, `0008`, `0006`, `workstream`, `docs/superpowers`).

**Scrub procedure (every task):**
1. Grep the task's files for `plan 0`, `Layer `, `speed-decoupling`, bare
   `0008`/`0006`, `workstream`, `docs/superpowers`.
2. Replace each hit's planning framing with the in-place technical reason for the
   code; do not delete the technical content of the comment.
3. Lower the touched files' rows in
   `contributing_guard.rs::PLANNING_REFERENCE_BASELINE` to the new count, or remove
   the row entirely if the file reaches zero (the guard prints a ready-to-paste
   table on drift).
4. Run `cargo test -p xagent-sandbox contributing_guard` and confirm green.

---

## 0001 — Vision-encoder & brain-shader reference scrub

### scrub-vision-encoder-modules — Strip Plan References from the DoG/Gabor/Complex Encoder Modules

`gabor.rs` (5 frozen refs), `dog.rs` (3), and `complex.rs` (1) carry intro
doc-comments and inline tags referencing the visual-cortex plan ("(plan 0008)",
"(plan 0003)"). These are the early-visual-cortex stages; their planning framing
belongs in the plan docs, not the module headers.

**Steps:**
1. Apply the scrub procedure to `crates/xagent-brain/src/gabor.rs`,
   `crates/xagent-brain/src/dog.rs`, and `crates/xagent-brain/src/complex.rs`
   (e.g. "Oriented Gabor simple-cell bank (plan 0008)" → "Oriented Gabor
   simple-cell bank").
2. Remove the three rows for these files from `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:37-39`) if they reach zero, else lower them.

- **Depends on:** —
- **Done when:** `gabor.rs`/`dog.rs`/`complex.rs` carry no planning references, the
  baseline rows are removed/lowered to match, and
  `planning_references_do_not_exceed_baseline` is green; cargo fmt/clippy/test green.

### scrub-brain-passes-shader — Strip Plan References from `brain_passes.wgsl`

`brain_passes.wgsl` carries the heaviest debt (20 frozen refs): section banners
"(plan 0006)", "(plan 0008 …)", "(plan 0009 …)" across the cooperative brain
passes. None affect the compiled passes.

**Steps:**
1. Apply the scrub procedure to
   `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, replacing each
   banner's plan tag with the pass's technical description (e.g. "feature_extract
   (plan 0008 visual cortex + plan 0009 danger percept)" → "feature_extract:
   visual-cortex features + danger percept").
2. Remove or lower the `brain_passes.wgsl` row in `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:41-44`) to the new count.

- **Depends on:** —
- **Done when:** `brain_passes.wgsl` carries no planning references (or a lowered,
  table-matching count), the shader still compiles and all
  `shader_*_constants_match_rust`/brain-pass tests stay green, and the guard is
  green; cargo fmt/clippy/test green.

### scrub-tiled-brain-shaders — Strip Plan References from the Tiled Brain-Pass Shaders

`brain_tick.wgsl` (1), `phase_brain_encode_tiled.wgsl` (1),
`phase_brain_encoder_credit_tiled.wgsl` (1), `phase_brain_features.wgsl` (3),
`phase_brain_predictor_tiled.wgsl` (1), and `phase_brain_tail_from_scratch.wgsl` (1)
each carry intro/banner plan tags (mostly "(plan 0006)"/"(plan 0008)").

**Steps:**
1. Apply the scrub procedure to all six shaders under
   `crates/xagent-brain/src/shaders/kernel/`.
2. Remove/lower their six rows in `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:45,48-67`).

- **Depends on:** —
- **Done when:** the six shaders carry no planning references, their baseline rows
  match, the shaders compile and brain-pass tests stay green, and the guard is
  green; cargo fmt/clippy/test green.

---

## 0002 — Sandbox binary reference scrub

### scrub-ui-and-sim-runtime — Strip Plan References from `ui.rs` and `sim_runtime.rs`

`ui.rs` (3 frozen refs, e.g. telemetry/tab labels referencing a plan) and
`sim_runtime.rs` (1, "plan 0008, task 0004") carry planning tags in comments and
UI/log strings. Runtime strings are in scope per CONTRIBUTING.

**Steps:**
1. Apply the scrub procedure to `crates/xagent-sandbox/src/ui.rs` and
   `crates/xagent-sandbox/src/sim_runtime.rs`.
2. Remove/lower their rows in `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:76,77`).

- **Depends on:** —
- **Done when:** both files carry no planning references, their baseline rows
  match, and the guard is green; cargo fmt/clippy/test green.

### scrub-bench — Strip Plan References from `bench.rs`

`bench.rs` (4 frozen refs) names planning artifacts in comments and labels:
"plan 0008", "0006-fused baseline", "0006-fused", "0008-VISUAL-CORTEX-BASELINE.md".

**Steps:**
1. Apply the scrub procedure to `crates/xagent-sandbox/src/bench.rs`, replacing the
   plan/doc references with technical prose (e.g. "0006-fused baseline" → "the
   fused-kernel throughput baseline").
2. Remove/lower the `bench.rs` row in `PLANNING_REFERENCE_BASELINE`
   (`contributing_guard.rs:73`).

- **Depends on:** —
- **Done when:** `bench.rs` carries no planning references, the baseline row
  matches, the bench harness still builds, and the guard is green; cargo
  fmt/clippy/test green.

---

**End of plan 0011 TASKS.** When every "Done when" bullet is green, the
`PLANNING_REFERENCE_BASELINE` carries no rows for this plan's files and the guard
debt is strictly smaller across the disjoint set Plan 0010 does not touch.
