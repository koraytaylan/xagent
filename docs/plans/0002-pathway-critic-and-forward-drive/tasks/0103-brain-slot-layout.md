---
id: brain-slot-layout
title: "Config slots, brain-state slot block and legacy upgrade function"
workstream: "0100"
kind: task
role: developer
depends_on:
  - default-fingerprint
  - route-flags
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib buffers"
    timeout_ms: 3600000
  - name: routes-off-fingerprint
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_default_fingerprint"
    timeout_ms: 3600000
  - name: vision-pathway
    command: "cargo test -p xagent-brain --test vision_pathway"
    timeout_ms: 3600000
  - name: crate-fmt
    command: "cargo fmt -p xagent-brain -- --check"
    fix: "cargo fmt -p xagent-brain"
  - name: crate-clippy
    command: "cargo clippy -p xagent-brain -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: contributing-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Config slots, brain-state slot block and legacy upgrade function

## Outcome

`buffers.rs` and `common.wgsl` gain two config slots, the 1321-value pathway
slot block after `O_TRACE_VISION`, and `upgrade_legacy_brain_state`. No shader
pass reads or writes the block, a fresh brain has it zero, and routes-off
behaviour is unchanged.

## Scope

- Config slots `CFG_CRITIC_INPUT` and `CFG_FORWARD_PATHWAY_ENABLED`, packed in
  `build_config_for` from the `BrainConfig` fields added by route-flags, with
  named Rust consts for the encodings 0.0, 1.0 and 2.0.
- The slot block in Rust and WGSL, with doc comments giving each slot's
  lifecycle (zero at birth, survives or is zeroed at death).
- `upgrade_legacy_brain_state` as specified below.
- Unit tests for every item below.

## Non-goals

- No edits to kernel passes, `governor.rs`, `config.rs` or `gpu_kernel.rs`.
  Blob-decode wiring belongs to legacy-decode-and-birth-state.
- No change to mutation, inheritance or any existing value.

## Design

**Config slots.** Read the `CFG_*` constants in `buffers.rs` and `common.wgsl`
and `CONFIG_SIZE` at dispatch. Expected, once plan 0001 has landed (slots
16-20 used, `CONFIG_SIZE` 24): `CFG_CRITIC_INPUT = 21`,
`CFG_FORWARD_PATHWAY_ENABLED = 22`, `CONFIG_SIZE` unchanged. If those indices
are taken, use the next two free indices after the highest `CFG_` index, grow
`CONFIG_SIZE` to the next multiple of 4 and the `brain_config` array in
`common.wgsl` to match. Mirror both names in `common.wgsl` as
`const NAME: u32 = Nu;`. Add `< CONFIG_SIZE` asserts beside the existing ones.

**Slot block.** In `buffers.rs`, with `PATHWAY_INPUT_COUNT = SCENT_CHANNELS +
VISION_PATHWAY_INPUTS` (10), in this order:

- `O_FORWARD_PATHWAY_WEIGHTS = O_TRACE_VISION + VISION_PATHWAY_INPUTS`
- `O_TRACE_FORWARD_PATHWAY = O_FORWARD_PATHWAY_WEIGHTS + PATHWAY_INPUT_COUNT`
- `O_CRITIC_PATHWAY_WEIGHTS = O_TRACE_FORWARD_PATHWAY + PATHWAY_INPUT_COUNT`
- `O_TRACE_CRITIC_PATHWAY = O_CRITIC_PATHWAY_WEIGHTS + PATHWAY_INPUT_COUNT`
- `O_PREV_PATHWAY_INPUT_NORM = O_TRACE_CRITIC_PATHWAY + PATHWAY_INPUT_COUNT`
- `O_RECENT_PATHWAY_KEYS = O_PREV_PATHWAY_INPUT_NORM + 1`
- `O_SENSORY_MEAN = O_RECENT_PATHWAY_KEYS + RECENT_CAP * PATHWAY_INPUT_COUNT`
- `PATHWAY_ROUTE_SLOT_COUNT` = the four `PATHWAY_INPUT_COUNT` regions, plus 1,
  plus `RECENT_CAP * PATHWAY_INPUT_COUNT` (1321).

`FIXED_TAIL_SIZE` grows by the block automatically; update its doc comment if
it describes the tail's end. In `common.wgsl`, add `const PATHWAY_INPUT_COUNT:
u32 = SCENT_CHANNELS + VISION_PATHWAY_INPUTS;` and one `override` line per
offset, after the `O_TRACE_VISION` line and replacing the `O_SENSORY_MEAN` line,
with expression text identical to the Rust expression.

**Legacy upgrade (D4).** `pub fn upgrade_legacy_brain_state(brain_state: &[f32])
-> Option<Vec<f32>>`:

1. If `feature_count_for_brain_stride(len)` is `Some`, return a copy.
2. `legacy_tail = FIXED_TAIL_SIZE - PATHWAY_ROUTE_SLOT_COUNT` (checked).
3. `legacy_base = ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION +
   legacy_tail` (checked). `variable = len - legacy_base` (checked). Require
   `variable % (ENCODED_DIMENSION + 1) == 0` and feature count `fc = variable /
   (ENCODED_DIMENSION + 1) >= 1`, else `None`.
4. `insert_at = fc * ENCODED_DIMENSION + ENCODED_DIMENSION +
   PREDICTOR_DIMENSION * ENCODED_DIMENSION + legacy_tail` (checked). Compute it
   arithmetically: `fixed_tail_base` decodes current-layout strides and must not
   be used here.
5. Return `[..insert_at]`, then `PATHWAY_ROUTE_SLOT_COUNT` zeros, then
   `[insert_at..]`.

Use no panicking indexing or unchecked arithmetic.

**Existing test.** The assertion `O_SENSORY_MEAN == O_TRACE_VISION +
VISION_PATHWAY_INPUTS` in `brain_stride_is_consistent` changes to include
`PATHWAY_ROUTE_SLOT_COUNT`. If any other test outside `touches` breaks, report
it as a contradiction; do not edit it.

## Steps

**Steps:**

1. In `buffers.rs`, add the two `CFG_*` consts at the expected or next free
   indices, the three named encoding consts with doc comments, packing in
   `build_config_for`, and the matching `const` lines in `common.wgsl` (growing
   `CONFIG_SIZE` and the array only if needed).
2. In `buffers.rs`, add `PATHWAY_INPUT_COUNT`, the block offsets,
   `PATHWAY_ROUTE_SLOT_COUNT` and the lifecycle doc comments, and update the
   `O_SENSORY_MEAN` assertion in `brain_stride_is_consistent`.
3. In `common.wgsl`, add `PATHWAY_INPUT_COUNT` and the block `override` lines
   with identical expression text, replacing the `O_SENSORY_MEAN` line. Add no
   shader pass.
4. In `buffers.rs`, implement `upgrade_legacy_brain_state` exactly as the Design
   algorithm.
5. In the `buffers.rs` tests module, add: (a) offsets contiguous and ordered,
   `PATHWAY_INPUT_COUNT == 10`, `PATHWAY_ROUTE_SLOT_COUNT == 1321`; (b)
   `FIXED_TAIL_SIZE - PATHWAY_ROUTE_SLOT_COUNT == O_TRACE_VISION +
   VISION_PATHWAY_INPUTS - O_PREDICTOR_CONTEXT_WEIGHT`; (c) WGSL parity: each
   block `override` line and each new `CFG_*` line in `common.wgsl` matches the
   Rust value (use `parse_wgsl_u32_constants` for `const` lines and exact-text
   `contains` for `override` lines); (d) a fresh `init_brain_state` has the
   block zero; (e) `build_config_for` writes 0.0, 1.0, 2.0 for the encodings and
   0.0 or 1.0 for the forward flag at the two `CFG_*` indices; (f) for `fc` in
   `1..=8192`, the legacy length `fc * (ENCODED_DIMENSION + 1) +
   ENCODED_DIMENSION + PREDICTOR_DIMENSION * ENCODED_DIMENSION + FIXED_TAIL_SIZE
   - PATHWAY_ROUTE_SLOT_COUNT` is never accepted by `feature_count_for_brain_stride`;
   (g) upgrading a default-layout state with the block removed returns the
   original state; (h) a current length returns an unchanged copy; (i) lengths
   `0`, `1`, `BRAIN_STRIDE - 1`, `BRAIN_STRIDE + 1` and the legacy length minus
   one return `None`.
6. Format the edited files, keep every new doc comment free of plan, task and
   workstream names, and avoid lossy `as` casts, so the crate-fmt, crate-clippy
   and contributing-guard checks pass.

- **Done when:** `cargo test -p xagent-brain --lib buffers` passes the block layout, WGSL parity, config packing and legacy upgrade tests, the routes-off-fingerprint check still matches the baseline fingerprint, and the contributing-guard check stays green (proves INV-1, INV-2, INV-3, INV-12).
