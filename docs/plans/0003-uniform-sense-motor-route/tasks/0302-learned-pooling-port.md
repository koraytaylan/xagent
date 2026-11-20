---
id: learned-pooling-port
title: "Apply the pooling verdict to the vision sense"
workstream: "0300"
kind: task
role: developer
depends_on:
  - pooling-study
gated: false
status: done
merged_as: "2f0624604fb8097443cb88954cba4be96a21bcfc"
touches:
  # All conditional on Verdict: PORTED in docs/decisions/pooling-study.md; with NOT PORTED no file changes.
  - crates/xagent-brain/src/senses.rs
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-shared/src/config.rs
  - crates/xagent-sandbox/tests/learned_pooling.rs
checks:
  - name: focused-proof
    command: >-
      sh -c 'if grep -qx "Verdict: PORTED" docs/decisions/pooling-study.md; then env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test learned_pooling -- --include-ignored; else grep -qx "Verdict: NOT PORTED" docs/decisions/pooling-study.md; fi'
    timeout_ms: 3600000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# 0302 — Apply the pooling verdict to the vision sense

## Outcome

The vision sense's input matches the verdict recorded in
`docs/decisions/pooling-study.md`, which
contains exactly one line `Verdict: PORTED` or `Verdict: NOT PORTED`.

- **NOT PORTED** (the expected case: the reference is +0.16 against +0.33).
  Make no source change. The focused-proof check verifies the verdict line and
  the parity check verifies that the default is unchanged.
- **PORTED.** Implement the winning arm named in the record (credit-trained
  column pooling, or Oja/PCA pooling) as a new `SenseInput` variant, implemented
  once in the generic sense block with no vision-specific code path. It is
  selected for vision only by `BrainConfig::vision_learned_pooling_enabled`.

## Decisions (PORTED only)

- Flag: `pub vision_learned_pooling_enabled: bool` in `BrainConfig`
  (`crates/xagent-shared/src/config.rs`), serde default `false`, not mutated and
  not heritable. It takes the next unused `CFG_*` index in `buffers.rs` (after
  every `CFG_*` constant present when this task starts), mirrored by the same
  name in `common.wgsl`. `CONFIG_SIZE` grows only if the existing alignment test
  requires it.
- Layout: the pooling-weight slots enter `BrainLayout` only when the flag is on,
  appended after the existing sense slots. With the flag off, the stride and
  every existing offset are unchanged, so stored databases and parity hold.
- Birth values and learning rule: exactly those named for the winning arm in
  the decision record (its fixed-decisions section), including its constants.

## Steps

**Steps:**

1. Read `docs/decisions/pooling-study.md`
   and note the single `Verdict:` line and the winning arm. If the verdict is
   `Verdict: NOT PORTED`, make no edit and go straight to the checks.
2. (PORTED) In `crates/xagent-shared/src/config.rs`, add the
   `vision_learned_pooling_enabled` field with its serde default and a doc
   comment. In `crates/xagent-brain/src/buffers.rs`, add the
   `CFG_VISION_LEARNED_POOLING_ENABLED` slot and pack it in the config upload.
   Mirror the slot constant in `crates/xagent-brain/src/shaders/kernel/common.wgsl`.
3. (PORTED) In `crates/xagent-brain/src/senses.rs`, add the new input variant
   (the name from the record). In `buffers.rs`, derive its pooling-weight slots
   through `BrainLayout` only when the flag is on, appended after the existing
   sense slots. Add a unit test that the flag-off layout equals the baseline
   layout offset for offset.
4. (PORTED) In `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl` and
   `common.wgsl`, route the new input through the generic sense block (gather,
   whiten, route dot product, trace and TD update) with the birth values and
   learning rule from the record. Add no smell- or vision-specific identifiers,
   and no plan, task or decision IDs in comments or log strings.
5. (PORTED) Add `crates/xagent-sandbox/tests/learned_pooling.rs` with `#[ignore]`
   tests that assert a GPU adapter under `XAGENT_REQUIRE_GPU`:
   `flag_on_kernel_runs`; `pooling_weights_birth_and_learning` (the weights equal
   their declared birth values at start and differ after a life); and
   `flag_off_matches_parity_golden`.

If the port needs any file outside `touches` (for example `gpu_kernel.rs` for
the input-kind map, or `agent/mod.rs` for full `BrainConfig` literals), stop and
report that file as a touch-limit blocker. Do not edit it.

## Non-goals

- No change to the default vision input: the hemifields remain the default.
- No re-recording of `crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden`,
  and no change to tests to make them pass.
- No edits to the free-life harness, governor schema or headless CLI. No git
  commands.

- **Done when:** INV-06 and INV-01: the focused-proof check passes, with the
  recorded verdict line matching (NOT PORTED) or the tests
  `flag_on_kernel_runs`, `pooling_weights_birth_and_learning` and
  `flag_off_matches_parity_golden` reported ok (PORTED); the parity check stays
  green with the golden byte-identical; and the fmt, clippy and guard checks pass.
