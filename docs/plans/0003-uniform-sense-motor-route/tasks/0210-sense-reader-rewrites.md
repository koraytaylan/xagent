---
id: sense-reader-rewrites
title: "Move out-of-file WGSL readers off the sense-specific names"
workstream: "0200"
kind: task
role: developer
depends_on:
  - sense-override-constants
  - generic-sense-helpers
  - generic-schedule-candidates
gated: false
status: done
merged_as: "f3e7cb0345f067c4101ba0ba497f2f40c61789c0"
touches:
  - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
  - crates/xagent-brain/src/gpu_kernel/whitening_validation.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- whitening_validation"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
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

# 0210 — Move out-of-file WGSL readers off the sense-specific names

## Outcome

`kernel_tick.wgsl`, `phase_death.wgsl` and `gpu_kernel/whitening_validation.rs`
no longer read `vision_pathway_learning_rate`, `SCENT_CHANNELS` or
`VISION_WHITENING_REFRESH`, so `remove-sense-specific-wgsl` can delete those
names from `common.wgsl` without touching these files. Behaviour, the parity
golden and every test result are unchanged.

## Decisions

- In `kernel_tick.wgsl` and `phase_death.wgsl` (terminal forward-pathway
  settle), replace each `vision_pathway_learning_rate(brain_base)` with
  `sense_learning_rate(brain_base, 1u)` (the wrapper's exact body in
  `common.wgsl`) and each `SCENT_CHANNELS` with `SENSE0_CHANNELS` (the
  generated override, default `2u`, equal to `SCENT_CHANNELS`). If these lines
  already read `sense_learning_rate(brain_base, 1u)` and
  `NON_VISUAL_FEATURE_CHANNELS` (same value), leave them as they are.
- In `gpu_kernel/whitening_validation.rs`, change only the anchor
  `const VISION_WHITENING_REFRESH: u32 = {REFRESH_CYCLES}u;` to
  `override SENSE1_REFRESH_TICKS: u32 = {REFRESH_CYCLES}u;` (default `20u`,
  equal to `REFRESH_CYCLES`). Leave the `REFRESH_BLOCK` literal as it is.
- Stop rule: if `sense_learning_rate`, `SENSE0_CHANNELS` or
  `override SENSE1_REFRESH_TICKS` is missing from `common.wgsl`, stop and
  report; edit no other file.

**Steps:**

1. Apply the two WGSL rewrites, changing nothing else on those lines.
2. Apply the one anchor change in `whitening_validation.rs`.
3. Run the checks.

- **Done when:** no touched file contains `vision_pathway_learning_rate`,
  `SCENT_CHANNELS` or `const VISION_WHITENING_REFRESH`; the focused-proof,
  brain-lib and parity (golden byte-identical) checks pass; fmt, clippy and
  guard pass.

## Non-goals

- No edit to `common.wgsl`, `brain_passes.wgsl` or any other file; no new name,
  value or test; no TODO, plan or task names in comments.
