---
id: pooling-study
title: "Learned pooling against the hand-chosen hemifields (study)"
workstream: "0300"
kind: task
role: architect
depends_on:
  - modulator-study
  - pooling-fit
gated: false
status: done
merged_as: "b1e97c0afb7d93c7a3bc1fe3ef4540744b1f5418"
touches:
  - crates/xagent-sandbox/examples/pooling_study.rs
  - docs/decisions/pooling-study.md
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib pooling_study"
    timeout_ms: 3600000
  - name: verdict
    command: >-
      sh -c 'grep -Eqx "Verdict: (PORTED|NOT PORTED)" docs/decisions/pooling-study.md'
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

# 0301 — Learned pooling study against the hemifields

## Scope

Run the offline comparison of the four input arms (A hemifields, B 32 columns,
C credit-trained pooling, D Oja/PCA pooling) on logged free lives, and record
the verdict. The fit library, log format, arms, score, gate (INV-06) and
sanity constants are delivered by task `pooling-fit` in
`crates/xagent-sandbox/src/pooling_study.rs`; this task adds the runner and the
decision record only and does not change any brain, config or shader file.

## Non-goals

- No change to any file outside `touches` (the fit library is read, not
  edited), including the brain, config, shader,
  parity golden and plan 0001 harness files. The golden is not re-recorded.
- No port of any kind. Porting a winning arm is owned by task `learned-pooling-port`.
- No change to the default vision input; the hemifields remain the default.
- No git commands. No planning IDs in source doc comments or log strings.

## Fixed decisions

- **Log format, arms, score, gate, sanity values**: exactly as fixed in task
  `pooling-fit` (`tasks/0300-pooling-fit.md`), used through its public API.
- **Harness reuse**: the runner reuses plan 0001's free-life harness
  (`crates/xagent-sandbox/src/free_life/`, `metrics.rs` and `tick_log.rs`) for the
  credit fields, ground truth, metrics and tick log, and adds only the 32 adapted
  column means. `headless::run_free_life` exposes no execution-schedule option and
  no per-tick hook, so the runner runs its own life loop: each brain tick it builds
  a `FreeLifeSample` field for field as `headless.rs` `build_sample` does, feeds it
  to the harness's `AgentAccumulator` and `TickLogWriter`, and takes reward, value,
  turn innovation, food bearing and death from that sample; food in view is
  restated from the harness rule (alive, distance `<= FOOD_IN_VIEW_DISTANCE`,
  `|bearing| < VIEW_HALF_ANGLE`). It may use a schedule that writes the adapted
  visual block to scratch, stated in the record.
  `FOOD_IN_VIEW_DISTANCE = 5.5` and `VIEW_HALF_ANGLE = pi/4` are the harness's own.
  No plan 0001 file is edited.
- **Column means**: read from each agent's adapted visual block through
  `read_brain_scratch_blocking` (public, though documented as test-only), with
  offsets derived from `BrainLayout`, never hardcoded. Default vision width is 8
  (`default_vision_width` in `crates/xagent-shared/src/config.rs`); any other
  width is an error and writes no logs.
- **Sanity**: arm B against `+0.16` and arm A against `+0.33`, the library's
  named constants. Reported only; a large miss is stated as a sanity failure
  for the architect.
- **Runs**: seeds 5, 6, 7, 8; 10 agents; 300,000 ticks; current defaults after
  `modulator-study`; fresh birth brains. Logs go to `--out-dir`, never the
  repository; the record names the directory as `<job-dir>`.
- **Prerequisite**: plan 0001's harness is a transitive prerequisite through
  `0002-pathway-critic-and-forward-drive`; its interfaces are as summarised in
  `docs/plans/0002-pathway-critic-and-forward-drive/ARCHITECTURE.md`.

## Steps

**Steps:**

1. Read the `pooling-fit` library API; do not edit it.
2. Create `crates/xagent-sandbox/examples/pooling_study.rs` with required
   `--out-dir <dir>` and `--fit-only <dir>`. The default run writes one log per seed
   through the free-life harness, then runs the fit and prints the per-seed and
   pooled A–D table, sanity lines and one `Verdict:` line. The example may print;
   library code may not.
3. Run the study (architect shell): check free memory, then run in the background
   with output under `$MAKINA_SHELL_JOB/output` and an explicit timeout:
   `cargo run --release -p xagent-sandbox --example pooling_study -- --out-dir <job-dir>`.
   Re-run `--fit-only <job-dir>` and confirm the same table and verdict.
4. Write `docs/decisions/pooling-study.md`: the
   method (fixed decisions above and in `pooling-fit`, in substance); the per-seed and pooled A–D table
   with excluded and above-zero counts; the gate applied line by line with numbers;
   the sanity lines; the winning arm by name; and, only if PORTED, the winning arm's
   birth values and learning rule precisely enough to implement. Finish with exactly
   one line `Verdict: PORTED` or `Verdict: NOT PORTED`.
5. Run the checks `focused-proof`, `verdict`, `parity`, `fmt`, `clippy` and `guard`.
   Fix only failures caused by this task's changes; report others as pre-existing
   with output. Do not edit the golden or change any test to pass.

- **Done when:** INV-06 is proved: the `focused-proof` check passes with
  `planted_side_cue_is_learned_by_reference_arm`, `cue_free_log_scores_near_zero`,
  `malformed_records_are_rejected` and `gate_matches_fixed_rule` reported ok; the
  `verdict` check passes against `docs/decisions/pooling-study.md`; and the `parity`
  check stays green on the unchanged golden, so the default input is still the hemifields.
