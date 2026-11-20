---
id: pooling-study
title: "Learned pooling against the hand-chosen hemifields (study)"
workstream: "0300"
kind: task
role: architect
depends_on:
  - modulator-study
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-sandbox/src/pooling_study.rs
  - crates/xagent-sandbox/src/lib.rs
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

Offline comparison, on logged free lives, of four input arms that all feed the
same whitened turn rule. This task produces the verdict and the decision record
only; it does not change any brain, config or shader file.

- **A** hemifields, whitened, learned (the reference; 8 inputs: mean adapted
  red, green, blue, depth over the left and right halves of the field).
- **B** 32 column means, whitened, learned (8 ray columns × 4 channels).
- **C** credit-trained pooling from the 32 column means to 2 sides × 4 channels
  (8 pooled inputs), feeding the same whitened turn rule.
- **D** Oja/PCA-learned pooling of the 32 column means to 8 components,
  whitened, learned.

Score, gate and verdict are fixed in `docs/plans/0003-uniform-sense-motor-route/ARCHITECTURE.md`
(Investigations: Pooling; INV-06).

## Non-goals

- No change to any file outside `touches`, including the brain, config, shader,
  parity golden and plan 0001 harness files. The golden is not re-recorded.
- No port of any kind. Porting a winning arm is owned by task `learned-pooling-port`.
- No change to the default vision input; the hemifields remain the default.
- No git commands. No planning IDs in source doc comments or log strings.

## Fixed decisions

- **Log format** (written by the runner, read by the fit): binary, little-endian.
  Header: the 4 bytes `XPOL`, `u32` column count (32), `u32` harness field count (8).
  Each record is 40 `f32` values: 32 column means (column `c` in `0..8`, channel
  `k` in red, green, blue, depth at `c * 4 + k`), then `reward`, `value`,
  `turn_innovation`, `food_bearing`, `food_in_view` (0 or 1), `death` (0 or 1),
  `agent`, `tick`. The reader checks file length = header + whole records and that
  every value is finite; any other length or bad magic is an error naming the file.
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
- **Hemifield input (A)**: mean of the 4 left-column means and of the 4
  right-column means, channel by channel.
- **Adaptation and whitening** (all arms): centre by running mean (rate `0.01`);
  running covariance at the same rate; ZCA whitening from a Jacobi
  eigendecomposition refreshed every `VISION_WHITENING_REFRESH` ticks (from
  `buffers.rs`); epsilon named in `pooling_study.rs`.
- **Turn output**: pre-noise `w · z`, evaluated on each record before its update.
- **Credit rule** (all arms): `delta = reward + gamma * V_next - value`, `V_next = 0`
  on death; trace `e = gamma * lambda * e + turn_innovation * z / (1 + |x|²)` with
  `x` the arm's raw input, zeroed on death; `w += ACTION_WEIGHT_LEARNING_RATE * delta * e`.
  Constants as named in `brain_passes.wgsl` and `config.rs`.
- **Arm C**: `P` (8 × 32) born as the hemifield averaging matrix; `g = Wᵀ w`;
  `P_ij += ETA * delta * g_i * c_j / (1 + |c|²)`, `ETA` equal to the turn learning rate.
- **Arm D**: Oja/Sanger, 8 components, `Q` (8 × 32) born as the first 8 standard
  basis vectors, on centred column means, step `OJA_RATE = 0.01`, normalised by
  `1 + |c|²`; the components then go through arm A's rule.
- **Score**: per agent, Pearson correlation over records with `food_in_view == 1`
  and tick `>= 0.7 * 300000`, between the turn output and the food side,
  `-sign(food_bearing)` (the free-life harness's convention, metrics.rs `food_side`).
  Agents with fewer than 30 such records are excluded and counted, not scored zero.
  Report mean ± standard error over scored agents and the count above zero.
- **Gate (INV-06)**: `best = max(mean(C), mean(D))`; `Verdict: PORTED` iff
  `best >= mean(A)` (ties port), else `Verdict: NOT PORTED`. Implemented as
  `pooling_study::gate_verdict`, used by the runner for the printed and recorded verdict.
- **Sanity**: arm B against `+0.16` and arm A against `+0.33` (investigation
  "A whitened visual pathway, offline", `docs/reviews/2026-09-25-learner-foraging-investigation.md`).
  Reported only; a large miss is stated as a sanity failure for the architect.
- **Runs**: seeds 5, 6, 7, 8; 10 agents; 300,000 ticks; current defaults after
  `modulator-study`; fresh birth brains. Logs go to `--out-dir`, never the
  repository; the record names the directory as `<job-dir>`.
- **Prerequisite**: plan 0001's harness is a transitive prerequisite through
  `0002-pathway-critic-and-forward-drive`; its interfaces are as summarised in
  `docs/plans/0002-pathway-critic-and-forward-drive/ARCHITECTURE.md`.

## Steps

**Steps:**

1. Create `crates/xagent-sandbox/src/pooling_study.rs` and add `pub mod pooling_study;`
   to `crates/xagent-sandbox/src/lib.rs`. Implement the log header, record reader
   and writer with the errors above; the four arms; the score; `gate_verdict`; and
   the sanity values as named constants. In a `tests` submodule add
   `planted_side_cue_is_learned_by_reference_arm` (a food side cue in the hemifield
   channels gives arm A a positive score), `cue_free_log_scores_near_zero` (arm A's
   mean within a named tolerance of zero), `malformed_records_are_rejected` (bad
   magic, short file, non-finite value and truncated last record each error) and
   `gate_matches_fixed_rule` (PORTED exactly when `max(C, D) >= A`, tie included).
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
   method (fixed decisions above, in substance); the per-seed and pooled A–D table
   with excluded and above-zero counts; the gate applied line by line with numbers;
   the sanity lines; the winning arm by name; and, only if PORTED, the winning arm's
   birth values and learning rule precisely enough to implement. Finish with exactly
   one line `Verdict: PORTED` or `Verdict: NOT PORTED`.
5. Run the checks `focused-proof`, `verdict`, `parity`, `fmt`, `clippy` and `guard`.
   Fix only failures caused by this task's changes; report others as pre-existing
   with output. Do not edit the golden or change any test to pass.
6. Before submitting, sweep every file in `touches` once, tests included, for the
   CONTRIBUTING.md Magic Numbers rule (only `0`, `1`, `-1`, `0.0`, `1.0` unnamed):
   `powi` exponents, array lengths, index and count thresholds, seeds, loop
   bounds, slice offsets and arm ids all need a named, documented `const` (or an
   equivalent form such as `d * d`), so review does not find sites one by one.

- **Done when:** INV-06 is proved: the `focused-proof` check passes with
  `planted_side_cue_is_learned_by_reference_arm`, `cue_free_log_scores_near_zero`,
  `malformed_records_are_rejected` and `gate_matches_fixed_rule` reported ok; the
  `verdict` check passes against `docs/decisions/pooling-study.md`; and the `parity`
  check stays green on the unchanged golden, so the default input is still the hemifields.
