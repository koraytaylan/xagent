---
id: pooling-fit
title: "Offline pooling fit library (log format, arms, score, gate)"
workstream: "0300"
kind: task
role: developer
depends_on:
  - modulator-study
gated: false
status: done
merged_as: "b8b32c467874a8f9dceb3ecdaa712ef0dca0d7ed"
touches:
  - crates/xagent-sandbox/src/pooling_study.rs
  - crates/xagent-sandbox/src/lib.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib pooling_study"
    timeout_ms: 3600000
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

# 0300 — Offline pooling fit library

## Scope

The CPU-only fit library that task `pooling-study` drives: the binary log
format, four input arms that all feed the same whitened turn rule, the score,
the INV-06 gate and the sanity reference values. Unit-tested on synthetic logs.
No runner, no GPU, no study run and no decision record (owned by
`pooling-study`).

- **A** hemifields, whitened, learned (the reference; 8 inputs: mean adapted
  red, green, blue, depth over the left and right halves of the field).
- **B** 32 column means, whitened, learned (8 ray columns × 4 channels).
- **C** credit-trained pooling from the 32 column means to 2 sides × 4 channels
  (8 pooled inputs), feeding the same whitened turn rule.
- **D** Oja/PCA-learned pooling of the 32 column means to 8 components,
  whitened, learned.

Score, gate and verdict are fixed in
`docs/plans/0003-uniform-sense-motor-route/ARCHITECTURE.md` (Investigations:
Pooling; INV-06).

## Non-goals

- No change to any file outside `touches`. No brain, config, shader, golden or
  plan 0001 harness edit. No printing from library code.
- No git commands. No planning IDs in source doc comments or log strings.
- No `as` casts; every literal other than 0, 1, -1, 0.0 and 1.0 is a named
  const with a doc comment (CONTRIBUTING.md magic-number rule).

## Fixed decisions

- **Log format**: binary, little-endian. Header: the 4 bytes `XPOL`, `u32`
  column count (32), `u32` harness field count (8). Each record is 40 `f32`
  values: 32 column means (column `c` in `0..8`, channel `k` in red, green,
  blue, depth at `c * 4 + k`), then `reward`, `value`, `turn_innovation`,
  `food_bearing`, `food_in_view` (0 or 1), `death` (0 or 1), `agent`, `tick`.
  Provide a writer and a reader. The reader checks file length = header + whole
  records and that every value is finite; any other length or bad magic is an
  error naming the file.
- **Hemifield input (A)**: mean of the 4 left-column means and of the 4
  right-column means, channel by channel.
- **Adaptation and whitening** (all arms): centre by running mean (rate
  `0.01`); running covariance at the same rate; ZCA whitening from a Jacobi
  eigendecomposition refreshed every `VISION_WHITENING_REFRESH` ticks (the
  constant in `xagent_brain::senses`); epsilon named in `pooling_study.rs`.
- **Turn output**: pre-noise `w · z`, evaluated on each record before its
  update.
- **Credit rule** (all arms): `delta = reward + gamma * V_next - value`,
  `V_next = 0` on death; trace
  `e = gamma * lambda * e + turn_innovation * z / (1 + |x|²)` with `x` the
  arm's raw input, zeroed on death; `w += ACTION_WEIGHT_LEARNING_RATE * delta * e`.
  Constants as named in `brain_passes.wgsl` and `config.rs`.
- **Arm C**: `P` (8 × 32) born as the hemifield averaging matrix; `g = Wᵀ w`;
  `P_ij += ETA * delta * g_i * c_j / (1 + |c|²)`, `ETA` equal to the turn
  learning rate.
- **Arm D**: Oja/Sanger, 8 components, `Q` (8 × 32) born as the first 8
  standard basis vectors, on centred column means, step `OJA_RATE = 0.01`,
  normalised by `1 + |c|²`; the components then go through arm A's rule.
- **Score**: per agent, Pearson correlation over records with
  `food_in_view == 1` and tick `>= 0.7 * 300000`, between the turn output and
  the food side, `-sign(food_bearing)` (the free-life harness's convention,
  metrics.rs `food_side`). Agents with fewer than 30 such records are excluded
  and counted, not scored zero. Summaries report mean ± standard error over
  scored agents and the count above zero.
- **Gate (INV-06)**: `best = max(mean(C), mean(D))`; `Verdict: PORTED` iff
  `best >= mean(A)` (ties port), else `Verdict: NOT PORTED`. Implemented as
  `pooling_study::gate_verdict`; the runner uses it for the printed and
  recorded verdict.
- **Sanity values**: arm B `+0.16` and arm A `+0.33` (investigation "A whitened
  visual pathway, offline",
  `docs/reviews/2026-09-25-learner-foraging-investigation.md`), as named
  constants.

## Steps

**Steps:**

1. Create `crates/xagent-sandbox/src/pooling_study.rs` and add
   `pub mod pooling_study;` to `crates/xagent-sandbox/src/lib.rs`. Implement
   the log header, record reader and writer with the errors above; the four
   arms; the score and per-arm summary; `gate_verdict`; and the sanity values.
   Expose a fit entry point that takes the records of one log and returns
   per-agent arm scores, for the runner.
2. In a `tests` submodule add `planted_side_cue_is_learned_by_reference_arm`
   (a food side cue in the hemifield channels gives arm A a positive score),
   `cue_free_log_scores_near_zero` (arm A's mean within a named tolerance of
   zero), `malformed_records_are_rejected` (bad magic, short file, non-finite
   value and truncated last record each error) and `gate_matches_fixed_rule`
   (PORTED exactly when `max(C, D) >= A`, tie included). Test logs go to a
   temporary directory, never the repository.
3. Run the checks `focused-proof`, `fmt`, `clippy` and `guard`. Fix only
   failures caused by this task; report others as pre-existing with output.

- **Done when:** the `focused-proof` check passes with
  `planted_side_cue_is_learned_by_reference_arm`,
  `cue_free_log_scores_near_zero`, `malformed_records_are_rejected` and
  `gate_matches_fixed_rule` reported ok, and the `fmt`, `clippy` and `guard`
  checks pass.
