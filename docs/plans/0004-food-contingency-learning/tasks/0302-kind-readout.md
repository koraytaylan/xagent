---
id: kind-readout
title: "Supervised kind readout (measurement only)"
workstream: "0300"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-brain/src/lib.rs
  - crates/xagent-brain/src/kind_readout.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib kind_readout"
---

# Supervised kind readout (measurement only)

## Outcome

A pure CPU module, `crates/xagent-brain/src/kind_readout.rs`, measures whether a
binary label is linearly readable: held-out balanced accuracy of an L2 logistic
regression under stratified k-fold, plus a shuffled-label control. It takes copied
values only. Purpose and gate: PLAN.md. Guard INV-06: ARCHITECTURE.md.

## Scope

- `crates/xagent-brain/src/lib.rs`: add `pub mod kind_readout;` after
  `pub mod gpu_kernel;`. Change nothing else.
- `crates/xagent-brain/src/kind_readout.rs` (new), in order:
  1. `//!` doc: purpose; functions only read their inputs. Name no plan, task,
     workstream, decision record or spec.
  2. Named constants, each with a `///` doc giving its technical reason:
     `READOUT_FOLDS: usize = 5`; `MIN_SAMPLES_PER_CLASS_PER_FOLD: usize = 2`
     (class minimum `READOUT_FOLDS * MIN_SAMPLES_PER_CLASS_PER_FOLD` = 10);
     `READOUT_L2_STRENGTHS: [f64; 3] = [0.01, 0.1, 1.0]`;
     `READOUT_ITERATIONS: usize = 300` (full-batch steps from zero weights);
     `READOUT_LEARNING_RATE: f64 = 1.0` (stable on standardised features: logistic
     curvature is at most a quarter of the covariance's top eigenvalue);
     `READOUT_STANDARD_DEVIATION_FLOOR: f64 = 1e-6` (finite standardisation for a
     constant feature); `READOUT_FOLD_SEED: u64` (fixed, so folds do not depend on
     the caller); `HALF: f64 = 0.5`; splitmix64 increment, two multipliers and
     three shifts, each named.
  3. `pub struct ReadoutSample { pub features: Vec<f32>, pub label: bool }` and
     `pub struct ReadoutScore { pub balanced_accuracy: f32, pub shuffled_balanced_accuracy: f32, pub samples: usize, pub positives: usize }`.
     Derive `Debug, Clone, PartialEq`; `///` on every field.
  4. Private helpers: splitmix64; Fisher-Yates over `usize` indices and `bool`
     labels; `to_f32` (`f64` to `f32`) with `#[allow(clippy::cast_possible_truncation)]`.
  5. `pub fn fit_kind_readout(samples: &[ReadoutSample], shuffle_seed: u64) -> Option<ReadoutScore>`
     and the private procedure under Decisions.
- Tests: the `#[cfg(test)] mod tests` block in `kind_readout.rs`. Names start with
  `readout_`. Synthetic sizes and thresholds are named test constants.
- Every `pub` item has a `///` doc. The file passes
  `cargo clippy --workspace --all-targets -- -D warnings` (pedantic). No `as` cast
  on an integer or float, except in `to_f32`.

Non-goals: no dependency or `Cargo.toml` change (no `rand`); no access to kernel,
buffers, physics, food, sense or config state; no CLI, file I/O, printing or
logging; no per-kind food logic; no re-export; no edits to other files.

Prerequisites: none (standard library only).

## Decisions

- **Entry point.** Return `None`, not a zero score, when: `samples` is empty; any
  feature length differs from the first sample's or is 0 (zero-length features);
  any feature is NaN or infinite; either class has fewer than
  `READOUT_FOLDS * MIN_SAMPLES_PER_CLASS_PER_FOLD`; the sample count does not fit
  `u32` (`u32::try_from`, then `f64::from`).
- **Score fields.** `samples` is the sample count; `positives` counts `label == true`.
- **Stratified folds (real labels).** Seed `READOUT_FOLD_SEED`, independent of
  `shuffle_seed`. Shuffle each class's indices with splitmix64 seeded by it, then cut
  each class into `READOUT_FOLDS` contiguous blocks: block `k` is
  `[k * len / READOUT_FOLDS, (k + 1) * len / READOUT_FOLDS)` (integer, `checked_mul`).
  Fold `k` tests on block `k` of both classes and trains on the rest.
- **Standardisation.** Per fold and feature, mean and population standard deviation
  come from the training fold only, in `f64`; the deviation is
  `max(sd, READOUT_STANDARD_DEVIATION_FLOOR)`. Test samples use those statistics.
- **Logistic fit.** Weights and bias start at zero. Each of `READOUT_ITERATIONS`
  full-batch steps computes `p = sigmoid(bias + w · x)`, the gradient
  `(1/n) Σ (p − y) x + λ w` for weights, `(1/n) Σ (p − y)` for bias (unregularised),
  then `w -= READOUT_LEARNING_RATE × grad`. `y` is 1 for `true`, 0 for `false`.
  Predict positive when `bias + w · x > 0`.
- **Strength selection.** Per strength, per-fold balanced accuracy is
  `HALF × (TP / P + TN / N)` with the fold's own `P` and `N`; the strength's score is
  its mean over folds. The best is the first strength with the highest mean (strict
  `>`, so ties keep the lower strength). The procedure returns that mean.
- **Shuffled control.** Fisher-Yates permutes the labels with splitmix64 seeded by
  `shuffle_seed`; the same procedure then runs on them with the same features and
  `READOUT_FOLD_SEED`, selecting its own best strength. The result is
  `shuffled_balanced_accuracy`. Class counts are preserved.
- **Splitmix64.** Add the increment to the `u64` state, two xor-shift-multiply mixes,
  then a final xor-shift. Shuffles reduce each draw modulo the bound and convert with
  `usize::try_from`; on error the enclosing function returns `None`. No `.expect()`.

## Steps

**Steps:**

1. Add the `lib.rs` line. Create `kind_readout.rs` with the `//!` doc, constants,
   `ReadoutSample`, `ReadoutScore`, splitmix64, shuffle helpers and `to_f32`. A stub
   `fit_kind_readout` is allowed only until step 2.
2. Implement validation as Decisions lists it, run the procedure on real and
   permuted labels, and assemble the score. Add
   `readout_insufficient_class_counts_is_none` (9 positives with 100 negatives is
   `None`; exactly 10 with 100 negatives is `Some`),
   `readout_non_finite_feature_is_none` (one NaN, one infinite feature, each on an
   otherwise valid set: `None`) and `readout_mismatched_or_empty_input_is_none`
   (empty slice, mismatched length, zero-length features: each `None`).
3. Implement the procedure in helpers, each under about 50 lines. Add
   `readout_separable_scores_high_and_shuffled_low`: 400 samples, 12 features uniform
   in `[-1, 1)` from splitmix64 with a fixed test seed; label `w · x + noise > 0`,
   `w` cycling `[1, -1, 0.5, -0.5]`, `noise` uniform in `[-0.1, 0.1)`. Assert
   `balanced_accuracy >= 0.95`, `shuffled_balanced_accuracy <= 0.65`,
   `samples == 400`, `positives` equal to the count of `true` labels.
4. Add `readout_noise_labels_near_chance` (400 samples, labels from independent
   generator bits; `balanced_accuracy` in `[0.35, 0.65]`) and
   `readout_imbalanced_separable_uses_balanced_accuracy` (1,200 step-3 candidates,
   keep every positive and every fourth negative; `negatives * 2 < positives` and
   `balanced_accuracy >= 0.90`).
5. Add `readout_is_deterministic` (two identical calls; every field equal by
   `to_bits()`) and `readout_shuffle_seed_changes_only_control` (two shuffle seeds on
   step 3's set; `balanced_accuracy`, `samples`, `positives` equal by bit; both
   shuffled values in `[0, 1]`).

- **Done when:** the focused-proof check passes on the host with every test named in Steps reported ok and zero failed, the separable test asserts step 3's thresholds, and the seed test shows only the shuffled control depends on the seed; this proves INV-06: fit_kind_readout is a pure function over a borrowed slice that constructs no kernel and writes no brain, physics, food or config state.

## Risks

- The worker has no shell. If a host run fails a statistical threshold, change an
  implementation constant (iterations or learning rate), never the test threshold,
  and report the observed value and the constant changed.
