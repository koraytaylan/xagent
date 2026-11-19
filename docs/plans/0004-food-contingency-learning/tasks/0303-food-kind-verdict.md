---
id: food-kind-verdict
title: "Gate verdict binary over the eight run summaries"
workstream: "0300"
kind: task
role: developer
depends_on:
  - food-kind-metrics
gated: false
touches:
  - crates/xagent-sandbox/src/bin/food_kind_verdict.rs
  - crates/xagent-sandbox/Cargo.toml
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --bin food_kind_verdict"
  - name: main-binary-default
    command: "cargo run -p xagent-sandbox -- --help"
---

# Gate verdict binary over the eight run summaries

## Outcome

A GPU-free binary, `crates/xagent-sandbox/src/bin/food_kind_verdict.rs`, that
recomputes the gate verdict from the eight saved per-run summaries and refuses
anything malformed. It reads no kernel and writes no run state.

## Scope

- `crates/xagent-sandbox/Cargo.toml`: add `default-run = "xagent"` under
  `[package]`. No other change and no new dependency.
- `crates/xagent-sandbox/src/bin/food_kind_verdict.rs` (new). It uses
  `xagent_sandbox::food_kind_metrics::{Arm, AgentFoodKindSummary, RunFoodKindSummary}`
  and `xagent_shared::FoodKind`.
  - CLI (clap 4 derive, existing dependency):
    `food_kind_verdict <INPUT> [--write-bundle <FILE>] [--expect-verdict <TEXT>]`.
  - Prints `Verdict: <text>`, then a GateReport table: per-seed B1/B2/B3 values for
    both arms, per-seed readout means, and per-pair link agent counts.
    User-facing println/eprintln is allowed here.

## Non-goals

- No GPU, kernel, config or metrics change. `food_kind_metrics.rs` is not edited.
- No write to `DECISION.md` or `results.json`; only `--write-bundle` writes a file.
- No plan, task, workstream, decision or spec wording in source, doc comments or
  test names (`contributing_guard.rs` enforces this).

## Decisions

- **Input.** A directory must hold exactly `learning-seed{5,6,7,8}.json` and
  `frozen-seed{5,6,7,8}.json`; any extra entry is an error. Otherwise INPUT is a
  bundle file, a JSON array of exactly those 8 runs. Both feed
  `fn verdict(runs: &[RunFoodKindSummary]) -> Result<Verdict, VerdictError>` and
  `fn gate_report(runs: &[RunFoodKindSummary]) -> Result<GateReport, VerdictError>`.
- **Validation, per run.** Exactly one run per (seed, arm) for seeds 5..=8;
  `ticks == 300_000`; `agent_count == 10 == agents.len()`; `fresh_brains`; both arms
  of a seed share `nourishing_kind`; every float finite. Each of the four turn fields
  must be `Some` for every agent, and `None` is an error. Readout fields may be
  `None`; that fails the readout link.
- **Constants.** `SEEDS = 5..=8`, `REQUIRED_TICKS: u64 = 300_000`,
  `REQUIRED_AGENTS: usize = 10`, `SEED_MAJORITY: usize = 3`,
  `READOUT_MIN_ACCURACY: f32 = 0.70`, `READOUT_MAX_SHUFFLED: f32 = 0.55`,
  `LINK_MIN_AGENTS: usize = 24` (of the 40 learning agents, per pair).
- **Signing.** Chosen reading: the metrics packet stores the four per-pair fields raw.
  `AgentFoodKindSummary` carries no nourishing kind, so it cannot sign them, and its
  accumulators sum raw values per pair kind. The verdict therefore signs each stored
  value exactly once with `food_kind_metrics::signed_for_nourishing(value, pair_kind,
  nourishing_kind)`, where `pair_kind` is First for `*_pair_first` and Second for
  `*_pair_second`, and `nourishing_kind` is that run's `nourishing_kind`. The stored
  weight-difference fields are already differences, so the verdict does not call
  `signed_weight_difference` on them, which would subtract again.
- **Behaviour (B1–B3).** A seed mean is the unweighted mean of its 10 agents.
  - B1a: mean of late toward-nourishing over all 40 learning agents is > 0.
    B1b: on at least 3 seeds, the learning seed mean exceeds the frozen seed mean.
  - B2a/B2b: the same for late away-from-harmful.
  - B3a: on at least 3 seeds, the learning pooled quarter-3 share exceeds its
    quarter-0 share. Pooled share is Σn / Σ(n+h) over the seed's agents; zero meals
    means the comparison is not met. B3b: on at least 3 seeds, the learning pooled
    quarter-3 share exceeds the frozen one.
- **Links, in order.**
  - readout: on every seed, learning mean `readout_balanced_accuracy` ≥ 0.70 and
    mean `readout_shuffled_accuracy` ≤ 0.55. Any `None` on a learning agent fails it.
  - credit: for each pair, at least 24 of the 40 learning agents have signed credit > 0.
  - weight change: the same count over the signed weight difference.
- **Verdict.** The first failing link gives `FAIL readout`, `FAIL credit` or
  `FAIL weight-change`. If all links pass and B1–B3 fail, the verdict is
  `FAIL weight-change`. Otherwise it is `PASS`.
- **Verdict text.** `Display` yields exactly `PASS`, `FAIL readout`, `FAIL credit`
  or `FAIL weight-change`. `FromStr` accepts exactly those four.
- **Exit.** I/O, parse, validation and `--expect-verdict` mismatch errors go to
  stderr with a non-zero exit.

## Steps

**Steps:**
1. Add `default-run = "xagent"` under `[package]` in Cargo.toml. Create the bin with
   a clap `Cli`, `Verdict` with Display/FromStr, and `VerdictError` with a manual
   Display (no new dependency).
2. Implement the input loader (exact-name directory or bundle) and `validate_runs`.
   Add tests `missing_seed_is_error`, `duplicate_run_is_error`, `wrong_ticks_is_error`,
   `wrong_agent_count_is_error`, `non_fresh_brains_is_error`,
   `mismatched_nourishing_kind_is_error`, `nan_is_error` and
   `directory_and_bundle_give_same_verdict`.
3. Implement `verdict` and `gate_report` with B1–B3, the three links and their order.
   Add tests `pass_fixture_is_pass` (learning late turns 0.2 against frozen 0.0,
   readout 0.9 and shuffled 0.5, every sign aligned), `readout_failure_is_named`,
   `credit_failure_alone_is_named`, `weight_change_failure_alone_is_named`,
   `readout_and_credit_failure_names_readout` and
   `behaviour_failure_with_links_passing_is_weight_change`.
4. Implement `--write-bundle` (validated runs sorted by seed then arm, pretty JSON
   array) and `--expect-verdict` (parsed with FromStr). Add tests
   `write_bundle_round_trip` and `expect_verdict_mismatch_is_error`.
5. Add tests `verdict_text_round_trip` (Display then FromStr for all four texts) and
   `verdict_text_rejects_other_strings` (for example `pass`, `FAIL  readout`, empty).
6. Wire `main` so a computed verdict prints `Verdict: <text>` and the table, then
   exits 0 when `--expect-verdict` is absent or matches. Errors go to stderr with a
   non-zero exit.

- **Done when:** the focused-proof and main-binary-default checks both pass on the host, every test named in Steps reports ok with zero failed, and INV-08 holds because the verdict uses exactly eight validated summaries with fixed thresholds, FAIL names the first failing link, and malformed input is an error.
