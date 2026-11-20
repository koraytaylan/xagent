---
id: supervisor-verdict
title: "Fail-closed supervisor verdict binary"
workstream: "0700"
kind: task
role: developer
depends_on: [run-summary]
gated: false
touches: [crates/xagent-sandbox/src/bin/supervisor_verdict.rs]
status: done
merged_as: "2482a7ce95cf9875544805ca216ee142218a57f6"
checks:
  - {name: focused-proof, command: "cargo test -p xagent-sandbox --bin supervisor_verdict"}
  - {name: main-binary-help, command: "cargo run -p xagent-sandbox -- --help"}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Fail-closed supervisor verdict binary

## Outcome

A GPU-free, auto-discovered binary `crates/xagent-sandbox/src/bin/supervisor_verdict.rs`
that applies the PLAN.md "Decision rule" to exactly 32 validated inputs and prints
`ADOPT`, `OPTION` or `REJECT`, or fails closed. Its first stdout line is `Verdict: <X>`.

## Scope

- The one new file. Reuse `xagent_sandbox::run_summary::{RunSummary, KindWindowTotals}`
  (0201) and `xagent_sandbox::headless::FreeLifeReport` (0001); do not redefine them.
- clap derive CLI: `supervisor_verdict <INPUT> [--write-bundle FILE] [--expect-verdict ADOPT|OPTION|REJECT]`.

## Non-goals

- No GPU, no run, no change to `RunSummary` or `FreeLifeReport`, no `Cargo.toml` edit.
  If `default-run = "xagent"` is missing from `crates/xagent-sandbox/Cargo.toml`, report it; do not edit.
- No plan, task, workstream or INV identifiers in source, doc comments or test names (`contributing_guard`).

## Prerequisites

- 0201 `run-summary`: `RunSummary` and `KindWindowTotals` as in ARCHITECTURE.md "Run summary".
- 0001: `FreeLifeReport` (`seed`, `agent_count`, `ticks`, `brain_source`, per-agent
  `AgentMetrics` with `late: WindowMetrics`); `WindowMetrics` fields `meals`,
  `alive_samples`, `turn_toward_food`, `sighting_meal_share`.
- 0004 layout: `src/bin` auto-discovery and `default-run`.

## Decisions

- Documented named consts, no literal copies: seeds `5..=8`, 10 agents, 32,000,000
  agent-ticks per summary, 300,000 ticks per free life, 3-of-4 seed majority.
- Input directory holds exactly the 32 files: `{generations,supervisor,weight-copy}-seed{N}.json`
  and `{generations,supervisor}-kinds-seed{N}.json` (`RunSummary`), and
  `free-life-{generations,supervisor,weight-copy}-seed{N}.json` (`FreeLifeReport`).
  Missing, duplicate or extra entries are errors. Any other INPUT is a bundle file.
- Per summary: mode `generations` (incl. `-kinds`) or `supervisor` (incl. `-kinds`, `weight-copy`);
  inheritance `None`, `Darwinian`, `WeightCopy` respectively; `seed` equals the file's seed;
  `agent_count` 10; `agent_ticks` 32,000,000; `food_kinds_enabled` false for the three base
  families, true for `-kinds`; `kinds` is `Some` exactly when food kinds are on; all numbers finite.
- Per free life: `seed` matches; `agent_count` 10 and `agents.len()` 10; `ticks` 300,000;
  `brain_source` `"champion"` for generations, `"archive"` for supervisor and weight-copy; finite.
- Metrics, per PLAN.md "Decision rule":
  - Per seed `F = nourishing_meals / agent_ticks`, `D = deaths / agent_ticks` (food-kinds-off runs).
    Pooled values are sums over the four seeds before dividing.
  - `L1 = Σ late.meals / Σ late.alive_samples`; `L2`, `L3` are means of `late.turn_toward_food`
    and `late.sighting_meal_share` over the 40 free-life agents.
  - `K1 = Σ late_toward_nourishing_sum / Σ late_toward_nourishing_samples`; `K2` likewise with
    `late_away_harmful_*`; `K3 = Σ late_nourishing_meals / (Σ late_nourishing_meals + Σ late_harmful_meals)`.
  - Any zero denominator: `VerdictError::UndefinedMetric(name)`. No defaults.
- `fn decide(supervisor: &Metrics, generations: &Metrics) -> Verdict` takes only those two
  arms, so weight copy cannot influence it. Use `>=` and `<=` exactly as PLAN.md writes them.
- Weight-copy metrics are computed and printed in the table only.
- Output: `Verdict: <X>`, then a table with supervisor, generations and weight-copy columns.
- Bundle: pretty JSON with `format` `"xagent-supervisor-verdict-bundle"`, `version` 1,
  `runs` and `free_lives` as objects keyed by file stem (BTreeMap), `metrics`, `verdict`.
  Bundle input revalidates, recomputes, and errors if stored `metrics` or `verdict` differ.
- Exit codes: success 0. Expect mismatch 2, via pure `fn expect_exit_code(computed: Verdict,
  expected: Option<Verdict>) -> i32` printing both verdicts to stderr. Any input error 1 with
  the error on stderr. Errors are manual `Display` and `Error` impls; no new dependency.
  `tempfile` is an existing dev-dependency.

## Steps

**Steps:**
1. Add the documented constants, `Verdict` (Display and FromStr for exactly ADOPT, OPTION, REJECT),
   `VerdictError`, the clap `Cli`, and the exact 32-name directory loader plus the bundle entry point.
2. Validate each `RunSummary` and `FreeLifeReport` as in Decisions. Tests: `verdict_rejects_missing_file`,
   `verdict_rejects_extra_file`, `verdict_rejects_wrong_agent_ticks`, `verdict_rejects_wrong_mode_or_inheritance`,
   `verdict_rejects_wrong_brain_source`, `verdict_rejects_non_finite_value`.
3. Implement the `Metrics` struct (F and D per seed, pooled F and D, L1 to L3, K1 to K3, weight-copy arm)
   with `UndefinedMetric` on zero denominators. Test: `verdict_rejects_undefined_metric`.
4. Implement `decide` as in Decisions. Tests: `verdict_adopt_when_three_seeds_and_learning_hold`,
   `verdict_option_when_two_seeds_but_nothing_worse`, `verdict_reject_when_a_learning_metric_is_worse`,
   `verdict_reject_when_pooled_deaths_worse`, `verdict_weight_copy_never_changes_verdict`.
5. Implement the stdout verdict line and metrics table, `--write-bundle`, and bundle input with the
   recompute check. Tests: `verdict_bundle_round_trip`, `verdict_bundle_with_tampered_verdict_is_error`.
6. Implement `expect_exit_code` and `main` exit handling (0, 1, 2). Test: `verdict_expect_mismatch_exit_code`.

Unit tests use synthetic in-memory inputs built by helpers, plus tempfile directories for directory input.

- **Done when:** the focused-proof check reports all fifteen tests named in Steps ok with zero failed, the main-binary-help check passes, and the binary's behaviour matches Decisions, proving INV-14: the verdict comes only from the 32 validated inputs under the PLAN.md rule, fails closed with exit codes 1 and 2, and weight copy never changes the verdict.

## Evidence and risks

- The worker has no shell; every proof is a host-run check named in the frontmatter.
- Field names come from the 0201 and 0001 packets. If the landed `FreeLifeReport` or `WindowMetrics`
  differs, report the mismatch with file and line; edit nothing outside the touched file.
- `contributing_guard` and `clippy -D warnings` apply: document every `pub` item and avoid unused code.
