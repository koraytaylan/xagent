---
id: "modulator-study"
title: "Judge, land or revert each modulator removal"
workstream: "0400"
kind: task
role: architect
depends_on:
  - modulator-flags
  - declared-sense-extension-test
gated: false
status: todo
merged_as: ""
touches:
  - docs/decisions/modulator-study.md
  - docs/decisions/modulator-study-evo-reference.json
  - crates/xagent-shared/src/config.rs
  - crates/xagent-sandbox/tests/modulator_flags.rs
  - crates/xagent-sandbox/tests/fixtures/sense_block_parity.golden
checks:
  - name: focused-proof
    command: "sh -c 'grep -Eq \"^Klinotaxis turn-gain: (LANDED|REVERTED)$\" docs/decisions/modulator-study.md && grep -Eq \"^Motor fatigue: (LANDED|REVERTED)$\" docs/decisions/modulator-study.md'"
  - name: flags
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test modulator_flags -- --include-ignored"
    timeout_ms: 3600000
  - name: config-tests
    command: "cargo test -p xagent-shared"
    timeout_ms: 1800000
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

# 0402 — Judge, land or revert each modulator removal

## Scope

- Run the modulator study (architect shell) against the fixed rule in
  `docs/plans/0003-uniform-sense-motor-route/ARCHITECTURE.md`
  (Investigations: Modulators; INV-07, INV-08), which mirrors plan 0002's
  route gates (`docs/plans/0002-pathway-critic-and-forward-drive/ARCHITECTURE.md`,
  D9).
- Record the reference config (with the commit it was dumped at), the
  within-life and 80-generation per-seed numbers, the rule applied line by
  line, and the two verdicts in
  `docs/decisions/modulator-study.md`.
- Land or revert each modulator independently, judged against the control
  alone (no combined arm). Landing flips only that flag's default to `false`;
  reverting changes no code.

## Non-goals

- No new study arm, metric, threshold or window beyond those fixed in
  ARCHITECTURE.md. The 80-generation comparison window is generations 60–78
  of an 80-generation run.
- No change to the free-life harness, the headless CLI, the governor schema or
  plan 0001/0002 files: they are read and called, never edited.
- No change to the champion database fixture (`sense_block_champion.db`).
- No change to any preset or default other than the flag(s) a LANDED verdict
  names.
- No git commands. The host commits this packet's files after checks.

## Study inputs

- Free-life runner: plan 0001's free-life harness (metrics in
  `crates/xagent-sandbox/src/free_life/`, CLI in `crates/xagent-sandbox/src/main.rs`,
  documented in `docs/plans/0001-free-life-harness-and-trunk-lesions/`). Plan 0001
  is a transitive prerequisite (this task depends on 0002, which depends on 0001),
  so its runner exists when this task runs. Copy
  the exact flag, metric and log names from those sources at execution time
  into the decision record. Do not guess names.
- Headless CLI spellings (`--seed`, `--no-render`, `--generations`, `--config`,
  `--db`, `--dump-config`) are read from `crates/xagent-sandbox/src/main.rs`
  and copied verbatim into the record.
- Reference config: `--dump-config` output (default config) with
  `governor.tick_budget` set to 40000, written to
  `docs/decisions/modulator-study-evo-reference.json`. Record
  the commit (`git rev-parse HEAD` output at dump time) and the exact commands
  in the decision record. The JSON must parse as `FullConfig`.
- Three arms, differing only in the modulator flags: control (both flags as
  defaulted at this point); klinotaxis off (`klinotaxis_turn_gain_enabled`
  false); fatigue off (`motor_fatigue_enabled` false). Off-variants are copies
  of the reference JSON with that one flag changed, kept in an ignored scratch
  directory under `target/`, never in tracked paths.
- Within-life arms: fresh birth brains, seeds 5, 6, 7, 8, 10 agents per seed
  per arm (40 agents per arm), 300,000 ticks. Per-agent values are compared
  control versus arm.
- 80-generation arms: seeds 5, 6, 7, 8, each arm, `--seed S --no-render
  --generations 80 --config <arm config> --db <fresh database under target/>`,
  one fresh database per run.
- Standard error: for each metric, SE is the Welch standard error of the
  difference between the 40 arm agents and the 40 control agents, computed on
  that metric's per-agent values (for the sightings share, the per-agent
  share).
- Machine limits: check available memory before starting; run long jobs in the
  background with output under `$MAKINA_SHELL_JOB/output`; at most two GPU jobs
  at once; use explicit timeouts.

## Verdict rule (applied per modulator against the control)

LAND iff ALL of:

1. arm minus control mean meals per agent ≥ −SE;
2. arm minus control mean deaths per agent ≤ +SE;
3. pooled share of sightings ending in a meal ≥ control share − SE;
4. in every seed 5–8, the arm's mean `composite_fitness` over generations
   60–78 of its 80-generation run is at least the control's over the same
   seed and window.

Otherwise REVERT.

**Steps:**

1. Dump the default config with `--dump-config`, set `governor.tick_budget` to
   40000, write `docs/decisions/modulator-study-evo-reference.json`,
   confirm it parses as `FullConfig` JSON, and create the two off-variant
   configs under `target/` from it. Record the dump commit and commands.
2. Run the within-life arms (control and both off-variants, seeds 5–8, 10
   agents, 300,000 ticks) through the plan 0001 free-life harness. Record per
   seed and arm: meals, deaths (and deaths on hazard ground), sightings, share
   of sightings ending in a meal, and size-weighted turn toward food and away
   from hazard, as per-seed tables in the decision record.
3. Run the 80-generation arms (seeds 5–8, each arm) with fresh databases under
   `target/`. Record per seed and arm: mean `composite_fitness` over
   generations 60–78 read from the governor schema, plus meals and deaths over
   the same window, as per-seed tables in the decision record.
4. Apply the verdict rule to klinotaxis and fatigue separately against the
   control. Write each rule clause with its numbers in the decision record,
   then exactly these two lines, each on its own line: `Klinotaxis turn-gain:
   LANDED` or `Klinotaxis turn-gain: REVERTED`, and `Motor fatigue: LANDED` or
   `Motor fatigue: REVERTED`.
5. Code change per verdict. For each LANDED modulator: in
   `crates/xagent-shared/src/config.rs` set its default to `false` in the
   `BrainConfig` `Default` impl, its serde default and every preset in that
   file; update the default assertions in
   `crates/xagent-sandbox/tests/modulator_flags.rs`; re-record only the parity
   golden with `env XAGENT_RECORD_SENSE_PARITY=1 XAGENT_REQUIRE_GPU=1 cargo
   test -p xagent-sandbox --test sense_block_parity -- --include-ignored
   --test-threads=1`, and state in the decision record that default behaviour
   legitimately changed, which is why the golden was re-recorded. The champion
   database fixture is untouched. For each REVERTED modulator make no code
   change. If both are reverted, leave `config.rs`, `modulator_flags.rs` and
   the golden unchanged and do not re-record. Then run the checks `flags`,
   `config-tests`, `parity`, `fmt`, `clippy`, `guard` and `focused-proof`; fix
   only failures caused by this task's changes and report any other failure as
   pre-existing with its output.

- **Done when:** INV-08 and INV-07 are proved: the `focused-proof` check finds both verdict lines in
  `docs/decisions/modulator-study.md`, that
  record holds the per-seed within-life and 80-generation numbers for control
  and each arm with the rule applied line by line (INV-08), and the
  `flags`, `config-tests` and `parity` checks pass on the resulting defaults,
  so each flag is off by default exactly when its verdict is LANDED (INV-07).
