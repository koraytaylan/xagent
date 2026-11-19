---
id: critic-input-decision
title: "Offline return fit and critic-input decision record"
workstream: "0400"
kind: task
role: architect
depends_on:
  - return-fit-tool
gated: false
status: todo
merged_as: ""
touches:
  - docs/decisions/pathway-critic-inputs.md
  - docs/decisions/pathway-critic-inputs.json
checks:
  - name: focused-proof
    command: "cargo run -p xagent-sandbox --bin pathway_return_fit -- verify docs/decisions/pathway-critic-inputs.json"
    timeout_ms: 3600000
---

# Offline return fit and critic-input decision record

## Outcome

A committed decision record for the critic input, produced by running the
built `xagent` binary on the host GPU and fitting the logged returns with the
`pathway_return_fit` tool from task 0301:

- `docs/decisions/pathway-critic-inputs.json` is the machine-readable record
  written by `pathway_return_fit fit` and accepted by its `verify` subcommand.
- `docs/decisions/pathway-critic-inputs.md` is the human record: question,
  setup, method, the per-set table, the winner under the D8 rule, and what the
  winner means for the critic route.

No file other than these two changes. No source, config default or route flag
changes in this task.

## Scope

- Prerequisites. Task 0301 is merged (the `pathway_return_fit` binary with
  `fit` and `verify`). Plan `0001-free-life-harness-and-trunk-lesions` is
  integrated before this task is dispatched: its free-life runner is in
  `crates/xagent-sandbox/src/headless.rs`, its CLI flags are in
  `crates/xagent-sandbox/src/main.rs`, and its report and per-brain-tick log
  are in `crates/xagent-sandbox/src/free_life/`
  (see `docs/plans/0001-free-life-harness-and-trunk-lesions/ARCHITECTURE.md`).
  Read the exact free-life flag spellings from `main.rs` at execution time and
  copy them verbatim into the Markdown record. Do not guess flag names. Free
  life requires `vision_stride` 1 (the default); do not change it.
  The route flags (`critic_input`, `forward_pathway_enabled`) are not set by
  this task: both routes are off by default, and the logs come from the
  current code at whatever commit this task runs on. If tasks 0102 or 0201
  have landed, that does not change the run; record the full commit SHA
  (`git rev-parse HEAD`) in the Markdown record.
- Config. Dump the default config with
  `cargo run --release -p xagent-sandbox --bin xagent -- --dump-config` into a scratch
  directory outside the repository, and use that dump unchanged for every run
  (no hand edits). Record the dump's SHA-256 in the Markdown record.
- Logged free lives. Fresh birth brains (no stored champion), 10 agents per
  run, 300,000 ticks, seeds 5, 6, 7 and 8, with the per-brain-tick log enabled,
  using the free-life flags read from `main.rs`. Before starting, check
  available memory (for example `free -m`, reading `MemAvailable`) and confirm
  it covers the runs; run at most four GPU jobs concurrently, and run them one
  after another when memory does not cover four side by side. Write each
  run's stdout and stderr to `$MAKINA_SHELL_JOB/output` (the job directory),
  and write the outputs and logs to scratch outside the repository. Logs are
  not committed. Every run must exit 0. A run that fails or produces a log
  with fewer than 300,000 ticks per agent is discarded and rerun once with the
  same seed; a second failure is a blocker, not a reason to change the seed or
  ticks.
- Fit and record. Run
  `cargo run -p xagent-sandbox --bin pathway_return_fit -- fit --log <the four logs> --out docs/decisions/pathway-critic-inputs.json`.
  Then write `docs/decisions/pathway-critic-inputs.md` with:
  - question, commit SHA, config SHA-256, seeds, agents, ticks, and the exact
    free-life and fit commands;
  - method: the D8 summary (returns with death cut and terminal −1, 300-tick
    truncation, one-row alignment, 1000-tick blocks, hold-out rule, ridge with
    inner λ choice, in-view and centred gaps, winner rule). The alignment: the
    critic reads the whitened pathway inputs stored by the previous brain tick,
    so the tool sets the pathway columns of row `t` to the logged pathway inputs
    of row `t − 1` of the same agent (encoding columns and the target stay the
    row's own). The first row of each agent's log and the first row after each
    death are dropped from all three input sets, so every set fits the same
    rows. The tool applies this alignment itself; the logs are not preprocessed
    by hand;
  - one table per input set (`pathway`, `encoding`, `encoding_and_pathway`)
    with pooled held-out R² ± SE and the number of agents with positive R²,
    in-view gap ± SE and centred gap ± SE, each with its positive-agent count.
    The task 0301 JSON record carries the per-set SEs and positive-agent
    counts; copy them from the JSON fields into the table verbatim, do not
    recompute them by hand, and name the JSON as their source;
  - the winner from the JSON and why it wins under the D8 rule;
  - what it means for the critic route: if the winner is `encoding`, the
    critic route stays off and task 0402 records it as not applicable.
  - how to regenerate the logs: the commands above, with the log files not
    committed and the record naming log file names only.
- Focused check. Run `focused-proof` (the `verify` command above). It must
  exit 0 on the committed JSON. If it fails, correct the record only by
  re-running `fit` on the same logs; never hand-edit the JSON numbers or the
  winner.

## Steps

**Steps:**

1. Confirm the prerequisites: task 0301's `pathway_return_fit` binary builds,
   and the free-life flags, runner and tick-log layout exist in `main.rs`,
   `headless.rs` and `free_life/` as Scope expects. Record `git rev-parse HEAD`.
   If any flag or layout differs from Scope, report the mismatch and stop; do
   not change 0001 files.
2. Dump the default config with `--dump-config` to a scratch directory outside
   the repository and record its SHA-256. Use that dump unchanged for every run.
3. Check memory, then run the four logged fresh-brain lives (seeds 5-8, 10
   agents, 300,000 ticks, per-brain-tick log on, `vision_stride` 1) with at
   most four concurrent GPU jobs, output to `$MAKINA_SHELL_JOB/output`. Discard
   and rerun once any run that fails or is short of ticks.
4. Run the `fit` command from Scope to write `docs/decisions/pathway-critic-inputs.json`
   from the four logs.
5. Write `docs/decisions/pathway-critic-inputs.md` as Scope describes: the
   commands, SHAs, method, per-set tables filled from the JSON fields, the
   winner and its critic-route meaning. Then run `focused-proof` and require
   exit 0.

- **Done when:** `cargo run -p xagent-sandbox --bin pathway_return_fit -- verify docs/decisions/pathway-critic-inputs.json` exits 0 on the committed record produced from 40 fresh-brain 300,000-tick lives on seeds 5-8 (proves INV-10).

## Non-goals

- No change to any file other than the two `touches`. No `Cargo.toml` edit,
  no new dependency, no route flag or config default change.
- No gate run, no critic-route or forward-route record (task 0402).
- No commit of logs, scratch outputs or the config dump; the host commits the
  two record files.
- No change to the D8 rule, λ grid, block size, hold-out share or truncation.
- No planning or task names in any file written.

## Evidence and risks

- The winner may be `encoding`. That is a valid, recorded result: the critic
  route then stays off, and 0402 records it as not applicable.
- Run cost: four 300,000-tick, 10-agent GPU runs per set of seeds. Runs must
  not be started on a memory-constrained host (see Scope).
- Reference for the investigation: the critic gap and R² figures in
  `docs/reviews/2026-09-25-learner-foraging-investigation.md` (best held-out R²
  about 0.26; in-view and centred gaps) are context only. This record's numbers
  come from the new runs, not from that review.
- The per-set SEs and positive-agent counts come from task 0301's JSON record.
  If the Markdown numbers cannot be matched to those JSON fields, that is a
  contradiction to report, not a reason to edit 0301.
- Plan 0001 is an integrated prerequisite outside this plan. If its free-life
  flags or log layout differ from what Scope expects, report the mismatch
  and stop; do not change 0001 files.
