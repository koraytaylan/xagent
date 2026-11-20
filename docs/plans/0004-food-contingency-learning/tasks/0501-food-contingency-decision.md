---
id: food-contingency-decision
title: "Food-contingency gate runs and decision record"
workstream: "0500"
kind: task
role: architect
depends_on:
  - food-kind-free-life
  - food-kind-verdict
gated: false
touches:
  - docs/decisions/food-contingency.md
  - docs/decisions/food-contingency-results.json
status: done
merged_as: "83777681dc29bbbac226b162fcfcae6903de1d6d"
checks:
  - name: decision-consistency
    command: |-
      sh -c 'v=$(grep -m1 "^Verdict: " docs/decisions/food-contingency.md | cut -d" " -f2-) && test -n "$v" && cargo run -p xagent-sandbox --bin food_kind_verdict -- docs/decisions/food-contingency-results.json --expect-verdict "$v"'
    timeout_ms: 3600000
  - name: release-gpu-bench
    command: |-
      cargo build --release -p xagent-sandbox
    timeout_ms: 3600000
---

# food-contingency-decision: Food-contingency gate runs and decision record

Operations: edit (DECISION.md) and shell (release build, GPU runs and the
verdict binary), architect-owned.

## Outcome

The food-contingency gate is run exactly as PLAN.md fixes it: eight
300,000-tick GPU lives on seeds 5 to 8, learning and frozen arms. The eight
per-run summaries are written to a scratch directory inside this job's durable
directory and are not committed. The verdict binary validates them fail-closed
and writes one bundle, `docs/decisions/food-contingency-results.json`.
`DECISION.md` states the verdict computed by that binary and the behaviour, link
and outcome evidence behind it. A negative result names the failed link
(readout, credit or weight change).

## Scope

- Scratch summaries: `$MAKINA_SHELL_JOB/summaries/` holds
  `learning-seed{5,6,7,8}.json` and `frozen-seed{5,6,7,8}.json`, written only by
  the harness through `--food-kind-summary`. `$TMPDIR` is not used.
- Stdout and stderr of each run, and the verdict output, go to
  `$MAKINA_SHELL_JOB/output/`.
- `docs/decisions/food-contingency-results.json` is written only by
  the verdict binary through `--write-bundle`.
- `docs/decisions/food-contingency.md` is the decision record.
- The host commits the touches of this task. The architect does not run Git.

## Non-goals

- No product code, config, test or PLAN.md change. No threshold change.
- No hand-edited, reconstructed or fabricated summary, no hand-edited or
  fabricated bundle, and no verdict text typed by hand. Every number in
  `DECISION.md` comes from a summary or from the verdict output.
- No per-tick logs in the repository.
- No retry of a run that already wrote its summary. A run that failed before
  writing one may be retried once; record the failure in Setup.
- No port of the verdict to evolution defaults, whatever the result.

## Decisions

- **Runs.** Seeds 5, 6, 7 and 8, one at a time, in that order. Each run uses 10
  agents, 300,000 ticks, fresh birth brains and `--food-kinds`. The 0001
  free-life flags for agent count, tick count, seed and fresh brains use the
  exact names in the landed `crates/xagent-sandbox/src/main.rs`; record them in
  Setup. The food-kind flags are the ones added by food-kind-free-life.
- **Learning arm.** `cargo run --release -p xagent-sandbox -- <0001 flags> --food-kinds --food-kind-summary $MAKINA_SHELL_JOB/summaries/learning-seed<N>.json`.
- **Frozen arm.** The same command for the same seed, plus `--freeze-policy`,
  with the summary `$MAKINA_SHELL_JOB/summaries/frozen-seed<N>.json`.
- **Verdict.** `cargo run --release -p xagent-sandbox --bin food_kind_verdict -- $MAKINA_SHELL_JOB/summaries --write-bundle docs/decisions/food-contingency-results.json`.
  Its `Verdict:` line is the only source of the verdict.
- **No adapter, no task.** If no GPU adapter is present, the task is BLOCKED.
  Write no summary, no bundle and no `DECISION.md`, and report the blocker.
- **Failed summary.** If the verdict rejects a summary, it is an error. Report
  BLOCKED with the error text and write no `DECISION.md`.
- **Run sequencing.** One GPU, so runs are sequential. Check each job state
  before starting the next run. Never start a second copy of a run that is still
  running.

## Steps

**Steps:**

1. Build and adapter check. Run `cargo build --release -p xagent-sandbox`, then
   `cargo run --release -p xagent-sandbox -- --bench --bench-ticks 2000 --bench-agents 2`.
   Record the GPU adapter name from the output. If none is present, stop and
   report BLOCKED.
2. Learning arm. For seeds 5, 6, 7 and 8, run the learning command from Decisions
   with stdout and stderr sent to `$MAKINA_SHELL_JOB/output/learning-seed<N>.log`.
   Record the wall time of each run. Stop and report on any failed run.
3. Frozen arm. For seeds 5, 6, 7 and 8, run the frozen command for the same seed
   and flags, with output to `$MAKINA_SHELL_JOB/output/frozen-seed<N>.log`.
   Verify that all eight summaries exist in `$MAKINA_SHELL_JOB/summaries/` and
   are non-empty.
4. Verdict. Run the verdict command from Decisions. Save its full output (the
   `Verdict:` line and the `GateReport` table) to
   `$MAKINA_SHELL_JOB/output/verdict.txt`. Confirm that `results.json` was written.
5. Write `docs/decisions/food-contingency.md`. Its first line is
   `Verdict: <PASS | FAIL readout | FAIL credit | FAIL weight-change>`, copied
   exactly from the verdict output. Then write these sections in order: Question;
   Setup (commands as run with flag names, build profile, GPU adapter name, wall
   time per run, any failed run; no transient worktree or TMPDIR paths); Nourishing
   kind per seed; Behaviour table per seed and arm (B1, B2, B3, meals of each kind,
   deaths); Links table (readout accuracy and shuffled control per seed, agents
   with positive signed credit per pair, agents with positive signed weight
   difference per pair); Outcome (which gate items passed; for FAIL, the named link
   and its evidence; for weight-change with links passing, state that the weights
   moved the right way but too little); What this does not show. Take every number
   from the summaries or the verdict output.
6. Run the `decision-consistency` check and the `release-gpu-bench` check on the
   written `DECISION.md` and `results.json`. Both must pass.

- **Done when:** check decision-consistency passes on the committed DECISION.md and results.json, so the verdict on the first line equals the verdict the verdict binary computes from the bundle, and a missing, malformed or mismatched summary is an error (INV-08).

## Evidence and risks

- The architect runs the release build, the adapter check and the eight runs on
  the host shell. The host runs the `decision-consistency` and `release-gpu-bench`
  checks after the packet lands.
- Wall time per run is long (300,000 ticks, 10 agents). Do not add runs to make
  the sample larger; the gate fixes seeds 5 to 8.
- A FAIL verdict is a valid, complete result. Record it as computed. Do not change
  thresholds or rerun a seed to reach a different verdict.
