---
id: supervisor-decision
title: "Supervisor evidence runs and decision record"
workstream: "0800"
kind: task
role: architect
depends_on: [evolution-cli, supervisor-verdict]
gated: false
touches: [docs/measurements/continuous-supervisor/DECISION.md, docs/measurements/continuous-supervisor/results.json]
status: todo
merged_as: ""
checks:
  - name: decision-consistency
    command: |-
      sh -c 'v=$(grep -m1 "^Verdict: " docs/measurements/continuous-supervisor/DECISION.md | cut -d" " -f2-) && test -n "$v" && cargo run -p xagent-sandbox --bin supervisor_verdict -- docs/measurements/continuous-supervisor/results.json --expect-verdict "$v"'
    timeout_ms: 3600000
  - name: release-build
    command: cargo build --release -p xagent-sandbox
    timeout_ms: 3600000
---

# Supervisor evidence runs and decision record

## Outcome

The PLAN.md "Decision rule" is run exactly as fixed: 20 in-run jobs and 12 free
lives on seeds 5 to 8. The verdict binary computes the verdict from the 32
validated summaries and writes `docs/measurements/continuous-supervisor/results.json`.
`DECISION.md` records that verdict. Run databases, summaries and configs stay in
ignored `target/continuous-supervisor/`.

## Scope

- Touches only `docs/measurements/continuous-supervisor/DECISION.md`, and
  `results.json` as written by the verdict binary.
- `mkdir -p target/continuous-supervisor/{db,summaries,config}` first.
- Every run is `cargo run --release -p xagent-sandbox -- <flags>`. Paths are
  checkout-relative. Stdout and stderr go to `$MAKINA_SHELL_JOB/output/`.

## Non-goals

- No product code, config default, test, plan or threshold change.
- No hand-edited summary, bundle or verdict. No run databases or per-tick logs committed.
- No adoption of weight copy, and no default flip even on ADOPT.

## Decisions

- **Configs.** `--dump-config` writes `config/default.json`. `config/base.json`:
  replace the line `"tick_budget": 1000000,` with `"tick_budget": 40000,`
  (sed on the exact line; grep confirms exactly one line changed). `config/base-kinds.json`:
  from base, set `"food_kinds_enabled"` and `"second_odour_sense_enabled"` to `true`
  (grep verifies each). All supervisor settings stay default.
- **In-run jobs**, for each seed S in 5, 6, 7, 8, with
  `--config <cfg> --seed S --no-render --agent-tick-budget 32000000 --db target/continuous-supervisor/db/<name>-sS.db --run-summary target/continuous-supervisor/summaries/<name>-seedS.json`:
  - `generations`: base, `--evolution-mode generations`
  - `supervisor`: base, `--evolution-mode supervisor`
  - `weight-copy`: base, `--evolution-mode supervisor --supervisor-inheritance weight-copy`
  - `generations-kinds`: base-kinds, `--evolution-mode generations`
  - `supervisor-kinds`: base-kinds, `--evolution-mode supervisor`
- **Free lives** (after their databases exist), base.json, `--free-life --free-life-agents 10 --free-life-ticks 300000 --seed S --free-life-out target/continuous-supervisor/summaries/free-life-<family>-seedS.json`.
  `generations` adds `--free-life-champion-db target/continuous-supervisor/db/generations-sS.db`.
  `supervisor` and `weight-copy` add `--free-life-archive-db target/continuous-supervisor/db/<family>-sS.db` (rank 0, the default).
- **Batching.** At most four GPU jobs at once. Check `free -h` and every job's state
  before each batch; never start a job still running. A run that failed before writing
  its output may be retried once, with the cause recorded. A run that wrote output is never re-run.
- **Verdict.** `cargo run --release -p xagent-sandbox --bin supervisor_verdict -- target/continuous-supervisor/summaries --write-bundle docs/measurements/continuous-supervisor/results.json`.
  Its `Verdict:` line is the only source of the verdict. Save the full output to `$MAKINA_SHELL_JOB/output/verdict.txt`.
- **BLOCKED.** No GPU adapter: write nothing and report BLOCKED. A verdict input error:
  report BLOCKED with the error text and write no DECISION.md. A run that cannot complete: BLOCKED with partial progress; never shorten budgets or drop seeds.
- **DECISION.md** first line `Verdict: <ADOPT|OPTION|REJECT>`, copied from the verdict output. Sections in order:
  Question; Rule (PLAN.md "Decision rule", quoted as fixed before the runs); Setup (commands with flag names,
  build profile, GPU adapter from the bench line, wall time per run, any retried run; no worktree or TMPDIR paths);
  Per-seed table (F and D for supervisor, generations, weight-copy); Learning table (L1-L3, K1-K3 for the three arms where defined);
  Outcome (which rule clauses held; weight copy recorded, never adopted); Consequence (ADOPT: removing generation mode is a later plan;
  OPTION/REJECT: the flag stays off; no default changes here); What this does not show. Every number comes from the verdict output or the summaries.

## Steps

**Steps:**

1. Run `cargo build --release -p xagent-sandbox`, then `cargo run --release -p xagent-sandbox -- --bench --bench-ticks 2000 --bench-agents 10`. Record the GPU adapter name from the bench line; no adapter means BLOCKED.
2. Run `mkdir -p` for the three work subdirectories and write the three configs per Decisions, grep-verifying each edit.
3. Run the 20 in-run jobs for seeds 5 to 8 in batches of at most four, per Decisions. Record each wall time.
4. Once the matching databases exist, run the 12 free lives per Decisions.
5. Run the verdict and bundle command; save its full output to `$MAKINA_SHELL_JOB/output/verdict.txt`.
6. Write `docs/measurements/continuous-supervisor/DECISION.md` per Decisions, then run the `decision-consistency` and `release-build` checks.

## Done when

- **Done when:** the `decision-consistency` check passes on the committed DECISION.md and results.json (its first-line verdict equals the binary's recomputed verdict over the bundle), and `release-build` passes, proving INV-14.

## Evidence and risks

- The architect runs all builds and runs on the host shell. The host runs both checks after the packet lands.
- 20 runs of 3.2M ticks plus 12 free lives are long. Do not add or drop runs.
- REJECT and OPTION are complete, valid results; record them as computed.
- The architect does not run Git. Job logs are not cited as durable evidence in DECISION.md.
