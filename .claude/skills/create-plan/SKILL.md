---
name: create-plan
description: Author the next implementation plan(s) under docs/plans/ in the exact house style the implement-plan workflow consumes. Use when the user says "create a plan", "/create-plan", "author the next plan", "plan the next work", "write a plan for <topic>", "generate the next plan(s)", or asks to scaffold a new numbered plan folder (SCOPE.md + ARCHITECTURE.md + TASKS.md + STATUS.md) under docs/plans/*. With no topic it auto-derives the next plan(s) from the deferred / gated / follow-on / debt work the STATUS docs record. This is the AUTHORING counterpart to implement-plan: create-plan WRITES a plan, implement-plan EXECUTES one — so this skill is NOT triggered by "implement plan NNNN", "run plan 0012", or "execute a task list" (that is implement-plan's domain). This skill runs the committed multi-agent engine at .claude/workflows/create-plan.js via the Workflow tool, so it works in cloud agents that only have the repo (not your personal ~/.claude). Requires the Workflow tool / sub-agent spawning.
---

# create-plan

This skill authors the project's next implementation plan(s) with the bundled deterministic workflow
engine — the counterpart to `implement-plan`: `create-plan` writes the plan, `implement-plan` executes
it. The engine is a JavaScript workflow; **do not reimplement its logic in prose**. Its correctness
comes from reading the live `docs/plans/README.md` authoring guide, deriving the next `NNNN` from git
ground truth, authoring the SCOPE/ARCHITECTURE/TASKS/STATUS triad from one shared research-backed
blueprint (so the workstream ids stay consistent and `TASKS.md` parses), and then validating the result
by running `implement-plan` itself in `dryRun`. Your job is only to launch it with the right arguments.

## How to run

Call the **Workflow** tool. Resolve the engine by name first (it lives in this repo's
`.claude/workflows/`), falling back to the explicit path:

- Preferred: `Workflow({ name: "create-plan", args: { brief: "<topic>", ... } })`
- Fallback if the name does not resolve: `Workflow({ scriptPath: ".claude/workflows/create-plan.js", args: { ... } })`

Map the user's request to `args`:

- `/create-plan` or "author the next plan" (no topic) → `args: {}` — the engine auto-derives the next
  plan from recorded deferred/gated/follow-on work.
- "create a plan for the credit-path learning fix" → `args: { brief: "the credit-path learning fix" }`.
- "draft the next 2 plans" → `args: { count: 2 }` (or `count: "auto"` to author every distinct
  high-priority candidate it finds, capped at 5).

**Recommend running `dryRun` first** on an open-ended request, so the user can confirm the chosen
topic(s), number(s), and workstream split before any files are written:
`Workflow({ name: "create-plan", args: { brief: "...", dryRun: true } })`.

## Arguments (pass inside `args`)

- `brief` — free text describing what the next plan should address. Omit to auto-derive from the
  deferred / gated / follow-on / debt work the STATUS docs, decision docs, and CONTRIBUTING record. (A
  bare string arg, e.g. `args: "the credit-path fix"`, is treated as the brief.)
- `count` (default 1) — how many plans to author. `"auto"` (or `"all"`) authors every distinct
  high-priority candidate found (capped at 5). Lower-priority candidates that are skipped are logged.
- `dryRun` — read-only: run Survey + Select, print the plans that WOULD be authored (numbers, slugs,
  rationale, provisional workstreams), then STOP. Writes nothing.
- `plansDir` (default `docs/plans`) — the plans root, if it lives elsewhere.
- `baseNumber` — force the first plan's `NNNN` (default: one past the highest existing plan).
- `verify` (default true) — run the adversarial critic + bounded fixer (checks the README new-plan
  checklist, triad consistency, anchor reality, junior-executability, and the parse contract).
- `validate` (default true) — run `implement-plan` in `dryRun` on each authored plan to prove its
  `TASKS.md` parses into an acyclic dependency DAG, folding any surfaced safety edges back into the file.
- `commit` (default false) — git-add + commit the authored files and the roll-up row. Off by default;
  files are left uncommitted for human review.
- `maxVerifyIters` (default 2) — critic→fix rounds before a plan is left as-authored for inspection.
- `authorModel` / `criticModel` / `choreModel` — model overrides for the authoring / critic / mechanical
  steps (authoring and critique inherit the session model by default — best for quality).

## What it produces

Per plan, a `docs/plans/NNNN-Title-Case-Kebab/` folder with `SCOPE.md`, `ARCHITECTURE.md`, `TASKS.md`,
and a `📋 Planned` `STATUS.md`, plus a `📋 Planned` row added to `docs/plans/STATUS.md`. The result is
exactly what `implement-plan` consumes — hand it straight to `implement-plan NNNN` (run its `dryRun`
first) to execute.

## After it runs

Read the engine's summary and tell the user what was authored (numbers, slugs, task counts) and whether
verify/validate passed. The plan files are the deliverable — point the user at them to review and refine
before implementing. If a plan's `verify` reported unresolved blockers or `validate` failed to parse,
say so plainly and surface the specific files to fix.

## Cloud-runner note (important)

The engine writes to the **local** working tree only: by default it leaves the authored files
UNCOMMITTED for review, and even with `commit:true` it commits to the *local* repo and never pushes. On
an ephemeral cloud runner the filesystem is discarded at session end, so persistence is YOUR job after
the workflow returns: commit the new files (or pass `commit:true`) and then `git push` the branch
yourself. If you do not push, the authored plan is lost when the sandbox is torn down. (Cloud agents
have a wired-in git credential, so `git push` works.)
