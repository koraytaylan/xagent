---
name: implement-plan
description: Implement a plan's task list by running its tasks in parallel with dependency-aware scheduling, a pre-execution dependency-verification pass, per-task developer→reviewer loops, and serialized 3-way merges into one integration branch. Use when the user says "implement plan NNNN", "/implement-plan NNNN", "run plan 0012", or asks to execute a task list under docs/plans/* (or docs/superpowers/plans/*). This skill runs the committed multi-agent engine at .claude/workflows/implement-plan.js via the Workflow tool, so it works in cloud agents that only have the repo (not your personal ~/.claude). Requires the Workflow tool / sub-agent spawning.
---

# implement-plan

This skill executes a plan's task list with the bundled deterministic workflow engine. The engine
is a JavaScript workflow — **do not reimplement its logic in prose**; its correctness comes from
deterministic DAG scheduling, a serialized merge queue, and the cycle-safe dependency-Verify phase.
Your job is only to launch it with the right arguments.

## How to run

Call the **Workflow** tool. Resolve the engine by name first (it lives in this repo's
`.claude/workflows/`), falling back to the explicit path:

- Preferred: `Workflow({ name: "implement-plan", args: { plan: "<selector>", ... } })`
- Fallback if the name does not resolve: `Workflow({ scriptPath: ".claude/workflows/implement-plan.js", args: { plan: "<selector>", ... } })`

`<selector>` is the plan number or slug, e.g. `"0012"` or `"docs/plans/0012-Some-Name"`. Take it from
the user's message (e.g. `/implement-plan 0012` → `plan: "0012"`).

## Arguments (pass inside `args`)

- `plan` (required) — the plan selector string.
- `dryRun` — read-only: run discovery + the dependency-Verify pass, print the parallel execution plan
  (waves, what would run vs. skip, and any auto-added safety edges), then STOP. **Recommend running
  this first** on an unfamiliar plan.
- `maxParallel` (default 4) — cap on concurrent task worktrees (each is a full checkout — bounds disk).
- `sequential` (default false) — force a strict one-task-at-a-time linear chain (overrides the graph).
- `integrationGate` (default true) — re-run the project's gates on the merged plan-branch tip before
  each task advances it. Leave on unless tasks are truly disjoint.
- `into` — base/integration branch to merge back into (defaults to the branch the run started on).
- `devModel` / `reviewModel` / `integrateModel` — model overrides for the developer / reviewer / merge
  steps.
- `maxReviewIters` (default 3) — review rounds before a task is left for inspection.

The run resumes by default: if a prior run left the plan's integration branch, already-landed tasks
(git ground truth) are skipped.

## Cloud-runner note (important)

The engine **commits but never pushes** — it squash-merges the plan branch into the *local* base
branch and stops. On an ephemeral cloud runner the filesystem is discarded at session end, so after
the workflow returns you MUST persist the result yourself: push the base branch (or the
`implement-plan/<plan>` integration branch) to the remote, and/or open a PR. If you do not push, the
work is lost when the sandbox is torn down. (Cloud agents have a wired-in git credential, so
`git push` works.)
