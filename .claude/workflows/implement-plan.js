export const meta = {
  name: 'implement-plan',
  description: 'Implement a plan\'s task list, running independent tasks in parallel. The discovery step reads the plan\'s TASKS.md when one exists and uses its authored "Depends on" edges as the dependency graph; if no TASKS.md exists it infers the graph. Before any execution a VERIFICATION step audits that graph: it re-derives each task\'s file footprint and what code each task needs from others, then AUTO-ADDS a serializing dependency edge between any two would-be-parallel tasks that would write the same files or where one needs another\'s code — so no two concurrent tasks can corrupt each other or fail to merge (authored edges are never removed, only added to; every added edge is logged). It then resumes by default: if a prior run left the plan\'s integration branch, tasks already landed on it (git ground truth) are skipped and only the unfinished ones run. Per run: create (or reuse) one integration branch for the plan off the base branch; the scheduler runs every task whose dependencies have landed concurrently (up to a configurable parallelism cap), each in its own worktree+branch off the plan branch (developer implements + self-gates, reviewer independently runs the project\'s gates and reviews, looping until approved or capped, escalating the developer to the stronger model after a rejected round). On approval the task branch is merged into the plan branch through a serialized 3-way merge (conflicts are surfaced, never clobbered), and its dependents build on the result. If no TASKS.md existed, the verified graph is written to one. At the end, if every non-gated task landed, the plan branch is squash-merged into the base branch as a single commit whose message lists what was done, and the plan\'s STATUS.md + the roll-up board are updated to reflect the outcome. (GATED) tasks are reported but never auto-run. Failed/blocked/conflicted tasks leave their worktree for inspection and block the final squash.',
  phases: [
    { title: 'Discover', detail: 'locate the plan, read/parse TASKS.md (or infer), extract each task\'s file footprint, read STATUS, detect resume state + git roots' },
    { title: 'Verify', detail: 'audit the dependency graph: detect would-be-parallel tasks that share files or have missing prerequisites and auto-add serializing edges so concurrent tasks cannot corrupt each other' },
    { title: 'Setup', detail: 'create the run\'s plan integration branch off the base (or reuse it to resume); write the verified TASKS.md if it was missing' },
    { title: 'Implement', detail: 'schedule tasks by dependency: ready tasks run in parallel (worktree → implement+gate → review, loop), serialized 3-way merge into the plan branch on approval; skip already-landed and gated tasks' },
    { title: 'Integrate', detail: 'squash-merge the plan branch into the base as one commit (only if all non-gated tasks landed)' },
    { title: 'Status', detail: 'update the plan\'s STATUS.md + the roll-up board to reflect the run outcome (best-effort, uncommitted)' },
  ],
}

// ── Inputs ──────────────────────────────────────────────────────────────────
// `args` is a plan selector string ("0012") or an object:
//   { plan: "0012", into: "develop", maxReviewIters: 3, devModel: "haiku",
//     reviewModel: "sonnet", integrateModel: "sonnet", maxParallel: 4,
//     sequential: false, integrationGate: true }
// `into`            overrides the base/integration branch (defaults to the branch the run started on).
// `maxParallel`     caps how many tasks hold a worktree + run concurrently (each worktree is a full
//                   checkout, so this bounds disk + machine load). Dependencies further constrain it.
//                   NOTE: maxParallel:1 only BOUNDS concurrency — it still honors the discovered DAG and
//                   reviews each task against its own base. For the legacy total-order behavior (each task
//                   built and reviewed on top of the previous task's merged result) use `sequential:true`.
// `sequential`      forces the legacy one-task-at-a-time behavior by chaining every task to the previous.
//                   It overrides the TASKS.md / inferred dependency graph with a linear chain.
// `dryRun`          read-only: run discovery AND verification (both read-only), print the parallel
//                   execution plan (waves, what would run vs skip, and any auto-added safety edges), and
//                   STOP. Creates no branch/worktree, edits no code, writes no TASKS.md/STATUS.

// `integrationGate` (default true) re-runs the project's gates on the MERGED plan-branch tip before each
//                   task advances it, so a clean-but-broken combination of independently-developed tasks is
//                   caught before it can reach the base branch. Set false only when tasks are truly disjoint.
//
// Dependency verification (runs before any branch/worktree/code work, on BOTH authored and inferred
// graphs — this is what stops conflicting tasks from running in parallel and corrupting each other):
//   • The scheduler runs every dependency-free task CONCURRENTLY. If the graph leaves two tasks that
//     write the same file (or where one needs the other's code) WITHOUT an edge between them, they run
//     in parallel and either merge-conflict or — worse — text-merge cleanly into a semantically broken
//     combination. The Verify phase finds those gaps and AUTO-ADDS the missing serializing edge.
//   • Two signals are used: (a) a deterministic file-collision check over each task's extracted
//     footprint — any unordered pair of would-be-concurrent tasks sharing a file with no path between
//     them is serialized in document order; (b) a semantic audit agent that re-derives, from the plan
//     docs, prerequisites file-overlap can't see (task B needs a type/function task A introduces).
//   • Edges are only ADDED, never removed: authored intent is preserved, and the run will simply not
//     schedule two file-colliding tasks at the same time. Every added edge is logged; suspected
//     wrong/unjustified existing edges are reported as warnings (not auto-applied). Skipped under
//     `sequential:true` (a linear chain is already collision-free).
//
// TASKS.md / STATUS.md behavior (the project's plan convention):
//   • If the plan folder has a TASKS.md, its authored "### {id} — {title}" tasks and "**Depends on:**"
//     edges seed the dependency graph; verification may ADD collision/prerequisite edges on top (logged).
//     The authored graph lists DIRECT prerequisites and is the lever for real parallelism; verification
//     only narrows it where two tasks would otherwise collide. When run against an authored file, added
//     edges apply to THAT run only — the log names them so they can be folded into the file permanently.
//   • If no TASKS.md exists (and the plan is a directory-style plan), the VERIFIED graph (inferred edges
//     plus any added collision/prerequisite edges) is WRITTEN to <planDir>/TASKS.md (untracked) so it
//     becomes a reviewable artifact.
//   • Tasks whose title ends with "(GATED)" are conditional Phase-2 follow-ups; they are reported but
//     NOT auto-run (their start is a human decision).
//   • Resume is UNCONDITIONAL — it always happens, with no opt-out: ground truth is git, not the doc —
//     a task is "already landed" iff its task branch is contained in the plan branch left by a prior
//     partial run. Those are skipped; setup reuses the existing plan branch instead of wiping it. To
//     force a clean restart, delete the plan branch by hand first (`git branch -D implement-plan/<plan>`),
//     since the workflow only resumes when that branch exists.
//   • At the end the plan's STATUS.md (per-task) and the roll-up board (one row per plan) are updated to
//     reflect the run. These edits are left UNCOMMITTED in the working tree for the user to review.
// `args` may arrive as an object, a bare selector string ("0012"), OR — depending on how the run is
// launched — a JSON-STRING of the options object. Normalize all three so options like dryRun survive.
let _args = args
if (typeof _args === 'string') {
  const s = _args.trim()
  if (s.startsWith('{') || s.startsWith('[')) { try { _args = JSON.parse(s) } catch { /* keep as string */ } }
}
const cfg = (_args && typeof _args === 'object') ? _args : { plan: _args }
const PLAN = String(cfg.plan || '')
const DEV_MODEL = cfg.devModel || 'haiku'
const REVIEW_MODEL = cfg.reviewModel || 'sonnet'
const MAX_ITERS = cfg.maxReviewIters || 3
const MAX_PARALLEL = Math.max(1, cfg.maxParallel || 4)
const FORCE_SEQUENTIAL = !!cfg.sequential
const DRY_RUN = !!cfg.dryRun                // read-only: discover + print the parallel plan, then stop
const INTEGRATION_GATE = cfg.integrationGate !== false
const INTEGRATE_MODEL = cfg.integrateModel || (INTEGRATION_GATE ? REVIEW_MODEL : 'haiku')

if (!PLAN) { log('No plan selector given. Pass e.g. args: "0012".'); return { error: 'no plan selector' } }
// Hard safety net: a valid plan selector is a number/slug/path ("0012", "docs/plans/0012-Name") — only
// word chars, '-', '_', '/', '.'. Anything with whitespace or JSON punctuation means the options object
// failed to parse and leaked into the selector. ABORT before any branch/worktree work rather than run the
// whole plan against a garbage id (this is exactly what an unparsed `{"plan":...}` arg would do).
if (/[^\w./-]/.test(PLAN)) {
  log(`Invalid plan selector ${JSON.stringify(PLAN)} (contains whitespace or JSON punctuation). The options object likely did not parse; aborting before doing any work. Pass args as {plan:"0012", ...} or a bare "0012".`)
  return { error: 'invalid plan selector', selector: PLAN }
}

const PLAN_BRANCH = `implement-plan/${PLAN}`            // the run's integration branch
const taskBranchOf = (id) => `implement-plan/${PLAN}--${id}`   // "--" so it never nests under PLAN_BRANCH

// ── Schemas ─────────────────────────────────────────────────────────────────
const DISCOVER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['repoRoot', 'startSha', 'startRef', 'planDir', 'tasksFromFile', 'planBranchExists', 'tasks'],
  properties: {
    repoRoot: { type: 'string', description: 'git rev-parse --show-toplevel' },
    startSha: { type: 'string', description: 'git rev-parse HEAD' },
    startRef: { type: 'string', description: 'git rev-parse --abbrev-ref HEAD (the base branch to integrate back into)' },
    planDir: { type: 'string', description: 'path to the plan directory, relative to repoRoot' },
    tasksPath: { type: 'string', description: 'path to the plan TASKS.md relative to repoRoot (where it exists or would be written); "" for a bare single-file plan with no dedicated folder' },
    statusPath: { type: 'string', description: 'path to the plan per-plan STATUS.md relative to repoRoot; "" if the plan has no dedicated folder' },
    rollupPath: { type: 'string', description: 'path to the plans-root roll-up STATUS.md (e.g. docs/plans/STATUS.md) relative to repoRoot; "" if none exists' },
    tasksFromFile: { type: 'boolean', description: 'true if tasks were PARSED from an existing TASKS.md; false if inferred from prose' },
    planBranchExists: { type: 'boolean', description: `true if a branch named ${PLAN_BRANCH} already exists` },
    tasks: {
      type: 'array',
      description: 'tasks in plan-document order',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title'],
        properties: {
          id: { type: 'string', description: 'the task id / heading (kebab-case)' },
          title: { type: 'string' },
          dependsOn: { type: 'array', items: { type: 'string' }, description: 'ids of tasks this one truly needs. From TASKS.md "**Depends on:**" verbatim, or inferred. Empty = independent.' },
          touches: { type: 'array', items: { type: 'string' }, description: 'repo-relative file paths (or directory paths) this task is expected to CREATE or MODIFY, read from the task steps / acceptance criteria / plan prose. Used to detect collisions between would-be-parallel tasks. Bias toward listing MORE: an over-broad footprint only costs a little parallelism, a missing one risks two tasks corrupting the same file. [] only if genuinely undeterminable.' },
          doneWhen: { type: 'string', description: 'the task\'s acceptance criteria text' },
          gated: { type: 'boolean', description: 'true if the task title ends with "(GATED)" — a conditional follow-up that must NOT be auto-run' },
          landed: { type: 'boolean', description: `true iff a prior run already landed this task: its branch ${taskBranchOf('<id>')} exists AND is an ancestor of ${PLAN_BRANCH}` },
        },
      },
    },
  },
}

const OK_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ok'],
  properties: { ok: { type: 'boolean' }, error: { type: 'string' } },
}

// The merge agent commits + folds + gates but NEVER advances the plan branch — advancement is
// script-controlled (see ADVANCE_SCHEMA) so no agent step-ordering can leave the plan branch
// advanced with unvalidated code.
const MERGE_TASK_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ok', 'committed'],
  properties: { ok: { type: 'boolean' }, committed: { type: 'boolean' }, conflict: { type: 'boolean' }, gatesFailed: { type: 'boolean' }, error: { type: 'string' } },
}

const ADVANCE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ok', 'advanced'],
  properties: { ok: { type: 'boolean' }, advanced: { type: 'boolean' }, error: { type: 'string' } },
}

const CONTAIN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['contained'],
  properties: { contained: { type: 'boolean' }, error: { type: 'string' } },
}

const INTEGRATE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['ok', 'committed'],
  properties: { ok: { type: 'boolean' }, committed: { type: 'boolean' }, sha: { type: 'string' }, error: { type: 'string' } },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['approved', 'gatesPass', 'findings', 'summary'],
  properties: {
    approved: { type: 'boolean', description: 'true ONLY if all gates pass and there are no blocker findings' },
    gatesPass: { type: 'boolean', description: 'true if the project build/test/lint/format gates all pass' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'note'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'nit'] },
          file: { type: 'string' },
          note: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

// The dependency-verification agent: audits the graph for edges that are MISSING and would damage a
// parallel run. It never removes edges; the script applies only `addEdges` (and only those that stay
// acyclic), and logs `warnings` without acting on them.
const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['addEdges', 'warnings'],
  properties: {
    addEdges: {
      type: 'array',
      description: 'edges that MUST exist but are absent: each means task "to" needs task "from" finished first — either because they write a common file (cannot safely run in parallel) or because "to" needs code/types/migrations "from" introduces to compile or pass its gates. Do NOT list edges already implied (directly or transitively) by the current graph.',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['from', 'to', 'reason'],
        properties: {
          from: { type: 'string', description: 'task id that must run BEFORE' },
          to: { type: 'string', description: 'task id that must run AFTER (gains the dependency on "from")' },
          reason: { type: 'string', description: 'the shared file(s), or the specific code/type/artifact "to" needs from "from"' },
        },
      },
    },
    warnings: {
      type: 'array',
      items: { type: 'string' },
      description: 'reported-only risks: existing dependsOn edges that look WRONG or unjustified, tasks whose file footprint could not be determined, or any other parallel-execution hazard. NOT auto-applied.',
    },
  },
}

// ── Prompt builders (function declarations hoist) ─────────────────────────────
function setupPrompt(repoRoot, planBranch, startSha) {
  return `Prepare the run's integration branch in the git repo at ${repoRoot}. Clean up any artifacts from a previous run of this plan first, then create a fresh branch.\n`
    + `1. git -C "${repoRoot}" worktree prune\n`
    + `2. Remove stale worktrees from a prior run: for each path under "${repoRoot}/.worktrees/implement-plan/${PLAN}--"*, run \`git -C "${repoRoot}" worktree remove --force <path>\` (ignore errors).\n`
    + `3. Delete stale branches from a prior run: \`git -C "${repoRoot}" branch --list 'implement-plan/${PLAN}*' | sed 's/^[* ]*//' | xargs -r -n1 git -C "${repoRoot}" branch -D\` (ignore errors).\n`
    + `4. Create the integration branch at the base commit WITHOUT switching the working tree: \`git -C "${repoRoot}" branch "${planBranch}" "${startSha}"\`\n`
    + `Confirm with \`git -C "${repoRoot}" rev-parse "${planBranch}"\`. Return ok=true on success, ok=false + error otherwise. Do NOT switch branches or modify tracked files.`
}

// Resume setup: a prior partial run left the plan branch with landed tasks on it. Do NOT delete or reset
// it — it IS the work to build on. Just prune stale per-task worktrees (keep every branch) so unfinished
// tasks can recreate their worktrees off the current plan-branch tip.
function setupResumePrompt(repoRoot, planBranch) {
  return `RESUME a partially-completed run on the existing integration branch "${planBranch}" in ${repoRoot}. It holds tasks that already landed — do NOT delete, reset, or fast-forward it, and do NOT delete any branch.\n`
    + `1. git -C "${repoRoot}" worktree prune\n`
    + `2. Remove stale per-task worktrees from the prior run (KEEP their branches): for each path under "${repoRoot}/.worktrees/implement-plan/${PLAN}--"*, run \`git -C "${repoRoot}" worktree remove --force <path>\` (ignore errors).\n`
    + `3. Confirm the integration branch still exists: \`git -C "${repoRoot}" rev-parse --verify "${planBranch}"\`. If it does NOT exist, return ok=false + error="plan branch missing, cannot resume".\n`
    + `Return ok=true if "${planBranch}" exists and stale worktrees are pruned. Do NOT switch branches, modify tracked files, or delete any branch.`
}

function generateTasksPrompt(repoRoot, tasksPath, planLabel, tasks) {
  const list = tasks.map(t => ({ id: t.id, title: t.title, dependsOn: t.deps || [], doneWhen: t.doneWhen || '' }))
  return `No TASKS.md exists for this plan, so the dependency graph was inferred from the plan prose. Persist it as a reviewable artifact at ${repoRoot}/${tasksPath}, in the project's TASKS.md house style, so future runs use the authored graph instead of re-inferring it.\n`
    + `Write the file with this structure:\n`
    + `  • A top H1 title and a one-paragraph summary of the plan.\n`
    + `  • A "**Conventions**" block noting: each task has a stable kebab-case id; "**Depends on**" lists direct prerequisites only ("—" means none); "**Done when**" is the verifiable acceptance criterion.\n`
    + `  • One or more "## NNNN — Workstream" phase headers grouping the tasks (a single "## 0001 — ${planLabel}" workstream is fine if no natural grouping exists), each containing the tasks as:\n`
    + `        ### {id} — {Title}\n`
    + `        <1-2 lines of context from the plan prose if available>\n`
    + `        - **Depends on:** {comma-separated dependency ids, or — if none}\n`
    + `        - **Done when:** {the acceptance criterion}\n`
    + `  • A note near the top that this file was AUTO-GENERATED from inferred dependencies by implement-plan and should be reviewed/refined by a human.\n`
    + `The authoritative task data (use these ids and edges EXACTLY):\n${JSON.stringify(list, null, 2)}\n`
    + `Create the file (and any missing parent directory). Do NOT commit, do NOT modify code or any other file. Return ok=true on success, ok=false + error otherwise.`
}

function verifyPrompt(repoRoot, planDir, tasks) {
  const list = tasks.map(t => ({ id: t.id, title: t.title, dependsOn: t.deps, touches: t.touches, doneWhen: t.doneWhen || '' }))
  return `You are the dependency-VERIFICATION step of an implement-plan run, auditing the task graph BEFORE anything is built. The scheduler runs every task whose dependencies have landed CONCURRENTLY, each in its own worktree, then merges them one at a time. Two tasks with NO dependency path between them WILL run in parallel — and if they modify the same file, or one needs code/types/migrations the other introduces, the parallel run corrupts the result or fails to merge. Your job: find the dependency edges that are MISSING and would damage execution, so they can be added before any work starts. This is READ-ONLY — make NO edits, NO commits, NO branch changes.\n`
    + `\n`
    + `Read the plan's task specs and design/overview docs from ${repoRoot}/${planDir}. Do NOT just trust the declared graph below — you are auditing it. For EACH task independently re-derive: (a) the concrete files it will create or modify, and (b) which other tasks' output it needs in order to compile or pass its own acceptance gates.\n`
    + `\n`
    + `Return addEdges — every edge that MUST exist but is ABSENT from the current graph (directly or transitively). "from" must run before "to"; "to" gains the dependency. Add an edge when EITHER:\n`
    + `  • two tasks touch a COMMON file (they cannot safely run in parallel — serialize them; make the later-defined task depend on the earlier-defined one), OR\n`
    + `  • one task NEEDS the other's code/types/migrations/artifacts to compile or to satisfy its "Done when". \n`
    + `Be thorough and err toward adding an edge when two tasks plausibly collide — a missing edge causes the exact corruption this step exists to prevent; a spurious one only costs a little parallelism. Do NOT list an edge that the current dependsOn already implies.\n`
    + `\n`
    + `Return warnings (reported only, NOT applied) for: existing dependsOn edges that look WRONG or unjustified (do not assume a written edge is correct), any task whose file footprint you could not determine, and any other parallel-execution hazard.\n`
    + `\n`
    + `The list below is ONLY the tasks that will run THIS session. A dependsOn id that is NOT in this list is an already-completed prerequisite (landed in a prior run) — its work is already present, so do NOT propose any edge whose "from" or "to" is an id not in this list. Both endpoints of every addEdges entry MUST be ids from this list.\n`
    + `Tasks to audit (id, declared dependsOn, declared touches, acceptance):\n${JSON.stringify(list, null, 2)}\n`
    + `Use the task ids EXACTLY as given; never invent ids. Return the structured result.`
}

function worktreeCreatePrompt(repoRoot, planBranch, wtPath, taskBranch) {
  return `Create an isolated git worktree for one plan task, branched off the run's integration branch. In ${repoRoot}:\n`
    + `1. Remove any stale worktree/branch for this task (ignore errors):\n`
    + `   git -C "${repoRoot}" worktree remove --force "${wtPath}" 2>/dev/null; git -C "${repoRoot}" branch -D "${taskBranch}" 2>/dev/null; true\n`
    + `2. Create the worktree on a new branch off the plan branch's CURRENT tip:\n`
    + `   git -C "${repoRoot}" worktree add -b "${taskBranch}" "${wtPath}" "${planBranch}"\n`
    + `Confirm with \`git -C "${wtPath}" status\`. Return ok=true on success, ok=false + error otherwise. Do not modify any tracked files.`
}

function developerPrompt(repoRoot, planDir, task, wtPath, findings) {
  let p = `Implement task "${task.id}" — ${task.title}.\n\n`
  p += `YOUR WORKING ROOT IS THE WORKTREE: ${wtPath}\n`
  p += `Make EVERY edit to files under ${wtPath} and run EVERY command there (cd into it). Never edit anything outside the worktree.\n`
  p += `The main checkout is at ${repoRoot}; if your toolchain has a shareable build/dependency cache, point it at the main checkout's to avoid a cold build.\n`
  p += `Read the task's full steps + the plan's design/overview docs from ${repoRoot}/${planDir} (the plan docs are NOT inside the worktree).\n`
  p += `Determine the project's gates (build/test/lint/format) from the task's acceptance criteria and the repo's conventions, and get them all green. Do NOT commit.`
  if (task.doneWhen) p += `\n\nAcceptance ("Done when"): ${task.doneWhen}`
  if (findings && findings.length) {
    p += `\n\nA reviewer REJECTED the previous attempt. Address each finding, then re-run the gates:\n`
    p += findings.map(f => `- [${f.severity}] ${f.file ? f.file + ': ' : ''}${f.note}`).join('\n')
  }
  return p
}

function reviewerPrompt(repoRoot, planDir, task, wtPath, implReport) {
  const report = typeof implReport === 'string' ? implReport : JSON.stringify(implReport || {})
  return `Review the changes in the worktree for task "${task.id}" — ${task.title}.\n\n`
    + `WORKTREE: ${wtPath}\n`
    + `Inspect there: \`cd "${wtPath}"\`, run \`git status\` and \`git diff\`, then determine and run the project's gates (build/test/lint/format) from the task's acceptance criteria and the repo's conventions.\n`
    + `Read the task spec + design docs from ${repoRoot}/${planDir}. Focus on this task's files; ignore unrelated changes.\n\n`
    + `The implementer reported:\n${report.slice(0, 4000)}`
}

function mergeTaskPrompt(repoRoot, wtPath, taskBranch, planBranch, task, gate) {
  let p = `Task "${task.id}" passed review. Commit it and fold in any work other parallel tasks landed in the meantime${gate ? ', then validate the merged result' : ''}. Do NOT advance "${planBranch}" (no \`git branch -f\`) and do NOT remove the worktree — those are separate steps the orchestrator performs. This step is serialized across tasks, so "${planBranch}" is stable while you run.\n`
  p += `1. Commit the work on its task branch:\n`
  p += `   cd "${wtPath}" && git add -A && git commit -m "task(${task.id}): ${task.title}"\n`
  p += `   (If there is nothing to commit, that is an error — return ok=false, committed=false.)\n`
  p += `2. Fold the latest plan branch into this task branch (other tasks may have landed since this worktree was created):\n`
  p += `   cd "${wtPath}" && git merge --no-edit "${planBranch}"\n`
  p += `   - If git reports a MERGE CONFLICT: run \`git merge --abort\`, leave the worktree in place, and return ok=false, committed=true, conflict=true, with the conflicting file paths in error. Do NOT attempt to resolve it.\n`
  if (gate) {
    p += `3. Validate the INTEGRATED tree. A clean text-level merge does NOT mean the combination builds — two independently-developed tasks can each pass review yet break when merged. Determine the project's gates (build/test/lint/format) the same way an implementer/reviewer would (from the task's acceptance criteria and the repo's conventions) and run them in "${wtPath}" on the merged tree. Point the build at the main checkout's shared build/dependency cache if it has one, to avoid a cold build.\n`
    p += `   - If ANY gate fails: leave the worktree in place and return ok=false, committed=true, gatesFailed=true, with the failing-gate output in error. Do NOT try to fix it.\n`
  }
  p += `Return ok (true ONLY if you committed, folded with NO conflict${gate ? ', AND all gates passed' : ''}), committed (bool), conflict (bool), gatesFailed (bool), and any error. Do NOT run \`git branch -f\` and do NOT remove the worktree.`
  return p
}

function advancePrompt(repoRoot, taskBranch, planBranch) {
  return `Task branch "${taskBranch}" has been merged and validated. Advance the run's plan branch "${planBranch}" to include it, as a FAST-FORWARD ONLY. Do NOT modify any files or any other branch.\n`
    + `1. Refuse a non-fast-forward (a non-FF advance would silently DROP already-landed tasks):\n`
    + `   git -C "${repoRoot}" merge-base --is-ancestor "${planBranch}" "${taskBranch}"\n`
    + `   - If that command exits NON-zero ("${planBranch}" is not contained in "${taskBranch}"): do NOT modify "${planBranch}"; return ok=false, advanced=false, error="refusing non-fast-forward advance".\n`
    + `   - Only if it exits 0: \`git -C "${repoRoot}" branch -f "${planBranch}" "${taskBranch}"\`\n`
    + `2. Confirm the advance is real (ground truth): \`git -C "${repoRoot}" merge-base --is-ancestor "${taskBranch}" "${planBranch}"\` — set advanced=true ONLY if this exits 0.\n`
    + `Return ok, advanced (bool — from the final ground-truth confirm), and any error.`
}

function cleanupWorktreePrompt(repoRoot, wtPath, taskBranch) {
  return `Task work has landed on the plan branch. Remove its now-merged worktree; its branch "${taskBranch}" MUST remain.\n`
    + `   git -C "${repoRoot}" worktree remove --force "${wtPath}"\n`
    + `Return ok=true if the worktree is gone (or was already absent), ok=false + error otherwise. Do NOT delete or modify any branch.`
}

function containmentCheckPrompt(repoRoot, taskBranch, planBranch) {
  return `In the git repo at ${repoRoot}, determine whether branch "${planBranch}" already contains every commit of branch "${taskBranch}".\n`
    + `   git -C "${repoRoot}" merge-base --is-ancestor "${taskBranch}" "${planBranch}"\n`
    + `Return contained=true iff that command exits 0, else contained=false. Do NOT modify anything.`
}

function integratePrompt(repoRoot, target, planBranch, message) {
  return `Every task in this plan passed review. Squash-merge the plan branch into the base branch as ONE commit, then delete the run's scratch branches.\n`
    + `In ${repoRoot}:\n`
    + `1. Make sure the working tree is on the base branch: \`git -C "${repoRoot}" switch "${target}"\`\n`
    + `2. Squash all of the plan branch's work into the index (no merge commit):\n`
    + `   git -C "${repoRoot}" merge --squash "${planBranch}"\n`
    + `3. Commit it with EXACTLY this message — write the message to a temp file and use \`git commit -F <file>\`:\n`
    + `-----\n${message}\n-----\n`
    + `4. Delete the now-merged scratch branches: \`git -C "${repoRoot}" branch --list 'implement-plan/${PLAN}*' | sed 's/^[* ]*//' | xargs -r -n1 git -C "${repoRoot}" branch -D\`\n`
    + `Do NOT push. Return ok, committed (bool), sha (the new commit on ${target}), and any error.`
}

// Best-effort: update the plan's STATUS.md (per-task) and the roll-up board (one row per plan) to
// reflect this run's outcome. PRESERVES human-authored prose; edits are left UNCOMMITTED. Never fails
// the run — a status-doc write is reporting, not part of the merge correctness.
function statusUpdatePrompt(repoRoot, statusPath, rollupPath, planLabel, base, planBranch, integratedSha, allPassed, taskRows, total, landedCount) {
  const planStatus = allPassed ? '✅ Complete' : (landedCount > 0 ? '🚧 In progress' : '⛔ Blocked')
  let p = `Update this plan's status docs to reflect the implement-plan run that just finished, in ${repoRoot}. First get today's date: run \`date +%Y-%m-%d\`. PRESERVE all human-authored prose (goal, baseline, outcome narrative); change ONLY status markers, per-task/workstream state cells, counts, and the "Last updated" line.\n`
  p += `Run result: plan "${planLabel}", base branch "${base}", ${landedCount}/${total} task(s) landed on ${planBranch}. `
  p += allPassed ? `All landed and squash-merged into ${base}${integratedSha ? ` as ${integratedSha}` : ''}.\n` : `NOT merged into ${base} — the landed tasks remain on ${planBranch} for a future resume.\n`
  p += `Plan-level status to set: ${planStatus}.\n`
  p += `Per-task outcome (id → state): ${JSON.stringify(taskRows)}\n`
  p += `Legend — plan-level: 📋 Planned · 🚧 In progress · ✅ Complete · ⛔ Blocked · 🗄️ Superseded. Task/workstream cells: ✅ Done / 🚧 In progress / ⛔ Blocked / ❌ Failed.\n`
  let n = 1
  if (statusPath) {
    p += `${n}. Per-plan STATUS at ${repoRoot}/${statusPath} (create it in the project's STATUS.md house style if missing): set each task or workstream State from the per-task outcomes above (a workstream is ✅ Done only if every one of its tasks landed; 🚧 if some did; ⛔/❌ if blocked/failed). Update the top "**Status:**" summary to ${planStatus} and refresh the "Last updated" line to today's date and "${base}". Do NOT invent measured results you were not given.\n`
    n++
  }
  if (rollupPath) {
    p += `${n}. Roll-up board at ${repoRoot}/${rollupPath}: update ONLY this plan's row — Status = ${planStatus}, the tasks-done/total count = ${landedCount}/${total}, keep or lightly trim the one-line outcome, and refresh the board's "Last updated" line to today's date. If no row exists for this plan, add one in plan-number order. Leave every other row byte-for-byte untouched.\n`
    n++
  }
  p += `Touch ONLY these status file(s); do NOT modify code, other docs, or git state, and do NOT commit. Return ok=true if the file(s) were updated, ok=false + error otherwise.`
  return p
}

// ── Concurrency helpers ───────────────────────────────────────────────────────
// A counting semaphore caps how many tasks hold a worktree at once (disk/load bound).
function makeSemaphore(max) {
  let avail = max
  const waiters = []
  return {
    async acquire() {
      if (avail > 0) { avail--; return }
      await new Promise(res => waiters.push(res))   // slot handed over on release()
    },
    release() {
      const w = waiters.shift()
      if (w) w()        // pass the slot directly to the next waiter (avail unchanged)
      else avail++
    },
  }
}

// A promise-chain mutex. Two are used: one serializes the integrate -> gate -> advance step
// (the "merge queue", so the plan branch advances one task at a time and integration gates don't
// interleave), the other serializes fast worktree add/remove git admin. They are distinct so a
// long integration build (under the merge lock) does not stall a ready task's worktree creation
// (under the worktree lock). The slow work — implement + review inside an isolated worktree —
// runs OUTSIDE both locks and stays fully parallel.
function makeMutex() {
  let chain = Promise.resolve()
  return (fn) => {
    const run = chain.then(fn, fn)          // run fn after the previous settles (ok or not)
    chain = run.then(() => {}, () => {})    // swallow result so the chain never breaks
    return run
  }
}

// ── 1. Discover plan + dependency graph + status + git roots ───────────────────
phase('Discover')
const d = await agent(
  `You are the discovery step of an implement-plan run for the plan selector "${PLAN}" in this git repository. Do everything READ-ONLY (no edits, no branch changes, no commits).\n`
  + `\n`
  + `1. LOCATE THE PLAN. Plans live either as a dedicated directory (e.g. docs/plans/${PLAN}-*/ holding TASKS.md, SCOPE.md, ARCHITECTURE.md, STATUS.md) or as a single Markdown file (e.g. docs/superpowers/plans/*.md, doc/plans/*, plans/*). The selector usually matches the start of the plan directory or file name.\n`
  + `   - planDir = the plan's directory relative to repoRoot (for a single-file plan, the directory that contains the file).\n`
  + `   - tasksPath = "<planDir>/TASKS.md" relative to repoRoot IF the plan has a dedicated folder; for a bare single-file plan whose folder is shared with other plans, set tasksPath = "".\n`
  + `   - statusPath = "<planDir>/STATUS.md" relative to repoRoot for a dedicated-folder plan, else "".\n`
  + `   - rollupPath = the plans-root roll-up board "<plans-root>/STATUS.md" (e.g. docs/plans/STATUS.md) relative to repoRoot if one exists, else "".\n`
  + `\n`
  + `2. GET THE TASK GRAPH.\n`
  + `   - IF a TASKS.md exists at tasksPath: PARSE it and set tasksFromFile=true. Convention: workstream phase headers "## NNNN — Name"; each task is "### {kebab-id} — {Title}"; under it a bullet "- **Depends on:** a, b" (a literal "—", "-", "none", or empty means NO dependencies) and a bullet "- **Done when:** <criterion>". For each task extract: id (the kebab id), title, dependsOn (array of the kebab ids listed; [] if none), doneWhen (the Done-when text). If a task's title ends with "(GATED)", set gated=true (else false). Use the ids and edges EXACTLY as written — the authored "Depends on" lines are the dependency graph as authored; do NOT add, infer, or "defensively" insert edges that are not written in the file (a later verification step audits the graph and adds any missing collision/prerequisite edges — that is not your job here).\n`
  + `   - ELSE (no TASKS.md at tasksPath): set tasksFromFile=false and INFER the tasks from the plan's prose. Return tasks in document order with id (unique kebab-case), title, doneWhen, gated=false, and a dependsOn array capturing the TRUE dependency graph: a task depends on another ONLY if it needs that task's code/files/types/artifacts to be implemented or to make its own gates pass; tasks touching disjoint files with no cross-reference get an empty dependsOn so they run concurrently; honor explicit "depends on"/"after"/"requires" wording; bias toward correctness — when you genuinely cannot tell whether two tasks would collide, add the dependency.\n`
  + `   - FOR EVERY TASK (both branches above), also populate "touches": the repo-relative files (or directory paths) the task is expected to CREATE or MODIFY. Read this from the task's steps / acceptance criteria, any explicit file paths it names, AND the plan's design docs in planDir (SCOPE.md / ARCHITECTURE.md / overview) — not TASKS.md alone, which often omits file lists. This is the input to collision detection: bias toward listing MORE files — an over-broad footprint only costs a little parallelism, a missing one risks two tasks corrupting the same file. Use [] only when the touched files are genuinely undeterminable from the plan.\n`
  + `\n`
  + `3. DETECT RESUME STATE (git is the ground truth, not any STATUS doc). Use EXACT ref names — never substring/fuzzy branch matching (other plans may share a number prefix). Check whether the integration branch exists EXACTLY: \`git -C <repoRoot> rev-parse --verify --quiet "refs/heads/${PLAN_BRANCH}"\` (exit 0 ⇒ exists). Set planBranchExists from THAT exact ref only — a branch like "implement-plan/<something-else>/${PLAN}" or "implement-plan/${PLAN}-other" does NOT count. If and only if it exists, then for EACH task set landed=true IFF \`git -C <repoRoot> rev-parse --verify --quiet "refs/heads/${PLAN_BRANCH}--<id>"\` succeeds AND \`git -C <repoRoot> merge-base --is-ancestor "refs/heads/${PLAN_BRANCH}--<id>" "refs/heads/${PLAN_BRANCH}"\` exits 0. Otherwise landed=false. If planBranchExists is false, landed=false for every task.\n`
  + `\n`
  + `4. GIT ROOTS: repoRoot (\`git rev-parse --show-toplevel\`), startSha (\`git rev-parse HEAD\`), startRef (\`git rev-parse --abbrev-ref HEAD\`).\n`
  + `Each task's id MUST be unique within the plan; if two headings would collide, qualify them. Return the full structured result. Do not modify anything.`,
  { label: `discover:${PLAN}`, phase: 'Discover', schema: DISCOVER_SCHEMA }
)
if (!d || !d.tasks || d.tasks.length === 0 || !d.repoRoot) {
  log(`Discovery failed for plan "${PLAN}" (no tasks or no repo root).`)
  return { plan: PLAN, error: 'discovery failed' }
}
const TARGET = cfg.into || d.startRef    // base branch to integrate back into
const PLAN_LABEL = (d.planDir || '').split('/').filter(Boolean).pop() || PLAN

// ── Normalize + validate the dependency graph ─────────────────────────────────
const idSet = new Set(d.tasks.map(t => t.id))
if (idSet.size !== d.tasks.length) {
  // Non-unique ids would collapse the promiseById map and make two tasks share a worktree/branch
  // (data race), and a dependsOn reference to a duplicated id is itself ambiguous — abort.
  const counts = new Map()
  for (const t of d.tasks) counts.set(t.id, (counts.get(t.id) || 0) + 1)
  const dupes = [...counts].filter(([, n]) => n > 1).map(([id]) => id)
  log(`Discovery returned non-unique task ids (${dupes.join(', ')}); aborting — ids must be unique to schedule and merge correctly.`)
  return { plan: d.planDir, error: 'duplicate task ids', duplicates: dupes }
}
for (let i = 0; i < d.tasks.length; i++) {
  const t = d.tasks[i]
  const raw = Array.isArray(t.dependsOn) ? t.dependsOn : []
  const known = raw.filter(x => x !== t.id && idSet.has(x))
  const unknown = raw.filter(x => x !== t.id && !idSet.has(x))
  if (unknown.length) log(`task ${t.id}: ignoring unknown dependency id(s) ${unknown.join(', ')}.`)
  t.deps = known
  t.gated = !!t.gated
  t.landed = !!(d.planBranchExists && t.landed)
  t.touches = Array.isArray(t.touches) ? t.touches.filter(x => typeof x === 'string' && x.trim()).map(x => x.trim()) : []
}
if (FORCE_SEQUENTIAL) {
  for (let i = 0; i < d.tasks.length; i++) d.tasks[i].deps = i > 0 ? [d.tasks[i - 1].id] : []
  log(`sequential mode: forcing a linear chain over ${d.tasks.length} tasks (overriding the authored/inferred graph).`)
}

// ── 2. Verify the dependency graph BEFORE any execution ────────────────────────
// The scheduler runs dependency-free tasks CONCURRENTLY. If the authored/inferred graph leaves two
// tasks that touch the same files (or where one needs the other's code) WITHOUT an edge between them,
// they run in parallel and corrupt each other or fail to merge. This phase audits the graph and
// AUTO-ADDS the missing serializing edges; it never removes an authored edge. It is fully read-only
// (analysis only — any inferred TASKS.md is written from the resulting verified graph in Setup), so it
// also runs under dryRun. Skipped under sequential mode (a linear chain is already collision-free).
phase('Verify')
const byIdV = new Map(d.tasks.map(t => [t.id, t]))
// Is `x` a (transitive) dependency of `y` over the LIVE deps, so edges added below are seen at once?
function isAncestor(x, y) {
  const seen = new Set()
  const start = byIdV.get(y)
  const stack = start ? [...start.deps] : []
  while (stack.length) {
    const cur = stack.pop()
    if (cur === x) return true
    if (seen.has(cur)) continue
    seen.add(cur)
    const ct = byIdV.get(cur)
    if (ct) for (const dep of ct.deps) stack.push(dep)
  }
  return false
}
const hasPath = (a, b) => isAncestor(a, b) || isAncestor(b, a)
// Add "to depends on from" if safe (known ids, not self, not already present, stays acyclic).
function addDependencyEdge(fromId, toId, reason) {
  if (fromId === toId) return false
  const from = byIdV.get(fromId), to = byIdV.get(toId)
  if (!from || !to) return false
  if (to.deps.includes(fromId) || isAncestor(fromId, toId)) return false   // already implied
  if (isAncestor(toId, fromId)) {                                          // reverse path exists → would cycle
    log(`verify: NOT adding "${toId} depends on ${fromId}" (${reason}) — the reverse dependency already exists; would create a cycle. Leaving as-is.`)
    return false
  }
  to.deps.push(fromId)
  return true
}

const addedEdges = []
if (FORCE_SEQUENTIAL) {
  log(`verify: skipped — sequential mode already serializes every task (no parallel collisions possible).`)
} else {
  // (a) Deterministic file-collision backstop over the extracted footprints. Among tasks that will
  // actually run concurrently (runnable = not gated, not already landed), any unordered pair that
  // shares a file but has no dependency path between them is serialized in document order. This does
  // not depend on the agent below — it is the reliable floor.
  const normPath = (p) => p.replace(/^\.?\/+/, '').replace(/\/+$/, '')
  const sharedFile = (af, bf) => {
    for (const a of af) for (const b of bf) {
      const na = normPath(a), nb = normPath(b)
      if (!na || !nb) continue
      if (na === nb) return na
      // The "+ '/'" makes this a directory-BOUNDARY test (normPath already stripped trailing slashes),
      // so "src" matches "src/foo.rs" but NOT "src-old/foo.rs".
      if (na.startsWith(nb + '/') || nb.startsWith(na + '/')) return `${na} ∩ ${nb}`
    }
    return null
  }
  const runnableTasks = d.tasks.filter(t => !t.gated && !t.landed)
  const noFootprint = runnableTasks.filter(t => !t.touches.length).length
  for (let i = 0; i < runnableTasks.length; i++) {
    for (let j = i + 1; j < runnableTasks.length; j++) {
      const a = runnableTasks[i], b = runnableTasks[j]   // a is earlier in document order → runs first
      const hit = sharedFile(a.touches, b.touches)
      if (hit && !hasPath(a.id, b.id) && addDependencyEdge(a.id, b.id, `shared file ${hit}`)) {
        addedEdges.push({ from: a.id, to: b.id, reason: `shared file ${hit}`, src: 'file-collision' })
        log(`verify: ${b.id} now depends on ${a.id} — both modify ${hit} and had no edge (they would have run in parallel and collided).`)
      }
    }
  }
  if (noFootprint) log(`verify: ${noFootprint}/${runnableTasks.length} runnable task(s) had no extractable file footprint — file-collision detection could not cover them; the semantic audit below may still catch prerequisites, and consider maxParallel:1 if unsure.`)

  // (b) Semantic audit (agent): prerequisites file-overlap can't see — task B needs a type/function/
  // migration task A introduces — plus warnings about wrong/unjustified existing edges. It audits ONLY
  // the runnable tasks and only edges BETWEEN runnable tasks are accepted: a LANDED task's work is
  // already in every new worktree's base (no edge needed), and a GATED task never auto-runs (it cannot
  // be a prerequisite, and a dep added to it would be inert) — accepting such edges would only pollute
  // the "fold into TASKS.md" advisory with permanent-but-spurious dependencies. Skipped entirely when
  // fewer than two runnable tasks exist (nothing could collide or depend), which also avoids an
  // avoidable agent call on a fully-resumed / all-gated run.
  const runnableIds = new Set(runnableTasks.map(t => t.id))
  if (runnableTasks.length >= 2) {
    const v = await agent(
      verifyPrompt(d.repoRoot, d.planDir, runnableTasks),
      { label: `verify:${PLAN}`, phase: 'Verify', model: REVIEW_MODEL, schema: VERIFY_SCHEMA }
    )
    if (v && Array.isArray(v.addEdges)) {
      for (const e of v.addEdges) {
        const fromId = String((e && e.from) || ''), toId = String((e && e.to) || '')
        if (!byIdV.has(fromId) || !byIdV.has(toId)) { log(`verify: ignoring proposed edge ${fromId || '?'} → ${toId || '?'} (unknown task id).`); continue }
        if (!runnableIds.has(fromId) || !runnableIds.has(toId)) { log(`verify: ignoring proposed edge ${fromId} → ${toId} — only edges between runnable (not gated, not already-landed) tasks are applied.`); continue }
        if (addDependencyEdge(fromId, toId, e.reason || 'verifier-required prerequisite')) {
          addedEdges.push({ from: fromId, to: toId, reason: e.reason || 'verifier-required prerequisite', src: 'verifier' })
          log(`verify: ${toId} now depends on ${fromId} — ${e.reason || 'verifier-flagged missing prerequisite'}.`)
        }
      }
    } else {
      log(`verify: semantic audit returned no result; relied on the deterministic file-collision backstop only.`)
    }
    if (v && Array.isArray(v.warnings)) for (const w of v.warnings) log(`verify ⚠: ${w}`)
  } else {
    log(`verify: skipping semantic audit — fewer than two runnable tasks, so nothing could collide or depend.`)
  }

  if (addedEdges.length) {
    log(`verify: added ${addedEdges.length} safety edge(s) to prevent parallel collisions / missing prerequisites; the schedule reflects them.`)
    if (d.tasksFromFile) log(`verify: these edges apply to THIS run only (the authored ${d.tasksPath || 'TASKS.md'} was not modified). Fold them into the file to make them permanent: ${addedEdges.map(e => `${e.to}←${e.from}`).join(', ')}.`)
  } else {
    const landedNow = d.tasks.filter(t => t.landed).length
    const scope = landedNow ? ` among the ${runnableTasks.length} runnable task(s) (${landedNow} already-landed not re-checked — they are baked into the integration branch and covered by the merge queue)` : ''
    log(`verify: graph is consistent — found no missing collision/prerequisite edges${scope}.`)
  }
}

// Topological sort (Kahn) — also detects cycles. On a cycle, fall back to a safe linear chain.
function topoOrder(tasks) {
  const byId = new Map(tasks.map(t => [t.id, t]))
  const indeg = new Map(tasks.map(t => [t.id, 0]))
  const adj = new Map(tasks.map(t => [t.id, []]))
  for (const t of tasks) for (const dep of t.deps) {
    adj.get(dep).push(t.id)
    indeg.set(t.id, indeg.get(t.id) + 1)
  }
  const queue = tasks.filter(t => indeg.get(t.id) === 0).map(t => t.id)  // document-order stable
  const order = []
  while (queue.length) {
    const id = queue.shift()
    order.push(id)
    for (const nxt of adj.get(id)) {
      indeg.set(nxt, indeg.get(nxt) - 1)
      if (indeg.get(nxt) === 0) queue.push(nxt)
    }
  }
  return order.length === tasks.length ? order.map(id => byId.get(id)) : null
}
let ordered = topoOrder(d.tasks)
if (!ordered) {
  log(`WARNING: dependency cycle detected; falling back to a safe sequential chain.`)
  for (let i = 0; i < d.tasks.length; i++) d.tasks[i].deps = i > 0 ? [d.tasks[i - 1].id] : []
  ordered = topoOrder(d.tasks)
}
const landedIds = new Set(d.tasks.filter(t => t.landed).map(t => t.id))
const gatedIds = new Set(d.tasks.filter(t => t.gated).map(t => t.id))
const RESUME = !!(d.planBranchExists && landedIds.size > 0)   // unconditional — always resume prior landed work
const rootCount = d.tasks.filter(t => t.deps.length === 0).length
const graphSrc = d.tasksFromFile ? `authored TASKS.md (${d.tasksPath})` : 'inferred from plan prose'
log(`Plan ${d.planDir}: ${d.tasks.length} tasks (${rootCount} with no deps; ${gatedIds.size} gated) — graph ${graphSrc}. Base "${TARGET}" (${d.startSha.slice(0, 8)}); integration branch ${PLAN_BRANCH}.`)
if (RESUME) log(`RESUME: ${PLAN_BRANCH} already holds ${landedIds.size} landed task(s) (${[...landedIds].join(', ')}); they will be skipped.`)
log(`Running up to ${MAX_PARALLEL} in parallel, honoring dependencies.`)

// ── Dry run: report the parallel execution plan and STOP. Read-only — discovery already ran
// (git reads + file reads only); nothing below this point executes, so no branch, worktree, edit,
// merge, generated TASKS.md, or STATUS write happens.
if (DRY_RUN) {
  phase('Dry run')
  // Execution "waves": a task's wave = 1 + max(dep waves); roots = wave 0. Tasks in one wave have no
  // dependency between them and run concurrently (bounded by maxParallel). This visualizes the fan-out.
  const byId = new Map(d.tasks.map(t => [t.id, t]))
  const waveOf = new Map()
  function computeWave(t) {
    if (waveOf.has(t.id)) return waveOf.get(t.id)
    let w = 0
    for (const dep of t.deps) { const dt = byId.get(dep); if (dt) w = Math.max(w, computeWave(dt) + 1) }
    waveOf.set(t.id, w)
    return w
  }
  for (const t of ordered) computeWave(t)
  const maxWave = ordered.length ? Math.max(...[...waveOf.values()]) : 0
  const waves = []
  for (let w = 0; w <= maxWave; w++) waves.push(ordered.filter(t => waveOf.get(t.id) === w))
  const widths = waves.map(w => w.filter(t => !t.gated && !t.landed).length)
  const wouldRun = ordered.filter(t => !t.gated && !t.landed)
  log(`DRY RUN — NOTHING is created or modified (no branch, worktree, code edit, merge, generated TASKS.md, or STATUS write). Plan ${d.planDir}; graph ${graphSrc}; base "${TARGET}"; resume=${RESUME}.`)
  log(`${d.tasks.length} task(s) across ${maxWave + 1} dependency wave(s); widest concurrent wave = ${Math.max(0, ...widths)} runnable (capped at maxParallel=${MAX_PARALLEL}). Would run ${wouldRun.length}; skip ${gatedIds.size} gated + ${landedIds.size} already-landed.`)
  for (let w = 0; w < waves.length; w++) {
    const names = waves[w].map(t => `${t.id}${t.gated ? ' [GATED→skip]' : t.landed ? ' [landed→skip]' : ''}`)
    log(`  wave ${w} (${waves[w].length}): ${names.join(', ')}`)
  }
  if (addedEdges.length) log(`  verification added ${addedEdges.length} safety edge(s): ${addedEdges.map(e => `${e.to}←${e.from} (${e.src})`).join(', ')}`)
  log(`Dry run complete. Re-run without dryRun to execute.`)
  return {
    plan: d.planDir, dryRun: true,
    graphSource: d.tasksFromFile ? 'tasks-md' : 'inferred',
    base: TARGET, planBranch: PLAN_BRANCH, resume: RESUME,
    taskCount: d.tasks.length, waveCount: maxWave + 1, widestWave: Math.max(0, ...widths),
    waves: waves.map(w => w.map(t => ({ id: t.id, deps: t.deps, gated: t.gated, landed: t.landed }))),
    wouldRun: wouldRun.map(t => t.id), gated: [...gatedIds], landed: [...landedIds],
    addedEdges,
  }
}

// ── 3. Setup: the run's plan integration branch (fresh, or reuse to resume) ────
phase('Setup')
const setup = await agent(
  RESUME ? setupResumePrompt(d.repoRoot, PLAN_BRANCH) : setupPrompt(d.repoRoot, PLAN_BRANCH, d.startSha),
  { label: `setup:${PLAN}`, phase: 'Setup', model: 'haiku', schema: OK_SCHEMA }
)
if (!setup || !setup.ok) {
  log(`Setup failed (${(setup && setup.error) || 'unknown'}); aborting.`)
  return { plan: d.planDir, error: 'setup failed', detail: setup && setup.error }
}

// Generate TASKS.md if it was missing (and the plan has a dedicated folder) so future runs use the
// authored graph instead of re-inferring it. Best-effort — never blocks the run.
if (!d.tasksFromFile && d.tasksPath) {
  const gen = await agent(
    generateTasksPrompt(d.repoRoot, d.tasksPath, PLAN_LABEL, d.tasks),
    { label: `gen-tasks:${PLAN}`, phase: 'Setup', model: REVIEW_MODEL, schema: OK_SCHEMA }
  )
  if (gen && gen.ok) log(`Generated ${d.tasksPath} from the inferred graph (UNCOMMITTED — review and refine, then commit).`)
  else log(`Could not generate TASKS.md (${(gen && gen.error) || 'unknown'}); continuing with the inferred graph in memory.`)
}

// ── 4. Schedule by dependency: ready tasks run in parallel; merges are serialized ─
phase('Implement')
const sem = makeSemaphore(MAX_PARALLEL)
const withMergeLock = makeMutex()      // serializes the integrate -> gate -> advance step (the "merge queue")
const withWorktreeLock = makeMutex()   // serializes worktree add/remove (fast shared-repo git admin)
const promiseById = new Map()          // id -> Promise<result>

async function runTask(task) {
  const taskBranch = taskBranchOf(task.id)
  try {
    // A GATED task is a conditional Phase-2 follow-up — its start is a human decision, so never
    // auto-run it. It is reported but does not block integration of the non-gated work.
    if (task.gated) {
      log(`task ${task.id}: GATED — conditional follow-up, not auto-run; left for a manual decision.`)
      return { id: task.id, title: task.title, approved: false, merged: false, gated: true, branch: taskBranch }
    }
    // Resume: this task already landed on the plan branch in a prior run (git ground truth). Skip it
    // and report it as merged so its dependents proceed without rebuilding it.
    if (task.landed) {
      log(`task ${task.id}: already landed on ${PLAN_BRANCH} (resume); skipping.`)
      return { id: task.id, title: task.title, approved: true, merged: true, skipped: true, conflict: false, gatesFailed: false, iterations: 0, branch: taskBranch, findings: [] }
    }

    // Wait for every dependency to finish. A dep that did not LAND (failed review, merge
    // conflict, worktree error, gated) blocks this task — we must not build on an absent base.
    for (const depId of task.deps) {
      const depResult = await (promiseById.get(depId) || Promise.resolve(null))
      if (!depResult || !depResult.merged) {
        log(`task ${task.id}: BLOCKED — dependency ${depId} did not land; skipping.`)
        return { id: task.id, title: task.title, approved: false, merged: false, blocked: true, reason: `dependency ${depId} did not land`, branch: taskBranch }
      }
    }

    await sem.acquire()
    try {
      const wtPath = `${d.repoRoot}/.worktrees/implement-plan/${PLAN}--${task.id}`
      // Worktree creation touches shared git admin — serialize it under the worktree lock.
      const wt = await withWorktreeLock(() => agent(
        worktreeCreatePrompt(d.repoRoot, PLAN_BRANCH, wtPath, taskBranch),
        { label: `worktree:${task.id}`, phase: 'Implement', model: 'haiku', schema: OK_SCHEMA }
      ))
      if (!wt || !wt.ok) {
        log(`task ${task.id}: worktree create FAILED (${(wt && wt.error) || 'unknown'}); skipping.`)
        return { id: task.id, title: task.title, approved: false, merged: false, error: 'worktree create failed', branch: taskBranch }
      }

      // implement → review loop (the parallel, expensive part — NOT under any lock)
      let findings = null
      let verdict = null
      let iters = 0
      let devModel = DEV_MODEL
      for (let i = 1; i <= MAX_ITERS; i++) {
        iters = i
        const implReport = await agent(
          developerPrompt(d.repoRoot, d.planDir, task, wtPath, findings),
          { label: `impl:${task.id}#${i}`, phase: 'Implement', model: devModel, agentType: 'developer' }
        )
        verdict = await agent(
          reviewerPrompt(d.repoRoot, d.planDir, task, wtPath, implReport),
          { label: `review:${task.id}#${i}`, phase: 'Implement', model: REVIEW_MODEL, agentType: 'reviewer', schema: VERDICT_SCHEMA }
        )
        if (verdict && verdict.approved && verdict.gatesPass) break
        findings = (verdict && verdict.findings && verdict.findings.length)
          ? verdict.findings
          : [{ severity: 'blocker', note: 'reviewer returned no verdict / no findings; treat as rejected' }]
        devModel = REVIEW_MODEL   // escalation ladder after the first rejected round
        log(`task ${task.id}: round ${i} not approved (${findings.length} finding(s)); escalating developer to ${devModel}.`)
      }

      const approved = !!(verdict && verdict.approved && verdict.gatesPass)
      if (!approved) {
        log(`task ${task.id}: STOPPED after ${iters} round(s); worktree kept at ${wtPath} (branch ${taskBranch}) for inspection.`)
        return { id: task.id, title: task.title, approved: false, merged: false, iterations: iters, branch: taskBranch, worktree: wtPath, findings: (verdict && verdict.findings) || [] }
      }

      // Serialized integrate -> gate -> advance (the "merge queue"). The merge+gate agent NEVER
      // touches the plan branch; the orchestrator advances it (a separate agent) ONLY after a clean,
      // gate-passing report — so no agent step-ordering can leave PLAN_BRANCH advanced with
      // unvalidated code. The whole sequence holds withMergeLock, so PLAN_BRANCH is stable.
      const outcome = await withMergeLock(async () => {
        const m = await agent(
          mergeTaskPrompt(d.repoRoot, wtPath, taskBranch, PLAN_BRANCH, task, INTEGRATION_GATE),
          { label: `merge:${task.id}`, phase: 'Implement', model: INTEGRATE_MODEL, schema: MERGE_TASK_SCHEMA }
        )
        const clean = !!(m && m.ok && m.committed && !m.conflict && !m.gatesFailed)
        if (!clean) return { report: m, advanced: false }
        let adv = await agent(
          advancePrompt(d.repoRoot, taskBranch, PLAN_BRANCH),
          { label: `advance:${task.id}`, phase: 'Implement', model: 'haiku', schema: ADVANCE_SCHEMA }
        )
        if (!adv) {
          // The advance agent died; it may have run branch -f before dying. Recover the ground
          // truth — it only runs post-gate, so containment here genuinely means "landed".
          const gt = await agent(
            containmentCheckPrompt(d.repoRoot, taskBranch, PLAN_BRANCH),
            { label: `advance-recover:${task.id}`, phase: 'Implement', model: 'haiku', schema: CONTAIN_SCHEMA }
          )
          adv = { ok: !!(gt && gt.contained), advanced: !!(gt && gt.contained) }
        }
        return { report: m, advanced: !!(adv && adv.advanced) }
      })
      const report = outcome && outcome.report
      const conflict = !!(report && report.conflict)
      const gatesFailed = !!(report && report.gatesFailed)
      const landed = !!(outcome && outcome.advanced)
      if (landed) {
        log(`task ${task.id}: APPROVED in ${iters} round(s); merged into ${PLAN_BRANCH}.`)
        const cleaned = await withWorktreeLock(() => agent(
          cleanupWorktreePrompt(d.repoRoot, wtPath, taskBranch),
          { label: `cleanup:${task.id}`, phase: 'Implement', model: 'haiku', schema: OK_SCHEMA }
        ))
        return { id: task.id, title: task.title, approved: true, merged: true, conflict: false, gatesFailed: false, iterations: iters, branch: taskBranch, worktree: (cleaned && cleaned.ok) ? null : wtPath, findings: verdict.findings || [] }
      }
      if (conflict) log(`task ${task.id}: approved but MERGE CONFLICT with ${PLAN_BRANCH} (${(report && report.error) || 'paths unknown'}); worktree kept at ${wtPath}.`)
      else if (gatesFailed) log(`task ${task.id}: approved but INTEGRATION GATES FAILED on the merged ${PLAN_BRANCH} tip (${(report && report.error) || 'see worktree'}); worktree kept at ${wtPath}.`)
      else log(`task ${task.id}: approved but merge/advance FAILED (${(report && report.error) || 'unknown'}); worktree kept at ${wtPath}.`)
      return { id: task.id, title: task.title, approved: true, merged: false, conflict, gatesFailed, iterations: iters, branch: taskBranch, worktree: wtPath, findings: verdict.findings || [] }
    } finally {
      sem.release()
    }
  } catch (e) {
    log(`task ${task.id}: unexpected error (${String((e && e.message) || e)}); treating as failed.`)
    return { id: task.id, title: task.title, approved: false, merged: false, error: String((e && e.message) || e), branch: taskBranch }
  }
}

// Launch every task. The Promise.resolve().then() defer guarantees all promiseById entries
// exist before any runTask body executes, so dependency lookups never miss regardless of order.
for (const task of d.tasks) {
  promiseById.set(task.id, Promise.resolve().then(() => runTask(task)))
}
// Collect in document order (all tasks run concurrently; this just awaits them).
const results = []
for (const task of d.tasks) results.push(await promiseById.get(task.id))

// ── 5. Integrate: squash-merge the plan branch into the base (only if all landed) ─
// Gated tasks are intentionally not run, so they do not count toward "all landed".
phase('Integrate')
const runnable = results.filter(r => r && !r.gated)
const landedTasks = runnable.filter(r => r && r.approved && r.merged)
const allPassed = runnable.length > 0 && landedTasks.length === runnable.length
const gatedCount = results.filter(r => r && r.gated).length
let integrated = null

if (allPassed) {
  const body = landedTasks.map(r => `- ${r.id}: ${r.title}`).join('\n')
  const gatedNote = gatedCount ? `\n(${gatedCount} gated task(s) intentionally not run.)\n` : ''
  const message = `feat(${PLAN_LABEL}): implement plan ${PLAN_LABEL}\n\n`
    + `Implemented and reviewed task-by-task via implement-plan (${landedTasks.length} task(s)):\n\n`
    + `${body}\n${gatedNote}`
  integrated = await agent(
    integratePrompt(d.repoRoot, TARGET, PLAN_BRANCH, message),
    { label: `integrate:${PLAN}`, phase: 'Integrate', model: 'haiku', schema: INTEGRATE_SCHEMA }
  )
  if (integrated && integrated.ok && integrated.committed) {
    log(`Integrated: squash-merged ${PLAN_BRANCH} into ${TARGET} as ${integrated.sha ? integrated.sha.slice(0, 8) : 'one commit'}; scratch branches deleted.`)
  } else {
    log(`Integration FAILED (${(integrated && integrated.error) || 'unknown'}); plan branch ${PLAN_BRANCH} left intact for manual merge.`)
  }
} else {
  const blocked = results.filter(r => r && r.blocked).length
  const conflicted = results.filter(r => r && r.conflict).length
  const gateFailed = results.filter(r => r && r.gatesFailed).length
  log(`Not all runnable tasks landed (${landedTasks.length}/${runnable.length}; ${blocked} blocked, ${conflicted} conflicted, ${gateFailed} integration-gate failed${gatedCount ? `, ${gatedCount} gated/not-run` : ''}); NOT squash-merging into ${TARGET}. Plan branch ${PLAN_BRANCH} holds the landed tasks (re-run to resume); failed tasks' worktrees are under .worktrees/implement-plan/ for inspection.`)
}

const integrateOk = !!(integrated && integrated.ok && integrated.committed)

// ── 6. Status: reflect the run in the plan's STATUS.md + the roll-up board ─────
// Best-effort and UNCOMMITTED — a status-doc write is reporting, never part of merge correctness.
phase('Status')
if (d.statusPath || d.rollupPath) {
  const stateOf = (r) => !r ? 'failed'
    : r.gated ? 'gated (not run)'
    : r.skipped ? 'done (prior run)'
    : (r.approved && r.merged) ? 'done'
    : r.blocked ? 'blocked'
    : r.conflict ? 'conflict'
    : r.gatesFailed ? 'gate-failed'
    : r.approved ? 'approved-not-merged'
    : 'failed'
  const taskRows = results.map(r => ({ id: r && r.id, title: r && r.title, state: stateOf(r) }))
  const st = await agent(
    statusUpdatePrompt(d.repoRoot, d.statusPath, d.rollupPath, PLAN_LABEL, TARGET, PLAN_BRANCH, integrateOk ? integrated.sha : '', allPassed, taskRows, results.length, landedTasks.length),
    { label: `status:${PLAN}`, phase: 'Status', model: REVIEW_MODEL, schema: OK_SCHEMA }
  )
  if (st && st.ok) log(`Updated STATUS docs (UNCOMMITTED — review and commit alongside the plan).`)
  else log(`STATUS update skipped/failed (${(st && st.error) || 'unknown'}); update the board by hand.`)
} else {
  log(`No STATUS.md / roll-up board for this plan; skipping status write-back.`)
}

log(`Done: ${landedTasks.length}/${runnable.length} runnable tasks landed on ${PLAN_BRANCH}${gatedCount ? ` (${gatedCount} gated/not-run)` : ''}; ${integrateOk ? `squashed into ${TARGET}` : `NOT merged into ${TARGET}`}.`)
return {
  plan: d.planDir,
  base: TARGET,
  planBranch: PLAN_BRANCH,
  resumed: RESUME,
  graphSource: d.tasksFromFile ? 'tasks-md' : 'inferred',
  addedEdges,
  landed: landedTasks.length,
  runnable: runnable.length,
  gated: gatedCount,
  total: results.length,
  integratedInto: integrateOk ? TARGET : null,
  integrationSha: integrateOk ? integrated.sha : null,
  results,
}
