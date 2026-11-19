# Makina plans

A Makina plan is an approved, executable specification: an objective and scope,
architecture and safety invariants, a dependency graph of independently verifiable
task packets, and durable progress records; it is not a monolithic checklist.

## Author, review, approve

1. Call plan with action=discover and objective <objective>: this main chat becomes the planning orchestrator
   and performs preflight itself; a discovery architect proposes the approach,
   number of plans, boundaries and dependencies, asking the user only for material choices.
2. Each plan architect owns architecture and task breakdown, delegating each packet's
   details to a separate reviewer-role author; another architect reviews that plan
   for medium-or-higher correctness findings, ignoring nitpicks; a final architect
   reviews only high-level cross-plan coverage, interfaces and dependencies.
3. One total concurrency cap includes nested authors; waiting architects release slots.
   Workers' owned files are committed incrementally by the host on their behalf on
   one isolated plans/<one-to-three-word-slug> branch, suffixed -2, -3, etc. on collisions;
   plans live under docs/plans/NNNN-slug/ and reserve numbers across pending batches.
   The invoking checkout and configured base branch stay unchanged while authoring.
4. Inspect the planning batch with /makina, then explicitly run /makina-approve <branch or number> to validate and squash-merge the exact independently reviewed batch into configured base_branch; the invoking checkout must be clean and on that branch; approval never implements or deletes the planning branch, and unapproved branches may remain idle indefinitely without rejection.

## Required layout and authoring rules

Each numbered plan directory contains:

- PLAN.md: YAML frontmatter objective, depends_on (sibling plan names), status,
  toolchains (every program the plan's checks and roles run from a shell, such
  as cargo or node: roles' sandboxed shells keep only these and system tools,
  and the host checks each resolves on the login PATH), and
  nonempty checks (named full/group commands executed by the host at integration);
  the body defines scope, non-goals, constraints, and acceptance requirements.
- ARCHITECTURE.md: explicit ## Invariants, ## State, and ## Untrusted inputs
  sections, including stable INV identifiers for the guarantees tasks must prove.
  New plans also require ## Feasibility: task | constraints and sources |
  compatibility witness | preflight check, exactly one row per runnable task;
  derive compatible boundary values from actual sources, with explicit units/scope.
- STATUS.md: plan status frontmatter and current progress, verification, and run notes.
- tasks/WWSS-slug.md: one task per packet, ordered by workstream/sequence prefix;
  filename slug matches frontmatter id, with title, workstream, kind (task or chore),
  depends_on (task IDs), gated: false (never true for new plans, and never an external
  owner, human verifier/validator or approval step), touches (repository-relative file paths),
  status, merged_as (initially an empty string), and nonempty checks (named lightweight
  commands proving this task's Done when, executed by the host with task scope).
- execution/: architect-owned static fixtures and executable lifecycle probes,
  committed by the host with the three plan documents, never product implementation.

Start plan status at todo and task status at todo, with no fabricated landing OIDs.
Every task has ordered `**Steps:**` (at most six) and exactly one falsifiable
`- **Done when:**` criterion citing the INV identifiers it proves.
A command the Done when spells in backticks must be run whole by one of the task's
checks or a PLAN.md check, options aside; otherwise name the check. A check that runs
it with a further argument proves less. No role has a shell, so a command no approved
check runs can never be proved.
Every declared invariant must be proved by a task, and no task may cite undeclared
invariants; use executable proofs rather than assertions of completion.

Each packet must be junior-friendly, settle design choices, and deliver exactly
one outcome with implementation and its tests together; assign one owner per file
and use disjoint paths for safe parallel work, not whole-tree construction globs.
Do not bundle unrelated goals or compound package proofs into a single Done when.
Use only genuine prerequisite edges: task IDs must be unique, dependencies valid,
and the graph acyclic; new plans may contain only unattended factory implementation,
so gated or externally owned tasks, verification/validation-only packets, all-gated
plans and historical monolithic task lists are refused.
Task count and wave width are not capped by execution concurrency.
Validate drafts with validate_plan and resolve every issue, including its
static baseline-source readiness report, before approval.

## Commence implementation

Initialization commits its newly created scaffold files on develop; commit any
intended project changes first, ensure Git identity is configured, and use the
configured base branch (develop by default).
Author checks as [{name: focused-proof, command: "exact proof command"}] in task
frontmatter and group/full checks in PLAN.md frontmatter; approval authorizes the
exact commands, and their document assigns task/integration scope automatically.
Optional timeout_ms and complete inputs globs follow the factory gate contract.
PLAN.md also declares nonempty preflight_checks with that schema: lightweight
baseline witnesses executable before future product code exists, proving the
Feasibility table's assumptions; approval, implementation and saved-run repair
acceptance execute these checks and refuse on failure or checkout mutation.
Preflights run in a fresh isolated checkout with no installed dependencies:
provision packages inside the preflight command itself, or use a dependency-free
source/API witness; setup_checks prepare execution witnesses, not preflights.
Never hand contradictory acceptance requirements to implementation; repairing
records or saved baselines alone does not establish the pending plan's feasibility.
Existing project checks in .makina/factory.toml remain supported: lightweight
checks use scope = "task"; full suites use scope = "integration"; missing task
proof routes refuse before workers start, and default true is not a project proof.
A task is judged only by its own checks and the task-scoped gates; integration-scoped
checks and gates judge the combined plan tree once, after the last task lands. A
repo-wide check any task can break (a formatter, a linter) therefore belongs in
that task's checks or in a task-scoped gate, and a check a command can correct
declares it as fix: the host runs the fix itself when the check fails.
Reviews must reject proofs requiring forbidden worker shell access, missing setup
ownership, and work no authorized architect can safely complete within approved scope.
Task role defaults to developer; declare role: architect for bootstrap, initial
dependency/lockfile creation, native setup or other work requiring raw shell
authority from the first attempt; assigned architects retain bounded correction
ownership and independent reviewers without first consuming developer retries.
Task setup_checks declares exact named task-scoped preparation commands run by
the host before every named proof; PLAN.md setup_checks prepares integration.
Preparation is uncached and may change only ignored dependency/build output,
never project files, index or HEAD; initial lockfile generation belongs to the
architect implementation, not read-only preparation in later fresh worktrees.
PLAN.md execution_checks must cover every pending task exactly once and include
a separate architect tasks: ["$integration"] witness without shell actions and
with no operations beyond edit (omit operations or list [edit]: its command is
host-run, so the host refuses shell or copy there).
With source-specific constraints covering $integration, its command may be a
narrow probe whose named constraint checks exercise the real joint baseline
sources; without constraints (legacy), its command must exactly equal one of the
commands declared in PLAN.md checks. A generic environment check is never an
approved integration witness.
Each entry uses name, tasks, role, command, optional fixtures [{source, target}]
and optional actions (named task-scoped commands, architect-owned tasks only).
Fixture sources are committed plan artifacts; targets must be owned by every
represented task/prerequisite chain, or be private .makina/lifecycle/ probes.
Prefer block-style YAML for long checks, preflight_checks and execution_checks
lists; one-line flow arrays can exceed the authoring read limit and hide entries.
Group tasks only with matching roles and identical setup_checks. The host seeds
fixtures in fresh disposable checkouts, executes authorized actions and the
actual task/integration preparation, then the probe; no warmed node_modules or
installation in an unrelated directory can establish this execution route.
Task witnesses must declare operations: [edit|copy|shell] matching every packet
step; developers may only edit, while exact filesystem copies and shell/native
generation require architect ownership and approved actions demonstrating them.
Host fixture seeding does not prove that a worker has lossless copy authority.
Prefer execution witness constraints: [{task, name, sources, checks}] covering
every represented pending task and a separate $integration constraint; sources
and fixture sources must be tracked regular files at the witness baseline (HEAD
for a new plan), never files the plan's tasks will create—list future paths in
touches. Named task-scoped checks must exercise real frozen naming/AST/API,
boundary, setup and proof-runner constraints on existing code or narrow samples.
Review must audit each task's actual operations and source constraints,
rejecting generic environment-only probes, omitted policies, checker stubs and
incompatible required external option/property names.
The host executes the witness and these source-specific checks, not automatic
future task/full-product acceptance gates: do not implement all unfinished tasks
or fake their tests just to establish feasibility. Legacy witnesses without
constraints explicitly opt into their matching acceptance command only.
Structural validation/loading is not readiness: older records remain readable
for authorized repair, which must still pass independent execution validation.
Saved-run evidence uses verified integration history and matching landing records
to exclude accepted tasks only in memory and seed fresh checkout proofs from the
pinned integration revision, never publishing premature base completion.
Planning runs these proofs before independent review and approval; repair must
audit every pending task's complete dependency, permission, setup, proof, review
and acceptance path and rerun all witnesses, not merely baseline compatibility.
Saved structural blockers are additionally checked against their actual preserved
implementation and approved gates in isolation, without accepting or landing it.
All saved failed tasks are replayed regardless of failure label, including tasks
that reached architect takeover or failed during independent review delivery.
During implementation, a developer's FAILED report is first put to one
read-only architect consultation, which returns the task to a fresh developer
with its direction when a developer path remains. Only when it finds none, or
repeated/deadline task gate failures or exhausted/unusable review delivery
occur, does the task transfer to one architect for in-scope resolution; changing requirements needs authorized
stopped-run repair, and gates plus fresh pinned independent review remain required.
Call implement with <number or docs/plans/NNNN-slug> explicitly after approval.
Plan-level depends_on prerequisites must be completed in both their STATUS.md
frontmatter and their docs/plans/STATUS.md roll-up row before implementation starts.

The host owns scheduling, isolated worktrees, branches, commits, gates, retries,
independent read-only reviews, task acceptance, landing OIDs, and final landing.
The host starts each developer, reviewer, and architect itself; developer
and reviewer roles request configured lightweight checks through verify,
not raw shell commands; architects may diagnose with broader checks.
Never substitute a worker's COMPLETE report for successful host proofs and a
fresh independent ACCEPT verdict with zero findings.

## Keep records truthful

Approved objectives, scope, architecture, task requirements, and Done whens are
immutable during execution: workers must not rewrite plan files to hide unfinished
work, weaken acceptance, or manually mark tasks accepted.
The host updates task status and merged_as from verified landing evidence, writes
run notes and plan status (todo, in_progress, completed, or failed), and maintains
the shared docs/plans/STATUS.md board; progress in an integration checkout can
legitimately precede published base-branch progress until final landing.
Only successful plan-level integration gates, independent review, and verified
landing justify completed; the plan's STATUS.md and roll-up must agree, or dependent
plans fail closed instead of guessing.

Call implement with plan=<plan> again to resume a stopped run using preserved host state;
use the repair tool with plan=<number or docs/plans/NNNN-slug> to recover a failed or stopped run's stale baseline from the published plan (it runs no independent review); a plan refused at admission is repaired automatically, with an independent review, never fabricate completion or hand-edit checkpoints to release dependencies.
Keep approved plans and their durable progress records tracked in Git; .makina/.gitignore
ignores all local .makina runtime state and settings, including factory.toml, and a run
adds those ignore rules and untracks a previously committed factory.toml without
changing its content; drafts and build caches normally live outside the repository.
