# docs/plans/

`docs/plans/` holds the project's implementation plans, one numbered folder per plan. This README is the authoring guide: it defines the folder/naming rule, the document triad every plan must contain, the per-task format the implement-plan workflow consumes, and the lifecycle from authored to merged. Follow it exactly — the conventions here are derived from the plans already in this folder, not invented.

Status tracking is two-tier: a folder-level **[STATUS.md](STATUS.md)** roll-up (one row per plan, beside this README) and a **per-plan `STATUS.md`** inside each plan folder (the task-level detail). Keep both current and in sync; see [STATUS.md](#statusmd) below.

## Folder & naming

One directory per plan, named `NNNN-Title-Case-Kebab`:

- `NNNN` — 4-digit zero-padded monotonic sequence number (`0001`, `0002`, ...). Not a date. Stable, orderable, mergeable, independent of authoring day.
- `Title-Case-Kebab` — the human title, Title-Cased, words joined by hyphens. Acronyms stay fully uppercased.

Real examples:

```
docs/plans/0001-Survival-Signal-Grounding/
docs/plans/0002-CPU-GPU-Runtime-Decoupling/
```

Three casing conventions coexist and must not be mixed:

- `NNNN-Title-Case-Kebab` — plan folders.
- `ALLCAPS.md` — the fixed, unnumbered core docs (`SCOPE.md`, `ARCHITECTURE.md`, `TASKS.md`) plus the per-plan `STATUS.md` tracker.
- `lower-kebab` — task ids inside `TASKS.md` (e.g. `sensory-tail-telemetry`). Each task id doubles as its git branch `task/{id}` and worktree directory `.makina/worktrees/{plan_slug}--{id}/`.

A `NNNN` decision doc may also live in the folder (see [Optional decision document](#optional-decision-document-nnnn-name-decisionmd)); it is `NNNN-NAME-DECISION.md` (number-prefixed, ALL-CAPS-kebab), numbered to the workstream it resolves — not given a fresh sequence number.

## STATUS.md

Execution status lives in two tiers so the top-level board stays small as plans accumulate:

- **Root [`STATUS.md`](STATUS.md)** — the roll-up. **One row per plan**, no per-task detail: status (📋 Planned / 🚧 In progress / ✅ Complete / ⛔ Blocked / 🗄️ Superseded), a tasks-done/total count, the one-line outcome, and a link to the plan's own `STATUS.md`. Answers "what is done, in flight, or blocked?" across all plans at a glance.
- **Per-plan `STATUS.md`** (inside each `NNNN-…/` folder) — the detail. The plan's status header, goal, measured outcome, and a per-workstream table marking each task's state. This is where task-level status grows, kept out of the root so the roll-up never bloats.

**A board is only as valuable as it is accurate, and a stale one is worse than none — it lies with authority.** Treat status as part of the change, not an afterthought:

- **Same change, every time.** Update status in the *same commit/PR* that moves reality — creating a plan, landing a task, merging a plan, or resolving a spike. A transition that does not touch STATUS.md is incomplete.
- **Keep the two in sync.** Per-task state changes in the plan's `STATUS.md`; whenever that flips the plan's overall status, tasks-done count, or outcome, update the matching root row in the same change. They must never disagree.
- **No drift from reality.** Status, counts, and outcome must match the plan's actual `TASKS.md` and the state of `develop`. If they disagree, the board is wrong — fix it.
- **Refresh the date.** Bump the `Last updated` line (and the branch it reflects) on every edit, in both files you touched.
- **Record terminal outcomes.** When a plan completes, capture the measured result; when a spike is rejected, point at its decision doc. The board is also the lightweight history of what shipped.

If you are unsure whether to update it, you should — an over-updated board costs a line; a wrong one costs trust.

## Plan structure

Every plan folder contains the same three documents. They form a hub-and-spoke triad threaded by a shared 4-digit **workstream** numbering (`0001`–`000N`): `SCOPE.md` defines the workstreams and maps findings onto them, `ARCHITECTURE.md` has a matching `## 000N` section of concrete deltas per workstream, and `TASKS.md` has a matching `## 000N` phase header per workstream holding its tasks.

| Document | Role |
|---|---|
| `SCOPE.md` | Boundaries and rationale: what the plan does and explicitly does not do, why, the workstream list, the finding→workstream mapping, and the locked design decisions with their gates. |
| `ARCHITECTURE.md` | The technical deltas: per workstream, what the code does today, the exact edits (constants/structs/signatures), and the safety/invariant argument that makes each edit correct. |
| `TASKS.md` | The execution checklist: dependency-ordered, independently-branchable tasks, each with a stable kebab id, prescriptive steps, a `Depends on` list, and a falsifiable `Done when` gate. The entry/index file the implement-plan skill consumes. |

Alongside these three authored docs, each plan folder also holds a living **`STATUS.md`** tracker — created with the plan and updated as tasks land (see [STATUS.md](#statusmd)). The three above are written once and rarely change; `STATUS.md` changes constantly.

The workstream number is a phase ordinal local to the plan, distinct from the plan number in the `NNNN` folder. Pick one typography (em-dash `—` / `→` or ASCII hyphen `-` / `->`) and keep it internally consistent across all three docs of a plan.

### SCOPE.md

Boundary-and-rationale contract. Coarse-grained — it lists workstreams, not tasks. Defers concrete edits to `ARCHITECTURE.md` and the executable task list to `TASKS.md` in its closing lines.

Fixed section order:

```markdown
# Scope — Plan NNNN

> One-sentence mission statement, set as a blockquote directly under the title, before any heading.

## Why this plan

Current state, then the diagnosed problems as a numbered list, each item
starting with a **bold lead-in.** Every load-bearing claim cites a precise
code location (`file.ext:line` or `file.ext:start-end`, symbol in backticks).

1. **First problem.** What is wrong and where (`kernel_tick.wgsl:383-391`).
2. **Second problem.** What is wrong and where (`governor.rs:473-479`).

(Optional, when adjudicating reviews: a falsification/provenance paragraph —
"verified against `develop` @ <sha>" — and a rejected-claims table so dead
claims are not re-litigated:)

**Review claims rejected during verification:**

| Claim | Source | Why rejected |
|---|---|---|
| ... | ... | ... |

## In scope

- **0001 — Workstream name.** What this workstream delivers. Keyed to its workstream id; see [TASKS.md](TASKS.md).
- **0002 — Workstream name.** ...

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| First problem (1) | `0001` |
| Second problem (2) | `0002` |

## Locked decisions

- **Decision title.** The commitment, its rationale, and the explicit gate or condition under which it holds or is revisited (data-gated, probe-regression-gated, measurement-gated).

## Out of scope

- **Excluded item.** Why it is excluded, or the gate that would unlock it later. Mirrors "In scope" to draw a hard boundary.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
```

Every numbered finding in "Why this plan" must appear in the origin→workstream table. State gate/success criteria as thresholds with decision rules (e.g. "ticks/sec cost under ~30% AND beats control on a fixed seed", "max 60 Hz publication"). Record negative results as the answer rather than re-running to confirm.

### ARCHITECTURE.md

Design and justification, organized by workstream. Describes the *deltas* against the current codebase — never a from-scratch design. Defers boundary/decision policy to `SCOPE.md` ("Decision rule in SCOPE (locked decisions).").

```markdown
# Architecture — Plan NNNN (deltas)

> Edits in `crates/xagent-brain/src/gpu_kernel.rs`, `crates/.../kernel_tick.wgsl`, ...
> (every touched file as a backtick repo-relative path).
> Line numbers are hints; locate by symbol.

## 0001 — Workstream name

Today `function_name` (`file.rs:1888-1903`) does X. (Each workstream/sub-design
opens with the word "Today" stating present behavior, anchored to file:line.)

### Named sub-design (optional H3)

Today ... .

Edits:

- **Bold-labeled edit** (`file.rs:193-210`): what changes, with a fenced block
  of the new constant/struct/signature carrying an in-place doc-comment.

​```rust
/// Why this value / what it means.
const TERMINAL_DEATH_TD_ERROR: f32 = -MAX_TD_ERROR;
​```

Properties that make this safe:
- `function_name` is thread-0-only with no barriers; barrier uniformity is untouched.
- L2-ball clamps / saturating arithmetic / ownership boundary holds because ... .

## 0002 — Workstream name

Today ... . Edits: ... .

(Optional trailing sections, present in 0001 only:)

## Test strategy

Named tests with file, setup, and exact assertions; ends with the CI gate
commands that must stay green.

## Interaction with prior work

Bold-led bullets tying the plan to prior specs/issues/reviews — what is honored, what is deferred.
```

Conventions: title always ends with the literal `(deltas)` suffix; constant names are full SCREAMING_SNAKE_CASE (no abbreviations); new code is shown as minimal fenced `rust`/`wgsl` snippets (the new constant/field/signature plus its doc-comment, never full function bodies); gated and not-yet-locked workstreams say so explicitly and may present labeled `Variant A` / `Variant B` alternatives with their gating conditions; use numbered ordered-list protocols and tables (sweep arms, experiment matrices) instead of ASCII diagrams; write formulas inline with unicode operators.

### TASKS.md

The execution checklist and the triad's entry/index file. Test-first and measurement-first: probe/baseline tasks are authored first and listed as dependencies of the change tasks that measure against them.

**Write it for a junior engineer who has never seen the codebase.** A task is correct only when an executor with zero prior context can complete it by following the steps literally — no guesswork, no improvisation, no design decisions left to them. Make every choice *here*, in the task: name the exact file and symbol, give the exact value with its rationale, pin the target signature/constant and the test verbatim (the test is the acceptance gate), and write a `Done when` the reviewer can mark pass/fail mechanically without interpreting intent. You are specifying the contract and the verification, not pre-writing the implementation — the executor still writes the body, but against rails tight enough that only one correct result fits. If a step would force the executor to ask "which file?", "what value?", or "how should this behave?", the task is underspecified — fix the task. Ambiguity is a defect in the plan, never a judgment call for the executor.

```markdown
# XAgent Plan NNNN — Plan Name

One short paragraph (3-8 lines) stating what the whole plan accomplishes,
as a run-on list of the concrete deltas in execution order.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" / "-" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (state as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Workstream name

### task-id — Task Title

1-2 prose paragraphs of current/broken behavior with precise code anchors
(`agent_death_respawn` zeroes the traces ... `kernel_tick.wgsl:383-391`) and
why the change is needed.

**Steps:**
1. Edit the exact file and symbol. Embed verbatim ```rust / ```wgsl / ```bash to paste.
2. Add the full test function here (GPU tests embed the self-skip guard verbatim).

- **Depends on:** —
- **Done when:** the falsifiable criterion (red-green for behavioral tasks: "the test fails before the change and passes after"); cargo fmt/clippy/test green.

---

## 0002 — Workstream name

### another-task-id — Another Task Title

...

---

**End of plan NNNN TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
```

#### Task format

Each task is one independently-branchable, independently-reviewable unit of work, sized to a single measurement or a single coherent code change plus its test. It must be self-contained and unambiguous: the executor should never need to open another plan doc, infer a missing value, or choose between approaches to finish it. Spell out the outcome so completion is binary.

- **Header:** `### {id} — {Title Case description}`. The `{id}` is lower-kebab-case and is reused verbatim as the git branch `task/{id}` and worktree `.makina/worktrees/{plan_slug}--{id}/`.
- **Context:** 1-2 prose paragraphs under the header, anchored to `symbol_name (file.ext:START-END)`. Line numbers are hints; the real locator is the symbol via grep.
- **`**Steps:**`** — an ordered (`1.`, `2.`, `3.`) imperative list. Prescriptive and copy-paste-ready: name the exact file and symbol, embed full verbatim fenced code (including complete test functions). Use a markdown table inside steps when a parameter matrix is clearer than prose. GPU-touching tests embed the standard guard verbatim:

  ```rust
  if !xagent_brain::GpuKernel::is_available() {
      eprintln!("Skipping: no GPU/fallback adapter available");
      return;
  }
  ```

- **`**Depends on:**`** — a bullet listing direct prerequisite task ids by their kebab id, comma-separated. `—` / `-` means no dependencies. Ordering is implied by workstream order plus these explicit id references; there is no separate DAG diagram.
- **`**Done when:**`** — a bullet giving the falsifiable acceptance criterion, almost always closing with "cargo fmt/clippy/test green". Behavioral tasks phrase it red-green ("the test fails before the change and passes after"). Tasks with no CPU-observable unit defer verification to a downstream re-measure probe and say so. A documentation-only task may exempt the green gate, but only if it explicitly says it is documentation-only.

The acceptance gate must cite the project's standing quality-gate commands. These are the canonical full forms, stated once in Conventions and abbreviated as "cargo fmt/clippy/test green" thereafter:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

**Status / gating markers:**

- A **GATED** task (a Phase-2 follow-up whose start is conditional) appends ` (GATED)` to its title, opens its context with a bold `**Gate:**` paragraph naming the upstream measurement tasks that must land first, and has a binary `Done when` ("either merged with all bands holding ... or reverted wholesale with the negative result recorded").
- A **spike / decision** task is framed as "Prototype only if ..." and its `Done when` is itself a decision (the note rejects the path with measured evidence, or a measured prototype justifies a follow-up plan). Its outcome is written up as the optional decision document below.

### Optional decision document (`NNNN-NAME-DECISION.md`)

Added inside a plan folder when a `TASKS.md` task is a spike/investigation whose `Done when` is a decision. Numbered to the workstream it resolves (e.g. `0005-SHARED-DEVICE-DECISION.md` resolves workstream `0005`), not given a fresh sequence number. Evidence-driven, leads with the decision, reversible by gate:

```markdown
> Restates the spike task and its "Done when:".

## Decision
The verdict up front (reject / accept) and which "Done when" branch it satisfies.

## The three paths
| Path | What it is | Cost it removes | Risk / cost to build |
|---|---|---|---|
| A | ... | ... | ... |

## Measured evidence
Raw instrumented numbers from live runs (e.g. RUST_LOG=debug).

## Why reject now
The reasoning.

## When to revisit
Concrete gating conditions that would reopen it; point at ARCHITECTURE.md §000N for the construction sketch.
```

## Lifecycle

1. **Create.** Author the numbered folder `docs/plans/NNNN-Title/` with the full triad. Make it workstream-numbered and measurement-first: every task carries a stable kebab id, a `Depends on` list, and a `Done when` that cites the gate commands. Probe/baseline tasks come first and gate the change tasks that measure against them. In the same change, create the plan's `STATUS.md` (status `📋 Planned`, the workstream/task table) and add its `📋 Planned` row to the root [STATUS.md](STATUS.md).
2. **Implement** (the `implement-plan` skill). One run creates a single integration branch for the whole plan off the base branch (`develop`). Then each task gets its own worktree + branch (`task/{id}` in `.makina/worktrees/{plan_slug}--{id}/`), respecting `Depends on` order so dependents build on already-merged prerequisites.
3. **Review.** Each task runs a two-role loop: a developer implements and self-gates, then an independent reviewer re-runs the project gates (`cargo fmt`/`clippy`/`cargo test -p xagent-sandbox`; GPU tests self-skip without an adapter, CI uses Mesa lavapipe) and reviews, looping until approved or capped, escalating the developer to a stronger model after a rejected round.
4. **Merge.** On approval a task branch merges into the plan branch. Once every task passes, the plan branch is squash-merged into `develop` as a single commit whose message lists what was done. A failed task leaves its worktree for inspection and blocks the final squash. Gated/spike tasks resolve into an in-folder decision doc rather than code. Keep both status files in step the whole way: mark each task done in the plan's `STATUS.md` as it lands, and reflect the plan-level status (`🚧 In progress` → `✅ Complete`), tasks-done count, and measured outcome in the root [STATUS.md](STATUS.md) row whenever those change.

Known gap: the per-task gate effectively runs only the changed crate's lib unit tests and can miss integration tests and doctests. After a plan lands, re-run from the repo root and fix/commit before the next plan, keeping `develop` always-green:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-fail-fast
```

New plans append their measured baselines / after-numbers back into the legacy baseline spec `docs/superpowers/specs/2026-06-10-learning-baseline.md` (a continuity bridge, not a replacement).

## Checklist for a new plan

- [ ] Folder named `NNNN-Title-Case-Kebab` with the next zero-padded sequence number.
- [ ] Status updated in the same change, both tiers in sync — the plan's own `STATUS.md` created on create and its tasks marked as they land, and the root [STATUS.md](STATUS.md) row kept current (status, tasks-done, outcome) with the `Last updated` line bumped.
- [ ] `SCOPE.md`: blockquote thesis; `## Why this plan` with numbered, code-cited findings; `## In scope` keyed to workstream ids; `## Origin -> workstream mapping` covering every finding; `## Locked decisions` with gates; `## Out of scope`; closing See-references to ARCHITECTURE and TASKS.
- [ ] `ARCHITECTURE.md`: title ends with `(deltas)`; opening blockquote file manifest ending "Line numbers are hints; locate by symbol."; one `## 000N` per workstream, each opening with "Today ..."; minimal fenced code with justifying doc-comments; safety/invariant argument per edit; decisions deferred to SCOPE.
- [ ] `TASKS.md`: title, summary paragraph, See-references line, `**Conventions**` block, `---` rules; one `## 000N` phase per workstream; each task with a kebab id header, context, `**Steps:**`, `**Depends on:**`, `**Done when:**`.
- [ ] Each task is junior-executable: exact files/symbols/values named, target signature and test pinned verbatim, zero design decisions or "which?/what?/how?" left open, and a `Done when` the reviewer can mark pass/fail mechanically — the contract is specified, not the whole implementation pre-written.
- [ ] Workstream ids `0001`–`000N` match one-to-one across all three docs.
- [ ] Probe/baseline tasks authored first and listed as `Depends on` of the changes they gate.
- [ ] Every `Done when` cites the quality gate (`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p xagent-sandbox`), except tasks explicitly marked documentation-only.
- [ ] GPU-touching tests embed the `GpuKernel::is_available()` self-skip guard.
- [ ] Gated tasks carry `(GATED)`, a `**Gate:**` precondition, and a binary land-or-revert-and-record `Done when`.
- [ ] Spike/decision tasks resolve into a `NNNN-NAME-DECISION.md` numbered to their workstream.
- [ ] Typography (em-dash/arrow vs ASCII) is internally consistent across the three docs.
