---
name: work-on-issue
description: Use this skill when the user asks Claude Code to work on a GitHub issue by number or URL (e.g. "work on issue 42", "work on https://github.com/koraytaylan/xagent/issues/42", "/work-on-issue 42"). Drives the full lifecycle — fetch issue, triage prior/open PRs, implement changes per CONTRIBUTING.md, open a PR, handle CI failures, iterate with Copilot review, merge to develop, and close the issue.
---

# Work on a GitHub Issue

End-to-end workflow that takes a single input — an issue number or URL — and carries the work through review and merge.

## Argument parsing

The user supplies either a bare issue number (`42`) or a full URL (`https://github.com/koraytaylan/xagent/issues/42`). Parse out the issue number. The repository is always `koraytaylan/xagent` (the only repo your GitHub MCP tools are scoped to). Reject any URL pointing at a different repo.

## Required toolset

Use the GitHub MCP tools (prefix `mcp__github__`) for every GitHub interaction. You do not have `gh`/`hub`/REST API. Relevant tools:

- `mcp__github__issue_read`, `mcp__github__list_issues`, `mcp__github__search_issues`, `mcp__github__issue_write`, `mcp__github__add_issue_comment`, `mcp__github__sub_issue_write`
- `mcp__github__list_pull_requests`, `mcp__github__search_pull_requests`, `mcp__github__pull_request_read`, `mcp__github__create_pull_request`, `mcp__github__update_pull_request`, `mcp__github__update_pull_request_branch`, `mcp__github__merge_pull_request`
- `mcp__github__list_commits`, `mcp__github__get_commit`, `mcp__github__get_file_contents`
- `mcp__github__request_copilot_review`
- `mcp__github__add_reply_to_pull_request_comment`, `mcp__github__resolve_review_thread`, `mcp__github__unresolve_review_thread`
- `mcp__github__subscribe_pr_activity`, `mcp__github__unsubscribe_pr_activity`

Load any tool you haven't used yet with `ToolSearch` (`select:<tool>`) before calling it.

## Step 1 — Load issue context

1. `mcp__github__issue_read` the issue. Capture title, body, labels, assignees, linked items, and current state (open/closed).
2. Refuse politely if the issue is already closed unless the user explicitly asks to re-open the work.
3. Summarise the requested scope in one or two sentences. This is the definition of "done" for this skill run.

## Step 2 — Triage prior work

Before writing any code, figure out what has already happened for this issue.

1. **Merged PRs referencing the issue.** Use `mcp__github__search_pull_requests` with queries like `is:pr is:merged #<num>` and `is:pr is:merged <num> in:body`. Also scan for `Closes #<num>`, `Fixes #<num>`, `Refs #<num>` in titles/bodies. Read each hit with `mcp__github__pull_request_read` and note what part of the scope is already delivered. Subtract that from the remaining work.
2. **Open PRs for this issue.** Use `mcp__github__list_pull_requests` (state: open) + `mcp__github__search_pull_requests` for `is:pr is:open #<num>`. If one exists and looks like the same scope:
   - Continue on that PR instead of opening a new one.
   - `git fetch` its branch and check it out locally (`git fetch origin <head>` then `git switch <head>`).
   - Treat its existing commits and review comments as your starting point.
3. If multiple open PRs cover overlapping scope, stop and ask the user which to continue.

State the triage outcome to the user in 2–3 lines before proceeding.

## Step 3 — Branch setup

If continuing an existing open PR, skip to step 4.

Otherwise create a fresh branch from `develop`:

```bash
git fetch origin develop
git switch -c <type>/issue-<num>-<kebab-slug> origin/develop
```

Use conventional `<type>` (`feat`, `fix`, `perf`, `refactor`, `doc`, `chore`) matching the issue.

## Step 4 — Implement per CONTRIBUTING.md

`CONTRIBUTING.md` is the single source of truth. Before and during the change, re-read the sections that apply to the files you're touching. Non-negotiable rules to check on every commit:

- **Tooling gate** — `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p xagent-sandbox` must pass locally before push.
- **Naming** — no single-letter vars (outside `x`/`y`/closures/iterators), no bare abbreviations, spell out names.
- **Magic numbers** — any literal that isn't `0`, `1`, `-1`, `0.0`, `1.0` becomes a documented `const`.
- **Function length** — split functions > ~50 lines; flag anything > 100.
- **Docs** — public items get `///`; docstrings must match behavior; remove stale comments/TODOs/commented-out code.
- **Numeric safety** — no lossy `as` casts, use `try_into`; guard divisors; `checked_mul` for size arithmetic; clamp invariants in the same scope.
- **GPU/WGSL** — buffer offsets from `BrainLayout`/config, not hardcoded; `#[repr(C)]` + 16-byte alignment for uniform structs; guard `select()` divisions with `max(denom, eps)`; uniform early-returns only.
- **Async readback** — track in-flight state; unmap on all paths; document data authority.
- **State invariants** — re-establish invariants in the same scope after mutating energy/integrity.
- **Concurrency** — SQLite `busy_timeout`; deterministic thread shutdown; no `.expect()` on I/O/GPU/thread paths.
- **Performance** — per-tick logic stays in WGSL; no deep clones in hot paths; squared-distance comparisons.
- **Serialization** — `#[serde(alias = "old_name")]` or migration when renaming fields; document endianness for binary formats.
- **Testing** — falsifiable tests, invariant assertions, no GPU-gated tests without `#[cfg]`.
- **Logging** — `log::warn!`/`log::error!` in library code, not `eprintln!`/`println!`.
- **Commits** — conventional prefixes (`feat:`, `fix:`, `perf:`, `refactor:`, `doc:`, `chore:`).

Run the full local gate before every push:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-sandbox
```

Do not push on hook failure. Fix the root cause and create a new commit.

## Step 5 — Open (or update) the PR

1. Push with `git push -u origin <branch>`. Retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s) on network failure only.
2. If no open PR exists, create one with `mcp__github__create_pull_request` targeting `develop`:
   - Title: conventional-commit style, < 70 chars.
   - Body: Summary (1–3 bullets), Test Plan (checklist), and `Closes #<num>` (or `Refs #<num>` if scope is partial).
3. Subscribe to PR activity: `mcp__github__subscribe_pr_activity` for the PR number. Tell the user you're now watching review events. **Do not rely on the subscription for CI status** — webhook events are lossy (e.g. PR #132 completed CI with zero events delivered). Treat any CI event that does arrive as a hint, but always confirm with an active poll (Step 6).

## Step 6 — CI babysitting (active polling, not event-driven)

After every push, actively poll the head commit's check runs until CI concludes. Do not wait passively for webhooks.

1. Capture the pushed SHA: `git rev-parse HEAD`.
2. Poll `mcp__github__get_commit` for that SHA (or `mcp__github__pull_request_read` for the PR — whichever exposes check-run/status data in this MCP server). Inspect every check run / status context.
3. Poll interval: start at 30s, back off to 60s after 3 min, cap at 2 min. Give up and surface the state to the user after ~20 min of `in_progress`/`queued` so the user can intervene.
4. Classify the aggregate state:
   - **All required checks `success`** → CI is green, proceed to Step 7.
   - **Any `failure`/`cancelled`/`timed_out`/`action_required`** → fix loop below.
   - **Any `in_progress`/`queued`/`pending`** → keep polling.
5. On failure:
   1. Read the failing check's log URL from the `get_commit` payload; fetch the log (or use `mcp__github__list_commits` → `get_commit` to drill into the check run if needed).
   2. Reproduce locally when possible (`cargo fmt`/`clippy`/`test`).
   3. Fix the root cause — never `--no-verify`, never disable a failing test, never mask a clippy lint without a justified `#[allow]` + comment.
   4. Commit (`fix:` or `chore:` prefix), push, and go back to step 1 with the new head SHA. Repeat until green.
6. If a webhook CI event does arrive while you're polling, fine — but never *skip* a poll because the webhook said something. The poll is authoritative.

## Step 7 — Copilot review loop

### 7a. Pre-review sync (run before **every** `request_copilot_review` call)

Copilot reviews the PR diff against `develop`. A dirty merge state produces noisy or misleading reviews, so sync first:

1. `git fetch origin develop`.
2. Check mergeability — either inspect the PR (`mcp__github__pull_request_read` → `mergeable`/`mergeable_state`) or attempt a local merge dry-run (`git merge --no-commit --no-ff origin/develop` then `git merge --abort`).
3. If the branch is behind but clean, fast-forward by rebasing or using `mcp__github__update_pull_request_branch`, then `git push`.
4. If there are conflicts, resolve them locally:
   - `git merge origin/develop` (preferred over rebase on a shared PR branch — preserves review anchors and avoids force-push).
   - Open each conflicted file, reconcile by hand. Prefer semantic merges over textual ones: re-apply the intent of both sides, don't just accept one.
   - Re-run the full local gate after resolution: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p xagent-sandbox`.
   - Commit with a `chore: merge develop into <branch>` (or `fix:` if the resolution involved behavior changes) — never amend an existing commit.
   - `git push`. If the remote rejected because someone else pushed meanwhile, `git pull --rebase` and retry.
5. Actively poll CI on the merge commit per Step 6 until it goes green. Do not assume success from silence — the webhook subscription is not reliable for CI status. Loop back to Step 6's fix path if it fails.

Only then call `mcp__github__request_copilot_review`.

### 7b. Processing review comments

For each review comment that arrives:

1. Read the comment and the referenced code carefully. Re-check against CONTRIBUTING.md — Copilot may be wrong about project-specific rules.
2. Decide one of three outcomes:
   - **Apply the fix.** Make the change, verify locally, commit, push. Reply on the comment explaining what you changed. Resolve the thread with `mcp__github__resolve_review_thread` once the push lands.
   - **Decline with reason.** If the suggestion conflicts with CONTRIBUTING.md, is wrong, or is a stylistic preference you disagree with, reply with a concrete rationale (cite the CONTRIBUTING.md section or the code invariant). Resolve the thread.
   - **Out of scope.** If a fix is valid but exceeds the scope defined in step 1, open a follow-up issue via `mcp__github__issue_write` (title + body describing the deferred work, labeled as tracked from this PR). Link it with `mcp__github__sub_issue_write` if appropriate. Reply on the comment with the new issue number and a short explanation. Resolve the thread.
3. Use `mcp__github__add_reply_to_pull_request_comment` for replies — keep them concrete, cite files/lines, avoid filler.

After **every** push that addresses review feedback, repeat step 7a (sync + conflict check) before calling `mcp__github__request_copilot_review` again so Copilot re-reads the updated diff against a clean merge.

Loop until Copilot's next review emits zero new comments.

## Step 8 — Merge

Preconditions before merge:

- CI is green on the latest commit.
- No unresolved review threads.
- Branch is up to date with `develop` and free of merge conflicts. If behind, sync via step 7a (`update_pull_request_branch` if clean, otherwise merge locally and resolve). Never rely on GitHub's web "resolve conflicts" flow — do it locally so the gate runs.

Merge with `mcp__github__merge_pull_request`, `merge_method: "merge"` (creates a merge commit, as requested). Use the PR title as the merge commit title.

After merge:

- `mcp__github__unsubscribe_pr_activity` for this PR.
- Locally `git switch develop && git pull origin develop` so the working tree tracks reality.

## Step 9 — Close the loop on the issue

Compose a single comment on the issue summarising:

- What shipped (PR link + one-line summary per notable change).
- What's deferred (links to any follow-up issues created in step 7).
- Whether all scope is covered.

Post with `mcp__github__add_issue_comment`. Then:

- **All scope covered** → close the issue with `mcp__github__issue_write` (state: closed).
- **Scope partially covered** → leave the issue open and make sure the comment explains what remains and references the follow-ups.

## Reporting back to the user

End your turn with a 1–2 sentence summary: PR URL, merge status, and issue state (closed/open-with-remaining-scope). If you stopped mid-workflow for a blocker, say exactly what you need from the user to continue.

## Non-negotiables

- Never force-push to `main`/`develop`. Never push to a branch other than the one this PR lives on.
- Never skip hooks (`--no-verify`, `--no-gpg-sign`) unless the user explicitly asks.
- Never mark a thread resolved without either applying the change, declining with a reason, or filing a follow-up.
- Never merge with red CI or unresolved threads.
- Never request a Copilot review while the PR has merge conflicts — resolve them first (step 7a).
- Never invent file paths, line numbers, or commit SHAs — read them first.
