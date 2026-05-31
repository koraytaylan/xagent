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
- `mcp__github__request_copilot_review` (use for re-requests too; fall back to a GitHub UI re-request only if the call no-ops — see Step 7c)
- `mcp__github__add_reply_to_pull_request_comment`, `mcp__github__resolve_review_thread` (reply, then resolve the thread yourself; hand off to the user only if resolve errors — see Step 7c)
- `mcp__github__subscribe_pr_activity`, `mcp__github__unsubscribe_pr_activity`

Load any tool you haven't used yet with `ToolSearch` (`select:<tool>`) before calling it.

## Step 1 — Load issue context

1. `mcp__github__issue_read` the issue. Capture title, body, labels, assignees, linked items, and current state (open/closed).
2. Refuse politely if the issue is already closed unless the user explicitly asks to re-open the work.
3. Summarise the requested scope in one or two sentences. This is the definition of "done" for this skill run.

## Step 2 — Triage prior work

Before writing any code, figure out what has already happened for this issue.

1. **Merged PRs referencing the issue.** Use `mcp__github__search_pull_requests` (already scoped to `is:pr`), and pass `owner`/`repo` (or add `repo:koraytaylan/xagent` to the query) so the search is repo-scoped, not global. GitHub search has no `#<num>` qualifier — `#42` is matched as the literal text `42` — so query the bare number where it counts: `is:merged <num> in:title,body`. Scan the hits for `Closes #<num>`, `Fixes #<num>`, `Refs #<num>` in titles/bodies, and cross-check the issue's own timeline/linked PRs. Read each hit with `mcp__github__pull_request_read` and note what part of the scope is already delivered. Subtract that from the remaining work.
2. **Open PRs for this issue.** Use `mcp__github__list_pull_requests` (state: open) + `mcp__github__search_pull_requests` for `is:open <num> in:title,body` (repo-scoped as above). If one exists and looks like the same scope:
   - Continue on that PR instead of opening a new one.
   - `git fetch` its branch and check it out locally (`git fetch origin <head>` then `git switch <head>`).
   - Treat its existing commits and review comments as your starting point.
3. If multiple open PRs cover overlapping scope, stop and ask the user which to continue.

State the triage outcome to the user in 2–3 lines before proceeding.

## Step 3 — Branch setup

If continuing an existing open PR, skip to step 4.

**Honor a session-designated branch first.** Some harnesses (e.g. Claude Code on the web) pin the session to a specific branch and forbid pushing anywhere else. If you were given a designated branch for this session, work on and push to *that* branch — do not create an `issue-<num>-<slug>` branch, and skip the rest of this step. Only create a fresh branch when no branch is pinned (e.g. a local CLI run).

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

The gate above is your release gate whether or not it's wired as a git hook — do not push if `fmt`, `clippy`, or `test` fails. Fix the root cause and create a new commit (never `--no-verify`).

## Step 5 — Open (or update) the PR

1. Push with `git push -u origin <branch>` (the session-designated branch if one was pinned in Step 3, otherwise the `issue-<num>-<slug>` branch). Retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s) on network failure only.
2. If no open PR exists, create one with `mcp__github__create_pull_request` with `base: develop` and `head` set to the branch you just pushed (the session-designated branch, or the `issue-<num>-<slug>` branch) — both are required:
   - Title: conventional-commit style, < 70 chars.
   - Body: Summary (1–3 bullets), Test Plan (checklist), and `Closes #<num>` (or `Refs #<num>` if scope is partial).
3. Subscribe to PR activity: `mcp__github__subscribe_pr_activity` for the PR number. Tell the user you're now watching CI + review events. The subscription brings Claude back into the conversation when an event arrives — it does **not** let Claude sleep inside a single turn with a fallback timer. Step 6 and Step 7b describe how to reconcile this: poll on each turn, end the turn to wait, and rely on webhook arrivals (or `/loop`) to resume.

## Step 6 — CI babysitting (turn-based polling)

Claude runs in a turn-based loop. It cannot pause a turn for N seconds waiting for a webhook and then time out — between turns Claude isn't running, so any "timer" described in the skill would never fire on its own. Treat CI status like this:

1. **Same turn as the push.** Capture the head SHA (`git rev-parse HEAD`) and poll `mcp__github__pull_request_read` with method `get_check_runs` — CI here runs as GitHub Actions (= check runs), so `get_commit`'s legacy status field reads empty and must not be relied on. `get_status` is a fine secondary probe for any legacy commit statuses. Report the current state to the user in one line.
2. **End the turn** if CI is still pending. Claude will be resumed by:
   - a `check_run.completed` / `check_suite.completed` / `status` webhook event (via the subscription), or
   - a user ping, or
   - a `/loop` tick if the user has set one up.
3. **On every resume,** re-poll the check-runs and reconcile:
   - **No check runs returned yet** → checks haven't registered (common in the first window after a push/PR creation). Treat empty as *pending*, not success — report and end the turn again. Conclude "no CI" only after several resumes with a persistently empty list (and this repo *does* run CI on PRs to `develop`, so empty here means "not yet").
   - **All required checks `success`** → proceed to Step 7.
   - **Any `failure` / `cancelled` / `timed_out` / `action_required`** → fix loop below.
   - **Any `in_progress` / `queued` / `pending`** → report status and end the turn again.
4. **If the user wants guaranteed forward progress** while CI runs (e.g. webhooks are dropping), offer `/loop 2m keep working on PR #<num>` so the harness brings Claude back on a timer. The skill itself has no way to self-schedule.
5. **Authority rule.** The API poll is authoritative; webhook payloads are hints. If they disagree, trust the poll.

On failure:

1. Read the failing check's details/log URL from the `get_check_runs` payload; fetch the log if needed.
2. Reproduce locally when possible (`cargo fmt` / `clippy` / `test`).
3. Fix the root cause — never `--no-verify`, never disable a failing test, never mask a clippy lint without a justified `#[allow]` + comment.
4. Commit (`fix:` or `chore:` prefix), push, and return to step 1 with the new head SHA. Repeat until green.

## Step 7 — Copilot review loop

### 7a. Pre-review sync (run before **every** `request_copilot_review` call)

Copilot reviews the PR diff against `develop`. A dirty merge state produces noisy or misleading reviews, so sync first:

1. `git fetch origin develop`.
2. Check mergeability — either inspect the PR (`mcp__github__pull_request_read` → `mergeable`/`mergeable_state`) or attempt a local merge dry-run (`git merge --no-commit --no-ff origin/develop` then `git merge --abort`). GitHub computes `mergeable` asynchronously, so right after a push it can be `null`/`unknown`; re-poll until it settles rather than acting on `unknown`. The local dry-run sidesteps the race and is the more reliable signal.
3. If the branch is behind but clean, fast-forward by rebasing or via `mcp__github__update_pull_request_branch`. `update_pull_request_branch` commits the merge on the *remote* head, leaving your local behind — immediately `git fetch && git pull --ff-only` (or `git reset --hard @{u}`) before any further local work, or your next push is rejected non-fast-forward. If you rebased locally instead, just `git push`.
4. If there are conflicts, resolve them locally:
   - `git merge origin/develop` (preferred over rebase on a shared PR branch — preserves review anchors and avoids force-push).
   - Open each conflicted file, reconcile by hand. Prefer semantic merges over textual ones: re-apply the intent of both sides, don't just accept one.
   - Re-run the full local gate after resolution: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p xagent-sandbox`.
   - Commit with a `chore: merge develop into <branch>` (or `fix:` if the resolution involved behavior changes) — never amend an existing commit.
   - `git push`. If the remote rejected because someone else pushed meanwhile, `git pull --rebase` and retry.
5. Wait for CI on the merge commit per Step 6 (poll on each turn; resume on webhook or `/loop` tick) until it goes green. Loop back to Step 6's fix path if it fails.

Only then call `mcp__github__request_copilot_review`. Use it for re-requests after later pushes too; if a re-request silently no-ops (no new review after several resumes), fall back to asking the user to re-request from the GitHub UI (see step 7c).

### 7b. Wait for the review (turn-based polling)

After calling `mcp__github__request_copilot_review`:

1. Tell the user the review is requested and end the turn. As with CI, Claude can't sleep inside a turn — it will be resumed by a `pull_request_review` / `pull_request_review_comment` webhook event, a user ping, or a `/loop` tick.
2. **On every resume,** poll `mcp__github__pull_request_read` (methods `get_reviews` and `get_review_comments`) and check whether Copilot has posted a review newer than the last request. If yes, proceed to 7c with the API data.
3. If several resumes pass without a Copilot review, surface to the user — Copilot may be disabled, rate-limited, or the request may have silently failed. Try `mcp__github__request_copilot_review` once more; if that still yields nothing, ask the user to re-request from the GitHub UI, and suggest `/loop 2m …` if they want timed polling.
4. **Authority rule.** The API is authoritative for "has Copilot reviewed yet" — a missing webhook does not mean no review exists.

### 7c. Processing review comments

`get_review_comments` is cursor-paginated — page through all results (bump `perPage`, follow `after` with the `endCursor` from each page's `PageInfo`) before deciding you've seen every thread. A single page can hide later threads, which would skip feedback here and break the "every thread resolved" gate in Step 8.

For each review comment (from the webhook payload or the poll result):

1. Read the comment and the referenced code carefully. Re-check against CONTRIBUTING.md — Copilot may be wrong about project-specific rules.
2. Decide one of three outcomes and do the outcome-specific work:
   - **Apply the fix.** Make the change, verify locally, commit, push.
   - **Decline with reason.** The suggestion conflicts with CONTRIBUTING.md, is wrong, or is a stylistic preference you disagree with.
   - **Out of scope.** The fix is valid but exceeds the scope defined in step 1 — open a follow-up issue via `mcp__github__issue_write` (title + body describing the deferred work, labeled as tracked from this PR). Link it with `mcp__github__sub_issue_write` if appropriate.
3. **Reply on the comment** with `mcp__github__add_reply_to_pull_request_comment`. Note the two different IDs in the `get_review_comments` payload: each *thread* carries a GraphQL node `threadId` (used to resolve, step 4), while each *comment inside it* carries a numeric `id` — pass that numeric comment `id` as `commentId` here (it's a `number`, not the node string). Reply to the first comment in the thread. Replies must be concrete, cite files/lines, avoid filler:
   - Apply → what you changed and the commit SHA.
   - Decline → concrete rationale citing the CONTRIBUTING.md section or the code invariant.
   - Out of scope → the new issue number and a one-line reason it's deferred.
4. **Resolve the thread** with `mcp__github__resolve_review_thread`, passing the thread's GraphQL `threadId` (the node-string thread id, *not* the numeric comment id from step 3). Resolve only after you've replied and the outcome is settled (fix pushed, declined-with-reason, or deferred to a follow-up). If the resolve call errors, fall back to the user hand-off below for the threads that wouldn't close.

**Idempotent across resumes.** Because the skill ends turns and re-polls on resume, key your work by `threadId`. Don't rely on `isResolved` alone as the reply ledger — you reply *before* you resolve, so a thread you've already answered can still read `isResolved: false` if the turn ended (or the resolve call errored) in between, and re-replying would double-post. Before replying, check the thread's comments for a reply you already authored; skip the reply if one exists and just (re)attempt the resolve. Keep a running summary comment on the PR as the durable ledger — a table of every thread processed: thread id / subject, outcome (apply/decline/out-of-scope), and commit SHA or follow-up issue number. Mark any thread the auto-resolve couldn't close with **"please resolve"** so the user can finish it from the GitHub UI. Merge (Step 8) waits until every thread reports `isResolved: true`.

After **every** push that addresses review feedback, repeat step 7a (sync + conflict check), then call `mcp__github__request_copilot_review` again so Copilot re-reads the updated diff. If the re-request no-ops (no new review arrives after several resumes), ask the user to re-request from the GitHub UI ("Reviewers → Copilot → Re-request review") and wait for confirmation before re-entering step 7b.

**No-change exit:** if an entire review cycle completes without any "apply the fix" outcomes — every comment was declined with reason or deferred to a follow-up issue, and no new commits were pushed — do **not** re-request a review. The diff Copilot reviewed hasn't changed, so a re-review would produce the same comments. Those outcomes are settled, so resolve their threads yourself (Step 7c step 4); only the threads `resolve_review_thread` couldn't close get handed to the user. Then go to Step 8.

Loop until either Copilot's next review emits zero new comments, or a cycle ends with zero code changes.

## Step 8 — Merge

Preconditions before merge:

- CI is green on the latest commit.
- All processed review threads are resolved. Verify via `mcp__github__pull_request_read` method `get_review_comments` — paging through every page (Step 7c) — that every thread reports `isResolved: true`. Resolve any you handled but left open with `mcp__github__resolve_review_thread` (Step 7c); only ask the user to resolve threads the call wouldn't close. If any remain open, wait before proceeding.
- Branch is up to date with `develop` and free of merge conflicts. If behind, sync via step 7a (`update_pull_request_branch` if clean — then `git pull --ff-only` locally so your tree matches the new remote head — otherwise merge locally and resolve). Never rely on GitHub's web "resolve conflicts" flow — do it locally so the gate runs.

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
- Resolve review threads with `resolve_review_thread` once their outcome is settled; only hand a thread off to the user if the resolve call errors (Step 7c).
- Re-request Copilot reviews with `request_copilot_review` after each feedback push; only ask the user to re-request from the GitHub UI if the call no-ops (Step 7c).
- Never merge with red CI or unresolved threads. If threads are still open, ask the user to resolve them and wait.
- Never request a Copilot review while the PR has merge conflicts — resolve them first (step 7a).
- Never invent file paths, line numbers, or commit SHAs — read them first.
