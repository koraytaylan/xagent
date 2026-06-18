# Reviewer Checklist

A focused pass over the [CONTRIBUTING.md](../CONTRIBUTING.md) rules that **no
automated gate can catch**. Run this on every change before approval — in the
`implement-plan` workflow this is the reviewer agent's job; for a human PR it is
the reviewer's.

## What is already machine-enforced — do not spend review effort here

These fail `cargo fmt` / `cargo clippy -D warnings` / `cargo test` automatically.
If the change is up for review, they already passed. Trust the gate:

- Formatting, pedantic-clippy lints, compile.
- **No `TODO`/`FIXME` markers** in `crates/**` — guarded by
  `tests/contributing_guard.rs::source_contains_no_pending_work_markers`.
- **No *new* planning references** (a plan/task/workstream/decision-doc/spec name)
  in `crates/**` source, comments, or runtime strings — ratcheted by
  `tests/contributing_guard.rs::planning_references_do_not_exceed_baseline`. The
  baseline only shrinks; you cannot add one without the test going red.

## What needs your eyes — the gate is blind to these

Check each. Most are judgment calls a linter cannot make.

### Naming & numbers
- [ ] No bare abbreviations or single-letter names outside closures/iterators/trivial math. Buffer-layout constants spell out descriptive words (`POS` → `POSITION`).
- [ ] Every numeric literal other than `0`/`1`/`-1`/`0.0`/`1.0` is a named `const` with a doc comment explaining *why* that value.

### Comments & docs match reality
- [ ] Doc/comment claims match the code **now** — especially contractual perf words ("non-blocking", "pre-allocates", "reuses allocation"). Renamed behavior → renamed function.
- [ ] A changed value/default/range has had its stale references grepped and updated (old literal *and* concept name).
- [ ] No commented-out code, tombstone comments, or "was previously…/will be replaced…" history in source.
- [ ] **Planning-agnostic, beyond the ratchet:** if you *touched* a file that still names a plan/task in a comment or string, clean those references in this change (file-by-file cleanup). The guard only blocks new ones; shrinking the debt is manual.
- [ ] No `eprintln!`/`println!` in library or long-lived runtime paths — use `log::warn!`/`error!`. (CLI entry points and tests are exempt; the guard cannot tell them apart, so **you** must.)
- [ ] No value written to a buffer or field that nothing reads.

### Numeric & GPU safety
- [ ] No lossy `as` integer conversions — `try_into()`/`TryFrom`, error handled. (The cast clippy lints are intentionally off; this is on you.)
- [ ] Buffer-size multiplications use `checked_mul()` or a validated bound.
- [ ] Every divisor guarded against zero (`max(value, epsilon)` or a check) — Rust **and** WGSL.
- [ ] Bounded quantities (energy, integrity) clamped in the *same scope* as the subtraction; death/invariant checks not deferred to next tick.
- [ ] Buffer offsets derive from `BrainLayout`/kernel config, never hardcoded strides; index/count validated before offset math.
- [ ] A Rust↔WGSL shared constant has a single canonical source (template *or* uniform, not both); uniform structs are 16-byte aligned.

### WGSL specifics
- [ ] `select()` arguments are all side-effect-safe (guard divisions with `max`, don't rely on the condition to skip them).
- [ ] Barriers correct: `storageBarrier()` + `workgroupBarrier()` where both visibility and execution sync are needed; early-return only when uniform across the workgroup and before the first barrier.
- [ ] Removed/renamed shared-header symbols grepped across every concatenating shader.

### Async readback & concurrency
- [ ] In-flight `map_async` tracked; both success and failure paths clean up and allow retry; staging buffers unmapped on success, error, agent switch, and reset.
- [ ] Data authority documented where two readbacks write the same field; stale async never overwrites fresher data.
- [ ] SQLite connections set `busy_timeout` (main and background). Background threads have a deterministic shutdown; `Drop` joins and logs panics.
- [ ] No `.expect()` for I/O, GPU, or thread operations — return `Result`/`Option`.

### State & performance
- [ ] Cause/effect temporal alignment: a gradient/reward recorded with a motor command reflects *that* command's outcome, not the previous tick's.
- [ ] State-machine transitions guard their preconditions (e.g. non-empty agent list) — no stuck states.
- [ ] Per-tick logic lives in WGSL, not Rust; no CPU work scales with `ticks_to_run`.
- [ ] No large-collection clones in per-frame hot paths (`clone_from()`/borrow); squared-distance (`_SQ`) for radius checks.

### Tests
- [ ] New tests are falsifiable — deleting the code under test breaks them; they assert invariants rather than filtering out violations and asserting on the remainder.
- [ ] Concurrency-sensitive paths have request→complete→consume *and* request→fail→cleanup coverage; cache paths have populate→invalidate→fresh coverage.
- [ ] Hardware-dependent (GPU) tests gated behind a feature/`#[cfg]`, not assumed present.
