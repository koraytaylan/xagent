---
id: archive-brain-source
title: "Free-life archive brain source"
workstream: "0500"
kind: task
role: developer
depends_on: [generation-run-summary, supervisor-verdict]
gated: false
touches: [crates/xagent-sandbox/src/headless.rs, crates/xagent-sandbox/tests/archive_free_life.rs, crates/xagent-sandbox/src/bin/pathway_route_gate.rs, crates/xagent-sandbox/src/bin/supervisor_verdict.rs]
status: done
merged_as: "6e7a7603731eededb4058a70945416a48b2088df"
checks:
  - {name: focused-proof, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test archive_free_life", timeout_ms: 3600000}
  - {name: headless-lib-tests, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --lib headless::tests::", timeout_ms: 3600000}
  - {name: seeded-digest-determinism, command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test seeded_headless_digest", timeout_ms: 3600000}
  - {name: sandbox-clippy, command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings", timeout_ms: 3600000}
  - {name: sandbox-fmt, command: "cargo fmt -p xagent-sandbox -- --check", fix: "cargo fmt -p xagent-sandbox"}
  - {name: sandbox-guard, command: "cargo test -p xagent-sandbox --test contributing_guard"}
---

# Free-life archive brain source

## Outcome

- `BrainSource::Archive { db_path: String, rank: usize }` is a free-life brain
  source. `run_free_life` loads it with
  `supervisor_store::load_archive_entry_read_only(db_path, rank)` and uses the
  entry's config as the brain config.
- `FreeLifeReport` gains `#[serde(default)] pub archive_entry_id: Option<i64>`,
  `Some` only for the archive source. Its `brain_source` is `"archive"`; champion
  fields stay `None`.

## Scope

- `headless.rs`: the variant, the archive branch of steps 2-3 of `run_free_life`,
  the two new `FreeLifeError` variants, and the report field.
- `src/bin/pathway_route_gate.rs` and `src/bin/supervisor_verdict.rs`: add
  `archive_entry_id: None` to their full `FreeLifeReport` literals, the only
  change the new field forces there; nothing else in those bins changes.
- `tests/archive_free_life.rs` (new): a fail-closed GPU helper copied from
  `gpu_adapter_present_when_required`, named consts for a small
  `SupervisorConfig`, and the tests below.

## Non-goals

- No CLI flags (`main.rs`, evolution-cli). No change to the supervisor or
  generation drivers, metrics, lesions or `governor.rs`.

## Prerequisites

- generation-run-summary (`headless.rs` ordering), which brings
  headless-supervisor-driver, supervisor-run and supervisor-store.
- Plan 0001: `run_free_life`, `FreeLifeOptions`, `BrainSource`, `FreeLifeReport`,
  `FreeLifeError`, and the free-life unit tests in `headless::tests`.
- From 0203: `load_archive_entry_read_only`, `StoredArchiveEntry` (with its entry
  id), `ArchiveLoadError { NoSupervisorRun, RankOutOfRange, MalformedBlob,
  MalformedConfig, Sqlite }`. A mismatch is a contradiction to report.

## Decisions

- Archive branch: load the entry; parse its config as the brain config; if the
  birth brain length differs from `BrainLayout::from_config(&config).brain_stride`,
  return a new `FreeLifeError` variant for archive layout, using the same
  pattern as the champion source's layout check. A load failure becomes
  `FreeLifeError::ArchiveLoad(ArchiveLoadError)`.
- The entry's birth brain is written unmutated to every agent, exactly as the
  champion source does. Nothing else in `run_free_life` changes.
- The archive load uses the read-only loader only; no write statement touches the
  database.
- `archive_entry_id` is set from the loaded entry; all other report fields follow
  the existing fresh/champion rules.
- Test DB: a short supervisor run through `run_headless_with_options` with a small
  `SupervisorConfig` (named consts) so the archive holds at least two entries.
- Strict serde: JSON without `archive_entry_id` deserialises to `None`.

## Steps

**Steps:**
1. Add `BrainSource::Archive { db_path, rank }`, the report field with
   `#[serde(default)]`, and the two `FreeLifeError` variants with Display.
2. Implement the archive branch of `run_free_life` steps 2-3 as in Decisions,
   leaving every other statement in the function unchanged.
3. Create `tests/archive_free_life.rs` with the GPU helper, consts, and a helper
   that builds the archive DB through `run_headless_with_options`.
4. Add the tests: `gpu_guard_fails_closed_when_required` (CPU);
   `archive_free_life_runs_rank_zero_entry` (`brain_source == "archive"`,
   `archive_entry_id` equals the rank-0 entry id from the loader, agent count and
   ticks as requested, champion fields `None`);
   `archive_free_life_leaves_database_bytes_unchanged` (hash of the main DB file
   before and after); `archive_free_life_rank_beyond_archive_is_error`;
   `archive_free_life_is_reproducible` (same options twice give equal reports);
   `free_life_report_without_archive_field_deserializes` (CPU).

- **Done when:** the focused-proof check (all six tests in `archive_free_life`
  pass), the headless-lib-tests check (0001's free-life unit tests still pass)
  and the seeded-digest-determinism check pass on the host, proving INV-11 (read-only
  archive loading leaves the DB unchanged and free lives run archive entries),
  INV-01 (the seeded digest is deterministic, and the generation path is
  untouched because this task changes only free-life code) and INV-15 (the GPU guard panics
  when required and no adapter exists).

## Evidence and risks

- The worker has no shell; the host-run checks in the frontmatter are the proof.
- Risk: the rank-0 entry must match the loader's ordering (`outcome DESC,
  entry_id ASC`); the test compares against the loader, not a hard-coded id.
- Risk: if the archive holds fewer than two entries after the short run, the
  test setup is wrong; report the measured count rather than lowering the
  requirement.
- Risk: the seeded digest depends on the champion branch staying byte-identical.
