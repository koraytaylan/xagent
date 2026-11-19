---
id: seeded-headless-digest
title: Capture the seeded headless evolution digest at baseline
workstream: "0100"
kind: task
role: architect
depends_on: []
gated: false
touches:
  - crates/xagent-sandbox/tests/seeded_headless_digest.rs
  - crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json
status: done
merged_as: "6615030e9b8b7b570baae8705cf68e73e003b94f"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --test seeded_headless_digest"
    timeout_ms: 3600000
  - name: digest-strict-capture
    command: "env XAGENT_REQUIRE_GPU=1 XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest -- --nocapture"
    timeout_ms: 3600000
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
---

# Capture the seeded headless evolution digest at baseline

## Outcome

- `crates/xagent-sandbox/tests/seeded_headless_digest.rs` runs a seeded
  headless evolution and reduces the stored result to a digest.
- `crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json` holds the
  digest captured from the unchanged baseline code at revision `0976448`,
  before any kernel change.

This is the reference that later kernel work must reproduce (INV-1).

## Decisions

- Run parameters: `FullConfig::default()` with `world.seed = SEED (5)`,
  `governor.tick_budget = TICK_BUDGET (2000)`,
  `governor.max_generations = GENERATIONS (2)`. Other defaults unchanged.
- Run with `xagent_sandbox::headless::run_headless(config, db_path, false,
  true)` into a file named `run.db` (the `*.db` pattern is git-ignored) inside a
  `tempfile::TempDir`. Open the finished database afterwards with
  `rusqlite::Connection::open_with_flags` and `OpenFlags::SQLITE_OPEN_READ_ONLY`,
  and call `busy_timeout` on it.
- Digest: FNV-1a 64-bit, implemented in the test. Named constants with `///`
  docs: `FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325`,
  `FNV_PRIME: u64 = 0x0000_0100_0000_01b3`. Hash input, in this order:
  1. a section tag, then every `node` row ordered by `id`, columns `id,
     parent_id, generation, config_json, best_fitness, avg_fitness,
     mutated_param, mutation_direction, status` (explicit list, in that
     order; `id` is hashed too);
  2. a section tag, then every `agent_result` row ordered by `id`, all columns
     except `id`, in schema column order (look up each column name with
     `Statement::column_name` and skip `id`);
  3. a section tag, then every `node_brain` row ordered by `node_id`:
     `node_id`, then the blob.
  Each value is hashed as: tag byte `0` for NULL; `1` + `i64` little-endian for
  INTEGER; `2` + `f64::to_bits` little-endian for REAL; `3` + `u64` length +
  UTF-8 bytes for TEXT; `4` + `u64` length + bytes for BLOB. Lengths are
  `u64` little-endian. Timestamps (`created_at`, `started_at`) and
  `wall_time_secs` are not read. The `run` table is not read.
- Output format: the digest is 16 lowercase hex digits, `{:016x}`.
- Modes, all inside a single `#[test]` function so that only one GPU kernel
  is created at a time:
  - Capture (`XAGENT_CAPTURE_SEEDED_DIGEST=1`): requires a GPU adapter (fail
    otherwise), runs once, prints `digest: <hex>` and `adapter: <description>`
    on stdout, and does not read the fixture.
  - Default (no capture or strict variable): if `GpuKernel::is_available()` is
    false, print `skipping seeded digest: no GPU adapter` and return, unless
    `XAGENT_REQUIRE_GPU=1` is set, in which case fail. Otherwise run twice into
    two temporary databases, assert the two digests are equal, parse the
    fixture with `serde_json`, and assert its `seed`, `generations`,
    `tick_budget` equal the constants and its `digest` is 16 lowercase hex
    digits.
  - Strict (`XAGENT_REQUIRE_SEEDED_DIGEST=1`, in addition to the default
    checks): assert the computed digest equals the fixture's `digest`; on
    mismatch the assertion message prints both digests.
- Adapter description: obtained in capture mode from
  `wgpu::Instance::new` with `Backends::all()`, `pollster::block_on` of
  `request_adapter`, then `get_info()` name and backend. The fixture's
  `adapter` field records that string from the capture output.
- Fixture JSON, exactly these keys: `{"seed":5,"generations":2,
  "tick_budget":2000,"digest":"<16 hex>","baseline_revision":"0976448",
  "adapter":"<from capture>"}`.
- No plan, task, workstream or decision-doc names in the test source; describe
  the test as "default kernel behaviour matches the recorded digest".
- Test helper functions and constants have `///` doc comments. Numeric literals
  other than 0 and 1 are named constants. No lossy `as` casts; use `try_into`
  or `to_bits`/`to_le_bytes`.

## Steps

This is an architect-owned task: step 2 needs shell for the baseline capture
(the capture `cargo test` runs in this task's own worktree), which the
architect route approves; no other step uses shell. No extra or sibling
worktree is created: this task has no prerequisites, so its working base
carries the unchanged baseline crate sources and the capture runs in place.

**Steps:**

1. Write `crates/xagent-sandbox/tests/seeded_headless_digest.rs` as described
   in Decisions: constants, the FNV-1a helper, the digest reducer, the
   `run_seeded_digest(db_path)` helper, and the single `#[test]`
   `default_kernel_matches_recorded_digest` with its three modes. Add a second
   CPU-only test `fnv1a_matches_reference_vectors` asserting the FNV-1a 64-bit
   values of the empty input (`0xcbf29ce484222325`) and of `b"a"`
   (`0xaf63dc4c8601ec8c`).
2. Capture in place on the working base, in this task's own worktree (no
   extra worktree, no copy of the test file elsewhere). First confirm with
   `git diff --quiet 0976448 -- crates Cargo.toml Cargo.lock ':!crates/xagent-sandbox/tests/seeded_headless_digest.rs'`
   (exit 0) that the only crate change from `0976448` is the new test file;
   if it exits non-zero, stop and report a blocker, since the working base no
   longer carries the baseline kernel. Then run
   `env XAGENT_CAPTURE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest -- --nocapture`
   twice in this worktree, redirecting each run to `$MAKINA_SHELL_JOB/output`.
   If the two printed digests differ, stop and report a blocker; do not write
   the fixture.
3. Write `crates/xagent-sandbox/tests/fixtures/seeded_headless_digest.json`
   from the captured digest, `baseline_revision` `"0976448"`, and the captured
   adapter string. Do not record worktree or temporary paths in the fixture or
   the source.
4. Run the `focused-proof` and `digest-strict-capture` checks in this worktree
   (host-run). If a failure is caused by the test code, fix the test code; do
   not change the fixture digest to match a new run without a new baseline
   capture.
5. Make `sandbox-clippy`, `sandbox-fmt` and `sandbox-guard` pass for the new
   test file, fixing lints and formatting (the `sandbox-fmt` check is
   auto-fixed by the host).

## Expected proof output

- `focused-proof`: `test result: ok. 2 passed; 0 failed` for the two tests
  above. On a host without a GPU adapter the default test prints the skip
  message and still passes, so the strict check is the decisive run.
- `digest-strict-capture`: `test result: ok. 2 passed; 0 failed`, with the
  printed digest equal to the fixture's `digest`.

- **Done when:** the `digest-strict-capture` check (`env XAGENT_REQUIRE_GPU=1 XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest -- --nocapture`) passes on the host GPU adapter against the fixture captured from revision `0976448`, which proves INV-1 at baseline (strict mode also runs the default `focused-proof` checks).

## Blockers to report, not work around

- Two capture runs at the baseline printing different digests: report the
  nondeterminism; no fixture is committed.
- No GPU adapter on the host: the strict check cannot run; report it.
