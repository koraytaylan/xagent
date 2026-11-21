---
id: resume-digest-gate
title: Stop-and-resume digest and refusal gate
workstream: "0400"
kind: task
role: developer
depends_on:
  - headless-resume
gated: false
touches:
  - crates/xagent-sandbox/tests/supervisor_resume_digest.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test supervisor_resume_digest
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: rustfmt --edition 2021 --check crates/xagent-sandbox/tests/supervisor_resume_digest.rs
    fix: rustfmt --edition 2021 crates/xagent-sandbox/tests/supervisor_resume_digest.rs
---

# Stop-and-resume digest and refusal gate

## Outcome

A new test file `crates/xagent-sandbox/tests/supervisor_resume_digest.rs` proves
INV-1 and INV-2 through the headless driver.

- INV-1: the uninterrupted run, which writes checkpoints, has the plan 1 supervisor
  digest equal to plan 0006's pinned golden for the same seeded config under 0006's
  adapter rule. A run stopped after a checkpoint and resumed has the same digest as
  the uninterrupted run.
- INV-2: truncated, byte-flipped, version-mismatched and config-mismatched
  checkpoints are refused with the matching `CheckpointError`, and the database is
  left unchanged.

Only this test file is added; no product code changes.

## Prerequisites used (not reimplemented)

- From `headless-resume`: `xagent_sandbox::headless::{run_headless_with_options,
  HeadlessError, HeadlessOptions}`; `HeadlessOptions.stop_after_checkpoint_at_or_after:
  Option<u64>`; `HeadlessError::Checkpoint(CheckpointError)`.
- From `checkpoint-envelope`: `xagent_sandbox::supervisor_checkpoint::CheckpointError`
  (`PartialEq`, `Display` containing `checkpoint`), variants `Truncated`,
  `ChecksumMismatch`, `VersionMismatch { found, expected }`, `ConfigMismatch`.
- From `checkpoint-store`: table `supervisor_checkpoint (run_id, tick,
  format_version, payload BLOB)`, one row per run.
- From landed 0006: `crates/xagent-sandbox/tests/supervisor_digest.rs` and fixture
  `crates/xagent-sandbox/tests/fixtures/supervisor_digest.json` (keys `seed`,
  `agent_tick_budget`, `inheritance`, `digest`, `adapter`). Read its `Fnv1a`,
  `hash_section`, `digest_database`, `adapter_description`, `env_flag`,
  `is_digest_hex` and seeded config from the landed file, not from memory.
- `GpuKernel::is_available()`, and `rusqlite`, `serde_json`, `tempfile`, `wgpu`,
  `pollster`, as already used by `tests/seeded_headless_digest.rs` and
  `tests/supervisor_headless.rs`.

## Decisions (explicit)

- **Digest copied, not exported.** Copy `Fnv1a` (with `write_value` and value tags),
  `fnv1a`, `hash_section`, `digest_database` verbatim from 0006's landed gate, with a
  comment naming `crates/xagent-sandbox/tests/supervisor_digest.rs`. Keep its SQL,
  skipped columns and table list.
- **Config.** Copy 0006's landed `seeded_config()`: `FullConfig::default()` with
  `world.seed = 5`, `governor.evolution_mode = EvolutionMode::Supervisor`,
  `governor.population_size = 10`, `governor.supervisor.agent_tick_budget = 600_000`;
  other supervisor settings stay default. If the landed 0006 config differs, use it.
- **Stop point.** `STOP_AT_TICK = 10_000`. Run B sets
  `stop_after_checkpoint_at_or_after: Some(STOP_AT_TICK)` and stops at the first
  checkpoint with tick `T >= STOP_AT_TICK`; then it resumes with `resume = true`.
- **Golden (no disable switch; checkpoints are always written).**
  `resume_digest_checkpointing_does_not_change_digest` runs the uninterrupted run A
  with a store and reads the fixture. It asserts `seed == 5`,
  `agent_tick_budget == 600000`, `inheritance == "weight_copy"`. As 0006's gate does,
  it asserts digest(A) equals the fixture `digest` only when `adapter_description()`
  equals the fixture `adapter` or `XAGENT_REQUIRE_SEEDED_DIGEST=1`; otherwise it prints
  one notice and skips only that comparison. A comment states the fixture was recorded
  by 0006 code with no checkpoints, so equality shows checkpoint writes change no
  decision.
- **Stop-and-resume equality.** `resume_digest_matches_uninterrupted` asserts
  digest(B) equals digest(A) for every adapter.
- **Refusal wording.** Each refusal test asserts
  `err.to_string().contains(&expected.to_string())`, since the caller may prepend
  run and path.
- **Byte mutations** (through `rusqlite` on the stopped db; only the payload changes):
  - truncated: `payload[..len / 2]`, expect `Truncated`.
  - corrupt byte: `payload[len / 2] ^= 0xff`, expect `ChecksumMismatch`.
  - version: bytes `8..12` = `2u32` LE, trailer recomputed as
    `fnv1a(&payload[..len - 8]).to_le_bytes()`, expect
    `VersionMismatch { found: 2, expected: 1 }`.
- **Config mismatch** (`resume_refuses_config_mismatch`): after the stop run, rewrite
  only the `world_config` column of the single `run` row. Parse it as
  `serde_json::Value`, set its serialised world `seed` key to `6` (confirm the key from
  the stored JSON), write it back, and leave the payload untouched. Resume must return
  `Err(HeadlessError::Checkpoint(CheckpointError::ConfigMismatch))`.
- **Non-checkpoint tables unchanged (INV-2).** Snapshot after the mutation and before
  the refused resume: hash every `sqlite_master` table except `supervisor_checkpoint`
  with `SELECT * FROM "<t>" ORDER BY rowid` using the copied hasher. After the refusal
  the hash is equal, and the payload (or `world_config` text) equals its pre-resume
  value.
- **Fail closed.** Copy `gpu_or_skip` and its guard from
  `tests/supervisor_headless.rs`; it panics when `XAGENT_REQUIRE_GPU=1` and no adapter.

## Non-goals

- No product code; no change to `headless.rs`, `supervisor_checkpoint.rs`,
  `supervisor_store.rs`, or any `gpu_kernel` file.
- No submit-count test and no INV-4 claim.
- No checkpoint history, no new dependency, no edit to the 0006 fixture or gate.

## Steps

**Steps:**

1. Create `crates/xagent-sandbox/tests/supervisor_resume_digest.rs` with a module doc,
   constants (`SEED = 5`, `AGENTS = 10`, `AGENT_TICK_BUDGET = 600_000`,
   `STOP_AT_TICK = 10_000`), the config helper, `gpu_or_skip`, and copied
   `adapter_description` and `env_flag`.
2. Copy the digest code from the landed 0006 `supervisor_digest.rs` (hasher, `fnv1a`,
   `hash_section`, `digest_database`, fixture read, `is_digest_hex`), each copy
   commented with its source file.
3. Add helpers: `run_uninterrupted(db)`, `run_stop_then_resume(db)`,
   `checkpoint_payload` and `set_payload` (read-write on the one `supervisor_checkpoint`
   row), `set_world_config`, `table_snapshot(db) -> u64`, and `refusal(db) ->
   CheckpointError` (resume with `resume = true` expects
   `Err(HeadlessError::Checkpoint(e))`).
4. Add tests `resume_digest_matches_uninterrupted`,
   `resume_digest_checkpointing_does_not_change_digest`, `resume_refuses_truncated`,
   `resume_refuses_corrupt_byte`, `resume_refuses_version_mismatch` and
   `resume_refuses_config_mismatch`, each refusal test following the decisions above.
5. Request `focused-proof` through `verify` and read its output: all six named tests
   pass, none skipped, with `XAGENT_REQUIRE_GPU=1` set. If one fails, fix this file
   only and request again.
6. Request `sandbox-clippy` and `sandbox-fmt` through `verify`. If `sandbox-fmt`
   fails, apply its fix and request all three again. Edit nothing outside `touches`.

- **Done when:** the focused-proof check passes with resume_digest_matches_uninterrupted, resume_digest_checkpointing_does_not_change_digest, resume_refuses_truncated, resume_refuses_corrupt_byte, resume_refuses_version_mismatch and resume_refuses_config_mismatch all passing and none skipped, and the sandbox-clippy and sandbox-fmt checks pass, proving INV-1 (uninterrupted checkpoint-writing digest equals the 0006 golden and the stop-and-resumed digest equals the uninterrupted digest) and INV-2 (truncated, corrupt and version-mismatched checkpoints are refused with the matching CheckpointError, config-mismatched with ConfigMismatch, and non-checkpoint tables are unchanged).
