---
id: supervisor-core-state
title: Counted supervisor streams and core capture/restore
workstream: "0100"
kind: task
role: developer
depends_on:
  - checkpoint-envelope
gated: false
touches:
  - crates/xagent-sandbox/src/supervisor.rs
status: done
merged_as: "34f2bebbfb73c159c2994982a34f7166377de4c9"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib supervisor::"
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
---

# Counted supervisor streams and core capture/restore

## Outcome

`crates/xagent-sandbox/src/supervisor.rs` gains a counted random stream,
`StreamRng`, used by every supervisor-owned random stream in place of
`SmallRng`. The core can be captured into a serialisable `SupervisorState` and
restored from it, so a restored `Supervisor` makes the same decisions and draws
the same values from that point on. Draw values, and so the plan 1 supervisor
digest, do not change. No other file changes.

## Decisions (explicit)

- **Scope of the file.** Edit only `crates/xagent-sandbox/src/supervisor.rs`.
- **Plan 0006 fields.** Any stream 0006 adds to `Supervisor` or `TickSchedule`
  becomes a `StreamRng`; any telemetry field it adds is covered by `capture`
  and `restore`. Find them by reading the file.
- **`StreamRng`.** Public, exactly these items:
  - `pub struct StreamRng { inner: SmallRng, seed: u64, steps: u64 }`
  - `pub fn seed_from(seed: u64) -> Self`: `SmallRng::seed_from_u64(seed)`,
    `steps = 0`.
  - `pub fn steps(&self) -> u64`.
  - `pub fn restore(seed: u64, steps: u64) -> Self`: `seed_from(seed)`, then
    call `next_u64()` on `inner` exactly `steps` times, discarding the values,
    leaving `self.steps == steps`. Restore must not use the counting `RngCore`
    methods, so the count is set once, not doubled.
  - `impl RngCore for StreamRng`: `next_u32` and `next_u64` delegate to `inner`
    and add 1 to `steps`; `fill_bytes(dest)` delegates and adds
    `ceil(dest.len() / 8)` (zero for an empty slice). Document this counting
    rule in the `StreamRng` doc comment. Step 4's `stream_rng_restore_resumes`
    checks it with `fill_bytes` lengths that are not multiples of 8. If that
    test shows the rule is wrong, stop and report the failing assertion; do not
    change the rule to pass.
  - Do not read files outside this checkout; the tests are the proof.
- **Stream fields.** Every `SmallRng` field in `TickSchedule` (`rng`) and
  `Supervisor` (`tiebreak_rng`, `parent_rng`, and any 0006 stream) becomes
  `StreamRng`, built with `StreamRng::seed_from(stream_seed(...))` or
  `seed_from(seed)` exactly where `SmallRng::seed_from_u64(...)` stood. Local
  `SmallRng` values (for example `initial` in `Supervisor::new`) may stay. A
  function that receives a stream field and is typed `&mut SmallRng` (for
  example near line 440) takes `&mut StreamRng` or a generic `R: Rng + ?Sized`.
- **Next boundary.** `pub fn next_boundary_tick(&self) -> u64` returns
  `self.boundary.next_tick()`.
- **Serde shape.** `#[derive(Clone, Debug, Serialize, Deserialize)] pub struct
  SupervisorState`. Types with private fields, or holding `AgentBrainState`, use
  module-private `...State` / `...Record` mirror structs. `AgentBrainState`
  (in `xagent-brain`, no serde, not editable) is stored as private
  `BrainStateRecord { #[serde(with = "crate::supervisor_checkpoint::f32_bits::vec")] brain_state: Vec<f32>, #[serde(with = "crate::supervisor_checkpoint::f32_bits::vec")] patterns: Vec<f32> }`
  with `From` conversions both ways. `BrainConfig`, `SupervisorConfig` and
  `GovernorConfig` already derive serde and are stored directly. Stream records
  are `StreamRngState { seed: u64, steps: u64 }`.
- **Floats.** Every `f32` in the state types uses
  `#[serde(with = "crate::supervisor_checkpoint::f32_bits")]`, every `Vec<f32>`
  uses `...f32_bits::vec`, and `Option<f32>` is stored as `Option<u32>` of
  `to_bits()`. No float is a JSON number.
- **Capture.** `pub fn capture(&self) -> SupervisorState` destructures with
  `let Supervisor { ... } = self;` naming every field, so a new field is a
  compile error until it is captured.
- **Restore.** `pub fn restore(state: SupervisorState) -> Result<Supervisor,
  SupervisorError>` returns an error for each of these and no other:
  - any stream `steps > 1 << 40` (`const MAX_STREAM_STEPS: u64 = 1 << 40;`);
  - empty `slots`, or `config.validate(slots.len())` failing (mapped with
    `SupervisorError::Config`);
  - `next_occupant < slots.len() as u64`;
  - any other inconsistency found while rebuilding (for example a
    `retired.slot` that is not a valid slot index).
  Add `InvalidState { reason: String }` to `SupervisorError`, with a `Display`
  arm writing `supervisor state is invalid: {reason}`.
- **Behaviour is unchanged.** No decision, draw order or tick logic changes. The
  only new public items are `StreamRng`, `next_boundary_tick`, `SupervisorState`,
  `capture`, `restore` and the `InvalidState` variant.

## Non-goals

- No change to other files, the store, the kernel, or the drivers. No checkpoint
  encoding, file I/O or SQLite; tests use `serde_json` directly. No dependency
  change. Do not edit existing tests; add new ones only.

## Steps

**Steps:**

1. In `crates/xagent-sandbox/src/supervisor.rs`, add `StreamRng` with its doc
   comment (counting rule), `seed_from`, `steps`, `restore` and the `RngCore`
   impl. Import `rand::RngCore`.
2. In the same file, change the `rng` field of `TickSchedule`, the `tiebreak_rng`
   and `parent_rng` fields of `Supervisor`, and any 0006 stream field from
   `SmallRng` to `StreamRng`, updating their constructors and any `&mut SmallRng`
   parameter that receives a stream field. Add `pub fn next_boundary_tick`.
   Existing tests must still compile and pass unchanged.
3. In the same file, add `StreamRngState`, `BrainStateRecord`, the `...State`
   mirrors for `TickSchedule`, `SlotLedger`, `SlotCounters` (if it holds floats
   or is needed), `Archive`, `ArchiveEntry` and the `RetiredSlot` record, with the
   serde attributes from Decisions. Add `SupervisorState` holding the config, the
   scalar `f32` fields as bits, the slot records, the archive record, both
   schedule records, the two stream records, `next_tick`, `next_occupant` and the
   retired record. Add `capture` and `restore` with the checks in Decisions, and
   the `InvalidState` variant with its `Display` arm.
4. Add these tests to the existing `#[cfg(test)] mod tests`:
   `stream_rng_matches_small_rng` (10,000 mixed draws of `next_u32`, `next_u64`
   and `fill_bytes` with lengths including 0, 1, 7, 8, 9 and 13, compared with a
   `SmallRng` seeded identically; every value equal);
   `stream_rng_restore_resumes` (draw a mixed prefix as above, record `steps()`,
   then `StreamRng::restore(seed, steps)`; the next 1,000 mixed draws equal the
   original's, and `steps()` agrees after each draw); and
   `supervisor_restore_continues_identically` (build a `Supervisor` with
   `Supervisor::new` and the existing test helpers for governor and brain config,
   drive it with synthetic `SlotCounters` over at least 200 samples, capture
   mid-run, round-trip through `serde_json::to_string` and `from_str`, restore,
   then drive both cores 1,000 further samples with the same inputs and assert
   equal decisions, outcomes, newborn brain states (by `to_bits`) and `steps()`).
5. Run the `focused-proof`, `sandbox-clippy` and `sandbox-fmt` checks. If
   `sandbox-fmt` fails, run its `fix` command and rerun the checks. Edit no file
   other than `crates/xagent-sandbox/src/supervisor.rs`.

- **Done when:** the `focused-proof` check passes `supervisor::tests::stream_rng_matches_small_rng`, `supervisor::tests::stream_rng_restore_resumes`, `supervisor::tests::supervisor_restore_continues_identically` and every pre-existing `supervisor::` test, and `sandbox-clippy` and `sandbox-fmt` pass, proving INV-6 (a `Supervisor::restore(sup.capture())` core yields the same decisions, outcomes and stream draws as the original, and a `StreamRng` matches a `SmallRng` with the same seed).
