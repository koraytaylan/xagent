---
id: checkpoint-envelope
title: Versioned checksummed checkpoint envelope
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-sandbox/src/supervisor_checkpoint.rs
  - crates/xagent-sandbox/src/lib.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: cargo test -p xagent-sandbox --lib supervisor_checkpoint
    timeout_ms: 3600000
  - name: sandbox-clippy
    command: cargo clippy -p xagent-sandbox --all-targets -- -D warnings
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: cargo fmt -p xagent-sandbox -- --check
    fix: cargo fmt -p xagent-sandbox
---

# Versioned checksummed checkpoint envelope

## Outcome

`crates/xagent-sandbox/src/supervisor_checkpoint.rs` is a pure, GPU-free codec
for one checkpoint: a list of named byte sections plus a tick, serialised with
the exact little-endian layout below and protected by a trailing FNV-1a-64
checksum. `decode` refuses every corrupt, truncated, oversized or
version-mismatched input with a `CheckpointError` and never returns partial
data. The module also provides the `f32_bits` serde helpers that later tasks
use to store floats by bit pattern. Nothing else in the crate changes.

## Decisions (explicit)

- Module declaration: add `pub mod supervisor_checkpoint;` in
  `crates/xagent-sandbox/src/lib.rs` between the existing `pub mod supervisor;`
  and `pub mod supervisor_run;` lines. This is the alphabetical position that
  rustfmt keeps; the existing `pub mod supervisor_store;` line stays unchanged.
- No new dependency. FNV-1a is hand-written. `serde` and `serde_json` are
  already sandbox dependencies (`crates/xagent-sandbox/Cargo.toml`); the tests
  use `serde_json` and serde derive for the `f32_bits` round trip.
- Public API, exactly these names:
  - `pub const FORMAT_VERSION: u32 = 1;`
  - `pub const MAX_CHECKPOINT_BYTES: usize = 512 * 1024 * 1024;`
  - private `const MAGIC: [u8; 8] = *b"XAGCKPT\0";`, `const MAX_SECTIONS: usize = 256;`,
    `const MAX_NAME_BYTES: usize = 64;`
  - `#[derive(Debug, Clone, PartialEq, Eq)] pub struct Checkpoint { pub tick: u64, pub sections: Vec<(String, Vec<u8>)> }`
  - `impl Checkpoint { pub fn section(&self, name: &str) -> Result<&[u8], CheckpointError> }`
    returns the bytes of the first section with that name, or `CheckpointError::Missing`.
  - `pub fn encode(checkpoint: &Checkpoint) -> Vec<u8>`. It is infallible, so it
    `assert!`s that `sections.len() <= 256` and every name is at most 64 bytes;
    these are caller bugs. It does not check duplicates or total size.
  - `pub fn decode(bytes: &[u8]) -> Result<Checkpoint, CheckpointError>`.
  - `#[derive(Debug, Clone, PartialEq, Eq)] pub enum CheckpointError` with exactly
    `Truncated`, `BadMagic`, `VersionMismatch { found: u32, expected: u32 }`,
    `ChecksumMismatch`, `TooLarge { len: usize, max: usize }`, `Malformed(String)`,
    `ConfigMismatch`, `Missing`. `Display` is plain English and gives the reason
    (for example "checkpoint is truncated; it ends before its declared content");
    the run and database path are added by the caller later, not here.
    Implements `std::error::Error`.
  - `pub mod f32_bits` with `pub fn serialize` and `pub fn deserialize` for `f32`
    (serialised as `to_bits()` `u32`), and `pub mod vec` with the same pair for
    `Vec<f32>`. Used as `#[serde(with = "...")]`.
- Layout, all little-endian, in this order: `MAGIC` (8 bytes), `u32 FORMAT_VERSION`,
  `u64 tick`, `u32 section_count`, then for each section `u16 name_len`, name bytes
  (UTF-8), `u64 len`, data bytes; then `u64` FNV-1a-64 of every preceding byte.
  FNV-1a-64: start `h = 0xcbf2_9ce4_8422_2325`; for each byte `h ^= byte` then
  `h = h.wrapping_mul(0x0000_0100_0000_01b3)`.
- `decode` checks in this order, and each step returns at its first failure:
  1. `bytes.len() > MAX_CHECKPOINT_BYTES` gives `TooLarge { len, max }`.
  2. Framing walk, reading only the length fields: the header needs 24 bytes
     (8 + 4 + 8 + 4) or `Truncated`; `section_count > 256` or `name_len > 64` gives
     `Malformed`; every name and data range must fit in the buffer or `Truncated`.
     After the last section, fewer than 8 bytes remaining gives `Truncated`,
     more than 8 gives `Malformed("trailing bytes")`. Every cut of a valid encoding
     therefore reports `Truncated`.
  3. Magic mismatch gives `BadMagic`.
  4. `u32` version at offset 8 not equal to `FORMAT_VERSION` gives
     `VersionMismatch { found, expected: FORMAT_VERSION }`.
  5. FNV-1a-64 of all bytes before the final 8 not equal to the trailer gives
     `ChecksumMismatch`.
  6. Section parsing: names must be valid UTF-8 and unique, else `Malformed`.
     Only now are sections copied into the returned `Checkpoint`.
- The codec does not interpret section contents. `config`, `run` and `kernel.*`
  are opaque to this module.

## Non-goals

- No SQLite, no store, no GPU, no kernel or supervisor types, no config
  parsing, no `ConfigMismatch` production (its variant exists for later callers).
- No edit to any file other than the two in `touches`; no dependency or
  version change.

## Steps

**Steps:**

1. Add `pub mod supervisor_checkpoint;` to `crates/xagent-sandbox/src/lib.rs`
   between `pub mod supervisor;` and `pub mod supervisor_run;`, leaving every
   other line unchanged.
2. Create `crates/xagent-sandbox/src/supervisor_checkpoint.rs` with the
   constants, `Checkpoint` and its `section` method, the `CheckpointError` enum
   with `Display` and `std::error::Error`, and the private `fnv1a64` helper.
3. In the same file, implement `encode` exactly per the layout and assertions in
   Decisions.
4. In the same file, implement `decode` with the six ordered checks in
   Decisions. Every check returns its named error; no check is skipped or
   reordered.
5. In the same file, add `pub mod f32_bits` (with `pub mod vec`) whose
   `serialize` and `deserialize` functions pass `u32` bit patterns through
   `to_bits` and `from_bits`.
6. In the same file, add `#[cfg(test)] mod tests` containing the tests
   `round_trip_restores_sections`, `every_truncation_is_refused_as_truncated`
   (for every `len < encoded.len()`, `decode(&encoded[..len])` equals
   `Err(Truncated)`), `every_flipped_byte_is_refused` (for every index, XOR the
   byte with `0xFF`; `decode` returns `Err`), `version_two_is_refused` (set
   bytes 8..12 to `2u32` little-endian; expect `VersionMismatch { found: 2,
   expected: 1 }`), `oversize_is_refused_as_too_large` (decode a
   `vec![0u8; MAX_CHECKPOINT_BYTES + 1]`, expect `TooLarge`),
   `bad_magic_is_refused`, `duplicate_names_are_malformed`, and
   `f32_bits_round_trip_bit_patterns` (a derive-serialised struct with
   `#[serde(with = "f32_bits")]` and `#[serde(with = "f32_bits::vec")]` fields
   holding `f32::from_bits(0x7FC0_1234)`, `-0.0`, `f32::from_bits(1)`,
   `f32::INFINITY` and `f32::NEG_INFINITY`, round-tripped through `serde_json`
   and compared with `to_bits`).

- **Done when:** the `focused-proof` check passes the tests `supervisor_checkpoint::tests::round_trip_restores_sections`, `every_truncation_is_refused_as_truncated`, `every_flipped_byte_is_refused`, `version_two_is_refused`, `oversize_is_refused_as_too_large`, `bad_magic_is_refused`, `duplicate_names_are_malformed` and `f32_bits_round_trip_bit_patterns`, and the `sandbox-clippy` and `sandbox-fmt` checks pass, proving INV-2 (a corrupt, truncated or version-mismatched checkpoint is refused with a named `CheckpointError`).
