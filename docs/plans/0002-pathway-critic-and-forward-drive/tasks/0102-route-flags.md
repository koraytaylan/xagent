---
id: route-flags
title: "Route flags in BrainConfig"
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-shared/src/lib.rs
  - crates/xagent-sandbox/src/agent/mod.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-shared"
    timeout_ms: 1800000
  - name: sandbox-compiles
    command: "cargo check -p xagent-sandbox --all-targets"
    timeout_ms: 3600000
  - name: crate-fmt
    command: "cargo fmt -p xagent-shared -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-shared -p xagent-sandbox"
  - name: crate-clippy
    command: "cargo clippy -p xagent-shared -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: contributing-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Route flags in BrainConfig

Add two route flags to `BrainConfig`: `critic_input` and
`forward_pathway_enabled`. Both default off and are locked per batch. Nothing
reads them yet, so behaviour is unchanged.

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, add `pub enum CriticInput { Encoding,
   Pathway, EncodingAndPathway }` with derives `Clone, Copy, Debug, PartialEq,
   Eq, Serialize, Deserialize` and `#[serde(rename_all = "snake_case")]`. Give
   each variant a doc comment: `Encoding` = the critic reads the centred 128-d
   encoding; `Pathway` = the ten whitened pathway inputs; `EncodingAndPathway` =
   both. Add documented `pub const DEFAULT_CRITIC_INPUT: CriticInput =
   CriticInput::Encoding;` and `pub const DEFAULT_FORWARD_PATHWAY_ENABLED: bool
   = false;`, with a doc line that each route is off until its gate passes.
2. In the same file, add fields `critic_input: CriticInput` and
   `forward_pathway_enabled: bool` to `BrainConfig`, each with
   `#[serde(default = "default_critic_input")]` or
   `#[serde(default = "default_forward_pathway_enabled")]`. Those private
   functions return the two consts. Field docs say: locked per batch, not
   heritable, not perturbed by `mutate_config`. Set both in `impl Default`, in
   `tiny()` and in `large()` using the consts.
3. In `crates/xagent-shared/src/lib.rs`, add `CriticInput`,
   `DEFAULT_CRITIC_INPUT` and `DEFAULT_FORWARD_PATHWAY_ENABLED` to the existing
   `pub use config::{...}` list.
4. In `crates/xagent-sandbox/src/agent/mod.rs`, edit only the two full
   `BrainConfig` literals: the mutation literal in `mutate_config_seeded` copies
   both fields from `parent`; the crossover literal in `crossover_config` copies
   both from `a`. Add a one-line `//` comment on each pair saying the flags are
   locked per batch and inherited unchanged.
5. In the `#[cfg(test)] mod tests` of `config.rs`, add these tests: (a) JSON
   without either key deserializes to `DEFAULT_CRITIC_INPUT` and
   `DEFAULT_FORWARD_PATHWAY_ENABLED`, asserted by comparing to the consts, never
   to literal values; (b) each of `"encoding"`, `"pathway"`,
   `"encoding_and_pathway"` round-trips through `serde_json`; (c) `"both"` fails to
   deserialize; (d) `forward_pathway_enabled: true` round-trips.
6. Run the host checks listed in `checks`. If one fails, fix it within the
   three touched files and rerun. Do not edit tests to pass them.

## Constraints

- Doc comments and source must not name any plan, task or decision document
  (INV-12). Describe behaviour only.
- No magic numbers outside the consts. Keep `#[serde(default)]` on every new
  field: no `deny_unknown_fields` and no field without a default.
- Do not change any other `BrainConfig` field, its serde attribute, or any
  behaviour in `mutate_config`, `crossover_config` or the presets beyond the new
  fields.
- Keep the `BrainConfig` struct-update (`..`) style of existing literals; if a
  full literal outside the touched files breaks, report it as a blocker rather
  than widening `touches`.

## Acceptance

- Test names: the four tests in step 5 pass under `focused-proof`.
- `sandbox-compiles` shows the sandbox's test targets still compile with the
  new fields.
- `contributing-guard` stays green.

- **Done when:** `cargo test -p xagent-shared` passes the flag default, round-trip and unknown-name tests, with the sandbox still compiling under the sandbox-compiles check and the contributing-guard check green (proves INV-5, INV-12).
