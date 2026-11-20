---
id: sense-descriptor
title: "Declarative sense descriptor table"
workstream: "0200"
kind: task
role: developer
depends_on:
  - parity-golden
gated: false
status: done
merged_as: "597683140a9739ebb74514d270817e16c0cd24d9"
touches:
  - crates/xagent-brain/src/senses.rs
  - crates/xagent-brain/src/lib.rs
  - docs/decisions/sense-route-wiring.md
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- senses::"
    timeout_ms: 3600000
  - name: fmt
    command: "cargo fmt --all -- --check"
    fix: "cargo fmt --all"
  - name: clippy
    command: "cargo clippy --workspace --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# 0201 — Declarative sense descriptor table

## Outcome

`crates/xagent-brain/src/senses.rs` holds the sense descriptor types and the
declared table `DECLARED_SENSES = [SMELL, VISION]`, re-exported from
`crates/xagent-brain/src/lib.rs`. The module is pure data: no caller, no GPU
code, no behaviour change. `docs/decisions/sense-route-wiring.md` records the route verdicts the
table was built from. Nothing in the crate reads the table yet.

## Baseline facts

- `SCENT_FEATURE_OFFSET` (25), `SCENT_WHITENING_RATE` (0.01),
  `VISION_PATHWAY_RATE` (0.01) and `VISION_WHITENING_REFRESH` (20) are WGSL
  `const`s in `crates/xagent-brain/src/shaders/kernel/common.wgsl` (about lines
  638–656), written with `u` and `f32` literal forms. They have no Rust source
  today; senses.rs restates them as `pub const`.
- Reused from `crates/xagent-brain/src/buffers.rs` (all `pub`): `SCENT_CHANNELS`
  (2), `VISION_PATHWAY_INPUTS` (8), `SCENT_INITIAL_VARIANCE` (1e-4),
  `VISION_PATHWAY_INITIAL_VARIANCE` (1e-4), `O_VISION_PLASTICITY`.
- The route records `docs/decisions/pathway-forward-route.{md,json}`,
  `docs/decisions/pathway-critic-route.{md,json}` and
  `docs/decisions/pathway-critic-inputs.{md,json}`, and the constants
  `DEFAULT_FORWARD_PATHWAY_ENABLED` and `DEFAULT_CRITIC_INPUT` in
  `crates/xagent-shared/src/config.rs`, are created by plan 0002. They are
  absent at baseline and must exist when this task starts.

## Decisions

- Route admission, fixed:
  - `Turn` (`Mutated`) on both senses, always.
  - `Forward` (`LearnedOnly`) on both senses iff the forward-route verdict is
    adopted AND `DEFAULT_FORWARD_PATHWAY_ENABLED` is `true`.
  - `Critic` (`LearnedOnly`) on both senses iff the critic-route verdict is
    adopted AND `DEFAULT_CRITIC_INPUT` is not `CriticInput::Encoding` AND the
    critic-inputs winner equals `DEFAULT_CRITIC_INPUT`.
  - A missing `.md` or `.json`, a `.md`/`.json` disagreement, or any
    record/default disagreement is BLOCKED: write nothing else and report the
    file and the observed values.
- Both senses share one `SENSE_ROUTES` slice in the order Turn, Forward, Critic,
  listing only the routes admitted above.
- Table values: `SMELL` = name `"smell"`, channels `SCENT_CHANNELS`, input
  `NonVisualFeatures { offset: SCENT_FEATURE_OFFSET }`, adaptation rate
  `SCENT_WHITENING_RATE`, whitening `ClosedForm2 { initial_variance:
  SCENT_INITIAL_VARIANCE }`, learning rate `Actor`. `VISION` = name `"vision"`,
  channels `VISION_PATHWAY_INPUTS`, input `VisualHemifields`, adaptation rate
  `VISION_PATHWAY_RATE`, whitening `Jacobi { refresh_ticks:
  VISION_WHITENING_REFRESH, initial_variance: VISION_PATHWAY_INITIAL_VARIANCE }`,
  learning rate `ActorTimesGene { gene_offset: O_VISION_PLASTICITY }`.
- Types, exactly as the architecture's Sense descriptor block, deriving
  `Clone, Copy, Debug, PartialEq`. `MAX_SENSE_SLOTS = 4`,
  `MAX_SENSE_CHANNELS = 8`.
- Every item has a `///` doc comment describing what it is and why its value
  holds, without naming any plan, task or decision document.
- Compare `f32` values in tests with `to_bits()`, not `==`.

**Steps:**

1. **Route gate and record.** Read the three decision records and
   `DEFAULT_FORWARD_PATHWAY_ENABLED` and `DEFAULT_CRITIC_INPUT` in
   `crates/xagent-shared/src/config.rs`. Apply the admission rules above; stop
   as BLOCKED if any gate fails. Otherwise write
   `docs/decisions/sense-route-wiring.md` with: each
   record path and its verdict, each default value, and a table of the resulting
   routes per sense (smell, vision).
2. **Types and constants.** Create `crates/xagent-brain/src/senses.rs` with a
   `//!` module doc. Define the enums and structs from the architecture, the
   `pub const` values `SCENT_FEATURE_OFFSET`, `SCENT_WHITENING_RATE`,
   `VISION_PATHWAY_RATE`, `VISION_WHITENING_REFRESH`, `MAX_SENSE_SLOTS`,
   `MAX_SENSE_CHANNELS`, each with a doc comment naming the WGSL constant it
   mirrors in `common.wgsl`.
3. **Table.** In the same file, define `SENSE_ROUTES`, `pub const SMELL` and
   `pub const VISION` with the values above, and
   `pub const DECLARED_SENSES: &[SenseDescriptor] = &[SMELL, VISION];`.
4. **Compile-time checks.** Add `const fn declared_senses_are_valid() -> bool`
   using `while` loops over `DECLARED_SENSES`, checking channels `<=
   MAX_SENSE_CHANNELS` and that every `ClosedForm2` sense has exactly 2
   channels, plus `DECLARED_SENSES.len() <= MAX_SENSE_SLOTS`. Then
   `const _: () = assert!(declared_senses_are_valid());`. Add
   `pub mod senses;` and `pub use senses::{SenseDescriptor, DECLARED_SENSES};`
   to `lib.rs` beside the existing `pub mod` and `pub use` lines.
5. **Tests.** In `senses.rs`, add `#[cfg(test)] mod tests` (tests named
   `senses::tests::*`) with:
   - `wgsl_constants_match_table`: a helper reads `const NAME: TYPE = VALUE;` from
     `include_str!("shaders/kernel/common.wgsl")` and parses VALUE; assert that
     `SCENT_FEATURE_OFFSET`, `SCENT_WHITENING_RATE`, `VISION_PATHWAY_RATE`,
     `VISION_WHITENING_REFRESH`, `SCENT_INITIAL_VARIANCE` and
     `VISION_PATHWAY_INITIAL_VARIANCE` equal the Rust values, and that
     `SMELL`/`VISION` carry them.
   - `only_turn_routes_are_mutated`: every `Mutated` route is `Turn` on `smell`
     or `vision`; every `Turn` route is `Mutated`; every non-`Turn` route is
     `LearnedOnly`.
   - `declared_routes_match_route_wiring`: `SMELL.routes` and `VISION.routes`
     equal a literal `&[RouteDescriptor]` expected list written from the
     verdicts in step 1.
   The step-1 verdicts are the source for the expected list, so the test fails
   if the table disagrees with the record.
   Finally, check every `senses.rs` line for `TODO` and for plan, task or
   decision names, and remove any found.

- **Done when:** INV-04 — the `focused-proof` check (`cargo test -p xagent-brain --lib -- senses::`) passes with the three named tests above, and `declared_routes_match_route_wiring` fails if any `Forward` or `Critic` route is added or removed without a matching verdict in `docs/decisions/sense-route-wiring.md`, proving INV-04.

## Non-goals

- No caller of `DECLARED_SENSES` and no layout, offset or WGSL change; those
  belong to `derived-sense-layout` and later tasks.
- No edit to `buffers.rs`, `common.wgsl`, `config.rs` or any file outside
  `touches`; the verdict records and defaults are read, never written.
- No declaration of a route the verdicts did not admit; rejected routes stay
  out of the table.
