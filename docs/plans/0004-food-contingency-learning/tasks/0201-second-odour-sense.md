---
id: second-odour-sense
title: "Second-odour descriptor and config-selected sense table"
workstream: "0200"
kind: task
role: developer
depends_on:
  - food-kind-config
gated: false
touches:
  - crates/xagent-brain/src/senses.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- senses::tests::"
    timeout_ms: 3600000
---

# 0201 Second-odour descriptor and config-selected sense table

## Outcome

`crates/xagent-brain/src/senses.rs` gains the second-odour sense as one
descriptor, a config-selected sense table, and two selector functions. With
`BrainConfig.second_odour_sense_enabled` off, the selection is the unchanged
declared table and there are zero extra channels. With it on, the table is the
declared table with the second-odour descriptor appended last. No other file
changes. See PLAN.md "Owner decisions on the 0003 interface" and ARCHITECTURE.md
"Second odour as a declared sense", INV-01 and INV-05.

## Baseline facts

- `senses.rs` is created by 0003 task `sense-descriptor`, which lands before this
  task starts. It defines `SenseDescriptor` (fields `name`, `channels`, `input`,
  `adaptation_rate`, `whitening`, `learning_rate`, `routes`), `SenseInput` with
  `NonVisualFeatures { offset }`, `Whitening` with `ClosedForm2 {
  initial_variance }`, `LearningRate::Actor`, `RouteDescriptor { route,
  heritability }`, `MotorRoute::Turn`, `Heritability::LearnedOnly`, the consts
  `SCENT_FEATURE_OFFSET` (25), `MAX_SENSE_SLOTS`, `SMELL`, `VISION` and
  `DECLARED_SENSES = &[SMELL, VISION]`, plus a const validity function
  asserted over `DECLARED_SENSES`.
- `SCENT_CHANNELS` is `pub const SCENT_CHANNELS: usize = 2;` in
  `crates/xagent-brain/src/buffers.rs`.
- `xagent-brain` depends on `xagent-shared`, which exports `BrainConfig`.
  `food-kind-config` adds `second_odour_sense_enabled: bool` (serde default
  `false`) to `BrainConfig`. Read that field; do not redefine it.
- Consumers reach the new public items as `xagent_brain::senses::...`.
  `crates/xagent-brain/src/lib.rs` is not in `touches`, so no re-export is added.

## Decisions

- **Feature offset.** `SECOND_ODOUR_FEATURE_OFFSET = SCENT_FEATURE_OFFSET +
  SCENT_CHANNELS`, which is 27. It is the first non-visual feature slot after the
  two scent features. The type follows the landed `SCENT_FEATURE_OFFSET` and the
  landed `NonVisualFeatures` field type. The PLAN's `usize` annotation applies only
  if the landed types are `usize`. If they differ, use the landed type and convert
  `SCENT_CHANNELS` with a named const or a checked conversion. Record the mapping
  in the report.
- **Routes: Turn only.** `SECOND_ODOUR_ROUTES` is the one-entry slice
  `[RouteDescriptor { route: MotorRoute::Turn, heritability:
  Heritability::LearnedOnly }]`. Its doc comment states the technical reason: the
  layout builder admits only `Turn` for a sense appended after the base senses and
  panics otherwise. The doc comment does not name plans or decisions. Smell's route
  slice is never reused.
- **Descriptor values.** `SECOND_ODOUR` has name `"second_odour"`, channels
  `SCENT_CHANNELS`, input `NonVisualFeatures { offset:
  SECOND_ODOUR_FEATURE_OFFSET }`, adaptation rate `SMELL.adaptation_rate`,
  whitening `SMELL.whitening`, learning rate `LearningRate::Actor`, and routes
  `SECOND_ODOUR_ROUTES`. Every value comes from a const field access or a named
  const. No literal is copied from smell.
- **Table.** `SENSES_WITH_SECOND_ODOUR: &[SenseDescriptor] = &[SMELL, VISION,
  SECOND_ODOUR]`. The validity function takes a slice parameter, so the existing
  `DECLARED_SENSES` assertion and a new assertion on `SENSES_WITH_SECOND_ODOUR`
  both use it. The new assertion also checks `len() <= MAX_SENSE_SLOTS`. The
  existing `DECLARED_SENSES` assertion stays intact.
- **Selectors.** `senses_for_config(config: &BrainConfig) -> &'static
  [SenseDescriptor]` returns `SENSES_WITH_SECOND_ODOUR` when
  `config.second_odour_sense_enabled`, and otherwise `DECLARED_SENSES`.
  `extra_sensory_channels(config: &BrainConfig) -> usize` returns
  `SECOND_ODOUR.channels` when the flag is on and `0` otherwise. Its doc comment
  says it is the number of sensory channels appended after the scent channels,
  consumed by the layout and the kernel override.
- **Compare floats by bits.** Tests compare `adaptation_rate` and the whitening
  `initial_variance` with `to_bits()`.

**Steps:**

1. Read the landed `crates/xagent-brain/src/senses.rs` and the `pub mod senses;`
   line in `crates/xagent-brain/src/lib.rs`. Confirm the names and types listed in
   Baseline facts. If a name or type differs, map it to the landed one when the
   meaning is identical, and record the mapping in the report. Stop and report only
   if the meaning differs.
2. Add `SECOND_ODOUR_FEATURE_OFFSET`, `SECOND_ODOUR_ROUTES` and `SECOND_ODOUR` in
   `senses.rs`, each with a `///` doc comment, following the Decisions above.
3. Generalise the existing const validity function to take `table:
   &[SenseDescriptor]`. Keep `const _: () = assert!(...DECLARED_SENSES...)` as it
   is. Add `SENSES_WITH_SECOND_ODOUR` and a `const _` assertion over it that also
   checks its length against `MAX_SENSE_SLOTS`.
4. Add `pub fn senses_for_config` and `pub fn extra_sensory_channels` with `///`
   doc comments. Add `use xagent_shared::BrainConfig;` at the top of `senses.rs`.
5. In the existing `#[cfg(test)] mod tests` of `senses.rs`, add these four tests
   with these exact names: `second_odour_descriptor_matches_decisions` (name,
   channels 2, input offset 27 equal to `SCENT_FEATURE_OFFSET + SCENT_CHANNELS`,
   adaptation rate and whitening bit-equal to `SMELL`'s, learning rate `Actor`,
   routes exactly one entry, `Turn` with `LearnedOnly`);
   `senses_for_config_off_is_declared_table` (`BrainConfig::default()` gives
   a table equal by value to `DECLARED_SENSES` (`assert_eq!` on the slices; no
   pointer-identity check, since a `const` has no single address), and `extra_sensory_channels` is 0);
   `senses_for_config_on_appends_second_odour_last` (flag on gives 3 entries, the
   first two equal `DECLARED_SENSES` in order, the last equals `SECOND_ODOUR`, and
   `extra_sensory_channels` is 2); `second_odour_routes_are_turn_only` (no
   `Forward` or `Critic` route, and no `Mutated` heritability). Leave the existing
   tests unchanged.
6. Hygiene: every new `pub` item has a `///` doc comment. `senses.rs` has no
   `TODO`, and no plan, task, workstream or decision name in source or comments,
   because `crates/xagent-sandbox/tests/contributing_guard.rs` enforces this. Run
   nothing else. The focused-proof check runs all `senses::tests::` tests.

- **Done when:** the `cargo test -p xagent-brain --lib -- senses::tests::` check
  (focused-proof) passes on the host, with output reporting `test result: ok` and
  the four tests `second_odour_descriptor_matches_decisions`,
  `senses_for_config_off_is_declared_table`,
  `senses_for_config_on_appends_second_odour_last` and
  `second_odour_routes_are_turn_only` each reporting `ok`. It proves INV-05: the
  second pair is a Turn/LearnedOnly descriptor on the existing `NonVisualFeatures`
  input at offset 27, appended last. It proves INV-01: with the flag off,
  `senses_for_config` returns `DECLARED_SENSES` and `extra_sensory_channels` is 0.

## Non-goals

- No layout (`buffers.rs`), kernel (`gpu_kernel.rs`), WGSL, encoder or mutation
  change. No `lib.rs` change.
- No new `SenseInput` kind.
- No route other than `Turn` for the second odour.
- No change to `DECLARED_SENSES`, `SMELL`, `VISION`, their routes or their
  heritability.
- No GPU test. The focused proof is CPU-only.
