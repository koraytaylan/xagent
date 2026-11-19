---
id: sense-override-constants
title: "Sense table as pipeline override constants"
workstream: "0200"
kind: task
role: developer
depends_on:
  - derived-sense-layout
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/senses.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib -- sense_override"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
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

# 0204 — Sense table as pipeline override constants

## Outcome

`vision_override_constants(layout)` in `gpu_kernel.rs` also emits `SENSE_COUNT`
and, for each slot `k` in `0..MAX_SENSE_SLOTS` (4), the per-sense overrides
below, generated from `layout.senses()` and `layout.sense_blocks()`. Kind codes
come from methods on the `senses.rs` enums, so the emitter names no sense and
no input kind. `common.wgsl` declares each name as an `override` whose default
equals the declared table, and mirrors the code constants. `GpuKernel::new_with_senses`
builds a kernel for any sense table; `GpuKernel::new` delegates with
`DECLARED_SENSES`. No WGSL code reads the new names yet, so generated code and
every trajectory stay byte-identical.

## Decisions

- Keep the signature `fn vision_override_constants(layout: &BrainLayout) ->
  HashMap<String, f64>` and its `map.insert("VISION_W", ...)` line unchanged.
  New inserts go in private `insert_sense_overrides(map: &mut HashMap<String,
  f64>, layout: &BrainLayout)`, called from it.
- Prerequisite, provided by `derived-sense-layout`: `BrainLayout::senses() ->
  &'static [SenseDescriptor]` in the same order as `sense_blocks()`, and the
  constructor `BrainLayout::with_senses(brain_config, senses)`.
- Per slot `k`, keys `SENSE{k}_<NAME>`, all emitted as numbers:
  - Kind and parameter keys: `INPUT_KIND` (`SenseInput::kind_code()`),
    `INPUT_OFFSET`, `CHANNELS`, `WHITENING_KIND` (`Whitening::kind_code()`),
    `REFRESH_TICKS` (0 unless `Jacobi`), `ADAPTATION_RATE` (f32),
    `LEARNING_RATE_GENE` (`LearningRate::gene_offset_or_zero()`), `ROUTE_MASK`
    (`SenseDescriptor::route_mask()`).
  - Slot offsets: `MEAN`, `COVARIANCE`, `WHITENING`, `INPUT`, `TURN_WEIGHTS`,
    `TURN_TRACE`, `FORWARD_WEIGHTS`, `FORWARD_TRACE`, `CRITIC_WEIGHTS`,
    `CRITIC_TRACE`.
- Offset semantics (explicit): slot offsets are per-agent `brain_state` indices
  copied unchanged from `sense_blocks()`, the same space as today's WGSL `O_*`
  overrides, which WGSL adds to `brain_base`. `INPUT_OFFSET` for
  `NonVisualFeatures` is the feature offset inside the non-visual tail; WGSL
  adds the visual-block width at use, as the smell code does today. For
  `VisualHemifields` it is `0`.
- A field the layout lacks (no whitening for `ClosedForm2`, no refresh, a route
  not declared) is `0`. Unused slots (`k >= sense count`) carry `0` for every key.
- Route mask bits, Rust `pub const` and WGSL `const` with identical names:
  `SENSE_ROUTE_TURN` = 1, `SENSE_ROUTE_FORWARD` = 2, `SENSE_ROUTE_CRITIC` = 4.
- Kind codes, identical names in Rust and WGSL: `SENSE_INPUT_KIND_NON_VISUAL_FEATURES`
  = 0, `SENSE_INPUT_KIND_VISUAL_HEMIFIELDS` = 1, `SENSE_WHITENING_KIND_CLOSED_FORM_2`
  = 0, `SENSE_WHITENING_KIND_JACOBI` = 1. Each Rust code constant has a `///` comment.
- `common.wgsl` declares, after `O_SENSORY_MEAN`, the mirrored `const` codes,
  then `override SENSE_COUNT: u32` and 18 per-slot overrides for each of the 4
  slots (8 parameter keys plus 10 offset keys). `u32` defaults use the `u`
  suffix and rates use `f32`.
- Default literals (the developer cannot run code): derive every default by
  hand from the cascade arithmetic in `buffers.rs` as the `derived-sense-layout`
  tests restate it. Smell is slot 0 (core 2+3+2+2+2 slots after `O_RECENT_NORMS`
  and `RECENT_CAP`), vision is slot 1 (core 8+64+64+8+8+8), and plan 0002's
  1321-slot route block sits between them and `O_SENSORY_MEAN`. Rates come from
  the same constants as the Rust table (`SCENT_WHITENING_RATE`,
  `VISION_PATHWAY_RATE`, refresh `VISION_WHITENING_REFRESH`). `SENSE_COUNT`
  defaults to `2u`; slots 2 and 3 default to `0u` / `0f32`.
- `pub fn new_with_senses(agent_count: u32, food_count: usize, brain_config:
  &BrainConfig, world_config: &WorldConfig, senses: &'static [SenseDescriptor])
  -> Self` takes the body of `new`, with `BrainLayout::with_senses(brain_config,
  senses)` in place of `from_config`. `new` delegates with `DECLARED_SENSES`.
- Conversions: no `as` casts. Use `u32::try_from(x).expect("...")` then
  `f64::from`, as the existing retina conversions do.
- Every new item has a `///` comment. Tests named `sense_override_*`, placed in
  `gpu_kernel.rs` `mod tests` and in `senses.rs` `mod tests`.

**Steps:**

1. **Code constants and kind methods.** In `senses.rs`, add the route-mask and
   kind-code `pub const`s with `///` comments, and the `const fn` methods
   `SenseInput::kind_code()`, `Whitening::kind_code()`,
   `LearningRate::gene_offset_or_zero()` and `SenseDescriptor::route_mask()`.
   Add `sense_override_kind_codes` asserting each method on every variant.
2. **Emitter.** In `gpu_kernel.rs`, add `insert_sense_overrides`, called from
   `vision_override_constants` after the `VISION_W` insert, which stays
   unchanged. It emits `SENSE_COUNT` and the 18 keys per slot `k < 4` from
   `layout.senses()` and `layout.sense_blocks()`, with `0` for unused slots.
3. **WGSL mirror and declarations.** In `common.wgsl`, add the mirrored `const`
   codes and route bits, `override SENSE_COUNT: u32` and the 72 per-slot
   overrides, each default derived as the Decisions section says and equal to
   the default table.
4. **Constructor.** Add `new_with_senses`, make `new` delegate with
   `DECLARED_SENSES`, and route through `BrainLayout::with_senses`.
5. **Table and mirror tests.** `sense_override_defaults_match_table` parses
   every `override SENSE` line of `COMMON_SRC` and asserts each literal equals
   the map value for the default layout, and that every `SENSE*` map key is
   declared. Its failure message prints the key, the expected map value and the
   declared literal. `sense_override_wgsl_mirrors_rust_codes` parses the
   mirrored `const SENSE_` lines and compares them with the Rust constants.
6. **Test-only extra sense.** `sense_override_test_sense_table` builds a static
   table of the declared senses plus a 2-channel `ClosedForm2` `Turn`-only sense
   and asserts `SENSE_COUNT` is 3, `SENSE2_*` equals `sense_blocks()[2]`, and
   slot 3 is all zero. `sense_override_extra_sense_composes` naga-parses and
   validates the composed brain shader for that table, following the existing
   composed-shader test pattern. Keep comments free of TODO, plan, task or
   decision names.

- **Done when:** INV-05 and INV-01 — the focused-proof check passes running
  `sense_override_kind_codes`, `sense_override_defaults_match_table`,
  `sense_override_wgsl_mirrors_rust_codes`, `sense_override_test_sense_table` and
  `sense_override_extra_sense_composes` (none skipped or edited), the brain-lib
  check passes, and the parity check passes with the golden unchanged, proving
  INV-05 (every per-sense parameter reaches WGSL from the layout, and defaults
  equal the table) and INV-01 (generated code and seeded trajectories
  byte-identical).

## Non-goals

- No WGSL reads of the new names, and no edit to `brain_passes.wgsl`.
- No edit to `buffers.rs`, `config.rs` or any file outside the three touches;
  the layout accessor and constructor belong to `derived-sense-layout`.
- No change to the value of any existing override, to `BRAIN_STRIDE`, or to the
  golden; no production sense, route or flag.
