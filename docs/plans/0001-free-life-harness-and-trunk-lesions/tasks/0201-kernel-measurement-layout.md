---
id: kernel-measurement-layout
title: Lay out the lesion and probe slots, types and decoder in Rust and WGSL
workstream: "0200"
kind: task
role: developer
depends_on: [seeded-headless-digest]
gated: false
touches:
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
status: done
merged_as: "a0fb08a0695064bd12b8c131373d0599c805a886"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib buffers::tests"
    timeout_ms: 3600000
  - name: brain-lib-tests
    command: "cargo test -p xagent-brain --lib"
    timeout_ms: 3600000
  - name: digest-strict-after-layout
    command: "env XAGENT_REQUIRE_GPU=1 XAGENT_REQUIRE_SEEDED_DIGEST=1 cargo test -p xagent-sandbox --test seeded_headless_digest -- --nocapture"
    timeout_ms: 3600000
  - name: brain-clippy
    command: "cargo clippy -p xagent-brain --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: brain-fmt
    command: "cargo fmt -p xagent-brain -- --check"
    fix: "cargo fmt -p xagent-brain"
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Lay out the lesion and probe slots, types and decoder in Rust and WGSL

Implements the Rust and WGSL items of ARCHITECTURE.md section
"Kernel measurement layout (kernel-measurement-layout)", proving INV-11 (and
keeping INV-1 bit-identical). Read that section for the slot table, probe
offsets and type signatures; do not copy them into source beyond the names.

Behaviour does not change: nothing writes the new slots or the probe block yet.
`gpu_kernel.rs` and every shader address the decision buffer and brain config
through the constants (`CONFIG_SIZE`, `DECISION_STRIDE`, `DECISION_MOTOR`), so
buffer sizes follow automatically. If any file outside `touches` hardcodes `16`,
the old stride or `array<vec4<f32>, 4>`, stop and report that as a blocker
instead of editing it.

Decisions (fixed, do not revisit): uniform grows to 6 vec4 (`CONFIG_SIZE = 24`);
`DECISION_PROBE = DECISION_MOTOR + 4`; `BRAIN_TICK_PROBE_LEN = 148`;
`DECISION_STRIDE = DECISION_PROBE + BRAIN_TICK_PROBE_LEN`. Every new Rust
constant has a WGSL twin with the same name and value.

Source rules: doc comments state what each slot or field does. Do not name any
plan, task, workstream or decision document in source or strings
(`contributing_guard.rs` enforces this).

**Steps:**
1. In `buffers.rs`, set `CONFIG_SIZE` to 24 and add `CFG_LESION_TURN_WEIGHTS` (16), `CFG_LESION_FORWARD_WEIGHTS` (17), `CFG_LESION_MEMORY_BLEND` (18), `CFG_LESION_CRITIC` (19) and `CFG_BRAIN_TICK_PROBE` (20), each documented with its meaning when `1.0`. Mirror them as `u32` constants in `common.wgsl` and change the `brain_config` binding to `array<vec4<f32>, 6>`.
2. Add `pub struct TrunkLesions` (four `bool` fields, derives `Clone, Copy, Debug, Default, PartialEq, Eq`, associated `NONE` and `ALL`, `is_none()`), `pack_trunk_lesions(cfg, lesions)` writing slots 16-19 as `1.0`/`0.0`, and `pack_brain_tick_probe(cfg, enabled)` writing slot 20. Padding slots 21-23 stay `0.0`.
3. Add `DECISION_PROBE`, `BRAIN_TICK_PROBE_LEN`, the redefined `DECISION_STRIDE`, and the `PROBE_*` offsets (`PROBE_ENCODING = 0`, `PROBE_PATHWAY_INPUTS = 128`, then `PROBE_REWARD` through `PROBE_ENCODING_FORWARD_OUTPUT = 147`) in Rust. In `common.wgsl` add `DECISION_PROBE`, `BRAIN_TICK_PROBE_LEN`, the new `DECISION_STRIDE` and the same `PROBE_*` values as literal `u32`s.
4. Add `pub struct BrainTickProbe` with the pub fields from the section, deriving `Clone, Copy, Debug, PartialEq`, and `from_decision_record(record: &[f32]) -> Option<Self>` returning `None` unless `record.len() >= DECISION_STRIDE`. Add `pub const TD_TRACE_DECAY: f32 = 0.97 * 0.9;` documented as mirroring `TD_DISCOUNT * TD_LAMBDA` in `common.wgsl`.
5. Add WGSL readers `lesion_turn_weights()`, `lesion_forward_weights()`, `lesion_memory_blend()`, `lesion_critic()` and `brain_tick_probe_enabled()`, each `bc_f32(<slot>) >= 0.5`, next to `steering_weights_learn()`.
6. Extend the `buffers::tests` module: add the new slots to `config_indices_within_bounds`, the new names to `shader_config_constants_match_rust` (parsed as `u32`), and assert `CONFIG_SIZE == 24`. Add `pack_trunk_lesions_places_each_switch_in_its_slot` (each switch alone, `NONE`, `ALL`, padding zero), `pack_brain_tick_probe_writes_only_slot_20`, `brain_tick_probe_decoder_places_each_field` (a record of length `DECISION_STRIDE` whose index `i` holds `i as f32`; every field equals its named offset, `from_decision_record` of length `DECISION_STRIDE - 1` is `None`), `wgsl_probe_layout_matches_rust` (parse the `PROBE_*` literals, and assert the exact WGSL text lines for `DECISION_PROBE` and `DECISION_STRIDE`) and `td_trace_decay_matches_wgsl_text` (parse `const TD_DISCOUNT: f32 = ...;` and `const TD_LAMBDA: f32 = ...;` from `common.wgsl` and assert `TD_TRACE_DECAY` equals their product within `1e-6`). The existing `shader_brain_config_array_size_matches_config_size` must pass unchanged at 6 vec4s. The `brain-lib-tests` check (`cargo test -p xagent-brain --lib`) must stay green with no existing test edited: every lib unit test that sizes the decision buffer or brain config does so through the constants. The crate-wide `cargo test -p xagent-brain` suite is not this task's check; it runs once as the PLAN.md `brain-crate-tests` integration check.

- **Done when:** INV-11 holds: `cargo test -p xagent-brain --lib buffers::tests` passes with the new layout, packer, decoder, WGSL-parity and TD-decay tests green, and `digest-strict-after-layout` still reproduces the baseline digest (INV-1).
