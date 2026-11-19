---
id: "modulator-flags"
title: "Flags for the klinotaxis turn-gain and motor-fatigue modulators"
workstream: "0400"
kind: task
role: developer
depends_on:
  - descriptor-driven-mutation
  - remove-sense-specific-wgsl
gated: false
status: todo
merged_as: ""
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-brain/src/buffers.rs
  - crates/xagent-brain/src/shaders/kernel/common.wgsl
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-sandbox/src/agent/mod.rs
  - crates/xagent-sandbox/tests/modulator_flags.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test modulator_flags -- --include-ignored"
    timeout_ms: 3600000
  - name: config-tests
    command: "cargo test -p xagent-shared"
    timeout_ms: 1800000
  - name: buffer-tests
    command: "cargo test -p xagent-brain --lib -- buffers::tests::"
    timeout_ms: 3600000
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

# 0401 — Flags for the klinotaxis turn-gain and motor-fatigue modulators

## Outcome

Two independent `BrainConfig` switches, `klinotaxis_turn_gain_enabled` and
`motor_fatigue_enabled`, each default `true`. Switch on: the brain is
byte-identical to today. Switch off: the matching factor is exactly `1.0` in the
executed motor command and in the credited exploration noise. Default runs stay
byte-identical to the `parity-golden` recording.

## Baseline facts

- Baseline `CONFIG_SIZE = 16` (`crates/xagent-brain/src/buffers.rs`), every slot
  `0..=15` used. Plans 0001 and 0002 grow the config first, so slot numbers are
  not fixed here. Rule at task start: let `H` be the highest `CFG_*` index in
  `buffers.rs` (expected 22). The flags take `H + 1` and `H + 2` (expected 23
  and 24). `CONFIG_SIZE` becomes the smallest multiple of 4 greater than the new
  highest index (expected 28). Confirm against integrated code and the existing
  size and index tests before writing constants.
- `common.wgsl` declares `brain_config` as `array<vec4<f32>, N>` (binding 14),
  `N = CONFIG_SIZE / 4` before this task. It grows by one vec4 to match.
- Factors in `brain_passes.wgsl`, located by text (line numbers move):
  `var fatigue_factor: f32 = 1.0;` followed by `if (p_len >= 4u) {` (staleness
  block); `let klinotaxis_factor = clamp(1.0 - gradient_deviation * KLINOTAXIS_SENSITIVITY, 0.3, 3.0);`
  followed by `turn *= klinotaxis_factor;`; and the `s_explore[...]` lines that
  multiply by `fatigue_factor` and `klinotaxis_factor`. The writes to
  `O_FATIGUE_FACTOR` and `P_FATIGUE_FACTOR_OUT` must keep happening.
- `mutate_config_with_strength_rng` and `crossover_config` in
  `crates/xagent-sandbox/src/agent/mod.rs` use full `BrainConfig` literals, so the
  new fields need explicit pass-through lines there. This is why `agent/mod.rs`
  is in `touches` and why `descriptor-driven-mutation` is in `depends_on`.

## Decisions

- Both flags use `#[serde(default = "default_modulator_gate_enabled")]`, returning
  `true`. They are population-wide runtime switches, not heritable:
  `mutate_config_with_strength_rng` copies them from `parent`,
  `crossover_config` from `a`.
- Gating is by branch, not multiplication. Fatigue off: the staleness branch is
  skipped and `fatigue_factor` stays `1.0`; the position-ring update before it
  still runs. Klinotaxis off: `klinotaxis_factor` stays `1.0` and
  `KLINOTAXIS_SENSITIVITY` is not read. Flag on: float expressions unchanged.
- Flags never move `brain_state` slots, so stored databases load either way.
- Slots: `CFG_KLINOTAXIS_TURN_GAIN_ENABLED = H + 1`, `CFG_MOTOR_FATIGUE_ENABLED = H + 2`.

**Steps:**

1. **Config fields and serde.** In `crates/xagent-shared/src/config.rs`, add two
   `pub` bool fields to `BrainConfig` after its last field, each with a doc
   comment saying it is a population-wide runtime switch, not heritable, default
   `true`. Add `fn default_modulator_gate_enabled() -> bool { true }` beside the
   other `default_*` functions. Set both `true` in `impl Default for BrainConfig`
   and in `tiny()` and `large()` if they use literals without
   `..Default::default()`. In `crates/xagent-sandbox/src/agent/mod.rs`, add
   `klinotaxis_turn_gain_enabled: parent.klinotaxis_turn_gain_enabled` and
   `motor_fatigue_enabled: parent.motor_fatigue_enabled` to
   `mutate_config_with_strength_rng`, and the same from `a` to `crossover_config`.
   Add a `config.rs` test that a JSON blob omitting both fields deserializes to
   `true`, and that `BrainConfig::default()` and every preset have both `true`.
   Fix any other full `BrainConfig { … }` literal in `config.rs` by adding both
   as `true`.
2. **CFG slots, packing, Rust mirrors.** In `crates/xagent-brain/src/buffers.rs`,
   add `pub const CFG_KLINOTAXIS_TURN_GAIN_ENABLED: usize` and
   `pub const CFG_MOTOR_FATIGUE_ENABLED: usize` after the highest existing `CFG_*`
   constant, with values from the Baseline rule, each with a doc comment naming
   its WGSL mirror. Set `CONFIG_SIZE` per the rule. In `build_config_for`, pack
   `1.0` or `0.0` at each slot from the booleans. Extend
   `config_indices_within_bounds` and `shader_config_constants_match_rust` with
   both names, and update the `brain_config` vec4-count check to the new size.
3. **WGSL mirrors and shader branches.** In `common.wgsl`, after the highest
   existing `CFG_*` constant, add `const CFG_KLINOTAXIS_TURN_GAIN_ENABLED: u32`
   and `const CFG_MOTOR_FATIGUE_ENABLED: u32` with the Rust values and a comment
   that they mirror `buffers.rs`. Grow the `brain_config` binding array by one
   vec4. In `brain_passes.wgsl`, before `var fatigue_factor: f32 = 1.0;` add
   `let motor_fatigue_on = bc_f32(CFG_MOTOR_FATIGUE_ENABLED) != 0.0;` and change
   `if (p_len >= 4u) {` to `if (motor_fatigue_on && p_len >= 4u) {`. Replace the
   klinotaxis `let` with `var klinotaxis_factor: f32 = 1.0;`,
   `let klinotaxis_on = bc_f32(CFG_KLINOTAXIS_TURN_GAIN_ENABLED) != 0.0;`, and
   `if (klinotaxis_on) { klinotaxis_factor = clamp(…); }` around the existing
   clamp. Leave `turn *= klinotaxis_factor;` and the `s_explore` lines unchanged.
4. **GPU and CPU proof file, then named checks.** Create
   `crates/xagent-sandbox/tests/modulator_flags.rs` with these tests:
   - CPU, normal: `default_modulator_flags_are_on`,
     `config_without_modulator_fields_defaults_on`,
     `build_config_packs_modulator_flags` (values at the two new `CFG_*` slots are
     `1.0` by default, `0.0` when off), and
     `mutation_and_crossover_keep_modulator_flags` (`mutate_config_seeded` and
     `crossover_config` on a parent with both flags `false` return both `false`).
   - GPU, `#[ignore]`, each asserting `GpuKernel::is_available()`:
     `motor_fatigue_off_pins_factor_to_one` runs 3000 ticks with
     `motor_fatigue_enabled = false`, sampling `fatigue_factor` via
     `read_agent_telemetry_blocking` for every agent at several points; every
     sample must be exactly `1.0`. The same seeded run with the default flag must
     produce at least one sample below `1.0`, or the test fails.
     `klinotaxis_off_pins_turn_gain_to_one` uses two runs from one seed, plants an
     energy drop so the gradient deviation drives `klinotaxis_factor` to a clamp,
     runs one brain tick, and reads `motor_turn` and `fatigue_factor`. With `on`
     and `off` the two runs' `motor_turn` values, the ratio `off / on` must be
     `≈ 1/0.3` or `≈ 1/3.0` (1e-5 relative), and the two `fatigue_factor` values
     must be equal. Use the stepping pattern in
     `crates/xagent-brain/tests/executed_action_credit.rs`.
   Each GPU test must fail if its flag branch in step 3 is deleted. Then make the
   frontmatter checks pass (`focused-proof`, `config-tests`, `buffer-tests`,
   `parity` with default flags, `fmt`, `clippy`, `guard`), fixing only failures
   caused by this task's edits to `touches`.

- **Done when:** the `focused-proof` check (`modulator_flags`, GPU tests included) passes with defaults on, old configs defaulting on, and flag-off runs giving factor exactly `1.0` in the executed command and the credit, and `parity` stays green with default flags, proving INV-07.

## Non-goals

- No default flip or golden re-record here; that belongs to `modulator-study`.
- No change to `O_FATIGUE_FLOOR`, `KLINOTAXIS_SENSITIVITY`, the position ring or
  the exploration noise beyond the gates above.
- No change to heritable genes, `mutate_brain_state`, or files outside `touches`.
