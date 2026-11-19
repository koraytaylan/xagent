---
id: second-odour-inheritance
title: "Second-odour block weights stay zero through birth, mutation and inheritance"
workstream: "0200"
kind: task
role: developer
depends_on:
  - extra-sensory-layout
gated: false
touches:
  - crates/xagent-sandbox/tests/second_odour_inheritance.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --test second_odour_inheritance"
    timeout_ms: 3600000
  - name: agent-unit
    command: "cargo test -p xagent-sandbox --lib agent::"
    timeout_ms: 3600000
---

# Second-odour block weights stay zero through birth, mutation and inheritance

## Outcome

A CPU-only integration test shows that, with `second_odour_sense_enabled` on,
the `second_odour` block's turn weights and trace are exactly `0.0` in every
birth brain and after every heritable path the sandbox applies to a brain:
steering mutation, steering population (the governor's per-generation
offspring), and whole-brain mutation. Smell and visual turn weights still
mutate, so the test is not vacuous (ARCHITECTURE.md INV-05: learned-only and
zero at birth; non-zero only through within-life learning).

## Why no product edit is expected

0003's `mutate_steering_weights` kicks only `Mutated` `Turn` routes of
`DECLARED_SENSES`, whose offsets sit at a fixed tail base that the extra
feature width does not move (0003 task 0203), and `mutate_brain_state_seeded`
kicks only the encoder, forward/turn action and predictor regions
(`crates/xagent-sandbox/src/agent/mod.rs`). The `second_odour` block is
`LearnedOnly` and lies after the 0003 route block, so no path writes it. This
task proves that; it edits no product file.

## Scope

- `crates/xagent-sandbox/tests/second_odour_inheritance.rs` (new), CPU only,
  using public items: `xagent_sandbox::agent::{fresh_brain_state,
  mutate_steering_weights, steering_population, mutate_brain_state_seeded}`,
  `xagent_brain::buffers::BrainLayout`, and `BrainConfig` with
  `second_odour_sense_enabled = true`. Block slots come from
  `BrainLayout::from_config(&config).sense_blocks()[2]` (turn weights and trace),
  indexed into `brain_state` exactly as the `extra-sensory-layout` test
  `second_odour_initial_state_is_zero_weights` indexes them. Seeds are named
  constants; use `rand::rngs::SmallRng::seed_from_u64`.

## Non-goals

- No edit to `agent/mod.rs`, the governor or any other product file. If a path
  writes the block, the test fails: report the function and line; do not loosen
  the assertion.
- No GPU; no planning vocabulary or `TODO` in the file.

## Steps

**Steps:**

1. Create the file with a module doc comment, the named seeds and a helper
   `second_odour_block(state, layout) -> Vec<f32>` returning the block's turn
   weights and trace.
2. `second_odour_birth_brain_is_zero`: for 8 seeds, `fresh_brain_state` gives a
   block of exactly `0.0`.
3. `second_odour_survives_steering_mutation`: for 8 seeds and strengths 0.1 and
   0.5, `mutate_steering_weights` leaves the block exactly `0.0` while at least
   one smell or visual turn weight changes (assert both).
4. `second_odour_survives_population_and_brain_mutation`: `steering_population`
   with 12 agents and group size 3 gives every genome a zero block, and
   `mutate_brain_state_seeded` at strength 0.5 for 8 seeds keeps it `0.0`.

- **Done when:** the focused-proof check passes with `second_odour_birth_brain_is_zero`, `second_odour_survives_steering_mutation` and `second_odour_survives_population_and_brain_mutation` ok and agent-unit stays green, proving INV-05's heritability clause: the `second_odour` block weights are zero in every birth brain and stay exactly zero through steering mutation, the steering population and whole-brain mutation, while smell and visual turn weights still mutate.
