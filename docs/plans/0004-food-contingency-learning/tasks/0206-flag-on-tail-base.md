---
id: flag-on-tail-base
title: "Fixed-tail base and feature count accept the second-odour brain stride"
workstream: "0200"
kind: task
role: developer
depends_on:
  - second-odour-inheritance
gated: false
touches:
  - crates/xagent-brain/src/buffers.rs
status: done
merged_as: "e19ef98db003ee9c46252c351b1501864733cd14"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-brain --lib buffers::"
    timeout_ms: 3600000
---

# Fixed-tail base and feature count accept the second-odour brain stride

## Outcome

`buffers::feature_count_for_brain_stride` and `buffers::fixed_tail_base`
accept the `brain_stride` of a layout built with
`second_odour_sense_enabled = true`, so the landed free-life seeding
(`GpuKernel::write_agent_heritable_config`, which calls `fixed_tail_base`)
no longer panics with 'brain_state length matches no BrainLayout' on a
flag-on kernel. Task food-kind-free-life needs this (AUTHORIZATION.md,
repair entry "flag-on fixed-tail base"); task second-odour-inheritance only
fixed the private steering inference in `agent/mod.rs`.

## Scope

- `crates/xagent-brain/src/buffers.rs` only: make the length-based inference
  also recognise a stride longer by the second-odour sense slots (derive the
  extra slot count from `BrainLayout::from_config` of the default and the
  flag-on config, not a literal), and add unit tests in the existing
  `buffers` test module:
  - `flag_on_stride_infers_feature_count`: for the flag-on layout `L`,
    `feature_count_for_brain_stride(L.brain_stride) == Some(L.feature_count)`.
  - `flag_on_fixed_tail_base_matches_layout`: `fixed_tail_base(L.brain_stride)`
    equals the offset at which `L` places `O_PREDICTOR_CONTEXT_WEIGHT`
    (as `init_brain_state_for` lays the state out).
  - `default_stride_inference_unchanged`: for feature counts 1..=64 the
    default-stride results are identical to the formula before this task, and
    a stride matching neither layout still gives `None`.

## Non-goals

- Default-config results stay bit-identical; no signature change to either
  function; no edit outside `buffers.rs`; no path writes the `second_odour`
  block.

## Steps

**Steps:**

1. Extend the inference as above.
2. Add the three tests.

- **Done when:** the focused-proof check passes with `flag_on_stride_infers_feature_count`, `flag_on_fixed_tail_base_matches_layout` and `default_stride_inference_unchanged` ok.
