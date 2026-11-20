---
id: kernel-routes
title: "Kernel critic input selection and forward pathway"
workstream: "0200"
kind: task
role: developer
depends_on:
  - brain-slot-layout
gated: false
status: done
merged_as: "74cba2d3f6afff0d759121e1b40cecd4e3b93889"
touches:
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl
  - crates/xagent-brain/src/shaders/kernel/phase_death.wgsl
  - crates/xagent-brain/README.md
  - crates/xagent-sandbox/tests/pathway_routes.rs
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_routes"
    timeout_ms: 3600000
  - name: routes-off-fingerprint
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_default_fingerprint"
    timeout_ms: 3600000
  - name: vision-pathway
    command: "cargo test -p xagent-brain --test vision_pathway"
    timeout_ms: 3600000
  - name: brain-lib
    command: "cargo test -p xagent-brain --lib"
    timeout_ms: 3600000
  - name: sandbox-integration
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test integration"
    timeout_ms: 7200000
  - name: crate-fmt
    command: "cargo fmt -p xagent-brain -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-brain -p xagent-sandbox"
  - name: crate-clippy
    command: "cargo clippy -p xagent-brain -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: contributing-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
    timeout_ms: 3600000
---

# Kernel critic input selection and forward pathway

## Outcome

With `critic_input` `pathway` or `encoding_and_pathway`, the critic reads the ten
whitened pathway inputs of the previous brain tick. With `forward_pathway_enabled`
on, the forward command gains a learned pathway term. With both routes off, no new
slot is read or written. Formulas are fixed in ARCHITECTURE.md D5 (nothing moves),
D6 (critic route) and D7 (forward route); implement them there, do not restate them.

## Prerequisites

- `brain-slot-layout` integrated: `CFG_CRITIC_INPUT`, `CFG_FORWARD_PATHWAY_ENABLED`,
  the block offsets and `PATHWAY_INPUT_COUNT` in `common.wgsl`. Use names, never
  indices. A missing name is a stop-and-report.
- Plan 0001 integrated: `CFG_LESION_TURN_WEIGHTS`, `CFG_LESION_FORWARD_WEIGHTS`,
  `CFG_LESION_MEMORY_BLEND`, `CFG_LESION_CRITIC`, `CFG_BRAIN_TICK_PROBE`. Define none.
- Mode constants local to `brain_passes.wgsl`: `CRITIC_INPUT_ENCODING` (0u),
  `CRITIC_INPUT_PATHWAY` (1u), `CRITIC_INPUT_ENCODING_AND_PATHWAY` (2u), read by
  `let m = u32(bc_f32(CFG_CRITIC_INPUT));`. A name collision is a stop-and-report.

## Pinned rules

1. Uniform branches only, on `m` and `CFG_FORWARD_PATHWAY_ENABLED`. Routes off stays
   bit-identical (INV-1): new branches are skipped and existing expressions keep order.
2. Critic input `c` and forward input `x` follow D5 exactly.
3. `s_pred_td` grows to 6 entries; `S_CRITIC_NORMALIZER = 5u` (D6). The turn channel
   always keeps `S_STEP_NORMALIZER`.
4. Balls (D6, D7): the ten critic pathway weights and each forward sense have their
   own `MAX_WEIGHT_NORM` ball. The forward weight step is not gated by
   `steering_weights_learn()`.
5. Existing-line edits are limited to the normaliser reads and the pathway-mode gates
   on the 128-d critic value dot, weight step, trace, replay and death step. Use
   `select` or uniform branches so encoding mode keeps its arithmetic. The 128-d
   forward term and forward lesion text stay untouched.
6. Death paths are inline in `kernel_tick.wgsl` and `phase_death.wgsl`, mirroring the
   terminal critic step; read the normaliser before `O_PREV_ENCODED` is zeroed. No
   helper in `common.wgsl`. Death keeps `O_PREV_PATHWAY_INPUT_NORM`, the ring and weights.
7. Lesions: `CFG_LESION_CRITIC` drops the whole value's pathway part in TD and replay.
   `CFG_LESION_FORWARD_WEIGHTS` drops only the 128-d forward term. The steering freeze
   does not cover the forward pathway.
8. No source comment names a plan, task, workstream or decision, or uses a D-label (INV-12).
9. Markers (INV-13): move, duplicate and re-indent no text in any file that
   `compose_brain_passes` or a `gpu_kernel` test matches. Keep text, count and order
   of `RECALL_DEFINITION`, `REFRESH_BLOCK`, `EXPLORATION_BOUNDARY`, `ORIGINAL_CALL`,
   `REFRESH_CALL`, the `s_reinf_dot` lines, `if (tid == 0u) { s_value = s_dense_partials[0]; }`,
   the `// ── Thread 0: exploration, noise, motor, telemetry` comment,
   `brain_state[brain_base + O_SCENT_WHITENED] = scent_whitened.x;`,
   `s_explore[0u] = noise_forward * exploration_rate * fatigue_factor;` and
   `+ s_explore[2u] * input / (1.0 + vision_sq);`. New text must not contain these,
   `fn compose_brain_passes(` or `vision_pathway_step(`. If a lib test fails, rephrase
   the new WGSL; never edit `gpu_kernel.rs` or `gpu_kernel/*.rs`.

## Non-goals

- No move or split of the value reduction, whitening or `vision_pathway_step` call.
- No change to turn policy, encoder, reward, lesion or config defaults, mutation,
  `common.wgsl`, `buffers.rs`, or any Rust file except `tests/pathway_routes.rs`.
- No new uniform slot, gene or inheritance write. A fingerprint change is a defect,
  not a reason to regenerate the 0101 fixture.

## Tests

`pathway_routes.rs` follows the `XAGENT_REQUIRE_GPU` convention of `tests/integration.rs`
and the probe style of `tests/vision_pathway.rs`. Each test reads `brain_state` before and
after one tick, sets `critic_input` and `forward_pathway_enabled` explicitly (INV-14),
asserts values, and is not ignored. Nine tests:

- `routes_off_keep_route_block_zero` (INV-1, INV-4): flags pinned to `Encoding` and
  `false`; route block stays zero over many ticks.
- `forward_rule_matches_d7_one_tick` (INV-6): trace and weight step equal D7; senses stay in balls.
- `forward_pathway_term_moves_command` (INV-6): `P_MOTOR_FWD_OUT` differs from the zero-weights run.
- `forward_death_settles_and_clears_traces` (INV-6): death settles, then zeroes forward traces.
- `critic_encoding_mode_zero_pathway_slots` (INV-7): value equals the baseline; pathway slots stay zero.
- `critic_pathway_mode_freezes_value_weights` (INV-7): value is bias plus pathway dot; `O_VALUE_WEIGHTS` unchanged.
- `critic_encoding_and_pathway_matches_d6` (INV-7): value and normaliser equal D6.
- `ring_pathway_keys_equal_stored_input` (INV-7): the store-slot key equals the stored `c`.
- `critic_death_zeroes_both_traces` (INV-7): both critic traces are zero after death; observed after the respawn tick with the stored pathway inputs set to zero, so all ten pathway trace entries read zero and all 128 `O_TRACE_CRITIC` entries minus the new life's first term read zero.

## Steps

**Steps:**
1. Critic value (D5, D6), `brain_passes.wgsl`: declare mode constants and
   `s_critic_pathway_input` as module-scope `var<private>` (never
   `var<workgroup>`: brain-lib's shared-memory bounds forbid it; see
   AUTHORIZATION.md); copy `c` and add the pathway dot in one uniform branch
   after the `s_value` line; gate the 128-d value dot by `[m != pathway]`.
2. Critic normaliser, step, trace, replay (D6), `brain_passes.wgsl`: grow `s_pred_td`;
   gate the 128-d step and trace; add the pathway step, trace, ball and
   `O_PREV_PATHWAY_INPUT_NORM` rewrite; write keys in the store pass before the
   existing `storageBarrier()`; gate replay and select the key norm.
3. Forward route (D7), `brain_passes.wgsl`: under `CFG_FORWARD_PATHWAY_ENABLED`, add the
   command term before the memory blend; weight step beside the smell turn-weight step
   using the pre-update trace; trace update after `s_explore[0u]` is assigned.
4. Death paths (D6, D7), `kernel_tick.wgsl` and `phase_death.wgsl`: inline the terminal
   normaliser, apply `TERMINAL_DEATH_TD_ERROR` through pathway and forward traces with
   balls, gate the 128-d terminal step and trace zeroing, then zero all four pathway traces.
5. Tests: write `crates/xagent-sandbox/tests/pathway_routes.rs` with the nine tests above.
6. README and checks: update `crates/xagent-brain/README.md` for the new slots and routes
   with no plan, task or decision references. Request host checks in this order:
   `focused-proof`, `routes-off-fingerprint`, `vision-pathway`, `brain-lib`,
   `sandbox-integration`, `crate-fmt` (fix on failure), `crate-clippy`,
   `contributing-guard`. Fix findings only inside `touches`.

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_routes` passes all nine named tests with none ignored, including `routes_off_keep_route_block_zero`, `forward_rule_matches_d7_one_tick` and `critic_encoding_and_pathway_matches_d6`; the routes-off-fingerprint check still matches the baseline with both routes off; and the brain-lib check passes with the composition markers unchanged (proves INV-1, INV-4, INV-6, INV-7, INV-12, INV-13, INV-14).

## Evidence and risks

- Baseline facts and the post-death input carry-over (D3) are in ARCHITECTURE.md; report any test that contradicts them.
