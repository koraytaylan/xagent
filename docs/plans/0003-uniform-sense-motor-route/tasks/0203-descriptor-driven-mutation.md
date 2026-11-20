---
id: descriptor-driven-mutation
title: "Steering mutation driven by the sense descriptor"
workstream: "0200"
kind: task
role: developer
depends_on:
  - derived-sense-layout
gated: false
status: done
merged_as: "e76884e763c5944f88d4a03f5e8990e6a9658129"
touches:
  - crates/xagent-sandbox/src/agent/mod.rs
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --lib agent::"
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

# 0203 — Steering mutation driven by the sense descriptor

## Outcome

In `crates/xagent-sandbox/src/agent/mod.rs`, steering mutation iterates the
declared sense table. `mutate_steering_weights_for(state, feature_count, senses,
strength, rng)` perturbs the action turn weights, then the `Mutated` route block
of each sense in declared sense order, then route order, rescaling each block
into the L2 ball. `mutate_steering_weights(state, strength, rng)` stays the public
entry point and calls the helper with `DECLARED_SENSES`. Default-configuration
draws and results stay bit-identical to today. No other file changes.

## Baseline facts

- `mutate_steering_weights` (about line 588) draws, in order: for each of
  `ENCODED_DIMENSION` action turn weights, a gate draw and, on a hit, a kick draw;
  then `rescale_turn_weights`. Then the scent block (`SCENT_CHANNELS`) and the
  vision block (`VISION_PATHWAY_INPUTS`), each channel gate-and-kick, then
  `rescale_into_ball` for that block.
- `brain_weight_offsets` maps an absolute layout slot `s` to
  `predictor_context + (s - O_PREDICTOR_CONTEXT_WEIGHT)`. Keep this mapping.
- Sense block offsets depend only on the sense table. The derived rule starts at
  the fixed `O_RECENT_NORMS + RECENT_CAP` base, so `feature_count` does not move
  them. If `with_senses` or `sense_blocks` differ from `derived-sense-layout`'s
  packet, stop as BLOCKED.
- Plan 0002 route slots (`O_FORWARD_PATHWAY_WEIGHTS`, `O_CRITIC_PATHWAY_WEIGHTS`
  and their traces) are never written by mutation.

## Decisions

- Helper: `pub fn mutate_steering_weights_for(state: &AgentBrainState,
  feature_count: usize, senses: &'static [SenseDescriptor], strength: f32, rng: &mut
  impl Rng) -> AgentBrainState`, same visibility and style as its neighbours.
- Per-sense turn offsets: from `BrainLayout::with_senses(&BrainConfig::default(),
  senses).sense_blocks()`, mapped by the `predictor_context` rule above.
  `BrainWeightOffsets` loses its scent and vision fields.
- Mutated routes only: a route with `Heritability::Mutated` must be
  `MotorRoute::Turn`. Assert this before any draw, with a message naming the
  sense. Only `Turn` blocks are mutated.
- Draw order: action turn loop, then each sense in slice order, each `Mutated`
  route in `routes` order, its channel loop (gate draw, then kick draw on a hit),
  then `rescale_into_ball` on that block. For `DECLARED_SENSES` this is the same
  RNG call sequence as today.
- Constants `STEERING_MUTATION_FRACTION`, `STEERING_WEIGHT_STEP` and
  `MAX_WEIGHT_NORM` keep their values. The hand-listed `(start, len)` array is
  deleted.
- Source comments name no plan, task or decision document. No `TODO` or `FIXME`.
  Every new item has a `///` comment. The later `modulator-flags` task edits this
  file's `BrainConfig` literals, so keep the `critic_input` and
  `forward_pathway_enabled` values as plan 0002 wrote them.

**Steps:**

1. **Helper and offsets.** Add `mutate_steering_weights_for` and per-sense turn
   offset derivation (see Decisions). Make `mutate_steering_weights` compute
   `feature_count_for_brain_stride(state.brain_state.len())` and delegate with
   `DECLARED_SENSES`. Import `DECLARED_SENSES`, `Heritability`, `MotorRoute`,
   `SenseDescriptor` and `BrainLayout` from `xagent_brain`. Remove any import or
   helper left unused.
2. **Draw order.** Keep the action turn loop and `rescale_turn_weights` exactly.
   Then implement the sense and route loops and the `Turn`-only assertion, in the
   order given in Decisions.
3. **Reference test.** Add `mutate_steering_weights_matches_two_block_reference`:
   for seeds 1 to 8 and strengths 0.1 and 0.5, a literal reference inside the test
   performs the pre-change draws (action turn loop, then the scent block at
   `O_SCENT_TURN_WEIGHTS` and the vision block at `O_VISION_TURN_WEIGHTS`, each
   rescaled). Assert `brain_state` equal under `f32::to_bits`, and assert the
   reference differs from the template so the check is not vacuous.
4. **Learned-only test.** Add `learned_only_sense_stays_zero_under_mutation`: a
   test-only `const` sense with 2 channels, `ClosedForm2`, and one `Turn` route
   that is `LearnedOnly`. Build `BrainLayout::with_senses(&config, &[SMELL,
   VISION, EXTRA])`, seed with `init_brain_state_for`, mutate through the helper
   with that layout's `feature_count`. Assert the extra block is exactly `0.0`
   before and after, and the smell and vision turn blocks change.
5. **Route and assertion tests.** Add `forward_and_critic_route_blocks_are_never_mutated`
   (config with `critic_input` and `forward_pathway_enabled` set explicitly;
   plan 0002 route slots set to nonzero values; bit-identical after mutation) and
   `mutated_non_turn_route_is_rejected` (`#[should_panic]` with the assertion's
   message, using a test-only base table whose sense has a `Mutated` `Forward`
   route). Existing tests `mutate_steering_weights_touches_only_the_turn_policy`
   and `steering_population_shares_a_genome_within_each_repeat_group` stay
   unchanged.
6. **Hygiene.** Confirm `agent/mod.rs` has no `TODO`, `FIXME`, plan, task or
   decision name in comments or strings, that every new item has a `///` comment,
   and that every new numeric literal is a named constant or `0`/`1`/`2.0`
   matching the existing kick code.

- **Done when:** INV-03 and INV-01 — the focused-proof check passes, running
  `agent::tests` with `mutate_steering_weights_matches_two_block_reference`,
  `learned_only_sense_stays_zero_under_mutation` and
  `mutated_non_turn_route_is_rejected` included and the existing tests unchanged,
  and the parity check stays green on the unchanged golden with no re-record,
  including the CPU `steering_population` scenario. This proves INV-03 (smell and
  vision turn weights mutated in the same draw order, other routes zero and never
  mutated) and INV-01 (default trajectories and the steering draw byte-identical).

## Non-goals

- No edit to `senses.rs`, `buffers.rs`, WGSL, `config.rs`, `governor.rs` or any
  file outside `touches`. The descriptor and layout APIs are read, never changed.
- No change to RNG draw order, kick size, mutation fraction, the `Mutated` set
  (smell and vision turn only), or the golden file.
- No new sense, route or flag in production; the extra sense exists only in tests.
