---
id: kernel-lesions-and-probe
title: Apply the trunk lesions and write the brain-tick probe in the fused brain pass
workstream: "0200"
kind: task
role: developer
depends_on:
  - kernel-measurement-layout
gated: false
touches:
  - crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/tests/trunk_lesions.rs
  - crates/xagent-brain/tests/brain_tick_probe.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test trunk_lesions --test brain_tick_probe"
    timeout_ms: 3600000
  - name: brain-kernel-lib-tests
    command: "cargo test -p xagent-brain --lib gpu_kernel"
    timeout_ms: 3600000
  - name: brain-policy-tests
    command: "cargo test -p xagent-brain --test scent_pathway --test vision_pathway --test turn_exploration --test executed_action_credit --test episodic_memory --test recent_replay"
    timeout_ms: 3600000
  - name: digest-strict-after-lesions
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
---

# Apply the trunk lesions and write the brain-tick probe

Binding spec: `ARCHITECTURE.md` sections "Lesion branches and probe
(kernel-lesions-and-probe)" and "Kernel measurement layout", invariants INV-2,
INV-3, INV-4 (INV-1 is held by the `digest-strict-after-lesions` check), and the
Feasibility row. The slots, `TrunkLesions`, `BrainTickProbe`, the packers and
`DECISION_PROBE` offsets come from `kernel-measurement-layout`; use those names
and `BrainLayout`/buffer constants only, no literal offsets.

Scope: edit only the four `touches` files. If a change needs any other file,
stop and report it as a blocker. No plan, task or workstream names in source or
comments.

**Steps:**
1. In `brain_passes.wgsl` `coop_predict_and_act`, wrap the four policy sites with
   uniform `select` or `if` on the `lesion_*()` readers: turn without
   `s_turn_dot` when `lesion_turn_weights()`; forward without `s_forward_dot`
   when `lesion_forward_weights()` (bias kept); skip the memory blend block when
   `lesion_memory_blend()`; `value` is `O_VALUE_BIAS` alone when
   `lesion_critic()`. Apply the same critic rule to `replay_value` in
   `coop_learn_and_store`. Keep every barrier reached by all invocations; with
   all flags `0.0` the arithmetic must be unchanged.
2. In `gpu_kernel.rs`, add fields `trunk_lesions: TrunkLesions` and
   `brain_tick_probe: bool` (off in `new`). `packed_brain_config` packs them with
   `pack_trunk_lesions` and `pack_brain_tick_probe`. Add `set_trunk_lesions`,
   `set_brain_tick_probe` (write their slots immediately, as
   `set_freeze_steering_weights` does), the two getters, and
   `read_brain_tick_probes_blocking` (one copy and map of the decision buffer,
   decoded per agent with `BrainTickProbe::from_decision_record`; on any failure
   log and leave `out` empty). Nothing is added to `BrainConfig` or
   `AgentBrainState`.
3. At the end of `coop_predict_and_act`, when `brain_tick_probe_enabled()`,
   write the probe block at `DECISION_PROBE` per the table in the spec. The
   pathway inputs are read from `O_SCENT_WHITENED` and `O_VISION_PATHWAY_INPUT`
   exactly where the trace block reads them. The vision input is written
   earlier in the tick (`brain_passes.wgsl` near 922 and
   `brain_cooperative_visual_pathway.wgsl`); if you find any writer that runs
   after `coop_predict_and_act`, stop and report a blocker. Value is the
   post-lesion value; forward and turn noise are `s_explore` entries; the turn
   and forward outputs are the computed dots even when lesioned.
4. Write `tests/trunk_lesions.rs`, following `tests/scent_pathway.rs` style:
   `defaults_off_and_survive_reset` (flags false at `new`, true after
   `set_trunk_lesions` survives `reset_agents_seeded`);
   `none_matches_unset_bit_for_bit` (two identically seeded kernels, one with
   `NONE` set explicitly, give identical full state); and per switch
   `each_lesion_removes_only_its_term`, comparing two identically seeded kernels
   that differ only in that lesion. Plant turn/forward weights with norm well
   below `MAX_WEIGHT_NORM` so the L2 ball does not rescale them. Add
   `critic_lesion_value_is_value_bias` (probe value equals `O_VALUE_BIAS`) and
   `memory_lesion_skips_recalled_blend` with planted recalled patterns.
5. Write `tests/brain_tick_probe.rs`: `probe_is_passive_and_exact` (probe on and
   off give bit-identical physics and brain state after the same ticks);
   `probe_fields_equal_brain_values` (value equals `O_PREV_VALUE`, `td_error`
   equals decision motor slot 3, pathway inputs equal `O_SCENT_WHITENED` and
   `O_VISION_PATHWAY_INPUT` after the tick, encoding equals the centred
   `s_memory_key` the critic reads, forward dot reads uncentred `s_encoded`);
   `readback_returns_agent_count_entries`.
6. Keep existing tests green unchanged: with every flag off the arithmetic is
   today's, so the `brain-kernel-lib-tests` check (the `gpu_kernel` lib tests)
   and the `brain-policy-tests` check (the scent, vision, turn-exploration,
   executed-credit, episodic-memory and replay integration tests that exercise
   the four policy sites) must pass with
   no existing test edited; a failure there is a defect in this change, not a
   test to update. The crate-wide `cargo test -p xagent-brain` suite is not
   this task's check; it runs once as the PLAN.md `brain-crate-tests`
   integration check.

- **Done when:** INV-2, INV-3 and INV-4 hold: `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test trunk_lesions --test brain_tick_probe` passes with the tests named in the Steps and no test skipped.
