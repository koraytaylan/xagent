# Authorization record

## Corrected requirement: `witness-config-size` baseline (2026-11-19)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: preflight `witness-config-size`, the `brain-slot-layout`
  constraint check of the same name, and `preflight-brain-slot-layout` required
  `pub const CONFIG_SIZE: usize = 16;` in `crates/xagent-brain/src/buffers.rs`.
- Contradiction: plan 0001 (landed in `abda4da`) added `CFG_LESION_*` and
  `CFG_BRAIN_TICK_PROBE` at slots 16-20 and grew `CONFIG_SIZE` to 24, exactly
  as ARCHITECTURE.md and task 0103 already anticipate; the witness therefore
  failed on the real baseline.
- Correction: the three checks now require
  `pub const CONFIG_SIZE: usize = 24;` (source: `buffers.rs` line with
  `CONFIG_SIZE: usize = 24`, slots 16-20 used, 21-23 padding). Task 0103's
  expected slot assignment (21, 22, size unchanged) is unaffected; no bound,
  task, dependency or behavior changed.
- Witness: `grep -q -F 'pub const CONFIG_SIZE: usize = 24;' crates/xagent-brain/src/buffers.rs`
  exits 0; all 58 preflight and constraint checks pass on the baseline.

## Corrected requirement: `s_critic_pathway_input` storage class (2026-11-19)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (task kernel-routes, plan contract blocker).

- Superseded: ARCHITECTURE.md D5 ("a new workgroup array
  `s_critic_pathway_input`") and the State table row
  "`s_critic_pathway_input` (workgroup, `PATHWAY_INPUT_COUNT`)".
- Contradiction: brain-lib, required by task 0201's Done when, pins total
  workgroup memory: `packed_encoder_shared_estimate_tracks_shader_declarations`
  asserts `FIXED_WORKGROUP_BYTES = 6_064` (gpu_kernel/packed_encoder.rs:25) and
  `four_dispatch_shared_bounds_follow_composed_declarations` asserts
  `RAW_MAIN_SHARED_BOUND = 9_200` (gpu_kernel/four_dispatch_validation.rs:44).
  A 10-float workgroup array yields 6_112 and 9_248, and both files are outside
  the task's touches; no in-budget workgroup array is free across the
  value-to-store window.
- Correction: the array is module-scope `var<private>` (literal length 10).
  Every write and read (fill after the `s_value` line, pathway dot, normaliser,
  trace update, store-pass `O_RECENT_PATHWAY_KEYS` write at `store_slot`) is in
  a `tid == 0u` block of one invocation, so private storage is semantically
  identical and adds no shared memory. The store-pass placement, same
  `store_slot` key, formulas, touches and the shared-memory constants are
  unchanged. Task 0201's `critic_death_zeroes_both_traces` line now states the
  observable reachable at the host tick boundary (zero stored pathway inputs:
  pathway trace reads zero; 128-d trace minus first term reads zero, all
  entries); the zeroing requirement is not weakened.
- Witness: the brain-lib check and the focused-proof check
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_routes`)
  pass on the preserved implementation.

## Repair: publish the kernel-routes storage-class amendment to the base (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (kernel-routes failed; route-gate-decisions blocked).

- Superseded: the published base records still carried ARCHITECTURE.md D5's
  "new workgroup array" and the State table "(workgroup, ...)" row, while the
  amendment above lived only on the plan branch (commit 1bcb430).
- Contradiction: the kernel-routes reviewer judged against the unamended base
  wording (it cites D5 line 131 and State table line 292, the base line
  numbers) and kept rejecting the `var<private>` declaration the amendment
  requires; the architect could not satisfy both, so the run stopped.
- Correction: the same amendment text (D5, State table, task 0201's
  `critic_death_zeroes_both_traces` line and the entry above) is now on the
  base, so every role reads one contract. No bound, proof, touch, dependency
  or behavior changes. Source check: every `s_critic_pathway_input` access
  (brain_passes.wgsl fill, dot/normaliser, trace, store-pass ring write) is in
  a `tid == 0u` block, and `coop_predict_and_act` and `coop_learn_and_store`
  run in the same entry point in every composed kernel (brain_tick.wgsl,
  brain_inner.wgsl, exact_tiled_brain.wgsl,
  phase_brain_tail_from_scratch.wgsl), so private storage persists across both.
- Witness: `grep -q -F 'module-scope `var<private>` array' docs/plans/0002-pathway-critic-and-forward-drive/ARCHITECTURE.md`
  exits 0; on the preserved kernel-routes branch the brain-lib check
  (`cargo test -p xagent-brain --lib`) and focused-proof
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test pathway_routes`) pass.

## Repair: name the `xagent` binary in `cargo run` commands

Cited authority: the plan owner's standing authorization for automatic repair of this run.

- Superseded: the `$integration`-scope witness action `dump-default-config`
  (`cargo run --release -p xagent-sandbox -- --dump-config`) and the task
  procedures in 0101, 0401 and 0402 that run `cargo run -p xagent-sandbox -- ...`
  without `--bin`.
- Contradiction: route-gate-tool and critic-input-decision add the
  `pathway_route_gate` and `pathway_return_fit` binaries to xagent-sandbox,
  and the package has no `default-run`. After that, cargo refuses a bare
  `cargo run -p xagent-sandbox` ("could not determine which binary to run",
  exit 101), so the host continuation proof failed.
- Correction: every such command now passes `--bin xagent`, the `[[bin]]`
  declared in crates/xagent-sandbox/Cargo.toml. It runs the same program with
  the same arguments, so no behavior, bound or proof changes.
- Witness: `cargo run --release -p xagent-sandbox --bin xagent -- --dump-config`
  exits 0 and prints the JSON config, and
  `grep -rn --exclude=AUTHORIZATION.md 'cargo run.*-p xagent-sandbox -- ' docs/plans/0002-pathway-critic-and-forward-drive`
  finds nothing.

## Repair: remove the residual workgroup wording behind the kernel-routes review (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (kernel-routes failed; route-gate-decisions blocked).

- Superseded: ARCHITECTURE.md baseline fact "so a workgroup variable written
  by the first is readable by the second", and task 0201 step 1's unqualified
  "declare ... `s_critic_pathway_input`".
- Contradiction: D5 and the State table already require module-scope
  `var<private>` (entries above), but the baseline fact still justified a
  workgroup variable and step 1 named no storage class; the kernel-routes
  reviewer read them as requiring `var<workgroup>` and rejected the
  conforming `var<private>` declaration (brain_passes.wgsl:71 on the preserved
  branch), while a workgroup array fails brain-lib (6_112 against 6_064 and
  9_248 against 9_200).
- Correction: the baseline fact now states that a thread-0 `var<private>`
  value persists from the value pass to the store pass of the same
  invocation, and step 1 names `var<private>` explicitly. No bound, proof,
  touch, dependency or behavior changes.
- Witness: `grep -q -F 'never
   `var<workgroup>`' docs/plans/0002-pathway-critic-and-forward-drive/tasks/0201-kernel-routes.md`
  and `! grep -q -F 'so a workgroup variable written' docs/plans/0002-pathway-critic-and-forward-drive/ARCHITECTURE.md`
  exit 0.
