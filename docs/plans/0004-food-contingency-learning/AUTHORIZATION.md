# Authorization record: food contingency learning

## Repair 2026-11-20: preflight sources moved by the 0003 landing

Authority: the plan owner's standing authorization for automatic repair of this run
(correct plan records contradicted by the shipped code, preserving the objective and
safety constraints). No requirement, bound, task, dependency or gated flag changed;
only baseline-proof source anchors were corrected to the code 0002/0003 actually landed.

1. **`pf-sense-interface` (and `pf-task-extra-sensory-layout`).**
   Superseded: `grep -qF "const SCENT_FEATURE_OFFSET: u32 = 25u;" crates/xagent-brain/src/shaders/kernel/common.wgsl`.
   Contradiction: 0003 task 0208 removed sense-specific WGSL names (`gpu_kernel.rs`
   `SENSE_SPECIFIC_NAMES` lists `SCENT_FEATURE_OFFSET`); the offset now lives in Rust.
   Correction: `grep -qF "pub const SCENT_FEATURE_OFFSET: usize = 25;" crates/xagent-brain/src/senses.rs`
   (same fact: scent offset 25, so 27 is the first free slot).
   Witness: the corrected `pf-sense-interface` command exits 0 at the baseline.
2. **`pf-extra-sensory-layout` (and `pf-task-extra-sensory-layout`).**
   Superseded: the override lines `... + 27u + SCENT_CHANNELS;` and
   `25u + SCENT_CHANNELS + 2u * DANGER_PERCEPT_FEATURES_ACTIVE;` in `common.wgsl`.
   Contradiction: 0003 renamed the WGSL constant to `NON_VISUAL_FEATURE_CHANNELS`
   (`common.wgsl` lines 168-205); the override expressions are otherwise identical.
   Correction: the same greps with `NON_VISUAL_FEATURE_CHANNELS`.
   Witness: the corrected `pf-extra-sensory-layout` command exits 0.
3. **`pf-verdict-binary-slot` (and the food-kind-verdict constraint).**
   Superseded: `test ! -e crates/xagent-sandbox/src/bin`.
   Contradiction: plan 0002 shipped `src/bin/pathway_return_fit.rs` and
   `src/bin/pathway_route_gate.rs`. The requirement is that the verdict binary's slot is free.
   Correction: `test ! -e crates/xagent-sandbox/src/bin/food_kind_verdict.rs`.
   Witness: the corrected `pf-verdict-binary-slot` command exits 0.
4. **`decision-route` execution check.**
   Superseded: `cargo run --release -p xagent-sandbox -- --bench ...`.
   Contradiction: with 0002's extra bins and no `default-run` (added later by
   food-kind-verdict), the baseline `cargo run` cannot pick a binary (exit 101).
   Correction: `cargo run --release -p xagent-sandbox --bin xagent -- --bench --bench-ticks 2000 --bench-agents 2`,
   the same binary and route that `default-run = "xagent"` selects.
   Witness: the corrected command completed 2000 ticks and exited 0 at the baseline.

## Amendment 2026-11-20: second-odour-sense off-case test compares by value

- Superseded: task 0201-second-odour-sense step 5, `senses_for_config_off_is_declared_table`
  asserts "`std::ptr::eq` with `DECLARED_SENSES`".
- Contradiction: `DECLARED_SENSES` is a landed `pub const` slice and the Non-goals say
  "No change to `DECLARED_SENSES`". Rust gives a `const` no single address (each use may
  get its own promoted allocation), so pointer identity is not guaranteed; the host run
  panicked on that assertion at `senses.rs:508`.
- Correction: the test asserts the off-case table equals `DECLARED_SENSES` by value
  (`assert_eq!` on the slices), which is what PLAN.md (line 471: the function "returns
  `DECLARED_SENSES`") requires observably. No other requirement, non-goal, touch, check,
  dependency or gated flag changed.
- Witness: the task's focused-proof `cargo test -p xagent-brain --lib -- senses::tests::`
  passed on the host with the value-equality test (8 passed).

## Repair 2026-11-20: preflight anchors accept the state their own task produces

Authority: the plan owner's standing authorization for automatic repair of this run
(correct a plan record that is contradictory, preserving the objective and safety
constraints). No requirement, bound, task, dependency, touch or gated flag changed;
each baseline proof still requires the exact baseline anchor at the baseline.

- Superseded: preflight/constraint greps that only the pre-task text satisfies:
  `pub const PHYS_STRIDE: usize = 53;` and `const PHYS_STRIDE: u32 = 53u;` and zero
  `WC_[A-Z_]*: usize = 27;` lines (`pf-food-kind-layout`, `pf-parity-physics-hash`,
  `pf-task-food-kind-layout`); a count of 1 for `hit_color = vec4<f32>(0.7, 0.95, 0.2, 1.0);`
  (`pf-vision-food-colour`, `pf-task-food-kind-colour`);
  `if alive && food_flags[food] == 0u {` (`pf-vision-scent-sites`, `pf-task-food-kind-senses`);
  `resolve_food_claim(agent_id);` (`pf-food-claim-site`, `pf-task-kernel-food-kind-meals`);
  `test ! -e` of `src/bin/food_kind_verdict.rs` / `src/kind_readout.rs` and zero `default-run`
  (`pf-verdict-binary-slot`, `pf-readout-crate`), plus their copies under `constraints`.
- Contradiction: the task packets require exactly these lines to change (0104 steps 1-2 and
  PLAN.md Owner decision item 4: `PHYS_STRIDE` 54, `WC_FOOD_KINDS` = 27; 0106 replaces the
  colour literal with `food_item_color(fidx)`; 0103 extends the `vision_prepare_scent`
  guard; 0102 passes the tick to `resolve_food_claim`; 0302/0303 create the readout module,
  verdict binary and `default-run`). Review judged food-kind-layout's correct result against
  the pre-task text and rejected it; the same would recur for every task listed.
- Correction: each anchor accepts the baseline text OR the packet's exact post-task text
  (54 together with `P_HARMFUL_MEAL_COUNT: usize = 53;`; exactly one slot-27 line, which is
  `pub const WC_FOOD_KINDS: usize = 27;`; one `hit_color = food_item_color(` per shader;
  the guard prefix `if alive && food_flags[food] == 0u`; the call prefix
  `resolve_food_claim(agent_id`; the module/binary present with `pub mod kind_readout;` /
  `fn main`, `default-run = "xagent"`).
- Witness: all 46 grep preflight/constraint commands of PLAN.md exit 0 at the baseline
  (f7b8d53), on the plan integration branch (1d450b3) and on the food-kind-layout task branch
  (c58fc83), where the superseded greps failed.

## Amendment 2026-11-20: extra-sensory-layout flag-on brain_stride growth

Authority: host-authorized plan amendment for task extra-sensory-layout. No task, dependency,
touch, check, gated flag or other requirement changed; the brain_stride formula stays frozen.

- Superseded: task 0202 test `second_odour_on_grows_sensory_and_feature_width_by_two`
  requiring `brain_stride` to grow by exactly `2 * (ENCODED_DIMENSION + 1)`.
- Contradiction: the same packet forbids changing the `brain_stride` formula and requires an
  11-slot third block (mean 2 + covariance 3 + whitened 2 + turn 2 + trace 2) inside
  `[end of the 0003 route block, sensory_mean_base)`. The landed formula in
  `crates/xagent-brain/src/buffers.rs` (`build`) adds `extra_sense_slots =
  sensory_mean_base - (route_base + PATHWAY_ROUTE_SLOT_COUNT)` to `brain_stride`, so that
  block necessarily adds 11 more slots. The host measured 71348 against the stated 71337.
- Correction: the required growth is `2 * (ENCODED_DIMENSION + 1) + 11`, the value the
  unchanged source formula yields; the 11 may be expressed as
  `on.sensory_mean_base() - off.sensory_mean_base()`.
- Witness: the task's focused-proof (`buffers::tests`) passed on the host with the assertion
  `off.brain_stride + 2 * (ENCODED_DIMENSION + 1) + (on.sensory_mean_base() -
  off.sensory_mean_base())` (measured 71348); gradient-audit and parity also passed.

## Amendment 2026-11-20: second-odour-inheritance steering feature-count inference

Authority: host-authorized plan amendment for task second-odour-inheritance. Task id,
dependency, gated flag, checks, Done when and frozen contracts (pf-inheritance-apis) are
unchanged; the touches gain two product files.

- Superseded: task 0205 "Why no product edit is expected", its Non-goal "No edit to
  `agent/mod.rs`, the governor or any other product file", and touches limited to the new
  test file.
- Contradiction: the Done when requires `mutate_steering_weights` and `steering_population`
  to run on a `second_odour_sense_enabled` brain. `mutate_steering_weights`
  (`crates/xagent-sandbox/src/agent/mod.rs`) derives the feature count with
  `feature_count_for_brain_stride(len).expect(..)` (`crates/xagent-brain/src/buffers.rs`),
  which assumes the default `FIXED_TAIL_SIZE`; the flag-on stride is larger by the
  11 extra-sense slots (previous amendment), so it panics ("brain_state length matches no
  BrainLayout"). `steering_population` has no other path and `mutate_steering_weights_for`
  builds offsets from `BrainConfig::default()`, so no test-only change can pass.
- Correction: touches add `crates/xagent-sandbox/src/agent/mod.rs` and
  `crates/xagent-brain/src/buffers.rs`; a new step 5 lets the task make the feature-count
  inference accept the flag-on length (the source offsets of the action turn weights and the
  0003 sense blocks relative to the predictor context are unchanged by the block, which lies
  after the route block), keeping signatures frozen, default behaviour bit-identical, no
  path writing the block, and every test assertion unweakened.
- Witness: the task's focused-proof (`cargo test -p xagent-sandbox --test
  second_odour_inheritance`) with all three tests ok, and agent-unit
  (`cargo test -p xagent-sandbox --lib agent::`) green; the failing witness is the recorded
  panic at agent/mod.rs:593 in both mutation tests.

## Amendment 2026-11-20: food-kind-colour hit-index recovery at two colour sites

Authority: host-authorized plan amendment for task food-kind-colour. Task id, dependencies,
touches, checks, Done when, gated flag and frozen contracts are unchanged.

- Superseded: task 0106 Scope "`phase_vision_parallel.wgsl` (line 191): ... using `fidx`
  (in scope from line 73)" and "`phase_vision_object_queries.wgsl` (line 196): ... use the
  food-index variable in scope there; invent no index", and Decision "Only the three literals
  change"; and the review reading (round 3) that recovering the hit index by re-running the
  existing food search is an unauthorized restructure.
- Contradiction: in the baseline `fidx` is local to `vision_parallel_sample` (line 73),
  which returns only `VISION_EVENT_FOOD`, while the colour site is in `vision_parallel_rays`;
  in object-queries `vision_object_write_ray(agent_id, ray_idx, first_event)` sees only the
  event word, the caller's `food_id` loop (line 268) folding into an `atomicMin`. So a
  literal-only change leaves Second items in the base colour in two paths, failing the Done
  when's INV-04 "in every vision path".
- Correction: Scope now states the index is not in scope at those sites and requires it be
  recovered at the colour site by re-running the existing food test (parallel: at the ray's
  first hit step, optionally via a helper shared with the sample function; object-queries:
  lowest `food_id` in `vision_object_cache` whose `vision_object_first_hit` step equals the
  first step). The Decision keeps no new storage buffer, binding or workgroup variable and an
  unchanged event word/encoding/`atomicMin` (so no food-index cap is introduced), and states
  that byte-identical surrounding shader code is not required.
- Witness: the task's focused-proof, vision-regression and parity checks, which passed with
  all=true on the round-3 implementation of exactly this design.

## Amendment 2026-11-20: food-kind-senses names the WGSL scent count by its 0003 name

Authority: host-authorized plan amendment for task food-kind-senses. Task id, dependencies,
touches, checks, Done when, gated flag and frozen contracts are unchanged; the guard and
offsets keep the same value (2u) and meaning.

- Superseded: task 0103 Outcome/Scope/WGSL-names/Non-goals text requiring the WGSL guard
  `EXTRA_SENSORY_CHANNELS >= SCENT_CHANNELS` and writes at `scent_base + SCENT_CHANNELS`
  and `scent_base + SCENT_CHANNELS + 1u`, the ARCHITECTURE.md food-kind-senses row's
  `scent_base + SCENT_CHANNELS`, and the review reading (rounds 2 and 4) that the WGSL must
  spell `SCENT_CHANNELS`.
- Contradiction: no WGSL file defines `SCENT_CHANNELS`; 0003 renamed it to
  `NON_VISUAL_FEATURE_CHANNELS` (`common.wgsl:168`, `const NON_VISUAL_FEATURE_CHANNELS: u32 = 2u;`;
  this record's Repair item 2), and `SENSE_SPECIFIC_NAMES` in `gpu_kernel.rs` bans
  `SCENT_CHANNELS` in WGSL. The packet forbids editing `common.wgsl`/`gpu_kernel.rs` and a
  literal offset, so the literal text is unsatisfiable within the touches.
- Correction: the same rename as Repair item 2: in WGSL the guard is
  `EXTRA_SENSORY_CHANNELS >= NON_VISUAL_FEATURE_CHANNELS` and the writes are
  `sensory_buffer[scent_base + NON_VISUAL_FEATURE_CHANNELS]` and
  `sensory_buffer[scent_base + NON_VISUAL_FEATURE_CHANNELS + 1u]`; WGSL must not write or
  define `SCENT_CHANNELS`; the Rust `buffers::SCENT_CHANNELS` stays usable in Rust tests.
- Witness: the task's focused-proof, scent-regression, sandbox-sensory-regression and parity
  checks, which passed with all=true on the round-4 implementation
  (`phase_vision.wgsl:414-417`, which already uses `NON_VISUAL_FEATURE_CHANNELS`).

## Amendment 2026-11-20: second-odour-features names the WGSL scent count by its 0003 name

Authority: host-authorized plan amendment for task second-odour-features. Task id,
dependencies, touches, checks, Done when, gated flag and frozen contracts are unchanged; the
offset keeps the same value (2u) and meaning, and no literal scent width is permitted.

- Superseded: task 0204 Scope loop `sensory_buffer[s_base + scent_offset + SCENT_CHANNELS + k]`
  (with "use `u32(SCENT_CHANNELS)`") and Decision "The loop names `EXTRA_SENSORY_CHANNELS`
  and `SCENT_CHANNELS`", and the review reading (rounds 1-3) that the WGSL must spell
  `SCENT_CHANNELS`.
- Contradiction: no WGSL file defines `SCENT_CHANNELS`; 0003 renamed it to
  `NON_VISUAL_FEATURE_CHANNELS` (`common.wgsl:168`, `const NON_VISUAL_FEATURE_CHANNELS: u32 = 2u;`;
  Repair item 2 and the food-kind-senses amendment above). The guard test
  `sense_specific_wgsl_has_no_sense_names` (`gpu_kernel.rs:4988-5015`, list
  `SENSE_SPECIFIC_NAMES`) fails if `SCENT_CHANNELS` appears in `brain_passes.wgsl` or
  `common.wgsl`, and the packet's Non-goals forbid editing `common.wgsl`/`gpu_kernel.rs`.
- Correction: both extractor loops read
  `sensory_buffer[s_base + scent_offset + NON_VISUAL_FEATURE_CHANNELS + k]`; WGSL must not
  write or define `SCENT_CHANNELS`; Rust tests may keep using `buffers::SCENT_CHANNELS`.
- Witness: the task's focused-proof, scent-regression, gradient-audit and parity checks, which
  passed with all=true on the round-3 implementation (`brain_passes.wgsl:248`,
  `phase_brain_features.wgsl:82`).

## Amendment 2026-11-20: food-kind-free-life depends on the flag-on stride inference

Authority: host-authorized plan amendment for task food-kind-free-life. Task id, touches,
gated flag, checks, Done when, Decisions, Non-goals and frozen contracts are unchanged;
depends_on gains one existing task.

- Superseded: task 0401 depends_on without second-odour-inheritance, and its Prerequisites
  treating the flag-on brain stride as usable by the landed seeding.
- Contradiction: the Decisions require `--food-kinds` to set
  `brain.second_odour_sense_enabled = true` before kernel construction and the Done when
  requires both GPU tests to pass under it, but the landed seeding calls
  `GpuKernel::write_agent_heritable_config` (`gpu_kernel.rs:3896`), which calls
  `buffers::fixed_tail_base`; `feature_count_for_brain_stride` (`buffers.rs:475`) assumes
  the default `FIXED_TAIL_SIZE` and returns `None` for the flag-on stride, so
  `fixed_tail_base` panics (`buffers.rs:520`). The Non-goals forbid this task editing
  buffers.rs, and skipping the call would restructure the run loop (INV-07). The fix is
  already authorized in task second-odour-inheritance (amendment "second-odour-inheritance
  steering feature-count inference" above, which owns buffers.rs); adding buffers.rs here
  instead would make two tasks claim the same path (validate_plan refuses it).
- Correction: food-kind-free-life depends_on second-odour-inheritance, whose landed
  flag-on-aware `feature_count_for_brain_stride` makes `fixed_tail_base` accept the
  second-odour stride; the Prerequisites name it. To stay within the 8000-character body limit, the
  Prerequisites phrase "(directly or transitively)" becomes "(or transitively)", which means the same. No cycle: second-odour-inheritance depends
  only on extra-sensory-layout.
- Witness: the task's focused-proof (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox
  --test food_kind_free_life`) with all three tests ok once second-odour-inheritance lands;
  the failing witness is the recorded panic at buffers.rs:520 in
  food_kind_free_life_frozen_arm_keeps_turn_weights and food_kind_free_life_summary_is_complete.

## Repair 2026-11-20: flag-on fixed-tail base (new task flag-on-tail-base)

Authority: the plan owner's standing authorization for automatic repair of this run (add a
non-gated implementation task for work an approved bound needs that no pending task delivers).

- Superseded: the food-kind-free-life amendment's claim that second-odour-inheritance's
  landed work makes `fixed_tail_base` accept the second-odour stride.
- Contradiction: task 0205's packet requires only the steering feature-count inference;
  its accepted-by-gates implementation is a private `steering_feature_count` in
  `crates/xagent-sandbox/src/agent/mod.rs` and leaves `buffers.rs` unchanged, so
  `buffers::fixed_tail_base` (`buffers.rs:518-520`) still panics for the flag-on stride
  inside `GpuKernel::write_agent_heritable_config` (`gpu_kernel.rs:3896`). The
  food-kind-free-life focused-proof cannot pass even after 0205 lands, and no pending task
  delivers the fix.
- Correction: new task `flag-on-tail-base` (tasks/0206, non-gated, touches only
  `buffers.rs`, depends_on second-odour-inheritance so the two `buffers.rs` owners are
  serialized) makes `feature_count_for_brain_stride` and `fixed_tail_base` accept the
  flag-on stride with default results bit-identical; food-kind-free-life gains depends_on
  flag-on-tail-base. Its Feasibility row, execution_checks entry (developer-route) and
  constraint `pf-flag-on-tail-base` are added. Every existing bound, proof and dependency
  stays.
- Witness: tasks/0206 focused-proof (`cargo test -p xagent-brain --lib buffers::`) with
  its three named tests ok; then food-kind-free-life's focused-proof with all three tests
  ok. Failing witness: the recorded panic at `buffers.rs:520`.

## Repair 2026-11-20: second-odour-inheritance review ledger record

Authority: the same standing authorization (correct a plan record the review could not judge).

- Superseded: nothing in the contract; the 0205 packet lacked a record of review ledger L2.
- Contradiction: the last review's only open finding (minor) asked for L2's full text in the
  0205 packet, which no approved check proves; the work itself (gates focused-proof and
  agent-unit green, diff from `bcd26ac^` limited to `agent/mod.rs` and the new test) was
  not found defective.
- Correction: 0205 gains a "Review record" section restating L1/L2 and their resolution
  (commit a9e1963). No requirement changes.
- Witness: `git diff bcd26ac^ a9e1963 -- crates/xagent-brain crates/xagent-sandbox/src/bin crates/xagent-sandbox/src/food_kind_metrics.rs` is empty; task gates focused-proof and agent-unit.

## Repair 2026-11-20: food-kind-senses packet body within the 8000-character limit

- Authority: the plan owner's standing authorization for automatic repair of this run.
- Superseded record: the `tasks/0103-food-kind-senses.md` body (8356 characters after
  the in-run amendment carried in 9edfd57), which validate_plan refuses (limit 8000).
- Contradiction: the amended packet text exceeded the packet size limit, so the published
  record failed validation.
- Correction: wording only. A duplicated sentence in Steps 1-2 that repeated Scope was
  removed and several sentences were made shorter. Frontmatter, checks, touches,
  dependencies, every requirement, name, bound and the amendment content are unchanged.
- Witness: `validate_plan docs/plans/0004-food-contingency-learning` returns ok, 17 tasks.

## Repair 2026-11-20: flag-on-tail-base preflight check declared

- Authority: the plan owner's standing authorization for automatic repair of this run.
- Superseded record: PLAN.md `preflight_checks`, which lacked `pf-flag-on-tail-base`
  although the ARCHITECTURE.md Feasibility row for flag-on-tail-base names it.
- Contradiction: the host continuation proof refused the plan because the feasibility
  witness names an unavailable preflight check.
- Correction: `pf-flag-on-tail-base` is added to `preflight_checks` with the same command
  as the existing named constraint check. No requirement, bound or dependency changes.
- Witness: the command exits 0 on the baseline (`cargo test -p xagent-brain --lib buffers:: --no-run`
  builds); validate_plan returns ok.
