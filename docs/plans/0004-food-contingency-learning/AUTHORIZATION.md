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
