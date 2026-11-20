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
