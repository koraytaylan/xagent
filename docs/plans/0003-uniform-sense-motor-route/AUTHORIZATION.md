# Authorization record

## Corrected requirements: `generic-sense-helpers` review contradictions (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (stopped-run repair after the task spent its three amendment
escalations).

- Superseded (PLAN.md constraint `helper-callers-known`): the check grepped
  `phase_death.wgsl` and `kernel_tick.wgsl` for `settle_turn_pathways_at_death`.
- Contradiction: packet step 5 and ARCHITECTURE.md refactor step 2 require both
  death callers to call `settle_senses_at_death(brain_base)` instead, so the
  check is false on the task's own required result.
- Correction: each caller grep now accepts
  `settle_(turn_pathways|senses)_at_death(brain_base)`, still proving the death
  call site is present before and after the task; the `whiten_scent` and
  `refresh_vision_whitening` greps are unchanged.
- Superseded (packet placement constraint): "The extracted slice must not
  contain `a[` or `v[` after its `a`/`v` rewrite", read by review as forbidding
  `a[`/`v[` in the shader source.
- Contradiction: `crates/xagent-brain/src/gpu_kernel/whitening_storage_validation.rs:50-56`
  (unchanged, part of the alternate-schedules gate) requires the slice to hold
  exactly one private `var a`/`var v` declaration and `a[`/`v[` indexing, and
  only its own rewrite to `whitening_scratch` removes them; renaming the arrays
  failed that test (left 0, right 1, exit 101).
- Correction: the packet now states that the slice keeps the private `a`/`v`
  matrices and their indexing, and is free of `a[`/`v[` only after the
  validation test's rewrite. No bound, task, dependency, gate or behavior
  changed.
- Witness: the corrected `helper-callers` command exits 0 on the baseline
  (`8c44cb1`) and on the `generic-sense-helpers` branch; the task branch's
  host verify (all seven gates, including alternate-schedules) passed on the
  unrenamed Jacobi arrays.

## Corrected requirement: `headless-cli-help` action (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: the `study-route-witness` execution check action
  `headless-cli-help` ran `cargo run -p xagent-sandbox -- --help`.
- Contradiction: plan 0002 (landed in `e246a6e`) added the
  `pathway_return_fit` and `pathway_route_gate` binaries to `xagent-sandbox`,
  so `cargo run` without `--bin` exits 101 ("could not determine which binary
  to run").
- Correction: the action now runs
  `cargo run -p xagent-sandbox --bin xagent -- --help`, the same CLI the check
  always targeted (`[[bin]] name = "xagent"` in
  `crates/xagent-sandbox/Cargo.toml`). No bound, task, dependency or behavior
  changed.
- Witness: `cargo run -p xagent-sandbox --bin xagent -- --help` exits 0 and
  prints the CLI help; `validate_plan` reports ok.
