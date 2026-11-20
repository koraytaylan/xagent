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

## Corrected plan record: carry in-run amendments into the published record (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: the published `develop` record of `ARCHITECTURE.md`, `PLAN.md`
  (decision-record list), `tasks/0101-parity-golden.md`,
  `tasks/0201-sense-descriptor.md`, `tasks/0203-descriptor-driven-mutation.md`
  and `tasks/0205-generic-sense-helpers.md`, which predated the run's
  architect-authorized amendments recorded below.
- Contradiction: host baseline recovery refused because the published record
  would revert those in-run amendments (latest `10af078192`) on the plan branch.
- Correction: the amended text and its authorization entries (below) are
  carried verbatim from `plan/0003-uniform-sense-motor-route`, keeping the
  `generic-sense-helpers` review corrections above. Task `status`/`merged_as`
  stay as published; landing remains the factory's. No bound, task,
  dependency, gate or behavior changed.
- Witness: `git diff plan/0003-uniform-sense-motor-route -- ARCHITECTURE.md`
  is empty and the task packets differ from the plan branch only in
  `status`/`merged_as` and the corrections above; `validate_plan` reports ok.

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

## Corrected requirement: parity-golden host recording (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `parity-golden` step 3 required recording the golden in the
  architect's shell with `env XAGENT_RECORD_SENSE_PARITY=1 XAGENT_REQUIRE_GPU=1
  cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored
  --test-threads=1`, and no host check ran that command.
- Contradiction: the task states the GPU hashes are host-specific and its
  Done when is judged by the host's `focused-proof`, but the sandboxed shell
  has no `/dev/dri` and records on lavapipe. Host compare then fails for
  `trajectory_raw_vision`, `trajectory_cortex` and `headless_run`
  (golden bed894a9f6ad3ce7/b3c3330f0b26e0c3/840121c20e66f90c vs host
  a44c0a6f1ca9dcf5/92903d9a95ded90f/5cfba02419b6d038), so Done when was
  unreachable without loosening the compare, which the task forbids.
- Correction: added task check `record-golden` running that same record
  command on the host, ordered before `focused-proof`; step 3 now records via
  `record-golden` and proves reproducibility by requesting `focused-proof`
  twice. Each compare is a fresh process, so run-to-run exactness is still
  proven; no tolerance, scenario, fixture rule, Done when, task id,
  dependency or gated flag changed. The record check exists only in this
  task (which changes no product code), so INV-01 and the no-re-record rule
  for later tasks are unchanged.
- Witness: host checks `record-golden` then `focused-proof`
  (`env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test
  sense_block_parity -- --include-ignored`) pass; `validate_plan` reports ok.

## Corrected requirement: parity-golden touches include `.gitignore` (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `parity-golden` touches listed only the parity test, the
  golden and `crates/xagent-sandbox/tests/fixtures/sense_block_champion.db`.
- Contradiction: the task requires committing the champion `.db` fixture, but
  the root `.gitignore` rule `*.db` ignores it, so the first implementer added
  `!/crates/xagent-sandbox/tests/fixtures/sense_block_champion.db`; the host's
  preservation commit `9dada31` carried that line onto the task branch. With
  `.gitignore` outside touches, keeping the line fails the branch-diff touches
  check and reverting it fails the working-tree touches check, so no
  implementation can pass.
- Correction: added `.gitignore` to the task's touches and stated in Scope
  that it may change only to un-ignore the champion fixture (or be restored to
  the `develop` content now that the fixture is tracked). No check, Done when,
  task id, dependency, gated flag or product-code rule changed.
- Witness: the host touches check accepts the branch with or without the
  exception line; checks `record-golden`, `focused-proof`, `fmt`, `clippy`
  and `guard` pass; `validate_plan` reports ok.

## Corrected requirement: parity-golden champion fixture recipe (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `parity-golden` Test design required the champion fixture
  to be the headless run's database "after deleting all
  `generation_recording` rows and running `VACUUM`", with no other change.
- Contradiction: that database stores wall-clock values (`run.started_at`,
  `run.wall_time_secs`, `node.created_at`), so every recording writes new
  fixture bytes. The task check `record-golden` runs in every host sweep, and
  the host refuses any gate that leaves a tracked file changed ("gates may
  write only ignored output"). Following the literal recipe fails
  `record-golden`; clearing the columns was rejected in review as off-recipe.
  No implementation could satisfy both.
- Correction: the recipe now also sets `run.started_at` and `node.created_at`
  to NULL and `run.wall_time_secs` to 0 before `VACUUM`, and forbids any other
  row or column change. These columns are run metadata outside every parity
  hash and are not read by `Governor::resume`/`champion_brain`, so the
  baseline, scenarios, compare exactness, Done when, task id, dependency,
  gated flag and checks are unchanged.
- Witness: host check `record-golden` (`env XAGENT_RECORD_SENSE_PARITY=1
  XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity
  -- --include-ignored --test-threads=1`) exits 0 leaving the committed
  fixture byte-identical (verified in an earlier round with this design),
  then `focused-proof` passes; `validate_plan` reports ok.
- Length: the task body already exceeded the 8000-character validator limit,
  so to stay under it the same edit shortened the wording of the `.gitignore`
  scope sentence, the probe path, the dependency note and steps 3-4. Their
  meaning is unchanged.

## Corrected requirement: sense-descriptor route record location (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `sense-descriptor` (touches, Outcome, step 1, Done when),
  PLAN.md "Decision records" and ARCHITECTURE.md (Routes paragraph, INV-04)
  required the developer to author
  `docs/plans/0003-uniform-sense-motor-route/0201-ROUTE-WIRING.md`.
- Contradiction: that path is inside the protected plan record. The host
  refuses developer writes to it ("not allowed for this role") and discards
  any non-amendment edit to it by other roles, so no implementation could
  produce the record that step 1, the Done when and INV-04 demand.
- Correction: the task-authored route record moves to
  `docs/decisions/sense-route-wiring.md`, beside the plan 0002 route records
  it summarizes (`docs/decisions/pathway-*.{md,json}`), which are task-authored
  outputs outside the plan record. Its required content, the admission rules,
  the BLOCKED conditions, the tests, checks, task id, dependency and gated flag
  are unchanged; only the path changes in the four places it is named.
- Witness: the developer can write `docs/decisions/sense-route-wiring.md`
  within touches; the `focused-proof` check (`cargo test -p xagent-brain --lib
  -- senses::`) together with `fmt`, `clippy` and `guard` passed on the
  preserved implementation in failure 2; `validate_plan` reports ok.

## Corrected requirement: descriptor-driven-mutation helper sense-table lifetime (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `descriptor-driven-mutation` Decisions, Helper signature
  `senses: &[SenseDescriptor]`.
- Contradiction: the same Decisions require per-sense turn offsets from
  `BrainLayout::with_senses(&BrainConfig::default(), senses).sense_blocks()`,
  and `with_senses` takes `senses: &'static [SenseDescriptor]`
  (crates/xagent-brain/src/buffers.rs:955, stored in a `'static` field at
  buffers.rs:771). A non-`'static` slice cannot be passed to it, and
  buffers.rs is outside this task's touches.
- Correction: the helper takes `senses: &'static [SenseDescriptor]`. Every
  caller the task names (`DECLARED_SENSES` and the test-only const tables) is
  already `'static`; the offset rule, tests, checks, task id, dependency and
  gated flag are unchanged.
- Witness: the preserved implementation with this signature passed
  `focused-proof`, `parity` (golden unchanged), `clippy` and `guard`, and
  `cargo fmt --all -- --check` exited 0 in failure 2.

## Corrected requirement: generic-sense-helpers WGSL channel-cap constant name (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `generic-sense-helpers` Decisions and Step 1, "Add
  `const SENSE_MAX_CHANNELS: u32 = 8u;`" in `common.wgsl`.
- Contradiction: the existing test
  `gpu_kernel::tests::sense_override_wgsl_mirrors_rust_codes`
  (crates/xagent-brain/src/gpu_kernel.rs, about lines 4738-4769) requires
  every top-level `common.wgsl` line starting with `const SENSE_` to be one of
  its 7 listed route/kind codes and asserts exactly 7 matches. The same packet
  forbids Rust and test-body edits and requires brain-lib to pass unedited, so
  the required name always fails brain-lib (failures 1 and 2: "SENSE_MAX_CHANNELS
  has no Rust constant").
- Correction: the WGSL constant is named `MAX_SENSE_CHANNELS`, the same name
  as its Rust source `pub const MAX_SENSE_CHANNELS: usize = 8;`
  (crates/xagent-brain/src/senses.rs, ARCHITECTURE.md:64). Value, type,
  placement, helpers, tests, checks, task id, dependency and gated flag are
  unchanged; the mirror test is untouched and not evaded.
- Witness: the `brain-lib` check, whose test
  `sense_override_wgsl_mirrors_rust_codes` ignores lines not starting with
  `const SENSE_`, plus `focused-proof` (which passed in failure 1);
  `validate_plan` reports ok.

## Corrected requirement: generic-sense-helpers layout-relative learning-rate gene (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `generic-sense-helpers` Non-goal "No edit to ... any Rust
  file" (and its touches limited to three WGSL files), as applied to the
  `SENSE{k}_LEARNING_RATE_GENE` value its Learning-rate decision reads.
- Contradiction: the Learning-rate and Route-settle decisions require
  `sense_learning_rate` to read `brain_state[brain_base + SENSE{k}_LEARNING_RATE_GENE]`
  and the death callers to use `settle_senses_at_death`, and Done when requires
  the parity golden byte-identical (INV-01). But `insert_sense_overrides`
  (crates/xagent-brain/src/gpu_kernel.rs:339-374) emits
  `gene_offset_or_zero()`, i.e. the static `O_VISION_PLASTICITY`
  (senses.rs:238, buffers.rs:137), while the WGSL `O_VISION_PLASTICITY`
  override (common.wgsl:273) shifts with `FEATURE_COUNT` (common.wgsl:213) and
  buffers.rs:1460 places the gene at a fixed delta from the layout's
  predictor-context weight. Under the visual cortex the override names the
  wrong slot, so `trajectory_cortex` diverges (failure 4: host bisect isolated
  the death-time settle; cortex hash 92903d9a95ded90f vs golden
  8386777063c48c62; pre-task shaders pass 4/4). No WGSL-only change can fix it
  without vision identifiers in generic helpers, which the packet forbids.
- Correction: add `crates/xagent-brain/src/gpu_kernel.rs` to the task's
  touches; `insert_sense_overrides` emits a nonzero gene as
  `gene - O_PREDICTOR_CONTEXT_WEIGHT` plus the layout's predictor-context-weight
  offset (buffers.rs:1434 formula), zero unchanged, plus one new unit test
  checking the cortex layout. The WGSL contract, objective, golden, existing
  test bodies, checks, task id, dependency and gated flag are unchanged.
- Witness: the `brain-lib` check (runs the new gene-offset test and the
  unchanged `sense_override_*` tests) and the `parity` check
  (`cortex_trajectory_matches_golden` against the unchanged golden).

## Corrected requirement: generic-sense-helpers layout-relative block slot offsets (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task `generic-sense-helpers` Decision "Layout-relative gene
  offset (Rust)" sentence "No other Rust change", and the Non-goal limiting
  Rust edits to "the `gpu_kernel.rs` gene-offset fix", as applied to the other
  `SENSE{k}_*` slot offsets.
- Contradiction: the helpers must read only the `SENSE{k}_*` accessors and
  parity must match the golden byte-identically (INV-01), but
  `derive_sense_layout` places every block from the static
  `SENSE_BASE = O_RECENT_NORMS + RECENT_CAP` (crates/xagent-brain/src/buffers.rs:270,395),
  while the WGSL `O_SCENT_*`/`O_VISION_*` overrides they replace
  (common.wgsl:334-351) derive from `O_RECENT_NORMS`, which moves with
  `FEATURE_COUNT`. With only the gene fixed, the cortex settle reads the wrong
  turn-weight and trace slots (failure 5: golden 92903d9a95ded90f, computed
  f499a9322a2fbeca, exit 101).
- Correction: `insert_sense_overrides` applies the same layout-relative shift
  to every nonzero `SENSE{k}_*` slot offset; zero unchanged; the new test also
  checks `SENSE1_TURN_WEIGHTS` under the cortex layout. WGSL contract, golden,
  existing tests, checks, task id, dependency and gated flag are unchanged.
- Witness: the `brain-lib` check (new test) and the `parity` check
  (`cortex_trajectory_matches_golden`, which passed 4/4 with this change in
  failure 5).
