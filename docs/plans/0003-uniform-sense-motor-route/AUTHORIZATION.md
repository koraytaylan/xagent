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

## Corrected requirement: generic-sense-helpers Jacobi placement wording, second clarification (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: packet `tasks/0205-generic-sense-helpers.md` placement sentence
  "it contains no `a[` or `v[` only after the `a`/`v` rewrite to
  `whitening_scratch` that `whitening_storage_validation.rs` performs".
- Contradiction: review again read it as forbidding `a[`/`v[` indexing in the
  shader source slice (finding on common.wgsl:1295-1357), while the unchanged
  gate test `crates/xagent-brain/src/gpu_kernel/whitening_storage_validation.rs:50-51`
  asserts the source slice holds one `PRIVATE_MATRICES` declaration and
  `a[`/`v[` indexing; renaming them failed it (left 0, right 1, exit 101).
- Correction: the sentence now states explicitly that the source slice must
  keep the declaration and all `a[`/`v[` indexing, that such indexing is not a
  defect, and that the "no `a[`/`v[`" property applies only to the test's
  in-memory rewritten copy (lines 52-56). Same requirement, unambiguous; no
  bound, task, dependency, gate, test or behavior changed.
- Witness: the `alternate-schedules` check
  (`workgroup_whitening_matches_complete_serial_state`) and host verify (all
  seven gates passed) on the task branch's unrenamed Jacobi arrays.

## Corrected requirement: generic-schedule-candidates gather wording (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: packet `tasks/0207-generic-schedule-candidates.md` sentence
  "Visual-pathway prepare runs for each Jacobi hemifield sense. Its gather is
  `vision_hemifields()`."
- Contradiction: review flagged `visual_pathway_validation.rs:77`
  (`let raw = vision_hemifields();`) as "the `sense_hemifield_input()` gather"
  and asked to change it to `vision_hemifields()`, which it already is. In the
  shipped code `sense_hemifield_input()` (`brain_passes.wgsl:911-913`) only
  returns `vision_hemifields()`, and its one call is the serial block
  (`brain_passes.wgsl:1785`), which Non-goals forbid editing. No edit could
  close the finding.
- Correction: the sentence now says the fragment and probe gather is a direct
  `let raw = vision_hemifields();` call, and that the serial block's
  `sense_hemifield_input()` wrapper call stays and is not a finding. Same
  requirement, unambiguous. No bound, task, dependency, gate, test or behavior
  changed.
- Witness: `grep -rn "sense_hemifield_input\|vision_hemifields" crates/xagent-brain/src`
  on the task branch shows the direct call at `visual_pathway_validation.rs:77`
  and `brain_cooperative_visual_pathway.wgsl:18`, and the wrapper only in
  `brain_passes.wgsl`. Host verify on that branch passed focused-proof,
  alternate-schedules, parity, clippy and guard. fmt failed only on a line
  wrap, which its declared fix `cargo fmt --all` rewrites.

## Corrected requirement: generic-schedule-candidates baseline constraint (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: PLAN.md constraint `generic-schedule-candidates /
  candidate-fragments-known / composition-tests`, which ran
  `visual_pathway_composition_preserves_other_passes` and
  `jacobi_composition_keeps_other_helpers_and_scratch` against the baseline.
- Contradiction: source proofs must hold on baseline code. On the plan
  baseline the landed `generic-sense-block-sites` task already rewrote the
  serial block, so the old visual-pathway test's `ORIGINAL_CALL` anchor matches
  0 times (`visual_pathway_validation.rs:64`, left 0 right 1). Updating that
  test to the generic block is this task's own deliverable, so the baseline
  proof could not pass before the task runs.
- Correction: the baseline constraint now checks that the visual-pathway test
  exists and runs the Jacobi composition test, which passes on baseline. The
  task's own gates still require both composition tests unchanged. No bound,
  gate, task, dependency or behavior changed.
- Witness: in the plan checkout the new command exits 0. On the task branch
  `visual_pathway_composition_preserves_other_passes` passes (1 passed).

## Corrected requirement: remove-sense-specific-wgsl touches and readers (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task 0208's Decisions claim that each renamed constant "is read
  only inside the two touched WGSL files", its step 1 confinement of readers to
  `common.wgsl` and `brain_passes.wgsl`, and its Non-goal forbidding edits to
  the validation modules and other `.wgsl` fragments, as applied to the names
  this task must remove.
- Contradiction: `kernel_tick.wgsl:786/789/798` and `phase_death.wgsl:209/212/221`
  call `vision_pathway_learning_rate` and read `SCENT_CHANNELS`, and
  `gpu_kernel/whitening_validation.rs:54` asserts
  `const VISION_WHITENING_REFRESH` in `common.wgsl`. Removing the names (required
  by the source test) breaks shader composition and brain-lib; keeping them
  fails the source test. The stop rule forbade the only fix.
- Correction: add those three files to the touches and permit exactly the
  value-preserving rewrites: `vision_pathway_learning_rate(brain_base)` becomes
  `sense_learning_rate(brain_base, 1u)` (the wrapper's own body,
  `common.wgsl:1061-1063`), `SCENT_CHANNELS` becomes the planned
  `NON_VISUAL_FEATURE_CHANNELS`, and the test anchor becomes
  `override SENSE1_REFRESH_TICKS: u32 = {REFRESH_CYCLES}u;` (existing default 20,
  `common.wgsl:407`, equal to `REFRESH_CYCLES`). `whitening_storage_validation.rs`
  and `visual_pathway_validation.rs` only assert absence and stay unedited. The
  removed list, source test, gates, dependencies and behavior are unchanged.
- Witness: the task's brain-lib, pathway-tests and parity checks, with
  focused-proof's `sense_specific_wgsl_has_no_sense_names`.

## Corrected requirement: remove-sense-specific-wgsl buffers.rs pinned lines (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: task 0208's Non-goals "No edit to `buffers.rs`" and "No change
  ... to existing test bodies", as applied to the WGSL lines pinned by
  `buffers::tests::pathway_block_and_cfg_slots_match_wgsl`.
- Contradiction: that test (`buffers.rs:2375/2381/2382`) asserts `common.wgsl`
  contains `override O_FORWARD_PATHWAY_WEIGHTS: u32 = O_TRACE_VISION +
  VISION_PATHWAY_INPUTS;`, the initialized `override O_SENSORY_MEAN: u32 =
  ...;` and `const PATHWAY_INPUT_COUNT: u32 = SCENT_CHANNELS +
  VISION_PATHWAY_INPUTS;`. The task must remove `O_TRACE_VISION`, rename
  `SCENT_CHANNELS` and declare `O_SENSORY_MEAN` uninitialized (step 2 and
  `sense_specific_*` tests), while brain-lib must pass. No implementation
  satisfies both.
- Correction: add `buffers.rs` to the touches and permit only those three
  string literals to be updated to the new `common.wgsl` lines (forward-weights
  default without any removed name and with unchanged value, e.g.
  `SENSE1_TURN_TRACE + SENSE1_CHANNELS`; `override O_SENSORY_MEAN: u32;`;
  `NON_VISUAL_FEATURE_CHANNELS + VISION_PATHWAY_INPUTS`). The removed list,
  source tests, gates, dependencies, values and behavior are unchanged; the
  test still pins every pathway-block line.
- Witness: focused-proof (`sense_specific_wgsl_has_no_sense_names`,
  `sense_specific_sensory_mean_matches_layout`), brain-lib (including
  `pathway_block_and_cfg_slots_match_wgsl`) and parity byte-identical.

## Corrected requirement: modulator-study output paths (2026-11-20)

Authority: host-authorized plan amendment for task `modulator-study`.

- Superseded: task 0402's touches, `focused-proof` check, Scope/steps/Done
  when, PLAN.md deliverables list and ARCHITECTURE.md INV-08 witness, which
  placed the task's outputs at
  `docs/plans/0003-uniform-sense-motor-route/0402-MODULATOR-DECISION.md` and
  `docs/plans/0003-uniform-sense-motor-route/0402-evo-reference.json`.
- Contradiction: those paths lie inside the protected plan record. The host
  discards any implementation edit there ("the architect edited protected plan
  record(s) ... the host discarded the edits"), so the task cannot both write
  its required records and pass; the `focused-proof` grep then fails on the
  missing file.
- Correction: the outputs move to `docs/decisions/modulator-study.md` and
  `docs/decisions/modulator-study-evo-reference.json`, the repository's
  existing location for study decision records (plan 0002's
  `docs/decisions/pathway-*-route.{md,json}`). The verdict lines, rule,
  required content, other touches, gates, dependencies and behavior are
  unchanged; only the path changes.
- Witness: the amended `focused-proof` command exits 0 against a file at
  `docs/decisions/modulator-study.md` holding the two verdict lines, and fails
  when it is absent.

## Corrected requirement: pooling-study decision record path (2026-11-20)

Authority: host-authorized plan amendment for task `pooling-study`.

- Superseded: task 0301's touches, `verdict` check, step 4 and Done when;
  task 0302's touches comment, `verdict` check and Context/step 1; PLAN.md's
  decision-record list; and ARCHITECTURE.md INV-06 witness, which placed the
  study's decision record at
  `docs/plans/0003-uniform-sense-motor-route/0301-POOLING-DECISION.md`.
- Contradiction: that path lies inside the protected plan record. The host
  discards any implementation edit there ("the architect edited protected plan
  record(s) (docs/plans/0003-uniform-sense-motor-route/0301-POOLING-DECISION.md);
  the host discarded the edits"), so the task cannot both write its required
  record and pass; the `verdict` grep then fails on the missing file, and
  0302's `verdict` check could never read it. This is the same defect corrected
  above for `modulator-study`.
- Correction: the record moves to `docs/decisions/pooling-study.md`, the
  repository's existing location for study decision records
  (`docs/decisions/modulator-study.md`, plan 0002's `pathway-*-route.md`).
  The verdict lines, gate rule, required content, other touches, gates,
  dependencies, gated flags and behavior are unchanged; only the path changes.
- Witness: the amended 0301 `verdict` command exits 0 against a file at
  `docs/decisions/pooling-study.md` ending in `Verdict: NOT PORTED`, and exits
  non-zero (2) when it is absent; the amended 0302 `verdict` command takes its NOT PORTED
  branch and exits 0 on the same file.

## Corrected requirement: pooling-study score food-side sign (2026-11-20)

Authority: host-authorized plan amendment for task `pooling-study`.

- Superseded: task 0301's fixed decision **Score**, which correlated the turn
  output with `sign(food_bearing)`.
- Contradiction: ARCHITECTURE.md (Investigations: Pooling, line 179) defines
  the score as the correlation of the policy's turn "with the food side", and
  the plan 0001 free-life harness the task reuses defines the food side as
  `food_side: -sign(bearing)` (crates/xagent-sandbox/src/free_life/metrics.rs:460).
  The task's own sanity references (B `+0.16`, A `+0.33`, from the
  learner-foraging investigation) are positive on that food-side scale. The
  literal `sign(food_bearing)` negates every score (measured A +0.2865,
  C +0.0435, D +0.0417 become -0.2865, -0.0435, -0.0417), so the fixed gate
  `max(C, D) >= A` would flip from NOT PORTED to PORTED and adopt a worse
  input, the outcome INV-06 exists to prevent, and both sanity lines would
  invert. No implementation satisfies both texts.
- Correction: the score correlates the turn output with the food side,
  `-sign(food_bearing)`, the harness's `food_side` convention. Gate rule,
  window, exclusion rule, sanity references, verdict lines, touches, checks,
  dependencies and gated flags are unchanged.
- Witness: the `focused-proof` check (`cargo test -p xagent-sandbox --lib
  pooling_study`) passes its `planted_side_cue_is_learned_by_reference_arm`
  test, a positive score for a cue on the food side under this convention; the
  `verdict` check reads `Verdict: NOT PORTED` from
  `docs/decisions/pooling-study.md`.

## Corrected requirement: pooling-study harness reuse (2026-11-20)

Authority: host-authorized plan amendment for task `pooling-study`.

- Superseded: task 0301's fixed decision **Harness reuse**, read as requiring the
  runner to drive plan 0001's own per-life run path for the life.
- Contradiction: that path is `headless::run_free_life`
  (crates/xagent-sandbox/src/headless.rs:1829). Its `FreeLifeOptions`
  (headless.rs:1480-1493) has no execution-schedule field, and the default fused
  schedule never writes the adapted visual block to scratch (the first run read
  all-zero column means); it has no per-tick hook, so the required
  `read_brain_scratch_blocking` column reads cannot happen during the life; and
  `build_sample` (headless.rs:1724) and `AgentAccumulator::step`
  (free_life/metrics.rs:439) are private. Changing any of this needs a plan 0001
  file edit, which the same decision forbids ("No plan 0001 file is edited").
  The named modules `metrics.rs` and `tick_log.rs` contain no life loop.
- Correction: the runner keeps its own life loop, builds each `FreeLifeSample` as
  `build_sample` does, feeds the harness's `AgentAccumulator` and
  `TickLogWriter`, takes the credit fields and ground truth from that sample, and
  restates food in view from `FOOD_IN_VIEW_DISTANCE` and `VIEW_HALF_ANGLE`. It may
  use a schedule that writes the adapted block to scratch, stated in the record.
  The no-edit rule for plan 0001 files, log format, score, gate, runs, touches,
  checks, dependencies and gated flags are unchanged.
- Witness: the `focused-proof` check (`cargo test -p xagent-sandbox --lib
  pooling_study`) passes, and the `verdict` check reads `Verdict: NOT PORTED`
  from `docs/decisions/pooling-study.md`, produced by the runner under this design.

## Corrected plan record: pooling-study split for the junior-task bounds (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (stopped on the junior-task bounds refusal).

- Superseded: task `pooling-study` (tasks/0301-pooling-study.md) as one packet
  owning the fit library (`crates/xagent-sandbox/src/pooling_study.rs`,
  `lib.rs`), the runner, the study run and the decision record.
- Contradiction: its body was 8222 characters, over the host's 8000-character
  bound for one task, so the run refused to resume; the task's last attempt
  also exhausted its review budget on the 1317-line library alone.
- Correction: the fit library moves, with every fixed decision it implements
  (log format, arms A-D, adaptation and whitening, credit rule, score, INV-06
  gate, sanity constants, the four named unit tests and their focused-proof),
  into the new non-gated developer task `pooling-fit`
  (tasks/0300-pooling-fit.md, depends on `modulator-study`). `pooling-study`
  now depends on `pooling-fit`, touches only the runner example and the
  decision record, and keeps every check (focused-proof, verdict, parity, fmt,
  clippy, guard), its role, the harness-reuse and run decisions, and its Done
  when. The whitening refresh constant is cited from `xagent_brain::senses`
  (where `sense-descriptor` placed `VISION_WHITENING_REFRESH`) instead of
  `buffers.rs`, which only names it in a comment. No bound, proof, gated flag
  or existing dependency changed; `learned-pooling-port` still waits for
  `pooling-study`.
- Witness: bodies are now 5318 (`pooling-fit`) and 4895 (`pooling-study`)
  characters with 2 touches each; the preflight `pf-task-pooling-fit` and the
  `fit-library-surfaces` constraint exit 0 on the baseline.
- `remove-sense-specific-wgsl` is split separately; see the next entry.

## Corrected requirement: remove-sense-specific-wgsl split into sense-reader-rewrites (2026-11-20)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run (stopped on the junior-task bounds refusal).

- Superseded: task `remove-sense-specific-wgsl` (tasks/0208) as one packet
  with 9 touches and an 8225-character body, including the out-of-file reader
  rewrites added by the "touches and readers" correction above.
- Contradiction: the host's task bounds allow at most 6 touches and 8000
  characters, and plan validation applies them to this record, so the run
  refused to resume.
- Correction: the three reader files (`kernel_tick.wgsl`, `phase_death.wgsl`,
  `gpu_kernel/whitening_validation.rs`) move into the new non-gated developer
  task `sense-reader-rewrites` (tasks/0210, depends on
  `sense-override-constants`, `generic-sense-helpers`,
  `generic-schedule-candidates`). It renames the readers, keeping their values,
  to names that exist both before and after removal: `sense_learning_rate(brain_base, 1u)`,
  `SENSE0_CHANNELS` (or the already-landed `NON_VISUAL_FEATURE_CHANNELS`,
  same value 2) and the anchor `override SENSE1_REFRESH_TICKS: u32 =
  {REFRESH_CYCLES}u;`. `remove-sense-specific-wgsl` now also depends on it and
  keeps 6 touches, every check, its removed list, source tests, buffers.rs
  literals and Done when. No bound, proof, gated flag or existing dependency
  changed. In the saved run the landed remove commit (a955d37) already contains
  these rewrites, so the new task's gates pass there with no edit needed.
- Witness: 0208 now has 6 touches and a 7466-character body; 0210 has 3
  touches; `pf-task-sense-reader-rewrites` and the `reader-anchors` constraint
  exit 0 on the baseline; validate_plan reports no issues.
