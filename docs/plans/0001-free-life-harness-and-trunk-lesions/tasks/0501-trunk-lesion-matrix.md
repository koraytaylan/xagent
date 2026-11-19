---
id: trunk-lesion-matrix
title: Run the trunk lesion matrix and record which trunk parts behaviour and within-life learning need
workstream: "0500"
kind: task
role: architect
depends_on: [free-life-cli]
gated: false
touches:
  - docs/measurements/trunk-lesions/lesion-matrix.json
  - docs/measurements/trunk-lesions/DECISION.md
  - crates/xagent-sandbox/tests/lesion_matrix_record.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "cargo test -p xagent-sandbox --test lesion_matrix_record"
    timeout_ms: 3600000
  - name: sandbox-guard
    command: "cargo test -p xagent-sandbox --test contributing_guard"
  - name: sandbox-clippy
    command: "cargo clippy -p xagent-sandbox --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: sandbox-fmt
    command: "cargo fmt -p xagent-sandbox -- --check"
    fix: "cargo fmt -p xagent-sandbox"
---

# Run the trunk lesion matrix and record which trunk parts behaviour and within-life learning need

## Outcome

- The reference lineages and 48 free lives run per ARCHITECTURE.md "Free-life
  run" and "Matrix and decision rule"; their reports (ignored
  `target/trunk-lesions/runs/`) are joined into the committed array
  `docs/measurements/trunk-lesions/lesion-matrix.json` of `FreeLifeReport`.
- `crates/xagent-sandbox/tests/lesion_matrix_record.rs` checks the matrix
  (INV-10, "Untrusted inputs"), recomputes the verdicts and compares them with
  `DECISION.md`, which records verdicts and numbers, null or negative included.

## Decisions

1. **Build and base config.** Run `cargo build --release -p xagent-sandbox`. Then run
   `mkdir -p target/trunk-lesions/runs`, because the in-checkout `target/` directory may
   not exist when the host redirects Cargo's target directory. Save
   `cargo run --release -p xagent-sandbox -- --dump-config` as `target/trunk-lesions/base.json`,
   set `governor.tick_budget` to `40000`, and confirm `vision_stride` is `1`.
   Every xagent invocation in Decisions 2-3 runs as `cargo run --release -p
   xagent-sandbox -- <flags>`, never via `target/release`; data paths under
   `target/trunk-lesions/` are plain checkout-relative paths.
2. **Reference lineages.** Seeds 5, 6, 7, 8, each as ARCHITECTURE.md "Matrix and
   decision rule" specifies: `cargo run --release -p xagent-sandbox -- --config target/trunk-lesions/base.json --seed S --no-render --db target/trunk-lesions/reference-sS.db --generations 80`.
   `--config target/trunk-lesions/base.json` is required on every lineage run, or
   the lineage uses the default tick budget instead of 40000.
3. **Free lives.** 48 runs (sources `fresh` and `champion`, arms `control`,
   `turn-weights`, `forward-weights`, `memory-blend`, `critic`, `all`, seeds 5-8)
   as ARCHITECTURE.md "Free-life run" and "CLI" specify: `--free-life
   --free-life-agents 10 --free-life-ticks 300000 --seed S`, the champion source
   adds `--free-life-champion-db target/trunk-lesions/reference-sS.db` and no node
   flag, `control` passes `--free-life-lesion none`, the other arms pass their own
   name, and each writes `--free-life-out target/trunk-lesions/runs/<source>-<arm>-seedS.json`.
4. **Jobs and resources.** Check `free -h` before the first batch and before each
   batch. Start each job in the background with its output in `$MAKINA_SHELL_JOB/output`.
   At most four GPU jobs run at once. Never start a job that is still listed as running.
   A champion run starts only after its seed's database exists. Re-run a non-zero
   exit only after its cause is recorded.
5. **Join.** A POSIX `sh` loop writes the 48 reports into
   `docs/measurements/trunk-lesions/lesion-matrix.json` as one JSON array: `[`,
   the objects separated by commas with no trailing comma, `]`. The result must
   parse as a JSON array of 48 objects.
6. **Record test.** Follow ARCHITECTURE.md "Matrix and decision rule" and "Untrusted
   inputs". Resolve the workspace root from `CARGO_MANIFEST_DIR` (`ancestors().nth(2)`),
   as `contributing_guard.rs` does. Deserialize `Vec<FreeLifeReport>`. Identify each
   report's source from `brain_source`, its arm from `lesions`, and its seed from
   `seed`; require exactly one report for each of the 48 combinations. Assert
   `agents.len() == 10`, `ticks == 300000`, `brain_ticks * kernel_batch_size == ticks`
   (checked multiplication), champion `champion_node_id` and `champion_generation`
   set and fresh ones unset, and every `f64` metric finite (`whole` and `late`, every
   agent, every `turn_credit` and `forward_credit` element). Use named constants with
   doc comments for the seeds, agent count, tick count and the `2.0` significance and
   `2.0` standard-error divisor. Recompute `d`, `se` and each verdict by the binding
   rule, print one line per part and source for each metric used, then compare with
   the `DECISION.md` table. Also assert `DECISION.md` has a line containing
   `--generations 80` and every such line contains
   `--config target/trunk-lesions/base.json`. Source rules: no `as` casts, no
   `unwrap`/`expect` on I/O in non-test code, functions under about 50 lines, no
   `TODO`, no plan name or `000N-x` slug.
7. **DECISION.md.** Sections: context, reproduction commands (Decisions 1-5, every
   xagent invocation in the `cargo run --release -p xagent-sandbox -- <flags>` form,
   each lineage command in full including `--config target/trunk-lesions/base.json`),
   reference champions (per seed: `champion_node_id` and `champion_generation` from
   the champion reports), the verdict table in the exact format of ARCHITECTURE.md
   "Matrix and decision rule" (header `| part | behaviour | within-life learning |`,
   separator `|---|---|---|`, then the four rows `turn-weights`, `forward-weights`,
   `memory-blend`, `critic` in that order), supporting numbers as `d ± se` per source
   for the metrics and all arms including `all` (`whole.meals`, `whole.deaths`,
   `whole.hazard_deaths`, `whole.sighting_meal_share`, `late.turn_toward_food`,
   `late.turn_away_from_hazard`, the three `late.pathway_food_correlation` values,
   and `late.value_gap_in_view` / `late.value_gap_centred`), and a plain statement
   of what the trunk is for by the verdicts. Plain language, no emoji.
8. **Stop and report.** If no GPU adapter is available, or the measured throughput
   puts the campaign beyond about 3-5 h, stop and report a blocker. Never shrink
   ticks, seeds, agents or generations. Never weaken the record test: on a verdict
   mismatch, correct the `DECISION.md` table to the recomputed value and re-run.

## Steps

**Steps:**

1. Do Decision 1, then run Decision 2's reference lineages and wait until all four exit 0 and each `target/trunk-lesions/reference-sS.db` exists.
2. Run Decisions 3-4: the 48 free lives in background jobs with the limits above, until all 48 files under `target/trunk-lesions/runs/` exist.
3. Do Decision 5 and confirm `lesion-matrix.json` parses as a 48-object array.
4. Write `crates/xagent-sandbox/tests/lesion_matrix_record.rs` per Decision 6. Run `cargo test -p xagent-sandbox --test lesion_matrix_record -- --nocapture` to read the recomputed verdicts; it fails until `DECISION.md` exists, which is expected here.
5. Write `docs/measurements/trunk-lesions/DECISION.md` per Decision 7 with the recomputed verdicts from step 4.
6. Run the checks on the host: focused-proof, sandbox-guard, sandbox-clippy and sandbox-fmt. Correct any verdict mismatch per Decision 8.

- **Done when:** The focused-proof check (`cargo test -p xagent-sandbox --test lesion_matrix_record`) passes, proving INV-10: `lesion-matrix.json` holds 48 reports (6 arms x 2 sources x 4 seeds) each with 10 agents and 300,000 ticks, and the `DECISION.md` verdict table equals the verdicts recomputed by the decision rule, and every `--generations 80` reproduction line in `DECISION.md` carries `--config target/trunk-lesions/base.json`, while sandbox-guard, sandbox-clippy and sandbox-fmt also pass.

## Non-goals

- No product code, existing test or default change; no other file edited.
- No tick logs, committed databases, or database writes outside `target/trunk-lesions/`.
- No shrinking of ticks, seeds, agents or generations; no verdict by hand.
