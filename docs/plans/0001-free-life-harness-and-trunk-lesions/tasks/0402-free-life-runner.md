---
id: free-life-runner
title: Run a seeded free life from a fresh or stored champion birth brain and report per-agent metrics
workstream: "0400"
kind: task
role: developer
depends_on: [kernel-lesions-and-probe, read-only-champion-loader, free-life-metrics-and-log]
gated: false
touches: [crates/xagent-sandbox/src/headless.rs, crates/xagent-sandbox/tests/free_life.rs]
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test free_life"
    timeout_ms: 3600000
  - name: headless-unit-tests
    command: "cargo test -p xagent-sandbox --lib headless::tests"
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

# Run a seeded free life and report per-agent metrics

## Outcome

- `crates/xagent-sandbox/src/headless.rs` gains the free-life API exactly as
  ARCHITECTURE.md section "Free-life run (free-life-runner)" specifies: the types,
  the two constants, `parse_lesions`, `lesion_names`, `FreeLifeError` (including
  the `ReadbackLength` variant), `FreeLifeReport` and `run_free_life`. The
  `pub` fields and derives that section pins on `BrainTickProbe` and the metrics
  types are required here too.
- `run_free_life` runs one seeded free life; identical options and config give an
  identical report (INV-8). Lesion and probe setters are called only from
  `run_free_life` (INV-2).
- `run_headless` and every other path are not edited.

## Consumed APIs

Use the names and signatures in ARCHITECTURE.md sections "Kernel measurement
layout", "Lesion branches and probe", "Read-only champion loader" and
"Free-life types". Do not restate them here.

## Decisions

1. Validation runs in the order of ARCHITECTURE "Free-life run" step 1, GPU
   check first. Because of that, the option-rejection tests need a GPU host and
   live in `tests/free_life.rs` under the focused proof.
2. The probe readback runs before the physics readback. The probe read borrows
   `&self` and the physics read borrows `&mut self`, so this order keeps both
   borrows valid.
3. `dispatch_batch` returns `true` by contract. Bind the result to `_dispatched`
   and say so in a comment.
4. A probe or physics length mismatch returns `ReadbackLength`; it never panics
   and never writes short data.
5. The lesion-parsing tests are CPU-only but live in `tests/free_life.rs`, without
   the GPU guard, so one command proves the whole API. `headless::tests` gains no
   new tests; its existing tests stay as the `headless-unit-tests` regression check.
6. Pedantic clippy is on: no `as` between integers or to float (use `try_from`,
   `from`, `checked_mul`), no `expect` or `unwrap` in non-test code, helpers under
   about 50 lines. No plan, task or workstream name appears in source, comments or
   strings.

## Steps

**Steps:**

1. In `headless.rs`, add the public types, the two constants, `FreeLifeError`,
   `FreeLifeReport`, `parse_lesions` and `lesion_names` as ARCHITECTURE specifies.
2. Add `run_free_life` in the validation order above: birth source and layout,
   vision stride, world and kernel build, then ticks against `kernel_batch_size()`.
   Nothing is uploaded or dispatched before validation passes.
3. Seed the birth brain from one `SmallRng`, write it and the heritable config to
   every agent, call `set_trunk_lesions(options.lesions)` and
   `set_brain_tick_probe(true)`, then create the optional log writer and one
   accumulator per agent.
4. Write the batch loop: dispatch, probe readback, physics readback, one sample per
   agent pushed to the accumulator and the writer. Keep the sample builder and the
   per-tick feed in separate helpers. Finish the accumulators and writer into the
   report.
5. Write `crates/xagent-sandbox/tests/free_life.rs` with these tests, named
   constants for every number, and a GPU helper that prints
   `skipping: no GPU adapter` unless `XAGENT_REQUIRE_GPU=1`, when it panics. CPU
   tests: `free_life_parses_lesion_lists`, `free_life_rejects_unknown_lesion_names`
   (`Critic`, empty item), `free_life_lesion_names_are_stable`. GPU tests:
   `free_life_is_reproducible_for_identical_options`,
   `free_life_different_seed_gives_different_report`,
   `free_life_brain_ticks_and_samples_match_ticks`,
   `free_life_lesions_change_report_and_are_listed`,
   `free_life_leaves_fresh_kernel_with_lesions_off`,
   `free_life_champion_source_reports_node_and_generation`,
   `free_life_champion_wrong_layout_is_rejected`,
   `free_life_missing_champion_node_is_reported`,
   `free_life_log_matches_report`,
   `free_life_rejects_vision_stride_other_than_one`,
   `free_life_rejects_agent_counts_outside_range`,
   `free_life_rejects_ticks_not_a_multiple_of_batch`. Use 2-4 agents and
   2,000-5,000 ticks rounded to a batch multiple. The champion fixture is a
   `TempDir` governor database whose current node holds `fresh_brain_state`
   stored with `store_champion_brain`; read the generation with `rusqlite`.
6. Fix findings from `focused-proof`, `sandbox-clippy` and `sandbox-fmt` in the two
   touched files only.

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test free_life` passes on the GPU host with every test named in Step 5 reported ok, proving INV-8 (equal options give equal reports, and every sample is one brain tick) and INV-2 (lesions reach the kernel only from run_free_life, and a fresh kernel starts with lesions and the probe off).

## Non-goals

- No change to `run_headless`, `Governor`, the kernel, WGSL, the metrics or tick-log
  modules, `main.rs` or any test file other than `tests/free_life.rs`. CLI flags
  belong to the CLI task.
- No learning, reward, world, default-value or database-write change. The champion
  is read only.
- No matrix, decision record or reference-run work.
- No setup checks, no dependency installation and no shell use; the host runs every
  check.
