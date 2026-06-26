# Scope — Plan 0019

> Fix the mechanical gate-integrity regressions and re-frame the spike
> mechanisms honestly to restore confidence in falsification discipline — none
> flips a default or changes shipped signal.

## Why this plan

A post-merge audit of `develop` after Plans 0017/0018 found that two mechanical
gate-integrity defects (a red `cargo fmt` gate and a cortex-budget skip on real
GPU) and three falsification re-framing concerns (an orphaned A/B harness and two
spikes tested at the wrong level) landed without process catching them. Each was
re-verified against the committed source before inclusion here. None is a research
gap; all are hygiene or documentation truthfulness.

1. **The `cargo fmt` gate is red on `develop` at `1da6e0f5`
   (`ui.rs:1142-1145`).** The UI texture-key assignment carries line breaks that
   rustfmt wants to collapse to single lines. `CONTRIBUTING.md` lists `cargo fmt`
   as the first CI gate; a red gate on `develop` means the pre-push gate was
   skipped or CI did not run. Pure formatting — zero logic impact — but it leaves
   the keystone gate failing on the mainline (`crates/xagent-sandbox/src/ui.rs:1142-1145`).

2. **The 0017 cortex throughput budget was met on lavapipe only; real GPU sits at
   1.1% vs the 50% target.** The committed
   `CORTEX-PROFILE-BASELINE.txt` records Metal baseline ≈ 7,625 tps and cortex-ON
   ≈ 83 tps (1.1%, ≈ 93× slower); the test self-skips on real GPU with a detailed
   18/256-lane-occupancy rationale and asserts the budget only on the software
   adapter. STATUS claims **Done**, so a reader concludes cortex is
   production-viable at 50% throughput when real hardware is 45× below target
   (`docs/plans/0017-Credit-Path-And-Emergent-Visual-Evolution/CORTEX-PROFILE-BASELINE.txt`,
   `crates/xagent-sandbox/tests/integration.rs:703`).

3. **The 0013 A/B harness `run_innate_instinct_ab` has no regression test.** The
   harness was run once on local Metal and produced a FAILED verdict recorded in
   the decision doc; no integration test invokes it. The harness is orphaned — if
   the learning signal changes, the gate cannot be re-run mechanically to confirm
   the failure still holds (`crates/xagent-sandbox/src/headless.rs:492`).

4. **The 0018 auxiliary-loss spike was not tested at the GPU-integrated level.**
   `auxiliary_steering_loss_converges_on_bearing` measures a CPU-side overlay with
   no GPU gradient injection; the decision doc correctly notes this. Yet the STATUS
   roll-up's **3/3 REJECT** framing can lead a reader to cite this as a
   falsification of auxiliary loss *as a mechanism*, when the truthful statement is
   that a CPU overlay — not the GPU-integrated mechanism — was tested
   (`crates/xagent-sandbox/tests/integration.rs:8806`).

5. **The 0018 frame-sync spike is disabled at `vision_stride=1`, so it was never
   tested under its activation condition.** The unit test confirms frame-sync is
   active at `vision_stride=10` (trace +54%), but the steering probe trains at
   `vision_stride=1`, where the mechanism is disabled. The probe never exercised
   the mechanism, yet the result is framed as a clean falsification
   (`docs/plans/0018-Credit-Path-Mechanism-Attack/0002-TRACE-HORIZON-DECISION.md:45`).

6. **Diagnostic tables in the probes print under `--nocapture`.**
   `cortex_throughput_profile_baseline`, the credit probes, and the auxiliary-loss
   tests all print large tables on every run. The `2026-06-19` grok-43 review
   recommended gating such output behind an env var so tests stay silent and
   falsifying-only by default (`crates/xagent-sandbox/tests/integration.rs:658`).

7. **Test file paths in STATUS docs are easy to misread.**
   `crates/xagent-brain/tests/intent_baseline_measurement.rs` and
   `crates/xagent-brain/tests/danger_percept_ablation_ab.rs` are referenced by bare
   filename in plan docs, but the project now has two test crates and omitting the
   crate prefix risks wrong-directory searches
   (`docs/plans/0015-Intent-Awareness-Measurement-Framework/STATUS.md`).

**Provenance.** Every finding re-verified against `develop` source: the
`ui.rs:1142-1145` assignments do trip rustfmt (confirmed); `CORTEX-PROFILE-BASELINE.txt`
records the real-GPU 1.1% number while the test asserts the budget only on
lavapipe (confirmed); `run_innate_instinct_ab` has no integration caller
(confirmed); the auxiliary-loss test measures a CPU overlay and the frame-sync
probe trains at `vision_stride=1` (both confirmed). The deeper credit-path
research gap and the cortex workgroup restructuring are addressed elsewhere or
deferred (see Out of scope).

## In scope

- **0001 — Fmt-Gate-Restore.** Run `cargo fmt` to collapse the `ui.rs:1142-1145`
  line breaks, verify `develop` is fmt-clean, and confirm CI enforces the fmt
  gate. Pure formatting, zero logic impact. See [TASKS.md](TASKS.md).
- **0002 — Cortex-Throughput-Honesty.** Correct the 0017 STATUS WS0002 row to
  state the real-GPU 1.1% budget miss and the lavapipe-only scope of the 50%
  assertion, and rename the cortex throughput test to reflect its software-adapter
  scope. Documentation only; cortex remains default-off. See [TASKS.md](TASKS.md).
- **0003 — Innate-Instinct-Harness-Regression-Test.** Add an integration test
  that invokes `run_innate_instinct_ab` with a tiny generation count, asserts
  well-formed `ValidationStats`, and guards against harness rot — guarding logic,
  not re-asserting the gate result. GPU-gated with graceful self-skip. See
  [TASKS.md](TASKS.md).
- **0004 — Auxiliary-Loss-Mechanism-Honesty.** Re-frame the 0018-0001
  auxiliary-loss decision as *mechanism not tested at the GPU-integrated level*,
  not falsified, and set the gate for a future GPU-integrated attempt.
  Documentation only. See [TASKS.md](TASKS.md).
- **0005 — Frame-Sync-Spike-Clarity.** Re-frame the 0018-0002 frame-sync spike as
  *not tested under its activation condition*, not falsified, and add a
  when-to-revisit note for sparse-stride evaluation. Documentation only. See
  [TASKS.md](TASKS.md).
- **0006 — Diagnostic-Table-Silencing.** Gate the cortex / credit / auxiliary
  probe `eprintln!` tables behind `XAGENT_VERBOSE_PROBES` so the tests are silent
  by default and the tables remain available on demand; assertions unchanged. See
  [TASKS.md](TASKS.md).
- **0007 — Test-File-Path-Traceability.** Update test file references in the plan
  STATUS and ARCHITECTURE docs to use full crate-relative paths (e.g.
  `crates/xagent-brain/tests/...`) so no reference is ambiguous about which crate
  owns which test. Documentation only. See [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| `cargo fmt` gate red on `develop` at `ui.rs:1142-1145` (1) | `0001` |
| 0017 cortex throughput 1.1% vs 50% target on real GPU (2) | `0002` |
| 0013 A/B harness `run_innate_instinct_ab` has no regression test (3) | `0003` |
| 0018 auxiliary-loss spike CPU-side only, not GPU-integrated (4) | `0004` |
| 0018 frame-sync spike disabled at `vision_stride=1` (5) | `0005` |
| Diagnostic tables print under `--nocapture` (6) | `0006` |
| Test file paths in STATUS docs easy to misread (7) | `0007` |

## Locked decisions

- **No default flips; all changes are hygiene or documentation.** None of the
  fixes flips a default or changes the shipped learning signal. The fmt fix is
  pure formatting; the cortex STATUS corrections are honest accounting only
  (cortex stays default-off); the A/B harness test guards logic without
  re-asserting the gate; the diagnostic tables are silenced for CI clarity; the
  spike re-framings are documentation only. Behavior is unaffected. **Gate:** all
  tasks complete.
- **Spike decisions re-framed, not reversed.** 0018-0001 and 0018-0002 are
  documented as *mechanism not tested at the relevant level* (or under its
  activation condition), not falsified. If GPU-integrated auxiliary loss or
  sparse-stride frame-sync is pursued in 0020, those tasks reference this plan's
  decision docs for scope and when-to-revisit gates. **Gate:** no code reverts or
  behavior changes — documentation-only re-framing for truthfulness.
- **Diagnostic tables optional for local dev; CI silent.**
  `XAGENT_VERBOSE_PROBES` defaults unset, so CI runs with silent probes
  (assertions only) and local developers set the var to see diagnostics while
  debugging. **Gate:** assertions stay unchanged and falsifiable; the env var only
  toggles output, aligning with the prior fitness-probe cleanup.

## Out of scope

- **The credit-path research hypothesis (triple-falsified steering, no new
  mechanism named).** This is a research-hypothesis gap, not a mechanical defect.
  0018 recorded three falsified mechanisms and the steering goal is triple-falsified
  with the mechanism in the chance band; naming a new hypothesis class is deferred
  to plan 0020.
- **Workgroup restructuring for cortex (7% lane occupancy, 18/256 threads).** The
  real-GPU cortex throughput fix requires restructuring the 256-lane workgroup to
  use more lanes — significant engineering, out of scope per 0017's SCOPE.md, and
  deferred or deprioritized versus the legacy raycast encoder.
- **Lever graduation decisions (`effort_rebased_fitness`,
  `danger_percept_enabled`).** 0014 fixed the effort-fitness math and 0010/0013
  carry decision docs; whether to flip those levers on production seeds is a
  separate decision (1–2 weeks of harness runs), deferred to 0020 or a parallel
  track.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
