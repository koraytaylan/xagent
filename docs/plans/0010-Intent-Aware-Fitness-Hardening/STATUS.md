# Plan 0010 — Intent-Aware Fitness Hardening — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** ✅ Complete. All 21 tasks landed via `implement-plan` and squash-merged
into `develop` as `b291dafb`. Authored from the four 2026-06-18 reviews
(Claude Opus 4.8, Grok 4.3, GPT-5 Codex, Gemini 3.1 Pro High) of Plan 0009; every
finding fact-checked against `claude/funny-cray-84end5` @ `5ddc976` before being
turned into a task (four review claims rejected during verification — see
[SCOPE.md](SCOPE.md)). `gate-includes-avoidance-floor` carried no unique diff — its
scope (the `AVOIDANCE_FLOOR` conjunct + `gate_rejects_avoidance_below_floor` test)
landed inside `harden-gate-predicate`. Recovered from a first parallel run whose
fused/split-shader edits collided; re-run serially (`maxParallel:1`) to convergence.
_Last updated: 2026-06-19, against `develop`._

- **Goal:** Make Plan 0009's flag-gated effort/danger machinery safe to graduate
  without flipping any default — re-derive the effort-fitness calibration from real
  recorded telemetry so the foraging/exploration axes stop collapsing to ≈0 at
  production scale, harden the speed-decoupling gate so it cannot pass on a
  worsening metric, fix the avoidance-intent percept so it measures avoidance
  rather than its opposite, and repair the acceptance-named tests that never assert
  their titular property.
- **Measured baseline (current code):** the default 0009 build is correct and
  93/93 green, but the effort axes collapse to ≈0.02 across the population at the
  1M-tick budget (`food/energy ≈ few-hundred / ~15,000`; `cells_per_dist ≈ 0.025`),
  `P_ENERGY_SPENT` omits the brain metabolic drain (18–62% under-count, scaling
  with brain size), the speed-decoupling gate passed while its own doc says the
  correlation rose 0.2238 → 0.2569, the A/B is uncontrolled (three unseeded RNG
  sites), two of three gate conjuncts carry no signal and the gate has zero unit
  tests, the avoidance-intent counter measures turning *toward* danger, the danger
  ring-scan runs unconditionally on the flag-off path, and three acceptance-named
  tests assert something other than their titular property.
- **Root cause:** Plan 0009 calibrated effort-rebased fitness against synthetic
  per-life profiles while production feeds it cumulative-per-generation
  accumulators 100–1000× larger, and validated the flag-flip with a gate that
  overclaims its evidence (passes on a worsening metric, uncontrolled A/B, two
  uninformative conjuncts, no unit tests). The percept/test defects are latent
  because the counter is observability-only and the defaults ship the flags off.
- **Approach:** four parallel workstreams on a largely disjoint file set — (0001)
  brain-drain accounting + recorded-replay recalibration + production-scale
  assertion; (0002) gate rewrite (strong baseline + strict improvement + uncapped
  viability + seeded paired A/B + unit tests + metadata); (0003) avoidance-intent
  sign/timing fix + scan flag-gating + `atan2` guard; (0004) repair the three
  acceptance-named tests + `WC_*`/`CFG_*` parity + serde defaults + migration order
  + stale-layout fixes + counter-persistence decision + magic-number naming.
- **Outcome:** ✅ All 21 tasks landed and squash-merged into `develop` (`b291dafb`).
  All flags stay default-off / byte-identical. Gate green: fmt + clippy clean,
  217 tests pass (lib 104, bin 14, contributing-guard 2, integration 97). The
  previously-failing `split_matches_fused_effort_telemetry` and
  `speed_cost_exponent_default_is_noop` (a fused/split divergence surfaced by the
  brain-drain accounting) pass after the serial recalibration. Several non-blocking
  doc/comment nits remain — see the "Follow-up nits" note below.

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Effort-fitness scale-invariance & recalibration | `brain-drain-energy-accounting`, `recorded-generation-calibration-replay`, `effort-axes-recalibration`, `calibration-test-falsifiability` | ✅ Done |
| 0002 | Decision-machinery hardening | `harden-gate-predicate`, `seeded-paired-ab-harness`, `gate-machinery-unit-tests`, `gate-includes-avoidance-floor` (folded into `harden-gate-predicate`), `share-danger-reduction`, `decision-doc-metadata-and-temp-cleanup` | ✅ Done |
| 0003 | Danger-percept correctness | `avoidance-intent-sign-timing`, `danger-percept-gpu-tests`, `danger-scan-gate-and-atan2-guard` | ✅ Done |
| 0004 | Test, layout & migration integrity | `recorded-telemetry-persistence-test`, `flag-off-byte-identity-golden`, `wc-cfg-constant-parity`, `serde-default-off-coverage`, `behavior-metric-migration-order`, `stale-layout-doc-fixes`, `avoidance-counter-persistence`, `shader-magic-number-naming` | ✅ Done |

## Verification

✅ Done. `0001`+`0002` landed, so the corrected speed-decoupling gate can now be
re-run to decide `0009`'s `default-flip-gate` on trustworthy evidence: the foraging
axis is re-based off real recorded telemetry (not ≈ 0.02), the gate requires a
strongly-positive baseline and strict improvement against a seeded paired A/B, the
gate math has GPU-free unit tests, and the avoidance-intent metric now measures
genuine turn-aways (sign inversion fixed). No 0009 default is flipped by this plan;
running the gate to a flip decision is the follow-on `0009 default-flip-gate` task.

## Follow-up nits (non-blocking, surfaced by reviewers — not yet addressed)

- `governor.rs` ~L2937: stale doc-comment still names `EXPLORATION_DISTANCE_BUDGET`
  (renamed to `EXPLORATION_RATE_TARGET`).
- `governor.rs` ~L4880/L4884: a diagnostic comment cites an inaccurate old-constant
  number for the competent-forager profile.
- `headless.rs` ~L1170: the `INCONCLUSIVE` decision branch renders the literal
  `{baseline_min:.2}` instead of the value (cosmetic; offline-tool prose only).
- `docs/plans/0009-Intent-Aware-Fitness/0009-FITNESS-CALIBRATION.md`: now stale after
  the recalibration (shows pre-Variant-B numbers + `…/1000` grid; should be `…/1024`).
- Pre-existing planning-reference comments touched in `kernel_tick.wgsl` /
  `phase_physics.wgsl` were not stripped on-touch — covered by Plan 0011.
