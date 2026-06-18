# Plan 0010 — Intent-Aware Fitness Hardening — status

Task-level execution status for this plan. Keep it current as tasks land, and
keep the roll-up row in [`../STATUS.md`](../STATUS.md) in sync.

**Status:** 📋 Planned. Authored from the four 2026-06-18 reviews
(Claude Opus 4.8, Grok 4.3, GPT-5 Codex, Gemini 3.1 Pro High) of Plan 0009; every
finding fact-checked against `claude/funny-cray-84end5` @ `5ddc976` before being
turned into a task (four review claims rejected during verification — see
[SCOPE.md](SCOPE.md)). No task started.
_Last updated: 2026-06-18, against `claude/funny-cray-84end5`._

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
- **Outcome:** _Pending — not started._

| WS | Workstream | Tasks | State |
|---|---|---|---|
| 0001 | Effort-fitness scale-invariance & recalibration | `brain-drain-energy-accounting`, `recorded-generation-calibration-replay`, `effort-axes-recalibration`, `calibration-test-falsifiability` | 📋 Planned |
| 0002 | Decision-machinery hardening | `harden-gate-predicate`, `seeded-paired-ab-harness`, `gate-machinery-unit-tests`, `gate-includes-avoidance-floor`, `share-danger-reduction`, `decision-doc-metadata-and-temp-cleanup` | 📋 Planned |
| 0003 | Danger-percept correctness | `avoidance-intent-sign-timing`, `danger-percept-gpu-tests`, `danger-scan-gate-and-atan2-guard` | 📋 Planned |
| 0004 | Test, layout & migration integrity | `recorded-telemetry-persistence-test`, `flag-off-byte-identity-golden`, `wc-cfg-constant-parity`, `serde-default-off-coverage`, `behavior-metric-migration-order`, `stale-layout-doc-fixes`, `avoidance-counter-persistence`, `shader-magic-number-naming` | 📋 Planned |

## Verification

_Pending._ The plan's success criterion is that, after `0001`+`0002` land, the
corrected speed-decoupling gate can be re-run to decide `0009`'s `default-flip-gate`
on trustworthy evidence: a competent forager reaches foraging ≈ 1.0 on real
recorded telemetry (not ≈ 0.02), the gate requires a strongly-positive baseline and
strict improvement against a seeded paired A/B, and the avoidance-intent metric
measures genuine turn-aways. No 0009 default is flipped by this plan.
