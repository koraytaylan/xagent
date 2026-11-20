# Modulator study (plan 0003, task 0402; INV-07, INV-08)

Rule: `docs/plans/0003-uniform-sense-motor-route/ARCHITECTURE.md` (Investigations: Modulators), mirroring plan 0002 D9. Each modulator is judged against the control alone.

## Setup

- Reference config: `docs/decisions/modulator-study-evo-reference.json` (sha256 `6ad3b9a254e52fa53e6ab3c9b71387d942aaa9118d965eac132fc64d69304f67`), dumped at commit `c6e4650c9fea0c0a39a3a930be6189238fc1be40` (`git rev-parse HEAD` at dump time). Commits after it up to this record change only plan documents (output-path amendment), not code.
- Commands: `xagent --dump-config > default.json`; `jq '.governor.tick_budget = 40000' default.json > modulator-study-evo-reference.json` (only difference from default: `tick_budget` 1000000 -> 40000). Parsed as `FullConfig` via `--config`, which every run below loaded.
- Arms (scratch copies under `target/modulator-study/`, ignored): control (both flags `true`, the defaults); klino-off (`klinotaxis_turn_gain_enabled` false); fatigue-off (`motor_fatigue_enabled` false). Each differs from the reference in that one key only.
- Within-life: `xagent --config <arm>.json --seed S --free-life --free-life-agents 10 --free-life-ticks 300000 --free-life-out <out>.json`, seeds 5-8 (40 agents per arm, fresh birth brains).
- 80-generation: `xagent --seed S --no-render --generations 80 --config <arm>.json --db target/modulator-study/gen/<arm>-sS.db`, a fresh database per run; `composite_fitness` read from the governor schema for generations 60-78. All 24 runs exited 0; every database reached generation 79.
- SE: Welch standard error of the arm-minus-control difference over the 40 per-agent values of each metric (per-agent share for the sightings share).

## Within-life per seed

| Arm | Seed | Meals/agent | Deaths/agent | Hazard deaths/agent | Sightings | Ending in meal | Share (pooled) | Turn toward food | Turn away from hazard |
|---|---|---|---|---|---|---|---|---|---|
| control | 5 | 167.3 | 35.2 | 21.5 | 2566 | 1506 | 0.5869 | 0.0067 | 0.0107 |
| control | 6 | 203.0 | 29.7 | 20.3 | 2931 | 1854 | 0.6325 | 0.0144 | 0.0096 |
| control | 7 | 144.4 | 40.0 | 25.3 | 2201 | 1314 | 0.5970 | 0.0085 | 0.0064 |
| control | 8 | 196.9 | 34.6 | 25.1 | 2975 | 1795 | 0.6034 | 0.0088 | 0.0087 |
| klino-off | 5 | 158.2 | 40.6 | 26.1 | 2295 | 1445 | 0.6296 | 0.0111 | 0.0052 |
| klino-off | 6 | 205.7 | 44.9 | 34.2 | 2902 | 1823 | 0.6282 | 0.0124 | 0.0061 |
| klino-off | 7 | 134.1 | 41.8 | 26.8 | 1988 | 1219 | 0.6132 | 0.0099 | 0.0045 |
| klino-off | 8 | 139.8 | 42.1 | 26.7 | 2107 | 1261 | 0.5985 | 0.0085 | 0.0050 |
| fatigue-off | 5 | 239.5 | 160.1 | 144.0 | 3173 | 1343 | 0.4233 | -0.0099 | -0.0014 |
| fatigue-off | 6 | 239.1 | 156.9 | 140.2 | 3232 | 1346 | 0.4165 | -0.0060 | 0.0191 |
| fatigue-off | 7 | 238.2 | 177.7 | 160.5 | 3353 | 1470 | 0.4384 | 0.0240 | 0.0250 |
| fatigue-off | 8 | 230.7 | 168.2 | 149.5 | 3054 | 1237 | 0.4050 | 0.0174 | -0.0092 |

## 80-generation per seed (generations 60-78)

| Arm | Seed | Mean composite_fitness | Mean meals/agent | Mean deaths/agent | Max generation in DB |
|---|---|---|---|---|---|
| control | 5 | 0.2064 | 46.21 | 0.36 | 79 |
| control | 6 | 0.2236 | 62.07 | 1.33 | 79 |
| control | 7 | 0.1546 | 43.19 | 1.66 | 79 |
| control | 8 | 0.2848 | 65.51 | 0.35 | 79 |
| klino-off | 5 | 0.2624 | 63.41 | 0.61 | 79 |
| klino-off | 6 | 0.3021 | 72.08 | 0.44 | 79 |
| klino-off | 7 | 0.2177 | 50.12 | 0.58 | 79 |
| klino-off | 8 | 0.3284 | 86.43 | 0.74 | 79 |
| fatigue-off | 5 | 0.1973 | 83.45 | 4.57 | 79 |
| fatigue-off | 6 | 0.1566 | 66.03 | 5.38 | 79 |
| fatigue-off | 7 | 0.1288 | 63.76 | 7.59 | 79 |
| fatigue-off | 8 | 0.1671 | 85.14 | 6.98 | 79 |

## Rule: Klinotaxis turn-gain (klino-off vs control)

- Clause 1: mean meals per agent arm 159.450 - control 177.900 = -18.450; Welch SE 17.719; needs >= -SE (-17.719): fail
- Clause 2: mean deaths per agent arm 42.350 - control 34.875 = 7.475; Welch SE 1.779; needs <= +SE (+1.779): fail
- Clause 3: pooled sighting meal share arm 0.6186 vs control 0.6061 - SE 0.0185 (Welch SE of per-agent shares) = 0.5876: pass
- Clause 4: mean composite_fitness generations 60-78, arm vs control per seed: seed 5 0.2624 vs 0.2064 (>=); seed 6 0.3021 vs 0.2236 (>=); seed 7 0.2177 vs 0.1546 (>=); seed 8 0.3284 vs 0.2848 (>=): pass
- Verdict: REVERT

## Rule: Motor fatigue (fatigue-off vs control)

- Clause 1: mean meals per agent arm 236.875 - control 177.900 = 58.975; Welch SE 22.754; needs >= -SE (-22.754): pass
- Clause 2: mean deaths per agent arm 165.725 - control 34.875 = 130.850; Welch SE 13.183; needs <= +SE (+13.183): fail
- Clause 3: pooled sighting meal share arm 0.4212 vs control 0.6061 - SE 0.0177 (Welch SE of per-agent shares) = 0.5884: fail
- Clause 4: mean composite_fitness generations 60-78, arm vs control per seed: seed 5 0.1973 vs 0.2064 (<); seed 6 0.1566 vs 0.2236 (<); seed 7 0.1288 vs 0.1546 (<); seed 8 0.1671 vs 0.2848 (<): fail
- Verdict: REVERT

## Verdicts

Klinotaxis turn-gain: REVERTED
Motor fatigue: REVERTED

Both are reverted, so no code changed: `crates/xagent-shared/src/config.rs`, `crates/xagent-sandbox/tests/modulator_flags.rs` and the parity golden are untouched, and the golden was not re-recorded. Both flags stay `true` by default.
