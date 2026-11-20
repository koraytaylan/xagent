# Pooling study: learned pooling against the hemifields

Rule: INV-06 and the Pooling investigation in
`docs/plans/0003-uniform-sense-motor-route/ARCHITECTURE.md`. Learned pooling is
ported only if its offline score reaches the hemifield score on the same logs.

## Method

- Runner: `crates/xagent-sandbox/examples/pooling_study.rs`, run as
  `cargo run --release -p xagent-sandbox --example pooling_study -- --out-dir <job-dir>`
  and re-fitted with `--fit-only <job-dir>`. `<job-dir>` is an architect-shell job
  directory outside the repository; it holds `seed-<n>.xpol` (pooling log) and
  `seed-<n>.ticklog` (free-life harness tick log) per seed. No log is in the repository.
- Runs: seeds 5, 6, 7, 8; 10 agents; 300,000 ticks; `FullConfig::default()` (the
  current defaults, with both modulators on); fresh birth brains seeded exactly as
  the free-life harness seeds them (world seed, `reset_agents_seeded`,
  `fresh_brain_state`, heritable config). The kernel batch is 10 ticks, so each
  agent has 30,000 brain-tick records per seed (300,000 records per log).
- Life loop: the runner runs its own loop because `headless::run_free_life` has no
  schedule option or per-tick hook. Each brain tick it builds a `FreeLifeSample`
  field for field as `headless.rs` `build_sample` does, feeds it to the harness's
  `AgentAccumulator` and `TickLogWriter`, and takes reward, value, turn
  innovation and food bearing from it. Death is 1 when the sample's death counter
  rose since the previous brain tick. Food in view restates the harness rule:
  alive, distance `<= FOOD_IN_VIEW_DISTANCE` (5.5) and `|bearing| < VIEW_HALF_ANGLE`
  (pi/4). The record's tick is the simulation tick at the batch start.
- Schedule: the kernel runs `BrainExecutionMode::ParallelTiled`, whose feature
  phase writes the adapted (running-mean-subtracted) feature vector to brain
  scratch every brain cycle and whose brain tail reads it back, so the logged
  values are the ones the brain used. The shipping fused schedule keeps them in
  workgroup memory only. The parallel-tiled schedule is the same brain math with
  a different reduction order (bounded drift, not byte-equal), so these lives do
  not reproduce the fused-schedule free lives of the modulator study exactly
  (harness totals here: meals 1047 / 1171 / 940 / 1174 and deaths 459 / 434 /
  483 / 435 over 10 agents for seeds 5 / 6 / 7 / 8).
- Column means: `read_brain_scratch_blocking` per brain tick; per agent the
  scratch row is `brain_scratch_stride` wide, the visual block starts at
  `SCRATCH_FEATURES`, colour of ray `r` channel `k` at `r * 4 + k`, depth at
  `vision_color_count + r`, all from `BrainLayout::from_config`. Column `c`
  channel `k` is the mean over the rows; logged at `c * 4 + k` (red, green, blue,
  depth). Vision width other than 8, stride other than 1 or an enabled visual
  cortex is an error before any log is written.
- Fit, arms, score and gate: the `pooling_study` library, unchanged. A: the 8
  hemifield means (left 4 columns, right 4 columns, per channel); B: the 32
  column means; C: credit-trained 32 -> 8 pooling; D: Oja/Sanger 32 -> 8
  pooling. Every arm feeds the same running whitening (rate 0.01, refresh every
  `VISION_WHITENING_REFRESH` records, epsilon 1e-3) and the kernel's TD(lambda)
  turn rule (discount 0.97, lambda 0.9, rate 0.10, trace zeroed on death). Score
  per agent: correlation of the arm's turn output with the food side over
  food-in-view records from 70% of the run on; agents with fewer than 30 such
  records are excluded, not scored zero. Gate: PORTED iff
  `max(mean C, mean D) >= mean A` over the pooled agents.

## Results

Mean score ± standard error; scored / excluded agents; agents above zero.

| Run | A hemifields | B 32 columns | C credit-trained pooling | D Oja/PCA pooling |
|---|---|---|---|---|
| seed 5 | +0.064 ± 0.079 (10 / 0, >0: 6) | +0.119 ± 0.083 (10 / 0, >0: 8) | -0.025 ± 0.059 (10 / 0, >0: 5) | +0.014 ± 0.057 (10 / 0, >0: 5) |
| seed 6 | +0.156 ± 0.062 (10 / 0, >0: 8) | +0.192 ± 0.049 (10 / 0, >0: 9) | +0.150 ± 0.055 (10 / 0, >0: 6) | +0.138 ± 0.044 (10 / 0, >0: 9) |
| seed 7 | +0.164 ± 0.072 (10 / 0, >0: 7) | +0.234 ± 0.063 (10 / 0, >0: 8) | +0.015 ± 0.033 (10 / 0, >0: 5) | +0.030 ± 0.046 (10 / 0, >0: 6) |
| seed 8 | +0.225 ± 0.054 (10 / 0, >0: 10) | +0.206 ± 0.065 (10 / 0, >0: 9) | +0.172 ± 0.048 (10 / 0, >0: 8) | +0.055 ± 0.054 (10 / 0, >0: 6) |
| pooled | +0.152 ± 0.034 (40 / 0, >0: 31) | +0.188 ± 0.033 (40 / 0, >0: 34) | +0.078 ± 0.027 (40 / 0, >0: 24) | +0.059 ± 0.025 (40 / 0, >0: 26) |

The `--fit-only <job-dir>` re-run printed the same table, gate line and verdict.

## Gate

- mean A (hemifields) = +0.1521
- mean C (credit-trained pooling) = +0.0781
- mean D (Oja/PCA pooling) = +0.0590
- max(C, D) = +0.0781 (arm C) < +0.1521 = mean A, so the gate fails.

Winning learned arm: C, credit-trained pooling; it does not reach the hemifields.
The best arm overall is B (32 columns), which is not a pooling arm and is not
judged by the gate.

## Sanity

- Arm B: +0.188 against the reference +0.16 (miss 0.028): consistent.
- Arm A: +0.152 against the reference +0.33 (miss 0.178): sanity failure, stated
  for the architect. The hemifield arm scores about half its earlier offline
  reference on these logs. Likely contributors are the different learner
  (current defaults with both modulators, versus the earlier investigation's
  learner) and the parallel-tiled schedule; the gate compares all arms on the
  same logs, so it is unaffected, but the absolute A score should not be read as
  a reproduction of the earlier +0.33.

Verdict: NOT PORTED
