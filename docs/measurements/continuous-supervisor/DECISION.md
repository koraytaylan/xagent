Verdict: REJECT

# Continuous-life supervisor: decision record

## Question

Does the continuous-life supervisor (Darwinian inheritance) do at least as well
as generation mode on food intake, deaths and within-life learning, at equal
agent-tick budgets, so that it could replace generation mode?

## Rule

Quoted from PLAN.md "Decision rule", fixed before any run and encoded in the
`supervisor_verdict` binary:

> Runs use seeds 5, 6, 7 and 8, 10 agents and exactly 32,000,000 agent-ticks per
> run (80 generations of 40,000 ticks in generation mode; 3,200,000 ticks in
> supervisor mode). The base config is the `--dump-config` default with
> `governor.tick_budget = 40000`; every supervisor setting stays at its default.
>
> Metrics (supervisor `S` against generations `G`):
>
> - Per seed: food per agent-tick `F = nourishing meals / agent-ticks` and deaths
>   per agent-tick `D = deaths / agent-ticks`, from the food-kinds-off runs.
> - Within-life learning, pooled over the four seeds, late-30% windows:
>   - from the free lives: `L1 = Σ late.meals / Σ late.alive_samples`,
>     `L2 = mean late.turn_toward_food`, `L3 = mean late.sighting_meal_share`
>     (means over all 40 agents);
>   - from the food-kinds runs, late window = last 30% of each season (a
>     generation is a season in generation mode): `K1` = pooled mean turn toward
>     the nourishing kind, `K2` = pooled mean turn away from the harmful kind,
>     `K3 = Σ late nourishing meals / Σ late meals`.
>
> Verdict:
>
> - `ADOPT` when `F_S ≥ F_G` and `D_S ≤ D_G` together hold on at least 3 of the 4
>   seeds, and `L1–L3` and `K1–K3` of `S` are each `≥` those of `G`.
> - `OPTION` when not `ADOPT`, but nothing gets worse: pooled `F_S ≥ F_G`, pooled
>   `D_S ≤ D_G` (pooled = sums over the four seeds) and every `L` and `K` of `S`
>   is `≥` that of `G`.
> - `REJECT` otherwise.
>
> The `weight-copy` arm is measured and reported with the same metrics but never
> enters the verdict and is never adopted.

## Setup

- Build profile: `release` (`cargo build --release -p xagent-sandbox`).
- GPU: bench line `[bench] Using GpuKernel (10 agents)` from
  `--bench --bench-ticks 2000 --bench-agents 10` (585 ticks/sec); the bench
  line names the GPU kernel backend, not a vendor adapter string.
- Configs: `--dump-config` default (`config/default.json`); `base.json` changes
  only `"tick_budget": 1000000,` to `"tick_budget": 40000,`; `base-kinds.json`
  additionally sets `"food_kinds_enabled": true` and
  `"second_odour_sense_enabled": true`. Supervisor settings at defaults.
- In-run jobs, seeds 5-8, each with `--config <cfg> --seed S --no-render
  --agent-tick-budget 32000000 --db target/continuous-supervisor/db/<name>-sS.db
  --run-summary target/continuous-supervisor/summaries/<name>-seedS.json`:

| Run family | Config | Extra flags | Wall time per run (s, seeds 5/6/7/8) |
|---|---|---|---|
| generations | base | `--evolution-mode generations` | 872 / 874 / 874 / 874 |
| supervisor | base | `--evolution-mode supervisor` | 732 / 734 / 734 / 734 |
| weight-copy | base | `--evolution-mode supervisor --supervisor-inheritance weight-copy` | 704 / 705 / 705 / 705 |
| generations-kinds | base-kinds | `--evolution-mode generations` | 769 / 768 / 768 / 768 |
| supervisor-kinds | base-kinds | `--evolution-mode supervisor` | 770 / 768 / 770 / 770 |

- Free lives, base config, `--free-life --free-life-agents 10
  --free-life-ticks 300000 --seed S --free-life-out
  target/continuous-supervisor/summaries/free-life-<family>-seedS.json`:

| Free life | Source flag | Wall time per run (s, seeds 5/6/7/8) |
|---|---|---|
| free-life-generations | `--free-life-champion-db .../generations-sS.db` | 86 / 86 / 86 / 86 |
| free-life-supervisor | `--free-life-archive-db .../supervisor-sS.db` (rank 0) | 79 / 78 / 78 / 78 |
| free-life-weight-copy | `--free-life-archive-db .../weight-copy-sS.db` (rank 0) | 71 / 70 / 70 / 71 |

- Jobs ran in batches of four. Every run exited 0 on its first attempt; no run
  was retried.
- Verdict: `cargo run --release -p xagent-sandbox --bin supervisor_verdict --
  target/continuous-supervisor/summaries --write-bundle
  docs/measurements/continuous-supervisor/results.json` over the 32 summaries.

## Per-seed table

Every run: 32,000,000 agent-ticks. F = nourishing meals / agent-ticks,
D = deaths / agent-ticks (from the run summaries).

| Seed | F supervisor | F generations | F weight-copy | D supervisor | D generations | D weight-copy | F_S ≥ F_G and D_S ≤ D_G |
|---|---|---|---|---|---|---|---|
| 5 | 0.001220 (39032) | 0.001226 (39229) | 0.001808 (57869) | 0.000033 (1049) | 0.000022 (715) | 0.000035 (1117) | no |
| 6 | 0.001058 (33852) | 0.001504 (48133) | 0.001775 (56798) | 0.000071 (2282) | 0.000040 (1294) | 0.000037 (1189) | no |
| 7 | 0.001064 (34045) | 0.001046 (33456) | 0.001497 (47913) | 0.000043 (1389) | 0.000046 (1468) | 0.000031 (982) | yes |
| 8 | 0.001235 (39525) | 0.001430 (45767) | 0.001796 (57479) | 0.000049 (1582) | 0.000028 (898) | 0.000044 (1411) | no |
| pooled | 0.001144 | 0.001301 | 0.001719 | 0.000049 | 0.000034 | 0.000037 | |

## Learning table

From the verdict output.

| Metric | supervisor | generations | weight-copy |
|---|---|---|---|
| L1 | 0.014156 | 0.014097 | 0.015658 |
| L2 | 0.077360 | 0.065371 | 0.079293 |
| L3 | 0.733065 | 0.691193 | 0.739407 |
| K1 | 0.010719 | 0.019780 | n/a |
| K2 | -0.010709 | -0.020261 | n/a |
| K3 | 0.511436 | 0.496561 | n/a |

## Outcome

- ADOPT seed clause: F_S ≥ F_G and D_S ≤ D_G held together on 1 of 4 seeds
  (seed 7 only); the rule needs 3. Not held.
- Pooled F_S ≥ F_G: 0.001144 < 0.001301. Not held.
- Pooled D_S ≤ D_G: 0.000049 > 0.000034. Not held.
- L1, L2, L3: supervisor ≥ generations on all three. Held.
- K1: 0.010719 < 0.019780. Not held. K2: -0.010709 ≥ -0.020261. Held. K3: 0.511436 ≥ 0.496561. Held.
- Neither ADOPT nor OPTION holds, so the verdict is REJECT, as computed by
  `supervisor_verdict`.
- Weight copy is recorded only: it shows the highest F on every seed and the
  highest L1-L3, but it never enters the verdict and is not adopted.

## Consequence

REJECT: the supervisor flag stays off and generation mode remains the default.
No default, config or code changes are made here.

## What this does not show

- Only four seeds, one population size (10 agents) and one budget
  (32,000,000 agent-ticks per run) were tested; other settings may differ.
- Supervisor settings were left at defaults; no tuning was attempted, so this
  does not show that no supervisor configuration could match generation mode.
- The weight-copy result is descriptive only; it is not tested against the
  rule and its advantage is not established as robust.
- The metrics measure food, deaths and the defined turn and meal-share
  measures; they do not establish broader cognitive differences.
