# Trunk lesion matrix: decision record

## Context

The brain trunk has four parts that can be switched off one at a time in a
free life: the encoding turn weights (`turn-weights`), the encoding forward
weights (`forward-weights`), the recalled-memory motor blend
(`memory-blend`) and the critic's learned value (`critic`). The arm `all`
switches off all four, and `control` switches off none. Each arm ran as a
free life of 10 agents for 300,000 ticks, from a fresh brain and from the
champion of a reference lineage, on seeds 5, 6, 7 and 8: 48 runs. The joined
reports are in `lesion-matrix.json`; `crates/xagent-sandbox/tests/lesion_matrix_record.rs`
checks the matrix and recomputes the verdicts below.

Decision rule. For lesion L, source X and metric m: per seed,
`d_s = mean over agents(m | L) - mean over agents(m | control)`;
`d = mean(d_s)`; `se = sd(d_s) / 2` (sample standard deviation over the four
seeds). A change is significant when `|d| > 2 * se` and `d != 0`.
Behaviour needs L when, in either source, `whole.meals` falls or
`whole.deaths` rises significantly. Within-life learning needs L when, in
fresh brains, `late.pathway_food_correlation.scent`,
`late.pathway_food_correlation.vision` or `late.turn_toward_food` falls
significantly.

## Reproduction commands

Build, base config and output directory:

```sh
cargo build --release -p xagent-sandbox
mkdir -p target/trunk-lesions/runs
cargo run --release -p xagent-sandbox -- --dump-config > target/trunk-lesions/base.json
# edit target/trunk-lesions/base.json: governor.tick_budget = 40000; vision_stride stays 1
```

Reference lineages (all four ran at once, each exited 0):

```sh
cargo run --release -p xagent-sandbox -- --config target/trunk-lesions/base.json --seed 5 --no-render --db target/trunk-lesions/reference-s5.db --generations 80
cargo run --release -p xagent-sandbox -- --config target/trunk-lesions/base.json --seed 6 --no-render --db target/trunk-lesions/reference-s6.db --generations 80
cargo run --release -p xagent-sandbox -- --config target/trunk-lesions/base.json --seed 7 --no-render --db target/trunk-lesions/reference-s7.db --generations 80
cargo run --release -p xagent-sandbox -- --config target/trunk-lesions/base.json --seed 8 --no-render --db target/trunk-lesions/reference-s8.db --generations 80
```

Free lives (48 runs, at most four at once, all exited 0). For source
`fresh` and each arm and seed S:

```sh
cargo run --release -p xagent-sandbox -- --free-life --free-life-agents 10 --free-life-ticks 300000 --seed S --free-life-lesion LESION --free-life-out target/trunk-lesions/runs/fresh-ARM-seedS.json
```

For source `champion`:

```sh
cargo run --release -p xagent-sandbox -- --free-life --free-life-agents 10 --free-life-ticks 300000 --seed S --free-life-champion-db target/trunk-lesions/reference-sS.db --free-life-lesion LESION --free-life-out target/trunk-lesions/runs/champion-ARM-seedS.json
```

ARM is one of `control`, `turn-weights`, `forward-weights`,
`memory-blend`, `critic`, `all`; LESION is `none` for `control` and
the arm name otherwise; S is 5, 6, 7 or 8.

Join:

```sh
out=docs/measurements/trunk-lesions/lesion-matrix.json
{ echo '['; first=1
  for src in fresh champion; do
    for arm in control turn-weights forward-weights memory-blend critic all; do
      for S in 5 6 7 8; do
        [ $first = 1 ] || echo ','; first=0
        cat target/trunk-lesions/runs/$src-$arm-seed$S.json
      done
    done
  done
  echo ']'; } > "$out"
```

Check: `cargo test -p xagent-sandbox --test lesion_matrix_record -- --nocapture`.

## Reference champions

| seed | champion_node_id | champion_generation |
|---|---|---|
| 5 | 78 | 77 |
| 6 | 78 | 77 |
| 7 | 75 | 74 |
| 8 | 79 | 78 |

## Verdicts

| part | behaviour | within-life learning |
|---|---|---|
| turn-weights | not needed | not needed |
| forward-weights | not needed | not needed |
| memory-blend | not needed | needed |
| critic | needed | not needed |

## Supporting numbers

Each cell is `d ± se` against `control` of the same source; `*` marks a
significant change by the rule above. The control mean is the mean over the
four seeds of the per-seed agent mean.

### fresh

| metric | control mean | turn-weights | forward-weights | memory-blend | critic | all |
|---|---|---|---|---|---|---|
| `whole.meals` | 165.3 | 56.98 ± 19.6 * | 10.8 ± 11.5 | 15.13 ± 15.4 | 28.8 ± 7.96 * | 37.8 ± 10.4 * |
| `whole.deaths` | 34.02 | -0.1 ± 4 | -1.075 ± 2.07 | -0.7 ± 2.97 | 10.22 ± 1.47 * | 13.45 ± 1.16 * |
| `whole.hazard_deaths` | 19.98 | 4.275 ± 2.98 | 0.575 ± 0.811 | 1.525 ± 1.45 | 18.38 ± 2.57 * | 22.35 ± 0.975 * |
| `whole.sighting_meal_share` | 0.5778 | 0.0492 ± 0.0239 * | 0.004973 ± 0.0178 | 0.02191 ± 0.0174 | 0.03039 ± 0.00906 * | 0.04742 ± 0.0163 * |
| `late.turn_toward_food` | 0.01459 | 0.006643 ± 0.00194 * | -0.002434 ± 0.00476 | -0.009517 ± 0.00901 | -0.001633 ± 0.00183 | 0.001388 ± 0.00128 |
| `late.turn_away_from_hazard` | 0.01268 | 0.004088 ± 0.00491 | 0.006484 ± 0.00265 * | -0.0001781 ± 0.00593 | -0.002677 ± 0.00468 | -0.003404 ± 0.004 |
| `late.pathway_food_correlation.encoding` | 0.02881 | -0.001978 ± 0.00868 | -0.02179 ± 0.00861 * | -0.01256 ± 0.0143 | -0.01027 ± 0.0172 | -0.01265 ± 0.00709 |
| `late.pathway_food_correlation.scent` | 0.336 | -0.03815 ± 0.0252 | 0.002389 ± 0.0351 | -0.07544 ± 0.0374 * | -0.00788 ± 0.0536 | -0.06793 ± 0.0114 * |
| `late.pathway_food_correlation.vision` | 0.2253 | 0.08034 ± 0.067 | 0.09424 ± 0.0304 * | 0.03673 ± 0.0619 | 0.09387 ± 0.0366 * | 0.165 ± 0.0483 * |
| `late.value_gap_in_view` | 0.07829 | 0.0381 ± 0.00427 * | 0.01233 ± 0.016 | 0.009632 ± 0.0062 | -0.07563 ± 0.00345 * | -0.07483 ± 0.00192 * |
| `late.value_gap_centred` | 0.004437 | 0.01812 ± 0.00599 * | 0.008049 ± 0.0126 | 0.009264 ± 0.00784 | -0.003841 ± 0.00748 | -0.005617 ± 0.00301 |

### champion

| metric | control mean | turn-weights | forward-weights | memory-blend | critic | all |
|---|---|---|---|---|---|---|
| `whole.meals` | 445.7 | 37.55 ± 28.6 | 6.35 ± 21.1 | 27.7 ± 15 | 40.1 ± 10.9 * | 36.95 ± 27.3 |
| `whole.deaths` | 9.675 | 0.475 ± 1.42 | -1.55 ± 0.741 * | 0.475 ± 0.728 | 0.2 ± 1.95 | -0.1 ± 2.14 |
| `whole.hazard_deaths` | 8.55 | 0.425 ± 1.1 | -1.475 ± 0.85 | 0.1 ± 0.644 | 0.65 ± 1.86 | -0.25 ± 1.65 |
| `whole.sighting_meal_share` | 0.7397 | 0.007029 ± 0.00692 | -2.353e-05 ± 0.00604 | -0.006108 ± 0.00849 | 0.01029 ± 0.00564 | 0.004182 ± 0.00957 |
| `late.turn_toward_food` | 0.065 | 0.01381 ± 0.00854 | 0.00839 ± 0.00754 | 0.007399 ± 0.00296 * | 0.01555 ± 0.00663 * | 0.01595 ± 0.00826 |
| `late.turn_away_from_hazard` | 0.04542 | -0.0003989 ± 0.00577 | 0.003651 ± 0.00479 | 0.001852 ± 0.00406 | 0.005637 ± 0.00438 | 0.004463 ± 0.00558 |
| `late.pathway_food_correlation.encoding` | -0.03571 | 0.04268 ± 0.0246 | 0.00411 ± 0.0108 | 0.01365 ± 0.0167 | 0.009612 ± 0.014 | 0.04626 ± 0.00991 * |
| `late.pathway_food_correlation.scent` | 0.3764 | -0.002432 ± 0.0198 | 0.009096 ± 0.0182 | -0.006031 ± 0.0162 | -0.009608 ± 0.0174 | -0.003032 ± 0.0167 |
| `late.pathway_food_correlation.vision` | 0.183 | -0.02986 ± 0.017 | -0.01177 ± 0.00592 | -0.01425 ± 0.0064 * | -0.02012 ± 0.00859 * | -0.0409 ± 0.0113 * |
| `late.value_gap_in_view` | 0.08037 | 0.007889 ± 0.00801 | -0.004778 ± 0.0137 | 0.007556 ± 0.00518 | -0.06934 ± 0.0231 * | -0.07382 ± 0.0201 * |
| `late.value_gap_centred` | 0.003941 | 0.006575 ± 0.00933 | 0.003361 ± 0.00654 | 0.002168 ± 0.00859 | 0.0005607 ± 0.00817 | 0.005203 ± 0.011 |


## What the trunk is for

By these verdicts the critic's learned value is the one trunk part that
behaviour needs: without it fresh brains die more often (deaths +10.2 ± 1.5,
almost all on hazard ground), even though they eat more. The memory blend is
the one part that within-life learning needs: without it the scent turn
output of fresh brains tracks food less well late in life (-0.075 ± 0.037).
The encoding turn and forward weights are needed for neither: switching
them off did not reduce meals, raise deaths, or reduce the scent, vision or
turn-toward-food learning measures; removing the turn weights instead raised
meals in fresh brains. In champions, removing the critic or the memory blend
significantly lowered the vision pathway-food correlation (-0.020 ± 0.009 and
-0.014 ± 0.006), removing the forward weights significantly lowered deaths
(-1.55 ± 0.74), and removing the critic raised meals (+40.1 ± 10.9); no
lesion produced a significant meal drop in champions. The
trunk's job, as measured here, is to keep the agent alive through a value
signal and to let recalled memory shape steering, not to steer through the
128-d encoding weights.
