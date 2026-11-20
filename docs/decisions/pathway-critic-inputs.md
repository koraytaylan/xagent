# Critic input decision

## Question

Should the critic (value head) read the whitened visual pathway inputs, in
place of or alongside the 128-wide encoding it reads today? The candidate
input sets are `pathway` (10 inputs), `encoding` (128, the current critic
input) and `encoding_and_pathway` (138). This record fits each set offline to
the returns of logged free lives and picks a winner with the rule below.

## Setup

- Commit: `e2dc157b8dcb14846d168eb75545a2cfa92bc713`, the output of
  `git rev-parse HEAD` when the four free-life runs and the fit were executed.
  The runs used this commit. The later commit
  `13591ac520647ca31cb50242c724959eca4ea92e` adds this record plus a
  one-line status change in a planning document, and changes no source, config or route flag.
  Both routes off (`critic_input` and `forward_pathway_enabled` at their
  defaults), `vision_stride` 1 (default).
- Config: default config from `--dump-config`, used unchanged for every run.
  SHA-256 of the dump: `78c12cf9477828b8a5f357f140ac0ae4a23adaebe359fe631765aa7940f00b4a`.
- Brains: fresh birth brains (no stored champion).
- Seeds 5, 6, 7, 8; 10 agents per run (40 agents); 300,000 ticks per run
  (30,000 brain ticks per agent). All four runs exited 0 and each report shows
  `"ticks": 300000`, `"brain_ticks": 30000`, `"brain_source": "fresh"`.

Commands (run from the repository root; scratch paths outside the repository):

```sh
cargo run --release -p xagent-sandbox -- --dump-config > config.json
sha256sum config.json

# once per seed S in 5 6 7 8
xagent --config config.json --free-life --seed S \
  --free-life-agents 10 --free-life-ticks 300000 \
  --free-life-log seedS.log --free-life-out seedS.json

cargo run -p xagent-sandbox --bin pathway_return_fit -- fit \
  --log seed5.log seed6.log seed7.log seed8.log \
  --out docs/decisions/pathway-critic-inputs.json

cargo run -p xagent-sandbox --bin pathway_return_fit -- verify \
  docs/decisions/pathway-critic-inputs.json
```

`xagent` is the release binary built by `cargo build --release -p xagent-sandbox`.
The free-life runs were started four at a time (host memory available about
84 GB before the runs).

## Method

- Target: per agent life, return `G_t = Σ_k γ^k r_{t+k+1}` with
  `γ = TD_DISCOUNT = 0.97`, cut at each death with the terminal outcome
  `TERMINAL_DEATH_TD_ERROR = −1` as the last reward. The final 300 brain ticks
  of each log are dropped (truncated returns).
- Alignment: the critic reads the whitened pathway inputs stored by the
  previous brain tick, so the tool sets the pathway columns of row `t` to the
  logged pathway inputs of row `t − 1` of the same agent; encoding columns and
  the target stay the row's own. The first row of each agent's log and the
  first row after each death are dropped from all three input sets, so every
  set fits the same rows. The tool applies this alignment itself; the logs are
  not preprocessed.
- Splits: blocks of 1000 brain ticks; a block is held out when
  `splitmix64(seed, agent, block) mod 10 < 3`. Here `seed` is the log's
  index in the `--log` list (0 to 3), which is what the JSON `agents[].seed`
  field records. Run seeds 5, 6, 7 and 8 map to indexes 0, 1, 2 and 3 in that
  order (`seed5.log` to `seed8.log`).
- Fit: ridge least squares with an intercept, λ chosen from
  `{1e-3, 1e-2, 1e-1, 1, 10}` on an inner split of the training blocks.
- Scores on held-out ticks: R²; in-view gap (mean fitted value with food in
  view minus not in view); centred gap (centred minus off-axis), using the same
  in-view and centred predicates as the free-life critic gap metrics.
- Winner rule: among sets whose pooled in-view gap and centred gap are both
  positive, the highest pooled mean held-out R²; `encoding_and_pathway` wins
  over `pathway` only if its paired per-agent R² advantage exceeds one standard
  error; if no set has both gaps positive, the winner is `encoding`.

## Results

Source: `docs/decisions/pathway-critic-inputs.json`. Pooled values, SEs and
gap positive-agent counts are copied from its `sets.<set>` fields
(`pooled_r2`, `pooled_r2_se`, `pooled_in_view_gap`, `pooled_in_view_gap_se`,
`in_view_positive_agents`, `pooled_centred_gap`, `pooled_centred_gap_se`,
`centred_positive_agents`). The JSON has no field for the number of agents
with positive R²; that count is the number of `sets.<set>.per_agent` entries
with `held_out_r2 > 0`, counted by script from the same JSON. 40 agents per set.

### `pathway`

| Metric | Pooled ± SE | Agents positive |
| --- | --- | --- |
| Held-out R² | -0.017632559112475 ± 0.01906992738991772 | 24 / 40 |
| In-view gap | 0.056289916438060236 ± 0.0032423605712854236 | 40 / 40 |
| Centred gap | -0.007177479096119705 ± 0.002198101354952649 | 11 / 40 |

### `encoding`

| Metric | Pooled ± SE | Agents positive |
| --- | --- | --- |
| Held-out R² | 0.08553721456958444 ± 0.02078014293912334 | 37 / 40 |
| In-view gap | 0.1064854532134063 ± 0.006108344743043743 | 40 / 40 |
| Centred gap | 0.017591688316976467 ± 0.0062654720515197895 | 29 / 40 |

### `encoding_and_pathway`

| Metric | Pooled ± SE | Agents positive |
| --- | --- | --- |
| Held-out R² | 0.06467065969500722 ± 0.02200631428617309 | 35 / 40 |
| In-view gap | 0.11381209050268762 ± 0.006319399722113118 | 40 / 40 |
| Centred gap | 0.022202722441585592 ± 0.006768963786463016 | 31 / 40 |

Paired per-agent R² advantage SE (`paired_r2_advantage_se`): 0.01234641142135113.

## Winner

`winner`: `encoding`.

`pathway` is not eligible: its pooled centred gap is negative. `encoding` and
`encoding_and_pathway` both have positive pooled in-view and centred gaps;
among them `encoding` has the higher pooled held-out R² (0.0855 against
0.0647), so it wins. Adding the pathway inputs to the encoding raised the
gaps slightly but lowered held-out R².

## Meaning for the critic route

The winner is `encoding`, the input the critic reads today. The critic route
stays off (`critic_input` keeps its default), and the critic-route gate record
is recorded as not applicable.

## Regenerating the logs

Run the commands under Setup. The logs (`seed5.log`, `seed6.log`,
`seed7.log`, `seed8.log`), the free-life reports and the config dump are not
committed; this record names the log files only.
