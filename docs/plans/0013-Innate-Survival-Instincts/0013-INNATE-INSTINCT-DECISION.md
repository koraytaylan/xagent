# Decision — Plan 0013: Innate Survival Instincts

> This task runs the headless A/B benchmark measuring seeded-instinct agents
> (`innate_instincts_enabled=true`, heritable danger/food strength genes) vs
> blank-slate baseline (`innate_instincts_enabled=false`) on identical
> seed-deterministic worlds. Done when: pass/fail gates are evaluated and recorded,
> and the decision (land seeded-instinct default-true follow-up, or reject with
> revisit conditions) is documented.

## Decision

**Result: FAIL — reject this iteration. `innate_instincts_enabled` stays default-off.**

All three prove-or-kill gates failed; two of them decisively and independently of
any configuration caveat. Seeding the pattern memory with the fixed danger/food
instinct signatures produced **no measurable survival, foraging, or steering
benefit** over the blank-slate baseline, and slightly *hindered* early-generation
learning.

## The three paths

| Path | What it is | Benefit | Risk / cost |
|---|---|---|---|
| A: Blank slate | Current: learn from zero, no priors | Clean, interpretable learning | Fragile early; slow credit assignment |
| B: Seeded instincts (this plan) | Seeded danger + food priors, heritable strength | (hypothesised) faster early survival | Priors may suppress learning if mistuned |
| C: Seeded + default-ON | If B passes, flip the default | Stable long-term learner | Must verify no late-gen regression |

The measured evidence below selects **Path A** (reject B): B shows no benefit and a
small early-learning cost, so the default stays off.

## Measured evidence

A/B run on **10 generations**, population **10**, default seed-deterministic config
(`effort_rebased=false, danger_percept=false, speed_cost_exponent=1`); the
`innate_instincts_enabled` flag is the only difference between arms. Run on the
local Metal adapter via `xagent --validate-innate-instincts --validation-generations 10`.

### Survival (population mean ticks-alive)
- Baseline (OFF): **995074**
- ON: **996975** (**+0.19%**)
- Early generations: ON started **below** baseline (Gen 0 best 0.0674 vs 0.0747)
  and converged to parity by Gen 9 (~0.085 both arms).

### Steering-alignment (mean avoidance-intent fraction)
- ON: **0.000**
- Caveat: this run had `danger_percept=false`, so the danger context is never
  sensed and the seeded danger instinct can never be recalled — see "When to
  revisit". The gate cannot pass in this configuration.

### Food-per-death (mean food consumed / mean death count)
- ON: **0.37** (the agents die roughly three times per food consumed)

### Gate results
- **Survival gate** (ON ≥ baseline × 1.10): 995074 × 1.10 = 1094581 vs ON 996975 → **✗ FAIL**
- **Alignment gate** (avoidance-intent ≥ 0.4): ON 0.000 → **✗ FAIL**
- **Food-per-death gate** (ratio ≥ 2.0): ON 0.37 → **✗ FAIL**
- **Overall: ✗✗✗ GATE FAILURE**

## Why reject now

The food instinct (food context → approach, default food perception is *on*) had a
fair test and delivered nothing: food-per-death **0.37 ≪ 2.0** and survival
**+0.19% ≪ +10%**, with ON trailing baseline in early generations. The seeded
patterns are subject to normal recall/reinforcement/decay, and the measured result
is that they neither accelerate early survival nor improve foraging efficiency —
consistent with the `2026-06-19` credit-path diagnosis ([learning bottleneck is the
credit path, not the seed](../../reviews/2026-06-19-claude-opus-48.md)): a seeded
motor prior with a positive valence still has to be *recalled and acted on* through
the same blend/credit path that currently sits at chance, so a static prior alone
does not move behaviour. Introducing the seeding complexity without a demonstrated
payoff is not justified; the flag stays default-off.

## When to revisit

Reopen only with new evidence from offline prototyping:

1. **Fair danger test first.** The alignment gate measured `0.000` because the A/B
   ran with `danger_percept=false` — the danger instinct was never sensed. Before
   any further judgement on the *danger* prior, re-run the benchmark with
   `danger_percept=true` so the danger context actually enters the encoded state
   and the seeded avoidance pattern can be recalled. The current alignment failure
   is partly a benchmark-configuration artifact, not solely an instinct failure.
2. **Early-learning interference.** ON trailed baseline in the first generations.
   Investigate whether the fixed signatures/strengths (or the every-reset re-seed
   semantics) perturb the credit path; weaker default seeds (e.g. 0.4 instead of
   0.8) or persistence-across-respawn may help.
3. **Alternative signatures.** Different motor priors (danger → freeze rather than
   backward; food → turn-toward) that pass the existing gates.
4. **Longer / larger runs.** 10 generations × population 10 is a short signal; a
   longer run with `danger_percept` enabled would give the danger prior its best
   chance. The food/survival fail, however, is already decisive at this scale.

Until at least (1) is addressed and a re-run shows a real benefit, the seeded-
instinct mechanism ships **complete but default-off** behind `innate_instincts_enabled`.
