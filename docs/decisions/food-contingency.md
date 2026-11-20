Verdict: FAIL readout

# Food-contingency gate decision

## Question

With two food kinds, one nourishing and one harmful and fixed per seed, does
the learning brain come to approach the nourishing kind and avoid the harmful
one more than a frozen-policy control? If it does not, which link fails first:
the kind readout, the credit signal, or the weight change?

## Setup

- Build: `cargo build --release -p xagent-sandbox` (release profile).
- Adapter check: `cargo run --release -p xagent-sandbox -- --bench --bench-ticks 2000 --bench-agents 2`.
  GPU adapter: `llvmpipe (LLVM 21.1.8, 256 bits)` (Mesa lavapipe, Vulkan).
- Learning arm, seed N in 5, 6, 7, 8, in that order:
  `cargo run --release -p xagent-sandbox -- --free-life --free-life-agents 10 --free-life-ticks 300000 --seed N --food-kinds --food-kind-summary <job>/summaries/learning-seedN.json`
- Frozen arm, same seeds: the same command plus `--freeze-policy`, summary
  `<job>/summaries/frozen-seedN.json`.
- Fresh birth brains: no `--free-life-champion-db` was given, so every agent
  starts from a fresh birth brain.
- Verdict: `cargo run --release -p xagent-sandbox --bin food_kind_verdict -- <job>/summaries --write-bundle docs/decisions/food-contingency-results.json`.
- Runs were sequential, one at a time. No run failed and none was retried.
  The eight summaries were produced once and the verdict was recomputed from the
  same summaries to write the bundle at this path.

| Run | Wall time |
|---|---|
| learning seed 5 | 102 s |
| learning seed 6 | 92 s |
| learning seed 7 | 92 s |
| learning seed 8 | 104 s |
| frozen seed 5 | 104 s |
| frozen seed 6 | 104 s |
| frozen seed 7 | 97 s |
| frozen seed 8 | 102 s |

## Nourishing kind per seed

| Seed | Nourishing kind |
|---|---|
| 5 | First |
| 6 | Second |
| 7 | Second |
| 8 | First |

## Behaviour

B1 is the late turn toward the nourishing kind, B2 is the late turn away from the
harmful kind, and B3 is the late (q3) nourishing meal share against q0 and the
frozen arm. Meal and death counts are totals over the ten agents.

| Seed | Arm | B1 | B2 | q0 share | q3 share | Nourishing meals | Harmful meals | Deaths |
|---|---|---|---|---|---|---|---|---|
| 5 | learning | -0.0007 | -0.0055 | 0.5181 | 0.6449 | 429 | 322 | 527 |
| 5 | frozen | -0.0004 | 0.0002 | | 0.4754 | 451 | 462 | 668 |
| 6 | learning | 0.0020 | 0.0039 | 0.4880 | 0.5983 | 433 | 427 | 562 |
| 6 | frozen | 0.0007 | -0.0011 | | 0.4444 | 384 | 405 | 569 |
| 7 | learning | 0.0100 | -0.0065 | 0.5095 | 0.4949 | 346 | 335 | 545 |
| 7 | frozen | -0.0007 | 0.0005 | | 0.4698 | 351 | 345 | 596 |
| 8 | learning | 0.0042 | -0.0013 | 0.4741 | 0.5525 | 549 | 552 | 571 |
| 8 | frozen | 0.0017 | -0.0003 | | 0.4659 | 462 | 456 | 658 |

Pooled late turn toward the nourishing kind: 0.0039; pooled late turn away from
the harmful kind: -0.0023. B1 true, B2 false, B3 true.

## Links

| Seed | Readout balanced accuracy | Shuffled control |
|---|---|---|
| 5 | 0.6880 | 0.5002 |
| 6 | 0.7029 | 0.5000 |
| 7 | 0.6865 | 0.5006 |
| 8 | 0.7144 | 0.5002 |

| Pair | Agents with positive signed credit (need 24) | Agents with positive signed weight difference (need 24) |
|---|---|---|
| first | 22 | 24 |
| second | 23 | 24 |

The verdict binary reports these counts per pair, pooled over seeds.

## Outcome

- B1 passed, B3 passed, B2 failed (pooled late turn away from the harmful kind
  is -0.0023).
- The first failed link is the readout: seeds 5 and 7 have balanced accuracy
  0.6880 and 0.6865, below the 0.70 floor, though every seed is well above its
  shuffled control (about 0.50). Seeds 6 and 8 reach 0.7029 and 0.7144.
- Credit also misses: 22 and 23 agents against the 24 required.
- Weight change meets its threshold on both pairs (24 and 24).

## What this does not show

- It does not show that the brain cannot learn food contingency. It shows that
  this gate, on seeds 5 to 8 at 300,000 ticks with 10 agents, did not pass.
- It does not test evolution defaults; nothing here is ported to them.
- Four seeds are a small sample, and the runs used a software Vulkan adapter.
- Per-seed credit and weight counts are not reported; only the verdict's
  pooled per-pair counts are used.
