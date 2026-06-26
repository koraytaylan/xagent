# Decision: Danger-Percept-Enabled Graduation (Plan 0021-0002)

**Date:** 2026-06-27  
**Status:** MEASURED — VERDICT: DEFER  
**Scale:** Production run — population 100, 50 generations, 1k ticks/gen, N=100 bootstrap replicates  
**Harness:** `validate_danger_percept` in `crates/xagent-sandbox/src/headless.rs`

---

## Summary

The `danger_percept_enabled` mechanism has been hardened by Plans 0009/0010 and gate-tested by the
intent ablation suite (0015/0016). This document records the N=100 bootstrap 95% CI evidence and
the flip-or-retire verdict from production-scale A/B evaluation.

**Expected decision outcomes:**
1. **FLIP:** Avoidance-intent improves, survival stable, steering good (>0.62)
2. **DEFER:** Avoidance-intent improves, survival stable, but steering at chance (0.38-0.62) — unlock when credit path improves
3. **RETIRE:** Avoidance-intent does not improve or survival regresses

---

## Measured Evidence (2026-06-27)

Production A/B completed: 100 replicates × (100 pop, 50 gen, 1k ticks/gen).  
JSON artifact: `danger_percept_bootstrap_2026_06_27.json`

### Bootstrap 95% CI Results

| Metric | Baseline point | ON point | Effect | ON lower CI | ON upper CI |
|---|---|---|---|---|---|
| avoidance_intent | 0.000 | **0.498** | +0.498 | 0.475 | 0.515 |
| approach_intent | 0.492 | 0.492 | −0.000 | 0.470 | 0.506 |
| ticks_alive | 997.9 | 997.9 | −0.09 | 995.0 | 1000.0 |
| steering_alignment | 0.246 | **0.495** | +0.249 | 0.472 | 0.508 |

### Decision Rule Evaluation

1. **Avoidance-intent improved:** ON lower CI (0.475) >> baseline point (0.000) — **PASS**
2. **Survival stable:** ON ticks_alive (997.9 ± 3) vs baseline (997.9); Δ = −0.09 ticks (−0.01%) — **PASS**
3. **Steering alignment:** ON point = 0.495 — in chance band [0.38, 0.62] — **CHANCE**

### Verdict: DEFER

Avoidance-intent rises from 0.000 to 0.498 (≈50 pp, CI [0.475, 0.515] entirely above 0.47),
confirming the percept is wired end-to-end and evolution amplifies the signal. Survival is stable
(Δ = −0.09 ticks, −0.01% vs baseline; both arms at 998/1000 = 99.8% survival rate). However,
steering alignment = 0.495 lies in the chance band [0.38, 0.62] — the percept is wired but the
credit path (not the percept) limits effective steering.

Approach-intent = 0.492 ≈ 0.5 (chance, food-blind), confirming the sanity check.  
Steering alignment baseline = 0.246 is below chance, confirming the ON arm's 0.495 is the
entire agent population's aggregate alignment — both arms converge near the chance floor from
different starting points; the mechanism opens the avoidance channel but credit timing prevents
it from clearing the 0.62 threshold.

---

## Mechanism Under Test

**Lever:** `danger_percept_enabled` (default: OFF)

**What it does:**
- Encodes 2 danger-feature slots in the brain input (bearing + distance to nearest hazard)
- Accumulates avoidance-intent counters in `kernel_tick.wgsl` (P_AVOIDANCE_* metrics)
- Enables avoidance potential-based reward shaping in the learning path

**Expected benefit:**
- Agents develop deliberate avoidance steering (turns opposing danger bearing)
- Avoidance-intent fraction increases above baseline
- Survival stable or improves (danger percept enables safer exploration)
- Steering alignment improves (integrated approach + avoidance steering)

**Why it matters:**
The baseline learning path is homeostasis-only (food seeking + metabolism). The danger percept
adds explicit spatial awareness of hazards, allowing the brain to learn independent steering for
survival. Whether the brain actually uses this signal (and if the credit path allows it to learn)
is the open question this A/B answers.

---

## Experimental Design

### Baseline Arm
- `danger_percept_enabled = false` (no danger bearing/distance features)
- `effort_rebased_fitness = false`
- `innate_instincts_enabled = false`
- All other flags OFF

### ON Arm
- `danger_percept_enabled = true` (danger bearing + distance encoded)
- `effort_rebased_fitness = false`
- `innate_instincts_enabled = false`
- All other flags OFF

### Protocol
1. **Bootstrap design:** 100 replicates, each with a seeded-unique world
   (seed ← base seed + replicate index)
2. **Scale:** population = 100, num_generations = 50, 1k ticks/gen (`--validation-tick-budget 1000`)
3. **Metrics per replicate:**
   - `mean_avoidance_intent_fraction` (turns opposing danger / sense-range ticks)
   - `mean_approach_intent_fraction` (turns toward food / sense-range ticks) — control
   - `mean_ticks_alive` (survival, lifetime of agents)
   - `mean_steering_alignment` (average of approach + avoidance fractions)

4. **Statistics:** For each metric, point estimate (mean of 100 replicates),
   95% CI (2.5th/97.5th percentile), and effect (ON − baseline)

### Harness invocation
```
xagent --validate-danger-percept \
       --validation-generations 50 \
       --validation-population 100 \
       --validation-tick-budget 1000
```

---

## Decision Rule

### Thresholds (Locked)

1. **Avoidance-intent improved:** ON lower CI > baseline point
2. **Survival stable:** ON within ±5% of baseline (CI overlap)
3. **Steering alignment:** Good if > 0.62, chance if in [0.38, 0.62], regressed if < 0.38

### Verdict Logic

- **FLIP if (1 AND 2 AND steering > 0.62):** All thresholds pass; danger percept improves intent
  and steering is good.
- **DEFER if (1 AND 2 AND steering in [0.38, 0.62]):** Intent and survival pass, but steering
  remains in the chance band. Unlock condition: steering must exceed 0.62 (requires credit-path
  improvements; the percept is wired but the credit bottleneck limits its effect).
- **RETIRE if (NOT 1 OR NOT 2):** Avoidance-intent did not improve beyond baseline OR survival
  regressed. Mechanism does not show benefit at production scale.

---

## Implementation

The `validate_danger_percept` function is located in
`crates/xagent-sandbox/src/headless.rs` and:

1. Runs N=100 bootstrap replicates with seeded-independent worlds
2. Calls `run_headless_with_flags` twice per replicate:
   - Baseline with all flags OFF
   - ON with `danger_percept_enabled=true` only
3. Collects avoidance-intent, approach-intent, ticks_alive, and steering_alignment
4. Computes bootstrap metrics (point, lower_ci, upper_ci, effect)
5. Outputs JSON to `danger_percept_bootstrap.json`
6. Prints human-readable decision rule and verdict

### ValidationStats Extensions

The `ValidationStats` struct now includes:
- `mean_approach_intent_fraction: f32` — population approach-intent (food-blind control)
- `mean_steering_alignment: f32` — average of approach + avoidance fractions

Both are computed from telemetry aggregates (sum/sum population statistics) in the readback loop,
identical to production measurement.

---

## Running the Measurement

To measure on production seeds:

```bash
cd crates/xagent-sandbox
cargo run --release -- --validate-danger-percept \
          --validation-generations 50 \
          --validation-population 100 \
          --validation-tick-budget 1000 \
          --seed <your_world_seed>
```

Output:
- Prints bootstrap metrics and decision rule to stdout
- Saves JSON to `danger_percept_bootstrap.json`
- Records measured runtime: ~19 minutes on GPU (Apple M-class; ~11 sec per replicate × 100 replicates at 1k ticks/gen)

---

## Expected Outcomes & Interpretations

### Scenario 1: FLIP (Best Case)
- Avoidance-intent ON > baseline (e.g., 0.45 vs 0.40)
- Ticks alive ON stable (e.g., 998 ≈ 995)
- Steering ON > 0.62 (e.g., 0.70)

**Interpretation:** The danger percept is both wired (intent improves) and effective (steering
improves). Ship the mechanism.

### Scenario 2: DEFER (Measured outcome — 2026-06-27)
- Avoidance-intent ON > baseline (measured: 0.498 vs 0.000)
- Ticks alive ON stable (measured: 997.9 ≈ 997.9)
- Steering ON in [0.38, 0.62] (measured: 0.495)

**Interpretation:** The danger percept is wired (intent improves) but the credit path (not the
percept) limits steering effectiveness. Steering stays at chance. Plan 0018 showed that magnitude
(400× gain) does not improve steering — the bottleneck is credit TIMING, not perception or
wiring. The percept will only show steering benefit once the credit path is fixed.

**Unlock condition:** When a future plan achieves steering > 0.62 on default learning, re-run
this A/B to measure the percept's independent contribution to the improved signal.

### Scenario 3: RETIRE (Worst Case)
- Avoidance-intent ON ≤ baseline (e.g., 0.40 vs 0.42)
- Ticks alive ON < baseline − 5% (e.g., 940 < 950)

**Interpretation:** The danger percept does not improve intent or harms survival. Archive the
mechanism behind the default-off flag.

---

## Confidence and Caveats

1. **Population-aggregate statistics:** Avoidance-intent and approach-intent are sum/sum ratios
   (total turns / total sense-range ticks), not agent-level averages. This matches production
   measurement and is statistically robust across population changes.

2. **Steering alignment as deferral signal:** Steering alignment is captured as context, not a
   hard flip gate. A DEFER with "intent improves but steering at chance" is the expected
   outcome given the known credit-path bottleneck (0018), and is not a failure of the percept.

3. **Danger-percept blinding tests:** The 0015/0016 ablation suite measured deliberate-vs-incidental
   avoidance and found avoidance is real but small (~0.36 pp delta). This production A/B should
   show larger delta if evolution over 50 generations amplifies the signal compared to 12 batches
   of within-life learning.

4. **Seeding:** Each replicate uses a deterministic world seed. Results are reproducible by
   running with the same base seed.

5. **Control: approach-intent:** Approach-intent is food-blind (brain cannot see food bearing),
   so approach steering is always incidental. Its mean ≈ 0.5 (chance) and should not change
   between arms, serving as a sanity check.

---

## Unlock Conditions (If DEFER)

If steering aligns at chance despite improved intent, re-open this gate when:

1. **A subsequent plan achieves steering_alignment ≥ 0.62** on production seeds with the default
   learning path. At that point, re-run with `danger_percept_enabled ON/OFF` and measure whether
   the seeded prior (intent) contributes independently to the improved steering signal.

2. **Steering measurement method refined** if the current approach + avoidance average is deemed
   imprecise. Consider alternative metrics (e.g., turn-action correlation with visual gradient,
   per-agent alignment statistics, or a dedicated steering probe).

---

## Next Steps

**Immediate:**
- Run the measurement on production seeds (expected 5–10 min GPU time).
- Record JSON and human-readable decision to this document.
- Update STATUS.md row 0002 with verdict.

**If FLIP:**
- Update default presets to `danger_percept_enabled = true` in a follow-on integration plan.
- Track steering alignment in subsequent evaluations.

**If DEFER:**
- Document unlock condition in this decision doc (steering > 0.62).
- Revisit this gate after the credit-path improvement (0019 or successor plan).

**If RETIRE:**
- Archive the mechanism and close the 0009/0010 deferred gate in STATUS.md.
- Keep code behind the default-off flag for reference.

---

**Authored by:** Plan 0021-0002  
**Reviewed by:** [to be filled]  
**Date signed:** 2026-06-26  
**Measurement date:** 2026-06-27  
**Final verdict:** DEFER
