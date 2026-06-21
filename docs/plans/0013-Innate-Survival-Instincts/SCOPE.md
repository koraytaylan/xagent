# Scope — Plan 0013

> Implement innate survival instinct priors (danger → avoidance, energy gain →
> approach) seeded at brain initialization, heritable and evolvable, behind a
> default-off flag, with a prove-or-kill A/B gate measuring survival,
> steering-alignment, and deaths-per-food before enabling by default.

## Why this plan

Today the brain initializes all pattern memory slots to zero, so agents learn
everything from scratch with no seed. The brief diagnoses this as too weak for
basic credit-path learning without standing external reward shaping. The seeded
instinct approach injects one-time evolved prior signatures into pattern memory
at initialization — danger context + strong negative valence + avoidance motor,
energy-gain context + strong positive valence + approach motor — and then
subjects them to normal recall, reinforcement, decay, and eviction
(`brain_passes.wgsl:1317-1341` memory-blend with valence weighting,
`buffers.rs:803-804` init_pattern_memory, `gpu_kernel.rs:527-571`
reset_agents_with_rng).

1. **Blank-slate pattern memory is too weak a starting point.**
   `init_pattern_memory()` (`buffers.rs:803-804`) fills all PATTERN_STRIDE ×
   MEMORY_CAP slots with zeros — no priors, no signal direction. Learning basic
   avoidance (danger → escape) or foraging (energy → approach) from that baseline
   competes against the exploration noise (`brain_passes.wgsl:1344-1357`
   exploration-rate formula) and requires tight credit paths through recall and
   valence reinforcement (`brain_passes.wgsl:1320-1342` memory-blend weighting),
   making agents fragile to early deaths before learning fires. The
   `2026-06-19-claude-opus-48.md` review localizes the project's actual learning
   bottleneck right here (H1/H2): mirrored-steering sits at **chance (0.474)** after
   120 training episodes even though food is **~55× linearly separable** in the
   encoded state — the encoder is *not* the limit; the **turn channel cannot
   accumulate sign-correct credit** (an action-blind TD error multiplying a
   forward-dominated gradient, so `E[td_error·noise_turn·enc] ≈ 0`). Seeding the
   turn/avoidance and approach behavior as an innate motor prior is a direct way to
   give the agent the basics the credit path cannot yet teach it — the mammalian-
   instinct intuition, grounded in the measured failure mode.

2. **Memory blend already supports valence-weighted motor priors.** The motor
   memory structure stores three channels per pattern: forward, turn,
   outcome_valence (`buffers.rs:134-135` O_PAT_MOTOR layout). The memory-blend
   kernel code (`brain_passes.wgsl:1317-1341`) already weights recalled memories
   by cosine similarity × valence and mixes the recalled motor output into learned
   policy at MEMORY_BLEND_STRENGTH=0.4 (`common.wgsl:517`). Seeded patterns with
   pre-filled motor vectors and valence signatures require no new shader logic —
   they are indistinguishable from learned patterns once in the buffer.

3. **Instinct strength is heritable and subject to evolution.** Following the
   visual-genome precedent (Gabor wavelength, aspect ratio, DoG surround,
   orientation offset are all heritable genes in `BrainConfig` and mutated during
   breeding, see `config.rs:150-178`), instinct strength should be a heritable
   scalar gene subject to mutate/crossover. Agents born with weak instincts die;
   those born with well-tuned instincts survive and breed, evolving the instinct
   strength without standing external reward. Respawn semantics (re-seed at birth
   vs persist evolved learned patterns) is a locked decision with explicit gate.

4. **Default-off gating is mandatory for byte-identical fallback.** Plan 0012
   removed standing reward shaping; this plan must ship with
   innate_instincts_enabled=false so the default code path is identical to
   pre-instinct behavior (zero priors, full learning-from-scratch). A failed A/B
   gate keeps the flag off; a passed gate flips the default. This preserves the
   ability to disable instincts if they later prove mistuned or harmful.

5. **A/B gating is the prove-or-kill criterion.** Plans 0004 (0.5 Mbps read
   amplification), 0005 (orientation selectivity), and 0006 (fused kernel) all
   used headless A/B validation with explicit pass/fail thresholds and recorded
   decision docs when rejected. Plan 0013 must do the same: run seeded-instinct
   agents (flag on, inherited instinct genes) vs blank-slate baseline (flag off)
   on identical seed-deterministic worlds using the existing headless validation
   framework (`headless.rs:508-580` run_headless_with_flags infrastructure),
   measure survival (ticks-alive mean ≥ baseline + ∆), steering-alignment (mean
   avoidance-intent ≥ threshold), and deaths-per-food (mean food count / mean
   death count ≥ threshold), and record the result with explicit land-or-revert
   decision logic.

## In scope

- **0001 — Innate-Pattern-Seeding.** Design and implement two seeded instinct
  pattern signatures (danger context with negative valence + avoidance motor
  prior, energy-gain context with positive valence + approach motor prior) and
  populate them into the pattern memory buffer at brain initialization via a new
  seed_instinct_patterns() helper, subject to heritable instinct-strength genes
  (new fields in BrainConfig). Keyed to its workstream id; see [TASKS.md](TASKS.md).
- **0002 — Heritable-Instinct-Config.** Thread the new instinct-strength genes
  (instinct_danger_strength, instinct_food_strength) through the BrainConfig
  struct, the breeding/mutation machinery, and the respawn semantics, ensuring
  they are heritable like visual-genome genes and subject to mutate/crossover
  during evolution. Keyed to its workstream id; see [TASKS.md](TASKS.md).
- **0003 — Default-Off-Gating.** Gate all instinct seeding behind
  innate_instincts_enabled flag (default false, byte-identical when off). Add the
  flag to BrainConfig, thread it through init and reset paths, and ensure the gate
  is the sole control point for seeding. Keyed to its workstream id; see
  [TASKS.md](TASKS.md).
- **0004 — Prove-Or-Kill-Gate.** Implement a headless A/B benchmark
  (seeded-instinct agents vs blank-slate baseline on identical seed-deterministic
  worlds) measuring population mean survival (ticks-alive), steering-alignment
  (mirrored-steering intent fraction vs danger bearing), and deaths-per-food (mean
  food count / mean death count). Define explicit pass thresholds. Author a
  decision doc recording the result and either landing the flag-default-true
  follow-up or recording rejection. Keyed to its workstream id; see
  [TASKS.md](TASKS.md).

## Origin -> workstream mapping

| Finding | Addressed by |
|---|---|
| Blank-slate pattern memory is too weak a starting point (1) | `0001` |
| Memory blend already supports valence-weighted motor priors (2) | `0001` |
| Instinct strength is heritable and subject to evolution (3) | `0002` |
| Default-off gating is mandatory for byte-identical fallback (4) | `0003` |
| A/B gating is the prove-or-kill criterion (5) | `0004` |

## Locked decisions

- **Respawn semantics: re-seed instinct patterns at every reset_agents() call.**
  Instinct patterns are re-seeded from the current heritable instinct-strength
  genes on every reset_agents() call (e.g. between generations), so evolved
  instinct strengths are inherited but learned reinforcement (pattern valence
  accumulated during one generation) does not persist across respawn. Rationale:
  seeded patterns are "one-time evolved priors", not learned memories; re-seeding
  ensures the instinct structure remains stable across generations and does not
  conflate inherited strength with learned associations. Gate condition: explicit
  measurement in the A/B gate (workstream 0004) confirms this does not harm
  learning speed or late-generation convergence (e.g. no regression in
  final-generation fitness). If later data shows learning suppression due to
  re-seeding, a follow-up plan may explore persistence semantics.
- **Instinct signature design is seeded offline and locked until A/B gate.** The
  danger-instinct encoded signature (all -0.5), motor vector (backward -0.7, turn
  +0.5), and food-instinct signature (all +0.5), motor vector (forward +0.7, no
  turn) are fixed constants in seed_instinct_patterns(). They are not heritable or
  mutated — only the strength multipliers (instinct_danger_strength,
  instinct_food_strength) evolve. Rationale: the instinct structure is a
  biological prior (analogous to mammalian innate reflexes); the strength is the
  evolvable parameter. Gate condition: the prove-or-kill benchmark (workstream
  0004) validates that these signatures improve survival/alignment; if they fail,
  alternative signatures are a revisit condition handled in a new plan iteration,
  not inline changes.
- **Default flag state is off until prove-or-kill gate passes.**
  innate_instincts_enabled defaults to false in BrainConfig, so the code is
  byte-identical to pre-0013 behavior on default init. The flag flips to true only
  after the A/B gate passes and a follow-up plan (out of scope) is authored.
  Rationale: default-off ensures no surprise behavior change on untested configs;
  it is the safest state for a new feature. Gate condition: the prove-or-kill gate
  (workstream 0004) must pass all three conditions (survival, alignment,
  food-per-death) before any follow-up plan considers flipping the default. If the
  gate fails, the flag stays false and is revisited only under explicit new
  evidence.

## Out of scope

- **Default-true follow-up plan.** Once this plan's A/B gate passes, a separate
  follow-up plan will flip innate_instincts_enabled=true by default and measure
  long-term learning impact (late-generation fitness, learning-rate convergence,
  genetic diversity). That plan is out of scope here; this plan gates on passing
  the prove-or-kill benchmark.
- **Alternative instinct signatures (different motor vectors, different encoded
  signatures).** The danger and food instinct signatures are locked (see Locked
  decisions). Offline prototyping can explore alternatives; if they show better
  results, they are a separate plan iteration, not inline changes to this plan.
- **Instinct strength curriculum (e.g. annealing strength over generations).**
  This plan seeds constant heritable strengths subject only to normal mutation.
  Annealing (e.g. starting strong, decaying over time) is a separate mechanism and
  a separate plan if the baseline seeding alone does not achieve desired outcomes.
- **Interaction with Plan 0012 reward shaping removal.** Plan 0012 removed
  standing external reward shaping. This plan seeds internal priors (pattern
  memory) instead. The two are orthogonal; no re-measurement of 0012's gates is
  required.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concrete edits.
See [TASKS.md](TASKS.md) for the executable task list with "Done when" criteria.
