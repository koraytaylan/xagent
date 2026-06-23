//! Seeded A/B percept-ablation measurement: is avoidance-intent DELIBERATE or INCIDENTAL?
//!
//! The `intent_baseline_measurement` probe measured the baseline intent fractions (approach
//! ≈ 0.498, avoidance ≈ 0.421) and left open the question those near-chance numbers raise:
//! when an agent turns away from
//! danger, is that turn *caused by* the agent sensing the danger bearing (deliberate), or is
//! it coincidental motion that the geometry-gated counter happens to score (incidental)?
//!
//! This is the validation harness that adjudicates it, for the one axis where a true ablation
//! is possible today. Architectural fact (see the brain README): the encoder packs the
//! nearest-DANGER bearing as a feature (gated by `danger_percept_enabled`), but it packs **no
//! food bearing at all** — so approach-intent is *incidental by construction* (the brain
//! cannot see which way food is) and needs no A/B. Danger is the testable axis.
//!
//! Method — a paired, seed-identical ablation. Both arms keep `danger_percept_enabled = true`
//! (so danger detection runs and the avoidance counters keep counting in **both** arms, and
//! the encoder width / brain layout is identical), and differ only in a runtime mask.
//! A (sighted) = `danger_percept_blinded = false`: the brain sees the true danger bearing.
//! B (blinded) = `danger_percept_blinded = true`: the encoder is fed the "no danger in range"
//! sentinel (distance 1.0, bearing 0.0) while the raw physics slots — and thus the avoidance
//! counters — are untouched.
//! Because A and B share seed, world, agent init, and network, the paired delta
//! `Δ = mean(avoidance_fraction | A) − mean(avoidance_fraction | B)` isolates the *causal
//! contribution of the brain seeing the danger bearing* to its steering. An A/A pair (two
//! sighted runs, same seed) gives the run-to-run noise floor so we know whether any Δ is real.
//!
//! Decision rule (stated, not asserted — this is a measurement probe).
//! Deliberate iff `Δ ≥ +0.05` and `Δ` clears the A/A noise floor with margin.
//! Incidental iff `|Δ| < 0.05` (within noise): avoidance turns happen at ~chance whether or
//! not the brain sees the bearing, locating the signal in the homeostasis-only credit path
//! (no spatial reward term), not the percept.
//!
//! ── MEASURED RESULT (Apple M3 Max, Metal; 16 agents × 12 batches × 8 paired seeds) ──
//! Record-then-paste idiom, mirroring `intent_baseline_measurement.rs`.
//!
//!   Per-seed avoidance-intent fraction:
//!     seed    A(sighted)  B(blinded)    Δ=A−B
//!     20250    0.427304   0.422838   +0.004467
//!     20251    0.430670   0.421724   +0.008946
//!     20252    0.432566   0.429445   +0.003121
//!     20253    0.420290   0.423350   −0.003061
//!     20254    0.430809   0.424034   +0.006775
//!     20255    0.428596   0.421165   +0.007431
//!     20256    0.425524   0.421320   +0.004205
//!     20257    0.422491   0.425237   −0.002746
//!
//!   mean A (sighted): 0.427281   mean B (blinded): 0.423639
//!   mean Δ (A−B):     +0.003642   std Δ: 0.004173   95% CI: [+0.000751, +0.006534]
//!   A/A noise floor:  0.000000 (kernel is bit-deterministic — the Δ is real, not jitter)
//!   approach control (food-blind): sighted 0.509871, blinded 0.506479
//!   Verdict: NEGLIGIBLE-BUT-REAL (|Δ| < 0.05, yet CI excludes 0)
//!
//! Interpretation (precise — do not over-read the label): the A/A noise floor is exactly 0
//! while the A/B Δ is +0.0036 with a CI that excludes 0, so the mask demonstrably changes the
//! brain's output — the danger percept IS wired and consumed, the steering is NOT causally
//! inert. But seeing the danger bearing buys only +0.36 pp of extra avoidance, ~14× below the
//! 0.05 practical-significance bar: the percept's contribution to steering is real but
//! negligible. Agents turn away from danger at ~chance (≈0.42, slightly below 0.5 from the
//! food/danger co-location geometry) almost regardless of whether the brain can see it.
//! What this RULES OUT: broken wiring / an unconsumed percept. What it does NOT isolate: the
//! residual ≈49.6 pp gap could be weak temporal credit assignment, weak bearing
//! representation in the encoded state, too-short within-life discovery, or the gap between
//! within-life TD (this regime) and cross-generation selection (not tested here). This run is
//! within-life only (random init + within-life TD, no evolution); an evolved-policy ablation
//! is needed before attributing the gap to any one cause. Consistent with the independently
//! measured chance-level steering (≈0.474) in `xagent-sandbox/tests/integration.rs`.
//!
//! No threshold is asserted (measurement-only). The test asserts only that the harness is
//! valid: the avoidance counters are non-zero in BOTH arms (gotcha: if blinding silently
//! zeroed the counter, B would be 0/0 and manufacture a fake delta), and the A/A pair is
//! deterministic (noise floor near zero).

use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

const LATTICE: usize = 4; // 4×4 = 16 agents, mirroring intent_baseline_measurement.rs
const AGENT_COUNT: u32 = (LATTICE * LATTICE) as u32;
const NUM_BATCHES: usize = 12;
const SEEDS: [u64; 8] = [20250, 20251, 20252, 20253, 20254, 20255, 20256, 20257];

/// One agent's intent fractions after a run. Avoidance is the axis under test; approach is
/// reported only as context (it is incidental by construction — the brain is food-bearing-blind).
struct AgentIntent {
    approach: Option<f32>,
    avoidance: Option<f32>,
    avoidance_sense_ticks: f32,
}

/// Run a single arm: build a fresh kernel with the given mask, seed it, advance it, and read
/// every agent's intent fractions. Both arms share an identical world and the same `seed`;
/// only `blinded` differs (and, for the noise floor, nothing differs between two A runs).
fn run_arm(seed: u64, blinded: bool) -> Vec<AgentIntent> {
    let mut brain_config = BrainConfig::default();
    // Danger percept ON in both arms so detection runs and the avoidance counter is alive;
    // the ablation is the blinding mask, not the percept gate.
    brain_config.danger_percept_enabled = true;
    brain_config.danger_percept_blinded = blinded;
    let world_config = WorldConfig::default();

    let mut kernel = GpuKernel::new(
        AGENT_COUNT,
        AGENT_COUNT as usize,
        &brain_config,
        &world_config,
    );
    kernel.reset_agents_seeded(&brain_config, seed);

    // World: flat terrain; a central danger biome; one food per agent near its spawn — the
    // exact layout `intent_baseline_measurement.rs` proved exercises both intent axes.
    let terrain_vps = 129;
    let heights = vec![0.0_f32; terrain_vps * terrain_vps];

    let biome_res = 256;
    let mut biomes = vec![0_u32; biome_res * biome_res];
    for i in 110..150 {
        for j in 110..150 {
            if i < biome_res && j < biome_res {
                biomes[i * biome_res + j] = 2u32; // BIOME_DANGER
            }
        }
    }

    let mut food_pos = Vec::new();
    let mut food_consumed = Vec::new();
    let mut food_timers = Vec::new();
    let mut agent_data = Vec::new();
    for i in 0..LATTICE {
        for j in 0..LATTICE {
            let spawn_x = (i as f32) * 12.0;
            let spawn_z = (j as f32) * 12.0;
            food_pos.push((spawn_x + 10.0, 0.0, spawn_z + 10.0));
            food_consumed.push(false);
            food_timers.push(0.0);
            agent_data.push((
                glam::Vec3::new(spawn_x, 1.0, spawn_z),
                100.0,
                100.0,
                brain_config.memory_capacity,
                brain_config.processing_slots,
            ));
        }
    }
    kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);
    kernel.upload_agents(&agent_data);

    let batch_size = kernel.kernel_batch_size();
    for batch_idx in 0..NUM_BATCHES {
        let start_tick = (batch_idx as u64) * batch_size as u64;
        kernel.dispatch_batch(start_tick, batch_size);
    }

    (0..AGENT_COUNT)
        .map(|agent_id| {
            let t = kernel.read_agent_telemetry_blocking(agent_id);
            let avoidance = if t.avoidance_sense_range_ticks > 1e-6 {
                Some(t.avoidance_turns_opposing / t.avoidance_sense_range_ticks)
            } else {
                None
            };
            let approach = if t.approach_sense_range_ticks > 1e-6 {
                Some(t.approach_turns_toward / t.approach_sense_range_ticks)
            } else {
                None
            };
            AgentIntent {
                approach,
                avoidance,
                avoidance_sense_ticks: t.avoidance_sense_range_ticks,
            }
        })
        .collect()
}

/// Mean over the defined samples; returns None if no agent had the axis in range.
fn mean_defined(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let defined: Vec<f32> = values.flatten().collect();
    if defined.is_empty() {
        None
    } else {
        Some(defined.iter().sum::<f32>() / defined.len() as f32)
    }
}

#[test]
fn danger_percept_ablation_ab() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let mut deltas: Vec<f32> = Vec::new();
    let mut sighted_means: Vec<f32> = Vec::new();
    let mut blinded_means: Vec<f32> = Vec::new();
    let mut approach_sighted_means: Vec<f32> = Vec::new();
    let mut approach_blinded_means: Vec<f32> = Vec::new();

    eprintln!("Danger-percept ablation A/B — per-seed avoidance-intent fraction:");
    eprintln!("  seed        A(sighted)  B(blinded)   Δ=A−B");
    for &seed in SEEDS.iter() {
        let a = run_arm(seed, false);
        let b = run_arm(seed, true);

        // Gotcha guard: the avoidance counter MUST be alive in both arms. If blinding
        // silently zeroed it, B would be 0/0 and any "delta" would be an artifact.
        let a_ticks: f32 = a.iter().map(|x| x.avoidance_sense_ticks).sum();
        let b_ticks: f32 = b.iter().map(|x| x.avoidance_sense_ticks).sum();
        assert!(
            a_ticks > 0.0 && b_ticks > 0.0,
            "avoidance counter degenerate (A ticks={a_ticks}, B ticks={b_ticks}) — \
             blinding must not zero the geometry-gated counter"
        );

        let a_av = mean_defined(a.iter().map(|x| x.avoidance)).unwrap();
        let b_av = mean_defined(b.iter().map(|x| x.avoidance)).unwrap();
        let delta = a_av - b_av;
        deltas.push(delta);
        sighted_means.push(a_av);
        blinded_means.push(b_av);
        if let Some(m) = mean_defined(a.iter().map(|x| x.approach)) {
            approach_sighted_means.push(m);
        }
        if let Some(m) = mean_defined(b.iter().map(|x| x.approach)) {
            approach_blinded_means.push(m);
        }
        eprintln!("  {seed:<10}  {a_av:>9.6}  {b_av:>9.6}  {delta:>+9.6}");
    }

    // Noise floor: two sighted runs at the same seed. With a deterministic kernel this is ~0;
    // any A/B delta must clear it to count as a real causal effect.
    let noise = {
        let a1 = run_arm(SEEDS[0], false);
        let a2 = run_arm(SEEDS[0], false);
        let m1 = mean_defined(a1.iter().map(|x| x.avoidance)).unwrap();
        let m2 = mean_defined(a2.iter().map(|x| x.avoidance)).unwrap();
        (m1 - m2).abs()
    };

    let n = deltas.len() as f32;
    let mean_delta = deltas.iter().sum::<f32>() / n;
    let var_delta = deltas
        .iter()
        .map(|&d| (d - mean_delta).powi(2))
        .sum::<f32>()
        / n;
    let std_delta = var_delta.sqrt();
    let sem = std_delta / n.sqrt();
    let ci95 = 1.96 * sem; // normal approximation; report mean_delta ± ci95
    let mean_sighted = sighted_means.iter().sum::<f32>() / n;
    let mean_blinded = blinded_means.iter().sum::<f32>() / n;
    let mean_approach_sighted = if approach_sighted_means.is_empty() {
        f32::NAN
    } else {
        approach_sighted_means.iter().sum::<f32>() / approach_sighted_means.len() as f32
    };
    let mean_approach_blinded = if approach_blinded_means.is_empty() {
        f32::NAN
    } else {
        approach_blinded_means.iter().sum::<f32>() / approach_blinded_means.len() as f32
    };

    eprintln!(
        "\nAvoidance-intent ablation summary ({} paired seeds):",
        SEEDS.len()
    );
    eprintln!("  mean A (sighted): {mean_sighted:.6}");
    eprintln!("  mean B (blinded): {mean_blinded:.6}");
    eprintln!("  mean Δ (A−B):     {mean_delta:+.6}");
    eprintln!("  std Δ:            {std_delta:.6}");
    eprintln!(
        "  95% CI of Δ:      [{:+.6}, {:+.6}]",
        mean_delta - ci95,
        mean_delta + ci95
    );
    eprintln!("  A/A noise floor:  {noise:.6}");
    eprintln!(
        "  approach (food-blind control): sighted {mean_approach_sighted:.6}, blinded {mean_approach_blinded:.6}"
    );
    let consumed = (mean_delta - ci95) > noise; // CI lower bound clears the noise floor
    let verdict = if mean_delta.abs() < 0.05 {
        if consumed {
            "NEGLIGIBLE-BUT-REAL (|Δ| < 0.05, CI excludes the noise floor: the percept is \
             consumed but contributes ~chance-level steering)"
        } else {
            "INERT (|Δ| < 0.05 and within the noise floor: no detectable percept effect)"
        }
    } else if mean_delta >= 0.05 && consumed {
        "DELIBERATE (Δ ≥ 0.05 and CI lower bound clears the noise floor)"
    } else {
        "INDETERMINATE (|Δ| ≥ 0.05 but CI/noise inconclusive)"
    };
    eprintln!("  Verdict: {verdict}");

    // Measurement-only: no threshold assertion. Sanity only — the A/A pair must be ~deterministic
    // so the A/B delta is interpretable, and the counters were already asserted alive per seed.
    assert!(
        noise < 0.02,
        "A/A noise floor {noise:.6} too high — paired delta would be uninterpretable"
    );
}
