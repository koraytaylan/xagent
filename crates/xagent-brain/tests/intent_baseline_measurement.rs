//! Measurement probe for baseline intent distribution at default config.
//!
//! This test captures the baseline distribution of approach-intent and avoidance-intent
//! fractions (mean, std, min, max, and p25/p50/p75 percentiles) over 16 agents under pure
//! homeostatic learning (post-Plan-0012, no PBRS shaping), serving as the reference
//! baseline a follow-up validation-harness plan will use to set deliberate-vs-incidental
//! classification thresholds.
//!
//! The world is constructed to exercise both intent axes: 16 agents on a 4×4 lattice near
//! the origin, each with food in sense range and danger biome enabled (danger_percept_enabled = true).
//! Each measurement point dispatches one full kernel batch so the brain pass executes.
//!
//! No assertions — this is a measurement probe, not a gate.

use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Baseline intent fractions under pure homeostatic learning (post-Plan-0012, no PBRS shaping).
/// Records mean/std/min/max + p25/p50/p75 per axis as the reference distribution
/// a follow-up validation-harness plan will use to set deliberate-vs-incidental thresholds.
///
/// World setup: 16 agents on a 4×4 lattice (~12 units apart), each agent spawned near origin
/// with food within FOOD_SENSE_RADIUS (30.0) and danger biome within DANGER_SENSE_RADIUS.
/// Baseline runs with danger_percept_enabled = true (off at shipped default) so avoidance axis
/// is exercised; approach axis is always measurable (food detection always runs).
///
/// Approach-intent fractions (toward food):
///   mean: 0.498238, std: 0.051976, min: 0.400000, max: 0.606061
///   p25: 0.475000, p50: 0.512821, p75: 0.537500
///
/// Avoidance-intent fractions (away from danger):
///   mean: 0.421316, std: 0.099750, min: 0.200000, max: 0.613636
///   p25: 0.366667, p50: 0.434783, p75: 0.487500
#[test]
fn intent_baseline_measurement() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let mut brain_config = BrainConfig::default();
    // Enable danger percept so avoidance axis is exercised; it is off at shipped default.
    brain_config.danger_percept_enabled = true;
    let world_config = WorldConfig::default();

    let agent_count: u32 = 16;
    let mut kernel = GpuKernel::new(
        agent_count,
        agent_count as usize,
        &brain_config,
        &world_config,
    );
    kernel.reset_agents_seeded(&brain_config, 12345);

    // World setup: flat terrain, danger biome, one food per agent near origin on 4×4 lattice.
    let terrain_vps = 129;
    let heights = vec![0.0_f32; terrain_vps * terrain_vps];

    // Biome grid: mark danger cells within DANGER_SENSE_RADIUS of agents on the lattice.
    // Lattice is ~12 units apart; DANGER_SENSE_RADIUS = 30.0, so a danger region covering
    // a 10×10 area around the origin (roughly cell 50-60 on a 256×256 grid) ensures all
    // agents sense danger.
    let biome_res = 256;
    let mut biomes = vec![0_u32; biome_res * biome_res];
    // Mark danger biome (BIOME_DANGER = 2) in a central 30×30 region around the origin
    // (cells 113-143 at 256×256 resolution, assuming the world maps [0, 256] to cells).
    for i in 110..150 {
        for j in 110..150 {
            if i < biome_res && j < biome_res {
                biomes[i * biome_res + j] = 2u32; // BIOME_DANGER
            }
        }
    }

    // Place one food item per agent, each within FOOD_SENSE_RADIUS (30.0) of spawn.
    let mut food_pos = Vec::new();
    let mut food_consumed = Vec::new();
    let mut food_timers = Vec::new();

    // 4×4 lattice near origin: agents at positions (x, y) where x, y ∈ {0, 12, 24, 36}.
    // Each agent spawned at (spawn_x, 1.0, spawn_z); place food at (spawn_x + 10, 0, spawn_z + 10)
    // (distance ≈ 14.1, well within FOOD_SENSE_RADIUS = 30.0).
    for i in 0..4 {
        for j in 0..4 {
            let spawn_x = (i as f32) * 12.0;
            let spawn_z = (j as f32) * 12.0;
            let food_x = spawn_x + 10.0;
            let food_z = spawn_z + 10.0;
            food_pos.push((food_x, 0.0, food_z));
            food_consumed.push(false);
            food_timers.push(0.0);
        }
    }

    kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agent data: spawn on the lattice.
    let mut agent_data = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let spawn_x = (i as f32) * 12.0;
            let spawn_z = (j as f32) * 12.0;
            agent_data.push((
                glam::Vec3::new(spawn_x, 1.0, spawn_z),
                100.0, // energy
                100.0, // integrity
                brain_config.memory_capacity,
                brain_config.processing_slots,
            ));
        }
    }
    kernel.upload_agents(&agent_data);

    // Run 8 batches to accumulate intent counters.
    let batch_size = kernel.kernel_batch_size();
    let num_batches = 8;

    for batch_idx in 0..num_batches {
        let start_tick = (batch_idx as u64) * batch_size as u64;
        kernel.dispatch_batch(start_tick, batch_size);
    }

    // Read intent fractions for each agent.
    let mut approach_fractions = Vec::new();
    let mut avoidance_fractions = Vec::new();
    let mut total_approach_sense_range_ticks = 0.0;
    let mut total_avoidance_sense_range_ticks = 0.0;

    for agent_id in 0..agent_count {
        let telemetry = kernel.read_agent_telemetry_blocking(agent_id);

        // Approach-intent: fraction of in-range ticks the agent turned toward food.
        let approach_sense_range = telemetry.approach_sense_range_ticks;
        let approach_turns = telemetry.approach_turns_toward;
        let approach_fraction = if approach_sense_range > 1e-6 {
            approach_turns / approach_sense_range
        } else {
            0.0
        };
        approach_fractions.push(approach_fraction);
        total_approach_sense_range_ticks += approach_sense_range;

        // Avoidance-intent: fraction of in-range ticks the agent turned away from danger.
        let avoidance_sense_range = telemetry.avoidance_sense_range_ticks;
        let avoidance_turns = telemetry.avoidance_turns_opposing;
        let avoidance_fraction = if avoidance_sense_range > 1e-6 {
            avoidance_turns / avoidance_sense_range
        } else {
            0.0
        };
        avoidance_fractions.push(avoidance_fraction);
        total_avoidance_sense_range_ticks += avoidance_sense_range;
    }

    // Validity guard: check that the world exercises both axes.
    if total_approach_sense_range_ticks < 1e-6 {
        eprintln!("WARNING: approach axis is degenerate (zero summed sense_range_ticks); world layout failed to exercise food detection");
    }
    if total_avoidance_sense_range_ticks < 1e-6 {
        eprintln!("WARNING: avoidance axis is degenerate (zero summed sense_range_ticks); world layout failed to exercise danger detection");
    }

    // Compute statistics for each axis.
    let compute_stats = |fractions: &[f32]| -> (f32, f32, f32, f32, f32, f32, f32) {
        let mean = fractions.iter().sum::<f32>() / fractions.len() as f32;
        let variance =
            fractions.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / fractions.len() as f32;
        let std = variance.sqrt();
        let min = fractions.iter().copied().fold(f32::INFINITY, f32::min);
        let max = fractions.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute percentiles (p25, p50, p75) by sorting.
        let mut sorted = fractions.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sorted.len();
        let p25_idx = ((0.25 * len as f32) as usize).min(len - 1);
        let p50_idx = ((0.5 * len as f32) as usize).min(len - 1);
        let p75_idx = ((0.75 * len as f32) as usize).min(len - 1);
        let p25 = sorted[p25_idx];
        let p50 = sorted[p50_idx];
        let p75 = sorted[p75_idx];

        (mean, std, min, max, p25, p50, p75)
    };

    let (
        approach_mean,
        approach_std,
        approach_min,
        approach_max,
        approach_p25,
        approach_p50,
        approach_p75,
    ) = compute_stats(&approach_fractions);
    let (
        avoidance_mean,
        avoidance_std,
        avoidance_min,
        avoidance_max,
        avoidance_p25,
        avoidance_p50,
        avoidance_p75,
    ) = compute_stats(&avoidance_fractions);

    eprintln!("Approach-intent baseline statistics (16 agents):");
    eprintln!("  mean: {approach_mean:.6}");
    eprintln!("  std:  {approach_std:.6}");
    eprintln!("  min:  {approach_min:.6}");
    eprintln!("  max:  {approach_max:.6}");
    eprintln!("  p25:  {approach_p25:.6}");
    eprintln!("  p50:  {approach_p50:.6}");
    eprintln!("  p75:  {approach_p75:.6}");

    eprintln!("Avoidance-intent baseline statistics (16 agents):");
    eprintln!("  mean: {avoidance_mean:.6}");
    eprintln!("  std:  {avoidance_std:.6}");
    eprintln!("  min:  {avoidance_min:.6}");
    eprintln!("  max:  {avoidance_max:.6}");
    eprintln!("  p25:  {avoidance_p25:.6}");
    eprintln!("  p50:  {avoidance_p50:.6}");
    eprintln!("  p75:  {avoidance_p75:.6}");
}
