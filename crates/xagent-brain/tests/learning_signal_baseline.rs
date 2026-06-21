//! Measurement probe for baseline raw_gradient distribution pre-removal.
//!
//! This test captures the pre-removal raw_gradient distribution (mean, std, min, max)
//! over 100 measurement points at default config, serving as the reference baseline for the
//! post-removal tests. The baseline is recorded in the doc comment below for use by
//! `add-homeostatic-only-gate` and other post-removal validation.
//!
//! Each measurement point dispatches one full kernel batch
//! (`kernel.kernel_batch_size()` = `vision_stride * brain_tick_stride` = 100 ticks at
//! default config) so that the brain pass executes at least once per sample.
//!
//! No assertions — this is a measurement probe, not a gate.

use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Baseline raw_gradient distribution over 100 measurement points at default config.
/// Each point dispatches one full kernel batch (brain_tick_stride * vision_stride ticks).
/// Recorded after running the test (pre-removal, approach-PBRS on, avoidance-PBRS off):
/// mean: 0.000412, std: 0.000173, min: -0.000381, max: 0.001878
///
/// This baseline is captured before removing approach-PBRS and avoidance-PBRS
/// terms, and is embedded in the homeostatic-only regression test for reference.
#[test]
fn learning_signal_baseline() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain_config = BrainConfig::default();
    let world_config = WorldConfig::default();

    let mut kernel = GpuKernel::new(1, 1, &brain_config, &world_config);
    kernel.reset_agents_seeded(&brain_config, 12345);

    // Minimal world setup (flat terrain, no biome, one food far away)
    let terrain_vps = 129;
    let heights = vec![0.0_f32; terrain_vps * terrain_vps];
    let biome_res = 256;
    let biomes = vec![0_u32; biome_res * biome_res];
    let food_pos = vec![(50.0, 0.0, 50.0)];
    let food_consumed = vec![false];
    let food_timers = vec![0.0];

    kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

    // Upload agent data: position, energy, integrity, memory capacity, processing slots
    let agent_data = vec![(
        glam::Vec3::new(0.0, 1.0, 0.0),
        100.0,
        100.0,
        brain_config.memory_capacity,
        brain_config.processing_slots,
    )];
    kernel.upload_agents(&agent_data);

    // One full batch per measurement point guarantees the brain pass executes.
    // At default config: vision_stride=10, brain_tick_stride=10, so batch_size=100.
    let batch_size = kernel.kernel_batch_size();
    let mut raw_gradients = Vec::new();

    for point_idx in 0..100u64 {
        let start_tick = point_idx * batch_size as u64;
        kernel.dispatch_batch(start_tick, batch_size);
        let telemetry = kernel.read_agent_telemetry_blocking(0);
        raw_gradients.push(telemetry.raw_gradient);
    }

    let mean = raw_gradients.iter().sum::<f32>() / raw_gradients.len() as f32;
    let variance = raw_gradients
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / raw_gradients.len() as f32;
    let std = variance.sqrt();
    let min = raw_gradients.iter().copied().fold(f32::INFINITY, f32::min);
    let max = raw_gradients
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "Learning signal baseline statistics (100 measurement points, batch_size={batch_size}):"
    );
    eprintln!("  mean: {mean:.6}");
    eprintln!("  std:  {std:.6}");
    eprintln!("  min:  {min:.6}");
    eprintln!("  max:  {max:.6}");
}
