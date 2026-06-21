//! Falsifiable test: raw_gradient contains only homeostatic deltas at default config.
//!
//! Verifies that raw_gradient contains only homeostatic deltas (energy + integrity),
//! with both approach-PBRS and avoidance-PBRS terms removed. This is a regression guard:
//! if a future edit reintroduces shaping, this test will fail.
//!
//! Pre-removal baseline (measured at default config before shaping term removal):
//! mean: 0.000412, std: 0.000173, min: -0.000381, max: 0.001878

use xagent_brain::{buffers::*, GpuKernel};
use xagent_shared::{BrainConfig, WorldConfig};

/// Fractional weight applied to the normalized energy delta when computing the homeostatic gradient.
/// Must match `ENERGY_WEIGHT` in `crates/xagent-brain/src/shaders/kernel/common.wgsl`.
const ENERGY_WEIGHT: f32 = 0.6;

/// Fractional weight applied to the normalized integrity delta when computing the homeostatic gradient.
/// Must match `INTEGRITY_WEIGHT` in `crates/xagent-brain/src/shaders/kernel/common.wgsl`.
const INTEGRITY_WEIGHT: f32 = 0.4;

#[test]
fn test_raw_gradient_is_homeostatic_only() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain_config = BrainConfig::default();
    let world_config = WorldConfig::default();

    // Test Case 1: Zero Deltas
    // Set P_ENERGY = P_PREV_ENERGY and P_INTEGRITY = P_PREV_INTEGRITY.
    // Upload sets both to the same value (max_energy/max_integrity).
    // Run only the brain pass (phase_mask=0x4) so physics pass doesn't modify them.
    //
    // Note: The brain pass compares current energy/integrity (from physics_state)
    // against previous values (stored in brain_state O_HOMEO+4/5). On the first
    // tick, these previous values are 0.0, so we run the brain pass twice: the first
    // run initializes the previous values in brain_state, and the second run computes
    // deltas with those values.
    {
        let mut kernel = GpuKernel::new(1, 1, &brain_config, &world_config);

        // Upload single agent with matching current and previous energy/integrity.
        // upload_agents sets both P_ENERGY and P_PREV_ENERGY to max_e = 100.0,
        // and both P_INTEGRITY and P_PREV_INTEGRITY to max_i = 100.0.
        let agent_data = vec![(
            glam::Vec3::new(0.0, 1.0, 0.0),
            100.0, // max_energy (sets P_ENERGY and P_PREV_ENERGY)
            100.0, // max_integrity (sets P_INTEGRITY and P_PREV_INTEGRITY)
            brain_config.memory_capacity,
            brain_config.processing_slots,
        )];
        kernel.upload_agents(&agent_data);

        // Minimal world setup
        let terrain_vps = 129;
        let heights = vec![0.0_f32; terrain_vps * terrain_vps];
        let biome_res = 256;
        let biomes = vec![0_u32; biome_res * biome_res];
        let food_pos = vec![(50.0, 0.0, 50.0)];
        let food_consumed = vec![false];
        let food_timers = vec![0.0];
        kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

        // Dispatch ONLY the brain pass (phase_mask=0x4) so physics pass doesn't execute.
        // This preserves P_ENERGY = P_PREV_ENERGY = 100.0 and P_INTEGRITY = P_PREV_INTEGRITY = 100.0.
        // First pass initializes the brain_state O_HOMEO+4/5 values.
        kernel.dispatch_batch_masked(0, kernel.brain_tick_stride(), 0x4);

        // Second pass: now O_HOMEO+4/5 are initialized to match P_ENERGY/P_INTEGRITY,
        // so the delta computation will use those values and compute zero deltas.
        kernel.dispatch_batch_masked(
            kernel.brain_tick_stride() as u64,
            kernel.brain_tick_stride(),
            0x4,
        );

        // Read raw_gradient: should be 0.0 (both deltas are zero)
        // raw_gradient = 0.0 * ENERGY_WEIGHT + 0.0 * INTEGRITY_WEIGHT = 0.0
        let telemetry = kernel.read_agent_telemetry_blocking(0);
        assert!(
            (telemetry.raw_gradient - 0.0).abs() < 1e-6,
            "Zero deltas should yield raw_gradient = 0.0, got {}",
            telemetry.raw_gradient
        );
    }

    // Test Case 2: Energy Only
    // Set P_ENERGY = 110.0, P_PREV_ENERGY = 100.0, so energy_delta = (110.0-100.0)/100.0 = 0.1.
    // Keep P_INTEGRITY = P_PREV_INTEGRITY = 100.0 so integrity_delta = 0.
    // Expected raw_gradient = 0.1 * ENERGY_WEIGHT (where ENERGY_WEIGHT = 0.6 per common.wgsl)
    // = 0.1 * 0.6 = 0.06
    {
        let mut kernel = GpuKernel::new(1, 1, &brain_config, &world_config);

        let agent_data = vec![(
            glam::Vec3::new(0.0, 1.0, 0.0),
            100.0,
            100.0,
            brain_config.memory_capacity,
            brain_config.processing_slots,
        )];
        kernel.upload_agents(&agent_data);

        // Minimal world setup
        let terrain_vps = 129;
        let heights = vec![0.0_f32; terrain_vps * terrain_vps];
        let biome_res = 256;
        let biomes = vec![0_u32; biome_res * biome_res];
        let food_pos = vec![(50.0, 0.0, 50.0)];
        let food_consumed = vec![false];
        let food_timers = vec![0.0];
        kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

        // First brain pass to initialize O_HOMEO+4/5 from current values.
        kernel.dispatch_batch_masked(0, kernel.brain_tick_stride(), 0x4);

        // Now set P_ENERGY = 110.0 (keeping P_PREV_ENERGY = 100.0 from initialization).
        // Note: we don't touch P_INTEGRITY/P_PREV_INTEGRITY, so they remain at 100.0.
        kernel.write_agent_physics_fields(0, &[(P_ENERGY, 110.0)]);

        // Second brain pass: now it will compute energy_delta = (110-100)/100 = 0.1,
        // and integrity_delta = (100-100)/100 = 0.0.
        kernel.dispatch_batch_masked(
            kernel.brain_tick_stride() as u64,
            kernel.brain_tick_stride(),
            0x4,
        );

        // Read raw_gradient: should be 0.1 * ENERGY_WEIGHT = 0.06
        let expected = 0.1 * ENERGY_WEIGHT;
        let telemetry = kernel.read_agent_telemetry_blocking(0);
        assert!(
            (telemetry.raw_gradient - expected).abs() < 1e-6,
            "Energy delta 0.1 should yield raw_gradient = {}, got {}",
            expected,
            telemetry.raw_gradient
        );
    }

    // Test Case 3: Integrity Only
    // Keep P_ENERGY = P_PREV_ENERGY = 100.0 so energy_delta = 0.
    // Set P_INTEGRITY = 105.0, P_PREV_INTEGRITY = 100.0, so integrity_delta = (105.0-100.0)/100.0 = 0.05.
    // Expected raw_gradient = 0.05 * INTEGRITY_WEIGHT (where INTEGRITY_WEIGHT = 0.4 per common.wgsl)
    // = 0.05 * 0.4 = 0.02
    {
        let mut kernel = GpuKernel::new(1, 1, &brain_config, &world_config);

        let agent_data = vec![(
            glam::Vec3::new(0.0, 1.0, 0.0),
            100.0,
            100.0,
            brain_config.memory_capacity,
            brain_config.processing_slots,
        )];
        kernel.upload_agents(&agent_data);

        // Minimal world setup
        let terrain_vps = 129;
        let heights = vec![0.0_f32; terrain_vps * terrain_vps];
        let biome_res = 256;
        let biomes = vec![0_u32; biome_res * biome_res];
        let food_pos = vec![(50.0, 0.0, 50.0)];
        let food_consumed = vec![false];
        let food_timers = vec![0.0];
        kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);

        // First brain pass to initialize O_HOMEO+4/5 from current values.
        kernel.dispatch_batch_masked(0, kernel.brain_tick_stride(), 0x4);

        // Now set P_INTEGRITY = 105.0 (keeping P_PREV_INTEGRITY = 100.0 from initialization).
        // Note: we don't touch P_ENERGY/P_PREV_ENERGY, so they remain at 100.0.
        kernel.write_agent_physics_fields(0, &[(P_INTEGRITY, 105.0)]);

        // Second brain pass: now it will compute integrity_delta = (105-100)/100 = 0.05,
        // and energy_delta = (100-100)/100 = 0.0.
        kernel.dispatch_batch_masked(
            kernel.brain_tick_stride() as u64,
            kernel.brain_tick_stride(),
            0x4,
        );

        // Read raw_gradient: should be 0.05 * INTEGRITY_WEIGHT = 0.02
        let expected = 0.05 * INTEGRITY_WEIGHT;
        let telemetry = kernel.read_agent_telemetry_blocking(0);
        assert!(
            (telemetry.raw_gradient - expected).abs() < 1e-6,
            "Integrity delta 0.05 should yield raw_gradient = {}, got {}",
            expected,
            telemetry.raw_gradient
        );
    }
}
