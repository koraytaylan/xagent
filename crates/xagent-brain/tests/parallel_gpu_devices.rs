//! Several tests each build a wgpu device. The Vulkan loader segfaults in
//! `vkSetDebugUtilsObjectNameEXT` when those calls overlap. This drives the
//! same entry points (`GpuKernel::new`, dispatch, blocking readback) from
//! four threads.

use xagent_brain::buffers::{PHYS_STRIDE, P_ALIVE};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Width that segfaulted in the loader when integration tests ran in parallel.
const PARALLEL_DEVICES: usize = 4;
const TERRAIN_SIDE: usize = 129;
const BIOME_SIDE: usize = 256;

#[test]
fn parallel_gpu_devices_complete_a_dispatch_and_readback() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    std::thread::scope(|scope| {
        for index in 0..PARALLEL_DEVICES {
            scope.spawn(move || {
                let brain = BrainConfig {
                    brain_tick_stride: 1,
                    vision_stride: 1,
                    movement_speed: 0.0,
                    ..BrainConfig::default()
                };
                let world = WorldConfig {
                    seed: u64::from(index as u32) + 1,
                    ..WorldConfig::default()
                };
                let mut kernel = GpuKernel::new(1, 1, &brain, &world);
                let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
                let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
                kernel.upload_world(&heights, &biomes, &[(40.0, 0.0, 40.0)], &[false], &[0.0]);
                kernel.upload_agents(&[(
                    glam::Vec3::new(0.0, 1.0, 0.0),
                    100.0,
                    100.0,
                    brain.memory_capacity,
                    brain.processing_slots,
                )]);
                kernel.dispatch_ticks(0, 1);
                let state = kernel.read_full_state_blocking();
                assert_eq!(state.len(), PHYS_STRIDE);
                assert!(
                    state[P_ALIVE] >= 0.5,
                    "device {index} lost the agent on a single full-energy tick"
                );
            });
        }
    });
}
