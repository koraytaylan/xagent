//! The extra sensory channels after the scent channels are copied into the
//! brain features by both feature extractors, directly after the scent
//! features. With the second odour sense off the features are unchanged.

use xagent_brain::buffers::{
    BrainLayout, ENCODED_DIMENSION, NON_VISUAL_COUNT, SCENT_CHANNELS, SCRATCH_FEATURES,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, FoodKind, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity meter value.
const FULL_METER: f32 = 100.0;
/// World seed shared by every kernel in this file.
const WORLD_SEED: u64 = 5;
/// Seed for the agent brain reset.
const RESET_SEED: u64 = 11;
/// Ticks for the fused-extractor tests.
const BRAIN_TICKS: u64 = 20;
/// Index of the first scent feature.
const SCENT_FEATURE: usize = 25;
/// Index of the first extra-sensory feature.
const EXTRA_FEATURE: usize = SCENT_FEATURE + SCENT_CHANNELS;
/// Number of extra sensory channels with the second sense on.
const EXTRA_CHANNELS: usize = 2;
/// Index of the `second_odour` block in the declared sense blocks.
const SECOND_ODOUR_BLOCK: usize = 2;
/// Width of the danger percept features when enabled.
const DANGER_FEATURE_WIDTH: usize = 2;
/// Number of kernels built to compare feature vectors with the flag off.
const KERNEL_RUN_COUNT: usize = 2;
/// Index of the left nostril of pair 1 in the non-visual tail.
const SCENT_SLOT: usize = NON_VISUAL_COUNT - SCENT_CHANNELS;
/// Index of the left nostril of pair 2 in the non-visual tail.
const SECOND_SLOT: usize = NON_VISUAL_COUNT;
/// Food spread over both kinds, all inside smelling range and out of reach.
const FOOD_POSITIONS: [(f32, f32, f32); 6] = [
    (-4.0, 0.35, 6.0),
    (5.0, 0.35, 4.0),
    (-6.0, 0.35, -5.0),
    (3.0, 0.35, 9.0),
    (-2.0, 0.35, 12.0),
    (8.0, 0.35, -3.0),
];
/// A spot far beyond smelling range of every agent.
const REMOTE_FOOD: (f32, f32, f32) = (-60.0, 0.35, -60.0);
/// Spawn positions of the two test agents.
const AGENT_POSITIONS: [(f32, f32, f32); 2] = [(0.0, 1.0, 0.0), (20.0, 1.0, 20.0)];

fn gpu_or_skip() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var_os("XAGENT_REQUIRE_GPU").is_none(),
        "no GPU adapter available but XAGENT_REQUIRE_GPU is set"
    );
    eprintln!("Skipping: no GPU/fallback adapter available");
    false
}

fn brain_config(second: bool) -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        second_odour_sense_enabled: second,
        ..BrainConfig::default()
    }
}

fn world_config() -> WorldConfig {
    WorldConfig {
        seed: WORLD_SEED,
        food_kinds_enabled: true,
        ..WorldConfig::default()
    }
}

/// The food list with every `Second` item moved far out of smelling range,
/// so item indices and kinds are unchanged.
fn first_only_food() -> Vec<(f32, f32, f32)> {
    FOOD_POSITIONS
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if FoodKind::of_item(i) == FoodKind::First {
                *p
            } else {
                REMOTE_FOOD
            }
        })
        .collect()
}

/// Build a kernel, upload the scene and run `ticks` ticks.
fn setup(brain: &BrainConfig, food: &[(f32, f32, f32)], ticks: u64) -> GpuKernel {
    let agent_count = u32::try_from(AGENT_POSITIONS.len()).expect("agent count fits u32");
    let mut kernel = GpuKernel::new(agent_count, food.len(), brain, &world_config());
    kernel.reset_agents_seeded(brain, RESET_SEED);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let eaten = vec![false; food.len()];
    kernel.upload_world(&heights, &biomes, food, &eaten, &vec![0.0; food.len()]);
    let agents: Vec<_> = AGENT_POSITIONS
        .iter()
        .map(|p| {
            (
                glam::Vec3::new(p.0, p.1, p.2),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    for i in 0..AGENT_POSITIONS.len() {
        let index = u32::try_from(i).expect("agent index fits u32");
        kernel.write_motor_decision(index, 0.0, 0.0, 0.0);
    }
    for tick in 0..ticks {
        kernel.dispatch_batch(tick, 1);
    }
    kernel
}

/// Whitened input of the third sense block for one agent.
fn third_block_input(kernel: &GpuKernel, agent: u32) -> Vec<f32> {
    let layout = kernel.brain_layout();
    let block = layout.sense_blocks()[SECOND_ODOUR_BLOCK];
    // The block offsets are written for the default feature width; the
    // encoder weights ahead of them grow with the feature count.
    let shift = (layout.feature_count - BrainLayout::default().feature_count) * ENCODED_DIMENSION;
    let start = block.whitened_input + shift;
    let state = kernel.read_agent_state(agent);
    state.brain_state[start..start + block.channels].to_vec()
}

#[test]
fn second_odour_scratch_features_follow_scent() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config(true);
    let mut kernel = setup(&brain, &FOOD_POSITIONS, 1);
    kernel.dispatch_feature_phase_for_test();
    let scratch = kernel.read_brain_scratch_blocking();
    let stride = kernel.brain_layout().brain_scratch_stride;
    let layout = kernel.brain_layout();
    let danger = if layout.danger_percept_enabled {
        DANGER_FEATURE_WIDTH
    } else {
        0
    };
    let non_visual_base = layout.feature_count
        - (SCENT_FEATURE + SCENT_CHANNELS + layout.extra_sensory_channels + danger);
    let mut any_extra = false;
    for agent in 0..AGENT_POSITIONS.len() {
        let tail = kernel
            .read_agent_telemetry_blocking(u32::try_from(agent).expect("agent fits u32"))
            .sensory_non_visual;
        let base = agent * stride + SCRATCH_FEATURES + non_visual_base;
        for c in 0..SCENT_CHANNELS {
            assert_eq!(
                scratch[base + SCENT_FEATURE + c].to_bits(),
                tail[SCENT_SLOT + c].to_bits(),
                "agent {agent} scent feature {c}"
            );
        }
        for c in 0..EXTRA_CHANNELS {
            assert_eq!(
                scratch[base + EXTRA_FEATURE + c].to_bits(),
                tail[SECOND_SLOT + c].to_bits(),
                "agent {agent} extra feature {c}"
            );
            any_extra |= tail[SECOND_SLOT + c] != 0.0;
        }
    }
    assert!(any_extra, "no second odour was smelt");
}

#[test]
fn second_odour_block_reads_fused_features() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config(true);
    let both = setup(&brain, &FOOD_POSITIONS, BRAIN_TICKS);
    let nonzero = (0..AGENT_POSITIONS.len()).any(|a| {
        third_block_input(&both, u32::try_from(a).expect("agent fits u32"))
            .iter()
            .any(|v| *v != 0.0)
    });
    assert!(nonzero, "second odour block input stayed zero");

    let first_only = first_only_food();
    let none = setup(&brain, &first_only, BRAIN_TICKS);
    for a in 0..AGENT_POSITIONS.len() {
        for (slot, v) in third_block_input(&none, u32::try_from(a).expect("agent fits u32"))
            .iter()
            .enumerate()
        {
            assert_eq!(
                *v, 0.0,
                "agent {a} slot {slot} nonzero without second odour"
            );
        }
    }
}

#[test]
fn second_odour_features_off_leave_features_unchanged() {
    if !gpu_or_skip() {
        return;
    }
    let brain = brain_config(false);
    let count = BrainLayout::default().feature_count;
    let mut features = Vec::new();
    for _ in 0..KERNEL_RUN_COUNT {
        let mut kernel = setup(&brain, &FOOD_POSITIONS, BRAIN_TICKS);
        kernel.dispatch_feature_phase_for_test();
        let scratch = kernel.read_brain_scratch_blocking();
        assert_eq!(kernel.brain_layout().feature_count, count);
        assert!(scratch.len() >= SCRATCH_FEATURES + count);
        let v: Vec<u32> = scratch[SCRATCH_FEATURES..SCRATCH_FEATURES + count]
            .iter()
            .map(|f| f.to_bits())
            .collect();
        features.push(v);
    }
    assert_eq!(features[0], features[1], "features differ across kernels");
}
