//! The nearest-danger measurement behind the avoidance-intent counters runs
//! whether or not the brain is given the danger percept: with the percept off
//! (the default) the kernel still finds the nearest hazard cell and its
//! bearing, matching a CPU scan of the same grid, and counts the in-range
//! ticks. Before, the percept flag also switched the measurement off, so the
//! counters read a stale distance with a zero bearing and never registered a
//! turn away.

use xagent_brain::buffers::{
    DANGER_SENSE_RADIUS, PHYS_STRIDE, P_AVOIDANCE_SENSE_RANGE_TICKS, P_FACING_X, P_FACING_Z,
    P_NEAREST_DANGER_BEARING, P_NEAREST_DANGER_DISTANCE, P_POS_X, P_POS_Z,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Mirrors `BIOME_DANGER` in `common.wgsl`.
const BIOME_DANGER: u32 = 2;
const FULL_METER: f32 = 100.0;
const TICKS: u64 = 4;
/// Hazard ground starts at this biome column; the agent stands this far west
/// of it, inside the sense radius.
const HAZARD_FIRST_COLUMN: usize = BIOME_SIDE / 2;
const AGENT_X: f32 = -10.0;
/// f32 rounding between the GPU and the f64 reference.
const TOLERANCE: f32 = 1e-4;

/// Nearest danger cell centre within the sense radius, scanned in row-major
/// order with ties to the first, and its signed bearing from the facing.
fn nearest_danger(
    biomes: &[u32],
    world_size: f32,
    pos: (f32, f32),
    facing: (f32, f32),
) -> (f32, f32) {
    let inv = BIOME_SIDE as f64 / f64::from(world_size);
    let half = f64::from(world_size) / 2.0;
    let radius = f64::from(DANGER_SENSE_RADIUS);
    let reach = (radius * inv).ceil() as i64 + 1;
    let (x, z) = (f64::from(pos.0), f64::from(pos.1));
    let col0 = ((x + half) * inv) as i64;
    let row0 = ((z + half) * inv) as i64;
    let max_index = BIOME_SIDE as i64 - 1;
    let mut best: Option<(f64, f64, f64)> = None;
    for dr in -reach..=reach {
        for dc in -reach..=reach {
            let row = (row0 + dr).clamp(0, max_index) as usize;
            let col = (col0 + dc).clamp(0, max_index) as usize;
            if biomes[row * BIOME_SIDE + col] != BIOME_DANGER {
                continue;
            }
            let cx = (col as f64 + 0.5) / inv - half - x;
            let cz = (row as f64 + 0.5) / inv - half - z;
            let distance = (cx * cx + cz * cz).sqrt();
            if distance < best.map_or(radius, |b| b.0) && distance > 1e-6 {
                best = Some((distance, cx, cz));
            }
        }
    }
    let (distance, cx, cz) = best.expect("a hazard cell lies within the sense radius");
    let (fx, fz) = (f64::from(facing.0), f64::from(facing.1));
    let bearing = (fx * cz - fz * cx).atan2(fx * cx + fz * cz);
    (distance as f32, bearing as f32)
}

#[test]
fn danger_is_measured_with_the_percept_off() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        danger_percept_enabled: false,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 6,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, 17);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes: Vec<u32> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            if cell % BIOME_SIDE >= HAZARD_FIRST_COLUMN {
                BIOME_DANGER
            } else {
                0
            }
        })
        .collect();
    kernel.upload_world(&heights, &biomes, &[(-60.0, 0.35, 6.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(AGENT_X, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    let physics = kernel.read_full_state_blocking()[..PHYS_STRIDE].to_vec();

    let (distance, bearing) = nearest_danger(
        &biomes,
        world.world_size,
        (physics[P_POS_X], physics[P_POS_Z]),
        (physics[P_FACING_X], physics[P_FACING_Z]),
    );
    assert!(distance < DANGER_SENSE_RADIUS);
    assert!(
        (physics[P_NEAREST_DANGER_DISTANCE] - distance).abs() < TOLERANCE,
        "nearest danger distance: GPU {} vs CPU {distance}",
        physics[P_NEAREST_DANGER_DISTANCE]
    );
    assert!(
        (physics[P_NEAREST_DANGER_BEARING] - bearing).abs() < TOLERANCE,
        "nearest danger bearing: GPU {} vs CPU {bearing}",
        physics[P_NEAREST_DANGER_BEARING]
    );
    assert!(
        physics[P_AVOIDANCE_SENSE_RANGE_TICKS] >= 1.0,
        "the in-range ticks were not counted: {}",
        physics[P_AVOIDANCE_SENSE_RANGE_TICKS]
    );
}
