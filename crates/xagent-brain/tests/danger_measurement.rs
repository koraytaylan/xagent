//! The nearest-danger measurement behind the avoidance-intent counters runs
//! whether or not the brain is given the danger percept: with the percept off
//! (the default) the kernel still finds the nearest hazard cell and its
//! bearing, matching a CPU scan of the same grid. Before, the percept flag
//! also switched the measurement off, so the counters read a stale distance
//! with a zero bearing and never registered a turn away. The counters count
//! only ticks with the hazard ahead, inside the agent's horizontal field of
//! view, and the agent off hazard ground.

use xagent_brain::buffers::{
    DANGER_SENSE_RADIUS, PHYS_STRIDE, P_AVOIDANCE_SENSE_RANGE_TICKS, P_FACING_X, P_FACING_Z,
    P_HAZARD_ENTRIES, P_NEAREST_DANGER_BEARING, P_NEAREST_DANGER_DISTANCE, P_POS_X, P_POS_Z,
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
/// The agent faces +Z, so hazard ground from this biome row on lies ahead of
/// an agent standing at `AGENT_Z_BEFORE_HAZARD`, and under one at
/// `AGENT_Z_ON_HAZARD`.
const HAZARD_FIRST_ROW: usize = BIOME_SIDE / 2;
const AGENT_Z_BEFORE_HAZARD: f32 = -10.0;
const AGENT_Z_ON_HAZARD: f32 = 10.0;
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

/// Run one agent at `position` over `biomes` for a few ticks with the danger
/// percept off; returns its physics row.
fn probe(biomes: &[u32], position: glam::Vec3) -> (Vec<f32>, f32) {
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
    kernel.upload_world(&heights, biomes, &[(-60.0, 0.35, 6.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        position,
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    (
        kernel.read_full_state_blocking()[..PHYS_STRIDE].to_vec(),
        world.world_size,
    )
}

fn hazard_where(is_hazard: impl Fn(usize, usize) -> bool) -> Vec<u32> {
    (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            if is_hazard(cell / BIOME_SIDE, cell % BIOME_SIDE) {
                BIOME_DANGER
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn danger_is_measured_with_the_percept_off() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Hazard ground to the east, beside an agent facing +Z.
    let biomes = hazard_where(|_, col| col >= HAZARD_FIRST_COLUMN);
    let (physics, world_size) = probe(&biomes, glam::Vec3::new(AGENT_X, 1.0, 0.0));

    let (distance, bearing) = nearest_danger(
        &biomes,
        world_size,
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
    // Beside the agent, outside its 90° view: not an avoidance tick.
    assert_eq!(
        physics[P_AVOIDANCE_SENSE_RANGE_TICKS], 0.0,
        "a hazard beside the agent should not count"
    );
}

#[test]
fn only_hazard_ahead_and_off_it_counts_for_avoidance() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let biomes = hazard_where(|row, _| row >= HAZARD_FIRST_ROW);

    let (ahead, _) = probe(&biomes, glam::Vec3::new(0.0, 1.0, AGENT_Z_BEFORE_HAZARD));
    assert!(
        ahead[P_AVOIDANCE_SENSE_RANGE_TICKS] >= 1.0,
        "hazard ahead in view should count: {}",
        ahead[P_AVOIDANCE_SENSE_RANGE_TICKS]
    );

    let (on, _) = probe(&biomes, glam::Vec3::new(0.0, 1.0, AGENT_Z_ON_HAZARD));
    assert_eq!(
        on[P_AVOIDANCE_SENSE_RANGE_TICKS], 0.0,
        "standing on hazard ground should not count"
    );
}

#[test]
fn a_step_onto_hazard_ground_is_counted_once() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let biomes = hazard_where(|row, _| row >= HAZARD_FIRST_ROW);

    // Placed on hazard ground: one step onto it (off it before the first
    // tick), however long it then stays.
    let (on, _) = probe(&biomes, glam::Vec3::new(0.0, 1.0, AGENT_Z_ON_HAZARD));
    assert_eq!(on[P_HAZARD_ENTRIES], 1.0, "one step onto hazard ground");

    let (off, _) = probe(&biomes, glam::Vec3::new(0.0, 1.0, AGENT_Z_BEFORE_HAZARD));
    assert_eq!(off[P_HAZARD_ENTRIES], 0.0, "never on hazard ground");
}
