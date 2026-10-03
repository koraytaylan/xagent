//! Food contested by several agents in the same cycle goes to the agent with
//! the lowest index, whichever workgroup runs first: every agent claims the
//! food in reach in one dispatch, and the claims are settled in the next. With
//! the grid cells kept in index order this makes a seeded run reproduce
//! exactly.

use xagent_brain::buffers::{PHYS_STRIDE, P_FOOD_COUNT};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
const FOOD_HEIGHT: f32 = 0.35;
/// Both claimants stand this far either side of the food, well inside the
/// eat radius (`WC_FOOD_RADIUS`, 2.0).
const CLAIMANT_OFFSET: f32 = 0.5;
/// A bystander far from the food, so the claimants are not agent 0.
const BYSTANDER_X: f32 = -40.0;
/// One brain cycle at the default stride.
const ONE_CYCLE: u32 = 10;

/// Meals of each agent after one cycle with the given agents around one food
/// item at the origin.
fn meals_after_one_cycle(positions: &[f32]) -> Vec<f32> {
    let brain = BrainConfig {
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 4,
        ..WorldConfig::default()
    };
    let agents = positions.len();
    let mut kernel = GpuKernel::new(agents as u32, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, 8);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(
        &heights,
        &biomes,
        &[(0.0, FOOD_HEIGHT, 0.0)],
        &[false],
        &[0.0],
    );
    let bodies: Vec<_> = positions
        .iter()
        .map(|&x| {
            (
                glam::Vec3::new(x, 1.0, 0.0),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&bodies);
    kernel.dispatch_batch(0, ONE_CYCLE);
    let physics = kernel.read_full_state_blocking().to_vec();
    (0..agents)
        .map(|a| physics[a * PHYS_STRIDE + P_FOOD_COUNT])
        .collect()
}

#[test]
fn contested_food_goes_to_the_lowest_index_agent() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    // Agents 0 and 1 contest the food.
    assert_eq!(
        meals_after_one_cycle(&[-CLAIMANT_OFFSET, CLAIMANT_OFFSET]),
        vec![1.0, 0.0]
    );
    // The same two positions, the other way round: still agent 0.
    assert_eq!(
        meals_after_one_cycle(&[CLAIMANT_OFFSET, -CLAIMANT_OFFSET]),
        vec![1.0, 0.0]
    );
    // Agents 1 and 2 contest it while agent 0 is far away: agent 1.
    assert_eq!(
        meals_after_one_cycle(&[BYSTANDER_X, CLAIMANT_OFFSET, -CLAIMANT_OFFSET]),
        vec![0.0, 1.0, 0.0]
    );
}
