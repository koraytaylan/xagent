//! Running the brain beside the vision pass (after the global pass, in one
//! dispatch with the vision workgroups) gives exactly the results of the
//! serial cycle, in which the brain runs inside the kernel before the global
//! pass and vision after it. The world is set up so that every input the
//! reorder could disturb is exercised: agents close enough to collide (the
//! global pass moves them before the brain now runs) and to see each other,
//! food to see and eat, and hazard ground.

use xagent_brain::buffers::PHYS_STRIDE;
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Mirrors `BIOME_DANGER` in `common.wgsl`.
const BIOME_DANGER: u32 = 2;
const FULL_METER: f32 = 100.0;
const AGENTS: usize = 6;
/// Agents start this far apart in a row: close enough to collide.
const AGENT_SPACING: f32 = 1.5;
const FOOD_ITEMS: usize = 12;
/// Food lies on a ring of this radius around the agents.
const FOOD_RING_RADIUS: f32 = 6.0;
const FOOD_HEIGHT: f32 = 0.35;
/// 300 brain cycles at the default stride of 10.
const TICKS: u32 = 3_000;

fn run(beside: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
    let brain = BrainConfig {
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 9,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(AGENTS as u32, FOOD_ITEMS, &brain, &world);
    kernel.set_brain_beside_vision(beside);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    // Hazard ground in a band to the east of the agents.
    let biomes: Vec<u32> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            let col = cell % BIOME_SIDE;
            if (BIOME_SIDE / 2 + 8..BIOME_SIDE / 2 + 20).contains(&col) {
                BIOME_DANGER
            } else {
                0
            }
        })
        .collect();
    let food: Vec<(f32, f32, f32)> = (0..FOOD_ITEMS)
        .map(|i| {
            let angle = i as f32 / FOOD_ITEMS as f32 * std::f32::consts::TAU;
            (
                FOOD_RING_RADIUS * angle.cos(),
                FOOD_HEIGHT,
                FOOD_RING_RADIUS * angle.sin(),
            )
        })
        .collect();
    kernel.upload_world(
        &heights,
        &biomes,
        &food,
        &vec![false; FOOD_ITEMS],
        &vec![0.0; FOOD_ITEMS],
    );
    let agents: Vec<_> = (0..AGENTS)
        .map(|i| {
            (
                glam::Vec3::new(i as f32 * AGENT_SPACING - 3.0, 1.0, 0.0),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    kernel.reset_agents_seeded(&brain, 31);
    kernel.dispatch_batch(0, TICKS);
    let physics = kernel.read_full_state_blocking()[..AGENTS * PHYS_STRIDE].to_vec();
    let brains = (0..AGENTS)
        .map(|a| kernel.read_agent_state(a as u32).brain_state)
        .collect();
    (physics, brains)
}

#[test]
fn brain_beside_vision_matches_the_serial_cycle() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let (serial_physics, serial_brains) = run(false);
    let (beside_physics, beside_brains) = run(true);
    for (i, (a, b)) in serial_physics.iter().zip(&beside_physics).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "physics slot {} of agent {}: serial {a} vs beside {b}",
            i % PHYS_STRIDE,
            i / PHYS_STRIDE
        );
    }
    for (agent, (a, b)) in serial_brains.iter().zip(&beside_brains).enumerate() {
        let differing = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "agent {agent}: {differing} brain_state values differ"
        );
    }
}
