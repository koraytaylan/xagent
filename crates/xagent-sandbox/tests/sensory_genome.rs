//! The heritable sensory genes reach the GPU senses: each agent's
//! horizontal and vertical angle of view shape its own rays, and its smell
//! strength scales the odour its two nostrils perceive.

use glam::Vec3;
use xagent_brain::buffers::MAX_TOUCH_CONTACTS;
use xagent_brain::GpuKernel;
use xagent_sandbox::agent::senses::sense_scent;
use xagent_sandbox::world::entity::FoodItem;
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer; at the
/// default 256-unit world one biome cell is one unit wide.
const BIOME_SIDE: usize = 256;
/// Half the default world: biome cell `c` spans `[c − half, c − half + 1)`.
const WORLD_HALF: f32 = 128.0;
const FULL_METER: f32 = 100.0;
const AGENT_Y: f32 = 1.0;
const FOOD_Y: f32 = 0.35;
/// Agents stand this far apart along X: beyond the 30-unit reach of sight
/// and smell from each other's probe food, with all five inside the world.
const AGENT_SPACING: f32 = 50.0;
/// First agent's X; the others follow at `AGENT_SPACING`.
const FIRST_AGENT_X: f32 = -100.0;
/// Probe food distance for the horizontal view checks.
const VIEW_FOOD_DISTANCE: f32 = 5.0;
/// Bearing (degrees, to the agent's right) inside a 90° view and outside a
/// 40° one.
const OFF_AXIS_BEARING: f32 = 35.0;
const WIDE_VIEW: f32 = 90.0;
const NARROW_VIEW: f32 = 40.0;
/// Radius of the near biome patch around the vertical-view agents: a wide
/// vertical view's middle-low row hits ground inside it, a narrow one's far
/// outside it.
const NEAR_PATCH_RADIUS: f32 = 6.0;
const TALL_VIEW: f32 = 120.0;
const SHORT_VIEW: f32 = 30.0;
/// Vision row (of 6) just below the horizon, the one that discriminates.
const LOW_ROW: usize = 3;
const VISION_COLUMNS: usize = 8;
/// Probe food distance for the smell checks, to the agent's left.
const SMELL_FOOD_DISTANCE: f32 = 4.0;
/// Food and ground colours written by the vision pass.
const FOOD_COLOR: [f32; 3] = [0.7, 0.95, 0.2];
const NEAR_BIOME_COLOR: [f32; 3] = [0.15, 0.5, 0.1];
const FAR_BIOME_COLOR: [f32; 3] = [0.5, 0.4, 0.2];
const COLOR_TOLERANCE: f32 = 1e-3;
/// Absorbs GPU/CPU float differences in the summed odour.
const SCENT_TOLERANCE: f32 = 1e-4;
/// Index of the left nostril in the non-visual telemetry tail: velocity(3),
/// facing(3), angular(1), energy, integrity, two deltas, then the touch
/// contacts.
const SCENT_SLOT: usize = 3 + 3 + 1 + 4 + MAX_TOUCH_CONTACTS * 4;

/// One probe agent: its genes and the world it is placed in.
struct Probe {
    config: BrainConfig,
    food: Option<Vec3>,
}

fn agent_position(index: usize) -> Vec3 {
    Vec3::new(FIRST_AGENT_X + AGENT_SPACING * index as f32, AGENT_Y, 0.0)
}

/// Food `distance` units from `index`'s agent at `bearing_degrees` to its
/// right (the agent faces +Z, so its right is +X).
fn food_at(index: usize, distance: f32, bearing_degrees: f32) -> Vec3 {
    let bearing = bearing_degrees.to_radians();
    let agent = agent_position(index);
    Vec3::new(
        agent.x + distance * bearing.sin(),
        FOOD_Y,
        agent.z + distance * bearing.cos(),
    )
}

/// Runs one brain tick (the vision pass at its end fills the sensory
/// buffer) and returns each agent's telemetry and the food positions.
fn sense(probes: &[Probe], biomes: &[u32]) -> (Vec<xagent_brain::AgentTelemetry>, Vec<Vec3>) {
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig::default();
    let foods: Vec<Vec3> = probes.iter().filter_map(|probe| probe.food).collect();
    let mut kernel = GpuKernel::new(probes.len() as u32, foods.len().max(1), &brain, &world);
    kernel.reset_agents_seeded(&brain, 9);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let food_positions: Vec<(f32, f32, f32)> = foods.iter().map(|f| (f.x, f.y, f.z)).collect();
    kernel.upload_world(
        &heights,
        biomes,
        &food_positions,
        &vec![false; foods.len()],
        &vec![0.0; foods.len()],
    );
    let agents: Vec<_> = (0..probes.len())
        .map(|i| {
            (
                agent_position(i),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    for (i, probe) in probes.iter().enumerate() {
        kernel.write_agent_heritable_config(i as u32, &probe.config);
    }
    kernel.dispatch_batch(0, 1);
    let telemetry = (0..probes.len())
        .map(|i| kernel.read_agent_telemetry_blocking(i as u32))
        .collect();
    (telemetry, foods)
}

fn color_at(telemetry: &xagent_brain::AgentTelemetry, ray: usize) -> [f32; 3] {
    let base = ray * 4;
    [
        telemetry.vision_color[base],
        telemetry.vision_color[base + 1],
        telemetry.vision_color[base + 2],
    ]
}

fn is_color(color: [f32; 3], expected: [f32; 3]) -> bool {
    color
        .iter()
        .zip(expected)
        .all(|(c, e)| (c - e).abs() < COLOR_TOLERANCE)
}

fn sees_food(telemetry: &xagent_brain::AgentTelemetry) -> bool {
    (0..telemetry.vision_color.len() / 4).any(|ray| is_color(color_at(telemetry, ray), FOOD_COLOR))
}

fn with_view(horizontal: f32, vertical: f32) -> BrainConfig {
    BrainConfig {
        horizontal_fov_degrees: horizontal,
        vertical_fov_degrees: vertical,
        ..BrainConfig::default()
    }
}

#[test]
fn each_agents_angle_of_view_shapes_its_own_rays() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let probes = [
        Probe {
            config: with_view(WIDE_VIEW, WIDE_VIEW),
            food: Some(food_at(0, VIEW_FOOD_DISTANCE, OFF_AXIS_BEARING)),
        },
        Probe {
            config: with_view(NARROW_VIEW, WIDE_VIEW),
            food: Some(food_at(1, VIEW_FOOD_DISTANCE, OFF_AXIS_BEARING)),
        },
        Probe {
            config: with_view(NARROW_VIEW, WIDE_VIEW),
            food: Some(food_at(2, VIEW_FOOD_DISTANCE, 0.0)),
        },
        Probe {
            config: with_view(WIDE_VIEW, TALL_VIEW),
            food: None,
        },
        Probe {
            config: with_view(WIDE_VIEW, SHORT_VIEW),
            food: None,
        },
    ];
    // Near biome (0) in a small patch around the two vertical-view agents,
    // far biome (1) everywhere else.
    let mut biomes = vec![1_u32; BIOME_SIDE * BIOME_SIDE];
    for row in 0..BIOME_SIDE {
        for col in 0..BIOME_SIDE {
            let x = col as f32 - WORLD_HALF + 0.5;
            let z = row as f32 - WORLD_HALF + 0.5;
            let near = [3, 4].iter().any(|&i| {
                let agent = agent_position(i);
                (x - agent.x).hypot(z - agent.z) < NEAR_PATCH_RADIUS
            });
            if near {
                biomes[row * BIOME_SIDE + col] = 0;
            }
        }
    }
    let (telemetry, _) = sense(&probes, &biomes);

    assert!(
        sees_food(&telemetry[0]),
        "a {WIDE_VIEW}° view should see food {OFF_AXIS_BEARING}° off-axis"
    );
    assert!(
        !sees_food(&telemetry[1]),
        "a {NARROW_VIEW}° view should not see food {OFF_AXIS_BEARING}° off-axis"
    );
    assert!(
        sees_food(&telemetry[2]),
        "a {NARROW_VIEW}° view should still see food straight ahead"
    );
    let low_row = |t: &xagent_brain::AgentTelemetry| -> Vec<[f32; 3]> {
        (0..VISION_COLUMNS)
            .map(|col| color_at(t, LOW_ROW * VISION_COLUMNS + col))
            .collect()
    };
    assert!(
        low_row(&telemetry[3])
            .iter()
            .all(|&c| is_color(c, NEAR_BIOME_COLOR)),
        "a {TALL_VIEW}° vertical view's low row should hit the ground close by"
    );
    assert!(
        low_row(&telemetry[4])
            .iter()
            .all(|&c| is_color(c, FAR_BIOME_COLOR)),
        "a {SHORT_VIEW}° vertical view's low row should hit the ground far away"
    );
}

#[test]
fn each_agents_smell_strength_scales_its_two_nostrils() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let with_smell = |smell_strength: f32| BrainConfig {
        smell_strength,
        ..BrainConfig::default()
    };
    // Food to each agent's left (−X).
    let probes = [
        Probe {
            config: with_smell(1.0),
            food: Some(food_at(0, SMELL_FOOD_DISTANCE, -90.0)),
        },
        Probe {
            config: with_smell(0.0),
            food: Some(food_at(1, SMELL_FOOD_DISTANCE, -90.0)),
        },
    ];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    let (telemetry, foods) = sense(&probes, &biomes);

    let scent = |t: &xagent_brain::AgentTelemetry| {
        [
            t.sensory_non_visual[SCENT_SLOT],
            t.sensory_non_visual[SCENT_SLOT + 1],
        ]
    };
    let [left, right] = scent(&telemetry[0]);
    assert!(
        left > right && right > 0.0,
        "food on the left should smell stronger on the left: left {left}, right {right}"
    );
    assert_eq!(
        scent(&telemetry[1]),
        [0.0, 0.0],
        "no nose should smell nothing"
    );

    // The GPU nose agrees with the CPU reference.
    let mut world = WorldState::new(WorldConfig::default());
    world.food_items = foods.iter().map(|&f| FoodItem::new(f)).collect();
    let reference = sense_scent(agent_position(0), Vec3::Z, 1.0, &world);
    for side in 0..2 {
        assert!(
            (scent(&telemetry[0])[side] - reference[side]).abs() < SCENT_TOLERANCE,
            "nostril {side}: GPU {} vs CPU reference {}",
            scent(&telemetry[0])[side],
            reference[side]
        );
    }
}
