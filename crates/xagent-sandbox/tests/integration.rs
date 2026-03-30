use glam::Vec3;
use xagent_shared::{MotorAction, MotorCommand, WorldConfig};

use xagent_sandbox::agent::AgentBody;
use xagent_sandbox::physics;
use xagent_sandbox::world::biome::BiomeType;
use xagent_sandbox::world::terrain::TerrainData;
use xagent_sandbox::world::WorldState;

// ── Helper ─────────────────────────────────────────────────────────────

fn test_world() -> WorldState {
    WorldState::new(WorldConfig::default())
}

fn test_world_with_seed(seed: u64) -> WorldState {
    WorldState::new(WorldConfig {
        seed,
        ..WorldConfig::default()
    })
}

fn agent_at(pos: Vec3) -> AgentBody {
    AgentBody::new(pos)
}

// ── Physics Tests ──────────────────────────────────────────────────────

#[test]
fn motor_forward_moves_agent() {
    let mut world = test_world();
    let spawn_x = 0.0;
    let spawn_z = 0.0;
    let spawn_y = world.terrain.height_at(spawn_x, spawn_z) + 2.0;
    let mut agent = agent_at(Vec3::new(spawn_x, spawn_y, spawn_z));

    let motor = MotorCommand {
        forward: 1.0,
        strafe: 0.0,
        turn: 0.0,
        action: None,
    };

    let start_pos = agent.body.position;
    let dt = 1.0 / 30.0;
    for _ in 0..10 {
        physics::step(&mut agent, &motor, &mut world, dt);
    }

    // Agent should have moved in its facing direction (initially Vec3::Z)
    let displacement = agent.body.position - start_pos;
    let forward_component = displacement.dot(Vec3::Z);
    assert!(
        forward_component > 0.5,
        "Agent should move forward, got displacement: {:?}",
        displacement
    );
}

#[test]
fn motor_turn_rotates_agent() {
    let mut world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let mut agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    let initial_facing = agent.body.facing;

    let motor = MotorCommand {
        forward: 0.0,
        strafe: 0.0,
        turn: 1.0,
        action: None,
    };

    let dt = 1.0 / 30.0;
    for _ in 0..10 {
        physics::step(&mut agent, &motor, &mut world, dt);
    }

    let dot = agent.body.facing.dot(initial_facing);
    assert!(
        dot < 0.99,
        "Facing direction should change after turning, dot product: {}",
        dot
    );
}

#[test]
fn gravity_keeps_agent_on_terrain() {
    let mut world = test_world();
    let ground = world.terrain.height_at(0.0, 0.0);
    // Place agent well above terrain
    let mut agent = agent_at(Vec3::new(0.0, ground + 50.0, 0.0));

    let motor = MotorCommand::idle();
    let dt = 1.0 / 30.0;

    // Step enough for gravity to bring agent down
    for _ in 0..300 {
        physics::step(&mut agent, &motor, &mut world, dt);
    }

    let terrain_height = world.terrain.height_at(agent.body.position.x, agent.body.position.z);
    let diff = agent.body.position.y - terrain_height;
    assert!(
        diff < 2.0,
        "Agent should be near terrain surface, but is {} above",
        diff
    );
}

#[test]
fn nan_motor_command_is_sanitized() {
    let mut world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let mut agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    let motor = MotorCommand {
        forward: f32::NAN,
        strafe: f32::INFINITY,
        turn: f32::NEG_INFINITY,
        action: None,
    };

    let dt = 1.0 / 30.0;
    // Should not panic and position should remain finite
    for _ in 0..10 {
        physics::step(&mut agent, &motor, &mut world, dt);
    }

    assert!(
        agent.body.position.x.is_finite(),
        "Position X should be finite after NaN motor"
    );
    assert!(
        agent.body.position.y.is_finite(),
        "Position Y should be finite after NaN motor"
    );
    assert!(
        agent.body.position.z.is_finite(),
        "Position Z should be finite after NaN motor"
    );
}

// ── Agent Tests ────────────────────────────────────────────────────────

#[test]
fn energy_depletes_over_time() {
    let mut world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let mut agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    let initial_energy = agent.body.internal.energy;
    let motor = MotorCommand::idle();
    let dt = 1.0 / 30.0;

    for _ in 0..100 {
        physics::step(&mut agent, &motor, &mut world, dt);
    }

    assert!(
        agent.body.internal.energy < initial_energy,
        "Energy should decrease over time: initial={}, current={}",
        initial_energy,
        agent.body.internal.energy
    );
}

#[test]
fn agent_dies_at_zero_energy() {
    let mut world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let mut agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    // Force energy to near-zero
    agent.body.internal.energy = 0.001;

    let motor = MotorCommand {
        forward: 1.0,
        strafe: 1.0,
        turn: 0.0,
        action: None,
    };
    let dt = 1.0 / 30.0;

    // Step until dead (should be very quick)
    for _ in 0..100 {
        physics::step(&mut agent, &motor, &mut world, dt);
        if !agent.body.alive {
            break;
        }
    }

    assert!(
        !agent.body.alive,
        "Agent should die when energy is depleted"
    );
}

#[test]
fn consume_near_food_restores_energy() {
    let mut world = test_world();

    // Find a food item position and place the agent near it
    let food_pos = if let Some(food) = world.food_items.iter().find(|f| !f.consumed) {
        food.position
    } else {
        // If no food (unlikely with default config), skip test meaningfully
        return;
    };

    let spawn_pos = Vec3::new(food_pos.x, food_pos.y + 1.0, food_pos.z);
    let mut agent = agent_at(spawn_pos);

    // Drain some energy first
    agent.body.internal.energy = 50.0;
    let energy_before = agent.body.internal.energy;

    let motor = MotorCommand {
        forward: 0.0,
        strafe: 0.0,
        turn: 0.0,
        action: Some(MotorAction::Consume),
    };
    let dt = 1.0 / 30.0;
    physics::step(&mut agent, &motor, &mut world, dt);

    // Energy should increase after consuming food (minus depletion cost)
    // food_energy_value is 20.0 by default, depletion is tiny per tick
    assert!(
        agent.body.internal.energy > energy_before,
        "Energy should increase after consuming food: before={}, after={}",
        energy_before,
        agent.body.internal.energy
    );
}

// ── World / Terrain Tests ──────────────────────────────────────────────

#[test]
fn terrain_height_is_deterministic() {
    let world_a = test_world_with_seed(12345);
    let world_b = test_world_with_seed(12345);

    let positions = [(0.0, 0.0), (10.0, 20.0), (-50.0, 30.0), (100.0, -100.0)];

    for (x, z) in positions {
        let ha = world_a.terrain.height_at(x, z);
        let hb = world_b.terrain.height_at(x, z);
        assert!(
            (ha - hb).abs() < 1e-6,
            "Terrain height should be deterministic at ({}, {}): {} vs {}",
            x,
            z,
            ha,
            hb
        );
    }
}

#[test]
fn biome_query_returns_valid_type() {
    let world = test_world();

    let positions = [
        (0.0, 0.0),
        (50.0, 50.0),
        (-100.0, 100.0),
        (120.0, -120.0),
        (-50.0, -50.0),
    ];

    for (x, z) in positions {
        let biome = world.biome_map.biome_at(x, z);
        // Just verify it returns a valid BiomeType (would panic at compile time if not)
        match biome {
            BiomeType::FoodRich | BiomeType::Barren | BiomeType::Danger => {}
        }
    }
}

#[test]
fn terrain_height_interpolation_is_smooth() {
    let world = test_world();

    // Walk along a line and check that adjacent heights aren't too different
    let step = 0.5;
    let mut prev_h = world.terrain.height_at(0.0, 0.0);

    for i in 1..100 {
        let x = i as f32 * step;
        let h = world.terrain.height_at(x, 0.0);
        let diff = (h - prev_h).abs();
        assert!(
            diff < 5.0,
            "Terrain height should be smooth: diff={} at x={}",
            diff,
            x
        );
        prev_h = h;
    }
}

#[test]
#[should_panic(expected = "Terrain subdivisions must be >= 2")]
fn terrain_rejects_zero_subdivisions() {
    TerrainData::generate(256.0, 0, 42);
}

#[test]
#[should_panic(expected = "Terrain subdivisions must be >= 2")]
fn terrain_rejects_one_subdivision() {
    TerrainData::generate(256.0, 1, 42);
}

// ── Sensory Tests ──────────────────────────────────────────────────────

#[test]
fn sensory_frame_has_correct_dimensions() {
    let world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    assert_eq!(frame.vision.width, 8, "Visual field width should be 8");
    assert_eq!(frame.vision.height, 6, "Visual field height should be 6");
    assert_eq!(
        frame.vision.color.len(),
        (8 * 6 * 4) as usize,
        "Color buffer should have width*height*4 elements"
    );
    assert_eq!(
        frame.vision.depth.len(),
        (8 * 6) as usize,
        "Depth buffer should have width*height elements"
    );
}

#[test]
fn interoception_matches_body_state() {
    let world = test_world();
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 2.0;
    let mut agent = agent_at(Vec3::new(0.0, spawn_y, 0.0));

    // Set specific energy/integrity values
    agent.body.internal.energy = 75.0;
    agent.body.internal.integrity = 60.0;

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    let expected_energy = agent.body.internal.energy_signal();
    let expected_integrity = agent.body.internal.integrity_signal();

    assert!(
        (frame.energy_signal - expected_energy).abs() < 1e-6,
        "energy_signal should match InternalState.energy_signal(): {} vs {}",
        frame.energy_signal,
        expected_energy
    );
    assert!(
        (frame.integrity_signal - expected_integrity).abs() < 1e-6,
        "integrity_signal should match InternalState.integrity_signal(): {} vs {}",
        frame.integrity_signal,
        expected_integrity
    );
}
