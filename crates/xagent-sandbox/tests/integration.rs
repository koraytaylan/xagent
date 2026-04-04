use glam::Vec3;
use xagent_shared::{BrainConfig, MotorAction, MotorCommand, WorldConfig};

use xagent_sandbox::agent::AgentBody;
use xagent_sandbox::bench;
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

#[test]
fn vision_detects_food_items() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return, // no food in default world (unlikely)
    };

    // Place agent looking directly at the food, close enough to see it
    let to_food = (food.position - Vec3::new(food.position.x - 10.0, food.position.y, food.position.z)).normalize();
    let agent_pos = food.position - to_food * 10.0;
    let spawn_y = world.terrain.height_at(agent_pos.x, agent_pos.z) + 2.0;
    let mut agent = agent_at(Vec3::new(agent_pos.x, spawn_y, agent_pos.z));
    // Face toward the food
    agent.body.facing = to_food;

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    // Check if any pixel in the vision field has the food color (lime green: R≈0.70, G≈0.95)
    let food_green_threshold_g = 0.90;
    let food_green_threshold_r = 0.60;
    let mut found_food_pixel = false;
    let pixels = (frame.vision.width * frame.vision.height) as usize;
    for px in 0..pixels {
        let base = px * 4;
        let r = frame.vision.color[base];
        let g = frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            found_food_pixel = true;
            break;
        }
    }

    // We can't guarantee the food is in the FOV (depends on world layout),
    // so this test checks the mechanism works rather than guaranteeing a hit.
    // Place agent very close and facing directly at food for a reliable check.
    let close_pos = food.position - Vec3::new(3.0, 0.0, 0.0);
    let close_y = world.terrain.height_at(close_pos.x, close_pos.z) + 2.0;
    let mut close_agent = agent_at(Vec3::new(close_pos.x, close_y, close_pos.z));
    close_agent.body.facing = Vec3::X; // face toward food (food is +X from agent)

    let mut close_frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&close_agent, &world, 0, &mut close_frame);

    let mut close_found = false;
    for px in 0..pixels {
        let base = px * 4;
        let r = close_frame.vision.color[base];
        let g = close_frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            close_found = true;
            break;
        }
    }

    assert!(
        found_food_pixel || close_found,
        "At least one vision approach should detect food as lime-green pixels"
    );
}

#[test]
fn vision_with_positions_detects_food_items() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return,
    };

    // Place agent close to food, facing it
    let close_pos = food.position - Vec3::new(3.0, 0.0, 0.0);
    let close_y = world.terrain.height_at(close_pos.x, close_pos.z) + 2.0;
    let mut agent = agent_at(Vec3::new(close_pos.x, close_y, close_pos.z));
    agent.body.facing = Vec3::X;

    // Use the positions-based extraction (the path used during evolution)
    let all_positions: Vec<(Vec3, bool)> = vec![(agent.body.position, true)];
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(&all_positions);
    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent, &world, 0, &all_positions, 0, &agent_grid, &mut frame,
    );

    let food_green_threshold_g = 0.90;
    let food_green_threshold_r = 0.60;
    let pixels = (frame.vision.width * frame.vision.height) as usize;
    let mut found_food = false;
    for px in 0..pixels {
        let base = px * 4;
        let r = frame.vision.color[base];
        let g = frame.vision.color[base + 1];
        if r > food_green_threshold_r && g > food_green_threshold_g {
            found_food = true;
            break;
        }
    }

    assert!(
        found_food,
        "extract_senses_with_positions should detect food as lime-green pixels (the critical evolution bug)"
    );
}

#[test]
fn touch_contacts_populated_near_food() {
    let world = test_world();

    // Find an unconsumed food item
    let food = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f,
        None => return,
    };

    // Place agent within touch range of food (< 3.0 units)
    let agent_pos = Vec3::new(food.position.x + 1.0, food.position.y, food.position.z);
    let spawn_y = world.terrain.height_at(agent_pos.x, agent_pos.z) + 2.0;
    let agent = agent_at(Vec3::new(agent_pos.x, spawn_y, agent_pos.z));

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    let has_food_touch = frame.touch_contacts.iter().any(|c| c.surface_tag == 1 && c.intensity > 0.0);
    assert!(
        has_food_touch,
        "Agent within 3 units of food should have a TOUCH_FOOD contact (tag=1)"
    );
}

// ── Metabolic Tests ────────────────────────────────────────────────────

#[test]
fn metabolic_cost_drains_energy_proportional_to_capacity() {
    use xagent_shared::BrainConfig;

    // Two configs: tiny brain vs large brain
    let small = BrainConfig { memory_capacity: 1, processing_slots: 1, ..BrainConfig::default() };
    let large = BrainConfig { memory_capacity: 512, processing_slots: 32, ..BrainConfig::default() };

    let small_drain = xagent_sandbox::physics::metabolic_drain_per_tick(
        small.memory_capacity,
        small.processing_slots,
    );
    let large_drain = xagent_sandbox::physics::metabolic_drain_per_tick(
        large.memory_capacity,
        large.processing_slots,
    );

    assert!(small_drain > 0.0, "Even small brains have baseline cost");
    assert!(
        large_drain > small_drain * 10.0,
        "Large brain should cost significantly more: small={small_drain}, large={large_drain}",
    );
}

// ── UI Snapshot Tests ─────────────────────────────────────────────────

#[test]
fn evolution_snapshot_default_tree_pane_fraction() {
    let snap = xagent_sandbox::ui::EvolutionSnapshot::default();
    assert!(
        (snap.tree_pane_fraction - 0.25).abs() < f32::EPSILON,
        "tree_pane_fraction should default to 0.25, got {}",
        snap.tree_pane_fraction,
    );
}

// ── Bench Tests ──────────────────────────────────────────────────────

#[test]
fn bench_runner_completes_and_reports_ticks_per_sec() {
    let brain = BrainConfig::default();
    let world = WorldConfig::default();
    let agent_count = 4;
    let total_ticks = 100;

    let result = bench::run_bench(brain, world, agent_count, total_ticks);

    assert_eq!(result.total_ticks, total_ticks, "total_ticks should match requested");
    assert_eq!(result.agent_count, agent_count, "agent_count should match requested");
    assert!(result.elapsed_secs > 0.0, "elapsed_secs should be positive");
    assert!(result.ticks_per_sec > 0.0, "ticks_per_sec should be positive");
    assert!(
        (result.ticks_per_sec - (total_ticks as f64 / result.elapsed_secs)).abs() < 1e-6,
        "ticks_per_sec should equal total_ticks / elapsed_secs"
    );
}

// ── AgentGrid Tests ─────────────────────────────────────────────────

#[test]
fn agent_grid_query_returns_nearby_agents() {
    use xagent_sandbox::world::spatial::AgentGrid;

    let positions: Vec<(Vec3, bool)> = vec![
        (Vec3::new(0.0, 0.0, 0.0), true),   // 0: at origin
        (Vec3::new(1.0, 0.0, 1.0), true),    // 1: nearby origin (same cell)
        (Vec3::new(100.0, 0.0, 100.0), true), // 2: far away
        (Vec3::new(2.0, 0.0, 2.0), false),   // 3: dead, near origin
    ];

    let grid = AgentGrid::from_positions(&positions);

    // Query near origin — should find agents 0 and 1 but not 2 (far) or 3 (dead)
    let nearby: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(nearby.contains(&0), "Should find agent 0 near origin");
    assert!(nearby.contains(&1), "Should find agent 1 near origin");
    assert!(!nearby.contains(&2), "Should NOT find distant agent 2");
    assert!(!nearby.contains(&3), "Should NOT find dead agent 3");

    // Query near the far agent — should find only agent 2
    let far_nearby: Vec<usize> = grid.query_nearby(100.0, 100.0).collect();
    assert!(far_nearby.contains(&2), "Should find agent 2 near (100,100)");
    assert!(!far_nearby.contains(&0), "Should NOT find agent 0 near (100,100)");

    // Query in empty area — should return nothing
    let empty: Vec<usize> = grid.query_nearby(-500.0, -500.0).collect();
    assert!(empty.is_empty(), "Should find no agents in empty area");
}

#[test]
fn agent_grid_rebuild_reuses_allocation() {
    use xagent_sandbox::world::spatial::AgentGrid;

    let positions_a: Vec<(Vec3, bool)> = vec![
        (Vec3::new(0.0, 0.0, 0.0), true),
        (Vec3::new(50.0, 0.0, 50.0), true),
    ];
    let mut grid = AgentGrid::from_positions(&positions_a);

    // After rebuild with different positions, old indices should be gone
    let positions_b: Vec<(Vec3, bool)> = vec![
        (Vec3::new(200.0, 0.0, 200.0), true),
    ];
    grid.rebuild(&positions_b);

    let near_origin: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(near_origin.is_empty(), "Old agent at origin should be gone after rebuild");

    let near_new: Vec<usize> = grid.query_nearby(200.0, 200.0).collect();
    assert!(near_new.contains(&0), "New agent 0 should be at (200,200)");
}

// ── ComputeBackend Tests ────────────────────────────────────────────

#[test]
fn compute_backend_probe_returns_a_tier() {
    let backend = xagent_sandbox::compute_backend::ComputeBackend::probe();
    // Should return at least CpuOptimized (rayon is always available)
    assert!(matches!(
        backend,
        xagent_sandbox::compute_backend::ComputeBackend::CpuOptimized
        | xagent_sandbox::compute_backend::ComputeBackend::GpuAccelerated { .. }
    ));
}

// ── step_pure Parity Tests ──────────────────────────────────────────

#[test]
fn step_pure_matches_step_for_movement() {
    let world_a = test_world();
    let world_b = test_world();

    let spawn_y = world_a.terrain.height_at(0.0, 0.0) + 2.0;
    let pos = Vec3::new(0.0, spawn_y, 0.0);
    let mut agent_a = agent_at(pos);
    let mut agent_b = agent_at(pos);

    let motor = MotorCommand {
        forward: 0.8,
        strafe: -0.3,
        turn: 0.5,
        action: None,
    };
    let dt = 1.0 / 30.0;

    // Run step() on agent_a (with mutable world — no food nearby so no mutation)
    let mut world_a_mut = world_a;
    physics::step(&mut agent_a, &motor, &mut world_a_mut, dt);

    // Run step_pure() on agent_b (with immutable world)
    let (_consumed, _died) = physics::step_pure(&mut agent_b, &motor, &world_b, dt);

    // Positions must match within f32 epsilon
    let eps = 1e-6;
    assert!(
        (agent_a.body.position - agent_b.body.position).length() < eps,
        "Positions diverge: step={:?} vs step_pure={:?}",
        agent_a.body.position,
        agent_b.body.position,
    );
    assert!(
        (agent_a.body.velocity - agent_b.body.velocity).length() < eps,
        "Velocities diverge: step={:?} vs step_pure={:?}",
        agent_a.body.velocity,
        agent_b.body.velocity,
    );
    assert!(
        (agent_a.body.internal.energy - agent_b.body.internal.energy).abs() < eps,
        "Energy diverges: step={} vs step_pure={}",
        agent_a.body.internal.energy,
        agent_b.body.internal.energy,
    );
    assert_eq!(
        agent_a.body.alive,
        agent_b.body.alive,
        "Alive state diverges",
    );
}

// ── GPU Vision Tests ────────────────────────────────────────────────

#[test]
fn gpu_vision_matches_cpu_vision() {
    // Skip if no GPU available
    let backend = xagent_sandbox::compute_backend::ComputeBackend::probe();
    if !backend.has_gpu() {
        println!("Skipping GPU test: no GPU available");
        return;
    }

    // Setup: create world, agent, positions
    let world = test_world();
    let agent = agent_at(Vec3::new(0.0, 5.0, 0.0));
    let positions = vec![
        (Vec3::new(0.0, 5.0, 0.0), true),   // self
        (Vec3::new(10.0, 5.0, 10.0), true),  // another agent
    ];

    // CPU vision
    let mut cpu_frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(&positions);
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent, &world, 0, &positions, 0, &agent_grid, &mut cpu_frame,
    );

    // GPU vision will be tested once the shader is implemented.
    // For now this test documents the expected interface.
    assert!(cpu_frame.vision.color.len() == 8 * 6 * 4);
}
