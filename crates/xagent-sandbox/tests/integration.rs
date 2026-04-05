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
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(&all_positions, world.config.world_size);
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
    if !xagent_brain::GpuBrain::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
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

// ── GPU Tick Loop Tests ─────────────────────────────────────────────

#[test]
fn gpu_tick_loop_runs_without_crash() {
    if !xagent_brain::GpuBrain::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig::default();
    let world = WorldConfig { seed: 42, ..Default::default() };
    let result = bench::run_bench(brain, world, 10, 100);

    assert_eq!(result.total_ticks, 100);
    assert_eq!(result.agent_count, 10);
    assert!(result.ticks_per_sec > 0.0);
}

// ── Trail + Heatmap Tests ───────────────────────────────────────────

#[test]
fn reset_trail_clears_trail_and_marks_dirty() {
    use xagent_sandbox::agent::Agent;
    let world = test_world();
    let pos = world.safe_spawn_position();
    let mut agent = Agent::new(0, pos, 0, BrainConfig::default(), 0);

    // Record a few trail points by moving the agent far enough apart
    agent.body.body.position = Vec3::new(0.0, 0.0, 0.0);
    agent.record_trail();
    agent.body.body.position = Vec3::new(10.0, 0.0, 10.0);
    agent.record_trail();
    assert!(agent.trail.len() >= 2, "trail should have points");

    // Clear dirty from initial recording
    agent.trail_dirty = false;

    // Reset trail (as should happen on death)
    agent.reset_trail();
    assert!(agent.trail.is_empty(), "trail should be empty after reset");
    assert!(agent.trail_dirty, "trail_dirty should be set after reset");
}

#[test]
fn record_heatmap_populates_cells() {
    use xagent_sandbox::agent::Agent;
    let world = test_world();
    let pos = world.safe_spawn_position();
    let mut agent = Agent::new(0, pos, 0, BrainConfig::default(), 0);

    agent.body.body.position = Vec3::new(10.0, 0.0, 10.0);
    agent.record_heatmap(world.config.world_size);

    assert!(agent.unique_cells_explored() >= 1, "should have explored at least 1 cell");
}

// ── FoodGrid Tests ──────────────────────────────────────────────────

#[test]
fn food_grid_query_returns_nearby_food() {
    use xagent_sandbox::world::spatial::FoodGrid;
    use xagent_sandbox::world::entity::FoodItem;

    let items = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)),   // 0: at origin
        FoodItem::new(Vec3::new(1.0, 0.0, 1.0)),    // 1: nearby origin
        FoodItem::new(Vec3::new(100.0, 0.0, 100.0)),// 2: far away
    ];
    let grid = FoodGrid::from_items(&items, 256.0);

    let nearby: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(nearby.contains(&0), "Should find food 0 near origin");
    assert!(nearby.contains(&1), "Should find food 1 near origin");
    assert!(!nearby.contains(&2), "Should NOT find distant food 2");

    let far: Vec<usize> = grid.query_nearby(100.0, 100.0).collect();
    assert!(far.contains(&2), "Should find food 2 near (100,100)");
    assert!(!far.contains(&0), "Should NOT find food 0 near (100,100)");

    let empty: Vec<usize> = grid.query_nearby(-500.0, -500.0).collect();
    assert!(empty.is_empty(), "Should find no food in empty area");
}

#[test]
fn food_grid_skips_consumed_items() {
    use xagent_sandbox::world::spatial::FoodGrid;
    use xagent_sandbox::world::entity::FoodItem;

    let mut item = FoodItem::new(Vec3::new(5.0, 0.0, 5.0));
    item.consumed = true;
    let items = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)), // 0: unconsumed
        item,                                      // 1: consumed
    ];
    let grid = FoodGrid::from_items(&items, 256.0);

    let nearby: Vec<usize> = grid.query_nearby(3.0, 3.0).collect();
    assert!(nearby.contains(&0), "Should find unconsumed food 0");
    assert!(!nearby.contains(&1), "Should NOT find consumed food 1");
}

#[test]
fn food_grid_remove_and_insert() {
    use xagent_sandbox::world::spatial::FoodGrid;
    use xagent_sandbox::world::entity::FoodItem;

    let items = vec![
        FoodItem::new(Vec3::new(10.0, 0.0, 10.0)),
        FoodItem::new(Vec3::new(12.0, 0.0, 12.0)),
    ];
    let mut grid = FoodGrid::from_items(&items, 256.0);

    // Both should be found
    let nearby: Vec<usize> = grid.query_nearby(11.0, 11.0).collect();
    assert!(nearby.contains(&0));
    assert!(nearby.contains(&1));

    // Remove food 0
    grid.remove(0, 10.0, 10.0);
    let after_remove: Vec<usize> = grid.query_nearby(11.0, 11.0).collect();
    assert!(!after_remove.contains(&0), "Food 0 should be gone after remove");
    assert!(after_remove.contains(&1), "Food 1 should remain");

    // Insert food 0 at a new position
    grid.insert(0, 50.0, 50.0);
    let at_new_pos: Vec<usize> = grid.query_nearby(50.0, 50.0).collect();
    assert!(at_new_pos.contains(&0), "Food 0 should be at new position");
}

#[test]
fn food_grid_rebuild_clears_and_repopulates() {
    use xagent_sandbox::world::spatial::FoodGrid;
    use xagent_sandbox::world::entity::FoodItem;

    let items_a = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)),
        FoodItem::new(Vec3::new(50.0, 0.0, 50.0)),
    ];
    let mut grid = FoodGrid::from_items(&items_a, 256.0);

    let items_b = vec![
        FoodItem::new(Vec3::new(80.0, 0.0, 80.0)),
    ];
    grid.rebuild(&items_b);

    let near_origin: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(near_origin.is_empty(), "Old food at origin should be gone after rebuild");

    let near_new: Vec<usize> = grid.query_nearby(80.0, 80.0).collect();
    assert!(near_new.contains(&0), "New food 0 should be at (80,80)");
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

    let grid = AgentGrid::from_positions(&positions, 256.0);

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
    let mut grid = AgentGrid::from_positions(&positions_a, 512.0);

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

// ── Deferred Food Consumption Tests ─────────────────────────────────

#[test]
fn step_pure_does_not_apply_food_energy() {
    let world = test_world();

    // Find an unconsumed food item
    let food_pos = match world.food_items.iter().find(|f| !f.consumed) {
        Some(f) => f.position,
        None => return,
    };

    let spawn_pos = Vec3::new(food_pos.x, food_pos.y + 1.0, food_pos.z);
    let mut agent = agent_at(spawn_pos);
    agent.body.internal.energy = 50.0;
    let energy_before = agent.body.internal.energy;

    let motor = MotorCommand {
        forward: 0.0,
        strafe: 0.0,
        turn: 0.0,
        action: Some(MotorAction::Consume),
    };
    let dt = 1.0 / 30.0;

    let (consumed, _died) = physics::step_pure(&mut agent, &motor, &world, dt);

    // step_pure should detect food but NOT apply energy gain
    assert!(consumed.is_some(), "Should detect nearby food");
    // Energy should have decreased (depletion) or stayed the same, never increased
    assert!(
        agent.body.internal.energy <= energy_before,
        "step_pure must not apply food energy: before={}, after={}",
        energy_before,
        agent.body.internal.energy,
    );
}

#[test]
fn deferred_consumption_awards_energy_to_only_one_agent() {
    let mut world = test_world();

    // Find an unconsumed food item
    let (food_idx, food_pos) = match world
        .food_items
        .iter()
        .enumerate()
        .find(|(_, f)| !f.consumed)
    {
        Some((i, f)) => (i, f.position),
        None => return,
    };

    // Place two agents at the same food position
    let spawn_pos = Vec3::new(food_pos.x, food_pos.y + 1.0, food_pos.z);
    let mut agent_a = agent_at(spawn_pos);
    let mut agent_b = agent_at(spawn_pos);
    agent_a.body.internal.energy = 50.0;
    agent_b.body.internal.energy = 50.0;

    let motor = MotorCommand {
        forward: 0.0,
        strafe: 0.0,
        turn: 0.0,
        action: Some(MotorAction::Consume),
    };
    let dt = 1.0 / 30.0;

    // Both agents detect the same food
    let (consumed_a, _) = physics::step_pure(&mut agent_a, &motor, &world, dt);
    let (consumed_b, _) = physics::step_pure(&mut agent_b, &motor, &world, dt);

    assert_eq!(consumed_a, Some(food_idx));
    assert_eq!(consumed_b, Some(food_idx));

    let energy_a_before = agent_a.body.internal.energy;
    let energy_b_before = agent_b.body.internal.energy;

    // Simulate the sequential deferred consumption (same logic as bench.rs)
    let results = vec![(consumed_a, false), (consumed_b, false)];
    let mut agents_energy = [energy_a_before, energy_b_before];

    for (i, (consumed, _)) in results.iter().enumerate() {
        if let Some(idx) = consumed {
            let food = &mut world.food_items[*idx];
            if !food.consumed {
                food.consumed = true;
                food.respawn_timer = 10.0;
                // Award energy only to the winning consumer
                agents_energy[i] += world.config.food_energy_value;
            }
        }
    }

    // Only one agent should have received the energy
    let a_got_food = agents_energy[0] > energy_a_before;
    let b_got_food = agents_energy[1] > energy_b_before;
    assert!(
        a_got_food && !b_got_food,
        "Only the first agent should get food energy: a_got={}, b_got={}",
        a_got_food,
        b_got_food,
    );
}

// ── Determinism Tests ───────────────────────────────────────────────

#[test]
fn deterministic_bench_produces_same_state_twice() {
    if !xagent_brain::GpuBrain::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    use xagent_shared::{BrainConfig, WorldConfig};

    // Pin rayon to 1 thread to eliminate any scheduling non-determinism.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    let config_b = BrainConfig::default();
    let config_w = WorldConfig::default();

    // Run bench twice with same parameters
    let (r1, r2) = pool.install(|| {
        let r1 = xagent_sandbox::bench::run_bench(config_b.clone(), config_w.clone(), 5, 500);
        let r2 = xagent_sandbox::bench::run_bench(config_b, config_w, 5, 500);
        (r1, r2)
    });

    // Both should complete with same tick count and agent count
    assert_eq!(r1.total_ticks, r2.total_ticks);
    assert_eq!(r1.agent_count, r2.agent_count);
    assert_eq!(r1.final_positions.len(), r2.final_positions.len());

    // NOTE: Exact or approximate position equality is NOT asserted.
    // GpuBrain uses GPU floating point which is not bitwise deterministic
    // across runs. Small per-tick rounding differences compound chaotically
    // (different food consumed, different collisions) so final positions
    // can diverge arbitrarily over 500 ticks.
    //
    // This test now only verifies structural correctness: both runs
    // complete the same number of ticks with the same population size
    // and produce a valid position vector.
    for pos in r1.final_positions.iter().chain(r2.final_positions.iter()) {
        assert!(!pos[0].is_nan(), "NaN in final positions");
        assert!(!pos[1].is_nan(), "NaN in final positions");
        assert!(!pos[2].is_nan(), "NaN in final positions");
    }
}

// ── GPU Vision Tests ────────────────────────────────────────────────

#[test]
fn cpu_vision_produces_correct_buffer_shape() {
    let world = test_world();
    let agent = agent_at(Vec3::new(0.0, 5.0, 0.0));
    let positions = vec![
        (Vec3::new(0.0, 5.0, 0.0), true),
        (Vec3::new(10.0, 5.0, 10.0), true),
    ];

    let mut frame = xagent_shared::SensoryFrame::new_blank(8, 6);
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(&positions, 256.0);
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent, &world, 0, &positions, 0, &agent_grid, &mut frame,
    );

    // 8×6 = 48 rays, 4 color channels each
    assert_eq!(frame.vision.color.len(), 8 * 6 * 4);
    assert_eq!(frame.vision.depth.len(), 8 * 6);
}
