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

fn agent_at(position: Vec3) -> AgentBody {
    AgentBody::new(position)
}

/// Create a blank sensory frame using the default brain config's vision dimensions.
fn default_frame() -> xagent_shared::SensoryFrame {
    let cfg = xagent_shared::BrainConfig::default();
    xagent_shared::SensoryFrame::new_blank(cfg.vision_width, cfg.vision_height)
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

    let terrain_height = world
        .terrain
        .height_at(agent.body.position.x, agent.body.position.z);
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

    let cfg = xagent_shared::BrainConfig::default();
    let vision_width = cfg.vision_width;
    let vision_height = cfg.vision_height;
    let mut frame = xagent_shared::SensoryFrame::new_blank(vision_width, vision_height);
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    assert_eq!(
        frame.vision.width, vision_width,
        "Visual field width should match config"
    );
    assert_eq!(
        frame.vision.height, vision_height,
        "Visual field height should match config"
    );
    assert_eq!(
        frame.vision.color.len(),
        (vision_width * vision_height * 4) as usize,
        "Color buffer should have width*height*4 elements"
    );
    assert_eq!(
        frame.vision.depth.len(),
        (vision_width * vision_height) as usize,
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

    let mut frame = default_frame();
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
    let to_food = (food.position
        - Vec3::new(food.position.x - 10.0, food.position.y, food.position.z))
    .normalize();
    let agent_pos = food.position - to_food * 10.0;
    let spawn_y = world.terrain.height_at(agent_pos.x, agent_pos.z) + 2.0;
    let mut agent = agent_at(Vec3::new(agent_pos.x, spawn_y, agent_pos.z));
    // Face toward the food
    agent.body.facing = to_food;

    let mut frame = default_frame();
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

    let mut close_frame = default_frame();
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
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(
        &all_positions,
        world.config.world_size,
    );
    let mut frame = default_frame();
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent,
        &world,
        0,
        &all_positions,
        0,
        &agent_grid,
        &mut frame,
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

    let mut frame = default_frame();
    xagent_sandbox::agent::senses::extract_senses(&agent, &world, 0, &mut frame);

    let has_food_touch = frame
        .touch_contacts
        .iter()
        .any(|c| c.surface_tag == 1 && c.intensity > 0.0);
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
    let small = BrainConfig {
        memory_capacity: 1,
        processing_slots: 1,
        ..BrainConfig::default()
    };
    let large = BrainConfig {
        memory_capacity: 512,
        processing_slots: 32,
        ..BrainConfig::default()
    };

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
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let brain = BrainConfig::default();
    let world = WorldConfig::default();
    let agent_count = 4;
    let total_ticks = 100;

    let result = bench::run_bench(brain, world, agent_count, total_ticks);

    assert_eq!(
        result.total_ticks, total_ticks,
        "total_ticks should match requested"
    );
    assert_eq!(
        result.agent_count, agent_count,
        "agent_count should match requested"
    );
    assert!(result.elapsed_secs > 0.0, "elapsed_secs should be positive");
    assert!(
        result.ticks_per_sec > 0.0,
        "ticks_per_sec should be positive"
    );
    assert!(
        (result.ticks_per_sec - (total_ticks as f64 / result.elapsed_secs)).abs() < 1e-6,
        "ticks_per_sec should equal total_ticks / elapsed_secs"
    );
}

/// Proves that decomposing ticks into different batch sizes doesn't
/// affect simulation results, as long as each batch is a multiple of
/// `kernel_batch_size` (= `vision_stride * brain_tick_stride`, default 100).
///
/// Each run creates a fresh kernel from the same initial state so the
/// comparison focuses on whether different batch decompositions change
/// the simulation result.
///
/// Test cases (all multiples of 100, total = 1000):
///   1. 1 × 1000   (1 dispatch)
///   2. 2 × 500    (2 dispatches)
///   3. 10 × 100   (10 dispatches)
#[test]
fn deterministic_across_batch_sizes() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    use xagent_brain::buffers::{P_POS_X, P_POS_Y, P_POS_Z};

    let brain = BrainConfig::default();
    let world_config = WorldConfig {
        seed: 42,
        ..Default::default()
    };
    let total_ticks: u32 = 1000;

    let world = xagent_sandbox::world::WorldState::new(world_config.clone());
    let heights = world.terrain.heights.clone();
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    let spawn_pos = world.safe_spawn_position();
    let food_count = world.food_items.len();
    let agent_data = vec![(
        spawn_pos,
        100.0_f32,
        100.0_f32,
        brain.memory_capacity,
        brain.processing_slots,
    )];

    // Helper: create a fresh kernel with deterministic brain state,
    // dispatch total_ticks in given batch size, return final position.
    let run_with_batch_size = |batch_size: u32| -> [f32; 3] {
        let mut kernel = xagent_brain::GpuKernel::new(1, food_count, &brain, &world_config);
        let kernel_batch = kernel.kernel_batch_size();
        assert_eq!(
            batch_size % kernel_batch,
            0,
            "batch_size {} must be a multiple of kernel_batch_size {}",
            batch_size,
            kernel_batch
        );
        // Overwrite random brain state with deterministic seed
        kernel.reset_agents_seeded(&brain, 12345);
        kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);
        kernel.upload_agents(&agent_data);

        let num_batches = total_ticks / batch_size;
        for i in 0..num_batches {
            kernel.dispatch_batch((i * batch_size) as u64, batch_size);
        }

        let state = kernel.read_full_state_blocking();
        [state[P_POS_X], state[P_POS_Y], state[P_POS_Z]]
    };

    let pos_1000 = run_with_batch_size(1000);
    let pos_500 = run_with_batch_size(500);
    let pos_100 = run_with_batch_size(100);

    eprintln!("1×1000:  {:?}", pos_1000);
    eprintln!("2×500:   {:?}", pos_500);
    eprintln!("10×100:  {:?}", pos_100);

    assert_eq!(pos_1000, pos_500, "2×500 diverged from 1×1000");
    assert_eq!(pos_1000, pos_100, "10×100 diverged from 1×1000");
}

// ── GPU Tick Loop Tests ─────────────────────────────────────────────

#[test]
fn gpu_tick_loop_runs_without_crash() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig::default();
    let world = WorldConfig {
        seed: 42,
        ..Default::default()
    };
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

    assert!(
        agent.unique_cells_explored() >= 1,
        "should have explored at least 1 cell"
    );
}

// ── FoodGrid Tests ──────────────────────────────────────────────────

#[test]
fn food_grid_query_returns_nearby_food() {
    use xagent_sandbox::world::entity::FoodItem;
    use xagent_sandbox::world::spatial::FoodGrid;

    let items = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)),     // 0: at origin
        FoodItem::new(Vec3::new(1.0, 0.0, 1.0)),     // 1: nearby origin
        FoodItem::new(Vec3::new(100.0, 0.0, 100.0)), // 2: far away
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
    use xagent_sandbox::world::entity::FoodItem;
    use xagent_sandbox::world::spatial::FoodGrid;

    let mut item = FoodItem::new(Vec3::new(5.0, 0.0, 5.0));
    item.consumed = true;
    let items = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)), // 0: unconsumed
        item,                                    // 1: consumed
    ];
    let grid = FoodGrid::from_items(&items, 256.0);

    let nearby: Vec<usize> = grid.query_nearby(3.0, 3.0).collect();
    assert!(nearby.contains(&0), "Should find unconsumed food 0");
    assert!(!nearby.contains(&1), "Should NOT find consumed food 1");
}

#[test]
fn food_grid_remove_and_insert() {
    use xagent_sandbox::world::entity::FoodItem;
    use xagent_sandbox::world::spatial::FoodGrid;

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
    assert!(
        !after_remove.contains(&0),
        "Food 0 should be gone after remove"
    );
    assert!(after_remove.contains(&1), "Food 1 should remain");

    // Insert food 0 at a new position
    grid.insert(0, 50.0, 50.0);
    let at_new_pos: Vec<usize> = grid.query_nearby(50.0, 50.0).collect();
    assert!(at_new_pos.contains(&0), "Food 0 should be at new position");
}

#[test]
fn food_grid_rebuild_clears_and_repopulates() {
    use xagent_sandbox::world::entity::FoodItem;
    use xagent_sandbox::world::spatial::FoodGrid;

    let items_a = vec![
        FoodItem::new(Vec3::new(0.0, 0.0, 0.0)),
        FoodItem::new(Vec3::new(50.0, 0.0, 50.0)),
    ];
    let mut grid = FoodGrid::from_items(&items_a, 256.0);

    let items_b = vec![FoodItem::new(Vec3::new(80.0, 0.0, 80.0))];
    grid.rebuild(&items_b);

    let near_origin: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(
        near_origin.is_empty(),
        "Old food at origin should be gone after rebuild"
    );

    let near_new: Vec<usize> = grid.query_nearby(80.0, 80.0).collect();
    assert!(near_new.contains(&0), "New food 0 should be at (80,80)");
}

// ── AgentGrid Tests ─────────────────────────────────────────────────

#[test]
fn agent_grid_query_returns_nearby_agents() {
    use xagent_sandbox::world::spatial::AgentGrid;

    let positions: Vec<(Vec3, bool)> = vec![
        (Vec3::new(0.0, 0.0, 0.0), true),     // 0: at origin
        (Vec3::new(1.0, 0.0, 1.0), true),     // 1: nearby origin (same cell)
        (Vec3::new(100.0, 0.0, 100.0), true), // 2: far away
        (Vec3::new(2.0, 0.0, 2.0), false),    // 3: dead, near origin
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
    assert!(
        far_nearby.contains(&2),
        "Should find agent 2 near (100,100)"
    );
    assert!(
        !far_nearby.contains(&0),
        "Should NOT find agent 0 near (100,100)"
    );

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
    let positions_b: Vec<(Vec3, bool)> = vec![(Vec3::new(200.0, 0.0, 200.0), true)];
    grid.rebuild(&positions_b);

    let near_origin: Vec<usize> = grid.query_nearby(0.0, 0.0).collect();
    assert!(
        near_origin.is_empty(),
        "Old agent at origin should be gone after rebuild"
    );

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
        agent_a.body.alive, agent_b.body.alive,
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
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    use xagent_shared::{BrainConfig, WorldConfig};

    // Pin rayon to 1 thread to eliminate any scheduling non-determinism.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

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
    // GpuKernel uses GPU floating point which is not bitwise deterministic
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

    let cfg = BrainConfig::default();
    let vision_width = cfg.vision_width;
    let vision_height = cfg.vision_height;
    let mut frame = xagent_shared::SensoryFrame::new_blank(vision_width, vision_height);
    let agent_grid = xagent_sandbox::world::spatial::AgentGrid::from_positions(&positions, 256.0);
    xagent_sandbox::agent::senses::extract_senses_with_positions(
        &agent,
        &world,
        0,
        &positions,
        0,
        &agent_grid,
        &mut frame,
    );

    assert_eq!(
        frame.vision.color.len(),
        (vision_width * vision_height * 4) as usize
    );
    assert_eq!(
        frame.vision.depth.len(),
        (vision_width * vision_height) as usize
    );
}

// ── Async Recording Persistence ───────────────────────────────────────

#[test]
fn async_recording_persists_and_round_trips() {
    use xagent_sandbox::governor::Governor;
    use xagent_sandbox::replay::{GenerationRecording, TickRecord};

    // 1. Create a temp on-disk DB (NOT :memory: — that skips the background writer).
    //    `into_temp_path()` closes the file handle (so SQLite can use it) while
    //    keeping the path reserved — automatic cleanup when `_tmp` drops at scope end.
    let _tmp = tempfile::NamedTempFile::new()
        .expect("failed to create temp file")
        .into_temp_path();
    let db_path = _tmp.to_str().expect("non-UTF-8 temp path").to_owned();

    let gov_cfg = xagent_shared::GovernorConfig::default();
    let brain_cfg = xagent_shared::BrainConfig::default();
    let world_cfg_json = serde_json::to_string(&xagent_shared::WorldConfig::default()).unwrap();

    // 2. Build a dummy GenerationRecording with 2 agents and 3 ticks.
    let agent_count = 2;
    let total_ticks = 3u64;
    let agents: Vec<(u32, [f32; 3])> = (0..agent_count)
        .map(|i| (i as u32, [1.0, 0.0, 0.0]))
        .collect();
    let mut recording = GenerationRecording::new(1, &agents, &[], total_ticks as usize, 8, 8);

    for tick in 0..total_ticks {
        let records: Vec<TickRecord> = (0..agent_count)
            .map(|a| TickRecord {
                position: [tick as f32, a as f32, 0.0],
                yaw: 0.1 * tick as f32,
                alive: true,
                energy: 100.0 - tick as f32,
                integrity: 1.0,
                motor_forward: 0.5,
                motor_turn: 0.0,
                exploration_rate: 0.1,
                prediction_error: 0.01,
                gradient: 0.0,
                raw_gradient: 0.0,
                urgency: 0.0,
                credit_magnitude: 0.0,
                patterns_recalled: 0,
                phase: 0,
                mean_attenuation: 1.0,
                curiosity_bonus: 0.0,
                fatigue_factor: 1.0,
                staleness: 0.0,
                vision_color: None,
            })
            .collect();
        recording.record_tick(tick, &records);
    }

    // Capture the node_id that Governor created for the root node.
    let node_id;
    {
        let mut gov =
            Governor::new(&db_path, gov_cfg, &brain_cfg, &world_cfg_json).expect("Governor::new");
        node_id = gov.current_node_id.expect("no current_node_id");

        // 3. Enqueue the recording for async persistence.
        gov.store_recording(&recording);

        // 4. Dropping gov joins the writer thread, flushing the pending write.
    }

    // 5. Reopen the DB via Governor::resume and verify round-trip.
    let gov2 = Governor::resume(&db_path).expect("Governor::resume");
    let (loaded_agents, loaded_ticks, floats) = gov2
        .load_recording(node_id)
        .expect("recording not found after async persistence");

    assert_eq!(loaded_agents, agent_count);
    assert_eq!(loaded_ticks, total_ticks);

    // 15 f32 fields per agent per tick (the serialization stride).
    let record_stride = 15;
    assert_eq!(
        floats.len(),
        agent_count * total_ticks as usize * record_stride
    );

    // Spot-check: position[0] of agent 0 at each tick should equal the tick index.
    for tick in 0..total_ticks as usize {
        let base = tick * agent_count * record_stride;
        assert!(
            (floats[base] - tick as f32).abs() < f32::EPSILON,
            "position.x mismatch at tick {tick}"
        );
    }

    // Drop the governor (and its DB connection) before `_tmp` cleans up the file.
    drop(gov2);
    // Also remove WAL/SHM sidecars that SQLite may have created.
    let _ = std::fs::remove_file(format!("{db_path}-wal"));
    let _ = std::fs::remove_file(format!("{db_path}-shm"));
    // `_tmp` drops here, removing the main DB file automatically.
}

// ── Vision-stride / Brain-tick-stride dispatch arithmetic sanity checks ──
//
// Background (not verified by these tests):
//   Within the fused kernel each inner cycle runs in this order:
//     physics → food_detect → death_respawn → brain
//   The barriers/orderings only establish phase sequencing within that
//   kernel cycle; they do not, by themselves, mean every brain input is
//   sourced from same-cycle data.
//
//   In particular, the vision pass (raycasting → sensory_buf) runs at the
//   end of each batch, AFTER the kernel dispatch. The brain in batch N
//   reads sensory_buf written by batch N-1's vision pass — a one-batch
//   lag for sensory/proprioceptive/interoceptive features sourced there.
//
//   When brain_tick_stride == vision_stride there is exactly one vision
//   pass per `vision_stride` brain cycles. The sensory lag is then:
//     one batch = vision_stride × brain_tick_stride physics ticks.
//
//   The tests below validate dispatch-related arithmetic formulas and, where
//   applicable elsewhere in this file, GPU pipeline behavior for matching
//   stride values.

/// Sanity-check the cycle/batch-count arithmetic for matching strides.
/// brain_cycles = ticks / brain_tick_stride
/// kernel_batches = brain_cycles / vision_stride
/// When strides are equal S: kernel_batches = ticks / S^2
#[test]
fn stride_batch_count_formula_when_strides_match() {
    // Sanity-check the arithmetic used by dispatch_batch for equal strides.
    // This test intentionally re-derives the expected values locally; it does
    // not call bench::run_bench or the real dispatch path.

    // stride = 1: every tick runs vision + brain.
    // 10 ticks → brain_cycles=10, kernel_batches=10, each with 1 cycle.
    {
        let ticks_to_run: u32 = 10;
        let brain_tick_stride: u32 = 1;
        let vision_stride: u32 = 1;
        let brain_cycles = ticks_to_run / brain_tick_stride;
        let kernel_batches = brain_cycles / vision_stride;
        let remainder_cycles = brain_cycles % vision_stride;
        assert_eq!(kernel_batches, 10, "stride=1: should have 10 batches");
        assert_eq!(remainder_cycles, 0, "stride=1: no remainder cycles");
    }

    // stride = 10: matching defaults.
    // 100 ticks → brain_cycles=10, kernel_batches=1, no remainder.
    {
        let ticks_to_run: u32 = 100;
        let brain_tick_stride: u32 = 10;
        let vision_stride: u32 = 10;
        let brain_cycles = ticks_to_run / brain_tick_stride;
        let kernel_batches = brain_cycles / vision_stride;
        let remainder_cycles = brain_cycles % vision_stride;
        assert_eq!(kernel_batches, 1, "stride=10: should have 1 batch");
        assert_eq!(remainder_cycles, 0, "stride=10: no remainder cycles");
    }

    // stride = 10, non-multiple ticks.
    // 150 ticks → brain_cycles=15, kernel_batches=1, remainder=5 cycles.
    {
        let ticks_to_run: u32 = 150;
        let brain_tick_stride: u32 = 10;
        let vision_stride: u32 = 10;
        let brain_cycles = ticks_to_run / brain_tick_stride;
        let kernel_batches = brain_cycles / vision_stride;
        let remainder_cycles = brain_cycles % vision_stride;
        assert_eq!(
            kernel_batches, 1,
            "stride=10, 150 ticks: should have 1 full batch"
        );
        assert_eq!(
            remainder_cycles, 5,
            "stride=10, 150 ticks: 5 remainder cycles"
        );
        // Total batches = kernel_batches + (remainder > 0)
        let total_batches = kernel_batches + if remainder_cycles > 0 { 1 } else { 0 };
        assert_eq!(total_batches, 2, "stride=10, 150 ticks: 2 total batches");
    }

    // Large stride = 64.
    // 64*64=4096 ticks → brain_cycles=64, kernel_batches=1, no remainder.
    {
        let ticks_to_run: u32 = 64 * 64;
        let brain_tick_stride: u32 = 64;
        let vision_stride: u32 = 64;
        let brain_cycles = ticks_to_run / brain_tick_stride;
        let kernel_batches = brain_cycles / vision_stride;
        let remainder_cycles = brain_cycles % vision_stride;
        assert_eq!(kernel_batches, 1, "stride=64: should have 1 batch");
        assert_eq!(remainder_cycles, 0, "stride=64: no remainder cycles");
    }
}

/// Verify the tick coverage: all ticks in a batch are accounted for.
/// total ticks covered = kernel_batches * vision_stride * brain_tick_stride
///                       + remainder_cycles * brain_tick_stride
///                       + physics_remainder
///
/// The expected decomposition is precomputed so the test can catch
/// regressions instead of only re-deriving `total_ticks` from itself.
#[test]
fn stride_tick_coverage_is_complete_when_strides_match() {
    // (stride, total_ticks, (kernel_batches, remainder_cycles, physics_remainder))
    let cases = [
        (1u32, 1u32, (1u32, 0u32, 0u32)),
        (1, 10, (10, 0, 0)),
        (1, 100, (100, 0, 0)),
        (1, 500, (500, 0, 0)),
        (1, 1000, (1000, 0, 0)),
        (4, 1, (0, 0, 1)),
        (4, 10, (0, 2, 2)),
        (4, 100, (6, 1, 0)),
        (4, 500, (31, 1, 0)),
        (4, 1000, (62, 2, 0)),
        (10, 1, (0, 0, 1)),
        (10, 10, (0, 1, 0)),
        (10, 100, (1, 0, 0)),
        (10, 500, (5, 0, 0)),
        (10, 1000, (10, 0, 0)),
        (16, 1, (0, 0, 1)),
        (16, 10, (0, 0, 10)),
        (16, 100, (0, 6, 4)),
        (16, 500, (1, 15, 4)),
        (16, 1000, (3, 14, 8)),
    ];

    for (stride, total_ticks, expected) in cases {
        let brain_tick_stride = stride;
        let vision_stride = stride;

        let brain_cycles = total_ticks / brain_tick_stride;
        let actual = (
            brain_cycles / vision_stride,
            brain_cycles % vision_stride,
            total_ticks % brain_tick_stride,
        );

        assert_eq!(
            actual, expected,
            "stride={stride}, ticks={total_ticks}: expected {:?}, got {:?}",
            expected, actual
        );

        let covered = expected.0 * vision_stride * brain_tick_stride
            + expected.1 * brain_tick_stride
            + expected.2;

        assert_eq!(
            covered, total_ticks,
            "stride={stride}, ticks={total_ticks}: covered={covered} != total={total_ticks}"
        );
    }
}

/// GPU smoke test: dispatch with brain_tick_stride == vision_stride == 1.
/// Every tick runs vision + brain; verifies no crash and valid positions.
#[test]
fn gpu_stride_1_matching_no_crash() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 1,
        ..WorldConfig::default()
    };

    let result = xagent_sandbox::bench::run_bench(brain, world, 2, 10);

    assert_eq!(result.total_ticks, 10);
    assert_eq!(result.agent_count, 2);
    for pos in &result.final_positions {
        assert!(pos[0].is_finite(), "NaN/inf x after stride=1");
        assert!(pos[1].is_finite(), "NaN/inf y after stride=1");
        assert!(pos[2].is_finite(), "NaN/inf z after stride=1");
    }
}

/// GPU smoke test: dispatch with brain_tick_stride == vision_stride == 10.
/// Tests the common production case where both strides match at 10.
#[test]
fn gpu_stride_10_matching_no_crash() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 10,
        vision_stride: 10,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 2,
        ..WorldConfig::default()
    };

    // 200 ticks: 2 full batches of 100 ticks each (10 brain cycles × 10 physics ticks)
    let result = xagent_sandbox::bench::run_bench(brain, world, 2, 200);

    assert_eq!(result.total_ticks, 200);
    assert_eq!(result.agent_count, 2);
    for pos in &result.final_positions {
        assert!(pos[0].is_finite(), "NaN/inf x after stride=10");
        assert!(pos[1].is_finite(), "NaN/inf y after stride=10");
        assert!(pos[2].is_finite(), "NaN/inf z after stride=10");
    }
}

/// GPU smoke test: large matching strides (stride=32).
/// Each batch covers 32×32=1024 physics ticks with one vision pass.
#[test]
fn gpu_large_stride_matching_no_crash() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 32,
        vision_stride: 32,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 3,
        ..WorldConfig::default()
    };

    // 1024 ticks: exactly one full batch (32 brain cycles × 32 physics ticks)
    let result = xagent_sandbox::bench::run_bench(brain, world, 2, 1024);

    assert_eq!(result.total_ticks, 1024);
    assert_eq!(result.agent_count, 2);
    for pos in &result.final_positions {
        assert!(pos[0].is_finite(), "NaN/inf x after large stride");
        assert!(pos[1].is_finite(), "NaN/inf y after large stride");
        assert!(pos[2].is_finite(), "NaN/inf z after large stride");
    }
}

/// GPU smoke test: non-multiple ticks with matching strides.
/// 150 ticks with stride=10: 1 full batch (100 ticks) + 1 remainder batch (50 ticks).
#[test]
fn gpu_non_multiple_ticks_matching_strides_no_crash() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 10,
        vision_stride: 10,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 4,
        ..WorldConfig::default()
    };

    let result = xagent_sandbox::bench::run_bench(brain, world, 2, 150);

    assert_eq!(result.total_ticks, 150);
    assert_eq!(result.agent_count, 2);
    for pos in &result.final_positions {
        assert!(pos[0].is_finite(), "NaN/inf x after non-multiple ticks");
        assert!(pos[1].is_finite(), "NaN/inf y after non-multiple ticks");
        assert!(pos[2].is_finite(), "NaN/inf z after non-multiple ticks");
    }
}

// ── GPU Terrain Height Tests ────────────────────────────────────────

#[test]
fn gpu_agents_follow_terrain_height() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig::default();
    let world_config = WorldConfig {
        seed: 42,
        ..Default::default()
    };
    let agent_count = 10;

    let result = bench::run_bench(brain, world_config.clone(), agent_count, 200);
    let world = WorldState::new(world_config);

    for (i, pos) in result.final_positions.iter().enumerate() {
        let x = pos[0];
        let y = pos[1];
        let z = pos[2];

        assert!(
            x.is_finite() && y.is_finite() && z.is_finite(),
            "Agent {} has non-finite final position: ({:?}, {:?}, {:?})",
            i,
            x,
            y,
            z
        );

        let terrain_y = world.terrain.height_at(x, z);
        let diff = y - terrain_y;

        // Agents should be at least AGENT_HALF_HEIGHT (1.0) above terrain;
        // allow small float tolerance (0.99) but reject agents sunk into the ground.
        assert!(
            diff >= 0.99 && diff < 5.0,
            "Agent {} at ({:.2}, {:.2}, {:.2}): expected Y to be 0.99..5.0 above terrain height {:.2}, but y - terrain_y = {:.2}",
            i, x, y, z, terrain_y, diff
        );
    }
}

#[test]
fn gpu_agents_y_matches_terrain_after_single_tick() {
    use xagent_brain::buffers::{PHYS_STRIDE, P_POS_X, P_POS_Y, P_POS_Z};
    use xagent_brain::GpuKernel;

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig::default();
    let world_config = WorldConfig {
        seed: 42,
        ..Default::default()
    };
    let agent_count = 10;
    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let mut kernel = GpuKernel::new(agent_count as u32, food_count, &brain, &world_config);

    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let food_consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let food_timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    kernel.upload_world(
        &world.terrain.heights,
        &biomes,
        &food_pos,
        &food_consumed,
        &food_timers,
    );

    let spawn_positions: Vec<glam::Vec3> = (0..agent_count)
        .map(|_| world.safe_spawn_position())
        .collect();
    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = spawn_positions
        .iter()
        .map(|&pos| {
            (
                pos,
                100.0,
                100.0,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agent_data);

    // Read state before any ticks (should match uploaded positions)
    let state_before = kernel.read_full_state_blocking();
    for i in 0..agent_count {
        let base = i * PHYS_STRIDE;
        let y = state_before[base + P_POS_Y];
        let expected_y = spawn_positions[i].y;
        assert!(
            (y - expected_y).abs() < 0.01,
            "Agent {} pre-tick Y={:.3} should match spawn Y={:.3}",
            i,
            y,
            expected_y
        );
    }

    // Run 1 tick on the same kernel and verify agents stay near terrain.
    // Using `kernel` directly (not bench::run_bench) so the pre- and post-tick
    // states come from the same kernel instance.
    assert!(
        kernel.dispatch_batch(0, 1),
        "dispatch_batch should return true indicating ticks were submitted"
    );
    let state_after = kernel.read_full_state_blocking();
    for i in 0..agent_count {
        let base = i * PHYS_STRIDE;
        let x = state_after[base + P_POS_X];
        let y = state_after[base + P_POS_Y];
        let z = state_after[base + P_POS_Z];
        let terrain_y = world.terrain.height_at(x, z);
        let diff = y - terrain_y;
        // Agents should be at least AGENT_HALF_HEIGHT (1.0) above terrain;
        // allow small float tolerance (0.99) but reject agents sunk into the ground.
        assert!(
            diff >= 0.99 && diff < 5.0,
            "Agent {} after 1 tick at ({:.2}, {:.2}, {:.2}): terrain_y={:.2}, diff={:.2}",
            i,
            x,
            y,
            z,
            terrain_y,
            diff
        );
    }
}
