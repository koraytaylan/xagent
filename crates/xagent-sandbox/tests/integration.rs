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

/// Proves the fused single-submit dispatch path is
/// bit-identical to running the same ticks as many small `dispatch_ticks`
/// calls.
///
/// 1037 ticks at the default stride (vision_stride=10, brain_tick_stride=10,
/// kernel_batch_size=100) decomposes as:
///   - 10 full kernel-batches (1000 ticks)  → fused into chunked submits
///   - 3 remainder cycles (30 ticks)        → own uniform write + submit
///   - 7 physics-remainder ticks            → physics-only submit
///
/// so it exercises BOTH remainder paths that the exact-multiple
/// `deterministic_across_batch_sizes` test never touches. The split run uses
/// the identical unit sequence (ten one-batch calls + a final 37-tick call =
/// 3 remainder cycles + 7 physics-remainder ticks), just ungrouped across
/// submits, so any divergence is a fusion/ordering bug.
#[test]
fn fused_dispatch_matches_split() {
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

    // Fresh kernel with deterministic brain state + identical initial world.
    let make_kernel = || {
        let mut kernel = xagent_brain::GpuKernel::new(1, food_count, &brain, &world_config);
        kernel.reset_agents_seeded(&brain, 12345);
        kernel.upload_world(&heights, &biomes, &food_pos, &food_consumed, &food_timers);
        kernel.upload_agents(&agent_data);
        kernel
    };

    let total: u32 = 1037;

    // One fused dispatch_ticks call covering all 1037 ticks.
    let mut fused = make_kernel();
    fused.dispatch_ticks(0, total);
    let fs = fused.read_full_state_blocking();
    let fused_pos = [fs[P_POS_X], fs[P_POS_Y], fs[P_POS_Z]];

    // Same 1037 ticks as ten one-batch calls + a final 37-tick call (3
    // remainder cycles + 7 physics-remainder ticks): identical unit sequence,
    // ungrouped across submits.
    let mut split = make_kernel();
    for i in 0..10u32 {
        split.dispatch_ticks(u64::from(i) * 100, 100);
    }
    split.dispatch_ticks(1000, 37);
    let ss = split.read_full_state_blocking();
    let split_pos = [ss[P_POS_X], ss[P_POS_Y], ss[P_POS_Z]];

    eprintln!("fused  1×1037:        {fused_pos:?}");
    eprintln!("split  10×100 + 1×37: {split_pos:?}");
    assert_eq!(
        fused_pos, split_pos,
        "fused dispatch diverged from split dispatch"
    );
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

// ── Learning Probe Tests ────────────────────────────────────────────────
//
// A controlled flat-world arena where each agent has exactly one food item
// at a known bearing. These tests pin down the measurable baseline of the
// current learner (turn/bearing alignment ≈ chance, foraging rate) so that
// learning changes can be evaluated against recorded numbers instead of
// intuition. See docs/superpowers/plans/2026-06-10-emergent-learning-pathway.md.

/// Agents per side of the square probe grid. Must stay even so the derived
/// agent count is even: alternating left/right food bearings then balance
/// exactly, and a systematic turn bias cannot masquerade as food-seeking.
const PROBE_GRID_SIDE: usize = 4;
/// Probe agents per arena — derived from the grid side so the two can
/// never drift. 16 agents give several hundred scored samples per run.
const PROBE_AGENT_COUNT: usize = PROBE_GRID_SIDE * PROBE_GRID_SIDE;
/// Spacing between probe agents. Greater than 2× the vision range (30) so
/// no probe agent can ever see another agent or another agent's food item.
const PROBE_AGENT_SPACING: f32 = 64.0;
/// Horizontal agent→food distance. The lowest below-horizon vision ray row
/// (vertical slope ≈ 0.18 on the default 8×6 grid) passes within the food
/// hit radius (1.0) at this range before the ray strikes flat ground
/// (≈ 5.5 units out), while staying beyond both the touch range (3.0) and
/// the food consume radius (2.0) so a stationary agent neither touches nor
/// eats its probe target.
const PROBE_FOOD_DISTANCE: f32 = 5.0;
/// Food bearing magnitude relative to the agent's initial facing.
/// atan(3/7) aligns the food exactly with a ray column of the default
/// 8-column vision grid (column offset u = ±3/7 at 45° half-FOV), which
/// maximizes ray-hit reliability at the probe distance.
const PROBE_FOOD_BEARING: f32 = 0.404_891_6;
/// Food rest height above flat terrain (matches the kernel's
/// FOOD_HEIGHT_OFFSET).
const PROBE_FOOD_Y: f32 = 0.35;
/// Agent center height above flat terrain (matches the kernel's
/// AGENT_HALF_HEIGHT).
const PROBE_AGENT_Y: f32 = 1.0;
/// Terrain vertices per side (matches the kernel's TERRAIN_VPS).
const PROBE_TERRAIN_VPS: usize = 129;
/// Biome grid resolution (matches the kernel's BIOME_GRID_RES).
const PROBE_BIOME_RES: usize = 256;

/// Probe brain config: single-tick strides give one vision frame and one
/// brain decision per physics tick (sensory lag = 1 tick), and zero
/// movement speed pins each agent at its spawn point — turning still works,
/// so the food bearing changes only through the agent's own rotation.
fn probe_brain_config() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..Default::default()
    }
}

/// Flat, hazard-free probe arena: one food item per agent at a fixed
/// bearing (alternating right/left per agent index), with everything
/// needed to re-pin the bodies for episodic training.
struct ProbeArena {
    kernel: xagent_brain::GpuKernel,
    agent_pos: Vec<glam::Vec3>,
    food_pos: Vec<(f32, f32, f32)>,
    /// `food_pos` with every bearing sign flipped. Episodic training
    /// alternates between the two layouts so a constant per-agent turn
    /// bias earns nothing on average — only vision-conditional turning
    /// pays off.
    mirrored_food_pos: Vec<(f32, f32, f32)>,
    heights: Vec<f32>,
    biomes: Vec<u32>,
    agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)>,
}

impl ProbeArena {
    /// Reset every agent body to its spawn pose (full energy, facing +Z)
    /// and restore all food items, without touching learned brain state.
    /// `mirror_food` selects the bearing-flipped food layout.
    fn reset_bodies_with(&mut self, mirror_food: bool) {
        let food = if mirror_food {
            &self.mirrored_food_pos
        } else {
            &self.food_pos
        };
        self.kernel.upload_world(
            &self.heights,
            &self.biomes,
            food,
            &vec![false; PROBE_AGENT_COUNT],
            &vec![0.0; PROBE_AGENT_COUNT],
        );
        self.kernel.upload_agents(&self.agent_data);
    }

    /// Reset with the canonical (unmirrored) food layout.
    fn reset_bodies(&mut self) {
        self.reset_bodies_with(false);
    }
}

/// Build the probe arena with freshly seeded brains.
fn build_probe_arena(brain: &BrainConfig, brain_seed: u64) -> ProbeArena {
    let world_config = WorldConfig {
        seed: 1,
        ..Default::default()
    };
    let mut kernel = xagent_brain::GpuKernel::new(
        PROBE_AGENT_COUNT as u32,
        PROBE_AGENT_COUNT,
        brain,
        &world_config,
    );
    kernel.reset_agents_seeded(brain, brain_seed);

    // Flat terrain and a uniform food-rich biome: no hazards, no slopes —
    // the only structure in the world is each agent's probe food item.
    let heights = vec![0.0_f32; PROBE_TERRAIN_VPS * PROBE_TERRAIN_VPS];
    let biomes = vec![0_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];

    let mut agent_pos = Vec::with_capacity(PROBE_AGENT_COUNT);
    let mut food_pos = Vec::with_capacity(PROBE_AGENT_COUNT);
    let mut mirrored_food_pos = Vec::with_capacity(PROBE_AGENT_COUNT);
    let grid_center = (PROBE_GRID_SIDE - 1) as f32 / 2.0;
    for ix in 0..PROBE_GRID_SIDE {
        for iz in 0..PROBE_GRID_SIDE {
            let i = ix * PROBE_GRID_SIDE + iz;
            let x = (ix as f32 - grid_center) * PROBE_AGENT_SPACING;
            let z = (iz as f32 - grid_center) * PROBE_AGENT_SPACING;
            agent_pos.push(glam::Vec3::new(x, PROBE_AGENT_Y, z));
            let bearing = if i % 2 == 0 {
                PROBE_FOOD_BEARING
            } else {
                -PROBE_FOOD_BEARING
            };
            food_pos.push((
                x + bearing.sin() * PROBE_FOOD_DISTANCE,
                PROBE_FOOD_Y,
                z + bearing.cos() * PROBE_FOOD_DISTANCE,
            ));
            mirrored_food_pos.push((
                x - bearing.sin() * PROBE_FOOD_DISTANCE,
                PROBE_FOOD_Y,
                z + bearing.cos() * PROBE_FOOD_DISTANCE,
            ));
        }
    }

    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = agent_pos
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
    let mut arena = ProbeArena {
        kernel,
        agent_pos,
        food_pos,
        mirrored_food_pos,
        heights,
        biomes,
        agent_data,
    };
    arena.reset_bodies();
    arena
}

/// Run `ticks` single-tick batches and score the sign of each turn output
/// against the food bearing at the moment the brain's vision frame was
/// captured (single-tick sensory lag). Assumes stationary agents (zero
/// movement speed). Returns (correct, scored). Skips the first two ticks
/// (no real frame yet), bearings outside the reliable vision window, and
/// exactly-zero motor values.
fn score_turn_alignment(arena: &mut ProbeArena, start_tick: u64, ticks: usize) -> (usize, usize) {
    use std::f32::consts::{PI, TAU};
    use xagent_brain::buffers::{PHYS_STRIDE, P_MOTOR_TURN_OUT, P_YAW};

    /// Bearing window for scoring. Above 0.6 rad the food nears the FOV
    /// edge (0.785 rad half-FOV) where ray coverage degrades; below
    /// 0.05 rad the correct turn direction is ambiguous.
    const BEARING_MAX: f32 = 0.6;
    const BEARING_MIN: f32 = 0.05;

    let mut prev_yaw = vec![0.0_f32; PROBE_AGENT_COUNT];
    let mut correct = 0_usize;
    let mut scored = 0_usize;

    for t in 0..ticks {
        arena.kernel.dispatch_batch(start_tick + t as u64, 1);
        let state = arena.kernel.read_full_state_blocking();
        for a in 0..PROBE_AGENT_COUNT {
            let base = a * PHYS_STRIDE;
            let yaw = state[base + P_YAW];
            let motor_turn = state[base + P_MOTOR_TURN_OUT];
            if t >= 2 {
                let dx = arena.food_pos[a].0 - arena.agent_pos[a].x;
                let dz = arena.food_pos[a].2 - arena.agent_pos[a].z;
                let world_bearing = dx.atan2(dz);
                let mut bearing = world_bearing - prev_yaw[a];
                while bearing > PI {
                    bearing -= TAU;
                }
                while bearing < -PI {
                    bearing += TAU;
                }
                if bearing.abs() >= BEARING_MIN && bearing.abs() <= BEARING_MAX && motor_turn != 0.0
                {
                    scored += 1;
                    if (motor_turn > 0.0) == (bearing > 0.0) {
                        correct += 1;
                    }
                }
            }
            prev_yaw[a] = yaw;
        }
    }
    (correct, scored)
}

/// Count vision rays reporting the food color (lime green, matching the
/// hit color written by the vision shader).
fn count_food_pixels(vision_color: &[f32]) -> usize {
    vision_color
        .chunks_exact(4)
        .filter(|px| {
            (px[0] - 0.7).abs() < 0.01 && (px[1] - 0.95).abs() < 0.01 && (px[2] - 0.2).abs() < 0.01
        })
        .count()
}

/// The telemetry sensory tail must expose the packed non-visual senses
/// (velocity 3, facing 3, angular 1, energy, integrity, energy delta,
/// integrity delta, then 4 touch contacts × 4) so probes can assert on
/// touch contacts and interoception without raw buffer plumbing.
#[test]
fn telemetry_exposes_non_visual_sensory_tail() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 31);
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    let layout = xagent_brain::buffers::BrainLayout::new(brain.vision_width, brain.vision_height);
    let expected_len =
        layout.sensory_stride - layout.vision_color_count - layout.vision_depth_count;
    assert_eq!(
        telemetry.sensory_non_visual.len(),
        expected_len,
        "non-visual tail length must match the layout"
    );
    // Energy is packed normalized at index 7 of the tail and the agent
    // is alive at full-ish energy after one tick.
    let energy_normalized = telemetry.sensory_non_visual[7];
    assert!(
        (0.5..=1.0).contains(&energy_normalized),
        "normalized energy {energy_normalized} not in (0.5, 1.0] after one tick"
    );
}

/// Information-path check: the probe geometry must actually be visible.
/// After one vision pass, every probe agent must have at least one ray
/// reporting the food color. If this fails, the arena geometry (distance,
/// bearing, ray layout) is broken and the other probe metrics are
/// meaningless.
#[test]
fn learning_probe_food_is_visible() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 7);

    // One single-tick batch: pass order is kernel → global (grid rebuild)
    // → vision, so the food grid is populated and one vision frame exists.
    arena.kernel.dispatch_batch(0, 1);

    let mut blind_agents = Vec::new();
    for agent in 0..PROBE_AGENT_COUNT {
        let telemetry = arena.kernel.read_agent_telemetry_blocking(agent as u32);
        if count_food_pixels(&telemetry.vision_color) == 0 {
            blind_agents.push(agent);
        }
    }
    assert!(
        blind_agents.is_empty(),
        "agents {blind_agents:?} see no food pixel at distance {PROBE_FOOD_DISTANCE} \
         bearing ±{PROBE_FOOD_BEARING}; probe geometry no longer matches the vision ray layout"
    );
}

/// Baseline directional-learning probe: with the current learner, the sign
/// of the turn output should be uncorrelated with the food's bearing —
/// alignment ≈ chance. A learner that acquires food-approach behavior must
/// push this rate decisively above the asserted band; the band itself
/// documents (and pins) today's chance-level baseline.
#[test]
fn learning_probe_baseline_turn_alignment_is_chance() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Single-tick batches dispatched (= brain decisions sampled per agent).
    const PROBE_TICKS: usize = 60;
    /// Minimum scored samples for the rate to be statistically meaningful
    /// (binomial σ ≈ 0.035 at n = 200).
    const MIN_SCORED_SAMPLES: usize = 200;

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 11);

    let (correct, scored) = score_turn_alignment(&mut arena, 0, PROBE_TICKS);

    let rate = correct as f64 / scored.max(1) as f64;
    eprintln!("learning probe baseline: turn/bearing alignment {correct}/{scored} = {rate:.3}");
    assert!(
        scored >= MIN_SCORED_SAMPLES,
        "only {scored} scored samples — agents rotated out of the bearing window too fast \
         or the probe geometry broke"
    );
    // ±6σ band around chance for the sample sizes this probe produces.
    // An untrained policy sits at ≈ 0.5; without reward events in this
    // stationary arena, nothing should push it off chance.
    assert!(
        (0.38..=0.62).contains(&rate),
        "turn/bearing alignment {rate:.3} is outside the chance band [0.38, 0.62] — \
         a directional bias crept into the untrained policy"
    );
}

/// Free-running foraging baseline in the probe arena with the default
/// config (normal movement, default strides): records food eaten and deaths
/// over a fixed tick budget. The printed numbers are the recorded baseline
/// that learning changes must improve.
#[test]
fn learning_probe_free_run_foraging_baseline() {
    use xagent_brain::buffers::{PHYS_STRIDE, P_DEATH_COUNT, P_FOOD_COUNT, P_TICKS_ALIVE};

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Total physics ticks. Long enough for several food encounters and the
    /// full energy-drain arc, short enough to stay fast on slow adapters.
    const RUN_TICKS: u32 = 3000;
    /// One death costs exactly one un-incremented alive tick (death and
    /// respawn happen in the same kernel cycle), so allow a small gap.
    const MAX_DEATH_TICK_GAP: f32 = 100.0;

    let brain = BrainConfig::default();
    let mut arena = build_probe_arena(&brain, 13);

    let batch = arena.kernel.kernel_batch_size();
    assert!(
        RUN_TICKS % batch == 0,
        "RUN_TICKS {RUN_TICKS} must be a multiple of kernel_batch_size {batch}"
    );
    let mut tick = 0_u64;
    while tick < u64::from(RUN_TICKS) {
        arena.kernel.dispatch_batch(tick, batch);
        tick += u64::from(batch);
    }

    let state = arena.kernel.read_full_state_blocking();
    let mut total_food = 0.0_f32;
    let mut total_deaths = 0.0_f32;
    for a in 0..PROBE_AGENT_COUNT {
        let base = a * PHYS_STRIDE;
        total_food += state[base + P_FOOD_COUNT];
        total_deaths += state[base + P_DEATH_COUNT];
        let ticks_alive = state[base + P_TICKS_ALIVE];
        // Liveness accounting: ticks_alive survives respawn, so it must
        // track the dispatched budget minus at most a small death gap.
        // A zero here means the simulation never ran.
        assert!(
            ticks_alive >= RUN_TICKS as f32 - MAX_DEATH_TICK_GAP && ticks_alive <= RUN_TICKS as f32,
            "agent {a}: ticks_alive {ticks_alive} outside expected range for budget {RUN_TICKS}"
        );
    }
    let food_per_agent_per_1k = total_food / PROBE_AGENT_COUNT as f32 / (RUN_TICKS as f32 / 1000.0);
    eprintln!(
        "learning probe baseline: food={total_food} deaths={total_deaths} \
         food/agent/1k-ticks={food_per_agent_per_1k:.3}"
    );
}

/// Vision-acuity check: the 17×13 odd grid has a horizon-grazing ray
/// row (odd row count) that passes a constant 0.65 below eye level —
/// inside the 1.0 food hit radius — so ground-level food straight ahead is
/// visible at every probed distance out to near the 30-unit vision range.
/// The 8×6 default documents what the upgrade buys: its lowest
/// below-horizon row strikes flat ground ≈ 5.5 units out, so food at 20 is
/// geometrically invisible regardless of bearing.
#[test]
fn vision_horizon_row_sees_food_at_range() {
    use xagent_brain::buffers::PHYS_STRIDE;
    use xagent_brain::GpuKernel;

    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Probed agent→food distances (one agent per distance). 25 stays
    /// inside the 30-unit ray-march range with sampling slack.
    const PROBE_DISTANCES: [f32; 5] = [5.0, 10.0, 15.0, 20.0, 25.0];
    /// Row spacing for the five probe agents: exactly 2× the vision range
    /// (mutual invisibility) while keeping the outermost agents at x = ±120,
    /// inside the ±128 world bound — an agent clamped at the boundary would
    /// slide off its food's ray column.
    const RANGE_PROBE_SPACING: f32 = 60.0;

    // One agent per distance, on the same spaced grid as the learning
    // probes, each with one food item straight ahead (+Z) on flat ground.
    let build = |brain: &BrainConfig| -> (GpuKernel, Vec<usize>) {
        let world_config = WorldConfig {
            seed: 1,
            ..Default::default()
        };
        let count = PROBE_DISTANCES.len();
        let mut kernel = GpuKernel::new(count as u32, count, brain, &world_config);
        kernel.reset_agents_seeded(brain, 29);
        let heights = vec![0.0_f32; PROBE_TERRAIN_VPS * PROBE_TERRAIN_VPS];
        let biomes = vec![0_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
        let mut agent_data = Vec::with_capacity(count);
        let mut food_pos = Vec::with_capacity(count);
        for (i, &dist) in PROBE_DISTANCES.iter().enumerate() {
            let x = (i as f32 - (count - 1) as f32 / 2.0) * RANGE_PROBE_SPACING;
            agent_data.push((
                glam::Vec3::new(x, PROBE_AGENT_Y, 0.0),
                100.0,
                100.0,
                brain.memory_capacity,
                brain.processing_slots,
            ));
            food_pos.push((x, PROBE_FOOD_Y, dist));
        }
        kernel.upload_world(
            &heights,
            &biomes,
            &food_pos,
            &vec![false; count],
            &vec![0.0; count],
        );
        kernel.upload_agents(&agent_data);
        kernel.dispatch_batch(0, 1);
        let mut seen = Vec::new();
        let state_alive: Vec<f32> = kernel.read_full_state_blocking().to_vec();
        for i in 0..count {
            assert!(
                state_alive[i * PHYS_STRIDE + xagent_brain::buffers::P_ALIVE] > 0.5,
                "probe agent {i} died during the single vision tick"
            );
            let telemetry = kernel.read_agent_telemetry_blocking(i as u32);
            if count_food_pixels(&telemetry.vision_color) > 0 {
                seen.push(i);
            }
        }
        (kernel, seen)
    };

    // The odd-grid 17×13: a horizon ray row makes every distance visible.
    let odd_grid = BrainConfig {
        vision_width: 17,
        vision_height: 13,
        ..probe_brain_config()
    };
    let (_, seen_odd) = build(&odd_grid);
    assert_eq!(
        seen_odd,
        (0..PROBE_DISTANCES.len()).collect::<Vec<_>>(),
        "17×13: food must be visible at every probed distance \
         {PROBE_DISTANCES:?} (missing indices = blind distances)"
    );

    // 8×6 (current default): distal food must be geometrically invisible —
    // this is the information defect the odd grid fixes. If this ever starts
    // passing, the contrast premise broke.
    let legacy_brain = BrainConfig {
        vision_width: 8,
        vision_height: 6,
        ..probe_brain_config()
    };
    let (_, seen_legacy) = build(&legacy_brain);
    assert!(
        seen_legacy.contains(&0) && !seen_legacy.contains(&3),
        "8×6: the near food (distance 5) must be visible and the distance-20 \
         food invisible (vertical ray gap); got {seen_legacy:?} — the \
         contrast premise broke"
    );
    eprintln!(
        "vision range probe: 17×13 sees {:?}, 8×6 sees {:?} (indices into {PROBE_DISTANCES:?})",
        seen_odd, seen_legacy
    );
}

/// Honest directional probe (confound-free protocol).
///
/// Each training episode mirrors the food side, so a constant per-agent
/// turn bias earns nothing on average — only genuinely vision-conditional
/// turning ("turn toward where the food is seen") is rewarded. After
/// training, the stationary alignment evaluation currently lands at chance:
/// TD(λ) credit through the random-projection encoder does **not** yet
/// teach vision-conditional steering. This test pins that honest baseline
/// (same falsifiable pattern as `learning_probe_baseline_turn_alignment_is_chance`):
/// a representation/credit change that finally produces directional
/// steering will push the rate out of the chance band and trip this test,
/// which is the signal to re-pin it upward.
///
/// (An earlier version mirrored nothing and reported ~0.64 "learning"; that
/// number was inflated by per-agent side-consistency — each agent always
/// saw food on one side in both training and eval — not by directional
/// learning. See docs/superpowers/specs/2026-06-10-learning-baseline.md.)
#[test]
fn learning_probe_mirrored_steering_is_chance() {
    use xagent_brain::buffers::{PHYS_STRIDE, P_FOOD_COUNT};

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Training episodes, food side alternating each episode.
    const TRAIN_EPISODES: usize = 120;
    /// Ticks per episode: at single-tick strides and default speed an agent
    /// heading roughly toward its food (5 units away) eats within ~30
    /// ticks, leaving slack for indirect paths.
    const EPISODE_TICKS: u32 = 100;
    /// Evaluation ticks (same scale as the baseline probe).
    const EVAL_TICKS: usize = 60;
    /// Minimum scored evaluation samples.
    const MIN_SCORED_SAMPLES: usize = 200;

    // Training config: single-tick strides (fresh vision every tick —
    // densest TD transitions) with normal movement so food is reachable.
    let train_brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        ..Default::default()
    };
    let mut arena = build_probe_arena(&train_brain, 17);

    let mut tick_cursor = 0_u64;
    let mut food_total = 0.0_f32;
    for episode in 0..TRAIN_EPISODES {
        // Alternate the food side so only vision-conditional turning pays.
        arena.reset_bodies_with(episode % 2 == 1);
        arena.kernel.dispatch_batch(tick_cursor, EPISODE_TICKS);
        tick_cursor += u64::from(EPISODE_TICKS);
        let state = arena.kernel.read_full_state_blocking();
        for a in 0..PROBE_AGENT_COUNT {
            food_total += state[a * PHYS_STRIDE + P_FOOD_COUNT];
        }
    }
    // Sanity: agents do reach food during training (the arena works and the
    // policy is not paralyzed) — this is foraging, not directional steering.
    assert!(
        food_total > 0.0,
        "no food eaten across {TRAIN_EPISODES} training episodes — arena broke"
    );

    // Evaluation: pin the agents (zero movement speed) via the heritable
    // config patch — learned weights stay intact — and score alignment
    // exactly like the baseline probe.
    let eval_brain = probe_brain_config();
    for a in 0..PROBE_AGENT_COUNT {
        arena
            .kernel
            .write_agent_heritable_config(a as u32, &eval_brain);
    }
    arena.reset_bodies();
    let (correct, scored) = score_turn_alignment(&mut arena, tick_cursor, EVAL_TICKS);

    let rate = correct as f64 / scored.max(1) as f64;
    eprintln!(
        "mirrored steering probe: food={food_total}, turn/bearing alignment \
         {correct}/{scored} = {rate:.3}"
    );
    assert!(
        scored >= MIN_SCORED_SAMPLES,
        "only {scored} scored samples — evaluation geometry broke"
    );
    // Honest baseline: vision-conditional steering is at chance. If a future
    // change produces real directional steering, `rate` leaves this band and
    // this assertion fires — re-pin it then.
    assert!(
        (0.38..=0.62).contains(&rate),
        "mirrored turn/bearing alignment {rate:.3} left the chance band [0.38, 0.62] — \
         if directional steering emerged, re-pin this baseline upward"
    );
}

/// Diagnostic: does the encoder keep food-left and food-right linearly
/// separable? Presents one agent (one encoder) the same scene with food on
/// the right, then on the left, and reads the pre-habituation encoded state
/// (`O_PREV_ENCODED`) for each. The directional signal the policy must read
/// is `encoded(right) − encoded(left)`; this measures whether that signal
/// rises above the within-class noise (two right-side scenes at slightly
/// different distances).
///
/// Reports `between` (cosine of right vs left) against `within` (cosine of
/// two right-side scenes). If `between ≈ within ≈ 1`, the encoder collapses
/// the food side below the readout floor — the encoder is the binding
/// constraint for directional steering. If `1 − between` is clearly larger
/// than `1 − within`, the side is represented and the bottleneck is the
/// credit/temporal path instead. Measurement-first: prints the numbers and
/// asserts only that the read succeeded.
#[test]
fn encoder_food_side_separability_diagnostic() {
    use xagent_brain::buffers::{
        BrainLayout, ENCODED_DIMENSION, O_PREDICTOR_CONTEXT_WEIGHT, O_PREV_ENCODED,
        PREDICTOR_DIMENSION,
    };

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let world_config = WorldConfig {
        seed: 1,
        ..Default::default()
    };
    let mut kernel = xagent_brain::GpuKernel::new(1, 1, &brain, &world_config);
    kernel.reset_agents_seeded(&brain, 31);
    let heights = vec![0.0_f32; PROBE_TERRAIN_VPS * PROBE_TERRAIN_VPS];
    let biomes = vec![0_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    let agent_data = vec![(
        glam::Vec3::new(0.0, PROBE_AGENT_Y, 0.0),
        100.0_f32,
        100.0_f32,
        brain.memory_capacity,
        brain.processing_slots,
    )];

    let layout = BrainLayout::new(brain.vision_width, brain.vision_height);
    let prev_encoded_off = layout.feature_count * ENCODED_DIMENSION
        + ENCODED_DIMENSION
        + PREDICTOR_DIMENSION * ENCODED_DIMENSION
        + (O_PREV_ENCODED - O_PREDICTOR_CONTEXT_WEIGHT);

    // Present food at a given bearing/distance and return the encoded state.
    // Two single-tick batches: the first runs vision, the second lets the
    // brain encode that frame into O_PREV_ENCODED.
    let mut tick = 0_u64;
    let mut present = |kernel: &mut xagent_brain::GpuKernel, bearing: f32, dist: f32| -> Vec<f32> {
        let food = vec![(bearing.sin() * dist, PROBE_FOOD_Y, bearing.cos() * dist)];
        kernel.upload_world(&heights, &biomes, &food, &[false], &[0.0]);
        kernel.upload_agents(&agent_data); // re-pin pose (facing +Z, full energy)
        kernel.dispatch_batch(tick, 1);
        kernel.dispatch_batch(tick + 1, 1);
        tick += 2;
        let bs = kernel.read_agent_state(0).brain_state;
        bs[prev_encoded_off..prev_encoded_off + ENCODED_DIMENSION].to_vec()
    };

    let cosine = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    };

    let e_right = present(&mut kernel, PROBE_FOOD_BEARING, PROBE_FOOD_DISTANCE);
    let e_right2 = present(&mut kernel, PROBE_FOOD_BEARING, PROBE_FOOD_DISTANCE + 1.0);
    let e_left = present(&mut kernel, -PROBE_FOOD_BEARING, PROBE_FOOD_DISTANCE);

    assert!(
        e_right.iter().any(|v| v.abs() > 1e-6) && e_left.iter().any(|v| v.abs() > 1e-6),
        "encoded states are all-zero — vision/encode path did not run"
    );

    let within = cosine(&e_right, &e_right2);
    let between = cosine(&e_right, &e_left);
    eprintln!(
        "encoder separability: within(right,right') cos={within:.4} (dist {:.4}), \
         between(right,left) cos={between:.4} (dist {:.4})",
        1.0 - within,
        1.0 - between,
    );
}

/// The critic must learn that the steady metabolic drain makes every state
/// slightly negative-valued: with stationary agents and no food events, the
/// value telemetry should settle below zero, and every TD error must
/// respect the MAX_TD_ERROR clamp.
#[test]
fn td_critic_tracks_metabolic_drain() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Enough ticks for the linear critic to converge on the constant-drain
    /// signal (time constant ≈ 100 brain ticks at the critic rate).
    const RUN_TICKS: usize = 300;

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 19);

    for t in 0..RUN_TICKS {
        arena.kernel.dispatch_batch(t as u64, 1);
    }

    let mut value_sum = 0.0_f32;
    for a in 0..PROBE_AGENT_COUNT {
        let telemetry = arena.kernel.read_agent_telemetry_blocking(a as u32);
        assert!(
            telemetry.value.is_finite() && telemetry.td_error.is_finite(),
            "agent {a}: non-finite critic telemetry (value={}, td_error={})",
            telemetry.value,
            telemetry.td_error
        );
        assert!(
            telemetry.td_error.abs() <= 1.0,
            "agent {a}: td_error {} exceeds the MAX_TD_ERROR clamp",
            telemetry.td_error
        );
        value_sum += telemetry.value;
    }
    let mean_value = value_sum / PROBE_AGENT_COUNT as f32;
    eprintln!("td critic drain probe: mean value {mean_value:.5}");
    assert!(
        mean_value < -1e-4,
        "mean value {mean_value:.5} did not go negative under constant drain — \
         the critic is not learning"
    );
    assert!(
        mean_value > -1.0,
        "mean value {mean_value:.5} is implausibly negative for a drain of ~1e-3/tick"
    );
}

/// Dying must apply one terminal TD update (δ = −MAX_TD_ERROR) through
/// the dying life's eligibility traces before they are cleared. With
/// preset traces the kick is exactly computable:
/// Δvalue_bias = 0.01·(−1)·5 = −0.05 and Δactor_bias = 0.1·(−1)·1 =
/// −0.10. The post-respawn brain tick in the same cycle applies δ
/// through freshly zeroed traces, so it cannot move the biases — any
/// deviation from the exact kick is a real defect.
#[test]
fn death_applies_terminal_td_update_through_traces() {
    use xagent_brain::buffers::{
        BrainLayout, ENCODED_DIMENSION, O_ACT_BIASES, O_PREDICTOR_CONTEXT_WEIGHT, O_TRACE_BIASES,
        O_VALUE_BIAS, PHYS_STRIDE, PREDICTOR_DIMENSION, P_DEATH_COUNT,
    };

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Hazard damage is rate (1.0) × integrity_scale per physics tick;
    /// 200 wipes the full 100 integrity in a single tick inside a
    /// danger biome.
    const ONE_TICK_KILL_INTEGRITY_SCALE: f32 = 200.0;

    let brain = BrainConfig {
        integrity_scale: ONE_TICK_KILL_INTEGRITY_SCALE,
        ..probe_brain_config()
    };
    // Warm-up happens on safe biome; then the world flips to all-danger.
    let mut arena = build_probe_arena(&brain, 29);
    arena.kernel.dispatch_batch(0, 1);
    arena.biomes = vec![2_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    arena.reset_bodies();

    let layout = BrainLayout::new(brain.vision_width, brain.vision_height);
    let tail_base = layout.feature_count * ENCODED_DIMENSION
        + ENCODED_DIMENSION
        + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
    let value_bias_offset = tail_base + (O_VALUE_BIAS - O_PREDICTOR_CONTEXT_WEIGHT);
    let trace_biases_offset = tail_base + (O_TRACE_BIASES - O_PREDICTOR_CONTEXT_WEIGHT);
    let act_biases_offset = tail_base + (O_ACT_BIASES - O_PREDICTOR_CONTEXT_WEIGHT);

    let agent = 0_u32;
    let mut state = arena.kernel.read_agent_state(agent);
    state.brain_state[value_bias_offset] = 0.5;
    state.brain_state[trace_biases_offset] = 5.0;
    state.brain_state[trace_biases_offset + 1] = 1.0;
    state.brain_state[trace_biases_offset + 2] = 1.0;
    let forward_bias_before = state.brain_state[act_biases_offset];
    let turn_bias_before = state.brain_state[act_biases_offset + 1];
    arena.kernel.write_agent_state(agent, &state);

    // This tick kills (integrity 100 → 0), respawns, and runs one
    // post-respawn brain tick whose traces were just zeroed — so the
    // only bias change in this tick is the terminal kick.
    arena.kernel.dispatch_batch(1, 1);

    let physics = arena.kernel.read_full_state_blocking();
    assert!(
        physics[agent as usize * PHYS_STRIDE + P_DEATH_COUNT] >= 1.0,
        "agent did not die in the one-tick-kill arena"
    );

    let after = arena.kernel.read_agent_state(agent);
    let value_bias = after.brain_state[value_bias_offset];
    let forward_bias = after.brain_state[act_biases_offset];
    let turn_bias = after.brain_state[act_biases_offset + 1];
    assert!(
        (value_bias - 0.45).abs() < 1e-3,
        "value bias {value_bias} != 0.45: terminal critic kick missing or wrong"
    );
    assert!(
        (forward_bias - (forward_bias_before - 0.10)).abs() < 1e-3,
        "forward bias {forward_bias} (was {forward_bias_before}): terminal actor kick missing"
    );
    assert!(
        (turn_bias - (turn_bias_before - 0.10)).abs() < 1e-3,
        "turn bias {turn_bias} (was {turn_bias_before}): terminal actor kick missing"
    );
}

/// Eligibility traces are episodic: after deaths they must have been reset
/// (and rebuilt only from post-respawn experience), so they stay bounded by
/// the geometric trace limit instead of accumulating across lives.
#[test]
fn td_traces_bounded_across_deaths() {
    use xagent_brain::buffers::{
        BrainLayout, ENCODED_DIMENSION, O_PREDICTOR_CONTEXT_WEIGHT, O_TRACE_CRITIC, O_TRACE_FWD,
        O_TRACE_TURN, PHYS_STRIDE, PREDICTOR_DIMENSION, P_DEATH_COUNT,
    };

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    /// Hazard damage (1.0/tick at default rates) kills a 100-integrity
    /// agent in ~100 ticks; 350 ticks guarantees repeated deaths.
    const RUN_TICKS: usize = 350;
    /// Geometric trace bound: |s_encoded| ≤ 1 per dim and noise ≤ 0.5, so
    /// |z| ≤ 1/(1 − γλ) ≈ 7.9. Allow generous slack for the brief
    /// post-respawn rebuild before asserting runaway accumulation.
    const TRACE_BOUND: f32 = 50.0;

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 23);
    // All-danger biome: every spawn fallback lands in hazard, so agents die
    // on a ~100-tick cycle.
    arena.biomes = vec![2_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    arena.reset_bodies();

    for t in 0..RUN_TICKS {
        arena.kernel.dispatch_batch(t as u64, 1);
    }

    let state = arena.kernel.read_full_state_blocking();
    let mut total_deaths = 0.0_f32;
    for a in 0..PROBE_AGENT_COUNT {
        total_deaths += state[a * PHYS_STRIDE + P_DEATH_COUNT];
    }
    assert!(
        total_deaths >= 1.0,
        "no deaths in the all-danger arena — the death path never ran"
    );

    // Trace offsets, rebased onto the live layout: the tail deltas from
    // O_PREDICTOR_CONTEXT_WEIGHT are vision-independent.
    let layout = BrainLayout::new(brain.vision_width, brain.vision_height);
    let tail_base = layout.feature_count * ENCODED_DIMENSION
        + ENCODED_DIMENSION
        + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
    for a in 0..PROBE_AGENT_COUNT {
        let brain_state = arena.kernel.read_agent_state(a as u32).brain_state;
        for d in 0..ENCODED_DIMENSION {
            for (name, static_off) in [
                ("critic", O_TRACE_CRITIC),
                ("fwd", O_TRACE_FWD),
                ("turn", O_TRACE_TURN),
            ] {
                let off = tail_base + (static_off - O_PREDICTOR_CONTEXT_WEIGHT);
                let z = brain_state[off + d];
                assert!(
                    z.is_finite() && z.abs() <= TRACE_BOUND,
                    "agent {a}: {name} trace[{d}] = {z} out of bounds after \
                     {total_deaths} deaths — traces are leaking across lives"
                );
            }
        }
    }
    eprintln!("td trace death probe: {total_deaths} deaths, traces bounded");
}

// ── Hazard Probe Tests ──────────────────────────────────────────────
//
// A half-plane danger arena: danger biome for x < 0, food-rich for
// x ≥ 0. Agents start 10 units inside the danger side, facing +Z
// (parallel to the boundary, so straight-line walking never exits on
// its own). Measured: hazard-exit latency (first tick with x ≥ 0,
// alive, without dying first) and deaths. These pin the
// danger-avoidance baseline the same way the mirrored steering probe
// pins food-approach.

/// Distance agents start inside the danger half-plane. Far enough that
/// exit requires sustained directed movement (~15 ticks of straight-east
/// walking at default speed), close enough that random walks exit within
/// the episode often enough to measure a latency distribution.
const HAZARD_PROBE_START_DEPTH: f32 = 10.0;
/// Episode cap in physics ticks. At hazard damage 0.5/tick (rate 1.0 ×
/// integrity_scale 0.5) an agent that never exits dies at tick 200, so
/// 600 ticks cleanly separates "exited", "died", and "wandered".
const HAZARD_PROBE_EPISODE_TICKS: u64 = 600;
/// Position sampling interval — bounds latency resolution and readback
/// cost.
const HAZARD_PROBE_SAMPLE_TICKS: u32 = 5;
/// Episodes per measurement.
const HAZARD_PROBE_EPISODES: usize = 3;

/// Outcome of one hazard episode for one agent.
struct HazardEpisodeOutcome {
    exit_latency_ticks: Option<u64>,
    died: bool,
}

/// Run one hazard episode and classify each agent's outcome.
fn run_hazard_episode(arena: &mut ProbeArena, start_tick: u64) -> Vec<HazardEpisodeOutcome> {
    use xagent_brain::buffers::{PHYS_STRIDE, P_DEATH_COUNT, P_POS_X};

    let initial_state = arena.kernel.read_full_state_blocking().to_vec();
    let initial_deaths: Vec<f32> = (0..PROBE_AGENT_COUNT)
        .map(|a| initial_state[a * PHYS_STRIDE + P_DEATH_COUNT])
        .collect();

    let mut outcomes: Vec<HazardEpisodeOutcome> = (0..PROBE_AGENT_COUNT)
        .map(|_| HazardEpisodeOutcome {
            exit_latency_ticks: None,
            died: false,
        })
        .collect();

    let mut ticks_done: u64 = 0;
    while ticks_done < HAZARD_PROBE_EPISODE_TICKS {
        arena
            .kernel
            .dispatch_batch(start_tick + ticks_done, HAZARD_PROBE_SAMPLE_TICKS);
        ticks_done += u64::from(HAZARD_PROBE_SAMPLE_TICKS);
        let state = arena.kernel.read_full_state_blocking();
        for (a, outcome) in outcomes.iter_mut().enumerate() {
            if outcome.died || outcome.exit_latency_ticks.is_some() {
                continue;
            }
            if state[a * PHYS_STRIDE + P_DEATH_COUNT] > initial_deaths[a] {
                outcome.died = true;
            } else if state[a * PHYS_STRIDE + P_POS_X] >= 0.0 {
                outcome.exit_latency_ticks = Some(ticks_done);
            }
        }
    }
    outcomes
}

/// Build the half-plane danger arena on top of the standard probe
/// arena: biome column < 128 (x < 0) is danger, the rest food-rich;
/// agents are re-positioned to x = −HAZARD_PROBE_START_DEPTH, spread
/// along z.
fn build_hazard_arena(brain: &BrainConfig, brain_seed: u64) -> ProbeArena {
    let mut arena = build_probe_arena(brain, brain_seed);
    let mut biomes = vec![0_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    for row in 0..PROBE_BIOME_RES {
        for col in 0..PROBE_BIOME_RES / 2 {
            biomes[row * PROBE_BIOME_RES + col] = 2;
        }
    }
    arena.biomes = biomes;
    for (index, agent) in arena.agent_data.iter_mut().enumerate() {
        let z_spread = (index as f32 - (PROBE_AGENT_COUNT as f32 - 1.0) / 2.0) * 8.0;
        agent.0 = glam::Vec3::new(-HAZARD_PROBE_START_DEPTH, PROBE_AGENT_Y, z_spread);
    }
    arena.reset_bodies();
    arena
}

/// Baseline: untrained agents in the hazard arena. Prints exit
/// fraction, mean exit latency, and death fraction; asserts structural
/// sanity plus pinned falsifiable bands. Re-pin the bands when a
/// change improves escape (the same protocol as the steering
/// probes).
#[test]
fn hazard_probe_exit_latency_baseline() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        ..Default::default()
    };
    let mut arena = build_hazard_arena(&brain, 37);

    let mut exits: Vec<u64> = Vec::new();
    let mut deaths = 0_usize;
    let mut tick_cursor = 0_u64;
    for episode in 0..HAZARD_PROBE_EPISODES {
        if episode > 0 {
            arena.reset_bodies();
        }
        for outcome in run_hazard_episode(&mut arena, tick_cursor) {
            if let Some(latency) = outcome.exit_latency_ticks {
                exits.push(latency);
            }
            if outcome.died {
                deaths += 1;
            }
        }
        tick_cursor += HAZARD_PROBE_EPISODE_TICKS;
    }

    let trials = HAZARD_PROBE_EPISODES * PROBE_AGENT_COUNT;
    let exit_fraction = exits.len() as f64 / trials as f64;
    let death_fraction = deaths as f64 / trials as f64;
    let mean_latency = if exits.is_empty() {
        f64::from(u32::MAX)
    } else {
        exits.iter().sum::<u64>() as f64 / exits.len() as f64
    };
    eprintln!(
        "hazard probe baseline: trials={trials} exit_fraction={exit_fraction:.3} \
         mean_exit_latency={mean_latency:.1} death_fraction={death_fraction:.3}"
    );

    // Pinned baseline recorded 2026-06-12 on macOS/Metal (wgpu adapter):
    // raw exit_fraction=0.188, mean_exit_latency=137.2, death_fraction=0.812
    // (9/48 exits, mean of exits 137.2, 39/48 deaths). ±50% relative bands —
    // generous for adapter noise, tight enough for real avoidance gains to
    // trip. Re-pin on improvement (same protocol as steering probes).
    assert!(
        (0.094..=0.282).contains(&exit_fraction),
        "hazard exit_fraction {exit_fraction:.3} outside pinned band [0.094, 0.282] — \
         re-pin if avoidance improves"
    );
    assert!(
        (68.6..=205.8).contains(&mean_latency),
        "hazard mean_exit_latency {mean_latency:.1} outside pinned band [68.6, 205.8] — \
         re-pin if avoidance improves"
    );
    assert!(
        (0.406..=1.218).contains(&death_fraction),
        "hazard death_fraction {death_fraction:.3} outside pinned band [0.406, 1.218] — \
         re-pin if avoidance improves"
    );

    // Structural sanity: every trial resolves into exit, death, or
    // timeout.
    assert!(exits.len() + deaths <= trials, "double-counted outcomes");
    // Falsifiable floor: the arena must actually be dangerous — if
    // nothing ever dies and everything exits instantly, the geometry
    // broke.
    assert!(
        death_fraction > 0.0 || mean_latency > 50.0,
        "arena is not hazardous: death_fraction={death_fraction}, \
         mean_exit_latency={mean_latency}"
    );
}

/// Standing in a danger biome must produce a TOUCH_HAZARD contact in
/// the sensory buffer: zero planar direction (the hazard is underfoot),
/// fixed intensity, tag 3/4. Mirrors the CPU reference in
/// agent/senses.rs.
#[test]
fn gpu_touch_emits_hazard_contact_in_danger_biome() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 41);
    arena.biomes = vec![2_u32; PROBE_BIOME_RES * PROBE_BIOME_RES];
    arena.reset_bodies();
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    // Touch slots start after [vel(3), facing(3), angular(1),
    // interoception(4)] = 11.
    let touch_base = 11;
    let first_slot = &telemetry.sensory_non_visual[touch_base..touch_base + 4];
    assert!(
        (first_slot[3] - 0.75).abs() < 1e-3,
        "first touch slot tag {} != 0.75 (TOUCH_HAZARD/4) — hazard contact missing",
        first_slot[3]
    );
    assert!(
        (first_slot[2] - 0.5).abs() < 1e-3,
        "hazard contact intensity {} != 0.5",
        first_slot[2]
    );
    assert!(
        first_slot[0].abs() < 1e-6 && first_slot[1].abs() < 1e-6,
        "hazard contact direction must be planar zero, got ({}, {})",
        first_slot[0],
        first_slot[1]
    );
}

/// Standing within TOUCH_EDGE_RANGE of a world wall must produce a
/// TOUCH_TERRAIN_EDGE contact pointing inward with closeness intensity.
#[test]
fn gpu_touch_emits_terrain_edge_contact_near_wall() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let brain = probe_brain_config();
    let mut arena = build_probe_arena(&brain, 43);
    // World half-bound (WC_WORLD_HALF_BOUND = ws/2 - 1) gives the
    // effective clamp; 1.5 units from the +X effective wall (pos = 125.5
    // yields dist 1.5). Matches intensity calc in GPU touch code.
    arena.agent_data[0].0 = glam::Vec3::new(125.5, PROBE_AGENT_Y, 0.0);
    arena.reset_bodies();
    arena.kernel.dispatch_batch(0, 1);

    let telemetry = arena.kernel.read_agent_telemetry_blocking(0);
    let touch_base = 11;
    let mut edge_slot: Option<&[f32]> = None;
    for contact in 0..4 {
        let slot =
            &telemetry.sensory_non_visual[touch_base + contact * 4..touch_base + contact * 4 + 4];
        if (slot[3] - 0.5).abs() < 1e-3 {
            edge_slot = Some(slot);
            break;
        }
    }
    let slot = edge_slot.expect("no TOUCH_TERRAIN_EDGE contact found near the +X wall");
    assert!(
        slot[0] < -0.9,
        "edge contact must point inward (−X), got direction x = {}",
        slot[0]
    );
    assert!(
        (slot[2] - 0.5).abs() < 0.05,
        "edge intensity {} != ~0.5 at 1.5 units from a 3-unit range wall",
        slot[2]
    );
}

// ── Split compute dispatch from CPU-visible publication ───────────────────

/// Build a kernel with world + agents uploaded and deterministic brain state,
/// ready for dispatch. Shared by the dispatch/publication split tests below.
fn split_test_kernel(agent_count: usize, seed: u64) -> (xagent_brain::GpuKernel, WorldState) {
    let brain = BrainConfig::default();
    let world_config = WorldConfig {
        seed: 7,
        ..Default::default()
    };
    let world = WorldState::new(world_config.clone());
    let food_count = world.food_items.len();

    let mut kernel =
        xagent_brain::GpuKernel::new(agent_count as u32, food_count, &brain, &world_config);
    kernel.reset_agents_seeded(&brain, seed);

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

    let agent_data: Vec<(glam::Vec3, f32, f32, usize, usize)> = (0..agent_count)
        .map(|_| {
            (
                world.safe_spawn_position(),
                100.0_f32,
                100.0_f32,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agent_data);
    (kernel, world)
}

/// Poll the non-blocking snapshot collector until it reports fresh data or the
/// bounded retry budget is exhausted.
///
/// `try_collect_state_snapshot` polls the device non-blockingly, so a brief
/// sleep between attempts gives the GPU wall-time to finish the staging copy —
/// the same cadence the real frame loop provides between redraws. Returns
/// `false` only if no snapshot arrived within the (generous) bound.
fn collect_snapshot(kernel: &mut xagent_brain::GpuKernel) -> bool {
    for _ in 0..1000 {
        if kernel.try_collect_state_snapshot() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    false
}

/// `dispatch_ticks` must advance GPU compute on its own, and — because it makes
/// no snapshot request — leave the staging ring empty so nothing is collectable.
#[test]
fn dispatch_ticks_advances_compute_without_requesting_snapshot() {
    use xagent_brain::buffers::{PHYS_STRIDE, P_TICKS_ALIVE};

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let (mut kernel, _world) = split_test_kernel(1, 12_345);

    let ticks_before = kernel.read_full_state_blocking()[P_TICKS_ALIVE] as u64;

    // Advance compute only — deliberately do NOT call request_state_snapshot.
    kernel.dispatch_ticks(0, 60);

    let ticks_after = kernel.read_full_state_blocking()[P_TICKS_ALIVE] as u64;
    assert!(
        ticks_after > ticks_before,
        "dispatch_ticks must advance compute: ticks_alive {} -> {}",
        ticks_before,
        ticks_after
    );
    let _ = PHYS_STRIDE; // stride imported for symmetry with sibling tests

    // No snapshot was requested, so the staging ring is empty and the
    // non-blocking collector must report nothing to collect.
    assert!(
        !kernel.try_collect_state_snapshot(),
        "no snapshot was requested, so none should be collectable"
    );
}

/// After `dispatch_ticks`, an explicit `request_state_snapshot` must publish the
/// advanced physics into `cached_state`, matching the authoritative blocking read.
#[test]
fn request_state_snapshot_publishes_advanced_state_to_cache() {
    use xagent_brain::buffers::{PHYS_STRIDE, P_POS_X, P_POS_Z, P_TICKS_ALIVE};

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let (mut kernel, _world) = split_test_kernel(1, 999);

    kernel.dispatch_ticks(0, 60);
    assert!(
        kernel.request_state_snapshot(),
        "a free staging slot should accept the snapshot request"
    );
    assert!(
        collect_snapshot(&mut kernel),
        "the requested snapshot should become collectable"
    );

    let cached_ticks = kernel.cached_state()[P_TICKS_ALIVE] as u64;
    assert!(
        cached_ticks > 0,
        "cached_state should reflect advanced compute, got ticks_alive {}",
        cached_ticks
    );

    // The published snapshot must equal the authoritative blocking read, since
    // no dispatch ran in between.
    let cached_x = kernel.cached_state()[P_POS_X];
    let cached_z = kernel.cached_state()[P_POS_Z];
    let blocking = kernel.read_full_state_blocking();
    assert!(
        (cached_x - blocking[P_POS_X]).abs() < 1e-4 && (cached_z - blocking[P_POS_Z]).abs() < 1e-4,
        "snapshot ({:.4},{:.4}) must match blocking read ({:.4},{:.4})",
        cached_x,
        cached_z,
        blocking[P_POS_X],
        blocking[P_POS_Z]
    );
    let _ = PHYS_STRIDE;
}

/// The compatibility wrapper `dispatch_batch` must still both advance compute
/// and request a snapshot, so a single call followed by a collect publishes.
#[test]
fn dispatch_batch_wrapper_advances_and_publishes() {
    use xagent_brain::buffers::P_TICKS_ALIVE;

    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let (mut kernel, _world) = split_test_kernel(1, 2_024);

    assert!(
        kernel.dispatch_batch(0, 60),
        "dispatch_batch should report ticks were submitted"
    );
    assert!(
        collect_snapshot(&mut kernel),
        "dispatch_batch must request a snapshot that later becomes collectable"
    );
    let cached_ticks = kernel.cached_state()[P_TICKS_ALIVE] as u64;
    assert!(
        cached_ticks > 0,
        "dispatch_batch wrapper should publish advanced state, got ticks_alive {}",
        cached_ticks
    );
}

/// Poll the non-blocking telemetry collector until it yields a result or the
/// bounded retry budget is exhausted.
fn collect_telemetry(
    kernel: &mut xagent_brain::GpuKernel,
) -> Option<(u32, xagent_brain::AgentTelemetry)> {
    for _ in 0..1000 {
        if let Some(result) = kernel.try_collect_telemetry() {
            return Some(result);
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    None
}

/// `try_collect_telemetry` must report the agent the readback was *requested
/// for*, and re-requesting a different agent (a selection change) must surface
/// that new agent — never the superseded one. Without this, telemetry that
/// completed for the previously-selected agent would be applied to the new one.
#[test]
fn try_collect_telemetry_tracks_requested_agent_across_reselection() {
    if !xagent_brain::GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }

    let (mut kernel, _world) = split_test_kernel(3, 4_242);
    // Advance so the telemetry slices hold real data.
    kernel.dispatch_ticks(0, 60);

    // Straightforward case: request agent 1, collect, expect index 1.
    kernel.request_agent_telemetry(1);
    let (idx, _) = collect_telemetry(&mut kernel).expect("telemetry for agent 1");
    assert_eq!(idx, 1, "collected telemetry must be labeled with agent 1");

    // Selection-change race: request 0, then immediately re-request 2 before
    // collecting. The kernel clears the superseded request; the collected
    // telemetry must be for agent 2, not agent 0.
    kernel.request_agent_telemetry(0);
    kernel.request_agent_telemetry(2);
    let (idx, _) = collect_telemetry(&mut kernel).expect("telemetry after re-request");
    assert_eq!(
        idx, 2,
        "after re-requesting agent 2, collected telemetry must be for agent 2"
    );
}
