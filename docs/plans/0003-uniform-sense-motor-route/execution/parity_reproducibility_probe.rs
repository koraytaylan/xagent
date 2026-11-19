//! Feasibility probe for the sense-block parity test: on this host a seeded
//! steering population, a seeded multi-agent GPU trajectory and a seeded
//! headless evolution run each reproduce byte for byte, so a golden captured
//! before a refactor can prove the refactor changed nothing.

use rand::SeedableRng;
use xagent_brain::GpuKernel;
use xagent_sandbox::agent::{fresh_brain_state, steering_population};
use xagent_sandbox::headless;
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, WorldConfig};

const AGENTS: u32 = 10;
const TICKS: u32 = 3000;
/// Ticks per `dispatch_ticks` call; the kernel is drained after each chunk so
/// no long queue of submits builds up on a shared GPU.
const CHUNK_TICKS: u32 = 100;
const WORLD_SEED: u64 = 5;
const BRAIN_SEED: u64 = 12345;

fn require_gpu() {
    assert!(
        GpuKernel::is_available(),
        "the parity probe needs a GPU or lavapipe adapter"
    );
}

fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn trajectory(brain: &BrainConfig) -> Vec<u8> {
    let world_config = WorldConfig {
        seed: WORLD_SEED,
        ..Default::default()
    };
    let world = WorldState::new(world_config.clone());
    let biomes = world.biome_map.grid_as_u32();
    let food_pos: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    let agents: Vec<_> = (0..AGENTS)
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
    let mut kernel = GpuKernel::new(AGENTS, world.food_items.len(), brain, &world_config);
    kernel.reset_agents_seeded(brain, BRAIN_SEED);
    kernel.upload_world(&world.terrain.heights, &biomes, &food_pos, &consumed, &timers);
    kernel.upload_agents(&agents);
    for chunk in 0..TICKS / CHUNK_TICKS {
        kernel.dispatch_ticks(u64::from(chunk * CHUNK_TICKS), CHUNK_TICKS);
        kernel.poll_wait();
    }
    let mut bytes = as_bytes(kernel.read_full_state_blocking());
    for agent in 0..AGENTS {
        let state = kernel.read_agent_state(agent);
        bytes.extend(as_bytes(&state.brain_state));
        bytes.extend(as_bytes(&state.patterns));
    }
    bytes
}

fn headless_outcome(dir: &std::path::Path, name: &str) -> Vec<u8> {
    let db = dir.join(name);
    let db_path = db.to_str().expect("utf-8 temp path");
    let config = FullConfig {
        brain: BrainConfig::default(),
        world: WorldConfig {
            seed: WORLD_SEED,
            ..Default::default()
        },
        governor: GovernorConfig {
            population_size: 4,
            tick_budget: 2000,
            max_generations: 2,
            ..Default::default()
        },
    };
    headless::run_headless(config, db_path, false, true);
    let connection = rusqlite::Connection::open(db_path).expect("open headless db");
    let mut bytes = Vec::new();
    let mut brains = connection
        .prepare("SELECT node_id, data FROM node_brain ORDER BY node_id")
        .expect("query node_brain");
    let rows = brains
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)))
        .expect("read node_brain");
    for row in rows {
        let (node, data) = row.expect("node_brain row");
        bytes.extend(node.to_le_bytes());
        bytes.extend(data);
    }
    let mut nodes = connection
        .prepare("SELECT id, best_fitness, avg_fitness FROM node ORDER BY id")
        .expect("query node");
    let rows = nodes
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<f64>>(1)?,
                row.get::<_, Option<f64>>(2)?,
            ))
        })
        .expect("read node");
    for row in rows {
        let (id, best, avg) = row.expect("node row");
        bytes.extend(id.to_le_bytes());
        bytes.extend(best.unwrap_or(f64::NAN).to_le_bytes());
        bytes.extend(avg.unwrap_or(f64::NAN).to_le_bytes());
    }
    bytes
}

#[test]
fn seeded_steering_population_is_reproducible() {
    let config = BrainConfig::default();
    let draw = || {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(BRAIN_SEED);
        let template = fresh_brain_state(&config, &mut rng);
        steering_population(&template, 10, 2, 0.3, &mut rng)
            .into_iter()
            .flat_map(|state| as_bytes(&state.brain_state))
            .collect::<Vec<u8>>()
    };
    assert_eq!(draw(), draw());
}

#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn seeded_gpu_trajectory_is_reproducible() {
    require_gpu();
    for cortex in [false, true] {
        let brain = BrainConfig {
            visual_cortex_enabled: cortex,
            ..Default::default()
        };
        let first = trajectory(&brain);
        let second = trajectory(&brain);
        assert!(!first.is_empty());
        assert!(
            first == second,
            "seeded {AGENTS}-agent trajectory diverged between runs (cortex {cortex})"
        );
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn seeded_headless_run_is_reproducible() {
    require_gpu();
    let dir = tempfile::tempdir().expect("temp dir");
    let first = headless_outcome(dir.path(), "first.db");
    let second = headless_outcome(dir.path(), "second.db");
    assert!(first.len() > 16, "the headless run stored no champion brain");
    assert!(first == second, "seeded headless run diverged between runs");
}
