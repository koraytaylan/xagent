//! Byte-level regression baseline for the sense/motor block.
//!
//! Each scenario runs seeded behaviour, hashes the resulting bytes with
//! FNV-1a 64-bit and compares the hash with the line of the same name in
//! `tests/fixtures/sense_block_parity.golden`. The comparison is exact: any
//! changed byte in any hashed state fails the scenario.
//!
//! Scenarios:
//!
//! - `steering_population` (CPU): a seeded fresh brain state and the steering
//!   population drawn from it; every returned `brain_state` is hashed.
//! - `trajectory_raw_vision` and `trajectory_cortex` (GPU): a seeded
//!   ten-agent world run for 3000 ticks with the visual cortex off and on.
//!   The full physics state and every agent's brain state and patterns are
//!   hashed.
//! - `headless_run` and `champion_replay` (GPU): a seeded two-generation
//!   headless evolution run, hashed from its stored brains and node fitness,
//!   and a 2000-tick replay of the lowest-id stored champion from the
//!   committed database fixture `tests/fixtures/sense_block_champion.db`.
//!
//! No separate route scenario exists: the critic reads the encoding by default
//! and the forward pathway route is off by default, and neither alternative
//! route setting passed its evaluation, so the default trajectories cover the
//! routes in use.
//!
//! Modes: compare is the default. With `XAGENT_RECORD_SENSE_PARITY=1` a
//! scenario rewrites only its own golden line (creating the file when absent)
//! and does not compare; the headless scenario also rewrites the champion
//! fixture. Record with `--test-threads=1`.
//!
//! The GPU hashes are specific to the host and adapter that recorded them.
//! The GPU scenarios are ignored by default and run only with
//! `--include-ignored` on that host; CI runs only `steering_population`.

use std::fs;
use std::path::{Path, PathBuf};

use rand::SeedableRng;
use rusqlite::Connection;
use xagent_brain::buffers::PHYS_STRIDE;
use xagent_brain::buffers::P_DEATH_COUNT;
use xagent_brain::{AgentBrainState, BrainLayout, GpuKernel};
use xagent_sandbox::agent::{fresh_brain_state, steering_population};
use xagent_sandbox::governor::Governor;
use xagent_sandbox::headless;
use xagent_sandbox::world::WorldState;
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, WorldConfig};

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Environment variable that switches the test into record mode.
const RECORD_VARIABLE: &str = "XAGENT_RECORD_SENSE_PARITY";
/// Golden file, relative to the crate root.
const GOLDEN_PATH: &str = "tests/fixtures/sense_block_parity.golden";
/// Champion database fixture, relative to the crate root.
const CHAMPION_FIXTURE_PATH: &str = "tests/fixtures/sense_block_champion.db";
/// Largest accepted champion fixture size in bytes (2 MB).
const CHAMPION_FIXTURE_MAX_BYTES: u64 = 2 * 1024 * 1024;

/// SQLite busy timeout for every connection the test opens.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Seed of the brain RNG for the steering population and the GPU kernel.
const BRAIN_SEED: u64 = 12345;
/// World seed of every GPU scenario.
const WORLD_SEED: u64 = 5;
/// Steering population size.
const STEERING_POPULATION_SIZE: usize = 10;
/// Number of steering-weight variants drawn into the population.
const STEERING_VARIANTS: usize = 2;
/// Steering perturbation scale.
const STEERING_SCALE: f32 = 0.3;

/// Agents in every GPU kernel scenario.
const AGENTS: u32 = 10;
/// Ticks of each trajectory scenario.
const TRAJECTORY_TICKS: u32 = 3000;
/// Ticks of the champion replay.
const REPLAY_TICKS: u32 = 2000;
/// Ticks per `dispatch_ticks` call; the kernel is drained after each chunk.
const CHUNK_TICKS: u32 = 100;
/// Energy and integrity capacity of every agent but the first.
const AGENT_CAPACITY: f32 = 100.0;
/// Energy and integrity capacity of agent 0, low enough that it dies.
const DYING_AGENT_CAPACITY: f32 = 1.0;

/// Headless population size.
const HEADLESS_POPULATION: usize = 4;
/// Headless ticks per generation.
const HEADLESS_TICK_BUDGET: u64 = 2000;
/// Headless generation count.
const HEADLESS_GENERATIONS: u64 = 2;

/// Folds `bytes` into an FNV-1a 64-bit hash state.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hashes `bytes` from the FNV-1a offset basis.
fn hash_bytes(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET_BASIS, bytes)
}

/// Little-endian bytes of `values`.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn record_mode() -> bool {
    std::env::var(RECORD_VARIABLE).is_ok_and(|value| value == "1")
}

fn require_gpu() {
    assert!(
        GpuKernel::is_available(),
        "this scenario requires a GPU or software Vulkan adapter"
    );
}

/// In record mode rewrites the scenario's golden line; otherwise asserts the
/// hash equals the golden line, failing when the file or line is absent.
fn check_golden(scenario: &str, hash: u64) {
    let path = crate_path(GOLDEN_PATH);
    let line = format!("{scenario} {hash:016x}");
    if record_mode() {
        let existing = fs::read_to_string(&path).unwrap_or_default();
        let mut lines: Vec<String> = existing
            .lines()
            .filter(|l| l.split_whitespace().next() != Some(scenario))
            .map(str::to_owned)
            .collect();
        lines.push(line);
        lines.sort();
        let parent = path.parent().expect("golden path has a parent directory");
        fs::create_dir_all(parent).expect("create the fixtures directory");
        fs::write(&path, lines.join("\n") + "\n").expect("write the golden file");
        return;
    }
    let golden = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("golden file {} is unreadable: {e}", path.display()));
    let recorded = golden
        .lines()
        .find_map(|l| {
            let mut parts = l.split_whitespace();
            (parts.next() == Some(scenario)).then(|| parts.next().unwrap_or("").to_owned())
        })
        .unwrap_or_else(|| panic!("golden file has no line for scenario {scenario}"));
    assert_eq!(
        recorded,
        format!("{hash:016x}"),
        "scenario {scenario} does not match its golden hash"
    );
}

fn world_config() -> WorldConfig {
    WorldConfig {
        seed: WORLD_SEED,
        ..Default::default()
    }
}

/// Builds a seeded kernel over a fresh world with agent 0 given a capacity
/// low enough to die.
fn seeded_kernel(brain: &BrainConfig) -> GpuKernel {
    let world_config = world_config();
    let world = WorldState::new(world_config.clone());
    let biomes = world.biome_map.grid_as_u32();
    let food_positions: Vec<(f32, f32, f32)> = world
        .food_items
        .iter()
        .map(|f| (f.position.x, f.position.y, f.position.z))
        .collect();
    let consumed: Vec<bool> = world.food_items.iter().map(|f| f.consumed).collect();
    let timers: Vec<f32> = world.food_items.iter().map(|f| f.respawn_timer).collect();
    let agents: Vec<_> = (0..AGENTS)
        .map(|index| {
            let capacity = if index == 0 {
                DYING_AGENT_CAPACITY
            } else {
                AGENT_CAPACITY
            };
            (
                world.safe_spawn_position(),
                capacity,
                capacity,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    let mut kernel = GpuKernel::new(AGENTS, world.food_items.len(), brain, &world_config);
    kernel.reset_agents_seeded(brain, BRAIN_SEED);
    kernel.upload_world(
        &world.terrain.heights,
        &biomes,
        &food_positions,
        &consumed,
        &timers,
    );
    kernel.upload_agents(&agents);
    kernel
}

fn run_chunked(kernel: &mut GpuKernel, ticks: u32) {
    for chunk in 0..ticks / CHUNK_TICKS {
        kernel.dispatch_ticks(u64::from(chunk * CHUNK_TICKS), CHUNK_TICKS);
        kernel.poll_wait();
    }
}

/// Full physics state plus every agent's brain state and patterns.
fn kernel_state_bytes(kernel: &mut GpuKernel) -> Vec<u8> {
    let mut bytes = f32_bytes(kernel.read_full_state_blocking());
    for agent in 0..AGENTS {
        let state = kernel.read_agent_state(agent);
        bytes.extend(f32_bytes(&state.brain_state));
        bytes.extend(f32_bytes(&state.patterns));
    }
    bytes
}

fn trajectory_hash(brain: &BrainConfig) -> u64 {
    let mut kernel = seeded_kernel(brain);
    run_chunked(&mut kernel, TRAJECTORY_TICKS);
    let physics = kernel.read_full_state_blocking();
    let deaths = physics
        .get(P_DEATH_COUNT)
        .copied()
        .expect("physics state holds agent 0's death counter");
    assert!(
        deaths > 0.0,
        "agent 0 must die at least once (death counter {deaths}, stride {PHYS_STRIDE})"
    );
    hash_bytes(&kernel_state_bytes(&mut kernel))
}

/// A seeded steering population hashes to its golden line.
#[test]
fn steering_population_matches_golden() {
    let config = BrainConfig::default();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(BRAIN_SEED);
    let template = fresh_brain_state(&config, &mut rng);
    let population = steering_population(
        &template,
        STEERING_POPULATION_SIZE,
        STEERING_VARIANTS,
        STEERING_SCALE,
        &mut rng,
    );
    assert_eq!(population.len(), STEERING_POPULATION_SIZE);
    let bytes: Vec<u8> = population
        .iter()
        .flat_map(|state| f32_bytes(&state.brain_state))
        .collect();
    check_golden("steering_population", hash_bytes(&bytes));
}

/// A seeded ten-agent trajectory with the visual cortex off kills agent 0 and
/// hashes to its golden line.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn raw_vision_trajectory_matches_golden() {
    require_gpu();
    let brain = BrainConfig {
        visual_cortex_enabled: false,
        ..Default::default()
    };
    check_golden("trajectory_raw_vision", trajectory_hash(&brain));
}

/// A seeded ten-agent trajectory with the visual cortex on kills agent 0 and
/// hashes to its golden line.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn cortex_trajectory_matches_golden() {
    require_gpu();
    let brain = BrainConfig {
        visual_cortex_enabled: true,
        ..Default::default()
    };
    check_golden("trajectory_cortex", trajectory_hash(&brain));
}

/// Hash of `node_brain` rows by `node_id` then `node` rows by `id`.
fn headless_hash(connection: &Connection) -> u64 {
    let mut bytes = Vec::new();
    let mut brains = connection
        .prepare("SELECT node_id, data FROM node_brain ORDER BY node_id")
        .expect("query node_brain");
    let rows = brains
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
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
    assert!(!bytes.is_empty(), "the headless run stored no nodes");
    hash_bytes(&bytes)
}

/// Strips recordings and wall-clock run metadata (`run.started_at`,
/// `run.wall_time_secs`, `node.created_at`) from `source`, so re-recording is
/// byte-identical, then compacts it and copies it to the fixture.
fn record_champion_fixture(source: &Path) {
    let connection = Connection::open(source).expect("open the headless database");
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .expect("set busy timeout");
    connection
        .execute_batch(
            "DELETE FROM generation_recording; \
             UPDATE run SET started_at = NULL, wall_time_secs = 0; \
             UPDATE node SET created_at = NULL; \
             PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE; VACUUM;",
        )
        .expect("compact the headless database");
    drop(connection);
    let fixture = crate_path(CHAMPION_FIXTURE_PATH);
    fs::copy(source, &fixture).expect("copy the champion fixture");
}

/// Lowest node id with a stored champion brain in `governor`'s database.
fn lowest_champion(governor: &Governor, db_path: &Path) -> AgentBrainState {
    let connection = Connection::open(db_path).expect("open the champion fixture copy");
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .expect("set busy timeout");
    let mut statement = connection
        .prepare("SELECT id FROM node ORDER BY id")
        .expect("query node ids");
    let ids: Vec<i64> = statement
        .query_map([], |row| row.get::<_, i64>(0))
        .expect("read node ids")
        .map(|id| id.expect("node id"))
        .collect();
    ids.into_iter()
        .find_map(|id| governor.champion_brain(id))
        .expect("the champion fixture holds a champion brain")
}

/// A seeded headless evolution run hashes to the `headless_run` line, and the
/// lowest-id champion of the committed fixture has the default brain stride
/// and replays for 2000 ticks to the `champion_replay` line.
#[test]
#[ignore = "requires a GPU; run explicitly with --include-ignored"]
fn headless_run_and_champion_replay_match_golden() {
    require_gpu();
    let run_dir = tempfile::tempdir().expect("temp dir");
    let db = run_dir.path().join("headless.db");
    let db_path = db.to_str().expect("utf-8 temp path");
    let config = FullConfig {
        brain: BrainConfig::default(),
        world: world_config(),
        governor: GovernorConfig {
            population_size: HEADLESS_POPULATION,
            tick_budget: HEADLESS_TICK_BUDGET,
            max_generations: HEADLESS_GENERATIONS,
            ..Default::default()
        },
    };
    headless::run_headless(config, db_path, false, true);
    let connection = Connection::open(&db).expect("open the headless database");
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .expect("set busy timeout");
    check_golden("headless_run", headless_hash(&connection));
    drop(connection);
    if record_mode() {
        record_champion_fixture(&db);
    }

    let fixture = crate_path(CHAMPION_FIXTURE_PATH);
    let size = fs::metadata(&fixture)
        .expect("champion fixture is present")
        .len();
    assert!(
        size < CHAMPION_FIXTURE_MAX_BYTES,
        "champion fixture is {size} bytes, limit {CHAMPION_FIXTURE_MAX_BYTES}"
    );
    let replay_dir = tempfile::tempdir().expect("temp dir");
    let replay_db = replay_dir.path().join("champion.db");
    fs::copy(&fixture, &replay_db).expect("copy the champion fixture");
    let governor = Governor::resume(replay_db.to_str().expect("utf-8 temp path"))
        .expect("resume the champion fixture");
    let champion = lowest_champion(&governor, &replay_db);
    let brain = BrainConfig::default();
    assert_eq!(
        champion.brain_state.len(),
        BrainLayout::from_config(&brain).brain_stride,
        "champion brain state length differs from the default brain stride"
    );
    let mut kernel = seeded_kernel(&brain);
    for agent in 0..AGENTS {
        kernel.write_agent_state(agent, &champion);
    }
    run_chunked(&mut kernel, REPLAY_TICKS);
    check_golden(
        "champion_replay",
        hash_bytes(&kernel_state_bytes(&mut kernel)),
    );
}
