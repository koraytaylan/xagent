//! Legacy brain decode, zero-at-birth pathway block and route-flag inheritance.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use xagent_brain::buffers::{
    feature_count_for_brain_stride, AgentBrainState, PATHWAY_ROUTE_SLOT_COUNT,
};
use xagent_sandbox::agent::{
    crossover_config, fresh_brain_state, mutate_brain_state_seeded, mutate_config_seeded,
};
use xagent_sandbox::governor::Governor;
use xagent_shared::{BrainConfig, CriticInput, GovernorConfig};

const NODE_ID: i64 = 1;
const SEEDS: u64 = 256;
/// RNG seed for the fresh brain.
const FRESH_BRAIN_SEED: u64 = 11;
/// Divisor that keeps only the leading fraction of a blob when truncating.
const TRUNCATED_FRACTION_DIVISOR: usize = 2;
/// Brain mutation strength used by the mutation test.
const MUTATION_STRENGTH_FOR_TEST: f32 = 0.1;
/// Governor population size for the test database.
const TEST_POPULATION_SIZE: usize = 10;
/// Governor tick budget for the test database.
const TEST_TICK_BUDGET: u64 = 100;
/// Governor elitism count for the test database.
const TEST_ELITISM_COUNT: usize = 3;
/// Governor patience for the test database.
const TEST_PATIENCE: u32 = 5;
/// Governor momentum decay for the test database.
const TEST_MOMENTUM_DECAY: f32 = 0.9;
const ALL_CRITIC_INPUTS: [CriticInput; 3] = [
    CriticInput::Encoding,
    CriticInput::Pathway,
    CriticInput::EncodingAndPathway,
];

fn governor_config() -> GovernorConfig {
    GovernorConfig {
        population_size: TEST_POPULATION_SIZE,
        tick_budget: TEST_TICK_BUDGET,
        elitism_count: TEST_ELITISM_COUNT,
        patience: TEST_PATIENCE,
        max_generations: 0,
        mutation_strength: MUTATION_STRENGTH_FOR_TEST,
        eval_repeats: 1,
        num_islands: 1,
        migration_interval: 0,
        momentum_decay: TEST_MOMENTUM_DECAY,
        ..GovernorConfig::default()
    }
}

fn open_governor(path: &str) -> Governor {
    Governor::new(path, governor_config(), &BrainConfig::default(), "{}").unwrap()
}

fn fresh() -> AgentBrainState {
    let mut rng = SmallRng::seed_from_u64(FRESH_BRAIN_SEED);
    fresh_brain_state(&BrainConfig::default(), &mut rng)
}

/// Range of the pathway block: it ends where the layout-sized sensory mean
/// (`feature_count` slots) begins, at the end of the state.
fn block_range(state: &AgentBrainState) -> std::ops::Range<usize> {
    let len = state.brain_state.len();
    let feature_count = feature_count_for_brain_stride(len).unwrap();
    let end = len - feature_count;
    end - PATHWAY_ROUTE_SLOT_COUNT..end
}

fn block(state: &AgentBrainState) -> &[f32] {
    &state.brain_state[block_range(state)]
}

#[test]
fn legacy_champion_loads_upgraded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gov.db");
    let governor = open_governor(path.to_str().unwrap());
    let current = fresh();
    let mut legacy = current.clone();
    let range = block_range(&current);
    legacy.brain_state.drain(range.clone());
    assert!(governor.store_champion_brain(NODE_ID, &legacy));
    let loaded = governor.champion_brain(NODE_ID).unwrap();
    assert_eq!(loaded.brain_state.len(), current.brain_state.len());
    assert!(block(&loaded).iter().all(|v| *v == 0.0));
    let mut without_block = loaded.brain_state.clone();
    without_block.drain(range);
    assert_eq!(without_block, legacy.brain_state);
    assert_eq!(loaded.patterns, legacy.patterns);
}

#[test]
fn current_length_blob_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gov.db");
    let governor = open_governor(path.to_str().unwrap());
    let current = fresh();
    assert!(governor.store_champion_brain(NODE_ID, &current));
    let loaded = governor.champion_brain(NODE_ID).unwrap();
    assert_eq!(loaded.brain_state, current.brain_state);
    assert_eq!(loaded.patterns, current.patterns);
}

#[test]
fn truncated_blob_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gov.db");
    let governor = open_governor(path.to_str().unwrap());
    assert!(governor.store_champion_brain(NODE_ID, &fresh()));
    let conn = rusqlite::Connection::open(&path).unwrap();
    let data: Vec<u8> = conn
        .query_row(
            "SELECT data FROM node_brain WHERE node_id = ?1",
            [NODE_ID],
            |row| row.get(0),
        )
        .unwrap();
    let truncated = &data[..data.len() / TRUNCATED_FRACTION_DIVISOR];
    conn.execute(
        "UPDATE node_brain SET data = ?1 WHERE node_id = ?2",
        rusqlite::params![truncated, NODE_ID],
    )
    .unwrap();
    drop(conn);
    assert!(governor.champion_brain(NODE_ID).is_none());
}

#[test]
fn fresh_brain_has_zero_pathway_block() {
    assert!(block(&fresh()).iter().all(|v| *v == 0.0));
}

#[test]
fn brain_mutation_keeps_block_zero() {
    let state = fresh();
    for seed in 0..SEEDS {
        let mutated = mutate_brain_state_seeded(&state, MUTATION_STRENGTH_FOR_TEST, seed);
        assert!(
            block(&mutated).iter().all(|v| *v == 0.0),
            "seed {seed} wrote the pathway block"
        );
    }
}

#[test]
fn config_inheritance_keeps_route_flags() {
    for critic_input in ALL_CRITIC_INPUTS {
        for forward in [false, true] {
            let parent = BrainConfig {
                critic_input,
                forward_pathway_enabled: forward,
                ..BrainConfig::default()
            };
            let other = BrainConfig {
                critic_input: CriticInput::Encoding,
                forward_pathway_enabled: !forward,
                ..BrainConfig::default()
            };
            for seed in 0..SEEDS {
                let child = mutate_config_seeded(&parent, seed);
                assert_eq!(child.critic_input, critic_input);
                assert_eq!(child.forward_pathway_enabled, forward);
                let crossed = crossover_config(&parent, &other);
                assert_eq!(crossed.critic_input, critic_input);
                assert_eq!(crossed.forward_pathway_enabled, forward);
            }
        }
    }
}
