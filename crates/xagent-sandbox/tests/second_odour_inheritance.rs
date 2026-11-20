//! The `second_odour` block is learned-only: its turn weights and trace are
//! exactly zero in every birth brain and stay zero through steering mutation,
//! the steering population and whole-brain mutation.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use xagent_brain::buffers::{
    BrainLayout, ENCODED_DIMENSION, O_PREDICTOR_CONTEXT_WEIGHT, PREDICTOR_DIMENSION,
};
use xagent_brain::AgentBrainState;
use xagent_sandbox::agent::{
    fresh_brain_state, mutate_brain_state_seeded, mutate_steering_weights, steering_population,
};
use xagent_shared::BrainConfig;

const BIRTH_SEED_BASE: u64 = 0x5EC0_0D00;
const STEERING_SEED_BASE: u64 = 0x5EC0_1D00;
const POPULATION_SEED: u64 = 0x5EC0_2D00;
const BRAIN_SEED_BASE: u64 = 0x5EC0_3D00;
const SEED_COUNT: u64 = 8;
const STRENGTHS: [f32; 2] = [0.1, 0.5];
const BRAIN_STRENGTH: f32 = 0.5;
const POPULATION_AGENTS: usize = 12;
const POPULATION_GROUP: usize = 3;
const POPULATION_STRENGTH: f32 = 0.5;
const SECOND_ODOUR_BLOCK: usize = 2;
const TURN_WIDTH: usize = 2;

fn config() -> BrainConfig {
    BrainConfig {
        second_odour_sense_enabled: true,
        ..BrainConfig::default()
    }
}

fn tail_base(layout: &BrainLayout) -> usize {
    let tail = layout.feature_count * ENCODED_DIMENSION
        + ENCODED_DIMENSION
        + PREDICTOR_DIMENSION * ENCODED_DIMENSION;
    tail - O_PREDICTOR_CONTEXT_WEIGHT
}

fn second_odour_block(state: &AgentBrainState, layout: &BrainLayout) -> Vec<f32> {
    let block = layout.sense_blocks()[SECOND_ODOUR_BLOCK];
    let base = tail_base(layout);
    let mut out = Vec::new();
    for i in 0..TURN_WIDTH {
        out.push(state.brain_state[base + block.turn_weights + i]);
    }
    for i in 0..TURN_WIDTH {
        out.push(state.brain_state[base + block.turn_trace + i]);
    }
    out
}

fn other_turn_weights(state: &AgentBrainState, layout: &BrainLayout) -> Vec<f32> {
    let base = tail_base(layout);
    let mut out = Vec::new();
    for block in &layout.sense_blocks()[..SECOND_ODOUR_BLOCK] {
        for i in 0..block.channels {
            out.push(state.brain_state[base + block.turn_weights + i]);
        }
    }
    out
}

fn assert_zero(state: &AgentBrainState, layout: &BrainLayout) {
    let block = second_odour_block(state, layout);
    assert!(block.iter().all(|w| *w == 0.0), "block was {block:?}");
}

#[test]
fn second_odour_birth_brain_is_zero() {
    let config = config();
    let layout = BrainLayout::from_config(&config);
    for seed in 0..SEED_COUNT {
        let mut rng = SmallRng::seed_from_u64(BIRTH_SEED_BASE + seed);
        let state = fresh_brain_state(&config, &mut rng);
        assert_zero(&state, &layout);
    }
}

#[test]
fn second_odour_survives_steering_mutation() {
    let config = config();
    let layout = BrainLayout::from_config(&config);
    for seed in 0..SEED_COUNT {
        let mut rng = SmallRng::seed_from_u64(STEERING_SEED_BASE + seed);
        let parent = fresh_brain_state(&config, &mut rng);
        let before = other_turn_weights(&parent, &layout);
        for strength in STRENGTHS {
            let child = mutate_steering_weights(&parent, strength, &mut rng);
            assert_zero(&child, &layout);
            assert_ne!(other_turn_weights(&child, &layout), before);
        }
    }
}

#[test]
fn second_odour_survives_population_and_brain_mutation() {
    let config = config();
    let layout = BrainLayout::from_config(&config);
    let mut rng = SmallRng::seed_from_u64(POPULATION_SEED);
    let template = fresh_brain_state(&config, &mut rng);
    let population = steering_population(
        &template,
        POPULATION_AGENTS,
        POPULATION_GROUP,
        POPULATION_STRENGTH,
        &mut rng,
    );
    assert_eq!(population.len(), POPULATION_AGENTS);
    for genome in &population {
        assert_zero(genome, &layout);
    }
    for seed in 0..SEED_COUNT {
        let mutated = mutate_brain_state_seeded(&template, BRAIN_STRENGTH, BRAIN_SEED_BASE + seed);
        assert_zero(&mutated, &layout);
    }
}
