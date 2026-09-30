//! The critic's recent-experience replay.
//!
//! Every brain tick is kept in a ring of the last `RECENT_CAP` moments,
//! regardless of outcome. Each moment gathers TD's own target as it unfolds:
//! the discounted TD reward of the next `REPLAY_RETURN_TICKS` brain ticks,
//! completed with the critic's value when it settles, or the terminal outcome
//! if the life ends first. One settled moment per brain tick steps the whole
//! value, bias included, toward its return.

use xagent_brain::buffers::{
    BRAIN_STRIDE, ENCODED_DIMENSION, O_PREV_VALUE, O_RECENT_KEYS, O_RECENT_NORMS, O_RECENT_RETURNS,
    O_RECENT_STATE, O_RECENT_TICKS, O_TICK_COUNT, O_VALUE_BIAS, O_VALUE_WEIGHTS, P_DEATH_COUNT,
    P_ENERGY, P_RAW_GRADIENT_OUT, P_URGENCY_OUT, RECENT_CAP, RECENT_OPEN, RECENT_SETTLED,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity at spawn.
const FULL_METER: f32 = 100.0;
/// Brain ticks run before a probe, so the ring holds settled moments.
const WARMUP_TICKS: u64 = 20;
/// Energy removed before the probe so the meal fits under the meter's cap.
const HUNGER_DEFICIT: f32 = 50.0;
/// Energy added in one step to simulate a meal.
const MEAL_ENERGY: f32 = 30.0;
/// Mirrors `TD_DISCOUNT` in `common.wgsl`.
const DISCOUNT: f32 = 0.97;
/// Mirrors `REPLAY_RETURN_TICKS` in `common.wgsl`.
const RETURN_TICKS: usize = 8;
/// Mirrors `TERMINAL_DEATH_TD_ERROR` in `common.wgsl`.
const TERMINAL_OUTCOME: f32 = -1.0;
/// Brain ticks recorded after the warmup: long enough for the first
/// recorded moment to settle, with later ones still open.
const RECORDED_TICKS: usize = 12;
/// Recorded tick on which the meal lands.
const MEAL_TICK: usize = 3;
/// Absorbs shader float rounding in the gathered returns.
const RETURN_TOLERANCE: f32 = 1e-5;
/// Return planted on every ring slot for the replay probe.
const PLANTED_RETURN: f32 = 0.5;
/// Age (brain ticks) given to the planted moments: long settled.
const PLANTED_AGE: f32 = 100.0;
/// Brain ticks the replay probe runs. Each replay of a planted moment closes
/// a tenth of the gap between its value and its return (CRITIC_REPLAY_RATE
/// 0.1, ‖key‖ = 1, bias and weight each moving half of it).
const REPLAY_TICKS: u64 = 20;
/// Share of the planted return the value must reach after the probe.
const MIN_REPLAYED_FRACTION: f32 = 0.5;

/// Stationary probe agent: every physics tick a brain tick. With no
/// predictive bonus, the TD reward is exactly the urgency-weighted
/// homeostatic change the physics state exports.
fn stationary_brain() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        homeo_predictive_credit_beta: 0.0,
        ..BrainConfig::default()
    }
}

/// One stationary agent on flat, hazard-free ground with its only food item
/// far out of reach.
fn stationary_kernel() -> GpuKernel {
    let brain = stationary_brain();
    let world = WorldConfig {
        seed: 3,
        ..WorldConfig::default()
    };
    let kernel = GpuKernel::new(1, 1, &brain, &world);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[(100.0, 0.0, 100.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    kernel
}

fn run_ticks(kernel: &mut GpuKernel, start: &mut u64, ticks: u64) {
    for _ in 0..ticks {
        kernel.dispatch_batch(*start, 1);
        *start += 1;
    }
}

fn set_energy(kernel: &mut GpuKernel, energy: f32) {
    kernel.write_agent_physics_fields(0, &[(P_ENERGY, energy)]);
}

/// Ring slot holding the moment stored at brain tick `created`.
fn slot_of(state: &[f32], created: f32) -> usize {
    (0..RECENT_CAP)
        .find(|&slot| state[O_RECENT_TICKS + slot] == created)
        .expect("the moment is in the ring")
}

#[test]
fn recent_moment_gathers_tds_own_return() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    set_energy(&mut kernel, FULL_METER - HUNGER_DEFICIT);
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    let start = kernel.read_agent_state(0).brain_state;
    assert_eq!(
        start.len(),
        BRAIN_STRIDE,
        "static offsets are only valid for the default vision layout"
    );
    let first = start[O_TICK_COUNT];

    // Record the TD reward and the critic's value of each following tick.
    let mut rewards = Vec::with_capacity(RECORDED_TICKS);
    let mut values = Vec::with_capacity(RECORDED_TICKS);
    for k in 0..RECORDED_TICKS {
        if k == MEAL_TICK {
            let energy = kernel.read_full_state_blocking()[P_ENERGY];
            set_energy(&mut kernel, energy + MEAL_ENERGY);
        }
        run_ticks(&mut kernel, &mut tick, 1);
        let physics = kernel.read_full_state_blocking();
        rewards.push(physics[P_RAW_GRADIENT_OUT] * (1.0 + physics[P_URGENCY_OUT]));
        values.push(kernel.read_agent_state(0).brain_state[O_PREV_VALUE]);
    }
    let state = kernel.read_agent_state(0).brain_state;

    // The moment stored just before the recording has settled with TD's
    // eight-tick return, the meal included, completed with the value then.
    let settled = slot_of(&state, first);
    let mut expected: f32 = (0..RETURN_TICKS)
        .map(|k| DISCOUNT.powi(k as i32) * rewards[k])
        .sum();
    expected += DISCOUNT.powi(RETURN_TICKS as i32) * values[RETURN_TICKS - 1];
    assert_eq!(state[O_RECENT_STATE + settled], RECENT_SETTLED);
    let gathered = state[O_RECENT_RETURNS + settled];
    assert!(
        (gathered - expected).abs() < RETURN_TOLERANCE,
        "settled moment gathered {gathered}, TD's own return is {expected}"
    );
    assert!(
        rewards[MEAL_TICK] > 0.0 && gathered > 0.0,
        "the meal should count toward the moment before it"
    );

    // A later moment is still open, with the rewards so far and no value yet.
    let open_age = RETURN_TICKS - 1;
    let open_created = first + (RECORDED_TICKS - open_age) as f32;
    let open = slot_of(&state, open_created);
    assert_eq!(state[O_RECENT_STATE + open], RECENT_OPEN);
    let offset = RECORDED_TICKS - open_age;
    let partial: f32 = (0..open_age)
        .map(|k| DISCOUNT.powi(k as i32) * rewards[offset + k])
        .sum();
    assert!(
        (state[O_RECENT_RETURNS + open] - partial).abs() < RETURN_TOLERANCE,
        "open moment gathered {}, expected {partial}",
        state[O_RECENT_RETURNS + open]
    );
}

#[test]
fn replay_steps_the_whole_value_toward_a_settled_return() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);

    // Fill the ring with one settled moment whose key is the first unit
    // vector, so the critic's value of it is bias + value_weights[0], and
    // start the critic from zero.
    let mut state = kernel.read_agent_state(0);
    let now = state.brain_state[O_TICK_COUNT];
    for d in 0..ENCODED_DIMENSION {
        state.brain_state[O_VALUE_WEIGHTS + d] = 0.0;
    }
    state.brain_state[O_VALUE_BIAS] = 0.0;
    for slot in 0..RECENT_CAP {
        for d in 0..ENCODED_DIMENSION {
            state.brain_state[O_RECENT_KEYS + slot * ENCODED_DIMENSION + d] =
                if d == 0 { 1.0 } else { 0.0 };
        }
        state.brain_state[O_RECENT_NORMS + slot] = 1.0;
        state.brain_state[O_RECENT_RETURNS + slot] = PLANTED_RETURN;
        state.brain_state[O_RECENT_TICKS + slot] = now - PLANTED_AGE;
        state.brain_state[O_RECENT_STATE + slot] = RECENT_SETTLED;
    }
    kernel.write_agent_state(0, &state);

    run_ticks(&mut kernel, &mut tick, REPLAY_TICKS);

    let after = kernel.read_agent_state(0).brain_state;
    let bias = after[O_VALUE_BIAS];
    let value = bias + after[O_VALUE_WEIGHTS];
    assert!(
        value > MIN_REPLAYED_FRACTION * PLANTED_RETURN && value <= PLANTED_RETURN,
        "the critic values the replayed moment at {value}; replay should move it toward \
         {PLANTED_RETURN} (without replay it stays near 0)"
    );
    assert!(
        bias > 0.0,
        "replay should move the bias with the weights, got bias {bias}"
    );
}

#[test]
fn death_settles_open_moments_with_the_terminal_outcome() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut tick = 0_u64;
    run_ticks(&mut kernel, &mut tick, WARMUP_TICKS);
    let before = kernel.read_agent_state(0).brain_state;
    let last_tick = before[O_TICK_COUNT];
    let open: Vec<usize> = (0..RECENT_CAP)
        .filter(|&slot| before[O_RECENT_STATE + slot] == RECENT_OPEN)
        .collect();
    assert_eq!(
        open.len(),
        RETURN_TICKS,
        "the last eight moments should still be gathering their returns"
    );

    set_energy(&mut kernel, 0.0);
    run_ticks(&mut kernel, &mut tick, 1);
    assert_eq!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT],
        1.0,
        "the probe agent should have died"
    );

    let after = kernel.read_agent_state(0).brain_state;
    for slot in open {
        let age = last_tick - before[O_RECENT_TICKS + slot];
        let expected = before[O_RECENT_RETURNS + slot] + DISCOUNT.powf(age) * TERMINAL_OUTCOME;
        assert_eq!(
            after[O_RECENT_STATE + slot],
            RECENT_SETTLED,
            "death left the moment of age {age} open"
        );
        assert!(
            (after[O_RECENT_RETURNS + slot] - expected).abs() < RETURN_TOLERANCE,
            "moment of age {age} holds {}, expected {expected}",
            after[O_RECENT_RETURNS + slot]
        );
    }
}
