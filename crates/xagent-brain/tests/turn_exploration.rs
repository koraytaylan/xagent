//! Turn exploration and the bias-free turn channel.
//!
//! The turn channel has no bias: a turn bias turns the same way in every
//! scene, so it could only ever learn a spin. Its exploration noise persists
//! across brain ticks (an AR(1) process with the variance of one uniform
//! draw), so an exploratory turn is held long enough to centre food.

use xagent_brain::buffers::{
    BRAIN_STRIDE, ENCODED_DIMENSION, O_ACTION_TURN_WEIGHTS, O_ACT_BIASES, O_TURN_NOISE,
    P_MOTOR_TURN_OUT,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity at spawn.
const FULL_METER: f32 = 100.0;
/// Brain ticks sampled: enough for the AR(1) statistics to settle (about
/// 250 effective independent samples at persistence 0.9).
const SAMPLE_TICKS: u64 = 5000;
/// Mirrors `TURN_NOISE_PERSISTENCE` in `common.wgsl`.
const PERSISTENCE: f64 = 0.9;
/// Tolerance on the measured lag-1 autocorrelation.
const AUTOCORRELATION_TOLERANCE: f64 = 0.05;
/// Variance of one uniform draw on [−0.5, 0.5].
const UNIFORM_DRAW_VARIANCE: f64 = 1.0 / 12.0;
/// Relative tolerance on the measured variance (its sampling error is ~9%).
const VARIANCE_TOLERANCE: f64 = 0.3;
/// A turn bias large enough to saturate the turn command if it were used.
const PLANTED_TURN_BIAS: f32 = 1.5;
/// Mean executed turn a used bias would produce is tanh(1.5) ≈ 0.9; without
/// one the zero-mean noise keeps the mean near 0.
const MAX_BIAS_FREE_MEAN_TURN: f64 = 0.2;

/// One stationary agent on flat, hazard-free ground; every physics tick is a
/// brain tick.
fn stationary_kernel() -> GpuKernel {
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 3,
        ..WorldConfig::default()
    };
    let kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.upload_world(
        &vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE],
        &vec![0_u32; BIOME_SIDE * BIOME_SIDE],
        &[(100.0, 0.0, 100.0)],
        &[false],
        &[0.0],
    );
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    kernel
}

#[test]
fn turn_noise_persists_with_the_variance_of_one_draw() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut noise = Vec::with_capacity(SAMPLE_TICKS as usize);
    for tick in 0..SAMPLE_TICKS {
        kernel.dispatch_batch(tick, 1);
        let state = kernel.read_agent_state(0).brain_state;
        assert_eq!(state.len(), BRAIN_STRIDE);
        noise.push(f64::from(state[O_TURN_NOISE]));
    }
    let n = noise.len() as f64;
    let mean = noise.iter().sum::<f64>() / n;
    let variance = noise.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let lag1 = noise
        .windows(2)
        .map(|pair| (pair[0] - mean) * (pair[1] - mean))
        .sum::<f64>()
        / (n - 1.0)
        / variance;
    assert!(
        (lag1 - PERSISTENCE).abs() < AUTOCORRELATION_TOLERANCE,
        "turn noise lag-1 autocorrelation {lag1:.3}, expected {PERSISTENCE} \
         (independent draws would give ~0)"
    );
    assert!(
        (variance / UNIFORM_DRAW_VARIANCE - 1.0).abs() < VARIANCE_TOLERANCE,
        "turn noise variance {variance:.4}, expected one uniform draw's {UNIFORM_DRAW_VARIANCE:.4}"
    );
}

#[test]
fn turn_channel_ignores_and_never_learns_a_bias() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = stationary_kernel();
    let mut state = kernel.read_agent_state(0);
    for d in 0..ENCODED_DIMENSION {
        state.brain_state[O_ACTION_TURN_WEIGHTS + d] = 0.0;
    }
    state.brain_state[O_ACT_BIASES + 1] = PLANTED_TURN_BIAS;
    kernel.write_agent_state(0, &state);

    let mut executed = 0.0_f64;
    for tick in 0..SAMPLE_TICKS {
        kernel.dispatch_batch(tick, 1);
        // Agent 0's physics row starts at 0.
        executed += f64::from(kernel.read_full_state_blocking()[P_MOTOR_TURN_OUT]);
    }
    let mean_turn = executed / SAMPLE_TICKS as f64;
    assert!(
        mean_turn.abs() < MAX_BIAS_FREE_MEAN_TURN,
        "mean executed turn {mean_turn:.3}: the planted turn bias steered the agent"
    );
    let after = kernel.read_agent_state(0).brain_state;
    assert_eq!(
        after[O_ACT_BIASES + 1],
        PLANTED_TURN_BIAS,
        "TD learning moved the unused turn-bias slot"
    );
}
