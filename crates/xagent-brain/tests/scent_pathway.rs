//! The whitened smell pathway to the turn policy. Each brain tick the two
//! nostrils are centred on their running mean and whitened by their running
//! covariance (C^(−1/2)); the statistics then move toward the new reading;
//! and the two scent turn weights' traces gather the turn noise's fresh
//! innovation times the whitened scent, normalised by 1 + |whitened|².

use xagent_brain::buffers::{
    MAX_TOUCH_CONTACTS, O_HOMEO, O_SCENT_COVARIANCE, O_SCENT_MEAN, O_SCENT_WHITENED, O_TRACE_SCENT,
    O_TURN_NOISE, P_EXPLORATION_RATE_OUT, P_FATIGUE_FACTOR_OUT,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
/// Ticks run first, so the nostrils smell the food and the statistics move.
const WARMUP_TICKS: u64 = 20;
/// Mirrors `SCENT_WHITENING_RATE` in `common.wgsl`.
const WHITENING_RATE: f32 = 0.01;
/// Mirrors `SCENT_EIGEN_FLOOR` in `common.wgsl`.
const EIGEN_FLOOR: f32 = 1e-8;
/// Mirrors `TD_DISCOUNT * TD_LAMBDA`.
const TRACE_DECAY: f32 = 0.97 * 0.9;
/// Mirrors `TURN_NOISE_PERSISTENCE`.
const NOISE_PERSISTENCE: f32 = 0.9;
/// Mirrors `KLINOTAXIS_SENSITIVITY` and the klinotaxis clamp.
const KLINOTAXIS_SENSITIVITY: f32 = 500.0;
const KLINOTAXIS_MIN: f32 = 0.3;
const KLINOTAXIS_MAX: f32 = 3.0;
/// Index of the left nostril in the non-visual telemetry tail.
const SCENT_SLOT: usize = 3 + 3 + 1 + 4 + MAX_TOUCH_CONTACTS * 4;
/// Relative tolerance for the whitened scent (it is divided by the square
/// root of a small eigenvalue) and absolute tolerance for the rest.
const RELATIVE_TOLERANCE: f32 = 1e-3;
const ABSOLUTE_TOLERANCE: f32 = 1e-6;

/// CPU mirror of `whiten_scent` in `common.wgsl`.
fn whiten(centred: [f32; 2], covariance: [f32; 3]) -> [f32; 2] {
    let [a, b, c] = covariance;
    let half_trace = 0.5 * (a + c);
    let radius = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let major = (half_trace + radius).max(EIGEN_FLOOR);
    let minor = (half_trace - radius).max(EIGEN_FLOOR);
    let mut axis = if a >= c { [1.0, 0.0] } else { [0.0, 1.0] };
    if b.abs() > 1e-12 {
        axis = [major - c, b];
    }
    let length = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt().max(1e-30);
    let unit = [axis[0] / length, axis[1] / length];
    let along_major = (unit[0] * centred[0] + unit[1] * centred[1]) / major.sqrt();
    let along_minor = (-unit[1] * centred[0] + unit[0] * centred[1]) / minor.sqrt();
    [
        unit[0] * along_major - unit[1] * along_minor,
        unit[1] * along_major + unit[0] * along_minor,
    ]
}

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected.abs()
}

#[test]
fn scent_is_whitened_and_credited_with_the_fresh_turn_innovation() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 5,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, 13);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    // Food ahead and to the left, within smelling range but out of reach.
    kernel.upload_world(&heights, &biomes, &[(-4.0, 0.35, 6.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..WARMUP_TICKS {
        kernel.dispatch_batch(tick, 1);
    }

    // The next brain tick reads the scent the last vision pass wrote.
    let telemetry = kernel.read_agent_telemetry_blocking(0);
    let scent = [
        telemetry.sensory_non_visual[SCENT_SLOT],
        telemetry.sensory_non_visual[SCENT_SLOT + 1],
    ];
    assert!(
        scent[0] > 0.0 && scent[1] > 0.0,
        "the probe food should be smelt"
    );
    let before = kernel.read_agent_state(0).brain_state;
    kernel.dispatch_batch(WARMUP_TICKS, 1);
    let after = kernel.read_agent_state(0).brain_state;
    let physics = kernel.read_full_state_blocking();

    let mean = [before[O_SCENT_MEAN], before[O_SCENT_MEAN + 1]];
    let covariance = [
        before[O_SCENT_COVARIANCE],
        before[O_SCENT_COVARIANCE + 1],
        before[O_SCENT_COVARIANCE + 2],
    ];
    let centred = [scent[0] - mean[0], scent[1] - mean[1]];
    let whitened = whiten(centred, covariance);
    for k in 0..2 {
        assert!(
            close(after[O_SCENT_WHITENED + k], whitened[k]),
            "whitened nostril {k}: GPU {} vs CPU {}",
            after[O_SCENT_WHITENED + k],
            whitened[k]
        );
        let expected_mean = mean[k] + WHITENING_RATE * centred[k];
        assert!(
            close(after[O_SCENT_MEAN + k], expected_mean),
            "scent mean {k}"
        );
    }
    let products = [
        centred[0] * centred[0],
        centred[0] * centred[1],
        centred[1] * centred[1],
    ];
    for k in 0..3 {
        let expected = covariance[k] + WHITENING_RATE * (products[k] - covariance[k]);
        assert!(
            close(after[O_SCENT_COVARIANCE + k], expected),
            "scent covariance {k}: GPU {} vs {expected}",
            after[O_SCENT_COVARIANCE + k]
        );
    }

    // The trace gathers the turn noise's fresh innovation, as executed.
    let klinotaxis = (1.0 - (after[O_HOMEO] - after[O_HOMEO + 1]) * KLINOTAXIS_SENSITIVITY)
        .clamp(KLINOTAXIS_MIN, KLINOTAXIS_MAX);
    let innovation = (after[O_TURN_NOISE] - NOISE_PERSISTENCE * before[O_TURN_NOISE])
        * physics[P_EXPLORATION_RATE_OUT]
        * physics[P_FATIGUE_FACTOR_OUT]
        * klinotaxis;
    let whitened_sq = whitened[0] * whitened[0] + whitened[1] * whitened[1];
    for k in 0..2 {
        let expected = TRACE_DECAY * before[O_TRACE_SCENT + k]
            + innovation * whitened[k] / (1.0 + whitened_sq);
        assert!(
            close(after[O_TRACE_SCENT + k], expected),
            "scent trace {k}: GPU {} vs {expected}",
            after[O_TRACE_SCENT + k]
        );
    }
}
