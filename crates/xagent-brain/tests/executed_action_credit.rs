//! The actors' eligibility traces credit the exploration noise as the body
//! executed it: scaled by the fatigue factor and, for turning, by the
//! klinotaxis factor, exactly as the motor command was. Crediting the full
//! draw credited actions the body only partly carried out.

use xagent_brain::buffers::{
    ENCODED_DIMENSION, O_HOMEO, O_RECENT_KEYS, O_TICK_COUNT, O_TRACE_TURN, O_TURN_NOISE,
    P_DEATH_COUNT, P_EXPLORATION_RATE_OUT, P_FATIGUE_FACTOR_OUT, RECENT_CAP,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity at spawn.
const FULL_METER: f32 = 100.0;
/// Brain ticks observed: enough for the moving agent to be slowed by fatigue
/// on a good share of them.
const OBSERVED_TICKS: u64 = 300;
/// Mirrors `TD_DISCOUNT * TD_LAMBDA` in `common.wgsl`.
const TRACE_DECAY: f32 = 0.97 * 0.9;
/// Mirrors `KLINOTAXIS_SENSITIVITY` in `common.wgsl`.
const KLINOTAXIS_SENSITIVITY: f32 = 500.0;
/// Mirrors the klinotaxis factor's clamp in `brain_passes.wgsl`.
const KLINOTAXIS_MIN: f32 = 0.3;
const KLINOTAXIS_MAX: f32 = 3.0;
/// Fatigue below this counts as a damped tick, where the executed and the
/// drawn noise clearly differ.
const DAMPED_FATIGUE: f32 = 0.95;
/// Damped ticks the probe must see for the check to discriminate.
const MIN_DAMPED_TICKS: usize = 20;
/// Absorbs shader float rounding in the trace increment.
const TRACE_TOLERANCE: f32 = 1e-4;

#[test]
fn turn_trace_credits_the_executed_noise() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: 4,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, 11);
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

    let mut previous_trace: Option<Vec<f32>> = None;
    let mut damped = 0;
    for tick in 0..OBSERVED_TICKS {
        kernel.dispatch_batch(tick, 1);
        let physics = kernel.read_full_state_blocking().to_vec();
        let state = kernel.read_agent_state(0).brain_state;
        let trace = state[O_TRACE_TURN..O_TRACE_TURN + ENCODED_DIMENSION].to_vec();
        assert_eq!(physics[P_DEATH_COUNT], 0.0, "the probe agent must not die");
        if let Some(previous) = previous_trace {
            let fatigue = physics[P_FATIGUE_FACTOR_OUT];
            let deviation = state[O_HOMEO] - state[O_HOMEO + 1];
            let klinotaxis =
                (1.0 - deviation * KLINOTAXIS_SENSITIVITY).clamp(KLINOTAXIS_MIN, KLINOTAXIS_MAX);
            let executed_noise =
                state[O_TURN_NOISE] * physics[P_EXPLORATION_RATE_OUT] * fatigue * klinotaxis;
            // This tick's key is the one the recent-experience ring stored.
            let slot = state[O_TICK_COUNT] as usize % RECENT_CAP;
            let key = &state[O_RECENT_KEYS + slot * ENCODED_DIMENSION..][..ENCODED_DIMENSION];
            for d in 0..ENCODED_DIMENSION {
                let increment = trace[d] - TRACE_DECAY * previous[d];
                let expected = executed_noise * key[d];
                assert!(
                    (increment - expected).abs() < TRACE_TOLERANCE,
                    "tick {tick}, dim {d}: turn trace grew by {increment}, executed noise × key \
                     is {expected} (fatigue {fatigue}, klinotaxis {klinotaxis})"
                );
            }
            if fatigue < DAMPED_FATIGUE {
                damped += 1;
            }
        }
        previous_trace = Some(trace);
    }
    assert!(
        damped >= MIN_DAMPED_TICKS,
        "only {damped} ticks were damped by fatigue; the probe must exercise the scaling"
    );
}
