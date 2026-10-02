//! The whitened visual pathway to the turn policy. Each brain tick the left
//! and right halves of the adapted field (red, green, blue, depth) are
//! centred on their running mean and whitened by C^(−1/2) of their running
//! covariance, refreshed every `VISION_WHITENING_REFRESH` brain ticks; the
//! visual turn weights' traces gather the turn noise's fresh innovation times
//! the whitened input, normalised by 1 + |input|². The weights learn at the
//! actors' rate scaled by the agent's heritable plasticity gene.

use xagent_brain::buffers::{
    BrainLayout, O_HOMEO, O_SCENT_TURN_WEIGHTS, O_SENSORY_MEAN, O_TICK_COUNT, O_TRACE_SCENT,
    O_TRACE_VISION, O_TURN_NOISE, O_VISION_PATHWAY_COVARIANCE, O_VISION_PATHWAY_INPUT,
    O_VISION_PATHWAY_MEAN, O_VISION_PATHWAY_WHITENING, O_VISION_PLASTICITY, O_VISION_TURN_WEIGHTS,
    P_EXPLORATION_RATE_OUT, P_FATIGUE_FACTOR_OUT, VISION_PATHWAY_INPUTS,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
const FULL_METER: f32 = 100.0;
/// Ticks run first, so the vision pass has filled the sensory buffer.
const WARMUP_TICKS: u64 = 5;
/// Mirrors `VISION_WHITENING_REFRESH` in `common.wgsl`: a brain tick count
/// that is a multiple of it refreshes the whitening matrix.
const REFRESH_TICK: f32 = 40.0;
/// A brain tick count that does not refresh it.
const PLAIN_TICK: f32 = 41.0;
/// Mirrors `TD_DISCOUNT * TD_LAMBDA`, `TURN_NOISE_PERSISTENCE`, and the
/// klinotaxis modulation.
const TRACE_DECAY: f32 = 0.97 * 0.9;
const NOISE_PERSISTENCE: f32 = 0.9;
const KLINOTAXIS_SENSITIVITY: f32 = 500.0;
const KLINOTAXIS_MIN: f32 = 0.3;
const KLINOTAXIS_MAX: f32 = 3.0;
/// Tolerances: the GPU's f32 Jacobi against an f64 reference, and float
/// rounding elsewhere.
const WHITENING_TOLERANCE: f32 = 2e-3;
const INPUT_TOLERANCE: f32 = 1e-4;
const N: usize = VISION_PATHWAY_INPUTS;
/// Mirrors `ACTION_WEIGHT_LEARNING_RATE` in `common.wgsl`.
const ACTOR_RATE: f32 = 0.10;
/// A plasticity gene away from its seed of 1, so the scaling shows.
const PLASTICITY: f32 = 3.0;
/// Traces planted before the step (non-zero, so both pathways move).
const PLANTED_SCENT_TRACE: f32 = 0.5;
const PLANTED_VISION_TRACE: f32 = 0.3;
/// Relative tolerance on the inferred step.
const STEP_TOLERANCE: f32 = 1e-3;

fn probe_config() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    }
}

fn probe_kernel() -> GpuKernel {
    let brain = probe_config();
    let world = WorldConfig {
        seed: 6,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, 17);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    // Hazard ground on the right half of the world, so the two halves of
    // the field differ.
    let biomes: Vec<u32> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            if cell % BIOME_SIDE >= BIOME_SIDE / 2 {
                2
            } else {
                0
            }
        })
        .collect();
    kernel.upload_world(&heights, &biomes, &[(-3.0, 0.35, 6.0)], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(2.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..WARMUP_TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    kernel
}

/// C^(−1/2) by Jacobi in f64 (reference for the shader's f32 version).
fn inverse_sqrt(cov: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut a = *cov;
    let mut v = [[0.0_f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut out = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = (0..N).map(|k| v[i][k] * v[j][k] / a[k][k].sqrt()).sum();
        }
    }
    out
}

#[test]
fn whitening_matrix_is_the_inverse_square_root_of_the_covariance() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = probe_kernel();
    // A correlated, positive-definite covariance: B·Bᵀ plus a small ridge.
    let mut cov = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            cov[i][j] = (0..N)
                .map(|k| {
                    let b = |r: usize, c: usize| ((r * 7 + c * 3) % 11) as f64 * 1e-2 - 0.04;
                    b(i, k) * b(j, k)
                })
                .sum::<f64>()
                + if i == j { 1e-3 } else { 0.0 };
        }
    }
    let mut state = kernel.read_agent_state(0);
    for i in 0..N {
        for j in 0..N {
            state.brain_state[O_VISION_PATHWAY_COVARIANCE + i * N + j] = cov[i][j] as f32;
        }
    }
    state.brain_state[O_TICK_COUNT] = REFRESH_TICK;
    kernel.write_agent_state(0, &state);
    kernel.dispatch_batch(WARMUP_TICKS, 1);

    let after = kernel.read_agent_state(0).brain_state;
    let expected = inverse_sqrt(&cov);
    let scale = expected
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs())) as f32;
    for i in 0..N {
        for j in 0..N {
            let gpu = after[O_VISION_PATHWAY_WHITENING + i * N + j];
            let cpu = expected[i][j] as f32;
            assert!(
                (gpu - cpu).abs() <= WHITENING_TOLERANCE * scale,
                "whitening[{i}][{j}]: GPU {gpu} vs CPU {cpu}"
            );
        }
    }
}

#[test]
fn pathway_reads_the_hemifields_and_credits_the_fresh_innovation() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = probe_kernel();
    // Identity whitening, zero mean and no sensory adaptation: the pathway's
    // input is then exactly the raw hemifield means.
    let mut state = kernel.read_agent_state(0);
    let layout = BrainLayout::default();
    let vision_count = layout.vision_color_count + layout.vision_depth_count;
    for k in 0..vision_count {
        state.brain_state[O_SENSORY_MEAN + k] = 0.0;
    }
    for i in 0..N {
        state.brain_state[O_VISION_PATHWAY_MEAN + i] = 0.0;
        for j in 0..N {
            state.brain_state[O_VISION_PATHWAY_WHITENING + i * N + j] =
                if i == j { 1.0 } else { 0.0 };
        }
    }
    state.brain_state[O_TICK_COUNT] = PLAIN_TICK;
    kernel.write_agent_state(0, &state);
    // The next brain tick reads the view the last vision pass wrote.
    let view = kernel.read_agent_telemetry_blocking(0).vision_color;
    let before = kernel.read_agent_state(0).brain_state;
    kernel.dispatch_batch(WARMUP_TICKS, 1);
    let after = kernel.read_agent_state(0).brain_state;
    let physics = kernel.read_full_state_blocking();

    let width = layout.vision_width as usize;
    let half = width / 2;
    for side in 0..2 {
        for channel in 0..3 {
            let (mut sum, mut count) = (0.0_f32, 0.0_f32);
            for row in 0..layout.vision_height as usize {
                for col in 0..width {
                    if (col >= half) == (side == 1) {
                        sum += view[(row * width + col) * 4 + channel];
                        count += 1.0;
                    }
                }
            }
            let gpu = after[O_VISION_PATHWAY_INPUT + side * 4 + channel];
            assert!(
                (gpu - sum / count).abs() < INPUT_TOLERANCE,
                "side {side} channel {channel}: GPU {gpu} vs hemifield mean {}",
                sum / count
            );
        }
    }
    let left_red = after[O_VISION_PATHWAY_INPUT];
    let right_red = after[O_VISION_PATHWAY_INPUT + 4];
    assert!(
        right_red > left_red,
        "the hazard ground on the right should redden the right half: left {left_red}, right {right_red}"
    );

    let klinotaxis = (1.0 - (after[O_HOMEO] - after[O_HOMEO + 1]) * KLINOTAXIS_SENSITIVITY)
        .clamp(KLINOTAXIS_MIN, KLINOTAXIS_MAX);
    let innovation = (after[O_TURN_NOISE] - NOISE_PERSISTENCE * before[O_TURN_NOISE])
        * physics[P_EXPLORATION_RATE_OUT]
        * physics[P_FATIGUE_FACTOR_OUT]
        * klinotaxis;
    let input: Vec<f32> = (0..N).map(|k| after[O_VISION_PATHWAY_INPUT + k]).collect();
    let input_sq: f32 = input.iter().map(|v| v * v).sum();
    for k in 0..N {
        let expected =
            TRACE_DECAY * before[O_TRACE_VISION + k] + innovation * input[k] / (1.0 + input_sq);
        assert!(
            (after[O_TRACE_VISION + k] - expected).abs() < INPUT_TOLERANCE,
            "vision trace {k}: GPU {} vs {expected}",
            after[O_TRACE_VISION + k]
        );
    }
}

#[test]
fn plasticity_gene_scales_the_visual_weight_step() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = probe_kernel();
    let config = BrainConfig {
        vision_plasticity: PLASTICITY,
        ..probe_config()
    };
    kernel.write_agent_heritable_config(0, &config);
    // The smell pathway learns at the actors' rate from the same TD error,
    // so its step reveals the error the visual step is scaled against.
    let mut state = kernel.read_agent_state(0);
    assert_eq!(state.brain_state[O_VISION_PLASTICITY], PLASTICITY);
    for k in 0..2 {
        state.brain_state[O_SCENT_TURN_WEIGHTS + k] = 0.0;
        state.brain_state[O_TRACE_SCENT + k] = PLANTED_SCENT_TRACE;
    }
    for k in 0..N {
        state.brain_state[O_VISION_TURN_WEIGHTS + k] = 0.0;
        state.brain_state[O_TRACE_VISION + k] = PLANTED_VISION_TRACE;
    }
    state.brain_state[O_TICK_COUNT] = PLAIN_TICK;
    kernel.write_agent_state(0, &state);
    kernel.dispatch_batch(WARMUP_TICKS, 1);
    let after = kernel.read_agent_state(0).brain_state;

    let td_error = after[O_SCENT_TURN_WEIGHTS] / (ACTOR_RATE * PLANTED_SCENT_TRACE);
    assert!(td_error.abs() > 1e-6, "the TD error vanished: {td_error}");
    let expected = ACTOR_RATE * PLASTICITY * td_error * PLANTED_VISION_TRACE;
    for k in 0..N {
        let step = after[O_VISION_TURN_WEIGHTS + k];
        assert!(
            (step - expected).abs() <= STEP_TOLERANCE * expected.abs(),
            "visual weight {k}: step {step} vs {expected} (plasticity {PLASTICITY})"
        );
    }
}
