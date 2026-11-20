//! The critic input route and the forward pathway route, probed on one agent
//! tick by tick: each test reads the brain state before and after a tick and
//! compares the change with the rule written in the kernel.

use xagent_brain::buffers::{
    BrainLayout, BrainTickProbe, ENCODED_DIMENSION, MAX_WEIGHT_NORM, O_CRITIC_PATHWAY_WEIGHTS,
    O_ENCODED_MEAN, O_FORWARD_PATHWAY_WEIGHTS, O_PREV_ENCODED, O_PREV_PATHWAY_INPUT_NORM,
    O_RECENT_PATHWAY_KEYS, O_SCENT_WHITENED, O_SENSORY_MEAN, O_TICK_COUNT, O_TRACE_CRITIC,
    O_TRACE_CRITIC_PATHWAY, O_TRACE_FORWARD_PATHWAY, O_VALUE_BIAS, O_VALUE_WEIGHTS,
    O_VISION_PATHWAY_INPUT, O_VISION_PATHWAY_MEAN, O_VISION_PATHWAY_WHITENING, O_VISION_PLASTICITY,
    PATHWAY_INPUT_COUNT, P_ENERGY, P_MOTOR_FWD_OUT, RECENT_CAP, SCENT_CHANNELS,
    VISION_PATHWAY_INPUTS,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, CriticInput, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Divisor of the biome grid side at which the right half begins.
const BIOME_HALF_DIVISOR: usize = 2;
/// Full energy and integrity meters of the probe agent.
const FULL_METER: f32 = 100.0;
/// Ticks run first, so the vision pass has filled the sensory buffer.
const WARMUP_TICKS: u64 = 5;
/// Mirrors `TD_DISCOUNT * TD_LAMBDA` in `common.wgsl`.
const TRACE_DECAY: f32 = 0.97 * 0.9;
/// Mirrors `ACTION_WEIGHT_LEARNING_RATE`, `CRITIC_LEARNING_RATE` and
/// `TERMINAL_DEATH_TD_ERROR` in `common.wgsl`.
const ACTOR_RATE: f32 = 0.10;
const CRITIC_RATE: f32 = 0.01;
const TERMINAL_ERROR: f32 = -1.0;
/// Heritable plasticity gene of the probe agent.
const PLASTICITY: f32 = 2.0;
/// Absolute tolerance on values and steps.
const TOLERANCE: f32 = 1e-4;
/// Planted trace and weight values.
const PLANTED_TRACE: f32 = 0.5;
const PLANTED_WEIGHT: f32 = 0.3;
const PLANTED_NORM: f32 = 0.7;
/// Number of pathway inputs (scent channels, then visual hemifields).
const N: usize = PATHWAY_INPUT_COUNT;
/// World seed of the probe world.
const WORLD_SEED: u64 = 6;
/// Seed for the probe agent's brain initialisation.
const AGENT_SEED: u64 = 17;
/// Biome id of the right half of the probe world; the left half is zero.
const RIGHT_HALF_BIOME: u32 = 2;
/// Food item (x, y, z) placed in the probe world.
const FOOD_POSITION: (f32, f32, f32) = (-3.0, 0.35, 6.0);
/// Spawn position (x, y, z) of the probe agent.
const SPAWN_POSITION: (f32, f32, f32) = (2.0, 1.0, 0.0);
/// Ticks run with routes off to check the route block stays zero.
const ROUTES_OFF_TICKS: u64 = 40;
/// Planted scent forward weight, close enough to the ball to cross it.
const NEAR_BALL_WEIGHT: f32 = 1.9;
/// Smallest TD error the one-tick forward check accepts as non-vanishing.
const MIN_TD_ERROR: f32 = 1e-6;
/// Forward pathway weight used to drive the command away from the zero run.
const FORWARD_PROBE_WEIGHT: f32 = 1.5;
/// Per-index growth of the critic pathway weight pattern.
const PATTERN_STEP: f32 = 0.1;
/// Period of the alternating sign in the probe weight pattern (even
/// indices positive, odd indices negative).
const PATTERN_SIGN_PERIOD: usize = 2;
/// Number of leading 128-d value weights planted with a ramp.
const VALUE_RAMP_WIDTH: usize = 16;
/// Slope of the planted value-weight ramp.
const VALUE_RAMP_STEP: f32 = 0.05;
/// Index at which the planted value-weight ramp crosses zero.
const VALUE_RAMP_CENTRE: f32 = 8.0;
/// Leading 128-d critic trace entries checked to have moved.
const TRACE_PROBE_WIDTH: usize = 4;

/// True when a GPU is usable; otherwise skips, or panics under
/// `XAGENT_REQUIRE_GPU=1`.
fn gpu_ready() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var("XAGENT_REQUIRE_GPU").as_deref() != Ok("1"),
        "XAGENT_REQUIRE_GPU=1 but no GPU adapter is available"
    );
    eprintln!("skipping: no GPU adapter");
    false
}

fn probe_config(critic_input: CriticInput, forward_pathway_enabled: bool) -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        vision_plasticity: PLASTICITY,
        critic_input,
        forward_pathway_enabled,
        ..BrainConfig::default()
    }
}

/// One agent with the route flags set, past warmup, with identity whitening
/// and a settled tick behind it, so the stored pathway inputs are non-zero.
/// Returns the kernel and the next tick number.
fn probe_kernel(critic_input: CriticInput, forward_pathway_enabled: bool) -> (GpuKernel, u64) {
    let brain = probe_config(critic_input, forward_pathway_enabled);
    let world = WorldConfig {
        seed: WORLD_SEED,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, AGENT_SEED);
    kernel.set_brain_tick_probe(true);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes: Vec<u32> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            if cell % BIOME_SIDE >= BIOME_SIDE / BIOME_HALF_DIVISOR {
                RIGHT_HALF_BIOME
            } else {
                0
            }
        })
        .collect();
    kernel.upload_world(&heights, &biomes, &[FOOD_POSITION], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(SPAWN_POSITION.0, SPAWN_POSITION.1, SPAWN_POSITION.2),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    for tick in 0..WARMUP_TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    let layout = BrainLayout::default();
    let vision_count = layout.vision_color_count + layout.vision_depth_count;
    let mut state = kernel.read_agent_state(0);
    for k in 0..vision_count {
        state.brain_state[O_SENSORY_MEAN + k] = 0.0;
    }
    for i in 0..VISION_PATHWAY_INPUTS {
        state.brain_state[O_VISION_PATHWAY_MEAN + i] = 0.0;
        for j in 0..VISION_PATHWAY_INPUTS {
            state.brain_state[O_VISION_PATHWAY_WHITENING + i * VISION_PATHWAY_INPUTS + j] =
                if i == j { 1.0 } else { 0.0 };
        }
    }
    kernel.write_agent_state(0, &state);
    kernel.dispatch_batch(WARMUP_TICKS, 1);
    (kernel, WARMUP_TICKS + 1)
}

fn brain_state(kernel: &GpuKernel) -> Vec<f32> {
    kernel.read_agent_state(0).brain_state
}

/// Plants `values` into the brain state of the agent.
fn plant(kernel: &GpuKernel, values: &[(usize, f32)]) {
    let mut state = kernel.read_agent_state(0);
    for &(slot, value) in values {
        state.brain_state[slot] = value;
    }
    kernel.write_agent_state(0, &state);
}

/// Runs one tick and returns the brain state and the tick's probe.
fn run_tick(kernel: &mut GpuKernel, tick: u64) -> (Vec<f32>, BrainTickProbe) {
    kernel.dispatch_batch(tick, 1);
    let mut probes = Vec::new();
    kernel.read_brain_tick_probes_blocking(&mut probes);
    (brain_state(kernel), probes.remove(0))
}

/// The ten whitened pathway inputs held in `state`: scent, then vision.
fn pathway_inputs(state: &[f32]) -> [f32; N] {
    let mut c = [0.0_f32; N];
    for (k, value) in c.iter_mut().enumerate() {
        *value = if k < SCENT_CHANNELS {
            state[O_SCENT_WHITENED + k]
        } else {
            state[O_VISION_PATHWAY_INPUT + k - SCENT_CHANNELS]
        };
    }
    c
}

fn assert_close(label: &str, gpu: f32, expected: f32) {
    assert!(
        (gpu - expected).abs() <= TOLERANCE * (1.0 + expected.abs()),
        "{label}: GPU {gpu} vs expected {expected}"
    );
}

fn norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Scales `values` back into the weight ball.
fn into_ball(values: &mut [f32]) {
    let length = norm(values);
    if length > MAX_WEIGHT_NORM {
        for value in values.iter_mut() {
            *value *= MAX_WEIGHT_NORM / length;
        }
    }
}

/// The ranges of the route block that must stay untouched with both routes off.
fn route_block(state: &[f32]) -> Vec<f32> {
    let mut block = state[O_FORWARD_PATHWAY_WEIGHTS..=O_PREV_PATHWAY_INPUT_NORM].to_vec();
    block.extend_from_slice(&state[O_RECENT_PATHWAY_KEYS..O_RECENT_PATHWAY_KEYS + RECENT_CAP * N]);
    block
}

/// Kills the agent and runs the tick whose cycle settles the dying life.
/// Returns the state before that tick and after it, with its probe.
fn die_and_respawn(kernel: &mut GpuKernel, tick: u64) -> (Vec<f32>, Vec<f32>, BrainTickProbe) {
    let before = brain_state(kernel);
    kernel.write_agent_physics_fields(0, &[(P_ENERGY, 0.0)]);
    let (after, probe) = run_tick(kernel, tick);
    (before, after, probe)
}

#[test]
fn routes_off_keep_route_block_zero() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, first) = probe_kernel(CriticInput::Encoding, false);
    for tick in first..first + ROUTES_OFF_TICKS {
        run_tick(&mut kernel, tick);
    }
    let block = route_block(&brain_state(&kernel));
    assert!(
        block.iter().all(|value| *value == 0.0),
        "the route block moved with both routes off"
    );
}

#[test]
fn forward_rule_matches_d7_one_tick() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::Encoding, true);
    let mut planted = Vec::new();
    for k in 0..N {
        planted.push((O_TRACE_FORWARD_PATHWAY + k, PLANTED_TRACE));
        planted.push((O_FORWARD_PATHWAY_WEIGHTS + k, 0.0));
    }
    // The scent senses start near the ball so the step crosses it.
    planted.push((O_FORWARD_PATHWAY_WEIGHTS, NEAR_BALL_WEIGHT));
    planted.push((O_FORWARD_PATHWAY_WEIGHTS + 1, NEAR_BALL_WEIGHT));
    plant(&kernel, &planted);
    let before = brain_state(&kernel);
    assert_eq!(before[O_VISION_PLASTICITY], PLASTICITY);
    let (after, probe) = run_tick(&mut kernel, tick);

    let x = probe.pathway_inputs;
    let delta = probe.td_error;
    assert!(delta.abs() > MIN_TD_ERROR, "the TD error vanished: {delta}");
    let scent_sq: f32 = x[..SCENT_CHANNELS].iter().map(|v| v * v).sum();
    let vision_sq: f32 = x[SCENT_CHANNELS..].iter().map(|v| v * v).sum();
    let mut scent_weights = [0.0_f32; SCENT_CHANNELS];
    let mut vision_weights = [0.0_f32; VISION_PATHWAY_INPUTS];
    for k in 0..N {
        let (rate, sense_sq) = if k < SCENT_CHANNELS {
            (ACTOR_RATE, scent_sq)
        } else {
            (ACTOR_RATE * PLASTICITY, vision_sq)
        };
        let weight = before[O_FORWARD_PATHWAY_WEIGHTS + k] + rate * delta * PLANTED_TRACE;
        if k < SCENT_CHANNELS {
            scent_weights[k] = weight;
        } else {
            vision_weights[k - SCENT_CHANNELS] = weight;
        }
        let trace = TRACE_DECAY * PLANTED_TRACE + probe.forward_noise * x[k] / (1.0 + sense_sq);
        assert_close(
            &format!("forward trace {k}"),
            after[O_TRACE_FORWARD_PATHWAY + k],
            trace,
        );
    }
    into_ball(&mut scent_weights);
    into_ball(&mut vision_weights);
    for k in 0..N {
        let expected = if k < SCENT_CHANNELS {
            scent_weights[k]
        } else {
            vision_weights[k - SCENT_CHANNELS]
        };
        assert_close(
            &format!("forward weight {k}"),
            after[O_FORWARD_PATHWAY_WEIGHTS + k],
            expected,
        );
    }
    let scent_norm =
        norm(&after[O_FORWARD_PATHWAY_WEIGHTS..O_FORWARD_PATHWAY_WEIGHTS + SCENT_CHANNELS]);
    let vision_norm =
        norm(&after[O_FORWARD_PATHWAY_WEIGHTS + SCENT_CHANNELS..O_FORWARD_PATHWAY_WEIGHTS + N]);
    assert!(
        scent_norm <= MAX_WEIGHT_NORM + TOLERANCE,
        "scent ball {scent_norm}"
    );
    assert!(
        vision_norm <= MAX_WEIGHT_NORM + TOLERANCE,
        "vision ball {vision_norm}"
    );
}

#[test]
fn forward_pathway_term_moves_command() {
    if !gpu_ready() {
        return;
    }
    let forward_out = |weight: f32| {
        let (mut kernel, tick) = probe_kernel(CriticInput::Encoding, true);
        let planted: Vec<(usize, f32)> = (0..N)
            .map(|k| (O_FORWARD_PATHWAY_WEIGHTS + k, weight))
            .collect();
        plant(&kernel, &planted);
        kernel.dispatch_batch(tick, 1);
        kernel.read_full_state_blocking()[P_MOTOR_FWD_OUT]
    };
    let baseline = forward_out(0.0);
    let driven = forward_out(FORWARD_PROBE_WEIGHT);
    assert!(
        (driven - baseline).abs() > TOLERANCE,
        "the forward command ignored the pathway weights: {baseline} vs {driven}"
    );
}

#[test]
fn forward_death_settles_and_clears_traces() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::Encoding, true);
    let mut planted = Vec::new();
    for k in 0..N {
        planted.push((O_TRACE_FORWARD_PATHWAY + k, PLANTED_TRACE));
        planted.push((O_FORWARD_PATHWAY_WEIGHTS + k, 0.0));
    }
    plant(&kernel, &planted);
    let (before, after, probe) = die_and_respawn(&mut kernel, tick);
    let x = probe.pathway_inputs;
    let scent_sq: f32 = x[..SCENT_CHANNELS].iter().map(|v| v * v).sum();
    let vision_sq: f32 = x[SCENT_CHANNELS..].iter().map(|v| v * v).sum();
    let mut scent_weights = [0.0_f32; SCENT_CHANNELS];
    let mut vision_weights = [0.0_f32; VISION_PATHWAY_INPUTS];
    for k in 0..N {
        let rate = if k < SCENT_CHANNELS {
            ACTOR_RATE
        } else {
            ACTOR_RATE * PLASTICITY
        };
        let weight = before[O_FORWARD_PATHWAY_WEIGHTS + k]
            + rate * TERMINAL_ERROR * before[O_TRACE_FORWARD_PATHWAY + k];
        if k < SCENT_CHANNELS {
            scent_weights[k] = weight;
        } else {
            vision_weights[k - SCENT_CHANNELS] = weight;
        }
        // The respawned life starts its trace from zero.
        let sense_sq = if k < SCENT_CHANNELS {
            scent_sq
        } else {
            vision_sq
        };
        assert_close(
            &format!("forward trace {k} after death"),
            after[O_TRACE_FORWARD_PATHWAY + k],
            probe.forward_noise * x[k] / (1.0 + sense_sq),
        );
    }
    into_ball(&mut scent_weights);
    into_ball(&mut vision_weights);
    let settled: f32 = scent_weights
        .iter()
        .chain(&vision_weights)
        .map(|w| w.abs())
        .sum();
    assert!(
        settled > TOLERANCE,
        "the terminal step moved no forward weight"
    );
    for k in 0..N {
        let expected = if k < SCENT_CHANNELS {
            scent_weights[k]
        } else {
            vision_weights[k - SCENT_CHANNELS]
        };
        assert_close(
            &format!("forward weight {k} after death"),
            after[O_FORWARD_PATHWAY_WEIGHTS + k],
            expected,
        );
    }
}

/// A critic weight pattern distinct per index.
fn pattern(k: usize) -> f32 {
    PLANTED_WEIGHT
        * (1.0 + PATTERN_STEP * k as f32)
        * if k % PATTERN_SIGN_PERIOD == 0 {
            1.0
        } else {
            -1.0
        }
}

#[test]
fn critic_encoding_mode_zero_pathway_slots() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::Encoding, false);
    let planted: Vec<(usize, f32)> = (0..VALUE_RAMP_WIDTH)
        .map(|k| {
            (
                O_VALUE_WEIGHTS + k,
                VALUE_RAMP_STEP * (k as f32 - VALUE_RAMP_CENTRE),
            )
        })
        .collect();
    plant(&kernel, &planted);
    let before = brain_state(&kernel);
    let (after, probe) = run_tick(&mut kernel, tick);
    let dot: f32 = (0..ENCODED_DIMENSION)
        .map(|k| before[O_VALUE_WEIGHTS + k] * probe.encoding[k])
        .sum();
    assert_close("value", probe.value, before[O_VALUE_BIAS] + dot);
    assert!(
        route_block(&after).iter().all(|value| *value == 0.0),
        "pathway slots moved in encoding mode"
    );
    assert!(after[O_TRACE_CRITIC..O_TRACE_CRITIC + TRACE_PROBE_WIDTH]
        .iter()
        .any(|v| *v != 0.0));
}

#[test]
fn critic_pathway_mode_freezes_value_weights() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::Pathway, false);
    let mut planted: Vec<(usize, f32)> = (0..N)
        .map(|k| (O_CRITIC_PATHWAY_WEIGHTS + k, pattern(k)))
        .collect();
    planted.extend((0..VALUE_RAMP_WIDTH).map(|k| {
        (
            O_VALUE_WEIGHTS + k,
            VALUE_RAMP_STEP * (k as f32 - VALUE_RAMP_CENTRE),
        )
    }));
    plant(&kernel, &planted);
    let before = brain_state(&kernel);
    let (after, probe) = run_tick(&mut kernel, tick);
    let c = pathway_inputs(&before);
    let dot: f32 = (0..N).map(|k| pattern(k) * c[k]).sum();
    assert_close("value", probe.value, before[O_VALUE_BIAS] + dot);
    let encoded = ENCODED_DIMENSION;
    assert_eq!(
        &after[O_VALUE_WEIGHTS..O_VALUE_WEIGHTS + encoded],
        &before[O_VALUE_WEIGHTS..O_VALUE_WEIGHTS + encoded],
        "the 128-d value weights moved in pathway mode"
    );
    assert_eq!(
        &after[O_TRACE_CRITIC..O_TRACE_CRITIC + encoded],
        &before[O_TRACE_CRITIC..O_TRACE_CRITIC + encoded],
        "the 128-d critic trace moved in pathway mode"
    );
}

#[test]
fn critic_encoding_and_pathway_matches_d6() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::EncodingAndPathway, false);
    let mut planted: Vec<(usize, f32)> = (0..N)
        .map(|k| (O_CRITIC_PATHWAY_WEIGHTS + k, pattern(k)))
        .collect();
    planted.extend((0..N).map(|k| (O_TRACE_CRITIC_PATHWAY + k, PLANTED_TRACE)));
    planted.push((O_PREV_PATHWAY_INPUT_NORM, PLANTED_NORM));
    planted.extend((0..VALUE_RAMP_WIDTH).map(|k| {
        (
            O_VALUE_WEIGHTS + k,
            VALUE_RAMP_STEP * (k as f32 - VALUE_RAMP_CENTRE),
        )
    }));
    plant(&kernel, &planted);
    let before = brain_state(&kernel);
    let (after, probe) = run_tick(&mut kernel, tick);
    let encoded = ENCODED_DIMENSION;
    let c = pathway_inputs(&before);
    let encoding_dot: f32 = (0..encoded)
        .map(|k| before[O_VALUE_WEIGHTS + k] * probe.encoding[k])
        .sum();
    let pathway_dot: f32 = (0..N).map(|k| pattern(k) * c[k]).sum();
    assert_close(
        "value",
        probe.value,
        before[O_VALUE_BIAS] + encoding_dot + pathway_dot,
    );
    let encoding_sq: f32 = (0..encoded)
        .map(|k| {
            let x = before[O_PREV_ENCODED + k] - before[O_ENCODED_MEAN + k];
            x * x
        })
        .sum();
    let normalizer = 1.0 / (1.0 + encoding_sq + PLANTED_NORM);
    let delta = probe.td_error;
    let mut weights = [0.0_f32; N];
    for k in 0..N {
        weights[k] = pattern(k) + CRITIC_RATE * normalizer * delta * PLANTED_TRACE;
    }
    into_ball(&mut weights);
    for k in 0..N {
        assert_close(
            &format!("critic pathway weight {k}"),
            after[O_CRITIC_PATHWAY_WEIGHTS + k],
            weights[k],
        );
        assert_close(
            &format!("critic pathway trace {k}"),
            after[O_TRACE_CRITIC_PATHWAY + k],
            TRACE_DECAY * PLANTED_TRACE + c[k],
        );
    }
    let c_sq: f32 = c.iter().map(|v| v * v).sum();
    assert_close("stored normaliser", after[O_PREV_PATHWAY_INPUT_NORM], c_sq);
}

#[test]
fn ring_pathway_keys_equal_stored_input() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::EncodingAndPathway, false);
    let before = brain_state(&kernel);
    let (after, _) = run_tick(&mut kernel, tick);
    let c = pathway_inputs(&before);
    let store_slot = (after[O_TICK_COUNT] as usize) % RECENT_CAP;
    for k in 0..N {
        assert_close(
            &format!("ring key {k}"),
            after[O_RECENT_PATHWAY_KEYS + store_slot * N + k],
            c[k],
        );
    }
}

#[test]
fn critic_death_zeroes_both_traces() {
    if !gpu_ready() {
        return;
    }
    let (mut kernel, tick) = probe_kernel(CriticInput::EncodingAndPathway, false);
    let mut planted: Vec<(usize, f32)> = (0..N)
        .map(|k| (O_TRACE_CRITIC_PATHWAY + k, PLANTED_TRACE))
        .collect();
    planted.extend((0..ENCODED_DIMENSION).map(|k| (O_TRACE_CRITIC + k, PLANTED_TRACE)));
    // The stored pathway inputs the respawn tick reads are zero, so the
    // pathway trace's first term after death is zero and the trace itself
    // must read zero.
    planted.extend((0..SCENT_CHANNELS).map(|k| (O_SCENT_WHITENED + k, 0.0)));
    planted.extend((0..N - SCENT_CHANNELS).map(|k| (O_VISION_PATHWAY_INPUT + k, 0.0)));
    plant(&kernel, &planted);
    let (before, after, probe) = die_and_respawn(&mut kernel, tick);
    // The planted traces are live before death, so a surviving decayed
    // remainder would show up in the checks below.
    for (slot, value) in &planted {
        assert_close(
            &format!("planted slot {slot} before death"),
            before[*slot],
            *value,
        );
    }
    // Death zeroes both traces: the pathway trace reads zero, and the 128-d
    // trace holds exactly the respawned life's first term (the centred
    // encoding), so its remainder over that term is zero.
    for k in 0..N {
        assert_close(
            &format!("critic pathway trace {k} after death"),
            after[O_TRACE_CRITIC_PATHWAY + k],
            0.0,
        );
    }
    for k in 0..ENCODED_DIMENSION {
        assert_close(
            &format!("critic trace {k} remainder after death"),
            after[O_TRACE_CRITIC + k] - probe.encoding[k],
            0.0,
        );
    }
}
