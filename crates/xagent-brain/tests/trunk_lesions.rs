//! Trunk lesions: each switch removes exactly one term of the policy or the
//! critic, and with every switch off the kernel computes what it always did.

use xagent_brain::buffers::{
    AgentBrainState, BrainTickProbe, TrunkLesions, ENCODED_DIMENSION, MEMORY_CAP,
    O_ACTION_FORWARD_WEIGHTS, O_ACTION_TURN_WEIGHTS, O_PAT_ACTIVE, O_PAT_META, O_PAT_MOTOR,
    O_PAT_NORMS, O_TRACE_FWD, O_TRACE_TURN, O_VALUE_BIAS, O_VALUE_WEIGHTS,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity for every placed agent.
const FULL_METER: f32 = 100.0;
/// Ticks run first, so the nostrils smell the food and the statistics move.
const WARMUP_TICKS: u64 = 20;
/// Planted policy and critic weight magnitude; the norm stays far below
/// `MAX_WEIGHT_NORM`, so the L2 ball never rescales it.
const PLANTED_WEIGHT: f32 = 0.05;
/// Planted critic value bias.
const PLANTED_VALUE_BIAS: f32 = 0.3;
/// Planted recalled forward output.
const PLANTED_MEMORY_FORWARD: f32 = 0.9;
/// Planted recalled turn output.
const PLANTED_MEMORY_TURN: f32 = -0.9;
/// Planted recalled valence.
const PLANTED_MEMORY_VALENCE: f32 = 1.0;
/// Smallest dot a planted weight must produce for a lesion to be observable.
const OBSERVABLE: f32 = 1e-6;
/// World seed shared by every kernel in these tests.
const WORLD_SEED: u64 = 5;
/// Seed for the agents' initial brains.
const AGENT_SEED: u64 = 13;
/// Position (x, y, z) of the one food item, within smelling range of the agent.
const FOOD_PLACEMENT: (f32, f32, f32) = (-4.0, 0.35, 6.0);
/// Each recalled pattern stores forward, turn and valence in consecutive
/// slots from `O_PAT_MOTOR`; this names the turn component.
/// Seed for the reset the lesion switches must survive.
const RESET_SEED: u64 = 7;
const MOTOR_TURN_COMPONENT: usize = 1;
/// Valence component of a recalled pattern's motor record.
const MOTOR_VALENCE_COMPONENT: usize = 2;

fn gpu_ready() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var("XAGENT_REQUIRE_GPU").is_err(),
        "XAGENT_REQUIRE_GPU is set but no adapter is available"
    );
    eprintln!("Skipping: no GPU/fallback adapter available");
    false
}

fn configs() -> (BrainConfig, WorldConfig) {
    (
        BrainConfig {
            brain_tick_stride: 1,
            vision_stride: 1,
            movement_speed: 0.0,
            ..BrainConfig::default()
        },
        WorldConfig {
            seed: WORLD_SEED,
            ..WorldConfig::default()
        },
    )
}

/// A seeded one-agent kernel with food in smelling range, the probe on, and
/// `WARMUP_TICKS` run with no lesion.
fn warmed_kernel() -> GpuKernel {
    let (brain, world) = configs();
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, AGENT_SEED);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[FOOD_PLACEMENT], &[false], &[0.0]);
    kernel.upload_agents(&[(
        glam::Vec3::new(0.0, 1.0, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    kernel.set_brain_tick_probe(true);
    for tick in 0..WARMUP_TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    kernel
}

fn last_probe(kernel: &GpuKernel) -> BrainTickProbe {
    let mut probes = Vec::new();
    kernel.read_brain_tick_probes_blocking(&mut probes);
    assert_eq!(probes.len(), 1, "probe readback failed");
    probes[0]
}

/// Plant policy and critic weights (small, in the key's direction) and clear
/// the policy eligibility traces, so the TD error a lesion changes cannot move
/// the policy weights before this tick's dots are taken.
fn plant_weights(kernel: &GpuKernel, key: &[f32; ENCODED_DIMENSION]) -> AgentBrainState {
    let mut state = kernel.read_agent_state(0);
    for d in 0..ENCODED_DIMENSION {
        let sign = if key[d] >= 0.0 { 1.0 } else { -1.0 };
        state.brain_state[O_ACTION_FORWARD_WEIGHTS + d] = PLANTED_WEIGHT;
        state.brain_state[O_ACTION_TURN_WEIGHTS + d] = PLANTED_WEIGHT * sign;
        state.brain_state[O_VALUE_WEIGHTS + d] = PLANTED_WEIGHT * sign;
        state.brain_state[O_TRACE_FWD + d] = 0.0;
        state.brain_state[O_TRACE_TURN + d] = 0.0;
    }
    state.brain_state[O_VALUE_BIAS] = PLANTED_VALUE_BIAS;
    state
}

/// Plant one active recalled pattern that matches `key`.
fn plant_memory(state: &mut AgentBrainState, key: &[f32; ENCODED_DIMENSION]) {
    for d in 0..ENCODED_DIMENSION {
        state.patterns[d * MEMORY_CAP] = key[d];
    }
    state.patterns[O_PAT_NORMS] = key.iter().map(|v| v * v).sum::<f32>().sqrt();
    state.patterns[O_PAT_ACTIVE] = 1.0;
    state.patterns[O_PAT_MOTOR] = PLANTED_MEMORY_FORWARD;
    state.patterns[O_PAT_MOTOR + MOTOR_TURN_COMPONENT] = PLANTED_MEMORY_TURN;
    state.patterns[O_PAT_MOTOR + MOTOR_VALENCE_COMPONENT] = PLANTED_MEMORY_VALENCE;
    state.patterns[O_PAT_META] = 0.0;
}

struct Outcome {
    probe: BrainTickProbe,
    motor_forward: f32,
    motor_turn: f32,
}

/// Run the planted scenario for one tick under `lesions`.
fn run(lesions: TrunkLesions, memory: bool) -> Outcome {
    let mut kernel = warmed_kernel();
    let key = last_probe(&kernel).encoding;
    let mut state = plant_weights(&kernel, &key);
    if memory {
        plant_memory(&mut state, &key);
    }
    kernel.write_agent_state(0, &state);
    kernel.set_trunk_lesions(lesions);
    kernel.dispatch_batch(WARMUP_TICKS, 1);
    let telemetry = kernel.read_agent_telemetry_blocking(0);
    Outcome {
        probe: last_probe(&kernel),
        motor_forward: telemetry.motor_fwd,
        motor_turn: telemetry.motor_turn,
    }
}

fn full_state(kernel: &mut GpuKernel) -> Vec<f32> {
    let agent = kernel.read_agent_state(0);
    let mut all = agent.brain_state;
    all.extend_from_slice(&agent.patterns);
    all.extend_from_slice(kernel.read_full_state_blocking());
    all
}

#[test]
fn defaults_off_and_survive_reset() {
    if !gpu_ready() {
        return;
    }
    let (brain, world) = configs();
    let mut kernel = GpuKernel::new(1, 1, &brain, &world);
    assert_eq!(kernel.trunk_lesions(), TrunkLesions::NONE);
    assert!(!kernel.brain_tick_probe_enabled());
    kernel.set_trunk_lesions(TrunkLesions::ALL);
    assert_eq!(kernel.trunk_lesions(), TrunkLesions::ALL);
    kernel.reset_agents_seeded(&brain, RESET_SEED);
    assert_eq!(kernel.trunk_lesions(), TrunkLesions::ALL);
    assert!(kernel.trunk_lesions().turn_weights);
    assert!(kernel.trunk_lesions().forward_weights);
    assert!(kernel.trunk_lesions().memory_blend);
    assert!(kernel.trunk_lesions().critic);
}

#[test]
fn none_matches_unset_bit_for_bit() {
    if !gpu_ready() {
        return;
    }
    let mut unset = warmed_kernel();
    let mut explicit = warmed_kernel();
    explicit.set_trunk_lesions(TrunkLesions::NONE);
    for tick in WARMUP_TICKS..WARMUP_TICKS + 5 {
        unset.dispatch_batch(tick, 1);
        explicit.dispatch_batch(tick, 1);
    }
    let a = full_state(&mut unset);
    let b = full_state(&mut explicit);
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "an explicit NONE changed the state"
    );
}

#[test]
fn each_lesion_removes_only_its_term() {
    if !gpu_ready() {
        return;
    }
    let base = run(TrunkLesions::NONE, false);
    assert!(base.probe.encoding_turn_output.abs() > OBSERVABLE);
    assert!(base.probe.encoding_forward_output.abs() > OBSERVABLE);

    let turn = run(
        TrunkLesions {
            turn_weights: true,
            ..TrunkLesions::NONE
        },
        false,
    );
    assert_eq!(turn.probe.encoding, base.probe.encoding);
    assert_eq!(turn.probe.value, base.probe.value);
    assert_eq!(
        turn.probe.encoding_forward_output,
        base.probe.encoding_forward_output
    );
    assert_eq!(
        turn.probe.encoding_turn_output, base.probe.encoding_turn_output,
        "the computed dot is reported even when lesioned"
    );
    assert_ne!(turn.motor_turn, base.motor_turn);

    let forward = run(
        TrunkLesions {
            forward_weights: true,
            ..TrunkLesions::NONE
        },
        false,
    );
    assert_eq!(forward.probe.encoding, base.probe.encoding);
    assert_eq!(forward.probe.value, base.probe.value);
    assert_eq!(
        forward.probe.encoding_turn_output,
        base.probe.encoding_turn_output
    );
    assert_eq!(
        forward.probe.encoding_forward_output, base.probe.encoding_forward_output,
        "the computed dot is reported even when lesioned"
    );
    assert_ne!(forward.motor_forward, base.motor_forward);

    let critic = run(
        TrunkLesions {
            critic: true,
            ..TrunkLesions::NONE
        },
        false,
    );
    assert_eq!(critic.probe.encoding, base.probe.encoding);
    assert_ne!(critic.probe.value, base.probe.value);
    assert_eq!(
        critic.probe.encoding_turn_output,
        base.probe.encoding_turn_output
    );
    assert_eq!(
        critic.probe.encoding_forward_output,
        base.probe.encoding_forward_output
    );

    let none_memory = run(TrunkLesions::NONE, true);
    let memory = run(
        TrunkLesions {
            memory_blend: true,
            ..TrunkLesions::NONE
        },
        true,
    );
    assert_eq!(memory.probe.encoding, none_memory.probe.encoding);
    assert_eq!(memory.probe.value, none_memory.probe.value);
    assert_eq!(
        memory.probe.encoding_turn_output,
        none_memory.probe.encoding_turn_output
    );
    assert_eq!(
        memory.probe.encoding_forward_output,
        none_memory.probe.encoding_forward_output
    );
    assert_ne!(
        (memory.motor_forward, memory.motor_turn),
        (none_memory.motor_forward, none_memory.motor_turn)
    );
}

#[test]
fn critic_lesion_value_is_value_bias() {
    if !gpu_ready() {
        return;
    }
    let lesioned = run(
        TrunkLesions {
            critic: true,
            ..TrunkLesions::NONE
        },
        false,
    );
    // The bias drifts a little in the tick (the TD update runs after the
    // value is formed), so read the value the pass formed, not the end state.
    assert_eq!(lesioned.probe.value, PLANTED_VALUE_BIAS);
}

#[test]
fn memory_lesion_skips_recalled_blend() {
    if !gpu_ready() {
        return;
    }
    let unplanted = run(TrunkLesions::NONE, false);
    let blended = run(TrunkLesions::NONE, true);
    let lesioned = run(
        TrunkLesions {
            memory_blend: true,
            ..TrunkLesions::NONE
        },
        true,
    );
    assert_ne!(
        (blended.motor_forward, blended.motor_turn),
        (unplanted.motor_forward, unplanted.motor_turn),
        "the planted pattern should blend into the motor"
    );
    assert_eq!(lesioned.motor_forward, unplanted.motor_forward);
    assert_eq!(lesioned.motor_turn, unplanted.motor_turn);
}
