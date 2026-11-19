//! The brain-tick probe is passive and reports the values the brain pass used.

use xagent_brain::buffers::{
    BrainTickProbe, ENCODED_DIMENSION, O_ACTION_FORWARD_WEIGHTS, O_ENCODED_MEAN, O_PREV_ENCODED,
    O_PREV_VALUE, O_SCENT_WHITENED, O_VISION_PATHWAY_INPUT,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Matches the terrain side used to size the kernel heightmap.
const TERRAIN_SIDE: usize = 129;
/// Matches the biome grid side used to size the kernel biome buffer.
const BIOME_SIDE: usize = 256;
/// Full energy and integrity for every placed agent.
const FULL_METER: f32 = 100.0;
/// Brain ticks run before the probe is read.
const TICKS: u64 = 20;
/// Agents in the readback-count test.
const READBACK_AGENTS: u32 = 3;
/// Scalars the probe reads from the scent pathway.
const SCENT_INPUTS: usize = 2;
/// Relative tolerance between the GPU tree-reduced forward dot and the
/// sequential CPU dot over the same operands.
const DOT_RELATIVE_TOLERANCE: f32 = 1e-4;
/// World seed shared by every kernel in these tests.
const WORLD_SEED: u64 = 5;
/// Seed for the agents' initial brains.
const AGENT_SEED: u64 = 13;
/// Position (x, y, z) of the one food item, within smelling range of the agents.
const FOOD_PLACEMENT: (f32, f32, f32) = (-4.0, 0.35, 6.0);
/// Spacing along x between consecutive agents.
const AGENT_SPACING: f32 = 3.0;

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

fn kernel_with(agents: u32, probe: bool) -> GpuKernel {
    let brain = BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: WORLD_SEED,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(agents, 1, &brain, &world);
    kernel.reset_agents_seeded(&brain, AGENT_SEED);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[FOOD_PLACEMENT], &[false], &[0.0]);
    let placed: Vec<_> = (0..agents)
        .map(|i| {
            (
                glam::Vec3::new(i as f32 * AGENT_SPACING, 1.0, 0.0),
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&placed);
    kernel.set_brain_tick_probe(probe);
    kernel
}

fn run_ticks(kernel: &mut GpuKernel) {
    for tick in 0..TICKS {
        kernel.dispatch_batch(tick, 1);
    }
}

#[test]
fn probe_is_passive_and_exact() {
    if !gpu_ready() {
        return;
    }
    let mut on = kernel_with(1, true);
    let mut off = kernel_with(1, false);
    run_ticks(&mut on);
    run_ticks(&mut off);
    let (a, b) = (on.read_agent_state(0), off.read_agent_state(0));
    assert!(a
        .brain_state
        .iter()
        .zip(&b.brain_state)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(a
        .patterns
        .iter()
        .zip(&b.patterns)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    let pa = on.read_full_state_blocking().to_vec();
    let pb = off.read_full_state_blocking().to_vec();
    assert_eq!(pa.len(), pb.len());
    assert!(pa.iter().zip(&pb).all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
fn probe_fields_equal_brain_values() {
    if !gpu_ready() {
        return;
    }
    let mut kernel = kernel_with(1, true);
    for tick in 0..TICKS - 1 {
        kernel.dispatch_batch(tick, 1);
    }
    // The running mean moves at the end of the tick, so the mean the key
    // was centred on is the one read before the last tick.
    let before = kernel.read_agent_state(0);
    kernel.dispatch_batch(TICKS - 1, 1);
    let state = kernel.read_agent_state(0);
    let telemetry = kernel.read_agent_telemetry_blocking(0);
    let mut probes: Vec<BrainTickProbe> = Vec::new();
    kernel.read_brain_tick_probes_blocking(&mut probes);
    assert_eq!(probes.len(), 1);
    let probe = probes[0];

    assert_eq!(probe.value, state.brain_state[O_PREV_VALUE]);
    assert_eq!(probe.td_error, telemetry.td_error);
    for k in 0..SCENT_INPUTS {
        assert_eq!(
            probe.pathway_inputs[k],
            state.brain_state[O_SCENT_WHITENED + k]
        );
    }
    for k in 0..probe.pathway_inputs.len() - SCENT_INPUTS {
        assert_eq!(
            probe.pathway_inputs[SCENT_INPUTS + k],
            state.brain_state[O_VISION_PATHWAY_INPUT + k]
        );
    }
    assert_eq!(probe.encoding.len(), ENCODED_DIMENSION);
    assert!(
        probe.encoding.iter().any(|v| v.abs() > 0.0),
        "the centred encoding should not be all zero"
    );
    assert!(probe.encoding.iter().all(|v| v.is_finite()));
    // The encoding is the centred memory key the critic reads: this tick's
    // encoding (stored as the previous encoding) minus the running mean.
    for d in 0..ENCODED_DIMENSION {
        let key = state.brain_state[O_PREV_ENCODED + d] - before.brain_state[O_ENCODED_MEAN + d];
        assert_eq!(
            probe.encoding[d].to_bits(),
            key.to_bits(),
            "encoding dimension {d}"
        );
    }
    // The forward dot reads the uncentred encoding.
    let (mut dot, mut magnitude) = (0.0_f32, 0.0_f32);
    for d in 0..ENCODED_DIMENSION {
        let term =
            state.brain_state[O_ACTION_FORWARD_WEIGHTS + d] * state.brain_state[O_PREV_ENCODED + d];
        dot += term;
        magnitude += term.abs();
    }
    assert!(
        (probe.encoding_forward_output - dot).abs() <= DOT_RELATIVE_TOLERANCE * magnitude,
        "forward dot {} differs from the uncentred dot {dot}",
        probe.encoding_forward_output
    );
}

#[test]
fn readback_returns_agent_count_entries() {
    if !gpu_ready() {
        return;
    }
    let mut kernel = kernel_with(READBACK_AGENTS, true);
    run_ticks(&mut kernel);
    let mut probes = Vec::new();
    kernel.read_brain_tick_probes_blocking(&mut probes);
    assert_eq!(probes.len(), READBACK_AGENTS as usize);
}
