//! A sense declared only in this test is carried by the generic sense block:
//! its slots append after the declared senses, start at zero and learn, with
//! no shader or kernel change.

use xagent_brain::buffers::{
    BrainLayout, MAX_WEIGHT_NORM, O_SENSORY_MEAN, O_TRACE_VISION, PATHWAY_ROUTE_SLOT_COUNT,
};
use xagent_brain::senses::{
    Heritability, LearningRate, MotorRoute, RouteDescriptor, SenseDescriptor, SenseInput,
    Whitening, SMELL, VISION,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

/// Index of the energy delta in the non-visual feature vector. The documented
/// order (27 features, `NON_VISUAL_FEATURE_COUNT`) is: speed (1) at 0, facing
/// (3) at 1..=3, angular velocity (1) at 4, energy (1) at 5, integrity (1) at
/// 6, energy delta (1) at 7, integrity delta (1) at 8, touch (16) at 9..=24,
/// scent (2) at 25..=26.
const ENERGY_DELTA_FEATURE: usize = 7;
/// Index of the integrity delta, the feature after the energy delta.
const INTEGRITY_DELTA_FEATURE: usize = 8;
/// Index of the first scent feature: 1 + 3 + 1 + 1 + 1 + 2 + 16.
const SCENT_FEATURE: usize = 25;

/// Channels of the probe sense: the energy and integrity deltas.
const PROBE_CHANNELS: usize = 2;
/// Slots a two-channel closed-form sense with a turn route adds: mean (2),
/// covariance (3), whitened input (2), turn weights (2) and traces (2).
const PROBE_SLOTS: usize = 11;
/// Slots of the vision sense's visual inputs between `O_TRACE_VISION` and the
/// route block.
const VISION_TRACE_SLOTS: usize = 8;
/// Running-mean adaptation rate of the probe, a slow average like the other senses.
const PROBE_ADAPTATION_RATE: f32 = 0.01;
/// Starting variance of the probe's whitening covariance: unit scale.
const PROBE_INITIAL_VARIANCE: f32 = 1.0;
/// Position of the probe in `EXTENDED`: after smell (0) and vision (1).
const PROBE_BLOCK_INDEX: usize = 2;
/// Position of the smell sense in the declared order.
const SMELL_BLOCK_INDEX: usize = 0;
/// Position of the vision sense in the declared order.
const VISION_BLOCK_INDEX: usize = 1;
/// Covariance slots of a two-channel closed-form whitener: variance of each
/// channel plus their cross term.
const PROBE_COVARIANCE_SLOTS: usize = 3;
/// World seed, fixed so the run is repeatable.
const WORLD_SEED: u64 = 9;
/// Agent reset seed, fixed so the run is repeatable.
const RESET_SEED: u64 = 21;
/// The hazard biome starts at this fraction of the biome side (one over this).
const HAZARD_BOUNDARY_DIVISOR: usize = 2;

const PROBE_ROUTES: &[RouteDescriptor] = &[RouteDescriptor {
    route: MotorRoute::Turn,
    heritability: Heritability::LearnedOnly,
}];

/// The extra sense: energy and integrity deltas, learning at the actor rate.
const PROBE: SenseDescriptor = SenseDescriptor {
    name: "probe",
    channels: PROBE_CHANNELS,
    input: SenseInput::NonVisualFeatures {
        offset: ENERGY_DELTA_FEATURE,
    },
    adaptation_rate: PROBE_ADAPTATION_RATE,
    whitening: Whitening::ClosedForm2 {
        initial_variance: PROBE_INITIAL_VARIANCE,
    },
    learning_rate: LearningRate::Actor,
    routes: PROBE_ROUTES,
};

const EXTENDED: &[SenseDescriptor] = &[SMELL, VISION, PROBE];

/// Heightmap samples per side, matching the default world terrain grid.
const TERRAIN_SIDE: usize = 129;
/// Biome cells per side, matching the default world biome grid.
const BIOME_SIDE: usize = 256;
/// Agents in the run: one per spot.
const AGENT_COUNT: u32 = 4;
/// Small, so eating changes the energy by varying amounts.
const MAX_ENERGY: f32 = 20.0;
/// Full integrity at spawn, the default world scale.
const MAX_INTEGRITY: f32 = 100.0;
/// Ticks stepped: long enough for agents to eat and take hazard damage.
const RUN_TICKS: u64 = 600;
/// Biome id of the damaging ground.
const HAZARD_BIOME: u32 = 2;
/// Food height just above flat terrain so agents reach it.
const FOOD_HEIGHT: f32 = 0.35;
/// Agent spots: the first two stand on hazard ground, the others on safe ground.
const SPOTS: [(f32, f32); 4] = [
    (SPOT_OFFSET, 0.0),
    (SPOT_OFFSET, SPOT_OFFSET),
    (-SPOT_OFFSET, 0.0),
    (-SPOT_OFFSET, SPOT_OFFSET),
];
/// Distance of each agent spot from the origin along an axis, wide enough to keep spots apart.
const SPOT_OFFSET: f32 = 10.0;
/// Food items placed at each agent spot, so each agent eats more than once.
const FOOD_PER_AGENT: usize = 2;

fn config() -> BrainConfig {
    BrainConfig {
        brain_tick_stride: 1,
        vision_stride: 1,
        movement_speed: 0.0,
        ..BrainConfig::default()
    }
}

fn world() -> WorldConfig {
    WorldConfig {
        seed: WORLD_SEED,
        ..WorldConfig::default()
    }
}

fn extended_kernel(brain: &BrainConfig) -> GpuKernel {
    let mut kernel = GpuKernel::new_with_senses(
        AGENT_COUNT,
        SPOTS.len() * FOOD_PER_AGENT,
        brain,
        &world(),
        EXTENDED,
    );
    kernel.reset_agents_seeded(brain, RESET_SEED);
    kernel
}

/// Resets seeded agents on hazard and safe ground beside food, steps the
/// kernel, and returns every agent's state.
fn run_agents() -> Vec<xagent_brain::AgentBrainState> {
    let brain = config();
    let mut kernel = extended_kernel(&brain);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes: Vec<u32> = (0..BIOME_SIDE * BIOME_SIDE)
        .map(|cell| {
            if cell % BIOME_SIDE >= BIOME_SIDE / HAZARD_BOUNDARY_DIVISOR {
                HAZARD_BIOME
            } else {
                0
            }
        })
        .collect();
    let food: Vec<(f32, f32, f32)> = SPOTS
        .iter()
        .flat_map(|&(x, z)| std::iter::repeat_n((x, FOOD_HEIGHT, z), FOOD_PER_AGENT))
        .collect();
    let consumed = vec![false; food.len()];
    let timers = vec![0.0_f32; food.len()];
    kernel.upload_world(&heights, &biomes, &food, &consumed, &timers);
    let agents: Vec<_> = SPOTS
        .iter()
        .map(|&(x, z)| {
            (
                glam::Vec3::new(x, 1.0, z),
                MAX_ENERGY,
                MAX_INTEGRITY,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    for tick in 0..RUN_TICKS {
        kernel.dispatch_batch(tick, 1);
    }
    (0..AGENT_COUNT)
        .map(|agent| kernel.read_agent_state(agent))
        .collect()
}

fn any_nonzero(slots: &[f32]) -> bool {
    slots.iter().any(|&v| v != 0.0)
}

#[test]
fn probe_layout_offsets_append_without_moving_existing_slots() {
    assert_eq!(INTEGRITY_DELTA_FEATURE, ENERGY_DELTA_FEATURE + 1);
    assert_eq!(
        SMELL.input,
        SenseInput::NonVisualFeatures {
            offset: SCENT_FEATURE
        }
    );
    let config = BrainConfig::default();
    let base = BrainLayout::from_config(&config);
    let extended = BrainLayout::with_senses(&config, EXTENDED);
    assert_eq!(extended.brain_stride, base.brain_stride + PROBE_SLOTS);
    assert_eq!(extended.sensory_mean_base(), O_SENSORY_MEAN + PROBE_SLOTS);
    assert_eq!(base.sensory_mean_base(), O_SENSORY_MEAN);
    let (base_blocks, extended_blocks) = (base.sense_blocks(), extended.sense_blocks());
    assert_eq!(extended_blocks.len(), base_blocks.len() + 1);
    assert_eq!(&extended_blocks[..base_blocks.len()], base_blocks);
    let probe = extended_blocks[PROBE_BLOCK_INDEX];
    assert_eq!(
        probe.mean,
        O_TRACE_VISION + VISION_TRACE_SLOTS + PATHWAY_ROUTE_SLOT_COUNT
    );
    assert_eq!(probe.channels, PROBE_CHANNELS);
    assert_eq!(probe.covariance, probe.mean + PROBE_CHANNELS);
    assert_eq!(
        probe.whitened_input,
        probe.covariance + PROBE_COVARIANCE_SLOTS
    );
    assert_eq!(probe.turn_weights, probe.whitened_input + PROBE_CHANNELS);
    assert_eq!(probe.turn_trace, probe.turn_weights + PROBE_CHANNELS);
    assert_eq!(
        probe.turn_trace + PROBE_CHANNELS,
        extended.sensory_mean_base()
    );
}

#[test]
fn probe_sense_starts_at_zero_on_gpu() {
    assert!(GpuKernel::is_available(), "no GPU adapter available");
    let brain = config();
    let kernel = extended_kernel(&brain);
    let layout = BrainLayout::with_senses(&brain, EXTENDED);
    let probe = layout.sense_blocks()[PROBE_BLOCK_INDEX];
    for agent in 0..AGENT_COUNT {
        let state = kernel.read_agent_state(agent).brain_state;
        let weights = &state[probe.turn_weights..probe.turn_weights + PROBE_CHANNELS];
        let traces = &state[probe.turn_trace..probe.turn_trace + PROBE_CHANNELS];
        assert!(weights.iter().all(|&v| v == 0.0), "weights {weights:?}");
        assert!(traces.iter().all(|&v| v == 0.0), "traces {traces:?}");
    }
}

#[test]
fn probe_sense_learns_through_generic_block() {
    assert!(GpuKernel::is_available(), "no GPU adapter available");
    let layout = BrainLayout::with_senses(&config(), EXTENDED);
    let blocks = layout.sense_blocks();
    let states = run_agents();
    let slice = |state: &[f32], start: usize| state[start..start + PROBE_CHANNELS].to_vec();
    let mut probe_inputs = Vec::new();
    let mut probe_weights = Vec::new();
    let mut smell_inputs = Vec::new();
    let mut vision_inputs = Vec::new();
    for state in &states {
        let brain = &state.brain_state;
        let (probe, smell, vision) = (
            blocks[PROBE_BLOCK_INDEX],
            blocks[SMELL_BLOCK_INDEX],
            blocks[VISION_BLOCK_INDEX],
        );
        probe_inputs.extend(slice(brain, probe.whitened_input));
        probe_weights.extend(slice(brain, probe.turn_weights));
        smell_inputs.extend(slice(brain, smell.whitened_input));
        vision_inputs.extend_from_slice(
            &brain[vision.whitened_input..vision.whitened_input + vision.channels],
        );
    }
    assert!(probe_inputs.iter().all(|v| v.is_finite()));
    assert!(any_nonzero(&probe_inputs), "probe inputs {probe_inputs:?}");
    assert!(probe_weights.iter().all(|v| v.is_finite()));
    assert!(
        any_nonzero(&probe_weights),
        "probe weights {probe_weights:?}"
    );
    for pair in probe_weights.chunks(PROBE_CHANNELS) {
        let norm = pair.iter().map(|w| w * w).sum::<f32>().sqrt();
        assert!(norm <= MAX_WEIGHT_NORM, "weight norm {norm}");
    }
    assert!(any_nonzero(&smell_inputs), "smell inputs {smell_inputs:?}");
    assert!(
        any_nonzero(&vision_inputs),
        "vision inputs {vision_inputs:?}"
    );
}
