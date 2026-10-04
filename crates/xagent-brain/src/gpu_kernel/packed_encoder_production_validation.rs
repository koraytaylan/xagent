//! Exercise the constructor's packed encoder through public lifecycle APIs.
//! The independent control keeps the same brain options and scalar global
//! credit. Every comparison includes all thirteen public mutable buffers;
//! packed storage is never exported to make a comparison pass.

use std::{collections::HashMap, error::Error};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore, Checkpoint};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::upload_random_scene;
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// Include a dying agent, a permanently inactive slot, and a live readback slot.
const AGENTS: u32 = 3;
const INACTIVE_AGENT: u32 = 1;
const READBACK_AGENT: u32 = 2;
const FOOD_ITEMS: usize = 104;
/// Exercise an odd raw feature count and the distinct cortical feature layout.
const FIELDS: [(u32, u32, bool); 2] = [(9, 7, false), (8, 6, true)];
/// Two independent seeds make the seeded reset change trained weights.
const INITIAL_SEED: u64 = 42;
const RESET_SEED: u64 = 314;
/// Every normal cycle runs all seven brain phases.
const COMPLETE_PHASES: u32 = 7;
const BRAIN_ONLY: u32 = 4;
const WITHOUT_LEARNING: u32 = COMPLETE_PHASES - 1;
/// A grouped vision route must leave the packed path and later reimport.
const ALTERNATE_VISION_STRIDE: u32 = 2;
/// Match the constructor's independent prefetch and context options.
const PREFETCH_FACTOR: u32 = 8;
const CONTEXT_LANES: u32 = 8;
/// Raw warmup crosses the refresh at twenty; cortex starts just before it.
const RAW_WARMUP: u32 = REFRESH_CYCLES + 1;
const CORTEX_WARMUP: u32 = 3;
/// Both explicit zero-energy injections must reach the death phase.
const EXPECTED_DEATHS: f32 = 2.0;
/// Raw transitions include a complete alternate vision batch.
const RAW_TRANSITION_CYCLES: u32 = 2;
const RAW_RESUME_CYCLES: u32 = 3;
/// A changed adaptation mean accompanies the deliberately changed matrix.
const WRITTEN_MEAN: f32 = 0.25;
const BRAIN_BUFFER: usize = 8;
const PATTERN_BUFFER: usize = 10;
const MUTABLE_BUFFERS: usize = 13;
const WORD_BYTES: usize = size_of::<f32>();

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn enabled(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// Match the optional arithmetic while excluding the packed representation.
fn scalar_passes() -> String {
    let prefetch = enabled("XAGENT_BRAIN_DENSE_PREFETCH");
    let lanes = std::env::var("XAGENT_BRAIN_PREDICTOR_LANES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| predictor_width::LANE_WIDTHS.contains(value))
        .unwrap_or(predictor_width::LANE_WIDTHS[0]);
    let mut passes = compose_brain_passes(enabled("XAGENT_BRAIN_COOPERATIVE_WHITENING"));
    if prefetch
        || lanes != predictor_width::LANE_WIDTHS[0]
        || enabled("XAGENT_BRAIN_FUSED_PREDICTOR")
    {
        passes = predictor_fusion::fuse_inline_predictor(&passes);
    }
    if prefetch {
        passes = dense_prefetch::prefetch_passes(&passes, PREFETCH_FACTOR);
    }
    passes = predictor_width::wider_predictor(&passes, lanes);
    if enabled("XAGENT_BRAIN_CONTEXT_GATHER") {
        passes = context_gather::gather_context(&passes, CONTEXT_LANES);
    }
    assert!(!passes.contains("packed_encoder."));
    passes
}

fn scalar_constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    let rays = kernel.layout.vision_width * kernel.layout.vision_height;
    let serial_groups = kernel.agent_count * (1 + rays.div_ceil(BRAIN_WORKGROUP_THREADS));
    let parallel = kernel.brain_vision_workgroups != serial_groups;
    if parallel {
        let rays_per_group = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
        assert_eq!(
            kernel.brain_vision_workgroups,
            kernel.agent_count * (1 + rays.div_ceil(rays_per_group))
        );
    }
    constants.insert(
        "VISION_PARALLEL_STEPS".into(),
        f64::from(u32::from(parallel)),
    );
    constants.insert(
        "VISION_AGENT_MASKS".into(),
        f64::from(u32::from(
            parallel && enabled("XAGENT_VISION_AGENT_MASKS") && kernel.agent_count <= u32::BITS,
        )),
    );
    constants
}

fn cache(kernel: &GpuKernel) -> Option<&packed_encoder::Cache> {
    kernel
        .global_credit
        .as_ref()
        .unwrap()
        .packed_encoder
        .as_ref()
}

fn assert_cache(kernel: &GpuKernel, packed: bool, valid: bool) {
    assert_eq!(cache(kernel).is_some(), packed);
    if let Some(cache) = cache(kernel) {
        assert_eq!(cache.is_valid(), valid, "packed cache validity");
    }
}

fn assert_warm_import_is_skipped(kernel: &GpuKernel) {
    let Some(cache) = cache(kernel) else {
        return;
    };
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    assert!(cache.is_valid());
    // A false return records no commands and makes no validity transition, so
    // this empty encoder can be dropped without abandoning an import.
    assert!(!cache.record_import(kernel, &mut encoder));
}

/// Restores made while the scalar arm is installed cannot invalidate a parked
/// cache. Explicitly invalidate it before putting those constructor handles back.
fn swap_credit(kernel: &mut GpuKernel, parked: &mut Option<global_credit::Pipelines>) {
    if let Some(cache) = parked
        .as_ref()
        .and_then(|credit| credit.packed_encoder.as_ref())
    {
        cache.invalidate();
    }
    std::mem::swap(&mut kernel.global_credit, parked);
}

fn active_route(kernel: &mut GpuKernel) {
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.vision_stride = 1;
    kernel.set_probe_pass_skips(false, false);
    kernel.probe.kernel_pass_limit = COMPLETE_PHASES;
    // Packed global credit must take precedence over the overlap preference.
    kernel.set_brain_beside_vision(true);
    assert!(kernel.global_credit_active());
}

fn prepare(brain: &BrainConfig) -> GpuKernel {
    let mut kernel = GpuKernel::new(AGENTS, FOOD_ITEMS, brain, &WorldConfig::default());
    assert!(
        cache(&kernel).is_some(),
        "run with XAGENT_BRAIN_PACKED_ENCODER=1 on a supported device"
    );
    if std::env::var("XAGENT_BRAIN_MAIN_THREADS").as_deref() == Ok("128") {
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            main_width::MAIN_THREADS,
        );
    }
    kernel.reset_agents_seeded(brain, INITIAL_SEED);
    active_route(&mut kernel);
    upload_random_scene(&kernel, 0, true, false);
    prepare_boundary_scene(&kernel);
    if brain.visual_cortex_enabled {
        let tick =
            fixed_tail_base(kernel.layout.brain_stride) + O_TICK_COUNT - O_PREDICTOR_CONTEXT_WEIGHT;
        for agent in 0..kernel.agent_count {
            let mut state = kernel.read_agent_state(agent);
            state.brain_state[tick] = f32::from(u16::try_from(REFRESH_CYCLES - 1).unwrap());
            kernel.write_agent_state(agent, &state);
        }
    }
    assert_cache(&kernel, true, false);
    kernel
}

fn advance(kernel: &mut GpuKernel, cycle: &mut u32, cycles: u32, mask: Option<u32>) {
    let end = *cycle + cycles;
    while *cycle < end {
        // Keep cortex to exactly one brain cycle in every driver submission.
        let count = if kernel.layout.visual_cortex_enabled {
            1
        } else {
            end - *cycle
        };
        let tick = u64::from(*cycle) * u64::from(kernel.brain_tick_stride);
        let ticks = count * kernel.brain_tick_stride;
        if let Some(mask) = mask {
            kernel.dispatch_batch_masked(tick, ticks, mask);
        } else {
            kernel.dispatch_ticks(tick, ticks);
        }
        kernel.poll_wait();
        *cycle += count;
    }
}

fn warm(kernel: &mut GpuKernel) -> u32 {
    let mut cycle = 0;
    let cycles = if kernel.layout.visual_cortex_enabled {
        CORTEX_WARMUP
    } else {
        RAW_WARMUP
    };
    let second_death = if kernel.layout.visual_cortex_enabled {
        1
    } else {
        REFRESH_CYCLES
    };
    while cycle < cycles {
        if cycle == second_death {
            force_death(kernel);
        }
        advance(kernel, &mut cycle, 1, None);
    }
    cycle
}

#[derive(Clone, Copy, Debug)]
enum Change {
    WriteOne,
    WriteBatch,
    ResetSeeded,
    ResetRandom,
    TryReset,
    Split,
    Tiled,
    VisionStride,
    SkipGlobal,
    PartialBrain,
    MaskedFull,
    MaskedBrain,
    SkipVision,
}

const CHANGES: [Change; 13] = [
    Change::WriteOne,
    Change::WriteBatch,
    Change::ResetSeeded,
    Change::ResetRandom,
    Change::TryReset,
    Change::Split,
    Change::Tiled,
    Change::VisionStride,
    Change::SkipGlobal,
    Change::PartialBrain,
    Change::MaskedFull,
    Change::MaskedBrain,
    Change::SkipVision,
];
/// Raw covers every transition; the slower cortical fixture repeats both write
/// APIs, a reset, and transitions through standalone and masked brain execution.
const CORTEX_CHANGES: [Change; 5] = [
    Change::WriteOne,
    Change::WriteBatch,
    Change::ResetSeeded,
    Change::Split,
    Change::MaskedFull,
];

impl Change {
    fn mask(self) -> Option<u32> {
        match self {
            Self::MaskedFull => Some(COMPLETE_PHASES),
            Self::MaskedBrain => Some(BRAIN_ONLY),
            _ => None,
        }
    }

    fn scalar_fallback(self) -> bool {
        matches!(
            self,
            Self::Split
                | Self::Tiled
                | Self::VisionStride
                | Self::SkipGlobal
                | Self::PartialBrain
                | Self::MaskedFull
                | Self::MaskedBrain
        )
    }

    fn writes_brain(self) -> bool {
        matches!(
            self,
            Self::WriteOne
                | Self::WriteBatch
                | Self::ResetSeeded
                | Self::ResetRandom
                | Self::TryReset
        )
    }

    fn random_reset(self) -> bool {
        matches!(self, Self::ResetRandom | Self::TryReset)
    }
}

fn changed_state(kernel: &GpuKernel, agent: u32) -> AgentBrainState {
    let mut state = kernel.read_agent_state(agent);
    let weights = kernel.layout.feature_count * ENCODED_DIMENSION;
    let before = state.brain_state[O_ENC_WEIGHTS].to_bits();
    for weight in &mut state.brain_state[O_ENC_WEIGHTS..O_ENC_WEIGHTS + weights] {
        *weight = -*weight;
    }
    assert_ne!(state.brain_state[O_ENC_WEIGHTS].to_bits(), before);
    let mean =
        fixed_tail_base(kernel.layout.brain_stride) + O_SENSORY_MEAN - O_PREDICTOR_CONTEXT_WEIGHT;
    state.brain_state[mean] = WRITTEN_MEAN;
    state
}

fn apply_change(kernel: &mut GpuKernel, brain: &BrainConfig, change: Change) {
    match change {
        Change::WriteOne => {
            kernel.write_agent_state(READBACK_AGENT, &changed_state(kernel, READBACK_AGENT))
        }
        Change::WriteBatch => {
            let states: Vec<_> = (0..kernel.agent_count)
                .map(|agent| changed_state(kernel, agent))
                .collect();
            kernel.batch_write_agent_states(states.len(), |agent| states[agent].clone());
        }
        Change::ResetSeeded => kernel.reset_agents_seeded(brain, RESET_SEED),
        Change::ResetRandom => kernel.reset_agents(brain),
        Change::TryReset => assert!(kernel.try_reset_agents(brain), "no readback is outstanding"),
        Change::Split => kernel.set_execution_mode(BrainExecutionMode::SplitSerial),
        Change::Tiled => kernel.set_execution_mode(BrainExecutionMode::ParallelTiled),
        Change::VisionStride => kernel.vision_stride = ALTERNATE_VISION_STRIDE,
        Change::SkipGlobal => kernel.set_probe_pass_skips(true, false),
        Change::PartialBrain => kernel.probe.kernel_pass_limit = WITHOUT_LEARNING,
        Change::SkipVision => kernel.set_probe_pass_skips(false, true),
        Change::MaskedFull | Change::MaskedBrain => (),
    }
}

fn assert_agent_snapshot(kernel: &GpuKernel, state: &State, actual: &AgentBrainState) {
    let agent = usize::try_from(READBACK_AGENT).unwrap();
    for (buffer, stride, actual) in [
        (
            BRAIN_BUFFER,
            kernel.layout.brain_stride,
            actual.brain_state.as_slice(),
        ),
        (PATTERN_BUFFER, PATTERN_STRIDE, actual.patterns.as_slice()),
    ] {
        let first = agent * stride * WORD_BYTES;
        let last = first + stride * WORD_BYTES;
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(actual),
            &state[buffer][first..last]
        );
    }
}

/// A queued read must retain its request-time snapshot while a later production
/// cycle runs; a subsequent blocking read must see the current scalar mirror.
fn readback_cycle(kernel: &mut GpuKernel, cycle: &mut u32, before: &State) -> TestResult<State> {
    assert_agent_snapshot(kernel, before, &kernel.read_agent_state(READBACK_AGENT));
    assert!(kernel.request_agent_state(READBACK_AGENT));
    advance(kernel, cycle, 1, None);
    let requested = kernel
        .try_collect_agent_state()
        .ok_or("agent read is still pending after poll")?
        .ok_or("agent read mapping failed")?;
    assert_agent_snapshot(kernel, before, &requested);
    let after = capture_state(kernel)?;
    assert_agent_snapshot(kernel, &after, &kernel.read_agent_state(READBACK_AGENT));
    Ok(after)
}

struct Trajectory {
    states: Vec<State>,
    /// Share random reset input bytes, never candidate output bytes.
    random_reset: Option<Checkpoint>,
}

fn trajectory(
    kernel: &mut GpuKernel,
    brain: &BrainConfig,
    change: Change,
    packed: bool,
    mut cycle: u32,
    shared_reset: Option<&Checkpoint>,
) -> TestResult<Trajectory> {
    active_route(kernel);
    assert_cache(kernel, packed, false);
    advance(kernel, &mut cycle, 1, None);
    assert_cache(kernel, packed, true);
    assert_warm_import_is_skipped(kernel);
    let mut states = vec![capture_state(kernel)?];
    apply_change(kernel, brain, change);
    // Check the API itself before a test restore could hide missed invalidation.
    assert_cache(kernel, packed, !change.writes_brain());
    let random_reset = if change.random_reset() {
        if let Some(initial) = shared_reset {
            restore(kernel, initial);
            None
        } else {
            Some(checkpoint(kernel))
        }
    } else {
        assert!(shared_reset.is_none());
        None
    };
    states.push(capture_state(kernel)?);
    if change.mask().is_none() {
        assert_eq!(
            kernel.global_credit_active(),
            !change.scalar_fallback(),
            "{change:?}"
        );
    }
    let transitions = if brain.visual_cortex_enabled {
        1
    } else {
        RAW_TRANSITION_CYCLES
    };
    advance(kernel, &mut cycle, transitions, change.mask());
    assert_cache(kernel, packed, !change.scalar_fallback());
    states.push(capture_state(kernel)?);
    active_route(kernel);
    let resume = if brain.visual_cortex_enabled {
        1
    } else {
        RAW_RESUME_CYCLES
    };
    advance(kernel, &mut cycle, resume, None);
    assert_cache(kernel, packed, true);
    states.push(capture_state(kernel)?);
    let readback_state = readback_cycle(kernel, &mut cycle, states.last().unwrap())?;
    assert_cache(kernel, packed, true);
    states.push(readback_state);
    Ok(Trajectory {
        states,
        random_reset,
    })
}

fn check_shape(brain: &BrainConfig) -> TestResult {
    let mut kernel = prepare(brain);
    let scalar =
        global_credit::Pipelines::new(&kernel, &scalar_passes(), &scalar_constants(&kernel))
            .ok_or("scalar global-credit control unavailable")?;
    assert!(scalar.packed_encoder.is_none());
    let mut parked = Some(scalar);
    let initial_state = capture_state(&kernel)?;
    let initial = checkpoint(&kernel);
    swap_credit(&mut kernel, &mut parked);
    let cycles = warm(&mut kernel);
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &initial);
    swap_credit(&mut kernel, &mut parked);
    assert_cache(&kernel, true, false);
    assert_eq!(warm(&mut kernel), cycles);
    assert_cache(&kernel, true, true);
    let warm_state = capture_state(&kernel)?;
    assert_eq!(warm_state.len(), MUTABLE_BUFFERS);
    assert_state_equal(&kernel, &expected, &warm_state);
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_DEATHS);
    assert_inactive_agent_unchanged(
        &kernel,
        &initial_state,
        &warm_state,
        INACTIVE_AGENT,
        "packed production warmup",
    );
    let warm = checkpoint(&kernel);
    let changes: &[Change] = if brain.visual_cortex_enabled {
        &CORTEX_CHANGES
    } else {
        &CHANGES
    };
    for &change in changes {
        println!(
            "PACKED_ENCODER_LIFECYCLE_STAGE width={} height={} cortex={} change={change:?}",
            brain.vision_width, brain.vision_height, brain.visual_cortex_enabled
        );
        active_route(&mut kernel);
        restore(&mut kernel, &warm);
        swap_credit(&mut kernel, &mut parked);
        let expected = trajectory(&mut kernel, brain, change, false, cycles, None)?;
        active_route(&mut kernel);
        restore(&mut kernel, &warm);
        swap_credit(&mut kernel, &mut parked);
        let actual = trajectory(
            &mut kernel,
            brain,
            change,
            true,
            cycles,
            expected.random_reset.as_ref(),
        )?;
        assert_eq!(actual.states.len(), expected.states.len());
        for (expected, actual) in expected.states.iter().zip(&actual.states) {
            assert_state_equal(&kernel, expected, actual);
            if !change.writes_brain() || matches!(change, Change::WriteOne) {
                assert_inactive_agent_unchanged(
                    &kernel,
                    &warm_state,
                    actual,
                    INACTIVE_AGENT,
                    "packed production transition",
                );
            }
        }
        println!("PACKED_ENCODER_LIFECYCLE width={} height={} cortex={} change={change:?} exact_buffers={MUTABLE_BUFFERS} checkpoints={} api_invalidation=true queued_read_snapshot=true public_mirror=true",
            brain.vision_width, brain.vision_height, brain.visual_cortex_enabled, actual.states.len());
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_PACKED_ENCODER=1; run explicitly"]
fn production_packed_encoder_preserves_lifecycle_and_public_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    assert!(enabled("XAGENT_BRAIN_PACKED_ENCODER"));
    for (width, height, cortex) in FIELDS {
        let brain = BrainConfig {
            vision_width: width,
            vision_height: height,
            visual_cortex_enabled: cortex,
            vision_stride: 1,
            ..BrainConfig::default()
        };
        check_shape(&brain)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_PACKED_ENCODER=1; run explicitly"]
fn production_packed_encoder_falls_back_at_workgroup_limit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    assert!(enabled("XAGENT_BRAIN_PACKED_ENCODER"));
    // This raw field leaves the ordinary main and combined pipelines within
    // the requested 16 KiB limit. Adding the packed encoder's 1 KiB scratch
    // crosses it. The independent source-declaration audit in packed_encoder
    // guards the production estimate used to reject this optional pipeline.
    const WIDTH: u32 = 20;
    const HEIGHT: u32 = 21;
    const FEATURES: usize = 2_127;
    const SCALAR_SHARED_BOUND: u32 = 15_616;
    const PACKED_SHARED_BOUND: u32 = 16_640;
    /// A few real cycles establish that the selected fallback still advances.
    const CYCLES: u32 = 3;
    let brain = BrainConfig {
        vision_width: WIDTH,
        vision_height: HEIGHT,
        visual_cortex_enabled: false,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    // Keep the actual constructor result. No limit override or synthetic
    // layout mutation manufactures the rejection after construction.
    let mut kernel = GpuKernel::new(AGENTS, FOOD_ITEMS, &brain, &WorldConfig::default());
    assert_eq!(kernel.layout.feature_count, FEATURES);
    let limit = kernel.device.limits().max_compute_workgroup_storage_size;
    assert!(SCALAR_SHARED_BOUND <= limit && limit < PACKED_SHARED_BOUND);
    assert!(
        kernel.global_credit.is_some(),
        "scalar credit must remain available"
    );
    assert!(
        cache(&kernel).is_none(),
        "constructor must select scalar credit"
    );
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        BRAIN_WORKGROUP_THREADS,
    );
    assert!(packed_encoder::Cache::new(&kernel).is_none());
    active_route(&mut kernel);
    kernel.reset_agents_seeded(&brain, INITIAL_SEED);
    upload_random_scene(&kernel, 0, true, false);
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let scalar =
        global_credit::Pipelines::new(&kernel, &scalar_passes(), &scalar_constants(&kernel))
            .ok_or("independent scalar control unavailable")?;
    assert!(scalar.packed_encoder.is_none());
    let mut parked = Some(scalar);
    swap_credit(&mut kernel, &mut parked);
    let mut cycle = 0;
    for _ in 0..CYCLES {
        advance(&mut kernel, &mut cycle, 1, None);
    }
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &initial);
    swap_credit(&mut kernel, &mut parked);
    cycle = 0;
    for _ in 0..CYCLES {
        assert!(kernel.global_credit_active());
        advance(&mut kernel, &mut cycle, 1, None);
        assert!(cache(&kernel).is_none());
    }
    let actual = capture_state(&kernel)?;
    assert_eq!(actual.len(), MUTABLE_BUFFERS);
    assert_state_equal(&kernel, &expected, &actual);
    let live = kernel.read_agent_state(READBACK_AGENT);
    let tick =
        fixed_tail_base(kernel.layout.brain_stride) + O_TICK_COUNT - O_PREDICTOR_CONTEXT_WEIGHT;
    assert_eq!(
        live.brain_state[tick],
        f32::from(u16::try_from(CYCLES).unwrap())
    );
    assert_agent_snapshot(&kernel, &actual, &live);
    println!("PACKED_ENCODER_RESOURCE_FALLBACK width={WIDTH} height={HEIGHT} features={FEATURES} device_workgroup_bytes={limit} scalar_shared_bound={SCALAR_SHARED_BOUND} packed_shared_bound={PACKED_SHARED_BOUND} cache_absent=true scalar_global_credit_active=true cycles={CYCLES} exact_buffers={MUTABLE_BUFFERS}");
    Ok(())
}
