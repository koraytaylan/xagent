//! Hardware checks of the production encoder-credit dispatch selection.
//! Removing the optional pipelines restores the independently retained inline
//! implementation; both paths start from identical bytes and run public APIs.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::upload_random_scene;
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// Match the measured small population while exercising a permanently dead slot.
const AGENTS: u32 = 10;
const FOOD_ITEMS: usize = 104;
/// Include an odd feature tail, a small field, and adapted cortical features.
const FIELDS: [(u32, u32, bool); 4] = [(8, 6, false), (9, 7, false), (1, 1, false), (8, 6, true)];
/// Visit two forced deaths and several whitening refresh boundaries.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const INACTIVE_AGENT: u32 = 1;
/// Reproducible weights are independent of the randomized geometry fixture.
const BRAIN_SEED: u64 = 42;
/// The fallback sample contains full vision batches and a physics remainder.
const FALLBACK_CYCLES: u32 = 4;
const PHYSICS_REMAINDER: u32 = 1;
/// The masked API executes physics, global/vision, and standalone brain.
const COMPLETE_PHASE_MASK: u32 = 7;
/// Learning is the seventh pass, so six is a deliberately incomplete brain.
const WITHOUT_LEARNING: u32 = 6;
/// Populate adapted features and memory before changing a live kernel's route.
const LIFECYCLE_WARMUP_CYCLES: u32 = 256;
/// Observe an active cycle, two transitional cycles, then three resumed cycles.
const LIFECYCLE_STAGE_CYCLES: [u32; 3] = [1, 2, 3];
/// Distinct deterministic weights prove that reset changes the warmed state.
const RESET_BRAIN_SEED: u64 = 314;
/// Reset positions are inside the default world and have normal body meters.
const RESET_AGENT_SPACING: f32 = 2.0;
const RESET_AGENT_HEIGHT: f32 = 1.0;
const RESET_BODY_METER: f32 = 100.0;
/// Change adapted input values through the public state-write API.
const WRITTEN_VISUAL_MEAN: f32 = 0.25;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn prepare(width: u32, height: u32, cortex: bool) -> GpuKernel {
    let brain = BrainConfig {
        vision_width: width,
        vision_height: height,
        visual_cortex_enabled: cortex,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut kernel = GpuKernel::new(AGENTS, FOOD_ITEMS, &brain, &WorldConfig::default());
    kernel.reset_agents_seeded(&brain, BRAIN_SEED);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_PHASE_MASK;
    upload_random_scene(&kernel, 0, true, false);
    prepare_boundary_scene(&kernel);
    assert!(
        kernel.global_credit.is_some(),
        "run with XAGENT_BRAIN_GLOBAL_CREDIT=1"
    );
    kernel
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for cycles in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        // Cortex parity uses bounded submissions so slow reference shaders
        // do not accumulate many expensive cycles in one GPU command buffer.
        // All cycles still execute, and checkpoints retain the same cadence.
        let per_submit = if kernel.layout.visual_cortex_enabled {
            1
        } else {
            cycles
        };
        let end = cycle + cycles;
        while cycle < end {
            let batch = per_submit.min(end - cycle);
            kernel.dispatch_ticks(
                u64::from(cycle * kernel.brain_tick_stride),
                batch * kernel.brain_tick_stride,
            );
            kernel.poll_wait();
            cycle += batch;
        }
        states.push(capture_state(kernel)?);
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_GLOBAL_CREDIT=1; run explicitly"]
fn production_global_credit_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height, cortex) in FIELDS {
        println!("PRODUCTION_GLOBAL_CREDIT_STAGE width={width} height={height} cortex={cortex} stage=prepare");
        let mut kernel = prepare(width, height, cortex);
        let option = kernel.global_credit.take();
        kernel.set_brain_beside_vision(false);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        println!("PRODUCTION_GLOBAL_CREDIT_STAGE width={width} height={height} cortex={cortex} stage=inline_reference");
        let expected = trajectory(&mut kernel)?;
        restore(&mut kernel, &initial);
        kernel.global_credit = option;
        // The option must override the default overlap request to run learning
        // before the global dispatch, where its independent updates execute.
        kernel.set_brain_beside_vision(true);
        assert!(kernel.global_credit_active());
        println!("PRODUCTION_GLOBAL_CREDIT_STAGE width={width} height={height} cortex={cortex} stage=offloaded");
        let actual = trajectory(&mut kernel)?;
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, expected, actual);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                actual,
                INACTIVE_AGENT,
                "production global credit",
            );
        }
        restore(&mut kernel, &initial);
        println!("PRODUCTION_GLOBAL_CREDIT_STAGE width={width} height={height} cortex={cortex} stage=repeat");
        let repeated = trajectory(&mut kernel)?;
        for (actual, repeated) in actual.iter().zip(&repeated) {
            assert_state_equal(&kernel, actual, repeated);
        }
        println!("PRODUCTION_GLOBAL_CREDIT width={width} height={height} cortex={cortex} cycles=100 exact_buffers=13 repeat_buffers=13 death_refresh=true requested_overlap=true");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Fallback {
    Split,
    Tiled,
    VisionStride,
    SkipGlobal,
    PassLimit,
    Masked,
}

fn fallback_dispatch(kernel: &mut GpuKernel, masked: bool) {
    let ticks = FALLBACK_CYCLES * kernel.brain_tick_stride + PHYSICS_REMAINDER;
    if masked {
        kernel.dispatch_batch_masked(0, ticks, COMPLETE_PHASE_MASK);
    } else {
        kernel.dispatch_ticks(0, ticks);
    }
    kernel.poll_wait();
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_GLOBAL_CREDIT=1; run explicitly"]
fn production_global_credit_keeps_unsupported_schedules() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    kernel.set_brain_beside_vision(false);
    let initial = checkpoint(&kernel);
    for case in [
        Fallback::Split,
        Fallback::Tiled,
        Fallback::VisionStride,
        Fallback::SkipGlobal,
        Fallback::PassLimit,
        Fallback::Masked,
    ] {
        kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
        kernel.vision_stride = 1;
        kernel.probe.skip_global = false;
        kernel.probe.kernel_pass_limit = COMPLETE_PHASE_MASK;
        match case {
            Fallback::Split => kernel.set_execution_mode(BrainExecutionMode::SplitSerial),
            Fallback::Tiled => kernel.set_execution_mode(BrainExecutionMode::ParallelTiled),
            Fallback::VisionStride => kernel.vision_stride = 2,
            Fallback::SkipGlobal => kernel.probe.skip_global = true,
            Fallback::PassLimit => kernel.probe.kernel_pass_limit = WITHOUT_LEARNING,
            Fallback::Masked => (),
        }
        let masked = matches!(case, Fallback::Masked);
        if !masked {
            assert!(!kernel.global_credit_active(), "{case:?}");
        }
        restore(&mut kernel, &initial);
        let option = kernel.global_credit.take();
        fallback_dispatch(&mut kernel, masked);
        let expected = capture_state(&kernel)?;
        restore(&mut kernel, &initial);
        kernel.global_credit = option;
        fallback_dispatch(&mut kernel, masked);
        assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        println!("PRODUCTION_GLOBAL_CREDIT_FALLBACK case={case:?} exact_buffers=13 physics_remainder=true");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Lifecycle {
    Split,
    Tiled,
    WriteState,
    Reset,
    SkipVision,
}

fn advance_lifecycle(kernel: &mut GpuKernel, cycle: &mut u32, cycles: u32) {
    kernel.dispatch_ticks(
        u64::from(*cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
    *cycle += cycles;
}

fn write_changed_agent_state(kernel: &GpuKernel) {
    let mut state = kernel.read_agent_state(0);
    let weights = kernel.layout.feature_count * ENCODED_DIMENSION;
    for weight in &mut state.brain_state[O_ENC_WEIGHTS..O_ENC_WEIGHTS + weights] {
        *weight = -*weight;
    }
    let mean =
        fixed_tail_base(kernel.layout.brain_stride) + O_SENSORY_MEAN - O_PREDICTOR_CONTEXT_WEIGHT;
    let previous = state.brain_state[mean];
    state.brain_state[mean] = if previous == WRITTEN_VISUAL_MEAN {
        -WRITTEN_VISUAL_MEAN
    } else {
        WRITTEN_VISUAL_MEAN
    };
    assert_ne!(previous.to_bits(), state.brain_state[mean].to_bits());
    kernel.write_agent_state(0, &state);
}

fn reset_lifecycle_agents(kernel: &mut GpuKernel) {
    let brain = BrainConfig {
        vision_width: FIELDS[0].0,
        vision_height: FIELDS[0].1,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    kernel.reset_agents_seeded(&brain, RESET_BRAIN_SEED);
    let agents: Vec<_> = (0..kernel.agent_count)
        .map(|agent| {
            (
                glam::Vec3::new(agent as f32 * RESET_AGENT_SPACING, RESET_AGENT_HEIGHT, 0.0),
                RESET_BODY_METER,
                RESET_BODY_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
}

fn lifecycle_trajectory(
    kernel: &mut GpuKernel,
    case: Lifecycle,
    enabled: bool,
) -> TestResult<Vec<State>> {
    assert_eq!(kernel.global_credit.is_some(), enabled);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.set_probe_pass_skips(false, false);
    assert_eq!(kernel.global_credit_active(), enabled);
    let mut cycle = LIFECYCLE_WARMUP_CYCLES;
    advance_lifecycle(kernel, &mut cycle, LIFECYCLE_STAGE_CYCLES[0]);
    let mut states = vec![capture_state(kernel)?];

    match case {
        Lifecycle::Split => kernel.set_execution_mode(BrainExecutionMode::SplitSerial),
        Lifecycle::Tiled => kernel.set_execution_mode(BrainExecutionMode::ParallelTiled),
        Lifecycle::WriteState => write_changed_agent_state(kernel),
        Lifecycle::Reset => reset_lifecycle_agents(kernel),
        Lifecycle::SkipVision => kernel.set_probe_pass_skips(false, true),
    }
    let supported = !matches!(case, Lifecycle::Split | Lifecycle::Tiled);
    assert_eq!(
        kernel.global_credit_active(),
        enabled && supported,
        "{case:?}: transition route"
    );
    advance_lifecycle(kernel, &mut cycle, LIFECYCLE_STAGE_CYCLES[1]);
    states.push(capture_state(kernel)?);

    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_probe_pass_skips(false, false);
    assert_eq!(
        kernel.global_credit_active(),
        enabled,
        "{case:?}: resumed route"
    );
    advance_lifecycle(kernel, &mut cycle, LIFECYCLE_STAGE_CYCLES[2]);
    states.push(capture_state(kernel)?);
    Ok(states)
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_GLOBAL_CREDIT=1; run explicitly"]
fn production_global_credit_preserves_warm_lifecycle_transitions() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    kernel.set_brain_beside_vision(false);
    assert!(kernel.global_credit_active());
    let mut cycle = 0;
    advance_lifecycle(&mut kernel, &mut cycle, LIFECYCLE_WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    for case in [
        Lifecycle::Split,
        Lifecycle::Tiled,
        Lifecycle::WriteState,
        Lifecycle::Reset,
        Lifecycle::SkipVision,
    ] {
        restore(&mut kernel, &warm);
        let option = kernel.global_credit.take();
        let expected = lifecycle_trajectory(&mut kernel, case, false)?;
        restore(&mut kernel, &warm);
        kernel.global_credit = option;
        // Private features are deliberately not checkpointed or restored.
        // The next live main must overwrite them before credit consumes them.
        let actual = lifecycle_trajectory(&mut kernel, case, true)?;
        assert_eq!(expected.len(), LIFECYCLE_STAGE_CYCLES.len());
        assert_eq!(actual.len(), expected.len());
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, expected, actual);
        }
        println!("PRODUCTION_GLOBAL_CREDIT_LIFECYCLE case={case:?} warmup_cycles={LIFECYCLE_WARMUP_CYCLES} stage_cycles={LIFECYCLE_STAGE_CYCLES:?} exact_buffers=13 selection_asserted=true private_features_restored=false");
    }
    Ok(())
}
