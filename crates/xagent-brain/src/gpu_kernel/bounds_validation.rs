//! Hardware-only measurement of compiler-inserted shader bounds checks.
//! Both arms use identical production cooperative-whitening and fused-predictor
//! source. Only the main module's bounds-check option changes; loop bounding,
//! default floating-point compilation, workgroup initialization, and every
//! other pipeline stay identical. Platform hardware robustness remains enabled.
//!
//! The trusted module is restricted to this seeded, fixed-layout fixture. It
//! accepts no saved or user-provided state and is not a production configuration.
//! Ring cursors and memory slot indices are asserted before dispatch. Their
//! shader updates preserve the checked ranges: rings use modulo, and selected
//! memory IDs are permutations/reductions of initialized IDs below MEMORY_CAP.
//! Recent-history slots use modulo, food claims come from the checked bounded
//! food scan, and terrain/biome lookups clamp their coordinates. Dense indices
//! derive from the asserted layout, bounded loops and exactly 256 invocations.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Matches the controlled small-world profile fixture.
const FIXTURE_AGENTS: u32 = 10;
/// The checked food-claim scan supplies IDs strictly below this population.
const FIXTURE_FOOD: usize = 104;
/// The index audit covers the default raw-vision layout only.
const FIXTURE_VISION_WIDTH: u32 = 8;
/// The index audit covers the default raw-vision layout only.
const FIXTURE_VISION_HEIGHT: u32 = 6;
/// Kernel push constants contain the start tick and complete brain-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// All seven brain passes must execute in both arms.
const COMPLETE_BRAIN: u32 = 7;
/// Food flags contain one consumed value and one claim value per item.
const FOOD_FLAG_SLOTS: usize = 2;
/// Deaths and whitening refreshes straddle these one-hundred-cycle checkpoints.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature memory exercises the complete recent-experience window.
const WARMUP_CYCLES: u32 = 256;
/// A bounded one-hundred-cycle trial amortizes submission overhead.
const TIMED_CYCLES: u32 = 100;
/// Alternating five pairs permit a median without a fixed execution-order bias.
const TIMING_ROUNDS: usize = 5;
/// The boundary fixture leaves this agent dead without a pending respawn.
const INACTIVE_AGENT: u32 = 1;
/// All mutable simulation buffers participate in byte-for-byte parity.
const MUTABLE_BUFFERS: usize = 13;
/// Brain ticks used as integers must remain exactly representable as f32.
const EXACT_FLOAT_INTEGER_LIMIT: u32 = 1 << 24;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn assert_fixture_layout(kernel: &GpuKernel) {
    assert_eq!(kernel.agent_count, FIXTURE_AGENTS);
    assert_eq!(kernel.food_count, FIXTURE_FOOD);
    assert_eq!(kernel.layout.vision_width, FIXTURE_VISION_WIDTH);
    assert_eq!(kernel.layout.vision_height, FIXTURE_VISION_HEIGHT);
    assert!(!kernel.layout.visual_cortex_enabled);
    assert!(!kernel.layout.danger_percept_enabled);
    assert_eq!(kernel.vision_stride, 1);
    assert!(!kernel.probe.brain_beside_vision);
    assert_eq!(kernel.probe.kernel_pass_limit, COMPLETE_BRAIN);
    assert!(!kernel.probe.skip_global && !kernel.probe.skip_vision);
    assert_eq!(kernel.execution_mode, BrainExecutionMode::FusedSerial);
    let expected = BrainLayout::new(FIXTURE_VISION_WIDTH, FIXTURE_VISION_HEIGHT);
    assert_eq!(kernel.layout.feature_count, expected.feature_count);
    assert_eq!(kernel.layout.brain_stride, expected.brain_stride);
    assert_eq!(kernel.layout.sensory_stride, expected.sensory_stride);
    let agents = usize::try_from(kernel.agent_count).unwrap();
    for (buffer, elements) in [
        (&kernel.agent_phys_buffer, agents * PHYS_STRIDE),
        (&kernel.decision_buffer, agents * DECISION_STRIDE),
        (
            &kernel.brain_state_buffer,
            agents * kernel.layout.brain_stride,
        ),
        (&kernel.pattern_buffer, agents * PATTERN_STRIDE),
        (
            &kernel.sensory_buffer,
            agents * kernel.layout.sensory_stride,
        ),
        (&kernel.food_state_buffer, FIXTURE_FOOD * FOOD_STATE_STRIDE),
        (&kernel.food_flags_buffer, FIXTURE_FOOD * FOOD_FLAG_SLOTS),
        (&kernel.heightmap_buffer, TERRAIN_VPS * TERRAIN_VPS),
        (&kernel.biome_buffer, BIOME_GRID_RES * BIOME_GRID_RES),
    ] {
        let bytes = elements.checked_mul(std::mem::size_of::<f32>()).unwrap();
        assert_eq!(buffer.size(), u64::try_from(bytes).unwrap());
    }
}

fn assert_integer_index(value: f32, exclusive_end: usize) {
    assert!(value.is_finite());
    assert!(value >= 0.0);
    assert_eq!(value.trunc().to_bits(), value.to_bits());
    assert!(f64::from(value) < f64::from(u32::try_from(exclusive_end).unwrap()));
}

fn assert_fixture_indices(kernel: &mut GpuKernel) {
    let tail = fixed_tail_base(kernel.layout.brain_stride);
    let offset = |reference| tail + reference - O_PREDICTOR_CONTEXT_WEIGHT;
    for agent in 0..kernel.agent_count {
        let state = kernel.read_agent_state(agent);
        assert_eq!(state.brain_state.len(), kernel.layout.brain_stride);
        assert_eq!(state.patterns.len(), PATTERN_STRIDE);
        assert!(state.brain_state.iter().all(|value| value.is_finite()));
        assert!(state.patterns.iter().all(|value| value.is_finite()));
        assert_integer_index(
            state.brain_state[offset(O_PREDICTION_ERROR_CURSOR)],
            ERROR_HISTORY_LEN,
        );
        assert_integer_index(state.brain_state[offset(O_POS_RING_CURSOR)], POS_RING_LEN);
        assert_integer_index(state.brain_state[offset(O_POS_RING_LEN)], POS_RING_LEN + 1);
        assert_integer_index(state.patterns[O_MIN_REINF_IDX], MEMORY_CAP);
        assert_integer_index(
            state.brain_state[offset(O_TICK_COUNT)],
            usize::try_from(EXACT_FLOAT_INTEGER_LIMIT).unwrap(),
        );
    }
    let physics = kernel.read_full_state_blocking();
    assert!(physics.iter().all(|value| value.is_finite()));
    for agent in physics.as_chunks::<PHYS_STRIDE>().0 {
        // Zero means no claim; valid food IDs are encoded as item + 1.
        assert_integer_index(agent[P_FOOD_CLAIM], FIXTURE_FOOD + 1);
    }
}

fn make_main_pipeline(
    kernel: &GpuKernel,
    source: &str,
    bounds_checks: bool,
) -> wgpu::ComputePipeline {
    let label = if bounds_checks {
        "bounds_checked_cooperative_fused_main"
    } else {
        "bounds_trusted_cooperative_fused_main"
    };
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    };
    let module = if bounds_checks {
        kernel.device.create_shader_module(descriptor)
    } else {
        // SAFETY: Only the fixed seeded fixture below can dispatch this module.
        // assert_fixture_layout validates every variable-size main-entry buffer;
        // assert_fixture_indices checks initial state-derived indices. Modulo
        // cursor updates, bounded food IDs and memory-ID permutations preserve
        // those invariants across the bounded trials. The main entry launches
        // exactly agent_count groups of 256 lanes; its dense and shared indices
        // stay inside the asserted default dimensions. Cortex is disabled and
        // no untrusted restore/upload or arbitrary dimensions are accepted.
        // WGSL validation and loop bounding remain enabled; no fast-math or
        // workgroup-zero-initialization option is changed.
        unsafe {
            kernel.device.create_shader_module_trusted(
                descriptor,
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: false,
                    force_loop_bounding: true,
                },
            )
        }
    };
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kernel_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn prepare_pair() -> (GpuKernel, wgpu::ComputePipeline) {
    let mut kernel = prepare_kernel();
    assert_fixture_layout(&kernel);
    assert_fixture_indices(&mut kernel);
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            &passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    kernel.kernel_pipeline = make_main_pipeline(&kernel, &source, true);
    let trusted = make_main_pipeline(&kernel, &source, false);
    (kernel, trusted)
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    assert!(cycles <= WARMUP_CYCLES);
    let ticks = cycles.checked_mul(kernel.brain_tick_stride).unwrap();
    assert!(start_tick.checked_add(ticks).unwrap() < EXACT_FLOAT_INTEGER_LIMIT);
    kernel.dispatch_ticks(u64::from(start_tick), ticks);
    kernel.poll_wait();
}

#[test]
#[ignore = "restricted GPU bounds-check diagnostic; run explicitly with --ignored --nocapture"]
fn trusted_main_matches_complete_checked_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate) = prepare_pair();
    prepare_boundary_scene(&kernel);
    assert_fixture_indices(&mut kernel);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        assert_fixture_indices(&mut kernel);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] > deaths_before);
    restore(&mut kernel, &initial);
    std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        assert_fixture_indices(&mut kernel);
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
    );
    println!("BOUNDS_PARITY cycles={cycle} exact_buffers={MUTABLE_BUFFERS} dead_agent_unchanged=true death_boundary=true loop_bounding=true production_opt_in=false");
    Ok(())
}

#[test]
#[ignore = "restricted GPU bounds-check benchmark; run in release mode with --ignored --nocapture"]
fn benchmark_trusted_main_against_checked() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate) = prepare_pair();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    assert_fixture_indices(&mut kernel);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut current_arm = 0;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            if arm != current_arm {
                std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate);
                current_arm = arm;
            }
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_fixture_indices(&mut kernel);
            states[arm] = Some(capture_state(&kernel)?);
        }
        assert_state_equal(
            &kernel,
            states[0].as_ref().unwrap(),
            states[1].as_ref().unwrap(),
        );
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let checked = timings[0][TIMING_ROUNDS / 2];
    let trusted = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "BOUNDS_RUNTIME agents={} ticks={ticks} rounds={TIMING_ROUNDS} checked_secs={checked:.9} trusted_secs={trusted:.9} checked_tps={:.3} trusted_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS} loop_bounding=true production_opt_in=false",
        kernel.agent_count, f64::from(ticks) / checked, f64::from(ticks) / trusted, checked / trusted,
    );
    Ok(())
}
