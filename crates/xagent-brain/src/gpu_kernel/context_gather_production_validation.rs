//! Compare constructor-created context-gather pipelines with an independent
//! ungathered control through the public dispatch routes. The control matches
//! the process's other brain options; the candidate pipelines are never rebuilt.
//! Each route compares its own schedule, including every mutable public buffer.

use std::{collections::HashMap, error::Error};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::upload_random_scene;
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// One agent dies, another stays inactive, and a third keeps its learned state.
const AGENTS: u32 = 3;
const INACTIVE_AGENT: u32 = 1;
const FOOD_ITEMS: usize = 104;
/// An odd retina exercises dynamic offsets; cortex uses its own feature layout.
const FIELDS: [(u32, u32, bool); 2] = [(9, 7, false), (8, 6, true)];
/// Visit both sides of two whitening refresh boundaries in the raw trajectory.
const RAW_CHUNKS: [u32; 6] = [1, 18, 1, 1, 19, 1];
/// Cortex starts next to a refresh boundary and uses short, polled submissions.
const CORTEX_CHUNKS: [u32; 3] = [1, 2, 1];
/// Complete dispatches enable physics, vision and all seven brain passes.
const COMPLETE_PHASES: u32 = 7;
const BRAIN_ONLY: u32 = 4;
/// The fallback route groups two brain cycles before global and vision.
const ALTERNATE_VISION_STRIDE: u32 = 2;
/// Keep every brain option except context gather equal to its constructor value.
const PREFETCH_FACTOR: u32 = 8;
const PUSH_CONSTANT_BYTES: u32 = 8;
const BRAIN_SEED: u64 = 42;
const MUTABLE_BUFFERS: usize = 13;
/// Antipodal unit-axis keys ensure both recall halves contain supplied patterns.
const PATTERN_SIGNS: usize = 2;
const SEEDED_PATTERNS: usize = RECALL_K * PATTERN_SIGNS;
const PATTERN_COMPONENT: f32 = 0.5;
const EXPECTED_DEATHS: f32 = 2.0;
/// The third agent is alive initially and retains the supplied recall fixture.
const RECALL_AGENT: usize = 2;
const PATTERN_BUFFER_INDEX: usize = 10;
/// Pattern metadata stores creation tick, last-use tick and usage count.
const PATTERN_METADATA_WORDS: usize = 3;
const PATTERN_USAGE_COLUMN: usize = PATTERN_METADATA_WORDS - 1;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn enabled(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

fn control_passes() -> String {
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
    assert!(!passes.contains("blend_gathered_recalled_context"));
    assert!(passes.contains("pattern_buffer[pattern_base + tid * MEMORY_CAP + idx] + encoded_mean"));
    passes
}

fn control_constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    let rays = kernel.layout.vision_width * kernel.layout.vision_height;
    // These retinas distinguish the constructor's serial and parallel combined
    // group counts, so the retained dispatch shape identifies its actual option.
    let serial_groups = kernel.agent_count * (1 + rays.div_ceil(BRAIN_WORKGROUP_THREADS));
    let parallel = kernel.brain_vision_workgroups != serial_groups;
    if parallel {
        let lanes = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
        assert_eq!(
            kernel.brain_vision_workgroups,
            kernel.agent_count * (1 + rays.div_ceil(lanes))
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

struct Control {
    main: wgpu::ComputePipeline,
    claim: wgpu::ComputePipeline,
    standalone: wgpu::ComputePipeline,
    combined: wgpu::ComputePipeline,
    tail: wgpu::ComputePipeline,
    credit: Option<global_credit::Pipelines>,
}

impl Control {
    fn new(kernel: &GpuKernel) -> Self {
        let passes = control_passes();
        let common = include_str!("../shaders/kernel/common.wgsl");
        let constants = control_constants(kernel);
        let kernel_source = [
            common,
            &passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n");
        let standalone = [
            common,
            &passes,
            include_str!("../shaders/kernel/brain_tick.wgsl"),
        ]
        .join("\n");
        let vision = [
            include_str!("../shaders/kernel/phase_vision.wgsl"),
            include_str!("../shaders/kernel/phase_vision_parallel.wgsl"),
        ]
        .join("\n")
        .replace("sensory_buffer[", "sensory_next[");
        assert!(!vision.contains("sensory_buffer"));
        let combined = [
            with_plain_grid_bindings(common),
            passes.clone(),
            include_str!("../shaders/kernel/brain_inner.wgsl").to_owned(),
            vision,
            include_str!("../shaders/kernel/brain_vision_tick.wgsl").to_owned(),
        ]
        .join("\n");
        let tail = [
            common,
            &passes,
            include_str!("../shaders/kernel/phase_brain_tail_from_scratch.wgsl"),
        ]
        .join("\n");
        let build = |source: &str, entry: &str, push: bool| {
            let module = kernel
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(
                        apply_subgroup_markers(source, kernel.has_subgroup).into(),
                    ),
                });
            let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
            let push_ranges = [wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }];
            let layout = kernel
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(entry),
                    bind_group_layouts: &[&bind_layout],
                    push_constant_ranges: if push { &push_ranges } else { &[] },
                });
            kernel
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                })
        };
        Self {
            main: build(&kernel_source, "kernel_tick", true),
            claim: build(&kernel_source, "kernel_claim_tick", true),
            standalone: build(&standalone, "brain_tick", false),
            combined: build(&combined, "brain_vision_tick", true),
            tail: build(&tail, "phase_brain_tail_from_scratch", false),
            credit: kernel
                .global_credit
                .as_ref()
                .map(|_| global_credit::Pipelines::new(kernel, &passes, &constants).unwrap()),
        }
    }

    /// Each second swap restores the exact constructor-created candidate handles.
    fn swap(&mut self, kernel: &mut GpuKernel) {
        std::mem::swap(&mut self.main, &mut kernel.kernel_pipeline);
        std::mem::swap(&mut self.claim, &mut kernel.kernel_claim_pipeline);
        std::mem::swap(&mut self.standalone, &mut kernel.brain_pipeline);
        std::mem::swap(&mut self.combined, &mut kernel.brain_vision_pipeline);
        std::mem::swap(&mut self.tail, &mut kernel.tail_pipeline);
        std::mem::swap(&mut self.credit, &mut kernel.global_credit);
    }
}

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
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_PHASES;
    upload_random_scene(&kernel, 0, true, false);
    prepare_boundary_scene(&kernel);
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.patterns.fill(0.0);
        for pattern in 0..SEEDED_PATTERNS {
            let dim = pattern / PATTERN_SIGNS;
            state.patterns[O_PAT_STATES + dim * MEMORY_CAP + pattern] =
                if pattern % PATTERN_SIGNS == 0 {
                    PATTERN_COMPONENT
                } else {
                    -PATTERN_COMPONENT
                };
            state.patterns[O_PAT_NORMS + pattern] = PATTERN_COMPONENT;
            state.patterns[O_PAT_REINF + pattern] = 1.0;
            state.patterns[O_PAT_ACTIVE + pattern] = 1.0;
        }
        state.patterns[O_ACTIVE_COUNT] = f32::from(u16::try_from(SEEDED_PATTERNS).unwrap());
        state.patterns[O_LAST_STORED_IDX] = f32::from(u16::try_from(SEEDED_PATTERNS).unwrap());
        state.patterns[O_MIN_REINF_IDX] = f32::from(u16::try_from(SEEDED_PATTERNS).unwrap());
        if cortex {
            let tick = fixed_tail_base(kernel.layout.brain_stride) + O_TICK_COUNT
                - O_PREDICTOR_CONTEXT_WEIGHT;
            state.brain_state[tick] = f32::from(u16::try_from(REFRESH_CYCLES - 1).unwrap());
        }
        kernel.write_agent_state(agent, &state);
    }
    kernel
}

#[derive(Clone, Copy, Debug)]
enum Route {
    FusedSeparate,
    FusedOverlapRequest,
    Split,
    Tiled,
    VisionStride,
    MaskedFull,
    MaskedBrain,
}

const ROUTES: [Route; 7] = [
    Route::FusedSeparate,
    Route::FusedOverlapRequest,
    Route::Split,
    Route::Tiled,
    Route::VisionStride,
    Route::MaskedFull,
    Route::MaskedBrain,
];

fn configure(kernel: &mut GpuKernel, route: Route) {
    kernel.set_execution_mode(match route {
        Route::Split => BrainExecutionMode::SplitSerial,
        Route::Tiled => BrainExecutionMode::ParallelTiled,
        _ => BrainExecutionMode::FusedSerial,
    });
    kernel.vision_stride = if matches!(route, Route::VisionStride) {
        ALTERNATE_VISION_STRIDE
    } else {
        1
    };
    kernel.set_brain_beside_vision(matches!(route, Route::FusedOverlapRequest));
    let offload = kernel.global_credit.is_some()
        && matches!(route, Route::FusedSeparate | Route::FusedOverlapRequest);
    if !matches!(route, Route::MaskedFull | Route::MaskedBrain) {
        assert_eq!(
            kernel.global_credit_active(),
            offload,
            "{route:?}: dispatch selection"
        );
    }
}

fn trajectory(kernel: &mut GpuKernel, route: Route) -> TestResult<Vec<State>> {
    let cortex = kernel.layout.visual_cortex_enabled;
    let chunks: &[u32] = if cortex { &CORTEX_CHUNKS } else { &RAW_CHUNKS };
    let death_cycle = if cortex { 1 } else { REFRESH_CYCLES };
    let mut cycle = 0;
    let mut states = Vec::new();
    for &cycles in chunks {
        if cycle == death_cycle {
            force_death(kernel);
        }
        let end = cycle + cycles;
        while cycle < end {
            let batch = if cortex { 1 } else { end - cycle };
            let start_tick = u64::from(cycle * kernel.brain_tick_stride);
            let ticks = batch * kernel.brain_tick_stride;
            match route {
                Route::MaskedFull => {
                    kernel.dispatch_batch_masked(start_tick, ticks, COMPLETE_PHASES)
                }
                Route::MaskedBrain => kernel.dispatch_batch_masked(start_tick, ticks, BRAIN_ONLY),
                _ => {
                    kernel.dispatch_ticks(start_tick, ticks);
                }
            }
            kernel.poll_wait();
            cycle += batch;
        }
        let state = capture_state(kernel)?;
        assert_eq!(state.len(), MUTABLE_BUFFERS);
        states.push(state);
    }
    if !matches!(route, Route::MaskedBrain) {
        assert!(
            kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_DEATHS,
            "{route:?}: both forced deaths must execute"
        );
    }
    Ok(states)
}

fn assert_recall_coverage(state: &State) {
    let patterns: &[f32] = bytemuck::cast_slice(&state[PATTERN_BUFFER_INDEX]);
    let base = RECALL_AGENT * PATTERN_STRIDE + O_PAT_META;
    let used = (0..SEEDED_PATTERNS)
        .filter(|pattern| {
            patterns[base + pattern * PATTERN_METADATA_WORDS + PATTERN_USAGE_COLUMN] >= 1.0
        })
        .count();
    assert!(
        used >= RECALL_K,
        "the first cycle must recall a complete context window"
    );
}

#[test]
#[ignore = "requires GPU and XAGENT_BRAIN_CONTEXT_GATHER=1; run explicitly"]
fn production_context_gather_preserves_dispatch_routes() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    assert!(
        enabled("XAGENT_BRAIN_CONTEXT_GATHER"),
        "run with XAGENT_BRAIN_CONTEXT_GATHER=1"
    );
    for (width, height, cortex) in FIELDS {
        println!("PRODUCTION_CONTEXT_GATHER_STAGE width={width} height={height} cortex={cortex} stage=construct");
        let mut kernel = prepare(width, height, cortex);
        let mut control = Control::new(&kernel);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        for route in ROUTES {
            configure(&mut kernel, route);
            restore(&mut kernel, &initial);
            control.swap(&mut kernel);
            let expected = trajectory(&mut kernel, route)?;
            assert_recall_coverage(&expected[0]);
            control.swap(&mut kernel);
            restore(&mut kernel, &initial);
            let actual = trajectory(&mut kernel, route)?;
            assert_recall_coverage(&actual[0]);
            assert_eq!(expected.len(), actual.len());
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    actual,
                    INACTIVE_AGENT,
                    "production context gather",
                );
            }
            let combined = matches!(route, Route::FusedOverlapRequest)
                && !kernel.global_credit_active()
                && !kernel.standalone_vision_required;
            println!("PRODUCTION_CONTEXT_GATHER route={route:?} width={width} height={height} cortex={cortex} exact_buffers={MUTABLE_BUFFERS} candidate=constructor control=context_disabled other_flags=matched actual_combined={combined} bounded_cortex=true seeded_patterns={SEEDED_PATTERNS}");
        }
    }
    Ok(())
}
