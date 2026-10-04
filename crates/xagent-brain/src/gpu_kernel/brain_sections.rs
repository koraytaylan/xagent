//! Hardware-only cumulative section timing of the optimized main kernel.
//! Runtime push-constant guards retain one compiled pipeline for all prefixes.
//! Full execution must match all thirteen persistent buffers; partial prefixes
//! each run one restored cycle and never become a later cycle's input.
//! XAGENT_SECTIONS_PREFETCH and XAGENT_SECTIONS_PREDICTOR_LANES select the
//! same production composition for warmup, guarded prefixes, and their control.
//! XAGENT_SECTIONS_PACKED_CONTEXT selects packed encoder, gathered context and
//! global credit; its private cache is imported before timestamped dispatches.
//! A separate frozen-cycle comparison isolates width changes from trajectories.

use std::error::Error;

use super::context_gather_production_validation::control_constants;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, make_kernel, restore};
use super::dense_prefetch::prefetch_passes;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::{wider_predictor, LANE_WIDTHS};
use super::rounding_validation::compare_rounding_state;
use super::vision_validation::read_buffer;
use super::*;

/// Low bits keep the production pass limit; high bits select a section stop.
const SECTION_SHIFT: u32 = 8;
const PASS_MASK: u32 = (1 << SECTION_SHIFT) - 1;
/// Main-kernel push constants remain the original two words.
const PUSH_BYTES: u32 = 8;
/// Prediction/action is the sixth cooperative pass, learning the seventh.
const PREDICT_PASS: u32 = 6;
const COMPLETE_BRAIN: u32 = 7;
/// All simulation phases participate in the full parity check.
const PHASE_MASK: u32 = 7;
/// Mature memory precedes the steady-state checkpoint.
const WARMUP_CYCLES: u32 = 256;
/// Cycle 260 reaches a scheduled refresh for agents that have not respawned.
const REFRESH_ADVANCE: u32 = 4;
/// An odd sample count allows medians while rotating the measured prefix.
const ROUNDS: usize = 5;
/// Boundaries surround claim and main inside one compute pass.
const QUERY_COUNT: u32 = 3;
/// The report uses microseconds rather than raw nanoseconds.
const NANOS_PER_MICRO: f64 = 1_000.0;
/// Production prefetch loads eight independent terms before arithmetic.
const PREFETCH_FACTOR: u32 = 8;
/// The frozen comparison isolates the selected wider production predictor.
const FROZEN_PREDICTOR_LANES: u32 = 16;
/// Production context gathering stages eight output dimensions per tile.
const CONTEXT_LANES: u32 = 8;
/// An odd count permits medians while alternating which arm runs first.
const FROZEN_ROUNDS: usize = 7;
/// The serial cycle preserves these production dispatch boundaries.
const CYCLE_STAGES: [&str; 4] = ["claim", "main", "global", "vision"];
/// One timestamp before the first dispatch and one after each stage.
const CYCLE_QUERY_COUNT: u32 = 5;

#[derive(Clone, Copy)]
struct BrainVariant {
    prefetch: bool,
    predictor_lanes: u32,
    packed_context: bool,
}

impl BrainVariant {
    fn from_env() -> TestResult<Self> {
        let prefetch = std::env::var("XAGENT_SECTIONS_PREFETCH").as_deref() == Ok("1");
        let predictor_lanes = match std::env::var("XAGENT_SECTIONS_PREDICTOR_LANES") {
            Ok(value) => value.parse()?,
            Err(std::env::VarError::NotPresent) => LANE_WIDTHS[0],
            Err(error) => return Err(error.into()),
        };
        if !LANE_WIDTHS.contains(&predictor_lanes) {
            return Err(format!("unsupported section predictor width {predictor_lanes}").into());
        }
        Ok(Self {
            prefetch,
            predictor_lanes,
            packed_context: std::env::var("XAGENT_SECTIONS_PACKED_CONTEXT").as_deref() == Ok("1"),
        })
    }

    fn passes(self) -> String {
        let passes = fuse_inline_predictor(&compose_brain_passes(true));
        let passes = if self.prefetch {
            prefetch_passes(&passes, PREFETCH_FACTOR)
        } else {
            passes
        };
        let passes = wider_predictor(&passes, self.predictor_lanes);
        if self.packed_context {
            context_gather::gather_context(&passes, CONTEXT_LANES)
        } else {
            passes
        }
    }
}

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct Boundary {
    stop: u32,
    pass: u32,
    completed: &'static str,
    marker: &'static str,
}

// Markers are complete unique line prefixes. Every injected return sits at
// function-body scope and depends only on a workgroup-uniform push constant.
const BOUNDARIES: &[Boundary] = &[
    Boundary {
        stop: 0,
        pass: 0,
        completed: "main_before_brain",
        marker: "",
    },
    Boundary {
        stop: 0,
        pass: 1,
        completed: "feature_extraction_and_adaptation",
        marker: "",
    },
    Boundary {
        stop: 0,
        pass: 2,
        completed: "encoder",
        marker: "",
    },
    Boundary {
        stop: 0,
        pass: 3,
        completed: "habituation_and_homeostasis",
        marker: "",
    },
    Boundary {
        stop: 0,
        pass: 4,
        completed: "recall_scoring",
        marker: "",
    },
    Boundary {
        stop: 0,
        pass: 5,
        completed: "recall_selection",
        marker: "",
    },
    Boundary {
        stop: 1,
        pass: PREDICT_PASS,
        completed: "before_predict",
        marker: "    // ── Predictor: train then predict",
    },
    Boundary {
        stop: 2,
        pass: PREDICT_PASS,
        completed: "predictor_train_and_dot",
        marker: "    // ── Recalled cosine similarities:",
    },
    Boundary {
        stop: 3,
        pass: PREDICT_PASS,
        completed: "recall_context_and_tanh",
        marker: "    // ── Prediction error reduction",
    },
    Boundary {
        stop: 4,
        pass: PREDICT_PASS,
        completed: "prediction_error_and_history",
        marker: "    // ── Homeostatic gradient predictor head",
    },
    Boundary {
        stop: 5,
        pass: PREDICT_PASS,
        completed: "homeostatic_predictor",
        marker: "    // ── TD(λ) credit:",
    },
    Boundary {
        stop: 6,
        pass: PREDICT_PASS,
        completed: "critic_normalizer_and_td_updates",
        marker: "    // ── Weight normalization:",
    },
    Boundary {
        stop: 7,
        pass: PREDICT_PASS,
        completed: "three_weight_norms_and_scales",
        marker: "    // ── Weight rescaling:",
    },
    Boundary {
        stop: 8,
        pass: PREDICT_PASS,
        completed: "weight_rescaling",
        marker: "    // ── Policy dot product reductions",
    },
    Boundary {
        stop: 9,
        pass: PREDICT_PASS,
        completed: "two_policy_dots",
        marker: "    // ── Attenuation sum reduction",
    },
    Boundary {
        stop: 10,
        pass: PREDICT_PASS,
        completed: "attenuation_sum",
        marker: "    cooperative_refresh_vision_whitening(brain_base, tid);",
    },
    Boundary {
        stop: 11,
        pass: PREDICT_PASS,
        completed: "cooperative_whitening_refresh",
        marker: "    // ── Thread 0: exploration, noise, motor, telemetry",
    },
    Boundary {
        stop: 12,
        pass: PREDICT_PASS,
        completed: "scent_vision_motor_and_telemetry",
        marker: "    // ── Per-dimension vector copies:",
    },
    Boundary {
        stop: 13,
        pass: PREDICT_PASS,
        completed: "prediction_and_credit_publication",
        marker: "    // ── Eligibility trace update",
    },
    Boundary {
        stop: 0,
        pass: PREDICT_PASS,
        completed: "eligibility_traces",
        marker: "",
    },
    Boundary {
        stop: 20,
        pass: COMPLETE_BRAIN,
        completed: "before_learn",
        marker: "    // Thread 0: context weight adaptation,",
    },
    Boundary {
        stop: 21,
        pass: COMPLETE_BRAIN,
        completed: "context_adaptation_and_encoder_credit",
        marker: "    // ── Compute memory-key norm ONCE",
    },
    Boundary {
        stop: 22,
        pass: COMPLETE_BRAIN,
        completed: "memory_key_norm",
        marker: "    // ── 7c. Memory reinforcement",
    },
    Boundary {
        stop: 23,
        pass: COMPLETE_BRAIN,
        completed: "memory_reinforcement",
        marker: "    // ── 7d. Memory store:",
    },
    Boundary {
        stop: 24,
        pass: COMPLETE_BRAIN,
        completed: "memory_store",
        marker: "    // ── 7e. Memory decay:",
    },
    Boundary {
        stop: 25,
        pass: COMPLETE_BRAIN,
        completed: "memory_decay",
        marker: "    // ── 7f. Min tracking + active count:",
    },
    Boundary {
        stop: 26,
        pass: COMPLETE_BRAIN,
        completed: "active_count",
        marker: "    // Argmin with first-minimum tie-break",
    },
    Boundary {
        stop: 27,
        pass: COMPLETE_BRAIN,
        completed: "eviction_argmin",
        marker: "    // ── 7g. Publish this tick's encoded state",
    },
    Boundary {
        stop: 28,
        pass: COMPLETE_BRAIN,
        completed: "encoded_state_and_mean_publication",
        marker: "    // ── 7h. Recent-experience value replay",
    },
    Boundary {
        stop: 29,
        pass: COMPLETE_BRAIN,
        completed: "recent_returns_and_store",
        marker: "    // One settled moment per brain tick,",
    },
    Boundary {
        stop: 0,
        pass: COMPLETE_BRAIN,
        completed: "critic_replay",
        marker: "",
    },
];

fn sources(kernel: &GpuKernel, instrumented: bool, variant: BrainVariant) -> String {
    let passes = variant.passes();
    let mut source = if variant.packed_context {
        let credit = kernel.global_credit.as_ref().unwrap();
        global_credit::main_source(&passes, kernel.has_subgroup, credit.packed_encoder.as_ref())
    } else {
        apply_subgroup_markers(
            &[
                include_str!("../shaders/kernel/common.wgsl"),
                passes.as_str(),
                include_str!("../shaders/kernel/brain_inner.wgsl"),
                include_str!("../shaders/kernel/phase_food_claim.wgsl"),
                include_str!("../shaders/kernel/kernel_tick.wgsl"),
            ]
            .join("\n"),
            kernel.has_subgroup,
        )
    };
    if instrumented {
        let limit = "let limit = kpc.pass_limit;";
        assert_eq!(source.matches(limit).count(), 1);
        source = source.replacen(
            limit,
            &format!("let limit = kpc.pass_limit & {PASS_MASK}u;"),
            1,
        );
        for boundary in BOUNDARIES.iter().filter(|boundary| boundary.stop != 0) {
            assert_eq!(
                source.matches(boundary.marker).count(),
                1,
                "{}",
                boundary.completed
            );
            source = source.replacen(
                boundary.marker,
                &format!(
                    "    if ((kpc.pass_limit >> {SECTION_SHIFT}u) == {}u) {{ return; }}\n{}",
                    boundary.stop, boundary.marker,
                ),
                1,
            );
        }
    }
    source
}

fn select_main(kernel: &mut GpuKernel, pipeline: &wgpu::ComputePipeline) {
    if let Some(credit) = &mut kernel.global_credit {
        credit.main = pipeline.clone();
    } else {
        kernel.kernel_pipeline = pipeline.clone();
    }
}

fn pipeline(
    kernel: &GpuKernel,
    instrumented: bool,
    variant: BrainVariant,
) -> wgpu::ComputePipeline {
    let label = format!(
        "brain_section_guarded_{instrumented}_prefetch_{}_lanes_{}",
        variant.prefetch, variant.predictor_lanes,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(sources(kernel, instrumented, variant).into()),
        });
    let bind = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &[&bind],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_BYTES,
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
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

struct Timer {
    queries: wgpu::QuerySet,
    results: wgpu::Buffer,
    query_count: u32,
}

impl Timer {
    fn new(kernel: &GpuKernel) -> Self {
        Self::with_query_count(kernel, QUERY_COUNT)
    }

    fn with_query_count(kernel: &GpuKernel, query_count: u32) -> Self {
        Self {
            queries: kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brain_section_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: query_count,
            }),
            results: kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brain_section_timestamp_results"),
                size: u64::from(query_count) * u64::from(wgpu::QUERY_SIZE),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            query_count,
        }
    }

    fn finish(
        &self,
        kernel: &GpuKernel,
        mut encoder: wgpu::CommandEncoder,
    ) -> TestResult<Vec<f64>> {
        encoder.resolve_query_set(&self.queries, 0..self.query_count, &self.results, 0);
        kernel.queue.submit([encoder.finish()]);
        let bytes = read_buffer(kernel, &self.results, self.results.size())?;
        let timestamps: &[u64] = bytemuck::cast_slice(&bytes);
        assert!(timestamps.windows(2).all(|pair| pair[1] >= pair[0]));
        Ok(timestamps
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64 * f64::from(kernel.queue.get_timestamp_period()))
            .collect())
    }

    fn measure(&self, kernel: &mut GpuKernel, tick: u32, limit: u32) -> TestResult<f64> {
        kernel.upload_world_config_with_cycles(
            u64::from(tick),
            kernel.kernel_batch_size(),
            PHASE_MASK,
            1,
        );
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        if let Some(cache) = kernel
            .global_credit
            .as_ref()
            .and_then(|credit| credit.packed_encoder.as_ref())
        {
            cache.record_import(kernel, &mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            let groups = kernel
                .global_credit
                .as_ref()
                .map_or(&kernel.bind_groups, |credit| &credit.bind_groups);
            pass.set_bind_group(0, &groups[kernel.active_config_index], &[]);
            pass.write_timestamp(&self.queries, 0);
            pass.set_pipeline(&kernel.kernel_claim_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.write_timestamp(&self.queries, 1);
            if let Some(credit) = &kernel.global_credit {
                pass.set_pipeline(&credit.main);
            } else {
                pass.set_pipeline(&kernel.kernel_pipeline);
            }
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, limit]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.write_timestamp(&self.queries, QUERY_COUNT - 1);
        }
        Ok(self.finish(kernel, encoder)?[1])
    }

    fn measure_cycle(
        &self,
        kernel: &mut GpuKernel,
        tick: u32,
    ) -> TestResult<[f64; CYCLE_STAGES.len()]> {
        assert_eq!(self.query_count, CYCLE_QUERY_COUNT);
        assert_eq!(kernel.vision_stride, 1);
        let batch_ticks = kernel.kernel_batch_size();
        kernel.upload_world_config_with_cycles(u64::from(tick), batch_ticks, PHASE_MASK, 1);
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
            pass.write_timestamp(&self.queries, 0);
            pass.set_pipeline(&kernel.kernel_claim_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.write_timestamp(&self.queries, 1);
            pass.set_pipeline(&kernel.kernel_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.write_timestamp(&self.queries, 2);
            pass.set_pipeline(&kernel.global_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick + batch_ticks, batch_ticks]));
            pass.dispatch_workgroups(1, 1, 1);
            pass.write_timestamp(&self.queries, 3);
            pass.set_pipeline(&kernel.vision_pipeline);
            pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
            pass.write_timestamp(&self.queries, CYCLE_QUERY_COUNT - 1);
        }
        let intervals = self.finish(kernel, encoder)?;
        kernel.active_config_index = 1 - kernel.active_config_index;
        Ok(intervals.try_into().unwrap())
    }
}

fn advance(kernel: &mut GpuKernel, tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn profile_checkpoint(
    kernel: &mut GpuKernel,
    reference: &wgpu::ComputePipeline,
    guarded: &wgpu::ComputePipeline,
    tick: u32,
    scene: &str,
    variant: BrainVariant,
) -> TestResult {
    let initial = checkpoint(kernel);
    select_main(kernel, reference);
    advance(kernel, tick, 1);
    let expected = capture_state(kernel)?;
    restore(kernel, &initial);
    select_main(kernel, guarded);
    advance(kernel, tick, 1);
    assert_state_equal(kernel, &expected, &capture_state(kernel)?);
    restore(kernel, &initial);

    let timer = Timer::new(kernel);
    let mut samples: Vec<Vec<f64>> = BOUNDARIES.iter().map(|_| Vec::new()).collect();
    let mut reference_samples = Vec::new();
    for round in 0..ROUNDS {
        // Include the unguarded full kernel in the rotation to expose any
        // instrumentation-induced compiler or scheduling cost explicitly.
        for offset in 0..=BOUNDARIES.len() {
            let index = (round + offset) % (BOUNDARIES.len() + 1);
            restore(kernel, &initial);
            if let Some(boundary) = BOUNDARIES.get(index) {
                select_main(kernel, guarded);
                let limit = boundary.pass | (boundary.stop << SECTION_SHIFT);
                samples[index].push(timer.measure(kernel, tick, limit)?);
            } else {
                select_main(kernel, reference);
                reference_samples.push(timer.measure(kernel, tick, COMPLETE_BRAIN)?);
            }
        }
    }
    restore(kernel, &initial);
    select_main(kernel, reference);
    let reference_nanos = median(&mut reference_samples);
    assert!(
        reference_nanos > 0.0,
        "full-kernel timestamp interval must be positive"
    );
    let mut previous = 0.0;
    for (index, (boundary, samples)) in BOUNDARIES.iter().zip(&mut samples).enumerate() {
        let nanos = median(samples);
        let completed = if variant.packed_context && boundary.stop == 21 {
            "context_adaptation_and_feature_publication"
        } else {
            boundary.completed
        };
        let is_start = index == 0;
        let delta = if is_start { 0.0 } else { nanos - previous };
        println!("BRAIN_SECTION scene={scene} prefetch={} predictor_lanes={} packed_context={} pass={} completed={} stop={} main_prefix_us={:.3} consecutive_delta_us={:.3} start_boundary={is_start} restored_single_cycle=true measurement_only=true", variant.prefetch, variant.predictor_lanes, variant.packed_context, boundary.pass, completed, boundary.stop, nanos / NANOS_PER_MICRO, delta / NANOS_PER_MICRO);
        if boundary.pass == COMPLETE_BRAIN && boundary.stop == 0 {
            println!("BRAIN_SECTION_CONTROL scene={scene} prefetch={} predictor_lanes={} packed_context={} reference_full_us={:.3} guarded_full_us={:.3} guarded_over_reference={:.4} full_all13_equal=true rounds={ROUNDS}", variant.prefetch, variant.predictor_lanes, variant.packed_context, reference_nanos / NANOS_PER_MICRO, nanos / NANOS_PER_MICRO, nanos / reference_nanos);
        }
        previous = nanos;
    }
    Ok(())
}

#[test]
#[ignore = "hardware section-prefix diagnostic; run in release mode with --ignored --nocapture"]
fn profile_optimized_brain_sections() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = make_kernel();
    // Prefix and complete-cycle arms must execute their selected main shader.
    kernel.global_credit = None;
    kernel.set_brain_beside_vision(false);
    assert!(kernel
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
    let variant = BrainVariant::from_env()?;
    if variant.packed_context {
        kernel.global_credit = global_credit::Pipelines::new_packed(
            &kernel,
            &variant.passes(),
            &control_constants(&kernel),
        );
        assert!(kernel.global_credit_active());
        assert!(kernel
            .global_credit
            .as_ref()
            .unwrap()
            .packed_encoder
            .is_some());
    }
    let reference = pipeline(&kernel, false, variant);
    let guarded = pipeline(&kernel, true, variant);
    select_main(&mut kernel, &reference);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let mut tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    profile_checkpoint(
        &mut kernel,
        &reference,
        &guarded,
        tick,
        "after_256_cycles",
        variant,
    )?;
    advance(&mut kernel, tick, REFRESH_ADVANCE);
    tick += REFRESH_ADVANCE * kernel.brain_tick_stride;
    profile_checkpoint(
        &mut kernel,
        &reference,
        &guarded,
        tick,
        "after_260_cycles",
        variant,
    )?;
    Ok(())
}

fn benchmark_frozen_checkpoint(
    kernel: &mut GpuKernel,
    pipelines: &[wgpu::ComputePipeline; 2],
    tick: u32,
    scene: &str,
) -> TestResult {
    let initial = checkpoint(kernel);
    let mut expected = Vec::with_capacity(pipelines.len());
    for pipeline in pipelines {
        restore(kernel, &initial);
        kernel.kernel_pipeline = pipeline.clone();
        advance(kernel, tick, 1);
        expected.push(capture_state(kernel)?);
    }
    compare_rounding_state(
        kernel,
        &expected[0],
        &expected[1],
        &format!("frozen_predictor/{scene}"),
    );
    let timer = Timer::with_query_count(kernel, CYCLE_QUERY_COUNT);
    // Validate timestamp recording against each arm's real dispatch_ticks
    // result, and warm both timed paths before collecting their measurements.
    for (arm, pipeline) in pipelines.iter().enumerate() {
        restore(kernel, &initial);
        kernel.kernel_pipeline = pipeline.clone();
        timer.measure_cycle(kernel, tick)?;
        assert_state_equal(kernel, &expected[arm], &capture_state(kernel)?);
    }
    let mut samples: [Vec<[f64; CYCLE_STAGES.len()]>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..FROZEN_ROUNDS {
        for offset in 0..pipelines.len() {
            let arm = (round + offset) % pipelines.len();
            restore(kernel, &initial);
            kernel.kernel_pipeline = pipelines[arm].clone();
            samples[arm].push(timer.measure_cycle(kernel, tick)?);
        }
    }
    restore(kernel, &initial);
    kernel.kernel_pipeline = pipelines[0].clone();
    for (stage, name) in CYCLE_STAGES.iter().enumerate() {
        let medians = samples.each_ref().map(|samples| {
            median(
                &mut samples
                    .iter()
                    .map(|sample| sample[stage])
                    .collect::<Vec<_>>(),
            )
        });
        assert!(medians.iter().all(|&nanos| nanos > 0.0));
        println!("FROZEN_PREDICTOR_STAGE scene={scene} stage={name} prefetch=true baseline_lanes={} candidate_lanes={FROZEN_PREDICTOR_LANES} baseline_us={:.3} candidate_us={:.3} speedup={:.4} rounds={FROZEN_ROUNDS} identical_start=true", LANE_WIDTHS[0], medians[0] / NANOS_PER_MICRO, medians[1] / NANOS_PER_MICRO, medians[0] / medians[1]);
    }
    let medians = samples.each_ref().map(|samples| {
        median(
            &mut samples
                .iter()
                .map(|sample| sample.iter().sum())
                .collect::<Vec<f64>>(),
        )
    });
    println!("FROZEN_PREDICTOR_CYCLE scene={scene} prefetch=true baseline_lanes={} candidate_lanes={FROZEN_PREDICTOR_LANES} baseline_us={:.3} candidate_us={:.3} speedup={:.4} rounds={FROZEN_ROUNDS} restore_and_readback_excluded=true each_arm_timestamp_all13_equal=true warmup_lanes={}", LANE_WIDTHS[0], medians[0] / NANOS_PER_MICRO, medians[1] / NANOS_PER_MICRO, medians[0] / medians[1], LANE_WIDTHS[0]);
    Ok(())
}

#[test]
#[ignore = "frozen-input GPU timestamps; run in release mode with --ignored --nocapture"]
fn benchmark_frozen_predictor_width_cycles() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = make_kernel();
    // Each frozen arm selects its own main, including the matching warmup.
    kernel.global_credit = None;
    kernel.set_brain_beside_vision(false);
    assert!(kernel
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
    let pipelines = [LANE_WIDTHS[0], FROZEN_PREDICTOR_LANES].map(|predictor_lanes| {
        pipeline(
            &kernel,
            false,
            BrainVariant {
                prefetch: true,
                predictor_lanes,
                packed_context: false,
            },
        )
    });
    kernel.kernel_pipeline = pipelines[0].clone();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let mut tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    benchmark_frozen_checkpoint(&mut kernel, &pipelines, tick, "after_256_cycles")?;
    advance(&mut kernel, tick, REFRESH_ADVANCE);
    tick += REFRESH_ADVANCE * kernel.brain_tick_stride;
    benchmark_frozen_checkpoint(&mut kernel, &pipelines, tick, "after_260_cycles")?;
    Ok(())
}
