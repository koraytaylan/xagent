//! Hardware-only cumulative section timing of the optimized main kernel.
//! Runtime push-constant guards retain one compiled pipeline for all prefixes.
//! Full execution must match all thirteen persistent buffers; partial prefixes
//! each run one restored cycle and never become a later cycle's input.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, make_kernel, restore};
use super::predictor_fusion::fuse_inline_predictor;
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

fn sources(instrumented: bool) -> String {
    let mut passes = fuse_inline_predictor(&compose_brain_passes(true));
    let mut inner = include_str!("../shaders/kernel/brain_inner.wgsl").to_owned();
    if instrumented {
        let limit = "let limit = kpc.pass_limit;";
        assert_eq!(inner.matches(limit).count(), 1);
        inner = inner.replacen(
            limit,
            &format!("let limit = kpc.pass_limit & {PASS_MASK}u;"),
            1,
        );
        for boundary in BOUNDARIES.iter().filter(|boundary| boundary.stop != 0) {
            assert_eq!(
                passes.matches(boundary.marker).count(),
                1,
                "{}",
                boundary.completed
            );
            passes = passes.replacen(
                boundary.marker,
                &format!(
                    "    if ((kpc.pass_limit >> {SECTION_SHIFT}u) == {}u) {{ return; }}\n{}",
                    boundary.stop, boundary.marker,
                ),
                1,
            );
        }
    }
    [
        include_str!("../shaders/kernel/common.wgsl"),
        passes.as_str(),
        inner.as_str(),
        include_str!("../shaders/kernel/phase_food_claim.wgsl"),
        include_str!("../shaders/kernel/kernel_tick.wgsl"),
    ]
    .join("\n")
}

fn pipeline(kernel: &GpuKernel, instrumented: bool) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain_section_prefixes"),
            source: wgpu::ShaderSource::Wgsl(
                apply_subgroup_markers(&sources(instrumented), kernel.has_subgroup).into(),
            ),
        });
    let bind = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain_section_prefixes"),
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
            label: Some(if instrumented {
                "brain_section_guarded"
            } else {
                "brain_section_reference"
            }),
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
}

impl Timer {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            queries: kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brain_section_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_COUNT,
            }),
            results: kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brain_section_timestamp_results"),
                size: u64::from(QUERY_COUNT) * u64::from(wgpu::QUERY_SIZE),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
        }
    }

    fn measure(&self, kernel: &mut GpuKernel, tick: u32, limit: u32) -> TestResult<f64> {
        kernel.upload_world_config_with_cycles(
            u64::from(tick),
            kernel.kernel_batch_size(),
            PHASE_MASK,
            1,
        );
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
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, limit]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.write_timestamp(&self.queries, QUERY_COUNT - 1);
        }
        encoder.resolve_query_set(&self.queries, 0..QUERY_COUNT, &self.results, 0);
        kernel.queue.submit([encoder.finish()]);
        let bytes = read_buffer(kernel, &self.results, self.results.size())?;
        let timestamps: &[u64] = bytemuck::cast_slice(&bytes);
        assert!(timestamps.windows(2).all(|pair| pair[1] >= pair[0]));
        Ok((timestamps[2] - timestamps[1]) as f64 * f64::from(kernel.queue.get_timestamp_period()))
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
) -> TestResult {
    let initial = checkpoint(kernel);
    kernel.kernel_pipeline = reference.clone();
    advance(kernel, tick, 1);
    let expected = capture_state(kernel)?;
    restore(kernel, &initial);
    kernel.kernel_pipeline = guarded.clone();
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
                kernel.kernel_pipeline = guarded.clone();
                let limit = boundary.pass | (boundary.stop << SECTION_SHIFT);
                samples[index].push(timer.measure(kernel, tick, limit)?);
            } else {
                kernel.kernel_pipeline = reference.clone();
                reference_samples.push(timer.measure(kernel, tick, COMPLETE_BRAIN)?);
            }
        }
    }
    restore(kernel, &initial);
    kernel.kernel_pipeline = reference.clone();
    let reference_nanos = median(&mut reference_samples);
    assert!(
        reference_nanos > 0.0,
        "full-kernel timestamp interval must be positive"
    );
    let mut previous_pass = 0;
    let mut previous = 0.0;
    for (boundary, samples) in BOUNDARIES.iter().zip(&mut samples) {
        let nanos = median(samples);
        let is_start = previous_pass != boundary.pass;
        let delta = if is_start { 0.0 } else { nanos - previous };
        println!("BRAIN_SECTION scene={scene} pass={} completed={} stop={} main_prefix_us={:.3} consecutive_delta_us={:.3} start_boundary={is_start} restored_single_cycle=true measurement_only=true", boundary.pass, boundary.completed, boundary.stop, nanos / NANOS_PER_MICRO, delta / NANOS_PER_MICRO);
        if boundary.pass == COMPLETE_BRAIN && boundary.stop == 0 {
            println!("BRAIN_SECTION_CONTROL scene={scene} reference_full_us={:.3} guarded_full_us={:.3} guarded_over_reference={:.4} full_all13_equal=true rounds={ROUNDS}", reference_nanos / NANOS_PER_MICRO, nanos / NANOS_PER_MICRO, nanos / reference_nanos);
        }
        previous = nanos;
        previous_pass = boundary.pass;
    }
    Ok(())
}

#[test]
#[ignore = "hardware section-prefix diagnostic; run in release mode with --ignored --nocapture"]
fn profile_optimized_brain_sections() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = make_kernel();
    kernel.set_brain_beside_vision(false);
    assert!(kernel
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
    let reference = pipeline(&kernel, false);
    let guarded = pipeline(&kernel, true);
    kernel.kernel_pipeline = reference.clone();
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let mut tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    profile_checkpoint(&mut kernel, &reference, &guarded, tick, "after_256_cycles")?;
    advance(&mut kernel, tick, REFRESH_ADVANCE);
    tick += REFRESH_ADVANCE * kernel.brain_tick_stride;
    profile_checkpoint(&mut kernel, &reference, &guarded, tick, "after_260_cycles")?;
    Ok(())
}
