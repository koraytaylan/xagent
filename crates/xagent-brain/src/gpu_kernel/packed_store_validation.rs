//! Validate production suppression of encoder stores whose updated FP32 words are
//! already bit-identical to the loaded vector. Both arms retain the production
//! packed cache, scalar mirror, arithmetic, dispatch recorder and lifecycle.
//! Only the candidate global shader changes; no counters enter either shader.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::packed_encoder_validation::{assert_credit_cases, credit_fixture};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP_CYCLES: [u32; 2] = [256, 1_000];
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
const CONTEXT_LANES: u32 = 8;
const VECTOR_WORDS: usize = 4;
const PUSH_CONSTANT_BYTES: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();
const MUTABLE_BUFFERS: usize = 13;
const BRAIN_BUFFER: usize = 8;
const INACTIVE_AGENT: u32 = 1;
const CREDIT_THRESHOLD: f32 = 1e-6;
const LARGE_CREDIT: f32 = 2.0;
const LARGE_FEATURE: f32 = 10_000.0;
const SMALL_FEATURE: f32 = 1e-20;
const ENCODER_LIMIT: f32 = 2.0;
const NEAR_LIMIT: f32 = 1.9;
const QUARTER_WEIGHT: f32 = 0.25;

const CREDIT_ENTRY: &str = r"
@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn encoder_credit_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    phase_encoder_credit(group.y, group.x * ENCODER_CREDIT_THREADS + tid);
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn optimized_brain() -> String {
    let fused = predictor_fusion::fuse_inline_predictor(&compose_brain_passes(true));
    let prefetched = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
    let widened = predictor_width::wider_predictor(&prefetched, PREDICTOR_LANES);
    context_gather::gather_context(&widened, CONTEXT_LANES)
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

/// Explicit production baseline, shared with snapshot-only opportunity probes.
/// The fixture fixes its vision and brain pipelines independently of process flags.
pub(super) fn prepare_kernel(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = super::cached_combined_validation::prepare(width, height, boundary);
    kernel.global_credit = Pipelines::new_packed(&kernel, &optimized_brain(), &constants(&kernel));
    assert!(kernel.global_credit_active());
    assert!(!cache(&kernel).is_valid());
    kernel
}

fn cache(kernel: &GpuKernel) -> &packed_encoder::Cache {
    kernel
        .global_credit
        .as_ref()
        .unwrap()
        .packed_encoder
        .as_ref()
        .unwrap()
}

fn pipeline(
    kernel: &GpuKernel,
    source: String,
    entry: &str,
    constants: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packed_store_candidate"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed_store_candidate"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("packed_store_candidate"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: Option<Pipelines>,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        let candidate = Pipelines::new_packed_with_store_suppression(
            kernel,
            &optimized_brain(),
            &constants(kernel),
            true,
        )
        .unwrap();
        Self {
            parked: Some(candidate),
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            // A checkpoint may have been restored while this cache was parked.
            self.parked
                .as_ref()
                .unwrap()
                .packed_encoder
                .as_ref()
                .unwrap()
                .invalidate();
            std::mem::swap(&mut kernel.global_credit, &mut self.parked);
            self.candidate = candidate;
        }
        assert!(kernel.global_credit_active());
    }
}

pub(super) fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    assert!(kernel.global_credit_active());
    kernel.dispatch_ticks(
        u64::from(cycle.checked_mul(kernel.brain_tick_stride).unwrap()),
        cycles.checked_mul(kernel.brain_tick_stride).unwrap(),
    );
    kernel.poll_wait();
}

/// Public state is compared without exporting private weights. Independently
/// assert the cache/mirror invariant that permits skipping both destinations.
fn assert_mirror(kernel: &GpuKernel, state: &State) -> TestResult {
    let packed = cache(kernel).buffer();
    let actual = read_buffer(kernel, packed, packed.size())?;
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let scratch_words = kernel
        .layout
        .brain_scratch_stride
        .checked_mul(agents)
        .unwrap();
    let prefix_words = scratch_words.div_ceil(VECTOR_WORDS) * VECTOR_WORDS;
    let matrix_words = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    for agent in 0..agents {
        let private_begin = usize::try_from(bytes(prefix_words + agent * matrix_words)).unwrap();
        let scalar_begin = usize::try_from(bytes(agent * kernel.layout.brain_stride)).unwrap();
        let length = usize::try_from(bytes(matrix_words)).unwrap();
        assert_eq!(
            &actual[private_begin..private_begin + length],
            &state[BRAIN_BUFFER][scalar_begin..scalar_begin + length],
            "packed/scalar mirror agent={agent}",
        );
    }
    Ok(())
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
        cycle += count;
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; exact full-state and cache-mirror comparison"]
fn packed_store_suppression_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel(width, height, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for replay in 0..2 {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "packed store suppression",
                );
            }
            println!("PACKED_STORE_PARITY width={width} height={height} cycles=100 replay={replay} exact_buffers={MUTABLE_BUFFERS} death_refresh=true private_mirror_exact=true no_export=true");
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_packed_store_suppression() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for warmup in WARMUP_CYCLES {
        let (width, height) = FIELDS[0];
        let mut kernel = prepare_kernel(width, height, false);
        let mut arms = Arms::new(&kernel);
        advance(&mut kernel, 0, warmup);
        let warm = checkpoint(&kernel);
        advance(&mut kernel, warmup, TIMED_CYCLES);
        let expected = capture_state(&kernel)?;
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, true);
        advance(&mut kernel, warmup, TIMED_CYCLES);
        let preflight = capture_state(&kernel)?;
        assert_state_equal(&kernel, &expected, &preflight);
        assert_mirror(&kernel, &preflight)?;
        let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for pair in 0..TIMING_PAIRS {
            for arm in [pair % 2, 1 - pair % 2] {
                restore(&mut kernel, &warm);
                arms.activate(&mut kernel, arm != 0);
                let start = Instant::now();
                advance(&mut kernel, warmup, TIMED_CYCLES);
                timings[arm].push(start.elapsed().as_secs_f64());
                let actual = capture_state(&kernel)?;
                assert_state_equal(&kernel, &expected, &actual);
                assert_mirror(&kernel, &actual)?;
            }
        }
        for values in &mut timings {
            values.sort_by(f64::total_cmp);
        }
        let reference = timings[0][TIMING_PAIRS / 2];
        let candidate = timings[1][TIMING_PAIRS / 2];
        println!("PACKED_STORE_TIMING warmup_cycles={warmup} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_production_recorder=true cold_import_timed=true state_comparison_timed=false", reference/candidate);
    }
    Ok(())
}

fn credit_pipeline(kernel: &GpuKernel, suppress: bool) -> wgpu::ComputePipeline {
    let source = [
        cache(kernel).common_source(),
        packed_encoder::credit_source(suppress),
        CREDIT_ENTRY.to_owned(),
    ]
    .join("\n");
    pipeline(kernel, source, "encoder_credit_probe", &constants(kernel))
}

fn run_credit(kernel: &GpuKernel, pipeline: &wgpu::ComputePipeline) {
    let packed = cache(kernel);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    packed.record_import(kernel, &mut encoder);
    // The fixture's public scratch supplies identical adapted features to both
    // private prefixes. This setup copy is outside the full-cycle experiment.
    encoder.copy_buffer_to_buffer(
        &kernel.brain_scratch_buffer,
        0,
        packed.buffer(),
        0,
        kernel.brain_scratch_buffer.size(),
    );
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(
            0,
            &kernel.global_credit.as_ref().unwrap().bind_groups[kernel.active_config_index],
            &[],
        );
        pass.dispatch_workgroups(packed.groups_per_agent(), kernel.agent_count, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

/// Include completely unchanged active vectors alongside signed-zero changes,
/// subnormal operands, mixed thresholds, and positive/negative saturation.
fn tiny_credit_fixture(kernel: &GpuKernel) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let subnormal = f32::from_bits(1);
    let below_normal = f32::from_bits(f32::MIN_POSITIVE.to_bits() - 1);
    let weights = [
        [1.0, -1.0, QUARTER_WEIGHT, -QUARTER_WEIGHT],
        [-0.0, 0.0, subnormal, -subnormal],
        [ENCODER_LIMIT, -ENCODER_LIMIT, NEAR_LIMIT, -NEAR_LIMIT],
        [
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            below_normal,
            -below_normal,
        ],
    ];
    let feature_values = [
        0.0,
        -0.0,
        subnormal,
        -subnormal,
        SMALL_FEATURE,
        LARGE_FEATURE,
    ];
    let credits = [
        CREDIT_THRESHOLD,
        -CREDIT_THRESHOLD,
        LARGE_CREDIT,
        -LARGE_CREDIT,
    ];
    let mut decisions = vec![0.0f32; agents * DECISION_STRIDE];
    let mut features = vec![0.0f32; agents * kernel.layout.brain_scratch_stride];
    for agent in 0..agents {
        let mut state = kernel.read_agent_state(u32::try_from(agent).unwrap());
        for feature in 0..kernel.layout.feature_count {
            features[agent * kernel.layout.brain_scratch_stride + feature] =
                feature_values[feature % feature_values.len()];
            for dimension in 0..ENCODED_DIMENSION {
                state.brain_state[feature * ENCODED_DIMENSION + dimension] =
                    weights[(dimension / VECTOR_WORDS) % weights.len()][dimension % VECTOR_WORDS];
                decisions[agent * DECISION_STRIDE + DECISION_CREDIT + dimension] =
                    credits[dimension % VECTOR_WORDS];
            }
        }
        kernel.write_agent_state(u32::try_from(agent).unwrap(), &state);
    }
    kernel
        .queue
        .write_buffer(&kernel.decision_buffer, 0, bytemuck::cast_slice(&decisions));
    kernel.queue.write_buffer(
        &kernel.brain_scratch_buffer,
        0,
        bytemuck::cast_slice(&features),
    );
}

fn credit_vector_counts(kernel: &GpuKernel, before: &State, after: &State) -> (usize, usize) {
    let before: &[u32] = bytemuck::cast_slice(&before[BRAIN_BUFFER]);
    let after: &[u32] = bytemuck::cast_slice(&after[BRAIN_BUFFER]);
    let mut unchanged = 0;
    let mut changed = 0;
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        if agent == usize::try_from(INACTIVE_AGENT).unwrap() {
            continue;
        }
        let base = agent * kernel.layout.brain_stride;
        for offset in (0..kernel.layout.feature_count * ENCODED_DIMENSION).step_by(VECTOR_WORDS) {
            if before[base + offset..base + offset + VECTOR_WORDS]
                == after[base + offset..base + offset + VECTOR_WORDS]
            {
                unchanged += 1;
            } else {
                changed += 1;
            }
        }
    }
    (unchanged, changed)
}

#[test]
#[ignore = "requires a GPU; original threshold/clamp fixture plus tiny updates"]
fn packed_store_suppression_preserves_credit_edges() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel(width, height, false);
        let mut arms = Arms::new(&kernel);
        let reference = credit_pipeline(&kernel, false);
        let candidate = credit_pipeline(&kernel, true);
        for tiny in [false, true] {
            arms.activate(&mut kernel, false);
            credit_fixture(&kernel, cache(&kernel).buffer());
            kernel.invalidate_packed_encoder();
            if tiny {
                tiny_credit_fixture(&kernel);
            }
            let initial = capture_state(&kernel)?;
            let saved = checkpoint(&kernel);
            run_credit(&kernel, &reference);
            let expected = capture_state(&kernel)?;
            assert_mirror(&kernel, &expected)?;
            if !tiny {
                assert_credit_cases(&kernel, &initial, &expected);
            }
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            run_credit(&kernel, &candidate);
            let actual = capture_state(&kernel)?;
            assert_state_equal(&kernel, &expected, &actual);
            assert_mirror(&kernel, &actual)?;
            let (unchanged, changed) = credit_vector_counts(&kernel, &initial, &actual);
            if tiny {
                assert!(unchanged > 0 && changed > 0);
            }
            println!("PACKED_STORE_CREDIT_EDGES width={width} height={height} tiny={tiny} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true unchanged_active_vectors={unchanged} changed_active_vectors={changed} signed_zero=true subnormal_operands={tiny} mixed_threshold=true clamp_edges=true inactive_unchanged=true");
        }
    }
    Ok(())
}
