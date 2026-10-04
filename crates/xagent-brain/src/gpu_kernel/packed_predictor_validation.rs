//! Isolated test-only row-major vec4 predictor storage. Four logical lanes of
//! the production width-sixteen sum share one invocation and one whole-vector
//! weight store. Its exact reduction tree is retained while 64 or 32 outputs
//! share each tile, reducing predictor barriers from 48 to four or eight.
//!
//! Binding13 contains the original scalar scratch prefix followed by private
//! predictor vectors. Encoder arithmetic/storage and global credit remain
//! scalar production code. Matrix import/export occurs only at test boundaries
//! outside timings; all other brain fields retain their original layout. All
//! arms execute through the same production dispatch recorder. Alternative
//! modes and ordinary host reset/write/export APIs are not integrated here.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Adjacent row-major inputs are exclusively owned as one vec4.
const VECTOR_WIDTH: usize = 4;
/// Match the current optimized predictor's FP32 addition tree exactly.
const LOGICAL_LANES: u32 = 16;
/// Two vector prefetches keep eight scalar weights live per invocation.
const VECTOR_PREFETCH: u32 = 2;
/// Scalar encoder/predictor reference uses the current production prefetch.
const SCALAR_PREFETCH: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Both source shapes retain the same square predictor but differ in offsets.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Full-state capture surrounds refresh, forced death and longer continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP_CYCLES: u32 = 256;
const TIMED_CYCLES: u32 = 100;
const TIMING_ROUNDS: usize = 5;
/// Compare the full workgroup with half as many active predictor invocations.
const ACTIVE_THREADS: [u32; 2] = [256, 128];
const ARM_COUNT: usize = 1 + ACTIVE_THREADS.len();
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

struct PackedShape {
    scratch_words: usize,
    prefix_bytes: u64,
    matrix_bytes: u64,
    original_offset_bytes: u64,
    total_bytes: u64,
}

impl PackedShape {
    fn new(kernel: &GpuKernel) -> Self {
        let agents = usize::try_from(kernel.agent_count).unwrap();
        assert!(agents > 0);
        let groups = LOGICAL_LANES / u32::try_from(VECTOR_WIDTH).unwrap();
        let tile = BRAIN_WORKGROUP_THREADS / groups;
        assert!(PREDICTOR_DIMENSION.is_multiple_of(usize::try_from(tile).unwrap()));
        assert!(ENCODED_DIMENSION
            .is_multiple_of(usize::try_from(LOGICAL_LANES * VECTOR_PREFETCH).unwrap()));
        let scalar_words = kernel
            .layout
            .brain_scratch_stride
            .checked_mul(agents)
            .unwrap();
        assert_eq!(bytes(scalar_words), kernel.brain_scratch_buffer.size());
        let scratch_words =
            scalar_words.checked_add(VECTOR_WIDTH - 1).unwrap() / VECTOR_WIDTH * VECTOR_WIDTH;
        let matrix_words = PREDICTOR_DIMENSION.checked_mul(ENCODED_DIMENSION).unwrap();
        let prefix_bytes = bytes(scratch_words);
        let matrix_bytes = bytes(matrix_words);
        let original_offset_words = kernel
            .layout
            .feature_count
            .checked_mul(ENCODED_DIMENSION)
            .unwrap()
            .checked_add(ENCODED_DIMENSION)
            .unwrap();
        assert!(
            original_offset_words.checked_add(matrix_words).unwrap() <= kernel.layout.brain_stride
        );
        let original_offset_bytes = bytes(original_offset_words);
        let total_bytes = prefix_bytes
            .checked_add(
                matrix_bytes
                    .checked_mul(u64::from(kernel.agent_count))
                    .unwrap(),
            )
            .unwrap();
        assert!(total_bytes <= u64::from(kernel.device.limits().max_storage_buffer_binding_size));
        assert!(total_bytes <= kernel.device.limits().max_buffer_size);
        u32::try_from(scratch_words).unwrap();
        u32::try_from(matrix_words.checked_mul(agents).unwrap() / VECTOR_WIDTH).unwrap();
        Self {
            scratch_words,
            prefix_bytes,
            matrix_bytes,
            original_offset_bytes,
            total_bytes,
        }
    }
}

/// Literal-sized scalar prefix followed by one row-major runtime vector array.
pub(super) fn packed_common(kernel: &GpuKernel) -> String {
    packed_common_with_threads(kernel, BRAIN_WORKGROUP_THREADS)
}

/// All workgroup invocations reach barriers; this sizes arithmetic and scratch.
pub(super) fn packed_common_with_threads(kernel: &GpuKernel, active_threads: u32) -> String {
    assert!(ACTIVE_THREADS.contains(&active_threads));
    let groups = LOGICAL_LANES / u32::try_from(VECTOR_WIDTH).unwrap();
    assert!(active_threads.is_multiple_of(groups));
    assert!(PREDICTOR_DIMENSION.is_multiple_of(usize::try_from(active_threads / groups).unwrap()));
    let shape = PackedShape::new(kernel);
    let original = include_str!("../shaders/kernel/common.wgsl");
    assert!(
        original.contains("override O_PREDICTOR_WEIGHTS: u32 = O_ENC_BIASES + ENCODED_DIMENSION;")
    );
    let declaration = format!("struct PackedPredictor {{\n    scratch: array<f32, {}>,\n    weights: array<vec4<f32>>,\n}}\n@group(0) @binding(13) var<storage, read_write> packed_predictor: PackedPredictor;", shape.scratch_words);
    let common = replace_once(
        original,
        "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
        &declaration,
    );
    format!("{common}\nconst PACKED_PREDICTOR_VECTOR_WIDTH: u32 = {VECTOR_WIDTH}u;\nconst PACKED_PREDICTOR_LANES: u32 = {LOGICAL_LANES}u;\nconst PACKED_PREDICTOR_GROUPS: u32 = PACKED_PREDICTOR_LANES / PACKED_PREDICTOR_VECTOR_WIDTH;\nconst PACKED_PREDICTOR_ACTIVE_THREADS: u32 = {active_threads}u;\nconst PACKED_PREDICTOR_OUTPUT_TILE: u32 = PACKED_PREDICTOR_ACTIVE_THREADS / PACKED_PREDICTOR_GROUPS;\nconst PACKED_PREDICTOR_INPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_PREDICTOR_VECTOR_WIDTH;\nconst PACKED_PREDICTOR_PREFETCH: u32 = {VECTOR_PREFETCH}u;\nconst PACKED_PREDICTOR_PARTIAL_WORDS: u32 = PACKED_PREDICTOR_OUTPUT_TILE * PACKED_PREDICTOR_LANES;\n")
}

/// Replace only inline predictor rows; all surrounding brain operations remain.
pub(super) fn packed_passes(baseline: &str) -> String {
    let function_start = baseline.find("fn coop_predict_and_act(").unwrap();
    let prefix_end = function_start
        + baseline[function_start..]
            .find("    // ── Recalled cosine similarities:")
            .unwrap();
    let prefix = &baseline[function_start..prefix_end];
    let begin = function_start + prefix.find("    } else {\n").unwrap() + "    } else {\n".len();
    let end = function_start + prefix.rfind("\n    }\n").unwrap();
    let block = &baseline[begin..end];
    assert!(block.contains("let output_in_tile = tid / 16u;"));
    assert!(block.contains("let lane = tid % 16u;"));
    assert_eq!(
        block.matches("O_PREDICTOR_WEIGHTS").count(),
        usize::try_from(SCALAR_PREFETCH * 2).unwrap()
    );
    assert!(block.contains("for (var stride = 16u / 2u; stride > 0u; stride /= 2u)"));
    let passes = replace_once(
        baseline,
        block,
        "        packed_predictor_rows(agent_id, tid);\n",
    );
    let passes = replace_once(
        &passes,
        "var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;",
        "var<workgroup> s_dense_partials: array<f32, PACKED_PREDICTOR_PARTIAL_WORDS>;",
    )
    .replace("brain_scratch[", "packed_predictor.scratch[");
    format!("{passes}\n{}", include_str!("packed_predictor.wgsl"))
}

/// Private storage owns the complete predictor matrix and scalar scratch prefix.
pub(super) fn packed_buffer(kernel: &GpuKernel) -> wgpu::Buffer {
    kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("packed_predictor_private"),
        size: PackedShape::new(kernel).total_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Every binding except private scratch/matrix binding13 remains original.
pub(super) fn packed_bind_group(
    kernel: &GpuKernel,
    packed: &wgpu::Buffer,
    config_index: usize,
) -> wgpu::BindGroup {
    super::exact_tiled_validation::make_transient_bind_group(
        kernel,
        &kernel.kernel_pipeline.get_bind_group_layout(0),
        packed,
        config_index,
    )
}

/// Copy unchanged row-major bytes at test boundaries. Export leaves packed state intact.
pub(super) fn copy_predictor_weights(kernel: &GpuKernel, packed: &wgpu::Buffer, export: bool) {
    let shape = PackedShape::new(kernel);
    assert_eq!(packed.size(), shape.total_bytes);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    for agent in 0..u64::from(kernel.agent_count) {
        let original_offset = agent
            .checked_mul(bytes(kernel.layout.brain_stride))
            .unwrap()
            .checked_add(shape.original_offset_bytes)
            .unwrap();
        let packed_offset = shape
            .prefix_bytes
            .checked_add(agent.checked_mul(shape.matrix_bytes).unwrap())
            .unwrap();
        if export {
            encoder.copy_buffer_to_buffer(
                packed,
                packed_offset,
                &kernel.brain_state_buffer,
                original_offset,
                shape.matrix_bytes,
            );
        } else {
            encoder.copy_buffer_to_buffer(
                &kernel.brain_state_buffer,
                original_offset,
                packed,
                packed_offset,
                shape.matrix_bytes,
            );
        }
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

fn optimized_brain() -> String {
    let fused = fuse_inline_predictor(&compose_brain_passes(true));
    let prefetched = dense_prefetch::prefetch_passes(&fused, SCALAR_PREFETCH);
    predictor_width::wider_predictor(&prefetched, LOGICAL_LANES)
}

fn publish_features(passes: &str) -> String {
    let begin = "    if (run_encoder_credit) {\n";
    let end = "    // ── Compute memory-key norm ONCE (memory reinforcement tiling)";
    assert_eq!(passes.matches(begin).count(), 1);
    assert_eq!(passes.matches(end).count(), 1);
    let start = passes.find(begin).unwrap();
    let last = passes.find(end).unwrap();
    let block = &passes[start..last];
    assert_eq!(block.matches("O_ENC_WEIGHTS").count(), 2);
    assert!(!block.contains("Barrier"));
    replace_once(
        passes,
        block,
        r"    if (run_encoder_credit) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            packed_predictor.scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];
        }
    }

",
    )
}

fn main_source(kernel: &GpuKernel, baseline: &str, active_threads: u32) -> String {
    let passes = publish_features(&packed_passes(baseline));
    // The exact identifier excludes the distinct homeostatic predictor weights.
    assert!(!passes.contains("brain_base + O_PREDICTOR_WEIGHTS"));
    apply_subgroup_markers(
        &[
            packed_common_with_threads(kernel, active_threads),
            passes,
            include_str!("../shaders/kernel/brain_inner.wgsl").to_owned(),
            include_str!("../shaders/kernel/phase_food_claim.wgsl").to_owned(),
            include_str!("../shaders/kernel/kernel_tick.wgsl").to_owned(),
        ]
        .join("\n"),
        kernel.has_subgroup,
    )
}

fn main_pipeline(
    kernel: &GpuKernel,
    source: String,
    constants: &HashMap<String, f64>,
    active_threads: u32,
) -> wgpu::ComputePipeline {
    eprintln!("PACKED_PREDICTOR_COMPILE_BEGIN entry=kernel_tick vector_width={VECTOR_WIDTH} logical_lanes={LOGICAL_LANES} active_threads={active_threads}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packed_predictor_main"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed_predictor_main"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("packed_predictor_main"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kernel_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        });
    eprintln!("PACKED_PREDICTOR_COMPILE_END entry=kernel_tick active_threads={active_threads}");
    pipeline
}

struct Arms {
    parked: [Option<Pipelines>; ARM_COUNT],
    active: usize,
    packed: wgpu::Buffer,
}

impl Arms {
    fn prepare(width: u32, height: u32, boundary: bool) -> (GpuKernel, Self) {
        let mut kernel = super::cached_combined_validation::prepare(width, height, boundary);
        let baseline = optimized_brain();
        let mut constants = vision_override_constants(&kernel.layout);
        constants.insert("VISION_AGENT_MASKS".into(), 1.0);
        kernel.global_credit = Some(Pipelines::new(&kernel, &baseline, &constants).unwrap());
        let packed = packed_buffer(&kernel);
        let parked = std::array::from_fn(|arm| {
            if arm == 0 {
                return None;
            }
            let active_threads = ACTIVE_THREADS[arm - 1];
            let mut candidate = Pipelines::new(&kernel, &baseline, &constants).unwrap();
            candidate.main = main_pipeline(
                &kernel,
                main_source(&kernel, &baseline, active_threads),
                &constants,
                active_threads,
            );
            candidate.bind_groups =
                std::array::from_fn(|index| packed_bind_group(&kernel, &packed, index));
            Some(candidate)
        });
        // The unchanged scalar global-credit module reads the identical scalar
        // prefix at binding13. It never accesses the packed predictor tail.
        (
            kernel,
            Self {
                parked,
                active: 0,
                packed,
            },
        )
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if arm != self.active {
            let next = self.parked[arm].take().unwrap();
            let previous = kernel.global_credit.replace(next).unwrap();
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
        assert!(kernel.global_credit_active());
    }
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    assert!(kernel.global_credit_active());
    kernel.dispatch_ticks(
        u64::from(cycle.checked_mul(kernel.brain_tick_stride).unwrap()),
        cycles.checked_mul(kernel.brain_tick_stride).unwrap(),
    );
    kernel.poll_wait();
}

fn trajectory(kernel: &mut GpuKernel, arms: &Arms) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, cycles);
        if arms.active != 0 {
            copy_predictor_weights(kernel, &arms.packed, true);
        }
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn packed_predictor_preserves_decoded_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let initial_state = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let reference = trajectory(&mut kernel, &arms)?;
        for (index, active_threads) in ACTIVE_THREADS.into_iter().enumerate() {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, index + 1);
            copy_predictor_weights(&kernel, &arms.packed, false);
            let candidate = trajectory(&mut kernel, &arms)?;
            for (reference, candidate) in reference.iter().zip(&candidate) {
                assert_state_equal(&kernel, reference, candidate);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    candidate,
                    INACTIVE_AGENT,
                    "packed predictor",
                );
            }
            restore(&mut kernel, &saved);
            copy_predictor_weights(&kernel, &arms.packed, false);
            let repeated = trajectory(&mut kernel, &arms)?;
            for (candidate, repeated) in candidate.iter().zip(&repeated) {
                assert_state_equal(&kernel, candidate, repeated);
            }
            println!("PACKED_PREDICTOR_PARITY width={width} height={height} cycles=100 decoded_exact_buffers={MUTABLE_BUFFERS} repeat_exact_buffers={MUTABLE_BUFFERS} death_refresh=true inactive_unchanged=true logical_lanes={LOGICAL_LANES} active_threads={active_threads} encoder_unchanged=true imports_exports=test_boundaries");
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_packed_predictor_against_current_global_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let (mut kernel, mut arms) = Arms::prepare(width, height, false);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    for arm in 1..ARM_COUNT {
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, arm);
        copy_predictor_weights(&kernel, &arms.packed, false);
        advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
        copy_predictor_weights(&kernel, &arms.packed, true);
        assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    }
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for position in 0..ARM_COUNT {
            let arm = (round + position) % ARM_COUNT;
            restore(&mut kernel, &warm);
            arms.activate(&mut kernel, arm);
            if arm != 0 {
                copy_predictor_weights(&kernel, &arms.packed, false);
            }
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            if arm != 0 {
                copy_predictor_weights(&kernel, &arms.packed, true);
            }
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        }
    }
    for times in &mut timings {
        times.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_ROUNDS / 2];
    for (index, active_threads) in ACTIVE_THREADS.into_iter().enumerate() {
        let candidate = timings[index + 1][TIMING_ROUNDS / 2];
        let partial_words = active_threads * u32::try_from(VECTOR_WIDTH).unwrap();
        println!("PACKED_PREDICTOR_TIMING cycles={TIMED_CYCLES} rounds={TIMING_ROUNDS} rotated_arms={ARM_COUNT} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} decoded_exact_buffers={MUTABLE_BUFFERS} logical_lanes={LOGICAL_LANES} active_threads={active_threads} predictor_partial_words={partial_words} packed_storage_bytes={} imports_exports_timed=false encoder_unchanged=true same_production_recorder=true full_simulation=true", reference/candidate, PackedShape::new(&kernel).total_bytes);
    }
    Ok(())
}
