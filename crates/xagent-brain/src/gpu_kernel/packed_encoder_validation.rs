//! Test-only typed vec4 encoder storage. Binding 13 contains a checked fixed
//! scalar scratch prefix followed by a runtime vec4 weight array. Adjacent
//! outputs keep their feature-major byte order; no overlapping scalar binding
//! aliases the packed matrix. Credit invocations own complete vectors.
//!
//! Import/export copies run only at test boundaries and outside timings. The
//! remaining brain state retains its production layout. Death preserves the
//! learned matrix; explicit reimport follows checkpoint restoration. Public
//! reset/write/export APIs and alternative schedules are not integrated here.
//! All timed arms use the existing production dispatch recorder and global
//! credit route, so this experiment changes shader/storage shape only.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Four consecutive feature-major outputs share one typed vector load/store.
const VECTOR_WIDTH: usize = 4;
/// Two vector prefetches keep eight weight scalars live per invocation.
const VECTOR_PREFETCH: u32 = 2;
/// Preserve the production encoder's four independent feature accumulations.
const ENCODER_INNER_LANES: u32 = 4;
/// The current optimized reference retains eight scalar prefetches and width16.
const SCALAR_PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// One world group precedes independent encoder-credit vector groups.
const WORLD_GROUPS: u32 = 1;
/// All production kernel layouts reserve two push-constant words.
const PUSH_CONSTANT_BYTES: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();
/// Default and odd input counts exercise different final credit-group tails.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Cross whitening refresh, forced death, and multi-submission boundaries.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP_CYCLES: u32 = 256;
const TIMED_CYCLES: u32 = 100;
/// Rotate all four arms through five rounds for median full-cycle timing.
const TIMING_ROUNDS: usize = 5;
const ARM_NAMES: [&str; 4] = ["scalar", "packed", "mirrored", "mirrored_context"];
const ARM_COUNT: usize = ARM_NAMES.len();
/// Only the unmirrored arm needs a decoded export before public comparisons.
const UNMIRRORED_ARM: usize = 1;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
/// Match the production credit gate; adjacent bit patterns straddle it.
const CREDIT_THRESHOLD: f32 = 1e-6;
/// Eight outputs repeat mixed inactive, threshold, and saturating components.
const CREDIT_CASE_WIDTH: usize = 8;
const LARGE_CREDIT: f32 = 2.0;
/// Large finite features make threshold-active updates observable in FP32.
const CREDIT_FEATURE: f32 = 10_000.0;
/// Inactive values deliberately include signed zero and values outside clamp.
const INACTIVE_WEIGHT: f32 = 3.5;
const SMALL_WEIGHT: f32 = 0.25;
const NEAR_CLAMP_WEIGHT: f32 = 1.9;
const ENCODER_LIMIT: f32 = 2.0;
/// Full-state captures place the brain in the ninth mutable buffer.
const BRAIN_BUFFER_INDEX: usize = 8;

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

struct PackedShape {
    scratch_words: usize,
    prefix_bytes: u64,
    matrix_bytes: u64,
    total_bytes: u64,
    vectors_per_agent: u32,
    groups_per_agent: u32,
}

impl PackedShape {
    fn new(kernel: &GpuKernel) -> Self {
        let agents = usize::try_from(kernel.agent_count).unwrap();
        assert!(agents > 0 && ENCODED_DIMENSION.is_multiple_of(VECTOR_WIDTH));
        let scalar_words = kernel
            .layout
            .brain_scratch_stride
            .checked_mul(agents)
            .unwrap();
        assert_eq!(bytes(scalar_words), kernel.brain_scratch_buffer.size());
        let scratch_words =
            scalar_words.checked_add(VECTOR_WIDTH - 1).unwrap() / VECTOR_WIDTH * VECTOR_WIDTH;
        let matrix_words = kernel
            .layout
            .feature_count
            .checked_mul(ENCODED_DIMENSION)
            .unwrap();
        let prefix_bytes = bytes(scratch_words);
        let matrix_bytes = bytes(matrix_words);
        let total_bytes = prefix_bytes
            .checked_add(
                matrix_bytes
                    .checked_mul(u64::from(kernel.agent_count))
                    .unwrap(),
            )
            .unwrap();
        let vectors_per_agent = u32::try_from(matrix_words / VECTOR_WIDTH).unwrap();
        let groups_per_agent = vectors_per_agent.div_ceil(BRAIN_WORKGROUP_THREADS);
        assert!(
            groups_per_agent
                .checked_mul(kernel.agent_count)
                .unwrap()
                .checked_add(WORLD_GROUPS)
                .unwrap()
                <= MAX_DISPATCH_WORKGROUPS
        );
        assert!(total_bytes <= u64::from(kernel.device.limits().max_storage_buffer_binding_size));
        assert!(total_bytes <= kernel.device.limits().max_buffer_size);
        u32::try_from(scratch_words).unwrap();
        vectors_per_agent.checked_mul(kernel.agent_count).unwrap();
        Self {
            scratch_words,
            prefix_bytes,
            matrix_bytes,
            total_bytes,
            vectors_per_agent,
            groups_per_agent,
        }
    }
}

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

/// Checked literal storage-array length; override-sized storage fields are not legal WGSL.
pub(super) fn packed_common(kernel: &GpuKernel) -> String {
    let shape = PackedShape::new(kernel);
    let common = include_str!("../shaders/kernel/common.wgsl");
    assert!(common.contains("const O_ENC_WEIGHTS: u32 = 0u;"));
    let declaration = format!(
        "struct PackedEncoder {{\n    scratch: array<f32, {}>,\n    weights: array<vec4<f32>>,\n}}\n@group(0) @binding(13) var<storage, read_write> packed_encoder: PackedEncoder;",
        shape.scratch_words,
    );
    let changed = replace_once(
        common,
        "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
        &declaration,
    );
    format!("{changed}\nconst PACKED_ENCODER_WIDTH: u32 = {VECTOR_WIDTH}u;\nconst PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;\nconst PACKED_ENCODER_PREFETCH: u32 = {VECTOR_PREFETCH}u;\nconst PACKED_ENCODER_INNER_LANES: u32 = {ENCODER_INNER_LANES}u;\nconst PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;\n")
}

/// Replace only the encoder arithmetic and enlarge its existing scalar scratch.
pub(super) fn packed_passes(baseline: &str) -> String {
    let start = baseline.find("fn coop_encode(").unwrap();
    let end = start + baseline[start..].find("\n}").unwrap() + "\n}".len();
    let original = &baseline[start..end];
    assert_eq!(
        original.matches("O_ENC_WEIGHTS").count(),
        usize::try_from(SCALAR_PREFETCH).unwrap()
    );
    assert!(baseline.contains("const DENSE_INNER_LANES: u32 = 4u;"));
    super::packed_encoder::packed_passes(baseline)
}

/// Independent test storage, with no alias to the ordinary brain/scratch bindings.
pub(super) fn packed_buffer(kernel: &GpuKernel) -> wgpu::Buffer {
    kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("packed_encoder_private"),
        size: PackedShape::new(kernel).total_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Bind the packed allocation only at binding13; every other buffer is original.
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

/// Preserve feature-major bytes while moving complete matrices at test boundaries.
/// Export does not modify packed state or alter any subsequent candidate query.
pub(super) fn copy_encoder_weights(kernel: &GpuKernel, packed: &wgpu::Buffer, export: bool) {
    let shape = PackedShape::new(kernel);
    assert_eq!(packed.size(), shape.total_bytes);
    let brain_stride = bytes(kernel.layout.brain_stride);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    for agent in 0..u64::from(kernel.agent_count) {
        let brain_offset = agent.checked_mul(brain_stride).unwrap();
        let packed_offset = shape
            .prefix_bytes
            .checked_add(agent.checked_mul(shape.matrix_bytes).unwrap())
            .unwrap();
        if export {
            encoder.copy_buffer_to_buffer(
                packed,
                packed_offset,
                &kernel.brain_state_buffer,
                brain_offset,
                shape.matrix_bytes,
            );
        } else {
            encoder.copy_buffer_to_buffer(
                &kernel.brain_state_buffer,
                brain_offset,
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
    predictor_width::wider_predictor(&prefetched, PREDICTOR_LANES)
}

fn publish_features(passes: &str) -> String {
    let start_marker = "    if (run_encoder_credit) {\n";
    let end_marker = "    // ── Compute memory-key norm ONCE (memory reinforcement tiling)";
    assert_eq!(passes.matches(start_marker).count(), 1);
    assert_eq!(passes.matches(end_marker).count(), 1);
    let start = passes.find(start_marker).unwrap();
    let end = passes.find(end_marker).unwrap();
    let block = &passes[start..end];
    assert_eq!(block.matches("O_ENC_WEIGHTS").count(), 2);
    assert!(!block.contains("Barrier"));
    assert!(!passes[end..].contains("O_ENC_WEIGHTS"));
    replace_once(
        passes,
        block,
        r"    if (run_encoder_credit) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            packed_encoder.scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];
        }
    }

",
    )
}

fn main_source(kernel: &GpuKernel, baseline: &str) -> String {
    let passes = publish_features(&packed_passes(baseline));
    assert!(!passes.contains("O_ENC_WEIGHTS"));
    apply_subgroup_markers(
        &[
            packed_common(kernel),
            passes,
            include_str!("../shaders/kernel/brain_inner.wgsl").to_owned(),
            include_str!("../shaders/kernel/phase_food_claim.wgsl").to_owned(),
            include_str!("../shaders/kernel/kernel_tick.wgsl").to_owned(),
        ]
        .join("\n"),
        kernel.has_subgroup,
    )
}

fn global_source(kernel: &GpuKernel, mirror: bool) -> String {
    let common = packed_common(kernel);
    let credit = if mirror {
        super::packed_encoder::CREDIT_SOURCE.to_owned()
    } else {
        let source = super::packed_encoder::CREDIT_SOURCE;
        let marker = "    // The scalar matrix remains authoritative for readback";
        assert_eq!(source.matches(marker).count(), 1);
        let start = source.find(marker).unwrap();
        let end = source.rfind("\n}").unwrap();
        replace_once(source, &source[start..end], "")
    };
    let global = replace_once(include_str!("../shaders/kernel/global_tick.wgsl"),
        "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;",
        "fn global_world_inner(tid: u32) {",
    );
    [
        common.as_str(),
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
        global.as_str(),
        &credit,
        include_str!("../shaders/kernel/global_credit_tick.wgsl"),
    ]
    .join("\n")
}

fn pipeline(
    kernel: &GpuKernel,
    source: String,
    entry: &str,
    constants: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let packed = source.contains("struct PackedEncoder");
    eprintln!("PACKED_ENCODER_COMPILE_BEGIN entry={entry} packed={packed}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        });
    eprintln!("PACKED_ENCODER_COMPILE_END entry={entry} packed={packed}");
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
                None
            } else {
                Some(packed_pipelines(
                    &kernel, &packed, &baseline, &constants, arm,
                ))
            }
        });
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
        if self.active != arm {
            let next = self.parked[arm].take().unwrap();
            let previous = kernel.global_credit.replace(next).unwrap();
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
        assert!(kernel.global_credit_active());
    }
}

fn packed_pipelines(
    kernel: &GpuKernel,
    packed: &wgpu::Buffer,
    baseline: &str,
    constants: &HashMap<String, f64>,
    arm: usize,
) -> Pipelines {
    let shape = PackedShape::new(kernel);
    let passes = if arm == ARM_COUNT - 1 {
        super::context_gather::gather_context(baseline, 8)
    } else {
        baseline.to_owned()
    };
    // Keep the production dispatch recorder unchanged. Its scalar scratch
    // allocation is unused by these test candidates.
    let mut candidate = Pipelines::new(kernel, baseline, constants).unwrap();
    candidate.main = pipeline(
        kernel,
        main_source(kernel, &passes),
        "kernel_tick",
        constants,
    );
    let mut constants = constants.clone();
    constants.insert(
        "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
        f64::from(shape.groups_per_agent),
    );
    let global = global_source(kernel, arm != UNMIRRORED_ARM);
    candidate.global = pipeline(kernel, global, "global_credit_tick", &constants);
    candidate.global_workgroups = shape
        .groups_per_agent
        .checked_mul(kernel.agent_count)
        .unwrap()
        .checked_add(WORLD_GROUPS)
        .unwrap();
    candidate.bind_groups = std::array::from_fn(|index| packed_bind_group(kernel, packed, index));
    candidate
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
        if arms.active == UNMIRRORED_ARM {
            copy_encoder_weights(kernel, &arms.packed, true);
        }
        states.push(capture_state(kernel)?);
        cycle += cycles;
    }
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn packed_encoder_preserves_decoded_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        let reference = trajectory(&mut kernel, &arms)?;
        for (arm, name) in ARM_NAMES.iter().enumerate().skip(1) {
            restore(&mut kernel, &initial);
            arms.activate(&mut kernel, arm);
            copy_encoder_weights(&kernel, &arms.packed, false);
            let candidate = trajectory(&mut kernel, &arms)?;
            for (reference, candidate) in reference.iter().zip(&candidate) {
                assert_state_equal(&kernel, reference, candidate);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    candidate,
                    INACTIVE_AGENT,
                    "packed encoder",
                );
            }
            restore(&mut kernel, &initial);
            copy_encoder_weights(&kernel, &arms.packed, false);
            let repeated = trajectory(&mut kernel, &arms)?;
            for (candidate, repeated) in candidate.iter().zip(&repeated) {
                assert_state_equal(&kernel, candidate, repeated);
            }
            println!("PACKED_ENCODER_PARITY arm={name} width={width} height={height} cycles=100 decoded_exact_buffers={MUTABLE_BUFFERS} repeat_exact_buffers={MUTABLE_BUFFERS} death_refresh=true inactive_unchanged=true imports=test_boundaries exports={}", arm == UNMIRRORED_ARM);
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_packed_encoder_against_current_global_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let (mut kernel, mut arms) = Arms::prepare(width, height, false);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &warm);
    arms.activate(&mut kernel, 1);
    copy_encoder_weights(&kernel, &arms.packed, false);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    copy_encoder_weights(&kernel, &arms.packed, true);
    assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        for position in 0..ARM_COUNT {
            let arm = (round + position) % ARM_COUNT;
            restore(&mut kernel, &warm);
            arms.activate(&mut kernel, arm);
            if arm != 0 {
                copy_encoder_weights(&kernel, &arms.packed, false);
            }
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            if arm == UNMIRRORED_ARM {
                copy_encoder_weights(&kernel, &arms.packed, true);
            }
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        }
    }
    for times in &mut timings {
        times.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_ROUNDS / 2];
    let shape = PackedShape::new(&kernel);
    for (arm, samples) in timings.iter().enumerate().skip(1) {
        let candidate = samples[TIMING_ROUNDS / 2];
        println!("PACKED_ENCODER_TIMING arm={} cycles={TIMED_CYCLES} rounds={TIMING_ROUNDS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} decoded_exact_buffers={MUTABLE_BUFFERS} vector_credit_groups_per_agent={} packed_vectors_per_agent={} packed_storage_bytes={} imports_exports_timed=false same_production_recorder=true full_simulation=true", ARM_NAMES[arm], reference/candidate, shape.groups_per_agent, shape.vectors_per_agent, shape.total_bytes);
    }
    Ok(())
}

pub(super) fn credit_fixture(kernel: &GpuKernel, packed: &wgpu::Buffer) {
    assert!(
        include_str!("../shaders/kernel/common.wgsl").contains("const CREDIT_EPSILON: f32 = 1e-6;")
    );
    let below = f32::from_bits(CREDIT_THRESHOLD.to_bits() - 1);
    let above = f32::from_bits(CREDIT_THRESHOLD.to_bits() + 1);
    let credit_cases = [
        below,
        CREDIT_THRESHOLD,
        above,
        LARGE_CREDIT,
        -below,
        -CREDIT_THRESHOLD,
        -above,
        -LARGE_CREDIT,
    ];
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let mut decisions = vec![0.0; agents.checked_mul(DECISION_STRIDE).unwrap()];
    let mut weights = vec![
        0.0;
        kernel
            .layout
            .feature_count
            .checked_mul(ENCODED_DIMENSION)
            .unwrap()
    ];
    for (index, weight) in weights.iter_mut().enumerate() {
        let feature = index / ENCODED_DIMENSION;
        *weight = match index % CREDIT_CASE_WIDTH {
            0 => {
                if feature % 2 == 0 {
                    -0.0
                } else {
                    INACTIVE_WEIGHT
                }
            }
            4 => {
                if feature % 2 == 0 {
                    0.0
                } else {
                    -INACTIVE_WEIGHT
                }
            }
            3 => NEAR_CLAMP_WEIGHT,
            7 => -NEAR_CLAMP_WEIGHT,
            _ => SMALL_WEIGHT,
        };
    }
    for agent in 0..agents {
        for dimension in 0..ENCODED_DIMENSION {
            decisions[agent * DECISION_STRIDE + DECISION_CREDIT + dimension] =
                credit_cases[dimension % CREDIT_CASE_WIDTH];
        }
        kernel.queue.write_buffer(
            &kernel.brain_state_buffer,
            bytes(agent.checked_mul(kernel.layout.brain_stride).unwrap()),
            bytemuck::cast_slice(&weights),
        );
        kernel.write_agent_physics_fields(
            u32::try_from(agent).unwrap(),
            &[(
                P_ALIVE,
                if agent == usize::try_from(INACTIVE_AGENT).unwrap() {
                    0.0
                } else {
                    1.0
                },
            )],
        );
    }
    kernel
        .queue
        .write_buffer(&kernel.decision_buffer, 0, bytemuck::cast_slice(&decisions));
    let features = vec![
        CREDIT_FEATURE;
        agents
            .checked_mul(kernel.layout.brain_scratch_stride)
            .unwrap()
    ];
    kernel.queue.write_buffer(
        &kernel.brain_scratch_buffer,
        0,
        bytemuck::cast_slice(&features),
    );
    kernel
        .queue
        .write_buffer(packed, 0, bytemuck::cast_slice(&features));
}

fn credit_probe_pipeline(kernel: &GpuKernel, packed: bool) -> wgpu::ComputePipeline {
    let common = if packed {
        packed_common(kernel)
    } else {
        include_str!("../shaders/kernel/common.wgsl").to_owned()
    };
    let credit = if packed {
        super::packed_encoder::CREDIT_SOURCE
    } else {
        include_str!("../shaders/kernel/phase_encoder_credit.wgsl")
    };
    let entry = r"
@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn encoder_credit_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    phase_encoder_credit(group.y, group.x * ENCODER_CREDIT_THREADS + tid);
}
";
    pipeline(
        kernel,
        [common.as_str(), credit, entry].join("\n"),
        "encoder_credit_probe",
        &vision_override_constants(&kernel.layout),
    )
}

fn run_credit(
    kernel: &GpuKernel,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    groups_per_agent: u32,
) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups_per_agent, kernel.agent_count, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

pub(super) fn assert_credit_cases(kernel: &GpuKernel, initial: &State, actual: &State) {
    let initial: &[u32] = bytemuck::cast_slice(&initial[BRAIN_BUFFER_INDEX]);
    let actual: &[u32] = bytemuck::cast_slice(&actual[BRAIN_BUFFER_INDEX]);
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        for feature in 0..kernel.layout.feature_count {
            for dimension in 0..ENCODED_DIMENSION {
                let index =
                    agent * kernel.layout.brain_stride + feature * ENCODED_DIMENSION + dimension;
                let case = dimension % CREDIT_CASE_WIDTH;
                if agent == usize::try_from(INACTIVE_AGENT).unwrap() || matches!(case, 0 | 4) {
                    assert_eq!(
                        actual[index], initial[index],
                        "inactive component must retain every bit"
                    );
                } else if case == 3 {
                    assert_eq!(
                        actual[index],
                        ENCODER_LIMIT.to_bits(),
                        "positive saturation"
                    );
                } else if case == 7 {
                    assert_eq!(
                        actual[index],
                        (-ENCODER_LIMIT).to_bits(),
                        "negative saturation"
                    );
                } else {
                    assert_ne!(
                        actual[index], initial[index],
                        "threshold-equal and above-threshold components must update"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; checks vector ownership and original per-component gates"]
fn packed_encoder_credit_preserves_threshold_clamp_and_inactive_components() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, arms) = Arms::prepare(width, height, false);
        credit_fixture(&kernel, &arms.packed);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        let reference = credit_probe_pipeline(&kernel, false);
        let candidate = credit_probe_pipeline(&kernel, true);
        let shape = PackedShape::new(&kernel);
        let scalar_groups = shape
            .vectors_per_agent
            .checked_mul(u32::try_from(VECTOR_WIDTH).unwrap())
            .unwrap()
            .div_ceil(BRAIN_WORKGROUP_THREADS);
        run_credit(
            &kernel,
            &reference,
            &kernel.bind_groups[kernel.active_config_index],
            scalar_groups,
        );
        let expected = capture_state(&kernel)?;
        assert_credit_cases(&kernel, &initial_state, &expected);
        restore(&mut kernel, &initial);
        copy_encoder_weights(&kernel, &arms.packed, false);
        let group = packed_bind_group(&kernel, &arms.packed, kernel.active_config_index);
        run_credit(&kernel, &candidate, &group, shape.groups_per_agent);
        copy_encoder_weights(&kernel, &arms.packed, true);
        let actual = capture_state(&kernel)?;
        assert_state_equal(&kernel, &expected, &actual);
        assert_credit_cases(&kernel, &initial_state, &actual);
        println!("PACKED_ENCODER_CREDIT_GATES width={width} height={height} decoded_exact_buffers={MUTABLE_BUFFERS} below_equal_above_threshold=true positive_negative_clamp=true signed_zero_and_outside_clamp_inactive_preserved=true dead_agent_unchanged=true tail_vectors=true");
    }
    Ok(())
}
