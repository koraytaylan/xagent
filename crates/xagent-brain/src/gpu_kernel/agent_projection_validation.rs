//! Test-only encoder projection computed beside actual packed credit updates.
//! One 256-thread global workgroup per agent updates all packed weights and
//! forms four original visual-prefix sums, seeded with the lane-zero bias.
//! Predicted inputs come from current raw vision and the stored adapted mean.
//! The next encoder reuses those partials only if all visual input bits match;
//! otherwise it executes the original packed encoder. There is no recurrent
//! projection approximation, delayed weight update, or change to public layout.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore, Checkpoint};
use super::packed_store_validation::{
    advance as production_advance, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::{
    assert_inactive_agent_unchanged, compare_rounding_state, extract_behavior,
    report_seeded_behavior,
};
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw vision sizes exercise both aligned and odd visual/nonvisual boundaries.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Existing whole-cycle measurements use ten independent brains.
const AGENTS: u32 = 10;
/// The shared vision fixture seeds both its brains and scene from this value.
const FIXTURE_SEED: u64 = 0x194f_c3a7;
/// Typed storage vectors own four adjacent outputs.
const VECTOR_WORDS: usize = 4;
/// World remains 256-thread; only 128 owners perform encoder arithmetic.
const CREDIT_THREADS: usize = 256;
/// Preserve the canonical four ascending feature recurrences.
const PARTIAL_LANES: usize = 4;
/// Each equality-gate invocation checks eight consecutive visual features.
const GATE_FEATURES: usize = 8;
/// Conservative per-declaration device shared-storage alignment.
const SHARED_ALIGNMENT: usize = 16;
/// Valid, last-used, hit-count, miss-count, and death generation are private.
const HEADER_WORDS: usize = 5;
const VALID_OFFSET: usize = 0;
const USED_OFFSET: usize = 1;
const HITS_OFFSET: usize = 2;
const MISSES_OFFSET: usize = 3;
const DEATH_OFFSET: usize = 4;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;
const BRAIN_BUFFER: usize = 8;
const MUTABLE_BUFFERS: usize = 13;
/// Includes both sides of periodic whitening and a second forced death.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Consecutive existing chunk boundaries guarantee one unchanged visual input
/// without relying on a moving terrain scene to produce the same whole frame.
const HELD_RAW_CYCLES: [u32; 2] = [REFRESH_CYCLES * 2, REFRESH_CYCLES * 2 + 1];
/// The canonical sky color and far-depth sentinel are legal raw vision inputs.
const HELD_SKY_RGBA: [f32; VECTOR_WORDS] = [0.53, 0.81, 0.92, 1.0];
const HELD_SKY_DEPTH: f32 = 1.0;
const PHYSICS_BUFFER: usize = 0;
const ALIVE_THRESHOLD: f32 = 0.5;
const INACTIVE_AGENT: u32 = 1;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn aligned_words(words: usize) -> usize {
    words
        .checked_add(VECTOR_WORDS - 1)
        .unwrap()
        .checked_div(VECTOR_WORDS)
        .unwrap()
        .checked_mul(VECTOR_WORDS)
        .unwrap()
}

/// Private appended storage follows the ordinary rounded scalar prefix.
pub(super) struct ProjectionLayout {
    pub(super) base_words: usize,
    pub(super) extra_words: usize,
    pub(super) prefix_words: usize,
    pub(super) agent_stride: usize,
    pub(super) partial_offset: usize,
    pub(super) raw_offset: usize,
    pub(super) valid_offset: usize,
    pub(super) used_offset: usize,
    pub(super) death_offset: usize,
    pub(super) visual_count: usize,
    pub(super) groups: usize,
    features_per_group: usize,
    gate_groups: usize,
    pub(super) shared_bytes: usize,
}

impl ProjectionLayout {
    pub(super) fn new(kernel: &GpuKernel) -> Self {
        assert!(!kernel.layout.visual_cortex_enabled);
        assert!(kernel.agent_count > 0);
        assert!(ENCODED_DIMENSION.is_multiple_of(VECTOR_WORDS));
        let output_vectors = ENCODED_DIMENSION / VECTOR_WORDS;
        assert!(CREDIT_THREADS.is_multiple_of(output_vectors));
        let features_per_group = GATE_FEATURES;
        let visual_count = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
        let gate_groups = visual_count.div_ceil(features_per_group);
        let groups = 1;
        // Encode borrows s_reinf_dot's 256 words for the equality gate.
        assert!(gate_groups < usize::try_from(main_width::MAIN_THREADS).unwrap());
        let shared_bytes = visual_count
            .checked_mul(WORD_BYTES)
            .unwrap()
            .checked_add(SHARED_ALIGNMENT - 1)
            .unwrap()
            / SHARED_ALIGNMENT
            * SHARED_ALIGNMENT;
        assert!(
            shared_bytes
                <= usize::try_from(kernel.device.limits().max_compute_workgroup_storage_size)
                    .unwrap()
        );
        let raw_offset = HEADER_WORDS;
        let partial_offset = raw_offset.checked_add(ENCODED_DIMENSION).unwrap();
        let partial_words = PARTIAL_LANES.checked_mul(ENCODED_DIMENSION).unwrap();
        let agent_stride = aligned_words(partial_offset.checked_add(partial_words).unwrap());
        let agents = usize::try_from(kernel.agent_count).unwrap();
        let base_words = aligned_words(
            kernel
                .layout
                .brain_scratch_stride
                .checked_mul(agents)
                .unwrap(),
        );
        let extra_words = agent_stride.checked_mul(agents).unwrap();
        let prefix_words = aligned_words(base_words.checked_add(extra_words).unwrap());
        u32::try_from(prefix_words).unwrap();
        Self {
            base_words,
            extra_words,
            prefix_words,
            agent_stride,
            partial_offset,
            raw_offset,
            valid_offset: VALID_OFFSET,
            used_offset: USED_OFFSET,
            death_offset: DEATH_OFFSET,
            visual_count,
            groups,
            features_per_group,
            gate_groups,
            shared_bytes,
        }
    }

    fn header(&self) -> String {
        format!(
            "\nconst PROJECTION_STORAGE_BASE: u32 = {}u;\nconst PROJECTION_AGENT_STRIDE: u32 = {}u;\nconst PROJECTION_VISUAL_COUNT: u32 = {}u;\nconst PROJECTION_GATE_GROUP_COUNT: u32 = {}u;\nconst PROJECTION_FEATURES_PER_GROUP: u32 = {}u;\nconst PROJECTION_VALID_OFFSET: u32 = {}u;\nconst PROJECTION_USED_OFFSET: u32 = {}u;\nconst PROJECTION_HITS_OFFSET: u32 = {HITS_OFFSET}u;\nconst PROJECTION_MISSES_OFFSET: u32 = {MISSES_OFFSET}u;\nconst PROJECTION_DEATH_OFFSET: u32 = {}u;\nconst PROJECTION_RAW_OFFSET: u32 = {}u;\nconst PROJECTION_PARTIAL_OFFSET: u32 = {}u;\n",
            self.base_words, self.agent_stride, self.visual_count, self.gate_groups,
            self.features_per_group, self.valid_offset, self.used_offset,
            self.death_offset, self.raw_offset, self.partial_offset,
        )
    }
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

/// Shared source for the timing shader and independent raw arithmetic probe.
pub(super) fn common_source(
    _kernel: &GpuKernel,
    packed: &packed_encoder::Cache,
    layout: &ProjectionLayout,
) -> String {
    format!("{}{}", packed.common_source(), layout.header())
}

/// Preserve original per-component arithmetic and mirrors; return the weight
/// actually used by storage even when its update is disabled or unchanged.
pub(super) fn credit_source() -> String {
    // Retain the established canonical update helper, dropping only its
    // eight-feature group driver and four-KiB temporary array.
    let source = super::fresh_projection_validation::credit_source();
    let driver = include_str!("fresh_projection_credit.wgsl");
    let source = replace_once(&source, driver, "");
    assert!(!source.contains("var<workgroup>"));
    format!("{source}\n{}", include_str!("agent_projection_credit.wgsl"))
}

/// Both routes publish their actual pre-tanh value in the timing source.
/// Removing no arithmetic for raw inspection avoids instrumentation drift.
pub(super) fn encode_source() -> String {
    let original = include_str!("../shaders/kernel/brain_packed_encoder.wgsl");
    let fresh = replace_once(
        original,
        "fn coop_encode(agent_id: u32, tid: u32) {",
        "fn fresh_projection_original_encode(agent_id: u32, tid: u32) {",
    );
    let fresh = replace_once(&fresh,
        "            s_encoded[base + component] = fast_tanh(reduced);",
        "            packed_encoder.scratch[PROJECTION_STORAGE_BASE + agent_id * PROJECTION_AGENT_STRIDE + PROJECTION_RAW_OFFSET + base + component] = reduced;\n            s_encoded[base + component] = fast_tanh(reduced);");
    format!("{fresh}\n{}", include_str!("agent_projection_encode.wgsl"))
}

pub(super) fn main_source(
    kernel: &GpuKernel,
    packed: &packed_encoder::Cache,
    layout: &ProjectionLayout,
) -> String {
    let source = global_credit::main_source(&optimized_brain(), kernel.has_subgroup, Some(packed));
    let source = main_width::try_transform(&source).unwrap();
    let original = include_str!("../shaders/kernel/brain_packed_encoder.wgsl");
    let source = replace_once(&source, original, &encode_source());
    format!("{source}{}", layout.header())
}

pub(super) fn global_source(
    _kernel: &GpuKernel,
    packed: &packed_encoder::Cache,
    layout: &ProjectionLayout,
) -> String {
    assert_eq!(layout.groups, 1, "one credit workgroup owns each agent");
    let source = global_credit::global_source_with_store_suppression(Some(packed), true);
    let source = replace_once(
        &source,
        &packed_encoder::credit_source(true),
        &credit_source(),
    );
    let source = replace_once(
        &source,
        "        phase_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x);",
        "        projection_credit_agent(agent_id, lid.x);",
    );
    let source = format!("{source}{}", layout.header());
    let shared: Vec<_> = source
        .lines()
        .filter(|line| line.starts_with("var<workgroup>"))
        .collect();
    assert_eq!(
        shared,
        ["var<workgroup> agent_projection_inputs: array<f32, PROJECTION_VISUAL_COUNT>;"]
    );
    source
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants.insert("GLOBAL_CREDIT_GROUPS_PER_AGENT".into(), 1.0);
    constants
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str) -> wgpu::ComputePipeline {
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
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel),
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: Option<global_credit::Pipelines>,
    candidate: bool,
    layout: ProjectionLayout,
}

impl Arms {
    fn prepare(width: u32, height: u32, boundary: bool) -> (GpuKernel, Self) {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
        kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
            &kernel,
            &optimized_brain(),
            &constants(&kernel),
            true,
        );
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            main_width::MAIN_THREADS
        );
        assert_eq!(kernel.agent_count, AGENTS);
        let layout = ProjectionLayout::new(&kernel);
        let packed =
            packed_encoder::Cache::new_with_extra_scratch(&kernel, layout.extra_words).unwrap();
        let mut candidate = global_credit::Pipelines::new_packed_with_main128(
            &kernel,
            &optimized_brain(),
            &constants(&kernel),
            true,
        )
        .unwrap();
        eprintln!(
            "AGENT_PROJECTION_COMPILE_MAIN_BEGIN width={width} height={height} main_threads={}",
            main_width::MAIN_THREADS
        );
        candidate.main = pipeline(
            &kernel,
            main_source(&kernel, &packed, &layout),
            "kernel_tick",
        );
        eprintln!("AGENT_PROJECTION_COMPILE_MAIN_END");
        eprintln!(
            "AGENT_PROJECTION_COMPILE_GLOBAL_BEGIN shared_bytes={}",
            layout.shared_bytes
        );
        candidate.global = pipeline(
            &kernel,
            global_source(&kernel, &packed, &layout),
            "global_credit_tick",
        );
        eprintln!("AGENT_PROJECTION_COMPILE_GLOBAL_END");
        let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
        candidate.bind_groups = std::array::from_fn(|index| {
            global_credit::private_bind_group(&kernel, &binding, packed.buffer(), index)
        });
        candidate.global_workgroups = kernel.agent_count.checked_add(1).unwrap();
        assert!(
            candidate.global_workgroups
                <= kernel.device.limits().max_compute_workgroups_per_dimension
        );
        assert_eq!(candidate.main_threads, main_width::MAIN_THREADS);
        candidate.packed_encoder = Some(packed);
        (
            kernel,
            Self {
                parked: Some(candidate),
                candidate: false,
                layout,
            },
        )
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
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
        if candidate && !cache(kernel).is_valid() {
            self.invalidate(kernel);
        }
        assert!(kernel.global_credit_active());
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            main_width::MAIN_THREADS
        );
    }

    fn invalidate(&self, kernel: &GpuKernel) {
        if self.candidate {
            for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
                kernel.queue.write_buffer(
                    cache(kernel).buffer(),
                    bytes(self.layout.base_words + agent * self.layout.agent_stride),
                    bytemuck::cast_slice(&[0.0f32; HEADER_WORDS]),
                );
            }
        }
    }

    fn advance(&self, kernel: &mut GpuKernel, cycle: u32, count: u32) {
        if !cache(kernel).is_valid() {
            self.invalidate(kernel);
        }
        production_advance(kernel, cycle, count);
    }

    fn restore(&self, kernel: &mut GpuKernel, saved: &Checkpoint) {
        restore(kernel, saved);
        self.invalidate(kernel);
    }

    fn state(&self, kernel: &GpuKernel) -> TestResult<State> {
        let state = capture_state(kernel)?;
        if self.candidate {
            let storage = cache(kernel).buffer();
            let actual = read_buffer(kernel, storage, storage.size())?;
            let matrix_words = kernel
                .layout
                .feature_count
                .checked_mul(ENCODED_DIMENSION)
                .unwrap();
            for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
                let private =
                    usize::try_from(bytes(self.layout.prefix_words + agent * matrix_words))
                        .unwrap();
                let public = usize::try_from(bytes(agent * kernel.layout.brain_stride)).unwrap();
                let size = usize::try_from(bytes(matrix_words)).unwrap();
                assert_eq!(
                    &actual[private..private + size],
                    &state[BRAIN_BUFFER][public..public + size],
                    "fresh projection private/scalar mirror agent={agent}"
                );
            }
        } else {
            super::packed_store_validation::assert_mirror(kernel, &state)?;
        }
        Ok(state)
    }

    fn counts(&self, kernel: &GpuKernel) -> TestResult<(f32, f32)> {
        assert!(self.candidate);
        let raw = read_buffer(
            kernel,
            cache(kernel).buffer(),
            bytes(self.layout.prefix_words),
        )?;
        let words: &[f32] = bytemuck::cast_slice(&raw);
        let mut hits = 0.0;
        let mut misses = 0.0;
        for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
            let base = self.layout.base_words + agent * self.layout.agent_stride;
            hits += words[base + HITS_OFFSET];
            misses += words[base + MISSES_OFFSET];
        }
        Ok((hits, misses))
    }
}

/// Fixture-only input injection; all world, vision and brain dispatches still
/// execute normally. Inactive agents and every nonvisual sensory word are left
/// untouched. Both arms and their replays receive the identical visual event.
fn hold_raw_visual(kernel: &GpuKernel, previous: &State) {
    let visual_count = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
    let mut visual = vec![HELD_SKY_DEPTH; visual_count];
    assert!(kernel
        .layout
        .vision_color_count
        .is_multiple_of(VECTOR_WORDS));
    for ray in 0..kernel.layout.vision_color_count / VECTOR_WORDS {
        let begin = ray * VECTOR_WORDS;
        visual[begin..begin + VECTOR_WORDS].copy_from_slice(&HELD_SKY_RGBA);
    }
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let alive_byte = (agent * PHYS_STRIDE + P_ALIVE) * WORD_BYTES;
        let alive = f32::from_le_bytes(
            previous[PHYSICS_BUFFER][alive_byte..alive_byte + WORD_BYTES]
                .try_into()
                .unwrap(),
        );
        if alive >= ALIVE_THRESHOLD {
            kernel.queue.write_buffer(
                &kernel.sensory_buffer,
                bytes(agent * kernel.layout.sensory_stride),
                bytemuck::cast_slice(&visual),
            );
        }
    }
}

fn trajectory(kernel: &mut GpuKernel, arms: &Arms, initial: &State) -> TestResult<Vec<State>> {
    let mut snapshots = Vec::new();
    let mut cycle = 0;
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        if HELD_RAW_CYCLES.contains(&cycle) {
            hold_raw_visual(kernel, snapshots.last().unwrap());
        }
        arms.advance(kernel, cycle, count);
        let state = arms.state(kernel)?;
        assert_inactive_agent_unchanged(
            kernel,
            initial,
            &state,
            INACTIVE_AGENT,
            "fresh projection",
        );
        snapshots.push(state);
        cycle += count;
    }
    assert_eq!(cycle, TIMED_CYCLES);
    Ok(snapshots)
}

#[test]
#[ignore = "requires GPU; strict full-state parity and same-arm replay"]
fn agent_projection_preserves_complete_state_and_repeatability() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let reference = trajectory(&mut kernel, &arms, &initial)?;
        arms.restore(&mut kernel, &saved);
        let reference_replay = trajectory(&mut kernel, &arms, &initial)?;
        for (expected, actual) in reference.iter().zip(&reference_replay) {
            assert_state_equal(&kernel, expected, actual);
        }
        arms.restore(&mut kernel, &saved);
        arms.activate(&mut kernel, true);
        let candidate = trajectory(&mut kernel, &arms, &initial)?;
        let (hits, misses) = arms.counts(&kernel)?;
        assert!(
            hits > 0.0 && misses > 0.0,
            "exercise cached and fresh routes"
        );
        arms.restore(&mut kernel, &saved);
        let repeated = trajectory(&mut kernel, &arms, &initial)?;
        let mut cycle = 0;
        for (((expected, actual), repeated), count) in reference
            .iter()
            .zip(&candidate)
            .zip(&repeated)
            .zip(PARITY_CHUNKS)
        {
            cycle += count;
            assert_state_equal(&kernel, actual, repeated);
            if expected != actual {
                compare_rounding_state(
                    &kernel,
                    expected,
                    actual,
                    &format!("agent_projection {width}x{height} cycles={cycle}"),
                );
            }
            assert_state_equal(&kernel, expected, actual);
        }
        report_seeded_behavior(
            &[(
                FIXTURE_SEED,
                extract_behavior(&kernel, reference.last().unwrap()),
                extract_behavior(&kernel, candidate.last().unwrap()),
            )],
            &format!("agent_projection {width}x{height} cycles={TIMED_CYCLES}"),
        );
        println!("AGENT_PROJECTION_PARITY width={width} height={height} cycles={TIMED_CYCLES} held_raw_cycles={HELD_RAW_CYCLES:?} world_vision_dispatches_unchanged=true same_arm_exact_buffers={MUTABLE_BUFFERS} cross_arm=all13_exact inactive_exact=true private_mirror_exact=true hits={hits} misses={misses} cache_error_accumulation=false");
    }
    Ok(())
}

#[test]
fn agent_projection_source_preserves_credit_math_and_fallback() {
    let credit = credit_source();
    let original = include_str!("../shaders/kernel/phase_packed_encoder_credit.wgsl");
    let begin = original
        .find("    // Skipped components execute neither addition")
        .unwrap();
    let end = original
        .find("    packed_encoder.weights[address] = weight;")
        .unwrap();
    assert!(credit.contains(&original[begin..end]));
    assert_eq!(
        credit.matches("brain_state[scalar_address").count(),
        VECTOR_WORDS
    );
    assert!(credit.contains("if !any(credit_enabled) { return original_weight; }"));
    assert!(credit.contains("P_ALIVE] < 0.5 { return original_weight; }"));
    let encode = encode_source();
    assert_eq!(encode.matches("fn coop_encode(").count(), 1);
    assert_eq!(
        encode
            .matches("fn fresh_projection_original_encode(")
            .count(),
        1
    );
    assert_eq!(encode.matches("fast_tanh(reduced)").count(), 2);
    assert!(encode.contains("bitcast<u32>(predicted) == bitcast<u32>(s_features[feature])"));
    assert!(encode.contains("workgroupUniformLoad(&s_reinf_dot[PROJECTION_GATE_GROUP_COUNT])"));
    assert_eq!(credit.matches("var<workgroup>").count(), 1);
    assert!(credit.contains("storageBarrier();\n    workgroupBarrier();"));
    assert!(encode.contains("PROJECTION_VISUAL_COUNT % PACKED_ENCODER_INNER_LANES"));
    assert!(encode.contains(
        "PROJECTION_DEATH_OFFSET] == physics_state[agent_id * PHYS_STRIDE + P_DEATH_COUNT]"
    ));
}

#[test]
fn agent_projection_owners_preserve_visual_prefix_and_nonvisual_continuation() {
    let output_vectors = ENCODED_DIMENSION / VECTOR_WORDS;
    for (width, height) in FIELDS {
        let visual_count = usize::try_from(width * height).unwrap() * (VECTOR_WORDS + 1);
        let feature_count = visual_count + NON_VISUAL_FEATURE_COUNT;
        let mut writes = vec![0_u32; feature_count.checked_mul(output_vectors).unwrap()];
        for owner in 0..PARTIAL_LANES * output_vectors {
            let lane = owner / output_vectors;
            let vector = owner % output_vectors;
            for feature in (lane..feature_count).step_by(PARTIAL_LANES) {
                writes[feature * output_vectors + vector] += 1;
            }
            let first_nonvisual = visual_count
                + (lane + PARTIAL_LANES - visual_count % PARTIAL_LANES) % PARTIAL_LANES;
            let cached_then_fresh: Vec<_> = (lane..visual_count)
                .step_by(PARTIAL_LANES)
                .chain((first_nonvisual..feature_count).step_by(PARTIAL_LANES))
                .collect();
            assert_eq!(
                cached_then_fresh,
                (lane..feature_count)
                    .step_by(PARTIAL_LANES)
                    .collect::<Vec<_>>()
            );
        }
        assert!(writes.iter().all(|count| *count == 1));
        let mut q_writes = vec![0_u32; visual_count];
        for thread in 0..CREDIT_THREADS {
            for feature in (thread..visual_count).step_by(CREDIT_THREADS) {
                q_writes[feature] += 1;
            }
        }
        assert!(q_writes.iter().all(|count| *count == 1));
    }
}

fn state_float(bytes: &[u8], word: usize) -> f32 {
    let begin = word.checked_mul(WORD_BYTES).unwrap();
    f32::from_le_bytes(bytes[begin..begin + WORD_BYTES].try_into().unwrap())
}

#[test]
#[ignore = "GPU source-publication check; identical one-cycle public state"]
fn agent_projection_publishes_current_raw_minus_stored_mean() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    const SENSORY_BUFFER: usize = 7;
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let before = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        arms.advance(&mut kernel, 0, 1);
        let reference = arms.state(&kernel)?;
        arms.restore(&mut kernel, &saved);
        arms.activate(&mut kernel, true);
        arms.advance(&mut kernel, 0, 1);
        let actual = arms.state(&kernel)?;
        assert_state_equal(&kernel, &reference, &actual);
        let prefix = read_buffer(
            &kernel,
            cache(&kernel).buffer(),
            bytes(arms.layout.prefix_words),
        )?;
        let mean_offset = kernel.layout.brain_stride - kernel.layout.feature_count;
        let mut checked = 0;
        for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
            let alive = state_float(&actual[PHYSICS_BUFFER], agent * PHYS_STRIDE + P_ALIVE)
                >= ALIVE_THRESHOLD;
            let header = arms.layout.base_words + agent * arms.layout.agent_stride;
            assert_eq!(
                state_float(&prefix, header + VALID_OFFSET),
                if alive { 1.0 } else { 0.0 }
            );
            assert_eq!(
                state_float(&prefix, header + DEATH_OFFSET).to_bits(),
                state_float(&actual[PHYSICS_BUFFER], agent * PHYS_STRIDE + P_DEATH_COUNT).to_bits()
            );
            if !alive {
                continue;
            }
            for feature in 0..arms.layout.visual_count {
                let raw = state_float(
                    &before[SENSORY_BUFFER],
                    agent * kernel.layout.sensory_stride + feature,
                );
                let mean = state_float(
                    &actual[BRAIN_BUFFER],
                    agent * kernel.layout.brain_stride + mean_offset + feature,
                );
                let predicted = state_float(
                    &prefix,
                    agent * kernel.layout.brain_scratch_stride + feature,
                );
                assert_eq!(
                    predicted.to_bits(),
                    (raw - mean).to_bits(),
                    "q publication agent={agent} feature={feature}"
                );
                checked += 1;
            }
        }
        println!("AGENT_PROJECTION_PUBLICATION width={width} height={height} checked_visual_words={checked} q_bits_exact=true death_header_exact=true exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true");
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle paired benchmark; run only after raw arithmetic validation"]
fn benchmark_agent_projection() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut arms) = Arms::prepare(FIELDS[0].0, FIELDS[0].1, false);
    arms.advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    let mut expected = Vec::new();
    for candidate in [false, true] {
        arms.restore(&mut kernel, &saved);
        arms.activate(&mut kernel, candidate);
        arms.advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
        expected.push(arms.state(&kernel)?);
    }
    if expected[0] != expected[1] {
        compare_rounding_state(
            &kernel,
            &expected[0],
            &expected[1],
            "agent_projection timing preflight",
        );
    }
    assert_state_equal(&kernel, &expected[0], &expected[1]);
    let (hits, misses) = arms.counts(&kernel)?;
    assert!(hits > 0.0 && misses > 0.0);
    let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            arms.restore(&mut kernel, &saved);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            arms.advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            times[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected[arm], &arms.state(&kernel)?);
        }
    }
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    let reference = times[0][TIMING_PAIRS / 2];
    let candidate = times[1][TIMING_PAIRS / 2];
    println!("AGENT_PROJECTION_TIMING warmup={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} baseline_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} hits={hits} misses={misses} same_arm_exact_buffers={MUTABLE_BUFFERS} cross_arm_all13_exact=true private_mirror_exact=true extra_storage_bytes={} shared_bytes={} global_groups={} added_dispatch=false raw_publication_timed=true full_simulation=true", reference / candidate, bytes(arms.layout.extra_words), arms.layout.shared_bytes, kernel.agent_count + 1);
    Ok(())
}
