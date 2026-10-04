//! Optional typed encoder weights with an authoritative scalar mirror.
//! Binding 13 holds ordinary scalar scratch followed by independent vec4
//! storage. Packed credit writes both matrices; host writes and scalar
//! schedules invalidate the packed copy for a GPU import before its next use.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{GpuKernel, BRAIN_WORKGROUP_THREADS, MAX_DISPATCH_WORKGROUPS};
use crate::buffers::{BrainLayout, ENCODED_DIMENSION};
use crate::complex::VISUAL_FEATURE_COUNT;

/// Adjacent feature-major outputs fill one aligned storage vector.
const VECTOR_WIDTH: usize = 4;
/// Two vector prefetches retain eight weight scalars per invocation.
const VECTOR_PREFETCH: u32 = 2;
/// Preserve the original encoder's four feature accumulation sequences.
const ENCODER_INNER_LANES: u32 = 4;
/// The first global workgroup executes the unchanged world phases.
const WORLD_WORKGROUPS: u32 = 1;
const WORD_BYTES: usize = size_of::<f32>();
/// Account for each workgroup global at a conservative 16-byte boundary.
const WORKGROUP_ALIGNMENT_BYTES: u64 = 16;
/// Rounded bytes of the 28 fixed globals in brain passes, brain inner, and
/// kernel tick; feature, visual, and dense-partial arrays are added separately.
const FIXED_WORKGROUP_BYTES: u64 = 6_064;
/// Cortex scratch stores luminance, surround, and horizontal-filter images.
const CORTEX_IMAGE_PLANES: usize = 3;
/// This fragment updates complete vectors and their scalar serialization copy.
pub(super) const CREDIT_SOURCE: &str =
    include_str!("../shaders/kernel/phase_packed_encoder_credit.wgsl");

/// Every derived size and shader index is checked before allocation.
struct Shape {
    agents: u32,
    scratch_words: u32,
    scalar_scratch_bytes: u64,
    prefix_bytes: u64,
    matrix_bytes: u64,
    brain_stride_bytes: u64,
    brain_bytes: u64,
    total_bytes: u64,
    groups_per_agent: u32,
    global_workgroups: u32,
}

fn bytes(words: usize) -> Option<u64> {
    u64::try_from(words.checked_mul(WORD_BYTES)?).ok()
}

fn aligned_workgroup_bytes(words: usize) -> Option<u64> {
    bytes(words)?
        .checked_add(WORKGROUP_ALIGNMENT_BYTES - 1)?
        .checked_div(WORKGROUP_ALIGNMENT_BYTES)?
        .checked_mul(WORKGROUP_ALIGNMENT_BYTES)
}

/// Packed encoding grows only the existing dense scratch, without adding a
/// Metal threadgroup resource. Reject layouts exceeding the device's storage
/// limit before compiling the optional pipeline.
fn workgroup_bytes(layout: &BrainLayout) -> Option<u64> {
    let visual_words = if layout.visual_cortex_enabled {
        layout
            .retina_pixel_count
            .checked_mul(CORTEX_IMAGE_PLANES)?
            .checked_add(VISUAL_FEATURE_COUNT)?
    } else {
        1
    };
    let partial_words =
        ENCODED_DIMENSION.checked_mul(usize::try_from(ENCODER_INNER_LANES).ok()?)?;
    FIXED_WORKGROUP_BYTES
        .checked_add(aligned_workgroup_bytes(layout.feature_count)?)?
        .checked_add(aligned_workgroup_bytes(visual_words)?)?
        .checked_add(aligned_workgroup_bytes(partial_words)?)
}

impl Shape {
    fn new(layout: &BrainLayout, agents: u32, limits: &wgpu::Limits) -> Option<Self> {
        if agents == 0
            || layout.feature_count == 0
            || layout.brain_scratch_stride == 0
            || !ENCODED_DIMENSION.is_multiple_of(VECTOR_WIDTH)
            || workgroup_bytes(layout)? > u64::from(limits.max_compute_workgroup_storage_size)
        {
            return None;
        }
        let count = usize::try_from(agents).ok()?;
        let matrix_words = layout.feature_count.checked_mul(ENCODED_DIMENSION)?;
        if matrix_words > layout.brain_stride {
            return None;
        }
        let scalar_scratch_words = layout.brain_scratch_stride.checked_mul(count)?;
        let scratch_words = scalar_scratch_words
            .checked_add(VECTOR_WIDTH - 1)?
            .checked_div(VECTOR_WIDTH)?
            .checked_mul(VECTOR_WIDTH)?;
        let scratch_words = u32::try_from(scratch_words).ok()?;
        let vectors_per_agent = u32::try_from(matrix_words / VECTOR_WIDTH).ok()?;
        vectors_per_agent.checked_mul(agents)?;
        let brain_words = layout.brain_stride.checked_mul(count)?;
        u32::try_from(brain_words).ok()?;
        let prefix_bytes = u64::from(scratch_words).checked_mul(u64::try_from(WORD_BYTES).ok()?)?;
        let matrix_bytes = bytes(matrix_words)?;
        let total_bytes = prefix_bytes.checked_add(matrix_bytes.checked_mul(u64::from(agents))?)?;
        let groups_per_agent = vectors_per_agent.div_ceil(BRAIN_WORKGROUP_THREADS);
        let global_workgroups = groups_per_agent
            .checked_mul(agents)?
            .checked_add(WORLD_WORKGROUPS)?;
        if global_workgroups > MAX_DISPATCH_WORKGROUPS
            || global_workgroups > limits.max_compute_workgroups_per_dimension
            || total_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || total_bytes > limits.max_buffer_size
        {
            return None;
        }
        Some(Self {
            agents,
            scratch_words,
            scalar_scratch_bytes: bytes(scalar_scratch_words)?,
            prefix_bytes,
            matrix_bytes,
            brain_stride_bytes: bytes(layout.brain_stride)?,
            brain_bytes: bytes(brain_words)?,
            total_bytes,
            groups_per_agent,
            global_workgroups,
        })
    }

    fn common_source(&self) -> String {
        let common = include_str!("../shaders/kernel/common.wgsl");
        assert!(common.contains("const O_ENC_WEIGHTS: u32 = 0u;"));
        let declaration = format!(
            "struct PackedEncoder {{\n    scratch: array<f32, {}>,\n    weights: array<vec4<f32>>,\n}}\n@group(0) @binding(13) var<storage, read_write> packed_encoder: PackedEncoder;",
            self.scratch_words,
        );
        let source = replace_once(
            common,
            "@group(0) @binding(13) var<storage, read_write> brain_scratch:       array<f32>;",
            &declaration,
        );
        // Storage members require creation-fixed array lengths. Only the
        // scalar prefix uses a checked host literal; weights remain runtime-sized.
        format!("{source}\nconst PACKED_ENCODER_WIDTH: u32 = {VECTOR_WIDTH}u;\nconst PACKED_ENCODER_OUTPUT_VECTORS: u32 = ENCODED_DIMENSION / PACKED_ENCODER_WIDTH;\nconst PACKED_ENCODER_PREFETCH: u32 = {VECTOR_PREFETCH}u;\nconst PACKED_ENCODER_INNER_LANES: u32 = {ENCODER_INNER_LANES}u;\nconst PACKED_ENCODER_PARTIAL_WORDS: u32 = ENCODED_DIMENSION * PACKED_ENCODER_INNER_LANES;\n")
    }
}

/// Packed weights are a disposable cache; the scalar brain remains authoritative.
pub(super) struct Cache {
    buffer: wgpu::Buffer,
    shape: Shape,
    valid: AtomicBool,
}

impl Cache {
    pub(super) fn new(kernel: &GpuKernel) -> Option<Self> {
        let shape = Shape::new(&kernel.layout, kernel.agent_count, &kernel.device.limits())?;
        Self::from_shape(kernel, shape)
    }

    /// Append private diagnostic scratch while retaining the production import
    /// and invalidation lifecycle; all scalar public scratch keeps its offsets.
    #[cfg(test)]
    pub(super) fn new_with_extra_scratch(kernel: &GpuKernel, extra_words: usize) -> Option<Self> {
        let limits = kernel.device.limits();
        let mut shape = Shape::new(&kernel.layout, kernel.agent_count, &limits)?;
        let words = usize::try_from(shape.scratch_words)
            .ok()?
            .checked_add(extra_words)?
            .checked_add(VECTOR_WIDTH - 1)?
            .checked_div(VECTOR_WIDTH)?
            .checked_mul(VECTOR_WIDTH)?;
        shape.scratch_words = u32::try_from(words).ok()?;
        shape.prefix_bytes = bytes(words)?;
        shape.total_bytes = shape
            .prefix_bytes
            .checked_add(shape.matrix_bytes.checked_mul(u64::from(shape.agents))?)?;
        if shape.total_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || shape.total_bytes > limits.max_buffer_size
        {
            return None;
        }
        Self::from_shape(kernel, shape)
    }

    fn from_shape(kernel: &GpuKernel, shape: Shape) -> Option<Self> {
        if shape.brain_bytes != kernel.brain_state_buffer.size()
            || shape.scalar_scratch_bytes != kernel.brain_scratch_buffer.size()
        {
            return None;
        }
        let buffer = kernel.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed_encoder_cache"),
            size: shape.total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            buffer,
            shape,
            valid: AtomicBool::new(false),
        })
    }

    pub(super) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub(super) fn groups_per_agent(&self) -> u32 {
        self.shape.groups_per_agent
    }

    pub(super) fn global_workgroups(&self) -> u32 {
        self.shape.global_workgroups
    }

    pub(super) fn common_source(&self) -> String {
        self.shape.common_source()
    }

    /// Call after any host or fallback shader write to the scalar encoder.
    pub(super) fn invalidate(&self) {
        self.valid.store(false, Ordering::Release);
    }

    /// Record a cold import before the first packed compute pass in a batch.
    /// The caller must submit this encoder before another packed dispatch, or
    /// invalidate the cache if recording is abandoned. Atomic claiming avoids
    /// overwriting an invalidation that arrives while copies are being recorded.
    /// Returns whether import copies were recorded.
    pub(super) fn record_import(
        &self,
        kernel: &GpuKernel,
        encoder: &mut wgpu::CommandEncoder,
    ) -> bool {
        if self.valid.swap(true, Ordering::AcqRel) {
            return false;
        }
        for agent in 0..u64::from(self.shape.agents) {
            // Shape checked the complete population ranges before allocation.
            let brain_offset = agent * self.shape.brain_stride_bytes;
            let packed_offset = self.shape.prefix_bytes + agent * self.shape.matrix_bytes;
            encoder.copy_buffer_to_buffer(
                &kernel.brain_state_buffer,
                brain_offset,
                &self.buffer,
                packed_offset,
                self.shape.matrix_bytes,
            );
        }
        true
    }

    #[cfg(test)]
    pub(super) fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
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

/// Optionally suppress both destinations only after the original arithmetic
/// produced four bit-identical words. Cache import and mirrored changed writes
/// maintain scalar == packed, so this needs no extra validity state. Comparing
/// integer bits preserves signed-zero changes that floating equality would hide.
pub(super) fn credit_source(skip_unchanged_stores: bool) -> String {
    if !skip_unchanged_stores {
        return CREDIT_SOURCE.to_owned();
    }
    assert!(!CREDIT_SOURCE.contains("Barrier"));
    let source = replace_once(
        CREDIT_SOURCE,
        "    var weight = packed_encoder.weights[address];",
        "    let original_weight = packed_encoder.weights[address];\n    var weight = original_weight;",
    );
    let source = replace_once(
        &source,
        "    packed_encoder.weights[address] = weight;",
        "    if all(bitcast<vec4<u32>>(weight) == bitcast<vec4<u32>>(original_weight)) { return; }\n    packed_encoder.weights[address] = weight;",
    );
    assert_eq!(
        source.matches("brain_state[scalar_address").count(),
        VECTOR_WIDTH
    );
    source
}

/// Replace only encoder arithmetic and redirect scalar scratch accesses.
/// Apply context gathering and feature publication first, so their scratch
/// references are redirected too. Predictor width and fusion are independent.
pub(super) fn packed_passes(source: &str) -> String {
    const ENCODE_DEFINITION: &str = "fn coop_encode(agent_id: u32, tid: u32) {";
    assert_eq!(source.matches(ENCODE_DEFINITION).count(), 1);
    assert!(source.contains("const DENSE_INNER_LANES: u32 = 4u;"));
    let start = source.find(ENCODE_DEFINITION).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + "\n}".len();
    let original = &source[start..end];
    // Plain and supported scalar-prefetch encoders all seed the same biases
    // and read the same matrix. Reapplying this transform fails these guards.
    assert_eq!(original.matches("O_ENC_BIASES").count(), 1);
    assert!(original.contains("O_ENC_WEIGHTS"));
    let changed = replace_once(
        source,
        original,
        include_str!("../shaders/kernel/brain_packed_encoder.wgsl"),
    );
    replace_once(
        &changed,
        "var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;",
        "var<workgroup> s_dense_partials: array<f32, PACKED_ENCODER_PARTIAL_WORDS>;",
    )
    .replace("brain_scratch[", "packed_encoder.scratch[")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_kernel::{
        compose_brain_passes, context_gather::gather_context, dense_prefetch,
        predictor_fusion::fuse_inline_predictor, predictor_width,
    };

    /// Default and odd retina sizes exercise complete and partial vector tiles.
    const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
    const AGENTS: u32 = 10;
    const DEFAULT_GLOBAL_GROUPS: u32 = 341;
    const DEFAULT_PACKED_BYTES: u64 = 1_400_000;
    /// Both existing scalar prefetch widths must be replaceable.
    const PREFETCH_FACTORS: [u32; 2] = [4, 8];
    /// The production context cache stages eight recalled patterns per lane set.
    const GATHER_LANES: u32 = 8;

    #[test]
    fn packed_encoder_shape_preserves_aligned_nonoverlapping_ranges() {
        for (width, height) in FIELDS {
            let layout = BrainLayout::new(width, height);
            let shape = Shape::new(&layout, AGENTS, &wgpu::Limits::default()).unwrap();
            assert_eq!(shape.prefix_bytes % bytes(VECTOR_WIDTH).unwrap(), 0);
            assert!(shape.prefix_bytes >= shape.scalar_scratch_bytes);
            assert!(shape.prefix_bytes - shape.scalar_scratch_bytes < bytes(VECTOR_WIDTH).unwrap());
            assert_eq!(
                shape.matrix_bytes,
                bytes(layout.feature_count * ENCODED_DIMENSION).unwrap()
            );
            assert_eq!(
                shape.total_bytes,
                shape.prefix_bytes + shape.matrix_bytes * u64::from(AGENTS)
            );
            assert!(shape.matrix_bytes <= shape.brain_stride_bytes);
            assert!(shape.common_source().contains("weights: array<vec4<f32>>"));
        }
        let shape = Shape::new(&BrainLayout::default(), AGENTS, &wgpu::Limits::default()).unwrap();
        assert_eq!(shape.global_workgroups, DEFAULT_GLOBAL_GROUPS);
        assert_eq!(shape.total_bytes, DEFAULT_PACKED_BYTES);
    }

    #[test]
    fn packed_encoder_shape_rejects_overflow_and_device_limits() {
        let layout = BrainLayout::default();
        let limits = wgpu::Limits::default();
        assert!(Shape::new(&layout, 0, &limits).is_none());
        assert!(Shape::new(&layout, u32::MAX, &limits).is_none());
        let mut overflow = layout.clone();
        overflow.feature_count = usize::MAX;
        assert!(Shape::new(&overflow, AGENTS, &limits).is_none());
        let too_small = wgpu::Limits {
            max_storage_buffer_binding_size: 1,
            ..limits.clone()
        };
        assert!(Shape::new(&layout, AGENTS, &too_small).is_none());
        let too_few_groups = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..limits.clone()
        };
        assert!(Shape::new(&layout, AGENTS, &too_few_groups).is_none());
        let shared_bytes = u32::try_from(workgroup_bytes(&layout).unwrap()).unwrap();
        let too_little_shared = wgpu::Limits {
            max_compute_workgroup_storage_size: shared_bytes - 1,
            ..limits.clone()
        };
        assert!(Shape::new(&layout, AGENTS, &too_little_shared).is_none());
        let sufficient_shared = wgpu::Limits {
            max_compute_workgroup_storage_size: shared_bytes,
            ..limits
        };
        assert!(Shape::new(&layout, AGENTS, &sufficient_shared).is_some());
    }

    #[test]
    fn packed_encoder_shared_estimate_tracks_shader_declarations() {
        use crate::buffers::{MEMORY_CAP, PREDICTOR_DIMENSION, RECALL_K};

        let sources = [
            include_str!("../shaders/kernel/brain_passes.wgsl"),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ];
        let mut fixed_bytes = 0;
        for declaration in sources
            .iter()
            .flat_map(|source| source.lines())
            .filter(|line| line.starts_with("var<workgroup>"))
        {
            let words = if let Some((_, array)) = declaration.split_once("array<") {
                let (kind, count) = array.trim_end_matches(">;").split_once(", ").unwrap();
                assert!(matches!(kind, "f32" | "u32"));
                match count {
                    "FEATURE_COUNT" | "VC_SCRATCH_LEN" | "BRAIN_WORKGROUP_SIZE" => continue,
                    "ENCODED_DIMENSION" => ENCODED_DIMENSION,
                    "PREDICTOR_DIMENSION" => PREDICTOR_DIMENSION,
                    "MEMORY_CAP" => MEMORY_CAP,
                    "RECALL_K" => RECALL_K,
                    literal => literal.parse().expect("known workgroup array size"),
                }
            } else {
                assert!(declaration.ends_with(": f32;") || declaration.ends_with(": u32;"));
                1
            };
            fixed_bytes += aligned_workgroup_bytes(words).unwrap();
        }
        assert_eq!(fixed_bytes, FIXED_WORKGROUP_BYTES);

        const CORTEX_RETINA_SIDE: usize = 24;
        const CORTEX_SHARED_BYTES: u64 = 15_296;
        let cortex = BrainLayout::with_retina_flagged(
            FIELDS[0].0,
            FIELDS[0].1,
            CORTEX_RETINA_SIDE,
            CORTEX_RETINA_SIDE,
            true,
            false,
        );
        assert_eq!(workgroup_bytes(&cortex), Some(CORTEX_SHARED_BYTES));
        assert!(Shape::new(&cortex, AGENTS, &wgpu::Limits::default()).is_some());
    }

    #[test]
    fn packed_encoder_composes_with_all_dense_and_context_options() {
        for whitening in [false, true] {
            let original = compose_brain_passes(whitening);
            let fused = fuse_inline_predictor(&original);
            let mut sources = vec![original, fused.clone()];
            for factor in PREFETCH_FACTORS {
                sources.push(dense_prefetch::prefetch_passes(&fused, factor));
            }
            for source in sources {
                for lanes in predictor_width::LANE_WIDTHS {
                    let widened = predictor_width::wider_predictor(&source, lanes);
                    for context in [false, true] {
                        let baseline = if context {
                            gather_context(&widened, GATHER_LANES)
                        } else {
                            widened.clone()
                        };
                        let packed = packed_passes(&baseline);
                        assert_ne!(packed, baseline);
                        assert_eq!(packed.matches("fn coop_encode(").count(), 1);
                        assert!(packed.contains("packed_encoder.weights["));
                        assert!(!packed.contains("brain_scratch["));
                        assert_eq!(
                            packed.matches("var<workgroup>").count(),
                            baseline.matches("var<workgroup>").count()
                        );
                        assert!(std::panic::catch_unwind(|| packed_passes(&packed)).is_err());
                    }
                }
            }
        }
    }
}
