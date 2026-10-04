//! Optional encoder-credit workgroups beside the single global world update.
//! The main kernel publishes its actual adapted features into private storage
//! at binding 13. Pointwise credit retains the original weight arithmetic and
//! clamps; the world branch does not access encoder weights or modify alive
//! flags. Optional packed encoder weights mirror every update into ordinary
//! brain storage, keeping public readbacks authoritative without extra dispatches.

use std::collections::HashMap;

use super::packed_encoder;
use super::{apply_subgroup_markers, GpuKernel, BRAIN_WORKGROUP_THREADS, MAX_DISPATCH_WORKGROUPS};
use crate::buffers::ENCODED_DIMENSION;

/// The first group runs the unchanged world phases and their local barriers.
const WORLD_WORKGROUPS: u32 = 1;
/// Both pipeline entries use a tick plus one phase-control push-constant word.
const PUSH_CONSTANT_BYTES: u32 = 8;

pub(super) struct Pipelines {
    pub(super) main: wgpu::ComputePipeline,
    pub(super) global: wgpu::ComputePipeline,
    pub(super) bind_groups: [wgpu::BindGroup; 2],
    pub(super) global_workgroups: u32,
    #[cfg(test)]
    pub(super) main_threads: u32,
    // Binding 13 differs from the ordinary group. Its lifetime follows both
    // groups, and every live agent rewrites its features before any read.
    _features: Option<wgpu::Buffer>,
    pub(super) packed_encoder: Option<packed_encoder::Cache>,
}

/// Checked shape calculation also bounds WGSL's per-agent u32 weight offset.
pub(super) fn workgroup_count(feature_count: usize, agent_count: u32) -> Option<u32> {
    let weights = u32::try_from(feature_count.checked_mul(ENCODED_DIMENSION)?).ok()?;
    let groups = weights
        .div_ceil(BRAIN_WORKGROUP_THREADS)
        .checked_mul(agent_count)?
        .checked_add(WORLD_WORKGROUPS)?;
    (weights != 0 && groups <= MAX_DISPATCH_WORKGROUPS).then_some(groups)
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

/// Replace only independent weight updates with publication of the feature
/// values actually consumed by encode. Inverting the updated running mean
/// would not recover these FP32 values exactly, so adaptation is never rerun.
fn publish_features(brain_passes: &str) -> String {
    const BEGIN: &str = "    if (run_encoder_credit) {\n";
    const END: &str = "    // ── Compute memory-key norm ONCE (memory reinforcement tiling)";
    assert_eq!(brain_passes.matches(BEGIN).count(), 1);
    assert_eq!(brain_passes.matches(END).count(), 1);
    let first = brain_passes.find(BEGIN).unwrap();
    let last = brain_passes.find(END).unwrap();
    assert!(first < last);
    let original = &brain_passes[first..last];
    assert_eq!(original.matches("O_ENC_WEIGHTS").count(), 2);
    assert!(!original.contains("Barrier"));
    assert!(!brain_passes[last..].contains("O_ENC_WEIGHTS"));
    replace_once(
        brain_passes,
        original,
        r"    if (run_encoder_credit) {
        let agent_scratch = agent_id * BRAIN_SCRATCH_STRIDE;
        for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
            brain_scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];
        }
    }

",
    )
}

pub(super) fn main_source(
    brain_passes: &str,
    has_subgroup: bool,
    packed: Option<&packed_encoder::Cache>,
) -> String {
    let passes = publish_features(brain_passes);
    let (common, passes) = if let Some(cache) = packed {
        (
            cache.common_source(),
            packed_encoder::packed_passes(&passes),
        )
    } else {
        (
            include_str!("../shaders/kernel/common.wgsl").to_owned(),
            passes,
        )
    };
    apply_subgroup_markers(
        &[
            &common,
            passes.as_str(),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        has_subgroup,
    )
}

pub(super) fn global_source(packed: Option<&packed_encoder::Cache>) -> String {
    global_source_with_store_suppression(packed, false)
}

pub(super) fn global_source_with_store_suppression(
    packed: Option<&packed_encoder::Cache>,
    skip_unchanged_stores: bool,
) -> String {
    let global = replace_once(include_str!("../shaders/kernel/global_tick.wgsl"),
        "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;",
        "fn global_world_inner(tid: u32) {");
    let phases = [
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
    ]
    .join("\n");
    assert!(!phases.contains("brain_state"));
    let common = packed.map_or_else(
        || include_str!("../shaders/kernel/common.wgsl").to_owned(),
        packed_encoder::Cache::common_source,
    );
    let credit = if packed.is_some() {
        packed_encoder::credit_source(skip_unchanged_stores)
    } else {
        include_str!("../shaders/kernel/phase_encoder_credit.wgsl").to_owned()
    };
    [
        &common,
        phases.as_str(),
        global.as_str(),
        credit.as_str(),
        include_str!("../shaders/kernel/global_credit_tick.wgsl"),
    ]
    .join("\n")
}

pub(super) fn private_bind_group(
    kernel: &GpuKernel,
    layout: &wgpu::BindGroupLayout,
    features: &wgpu::Buffer,
    config_index: usize,
) -> wgpu::BindGroup {
    let buffers = [
        &kernel.agent_phys_buffer,
        &kernel.decision_buffer,
        &kernel.heightmap_buffer,
        &kernel.biome_buffer,
        &kernel.world_config_bufs[config_index],
        &kernel.food_state_buffer,
        &kernel.food_flags_buffer,
        &kernel.food_grid_buffer,
        &kernel.agent_grid_buffer,
        &kernel.collision_scratch_buffer,
        &kernel.sensory_buffer,
        &kernel.brain_state_buffer,
        &kernel.pattern_buffer,
        features,
        &kernel.brain_config_buffer,
        &kernel.dispatch_args_buffer,
        &kernel.trail_ring_buffer,
        &kernel._sensory_next_buffer,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: u32::try_from(binding).unwrap(),
            resource: buffer.as_entire_binding(),
        })
        .collect();
    kernel.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("global_credit_private_features"),
        layout,
        entries: &entries,
    })
}

impl Pipelines {
    pub(super) fn new(
        kernel: &GpuKernel,
        brain_passes: &str,
        constants: &HashMap<String, f64>,
    ) -> Option<Self> {
        Self::new_variant(kernel, brain_passes, constants, false, false, false)
    }

    pub(super) fn new_packed(
        kernel: &GpuKernel,
        brain_passes: &str,
        constants: &HashMap<String, f64>,
    ) -> Option<Self> {
        Self::new_variant(kernel, brain_passes, constants, true, false, false)
    }

    /// Explicit selection keeps reference pipelines independent of process flags.
    pub(super) fn new_packed_with_store_suppression(
        kernel: &GpuKernel,
        brain_passes: &str,
        constants: &HashMap<String, f64>,
        skip_unchanged_stores: bool,
    ) -> Option<Self> {
        if !skip_unchanged_stores {
            return Self::new_packed(kernel, brain_passes, constants);
        }
        Self::new_variant(kernel, brain_passes, constants, true, true, false)
    }

    /// Use 128 main invocations only for the supported packed/predictor/context
    /// composition; unsupported source or resource shapes keep the 256 entry.
    pub(super) fn new_packed_with_main128(
        kernel: &GpuKernel,
        brain_passes: &str,
        constants: &HashMap<String, f64>,
        skip_unchanged_stores: bool,
    ) -> Option<Self> {
        Self::new_variant(
            kernel,
            brain_passes,
            constants,
            true,
            skip_unchanged_stores,
            true,
        )
    }

    fn new_variant(
        kernel: &GpuKernel,
        brain_passes: &str,
        constants: &HashMap<String, f64>,
        packed_requested: bool,
        skip_unchanged_stores: bool,
        main128_requested: bool,
    ) -> Option<Self> {
        if kernel.vision_stride != 1 {
            log::warn!("[GpuKernel] Global encoder credit requires vision stride 1; retaining the original schedule");
            return None;
        }
        let packed = if packed_requested {
            let cache = packed_encoder::Cache::new(kernel);
            if cache.is_none() {
                log::warn!("[GpuKernel] Packed encoder exceeds buffer, workgroup storage or dispatch limits; retaining scalar global credit");
            }
            cache
        } else {
            None
        };
        let Some(global_workgroups) = packed.as_ref().map_or_else(
            || workgroup_count(kernel.layout.feature_count, kernel.agent_count),
            |cache| Some(cache.global_workgroups()),
        ) else {
            log::warn!("[GpuKernel] Global encoder credit exceeds the dispatch dimension limit; retaining the original schedule");
            return None;
        };
        let mut constants = constants.clone();
        if let Some(cache) = &packed {
            constants.insert(
                "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
                f64::from(cache.groups_per_agent()),
            );
        }
        let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
        let layout = kernel
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("global_credit_layout"),
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..PUSH_CONSTANT_BYTES,
                }],
            });
        let create = |source: String, entry: &str| {
            let module = kernel
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
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
        let mut main_src = main_source(brain_passes, kernel.has_subgroup, packed.as_ref());
        let mut main_threads = BRAIN_WORKGROUP_THREADS;
        if main128_requested {
            if let Some(candidate) = super::main_width::try_transform(&main_src) {
                main_src = candidate;
                main_threads = super::main_width::MAIN_THREADS;
            } else {
                log::warn!("[GpuKernel] 128-thread main requires packed encoder, predictor16 and context8; retaining 256 threads");
            }
        }
        let main = create(main_src, "kernel_tick");
        let global_src = if skip_unchanged_stores {
            global_source_with_store_suppression(packed.as_ref(), true)
        } else {
            global_source(packed.as_ref())
        };
        let global = create(global_src, "global_credit_tick");
        let features = packed.is_none().then(|| {
            kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global_credit_features"),
                size: kernel.brain_scratch_buffer.size(),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let buffer = packed
            .as_ref()
            .map_or_else(|| features.as_ref().unwrap(), packed_encoder::Cache::buffer);
        let bind_groups =
            std::array::from_fn(|index| private_bind_group(kernel, &bind_layout, buffer, index));
        log::info!("[GpuKernel] Global encoder credit enabled: {global_workgroups} groups, packed encoder={}, unchanged vector stores skipped={}, main threads={main_threads}; fused serial uses a separate brain, other modes and partial/global-skipping probes retain the original schedule", packed.is_some(), packed.is_some() && skip_unchanged_stores);
        Some(Self {
            main,
            global,
            bind_groups,
            global_workgroups,
            #[cfg(test)]
            main_threads,
            _features: features,
            packed_encoder: packed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default raw 8×6 vision supplies 240 visual and 27 other features.
    const DEFAULT_FEATURES: usize = 267;
    /// 34,176 weights occupy 133 full groups and one half-full group.
    const DEFAULT_AGENTS: u32 = 10;
    const DEFAULT_GROUPS: u32 = 1_341;
    /// The dispatch ceiling admits 489 such agents, including the world group.
    const LAST_SUPPORTED_AGENTS: u32 = 489;
    const LAST_SUPPORTED_GROUPS: u32 = 65_527;

    #[test]
    fn dispatch_shape_preserves_partial_tiles_and_rejects_overflow() {
        assert_eq!(
            workgroup_count(DEFAULT_FEATURES, DEFAULT_AGENTS),
            Some(DEFAULT_GROUPS)
        );
        assert_eq!(
            workgroup_count(DEFAULT_FEATURES, LAST_SUPPORTED_AGENTS),
            Some(LAST_SUPPORTED_GROUPS)
        );
        assert_eq!(
            workgroup_count(DEFAULT_FEATURES, LAST_SUPPORTED_AGENTS + 1),
            None
        );
        assert_eq!(workgroup_count(usize::MAX, 1), None);
        assert_eq!(workgroup_count(1, u32::MAX), None);
        assert_eq!(workgroup_count(0, 1), None);
        let oversized_weight_offset = usize::try_from(u32::MAX).unwrap() / ENCODED_DIMENSION + 1;
        assert_eq!(workgroup_count(oversized_weight_offset, 1), None);
    }
}
