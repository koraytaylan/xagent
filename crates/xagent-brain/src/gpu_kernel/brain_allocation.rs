//! Compile-only register-allocation diagnostics for the seven brain stages.
//! Each entry keeps the production cooperative function and layout overrides.
//! Runtime storage supplies cross-stage inputs, and storage sinks retain
//! shared-memory outputs. These entries are never dispatched: they isolate
//! compiler allocation, not stage timing or simulation correctness.

use std::{error::Error, io::Write};

use super::*;

/// The seven logical stages follow the production brain entry's boundaries.
const STAGES: [(&str, &str); 7] = [
    (
        "features",
        r"
    coop_feature_extract(agent_id, tid);
    workgroupBarrier();
    coop_visual_cortex(agent_id, tid);
    workgroupBarrier();
    coop_sensory_adapt(agent_id, tid);
    workgroupBarrier();
    for (var index = tid; index < FEATURE_COUNT; index += BRAIN_WORKGROUP_SIZE) {
        brain_scratch[scratch_base + SCRATCH_FEATURES + index] = s_features[index];
    }
",
    ),
    (
        "encode",
        r"
    allocation_load_features(agent_id, tid);
    workgroupBarrier();
    coop_encode(agent_id, tid);
    workgroupBarrier();
    if (tid < ENCODED_DIMENSION) {
        brain_scratch[scratch_base + SCRATCH_ENCODED + tid] = s_encoded[tid];
    }
",
    ),
    (
        "homeostasis",
        r"
    if (tid < ENCODED_DIMENSION) {
        s_encoded[tid] = brain_scratch[scratch_base + SCRATCH_ENCODED + tid];
    }
    workgroupBarrier();
    coop_habituate_homeo(agent_id, tid);
    workgroupBarrier();
    if (tid < ENCODED_DIMENSION) {
        brain_scratch[scratch_base + SCRATCH_HABITUATED + tid] = s_memory_key[tid];
    }
    if (tid < SCRATCH_RECALL - SCRATCH_HOMEO) {
        brain_scratch[scratch_base + SCRATCH_HOMEO + tid] = s_homeo[tid];
    }
",
    ),
    (
        "recall_score",
        r"
    if (tid < ENCODED_DIMENSION) {
        s_memory_key[tid] = brain_scratch[scratch_base + SCRATCH_HABITUATED + tid];
    }
    workgroupBarrier();
    coop_recall_score(agent_id, tid);
    workgroupBarrier();
    if (tid < MEMORY_CAP) {
        pattern_buffer[agent_id * PATTERN_STRIDE + O_PAT_REINF + tid] = s_similarities[tid];
    }
",
    ),
    (
        "recall_topk",
        r"
    if (tid < MEMORY_CAP) {
        s_similarities[tid] = pattern_buffer[agent_id * PATTERN_STRIDE + O_PAT_REINF + tid];
    }
    workgroupBarrier();
    coop_recall_topk(agent_id, tid /* SUBGROUP_TOPK_ARGS */);
    workgroupBarrier();
    if (tid < RECALL_IDX_STRIDE) {
        brain_scratch[scratch_base + SCRATCH_RECALL + tid] = s_recall[tid];
    }
    if (tid < MEMORY_CAP) {
        pattern_buffer[agent_id * PATTERN_STRIDE + O_PAT_REINF + tid] = s_similarities[tid];
    }
",
    ),
    (
        "predict_and_act",
        r"
    allocation_load_features(agent_id, tid);
    allocation_load_encoding(agent_id, tid);
    if (tid < SCRATCH_RECALL - SCRATCH_HOMEO) {
        s_homeo[tid] = brain_scratch[scratch_base + SCRATCH_HOMEO + tid];
    }
    if (tid < RECALL_IDX_STRIDE) {
        s_recall[tid] = brain_scratch[scratch_base + SCRATCH_RECALL + tid];
    }
    if (tid < RECALL_K) {
        s_similarities[tid] = brain_scratch[scratch_base + SCRATCH_RECALL_SIMILARITY + tid];
    }
    workgroupBarrier();
    coop_predict_and_act(agent_id, tid, false);
    workgroupBarrier();
    if (tid == 0u) {
        brain_scratch[scratch_base + SCRATCH_SCALARS] = s_pred_td[S_REWARD];
    }
",
    ),
    (
        "learn_and_store",
        r"
    allocation_load_features(agent_id, tid);
    allocation_load_encoding(agent_id, tid);
    if (tid == 0u) {
        s_pred_td[S_PRED_ERROR] = physics_state[agent_id * PHYS_STRIDE + P_PREDICTION_ERROR];
        s_pred_td[S_REWARD] = brain_scratch[scratch_base + SCRATCH_SCALARS];
    }
    workgroupBarrier();
    coop_learn_and_store(agent_id, tid, true);
",
    ),
];

/// Only cross-stage inputs are loaded; reduction scratch remains owned by
/// each unmodified cooperative function. The synthetic storage mapping is
/// intentionally not a runnable schedule between these independent entries.
const INPUT_HELPERS: &str = r"
fn allocation_load_features(agent_id: u32, tid: u32) {
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    for (var index = tid; index < FEATURE_COUNT; index += BRAIN_WORKGROUP_SIZE) {
        s_features[index] = brain_scratch[scratch_base + SCRATCH_FEATURES + index];
    }
}

fn allocation_load_encoding(agent_id: u32, tid: u32) {
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    if (tid < ENCODED_DIMENSION) {
        s_encoded[tid] = brain_scratch[scratch_base + SCRATCH_ENCODED + tid];
        s_memory_key[tid] = brain_scratch[scratch_base + SCRATCH_HABITUATED + tid];
    }
}
";

fn stage_source(stage: &str, body: &str, has_subgroup: bool) -> String {
    let entry = format!(
        r"
@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn allocation_{stage}(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
    // SUBGROUP_ENTRY_PARAMS
) {{
    let agent_id = wgid.x;
    let tid = lid.x;
    let scratch_base = agent_id * BRAIN_SCRATCH_STRIDE;
    {body}
}}
"
    );
    apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            include_str!("../shaders/kernel/brain_passes.wgsl"),
            INPUT_HELPERS,
            &entry,
        ]
        .join("\n"),
        has_subgroup,
    )
}

fn stage_marker(boundary: &str, stage: &str) -> std::io::Result<()> {
    let mut stderr = std::io::stderr().lock();
    writeln!(stderr, "BRAIN_ALLOCATION {boundary} stage={stage}")?;
    stderr.flush()
}

#[test]
#[ignore = "hardware compiler diagnostic; run with RADV_DEBUG=shaderstats --ignored --nocapture"]
fn diagnose_brain_stage_register_allocation() -> Result<(), Box<dyn Error>> {
    let _vulkan = vulkan_gate::enter();
    stage_marker("BEGIN", "production_initialization")?;
    let kernel = GpuKernel::new(1, 1, &BrainConfig::default(), &WorldConfig::default());
    stage_marker("END", "production_initialization")?;
    eprintln!(
        "BRAIN_ALLOCATION_CONFIG features={} cortex={} subgroup={} dispatches=0",
        kernel.layout.feature_count, kernel.layout.visual_cortex_enabled, kernel.has_subgroup
    );
    let bind_layout = kernel.brain_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain_allocation_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
    let constants = vision_override_constants(&kernel.layout);
    for (stage, body) in STAGES {
        let entry = format!("allocation_{stage}");
        stage_marker("BEGIN", stage)?;
        let module = kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&entry),
                source: wgpu::ShaderSource::Wgsl(
                    stage_source(stage, body, kernel.has_subgroup).into(),
                ),
            });
        let _pipeline = kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });
        stage_marker("END", stage)?;
    }
    Ok(())
}
