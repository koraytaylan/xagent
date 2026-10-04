//! Hardware-only whole-state parity for the combined production optimizations.
//! Each layout uses one kernel and one checkpoint. The production claim, main
//! and vision pipelines are compared with explicitly compiled untouched brain
//! entries and pure serial vision; both use the separate-brain schedule.
//! Enable all five flags in REQUIRED_FLAGS before running the ignored test.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::vision_validation::{make_kernel, pure_serial_pipeline, upload_random_scene};
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// Production chooses the tested object cache for this default population.
const AGENTS: u32 = 10;
/// Default-sized food population fits the complete cooperative object cache.
const FOOD_ITEMS: usize = 104;
/// Default and odd fields exercise different feature strides and partial ray groups.
const VISION_FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Compare refresh/death boundaries and a longer evolving-state checkpoint.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// All seven cooperative brain passes are part of the serial reference.
const COMPLETE_BRAIN: u32 = 7;
/// Kernel push constants carry the starting tick and complete-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Production object queries place eight rays in one 256-thread workgroup.
const OBJECT_RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;
/// Nearby agents exercise collisions and registered-agent masks after movement.
const AGENT_SEPARATION: f32 = 1.5;
/// Every mutable storage buffer, including flags, scratch, masks and depth.
const MUTABLE_BUFFERS: usize = 13;
/// The fixture forces death initially and on a scheduled whitening refresh.
const FORCED_DEATHS: f32 = 2.0;
/// The fixture leaves this agent dead without a pending respawn.
const INACTIVE_AGENT: u32 = 1;
/// Require real production construction without mutating process environment.
const REQUIRED_FLAGS: [&str; 5] = [
    "XAGENT_BRAIN_COOPERATIVE_WHITENING",
    "XAGENT_BRAIN_FUSED_PREDICTOR",
    "XAGENT_VISION_OBJECT_QUERIES",
    "XAGENT_VISION_PARALLEL_SCENT",
    "XAGENT_VISION_AGENT_MASKS",
];

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct CyclePipelines {
    claim: wgpu::ComputePipeline,
    main: wgpu::ComputePipeline,
    vision: wgpu::ComputePipeline,
    vision_workgroups: u32,
}

impl CyclePipelines {
    fn swap_with(&mut self, kernel: &mut GpuKernel) {
        std::mem::swap(&mut self.claim, &mut kernel.kernel_claim_pipeline);
        std::mem::swap(&mut self.main, &mut kernel.kernel_pipeline);
        std::mem::swap(&mut self.vision, &mut kernel.vision_pipeline);
        std::mem::swap(&mut self.vision_workgroups, &mut kernel.vision_workgroups);
    }
}

fn make_serial_oracle(kernel: &GpuKernel) -> CyclePipelines {
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            include_str!("../shaders/kernel/brain_passes.wgsl"),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("combined_untouched_brain_oracle"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("combined_untouched_brain_oracle_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let constants = vision_override_constants(&kernel.layout);
    let create = |entry| {
        kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("combined_untouched_oracle_{entry}")),
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
    let (vision, vision_workgroups) = pure_serial_pipeline(kernel);
    CyclePipelines {
        claim: create("kernel_claim_tick"),
        main: create("kernel_tick"),
        vision,
        vision_workgroups,
    }
}

fn prepare_optimized_kernel(width: u32, height: u32) -> GpuKernel {
    let mut kernel = make_kernel(width, height, AGENTS, FOOD_ITEMS);
    // This comparison swaps the retained main pipeline with an explicit oracle.
    kernel.global_credit = None;
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    assert_eq!(kernel.vision_stride, 1);
    assert!(kernel.standalone_vision_required);
    assert_eq!(
        kernel.vision_workgroups,
        AGENTS * (width * height).div_ceil(OBJECT_RAYS_PER_GROUP)
    );
    upload_random_scene(&kernel, 0, true, false);
    for agent in 0..AGENTS {
        kernel.write_agent_physics_fields(
            agent,
            &[(P_POS_X, agent as f32 * AGENT_SEPARATION), (P_POS_Z, 0.0)],
        );
    }
    prepare_boundary_scene(&kernel);
    kernel
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    if cycle == REFRESH_CYCLES {
        force_death(kernel);
    }
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

fn compare_layout(width: u32, height: u32) -> TestResult {
    let mut kernel = prepare_optimized_kernel(width, height);
    let candidate_workgroups = kernel.vision_workgroups;
    let mut alternate = make_serial_oracle(&kernel);
    let reference_workgroups = alternate.vision_workgroups;
    assert_ne!(candidate_workgroups, reference_workgroups);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
    // Capture all production candidate pipelines and their workgroup count
    // together before executing any reference cycle.
    alternate.swap_with(&mut kernel);
    assert_eq!(alternate.vision_workgroups, candidate_workgroups);
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        advance(&mut kernel, cycle, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    restore(&mut kernel, &initial);
    alternate.swap_with(&mut kernel);
    assert_eq!(kernel.vision_workgroups, candidate_workgroups);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        advance(&mut kernel, cycle, cycles);
        assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state)
    );
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
    println!(
        "COMBINED_FULL_STATE_PARITY width={width} height={height} agents={AGENTS} cycles={cycle} exact_buffers={MUTABLE_BUFFERS} candidate_vision_groups={candidate_workgroups} reference_vision_groups={reference_workgroups} death_refresh=true"
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU and all five REQUIRED_FLAGS=1; run explicitly with --ignored --nocapture"]
fn combined_brain_and_vision_match_all_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for flag in REQUIRED_FLAGS {
        assert_eq!(
            std::env::var(flag).as_deref(),
            Ok("1"),
            "set {flag}=1 before running this test"
        );
    }
    for (width, height) in VISION_FIELDS {
        compare_layout(width, height)?;
    }
    Ok(())
}
