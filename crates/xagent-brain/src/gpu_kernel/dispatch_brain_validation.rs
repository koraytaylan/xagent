//! Hardware checks for combined-brain, split-serial and masked-dispatch paths.
//! Each mode compares its own original-source schedule against optional brain
//! transforms at death/refresh boundaries. Four predictor lanes require exact
//! state parity; sixteen lanes report FP32 drift without asserting equivalence.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::wider_predictor;
use super::rounding_validation::{assert_inactive_agent_unchanged, compare_rounding_state};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, PARITY_CHUNKS, REFRESH_CYCLES,
};
use super::*;

/// Tick and brain-pass limit occupy two push-constant words.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// The fixture leaves one agent dead without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Initial death and the scheduled-refresh boundary must both execute.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
/// Includes sensory_next, depth, claims and every persistent brain scratch.
const STATE_BUFFERS: usize = 13;
/// The masked public route runs physics, vision and standalone brain together.
const COMPLETE_PHASE_MASK: u32 = 7;
/// The production ordered predictor has four lanes per output row.
const EXACT_PREDICTOR_LANES: u32 = 4;
/// The rounding candidate uses sixteen lanes and a balanced final sum.
const WIDE_PREDICTOR_LANES: u32 = 16;

enum BrainEntry {
    Main,
    Combined,
    Standalone,
}

struct BrainPipelines {
    main: wgpu::ComputePipeline,
    combined: wgpu::ComputePipeline,
    standalone: wgpu::ComputePipeline,
}

fn shader_module(
    kernel: &GpuKernel,
    passes: &str,
    entry: BrainEntry,
    has_subgroup: bool,
) -> wgpu::ShaderModule {
    let common = include_str!("../shaders/kernel/common.wgsl");
    let source = match entry {
        BrainEntry::Combined => {
            let common = with_plain_grid_bindings(common);
            let vision = [
                include_str!("../shaders/kernel/phase_vision.wgsl"),
                include_str!("../shaders/kernel/phase_vision_parallel.wgsl"),
            ]
            .join("\n")
            .replace("sensory_buffer[", "sensory_next[");
            assert!(!vision.contains("sensory_buffer"));
            [
                common.as_str(),
                passes,
                include_str!("../shaders/kernel/brain_inner.wgsl"),
                vision.as_str(),
                include_str!("../shaders/kernel/brain_vision_tick.wgsl"),
            ]
            .join("\n")
        }
        BrainEntry::Main => [
            common,
            passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        BrainEntry::Standalone => [
            common,
            passes,
            include_str!("../shaders/kernel/brain_tick.wgsl"),
        ]
        .join("\n"),
    };
    kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dispatch_brain_parity"),
            source: wgpu::ShaderSource::Wgsl(apply_subgroup_markers(&source, has_subgroup).into()),
        })
}

fn make_variants(
    kernel: &mut GpuKernel,
    has_subgroup: bool,
    predictor_lanes: u32,
) -> (BrainPipelines, BrainPipelines) {
    let original_passes = include_str!("../shaders/kernel/brain_passes.wgsl");
    let candidate_passes = fuse_inline_predictor(&compose_brain_passes(true));
    // Exercise the production prefetch factor through every brain entry point.
    const PREFETCH_FACTOR: u32 = 8;
    let candidate_passes =
        super::dense_prefetch::prefetch_passes(&candidate_passes, PREFETCH_FACTOR);
    let candidate_passes = wider_predictor(&candidate_passes, predictor_lanes);
    let original_main = shader_module(kernel, original_passes, BrainEntry::Main, has_subgroup);
    let original_combined =
        shader_module(kernel, original_passes, BrainEntry::Combined, has_subgroup);
    let original_standalone = shader_module(
        kernel,
        original_passes,
        BrainEntry::Standalone,
        has_subgroup,
    );
    let candidate_main = shader_module(kernel, &candidate_passes, BrainEntry::Main, has_subgroup);
    let candidate_combined = shader_module(
        kernel,
        &candidate_passes,
        BrainEntry::Combined,
        has_subgroup,
    );
    let candidate_standalone = shader_module(
        kernel,
        &candidate_passes,
        BrainEntry::Standalone,
        has_subgroup,
    );
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dispatch_brain_parity"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    // The standalone brain entry has no push constants, as in production.
    let standalone_layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dispatch_brain_standalone_parity"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
    // Serial ray overrides select the original combined dispatch's shape.
    // Optional standalone vision flags cannot affect these explicit modules.
    let constants = vision_override_constants(&kernel.layout);
    let create = |module: &wgpu::ShaderModule, entry| {
        kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(if entry == "brain_tick" {
                    &standalone_layout
                } else {
                    &layout
                }),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
    };
    let claim = create(&original_main, "kernel_claim_tick");
    let publish = create(&original_combined, "sensory_publish");
    let original = BrainPipelines {
        main: create(&original_main, "kernel_tick"),
        combined: create(&original_combined, "brain_vision_tick"),
        standalone: create(&original_standalone, "brain_tick"),
    };
    let candidate = BrainPipelines {
        main: create(&candidate_main, "kernel_tick"),
        combined: create(&candidate_combined, "brain_vision_tick"),
        standalone: create(&candidate_standalone, "brain_tick"),
    };
    kernel.kernel_claim_pipeline = claim;
    kernel.sensory_publish_pipeline = publish;
    (original, candidate)
}

fn install(kernel: &mut GpuKernel, pipelines: &BrainPipelines) {
    kernel.kernel_pipeline = pipelines.main.clone();
    kernel.brain_vision_pipeline = pipelines.combined.clone();
    kernel.brain_pipeline = pipelines.standalone.clone();
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32, masked: bool) {
    if cycle == REFRESH_CYCLES {
        force_death(kernel);
    }
    let tick = cycle * kernel.brain_tick_stride;
    let ticks = cycles * kernel.brain_tick_stride;
    if masked {
        kernel.dispatch_batch_masked(u64::from(tick), ticks, COMPLETE_PHASE_MASK);
    } else {
        kernel.dispatch_ticks(u64::from(tick), ticks);
    }
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn optional_brain_preserves_combined_and_split_dispatches() -> Result<(), Box<dyn Error>> {
    check_dispatches(EXACT_PREDICTOR_LANES)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn wider_predictor_reports_rounding_across_dispatch_routes() -> Result<(), Box<dyn Error>> {
    check_dispatches(WIDE_PREDICTOR_LANES)
}

fn check_dispatches(predictor_lanes: u32) -> Result<(), Box<dyn Error>> {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    // Ensure the combined case actually dispatches brain_vision_tick even if
    // the process enabled standalone object/scent variants for other tests.
    kernel.standalone_vision_required = false;
    let rays = kernel.layout.vision_width * kernel.layout.vision_height;
    kernel.brain_vision_workgroups =
        kernel.agent_count * (1 + rays.div_ceil(BRAIN_WORKGROUP_THREADS));
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let initial_state = capture_state(&kernel)?;
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);

    let subgroup_modes = if kernel.has_subgroup {
        vec![true, false]
    } else {
        vec![false]
    };
    for has_subgroup in subgroup_modes {
        let (original, candidate) = make_variants(&mut kernel, has_subgroup, predictor_lanes);
        for (label, mode, beside, masked) in [
            ("combined", BrainExecutionMode::FusedSerial, true, false),
            (
                "split_serial",
                BrainExecutionMode::SplitSerial,
                false,
                false,
            ),
            (
                "masked_standalone",
                BrainExecutionMode::FusedSerial,
                false,
                true,
            ),
        ] {
            kernel.set_execution_mode(mode);
            kernel.set_brain_beside_vision(beside);
            assert_eq!(kernel.probe.brain_beside_vision, beside);
            assert_eq!(kernel.vision_stride, 1);
            assert!(!kernel.probe.skip_vision && !kernel.probe.skip_global);
            restore(&mut kernel, &initial);
            install(&mut kernel, &original);
            let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
            let mut cycle = 0;
            for cycles in PARITY_CHUNKS {
                advance(&mut kernel, cycle, cycles, masked);
                cycle += cycles;
                let state = capture_state(&kernel)?;
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    &state,
                    INACTIVE_AGENT,
                    &format!(
                        "dispatch_reference mode={label} subgroup={has_subgroup} cycles={cycle}"
                    ),
                );
                expected.push(state);
            }
            assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);

            restore(&mut kernel, &initial);
            install(&mut kernel, &candidate);
            cycle = 0;
            for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
                advance(&mut kernel, cycle, cycles, masked);
                cycle += cycles;
                let actual = capture_state(&kernel)?;
                let checkpoint_label = format!(
                    "dispatch_predictor_width mode={label} subgroup={has_subgroup} predictor_lanes={predictor_lanes} cycles={cycle}"
                );
                if predictor_lanes == EXACT_PREDICTOR_LANES {
                    assert_state_equal(&kernel, expected, &actual);
                } else {
                    compare_rounding_state(&kernel, expected, &actual, &checkpoint_label);
                }
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial_state,
                    &actual,
                    INACTIVE_AGENT,
                    &checkpoint_label,
                );
            }
            assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);
            let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
            assert_eq!(
                bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
                bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
            );
            if predictor_lanes == EXACT_PREDICTOR_LANES {
                println!("BRAIN_DISPATCH_PARITY mode={label} subgroup={has_subgroup} cycles={cycle} exact_buffers={STATE_BUFFERS} original_source_oracle=true death_boundary=true");
            } else {
                println!("BRAIN_DISPATCH_ROUNDING mode={label} subgroup={has_subgroup} predictor_lanes={predictor_lanes} cycles={cycle} compared_buffers={STATE_BUFFERS} original_source_oracle=true inactive_brain_patterns_exact=true death_boundary=true equivalence=not_asserted");
            }
        }
    }
    Ok(())
}
