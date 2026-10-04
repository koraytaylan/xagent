//! Hardware-only FP32 subgroup reductions for the cooperative action/learning
//! tail. Both arms explicitly use cooperative whitening and fused prediction.
//! Dense encoder/predictor arithmetic is unchanged. Native subgroup sums may
//! round differently, so comparisons report numerical and behavioral drift
//! while requiring finite valid state and unchanged inactive brain/pattern state.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::{
    assert_inactive_agent_unchanged, compare_rounding_state, extract_behavior,
    report_seeded_behavior,
};
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// The kernel accepts its starting tick and cooperative-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Boundaries include immediate error, refresh/death events and 1,000 cycles.
const CONTINUATION_CHUNKS: [u32; 8] = [1, 18, 1, 1, 19, 1, 59, 900];
/// Independent brain seeds retain the same seeded flat-world geometry.
const BRAIN_SEEDS: [u64; 3] = [42, 314, 2026];
/// Full episodic memory before the paired throughput measurement.
const WARMUP_CYCLES: u32 = 256;
/// One hundred full cycles amortize recording and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Rotate which arm runs first across five paired trials.
const TIMING_ROUNDS: usize = 5;
/// This fixture agent remains inactive without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Source assertions keep the reduction experiment's scope explicit.
const REDUCTION_CALLS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct KernelPipelines {
    claim: wgpu::ComputePipeline,
    main: wgpu::ComputePipeline,
}

fn candidate_sources(passes: &str) -> (String, String) {
    const DEFINITION: &str = "fn wg_reduce_dense(tid: u32) {";
    const SUBGROUP_MARKER: &str = "    // KERNEL_SUBGROUP_ENTRY_PARAMS\n";
    const MAIN_START: &str = "    let tid = lid.x;\n    let base_tick = kpc.start_tick;\n";
    assert_eq!(passes.matches(DEFINITION).count(), 1);
    assert_eq!(
        passes.matches("wg_reduce_dense(tid);").count(),
        REDUCTION_CALLS
    );
    assert!(passes.contains("var<workgroup> s_dense_partials: array<f32, BRAIN_WORKGROUP_SIZE>;"));
    assert!(passes.contains("const BRAIN_WORKGROUP_SIZE: u32 = 256u;"));
    let renamed = passes.replacen(DEFINITION, "fn tree_reduce_dense_reference(tid: u32) {", 1);
    let passes = [renamed.as_str(), include_str!("subgroup_reductions.wgsl")].join("\n");
    let kernel = include_str!("../shaders/kernel/kernel_tick.wgsl");
    assert_eq!(kernel.matches(SUBGROUP_MARKER).count(), 1);
    assert_eq!(kernel.matches(MAIN_START).count(), 1);
    let kernel = kernel
        .replacen(
            SUBGROUP_MARKER,
            &format!(
                "    @builtin(subgroup_id) reduction_group: u32,\n    @builtin(num_subgroups) reduction_groups: u32,\n{SUBGROUP_MARKER}"
            ),
            1,
        )
        .replacen(
            MAIN_START,
            &format!("{MAIN_START}    reduction_subgroup = vec3<u32>(sgid, reduction_group, reduction_groups);\n"),
            1,
        );
    (passes, kernel)
}

fn make_pipelines(kernel: &GpuKernel, subgroup: bool) -> KernelPipelines {
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let (passes, entry) = if subgroup {
        candidate_sources(&passes)
    } else {
        (
            passes,
            include_str!("../shaders/kernel/kernel_tick.wgsl").to_owned(),
        )
    };
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            passes.as_str(),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            entry.as_str(),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let label = if subgroup {
        "fp32_subgroup_sum_candidate"
    } else {
        "fp32_tree_sum_optimized_baseline"
    };
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
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
                label: Some(&format!("{label}_{entry}")),
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
    KernelPipelines {
        claim: create("kernel_claim_tick"),
        main: create("kernel_tick"),
    }
}

fn prepare() -> TestResult<(GpuKernel, wgpu::ComputePipeline)> {
    let mut kernel = prepare_kernel();
    if !kernel.has_subgroup {
        return Err("Subgroup sum experiment requires supported native subgroups".into());
    }
    // has_subgroup guarantees at least the supported bitonic subgroup width;
    // its maximum subgroup count fits the unused upper half of dense scratch.
    let dense_inputs = u32::try_from(ENCODED_DIMENSION)?;
    assert!(
        BRAIN_WORKGROUP_THREADS / MIN_SUBGROUP_WIDTH_FOR_BITONIC
            <= BRAIN_WORKGROUP_THREADS - dense_inputs
    );
    let baseline = make_pipelines(&kernel, false);
    let candidate = make_pipelines(&kernel, true);
    kernel.kernel_claim_pipeline = baseline.claim;
    kernel.kernel_pipeline = baseline.main;
    Ok((kernel, candidate.main))
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU with subgroups; run explicitly with --ignored --nocapture"]
fn subgroup_sums_report_rounding_and_behavior() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut behavior_pairs = Vec::new();
    for seed in BRAIN_SEEDS {
        let (mut kernel, mut alternate_pipeline) = prepare()?;
        let brain = BrainConfig {
            vision_stride: 1,
            ..BrainConfig::default()
        };
        kernel.reset_agents_seeded(&brain, seed);
        prepare_boundary_scene(&kernel);
        let initial_state = capture_state(&kernel)?;
        let initial = checkpoint(&kernel);
        let mut expected = Vec::with_capacity(CONTINUATION_CHUNKS.len());
        let mut cycle = 0;
        for cycles in CONTINUATION_CHUNKS {
            if cycle == REFRESH_CYCLES {
                force_death(&kernel);
            }
            let tick = cycle * kernel.brain_tick_stride;
            advance(&mut kernel, tick, cycles);
            expected.push(capture_state(&kernel)?);
            cycle += cycles;
        }
        restore(&mut kernel, &initial);
        std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate_pipeline);
        cycle = 0;
        for (cycles, expected) in CONTINUATION_CHUNKS.into_iter().zip(&expected) {
            if cycle == REFRESH_CYCLES {
                force_death(&kernel);
            }
            let tick = cycle * kernel.brain_tick_stride;
            advance(&mut kernel, tick, cycles);
            cycle += cycles;
            let actual = capture_state(&kernel)?;
            let label = format!("subgroup_sum brain_seed={seed} cycles={cycle}");
            let _metrics = compare_rounding_state(&kernel, expected, &actual, &label);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial_state,
                &actual,
                INACTIVE_AGENT,
                &label,
            );
            if cycle == CONTINUATION_CHUNKS.iter().sum::<u32>() {
                behavior_pairs.push((
                    seed,
                    extract_behavior(&kernel, expected),
                    extract_behavior(&kernel, &actual),
                ));
            }
        }
    }
    report_seeded_behavior(&behavior_pairs, "subgroup_sum_flat_world_brain_seeds");
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_subgroup_sums_against_optimized_tree() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut alternate_pipeline) = prepare()?;
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut current_arm = 0;
    let mut timings = [Vec::new(), Vec::new()];
    let mut repeat_reference: [Option<Vec<Vec<u8>>>; 2] = [None, None];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            if arm != current_arm {
                std::mem::swap(&mut kernel.kernel_pipeline, &mut alternate_pipeline);
                current_arm = arm;
            }
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            let state = capture_state(&kernel)?;
            if let Some(reference) = &repeat_reference[arm] {
                assert_state_equal(&kernel, reference, &state);
            } else {
                repeat_reference[arm] = Some(state.clone());
            }
            states[arm] = Some(state);
        }
        let label = format!("subgroup_sum_timing round={round} cycles={TIMED_CYCLES}");
        let _metrics = compare_rounding_state(
            &kernel,
            states[0].as_ref().unwrap(),
            states[1].as_ref().unwrap(),
            &label,
        );
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_ROUNDS / 2];
    let subgroup = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "SUBGROUP_SUM agents={} ticks={ticks} rounds={TIMING_ROUNDS} baseline=cooperative_whitening_fused_predictor baseline_secs={baseline:.9} subgroup_secs={subgroup:.9} baseline_tps={:.3} subgroup_tps={:.3} speedup={:.3} repeat_state_exact=true cross_arm_rounding_reported=true",
        kernel.agent_count, f64::from(ticks) / baseline, f64::from(ticks) / subgroup, baseline / subgroup,
    );
    Ok(())
}
