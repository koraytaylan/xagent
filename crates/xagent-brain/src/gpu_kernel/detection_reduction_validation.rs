//! Hardware-only exact tree reductions for food and danger detection.
//! Original local scans and their first 256-to-128 merge stay unchanged.
//! The final tree preserves the merged-slot rank for eat ties, independently
//! of the food-index ordering used for bearings. Both arms explicitly compose
//! cooperative whitening and fused prediction before comparing whole cycles.

use std::{error::Error, time::Instant};

use glam::Vec3;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::predictor_fusion::fuse_inline_predictor;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, REFRESH_CYCLES,
};
use super::*;

/// Both kernel entries use starting tick and cooperative-pass limit words.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Compare scheduled refreshes, repeated death and a longer continuation.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature the learned memory before paired full-cycle timing.
const WARMUP_CYCLES: u32 = 256;
/// One hundred cycles amortize recording and completion overhead.
const TIMED_CYCLES: u32 = 100;
/// Alternating arm order across five pairs gives one median per arm.
const TIMING_ROUNDS: usize = 5;
/// The boundary fixture leaves this agent dead without requesting a respawn.
const INACTIVE_AGENT: u32 = 1;
/// Compare all mutable production simulation buffers, including food claims.
const MUTABLE_BUFFERS: usize = 13;
/// Force death initially and on the first scheduled-refresh boundary.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
/// More foods than invocations make food-index and slot-rank orders disagree.
const TIE_FOOD_COUNT: usize = 320;
/// Slot 32 wins the original serial slot scan despite containing food 288.
const EARLIER_SLOT_FOOD: usize = 288;
/// Slot 64 contains a lower food index but must lose the equal-distance eat tie.
const LATER_SLOT_FOOD: usize = 64;
/// Exactly representable opposite offsets produce identical squared distances.
const TIE_FOOD_OFFSET: f32 = 0.25;
/// Normal food height keeps the two bearing candidates symmetric as well.
const FOOD_HEIGHT: f32 = 0.35;
/// Unavailable foods need no grid occupancy and stay outside the local scene.
const UNUSED_FOOD_POSITION: f32 = 40.0;
/// Prevent unavailable filler food from respawning during the one-cycle probe.
const UNAVAILABLE_TIMER: f32 = 1000.0;
/// Full physiological meters avoid a death unrelated to the tie fixture.
const FULL_METER: f32 = 100.0;
/// Normal initial eye height above the flat terrain.
const EYE_HEIGHT: f32 = 1.0;
/// Explicit seeded brain initialization makes the tie fixture repeatable.
const TIE_SEED: u64 = 42;
/// All seven cooperative stages execute in the tie fixture.
const COMPLETE_BRAIN: u32 = 7;
/// Fourth entry in cycle_profile's complete mutable-state snapshot.
const FOOD_FLAGS_STATE_INDEX: usize = 3;
/// The shader stores consumed food as an atomic one.
const FOOD_CONSUMED: u32 = 1;
/// Value used by the shader's biome lookup for harmful ground.
const DANGER_BIOME: u32 = 2;

const DANGER_SCAN: &str = r"    if (tid == 0u) {
        var best_distance = DANGER_SENSE_RADIUS;
        var best_cell = NO_DANGER_CELL;
        for (var i = 0u; i < MEMORY_CAP; i++) {
            if (danger_cell_precedes(s_similarities[i], shared_sort_indices[i],
                                     best_distance, best_cell)) {
                best_distance = s_similarities[i];
                best_cell = shared_sort_indices[i];
            }
        }
";

const FOOD_SCAN: &str = r"    if (tid == 0u && alive) {
        var best_idx = 0xFFFFFFFFu;
        var best_dist_sq = 1e12;
        var best_food_dist_sq = 1e12;
        var bearing_dist_sq = FOOD_SENSE_RADIUS * FOOD_SENSE_RADIUS;
        var best_food_idx = NO_FOOD;
        for (var i = 0u; i < 128u; i++) {
            if (s_similarities[i] < best_dist_sq) {
                best_dist_sq = s_similarities[i];
                best_idx = shared_sort_indices[i];
            }
            best_food_dist_sq = min(best_food_dist_sq, s_food_dist_sq[i]);
            if (food_precedes(s_argmin_val[i], s_argmin_idx[i], bearing_dist_sq, best_food_idx)) {
                bearing_dist_sq = s_argmin_val[i];
                best_food_idx = s_argmin_idx[i];
            }
        }
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct KernelPipelines {
    claim: wgpu::ComputePipeline,
    main: wgpu::ComputePipeline,
}

impl KernelPipelines {
    fn swap_with(&mut self, kernel: &mut GpuKernel) {
        std::mem::swap(&mut kernel.kernel_claim_pipeline, &mut self.claim);
        std::mem::swap(&mut kernel.kernel_pipeline, &mut self.main);
    }
}

fn reduced_kernel_source() -> String {
    const RANK_INITIALIZATION: &str = "        s_argmin_idx[tid] = local_bearing_idx;\n";
    const DANGER_TREE: &str = "    reduce_danger_detection(tid);\n\n    if (tid == 0u) {\n        let best_distance = s_similarities[0u];\n        let best_cell = shared_sort_indices[0u];\n";
    const FOOD_TREE: &str = "    reduce_food_detection(tid);\n\n    if (tid == 0u && alive) {\n        let best_idx = shared_sort_indices[0u];\n        let best_food_dist_sq = s_food_dist_sq[0u];\n        let best_food_idx = s_argmin_idx[0u];\n";
    let original = include_str!("../shaders/kernel/kernel_tick.wgsl");
    for marker in [DANGER_SCAN, FOOD_SCAN, RANK_INITIALIZATION] {
        assert_eq!(original.matches(marker).count(), 1);
    }
    let reduced = original
        .replacen(DANGER_SCAN, DANGER_TREE, 1)
        .replacen(FOOD_SCAN, FOOD_TREE, 1)
        .replacen(
            RANK_INITIALIZATION,
            &format!("{RANK_INITIALIZATION}        s_dense_partials[tid] = f32(tid);\n"),
            1,
        );
    [reduced.as_str(), include_str!("detection_reductions.wgsl")].join("\n")
}

fn make_pipelines(kernel: &GpuKernel, reduced: bool) -> KernelPipelines {
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let entry = if reduced {
        reduced_kernel_source()
    } else {
        include_str!("../shaders/kernel/kernel_tick.wgsl").to_owned()
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
    let label = if reduced {
        "detection_ordered_tree_candidate"
    } else {
        "detection_serial_scan_optimized_brain_baseline"
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

fn install_baseline(kernel: &mut GpuKernel) -> KernelPipelines {
    let baseline = make_pipelines(kernel, false);
    let candidate = make_pipelines(kernel, true);
    kernel.kernel_claim_pipeline = baseline.claim;
    kernel.kernel_pipeline = baseline.main;
    candidate
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn detection_trees_match_complete_optimized_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut candidate = install_baseline(&mut kernel);
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
    let inactive_before = kernel.read_agent_state(INACTIVE_AGENT);
    let deaths_before = kernel.read_full_state_blocking()[P_DEATH_COUNT];
    let mut expected = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        expected.push(capture_state(&kernel)?);
        cycle += cycles;
    }
    assert!(
        kernel.read_full_state_blocking()[P_DEATH_COUNT] >= deaths_before + EXPECTED_FORCED_DEATHS
    );
    restore(&mut kernel, &initial);
    candidate.swap_with(&mut kernel);
    cycle = 0;
    for (cycles, expected) in PARITY_CHUNKS.into_iter().zip(&expected) {
        if cycle == REFRESH_CYCLES {
            force_death(&kernel);
        }
        let tick = cycle * kernel.brain_tick_stride;
        advance(&mut kernel, tick, cycles);
        assert_state_equal(&kernel, expected, &capture_state(&kernel)?);
        cycle += cycles;
    }
    let inactive_after = kernel.read_agent_state(INACTIVE_AGENT);
    assert_eq!(
        bytemuck::cast_slice::<f32, u32>(&inactive_before.brain_state),
        bytemuck::cast_slice::<f32, u32>(&inactive_after.brain_state),
    );
    println!("DETECTION_TREE_PARITY cycles={cycle} exact_buffers={MUTABLE_BUFFERS} death_boundaries=2 dead_agent_unchanged=true");
    Ok(())
}

fn tie_kernel() -> GpuKernel {
    let brain = BrainConfig {
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let world = WorldConfig {
        seed: TIE_SEED,
        ..WorldConfig::default()
    };
    let mut kernel = GpuKernel::new(1, TIE_FOOD_COUNT, &brain, &world);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.probe.skip_global = false;
    kernel.probe.skip_vision = false;
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    kernel.reset_agents_seeded(&brain, TIE_SEED);
    let mut food = vec![(UNUSED_FOOD_POSITION, FOOD_HEIGHT, UNUSED_FOOD_POSITION); TIE_FOOD_COUNT];
    food[EARLIER_SLOT_FOOD] = (-TIE_FOOD_OFFSET, FOOD_HEIGHT, 0.0);
    food[LATER_SLOT_FOOD] = (TIE_FOOD_OFFSET, FOOD_HEIGHT, 0.0);
    let mut consumed = vec![true; TIE_FOOD_COUNT];
    consumed[EARLIER_SLOT_FOOD] = false;
    consumed[LATER_SLOT_FOOD] = false;
    let mut biomes = vec![0; BIOME_GRID_RES * BIOME_GRID_RES];
    // The four central cells are equidistant from the unmoving origin. Their
    // row-major cell ordering independently exercises the danger tie rule.
    let middle = BIOME_GRID_RES / 2;
    for row in [middle - 1, middle] {
        for column in [middle - 1, middle] {
            biomes[row * BIOME_GRID_RES + column] = DANGER_BIOME;
        }
    }
    kernel.upload_world(
        &vec![0.0; TERRAIN_VPS * TERRAIN_VPS],
        &biomes,
        &food,
        &consumed,
        &vec![UNAVAILABLE_TIMER; TIE_FOOD_COUNT],
    );
    kernel.upload_agents(&[(
        Vec3::new(0.0, EYE_HEIGHT, 0.0),
        FULL_METER,
        FULL_METER,
        brain.memory_capacity,
        brain.processing_slots,
    )]);
    kernel.write_motor_decision(0, 0.0, 0.0, 0.0);
    kernel
}

fn captured_food_flag(state: &[Vec<u8>], food: usize) -> u32 {
    let bytes = &state[FOOD_FLAGS_STATE_INDEX];
    let offset = food.checked_mul(std::mem::size_of::<u32>()).unwrap();
    bytemuck::pod_read_unaligned(&bytes[offset..offset + std::mem::size_of::<u32>()])
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn detection_tree_preserves_merged_food_slot_ties() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = tie_kernel();
    let mut candidate = install_baseline(&mut kernel);
    let initial = checkpoint(&kernel);
    advance(&mut kernel, 0, 1);
    let expected = capture_state(&kernel)?;
    assert_eq!(
        captured_food_flag(&expected, EARLIER_SLOT_FOOD),
        FOOD_CONSUMED
    );
    assert_eq!(captured_food_flag(&expected, LATER_SLOT_FOOD), 0);
    let physics = kernel.read_full_state_blocking();
    assert_eq!(physics[P_FOOD_COUNT].to_bits(), 1.0_f32.to_bits());
    assert!(physics[P_NEAREST_DANGER_BEARING] > std::f32::consts::FRAC_PI_2);
    restore(&mut kernel, &initial);
    candidate.swap_with(&mut kernel);
    advance(&mut kernel, 0, 1);
    assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    println!("DETECTION_TREE_TIES foods={TIE_FOOD_COUNT} winner={EARLIER_SLOT_FOOD} losing_lower_food_index={LATER_SLOT_FOOD} exact_buffers={MUTABLE_BUFFERS} danger_ties=true");
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_detection_trees_against_serial_scans() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut candidate = install_baseline(&mut kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut current_arm = 0;
    let mut timings = [Vec::new(), Vec::new()];
    for round in 0..TIMING_ROUNDS {
        let mut states = [None, None];
        for offset in 0..timings.len() {
            let arm = (round + offset) % timings.len();
            restore(&mut kernel, &warm);
            if arm != current_arm {
                candidate.swap_with(&mut kernel);
                current_arm = arm;
            }
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        assert_state_equal(
            &kernel,
            states[0].as_ref().unwrap(),
            states[1].as_ref().unwrap(),
        );
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_ROUNDS / 2];
    let tree = timings[1][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    println!(
        "DETECTION_TREE agents={} ticks={ticks} rounds={TIMING_ROUNDS} brain=cooperative_whitening_fused_predictor baseline_secs={baseline:.9} tree_secs={tree:.9} baseline_tps={:.3} tree_tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
        kernel.agent_count, f64::from(ticks) / baseline, f64::from(ticks) / tree, baseline / tree,
    );
    Ok(())
}
