//! Test-only 128-invocation main with the complete packed production schedule.
//! Claim, global credit/world, and vision retain their 256-thread pipelines.
//! The predictor keeps its sixteen-lane addition tree, and each reinforcement
//! thread computes both original stride-two chains independently. Context and
//! predictor output tiles shrink; packed encoder ownership stays unchanged.
//!
//! Shared allocations stay unchanged, including the 512-word packed encoder
//! scratch. The cortex normalization explicitly retains its original logical
//! 256-lane tree. This measures scheduling/occupancy against the extra tiles
//! without changing storage authority, dispatch count, or host tick logic.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::main_width::MAIN_THREADS;
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::upload_random_scene;
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// Default and odd raw fields exercise feature and final-credit tails.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Endpoints straddle death, whitening, and chunk boundaries through cycle100.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Mature credit opportunities match the current packed/store-skip baseline.
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const REPLAYS: usize = 2;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
/// Keep cortical driver submissions bounded while including inactive/dead/live.
const CORTEX_AGENTS: u32 = 3;
const CORTEX_CYCLES: u32 = 3;
const FOOD_ITEMS: usize = 104;
const CORTEX_SEED: u64 = 42;
const COMPLETE_BRAIN: u32 = 7;
const PUSH_CONSTANT_BYTES: u32 = 8;
const RAYS_PER_GROUP: u32 = BRAIN_WORKGROUP_THREADS / PARALLEL_VISION_LANES;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

fn pipeline(
    kernel: &GpuKernel,
    source: &str,
    entry: &str,
    constants: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_width_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_width_probe"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("main_width_probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: Option<global_credit::Pipelines>,
    candidate: bool,
    expected_threads: u32,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        Self::with_passes(kernel, &optimized_brain(), MAIN_THREADS)
    }

    fn with_passes(kernel: &GpuKernel, passes: &str, expected_threads: u32) -> Self {
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            BRAIN_WORKGROUP_THREADS
        );
        eprintln!(
            "MAIN128_COMPILE_BEGIN width={} height={} cortex={} expected_threads={expected_threads}",
            kernel.layout.vision_width,
            kernel.layout.vision_height,
            kernel.layout.visual_cortex_enabled
        );
        let parked = global_credit::Pipelines::new_packed_with_main128(
            kernel,
            passes,
            &constants(kernel),
            true,
        )
        .unwrap();
        assert_eq!(parked.main_threads, expected_threads);
        assert!(parked.packed_encoder.is_some());
        eprintln!("MAIN128_COMPILE_END");
        Self {
            parked: Some(parked),
            candidate: false,
            expected_threads,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            // Checkpoint restores cannot reach a parked cache. Invalidate it
            // before activation so the public authoritative matrix is imported.
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
        assert!(kernel.global_credit_active());
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            if candidate {
                self.expected_threads
            } else {
                BRAIN_WORKGROUP_THREADS
            }
        );
    }
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
        cycle += count;
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; exact full-state and private cache comparison"]
fn main128_preserves_raw_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, true, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for replay in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "main128 raw",
                );
            }
            println!("MAIN128_PARITY width={width} height={height} cortex=false cycles=100 replay={replay} exact_buffers={MUTABLE_BUFFERS} death_refresh=true private_mirror_exact=true no_export=true main_threads={MAIN_THREADS} other_threads={BRAIN_WORKGROUP_THREADS}");
        }
    }
    Ok(())
}

fn prepare_cortex() -> GpuKernel {
    let (width, height) = FIELDS[0];
    let brain = BrainConfig {
        vision_width: width,
        vision_height: height,
        visual_cortex_enabled: true,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut kernel = GpuKernel::new(CORTEX_AGENTS, FOOD_ITEMS, &brain, &WorldConfig::default());
    kernel.reset_agents_seeded(&brain, CORTEX_SEED);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.set_probe_pass_skips(false, false);
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    let mut constants = constants(&kernel);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_store_suppression(
        &kernel,
        &optimized_brain(),
        &constants,
        true,
    );
    assert!(kernel.global_credit_active());
    assert!(!cache(&kernel).is_valid());
    // Explicit claim and standalone vision keep these arms independent of
    // ambient constructor options. Only the main pipeline is ever swapped.
    let source = global_credit::main_source(
        &optimized_brain(),
        kernel.has_subgroup,
        Some(cache(&kernel)),
    );
    kernel.kernel_claim_pipeline = pipeline(&kernel, &source, "kernel_claim_tick", &constants);
    for name in [
        "VISION_PARALLEL_STEPS",
        "VISION_OBJECT_QUERIES",
        "VISION_PARALLEL_SCENT",
    ] {
        constants.insert(name.into(), 1.0);
    }
    constants.insert(
        "VISION_RAYS_PER_WORKGROUP".into(),
        f64::from(RAYS_PER_GROUP),
    );
    let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
    let source = compose_vision_source(&common, true, true);
    kernel.vision_pipeline = pipeline(&kernel, &source, "vision_tick", &constants);
    kernel.vision_workgroups = kernel.agent_count * (width * height).div_ceil(RAYS_PER_GROUP);
    kernel.standalone_vision_required = true;
    upload_random_scene(&kernel, 0, true, false);
    prepare_boundary_scene(&kernel);
    let tick =
        fixed_tail_base(kernel.layout.brain_stride) + O_TICK_COUNT - O_PREDICTOR_CONTEXT_WEIGHT;
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.brain_state[tick] = f32::from(u16::try_from(REFRESH_CYCLES - 1).unwrap());
        kernel.write_agent_state(agent, &state);
    }
    kernel
}

fn cortex_trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    for cycle in 0..CORTEX_CYCLES {
        if cycle == 1 {
            force_death(kernel);
        }
        advance(kernel, cycle, 1);
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= EXPECTED_FORCED_DEATHS);
    Ok(states)
}

#[test]
#[ignore = "requires a GPU; bounded cortex parity with one cycle per submission"]
fn main128_preserves_bounded_cortex_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_cortex();
    let mut arms = Arms::new(&kernel);
    let initial = capture_state(&kernel)?;
    let saved = checkpoint(&kernel);
    let expected = cortex_trajectory(&mut kernel)?;
    for replay in 0..REPLAYS {
        restore(&mut kernel, &saved);
        arms.activate(&mut kernel, true);
        let actual = cortex_trajectory(&mut kernel)?;
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, expected, actual);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial,
                actual,
                INACTIVE_AGENT,
                "main128 cortex",
            );
        }
        println!("MAIN128_PARITY width={} height={} cortex=true cycles={CORTEX_CYCLES} agents={CORTEX_AGENTS} replay={replay} exact_buffers={MUTABLE_BUFFERS} death_tick_boundary=true private_mirror_exact=true no_export=true", kernel.layout.vision_width, kernel.layout.vision_height);
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_main128_complete_cycles() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare_kernel_with_store_suppression(width, height, false, true);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    assert_mirror(&kernel, &expected)?;
    restore(&mut kernel, &saved);
    arms.activate(&mut kernel, true);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let preflight = capture_state(&kernel)?;
    assert_state_equal(&kernel, &expected, &preflight);
    assert_mirror(&kernel, &preflight)?;
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            assert_state_equal(&kernel, &expected, &actual);
            assert_mirror(&kernel, &actual)?;
        }
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("MAIN128_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_production_recorder=true cold_import_timed=true state_comparison_timed=false shared_storage_unchanged=true main_threads={MAIN_THREADS} other_threads={BRAIN_WORKGROUP_THREADS}", reference / candidate);
    Ok(())
}

#[test]
#[ignore = "requires a GPU; explicit unsupported source must preserve the packed fallback"]
fn main128_unsupported_brain_uses_original_main() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare_kernel_with_store_suppression(width, height, true, true);
    // Four-lane fused/prefetched predictor and no context gather deliberately
    // violate two independent conditions of the compact-main source contract.
    const PREFETCH_FACTOR: u32 = 8;
    let fused = predictor_fusion::fuse_inline_predictor(&compose_brain_passes(true));
    let passes = dense_prefetch::prefetch_passes(&fused, PREFETCH_FACTOR);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_store_suppression(
        &kernel,
        &passes,
        &constants(&kernel),
        true,
    );
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        BRAIN_WORKGROUP_THREADS
    );
    let mut arms = Arms::with_passes(&kernel, &passes, BRAIN_WORKGROUP_THREADS);
    let initial = capture_state(&kernel)?;
    let saved = checkpoint(&kernel);
    let expected = cortex_trajectory(&mut kernel)?;
    for replay in 0..REPLAYS {
        restore(&mut kernel, &saved);
        arms.activate(&mut kernel, true);
        let actual = cortex_trajectory(&mut kernel)?;
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, expected, actual);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial,
                actual,
                INACTIVE_AGENT,
                "main128 unsupported source",
            );
        }
        println!("MAIN128_FALLBACK predictor_lanes=4 context_gather=false main_threads={BRAIN_WORKGROUP_THREADS} cycles={CORTEX_CYCLES} replay={replay} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true");
    }
    Ok(())
}
