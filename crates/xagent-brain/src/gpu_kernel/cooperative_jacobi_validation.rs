//! Hardware-only cooperative Jacobi rotations. All arms retain the explicit
//! production packed encoder, store suppression, context gather and dense
//! predictor configuration. Only the whitening refresh function changes.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore, Checkpoint};
use super::global_credit::Pipelines;
use super::packed_store_validation::{
    advance, assert_mirror, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Include the default grid and an odd grid with a different brain layout.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Check immediately around death/refresh boundaries, then continue to 100.
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Independent candidate replays expose unpublished workgroup scratch.
const REPLAYS: usize = 2;
/// Populate the episodic memory before all latency measurements.
const WARMUP_CYCLES: u32 = 256;
/// A full trajectory amortizes submission overhead and includes five refreshes.
const TIMED_CYCLES: u32 = 100;
/// Rotate arm order and use the median of an odd number of pairs.
const TIMING_PAIRS: usize = 5;
/// The boundary fixture leaves this agent inactive without requesting respawn.
const INACTIVE_AGENT: u32 = 1;
/// The two forced deaths must occur, including one at a refresh boundary.
const FORCED_DEATHS: f32 = 2.0;
/// Complete simulation snapshots contain every mutable storage buffer.
const MUTABLE_BUFFERS: usize = 13;
/// Kernel push constants hold the starting tick and brain pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Ten fixtures fit the existing ten-agent validation scene.
const MATRIX_CASES: usize = 10;
/// Matrix data is serialized in the brain snapshot's ninth buffer.
const BRAIN_BUFFER: usize = 8;
/// The smallest bounded cortex fixture still uses all cortex stages.
const CORTEX_CYCLES: u32 = 4;
const SCENE_SEED: u64 = 42;
const FOOD_ITEMS: usize = 104;
/// Finite diagonal/off-diagonal scales keep the fixtures positive semidefinite.
const DIAGONAL: f32 = 0.02;
const OFF_DIAGONAL: f32 = 0.001;
const SMALL_DIAGONAL: f32 = 1e-8;
/// Exercise the pair skip and sweep convergence thresholds from the shader.
const PAIR_BELOW: f32 = 0.5e-30;
const PAIR_ABOVE: f32 = 2e-30;
const SWEEP_BELOW: f32 = 0.5e-15;
const SWEEP_ABOVE: f32 = 2e-15;
/// A finite sentinel makes a deleted refresh detectable in the raw probe.
const WHITENING_SENTINEL: f32 = -7.0;

const REFRESH_SIGNATURE: &str =
    "fn cooperative_refresh_vision_whitening(brain_base: u32, tid: u32) {";
const CANDIDATE: &str = include_str!("../shaders/kernel/brain_cooperative_jacobi.wgsl");
const PROBE_ENTRY: &str = r"
@compute @workgroup_size(256)
fn jacobi_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    cooperative_refresh_vision_whitening(group.x * BRAIN_STRIDE, tid);
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn cooperative_rotations(original: &str) -> String {
    assert_eq!(original.matches(REFRESH_SIGNATURE).count(), 1);
    let start = original.find(REFRESH_SIGNATURE).unwrap();
    let end = start + original[start..].find("\n}").unwrap() + "\n}".len();
    let mut candidate = original.to_owned();
    candidate.replace_range(start..end, CANDIDATE.trim_end());
    assert_ne!(candidate, original, "the refresh function must change");
    assert_eq!(candidate.matches(REFRESH_SIGNATURE).count(), 1);
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        original.matches("var<workgroup>").count(),
        "the rotation helper borrows existing scratch"
    );
    candidate
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

struct Arms {
    parked: Option<Pipelines>,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        let original = optimized_brain();
        let candidate = cooperative_rotations(&original);
        assert_ne!(candidate, original);
        Self {
            parked: Pipelines::new_packed_with_store_suppression(
                kernel,
                &candidate,
                &constants(kernel),
                true,
            ),
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            // Checkpoint restoration cannot invalidate a parked private cache.
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
    }
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut states = Vec::with_capacity(PARITY_CHUNKS.len());
    let mut cycle = 0;
    for cycles in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, cycles);
        cycle += cycles;
        let state = capture_state(kernel)?;
        assert_eq!(state.len(), MUTABLE_BUFFERS);
        assert_mirror(kernel, &state)?;
        states.push(state);
    }
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
    Ok(states)
}

#[test]
fn jacobi_composition_keeps_other_helpers_and_scratch() {
    let original = optimized_brain();
    let candidate = cooperative_rotations(&original);
    let recall = original.find("fn coop_recall_score(").unwrap();
    assert!(candidate.ends_with(&original[recall..]));
    assert!(candidate.contains("var<workgroup> s_reinf_dot: array<f32, 256>"));
    assert_eq!(CANDIDATE.matches("workgroupUniformLoad(").count(), 2);
    for expression in [
        "c * akp - s * akq",
        "s * akp + c * akq",
        "c * apk - s * aqk",
        "s * apk + c * aqk",
        "c * vkp - s * vkq",
        "s * vkp + c * vkq",
    ] {
        assert!(original.contains(expression));
        assert!(candidate.contains(expression));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = apply_subgroup_markers(
            &[
                include_str!("../shaders/kernel/common.wgsl"),
                candidate.as_str(),
                PROBE_ENTRY,
            ]
            .join("\n"),
            false,
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_jacobi_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, true, true);
        let mut arms = Arms::new(&kernel);
        let saved = checkpoint(&kernel);
        let initial = capture_state(&kernel)?;
        let expected = trajectory(&mut kernel)?;
        for _ in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(actual) {
                assert_state_equal(&kernel, expected, &actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    &actual,
                    INACTIVE_AGENT,
                    "cooperative Jacobi",
                );
            }
        }
        println!("COOPERATIVE_JACOBI_PARITY vision={width}x{height} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} replays={REPLAYS} death_refresh=true private_mirror_exact=true");
    }
    Ok(())
}

fn probe_pipeline(kernel: &GpuKernel, candidate: bool) -> wgpu::ComputePipeline {
    let original = optimized_brain();
    let passes = if candidate {
        cooperative_rotations(&original)
    } else {
        original
    };
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            passes.as_str(),
            PROBE_ENTRY,
        ]
        .join("\n"),
        false,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cooperative_jacobi_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cooperative_jacobi_probe"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cooperative_jacobi_probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("jacobi_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel),
                ..Default::default()
            },
            cache: None,
        })
}

fn run_probe(kernel: &GpuKernel, pipeline: &wgpu::ComputePipeline) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

fn tail_offset(kernel: &GpuKernel, offset: usize) -> usize {
    fixed_tail_base(kernel.layout.brain_stride) + offset - O_PREDICTOR_CONTEXT_WEIGHT
}

fn covariance(case: usize, row: usize, column: usize) -> f32 {
    let diagonal = row == column;
    match case {
        0 => 0.0,
        1 => {
            if diagonal {
                DIAGONAL * f32::from(u16::try_from(row + 1).unwrap())
            } else {
                0.0
            }
        }
        2 => {
            if diagonal {
                DIAGONAL
            } else {
                0.0
            }
        }
        3 => {
            if diagonal {
                DIAGONAL
            } else {
                OFF_DIAGONAL
            }
        }
        4 => {
            if diagonal {
                DIAGONAL
            } else if (row + column).is_multiple_of(2) {
                OFF_DIAGONAL
            } else {
                -OFF_DIAGONAL
            }
        }
        5 => OFF_DIAGONAL + if diagonal { SMALL_DIAGONAL } else { 0.0 },
        6 | 7 => {
            if diagonal {
                DIAGONAL
            } else if (row == 0 && column == 1) || (row == 1 && column == 0) {
                if case == 6 {
                    PAIR_BELOW
                } else {
                    PAIR_ABOVE
                }
            } else if (row == 2 && column == 3) || (row == 3 && column == 2) {
                OFF_DIAGONAL
            } else {
                0.0
            }
        }
        8 | 9 => {
            if diagonal {
                DIAGONAL
            } else if (row == 0 && column == 1) || (row == 1 && column == 0) {
                if case == 8 {
                    SWEEP_BELOW
                } else {
                    SWEEP_ABOVE
                }
            } else {
                0.0
            }
        }
        _ => unreachable!(),
    }
}

fn upload_matrices(kernel: &GpuKernel, tick: u32) {
    let covariance_offset = tail_offset(kernel, O_VISION_PATHWAY_COVARIANCE);
    let whitening_offset = tail_offset(kernel, O_VISION_PATHWAY_WHITENING);
    let tick_offset = tail_offset(kernel, O_TICK_COUNT);
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.brain_state[tick_offset] = f32::from(u16::try_from(tick).unwrap());
        for row in 0..VISION_PATHWAY_INPUTS {
            for column in 0..VISION_PATHWAY_INPUTS {
                let cell = row * VISION_PATHWAY_INPUTS + column;
                state.brain_state[covariance_offset + cell] =
                    covariance(usize::try_from(agent).unwrap() % MATRIX_CASES, row, column);
                state.brain_state[whitening_offset + cell] = WHITENING_SENTINEL;
            }
        }
        kernel.write_agent_state(agent, &state);
    }
}

fn assert_whitening_written(kernel: &GpuKernel, state: &State) {
    let brain: &[f32] = bytemuck::cast_slice(&state[BRAIN_BUFFER]);
    let offset = tail_offset(kernel, O_VISION_PATHWAY_WHITENING);
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * kernel.layout.brain_stride + offset;
        let cells = &brain[base..base + VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS];
        assert!(cells.iter().all(|value| value.is_finite()));
        assert!(cells
            .iter()
            .any(|value| value.to_bits() != WHITENING_SENTINEL.to_bits()));
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_jacobi_covariance_edges_and_skipped_refresh() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(FIELDS[0].0, FIELDS[0].1, false, true);
    assert_eq!(usize::try_from(kernel.agent_count).unwrap(), MATRIX_CASES);
    let reference = probe_pipeline(&kernel, false);
    let candidate = probe_pipeline(&kernel, true);
    for tick in [0, 1] {
        upload_matrices(&kernel, tick);
        let saved = checkpoint(&kernel);
        let initial = capture_state(&kernel)?;
        run_probe(&kernel, &reference);
        let expected = capture_state(&kernel)?;
        if tick == 0 {
            assert_whitening_written(&kernel, &expected);
        } else {
            assert_state_equal(&kernel, &initial, &expected);
        }
        for _ in 0..REPLAYS {
            restore(&mut kernel, &saved);
            run_probe(&kernel, &candidate);
            assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        }
        println!("COOPERATIVE_JACOBI_EDGES cases={MATRIX_CASES} tick={tick} exact_buffers={MUTABLE_BUFFERS} replays={REPLAYS}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_jacobi_preserves_cortex_skip() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let brain = BrainConfig {
        vision_width: FIELDS[0].0,
        vision_height: FIELDS[0].1,
        visual_cortex_enabled: true,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut kernel = GpuKernel::new(1, FOOD_ITEMS, &brain, &WorldConfig::default());
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.set_brain_beside_vision(false);
    kernel.reset_agents_seeded(&brain, SCENE_SEED);
    super::vision_validation::upload_random_scene(&kernel, SCENE_SEED, false, false);
    kernel.global_credit = Pipelines::new_packed_with_store_suppression(
        &kernel,
        &optimized_brain(),
        &constants(&kernel),
        true,
    );
    let mut arms = Arms::new(&kernel);
    upload_matrices(&kernel, 0);
    let reference_probe = probe_pipeline(&kernel, false);
    let candidate_probe = probe_pipeline(&kernel, true);
    let initial = capture_state(&kernel)?;
    run_probe(&kernel, &reference_probe);
    assert_state_equal(&kernel, &initial, &capture_state(&kernel)?);
    run_probe(&kernel, &candidate_probe);
    assert_state_equal(&kernel, &initial, &capture_state(&kernel)?);
    let saved = checkpoint(&kernel);
    for cycle in 0..CORTEX_CYCLES {
        advance(&mut kernel, cycle, 1);
    }
    let expected = capture_state(&kernel)?;
    restore(&mut kernel, &saved);
    arms.activate(&mut kernel, true);
    for cycle in 0..CORTEX_CYCLES {
        advance(&mut kernel, cycle, 1);
    }
    assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
    assert_mirror(&kernel, &expected)?;
    println!("COOPERATIVE_JACOBI_CORTEX cycles={CORTEX_CYCLES} exact_buffers={MUTABLE_BUFFERS} raw_refresh_skipped=true");
    Ok(())
}

fn set_ticks(kernel: &GpuKernel, tick: u32) {
    let offset = tail_offset(kernel, O_TICK_COUNT);
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.brain_state[offset] = f32::from(u16::try_from(tick).unwrap());
        kernel.write_agent_state(agent, &state);
    }
}

fn paired_timing(
    kernel: &mut GpuKernel,
    arms: &mut Arms,
    saved: &Checkpoint,
    cycles: u32,
    label: &str,
) -> TestResult {
    restore(kernel, saved);
    arms.activate(kernel, false);
    advance(kernel, WARMUP_CYCLES, cycles);
    let expected = capture_state(kernel)?;
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(kernel, saved);
            arms.activate(kernel, arm != 0);
            let start = Instant::now();
            advance(kernel, WARMUP_CYCLES, cycles);
            timings[arm].push(start.elapsed().as_secs_f64());
            let actual = capture_state(kernel)?;
            assert_state_equal(kernel, &expected, &actual);
            assert_mirror(kernel, &actual)?;
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("COOPERATIVE_JACOBI_TIMING case={label} warmup_cycles={WARMUP_CYCLES} cycles={cycles} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} full_simulation=true cold_import_timed=true state_comparison_timed=false", reference/candidate);
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_cooperative_jacobi() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(FIELDS[0].0, FIELDS[0].1, false, true);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    paired_timing(&mut kernel, &mut arms, &warm, TIMED_CYCLES, "evolving")?;
    for (tick, label) in [
        (REFRESH_CYCLES, "refresh"),
        (REFRESH_CYCLES + 1, "ordinary"),
    ] {
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, false);
        set_ticks(&kernel, tick);
        let saved = checkpoint(&kernel);
        paired_timing(&mut kernel, &mut arms, &saved, 1, label)?;
    }
    Ok(())
}
