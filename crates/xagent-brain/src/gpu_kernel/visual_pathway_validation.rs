//! Hardware-only parallelization of independent visual-pathway outputs. The
//! reference retains the current production packed brain and its ordered scalar
//! pathway. Hemifield pooling calls the original scalar function; candidate
//! whitening rows keep their original FP32 accumulation order.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::packed_store_validation::{
    advance, assert_mirror, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 256;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const INACTIVE_AGENT: u32 = 1;
const MUTABLE_BUFFERS: usize = 13;
const BRAIN_BUFFER: usize = 8;
const PUSH_CONSTANT_BYTES: u32 = 8;
const CORTEX_CYCLES: u32 = 4;
const SCENE_SEED: u64 = 42;
const FOOD_ITEMS: usize = 104;
const DIAGONAL: f32 = 0.02;
const OFF_DIAGONAL: f32 = 0.001;
const INPUT_SENTINEL: f32 = -17.0;
const TURN_SENTINEL: f32 = -19.0;
const WHITENING_SENTINEL: f32 = -7.0;
/// Bound host-only failure output without instrumenting the tested shader.
const MAX_DIFFERENCE_SAMPLES: usize = 12;
const HELPER: &str = include_str!("../shaders/kernel/brain_cooperative_visual_pathway.wgsl");
const REFRESH_CALL: &str = "    cooperative_refresh_vision_whitening(brain_base, tid);\n";
const ORIGINAL_CALL: &str = "let vision_dot = vision_pathway_step(\n            brain_base, brain_state[brain_base + O_TICK_COUNT], visual_cortex_on);";

const PROBE_ENTRY: &str = r"
@compute @workgroup_size(256)
fn visual_pathway_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let brain_base = group.x * BRAIN_STRIDE;
    for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
        s_features[feature] = brain_scratch[group.x * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature];
    }
    workgroupBarrier();
    cooperative_refresh_vision_whitening(brain_base, tid);
    // PREPARE_PATHWAY
    if (tid == 0u) {
        let visual_cortex_on = bc_f32(CFG_VISUAL_CORTEX_ENABLED) != 0.0;
        let turn = vision_pathway_step(brain_base, brain_state[brain_base + O_TICK_COUNT], visual_cortex_on);
        decision_buffer[group.x * DECISION_STRIDE + DECISION_MOTOR + 1u] = turn;
    }
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn cooperative_pathway(original: &str) -> String {
    assert_eq!(original.matches(REFRESH_CALL).count(), 1);
    assert_eq!(original.matches(ORIGINAL_CALL).count(), 1);
    let refresh = original.find(REFRESH_CALL).unwrap();
    let exploration = original
        .find("    // ── Thread 0: exploration, noise, motor, telemetry")
        .unwrap();
    let call = original.find(ORIGINAL_CALL).unwrap();
    assert!(refresh < exploration && exploration < call);
    assert!(original[refresh + REFRESH_CALL.len()..exploration]
        .trim()
        .is_empty());
    assert!(original.contains("var<workgroup> s_reinf_dot: array<f32, 256>;"));
    assert!(original.contains("        s_reinf_dot[tid] = dot;\n    }\n    workgroupBarrier();"));
    let candidate = original
        .replacen(
            REFRESH_CALL,
            &format!("{REFRESH_CALL}    cooperative_prepare_vision_pathway(brain_base, tid);\n"),
            1,
        )
        .replacen(
            ORIGINAL_CALL,
            "let vision_dot = prepared_vision_pathway_turn(brain_base, visual_cortex_on);",
            1,
        );
    let candidate = format!("{candidate}\n{HELPER}");
    assert_ne!(candidate, original);
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        original.matches("var<workgroup>").count()
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
        let candidate = cooperative_pathway(&original);
        assert_ne!(candidate, original);
        let pipelines = Pipelines::new_packed_with_store_suppression(
            kernel,
            &candidate,
            &constants(kernel),
            true,
        )
        .unwrap();
        assert!(pipelines.packed_encoder.is_some());
        Self {
            parked: Some(pipelines),
            candidate: false,
        }
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
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= 2.0);
    Ok(states)
}

#[test]
fn visual_pathway_composition_preserves_other_passes() {
    let original = optimized_brain();
    let candidate = cooperative_pathway(&original);
    let refresh_definition = original
        .find("fn cooperative_refresh_vision_whitening(")
        .unwrap();
    assert!(candidate.contains(&original[refresh_definition..]));
    for expression in [
        "let raw = vision_hemifields();",
        "s_reinf_dot[k] = raw[k] - brain_state[brain_base + O_VISION_PATHWAY_MEAN + k];",
        "brain_state[slot] += VISION_PATHWAY_RATE * (s_reinf_dot[i] * s_reinf_dot[j] - brain_state[slot]);",
    ] { assert!(HELPER.contains(expression)); }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = apply_subgroup_markers(
            &[
                include_str!("../shaders/kernel/common.wgsl"),
                candidate.as_str(),
                &probe_entry(true),
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
fn cooperative_visual_pathway_preserves_complete_state() -> TestResult {
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
            for (expected, actual) in expected.iter().zip(trajectory(&mut kernel)?) {
                assert_state_equal(&kernel, expected, &actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    &actual,
                    INACTIVE_AGENT,
                    "cooperative visual pathway",
                );
            }
        }
        println!("COOPERATIVE_VISUAL_PATHWAY_PARITY vision={width}x{height} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} replays={REPLAYS} death_refresh=true private_mirror_exact=true");
    }
    Ok(())
}

fn probe_entry(candidate: bool) -> String {
    if candidate {
        PROBE_ENTRY.replace("    // PREPARE_PATHWAY", "    cooperative_prepare_vision_pathway(brain_base, tid);")
            .replace("vision_pathway_step(brain_base, brain_state[brain_base + O_TICK_COUNT], visual_cortex_on)", "prepared_vision_pathway_turn(brain_base, visual_cortex_on)")
    } else {
        PROBE_ENTRY.to_owned()
    }
}

fn probe_pipeline(kernel: &GpuKernel, candidate: bool) -> wgpu::ComputePipeline {
    let original = optimized_brain();
    let passes = if candidate {
        cooperative_pathway(&original)
    } else {
        original
    };
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            passes.as_str(),
            &probe_entry(candidate),
        ]
        .join("\n"),
        false,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cooperative_visual_pathway_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cooperative_visual_pathway_probe"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cooperative_visual_pathway_probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("visual_pathway_probe"),
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

fn tail(kernel: &GpuKernel, offset: usize) -> usize {
    fixed_tail_base(kernel.layout.brain_stride) + offset - O_PREDICTOR_CONTEXT_WEIGHT
}

fn small_float(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap())
}

fn upload_phase_fixture(kernel: &GpuKernel, tick: u32) {
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        let agent_index = usize::try_from(agent).unwrap();
        state.brain_state[tail(kernel, O_TICK_COUNT)] = f32::from(u16::try_from(tick).unwrap());
        for i in 0..VISION_PATHWAY_INPUTS {
            state.brain_state[tail(kernel, O_VISION_PATHWAY_MEAN) + i] =
                (small_float(i) - 3.0) * 0.125;
            state.brain_state[tail(kernel, O_VISION_PATHWAY_INPUT) + i] = INPUT_SENTINEL;
            state.brain_state[tail(kernel, O_VISION_TURN_WEIGHTS) + i] =
                (small_float(i) - 4.0) * 0.0625;
            for j in 0..VISION_PATHWAY_INPUTS {
                let cell = i * VISION_PATHWAY_INPUTS + j;
                state.brain_state[tail(kernel, O_VISION_PATHWAY_COVARIANCE) + cell] = if i == j {
                    DIAGONAL * small_float(i + 1)
                } else {
                    OFF_DIAGONAL
                };
                state.brain_state[tail(kernel, O_VISION_PATHWAY_WHITENING) + cell] = if tick == 0 {
                    WHITENING_SENTINEL
                } else {
                    (small_float((cell + agent_index) % 13) - 6.0) * 0.125
                };
            }
        }
        kernel.write_agent_state(agent, &state);
        let features: Vec<f32> = (0..kernel.layout.feature_count)
            .map(|feature| {
                let signed = (small_float((feature * 7 + agent_index * 11) % 31) - 15.0) * 0.0625;
                match agent_index % 4 {
                    0 => 0.0,
                    1 => 0.25,
                    2 => signed,
                    _ => {
                        if feature.is_multiple_of(2) {
                            signed
                        } else {
                            -signed
                        }
                    }
                }
            })
            .collect();
        let offset =
            usize::try_from(agent).unwrap() * kernel.layout.brain_scratch_stride + SCRATCH_FEATURES;
        kernel.queue.write_buffer(
            &kernel.brain_scratch_buffer,
            u64::try_from(offset * size_of::<f32>()).unwrap(),
            bytemuck::cast_slice(&features),
        );
        let motor = usize::try_from(agent).unwrap() * DECISION_STRIDE + DECISION_MOTOR + 1;
        kernel.queue.write_buffer(
            &kernel.decision_buffer,
            u64::try_from(motor * size_of::<f32>()).unwrap(),
            bytemuck::bytes_of(&TURN_SENTINEL),
        );
    }
}

fn assert_phase_changed(
    kernel: &GpuKernel,
    initial: &State,
    actual: &State,
    tick: u32,
    cortex: bool,
) {
    let before: &[f32] = bytemuck::cast_slice(&initial[BRAIN_BUFFER]);
    let after: &[f32] = bytemuck::cast_slice(&actual[BRAIN_BUFFER]);
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * kernel.layout.brain_stride;
        let input = base + tail(kernel, O_VISION_PATHWAY_INPUT);
        let inputs = &after[input..input + VISION_PATHWAY_INPUTS];
        assert!(inputs
            .iter()
            .all(|value| value.is_finite() && value.to_bits() != INPUT_SENTINEL.to_bits()));
        if cortex {
            assert!(inputs
                .iter()
                .all(|value| value.to_bits() == 0.0_f32.to_bits()));
        }
        for (offset, length, changed) in [
            (O_VISION_PATHWAY_MEAN, VISION_PATHWAY_INPUTS, !cortex),
            (
                O_VISION_PATHWAY_COVARIANCE,
                VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS,
                !cortex,
            ),
            (
                O_VISION_PATHWAY_WHITENING,
                VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS,
                !cortex && tick == 0,
            ),
        ] {
            let begin = base + tail(kernel, offset);
            let equal = before[begin..begin + length]
                .iter()
                .zip(&after[begin..begin + length])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert_eq!(
                !equal, changed,
                "phase fixture agent={agent} offset={offset} tick={tick} cortex={cortex}"
            );
        }
    }
}

fn report_phase_difference(kernel: &GpuKernel, expected: &State, actual: &State, tick: u32) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    for (name, buffer, stride, offset, length) in [
        (
            "mean",
            BRAIN_BUFFER,
            kernel.layout.brain_stride,
            tail(kernel, O_VISION_PATHWAY_MEAN),
            VISION_PATHWAY_INPUTS,
        ),
        (
            "whitened_input",
            BRAIN_BUFFER,
            kernel.layout.brain_stride,
            tail(kernel, O_VISION_PATHWAY_INPUT),
            VISION_PATHWAY_INPUTS,
        ),
        (
            "covariance",
            BRAIN_BUFFER,
            kernel.layout.brain_stride,
            tail(kernel, O_VISION_PATHWAY_COVARIANCE),
            VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS,
        ),
        (
            "whitening",
            BRAIN_BUFFER,
            kernel.layout.brain_stride,
            tail(kernel, O_VISION_PATHWAY_WHITENING),
            VISION_PATHWAY_INPUTS * VISION_PATHWAY_INPUTS,
        ),
        ("turn", 1, DECISION_STRIDE, DECISION_MOTOR + 1, 1),
    ] {
        let reference: &[f32] = bytemuck::cast_slice(&expected[buffer]);
        let candidate: &[f32] = bytemuck::cast_slice(&actual[buffer]);
        let mut changed = 0;
        let mut max_abs = 0.0_f64;
        for agent in 0..agents {
            for component in 0..length {
                let index = agent * stride + offset + component;
                let before = reference[index];
                let after = candidate[index];
                if before.to_bits() == after.to_bits() {
                    continue;
                }
                changed += 1;
                max_abs = max_abs.max((f64::from(after) - f64::from(before)).abs());
                if changed <= MAX_DIFFERENCE_SAMPLES {
                    eprintln!("VISUAL_PATHWAY_PHASE_DIFFERENCE tick={tick} region={name} agent={agent} component={component} expected={before:.9e} actual={after:.9e} expected_bits={:08x} actual_bits={:08x}", before.to_bits(), after.to_bits());
                }
            }
        }
        eprintln!("VISUAL_PATHWAY_PHASE_REGION tick={tick} region={name} changed={changed} compared={} max_abs={max_abs:.9e}", agents * length);
    }
}

fn phase_fixture(kernel: &mut GpuKernel, cortex: bool) -> TestResult {
    let reference = probe_pipeline(kernel, false);
    let candidate = probe_pipeline(kernel, true);
    for tick in [0, 1] {
        upload_phase_fixture(kernel, tick);
        let saved = checkpoint(kernel);
        let initial = capture_state(kernel)?;
        run_probe(kernel, &reference);
        let expected = capture_state(kernel)?;
        assert_phase_changed(kernel, &initial, &expected, tick, cortex);
        for _ in 0..REPLAYS {
            restore(kernel, &saved);
            run_probe(kernel, &candidate);
            let actual = capture_state(kernel)?;
            if expected != actual {
                report_phase_difference(kernel, &expected, &actual, tick);
            }
            assert_state_equal(kernel, &expected, &actual);
        }
        println!("COOPERATIVE_VISUAL_PATHWAY_PHASE vision={}x{} cortex={cortex} tick={tick} exact_buffers={MUTABLE_BUFFERS} replays={REPLAYS} old_covariance_refresh=true old_mean_centering=true", kernel.layout.vision_width, kernel.layout.vision_height);
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_visual_pathway_preserves_phase_ordering() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, false, true);
        phase_fixture(&mut kernel, false)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn cooperative_visual_pathway_preserves_cortex_skip() -> TestResult {
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
    phase_fixture(&mut kernel, true)?;
    let saved = checkpoint(&kernel);
    for cycle in 0..CORTEX_CYCLES {
        advance(&mut kernel, cycle, 1);
    }
    let expected = capture_state(&kernel)?;
    assert_mirror(&kernel, &expected)?;
    for _ in 0..REPLAYS {
        restore(&mut kernel, &saved);
        arms.activate(&mut kernel, true);
        for cycle in 0..CORTEX_CYCLES {
            advance(&mut kernel, cycle, 1);
        }
        let actual = capture_state(&kernel)?;
        assert_state_equal(&kernel, &expected, &actual);
        assert_mirror(&kernel, &actual)?;
    }
    println!("COOPERATIVE_VISUAL_PATHWAY_CORTEX cycles={CORTEX_CYCLES} exact_buffers={MUTABLE_BUFFERS} replays={REPLAYS} one_cycle_per_poll=true private_mirror_exact=true");
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_cooperative_visual_pathway() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(FIELDS[0].0, FIELDS[0].1, false, true);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
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
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("COOPERATIVE_VISUAL_PATHWAY_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} full_simulation=true cold_import_timed=true state_comparison_timed=false", reference/candidate);
    Ok(())
}
