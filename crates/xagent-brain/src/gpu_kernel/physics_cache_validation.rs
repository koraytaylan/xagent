//! Keep an agent's physics fields in function storage across its sub-ticks.
//! The canonical per-tick expressions remain unchanged; only field accesses
//! are redirected, with one load before the loop and publication before claim.

use std::{collections::BTreeSet, error::Error, fmt::Write, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const SUBTICKS: [u32; 2] = [1, 10];
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const PUSH_BYTES: u32 = 8;
const INACTIVE_AGENT: u32 = 1;
/// The boundary fixture forces agent zero to die initially and at cycle twenty.
const EXPECTED_FORCED_DEATHS: f32 = 2.0;
const MUTABLE_BUFFERS: usize = 13;
const FIELD_ACCESS: &str = "physics_state[b + ";
const STEP_SIGNATURE: &str = "fn agent_physics(agent_id: u32, tick: u32) {";
const STEP_LOOP: &str = "        for (var t = 0u; t < stride; t++) {\n            agent_physics(agent_id, base_tick + t);\n        }\n";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn cached_physics_source(source: &str) -> String {
    assert_eq!(source.matches(STEP_SIGNATURE).count(), 1);
    assert_eq!(source.matches(STEP_LOOP).count(), 1);
    let start = source.find(STEP_SIGNATURE).unwrap();
    let end = start + source[start..].find("\n}").unwrap() + "\n}".len();
    let original = &source[start..end];
    let mut reads = BTreeSet::new();
    let mut writes = BTreeSet::new();
    for (index, _) in original.match_indices(FIELD_ACCESS) {
        let rest = &original[index + FIELD_ACCESS.len()..];
        let end = rest.find(']').unwrap();
        let field = &rest[..end];
        assert!(field.starts_with("P_"));
        assert!(field
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'_'));
        reads.insert(field);
        let after = rest[end + 1..].trim_start();
        if (after.starts_with('=') && !after.starts_with("=="))
            || ["+=", "-=", "*=", "/="]
                .iter()
                .any(|operator| after.starts_with(operator))
        {
            writes.insert(field);
        }
    }
    assert!(writes.contains("P_ALIVE"));
    assert!(writes.contains("P_LAST_DEATH_TICK"));
    assert!(!writes.contains("P_MAX_ENERGY"));
    assert!(writes.len() < reads.len());
    let step = original
        .replacen(STEP_SIGNATURE, "fn cached_physics_step(agent_id: u32, tick: u32, state: ptr<function, array<f32, PHYS_STRIDE>>) {", 1)
        .replacen("    let b = agent_id * PHYS_STRIDE;\n", "", 1)
        .replace(FIELD_ACCESS, "(*state)[");
    assert!(!step.contains("physics_state["));
    assert!(!step.contains("Barrier"));
    let mut wrapper = String::from("fn cached_physics_subticks(agent_id: u32, base_tick: u32, stride: u32) {\n    let b = agent_id * PHYS_STRIDE;\n    if (stride == 0u || physics_state[b + P_ALIVE] < 0.5) { return; }\n    var state: array<f32, PHYS_STRIDE>;\n");
    for field in &reads {
        writeln!(wrapper, "    state[{field}] = physics_state[b + {field}];").unwrap();
    }
    wrapper.push_str("    for (var tick = 0u; tick < stride; tick++) {\n        cached_physics_step(agent_id, base_tick + tick, &state);\n    }\n");
    for field in &writes {
        writeln!(wrapper, "    physics_state[b + {field}] = state[{field}];").unwrap();
    }
    wrapper.push_str("}\n");
    let candidate = source.replacen(
        STEP_LOOP,
        "        cached_physics_subticks(agent_id, base_tick, stride);\n",
        1,
    );
    format!("{candidate}\n{step}\n{wrapper}")
}

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
        &kernel,
        &optimized_brain(),
        &constants,
        true,
    );
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        main_width::MAIN_THREADS
    );
    kernel
}

struct Arms {
    parked: wgpu::ComputePipeline,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        let source = global_credit::main_source(
            &optimized_brain(),
            kernel.has_subgroup,
            Some(cache(kernel)),
        );
        let source = cached_physics_source(&source);
        let module = kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cached_physics"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
        let layout = kernel
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cached_physics"),
                bind_group_layouts: &[&binding],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..PUSH_BYTES,
                }],
            });
        let mut constants = vision_override_constants(&kernel.layout);
        constants.insert("VISION_AGENT_MASKS".into(), 1.0);
        let parked = kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cached_physics"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("kernel_claim_tick"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
        Self {
            parked,
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            std::mem::swap(&mut self.parked, &mut kernel.kernel_claim_pipeline);
            self.candidate = candidate;
        }
    }
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in CHUNKS {
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
fn physics_cache_composition_retains_the_step_expressions() {
    let original = global_credit::main_source(&optimized_brain(), false, None);
    let candidate = cached_physics_source(&original);
    assert_ne!(candidate, original);
    assert_eq!(
        candidate
            .matches("        cached_physics_subticks(agent_id, base_tick, stride);\n")
            .count(),
        1
    );
    assert!(!candidate.contains(STEP_LOOP));
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        original.matches("var<workgroup>").count()
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let module = wgpu::naga::front::wgsl::parse_str(&candidate)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&candidate)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&candidate)));
    }
}

#[test]
#[ignore = "requires GPU; full-state and mirror validation across sub-tick counts"]
fn physics_cache_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        for stride in SUBTICKS {
            let mut kernel = prepare(width, height, true);
            kernel.brain_tick_stride = stride;
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
                        "physics cache",
                    );
                }
                println!("PHYSICS_CACHE_PARITY width={width} height={height} stride={stride} replay={replay} cycles=100 exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true death_refresh=true");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU complete simulation benchmark; run explicitly in release mode"]
fn benchmark_cached_physics() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
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
    println!("PHYSICS_CACHE_TIMING warmup={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true main_threads=128 full_simulation=true cold_import_timed=true readback_timed=false", reference/candidate);
    Ok(())
}
