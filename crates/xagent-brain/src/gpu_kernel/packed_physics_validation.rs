//! Hardware-only grouping of independent agents' physics into one invocation
//! per agent, followed by the unchanged cooperative food claim and full cycle.
//! Both timed arms use identical command chunking; a separate production arm
//! validates the recorder against all thirteen mutable buffers.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// One compact workgroup covers the ten-agent scene without subgroup operations.
const PHYSICS_THREADS: u32 = 32;
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const REPLAYS: usize = 2;
const COMPLETE_BRAIN: u32 = 7;
const PHASE_MASK: u32 = 7;
const PUSH_CONSTANT_BYTES: u32 = 8;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;

const PHYSICS_LOOP: &str = "        for (var t = 0u; t < stride; t++) {\n            agent_physics(agent_id, base_tick + t);\n        }\n";
const PHYSICS_ENTRY: &str = r"
@compute @workgroup_size(32)
fn packed_physics_tick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= wc_u32(WC_AGENT_COUNT)) { return; }
    let stride = wc_u32(WC_BRAIN_TICK_STRIDE);
    for (var t = 0u; t < stride; t++) {
        agent_physics(agent_id, kpc.start_tick + t);
    }
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn claim_without_physics(original: &str) -> String {
    assert_eq!(original.matches(PHYSICS_LOOP).count(), 1);
    let start = original.find("fn kernel_claim_tick(").unwrap();
    let end = start + original[start..].find("\n}").unwrap();
    assert!(original[start..end].contains(PHYSICS_LOOP));
    let candidate = original.replacen(PHYSICS_LOOP, "", 1);
    assert!(candidate.contains("fn agent_physics("));
    assert!(candidate.contains("    agent_food_detect(agent_id, tid);"));
    candidate
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str) -> wgpu::ComputePipeline {
    let mut constants: HashMap<String, f64> = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packed_physics"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed_physics"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("packed_physics"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

struct Candidate {
    physics: wgpu::ComputePipeline,
    claim: wgpu::ComputePipeline,
}

impl Candidate {
    fn new(kernel: &GpuKernel) -> Self {
        let source = global_credit::main_source(&optimized_brain(), false, Some(cache(kernel)));
        let physics = pipeline(
            kernel,
            format!("{source}\n{PHYSICS_ENTRY}"),
            "packed_physics_tick",
        );
        let claim = pipeline(kernel, claim_without_physics(&source), "kernel_claim_tick");
        Self { physics, claim }
    }

    fn advance(&self, kernel: &mut GpuKernel, cycle: u32, cycles: u32, packed: bool) {
        assert!(kernel.global_credit_active());
        assert_eq!(kernel.vision_stride, 1);
        assert!(!kernel.probe.skip_vision);
        let stride = kernel.brain_tick_stride;
        let mut tick = cycle.checked_mul(stride).unwrap();
        kernel.upload_world_config_with_cycles(u64::from(tick), stride, PHASE_MASK, 1);
        let mut batch = 0;
        while batch < cycles {
            let end = (batch + MAX_FUSED_BATCHES).min(cycles);
            let mut encoder = kernel.device.create_command_encoder(&Default::default());
            cache(kernel).record_import(kernel, &mut encoder);
            {
                let credit = kernel.global_credit.as_ref().unwrap();
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &credit.bind_groups[kernel.active_config_index], &[]);
                for _ in batch..end {
                    let push = [tick, COMPLETE_BRAIN];
                    if packed {
                        pass.set_pipeline(&self.physics);
                        pass.set_push_constants(0, bytemuck::cast_slice(&push));
                        pass.dispatch_workgroups(
                            kernel.agent_count.div_ceil(PHYSICS_THREADS),
                            1,
                            1,
                        );
                    }
                    pass.set_pipeline(if packed {
                        &self.claim
                    } else {
                        &kernel.kernel_claim_pipeline
                    });
                    pass.set_push_constants(0, bytemuck::cast_slice(&push));
                    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
                    pass.set_pipeline(&credit.main);
                    pass.set_push_constants(0, bytemuck::cast_slice(&push));
                    pass.dispatch_workgroups(kernel.agent_count, 1, 1);
                    tick = tick.checked_add(stride).unwrap();
                    pass.set_pipeline(&credit.global);
                    pass.set_push_constants(0, bytemuck::cast_slice(&[tick, stride]));
                    pass.dispatch_workgroups(credit.global_workgroups, 1, 1);
                    pass.set_pipeline(&kernel.vision_pipeline);
                    pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
                }
            }
            kernel.queue.submit([encoder.finish()]);
            batch = end;
        }
        kernel.active_config_index = 1 - kernel.active_config_index;
        kernel.poll_wait();
    }
}

fn trajectory(kernel: &mut GpuKernel, candidate: &Candidate, arm: usize) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    let mut cycle = 0;
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        if arm == 0 {
            advance(kernel, cycle, count);
        } else {
            candidate.advance(kernel, cycle, count, arm == 2);
        }
        let state = capture_state(kernel)?;
        assert_mirror(kernel, &state)?;
        states.push(state);
        cycle += count;
    }
    Ok(states)
}

#[test]
fn packed_physics_composition_keeps_subtick_order() {
    let source = global_credit::main_source(&optimized_brain(), false, None);
    let candidate = claim_without_physics(&source);
    assert_eq!(source.replace(PHYSICS_LOOP, ""), candidate);
    assert!(PHYSICS_ENTRY.contains(&format!("@workgroup_size({PHYSICS_THREADS})")));
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = format!("{candidate}\n{PHYSICS_ENTRY}");
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
#[ignore = "requires GPU; exact thirteen-buffer comparison against production"]
fn packed_physics_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, true, true);
        let candidate = Candidate::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, &candidate, 0)?;
        for arm in 1..=2 {
            for replay in 0..REPLAYS {
                restore(&mut kernel, &saved);
                let actual = trajectory(&mut kernel, &candidate, arm)?;
                for (expected, actual) in expected.iter().zip(&actual) {
                    assert_state_equal(&kernel, expected, actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        actual,
                        INACTIVE_AGENT,
                        "packed physics",
                    );
                }
                println!("PACKED_PHYSICS_PARITY width={width} height={height} arm={arm} replay={replay} cycles=100 exact_buffers={MUTABLE_BUFFERS} death_refresh=true private_mirror_exact=true production_reference=true");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU complete-cycle benchmark; run explicitly in release mode"]
fn benchmark_packed_physics() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare_kernel_with_store_suppression(width, height, false, true);
    let candidate = Candidate::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &saved);
            let start = Instant::now();
            candidate.advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES, arm != 0);
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
    println!("PACKED_PHYSICS_TIMING warmup_cycles={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true full_simulation=true same_chunking=true extra_dispatch_timed=true cold_import_timed=true readback_timed=false", reference/candidate);
    Ok(())
}
