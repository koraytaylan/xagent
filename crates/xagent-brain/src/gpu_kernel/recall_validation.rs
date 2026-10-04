//! Hardware-only recall lifetime experiment. Interleaved norm and dot loops
//! retain each original sum order. Both private and shared whitening matrices
//! are tested to expose independent register-pressure limits in one kernel.

use std::{error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::whitening_storage_validation::workgroup_whitening_common;
use super::whitening_validation::{
    force_death, prepare_boundary_scene, prepare_kernel, PARITY_CHUNKS, REFRESH_CYCLES,
};
use super::*;

/// Baseline, recall only, and recall plus shared whitening are distinct arms.
const ARM_COUNT: usize = 3;
const BASELINE_ARM: usize = 0;
/// Combined arm removes the independent private whitening-matrix lifetime.
const SHARED_WHITENING_ARM: usize = 2;
/// The original recall calculates norm and dot in two separate loops.
const ORIGINAL_RECALL_LOOPS: usize = 2;
const ARM_NAMES: [&str; ARM_COUNT] = [
    "serial",
    "recall_interleaved",
    "recall_and_shared_whitening",
];
/// Kernel push constants contain tick and complete-pass limit.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Mature memory includes every recent-experience slot and recall pattern.
const WARMUP_CYCLES: u32 = 256;
/// Four normal command chunks amortize submission and completion overhead.
const TIMED_CYCLES: u32 = MAX_FUSED_BATCHES * 4;
/// Rotate three arms over five checkpoint-matched timing rounds.
const TIMING_ROUNDS: usize = 5;
/// Every mutable storage buffer in the fused cycle is compared exactly.
const MUTABLE_BUFFERS: usize = 13;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct PipelineArms {
    parked: [Option<wgpu::ComputePipeline>; ARM_COUNT],
    active: usize,
}

impl PipelineArms {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            parked: [
                None,
                Some(make_pipeline(kernel, false)),
                Some(make_pipeline(kernel, true)),
            ],
            active: BASELINE_ARM,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if arm != self.active {
            let next = self.parked[arm].take().unwrap();
            let previous = std::mem::replace(&mut kernel.kernel_pipeline, next);
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
    }
}

pub(super) fn interleaved_brain_passes() -> String {
    let source = include_str!("../shaders/kernel/brain_passes.wgsl");
    const BEGIN: &str = "fn coop_recall_score(agent_id: u32, tid: u32) {";
    assert_eq!(source.matches(BEGIN).count(), 1);
    let start = source.find(BEGIN).unwrap();
    let opening = start + BEGIN.len() - 1;
    let mut nesting = 0_u32;
    let end = source[opening..]
        .bytes()
        .enumerate()
        .find_map(|(offset, byte)| {
            match byte {
                b'{' => nesting += 1,
                b'}' => {
                    nesting -= 1;
                    if nesting == 0 {
                        return Some(opening + offset + 1);
                    }
                }
                _ => {}
            }
            None
        })
        .unwrap();
    let original = &source[start..end];
    assert_eq!(
        original
            .matches("for (var d: u32 = 0u; d < ENCODED_DIMENSION; d = d + 1u)")
            .count(),
        ORIGINAL_RECALL_LOOPS
    );
    assert!(original.contains("q_norm_sq += v * v;"));
    assert!(original
        .contains("dot += s_memory_key[d] * pattern_buffer[pattern_base + d * MEMORY_CAP + tid];"));
    [
        &source[..start],
        include_str!("recall_interleaved.wgsl"),
        &source[end..],
    ]
    .concat()
}

fn make_pipeline(kernel: &GpuKernel, shared_whitening: bool) -> wgpu::ComputePipeline {
    let common = if shared_whitening {
        workgroup_whitening_common()
    } else {
        include_str!("../shaders/kernel/common.wgsl").to_owned()
    };
    let passes = interleaved_brain_passes();
    let source = apply_subgroup_markers(
        &[
            common.as_str(),
            passes.as_str(),
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            include_str!("../shaders/kernel/phase_food_claim.wgsl"),
            include_str!("../shaders/kernel/kernel_tick.wgsl"),
        ]
        .join("\n"),
        kernel.has_subgroup,
    );
    let label = if shared_whitening {
        ARM_NAMES[SHARED_WHITENING_ARM]
    } else {
        ARM_NAMES[1]
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
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kernel_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn advance(kernel: &mut GpuKernel, start_tick: u32, cycles: u32) {
    kernel.dispatch_ticks(u64::from(start_tick), cycles * kernel.brain_tick_stride);
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn interleaved_recall_matches_complete_serial_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut pipelines = PipelineArms::new(&kernel);
    prepare_boundary_scene(&kernel);
    let initial = checkpoint(&kernel);
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
    for (arm, name) in ARM_NAMES.iter().enumerate().skip(1) {
        restore(&mut kernel, &initial);
        pipelines.activate(&mut kernel, arm);
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
        println!("RECALL_PARITY variant={name} cycles={cycle} exact_buffers={MUTABLE_BUFFERS}");
    }
    Ok(())
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_interleaved_recall_against_serial() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let mut pipelines = PipelineArms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let tick = WARMUP_CYCLES * kernel.brain_tick_stride;
    let mut timings: [Vec<f64>; ARM_COUNT] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_ROUNDS {
        let mut states: [Option<Vec<Vec<u8>>>; ARM_COUNT] = std::array::from_fn(|_| None);
        for offset in 0..ARM_COUNT {
            let arm = (round + offset) % ARM_COUNT;
            restore(&mut kernel, &warm);
            pipelines.activate(&mut kernel, arm);
            let start = Instant::now();
            advance(&mut kernel, tick, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            states[arm] = Some(capture_state(&kernel)?);
        }
        for state in states.iter().skip(1) {
            assert_state_equal(
                &kernel,
                states[BASELINE_ARM].as_ref().unwrap(),
                state.as_ref().unwrap(),
            );
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[BASELINE_ARM][TIMING_ROUNDS / 2];
    let ticks = TIMED_CYCLES * kernel.brain_tick_stride;
    for (name, samples) in ARM_NAMES.iter().zip(timings) {
        let seconds = samples[TIMING_ROUNDS / 2];
        println!(
            "RECALL_INTERLEAVING variant={name} agents={} ticks={ticks} rounds={TIMING_ROUNDS} seconds={seconds:.9} tps={:.3} speedup={:.3} exact_buffers={MUTABLE_BUFFERS}",
            kernel.agent_count, f64::from(ticks) / seconds, baseline / seconds,
        );
    }
    Ok(())
}
