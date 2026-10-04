//! Frozen-cycle attribution of the global world's cost beside packed credit.
//! The measurement-only arm dispatches just the original world workgroup;
//! it never feeds its omitted encoder update into a subsequent cycle.
//! Full recording must match production, and the reduced arm may differ
//! only in the encoder matrices whose updates it deliberately omits.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance, assert_mirror, cache, prepare_kernel_with_store_suppression,
};
use super::vision_validation::read_buffer;
use super::*;

/// Match the complete-cycle packed encoder measurement's raw visual field.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
/// Sample the same early and mature windows as unchanged-store diagnostics.
const WARMUP_CYCLES: [u32; 2] = [256, 1_000];
/// Alternate arm order across an odd number of restored-input trials.
const ROUNDS: usize = 7;
/// One query immediately before and one immediately after global dispatch.
const QUERY_COUNT: u32 = 2;
/// Production fused execution includes all seven brain and world phases.
const COMPLETE_BRAIN: u32 = 7;
const PHASE_MASK: u32 = 7;
/// Workgroup zero performs the ordinary world update in the combined shader.
const WORLD_GROUPS: u32 = 1;
/// Brain state is the ninth entry in the canonical thirteen-buffer capture.
const BRAIN_BUFFER: usize = 8;
const MUTABLE_BUFFERS: usize = 13;
const NANOS_PER_MICROSECOND: f64 = 1_000.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct Timer {
    queries: wgpu::QuerySet,
    results: wgpu::Buffer,
}

impl Timer {
    fn new(kernel: &GpuKernel) -> Self {
        Self {
            queries: kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("global_world_attribution"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_COUNT,
            }),
            results: kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("global_world_timestamps"),
                size: u64::from(QUERY_COUNT) * u64::from(wgpu::QUERY_SIZE),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
        }
    }

    fn measure(&self, kernel: &mut GpuKernel, cycle: u32, world_only: bool) -> TestResult<f64> {
        assert!(kernel.global_credit_active());
        assert_eq!(kernel.vision_stride, 1);
        let tick = cycle.checked_mul(kernel.brain_tick_stride).unwrap();
        let batch_ticks = kernel.kernel_batch_size();
        assert_eq!(batch_ticks, kernel.brain_tick_stride);
        kernel.upload_world_config_with_cycles(u64::from(tick), batch_ticks, PHASE_MASK, 1);
        let credit = kernel.global_credit.as_ref().unwrap();
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        cache(kernel).record_import(kernel, &mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &credit.bind_groups[kernel.active_config_index], &[]);
            pass.set_pipeline(&kernel.kernel_claim_pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.set_pipeline(&credit.main);
            pass.set_push_constants(0, bytemuck::cast_slice(&[tick, COMPLETE_BRAIN]));
            pass.dispatch_workgroups(kernel.agent_count, 1, 1);
            pass.set_pipeline(&credit.global);
            pass.set_push_constants(
                0,
                bytemuck::cast_slice(&[tick.checked_add(batch_ticks).unwrap(), batch_ticks]),
            );
            pass.write_timestamp(&self.queries, 0);
            pass.dispatch_workgroups(
                if world_only {
                    WORLD_GROUPS
                } else {
                    credit.global_workgroups
                },
                1,
                1,
            );
            pass.write_timestamp(&self.queries, QUERY_COUNT - 1);
            pass.set_pipeline(&kernel.vision_pipeline);
            pass.dispatch_workgroups(kernel.vision_workgroups, 1, 1);
        }
        encoder.resolve_query_set(&self.queries, 0..QUERY_COUNT, &self.results, 0);
        kernel.queue.submit([encoder.finish()]);
        let raw = read_buffer(kernel, &self.results, self.results.size())?;
        let timestamps: &[u64] = bytemuck::cast_slice(&raw);
        assert_eq!(timestamps.len(), usize::try_from(QUERY_COUNT).unwrap());
        assert!(timestamps[1] > timestamps[0]);
        kernel.active_config_index = 1 - kernel.active_config_index;
        Ok(
            (timestamps[1] - timestamps[0]) as f64 * f64::from(kernel.queue.get_timestamp_period())
                / NANOS_PER_MICROSECOND,
        )
    }
}

/// Copy only the deliberately omitted matrices into an otherwise complete
/// reference. Every other public word remains subject to exact comparison.
fn without_encoder_update(
    kernel: &GpuKernel,
    initial: &[Vec<u8>],
    complete: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    assert_eq!(initial.len(), MUTABLE_BUFFERS);
    assert_eq!(complete.len(), MUTABLE_BUFFERS);
    let mut expected = complete.to_vec();
    let matrix_bytes = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap()
        .checked_mul(size_of::<f32>())
        .unwrap();
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let first = (agent * kernel.layout.brain_stride + O_ENC_WEIGHTS) * size_of::<f32>();
        let last = first + matrix_bytes;
        expected[BRAIN_BUFFER][first..last].copy_from_slice(&initial[BRAIN_BUFFER][first..last]);
    }
    expected
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "requires GPU timestamps; restored-cycle attribution, not a speedup benchmark"]
fn profile_global_world_beside_packed_credit() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for warmup in WARMUP_CYCLES {
        let mut kernel = prepare_kernel_with_store_suppression(WIDTH, HEIGHT, false, true);
        assert!(kernel
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES));
        advance(&mut kernel, 0, warmup);
        let saved = checkpoint(&kernel);
        let initial = capture_state(&kernel)?;
        advance(&mut kernel, warmup, 1);
        let complete = capture_state(&kernel)?;
        let world_only = without_encoder_update(&kernel, &initial, &complete);
        assert_ne!(world_only[BRAIN_BUFFER], complete[BRAIN_BUFFER]);
        let expected = [&complete, &world_only];
        let timer = Timer::new(&kernel);
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..ROUNDS {
            for arm in [round % 2, 1 - round % 2] {
                restore(&mut kernel, &saved);
                samples[arm].push(timer.measure(&mut kernel, warmup, arm != 0)?);
                let actual = capture_state(&kernel)?;
                assert_state_equal(&kernel, expected[arm], &actual);
                assert_mirror(&kernel, &actual)?;
            }
        }
        restore(&mut kernel, &saved);
        let full_micros = median(&mut samples[0]);
        let world_micros = median(&mut samples[1]);
        println!("GLOBAL_WORLD_ATTRIBUTION warmup_cycles={warmup} rounds={ROUNDS} full_global_us={full_micros:.3} world_only_us={world_micros:.3} difference_us={:.3} restored_single_cycle=true full_recorder_exact_buffers={MUTABLE_BUFFERS} world_only_differs_only_in_omitted_encoder_matrices=true private_scalar_mirror_exact=true same_compiled_global_shader=true measurement_only=true non_additive=true", full_micros-world_micros);
    }
    Ok(())
}
