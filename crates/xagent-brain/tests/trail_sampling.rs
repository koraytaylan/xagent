//! The GPU trail ring records agent positions at tick-aligned boundaries, so
//! the sampled path depends on simulated ticks alone — not on how the host
//! splits ticks into dispatches or when it reads state back.

use xagent_brain::buffers::{
    TrailSample, PHYS_STRIDE, P_POS_X, P_POS_Y, P_POS_Z, TRAIL_RECORD_STRIDE, TRAIL_RING_SLOTS,
};
use xagent_brain::GpuKernel;
use xagent_shared::{BrainConfig, WorldConfig};

const TERRAIN_SIDE: usize = 129;
const BIOME_SIDE: usize = 256;
const AGENT_COUNT: u32 = 3;
const SNAPSHOT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

fn kernel() -> GpuKernel {
    // Several brain cycles per batch, so a dispatch can end inside a batch.
    let brain = BrainConfig {
        brain_tick_stride: 5,
        vision_stride: 4,
        ..BrainConfig::default()
    };
    let world = WorldConfig::default();
    let kernel = GpuKernel::new(AGENT_COUNT, 1, &brain, &world);
    let heights = vec![0.0_f32; TERRAIN_SIDE * TERRAIN_SIDE];
    let biomes = vec![0_u32; BIOME_SIDE * BIOME_SIDE];
    kernel.upload_world(&heights, &biomes, &[(40.0, 0.0, 40.0)], &[false], &[0.0]);
    let agents: Vec<_> = (0..AGENT_COUNT)
        .map(|index| {
            (
                glam::Vec3::new(index as f32 * 10.0, 1.0, 0.0),
                100.0,
                100.0,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&agents);
    kernel
}

/// Stage a state snapshot and wait for it to land in the kernel's caches.
fn collect_snapshot(kernel: &mut GpuKernel) -> Vec<TrailSample> {
    assert!(
        kernel.request_state_snapshot(),
        "a staging slot must be free"
    );
    let started = std::time::Instant::now();
    while started.elapsed() < SNAPSHOT_TIMEOUT {
        if kernel.try_collect_state_snapshot() {
            return kernel.collected_trail_samples();
        }
        std::thread::yield_now();
    }
    panic!("state snapshot never completed");
}

fn sample_numbers(samples: &[TrailSample]) -> Vec<u64> {
    samples.iter().map(|sample| sample.sample_number).collect()
}

#[test]
fn trail_samples_land_on_kernel_batch_boundaries_whatever_the_dispatch_split() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = kernel();
    let batch = kernel.kernel_batch_size();

    // One dispatch of three batches, then a dispatch of one batch, split
    // differently from how a faster or slower host clock would split them.
    kernel.dispatch_ticks(0, 3 * batch);
    kernel.dispatch_ticks(u64::from(3 * batch), batch);
    let samples = collect_snapshot(&mut kernel);
    assert_eq!(sample_numbers(&samples), vec![1, 2, 3, 4]);

    // The newest sample is every agent's position at the end of the last batch.
    let state = kernel.read_full_state_blocking().to_vec();
    let newest = samples.last().expect("four samples were recorded");
    for agent in 0..AGENT_COUNT as usize {
        let base = agent * PHYS_STRIDE;
        let record = &newest.records[agent * TRAIL_RECORD_STRIDE..][..TRAIL_RECORD_STRIDE];
        assert_eq!(record[0], state[base + P_POS_X]);
        assert_eq!(record[1], state[base + P_POS_Y]);
        assert_eq!(record[2], state[base + P_POS_Z]);
        assert_eq!(record[3], 0.0, "no agent has died yet");
    }
}

#[test]
fn a_dispatch_ending_inside_a_batch_records_no_sample() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = kernel();
    let batch = kernel.kernel_batch_size();
    let brain_stride = kernel.brain_tick_stride();

    kernel.dispatch_ticks(0, batch + brain_stride);
    let samples = collect_snapshot(&mut kernel);
    assert_eq!(sample_numbers(&samples), vec![1]);

    // Finishing the batch records the second sample.
    kernel.dispatch_ticks(u64::from(batch + brain_stride), batch - brain_stride);
    let samples = collect_snapshot(&mut kernel);
    assert_eq!(sample_numbers(&samples), vec![1, 2]);
}

#[test]
fn the_ring_keeps_only_the_newest_samples() {
    if !GpuKernel::is_available() {
        eprintln!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let mut kernel = kernel();
    let batch = kernel.kernel_batch_size();
    let total = u32::try_from(TRAIL_RING_SLOTS).expect("ring size fits u32") + 5;

    kernel.dispatch_ticks(0, total * batch);
    let samples = collect_snapshot(&mut kernel);
    assert_eq!(samples.len(), TRAIL_RING_SLOTS);
    assert_eq!(
        samples.last().map(|s| s.sample_number),
        Some(u64::from(total))
    );
    assert!(
        samples
            .windows(2)
            .all(|pair| pair[1].sample_number == pair[0].sample_number + 1),
        "retained samples must be consecutive"
    );
}
