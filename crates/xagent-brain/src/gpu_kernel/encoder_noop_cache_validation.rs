//! Hardware-only block-gap caching for encoder credit. The independent
//! reference is the production global-credit option with the current optimized
//! brain and vision. Only the candidate global shader and private binding 13
//! differ. All lifecycle invalidations below are explicit test-harness work;
//! production APIs do not maintain this experimental cache.

use std::{error::Error, time::Instant};

use super::cached_combined_validation::prepare;
use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::exact_tiled_validation::make_transient_bind_group;
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, prepare_boundary_scene, REFRESH_CYCLES};
use super::*;

/// Each cache entry bounds 32 contiguous output dimensions of one feature.
const BLOCK_WIDTH: usize = 32;
/// One cached half-gap followed by one last-cycle skipped-update count.
const RECORD_WORDS: usize = 2;
const WORD_BYTES: usize = std::mem::size_of::<f32>();
/// Match the optional production brain configuration in both arms.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Exercise the default layout and a padded final credit workgroup.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Two deaths and whitening refreshes through cycle 100.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const INACTIVE_AGENT: u32 = 1;
const FORCED_DEATHS: f32 = 2.0;
/// Mature adaptation makes the measured no-op density relevant.
const WARMUP_CYCLES: u32 = 1_000;
/// The first candidate cycle builds metadata before the common timing state.
const CACHE_WARMUP_CYCLES: u32 = 1;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
/// An active cycle, two transition cycles, then three resumed cycles.
const LIFECYCLE_CYCLES: [u32; 3] = [1, 2, 3];
const RESET_SEED: u64 = 314;
/// A much smaller weight invalidates previously cached neighbor gaps.
const HOST_WEIGHT: f32 = 1.0 / 1_099_511_627_776.0;
const HOST_VISUAL_MEAN: f32 = 0.25;
/// Exercise signed zero, subnormals, normal underflow boundary, clamp endpoints,
/// and finite values outside the encoder's allowed interval.
const GUARD_WEIGHTS: [f32; 9] = [
    0.0,
    -0.0,
    f32::from_bits(1),
    -f32::from_bits(1),
    f32::MIN_POSITIVE,
    2.0,
    -2.0,
    3.0,
    -3.0,
];
const COMPLETE_BRAIN: u32 = 7;
const WITHOUT_LEARNING: u32 = COMPLETE_BRAIN - 1;
const PUSH_CONSTANT_BYTES: u32 = 8;
/// The ninth state capture is the complete brain buffer.
const BRAIN_BUFFER: usize = 8;
const ENCODER_LIMIT: f64 = 2.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

struct CacheVariant {
    parked: Option<global_credit::Pipelines>,
    storage: wgpu::Buffer,
    metadata_offset: u64,
    metadata_bytes_per_agent: usize,
    zeroes: Vec<u8>,
}

#[derive(Default)]
struct CacheStats {
    valid_blocks: usize,
    skipped_updates: u64,
}

fn candidate_source() -> String {
    let world = include_str!("../shaders/kernel/global_tick.wgsl");
    const ENTRY: &str = "@compute @workgroup_size(256)\nfn global_tick(@builtin(local_invocation_id) lid: vec3u) {\n    let tid = lid.x;";
    assert_eq!(world.matches(ENTRY).count(), 1);
    let world = world.replacen(ENTRY, "fn global_world_inner(tid: u32) {", 1);
    [
        include_str!("../shaders/kernel/common.wgsl"),
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("../shaders/kernel/phase_grid_order.wgsl"),
        include_str!("../shaders/kernel/phase_collision.wgsl"),
        include_str!("../shaders/kernel/phase_trail_sample.wgsl"),
        world.as_str(),
        include_str!("../shaders/kernel/phase_encoder_credit.wgsl"),
        include_str!("../shaders/kernel/encoder_noop_cache.wgsl"),
    ]
    .join("\n")
}

fn make_variant(kernel: &mut GpuKernel) -> CacheVariant {
    let passes = predictor_width::wider_predictor(
        &dense_prefetch::prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH_FACTOR,
        ),
        PREDICTOR_LANES,
    );
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    kernel.global_credit = global_credit::Pipelines::new(kernel, &passes, &constants);
    assert!(kernel.global_credit_active());
    let mut candidate = global_credit::Pipelines::new(kernel, &passes, &constants).unwrap();
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("encoder_noop_cache_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("encoder_noop_cache_tick"),
            source: wgpu::ShaderSource::Wgsl(candidate_source().into()),
        });
    candidate.global = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("encoder_noop_cache_tick"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("encoder_noop_cache_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
    let blocks_per_agent = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap()
        / BLOCK_WIDTH;
    let metadata_bytes_per_agent = blocks_per_agent
        .checked_mul(RECORD_WORDS * WORD_BYTES)
        .unwrap();
    let metadata_bytes = metadata_bytes_per_agent
        .checked_mul(usize::try_from(kernel.agent_count).unwrap())
        .unwrap();
    let metadata_offset = kernel.brain_scratch_buffer.size();
    let size = metadata_offset
        .checked_add(u64::try_from(metadata_bytes).unwrap())
        .unwrap();
    let storage = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("encoder_noop_private_features_and_metadata"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    candidate.bind_groups = std::array::from_fn(|index| {
        make_transient_bind_group(kernel, &bind_layout, &storage, index)
    });
    let variant = CacheVariant {
        parked: Some(candidate),
        storage,
        metadata_offset,
        metadata_bytes_per_agent,
        zeroes: vec![0; metadata_bytes],
    };
    variant.invalidate_all(kernel);
    variant
}

impl CacheVariant {
    fn invalidate_all(&self, kernel: &GpuKernel) {
        kernel
            .queue
            .write_buffer(&self.storage, self.metadata_offset, &self.zeroes);
    }

    fn invalidate_agent(&self, kernel: &GpuKernel, agent: u32) {
        assert!(agent < kernel.agent_count);
        let offset = usize::try_from(agent)
            .unwrap()
            .checked_mul(self.metadata_bytes_per_agent)
            .unwrap();
        kernel.queue.write_buffer(
            &self.storage,
            self.metadata_offset + u64::try_from(offset).unwrap(),
            &self.zeroes[..self.metadata_bytes_per_agent],
        );
    }

    fn snapshot(&self, kernel: &GpuKernel) -> wgpu::Buffer {
        let snapshot = kernel.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoder_noop_private_checkpoint"),
            size: self.storage.size(),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        copy_private(kernel, &self.storage, &snapshot);
        snapshot
    }

    fn check(&self, kernel: &GpuKernel, state: &State) -> TestResult<CacheStats> {
        let bytes = read_buffer(kernel, &self.storage, self.storage.size())?;
        let metadata = &bytes[usize::try_from(self.metadata_offset)?..];
        let brain = as_floats(&state[BRAIN_BUFFER]);
        let blocks_per_agent = self.metadata_bytes_per_agent / (RECORD_WORDS * WORD_BYTES);
        let mut stats = CacheStats::default();
        for (index, record) in as_floats(metadata)
            .as_chunks::<RECORD_WORDS>()
            .0
            .iter()
            .enumerate()
        {
            let gap = record[0];
            let skipped = record[1];
            assert!(
                skipped.is_finite()
                    && (0.0..=BLOCK_WIDTH as f32).contains(&skipped)
                    && skipped.fract() == 0.0
            );
            stats.skipped_updates += skipped as u64;
            if gap == 0.0 {
                continue;
            }
            assert!(gap.is_normal() && gap > 0.0);
            stats.valid_blocks += 1;
            let agent = index / blocks_per_agent;
            let block = index % blocks_per_agent;
            let first = agent * kernel.layout.brain_stride + block * BLOCK_WIDTH;
            for &weight in &brain[first..first + BLOCK_WIDTH] {
                assert!(weight.is_normal() && f64::from(weight).abs() <= ENCODER_LIMIT);
                let above = f64::from(weight.next_up()) - f64::from(weight);
                let below = f64::from(weight) - f64::from(weight.next_down());
                assert!(
                    f64::from(gap) <= above.min(below) / 2.0,
                    "cached half-gap {gap:e} overstates current weight {weight:e}"
                );
            }
        }
        Ok(stats)
    }
}

fn copy_private(kernel: &GpuKernel, source: &wgpu::Buffer, target: &wgpu::Buffer) {
    assert_eq!(source.size(), target.size());
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

fn as_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<WORD_BYTES>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn advance(
    kernel: &mut GpuKernel,
    cache: &mut CacheVariant,
    enabled: bool,
    cycle: &mut u32,
    cycles: u32,
) {
    // A fallback can update weights without refreshing derived metadata. Clear
    // it before dispatch, including partial/global-skipping probe routes.
    if !kernel.global_credit_active() {
        cache.invalidate_all(kernel);
    }
    if enabled {
        std::mem::swap(&mut kernel.global_credit, &mut cache.parked);
    }
    kernel.dispatch_ticks(
        u64::from(*cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
    if enabled {
        std::mem::swap(&mut kernel.global_credit, &mut cache.parked);
    }
    *cycle += cycles;
}

fn trajectory(
    kernel: &mut GpuKernel,
    cache: &mut CacheVariant,
    enabled: bool,
) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for cycles in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cache, enabled, &mut cycle, cycles);
        let state = capture_state(kernel)?;
        if enabled {
            let stats = cache.check(kernel, &state)?;
            if cycle == CHUNKS[0] {
                assert_eq!(
                    stats.skipped_updates, 0,
                    "cold cache executes every original attempted update"
                );
                assert!(stats.valid_blocks > 0);
            }
            println!(
                "ENCODER_NOOP_CACHE_STATE cycles={cycle} valid_blocks={} skipped_last_cycle={}",
                stats.valid_blocks, stats.skipped_updates
            );
        }
        states.push(state);
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU; run explicitly in release mode with --ignored --nocapture"]
fn encoder_noop_cache_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut cache = make_variant(&mut kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, &mut cache, false)?;
        restore(&mut kernel, &saved);
        cache.invalidate_all(&kernel);
        let actual = trajectory(&mut kernel, &mut cache, true)?;
        for (reference, candidate) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, reference, candidate);
            assert_inactive_agent_unchanged(
                &kernel,
                &initial,
                candidate,
                INACTIVE_AGENT,
                "encoder no-op cache",
            );
        }
        assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
        restore(&mut kernel, &saved);
        cache.invalidate_all(&kernel);
        let replay = trajectory(&mut kernel, &mut cache, true)?;
        for (candidate, replay) in actual.iter().zip(&replay) {
            assert_state_equal(&kernel, candidate, replay);
        }
        println!(
            "ENCODER_NOOP_CACHE_PARITY width={width} height={height} cycles=100 exact_buffers=13 block_width={BLOCK_WIDTH} rounding_scope=observed_compilation cold_cache=true death_refresh=true"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Mutation {
    WriteAgent,
    GuardWeights,
    WriteBatch,
    Reset,
    Split,
    Tiled,
    PartialBrain,
    SkipGlobal,
}

fn write_small_weights(kernel: &GpuKernel, state: &mut AgentBrainState) {
    let end = kernel.layout.feature_count * ENCODED_DIMENSION;
    state.brain_state[..end].fill(HOST_WEIGHT);
    let mean =
        fixed_tail_base(kernel.layout.brain_stride) + O_SENSORY_MEAN - O_PREDICTOR_CONTEXT_WEIGHT;
    state.brain_state[mean] = HOST_VISUAL_MEAN;
}

fn mutate(kernel: &mut GpuKernel, cache: &CacheVariant, mutation: Mutation) {
    match mutation {
        Mutation::WriteAgent => {
            let mut state = kernel.read_agent_state(0);
            write_small_weights(kernel, &mut state);
            kernel.write_agent_state(0, &state);
            cache.invalidate_agent(kernel, 0);
        }
        Mutation::GuardWeights => {
            let mut state = kernel.read_agent_state(0);
            state.brain_state[..GUARD_WEIGHTS.len()].copy_from_slice(&GUARD_WEIGHTS);
            kernel.write_agent_state(0, &state);
            cache.invalidate_agent(kernel, 0);
        }
        Mutation::WriteBatch => {
            let states: Vec<_> = (0..kernel.agent_count)
                .map(|agent| {
                    let mut state = kernel.read_agent_state(agent);
                    write_small_weights(kernel, &mut state);
                    state
                })
                .collect();
            kernel.batch_write_agent_states(states.len(), |index| states[index].clone());
            cache.invalidate_all(kernel);
        }
        Mutation::Reset => {
            let brain = BrainConfig {
                vision_stride: 1,
                ..BrainConfig::default()
            };
            kernel.reset_agents_seeded(&brain, RESET_SEED);
            prepare_boundary_scene(kernel);
            cache.invalidate_all(kernel);
        }
        Mutation::Split => kernel.set_execution_mode(BrainExecutionMode::SplitSerial),
        Mutation::Tiled => kernel.set_execution_mode(BrainExecutionMode::ParallelTiled),
        Mutation::PartialBrain => kernel.probe.kernel_pass_limit = WITHOUT_LEARNING,
        Mutation::SkipGlobal => kernel.set_probe_pass_skips(true, false),
    }
}

fn lifecycle(
    kernel: &mut GpuKernel,
    cache: &mut CacheVariant,
    mutation: Mutation,
    enabled: bool,
) -> TestResult<Vec<State>> {
    let mut cycle = WARMUP_CYCLES;
    advance(kernel, cache, enabled, &mut cycle, LIFECYCLE_CYCLES[0]);
    let mut states = vec![capture_state(kernel)?];
    mutate(kernel, cache, mutation);
    advance(kernel, cache, enabled, &mut cycle, LIFECYCLE_CYCLES[1]);
    let transition_state = capture_state(kernel)?;
    if enabled && !kernel.global_credit_active() {
        let stats = cache.check(kernel, &transition_state)?;
        assert_eq!(stats.valid_blocks, 0, "fallback must invalidate metadata");
        assert_eq!(stats.skipped_updates, 0);
    }
    states.push(transition_state);
    kernel.set_execution_mode(BrainExecutionMode::FusedSerial);
    kernel.probe.kernel_pass_limit = COMPLETE_BRAIN;
    kernel.set_probe_pass_skips(false, false);
    advance(kernel, cache, enabled, &mut cycle, LIFECYCLE_CYCLES[2]);
    let final_state = capture_state(kernel)?;
    if enabled {
        cache.check(kernel, &final_state)?;
    }
    states.push(final_state);
    Ok(states)
}

#[test]
#[ignore = "requires GPU; run explicitly in release mode with --ignored --nocapture"]
fn encoder_noop_cache_preserves_explicit_invalidations() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, true);
    let mut cache = make_variant(&mut kernel);
    let mut cycle = 0;
    advance(&mut kernel, &mut cache, true, &mut cycle, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let metadata = cache.snapshot(&kernel);
    let warm_state = capture_state(&kernel)?;
    let stats = cache.check(&kernel, &warm_state)?;
    assert!(stats.valid_blocks > 0 && stats.skipped_updates > 0);
    for mutation in [
        Mutation::WriteAgent,
        Mutation::GuardWeights,
        Mutation::WriteBatch,
        Mutation::Reset,
        Mutation::Split,
        Mutation::Tiled,
        Mutation::PartialBrain,
        Mutation::SkipGlobal,
    ] {
        restore(&mut kernel, &warm);
        copy_private(&kernel, &metadata, &cache.storage);
        let expected = lifecycle(&mut kernel, &mut cache, mutation, false)?;
        restore(&mut kernel, &warm);
        copy_private(&kernel, &metadata, &cache.storage);
        let actual = lifecycle(&mut kernel, &mut cache, mutation, true)?;
        for (reference, candidate) in expected.iter().zip(&actual) {
            assert_state_equal(&kernel, reference, candidate);
        }
        println!(
            "ENCODER_NOOP_CACHE_LIFECYCLE mutation={mutation:?} exact_buffers=13 explicit_harness_invalidation=true production_invalidation_unimplemented=true"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires GPU; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_encoder_noop_cache_full_cycles() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    let mut cache = make_variant(&mut kernel);
    let mut cycle = 0;
    advance(&mut kernel, &mut cache, false, &mut cycle, WARMUP_CYCLES);
    advance(
        &mut kernel,
        &mut cache,
        true,
        &mut cycle,
        CACHE_WARMUP_CYCLES,
    );
    let start_cycle = cycle;
    let warm = checkpoint(&kernel);
    let metadata = cache.snapshot(&kernel);
    let warm_state = capture_state(&kernel)?;
    let stats = cache.check(&kernel, &warm_state)?;
    assert!(stats.valid_blocks > 0);
    advance(&mut kernel, &mut cache, false, &mut cycle, TIMED_CYCLES);
    let expected = capture_state(&kernel)?;
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for round in 0..TIMING_PAIRS {
        for offset in 0..samples.len() {
            let arm = (round + offset) % samples.len();
            restore(&mut kernel, &warm);
            copy_private(&kernel, &metadata, &cache.storage);
            cycle = start_cycle;
            let started = Instant::now();
            advance(&mut kernel, &mut cache, arm != 0, &mut cycle, TIMED_CYCLES);
            samples[arm].push(started.elapsed().as_secs_f64());
            let actual = capture_state(&kernel)?;
            assert_state_equal(&kernel, &expected, &actual);
            if arm != 0 {
                let stats = cache.check(&kernel, &actual)?;
                assert!(
                    stats.skipped_updates > 0,
                    "candidate must skip real attempted updates"
                );
            }
        }
    }
    for times in &mut samples {
        times.sort_by(f64::total_cmp);
    }
    let baseline = samples[0][TIMING_PAIRS / 2];
    let candidate = samples[1][TIMING_PAIRS / 2];
    println!(
        "ENCODER_NOOP_CACHE_TIMING cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} baseline=production_global_credit baseline_seconds={baseline:.9} candidate_seconds={candidate:.9} speedup={:.3} exact_buffers=13 block_width={BLOCK_WIDTH} warmup_cycles={WARMUP_CYCLES} metadata_bytes={} dispatches_per_cycle=4 rounding_scope=observed_compilation",
        baseline / candidate,
        cache.zeroes.len()
    );
    Ok(())
}
