//! Observe unchanged packed encoder vectors from ordinary GPU snapshots.
//! No counters or branches are inserted into a shader. Decision credit after
//! the cycle supplies the exact gates consumed by global credit; death and
//! inactive transitions are excluded from the store opportunity denominator.
//! These counts describe the supplied trajectories, not a timing prediction.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{advance, prepare_kernel};
use super::vision_validation::read_buffer;
use super::*;

/// Match the independent suppression benchmark's ordinary raw-vision field.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
/// Independent learned trajectories share the identical initial world.
const SEEDS: [u64; 3] = [42, 314, 2026];
/// Observe both the populated-memory and later evolving benchmark states.
const WINDOW_STARTS: [u32; 2] = [256, 1_000];
/// Consecutive samples avoid treating a single sparse update as representative.
const WINDOW_CYCLES: u32 = 16;
/// Each packed invocation owns four adjacent feature-major output weights.
const VECTOR_WIDTH: usize = 4;
const WORD_BYTES: usize = size_of::<f32>();
const PHYSICS: usize = 0;
const DECISIONS: usize = 1;
const BRAIN: usize = 8;
const MUTABLE_BUFFERS: usize = 13;
const ALIVE_THRESHOLD: f32 = 0.5;
const PERCENT_SCALE: f64 = 100.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Default)]
struct Counts {
    eligible_agent_cycles: u64,
    inactive_agent_cycles: u64,
    transition_agent_cycles: u64,
    credit_disabled_vectors: u64,
    attempted_vectors: u64,
    /// Index is the number of changed components in one attempted vector.
    changed_components_per_vector: [u64; VECTOR_WIDTH + 1],
    enabled_components: u64,
    unchanged_enabled_components: u64,
}

impl Counts {
    fn merge(&mut self, other: &Self) {
        self.eligible_agent_cycles += other.eligible_agent_cycles;
        self.inactive_agent_cycles += other.inactive_agent_cycles;
        self.transition_agent_cycles += other.transition_agent_cycles;
        self.credit_disabled_vectors += other.credit_disabled_vectors;
        self.attempted_vectors += other.attempted_vectors;
        self.enabled_components += other.enabled_components;
        self.unchanged_enabled_components += other.unchanged_enabled_components;
        for (total, count) in self
            .changed_components_per_vector
            .iter_mut()
            .zip(other.changed_components_per_vector)
        {
            *total += count;
        }
    }

    fn report(self, label: &str) {
        let unchanged = self.changed_components_per_vector[0];
        let changed = self.attempted_vectors - unchanged;
        assert_eq!(
            self.changed_components_per_vector.iter().sum::<u64>(),
            self.attempted_vectors
        );
        let unchanged_percent = if self.attempted_vectors == 0 {
            "n/a".to_owned()
        } else {
            format!(
                "{:.6}",
                PERCENT_SCALE * unchanged as f64 / self.attempted_vectors as f64
            )
        };
        println!("PACKED_STORE_OPPORTUNITY {label} eligible_agent_cycles={} inactive_agent_cycles={} death_or_alive_transitions_excluded={} vectors_already_credit_disabled={} attempted_vectors={} all_four_bits_unchanged={} any_component_changed={} unchanged_vector_percent={} enabled_components={} unchanged_enabled_components={} changed_components_histogram={:?} after_cycle_credit=true private_scalar_mirror_exact=true replay_exact_buffers={MUTABLE_BUFFERS} shader_instrumentation=false timing_not_measured=true",
            self.eligible_agent_cycles, self.inactive_agent_cycles,
            self.transition_agent_cycles, self.credit_disabled_vectors,
            self.attempted_vectors, unchanged, changed, unchanged_percent,
            self.enabled_components, self.unchanged_enabled_components,
            self.changed_components_per_vector);
    }
}

fn bits(bytes: &[u8], index: usize) -> u32 {
    let first = index * WORD_BYTES;
    u32::from_le_bytes(bytes[first..first + WORD_BYTES].try_into().unwrap())
}

fn number(bytes: &[u8], index: usize) -> f32 {
    let value = f32::from_bits(bits(bytes, index));
    assert!(value.is_finite());
    value
}

fn credit_epsilon() -> f32 {
    let prefix = "const CREDIT_EPSILON: f32 = ";
    let common = include_str!("../shaders/kernel/common.wgsl");
    let declarations: Vec<_> = common
        .lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .collect();
    assert_eq!(declarations.len(), 1);
    let epsilon: f32 = declarations[0].split(';').next().unwrap().parse().unwrap();
    assert!(epsilon.is_normal() && epsilon > 0.0);
    let credit = packed_encoder::CREDIT_SOURCE;
    assert!(credit.contains("let credit_enabled = abs(credits) >= vec4<f32>(CREDIT_EPSILON);"));
    assert!(credit.contains("if !any(credit_enabled) { return; }"));
    assert!(credit.contains("if physics_state[agent_id * PHYS_STRIDE + P_ALIVE] < 0.5 { return; }"));
    epsilon
}

fn encoder_range(kernel: &GpuKernel, agent: usize) -> std::ops::Range<usize> {
    let first = (agent * kernel.layout.brain_stride + O_ENC_WEIGHTS) * WORD_BYTES;
    first..first + kernel.layout.feature_count * ENCODED_DIMENSION * WORD_BYTES
}

/// Read the actual private vector matrix at both window boundaries, proving
/// that the public mirror used for observations represents those same bits.
fn assert_private_mirror(kernel: &GpuKernel, state: &[Vec<u8>]) -> TestResult {
    let cache = kernel
        .global_credit
        .as_ref()
        .unwrap()
        .packed_encoder
        .as_ref()
        .unwrap();
    assert!(cache.is_valid());
    let packed = read_buffer(kernel, cache.buffer(), cache.buffer().size())?;
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let matrix_bytes = kernel.layout.feature_count * ENCODED_DIMENSION * WORD_BYTES;
    let prefix = packed.len().checked_sub(agents * matrix_bytes).unwrap();
    let scratch_words = kernel.layout.brain_scratch_stride * agents;
    assert_eq!(
        prefix,
        scratch_words.div_ceil(VECTOR_WIDTH) * VECTOR_WIDTH * WORD_BYTES
    );
    for agent in 0..agents {
        let first = prefix + agent * matrix_bytes;
        assert_eq!(
            &packed[first..first + matrix_bytes],
            &state[BRAIN][encoder_range(kernel, agent)],
            "private vector matrix and scalar mirror differ for agent {agent}"
        );
    }
    Ok(())
}

fn observe_vectors(
    kernel: &GpuKernel,
    agent: usize,
    before: &[Vec<u8>],
    after: &[Vec<u8>],
    epsilon: f32,
    counts: &mut Counts,
) {
    let decision_base = agent * DECISION_STRIDE + DECISION_CREDIT;
    let brain_base = agent * kernel.layout.brain_stride + O_ENC_WEIGHTS;
    for dimension in (0..ENCODED_DIMENSION).step_by(VECTOR_WIDTH) {
        let enabled: [bool; VECTOR_WIDTH] = std::array::from_fn(|component| {
            number(&after[DECISIONS], decision_base + dimension + component).abs() >= epsilon
        });
        let any_enabled = enabled.iter().any(|&enabled| enabled);
        for feature in 0..kernel.layout.feature_count {
            let first = brain_base + feature * ENCODED_DIMENSION + dimension;
            let mut changed_components = 0;
            for (component, &enabled) in enabled.iter().enumerate() {
                let previous = bits(&before[BRAIN], first + component);
                let current = bits(&after[BRAIN], first + component);
                assert!(
                    f32::from_bits(previous).is_finite() && f32::from_bits(current).is_finite()
                );
                let unchanged = previous == current;
                if enabled {
                    counts.enabled_components += 1;
                    counts.unchanged_enabled_components += u64::from(unchanged);
                } else {
                    assert!(
                        unchanged,
                        "disabled credit changed agent={agent} feature={feature} dimension={}",
                        dimension + component
                    );
                }
                changed_components += usize::from(!unchanged);
            }
            if any_enabled {
                counts.attempted_vectors += 1;
                counts.changed_components_per_vector[changed_components] += 1;
            } else {
                counts.credit_disabled_vectors += 1;
            }
        }
    }
}

fn observe(kernel: &GpuKernel, before: &[Vec<u8>], after: &[Vec<u8>], epsilon: f32) -> Counts {
    assert_eq!(before.len(), MUTABLE_BUFFERS);
    assert_eq!(after.len(), MUTABLE_BUFFERS);
    let mut counts = Counts::default();
    let tick_offset =
        fixed_tail_base(kernel.layout.brain_stride) + O_TICK_COUNT - O_PREDICTOR_CONTEXT_WEIGHT;
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let physics = agent * PHYS_STRIDE;
        let was_alive = number(&before[PHYSICS], physics + P_ALIVE) >= ALIVE_THRESHOLD;
        let is_alive = number(&after[PHYSICS], physics + P_ALIVE) >= ALIVE_THRESHOLD;
        let death_changed = bits(&before[PHYSICS], physics + P_DEATH_COUNT)
            != bits(&after[PHYSICS], physics + P_DEATH_COUNT);
        if !was_alive && !is_alive && !death_changed {
            counts.inactive_agent_cycles += 1;
            let matrix = encoder_range(kernel, agent);
            assert_eq!(before[BRAIN][matrix.clone()], after[BRAIN][matrix]);
            continue;
        }
        if !was_alive || !is_alive || death_changed {
            counts.transition_agent_cycles += 1;
            continue;
        }
        let tick = agent * kernel.layout.brain_stride + tick_offset;
        assert_eq!(
            number(&after[BRAIN], tick),
            number(&before[BRAIN], tick) + 1.0,
            "one complete brain cycle must separate observations"
        );
        counts.eligible_agent_cycles += 1;
        observe_vectors(kernel, agent, before, after, epsilon, &mut counts);
    }
    assert_eq!(
        counts.eligible_agent_cycles
            + counts.inactive_agent_cycles
            + counts.transition_agent_cycles,
        u64::from(kernel.agent_count)
    );
    let vectors_per_agent = kernel.layout.feature_count * ENCODED_DIMENSION / VECTOR_WIDTH;
    assert_eq!(
        counts.attempted_vectors + counts.credit_disabled_vectors,
        counts.eligible_agent_cycles * u64::try_from(vectors_per_agent).unwrap()
    );
    counts
}

#[test]
#[ignore = "requires GPU; run explicitly in release mode with --ignored --nocapture"]
fn packed_encoder_unchanged_vector_store_opportunity() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel(WIDTH, HEIGHT, false);
    assert!(kernel.global_credit_active());
    assert!(kernel
        .global_credit
        .as_ref()
        .unwrap()
        .packed_encoder
        .is_some());
    let initial_world = checkpoint(&kernel);
    let epsilon = credit_epsilon();
    let brain = BrainConfig {
        vision_width: WIDTH,
        vision_height: HEIGHT,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let mut totals = [Counts::default(); WINDOW_STARTS.len()];
    for seed in SEEDS {
        restore(&mut kernel, &initial_world);
        kernel.reset_agents_seeded(&brain, seed);
        let mut cycle = 0;
        for (window, start) in WINDOW_STARTS.into_iter().enumerate() {
            advance(&mut kernel, cycle, start - cycle);
            cycle = start;
            let mut previous = capture_state(&kernel)?;
            assert_private_mirror(&kernel, &previous)?;
            let mut counts = Counts::default();
            for _ in 0..WINDOW_CYCLES {
                let initial = checkpoint(&kernel);
                advance(&mut kernel, cycle, 1);
                let actual = capture_state(&kernel)?;
                counts.merge(&observe(&kernel, &previous, &actual, epsilon));
                restore(&mut kernel, &initial);
                advance(&mut kernel, cycle, 1);
                assert_state_equal(&kernel, &actual, &capture_state(&kernel)?);
                previous = actual;
                cycle += 1;
            }
            assert_private_mirror(&kernel, &previous)?;
            counts.report(&format!("seed={seed} start_cycle={start} cycles={WINDOW_CYCLES} width={WIDTH} height={HEIGHT}"));
            totals[window].merge(&counts);
        }
    }
    for (start, counts) in WINDOW_STARTS.into_iter().zip(totals) {
        counts.report(&format!("seed=all seeds={} start_cycle={start} cycles_per_seed={WINDOW_CYCLES} width={WIDTH} height={HEIGHT}", SEEDS.len()));
    }
    Ok(())
}
