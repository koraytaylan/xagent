//! Test-only numerical diagnostics for FP32 schedule experiments. Full-state
//! comparisons enforce layout and semantic invariants, then report drift;
//! they do not impose an undocumented tolerance on nonlinear trajectories.
//! Isolated dot checks separately use a derived forward-error bound against
//! f64 arithmetic. Passing a state report is not an f64 accuracy proof.

use super::*;

/// The capture_state helper returns these thirteen buffers in this order.
const BUFFER_COUNT: usize = 13;
/// GPU buffer words contain either an IEEE binary32 value or a 32-bit integer.
const WORD_BYTES: usize = std::mem::size_of::<u32>();
/// Named subregions prevent large weight buffers from hiding motor drift.
const REGION_NAMES: [&str; 20] = [
    "physics_other",
    "decisions_other",
    "food",
    "food_flags_and_claims",
    "food_grid",
    "agent_grid_and_masks",
    "collision_scratch",
    "sensory",
    "brain_other",
    "brain_scratch",
    "patterns",
    "trail_ring",
    "sensory_next",
    "encoder_weights",
    "predictor_weights",
    "predictions",
    "previous_encoding",
    "motor_decisions",
    "positions",
    "energy_and_integrity",
];
/// Indices mirror cycle_profile::state_buffers, not shader binding numbers.
const PHYSICS: usize = 0;
const DECISIONS: usize = 1;
const FOOD: usize = 2;
const FLAGS: usize = 3;
const FOOD_GRID: usize = 4;
const AGENT_GRID: usize = 5;
const COLLISION: usize = 6;
const BRAIN: usize = 8;
const PATTERNS: usize = 10;
const TRAIL: usize = 11;
const ENCODER_REGION: usize = 13;
const PREDICTOR_REGION: usize = 14;
const PREDICTIONS_REGION: usize = 15;
const ENCODING_REGION: usize = 16;
const MOTOR_REGION: usize = 17;
const POSITION_REGION: usize = 18;
const ENERGY_REGION: usize = 19;
/// Inline predictor training clamps every updated weight to this interval.
const PREDICTOR_WEIGHT_LIMIT: f32 = 3.0;
/// Encoder-credit updates clamp weights to this interval.
const ENCODER_WEIGHT_LIMIT: f32 = 2.0;
/// Turning applies a clamped multiplier after its initial unit clamp.
const MOTOR_TURN_LIMIT: f32 = 3.0;
/// Binary32 round-to-nearest unit roundoff, half its machine epsilon.
const FP32_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
/// A separate multiply and add is a conservative dot-operation count.
const DOT_OPERATIONS_PER_TERM: u32 = 2;
/// Summary fields retain natural simulation units rather than one global scale.
const BEHAVIOR_NAMES: [&str; 12] = [
    "alive",
    "deaths",
    "food_eaten",
    "energy",
    "integrity",
    "distance",
    "energy_spent",
    "danger_distance",
    "hazard_entries",
    "abs_motor_forward",
    "abs_motor_turn",
    "prediction_error",
];

/// Descriptive numerical differences; no field is an implicit acceptance limit.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RoundingMetrics {
    pub max_abs: f64,
    pub rms: f64,
    /// Symmetric L2 error divided by the larger input L2 norm; zero for two zeros.
    pub normalized_l2: f64,
    pub changed_float_words: usize,
    pub changed_discrete_words: usize,
}

#[derive(Clone, Copy, Default)]
struct Accumulator {
    words: u32,
    changed: usize,
    max_abs: f64,
    squared_error: f64,
    reference_squared: f64,
    candidate_squared: f64,
}

impl Accumulator {
    fn add(&mut self, reference: f32, candidate: f32) {
        self.words += 1;
        self.changed += usize::from(reference.to_bits() != candidate.to_bits());
        let reference = f64::from(reference);
        let candidate = f64::from(candidate);
        let error = candidate - reference;
        self.max_abs = self.max_abs.max(error.abs());
        self.squared_error += error * error;
        self.reference_squared += reference * reference;
        self.candidate_squared += candidate * candidate;
    }

    fn metrics(self) -> RoundingMetrics {
        let scale = self.reference_squared.max(self.candidate_squared);
        RoundingMetrics {
            max_abs: self.max_abs,
            rms: if self.words == 0 {
                0.0
            } else {
                (self.squared_error / f64::from(self.words)).sqrt()
            },
            normalized_l2: if scale == 0.0 {
                0.0
            } else {
                (self.squared_error / scale).sqrt()
            },
            changed_float_words: self.changed,
            changed_discrete_words: 0,
        }
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let offset = index.checked_mul(WORD_BYTES).unwrap();
    u32::from_le_bytes(bytes[offset..offset + WORD_BYTES].try_into().unwrap())
}

fn value(bytes: &[u8], index: usize) -> f32 {
    f32::from_bits(word(bytes, index))
}

fn tail(kernel: &GpuKernel, offset: usize) -> usize {
    fixed_tail_base(kernel.layout.brain_stride) + offset - O_PREDICTOR_CONTEXT_WEIGHT
}

fn state_sizes(kernel: &GpuKernel) -> [u64; BUFFER_COUNT] {
    [
        &kernel.agent_phys_buffer,
        &kernel.decision_buffer,
        &kernel.food_state_buffer,
        &kernel.food_flags_buffer,
        &kernel.food_grid_buffer,
        &kernel.agent_grid_buffer,
        &kernel.collision_scratch_buffer,
        &kernel.sensory_buffer,
        &kernel.brain_state_buffer,
        &kernel.brain_scratch_buffer,
        &kernel.pattern_buffer,
        &kernel.trail_ring_buffer,
        &kernel._sensory_next_buffer,
    ]
    .map(wgpu::Buffer::size)
}

fn is_integer_storage(kernel: &GpuKernel, buffer: usize, index: usize) -> bool {
    if (FLAGS..=COLLISION).contains(&buffer) {
        return true;
    }
    let agents = usize::try_from(kernel.agent_count).unwrap();
    buffer == TRAIL && index % ((agents + 1) * TRAIL_RECORD_STRIDE) == agents * TRAIL_RECORD_STRIDE
}

fn discrete_float(kernel: &GpuKernel, buffer: usize, index: usize) -> bool {
    if buffer == PHYSICS {
        return [
            P_ALIVE,
            P_DIED_FLAG,
            P_FOOD_COUNT,
            P_TICKS_ALIVE,
            P_DEATH_COUNT,
            P_LAST_DEATH_TICK,
            P_IN_DANGER_BIOME,
            P_AVOIDANCE_SENSE_RANGE_TICKS,
            P_AVOIDANCE_TURNS_OPPOSING,
            P_APPROACH_SENSE_RANGE_TICKS,
            P_APPROACH_TURNS_TOWARD,
            P_HAZARD_ENTRIES,
            P_FOOD_CLAIM,
        ]
        .contains(&(index % PHYS_STRIDE));
    }
    if buffer == PATTERNS {
        return index % PATTERN_STRIDE >= O_PAT_META;
    }
    if buffer == BRAIN {
        let offset = index % kernel.layout.brain_stride;
        return [
            O_PREDICTION_ERROR_CURSOR,
            O_PREDICTION_ERROR_COUNT,
            O_POS_RING_CURSOR,
            O_POS_RING_LEN,
            O_TICK_COUNT,
            O_SALIENCE_LABEL,
        ]
        .iter()
        .any(|&field| offset == tail(kernel, field))
            || (tail(kernel, O_RECENT_TICKS)..tail(kernel, O_RECENT_NORMS)).contains(&offset);
    }
    buffer == TRAIL && index % TRAIL_RECORD_STRIDE == TRAIL_RECORD_STRIDE - 1
}

fn region(kernel: &GpuKernel, buffer: usize, index: usize) -> usize {
    match buffer {
        PHYSICS => match index % PHYS_STRIDE {
            P_POS_X..=P_POS_Z => POSITION_REGION,
            P_ENERGY | P_INTEGRITY | P_MAX_ENERGY | P_MAX_INTEGRITY => ENERGY_REGION,
            _ => buffer,
        },
        DECISIONS => {
            let offset = index % DECISION_STRIDE;
            if offset < DECISION_CREDIT {
                PREDICTIONS_REGION
            } else if offset == DECISION_MOTOR || offset == DECISION_MOTOR + 1 {
                MOTOR_REGION
            } else {
                buffer
            }
        }
        BRAIN => {
            let offset = index % kernel.layout.brain_stride;
            let encoder_end = kernel.layout.feature_count * ENCODED_DIMENSION;
            if offset < encoder_end {
                ENCODER_REGION
            } else if (encoder_end + ENCODED_DIMENSION..fixed_tail_base(kernel.layout.brain_stride))
                .contains(&offset)
            {
                PREDICTOR_REGION
            } else if (tail(kernel, O_PREV_PREDICTION)..tail(kernel, O_TICK_COUNT))
                .contains(&offset)
            {
                PREDICTIONS_REGION
            } else if (tail(kernel, O_PREV_ENCODED)..tail(kernel, O_HOMEO)).contains(&offset) {
                ENCODING_REGION
            } else {
                buffer
            }
        }
        _ => buffer,
    }
}

fn assert_integer(value: f32, maximum: f64, label: &str) {
    assert!(
        value >= 0.0 && f64::from(value) <= maximum && value.fract() == 0.0,
        "{label}: invalid nonnegative integer {value}, maximum {maximum}"
    );
}

fn validate_agents(kernel: &GpuKernel, state: &[Vec<u8>], label: &str) {
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * PHYS_STRIDE;
        let physics = |offset| value(&state[PHYSICS], base + offset);
        for flag in [P_ALIVE, P_DIED_FLAG, P_IN_DANGER_BIOME] {
            assert_integer(physics(flag), 1.0, label);
        }
        // Food settlement follows the physics clamp and can leave energy
        // above its nominal maximum until the next physics update.
        for (meter, maximum) in [(P_ENERGY, P_MAX_ENERGY), (P_INTEGRITY, P_MAX_INTEGRITY)] {
            assert!(
                physics(maximum) > 0.0 && physics(meter) >= 0.0,
                "{label}: agent {agent} meter {meter}={} maximum={}",
                physics(meter),
                physics(maximum)
            );
        }
        assert!(
            physics(P_INTEGRITY) <= physics(P_MAX_INTEGRITY),
            "{label}: integrity clamp"
        );
        let decision_base = agent * DECISION_STRIDE + DECISION_MOTOR;
        assert!(
            value(&state[DECISIONS], decision_base).abs() <= 1.0,
            "{label}: forward clamp"
        );
        assert!(
            value(&state[DECISIONS], decision_base + 1).abs() <= MOTOR_TURN_LIMIT,
            "{label}: turn clamp"
        );
        assert_eq!(
            value(&state[DECISIONS], decision_base + 2),
            0.0,
            "{label}: reserved strafe"
        );
        let brain_base = agent * kernel.layout.brain_stride;
        for (offset, count) in [
            (O_PREDICTION_ERROR_CURSOR, ERROR_HISTORY_LEN - 1),
            (O_PREDICTION_ERROR_COUNT, ERROR_HISTORY_LEN),
            (O_POS_RING_CURSOR, POS_RING_LEN - 1),
            (O_POS_RING_LEN, POS_RING_LEN),
        ] {
            assert_integer(
                value(&state[BRAIN], brain_base + tail(kernel, offset)),
                f64::from(u32::try_from(count).unwrap()),
                label,
            );
        }
        let pattern_base = agent * PATTERN_STRIDE;
        for pattern in 0..MEMORY_CAP {
            assert_integer(
                value(&state[PATTERNS], pattern_base + O_PAT_ACTIVE + pattern),
                1.0,
                label,
            );
            assert!(
                value(&state[PATTERNS], pattern_base + O_PAT_NORMS + pattern) >= 0.0,
                "{label}: negative pattern norm"
            );
        }
        assert_integer(
            value(&state[PATTERNS], pattern_base + O_ACTIVE_COUNT),
            f64::from(u32::try_from(MEMORY_CAP).unwrap()),
            label,
        );
    }
}

fn validate_state(kernel: &GpuKernel, state: &[Vec<u8>], label: &str) {
    assert_eq!(state.len(), BUFFER_COUNT, "{label}: captured buffer count");
    for (buffer, (bytes, expected)) in state.iter().zip(state_sizes(kernel)).enumerate() {
        assert_eq!(
            u64::try_from(bytes.len()).unwrap(),
            expected,
            "{label}: buffer {buffer} size"
        );
        assert_eq!(bytes.len() % WORD_BYTES, 0);
        for index in 0..bytes.len() / WORD_BYTES {
            if !is_integer_storage(kernel, buffer, index) {
                let number = value(bytes, index);
                assert!(
                    number.is_finite(),
                    "{label}: nonfinite buffer={buffer} word={index} bits={:08x}",
                    word(bytes, index)
                );
                if region(kernel, buffer, index) == ENCODER_REGION {
                    assert!(
                        number.abs() <= ENCODER_WEIGHT_LIMIT,
                        "{label}: encoder clamp at {index}: {number}"
                    );
                } else if region(kernel, buffer, index) == PREDICTOR_REGION {
                    assert!(
                        number.abs() <= PREDICTOR_WEIGHT_LIMIT,
                        "{label}: predictor clamp at {index}: {number}"
                    );
                }
            }
        }
    }
    for item in 0..kernel.food_count {
        assert!(
            word(&state[FLAGS], item) <= 1,
            "{label}: food consumed flag"
        );
        let claimant = word(&state[FLAGS], kernel.food_count + item);
        assert!(
            claimant == u32::MAX || claimant < kernel.agent_count,
            "{label}: food claimant"
        );
        assert!(
            value(&state[FOOD], item * FOOD_STATE_STRIDE + FOOD_RESPAWN_TIMER) >= 0.0,
            "{label}: respawn timer"
        );
    }
    for (buffer, stride, population) in [
        (FOOD_GRID, FOOD_GRID_CELL_STRIDE, kernel.food_count),
        (
            AGENT_GRID,
            AGENT_GRID_CELL_STRIDE,
            usize::try_from(kernel.agent_count).unwrap(),
        ),
    ] {
        let words = state[buffer].len() / WORD_BYTES;
        let cells = if buffer == AGENT_GRID {
            words / (stride + 1)
        } else {
            words / stride
        };
        for cell in 0..cells {
            let count = usize::try_from(word(&state[buffer], cell * stride)).unwrap();
            assert!(
                count <= population,
                "{label}: grid count exceeds population"
            );
            for slot in 0..count.min(stride - 1) {
                assert!(
                    usize::try_from(word(&state[buffer], cell * stride + 1 + slot)).unwrap()
                        < population,
                    "{label}: retained grid index out of bounds"
                );
            }
        }
    }
    validate_agents(kernel, state, label);
}

/// Validate both states and report numerical/discrete drift without a global epsilon.
/// Near-tie branch causes cannot be inferred from final buffers, so output labels
/// them as uninstrumented rather than claiming every discrete difference is a tie.
pub(super) fn compare_rounding_state(
    kernel: &GpuKernel,
    reference: &[Vec<u8>],
    candidate: &[Vec<u8>],
    label: &str,
) -> RoundingMetrics {
    validate_state(kernel, reference, &format!("{label}/reference"));
    validate_state(kernel, candidate, &format!("{label}/candidate"));
    let mut regions = [Accumulator::default(); REGION_NAMES.len()];
    let mut combined = Accumulator::default();
    let mut changed_discrete_words = 0;
    for (buffer, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        for index in 0..reference.len() / WORD_BYTES {
            if is_integer_storage(kernel, buffer, index) || discrete_float(kernel, buffer, index) {
                changed_discrete_words +=
                    usize::from(word(reference, index) != word(candidate, index));
            } else {
                let expected = value(reference, index);
                let actual = value(candidate, index);
                regions[region(kernel, buffer, index)].add(expected, actual);
                combined.add(expected, actual);
            }
        }
    }
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        for field in [
            P_MAX_ENERGY,
            P_MAX_INTEGRITY,
            P_MEMORY_CAP,
            P_PROCESSING_SLOTS,
        ] {
            let index = agent * PHYS_STRIDE + field;
            assert_eq!(
                word(&reference[PHYSICS], index),
                word(&candidate[PHYSICS], index),
                "{label}: immutable physical field {field}"
            );
        }
    }
    for (name, accumulator) in REGION_NAMES.iter().zip(regions) {
        if accumulator.changed > 0 {
            let metrics = accumulator.metrics();
            println!("ROUNDING_REGION label={label} region={name} changed={} words={} max_abs={:.9e} rms={:.9e} normalized_l2={:.9e}", accumulator.changed, accumulator.words, metrics.max_abs, metrics.rms, metrics.normalized_l2);
        }
    }
    let mut metrics = combined.metrics();
    metrics.changed_discrete_words = changed_discrete_words;
    println!("ROUNDING_STATE label={label} changed_floats={} changed_discrete={} max_abs={:.9e} rms={:.9e} normalized_l2={:.9e} invariants=pass branch_ties=uninstrumented accuracy_bound=not_a_pipeline_bound", metrics.changed_float_words, metrics.changed_discrete_words, metrics.max_abs, metrics.rms, metrics.normalized_l2);
    let reference_behavior = extract_behavior(kernel, reference);
    let candidate_behavior = extract_behavior(kernel, candidate);
    print_behavior_pair(label, &reference_behavior, &candidate_behavior);
    metrics
}

/// Require an explicitly identified permanently inactive agent to retain state.
/// Vision may clear inactive sensory output, so only brain and pattern
/// storage are checked. Both snapshots must mark the agent dead without a reset.
pub(super) fn assert_inactive_agent_unchanged(
    kernel: &GpuKernel,
    initial: &[Vec<u8>],
    actual: &[Vec<u8>],
    agent: u32,
    label: &str,
) {
    assert!(agent < kernel.agent_count);
    let agent = usize::try_from(agent).unwrap();
    for state in [initial, actual] {
        assert_eq!(value(&state[PHYSICS], agent * PHYS_STRIDE + P_ALIVE), 0.0);
        assert_eq!(
            value(&state[PHYSICS], agent * PHYS_STRIDE + P_DIED_FLAG),
            0.0
        );
    }
    for (buffer, stride) in [
        (BRAIN, kernel.layout.brain_stride),
        (PATTERNS, PATTERN_STRIDE),
    ] {
        let begin = agent * stride * WORD_BYTES;
        let end = begin + stride * WORD_BYTES;
        assert_eq!(
            initial[buffer][begin..end],
            actual[buffer][begin..end],
            "{label}: inactive agent {agent} buffer {buffer}"
        );
    }
}

/// Population totals in the natural units named by BEHAVIOR_NAMES.
#[derive(Clone, Copy, Debug)]
pub(super) struct BehaviorSummary {
    pub values: [f64; BEHAVIOR_NAMES.len()],
}

/// Extract comparable behavior outcomes; this performs no GPU readback.
pub(super) fn extract_behavior(kernel: &GpuKernel, state: &[Vec<u8>]) -> BehaviorSummary {
    let fields = [
        P_ALIVE,
        P_DEATH_COUNT,
        P_FOOD_COUNT,
        P_ENERGY,
        P_INTEGRITY,
        P_DISTANCE_TRAVELED,
        P_ENERGY_SPENT,
        P_DANGER_PATH_LENGTH,
        P_HAZARD_ENTRIES,
        P_MOTOR_FWD_OUT,
        P_MOTOR_TURN_OUT,
        P_PREDICTION_ERROR,
    ];
    let mut values = [0.0; BEHAVIOR_NAMES.len()];
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        for (index, field) in fields.iter().enumerate() {
            let number = f64::from(value(&state[PHYSICS], agent * PHYS_STRIDE + field));
            values[index] += if [P_MOTOR_FWD_OUT, P_MOTOR_TURN_OUT].contains(field) {
                number.abs()
            } else {
                number
            };
        }
    }
    assert!(values.iter().all(|number| number.is_finite()));
    BehaviorSummary { values }
}

fn print_behavior_pair(label: &str, reference: &BehaviorSummary, candidate: &BehaviorSummary) {
    for (index, name) in BEHAVIOR_NAMES.iter().enumerate() {
        println!("ROUNDING_BEHAVIOR label={label} metric={name} reference={:.9e} candidate={:.9e} delta={:.9e}", reference.values[index], candidate.values[index], candidate.values[index] - reference.values[index]);
    }
}

/// Report matched seeds and empirical distribution drift, not statistical equivalence.
/// An explicit caller-owned practical margin is required before promoting a
/// numerical schedule; lack of a detected difference is not an acceptance test.
pub(super) fn report_seeded_behavior(
    samples: &[(u64, BehaviorSummary, BehaviorSummary)],
    label: &str,
) {
    assert!(
        !samples.is_empty(),
        "{label}: at least one seed is required"
    );
    let mut seeds = std::collections::HashSet::new();
    for (seed, reference, candidate) in samples {
        assert!(seeds.insert(seed), "{label}: duplicate seed {seed}");
        assert!(reference
            .values
            .iter()
            .chain(&candidate.values)
            .all(|number| number.is_finite()));
        print_behavior_pair(&format!("{label}/seed={seed}"), reference, candidate);
    }
    let count = f64::from(u32::try_from(samples.len()).unwrap());
    for (index, name) in BEHAVIOR_NAMES.iter().enumerate() {
        let mut reference: Vec<_> = samples
            .iter()
            .map(|sample| sample.1.values[index])
            .collect();
        let mut candidate: Vec<_> = samples
            .iter()
            .map(|sample| sample.2.values[index])
            .collect();
        let mean_reference = reference.iter().sum::<f64>() / count;
        let mean_candidate = candidate.iter().sum::<f64>() / count;
        let paired_rms = (samples
            .iter()
            .map(|sample| (sample.2.values[index] - sample.1.values[index]).powi(2))
            .sum::<f64>()
            / count)
            .sqrt();
        reference.sort_by(f64::total_cmp);
        candidate.sort_by(f64::total_cmp);
        let wasserstein = reference
            .iter()
            .zip(&candidate)
            .map(|(left, right)| (left - right).abs())
            .sum::<f64>()
            / count;
        println!("ROUNDING_SEEDS label={label} metric={name} seeds={} reference_mean={mean_reference:.9e} candidate_mean={mean_candidate:.9e} paired_rms={paired_rms:.9e} empirical_wasserstein={wasserstein:.9e} reference_range=[{:.9e},{:.9e}] candidate_range=[{:.9e},{:.9e}] equivalence=not_asserted", samples.len(), reference[0], reference[reference.len()-1], candidate[0], candidate[candidate.len()-1]);
    }
}

/// Forward error of one supplied FP32 dot, not of downstream nonlinear state.
#[derive(Clone, Copy, Debug)]
pub(super) struct DotMetrics {
    pub reference_f64: f64,
    pub absolute_error: f64,
    pub forward_bound: f64,
    pub absolute_product_sum: f64,
}

/// Reference and error budget without claiming an observed GPU dot value.
#[derive(Clone, Copy, Debug)]
pub(super) struct DotReference {
    pub reference_f64: f64,
    pub forward_bound: f64,
    pub absolute_product_sum: f64,
}

/// Assert the conservative gamma_(2n) dot bound, including f64-reference error
/// and a worst-case binary32 flush-to-zero allowance per operation. Products
/// of two f32 inputs are exact in f64; only the reference summation rounds.
/// This model assumes finite input products and no intermediate FP32 overflow.
pub(super) fn validate_fp32_dot(
    left: &[f32],
    right: &[f32],
    observed: f32,
    label: &str,
) -> DotMetrics {
    let metrics = check_fp32_dot(left, right, observed, label);
    println!("ROUNDING_DOT label={label} terms={} reference_f64={:.9e} absolute_error={:.9e} forward_bound={:.9e} absolute_product_sum={:.9e}", left.len(), metrics.reference_f64, metrics.absolute_error, metrics.forward_bound, metrics.absolute_product_sum);
    metrics
}

/// Check the same derived dot bound without logging every row of a large probe.
pub(super) fn check_fp32_dot(
    left: &[f32],
    right: &[f32],
    observed: f32,
    label: &str,
) -> DotMetrics {
    assert_eq!(left.len(), right.len());
    assert!(observed.is_finite(), "{label}: nonfinite observed dot");
    let DotReference {
        reference_f64,
        forward_bound,
        absolute_product_sum,
    } = fp32_dot_reference(left, right, label);
    let absolute_error = (f64::from(observed) - reference_f64).abs();
    assert!(absolute_error <= forward_bound, "{label}: dot error {absolute_error:.9e} exceeds derived bound {forward_bound:.9e}; f64={reference_f64:.9e}, observed={observed:.9e}");
    DotMetrics {
        reference_f64,
        absolute_error,
        forward_bound,
        absolute_product_sum,
    }
}

/// Compute the existing nearest-rounding/FTZ forward budget and f64 reference.
/// This does not assert that any GPU result or proposed recurrence meets it.
pub(super) fn fp32_dot_reference(left: &[f32], right: &[f32], label: &str) -> DotReference {
    assert_eq!(left.len(), right.len());
    let terms = u32::try_from(left.len()).unwrap();
    let operations = f64::from(terms.checked_mul(DOT_OPERATIONS_PER_TERM).unwrap());
    let scaled_roundoff = operations * FP32_UNIT_ROUNDOFF;
    assert!(
        scaled_roundoff < 1.0,
        "{label}: dot too long for this forward bound"
    );
    let mut reference_f64 = 0.0;
    let mut absolute_product_sum = 0.0;
    let mut input_flush_allowance = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        assert!(
            left.is_finite() && right.is_finite(),
            "{label}: nonfinite dot input"
        );
        let product = f64::from(left) * f64::from(right);
        reference_f64 += product;
        absolute_product_sum += product.abs();
        // Flushing a subnormal input before multiplying by a large normal
        // value can remove a normal-sized product, not just a tiny result.
        if left.is_subnormal() || right.is_subnormal() {
            input_flush_allowance += product.abs();
        }
    }
    assert!(
        absolute_product_sum <= f64::from(f32::MAX),
        "{label}: possible intermediate overflow is outside this bound"
    );
    let double_roundoff = f64::from(terms) * (f64::EPSILON / 2.0);
    let reference_gamma = double_roundoff / (1.0 - double_roundoff);
    let upper_product_sum = absolute_product_sum / (1.0 - double_roundoff);
    let gamma = scaled_roundoff / (1.0 - scaled_roundoff);
    let underflow_allowance = operations * f64::from(f32::MIN_POSITIVE) / (1.0 - scaled_roundoff);
    let forward_bound = (gamma + reference_gamma) * upper_product_sum
        + underflow_allowance
        + input_flush_allowance / (1.0 - double_roundoff);
    DotReference {
        reference_f64,
        forward_bound,
        absolute_product_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_bound_accepts_cancellation_rounding() {
        let left = [100_000_000.0, 1.0, -100_000_000.0];
        let right = [1.0; 3];
        let serial = left
            .iter()
            .zip(right)
            .fold(0.0, |sum, (left, right)| sum + left * right);
        let regrouped = (left[0] + left[2]) + left[1];
        assert_ne!(serial, regrouped);
        validate_fp32_dot(&left, &right, serial, "serial_cancellation");
        validate_fp32_dot(&left, &right, regrouped, "regrouped_cancellation");
    }

    #[test]
    #[should_panic(expected = "exceeds derived bound")]
    fn dot_bound_rejects_missing_contribution() {
        validate_fp32_dot(&[1.0, 1.0], &[1.0, 1.0], 1.0, "missing_term");
    }

    #[test]
    fn seeded_report_accepts_distinct_finite_seeds() {
        let summary = BehaviorSummary {
            values: [0.0; BEHAVIOR_NAMES.len()],
        };
        report_seeded_behavior(
            &[(0, summary, summary), (1, summary, summary)],
            "same_behavior",
        );
    }
}
