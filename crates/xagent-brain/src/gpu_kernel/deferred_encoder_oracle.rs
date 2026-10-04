//! Numerical checks for a deferred encoder window, relative to sequential
//! FP32 weight updates. Materialization must reproduce the supplied GPU
//! sequential matrix exactly. Deferred raw dots may differ: their bound
//! includes both arithmetic schedules and weight increments discarded by
//! sequential rounding. A window that may clamp must first materialize.
//!
//! Inputs are already-rounded FP32 scales and adapted features. This module
//! does not certify how a shader computes those inputs, approximate tanh,
//! downstream branches, or an evolving simulation trajectory.
//! Rounding and flush-to-zero allowances follow the WGSL accuracy contract:
//! https://www.w3.org/TR/WGSL/#floating-point-accuracy

/// Encoder credit clamps every updated weight to this interval.
const WEIGHT_LIMIT: f64 = 2.0;
/// WGSL allows either adjacent FP32 rounding result, so use a full epsilon.
const FP32_ERROR: f64 = f32::EPSILON as f64;
/// Flush-to-zero may affect a subnormal input or result.
const NORMAL_MINIMUM: f64 = f32::MIN_POSITIVE as f64;
/// Dot products perform at most one multiply and one add per input.
const DOT_OPERATIONS: f64 = 2.0;

/// One recorded rank-one update. Inactive dimensions retain the original skip
/// decision; their finite scale values are ignored rather than read as updates.
pub(super) struct RankOneStep<'a> {
    pub(super) features: &'a [f32],
    pub(super) scales: &'a [f32],
    pub(super) active: &'a [bool],
}

/// Observed errors and derived allowances, without an arbitrary tolerance.
#[derive(Default, Debug)]
pub(super) struct DeferredMetrics {
    pub(super) max_matrix_rounding_error: f64,
    pub(super) max_raw_dot_error: f64,
    pub(super) max_raw_dot_bound: f64,
    pub(super) max_error_to_bound: f64,
    pub(super) rms_raw_dot_error: f64,
    pub(super) cpu_sequential_differences: usize,
}

/// Symmetric interval for an FP32 expression around an f64 reference value.
#[derive(Clone, Copy)]
struct Bounded {
    value: f64,
    error: f64,
}

impl Bounded {
    fn input(value: f32) -> Self {
        assert!(value.is_finite());
        Self {
            value: f64::from(value),
            error: 0.0,
        }
    }

    fn magnitude(self) -> f64 {
        self.value.abs() + self.error
    }

    fn contains(self, value: f32) -> bool {
        value.is_finite() && (f64::from(value) - self.value).abs() <= self.error
    }

    fn multiply(self, other: Self) -> Self {
        let value = self.value * other.value;
        let propagated = self.error * other.magnitude() + self.value.abs() * other.error;
        // A flushed input can lose a normal-sized product when the other
        // operand is large. Account for that before rounding the product.
        let flush = NORMAL_MINIMUM * (self.magnitude() + other.magnitude())
            + NORMAL_MINIMUM * NORMAL_MINIMUM;
        let error = propagated
            + flush
            + FP32_ERROR * (value.abs() + propagated + flush)
            + NORMAL_MINIMUM
            + f64::EPSILON * value.abs();
        assert!(value.abs() + error <= f64::from(f32::MAX));
        Self { value, error }
    }

    fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        let propagated = self.error + other.error + DOT_OPERATIONS * NORMAL_MINIMUM;
        let error = propagated
            + FP32_ERROR * (value.abs() + propagated)
            + NORMAL_MINIMUM
            + f64::EPSILON * (self.value.abs() + other.value.abs());
        assert!(value.abs() + error <= f64::from(f32::MAX));
        Self { value, error }
    }

    fn clamp(self) -> Self {
        let low = (self.value - self.error).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        let high = (self.value + self.error).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        let value = self.value.clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        Self {
            value,
            error: (value - low).max(high - value),
        }
    }
}

fn gamma(operations: usize, epsilon: f64) -> f64 {
    let scaled = f64::from(u32::try_from(operations).unwrap()) * epsilon;
    assert!(scaled < 1.0);
    scaled / (1.0 - scaled)
}

/// Products of supplied f32 values are exact in f64; the sum can still round.
fn real_dot(left: &[f32], right: &[f32]) -> Bounded {
    assert_eq!(left.len(), right.len());
    let mut value = 0.0;
    let mut magnitude = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        assert!(left.is_finite() && right.is_finite());
        let product = f64::from(left) * f64::from(right);
        value += product;
        magnitude += product.abs();
    }
    let reference_gamma = gamma(left.len(), f64::EPSILON);
    Bounded {
        value,
        error: reference_gamma * magnitude * (1.0 + reference_gamma),
    }
}

/// Any binary reduction of these products has at most n rounding operations
/// on one contribution's path, including its multiply. This covers the
/// original four-lane tree and serial or balanced feature-dot schedules.
fn rounded_dot(left: &[f32], right: &[f32]) -> Bounded {
    let reference = real_dot(left, right);
    let mut magnitude = 0.0;
    let mut flushed_inputs = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let product = (f64::from(left) * f64::from(right)).abs();
        magnitude += product;
        if left.is_subnormal() || right.is_subnormal() {
            flushed_inputs += product;
        }
    }
    let reference_gamma = gamma(left.len(), f64::EPSILON);
    let upper_magnitude = magnitude * (1.0 + reference_gamma);
    let float_gamma = gamma(left.len(), FP32_ERROR);
    let operations = DOT_OPERATIONS * f64::from(u32::try_from(left.len()).unwrap());
    let error = reference.error
        + float_gamma * upper_magnitude
        + operations * NORMAL_MINIMUM * (1.0 + float_gamma)
        + flushed_inputs * (1.0 + reference_gamma);
    assert!(upper_magnitude + error <= f64::from(f32::MAX));
    Bounded {
        value: reference.value,
        error,
    }
}

fn validate_shape(base: &[f32], features: usize, outputs: usize, steps: &[RankOneStep<'_>]) {
    assert!(features > 0 && outputs > 0);
    assert_eq!(base.len(), features.checked_mul(outputs).unwrap());
    assert!(base
        .iter()
        .all(|weight| weight.is_finite() && f64::from(*weight).abs() <= WEIGHT_LIMIT));
    for step in steps {
        assert_eq!(step.features.len(), features);
        assert_eq!(step.scales.len(), outputs);
        assert_eq!(step.active.len(), outputs);
        assert!(step.features.iter().all(|value| value.is_finite()));
        for &scale in step.scales {
            assert!(scale.is_finite());
        }
    }
}

/// The least f32 at least as large as a nonnegative two-term f64 expansion.
/// f32 products are exact in f64; the residual also handles addition of a
/// tiny positive value that an f64 sum alone would discard.
fn positive_ceiling(value: f64, residual: f64) -> Option<f32> {
    assert!(value >= 0.0 && value.is_finite() && residual.is_finite());
    let rounded = value as f32;
    if !rounded.is_finite() {
        return None;
    }
    let represented = f64::from(rounded);
    let needs_next = represented < value || (represented == value && residual > 0.0);
    let upper = if needs_next {
        f32::from_bits(rounded.to_bits() + 1)
    } else {
        rounded
    };
    upper.is_finite().then_some(upper)
}

fn positive_sum_ceiling(left: f32, right: f32) -> Option<f32> {
    let left = f64::from(left);
    let right = f64::from(right);
    let sum = left + right;
    // Error-free TwoSum: both inputs are finite nonnegative f32, so neither
    // f64 overflow nor f64 underflow can occur in this expansion.
    let virtual_right = sum - left;
    let residual = (left - (sum - virtual_right)) + (right - virtual_right);
    positive_ceiling(sum, residual)
}

/// Independently bound every prefix using positive, directed-upward f32
/// arithmetic. This is tighter than inflating by a relative epsilon and no
/// stronger than the shader's next-positive-f32 inflation. It includes the
/// sequential product/add rounding before each clamp; a final in-range real
/// sum alone would not certify the intervening prefixes.
pub(super) fn certify_deferred_window(
    base: &[f32],
    features: usize,
    outputs: usize,
    steps: &[RankOneStep<'_>],
) -> bool {
    validate_shape(base, features, outputs, steps);
    let mut bound = base.iter().copied().map(f32::abs).fold(0.0, f32::max);
    for step in steps {
        let maximum_feature = step
            .features
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        let maximum_scale = step
            .scales
            .iter()
            .zip(step.active)
            .filter(|(_, active)| **active)
            .map(|(scale, _)| scale.abs())
            .fold(0.0, f32::max);
        if maximum_feature == 0.0 || maximum_scale == 0.0 {
            continue;
        }
        let product = f64::from(maximum_feature) * f64::from(maximum_scale);
        let Some(increment) = positive_ceiling(product, 0.0) else {
            return false;
        };
        let Some(next) = positive_sum_ceiling(bound, increment) else {
            return false;
        };
        if f64::from(next) > WEIGHT_LIMIT {
            return false;
        }
        bound = next;
    }
    true
}

/// Explicit scalar CPU schedule: multiply, round, add, round, then clamp at
/// every step. It is an adversarial reference, not a claim about GPU FMA use.
pub(super) fn sequential_fp32_weights(
    base: &[f32],
    features: usize,
    outputs: usize,
    steps: &[RankOneStep<'_>],
) -> Vec<f32> {
    validate_shape(base, features, outputs, steps);
    let mut weights = base.to_vec();
    for step in steps {
        for feature in 0..features {
            for dim in 0..outputs {
                if step.active[dim] {
                    let product = step.scales[dim] * step.features[feature];
                    let index = feature * outputs + dim;
                    weights[index] = (weights[index] + product)
                        .clamp(-(WEIGHT_LIMIT as f32), WEIGHT_LIMIT as f32);
                }
            }
        }
    }
    weights
}

fn real_weight(base: f32, feature: usize, dim: usize, steps: &[RankOneStep<'_>]) -> Bounded {
    let mut value = f64::from(base);
    let mut magnitude = value.abs();
    for step in steps {
        if step.active[dim] {
            let product = f64::from(step.scales[dim]) * f64::from(step.features[feature]);
            value += product;
            magnitude += product.abs();
        }
    }
    let reference_gamma = gamma(steps.len() + 1, f64::EPSILON);
    Bounded {
        value,
        error: reference_gamma * magnitude * (1.0 + reference_gamma),
    }
}

/// Check one certified pending window from its last materialized base. The
/// supplied sequential/materialized matrices are feature-major. Raw dots are
/// before bias, tanh, habituation or any other nonlinear downstream operation.
///
/// After a saturation fallback, pass its updated base and only the subsequent
/// pending factors. An empty window cannot hide a changed base-dot schedule.
pub(super) fn validate_deferred_encoder_prefix(
    base: &[f32],
    features: usize,
    outputs: usize,
    steps: &[RankOneStep<'_>],
    query: &[f32],
    sequential: &[f32],
    materialized: &[f32],
    sequential_dots: &[f32],
    deferred_dots: &[f32],
    label: &str,
) -> DeferredMetrics {
    validate_shape(base, features, outputs, steps);
    assert_eq!(query.len(), features);
    assert_eq!(sequential.len(), base.len());
    assert_eq!(materialized.len(), base.len());
    assert_eq!(sequential_dots.len(), outputs);
    assert_eq!(deferred_dots.len(), outputs);
    assert!(
        certify_deferred_window(base, features, outputs, steps),
        "{label}: an uncertified clamp prefix requires materialization/fallback"
    );
    let cpu = sequential_fp32_weights(base, features, outputs, steps);
    let mut metrics = DeferredMetrics::default();
    let mut dot_squared_error = 0.0;
    for (index, (&sequential, &materialized)) in sequential.iter().zip(materialized).enumerate() {
        assert!(sequential.is_finite() && f64::from(sequential).abs() <= WEIGHT_LIMIT);
        assert_eq!(
            materialized.to_bits(),
            sequential.to_bits(),
            "{label}: materialized weight {index} differs from sequential GPU FP32"
        );
        metrics.cpu_sequential_differences +=
            usize::from(cpu[index].to_bits() != sequential.to_bits());
        let feature = index / outputs;
        let dim = index % outputs;
        let mut envelope = Bounded::input(base[index]);
        for step in steps {
            if step.active[dim] {
                envelope = envelope
                    .add(
                        Bounded::input(step.scales[dim])
                            .multiply(Bounded::input(step.features[feature])),
                    )
                    .clamp();
            }
        }
        assert!(
            envelope.contains(sequential),
            "{label}: sequential weight {index} outside FP32 update envelope"
        );
        let real = real_weight(base[index], feature, dim, steps);
        metrics.max_matrix_rounding_error = metrics
            .max_matrix_rounding_error
            .max((real.value - f64::from(sequential)).abs() + real.error);
    }
    for dim in 0..outputs {
        let base_row: Vec<_> = (0..features)
            .map(|feature| base[feature * outputs + dim])
            .collect();
        let sequential_row: Vec<_> = (0..features)
            .map(|feature| sequential[feature * outputs + dim])
            .collect();
        let mut deferred = rounded_dot(&base_row, query);
        let mut matrix_projection = 0.0;
        for feature in 0..features {
            let real = real_weight(base_row[feature], feature, dim, steps);
            matrix_projection += ((real.value - f64::from(sequential_row[feature])).abs()
                + real.error)
                * f64::from(query[feature]).abs();
        }
        for step in steps {
            if step.active[dim] {
                deferred = deferred.add(
                    Bounded::input(step.scales[dim]).multiply(rounded_dot(step.features, query)),
                );
            }
        }
        let sequential_dot = rounded_dot(&sequential_row, query);
        assert!(
            sequential_dot.contains(sequential_dots[dim]),
            "{label}: sequential dot {dim} outside forward bound"
        );
        assert!(
            deferred.contains(deferred_dots[dim]),
            "{label}: deferred dot {dim} outside its arithmetic bound"
        );
        let reference_gamma = gamma(features.checked_mul(2).unwrap(), f64::EPSILON);
        let bound =
            matrix_projection * (1.0 + reference_gamma) + deferred.error + sequential_dot.error;
        let error = (f64::from(deferred_dots[dim]) - f64::from(sequential_dots[dim])).abs();
        assert!(
            error <= bound,
            "{label}: dot difference {error:e} exceeds bound {bound:e}, dim {dim}"
        );
        if steps.is_empty() {
            assert_eq!(
                deferred_dots[dim].to_bits(),
                sequential_dots[dim].to_bits(),
                "{label}: empty-window raw dot changed its original schedule"
            );
        }
        metrics.max_raw_dot_error = metrics.max_raw_dot_error.max(error);
        dot_squared_error += error * error;
        metrics.max_raw_dot_bound = metrics.max_raw_dot_bound.max(bound);
        if bound > 0.0 {
            metrics.max_error_to_bound = metrics.max_error_to_bound.max(error / bound);
        }
    }
    metrics.rms_raw_dot_error =
        (dot_squared_error / f64::from(u32::try_from(outputs).unwrap())).sqrt();
    println!("DEFERRED_ENCODER_ORACLE label={label} pending={} features={features} outputs={outputs} materialized_bits_equal=true max_matrix_rounding_error={:.9e} max_raw_dot_error={:.9e} rms_raw_dot_error={:.9e} max_raw_dot_bound={:.9e} max_error_to_bound={:.9e} cpu_sequential_differences={} nonlinear_behavior_equivalence=not_asserted", steps.len(), metrics.max_matrix_rounding_error, metrics.max_raw_dot_error, metrics.rms_raw_dot_error, metrics.max_raw_dot_bound, metrics.max_error_to_bound, metrics.cpu_sequential_differences);
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight updates no larger than half the nearer gap disappear at ±1.
    const TINY_STEPS: usize = 8;
    const TINY_SCALE: f32 = 1.0 / 33_554_432.0;
    /// A positive step saturates; the negative step must start at the clamp.
    const SATURATING_BASE: f32 = 1.875;
    const SATURATING_SCALE: f32 = 0.25;

    fn steps<'a>(
        features: &'a [f32],
        scales: &'a [f32],
        active: &'a [bool],
        count: usize,
    ) -> Vec<RankOneStep<'a>> {
        (0..count)
            .map(|_| RankOneStep {
                features,
                scales,
                active,
            })
            .collect()
    }

    #[test]
    fn tiny_steps_expose_lost_sequential_rounding() {
        let base = [1.0, -1.0];
        let query = [1.0, 1.0];
        let updates = steps(&query, &[TINY_SCALE], &[true], TINY_STEPS);
        let sequential = sequential_fp32_weights(&base, query.len(), 1, &updates);
        assert_eq!(sequential, base);
        // The base dot cancels. Each ordered low-rank correction is nonzero,
        // even though every independently stored weight update rounds away.
        let correction = TINY_SCALE * (query[0] + query[1]);
        let deferred = (0..TINY_STEPS).fold(0.0_f32, |sum, _| sum + correction);
        assert!(deferred > 0.0);
        let metrics = validate_deferred_encoder_prefix(
            &base,
            query.len(),
            1,
            &updates,
            &query,
            &sequential,
            &sequential,
            &[0.0],
            &[deferred],
            "tiny_updates",
        );
        assert!(metrics.max_matrix_rounding_error >= f64::from(TINY_SCALE * TINY_STEPS as f32));
        assert!(metrics.max_raw_dot_error > 0.0);
    }

    #[test]
    fn positive_certificate_retains_a_term_lost_even_in_f64() {
        assert_eq!(1.0_f64 + f64::from(f32::MIN_POSITIVE), 1.0);
        assert_eq!(
            positive_sum_ceiling(1.0, f32::MIN_POSITIVE)
                .unwrap()
                .to_bits(),
            1.0_f32.to_bits() + 1
        );
    }

    #[test]
    fn adding_zero_accounts_for_flushed_subnormal_input() {
        let tiny = f32::from_bits(1);
        let sum = Bounded::input(tiny).add(Bounded::input(0.0));
        assert!(sum.contains(tiny));
        assert!(sum.contains(0.0));
    }

    #[test]
    fn saturation_requires_fallback_despite_in_range_final_sum() {
        let updates = [
            RankOneStep {
                features: &[1.0],
                scales: &[SATURATING_SCALE],
                active: &[true],
            },
            RankOneStep {
                features: &[1.0],
                scales: &[-SATURATING_SCALE],
                active: &[true],
            },
        ];
        let sequential = sequential_fp32_weights(&[SATURATING_BASE], 1, 1, &updates);
        assert_eq!(sequential, [2.0 - SATURATING_SCALE]);
        assert!(!certify_deferred_window(&[SATURATING_BASE], 1, 1, &updates));
        assert_ne!(sequential[0], SATURATING_BASE);
        validate_deferred_encoder_prefix(
            &sequential,
            1,
            1,
            &[],
            &[1.0],
            &sequential,
            &sequential,
            &sequential,
            &sequential,
            "post_saturation_fallback",
        );
    }

    #[test]
    fn inactive_dimension_retains_its_weight_bits() {
        let updates = steps(&[1.0], &[SATURATING_SCALE], &[false], TINY_STEPS);
        let sequential = sequential_fp32_weights(&[-0.0], 1, 1, &updates);
        assert_eq!(sequential[0].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    #[should_panic(expected = "materialized weight")]
    fn one_shot_materialization_is_rejected() {
        let updates = steps(&[1.0], &[TINY_SCALE], &[true], TINY_STEPS);
        let one_shot = 1.0 + TINY_SCALE * TINY_STEPS as f32;
        validate_deferred_encoder_prefix(
            &[1.0],
            1,
            1,
            &updates,
            &[1.0],
            &[1.0],
            &[one_shot],
            &[1.0],
            &[one_shot],
            "incorrect_materialization",
        );
    }

    #[test]
    #[should_panic(expected = "deferred dot")]
    fn omitted_rank_one_contribution_is_rejected() {
        let updates = steps(&[1.0], &[SATURATING_SCALE], &[true], 1);
        let sequential = sequential_fp32_weights(&[1.0], 1, 1, &updates);
        validate_deferred_encoder_prefix(
            &[1.0],
            1,
            1,
            &updates,
            &[1.0],
            &sequential,
            &sequential,
            &sequential,
            &[1.0],
            "missing_factor",
        );
    }
}
