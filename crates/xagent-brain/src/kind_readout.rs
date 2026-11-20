//! Supervised linear readout of a binary label.
//!
//! Measures whether a binary label is linearly readable from a feature
//! vector: held-out balanced accuracy of an L2 logistic regression under
//! stratified k-fold, plus a shuffled-label control. All functions only read
//! their inputs; they take copied values and touch no other state.

/// Number of stratified cross-validation folds.
pub const READOUT_FOLDS: usize = 5;
/// Minimum samples of each class per fold, so every test block holds both classes.
pub const MIN_SAMPLES_PER_CLASS_PER_FOLD: usize = 2;
/// Candidate L2 strengths; the best by mean held-out balanced accuracy is kept.
pub const READOUT_L2_STRENGTHS: [f64; 3] = [0.01, 0.1, 1.0];
/// Full-batch gradient steps from zero weights.
pub const READOUT_ITERATIONS: usize = 300;
/// Step size; stable on standardised features because logistic curvature is at
/// most a quarter of the covariance's top eigenvalue.
pub const READOUT_LEARNING_RATE: f64 = 1.0;
/// Lower bound on a feature's standard deviation, keeping standardisation
/// finite for a constant feature.
pub const READOUT_STANDARD_DEVIATION_FLOOR: f64 = 1e-6;
/// Fixed fold seed, so folds do not depend on the caller.
pub const READOUT_FOLD_SEED: u64 = 0x5EED_F01D_C0DE_0001;
/// One half, used by balanced accuracy.
const HALF: f64 = 0.5;
/// Splitmix64 state increment (golden-ratio constant).
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
/// Splitmix64 first mixing multiplier.
const SPLITMIX_MULTIPLIER_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// Splitmix64 second mixing multiplier.
const SPLITMIX_MULTIPLIER_B: u64 = 0x94D0_49BB_1331_11EB;
/// Splitmix64 first xor-shift.
const SPLITMIX_SHIFT_A: u32 = 30;
/// Splitmix64 second xor-shift.
const SPLITMIX_SHIFT_B: u32 = 27;
/// Splitmix64 final xor-shift.
const SPLITMIX_SHIFT_C: u32 = 31;

/// One labelled feature vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutSample {
    /// Copied feature values.
    pub features: Vec<f32>,
    /// Binary label to read out.
    pub label: bool,
}

/// Result of a readout measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadoutScore {
    /// Held-out balanced accuracy on the real labels.
    pub balanced_accuracy: f32,
    /// Held-out balanced accuracy on permuted labels (control).
    pub shuffled_balanced_accuracy: f32,
    /// Number of samples measured.
    pub samples: usize,
    /// Number of samples with a true label.
    pub positives: usize,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut z = *state;
    z = (z ^ (z >> SPLITMIX_SHIFT_A)).wrapping_mul(SPLITMIX_MULTIPLIER_A);
    z = (z ^ (z >> SPLITMIX_SHIFT_B)).wrapping_mul(SPLITMIX_MULTIPLIER_B);
    z ^ (z >> SPLITMIX_SHIFT_C)
}

/// Fisher-Yates shuffle (used for `usize` indices and `bool` labels).
fn shuffle<T>(items: &mut [T], state: &mut u64) -> Option<()> {
    for i in (1..items.len()).rev() {
        let bound = u64::try_from(i.checked_add(1)?).ok()?;
        let j = usize::try_from(splitmix64(state) % bound).ok()?;
        items.swap(i, j);
    }
    Some(())
}

#[allow(clippy::cast_possible_truncation)]
fn to_f32(value: f64) -> f32 {
    value as f32
}

fn count_f64(n: usize) -> Option<f64> {
    u32::try_from(n).ok().map(f64::from)
}

/// Train and test data for one fold, standardised with training statistics.
struct FoldData {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<bool>,
    test_x: Vec<Vec<f64>>,
    test_y: Vec<bool>,
}

/// Test index blocks per fold, stratified by label.
fn stratified_blocks(labels: &[bool]) -> Option<Vec<Vec<usize>>> {
    let mut state = READOUT_FOLD_SEED;
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); READOUT_FOLDS];
    for class in [true, false] {
        let mut idx: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == class).collect();
        shuffle(&mut idx, &mut state)?;
        let len = idx.len();
        for (k, fold) in folds.iter_mut().enumerate() {
            let lo = k.checked_mul(len)? / READOUT_FOLDS;
            let hi = (k + 1).checked_mul(len)? / READOUT_FOLDS;
            fold.extend_from_slice(&idx[lo..hi]);
        }
    }
    Some(folds)
}

fn build_fold(
    features: &[Vec<f64>],
    labels: &[bool],
    test_idx: &[usize],
    mask: &mut [bool],
) -> Option<FoldData> {
    mask.fill(false);
    for &i in test_idx {
        mask[i] = true;
    }
    let train: Vec<usize> = (0..labels.len()).filter(|&i| !mask[i]).collect();
    let n = count_f64(train.len())?;
    let dim = features.first()?.len();
    let mut mean = vec![0.0_f64; dim];
    for &i in &train {
        for (m, v) in mean.iter_mut().zip(&features[i]) {
            *m += v / n;
        }
    }
    let mut sd = vec![0.0_f64; dim];
    for &i in &train {
        for ((s, v), m) in sd.iter_mut().zip(&features[i]).zip(&mean) {
            *s += (v - m) * (v - m) / n;
        }
    }
    for s in &mut sd {
        *s = s.sqrt().max(READOUT_STANDARD_DEVIATION_FLOOR);
    }
    let standardise = |i: usize| -> Vec<f64> {
        features[i]
            .iter()
            .zip(&mean)
            .zip(&sd)
            .map(|((v, m), s)| (v - m) / s)
            .collect()
    };
    Some(FoldData {
        train_x: train.iter().map(|&i| standardise(i)).collect(),
        train_y: train.iter().map(|&i| labels[i]).collect(),
        test_x: test_idx.iter().map(|&i| standardise(i)).collect(),
        test_y: test_idx.iter().map(|&i| labels[i]).collect(),
    })
}

fn decision(weights: &[f64], bias: f64, x: &[f64]) -> f64 {
    bias + weights.iter().zip(x).map(|(w, v)| w * v).sum::<f64>()
}

fn fit_logistic(data: &FoldData, lambda: f64) -> Option<(Vec<f64>, f64)> {
    let dim = data.train_x.first()?.len();
    let n = count_f64(data.train_x.len())?;
    let mut weights = vec![0.0_f64; dim];
    let mut bias = 0.0_f64;
    for _ in 0..READOUT_ITERATIONS {
        let mut grad = vec![0.0_f64; dim];
        let mut grad_bias = 0.0_f64;
        for (x, &y) in data.train_x.iter().zip(&data.train_y) {
            let p = 1.0 / (1.0 + (-decision(&weights, bias, x)).exp());
            let err = (p - if y { 1.0 } else { 0.0 }) / n;
            grad_bias += err;
            for (g, v) in grad.iter_mut().zip(x) {
                *g += err * v;
            }
        }
        for (w, g) in weights.iter_mut().zip(&grad) {
            *w -= READOUT_LEARNING_RATE * (g + lambda * *w);
        }
        bias -= READOUT_LEARNING_RATE * grad_bias;
    }
    Some((weights, bias))
}

fn fold_balanced_accuracy(data: &FoldData, weights: &[f64], bias: f64) -> Option<f64> {
    let (mut tp, mut tn, mut p, mut n) = (0_usize, 0_usize, 0_usize, 0_usize);
    for (x, &y) in data.test_x.iter().zip(&data.test_y) {
        let predicted = decision(weights, bias, x) > 0.0;
        if y {
            p += 1;
            tp += usize::from(predicted);
        } else {
            n += 1;
            tn += usize::from(!predicted);
        }
    }
    Some(HALF * (count_f64(tp)? / count_f64(p)? + count_f64(tn)? / count_f64(n)?))
}

/// Best-strength mean held-out balanced accuracy for the given labels.
fn cross_validate(features: &[Vec<f64>], labels: &[bool]) -> Option<f64> {
    let blocks = stratified_blocks(labels)?;
    let mut mask = vec![false; labels.len()];
    let mut folds = Vec::with_capacity(READOUT_FOLDS);
    for block in &blocks {
        folds.push(build_fold(features, labels, block, &mut mask)?);
    }
    let fold_count = count_f64(READOUT_FOLDS)?;
    let mut best: Option<f64> = None;
    for lambda in READOUT_L2_STRENGTHS {
        let mut total = 0.0;
        for data in &folds {
            let (weights, bias) = fit_logistic(data, lambda)?;
            total += fold_balanced_accuracy(data, &weights, bias)?;
        }
        let mean = total / fold_count;
        if best.is_none_or(|b| mean > b) {
            best = Some(mean);
        }
    }
    best
}

/// Measures how linearly readable the label is from the features.
///
/// Returns `None` for empty or ragged input, zero-length or non-finite
/// features, or fewer than `READOUT_FOLDS * MIN_SAMPLES_PER_CLASS_PER_FOLD`
/// samples in either class.
#[must_use]
pub fn fit_kind_readout(samples: &[ReadoutSample], shuffle_seed: u64) -> Option<ReadoutScore> {
    let dim = samples.first()?.features.len();
    if dim == 0
        || samples
            .iter()
            .any(|s| s.features.len() != dim || s.features.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    u32::try_from(samples.len()).ok()?;
    let positives = samples.iter().filter(|s| s.label).count();
    let negatives = samples.len() - positives;
    let minimum = READOUT_FOLDS * MIN_SAMPLES_PER_CLASS_PER_FOLD;
    if positives < minimum || negatives < minimum {
        return None;
    }
    let features: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| s.features.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let labels: Vec<bool> = samples.iter().map(|s| s.label).collect();
    let real = cross_validate(&features, &labels)?;
    let mut permuted = labels;
    let mut state = shuffle_seed;
    shuffle(&mut permuted, &mut state)?;
    let shuffled = cross_validate(&features, &permuted)?;
    Some(ReadoutScore {
        balanced_accuracy: to_f32(real),
        shuffled_balanced_accuracy: to_f32(shuffled),
        samples: samples.len(),
        positives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED: u64 = 0xABCD_1234;
    const SEPARABLE_SAMPLES: usize = 400;
    const SEPARABLE_FEATURES: usize = 12;
    const TEST_WEIGHTS: [f64; 4] = [1.0, -1.0, 0.5, -0.5];
    const NOISE_HALF_WIDTH: f64 = 0.1;
    const UNIT_SCALE: f64 = 4_294_967_296.0;
    const HIGH_ACCURACY: f32 = 0.95;
    const CONTROL_CEILING: f32 = 0.65;
    const CHANCE_LOW: f32 = 0.35;
    const CHANCE_HIGH: f32 = 0.65;
    const IMBALANCED_CANDIDATES: usize = 1200;
    const NEGATIVE_KEEP_EVERY: usize = 4;
    const IMBALANCED_ACCURACY: f32 = 0.90;
    const MIN_CLASS: usize = READOUT_FOLDS * MIN_SAMPLES_PER_CLASS_PER_FOLD;
    const FILLER_NEGATIVES: usize = 100;
    const SEED_A: u64 = 11;
    const SEED_B: u64 = 987_654_321;
    /// Shift that keeps the high 32 bits of a draw.
    const UNIT_SHIFT: u32 = 32;
    /// Factor mapping a unit value to a symmetric range of width 2.
    const UNIT_RANGE_SCALE: f64 = 2.0;
    /// The imbalanced set must have fewer than 1/this negatives per positive.
    const MAJORITY_RATIO: usize = 2;
    /// Sample index inside the positive block used to corrupt a value.
    const POSITIVE_BLOCK_INDEX: usize = 3;
    /// Sample index used to make a feature vector ragged.
    const RAGGED_INDEX: usize = 5;

    /// Uniform in [0, 1).
    fn unit(state: &mut u64) -> f64 {
        f64::from(u32::try_from(splitmix64(state) >> UNIT_SHIFT).unwrap()) / UNIT_SCALE
    }

    fn separable(count: usize) -> Vec<ReadoutSample> {
        let mut state = TEST_SEED;
        (0..count)
            .map(|_| {
                let features: Vec<f32> = (0..SEPARABLE_FEATURES)
                    .map(|_| to_f32(unit(&mut state) * UNIT_RANGE_SCALE - 1.0))
                    .collect();
                let dot: f64 = features
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| TEST_WEIGHTS[i % TEST_WEIGHTS.len()] * f64::from(v))
                    .sum();
                let noise = (unit(&mut state) * UNIT_RANGE_SCALE - 1.0) * NOISE_HALF_WIDTH;
                ReadoutSample {
                    features,
                    label: dot + noise > 0.0,
                }
            })
            .collect()
    }

    fn with_counts(pos: usize, neg: usize) -> Vec<ReadoutSample> {
        let mut out = Vec::new();
        for i in 0..pos + neg {
            let v = to_f32(f64::from(u32::try_from(i).unwrap()));
            out.push(ReadoutSample {
                features: vec![v, -v],
                label: i < pos,
            });
        }
        out
    }

    #[test]
    fn readout_insufficient_class_counts_is_none() {
        assert!(fit_kind_readout(&with_counts(MIN_CLASS - 1, FILLER_NEGATIVES), 1).is_none());
        assert!(fit_kind_readout(&with_counts(MIN_CLASS, FILLER_NEGATIVES), 1).is_some());
    }

    #[test]
    fn readout_non_finite_feature_is_none() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut set = with_counts(MIN_CLASS, FILLER_NEGATIVES);
            set[POSITIVE_BLOCK_INDEX].features[0] = bad;
            assert!(fit_kind_readout(&set, 1).is_none());
        }
    }

    #[test]
    fn readout_mismatched_or_empty_input_is_none() {
        assert!(fit_kind_readout(&[], 1).is_none());
        let mut set = with_counts(MIN_CLASS, FILLER_NEGATIVES);
        set[RAGGED_INDEX].features.push(0.0);
        assert!(fit_kind_readout(&set, 1).is_none());
        let mut empty = with_counts(MIN_CLASS, FILLER_NEGATIVES);
        for s in &mut empty {
            s.features.clear();
        }
        assert!(fit_kind_readout(&empty, 1).is_none());
    }

    #[test]
    fn readout_separable_scores_high_and_shuffled_low() {
        let set = separable(SEPARABLE_SAMPLES);
        let score = fit_kind_readout(&set, SEED_A).unwrap();
        assert!(score.balanced_accuracy >= HIGH_ACCURACY, "{score:?}");
        assert!(
            score.shuffled_balanced_accuracy <= CONTROL_CEILING,
            "{score:?}"
        );
        assert_eq!(score.samples, SEPARABLE_SAMPLES);
        assert_eq!(score.positives, set.iter().filter(|s| s.label).count());
    }

    #[test]
    fn readout_noise_labels_near_chance() {
        let mut state = TEST_SEED;
        let set: Vec<ReadoutSample> = (0..SEPARABLE_SAMPLES)
            .map(|_| ReadoutSample {
                features: (0..SEPARABLE_FEATURES)
                    .map(|_| to_f32(unit(&mut state)))
                    .collect(),
                label: splitmix64(&mut state) & 1 == 1,
            })
            .collect();
        let score = fit_kind_readout(&set, SEED_A).unwrap();
        assert!(
            (CHANCE_LOW..=CHANCE_HIGH).contains(&score.balanced_accuracy),
            "{score:?}"
        );
    }

    #[test]
    fn readout_imbalanced_separable_uses_balanced_accuracy() {
        let mut seen_negatives = 0_usize;
        let set: Vec<ReadoutSample> = separable(IMBALANCED_CANDIDATES)
            .into_iter()
            .filter(|s| {
                if s.label {
                    return true;
                }
                seen_negatives += 1;
                (seen_negatives - 1) % NEGATIVE_KEEP_EVERY == 0
            })
            .collect();
        let positives = set.iter().filter(|s| s.label).count();
        let negatives = set.len() - positives;
        assert!(negatives * MAJORITY_RATIO < positives);
        let score = fit_kind_readout(&set, SEED_A).unwrap();
        assert!(score.balanced_accuracy >= IMBALANCED_ACCURACY, "{score:?}");
    }

    #[test]
    fn readout_is_deterministic() {
        let set = separable(SEPARABLE_SAMPLES);
        let a = fit_kind_readout(&set, SEED_A).unwrap();
        let b = fit_kind_readout(&set, SEED_A).unwrap();
        assert_eq!(a.balanced_accuracy.to_bits(), b.balanced_accuracy.to_bits());
        assert_eq!(
            a.shuffled_balanced_accuracy.to_bits(),
            b.shuffled_balanced_accuracy.to_bits()
        );
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.positives, b.positives);
    }

    #[test]
    fn readout_shuffle_seed_changes_only_control() {
        let set = separable(SEPARABLE_SAMPLES);
        let a = fit_kind_readout(&set, SEED_A).unwrap();
        let b = fit_kind_readout(&set, SEED_B).unwrap();
        assert_eq!(a.balanced_accuracy.to_bits(), b.balanced_accuracy.to_bits());
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.positives, b.positives);
        for v in [a.shuffled_balanced_accuracy, b.shuffled_balanced_accuracy] {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
