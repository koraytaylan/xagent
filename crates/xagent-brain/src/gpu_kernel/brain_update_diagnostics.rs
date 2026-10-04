//! Snapshot-only diagnostics for dense-update sparsity and clamp envelopes.
//! The production shader is unmodified. Exact matrix bit changes come from
//! before/after captures; conservative FP32 intervals classify clamp decisions
//! as definite, absent, or unknown. These observations are not timings or a
//! proof for a deferred representation with a different rounding schedule.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::dense_prefetch::prefetch_passes;
use super::dense_prefetch_validation::make_pipeline;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::wider_predictor;
use super::whitening_validation::{prepare_boundary_scene, prepare_kernel};
use super::*;

/// Independent brain seeds share the restored flat-world scene.
const BRAIN_SEEDS: [u64; 3] = [42, 314, 2026];
/// Sample populated memories and a later evolving population.
const WINDOW_STARTS: [u32; 2] = [256, 1_000];
/// Consecutive observations permit one-, eight-, and sixteen-cycle envelopes.
const WINDOW_CYCLES: u32 = 16;
const ENVELOPE_WINDOWS: [usize; 3] = [1, 8, 16];
/// Current optional production predictor configuration.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Compare contiguous dimension blocks with the complete feature row.
const CERTIFICATE_BLOCK_WIDTHS: [usize; 3] = [32, 64, ENCODED_DIMENSION];
/// Indices are the capture_state buffer order.
const PHYSICS: usize = 0;
const DECISIONS: usize = 1;
const SENSORY: usize = 7;
const BRAIN: usize = 8;
/// Each dense update applies these symmetric weight limits.
const ENCODER_LIMIT: f64 = 2.0;
const PREDICTOR_LIMIT: f64 = 3.0;
/// A deliberately loose 32-rounding margin covers each interval operation,
/// including permitted contraction and division/sqrt approximation. The
/// normal-minimum allowance also covers flushed subnormal operands/results.
/// Every resulting weight interval is checked against the actual GPU value.
const ROUNDING_MARGIN_OPERATIONS: f64 = 32.0;
/// Interoception denominators in coop_feature_extract use this exact f32 floor.
const INTEROCEPTION_FLOOR: f32 = 1e-6;
/// Sensory layout starts with velocity xyz, facing xyz, angular velocity,
/// four interoception values, then touch and scent.
const VECTOR_CHANNELS: usize = 3;
const SCALAR_INTEROCEPTION_CHANNELS: usize = 4;
/// Upper tail reported for active-agent-cycle counter distributions.
const PERCENTILE_NUMERATOR: usize = 95;
const PERCENTILE_DENOMINATOR: usize = 100;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Default)]
struct Interval {
    low: f64,
    high: f64,
}

impl Interval {
    fn exact(value: f32) -> Self {
        assert!(value.is_finite());
        Self {
            low: f64::from(value),
            high: f64::from(value),
        }
    }

    fn rounded(low: f64, high: f64, scale: f64) -> Self {
        let unit = f64::from(f32::EPSILON) / 2.0;
        let factor = ROUNDING_MARGIN_OPERATIONS * unit;
        let margin = factor / (1.0 - factor) * scale
            + ROUNDING_MARGIN_OPERATIONS * f64::from(f32::MIN_POSITIVE) * (1.0 + scale);
        assert!(low.is_finite() && high.is_finite() && scale.is_finite());
        Self {
            low: low - margin,
            high: high + margin,
        }
    }

    fn magnitude(self) -> f64 {
        self.low.abs().max(self.high.abs())
    }

    fn add(self, other: Self) -> Self {
        Self::rounded(
            self.low + other.low,
            self.high + other.high,
            self.magnitude() + other.magnitude(),
        )
    }

    fn subtract(self, other: Self) -> Self {
        self.add(Self {
            low: -other.high,
            high: -other.low,
        })
    }

    fn multiply(self, other: Self) -> Self {
        let products = [
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        ];
        // Flushing a tiny operand can erase its product with a larger normal
        // operand; a result-only minimum-normal allowance does not cover that.
        let input_flush = f64::from(f32::MIN_POSITIVE)
            * (self.magnitude() + other.magnitude() + f64::from(f32::MIN_POSITIVE));
        Self::rounded(
            products.into_iter().fold(f64::INFINITY, f64::min) - input_flush,
            products.into_iter().fold(f64::NEG_INFINITY, f64::max) + input_flush,
            self.magnitude() * other.magnitude(),
        )
    }

    fn clamp(self, limit: f64) -> Self {
        Self {
            low: self.low.clamp(-limit, limit),
            high: self.high.clamp(-limit, limit),
        }
    }

    fn includes(self, value: f32) -> bool {
        self.low <= f64::from(value) && f64::from(value) <= self.high
    }

    fn nonzero(self) -> bool {
        self.low > f64::from(f32::MIN_POSITIVE) || self.high < -f64::from(f32::MIN_POSITIVE)
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum ClampClass {
    Absent,
    Definite,
    Unknown,
}

fn classify_clamp(value: Interval, limit: f64) -> ClampClass {
    if value.low > limit || value.high < -limit {
        ClampClass::Definite
    } else if value.low >= -limit && value.high <= limit {
        ClampClass::Absent
    } else {
        ClampClass::Unknown
    }
}

#[derive(Clone, Copy, Default)]
struct Counts {
    attempted: u64,
    unchanged: u64,
    nonzero_step_unchanged: u64,
    unchanged_step_zero_or_unknown: u64,
    weight_clamp: [u64; 3],
    gradient_clamp: [u64; 3],
}

impl Counts {
    fn update(&mut self, before: f32, after: f32, step: Interval, raw: Interval, limit: f64) {
        assert!(
            raw.clamp(limit).includes(after),
            "GPU updated weight {after:e} outside snapshot interval [{:e}, {:e}], old {before:e}",
            raw.low,
            raw.high
        );
        self.attempted += 1;
        if before.to_bits() == after.to_bits() {
            self.unchanged += 1;
            if step.nonzero() {
                self.nonzero_step_unchanged += 1;
            } else {
                self.unchanged_step_zero_or_unknown += 1;
            }
        }
        // A clamp that changes a finite value always stores an exact endpoint.
        let classification = if f64::from(after).abs() < limit {
            ClampClass::Absent
        } else {
            classify_clamp(raw, limit)
        };
        self.weight_clamp[classification as usize] += 1;
    }
}

#[derive(Clone, Copy, Default)]
struct RowBound {
    present: bool,
    active: bool,
    maximum_weight: f64,
    maximum_step: f64,
    maximum_gradient: f64,
}

#[derive(Clone, Copy, Default)]
struct FeatureCertificates {
    total: u64,
    all_unchanged: u64,
    no_active_credit: u64,
    eligible: u64,
    certified: u64,
    certified_active: u64,
    certified_attempted_weights: u64,
    all_unchanged_attempted_weights: u64,
}

struct Observation {
    encoder_counts: Vec<Counts>,
    predictor_counts: Vec<Counts>,
    encoder: Vec<RowBound>,
    predictor: Vec<RowBound>,
    feature_certificates: Vec<[FeatureCertificates; CERTIFICATE_BLOCK_WIDTHS.len()]>,
    respawns: u64,
}

impl Observation {
    fn rows(&self, predictor: bool) -> &[RowBound] {
        if predictor {
            &self.predictor
        } else {
            &self.encoder
        }
    }
}

fn shader_constant(name: &str) -> f32 {
    let prefix = format!("const {name}: f32 = ");
    let common = include_str!("../shaders/kernel/common.wgsl");
    let line = common
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap();
    line.split(';').next().unwrap().parse().unwrap()
}

fn words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn tail(kernel: &GpuKernel, field: usize) -> usize {
    fixed_tail_base(kernel.layout.brain_stride) + field - O_PREDICTOR_CONTEXT_WEIGHT
}

// The raw-vision fixture reads the previous sensory frame. Only interoception
// reads same-cycle physics; the later global collision pass leaves these
// energy/integrity fields unchanged. No fixed feature clamp is
// assumed: adaptation subtracts each visual channel's stored running mean.
fn feature_intervals(
    kernel: &GpuKernel,
    sensory: &[f32],
    before: &[f32],
    physics: &[f32],
) -> Vec<Interval> {
    assert!(!kernel.layout.visual_cortex_enabled && !kernel.layout.danger_percept_enabled);
    let visual = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
    let mean = tail(kernel, O_SENSORY_MEAN);
    let mut features: Vec<_> = (0..visual)
        .map(|index| {
            Interval::exact(sensory[index]).subtract(Interval::exact(before[mean + index]))
        })
        .collect();
    let mut speed_squared = Interval::default();
    for &component in &sensory[visual..visual + VECTOR_CHANNELS] {
        let component = Interval::exact(component);
        speed_squared = speed_squared.add(component.multiply(component));
    }
    let low = speed_squared.low.max(0.0).sqrt();
    let high = speed_squared.high.max(0.0).sqrt();
    features.push(Interval::rounded(low, high, high));
    let facing = visual + VECTOR_CHANNELS;
    features.extend(
        sensory[facing..=facing + VECTOR_CHANNELS]
            .iter()
            .copied()
            .map(Interval::exact),
    );
    for (current, maximum) in [(P_ENERGY, P_MAX_ENERGY), (P_INTEGRITY, P_MAX_INTEGRITY)] {
        let denominator = f64::from(physics[maximum].max(INTEROCEPTION_FLOOR));
        let quotient = f64::from(physics[current]) / denominator;
        features.push(Interval::rounded(quotient, quotient, quotient.abs()));
    }
    for (current, previous) in [(P_ENERGY, P_PREV_ENERGY), (P_INTEGRITY, P_PREV_INTEGRITY)] {
        features
            .push(Interval::exact(physics[current]).subtract(Interval::exact(physics[previous])));
    }
    let touch = facing + VECTOR_CHANNELS + 1 + SCALAR_INTEROCEPTION_CHANNELS;
    let end = touch + MAX_TOUCH_CONTACTS * TOUCH_FEATURES + SCENT_CHANNELS;
    features.extend(sensory[touch..end].iter().copied().map(Interval::exact));
    assert_eq!(features.len(), kernel.layout.feature_count);
    features
}

fn observe_encoder(
    kernel: &GpuKernel,
    before: &[f32],
    after: &[f32],
    decision: &[f32],
    features: &[Interval],
    learning_rate: f32,
) -> (Counts, Vec<RowBound>) {
    let mut counts = Counts::default();
    let mut rows = Vec::new();
    for dim in 0..ENCODED_DIMENSION {
        let credit = decision[DECISION_CREDIT + dim];
        let active = credit.abs() >= shader_constant("CREDIT_EPSILON");
        let scale = Interval::exact(learning_rate)
            .multiply(Interval::exact(credit))
            .multiply(Interval::exact(shader_constant("ENCODER_CREDIT_SCALE")));
        let mut row = RowBound {
            present: true,
            active,
            ..RowBound::default()
        };
        for (feature, &input) in features.iter().enumerate() {
            let index = feature * ENCODED_DIMENSION + dim;
            row.maximum_weight = row.maximum_weight.max(f64::from(before[index]).abs());
            if active {
                let step = scale.multiply(input);
                row.maximum_step = row.maximum_step.max(step.magnitude());
                counts.update(
                    before[index],
                    after[index],
                    step,
                    Interval::exact(before[index]).add(step),
                    ENCODER_LIMIT,
                );
            } else {
                assert_eq!(before[index].to_bits(), after[index].to_bits());
            }
        }
        rows.push(row);
    }
    assert_eq!(
        counts.attempted,
        u64::try_from(rows.iter().filter(|row| row.active).count() * kernel.layout.feature_count)
            .unwrap()
    );
    (counts, rows)
}

fn observe_predictor(
    kernel: &GpuKernel,
    before: &[f32],
    after: &[f32],
    learning_rate: f32,
    respawned: bool,
) -> (Counts, Vec<RowBound>) {
    let mut counts = Counts::default();
    let mut rows = Vec::new();
    let matrix = kernel.layout.feature_count * ENCODED_DIMENSION + ENCODED_DIMENSION;
    let encoding = tail(kernel, O_PREV_ENCODED);
    let prediction = tail(kernel, O_PREV_PREDICTION);
    for dim in 0..PREDICTOR_DIMENSION {
        let old_prediction = Interval::exact(before[prediction + dim]);
        let error = old_prediction.subtract(Interval::exact(after[encoding + dim]));
        let derivative = Interval::exact(1.0).subtract(old_prediction.multiply(old_prediction));
        let coefficient = error.multiply(derivative);
        let mut row = RowBound {
            present: true,
            active: true,
            ..RowBound::default()
        };
        for input in 0..ENCODED_DIMENSION {
            // agent_death_respawn clears previous encoding but retains the
            // predictor and its previous prediction before this cycle trains.
            let previous_input = if respawned {
                0.0
            } else {
                before[encoding + input]
            };
            let gradient = coefficient.multiply(Interval::exact(previous_input));
            let step = Interval::exact(learning_rate).multiply(gradient.clamp(1.0));
            let index = matrix + dim * ENCODED_DIMENSION + input;
            row.maximum_weight = row.maximum_weight.max(f64::from(before[index]).abs());
            row.maximum_step = row.maximum_step.max(step.magnitude());
            row.maximum_gradient = row.maximum_gradient.max(gradient.magnitude());
            counts.gradient_clamp[classify_clamp(gradient, 1.0) as usize] += 1;
            counts.update(
                before[index],
                after[index],
                step,
                Interval::exact(before[index]).subtract(step),
                PREDICTOR_LIMIT,
            );
        }
        rows.push(row);
    }
    (counts, rows)
}

// A finite normal weight inside the clamp interval retains its exact bits
// when the update magnitude is strictly below half BOTH neighboring gaps.
// Strictness excludes midpoint ties, and excluding zeros/subnormals avoids
// signed-zero and flush-to-zero changes. f64 represents every f32 neighbor
// gap and its half exactly, including gaps around powers of two.
fn minimum_neighbor_gap(weights: &[f32]) -> Option<f64> {
    let mut minimum = f64::INFINITY;
    for &weight in weights {
        if !weight.is_normal() || f64::from(weight).abs() > ENCODER_LIMIT {
            return None;
        }
        let above = f64::from(weight.next_up()) - f64::from(weight);
        let below = f64::from(weight) - f64::from(weight.next_down());
        minimum = minimum.min(above).min(below);
    }
    Some(minimum)
}

// This recomputes metadata from captured weights solely to measure potential
// coverage. No GPU cache, reduction, dispatch, or skipped update is implemented.
// The interval includes uncertainty in the actual adapted feature and scale,
// and covers both a separately rounded product and a contracted multiply/add.
fn observe_feature_certificates(
    before: &[f32],
    after: &[f32],
    decision: &[f32],
    features: &[Interval],
    learning_rate: f32,
    block_width: usize,
) -> FeatureCertificates {
    assert!(block_width != 0 && ENCODED_DIMENSION % block_width == 0);
    let epsilon = shader_constant("CREDIT_EPSILON");
    let credit_scale = Interval::exact(shader_constant("ENCODER_CREDIT_SCALE"));
    let mut counts = FeatureCertificates::default();
    for (block, credits) in decision[DECISION_CREDIT..DECISION_CREDIT + ENCODED_DIMENSION]
        .chunks_exact(block_width)
        .enumerate()
    {
        let mut active_dimensions = 0_u64;
        let mut maximum_scale = 0.0_f64;
        for &credit in credits {
            if credit.abs() >= epsilon {
                active_dimensions += 1;
                let scale = Interval::exact(learning_rate)
                    .multiply(Interval::exact(credit))
                    .multiply(credit_scale);
                maximum_scale = maximum_scale.max(scale.magnitude());
            }
        }
        let scale_bound = Interval {
            low: 0.0,
            high: maximum_scale,
        };
        for (feature, &input) in features.iter().enumerate() {
            let first = feature * ENCODED_DIMENSION + block * block_width;
            let old = &before[first..first + block_width];
            let new = &after[first..first + block_width];
            let unchanged = old
                .iter()
                .zip(new)
                .all(|(old, new)| old.to_bits() == new.to_bits());
            counts.total += 1;
            counts.all_unchanged += u64::from(unchanged);
            counts.no_active_credit += u64::from(active_dimensions == 0);
            counts.all_unchanged_attempted_weights += u64::from(unchanged) * active_dimensions;
            let Some(minimum_gap) = minimum_neighbor_gap(old) else {
                continue;
            };
            counts.eligible += 1;
            let update_bound = scale_bound.multiply(input).magnitude();
            if update_bound < minimum_gap / 2.0 {
                assert!(
                    unchanged,
                    "feature {feature} block {block} width {block_width}: certified no-op changed a weight, update bound {update_bound:e}, minimum neighbor gap {minimum_gap:e}"
                );
                counts.certified += 1;
                counts.certified_active += u64::from(active_dimensions != 0);
                counts.certified_attempted_weights += active_dimensions;
            }
        }
    }
    counts
}

fn observe(
    kernel: &GpuKernel,
    before: &[Vec<u8>],
    after: &[Vec<u8>],
    learning_rate: f32,
) -> Observation {
    let old_brain = words(&before[BRAIN]);
    let new_brain = words(&after[BRAIN]);
    let old_physics = words(&before[PHYSICS]);
    let new_physics = words(&after[PHYSICS]);
    let sensory = words(&before[SENSORY]);
    let decisions = words(&after[DECISIONS]);
    let mut observation = Observation {
        encoder_counts: Vec::new(),
        predictor_counts: Vec::new(),
        encoder: Vec::new(),
        predictor: Vec::new(),
        feature_certificates: Vec::new(),
        respawns: 0,
    };
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * kernel.layout.brain_stride;
        let old = &old_brain[base..base + kernel.layout.brain_stride];
        let new = &new_brain[base..base + kernel.layout.brain_stride];
        let physics = &new_physics[agent * PHYS_STRIDE..(agent + 1) * PHYS_STRIDE];
        if physics[P_ALIVE] < 0.5 {
            assert_eq!(
                &before[BRAIN][base * size_of::<f32>()
                    ..(base + kernel.layout.brain_stride) * size_of::<f32>()],
                &after[BRAIN][base * size_of::<f32>()
                    ..(base + kernel.layout.brain_stride) * size_of::<f32>()]
            );
            observation.encoder_counts.push(Counts::default());
            observation.predictor_counts.push(Counts::default());
            observation
                .encoder
                .extend([RowBound::default(); ENCODED_DIMENSION]);
            observation
                .predictor
                .extend([RowBound::default(); PREDICTOR_DIMENSION]);
            continue;
        }
        assert_eq!(
            new[tail(kernel, O_TICK_COUNT)],
            old[tail(kernel, O_TICK_COUNT)] + 1.0
        );
        let respawned = physics[P_DEATH_COUNT] > old_physics[agent * PHYS_STRIDE + P_DEATH_COUNT];
        observation.respawns += u64::from(respawned);
        let sensory_base = agent * kernel.layout.sensory_stride;
        let features = feature_intervals(
            kernel,
            &sensory[sensory_base..sensory_base + kernel.layout.sensory_stride],
            old,
            physics,
        );
        let decision = &decisions[agent * DECISION_STRIDE..(agent + 1) * DECISION_STRIDE];
        observation
            .feature_certificates
            .push(CERTIFICATE_BLOCK_WIDTHS.map(|block_width| {
                observe_feature_certificates(
                    old,
                    new,
                    decision,
                    &features,
                    learning_rate,
                    block_width,
                )
            }));
        let (counts, rows) = observe_encoder(kernel, old, new, decision, &features, learning_rate);
        observation.encoder_counts.push(counts);
        observation.encoder.extend(rows);
        let (counts, rows) = observe_predictor(kernel, old, new, learning_rate, respawned);
        observation.predictor_counts.push(counts);
        observation.predictor.extend(rows);
    }
    observation
}

fn report_distribution(label: &str, metric: &str, mut values: Vec<u64>) {
    assert!(!values.is_empty());
    values.sort_unstable();
    let count = u32::try_from(values.len()).unwrap();
    let total: u64 = values.iter().sum();
    let mean = total as f64 / f64::from(count);
    let upper = ((values.len() - 1) * PERCENTILE_NUMERATOR).div_ceil(PERCENTILE_DENOMINATOR);
    println!(
        "DENSE_UPDATE_DISTRIBUTION label={label} metric={metric} samples={count} total={total} mean={mean:.6} min={} median={} p95={} max={} sample_unit=active_agent_cycle",
        values[0],
        values[values.len() / 2],
        values[upper],
        values[values.len() - 1]
    );
}

fn report_counts(label: &str, matrix: &str, counts: &[Counts]) {
    let metrics: [(&str, fn(&Counts) -> u64); 10] = [
        ("attempted", |c| c.attempted),
        ("unchanged", |c| c.unchanged),
        ("nonzero_step_unchanged", |c| c.nonzero_step_unchanged),
        ("unchanged_step_zero_or_unknown", |c| {
            c.unchanged_step_zero_or_unknown
        }),
        ("weight_clamp_absent", |c| {
            c.weight_clamp[ClampClass::Absent as usize]
        }),
        ("weight_clamp_definite", |c| {
            c.weight_clamp[ClampClass::Definite as usize]
        }),
        ("weight_clamp_unknown", |c| {
            c.weight_clamp[ClampClass::Unknown as usize]
        }),
        ("gradient_clamp_absent", |c| {
            c.gradient_clamp[ClampClass::Absent as usize]
        }),
        ("gradient_clamp_definite", |c| {
            c.gradient_clamp[ClampClass::Definite as usize]
        }),
        ("gradient_clamp_unknown", |c| {
            c.gradient_clamp[ClampClass::Unknown as usize]
        }),
    ];
    for (name, extract) in metrics {
        if matrix == "encoder" && name.starts_with("gradient_") {
            continue;
        }
        report_distribution(
            label,
            &format!("{matrix}_{name}"),
            counts.iter().map(extract).collect(),
        );
    }
}

fn report_envelopes(label: &str, observations: &[Observation], predictor: bool) {
    let matrix = if predictor { "predictor" } else { "encoder" };
    let limit = if predictor {
        PREDICTOR_LIMIT
    } else {
        ENCODER_LIMIT
    };
    for length in ENVELOPE_WINDOWS {
        let (mut eligible, mut active, mut weight_safe, mut gradient_safe, mut combined_safe) =
            (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
        for window in observations.chunks_exact(length) {
            for row in 0..window[0].rows(predictor).len() {
                if window
                    .iter()
                    .any(|sample| !sample.rows(predictor)[row].present)
                {
                    continue;
                }
                eligible += 1;
                let any_active = window
                    .iter()
                    .any(|sample| sample.rows(predictor)[row].active);
                active += u64::from(any_active);
                let mut maximum = window[0].rows(predictor)[row].maximum_weight;
                let mut unclipped_gradient = true;
                for sample in window {
                    let value = sample.rows(predictor)[row];
                    let sum = maximum + value.maximum_step;
                    maximum = Interval::rounded(sum, sum, sum).high;
                    unclipped_gradient &= !predictor || value.maximum_gradient <= 1.0;
                }
                let safe = maximum < limit;
                weight_safe += u64::from(safe);
                gradient_safe += u64::from(unclipped_gradient);
                combined_safe += u64::from(any_active && safe && unclipped_gradient);
            }
        }
        println!(
            "DENSE_UPDATE_ENVELOPE label={label} matrix={matrix} cycles={length} row_windows={eligible} active_row_windows={active} weight_safe={weight_safe} gradient_safe={gradient_safe} active_rank_one_safe={combined_safe} method=snapshot_fp64_intervals future_representation_proof=required"
        );
    }
}

fn report_window(label: &str, observations: &[Observation]) {
    for predictor in [false, true] {
        let counts: Vec<_> = observations
            .iter()
            .flat_map(|sample| {
                let counts = if predictor {
                    &sample.predictor_counts
                } else {
                    &sample.encoder_counts
                };
                counts
                    .iter()
                    .zip(sample.rows(predictor).as_chunks::<ENCODED_DIMENSION>().0)
                    .filter(|(_, rows)| rows[0].present)
                    .map(|(count, _)| *count)
            })
            .collect();
        report_counts(
            label,
            if predictor { "predictor" } else { "encoder" },
            &counts,
        );
        report_envelopes(label, observations, predictor);
    }
    report_distribution(
        label,
        "active_encoder_credit_dimensions",
        observations
            .iter()
            .flat_map(|sample| {
                sample
                    .encoder
                    .as_chunks::<ENCODED_DIMENSION>()
                    .0
                    .iter()
                    .filter(|rows| rows[0].present)
                    .map(|rows| {
                        u64::try_from(rows.iter().filter(|row| row.active).count()).unwrap()
                    })
            })
            .collect(),
    );
    println!(
        "DENSE_UPDATE_SCOPE label={label} production_replay_buffers_exact=13 observation=snapshot_only respawn_agent_cycles={} timing_not_measured=true cache_effects_not_measured=true materialized_reads_and_writes_require_export_and_invalidation=true",
        observations
            .iter()
            .map(|sample| sample.respawns)
            .sum::<u64>()
    );
    report_feature_certificates(label, observations);
}

fn feature_certificate_totals(
    observations: &[Observation],
    width_index: usize,
) -> FeatureCertificates {
    let mut total = FeatureCertificates::default();
    for widths in observations
        .iter()
        .flat_map(|sample| &sample.feature_certificates)
    {
        let counts = widths[width_index];
        total.total += counts.total;
        total.all_unchanged += counts.all_unchanged;
        total.no_active_credit += counts.no_active_credit;
        total.eligible += counts.eligible;
        total.certified += counts.certified;
        total.certified_active += counts.certified_active;
        total.certified_attempted_weights += counts.certified_attempted_weights;
        total.all_unchanged_attempted_weights += counts.all_unchanged_attempted_weights;
    }
    total
}

fn report_feature_certificates(label: &str, observations: &[Observation]) {
    let attempted_weights: u64 = observations
        .iter()
        .flat_map(|sample| &sample.encoder_counts)
        .map(|counts| counts.attempted)
        .sum();
    for (width_index, block_width) in CERTIFICATE_BLOCK_WIDTHS.into_iter().enumerate() {
        let total = feature_certificate_totals(observations, width_index);
        println!(
            "DENSE_ENCODER_BLOCK_CERTIFICATE label={label} block_width={block_width} total_blocks={} all_block_weights_unchanged={} no_active_credit_blocks={} eligible_blocks={} invalid_blocks={} certificate_hits={} certificate_hits_with_active_credit={} attempted_weights={} certified_attempted_weights={} all_unchanged_block_attempted_weights={} false_positives=0 proof_scope=observed_compilation future_gpu_cache_unimplemented=true",
            total.total,
            total.all_unchanged,
            total.no_active_credit,
            total.eligible,
            total.total - total.eligible,
            total.certified,
            total.certified_active,
            attempted_weights,
            total.certified_attempted_weights,
            total.all_unchanged_attempted_weights
        );
    }
    // The full-row report expresses the same coverage in complete features.
    let full_row_index = CERTIFICATE_BLOCK_WIDTHS
        .iter()
        .position(|&width| width == ENCODED_DIMENSION)
        .unwrap();
    let total = feature_certificate_totals(observations, full_row_index);
    println!(
        "DENSE_ENCODER_FEATURE_CERTIFICATE label={label} total_features={} all_128_unchanged={} no_active_credit_features={} eligible_features={} invalid_features={} certificate_hits={} certificate_hits_with_active_credit={} attempted_weights={} certified_attempted_weights={} all_128_unchanged_attempted_weights={} false_positives=0 proof_scope=observed_compilation future_gpu_cache_unimplemented=true",
        total.total,
        total.all_unchanged,
        total.no_active_credit,
        total.eligible,
        total.total - total.eligible,
        total.certified,
        total.certified_active,
        attempted_weights,
        total.certified_attempted_weights,
        total.all_unchanged_attempted_weights
    );
}

fn advance(kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
    kernel.dispatch_ticks(
        u64::from(cycle * kernel.brain_tick_stride),
        cycles * kernel.brain_tick_stride,
    );
    kernel.poll_wait();
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn dense_update_sparsity_and_clamp_envelopes() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel();
    let passes = wider_predictor(
        &prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH_FACTOR,
        ),
        PREDICTOR_LANES,
    );
    kernel.kernel_pipeline = make_pipeline(&kernel, &passes, "brain_update_snapshot_reference");
    let common_world = checkpoint(&kernel);
    let brain = BrainConfig {
        vision_stride: 1,
        ..BrainConfig::default()
    };
    for seed in BRAIN_SEEDS {
        restore(&mut kernel, &common_world);
        kernel.reset_agents_seeded(&brain, seed);
        prepare_boundary_scene(&kernel);
        let mut cycle = 0;
        for start in WINDOW_STARTS {
            advance(&mut kernel, cycle, start - cycle);
            cycle = start;
            let mut previous = capture_state(&kernel)?;
            let mut observations = Vec::new();
            for _ in 0..WINDOW_CYCLES {
                let initial = checkpoint(&kernel);
                advance(&mut kernel, cycle, 1);
                let expected = capture_state(&kernel)?;
                let observation = observe(&kernel, &previous, &expected, brain.learning_rate);
                restore(&mut kernel, &initial);
                advance(&mut kernel, cycle, 1);
                let replay = capture_state(&kernel)?;
                assert_state_equal(&kernel, &expected, &replay);
                observations.push(observation);
                previous = expected;
                cycle += 1;
            }
            report_window(
                &format!(
                    "brain_seed={seed} start_cycle={start} cycles={WINDOW_CYCLES} lanes={PREDICTOR_LANES} prefetch={PREFETCH_FACTOR}"
                ),
                &observations,
            );
        }
    }
    Ok(())
}
