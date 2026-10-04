//! Raw GPU checks for both recall-reuse sources. Dot and squared-norm outputs
//! are checked against f64 forward bounds. Cosine intervals propagate those
//! bounds through sqrt, multiplication, division, clamps and the norm guard.
//! Threshold crossings are reported separately from numerical error.
//!
//! WGSL accuracy permits 2 ULP for inverseSqrt, 2.5 ULP for division, and
//! defines sqrt accuracy through their composition:
//! https://www.w3.org/TR/WGSL/#floating-point-accuracy

use std::error::Error;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::cycle_profile::make_kernel;
use super::dense_prefetch::prefetch_passes;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::wider_predictor;
use super::recall_reuse_validation::{cached_passes, cooperative_cached_passes};
use super::rounding_validation::{check_fp32_dot, DotMetrics};
use super::vision_validation::read_buffer;
use super::*;

/// Actual production composition surrounding both tested recall sources.
const PREFETCH: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Stable independent raw-input seed.
const SEED: u64 = 20_261_005;
/// Production norm guard and reinforcement threshold, both materialized as f32.
const NORM_GUARD: f32 = 1e-8;
const REINFORCEMENT_THRESHOLD: f32 = 0.3;
/// Five per-pattern arrays fit the kernel's existing per-agent scratch.
const OUTPUT_ARRAYS: usize = 5;
const RAW_CACHE: usize = MEMORY_CAP;
const RAW_DOT: usize = RAW_CACHE + MEMORY_CAP;
const RAW_NORM: usize = RAW_DOT + MEMORY_CAP;
const RAW_SQUARE: usize = RAW_NORM + MEMORY_CAP;
/// Pipeline layout follows the kernel's two-word push-constant range.
const PUSH_BYTES: u32 = 8;
/// WGSL allows rounding toward either adjacent f32, so use a full epsilon.
const ROUND_RELATIVE: f64 = f32::EPSILON as f64;
/// Maximum relative ULP allowance for a normal f32 inverse square root.
const INVERSE_SQRT_RELATIVE: f64 = 2.0 * ROUND_RELATIVE;
/// Maximum relative ULP allowance for normal-range WGSL division.
const DIVIDE_RELATIVE: f64 = 2.5 * ROUND_RELATIVE;
/// Moderate exponent span keeps all nonzero squares and denominators normal.
const EXPONENT_SPAN: i32 = 12;
/// Even/odd pairing constructs cancellations and positive/negative alignments.
const PAIR: usize = 2;
/// Threshold fixtures rotate one representable value below, at, and above it.
const THRESHOLD_VARIANTS: usize = 3;
/// Distinct query norms below the guard remain normal FP32 values.
const BELOW_GUARD_SCALE: f32 = 0.5;
/// Division's WGSL ULP guarantee covers denominators through 2^126.
const DIVISOR_MAX_EXPONENT: i32 = 126;

#[derive(Clone, Copy, Debug)]
enum QueryCase {
    Seeded,
    Zero,
    BelowGuard,
    AtGuard,
    AboveGuard,
    Tiny,
    Cancellation,
    Dynamic,
    Reinforcement,
    Clamp,
}
const QUERIES: [QueryCase; 10] = [
    QueryCase::Seeded,
    QueryCase::Zero,
    QueryCase::BelowGuard,
    QueryCase::AtGuard,
    QueryCase::AboveGuard,
    QueryCase::Tiny,
    QueryCase::Cancellation,
    QueryCase::Dynamic,
    QueryCase::Reinforcement,
    QueryCase::Clamp,
];

#[derive(Clone, Copy)]
enum PatternCase {
    Ordinary,
    Inactive,
    ZeroNorm,
    BelowGuard,
    AtGuard,
    AboveGuard,
}
const PATTERNS: [PatternCase; 6] = [
    PatternCase::Ordinary,
    PatternCase::Inactive,
    PatternCase::ZeroNorm,
    PatternCase::BelowGuard,
    PatternCase::AtGuard,
    PatternCase::AboveGuard,
];

struct Row {
    values: Vec<f32>,
    norm: f32,
    active: bool,
}
struct Fixture {
    keys: Vec<Vec<f32>>,
    rows: Vec<Row>,
}

#[derive(Clone, Copy)]
struct Interval {
    low: f64,
    high: f64,
}

impl Interval {
    fn contains(self, value: f64) -> bool {
        value >= self.low && value <= self.high
    }
    fn clamp(self) -> Self {
        Self {
            low: self.low.clamp(-1.0, 1.0),
            high: self.high.clamp(-1.0, 1.0),
        }
    }
}

#[derive(Default)]
struct Summary {
    active: usize,
    inactive: usize,
    norm_guard_flips: usize,
    reinforcement_flips: usize,
    max_norm_error: f64,
    max_cosine_error: f64,
    max_dot_bound_fraction: f64,
}

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn function(source: &str, declaration: &str) -> String {
    assert_eq!(source.matches(declaration).count(), 1);
    let start = source.find(declaration).unwrap();
    let end = start + source[start..].find("\n}\n").unwrap() + "\n}\n".len();
    source[start..end].to_owned()
}

fn probe_source(cooperative: bool) -> String {
    let production = wider_predictor(
        &prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH,
        ),
        PREDICTOR_LANES,
    );
    let passes = if cooperative {
        cooperative_cached_passes(&production)
    } else {
        cached_passes(&production)
    };
    let mut recall = function(&passes, "fn coop_recall_score(agent_id: u32, tid: u32) {");
    if !cooperative {
        let marker = "            let q_norm = sqrt(q_norm_sq);";
        assert_eq!(recall.matches(marker).count(), 1);
        recall = recall.replacen(
            marker,
            &format!(
                r"{marker}
            let output = agent_id * BRAIN_SCRATCH_STRIDE;
            brain_scratch[output + RAW_DOT + tid] = dot;
            brain_scratch[output + RAW_NORM + tid] = q_norm;
            brain_scratch[output + RAW_SQUARE + tid] = q_norm_sq;"
            ),
            1,
        );
    }
    let mut source = include_str!("../shaders/kernel/common.wgsl").to_owned();
    for declaration in [
        "const BRAIN_WORKGROUP_SIZE:",
        "var<workgroup> s_memory_key:",
        "var<workgroup> s_similarities:",
        "var<workgroup> s_argmin_val:",
        "var<workgroup> s_dense_partials:",
        "var<workgroup> s_reinf_dot:",
        "var<workgroup> s_enc_norm:",
    ] {
        let matches: Vec<_> = passes
            .lines()
            .filter(|line| line.starts_with(declaration))
            .collect();
        assert_eq!(matches.len(), 1);
        source.push('\n');
        source.push_str(matches[0]);
    }
    source.push('\n');
    source.push_str(&function(&passes, "fn wg_reduce_dense(tid: u32) {"));
    source.push_str(&recall);
    source.push_str(include_str!("recall_cosine_probe.wgsl"));
    source
}

fn pipeline(kernel: &GpuKernel, cooperative: bool) -> wgpu::ComputePipeline {
    let label = format!("raw_recall_cosine_cooperative_{cooperative}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(probe_source(cooperative).into()),
        });
    let bind = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &[&bind],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_BYTES,
            }],
        });
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert(
        "RAW_COOPERATIVE".to_owned(),
        if cooperative { 1.0 } else { 0.0 },
    );
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("recall_cosine_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn norm_f64(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt()
}

fn key(case: QueryCase, rng: &mut StdRng) -> Vec<f32> {
    let mut values = vec![0.0; ENCODED_DIMENSION];
    match case {
        QueryCase::Seeded => values
            .iter_mut()
            .for_each(|value| *value = rng.random_range(-1.0..1.0)),
        QueryCase::Zero => {}
        QueryCase::BelowGuard => values[0] = f32::from_bits(NORM_GUARD.to_bits() - 1),
        QueryCase::AtGuard => values[0] = NORM_GUARD,
        QueryCase::AboveGuard => values[0] = f32::from_bits(NORM_GUARD.to_bits() + 1),
        QueryCase::Tiny => values[0] = NORM_GUARD * BELOW_GUARD_SCALE,
        QueryCase::Cancellation => values
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| *value = if index % PAIR == 0 { 1.0 } else { -1.0 }),
        QueryCase::Dynamic => values.iter_mut().for_each(|value| {
            *value = rng.random_range(-1.0..1.0)
                * 2.0_f32.powi(rng.random_range(-EXPONENT_SPAN..=EXPONENT_SPAN))
        }),
        QueryCase::Reinforcement | QueryCase::Clamp => values[0] = 1.0,
    }
    values
}

fn row(case: QueryCase, index: usize, rng: &mut StdRng) -> Row {
    let mut values: Vec<f32> = (0..ENCODED_DIMENSION)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    match case {
        QueryCase::Reinforcement => {
            values.fill(0.0);
            let bits = REINFORCEMENT_THRESHOLD.to_bits();
            let cosine = match index % THRESHOLD_VARIANTS {
                0 => f32::from_bits(bits - 1),
                1 => REINFORCEMENT_THRESHOLD,
                _ => f32::from_bits(bits + 1),
            };
            values[0] = cosine;
            values[1] = (1.0 - f64::from(cosine).powi(2)).sqrt() as f32;
        }
        QueryCase::Clamp => {
            values.fill(0.0);
            values[0] = if index % PAIR == 0 { 1.0 } else { -1.0 };
        }
        QueryCase::Cancellation => {
            for pair in values.as_chunks_mut::<PAIR>().0 {
                pair[1] = pair[0];
            }
        }
        _ => {}
    }
    let kind = PATTERNS[index % PATTERNS.len()];
    let mut norm = norm_f64(&values) as f32;
    let target = match kind {
        PatternCase::ZeroNorm => Some(0.0),
        PatternCase::BelowGuard => Some(f32::from_bits(NORM_GUARD.to_bits() - 1)),
        PatternCase::AtGuard => Some(NORM_GUARD),
        PatternCase::AboveGuard => Some(f32::from_bits(NORM_GUARD.to_bits() + 1)),
        _ => None,
    };
    if let Some(target) = target {
        assert!(
            norm > 0.0,
            "fixture row must have a positive norm before scaling"
        );
        let scale = f64::from(target) / f64::from(norm);
        for value in &mut values {
            *value = (f64::from(*value) * scale) as f32;
        }
        norm = target;
    }
    Row {
        values,
        norm,
        active: !matches!(kind, PatternCase::Inactive),
    }
}

fn fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut keys = Vec::new();
    let mut rows = Vec::new();
    for case in QUERIES {
        keys.push(key(case, &mut rng));
        for index in 0..MEMORY_CAP {
            rows.push(row(case, index, &mut rng));
        }
    }
    Fixture { keys, rows }
}

fn upload(kernel: &GpuKernel, fixture: &Fixture) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    assert_eq!(agents, QUERIES.len());
    assert!(kernel.layout.brain_scratch_stride >= OUTPUT_ARRAYS * MEMORY_CAP);
    let mut brain = vec![0.0_f32; agents.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut patterns = vec![0.0_f32; agents.checked_mul(PATTERN_STRIDE).unwrap()];
    for agent in 0..agents {
        let key = agent * kernel.layout.brain_stride
            + fixed_tail_base(kernel.layout.brain_stride)
            + O_PREV_ENCODED
            - O_PREDICTOR_CONTEXT_WEIGHT;
        brain[key..key + ENCODED_DIMENSION].copy_from_slice(&fixture.keys[agent]);
        for pattern in 0..MEMORY_CAP {
            let row = &fixture.rows[agent * MEMORY_CAP + pattern];
            for (dimension, value) in row.values.iter().enumerate() {
                patterns[agent * PATTERN_STRIDE + dimension * MEMORY_CAP + pattern] = *value;
            }
            patterns[agent * PATTERN_STRIDE + O_PAT_NORMS + pattern] = row.norm;
            patterns[agent * PATTERN_STRIDE + O_PAT_ACTIVE + pattern] =
                if row.active { 1.0 } else { 0.0 };
        }
    }
    let scratch = vec![
        f32::NAN;
        agents
            .checked_mul(kernel.layout.brain_scratch_stride)
            .unwrap()
    ];
    for (buffer, values) in [
        (&kernel.brain_state_buffer, &brain),
        (&kernel.pattern_buffer, &patterns),
        (&kernel.brain_scratch_buffer, &scratch),
    ] {
        kernel
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(values));
    }
}

fn execute(kernel: &GpuKernel, cooperative: bool, fixture: &Fixture) -> TestResult<Vec<f32>> {
    upload(kernel, fixture);
    let pipeline = pipeline(kernel, cooperative);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.set_push_constants(0, bytemuck::cast_slice(&[0_u32, 0_u32]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    let bytes = read_buffer(
        kernel,
        &kernel.brain_scratch_buffer,
        kernel.brain_scratch_buffer.size(),
    )?;
    Ok(bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

fn norm_interval(square: DotMetrics) -> Interval {
    if square.reference_f64 == 0.0 {
        return Interval {
            low: 0.0,
            high: 0.0,
        };
    }
    let low = (square.reference_f64 - square.forward_bound)
        .max(0.0)
        .sqrt();
    let high = (square.reference_f64 + square.forward_bound).sqrt();
    Interval {
        low: low * (1.0 - DIVIDE_RELATIVE) / (1.0 + INVERSE_SQRT_RELATIVE),
        high: high * (1.0 + DIVIDE_RELATIVE) / (1.0 - INVERSE_SQRT_RELATIVE),
    }
}

fn cosine_interval(dot: DotMetrics, norm: Interval, pattern_norm: f32) -> Interval {
    let guard = f64::from(NORM_GUARD);
    if norm.high < guard || pattern_norm < NORM_GUARD {
        return Interval {
            low: 0.0,
            high: 0.0,
        };
    }
    let denominator_low = norm.low.max(guard) * f64::from(pattern_norm) * (1.0 - ROUND_RELATIVE);
    let denominator_high = norm.high * f64::from(pattern_norm) * (1.0 + ROUND_RELATIVE);
    assert!(denominator_low >= f64::from(f32::MIN_POSITIVE));
    assert!(denominator_high <= 2.0_f64.powi(DIVISOR_MAX_EXPONENT));
    let numerator_low = dot.reference_f64 - dot.forward_bound;
    let numerator_high = dot.reference_f64 + dot.forward_bound;
    let corners = [
        numerator_low / denominator_low,
        numerator_low / denominator_high,
        numerator_high / denominator_low,
        numerator_high / denominator_high,
    ];
    let mut low = corners.into_iter().fold(f64::INFINITY, f64::min);
    let mut high = corners.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let division_error = low.abs().max(high.abs()) * DIVIDE_RELATIVE + f64::from(f32::MIN_POSITIVE);
    low -= division_error;
    high += division_error;
    if norm.low < guard {
        low = low.min(0.0);
        high = high.max(0.0);
    }
    Interval { low, high }.clamp()
}

fn check_row(
    key: &[f32],
    row: &Row,
    output: &[f32],
    pattern: usize,
    summary: &mut Summary,
    label: &str,
) {
    let similarity = output[pattern];
    assert_eq!(
        similarity.to_bits(),
        output[RAW_CACHE + pattern].to_bits(),
        "{label}: cached recall differs"
    );
    if !row.active {
        assert_eq!(similarity, -2.0, "{label}: inactive sentinel");
        summary.inactive += 1;
        return;
    }
    summary.active += 1;
    let dot = check_fp32_dot(key, &row.values, output[RAW_DOT + pattern], label);
    let square = check_fp32_dot(key, key, output[RAW_SQUARE + pattern], label);
    let norm = norm_interval(square);
    let actual_norm = f64::from(output[RAW_NORM + pattern]);
    assert!(
        norm.contains(actual_norm),
        "{label}: norm {actual_norm:.9e} outside [{:.9e},{:.9e}]",
        norm.low,
        norm.high
    );
    let expected_norm = square.reference_f64.sqrt();
    summary.max_norm_error = summary
        .max_norm_error
        .max((actual_norm - expected_norm).abs());
    if dot.forward_bound > 0.0 {
        summary.max_dot_bound_fraction = summary
            .max_dot_bound_fraction
            .max(dot.absolute_error / dot.forward_bound);
    }
    let actual_guard = actual_norm >= f64::from(NORM_GUARD) && row.norm >= NORM_GUARD;
    let expected_guard = expected_norm >= f64::from(NORM_GUARD) && row.norm >= NORM_GUARD;
    if actual_guard != expected_guard {
        assert!(norm.contains(f64::from(NORM_GUARD)));
        summary.norm_guard_flips += 1;
    }
    if !actual_guard {
        assert_eq!(similarity, 0.0, "{label}: guarded zero cosine");
    }
    let interval = cosine_interval(dot, norm, row.norm);
    assert!(
        similarity.is_finite() && interval.contains(f64::from(similarity)),
        "{label}: cosine {similarity:.9e} outside [{:.9e},{:.9e}]",
        interval.low,
        interval.high
    );
    let reference = if expected_guard {
        (dot.reference_f64 / (expected_norm * f64::from(row.norm))).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    summary.max_cosine_error = summary
        .max_cosine_error
        .max((f64::from(similarity) - reference).abs());
    if (similarity > REINFORCEMENT_THRESHOLD) != (reference > f64::from(REINFORCEMENT_THRESHOLD)) {
        assert!(
            interval.contains(f64::from(REINFORCEMENT_THRESHOLD)),
            "{label}: unjustified reinforcement flip"
        );
        summary.reinforcement_flips += 1;
    }
}

#[test]
#[ignore = "raw recall GPU accuracy probe; run in release mode with --ignored --nocapture"]
fn recall_cosines_and_norms_satisfy_fp32_bounds() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let kernel = make_kernel();
    let fixture = fixture();
    for cooperative in [false, true] {
        let output = execute(&kernel, cooperative, &fixture)?;
        for (agent, case) in QUERIES.iter().enumerate() {
            let mut summary = Summary::default();
            let base = agent * kernel.layout.brain_scratch_stride;
            for pattern in 0..MEMORY_CAP {
                check_row(
                    &fixture.keys[agent],
                    &fixture.rows[agent * MEMORY_CAP + pattern],
                    &output[base..],
                    pattern,
                    &mut summary,
                    &format!("recall/cooperative={cooperative}/case={case:?}/pattern={pattern}"),
                );
            }
            println!("RECALL_RAW_COSINE cooperative={cooperative} case={case:?} active={} inactive={} max_norm_error={:.9e} max_cosine_error={:.9e} max_dot_error_to_bound={:.9e} norm_guard_flips={} reinforcement_threshold_flips={} guard_and_rounding_intervals_pass=true reference=f64_with_stored_f32_pattern_norm", summary.active, summary.inactive, summary.max_norm_error, summary.max_cosine_error, summary.max_dot_bound_fraction, summary.norm_guard_flips, summary.reinforcement_flips);
        }
    }
    Ok(())
}
