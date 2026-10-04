//! Snapshot-only FP64 visual-projection recurrence with rounded weight updates.
//! Each cached projection is initialized from actual storage, then advanced for
//! one, eight or sixteen cycles. The current matrix is still read in full here;
//! this measures numerical opportunity, not GPU work saved or an implementation.
//! Candidate GPU roundoff and altered downstream trajectories are not modeled.

use std::error::Error;

use super::cycle_profile::{assert_state_equal, checkpoint, restore};
use super::packed_store_validation::{advance, prepare_kernel_with_store_suppression};
use super::rounding_validation::fp32_dot_reference;
use super::vision_validation::read_buffer;
use super::visual_event_diagnostics::{
    adaptation_rate, check_adaptation, number, sample, steady_agent, Sample,
};
use super::*;

/// Match the raw-field production store-suppression fixture.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
const SEEDS: [u64; 3] = [42, 314, 2026];
const WARMUP_CYCLES: [u32; 2] = [256, 1_000];
const WINDOW_LENGTHS: [usize; 3] = [1, 8, 16];
/// Sixteen observed transitions give two complete eight-cycle windows.
const OBSERVATIONS: usize = 16;
const PHYSICS: usize = 0;
const DECISIONS: usize = 1;
const SENSORY: usize = 7;
const BRAIN: usize = 8;
const MUTABLE_BUFFERS: usize = 13;
const WORD_BYTES: usize = size_of::<f32>();
/// Encoder updates clamp enabled components to the canonical signed range.
const WEIGHT_LIMIT: f64 = 2.0;
/// Uniform storage uses the same four-word vectors as common.wgsl.
const UNIFORM_VECTOR_WORDS: usize = 4;
const ALIVE_THRESHOLD: f32 = 0.5;
const METRICS: [&str; 7] = [
    "fresh_minus_cached",
    "adaptation_residual",
    "stored_weight_residual",
    "ideal_clamp_residual",
    "post_clamp_rounding_residual",
    "discarded_increment_residual",
    "absolute_residual_envelope",
];

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn shader_constant(name: &str) -> f32 {
    let prefix = format!("const {name}: f32 = ");
    let values: Vec<_> = include_str!("../shaders/kernel/common.wgsl")
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .collect();
    assert_eq!(values.len(), 1);
    let value: f32 = values[0].split(';').next().unwrap().parse().unwrap();
    assert!(value.is_normal() && value > 0.0);
    value
}

/// The production uniform has no COPY_SRC usage. This independent read-only
/// dispatch copies one word into a diagnostic buffer without simulation writes.
fn read_learning_rate(kernel: &GpuKernel) -> TestResult<f32> {
    assert_eq!(CONFIG_SIZE % UNIFORM_VECTOR_WORDS, 0);
    let source = format!(
        "@group(0) @binding(0) var<uniform> config: array<vec4<f32>, {}>;\n\
         @group(0) @binding(1) var<storage, read_write> output: array<u32>;\n\
         @compute @workgroup_size(1) fn read_rate() {{\n\
         output[0] = bitcast<u32>(config[{}][{}]);\n}}",
        CONFIG_SIZE / UNIFORM_VECTOR_WORDS,
        CFG_LEARNING_RATE / UNIFORM_VECTOR_WORDS,
        CFG_LEARNING_RATE % UNIFORM_VECTOR_WORDS,
    );
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recurrence_learning_rate_readback"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("recurrence_learning_rate_readback"),
            layout: None,
            module: &module,
            entry_point: Some("read_rate"),
            compilation_options: Default::default(),
            cache: None,
        });
    let output = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("recurrence_learning_rate_word"),
        size: u64::try_from(WORD_BYTES).unwrap(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bindings = kernel.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("recurrence_learning_rate_readback"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: kernel.brain_config_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    let rate = number(&read_buffer(kernel, &output, output.size())?, 0);
    assert!(rate.is_normal() && rate > 0.0);
    Ok(rate)
}

#[derive(Clone, Copy, Default)]
struct UpdateCounts {
    enabled_outputs: u64,
    disabled_outputs: u64,
    threshold_equal_outputs: u64,
    visual_weights: u64,
    nonzero_ideal_increments: u64,
    unchanged_nonzero_increments: u64,
    ideal_clamp_crossings: u64,
    stored_at_limit: u64,
    residual_max_abs: f64,
    residual_squared_sum: f64,
}

impl UpdateCounts {
    fn merge(&mut self, other: Self) {
        self.enabled_outputs += other.enabled_outputs;
        self.disabled_outputs += other.disabled_outputs;
        self.threshold_equal_outputs += other.threshold_equal_outputs;
        self.visual_weights += other.visual_weights;
        self.nonzero_ideal_increments += other.nonzero_ideal_increments;
        self.unchanged_nonzero_increments += other.unchanged_nonzero_increments;
        self.ideal_clamp_crossings += other.ideal_clamp_crossings;
        self.stored_at_limit += other.stored_at_limit;
        self.residual_max_abs = self.residual_max_abs.max(other.residual_max_abs);
        self.residual_squared_sum += other.residual_squared_sum;
    }

    fn report(self, label: &str, learning_rate: f32, scale: f32, threshold: f32) {
        let rms = (self.residual_squared_sum / self.visual_weights.max(1) as f64).sqrt();
        println!("RAW_RECURRENCE_UPDATES {label} enabled_outputs={} below_threshold_outputs={} exactly_at_threshold_outputs={} visual_weights={} nonzero_ideal_increments={} bit_unchanged_nonzero_increments={} ideal_unrounded_clamp_crossings={} observed_stored_at_limit={} update_residual_max_abs={:.9e} update_residual_rms={rms:.9e} learning_rate={learning_rate:.9e} credit_scale={scale:.9e} credit_threshold={threshold:.9e} learning_rate_source=actual_gpu_uniform coefficient=ideal_f64_product_of_f32_inputs coefficient_rounding_included_in_residual=true clamp_events_not_inferred_from_stored_endpoint=true unchanged_increment_subset_includes_saturation=true counts_once_per_transition=true",
            self.enabled_outputs, self.disabled_outputs, self.threshold_equal_outputs,
            self.visual_weights, self.nonzero_ideal_increments, self.unchanged_nonzero_increments,
            self.ideal_clamp_crossings, self.stored_at_limit, self.residual_max_abs);
    }
}

#[derive(Clone, Copy, Default)]
struct Metric {
    rows: u64,
    exceeds: u64,
    max_abs: f64,
    squared_sum: f64,
    max_over_budget: f64,
}

impl Metric {
    fn observe(&mut self, value: f64, budget: f64) {
        assert!(value.is_finite() && budget.is_finite() && budget > 0.0);
        self.rows += 1;
        self.exceeds += u64::from(value.abs() > budget);
        self.max_abs = self.max_abs.max(value.abs());
        self.squared_sum += value * value;
        self.max_over_budget = self.max_over_budget.max(value.abs() / budget);
    }

    fn merge(&mut self, other: Self) {
        self.rows += other.rows;
        self.exceeds += other.exceeds;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.squared_sum += other.squared_sum;
        self.max_over_budget = self.max_over_budget.max(other.max_over_budget);
    }
}

#[derive(Clone, Copy, Default)]
struct WindowCounts {
    eligible_agent_prefixes: u64,
    inactive_agent_prefixes: u64,
    transition_or_invalid_window_prefixes: u64,
    metrics: [Metric; METRICS.len()],
    endpoint: Metric,
    total_error_by_prefix: [Metric; OBSERVATIONS],
    decomposition_residual_max: f64,
}

impl WindowCounts {
    fn merge(&mut self, other: Self) {
        self.eligible_agent_prefixes += other.eligible_agent_prefixes;
        self.inactive_agent_prefixes += other.inactive_agent_prefixes;
        self.transition_or_invalid_window_prefixes += other.transition_or_invalid_window_prefixes;
        for (total, value) in self.metrics.iter_mut().zip(other.metrics) {
            total.merge(value);
        }
        self.endpoint.merge(other.endpoint);
        for (total, value) in self
            .total_error_by_prefix
            .iter_mut()
            .zip(other.total_error_by_prefix)
        {
            total.merge(value);
        }
        self.decomposition_residual_max = self
            .decomposition_residual_max
            .max(other.decomposition_residual_max);
    }

    fn report(self, label: &str, length: usize, beta: f64) {
        let ratios: Vec<_> = self.total_error_by_prefix[..length]
            .iter()
            .map(|metric| metric.max_over_budget)
            .collect();
        println!("RAW_RECURRENCE_WINDOW {label} window_cycles={length} eligible_agent_prefixes={} inactive_prefixes_excluded={} death_reset_or_invalid_window_prefixes_excluded={} endpoint_rows={} endpoint_exceeds_budget={} endpoint_max_over_budget={:.9e} max_over_budget_by_prefix={ratios:?} decomposition_residual_max={:.9e} beta_f64={beta:.17e} initialize=actual_before_weight_times_actual_adapted_visual delta_raw=before_cycle_sensory current_weight=before_current_credit norm=actual_previous_visual_only nonvisual_and_bias=fresh budget=full_feature_plus_bias_nearest_gamma_2n_f64_ftz candidate_gpu_roundoff=omitted altered_trajectory=unmodeled acceptance_not_asserted=true timing_not_measured=true replay_exact_buffers={MUTABLE_BUFFERS}",
            self.eligible_agent_prefixes, self.inactive_agent_prefixes,
            self.transition_or_invalid_window_prefixes, self.endpoint.rows,
            self.endpoint.exceeds, self.endpoint.max_over_budget, self.decomposition_residual_max);
        for (name, metric) in METRICS.into_iter().zip(self.metrics) {
            let rms = (metric.squared_sum / metric.rows.max(1) as f64).sqrt();
            println!("RAW_RECURRENCE_ERROR {label} window_cycles={length} component={name} rows={} exceeds_fresh_dot_budget={} max_abs={:.9e} rms={rms:.9e} max_over_budget={:.9e}", metric.rows, metric.exceeds, metric.max_abs, metric.max_over_budget);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct RowTransition {
    previous_projection: f64,
    current_projection: f64,
    raw_delta_projection: f64,
    ideal_update_projection: f64,
    adaptation_projection: f64,
    weight_projection: f64,
    clamp_projection: f64,
    rounding_projection: f64,
    discarded_projection: f64,
    adaptation_envelope: f64,
    weight_envelope: f64,
    budget: f64,
}

/// Classify saturation without adding the increment to the stored weight.
/// Even FP64 addition can discard a tiny outward increment at a clamp limit.
fn ideal_clamp_residual(old_weight: f32, increment: f64, enabled: bool) -> Option<f64> {
    if !enabled {
        return None;
    }
    let upper_margin = WEIGHT_LIMIT - f64::from(old_weight);
    let lower_margin = -WEIGHT_LIMIT - f64::from(old_weight);
    if increment > upper_margin {
        Some(upper_margin - increment)
    } else if increment < lower_margin {
        Some(lower_margin - increment)
    } else {
        None
    }
}

#[test]
fn clamp_classification_preserves_sub_float64_ulp_outward_increments() {
    /// The stored encoder endpoints are exactly representable in both formats.
    const STORED_LIMIT: f32 = 2.0;
    /// Smaller than the FP64 spacing at either clamp endpoint.
    const TINY_INCREMENT: f64 = f64::EPSILON * f64::EPSILON;
    /// A normal interior point stays inside the range after this exact update.
    const INTERIOR_WEIGHT: f32 = 1.0;
    const INTERIOR_INCREMENT: f64 = 0.25;
    assert_eq!(
        (WEIGHT_LIMIT + TINY_INCREMENT).to_bits(),
        WEIGHT_LIMIT.to_bits(),
        "adding first loses the positive outward increment"
    );
    assert_eq!(
        (-WEIGHT_LIMIT - TINY_INCREMENT).to_bits(),
        (-WEIGHT_LIMIT).to_bits(),
        "adding first loses the negative outward increment"
    );
    assert_eq!(
        ideal_clamp_residual(STORED_LIMIT, TINY_INCREMENT, true),
        Some(-TINY_INCREMENT)
    );
    assert_eq!(
        ideal_clamp_residual(-STORED_LIMIT, -TINY_INCREMENT, true),
        Some(TINY_INCREMENT)
    );
    assert_eq!(
        ideal_clamp_residual(STORED_LIMIT, TINY_INCREMENT, false),
        None,
        "disabled credit does not execute the clamp"
    );
    assert_eq!(
        ideal_clamp_residual(INTERIOR_WEIGHT, INTERIOR_INCREMENT, true),
        None
    );
    assert_eq!(
        ideal_clamp_residual(STORED_LIMIT, -TINY_INCREMENT, true),
        None,
        "an inward increment does not cross the limit"
    );
}

/// W_next is the current BEFORE matrix, and equals the previous AFTER matrix.
/// R = W_next - (W_previous + previous_adapted * ideal_credit_scale).
/// Every projection and norm here uses visual rows only.
fn transitions(
    kernel: &GpuKernel,
    previous: &Sample,
    current: &Sample,
    agent: usize,
    learning_rate: f32,
    credit_scale: f32,
    threshold: f32,
    beta: f64,
) -> (Vec<RowTransition>, UpdateCounts) {
    let features = kernel.layout.feature_count;
    let visual = kernel.layout.vision_color_count + kernel.layout.vision_depth_count;
    let brain = agent * kernel.layout.brain_stride;
    let feature_base = agent * features;
    let sensory = agent * kernel.layout.sensory_stride;
    let mut input = current.features[feature_base..feature_base + features].to_vec();
    input.push(1.0);
    let previous_norm: f64 = previous.features[feature_base..feature_base + visual]
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum();
    let mut counts = UpdateCounts::default();
    let mut rows = Vec::with_capacity(ENCODED_DIMENSION);
    for output in 0..ENCODED_DIMENSION {
        let credit = number(
            &previous.after[DECISIONS],
            agent * DECISION_STRIDE + DECISION_CREDIT + output,
        );
        let enabled = credit.abs() >= threshold;
        counts.enabled_outputs += u64::from(enabled);
        counts.disabled_outputs += u64::from(!enabled);
        counts.threshold_equal_outputs += u64::from(credit.abs() == threshold);
        let coefficient = if enabled {
            f64::from(learning_rate) * f64::from(credit) * f64::from(credit_scale)
        } else {
            0.0
        };
        let mut row = RowTransition {
            ideal_update_projection: coefficient * previous_norm,
            ..Default::default()
        };
        let mut weights = Vec::with_capacity(features + 1);
        for feature in 0..features {
            let address = brain + O_ENC_WEIGHTS + feature * ENCODED_DIMENSION + output;
            let old_weight = number(&previous.before[BRAIN], address);
            let weight = number(&current.before[BRAIN], address);
            weights.push(weight);
            if !enabled {
                assert_eq!(
                    old_weight.to_bits(),
                    weight.to_bits(),
                    "below-threshold credit changed a weight"
                );
            } else {
                assert!(f64::from(weight).abs() <= WEIGHT_LIMIT);
            }
            if feature >= visual {
                continue;
            }
            let old_adapted = f64::from(previous.features[feature_base + feature]);
            let adapted = f64::from(current.features[feature_base + feature]);
            let delta_raw = f64::from(number(&current.before[SENSORY], sensory + feature))
                - f64::from(number(&previous.before[SENSORY], sensory + feature));
            let adaptation_residual = adapted - (beta * old_adapted + delta_raw);
            let increment = coefficient * old_adapted;
            // Subtract the two stored f32 values first. Forming old+increment
            // first could discard an increment even in this FP64 diagnostic.
            let residual = (f64::from(weight) - f64::from(old_weight)) - increment;
            let clamp = ideal_clamp_residual(old_weight, increment, enabled);
            let clamp_residual = clamp.unwrap_or(0.0);
            let rounding_residual = residual - clamp_residual;
            let discarded = enabled && increment != 0.0 && old_weight.to_bits() == weight.to_bits();
            counts.visual_weights += 1;
            counts.nonzero_ideal_increments += u64::from(enabled && increment != 0.0);
            counts.unchanged_nonzero_increments += u64::from(discarded);
            counts.ideal_clamp_crossings += u64::from(clamp.is_some());
            counts.stored_at_limit += u64::from(enabled && f64::from(weight).abs() == WEIGHT_LIMIT);
            counts.residual_max_abs = counts.residual_max_abs.max(residual.abs());
            counts.residual_squared_sum += residual * residual;
            row.previous_projection += f64::from(old_weight) * old_adapted;
            row.current_projection += f64::from(weight) * adapted;
            row.raw_delta_projection += f64::from(weight) * delta_raw;
            row.adaptation_projection += f64::from(weight) * adaptation_residual;
            row.weight_projection += residual * old_adapted;
            row.clamp_projection += clamp_residual * old_adapted;
            row.rounding_projection += rounding_residual * old_adapted;
            if discarded {
                row.discarded_projection += residual * old_adapted;
            }
            row.adaptation_envelope += (f64::from(weight) * adaptation_residual).abs();
            row.weight_envelope += (residual * old_adapted).abs();
        }
        weights.push(number(
            &current.before[BRAIN],
            brain + features * ENCODED_DIMENSION + output,
        ));
        row.budget =
            fp32_dot_reference(&input, &weights, "fresh full encoder dot with bias").forward_bound;
        rows.push(row);
    }
    (rows, counts)
}

#[derive(Clone, Copy, Default)]
struct CachedRow {
    projection: f64,
    adaptation: f64,
    weight: f64,
    clamp: f64,
    rounding: f64,
    discarded: f64,
    envelope: f64,
}

impl CachedRow {
    fn advance(&mut self, row: RowTransition, beta: f64) -> [f64; METRICS.len()] {
        // z_next = beta * (z + c * ||v_previous_visual||²) + W_nextᵀ Δraw.
        self.projection =
            beta * (self.projection + row.ideal_update_projection) + row.raw_delta_projection;
        self.adaptation = beta * self.adaptation + row.adaptation_projection;
        self.weight = beta * (self.weight + row.weight_projection);
        self.clamp = beta * (self.clamp + row.clamp_projection);
        self.rounding = beta * (self.rounding + row.rounding_projection);
        self.discarded = beta * (self.discarded + row.discarded_projection);
        self.envelope = beta * (self.envelope + row.weight_envelope) + row.adaptation_envelope;
        [
            row.current_projection - self.projection,
            self.adaptation,
            self.weight,
            self.clamp,
            self.rounding,
            self.discarded,
            self.envelope,
        ]
    }
}

struct Window {
    length: usize,
    agents: Vec<Option<Vec<CachedRow>>>,
    counts: WindowCounts,
}

impl Window {
    fn new(length: usize, agents: usize) -> Self {
        assert_eq!(OBSERVATIONS % length, 0);
        Self {
            length,
            agents: vec![None; agents],
            counts: WindowCounts::default(),
        }
    }

    fn observe(
        &mut self,
        index: usize,
        agent: usize,
        rows: Option<&[RowTransition]>,
        inactive: bool,
        beta: f64,
    ) {
        let prefix = index % self.length;
        if prefix == 0 {
            self.agents[agent] = rows.map(|rows| {
                rows.iter()
                    .map(|row| CachedRow {
                        projection: row.previous_projection,
                        ..Default::default()
                    })
                    .collect()
            });
        }
        if rows.is_none() {
            self.agents[agent] = None;
        }
        let Some((cached, rows)) = self.agents[agent].as_mut().zip(rows) else {
            if inactive {
                self.counts.inactive_agent_prefixes += 1;
            } else {
                self.counts.transition_or_invalid_window_prefixes += 1;
            }
            return;
        };
        self.counts.eligible_agent_prefixes += 1;
        for (cached, &row) in cached.iter_mut().zip(rows) {
            let errors = cached.advance(row, beta);
            for (metric, error) in self.counts.metrics.iter_mut().zip(errors) {
                metric.observe(error, row.budget);
            }
            self.counts.total_error_by_prefix[prefix].observe(errors[0], row.budget);
            if prefix + 1 == self.length {
                self.counts.endpoint.observe(errors[0], row.budget);
            }
            // This residual reports host FP64 cancellation/summation noise;
            // it is not an additional FP32 acceptance allowance.
            self.counts.decomposition_residual_max = self
                .counts
                .decomposition_residual_max
                .max((errors[0] - errors[1] - errors[2]).abs());
        }
    }
}

fn all_inactive(previous: &Sample, current: &Sample, agent: usize) -> bool {
    [
        &previous.before,
        &previous.after,
        &current.before,
        &current.after,
    ]
    .iter()
    .all(|state| number(&state[PHYSICS], agent * PHYS_STRIDE + P_ALIVE) < ALIVE_THRESHOLD)
}

fn assert_source_contract() {
    let source = packed_encoder::CREDIT_SOURCE;
    assert!(source.contains("let learning_rate = brain_config[1].x;"));
    assert_eq!(CFG_LEARNING_RATE, UNIFORM_VECTOR_WORDS);
    assert!(source.contains("let credit_enabled = abs(credits) >= vec4<f32>(CREDIT_EPSILON);"));
    for component in ["x", "y", "z", "w"] {
        assert!(source.contains(&format!(
            "let scale = learning_rate * credits.{component} * ENCODER_CREDIT_SCALE;"
        )));
        assert!(source.contains(&format!(
            "weight.{component} = clamp(weight.{component} + scale * input, -2.0, 2.0);"
        )));
    }
}

#[test]
#[ignore = "requires GPU; snapshot-only FP64 recurrence opportunity, not numerical acceptance"]
fn cached_visual_projection_with_stored_weight_residuals() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    assert_source_contract();
    let mut kernel = prepare_kernel_with_store_suppression(WIDTH, HEIGHT, false, true);
    assert!(!kernel.layout.visual_cortex_enabled);
    let world = checkpoint(&kernel);
    let rate = adaptation_rate();
    let beta = 1.0 - f64::from(rate);
    let scale = shader_constant("ENCODER_CREDIT_SCALE");
    let threshold = shader_constant("CREDIT_EPSILON");
    let brain = BrainConfig {
        vision_width: WIDTH,
        vision_height: HEIGHT,
        vision_stride: 1,
        ..BrainConfig::default()
    };
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let mut totals = [[WindowCounts::default(); WINDOW_LENGTHS.len()]; WARMUP_CYCLES.len()];
    let mut total_updates = [UpdateCounts::default(); WARMUP_CYCLES.len()];
    for seed in SEEDS {
        restore(&mut kernel, &world);
        kernel.reset_agents_seeded(&brain, seed);
        let learning_rate = read_learning_rate(&kernel)?;
        assert_eq!(
            learning_rate.to_bits(),
            build_config_for(&brain, &kernel.layout)[CFG_LEARNING_RATE].to_bits()
        );
        let mut cycle = 0;
        for (warm_index, start) in WARMUP_CYCLES.into_iter().enumerate() {
            advance(&mut kernel, cycle, start - 1 - cycle);
            let mut previous = sample(&mut kernel, start - 1)?;
            cycle = start;
            let mut windows: Vec<_> = WINDOW_LENGTHS
                .into_iter()
                .map(|length| Window::new(length, agents))
                .collect();
            let mut updates = UpdateCounts::default();
            for index in 0..OBSERVATIONS {
                let current = sample(&mut kernel, cycle)?;
                assert_state_equal(&kernel, &previous.after, &current.before);
                for agent in 0..agents {
                    let rows = if steady_agent(&previous, agent) && steady_agent(&current, agent) {
                        check_adaptation(&kernel, &previous, agent, rate);
                        check_adaptation(&kernel, &current, agent, rate);
                        let tick = agent * kernel.layout.brain_stride
                            + fixed_tail_base(kernel.layout.brain_stride)
                            + O_TICK_COUNT
                            - O_PREDICTOR_CONTEXT_WEIGHT;
                        assert_eq!(
                            number(&current.after[BRAIN], tick),
                            number(&previous.after[BRAIN], tick) + 1.0
                        );
                        let (rows, counts) = transitions(
                            &kernel,
                            &previous,
                            &current,
                            agent,
                            learning_rate,
                            scale,
                            threshold,
                            beta,
                        );
                        updates.merge(counts);
                        Some(rows)
                    } else {
                        None
                    };
                    for window in &mut windows {
                        window.observe(
                            index,
                            agent,
                            rows.as_deref(),
                            all_inactive(&previous, &current, agent),
                            beta,
                        );
                    }
                }
                previous = current;
                cycle += 1;
            }
            let label =
                format!("seed={seed} warmup_cycles={start} observed_transitions={OBSERVATIONS}");
            updates.report(&label, learning_rate, scale, threshold);
            total_updates[warm_index].merge(updates);
            for (window_index, window) in windows.into_iter().enumerate() {
                let counts = window.counts;
                assert!(counts.eligible_agent_prefixes > 0);
                assert!(counts.endpoint.rows > 0);
                assert_eq!(
                    counts.eligible_agent_prefixes
                        + counts.inactive_agent_prefixes
                        + counts.transition_or_invalid_window_prefixes,
                    u64::try_from(OBSERVATIONS * agents).unwrap()
                );
                assert_eq!(
                    counts.metrics[0].rows,
                    counts.eligible_agent_prefixes * u64::try_from(ENCODED_DIMENSION).unwrap()
                );
                counts.report(&label, window.length, beta);
                totals[warm_index][window_index].merge(counts);
            }
        }
    }
    let learning_rate = read_learning_rate(&kernel)?;
    for (warm_index, start) in WARMUP_CYCLES.into_iter().enumerate() {
        let label = format!(
            "seed=all seeds={} warmup_cycles={start} observed_transitions_per_seed={OBSERVATIONS}",
            SEEDS.len()
        );
        total_updates[warm_index].report(&label, learning_rate, scale, threshold);
        for (length, counts) in WINDOW_LENGTHS.into_iter().zip(totals[warm_index]) {
            counts.report(&label, length, beta);
        }
    }
    Ok(())
}
