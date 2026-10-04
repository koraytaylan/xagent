//! Independent raw arithmetic checks for the next-input projection prototype.
//! Candidate functions come from the actual timing source. Matrix updates use
//! the original scalar credit shader as an oracle, and raw encoder outputs use
//! the existing fresh-dot FP64 reference and FP32 forward budget unchanged.

use std::error::Error;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::fresh_projection_validation::{
    common_source, credit_source, encode_source, ProjectionLayout,
};
use super::packed_store_validation::optimized_brain;
use super::rounding_validation::{check_fp32_dot, DotMetrics};
use super::vision_validation::{make_kernel, read_buffer};
use super::*;

/// Two raw fields exercise different final eight-feature credit tiles.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const CASE_COUNT: usize = 12;
const SEED: u64 = 20_261_014;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;
const THREADS: u32 = 256;
const WEIGHT_LIMIT: f32 = 2.0;
const OUTSIDE_LIMIT: f32 = 3.0;
const NEAR_LIMIT: f32 = 1.9;
const CREDIT_THRESHOLD: f32 = 1e-6;
const LARGE_CREDIT: f32 = 2.0;
const LARGE_CREDIT_INPUT: f32 = 10_000.0;
const ORDINARY_CREDIT: f32 = 0.25;
const CREDIT_CASES: usize = 8;
const FEATURE_LANES: usize = 4;
const PAIR: usize = 2;
const EXPONENT_LIMIT: i32 = 30;
const RESIDUAL_EXPONENT: i32 = -24;
const TINY_EXPONENT: i32 = -120;
/// Finite, deliberately implausible value detects unwritten diagnostic slots.
const SENTINEL: f32 = -123_456.0;
const MIXED_CREDIT_AGENT: usize = 9;
const NEAR_ZERO_AGENT: usize = 10;
const RESET_SEED: u64 = 2026;

const CREDIT_ENTRY: &str = r"
@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn credit_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    phase_encoder_credit(group.y, group.x * ENCODER_CREDIT_THREADS + tid);
}
";
const ENCODE_ENTRY: &str = r"
var<workgroup> raw_probe_alive: u32;
@compute @workgroup_size(BRAIN_WORKGROUP_SIZE)
fn encode_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let agent_id = group.x;
    if (tid == 0u) {
        raw_probe_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
    }
    if (workgroupUniformLoad(&raw_probe_alive) == 0u) { return; }
    for (var feature = tid; feature < FEATURE_COUNT; feature += BRAIN_WORKGROUP_SIZE) {
        s_features[feature] = sensory_buffer[agent_id * SENSORY_STRIDE + feature];
    }
    workgroupBarrier();
    coop_encode(agent_id, tid);
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum Case {
    Seeded,
    Positive,
    PairCancellation,
    ResidualCancellation,
    WideExponent,
    TinyNormal,
    Subnormal,
    SparseTail,
    BiasOnly,
    MixedCredit,
    NearZeroChanged,
    SignedZero,
}

const CASES: [Case; CASE_COUNT] = [
    Case::Seeded,
    Case::Positive,
    Case::PairCancellation,
    Case::ResidualCancellation,
    Case::WideExponent,
    Case::TinyNormal,
    Case::Subnormal,
    Case::SparseTail,
    Case::BiasOnly,
    Case::MixedCredit,
    Case::NearZeroChanged,
    Case::SignedZero,
];

struct AgentFixture {
    predicted_inputs: Vec<f32>,
    credit_inputs: Vec<f32>,
    weights: Vec<f32>,
    biases: Vec<f32>,
    credits: Vec<f32>,
}

fn fixtures(features: usize, visual: usize) -> Vec<AgentFixture> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let below = f32::from_bits(CREDIT_THRESHOLD.to_bits() - 1);
    let above = f32::from_bits(CREDIT_THRESHOLD.to_bits() + 1);
    let gates = [
        below,
        CREDIT_THRESHOLD,
        above,
        LARGE_CREDIT,
        -below,
        -CREDIT_THRESHOLD,
        -above,
        -LARGE_CREDIT,
    ];
    CASES
        .into_iter()
        .map(|case| {
            let predicted_inputs = (0..features)
                .map(|feature| match case {
                    Case::Seeded => rng.random_range(-1.0..1.0),
                    Case::Positive => rng.random_range(0.0..1.0),
                    Case::WideExponent => {
                        rng.random_range(-1.0..1.0)
                            * 2.0_f32.powi(rng.random_range(-EXPONENT_LIMIT..=EXPONENT_LIMIT))
                    }
                    Case::TinyNormal => rng.random_range(-1.0..1.0) * 2.0_f32.powi(TINY_EXPONENT),
                    Case::Subnormal => {
                        f32::from_bits(rng.random_range(1..f32::MIN_POSITIVE.to_bits()))
                    }
                    Case::SparseTail => {
                        if feature + 1 == visual || feature + 1 == features {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Case::BiasOnly => 0.0,
                    Case::SignedZero => {
                        if feature % PAIR == 0 {
                            -0.0
                        } else {
                            0.0
                        }
                    }
                    Case::NearZeroChanged => {
                        if feature < visual {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0,
                })
                .collect();
            let credit_inputs = (0..features)
                .map(|_| match case {
                    Case::MixedCredit => LARGE_CREDIT_INPUT,
                    Case::Seeded | Case::Positive => rng.random_range(-1.0..1.0),
                    _ => 0.0,
                })
                .collect();
            let mut weights = vec![0.0; features.checked_mul(ENCODED_DIMENSION).unwrap()];
            for feature in 0..features {
                for output in 0..ENCODED_DIMENSION {
                    let address = feature * ENCODED_DIMENSION + output;
                    weights[address] = match case {
                        Case::Positive => rng.random_range(0.0..=WEIGHT_LIMIT),
                        Case::PairCancellation if feature % PAIR != 0 => {
                            -weights[address - ENCODED_DIMENSION]
                        }
                        Case::ResidualCancellation => match feature % FEATURE_LANES {
                            0 => WEIGHT_LIMIT,
                            PAIR => -WEIGHT_LIMIT,
                            _ => 2.0_f32.powi(RESIDUAL_EXPONENT),
                        },
                        Case::MixedCredit => match output % CREDIT_CASES {
                            0 => {
                                if feature % PAIR == 0 {
                                    -0.0
                                } else {
                                    OUTSIDE_LIMIT
                                }
                            }
                            FEATURE_LANES => {
                                if feature % PAIR == 0 {
                                    0.0
                                } else {
                                    -OUTSIDE_LIMIT
                                }
                            }
                            3 => NEAR_LIMIT,
                            7 => -NEAR_LIMIT,
                            _ => ORDINARY_CREDIT,
                        },
                        _ => rng.random_range(-WEIGHT_LIMIT..=WEIGHT_LIMIT),
                    };
                }
            }
            let biases = (0..ENCODED_DIMENSION)
                .map(|_| match case {
                    Case::TinyNormal | Case::Subnormal | Case::NearZeroChanged => 0.0,
                    Case::SignedZero => -0.0,
                    _ => rng.random_range(-1.0..1.0),
                })
                .collect();
            let credits = (0..ENCODED_DIMENSION)
                .map(|output| match case {
                    Case::MixedCredit => gates[output % CREDIT_CASES],
                    Case::Seeded | Case::Positive => {
                        if output % PAIR == 0 {
                            ORDINARY_CREDIT
                        } else {
                            -ORDINARY_CREDIT
                        }
                    }
                    _ => 0.0,
                })
                .collect();
            AgentFixture {
                predicted_inputs,
                credit_inputs,
                weights,
                biases,
                credits,
            }
        })
        .collect()
}

#[derive(Default)]
struct Summary {
    rows: usize,
    max_abs: f64,
    squared_error: f64,
    max_over_bound: f64,
    reference_bit_differences: usize,
}

impl Summary {
    fn add(&mut self, metrics: DotMetrics, actual: f32, reference: f32) {
        self.rows += 1;
        self.max_abs = self.max_abs.max(metrics.absolute_error);
        self.squared_error += metrics.absolute_error * metrics.absolute_error;
        self.max_over_bound = self
            .max_over_bound
            .max(metrics.absolute_error / metrics.forward_bound);
        self.reference_bit_differences += usize::from(actual.to_bits() != reference.to_bits());
    }
}

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn word(raw: &[u8], index: usize) -> f32 {
    let start = index.checked_mul(WORD_BYTES).unwrap();
    f32::from_le_bytes(raw[start..start + WORD_BYTES].try_into().unwrap())
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fresh_projection_raw_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fresh_projection_raw_probe"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fresh_projection_raw_probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &vision_override_constants(&kernel.layout),
                ..Default::default()
            },
            cache: None,
        })
}

struct Probe {
    layout: ProjectionLayout,
    cache: packed_encoder::Cache,
    binding: wgpu::BindGroup,
    original_credit: wgpu::ComputePipeline,
    candidate_credit: wgpu::ComputePipeline,
    original_encode: wgpu::ComputePipeline,
    candidate_encode: wgpu::ComputePipeline,
}

impl Probe {
    fn new(kernel: &GpuKernel) -> Self {
        let layout = ProjectionLayout::new(kernel);
        let cache =
            packed_encoder::Cache::new_with_extra_scratch(kernel, layout.extra_words).unwrap();
        let binding = global_credit::private_bind_group(
            kernel,
            &kernel.kernel_pipeline.get_bind_group_layout(0),
            cache.buffer(),
            kernel.active_config_index,
        );
        let common = common_source(kernel, &cache, &layout);
        let original_credit = pipeline(
            kernel,
            [
                include_str!("../shaders/kernel/common.wgsl"),
                include_str!("../shaders/kernel/phase_encoder_credit.wgsl"),
                CREDIT_ENTRY,
            ]
            .join("\n"),
            "credit_probe",
        );
        let candidate_entry = CREDIT_ENTRY.replace(
            "phase_encoder_credit(group.y, group.x * ENCODER_CREDIT_THREADS + tid);",
            "projection_credit_group(group.y, group.x, tid);",
        );
        assert_ne!(candidate_entry, CREDIT_ENTRY);
        let candidate_credit = pipeline(
            kernel,
            format!("{common}\n{}\n{candidate_entry}", credit_source()),
            "credit_probe",
        );
        let passes = packed_encoder::packed_passes(&optimized_brain());
        let mut declarations = String::new();
        for prefix in [
            "const BRAIN_WORKGROUP_SIZE:",
            "const DENSE_OUTPUT_TILE:",
            "const DENSE_INNER_LANES:",
            "var<workgroup> s_features:",
            "var<workgroup> s_encoded:",
            "var<workgroup> s_dense_partials:",
            "var<workgroup> s_reinf_dot:",
        ] {
            let lines: Vec<_> = passes
                .lines()
                .filter(|line| line.starts_with(prefix))
                .collect();
            assert_eq!(lines.len(), 1, "unique encoder dependency {prefix}");
            declarations.push_str(lines[0]);
            declarations.push('\n');
        }
        let encode = encode_source();
        assert!(encode.contains("PROJECTION_RAW_OFFSET"));
        assert!(encode.contains("PROJECTION_USED_OFFSET"));
        let candidate_encode = pipeline(
            kernel,
            format!("{common}\n{declarations}\n{encode}\n{ENCODE_ENTRY}"),
            "encode_probe",
        );
        // The reference body is the original canonical packed encoder with
        // the timing source's raw publication. No tanh inverse or duplicate
        // arithmetic is used, and the cached function is omitted entirely.
        let marker = "// Cached visual partials";
        assert_eq!(encode.matches(marker).count(), 1);
        let fresh = encode.split_once(marker).unwrap().0;
        assert!(!fresh.contains("fn coop_encode("));
        let entry = ENCODE_ENTRY.replace(
            "coop_encode(agent_id, tid);",
            "fresh_projection_original_encode(agent_id, tid);",
        );
        let original_encode = pipeline(
            kernel,
            format!("{common}\n{declarations}\n{fresh}\n{entry}"),
            "encode_probe",
        );
        Self {
            layout,
            cache,
            binding,
            original_credit,
            candidate_credit,
            original_encode,
            candidate_encode,
        }
    }

    fn private_word(&self, agent: usize, offset: usize) -> usize {
        self.layout.base_words + agent * self.layout.agent_stride + offset
    }

    fn set_private(&self, kernel: &GpuKernel, agent: usize, offset: usize, value: f32) {
        kernel.queue.write_buffer(
            self.cache.buffer(),
            bytes(self.private_word(agent, offset)),
            bytemuck::bytes_of(&value),
        );
    }

    fn read(&self, kernel: &GpuKernel) -> TestResult<Vec<u8>> {
        read_buffer(kernel, self.cache.buffer(), self.cache.buffer().size())
    }

    fn import(&self, kernel: &GpuKernel) {
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        assert!(self.cache.record_import(kernel, &mut encoder));
        kernel.queue.submit([encoder.finish()]);
        kernel.poll_wait();
    }
}

fn dispatch(
    kernel: &GpuKernel,
    pipeline: &wgpu::ComputePipeline,
    binding: &wgpu::BindGroup,
    groups: [u32; 2],
) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, binding, &[]);
        pass.set_push_constants(0, bytemuck::cast_slice(&[0_u32, 0_u32]));
        pass.dispatch_workgroups(groups[0], groups[1], 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

fn upload_inputs(kernel: &GpuKernel, inputs: &[Vec<f32>]) {
    assert!(kernel.layout.feature_count <= kernel.layout.sensory_stride);
    let mut sensory = vec![
        0.0;
        CASE_COUNT
            .checked_mul(kernel.layout.sensory_stride)
            .unwrap()
    ];
    for (agent, values) in inputs.iter().enumerate() {
        assert_eq!(values.len(), kernel.layout.feature_count);
        let begin = agent * kernel.layout.sensory_stride;
        sensory[begin..begin + values.len()].copy_from_slice(values);
    }
    kernel
        .queue
        .write_buffer(&kernel.sensory_buffer, 0, bytemuck::cast_slice(&sensory));
}

fn assert_matrices(kernel: &GpuKernel, probe: &Probe, brain: &[u8], private: &[u8]) {
    let words = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    for agent in 0..CASE_COUNT {
        let public_first =
            usize::try_from(bytes(agent * kernel.layout.brain_stride + O_ENC_WEIGHTS)).unwrap();
        let private_first =
            usize::try_from(bytes(probe.layout.prefix_words + agent * words)).unwrap();
        let count = usize::try_from(bytes(words)).unwrap();
        assert_eq!(
            &brain[public_first..public_first + count],
            &private[private_first..private_first + count],
            "private/scalar matrix agent={agent}"
        );
    }
}

/// Independent scalar credit executes before resetting the same input matrix
/// for the candidate. Private export is never used to make scalar parity pass.
fn prepare_credit(
    kernel: &GpuKernel,
    probe: &Probe,
    fixture: &[AgentFixture],
    dead: Option<usize>,
) -> TestResult<Vec<u8>> {
    let features = kernel.layout.feature_count;
    let matrix_words = features.checked_mul(ENCODED_DIMENSION).unwrap();
    let mut brain = vec![0.0; CASE_COUNT.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut decisions = vec![0.0; CASE_COUNT.checked_mul(DECISION_STRIDE).unwrap()];
    let mut scratch = vec![0.0; probe.layout.prefix_words];
    for (agent, fixture) in fixture.iter().enumerate() {
        let base = agent * kernel.layout.brain_stride + O_ENC_WEIGHTS;
        brain[base..base + matrix_words].copy_from_slice(&fixture.weights);
        brain[base + matrix_words..base + matrix_words + ENCODED_DIMENSION]
            .copy_from_slice(&fixture.biases);
        let begin = agent * kernel.layout.brain_scratch_stride + SCRATCH_FEATURES;
        scratch[begin..begin + features].copy_from_slice(&fixture.credit_inputs);
        let begin = agent * DECISION_STRIDE + DECISION_CREDIT;
        decisions[begin..begin + ENCODED_DIMENSION].copy_from_slice(&fixture.credits);
        let raw = probe.private_word(agent, probe.layout.raw_offset);
        scratch[raw..raw + ENCODED_DIMENSION].fill(SENTINEL);
        kernel.write_agent_physics_fields(
            u32::try_from(agent).unwrap(),
            &[
                (P_ALIVE, if dead == Some(agent) { 0.0 } else { 1.0 }),
                (P_DEATH_COUNT, 0.0),
            ],
        );
    }
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    kernel
        .queue
        .write_buffer(&kernel.decision_buffer, 0, bytemuck::cast_slice(&decisions));
    kernel.queue.write_buffer(
        &kernel.brain_scratch_buffer,
        0,
        bytemuck::cast_slice(&scratch[..CASE_COUNT * kernel.layout.brain_scratch_stride]),
    );
    kernel
        .queue
        .write_buffer(probe.cache.buffer(), 0, bytemuck::cast_slice(&scratch));
    upload_inputs(
        kernel,
        &fixture
            .iter()
            .map(|agent| agent.predicted_inputs.clone())
            .collect::<Vec<_>>(),
    );
    let scalar_groups = u32::try_from(matrix_words).unwrap().div_ceil(THREADS);
    dispatch(
        kernel,
        &probe.original_credit,
        &kernel.bind_groups[kernel.active_config_index],
        [scalar_groups, kernel.agent_count],
    );
    let expected = read_buffer(
        kernel,
        &kernel.brain_state_buffer,
        kernel.brain_state_buffer.size(),
    )?;
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    probe.cache.invalidate();
    probe.import(kernel);
    dispatch(
        kernel,
        &probe.candidate_credit,
        &probe.binding,
        [probe.cache.groups_per_agent(), kernel.agent_count],
    );
    let actual = read_buffer(
        kernel,
        &kernel.brain_state_buffer,
        kernel.brain_state_buffer.size(),
    )?;
    assert_eq!(
        expected, actual,
        "actual candidate credit must match scalar shader, dead={dead:?}"
    );
    assert_matrices(kernel, probe, &actual, &probe.read(kernel)?);
    let mixed = &fixture[MIXED_CREDIT_AGENT];
    for feature in 0..features {
        for output in 0..ENCODED_DIMENSION {
            let slot = feature * ENCODED_DIMENSION + output;
            let value = word(
                &actual,
                MIXED_CREDIT_AGENT * kernel.layout.brain_stride + O_ENC_WEIGHTS + slot,
            );
            if dead == Some(MIXED_CREDIT_AGENT)
                || matches!(output % CREDIT_CASES, 0 | FEATURE_LANES)
            {
                assert_eq!(value.to_bits(), mixed.weights[slot].to_bits());
            } else if output % CREDIT_CASES == 3 {
                assert_eq!(value, WEIGHT_LIMIT);
            } else if output % CREDIT_CASES == 7 {
                assert_eq!(value, -WEIGHT_LIMIT);
            } else {
                assert_ne!(
                    value.to_bits(),
                    mixed.weights[slot].to_bits(),
                    "threshold-equal and above must update"
                );
            }
        }
    }
    Ok(actual)
}

fn predicted_inputs(
    kernel: &GpuKernel,
    probe: &Probe,
    fixture: &[AgentFixture],
) -> TestResult<Vec<Vec<f32>>> {
    let private = probe.read(kernel)?;
    Ok(fixture
        .iter()
        .enumerate()
        .map(|(agent, fixture)| {
            let mut input = fixture.predicted_inputs.clone();
            for feature in 0..probe.layout.visual_count {
                input[feature] = word(
                    &private,
                    agent * kernel.layout.brain_scratch_stride + SCRATCH_FEATURES + feature,
                );
            }
            input
        })
        .collect())
}

fn encode(
    kernel: &GpuKernel,
    probe: &Probe,
    candidate: bool,
    inputs: &[Vec<f32>],
) -> TestResult<Vec<u8>> {
    upload_inputs(kernel, inputs);
    for agent in 0..CASE_COUNT {
        kernel.queue.write_buffer(
            probe.cache.buffer(),
            bytes(probe.private_word(agent, probe.layout.raw_offset)),
            bytemuck::cast_slice(&[SENTINEL; ENCODED_DIMENSION]),
        );
    }
    dispatch(
        kernel,
        if candidate {
            &probe.candidate_encode
        } else {
            &probe.original_encode
        },
        &probe.binding,
        [kernel.agent_count, 1],
    );
    probe.read(kernel)
}

fn check_outputs(
    kernel: &GpuKernel,
    probe: &Probe,
    brain: &[u8],
    inputs: &[Vec<f32>],
    reference: &[u8],
    candidate: &[u8],
    expected_reuse: &[bool],
    dead: Option<usize>,
    label: &str,
) {
    for (agent, case) in CASES.iter().enumerate() {
        let output = probe.private_word(agent, probe.layout.raw_offset);
        if dead == Some(agent) {
            for dimension in 0..ENCODED_DIMENSION {
                assert_eq!(
                    word(reference, output + dimension).to_bits(),
                    SENTINEL.to_bits()
                );
                assert_eq!(
                    word(candidate, output + dimension).to_bits(),
                    SENTINEL.to_bits()
                );
            }
            continue;
        }
        assert_eq!(
            word(
                candidate,
                probe.private_word(agent, probe.layout.used_offset)
            ),
            f32::from(u8::from(expected_reuse[agent])),
            "reuse gate {label} agent={agent}"
        );
        let mut input = inputs[agent].clone();
        input.push(1.0);
        let mut summary = Summary::default();
        for dimension in 0..ENCODED_DIMENSION {
            let mut weights: Vec<_> = (0..kernel.layout.feature_count)
                .map(|feature| {
                    word(
                        brain,
                        agent * kernel.layout.brain_stride
                            + O_ENC_WEIGHTS
                            + feature * ENCODED_DIMENSION
                            + dimension,
                    )
                })
                .collect();
            weights.push(word(
                brain,
                agent * kernel.layout.brain_stride
                    + O_ENC_WEIGHTS
                    + kernel.layout.feature_count * ENCODED_DIMENSION
                    + dimension,
            ));
            let actual = word(candidate, output + dimension);
            let original = word(reference, output + dimension);
            let name = format!("fresh projection {label} {case:?} dim={dimension}");
            check_fp32_dot(&input, &weights, original, &format!("original {name}"));
            let metrics = check_fp32_dot(&input, &weights, actual, &name);
            if !expected_reuse[agent] {
                assert_eq!(
                    actual.to_bits(),
                    original.to_bits(),
                    "canonical fresh fallback {name}"
                );
            }
            summary.add(metrics, actual, original);
        }
        let rms = (summary.squared_error / summary.rows as f64).sqrt();
        println!("FRESH_PROJECTION_RAW label={label} case={case:?} features={} rows={} cached={} max_abs={:.9e} rms={rms:.9e} max_error_to_fresh_bound={:.9e} raw_bit_differences={} actual_credit_matrix_exact=true private_mirror_exact=true bound=unchanged_nearest_gamma_2n_f64_ftz timing_source_raw_publication=true observed_compilation=true", kernel.layout.feature_count, summary.rows, expected_reuse[agent], summary.max_abs, summary.max_over_bound, summary.reference_bit_differences);
    }
}

fn compare_encode(
    kernel: &GpuKernel,
    probe: &Probe,
    brain: &[u8],
    inputs: &[Vec<f32>],
    expected_reuse: &[bool],
    dead: Option<usize>,
    label: &str,
) -> TestResult<Vec<u8>> {
    let reference = encode(kernel, probe, false, inputs)?;
    let candidate = encode(kernel, probe, true, inputs)?;
    check_outputs(
        kernel,
        probe,
        brain,
        inputs,
        &reference,
        &candidate,
        expected_reuse,
        dead,
        label,
    );
    Ok(candidate)
}

#[test]
#[ignore = "requires GPU; actual cached/fallback raw dots and independent scalar credit"]
fn fresh_projection_raw_dots_and_fallback_satisfy_fresh_bounds() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = make_kernel(width, height, u32::try_from(CASE_COUNT).unwrap(), 1);
        let probe = Probe::new(&kernel);
        let fixture = fixtures(kernel.layout.feature_count, probe.layout.visual_count);
        let brain = prepare_credit(&kernel, &probe, &fixture, None)?;
        let predicted = predicted_inputs(&kernel, &probe, &fixture)?;
        let fast = compare_encode(
            &kernel,
            &probe,
            &brain,
            &predicted,
            &[true; CASE_COUNT],
            None,
            "matching_prediction",
        )?;

        // Deliberately invalid cache data proves the fast route reads cached
        // partials. These poisoned results are excluded from numerical checks.
        for agent in 0..CASE_COUNT {
            probe.set_private(&kernel, agent, probe.layout.partial_offset, SENTINEL);
        }
        let poisoned = encode(&kernel, &probe, true, &predicted)?;
        for agent in 0..CASE_COUNT {
            assert_ne!(
                word(
                    &poisoned,
                    probe.private_word(agent, probe.layout.raw_offset)
                )
                .to_bits(),
                word(&fast, probe.private_word(agent, probe.layout.raw_offset)).to_bits(),
                "cached path must consume its partials"
            );
        }
        let mut changed = predicted.clone();
        for (agent, inputs) in changed.iter_mut().enumerate() {
            let last = probe.layout.visual_count - 1;
            inputs[last] = f32::from_bits(inputs[last].to_bits() ^ 1);
            if agent == NEAR_ZERO_AGENT {
                inputs[..probe.layout.visual_count].fill(2.0_f32.powi(RESIDUAL_EXPONENT));
            }
        }
        compare_encode(
            &kernel,
            &probe,
            &brain,
            &changed,
            &[false; CASE_COUNT],
            None,
            "changed_input_poisoned_cache",
        )?;

        // A changed death generation must reject otherwise matching q and
        // deliberately corrupt cached partials without reading them.
        for agent in 0..CASE_COUNT {
            kernel
                .write_agent_physics_fields(u32::try_from(agent).unwrap(), &[(P_DEATH_COUNT, 1.0)]);
        }
        compare_encode(
            &kernel,
            &probe,
            &brain,
            &predicted,
            &[false; CASE_COUNT],
            None,
            "death_generation_poisoned_cache",
        )?;
        for agent in 0..CASE_COUNT {
            kernel
                .write_agent_physics_fields(u32::try_from(agent).unwrap(), &[(P_DEATH_COUNT, 0.0)]);
            probe.set_private(&kernel, agent, probe.layout.valid_offset, 0.0);
        }
        compare_encode(
            &kernel,
            &probe,
            &brain,
            &predicted,
            &[false; CASE_COUNT],
            None,
            "invalid_poisoned_cache",
        )?;

        // Rebuild a valid projection after the invalid-cache case. Verify that
        // matching input actually reuses it, then poison its first partial.
        // Thus reset cannot pass merely because metadata was already invalid.
        assert_eq!(prepare_credit(&kernel, &probe, &fixture, None)?, brain);
        let warm = encode(&kernel, &probe, true, &predicted)?;
        for agent in 0..CASE_COUNT {
            assert_eq!(
                word(&warm, probe.private_word(agent, probe.layout.valid_offset)),
                1.0
            );
            assert_eq!(
                word(&warm, probe.private_word(agent, probe.layout.used_offset)),
                1.0
            );
            probe.set_private(&kernel, agent, probe.layout.partial_offset, SENTINEL);
        }
        let before_reset = probe.read(&kernel)?;
        for agent in 0..CASE_COUNT {
            assert_eq!(
                word(
                    &before_reset,
                    probe.private_word(agent, probe.layout.valid_offset)
                ),
                1.0,
                "host reset must start with a warm valid projection"
            );
            assert_eq!(
                word(
                    &before_reset,
                    probe.private_word(agent, probe.layout.partial_offset)
                ),
                SENTINEL
            );
        }

        // Use the real host reset, then explicitly clear the private header
        // and reimport scalar matrices, as the prototype recorder does. This
        // detached test cache is not part of the production constructor API.
        kernel.reset_agents_seeded(
            &BrainConfig {
                vision_width: width,
                vision_height: height,
                vision_stride: 1,
                ..BrainConfig::default()
            },
            RESET_SEED,
        );
        probe.cache.invalidate();
        assert!(!probe.cache.is_valid());
        for agent in 0..CASE_COUNT {
            kernel.queue.write_buffer(
                probe.cache.buffer(),
                bytes(probe.private_word(agent, 0)),
                bytemuck::cast_slice(&vec![0.0_f32; probe.layout.raw_offset]),
            );
        }
        probe.import(&kernel);
        let reset_brain = read_buffer(
            &kernel,
            &kernel.brain_state_buffer,
            kernel.brain_state_buffer.size(),
        )?;
        assert_matrices(&kernel, &probe, &reset_brain, &probe.read(&kernel)?);
        compare_encode(
            &kernel,
            &probe,
            &reset_brain,
            &predicted,
            &[false; CASE_COUNT],
            None,
            "host_reset_import_fallback",
        )?;

        let dead_brain = prepare_credit(&kernel, &probe, &fixture, Some(MIXED_CREDIT_AGENT))?;
        let predicted = predicted_inputs(&kernel, &probe, &fixture)?;
        compare_encode(
            &kernel,
            &probe,
            &dead_brain,
            &predicted,
            &[true; CASE_COUNT],
            Some(MIXED_CREDIT_AGENT),
            "inactive_credit_and_brain",
        )?;
        println!("FRESH_PROJECTION_GATES width={width} height={height} fixtures={CASE_COUNT} scalar_credit_exact=true clamp_threshold_disabled_outside_limit=true partial_poison_falsifiable=true changed_input_fallback=true death_generation_fallback=true host_reset_import_fallback=true reset_started_with_valid_poisoned_projection=true detached_header_reset_explicit=true inactive_agent_preserved=true");
    }
    Ok(())
}
