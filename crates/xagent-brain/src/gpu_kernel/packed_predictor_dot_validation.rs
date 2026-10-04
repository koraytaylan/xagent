//! Raw arithmetic checks of the actual scalar and packed predictor bodies.
//! GPU-trained matrices must match the independent scalar update bit for bit.
//! Raw dots use each arm's observed updated weights for an f64 forward bound,
//! separating update correctness from reduction error. The shared bound uses
//! nearest rounding plus FTZ allowances: it validates an observed compilation,
//! not a portable WGSL rounding guarantee or downstream nonlinear trajectories.

use std::error::Error;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::packed_predictor_validation::{
    copy_predictor_weights, packed_bind_group, packed_buffer, packed_common,
    packed_common_with_threads, packed_passes,
};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::check_fp32_dot;
use super::vision_validation::{make_kernel, read_buffer};
use super::*;

/// Different feature counts move the predictor's dynamic brain-state offset.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Preserve the measured scalar source's prefetch and sixteen-lane association.
const PREFETCH_FACTOR: u32 = 8;
const PREDICTOR_LANES: u32 = 16;
/// Cover both packed occupancy choices; the dispatch itself always has 256 lanes.
const PACKED_THREADS: [u32; 2] = [128, BRAIN_WORKGROUP_THREADS];
/// Reproducible inputs and weights are independent of simulation RNG state.
const DOT_SEED: u64 = 20_261_007;
/// Ten families separate training, clamp boundaries and reduction conditioning.
const CASE_COUNT: usize = 10;
const WEIGHT_LIMIT: f32 = 3.0;
const NEAR_LIMIT: f32 = 2.9999;
/// Large previous inputs force the scalar gradient clamp to ±1 exactly.
const GRADIENT_INPUT: f32 = 8.0;
/// Ordinary rows stay comfortably inside the post-update clamp.
const ORDINARY_WEIGHT: f32 = 2.0;
const PREVIOUS_PREDICTION_LIMIT: f32 = 0.75;
/// Keep derivative-zero weights nonzero so signed-zero addition is unambiguous.
const MIN_WEIGHT: f32 = 0.25;
/// Wide finite dot magnitudes remain below FP32 overflow at every intermediate.
const EXPONENT_LIMIT: i32 = 70;
const TINY_EXPONENT: i32 = -120;
const RESIDUAL_EXPONENT: i32 = -24;
/// Alternating signs and four-term residual patterns stress cancellation.
const PAIR_WIDTH: usize = 2;
const RESIDUAL_WIDTH: usize = 4;
/// Kernel-compatible layouts reserve two u32 push constants even when unused.
const PUSH_CONSTANT_BYTES: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum Case {
    Seeded,
    GradientClamp,
    WeightSaturation,
    DerivativeZero,
    Cancellation,
    Residual,
    WideExponent,
    TinyNormal,
    Subnormal,
    LastInput,
}

const CASES: [Case; CASE_COUNT] = [
    Case::Seeded,
    Case::GradientClamp,
    Case::WeightSaturation,
    Case::DerivativeZero,
    Case::Cancellation,
    Case::Residual,
    Case::WideExponent,
    Case::TinyNormal,
    Case::Subnormal,
    Case::LastInput,
];

struct Fixture {
    encoded: Vec<Vec<f32>>,
    previous: Vec<Vec<f32>>,
    predictions: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
}

struct Output {
    dots: Vec<f32>,
    weights: Vec<Vec<f32>>,
}

fn original_passes() -> String {
    predictor_width::wider_predictor(
        &dense_prefetch::prefetch_passes(
            &fuse_inline_predictor(&compose_brain_passes(true)),
            PREFETCH_FACTOR,
        ),
        PREDICTOR_LANES,
    )
}

fn function(source: &str, name: &str) -> String {
    let marker = format!("fn {name}(");
    assert_eq!(source.matches(&marker).count(), 1);
    let first = source.find(&marker).unwrap();
    let last = first + source[first..].find("\n}").unwrap() + "\n}".len();
    source[first..last].to_owned()
}

fn source(kernel: &GpuKernel, packed: bool, active_threads: u32) -> String {
    let original = original_passes();
    let passes = if packed {
        let candidate = packed_passes(&original);
        assert_ne!(candidate, original, "packed transform must execute");
        candidate
    } else {
        original
    };
    let mut source = if packed {
        if active_threads == BRAIN_WORKGROUP_THREADS {
            packed_common(kernel)
        } else {
            packed_common_with_threads(kernel, active_threads)
        }
    } else {
        include_str!("../shaders/kernel/common.wgsl").to_owned()
    };
    for declaration in [
        "const BRAIN_WORKGROUP_SIZE:",
        "const DENSE_OUTPUT_TILE:",
        "const DENSE_INNER_LANES:",
        "var<workgroup> s_encoded:",
        "var<workgroup> s_prediction:",
        "var<workgroup> s_recall:",
        "var<workgroup> s_dense_partials:",
    ] {
        let matches: Vec<_> = passes
            .lines()
            .filter(|line| line.starts_with(declaration))
            .collect();
        assert_eq!(matches.len(), 1, "unique declaration {declaration}");
        source.push('\n');
        source.push_str(matches[0]);
    }
    let body = if packed {
        function(&passes, "packed_predictor_rows")
    } else {
        let first = passes.find("fn coop_predict_and_act(").unwrap();
        let last = first
            + passes[first..]
                .find("    // ── Recalled cosine similarities:")
                .unwrap();
        format!("{}\n}}", &passes[first..last])
    };
    assert!(!body.contains("fast_tanh("));
    source.push('\n');
    source.push_str(&body);
    let entry = include_str!("packed_predictor_dot_probe.wgsl");
    assert_eq!(entry.matches("// RAW_PREDICTOR_CALL").count(), 1);
    let entry = if packed {
        entry.replace("brain_scratch[", "packed_predictor.scratch[")
    } else {
        entry.to_owned()
    };
    let call = if packed {
        "packed_predictor_rows(agent_id, tid);"
    } else {
        "coop_predict_and_act(agent_id, tid, false);"
    };
    source.push('\n');
    source.push_str(&entry.replace("// RAW_PREDICTOR_CALL", call));
    source
}

fn pipeline(kernel: &GpuKernel, packed: bool, active_threads: u32) -> wgpu::ComputePipeline {
    let label = format!("packed_predictor_raw_{packed}_{active_threads}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(source(kernel, packed, active_threads).into()),
        });
    let bind_layout = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("packed_predictor_dot_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &vision_override_constants(&kernel.layout),
                ..Default::default()
            },
            cache: None,
        })
}

fn sign(index: usize) -> f32 {
    if index % PAIR_WIDTH == 0 {
        1.0
    } else {
        -1.0
    }
}

fn fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(DOT_SEED);
    let mut fixture = Fixture {
        encoded: Vec::new(),
        previous: Vec::new(),
        predictions: Vec::new(),
        weights: Vec::new(),
    };
    for case in CASES {
        let encoded = (0..ENCODED_DIMENSION)
            .map(|input| match case {
                Case::Seeded | Case::DerivativeZero => rng.random_range(-1.0..1.0),
                Case::WideExponent => {
                    rng.random_range(-1.0..1.0)
                        * 2.0_f32.powi(rng.random_range(-EXPONENT_LIMIT..=EXPONENT_LIMIT))
                }
                Case::TinyNormal => rng.random_range(-1.0..1.0) * 2.0_f32.powi(TINY_EXPONENT),
                Case::Subnormal => f32::from_bits(rng.random_range(1..f32::MIN_POSITIVE.to_bits())),
                Case::LastInput => {
                    if input + 1 == ENCODED_DIMENSION {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 1.0,
            })
            .collect();
        let previous = (0..ENCODED_DIMENSION)
            .map(|input| match case {
                Case::Seeded | Case::DerivativeZero | Case::TinyNormal => {
                    rng.random_range(-1.0..1.0)
                }
                Case::GradientClamp | Case::WeightSaturation => sign(input) * GRADIENT_INPUT,
                _ => 0.0,
            })
            .collect();
        let predictions = (0..PREDICTOR_DIMENSION)
            .map(|dim| match case {
                Case::Seeded => {
                    rng.random_range(-PREVIOUS_PREDICTION_LIMIT..PREVIOUS_PREDICTION_LIMIT)
                }
                Case::DerivativeZero => sign(dim),
                _ => 0.0,
            })
            .collect();
        let mut weights = vec![0.0; PREDICTOR_DIMENSION.checked_mul(ENCODED_DIMENSION).unwrap()];
        for dim in 0..PREDICTOR_DIMENSION {
            for input in 0..ENCODED_DIMENSION {
                let slot = dim * ENCODED_DIMENSION + input;
                weights[slot] = match case {
                    Case::GradientClamp => 0.0,
                    Case::WeightSaturation => sign(input) * NEAR_LIMIT,
                    Case::DerivativeZero => {
                        sign(input) * rng.random_range(MIN_WEIGHT..ORDINARY_WEIGHT)
                    }
                    Case::Cancellation if input % PAIR_WIDTH != 0 => -weights[slot - 1],
                    Case::Residual => match input % RESIDUAL_WIDTH {
                        0 => ORDINARY_WEIGHT,
                        PAIR_WIDTH => -ORDINARY_WEIGHT,
                        _ => 2.0_f32.powi(RESIDUAL_EXPONENT),
                    },
                    _ => rng.random_range(-ORDINARY_WEIGHT..ORDINARY_WEIGHT),
                };
            }
        }
        fixture.encoded.push(encoded);
        fixture.previous.push(previous);
        fixture.predictions.push(predictions);
        fixture.weights.push(weights);
    }
    fixture
}

fn words(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(WORD_BYTES));
    bytes
        .as_chunks::<WORD_BYTES>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn execute(
    kernel: &GpuKernel,
    fixture: &Fixture,
    packed: bool,
    active_threads: u32,
) -> TestResult<Output> {
    assert_eq!(usize::try_from(kernel.agent_count).unwrap(), CASE_COUNT);
    let mut brain = vec![0.0_f32; CASE_COUNT.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut scratch = vec![
        0.0_f32;
        CASE_COUNT
            .checked_mul(kernel.layout.brain_scratch_stride)
            .unwrap()
    ];
    let matrix_offset = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap()
        + ENCODED_DIMENSION;
    let matrix_words = ENCODED_DIMENSION.checked_mul(PREDICTOR_DIMENSION).unwrap();
    let fixed = fixed_tail_base(kernel.layout.brain_stride);
    let previous_offset = fixed + O_PREV_ENCODED - O_PREDICTOR_CONTEXT_WEIGHT;
    let prediction_offset = fixed + O_PREV_PREDICTION - O_PREDICTOR_CONTEXT_WEIGHT;
    let output_offset = kernel.layout.feature_count + SCRATCH_PREDICTION - FEATURES_STRIDE;
    for agent in 0..CASE_COUNT {
        let base = agent * kernel.layout.brain_stride;
        brain[base + matrix_offset..base + matrix_offset + matrix_words]
            .copy_from_slice(&fixture.weights[agent]);
        brain[base + previous_offset..base + previous_offset + ENCODED_DIMENSION]
            .copy_from_slice(&fixture.previous[agent]);
        brain[base + prediction_offset..base + prediction_offset + PREDICTOR_DIMENSION]
            .copy_from_slice(&fixture.predictions[agent]);
        let base = agent * kernel.layout.brain_scratch_stride;
        scratch[base..base + ENCODED_DIMENSION].copy_from_slice(&fixture.encoded[agent]);
        // The shader only writes these output slots; no NaN enters arithmetic.
        scratch[base + output_offset..base + output_offset + PREDICTOR_DIMENSION].fill(f32::NAN);
    }
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    let pipeline = pipeline(kernel, packed, active_threads);
    let packed_storage = packed.then(|| packed_buffer(kernel));
    let packed_group = packed_storage
        .as_ref()
        .map(|buffer| packed_bind_group(kernel, buffer, kernel.active_config_index));
    let (storage, group) = if let (Some(storage), Some(group)) = (&packed_storage, &packed_group) {
        copy_predictor_weights(kernel, storage, false);
        (storage, group)
    } else {
        (
            &kernel.brain_scratch_buffer,
            &kernel.bind_groups[kernel.active_config_index],
        )
    };
    kernel
        .queue
        .write_buffer(storage, 0, bytemuck::cast_slice(&scratch));
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_push_constants(0, bytemuck::cast_slice(&[0_u32, 0_u32]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
    if packed {
        copy_predictor_weights(kernel, storage, true);
    }
    let scratch = words(&read_buffer(
        kernel,
        storage,
        kernel.brain_scratch_buffer.size(),
    )?);
    let actual = words(&read_buffer(
        kernel,
        &kernel.brain_state_buffer,
        kernel.brain_state_buffer.size(),
    )?);
    let mut output = Output {
        dots: Vec::new(),
        weights: Vec::new(),
    };
    for agent in 0..CASE_COUNT {
        let base = agent * kernel.layout.brain_scratch_stride + output_offset;
        output
            .dots
            .extend_from_slice(&scratch[base..base + PREDICTOR_DIMENSION]);
        let base = agent * kernel.layout.brain_stride;
        let first = base + matrix_offset;
        let last = first + matrix_words;
        for index in (base..first).chain(last..base + kernel.layout.brain_stride) {
            assert_eq!(
                actual[index].to_bits(),
                brain[index].to_bits(),
                "only predictor weights may change: {index}"
            );
        }
        output.weights.push(actual[first..last].to_vec());
    }
    Ok(output)
}

fn assert_updates(fixture: &Fixture, output: &Output) {
    let learning_rate = BrainConfig::default().learning_rate;
    assert!(learning_rate > WEIGHT_LIMIT - NEAR_LIMIT && learning_rate < WEIGHT_LIMIT);
    for (agent, case) in CASES.iter().enumerate() {
        for (index, &weight) in output.weights[agent].iter().enumerate() {
            assert!(weight.is_finite() && weight.abs() <= WEIGHT_LIMIT);
            let expected = match case {
                Case::GradientClamp => Some(sign(index) * learning_rate),
                Case::WeightSaturation => Some(sign(index) * WEIGHT_LIMIT),
                Case::DerivativeZero => Some(fixture.weights[agent][index]),
                _ => None,
            };
            if let Some(expected) = expected {
                assert_eq!(
                    weight.to_bits(),
                    expected.to_bits(),
                    "{case:?} weight={index}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn packed_predictor_training_and_raw_dots_match_scalar_reference() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let fixture = fixture();
    for (width, height) in FIELDS {
        let kernel = make_kernel(width, height, u32::try_from(CASE_COUNT).unwrap(), 1);
        let original = execute(&kernel, &fixture, false, BRAIN_WORKGROUP_THREADS)?;
        assert_updates(&fixture, &original);
        for threads in std::iter::once(None).chain(PACKED_THREADS.into_iter().map(Some)) {
            let packed = threads.is_some();
            let active_threads = threads.unwrap_or(BRAIN_WORKGROUP_THREADS);
            let candidate = if packed {
                Some(execute(&kernel, &fixture, true, active_threads)?)
            } else {
                None
            };
            let output = candidate.as_ref().unwrap_or(&original);
            assert_updates(&fixture, output);
            for agent in 0..CASE_COUNT {
                for (index, (&original, &candidate)) in original.weights[agent]
                    .iter()
                    .zip(&output.weights[agent])
                    .enumerate()
                {
                    assert_eq!(
                        candidate.to_bits(),
                        original.to_bits(),
                        "trained matrix active_threads={active_threads} agent={agent} weight={index}"
                    );
                }
            }
            for (agent, case) in CASES.iter().enumerate() {
                let mut max_absolute = 0.0_f64;
                let mut squared = 0.0;
                let mut max_ratio = 0.0_f64;
                let mut changed = 0;
                for dim in 0..PREDICTOR_DIMENSION {
                    let row = dim * ENCODED_DIMENSION;
                    let index = agent * PREDICTOR_DIMENSION + dim;
                    let metrics = check_fp32_dot(
                        &fixture.encoded[agent],
                        &output.weights[agent][row..row + ENCODED_DIMENSION],
                        output.dots[index],
                        &format!("packed_predictor={packed} active_threads={active_threads} {width}x{height} {case:?} dim={dim}"),
                    );
                    max_absolute = max_absolute.max(metrics.absolute_error);
                    squared += metrics.absolute_error * metrics.absolute_error;
                    if metrics.forward_bound > 0.0 {
                        max_ratio = max_ratio.max(metrics.absolute_error / metrics.forward_bound);
                    }
                    changed +=
                        usize::from(output.dots[index].to_bits() != original.dots[index].to_bits());
                }
                let rms = (squared / f64::from(u32::try_from(PREDICTOR_DIMENSION).unwrap())).sqrt();
                println!("PACKED_PREDICTOR_RAW packed={packed} active_threads={active_threads} width={width} height={height} case={case:?} seed={DOT_SEED} rows={PREDICTOR_DIMENSION} terms={ENCODED_DIMENSION} max_abs={max_absolute:.9e} rms={rms:.9e} max_error_to_forward_bound={max_ratio:.9e} raw_bit_differences={changed} trained_matrix_exact=true all_bounds_pass=true bound=nearest_gamma_2n_plus_f64_ftz observed_compilation=true");
            }
        }
    }
    Ok(())
}
