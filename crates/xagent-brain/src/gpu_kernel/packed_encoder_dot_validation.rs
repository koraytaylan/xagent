//! Raw-dot validation of the actual scalar and packed encoder functions.
//! Only terminal tanh calls are removed. Bias-plus-dot outputs are checked
//! against f64 references and derived FP32 forward bounds. Raw bit differences
//! are reported separately; these bounds do not validate nonlinear trajectories.
//! The shared checker uses round-to-nearest unit roundoff plus an FTZ allowance.
//! Passing describes the observed compilation, not a portable WGSL rounding proof.

use std::error::Error;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::packed_encoder_validation::{
    copy_encoder_weights, packed_bind_group, packed_buffer, packed_common, packed_passes,
};
use super::predictor_fusion::fuse_inline_predictor;
use super::rounding_validation::{check_fp32_dot, DotMetrics};
use super::vision_validation::{make_kernel, read_buffer};
use super::*;

/// Feature counts 267 and 342 exercise different partial prefetch chunks.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Independent fixtures cover cancellation, underflow, bias and feature tails.
const CASE_COUNT: usize = 10;
/// Match production scalar prefetch while retaining its original reduction.
const PREFETCH_FACTOR: u32 = 8;
/// Independent reproducible input generation does not alter simulation RNG.
const DOT_SEED: u64 = 20_261_006;
/// Supplied encoder weights obey the production learning clamp.
const WEIGHT_LIMIT: f32 = 2.0;
/// Broad feature magnitudes remain well below intermediate FP32 overflow.
const EXPONENT_LIMIT: i32 = 70;
/// Near-underflow normal features stress multiplication and input flushing.
const TINY_EXPONENT: i32 = -120;
/// Small terms between cancelling large terms stress discarded low bits.
const RESIDUAL_EXPONENT: i32 = -24;
/// The original encoder assigns features to four independent lane sequences.
const FEATURE_LANES: usize = 4;
/// Adjacent cancelling terms use identical feature magnitudes.
const PAIR_WIDTH: usize = 2;
/// Existing kernel layouts reserve two u32 push constants.
const PUSH_CONSTANT_BYTES: u32 = 8;
const WORD_BYTES: usize = size_of::<f32>();

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum DotCase {
    Seeded,
    Positive,
    PairCancellation,
    ResidualCancellation,
    WideExponent,
    TinyNormal,
    Subnormal,
    SparseTail,
    BiasOnly,
    SignedZero,
}

const CASES: [DotCase; CASE_COUNT] = [
    DotCase::Seeded,
    DotCase::Positive,
    DotCase::PairCancellation,
    DotCase::ResidualCancellation,
    DotCase::WideExponent,
    DotCase::TinyNormal,
    DotCase::Subnormal,
    DotCase::SparseTail,
    DotCase::BiasOnly,
    DotCase::SignedZero,
];

struct Fixture {
    features: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

#[derive(Default)]
struct Summary {
    max_absolute_error: f64,
    squared_error: f64,
    max_error_to_bound: f64,
    changed_bits: usize,
}

impl Summary {
    fn add(&mut self, metrics: DotMetrics, observed: f32, original: f32) {
        self.max_absolute_error = self.max_absolute_error.max(metrics.absolute_error);
        self.squared_error += metrics.absolute_error * metrics.absolute_error;
        if metrics.forward_bound > 0.0 {
            self.max_error_to_bound = self
                .max_error_to_bound
                .max(metrics.absolute_error / metrics.forward_bound);
        }
        self.changed_bits += usize::from(observed.to_bits() != original.to_bits());
    }
}

fn original_passes() -> String {
    dense_prefetch::prefetch_passes(
        &fuse_inline_predictor(&compose_brain_passes(true)),
        PREFETCH_FACTOR,
    )
}

fn raw_encode(passes: &str) -> String {
    let marker = "fn coop_encode(";
    assert_eq!(passes.matches(marker).count(), 1);
    let first = passes.find(marker).unwrap();
    let last = first + passes[first..].find("\n}").unwrap() + "\n}".len();
    let encode = &passes[first..last];
    assert_eq!(encode.matches("fast_tanh(").count(), 1);
    assert_eq!(encode.matches("= fast_tanh(reduced);").count(), 1);
    // Removing only the identifier leaves (operand), without reassociation.
    encode.replacen("fast_tanh(reduced)", "(reduced)", 1)
}

fn source(kernel: &GpuKernel, packed: bool) -> String {
    let original = original_passes();
    let passes = if packed {
        let candidate = packed_passes(&original);
        assert_ne!(candidate, original, "packed transform must execute");
        candidate
    } else {
        original
    };
    let mut source = if packed {
        packed_common(kernel)
    } else {
        include_str!("../shaders/kernel/common.wgsl").to_owned()
    };
    for declaration in [
        "const BRAIN_WORKGROUP_SIZE:",
        "const DENSE_OUTPUT_TILE:",
        "const DENSE_INNER_LANES:",
        "var<workgroup> s_features:",
        "var<workgroup> s_encoded:",
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
    source.push('\n');
    source.push_str(&raw_encode(&passes));
    source.push('\n');
    let entry = include_str!("packed_encoder_dot_probe.wgsl");
    if packed {
        source.push_str(&entry.replace("brain_scratch[", "packed_encoder.scratch["));
    } else {
        source.push_str(entry);
    }
    source
}

fn pipeline(kernel: &GpuKernel, packed: bool) -> wgpu::ComputePipeline {
    let label = format!("packed_encoder_raw_{packed}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(source(kernel, packed).into()),
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
            entry_point: Some("packed_encoder_dot_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &vision_override_constants(&kernel.layout),
                ..Default::default()
            },
            cache: None,
        })
}

fn fixture(features: usize) -> Fixture {
    let mut rng = StdRng::seed_from_u64(DOT_SEED);
    let mut fixture = Fixture {
        features: Vec::new(),
        weights: Vec::new(),
        biases: Vec::new(),
    };
    for case in CASES {
        let inputs = (0..features)
            .map(|feature| match case {
                DotCase::Seeded => rng.random_range(-1.0..1.0),
                DotCase::Positive => rng.random_range(0.0..1.0),
                DotCase::WideExponent => {
                    rng.random_range(-1.0..1.0)
                        * 2.0_f32.powi(rng.random_range(-EXPONENT_LIMIT..=EXPONENT_LIMIT))
                }
                DotCase::TinyNormal => rng.random_range(-1.0..1.0) * 2.0_f32.powi(TINY_EXPONENT),
                DotCase::Subnormal => {
                    f32::from_bits(rng.random_range(1..f32::MIN_POSITIVE.to_bits()))
                }
                DotCase::SparseTail => {
                    if feature + 1 == features {
                        1.0
                    } else {
                        0.0
                    }
                }
                DotCase::BiasOnly => 0.0,
                DotCase::SignedZero => {
                    if feature % PAIR_WIDTH == 0 {
                        -0.0
                    } else {
                        0.0
                    }
                }
                _ => 1.0,
            })
            .collect();
        let mut matrix = vec![0.0; features.checked_mul(ENCODED_DIMENSION).unwrap()];
        for feature in 0..features {
            for dim in 0..ENCODED_DIMENSION {
                let slot = feature * ENCODED_DIMENSION + dim;
                matrix[slot] = match case {
                    DotCase::Positive => rng.random_range(0.0..=WEIGHT_LIMIT),
                    DotCase::PairCancellation if feature % PAIR_WIDTH != 0 => {
                        -matrix[slot - ENCODED_DIMENSION]
                    }
                    DotCase::ResidualCancellation => match feature % FEATURE_LANES {
                        0 => WEIGHT_LIMIT,
                        PAIR_WIDTH => -WEIGHT_LIMIT,
                        _ => 2.0_f32.powi(RESIDUAL_EXPONENT),
                    },
                    _ => rng.random_range(-WEIGHT_LIMIT..=WEIGHT_LIMIT),
                };
            }
        }
        let biases = (0..ENCODED_DIMENSION)
            .map(|_| match case {
                DotCase::TinyNormal | DotCase::Subnormal => 0.0,
                DotCase::SignedZero => -0.0,
                _ => rng.random_range(-1.0..1.0),
            })
            .collect();
        fixture.features.push(inputs);
        fixture.weights.push(matrix);
        fixture.biases.push(biases);
    }
    fixture
}

fn execute(kernel: &GpuKernel, fixture: &Fixture, packed: bool) -> TestResult<Vec<f32>> {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    assert_eq!(agents, CASE_COUNT);
    let mut brain = vec![0.0_f32; agents.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut scratch = vec![
        0.0_f32;
        agents
            .checked_mul(kernel.layout.brain_scratch_stride)
            .unwrap()
    ];
    let weights = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    for agent in 0..agents {
        let brain_base = agent * kernel.layout.brain_stride;
        brain[brain_base..brain_base + weights].copy_from_slice(&fixture.weights[agent]);
        brain[brain_base + weights..brain_base + weights + ENCODED_DIMENSION]
            .copy_from_slice(&fixture.biases[agent]);
        let scratch_base = agent * kernel.layout.brain_scratch_stride;
        scratch[scratch_base..scratch_base + kernel.layout.feature_count]
            .copy_from_slice(&fixture.features[agent]);
        let output = scratch_base + kernel.layout.feature_count;
        // Only the final publication accesses these slots in this shader;
        // sentinels detect unwritten outputs and never enter GPU arithmetic.
        scratch[output..output + ENCODED_DIMENSION].fill(f32::NAN);
    }
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    let pipeline = pipeline(kernel, packed);
    let packed_storage = packed.then(|| packed_buffer(kernel));
    let packed_group = packed_storage
        .as_ref()
        .map(|buffer| packed_bind_group(kernel, buffer, kernel.active_config_index));
    let (storage, group) = if let (Some(storage), Some(group)) = (&packed_storage, &packed_group) {
        copy_encoder_weights(kernel, storage, false);
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
    let bytes = read_buffer(kernel, storage, kernel.brain_scratch_buffer.size())?;
    let words: Vec<_> = bytes
        .as_chunks::<WORD_BYTES>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let mut output = Vec::with_capacity(agents.checked_mul(ENCODED_DIMENSION).unwrap());
    for agent in 0..agents {
        let first = agent * kernel.layout.brain_scratch_stride + kernel.layout.feature_count;
        output.extend_from_slice(&words[first..first + ENCODED_DIMENSION]);
    }
    Ok(output)
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn packed_encoder_raw_dots_satisfy_fp32_bounds() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let kernel = make_kernel(width, height, u32::try_from(CASE_COUNT).unwrap(), 1);
        let fixture = fixture(kernel.layout.feature_count);
        let original = execute(&kernel, &fixture, false)?;
        let candidate = execute(&kernel, &fixture, true)?;
        for (packed, outputs) in [(false, &original), (true, &candidate)] {
            for (agent, case) in CASES.iter().enumerate() {
                let mut summary = Summary::default();
                let mut inputs = fixture.features[agent].clone();
                inputs.push(1.0);
                for dim in 0..ENCODED_DIMENSION {
                    let mut weights: Vec<_> = (0..kernel.layout.feature_count)
                        .map(|feature| fixture.weights[agent][feature * ENCODED_DIMENSION + dim])
                        .collect();
                    weights.push(fixture.biases[agent][dim]);
                    let index = agent * ENCODED_DIMENSION + dim;
                    let metrics = check_fp32_dot(
                        &inputs,
                        &weights,
                        outputs[index],
                        &format!("packed_encoder={packed} {width}x{height} {case:?} dim={dim}"),
                    );
                    summary.add(metrics, outputs[index], original[index]);
                }
                let rms = (summary.squared_error
                    / f64::from(u32::try_from(ENCODED_DIMENSION).unwrap()))
                .sqrt();
                println!("PACKED_ENCODER_RAW packed={packed} width={width} height={height} features={} rows={ENCODED_DIMENSION} case={case:?} seed={DOT_SEED} max_abs={:.9e} rms={rms:.9e} max_error_to_forward_bound={:.9e} raw_bit_differences={} includes_bias=true all_bounds_pass=true bound=nearest_gamma_2n_plus_f64_ftz observed_compilation=true", kernel.layout.feature_count, summary.max_absolute_error, summary.max_error_to_bound, summary.changed_bits);
            }
        }
    }
    Ok(())
}
