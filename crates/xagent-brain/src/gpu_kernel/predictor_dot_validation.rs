//! Hardware-only raw predictor-dot validation. The GPU executes the production
//! fused update/partial-sum prefix and the actual width transformation's
//! reduction, with zero training inputs. Scalar and production-prefetch source
//! paths both run every width. Raw FP32 outputs before context/tanh are checked
//! against f64 forward bounds for seeded and adversarial rows.

use std::error::Error;

use rand::{rngs::StdRng, Rng, SeedableRng};

use super::cycle_profile::make_kernel;
use super::dense_prefetch::prefetch_passes;
use super::predictor_fusion::fuse_inline_predictor;
use super::predictor_width::{wider_predictor, LANE_WIDTHS};
use super::rounding_validation::{check_fp32_dot, DotMetrics};
use super::vision_validation::read_buffer;
use super::*;

/// Cover the scalar loop and production prefetch composition independently.
const PREFETCH_ARMS: [bool; 2] = [false, true];
/// Production prefetch loads this many independent terms before arithmetic.
const PREFETCH_FACTOR: u32 = 8;
/// Reproducible FP32 input and row-weight generation, independent of world RNG.
const DOT_SEED: u64 = 20_261_004;
/// The existing cycle-profile fixture provides one workgroup per case.
const CASE_COUNT: usize = 10;
/// Kernel layouts reserve the two-u32 push-constant range even when unused.
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Every supplied predictor weight lies within the production update clamp.
const WEIGHT_LIMIT: f32 = 3.0;
/// Keep nonzero mantissas away from zero while varying their FP32 low bits.
const MIN_MANTISSA: f32 = 0.5;
/// Broad input magnitudes stress cancellation without approaching overflow.
const EXPONENT_LIMIT: i32 = 80;
/// Tiny normal inputs also exercise products near the subnormal boundary.
const TINY_NORMAL_EXPONENT: i32 = -120;
/// This residual is lost when added to the largest supplied weight.
const SMALLEST_RESIDUAL_EXPONENT: i32 = -24;
/// Larger residuals cover cancellation with multiple rounding severities.
const LARGEST_RESIDUAL_EXPONENT: i32 = -12;
/// Adjacent pairs have identical input magnitudes and opposite weights.
const PAIR_WIDTH: usize = 2;
/// The cancellation pattern separates a large cancelling pair by small terms.
const RESIDUAL_PATTERN_WIDTH: usize = 4;

#[derive(Clone, Copy, Debug)]
enum DotCase {
    Seeded,
    Positive,
    AlternatingCancellation,
    CancellationResidual,
    ExponentSweep,
    TinyNormal,
    Subnormal,
    PairedWideCancellation,
    Sparse,
    ClusteredCancellation,
}

const CASES: [DotCase; CASE_COUNT] = [
    DotCase::Seeded,
    DotCase::Positive,
    DotCase::AlternatingCancellation,
    DotCase::CancellationResidual,
    DotCase::ExponentSweep,
    DotCase::TinyNormal,
    DotCase::Subnormal,
    DotCase::PairedWideCancellation,
    DotCase::Sparse,
    DotCase::ClusteredCancellation,
];

struct Fixture {
    inputs: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
}

#[derive(Default)]
struct DotSummary {
    max_absolute_error: f64,
    squared_error: f64,
    max_error_to_bound: f64,
    max_error_to_product_sum: f64,
    different_from_scalar_four_lanes: usize,
}

impl DotSummary {
    fn add(&mut self, metrics: DotMetrics, observed: f32, original: f32) {
        self.max_absolute_error = self.max_absolute_error.max(metrics.absolute_error);
        self.squared_error += metrics.absolute_error * metrics.absolute_error;
        if metrics.forward_bound > 0.0 {
            self.max_error_to_bound = self
                .max_error_to_bound
                .max(metrics.absolute_error / metrics.forward_bound);
        }
        if metrics.absolute_product_sum > 0.0 {
            self.max_error_to_product_sum = self
                .max_error_to_product_sum
                .max(metrics.absolute_error / metrics.absolute_product_sum);
        }
        self.different_from_scalar_four_lanes +=
            usize::from(observed.to_bits() != original.to_bits());
    }
}

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn dot_source(lanes: u32, prefetch: bool) -> String {
    let passes = fuse_inline_predictor(&compose_brain_passes(true));
    let passes = if prefetch {
        prefetch_passes(&passes, PREFETCH_FACTOR)
    } else {
        passes
    };
    let passes = wider_predictor(&passes, lanes);
    let start = passes.find("fn coop_predict_and_act(").unwrap();
    let end = start
        + passes[start..]
            .find("    // ── Recalled cosine similarities:")
            .unwrap();
    let prefix = &passes[start..end];
    if prefetch {
        for item in 0..PREFETCH_FACTOR {
            assert!(prefix.contains(&format!(
                "partial += encoded_{item} * updated_weight_{item};"
            )));
        }
        assert!(!prefix.contains("partial += s_encoded[j] * w;"));
    } else {
        assert!(prefix.contains("partial += s_encoded[j] * w;"));
    }
    assert!(!prefix.contains("fast_tanh("));
    let mut source = include_str!("../shaders/kernel/common.wgsl").to_owned();
    // Extract declarations from the same source rather than duplicating their
    // dimensions or the arithmetic/reduction implementation in this probe.
    for declaration in [
        "const BRAIN_WORKGROUP_SIZE:",
        "const DENSE_OUTPUT_TILE:",
        "const DENSE_INNER_LANES:",
        "var<workgroup> s_encoded:",
        "var<workgroup> s_recall:",
        "var<workgroup> s_prediction:",
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
    source.push_str(prefix);
    source.push_str("}\n");
    source.push_str(include_str!("predictor_dot_probe.wgsl"));
    source
}

fn make_pipeline(kernel: &GpuKernel, lanes: u32, prefetch: bool) -> wgpu::ComputePipeline {
    let label = format!("raw_predictor_dot_{lanes}_prefetch_{prefetch}");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(dot_source(lanes, prefetch).into()),
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
    let constants = vision_override_constants(&kernel.layout);
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("predictor_dot_probe"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn case_inputs(case: DotCase, rng: &mut StdRng) -> Vec<f32> {
    let mut inputs = vec![1.0; ENCODED_DIMENSION];
    for index in 0..inputs.len() {
        inputs[index] = match case {
            DotCase::Seeded | DotCase::Sparse => rng.random_range(-1.0..1.0),
            DotCase::Positive => rng.random_range(MIN_MANTISSA..1.0),
            DotCase::ExponentSweep => {
                rng.random_range(-1.0..1.0)
                    * 2.0_f32.powi(rng.random_range(-EXPONENT_LIMIT..=EXPONENT_LIMIT))
            }
            DotCase::TinyNormal => rng.random_range(-1.0..1.0) * 2.0_f32.powi(TINY_NORMAL_EXPONENT),
            DotCase::Subnormal => f32::from_bits(rng.random_range(1..f32::MIN_POSITIVE.to_bits())),
            DotCase::PairedWideCancellation => {
                if index % PAIR_WIDTH == 0 {
                    rng.random_range(MIN_MANTISSA..1.0)
                        * 2.0_f32.powi(rng.random_range(-EXPONENT_LIMIT..=EXPONENT_LIMIT))
                } else {
                    inputs[index - 1]
                }
            }
            _ => 1.0,
        };
    }
    inputs
}

fn case_weights(case: DotCase, row: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut weights = vec![0.0; ENCODED_DIMENSION];
    let residual =
        2.0_f32.powi(rng.random_range(SMALLEST_RESIDUAL_EXPONENT..=LARGEST_RESIDUAL_EXPONENT));
    for index in 0..weights.len() {
        weights[index] = match case {
            DotCase::Positive => rng.random_range(MIN_MANTISSA..=WEIGHT_LIMIT),
            DotCase::AlternatingCancellation | DotCase::PairedWideCancellation => {
                if index % PAIR_WIDTH == 0 {
                    rng.random_range(MIN_MANTISSA..=WEIGHT_LIMIT)
                } else {
                    -weights[index - 1]
                }
            }
            DotCase::CancellationResidual => match index % RESIDUAL_PATTERN_WIDTH {
                0 => WEIGHT_LIMIT,
                1 => residual,
                remainder if remainder == PAIR_WIDTH => -WEIGHT_LIMIT,
                _ => residual,
            },
            DotCase::Sparse => {
                if index == row && row != 0 {
                    rng.random_range(-WEIGHT_LIMIT..=WEIGHT_LIMIT)
                } else {
                    0.0
                }
            }
            DotCase::ClusteredCancellation => {
                if index < ENCODED_DIMENSION / PAIR_WIDTH {
                    rng.random_range(MIN_MANTISSA..=WEIGHT_LIMIT)
                } else {
                    -weights[index - ENCODED_DIMENSION / PAIR_WIDTH]
                }
            }
            _ => rng.random_range(-WEIGHT_LIMIT..=WEIGHT_LIMIT),
        };
    }
    weights
}

fn fixture() -> Fixture {
    let mut rng = StdRng::seed_from_u64(DOT_SEED);
    let mut inputs = Vec::with_capacity(CASE_COUNT);
    let mut weights = Vec::with_capacity(CASE_COUNT * PREDICTOR_DIMENSION);
    for case in CASES {
        inputs.push(case_inputs(case, &mut rng));
        for row in 0..PREDICTOR_DIMENSION {
            weights.push(case_weights(case, row, &mut rng));
        }
    }
    Fixture { inputs, weights }
}

fn scratch_output_offset(kernel: &GpuKernel) -> usize {
    kernel.layout.feature_count + SCRATCH_PREDICTION - FEATURES_STRIDE
}

fn upload_fixture(kernel: &GpuKernel, fixture: &Fixture) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    assert_eq!(agents, CASE_COUNT);
    assert!(kernel.layout.feature_count >= ENCODED_DIMENSION);
    let mut brain = vec![0.0_f32; agents.checked_mul(kernel.layout.brain_stride).unwrap()];
    let mut scratch = vec![
        0.0_f32;
        agents
            .checked_mul(kernel.layout.brain_scratch_stride)
            .unwrap()
    ];
    let weights_offset = kernel.layout.feature_count * ENCODED_DIMENSION + ENCODED_DIMENSION;
    for agent in 0..agents {
        let scratch_base = agent * kernel.layout.brain_scratch_stride;
        scratch[scratch_base..scratch_base + ENCODED_DIMENSION]
            .copy_from_slice(&fixture.inputs[agent]);
        let output = scratch_base + scratch_output_offset(kernel);
        scratch[output..output + PREDICTOR_DIMENSION].fill(f32::NAN);
        for row in 0..PREDICTOR_DIMENSION {
            let offset =
                agent * kernel.layout.brain_stride + weights_offset + row * ENCODED_DIMENSION;
            brain[offset..offset + ENCODED_DIMENSION]
                .copy_from_slice(&fixture.weights[agent * PREDICTOR_DIMENSION + row]);
        }
    }
    kernel
        .queue
        .write_buffer(&kernel.brain_state_buffer, 0, bytemuck::cast_slice(&brain));
    kernel.queue.write_buffer(
        &kernel.brain_scratch_buffer,
        0,
        bytemuck::cast_slice(&scratch),
    );
}

fn f32_words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn execute(
    kernel: &GpuKernel,
    pipeline: &wgpu::ComputePipeline,
    fixture: &Fixture,
) -> TestResult<Vec<f32>> {
    upload_fixture(kernel, fixture);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.set_push_constants(0, bytemuck::cast_slice(&[0_u32, 0_u32]));
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    let scratch = f32_words(&read_buffer(
        kernel,
        &kernel.brain_scratch_buffer,
        kernel.brain_scratch_buffer.size(),
    )?);
    let brain = f32_words(&read_buffer(
        kernel,
        &kernel.brain_state_buffer,
        kernel.brain_state_buffer.size(),
    )?);
    let weights_offset = kernel.layout.feature_count * ENCODED_DIMENSION + ENCODED_DIMENSION;
    let mut output = Vec::with_capacity(fixture.weights.len());
    for agent in 0..CASE_COUNT {
        let start = agent * kernel.layout.brain_scratch_stride + scratch_output_offset(kernel);
        output.extend_from_slice(&scratch[start..start + PREDICTOR_DIMENSION]);
        for row in 0..PREDICTOR_DIMENSION {
            let offset =
                agent * kernel.layout.brain_stride + weights_offset + row * ENCODED_DIMENSION;
            assert_eq!(
                brain[offset..offset + ENCODED_DIMENSION],
                fixture.weights[agent * PREDICTOR_DIMENSION + row],
                "zero-gradient training must preserve supplied weights"
            );
        }
    }
    Ok(output)
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn predictor_raw_lane_reductions_satisfy_fp32_dot_bounds() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let kernel = make_kernel();
    let fixture = fixture();
    let mut original = Vec::new();
    for prefetch in PREFETCH_ARMS {
        for lanes in LANE_WIDTHS {
            let output = execute(&kernel, &make_pipeline(&kernel, lanes, prefetch), &fixture)?;
            if original.is_empty() {
                original.clone_from(&output);
            }
            for (agent, case) in CASES.iter().enumerate() {
                let mut summary = DotSummary::default();
                for row in 0..PREDICTOR_DIMENSION {
                    let index = agent * PREDICTOR_DIMENSION + row;
                    let label = format!(
                        "predictor_raw/lanes={lanes}/prefetch={prefetch}/case={case:?}/row={row}"
                    );
                    let metrics = check_fp32_dot(
                        &fixture.inputs[agent],
                        &fixture.weights[index],
                        output[index],
                        &label,
                    );
                    summary.add(metrics, output[index], original[index]);
                }
                let rms =
                    (summary.squared_error / f64::from(u32::try_from(PREDICTOR_DIMENSION)?)).sqrt();
                println!("PREDICTOR_RAW_DOT lanes={lanes} prefetch={prefetch} case={case:?} seed={DOT_SEED} rows={PREDICTOR_DIMENSION} terms={ENCODED_DIMENSION} max_abs={:.9e} rms={rms:.9e} max_error_to_forward_bound={:.9e} max_error_to_abs_products={:.9e} changed_from_scalar_four_lanes={} bound=gamma_2n_plus_f64_and_ftz all_rows_pass=true", summary.max_absolute_error, summary.max_error_to_bound, summary.max_error_to_product_sum, summary.different_from_scalar_four_lanes);
            }
        }
    }
    let arms = PREFETCH_ARMS.len() * LANE_WIDTHS.len();
    println!("PREDICTOR_RAW_DOT_COMPLETE arms={arms} rows_per_arm={} total_dot_checks={} all_rows_pass=true", fixture.weights.len(), arms * fixture.weights.len());
    Ok(())
}
