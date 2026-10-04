//! GPU-only primitive for deferred rank-one encoder updates, not a simulation
//! integration. Base weights plus eight or sixteen recorded feature/scale
//! vectors replace repeated dense weight writes. Queries use a base matvec
//! plus FP32 low-rank corrections; this changes rounding, including increments
//! lost by sequential per-weight updates, and requires numerical bounds.
//!
//! Materialization replays every original gated update and clamp in order in a
//! register, with one matrix load/store per window. A conservative no-clamp
//! certificate guards deferred queries. Failure flushes the certified prefix
//! before applying the uncertified update densely. Final flush is mandatory.
//! The measured primitive saves matrix traffic, not update multiplication/add
//! count, and includes certificate/reduction overhead. It is not a whole-cycle
//! benchmark and does not establish evolving-simulation behavioral equivalence.
//!
//! An integrated representation would export a materialized GPU copy without
//! changing the base or pending factors: flushing live state on readback would
//! make future query rounding depend on observation frequency. Intentional
//! mode/layout transitions may flush before replacing storage. Weight imports
//! must invalidate pending factors. Death/respawn preserves encoder learning,
//! so it must retain the factors or materialize at a deterministic transition;
//! dropping them would lose updates. Inactive agents must not append factors.
//! No such lifecycle is installed by this isolated test. Factors here are
//! finite supplied FP32 values; a real producer must capture actual adapted
//! features, rounded scales, and gates.
//! Finite factors alone do not guarantee finite feature/query overlaps: an
//! integrated producer must also establish a dot-range bound before querying.

use std::{error::Error, fmt::Write, time::Instant};

use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::util::DeviceExt;

use super::cycle_profile::make_kernel;
use super::deferred_encoder_oracle::{validate_deferred_encoder_prefix, RankOneStep};
use super::vision_validation::read_buffer;
use super::GpuKernel;

/// Default eight-by-six sensory layout has this many adapted encoder inputs.
const FEATURE_COUNT: usize = 267;
/// Match production encoded state and its four-lane, 64-output dot schedule.
const OUTPUT_DIMENSION: usize = 128;
const INNER_LANES: usize = 4;
const OUTPUT_TILE: usize = 64;
const WORKGROUP_SIZE: usize = 256;
/// Fixed flush windows under comparison; both fit the same shared storage.
const WINDOWS: [usize; 2] = [8, 16];
const MAX_WINDOW: usize = 16;
/// Two full longest windows plus a short final window exercise invalidation.
const STEP_COUNT: usize = 2 * MAX_WINDOW + 1;
/// The production encoder clamp is symmetric around zero.
const WEIGHT_LIMIT: f32 = 2.0;
/// Reproducible source data independent of the simulation's RNG or env flags.
const SEED: u64 = 20_261_004;
/// Match the default population for the isolated throughput experiment.
const BENCHMARK_CASES: usize = 10;
/// Repeated submissions amortize completion overhead without a long GPU batch.
const BENCHMARK_REPETITIONS: u32 = 8;
/// Rotate all three arms through each position to limit ordering bias.
const TIMING_ROUNDS: usize = 5;
const ARM_COUNT: usize = 3;
const ARM_NAMES: [&str; ARM_COUNT] = ["sequential", "deferred_8", "deferred_16"];
/// Metadata is encoded as u32 bits except for the final positive FP32 bound.
const META_PENDING_START: usize = 0;
const META_PENDING_COUNT: usize = 1;
const META_ACCEPTED: usize = 2;
const META_FALLBACKS: usize = 3;
const META_FLUSHES: usize = 4;
const META_BOUND: usize = 5;
const METADATA_WORDS: usize = 6;
/// Finite input magnitudes keep the primitive far from FP32 overflow.
const SEEDED_WEIGHT: f32 = 0.4;
const SEEDED_SCALE: f32 = 0.001;
/// This update is below half an ULP at a weight of one and is discarded.
const LOST_INCREMENT_EXPONENT: i32 = -27;
/// Safe updates approach the clamp before an intentionally uncertified step.
const NEAR_CLAMP_WEIGHT: f32 = 1.9;
const NEAR_CLAMP_SCALE: f32 = 0.02;
const SATURATING_STEP: usize = 4;
const SATURATING_SCALE: f32 = 0.2;
const RECOVERY_SCALE: f32 = -0.25;
/// Alternating steps demonstrate a conservative false-positive certificate.
const CANCELLING_WEIGHT: f32 = 1.75;
const CANCELLING_SCALE: f32 = 0.05;
/// Explicit inactive gates must ignore even large finite supplied scales.
const INACTIVE_SCALE: f32 = 500.0;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum Case {
    Seeded,
    LostTinyUpdates,
    AlternatingUpdates,
    ClampThenRecovery,
    InactiveDimensions,
    ZeroUpdates,
    SubnormalUpdates,
}

const CASES: [Case; 7] = [
    Case::Seeded,
    Case::LostTinyUpdates,
    Case::AlternatingUpdates,
    Case::ClampThenRecovery,
    Case::InactiveDimensions,
    Case::ZeroUpdates,
    Case::SubnormalUpdates,
];

struct Fixture {
    weights: Vec<f32>,
    features: Vec<f32>,
    scales: Vec<f32>,
    active: Vec<bool>,
    queries: Vec<f32>,
}

impl Fixture {
    fn new(case: Case, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = vec![0.0; FEATURE_COUNT * OUTPUT_DIMENSION];
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = match case {
                Case::LostTinyUpdates => {
                    if (index / OUTPUT_DIMENSION) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Case::AlternatingUpdates => CANCELLING_WEIGHT,
                Case::ClampThenRecovery => NEAR_CLAMP_WEIGHT,
                _ => rng.random_range(-SEEDED_WEIGHT..SEEDED_WEIGHT),
            };
        }
        let features = (0..STEP_COUNT * FEATURE_COUNT)
            .map(|_| match case {
                Case::Seeded | Case::InactiveDimensions => rng.random_range(-1.0..1.0),
                _ => 1.0,
            })
            .collect();
        let mut scales = vec![0.0; STEP_COUNT * OUTPUT_DIMENSION];
        let mut active = vec![true; scales.len()];
        for step in 0..STEP_COUNT {
            for dimension in 0..OUTPUT_DIMENSION {
                let index = step * OUTPUT_DIMENSION + dimension;
                scales[index] = match case {
                    Case::Seeded => rng.random_range(-SEEDED_SCALE..SEEDED_SCALE),
                    Case::LostTinyUpdates => 2.0_f32.powi(LOST_INCREMENT_EXPONENT),
                    Case::AlternatingUpdates => {
                        if step % 2 == 0 {
                            CANCELLING_SCALE
                        } else {
                            -CANCELLING_SCALE
                        }
                    }
                    Case::ClampThenRecovery => match step {
                        SATURATING_STEP => SATURATING_SCALE,
                        step if step == SATURATING_STEP + 1 => RECOVERY_SCALE,
                        step if step < SATURATING_STEP => NEAR_CLAMP_SCALE,
                        _ => SEEDED_SCALE,
                    },
                    Case::InactiveDimensions => {
                        active[index] = dimension % 2 == 0;
                        if active[index] {
                            SEEDED_SCALE
                        } else {
                            INACTIVE_SCALE
                        }
                    }
                    Case::ZeroUpdates => 0.0,
                    Case::SubnormalUpdates => f32::from_bits(1),
                };
            }
        }
        let queries = (0..STEP_COUNT * FEATURE_COUNT)
            .map(|_| {
                if matches!(case, Case::LostTinyUpdates) {
                    1.0
                } else {
                    rng.random_range(-1.0..1.0)
                }
            })
            .collect();
        Self {
            weights,
            features,
            scales,
            active,
            queries,
        }
    }

    fn steps(&self, start: usize, count: usize) -> Vec<RankOneStep<'_>> {
        (start..start + count)
            .map(|step| RankOneStep {
                features: &self.features[step * FEATURE_COUNT..(step + 1) * FEATURE_COUNT],
                scales: &self.scales[step * OUTPUT_DIMENSION..(step + 1) * OUTPUT_DIMENSION],
                active: &self.active[step * OUTPUT_DIMENSION..(step + 1) * OUTPUT_DIMENSION],
            })
            .collect()
    }

    fn packed(&self) -> Vec<f32> {
        let mut result = self.weights.clone();
        result.extend_from_slice(&self.features);
        result.extend_from_slice(&self.scales);
        result.extend(
            self.active
                .iter()
                .map(|active| f32::from(u8::from(*active))),
        );
        result.extend_from_slice(&self.queries);
        assert!(result.iter().all(|value| value.is_finite()));
        result
    }
}

struct Shape {
    matrix_words: usize,
    input_stride: usize,
    snapshot_words: usize,
    output_stride: usize,
}

impl Shape {
    fn new(capture: bool) -> Self {
        let matrix_words = FEATURE_COUNT.checked_mul(OUTPUT_DIMENSION).unwrap();
        let factor_words = FEATURE_COUNT.checked_add(OUTPUT_DIMENSION).unwrap();
        let input_stride = matrix_words
            .checked_add(
                STEP_COUNT
                    .checked_mul(factor_words)
                    .unwrap()
                    .checked_mul(2)
                    .unwrap(),
            )
            .unwrap();
        let snapshot_words = if capture {
            matrix_words.checked_mul(2).unwrap()
        } else {
            0
        };
        let output_stride = snapshot_words
            .checked_add(OUTPUT_DIMENSION)
            .unwrap()
            .checked_add(METADATA_WORDS)
            .unwrap();
        Self {
            matrix_words,
            input_stride,
            snapshot_words,
            output_stride,
        }
    }

    fn source(&self, window: usize, capture: bool) -> String {
        assert!(WINDOWS.contains(&window));
        let features_offset = self.matrix_words;
        let scales_offset = features_offset + STEP_COUNT * FEATURE_COUNT;
        let active_offset = scales_offset + STEP_COUNT * OUTPUT_DIMENSION;
        let queries_offset = active_offset + STEP_COUNT * OUTPUT_DIMENSION;
        assert_eq!(
            queries_offset + STEP_COUNT * FEATURE_COUNT,
            self.input_stride
        );
        let integers = [
            ("FEATURE_COUNT", FEATURE_COUNT),
            ("OUTPUT_DIMENSION", OUTPUT_DIMENSION),
            ("INNER_LANES", INNER_LANES),
            ("OUTPUT_TILE", OUTPUT_TILE),
            ("WORKGROUP_SIZE", WORKGROUP_SIZE),
            ("MAX_WINDOW", MAX_WINDOW),
            ("WINDOW_SIZE", window),
            ("STEP_COUNT", STEP_COUNT),
            ("MATRIX_WORDS", self.matrix_words),
            ("INPUT_STRIDE", self.input_stride),
            ("SNAPSHOT_WORDS", self.snapshot_words),
            ("OUTPUT_STRIDE", self.output_stride),
            ("FEATURES_OFFSET", features_offset),
            ("SCALES_OFFSET", scales_offset),
            ("ACTIVE_OFFSET", active_offset),
            ("QUERIES_OFFSET", queries_offset),
            ("META_PENDING_START", META_PENDING_START),
            ("META_PENDING_COUNT", META_PENDING_COUNT),
            ("META_ACCEPTED", META_ACCEPTED),
            ("META_FALLBACKS", META_FALLBACKS),
            ("META_FLUSHES", META_FLUSHES),
            ("META_BOUND", META_BOUND),
        ];
        let mut source = String::new();
        for (name, value) in integers {
            writeln!(
                source,
                "const {name}: u32 = {}u;",
                u32::try_from(value).unwrap()
            )
            .unwrap();
        }
        writeln!(source,"const CAPTURE_SNAPSHOTS: bool = {capture};\nconst WEIGHT_LIMIT: f32 = {WEIGHT_LIMIT:?};\nconst MIN_NORMAL: f32 = {:.17e};\nconst MAX_FINITE: f32 = {:.17e};", f64::from(f32::MIN_POSITIVE), f64::from(f32::MAX)).unwrap();
        source.push_str(include_str!("deferred_encoder_probe.wgsl"));
        source
    }
}

struct Probe {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    weights: wgpu::Buffer,
    output: wgpu::Buffer,
    shape: Shape,
    cases: u32,
}

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(std::mem::size_of::<f32>()).unwrap()).unwrap()
}

impl Probe {
    fn new(
        kernel: &GpuKernel,
        fixtures: &[Fixture],
        window: usize,
        capture: bool,
        deferred: bool,
    ) -> Self {
        assert!(!fixtures.is_empty());
        let shape = Shape::new(capture);
        let input: Vec<f32> = fixtures.iter().flat_map(Fixture::packed).collect();
        assert_eq!(
            input.len(),
            shape.input_stride.checked_mul(fixtures.len()).unwrap()
        );
        let input = kernel
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("deferred_encoder_inputs"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buffer = |words, label| {
            let size = bytes(words);
            assert!(size <= u64::from(kernel.device.limits().max_storage_buffer_binding_size));
            kernel.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let weights = buffer(
            shape.matrix_words.checked_mul(fixtures.len()).unwrap(),
            "deferred_encoder_weights",
        );
        let output = buffer(
            shape
                .output_stride
                .checked_mul(STEP_COUNT)
                .unwrap()
                .checked_mul(fixtures.len())
                .unwrap(),
            "deferred_encoder_output",
        );
        let module = kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("deferred_encoder_primitive"),
                source: wgpu::ShaderSource::Wgsl(shape.source(window, capture).into()),
            });
        let entry = if deferred {
            "deferred_encoder_probe"
        } else {
            "sequential_encoder_probe"
        };
        let pipeline = kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = kernel.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            bind_group,
            weights,
            output,
            shape,
            cases: u32::try_from(fixtures.len()).unwrap(),
        }
    }

    fn run(&self, kernel: &GpuKernel, repetitions: u32) {
        let mut encoder = kernel.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..repetitions {
                pass.dispatch_workgroups(self.cases, 1, 1);
            }
        }
        kernel.queue.submit([encoder.finish()]);
        kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    }

    fn read(&self, kernel: &GpuKernel) -> TestResult<ProbeResult> {
        let words = |data: Vec<u8>| {
            let (words, remainder) = data.as_chunks::<4>();
            assert!(remainder.is_empty());
            words.iter().map(|word| f32::from_le_bytes(*word)).collect()
        };
        Ok(ProbeResult {
            weights: words(read_buffer(kernel, &self.weights, self.weights.size())?),
            output: words(read_buffer(kernel, &self.output, self.output.size())?),
        })
    }
}

struct ProbeResult {
    weights: Vec<f32>,
    output: Vec<f32>,
}

fn assert_bits_equal(expected: &[f32], actual: &[f32], label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label}");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "{label} word={index} expected={expected} actual={actual}"
        );
    }
}

fn validate_without_snapshots(
    kernel: &GpuKernel,
    fixture: &Fixture,
    captured: &ProbeResult,
    captured_shape: &Shape,
    window: usize,
    deferred: bool,
) -> TestResult {
    let probe = Probe::new(
        kernel,
        std::slice::from_ref(fixture),
        window,
        false,
        deferred,
    );
    probe.run(kernel, 1);
    let actual = probe.read(kernel)?;
    assert_bits_equal(
        &captured.weights,
        &actual.weights,
        "snapshot-free materialization",
    );
    for step in 0..STEP_COUNT {
        let expected_offset = step * captured_shape.output_stride + captured_shape.snapshot_words;
        let actual_offset = step * probe.shape.output_stride;
        // Exact projection parity makes every bounded raw-dot observation apply
        // to the uninstrumented timing source too. Code-generation changes in
        // arithmetic, certificates, or window transitions fail this gate.
        assert_bits_equal(
            &captured.output[expected_offset..expected_offset + probe.shape.output_stride],
            &actual.output[actual_offset..actual_offset + probe.shape.output_stride],
            "snapshot-free raw dots and certificate metadata",
        );
    }
    Ok(())
}

fn validate_prefixes(
    fixture: &Fixture,
    shape: &Shape,
    reference: &ProbeResult,
    candidate: &ProbeResult,
    case: Case,
    window: usize,
) {
    assert_bits_equal(
        &reference.weights,
        &candidate.weights,
        "final materialized weights",
    );
    let mut max_error = 0.0_f64;
    let mut squared_error = 0.0;
    let mut changed = 0;
    let mut fallbacks = 0;
    let mut flushes = 0;
    let mut max_pending = 0;
    for step in 0..STEP_COUNT {
        let offset = step * shape.output_stride;
        let metadata = offset + shape.snapshot_words + OUTPUT_DIMENSION;
        let start =
            usize::try_from(candidate.output[metadata + META_PENDING_START].to_bits()).unwrap();
        let count =
            usize::try_from(candidate.output[metadata + META_PENDING_COUNT].to_bits()).unwrap();
        assert!(count <= window && start + count == step + 1);
        let accepted = candidate.output[metadata + META_ACCEPTED].to_bits() != 0;
        assert_eq!(accepted, count != 0);
        if accepted {
            assert!(candidate.output[metadata + META_BOUND] <= WEIGHT_LIMIT);
        }
        max_pending = max_pending.max(count);
        fallbacks = candidate.output[metadata + META_FALLBACKS].to_bits();
        flushes = candidate.output[metadata + META_FLUSHES].to_bits();
        let base = &candidate.output[offset..offset + shape.matrix_words];
        let materialized =
            &candidate.output[offset + shape.matrix_words..offset + shape.snapshot_words];
        let sequential =
            &reference.output[offset + shape.matrix_words..offset + shape.snapshot_words];
        let query = &fixture.queries[step * FEATURE_COUNT..(step + 1) * FEATURE_COUNT];
        let raw = offset + shape.snapshot_words;
        let sequential_dots = &reference.output[raw..raw + OUTPUT_DIMENSION];
        let deferred_dots = &candidate.output[raw..raw + OUTPUT_DIMENSION];
        let label = format!("deferred_encoder/case={case:?}/window={window}/prefix={step}");
        validate_deferred_encoder_prefix(
            base,
            FEATURE_COUNT,
            OUTPUT_DIMENSION,
            &fixture.steps(start, count),
            query,
            sequential,
            materialized,
            sequential_dots,
            deferred_dots,
            &label,
        );
        for (expected, actual) in sequential_dots.iter().zip(deferred_dots) {
            let error = f64::from(*actual) - f64::from(*expected);
            max_error = max_error.max(error.abs());
            squared_error += error * error;
            changed += usize::from(actual.to_bits() != expected.to_bits());
        }
    }
    if matches!(case, Case::ClampThenRecovery | Case::AlternatingUpdates) {
        assert!(fallbacks > 0, "adversary must exercise dense fallback");
    } else {
        assert_eq!(fallbacks, 0, "safe fixture must remain deferred");
        assert_eq!(max_pending, window, "exercise full-window flush");
        assert!(flushes > 0, "exercise fixed-window invalidation");
    }
    if matches!(
        case,
        Case::LostTinyUpdates | Case::ZeroUpdates | Case::SubnormalUpdates
    ) {
        assert_bits_equal(
            &fixture.weights,
            &reference.weights,
            "discarded or zero update",
        );
    }
    if matches!(case, Case::LostTinyUpdates) {
        assert!(
            changed > 0,
            "lost weight increments must remain visible in deferred cancellation dots"
        );
    }
    let rows = u32::try_from(STEP_COUNT * OUTPUT_DIMENSION).unwrap();
    let rms = (squared_error / f64::from(rows)).sqrt();
    println!("DEFERRED_ENCODER_PRIMITIVE case={case:?} window={window} steps={STEP_COUNT} max_abs_dot={max_error:.9e} rms_dot={rms:.9e} changed_dots={changed} fallbacks={fallbacks} completed_flushes={flushes} max_pending={max_pending} final_materialization_exact=true all_prefix_bounds_pass=true full_simulation=false");
}

#[test]
#[ignore = "requires a GPU; run explicitly in release mode with --ignored --nocapture"]
fn deferred_encoder_primitive_materializes_exactly_and_bounds_raw_dots() -> TestResult {
    let _vulkan = super::vulkan_gate::enter();
    let kernel = make_kernel();
    for case in CASES {
        let fixture = Fixture::new(case, SEED);
        let fixtures = std::slice::from_ref(&fixture);
        let reference = Probe::new(&kernel, fixtures, MAX_WINDOW, true, false);
        reference.run(&kernel, 1);
        let expected = reference.read(&kernel)?;
        validate_without_snapshots(
            &kernel,
            &fixture,
            &expected,
            &reference.shape,
            MAX_WINDOW,
            false,
        )?;
        for window in WINDOWS {
            let candidate = Probe::new(&kernel, fixtures, window, true, true);
            candidate.run(&kernel, 1);
            let actual = candidate.read(&kernel)?;
            validate_prefixes(&fixture, &candidate.shape, &expected, &actual, case, window);
            validate_without_snapshots(&kernel, &fixture, &actual, &candidate.shape, window, true)?;
            candidate.run(&kernel, 1);
            let repeat = candidate.read(&kernel)?;
            assert_bits_equal(
                &actual.weights,
                &repeat.weights,
                "candidate weight repeatability",
            );
            assert_bits_equal(
                &actual.output,
                &repeat.output,
                "candidate prefix repeatability",
            );
        }
        println!("DEFERRED_ENCODER_TIMING_SOURCE case={case:?} snapshot_free_raw_dots_and_final_matrix_match_bounded_shader=true");
    }
    Ok(())
}

#[test]
#[ignore = "GPU primitive benchmark, not a full simulation benchmark"]
fn benchmark_deferred_encoder_primitive_matrix_traffic() -> TestResult {
    let _vulkan = super::vulkan_gate::enter();
    let kernel = make_kernel();
    let fixtures: Vec<_> = (0..BENCHMARK_CASES)
        .map(|index| Fixture::new(Case::Seeded, SEED + u64::try_from(index).unwrap()))
        .collect();
    let probes = [
        Probe::new(&kernel, &fixtures, MAX_WINDOW, false, false),
        Probe::new(&kernel, &fixtures, WINDOWS[0], false, true),
        Probe::new(&kernel, &fixtures, WINDOWS[1], false, true),
    ];
    let mut expected = Vec::new();
    let mut repeats = Vec::new();
    for (arm, probe) in probes.iter().enumerate() {
        probe.run(&kernel, 1);
        let observed = probe.read(&kernel)?;
        if arm == 0 {
            expected.clone_from(&observed.weights);
        } else {
            for case in 0..BENCHMARK_CASES {
                for step in 0..STEP_COUNT {
                    let metadata =
                        (case * STEP_COUNT + step) * probe.shape.output_stride + OUTPUT_DIMENSION;
                    assert_eq!(
                        observed.output[metadata + META_FALLBACKS].to_bits(),
                        0,
                        "closed-form traffic counts require no fallback"
                    );
                    assert_eq!(observed.output[metadata + META_ACCEPTED].to_bits(), 1);
                    assert!(observed.output[metadata + META_BOUND] <= WEIGHT_LIMIT);
                }
            }
        }
        assert_bits_equal(
            &expected,
            &observed.weights,
            "benchmark final materialization",
        );
        repeats.push(observed);
    }
    let mut seconds = [0.0; ARM_COUNT];
    for round in 0..TIMING_ROUNDS {
        for position in 0..ARM_COUNT {
            let arm = (round + position) % ARM_COUNT;
            let start = Instant::now();
            probes[arm].run(&kernel, BENCHMARK_REPETITIONS);
            let elapsed = start.elapsed().as_secs_f64();
            seconds[arm] += elapsed;
            let actual = probes[arm].read(&kernel)?;
            assert_bits_equal(
                &repeats[arm].weights,
                &actual.weights,
                "timed arm weight repeatability",
            );
            assert_bits_equal(
                &repeats[arm].output,
                &actual.output,
                "timed arm output repeatability",
            );
            println!("DEFERRED_ENCODER_PRIMITIVE_TIME arm={} round={round} seconds={elapsed:.9} agents={BENCHMARK_CASES} steps_per_dispatch={STEP_COUNT} dispatches={BENCHMARK_REPETITIONS} capture_snapshots=false full_simulation=false", ARM_NAMES[arm]);
        }
    }
    for (arm, window) in WINDOWS.into_iter().enumerate() {
        let matrix_bytes = bytes(FEATURE_COUNT * OUTPUT_DIMENSION * BENCHMARK_CASES);
        let sequential_update_bytes = matrix_bytes * 2 * u64::try_from(STEP_COUNT).unwrap();
        let deferred_update_bytes =
            matrix_bytes * 2 * u64::try_from(STEP_COUNT.div_ceil(window)).unwrap();
        let factor_bytes = bytes(window * (FEATURE_COUNT + OUTPUT_DIMENSION) * BENCHMARK_CASES);
        println!("DEFERRED_ENCODER_PRIMITIVE_SUMMARY window={window} paired_seconds_sequential={:.9} paired_seconds_candidate={:.9} speedup={:.6} logical_update_matrix_bytes_sequential={sequential_update_bytes} logical_update_matrix_bytes_candidate={deferred_update_bytes} pending_factor_float_bytes={factor_bytes} pending_gate_bytes={} update_arithmetic_reduced=false dot_reads_unchanged=true initial_certificate_matrix_read=true full_simulation=false", seconds[0], seconds[arm + 1], seconds[0] / seconds[arm + 1], bytes(window * OUTPUT_DIMENSION * BENCHMARK_CASES));
    }
    Ok(())
}
