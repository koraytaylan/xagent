//! Test-only packed encoder authority between explicit export boundaries.
//! The candidate removes only scalar mirror stores from production credit.
//! It retains main128, context8, predictor16, unchanged-store suppression, the
//! private allocation and the production dispatch recorder. Public state is
//! stale between exports; this module does not implement a public lifecycle.
//!
//! Timings include export submissions and their completion. Reference and
//! candidate use identical dispatch-call boundaries and completion polling.
//! Private matrices are checked against reference bytes before the untimed
//! final export, so exporting cannot hide an incorrect private trajectory.

use std::{collections::HashMap, error::Error, ops::Range, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::packed_store_validation::{
    cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw fields exercise both aligned and partial feature tiles.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Export every cycle, at the production submission limit, or at the endpoint.
const EXPORT_INTERVALS: [u32; 3] = [1, MAX_FUSED_BATCHES, 100];
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const REPLAYS: usize = 2;
const MUTABLE_BUFFERS: usize = 13;
const BRAIN_BUFFER: usize = 8;
const INACTIVE_AGENT: u32 = 1;
const FORCED_DEATHS: f32 = 2.0;
const VECTOR_BYTES: u64 = 16;
const VECTOR_WORDS: usize = 4;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;

const MIRROR_TAIL: &str = "    // The scalar matrix remains authoritative for readback and every fallback\n    // schedule. Each invocation owns these same four scalar locations.\n    let scalar_address = agent_id * BRAIN_STRIDE + O_ENC_WEIGHTS\n        + weight_vector * PACKED_ENCODER_WIDTH;\n    brain_state[scalar_address] = weight.x;\n    brain_state[scalar_address + 1u] = weight.y;\n    brain_state[scalar_address + 2u] = weight.z;\n    brain_state[scalar_address + 3u] = weight.w;\n";
const PRIVATE_STORE: &str = "    packed_encoder.weights[address] = weight;\n";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn without_scalar_mirror(credit: &str) -> String {
    assert_eq!(credit.matches(MIRROR_TAIL).count(), 1);
    assert_eq!(credit.matches(PRIVATE_STORE).count(), 1);
    assert_eq!(
        credit.matches("brain_state[scalar_address").count(),
        VECTOR_WORDS
    );
    let candidate = credit.replacen(MIRROR_TAIL, "", 1);
    assert_ne!(candidate, credit);
    assert!(!candidate.contains("scalar_address"));
    assert_eq!(candidate.matches(PRIVATE_STORE).count(), 1);
    candidate
}

/// Derive the matrix prefix from the actual production allocation, then check
/// it against the array length emitted by the production source accessor.
/// This deliberately does not duplicate the cache's padding calculation.
struct MatrixLayout {
    prefix_bytes: u64,
    matrix_bytes: u64,
    brain_stride_bytes: u64,
    agents: u32,
}

impl MatrixLayout {
    fn new(kernel: &GpuKernel) -> Self {
        let packed = cache(kernel);
        let matrix_bytes = bytes(
            kernel
                .layout
                .feature_count
                .checked_mul(ENCODED_DIMENSION)
                .unwrap(),
        );
        let population_bytes = matrix_bytes
            .checked_mul(u64::from(kernel.agent_count))
            .unwrap();
        let prefix_bytes = packed
            .buffer()
            .size()
            .checked_sub(population_bytes)
            .unwrap();
        let brain_stride_bytes = bytes(kernel.layout.brain_stride);
        assert!(prefix_bytes.is_multiple_of(VECTOR_BYTES));
        assert!(prefix_bytes >= kernel.brain_scratch_buffer.size());
        assert!(matrix_bytes <= brain_stride_bytes);
        let common = packed.common_source();
        let words = prefix_bytes / u64::try_from(WORD_BYTES).unwrap();
        assert_eq!(
            common
                .matches(&format!("scratch: array<f32, {words}>,"))
                .count(),
            1
        );
        assert!(common.contains("const O_ENC_WEIGHTS: u32 = 0u;"));
        assert_eq!(
            brain_stride_bytes
                .checked_mul(u64::from(kernel.agent_count))
                .unwrap(),
            kernel.brain_state_buffer.size()
        );
        Self {
            prefix_bytes,
            matrix_bytes,
            brain_stride_bytes,
            agents: kernel.agent_count,
        }
    }

    fn private_offset(&self, agent: u32) -> u64 {
        assert!(agent < self.agents);
        self.prefix_bytes
            .checked_add(u64::from(agent).checked_mul(self.matrix_bytes).unwrap())
            .unwrap()
    }

    fn scalar_offset(&self, agent: u32) -> u64 {
        assert!(agent < self.agents);
        u64::from(agent)
            .checked_mul(self.brain_stride_bytes)
            .unwrap()
    }

    fn range(&self, offset: u64) -> Range<usize> {
        usize::try_from(offset).unwrap()
            ..usize::try_from(offset.checked_add(self.matrix_bytes).unwrap()).unwrap()
    }
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants
}

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    kernel.global_credit =
        Pipelines::new_packed_with_main128(&kernel, &optimized_brain(), &constants(&kernel), true);
    assert!(kernel.global_credit_active());
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        main_width::MAIN_THREADS
    );
    assert!(!cache(&kernel).is_valid());
    kernel
}

struct Arms {
    parked: wgpu::ComputePipeline,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        let canonical = packed_encoder::credit_source(true);
        let unmirrored = without_scalar_mirror(&canonical);
        let original =
            global_credit::global_source_with_store_suppression(Some(cache(kernel)), true);
        assert_eq!(original.matches(&canonical).count(), 1);
        let source = original.replacen(&canonical, &unmirrored, 1);
        assert_ne!(source, original);
        let module = kernel
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("authoritative_packed_credit"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
        let layout = kernel
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("authoritative_packed_credit"),
                bind_group_layouts: &[&binding],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..PUSH_CONSTANT_BYTES,
                }],
            });
        let mut constants = constants(kernel);
        constants.insert(
            "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
            f64::from(cache(kernel).groups_per_agent()),
        );
        let parked = kernel
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("authoritative_packed_credit"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("global_credit_tick"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
        Self {
            parked,
            candidate: false,
        }
    }

    /// Call only after checkpoint restoration has discarded the prior arm's
    /// private authority. This prototype deliberately has no live transition.
    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        assert!(
            !cache(kernel).is_valid(),
            "restore before changing experiment arms"
        );
        if candidate != self.candidate {
            std::mem::swap(
                &mut kernel.global_credit.as_mut().unwrap().global,
                &mut self.parked,
            );
            self.candidate = candidate;
        }
    }
}

fn import(kernel: &GpuKernel) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    assert!(cache(kernel).record_import(kernel, &mut encoder));
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

/// Queue copies after the last production dispatch. The caller's poll covers
/// both dispatch and export; no staging/map operation is inside the timer.
fn submit_export(kernel: &GpuKernel, layout: &MatrixLayout) {
    assert!(cache(kernel).is_valid());
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    for agent in 0..layout.agents {
        encoder.copy_buffer_to_buffer(
            cache(kernel).buffer(),
            layout.private_offset(agent),
            &kernel.brain_state_buffer,
            layout.scalar_offset(agent),
            layout.matrix_bytes,
        );
    }
    kernel.queue.submit([encoder.finish()]);
}

fn dispatch(kernel: &mut GpuKernel, cycle: u32, count: u32) {
    assert!(kernel.global_credit_active());
    assert!(cache(kernel).is_valid());
    let stride = kernel.brain_tick_stride;
    kernel.dispatch_ticks(
        u64::from(cycle.checked_mul(stride).unwrap()),
        count.checked_mul(stride).unwrap(),
    );
    assert!(cache(kernel).is_valid());
}

fn run_chunks(
    kernel: &mut GpuKernel,
    layout: &MatrixLayout,
    start: u32,
    interval: u32,
    candidate: bool,
    export_final: bool,
) {
    assert!(EXPORT_INTERVALS.contains(&interval));
    let mut completed = 0;
    while completed < TIMED_CYCLES {
        let count = interval.min(TIMED_CYCLES - completed);
        dispatch(kernel, start.checked_add(completed).unwrap(), count);
        completed += count;
        if candidate && (export_final || completed < TIMED_CYCLES) {
            submit_export(kernel, layout);
        }
        kernel.poll_wait();
    }
}

fn assert_private(kernel: &GpuKernel, layout: &MatrixLayout, expected: &State) -> TestResult {
    let buffer = cache(kernel).buffer();
    let packed = read_buffer(kernel, buffer, buffer.size())?;
    for agent in 0..layout.agents {
        assert_eq!(
            &packed[layout.range(layout.private_offset(agent))],
            &expected[BRAIN_BUFFER][layout.range(layout.scalar_offset(agent))],
            "private encoder differs before export, agent={agent}",
        );
    }
    Ok(())
}

fn reference_trajectory(kernel: &mut GpuKernel, layout: &MatrixLayout) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        dispatch(kernel, cycle, count);
        kernel.poll_wait();
        let state = capture_state(kernel)?;
        assert_private(kernel, layout, &state)?;
        states.push(state);
        cycle += count;
    }
    assert_eq!(cycle, TIMED_CYCLES);
    Ok(states)
}

#[test]
fn authoritative_packed_removes_only_scalar_mirror_stores() {
    for suppress in [false, true] {
        let canonical = packed_encoder::credit_source(suppress);
        let candidate = without_scalar_mirror(&canonical);
        let restored =
            candidate.replacen(PRIVATE_STORE, &format!("{PRIVATE_STORE}{MIRROR_TAIL}"), 1);
        assert_eq!(restored, canonical);
        assert!(!candidate.contains("brain_state["));
        assert!(!candidate.contains("Barrier"));
        assert_eq!(
            candidate.matches("weight.").count(),
            canonical.matches("weight.").count() - VECTOR_WORDS
        );
    }
}

#[test]
#[ignore = "requires a GPU; private authority, exported all13 state, death and refresh"]
fn authoritative_packed_preserves_exported_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let layout = MatrixLayout::new(&kernel);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        import(&kernel);
        let expected = reference_trajectory(&mut kernel, &layout)?;
        let mut observed_stale_scalar = false;
        for replay in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            import(&kernel);
            let mut cycle = 0;
            for (count, reference) in PARITY_CHUNKS.into_iter().zip(&expected) {
                if cycle == REFRESH_CYCLES {
                    force_death(&kernel);
                }
                dispatch(&mut kernel, cycle, count);
                kernel.poll_wait();
                assert_private(&kernel, &layout, reference)?;
                let scalar = read_buffer(
                    &kernel,
                    &kernel.brain_state_buffer,
                    kernel.brain_state_buffer.size(),
                )?;
                for agent in 0..layout.agents {
                    let range = layout.range(layout.scalar_offset(agent));
                    observed_stale_scalar |=
                        scalar[range.clone()] != reference[BRAIN_BUFFER][range];
                }
                submit_export(&kernel, &layout);
                kernel.poll_wait();
                let actual = capture_state(&kernel)?;
                assert_eq!(actual.len(), MUTABLE_BUFFERS);
                assert_state_equal(&kernel, reference, &actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    &actual,
                    INACTIVE_AGENT,
                    "authoritative packed encoder",
                );
                cycle += count;
            }
            assert_eq!(cycle, TIMED_CYCLES);
            assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
            println!("AUTHORITATIVE_PACKED_PARITY width={width} height={height} replay={replay} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} private_before_export_exact=true death_refresh=true lifecycle_integrated=false");
        }
        assert!(
            observed_stale_scalar,
            "candidate must actually omit a changed scalar mirror"
        );
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark including explicit export copies and completion"]
fn benchmark_authoritative_packed_export_frequency() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (width, height) = FIELDS[0];
    let mut kernel = prepare(width, height, false);
    let layout = MatrixLayout::new(&kernel);
    let mut arms = Arms::new(&kernel);
    import(&kernel);
    dispatch(&mut kernel, 0, WARMUP_CYCLES);
    kernel.poll_wait();
    let warm = checkpoint(&kernel);
    for interval in EXPORT_INTERVALS {
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, false);
        import(&kernel);
        run_chunks(&mut kernel, &layout, WARMUP_CYCLES, interval, false, false);
        let expected = capture_state(&kernel)?;
        assert_private(&kernel, &layout, &expected)?;
        restore(&mut kernel, &warm);
        arms.activate(&mut kernel, true);
        import(&kernel);
        run_chunks(&mut kernel, &layout, WARMUP_CYCLES, interval, true, false);
        assert_private(&kernel, &layout, &expected)?;
        submit_export(&kernel, &layout);
        kernel.poll_wait();
        assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
        let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for pair in 0..TIMING_PAIRS {
            for position in 0..timings.len() {
                let arm = (pair + position) % timings.len();
                restore(&mut kernel, &warm);
                arms.activate(&mut kernel, arm != 0);
                import(&kernel);
                let start = Instant::now();
                run_chunks(
                    &mut kernel,
                    &layout,
                    WARMUP_CYCLES,
                    interval,
                    arm != 0,
                    true,
                );
                timings[arm].push(start.elapsed().as_secs_f64());
                assert_state_equal(&kernel, &expected, &capture_state(&kernel)?);
                assert_private(&kernel, &layout, &expected)?;
            }
        }
        for values in &mut timings {
            values.sort_by(f64::total_cmp);
        }
        let reference = timings[0][TIMING_PAIRS / 2];
        let candidate = timings[1][TIMING_PAIRS / 2];
        println!("AUTHORITATIVE_PACKED_TIMING cycles={TIMED_CYCLES} warmup={WARMUP_CYCLES} pairs={TIMING_PAIRS} export_interval={interval} exports={} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_before_export_gate=true exports_and_completion_timed=true import_timed=false matched_dispatch_chunks=true export_bytes={} lifecycle_integrated=false", TIMED_CYCLES.div_ceil(interval), reference / candidate, layout.matrix_bytes * u64::from(layout.agents));
    }
    Ok(())
}
