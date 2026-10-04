//! Packed global-credit row batching. Both arms retain production main128,
//! context8, predictor16, mirrored packed weights and unchanged-store suppression.
//! Reusing each invocation's credit vector across rows changes only the global
//! credit source and workgroup count; every matrix vector still has one writer.

use std::{collections::HashMap, error::Error, fmt::Write, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::packed_encoder_validation::{assert_credit_cases, credit_fixture};
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const ROW_COUNTS: [u32; 4] = [1, 2, 4, 8];
const THREADS: u32 = 256;
const VECTOR_WORDS: usize = 4;
const REPLAYS: usize = 2;
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const MUTABLE_BUFFERS: usize = 13;
const BRAIN_BUFFER: usize = 8;
const INACTIVE_AGENT: u32 = 1;
const PUSH_CONSTANT_BYTES: u32 = 8;
const CREDIT_THRESHOLD: f32 = 1e-6;
const LARGE_FEATURE: f32 = 10_000.0;
const FEATURE_BLOCK: usize = 8;
const CLAMP_CASE: usize = 3;
const ENCODER_LIMIT: f32 = 2.0;
const WRAPPER: &str = include_str!("packed_credit_batch.wgsl");
const SIGNATURE: &str = "fn phase_encoder_credit(agent_id: u32, weight_vector: u32) {\n";
const ADDRESS: &str =
    "    let address = agent_id * FEATURE_COUNT * PACKED_ENCODER_OUTPUT_VECTORS + weight_vector;\n";
const RATE: &str = "    let learning_rate = brain_config[1].x;\n";
const PROBE: &str = r"
@compute @workgroup_size(ENCODER_CREDIT_THREADS)
fn packed_credit_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    phase_encoder_credit(group.y, group.x * ENCODER_CREDIT_THREADS + tid);
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "unique source target: {before}"
    );
    source.replacen(before, after, 1)
}

/// The gate, clamp/update and mirror-store expressions come from the canonical
/// production shader. A suppressed row returns only from this inner helper;
/// the outer row loop must continue to every later feature it owns.
fn batched_credit() -> String {
    let canonical = packed_encoder::credit_source(true);
    assert!(!canonical.contains("Barrier"));
    assert!(!canonical.contains("var<workgroup>"));
    assert_eq!(canonical.matches(SIGNATURE).count(), 1);
    assert_eq!(canonical.matches(ADDRESS).count(), 1);
    let gate_start = canonical.find(SIGNATURE).unwrap() + SIGNATURE.len();
    let body_start = canonical.find(ADDRESS).unwrap();
    let body_end = canonical.rfind('\n').unwrap();
    assert!(canonical[..body_end].ends_with('}'));
    let body_end = body_end - 1;
    let gate = &canonical[gate_start..body_start];
    assert!(gate.contains("if !any(credit_enabled) { return; }"));
    let mut body = replace_once(&canonical[body_start..body_end], RATE, "");
    let mut scales = String::new();
    for component in ["x", "y", "z", "w"] {
        let expression = format!("learning_rate * credits.{component} * ENCODER_CREDIT_SCALE");
        body = replace_once(
            &body,
            &format!("let scale = {expression};"),
            &format!("let scale = scales.{component};"),
        );
        writeln!(
            scales,
            "    if credit_enabled.{component} {{ scales.{component} = {expression}; }}"
        )
        .unwrap();
    }
    assert_eq!(
        body.matches("return;").count(),
        1,
        "only the unchanged-vector return belongs to the row helper"
    );
    assert!(body.contains(
        "if all(bitcast<vec4<u32>>(weight) == bitcast<vec4<u32>>(original_weight)) { return; }"
    ));
    assert_eq!(
        body.matches("brain_state[scalar_address").count(),
        VECTOR_WORDS
    );
    let wrapper = replace_once(WRAPPER, "    // CREDIT_BATCH_GATE\n", gate);
    let wrapper = replace_once(&wrapper, "    // CREDIT_BATCH_SCALES\n", &scales);
    let declarations = &canonical[..canonical.find(SIGNATURE).unwrap()];
    format!("{declarations}\nfn packed_credit_apply_row(agent_id: u32, weight_vector: u32, feature: u32, credit_enabled: vec4<bool>, scales: vec4<f32>) {{\n{body}\n}}\n{wrapper}")
}

fn group_count(features: usize, rows: u32) -> Option<u32> {
    if !ROW_COUNTS.contains(&rows) {
        return None;
    }
    let vectors = u32::try_from(features.checked_mul(ENCODED_DIMENSION / VECTOR_WORDS)?).ok()?;
    if vectors == 0 {
        return None;
    }
    let per_group = THREADS.checked_mul(rows)?;
    // Also protect the padded final workgroup's row-vector arithmetic.
    vectors.checked_add(per_group - 1)?;
    Some(vectors.div_ceil(per_group))
}

fn constants(kernel: &GpuKernel, rows: u32) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants.insert(
        "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
        f64::from(group_count(kernel.layout.feature_count, rows).unwrap()),
    );
    if rows != 1 {
        constants.insert("CREDIT_ROWS_PER_INVOCATION".into(), f64::from(rows));
    }
    constants
}

fn prepare(width: u32, height: u32, boundary: bool) -> GpuKernel {
    let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
    kernel.global_credit = Pipelines::new_packed_with_main128(
        &kernel,
        &optimized_brain(),
        &constants(&kernel, 1),
        true,
    );
    assert_eq!(
        kernel.global_credit.as_ref().unwrap().main_threads,
        main_width::MAIN_THREADS
    );
    assert!(kernel.global_credit_active());
    assert_eq!(
        cache(&kernel).groups_per_agent(),
        group_count(kernel.layout.feature_count, 1).unwrap()
    );
    kernel
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str, rows: u32) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packed_credit_batch"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed_credit_batch"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("packed_credit_batch"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel, rows),
                ..Default::default()
            },
            cache: None,
        })
}

struct Arm {
    pipeline: wgpu::ComputePipeline,
    workgroups: u32,
}

struct Arms {
    parked: Vec<Option<Arm>>,
    active: usize,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        let canonical = packed_encoder::credit_source(true);
        let batched = batched_credit();
        assert_ne!(canonical, batched);
        let original =
            global_credit::global_source_with_store_suppression(Some(cache(kernel)), true);
        let candidate = replace_once(&original, &canonical, &batched);
        let mut parked = vec![None];
        for rows in ROW_COUNTS.into_iter().skip(1) {
            let workgroups = group_count(kernel.layout.feature_count, rows)
                .unwrap()
                .checked_mul(kernel.agent_count)
                .unwrap()
                .checked_add(1)
                .unwrap();
            assert!(workgroups <= MAX_DISPATCH_WORKGROUPS);
            assert!(workgroups <= kernel.device.limits().max_compute_workgroups_per_dimension);
            parked.push(Some(Arm {
                pipeline: pipeline(kernel, candidate.clone(), "global_credit_tick", rows),
                workgroups,
            }));
        }
        Self { parked, active: 0 }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, arm: usize) {
        if self.active != arm {
            let next = self.parked[arm].take().unwrap();
            let pipelines = kernel.global_credit.as_mut().unwrap();
            let previous = Arm {
                pipeline: std::mem::replace(&mut pipelines.global, next.pipeline),
                workgroups: std::mem::replace(&mut pipelines.global_workgroups, next.workgroups),
            };
            assert!(self.parked[self.active].replace(previous).is_none());
            self.active = arm;
        }
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            main_width::MAIN_THREADS
        );
        assert!(kernel.global_credit_active());
    }
}

fn checked_state(kernel: &GpuKernel) -> TestResult<State> {
    let state = capture_state(kernel)?;
    assert_eq!(state.len(), MUTABLE_BUFFERS);
    assert_mirror(kernel, &state)?;
    Ok(state)
}

fn trajectory(kernel: &mut GpuKernel) -> TestResult<Vec<State>> {
    let mut cycle = 0;
    let mut states = Vec::new();
    for count in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        states.push(checked_state(kernel)?);
        cycle += count;
    }
    assert_eq!(cycle, TIMED_CYCLES);
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= 2.0);
    Ok(states)
}

#[test]
fn packed_credit_batch_composition_and_tail_ownership() {
    let candidate = batched_credit();
    assert_ne!(candidate, packed_encoder::credit_source(true));
    assert!(!candidate.contains("Barrier"));
    assert!(!candidate.contains("var<workgroup>"));
    assert_eq!(
        candidate.matches("learning_rate * credits.").count(),
        VECTOR_WORDS
    );
    assert_eq!(
        candidate.matches("brain_state[scalar_address").count(),
        VECTOR_WORDS
    );
    for features in [1, 7, 8, 9, 16, 17, 267, 342] {
        let vectors = features * (ENCODED_DIMENSION / VECTOR_WORDS);
        for rows in ROW_COUNTS {
            let groups = group_count(features, rows).unwrap();
            let mut writes = vec![0u8; vectors];
            for group in 0..groups {
                for tid in 0..THREADS {
                    for row in 0..rows {
                        let index =
                            usize::try_from(group * THREADS * rows + tid + row * THREADS).unwrap();
                        if index < vectors {
                            writes[index] += 1;
                        }
                    }
                }
            }
            assert!(writes.iter().all(|&writers| writers == 1));
        }
    }
    assert!(group_count(usize::MAX, 8).is_none());
    assert!(group_count(267, 3).is_none());
}

#[test]
#[ignore = "requires a GPU; all13 state, private mirror, death and refresh parity"]
fn packed_credit_batches_preserve_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for (arm, rows) in ROW_COUNTS.into_iter().enumerate().skip(1) {
            for replay in 0..REPLAYS {
                restore(&mut kernel, &saved);
                arms.activate(&mut kernel, arm);
                for (reference, actual) in expected.iter().zip(trajectory(&mut kernel)?) {
                    assert_state_equal(&kernel, reference, &actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        &actual,
                        INACTIVE_AGENT,
                        "packed credit row batching",
                    );
                }
                println!("PACKED_CREDIT_BATCH_PARITY width={width} height={height} rows={rows} replay={replay} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} main_threads=128 private_mirror_exact=true death_refresh=true");
            }
        }
    }
    Ok(())
}

fn credit_probe(kernel: &GpuKernel, rows: u32) -> wgpu::ComputePipeline {
    let credit = if rows == 1 {
        packed_encoder::credit_source(true)
    } else {
        batched_credit()
    };
    pipeline(
        kernel,
        [cache(kernel).common_source(), credit, PROBE.to_owned()].join("\n"),
        "packed_credit_probe",
        rows,
    )
}

fn run_credit(kernel: &GpuKernel, pipeline: &wgpu::ComputePipeline, rows: u32) {
    let packed = cache(kernel);
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    packed.record_import(kernel, &mut encoder);
    encoder.copy_buffer_to_buffer(
        &kernel.brain_scratch_buffer,
        0,
        packed.buffer(),
        0,
        kernel.brain_scratch_buffer.size(),
    );
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(
            0,
            &kernel.global_credit.as_ref().unwrap().bind_groups[kernel.active_config_index],
            &[],
        );
        pass.dispatch_workgroups(
            group_count(kernel.layout.feature_count, rows).unwrap(),
            kernel.agent_count,
            1,
        );
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

/// First owned row is unchanged while the next owned row updates, exposing an
/// incorrectly propagated row-helper return. The last vector is fully gated.
fn unchanged_then_updated_fixture(kernel: &GpuKernel) {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let mut features = vec![0.0f32; agents * kernel.layout.brain_scratch_stride];
    let disabled = [f32::from_bits(CREDIT_THRESHOLD.to_bits() - 1); VECTOR_WORDS];
    for agent in 0..agents {
        for feature in 0..kernel.layout.feature_count {
            features[agent * kernel.layout.brain_scratch_stride + feature] =
                if (feature / FEATURE_BLOCK).is_multiple_of(2) {
                    0.0
                } else {
                    LARGE_FEATURE
                };
        }
        let offset = agent * DECISION_STRIDE + DECISION_CREDIT + ENCODED_DIMENSION - VECTOR_WORDS;
        kernel.queue.write_buffer(
            &kernel.decision_buffer,
            u64::try_from(offset * size_of::<f32>()).unwrap(),
            bytemuck::cast_slice(&disabled),
        );
    }
    kernel.queue.write_buffer(
        &kernel.brain_scratch_buffer,
        0,
        bytemuck::cast_slice(&features),
    );
}

fn assert_later_row_updated(kernel: &GpuKernel, initial: &State, actual: &State) {
    let before: &[u32] = bytemuck::cast_slice(&initial[BRAIN_BUFFER]);
    let after: &[u32] = bytemuck::cast_slice(&actual[BRAIN_BUFFER]);
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        let base = agent * kernel.layout.brain_stride;
        if agent == usize::try_from(INACTIVE_AGENT).unwrap() {
            continue;
        }
        assert_eq!(
            before[base + CLAMP_CASE],
            after[base + CLAMP_CASE],
            "zero first row"
        );
        let later = base + FEATURE_BLOCK * ENCODED_DIMENSION + CLAMP_CASE;
        assert_ne!(
            before[later], after[later],
            "the next owned row must execute after a no-store row"
        );
        assert_eq!(after[later], ENCODER_LIMIT.to_bits());
        for feature in 0..kernel.layout.feature_count {
            let first = base + feature * ENCODED_DIMENSION + ENCODED_DIMENSION - VECTOR_WORDS;
            assert_eq!(
                &before[first..first + VECTOR_WORDS],
                &after[first..first + VECTOR_WORDS],
                "all-credit-disabled vector"
            );
        }
    }
}

#[test]
#[ignore = "requires a GPU; canonical credit gates, clamps, no-store rows and tails"]
fn packed_credit_batches_preserve_update_edges() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare(width, height, false);
        let probes = ROW_COUNTS.map(|rows| credit_probe(&kernel, rows));
        for no_store_first in [false, true] {
            credit_fixture(&kernel, cache(&kernel).buffer());
            cache(&kernel).invalidate();
            if no_store_first {
                unchanged_then_updated_fixture(&kernel);
            }
            let initial = capture_state(&kernel)?;
            let saved = checkpoint(&kernel);
            run_credit(&kernel, &probes[0], 1);
            let expected = checked_state(&kernel)?;
            if no_store_first {
                assert_later_row_updated(&kernel, &initial, &expected);
            } else {
                assert_credit_cases(&kernel, &initial, &expected);
            }
            for (index, rows) in ROW_COUNTS.into_iter().enumerate().skip(1) {
                for _ in 0..REPLAYS {
                    restore(&mut kernel, &saved);
                    run_credit(&kernel, &probes[index], rows);
                    let actual = checked_state(&kernel)?;
                    assert_state_equal(&kernel, &expected, &actual);
                    assert_inactive_agent_unchanged(
                        &kernel,
                        &initial,
                        &actual,
                        INACTIVE_AGENT,
                        "packed credit raw edges",
                    );
                    if no_store_first {
                        assert_later_row_updated(&kernel, &initial, &actual);
                    } else {
                        assert_credit_cases(&kernel, &initial, &actual);
                    }
                }
                println!("PACKED_CREDIT_BATCH_EDGES width={width} height={height} rows={rows} no_store_first={no_store_first} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true threshold_clamp=true inactive_exact=true partial_rows=true replays={REPLAYS}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU paired full-cycle benchmark; run explicitly in release mode"]
fn benchmark_packed_credit_batches() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare(FIELDS[0].0, FIELDS[0].1, false);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP, TIMED_CYCLES);
    let expected = checked_state(&kernel)?;
    for (arm, rows) in ROW_COUNTS.into_iter().enumerate().skip(1) {
        let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for pair in 0..TIMING_PAIRS {
            for choice in [pair % 2, 1 - pair % 2] {
                restore(&mut kernel, &saved);
                arms.activate(&mut kernel, if choice == 0 { 0 } else { arm });
                let start = Instant::now();
                advance(&mut kernel, WARMUP, TIMED_CYCLES);
                timings[choice].push(start.elapsed().as_secs_f64());
                assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
            }
        }
        for samples in &mut timings {
            samples.sort_by(f64::total_cmp);
        }
        let reference = timings[0][TIMING_PAIRS / 2];
        let candidate = timings[1][TIMING_PAIRS / 2];
        let groups = group_count(kernel.layout.feature_count, rows).unwrap();
        println!("PACKED_CREDIT_BATCH_TIMING rows={rows} warmup={WARMUP} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} groups_per_agent={groups} global_workgroups={} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true main_threads=128 full_simulation=true cold_import_timed=true readbacks_timed=false new_shared_storage=false", reference/candidate, groups*kernel.agent_count+1);
    }
    Ok(())
}
