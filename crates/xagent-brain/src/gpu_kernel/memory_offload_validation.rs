//! Test-only pattern maintenance beside packed credit and world updates.
//! Main retains the norm, current-slot store, encoded-state publication and
//! critic replay. One additional workgroup per agent reinforces every other
//! slot, then performs canonical decay, active count and eviction selection.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::global_credit::Pipelines;
use super::packed_store_validation::{
    advance, assert_mirror, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Raw layouts exercise ordinary and odd feature counts.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// Check both sides of scheduled refresh and a second forced death.
const CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
/// Independent repetitions expose unpublished storage and shared scratch.
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const FORCED_DEATHS: f32 = 2.0;
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Four full arrays and one scalar, each conservatively rounded to 16 bytes.
const GLOBAL_SHARED_BYTES: u64 = 2_576;
const SHARED_ALIGNMENT: usize = 16;
/// Distinct slots test overwrite, deactivation and first-index eviction ties.
const STORED_SLOT: usize = 3;
const REINFORCED_SLOT: usize = 4;
const DECAY_SLOTS: [usize; 2] = [5, 6];
const EMPTY_SLOT: usize = 7;
const FIXTURE_TICK: f32 = 40.0;
const OLD_REINFORCEMENT: f32 = 19.999;
const ORDINARY_REINFORCEMENT: f32 = 0.5;
const EXPIRING_REINFORCEMENT: f32 = 1e-12;
const PREDICTION_ERROR: f32 = 0.25;
const PATTERN_BUFFER: usize = 10;

const REINFORCEMENT: &str = "    // ── 7c. Memory reinforcement + episodic credit:";
const STORE: &str = "    // ── 7d. Memory store:";
const DECAY: &str = "    // ── 7e. Memory decay:";
const PUBLISH: &str = "    // ── 7g. Publish this tick's encoded state";
const REINFORCEMENT_PLACEHOLDER: &str = "        // MEMORY_OFFLOAD_REINFORCEMENT";
const DECAY_PLACEHOLDER: &str = "        // MEMORY_OFFLOAD_DECAY_AND_MINIMUM";
const MAINTENANCE_ENTRY: &str = r"
@compute @workgroup_size(256)
fn memory_maintenance_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    maintain_stored_memory(group.x, tid);
}
";
const LEARN_ENTRY: &str = r"
@compute @workgroup_size(256)
fn memory_learn_probe(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let agent_id = group.x;
    let brain_base = agent_id * BRAIN_STRIDE;
    if (tid < ENCODED_DIMENSION) {
        s_memory_key[tid] = brain_state[brain_base + O_PREV_ENCODED + tid];
        s_encoded[tid] = s_memory_key[tid];
    }
    if (tid == 0u) {
        s_alive = select(0u, 1u, physics_state[agent_id * PHYS_STRIDE + P_ALIVE] >= 0.5);
        s_pred_td[S_PRED_ERROR] = physics_state[agent_id * PHYS_STRIDE + P_PREDICTION_ERROR];
        s_pred_td[S_REWARD] = 0.0;
    }
    workgroupBarrier();
    if (s_alive != 0u) { coop_learn_and_store(agent_id, tid, false); }
}
";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn between<'a>(source: &'a str, first: &str, last: &str) -> &'a str {
    assert_eq!(source.matches(first).count(), 1);
    assert_eq!(source.matches(last).count(), 1);
    let start = source.find(first).unwrap();
    let end = source.find(last).unwrap();
    assert!(start < end);
    &source[start..end]
}

fn deferred_passes(source: &str) -> String {
    let reinforcement = between(source, REINFORCEMENT, STORE);
    let decay = between(source, DECAY, PUBLISH);
    let candidate = replace_once(source, reinforcement, "");
    let candidate = replace_once(&candidate, decay, "");
    assert_ne!(source, candidate);
    assert!(candidate.contains(STORE));
    assert!(candidate.contains("s_enc_norm = sqrt(s_dense_partials[0]);"));
    let remaining = &candidate[candidate.find(PUBLISH).unwrap()..];
    let end = remaining.find("\n}").unwrap();
    assert!(!remaining[..end].contains("pattern_buffer["));
    candidate
}

fn maintenance_source() -> String {
    assert_eq!(ENCODED_DIMENSION, MEMORY_CAP);
    assert_eq!(
        usize::try_from(BRAIN_WORKGROUP_THREADS).unwrap(),
        MEMORY_CAP * 2
    );
    let original = optimized_brain();
    let reinforcement = between(&original, REINFORCEMENT, STORE);
    let reinforcement = replace_once(
        reinforcement,
        "    if (tid < MEMORY_CAP) {",
        "    if (tid < MEMORY_CAP && pattern != stored_idx) {",
    );
    let decay = between(&original, DECAY, PUBLISH);
    let decay = replace_once(
        decay,
        "        s_argmin_val[tid] = s_similarities[tid];\n",
        "",
    );
    let template = include_str!("memory_offload.wgsl");
    let source = replace_once(template, REINFORCEMENT_PLACEHOLDER, &reinforcement);
    let source = replace_once(&source, DECAY_PLACEHOLDER, &decay);
    source
        .replace("s_pred_td[S_PRED_ERROR]", "prediction_error")
        .replace("s_enc_norm", "encoded_norm")
        .replace("s_memory_key[", "memory_offload_key[")
        .replace("s_reinf_dot[", "memory_offload_dot[")
        .replace("s_dense_partials[", "memory_offload_dot[")
        .replace("s_similarities[", "memory_offload_score[")
        .replace("s_argmin_val[", "memory_offload_score[")
        .replace("s_argmin_idx[", "memory_offload_index[")
        .replace("wg_reduce_dense(tid)", "memory_offload_reduce(tid)")
}

fn global_source(kernel: &GpuKernel) -> String {
    let source = global_credit::global_source_with_store_suppression(Some(cache(kernel)), true);
    assert!(!source
        .lines()
        .any(|line| line.trim_start().starts_with("var<workgroup>")));
    let source = replace_once(&source,
        "    } else {\n        let credit_group = wgid.x - GLOBAL_WORLD_WORKGROUPS;",
        "    } else if wgid.x < GLOBAL_WORLD_WORKGROUPS + wc_u32(WC_AGENT_COUNT) * GLOBAL_CREDIT_GROUPS_PER_AGENT {\n        let credit_group = wgid.x - GLOBAL_WORLD_WORKGROUPS;");
    let source = replace_once(&source,
        "        phase_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x);\n    }\n}",
        "        phase_encoder_credit(agent_id, tile * ENCODER_CREDIT_THREADS + lid.x);\n    } else {\n        let agent_id = wgid.x - GLOBAL_WORLD_WORKGROUPS - wc_u32(WC_AGENT_COUNT) * GLOBAL_CREDIT_GROUPS_PER_AGENT;\n        maintain_stored_memory(agent_id, lid.x);\n    }\n}");
    format!("{source}\n{}", maintenance_source())
}

fn constants(kernel: &GpuKernel) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants.insert(
        "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
        f64::from(cache(kernel).groups_per_agent()),
    );
    constants
}

fn pipeline(kernel: &GpuKernel, source: String, entry: &str) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants(kernel),
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: Option<Pipelines>,
    candidate: bool,
}

impl Arms {
    fn new(kernel: &GpuKernel) -> Self {
        assert!(
            GLOBAL_SHARED_BYTES
                <= u64::from(kernel.device.limits().max_compute_workgroup_storage_size)
        );
        let mut candidate = Pipelines::new_packed_with_store_suppression(
            kernel,
            &deferred_passes(&optimized_brain()),
            &constants(kernel),
            true,
        )
        .unwrap();
        candidate.global = pipeline(kernel, global_source(kernel), "global_credit_tick");
        candidate.global_workgroups = candidate
            .global_workgroups
            .checked_add(kernel.agent_count)
            .unwrap();
        assert!(candidate.global_workgroups <= MAX_DISPATCH_WORKGROUPS);
        assert!(
            candidate.global_workgroups
                <= kernel.device.limits().max_compute_workgroups_per_dimension
        );
        Self {
            parked: Some(candidate),
            candidate: false,
        }
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if self.candidate != candidate {
            self.parked
                .as_ref()
                .unwrap()
                .packed_encoder
                .as_ref()
                .unwrap()
                .invalidate();
            std::mem::swap(&mut kernel.global_credit, &mut self.parked);
            self.candidate = candidate;
        }
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
    let mut states = Vec::new();
    let mut cycle = 0;
    for count in CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        cycle += count;
        states.push(checked_state(kernel)?);
    }
    assert_eq!(cycle, TIMED_CYCLES);
    assert!(kernel.read_full_state_blocking()[P_DEATH_COUNT] >= FORCED_DEATHS);
    Ok(states)
}

#[test]
fn memory_offload_composes_canonical_updates_and_preserves_norm() {
    let original = optimized_brain();
    let main = deferred_passes(&original);
    assert!(!main.contains(REINFORCEMENT));
    assert!(!main.contains(DECAY));
    let maintenance = maintenance_source();
    assert!(maintenance.contains("if (tid < MEMORY_CAP && pattern != stored_idx)"));
    assert!(maintenance.contains("for (var d = lane; d < ENCODED_DIMENSION; d += 2u)"));
    assert!(maintenance.contains("vo < vt || (vo == vt && io < it)"));
    assert_eq!(maintenance.matches("var<workgroup>").count(), 5);
    let mut shared_bytes = 0usize;
    for declaration in maintenance
        .lines()
        .filter(|line| line.starts_with("var<workgroup>"))
    {
        let words = if let Some((_, array)) = declaration.split_once("array<") {
            let (kind, count) = array.trim_end_matches(">;").split_once(", ").unwrap();
            assert!(matches!(kind, "f32" | "u32"));
            match count {
                "ENCODED_DIMENSION" => ENCODED_DIMENSION,
                "MEMORY_CAP" => MEMORY_CAP,
                literal => literal.parse().unwrap(),
            }
        } else {
            assert!(declaration.ends_with(": u32;"));
            1
        };
        shared_bytes += words
            .checked_mul(size_of::<f32>())
            .unwrap()
            .div_ceil(SHARED_ALIGNMENT)
            * SHARED_ALIGNMENT;
    }
    assert_eq!(u64::try_from(shared_bytes).unwrap(), GLOBAL_SHARED_BYTES);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let source = [
            include_str!("../shaders/kernel/common.wgsl"),
            &maintenance,
            MAINTENANCE_ENTRY,
        ]
        .join("\n");
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }
}

#[test]
#[ignore = "requires GPU; exact all-buffer comparison through death and refresh"]
fn memory_offload_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, true, true);
        let mut arms = Arms::new(&kernel);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel)?;
        for _ in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "memory offload",
                );
            }
        }
        println!("MEMORY_OFFLOAD_PARITY vision={width}x{height} cycles={TIMED_CYCLES} exact_buffers={MUTABLE_BUFFERS} candidate_replays={REPLAYS} private_mirror_exact=true death_refresh=true inactive_exact=true");
    }
    Ok(())
}

fn tail(kernel: &GpuKernel, offset: usize) -> usize {
    fixed_tail_base(kernel.layout.brain_stride) + offset - O_PREDICTOR_CONTEXT_WEIGHT
}

fn edge_fixture(kernel: &GpuKernel) {
    for agent in 0..kernel.agent_count {
        let mut state = kernel.read_agent_state(agent);
        state.brain_state[tail(kernel, O_TICK_COUNT)] = FIXTURE_TICK;
        state.brain_state[tail(kernel, O_SALIENCE_LABEL)] = 1.0;
        for dimension in 0..ENCODED_DIMENSION {
            let key = if dimension == 0 { 1.0 } else { 0.0 };
            state.brain_state[tail(kernel, O_PREV_ENCODED) + dimension] = key;
            for pattern in 0..MEMORY_CAP {
                state.patterns[dimension * MEMORY_CAP + pattern] = if DECAY_SLOTS.contains(&pattern)
                {
                    -key
                } else {
                    key
                };
            }
        }
        for pattern in 0..MEMORY_CAP {
            state.patterns[O_PAT_NORMS + pattern] = 1.0;
            state.patterns[O_PAT_ACTIVE + pattern] = if pattern == EMPTY_SLOT { 0.0 } else { 1.0 };
            state.patterns[O_PAT_REINF + pattern] = if pattern == STORED_SLOT {
                OLD_REINFORCEMENT
            } else if DECAY_SLOTS.contains(&pattern) {
                EXPIRING_REINFORCEMENT
            } else {
                ORDINARY_REINFORCEMENT
            };
            state.patterns[O_PAT_MOTOR + pattern * 3 + 2] =
                if pattern == STORED_SLOT { 1.0 } else { 0.0 };
            state.patterns[O_PAT_META + pattern * 3] = if DECAY_SLOTS.contains(&pattern) {
                0.0
            } else {
                FIXTURE_TICK - 1.0
            };
            state.patterns[O_PAT_META + pattern * 3 + 1] = FIXTURE_TICK - 1.0;
            state.patterns[O_PAT_META + pattern * 3 + 2] = 1.0;
        }
        state.patterns[O_MIN_REINF_IDX] = f32::from(u16::try_from(STORED_SLOT).unwrap());
        kernel.write_agent_state(agent, &state);
        kernel.write_agent_physics_fields(
            agent,
            &[
                (P_PREDICTION_ERROR, PREDICTION_ERROR),
                (P_ALIVE, if agent == INACTIVE_AGENT { 0.0 } else { 1.0 }),
            ],
        );
    }
}

fn learn_probe(kernel: &GpuKernel, deferred: bool) -> wgpu::ComputePipeline {
    let original = optimized_brain();
    let passes = if deferred {
        deferred_passes(&original)
    } else {
        original
    };
    let source = apply_subgroup_markers(
        &[
            include_str!("../shaders/kernel/common.wgsl"),
            &passes,
            include_str!("../shaders/kernel/brain_inner.wgsl"),
            LEARN_ENTRY,
        ]
        .join("\n"),
        false,
    );
    pipeline(kernel, source, "memory_learn_probe")
}

fn run_probe(kernel: &GpuKernel, pipeline: &wgpu::ComputePipeline) {
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    cache(kernel).record_import(kernel, &mut encoder);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.dispatch_workgroups(kernel.agent_count, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
    kernel.poll_wait();
}

fn patterns(state: &State, agent: usize) -> &[f32] {
    let values: &[f32] = bytemuck::cast_slice(&state[PATTERN_BUFFER]);
    &values[agent * PATTERN_STRIDE..(agent + 1) * PATTERN_STRIDE]
}

fn assert_edge_result(kernel: &GpuKernel, state: &State) {
    for agent in 0..usize::try_from(kernel.agent_count).unwrap() {
        if agent == usize::try_from(INACTIVE_AGENT).unwrap() {
            continue;
        }
        let memory = patterns(state, agent);
        assert!(memory[O_PAT_REINF + STORED_SLOT] <= 1.0);
        assert_eq!(
            memory[O_PAT_MOTOR + STORED_SLOT * 3 + 2].to_bits(),
            0.0f32.to_bits()
        );
        assert!(memory[O_PAT_REINF + REINFORCED_SLOT] > ORDINARY_REINFORCEMENT);
        assert!(memory[O_PAT_MOTOR + REINFORCED_SLOT * 3 + 2] > 0.0);
        for slot in DECAY_SLOTS {
            assert_eq!(memory[O_PAT_ACTIVE + slot].to_bits(), 0.0f32.to_bits());
        }
        assert_eq!(
            memory[O_MIN_REINF_IDX],
            f32::from(u16::try_from(DECAY_SLOTS[0]).unwrap())
        );
        assert_eq!(
            memory[O_ACTIVE_COUNT],
            f32::from(u16::try_from(MEMORY_CAP - DECAY_SLOTS.len() - 1).unwrap())
        );
    }
}

#[test]
#[ignore = "requires GPU; overwritten slot, episodic credit, decay and minimum ties"]
fn memory_offload_preserves_overwrite_and_eviction_edges() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(FIELDS[0].0, FIELDS[0].1, false, true);
    edge_fixture(&kernel);
    let initial = capture_state(&kernel)?;
    let saved = checkpoint(&kernel);
    let reference = learn_probe(&kernel, false);
    let deferred = learn_probe(&kernel, true);
    let maintenance = pipeline(
        &kernel,
        [
            include_str!("../shaders/kernel/common.wgsl"),
            &maintenance_source(),
            MAINTENANCE_ENTRY,
        ]
        .join("\n"),
        "memory_maintenance_probe",
    );
    run_probe(&kernel, &reference);
    let expected = checked_state(&kernel)?;
    assert_edge_result(&kernel, &expected);
    for _ in 0..REPLAYS {
        restore(&mut kernel, &saved);
        run_probe(&kernel, &deferred);
        let stored = checked_state(&kernel)?;
        assert_eq!(
            patterns(&stored, 0)[O_PAT_REINF + STORED_SLOT].to_bits(),
            1.0f32.to_bits()
        );
        assert_eq!(
            patterns(&stored, 0)[O_PAT_MOTOR + STORED_SLOT * 3 + 2].to_bits(),
            0.0f32.to_bits()
        );
        run_probe(&kernel, &maintenance);
        let actual = checked_state(&kernel)?;
        assert_state_equal(&kernel, &expected, &actual);
        assert_inactive_agent_unchanged(
            &kernel,
            &initial,
            &actual,
            INACTIVE_AGENT,
            "memory offload raw edges",
        );
        assert_edge_result(&kernel, &actual);
    }
    println!("MEMORY_OFFLOAD_EDGES exact_buffers={MUTABLE_BUFFERS} fresh_slot_overwritten=true other_slot_reinforced=true episodic_credit=true deactivation=true first_index_minimum_tie=true private_mirror_exact=true candidate_replays={REPLAYS}");
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle paired benchmark; run explicitly in release mode"]
fn benchmark_memory_offload() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let mut kernel = prepare_kernel_with_store_suppression(FIELDS[0].0, FIELDS[0].1, false, true);
    let mut arms = Arms::new(&kernel);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let saved = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = checked_state(&kernel)?;
    let mut times: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            times[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
        }
    }
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    let reference = times[0][TIMING_PAIRS / 2];
    let candidate = times[1][TIMING_PAIRS / 2];
    println!("MEMORY_OFFLOAD_TIMING warmup={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true extra_groups={} extra_global_shared_bytes={GLOBAL_SHARED_BYTES} added_dispatch=false cold_import_timed=true full_simulation=true", reference/candidate, kernel.agent_count);
    Ok(())
}
