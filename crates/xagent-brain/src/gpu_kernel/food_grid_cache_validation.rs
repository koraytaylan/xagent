//! Exact test-only food-grid reuse in the current packed production schedule.
//! The first agent's private SCRATCH_PREDICTION holds 104 membership codes and
//! four cache words. Fused prediction never reads this tiled-only region, and
//! concurrent encoder credit reads only the preceding adapted-feature prefix.
//! Public grid bytes, including stale unused slots, remain fully observable.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore, Checkpoint};
use super::packed_store_validation::{
    advance as production_advance, assert_mirror, cache, optimized_brain,
    prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
const AGENTS: u32 = 10;
const FOOD_ITEMS: usize = 104;
const CACHE_HEADER_WORDS: usize = 4;
const CACHE_VALID: usize = 0;
const CACHE_OVERFLOW: usize = 1;
const CACHE_REUSED: usize = 2;
const CACHE_EXPIRY: usize = 3;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;
const EXACT_INTEGER_LIMIT: usize = 1 << 24;
const MUTABLE_BUFFERS: usize = 13;
const INACTIVE_AGENT: u32 = 1;
const PARITY_CHUNKS: [u32; 7] = [1, 18, 1, 1, 19, 1, 59];
const WARMUP_CYCLES: u32 = 1_000;
const TIMED_CYCLES: u32 = 100;
const TIMING_PAIRS: usize = 5;
const PHYSICS_BUFFER_INDEX: usize = 0;
const FOOD_GRID_BUFFER_INDEX: usize = 4;
const FOOD_STATE_BUFFER_INDEX: usize = 2;
const FOOD_FLAGS_BUFFER_INDEX: usize = 3;
const GRID_COLUMNS: usize = 13;
const FOOD_GRID_X_ORIGIN: f32 = -48.0;
const FOOD_GRID_Z_ORIGIN: f32 = 24.0;
const FOOD_HEIGHT: f32 = 0.35;
const OVERFLOW_POSITION: f32 = 96.0;
const WITHIN_CELL_SHIFT: f32 = 0.25;
const BELOW_GROUND_Y_SHIFT: f32 = 0.125;
const EXPIRING_TIMER: f32 = 0.000_001;
const TIMER_INITIALIZED: f32 = 10.0;
const CONSUMED_ITEM: usize = 0;
const CLAIMED_ITEM: usize = 1;
const CLAIMING_AGENT: u32 = 2;
const AGENT_GRID_Z: f32 = -24.0;
const TRANSITION_STEPS: usize = 12;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

fn metadata_base(kernel: &GpuKernel) -> usize {
    kernel
        .layout
        .feature_count
        .checked_add(SCRATCH_PREDICTION - SCRATCH_ENCODED)
        .unwrap()
}

fn check_scratch_contract(kernel: &GpuKernel) {
    assert_eq!(kernel.agent_count, AGENTS);
    assert_eq!(kernel.food_count, FOOD_ITEMS);
    assert!(CACHE_HEADER_WORDS + kernel.food_count <= PREDICTOR_DIMENSION);
    let base = metadata_base(kernel);
    assert!(base >= kernel.layout.feature_count);
    assert!(base + PREDICTOR_DIMENSION <= kernel.layout.brain_scratch_stride);
    let cells =
        usize::try_from(kernel.food_grid_buffer.size() / bytes(FOOD_GRID_CELL_STRIDE)).unwrap();
    assert!(cells < EXACT_INTEGER_LIMIT);
    let inner = include_str!("../shaders/kernel/brain_inner.wgsl");
    assert!(inner.contains("coop_predict_and_act(agent_id, tid, false);"));
    let main =
        global_credit::main_source(&optimized_brain(), kernel.has_subgroup, Some(cache(kernel)));
    assert_eq!(main.matches("packed_encoder.scratch[").count(), 2);
    assert!(main.contains("if (use_scratch_prediction) {"));
    assert!(main.contains(
        "s_prediction[i] = packed_encoder.scratch[agent_scratch + SCRATCH_PREDICTION + i];"
    ));
    assert!(main.contains(
        "packed_encoder.scratch[agent_scratch + SCRATCH_FEATURES + feature] = s_features[feature];"
    ));
    let credit = packed_encoder::credit_source(true);
    assert_eq!(credit.matches("packed_encoder.scratch[").count(), 1);
    assert!(credit.contains(
        "packed_encoder.scratch[agent_id * BRAIN_SCRATCH_STRIDE + SCRATCH_FEATURES + feature]"
    ));
}

fn cache_source(kernel: &GpuKernel) -> String {
    check_scratch_contract(kernel);
    let source = global_credit::global_source_with_store_suppression(Some(cache(kernel)), true);
    let source = replace_once(&source,
        "    let agent_count = wc_u32(WC_AGENT_COUNT);\n\n    phase_clear(tid);",
        "    let agent_count = wc_u32(WC_AGENT_COUNT);\n\n    food_cache_begin(tid);\n    phase_clear(tid);");
    let source = replace_once(&source,
        "        atomicStore(&food_grid[cell * FOOD_GRID_CELL_STRIDE], 0u);",
        "        if food_cache_rebuild() { atomicStore(&food_grid[cell * FOOD_GRID_CELL_STRIDE], 0u); }");
    let source = replace_once(
        &source,
        "    phase_food_grid(tid);\n",
        "    if food_cache_rebuild() { phase_food_grid(tid); }\n",
    );
    let first = source
        .find("        let food_base = cell * FOOD_GRID_CELL_STRIDE;")
        .unwrap();
    let last = first
        + source[first..]
            .find("\n        let agent_base = cell * AGENT_GRID_CELL_STRIDE;")
            .unwrap();
    let old = &source[first..last];
    let food = replace_once(old,
        "        let food_count = min(atomicLoad(&food_grid[food_base]), FOOD_GRID_MAX_PER_CELL);",
        "        let raw_food_count = atomicLoad(&food_grid[food_base]);\n        if raw_food_count > FOOD_GRID_MAX_PER_CELL { atomicStore(&food_cache_flags[FOOD_CACHE_OVERFLOW_FLAG], 1u); }\n        let food_count = min(raw_food_count, FOOD_GRID_MAX_PER_CELL);");
    let source = replace_once(
        &source,
        old,
        &format!("        if food_cache_rebuild() {{\n{food}        }}\n"),
    );
    let source = replace_once(&source,
        "    if gpc.trail_interval != 0u && gpc.tick % gpc.trail_interval == 0u {",
        "    food_cache_finish(tid);\n    if gpc.trail_interval != 0u && gpc.tick % gpc.trail_interval == 0u {");
    // Respawn timing/RNG/append order and claim clearing are canonical.
    assert!(source.contains(include_str!("../shaders/kernel/phase_food_respawn.wgsl")));
    assert!(source.contains("atomicStore(&food_flags[food_claim_slot(item)], FOOD_UNCLAIMED);"));
    format!("{source}\n{}", include_str!("food_grid_cache.wgsl"))
}

fn pipeline(
    kernel: &GpuKernel,
    source: String,
    constants: &HashMap<String, f64>,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("food_grid_cache_global"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("food_grid_cache_global"),
            bind_group_layouts: &[&binding],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
    kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("food_grid_cache_global"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("global_credit_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

fn invalidate_metadata(kernel: &GpuKernel) {
    kernel.queue.write_buffer(
        cache(kernel).buffer(),
        bytes(metadata_base(kernel)),
        bytemuck::cast_slice(&[0.0f32; CACHE_HEADER_WORDS]),
    );
}

fn restore_case(kernel: &mut GpuKernel, checkpoint: &Checkpoint) {
    restore(kernel, checkpoint);
    invalidate_metadata(kernel);
}

fn advance(kernel: &mut GpuKernel, cycle: u32, count: u32) {
    // The ordinary importer copies weights only; it cannot restore this
    // test-private metadata after a public checkpoint/reset or arm transition.
    if !cache(kernel).is_valid() {
        invalidate_metadata(kernel);
    }
    production_advance(kernel, cycle, count);
}

struct Arms {
    parked: Option<global_credit::Pipelines>,
    candidate: bool,
}

impl Arms {
    fn prepare(width: u32, height: u32, boundary: bool) -> (GpuKernel, Self) {
        let kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
        check_scratch_contract(&kernel);
        invalidate_metadata(&kernel);
        let mut constants = vision_override_constants(&kernel.layout);
        constants.insert("VISION_AGENT_MASKS".into(), 1.0);
        let mut candidate = global_credit::Pipelines::new_packed_with_store_suppression(
            &kernel,
            &optimized_brain(),
            &constants,
            true,
        )
        .unwrap();
        constants.insert(
            "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
            f64::from(
                candidate
                    .packed_encoder
                    .as_ref()
                    .unwrap()
                    .groups_per_agent(),
            ),
        );
        candidate.global = pipeline(&kernel, cache_source(&kernel), &constants);
        // Equal layouts mean the source's literal prefix fits this second cache.
        assert_eq!(
            cache(&kernel).buffer().size(),
            candidate.packed_encoder.as_ref().unwrap().buffer().size()
        );
        (
            kernel,
            Self {
                parked: Some(candidate),
                candidate: false,
            },
        )
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
            invalidate_metadata(kernel);
            self.candidate = candidate;
        }
        assert!(kernel.global_credit_active());
    }
}

fn metadata(kernel: &GpuKernel) -> TestResult<[f32; CACHE_HEADER_WORDS]> {
    let base = metadata_base(kernel);
    let raw = read_buffer(
        kernel,
        cache(kernel).buffer(),
        bytes(base + CACHE_HEADER_WORDS),
    )?;
    let words: &[f32] = bytemuck::cast_slice(&raw);
    Ok(words[base..base + CACHE_HEADER_WORDS].try_into().unwrap())
}

fn checked_state(kernel: &GpuKernel) -> TestResult<State> {
    let state = capture_state(kernel)?;
    assert_mirror(kernel, &state)?;
    Ok(state)
}

fn trajectory(kernel: &mut GpuKernel, candidate: bool) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    let mut cycle = 0;
    let mut reused = false;
    for count in PARITY_CHUNKS {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        advance(kernel, cycle, count);
        states.push(checked_state(kernel)?);
        if candidate {
            reused |= metadata(kernel)?[CACHE_REUSED] == 1.0;
        }
        cycle += count;
    }
    if candidate {
        assert!(reused, "fixture must exercise grid reuse");
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU; exact full-grid, public-state and private-mirror comparisons"]
fn food_grid_cache_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, false)?;
        for replay in 0..2 {
            restore_case(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel, true)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "food grid cache",
                );
            }
            println!("FOOD_GRID_CACHE_PARITY width={width} height={height} cycles=100 replay={replay} exact_buffers={MUTABLE_BUFFERS} entire_grid_including_stale_slots=true death_refresh=true private_mirror_exact=true");
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU full-cycle benchmark; run explicitly in release mode"]
fn benchmark_food_grid_cache() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut arms) = Arms::prepare(FIELDS[0].0, FIELDS[0].1, false);
    advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    let expected = checked_state(&kernel)?;
    restore_case(&mut kernel, &warm);
    arms.activate(&mut kernel, true);
    advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
    assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore_case(&mut kernel, &warm);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            advance(&mut kernel, WARMUP_CYCLES, TIMED_CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
        }
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!("FOOD_GRID_CACHE_TIMING warmup={WARMUP_CYCLES} cycles={TIMED_CYCLES} pairs={TIMING_PAIRS} baseline_seconds={baseline:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true metadata_words={} added_binding=false added_dispatch=false cold_import_timed=true full_simulation=true", baseline / candidate, CACHE_HEADER_WORDS + FOOD_ITEMS);
    Ok(())
}

fn write_food(kernel: &GpuKernel, item: usize, position: [f32; 3], consumed: u32, timer: f32) {
    let words = [position[0], position[1], position[2], timer];
    kernel.queue.write_buffer(
        &kernel.food_state_buffer,
        bytes(item * FOOD_STATE_STRIDE),
        bytemuck::cast_slice(&words),
    );
    kernel.queue.write_buffer(
        &kernel.food_flags_buffer,
        bytes(item),
        bytemuck::cast_slice(&[consumed]),
    );
}

fn food_position(item: usize) -> [f32; 3] {
    [
        FOOD_GRID_X_ORIGIN + (item % GRID_COLUMNS) as f32 * GRID_CELL_SIZE,
        FOOD_HEIGHT,
        FOOD_GRID_Z_ORIGIN + (item / GRID_COLUMNS) as f32 * GRID_CELL_SIZE,
    ]
}

fn transition_fixture(kernel: &GpuKernel) {
    for item in 0..FOOD_ITEMS {
        write_food(kernel, item, food_position(item), 0, 0.0);
    }
    for agent in 0..AGENTS {
        kernel.write_agent_physics_fields(
            agent,
            &[
                (P_POS_X, agent as f32 * GRID_CELL_SIZE),
                (P_POS_Z, AGENT_GRID_Z),
            ],
        );
    }
}

fn float_at(state: &State, buffer: usize, word: usize) -> f32 {
    let begin = word * WORD_BYTES;
    f32::from_le_bytes(state[buffer][begin..begin + WORD_BYTES].try_into().unwrap())
}

fn flag_at(state: &State, item: usize) -> u32 {
    let begin = item * WORD_BYTES;
    u32::from_le_bytes(
        state[FOOD_FLAGS_BUFFER_INDEX][begin..begin + WORD_BYTES]
            .try_into()
            .unwrap(),
    )
}

fn transition_trajectory(kernel: &mut GpuKernel, candidate: bool) -> TestResult<Vec<State>> {
    let mut states: Vec<State> = Vec::new();
    for step in 0..TRANSITION_STEPS {
        match step {
            2 => {
                let mut position = food_position(CONSUMED_ITEM);
                position[0] -= GRID_CELL_SIZE;
                write_food(kernel, CONSUMED_ITEM, position, 0, 0.0);
            }
            3 => {
                let mut position = food_position(CONSUMED_ITEM);
                position[0] -= GRID_CELL_SIZE - WITHIN_CELL_SHIFT;
                position[1] += BELOW_GROUND_Y_SHIFT;
                write_food(kernel, CONSUMED_ITEM, position, 0, 0.0);
            }
            4 => write_food(kernel, CONSUMED_ITEM, food_position(CONSUMED_ITEM), 1, 0.0),
            6 => write_food(
                kernel,
                CONSUMED_ITEM,
                food_position(CONSUMED_ITEM),
                1,
                EXPIRING_TIMER,
            ),
            8 => {
                let previous = states.last().unwrap();
                let base = usize::try_from(CLAIMING_AGENT).unwrap() * PHYS_STRIDE;
                let position = [
                    float_at(previous, PHYSICS_BUFFER_INDEX, base + P_POS_X),
                    float_at(previous, PHYSICS_BUFFER_INDEX, base + P_POS_Y),
                    float_at(previous, PHYSICS_BUFFER_INDEX, base + P_POS_Z),
                ];
                // Locomotion overwrites x/z velocity from these commands, so
                // the following claim sees the food at its exact x/z position.
                kernel.write_motor_decision(CLAIMING_AGENT, 0.0, 0.0, 0.0);
                write_food(kernel, CLAIMED_ITEM, position, 0, 0.0);
            }
            9 => {
                for item in 0..=FOOD_GRID_MAX_PER_CELL {
                    write_food(
                        kernel,
                        item,
                        [OVERFLOW_POSITION, FOOD_HEIGHT, OVERFLOW_POSITION],
                        0,
                        0.0,
                    );
                }
            }
            11 => {
                for item in 0..=FOOD_GRID_MAX_PER_CELL {
                    write_food(kernel, item, food_position(item), 0, 0.0);
                }
            }
            _ => {}
        }
        advance(kernel, u32::try_from(step).unwrap(), 1);
        let state = checked_state(kernel)?;
        if matches!(step, 1 | 3 | 5) {
            assert_eq!(
                states.last().unwrap()[FOOD_GRID_BUFFER_INDEX],
                state[FOOD_GRID_BUFFER_INDEX],
                "membership-stable entire grid"
            );
        }
        if step == 4 {
            assert_eq!(
                float_at(&state, FOOD_STATE_BUFFER_INDEX, FOOD_RESPAWN_TIMER),
                TIMER_INITIALIZED
            );
        }
        if step == 5 {
            let before = float_at(
                states.last().unwrap(),
                FOOD_STATE_BUFFER_INDEX,
                FOOD_RESPAWN_TIMER,
            );
            let after = float_at(&state, FOOD_STATE_BUFFER_INDEX, FOOD_RESPAWN_TIMER);
            assert!(
                after < before && after > 0.0,
                "reuse must still decrement timers"
            );
        }
        if step == 6 {
            assert_eq!(flag_at(&state, CONSUMED_ITEM), 0);
            assert_eq!(
                float_at(&state, FOOD_STATE_BUFFER_INDEX, FOOD_RESPAWN_TIMER),
                0.0
            );
        }
        if step == 8 {
            assert_eq!(
                flag_at(&state, CLAIMED_ITEM),
                1,
                "forced food must be consumed"
            );
        }
        if candidate {
            let meta = metadata(kernel)?;
            assert_eq!(
                meta[CACHE_REUSED],
                if matches!(step, 1 | 3 | 5) { 1.0 } else { 0.0 },
                "cache eligibility step={step}"
            );
            assert_eq!(meta[CACHE_EXPIRY], if step == 6 { 1.0 } else { 0.0 });
            assert_eq!(meta[CACHE_VALID], if step == 6 { 0.0 } else { 1.0 });
            assert_eq!(
                meta[CACHE_OVERFLOW],
                if matches!(step, 9 | 10) { 1.0 } else { 0.0 }
            );
        }
        states.push(state);
    }
    Ok(states)
}

#[test]
#[ignore = "requires GPU; includes consumption, timer expiry, overflow and host membership edits"]
fn food_grid_cache_preserves_transition_order_and_overflow_fallback() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, false);
        transition_fixture(&kernel);
        let saved = checkpoint(&kernel);
        let expected = transition_trajectory(&mut kernel, false)?;
        for replay in 0..2 {
            restore_case(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = transition_trajectory(&mut kernel, true)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
            }
            println!("FOOD_GRID_CACHE_TRANSITIONS width={width} height={height} steps={TRANSITION_STEPS} replay={replay} exact_buffers={MUTABLE_BUFFERS} stable_grid_reused=true host_membership_checked=true timer_progress=true respawn_phase_unchanged=true overflow_rebuild=true stale_slots_compared=true private_mirror_exact=true");
        }
    }
    Ok(())
}
