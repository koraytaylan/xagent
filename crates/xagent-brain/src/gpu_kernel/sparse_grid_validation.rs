//! Test-only occupied-cell clear/sort in the production packed/main128 schedule.
//! Original food/agent registration, respawn, collision, trail and encoder-credit
//! arithmetic remain intact. Checked private metadata records insertion cells
//! before collisions. Cold caches take the canonical complete grid scans.
//! Public counts, sorted entries, stale tails and visibility masks are compared.
//! This bounded fixture excludes overflowing cells because atomic insertion
//! winners in a capped overflowing grid are not guaranteed across pipelines.
//! Cache invalidation is explicit in this harness; no public lifecycle is added.

use std::{collections::HashMap, error::Error, time::Instant};

use super::cycle_profile::{assert_state_equal, capture_state, checkpoint, restore};
use super::packed_store_validation::{
    advance as production_advance, cache, optimized_brain, prepare_kernel_with_store_suppression,
};
use super::rounding_validation::assert_inactive_agent_unchanged;
use super::vision_validation::read_buffer;
use super::whitening_validation::{force_death, REFRESH_CYCLES};
use super::*;

/// Default and odd fields exercise both packed-layout and final-ray tails.
const FIELDS: [(u32, u32); 2] = [(8, 6), (9, 7)];
/// One invocation can snapshot one food and one agent sorting owner.
const MAX_ITEMS: u32 = 256;
/// Header holds valid, used-sparse, width, offset and both population counts.
const HEADER_WORDS: usize = 6;
const VALID: usize = 0;
const USED: usize = 1;
const WIDTH: usize = 2;
const OFFSET: usize = 3;
const VECTOR_WORDS: usize = 4;
const WORD_BYTES: usize = size_of::<f32>();
const PUSH_CONSTANT_BYTES: u32 = 8;
/// Positive cell+1 codes must remain exact when stored as f32.
const EXACT_INTEGER_LIMIT: usize = 1 << 24;
/// Complete mutable public state plus a separate private-matrix comparison.
const MUTABLE_BUFFERS: usize = 13;
const PHYSICS_BUFFER: usize = 0;
const FOOD_GRID_BUFFER: usize = 4;
const AGENT_GRID_BUFFER: usize = 5;
const BRAIN_BUFFER: usize = 8;
const INACTIVE_AGENT: u32 = 1;
/// One hundred cycles include multiple whitening and forced-death boundaries.
const CYCLES: u32 = 100;
const REPLAYS: usize = 2;
const WARMUP_CYCLES: u32 = 1_000;
const TIMING_PAIRS: usize = 5;
/// Spread foods across thirteen columns, away from the agent collision fixture.
const FOOD_COLUMNS: usize = 13;
const FOOD_X: f32 = -48.0;
const FOOD_Z: f32 = 24.0;
const FOOD_Y: f32 = 0.35;
const AGENT_Z: f32 = -24.0;
const OTHER_AGENTS_Z: f32 = -64.0;
const COLLISION_RIGHT_AGENT: u32 = 2;
const COLLISION_LEFT_AGENT: u32 = 3;
const EDGE_DISTANCE: f32 = 0.1;
const PAIR_DISTANCE: f32 = 0.3;
const OUTSIDE_POSITION: f32 = 10_000.0;
const EXPIRING_TIMER: f32 = 0.000_001;
const TRANSITIONS: u32 = 14;
const RESET_SEED: u64 = 314;
const HOST_WEIGHT: f32 = 0.125;
const FOOD_FLAGS_BUFFER: usize = 3;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
type State = Vec<Vec<u8>>;

fn bytes(words: usize) -> u64 {
    u64::try_from(words.checked_mul(WORD_BYTES).unwrap()).unwrap()
}

fn aligned_words(words: usize) -> usize {
    words.checked_add(VECTOR_WORDS - 1).unwrap() / VECTOR_WORDS * VECTOR_WORDS
}

struct MetadataLayout {
    base: usize,
    extra: usize,
    prefix: usize,
    agent_cells: usize,
}

impl MetadataLayout {
    fn new(kernel: &GpuKernel) -> Self {
        assert!(!kernel.layout.visual_cortex_enabled);
        assert!(kernel.agent_count > 0 && kernel.agent_count <= MAX_ITEMS);
        assert!(kernel.food_count > 0 && kernel.food_count <= usize::try_from(MAX_ITEMS).unwrap());
        let agents = usize::try_from(kernel.agent_count).unwrap();
        let base = aligned_words(
            kernel
                .layout
                .brain_scratch_stride
                .checked_mul(agents)
                .unwrap(),
        );
        let agent_cells = HEADER_WORDS.checked_add(kernel.food_count).unwrap();
        let extra = agent_cells.checked_add(agents).unwrap();
        let prefix = aligned_words(base.checked_add(extra).unwrap());
        u32::try_from(prefix).unwrap();
        let cells =
            usize::try_from(kernel.food_grid_buffer.size() / bytes(FOOD_GRID_CELL_STRIDE)).unwrap();
        assert!(cells.checked_add(1).unwrap() <= EXACT_INTEGER_LIMIT);
        Self {
            base,
            extra,
            prefix,
            agent_cells,
        }
    }

    fn source(&self) -> String {
        format!(
            "\nconst SPARSE_GRID_BASE: u32 = {}u;\nconst SPARSE_VALID: u32 = 0u;\nconst SPARSE_USED: u32 = 1u;\nconst SPARSE_WIDTH: u32 = 2u;\nconst SPARSE_OFFSET: u32 = 3u;\nconst SPARSE_FOODS: u32 = 4u;\nconst SPARSE_AGENTS: u32 = 5u;\nconst SPARSE_FOOD_CELLS: u32 = {HEADER_WORDS}u;\nconst SPARSE_AGENT_CELLS: u32 = {}u;\n",
            self.base, self.agent_cells
        )
    }
}

fn replace_once(source: &str, old: &str, new: &str) -> String {
    assert_eq!(
        source.matches(old).count(),
        1,
        "unique source target: {old}"
    );
    source.replacen(old, new, 1)
}

/// Preserve the complete insertion-sort bodies without duplicating their math.
fn sorting_helpers() -> String {
    let original = include_str!("../shaders/kernel/phase_grid_order.wgsl");
    let food = original
        .find("        let food_base = cell * FOOD_GRID_CELL_STRIDE;")
        .unwrap();
    let agent = original
        .find("        let agent_base = cell * AGENT_GRID_CELL_STRIDE;")
        .unwrap();
    let end = original.rfind("    }\n}").unwrap();
    format!(
        "\nfn sparse_sort_food(cell: u32) {{\n{}\n}}\nfn sparse_sort_agents(cell: u32) {{\n{}\n}}\n",
        &original[food..agent],
        &original[agent..end]
    )
}

fn clear_remainder() -> String {
    let original = include_str!("../shaders/kernel/phase_clear.wgsl");
    let begin = original
        .find("    // No food is claimed until the next claim step")
        .unwrap();
    format!(
        "\nfn sparse_clear_remainder(tid: u32) {{\n{}",
        &original[begin..]
    )
}

fn registration_sources() -> [String; 3] {
    let food = include_str!("../shaders/kernel/phase_food_grid.wgsl");
    let food = replace_once(
        food,
        "    for (var i = tid; i < food_count; i += 256u) {",
        "    for (var i = tid; i < food_count; i += 256u) {\n        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOOD_CELLS + i] = 0.0;",
    );
    let food = replace_once(
        &food,
        "        let cell_idx = ucx * grid_w + ucz;",
        "        let cell_idx = ucx * grid_w + ucz;\n        packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOOD_CELLS + i] = f32(cell_idx + 1u);",
    );
    let respawn = replace_once(
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        "                let cell_idx = ucx * grid_w + ucz;",
        "                let cell_idx = ucx * grid_w + ucz;\n                packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_FOOD_CELLS + i] = f32(cell_idx + 1u);",
    );
    let agents = replace_once(
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        "    if agent >= agent_count { return; }",
        "    if agent >= agent_count { return; }\n    packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENT_CELLS + agent] = 0.0;",
    );
    let agents = replace_once(
        &agents,
        "    let cell_idx = ucx * grid_w + ucz;",
        "    let cell_idx = ucx * grid_w + ucz;\n    packed_encoder.scratch[SPARSE_GRID_BASE + SPARSE_AGENT_CELLS + agent] = f32(cell_idx + 1u);",
    );
    [food, respawn, agents]
}

fn sparse_source(packed: &packed_encoder::Cache, layout: &MetadataLayout) -> String {
    let mut source = global_credit::global_source_with_store_suppression(Some(packed), true);
    let original = source.clone();
    let phases = [
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
    ];
    for (original, candidate) in phases.into_iter().zip(registration_sources()) {
        source = replace_once(&source, original, &candidate);
    }
    source = replace_once(
        &source,
        "    phase_clear(tid);",
        "    phase_sparse_clear(tid);",
    );
    source = replace_once(
        &source,
        "    phase_sort_grid_cells(tid);",
        "    phase_sparse_sort_grid_cells(tid);",
    );
    source = replace_once(
        &source,
        "    if gpc.trail_interval != 0u && gpc.tick % gpc.trail_interval == 0u {",
        "    sparse_grid_finish(tid);\n    if gpc.trail_interval != 0u && gpc.tick % gpc.trail_interval == 0u {",
    );
    source.push_str(&layout.source());
    source.push_str(&sorting_helpers());
    source.push_str(&clear_remainder());
    source.push_str(include_str!("sparse_grid.wgsl"));
    assert_ne!(source, original);
    assert!(!source
        .lines()
        .any(|line| line.trim().starts_with("var<workgroup>")));
    assert!(source.contains(&packed_encoder::credit_source(true)));
    source
}

fn constants(kernel: &GpuKernel, packed: &packed_encoder::Cache) -> HashMap<String, f64> {
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    constants.insert(
        "GLOBAL_CREDIT_GROUPS_PER_AGENT".into(),
        f64::from(packed.groups_per_agent()),
    );
    constants
}

fn pipeline(
    kernel: &GpuKernel,
    packed: &packed_encoder::Cache,
    source: String,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sparse_grid_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sparse_grid_probe"),
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
                constants: &constants(kernel, packed),
                ..Default::default()
            },
            cache: None,
        })
}

struct Arms {
    parked: Option<global_credit::Pipelines>,
    candidate: bool,
    layout: MetadataLayout,
}

impl Arms {
    fn prepare(width: u32, height: u32, boundary: bool) -> (GpuKernel, Self) {
        let mut kernel = prepare_kernel_with_store_suppression(width, height, boundary, true);
        kernel.global_credit = global_credit::Pipelines::new_packed_with_main128(
            &kernel,
            &optimized_brain(),
            &constants(&kernel, cache(&kernel)),
            true,
        );
        let layout = MetadataLayout::new(&kernel);
        let packed = packed_encoder::Cache::new_with_extra_scratch(&kernel, layout.extra).unwrap();
        assert!(packed
            .common_source()
            .contains(&format!("scratch: array<f32, {}>", layout.prefix)));
        let mut candidate = global_credit::Pipelines::new_packed_with_main128(
            &kernel,
            &optimized_brain(),
            &constants(&kernel, &packed),
            true,
        )
        .unwrap();
        let main =
            global_credit::main_source(&optimized_brain(), kernel.has_subgroup, Some(&packed));
        candidate.main = pipeline(
            &kernel,
            &packed,
            main_width::try_transform(&main).unwrap(),
            "kernel_tick",
        );
        candidate.global = pipeline(
            &kernel,
            &packed,
            sparse_source(&packed, &layout),
            "global_credit_tick",
        );
        let binding = kernel.kernel_pipeline.get_bind_group_layout(0);
        candidate.bind_groups = std::array::from_fn(|index| {
            global_credit::private_bind_group(&kernel, &binding, packed.buffer(), index)
        });
        candidate.packed_encoder = Some(packed);
        assert_eq!(candidate.main_threads, main_width::MAIN_THREADS);
        assert_eq!(
            kernel.global_credit.as_ref().unwrap().main_threads,
            main_width::MAIN_THREADS
        );
        (
            kernel,
            Self {
                parked: Some(candidate),
                candidate: false,
                layout,
            },
        )
    }

    fn invalidate_metadata(&self, kernel: &GpuKernel) {
        assert!(self.candidate);
        kernel.queue.write_buffer(
            cache(kernel).buffer(),
            bytes(self.layout.base),
            bytemuck::cast_slice(&[0.0f32; HEADER_WORDS]),
        );
    }

    fn activate(&mut self, kernel: &mut GpuKernel, candidate: bool) {
        if candidate != self.candidate {
            self.parked
                .as_ref()
                .unwrap()
                .packed_encoder
                .as_ref()
                .unwrap()
                .invalidate();
            std::mem::swap(&mut self.parked, &mut kernel.global_credit);
            self.candidate = candidate;
            if candidate {
                self.invalidate_metadata(kernel);
            }
        }
        assert!(kernel.global_credit_active());
    }

    fn advance(&self, kernel: &mut GpuKernel, cycle: u32, cycles: u32) {
        assert!(
            kernel.agent_count <= MAX_ITEMS
                && kernel.food_count <= usize::try_from(MAX_ITEMS).unwrap()
        );
        if self.candidate && !cache(kernel).is_valid() {
            self.invalidate_metadata(kernel);
        }
        production_advance(kernel, cycle, cycles);
    }

    fn metadata(&self, kernel: &GpuKernel) -> TestResult<Vec<f32>> {
        assert!(self.candidate);
        let raw = read_buffer(
            kernel,
            cache(kernel).buffer(),
            bytes(self.layout.base + self.layout.extra),
        )?;
        let values: &[f32] = bytemuck::cast_slice(&raw);
        Ok(values[self.layout.base..self.layout.base + self.layout.extra].to_vec())
    }
}

fn assert_mirror(kernel: &GpuKernel, state: &State) -> TestResult {
    let agents = usize::try_from(kernel.agent_count).unwrap();
    let matrix_words = kernel
        .layout
        .feature_count
        .checked_mul(ENCODED_DIMENSION)
        .unwrap();
    let matrix_bytes = bytes(matrix_words);
    let packed = cache(kernel).buffer();
    let prefix = packed
        .size()
        .checked_sub(
            matrix_bytes
                .checked_mul(u64::from(kernel.agent_count))
                .unwrap(),
        )
        .unwrap();
    let data = read_buffer(kernel, packed, packed.size())?;
    for agent in 0..agents {
        let private = usize::try_from(prefix + bytes(agent * matrix_words)).unwrap();
        let public = usize::try_from(bytes(agent * kernel.layout.brain_stride)).unwrap();
        let length = usize::try_from(matrix_bytes).unwrap();
        assert_eq!(
            &data[private..private + length],
            &state[BRAIN_BUFFER][public..public + length],
            "packed mirror agent={agent}"
        );
    }
    Ok(())
}

fn assert_grid_counts(state: &State) {
    let foods: &[u32] = bytemuck::cast_slice(&state[FOOD_GRID_BUFFER]);
    let agents: &[u32] = bytemuck::cast_slice(&state[AGENT_GRID_BUFFER]);
    assert_eq!(foods.len() % FOOD_GRID_CELL_STRIDE, 0);
    let cells = foods.len() / FOOD_GRID_CELL_STRIDE;
    for cell in 0..cells {
        assert!(
            foods[cell * FOOD_GRID_CELL_STRIDE] <= u32::try_from(FOOD_GRID_MAX_PER_CELL).unwrap(),
            "food overflow cell={cell}"
        );
        assert!(
            agents[cell * AGENT_GRID_CELL_STRIDE]
                <= u32::try_from(AGENT_GRID_MAX_PER_CELL).unwrap(),
            "agent overflow cell={cell}"
        );
    }
}

fn checked_state(kernel: &GpuKernel) -> TestResult<State> {
    let state = capture_state(kernel)?;
    assert_grid_counts(&state);
    assert_mirror(kernel, &state)?;
    Ok(state)
}

fn trajectory(kernel: &mut GpuKernel, arms: &Arms) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    for cycle in 0..CYCLES {
        if cycle == REFRESH_CYCLES {
            force_death(kernel);
        }
        arms.advance(kernel, cycle, 1);
        states.push(checked_state(kernel)?);
        if arms.candidate {
            let metadata = arms.metadata(kernel)?;
            assert_eq!(metadata[VALID], 1.0);
            assert_eq!(metadata[USED], if cycle == 0 { 0.0 } else { 1.0 });
        }
    }
    Ok(states)
}

#[test]
#[ignore = "GPU all13, complete grids/masks, pre-collision metadata and private mirror"]
fn sparse_grid_preserves_complete_state() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, true);
        let initial = capture_state(&kernel)?;
        let saved = checkpoint(&kernel);
        let expected = trajectory(&mut kernel, &arms)?;
        for replay in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = trajectory(&mut kernel, &arms)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
                assert_inactive_agent_unchanged(
                    &kernel,
                    &initial,
                    actual,
                    INACTIVE_AGENT,
                    "sparse grid",
                );
            }
            println!(
                "SPARSE_GRID_PARITY width={width} height={height} cycles={CYCLES} replay={replay} exact_buffers={MUTABLE_BUFFERS} every_cycle_no_overflow=true complete_grid_stale_tails_and_masks=true private_mirror_exact=true death_refresh=true cold_then_warm=true main_threads=128 global_threads=256"
            );
        }
    }
    Ok(())
}

/// Inspect every untimed evolving cycle, since a later clear hides old overflow.
fn timing_preflight(kernel: &mut GpuKernel, arms: &Arms) -> TestResult<State> {
    let mut last = Vec::new();
    for cycle in WARMUP_CYCLES..WARMUP_CYCLES + CYCLES {
        arms.advance(kernel, cycle, 1);
        last = checked_state(kernel)?;
    }
    Ok(last)
}

#[test]
#[ignore = "GPU full-cycle timing; run explicitly in release mode"]
fn benchmark_sparse_grid_maintenance() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let (mut kernel, mut arms) = Arms::prepare(FIELDS[0].0, FIELDS[0].1, false);
    arms.advance(&mut kernel, 0, WARMUP_CYCLES);
    let warm = checkpoint(&kernel);
    let expected = timing_preflight(&mut kernel, &arms)?;
    restore(&mut kernel, &warm);
    arms.activate(&mut kernel, true);
    let preflight = timing_preflight(&mut kernel, &arms)?;
    assert_state_equal(&kernel, &expected, &preflight);
    let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    for pair in 0..TIMING_PAIRS {
        for arm in [pair % 2, 1 - pair % 2] {
            restore(&mut kernel, &warm);
            arms.activate(&mut kernel, arm != 0);
            let start = Instant::now();
            arms.advance(&mut kernel, WARMUP_CYCLES, CYCLES);
            timings[arm].push(start.elapsed().as_secs_f64());
            assert_state_equal(&kernel, &expected, &checked_state(&kernel)?);
        }
    }
    for values in &mut timings {
        values.sort_by(f64::total_cmp);
    }
    let reference = timings[0][TIMING_PAIRS / 2];
    let candidate = timings[1][TIMING_PAIRS / 2];
    println!(
        "SPARSE_GRID_TIMING warmup_cycles={WARMUP_CYCLES} cycles={CYCLES} pairs={TIMING_PAIRS} reference_seconds={reference:.9} candidate_seconds={candidate:.9} speedup={:.6} exact_buffers={MUTABLE_BUFFERS} private_mirror_exact=true metadata_words={} added_workgroup_bytes=0 same_dispatches=true same_production_recorder=true first_cold_rebuild_and_import_timed=true full_simulation=true",
        reference / candidate,
        arms.layout.extra
    );
    Ok(())
}

fn food_position(item: usize) -> [f32; 3] {
    [
        FOOD_X + (item % FOOD_COLUMNS) as f32 * GRID_CELL_SIZE,
        FOOD_Y,
        FOOD_Z + (item / FOOD_COLUMNS) as f32 * GRID_CELL_SIZE,
    ]
}

fn write_food(kernel: &GpuKernel, item: usize, position: [f32; 3], consumed: u32, timer: f32) {
    kernel.queue.write_buffer(
        &kernel.food_state_buffer,
        bytes(item * FOOD_STATE_STRIDE),
        bytemuck::cast_slice(&[position[0], position[1], position[2], timer]),
    );
    kernel.queue.write_buffer(
        &kernel.food_flags_buffer,
        bytes(item),
        bytemuck::cast_slice(&[consumed]),
    );
}

fn transition_fixture(kernel: &GpuKernel) {
    for item in 0..kernel.food_count {
        write_food(kernel, item, food_position(item), 0, 0.0);
    }
    for agent in 0..kernel.agent_count {
        kernel.write_motor_decision(agent, 0.0, 0.0, 0.0);
        kernel.write_agent_physics_fields(
            agent,
            &[
                (P_POS_X, agent as f32 * GRID_CELL_SIZE),
                (P_POS_Z, OTHER_AGENTS_Z),
            ],
        );
    }
    for (agent, distance) in [
        (COLLISION_RIGHT_AGENT, EDGE_DISTANCE),
        (COLLISION_LEFT_AGENT, EDGE_DISTANCE + PAIR_DISTANCE),
    ] {
        kernel.write_agent_physics_fields(
            agent,
            &[(P_POS_X, GRID_CELL_SIZE - distance), (P_POS_Z, AGENT_Z)],
        );
    }
}

fn float_at(state: &State, buffer: usize, word: usize) -> f32 {
    let begin = word * WORD_BYTES;
    f32::from_le_bytes(state[buffer][begin..begin + WORD_BYTES].try_into().unwrap())
}

fn transition_trajectory(kernel: &mut GpuKernel, arms: &Arms) -> TestResult<Vec<State>> {
    let mut states = Vec::new();
    for step in 0..TRANSITIONS {
        match step {
            2 => {
                let mut position = food_position(0);
                position[0] += GRID_CELL_SIZE;
                write_food(kernel, 0, position, 0, 0.0);
            }
            3 => write_food(kernel, 0, food_position(0), 1, 0.0),
            4 => write_food(kernel, 0, food_position(0), 1, EXPIRING_TIMER),
            5 => write_food(
                kernel,
                0,
                [OUTSIDE_POSITION, FOOD_Y, OUTSIDE_POSITION],
                0,
                0.0,
            ),
            6 => {
                write_food(kernel, 0, food_position(0), 0, 0.0);
                kernel.write_agent_physics_fields(
                    COLLISION_RIGHT_AGENT,
                    &[(P_ALIVE, 0.0), (P_DIED_FLAG, 0.0)],
                );
            }
            7 => kernel.write_agent_physics_fields(
                COLLISION_RIGHT_AGENT,
                &[(P_POS_X, OUTSIDE_POSITION), (P_POS_Z, OUTSIDE_POSITION)],
            ),
            8 => kernel.write_agent_physics_fields(
                COLLISION_RIGHT_AGENT,
                &[
                    (P_ALIVE, 1.0),
                    (P_POS_X, GRID_CELL_SIZE),
                    (P_POS_Z, AGENT_Z),
                ],
            ),
            9 => {
                if arms.candidate {
                    kernel.queue.write_buffer(
                        cache(kernel).buffer(),
                        bytes(arms.layout.base + WIDTH),
                        bytemuck::cast_slice(&[0.0f32]),
                    );
                }
            }
            10 => {
                let config = BrainConfig {
                    vision_width: kernel.layout.vision_width,
                    vision_height: kernel.layout.vision_height,
                    ..BrainConfig::default()
                };
                kernel.reset_agents_seeded(&config, RESET_SEED);
                assert!(!cache(kernel).is_valid());
            }
            12 => {
                let mut state = kernel.read_agent_state(0);
                state.brain_state[0] = HOST_WEIGHT;
                kernel.write_agent_state(0, &state);
                assert!(!cache(kernel).is_valid());
            }
            _ => {}
        }
        arms.advance(kernel, step, 1);
        let state = checked_state(kernel)?;
        if step == 0 {
            let base = usize::try_from(COLLISION_RIGHT_AGENT).unwrap() * PHYS_STRIDE;
            let registered_x = float_at(&state, PHYSICS_BUFFER, base + P_BRAIN_POS_X);
            let final_x = float_at(&state, PHYSICS_BUFFER, base + P_POS_X);
            assert_ne!(
                (registered_x / GRID_CELL_SIZE).floor(),
                (final_x / GRID_CELL_SIZE).floor(),
                "fixture must cross a registration boundary after collision"
            );
            if arms.candidate {
                let metadata = arms.metadata(kernel)?;
                let registered_z = float_at(&state, PHYSICS_BUFFER, base + P_BRAIN_POS_Z);
                let column = (registered_x / GRID_CELL_SIZE).floor() + metadata[OFFSET];
                let row = (registered_z / GRID_CELL_SIZE).floor() + metadata[OFFSET];
                let code = column * metadata[WIDTH] + row + 1.0;
                let agent = usize::try_from(COLLISION_RIGHT_AGENT).unwrap();
                assert_eq!(
                    metadata[arms.layout.agent_cells + agent],
                    code,
                    "cache must record the cell before collision movement"
                );
            }
        }
        if arms.candidate {
            let metadata = arms.metadata(kernel)?;
            assert_eq!(metadata[VALID], 1.0);
            assert_eq!(
                metadata[USED],
                if matches!(step, 0 | 9 | 10 | 12) {
                    0.0
                } else {
                    1.0
                },
                "cold/geometry/reset/host-write gate step={step}"
            );
            if matches!(step, 3 | 5) {
                assert_eq!(metadata[HEADER_WORDS], 0.0);
            }
            if step == 4 {
                let flags: &[u32] = bytemuck::cast_slice(&state[FOOD_FLAGS_BUFFER]);
                assert_eq!(flags[0], 0, "expired food must become available");
                assert!(
                    metadata[HEADER_WORDS] > 0.0,
                    "respawn must publish its new registration"
                );
            }
            if matches!(step, 6 | 7) {
                assert_eq!(
                    metadata
                        [arms.layout.agent_cells + usize::try_from(COLLISION_RIGHT_AGENT).unwrap()],
                    0.0
                );
            }
        }
        states.push(state);
    }
    Ok(states)
}

#[test]
#[ignore = "GPU movement, consumed/respawned/outside objects, cache reset and host write"]
fn sparse_grid_preserves_transitions_and_cold_fallback() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in FIELDS {
        let (mut kernel, mut arms) = Arms::prepare(width, height, false);
        transition_fixture(&kernel);
        let saved = checkpoint(&kernel);
        let expected = transition_trajectory(&mut kernel, &arms)?;
        for replay in 0..REPLAYS {
            restore(&mut kernel, &saved);
            arms.activate(&mut kernel, true);
            let actual = transition_trajectory(&mut kernel, &arms)?;
            for (expected, actual) in expected.iter().zip(&actual) {
                assert_state_equal(&kernel, expected, actual);
            }
            println!(
                "SPARSE_GRID_TRANSITIONS width={width} height={height} steps={TRANSITIONS} replay={replay} exact_buffers={MUTABLE_BUFFERS} post_collision_crossing=true consumed_respawn_outside=true inactive_reactivation=true geometry_guard=true reset_and_host_write_cold=true private_mirror_exact=true entire_grid_stale_tails_and_masks=true"
            );
        }
    }
    Ok(())
}

#[test]
fn sparse_grid_keeps_canonical_sorts_and_registration_arithmetic() {
    let helpers = sorting_helpers();
    let original = include_str!("../shaders/kernel/phase_grid_order.wgsl");
    for line in original.lines().filter(|line| {
        line.trim_start().starts_with("atomicStore(") || line.trim_start().starts_with("let key =")
    }) {
        assert!(helpers.contains(line));
    }
    for (candidate, original) in registration_sources().into_iter().zip([
        include_str!("../shaders/kernel/phase_food_grid.wgsl"),
        include_str!("../shaders/kernel/phase_food_respawn.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
    ]) {
        assert_ne!(candidate, original);
        let restored = candidate
            .lines()
            .filter(|line| !line.contains("packed_encoder.scratch["))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(restored, original.trim_end());
    }
    let sparse = include_str!("sparse_grid.wgsl");
    let selection = sparse.find("agent_owner = atomicLoad").unwrap();
    let publish = sparse[selection..]
        .find("storageBarrier(); workgroupBarrier();")
        .unwrap()
        + selection;
    let first_sort = sparse.find("if food_owner { sparse_sort_food").unwrap();
    assert!(selection < publish && publish < first_sort);
    assert!(!sparse.contains("var<workgroup>"));
}
