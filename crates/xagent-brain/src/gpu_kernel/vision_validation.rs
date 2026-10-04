//! Opt-in GPU tests compare every sensory bit against the serial ray marcher
//! on identical buffers, then time repeated vision dispatches with GPU queries.

use std::error::Error;

use glam::Vec3;
use rand::{rngs::StdRng, Rng, SeedableRng};
use xagent_shared::{NOSTRIL_FORWARD_OFFSET, NOSTRIL_SIDE_OFFSET, SCENT_RANGE};

use super::*;

/// Small odd dimensions contain an exactly forward-facing centre ray.
const CENTRE_SIDE: u32 = 3;
/// Position of the centre ray in a three-by-three visual field.
const CENTRE_RAY: usize = 4;
/// The upper centre ray tilts upward without a horizontal component.
const UPPER_RAY: usize = 1;
/// The production terrain lattice has 128 segments per axis.
const TERRAIN_SIDE: usize = 129;
/// Full initial energy and integrity avoid irrelevant state boundaries.
const FULL_METER: f32 = 100.0;
/// First sampled distance in the production marcher.
const FIRST_SAMPLE: f32 = 1.2;
/// The production ray limit normalizes depth.
const MAX_DISTANCE: f32 = 30.0;
/// A finite sentinel detects unwritten sensory values and dead-agent writes.
const SENTINEL: f32 = -123.5;
/// Colours intentionally mirror observable shader output, not its control flow.
const FOOD_COLOR: [f32; 4] = [0.7, 0.95, 0.2, 1.0];
const AGENT_COLOR: [f32; 4] = [0.9, 0.2, 0.6, 1.0];
const TERRAIN_COLOR: [f32; 4] = [0.15, 0.5, 0.1, 1.0];
const SKY_COLOR: [f32; 4] = [0.53, 0.81, 0.92, 1.0];
/// More than one wave's worth of repeated calls amortizes timestamp overhead.
const TIMED_DISPATCHES: u32 = 128;
/// An odd sample count yields an unambiguous median.
const TIMING_ROUNDS: usize = 7;
/// Draws cover different worlds without relying on run-dependent randomness.
const SCENE_SEED: u64 = 0x194f_c3a7;
/// A representative default population from the cycle latency investigation.
const DEFAULT_POPULATION: u32 = 10;
/// Enough food to exercise repeated grid reads and nonvisual smell work.
const RANDOM_FOOD_COUNT: usize = 104;
/// Sweeps include minimal, odd, default, and larger visual fields.
const VISION_DIMENSIONS: [(u32, u32); 4] = [(2, 2), (3, 3), (8, 6), (13, 9)];
/// Inputs outside the allowed gene range verify the same clamping as serial.
const FOV_PAIRS: [(f32, f32); 4] = [(-10.0, -10.0), (60.0, 40.0), (90.0, 60.0), (500.0, 500.0)];
/// Cooperative groups can hold at most eight 32-lane rays.
const PARALLEL_GROUP_WIDTHS: [u32; 3] = [1, 4, 8];

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Default)]
struct VisionOptions {
    parallel_steps: bool,
    agent_masks: bool,
    object_queries: bool,
    parallel_scent: bool,
}

struct VisionPipeline {
    pipeline: wgpu::ComputePipeline,
    workgroups: u32,
    rays_per_group: u32,
    options: VisionOptions,
}

fn make_pipeline(
    kernel: &GpuKernel,
    rays_per_group: u32,
    include_senses: bool,
    mut options: VisionOptions,
) -> VisionPipeline {
    options.parallel_scent &= include_senses;
    let common = with_plain_grid_bindings(include_str!("../shaders/kernel/common.wgsl"));
    let source = if options.parallel_steps {
        compose_vision_source(&common, options.object_queries, options.parallel_scent)
    } else {
        assert!(!options.agent_masks && !options.object_queries && !options.parallel_scent);
        [
            common.as_str(),
            include_str!("../shaders/kernel/phase_vision.wgsl"),
            include_str!("vision_serial_reference.wgsl"),
        ]
        .join("\n")
    };
    let source = if include_senses {
        source
    } else {
        let senses_call = "phase_vision_senses(agent_id);";
        assert_eq!(source.matches(senses_call).count(), 1);
        source.replace(senses_call, "")
    };
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vision_parity_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.vision_pipeline.get_bind_group_layout(0);
    let pipeline_layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vision_parity_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert(
        "VISION_RAYS_PER_WORKGROUP".into(),
        f64::from(rays_per_group),
    );
    if options.parallel_steps {
        constants.insert("VISION_PARALLEL_STEPS".into(), 1.0);
        constants.insert(
            "VISION_AGENT_MASKS".into(),
            f64::from(u32::from(options.agent_masks)),
        );
        if options.object_queries {
            constants.insert("VISION_OBJECT_QUERIES".into(), 1.0);
        }
        if options.parallel_scent {
            constants.insert("VISION_PARALLEL_SCENT".into(), 1.0);
        }
    }
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(if options.parallel_steps {
                "parallel_vision_probe"
            } else {
                "serial_vision_reference"
            }),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("vision_tick"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
    let rays = kernel.layout.vision_width * kernel.layout.vision_height;
    VisionPipeline {
        pipeline,
        workgroups: kernel.agent_count * rays.div_ceil(rays_per_group),
        rays_per_group,
        options,
    }
}

pub(super) fn make_kernel(width: u32, height: u32, agents: u32, food_count: usize) -> GpuKernel {
    let brain = BrainConfig {
        vision_width: width,
        vision_height: height,
        ..BrainConfig::default()
    };
    let mut kernel = GpuKernel::new(agents, food_count, &brain, &WorldConfig::default());
    kernel.reset_agents_seeded(&brain, SCENE_SEED);
    kernel.upload_world_config(0, kernel.kernel_batch_size());
    kernel
}

/// Build the original ray and senses entry without any cooperative fragments.
/// Return its own dispatch count so callers cannot reuse an optimized count.
pub(super) fn pure_serial_pipeline(kernel: &GpuKernel) -> (wgpu::ComputePipeline, u32) {
    let reference = make_pipeline(
        kernel,
        SMALL_POPULATION_RAYS_PER_WORKGROUP,
        true,
        VisionOptions::default(),
    );
    (reference.pipeline, reference.workgroups)
}

fn dispatch_mask_builder(kernel: &GpuKernel, entry: &str) {
    let source = [
        include_str!("../shaders/kernel/common.wgsl"),
        include_str!("../shaders/kernel/phase_clear.wgsl"),
        include_str!("../shaders/kernel/phase_agent_grid.wgsl"),
        include_str!("vision_mask_validation.wgsl"),
    ]
    .join("\n");
    let module = kernel
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production_mask_builder_probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let bind_layout = kernel.vision_pipeline.get_bind_group_layout(0);
    let layout = kernel
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mask_builder_probe_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
    let mut constants = vision_override_constants(&kernel.layout);
    constants.insert("VISION_AGENT_MASKS".into(), 1.0);
    let pipeline = kernel
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        });
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    kernel.queue.submit([encoder.finish()]);
}

fn build_retained_grid_masks(kernel: &GpuKernel) {
    dispatch_mask_builder(kernel, "masks_from_retained_grid");
}

fn assert_grid_masks(kernel: &GpuKernel, expected_entries: usize) -> TestResult {
    let raw = read_buffer(
        kernel,
        &kernel.agent_grid_buffer,
        kernel.agent_grid_buffer.size(),
    )?;
    let grid: &[u32] = bytemuck::cast_slice(&raw);
    let width = grid_width(kernel.world_config.world_size);
    let cells = width.checked_mul(width).unwrap();
    let mut expected = vec![0_u32; cells];
    let mut entries = 0;
    for cell in 0..cells {
        let base = cell * AGENT_GRID_CELL_STRIDE;
        let count = usize::try_from(grid[base])?.min(AGENT_GRID_MAX_PER_CELL);
        entries += count;
        for &agent in &grid[base + 1..base + 1 + count] {
            let column = cell / width;
            let row = cell % width;
            for center_column in column.saturating_sub(1)..=(column + 1).min(width - 1) {
                for center_row in row.saturating_sub(1)..=(row + 1).min(width - 1) {
                    expected[center_column * width + center_row] |= 1_u32 << agent;
                }
            }
        }
    }
    assert_eq!(
        entries, expected_entries,
        "production grid must retain every live test agent"
    );
    assert_eq!(
        &grid[cells * AGENT_GRID_CELL_STRIDE..],
        expected.as_slice(),
        "every mask must describe exactly the retained neighboring registrations"
    );
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn visibility_masks_build_and_clear_from_retained_production_grids() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    const MASK_AGENTS: u32 = u32::BITS;
    let kernel = make_kernel(CENTRE_SIDE, CENTRE_SIDE, MASK_AGENTS, 1);
    let agents: Vec<_> = (0..MASK_AGENTS)
        .map(|agent| {
            Vec3::new(
                (agent as f32 - MASK_AGENTS as f32 * 0.5) * GRID_CELL_SIZE,
                1.0,
                0.0,
            )
        })
        .collect();
    upload_scene(&kernel, &agents, &[Vec3::ZERO], 0.0);
    kernel.write_agent_physics_fields(0, &[(P_ALIVE, 0.0)]);
    dispatch_mask_builder(&kernel, "masks_from_production_grid");
    assert_grid_masks(&kernel, usize::try_from(MASK_AGENTS - 1)?)?;

    // Moving every agent to a single cell exercises complete capacity, bit31,
    // and clearing masks whose former registrations are now elsewhere.
    upload_scene(
        &kernel,
        &vec![Vec3::Y; usize::try_from(MASK_AGENTS)?],
        &[Vec3::ZERO],
        0.0,
    );
    dispatch_mask_builder(&kernel, "masks_from_production_grid");
    assert_grid_masks(&kernel, usize::try_from(MASK_AGENTS)?)?;
    Ok(())
}

fn upload_grid(kernel: &GpuKernel, positions: &[Vec3], stride: usize, target: &wgpu::Buffer) {
    let width = grid_width(kernel.world_config.world_size);
    let offset = i32::try_from(width / 2).unwrap();
    let mut grid = vec![
        0_u32;
        width
            .checked_mul(width)
            .unwrap()
            .checked_mul(stride)
            .unwrap()
    ];
    for (index, position) in positions.iter().enumerate() {
        let cell_x = (position.x / GRID_CELL_SIZE).floor() as i32 + offset;
        let cell_z = (position.z / GRID_CELL_SIZE).floor() as i32 + offset;
        let (Ok(cell_x), Ok(cell_z)) = (usize::try_from(cell_x), usize::try_from(cell_z)) else {
            continue;
        };
        if cell_x >= width || cell_z >= width {
            continue;
        }
        let base = (cell_x * width + cell_z) * stride;
        let count = usize::try_from(grid[base]).unwrap();
        // Production grids retain only the first capacity entries in a cell.
        if count < stride - 1 {
            grid[base + 1 + count] = u32::try_from(index).unwrap();
            grid[base] += 1;
        }
    }
    kernel
        .queue
        .write_buffer(target, 0, bytemuck::cast_slice(&grid));
}

fn upload_scene(kernel: &GpuKernel, agents: &[Vec3], food: &[Vec3], terrain_height: f32) {
    assert_eq!(agents.len(), usize::try_from(kernel.agent_count).unwrap());
    assert_eq!(food.len(), kernel.food_count);
    let brain = BrainConfig::default();
    let physics: Vec<_> = agents
        .iter()
        .map(|&position| {
            (
                position,
                FULL_METER,
                FULL_METER,
                brain.memory_capacity,
                brain.processing_slots,
            )
        })
        .collect();
    kernel.upload_agents(&physics);
    let food_positions: Vec<_> = food
        .iter()
        .map(|position| (position.x, position.y, position.z))
        .collect();
    kernel.upload_world(
        &vec![terrain_height; TERRAIN_SIDE * TERRAIN_SIDE],
        &vec![0; BIOME_GRID_RES * BIOME_GRID_RES],
        &food_positions,
        &vec![false; food.len()],
        &vec![0.0; food.len()],
    );
    upload_grid(
        kernel,
        agents,
        AGENT_GRID_CELL_STRIDE,
        &kernel.agent_grid_buffer,
    );
    upload_grid(
        kernel,
        food,
        FOOD_GRID_CELL_STRIDE,
        &kernel.food_grid_buffer,
    );
}

pub(super) fn read_buffer(
    kernel: &GpuKernel,
    buffer: &wgpu::Buffer,
    size: u64,
) -> TestResult<Vec<u8>> {
    let staging = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vision_probe_readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    kernel.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    kernel.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    receiver.recv()??;
    let bytes = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(bytes)
}

fn record_dispatch<'pass>(
    kernel: &'pass GpuKernel,
    pipeline: &'pass VisionPipeline,
    pass: &mut wgpu::ComputePass<'pass>,
) {
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, &kernel.bind_groups[kernel.active_config_index], &[]);
    pass.dispatch_workgroups(pipeline.workgroups, 1, 1);
}

fn run_vision(kernel: &GpuKernel, pipeline: &VisionPipeline) -> TestResult<Vec<u32>> {
    let values = usize::try_from(kernel.agent_count)
        .unwrap()
        .checked_mul(kernel.layout.sensory_stride)
        .unwrap();
    kernel.queue.write_buffer(
        &kernel.sensory_buffer,
        0,
        bytemuck::cast_slice(&vec![SENTINEL; values]),
    );
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        record_dispatch(kernel, pipeline, &mut pass);
    }
    kernel.queue.submit([encoder.finish()]);
    let byte_count = u64::try_from(values.checked_mul(std::mem::size_of::<f32>()).unwrap())?;
    let bytes = read_buffer(kernel, &kernel.sensory_buffer, byte_count)?;
    Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
}

fn assert_parity(
    kernel: &GpuKernel,
    serial: &VisionPipeline,
    parallel: &[VisionPipeline],
    label: &str,
) -> TestResult<Vec<u32>> {
    build_retained_grid_masks(kernel);
    let expected = run_vision(kernel, serial)?;
    for pipeline in parallel {
        let group = pipeline.rays_per_group;
        let masks = pipeline.options.agent_masks;
        let objects = pipeline.options.object_queries;
        let scent = pipeline.options.parallel_scent;
        let actual = run_vision(kernel, pipeline)?;
        assert_eq!(actual.len(), expected.len());
        if let Some((slot, (actual, expected))) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!("{label}, rays/group={group}, masks={masks}, objects={objects}, scent={scent}, agent={}, sensory slot={}: parallel={:#010x} ({}) serial={:#010x} ({})",
                slot / kernel.layout.sensory_stride, slot % kernel.layout.sensory_stride,
                actual, f32::from_bits(*actual), expected, f32::from_bits(*expected));
        }
    }
    Ok(expected)
}

fn assert_ray(bits: &[u32], layout: &BrainLayout, ray: usize, color: [f32; 4], depth: f32) {
    assert_eq!(
        &bits[ray * color.len()..(ray + 1) * color.len()],
        &color.map(f32::to_bits)
    );
    assert_eq!(bits[layout.vision_color_count + ray], depth.to_bits());
}

fn pipeline_pair(
    kernel: &GpuKernel,
    include_senses: bool,
) -> (VisionPipeline, Vec<VisionPipeline>) {
    let serial = make_pipeline(
        kernel,
        vision_rays_per_workgroup(
            kernel.agent_count,
            kernel.layout.vision_width * kernel.layout.vision_height,
        ),
        include_senses,
        VisionOptions::default(),
    );
    let mut parallel = Vec::new();
    for width in PARALLEL_GROUP_WIDTHS {
        for masks in [false, true] {
            for scent in [false, true]
                .into_iter()
                .filter(|&scent| !scent || include_senses)
            {
                parallel.push(make_pipeline(
                    kernel,
                    width,
                    include_senses,
                    VisionOptions {
                        parallel_steps: true,
                        agent_masks: masks,
                        parallel_scent: scent,
                        ..VisionOptions::default()
                    },
                ));
            }
        }
    }
    /// The object cache has a fixed capacity of 256 food positions.
    const OBJECT_FOOD_CAPACITY: usize = 256;
    if kernel.agent_count <= u32::BITS && kernel.food_count <= OBJECT_FOOD_CAPACITY {
        let width = *PARALLEL_GROUP_WIDTHS.last().unwrap();
        for masks in [false, true] {
            for scent in [false, true]
                .into_iter()
                .filter(|&scent| !scent || include_senses)
            {
                parallel.push(make_pipeline(
                    kernel,
                    width,
                    include_senses,
                    VisionOptions {
                        parallel_steps: true,
                        agent_masks: masks,
                        object_queries: true,
                        parallel_scent: scent,
                    },
                ));
            }
        }
    }
    // Packed serial waves distinguish useful step parallelism from simply
    // using more of the hardware lanes than the production small-world shape.
    const PACKED_SERIAL_WIDTHS: [u32; 2] = [32, 64];
    parallel.extend(
        PACKED_SERIAL_WIDTHS
            .map(|width| make_pipeline(kernel, width, include_senses, VisionOptions::default())),
    );
    (serial, parallel)
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn parallel_vision_preserves_hit_order_sky_exit_and_dead_agents() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let kernel = make_kernel(CENTRE_SIDE, CENTRE_SIDE, 2, 1);
    let (serial, parallel) = pipeline_pair(&kernel, true);
    let observer = Vec3::new(0.0, 1.0, 0.0);
    let sample = Vec3::new(0.0, 1.0, FIRST_SAMPLE);
    let far = Vec3::new(MAX_DISTANCE, MAX_DISTANCE, MAX_DISTANCE);
    let same_step_depth = FIRST_SAMPLE / MAX_DISTANCE;
    for (label, other, food, expected) in [
        (
            "food precedes agent and terrain",
            sample,
            sample,
            FOOD_COLOR,
        ),
        ("agent precedes terrain", sample, far, AGENT_COLOR),
        (
            "near terrain precedes later food",
            far,
            observer + Vec3::Z * (FIRST_SAMPLE * CENTRE_SIDE as f32),
            TERRAIN_COLOR,
        ),
    ] {
        upload_scene(&kernel, &[observer, other], &[food], 2.0);
        let bits = assert_parity(&kernel, &serial, &parallel, label)?;
        assert_ray(&bits, &kernel.layout, CENTRE_RAY, expected, same_step_depth);
    }

    upload_scene(&kernel, &[observer, sample], &[sample], 2.0);
    kernel
        .queue
        .write_buffer(&kernel.food_flags_buffer, 0, bytemuck::cast_slice(&[1_u32]));
    let bits = assert_parity(&kernel, &serial, &parallel, "consumed food exposes agent")?;
    assert_ray(
        &bits,
        &kernel.layout,
        CENTRE_RAY,
        AGENT_COLOR,
        same_step_depth,
    );

    upload_scene(
        &kernel,
        &[observer, far],
        &[Vec3::new(1.0, 1.0, FIRST_SAMPLE)],
        0.0,
    );
    let bits = assert_parity(&kernel, &serial, &parallel, "strict food radius")?;
    assert_ray(&bits, &kernel.layout, CENTRE_RAY, SKY_COLOR, 1.0);

    const AGENT_RADIUS: f32 = 1.5;
    upload_scene(
        &kernel,
        &[observer, Vec3::new(AGENT_RADIUS, 1.0, FIRST_SAMPLE)],
        &[far],
        0.0,
    );
    let bits = assert_parity(&kernel, &serial, &parallel, "strict agent radius")?;
    assert_ray(&bits, &kernel.layout, CENTRE_RAY, SKY_COLOR, 1.0);

    upload_scene(&kernel, &[observer, sample], &[far], 0.0);
    kernel.write_agent_physics_fields(1, &[(P_ALIVE, 0.0)]);
    let bits = assert_parity(&kernel, &serial, &parallel, "dead target ignored")?;
    assert_ray(&bits, &kernel.layout, CENTRE_RAY, SKY_COLOR, 1.0);
    assert!(
        bits[kernel.layout.sensory_stride..]
            .iter()
            .all(|&value| value == SENTINEL.to_bits()),
        "dead observer must leave every sensory slot untouched"
    );

    // The object's registered cell is adjacent to its actual post-collision cell.
    upload_scene(&kernel, &[observer, sample], &[far], 0.0);
    upload_grid(
        &kernel,
        &[observer, sample - Vec3::X * GRID_CELL_SIZE],
        AGENT_GRID_CELL_STRIDE,
        &kernel.agent_grid_buffer,
    );
    let bits = assert_parity(&kernel, &serial, &parallel, "stale agent registration")?;
    assert_ray(
        &bits,
        &kernel.layout,
        CENTRE_RAY,
        AGENT_COLOR,
        same_step_depth,
    );

    // A sample outside the allocated mask grid still has an in-range neighbor
    // cell. The retained registration remains visible through that neighbor.
    let grid_edge = (grid_width(kernel.world_config.world_size) / 2) as f32 * GRID_CELL_SIZE;
    let edge_observer = Vec3::new(grid_edge - FIRST_SAMPLE, 1.0, 0.0);
    let edge_target = Vec3::new(grid_edge, 1.0, 0.0);
    upload_scene(&kernel, &[edge_observer, edge_target], &[far], 0.0);
    kernel.write_agent_physics_fields(0, &[(P_FACING_X, 1.0), (P_FACING_Z, 0.0)]);
    upload_grid(
        &kernel,
        &[edge_observer, edge_observer],
        AGENT_GRID_CELL_STRIDE,
        &kernel.agent_grid_buffer,
    );
    let bits = assert_parity(&kernel, &serial, &parallel, "sample outside mask grid")?;
    assert_ray(
        &bits,
        &kernel.layout,
        CENTRE_RAY,
        AGENT_COLOR,
        same_step_depth,
    );

    const UPWARD_FOV: f32 = 120.0;
    const LATE_SAMPLE: f32 = 8.0;
    let angle = (UPWARD_FOV * 0.5).to_radians();
    let upward = Vec3::new(0.0, angle.sin(), angle.cos());
    upload_scene(
        &kernel,
        &[observer, far],
        &[observer + upward * (FIRST_SAMPLE * LATE_SAMPLE)],
        0.0,
    );
    kernel.write_agent_heritable_config(
        0,
        &BrainConfig {
            vertical_fov_degrees: UPWARD_FOV,
            ..BrainConfig::default()
        },
    );
    let bits = assert_parity(&kernel, &serial, &parallel, "sky exit precedes later food")?;
    assert_ray(&bits, &kernel.layout, UPPER_RAY, SKY_COLOR, 1.0);
    Ok(())
}

pub(super) fn upload_random_scene(
    kernel: &GpuKernel,
    scene_index: u64,
    terrain: bool,
    sparse: bool,
) {
    const SCENE_HALF_WIDTH: f32 = 28.0;
    const FOOD_HEIGHT_LIMIT: f32 = 8.0;
    let half_width = if sparse {
        kernel.world_config.world_size * 0.5 - MAX_DISTANCE
    } else {
        SCENE_HALF_WIDTH
    };
    let mut rng = StdRng::seed_from_u64(SCENE_SEED + scene_index);
    let agents: Vec<_> = (0..kernel.agent_count)
        .map(|_| {
            Vec3::new(
                rng.random_range(-half_width..half_width),
                1.0,
                rng.random_range(-half_width..half_width),
            )
        })
        .collect();
    let food: Vec<_> = (0..kernel.food_count)
        .map(|_| {
            Vec3::new(
                rng.random_range(-half_width..half_width),
                rng.random_range(0.0..FOOD_HEIGHT_LIMIT),
                rng.random_range(-half_width..half_width),
            )
        })
        .collect();
    upload_scene(kernel, &agents, &food, 0.0);
    for agent in 0..kernel.agent_count {
        let facing = rng.random_range(0.0..std::f32::consts::TAU);
        kernel.write_agent_physics_fields(
            agent,
            &[(P_FACING_X, facing.sin()), (P_FACING_Z, facing.cos())],
        );
    }
    if terrain {
        const TERRAIN_AMPLITUDE: f32 = 4.0;
        let heights: Vec<f32> = (0..TERRAIN_SIDE * TERRAIN_SIDE)
            .map(|_| rng.random_range(-TERRAIN_AMPLITUDE..TERRAIN_AMPLITUDE))
            .collect();
        const BIOME_TYPES: u32 = 3;
        let biomes: Vec<u32> = (0..BIOME_GRID_RES * BIOME_GRID_RES)
            .map(|_| rng.random_range(0..BIOME_TYPES))
            .collect();
        kernel
            .queue
            .write_buffer(&kernel.heightmap_buffer, 0, bytemuck::cast_slice(&heights));
        kernel
            .queue
            .write_buffer(&kernel.biome_buffer, 0, bytemuck::cast_slice(&biomes));
    }
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn parallel_vision_matches_all_sensory_bits_across_fields_and_scenes() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    for (width, height) in VISION_DIMENSIONS {
        let kernel = make_kernel(width, height, DEFAULT_POPULATION, RANDOM_FOOD_COUNT);
        let (serial, parallel) = pipeline_pair(&kernel, true);
        for (scene, &(horizontal, vertical)) in FOV_PAIRS.iter().enumerate() {
            upload_random_scene(&kernel, u64::try_from(scene)?, true, false);
            for agent in 0..kernel.agent_count {
                kernel.write_agent_heritable_config(
                    agent,
                    &BrainConfig {
                        horizontal_fov_degrees: horizontal,
                        vertical_fov_degrees: vertical,
                        ..BrainConfig::default()
                    },
                );
            }
            assert_parity(
                &kernel,
                &serial,
                &parallel,
                &format!("dimensions={width}x{height}, scene={scene}, fov={horizontal}/{vertical}"),
            )?;
            for agent in 0..kernel.agent_count {
                let (horizontal, vertical) =
                    FOV_PAIRS[(usize::try_from(agent)? + scene) % FOV_PAIRS.len()];
                kernel.write_agent_heritable_config(
                    agent,
                    &BrainConfig {
                        horizontal_fov_degrees: horizontal,
                        vertical_fov_degrees: vertical,
                        ..BrainConfig::default()
                    },
                );
            }
            const CONSUMED_PERIOD: usize = 3;
            let flags: Vec<u32> = (0..kernel.food_count)
                .map(|food| u32::from((food + scene) % CONSUMED_PERIOD == 0))
                .collect();
            kernel
                .queue
                .write_buffer(&kernel.food_flags_buffer, 0, bytemuck::cast_slice(&flags));
            assert_parity(
                &kernel,
                &serial,
                &parallel,
                &format!(
                    "dimensions={width}x{height}, scene={scene}, per-agent fov and consumed food"
                ),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn parallel_vision_matches_at_agent_mask_population_boundary() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    const POPULATIONS: [u32; 3] = [u32::BITS - 1, u32::BITS, u32::BITS + 1];
    for population in POPULATIONS {
        let kernel = make_kernel(CENTRE_SIDE, CENTRE_SIDE, population, 1);
        let (serial, parallel) = pipeline_pair(&kernel, true);
        let far = Vec3::splat(MAX_DISTANCE);
        let mut agents = vec![far; usize::try_from(population)?];
        agents[0] = Vec3::Y;
        agents[usize::try_from(population - 1)?] = Vec3::new(0.0, 1.0, FIRST_SAMPLE);
        upload_scene(&kernel, &agents, &[far], 0.0);
        let bits = assert_parity(
            &kernel,
            &serial,
            &parallel,
            &format!("population={population}, final agent alive"),
        )?;
        assert_ray(
            &bits,
            &kernel.layout,
            CENTRE_RAY,
            AGENT_COLOR,
            FIRST_SAMPLE / MAX_DISTANCE,
        );
        kernel.write_agent_physics_fields(population - 1, &[(P_ALIVE, 0.0)]);
        let bits = assert_parity(
            &kernel,
            &serial,
            &parallel,
            &format!("population={population}, final agent dead"),
        )?;
        assert_ray(&bits, &kernel.layout, CENTRE_RAY, SKY_COLOR, 1.0);
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn object_queries_respect_retained_food_and_late_agent_eligibility() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    let kernel = make_kernel(CENTRE_SIDE, CENTRE_SIDE, 2, FOOD_GRID_MAX_PER_CELL + 1);
    let (serial, parallel) = pipeline_pair(&kernel, true);
    let observer = Vec3::Y;
    let sample = Vec3::new(0.0, 1.0, FIRST_SAMPLE);
    let far_agent = Vec3::splat(MAX_DISTANCE);
    /// All retained food shares the sample's grid cell but is outside the ray.
    const MISSED_FOOD: Vec3 = Vec3::new(6.0, 6.0, 6.0);
    let mut food = vec![MISSED_FOOD; FOOD_GRID_MAX_PER_CELL + 1];
    food[FOOD_GRID_MAX_PER_CELL] = sample;
    for retained in [false, true] {
        if retained {
            food.swap(0, FOOD_GRID_MAX_PER_CELL);
        }
        upload_scene(&kernel, &[observer, far_agent], &food, 0.0);
        // Production insertion increments the count even when storage is full.
        let width = grid_width(kernel.world_config.world_size);
        let cell = width / 2 * width + width / 2;
        let offset = u64::try_from(cell * FOOD_GRID_CELL_STRIDE * std::mem::size_of::<u32>())?;
        kernel.queue.write_buffer(
            &kernel.food_grid_buffer,
            offset,
            bytemuck::cast_slice(&[u32::try_from(food.len())?]),
        );
        let bits = assert_parity(
            &kernel,
            &serial,
            &parallel,
            &format!("food overflow target retained={retained}"),
        )?;
        assert_ray(
            &bits,
            &kernel.layout,
            CENTRE_RAY,
            if retained { FOOD_COLOR } else { SKY_COLOR },
            if retained {
                FIRST_SAMPLE / MAX_DISTANCE
            } else {
                1.0
            },
        );
    }

    let target = Vec3::new(0.0, 1.0, GRID_CELL_SIZE);
    upload_scene(
        &kernel,
        &[observer, target],
        &vec![far_agent; food.len()],
        0.0,
    );
    upload_grid(
        &kernel,
        &[observer, Vec3::new(0.0, 1.0, GRID_CELL_SIZE * 2.0)],
        AGENT_GRID_CELL_STRIDE,
        &kernel.agent_grid_buffer,
    );
    let bits = assert_parity(
        &kernel,
        &serial,
        &parallel,
        "earlier geometric hit ineligible, later sample eligible",
    )?;
    /// Sample seven is the first centre in the neighboring eight-unit cell.
    const FIRST_REGISTERED_SAMPLE: f32 = 7.0;
    assert_ray(
        &bits,
        &kernel.layout,
        CENTRE_RAY,
        AGENT_COLOR,
        (FIRST_REGISTERED_SAMPLE * FIRST_SAMPLE) / MAX_DISTANCE,
    );
    Ok(())
}

#[test]
#[ignore = "requires a GPU; run explicitly with --ignored --nocapture"]
fn parallel_scent_preserves_order_chunks_range_and_heritable_strength() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    /// Counts cover empty worlds and one/two incomplete 256-item chunks.
    const FOOD_COUNTS: [usize; 4] = [0, 1, 257, 513];
    /// Small strengths expose accumulation changes without saturating the output.
    const STRENGTHS: [f32; 6] = [0.0, 0.001, 0.05, 1.0, 5.0, 500.0];
    let just_inside = f32::from_bits(SCENT_RANGE.to_bits() - 1);
    let just_outside = f32::from_bits(SCENT_RANGE.to_bits() + 1);
    for food_count in FOOD_COUNTS {
        let kernel = make_kernel(
            CENTRE_SIDE,
            CENTRE_SIDE,
            u32::try_from(STRENGTHS.len())?,
            food_count,
        );
        let (serial, parallel) = pipeline_pair(&kernel, true);
        for distance in [just_inside, SCENT_RANGE, just_outside] {
            let food: Vec<_> = (0..food_count)
                .map(|item| {
                    if item == 0 {
                        Vec3::new(-NOSTRIL_SIDE_OFFSET - distance, 0.0, NOSTRIL_FORWARD_OFFSET)
                    } else {
                        /// Vary contribution magnitude and separate the two nostrils.
                        const POSITION_PERIOD: usize = 29;
                        let horizontal = (item % POSITION_PERIOD) as f32 - SCENT_RANGE * 0.5;
                        let depth =
                            ((item / POSITION_PERIOD) % POSITION_PERIOD) as f32 - SCENT_RANGE * 0.5;
                        Vec3::new(horizontal, 0.0, depth)
                    }
                })
                .collect();
            upload_scene(&kernel, &vec![Vec3::Y; STRENGTHS.len()], &food, 0.0);
            for (agent, &strength) in STRENGTHS.iter().enumerate() {
                kernel.write_agent_heritable_config(
                    u32::try_from(agent)?,
                    &BrainConfig {
                        smell_strength: strength,
                        ..BrainConfig::default()
                    },
                );
            }
            /// Mix live and consumed items across every chunk, keeping item zero.
            const CONSUMED_PERIOD: usize = 3;
            let flags: Vec<u32> = (0..food_count)
                .map(|item| u32::from(item % CONSUMED_PERIOD == 1))
                .collect();
            kernel
                .queue
                .write_buffer(&kernel.food_flags_buffer, 0, bytemuck::cast_slice(&flags));
            let bits = assert_parity(
                &kernel,
                &serial,
                &parallel,
                &format!("food={food_count}, first distance={distance:?}"),
            )?;
            let scent_start = kernel.layout.sensory_stride - SCENT_CHANNELS;
            for (agent, values) in bits.chunks_exact(kernel.layout.sensory_stride).enumerate() {
                let scent = &values[scent_start..];
                if food_count == 0
                    || (food_count == 1 && distance >= SCENT_RANGE)
                    || STRENGTHS[agent] == 0.0
                {
                    assert!(
                        scent.iter().all(|&value| f32::from_bits(value) == 0.0),
                        "empty, out-of-range, or disabled scent must be zero"
                    );
                } else {
                    assert!(
                        f32::from_bits(scent[0]) > 0.0,
                        "live in-range food must produce scent"
                    );
                }
            }
        }
    }
    Ok(())
}

fn time_vision(kernel: &GpuKernel, pipeline: &VisionPipeline) -> TestResult<f64> {
    const QUERY_COUNT: u32 = 2;
    let queries = kernel.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("vision_timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: QUERY_COUNT,
    });
    let bytes = u64::from(QUERY_COUNT) * u64::try_from(std::mem::size_of::<u64>())?;
    let resolved = kernel.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vision_timestamp_results"),
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = kernel.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timed_vision_dispatches"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        for _ in 0..TIMED_DISPATCHES {
            record_dispatch(kernel, pipeline, &mut pass);
        }
    }
    encoder.resolve_query_set(&queries, 0..QUERY_COUNT, &resolved, 0);
    kernel.queue.submit([encoder.finish()]);
    let result = read_buffer(kernel, &resolved, bytes)?;
    let timestamps: &[u64] = bytemuck::cast_slice(&result);
    assert!(
        timestamps[1] > timestamps[0],
        "GPU timestamp interval must be positive"
    );
    Ok(
        (timestamps[1] - timestamps[0]) as f64 * f64::from(kernel.queue.get_timestamp_period())
            / f64::from(TIMED_DISPATCHES),
    )
}

#[test]
#[ignore = "GPU benchmark; run explicitly in release mode with --ignored --nocapture"]
fn benchmark_parallel_vision_gpu_timestamps() -> TestResult {
    let _vulkan = vulkan_gate::enter();
    const POPULATIONS: [u32; 3] = [1, DEFAULT_POPULATION, 40];
    for population in POPULATIONS {
        let kernel = make_kernel(
            BrainConfig::default().vision_width,
            BrainConfig::default().vision_height,
            population,
            RANDOM_FOOD_COUNT,
        );
        assert!(
            kernel
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY),
            "this benchmark requires an adapter with timestamp queries"
        );
        let complete = pipeline_pair(&kernel, true);
        let rays_only = pipeline_pair(&kernel, false);
        for (scene, terrain, sparse) in [
            ("dense_flat", false, false),
            ("dense_terrain", true, false),
            ("sparse_flat", false, true),
            ("synthetic_long_open", false, true),
        ] {
            upload_random_scene(&kernel, 0, terrain, sparse);
            if scene == "synthetic_long_open" {
                // Food and terrain stay below all samples, other observers
                // stay beyond the ray range, and every ray points downward.
                const HIGH_OBSERVER: f32 = 100.0;
                /// Eight units of separation beyond a ray's range also keeps
                /// every other agent's hit sphere out of reach.
                const OBSERVER_SPACING: f32 = MAX_DISTANCE + GRID_CELL_SIZE;
                let side = (kernel.agent_count as f32).sqrt().ceil() as u32;
                let half_span = (side - 1) as f32 * OBSERVER_SPACING * 0.5;
                let observers: Vec<_> = (0..kernel.agent_count)
                    .map(|agent| {
                        Vec3::new(
                            (agent % side) as f32 * OBSERVER_SPACING - half_span,
                            HIGH_OBSERVER,
                            (agent / side) as f32 * OBSERVER_SPACING - half_span,
                        )
                    })
                    .collect();
                for agent in 0..kernel.agent_count {
                    let observer = observers[usize::try_from(agent)?];
                    kernel.write_agent_physics_fields(
                        agent,
                        &[
                            (P_POS_X, observer.x),
                            (P_POS_Y, observer.y),
                            (P_POS_Z, observer.z),
                            (P_FACING_Y, -1.0),
                        ],
                    );
                }
                upload_grid(
                    &kernel,
                    &observers,
                    AGENT_GRID_CELL_STRIDE,
                    &kernel.agent_grid_buffer,
                );
            }
            for (scope, (serial, parallel)) in
                [("vision_and_senses", &complete), ("rays_only", &rays_only)]
            {
                let bits = assert_parity(&kernel, serial, parallel, scene)?;
                if scene == "synthetic_long_open" {
                    for agent in bits.chunks_exact(kernel.layout.sensory_stride) {
                        for ray in 0..kernel.layout.vision_depth_count {
                            assert_ray(agent, &kernel.layout, ray, SKY_COLOR, 1.0);
                        }
                    }
                }
                benchmark_pipelines(&kernel, serial, parallel, scene, scope)?;
            }
        }
    }
    Ok(())
}

fn benchmark_pipelines(
    kernel: &GpuKernel,
    serial: &VisionPipeline,
    parallel: &[VisionPipeline],
    scene: &str,
    scope: &str,
) -> TestResult {
    let all_pipelines: Vec<_> = std::iter::once(serial).chain(parallel.iter()).collect();
    for pipeline in &all_pipelines {
        time_vision(kernel, pipeline)?;
    }
    let mut timings = vec![Vec::with_capacity(TIMING_ROUNDS); all_pipelines.len()];
    // Rotate the measurement order to reduce thermal/order bias.
    for round in 0..TIMING_ROUNDS {
        for offset in 0..all_pipelines.len() {
            let index = (round + offset) % all_pipelines.len();
            timings[index].push(time_vision(kernel, all_pipelines[index])?);
        }
    }
    for samples in &mut timings {
        samples.sort_by(f64::total_cmp);
    }
    let baseline = timings[0][TIMING_ROUNDS / 2];
    let best_serial = parallel
        .iter()
        .zip(&timings[1..])
        .filter(|(pipeline, _)| !pipeline.options.parallel_steps)
        .map(|(_, samples)| samples[TIMING_ROUNDS / 2])
        .fold(baseline, f64::min);
    let population = kernel.agent_count;
    println!("VISION_GPU_NS agents={population} scene={scene} scope={scope} serial_median={baseline:.1} serial_min={:.1} best_serial_median={best_serial:.1}", timings[0][0]);
    for (pipeline, samples) in parallel.iter().zip(&timings[1..]) {
        let median = samples[TIMING_ROUNDS / 2];
        let group = pipeline.rays_per_group;
        let masks = pipeline.options.agent_masks;
        let parallel_steps = pipeline.options.parallel_steps;
        let objects = pipeline.options.object_queries;
        let scent = pipeline.options.parallel_scent;
        println!("VISION_GPU_NS agents={population} scene={scene} scope={scope} parallel_steps={parallel_steps} rays_per_group={group} masks={masks} objects={objects} scent={scent} median={median:.1} min={:.1} speedup={:.3} best_serial_speedup={:.3}", samples[0], baseline / median, best_serial / median);
    }
    Ok(())
}
