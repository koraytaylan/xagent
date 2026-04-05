//! GPU compute pipeline for vision raycast operations.
//!
//! Runs GPU-accelerated ray marching for each agent's sensory input.
//! Output is RGBA+depth per ray, identical to CPU `sample_vision_positions`.
//!
//! # Architecture
//!
//! The module creates its own wgpu device and queue, independent of the renderer.
//! Buffer capacities are fixed when the pipeline is constructed.
//!
//! # Performance
//!
//! For 50 agents × 48 rays each, one GPU dispatch is faster than CPU rayon.
//! The caller uses `ComputeBackend::probe()` at startup to decide whether
//! GPU is available.

use glam::Vec3;
use wgpu::util::DeviceExt;

// ── Vision Raycast Shader ────────────────────────────────────────────

pub const VISION_SHADER: &str = r#"
struct VisionParams {
    num_agents: u32,
    num_rays: u32,       // 48 (8x6)
    max_dist: f32,       // 50.0
    step_size: f32,      // 1.0
    world_size: f32,
    terrain_vps: u32,    // verts per side
    terrain_size: f32,
    num_food: u32,
    num_agents_total: u32,
    agent_radius_sq: f32, // 2.25
    food_radius_sq: f32,  // 1.0
    biome_res: u32,      // biome grid resolution (e.g. 256)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: VisionParams;
@group(0) @binding(1) var<storage, read> terrain_heights: array<f32>;
@group(0) @binding(2) var<storage, read> biome_types: array<u32>;
@group(0) @binding(3) var<storage, read> food_positions: array<f32>;  // [x, y, z, consumed] per food
@group(0) @binding(4) var<storage, read> agent_positions: array<f32>; // [x, y, z, alive] per agent
@group(0) @binding(5) var<storage, read> ray_origins: array<f32>;     // [x, y, z] per agent
@group(0) @binding(6) var<storage, read> ray_dirs: array<f32>;        // [x, y, z] per ray per agent
@group(0) @binding(7) var<storage, read_write> vision_output: array<f32>; // [r, g, b, a, depth] per ray

fn terrain_height(x: f32, z: f32) -> f32 {
    let half = params.terrain_size / 2.0;
    let step = params.terrain_size / f32(params.terrain_vps - 1u);
    let vps = params.terrain_vps;

    let gx = clamp((x + half) / step, 0.0, f32(vps - 1u));
    let gz = clamp((z + half) / step, 0.0, f32(vps - 1u));

    let ix = min(u32(floor(gx)), vps - 2u);
    let iz = min(u32(floor(gz)), vps - 2u);
    let fx = gx - f32(ix);
    let fz = gz - f32(iz);

    let h00 = terrain_heights[iz * vps + ix];
    let h10 = terrain_heights[iz * vps + ix + 1u];
    let h01 = terrain_heights[(iz + 1u) * vps + ix];
    let h11 = terrain_heights[(iz + 1u) * vps + ix + 1u];

    let h0 = h00 + (h10 - h00) * fx;
    let h1 = h01 + (h11 - h01) * fx;
    return h0 + (h1 - h0) * fz;
}

fn biome_color(x: f32, z: f32) -> vec4<f32> {
    let half = params.world_size / 2.0;
    let inv_cell = f32(params.biome_res) / params.world_size;
    // Clamp in float space before u32 conversion to avoid negative→u32 issues
    // when rays extend beyond world bounds (x/z < -half).
    let gx = clamp((x + half) * inv_cell, 0.0, f32(params.biome_res - 1u));
    let gz = clamp((z + half) * inv_cell, 0.0, f32(params.biome_res - 1u));
    let col = u32(floor(gx));
    let row = u32(floor(gz));
    let biome = biome_types[row * params.biome_res + col];
    if biome == 0u { // FoodRich
        return vec4(0.15, 0.50, 0.10, 1.0);
    } else if biome == 1u { // Barren
        return vec4(0.50, 0.40, 0.20, 1.0);
    } else { // Danger
        return vec4(0.60, 0.20, 0.10, 1.0);
    }
}

@compute @workgroup_size(48, 1, 1)
fn vision_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_idx = gid.x;
    let agent_idx = gid.y;

    if ray_idx >= params.num_rays || agent_idx >= params.num_agents {
        return;
    }

    let origin_base = agent_idx * 3u;
    let origin = vec3(
        ray_origins[origin_base],
        ray_origins[origin_base + 1u],
        ray_origins[origin_base + 2u],
    );

    let dir_base = (agent_idx * params.num_rays + ray_idx) * 3u;
    let dir = vec3(
        ray_dirs[dir_base],
        ray_dirs[dir_base + 1u],
        ray_dirs[dir_base + 2u],
    );

    let sky = vec4(0.53, 0.81, 0.92, 1.0);
    let agent_color = vec4(0.9, 0.2, 0.6, 1.0);
    let food_color = vec4(0.70, 0.95, 0.20, 1.0);

    // Early-out for upward rays above terrain
    if dir.y > 0.3 {
        let origin_h = terrain_height(origin.x, origin.z);
        if origin.y > origin_h {
            let out_base = (agent_idx * params.num_rays + ray_idx) * 5u;
            vision_output[out_base] = sky.x;
            vision_output[out_base + 1u] = sky.y;
            vision_output[out_base + 2u] = sky.z;
            vision_output[out_base + 3u] = sky.w;
            vision_output[out_base + 4u] = 1.0; // max depth normalized
            return;
        }
    }

    var t: f32 = 0.0;
    var hit_color = sky;
    var hit_depth = params.max_dist;

    while t < params.max_dist {
        let p = origin + dir * t;

        // Check food items (brute force - N_food is small)
        for (var fi = 0u; fi < params.num_food; fi = fi + 1u) {
            let fb = fi * 4u;
            if food_positions[fb + 3u] > 0.5 { continue; } // consumed
            let diff = p - vec3(food_positions[fb], food_positions[fb + 1u], food_positions[fb + 2u]);
            if dot(diff, diff) < params.food_radius_sq {
                hit_color = food_color;
                hit_depth = t;
                let out_base = (agent_idx * params.num_rays + ray_idx) * 5u;
                vision_output[out_base] = hit_color.x;
                vision_output[out_base + 1u] = hit_color.y;
                vision_output[out_base + 2u] = hit_color.z;
                vision_output[out_base + 3u] = hit_color.w;
                vision_output[out_base + 4u] = hit_depth / params.max_dist;
                return;
            }
        }

        // Check other agents
        for (var ai = 0u; ai < params.num_agents_total; ai = ai + 1u) {
            if ai == agent_idx { continue; }
            let ab = ai * 4u;
            if agent_positions[ab + 3u] < 0.5 { continue; } // dead
            let diff = p - vec3(agent_positions[ab], agent_positions[ab + 1u], agent_positions[ab + 2u]);
            if dot(diff, diff) < params.agent_radius_sq {
                hit_color = agent_color;
                hit_depth = t;
                let out_base = (agent_idx * params.num_rays + ray_idx) * 5u;
                vision_output[out_base] = hit_color.x;
                vision_output[out_base + 1u] = hit_color.y;
                vision_output[out_base + 2u] = hit_color.z;
                vision_output[out_base + 3u] = hit_color.w;
                vision_output[out_base + 4u] = hit_depth / params.max_dist;
                return;
            }
        }

        // Check terrain
        let gh = terrain_height(p.x, p.z);
        if p.y <= gh {
            hit_color = biome_color(p.x, p.z);
            hit_depth = t;
            let out_base = (agent_idx * params.num_rays + ray_idx) * 5u;
            vision_output[out_base] = hit_color.x;
            vision_output[out_base + 1u] = hit_color.y;
            vision_output[out_base + 2u] = hit_color.z;
            vision_output[out_base + 3u] = hit_color.w;
            vision_output[out_base + 4u] = hit_depth / params.max_dist;
            return;
        }

        t = t + params.step_size;
    }

    // No hit - sky
    let out_base = (agent_idx * params.num_rays + ray_idx) * 5u;
    vision_output[out_base] = sky.x;
    vision_output[out_base + 1u] = sky.y;
    vision_output[out_base + 2u] = sky.z;
    vision_output[out_base + 3u] = sky.w;
    vision_output[out_base + 4u] = 1.0;
}
"#;

// ── VisionParams (Rust side, matches WGSL struct layout) ────────────

/// Uniform parameters for the vision raycast compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisionParams {
    pub num_agents: u32,
    pub num_rays: u32,
    pub max_dist: f32,
    pub step_size: f32,
    pub world_size: f32,
    pub terrain_vps: u32,
    pub terrain_size: f32,
    pub num_food: u32,
    pub num_agents_total: u32,
    pub agent_radius_sq: f32,
    pub food_radius_sq: f32,
    pub biome_res: u32,
    // 12 fields × 4 bytes = 48 bytes (already 16-byte aligned).
    // Explicit padding to 64 bytes for future-proofing against
    // stricter WGSL uniform alignment requirements on some drivers.
    pub _padding: [u32; 4],
}

// ── GpuVisionCompute ─────────────────────────────────────────────────

/// GPU vision raycast pipeline.
///
/// Uploads terrain heightmap, food positions, agent positions, and ray
/// parameters. Each thread marches one ray for one agent. Output is
/// RGBA+depth per ray, identical to CPU `sample_vision_positions`.
///
/// Buffer capacities are fixed when the pipeline is constructed. Callers
/// must not submit more food positions, agent positions, or rays than the
/// counts this instance was created to support; create a new
/// `GpuVisionCompute` if larger capacities are needed.
pub struct GpuVisionCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    _bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // Buffers -- allocated at construction for fixed capacities
    params_buf: wgpu::Buffer,
    _terrain_buf: wgpu::Buffer,
    _biome_buf: wgpu::Buffer,
    food_buf: wgpu::Buffer,
    agent_pos_buf: wgpu::Buffer,
    ray_origins_buf: wgpu::Buffer,
    ray_dirs_buf: wgpu::Buffer,
    vision_buf: wgpu::Buffer,
    vision_staging: wgpu::Buffer,
    num_agents: u32,
    num_rays: u32,
    has_in_flight: bool,
    // Invariant terrain parameters (set at construction, never change)
    terrain_vps: u32,
    terrain_size_val: f32,
    biome_res: u32,
}

impl GpuVisionCompute {
    /// Create a new GPU vision compute pipeline.
    ///
    /// Accepts device and queue from `ComputeBackend::GpuAccelerated`.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        num_agents: u32,
        terrain_heights: &[f32],
        biome_types: &[u32],
        terrain_vps: u32,
        terrain_size: f32,
        biome_res: u32,
        num_food: usize,
        num_agents_total: usize,
    ) -> Self {
        let num_rays = 48u32; // 8x6

        // Compile shader
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vision.wgsl"),
            source: wgpu::ShaderSource::Wgsl(VISION_SHADER.into()),
        });

        // Create pipeline with auto-derived layout
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vision"),
            layout: None,
            module: &module,
            entry_point: Some("vision_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Buffer sizes (bytes)
        // Minimum 16 bytes for storage buffers (wgpu requires non-zero)
        let food_size_bytes = ((num_food.max(1) * 4) as u64) * 4;
        let agent_pos_size_bytes = ((num_agents_total.max(1) * 4) as u64) * 4;
        let ray_origins_size_bytes = (num_agents.max(1) as u64) * 3 * 4;
        let ray_dirs_size_bytes = (num_agents.max(1) as u64) * (num_rays as u64) * 3 * 4;
        let vision_size_bytes = (num_agents.max(1) as u64) * (num_rays as u64) * 5 * 4;

        // Params
        let params = VisionParams {
            num_agents,
            num_rays,
            max_dist: 50.0,
            step_size: 1.0,
            world_size: terrain_size,
            terrain_vps,
            terrain_size,
            num_food: num_food as u32,
            num_agents_total: num_agents_total as u32,
            agent_radius_sq: 2.25,
            food_radius_sq: 1.0,
            biome_res,
            _padding: [0; 4],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Terrain and biome are uploaded once at construction
        let terrain_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-terrain"),
            contents: bytemuck::cast_slice(terrain_heights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let biome_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-biome"),
            contents: bytemuck::cast_slice(biome_types),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Per-tick buffers
        let food_buf = make_storage_in(&device, "vision-food", food_size_bytes);
        let agent_pos_buf = make_storage_in(&device, "vision-agent-pos", agent_pos_size_bytes);
        let ray_origins_buf = make_storage_in(&device, "vision-ray-origins", ray_origins_size_bytes);
        let ray_dirs_buf = make_storage_in(&device, "vision-ray-dirs", ray_dirs_size_bytes);

        let vision_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vision-output"),
            size: vision_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vision_staging = make_staging(&device, "vision-staging", vision_size_bytes);

        // Create bind group once — buffers are fixed for the pipeline's lifetime
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-bg"),
            layout: &bind_group_layout,
            entries: &[
                bg_buf(0, &params_buf),
                bg_buf(1, &terrain_buf),
                bg_buf(2, &biome_buf),
                bg_buf(3, &food_buf),
                bg_buf(4, &agent_pos_buf),
                bg_buf(5, &ray_origins_buf),
                bg_buf(6, &ray_dirs_buf),
                bg_buf(7, &vision_buf),
            ],
        });

        log::info!(
            "[GPU-VISION] Ready: {} agents, {} rays/agent, terrain_vps={}, {} food, {} agents_total",
            num_agents, num_rays, terrain_vps, num_food, num_agents_total,
        );

        Self {
            device,
            queue,
            pipeline,
            _bind_group_layout: bind_group_layout,
            bind_group,
            params_buf,
            _terrain_buf: terrain_buf,
            _biome_buf: biome_buf,
            food_buf,
            agent_pos_buf,
            ray_origins_buf,
            ray_dirs_buf,
            vision_buf,
            vision_staging,
            num_agents,
            num_rays,
            has_in_flight: false,
            terrain_vps,
            terrain_size_val: terrain_size,
            biome_res,
        }
    }

    /// Upload per-tick data and dispatch the vision compute shader.
    ///
    /// - `food_positions`: `[x, y, z, consumed_f32]` per food item
    /// - `agent_positions`: `[x, y, z, alive_f32]` per agent
    /// - `ray_origins`: `[x, y, z]` per agent
    /// - `ray_dirs`: `[x, y, z]` per ray per agent (48 rays per agent)
    pub fn submit(
        &mut self,
        food_positions: &[f32],
        agent_positions: &[f32],
        ray_origins: &[f32],
        ray_dirs: &[f32],
    ) {
        // Validate that slice sizes fit the pre-allocated buffers
        assert!(
            food_positions.len() * 4 <= self.food_buf.size() as usize,
            "food_positions ({} floats) exceeds food buffer capacity",
            food_positions.len(),
        );
        assert!(
            agent_positions.len() * 4 <= self.agent_pos_buf.size() as usize,
            "agent_positions ({} floats) exceeds agent_pos buffer capacity",
            agent_positions.len(),
        );
        assert!(
            ray_origins.len() * 4 <= self.ray_origins_buf.size() as usize,
            "ray_origins ({} floats) exceeds ray_origins buffer capacity",
            ray_origins.len(),
        );
        assert!(
            ray_dirs.len() * 4 <= self.ray_dirs_buf.size() as usize,
            "ray_dirs ({} floats) exceeds ray_dirs buffer capacity",
            ray_dirs.len(),
        );

        // Update params (num_food/num_agents_total may change per tick
        // within the pre-allocated capacity)
        let num_food = (food_positions.len() / 4) as u32;
        let num_agents_total = (agent_positions.len() / 4) as u32;
        let params = VisionParams {
            num_agents: self.num_agents,
            num_rays: self.num_rays,
            max_dist: 50.0,
            step_size: 1.0,
            world_size: self.terrain_size_val,
            terrain_vps: self.terrain_vps,
            terrain_size: self.terrain_size_val,
            num_food,
            num_agents_total,
            agent_radius_sq: 2.25,
            food_radius_sq: 1.0,
            biome_res: self.biome_res,
            _padding: [0; 4],
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Upload per-tick data
        self.queue.write_buffer(&self.food_buf, 0, bytemuck::cast_slice(food_positions));
        self.queue.write_buffer(&self.agent_pos_buf, 0, bytemuck::cast_slice(agent_positions));
        self.queue.write_buffer(&self.ray_origins_buf, 0, bytemuck::cast_slice(ray_origins));
        self.queue.write_buffer(&self.ray_dirs_buf, 0, bytemuck::cast_slice(ray_dirs));

        // Reuse bind group created at construction (buffers are fixed)
        let mut cmd = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vision-compute"),
        });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // dispatch(1, num_agents, 1) — workgroup_size is (48, 1, 1)
            pass.dispatch_workgroups(1, self.num_agents, 1);
        }

        // Copy output to staging
        let vision_size = (self.num_agents as u64) * (self.num_rays as u64) * 5 * 4;
        cmd.copy_buffer_to_buffer(&self.vision_buf, 0, &self.vision_staging, 0, vision_size);

        self.queue.submit(std::iter::once(cmd.finish()));

        // Request async map
        self.vision_staging
            .slice(..vision_size)
            .map_async(wgpu::MapMode::Read, |result| {
                if let Err(e) = result {
                    log::error!("[GPU-VISION] map_async failed: {}", e);
                }
            });
        self.has_in_flight = true;
    }

    /// Read back vision output. Blocks until the GPU is done.
    ///
    /// Returns 5 floats per ray per agent: `[r, g, b, a, depth]`.
    /// Total length = num_agents * num_rays * 5.
    pub fn collect_blocking(&mut self) -> Option<Vec<f32>> {
        if !self.has_in_flight {
            return None;
        }
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let vision_len = (self.num_agents as usize) * (self.num_rays as usize) * 5;
        let vision_size = (vision_len as u64) * 4;

        let slice = self.vision_staging.slice(..vision_size);
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.vision_staging.unmap();
        self.has_in_flight = false;

        Some(result)
    }
}

// ── Ray direction computation (CPU helper) ───────────────────────────

/// Compute ray origins and directions for all agents.
///
/// Returns `(origins, dirs)` where:
/// - `origins`: `[x, y, z]` per agent (length = agents.len() * 3)
/// - `dirs`: `[x, y, z]` per ray per agent (length = agents.len() * 48 * 3)
///
/// Ray layout: 8 columns x 6 rows, FOV = 90 degrees.
pub fn compute_ray_params(
    agents: &[(Vec3, f32)], // (position, yaw) per agent
) -> (Vec<f32>, Vec<f32>) {
    let num_rays = 48u32; // 8x6
    let w = 8u32;
    let h = 6u32;
    let half_fov = (90.0_f32 / 2.0).to_radians();
    let tan_hf = half_fov.tan();

    let mut origins = Vec::with_capacity(agents.len() * 3);
    let mut dirs = Vec::with_capacity(agents.len() * num_rays as usize * 3);

    for (pos, yaw) in agents {
        origins.push(pos.x);
        origins.push(pos.y);
        origins.push(pos.z);

        let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let right = Vec3::new(fwd.z, 0.0, -fwd.x).normalize_or_zero();

        for row in 0..h {
            for col in 0..w {
                let u = (col as f32 / (w - 1) as f32) * 2.0 - 1.0;
                let v = (row as f32 / (h - 1) as f32) * 2.0 - 1.0;
                let ray = (fwd + right * u * tan_hf + Vec3::Y * (-v) * tan_hf).normalize_or_zero();
                dirs.push(ray.x);
                dirs.push(ray.y);
                dirs.push(ray.z);
            }
        }
    }

    (origins, dirs)
}

// ── Helpers ───────────────────────────────────────────────────────────

fn make_storage_in(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_staging(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bg_buf<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_params_is_64_bytes() {
        assert_eq!(
            std::mem::size_of::<VisionParams>(),
            64,
            "VisionParams must be 64 bytes to match WGSL uniform layout"
        );
    }
}
