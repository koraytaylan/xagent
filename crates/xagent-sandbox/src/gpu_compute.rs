//! GPU compute pipeline for brain operations (encode + memory recall).
//!
//! Offloads the two most expensive brain operations to the GPU:
//! - Sensory encoding: matrix-vector multiply + tanh (~3% of brain FLOPs)
//! - Memory recall: batch cosine similarity (~80% of brain FLOPs)
//!
//! Together these account for ~83% of per-tick brain computation. The remaining
//! operations (prediction, action selection, learning) stay on CPU where they
//! need mutable access to brain internals.
//!
//! # Architecture
//!
//! The module creates its own wgpu device and queue, independent of the renderer.
//! Note: even on Apple Silicon unified memory, `queue.write_buffer` performs a
//! CPU-side staging copy internally. The pipeline uses double-buffered staging
//! and async readback to overlap GPU compute with CPU sensory extraction.
//!
//! Two compute passes per dispatch:
//! 1. **Encode**: features × weights + biases → tanh → encoded state (per agent)
//! 2. **Recall**: cosine similarity between encoded state and all memory patterns
//!    (per agent × per pattern), using workgroup-level parallel reduction
//!
//! # Pipelining
//!
//! The `submit()` / `collect()` API allows the caller to overlap GPU compute
//! with CPU work:
//! 1. `collect()` — read previous dispatch results (GPU already finished)
//! 2. CPU does sensory extraction in parallel (rayon)
//! 3. `submit()` — kick off new GPU dispatch (returns immediately)
//! 4. GPU computes while CPU prepares next tick
//!
//! This introduces a 1-brain-tick latency in motor responses (negligible for
//! evolution) but eliminates synchronous GPU blocking.
//!
//! **Adaptive scheduling:** The caller decides per-frame whether to use the
//! GPU path (1 dispatch/frame) or CPU rayon (brain_stride-decimated ticks
//! inside the simulation loop). When speed is low (≤~16×), expected brain
//! ticks per frame ≤ 2, so one GPU dispatch matches CPU throughput. At higher
//! speeds, the caller automatically switches to CPU rayon to preserve
//! cognitive throughput. This ensures agents get equivalent brain tick rates
//! regardless of which compute path is active.
//!
//! # Performance
//!
//! # Performance
//!
//! For 10 agents the GPU dispatch overhead (~50-100μs) dominates the actual
//! compute time. GPU becomes advantageous above ~50 agents. Use `--gpu-brain`
//! to opt in; the caller uses an adaptive scheduler that automatically falls
//! back to CPU rayon when the speed multiplier would cause GPU mode to deliver
//! fewer brain ticks than the CPU path (roughly above 10-16× speed).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters passed to both compute shaders as a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    num_agents: u32,
    dim: u32,
    feature_count: u32,
    memory_capacity: u32,
}

// ── WGSL Shaders ──────────────────────────────────────────────────────

const ENCODE_SHADER: &str = r#"
struct Params {
    num_agents: u32,
    dim: u32,
    feature_count: u32,
    memory_capacity: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> enc_weights: array<f32>;
@group(0) @binding(3) var<storage, read> enc_biases: array<f32>;
@group(0) @binding(4) var<storage, read_write> encoded: array<f32>;

// One thread per output dimension, one workgroup-row per agent.
// dispatch(ceil(dim/32), num_agents, 1)
@compute @workgroup_size(32, 1, 1)
fn encode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;   // output dimension index
    let a = gid.y;   // agent index

    if (a >= params.num_agents || d >= params.dim) {
        return;
    }

    let feat_base   = a * params.feature_count;
    let weight_base = a * params.feature_count * params.dim;
    let out_base    = a * params.dim;

    var sum = enc_biases[out_base + d];
    for (var j = 0u; j < params.feature_count; j = j + 1u) {
        sum = sum + features[feat_base + j] * enc_weights[weight_base + j * params.dim + d];
    }
    encoded[out_base + d] = tanh(sum);
}
"#;

const RECALL_SHADER: &str = r#"
struct Params {
    num_agents: u32,
    dim: u32,
    feature_count: u32,
    memory_capacity: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> encoded: array<f32>;
@group(0) @binding(2) var<storage, read> mem_patterns: array<f32>;
@group(0) @binding(3) var<storage, read> mem_active: array<u32>;
@group(0) @binding(4) var<storage, read_write> similarities: array<f32>;

// Workgroup-shared accumulators for parallel dot-product reduction.
var<workgroup> s_dot: array<f32, 32>;
var<workgroup> s_mag_a: array<f32, 32>;
var<workgroup> s_mag_b: array<f32, 32>;

// One workgroup per (agent, pattern) pair. 32 threads cooperate on the
// cosine similarity computation over `dim` dimensions.
// dispatch(memory_capacity, num_agents, 1)
@compute @workgroup_size(32, 1, 1)
fn recall_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let pat_idx   = wid.x;   // pattern slot index
    let agent_idx = wid.y;   // agent index
    let tid       = lid.x;   // thread within workgroup

    if (agent_idx >= params.num_agents || pat_idx >= params.memory_capacity) {
        return;
    }

    let slot = agent_idx * params.memory_capacity + pat_idx;

    // Skip inactive (empty) pattern slots
    if (mem_active[slot] == 0u) {
        if (tid == 0u) {
            similarities[slot] = -2.0;
        }
        return;
    }

    let enc_base = agent_idx * params.dim;
    let pat_base = slot * params.dim;

    // Each thread handles dimensions [tid, tid+32, tid+64, ...]
    var local_dot: f32 = 0.0;
    var local_mag_a: f32 = 0.0;
    var local_mag_b: f32 = 0.0;

    var i = tid;
    while (i < params.dim) {
        let a = encoded[enc_base + i];
        let b = mem_patterns[pat_base + i];
        local_dot   = local_dot   + a * b;
        local_mag_a = local_mag_a + a * a;
        local_mag_b = local_mag_b + b * b;
        i = i + 32u;
    }

    s_dot[tid]   = local_dot;
    s_mag_a[tid] = local_mag_a;
    s_mag_b[tid] = local_mag_b;
    workgroupBarrier();

    // Tree reduction: 32 → 1
    for (var s = 16u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            s_dot[tid]   += s_dot[tid + s];
            s_mag_a[tid] += s_mag_a[tid + s];
            s_mag_b[tid] += s_mag_b[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let dot_val = s_dot[0];
        let ma = sqrt(s_mag_a[0]);
        let mb = sqrt(s_mag_b[0]);

        var sim: f32 = 0.0;
        if (ma >= 1e-8 && mb >= 1e-8) {
            sim = clamp(dot_val / (ma * mb), -1.0, 1.0);
        }
        similarities[slot] = sim;
    }
}
"#;

// ── Rust infrastructure ───────────────────────────────────────────────

/// GPU compute pipeline for batched brain encode + recall.
///
/// Owns its own wgpu device and queue (independent of the renderer).
pub struct GpuBrainCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,

    num_agents: u32,
    dim: u32,
    feature_count: u32,
    memory_capacity: u32,

    encode_pipeline: wgpu::ComputePipeline,
    recall_pipeline: wgpu::ComputePipeline,

    // GPU storage buffers
    #[allow(dead_code)]
    params_buf: wgpu::Buffer,
    features_buf: wgpu::Buffer,
    enc_weights_buf: wgpu::Buffer,
    enc_biases_buf: wgpu::Buffer,
    encoded_buf: wgpu::Buffer,
    mem_patterns_buf: wgpu::Buffer,
    mem_active_buf: wgpu::Buffer,
    similarities_buf: wgpu::Buffer,

    // Double-buffered staging for pipelined async readback
    encoded_staging: [wgpu::Buffer; 2],
    similarities_staging: [wgpu::Buffer; 2],
    staging_idx: usize,
    has_in_flight: bool,

    // Non-blocking completion flags set by map_async callbacks
    enc_mapped: Arc<AtomicBool>,
    sim_mapped: Arc<AtomicBool>,

    encode_bind_group: wgpu::BindGroup,
    recall_bind_group: wgpu::BindGroup,
}

impl GpuBrainCompute {
    /// Attempt to create a GPU compute pipeline.
    /// Returns `None` if no suitable GPU adapter is available.
    pub fn try_new(
        num_agents: u32,
        dim: u32,
        feature_count: u32,
        memory_capacity: u32,
    ) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        log::info!("[GPU-COMPUTE] Adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xagent-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;

        // Compile shaders
        let encode_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("encode.wgsl"),
            source: wgpu::ShaderSource::Wgsl(ENCODE_SHADER.into()),
        });
        let recall_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recall.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RECALL_SHADER.into()),
        });

        // Create pipelines with auto-derived layouts
        let encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("encode"),
                layout: None,
                module: &encode_module,
                entry_point: Some("encode_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let recall_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("recall"),
                layout: None,
                module: &recall_module,
                entry_point: Some("recall_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Buffer sizes (in bytes)
        let n = num_agents as u64;
        let d = dim as u64;
        let f = feature_count as u64;
        let c = memory_capacity as u64;

        let features_size = n * f * 4;
        let enc_weights_size = n * f * d * 4;
        let enc_biases_size = n * d * 4;
        let encoded_size = n * d * 4;
        let mem_patterns_size = n * c * d * 4;
        let mem_active_size = n * c * 4;
        let similarities_size = n * c * 4;

        // Create buffers
        let params = GpuParams {
            num_agents,
            dim,
            feature_count,
            memory_capacity,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let features_buf = make_storage_in(&device, "features", features_size);
        let enc_weights_buf = make_storage_in(&device, "enc-weights", enc_weights_size);
        let enc_biases_buf = make_storage_in(&device, "enc-biases", enc_biases_size);
        let encoded_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded"),
            size: encoded_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mem_patterns_buf = make_storage_in(&device, "mem-patterns", mem_patterns_size);
        let mem_active_buf = make_storage_in(&device, "mem-active", mem_active_size);
        let similarities_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("similarities"),
            size: similarities_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let encoded_staging = [
            make_staging(&device, "encoded-staging-0", encoded_size),
            make_staging(&device, "encoded-staging-1", encoded_size),
        ];
        let similarities_staging = [
            make_staging(&device, "similarities-staging-0", similarities_size),
            make_staging(&device, "similarities-staging-1", similarities_size),
        ];

        // Bind groups (auto-derived layouts from pipelines)
        let encode_bgl = encode_pipeline.get_bind_group_layout(0);
        let encode_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode-bg"),
            layout: &encode_bgl,
            entries: &[
                bg_buf(0, &params_buf),
                bg_buf(1, &features_buf),
                bg_buf(2, &enc_weights_buf),
                bg_buf(3, &enc_biases_buf),
                bg_buf(4, &encoded_buf),
            ],
        });

        let recall_bgl = recall_pipeline.get_bind_group_layout(0);
        let recall_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("recall-bg"),
            layout: &recall_bgl,
            entries: &[
                bg_buf(0, &params_buf),
                bg_buf(1, &encoded_buf),
                bg_buf(2, &mem_patterns_buf),
                bg_buf(3, &mem_active_buf),
                bg_buf(4, &similarities_buf),
            ],
        });

        log::info!(
            "[GPU-COMPUTE] Ready: {} agents, dim={}, features={}, memory_capacity={}",
            num_agents, dim, feature_count, memory_capacity,
        );

        Some(Self {
            device,
            queue,
            num_agents,
            dim,
            feature_count,
            memory_capacity,
            encode_pipeline,
            recall_pipeline,
            params_buf,
            features_buf,
            enc_weights_buf,
            enc_biases_buf,
            encoded_buf,
            mem_patterns_buf,
            mem_active_buf,
            similarities_buf,
            encoded_staging,
            similarities_staging,
            staging_idx: 0,
            has_in_flight: false,
            enc_mapped: Arc::new(AtomicBool::new(false)),
            sim_mapped: Arc::new(AtomicBool::new(false)),
            encode_bind_group,
            recall_bind_group,
        })
    }

    /// Submit GPU compute work asynchronously.
    ///
    /// Uploads data, dispatches encode + recall shaders, copies results to a
    /// staging buffer, and requests an async map — then returns immediately
    /// without waiting for the GPU. Call `collect()` later to read back the
    /// results (typically after doing CPU-side sensory extraction in parallel).
    pub fn submit(
        &mut self,
        features: &[f32],
        enc_weights: &[f32],
        enc_biases: &[f32],
        mem_patterns: &[f32],
        mem_active: &[u32],
    ) {
        let widx = self.staging_idx;

        // If a previous dispatch targeting this staging pair was never collected,
        // unmap the buffers so they can be used as copy destinations again.
        // This prevents a wgpu validation error when try_collect() returns None
        // for 2+ consecutive frames (the double-buffer wraps around).
        if self.has_in_flight {
            self.encoded_staging[widx].unmap();
            self.similarities_staging[widx].unmap();
        }

        // Upload input data
        self.queue
            .write_buffer(&self.features_buf, 0, bytemuck::cast_slice(features));
        self.queue
            .write_buffer(&self.enc_weights_buf, 0, bytemuck::cast_slice(enc_weights));
        self.queue
            .write_buffer(&self.enc_biases_buf, 0, bytemuck::cast_slice(enc_biases));
        self.queue.write_buffer(
            &self.mem_patterns_buf,
            0,
            bytemuck::cast_slice(mem_patterns),
        );
        self.queue
            .write_buffer(&self.mem_active_buf, 0, bytemuck::cast_slice(mem_active));

        let mut cmd = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("brain-compute"),
            });

        // Pass 1: Encode (features → encoded states)
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.encode_pipeline);
            pass.set_bind_group(0, &self.encode_bind_group, &[]);
            let x_groups = (self.dim + 31) / 32;
            pass.dispatch_workgroups(x_groups, self.num_agents, 1);
        }

        // Pass 2: Recall (encoded states × memory patterns → similarity scores)
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("recall"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.recall_pipeline);
            pass.set_bind_group(0, &self.recall_bind_group, &[]);
            pass.dispatch_workgroups(self.memory_capacity, self.num_agents, 1);
        }

        // Copy results to the current staging pair
        let enc_size = (self.num_agents * self.dim) as u64 * 4;
        let sim_size = (self.num_agents * self.memory_capacity) as u64 * 4;
        cmd.copy_buffer_to_buffer(
            &self.encoded_buf, 0, &self.encoded_staging[widx], 0, enc_size,
        );
        cmd.copy_buffer_to_buffer(
            &self.similarities_buf, 0, &self.similarities_staging[widx], 0, sim_size,
        );

        self.queue.submit(std::iter::once(cmd.finish()));

        // Request async mapping with completion flags
        self.enc_mapped.store(false, Ordering::Release);
        self.sim_mapped.store(false, Ordering::Release);

        let enc_flag = self.enc_mapped.clone();
        self.encoded_staging[widx]
            .slice(..enc_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    enc_flag.store(true, Ordering::Release);
                }
            });
        let sim_flag = self.sim_mapped.clone();
        self.similarities_staging[widx]
            .slice(..sim_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    sim_flag.store(true, Ordering::Release);
                }
            });

        self.staging_idx = 1 - self.staging_idx;
        self.has_in_flight = true;
    }

    /// Non-blocking collect: poll GPU and return results if ready.
    ///
    /// If the GPU hasn't finished yet, returns `None` — agents should
    /// keep using their cached motor commands. Never blocks the caller.
    pub fn try_collect(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        if !self.has_in_flight {
            return None;
        }

        self.device.poll(wgpu::Maintain::Poll);

        if !self.enc_mapped.load(Ordering::Acquire)
            || !self.sim_mapped.load(Ordering::Acquire)
        {
            return None;
        }

        self.read_mapped_staging()
    }

    /// Blocking collect: waits for GPU results. Used by `dispatch()` and
    /// end-of-generation flush where we must have results.
    pub fn collect_blocking(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        if !self.has_in_flight {
            return None;
        }

        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        self.read_mapped_staging()
    }

    fn read_mapped_staging(&mut self) -> Option<(Vec<f32>, Vec<f32>)> {
        let read_idx = 1 - self.staging_idx;

        let enc_size = (self.num_agents * self.dim) as u64 * 4;
        let sim_size = (self.num_agents * self.memory_capacity) as u64 * 4;

        let enc_slice = self.encoded_staging[read_idx].slice(..enc_size);
        let enc_data = enc_slice.get_mapped_range();
        let encoded: Vec<f32> = bytemuck::cast_slice(&enc_data).to_vec();
        drop(enc_data);
        self.encoded_staging[read_idx].unmap();

        let sim_slice = self.similarities_staging[read_idx].slice(..sim_size);
        let sim_data = sim_slice.get_mapped_range();
        let similarities: Vec<f32> = bytemuck::cast_slice(&sim_data).to_vec();
        drop(sim_data);
        self.similarities_staging[read_idx].unmap();

        self.has_in_flight = false;

        Some((encoded, similarities))
    }

    /// Synchronous dispatch: submit + blocking collect. Used by tests.
    pub fn dispatch(
        &mut self,
        features: &[f32],
        enc_weights: &[f32],
        enc_biases: &[f32],
        mem_patterns: &[f32],
        mem_active: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        self.submit(features, enc_weights, enc_biases, mem_patterns, mem_active);
        self.collect_blocking().expect("collect after submit must return results")
    }

    /// Whether the current config matches the given dimensions.
    pub fn matches_config(
        &self,
        num_agents: u32,
        dim: u32,
        feature_count: u32,
        memory_capacity: u32,
    ) -> bool {
        self.num_agents == num_agents
            && self.dim == dim
            && self.feature_count == feature_count
            && self.memory_capacity == memory_capacity
    }

    pub fn num_agents(&self) -> u32 {
        self.num_agents
    }
    pub fn dim(&self) -> u32 {
        self.dim
    }
    pub fn feature_count(&self) -> u32 {
        self.feature_count
    }
    pub fn memory_capacity(&self) -> u32 {
        self.memory_capacity
    }

    /// Append one agent's brain data to the batch accumulators, padding to
    /// the GPU pipeline's expected (dim, feature_count, memory_capacity).
    ///
    /// Call this for every agent slot (active or inactive). For inactive
    /// agents pass empty slices and `agent_dim = agent_fc = agent_cap = 0`.
    pub fn pad_agent_into(
        &self,
        features: &[f32],
        weights: &[f32],
        biases: &[f32],
        patterns: &[f32],
        active: &[u32],
        agent_dim: usize,
        agent_fc: usize,
        agent_cap: usize,
        all_features: &mut Vec<f32>,
        all_weights: &mut Vec<f32>,
        all_biases: &mut Vec<f32>,
        all_patterns: &mut Vec<f32>,
        all_active: &mut Vec<u32>,
    ) {
        let dim = self.dim as usize;
        let fc = self.feature_count as usize;
        let cap = self.memory_capacity as usize;

        // Features: [agent_fc] → [fc] (pad trailing zeros)
        let f_copy = features.len().min(fc);
        all_features.extend_from_slice(&features[..f_copy]);
        if fc > f_copy {
            all_features.extend(std::iter::repeat(0.0f32).take(fc - f_copy));
        }

        // Biases: [agent_dim] → [dim]
        let b_copy = biases.len().min(dim);
        all_biases.extend_from_slice(&biases[..b_copy]);
        if dim > b_copy {
            all_biases.extend(std::iter::repeat(0.0f32).take(dim - b_copy));
        }

        // Weights: [agent_fc × agent_dim] row-major → [fc × dim]
        for r in 0..fc {
            if r < agent_fc && agent_dim > 0 {
                let src = r * agent_dim;
                let copy_len = agent_dim.min(dim);
                all_weights.extend_from_slice(&weights[src..src + copy_len]);
                if dim > agent_dim {
                    all_weights.extend(std::iter::repeat(0.0f32).take(dim - agent_dim));
                }
            } else {
                all_weights.extend(std::iter::repeat(0.0f32).take(dim));
            }
        }

        // Patterns: [agent_cap × agent_dim] → [cap × dim]
        for p in 0..cap {
            if p < agent_cap && agent_dim > 0 {
                let src = p * agent_dim;
                let copy_len = agent_dim.min(dim);
                all_patterns.extend_from_slice(&patterns[src..src + copy_len]);
                if dim > agent_dim {
                    all_patterns.extend(std::iter::repeat(0.0f32).take(dim - agent_dim));
                }
            } else {
                all_patterns.extend(std::iter::repeat(0.0f32).take(dim));
            }
        }

        // Active mask: [agent_cap] → [cap]
        let a_copy = active.len().min(cap);
        all_active.extend_from_slice(&active[..a_copy]);
        if cap > a_copy {
            all_active.extend(std::iter::repeat(0u32).take(cap - a_copy));
        }
    }
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

    fn try_gpu(
        num_agents: u32,
        dim: u32,
        feature_count: u32,
        memory_capacity: u32,
    ) -> Option<GpuBrainCompute> {
        GpuBrainCompute::try_new(num_agents, dim, feature_count, memory_capacity)
    }

    fn cpu_encode(features: &[f32], weights: &[f32], biases: &[f32], dim: usize) -> Vec<f32> {
        let feature_count = features.len();
        let mut out = vec![0.0f32; dim];
        for i in 0..dim {
            let mut sum = biases[i];
            for j in 0..feature_count {
                sum += features[j] * weights[j * dim + i];
            }
            out[i] = sum.tanh();
        }
        out
    }

    fn cpu_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if ma < 1e-8 || mb < 1e-8 {
            return 0.0;
        }
        (dot / (ma * mb)).clamp(-1.0, 1.0)
    }

    #[test]
    fn gpu_encode_matches_cpu() {
        let mut gpu = match try_gpu(1, 4, 3, 2) {
            Some(g) => g,
            None => {
                eprintln!("No GPU available, skipping test");
                return;
            }
        };

        let features = vec![0.5, -0.3, 0.8];
        let weights = vec![
            0.1, 0.2, -0.1, 0.3, // feature 0 → dim 0-3
            -0.2, 0.5, 0.1, -0.4, // feature 1 → dim 0-3
            0.3, -0.1, 0.4, 0.2, // feature 2 → dim 0-3
        ];
        let biases = vec![0.01, -0.02, 0.03, -0.01];
        let mem_patterns = vec![0.0f32; 2 * 4]; // 2 patterns × 4 dim, all zeros
        let mem_active = vec![0u32; 2]; // no active patterns

        let (gpu_encoded, _) = gpu.dispatch(&features, &weights, &biases, &mem_patterns, &mem_active);
        let cpu_encoded = cpu_encode(&features, &weights, &biases, 4);

        for (i, (g, c)) in gpu_encoded.iter().zip(cpu_encoded.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "Encode dim {}: GPU={} CPU={} diff={}",
                i,
                g,
                c,
                (g - c).abs()
            );
        }
    }

    #[test]
    fn gpu_recall_matches_cpu() {
        let dim = 4u32;
        let capacity = 3u32;
        let mut gpu = match try_gpu(1, dim, 3, capacity) {
            Some(g) => g,
            None => {
                eprintln!("No GPU available, skipping test");
                return;
            }
        };

        // Pre-compute encoded state on GPU (dummy encode)
        let features = vec![1.0, 0.0, 0.0];
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // identity-ish
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let biases = vec![0.0; 4];

        // 3 patterns: one similar to encoded, one orthogonal, one inactive
        let pattern0 = vec![0.7, 0.1, 0.0, 0.0]; // similar to tanh([1,0,0,0])
        let pattern1 = vec![0.0, 0.0, 0.0, 1.0]; // orthogonal
        let pattern2 = vec![0.0; 4]; // inactive

        let mut mem_patterns = Vec::new();
        mem_patterns.extend_from_slice(&pattern0);
        mem_patterns.extend_from_slice(&pattern1);
        mem_patterns.extend_from_slice(&pattern2);

        let mem_active = vec![1u32, 1, 0]; // pattern 2 inactive

        let (gpu_encoded, gpu_sims) =
            gpu.dispatch(&features, &weights, &biases, &mem_patterns, &mem_active);

        // CPU reference
        let cpu_encoded = cpu_encode(&features, &weights, &biases, dim as usize);
        let cpu_sim0 = cpu_cosine_sim(&cpu_encoded, &pattern0);
        let cpu_sim1 = cpu_cosine_sim(&cpu_encoded, &pattern1);

        // Verify encoded states match
        for (i, (g, c)) in gpu_encoded.iter().zip(cpu_encoded.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "Encode dim {}: GPU={} CPU={}",
                i, g, c
            );
        }

        // Verify similarity scores match
        assert!(
            (gpu_sims[0] - cpu_sim0).abs() < 1e-4,
            "Sim0: GPU={} CPU={}",
            gpu_sims[0],
            cpu_sim0
        );
        assert!(
            (gpu_sims[1] - cpu_sim1).abs() < 1e-4,
            "Sim1: GPU={} CPU={}",
            gpu_sims[1],
            cpu_sim1
        );
        assert!(
            gpu_sims[2] < -1.5,
            "Inactive pattern should be -2.0, got {}",
            gpu_sims[2]
        );
    }

    #[test]
    fn gpu_multi_agent_dispatch() {
        let n = 3u32;
        let dim = 4u32;
        let fc = 2u32;
        let cap = 2u32;
        let mut gpu = match try_gpu(n, dim, fc, cap) {
            Some(g) => g,
            None => {
                eprintln!("No GPU available, skipping test");
                return;
            }
        };

        // Each agent has different features
        let mut features = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut patterns = Vec::new();
        let mut active = Vec::new();

        for agent in 0..n {
            let f = agent as f32 * 0.3;
            features.extend_from_slice(&[f, 1.0 - f]);

            // Same weights for all agents (identity-like)
            for j in 0..fc {
                for i in 0..dim {
                    weights.push(if i == j { 1.0 } else { 0.0 });
                }
            }
            biases.extend_from_slice(&vec![0.0; dim as usize]);

            // One active pattern, one inactive
            patterns.extend_from_slice(&vec![0.5; dim as usize]);
            patterns.extend_from_slice(&vec![0.0; dim as usize]);
            active.extend_from_slice(&[1, 0]);
        }

        let (encoded, sims) = gpu.dispatch(&features, &weights, &biases, &patterns, &active);

        assert_eq!(encoded.len(), (n * dim) as usize);
        assert_eq!(sims.len(), (n * cap) as usize);

        // Each agent's encoded state should be different (different features)
        let enc0 = &encoded[0..dim as usize];
        let enc1 = &encoded[dim as usize..2 * dim as usize];
        let diff: f32 = enc0.iter().zip(enc1).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "Different agents should produce different encodings");

        // Second pattern of each agent should be inactive
        for agent in 0..n {
            let idx = (agent * cap + 1) as usize;
            assert!(
                sims[idx] < -1.5,
                "Agent {} inactive pattern sim should be -2.0, got {}",
                agent,
                sims[idx]
            );
        }
    }

    /// Verify that `pad_agent_into` correctly handles agents with smaller
    /// dimensions than the GPU pipeline expects — the exact scenario that
    /// causes buffer overruns when mutations change structural params.
    #[test]
    fn gpu_padding_preserves_encode_results() {
        // GPU pipeline sized for the larger agent: dim=4, fc=3, cap=3
        let gpu_dim = 4u32;
        let gpu_fc = 3u32;
        let gpu_cap = 3u32;
        let mut gpu = match try_gpu(2, gpu_dim, gpu_fc, gpu_cap) {
            Some(g) => g,
            None => {
                eprintln!("No GPU available, skipping test");
                return;
            }
        };

        // Agent 0: uses the full pipeline dimensions (dim=4, fc=3, cap=3)
        let feats_0 = vec![1.0, 0.5, 0.2];
        let weights_0: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect(); // 3×4
        let biases_0 = vec![0.01, 0.02, 0.03, 0.04];
        let pats_0 = vec![0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // 3×4
        let active_0 = vec![1u32, 1, 0];

        // Agent 1: smaller dimensions (dim=3, fc=2, cap=2) — the mutation case
        let feats_1 = vec![0.8, 0.3];
        let weights_1: Vec<f32> = (0..6).map(|i| (i as f32) * 0.15).collect(); // 2×3
        let biases_1 = vec![0.05, 0.06, 0.07];
        let pats_1 = vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0]; // 2×3
        let active_1 = vec![1u32, 0];

        let dim = gpu_dim as usize;
        let fc = gpu_fc as usize;
        let cap = gpu_cap as usize;

        let mut all_features = Vec::with_capacity(2 * fc);
        let mut all_weights = Vec::with_capacity(2 * fc * dim);
        let mut all_biases = Vec::with_capacity(2 * dim);
        let mut all_patterns = Vec::with_capacity(2 * cap * dim);
        let mut all_active = Vec::with_capacity(2 * cap);

        // Agent 0: no padding needed (exact match)
        gpu.pad_agent_into(
            &feats_0, &weights_0, &biases_0, &pats_0, &active_0,
            4, 3, 3,
            &mut all_features, &mut all_weights, &mut all_biases,
            &mut all_patterns, &mut all_active,
        );
        // Agent 1: needs padding
        gpu.pad_agent_into(
            &feats_1, &weights_1, &biases_1, &pats_1, &active_1,
            3, 2, 2,
            &mut all_features, &mut all_weights, &mut all_biases,
            &mut all_patterns, &mut all_active,
        );

        // Verify buffer sizes are correct
        assert_eq!(all_features.len(), 2 * fc, "features buffer wrong size");
        assert_eq!(all_weights.len(), 2 * fc * dim, "weights buffer wrong size");
        assert_eq!(all_biases.len(), 2 * dim, "biases buffer wrong size");
        assert_eq!(all_patterns.len(), 2 * cap * dim, "patterns buffer wrong size");
        assert_eq!(all_active.len(), 2 * cap, "active buffer wrong size");

        // Dispatch on GPU
        let (encoded, sims) = gpu.dispatch(
            &all_features, &all_weights, &all_biases, &all_patterns, &all_active,
        );

        // Verify agent 0 encode matches CPU reference (full dims)
        let cpu_enc_0 = cpu_encode(&feats_0, &weights_0, &biases_0, 4);
        for d in 0..4 {
            assert!(
                (encoded[d] - cpu_enc_0[d]).abs() < 1e-4,
                "Agent 0 dim {}: GPU={} CPU={}",
                d, encoded[d], cpu_enc_0[d]
            );
        }

        // Verify agent 1 encode: first 3 dims should match CPU with padded layout
        // The padded weights for agent 1 are: row0=[0,0.15,0.3,0], row1=[0.45,0.6,0.75,0], row2=[0,0,0,0]
        // So CPU encode with padded layout gives same first 3 dims
        let padded_weights_1: Vec<f32> = {
            let mut w = Vec::new();
            // row 0 of agent 1: weights_1[0..3] padded to dim=4
            w.extend_from_slice(&weights_1[0..3]);
            w.push(0.0);
            // row 1
            w.extend_from_slice(&weights_1[3..6]);
            w.push(0.0);
            // row 2 (padding row)
            w.extend_from_slice(&[0.0; 4]);
            w
        };
        let padded_feats_1 = vec![0.8, 0.3, 0.0];
        let padded_biases_1 = vec![0.05, 0.06, 0.07, 0.0];
        let cpu_enc_1 = cpu_encode(&padded_feats_1, &padded_weights_1, &padded_biases_1, 4);
        for d in 0..3 {
            assert!(
                (encoded[dim + d] - cpu_enc_1[d]).abs() < 1e-4,
                "Agent 1 dim {}: GPU={} CPU={}",
                d, encoded[dim + d], cpu_enc_1[d]
            );
        }
        // Dim 3 of agent 1 should be tanh(0) = 0 (padding)
        assert!(
            encoded[dim + 3].abs() < 1e-4,
            "Agent 1 padded dim should be ~0, got {}",
            encoded[dim + 3]
        );

        // Verify agent 1 inactive pattern (slot 1) and padding slot (slot 2)
        assert!(
            sims[cap + 1] < -1.5,
            "Agent 1 inactive pattern should be -2.0, got {}",
            sims[cap + 1]
        );
        assert!(
            sims[cap + 2] < -1.5,
            "Agent 1 padding slot should be -2.0, got {}",
            sims[cap + 2]
        );
    }
}
