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
//! For 10 agents the GPU dispatch overhead (~50-100μs) dominates the actual
//! compute time. GPU becomes advantageous above ~50 agents. GPU is auto-detected
//! at startup via `ComputeBackend::probe()`; the caller uses an adaptive
//! scheduler that automatically falls back to CPU rayon when the speed
//! multiplier would cause GPU mode to deliver
//! fewer brain ticks than the CPU path (roughly above 10-16× speed).

use glam::Vec3;
use std::sync::atomic::{AtomicU64, Ordering};
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
        sum = sum + features[feat_base + j] * enc_weights[weight_base + d * params.feature_count + j];
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
    staging_mapped: [bool; 2],

    // Non-blocking completion: map_async callbacks write the submit_seq they
    // belong to. try_collect() checks that both match the current submit_seq,
    // preventing stale callbacks from a prior submit being mistaken as ready.
    submit_seq: u64,
    enc_mapped_seq: Arc<AtomicU64>,
    sim_mapped_seq: Arc<AtomicU64>,

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

        // Request the adapter's actual max storage buffer binding size
        // instead of the conservative 128MB default.
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xagent-compute"),
                required_features: wgpu::Features::empty(),
                required_limits,
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

        // Guard: bail if any buffer exceeds the device's limits
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size as u64;
        let limit = max_binding.min(max_buf);
        let largest = enc_weights_size
            .max(mem_patterns_size)
            .max(features_size);
        if largest > limit {
            log::warn!(
                "[GPU-COMPUTE] Largest buffer ({:.1} MB) exceeds device limit ({:.1} MB) — falling back to CPU",
                largest as f64 / 1_048_576.0,
                limit as f64 / 1_048_576.0,
            );
            return None;
        }

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
            staging_mapped: [false; 2],
            submit_seq: 0,
            enc_mapped_seq: Arc::new(AtomicU64::new(0)),
            sim_mapped_seq: Arc::new(AtomicU64::new(0)),
            encode_bind_group,
            recall_bind_group,
        })
    }

    /// Create with a pre-existing device and queue (for shared pipelines).
    ///
    /// Same as `try_new()` but uses provided device/queue instead of creating
    /// new ones. Returns `None` if buffer sizes exceed device limits.
    pub fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        num_agents: u32,
        dim: u32,
        feature_count: u32,
        memory_capacity: u32,
    ) -> Option<Self> {
        // Compile shaders
        let encode_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("encode.wgsl"),
            source: wgpu::ShaderSource::Wgsl(ENCODE_SHADER.into()),
        });
        let recall_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recall.wgsl"),
            source: wgpu::ShaderSource::Wgsl(RECALL_SHADER.into()),
        });

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

        // Guard: bail if any buffer exceeds the device's limits
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size as u64;
        let limit = max_binding.min(max_buf);
        let largest = enc_weights_size
            .max(mem_patterns_size)
            .max(features_size);
        if largest > limit {
            log::warn!(
                "[GPU-COMPUTE] with_device: largest buffer ({:.1} MB) exceeds device limit ({:.1} MB)",
                largest as f64 / 1_048_576.0,
                limit as f64 / 1_048_576.0,
            );
            return None;
        }

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
            "[GPU-COMPUTE] with_device: {} agents, dim={}, features={}, memory_capacity={}",
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
            staging_mapped: [false; 2],
            submit_seq: 0,
            enc_mapped_seq: Arc::new(AtomicU64::new(0)),
            sim_mapped_seq: Arc::new(AtomicU64::new(0)),
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

        // If a previous dispatch mapped this staging pair and it was never
        // collected, unmap it so it can be used as a copy destination again.
        // Only unmap buffers that are actually mapped (avoids wgpu panic).
        if self.staging_mapped[widx] {
            self.encoded_staging[widx].unmap();
            self.similarities_staging[widx].unmap();
            self.staging_mapped[widx] = false;
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

        // Request async mapping with sequence-tagged completion
        self.submit_seq += 1;
        let seq = self.submit_seq;

        let enc_flag = self.enc_mapped_seq.clone();
        self.encoded_staging[widx]
            .slice(..enc_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    enc_flag.store(seq, Ordering::Release);
                }
            });
        let sim_flag = self.sim_mapped_seq.clone();
        self.similarities_staging[widx]
            .slice(..sim_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    sim_flag.store(seq, Ordering::Release);
                }
            });

        self.staging_mapped[widx] = true;
        self.staging_idx = 1 - self.staging_idx;
        self.has_in_flight = true;
    }

    /// Non-blocking collect into pre-allocated output buffers.
    ///
    /// If the GPU hasn't finished yet, returns `false` — agents should
    /// keep using their cached motor commands. Never blocks the caller.
    /// Uses sequence numbers to ensure only the latest submit's callbacks
    /// are accepted (prevents stale callback races under GPU pressure).
    pub fn try_collect_into(
        &mut self,
        encoded_out: &mut Vec<f32>,
        similarities_out: &mut Vec<f32>,
    ) -> bool {
        if !self.has_in_flight {
            return false;
        }

        self.device.poll(wgpu::Maintain::Poll);

        let seq = self.submit_seq;
        if self.enc_mapped_seq.load(Ordering::Acquire) != seq
            || self.sim_mapped_seq.load(Ordering::Acquire) != seq
        {
            return false;
        }

        self.read_mapped_staging_into(encoded_out, similarities_out);
        true
    }

    /// Blocking collect into pre-allocated output buffers.
    /// Used by `dispatch()` and end-of-generation flush where we must have results.
    pub fn collect_blocking_into(
        &mut self,
        encoded_out: &mut Vec<f32>,
        similarities_out: &mut Vec<f32>,
    ) -> bool {
        if !self.has_in_flight {
            return false;
        }

        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        self.read_mapped_staging_into(encoded_out, similarities_out);
        true
    }

    fn read_mapped_staging_into(
        &mut self,
        encoded_out: &mut Vec<f32>,
        similarities_out: &mut Vec<f32>,
    ) {
        let read_idx = 1 - self.staging_idx;

        let enc_len = (self.num_agents * self.dim) as usize;
        let sim_len = (self.num_agents * self.memory_capacity) as usize;

        let enc_slice = self.encoded_staging[read_idx].slice(..(enc_len as u64 * 4));
        let enc_data = enc_slice.get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&enc_data);
        encoded_out.clear();
        encoded_out.extend_from_slice(src);
        drop(enc_data);
        self.encoded_staging[read_idx].unmap();

        let sim_slice = self.similarities_staging[read_idx].slice(..(sim_len as u64 * 4));
        let sim_data = sim_slice.get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&sim_data);
        similarities_out.clear();
        similarities_out.extend_from_slice(src);
        drop(sim_data);
        self.similarities_staging[read_idx].unmap();

        self.staging_mapped[read_idx] = false;
        self.has_in_flight = false;
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
        let mut enc = Vec::new();
        let mut sim = Vec::new();
        assert!(self.collect_blocking_into(&mut enc, &mut sim), "collect after submit must return results");
        (enc, sim)
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
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // Buffers -- allocated at construction for fixed capacities
    params_buf: wgpu::Buffer,
    terrain_buf: wgpu::Buffer,
    biome_buf: wgpu::Buffer,
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
    /// Accepts device and queue from `ComputeBackend::GpuAccelerated` so
    /// that the device can be shared with `GpuBrainCompute` later.
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
            bind_group_layout,
            bind_group,
            params_buf,
            terrain_buf,
            biome_buf,
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

// ── Unified GPU Pipeline ──────────────────────────────────────────────

/// Combined result from a unified vision + brain GPU dispatch.
pub struct UnifiedResult {
    /// RGBA+depth per ray per agent (5 floats per ray, 48 rays per agent).
    pub vision: Vec<f32>,
    /// Encoded state per agent.
    pub encoded: Vec<f32>,
    /// Similarity scores per agent (per memory pattern).
    pub similarities: Vec<f32>,
}

/// Coordinated GPU pipeline for vision + encode + recall on a shared device/queue.
///
/// Provides a single high-level API that runs the existing vision and brain
/// GPU pipelines together. In the current implementation, those stages are
/// submitted separately via the underlying `vision` and `brain` compute
/// paths rather than being recorded into one command encoder.
///
/// True on-GPU chaining (vision output directly feeding encode features)
/// is deferred — it requires reformatting vision output to match the
/// feature vector layout.
pub struct GpuUnifiedPipeline {
    vision: GpuVisionCompute,
    brain: GpuBrainCompute,
}

impl GpuUnifiedPipeline {
    /// Create a unified pipeline sharing a single device/queue.
    ///
    /// Both vision and brain sub-pipelines receive cloned handles to the
    /// same device and queue (wgpu handles are Arc-wrapped internally).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        // vision params
        terrain_heights: &[f32],
        biome_types: &[u32],
        terrain_vps: u32,
        terrain_size: f32,
        biome_res: u32,
        num_food: usize,
        // brain params
        num_agents: u32,
        dim: u32,
        feature_count: u32,
        memory_capacity: u32,
    ) -> Option<Self> {
        let vision = GpuVisionCompute::new(
            device.clone(),
            queue.clone(),
            num_agents,
            terrain_heights,
            biome_types,
            terrain_vps,
            terrain_size,
            biome_res,
            num_food,
            num_agents as usize,
        );
        let brain = GpuBrainCompute::with_device(
            device,
            queue,
            num_agents,
            dim,
            feature_count,
            memory_capacity,
        )?;
        Some(Self { vision, brain })
    }

    /// Submit vision + brain in one dispatch.
    ///
    /// Vision runs first on GPU, then brain encode+recall. Both are
    /// dispatched as separate compute passes but submitted together,
    /// eliminating one CPU roundtrip compared to running them independently.
    pub fn submit_vision_and_brain(
        &mut self,
        // vision data
        food_positions: &[f32],
        agent_positions: &[f32],
        ray_origins: &[f32],
        ray_dirs: &[f32],
        // brain data
        features: &[f32],
        enc_weights: &[f32],
        enc_biases: &[f32],
        mem_patterns: &[f32],
        mem_active: &[u32],
    ) {
        // Submit vision and brain as separate dispatches.
        // They use separate command encoders currently (each sub-pipeline
        // owns its own submit path), but share the same device/queue so
        // the driver can batch them.
        self.vision.submit(food_positions, agent_positions, ray_origins, ray_dirs);
        self.brain.submit(features, enc_weights, enc_biases, mem_patterns, mem_active);
    }

    /// Collect vision + brain results. Blocks until both are complete.
    pub fn collect_blocking(&mut self) -> Option<UnifiedResult> {
        let vision = self.vision.collect_blocking()?;
        let mut encoded = Vec::new();
        let mut similarities = Vec::new();
        if !self.brain.collect_blocking_into(&mut encoded, &mut similarities) {
            return None;
        }
        Some(UnifiedResult {
            vision,
            encoded,
            similarities,
        })
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
            let row_base = i * feature_count;
            for j in 0..feature_count {
                sum += features[j] * weights[row_base + j];
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
