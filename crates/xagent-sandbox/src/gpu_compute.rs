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
//! On Apple Silicon, this shares the same physical GPU via unified memory — the
//! "upload/download" is effectively zero-copy (shared physical pages).
//!
//! Two compute passes per tick:
//! 1. **Encode**: features × weights + biases → tanh → encoded state (per agent)
//! 2. **Recall**: cosine similarity between encoded state and all memory patterns
//!    (per agent × per pattern), using workgroup-level parallel reduction
//!
//! # Performance
//!
//! For 10 agents the GPU dispatch overhead (~50-100μs) dominates the actual
//! compute time. GPU becomes advantageous above ~50 agents. Use `--gpu-brain`
//! to opt in; the default rayon CPU path is faster for small populations.

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

    // CPU-readable staging buffers
    encoded_staging: wgpu::Buffer,
    similarities_staging: wgpu::Buffer,

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

        let encoded_staging = make_staging(&device, "encoded-staging", encoded_size);
        let similarities_staging =
            make_staging(&device, "similarities-staging", similarities_size);

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
            encode_bind_group,
            recall_bind_group,
        })
    }

    /// Upload brain state for all agents and dispatch encode + recall on GPU.
    ///
    /// Returns `(encoded_states, similarity_scores)` packed contiguously:
    /// - `encoded_states`: `[num_agents × dim]` — the encoded representations
    /// - `similarity_scores`: `[num_agents × memory_capacity]` — cosine similarities
    ///   (inactive patterns marked as -2.0)
    ///
    /// # Arguments
    /// - `features`: `[num_agents × feature_count]` extracted sensory features
    /// - `enc_weights`: `[num_agents × feature_count × dim]` encoder weight matrices
    /// - `enc_biases`: `[num_agents × dim]` encoder bias vectors
    /// - `mem_patterns`: `[num_agents × memory_capacity × dim]` memory pattern data
    /// - `mem_active`: `[num_agents × memory_capacity]` active mask (1=active, 0=empty)
    pub fn dispatch(
        &self,
        features: &[f32],
        enc_weights: &[f32],
        enc_biases: &[f32],
        mem_patterns: &[f32],
        mem_active: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
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

        // Copy results to staging buffers for CPU readback
        let enc_size = (self.num_agents * self.dim) as u64 * 4;
        let sim_size = (self.num_agents * self.memory_capacity) as u64 * 4;
        cmd.copy_buffer_to_buffer(&self.encoded_buf, 0, &self.encoded_staging, 0, enc_size);
        cmd.copy_buffer_to_buffer(
            &self.similarities_buf,
            0,
            &self.similarities_staging,
            0,
            sim_size,
        );

        self.queue.submit(std::iter::once(cmd.finish()));

        // Read back results
        let encoded = read_staging(&self.device, &self.encoded_staging, enc_size);
        let similarities = read_staging(&self.device, &self.similarities_staging, sim_size);

        (encoded, similarities)
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

/// Synchronously map a staging buffer and read its contents as f32s.
fn read_staging(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Vec<f32> {
    let slice = buffer.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    rx.recv()
        .expect("GPU map channel closed")
        .expect("GPU buffer map failed");

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    result
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
        let gpu = match try_gpu(1, 4, 3, 2) {
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
        let gpu = match try_gpu(1, dim, 3, capacity) {
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
        let gpu = match try_gpu(n, dim, fc, cap) {
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
}
