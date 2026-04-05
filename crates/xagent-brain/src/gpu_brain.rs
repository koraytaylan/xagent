//! GPU-resident brain: all agent brain computation runs on GPU.
//!
//! Brain state lives permanently on GPU storage buffers. Each tick:
//! 1. CPU packs sensory frames and uploads (~52KB for 50 agents)
//! 2. GPU runs 7 compute passes in a single queue.submit()
//! 3. CPU reads back motor commands (~800 bytes for 50 agents)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use wgpu;
use xagent_shared::{BrainConfig, MotorCommand, SensoryFrame};

use crate::buffers::*;

pub struct GpuBrain {
    device: wgpu::Device,
    queue: wgpu::Queue,

    agent_count: u32,
    config: BrainConfig,

    // ── Persistent state (lives on GPU across ticks) ──
    brain_state_buf: wgpu::Buffer,
    pattern_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    config_buf: wgpu::Buffer,

    // ── Transient (overwritten each tick) ──
    sensory_buf: wgpu::Buffer,
    features_buf: wgpu::Buffer,
    encoded_buf: wgpu::Buffer,
    habituated_buf: wgpu::Buffer,
    homeo_out_buf: wgpu::Buffer,
    similarities_buf: wgpu::Buffer,
    recall_buf: wgpu::Buffer,
    decision_buf: wgpu::Buffer,

    // ── Readback staging ──
    motor_staging: [wgpu::Buffer; 2],
    staging_idx: usize,
    staging_mapped: [bool; 2],
    submit_seq: u64,
    mapped_seq: Arc<AtomicU64>,
    has_in_flight: bool,

    // ── Compute pipelines ──
    feature_extract_pipeline: wgpu::ComputePipeline,
    feature_extract_bind_group: wgpu::BindGroup,
    encode_pipeline: wgpu::ComputePipeline,
    encode_bind_group: wgpu::BindGroup,
    habituate_homeo_pipeline: wgpu::ComputePipeline,
    habituate_homeo_bind_group: wgpu::BindGroup,

    // ── Packing scratch ──
    sensory_scratch: Vec<f32>,
}

impl GpuBrain {
    fn create_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Create a GPU brain for `agent_count` agents.
    pub fn new(agent_count: u32, config: &BrainConfig) -> Self {
        let n = agent_count as usize;

        // ── wgpu device ──
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter found");

        log::info!("[GpuBrain] Adapter: {:?}", adapter.get_info());

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-brain"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // ── Persistent buffers ──
        let brain_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain_state"),
            size: (n * BRAIN_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let pattern_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("patterns"),
            size: (n * PATTERN_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("history"),
            size: (n * HISTORY_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("config"),
            size: (CONFIG_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Transient buffers ──
        let sensory_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sensory_input"),
            size: (n * SENSORY_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let features_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("features"),
            size: (n * FEATURES_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let encoded_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoded"),
            size: (n * ENCODED_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let habituated_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("habituated"),
            size: (n * HABITUATED_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let homeo_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("homeo_out"),
            size: (n * HOMEO_OUT_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let similarities_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("similarities"),
            size: (n * SIMILARITIES_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let recall_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("recall"),
            size: (n * RECALL_IDX_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });
        let decision_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decision"),
            size: (n * DECISION_STRIDE * 4) as u64,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // ── Readback staging (double-buffered) ──
        let motor_size = (n * 4 * 4) as u64; // N × 4 f32
        let motor_staging = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("motor_staging_0"),
                size: motor_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("motor_staging_1"),
                size: motor_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // ── Initialize persistent state ──
        let mut rng = rand::rng();
        let mut brain_data = Vec::with_capacity(n * BRAIN_STRIDE);
        let mut pattern_data = Vec::with_capacity(n * PATTERN_STRIDE);
        let mut history_data = Vec::with_capacity(n * HISTORY_STRIDE);
        for _ in 0..n {
            brain_data.extend_from_slice(&init_brain_state(config, &mut rng));
            pattern_data.extend_from_slice(&init_pattern_memory());
            history_data.extend_from_slice(&init_action_history());
        }
        queue.write_buffer(&brain_state_buf, 0, bytemuck::cast_slice(&brain_data));
        queue.write_buffer(&pattern_buf, 0, bytemuck::cast_slice(&pattern_data));
        queue.write_buffer(&history_buf, 0, bytemuck::cast_slice(&history_data));
        queue.write_buffer(&config_buf, 0, bytemuck::cast_slice(&build_config(config)));

        // ── Compute pipelines ──
        let constants = crate::buffers::wgsl_constants();
        let fe_source = format!("{}\n{}", constants, include_str!("shaders/feature_extract.wgsl"));
        let feature_extract_pipeline = Self::create_pipeline(&device, "feature_extract", &fe_source);
        let feature_extract_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("feature_extract_bg"),
            layout: &feature_extract_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sensory_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: features_buf.as_entire_binding() },
            ],
        });

        let enc_source = format!("{}\n{}", constants, include_str!("shaders/encode.wgsl"));
        let encode_pipeline = Self::create_pipeline(&device, "encode", &enc_source);
        let encode_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode_bg"),
            layout: &encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: features_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brain_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: encoded_buf.as_entire_binding() },
            ],
        });

        let hh_source = format!("{}\n{}", constants, include_str!("shaders/habituate_homeo.wgsl"));
        let habituate_homeo_pipeline = Self::create_pipeline(&device, "habituate_homeo", &hh_source);
        let habituate_homeo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("habituate_homeo_bg"),
            layout: &habituate_homeo_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: encoded_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sensory_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: brain_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: habituated_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: homeo_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: config_buf.as_entire_binding() },
            ],
        });

        Self {
            device,
            queue,
            agent_count,
            config: config.clone(),
            brain_state_buf,
            pattern_buf,
            history_buf,
            config_buf,
            sensory_buf,
            features_buf,
            encoded_buf,
            habituated_buf,
            homeo_out_buf,
            similarities_buf,
            recall_buf,
            decision_buf,
            motor_staging,
            staging_idx: 0,
            staging_mapped: [false, false],
            submit_seq: 0,
            mapped_seq: Arc::new(AtomicU64::new(0)),
            has_in_flight: false,
            feature_extract_pipeline,
            feature_extract_bind_group,
            encode_pipeline,
            encode_bind_group,
            habituate_homeo_pipeline,
            habituate_homeo_bind_group,
            sensory_scratch: vec![0.0; n * SENSORY_STRIDE],
        }
    }

    /// Number of agents this brain manages.
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Upload packed sensory data to GPU.
    pub fn upload_sensory(&mut self, frames: &[SensoryFrame]) {
        assert_eq!(frames.len(), self.agent_count as usize);
        for (i, frame) in frames.iter().enumerate() {
            let offset = i * SENSORY_STRIDE;
            pack_sensory_frame(frame, &mut self.sensory_scratch[offset..offset + SENSORY_STRIDE]);
        }
        self.queue.write_buffer(
            &self.sensory_buf,
            0,
            bytemuck::cast_slice(&self.sensory_scratch),
        );
    }

    /// Read back motor commands from staging buffer.
    pub fn read_motor_output(&mut self) -> Vec<MotorCommand> {
        let n = self.agent_count as usize;
        if !self.has_in_flight {
            return vec![MotorCommand::default(); n];
        }

        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let read_idx = 1 - self.staging_idx;
        let motor_size = (n * 4 * 4) as u64;
        let slice = self.motor_staging[read_idx].slice(..motor_size);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let mut commands = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 4;
            commands.push(MotorCommand {
                forward: floats[base],
                turn: floats[base + 1],
                strafe: floats[base + 2],
                action: None,
            });
        }
        drop(data);
        self.motor_staging[read_idx].unmap();
        self.staging_mapped[read_idx] = false;
        self.has_in_flight = false;

        commands
    }

    /// Read back full brain state for one agent (blocking GPU readback).
    pub fn read_agent_state(&mut self, index: u32) -> AgentBrainState {
        let i = index as usize;
        let mut state = AgentBrainState::new_blank();

        let brain_offset = (i * BRAIN_STRIDE * 4) as u64;
        let brain_size = (BRAIN_STRIDE * 4) as u64;
        self.read_buffer_range(&self.brain_state_buf, brain_offset, brain_size, &mut state.brain_state);

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        let pat_size = (PATTERN_STRIDE * 4) as u64;
        self.read_buffer_range(&self.pattern_buf, pat_offset, pat_size, &mut state.patterns);

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        let hist_size = (HISTORY_STRIDE * 4) as u64;
        self.read_buffer_range(&self.history_buf, hist_offset, hist_size, &mut state.history);

        state
    }

    /// Write full brain state for one agent (blocking GPU upload).
    pub fn write_agent_state(&mut self, index: u32, state: &AgentBrainState) {
        let i = index as usize;
        let brain_offset = (i * BRAIN_STRIDE * 4) as u64;
        self.queue.write_buffer(&self.brain_state_buf, brain_offset, bytemuck::cast_slice(&state.brain_state));

        let pat_offset = (i * PATTERN_STRIDE * 4) as u64;
        self.queue.write_buffer(&self.pattern_buf, pat_offset, bytemuck::cast_slice(&state.patterns));

        let hist_offset = (i * HISTORY_STRIDE * 4) as u64;
        self.queue.write_buffer(&self.history_buf, hist_offset, bytemuck::cast_slice(&state.history));
    }

    /// Signal agent death: halve all pattern reinforcements and reset homeostasis.
    pub fn death_signal(&mut self, index: u32) {
        let mut state = self.read_agent_state(index);

        // Halve all pattern reinforcements (trauma)
        let base = O_PAT_REINF;
        for j in 0..MEMORY_CAP {
            state.patterns[base + j] *= 0.5;
        }

        // Reset homeostasis EMAs
        let h = O_HOMEO;
        for j in 0..6 {
            state.brain_state[h + j] = 0.0;
        }

        // Reset action history
        state.history = init_action_history();

        // Reset exploration rate to 0.5
        state.brain_state[O_EXPLORATION_RATE] = 0.5;

        self.write_agent_state(index, &state);
    }

    /// Dispatch feature extraction pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_feature_extract(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_feature_extract"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("feature_extract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.feature_extract_pipeline);
            pass.set_bind_group(0, &self.feature_extract_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch encode pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_encode(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encode"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.encode_pipeline);
            pass.set_bind_group(0, &self.encode_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch habituate+homeo pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_habituate_homeo(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_habituate_homeo"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("habituate_homeo"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.habituate_homeo_pipeline);
            pass.set_bind_group(0, &self.habituate_homeo_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Helper: blocking read of a buffer range into a pre-sized Vec<f32>.
    pub(crate) fn read_buffer_range(&self, buffer: &wgpu::Buffer, offset: u64, size: u64, out: &mut Vec<f32>) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_copy"),
        });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        out.clear();
        out.extend_from_slice(floats);
        drop(data);
        staging.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_brain_initializes() {
        let config = BrainConfig::default();
        let brain = GpuBrain::new(2, &config);
        assert_eq!(brain.agent_count(), 2);
    }

    #[test]
    fn read_write_state_roundtrip() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(2, &config);

        let state0 = brain.read_agent_state(0);
        assert_eq!(state0.brain_state.len(), BRAIN_STRIDE);
        assert_eq!(state0.patterns.len(), PATTERN_STRIDE);

        let mut modified = state0.clone();
        modified.brain_state[O_ENC_BIASES] = 42.0;
        brain.write_agent_state(0, &modified);

        let readback = brain.read_agent_state(0);
        assert_eq!(readback.brain_state[O_ENC_BIASES], 42.0);

        let state1 = brain.read_agent_state(1);
        assert_ne!(state1.brain_state[O_ENC_BIASES], 42.0);
    }

    #[test]
    fn feature_extract_produces_correct_output() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        // Create a frame with known values
        let mut frame = SensoryFrame::new_blank(8, 6);
        frame.vision.color[0] = 0.5; // First pixel R
        frame.vision.color[1] = 0.3; // First pixel G
        frame.velocity = glam::Vec3::new(3.0, 4.0, 0.0); // magnitude = 5.0
        frame.energy_signal = 0.8;

        brain.upload_sensory(&[frame]);
        brain.run_feature_extract();

        // Read back features
        let mut features = vec![0.0_f32; FEATURES_STRIDE];
        brain.read_buffer_range(&brain.features_buf, 0, (FEATURES_STRIDE * 4) as u64, &mut features);

        // Vision color should be copied directly
        assert!((features[0] - 0.5).abs() < 0.001, "pixel R");
        assert!((features[1] - 0.3).abs() < 0.001, "pixel G");

        // Velocity magnitude at index 192
        assert!((features[192] - 5.0).abs() < 0.01, "vel magnitude");

        // Energy at index 197 (192 + vel_mag(1) + facing(3) + ang_vel(1) = 197)
        assert!((features[197] - 0.8).abs() < 0.001, "energy");
    }

    #[test]
    fn death_signal_halves_reinforcement() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        let mut state = brain.read_agent_state(0);
        state.patterns[O_PAT_ACTIVE] = 1.0;
        state.patterns[O_PAT_REINF] = 2.0;
        state.patterns[O_ACTIVE_COUNT] = 1.0;
        brain.write_agent_state(0, &state);

        brain.death_signal(0);

        let after = brain.read_agent_state(0);
        assert!((after.patterns[O_PAT_REINF] - 1.0).abs() < 0.01);

        for j in 0..6 {
            assert_eq!(after.brain_state[O_HOMEO + j], 0.0);
        }
    }

    #[test]
    fn encode_produces_tanh_output() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        // Set up identity-like weights: weight[0][d] = 1.0 for all d, rest = 0
        // This means encoded[d] = fast_tanh(features[0] * 1.0 + bias[d])
        let mut state = brain.read_agent_state(0);
        // Zero all encoder weights
        for i in 0..(FEATURE_COUNT * DIM) {
            state.brain_state[O_ENC_WEIGHTS + i] = 0.0;
        }
        // Set weight[feature=0, dim=d] = 1.0 for all d
        for d in 0..DIM {
            state.brain_state[O_ENC_WEIGHTS + 0 * DIM + d] = 1.0;
        }
        // Zero biases
        for d in 0..DIM {
            state.brain_state[O_ENC_BIASES + d] = 0.0;
        }
        brain.write_agent_state(0, &state);

        // Create a frame with known first pixel
        let mut frame = SensoryFrame::new_blank(8, 6);
        frame.vision.color[0] = 0.5; // features[0] = 0.5
        brain.upload_sensory(&[frame]);
        brain.run_feature_extract();
        brain.run_encode();

        // Read back encoded
        let mut encoded = vec![0.0_f32; DIM];
        brain.read_buffer_range(&brain.encoded_buf, 0, (DIM * 4) as u64, &mut encoded);

        // All dims should be fast_tanh(0.5)
        let expected = crate::fast_tanh(0.5);
        for d in 0..DIM {
            assert!((encoded[d] - expected).abs() < 0.01,
                "dim {} expected {} got {}", d, expected, encoded[d]);
        }
    }

    #[test]
    fn habituation_attenuates_repeated_input() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        let frame = SensoryFrame::new_blank(8, 6);

        // Run feature_extract + encode + habituate_homeo multiple times with constant input
        for _ in 0..100 {
            brain.upload_sensory(&[frame.clone()]);
            brain.run_feature_extract();
            brain.run_encode();
            brain.run_habituate_homeo();
        }

        // Read back attenuation values from brain_state
        let state = brain.read_agent_state(0);
        let mut mean_atten = 0.0;
        for d in 0..DIM {
            mean_atten += state.brain_state[O_HAB_ATTEN + d];
        }
        mean_atten /= DIM as f32;

        // Constant input should drive attenuation down toward floor (0.1)
        assert!(mean_atten < 0.5, "Constant input should reduce attenuation, got mean={}", mean_atten);
    }
}
