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

    // ── Packing scratch ──
    sensory_scratch: Vec<f32>,
}

impl GpuBrain {
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
}
