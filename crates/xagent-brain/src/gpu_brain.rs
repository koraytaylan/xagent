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

#[allow(dead_code)] // Transient GPU buffers are only read via bind groups, not Rust field access
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
    recall_score_pipeline: wgpu::ComputePipeline,
    recall_score_bind_group: wgpu::BindGroup,
    recall_topk_pipeline: wgpu::ComputePipeline,
    recall_topk_bind_group: wgpu::BindGroup,
    predict_act_pipeline: wgpu::ComputePipeline,
    predict_act_bind_group: wgpu::BindGroup,
    learn_store_pipeline: wgpu::ComputePipeline,
    learn_store_bind_group: wgpu::BindGroup,

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
        .or_else(|| {
            log::warn!("[GpuBrain] No GPU adapter found, trying fallback (CPU) adapter");
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .expect("No GPU or fallback adapter found");

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

        let rs_source = format!("{}\n{}", constants, include_str!("shaders/recall_score.wgsl"));
        let recall_score_pipeline = Self::create_pipeline(&device, "recall_score", &rs_source);
        let recall_score_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("recall_score_bg"),
            layout: &recall_score_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: habituated_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: similarities_buf.as_entire_binding() },
            ],
        });

        let rt_source = format!("{}\n{}", constants, include_str!("shaders/recall_topk.wgsl"));
        let recall_topk_pipeline = Self::create_pipeline(&device, "recall_topk", &rt_source);
        let recall_topk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("recall_topk_bg"),
            layout: &recall_topk_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: similarities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: recall_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: brain_state_buf.as_entire_binding() },
            ],
        });

        let pa_source = format!("{}\n{}", constants, include_str!("shaders/predict_and_act.wgsl"));
        let predict_act_pipeline = Self::create_pipeline(&device, "predict_and_act", &pa_source);
        let predict_act_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("predict_act_bg"),
            layout: &predict_act_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: habituated_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brain_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: recall_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: homeo_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: history_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: decision_buf.as_entire_binding() },
            ],
        });

        let ls_source = format!("{}\n{}", constants, include_str!("shaders/learn_and_store.wgsl"));
        let learn_store_pipeline = Self::create_pipeline(&device, "learn_and_store", &ls_source);
        let learn_store_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("learn_store_bg"),
            layout: &learn_store_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: habituated_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: features_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: brain_state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pattern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: decision_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: homeo_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: config_buf.as_entire_binding() },
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
            recall_score_pipeline,
            recall_score_bind_group,
            recall_topk_pipeline,
            recall_topk_bind_group,
            predict_act_pipeline,
            predict_act_bind_group,
            learn_store_pipeline,
            learn_store_bind_group,
            sensory_scratch: vec![0.0; n * SENSORY_STRIDE],
        }
    }

    /// Number of agents this brain manages.
    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    /// Resize for a new population size. Reallocates all buffers.
    /// Called between generations when population_size changes.
    pub fn resize(&mut self, agent_count: u32) {
        *self = Self::new(agent_count, &self.config.clone());
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

    /// Upload sensory frames, run full brain tick (7 passes), return motor commands.
    pub fn tick(&mut self, frames: &[SensoryFrame]) -> Vec<MotorCommand> {
        self.submit(frames);
        self.collect()
    }

    /// Non-blocking submit: upload sensory data, dispatch all 7 passes.
    pub fn submit(&mut self, frames: &[SensoryFrame]) {
        self.upload_sensory(frames);

        let widx = self.staging_idx;
        if self.staging_mapped[widx] {
            self.motor_staging[widx].unmap();
            self.staging_mapped[widx] = false;
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brain_tick"),
        });

        // Pass 1: feature_extract
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.feature_extract_pipeline);
            pass.set_bind_group(0, &self.feature_extract_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 2: encode
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.encode_pipeline);
            pass.set_bind_group(0, &self.encode_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 3: habituate_homeo
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.habituate_homeo_pipeline);
            pass.set_bind_group(0, &self.habituate_homeo_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 4: recall_score
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.recall_score_pipeline);
            pass.set_bind_group(0, &self.recall_score_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 5: recall_topk
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.recall_topk_pipeline);
            pass.set_bind_group(0, &self.recall_topk_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 6: predict_and_act
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.predict_act_pipeline);
            pass.set_bind_group(0, &self.predict_act_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        // Pass 7: learn_and_store
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.learn_store_pipeline);
            pass.set_bind_group(0, &self.learn_store_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }

        // Copy motor output from decision_buf to staging buffer.
        // Motor is at offset DIM+DIM (=64 f32) within each agent's DECISION_STRIDE block,
        // so the data is not contiguous across agents. Issue N copy commands to gather it.
        let n = self.agent_count as usize;
        for i in 0..n {
            let src_offset = ((i * DECISION_STRIDE + DIM + DIM) * 4) as u64;
            let dst_offset = (i * 4 * 4) as u64; // 4 f32 * 4 bytes per agent
            encoder.copy_buffer_to_buffer(
                &self.decision_buf,
                src_offset,
                &self.motor_staging[widx],
                dst_offset,
                16, // 4 f32 * 4 bytes
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer for reading
        let motor_size = (n * 4 * 4) as u64;
        let seq = self.submit_seq + 1;
        self.submit_seq = seq;
        let flag = self.mapped_seq.clone();
        self.motor_staging[widx]
            .slice(..motor_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(seq, Ordering::Release);
                }
            });
        self.staging_mapped[widx] = true;
        self.staging_idx = 1 - self.staging_idx;
        self.has_in_flight = true;
    }

    /// Blocking collect: wait for GPU and return motor commands.
    pub fn collect(&mut self) -> Vec<MotorCommand> {
        self.read_motor_output()
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

    /// Dispatch recall_score pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_recall_score(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_recall_score"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("recall_score"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.recall_score_pipeline);
            pass.set_bind_group(0, &self.recall_score_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch recall_topk pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_recall_topk(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_recall_topk"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("recall_topk"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.recall_topk_pipeline);
            pass.set_bind_group(0, &self.recall_topk_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch predict_and_act pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_predict_and_act(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_predict_and_act"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predict_and_act"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.predict_act_pipeline);
            pass.set_bind_group(0, &self.predict_act_bind_group, &[]);
            pass.dispatch_workgroups(self.agent_count, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch learn_and_store pass (test-only).
    #[cfg(test)]
    pub(crate) fn run_learn_and_store(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_learn_and_store"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("learn_and_store"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.learn_store_pipeline);
            pass.set_bind_group(0, &self.learn_store_bind_group, &[]);
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

    #[test]
    fn recall_score_computes_cosine_similarity() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        // Write a known pattern to slot 0 that matches a known query
        let mut state = brain.read_agent_state(0);

        // Set pattern slot 0 as active with a known state
        state.patterns[O_PAT_ACTIVE] = 1.0;  // slot 0 active
        state.patterns[O_ACTIVE_COUNT] = 1.0;

        // Pattern state: all 1.0 (norm = sqrt(32))
        for d in 0..DIM {
            state.patterns[O_PAT_STATES + d] = 1.0;
        }
        state.patterns[O_PAT_NORMS] = (DIM as f32).sqrt(); // pre-cached norm

        brain.write_agent_state(0, &state);

        // Upload habituated data directly: all 1.0 (same direction as pattern)
        let hab_data: Vec<f32> = vec![1.0; DIM];
        brain.queue.write_buffer(&brain.habituated_buf, 0, bytemuck::cast_slice(&hab_data));

        brain.run_recall_score();

        // Read similarities
        let mut sims = vec![0.0_f32; MEMORY_CAP];
        brain.read_buffer_range(&brain.similarities_buf, 0, (MEMORY_CAP * 4) as u64, &mut sims);

        // Slot 0 should have similarity ≈ 1.0 (identical direction)
        assert!((sims[0] - 1.0).abs() < 0.01, "slot 0 sim should be ~1.0, got {}", sims[0]);

        // Slot 1 should be -2.0 (inactive)
        assert!((sims[1] - (-2.0)).abs() < 0.01, "slot 1 should be -2.0, got {}", sims[1]);
    }

    #[test]
    fn recall_topk_selects_best_patterns() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);

        // Write similarities directly: slot 5 = 0.9, slot 10 = 0.7, rest = -2.0
        let mut sims = vec![-2.0_f32; MEMORY_CAP];
        sims[5] = 0.9;
        sims[10] = 0.7;
        brain.queue.write_buffer(&brain.similarities_buf, 0, bytemuck::cast_slice(&sims));

        // Mark slots 5 and 10 as active in patterns (needed for metadata update)
        let mut state = brain.read_agent_state(0);
        state.patterns[O_PAT_ACTIVE + 5] = 1.0;
        state.patterns[O_PAT_ACTIVE + 10] = 1.0;
        state.patterns[O_ACTIVE_COUNT] = 2.0;
        brain.write_agent_state(0, &state);

        brain.run_recall_topk();

        // Read recall buffer
        let mut recall = vec![0.0_f32; RECALL_IDX_STRIDE];
        brain.read_buffer_range(&brain.recall_buf, 0, (RECALL_IDX_STRIDE * 4) as u64, &mut recall);

        // Count should be 2
        let count = recall[RECALL_K] as u32;
        assert_eq!(count, 2, "should recall 2 patterns, got {}", count);

        // First should be slot 5 (highest sim), second slot 10
        assert_eq!(recall[0] as u32, 5, "first recalled should be slot 5");
        assert_eq!(recall[1] as u32, 10, "second recalled should be slot 10");
    }

    #[test]
    fn predict_and_act_produces_valid_motor() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(2, &config);
        let frames: Vec<SensoryFrame> = (0..2)
            .map(|_| SensoryFrame::new_blank(8, 6))
            .collect();

        // Run full pipeline up to predict_and_act
        brain.upload_sensory(&frames);
        brain.run_feature_extract();
        brain.run_encode();
        brain.run_habituate_homeo();
        brain.run_recall_score();
        brain.run_recall_topk();
        brain.run_predict_and_act();

        // Read decision buffer
        let total = 2 * DECISION_STRIDE;
        let mut dec = vec![0.0_f32; total];
        brain.read_buffer_range(&brain.decision_buf, 0, (total * 4) as u64, &mut dec);

        // Motor output should be finite and in [-1, 1]
        for agent in 0..2 {
            let base = agent * DECISION_STRIDE;
            let fwd = dec[base + DIM + DIM];
            let trn = dec[base + DIM + DIM + 1];
            assert!(fwd.is_finite(), "agent {} forward not finite: {}", agent, fwd);
            assert!(trn.is_finite(), "agent {} turn not finite: {}", agent, trn);
            assert!(
                fwd >= -1.0 && fwd <= 1.0,
                "agent {} forward out of range: {}",
                agent,
                fwd
            );
            assert!(
                trn >= -1.0 && trn <= 1.0,
                "agent {} turn out of range: {}",
                agent,
                trn
            );
        }
    }

    #[test]
    fn learn_and_store_modifies_weights_and_stores_pattern() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);
        let frame = SensoryFrame::new_blank(8, 6);

        // Run full pipeline through all 7 passes
        brain.upload_sensory(&[frame.clone()]);
        brain.run_feature_extract();
        brain.run_encode();
        brain.run_habituate_homeo();
        brain.run_recall_score();
        brain.run_recall_topk();
        brain.run_predict_and_act();
        brain.run_learn_and_store();

        let after = brain.read_agent_state(0);

        // Pattern should have been stored (active_count should be > 0)
        let active = after.patterns[O_ACTIVE_COUNT];
        assert!(active >= 1.0, "should have stored at least one pattern, active_count={}", active);

        // Verify a pattern slot is active
        let stored_idx = after.patterns[O_LAST_STORED_IDX] as usize;
        assert_eq!(after.patterns[O_PAT_ACTIVE + stored_idx], 1.0, "stored slot should be active");
    }

    #[test]
    fn tick_produces_valid_motor_commands() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(4, &config);
        let frames: Vec<SensoryFrame> = (0..4)
            .map(|_| SensoryFrame::new_blank(8, 6))
            .collect();

        for _ in 0..100 {
            let commands = brain.tick(&frames);
            assert_eq!(commands.len(), 4);
            for cmd in &commands {
                assert!(cmd.forward.is_finite(), "forward must be finite");
                assert!(cmd.turn.is_finite(), "turn must be finite");
                assert!(cmd.forward >= -1.0 && cmd.forward <= 1.0, "forward in range: {}", cmd.forward);
                assert!(cmd.turn >= -1.0 && cmd.turn <= 1.0, "turn in range: {}", cmd.turn);
            }
        }
    }

    #[test]
    fn learning_changes_weights() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);
        let frame = SensoryFrame::new_blank(8, 6);

        let before = brain.read_agent_state(0);

        for _ in 0..50 {
            brain.tick(&[frame.clone()]);
        }

        let after = brain.read_agent_state(0);

        // Predictor weights should shift due to gradient descent on prediction error
        let weight_delta: f32 = before.brain_state[O_PRED_WEIGHTS..O_PRED_WEIGHTS + 100]
            .iter()
            .zip(&after.brain_state[O_PRED_WEIGHTS..O_PRED_WEIGHTS + 100])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(weight_delta > 0.0, "predictor weights should change after learning");
    }

    #[test]
    fn memory_fills_over_time() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(1, &config);
        let frame = SensoryFrame::new_blank(8, 6);

        for _ in 0..200 {
            brain.tick(&[frame.clone()]);
        }

        let state = brain.read_agent_state(0);
        let active = state.patterns[O_ACTIVE_COUNT];
        assert!(active > 0.0, "memory should have stored patterns after 200 ticks, got active_count={}", active);
    }

    #[test]
    fn resize_changes_agent_count() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(10, &config);
        assert_eq!(brain.agent_count(), 10);
        brain.resize(20);
        assert_eq!(brain.agent_count(), 20);
    }

    #[test]
    fn agents_produce_varied_motor_output() {
        let config = BrainConfig::default();
        let mut brain = GpuBrain::new(10, &config);
        let frames: Vec<SensoryFrame> = (0..10)
            .map(|_| SensoryFrame::new_blank(8, 6))
            .collect();

        let mut all_same = true;
        for _ in 0..50 {
            let commands = brain.tick(&frames);
            let first = commands[0].forward;
            if commands.iter().any(|c| (c.forward - first).abs() > 0.01) {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "different agents should produce varied motor output");
    }
}
