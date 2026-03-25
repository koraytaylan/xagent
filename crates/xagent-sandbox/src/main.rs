use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use glam::Vec3;
use log::info;
use rand::Rng;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};
use xagent_brain::Brain;
use xagent_shared::{BrainConfig, FullConfig, WorldConfig};

use xagent_sandbox::agent::senses::OtherAgent;
use xagent_sandbox::agent::{
    mutate_config, senses, Agent, AgentBody, MAX_AGENTS,
};
use xagent_sandbox::recording::MetricsLogger;
use xagent_sandbox::renderer::camera::Camera;
use xagent_sandbox::renderer::font::TextItem;
use xagent_sandbox::renderer::hud::HudBar;
use xagent_sandbox::renderer::{GpuMesh, InstanceData, Renderer};

use xagent_sandbox::world::WorldState;

// ── CLI ────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "xagent", about = "Emergent Cognitive Agent Sandbox")]
struct Cli {
    /// Brain preset: tiny, default, large
    #[arg(long, default_value = "default")]
    brain_preset: String,

    /// World preset: easy, normal, hard
    #[arg(long, default_value = "normal")]
    world_preset: String,

    /// Load full config from JSON file
    #[arg(long)]
    config: Option<String>,

    /// Random seed for world generation
    #[arg(long)]
    seed: Option<u64>,

    /// Simulation ticks per second
    #[arg(long)]
    tick_rate: Option<f32>,

    /// Run headless (no window, just simulation + logging)
    #[arg(long)]
    no_render: bool,

    /// Print current config as JSON to stdout and exit
    #[arg(long)]
    dump_config: bool,

    /// Enable CSV metrics logging to file
    #[arg(long)]
    log: bool,
}

fn resolve_config(cli: &Cli) -> FullConfig {
    let mut config = if let Some(path) = &cli.config {
        let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error: Failed to read config file '{}': {}", path, e);
            std::process::exit(1);
        });
        serde_json::from_str::<FullConfig>(&data).unwrap_or_else(|e| {
            eprintln!("Error: Failed to parse config file '{}': {}", path, e);
            std::process::exit(1);
        })
    } else {
        let brain = match cli.brain_preset.as_str() {
            "tiny" => BrainConfig::tiny(),
            "default" => BrainConfig::default(),
            "large" => BrainConfig::large(),
            other => {
                eprintln!("Error: Unknown brain preset '{}'. Choose: tiny, default, large", other);
                std::process::exit(1);
            }
        };
        let world = match cli.world_preset.as_str() {
            "easy" => WorldConfig::easy(),
            "normal" => WorldConfig::default(),
            "hard" => WorldConfig::hard(),
            other => {
                eprintln!("Error: Unknown world preset '{}'. Choose: easy, normal, hard", other);
                std::process::exit(1);
            }
        };
        FullConfig { brain, world }
    };

    // CLI overrides
    if let Some(seed) = cli.seed {
        config.world.seed = seed;
    }
    if let Some(tick_rate) = cli.tick_rate {
        config.world.tick_rate = tick_rate;
    }

    config
}

fn print_config(config: &FullConfig) {
    println!("── Active Configuration ──────────────────────────");
    println!(
        "  Brain: capacity={} slots={} vis={} dim={} lr={} decay={}",
        config.brain.memory_capacity,
        config.brain.processing_slots,
        config.brain.visual_encoding_size,
        config.brain.representation_dim,
        config.brain.learning_rate,
        config.brain.decay_rate,
    );
    println!(
        "  World: size={} depletion={} move_cost={} hazard={} regen={} food_val={} food_den={} tick_rate={} seed={}",
        config.world.world_size,
        config.world.energy_depletion_rate,
        config.world.movement_energy_cost,
        config.world.hazard_damage_rate,
        config.world.integrity_regen_rate,
        config.world.food_energy_value,
        config.world.food_density,
        config.world.tick_rate,
        config.world.seed,
    );
    println!("──────────────────────────────────────────────────");
}

const RESPAWN_COOLDOWN_FRAMES: u32 = 60;

/// Base simulation rate in Hz. The simulation runs at this rate (scaled by
/// speed multiplier) regardless of rendering frame rate.
const SIM_RATE: f32 = 60.0;
const SIM_DT: f32 = 1.0 / SIM_RATE;

struct App {
    brain_config: BrainConfig,
    world_config: WorldConfig,

    renderer: Option<Renderer>,
    window: Option<Arc<Window>>,
    camera: Camera,

    // Static terrain mesh (built once)
    terrain_gpu: Option<GpuMesh>,

    // Simulation
    world: Option<WorldState>,
    agents: Vec<Agent>,
    next_agent_id: u32,
    tick: u64,
    food_dirty: bool,
    food_gpu: Option<GpuMesh>,

    last_frame: Instant,

    /// Fixed-timestep accumulator — simulation ticks run at SIM_RATE Hz,
    /// decoupled from the render frame rate.
    sim_accumulator: f32,

    // Speed controls (multiplier for simulation rate)
    speed_multiplier: u32,
    paused: bool,

    // Brain persistence default
    persist_brain: bool,

    // Session statistics
    total_prediction_error: f64,
    error_count: u64,

    // Telemetry selection
    selected_agent_idx: usize,

    // CSV metrics logger
    logger: Option<MetricsLogger>,

    // GPU instancing for agents
    agent_instance_buffer: Option<wgpu::Buffer>,
    agent_instance_count: u32,

    // FPS tracking
    frame_times: VecDeque<Instant>,
    fps: f32,

    // Cached HUD data — only rebuilt when a simulation tick runs
    cached_hud_bars: Vec<HudBar>,
    hud_dirty: bool,

    // Cursor position for click-to-select
    cursor_pos: (f64, f64),

    // Heatmap overlay
    heatmap_enabled: bool,
    heatmap_gpu: Option<GpuMesh>,

    // Trail overlay for selected agent
    trail_gpu: Option<GpuMesh>,

    // Selection marker above focused agent
    marker_gpu: Option<GpuMesh>,
}

impl App {
    fn new(brain_config: BrainConfig, world_config: WorldConfig, enable_logging: bool) -> Self {
        let logger = if enable_logging {
            match MetricsLogger::new() {
                Ok(l) => {
                    println!("[RECORDING] Logging metrics to: {}", l.file_name);
                    Some(l)
                }
                Err(e) => {
                    eprintln!("[RECORDING] Failed to create metrics log: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            brain_config,
            world_config,
            renderer: None,
            window: None,
            camera: Camera::new(16.0 / 9.0),
            terrain_gpu: None,
            world: None,
            agents: Vec::new(),
            next_agent_id: 0,
            tick: 0,
            food_dirty: true,
            food_gpu: None,
            last_frame: Instant::now(),
            sim_accumulator: 0.0,
            speed_multiplier: 1,
            paused: false,
            persist_brain: true,
            total_prediction_error: 0.0,
            error_count: 0,
            selected_agent_idx: 0,
            logger,
            agent_instance_buffer: None,
            agent_instance_count: 0,
            frame_times: VecDeque::new(),
            fps: 0.0,
            cached_hud_bars: Vec::new(),
            hud_dirty: true,
            cursor_pos: (0.0, 0.0),
            heatmap_enabled: false,
            heatmap_gpu: None,
            trail_gpu: None,
            marker_gpu: None,
        }
    }

    /// Spawn a new agent with the given BrainConfig at a random position.
    fn spawn_agent(&mut self, config: BrainConfig, generation: u32) {
        if self.agents.len() >= MAX_AGENTS {
            println!(
                "[SPAWN] Max agents ({}) reached, cannot spawn more",
                MAX_AGENTS
            );
            return;
        }
        let Some(world) = &self.world else { return };

        let mut rng = rand::rng();
        let half = world.config.world_size / 2.0 - 5.0;
        let x: f32 = rng.random_range(-half..half);
        let z: f32 = rng.random_range(-half..half);
        let y = world.terrain.height_at(x, z) + 1.0;

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let mut agent = Agent::new(id, Vec3::new(x, y, z), config, self.tick);
        agent.generation = generation;
        agent.persist_brain = self.persist_brain;

        let color = agent.color;
        println!(
            "[SPAWN] Agent {} (gen {}) spawned at ({:.1}, {:.1}) — color: ({:.2}, {:.2}, {:.2})",
            id, generation, x, z, color[0], color[1], color[2]
        );

        self.agents.push(agent);
    }

    /// Spawn a child agent near a parent, with mutated config.
    #[allow(dead_code)]
    fn spawn_child(&mut self, parent_idx: usize) {
        if self.agents.len() >= MAX_AGENTS {
            return;
        }
        let parent = &self.agents[parent_idx];
        let parent_id = parent.id;
        let parent_gen = parent.generation;
        let parent_pos = parent.body.body.position;
        let parent_config = parent.brain.config.clone();

        let child_config = mutate_config(&parent_config);
        let mut rng = rand::rng();
        let offset = Vec3::new(
            rng.random_range(-5.0..5.0_f32),
            0.0,
            rng.random_range(-5.0..5.0_f32),
        );
        let child_pos = parent_pos + offset;

        let id = self.next_agent_id;
        self.next_agent_id += 1;

        let Some(world) = &self.world else { return };
        let half = world.config.world_size / 2.0 - 1.0;
        let cx = child_pos.x.clamp(-half, half);
        let cz = child_pos.z.clamp(-half, half);
        let cy = world.terrain.height_at(cx, cz) + 1.0;

        let mut child = Agent::new(id, Vec3::new(cx, cy, cz), child_config, self.tick);
        child.generation = parent_gen + 1;
        child.persist_brain = self.persist_brain;

        println!(
            "[REPRODUCE] Agent {} (gen {}) → child Agent {} (gen {}) at ({:.1}, {:.1})",
            parent_id, parent_gen, id, child.generation, cx, cz
        );
        println!(
            "  Child config: cap={} slots={} dim={} lr={:.4} decay={:.4}",
            child.brain.config.memory_capacity,
            child.brain.config.processing_slots,
            child.brain.config.representation_dim,
            child.brain.config.learning_rate,
            child.brain.config.decay_rate,
        );

        self.agents.push(child);
    }

    /// Pick the agent closest to the cursor via screen-space projection.
    fn pick_agent_at_cursor(&mut self) {
        let Some(renderer) = &self.renderer else { return };
        let w = renderer.config.width as f32;
        let h = renderer.config.height as f32;
        if w < 1.0 || h < 1.0 { return; }

        // Normalized device coordinates [-1, 1]
        let ndc_x = (self.cursor_pos.0 as f32 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.cursor_pos.1 as f32 / h) * 2.0;

        let vp = self.camera.view_projection_matrix();
        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = f32::MAX;

        for (i, agent) in self.agents.iter().enumerate() {
            if !agent.body.body.alive { continue; }
            let pos = agent.body.body.position;
            let clip = vp * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w <= 0.0 { continue; } // behind camera
            let sx = clip.x / clip.w;
            let sy = clip.y / clip.w;
            let d = (sx - ndc_x).powi(2) + (sy - ndc_y).powi(2);
            if d < best_dist_sq {
                best_dist_sq = d;
                best_idx = Some(i);
            }
        }

        // Only select if click is reasonably close (within ~10% of screen)
        if let Some(idx) = best_idx {
            if best_dist_sq < 0.05 {
                self.selected_agent_idx = idx;
                self.agents[idx].trail_dirty = true;
                let a = &self.agents[idx];
                println!(
                    "[SELECT] Agent {} (gen {}) — phase: {}",
                    a.id, a.generation, a.brain.telemetry().behavior_phase()
                );
                self.hud_dirty = true;
            }
        }
    }

    fn print_session_summary(&mut self) {
        if let Some(logger) = &mut self.logger {
            let _ = logger.flush();
        }

        let avg_err = if self.error_count > 0 {
            self.total_prediction_error / self.error_count as f64
        } else {
            0.0
        };

        let total_deaths: u32 = self.agents.iter().map(|a| a.death_count).sum();

        let log_name = self
            .logger
            .as_ref()
            .map(|l| l.file_name.as_str())
            .unwrap_or("(none)");

        println!();
        println!("=== xagent Session Summary ===");
        println!("Total ticks: {}", self.tick);
        println!("Total agents spawned: {}", self.next_agent_id);
        println!("Living agents: {}", self.agents.len());
        println!("Total deaths: {}", total_deaths);
        println!("Avg prediction error: {:.2}", avg_err);
        println!("Log file: {}", log_name);
        println!();
    }

    /// Build HUD overlay bars for the selected agent.
    fn build_hud_bars(&self) -> Vec<HudBar> {
        let Some(agent) = self.agents.get(self.selected_agent_idx) else {
            return Vec::new();
        };

        let energy = agent.body.body.internal.energy_signal();
        let integrity = agent.body.body.internal.integrity_signal();
        let pred_err = agent.brain.telemetry().prediction_error.clamp(0.0, 1.0);
        let explore = agent.brain.telemetry().exploration_rate.clamp(0.0, 1.0);

        let bar_w = 0.35;
        let bar_h = 0.025;
        let left = -0.98;
        let top = 0.97;
        let gap = 0.035;
        let bg = [0.15, 0.15, 0.15];

        vec![
            // Energy — green
            HudBar {
                x: left,
                y: top,
                width: bar_w,
                height: bar_h,
                fill: energy,
                color: [0.2, 0.85, 0.2],
                bg_color: bg,
            },
            // Integrity — blue
            HudBar {
                x: left,
                y: top - gap,
                width: bar_w,
                height: bar_h,
                fill: integrity,
                color: [0.3, 0.5, 1.0],
                bg_color: bg,
            },
            // Prediction error — red
            HudBar {
                x: left,
                y: top - gap * 2.0,
                width: bar_w,
                height: bar_h,
                fill: pred_err,
                color: [0.95, 0.2, 0.15],
                bg_color: bg,
            },
            // Exploration rate — yellow
            HudBar {
                x: left,
                y: top - gap * 3.0,
                width: bar_w,
                height: bar_h,
                fill: explore,
                color: [0.95, 0.85, 0.1],
                bg_color: bg,
            },
        ]
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Spin the event loop continuously for uncapped frame rate
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("xagent — Emergent Cognitive Agent Sandbox")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window =
            Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height.max(1) as f32;

        let renderer = Renderer::new(window.clone());

        // ── world ──────────────────────────────────────────────────
        let world = WorldState::new(self.world_config.clone());

        let terrain_mesh = world.terrain_mesh();
        self.terrain_gpu = Some(GpuMesh::from_mesh(&renderer.device, &terrain_mesh));

        // Dynamic food buffer — max ~500 food items × 24 vertices each
        let food_mesh = world.food_mesh();
        let mut food_gpu = GpuMesh::new_dynamic(&renderer.device, 12000, 18000);
        food_gpu.update_from_mesh(&renderer.queue, &food_mesh);
        self.food_gpu = Some(food_gpu);
        self.food_dirty = false;

        // Dynamic heatmap overlay buffer — max 4096 cells × 4 vertices each
        let heatmap_res = xagent_sandbox::agent::HEATMAP_RES;
        let max_verts = (heatmap_res * heatmap_res * 4) as u64;
        let max_idx = (heatmap_res * heatmap_res * 6) as u64;
        self.heatmap_gpu = Some(GpuMesh::new_dynamic(&renderer.device, max_verts, max_idx));

        // Trail overlay: linear ribbon, up to MAX_TRAIL_POINTS control points.
        // Each segment = 4 verts, 6 indices.
        let max_trail_segs = xagent_sandbox::agent::MAX_TRAIL_POINTS as u64;
        self.trail_gpu = Some(GpuMesh::new_dynamic(
            &renderer.device,
            max_trail_segs * 4,
            max_trail_segs * 6,
        ));

        // Selection marker: diamond (24 verts, 24 indices)
        self.marker_gpu = Some(GpuMesh::new_dynamic(&renderer.device, 24, 24));

        // Pre-allocate agent instance buffer
        let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_instance_buffer"),
            size: (MAX_AGENTS * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.agent_instance_buffer = Some(instance_buffer);

        self.world = Some(world);
        self.renderer = Some(renderer);
        self.window = Some(window);

        // ── spawn initial agent ────────────────────────────────────
        self.spawn_agent(self.brain_config.clone(), 0);

        self.tick = 0;
        self.last_frame = Instant::now();

        println!(
            "[CONTROLS] P/Space = pause | 1-4 = speed | R = toggle brain persist | ESC = quit"
        );
        println!(
            "[CONTROLS] N = spawn agent | M = spawn mutated agent | Tab = cycle telemetry"
        );
        println!(
            "[SIM] Brain on death: {}",
            if self.persist_brain {
                "PERSIST (learning preserved)"
            } else {
                "RESET (fresh brain)"
            }
        );
        info!("Renderer + world + brain initialized — agent is alive");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Window close requested, shutting down");
                self.print_session_summary();
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                    self.camera.aspect =
                        new_size.width as f32 / new_size.height.max(1) as f32;
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    // ── camera controls ─────────────────────────────
                    PhysicalKey::Code(KeyCode::KeyW) => self.camera.move_forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.camera.move_backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.camera.move_left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.camera.move_right = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => self.camera.move_up = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => {
                        self.camera.move_down = pressed;
                    }
                    PhysicalKey::Code(KeyCode::Digit0) if pressed => {
                        self.camera.reset();
                    }

                    // ── simulation controls ─────────────────────────
                    PhysicalKey::Code(KeyCode::Escape) if pressed => {
                        self.print_session_summary();
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::KeyP | KeyCode::Space) if pressed => {
                        self.paused = !self.paused;
                        println!(
                            "[SIM] {}",
                            if self.paused { "PAUSED" } else { "RESUMED" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit1) if pressed => {
                        self.speed_multiplier = 1;
                        println!("[SIM] Speed: 1x ({} ticks/sec)", SIM_RATE as u32);
                    }
                    PhysicalKey::Code(KeyCode::Digit2) if pressed => {
                        self.speed_multiplier = 2;
                        println!("[SIM] Speed: 2x ({} ticks/sec)", SIM_RATE as u32 * 2);
                    }
                    PhysicalKey::Code(KeyCode::Digit3) if pressed => {
                        self.speed_multiplier = 5;
                        println!("[SIM] Speed: 5x ({} ticks/sec)", SIM_RATE as u32 * 5);
                    }
                    PhysicalKey::Code(KeyCode::Digit4) if pressed => {
                        self.speed_multiplier = 10;
                        println!("[SIM] Speed: 10x ({} ticks/sec)", SIM_RATE as u32 * 10);
                    }
                    PhysicalKey::Code(KeyCode::Digit5) if pressed => {
                        self.speed_multiplier = 100;
                        println!("[SIM] Speed: 100x ({} ticks/sec)", SIM_RATE as u32 * 100);
                    }
                    PhysicalKey::Code(KeyCode::Digit6) if pressed => {
                        self.speed_multiplier = 1000;
                        println!("[SIM] Speed: 1000x ({} ticks/sec)", SIM_RATE as u32 * 1000);
                    }
                    PhysicalKey::Code(KeyCode::KeyR) if pressed => {
                        self.persist_brain = !self.persist_brain;
                        for a in &mut self.agents {
                            a.persist_brain = self.persist_brain;
                        }
                        println!(
                            "[SIM] Brain on death: {}",
                            if self.persist_brain {
                                "PERSIST (learning preserved)"
                            } else {
                                "RESET (fresh brain)"
                            }
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyH) if pressed => {
                        self.heatmap_enabled = !self.heatmap_enabled;
                        println!(
                            "[SIM] Heatmap: {}",
                            if self.heatmap_enabled { "ON" } else { "OFF" }
                        );
                    }

                    // ── agent spawning ──────────────────────────────
                    PhysicalKey::Code(KeyCode::KeyN) if pressed => {
                        self.spawn_agent(self.brain_config.clone(), 0);
                    }
                    PhysicalKey::Code(KeyCode::KeyM) if pressed => {
                        let mutated = mutate_config(&self.brain_config);
                        self.spawn_agent(mutated, 0);
                    }

                    // ── telemetry cycling ───────────────────────────
                    PhysicalKey::Code(KeyCode::Tab) if pressed => {
                        if !self.agents.is_empty() {
                            self.selected_agent_idx =
                                (self.selected_agent_idx + 1) % self.agents.len();
                            self.agents[self.selected_agent_idx].trail_dirty = true;
                            let a = &self.agents[self.selected_agent_idx];
                            println!(
                                "[TELEMETRY] Now showing Agent {} (gen {}, color: ({:.2},{:.2},{:.2}))",
                                a.id, a.generation, a.color[0], a.color[1], a.color[2]
                            );
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.camera.is_mouse_dragging = state == ElementState::Pressed;
                    if state == ElementState::Released {
                        self.camera.last_mouse_pos = None;
                    }
                }
                // Right-click to select nearest agent
                if button == MouseButton::Right && state == ElementState::Released {
                    self.pick_agent_at_cursor();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
                self.camera.process_mouse_move(position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.process_scroll(scroll);
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;

                // ── FPS tracking ──────────────────────────────────
                self.frame_times.push_back(now);
                while self.frame_times.len() > 300 {
                    self.frame_times.pop_front();
                }
                if self.frame_times.len() >= 2 {
                    let elapsed = (*self.frame_times.back().unwrap()
                        - *self.frame_times.front().unwrap())
                    .as_secs_f32();
                    self.fps = (self.frame_times.len() - 1) as f32 / elapsed.max(0.001);
                }

                // ── camera ─────────────────────────────────────────
                self.camera.update(dt);

                // ── simulation ticks (fixed timestep) ──────────
                if !self.paused {
                    self.sim_accumulator += dt * self.speed_multiplier as f32;
                    // Cap per-frame ticks proportional to speed, with a reasonable ceiling
                    let max_ticks = (self.speed_multiplier * 2).min(2000);
                    let mut ticks_run = 0u32;

                    // Brain decimation: at high speed, run brain every Nth tick
                    // to keep CPU load manageable. Physics still runs every tick.
                    // sqrt scaling: 1x→1, 10x→3, 100x→10, 1000x→31
                    let brain_stride = ((self.speed_multiplier as f32).sqrt() as u64).max(1);

                    // Pre-allocate buffers outside the tick loop to avoid
                    // repeated heap allocations (12 agents × 10 ticks = 120/frame).
                    let mut all_positions: Vec<(Vec3, bool)> =
                        Vec::with_capacity(self.agents.len());
                    let mut others_buf: Vec<OtherAgent> =
                        Vec::with_capacity(self.agents.len());

                    while self.sim_accumulator >= SIM_DT && ticks_run < max_ticks {
                        self.sim_accumulator -= SIM_DT;
                        ticks_run += 1;

                        if let Some(world) = &mut self.world {
                            all_positions.clear();
                            all_positions.extend(
                                self.agents
                                    .iter()
                                    .map(|a| (a.body.body.position, a.body.body.alive)),
                            );

                            for i in 0..self.agents.len() {
                                let agent = &mut self.agents[i];
                                if !agent.body.body.alive {
                                    continue;
                                }

                                // Stagger brain ticks across agents so CPU load
                                // is spread evenly across frames.
                                let run_brain = (self.tick + i as u64) % brain_stride == 0;

                                let motor = if run_brain {
                                    others_buf.clear();
                                    for (j, (pos, alive)) in all_positions.iter().enumerate() {
                                        if j != i {
                                            others_buf.push(OtherAgent {
                                                position: *pos,
                                                alive: *alive,
                                            });
                                        }
                                    }

                                    let frame = senses::extract_senses_with_others(
                                        &agent.body, world, self.tick, &others_buf,
                                    );

                                    let m = agent.brain.tick(&frame);
                                    agent.cached_motor = m.clone();
                                    m
                                } else {
                                    agent.cached_motor.clone()
                                };

                                let consumed = xagent_sandbox::physics::step(
                                    &mut agent.body, &motor, world, SIM_DT,
                                );
                                if consumed {
                                    self.food_dirty = true;
                                }

                                // Record position for heatmap
                                agent.record_heatmap(world.config.world_size);
                                // Record trail breadcrumb
                                agent.record_trail();

                                if run_brain && i == self.selected_agent_idx {
                                    let life_ticks = agent.age(self.tick);
                                    log_tick_to_csv(
                                        &mut self.logger, agent, world, &motor, life_ticks,
                                    );
                                    self.total_prediction_error +=
                                        agent.brain.telemetry().prediction_error as f64;
                                    self.error_count += 1;
                                }
                            }

                            // Agent-agent collision resolution
                            {
                                let min_dist: f32 = 2.0;
                                let min_dist_sq = min_dist * min_dist;
                                let n = self.agents.len();
                                for i in 0..n {
                                    if !self.agents[i].body.body.alive {
                                        continue;
                                    }
                                    for j in (i + 1)..n {
                                        if !self.agents[j].body.body.alive {
                                            continue;
                                        }
                                        let diff = self.agents[j].body.body.position
                                            - self.agents[i].body.body.position;
                                        let dist_sq = diff.length_squared();
                                        if dist_sq < min_dist_sq && dist_sq > 0.001 {
                                            let dist = dist_sq.sqrt();
                                            let overlap = min_dist - dist;
                                            let push = diff.normalize() * (overlap * 0.5);
                                            let (left, right) = self.agents.split_at_mut(j);
                                            left[i].body.body.position -= push;
                                            right[0].body.body.position += push;
                                        }
                                    }
                                }
                            }

                            let respawned = world.update(SIM_DT);
                            if respawned {
                                self.food_dirty = true;
                            }

                            // Handle death/respawn
                            for agent in &mut self.agents {
                                if !agent.body.body.alive && agent.respawn_cooldown == 0 {
                                    let life_ticks = agent.age(self.tick);
                                    agent.longest_life = agent.longest_life.max(life_ticks);
                                    agent.death_count += 1;
                                    let cause = if agent.body.body.internal.energy <= 0.0 {
                                        "energy depletion"
                                    } else {
                                        "integrity failure"
                                    };
                                    // Death signal: retroactively punish actions that led here
                                    agent.brain.death_signal();
                                    log::debug!(
                                        "[DEATH] Agent {} (gen {}) died at tick {} (lived {} ticks) — cause: {}",
                                        agent.id, agent.generation, self.tick, life_ticks, cause
                                    );
                                    agent.respawn_cooldown = RESPAWN_COOLDOWN_FRAMES;
                                } else if !agent.body.body.alive && agent.respawn_cooldown > 0 {
                                    agent.respawn_cooldown -= 1;
                                    if agent.respawn_cooldown == 0 {
                                        let mut rng = rand::rng();
                                        let half = world.config.world_size / 2.0 - 5.0;
                                        let x: f32 = rng.random_range(-half..half);
                                        let z: f32 = rng.random_range(-half..half);
                                        let y = world.terrain.height_at(x, z) + 1.0;
                                        agent.body = AgentBody::new(Vec3::new(x, y, z));
                                        // Partial respawn: no "free heal" from dying
                                        agent.body.body.internal.energy =
                                            agent.body.body.internal.max_energy * 0.5;
                                        agent.body.body.internal.integrity =
                                            agent.body.body.internal.max_integrity * 0.7;
                                        agent.life_start_tick = self.tick;
                                        agent.has_reproduced = false;
                                        agent.generation += 1;
                                        agent.reset_trail();
                                        if !agent.persist_brain {
                                            agent.brain = Brain::new(agent.brain.config.clone());
                                        } else {
                                            // Death trauma: 20% memory reinforcement decay
                                            agent.brain.trauma(0.2);
                                        }
                                        log::debug!(
                                            "[RESPAWN] Agent {} respawned at ({:.1}, {:.1}) — brain {}",
                                            agent.id, x, z,
                                            if agent.persist_brain { "PERSISTED" } else { "RESET" }
                                        );
                                    }
                                }
                            }
                        }

                        self.tick += 1;

                        // Reproduction — disabled: agents now interact and
                        // compete over scarce food instead of reproducing.
                        // children_to_spawn.clear();
                        // for (i, agent) in self.agents.iter().enumerate() {
                        //     if agent.can_reproduce(self.tick)
                        //         && !agent.has_reproduced
                        //         && self.agents.len() + children_to_spawn.len() < MAX_AGENTS
                        //     {
                        //         children_to_spawn.push(i);
                        //     }
                        // }
                        // for &parent_idx in &children_to_spawn {
                        //     self.agents[parent_idx].has_reproduced = true;
                        //     self.spawn_child(parent_idx);
                        // }

                        // (telemetry is shown on-screen via HUD — no console spam)
                    }

                    // Clamp accumulator to prevent unbounded buildup
                    self.sim_accumulator = self.sim_accumulator.min(SIM_DT * 3.0);

                    // Mark HUD dirty if any ticks ran this frame
                    if ticks_run > 0 {
                        self.hud_dirty = true;
                    }
                }

                // Fix selected index if agents were removed
                if !self.agents.is_empty() {
                    self.selected_agent_idx =
                        self.selected_agent_idx.min(self.agents.len() - 1);
                }

                // ── rebuild dynamic meshes ─────────────────────────
                if self.food_dirty {
                    if let (Some(renderer), Some(world), Some(food_gpu)) =
                        (&self.renderer, &self.world, &mut self.food_gpu)
                    {
                        let fm = world.food_mesh();
                        food_gpu.update_from_mesh(&renderer.queue, &fm);
                        self.food_dirty = false;
                    }
                }

                // ── rebuild heatmap overlay ─────────────────────────
                if self.heatmap_enabled {
                    if let (Some(renderer), Some(world), Some(heatmap_gpu)) =
                        (&self.renderer, &self.world, &mut self.heatmap_gpu)
                    {
                        if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                            let mesh = build_heatmap_mesh(
                                &agent.heatmap,
                                world.config.world_size,
                                &world.terrain,
                            );
                            heatmap_gpu.update_from_mesh(&renderer.queue, &mesh);
                        }
                    }
                } else if let Some(heatmap_gpu) = &mut self.heatmap_gpu {
                    heatmap_gpu.num_indices = 0;
                }

                // ── rebuild trail overlay for selected agent (only when changed) ──
                if let (Some(renderer), Some(trail_gpu)) =
                    (&self.renderer, &mut self.trail_gpu)
                {
                    if let Some(agent) = self.agents.get_mut(self.selected_agent_idx) {
                        if agent.trail_dirty {
                            if agent.trail.len() > 1 && agent.body.body.alive {
                                let mesh = build_trail_mesh(
                                    &agent.trail,
                                    &agent.color,
                                );
                                trail_gpu.update_from_mesh(&renderer.queue, &mesh);
                            } else {
                                trail_gpu.num_indices = 0;
                            }
                            agent.trail_dirty = false;
                        }
                    } else {
                        trail_gpu.num_indices = 0;
                    }
                }

                // ── rebuild selection marker above focused agent ──────
                if let (Some(renderer), Some(marker_gpu)) =
                    (&self.renderer, &mut self.marker_gpu)
                {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        if agent.body.body.alive {
                            let mesh = build_marker_mesh(agent.body.body.position);
                            marker_gpu.update_from_mesh(&renderer.queue, &mesh);
                        } else {
                            marker_gpu.num_indices = 0;
                        }
                    } else {
                        marker_gpu.num_indices = 0;
                    }
                }

                // ── update agent instance buffer (only when sim ticked) ──
                if self.hud_dirty {
                    let instances: Vec<InstanceData> = self
                        .agents
                        .iter()
                        .map(|a| {
                            let color = if !a.body.body.alive {
                                [0.25, 0.25, 0.25] // dead
                            } else {
                                // Color by behavioral significance:
                                // gray (random) → bright red (truly adapted)
                                let t = &a.brain.telemetry();
                                let sig = t.exploitation_ratio
                                    * (1.0 - t.prediction_error.clamp(0.0, 1.0))
                                    * t.memory_utilization;
                                // Cubic curve: agents must genuinely adapt before
                                // turning red. Linear was too generous — 0.3 raw score
                                // already looked bright. Now 0.3³ = 0.027, barely tinted.
                                let s = sig.clamp(0.0, 1.0).powi(3);
                                // Lerp: gray [0.55, 0.55, 0.55] → red [0.95, 0.15, 0.10]
                                [
                                    0.55 + s * 0.40,
                                    0.55 - s * 0.40,
                                    0.55 - s * 0.45,
                                ]
                            };
                            InstanceData {
                                position: a.body.body.position.into(),
                                color,
                                scale: 2.0,
                                _pad: 0.0,
                            }
                        })
                        .collect();
                    self.agent_instance_count = instances.len() as u32;

                    if let (Some(renderer), Some(buf)) =
                        (&self.renderer, &self.agent_instance_buffer)
                    {
                        if !instances.is_empty() {
                            renderer
                                .queue
                                .write_buffer(buf, 0, bytemuck::cast_slice(&instances));
                        }
                    }
                }

                // ── update HUD + text geometry (only when dirty) ──
                if self.hud_dirty {
                    self.cached_hud_bars = self.build_hud_bars();

                    let mut text_items: Vec<TextItem> = Vec::new();

                    // FPS counter at top-right
                    text_items.push(TextItem {
                        text: format!("FPS: {:.0}", self.fps),
                        x: 0.62,
                        y: 0.97,
                        scale: 0.04,
                        color: [0.0, 0.0, 0.0, 1.0],
                    });

                    // Agent count below FPS
                    let alive = self.agents.iter().filter(|a| a.body.body.alive).count();
                    text_items.push(TextItem {
                        text: format!("Agents: {}/{}", alive, self.agents.len()),
                        x: 0.62,
                        y: 0.93,
                        scale: 0.03,
                        color: [0.0, 0.0, 0.0, 0.8],
                    });

                    // HUD bar labels
                    let label_x = -0.98 + 0.36;
                    let label_scale = 0.03;
                    let label_color = [0.0, 0.0, 0.0, 0.9];
                    let label_top = 0.97;
                    let label_gap = 0.035;

                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let energy = agent.body.body.internal.energy_signal();
                        let integrity = agent.body.body.internal.integrity_signal();
                        let pred_err = agent.brain.telemetry().prediction_error;
                        let explore = agent.brain.telemetry().exploration_rate;

                        text_items.push(TextItem {
                            text: format!("Energy: {:.0}%", energy * 100.0),
                            x: label_x,
                            y: label_top,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Integrity: {:.0}%", integrity * 100.0),
                            x: label_x,
                            y: label_top - label_gap,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("PredErr: {:.2}", pred_err),
                            x: label_x,
                            y: label_top - label_gap * 2.0,
                            scale: label_scale,
                            color: label_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Explore: {:.0}%", explore * 100.0),
                            x: label_x,
                            y: label_top - label_gap * 3.0,
                            scale: label_scale,
                            color: label_color,
                        });

                        // Info block below bars
                        let info_y = label_top - label_gap * 4.5;
                        let info_scale = 0.028;
                        let info_color = [0.1, 0.1, 0.2, 0.9];
                        text_items.push(TextItem {
                            text: format!(
                                "Agent {} | Gen {} | Deaths: {}",
                                agent.id, agent.generation, agent.death_count
                            ),
                            x: -0.98,
                            y: info_y,
                            scale: info_scale,
                            color: info_color,
                        });
                        text_items.push(TextItem {
                            text: format!(
                                "Phase: {} | Quality: {:.0}%",
                                agent.brain.telemetry().behavior_phase(),
                                agent.brain.telemetry().decision_quality * 100.0
                            ),
                            x: -0.98,
                            y: info_y - 0.035,
                            scale: info_scale,
                            color: info_color,
                        });
                        text_items.push(TextItem {
                            text: format!("Tick: {}  Speed: {}x", self.tick, self.speed_multiplier),
                            x: -0.98,
                            y: info_y - 0.07,
                            scale: info_scale,
                            color: info_color,
                        });
                    }

                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_hud(&self.cached_hud_bars, &[]);
                        renderer.update_text(&text_items);
                    }

                    self.hud_dirty = false;
                }

                // ── render (just draws — no geometry rebuilding) ──
                let inst_buf = self.agent_instance_buffer.as_ref();
                let inst_count = self.agent_instance_count;

                if let Some(renderer) = &mut self.renderer {
                    let t = self.terrain_gpu.as_ref();
                    let f = self.food_gpu.as_ref();
                    let h = self.heatmap_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let tr = self.trail_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let mk = self.marker_gpu.as_ref().filter(|g| g.num_indices > 0);
                    let mut mesh_vec: Vec<&GpuMesh> = Vec::with_capacity(5);
                    if let Some(t) = t { mesh_vec.push(t); }
                    if let Some(f) = f { mesh_vec.push(f); }
                    if let Some(h) = h { mesh_vec.push(h); }
                    if let Some(tr) = tr { mesh_vec.push(tr); }
                    if let Some(mk) = mk { mesh_vec.push(mk); }

                    let vp = self.camera.view_projection_matrix();
                    match renderer.render_frame(
                        &mesh_vec,
                        &vp,
                        inst_buf,
                        inst_count,
                    ) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let w = renderer.config.width;
                            let h = renderer.config.height;
                            renderer.resize(w, h);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            self.print_session_summary();
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::warn!("Surface error: {:?}", e);
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ── Free functions to avoid borrow conflicts ───────────────────────────

/// Build a flat-quad mesh showing the selected agent's position heatmap.
/// Each non-zero cell becomes a colored quad slightly above the terrain.
fn build_heatmap_mesh(
    heatmap: &[u32],
    world_size: f32,
    terrain: &xagent_sandbox::world::terrain::TerrainData,
) -> xagent_sandbox::world::Mesh {
    use xagent_sandbox::agent::HEATMAP_RES;
    use xagent_sandbox::renderer::Vertex;

    let max_count = heatmap.iter().copied().max().unwrap_or(1).max(1) as f32;
    let half = world_size / 2.0;
    let cell = world_size / HEATMAP_RES as f32;
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for cz in 0..HEATMAP_RES {
        for cx in 0..HEATMAP_RES {
            let count = heatmap[cz * HEATMAP_RES + cx];
            if count == 0 {
                continue;
            }
            let t = (count as f32 / max_count).clamp(0.0, 1.0);
            // blue → yellow → red
            let color = if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0 - s]
            } else {
                let s = (t - 0.5) * 2.0;
                [1.0, 1.0 - s, 0.0]
            };

            let x0 = -half + cx as f32 * cell;
            let z0 = -half + cz as f32 * cell;
            let xm = x0 + cell * 0.5;
            let zm = z0 + cell * 0.5;
            let y = terrain.height_at(xm, zm) + 0.5;

            let base = vertices.len() as u32;
            vertices.push(Vertex { position: [x0, y, z0], color });
            vertices.push(Vertex { position: [x0 + cell, y, z0], color });
            vertices.push(Vertex { position: [x0 + cell, y, z0 + cell], color });
            vertices.push(Vertex { position: [x0, y, z0 + cell], color });
            // CCW winding from above (normal pointing +Y)
            indices.extend_from_slice(&[
                base, base + 2, base + 1,
                base, base + 3, base + 2,
            ]);
        }
    }

    xagent_sandbox::world::Mesh { vertices, indices }
}

/// Build a linear ribbon trail from the agent's distance-sampled control points.
/// No spline interpolation — the 3-unit sampling distance is dense enough for
/// visually smooth curves while keeping CPU cost negligible.
/// Only rebuilt when the trail dirty flag is set.
fn build_trail_mesh(
    points: &[[f32; 3]],
    agent_color: &[f32; 3],
) -> xagent_sandbox::world::Mesh {
    use xagent_sandbox::renderer::Vertex;

    let n = points.len();
    if n < 2 {
        return xagent_sandbox::world::Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let num_segments = n - 1;
    let mut vertices = Vec::with_capacity(num_segments * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(num_segments * 6);

    let ribbon_half_width: f32 = 0.3;
    let y_offset: f32 = 0.3;

    for i in 0..num_segments {
        let a = points[i];
        let b = points[i + 1];

        // Gentle fade: oldest 20% fades from 0.3→0.7, rest is 0.7
        let progress = i as f32 / num_segments as f32;
        let brightness = if progress < 0.2 {
            0.3 + (progress / 0.2) * 0.4
        } else {
            0.7
        };
        let color = [
            agent_color[0] * brightness,
            agent_color[1] * brightness,
            agent_color[2] * brightness,
        ];

        // Perpendicular in XZ plane for ribbon width
        let dx = b[0] - a[0];
        let dz = b[2] - a[2];
        let len = (dx * dx + dz * dz).sqrt().max(0.001);
        let px = -dz / len * ribbon_half_width;
        let pz = dx / len * ribbon_half_width;

        let base = vertices.len() as u32;
        vertices.push(Vertex {
            position: [a[0] + px, a[1] + y_offset, a[2] + pz],
            color,
        });
        vertices.push(Vertex {
            position: [a[0] - px, a[1] + y_offset, a[2] - pz],
            color,
        });
        vertices.push(Vertex {
            position: [b[0] - px, b[1] + y_offset, b[2] - pz],
            color,
        });
        vertices.push(Vertex {
            position: [b[0] + px, b[1] + y_offset, b[2] + pz],
            color,
        });

        indices.extend_from_slice(&[
            base, base + 2, base + 1,
            base, base + 3, base + 2,
        ]);
    }

    xagent_sandbox::world::Mesh { vertices, indices }
}

/// Build a small diamond marker hovering above the given position.
fn build_marker_mesh(position: glam::Vec3) -> xagent_sandbox::world::Mesh {
    use xagent_sandbox::renderer::Vertex;

    let cx = position.x;
    let cy = position.y + 5.0; // float above agent
    let cz = position.z;
    let r: f32 = 1.2; // diamond radius
    let h: f32 = 1.8; // diamond height (top to bottom)
    let color = [1.0, 1.0, 0.2]; // bright yellow

    // Diamond: 4 equatorial points + top + bottom
    let top = [cx, cy + h * 0.5, cz];
    let bot = [cx, cy - h * 0.5, cz];
    let n = [cx, cy, cz - r]; // north
    let s = [cx, cy, cz + r]; // south
    let e = [cx + r, cy, cz];
    let w = [cx - r, cy, cz];

    let shade_top = [color[0], color[1], color[2]];
    let shade_side = [color[0] * 0.75, color[1] * 0.75, color[2] * 0.75];
    let shade_bot = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];

    // 8 triangular faces (4 upper, 4 lower)
    let mut vertices = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(24);

    // Upper faces
    let upper_faces = [(n, e), (e, s), (s, w), (w, n)];
    for (a, b) in &upper_faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: top, color: shade_top });
        vertices.push(Vertex { position: *a, color: shade_side });
        vertices.push(Vertex { position: *b, color: shade_side });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Lower faces
    let lower_faces = [(e, n), (s, e), (w, s), (n, w)];
    for (a, b) in &lower_faces {
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: bot, color: shade_bot });
        vertices.push(Vertex { position: *a, color: shade_side });
        vertices.push(Vertex { position: *b, color: shade_side });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    xagent_sandbox::world::Mesh { vertices, indices }
}

fn log_tick_to_csv(
    logger: &mut Option<MetricsLogger>,
    agent: &Agent,
    world: &WorldState,
    motor: &xagent_shared::MotorCommand,
    life_ticks: u64,
) {
    if let Some(logger) = logger {
        let biome = world
            .biome_map
            .biome_at(agent.body.body.position.x, agent.body.body.position.z);
        let _ = logger.log_tick(
            agent.id,
            agent.brain.telemetry(),
            agent.brain.config.memory_capacity,
            agent.body.body.internal.energy,
            agent.body.body.internal.max_energy,
            agent.body.body.internal.integrity,
            agent.body.body.internal.max_integrity,
            agent.body.body.position,
            agent.body.body.facing,
            biome,
            motor,
            agent.body.body.alive,
            agent.death_count,
            life_ticks,
            agent.generation,
        );
    }
}

// ── Headless simulation loop ───────────────────────────────────────────

fn run_headless(config: FullConfig) {
    info!("Running in headless mode (no window)");
    let tick_duration = Duration::from_secs_f32(1.0 / config.world.tick_rate);
    let dt = 1.0 / config.world.tick_rate;

    let mut world = WorldState::new(config.world);
    let spawn_y = world.terrain.height_at(0.0, 0.0) + 1.0;
    let mut agent_body = AgentBody::new(Vec3::new(0.0, spawn_y, 0.0));
    let mut brain = Brain::new(config.brain.clone());
    let mut tick: u64 = 0;

    loop {
        if agent_body.body.alive {
            let frame = senses::extract_senses(&agent_body, &world, tick);
            let motor = brain.tick(&frame);
            xagent_sandbox::physics::step(&mut agent_body, &motor, &mut world, dt);
            world.update(dt);
            tick += 1;

            if tick % 100 == 0 {
                let t = brain.telemetry();
                let energy_pct =
                    agent_body.body.internal.energy_signal() * 100.0;
                let integrity_pct =
                    agent_body.body.internal.integrity_signal() * 100.0;
                println!(
                    "[Tick {:>5}] Energy: {:.1}% | Integrity: {:.1}% | PredErr: {:.2} | Mem: {}/{}",
                    tick,
                    energy_pct,
                    integrity_pct,
                    t.prediction_error,
                    t.memory_active_count,
                    brain.config.memory_capacity,
                );
            }
        } else if tick > 0 {
            let spawn_y = world.terrain.height_at(0.0, 0.0) + 1.0;
            agent_body = AgentBody::new(Vec3::new(0.0, spawn_y, 0.0));
            brain = Brain::new(config.brain.clone());
            tick = 0;
            log::debug!(
                "[RESPAWN] Agent died and has been respawned with a fresh brain"
            );
        }

        std::thread::sleep(tick_duration);
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    info!("xagent sandbox starting...");
    println!("xagent v0.1.0 \u{2014} Emergent Cognitive Agent Sandbox");

    let config = resolve_config(&cli);

    if cli.dump_config {
        let json = serde_json::to_string_pretty(&config)
            .expect("Failed to serialize config");
        println!("{}", json);
        return;
    }

    print_config(&config);

    if cli.no_render {
        run_headless(config);
    } else {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App::new(config.brain, config.world, cli.log);
        event_loop.run_app(&mut app).expect("Event loop error");
    }
}
