mod app;
mod evolution;
mod gpu_orchestration;
mod render_pipeline;
mod replay_coord;
mod sim_runtime;
mod snapshot;
mod ui_chrome;

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use log::info;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{WindowAttributes, WindowId};
use xagent_shared::{BrainConfig, FullConfig, GovernorConfig, WorldConfig};

use xagent_sandbox::agent::{mutate_config, MAX_AGENTS};
use xagent_sandbox::headless;
use xagent_sandbox::renderer::{GpuMesh, InstanceData, Renderer};
use xagent_sandbox::ui::{EguiIntegration, EvolutionAction, EvolutionState};

use xagent_sandbox::world::WorldState;

use crate::app::{App, SIM_RATE};

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

    /// Run headless (no window, simulation only)
    #[arg(long)]
    no_render: bool,

    /// Print current config as JSON to stdout and exit
    #[arg(long)]
    dump_config: bool,

    /// SQLite database path for evolution state (default: xagent.db)
    #[arg(long, default_value = "xagent.db")]
    db: String,

    /// Resume evolution from existing database
    #[arg(long)]
    resume: bool,

    /// Maximum generations to run (0 = unlimited)
    #[arg(long)]
    generations: Option<u64>,

    /// Print evolution tree from database and exit
    #[arg(long)]
    dump_tree: bool,

    /// Run headless benchmark (no UI, no DB) and print ticks/sec
    #[arg(long)]
    bench: bool,

    /// Number of ticks for --bench mode (default: 10000)
    #[arg(long, default_value_t = 10_000)]
    bench_ticks: u64,

    /// Number of agents for --bench mode (default: 10)
    #[arg(long, default_value_t = 10)]
    bench_agents: usize,

    /// Override world size for --bench mode (default: from preset)
    #[arg(long)]
    world_size: Option<f32>,

    /// Run phase profiler: breaks down time by physics/vision/brain
    #[arg(long)]
    bench_profile: bool,

    /// Run fused-dispatch pass-isolation A/B (full vs skip global/vision) to
    /// locate the throughput ceiling. Honors --bench-ticks / --bench-agents.
    #[arg(long)]
    bench_phase_ab: bool,

    /// Sweep agent counts to locate the GPU occupancy knee: prints, per N, tps
    /// and agent-ticks/sec (tps × N) and flags the knee. Honors --bench-ticks /
    /// --world-size (--bench-agents is ignored — the sweep sets N itself).
    #[arg(long)]
    bench_agent_sweep: bool,

    /// A/B the visual-cortex pass cost: full-pipeline tps with
    /// visual_cortex_enabled OFF vs. ON at the configured retina resolution
    /// (plan 0008). Honors --bench-ticks / --bench-agents / --world-size; the
    /// retina dimensions come from the brain preset / --config.
    #[arg(long)]
    bench_visual_cortex: bool,

    /// Run speed-decoupling validation (plan 0009): A/B test with all 0009 flags
    /// off (baseline) vs on, measuring speed↔fitness correlation and other metrics.
    #[arg(long)]
    validate_speed_decoupling: bool,

    /// Number of generations for speed-decoupling validation (default: 10)
    #[arg(long, default_value_t = 10)]
    validation_generations: u64,
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
                eprintln!(
                    "Error: Unknown brain preset '{}'. Choose: tiny, default, large",
                    other
                );
                std::process::exit(1);
            }
        };
        let world = match cli.world_preset.as_str() {
            "easy" => WorldConfig::easy(),
            "normal" => WorldConfig::default(),
            "hard" => WorldConfig::hard(),
            other => {
                eprintln!(
                    "Error: Unknown world preset '{}'. Choose: easy, normal, hard",
                    other
                );
                std::process::exit(1);
            }
        };
        FullConfig {
            brain,
            world,
            governor: GovernorConfig::default(),
        }
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
    // `mem_cost` / `proc_cost` reflect the metabolic-cost proxies — the
    // kernel's actual pattern-memory and recall widths are fixed constants
    // (see `xagent_brain::buffers::{MEMORY_CAP, RECALL_K}`). `visual_encoding_size`
    // is currently unused and omitted from this summary (see issue #106).
    println!(
        "  Brain: mem_cost={} proc_cost={} dim={} lr={} decay={}",
        config.brain.memory_capacity,
        config.brain.processing_slots,
        config.brain.representation_dimension,
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

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );
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

        // Trail overlay: combined ribbons for ALL agents.
        // Each segment = 4 verts, 6 indices.
        let max_trail_segs = xagent_sandbox::agent::MAX_TRAIL_POINTS as u64 * MAX_AGENTS as u64;
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
        self.window = Some(window.clone());

        // ── egui integration ───────────────────────────────────────
        {
            let r = self.renderer.as_ref().unwrap();
            self.egui = Some(EguiIntegration::new(
                &r.device,
                r.config.format,
                &window,
                r.config.width,
                r.config.height,
            ));
        }

        self.tick = 0;
        self.last_frame = Instant::now();

        println!("[CONTROLS] P/Space = pause | 1-6 = speed | G = toggle 3D");
        println!("[CONTROLS] N = spawn agent | M = spawn mutated agent | Tab = cycle telemetry");
        info!("Renderer + world + brain initialized — agent is alive");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Forward every event to egui first.
        // If egui consumed it (pointer over a panel, typing in a text field, etc.)
        // we skip our own camera/sim key handling for that event.
        let egui_consumed = if let (Some(egui), Some(window)) = (&mut self.egui, &self.window) {
            egui.on_window_event(window, &event)
        } else {
            false
        };

        // For mouse/scroll events over the 3D viewport, let camera controls through.
        // The viewport_hovered flag is set each frame when the pointer is over the
        // viewport image. When it's over other egui panels, block camera input.
        let pointer_on_viewport = self.viewport_hovered;

        match event {
            WindowEvent::CloseRequested => {
                info!("Window close requested, shutting down");
                self.print_session_summary();
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size.width, new_size.height);
                    self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
                }
            }

            WindowEvent::KeyboardInput { event, .. } if !egui_consumed => {
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
                    PhysicalKey::Code(KeyCode::KeyP | KeyCode::Space) if pressed => {
                        match self.evo_snapshot.state {
                            EvolutionState::Running => {
                                self.handle_evolution_action(EvolutionAction::Pause);
                            }
                            EvolutionState::Paused => {
                                self.handle_evolution_action(EvolutionAction::Unpause);
                            }
                            _ => {}
                        }
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
                    PhysicalKey::Code(KeyCode::Digit7) if pressed => {
                        self.speed_multiplier = 10000;
                        println!(
                            "[SIM] Speed: 10000x ({} ticks/sec)",
                            SIM_RATE as u32 * 10000
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit8) if pressed => {
                        self.speed_multiplier = 100000;
                        println!(
                            "[SIM] Speed: 100000x ({} ticks/sec)",
                            SIM_RATE as u32 * 100000
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit9) if pressed => {
                        self.speed_multiplier = 1000000;
                        println!(
                            "[SIM] Speed: 1000000x ({} ticks/sec)",
                            SIM_RATE as u32 * 1000000
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyH) if pressed => {
                        self.heatmap_enabled = !self.heatmap_enabled;
                        self.heatmap_dirty = true;
                        println!(
                            "[SIM] Heatmap: {}",
                            if self.heatmap_enabled { "ON" } else { "OFF" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::KeyG) if pressed => {
                        self.render_3d = !self.render_3d;
                        self.log_msg(format!(
                            "[SIM] 3D render: {}",
                            if self.render_3d {
                                "ON"
                            } else {
                                "OFF (fast mode)"
                            }
                        ));
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
                    PhysicalKey::Code(KeyCode::Tab) if pressed && !self.agents.is_empty() => {
                        let next_agent_idx = (self.selected_agent_idx + 1) % self.agents.len();
                        self.selected_agent_idx = next_agent_idx;
                        self.agents[self.selected_agent_idx].trail_dirty = true;
                        let a = &self.agents[self.selected_agent_idx];
                        println!(
                            "[TELEMETRY] Now showing Agent {} (gen {}, color: ({:.2},{:.2},{:.2}))",
                            a.id, a.generation, a.color[0], a.color[1], a.color[2]
                        );
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } if pointer_on_viewport => {
                if button == MouseButton::Right {
                    self.camera.is_mouse_dragging = state == ElementState::Pressed;
                    if state == ElementState::Released {
                        self.camera.last_mouse_pos = None;
                    }
                }
                // Left-click to select nearest agent
                if button == MouseButton::Left && state == ElementState::Released {
                    self.pick_agent_at_cursor();
                }
            }

            WindowEvent::CursorMoved { position, .. }
                if pointer_on_viewport || self.camera.is_mouse_dragging =>
            {
                self.cursor_pos = (position.x, position.y);
                if self.camera.orbit_mode {
                    self.camera.process_orbit_mouse_move(position.x, position.y);
                } else {
                    self.camera.process_mouse_move(position.x, position.y);
                }
            }

            WindowEvent::MouseWheel { delta, .. } if pointer_on_viewport => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                if self.camera.orbit_mode {
                    self.camera.process_orbit_scroll(scroll);
                } else {
                    self.camera.process_scroll(scroll);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;
                self.runtime_counters.frames_rendered += 1;
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
                // Detect orbit toggle: initialize orbit parameters from current camera
                if self.orbit_mode && !self.camera.orbit_mode {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let diff = self.camera.position - agent.body.body.position;
                        let len = diff.length().max(0.001);
                        self.camera.orbit_distance = len.clamp(5.0, 200.0);
                        self.camera.orbit_yaw = diff.z.atan2(diff.x);
                        self.camera.orbit_pitch = (diff.y / len).asin().clamp(0.05, 1.4);
                    }
                }
                self.camera.orbit_mode = self.orbit_mode;
                if self.camera.orbit_mode {
                    if let Some(agent) = self.agents.get(self.selected_agent_idx) {
                        let target = agent.body.body.position;
                        self.camera.update_orbit(target);
                    }
                } else {
                    self.camera.update(dt);
                }

                // ── consume simulation-worker events ──
                // The worker owns the GPU kernel and advances ticks on its own
                // cadence; the redraw path only drains events, applies the
                // newest snapshot to CPU caches, and drives the generation
                // handoff. No GPU dispatch, readback, or device poll happens here.
                self.drain_sim_events();

                // Advance replay playback
                self.advance_replay_playback();

                // Fix selected index if agents were removed
                if !self.agents.is_empty() {
                    self.selected_agent_idx = self.selected_agent_idx.min(self.agents.len() - 1);
                }

                // ── rebuild dynamic meshes (throttled) + per-frame marker ──
                self.rebuild_dynamic_meshes();

                // ── update agent instance buffer + HUD/text (only when dirty) ──
                self.update_agent_instances();
                self.rebuild_hud_text();

                // ── render (3D offscreen + egui chrome) ──
                let pending_evo_action = self.render_frame(event_loop);

                // Handle evolution actions (outside renderer borrow); may start,
                // stop, or reconfigure the simulation worker.
                self.handle_evolution_action(pending_evo_action);

                // Forward any speed/pause/selection changes to the worker.
                self.sync_worker_controls();

                // Emit runtime-decoupling diagnostics (rate-limited internally).
                self.log_runtime_counters();
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

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    info!("xagent sandbox starting...");
    // Banner goes to stderr so stdout stays pure for data-producing modes
    // (notably `--dump-config`, whose stdout must be valid JSON for piping).
    eprintln!("xagent v0.1.0 \u{2014} Emergent Cognitive Agent Sandbox");

    let mut config = resolve_config(&cli);

    // Override generations from CLI
    if let Some(gens) = cli.generations {
        config.governor.max_generations = gens;
    }

    if cli.dump_config {
        let json = serde_json::to_string_pretty(&config).expect("Failed to serialize config");
        println!("{}", json);
        return;
    }

    if cli.bench_profile {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        xagent_sandbox::bench::run_profile(config.brain, config.world, agent_count, total_ticks);
        return;
    }

    if cli.bench_phase_ab {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        xagent_sandbox::bench::run_phase_ab(config.brain, config.world, agent_count, total_ticks);
        return;
    }

    if cli.bench_agent_sweep {
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        // Default N list spans the latency-bound floor (1–50), the occupancy
        // knee neighborhood (100–200), and the plateau (400–1000).
        let counts = [1usize, 4, 10, 50, 100, 200, 400, 1000];
        xagent_sandbox::bench::run_agent_sweep(config.brain, config.world, total_ticks, &counts);
        return;
    }

    if cli.bench_visual_cortex {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        xagent_sandbox::bench::run_visual_cortex_ab(
            config.brain,
            config.world,
            agent_count,
            total_ticks,
        );
        return;
    }

    if cli.bench {
        let agent_count = cli.bench_agents;
        let total_ticks = cli.bench_ticks;
        if let Some(ws) = cli.world_size {
            config.world.world_size = ws;
        }
        println!("Benchmark: {} agents, {} ticks", agent_count, total_ticks);
        let result =
            xagent_sandbox::bench::run_bench(config.brain, config.world, agent_count, total_ticks);
        println!(
            "Completed {} ticks in {:.2}s ({:.0} ticks/sec)",
            result.total_ticks, result.elapsed_secs, result.ticks_per_sec,
        );
        return;
    }

    if cli.dump_tree {
        headless::dump_tree(&cli.db);
        return;
    }

    if cli.validate_speed_decoupling {
        headless::validate_speed_decoupling(config, cli.validation_generations);
        return;
    }

    print_config(&config);

    if cli.no_render {
        headless::run_headless(config, &cli.db, cli.resume, true);
    } else {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App::new(config.brain, config.world, config.governor, &cli.db);
        event_loop.run_app(&mut app).expect("Event loop error");
    }
}

// ── Pure helpers (extracted for testability) ─────────────────────────────

/// Map speed multiplier to a compact display label.
fn speed_label(multiplier: u32) -> &'static str {
    match multiplier {
        1 => "1×",
        2 => "2×",
        5 => "5×",
        10 => "10×",
        100 => "100×",
        1_000 => "1k×",
        10_000 => "10k×",
        100_000 => "100k×",
        1_000_000 => "1000k×",
        _ => "?×",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── speed_label ──────────────────────────────────────────────────

    #[test]
    fn speed_label_all_known_levels() {
        assert_eq!(speed_label(1), "1×");
        assert_eq!(speed_label(2), "2×");
        assert_eq!(speed_label(5), "5×");
        assert_eq!(speed_label(10), "10×");
        assert_eq!(speed_label(100), "100×");
        assert_eq!(speed_label(1_000), "1k×");
        assert_eq!(speed_label(10_000), "10k×");
        assert_eq!(speed_label(100_000), "100k×");
        assert_eq!(speed_label(1_000_000), "1000k×");
    }

    #[test]
    fn speed_label_unknown_returns_fallback() {
        assert_eq!(speed_label(42), "?×");
        assert_eq!(speed_label(0), "?×");
        assert_eq!(speed_label(999), "?×");
    }

    // ── key-to-multiplier mapping ────────────────────────────────────

    #[test]
    fn all_speed_levels_have_labels() {
        let levels: &[u32] = &[1, 2, 5, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
        for &m in levels {
            assert_ne!(speed_label(m), "?×", "missing label for multiplier {}", m);
        }
    }
}
