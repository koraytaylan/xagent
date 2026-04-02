//! egui integration layer — immediate-mode UI rendered on top of the 3D scene.
//!
//! Manages the offscreen render target for the 3D viewport and the egui overlay.

use egui_wgpu::ScreenDescriptor;

/// Tab types for the dock area.
#[derive(Clone, Debug, PartialEq)]
pub enum Tab {
    /// The 3D sandbox viewport (always open, cannot be closed).
    Sandbox,
    /// Evolution dashboard (always open, cannot be closed).
    Evolution,
    /// Agent detail view, keyed by agent ID.
    AgentDetail(u32),
}

/// Sort modes for the agent list sidebar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SortMode {
    Id,
    Energy,
    Integrity,
    Deaths,
    LongestLife,
    PredictionError,
    Fitness,
}

impl SortMode {
    pub const ALL: [SortMode; 7] = [
        SortMode::Id,
        SortMode::Energy,
        SortMode::Integrity,
        SortMode::Deaths,
        SortMode::LongestLife,
        SortMode::PredictionError,
        SortMode::Fitness,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            SortMode::Id => "ID",
            SortMode::Energy => "Energy",
            SortMode::Integrity => "Integrity",
            SortMode::Deaths => "Deaths",
            SortMode::LongestLife => "Longest Life",
            SortMode::PredictionError => "Pred. Error",
            SortMode::Fitness => "Fitness",
        }
    }
}

/// Snapshot of a single agent's state, copied once per frame for the UI.
pub struct AgentSnapshot {
    pub id: u32,
    pub gen: u32,
    pub energy: f32,
    pub max_energy: f32,
    pub integrity: f32,
    pub max_integrity: f32,
    pub alive: bool,
    pub deaths: u32,
    pub color: [f32; 3],
    pub longest_life: u64,
    pub exploration_rate: f32,
    pub prediction_error: f32,
    /// Forward channel weight norm (policy strength).
    pub forward_weight_norm: f32,
    /// Turn channel weight norm (policy strength).
    pub turn_weight_norm: f32,
    pub prediction_error_history: Vec<f32>,
    pub exploration_rate_history: Vec<f32>,
    pub energy_history: Vec<f32>,
    pub integrity_history: Vec<f32>,
    /// Recent decision snapshots for the brain inspector stream.
    pub decision_log: Vec<xagent_brain::DecisionSnapshot>,
    /// Homeostatic gradient (composite).
    pub gradient: f32,
    /// Urgency level.
    pub urgency: f32,
    /// Food consumed (cumulative).
    pub food_consumed: u32,
    /// Total ticks alive (cumulative).
    pub total_ticks_alive: u64,
    /// Current motor forward output.
    pub motor_forward: f32,
    /// Current motor turn output.
    pub motor_turn: f32,
    /// Current behavior phase label.
    pub phase: &'static str,
    /// Agent's 8x6 visual field as RGBA floats (192 values: width*height*4).
    pub vision_color: Vec<f32>,
    /// Visual field dimensions.
    pub vision_width: u32,
    pub vision_height: u32,
    /// Agent position in world space.
    pub position: [f32; 3],
    /// Agent yaw (rotation around Y axis, radians).
    pub yaw: f32,
    /// Sensory habituation: mean attenuation [0.1, 1.0].
    pub mean_attenuation: f32,
    /// Curiosity bonus from sensory monotony.
    pub curiosity_bonus: f32,
    /// Motor fatigue factor [fatigue_floor, 1.0]. Low = fatigued.
    pub fatigue_factor: f32,
    /// Motor output variance.
    pub motor_variance: f32,
    /// Fatigue factor history for the chart.
    pub fatigue_history: Vec<f32>,
}

/// The lifecycle state of the evolution system.
#[derive(Clone, Debug, PartialEq)]
pub enum EvolutionState {
    /// No session in the DB — user should configure params and press Start.
    Idle,
    /// A previous session was found — show summary, offer Resume or Reset.
    HasSession { generation: u32 },
    /// Evolution is actively running.
    Running,
    /// Evolution is paused (user hit Pause).
    Paused,
}

/// Actions the evolution tab can request from the main loop.
#[derive(Clone, Debug, PartialEq)]
pub enum EvolutionAction {
    None,
    Start,
    Resume,
    Pause,
    Unpause,
    Reset,
}

/// Snapshot of evolution state for UI rendering (rebuilt from governor each frame).
pub struct EvolutionSnapshot {
    pub state: EvolutionState,
    pub generation: u32,
    pub gen_tick: u64,
    pub tick_budget: u64,
    pub population_size: usize,
    pub elitism_count: usize,
    pub patience: u32,
    pub max_generations: u64,
    pub eval_repeats: usize,
    pub num_islands: usize,
    pub migration_interval: u32,
    pub wall_time_secs: f64,
    pub ticks_per_sec: f64,
    pub best_fitness: f32,
    pub tree_nodes: Vec<crate::governor::TreeNode>,
    pub current_node_id: Option<i64>,
    pub current_config: Option<xagent_shared::BrainConfig>,
    pub fitness_history: std::collections::HashMap<i64, Vec<(u32, f32, f32)>>, // island_id → (generation, best, avg)
    pub selected_node_id: Option<i64>,
    /// Fraction of available width for the tree pane (0.0–1.0). Persisted across frames.
    pub tree_pane_fraction: f32,
    // Editable fields for Idle state (pre-start configuration)
    pub edit_brain: xagent_shared::BrainConfig,
    pub edit_governor: xagent_shared::GovernorConfig,
}

impl Default for EvolutionSnapshot {
    fn default() -> Self {
        Self {
            state: EvolutionState::Idle,
            generation: 0,
            gen_tick: 0,
            tick_budget: 50_000,
            population_size: 10,
            elitism_count: 3,
            patience: 5,
            max_generations: 0,
            eval_repeats: 2,
            num_islands: 3,
            migration_interval: 5,
            wall_time_secs: 0.0,
            ticks_per_sec: 0.0,
            best_fitness: -1.0,
            tree_nodes: Vec::new(),
            current_node_id: None,
            current_config: None,
            fitness_history: std::collections::HashMap::new(),
            selected_node_id: None,
            tree_pane_fraction: 0.25,
            edit_brain: xagent_shared::BrainConfig::default(),
            edit_governor: xagent_shared::GovernorConfig::default(),
        }
    }
}

/// Per-frame world state snapshot for the mini-map.
pub struct WorldSnapshot {
    /// World size in units (e.g., 200.0).
    pub world_size: f32,
    /// Positions of non-consumed food items (x, z pairs).
    pub food_positions: Vec<[f32; 2]>,
    /// Pre-rendered biome image as egui texture, built once on world creation.
    pub biome_texture: Option<egui::TextureHandle>,
}

impl Default for WorldSnapshot {
    fn default() -> Self {
        Self {
            world_size: 200.0,
            food_positions: Vec::new(),
            biome_texture: None,
        }
    }
}

/// State for the replay playback controls.
pub struct ReplayState {
    /// Whether replay mode is active.
    pub active: bool,
    /// Current playback tick.
    pub current_tick: u64,
    /// Whether playback is running (auto-advancing).
    pub playing: bool,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Total ticks available in the recording.
    pub total_ticks: u64,
    /// Selected agent index in the recording.
    pub selected_agent_idx: usize,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            active: false,
            current_tick: 0,
            playing: false,
            speed: 1.0,
            total_ticks: 0,
            selected_agent_idx: 0,
        }
    }
}

/// Context passed to the TabViewer so it can render both viewport and agent detail tabs.
pub struct TabContext<'a> {
    pub viewport_tex_id: egui::TextureId,
    pub ppp: f32,
    pub desired_vp: &'a mut (u32, u32),
    pub viewport_hovered: &'a mut bool,
    /// Number of history samples visible in the detail chart (zoom level).
    pub chart_window: &'a mut usize,
    pub agents: &'a [AgentSnapshot],
    pub evolution: &'a mut EvolutionSnapshot,
    pub evolution_action: &'a mut EvolutionAction,
    pub world: &'a WorldSnapshot,
    pub replay: &'a mut ReplayState,
    pub recording: Option<&'a crate::replay::GenerationRecording>,
    pub orbit_mode: &'a mut bool,
}

/// Holds egui state needed across frames: context, winit integration, wgpu renderer,
/// and the offscreen texture used to embed the 3D viewport inside an egui panel.
pub struct EguiIntegration {
    pub ctx: egui::Context,
    winit_state: egui_winit::State,
    wgpu_renderer: egui_wgpu::Renderer,

    // Offscreen 3D viewport
    viewport_color_format: wgpu::TextureFormat,
    pub viewport_color: wgpu::Texture,
    pub viewport_color_view: wgpu::TextureView,
    pub viewport_depth_view: wgpu::TextureView,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub viewport_texture_id: egui::TextureId,
}

impl EguiIntegration {
    /// Create a new egui integration sharing the existing wgpu device.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &winit::window::Window,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        let ctx = egui::Context::default();

        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None, // native_pixels_per_point — let egui auto-detect
            None, // theme
            None, // max_texture_side
        );

        let mut wgpu_renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            None, // no depth
            1,    // msaa samples
            false, // dithering
        );

        let (color_tex, color_view, depth_view) =
            Self::create_offscreen_textures(device, viewport_width, viewport_height, surface_format);

        let texture_id = wgpu_renderer.register_native_texture(
            device,
            &color_view,
            wgpu::FilterMode::Linear,
        );

        Self {
            ctx,
            winit_state,
            wgpu_renderer,
            viewport_color_format: surface_format,
            viewport_color: color_tex,
            viewport_color_view: color_view,
            viewport_depth_view: depth_view,
            viewport_width,
            viewport_height,
            viewport_texture_id: texture_id,
        }
    }

    fn create_offscreen_textures(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let w = width.max(1);
        let h = height.max(1);

        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        (color_tex, color_view, depth_view)
    }

    /// Resize the offscreen viewport textures. Call when the egui panel changes size.
    pub fn resize_viewport(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if w == self.viewport_width && h == self.viewport_height {
            return;
        }
        self.viewport_width = w;
        self.viewport_height = h;

        let (color_tex, color_view, depth_view) =
            Self::create_offscreen_textures(device, w, h, self.viewport_color_format);
        self.viewport_color = color_tex;
        self.viewport_color_view = color_view;
        self.viewport_depth_view = depth_view;

        // Re-register the texture with egui using the same TextureId
        self.wgpu_renderer.update_egui_texture_from_wgpu_texture(
            device,
            &self.viewport_color_view,
            wgpu::FilterMode::Linear,
            self.viewport_texture_id,
        );
    }

    /// Forward a winit event to egui. Returns `true` if egui consumed it
    /// (meaning the app should NOT process it as a camera/sim control).
    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.winit_state.on_window_event(window, event);
        response.consumed
    }

    /// Run the egui frame, tessellate, and record a render pass into `encoder`.
    ///
    /// `ui_fn` receives the egui context and should call `egui::SidePanel`, `Window`, etc.
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen: ScreenDescriptor,
        ui_fn: impl FnMut(&egui::Context),
    ) {
        let raw_input = self.winit_state.take_egui_input(window);
        let full_output = self.ctx.run(raw_input, ui_fn);

        self.winit_state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        // Upload changed textures (font atlas on first frame, etc.)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.wgpu_renderer
                .update_texture(device, queue, *id, image_delta);
        }

        self.wgpu_renderer
            .update_buffers(device, queue, encoder, &tris, &screen);

        // egui render pass — Clear to dark background, then draw UI (which embeds viewport texture)
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.12, g: 0.12, b: 0.14, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // wgpu 24 returns RenderPass<'encoder>; egui-wgpu 0.31 expects
            // RenderPass<'static>. forget_lifetime() drops the borrow.
            let mut pass = pass.forget_lifetime();
            self.wgpu_renderer.render(&mut pass, &tris, &screen);
        }

        // Free textures no longer needed
        for id in &full_output.textures_delta.free {
            self.wgpu_renderer.free_texture(id);
        }
    }
}

// ── TabViewer for egui_dock ──────────────────────────────────────────────

impl<'a> egui_dock::TabViewer for TabContext<'a> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        match tab {
            Tab::Sandbox => egui::Id::new("tab_sandbox"),
            Tab::Evolution => egui::Id::new("tab_evolution"),
            Tab::AgentDetail(id) => egui::Id::new("tab_agent_detail").with(*id),
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Sandbox => "🌍 Sandbox".into(),
            Tab::Evolution => "📊 Evolution".into(),
            Tab::AgentDetail(id) => {
                let name = self.agents.iter()
                    .find(|a| a.id == *id)
                    .map(|a| format!("Agent {} (g{})", a.id, a.gen))
                    .unwrap_or_else(|| format!("Agent {} (?)", id));
                format!("🧠 {}", name).into()
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Sandbox => {
                // Toolbar
                ui.horizontal(|ui| {
                    let orbit_label = if *self.orbit_mode { "\u{1F3AF} Orbiting" } else { "\u{1F3AF} Orbit Agent" };
                    if ui.selectable_label(*self.orbit_mode, orbit_label).clicked() {
                        *self.orbit_mode = !*self.orbit_mode;
                    }
                });
                ui.separator();

                // Viewport
                let avail = ui.available_size();
                *self.desired_vp = (
                    (avail.x * self.ppp) as u32,
                    (avail.y * self.ppp) as u32,
                );
                let resp = ui.add(
                    egui::Image::new(egui::load::SizedTexture::new(
                        self.viewport_tex_id,
                        avail,
                    ))
                    .sense(egui::Sense::click_and_drag()),
                );
                *self.viewport_hovered = resp.hovered() || resp.dragged();
            }
            Tab::Evolution => {
                let action = Self::render_evolution_tab(ui, self.evolution);
                if action != EvolutionAction::None {
                    *self.evolution_action = action;
                }
            }
            Tab::AgentDetail(id) => {
                if let Some(snap) = self.agents.iter().find(|a| a.id == *id) {
                    egui::ScrollArea::vertical()
                        .id_salt(format!("agent_detail_scroll_{}", id))
                        .show(ui, |ui| {
                            Self::render_agent_detail(ui, snap, self.chart_window, self.world, self.agents, self.replay, self.recording);
                        });
                } else {
                    ui.label(format!("Agent {} no longer exists.", id));
                }
            }
        }
    }

    fn closeable(&mut self, tab: &mut Self::Tab) -> bool {
        !matches!(tab, Tab::Sandbox | Tab::Evolution)
    }
}

impl<'a> TabContext<'a> {
    fn render_agent_detail(
        ui: &mut egui::Ui,
        snap: &AgentSnapshot,
        chart_window: &mut usize,
        world: &WorldSnapshot,
        all_agents: &[AgentSnapshot],
        replay: &mut ReplayState,
        recording: Option<&crate::replay::GenerationRecording>,
    ) {
        let color = egui::Color32::from_rgb(
            (snap.color[0] * 255.0) as u8,
            (snap.color[1] * 255.0) as u8,
            (snap.color[2] * 255.0) as u8,
        );

        // Header
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
            ui.painter().circle_filled(rect.center(), 7.0, color);
            ui.heading(format!("Agent {} (Gen {})", snap.id, snap.gen));
            if !snap.alive {
                ui.label(egui::RichText::new("DEAD").color(egui::Color32::RED));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(snap.phase)
                        .strong()
                        .color(match snap.phase {
                            "ADAPTED" => egui::Color32::from_rgb(80, 200, 80),
                            "LEARNING" => egui::Color32::from_rgb(200, 200, 80),
                            "EXPLORING" => egui::Color32::from_rgb(200, 140, 60),
                            _ => egui::Color32::from_rgb(150, 150, 150),
                        }),
                );
            });
        });
        ui.separator();

        // ── Replay Controls ──
        if let Some(rec) = recording {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if replay.active {
                    if ui.button("Live").clicked() {
                        replay.active = false;
                        replay.playing = false;
                    }
                } else if ui.button(format!("Replay Gen {}", rec.generation)).clicked() {
                    replay.active = true;
                    replay.current_tick = 0;
                    replay.total_ticks = rec.total_ticks;
                    replay.playing = false;
                    replay.selected_agent_idx = rec.agent_info.iter()
                        .position(|(id, _)| *id == snap.id)
                        .unwrap_or(0);
                }

                if replay.active {
                    ui.separator();
                    if replay.playing {
                        if ui.button("Pause").clicked() {
                            replay.playing = false;
                        }
                    } else if ui.button("Play").clicked() {
                        replay.playing = true;
                    }

                    if ui.button("Reset").clicked() {
                        replay.current_tick = 0;
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Speed:").small());
                    for &spd in &[0.5_f32, 1.0, 2.0, 4.0, 8.0] {
                        let label = if spd == 1.0 { "1x".to_string() } else { format!("{}x", spd) };
                        if ui.selectable_label((replay.speed - spd).abs() < 0.01, &label).clicked() {
                            replay.speed = spd;
                        }
                    }

                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Tick {}/{}", replay.current_tick, replay.total_ticks
                        ))
                        .monospace()
                        .small(),
                    );
                }
            });

            if replay.active {
                let mut tick_f = replay.current_tick as f32;
                let max_f = replay.total_ticks.saturating_sub(1).max(1) as f32;
                ui.add(egui::Slider::new(&mut tick_f, 0.0..=max_f).show_value(false));
                replay.current_tick = tick_f as u64;
            }
            ui.separator();
        }

        // If replay is active, build snapshot from recording
        let replay_snap;
        let effective_snap = if replay.active {
            if let Some(rec) = recording {
                if let Some(record) = rec.get(replay.current_tick, replay.selected_agent_idx) {
                    let (id, color) = rec.agent_info[replay.selected_agent_idx];
                    replay_snap = AgentSnapshot {
                        id,
                        gen: rec.generation,
                        energy: record.energy,
                        max_energy: 1.0,
                        integrity: record.integrity,
                        max_integrity: 1.0,
                        alive: record.alive,
                        deaths: 0,
                        color,
                        longest_life: 0,
                        exploration_rate: record.exploration_rate,
                        prediction_error: record.prediction_error,
                        forward_weight_norm: 0.0,
                        turn_weight_norm: 0.0,
                        prediction_error_history: Vec::new(),
                        exploration_rate_history: Vec::new(),
                        energy_history: Vec::new(),
                        integrity_history: Vec::new(),
                        decision_log: Vec::new(),
                        gradient: record.gradient,
                        urgency: record.urgency,
                        food_consumed: 0,
                        total_ticks_alive: replay.current_tick,
                        motor_forward: record.motor_forward,
                        motor_turn: record.motor_turn,
                        phase: crate::replay::GenerationRecording::phase_label(record.phase),
                        vision_color: record.vision_color.clone().unwrap_or_default(),
                        vision_width: 8,
                        vision_height: 6,
                        position: record.position,
                        yaw: record.yaw,
                        mean_attenuation: record.mean_attenuation,
                        curiosity_bonus: record.curiosity_bonus,
                        fatigue_factor: record.fatigue_factor,
                        motor_variance: record.motor_variance,
                        fatigue_history: Vec::new(),
                    };
                    &replay_snap
                } else {
                    snap
                }
            } else {
                snap
            }
        } else {
            snap
        };

        // Build effective world for replay (food positions from recording)
        let replay_world;
        let effective_world = if replay.active {
            if let Some(rec) = recording {
                let food = rec.food_at_tick(replay.current_tick);
                replay_world = WorldSnapshot {
                    world_size: world.world_size,
                    food_positions: food.iter()
                        .filter(|(_, consumed)| !consumed)
                        .map(|(p, _)| [p[0], p[2]])
                        .collect(),
                    biome_texture: world.biome_texture.clone(),
                };
                &replay_world
            } else {
                world
            }
        } else {
            world
        };

        // Build effective agents for replay (all agents at this tick)
        let replay_agents;
        let effective_agents = if replay.active {
            if let Some(rec) = recording {
                if let Some(tick_records) = rec.get_tick(replay.current_tick) {
                    replay_agents = tick_records.iter().enumerate().map(|(i, r)| {
                        let (id, color) = rec.agent_info[i];
                        AgentSnapshot {
                            id,
                            gen: rec.generation,
                            energy: r.energy,
                            max_energy: 1.0,
                            integrity: r.integrity,
                            max_integrity: 1.0,
                            alive: r.alive,
                            deaths: 0,
                            color,
                            longest_life: 0,
                            exploration_rate: r.exploration_rate,
                            prediction_error: r.prediction_error,
                            forward_weight_norm: 0.0,
                            turn_weight_norm: 0.0,
                            prediction_error_history: Vec::new(),
                            exploration_rate_history: Vec::new(),
                            energy_history: Vec::new(),
                            integrity_history: Vec::new(),
                            decision_log: Vec::new(),
                            gradient: r.gradient,
                            urgency: r.urgency,
                            food_consumed: 0,
                            total_ticks_alive: 0,
                            motor_forward: r.motor_forward,
                            motor_turn: r.motor_turn,
                            phase: crate::replay::GenerationRecording::phase_label(r.phase),
                            vision_color: Vec::new(),
                            vision_width: 8,
                            vision_height: 6,
                            position: r.position,
                            yaw: r.yaw,
                            mean_attenuation: r.mean_attenuation,
                            curiosity_bonus: r.curiosity_bonus,
                            fatigue_factor: r.fatigue_factor,
                            motor_variance: r.motor_variance,
                            fatigue_history: Vec::new(),
                        }
                    }).collect::<Vec<_>>();
                    &replay_agents[..]
                } else {
                    all_agents
                }
            } else {
                all_agents
            }
        } else {
            all_agents
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            // -- Vitals + Motor --
            ui.columns(2, |cols| {
                // Left column: vitals + statistics
                cols[0].label(egui::RichText::new("Vitals").strong());
                cols[0].add_space(4.0);

                let energy_frac = effective_snap.energy / effective_snap.max_energy.max(0.001);
                let integrity_frac = effective_snap.integrity / effective_snap.max_integrity.max(0.001);

                egui::Grid::new("vitals_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(&mut cols[0], |ui| {
                        ui.label("Energy:");
                        ui.add(egui::ProgressBar::new(energy_frac)
                            .text(format!("{:.1}/{:.0}", effective_snap.energy, effective_snap.max_energy))
                            .fill(if energy_frac > 0.5 {
                                egui::Color32::from_rgb(80, 200, 80)
                            } else if energy_frac > 0.25 {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::from_rgb(220, 60, 60)
                            }));
                        ui.end_row();

                        ui.label("Integrity:");
                        ui.add(egui::ProgressBar::new(integrity_frac)
                            .text(format!("{:.1}/{:.0}", effective_snap.integrity, effective_snap.max_integrity))
                            .fill(egui::Color32::from_rgb(100, 150, 255)));
                        ui.end_row();
                    });

                cols[0].add_space(8.0);
                cols[0].label(egui::RichText::new("Statistics").strong());
                cols[0].add_space(4.0);
                cols[0].label(format!("Deaths: {}", effective_snap.deaths));
                cols[0].label(format!("Food consumed: {}", effective_snap.food_consumed));
                cols[0].label(format!("Longest life: {} ticks", effective_snap.longest_life));
                cols[0].label(format!("Total alive: {} ticks", effective_snap.total_ticks_alive));

                // Right column: brain + motor
                cols[1].label(egui::RichText::new("Brain").strong());
                cols[1].add_space(4.0);
                cols[1].label(format!("Exploration: {:.1}%", effective_snap.exploration_rate * 100.0));
                cols[1].label(format!("Pred. error: {:.4}", effective_snap.prediction_error));
                cols[1].label(format!("Gradient: {:+.4}", effective_snap.gradient));
                cols[1].label(format!("Urgency: {:.2}", effective_snap.urgency));
                cols[1].add_space(4.0);
                cols[1].label(
                    egui::RichText::new(format!(
                        "Habituation: {:.0}%",
                        effective_snap.mean_attenuation * 100.0
                    ))
                    .color(if effective_snap.mean_attenuation < 0.4 {
                        egui::Color32::from_rgb(220, 100, 60)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Curiosity: {:.3}",
                        effective_snap.curiosity_bonus
                    ))
                    .color(if effective_snap.curiosity_bonus > 0.2 {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Fatigue: {:.0}%",
                        effective_snap.fatigue_factor * 100.0
                    ))
                    .color(if effective_snap.fatigue_factor < 0.5 {
                        egui::Color32::from_rgb(220, 100, 60)
                    } else {
                        egui::Color32::GRAY
                    }),
                );
                cols[1].label(
                    egui::RichText::new(format!(
                        "Motor var: {:.4}",
                        effective_snap.motor_variance
                    ))
                    .small()
                    .color(egui::Color32::GRAY),
                );

                cols[1].add_space(8.0);
                cols[1].label(egui::RichText::new("Motor Output").strong());
                cols[1].add_space(4.0);

                egui::Grid::new("motor_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(&mut cols[1], |ui| {
                        ui.label("Forward:");
                        let fwd = effective_snap.motor_forward;
                        let fwd_color = if fwd >= 0.0 {
                            egui::Color32::from_rgb(80, 200, 80)
                        } else {
                            egui::Color32::from_rgb(220, 60, 60)
                        };
                        ui.add(egui::ProgressBar::new(fwd.abs())
                            .text(format!("{:+.3}", fwd))
                            .fill(fwd_color)
                            .desired_width(120.0));
                        ui.end_row();

                        ui.label("Turn:");
                        let trn = effective_snap.motor_turn;
                        let trn_color = if trn >= 0.0 {
                            egui::Color32::from_rgb(100, 150, 255)
                        } else {
                            egui::Color32::from_rgb(200, 100, 255)
                        };
                        ui.add(egui::ProgressBar::new(trn.abs())
                            .text(format!("{:+.3} {}", trn, if trn > 0.05 { "R" } else if trn < -0.05 { "L" } else { "" }))
                            .fill(trn_color)
                            .desired_width(120.0));
                        ui.end_row();

                        ui.label("Fwd norm:");
                        ui.label(format!("{:.3}", effective_snap.forward_weight_norm));
                        ui.end_row();

                        ui.label("Turn norm:");
                        ui.label(format!("{:.3}", effective_snap.turn_weight_norm));
                        ui.end_row();
                    });
            });

            // ── Agent Vision + Mini-Map (side by side) ────────
            ui.add_space(8.0);
            ui.columns(2, |cols| {
                // ── Left column: Vision ──────────────────────
                cols[0].label(egui::RichText::new("Vision").strong());
                cols[0].add_space(4.0);

                if effective_snap.vision_color.len() >= (effective_snap.vision_width * effective_snap.vision_height * 4) as usize {
                    let vw = effective_snap.vision_width as usize;
                    let vh = effective_snap.vision_height as usize;
                    let display_w = cols[0].available_width().max(80.0);
                    let cell_w = display_w / vw as f32;
                    let cell_h = cell_w; // square pixels
                    let display_h = cell_h * vh as f32;

                    let (rect, _) = cols[0].allocate_exact_size(
                        egui::vec2(display_w, display_h),
                        egui::Sense::hover(),
                    );
                    let p = cols[0].painter();

                    // Background
                    p.rect_filled(rect, 2.0, egui::Color32::from_gray(15));

                    // Paint each pixel as a colored rectangle
                    for row in 0..vh {
                        for col in 0..vw {
                            let idx = (row * vw + col) * 4;
                            let r = (effective_snap.vision_color[idx] * 255.0).clamp(0.0, 255.0) as u8;
                            let g = (effective_snap.vision_color[idx + 1] * 255.0).clamp(0.0, 255.0) as u8;
                            let b = (effective_snap.vision_color[idx + 2] * 255.0).clamp(0.0, 255.0) as u8;
                            let pixel_rect = egui::Rect::from_min_size(
                                egui::pos2(
                                    rect.left() + col as f32 * cell_w,
                                    rect.top() + row as f32 * cell_h,
                                ),
                                egui::vec2(cell_w, cell_h),
                            );
                            p.rect_filled(pixel_rect, 0.0, egui::Color32::from_rgb(r, g, b));
                        }
                    }

                    // Grid lines (subtle)
                    for col in 0..=vw {
                        let x = rect.left() + col as f32 * cell_w;
                        p.line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            egui::Stroke::new(0.5, egui::Color32::from_gray(40)),
                        );
                    }
                    for row in 0..=vh {
                        let y = rect.top() + row as f32 * cell_h;
                        p.line_segment(
                            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                            egui::Stroke::new(0.5, egui::Color32::from_gray(40)),
                        );
                    }
                } else {
                    cols[0].label(
                        egui::RichText::new("No vision data available.")
                            .italics()
                            .color(egui::Color32::GRAY),
                    );
                }

                // ── Right column: Mini-Map ────────────────────
                cols[1].horizontal(|ui| {
                    ui.label(egui::RichText::new("Map").strong());
                    ui.label(
                        egui::RichText::new("(top-down)")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                });
                cols[1].add_space(4.0);

                let map_size = cols[1].available_width().max(80.0);
                let (map_rect, _) = cols[1].allocate_exact_size(
                    egui::vec2(map_size, map_size),
                    egui::Sense::hover(),
                );
                let p = cols[1].painter();

                // Draw biome texture as background
                if let Some(ref tex) = effective_world.biome_texture {
                    p.image(
                        tex.id(),
                        map_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                } else {
                    p.rect_filled(map_rect, 2.0, egui::Color32::from_gray(30));
                }

                let ws = effective_world.world_size;
                let half = ws / 2.0;

                // Helper: world (x, z) → map pixel position
                let to_map = |wx: f32, wz: f32| -> egui::Pos2 {
                    let nx = (wx + half) / ws;
                    let nz = (wz + half) / ws;
                    egui::pos2(
                        map_rect.left() + nx * map_rect.width(),
                        map_rect.top() + nz * map_rect.height(),
                    )
                };

                // Draw food items as small yellow-green dots
                for fp in &effective_world.food_positions {
                    let pos = to_map(fp[0], fp[1]);
                    p.circle_filled(pos, 2.0, egui::Color32::from_rgb(140, 200, 40));
                }

                // Draw all other agents as small dots
                for other in effective_agents {
                    if other.id == effective_snap.id {
                        continue;
                    }
                    let acolor = if other.alive {
                        egui::Color32::from_rgb(
                            (other.color[0] * 255.0) as u8,
                            (other.color[1] * 255.0) as u8,
                            (other.color[2] * 255.0) as u8,
                        )
                    } else {
                        egui::Color32::from_gray(80)
                    };
                    let pos = to_map(other.position[0], other.position[2]);
                    p.circle_filled(pos, 3.0, acolor);
                }

                // Draw selected agent as a larger dot with facing direction arrow
                {
                    let agent_pos = to_map(effective_snap.position[0], effective_snap.position[2]);
                    let agent_color = egui::Color32::from_rgb(
                        (effective_snap.color[0] * 255.0) as u8,
                        (effective_snap.color[1] * 255.0) as u8,
                        (effective_snap.color[2] * 255.0) as u8,
                    );

                    // Agent dot
                    p.circle_filled(agent_pos, 5.0, agent_color);
                    p.circle_stroke(agent_pos, 5.0, egui::Stroke::new(1.5, egui::Color32::WHITE));

                    // Facing direction arrow
                    let arrow_len = 12.0;
                    let dx = effective_snap.yaw.sin() * arrow_len;
                    let dz = effective_snap.yaw.cos() * arrow_len;
                    let arrow_end = egui::pos2(agent_pos.x + dx, agent_pos.y + dz);
                    p.line_segment(
                        [agent_pos, arrow_end],
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                }

                // Map border
                p.rect_stroke(map_rect, 2.0, egui::Stroke::new(1.0, egui::Color32::from_gray(80)), egui::StrokeKind::Inside);
            });

            // -- History chart --
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("History").strong());
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!("Window: {} ticks  (scroll to zoom)", *chart_window))
                        .small()
                        .color(egui::Color32::GRAY),
                );
            });
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                for (label, color) in [
                    ("Energy", egui::Color32::from_rgb(80, 200, 80)),
                    ("Integrity", egui::Color32::from_rgb(100, 150, 255)),
                    ("Pred. Error", egui::Color32::from_rgb(200, 140, 60)),
                    ("Exploration", egui::Color32::from_rgb(180, 100, 220)),
                    ("Fatigue", egui::Color32::from_rgb(220, 120, 60)),
                ] {
                    let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot.center(), 4.0, color);
                    ui.label(egui::RichText::new(label).small().color(egui::Color32::GRAY));
                    ui.add_space(6.0);
                }
            });
            ui.add_space(2.0);

            let chart_height = 120.0;
            let avail_w = ui.available_width().max(60.0);
            let (rect, chart_resp) = ui.allocate_exact_size(
                egui::vec2(avail_w, chart_height),
                egui::Sense::hover(),
            );

            if chart_resp.hovered() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll != 0.0 {
                    let factor = if scroll > 0.0 { 0.8 } else { 1.25 };
                    let new_w = ((*chart_window as f32) * factor).round() as usize;
                    *chart_window = new_w.clamp(30, 10_000);
                }
            }

            let painter = ui.painter();
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(25));
            for frac in [0.25, 0.5, 0.75] {
                let y = rect.bottom() - frac * rect.height();
                painter.line_segment(
                    [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                    egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
                );
            }

            let window = *chart_window;
            let series_data: [(&[f32], egui::Color32); 5] = [
                (&effective_snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
                (&effective_snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
                (&effective_snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
                (&effective_snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
                (&effective_snap.fatigue_history, egui::Color32::from_rgb(220, 120, 60)),
            ];
            for &(full_data, color) in &series_data {
                let start = full_data.len().saturating_sub(window);
                let data = &full_data[start..];
                if data.len() < 2 {
                    continue;
                }
                let n = data.len();
                let points: Vec<egui::Pos2> = data.iter().enumerate().map(|(i, &v)| {
                    let x = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                    let y = rect.bottom() - v.clamp(0.0, 1.0) * rect.height();
                    egui::pos2(x, y)
                }).collect();
                let stroke = egui::Stroke::new(1.5, color);
                for pair in points.windows(2) {
                    painter.line_segment([pair[0], pair[1]], stroke);
                }
            }

            // -- Brain Decision Stream --
            ui.add_space(12.0);
            ui.label(egui::RichText::new("Decision Stream").strong().size(14.0));
            ui.add_space(4.0);

            if effective_snap.decision_log.is_empty() {
                ui.label(
                    egui::RichText::new("No decisions recorded yet.")
                        .italics()
                        .color(egui::Color32::GRAY),
                );
            } else {
                let display_count = effective_snap.decision_log.len().min(64);
                let start = effective_snap.decision_log.len() - display_count;

                // Column headers
                ui.horizontal(|ui| {
                    let mono_gray = |text: &str| egui::RichText::new(text).small().strong().color(egui::Color32::from_gray(180)).monospace();
                    ui.label(mono_gray("TICK"));
                    ui.add_space(8.0);
                    ui.label(mono_gray("MOTOR"));
                    ui.add_space(20.0);
                    ui.label(mono_gray("FEELING"));
                    ui.add_space(16.0);
                    ui.label(mono_gray("SIGNAL"));
                    ui.add_space(16.0);
                    ui.label(mono_gray("CONTEXT"));
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("decision_stream")
                    .max_height(300.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for d in effective_snap.decision_log[start..].iter().rev() {
                            let bg = if d.credit_magnitude > 0.05 {
                                if d.raw_gradient > 0.0 {
                                    egui::Color32::from_rgba_premultiplied(30, 60, 30, 255)
                                } else {
                                    egui::Color32::from_rgba_premultiplied(60, 30, 30, 255)
                                }
                            } else {
                                egui::Color32::TRANSPARENT
                            };

                            egui::Frame::NONE
                                .fill(bg)
                                .inner_margin(egui::Margin::symmetric(2, 1))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:>6}", d.tick))
                                                .monospace()
                                                .small()
                                                .color(egui::Color32::from_gray(120)),
                                        );

                                        let fwd_char = if d.motor_forward > 0.3 { "^^" }
                                            else if d.motor_forward > 0.05 { "^" }
                                            else if d.motor_forward < -0.3 { "vv" }
                                            else if d.motor_forward < -0.05 { "v" }
                                            else { "--" };
                                        let trn_char = if d.motor_turn > 0.3 { ">>" }
                                            else if d.motor_turn > 0.05 { ">" }
                                            else if d.motor_turn < -0.3 { "<<" }
                                            else if d.motor_turn < -0.05 { "<" }
                                            else { "--" };
                                        ui.label(
                                            egui::RichText::new(format!("{}{}", fwd_char, trn_char))
                                                .monospace()
                                                .small()
                                                .color(egui::Color32::WHITE),
                                        );

                                        let grad_color = if d.raw_gradient > 0.01 {
                                            egui::Color32::from_rgb(80, 200, 80)
                                        } else if d.raw_gradient < -0.01 {
                                            egui::Color32::from_rgb(220, 60, 60)
                                        } else {
                                            egui::Color32::from_gray(120)
                                        };
                                        ui.label(
                                            egui::RichText::new(format!("g:{:+.3}", d.raw_gradient))
                                                .monospace()
                                                .small()
                                                .color(grad_color),
                                        );

                                        let urgency_color = if d.urgency > 5.0 {
                                            egui::Color32::from_rgb(220, 60, 60)
                                        } else if d.urgency > 2.0 {
                                            egui::Color32::YELLOW
                                        } else {
                                            egui::Color32::from_gray(120)
                                        };
                                        ui.label(
                                            egui::RichText::new(format!("u:{:.1}", d.urgency))
                                                .monospace()
                                                .small()
                                                .color(urgency_color),
                                        );

                                        if d.credit_magnitude > 0.01 {
                                            let credit_color = if d.raw_gradient > 0.0 {
                                                egui::Color32::from_rgb(80, 200, 80)
                                            } else {
                                                egui::Color32::from_rgb(220, 60, 60)
                                            };
                                            ui.label(
                                                egui::RichText::new(format!("cr:{:.3}", d.credit_magnitude))
                                                    .monospace()
                                                    .small()
                                                    .color(credit_color),
                                            );
                                        } else {
                                            ui.label(
                                                egui::RichText::new("cr:---")
                                                    .monospace()
                                                    .small()
                                                    .color(egui::Color32::from_gray(60)),
                                            );
                                        }

                                        ui.label(
                                            egui::RichText::new(format!("pe:{:.3}", d.prediction_error))
                                                .monospace()
                                                .small()
                                                .color(egui::Color32::from_rgb(200, 140, 60)),
                                        );

                                        ui.label(
                                            egui::RichText::new(format!(
                                                "ex:{:.0}% mem:{}",
                                                d.exploration_rate * 100.0,
                                                d.patterns_recalled
                                            ))
                                            .monospace()
                                            .small()
                                            .color(egui::Color32::from_rgb(180, 100, 220)),
                                        );
                                    });
                                });
                        }
                    });
            }
        });
    }

    fn render_evolution_tab(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) -> EvolutionAction {
        let mut action = EvolutionAction::None;

        let state = evo.state.clone();
        match state {
            EvolutionState::Idle => {
                ui.heading("🧬 New Evolution Run");
                ui.add_space(8.0);
                ui.label("Configure parameters below, then press Start.");
                ui.add_space(12.0);

                Self::render_config_editor(ui, evo);
                ui.add_space(12.0);

                if ui.add(egui::Button::new(
                    egui::RichText::new("▶  Start Evolution").strong(),
                ).min_size(egui::vec2(200.0, 36.0))).clicked() {
                    action = EvolutionAction::Start;
                }
            }

            EvolutionState::HasSession { generation } => {
                ui.heading("🧬 Previous Session Found");
                ui.add_space(8.0);
                ui.label(format!(
                    "An evolution run was found at generation {}.",
                    generation,
                ));
                ui.add_space(8.0);

                Self::render_config_summary(ui, evo);
                ui.add_space(12.0);

                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(
                        egui::RichText::new("▶  Resume").strong(),
                    ).min_size(egui::vec2(140.0, 36.0))).clicked() {
                        action = EvolutionAction::Resume;
                    }
                    ui.add_space(16.0);
                    if ui.add(egui::Button::new(
                        egui::RichText::new("🗑  Reset & Start Fresh")
                            .color(egui::Color32::from_rgb(220, 80, 80)),
                    ).min_size(egui::vec2(180.0, 36.0))).clicked() {
                        action = EvolutionAction::Reset;
                    }
                });
            }

            EvolutionState::Running | EvolutionState::Paused => {
                Self::render_running_dashboard(ui, evo);
            }
        }

        action
    }

    fn render_config_editor(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {
        ui.collapsing("🧠 Brain Config", |ui| {
            egui::Grid::new("brain_edit_grid")
                .num_columns(2)
                .spacing([20.0, 6.0])
                .show(ui, |ui| {
                    let b = &mut evo.edit_brain;

                    ui.label("memory_capacity");
                    let mut mc = b.memory_capacity as i32;
                    ui.add(egui::DragValue::new(&mut mc).range(4..=8192).speed(1));
                    b.memory_capacity = mc.max(1) as usize;
                    ui.end_row();

                    ui.label("processing_slots");
                    let mut ps = b.processing_slots as i32;
                    ui.add(egui::DragValue::new(&mut ps).range(1..=256).speed(1));
                    b.processing_slots = ps.max(1) as usize;
                    ui.end_row();

                    ui.label("visual_encoding_size");
                    let mut ve = b.visual_encoding_size as i32;
                    ui.add(egui::DragValue::new(&mut ve).range(2..=512).speed(1));
                    b.visual_encoding_size = ve.max(1) as usize;
                    ui.end_row();

                    ui.label("representation_dim");
                    let mut rd = b.representation_dim as i32;
                    ui.add(egui::DragValue::new(&mut rd).range(4..=128).speed(1));
                    b.representation_dim = rd.max(1) as usize;
                    ui.end_row();

                    ui.label("learning_rate");
                    ui.add(egui::DragValue::new(&mut b.learning_rate)
                        .range(0.0001..=1.0).speed(0.001).max_decimals(5));
                    ui.end_row();

                    ui.label("decay_rate");
                    ui.add(egui::DragValue::new(&mut b.decay_rate)
                        .range(0.0001..=1.0).speed(0.001).max_decimals(5));
                    ui.end_row();
                });
        });

        ui.collapsing("⚙ Governor Config", |ui| {
            egui::Grid::new("gov_edit_grid")
                .num_columns(2)
                .spacing([20.0, 6.0])
                .show(ui, |ui| {
                    let g = &mut evo.edit_governor;

                    ui.label("population_size");
                    let mut ps = g.population_size as i32;
                    ui.add(egui::DragValue::new(&mut ps).range(2..=100).speed(1));
                    g.population_size = ps.max(2) as usize;
                    ui.end_row();

                    ui.label("tick_budget");
                    let mut tb = g.tick_budget as i64;
                    ui.add(egui::DragValue::new(&mut tb).range(1000..=1_000_000).speed(1000));
                    g.tick_budget = tb.max(1000) as u64;
                    ui.end_row();

                    ui.label("elitism_count");
                    let mut ec = g.elitism_count as i32;
                    ui.add(egui::DragValue::new(&mut ec).range(1..=50).speed(1));
                    g.elitism_count = ec.max(1) as usize;
                    ui.end_row();

                    ui.label("max_generations");
                    let mut mg = g.max_generations as i64;
                    ui.add(egui::DragValue::new(&mut mg).range(0..=100_000).speed(1));
                    g.max_generations = mg.max(0) as u64;
                    ui.end_row();

                    ui.label("patience");
                    let mut p = g.patience as i32;
                    ui.add(egui::DragValue::new(&mut p).range(1..=20).speed(1));
                    g.patience = p.max(1) as u32;
                    ui.end_row();

                    ui.label("eval_repeats");
                    let mut er = g.eval_repeats as i32;
                    ui.add(egui::DragValue::new(&mut er).range(1..=5).speed(1));
                    g.eval_repeats = er.max(1) as usize;
                    ui.end_row();

                    ui.label("num_islands");
                    let mut ni = g.num_islands as i32;
                    ui.add(egui::DragValue::new(&mut ni).range(1..=10).speed(1));
                    g.num_islands = ni.max(1) as usize;
                    ui.end_row();

                    ui.label("migration_interval");
                    let mut mi = g.migration_interval as i32;
                    ui.add(egui::DragValue::new(&mut mi).range(0..=50).speed(1));
                    g.migration_interval = mi.max(0) as u32;
                    ui.end_row();
                });
        });
    }

    fn render_config_summary(ui: &mut egui::Ui, evo: &mut EvolutionSnapshot) {
        if let Some(cfg) = &evo.current_config {
            ui.collapsing("🧠 Brain Config", |ui| {
                egui::Grid::new("brain_summary_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("memory_capacity");
                        ui.monospace(format!("{}", cfg.memory_capacity));
                        ui.end_row();
                        ui.label("processing_slots");
                        ui.monospace(format!("{}", cfg.processing_slots));
                        ui.end_row();
                        ui.label("visual_encoding_size");
                        ui.monospace(format!("{}", cfg.visual_encoding_size));
                        ui.end_row();
                        ui.label("representation_dim");
                        ui.monospace(format!("{}", cfg.representation_dim));
                        ui.end_row();
                        ui.label("learning_rate");
                        ui.monospace(format!("{:.5}", cfg.learning_rate));
                        ui.end_row();
                        ui.label("decay_rate");
                        ui.monospace(format!("{:.5}", cfg.decay_rate));
                        ui.end_row();
                    });
            });
        }

        ui.collapsing("⚙ Governor Config", |ui| {
            egui::Grid::new("gov_summary_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    ui.label("population_size");
                    ui.monospace(format!("{}", evo.population_size));
                    ui.end_row();
                    ui.label("tick_budget");
                    ui.monospace(format!("{}", evo.tick_budget));
                    ui.end_row();
                    ui.label("elitism_count");
                    ui.monospace(format!("{}", evo.elitism_count));
                    ui.end_row();
                    ui.label("max_generations");
                    ui.monospace(format!("{}", evo.max_generations));
                    ui.end_row();
                    ui.label("patience");
                    ui.monospace(format!("{}", evo.patience));
                    ui.end_row();
                    ui.label("eval_repeats");
                    ui.monospace(format!("{}", evo.eval_repeats));
                    ui.end_row();
                    ui.label("num_islands");
                    ui.monospace(format!("{}", evo.num_islands));
                    ui.end_row();
                    ui.label("migration_interval");
                    ui.monospace(format!("{}", evo.migration_interval));
                    ui.end_row();
                });
        });

        if !evo.fitness_history.is_empty() {
            Self::render_fitness_chart(ui, &evo.fitness_history);
        }

        if !evo.tree_nodes.is_empty() {
            ui.collapsing("Evolution Tree", |ui| {
                Self::render_tree(ui, &evo.tree_nodes, evo.current_node_id, &mut evo.selected_node_id);
            });
        }
    }

    fn render_running_dashboard(
        ui: &mut egui::Ui,
        evo: &mut EvolutionSnapshot,
    ) {
        // ── Fitness chart at top of tab (always visible) ────────
        if !evo.fitness_history.is_empty() {
            Self::render_fitness_chart(ui, &evo.fitness_history);
        }

        let tree_has_nodes = !evo.tree_nodes.is_empty();

        if tree_has_nodes {
            let tree_nodes = evo.tree_nodes.clone();
            let current_node_id = evo.current_node_id;

            let selected_node_id = evo.selected_node_id;
            let generation = evo.generation;
            let current_config = &evo.current_config;

            // Reserve the full remaining rect for the two-pane layout
            let pane_rect = ui.available_rect_before_wrap();
            let gap = 4.0;
            let sep_x = pane_rect.left() + pane_rect.width() * evo.tree_pane_fraction;

            // Draggable separator
            let sep_interact_rect = egui::Rect::from_x_y_ranges(
                (sep_x - gap)..=(sep_x + gap),
                pane_rect.y_range(),
            );
            let sep_id = ui.id().with("evo_pane_sep");
            let sep_sense = ui.interact(sep_interact_rect, sep_id, egui::Sense::drag());
            if sep_sense.dragged() {
                let new_x = sep_x + sep_sense.drag_delta().x;
                evo.tree_pane_fraction =
                    ((new_x - pane_rect.left()) / pane_rect.width()).clamp(0.15, 0.85);
            }
            if sep_sense.hovered() || sep_sense.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            ui.painter().vline(
                sep_x,
                pane_rect.y_range(),
                ui.visuals().widgets.noninteractive.bg_stroke,
            );

            // Left pane: tree
            let left_rect = egui::Rect::from_x_y_ranges(
                pane_rect.left()..=(sep_x - gap),
                pane_rect.y_range(),
            );
            let vertical = egui::Layout::top_down(egui::Align::LEFT);
            let mut left_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(left_rect)
                    .layout(vertical),
            );
            left_ui.group(|ui| {
                ui.set_min_size(egui::vec2(left_rect.width() - 8.0, left_rect.height() - 8.0));
                ui.label(egui::RichText::new("Evolution Tree").strong().size(13.0));
                ui.add_space(4.0);
                let scroll_width = ui.available_width();
                egui::ScrollArea::both()
                    .id_salt("evo_tree_scroll")
                    .min_scrolled_width(scroll_width)
                    .show(ui, |ui| {
                        ui.set_min_width(scroll_width);
                        Self::render_tree(
                            ui,
                            &tree_nodes,
                            current_node_id,
                            &mut evo.selected_node_id,
                        );
                    });
            });

            // Right pane: detail / overview
            let right_rect = egui::Rect::from_x_y_ranges(
                (sep_x + gap)..=pane_rect.right(),
                pane_rect.y_range(),
            );
            let mut right_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(right_rect)
                    .layout(vertical),
            );
            right_ui.group(|ui| {
                ui.set_min_size(egui::vec2(right_rect.width() - 8.0, right_rect.height() - 8.0));
                if let Some(sel_id) = selected_node_id {
                    let heading_text = tree_nodes.iter()
                        .find(|n| n.id == sel_id)
                        .map(|n| format!("Generation {}", n.generation))
                        .unwrap_or_else(|| format!("Generation {}", generation));
                    ui.heading(heading_text);
                } else {
                    ui.heading(format!("Generation {}", generation));
                }
                ui.add_space(4.0);
                let scroll_width = ui.available_width();
                egui::ScrollArea::vertical()
                    .id_salt("evo_detail_scroll")
                    .min_scrolled_width(scroll_width)
                    .show(ui, |ui| {
                        ui.set_min_width(scroll_width);
                        if let Some(sel_id) = selected_node_id {
                            Self::render_node_detail(ui, &tree_nodes, sel_id);
                        }

                        if let Some(cfg) = current_config {
                            ui.collapsing("Best Config", |ui| {
                                egui::Grid::new("config_grid")
                                    .num_columns(2)
                                    .spacing([20.0, 4.0])
                                    .show(ui, |ui| {
                                        ui.label("memory_capacity");
                                        ui.monospace(format!("{}", cfg.memory_capacity));
                                        ui.end_row();
                                        ui.label("processing_slots");
                                        ui.monospace(format!("{}", cfg.processing_slots));
                                        ui.end_row();
                                        ui.label("visual_encoding_size");
                                        ui.monospace(format!("{}", cfg.visual_encoding_size));
                                        ui.end_row();
                                        ui.label("representation_dim");
                                        ui.monospace(format!("{}", cfg.representation_dim));
                                        ui.end_row();
                                        ui.label("learning_rate");
                                        ui.monospace(format!("{:.5}", cfg.learning_rate));
                                        ui.end_row();
                                        ui.label("decay_rate");
                                        ui.monospace(format!("{:.5}", cfg.decay_rate));
                                        ui.end_row();
                                    });
                            });
                            ui.add_space(4.0);
                        }
                    });
            });

            // Tell the parent layout we consumed the full rect
            ui.advance_cursor_after_rect(pane_rect);
        } else {
            ui.heading(format!("Generation {}", evo.generation));
            ui.add_space(4.0);
            if let Some(cfg) = &evo.current_config {
                ui.collapsing("Best Config", |ui| {
                    egui::Grid::new("config_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("memory_capacity");
                            ui.monospace(format!("{}", cfg.memory_capacity));
                            ui.end_row();
                            ui.label("processing_slots");
                            ui.monospace(format!("{}", cfg.processing_slots));
                            ui.end_row();
                            ui.label("visual_encoding_size");
                            ui.monospace(format!("{}", cfg.visual_encoding_size));
                            ui.end_row();
                            ui.label("representation_dim");
                            ui.monospace(format!("{}", cfg.representation_dim));
                            ui.end_row();
                            ui.label("learning_rate");
                            ui.monospace(format!("{:.5}", cfg.learning_rate));
                            ui.end_row();
                            ui.label("decay_rate");
                            ui.monospace(format!("{:.5}", cfg.decay_rate));
                            ui.end_row();
                        });
                });
                ui.add_space(4.0);
            }
        }
    }

    fn render_fitness_chart(
        ui: &mut egui::Ui,
        fitness_history: &std::collections::HashMap<i64, Vec<(u32, f32, f32)>>,
    ) {
        {
            let ui = &mut *ui;
            let avail = ui.available_width().max(200.0);
            let chart_height = 150.0;
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(avail, chart_height),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_gray(20));

            if fitness_history.is_empty() {
                return;
            }

            // Per-island colors (up to 8 islands, wraps)
            let island_colors = [
                egui::Color32::from_rgb(50, 200, 80),   // green
                egui::Color32::from_rgb(80, 140, 255),   // blue
                egui::Color32::from_rgb(255, 140, 50),    // orange
                egui::Color32::from_rgb(200, 80, 200),    // purple
                egui::Color32::from_rgb(255, 220, 50),    // yellow
                egui::Color32::from_rgb(50, 220, 220),    // cyan
                egui::Color32::from_rgb(255, 80, 80),     // red
                egui::Color32::from_rgb(180, 180, 180),   // grey
            ];

            // Find global max fitness and generation range across all islands
            let mut global_max_fit: f32 = 0.01;
            let mut global_min_gen: u32 = u32::MAX;
            let mut global_max_gen: u32 = 0;
            for points in fitness_history.values() {
                for &(gen, best, _) in points {
                    global_max_fit = global_max_fit.max(best);
                    global_min_gen = global_min_gen.min(gen);
                    global_max_gen = global_max_gen.max(gen);
                }
            }

            if global_min_gen == u32::MAX {
                return; // all vecs were empty — nothing to draw
            }

            let gen_range = (global_max_gen - global_min_gen).max(1) as f32;

            // Sort island keys for deterministic rendering
            let mut island_keys: Vec<i64> = fitness_history.keys().copied().collect();
            island_keys.sort();

            // Draw best-fitness line per island
            for &island_id in &island_keys {
                let points = &fitness_history[&island_id];
                if points.len() < 2 {
                    continue;
                }
                let color = island_colors[island_id.unsigned_abs() as usize % island_colors.len()];

                let pts: Vec<egui::Pos2> = points
                    .iter()
                    .map(|&(gen, best, _)| {
                        egui::pos2(
                            rect.left() + ((gen - global_min_gen) as f32 / gen_range) * rect.width(),
                            rect.bottom() - (best / global_max_fit) * rect.height(),
                        )
                    })
                    .collect();
                for pair in pts.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(2.0, color),
                    );
                }
            }

            // Legend
            ui.horizontal(|ui| {
                for &island_id in &island_keys {
                    let color = island_colors[island_id.unsigned_abs() as usize % island_colors.len()];
                    let label = if island_id < 0 { "Global".to_string() } else { format!("Island {}", island_id) };
                    ui.colored_label(color, label);
                }
                ui.label(format!("Max: {:.4}", global_max_fit));
            });
        }
        ui.add_space(4.0);
    }

    fn render_node_detail(ui: &mut egui::Ui, nodes: &[crate::governor::TreeNode], node_id: i64) {
        let node = match nodes.iter().find(|n| n.id == node_id) {
            Some(n) => n,
            None => {
                ui.label("Node not found.");
                return;
            }
        };

        egui::Grid::new("node_detail_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Status");
                let status_color = match node.status.as_str() {
                    "failed" => egui::Color32::from_rgb(180, 60, 60),
                    "exhausted" => egui::Color32::GRAY,
                    "successful" => egui::Color32::from_rgb(100, 200, 100),
                    "active" => egui::Color32::from_rgb(80, 220, 120),
                    _ => egui::Color32::from_gray(180),
                };
                ui.colored_label(status_color, &node.status);
                ui.end_row();

                if let Some(island) = node.island_id {
                    ui.label("Island");
                    ui.monospace(format!("{}", island));
                    ui.end_row();
                }

                if let Some(fit) = node.best_fitness {
                    ui.label("Best Fitness");
                    ui.monospace(format!("{:.6}", fit));
                    ui.end_row();
                }
                if let Some(avg) = node.avg_fitness {
                    ui.label("Avg Fitness");
                    ui.monospace(format!("{:.6}", avg));
                    ui.end_row();
                }
            });

        if !node.mutations.is_empty() {
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Mutations").strong());
            egui::Grid::new("node_mutations_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    for (param, dir) in &node.mutations {
                        let arrow = if *dir > 0.0 { "+" } else { "-" };
                        ui.label(param);
                        ui.monospace(format!("{}{:.4}", arrow, dir.abs()));
                        ui.end_row();
                    }
                });
        }

        if let Some(cfg) = &node.config {
            ui.add_space(8.0);
            ui.collapsing("Config", |ui| {
                egui::Grid::new("node_config_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("memory_capacity");
                        ui.monospace(format!("{}", cfg.memory_capacity));
                        ui.end_row();
                        ui.label("processing_slots");
                        ui.monospace(format!("{}", cfg.processing_slots));
                        ui.end_row();
                        ui.label("visual_encoding_size");
                        ui.monospace(format!("{}", cfg.visual_encoding_size));
                        ui.end_row();
                        ui.label("representation_dim");
                        ui.monospace(format!("{}", cfg.representation_dim));
                        ui.end_row();
                        ui.label("learning_rate");
                        ui.monospace(format!("{:.5}", cfg.learning_rate));
                        ui.end_row();
                        ui.label("decay_rate");
                        ui.monospace(format!("{:.5}", cfg.decay_rate));
                        ui.end_row();
                        ui.label("distress_exponent");
                        ui.monospace(format!("{:.2}", cfg.distress_exponent));
                        ui.end_row();
                        ui.label("habituation_sensitivity");
                        ui.monospace(format!("{:.1}", cfg.habituation_sensitivity));
                        ui.end_row();
                        ui.label("max_curiosity_bonus");
                        ui.monospace(format!("{:.2}", cfg.max_curiosity_bonus));
                        ui.end_row();
                        ui.label("fatigue_recovery_sensitivity");
                        ui.monospace(format!("{:.1}", cfg.fatigue_recovery_sensitivity));
                        ui.end_row();
                        ui.label("fatigue_floor");
                        ui.monospace(format!("{:.2}", cfg.fatigue_floor));
                        ui.end_row();
                    });
            });
        }
    }

    fn render_tree(
        ui: &mut egui::Ui,
        nodes: &[crate::governor::TreeNode],
        current_id: Option<i64>,
        selected_node_id: &mut Option<i64>,
    ) {
        if nodes.is_empty() {
            return;
        }

        let mut children_map: std::collections::HashMap<Option<i64>, Vec<&crate::governor::TreeNode>> =
            std::collections::HashMap::new();
        for node in nodes {
            children_map.entry(node.parent_id).or_default().push(node);
        }

        let mut expanded_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
        if let Some(current) = current_id {
            let node_map: std::collections::HashMap<i64, &crate::governor::TreeNode> =
                nodes.iter().map(|n| (n.id, n)).collect();
            let mut id = Some(current);
            while let Some(nid) = id {
                expanded_ids.insert(nid);
                id = node_map.get(&nid).and_then(|n| n.parent_id);
            }
        }

        // Sort children: active/successful first, then exhausted, then failed
        for children in children_map.values_mut() {
            children.sort_unstable_by_key(|node| {
                match node.status.as_str() {
                    "active" => 0,
                    "successful" => 1,
                    "exhausted" => 2,
                    "failed" => 3,
                    _ => 4,
                }
            });
        }

        if let Some(roots) = children_map.get(&None) {
            for root in roots {
                Self::render_tree_node(ui, root, &children_map, &expanded_ids, current_id, selected_node_id);
            }
        }
    }

    fn render_tree_node(
        ui: &mut egui::Ui,
        node: &crate::governor::TreeNode,
        children_map: &std::collections::HashMap<Option<i64>, Vec<&crate::governor::TreeNode>>,
        expanded_ids: &std::collections::HashSet<i64>,
        current_id: Option<i64>,
        selected_node_id: &mut Option<i64>,
    ) {
        let has_children = children_map.get(&Some(node.id)).map_or(false, |c| !c.is_empty());
        let is_current = Some(node.id) == current_id;
        let is_on_path = expanded_ids.contains(&node.id);
        let is_selected = *selected_node_id == Some(node.id);

        let label = Self::format_tree_label(node, is_current);
        let mut color = match node.status.as_str() {
            "failed" => egui::Color32::from_rgb(180, 60, 60),
            "exhausted" => egui::Color32::GRAY,
            "successful" => egui::Color32::from_rgb(100, 200, 100),
            _ if is_current => egui::Color32::from_rgb(80, 220, 120),
            _ => egui::Color32::from_gray(180),
        };
        if is_selected {
            color = egui::Color32::from_rgb(120, 200, 255);
        }

        if has_children {
            let header = egui::CollapsingHeader::new(
                egui::RichText::new(&label).color(color).monospace().size(11.0),
            )
            .id_salt(node.id)
            .default_open(is_on_path);

            let resp = header.show(ui, |ui| {
                if let Some(children) = children_map.get(&Some(node.id)) {
                    for child in children {
                        Self::render_tree_node(ui, child, children_map, expanded_ids, current_id, selected_node_id);
                    }
                }
            });
            if resp.header_response.clicked() {
                *selected_node_id = Some(node.id);
            }
        } else {
            ui.horizontal(|ui| {
                ui.add_space(18.0);
                let resp = ui.add(
                    egui::Label::new(
                        egui::RichText::new(&label).color(color).monospace().size(11.0),
                    ).sense(egui::Sense::click()),
                );
                if resp.clicked() {
                    *selected_node_id = Some(node.id);
                }
            });
        }
    }

    fn format_tree_label(node: &crate::governor::TreeNode, is_current: bool) -> String {
        let fitness_str = node
            .best_fitness
            .map(|f| format!("{:.4}", f))
            .unwrap_or_else(|| "—".into());

        let current_marker = if is_current { ">> " } else { "" };

        format!("{}Gen {} {}", current_marker, node.generation, fitness_str)
    }
}
