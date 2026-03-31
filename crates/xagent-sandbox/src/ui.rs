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
    pub tree_nodes: Vec<crate::governor::TreeNode>,
    pub current_node_id: Option<i64>,
    pub current_config: Option<xagent_shared::BrainConfig>,
    pub fitness_history: Vec<(u32, f32, f32)>, // (generation, best, avg)
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
            tree_nodes: Vec::new(),
            current_node_id: None,
            current_config: None,
            fitness_history: Vec::new(),
            edit_brain: xagent_shared::BrainConfig::default(),
            edit_governor: xagent_shared::GovernorConfig::default(),
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
                egui::ScrollArea::vertical()
                    .id_salt("evolution_tab_scroll")
                    .show(ui, |ui| {
                        let action = Self::render_evolution_tab(ui, self.evolution);
                        if action != EvolutionAction::None {
                            *self.evolution_action = action;
                        }
                    });
            }
            Tab::AgentDetail(id) => {
                if let Some(snap) = self.agents.iter().find(|a| a.id == *id) {
                    egui::ScrollArea::vertical()
                        .id_salt(format!("agent_detail_scroll_{}", id))
                        .show(ui, |ui| {
                            Self::render_agent_detail(ui, snap, self.chart_window);
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
    fn render_agent_detail(ui: &mut egui::Ui, snap: &AgentSnapshot, chart_window: &mut usize) {
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

        egui::ScrollArea::vertical().show(ui, |ui| {
            // -- Vitals + Motor --
            ui.columns(2, |cols| {
                // Left column: vitals + statistics
                cols[0].label(egui::RichText::new("Vitals").strong());
                cols[0].add_space(4.0);

                let energy_frac = snap.energy / snap.max_energy.max(0.001);
                let integrity_frac = snap.integrity / snap.max_integrity.max(0.001);

                egui::Grid::new("vitals_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(&mut cols[0], |ui| {
                        ui.label("Energy:");
                        ui.add(egui::ProgressBar::new(energy_frac)
                            .text(format!("{:.1}/{:.0}", snap.energy, snap.max_energy))
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
                            .text(format!("{:.1}/{:.0}", snap.integrity, snap.max_integrity))
                            .fill(egui::Color32::from_rgb(100, 150, 255)));
                        ui.end_row();
                    });

                cols[0].add_space(8.0);
                cols[0].label(egui::RichText::new("Statistics").strong());
                cols[0].add_space(4.0);
                cols[0].label(format!("Deaths: {}", snap.deaths));
                cols[0].label(format!("Food consumed: {}", snap.food_consumed));
                cols[0].label(format!("Longest life: {} ticks", snap.longest_life));
                cols[0].label(format!("Total alive: {} ticks", snap.total_ticks_alive));

                // Right column: brain + motor
                cols[1].label(egui::RichText::new("Brain").strong());
                cols[1].add_space(4.0);
                cols[1].label(format!("Exploration: {:.1}%", snap.exploration_rate * 100.0));
                cols[1].label(format!("Pred. error: {:.4}", snap.prediction_error));
                cols[1].label(format!("Gradient: {:+.4}", snap.gradient));
                cols[1].label(format!("Urgency: {:.2}", snap.urgency));

                cols[1].add_space(8.0);
                cols[1].label(egui::RichText::new("Motor Output").strong());
                cols[1].add_space(4.0);

                egui::Grid::new("motor_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(&mut cols[1], |ui| {
                        ui.label("Forward:");
                        let fwd = snap.motor_forward;
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
                        let trn = snap.motor_turn;
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
                        ui.label(format!("{:.3}", snap.forward_weight_norm));
                        ui.end_row();

                        ui.label("Turn norm:");
                        ui.label(format!("{:.3}", snap.turn_weight_norm));
                        ui.end_row();
                    });
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
            let series_data: [(&[f32], egui::Color32); 4] = [
                (&snap.energy_history, egui::Color32::from_rgb(80, 200, 80)),
                (&snap.integrity_history, egui::Color32::from_rgb(100, 150, 255)),
                (&snap.prediction_error_history, egui::Color32::from_rgb(200, 140, 60)),
                (&snap.exploration_rate_history, egui::Color32::from_rgb(180, 100, 220)),
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

            if snap.decision_log.is_empty() {
                ui.label(
                    egui::RichText::new("No decisions recorded yet.")
                        .italics()
                        .color(egui::Color32::GRAY),
                );
            } else {
                let display_count = snap.decision_log.len().min(64);
                let start = snap.decision_log.len() - display_count;

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
                        for d in snap.decision_log[start..].iter().rev() {
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

        match &evo.state {
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
                let is_running = evo.state == EvolutionState::Running;
                Self::render_running_dashboard(ui, evo, is_running, &mut action);
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

    fn render_config_summary(ui: &mut egui::Ui, evo: &EvolutionSnapshot) {
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
            Self::render_fitness_chart(ui, evo);
        }

        if !evo.tree_nodes.is_empty() {
            ui.collapsing("🌳 Evolution Tree", |ui| {
                Self::render_tree(ui, &evo.tree_nodes, evo.current_node_id);
            });
        }
    }

    fn render_running_dashboard(
        ui: &mut egui::Ui,
        evo: &EvolutionSnapshot,
        is_running: bool,
        action: &mut EvolutionAction,
    ) {
        // ── Header + controls ───────────────────────────────
        ui.horizontal(|ui| {
            ui.heading(format!("Generation {}", evo.generation));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new(
                    egui::RichText::new("🗑 Reset")
                        .color(egui::Color32::from_rgb(220, 80, 80)),
                )).clicked() {
                    *action = EvolutionAction::Reset;
                }
                if is_running {
                    if ui.button("⏸ Pause").clicked() {
                        *action = EvolutionAction::Pause;
                    }
                } else {
                    if ui.button("▶ Resume").clicked() {
                        *action = EvolutionAction::Unpause;
                    }
                }
            });
        });
        ui.add_space(4.0);

        // ── Progress bar ────────────────────────────────────
        let progress = if evo.tick_budget > 0 {
            evo.gen_tick as f32 / evo.tick_budget as f32
        } else {
            0.0
        };
        ui.add(
            egui::ProgressBar::new(progress)
                .text(format!(
                    "{} / {} ticks ({:.0}%)",
                    evo.gen_tick, evo.tick_budget, progress * 100.0,
                ))
                .animate(is_running),
        );
        ui.add_space(8.0);

        // ── Timing stats ────────────────────────────────────
        ui.horizontal(|ui| {
            let hours = (evo.wall_time_secs / 3600.0) as u64;
            let mins = ((evo.wall_time_secs % 3600.0) / 60.0) as u64;
            let secs = (evo.wall_time_secs % 60.0) as u64;
            ui.label(format!("⏱ {}h {:02}m {:02}s", hours, mins, secs));
            ui.separator();
            ui.label(format!("{:.0} ticks/sec", evo.ticks_per_sec));
            ui.separator();
            ui.label(format!("Pop: {} | Elite: {} | Patience: {}",
                             evo.population_size, evo.elitism_count, evo.patience));
        });
        ui.separator();

        // ── Current best config ─────────────────────────────
        if let Some(cfg) = &evo.current_config {
            ui.collapsing("🧬 Best Config", |ui| {
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

        // ── Fitness chart ───────────────────────────────────
        if !evo.fitness_history.is_empty() {
            Self::render_fitness_chart(ui, evo);
        }

        // ── Evolution tree ──────────────────────────────────
        if !evo.tree_nodes.is_empty() {
            ui.collapsing("🌳 Evolution Tree", |ui| {
                Self::render_tree(ui, &evo.tree_nodes, evo.current_node_id);
            });
        }
    }

    fn render_fitness_chart(ui: &mut egui::Ui, evo: &EvolutionSnapshot) {
        ui.collapsing("📈 Fitness Over Generations", |ui| {
            let avail = ui.available_width().max(200.0);
            let chart_height = 120.0;
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(avail, chart_height),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_gray(20));

            let max_fit = evo
                .fitness_history
                .iter()
                .map(|(_, b, _)| *b)
                .fold(0.0f32, f32::max)
                .max(0.01);

            let n = evo.fitness_history.len();
            if n >= 2 {
                let best_pts: Vec<egui::Pos2> = evo
                    .fitness_history
                    .iter()
                    .enumerate()
                    .map(|(i, (_, best, _))| {
                        egui::pos2(
                            rect.left() + (i as f32 / (n - 1) as f32) * rect.width(),
                            rect.bottom() - (best / max_fit) * rect.height(),
                        )
                    })
                    .collect();
                for pair in best_pts.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(50, 200, 80)),
                    );
                }

                let avg_pts: Vec<egui::Pos2> = evo
                    .fitness_history
                    .iter()
                    .enumerate()
                    .map(|(i, (_, _, avg))| {
                        egui::pos2(
                            rect.left() + (i as f32 / (n - 1) as f32) * rect.width(),
                            rect.bottom() - (avg / max_fit) * rect.height(),
                        )
                    })
                    .collect();
                for pair in avg_pts.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 180, 50)),
                    );
                }
            }

            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::from_rgb(50, 200, 80), "● Best");
                ui.colored_label(egui::Color32::from_rgb(200, 180, 50), "● Avg");
                ui.label(format!("Max: {:.4}", max_fit));
            });
        });
        ui.add_space(4.0);
    }

    fn render_tree(
        ui: &mut egui::Ui,
        nodes: &[crate::governor::TreeNode],
        current_id: Option<i64>,
    ) {
        if nodes.is_empty() {
            return;
        }

        // Build children map: parent_id → list of child nodes
        let mut children_map: std::collections::HashMap<Option<i64>, Vec<&crate::governor::TreeNode>> =
            std::collections::HashMap::new();
        for node in nodes {
            children_map.entry(node.parent_id).or_default().push(node);
        }

        // Find path from current node to root for default expansion
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

        // Render from root nodes (parent_id = None)
        if let Some(roots) = children_map.get(&None) {
            for root in roots {
                Self::render_tree_node(ui, root, &children_map, &expanded_ids, current_id);
            }
        }
    }

    fn render_tree_node(
        ui: &mut egui::Ui,
        node: &crate::governor::TreeNode,
        children_map: &std::collections::HashMap<Option<i64>, Vec<&crate::governor::TreeNode>>,
        expanded_ids: &std::collections::HashSet<i64>,
        current_id: Option<i64>,
    ) {
        let has_children = children_map.get(&Some(node.id)).map_or(false, |c| !c.is_empty());
        let is_current = Some(node.id) == current_id;
        let is_on_path = expanded_ids.contains(&node.id);

        let label = Self::format_tree_label(node, is_current);
        let color = match node.status.as_str() {
            "failed" => egui::Color32::from_rgb(180, 60, 60),
            "exhausted" => egui::Color32::GRAY,
            "successful" => egui::Color32::from_rgb(100, 200, 100),
            _ if is_current => egui::Color32::from_rgb(80, 220, 120),
            _ => egui::Color32::from_gray(180),
        };

        let is_dead_end = matches!(node.status.as_str(), "failed" | "exhausted");

        if has_children {
            let mut header = egui::CollapsingHeader::new(
                egui::RichText::new(&label).color(color).monospace().size(11.0),
            )
            .id_salt(node.id)
            .default_open(is_on_path);

            // Force-collapse dead-end branches not on the active path
            if is_dead_end && !is_on_path {
                header = header.open(Some(false));
            }

            header.show(ui, |ui| {
                if let Some(children) = children_map.get(&Some(node.id)) {
                    for child in children {
                        Self::render_tree_node(ui, child, children_map, expanded_ids, current_id);
                    }
                }
            });
        } else {
            // Leaf node — label with small indent to align with collapsible siblings
            ui.horizontal(|ui| {
                ui.add_space(18.0);
                ui.label(egui::RichText::new(&label).color(color).monospace().size(11.0));
            });
        }
    }

    fn format_tree_label(node: &crate::governor::TreeNode, is_current: bool) -> String {
        let fitness_str = node
            .best_fitness
            .map(|f| format!("{:.4}", f))
            .unwrap_or_else(|| "—".into());

        let mutation_str = if node.mutations.is_empty() {
            String::new()
        } else if let Some(config) = &node.config {
            let parts: Vec<String> = node
                .mutations
                .iter()
                .map(|(p, d)| {
                    let arrow = if *d > 0.0 { "↑" } else { "↓" };
                    match p.as_str() {
                        "memory_capacity" => format!("mem{}{}", arrow, config.memory_capacity),
                        "processing_slots" => format!("slots{}{}", arrow, config.processing_slots),
                        "representation_dim" => format!("repr{}{}", arrow, config.representation_dim),
                        "learning_rate" => format!("lr{}{:.4}", arrow, config.learning_rate),
                        "decay_rate" => format!("decay{}{:.4}", arrow, config.decay_rate),
                        other => format!("{}{}", other, arrow),
                    }
                })
                .collect();
            format!(" ({})", parts.join(" "))
        } else {
            let parts: Vec<String> = node
                .mutations
                .iter()
                .map(|(p, d)| {
                    let short = match p.as_str() {
                        "memory_capacity" => "mem",
                        "processing_slots" => "slots",
                        "representation_dim" => "repr",
                        "learning_rate" => "lr",
                        "decay_rate" => "decay",
                        other => other,
                    };
                    format!("{}{}", short, if *d > 0.0 { "↑" } else { "↓" })
                })
                .collect();
            format!(" ({})", parts.join(" "))
        };

        let status_icon = match node.status.as_str() {
            "failed" => " [X]",
            "exhausted" => " [--]",
            "successful" => " [OK]",
            _ => "",
        };

        let current_marker = if is_current { ">> " } else { "" };

        format!(
            "{}Gen {} fit={}{}{}",
            current_marker, node.generation, fitness_str, mutation_str, status_icon,
        )
    }
}
