//! Playback state and egui rendering for a recorded tick window.

use xagent_sandbox::tick_window::{Pin, PinKind, TickWindow, WindowFrame};

/// Frames stepped per redraw by default.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) const DEFAULT_PLAYBACK_SPEED: u32 = 1;
/// Upper bound of the speed control.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) const MAX_PLAYBACK_SPEED: u32 = 64;
/// Decimal places used for value labels.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) const DISPLAY_DECIMALS: usize = 2;

/// Playback state over a tick window.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) struct TickWindowView {
    cursor_tick: Option<u64>,
    playing: bool,
    speed: u32,
    selected_agent: Option<usize>,
    pending_jump: Option<u64>,
}

impl Default for TickWindowView {
    fn default() -> Self {
        Self {
            cursor_tick: None,
            playing: false,
            speed: DEFAULT_PLAYBACK_SPEED,
            selected_agent: None,
            pending_jump: None,
        }
    }
}

/// First frame tick at or after `target`, else the last frame tick.
fn first_tick_at_or_after(window: &TickWindow, target: u64) -> Option<u64> {
    window
        .frames()
        .map(|f| f.tick)
        .find(|&t| t >= target)
        .or_else(|| last_tick(window))
}

/// Tick of the last frame.
fn last_tick(window: &TickWindow) -> Option<u64> {
    window.frames().last().map(|f| f.tick)
}

impl TickWindowView {
    /// Requests that the next advance jumps to `tick`.
    #[allow(dead_code)] // no caller until the view is wired into the GUI
    pub(crate) fn request_jump(&mut self, tick: u64) {
        self.pending_jump = Some(tick);
    }

    /// Applies a pending jump, clamps the cursor, and steps playback.
    #[allow(dead_code)] // no caller until the view is wired into the GUI
    pub(crate) fn advance(&mut self, window: &TickWindow, frames: u32) {
        let target = self.pending_jump.take().or(self.cursor_tick).unwrap_or(0);
        self.cursor_tick = first_tick_at_or_after(window, target);
        if self.playing {
            if let Some(cursor) = self.cursor_tick {
                let step = usize::try_from(frames).unwrap_or(usize::MAX);
                let index = window.frames().position(|f| f.tick == cursor).unwrap_or(0);
                self.cursor_tick = window
                    .frames()
                    .nth(index.saturating_add(step))
                    .map(|f| f.tick)
                    .or_else(|| last_tick(window));
            }
        }
        if self.cursor_tick.is_none() || self.cursor_tick == last_tick(window) {
            self.playing = false;
        }
        let agent_count = self.current_frame(window).map_or(0, |f| f.agents.len());
        if self.selected_agent.is_some_and(|i| i >= agent_count) {
            self.selected_agent = None;
        }
    }

    /// The frame whose tick equals the cursor.
    #[allow(dead_code)] // no caller until the view is wired into the GUI
    pub(crate) fn current_frame<'a>(&self, window: &'a TickWindow) -> Option<&'a WindowFrame> {
        let cursor = self.cursor_tick?;
        window.frames().find(|f| f.tick == cursor)
    }
}

/// Draws a button that records a jump target when clicked.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) fn jump_button(ui: &mut egui::Ui, tick: u64, label: &str, jump: &mut Option<u64>) {
    if ui.button(label).clicked() {
        *jump = Some(tick);
    }
}

fn value_label(ui: &mut egui::Ui, name: &str, v: f32) {
    ui.label(format!("{}: {:.*}", name, DISPLAY_DECIMALS, v));
}

/// Renders the playback controls and the current frame.
#[allow(dead_code)] // no caller until the view is wired into the GUI
pub(crate) fn render(ui: &mut egui::Ui, view: &mut TickWindowView, window: &TickWindow) {
    let speed = view.speed;
    view.advance(window, speed);
    if window.is_empty() {
        ui.label("No frames recorded");
        return;
    }
    let mut jump: Option<u64> = None;
    ui.horizontal(|ui| {
        let text = if view.playing { "Pause" } else { "Play" };
        if ui.button(text).clicked() {
            view.playing = !view.playing;
        }
        let last_index = window.len().saturating_sub(1);
        let mut index = view
            .cursor_tick
            .and_then(|c| window.frames().position(|f| f.tick == c))
            .unwrap_or(0);
        if ui
            .add(egui::Slider::new(&mut index, 0..=last_index).text("Frame"))
            .changed()
        {
            view.cursor_tick = window.frames().nth(index).map(|f| f.tick);
        }
    });
    ui.add(egui::Slider::new(&mut view.speed, 1..=MAX_PLAYBACK_SPEED).text("Speed"));

    let frame = view.current_frame(window);
    let agent_count = frame.map_or(0, |f| f.agents.len());
    let selected_text = view
        .selected_agent
        .map_or_else(|| "None".to_string(), |i| format!("Agent {i}"));
    egui::ComboBox::from_label("Agent")
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut view.selected_agent, None, "None");
            for i in 0..agent_count {
                ui.selectable_value(&mut view.selected_agent, Some(i), format!("Agent {i}"));
            }
        });
    if let Some(rec) = frame.and_then(|f| view.selected_agent.and_then(|i| f.agents.get(i))) {
        let [x, y, z] = rec.position;
        ui.label(format!(
            "Position: ({:.*}, {:.*}, {:.*})",
            DISPLAY_DECIMALS, x, DISPLAY_DECIMALS, y, DISPLAY_DECIMALS, z
        ));
        value_label(ui, "Energy", rec.energy);
        value_label(ui, "Integrity", rec.integrity);
        value_label(ui, "Motor forward", rec.motor_forward);
        value_label(ui, "Motor turn", rec.motor_turn);
        ui.label(format!("Alive: {}", rec.alive));
    }
    ui.label("Phase: not recorded");
    ui.label("Vision: not recorded");

    for pin in window.pins() {
        let Pin { tick, kind } = *pin;
        let label = match kind {
            PinKind::Replacement { slot } => format!("Replacement slot {slot} at tick {tick}"),
            PinKind::SeasonFlip => format!("Season flip at tick {tick}"),
        };
        ui.horizontal(|ui| {
            ui.label(&label);
            jump_button(ui, tick, "Jump", &mut jump);
        });
    }
    if let Some(tick) = jump {
        view.request_jump(tick);
    }
    if view.playing || view.pending_jump.is_some() {
        ui.ctx().request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_sandbox::tick_window::{record_from_physics_row, TickWindowConfig};

    fn frame(tick: u64) -> WindowFrame {
        WindowFrame {
            tick,
            agents: vec![record_from_physics_row(
                &[0.0; xagent_brain::buffers::PHYS_STRIDE],
            )],
            food: vec![],
        }
    }

    fn window(ticks: &[u64]) -> TickWindow {
        let mut w = TickWindow::new(TickWindowConfig::default()).expect("default config");
        for &t in ticks {
            w.capture(frame(t));
        }
        w
    }

    #[test]
    fn cursor_clamped_into_window_on_advance() {
        let w = window(&[10, 20, 30]);
        let mut v = TickWindowView {
            cursor_tick: Some(5),
            ..Default::default()
        };
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(10));
        v.cursor_tick = Some(100);
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(30));
    }

    #[test]
    fn play_stops_at_last_frame() {
        let w = window(&[10, 20, 30]);
        let mut v = TickWindowView {
            cursor_tick: Some(10),
            playing: true,
            ..Default::default()
        };
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(20));
        assert!(v.playing);
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(30));
        assert!(!v.playing);
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(30));
        assert!(!v.playing);
    }

    #[test]
    fn jump_selects_nearest_at_or_after_and_past_end_goes_last() {
        let w = window(&[10, 20, 30]);
        let mut v = TickWindowView::default();
        v.request_jump(15);
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(20));
        assert!(!v.playing);
        v.request_jump(1000);
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, Some(30));
        assert!(!v.playing);
    }

    #[test]
    fn empty_window_yields_no_frame() {
        let w = window(&[]);
        let mut v = TickWindowView {
            playing: true,
            ..Default::default()
        };
        v.advance(&w, 1);
        assert_eq!(v.cursor_tick, None);
        assert!(v.current_frame(&w).is_none());
        assert!(!v.playing);
    }

    #[test]
    fn selected_agent_cleared_when_out_of_range() {
        let w = window(&[10]);
        let mut v = TickWindowView {
            selected_agent: Some(3),
            ..Default::default()
        };
        v.advance(&w, 1);
        assert_eq!(v.selected_agent, None);
        v.selected_agent = Some(0);
        v.advance(&w, 1);
        assert_eq!(v.selected_agent, Some(0));
    }

    fn run_render(view: &mut TickWindowView, w: &TickWindow) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| render(ui, view, w));
        });
    }

    #[test]
    fn render_smoke_empty_window() {
        let w = window(&[]);
        let mut v = TickWindowView::default();
        run_render(&mut v, &w);
        assert_eq!(v.cursor_tick, None);
    }

    #[test]
    fn render_smoke_populated_window() {
        let mut w = window(&[10, 20, 30]);
        w.pin(Pin {
            tick: 20,
            kind: PinKind::Replacement { slot: 1 },
        });
        w.pin(Pin {
            tick: 30,
            kind: PinKind::SeasonFlip,
        });
        let mut v = TickWindowView {
            selected_agent: Some(0),
            ..Default::default()
        };
        run_render(&mut v, &w);
        assert_eq!(v.cursor_tick, Some(10));
    }
}
