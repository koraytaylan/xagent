//! Slot table, boundary inspector and run-progress header views.

use super::{
    BoundaryRecord, EligibilityGround, InheritedState, ObservatoryModel, ParentRef, ParentSource,
    Progress, SeasonState,
};

/// Text shown when a value is unknown.
const NOT_AVAILABLE: &str = "n/a";
/// Column headings of the slot table.
const SLOT_HEADINGS: [&str; 7] = [
    "occupant", "age", "state", "meals", "deaths", "score", "parent",
];

/// Selection state of the slot table.
#[derive(Clone, Debug, Default)]
pub struct SlotsViewState {
    /// Slot selected by the user.
    pub selected: Option<usize>,
}

/// Selection state of the boundary inspector.
#[derive(Clone, Debug, Default)]
pub struct BoundaryViewState {
    /// Tick of the boundary selected by the user.
    pub selected_tick: Option<u64>,
}

fn state_text(state: InheritedState) -> &'static str {
    match state {
        InheritedState::Birth => "birth",
        InheritedState::Learned => "learned",
    }
}

fn parent_text(parent: Option<&ParentRef>) -> String {
    match parent {
        None => NOT_AVAILABLE.to_string(),
        Some(p) => match p.source {
            ParentSource::Archive { entry_id } => {
                format!("archive #{entry_id} ({})", state_text(p.state))
            }
            ParentSource::Peer { slot } => {
                format!("peer slot {slot} ({})", state_text(p.state))
            }
        },
    }
}

fn score_text(score: Option<f32>, median: Option<f32>) -> String {
    match (score, median) {
        (Some(s), Some(m)) if s.is_finite() && m.is_finite() => {
            format!("{s:.3} ({:+.3})", s - m)
        }
        _ => NOT_AVAILABLE.to_string(),
    }
}

/// One row of seven text cells per slot.
pub fn slot_rows(model: &ObservatoryModel) -> Vec<[String; 7]> {
    model
        .slots
        .iter()
        .map(|row| {
            [
                row.occupant_id.to_string(),
                row.age_ticks.to_string(),
                if row.alive { "alive" } else { "dead" }.to_string(),
                row.meals.to_string(),
                row.deaths.to_string(),
                score_text(row.window_score, row.peer_median),
                parent_text(row.parent.as_ref()),
            ]
        })
        .collect()
}

/// Text lines describing one boundary.
pub fn boundary_lines(record: &BoundaryRecord) -> Vec<String> {
    if let Some(reason) = &record.idle_reason {
        return vec![format!("idle: {reason}")];
    }
    let mut lines = Vec::new();
    for eligible in &record.eligible {
        lines.push(match eligible.ground {
            EligibilityGround::Died => format!("slot {}: died", eligible.slot),
            EligibilityGround::BelowMedian { score, median } => format!(
                "slot {}: score {score:.3} below median {median:.3}",
                eligible.slot
            ),
        });
    }
    if let Some(slot) = record.picked {
        lines.push(format!("picked slot {slot}"));
    }
    if let Some(text) = &record.tie_break {
        lines.push(format!("tie-break: {text}"));
    }
    lines.push(parent_text(record.parent.as_ref()));
    for delta in &record.config_diff {
        lines.push(format!(
            "{}: {} -> {}",
            delta.field, delta.parent, delta.child
        ));
    }
    lines
}

/// Run progress as one line of text.
pub fn progress_text(progress: &Progress) -> String {
    let budget = progress
        .budget_ticks
        .map_or_else(|| "unlimited".to_string(), |b| b.to_string());
    let mut parts = vec![format!("tick {} / {budget}", progress.tick)];
    if let Some(rate) = progress.ticks_per_sec {
        parts.push(format!("{rate:.1} ticks/s"));
    }
    if !progress.finished {
        if let Some(next) = progress.next_boundary_tick {
            parts.push(format!(
                "next boundary in {} ticks",
                next.saturating_sub(progress.tick)
            ));
        }
    }
    parts.join("  ")
}

/// Season state as one line of text.
pub fn season_text(season: Option<&SeasonState>, tick: u64) -> String {
    let Some(season) = season else {
        return "season: n/a".to_string();
    };
    let nourishing = format!("nourishing: {:?}", season.nourishing);
    match season.next_flip_tick {
        Some(flip) => format!(
            "{nourishing}  next flip in {} ticks",
            flip.saturating_sub(tick)
        ),
        None => nourishing,
    }
}

/// Draws the slot table; the caller owns scrolling.
pub fn show_slots(ui: &mut egui::Ui, model: &ObservatoryModel, state: &mut SlotsViewState) {
    let rows = slot_rows(model);
    egui::Grid::new("observatory_slots")
        .striped(true)
        .show(ui, |ui| {
            ui.strong("slot");
            for heading in SLOT_HEADINGS {
                ui.strong(heading);
            }
            ui.end_row();
            for (slot_row, cells) in model.slots.iter().zip(rows.iter()) {
                let chosen = state.selected == Some(slot_row.slot);
                if ui
                    .selectable_label(chosen, slot_row.slot.to_string())
                    .clicked()
                {
                    state.selected = Some(slot_row.slot);
                }
                for cell in cells {
                    ui.label(cell);
                }
                ui.end_row();
            }
        });
}

/// Draws the boundary inspector, newest first; the caller owns scrolling.
pub fn show_boundaries(ui: &mut egui::Ui, model: &ObservatoryModel, state: &mut BoundaryViewState) {
    for record in model.boundaries.iter().rev() {
        let title = format!("tick {}", record.tick);
        let header = egui::CollapsingHeader::new(title)
            .id_salt(("observatory_boundary", record.tick))
            .show(ui, |ui| {
                for line in boundary_lines(record) {
                    ui.label(line);
                }
            });
        if header.header_response.clicked() {
            state.selected_tick = Some(record.tick);
        }
    }
}

/// Draws the progress and season header as one wrapped row.
pub fn show_header(ui: &mut egui::Ui, model: &ObservatoryModel) {
    ui.horizontal_wrapped(|ui| {
        ui.label(progress_text(&model.progress));
        ui.separator();
        ui.label(season_text(model.season.as_ref(), model.progress.tick));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observatory::{ConfigDelta, EligibleSlot, SlotRow};
    use xagent_shared::FoodKind;

    fn slot(
        slot: usize,
        alive: bool,
        score: Option<f32>,
        median: Option<f32>,
        parent: Option<ParentRef>,
    ) -> SlotRow {
        SlotRow {
            slot,
            occupant_id: 10 + u64::try_from(slot).unwrap_or(0),
            age_ticks: 500,
            alive,
            meals: 4,
            deaths: 2,
            window_score: score,
            peer_median: median,
            parent,
        }
    }

    fn archive_parent() -> ParentRef {
        ParentRef {
            occupant_id: 1,
            source: ParentSource::Archive { entry_id: 7 },
            state: InheritedState::Birth,
        }
    }

    fn peer_parent() -> ParentRef {
        ParentRef {
            occupant_id: 2,
            source: ParentSource::Peer { slot: 3 },
            state: InheritedState::Learned,
        }
    }

    #[test]
    fn tables_slot_rows_show_every_column() {
        let model = ObservatoryModel {
            slots: vec![
                slot(0, true, Some(1.5), Some(2.0), Some(archive_parent())),
                slot(1, false, Some(2.5), None, Some(peer_parent())),
                slot(2, true, None, None, None),
            ],
            ..ObservatoryModel::default()
        };
        let rows = slot_rows(&model);
        assert_eq!(
            rows[0],
            [
                "10",
                "500",
                "alive",
                "4",
                "2",
                "1.500 (-0.500)",
                "archive #7 (birth)"
            ]
        );
        assert_eq!(rows[1][2], "dead");
        assert_eq!(rows[1][5], "n/a");
        assert_eq!(rows[1][6], "peer slot 3 (learned)");
        assert_eq!(rows[2][5], "n/a");
        assert_eq!(rows[2][6], "n/a");
    }

    #[test]
    fn tables_boundary_lines_replacement() {
        let record = BoundaryRecord {
            tick: 100,
            eligible: vec![
                EligibleSlot {
                    slot: 0,
                    ground: EligibilityGround::Died,
                },
                EligibleSlot {
                    slot: 2,
                    ground: EligibilityGround::BelowMedian {
                        score: 0.25,
                        median: 1.0,
                    },
                },
            ],
            picked: Some(2),
            tie_break: Some("lowest slot".to_string()),
            parent: Some(peer_parent()),
            config_diff: vec![ConfigDelta {
                field: "lr".to_string(),
                parent: "0.1".to_string(),
                child: "0.2".to_string(),
            }],
            idle_reason: None,
        };
        assert_eq!(
            boundary_lines(&record),
            vec![
                "slot 0: died",
                "slot 2: score 0.250 below median 1.000",
                "picked slot 2",
                "tie-break: lowest slot",
                "peer slot 3 (learned)",
                "lr: 0.1 -> 0.2",
            ]
        );
    }

    #[test]
    fn tables_boundary_lines_idle() {
        let record = BoundaryRecord {
            tick: 5,
            eligible: Vec::new(),
            picked: None,
            tie_break: None,
            parent: None,
            config_diff: Vec::new(),
            idle_reason: Some("no eligible slot".to_string()),
        };
        assert_eq!(boundary_lines(&record), vec!["idle: no eligible slot"]);
    }

    #[test]
    fn tables_progress_text_never_underflows() {
        let progress = Progress {
            tick: 50,
            budget_ticks: None,
            ticks_per_sec: None,
            next_boundary_tick: Some(40),
            finished: false,
        };
        let text = progress_text(&progress);
        assert!(text.contains("tick 50 / unlimited"));
        assert!(text.contains("next boundary in 0 ticks"));
        assert!(!text.contains("ticks/s"));
        let live = Progress {
            budget_ticks: Some(100),
            ticks_per_sec: Some(12.34),
            ..progress
        };
        let text = progress_text(&live);
        assert!(text.contains("tick 50 / 100"));
        assert!(text.contains("12.3 ticks/s"));
    }

    #[test]
    fn tables_season_text_counts_to_flip() {
        let season = SeasonState {
            nourishing: FoodKind::First,
            next_flip_tick: Some(130),
        };
        let text = season_text(Some(&season), 100);
        assert!(text.contains("nourishing: First"));
        assert!(text.contains("next flip in 30 ticks"));
        assert!(season_text(Some(&season), 200).contains("next flip in 0 ticks"));
        assert_eq!(season_text(None, 0), "season: n/a");
    }
}
