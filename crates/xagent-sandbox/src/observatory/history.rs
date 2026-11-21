//! Lineage forest, archive browser and event timeline views.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::{ArchiveRow, ObservatoryModel, TimelineEvent, TimelineKind, MAX_DISPLAY_TEXT_BYTES};

/// Every timeline kind, in toggle order.
const ALL_KINDS: [TimelineKind; 6] = [
    TimelineKind::Replacement,
    TimelineKind::Idle,
    TimelineKind::Season,
    TimelineKind::Archive,
    TimelineKind::Death,
    TimelineKind::Run,
];

/// Spaces of indentation per lineage depth level.
const INDENT_SPACES: usize = 2;

/// One occupant in the lineage forest, in pre-order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageNode {
    /// Occupant id.
    pub occupant_id: u64,
    /// Distance from the root of its tree.
    pub depth: usize,
    /// Whether the occupant's parent is no longer tracked.
    pub parent_evicted: bool,
}

/// Cuts `text` to [`MAX_DISPLAY_TEXT_BYTES`] on a char boundary.
#[must_use]
pub fn truncate_display(text: &str) -> &str {
    if text.len() <= MAX_DISPLAY_TEXT_BYTES {
        return text;
    }
    let mut end = MAX_DISPLAY_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Walks the subtree under `start` in pre-order, skipping visited occupants.
fn walk(
    start: (u64, bool),
    children: &BTreeMap<u64, Vec<u64>>,
    visited: &mut BTreeSet<u64>,
    out: &mut Vec<LineageNode>,
) {
    let mut stack = vec![(start.0, 0_usize, start.1)];
    while let Some((id, depth, parent_evicted)) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        out.push(LineageNode {
            occupant_id: id,
            depth,
            parent_evicted,
        });
        if let Some(kids) = children.get(&id) {
            for kid in kids.iter().rev() {
                if !visited.contains(kid) {
                    stack.push((*kid, depth.saturating_add(1), false));
                }
            }
        }
    }
}

/// Builds the lineage forest in pre-order; each occupant appears exactly once.
#[must_use]
pub fn lineage_forest(model: &ObservatoryModel) -> Vec<LineageNode> {
    let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut roots: Vec<(u64, bool)> = Vec::new();
    for (id, record) in &model.occupants {
        match record.parent {
            None => roots.push((*id, false)),
            Some(parent) if !model.occupants.contains_key(&parent.occupant_id) => {
                roots.push((*id, true));
            }
            Some(parent) => children.entry(parent.occupant_id).or_default().push(*id),
        }
    }
    let mut visited = BTreeSet::new();
    let mut out = Vec::with_capacity(model.occupants.len());
    for root in roots {
        walk(root, &children, &mut visited, &mut out);
    }
    for id in model.occupants.keys() {
        if !visited.contains(id) {
            walk((*id, false), &children, &mut visited, &mut out);
        }
    }
    out
}

/// Archive rows by outcome descending, NaN last, ties by entry id.
#[must_use]
pub fn archive_rows(model: &ObservatoryModel) -> Vec<&ArchiveRow> {
    let mut rows: Vec<&ArchiveRow> = model.archive.iter().collect();
    rows.sort_by(|a, b| {
        let order = match (a.outcome.is_nan(), b.outcome.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => b.outcome.total_cmp(&a.outcome),
        };
        order.then(a.entry_id.cmp(&b.entry_id))
    });
    rows
}

/// Which timeline events to show.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimelineFilter {
    /// Kinds to show; empty means all.
    pub kinds: Vec<TimelineKind>,
    /// First tick shown, inclusive.
    pub from_tick: Option<u64>,
    /// Last tick shown, inclusive.
    pub to_tick: Option<u64>,
}

/// Events matching `filter`, newest first.
#[must_use]
pub fn filtered<'a>(
    model: &'a ObservatoryModel,
    filter: &TimelineFilter,
) -> Vec<&'a TimelineEvent> {
    model
        .timeline
        .iter()
        .rev()
        .filter(|event| filter.kinds.is_empty() || filter.kinds.contains(&event.kind))
        .filter(|event| filter.from_tick.is_none_or(|from| event.tick >= from))
        .filter(|event| filter.to_tick.is_none_or(|to| event.tick <= to))
        .collect()
}

/// View state of the lineage view.
#[derive(Clone, Debug, Default)]
pub struct LineageViewState {
    /// Hide roots whose parent was evicted.
    pub hide_evicted_roots: bool,
}

/// View state of the timeline view.
#[derive(Clone, Debug, Default)]
pub struct TimelineViewState {
    /// Kinds toggled on; empty means all.
    pub kinds: Vec<TimelineKind>,
    /// Text of the first-tick field.
    pub from_text: String,
    /// Text of the last-tick field.
    pub to_text: String,
}

impl TimelineViewState {
    /// The filter the current fields describe.
    #[must_use]
    pub fn filter(&self) -> TimelineFilter {
        TimelineFilter {
            kinds: self.kinds.clone(),
            from_tick: self.from_text.trim().parse().ok(),
            to_tick: self.to_text.trim().parse().ok(),
        }
    }
}

/// Shows the lineage forest.
pub fn show_lineage(ui: &mut egui::Ui, model: &ObservatoryModel, state: &mut LineageViewState) {
    ui.checkbox(&mut state.hide_evicted_roots, "Hide evicted-parent roots");
    egui::ScrollArea::vertical().show(ui, |ui| {
        let mut skip_tree = false;
        for node in lineage_forest(model) {
            if node.depth == 0 {
                skip_tree = state.hide_evicted_roots && node.parent_evicted;
            }
            if skip_tree {
                continue;
            }
            let indent = " ".repeat(node.depth.saturating_mul(INDENT_SPACES));
            let marker = if node.parent_evicted {
                " (parent evicted)"
            } else {
                ""
            };
            ui.label(format!("{indent}#{}{marker}", node.occupant_id));
        }
    });
}

/// Shows the archive as a grid.
pub fn show_archive(ui: &mut egui::Ui, model: &ObservatoryModel) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::Grid::new("observatory_archive_grid")
            .striped(true)
            .show(ui, |ui| {
                for header in ["Entry", "Occupant", "Outcome", "Age", "Config"] {
                    ui.strong(header);
                }
                ui.end_row();
                for row in archive_rows(model) {
                    ui.label(row.entry_id.to_string());
                    ui.label(row.occupant_id.to_string());
                    ui.label(format!("{:.3}", row.outcome));
                    ui.label(row.age_ticks.to_string());
                    ui.collapsing(format!("config {}", row.entry_id), |ui| {
                        let summary = format!("{:?}", row.config);
                        ui.label(truncate_display(&summary));
                    });
                    ui.end_row();
                }
            });
    });
}

/// Shows the event timeline with filters.
pub fn show_timeline(ui: &mut egui::Ui, model: &ObservatoryModel, state: &mut TimelineViewState) {
    ui.horizontal_wrapped(|ui| {
        for kind in ALL_KINDS {
            let mut on = state.kinds.contains(&kind);
            if ui.checkbox(&mut on, format!("{kind:?}")).changed() {
                if on {
                    state.kinds.push(kind);
                } else {
                    state.kinds.retain(|held| *held != kind);
                }
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("From tick");
        ui.text_edit_singleline(&mut state.from_text);
        ui.label("To tick");
        ui.text_edit_singleline(&mut state.to_text);
    });
    let events = filtered(model, &state.filter());
    let row_height = ui.text_style_height(&egui::TextStyle::Body);
    egui::ScrollArea::vertical().show_rows(ui, row_height, events.len(), |ui, range| {
        for event in events.iter().skip(range.start).take(range.len()) {
            ui.label(format!(
                "{} {:?} {}",
                event.tick,
                event.kind,
                truncate_display(&event.text)
            ));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observatory::{
        InheritedState, ObservatoryRecord, OccupantRecord, ParentRef, ParentSource,
    };
    use xagent_shared::BrainConfig;

    /// Id of the grandparent root.
    const ROOT: u64 = 1;
    /// Id of the child.
    const CHILD: u64 = 2;
    /// Id of the grandchild.
    const GRANDCHILD: u64 = 3;
    /// Id of a second child.
    const SIBLING: u64 = 4;
    /// Id whose parent is absent.
    const ORPHAN: u64 = 5;
    /// Id of a parent that is not tracked.
    const MISSING: u64 = 99;
    /// First tick of the inclusive range.
    const FROM: u64 = 3;
    /// Last tick of the inclusive range.
    const TO: u64 = 5;
    /// Events folded in the timeline test.
    const EVENTS: u64 = 8;
    /// Every second tick is a Death event in the timeline test.
    const DEATH_TICK_PERIOD: u64 = 2;
    /// The Death tick that lies inside the inclusive range.
    const IN_RANGE_DEATH_TICK: u64 = 4;
    /// Depth of the grandchild in the family.
    const GRANDCHILD_DEPTH: usize = 2;
    /// Index of the evicted-parent root in the pre-order forest.
    const ORPHAN_INDEX: usize = 4;
    /// The higher outcome in the archive test.
    const HIGH_OUTCOME: f32 = 2.0;

    fn occupant(id: u64, parent: Option<u64>) -> ObservatoryRecord {
        ObservatoryRecord::Occupant(OccupantRecord {
            occupant_id: id,
            parent: parent.map(|occupant_id| ParentRef {
                occupant_id,
                source: ParentSource::Peer { slot: 0 },
                state: InheritedState::Birth,
            }),
            born_tick: 0,
            retired_tick: None,
            intake: Vec::new(),
            late: None,
        })
    }

    fn model_of(records: Vec<ObservatoryRecord>) -> ObservatoryModel {
        let mut model = ObservatoryModel::default();
        for record in records {
            model.fold(record);
        }
        model
    }

    #[test]
    fn history_lineage_each_occupant_once() {
        let model = model_of(vec![
            occupant(ROOT, None),
            occupant(CHILD, Some(ROOT)),
            occupant(GRANDCHILD, Some(CHILD)),
            occupant(SIBLING, Some(ROOT)),
            occupant(ORPHAN, Some(MISSING)),
        ]);
        let forest = lineage_forest(&model);
        let ids: Vec<u64> = forest.iter().map(|n| n.occupant_id).collect();
        assert_eq!(ids, vec![ROOT, CHILD, GRANDCHILD, SIBLING, ORPHAN]);
        let depths: Vec<usize> = forest.iter().map(|n| n.depth).collect();
        assert_eq!(depths, vec![0, 1, GRANDCHILD_DEPTH, 1, 0]);
        assert!(forest[ORPHAN_INDEX].parent_evicted);
        assert!(!forest[0].parent_evicted);
    }

    #[test]
    fn history_lineage_cycle_terminates() {
        let model = model_of(vec![
            occupant(CHILD, Some(GRANDCHILD)),
            occupant(GRANDCHILD, Some(CHILD)),
        ]);
        let forest = lineage_forest(&model);
        let ids: Vec<u64> = forest.iter().map(|n| n.occupant_id).collect();
        assert_eq!(ids, vec![CHILD, GRANDCHILD]);
        assert_eq!(forest[0].depth, 0);
    }

    #[test]
    fn history_archive_sorted_by_outcome() {
        let row = |entry_id: u64, outcome: f32| {
            ObservatoryRecord::Archive(ArchiveRow {
                entry_id,
                occupant_id: entry_id,
                outcome,
                age_ticks: 0,
                config: BrainConfig::default(),
            })
        };
        let model = model_of(vec![
            row(ROOT, f32::NAN),
            row(CHILD, 1.0),
            row(GRANDCHILD, HIGH_OUTCOME),
            row(SIBLING, 1.0),
        ]);
        let ids: Vec<u64> = archive_rows(&model).iter().map(|r| r.entry_id).collect();
        assert_eq!(ids, vec![GRANDCHILD, CHILD, SIBLING, ROOT]);
    }

    #[test]
    fn history_timeline_filter_kind_and_range() {
        let mut records = Vec::new();
        for tick in 0..EVENTS {
            let kind = if tick % DEATH_TICK_PERIOD == 0 {
                TimelineKind::Death
            } else {
                TimelineKind::Run
            };
            records.push(ObservatoryRecord::Event(TimelineEvent {
                tick,
                kind,
                text: String::new(),
            }));
        }
        let model = model_of(records);
        let filter = TimelineFilter {
            kinds: vec![TimelineKind::Death],
            from_tick: Some(FROM),
            to_tick: Some(TO),
        };
        let ticks: Vec<u64> = filtered(&model, &filter).iter().map(|e| e.tick).collect();
        assert_eq!(ticks, vec![IN_RANGE_DEATH_TICK]);
        let all = TimelineFilter {
            from_tick: Some(FROM),
            to_tick: Some(TO),
            ..TimelineFilter::default()
        };
        let ticks: Vec<u64> = filtered(&model, &all).iter().map(|e| e.tick).collect();
        assert_eq!(ticks, vec![TO, IN_RANGE_DEATH_TICK, FROM]);
    }

    #[test]
    fn history_truncate_on_char_boundary() {
        let text = "é".repeat(MAX_DISPLAY_TEXT_BYTES);
        let cut = truncate_display(&text);
        assert!(cut.len() <= MAX_DISPLAY_TEXT_BYTES);
        assert!(!cut.is_empty());
        assert!(cut.chars().all(|c| c == 'é'));
        assert_eq!(truncate_display("short"), "short");
    }
}
