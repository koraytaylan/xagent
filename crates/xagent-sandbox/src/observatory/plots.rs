//! Population charts and the learning view.

use super::{ObservatoryModel, TimelineKind};

/// Smallest chart width, used when the available width is not finite.
pub const MIN_CHART_WIDTH: f32 = 200.0;
/// Height of every chart.
pub const CHART_HEIGHT: f32 = 150.0;
/// Background grey level of a chart.
const CHART_BACKGROUND_GREY: u8 = 20;
/// Stroke width of a series line.
const SERIES_STROKE_WIDTH: f32 = 1.5;
/// Stroke width of a marker line.
const MARKER_STROKE_WIDTH: f32 = 1.0;
/// Inner padding of the plot area.
const CHART_PADDING: f32 = 4.0;
/// Colour of replacement markers.
const REPLACEMENT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 140, 50);
/// Colour of season flip markers.
const SEASON_FLIP_COLOR: egui::Color32 = egui::Color32::from_rgb(80, 200, 220);
/// Series colours, cycled by series index.
const SERIES_COLORS: [egui::Color32; 4] = [
    egui::Color32::from_rgb(50, 200, 80),
    egui::Color32::from_rgb(80, 140, 255),
    egui::Color32::from_rgb(255, 220, 50),
    egui::Color32::from_rgb(200, 80, 200),
];
/// Text shown for a measure that has no value.
const NOT_AVAILABLE: &str = "n/a";
/// Decimal places shown for late-window measures.
const LATE_WINDOW_DECIMALS: usize = 3;

/// One labelled line of a chart.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartSeries {
    /// Series label.
    pub label: String,
    /// Points as (x, y).
    pub points: Vec<(f64, f64)>,
}

/// Kind of an axis marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    /// A slot was replaced.
    Replacement,
    /// The season flipped.
    SeasonFlip,
}

/// A vertical marker at a tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisMarker {
    /// Tick of the marker.
    pub tick: u64,
    /// Kind of the marker.
    pub kind: MarkerKind,
}

/// Which occupants the learning view plots.
#[derive(Clone, Debug, Default)]
pub struct LearningViewState {
    /// Selected occupant ids.
    pub selected: Vec<u64>,
}

/// Converts a tick or count to `f64` for plotting; precision loss above 2^53
/// is irrelevant for a chart axis.
#[allow(clippy::cast_precision_loss)]
fn tick_to_f64(value: u64) -> f64 {
    value as f64
}

/// Narrows a plot coordinate to the single precision egui uses.
#[allow(clippy::cast_possible_truncation)]
fn to_screen(value: f64) -> f32 {
    value as f32
}

/// Markers for replacements and season flips, sorted by tick.
#[must_use]
pub fn axis_markers(model: &ObservatoryModel) -> Vec<AxisMarker> {
    let replacements = model
        .boundaries
        .iter()
        .filter(|boundary| boundary.picked.is_some())
        .map(|boundary| AxisMarker {
            tick: boundary.tick,
            kind: MarkerKind::Replacement,
        });
    let flips = model
        .timeline
        .iter()
        .filter(|event| event.kind == TimelineKind::Season)
        .map(|event| AxisMarker {
            tick: event.tick,
            kind: MarkerKind::SeasonFlip,
        });
    let mut markers: Vec<AxisMarker> = replacements.chain(flips).collect();
    markers.sort_by_key(|marker| marker.tick);
    markers
}

fn series_of(
    model: &ObservatoryModel,
    label: &str,
    value: impl Fn(&super::SeriesPoint) -> Option<f64>,
) -> ChartSeries {
    let points = model
        .series
        .iter()
        .filter_map(|point| {
            value(point)
                .filter(|v| v.is_finite())
                .map(|v| (tick_to_f64(point.tick), v))
        })
        .collect();
    ChartSeries {
        label: label.to_owned(),
        points,
    }
}

/// The series of the four population charts.
#[must_use]
pub fn chart_series(model: &ObservatoryModel) -> [Vec<ChartSeries>; 4] {
    [
        vec![
            series_of(model, "food per agent-tick", |p| {
                Some(f64::from(p.food_per_agent_tick))
            }),
            series_of(model, "deaths per agent-tick", |p| {
                Some(f64::from(p.deaths_per_agent_tick))
            }),
        ],
        vec![series_of(model, "window score spread", |p| {
            p.score_spread.map(f64::from)
        })],
        vec![series_of(model, "retired lifespan", |p| {
            p.retired_lifespan.map(tick_to_f64)
        })],
        vec![
            series_of(model, "archive best", |p| p.archive_best.map(f64::from)),
            series_of(model, "archive median", |p| p.archive_median.map(f64::from)),
        ],
    ]
}

/// One cumulative-intake series per selected tracked occupant.
#[must_use]
pub fn intake_curves(model: &ObservatoryModel, selected: &[u64]) -> Vec<ChartSeries> {
    selected
        .iter()
        .filter_map(|id| model.occupants.get(id))
        .map(|occupant| ChartSeries {
            label: format!("occupant {}", occupant.occupant_id),
            points: occupant
                .intake
                .iter()
                .map(|&(age, meals)| (tick_to_f64(age), f64::from(meals)))
                .collect(),
        })
        .collect()
}

/// Late-window measures per tracked occupant.
#[must_use]
pub fn late_window_rows(model: &ObservatoryModel) -> Vec<[String; 4]> {
    model
        .occupants
        .values()
        .map(|occupant| match &occupant.late {
            Some(late) => [
                occupant.occupant_id.to_string(),
                format!("{:.*}", LATE_WINDOW_DECIMALS, late.meals_per_alive),
                format!("{:.*}", LATE_WINDOW_DECIMALS, late.turn_toward_food),
                format!("{:.*}", LATE_WINDOW_DECIMALS, late.sighting_meal_share),
            ],
            None => [
                occupant.occupant_id.to_string(),
                NOT_AVAILABLE.to_owned(),
                NOT_AVAILABLE.to_owned(),
                NOT_AVAILABLE.to_owned(),
            ],
        })
        .collect()
}

/// Draws one line chart with a vertical line per marker.
pub fn line_chart(ui: &mut egui::Ui, title: &str, series: &[ChartSeries], markers: &[AxisMarker]) {
    ui.label(title);
    let available = ui.available_width();
    let width = if available.is_finite() {
        available.max(MIN_CHART_WIDTH)
    } else {
        MIN_CHART_WIDTH
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, CHART_HEIGHT), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(CHART_BACKGROUND_GREY));
    let plot = rect.shrink(CHART_PADDING);

    let all = series.iter().flat_map(|s| s.points.iter());
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in all {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() {
        return;
    }
    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(f64::EPSILON);
    let to_x = |x: f64| plot.left() + to_screen((x - min_x) / span_x) * plot.width();
    let to_y = |y: f64| plot.bottom() - to_screen((y - min_y) / span_y) * plot.height();

    for marker in markers {
        let x = tick_to_f64(marker.tick);
        if x < min_x || x > max_x {
            continue;
        }
        let color = match marker.kind {
            MarkerKind::Replacement => REPLACEMENT_COLOR,
            MarkerKind::SeasonFlip => SEASON_FLIP_COLOR,
        };
        painter.vline(
            to_x(x),
            plot.y_range(),
            egui::Stroke::new(MARKER_STROKE_WIDTH, color),
        );
    }
    for (index, line) in series.iter().enumerate() {
        let color = SERIES_COLORS[index % SERIES_COLORS.len()];
        let points: Vec<egui::Pos2> = line
            .points
            .iter()
            .map(|&(x, y)| egui::pos2(to_x(x), to_y(y)))
            .collect();
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(SERIES_STROKE_WIDTH, color),
        ));
    }
    let legend = series
        .iter()
        .map(|s| s.label.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    ui.small(legend);
}

/// Draws the four population charts.
pub fn show_charts(ui: &mut egui::Ui, model: &ObservatoryModel) {
    let markers = axis_markers(model);
    let charts = chart_series(model);
    let titles = [
        "Food and deaths per agent-tick",
        "Window score spread",
        "Retired occupant lifespan",
        "Archive best and median",
    ];
    for (title, series) in titles.iter().zip(charts.iter()) {
        line_chart(ui, title, series, &markers);
    }
}

/// Draws the learning view.
pub fn show_learning(ui: &mut egui::Ui, model: &ObservatoryModel, state: &mut LearningViewState) {
    for &id in model.occupants.keys() {
        let mut checked = state.selected.contains(&id);
        if ui
            .checkbox(&mut checked, format!("occupant {id}"))
            .changed()
        {
            if checked {
                state.selected.push(id);
            } else {
                state.selected.retain(|held| *held != id);
            }
        }
    }
    let curves = intake_curves(model, &state.selected);
    line_chart(ui, "Cumulative intake by age", &curves, &[]);
    egui::Grid::new("late_window_grid").show(ui, |ui| {
        for header in [
            "occupant",
            "meals per alive tick",
            "turn toward food",
            "sighting meal share",
        ] {
            ui.strong(header);
        }
        ui.end_row();
        for row in late_window_rows(model) {
            for cell in row {
                ui.label(cell);
            }
            ui.end_row();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observatory::{
        BoundaryRecord, LateWindow, ObservatoryRecord, OccupantRecord, SeriesPoint, TimelineEvent,
    };

    /// Tick of the replacement boundary.
    const REPLACE_TICK: u64 = 10;
    /// Tick of the idle boundary.
    const IDLE_TICK: u64 = 20;
    /// Tick of the season flip.
    const FLIP_TICK: u64 = 5;
    /// Occupant used in intake tests.
    const OCCUPANT_A: u64 = 1;
    /// Second occupant.
    const OCCUPANT_B: u64 = 2;

    fn boundary(tick: u64, picked: Option<usize>) -> ObservatoryRecord {
        ObservatoryRecord::Boundary(BoundaryRecord {
            tick,
            eligible: Vec::new(),
            picked,
            tie_break: None,
            parent: None,
            config_diff: Vec::new(),
            idle_reason: None,
        })
    }

    fn point(tick: u64) -> SeriesPoint {
        SeriesPoint {
            tick,
            food_per_agent_tick: 0.5,
            deaths_per_agent_tick: 0.25,
            score_spread: None,
            retired_lifespan: None,
            archive_best: Some(f32::NAN),
            archive_median: Some(1.0),
        }
    }

    fn occupant(id: u64, late: Option<LateWindow>) -> OccupantRecord {
        OccupantRecord {
            occupant_id: id,
            parent: None,
            born_tick: 0,
            retired_tick: None,
            intake: Vec::new(),
            late,
        }
    }

    #[test]
    fn plots_markers_at_replacements_and_flips() {
        let mut model = ObservatoryModel::default();
        model.fold(boundary(IDLE_TICK, None));
        model.fold(boundary(REPLACE_TICK, Some(0)));
        model.fold(ObservatoryRecord::Event(TimelineEvent {
            tick: FLIP_TICK,
            kind: TimelineKind::Season,
            text: String::new(),
        }));
        model.fold(ObservatoryRecord::Event(TimelineEvent {
            tick: IDLE_TICK,
            kind: TimelineKind::Idle,
            text: String::new(),
        }));
        assert_eq!(
            axis_markers(&model),
            vec![
                AxisMarker {
                    tick: FLIP_TICK,
                    kind: MarkerKind::SeasonFlip
                },
                AxisMarker {
                    tick: REPLACE_TICK,
                    kind: MarkerKind::Replacement
                },
            ]
        );
    }

    #[test]
    fn plots_chart_series_cover_four_charts() {
        let mut model = ObservatoryModel::default();
        let mut with_values = point(2);
        with_values.score_spread = Some(3.0);
        with_values.retired_lifespan = Some(7);
        model.fold(ObservatoryRecord::Series(point(1)));
        model.fold(ObservatoryRecord::Series(with_values));
        let charts = chart_series(&model);
        assert_eq!(charts[0].len(), 2);
        assert_eq!(charts[0][0].label, "food per agent-tick");
        assert_eq!(charts[0][1].label, "deaths per agent-tick");
        assert_eq!(charts[1][0].label, "window score spread");
        assert_eq!(charts[2][0].label, "retired lifespan");
        assert_eq!(charts[0][0].points, vec![(1.0, 0.5), (2.0, 0.5)]);
        assert_eq!(charts[0][1].points, vec![(1.0, 0.25), (2.0, 0.25)]);
        assert_eq!(charts[1][0].points, vec![(2.0, 3.0)]);
        assert_eq!(charts[2][0].points, vec![(2.0, 7.0)]);
        assert_eq!(charts[3][0].label, "archive best");
        assert!(charts[3][0].points.is_empty());
        assert_eq!(charts[3][1].label, "archive median");
        assert_eq!(charts[3][1].points.len(), 2);
    }

    #[test]
    fn plots_intake_curves_use_age() {
        let mut model = ObservatoryModel::default();
        model.fold(ObservatoryRecord::Occupant(occupant(OCCUPANT_A, None)));
        model.fold(ObservatoryRecord::Occupant(occupant(OCCUPANT_B, None)));
        for (id, age, meals) in [(OCCUPANT_A, 3, 1), (OCCUPANT_A, 6, 4), (OCCUPANT_B, 1, 1)] {
            model.fold(ObservatoryRecord::Intake {
                occupant_id: id,
                age_ticks: age,
                meals,
            });
        }
        let curves = intake_curves(&model, &[OCCUPANT_A]);
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].points, vec![(3.0, 1.0), (6.0, 4.0)]);
    }

    #[test]
    fn plots_late_window_rows_list_measures() {
        let mut model = ObservatoryModel::default();
        model.fold(ObservatoryRecord::Occupant(occupant(
            OCCUPANT_A,
            Some(LateWindow {
                meals_per_alive: 0.5,
                turn_toward_food: 0.25,
                sighting_meal_share: 0.75,
            }),
        )));
        model.fold(ObservatoryRecord::Occupant(occupant(OCCUPANT_B, None)));
        let rows = late_window_rows(&model);
        assert_eq!(rows[0], ["1", "0.500", "0.250", "0.750"].map(String::from));
        assert_eq!(rows[1], ["2", "n/a", "n/a", "n/a"].map(String::from));
    }
}
