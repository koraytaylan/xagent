//! Record test for the trunk lesion matrix.
//!
//! Reads the committed `docs/measurements/trunk-lesions/lesion-matrix.json`
//! (an array of free-life reports), checks that it holds the complete matrix
//! with the expected parameters and only finite metrics, recomputes every
//! verdict by the decision rule, and compares them with the verdict table in
//! `docs/measurements/trunk-lesions/DECISION.md`.

use std::path::{Path, PathBuf};

use xagent_sandbox::free_life::metrics::{AgentMetrics, WindowMetrics};
use xagent_sandbox::headless::FreeLifeReport;

/// Seeds of the reference lineages and free lives.
const SEEDS: [u64; 4] = [5, 6, 7, 8];
/// Agents in every free life.
const AGENT_COUNT: usize = 10;
/// Simulation ticks in every free life.
const TICK_COUNT: u64 = 300_000;
/// A change is significant when `|d|` exceeds this multiple of `se`.
const SIGNIFICANCE_MULTIPLE: f64 = 2.0;
/// `se = sd(d_s) / SE_DIVISOR` (the square root of the four seeds).
const SE_DIVISOR: f64 = 2.0;
/// Brain sources of the matrix.
const SOURCES: [&str; 2] = ["fresh", "champion"];
/// The unlesioned arm every lesion is compared with.
const CONTROL_ARM: &str = "control";
/// The arm that lesions every part at once.
const ALL_ARM: &str = "all";
/// The four trunk parts, in table order; each is also a single-lesion arm.
const PARTS: [&str; 4] = ["turn-weights", "forward-weights", "memory-blend", "critic"];
/// Matrix file, relative to the workspace root.
const MATRIX_PATH: &str = "docs/measurements/trunk-lesions/lesion-matrix.json";
/// Decision record, relative to the workspace root.
const DECISION_PATH: &str = "docs/measurements/trunk-lesions/DECISION.md";
/// Flag that marks a reference lineage command.
const LINEAGE_FLAG: &str = "--generations 80";
/// Config flag every lineage command must carry.
const LINEAGE_CONFIG_FLAG: &str = "--config target/trunk-lesions/base.json";
/// Table cell for a part the capability needs.
const NEEDED: &str = "needed";
/// Table cell for a part the capability does not need.
const NOT_NEEDED: &str = "not needed";

/// A per-agent metric extractor with its printable name.
type Metric = (&'static str, fn(&AgentMetrics) -> f64);

/// Metrics whose fall or rise decides the behaviour verdict.
const MEALS: Metric = ("whole.meals", |a| count(a.whole.meals));
const DEATHS: Metric = ("whole.deaths", |a| count(a.whole.deaths));
/// Metrics whose fall in fresh brains decides the learning verdict.
const LEARNING_METRICS: [Metric; 3] = [
    ("late.pathway_food_correlation.scent", |a| {
        a.late.pathway_food_correlation.scent
    }),
    ("late.pathway_food_correlation.vision", |a| {
        a.late.pathway_food_correlation.vision
    }),
    ("late.turn_toward_food", |a| a.late.turn_toward_food),
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| panic!("workspace root above CARGO_MANIFEST_DIR"))
}

fn read_text(relative: &str) -> String {
    let path = workspace_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn load_matrix() -> Vec<FreeLifeReport> {
    serde_json::from_str(&read_text(MATRIX_PATH))
        .unwrap_or_else(|e| panic!("{MATRIX_PATH} is not an array of reports: {e}"))
}

fn count(value: u64) -> f64 {
    let small = u32::try_from(value).unwrap_or_else(|_| panic!("count {value} too large"));
    f64::from(small)
}

fn arm_of(report: &FreeLifeReport) -> String {
    match report.lesions.len() {
        0 => CONTROL_ARM.to_string(),
        1 => report.lesions[0].clone(),
        n if n == PARTS.len() && report.lesions.iter().zip(PARTS).all(|(a, b)| a == b) => {
            ALL_ARM.to_string()
        }
        _ => panic!("unexpected lesion list {:?}", report.lesions),
    }
}

fn all_arms() -> Vec<&'static str> {
    let mut arms = vec![CONTROL_ARM];
    arms.extend(PARTS);
    arms.push(ALL_ARM);
    arms
}

fn find<'a>(
    matrix: &'a [FreeLifeReport],
    source: &str,
    arm: &str,
    seed: u64,
) -> &'a FreeLifeReport {
    let hits: Vec<_> = matrix
        .iter()
        .filter(|r| r.brain_source == source && arm_of(r) == arm && r.seed == seed)
        .collect();
    assert_eq!(hits.len(), 1, "reports for {source} {arm} seed {seed}");
    hits[0]
}

fn window_finite(w: &WindowMetrics) -> bool {
    let scalars = [
        w.sighting_meal_share,
        w.turn_toward_food,
        w.turn_away_from_hazard,
        w.pathway_food_correlation.encoding,
        w.pathway_food_correlation.scent,
        w.pathway_food_correlation.vision,
        w.value_gap_in_view,
        w.value_gap_centred,
    ];
    scalars
        .iter()
        .chain(w.turn_credit.iter())
        .chain(w.forward_credit.iter())
        .all(|v| v.is_finite())
}

fn check_report(r: &FreeLifeReport) {
    let label = format!("{} {} seed {}", r.brain_source, arm_of(r), r.seed);
    assert_eq!(r.agents.len(), AGENT_COUNT, "{label}: agents");
    assert_eq!(r.agent_count, AGENT_COUNT, "{label}: agent_count");
    assert_eq!(r.ticks, TICK_COUNT, "{label}: ticks");
    let product = r.brain_ticks.checked_mul(u64::from(r.kernel_batch_size));
    assert_eq!(product, Some(r.ticks), "{label}: brain_ticks * batch");
    let champion = r.brain_source == "champion";
    assert_eq!(r.champion_node_id.is_some(), champion, "{label}: node id");
    assert_eq!(
        r.champion_generation.is_some(),
        champion,
        "{label}: generation"
    );
    for agent in &r.agents {
        assert!(
            window_finite(&agent.whole),
            "{label}: non-finite whole metric"
        );
        assert!(
            window_finite(&agent.late),
            "{label}: non-finite late metric"
        );
    }
}

fn mean(values: &[f64]) -> f64 {
    let n = u32::try_from(values.len()).unwrap_or_else(|_| panic!("too many values"));
    values.iter().sum::<f64>() / f64::from(n)
}

fn sample_sd(values: &[f64]) -> f64 {
    let m = mean(values);
    let n = u32::try_from(values.len()).unwrap_or_else(|_| panic!("too many values"));
    let sq: f64 = values.iter().map(|v| (v - m) * (v - m)).sum();
    (sq / (f64::from(n) - 1.0)).sqrt()
}

fn agent_mean(report: &FreeLifeReport, metric: Metric) -> f64 {
    let values: Vec<f64> = report.agents.iter().map(metric.1).collect();
    mean(&values)
}

/// Returns `(d, se)` for an arm against control in one source.
fn effect(matrix: &[FreeLifeReport], source: &str, arm: &str, metric: Metric) -> (f64, f64) {
    let ds: Vec<f64> = SEEDS
        .iter()
        .map(|&s| {
            agent_mean(find(matrix, source, arm, s), metric)
                - agent_mean(find(matrix, source, CONTROL_ARM, s), metric)
        })
        .collect();
    (mean(&ds), sample_sd(&ds) / SE_DIVISOR)
}

fn significant(d: f64, se: f64) -> bool {
    d != 0.0 && d.abs() > SIGNIFICANCE_MULTIPLE * se
}

fn report_effect(
    matrix: &[FreeLifeReport],
    source: &str,
    part: &str,
    metric: Metric,
) -> (f64, f64) {
    let (d, se) = effect(matrix, source, part, metric);
    println!("{part} {source} {}: d = {d:.6} ± {se:.6}", metric.0);
    (d, se)
}

fn behaviour_needs(matrix: &[FreeLifeReport], part: &str) -> bool {
    let mut needed = false;
    for source in SOURCES {
        let (d, se) = report_effect(matrix, source, part, MEALS);
        needed |= d < 0.0 && significant(d, se);
        let (d, se) = report_effect(matrix, source, part, DEATHS);
        needed |= d > 0.0 && significant(d, se);
    }
    needed
}

fn learning_needs(matrix: &[FreeLifeReport], part: &str) -> bool {
    let mut needed = false;
    for metric in LEARNING_METRICS {
        let (d, se) = report_effect(matrix, "fresh", part, metric);
        needed |= d < 0.0 && significant(d, se);
    }
    needed
}

fn cell(needed: bool) -> &'static str {
    if needed {
        NEEDED
    } else {
        NOT_NEEDED
    }
}

fn table_row(decision: &str, part: &str) -> (String, String) {
    let prefix = format!("| {part} |");
    let rows: Vec<&str> = decision
        .lines()
        .filter(|l| l.trim_start().starts_with(&prefix))
        .collect();
    assert_eq!(rows.len(), 1, "DECISION.md table rows for {part}");
    let cells: Vec<&str> = rows[0]
        .trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect();
    assert_eq!(cells.len(), 3, "DECISION.md row for {part}: {}", rows[0]);
    (cells[1].to_string(), cells[2].to_string())
}

#[test]
fn matrix_is_complete_and_finite() {
    let matrix = load_matrix();
    assert_eq!(matrix.len(), SOURCES.len() * all_arms().len() * SEEDS.len());
    for source in SOURCES {
        for arm in all_arms() {
            for seed in SEEDS {
                check_report(find(&matrix, source, arm, seed));
            }
        }
    }
}

#[test]
fn decision_table_matches_recomputed_verdicts() {
    let matrix = load_matrix();
    let decision = read_text(DECISION_PATH);
    let header = "| part | behaviour | within-life learning |";
    let lines: Vec<&str> = decision.lines().map(str::trim).collect();
    let at = lines.iter().position(|l| *l == header);
    let at = at.unwrap_or_else(|| panic!("DECISION.md lacks the verdict table header"));
    assert_eq!(lines.get(at + 1).copied(), Some("|---|---|---|"));
    for (offset, part) in PARTS.iter().enumerate() {
        let expected_prefix = format!("| {part} |");
        let row = lines.get(at + 2 + offset).copied().unwrap_or_default();
        assert!(
            row.starts_with(&expected_prefix),
            "table row {offset} is {row}"
        );
        let behaviour = cell(behaviour_needs(&matrix, part));
        let learning = cell(learning_needs(&matrix, part));
        println!("verdict {part}: behaviour {behaviour}, within-life learning {learning}");
        let (b, l) = table_row(&decision, part);
        assert_eq!(b, behaviour, "{part} behaviour verdict");
        assert_eq!(l, learning, "{part} within-life learning verdict");
    }
}

#[test]
fn lineage_commands_carry_the_base_config() {
    let decision = read_text(DECISION_PATH);
    let lineage: Vec<&str> = decision
        .lines()
        .filter(|l| l.contains(LINEAGE_FLAG))
        .collect();
    assert!(!lineage.is_empty(), "DECISION.md has no lineage command");
    for line in lineage {
        assert!(
            line.contains(LINEAGE_CONFIG_FLAG),
            "lineage line lacks base config: {line}"
        );
    }
}
