//! Recomputes the gate verdict from the eight saved per-run summaries.
//!
//! Reads no kernel and writes no run state; only `--write-bundle` writes a file.

use std::collections::BTreeSet;
use std::fmt;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

use clap::Parser;
use xagent_sandbox::food_kind_metrics::{AgentFoodKindSummary, Arm, RunFoodKindSummary};
use xagent_shared::FoodKind;

/// Seeds every verdict must cover.
const SEEDS: RangeInclusive<u64> = 5..=8;
/// Ticks every run must have simulated.
const REQUIRED_TICKS: u64 = 300_000;
/// Runs a verdict needs: two arms times four seeds.
const REQUIRED_RUNS: usize = 8;
/// Agents every run must hold.
const REQUIRED_AGENTS: usize = 10;
/// Seeds on which a comparison must hold.
const SEED_MAJORITY: usize = 3;
/// First quarter of the run, the baseline share for the meal-share comparison.
const FIRST_QUARTER: usize = 0;
/// Last quarter of the run, the late share compared against the start.
const LAST_QUARTER: usize = 3;
/// Minimum mean readout balanced accuracy per seed.
const READOUT_MIN_ACCURACY: f32 = 0.70;
/// Maximum mean shuffled-label readout accuracy per seed.
const READOUT_MAX_SHUFFLED: f32 = 0.55;
/// Minimum learning agents (of 40) with a positive signed value, per pair.
const LINK_MIN_AGENTS: usize = 24;

/// Command line of the verdict binary.
#[derive(Parser, Debug)]
#[command(about = "Recompute the gate verdict from eight run summaries")]
struct Cli {
    /// Directory of run files or a bundle file.
    input: PathBuf,
    /// Write the validated runs as a bundle file.
    #[arg(long)]
    write_bundle: Option<PathBuf>,
    /// Fail unless the verdict text equals this.
    #[arg(long)]
    expect_verdict: Option<String>,
}

/// Gate verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pass,
    FailReadout,
    FailCredit,
    FailWeightChange,
}

impl fmt::Display for Verdict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::FailReadout => "FAIL readout",
            Verdict::FailCredit => "FAIL credit",
            Verdict::FailWeightChange => "FAIL weight-change",
        })
    }
}

impl FromStr for Verdict {
    type Err = VerdictError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "PASS" => Ok(Verdict::Pass),
            "FAIL readout" => Ok(Verdict::FailReadout),
            "FAIL credit" => Ok(Verdict::FailCredit),
            "FAIL weight-change" => Ok(Verdict::FailWeightChange),
            other => Err(VerdictError::Invalid(format!(
                "unknown verdict text {other:?}"
            ))),
        }
    }
}

/// Failure to read, validate or match.
#[derive(Debug)]
enum VerdictError {
    Io(String),
    Parse(String),
    Invalid(String),
    Mismatch { expected: Verdict, actual: Verdict },
}

impl fmt::Display for VerdictError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerdictError::Io(message) => write!(formatter, "io error: {message}"),
            VerdictError::Parse(message) => write!(formatter, "parse error: {message}"),
            VerdictError::Invalid(message) => write!(formatter, "invalid input: {message}"),
            VerdictError::Mismatch { expected, actual } => {
                write!(
                    formatter,
                    "verdict mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for VerdictError {}

fn invalid<T>(msg: impl Into<String>) -> Result<T, VerdictError> {
    Err(VerdictError::Invalid(msg.into()))
}

fn file_name(arm: Arm, seed: u64) -> String {
    let prefix = match arm {
        Arm::Learning => "learning",
        Arm::Frozen => "frozen",
    };
    format!("{prefix}-seed{seed}.json")
}

fn read_run(path: &Path) -> Result<RunFoodKindSummary, VerdictError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| VerdictError::Parse(format!("{}: {e}", path.display())))
}

/// Loads runs from an exact-name directory or a bundle file.
fn load_input(path: &Path) -> Result<Vec<RunFoodKindSummary>, VerdictError> {
    if path.is_dir() {
        let expected: BTreeSet<String> = SEEDS
            .flat_map(|seed| [file_name(Arm::Learning, seed), file_name(Arm::Frozen, seed)])
            .collect();
        let mut found = BTreeSet::new();
        let entries = std::fs::read_dir(path)
            .map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))?;
        for entry in entries {
            let entry = entry.map_err(|e| VerdictError::Io(e.to_string()))?;
            found.insert(entry.file_name().to_string_lossy().into_owned());
        }
        if found != expected {
            return invalid(format!(
                "directory must hold exactly {expected:?}, found {found:?}"
            ));
        }
        expected
            .iter()
            .map(|name| read_run(&path.join(name)))
            .collect()
    } else {
        let text = std::fs::read_to_string(path)
            .map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text)
            .map_err(|e| VerdictError::Parse(format!("{}: {e}", path.display())))
    }
}

/// Number of food kinds a pair statistic covers: First and Second.
const PAIR_KINDS: usize = 2;

fn agent_floats(agent: &AgentFoodKindSummary) -> Vec<f32> {
    let optional = [
        agent.turn_toward_nourishing_whole,
        agent.turn_toward_nourishing_late,
        agent.turn_away_harmful_whole,
        agent.turn_away_harmful_late,
        agent.readout_balanced_accuracy,
        agent.readout_shuffled_accuracy,
    ];
    let mut values: Vec<f32> = optional.into_iter().flatten().collect();
    values.extend([
        agent.signed_credit_pair_first,
        agent.signed_credit_pair_second,
        agent.signed_weight_difference_pair_first,
        agent.signed_weight_difference_pair_second,
    ]);
    values
}

/// Checks the eight runs are complete and well formed.
fn validate_runs(runs: &[RunFoodKindSummary]) -> Result<(), VerdictError> {
    for run in runs {
        if !SEEDS.contains(&run.seed) {
            return invalid(format!("unexpected seed {}", run.seed));
        }
        let tag = format!("seed {} {:?}", run.seed, run.arm);
        if runs
            .iter()
            .filter(|r| r.seed == run.seed && r.arm == run.arm)
            .count()
            != 1
        {
            return invalid(format!("duplicate run for {tag}"));
        }
        if run.ticks != REQUIRED_TICKS {
            return invalid(format!("{tag}: ticks {} != {REQUIRED_TICKS}", run.ticks));
        }
        if run.agent_count != REQUIRED_AGENTS || run.agents.len() != REQUIRED_AGENTS {
            return invalid(format!("{tag}: agent count is not {REQUIRED_AGENTS}"));
        }
        if !run.fresh_brains {
            return invalid(format!("{tag}: brains were not fresh"));
        }
        for agent in &run.agents {
            if agent.turn_toward_nourishing_whole.is_none()
                || agent.turn_toward_nourishing_late.is_none()
                || agent.turn_away_harmful_whole.is_none()
                || agent.turn_away_harmful_late.is_none()
            {
                return invalid(format!("{tag}: missing turn field"));
            }
            if agent_floats(agent).iter().any(|x| !x.is_finite()) {
                return invalid(format!("{tag}: non-finite value"));
            }
        }
    }
    for seed in SEEDS {
        let mut kinds = Vec::new();
        for arm in [Arm::Learning, Arm::Frozen] {
            match runs.iter().find(|r| r.seed == seed && r.arm == arm) {
                Some(run) => kinds.push(run.nourishing_kind),
                None => return invalid(format!("missing run for seed {seed} {arm:?}")),
            }
        }
        if kinds[0] != kinds[1] {
            return invalid(format!("seed {seed}: arms disagree on nourishing kind"));
        }
    }
    if runs.len() != REQUIRED_RUNS {
        return invalid(format!("expected exactly {REQUIRED_RUNS} runs"));
    }
    Ok(())
}

fn find(
    runs: &[RunFoodKindSummary],
    seed: u64,
    arm: Arm,
) -> Result<&RunFoodKindSummary, VerdictError> {
    runs.iter()
        .find(|r| r.seed == seed && r.arm == arm)
        .ok_or_else(|| VerdictError::Invalid(format!("missing run for seed {seed} {arm:?}")))
}

/// Mean of `values`; zero when empty.
#[allow(clippy::cast_possible_truncation)]
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    let count = f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
    (sum / count) as f32
}

fn late_toward(agent: &AgentFoodKindSummary) -> f32 {
    agent.turn_toward_nourishing_late.unwrap_or_default()
}

fn late_away(agent: &AgentFoodKindSummary) -> f32 {
    agent.turn_away_harmful_late.unwrap_or_default()
}

/// Pooled nourishing share of meals in `quarter`; `None` without meals.
#[allow(clippy::cast_possible_truncation)]
fn pooled_share(agents: &[AgentFoodKindSummary], quarter: usize) -> Option<f32> {
    let (mut nourishing, mut harmful) = (0.0f64, 0.0f64);
    for agent in agents {
        nourishing += f64::from(agent.nourishing_meals_by_quarter[quarter]);
        harmful += f64::from(agent.harmful_meals_by_quarter[quarter]);
    }
    if nourishing + harmful == 0.0 {
        None
    } else {
        Some((nourishing / (nourishing + harmful)) as f32)
    }
}

fn greater(left: Option<f32>, right: Option<f32>) -> bool {
    matches!((left, right), (Some(high), Some(low)) if high > low)
}

/// Per-seed values of the report.
#[derive(Debug, Clone, PartialEq)]
struct SeedRow {
    seed: u64,
    toward_learning: f32,
    toward_frozen: f32,
    away_learning: f32,
    away_frozen: f32,
    share_q0_learning: Option<f32>,
    share_q3_learning: Option<f32>,
    share_q3_frozen: Option<f32>,
    readout_accuracy: Option<f32>,
    readout_shuffled: Option<f32>,
}

/// All values behind the verdict.
#[derive(Debug, Clone, PartialEq)]
struct GateReport {
    rows: Vec<SeedRow>,
    toward_pooled: f32,
    away_pooled: f32,
    credit_agents: [usize; PAIR_KINDS],
    weight_agents: [usize; PAIR_KINDS],
    b1: bool,
    b2: bool,
    b3: bool,
    readout_ok: bool,
    credit_ok: bool,
    weight_ok: bool,
}

fn mean_option(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let all: Option<Vec<f32>> = values.collect();
    all.map(|v| mean(&v))
}

fn gate_report(runs: &[RunFoodKindSummary]) -> Result<GateReport, VerdictError> {
    validate_runs(runs)?;
    let mut rows = Vec::new();
    let mut learning_agents: Vec<(&AgentFoodKindSummary, FoodKind)> = Vec::new();
    for seed in SEEDS {
        let learning = find(runs, seed, Arm::Learning)?;
        let frozen = find(runs, seed, Arm::Frozen)?;
        learning_agents.extend(
            learning
                .agents
                .iter()
                .map(|a| (a, learning.nourishing_kind)),
        );
        let col = |agents: &[AgentFoodKindSummary], g: fn(&AgentFoodKindSummary) -> f32| {
            mean(&agents.iter().map(g).collect::<Vec<_>>())
        };
        rows.push(SeedRow {
            seed,
            toward_learning: col(&learning.agents, late_toward),
            toward_frozen: col(&frozen.agents, late_toward),
            away_learning: col(&learning.agents, late_away),
            away_frozen: col(&frozen.agents, late_away),
            share_q0_learning: pooled_share(&learning.agents, FIRST_QUARTER),
            share_q3_learning: pooled_share(&learning.agents, LAST_QUARTER),
            share_q3_frozen: pooled_share(&frozen.agents, LAST_QUARTER),
            readout_accuracy: mean_option(
                learning.agents.iter().map(|a| a.readout_balanced_accuracy),
            ),
            readout_shuffled: mean_option(
                learning.agents.iter().map(|a| a.readout_shuffled_accuracy),
            ),
        });
    }
    let majority =
        |pred: &dyn Fn(&SeedRow) -> bool| rows.iter().filter(|r| pred(r)).count() >= SEED_MAJORITY;
    let toward_pooled = mean(
        &learning_agents
            .iter()
            .map(|(a, _)| late_toward(a))
            .collect::<Vec<_>>(),
    );
    let away_pooled = mean(
        &learning_agents
            .iter()
            .map(|(a, _)| late_away(a))
            .collect::<Vec<_>>(),
    );
    let b1 = toward_pooled > 0.0 && majority(&|r| r.toward_learning > r.toward_frozen);
    let b2 = away_pooled > 0.0 && majority(&|r| r.away_learning > r.away_frozen);
    let b3 = majority(&|r| greater(r.share_q3_learning, r.share_q0_learning))
        && majority(&|r| greater(r.share_q3_learning, r.share_q3_frozen));
    let readout_ok = rows.iter().all(|r| {
        matches!((r.readout_accuracy, r.readout_shuffled),
            (Some(accuracy), Some(shuffled))
                if accuracy >= READOUT_MIN_ACCURACY && shuffled <= READOUT_MAX_SHUFFLED)
    });
    let count = |pick: fn(&AgentFoodKindSummary) -> [f32; PAIR_KINDS]| {
        let mut counts = [0usize; PAIR_KINDS];
        for (agent, nourishing_kind) in &learning_agents {
            let raw = pick(agent);
            for (index, kind) in [FoodKind::First, FoodKind::Second].into_iter().enumerate() {
                if xagent_sandbox::food_kind_metrics::signed_for_nourishing(
                    raw[index],
                    kind,
                    *nourishing_kind,
                ) > 0.0
                {
                    counts[index] += 1;
                }
            }
        }
        counts
    };
    let credit_agents = count(|a| [a.signed_credit_pair_first, a.signed_credit_pair_second]);
    let weight_agents = count(|a| {
        [
            a.signed_weight_difference_pair_first,
            a.signed_weight_difference_pair_second,
        ]
    });
    Ok(GateReport {
        rows,
        toward_pooled,
        away_pooled,
        credit_agents,
        weight_agents,
        b1,
        b2,
        b3,
        readout_ok,
        credit_ok: credit_agents.iter().all(|&c| c >= LINK_MIN_AGENTS),
        weight_ok: weight_agents.iter().all(|&c| c >= LINK_MIN_AGENTS),
    })
}

fn verdict(runs: &[RunFoodKindSummary]) -> Result<Verdict, VerdictError> {
    let report = gate_report(runs)?;
    Ok(if !report.readout_ok {
        Verdict::FailReadout
    } else if !report.credit_ok {
        Verdict::FailCredit
    } else if !(report.weight_ok && report.b1 && report.b2 && report.b3) {
        Verdict::FailWeightChange
    } else {
        Verdict::Pass
    })
}

fn fmt_opt(value: Option<f32>) -> String {
    value.map_or_else(|| "none".to_string(), |shown| format!("{shown:.4}"))
}

impl fmt::Display for GateReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "seed | B1 learn | B1 frozen | B2 learn | B2 frozen | q0 share | q3 share | q3 frozen | readout | shuffled"
        )?;
        for row in &self.rows {
            writeln!(
                formatter,
                "{} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {} | {} | {} | {}",
                row.seed,
                row.toward_learning,
                row.toward_frozen,
                row.away_learning,
                row.away_frozen,
                fmt_opt(row.share_q0_learning),
                fmt_opt(row.share_q3_learning),
                fmt_opt(row.share_q3_frozen),
                fmt_opt(row.readout_accuracy),
                fmt_opt(row.readout_shuffled),
            )?;
        }
        writeln!(
            formatter,
            "pooled late toward {:.4}, away {:.4}; B1 {} B2 {} B3 {}",
            self.toward_pooled, self.away_pooled, self.b1, self.b2, self.b3
        )?;
        writeln!(
            formatter,
            "credit agents (first, second): {}, {} (need {LINK_MIN_AGENTS})",
            self.credit_agents[0], self.credit_agents[1]
        )?;
        write!(
            formatter,
            "weight agents (first, second): {}, {} (need {LINK_MIN_AGENTS})",
            self.weight_agents[0], self.weight_agents[1]
        )
    }
}

/// Writes the validated runs sorted by seed then arm as a JSON array.
fn write_bundle(runs: &[RunFoodKindSummary], path: &Path) -> Result<(), VerdictError> {
    validate_runs(runs)?;
    let mut sorted = runs.to_vec();
    sorted.sort_by_key(|r| (r.seed, r.arm == Arm::Frozen));
    let text =
        serde_json::to_string_pretty(&sorted).map_err(|e| VerdictError::Parse(e.to_string()))?;
    std::fs::write(path, text).map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))
}

fn check_expected(actual: Verdict, expected: Option<&str>) -> Result<(), VerdictError> {
    match expected {
        None => Ok(()),
        Some(text) => {
            let expected = text.parse::<Verdict>()?;
            if expected == actual {
                Ok(())
            } else {
                Err(VerdictError::Mismatch { expected, actual })
            }
        }
    }
}

fn run(cli: &Cli) -> Result<(), VerdictError> {
    let runs = load_input(&cli.input)?;
    let outcome = verdict(&runs)?;
    let report = gate_report(&runs)?;
    println!("Verdict: {outcome}");
    println!("{report}");
    if let Some(path) = &cli.write_bundle {
        write_bundle(&runs, path)?;
    }
    check_expected(outcome, cli.expect_verdict.as_deref())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> AgentFoodKindSummary {
        AgentFoodKindSummary {
            turn_toward_nourishing_whole: Some(0.0),
            turn_toward_nourishing_late: Some(0.0),
            turn_away_harmful_whole: Some(0.0),
            turn_away_harmful_late: Some(0.0),
            nourishing_meals_by_quarter: [2; 4],
            harmful_meals_by_quarter: [2; 4],
            deaths: 0,
            readout_balanced_accuracy: Some(0.9),
            readout_shuffled_accuracy: Some(0.5),
            signed_credit_pair_first: 1.0,
            signed_credit_pair_second: -1.0,
            signed_weight_difference_pair_first: 1.0,
            signed_weight_difference_pair_second: -1.0,
        }
    }

    fn make_run(seed: u64, arm: Arm) -> RunFoodKindSummary {
        let mut base = agent();
        if arm == Arm::Learning {
            base.turn_toward_nourishing_late = Some(0.2);
            base.turn_away_harmful_late = Some(0.2);
            base.nourishing_meals_by_quarter = [1, 1, 1, 3];
            base.harmful_meals_by_quarter = [3, 1, 1, 1];
        }
        RunFoodKindSummary {
            seed,
            arm,
            ticks: REQUIRED_TICKS,
            agent_count: REQUIRED_AGENTS,
            fresh_brains: true,
            nourishing_kind: FoodKind::First,
            agents: vec![base; REQUIRED_AGENTS],
        }
    }

    fn fixture() -> Vec<RunFoodKindSummary> {
        SEEDS
            .flat_map(|s| [make_run(s, Arm::Learning), make_run(s, Arm::Frozen)])
            .collect()
    }

    fn learning_agents_mut(
        runs: &mut [RunFoodKindSummary],
    ) -> impl Iterator<Item = &mut AgentFoodKindSummary> {
        runs.iter_mut()
            .filter(|r| r.arm == Arm::Learning)
            .flat_map(|r| r.agents.iter_mut())
    }

    #[test]
    fn missing_seed_is_error() {
        let mut runs = fixture();
        runs.pop();
        assert!(validate_runs(&runs).is_err());
        assert!(verdict(&runs).is_err());
    }

    #[test]
    fn duplicate_run_is_error() {
        let mut runs = fixture();
        runs[7] = runs[6].clone();
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn wrong_ticks_is_error() {
        let mut runs = fixture();
        runs[0].ticks = 1;
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn wrong_agent_count_is_error() {
        let mut runs = fixture();
        runs[0].agents.pop();
        assert!(validate_runs(&runs).is_err());
        let mut runs = fixture();
        runs[0].agent_count = 9;
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn non_fresh_brains_is_error() {
        let mut runs = fixture();
        runs[0].fresh_brains = false;
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn mismatched_nourishing_kind_is_error() {
        let mut runs = fixture();
        runs[0].nourishing_kind = FoodKind::Second;
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn nan_is_error() {
        let mut runs = fixture();
        runs[0].agents[0].signed_credit_pair_first = f32::NAN;
        assert!(validate_runs(&runs).is_err());
        let mut runs = fixture();
        runs[0].agents[0].turn_toward_nourishing_late = None;
        assert!(validate_runs(&runs).is_err());
    }

    #[test]
    fn directory_and_bundle_give_same_verdict() {
        let dir = tempfile::tempdir().unwrap();
        let runs = fixture();
        for run in &runs {
            let path = dir.path().join(file_name(run.arm, run.seed));
            std::fs::write(path, serde_json::to_string(run).unwrap()).unwrap();
        }
        let bundle_dir = tempfile::tempdir().unwrap();
        let bundle_path = bundle_dir.path().join("bundle.json");
        write_bundle(&runs, &bundle_path).unwrap();
        let from_dir = verdict(&load_input(dir.path()).unwrap()).unwrap();
        let from_bundle = verdict(&load_input(&bundle_path).unwrap()).unwrap();
        assert_eq!(from_dir, from_bundle);
        std::fs::write(dir.path().join("extra.json"), "{}").unwrap();
        assert!(load_input(dir.path()).is_err());
    }

    #[test]
    fn pass_fixture_is_pass() {
        assert_eq!(verdict(&fixture()).unwrap(), Verdict::Pass);
    }

    #[test]
    fn readout_failure_is_named() {
        let mut runs = fixture();
        for agent in learning_agents_mut(&mut runs) {
            agent.readout_balanced_accuracy = None;
        }
        assert_eq!(verdict(&runs).unwrap(), Verdict::FailReadout);
    }

    #[test]
    fn credit_failure_alone_is_named() {
        let mut runs = fixture();
        for agent in learning_agents_mut(&mut runs) {
            agent.signed_credit_pair_first = -1.0;
        }
        assert_eq!(verdict(&runs).unwrap(), Verdict::FailCredit);
    }

    #[test]
    fn weight_change_failure_alone_is_named() {
        let mut runs = fixture();
        for agent in learning_agents_mut(&mut runs) {
            agent.signed_weight_difference_pair_second = 1.0;
        }
        assert_eq!(verdict(&runs).unwrap(), Verdict::FailWeightChange);
    }

    #[test]
    fn readout_and_credit_failure_names_readout() {
        let mut runs = fixture();
        for agent in learning_agents_mut(&mut runs) {
            agent.readout_shuffled_accuracy = Some(0.9);
            agent.signed_credit_pair_first = -1.0;
        }
        assert_eq!(verdict(&runs).unwrap(), Verdict::FailReadout);
    }

    #[test]
    fn behaviour_failure_with_links_passing_is_weight_change() {
        let mut runs = fixture();
        for agent in learning_agents_mut(&mut runs) {
            agent.turn_toward_nourishing_late = Some(0.0);
        }
        assert_eq!(verdict(&runs).unwrap(), Verdict::FailWeightChange);
    }

    #[test]
    fn write_bundle_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bundle.json");
        let mut runs = fixture();
        runs.reverse();
        write_bundle(&runs, &path).unwrap();
        let back = load_input(&path).unwrap();
        let mut expected = runs.clone();
        expected.sort_by_key(|r| (r.seed, r.arm == Arm::Frozen));
        assert_eq!(back, expected);
    }

    #[test]
    fn expect_verdict_mismatch_is_error() {
        assert!(check_expected(Verdict::Pass, Some("PASS")).is_ok());
        assert!(check_expected(Verdict::Pass, None).is_ok());
        assert!(check_expected(Verdict::Pass, Some("FAIL credit")).is_err());
        assert!(check_expected(Verdict::Pass, Some("bogus")).is_err());
    }

    #[test]
    fn verdict_text_round_trip() {
        for outcome in [
            Verdict::Pass,
            Verdict::FailReadout,
            Verdict::FailCredit,
            Verdict::FailWeightChange,
        ] {
            assert_eq!(outcome.to_string().parse::<Verdict>().unwrap(), outcome);
        }
    }

    #[test]
    fn verdict_text_rejects_other_strings() {
        for text in ["pass", "FAIL  readout", "", "FAIL", "PASS "] {
            assert!(text.parse::<Verdict>().is_err());
        }
    }
}
