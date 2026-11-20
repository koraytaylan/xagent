//! Fail-closed verdict over the supervisor evidence.
//!
//! Reads exactly 32 validated inputs (a directory of result files, or a
//! previously written bundle), applies the fixed decision rule and prints
//! `ADOPT`, `OPTION` or `REJECT`. Any malformed, missing or surplus input is an
//! error; no GPU is touched.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;
use serde::{Deserialize, Serialize};
use xagent_sandbox::headless::FreeLifeReport;
use xagent_sandbox::run_summary::{KindWindowTotals, RunSummary};
use xagent_shared::{EvolutionMode, Inheritance};

/// First seed of the evidence runs.
const SEED_FIRST: u64 = 5;
/// Last seed of the evidence runs.
const SEED_LAST: u64 = 8;
/// Agents in every run and free life.
const AGENT_COUNT: usize = 10;
/// Agent-ticks every run summary must cover.
const AGENT_TICKS_PER_SUMMARY: u64 = 32_000_000;
/// Ticks every free life must cover.
const FREE_LIFE_TICKS: u64 = 300_000;
/// Seeds that must favour the supervisor for an adoption.
const SEED_MAJORITY: usize = 3;
/// Bundle `format` field.
const BUNDLE_FORMAT: &str = "xagent-supervisor-verdict-bundle";
/// Bundle `version` field.
const BUNDLE_VERSION: u32 = 1;
/// Exit code of an input error.
const EXIT_INPUT_ERROR: i32 = 1;
/// Exit code of an expectation mismatch.
const EXIT_EXPECT_MISMATCH: i32 = 2;

/// Outcome of the decision rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum Verdict {
    /// Adopt the supervisor.
    Adopt,
    /// Keep the supervisor as an option.
    Option,
    /// Reject the supervisor.
    Reject,
}

impl fmt::Display for Verdict {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Adopt => "ADOPT",
            Self::Option => "OPTION",
            Self::Reject => "REJECT",
        })
    }
}

impl FromStr for Verdict {
    type Err = VerdictError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "ADOPT" => Ok(Self::Adopt),
            "OPTION" => Ok(Self::Option),
            "REJECT" => Ok(Self::Reject),
            other => Err(VerdictError::BadVerdictName(other.to_string())),
        }
    }
}

/// Why the verdict could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VerdictError {
    /// A file could not be read or written.
    Io(String),
    /// A file is not valid JSON of the expected type.
    Parse(String),
    /// An expected input is absent.
    MissingFile(String),
    /// An input is not one of the expected names.
    ExtraFile(String),
    /// An input name appears twice.
    DuplicateFile(String),
    /// An input has a wrong or non-finite value.
    Invalid { file: String, reason: String },
    /// A metric has a zero denominator.
    UndefinedMetric(&'static str),
    /// The bundle is malformed or disagrees with its recomputation.
    Bundle(String),
    /// Unknown verdict name.
    BadVerdictName(String),
}

impl fmt::Display for VerdictError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "i/o error: {message}"),
            Self::Parse(message) => write!(formatter, "parse error: {message}"),
            Self::MissingFile(name) => write!(formatter, "missing input: {name}"),
            Self::ExtraFile(name) => write!(formatter, "unexpected input: {name}"),
            Self::DuplicateFile(name) => write!(formatter, "duplicate input: {name}"),
            Self::Invalid { file, reason } => {
                write!(formatter, "invalid input {file}: {reason}")
            }
            Self::UndefinedMetric(name) => {
                write!(formatter, "metric {name} has a zero denominator")
            }
            Self::Bundle(message) => write!(formatter, "bundle error: {message}"),
            Self::BadVerdictName(name) => {
                write!(
                    formatter,
                    "unknown verdict {name:?}; use ADOPT, OPTION or REJECT"
                )
            }
        }
    }
}

impl std::error::Error for VerdictError {}

/// Command line.
#[derive(Parser, Debug)]
#[command(about = "Apply the fixed supervisor decision rule to 32 validated inputs")]
struct Cli {
    /// Directory holding the 32 result files, or a bundle file.
    input: PathBuf,
    /// Write the inputs, metrics and verdict as a bundle file.
    #[arg(long, value_name = "FILE")]
    write_bundle: Option<PathBuf>,
    /// Exit with code 2 unless the computed verdict equals this one.
    #[arg(long, value_name = "ADOPT|OPTION|REJECT")]
    expect_verdict: Option<Verdict>,
}

/// Run family of an input file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Generations,
    Supervisor,
    WeightCopy,
}

impl Family {
    const ALL: [Self; ARM_COUNT] = [Self::Generations, Self::Supervisor, Self::WeightCopy];

    fn label(self) -> &'static str {
        match self {
            Self::Generations => "generations",
            Self::Supervisor => "supervisor",
            Self::WeightCopy => "weight-copy",
        }
    }

    fn mode(self) -> EvolutionMode {
        match self {
            Self::Generations => EvolutionMode::Generations,
            Self::Supervisor | Self::WeightCopy => EvolutionMode::Supervisor,
        }
    }

    fn inheritance(self) -> Option<Inheritance> {
        match self {
            Self::Generations => None,
            Self::Supervisor => Some(Inheritance::Darwinian),
            Self::WeightCopy => Some(Inheritance::WeightCopy),
        }
    }

    fn brain_source(self) -> &'static str {
        match self {
            Self::Generations => "champion",
            Self::Supervisor | Self::WeightCopy => "archive",
        }
    }

    fn has_kinds_runs(self) -> bool {
        self != Self::WeightCopy
    }
}

/// What kind of file a stem names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Run { kinds: bool },
    FreeLife,
}

/// One expected input.
#[derive(Clone, Debug)]
struct Spec {
    stem: String,
    family: Family,
    role: Role,
    seed: u64,
}

fn run_stem(family: Family, kinds: bool, seed: u64) -> String {
    let infix = if kinds { "-kinds" } else { "" };
    format!("{}{infix}-seed{seed}", family.label())
}

fn free_life_stem(family: Family, seed: u64) -> String {
    format!("free-life-{}-seed{seed}", family.label())
}

/// The exact set of expected inputs.
fn specs() -> Vec<Spec> {
    let mut out = Vec::new();
    for family in Family::ALL {
        for seed in SEED_FIRST..=SEED_LAST {
            out.push(Spec {
                stem: run_stem(family, false, seed),
                family,
                role: Role::Run { kinds: false },
                seed,
            });
            if family.has_kinds_runs() {
                out.push(Spec {
                    stem: run_stem(family, true, seed),
                    family,
                    role: Role::Run { kinds: true },
                    seed,
                });
            }
            out.push(Spec {
                stem: free_life_stem(family, seed),
                family,
                role: Role::FreeLife,
                seed,
            });
        }
    }
    out
}

/// All validated inputs keyed by file stem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Inputs {
    runs: BTreeMap<String, RunSummary>,
    free_lives: BTreeMap<String, FreeLifeReport>,
}

fn invalid(file: &str, reason: impl Into<String>) -> VerdictError {
    VerdictError::Invalid {
        file: file.to_string(),
        reason: reason.into(),
    }
}

fn check_finite(file: &str, name: &str, value: f64) -> Result<(), VerdictError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid(file, format!("{name} is not finite")))
    }
}

fn validate_run(spec: &Spec, kinds: bool, run: &RunSummary) -> Result<(), VerdictError> {
    let file = spec.stem.as_str();
    if run.mode != spec.family.mode() {
        return Err(invalid(file, "wrong evolution mode"));
    }
    if run.inheritance != spec.family.inheritance() {
        return Err(invalid(file, "wrong inheritance"));
    }
    if run.seed != spec.seed {
        return Err(invalid(file, "seed does not match the file name"));
    }
    if run.agent_count != AGENT_COUNT {
        return Err(invalid(file, "wrong agent count"));
    }
    if run.agent_ticks != AGENT_TICKS_PER_SUMMARY {
        return Err(invalid(file, "wrong agent-tick count"));
    }
    if run.food_kinds_enabled != kinds {
        return Err(invalid(file, "wrong food-kinds setting"));
    }
    if run.kinds.is_some() != kinds {
        return Err(invalid(file, "kind totals present only with food kinds"));
    }
    if let Some(kind_totals) = &run.kinds {
        check_finite(
            file,
            "late_toward_nourishing_sum",
            kind_totals.late_toward_nourishing_sum,
        )?;
        check_finite(
            file,
            "late_away_harmful_sum",
            kind_totals.late_away_harmful_sum,
        )?;
    }
    Ok(())
}

fn validate_free_life(spec: &Spec, report: &FreeLifeReport) -> Result<(), VerdictError> {
    let file = spec.stem.as_str();
    if report.seed != spec.seed {
        return Err(invalid(file, "seed does not match the file name"));
    }
    if report.agent_count != AGENT_COUNT || report.agents.len() != AGENT_COUNT {
        return Err(invalid(file, "wrong agent count"));
    }
    if report.ticks != FREE_LIFE_TICKS {
        return Err(invalid(file, "wrong tick count"));
    }
    if report.brain_source != spec.family.brain_source() {
        return Err(invalid(file, "wrong brain source"));
    }
    for agent in &report.agents {
        for window in [&agent.whole, &agent.late] {
            check_finite(file, "sighting_meal_share", window.sighting_meal_share)?;
            check_finite(file, "turn_toward_food", window.turn_toward_food)?;
            check_finite(file, "turn_away_from_hazard", window.turn_away_from_hazard)?;
            check_finite(file, "value_gap_in_view", window.value_gap_in_view)?;
            check_finite(file, "value_gap_centred", window.value_gap_centred)?;
            let correlation = &window.pathway_food_correlation;
            check_finite(
                file,
                "pathway_food_correlation.encoding",
                correlation.encoding,
            )?;
            check_finite(file, "pathway_food_correlation.scent", correlation.scent)?;
            check_finite(file, "pathway_food_correlation.vision", correlation.vision)?;
            for value in window.turn_credit.iter().chain(&window.forward_credit) {
                check_finite(file, "pathway credit", *value)?;
            }
        }
    }
    Ok(())
}

/// Checks that the maps hold exactly the expected inputs and that each is valid.
fn validate_inputs(inputs: &Inputs) -> Result<(), VerdictError> {
    let specs = specs();
    for key in inputs.runs.keys().chain(inputs.free_lives.keys()) {
        if !specs.iter().any(|s| &s.stem == key) {
            return Err(VerdictError::ExtraFile(key.clone()));
        }
    }
    for spec in &specs {
        match spec.role {
            Role::Run { kinds } => {
                let run = inputs
                    .runs
                    .get(&spec.stem)
                    .ok_or_else(|| VerdictError::MissingFile(spec.stem.clone()))?;
                validate_run(spec, kinds, run)?;
            }
            Role::FreeLife => {
                let report = inputs
                    .free_lives
                    .get(&spec.stem)
                    .ok_or_else(|| VerdictError::MissingFile(spec.stem.clone()))?;
                validate_free_life(spec, report)?;
            }
        }
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, VerdictError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| VerdictError::Parse(format!("{}: {e}", path.display())))
}

/// Loads and validates the 32 files of a directory.
fn load_directory(dir: &Path) -> Result<Inputs, VerdictError> {
    let specs = specs();
    let mut inputs = Inputs {
        runs: BTreeMap::new(),
        free_lives: BTreeMap::new(),
    };
    let entries =
        std::fs::read_dir(dir).map_err(|e| VerdictError::Io(format!("{}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry.map_err(|e| VerdictError::Io(e.to_string()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let spec = name
            .strip_suffix(".json")
            .and_then(|stem| specs.iter().find(|s| s.stem == stem))
            .ok_or_else(|| VerdictError::ExtraFile(name.clone()))?;
        let duplicate = match spec.role {
            Role::Run { .. } => inputs
                .runs
                .insert(spec.stem.clone(), read_json(&entry.path())?)
                .is_some(),
            Role::FreeLife => inputs
                .free_lives
                .insert(spec.stem.clone(), read_json(&entry.path())?)
                .is_some(),
        };
        if duplicate {
            return Err(VerdictError::DuplicateFile(name));
        }
    }
    validate_inputs(&inputs)?;
    Ok(inputs)
}

/// Number of arms shown in the metrics table.
const ARM_COUNT: usize = 3;
/// Number of learning metrics per family (L1 to L3 and K1 to K3).
const METRIC_COUNT: usize = 3;
/// Decimal places printed for each metric value.
const VALUE_PRECISION: usize = 6;
/// Column width of the metric name in the table.
const NAME_WIDTH: usize = 10;
/// Column width of each arm value in the table.
const VALUE_WIDTH: usize = 14;

/// Metrics of one arm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Metrics {
    /// Nourishing meals per agent-tick, per seed.
    food_per_seed: Vec<f64>,
    /// Deaths per agent-tick, per seed.
    deaths_per_seed: Vec<f64>,
    /// Pooled nourishing meals per agent-tick.
    food_pooled: f64,
    /// Pooled deaths per agent-tick.
    deaths_pooled: f64,
    /// Free-life learning metrics L1 to L3.
    l: [f64; METRIC_COUNT],
    /// Food-kind learning metrics K1 to K3; absent for the weight-copy arm.
    k: Option<[f64; METRIC_COUNT]>,
}

/// Metrics of the three arms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct AllMetrics {
    supervisor: Metrics,
    generations: Metrics,
    weight_copy: Metrics,
}

/// Converts a count to a float, failing when it exceeds the exactly representable range.
fn count_to_f64(name: &str, count: u64) -> Result<f64, VerdictError> {
    u32::try_from(count)
        .map(f64::from)
        .map_err(|_| invalid(name, "count exceeds the exactly representable range"))
}

fn ratio(name: &'static str, numerator: f64, denominator: f64) -> Result<f64, VerdictError> {
    if denominator == 0.0 {
        Err(VerdictError::UndefinedMetric(name))
    } else {
        Ok(numerator / denominator)
    }
}

fn run_of(
    inputs: &Inputs,
    family: Family,
    kinds: bool,
    seed: u64,
) -> Result<&RunSummary, VerdictError> {
    let stem = run_stem(family, kinds, seed);
    inputs
        .runs
        .get(&stem)
        .ok_or(VerdictError::MissingFile(stem))
}

fn kind_totals(inputs: &Inputs, family: Family) -> Result<KindWindowTotals, VerdictError> {
    let mut total = KindWindowTotals {
        late_toward_nourishing_sum: 0.0,
        late_toward_nourishing_samples: 0,
        late_away_harmful_sum: 0.0,
        late_away_harmful_samples: 0,
        late_nourishing_meals: 0,
        late_harmful_meals: 0,
        early_nourishing_meals: 0,
        early_harmful_meals: 0,
    };
    for seed in SEED_FIRST..=SEED_LAST {
        let run = run_of(inputs, family, true, seed)?;
        let seed_totals = run
            .kinds
            .as_ref()
            .ok_or_else(|| invalid(&run_stem(family, true, seed), "kind totals missing"))?;
        total.late_toward_nourishing_sum += seed_totals.late_toward_nourishing_sum;
        total.late_toward_nourishing_samples += seed_totals.late_toward_nourishing_samples;
        total.late_away_harmful_sum += seed_totals.late_away_harmful_sum;
        total.late_away_harmful_samples += seed_totals.late_away_harmful_samples;
        total.late_nourishing_meals += seed_totals.late_nourishing_meals;
        total.late_harmful_meals += seed_totals.late_harmful_meals;
    }
    Ok(total)
}

fn arm_metrics(inputs: &Inputs, family: Family) -> Result<Metrics, VerdictError> {
    let mut food_per_seed = Vec::new();
    let mut deaths_per_seed = Vec::new();
    let (mut meals, mut deaths, mut ticks) = (0.0_f64, 0.0_f64, 0.0_f64);
    for seed in SEED_FIRST..=SEED_LAST {
        let run = run_of(inputs, family, false, seed)?;
        let (seed_meals, seed_deaths, seed_ticks) = (
            count_to_f64("nourishing_meals", run.nourishing_meals)?,
            count_to_f64("deaths", run.deaths)?,
            count_to_f64("agent_ticks", run.agent_ticks)?,
        );
        food_per_seed.push(ratio("F", seed_meals, seed_ticks)?);
        deaths_per_seed.push(ratio("D", seed_deaths, seed_ticks)?);
        meals += seed_meals;
        deaths += seed_deaths;
        ticks += seed_ticks;
    }

    let (mut late_meals, mut alive, mut turn, mut share, mut agents) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for seed in SEED_FIRST..=SEED_LAST {
        let stem = free_life_stem(family, seed);
        let report = inputs
            .free_lives
            .get(&stem)
            .ok_or(VerdictError::MissingFile(stem))?;
        for agent in &report.agents {
            late_meals += count_to_f64("late.meals", agent.late.meals)?;
            alive += count_to_f64("late.alive_samples", agent.late.alive_samples)?;
            turn += agent.late.turn_toward_food;
            share += agent.late.sighting_meal_share;
            agents += 1.0;
        }
    }

    let kind_metrics = if family.has_kinds_runs() {
        let totals = kind_totals(inputs, family)?;
        let nourishing = count_to_f64("late_nourishing_meals", totals.late_nourishing_meals)?;
        let harmful = count_to_f64("late_harmful_meals", totals.late_harmful_meals)?;
        let toward_samples = count_to_f64(
            "late_toward_nourishing_samples",
            totals.late_toward_nourishing_samples,
        )?;
        let away_samples = count_to_f64(
            "late_away_harmful_samples",
            totals.late_away_harmful_samples,
        )?;
        Some([
            ratio("K1", totals.late_toward_nourishing_sum, toward_samples)?,
            ratio("K2", totals.late_away_harmful_sum, away_samples)?,
            ratio("K3", nourishing, nourishing + harmful)?,
        ])
    } else {
        None
    };

    Ok(Metrics {
        food_per_seed,
        deaths_per_seed,
        food_pooled: ratio("pooled F", meals, ticks)?,
        deaths_pooled: ratio("pooled D", deaths, ticks)?,
        l: [
            ratio("L1", late_meals, alive)?,
            ratio("L2", turn, agents)?,
            ratio("L3", share, agents)?,
        ],
        k: kind_metrics,
    })
}

fn compute_metrics(inputs: &Inputs) -> Result<AllMetrics, VerdictError> {
    Ok(AllMetrics {
        supervisor: arm_metrics(inputs, Family::Supervisor)?,
        generations: arm_metrics(inputs, Family::Generations)?,
        weight_copy: arm_metrics(inputs, Family::WeightCopy)?,
    })
}

/// Applies the decision rule to the supervisor and generations arms only.
fn decide(supervisor: &Metrics, generations: &Metrics) -> Verdict {
    let seeds_ok = supervisor
        .food_per_seed
        .iter()
        .zip(&generations.food_per_seed)
        .zip(
            supervisor
                .deaths_per_seed
                .iter()
                .zip(&generations.deaths_per_seed),
        )
        .filter(|((fs, fg), (ds, dg))| fs >= fg && ds <= dg)
        .count();
    let l_ok = supervisor.l.iter().zip(&generations.l).all(|(s, g)| s >= g);
    let k_ok = match (supervisor.k, generations.k) {
        (Some(supervisor_k), Some(generations_k)) => supervisor_k
            .iter()
            .zip(&generations_k)
            .all(|(own, other)| own >= other),
        _ => false,
    };
    let learning = l_ok && k_ok;
    let pooled = supervisor.food_pooled >= generations.food_pooled
        && supervisor.deaths_pooled <= generations.deaths_pooled;
    if seeds_ok >= SEED_MAJORITY && learning {
        Verdict::Adopt
    } else if pooled && learning {
        Verdict::Option
    } else {
        Verdict::Reject
    }
}

/// Bundle file contents.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Bundle {
    format: String,
    version: u32,
    runs: BTreeMap<String, RunSummary>,
    free_lives: BTreeMap<String, FreeLifeReport>,
    metrics: AllMetrics,
    verdict: Verdict,
}

/// Computes metrics and verdict from validated inputs.
fn evaluate(inputs: &Inputs) -> Result<(AllMetrics, Verdict), VerdictError> {
    let metrics = compute_metrics(inputs)?;
    let verdict = decide(&metrics.supervisor, &metrics.generations);
    Ok((metrics, verdict))
}

fn make_bundle(inputs: &Inputs, metrics: AllMetrics, verdict: Verdict) -> Bundle {
    Bundle {
        format: BUNDLE_FORMAT.to_string(),
        version: BUNDLE_VERSION,
        runs: inputs.runs.clone(),
        free_lives: inputs.free_lives.clone(),
        metrics,
        verdict,
    }
}

/// Loads a bundle, revalidates it and checks the stored results.
fn load_bundle(path: &Path) -> Result<(Inputs, AllMetrics, Verdict), VerdictError> {
    let bundle: Bundle = read_json(path)?;
    if bundle.format != BUNDLE_FORMAT || bundle.version != BUNDLE_VERSION {
        return Err(VerdictError::Bundle(
            "unknown format or version".to_string(),
        ));
    }
    let inputs = Inputs {
        runs: bundle.runs,
        free_lives: bundle.free_lives,
    };
    validate_inputs(&inputs)?;
    let (metrics, verdict) = evaluate(&inputs)?;
    // Stored numbers went through text; push the recomputation through the same
    // text so both sides parse identically.
    let text = serde_json::to_string(&metrics).map_err(|e| VerdictError::Parse(e.to_string()))?;
    let reparsed: AllMetrics =
        serde_json::from_str(&text).map_err(|e| VerdictError::Parse(e.to_string()))?;
    if reparsed != bundle.metrics {
        return Err(VerdictError::Bundle(
            "stored metrics differ from recomputation".to_string(),
        ));
    }
    if verdict != bundle.verdict {
        return Err(VerdictError::Bundle(
            "stored verdict differs from recomputation".to_string(),
        ));
    }
    Ok((inputs, metrics, verdict))
}

fn load(path: &Path) -> Result<(Inputs, AllMetrics, Verdict), VerdictError> {
    if path.is_dir() {
        let inputs = load_directory(path)?;
        let (metrics, verdict) = evaluate(&inputs)?;
        Ok((inputs, metrics, verdict))
    } else {
        load_bundle(path)
    }
}

fn render(verdict: Verdict, all_metrics: &AllMetrics) -> String {
    let cell =
        |v: Option<f64>| v.map_or_else(|| "n/a".to_string(), |x| format!("{x:.VALUE_PRECISION$}"));
    let mut rows: Vec<(String, [Option<f64>; ARM_COUNT])> = Vec::new();
    let arms = [
        &all_metrics.supervisor,
        &all_metrics.generations,
        &all_metrics.weight_copy,
    ];
    rows.push(("F pooled".into(), arms.map(|a| Some(a.food_pooled))));
    rows.push(("D pooled".into(), arms.map(|a| Some(a.deaths_pooled))));
    for i in 0..METRIC_COUNT {
        rows.push((format!("L{}", i + 1), arms.map(|a| Some(a.l[i]))));
    }
    for i in 0..METRIC_COUNT {
        rows.push((format!("K{}", i + 1), arms.map(|a| a.k.map(|k| k[i]))));
    }
    let mut out = format!("Verdict: {verdict}\n");
    let _ = writeln!(
        out,
        "{:<NAME_WIDTH$} {:>VALUE_WIDTH$} {:>VALUE_WIDTH$} {:>VALUE_WIDTH$}",
        "metric", "supervisor", "generations", "weight-copy"
    );
    for (name, [supervisor, generations, weight_copy]) in rows {
        let _ = writeln!(
            out,
            "{:<NAME_WIDTH$} {:>VALUE_WIDTH$} {:>VALUE_WIDTH$} {:>VALUE_WIDTH$}",
            name,
            cell(supervisor),
            cell(generations),
            cell(weight_copy)
        );
    }
    out
}

/// Exit code for the computed verdict against an optional expectation.
fn expect_exit_code(computed: Verdict, expected: Option<Verdict>) -> i32 {
    match expected {
        Some(want) if want != computed => {
            eprintln!("expected verdict {want}, computed {computed}");
            EXIT_EXPECT_MISMATCH
        }
        _ => 0,
    }
}

/// Exit code for a command-line parse outcome: help and version succeed, usage errors are input errors.
fn usage_exit_code(error: &clap::Error) -> i32 {
    if error.use_stderr() {
        EXIT_INPUT_ERROR
    } else {
        0
    }
}

fn run(cli: &Cli) -> Result<i32, VerdictError> {
    let (inputs, metrics, verdict) = load(&cli.input)?;
    print!("{}", render(verdict, &metrics));
    if let Some(path) = &cli.write_bundle {
        let bundle = make_bundle(&inputs, metrics, verdict);
        let text = serde_json::to_string_pretty(&bundle)
            .map_err(|e| VerdictError::Parse(e.to_string()))?;
        std::fs::write(path, text)
            .map_err(|e| VerdictError::Io(format!("{}: {e}", path.display())))?;
    }
    Ok(expect_exit_code(verdict, cli.expect_verdict))
}

fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = usage_exit_code(&error);
            // Printing can only fail on a closed stream; the exit code still reports the outcome.
            let _ = error.print();
            std::process::exit(code);
        }
    };
    match run(&cli) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(EXIT_INPUT_ERROR);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_sandbox::free_life::metrics::{AgentMetrics, WindowMetrics};

    /// Meals of a fixture run that beats the base fixture.
    const BETTER_MEALS: u64 = 1100;
    /// Deaths of a fixture run that beats the base fixture.
    const FEWER_DEATHS: u64 = 9;
    /// Deaths of a fixture run that is worse than the base fixture.
    const WORSE_DEATHS: u64 = 11;
    /// Nourishing meals above the base fixture.
    const HIGH_NOURISHING_MEALS: u64 = 1300;
    /// Nourishing meals below the base fixture.
    const REDUCED_NOURISHING_MEALS: u64 = 900;
    /// Turn-toward-food value worse than the base fixture.
    const WORSE_TURN: f64 = 0.05;
    /// Away-from-harmful value worse than the base fixture.
    const WORSE_AWAY: f64 = 5.0;
    /// Meals of an extremely good weight-copy fixture.
    const GREAT_MEALS: u64 = 5000;
    /// Turn-toward-food value of an extremely good weight-copy fixture.
    const GREAT_TURN: f64 = 0.9;
    /// Meals of an extremely bad weight-copy fixture.
    const AWFUL_MEALS: u64 = 1;
    /// Deaths of an extremely bad weight-copy fixture.
    const AWFUL_DEATHS: u64 = 900;
    /// Seeds given the high nourishing-meal fixture.
    const HIGH_SEEDS: [u64; 2] = [5, 6];
    /// Seeds given the reduced nourishing-meal fixture.
    const LOW_SEEDS: [u64; 2] = [7, 8];
    /// Number of input files in a complete input set.
    const INPUT_FILE_COUNT: usize = 32;

    /// Meals of the base fixture run.
    const BASE_MEALS: u64 = 1000;
    /// Deaths of the base fixture run.
    const BASE_DEATHS: u64 = 10;
    /// Late-window meals of the base fixture free life.
    const BASE_LATE_MEALS: u64 = 100;
    /// Turn-toward-food value of the base fixture.
    const BASE_TURN: f64 = 0.1;
    /// Sighting meal share of the base fixture.
    const BASE_SHARE: f64 = 0.1;
    /// Toward-nourishing sum of the base fixture.
    const BASE_TOWARD: f64 = 10.0;
    /// Away-from-harmful sum of the base fixture.
    const BASE_AWAY: f64 = 10.0;
    /// Late nourishing meals of the base fixture.
    const BASE_NOURISHING_LATE: u64 = 50;
    /// Sample count of a fixture window.
    const WINDOW_SAMPLES: u64 = 1000;
    /// Sample count of the fixture kind totals.
    const KIND_SAMPLES: u64 = 100;
    /// Late harmful meals of the fixture kind totals.
    const KIND_HARMFUL_MEALS: u64 = 50;

    #[derive(Clone)]
    struct Params {
        meals: u64,
        deaths: u64,
        late_meals: u64,
        turn: f64,
        share: f64,
        toward: f64,
        away: f64,
        nourishing_late: u64,
    }

    fn base() -> Params {
        Params {
            meals: BASE_MEALS,
            deaths: BASE_DEATHS,
            late_meals: BASE_LATE_MEALS,
            turn: BASE_TURN,
            share: BASE_SHARE,
            toward: BASE_TOWARD,
            away: BASE_AWAY,
            nourishing_late: BASE_NOURISHING_LATE,
        }
    }

    fn window(params: &Params) -> WindowMetrics {
        WindowMetrics {
            samples: WINDOW_SAMPLES,
            alive_samples: WINDOW_SAMPLES,
            meals: params.late_meals,
            deaths: 0,
            hazard_deaths: 0,
            sightings: 0,
            sightings_ending_in_meal: 0,
            sighting_meal_share: params.share,
            turn_toward_food: params.turn,
            turn_away_from_hazard: 0.0,
            pathway_food_correlation: Default::default(),
            turn_credit: Default::default(),
            forward_credit: Default::default(),
            value_gap_in_view: 0.0,
            value_gap_centred: 0.0,
        }
    }

    fn run_summary(family: Family, kinds: bool, seed: u64, params: &Params) -> RunSummary {
        RunSummary {
            mode: family.mode(),
            inheritance: family.inheritance(),
            seed,
            agent_count: AGENT_COUNT,
            ticks: AGENT_TICKS_PER_SUMMARY / AGENT_COUNT as u64,
            agent_ticks: AGENT_TICKS_PER_SUMMARY,
            food_kinds_enabled: kinds,
            nourishing_meals: params.meals,
            harmful_meals: 0,
            deaths: params.deaths,
            seasons: 1,
            kinds: kinds.then_some(KindWindowTotals {
                late_toward_nourishing_sum: params.toward,
                late_toward_nourishing_samples: KIND_SAMPLES,
                late_away_harmful_sum: params.away,
                late_away_harmful_samples: KIND_SAMPLES,
                late_nourishing_meals: params.nourishing_late,
                late_harmful_meals: KIND_HARMFUL_MEALS,
                early_nourishing_meals: 0,
                early_harmful_meals: 0,
            }),
        }
    }

    fn free_life(family: Family, seed: u64, params: &Params) -> FreeLifeReport {
        let agent = AgentMetrics {
            whole: window(params),
            late: window(params),
        };
        FreeLifeReport {
            agent_count: AGENT_COUNT,
            ticks: FREE_LIFE_TICKS,
            seed,
            brain_source: family.brain_source().to_string(),
            champion_node_id: None,
            champion_generation: None,
            archive_entry_id: None,
            lesions: Vec::new(),
            kernel_batch_size: 1,
            brain_ticks: 0,
            agents: vec![agent; AGENT_COUNT],
        }
    }

    fn inputs(generations: &Params, supervisor: &Params, weight_copy: &Params) -> Inputs {
        let mut out = Inputs {
            runs: BTreeMap::new(),
            free_lives: BTreeMap::new(),
        };
        for spec in specs() {
            let params = match spec.family {
                Family::Generations => generations,
                Family::Supervisor => supervisor,
                Family::WeightCopy => weight_copy,
            };
            match spec.role {
                Role::Run { kinds } => {
                    out.runs.insert(
                        spec.stem,
                        run_summary(spec.family, kinds, spec.seed, params),
                    );
                }
                Role::FreeLife => {
                    out.free_lives
                        .insert(spec.stem, free_life(spec.family, spec.seed, params));
                }
            }
        }
        out
    }

    fn equal_inputs() -> Inputs {
        inputs(&base(), &base(), &base())
    }

    fn write_dir(inputs: &Inputs, dir: &Path) {
        for (stem, run) in &inputs.runs {
            std::fs::write(
                dir.join(format!("{stem}.json")),
                serde_json::to_string(run).unwrap(),
            )
            .unwrap();
        }
        for (stem, report) in &inputs.free_lives {
            std::fs::write(
                dir.join(format!("{stem}.json")),
                serde_json::to_string(report).unwrap(),
            )
            .unwrap();
        }
    }

    fn verdict_of(inputs: &Inputs) -> Verdict {
        validate_inputs(inputs).unwrap();
        evaluate(inputs).unwrap().1
    }

    #[test]
    fn verdict_rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        write_dir(&equal_inputs(), dir.path());
        std::fs::remove_file(dir.path().join("supervisor-seed5.json")).unwrap();
        assert_eq!(
            load_directory(dir.path()).unwrap_err(),
            VerdictError::MissingFile("supervisor-seed5".to_string())
        );
    }

    #[test]
    fn verdict_rejects_extra_file() {
        let dir = tempfile::tempdir().unwrap();
        write_dir(&equal_inputs(), dir.path());
        std::fs::write(dir.path().join("notes.json"), "{}").unwrap();
        assert!(matches!(
            load_directory(dir.path()),
            Err(VerdictError::ExtraFile(_))
        ));
    }

    #[test]
    fn verdict_directory_loads_when_complete() {
        let dir = tempfile::tempdir().unwrap();
        write_dir(&equal_inputs(), dir.path());
        assert_eq!(load_directory(dir.path()).unwrap(), equal_inputs());
    }

    #[test]
    fn verdict_rejects_wrong_agent_ticks() {
        let mut tampered = equal_inputs();
        tampered
            .runs
            .get_mut("generations-seed5")
            .unwrap()
            .agent_ticks = 1;
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
    }

    #[test]
    fn verdict_rejects_wrong_mode_or_inheritance() {
        let mut tampered = equal_inputs();
        tampered
            .runs
            .get_mut("supervisor-seed5")
            .unwrap()
            .inheritance = Some(Inheritance::WeightCopy);
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
        let mut tampered = equal_inputs();
        tampered.runs.get_mut("generations-seed6").unwrap().mode = EvolutionMode::Supervisor;
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
    }

    #[test]
    fn verdict_rejects_wrong_brain_source() {
        let mut tampered = equal_inputs();
        tampered
            .free_lives
            .get_mut("free-life-supervisor-seed7")
            .unwrap()
            .brain_source = "champion".to_string();
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
    }

    #[test]
    fn verdict_rejects_non_finite_value() {
        let mut tampered = equal_inputs();
        tampered
            .free_lives
            .get_mut("free-life-generations-seed5")
            .unwrap()
            .agents[0]
            .late
            .turn_toward_food = f64::NAN;
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
        let mut tampered = equal_inputs();
        tampered
            .runs
            .get_mut("supervisor-kinds-seed5")
            .unwrap()
            .kinds
            .as_mut()
            .unwrap()
            .late_away_harmful_sum = f64::INFINITY;
        assert!(matches!(
            validate_inputs(&tampered),
            Err(VerdictError::Invalid { .. })
        ));
    }

    #[test]
    fn verdict_rejects_undefined_metric() {
        let mut tampered = equal_inputs();
        for seed in SEED_FIRST..=SEED_LAST {
            tampered
                .runs
                .get_mut(&run_stem(Family::Supervisor, true, seed))
                .unwrap()
                .kinds
                .as_mut()
                .unwrap()
                .late_toward_nourishing_samples = 0;
        }
        validate_inputs(&tampered).unwrap();
        assert_eq!(
            evaluate(&tampered).unwrap_err(),
            VerdictError::UndefinedMetric("K1")
        );
    }

    #[test]
    fn verdict_adopt_when_three_seeds_and_learning_hold() {
        let better = Params {
            meals: BETTER_MEALS,
            deaths: FEWER_DEATHS,
            ..base()
        };
        assert_eq!(
            verdict_of(&inputs(&base(), &better, &base())),
            Verdict::Adopt
        );
    }

    #[test]
    fn verdict_option_when_two_seeds_but_nothing_worse() {
        let mut tampered = equal_inputs();
        for seed in HIGH_SEEDS {
            tampered
                .runs
                .get_mut(&run_stem(Family::Supervisor, false, seed))
                .unwrap()
                .nourishing_meals = HIGH_NOURISHING_MEALS;
        }
        for seed in LOW_SEEDS {
            tampered
                .runs
                .get_mut(&run_stem(Family::Supervisor, false, seed))
                .unwrap()
                .nourishing_meals = REDUCED_NOURISHING_MEALS;
        }
        assert_eq!(verdict_of(&tampered), Verdict::Option);
    }

    #[test]
    fn verdict_reject_when_a_learning_metric_is_worse() {
        let worse = Params {
            meals: BETTER_MEALS,
            turn: WORSE_TURN,
            ..base()
        };
        assert_eq!(
            verdict_of(&inputs(&base(), &worse, &base())),
            Verdict::Reject
        );
        let worse_k = Params {
            away: WORSE_AWAY,
            ..base()
        };
        assert_eq!(
            verdict_of(&inputs(&base(), &worse_k, &base())),
            Verdict::Reject
        );
    }

    #[test]
    fn verdict_reject_when_pooled_deaths_worse() {
        let worse = Params {
            meals: BETTER_MEALS,
            deaths: WORSE_DEATHS,
            ..base()
        };
        assert_eq!(
            verdict_of(&inputs(&base(), &worse, &base())),
            Verdict::Reject
        );
    }

    #[test]
    fn verdict_weight_copy_never_changes_verdict() {
        let better = Params {
            meals: BETTER_MEALS,
            ..base()
        };
        let great = Params {
            meals: GREAT_MEALS,
            deaths: 0,
            turn: GREAT_TURN,
            ..base()
        };
        let awful = Params {
            meals: AWFUL_MEALS,
            deaths: AWFUL_DEATHS,
            turn: 0.0,
            ..base()
        };
        let with_great = verdict_of(&inputs(&base(), &better, &great));
        let with_awful = verdict_of(&inputs(&base(), &better, &awful));
        assert_eq!(with_great, with_awful);
        assert_eq!(with_great, Verdict::Adopt);
    }

    #[test]
    fn verdict_bundle_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let source = inputs(
            &base(),
            &Params {
                meals: BETTER_MEALS,
                ..base()
            },
            &base(),
        );
        let (metrics, verdict) = evaluate(&source).unwrap();
        let path = dir.path().join("bundle.json");
        let bundle = make_bundle(&source, metrics.clone(), verdict);
        std::fs::write(&path, serde_json::to_string_pretty(&bundle).unwrap()).unwrap();
        let (loaded, _, loaded_verdict) = load(&path).unwrap();
        assert_eq!(
            loaded.runs.len() + loaded.free_lives.len(),
            INPUT_FILE_COUNT
        );
        assert_eq!(loaded_verdict, verdict);
    }

    #[test]
    fn verdict_bundle_with_tampered_verdict_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let source = equal_inputs();
        let (metrics, verdict) = evaluate(&source).unwrap();
        let tampered = if verdict == Verdict::Reject {
            Verdict::Adopt
        } else {
            Verdict::Reject
        };
        let path = dir.path().join("bundle.json");
        let bundle = make_bundle(&source, metrics, tampered);
        std::fs::write(&path, serde_json::to_string_pretty(&bundle).unwrap()).unwrap();
        assert!(matches!(load(&path), Err(VerdictError::Bundle(_))));
    }

    #[test]
    fn verdict_usage_error_exit_code() {
        let bad_value =
            Cli::try_parse_from(["supervisor_verdict", "dir", "--expect-verdict", "MAYBE"])
                .expect_err("invalid expectation must not parse");
        assert_eq!(usage_exit_code(&bad_value), EXIT_INPUT_ERROR);
        let help = Cli::try_parse_from(["supervisor_verdict", "--help"])
            .expect_err("help is reported as a parse outcome");
        assert_eq!(usage_exit_code(&help), 0);
    }

    #[test]
    fn verdict_expect_mismatch_exit_code() {
        assert_eq!(expect_exit_code(Verdict::Adopt, None), 0);
        assert_eq!(expect_exit_code(Verdict::Adopt, Some(Verdict::Adopt)), 0);
        assert_eq!(
            expect_exit_code(Verdict::Adopt, Some(Verdict::Reject)),
            EXIT_EXPECT_MISMATCH
        );
        assert_eq!("OPTION".parse::<Verdict>().unwrap(), Verdict::Option);
        assert!("adopt".parse::<Verdict>().is_err());
    }
}
