//! Route gate: decides whether a pathway route earns its default.
//!
//! The tool compares a route arm (4 seeds, 10 agents each, fresh birth brains)
//! against a control arm run with both routes off, and records the verdicts of
//! four gates:
//!
//! - G1 (forward route only): conditional credit `C = sum_k g_k * m_k` per
//!   agent, where `g_k` is the life mean of `td_error * e_k` (the forward
//!   pathway trace read before the tick's update) and `m_k` is the mean of
//!   input `k` on ticks with food in view and centred minus its life mean. It
//!   passes when `C > 0` in at least 21 of 40 agents and the pooled mean of `C`
//!   is positive. Unconditional credit `U = sum_k g_k` is recorded only.
//! - G2: pooled share of sightings ending in a meal is higher than control and
//!   higher in at least 3 of 4 seeds.
//! - G3: route minus control mean meals is at least minus the Welch standard
//!   error, and mean deaths is at most plus the Welch standard error.
//! - G4: in every seed the route's mean composite fitness over generations
//!   60..=78 is at least the control's. It is evaluated only when the earlier
//!   gates pass.
//!
//! Trace conventions: the trace is read before the tick's update and is zero at
//! the start of each life, it is zeroed after a tick on which the death counter
//! rose, `g_k` is the mean over every logged tick, and the trace increment is
//! `forward_noise * x_k / (1 + |x_sense|^2)` with the norm taken per sense
//! (smell inputs 0..2, vision inputs 2..10), decayed by `TD_DISCOUNT * TD_LAMBDA`.
//! The logged `td_error` column is used as is. The ten-value forward credit of
//! the free-life metrics is copied verbatim and never recomputed.
//!
//! Input formats come from the library: this binary calls the free-life
//! metrics report (`xagent_sandbox::headless::FreeLifeReport`), the tick log
//! reader (`xagent_sandbox::free_life::tick_log::read_tick_log`) and the metric
//! predicate constants through the public `xagent_sandbox` paths, and re-implements
//! nothing. The tick log header fields are `format`, `version`, `endianness`,
//! `columns` and `rows`; the columns read here are `agent`, `brain_tick`,
//! `alive`, `deaths_total`, `food_distance`, `food_bearing`, `forward_noise`,
//! `td_error` and `pathway_input_0` to `pathway_input_9`. A death is a row whose
//! `deaths_total` exceeds the previous row of the same agent.
//!
//! The tool never runs an external program.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use xagent_sandbox::free_life::metrics::{
    count_as_f64, AgentMetrics, WindowMetrics, CENTRED_BEARING, FOOD_IN_VIEW_DISTANCE,
    PATHWAY_INPUTS, VIEW_HALF_ANGLE,
};
use xagent_sandbox::free_life::tick_log::{read_tick_log, TickLog};
use xagent_sandbox::headless::FreeLifeReport;
use xagent_shared::{CriticInput, DEFAULT_CRITIC_INPUT, DEFAULT_FORWARD_PATHWAY_ENABLED};

/// TD discount of the kernel (`TD_DISCOUNT` in the shared shader header).
const TD_DISCOUNT: f64 = 0.97;
/// TD trace decay factor of the kernel (`TD_LAMBDA` in the shared shader header).
const TD_LAMBDA: f64 = 0.9;
/// Per-tick decay of the forward pathway trace, `TD_DISCOUNT * TD_LAMBDA`.
const TRACE_DECAY: f64 = TD_DISCOUNT * TD_LAMBDA;
/// Seeds of the gate arms, one free life each.
const SEEDS: [u64; 4] = [5, 6, 7, 8];
/// Agents per seed in each arm.
const AGENTS_PER_SEED: usize = 10;
/// Simulation ticks of every free life.
const TICKS_PER_SEED: u64 = 300_000;
/// Number of smell inputs among the pathway inputs; the rest are vision inputs.
const SMELL_INPUTS: usize = 2;
/// Logged `alive` values above this count as alive (the column holds 0 or 1).
const ALIVE_THRESHOLD: f64 = 0.5;
/// G1 needs at least this many agents with positive conditional credit.
const G1_MIN_POSITIVE_AGENTS: usize = 21;
/// G2 needs at least this many seeds with a higher sighting meal share.
const G2_MIN_HIGHER_SEEDS: usize = 3;
/// Slack for the G3 comparisons, so a value exactly on the bound passes.
const GATE_TOLERANCE: f64 = 1e-9;
/// Relative tolerance when `verify` compares recomputed numbers with recorded ones.
const RELATIVE_TOLERANCE: f64 = 1e-9;
/// First generation of the G4 window.
const FIRST_GENERATION: i64 = 60;
/// Last generation of the G4 window.
const LAST_GENERATION: i64 = 78;
/// Seconds a database read waits on a lock before failing.
const DB_BUSY_TIMEOUT_SECONDS: u64 = 5;
/// Brain source of a fresh birth brain in the metrics report.
const FRESH_BRAIN_SOURCE: &str = "fresh";

const PASS: &str = "pass";
const FAIL: &str = "fail";
const PENDING: &str = "pending";
const NOT_APPLICABLE: &str = "not_applicable";
const SETTING_FORWARD: &str = "forward_pathway";
const SETTING_ENCODING: &str = "encoding";
const SETTING_PATHWAY: &str = "pathway";
const SETTING_BOTH: &str = "encoding_and_pathway";
const GATE_NAMES: [&str; 4] = ["G1", "G2", "G3", "G4"];

/// Number of agents in one arm.
const TOTAL_AGENTS: usize = SEEDS.len() * AGENTS_PER_SEED;

/// Which route a record judges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Route {
    Critic,
    Forward,
}

/// Which arm an agent belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Control,
    Route,
}

/// Code defaults a record is checked against.
#[derive(Clone, Copy, Debug)]
struct CodeDefaults {
    critic_input: CriticInput,
    forward_pathway_enabled: bool,
}

/// Thresholds and constants the numbers of a record were derived with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    td_discount: f64,
    td_lambda: f64,
    trace_decay: f64,
    min_positive_agents: usize,
    min_higher_seeds: usize,
    first_generation: i64,
    last_generation: i64,
}

impl Config {
    fn current() -> Self {
        Self {
            td_discount: TD_DISCOUNT,
            td_lambda: TD_LAMBDA,
            trace_decay: TRACE_DECAY,
            min_positive_agents: G1_MIN_POSITIVE_AGENTS,
            min_higher_seeds: G2_MIN_HIGHER_SEEDS,
            first_generation: FIRST_GENERATION,
            last_generation: LAST_GENERATION,
        }
    }
}

/// One agent's numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentRecord {
    arm: Arm,
    seed: u64,
    agent: usize,
    meals: u64,
    deaths: u64,
    sightings: u64,
    sightings_ending_in_meal: u64,
    sighting_meal_share: f64,
    /// `g_k`, the life mean of `td_error * trace` per input (forward route arm only).
    life_mean_credit: Option<Vec<f64>>,
    /// `m_k`, the centred-tick mean of input `k` minus its life mean.
    centred_input_excess: Option<Vec<f64>>,
    /// Conditional credit `C`.
    conditional_credit: Option<f64>,
    /// Unconditional credit `U`, information only.
    unconditional_credit: Option<f64>,
    /// The metrics report's whole-life forward credit, copied verbatim.
    free_life_forward_credit: Option<Vec<f64>>,
}

/// Pooled numbers of one arm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmPooled {
    agents: usize,
    mean_meals: f64,
    mean_deaths: f64,
    meals_variance: f64,
    deaths_variance: f64,
    sightings: u64,
    sightings_ending_in_meal: u64,
    sighting_meal_share: f64,
    positive_credit_agents: Option<usize>,
    mean_conditional_credit: Option<f64>,
    mean_unconditional_credit: Option<f64>,
}

/// Pooled numbers of both arms and their differences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pooled {
    control: ArmPooled,
    route: ArmPooled,
    meals_difference: f64,
    meals_standard_error: f64,
    deaths_difference: f64,
    deaths_standard_error: f64,
}

/// Per-seed numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedRecord {
    seed: u64,
    control_sighting_meal_share: f64,
    route_sighting_meal_share: f64,
    share_higher: bool,
    control_mean_meals: f64,
    route_mean_meals: f64,
    control_mean_deaths: f64,
    route_mean_deaths: f64,
    control_generation_mean: Option<f64>,
    route_generation_mean: Option<f64>,
    g4: String,
}

/// The gate record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    route: Route,
    commit: String,
    setting_under_test: String,
    verdict: String,
    failed_gate: Option<String>,
    g1: String,
    g2: String,
    g3: String,
    g4: String,
    seeds: Vec<u64>,
    agents_per_seed: usize,
    ticks_per_seed: Option<u64>,
    config: Option<Config>,
    agents: Vec<AgentRecord>,
    per_seed: Vec<SeedRecord>,
    pooled: Option<Pooled>,
}

/// Fields of a record that the caller chooses.
#[derive(Clone, Debug)]
struct Header {
    route: Route,
    setting: String,
    commit: String,
}

/// Mean composite fitness of both arms of one seed over the G4 window.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SeedGenerations {
    seed: u64,
    control_mean: f64,
    route_mean: f64,
}

fn usize_f64(n: usize) -> f64 {
    count_as_f64(u64::try_from(n).unwrap_or(u64::MAX))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / usize_f64(values.len())
    }
}

fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let centre = mean(values);
    let squares: f64 = values.iter().map(|v| (v - centre) * (v - centre)).sum();
    squares / usize_f64(values.len() - 1)
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= RELATIVE_TOLERANCE * a.abs().max(b.abs())
}

// ---------------------------------------------------------------------------
// Credit rebuild
// ---------------------------------------------------------------------------

/// The per-tick inputs of the credit rebuild.
#[derive(Clone, Copy, Debug)]
struct CreditTick {
    inputs: [f64; PATHWAY_INPUTS],
    forward_noise: f64,
    td_error: f64,
    death: bool,
    food_centred: bool,
}

/// Credit numbers of one agent life.
#[derive(Clone, Debug)]
struct AgentCredit {
    life_mean_credit: [f64; PATHWAY_INPUTS],
    centred_input_excess: [f64; PATHWAY_INPUTS],
    conditional: f64,
    unconditional: f64,
}

/// Trace increment of one tick, normalised per sense.
fn trace_increment(tick: &CreditTick) -> [f64; PATHWAY_INPUTS] {
    let square_sum = |inputs: &[f64]| 1.0 + inputs.iter().map(|v| v * v).sum::<f64>();
    let smell_norm = square_sum(&tick.inputs[..SMELL_INPUTS]);
    let vision_norm = square_sum(&tick.inputs[SMELL_INPUTS..]);
    std::array::from_fn(|k| {
        let norm = if k < SMELL_INPUTS {
            smell_norm
        } else {
            vision_norm
        };
        tick.forward_noise * tick.inputs[k] / norm
    })
}

/// The trace as it stands before each tick's update.
fn pre_update_traces(ticks: &[CreditTick]) -> Vec<[f64; PATHWAY_INPUTS]> {
    let mut trace = [0.0; PATHWAY_INPUTS];
    let mut out = Vec::with_capacity(ticks.len());
    for tick in ticks {
        out.push(trace);
        if tick.death {
            trace = [0.0; PATHWAY_INPUTS];
        } else {
            let increment = trace_increment(tick);
            for k in 0..PATHWAY_INPUTS {
                trace[k] = TRACE_DECAY * trace[k] + increment[k];
            }
        }
    }
    out
}

fn rebuild_credit(ticks: &[CreditTick]) -> Result<AgentCredit, String> {
    if ticks.is_empty() {
        return Err("no ticks to rebuild credit from".to_string());
    }
    let traces = pre_update_traces(ticks);
    let total = usize_f64(ticks.len());
    let mut credit_sum = [0.0; PATHWAY_INPUTS];
    let mut life_sum = [0.0; PATHWAY_INPUTS];
    let mut centred_sum = [0.0; PATHWAY_INPUTS];
    let mut centred_ticks = 0_usize;
    for (tick, trace) in ticks.iter().zip(&traces) {
        for k in 0..PATHWAY_INPUTS {
            credit_sum[k] += tick.td_error * trace[k];
            life_sum[k] += tick.inputs[k];
            if tick.food_centred {
                centred_sum[k] += tick.inputs[k];
            }
        }
        if tick.food_centred {
            centred_ticks += 1;
        }
    }
    let life_mean_credit = credit_sum.map(|v| v / total);
    let centred_input_excess: [f64; PATHWAY_INPUTS] = std::array::from_fn(|k| {
        if centred_ticks == 0 {
            0.0
        } else {
            centred_sum[k] / usize_f64(centred_ticks) - life_sum[k] / total
        }
    });
    let conditional: f64 = life_mean_credit
        .iter()
        .zip(&centred_input_excess)
        .map(|(g, m)| g * m)
        .sum();
    let unconditional: f64 = life_mean_credit.iter().sum();
    Ok(AgentCredit {
        life_mean_credit,
        centred_input_excess,
        conditional,
        unconditional,
    })
}

// ---------------------------------------------------------------------------
// Tick log
// ---------------------------------------------------------------------------

struct LogColumns {
    agent: usize,
    brain_tick: usize,
    alive: usize,
    deaths_total: usize,
    food_distance: usize,
    food_bearing: usize,
    forward_noise: usize,
    td_error: usize,
    inputs: Vec<usize>,
}

fn log_columns(name: &str, log: &TickLog) -> Result<LogColumns, String> {
    let find = |column: &str| {
        log.column_index(column)
            .ok_or_else(|| format!("{name}: tick log has no column {column}"))
    };
    let mut inputs = Vec::with_capacity(PATHWAY_INPUTS);
    for k in 0..PATHWAY_INPUTS {
        inputs.push(find(&format!("pathway_input_{k}"))?);
    }
    Ok(LogColumns {
        agent: find("agent")?,
        brain_tick: find("brain_tick")?,
        alive: find("alive")?,
        deaths_total: find("deaths_total")?,
        food_distance: find("food_distance")?,
        food_bearing: find("food_bearing")?,
        forward_noise: find("forward_noise")?,
        td_error: find("td_error")?,
        inputs,
    })
}

fn agent_index(value: f32, agent_count: usize) -> Option<usize> {
    (0..agent_count).find(|a| u16::try_from(*a).map(f32::from) == Ok(value))
}

fn finite_cell(name: &str, row: &[f32], column: usize, label: &str) -> Result<f64, String> {
    let value = row
        .get(column)
        .copied()
        .ok_or_else(|| format!("{name}: row is shorter than the column list"))?;
    if value.is_finite() {
        Ok(f64::from(value))
    } else {
        Err(format!("{name}: {label} holds a non-finite value"))
    }
}

/// Per-agent credit ticks of a tick log holding `agent_count` agents.
fn credit_ticks_from_log(
    name: &str,
    log: &TickLog,
    agent_count: usize,
    brain_ticks: u64,
) -> Result<Vec<Vec<CreditTick>>, String> {
    let cols = log_columns(name, log)?;
    let width = log.columns.len();
    let expected = usize::try_from(log.rows)
        .ok()
        .and_then(|rows| rows.checked_mul(width));
    if width == 0 || expected != Some(log.values.len()) {
        return Err(format!("{name}: tick log values do not match its header"));
    }
    let mut per_agent: Vec<Vec<CreditTick>> = vec![Vec::new(); agent_count];
    let mut previous_deaths: Vec<Option<f64>> = vec![None; agent_count];
    let mut previous_tick: Vec<Option<f64>> = vec![None; agent_count];
    for row in log.values.chunks_exact(width) {
        let agent = row
            .get(cols.agent)
            .and_then(|v| agent_index(*v, agent_count))
            .ok_or_else(|| format!("{name}: a row names an agent outside 0..{agent_count}"))?;
        let tick_number = finite_cell(name, row, cols.brain_tick, "brain_tick")?;
        if previous_tick[agent].is_some_and(|p| tick_number <= p) {
            return Err(format!(
                "{name}: brain_tick does not increase for agent {agent}"
            ));
        }
        previous_tick[agent] = Some(tick_number);
        let deaths = finite_cell(name, row, cols.deaths_total, "deaths_total")?;
        let death = previous_deaths[agent].is_some_and(|p| deaths > p);
        previous_deaths[agent] = Some(deaths);
        let alive = finite_cell(name, row, cols.alive, "alive")? > ALIVE_THRESHOLD;
        let distance = f64::from(
            *row.get(cols.food_distance)
                .ok_or_else(|| format!("{name}: row is shorter than the column list"))?,
        );
        if distance.is_nan() {
            return Err(format!("{name}: food_distance is not a number"));
        }
        let bearing = finite_cell(name, row, cols.food_bearing, "food_bearing")?.abs();
        let mut inputs = [0.0; PATHWAY_INPUTS];
        for (k, column) in cols.inputs.iter().enumerate() {
            inputs[k] = finite_cell(name, row, *column, "pathway input")?;
        }
        per_agent[agent].push(CreditTick {
            inputs,
            forward_noise: finite_cell(name, row, cols.forward_noise, "forward_noise")?,
            td_error: finite_cell(name, row, cols.td_error, "td_error")?,
            death,
            food_centred: alive
                && distance <= FOOD_IN_VIEW_DISTANCE
                && bearing < VIEW_HALF_ANGLE
                && bearing < CENTRED_BEARING,
        });
    }
    let expected_ticks = usize::try_from(brain_ticks)
        .map_err(|e| format!("{name}: brain tick count {brain_ticks}: {e}"))?;
    for (agent, ticks) in per_agent.iter().enumerate() {
        if ticks.len() != expected_ticks {
            return Err(format!(
                "{name}: agent {agent} has {} rows, expected {expected_ticks}",
                ticks.len()
            ));
        }
    }
    Ok(per_agent)
}

// ---------------------------------------------------------------------------
// Metrics reports
// ---------------------------------------------------------------------------

/// A metrics report with the file it came from and its tick log, if any.
struct Loaded {
    name: String,
    report: FreeLifeReport,
    log_path: Option<PathBuf>,
}

fn read_report(path: &Path) -> Result<FreeLifeReport, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn window_values(window: &WindowMetrics) -> Vec<f64> {
    let mut values = vec![
        window.sighting_meal_share,
        window.turn_toward_food,
        window.turn_away_from_hazard,
        window.pathway_food_correlation.encoding,
        window.pathway_food_correlation.scent,
        window.pathway_food_correlation.vision,
        window.value_gap_in_view,
        window.value_gap_centred,
    ];
    values.extend(window.turn_credit);
    values.extend(window.forward_credit);
    values
}

fn validate_report(name: &str, report: &FreeLifeReport) -> Result<(), String> {
    if !SEEDS.contains(&report.seed) {
        return Err(format!(
            "{name}: seed {} is not one of {SEEDS:?}",
            report.seed
        ));
    }
    if report.agent_count != AGENTS_PER_SEED || report.agents.len() != AGENTS_PER_SEED {
        return Err(format!(
            "{name}: {} agents ({} metrics rows), expected {AGENTS_PER_SEED}",
            report.agent_count,
            report.agents.len()
        ));
    }
    if report.ticks != TICKS_PER_SEED {
        return Err(format!(
            "{name}: {} ticks, expected {TICKS_PER_SEED}",
            report.ticks
        ));
    }
    if report.brain_source != FRESH_BRAIN_SOURCE || !report.lesions.is_empty() {
        return Err(format!(
            "{name}: the free life is not a fresh, unlesioned one"
        ));
    }
    for (index, agent) in report.agents.iter().enumerate() {
        for window in [&agent.whole, &agent.late] {
            if window_values(window).iter().any(|v| !v.is_finite()) {
                return Err(format!("{name}: agent {index} has a non-finite metric"));
            }
        }
    }
    Ok(())
}

fn validate_arm(label: &str, arm: &[Loaded]) -> Result<(), String> {
    let names: Vec<&str> = arm.iter().map(|l| l.name.as_str()).collect();
    if arm.len() != SEEDS.len() {
        return Err(format!(
            "{label} arm has {} metrics files ({}), expected {}",
            arm.len(),
            names.join(", "),
            SEEDS.len()
        ));
    }
    for loaded in arm {
        validate_report(&loaded.name, &loaded.report)?;
    }
    for seed in SEEDS {
        let holders: Vec<&str> = arm
            .iter()
            .filter(|l| l.report.seed == seed)
            .map(|l| l.name.as_str())
            .collect();
        if holders.len() != 1 {
            return Err(format!(
                "{label} arm has {} files for seed {seed} ({}), expected 1",
                holders.len(),
                names.join(", ")
            ));
        }
    }
    Ok(())
}

fn base_record(arm: Arm, seed: u64, agent: usize, metrics: &AgentMetrics) -> AgentRecord {
    let window = &metrics.whole;
    AgentRecord {
        arm,
        seed,
        agent,
        meals: window.meals,
        deaths: window.deaths,
        sightings: window.sightings,
        sightings_ending_in_meal: window.sightings_ending_in_meal,
        sighting_meal_share: share(window.sightings_ending_in_meal, window.sightings),
        life_mean_credit: None,
        centred_input_excess: None,
        conditional_credit: None,
        unconditional_credit: None,
        free_life_forward_credit: None,
    }
}

fn forward_record(
    seed: u64,
    agent: usize,
    metrics: &AgentMetrics,
    credit: &AgentCredit,
) -> AgentRecord {
    AgentRecord {
        life_mean_credit: Some(credit.life_mean_credit.to_vec()),
        centred_input_excess: Some(credit.centred_input_excess.to_vec()),
        conditional_credit: Some(credit.conditional),
        unconditional_credit: Some(credit.unconditional),
        free_life_forward_credit: Some(metrics.whole.forward_credit.to_vec()),
        ..base_record(Arm::Route, seed, agent, metrics)
    }
}

fn share(ending: u64, sightings: u64) -> f64 {
    if sightings == 0 {
        0.0
    } else {
        count_as_f64(ending) / count_as_f64(sightings)
    }
}

fn arm_records(arm: Arm, loaded: &[Loaded]) -> Vec<AgentRecord> {
    let mut sorted: Vec<&Loaded> = loaded.iter().collect();
    sorted.sort_by_key(|l| l.report.seed);
    sorted
        .iter()
        .flat_map(|l| {
            l.report
                .agents
                .iter()
                .enumerate()
                .map(|(index, metrics)| base_record(arm, l.report.seed, index, metrics))
        })
        .collect()
}

fn forward_records(loaded: &[Loaded]) -> Result<Vec<AgentRecord>, String> {
    let mut sorted: Vec<&Loaded> = loaded.iter().collect();
    sorted.sort_by_key(|l| l.report.seed);
    let mut records = Vec::new();
    for item in sorted {
        let path = item
            .log_path
            .as_ref()
            .ok_or_else(|| format!("{}: no route log was given", item.name))?;
        let log_name = path.display().to_string();
        let log = read_tick_log(path).map_err(|e| format!("{log_name}: {e}"))?;
        let ticks = credit_ticks_from_log(
            &log_name,
            &log,
            item.report.agent_count,
            item.report.brain_ticks,
        )?;
        for (index, (agent_ticks, metrics)) in ticks.iter().zip(&item.report.agents).enumerate() {
            let credit = rebuild_credit(agent_ticks).map_err(|e| format!("{log_name}: {e}"))?;
            records.push(forward_record(item.report.seed, index, metrics, &credit));
        }
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// G1: enough agents with positive credit and a positive pooled mean.
fn g1_passes(positive_agents: usize, pooled_mean: f64) -> bool {
    positive_agents >= G1_MIN_POSITIVE_AGENTS && pooled_mean > 0.0
}

/// G2: the pooled share is higher and enough seeds are higher.
fn g2_passes(route_share: f64, control_share: f64, seeds_higher: usize) -> bool {
    route_share > control_share && seeds_higher >= G2_MIN_HIGHER_SEEDS
}

/// G3: meals not worse than minus one standard error, deaths not worse than plus one.
fn g3_passes(
    meals_difference: f64,
    meals_standard_error: f64,
    deaths_difference: f64,
    deaths_standard_error: f64,
) -> bool {
    meals_difference >= -meals_standard_error - GATE_TOLERANCE
        && deaths_difference <= deaths_standard_error + GATE_TOLERANCE
}

fn status(passed: bool) -> String {
    (if passed { PASS } else { FAIL }).to_string()
}

fn pool(agents: &[&AgentRecord]) -> ArmPooled {
    let meals: Vec<f64> = agents.iter().map(|a| count_as_f64(a.meals)).collect();
    let deaths: Vec<f64> = agents.iter().map(|a| count_as_f64(a.deaths)).collect();
    let sightings: u64 = agents.iter().map(|a| a.sightings).sum();
    let ending: u64 = agents.iter().map(|a| a.sightings_ending_in_meal).sum();
    let conditional: Option<Vec<f64>> = agents.iter().map(|a| a.conditional_credit).collect();
    let unconditional: Option<Vec<f64>> = agents.iter().map(|a| a.unconditional_credit).collect();
    ArmPooled {
        agents: agents.len(),
        mean_meals: mean(&meals),
        mean_deaths: mean(&deaths),
        meals_variance: sample_variance(&meals),
        deaths_variance: sample_variance(&deaths),
        sightings,
        sightings_ending_in_meal: ending,
        sighting_meal_share: share(ending, sightings),
        positive_credit_agents: conditional
            .as_ref()
            .map(|c| c.iter().filter(|v| **v > 0.0).count()),
        mean_conditional_credit: conditional.as_deref().map(mean),
        mean_unconditional_credit: unconditional.as_deref().map(mean),
    }
}

fn welch_standard_error(
    route_variance: f64,
    route_agents: usize,
    control_variance: f64,
    control_agents: usize,
) -> f64 {
    (route_variance / usize_f64(route_agents) + control_variance / usize_f64(control_agents)).sqrt()
}

fn arm_agents(agents: &[AgentRecord], arm: Arm, seed: Option<u64>) -> Vec<&AgentRecord> {
    agents
        .iter()
        .filter(|a| a.arm == arm && seed.is_none_or(|s| a.seed == s))
        .collect()
}

/// Everything derived from the agents and the generation means.
struct Derived {
    pooled: Pooled,
    per_seed: Vec<SeedRecord>,
    gates: [String; 4],
    verdict: String,
    failed_gate: Option<String>,
}

fn seed_record(
    seed: u64,
    agents: &[AgentRecord],
    generations: Option<GenerationMeans>,
    g4: &str,
) -> Result<SeedRecord, String> {
    let control = arm_agents(agents, Arm::Control, Some(seed));
    let route = arm_agents(agents, Arm::Route, Some(seed));
    if control.len() != AGENTS_PER_SEED || route.len() != AGENTS_PER_SEED {
        return Err(format!(
            "agents: seed {seed} has {} control and {} route agents, expected {AGENTS_PER_SEED} each",
            control.len(),
            route.len()
        ));
    }
    let (control_pool, route_pool) = (pool(&control), pool(&route));
    let gated = g4 != NOT_APPLICABLE;
    Ok(SeedRecord {
        seed,
        control_sighting_meal_share: control_pool.sighting_meal_share,
        route_sighting_meal_share: route_pool.sighting_meal_share,
        share_higher: route_pool.sighting_meal_share > control_pool.sighting_meal_share,
        control_mean_meals: control_pool.mean_meals,
        route_mean_meals: route_pool.mean_meals,
        control_mean_deaths: control_pool.mean_deaths,
        route_mean_deaths: route_pool.mean_deaths,
        control_generation_mean: generations.filter(|_| gated).map(|g| g.control),
        route_generation_mean: generations.filter(|_| gated).map(|g| g.route),
        g4: g4.to_string(),
    })
}

/// Mean composite fitness of both arms over the G4 window, as recorded.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GenerationMeans {
    control: f64,
    route: f64,
}

fn derive(
    route: Route,
    agents: &[AgentRecord],
    generations: &[Option<GenerationMeans>],
) -> Result<Derived, String> {
    if generations.len() != SEEDS.len() {
        return Err(format!(
            "per_seed: {} entries, expected {}",
            generations.len(),
            SEEDS.len()
        ));
    }
    let control = arm_agents(agents, Arm::Control, None);
    let route_agents = arm_agents(agents, Arm::Route, None);
    if control.len() != TOTAL_AGENTS || route_agents.len() != TOTAL_AGENTS {
        return Err(format!(
            "agents: {} control and {} route agents, expected {TOTAL_AGENTS} each",
            control.len(),
            route_agents.len()
        ));
    }
    let (control_pool, route_pool) = (pool(&control), pool(&route_agents));
    let seed_stats = SEEDS
        .iter()
        .map(|seed| seed_record(*seed, agents, None, NOT_APPLICABLE))
        .collect::<Result<Vec<_>, _>>()?;
    let seeds_higher = seed_stats.iter().filter(|s| s.share_higher).count();
    let g1 = if route == Route::Forward {
        let credits: Option<Vec<f64>> = route_agents.iter().map(|a| a.conditional_credit).collect();
        let credits = credits.ok_or("agents: a route agent has no conditional credit")?;
        let positive = credits.iter().filter(|v| **v > 0.0).count();
        status(g1_passes(positive, mean(&credits)))
    } else {
        NOT_APPLICABLE.to_string()
    };
    let g2 = status(g2_passes(
        route_pool.sighting_meal_share,
        control_pool.sighting_meal_share,
        seeds_higher,
    ));
    let pooled = Pooled {
        meals_difference: route_pool.mean_meals - control_pool.mean_meals,
        meals_standard_error: welch_standard_error(
            route_pool.meals_variance,
            route_pool.agents,
            control_pool.meals_variance,
            control_pool.agents,
        ),
        deaths_difference: route_pool.mean_deaths - control_pool.mean_deaths,
        deaths_standard_error: welch_standard_error(
            route_pool.deaths_variance,
            route_pool.agents,
            control_pool.deaths_variance,
            control_pool.agents,
        ),
        control: control_pool,
        route: route_pool,
    };
    let g3 = status(g3_passes(
        pooled.meals_difference,
        pooled.meals_standard_error,
        pooled.deaths_difference,
        pooled.deaths_standard_error,
    ));
    let early_failed = [&g1, &g2, &g3].iter().any(|g| g.as_str() == FAIL);
    let (per_seed, g4) = if early_failed {
        (seed_stats, NOT_APPLICABLE.to_string())
    } else {
        gated_seeds(agents, generations)?
    };
    let gates = [g1, g2, g3, g4];
    let failed = gates.iter().position(|g| g == FAIL);
    let verdict = if failed.is_some() {
        FAIL
    } else if gates.iter().any(|g| g == PENDING) {
        PENDING
    } else {
        PASS
    };
    Ok(Derived {
        pooled,
        per_seed,
        failed_gate: failed.map(|i| GATE_NAMES[i].to_string()),
        verdict: verdict.to_string(),
        gates,
    })
}

fn gated_seeds(
    agents: &[AgentRecord],
    generations: &[Option<GenerationMeans>],
) -> Result<(Vec<SeedRecord>, String), String> {
    let supplied = generations.iter().filter(|g| g.is_some()).count();
    if supplied != 0 && supplied != SEEDS.len() {
        return Err("per_seed: generation means are present for only some seeds".to_string());
    }
    let mut records = Vec::new();
    for (seed, means) in SEEDS.iter().zip(generations) {
        let seed_status = match means {
            None => PENDING,
            Some(g) if g.route >= g.control => PASS,
            Some(_) => FAIL,
        };
        records.push(seed_record(*seed, agents, *means, seed_status)?);
    }
    let overall = if records.iter().any(|r| r.g4 == FAIL) {
        FAIL
    } else if records.iter().any(|r| r.g4 == PENDING) {
        PENDING
    } else {
        PASS
    };
    Ok((records, overall.to_string()))
}

fn assemble(
    header: &Header,
    agents: Vec<AgentRecord>,
    generations: &[Option<GenerationMeans>],
) -> Result<Record, String> {
    if header.commit.is_empty() {
        return Err("commit must not be empty".to_string());
    }
    let derived = derive(header.route, &agents, generations)?;
    let [g1, g2, g3, g4] = derived.gates;
    Ok(Record {
        route: header.route,
        commit: header.commit.clone(),
        setting_under_test: header.setting.clone(),
        verdict: derived.verdict,
        failed_gate: derived.failed_gate,
        g1,
        g2,
        g3,
        g4,
        seeds: SEEDS.to_vec(),
        agents_per_seed: AGENTS_PER_SEED,
        ticks_per_seed: Some(TICKS_PER_SEED),
        config: Some(Config::current()),
        agents,
        per_seed: derived.per_seed,
        pooled: Some(derived.pooled),
    })
}

fn not_applicable_record(commit: &str) -> Result<Record, String> {
    if commit.is_empty() {
        return Err("commit must not be empty".to_string());
    }
    Ok(Record {
        route: Route::Critic,
        commit: commit.to_string(),
        setting_under_test: SETTING_ENCODING.to_string(),
        verdict: NOT_APPLICABLE.to_string(),
        failed_gate: None,
        g1: NOT_APPLICABLE.to_string(),
        g2: NOT_APPLICABLE.to_string(),
        g3: NOT_APPLICABLE.to_string(),
        g4: NOT_APPLICABLE.to_string(),
        seeds: Vec::new(),
        agents_per_seed: 0,
        ticks_per_seed: None,
        config: None,
        agents: Vec::new(),
        per_seed: Vec::new(),
        pooled: None,
    })
}

// ---------------------------------------------------------------------------
// Generations (G4)
// ---------------------------------------------------------------------------

/// Mean over the G4 window of the per-generation mean composite fitness.
fn generation_mean(conn: &Connection, name: &str) -> Result<f64, String> {
    let mut statement = conn
        .prepare(
            "SELECT n.generation, AVG(a.composite_fitness)
             FROM node n JOIN agent_result a ON a.node_id = n.id
             WHERE n.generation BETWEEN ?1 AND ?2
             GROUP BY n.generation ORDER BY n.generation",
        )
        .map_err(|e| format!("{name}: {e}"))?;
    let rows = statement
        .query_map(params![FIRST_GENERATION, LAST_GENERATION], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<f64>>(1)?))
        })
        .map_err(|e| format!("{name}: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{name}: {e}"))?;
    let mut means = Vec::new();
    for generation in FIRST_GENERATION..=LAST_GENERATION {
        let value = rows
            .iter()
            .find(|(g, _)| *g == generation)
            .and_then(|(_, v)| *v)
            .ok_or_else(|| format!("{name}: generation {generation} has no agent results"))?;
        if !value.is_finite() {
            return Err(format!(
                "{name}: generation {generation} fitness is not finite"
            ));
        }
        means.push(value);
    }
    Ok(mean(&means))
}

fn seed_generations(
    seed: u64,
    control: (&Connection, &str),
    route: (&Connection, &str),
) -> Result<SeedGenerations, String> {
    Ok(SeedGenerations {
        seed,
        control_mean: generation_mean(control.0, control.1)?,
        route_mean: generation_mean(route.0, route.1)?,
    })
}

fn failing_seeds(means: &[SeedGenerations]) -> Vec<u64> {
    means
        .iter()
        .filter(|m| m.route_mean < m.control_mean)
        .map(|m| m.seed)
        .collect()
}

fn open_read_only(path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(DB_BUSY_TIMEOUT_SECONDS))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(conn)
}

fn apply_generations(record: Record, means: &[SeedGenerations]) -> Result<Record, String> {
    if record.verdict != PENDING || record.g4 != PENDING {
        return Err(format!(
            "record is not awaiting generations (verdict {}, g4 {})",
            record.verdict, record.g4
        ));
    }
    let seeds: Vec<u64> = means.iter().map(|m| m.seed).collect();
    if seeds != record.seeds {
        return Err(format!(
            "generation seeds {seeds:?} differ from the record's {:?}",
            record.seeds
        ));
    }
    let header = Header {
        route: record.route,
        setting: record.setting_under_test.clone(),
        commit: record.commit.clone(),
    };
    let agents = normalise_agents(record.route, &record.agents)?;
    let generations: Vec<Option<GenerationMeans>> = means
        .iter()
        .map(|m| {
            Some(GenerationMeans {
                control: m.control_mean,
                route: m.route_mean,
            })
        })
        .collect();
    assemble(&header, agents, &generations)
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

fn finite_vector(field: &str, values: &[f64]) -> Result<(), String> {
    if values.len() != PATHWAY_INPUTS || values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{field}: expected {PATHWAY_INPUTS} finite values"));
    }
    Ok(())
}

/// Recomputes an agent's share and credits from its counts and credit parts.
fn normalise_agent(route: Route, agent: &AgentRecord) -> Result<AgentRecord, String> {
    let mut out = agent.clone();
    out.sighting_meal_share = share(agent.sightings_ending_in_meal, agent.sightings);
    let has_credit = route == Route::Forward && agent.arm == Arm::Route;
    if !has_credit {
        let absent = agent.life_mean_credit.is_none()
            && agent.centred_input_excess.is_none()
            && agent.conditional_credit.is_none()
            && agent.unconditional_credit.is_none()
            && agent.free_life_forward_credit.is_none();
        if absent {
            return Ok(out);
        }
        return Err(format!(
            "agents: agent {} of seed {} carries credit it should not have",
            agent.agent, agent.seed
        ));
    }
    let (Some(g), Some(m), Some(copied)) = (
        &agent.life_mean_credit,
        &agent.centred_input_excess,
        &agent.free_life_forward_credit,
    ) else {
        return Err(format!(
            "agents: agent {} of seed {} lacks credit parts",
            agent.agent, agent.seed
        ));
    };
    finite_vector("life_mean_credit", g)?;
    finite_vector("centred_input_excess", m)?;
    finite_vector("free_life_forward_credit", copied)?;
    out.conditional_credit = Some(g.iter().zip(m).map(|(a, b)| a * b).sum());
    out.unconditional_credit = Some(g.iter().sum());
    Ok(out)
}

fn normalise_agents(route: Route, agents: &[AgentRecord]) -> Result<Vec<AgentRecord>, String> {
    agents.iter().map(|a| normalise_agent(route, a)).collect()
}

fn join_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

/// First path at which `actual` differs from `expected`, numbers within tolerance.
fn json_diff(path: &str, expected: &Value, actual: &Value) -> Option<String> {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) if close(x, y) => None,
            _ => Some(path.to_string()),
        },
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let child = join_path(path, key);
                match b.get(key) {
                    Some(other) => {
                        if let Some(diff) = json_diff(&child, value, other) {
                            return Some(diff);
                        }
                    }
                    None => return Some(child),
                }
            }
            b.keys()
                .find(|k| !a.contains_key(*k))
                .map(|k| join_path(path, k))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(path.to_string());
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (x, y))| json_diff(&format!("{path}[{i}]"), x, y))
        }
        _ => (expected != actual).then(|| path.to_string()),
    }
}

fn record_value(record: &Record) -> Result<Value, String> {
    serde_json::to_value(record).map_err(|e| format!("record: {e}"))
}

fn critic_setting(setting: &str) -> Option<CriticInput> {
    match setting {
        SETTING_ENCODING => Some(CriticInput::Encoding),
        SETTING_PATHWAY => Some(CriticInput::Pathway),
        SETTING_BOTH => Some(CriticInput::EncodingAndPathway),
        _ => None,
    }
}

fn verify_record(record: &Record, defaults: CodeDefaults) -> Result<(), String> {
    if record.commit.is_empty() {
        return Err("commit: must not be empty".to_string());
    }
    if record.verdict == NOT_APPLICABLE {
        let expected = not_applicable_record(&record.commit)?;
        if let Some(path) = json_diff("", &record_value(&expected)?, &record_value(record)?) {
            return Err(format!("{path}: disagrees with the not-applicable record"));
        }
        return check_critic_default(defaults, CriticInput::Encoding, "verdict");
    }
    let setting_ok = match record.route {
        Route::Forward => record.setting_under_test == SETTING_FORWARD,
        Route::Critic => matches!(
            record.setting_under_test.as_str(),
            SETTING_PATHWAY | SETTING_BOTH
        ),
    };
    if !setting_ok {
        return Err(format!(
            "setting_under_test: {} does not fit the route",
            record.setting_under_test
        ));
    }
    let agents = normalise_agents(record.route, &record.agents)?;
    let generations: Vec<Option<GenerationMeans>> = record
        .per_seed
        .iter()
        .map(|s| {
            s.control_generation_mean
                .zip(s.route_generation_mean)
                .map(|(control, route)| GenerationMeans { control, route })
        })
        .collect();
    let header = Header {
        route: record.route,
        setting: record.setting_under_test.clone(),
        commit: record.commit.clone(),
    };
    let expected = assemble(&header, agents, &generations)?;
    if let Some(path) = json_diff("", &record_value(&expected)?, &record_value(record)?) {
        return Err(format!("{path}: disagrees with the recomputed value"));
    }
    let adopted = record.verdict == PASS;
    match record.route {
        Route::Forward => {
            if adopted != defaults.forward_pathway_enabled {
                return Err(format!(
                    "verdict {} disagrees with the forward_pathway_enabled default {}",
                    record.verdict, defaults.forward_pathway_enabled
                ));
            }
            Ok(())
        }
        Route::Critic => {
            let setting = critic_setting(&record.setting_under_test)
                .ok_or("setting_under_test: unknown critic input")?;
            let expected_default = if adopted {
                setting
            } else {
                CriticInput::Encoding
            };
            check_critic_default(defaults, expected_default, "verdict")
        }
    }
}

fn check_critic_default(
    defaults: CodeDefaults,
    expected: CriticInput,
    field: &str,
) -> Result<(), String> {
    if defaults.critic_input == expected {
        Ok(())
    } else {
        Err(format!(
            "{field}: the critic_input default is {:?} but the record requires {expected:?}",
            defaults.critic_input
        ))
    }
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "pathway_route_gate",
    about = "Decides whether a pathway route earns its default"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute credit and gates G1-G3 from the free-life arms.
    FreeLife(FreeLifeArgs),
    /// Write the empty-arm critic record when the encoding input won.
    NotApplicable(NotApplicableArgs),
    /// Add G4 from the generation databases and the final verdict.
    Generations(GenerationsArgs),
    /// Recompute every number, gate and verdict of a record.
    Verify(VerifyArgs),
}

#[derive(Args)]
struct FreeLifeArgs {
    #[arg(long, value_enum)]
    route: Route,
    /// Required for the critic route, rejected for the forward route.
    #[arg(long)]
    critic_input: Option<String>,
    #[arg(long)]
    commit: String,
    /// One metrics report per seed of the control arm.
    #[arg(long, num_args = 1.., required = true)]
    control_metrics: Vec<PathBuf>,
    /// One metrics report per seed of the route arm.
    #[arg(long, num_args = 1.., required = true)]
    route_metrics: Vec<PathBuf>,
    /// One tick log per route metrics report, in the same order (forward route only).
    #[arg(long, num_args = 1..)]
    route_log: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct NotApplicableArgs {
    #[arg(long, value_enum)]
    route: Route,
    #[arg(long)]
    critic_input: String,
    #[arg(long)]
    winner_record: PathBuf,
    #[arg(long)]
    commit: String,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct GenerationsArgs {
    #[arg(long)]
    record: PathBuf,
    /// One control database per seed, in ascending seed order.
    #[arg(long, num_args = 1.., required = true)]
    control_db: Vec<PathBuf>,
    /// One route database per seed, in ascending seed order.
    #[arg(long, num_args = 1.., required = true)]
    route_db: Vec<PathBuf>,
}

#[derive(Args)]
struct VerifyArgs {
    record: PathBuf,
}

fn write_record(path: &Path, record: &Record) -> Result<(), String> {
    let text = serde_json::to_string_pretty(record).map_err(|e| format!("record: {e}"))?;
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_record(path: &Path) -> Result<Record, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_critic_input(text: &str) -> Result<CriticInput, String> {
    critic_setting(text).ok_or_else(|| {
        format!("unknown critic input {text}; expected {SETTING_ENCODING}, {SETTING_PATHWAY} or {SETTING_BOTH}")
    })
}

fn read_arm(paths: &[PathBuf], logs: &[PathBuf]) -> Result<Vec<Loaded>, String> {
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| -> Result<Loaded, String> {
            Ok(Loaded {
                name: path.display().to_string(),
                report: read_report(path)?,
                log_path: logs.get(index).cloned(),
            })
        })
        .collect()
}

fn setting_for(args: &FreeLifeArgs) -> Result<String, String> {
    match args.route {
        Route::Critic => {
            let text = args
                .critic_input
                .as_deref()
                .ok_or("--critic-input is required for --route critic")?;
            if !args.route_log.is_empty() {
                return Err("--route-log is rejected for --route critic".to_string());
            }
            match parse_critic_input(text)? {
                CriticInput::Encoding => {
                    Err("critic input encoding has no route arm; use not-applicable".to_string())
                }
                CriticInput::Pathway => Ok(SETTING_PATHWAY.to_string()),
                CriticInput::EncodingAndPathway => Ok(SETTING_BOTH.to_string()),
            }
        }
        Route::Forward => {
            if args.critic_input.is_some() {
                return Err("--critic-input is rejected for --route forward".to_string());
            }
            if args.route_log.len() != args.route_metrics.len() {
                return Err(format!(
                    "--route-log needs one log per route metrics file ({} given, {} metrics files)",
                    args.route_log.len(),
                    args.route_metrics.len()
                ));
            }
            Ok(SETTING_FORWARD.to_string())
        }
    }
}

fn run_free_life(args: &FreeLifeArgs) -> Result<String, String> {
    if args.commit.is_empty() {
        return Err("--commit must not be empty".to_string());
    }
    let setting = setting_for(args)?;
    let control = read_arm(&args.control_metrics, &[])?;
    let route_arm = read_arm(&args.route_metrics, &args.route_log)?;
    validate_arm("control", &control)?;
    validate_arm("route", &route_arm)?;
    let mut agents = arm_records(Arm::Control, &control);
    if args.route == Route::Forward {
        agents.extend(forward_records(&route_arm)?);
    } else {
        agents.extend(arm_records(Arm::Route, &route_arm));
    }
    let header = Header {
        route: args.route,
        setting,
        commit: args.commit.clone(),
    };
    let record = assemble(&header, agents, &[None; SEEDS.len()])?;
    write_record(&args.out, &record)?;
    Ok(format!("verdict {}", record.verdict))
}

fn run_not_applicable(args: &NotApplicableArgs) -> Result<String, String> {
    if args.route != Route::Critic {
        return Err("not-applicable only applies to --route critic".to_string());
    }
    if parse_critic_input(&args.critic_input)? != CriticInput::Encoding {
        return Err("not-applicable requires --critic-input encoding".to_string());
    }
    let name = args.winner_record.display().to_string();
    let text = fs::read_to_string(&args.winner_record).map_err(|e| format!("{name}: {e}"))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("{name}: {e}"))?;
    let winner = value
        .get("winner")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{name}: no top-level string field winner"))?;
    if winner != SETTING_ENCODING {
        return Err(format!(
            "{name}: winner is {winner}, not-applicable requires {SETTING_ENCODING}"
        ));
    }
    let record = not_applicable_record(&args.commit)?;
    write_record(&args.out, &record)?;
    Ok(format!("verdict {NOT_APPLICABLE}"))
}

fn run_generations(args: &GenerationsArgs) -> Result<String, String> {
    let record = read_record(&args.record)?;
    if args.control_db.len() != record.seeds.len() || args.route_db.len() != record.seeds.len() {
        return Err(format!(
            "expected {} control and {} route databases, got {} and {}",
            record.seeds.len(),
            record.seeds.len(),
            args.control_db.len(),
            args.route_db.len()
        ));
    }
    let mut means = Vec::new();
    for ((seed, control_path), route_path) in record
        .seeds
        .iter()
        .zip(&args.control_db)
        .zip(&args.route_db)
    {
        let control = open_read_only(control_path)?;
        let route = open_read_only(route_path)?;
        means.push(seed_generations(
            *seed,
            (&control, &control_path.display().to_string()),
            (&route, &route_path.display().to_string()),
        )?);
    }
    let finished = apply_generations(record, &means)?;
    write_record(&args.record, &finished)?;
    let failing = failing_seeds(&means);
    if failing.is_empty() {
        Ok(format!("verdict {}", finished.verdict))
    } else {
        Ok(format!(
            "verdict {}: G4 fails in seeds {failing:?}",
            finished.verdict
        ))
    }
}

fn run_verify(args: &VerifyArgs) -> Result<String, String> {
    let record = read_record(&args.record)?;
    let defaults = CodeDefaults {
        critic_input: DEFAULT_CRITIC_INPUT,
        forward_pathway_enabled: DEFAULT_FORWARD_PATHWAY_ENABLED,
    };
    verify_record(&record, defaults).map_err(|e| format!("{}: {e}", args.record.display()))?;
    Ok(format!("verified, verdict {}", record.verdict))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match &cli.command {
        Command::FreeLife(args) => run_free_life(args),
        Command::NotApplicable(args) => run_not_applicable(args),
        Command::Generations(args) => run_generations(args),
        Command::Verify(args) => run_verify(args),
    };
    match outcome {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_sandbox::free_life::metrics::PathwayCorrelation;
    use xagent_sandbox::free_life::tick_log::columns;

    const CREDIT_TOLERANCE: f64 = 1e-12;
    const CONTROL_SHARE_ENDING: u64 = 3;
    const ROUTE_SHARE_ENDING: u64 = 5;
    const SIGHTINGS: u64 = 10;
    const FIXTURE_SE: f64 = 0.5;
    const BOUNDARY_NUDGE: f64 = 1e-6;
    const SIGN_PHASE_TICKS: usize = 5;
    /// Hand-computed TD error of tick 1 in the credit fixture.
    const SPEC_DELTA_TICK_ONE: f64 = 0.685;
    /// Hand-computed TD error of tick 2 in the credit fixture.
    const SPEC_DELTA_TICK_TWO: f64 = -0.4;
    /// Hand-computed TD error of tick 3 in the credit fixture.
    const SPEC_DELTA_TICK_THREE: f64 = 0.3;
    /// Single-precision copy of `SPEC_DELTA_TICK_ONE` for the logged column.
    const LOG_DELTA_TICK_ONE: f32 = 0.685;
    /// Single-precision copy of `SPEC_DELTA_TICK_TWO` for the logged column.
    const LOG_DELTA_TICK_TWO: f32 = -0.4;
    /// Single-precision copy of `SPEC_DELTA_TICK_THREE` for the logged column.
    const LOG_DELTA_TICK_THREE: f32 = 0.3;
    /// Agents in the single-agent log fixtures.
    const LOG_ONE_AGENT: usize = 1;
    /// Agents in the log fixture that expects a missing agent.
    const LOG_TWO_AGENTS: usize = 2;
    /// Ticks in the credit log fixture.
    const LOG_FOUR_TICKS: u64 = 4;
    /// Ticks in the missing-agent log fixture.
    const LOG_THREE_TICKS: u64 = 3;
    /// Ticks in the NaN log fixture.
    const LOG_TWO_TICKS: u64 = 2;
    /// Period of the alternating per-agent meal and death wobble.
    const WOBBLE_PERIOD: usize = 2;
    /// Control-arm base meals per agent.
    const CONTROL_BASE_MEALS: u64 = 5;
    /// Control-arm base deaths per agent.
    const CONTROL_BASE_DEATHS: u64 = 2;
    /// Route-arm base meals per agent.
    const ROUTE_BASE_MEALS: u64 = 6;
    /// Control-arm generation fitness in the synthetic record.
    const CONTROL_FITNESS: f64 = 1.0;
    /// Route-arm generation fitness in the synthetic record and G4 passes.
    const ROUTE_FITNESS: f64 = 2.0;
    /// Route-arm fitness on the one seed that is lower than control.
    const LOWER_ROUTE_FITNESS: f64 = 0.5;
    /// Index into `SEEDS` of the seed whose route fitness is lower.
    const LOWER_SEED_INDEX: usize = 1;
    /// Brain ticks recorded in the loaded report fixture.
    const FIXTURE_BRAIN_TICKS: u64 = 4;
    /// Food distance logged in the log fixture rows.
    const FIXTURE_FOOD_DISTANCE: f32 = 100.0;
    /// Seed count as used by the G4 fixtures.
    const SEED_COUNT: usize = SEEDS.len();
    /// Hand-computed pre-update trace of tick 1 in the credit fixture.
    const SPEC_TRACE_TICK_ONE: f64 = 0.5;
    /// Hand-computed pre-update trace of tick 2 in the credit fixture.
    const SPEC_TRACE_TICK_TWO: f64 = 0.9365;
    /// Hand-computed life mean credit of the credit fixture.
    const SPEC_MEAN_CREDIT: f64 = -0.008_025;
    /// Smallest change in mean credit that proves the death flag matters.
    const DEATH_FLAG_EFFECT: f64 = 1e-3;
    /// Smell input value in the normalisation fixture.
    const NORMALISE_SMELL_INPUT: f64 = 2.0;
    /// Expected smell trace increment for `NORMALISE_SMELL_INPUT`.
    const NORMALISE_SMELL_INCREMENT: f64 = 0.4;
    /// Vision input value in the normalisation fixture.
    const NORMALISE_VISION_INPUT: f64 = 3.0;
    /// Expected vision trace increment for `NORMALISE_VISION_INPUT`.
    const NORMALISE_VISION_INCREMENT: f64 = 0.3;
    /// Magnitude of the TD error in the sign fixtures.
    const SIGN_DELTA: f64 = 0.5;
    /// Sighting meal share stored in the window fixture.
    const WINDOW_MEAL_SHARE: f64 = 0.3;
    /// Per-input offset of the copied forward credit fixture.
    const COPIED_CREDIT_OFFSET: f64 = 0.25;
    /// Positive pooled mean credit in the G1 fixtures.
    const G1_POSITIVE_MEAN: f64 = 0.25;
    /// Negative pooled mean credit in the G1 fixtures.
    const G1_NEGATIVE_MEAN: f64 = -0.1;
    /// Higher pooled share in the G2 fixtures.
    const G2_HIGH_SHARE: f64 = 0.5;
    /// Lower and control pooled share in the G2 fixtures.
    const G2_LOW_SHARE: f64 = 0.3;
    /// Generation left out of the G4 database to force an error.
    const MISSING_GENERATION: i64 = 71;
    /// Log sample values of the log credit fixture, as the first value.
    const LOG_SAMPLE_FIRST: f32 = 0.1;
    /// Tolerance for credit rebuilt from single-precision log cells.
    const LOG_CREDIT_TOLERANCE: f64 = 1e-6;
    /// Log sample values of the log credit fixture, as the second value.
    const LOG_SAMPLE_SECOND: f32 = 0.2;

    fn tick(inputs: [f64; PATHWAY_INPUTS], td_error: f64, death: bool) -> CreditTick {
        CreditTick {
            inputs,
            forward_noise: 1.0,
            td_error,
            death,
            food_centred: false,
        }
    }

    fn smell_unit() -> [f64; PATHWAY_INPUTS] {
        let mut inputs = [0.0; PATHWAY_INPUTS];
        inputs[0] = 1.0;
        inputs
    }

    fn spec_ticks(with_second_death: bool) -> Vec<CreditTick> {
        let deltas = [
            0.0,
            SPEC_DELTA_TICK_ONE,
            SPEC_DELTA_TICK_TWO,
            SPEC_DELTA_TICK_THREE,
        ];
        let deaths = [false, false, with_second_death, true];
        deltas
            .iter()
            .zip(deaths)
            .map(|(d, death)| tick(smell_unit(), *d, death))
            .collect()
    }

    #[test]
    fn credit_rebuild_matches_hand_computed_trace_and_mean() {
        let ticks = spec_ticks(true);
        let traces = pre_update_traces(&ticks);
        let expected = [0.0, SPEC_TRACE_TICK_ONE, SPEC_TRACE_TICK_TWO, 0.0];
        for (trace, want) in traces.iter().zip(expected) {
            assert!((trace[0] - want).abs() < CREDIT_TOLERANCE, "{trace:?}");
        }
        let credit = rebuild_credit(&ticks).unwrap();
        assert!((credit.life_mean_credit[0] - SPEC_MEAN_CREDIT).abs() < CREDIT_TOLERANCE);
    }

    #[test]
    fn credit_rebuild_depends_on_the_death_flag() {
        let with_death = rebuild_credit(&spec_ticks(true)).unwrap();
        let without = rebuild_credit(&spec_ticks(false)).unwrap();
        assert!(
            (with_death.life_mean_credit[0] - without.life_mean_credit[0]).abs()
                > DEATH_FLAG_EFFECT
        );
    }

    #[test]
    fn trace_normalisation_is_per_sense() {
        let mut smell = [0.0; PATHWAY_INPUTS];
        smell[0] = NORMALISE_SMELL_INPUT;
        let increment = trace_increment(&tick(smell, 0.0, false));
        assert!((increment[0] - NORMALISE_SMELL_INCREMENT).abs() < CREDIT_TOLERANCE);
        let mut vision = [0.0; PATHWAY_INPUTS];
        vision[PATHWAY_INPUTS - 1] = NORMALISE_VISION_INPUT;
        let increment = trace_increment(&tick(vision, 0.0, false));
        assert!(
            (increment[PATHWAY_INPUTS - 1] - NORMALISE_VISION_INCREMENT).abs() < CREDIT_TOLERANCE
        );
        assert!(increment[..SMELL_INPUTS].iter().all(|v| *v == 0.0));
    }

    fn sign_ticks(delta: f64) -> Vec<CreditTick> {
        let ahead = (0..SIGN_PHASE_TICKS).map(|_| CreditTick {
            food_centred: true,
            ..tick(smell_unit(), delta, false)
        });
        let elsewhere = (0..SIGN_PHASE_TICKS).map(|_| tick([0.0; PATHWAY_INPUTS], 0.0, false));
        ahead.chain(elsewhere).collect()
    }

    #[test]
    fn conditional_credit_follows_the_sign_of_delta_where_food_is_ahead() {
        let positive = rebuild_credit(&sign_ticks(SIGN_DELTA)).unwrap();
        let negative = rebuild_credit(&sign_ticks(-SIGN_DELTA)).unwrap();
        assert!(positive.conditional > 0.0 && positive.conditional.is_finite());
        assert!(negative.conditional < 0.0 && negative.conditional.is_finite());
    }

    fn window(forward_credit: [f64; PATHWAY_INPUTS]) -> WindowMetrics {
        WindowMetrics {
            samples: 0,
            alive_samples: 0,
            meals: 1,
            deaths: 1,
            hazard_deaths: 0,
            sightings: SIGHTINGS,
            sightings_ending_in_meal: CONTROL_SHARE_ENDING,
            sighting_meal_share: WINDOW_MEAL_SHARE,
            turn_toward_food: 0.0,
            turn_away_from_hazard: 0.0,
            pathway_food_correlation: PathwayCorrelation::default(),
            turn_credit: [0.0; PATHWAY_INPUTS],
            forward_credit,
            value_gap_in_view: 0.0,
            value_gap_centred: 0.0,
        }
    }

    fn metrics(forward_credit: [f64; PATHWAY_INPUTS]) -> AgentMetrics {
        AgentMetrics {
            whole: window(forward_credit),
            late: window([0.0; PATHWAY_INPUTS]),
        }
    }

    #[test]
    fn copied_forward_credit_is_verbatim_and_independent_of_conditional_credit() {
        let credit = rebuild_credit(&sign_ticks(SIGN_DELTA)).unwrap();
        let original: [f64; PATHWAY_INPUTS] =
            std::array::from_fn(|k| usize_f64(k) + COPIED_CREDIT_OFFSET);
        let first = forward_record(SEEDS[0], 0, &metrics(original), &credit);
        assert_eq!(
            first.free_life_forward_credit.as_deref(),
            Some(original.as_slice())
        );
        let mut changed = original;
        changed[3] += 1.0;
        let second = forward_record(SEEDS[0], 0, &metrics(changed), &credit);
        assert_eq!(
            second.free_life_forward_credit.unwrap()[3],
            original[3] + 1.0
        );
        assert_eq!(second.conditional_credit, first.conditional_credit);
        assert_eq!(second.unconditional_credit, first.unconditional_credit);
    }

    #[test]
    fn g1_needs_twenty_one_positive_agents_and_a_positive_mean() {
        let boundary = G1_MIN_POSITIVE_AGENTS;
        assert!(!g1_passes(boundary - 1, G1_POSITIVE_MEAN));
        assert!(g1_passes(boundary, G1_POSITIVE_MEAN));
        assert!(!g1_passes(TOTAL_AGENTS, 0.0));
        assert!(!g1_passes(TOTAL_AGENTS, G1_NEGATIVE_MEAN));
    }

    #[test]
    fn g2_needs_a_higher_pooled_share_and_three_higher_seeds() {
        assert!(!g2_passes(
            G2_HIGH_SHARE,
            G2_LOW_SHARE,
            G2_MIN_HIGHER_SEEDS - 1
        ));
        assert!(g2_passes(G2_HIGH_SHARE, G2_LOW_SHARE, G2_MIN_HIGHER_SEEDS));
        assert!(!g2_passes(G2_LOW_SHARE, G2_LOW_SHARE, SEEDS.len()));
    }

    #[test]
    fn g3_boundaries_are_inclusive_at_one_standard_error() {
        let se = FIXTURE_SE;
        assert!(g3_passes(-se, se, 0.0, se));
        assert!(!g3_passes(-se - BOUNDARY_NUDGE, se, 0.0, se));
        assert!(g3_passes(0.0, se, se, se));
        assert!(!g3_passes(0.0, se, se + BOUNDARY_NUDGE, se));
    }

    #[allow(clippy::too_many_arguments)]
    fn agent_record(
        arm: Arm,
        seed: u64,
        agent: usize,
        meals: u64,
        deaths: u64,
        ending: u64,
        credit: Option<f64>,
    ) -> AgentRecord {
        let mut parts = [0.0; PATHWAY_INPUTS];
        parts[0] = credit.unwrap_or(0.0);
        let mut excess = [0.0; PATHWAY_INPUTS];
        excess[0] = 1.0;
        AgentRecord {
            arm,
            seed,
            agent,
            meals,
            deaths,
            sightings: SIGHTINGS,
            sightings_ending_in_meal: ending,
            sighting_meal_share: share(ending, SIGHTINGS),
            life_mean_credit: credit.map(|_| parts.to_vec()),
            centred_input_excess: credit.map(|_| excess.to_vec()),
            conditional_credit: credit,
            unconditional_credit: credit,
            free_life_forward_credit: credit.map(|_| vec![0.0; PATHWAY_INPUTS]),
        }
    }

    fn synthetic_agents(route: Route) -> Vec<AgentRecord> {
        let mut agents = Vec::new();
        for arm in [Arm::Control, Arm::Route] {
            for seed in SEEDS {
                for index in 0..AGENTS_PER_SEED {
                    let wobble = u64::from(index % WOBBLE_PERIOD == 1);
                    let (meals, deaths, ending, credit) = match arm {
                        Arm::Control => (
                            CONTROL_BASE_MEALS + wobble,
                            CONTROL_BASE_DEATHS + wobble,
                            CONTROL_SHARE_ENDING,
                            None,
                        ),
                        Arm::Route => (
                            ROUTE_BASE_MEALS + wobble,
                            1 + wobble,
                            ROUTE_SHARE_ENDING,
                            (route == Route::Forward).then_some(1.0),
                        ),
                    };
                    agents.push(agent_record(
                        arm, seed, index, meals, deaths, ending, credit,
                    ));
                }
            }
        }
        agents
    }

    fn header(route: Route, setting: &str) -> Header {
        Header {
            route,
            setting: setting.to_string(),
            commit: "abc123".to_string(),
        }
    }

    fn generations(control: f64, route: f64) -> Vec<Option<GenerationMeans>> {
        vec![Some(GenerationMeans { control, route }); SEEDS.len()]
    }

    fn passing_record(route: Route, setting: &str) -> Record {
        assemble(
            &header(route, setting),
            synthetic_agents(route),
            &generations(CONTROL_FITNESS, ROUTE_FITNESS),
        )
        .unwrap()
    }

    fn pending_record(route: Route, setting: &str) -> Record {
        assemble(
            &header(route, setting),
            synthetic_agents(route),
            &[None; SEEDS.len()],
        )
        .unwrap()
    }

    fn defaults(critic_input: CriticInput, forward: bool) -> CodeDefaults {
        CodeDefaults {
            critic_input,
            forward_pathway_enabled: forward,
        }
    }

    fn fitness_db(skip: Option<i64>, fitness: f64) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE node (id INTEGER PRIMARY KEY, run_id INTEGER, generation INTEGER NOT NULL);
             CREATE TABLE agent_result (id INTEGER PRIMARY KEY, node_id INTEGER REFERENCES node(id),
                 agent_index INTEGER NOT NULL, composite_fitness REAL);",
        )
        .unwrap();
        for generation in FIRST_GENERATION..=LAST_GENERATION {
            if skip == Some(generation) {
                continue;
            }
            conn.execute(
                "INSERT INTO node (id, run_id, generation) VALUES (?1, 1, ?1)",
                params![generation],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO agent_result (node_id, agent_index, composite_fitness) VALUES (?1, 0, ?2)",
                params![generation, fitness],
            )
            .unwrap();
        }
        conn
    }

    fn generation_outcome(route_fitness: [f64; SEED_COUNT]) -> Vec<SeedGenerations> {
        SEEDS
            .iter()
            .zip(route_fitness)
            .map(|(seed, fitness)| {
                let control = fitness_db(None, 1.0);
                let route = fitness_db(None, fitness);
                seed_generations(*seed, (&control, "control.db"), (&route, "route.db")).unwrap()
            })
            .collect()
    }

    #[test]
    fn g4_names_the_one_seed_where_the_route_is_lower() {
        let means = generation_outcome({
            let mut fitness = [ROUTE_FITNESS; SEED_COUNT];
            fitness[LOWER_SEED_INDEX] = LOWER_ROUTE_FITNESS;
            fitness
        });
        assert_eq!(failing_seeds(&means), vec![SEEDS[LOWER_SEED_INDEX]]);
        let record =
            apply_generations(pending_record(Route::Critic, SETTING_PATHWAY), &means).unwrap();
        assert_eq!(record.g4, FAIL);
        assert_eq!(record.verdict, FAIL);
        assert_eq!(record.failed_gate.as_deref(), Some("G4"));
    }

    #[test]
    fn g4_passes_when_every_seed_is_higher() {
        let means = generation_outcome([ROUTE_FITNESS; SEED_COUNT]);
        assert!(failing_seeds(&means).is_empty());
        let record =
            apply_generations(pending_record(Route::Critic, SETTING_PATHWAY), &means).unwrap();
        assert_eq!(record.verdict, PASS);
        assert_eq!(record.failed_gate, None);
    }

    #[test]
    fn g4_missing_generation_is_an_error_naming_the_file() {
        let conn = fitness_db(Some(MISSING_GENERATION), CONTROL_FITNESS);
        let err = generation_mean(&conn, "route_7.db").unwrap_err();
        assert!(
            err.contains("route_7.db") && err.contains(&MISSING_GENERATION.to_string()),
            "{err}"
        );
    }

    fn write_winner(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join("winner.json");
        fs::write(&path, text).unwrap();
        path
    }

    fn not_applicable_args(winner: PathBuf, out: PathBuf) -> NotApplicableArgs {
        NotApplicableArgs {
            route: Route::Critic,
            critic_input: SETTING_ENCODING.to_string(),
            winner_record: winner,
            commit: "abc123".to_string(),
            out,
        }
    }

    #[test]
    fn not_applicable_writes_the_empty_arm_record_for_an_encoding_winner() {
        let dir = tempfile::tempdir().unwrap();
        let winner = write_winner(dir.path(), r#"{"winner":"encoding"}"#);
        let out = dir.path().join("record.json");
        run_not_applicable(&not_applicable_args(winner, out.clone())).unwrap();
        let record = read_record(&out).unwrap();
        assert_eq!(record.verdict, NOT_APPLICABLE);
        assert!(record.seeds.is_empty() && record.agents.is_empty() && record.pooled.is_none());
        verify_record(&record, defaults(CriticInput::Encoding, false)).unwrap();
        assert!(verify_record(&record, defaults(CriticInput::Pathway, false)).is_err());
    }

    #[test]
    fn not_applicable_rejects_other_winners_and_missing_fields_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        for text in [r#"{"winner":"pathway"}"#, r#"{"other":1}"#] {
            let winner = write_winner(dir.path(), text);
            let out = dir.path().join("record.json");
            let err =
                run_not_applicable(&not_applicable_args(winner.clone(), out.clone())).unwrap_err();
            assert!(err.contains(&winner.display().to_string()), "{err}");
            assert!(!out.exists());
        }
        let missing = dir.path().join("absent.json");
        let err = run_not_applicable(&not_applicable_args(
            missing.clone(),
            dir.path().join("x.json"),
        ))
        .unwrap_err();
        assert!(err.contains(&missing.display().to_string()), "{err}");
    }

    #[test]
    fn verify_accepts_the_tools_own_records() {
        for route in [Route::Forward, Route::Critic] {
            let setting = if route == Route::Forward {
                SETTING_FORWARD
            } else {
                SETTING_BOTH
            };
            let record = pending_record(route, setting);
            verify_record(&record, defaults(CriticInput::Encoding, false)).unwrap();
        }
    }

    #[test]
    fn verify_rejects_edited_verdicts_numbers_gates_and_commits() {
        let base = passing_record(Route::Forward, SETTING_FORWARD);
        let ok = defaults(CriticInput::Encoding, true);
        verify_record(&base, ok).unwrap();

        let mut edited = base.clone();
        edited.verdict = FAIL.to_string();
        assert!(verify_record(&edited, ok).unwrap_err().contains("verdict"));

        let mut edited = base.clone();
        if let Some(pooled) = edited.pooled.as_mut() {
            pooled.route.mean_meals *= 1.0 + BOUNDARY_NUDGE;
        }
        assert!(verify_record(&edited, ok)
            .unwrap_err()
            .contains("pooled.route.mean_meals"));

        let mut edited = base.clone();
        edited.failed_gate = Some("G2".to_string());
        assert!(verify_record(&edited, ok)
            .unwrap_err()
            .contains("failed_gate"));

        let mut edited = base;
        edited.commit = String::new();
        assert!(verify_record(&edited, ok).unwrap_err().contains("commit"));
    }

    #[test]
    fn verify_ties_the_forward_default_to_the_verdict() {
        let record = passing_record(Route::Forward, SETTING_FORWARD);
        assert!(verify_record(&record, defaults(CriticInput::Encoding, false)).is_err());
        verify_record(&record, defaults(CriticInput::Encoding, true)).unwrap();
    }

    #[test]
    fn verify_ties_the_critic_default_to_the_winner() {
        let record = passing_record(Route::Critic, SETTING_PATHWAY);
        assert!(verify_record(&record, defaults(CriticInput::Encoding, false)).is_err());
        verify_record(&record, defaults(CriticInput::Pathway, false)).unwrap();
    }

    fn loaded(seed: u64, name: &str, agent_count: usize) -> Loaded {
        Loaded {
            name: name.to_string(),
            report: FreeLifeReport {
                agent_count,
                ticks: TICKS_PER_SEED,
                seed,
                brain_source: FRESH_BRAIN_SOURCE.to_string(),
                champion_node_id: None,
                champion_generation: None,
                lesions: Vec::new(),
                kernel_batch_size: 1,
                brain_ticks: FIXTURE_BRAIN_TICKS,
                agents: (0..agent_count)
                    .map(|_| metrics([0.0; PATHWAY_INPUTS]))
                    .collect(),
            },
            log_path: None,
        }
    }

    fn full_arm(label: &str) -> Vec<Loaded> {
        SEEDS
            .iter()
            .map(|seed| loaded(*seed, &format!("{label}_{seed}.json"), AGENTS_PER_SEED))
            .collect()
    }

    #[test]
    fn a_well_formed_arm_validates() {
        validate_arm("control", &full_arm("control")).unwrap();
    }

    /// Index of the arm entry whose metrics carry the NaN cell.
    const NAN_ARM_INDEX: usize = 1;
    /// Index of the agent within that entry that carries the NaN cell.
    const NAN_AGENT_INDEX: usize = AGENTS_PER_SEED - 1;
    /// Index of the arm entry that is short of one agent.
    const SHORT_ARM_INDEX: usize = 2;

    #[test]
    fn nan_metrics_are_rejected_naming_the_file() {
        let mut arm = full_arm("control");
        arm[NAN_ARM_INDEX].report.agents[NAN_AGENT_INDEX]
            .whole
            .sighting_meal_share = f64::NAN;
        let err = validate_arm("control", &arm).unwrap_err();
        let file = format!("control_{}.json", SEEDS[NAN_ARM_INDEX]);
        assert!(err.contains(&file), "{err}");
    }

    #[test]
    fn three_seeds_are_rejected_naming_the_files() {
        let mut arm = full_arm("control");
        arm.pop();
        let err = validate_arm("control", &arm).unwrap_err();
        let file = format!("control_{}.json", SEEDS[0]);
        assert!(err.contains(&file), "{err}");
    }

    #[test]
    fn thirty_nine_route_agents_are_rejected_naming_the_file() {
        let mut arm = full_arm("route");
        let file = format!("route_{}.json", SEEDS[SHORT_ARM_INDEX]);
        arm[SHORT_ARM_INDEX] = loaded(SEEDS[SHORT_ARM_INDEX], &file, AGENTS_PER_SEED - 1);
        let err = validate_arm("route", &arm).unwrap_err();
        assert!(err.contains(&file), "{err}");
    }

    fn log_row(names: &[String], cells: &[(&str, f32)]) -> Vec<f32> {
        let mut row = vec![0.0; names.len()];
        for (name, value) in cells {
            let index = names.iter().position(|n| n == name).unwrap();
            row[index] = *value;
        }
        row
    }

    fn log_of(rows: &[Vec<f32>]) -> TickLog {
        TickLog {
            columns: columns(),
            rows: u64::try_from(rows.len()).unwrap(),
            values: rows.iter().flatten().copied().collect(),
        }
    }

    fn log_rows(agent: f32, deltas: &[f32], deaths: &[f32]) -> Vec<Vec<f32>> {
        let names = columns();
        let mut rows = Vec::new();
        for (tick_number, (delta, death)) in deltas.iter().zip(deaths).enumerate() {
            let tick_value = f32::from(u16::try_from(tick_number).unwrap());
            rows.push(log_row(
                &names,
                &[
                    ("brain_tick", tick_value),
                    ("agent", agent),
                    ("alive", 1.0),
                    ("deaths_total", *death),
                    ("food_distance", FIXTURE_FOOD_DISTANCE),
                    ("pathway_input_0", 1.0),
                    ("forward_noise", 1.0),
                    ("td_error", *delta),
                ],
            ));
        }
        rows
    }

    #[test]
    fn log_credit_matches_the_hand_computed_credit() {
        let rows = log_rows(
            0.0,
            &[
                0.0,
                LOG_DELTA_TICK_ONE,
                LOG_DELTA_TICK_TWO,
                LOG_DELTA_TICK_THREE,
            ],
            &[0.0, 0.0, 1.0, 1.0],
        );
        let per_agent =
            credit_ticks_from_log("log.bin", &log_of(&rows), LOG_ONE_AGENT, LOG_FOUR_TICKS)
                .unwrap();
        let credit = rebuild_credit(&per_agent[0]).unwrap();
        assert!((credit.life_mean_credit[0] - SPEC_MEAN_CREDIT).abs() < LOG_CREDIT_TOLERANCE);
    }

    #[test]
    fn a_log_missing_an_agent_is_rejected_naming_the_file() {
        let rows = log_rows(
            0.0,
            &[0.0, LOG_SAMPLE_FIRST, LOG_SAMPLE_SECOND],
            &[0.0, 0.0, 0.0],
        );
        let err = credit_ticks_from_log(
            "route_log.bin",
            &log_of(&rows),
            LOG_TWO_AGENTS,
            LOG_THREE_TICKS,
        )
        .unwrap_err();
        assert!(err.contains("route_log.bin"), "{err}");
    }

    #[test]
    fn a_nan_log_cell_is_rejected_naming_the_file() {
        let mut rows = log_rows(0.0, &[0.0, LOG_SAMPLE_FIRST], &[0.0, 0.0]);
        let names = columns();
        let index = names.iter().position(|n| n == "td_error").unwrap();
        rows[1][index] = f32::NAN;
        let err = credit_ticks_from_log(
            "route_log.bin",
            &log_of(&rows),
            LOG_ONE_AGENT,
            LOG_TWO_TICKS,
        )
        .unwrap_err();
        assert!(err.contains("route_log.bin"), "{err}");
    }
}
