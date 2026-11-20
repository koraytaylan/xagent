//! Offline return fit: decides which critic input predicts discounted returns
//! best, from per-brain-tick free-life logs.
//!
//! * `fit --log <path>... --out <json>` reads the logs and writes a record for
//!   the input sets `pathway`, `encoding` and `encoding_and_pathway`, with the
//!   winner. Each log's position among the `--log` values is its seed.
//! * `verify <json>` re-derives constants, pooled values, standard errors,
//!   counts, the paired standard error and the winner from the per-agent
//!   scores, and exits non-zero on the first mismatch.
//!
//! Reuse of the free-life module, per item:
//!
//! * tick log reader, imported: `xagent_sandbox::free_life::tick_log::read_tick_log`
//!   and `MAX_EXACT_F32_INTEGER` (`crates/xagent-sandbox/src/free_life/tick_log.rs`).
//! * in-view and centred predicates, imported: `FOOD_IN_VIEW_DISTANCE`,
//!   `VIEW_HALF_ANGLE`, `CENTRED_BEARING`, `OFF_AXIS_BEARING` and
//!   `count_as_f64` (`crates/xagent-sandbox/src/free_life/metrics.rs`).
//! * the predicates themselves are private there, so they are re-implemented
//!   here (`Row` construction) with the same thresholds.
//!
//! The logged encoding is the centred memory key, used as logged.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use xagent_sandbox::free_life::metrics::{
    count_as_f64, CENTRED_BEARING, FOOD_IN_VIEW_DISTANCE, OFF_AXIS_BEARING, VIEW_HALF_ANGLE,
};
use xagent_sandbox::free_life::tick_log::{read_tick_log, TickLog, MAX_EXACT_F32_INTEGER};

/// Per-tick discount of the critic return; mirrors `common.wgsl`.
const TD_DISCOUNT: f64 = 0.97;
/// Reward used in place of the logged one on a tick that ends in death.
const TERMINAL_DEATH_TD_ERROR: f64 = -1.0;
/// Trailing ticks dropped from every log because their returns are truncated.
const TRUNCATION_TICKS: usize = 300;
/// Ticks per split block.
const BLOCK_TICKS: usize = 1000;
/// Hold-out hash modulus.
const HOLDOUT_MODULUS: u64 = 10;
/// A block is held out when its hash modulo `HOLDOUT_MODULUS` is below this.
const HOLDOUT_BELOW: u64 = 3;
/// Training blocks whose index modulo this is zero form the inner validation.
const INNER_VALIDATION_MODULUS: u64 = 5;
/// Ridge penalties tried, ascending.
const LAMBDA_GRID: [f64; 5] = [1e-3, 1e-2, 1e-1, 1.0, 10.0];
/// Number of logged whitened pathway inputs.
const PATHWAY_WIDTH: usize = 10;
/// Number of logged encoding columns.
const ENCODING_WIDTH: usize = 128;
/// Width of the combined set.
const ENCODING_AND_PATHWAY_WIDTH: usize = 138;
/// Splitmix64 state increment.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
/// First splitmix64 mixing multiplier.
const SPLITMIX_MIXER_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second splitmix64 mixing multiplier.
const SPLITMIX_MIXER_B: u64 = 0x94D0_49BB_1331_11EB;
/// First splitmix64 xor-shift.
const SPLITMIX_SHIFT_A: u32 = 30;
/// Second splitmix64 xor-shift.
const SPLITMIX_SHIFT_B: u32 = 27;
/// Final splitmix64 xor-shift.
const SPLITMIX_SHIFT_C: u32 = 31;
/// Rows between a fitted row and the row whose pathway inputs the critic read.
const PATHWAY_INPUT_LAG_ROWS: usize = 1;
/// Record layout version.
const SCHEMA_VERSION: u32 = 1;
/// Relative tolerance for real numbers in `verify`.
const VERIFY_RELATIVE_TOLERANCE: f64 = 1e-9;
/// A Cholesky pivot at or below this fraction of its diagonal is singular.
const PIVOT_RELATIVE_TOLERANCE: f64 = 1e-12;
/// Fewest agents a pooled value needs.
const MIN_AGENTS: usize = 2;

/// Critic input choice; serde names match the brain configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CriticInput {
    Encoding,
    Pathway,
    EncodingAndPathway,
}

/// One logged brain tick of one agent.
struct Tick {
    reward: f64,
    died: bool,
    alive: bool,
    food_distance: f64,
    food_bearing: f64,
    pathway: [f32; PATHWAY_WIDTH],
    encoding: Vec<f32>,
}

/// One fitted row: features are the pathway inputs then the encoding.
struct Row {
    /// Original row index; only the alignment tests read it.
    #[cfg_attr(not(test), allow(dead_code))]
    tick: usize,
    block: u64,
    target: f64,
    alive: bool,
    in_view: bool,
    centred: bool,
    off_axis: bool,
    features: Vec<f32>,
}

struct AgentData {
    seed: u64,
    agent: u64,
    file: String,
    rows: Vec<Row>,
}

#[derive(Clone, Copy)]
enum Set {
    Pathway,
    Encoding,
    EncodingAndPathway,
}

impl Set {
    const ALL: [Set; 3] = [Set::Pathway, Set::Encoding, Set::EncodingAndPathway];

    fn range(self) -> std::ops::Range<usize> {
        match self {
            Set::Pathway => 0..PATHWAY_WIDTH,
            Set::Encoding => PATHWAY_WIDTH..ENCODING_AND_PATHWAY_WIDTH,
            Set::EncodingAndPathway => 0..ENCODING_AND_PATHWAY_WIDTH,
        }
    }

    fn width(self) -> usize {
        match self {
            Set::Pathway => PATHWAY_WIDTH,
            Set::Encoding => ENCODING_WIDTH,
            Set::EncodingAndPathway => ENCODING_AND_PATHWAY_WIDTH,
        }
    }
}

// ---------------------------------------------------------------- returns

/// Discounted returns with terminal death cut; the log end counts as 0.
fn returns(rewards: &[f64], died: &[bool]) -> Vec<f64> {
    let mut out = vec![0.0; rewards.len()];
    let mut next = 0.0;
    for (t, slot) in out.iter_mut().enumerate().rev() {
        let dead = died.get(t).copied().unwrap_or(false);
        let reward = if dead {
            TERMINAL_DEATH_TD_ERROR
        } else {
            rewards.get(t).copied().unwrap_or(0.0)
        };
        let carry = if dead { 0.0 } else { 1.0 };
        next = reward + TD_DISCOUNT * carry * next;
        *slot = next;
    }
    out
}

/// Returns with the truncated tail removed.
fn truncated_returns(rewards: &[f64], died: &[bool]) -> Vec<f64> {
    let mut g = returns(rewards, died);
    g.truncate(g.len().saturating_sub(TRUNCATION_TICKS));
    g
}

// ------------------------------------------------------------------ split

/// One splitmix64 step: advances the state and returns the mixed value.
fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut z = *state;
    z = (z ^ (z >> SPLITMIX_SHIFT_A)).wrapping_mul(SPLITMIX_MIXER_A);
    z = (z ^ (z >> SPLITMIX_SHIFT_B)).wrapping_mul(SPLITMIX_MIXER_B);
    z ^ (z >> SPLITMIX_SHIFT_C)
}

fn hold_out_hash(seed: u64, agent: u64, block: u64) -> u64 {
    let mut state = seed;
    let h = splitmix_next(&mut state);
    let mut state = h ^ agent;
    let h = splitmix_next(&mut state);
    let mut state = h ^ block;
    splitmix_next(&mut state)
}

fn is_held_out(seed: u64, agent: u64, block: u64) -> bool {
    hold_out_hash(seed, agent, block) % HOLDOUT_MODULUS < HOLDOUT_BELOW
}

fn block_of(tick: usize) -> u64 {
    u64::try_from(tick / BLOCK_TICKS).unwrap_or(u64::MAX)
}

/// True when the block holding `tick` has a full `BLOCK_TICKS` ticks inside
/// `kept_len`; a trailing partial block is dropped.
fn block_is_complete(tick: usize, kept_len: usize) -> bool {
    (tick / BLOCK_TICKS + 1).saturating_mul(BLOCK_TICKS) <= kept_len
}

/// Fitted rows of one agent: returns, one-row pathway lag, life-start drop,
/// complete blocks only.
fn build_rows(ticks: &[Tick]) -> Vec<Row> {
    let rewards: Vec<f64> = ticks.iter().map(|t| t.reward).collect();
    let died: Vec<bool> = ticks.iter().map(|t| t.died).collect();
    let targets = truncated_returns(&rewards, &died);
    let kept = targets.len();
    let mut rows = Vec::new();
    for (t, target) in targets.iter().enumerate() {
        if t < PATHWAY_INPUT_LAG_ROWS || !block_is_complete(t, kept) {
            continue;
        }
        if t >= 1 && died.get(t - 1).copied().unwrap_or(false) {
            continue;
        }
        let (Some(own), Some(lagged)) = (ticks.get(t), ticks.get(t - PATHWAY_INPUT_LAG_ROWS))
        else {
            continue;
        };
        let mut features: Vec<f32> = lagged.pathway.to_vec();
        features.extend_from_slice(&own.encoding);
        let bearing = own.food_bearing.abs();
        let in_view =
            own.alive && own.food_distance <= FOOD_IN_VIEW_DISTANCE && bearing < VIEW_HALF_ANGLE;
        rows.push(Row {
            tick: t,
            block: block_of(t),
            target: *target,
            alive: own.alive,
            in_view,
            centred: in_view && bearing < CENTRED_BEARING,
            off_axis: in_view && bearing > OFF_AXIS_BEARING,
            features,
        });
    }
    rows
}

// ------------------------------------------------------------------ ridge

struct Obs<'a> {
    x: &'a [f32],
    y: f64,
}

/// Solves `a x = b` for symmetric positive definite `a` (row-major, lower
/// triangle used) by Cholesky.
#[allow(clippy::needless_range_loop)]
fn solve_spd(mut a: Vec<f64>, mut b: Vec<f64>, p: usize, file: &str) -> Result<Vec<f64>, String> {
    for j in 0..p {
        for i in j..p {
            let mut s = a[i * p + j];
            let original = s;
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                if !s.is_finite() || s <= 0.0 || s <= PIVOT_RELATIVE_TOLERANCE * original {
                    return Err(format!(
                        "{file}: singular normal equations at column {j} (pivot {s})"
                    ));
                }
                a[j * p + j] = s.sqrt();
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * b[k];
        }
        b[i] = s / a[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= a[k * p + i] * b[k];
        }
        b[i] = s / a[i * p + i];
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(format!("{file}: non-finite ridge solution"));
    }
    Ok(b)
}

/// Ridge fit; coefficient 0 is the unpenalised intercept.
fn ridge_fit(obs: &[Obs], width: usize, lambda: f64, file: &str) -> Result<Vec<f64>, String> {
    if obs.is_empty() {
        return Err(format!("{file}: no rows to fit"));
    }
    let p = width + 1;
    let mut gram = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    let mut v = vec![1.0; p];
    for o in obs {
        if o.x.len() != width {
            return Err(format!(
                "{file}: row has {} values, expected {width}",
                o.x.len()
            ));
        }
        if !o.y.is_finite() {
            return Err(format!("{file}: non-finite target"));
        }
        for (dst, src) in v.iter_mut().skip(1).zip(o.x) {
            if !src.is_finite() {
                return Err(format!("{file}: non-finite feature"));
            }
            *dst = f64::from(*src);
        }
        for (i, vi) in v.iter().enumerate() {
            if let Some(slot) = xty.get_mut(i) {
                *slot += vi * o.y;
            }
            for (j, vj) in v.iter().enumerate().take(i + 1) {
                if let Some(slot) = gram.get_mut(i * p + j) {
                    *slot += vi * vj;
                }
            }
        }
    }
    for i in 1..p {
        if let Some(slot) = gram.get_mut(i * p + i) {
            *slot += lambda;
        }
    }
    solve_spd(gram, xty, p, file)
}

fn predict(coefs: &[f64], x: &[f32], file: &str) -> Result<f64, String> {
    if x.len() + 1 != coefs.len() {
        return Err(format!(
            "{file}: row has {} values, expected {}",
            x.len(),
            coefs.len().saturating_sub(1)
        ));
    }
    let intercept = coefs.first().copied().unwrap_or(0.0);
    Ok(intercept
        + coefs
            .iter()
            .skip(1)
            .zip(x)
            .map(|(c, v)| c * f64::from(*v))
            .sum::<f64>())
}

/// Coefficient of determination; `None` when the targets have no variance.
fn r_squared(pred: &[f64], y: &[f64]) -> Option<f64> {
    if y.is_empty() || pred.len() != y.len() {
        return None;
    }
    let mean = y.iter().sum::<f64>() / count_as_f64(u64::try_from(y.len()).ok()?);
    let sst: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    let sse: f64 = pred.iter().zip(y).map(|(p, v)| (v - p) * (v - p)).sum();
    if sst > 0.0 && sst.is_finite() {
        Some(1.0 - sse / sst)
    } else {
        None
    }
}

fn inner_scores(fit: &[Obs], val: &[Obs], width: usize, file: &str) -> Result<Vec<f64>, String> {
    let ys: Vec<f64> = val.iter().map(|o| o.y).collect();
    let mut scores = Vec::new();
    for lambda in LAMBDA_GRID {
        let coefs = ridge_fit(fit, width, lambda, file)?;
        let pred = val
            .iter()
            .map(|o| predict(&coefs, o.x, file))
            .collect::<Result<Vec<f64>, String>>()?;
        scores.push(
            r_squared(&pred, &ys)
                .ok_or_else(|| format!("{file}: inner validation targets have no variance"))?,
        );
    }
    Ok(scores)
}

/// Index of the highest score; ties go to the smaller index.
fn best_index(scores: &[f64]) -> usize {
    let mut best = 0;
    for (i, s) in scores.iter().enumerate() {
        if scores.get(best).is_none_or(|b| s > b) {
            best = i;
        }
    }
    best
}

fn select_lambda(fit: &[Obs], val: &[Obs], width: usize, file: &str) -> Result<f64, String> {
    let scores = inner_scores(fit, val, width, file)?;
    LAMBDA_GRID
        .get(best_index(&scores))
        .copied()
        .ok_or_else(|| format!("{file}: empty lambda grid"))
}

// ---------------------------------------------------------------- scoring

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / count_as_f64(u64::try_from(values.len()).ok()?))
    }
}

/// Mean of `pos` minus mean of `neg`; `None` when either is empty.
fn gap(pos: &[f64], neg: &[f64]) -> Option<f64> {
    Some(mean(pos)? - mean(neg)?)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerAgent {
    agent: u64,
    lambda: f64,
    held_out_r2: f64,
    in_view_gap: f64,
    centred_gap: f64,
}

fn obs_of<'a>(r: &'a Row, range: &std::ops::Range<usize>) -> Obs<'a> {
    Obs {
        x: r.features.get(range.clone()).unwrap_or(&[]),
        y: r.target,
    }
}

fn score_agent(data: &AgentData, set: Set) -> Result<PerAgent, String> {
    let file = data.file.as_str();
    let width = set.width();
    let range = set.range();
    let (held, train): (Vec<&Row>, Vec<&Row>) = data
        .rows
        .iter()
        .partition(|r| is_held_out(data.seed, data.agent, r.block));
    if held.is_empty() {
        return Err(format!("{file}: agent {} has no held-out rows", data.agent));
    }
    let (val, fit): (Vec<&Row>, Vec<&Row>) = train
        .iter()
        .partition(|r| r.block % INNER_VALIDATION_MODULUS == 0);
    if val.is_empty() || fit.is_empty() {
        return Err(format!(
            "{file}: agent {} has no inner validation or inner fit rows",
            data.agent
        ));
    }
    let fit_obs: Vec<Obs> = fit.iter().map(|r| obs_of(r, &range)).collect();
    let val_obs: Vec<Obs> = val.iter().map(|r| obs_of(r, &range)).collect();
    let lambda = select_lambda(&fit_obs, &val_obs, width, file)?;
    let train_obs: Vec<Obs> = train.iter().map(|r| obs_of(r, &range)).collect();
    let coefs = ridge_fit(&train_obs, width, lambda, file)?;
    let mut preds = Vec::new();
    let mut ys = Vec::new();
    let (mut in_view, mut others, mut centred, mut off) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in &held {
        let o = obs_of(r, &range);
        let p = predict(&coefs, o.x, file)?;
        preds.push(p);
        ys.push(r.target);
        if r.in_view {
            in_view.push(p);
        } else if r.alive {
            others.push(p);
        }
        if r.centred {
            centred.push(p);
        }
        if r.off_axis {
            off.push(p);
        }
    }
    let r2 = r_squared(&preds, &ys)
        .ok_or_else(|| format!("{file}: held-out targets have no variance"))?;
    let in_view_gap = gap(&in_view, &others)
        .ok_or_else(|| format!("{file}: agent {} has an empty view class", data.agent))?;
    let centred_gap = gap(&centred, &off)
        .ok_or_else(|| format!("{file}: agent {} has an empty bearing class", data.agent))?;
    Ok(PerAgent {
        agent: data.agent,
        lambda,
        held_out_r2: r2,
        in_view_gap,
        centred_gap,
    })
}

// ----------------------------------------------------------------- pooled

/// Mean and standard error (`sample sd / sqrt(n)`); `None` below two values.
fn spread(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < MIN_AGENTS {
        return None;
    }
    let n = count_as_f64(u64::try_from(values.len()).ok()?);
    let m = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
    Some((m, var.sqrt() / n.sqrt()))
}

fn positive_count(values: &[f64]) -> usize {
    values.iter().filter(|v| **v > 0.0).count()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetRecord {
    per_agent: Vec<PerAgent>,
    pooled_r2: f64,
    pooled_r2_se: f64,
    pooled_in_view_gap: f64,
    pooled_in_view_gap_se: f64,
    pooled_centred_gap: f64,
    pooled_centred_gap_se: f64,
    in_view_positive_agents: usize,
    centred_positive_agents: usize,
}

fn summarize_set(per_agent: Vec<PerAgent>, file: &str) -> Result<SetRecord, String> {
    if per_agent.len() < MIN_AGENTS {
        return Err(format!("{file}: fewer than {MIN_AGENTS} agents"));
    }
    let col = |f: fn(&PerAgent) -> f64| -> Result<Vec<f64>, String> {
        let v: Vec<f64> = per_agent.iter().map(f).collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("{file}: non-finite per-agent value"));
        }
        Ok(v)
    };
    let r2 = col(|a| a.held_out_r2)?;
    let iv = col(|a| a.in_view_gap)?;
    let cg = col(|a| a.centred_gap)?;
    col(|a| a.lambda)?;
    let missing = || format!("{file}: fewer than {MIN_AGENTS} agents");
    let (pooled_r2, pooled_r2_se) = spread(&r2).ok_or_else(missing)?;
    let (pooled_in_view_gap, pooled_in_view_gap_se) = spread(&iv).ok_or_else(missing)?;
    let (pooled_centred_gap, pooled_centred_gap_se) = spread(&cg).ok_or_else(missing)?;
    Ok(SetRecord {
        pooled_r2,
        pooled_r2_se,
        pooled_in_view_gap,
        pooled_in_view_gap_se,
        pooled_centred_gap,
        pooled_centred_gap_se,
        in_view_positive_agents: positive_count(&iv),
        centred_positive_agents: positive_count(&cg),
        per_agent,
    })
}

/// Winner from pooled R² and eligibility, ordered pathway, encoding,
/// encoding-and-pathway.
fn decide_winner(
    r2: [f64; 3],
    eligible: [bool; 3],
    advantage: f64,
    advantage_se: f64,
) -> CriticInput {
    const ORDER: [CriticInput; 3] = [
        CriticInput::Pathway,
        CriticInput::Encoding,
        CriticInput::EncodingAndPathway,
    ];
    let mut best: Option<usize> = None;
    for (i, ok) in eligible.iter().enumerate() {
        if *ok && best.is_none_or(|b| r2.get(i) > r2.get(b)) {
            best = Some(i);
        }
    }
    match best.and_then(|b| ORDER.get(b).copied().map(|w| (b, w))) {
        None => CriticInput::Encoding,
        Some((_, CriticInput::EncodingAndPathway)) => {
            if eligible.first().copied().unwrap_or(false) && advantage <= advantage_se {
                CriticInput::Pathway
            } else {
                CriticInput::EncodingAndPathway
            }
        }
        Some((_, w)) => w,
    }
}

// ----------------------------------------------------------------- record

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Constants {
    td_discount: f64,
    terminal_death_td_error: f64,
    truncation_ticks: usize,
    block_ticks: usize,
    holdout_modulus: u64,
    holdout_below: u64,
    inner_validation_modulus: u64,
    lambda_grid: [f64; 5],
    pathway_width: usize,
    encoding_width: usize,
    encoding_and_pathway_width: usize,
    splitmix_increment: u64,
    splitmix_mixer_a: u64,
    splitmix_mixer_b: u64,
    pathway_input_lag_rows: usize,
}

impl Constants {
    fn current() -> Self {
        Self {
            td_discount: TD_DISCOUNT,
            terminal_death_td_error: TERMINAL_DEATH_TD_ERROR,
            truncation_ticks: TRUNCATION_TICKS,
            block_ticks: BLOCK_TICKS,
            holdout_modulus: HOLDOUT_MODULUS,
            holdout_below: HOLDOUT_BELOW,
            inner_validation_modulus: INNER_VALIDATION_MODULUS,
            lambda_grid: LAMBDA_GRID,
            pathway_width: PATHWAY_WIDTH,
            encoding_width: ENCODING_WIDTH,
            encoding_and_pathway_width: ENCODING_AND_PATHWAY_WIDTH,
            splitmix_increment: SPLITMIX_INCREMENT,
            splitmix_mixer_a: SPLITMIX_MIXER_A,
            splitmix_mixer_b: SPLITMIX_MIXER_B,
            pathway_input_lag_rows: PATHWAY_INPUT_LAG_ROWS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentId {
    seed: u64,
    agent: u64,
    log_file: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sets {
    pathway: SetRecord,
    encoding: SetRecord,
    encoding_and_pathway: SetRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    schema_version: u32,
    constants: Constants,
    agents: Vec<AgentId>,
    sets: Sets,
    paired_r2_advantage_se: f64,
    winner: CriticInput,
}

/// Builds the record from per-agent scores; shared by `fit` and `verify`.
fn build_record(
    agents: Vec<AgentId>,
    per_agent: [Vec<PerAgent>; 3],
    file: &str,
) -> Result<Record, String> {
    let [pathway_scores, encoding_scores, both_scores] = per_agent;
    if pathway_scores.len() != both_scores.len() {
        return Err(format!("{file}: sets list different agent counts"));
    }
    let diffs: Vec<f64> = both_scores
        .iter()
        .zip(&pathway_scores)
        .map(|(b, p)| b.held_out_r2 - p.held_out_r2)
        .collect();
    let pathway = summarize_set(pathway_scores, file)?;
    let encoding = summarize_set(encoding_scores, file)?;
    let both = summarize_set(both_scores, file)?;
    let (advantage, advantage_se) =
        spread(&diffs).ok_or_else(|| format!("{file}: fewer than {MIN_AGENTS} agents"))?;
    let eligible = |s: &SetRecord| s.pooled_in_view_gap > 0.0 && s.pooled_centred_gap > 0.0;
    let winner = decide_winner(
        [pathway.pooled_r2, encoding.pooled_r2, both.pooled_r2],
        [eligible(&pathway), eligible(&encoding), eligible(&both)],
        advantage,
        advantage_se,
    );
    Ok(Record {
        schema_version: SCHEMA_VERSION,
        constants: Constants::current(),
        agents,
        sets: Sets {
            pathway,
            encoding,
            encoding_and_pathway: both,
        },
        paired_r2_advantage_se: advantage_se,
        winner,
    })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= VERIFY_RELATIVE_TOLERANCE * a.abs().max(b.abs())
}

fn check_real(field: &str, recorded: f64, derived: f64) -> Result<(), String> {
    if close(recorded, derived) {
        Ok(())
    } else {
        Err(format!("{field}: recorded {recorded}, derived {derived}"))
    }
}

fn check_count(field: &str, recorded: usize, derived: usize) -> Result<(), String> {
    if recorded == derived {
        Ok(())
    } else {
        Err(format!("{field}: recorded {recorded}, derived {derived}"))
    }
}

fn check_set(name: &str, rec: &SetRecord, der: &SetRecord) -> Result<(), String> {
    let reals = [
        ("pooled_r2", rec.pooled_r2, der.pooled_r2),
        ("pooled_r2_se", rec.pooled_r2_se, der.pooled_r2_se),
        (
            "pooled_in_view_gap",
            rec.pooled_in_view_gap,
            der.pooled_in_view_gap,
        ),
        (
            "pooled_in_view_gap_se",
            rec.pooled_in_view_gap_se,
            der.pooled_in_view_gap_se,
        ),
        (
            "pooled_centred_gap",
            rec.pooled_centred_gap,
            der.pooled_centred_gap,
        ),
        (
            "pooled_centred_gap_se",
            rec.pooled_centred_gap_se,
            der.pooled_centred_gap_se,
        ),
    ];
    for (field, a, b) in reals {
        check_real(&format!("sets.{name}.{field}"), a, b)?;
    }
    check_count(
        &format!("sets.{name}.in_view_positive_agents"),
        rec.in_view_positive_agents,
        der.in_view_positive_agents,
    )?;
    check_count(
        &format!("sets.{name}.centred_positive_agents"),
        rec.centred_positive_agents,
        der.centred_positive_agents,
    )
}

/// Re-derives everything from `per_agent` and compares.
fn verify_record(rec: &Record, file: &str) -> Result<(), String> {
    if rec.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "schema_version: recorded {}, derived {SCHEMA_VERSION}",
            rec.schema_version
        ));
    }
    let current = Constants::current();
    if rec.constants != current {
        return Err(format!(
            "constants: recorded {:?}, derived {current:?}",
            rec.constants
        ));
    }
    let sets = [
        &rec.sets.pathway,
        &rec.sets.encoding,
        &rec.sets.encoding_and_pathway,
    ];
    for set in sets {
        let ids_match = set.per_agent.len() == rec.agents.len()
            && set
                .per_agent
                .iter()
                .zip(&rec.agents)
                .all(|(p, a)| p.agent == a.agent);
        if !ids_match {
            return Err(format!("{file}: per_agent does not match agents"));
        }
    }
    let der = build_record(rec.agents.clone(), sets.map(|s| s.per_agent.clone()), file)?;
    check_set("pathway", &rec.sets.pathway, &der.sets.pathway)?;
    check_set("encoding", &rec.sets.encoding, &der.sets.encoding)?;
    check_set(
        "encoding_and_pathway",
        &rec.sets.encoding_and_pathway,
        &der.sets.encoding_and_pathway,
    )?;
    check_real(
        "paired_r2_advantage_se",
        rec.paired_r2_advantage_se,
        der.paired_r2_advantage_se,
    )?;
    if rec.winner != der.winner {
        return Err(format!(
            "winner: recorded {:?}, derived {:?}",
            rec.winner, der.winner
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- loading

/// Integer stored exactly in an `f32` column.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn exact_count(x: f32, what: &str, file: &str) -> Result<u64, String> {
    let wide = f64::from(x);
    if !wide.is_finite()
        || wide < 0.0
        || wide.fract() != 0.0
        || wide > count_as_f64(MAX_EXACT_F32_INTEGER)
    {
        return Err(format!("{file}: {what} {x} is not an exact count"));
    }
    Ok(wide as u64)
}

fn agents_from_log(log: &TickLog, seed: u64, file: &str) -> Result<Vec<AgentData>, String> {
    let width = log.columns.len();
    if width == 0 {
        return Err(format!("{file}: no columns"));
    }
    if let Some(i) = log.values.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "{file}: non-finite value at row {} column {}",
            i / width,
            i % width
        ));
    }
    let col = |name: &str| {
        log.column_index(name)
            .ok_or_else(|| format!("{file}: missing column {name}"))
    };
    let (c_agent, c_alive, c_deaths) = (col("agent")?, col("alive")?, col("deaths_total")?);
    let (c_dist, c_bear, c_reward) = (col("food_distance")?, col("food_bearing")?, col("reward")?);
    let enc_cols = (0..ENCODING_WIDTH)
        .map(|i| col(&format!("encoding_{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let path_cols = (0..PATHWAY_WIDTH)
        .map(|i| col(&format!("pathway_input_{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let rows: Vec<&[f32]> = log.values.chunks_exact(width).collect();
    let mut groups: Vec<(u64, Vec<usize>)> = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let id = exact_count(row.get(c_agent).copied().unwrap_or(f32::NAN), "agent", file)?;
        match groups.iter_mut().find(|(g, _)| *g == id) {
            Some((_, members)) => members.push(r),
            None => groups.push((id, vec![r])),
        }
    }
    groups.sort_by_key(|(id, _)| *id);
    let at = |row: &[f32], c: usize| row.get(c).copied().unwrap_or(0.0);
    let mut out = Vec::new();
    for (agent, members) in groups {
        let mut ticks = Vec::new();
        for (n, r) in members.iter().enumerate() {
            let row = rows.get(*r).copied().unwrap_or(&[]);
            let previous_deaths = n
                .checked_sub(1)
                .and_then(|p| members.get(p))
                .and_then(|pr| rows.get(*pr))
                .map(|prow| at(prow, c_deaths));
            let mut pathway = [0.0_f32; PATHWAY_WIDTH];
            for (dst, c) in pathway.iter_mut().zip(&path_cols) {
                *dst = at(row, *c);
            }
            ticks.push(Tick {
                reward: f64::from(at(row, c_reward)),
                died: previous_deaths.is_some_and(|d| at(row, c_deaths) > d),
                alive: at(row, c_alive) > 0.0,
                food_distance: f64::from(at(row, c_dist)),
                food_bearing: f64::from(at(row, c_bear)),
                pathway,
                encoding: enc_cols.iter().map(|c| at(row, *c)).collect(),
            });
        }
        out.push(AgentData {
            seed,
            agent,
            file: file.to_string(),
            rows: build_rows(&ticks),
        });
    }
    Ok(out)
}

fn file_label(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn load_agents(path: &Path, seed: u64) -> Result<Vec<AgentData>, String> {
    let label = file_label(path);
    let log = read_tick_log(path).map_err(|e| format!("{label}: {e}"))?;
    agents_from_log(&log, seed, &label)
}

fn fit_agents(data: &[AgentData], file: &str) -> Result<Record, String> {
    if data.len() < MIN_AGENTS {
        return Err(format!("{file}: fewer than {MIN_AGENTS} agents"));
    }
    let mut scores: [Vec<PerAgent>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for agent in data {
        for (set, slot) in Set::ALL.into_iter().zip(scores.iter_mut()) {
            slot.push(score_agent(agent, set)?);
        }
    }
    let agents = data
        .iter()
        .map(|a| AgentId {
            seed: a.seed,
            agent: a.agent,
            log_file: a.file.clone(),
        })
        .collect();
    build_record(agents, scores, file)
}

fn run_fit(logs: &[PathBuf], out: &Path) -> Result<(), String> {
    let mut data = Vec::new();
    for (i, path) in logs.iter().enumerate() {
        let seed = u64::try_from(i).map_err(|e| e.to_string())?;
        data.extend(load_agents(path, seed)?);
    }
    let label = file_label(out);
    let record = fit_agents(&data, &label)?;
    let text = serde_json::to_string_pretty(&record).map_err(|e| format!("{label}: {e}"))?;
    fs::write(out, text).map_err(|e| format!("{label}: {e}"))
}

fn run_verify(path: &Path) -> Result<(), String> {
    let label = file_label(path);
    let text = fs::read_to_string(path).map_err(|e| format!("{label}: {e}"))?;
    let record: Record = serde_json::from_str(&text).map_err(|e| format!("{label}: {e}"))?;
    verify_record(&record, &label)
}

#[derive(Parser)]
#[command(about = "Offline return fit for critic input sets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit the three input sets and write the record.
    Fit {
        /// Per-brain-tick logs.
        #[arg(long, num_args = 1.., required = true)]
        log: Vec<PathBuf>,
        /// Output record.
        #[arg(long)]
        out: PathBuf,
    },
    /// Re-derive a record and fail on any mismatch.
    Verify {
        /// Record to check.
        path: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Fit { log, out } => run_fit(log, out),
        Command::Verify { path } => run_verify(path),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xagent_sandbox::free_life::metrics::{FreeLifeSample, PATHWAY_INPUTS};
    use xagent_sandbox::free_life::tick_log::TickLogWriter;

    const FIRST_SPLITMIX_OUTPUT: u64 = 0xE220_A839_7B1D_CDAF;
    const SECOND_SPLITMIX_OUTPUT: u64 = 0x6E78_9E6A_A1B9_65F4;
    const SHARE_BLOCKS: u64 = 10_000;
    const SHARE_TOLERANCE: f64 = 0.02;
    const RIDGE_ROWS: usize = 200;
    const COEFFICIENT_TOLERANCE: f64 = 1e-3;
    const EXACT_R2_FLOOR: f64 = 0.999;
    const NOISE_R2_CEILING: f64 = 0.05;
    const SMALL_LAMBDA: f64 = 1e-3;
    const INTERCEPT: f64 = 2.0;
    const SLOPE_ONE: f64 = 3.0;
    const SLOPE_TWO: f64 = -1.0;
    const TEST_FILE: &str = "case.bin";
    const SCALE_BITS: u32 = 48;
    /// Seed of the exact-data generator.
    const EXACT_SEED: u64 = 7;
    /// Seed of the hold-out noise generator.
    const NOISE_SEED: u64 = 99;
    /// Offset of the encoding value from the pathway marker.
    const ENCODING_OFFSET: f32 = 0.5;
    /// Second reward of the return fixtures.
    const REWARD_B: f64 = 2.0;
    /// Third reward of the return fixtures.
    const REWARD_C: f64 = 3.0;
    /// Plateau reward of the second-life fixture.
    const PLATEAU_REWARD: f64 = 5.0;
    /// Reward of the second life, which must not leak back.
    const LARGE_REWARD: f64 = 100.0;
    /// Ticks beyond the truncation window.
    const EXTRA_TICKS: usize = 2;
    /// Number of feature columns in the ridge fixtures.
    const FEATURE_COUNT: usize = 2;
    /// Number of input sets.
    const SET_COUNT: usize = 3;
    /// Period of the duplicate-column feature.
    const DUPLICATE_PERIOD: usize = 50;
    /// Divisor that halves a count.
    const HALF_DIVISOR: usize = 2;
    /// Block count of the complete-block fixture.
    const TWO_BLOCKS: usize = 2;
    /// Highest score in the tie fixtures.
    const SCORE_HIGH: f64 = 0.5;
    /// Middle score in the tie fixtures.
    const SCORE_MID: f64 = 0.4;
    /// Lowest score in the tie fixtures.
    const SCORE_LOW: f64 = 0.1;
    /// Pathway marker of the second alignment row.
    const MARKER_SECOND: f32 = 2.0;
    /// Pathway marker of the third alignment row.
    const MARKER_THIRD: f32 = 3.0;
    /// Pathway marker of the fourth alignment row.
    const MARKER_FOURTH: f32 = 4.0;
    /// Pathway marker of the fifth alignment row.
    const MARKER_FIFTH: f32 = 5.0;
    /// Pathway marker of the filler alignment rows.
    const MARKER_FILL: f32 = 6.0;
    /// Rows inspected in the alignment test.
    const INSPECTED_ROWS: usize = 3;
    /// Second kept tick of the alignment fixture.
    const SECOND_KEPT_TICK: usize = 2;
    /// Third kept tick of the alignment fixture.
    const THIRD_KEPT_TICK: usize = 4;
    /// First tick of the second life in the alignment fixture.
    const LIFE_START_TICK: usize = 3;
    /// Sample whose mean and standard error are known.
    const SPREAD_SAMPLE: &[f64] = &[1.0, 2.0, 3.0, 4.0];
    /// Mean of `SPREAD_SAMPLE`.
    const SPREAD_MEAN: f64 = 2.5;
    /// Numerator of the sample variance of `SPREAD_SAMPLE`.
    const VARIANCE_NUMERATOR: f64 = 5.0;
    /// Denominator of the sample variance of `SPREAD_SAMPLE`.
    const VARIANCE_DENOMINATOR: f64 = 3.0;
    /// Square root of the sample size of `SPREAD_SAMPLE`.
    const SQRT_SAMPLE_SIZE: f64 = 2.0;
    /// Tolerance of the spread comparison.
    const SPREAD_TOLERANCE: f64 = 1e-12;
    /// Sample with a known positive count.
    const POSITIVE_SAMPLE: &[f64] = &[1.0, -1.0, 0.0, 2.0];
    /// Positive count of `POSITIVE_SAMPLE`.
    const POSITIVE_EXPECTED: usize = 2;
    /// Pooled R2 values of the winner fixtures.
    const WINNER_R2: [f64; SET_COUNT] = [0.2, 0.3, 0.5];
    /// R2 values where encoding alone is best.
    const ENCODING_BEST_R2: [f64; SET_COUNT] = [0.2, 0.6, 0.5];
    /// Standard error of the paired advantage.
    const ADVANTAGE_SE: f64 = 0.1;
    /// Advantage above `ADVANTAGE_SE`.
    const ADVANTAGE_ABOVE: f64 = 0.2;
    /// Agents in the record fixture.
    const AGENT_COUNT: usize = 3;
    /// Pathway R2 per agent.
    const PATHWAY_R2: [f64; AGENT_COUNT] = [0.1, 0.2, 0.15];
    /// Encoding R2 per agent.
    const ENCODING_R2: [f64; AGENT_COUNT] = [0.3, 0.1, 0.2];
    /// Combined R2 per agent.
    const BOTH_R2: [f64; AGENT_COUNT] = [0.5, 0.45, 0.6];
    /// Gap of the encoding fixture (kept positive).
    const ENCODING_GAP: f64 = 0.5;
    /// Perturbation of a pooled R2 beyond tolerance.
    const R2_PERTURBATION: f64 = 1e-6;
    /// Scale applied to an edited standard error.
    const SE_SCALE: f64 = 2.0;
    /// Agents in the malformed-log fixtures.
    const LOG_AGENTS: u32 = 2;
    /// Ticks in the malformed-log fixtures.
    const LOG_TICKS: u64 = 20;
    /// A row width that is not a valid set width.
    const WRONG_ROW_WIDTH: usize = 9;

    fn unit(state: &mut u64) -> f32 {
        let bits = u16::try_from(splitmix_next(state) >> SCALE_BITS).unwrap();
        f32::from(bits) / (f32::from(u16::MAX) + 1.0)
    }

    fn exact_data(noise: bool) -> (Vec<[f32; FEATURE_COUNT]>, Vec<f64>) {
        let mut state = EXACT_SEED;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..RIDGE_ROWS {
            let x = [unit(&mut state), unit(&mut state)];
            ys.push(if noise {
                f64::from(unit(&mut state))
            } else {
                INTERCEPT + SLOPE_ONE * f64::from(x[0]) + SLOPE_TWO * f64::from(x[1])
            });
            xs.push(x);
        }
        (xs, ys)
    }

    fn obs_of<'a>(xs: &'a [[f32; FEATURE_COUNT]], ys: &[f64]) -> Vec<Obs<'a>> {
        xs.iter().zip(ys).map(|(x, y)| Obs { x, y: *y }).collect()
    }

    fn tick(reward: f64, died: bool, marker: f32) -> Tick {
        Tick {
            reward,
            died,
            alive: true,
            food_distance: 1.0,
            food_bearing: 0.0,
            pathway: [marker; PATHWAY_WIDTH],
            encoding: vec![marker + ENCODING_OFFSET; ENCODING_WIDTH],
        }
    }

    // Returns ----------------------------------------------------------

    #[test]
    fn returns_discount_backward_without_death() {
        let g = returns(&[1.0, REWARD_B, REWARD_C], &[false, false, false]);
        assert_eq!(g[2], REWARD_C);
        assert_eq!(g[1], REWARD_B + TD_DISCOUNT * REWARD_C);
        assert_eq!(g[0], 1.0 + TD_DISCOUNT * g[1]);
    }

    #[test]
    fn death_on_last_tick_uses_terminal_value() {
        let g = returns(&[1.0, REWARD_B, REWARD_C], &[false, false, true]);
        assert_eq!(g[2], TERMINAL_DEATH_TD_ERROR);
    }

    #[test]
    fn second_life_does_not_leak_into_first() {
        let g = returns(
            &[PLATEAU_REWARD, PLATEAU_REWARD, LARGE_REWARD],
            &[false, true, false],
        );
        assert_eq!(g[1], TERMINAL_DEATH_TD_ERROR);
        assert_eq!(g[0], PLATEAU_REWARD + TD_DISCOUNT * TERMINAL_DEATH_TD_ERROR);
    }

    #[test]
    fn truncation_drops_the_final_ticks() {
        let n = TRUNCATION_TICKS + EXTRA_TICKS;
        assert_eq!(
            truncated_returns(&vec![1.0; n], &vec![false; n]).len(),
            EXTRA_TICKS
        );
    }

    // Splitmix ---------------------------------------------------------

    #[test]
    fn splitmix_matches_reference_outputs() {
        let mut state = 0;
        assert_eq!(splitmix_next(&mut state), FIRST_SPLITMIX_OUTPUT);
        assert_eq!(splitmix_next(&mut state), SECOND_SPLITMIX_OUTPUT);
    }

    #[test]
    fn held_out_share_matches_the_modulus_ratio() {
        let held = (0..SHARE_BLOCKS).filter(|b| is_held_out(1, 0, *b)).count();
        let share = count_as_f64(u64::try_from(held).unwrap()) / count_as_f64(SHARE_BLOCKS);
        let target = count_as_f64(HOLDOUT_BELOW) / count_as_f64(HOLDOUT_MODULUS);
        assert!((share - target).abs() < SHARE_TOLERANCE);
    }

    #[test]
    fn changing_the_agent_changes_the_hold_out_pattern() {
        let a: Vec<bool> = (0..SHARE_BLOCKS).map(|b| is_held_out(1, 0, b)).collect();
        let b: Vec<bool> = (0..SHARE_BLOCKS).map(|b| is_held_out(1, 1, b)).collect();
        assert_ne!(a, b);
    }

    // Ridge ------------------------------------------------------------

    #[test]
    fn ridge_recovers_known_coefficients() {
        let (xs, ys) = exact_data(false);
        let coefs = ridge_fit(&obs_of(&xs, &ys), FEATURE_COUNT, SMALL_LAMBDA, TEST_FILE).unwrap();
        for (got, want) in coefs.iter().zip([INTERCEPT, SLOPE_ONE, SLOPE_TWO]) {
            assert!((got - want).abs() < COEFFICIENT_TOLERANCE);
        }
    }

    #[test]
    fn ridge_scores_exact_data_high_and_noise_low() {
        for (noise, exact) in [(false, true), (true, false)] {
            let (xs, ys) = exact_data(noise);
            let coefs =
                ridge_fit(&obs_of(&xs, &ys), FEATURE_COUNT, SMALL_LAMBDA, TEST_FILE).unwrap();
            let (hx, _) = exact_data(false);
            let mut state = NOISE_SEED;
            let hy: Vec<f64> = hx
                .iter()
                .map(|x| {
                    if exact {
                        INTERCEPT + SLOPE_ONE * f64::from(x[0]) + SLOPE_TWO * f64::from(x[1])
                    } else {
                        f64::from(unit(&mut state))
                    }
                })
                .collect();
            let pred: Vec<f64> = hx
                .iter()
                .map(|x| predict(&coefs, x, TEST_FILE).unwrap())
                .collect();
            let r2 = r_squared(&pred, &hy).unwrap();
            if exact {
                assert!(r2 > EXACT_R2_FLOOR);
            } else {
                assert!(r2 < NOISE_R2_CEILING);
            }
        }
    }

    #[test]
    fn duplicate_column_without_penalty_errors_naming_the_file() {
        let xs: Vec<[f32; FEATURE_COUNT]> = (0..RIDGE_ROWS)
            .map(|i| {
                let v = f32::from(u8::try_from(i % DUPLICATE_PERIOD).unwrap());
                [v, v]
            })
            .collect();
        let ys: Vec<f64> = xs.iter().map(|x| f64::from(x[0])).collect();
        let err = ridge_fit(&obs_of(&xs, &ys), FEATURE_COUNT, 0.0, TEST_FILE).unwrap_err();
        assert!(err.contains(TEST_FILE));
    }

    // Split ------------------------------------------------------------

    #[test]
    fn trailing_partial_block_is_dropped() {
        let kept = BLOCK_TICKS + BLOCK_TICKS / HALF_DIVISOR;
        assert!(block_is_complete(0, kept));
        assert!(!block_is_complete(BLOCK_TICKS, kept));
        assert!(block_is_complete(BLOCK_TICKS, TWO_BLOCKS * BLOCK_TICKS));
    }

    #[test]
    fn ties_choose_the_smaller_lambda_index() {
        assert_eq!(
            best_index(&[SCORE_HIGH, SCORE_HIGH, SCORE_MID, SCORE_HIGH, SCORE_LOW]),
            0
        );
        assert_eq!(
            best_index(&[SCORE_LOW, SCORE_HIGH, SCORE_MID, SCORE_HIGH, SCORE_LOW]),
            1
        );
    }

    #[test]
    fn chosen_lambda_has_the_highest_inner_validation_score() {
        let (xs, ys) = exact_data(true);
        let half = RIDGE_ROWS / HALF_DIVISOR;
        let fit = obs_of(xs.get(..half).unwrap(), ys.get(..half).unwrap());
        let val = obs_of(xs.get(half..).unwrap(), ys.get(half..).unwrap());
        let scores = inner_scores(&fit, &val, FEATURE_COUNT, TEST_FILE).unwrap();
        let chosen = select_lambda(&fit, &val, FEATURE_COUNT, TEST_FILE).unwrap();
        let index = LAMBDA_GRID.iter().position(|l| *l == chosen).unwrap();
        assert!(scores.iter().all(|s| *s <= scores[index]));
    }

    // Alignment --------------------------------------------------------

    #[test]
    fn pathway_columns_lag_one_row_and_drop_life_starts() {
        let mut ticks = vec![
            tick(0.0, false, 1.0),
            tick(0.0, false, MARKER_SECOND),
            tick(0.0, true, MARKER_THIRD),
            tick(0.0, false, MARKER_FOURTH),
            tick(0.0, false, MARKER_FIFTH),
        ];
        let total = BLOCK_TICKS + TRUNCATION_TICKS;
        while ticks.len() < total {
            ticks.push(tick(0.0, false, MARKER_FILL));
        }
        let rows = build_rows(&ticks);
        let kept: Vec<usize> = rows.iter().take(INSPECTED_ROWS).map(|r| r.tick).collect();
        assert_eq!(kept, vec![1, SECOND_KEPT_TICK, THIRD_KEPT_TICK]);
        for row in rows.iter().take(INSPECTED_ROWS) {
            let prev = &ticks[row.tick - PATHWAY_INPUT_LAG_ROWS];
            let own = &ticks[row.tick];
            assert_eq!(row.features[..PATHWAY_WIDTH], prev.pathway[..]);
            assert_eq!(row.features[PATHWAY_WIDTH..], own.encoding[..]);
        }
        assert!(rows
            .iter()
            .all(|r| r.tick != 0 && r.tick != LIFE_START_TICK));
    }

    // Gaps and spread --------------------------------------------------

    #[test]
    fn gap_is_positive_when_in_view_scores_higher() {
        assert_eq!(gap(&[1.0, 1.0], &[0.0, 0.0]), Some(1.0));
        assert_eq!(gap(&[0.0], &[1.0]), Some(-1.0));
        assert_eq!(gap(&[], &[1.0]), None);
    }

    #[test]
    fn spread_and_positive_counts_follow_the_definition() {
        let (m, se) = spread(SPREAD_SAMPLE).unwrap();
        assert_eq!(m, SPREAD_MEAN);
        let expected = (VARIANCE_NUMERATOR / VARIANCE_DENOMINATOR).sqrt() / SQRT_SAMPLE_SIZE;
        assert!((se - expected).abs() < SPREAD_TOLERANCE);
        assert_eq!(positive_count(POSITIVE_SAMPLE), POSITIVE_EXPECTED);
        assert!(spread(&[1.0]).is_none());
    }

    // Winner and verify ------------------------------------------------

    #[test]
    fn winner_rule_follows_gaps_and_paired_advantage() {
        let r2 = WINNER_R2;
        assert_eq!(
            decide_winner(r2, [false; SET_COUNT], 1.0, ADVANTAGE_SE),
            CriticInput::Encoding
        );
        assert_eq!(
            decide_winner(r2, [true; SET_COUNT], ADVANTAGE_SE, ADVANTAGE_SE),
            CriticInput::Pathway
        );
        assert_eq!(
            decide_winner(r2, [true; SET_COUNT], ADVANTAGE_ABOVE, ADVANTAGE_SE),
            CriticInput::EncodingAndPathway
        );
        assert_eq!(
            decide_winner(ENCODING_BEST_R2, [true; SET_COUNT], 0.0, ADVANTAGE_SE),
            CriticInput::Encoding
        );
    }

    fn scores(r2: [f64; AGENT_COUNT], gap: f64) -> Vec<PerAgent> {
        r2.iter()
            .enumerate()
            .map(|(i, r)| PerAgent {
                agent: u64::try_from(i).unwrap(),
                lambda: 1.0,
                held_out_r2: *r,
                in_view_gap: gap,
                centred_gap: gap,
            })
            .collect()
    }

    fn sample_record() -> Record {
        let agents = (0..AGENT_COUNT)
            .map(|a| AgentId {
                seed: 0,
                agent: u64::try_from(a).unwrap(),
                log_file: "log.bin".to_string(),
            })
            .collect();
        build_record(
            agents,
            [
                scores(PATHWAY_R2, 1.0),
                scores(ENCODING_R2, ENCODING_GAP),
                scores(BOTH_R2, 1.0),
            ],
            "rec.json",
        )
        .unwrap()
    }

    #[test]
    fn verify_accepts_an_unchanged_record() {
        let rec = sample_record();
        verify_record(&rec, "rec.json").unwrap();
        let text = serde_json::to_string(&rec).unwrap();
        let back: Record = serde_json::from_str(&text).unwrap();
        verify_record(&back, "rec.json").unwrap();
    }

    #[test]
    fn verify_rejects_edited_fields() {
        let mut rec = sample_record();
        rec.winner = CriticInput::Encoding;
        assert!(verify_record(&rec, "rec.json")
            .unwrap_err()
            .contains("winner"));

        let mut rec = sample_record();
        rec.sets.pathway.pooled_r2 += R2_PERTURBATION;
        assert!(verify_record(&rec, "rec.json").is_err());

        let mut rec = sample_record();
        rec.sets.encoding.pooled_r2_se *= SE_SCALE;
        assert!(verify_record(&rec, "rec.json").is_err());

        let mut rec = sample_record();
        rec.sets.encoding.in_view_positive_agents += 1;
        assert!(verify_record(&rec, "rec.json").is_err());

        let mut rec = sample_record();
        rec.paired_r2_advantage_se += 1.0;
        assert!(verify_record(&rec, "rec.json").is_err());
    }

    #[test]
    fn verify_rejects_a_missing_paired_se() {
        let rec = sample_record();
        let mut value = serde_json::to_value(&rec).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("paired_r2_advantage_se");
        assert!(serde_json::from_value::<Record>(value).is_err());
    }

    #[test]
    fn unknown_set_names_are_rejected() {
        let rec = sample_record();
        let mut value = serde_json::to_value(&rec).unwrap();
        value["sets"]["extra"] = serde_json::json!({});
        assert!(serde_json::from_value::<Record>(value).is_err());
    }

    // Malformed input --------------------------------------------------

    fn sample(tick: u64, agent_marker: f32, reward: f32) -> FreeLifeSample {
        FreeLifeSample {
            brain_tick: tick,
            alive: true,
            meals_total: 0,
            deaths_total: 0,
            on_hazard: false,
            motor_turn: 0.0,
            food_distance: 1.0,
            food_bearing: 0.0,
            hazard_distance: 1.0,
            hazard_bearing: 0.0,
            encoding: vec![agent_marker; ENCODING_WIDTH],
            pathway_inputs: [agent_marker; PATHWAY_INPUTS],
            reward,
            value: 0.0,
            td_error: 0.0,
            forward_noise: 0.0,
            turn_noise: 0.0,
            turn_innovation: 0.0,
            encoding_turn_output: 0.0,
            scent_turn_output: 0.0,
            vision_turn_output: 0.0,
            encoding_forward_output: 0.0,
        }
    }

    fn write_log(dir: &Path, name: &str, agents: u32, ticks: u64, reward: f32) -> PathBuf {
        let path = dir.join(name);
        let mut w = TickLogWriter::create(&path, ticks * u64::from(agents)).unwrap();
        for t in 0..ticks {
            for a in 0..agents {
                w.write_sample(a, &sample(t, 1.0, reward)).unwrap();
            }
        }
        w.finish().unwrap();
        path
    }

    #[test]
    fn nan_in_a_log_errors_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_log(dir.path(), "nan.bin", LOG_AGENTS, LOG_TICKS, f32::NAN);
        let err = load_agents(&path, 0).map(|_| ()).unwrap_err();
        assert!(err.contains("nan.bin"));
    }

    #[test]
    fn wrong_row_width_errors_naming_the_file() {
        let x = [0.0_f32; WRONG_ROW_WIDTH];
        let obs = [Obs { x: &x, y: 0.0 }];
        let err = ridge_fit(&obs, PATHWAY_WIDTH, 1.0, TEST_FILE).unwrap_err();
        assert!(err.contains(TEST_FILE));
    }

    #[test]
    fn too_few_ticks_for_a_held_out_set_errors_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_log(dir.path(), "short.bin", LOG_AGENTS, LOG_TICKS, 0.0);
        let data = load_agents(&path, 0).unwrap();
        let err = fit_agents(&data, "short.bin").map(|_| ()).unwrap_err();
        assert!(err.contains("short.bin"));
    }

    #[test]
    fn a_single_agent_errors_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_log(dir.path(), "one.bin", 1, LOG_TICKS, 0.0);
        let data = load_agents(&path, 0).unwrap();
        let err = fit_agents(&data, "one.bin").map(|_| ()).unwrap_err();
        assert!(err.contains("one.bin"));
    }
}
