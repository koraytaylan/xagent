//! Offline fit library for comparing visual pooling schemes.
//!
//! A log holds per-tick column means of the vision field together with the
//! reward, value and turn bookkeeping of a free-life run. Four input arms feed
//! the same whitened, credit-trained turn rule and are scored by how well the
//! turn output tracks the side of the food.
//!
//! Log format (little-endian): the bytes `XPOL`, a `u32` column count, a `u32`
//! harness field count, then records of 40 `f32` values each: 32 column means
//! (column `c`, channel `k` at `c * 4 + k`, channels red, green, blue, depth),
//! then reward, value, turn innovation, food bearing, food in view (0 or 1),
//! death (0 or 1), agent and tick.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate
)]

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use xagent_brain::senses::VISION_WHITENING_REFRESH;

/// Magic bytes opening every log.
pub const LOG_MAGIC: [u8; 4] = *b"XPOL";
/// Number of column-mean values in a record (8 ray columns x 4 channels).
pub const COLUMN_COUNT: usize = 32;
/// Number of bookkeeping values following the column means in a record.
pub const HARNESS_FIELD_COUNT: usize = 8;
/// Channels per ray column: red, green, blue, depth.
pub const CHANNEL_COUNT: usize = 4;
/// Ray columns in the field.
pub const RAY_COLUMNS: usize = 8;
/// Ray columns in each hemifield.
const HEMIFIELD_COLUMNS: usize = 4;
/// Averaging weight of one column inside a hemifield (1 / `HEMIFIELD_COLUMNS`).
const HEMIFIELD_WEIGHT: f32 = 0.25;
/// Inputs and pooled components of arms A, C and D (2 sides x 4 channels).
pub const POOLED_COUNT: usize = 8;
/// Bytes in the log header: magic plus two `u32` counts.
const HEADER_BYTES: usize = 12;
/// Byte offset of the column count in the header.
const COLUMN_COUNT_OFFSET: usize = 4;
/// Byte offset of the harness field count in the header.
const FIELD_COUNT_OFFSET: usize = 8;
/// Bytes in one stored `f32`.
const FLOAT_BYTES: usize = 4;
/// Values in one record.
pub const VALUES_PER_RECORD: usize = COLUMN_COUNT + HARNESS_FIELD_COUNT;
/// Bytes in one record.
const RECORD_BYTES: usize = VALUES_PER_RECORD * FLOAT_BYTES;
/// Number of fitted arms (A, B, C, D).
pub const ARM_COUNT: usize = 4;

/// Rate of the running mean and covariance used for centring and whitening.
const ADAPTATION_RATE: f32 = 0.01;
/// Regulariser added to eigenvalues before the inverse square root.
const WHITENING_EPSILON: f32 = 1.0e-3;
/// Maximum Jacobi sweeps per eigendecomposition.
const JACOBI_SWEEPS: usize = 12;
/// Off-diagonal magnitude below which Jacobi skips a rotation.
const JACOBI_TOLERANCE: f32 = 1.0e-9;
/// Temporal-difference discount, mirroring `TD_DISCOUNT` in the kernel.
const TD_DISCOUNT: f32 = 0.97;
/// Trace decay, mirroring `TD_LAMBDA` in the kernel.
const TD_LAMBDA: f32 = 0.9;
/// Turn weight learning rate, mirroring `ACTION_WEIGHT_LEARNING_RATE`.
const ACTION_WEIGHT_LEARNING_RATE: f32 = 0.10;
/// Learning rate of the credit-trained pooling matrix of arm C.
const POOLING_ETA: f32 = ACTION_WEIGHT_LEARNING_RATE;
/// Oja/Sanger step of arm D.
const OJA_RATE: f32 = 0.01;
/// Length of the free-life run in ticks.
const RUN_TICKS: f32 = 300_000.0;
/// Fraction of the run before which records are not scored.
const SCORE_START_FRACTION: f32 = 0.7;
/// Minimum scored records for an agent to be scored.
const MIN_SCORED_RECORDS: usize = 30;
/// Sanity reference: arm B mean score from the earlier offline investigation.
pub const SANITY_ARM_B: f64 = 0.16;
/// Sanity reference: arm A mean score from the earlier offline investigation.
pub const SANITY_ARM_A: f64 = 0.33;

/// One log record.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// Column means, column `c` channel `k` at `c * 4 + k`.
    pub columns: [f32; COLUMN_COUNT],
    /// Reward of the tick.
    pub reward: f32,
    /// Critic value at the tick.
    pub value: f32,
    /// Turn innovation (noise injected into the turn).
    pub turn_innovation: f32,
    /// Signed bearing of the nearest food.
    pub food_bearing: f32,
    /// 1 when food is in view.
    pub food_in_view: f32,
    /// 1 when the agent died this tick.
    pub death: f32,
    /// Agent index.
    pub agent: f32,
    /// Tick number.
    pub tick: f32,
}

impl Record {
    fn to_values(&self) -> [f32; VALUES_PER_RECORD] {
        let mut values = [0.0; VALUES_PER_RECORD];
        values[..COLUMN_COUNT].copy_from_slice(&self.columns);
        values[COLUMN_COUNT..].copy_from_slice(&[
            self.reward,
            self.value,
            self.turn_innovation,
            self.food_bearing,
            self.food_in_view,
            self.death,
            self.agent,
            self.tick,
        ]);
        values
    }

    fn from_values(values: &[f32; VALUES_PER_RECORD]) -> Self {
        let mut columns = [0.0; COLUMN_COUNT];
        columns.copy_from_slice(&values[..COLUMN_COUNT]);
        let [reward, value, turn_innovation, food_bearing, food_in_view, death, agent, tick] =
            std::array::from_fn(|i| values[COLUMN_COUNT + i]);
        Self {
            columns,
            reward,
            value,
            turn_innovation,
            food_bearing,
            food_in_view,
            death,
            agent,
            tick,
        }
    }
}

/// Error reading or writing a log; always names the file.
#[derive(Debug)]
pub struct PoolingError {
    /// File the error concerns.
    pub path: PathBuf,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for PoolingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

impl std::error::Error for PoolingError {}

fn error(path: &Path, message: impl Into<String>) -> PoolingError {
    PoolingError {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn read_f32(chunk: &[u8]) -> f32 {
    let [b0, b1, b2, b3] = *chunk else {
        return 0.0;
    };
    f32::from_le_bytes([b0, b1, b2, b3])
}

fn read_u32(chunk: &[u8]) -> u32 {
    let [b0, b1, b2, b3] = *chunk else {
        return 0;
    };
    u32::from_le_bytes([b0, b1, b2, b3])
}

/// Writes `records` to `path` in the log format.
pub fn write_log(path: &Path, records: &[Record]) -> Result<(), PoolingError> {
    let column_count =
        u32::try_from(COLUMN_COUNT).map_err(|_| error(path, "column count overflow"))?;
    let field_count =
        u32::try_from(HARNESS_FIELD_COUNT).map_err(|_| error(path, "field count overflow"))?;
    let mut bytes = Vec::with_capacity(HEADER_BYTES + records.len() * RECORD_BYTES);
    bytes.extend_from_slice(&LOG_MAGIC);
    bytes.extend_from_slice(&column_count.to_le_bytes());
    bytes.extend_from_slice(&field_count.to_le_bytes());
    for record in records {
        for value in record.to_values() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).map_err(|err| error(path, format!("write failed: {err}")))
}

/// Reads and validates a log: magic, header counts, whole records, finite values.
pub fn read_log(path: &Path) -> Result<Vec<Record>, PoolingError> {
    let bytes = std::fs::read(path).map_err(|err| error(path, format!("read failed: {err}")))?;
    if bytes.len() < HEADER_BYTES {
        return Err(error(path, "file shorter than the header"));
    }
    if bytes[..LOG_MAGIC.len()] != LOG_MAGIC {
        return Err(error(path, "bad magic"));
    }
    let columns = usize::try_from(read_u32(&bytes[COLUMN_COUNT_OFFSET..FIELD_COUNT_OFFSET]))
        .unwrap_or(usize::MAX);
    let fields =
        usize::try_from(read_u32(&bytes[FIELD_COUNT_OFFSET..HEADER_BYTES])).unwrap_or(usize::MAX);
    if columns != COLUMN_COUNT || fields != HARNESS_FIELD_COUNT {
        return Err(error(path, "unexpected column or harness field count"));
    }
    let body = &bytes[HEADER_BYTES..];
    if body.len() % RECORD_BYTES != 0 {
        return Err(error(path, "length is not header plus whole records"));
    }
    let mut records = Vec::with_capacity(body.len() / RECORD_BYTES);
    for chunk in body.as_chunks::<RECORD_BYTES>().0 {
        let mut values = [0.0; VALUES_PER_RECORD];
        for (slot, raw) in values.iter_mut().zip(chunk.as_chunks::<FLOAT_BYTES>().0) {
            *slot = read_f32(raw);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(error(path, "record contains a non-finite value"));
        }
        records.push(Record::from_values(&values));
    }
    Ok(records)
}

/// Mean adapted hemifield input: left then right, channel by channel.
fn hemifield(columns: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; POOLED_COUNT];
    for side in 0..POOLED_COUNT / CHANNEL_COUNT {
        for channel in 0..CHANNEL_COUNT {
            let mut sum = 0.0;
            for column in 0..HEMIFIELD_COLUMNS {
                sum += columns[(side * HEMIFIELD_COLUMNS + column) * CHANNEL_COUNT + channel];
            }
            out[side * CHANNEL_COUNT + channel] = sum * HEMIFIELD_WEIGHT;
        }
    }
    out
}

/// Symmetric eigendecomposition by cyclic Jacobi; returns (eigenvalues, eigenvectors as columns).
fn jacobi(matrix: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }
    for _ in 0..JACOBI_SWEEPS {
        let mut rotated = false;
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = a[p * dim + q];
                if apq.abs() < JACOBI_TOLERANCE {
                    continue;
                }
                rotated = true;
                let tau = (a[q * dim + q] - a[p * dim + p]) / (apq + apq);
                let sign = if tau < 0.0 { -1.0 } else { 1.0 };
                let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..dim {
                    let (kp, kq) = (a[k * dim + p], a[k * dim + q]);
                    a[k * dim + p] = c * kp - s * kq;
                    a[k * dim + q] = s * kp + c * kq;
                    let (vp, vq) = (v[k * dim + p], v[k * dim + q]);
                    v[k * dim + p] = c * vp - s * vq;
                    v[k * dim + q] = s * vp + c * vq;
                }
                for k in 0..dim {
                    let (pk, qk) = (a[p * dim + k], a[q * dim + k]);
                    a[p * dim + k] = c * pk - s * qk;
                    a[q * dim + k] = s * pk + c * qk;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let values = (0..dim).map(|i| a[i * dim + i]).collect();
    (values, v)
}

/// Running-mean centring, running covariance, ZCA whitening and the credit-trained turn weights.
struct Learner {
    dim: usize,
    mean: Vec<f32>,
    covariance: Vec<f32>,
    whiten: Vec<f32>,
    weights: Vec<f32>,
    trace: Vec<f32>,
    whitened: Vec<f32>,
    ticks: u32,
}

impl Learner {
    fn new(dim: usize) -> Self {
        let mut whiten = vec![0.0; dim * dim];
        for i in 0..dim {
            whiten[i * dim + i] = 1.0;
        }
        Self {
            dim,
            mean: vec![0.0; dim],
            covariance: vec![0.0; dim * dim],
            whiten,
            weights: vec![0.0; dim],
            trace: vec![0.0; dim],
            whitened: vec![0.0; dim],
            ticks: 0,
        }
    }

    /// Pre-noise turn output `w . z` for raw input `x`, before this record's update.
    fn output(&mut self, x: &[f32]) -> f32 {
        let dim = self.dim;
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += self.whiten[i * dim + j] * (x[j] - self.mean[j]);
            }
            self.whitened[i] = sum;
        }
        self.weights
            .iter()
            .zip(&self.whitened)
            .map(|(w, z)| w * z)
            .sum()
    }

    /// Gradient of the output with respect to the raw input: whitening transpose times weights.
    fn input_gradient(&self) -> Vec<f32> {
        let dim = self.dim;
        (0..dim)
            .map(|j| {
                (0..dim)
                    .map(|i| self.whiten[i * dim + j] * self.weights[i])
                    .sum()
            })
            .collect()
    }

    /// Applies the credit rule and adapts statistics; returns the TD error.
    fn update(&mut self, x: &[f32], record: &Record, next_value: f32) -> f32 {
        let dim = self.dim;
        let died = record.death == 1.0;
        let delta = record.reward + TD_DISCOUNT * next_value - record.value;
        let norm = 1.0 + x.iter().map(|v| v * v).sum::<f32>();
        for i in 0..dim {
            // On death the trace is zeroed before the update, so e = 0 and it
            // does not carry into the next record.
            self.trace[i] = if died {
                0.0
            } else {
                TD_DISCOUNT * TD_LAMBDA * self.trace[i]
                    + record.turn_innovation * self.whitened[i] / norm
            };
            self.weights[i] += ACTION_WEIGHT_LEARNING_RATE * delta * self.trace[i];
        }
        let centred: Vec<f32> = (0..dim).map(|i| x[i] - self.mean[i]).collect();
        for i in 0..dim {
            self.mean[i] += ADAPTATION_RATE * centred[i];
            for j in 0..dim {
                let cell = &mut self.covariance[i * dim + j];
                *cell += ADAPTATION_RATE * (centred[i] * centred[j] - *cell);
            }
        }
        self.ticks += 1;
        if self.ticks % VISION_WHITENING_REFRESH == 0 {
            self.refresh_whitening();
        }
        delta
    }

    fn refresh_whitening(&mut self) {
        let dim = self.dim;
        let (values, vectors) = jacobi(&self.covariance, dim);
        let scales: Vec<f32> = values
            .iter()
            .map(|value| 1.0 / (value.max(0.0) + WHITENING_EPSILON).sqrt())
            .collect();
        for i in 0..dim {
            for j in 0..dim {
                self.whiten[i * dim + j] = (0..dim)
                    .map(|k| vectors[i * dim + k] * scales[k] * vectors[j * dim + k])
                    .sum();
            }
        }
    }
}

/// Maps column means to the learner input and adapts any learned pooling.
trait FrontEnd {
    fn dim(&self) -> usize;
    fn input(&mut self, columns: &[f32]) -> Vec<f32>;
    fn adapt(&mut self, columns: &[f32], delta: f32, gradient: &[f32]);
}

/// Arm A: fixed hemifield means.
struct HemifieldFront;

impl FrontEnd for HemifieldFront {
    fn dim(&self) -> usize {
        POOLED_COUNT
    }
    fn input(&mut self, columns: &[f32]) -> Vec<f32> {
        hemifield(columns)
    }
    fn adapt(&mut self, _columns: &[f32], _delta: f32, _gradient: &[f32]) {}
}

/// Arm B: the 32 column means unchanged.
struct ColumnFront;

impl FrontEnd for ColumnFront {
    fn dim(&self) -> usize {
        COLUMN_COUNT
    }
    fn input(&mut self, columns: &[f32]) -> Vec<f32> {
        columns.to_vec()
    }
    fn adapt(&mut self, _columns: &[f32], _delta: f32, _gradient: &[f32]) {}
}

/// Arm C: pooling matrix born as the hemifield average and trained by credit.
struct CreditPoolFront {
    pooling: Vec<f32>,
}

impl CreditPoolFront {
    fn new() -> Self {
        let mut pooling = vec![0.0; POOLED_COUNT * COLUMN_COUNT];
        for row in 0..POOLED_COUNT {
            let (side, channel) = (row / CHANNEL_COUNT, row % CHANNEL_COUNT);
            for column in 0..HEMIFIELD_COLUMNS {
                let source = (side * HEMIFIELD_COLUMNS + column) * CHANNEL_COUNT + channel;
                pooling[row * COLUMN_COUNT + source] = HEMIFIELD_WEIGHT;
            }
        }
        Self { pooling }
    }
}

impl FrontEnd for CreditPoolFront {
    fn dim(&self) -> usize {
        POOLED_COUNT
    }
    fn input(&mut self, columns: &[f32]) -> Vec<f32> {
        (0..POOLED_COUNT)
            .map(|i| {
                (0..COLUMN_COUNT)
                    .map(|j| self.pooling[i * COLUMN_COUNT + j] * columns[j])
                    .sum()
            })
            .collect()
    }
    fn adapt(&mut self, columns: &[f32], delta: f32, gradient: &[f32]) {
        let norm = 1.0 + columns.iter().map(|v| v * v).sum::<f32>();
        for i in 0..POOLED_COUNT {
            for j in 0..COLUMN_COUNT {
                self.pooling[i * COLUMN_COUNT + j] +=
                    POOLING_ETA * delta * gradient[i] * columns[j] / norm;
            }
        }
    }
}

/// Arm D: Oja/Sanger components of the centred column means.
struct OjaFront {
    components: Vec<f32>,
    mean: Vec<f32>,
    centred: Vec<f32>,
    projected: Vec<f32>,
}

impl OjaFront {
    fn new() -> Self {
        let mut components = vec![0.0; POOLED_COUNT * COLUMN_COUNT];
        for i in 0..POOLED_COUNT {
            components[i * COLUMN_COUNT + i] = 1.0;
        }
        Self {
            components,
            mean: vec![0.0; COLUMN_COUNT],
            centred: vec![0.0; COLUMN_COUNT],
            projected: vec![0.0; POOLED_COUNT],
        }
    }
}

impl FrontEnd for OjaFront {
    fn dim(&self) -> usize {
        POOLED_COUNT
    }
    fn input(&mut self, columns: &[f32]) -> Vec<f32> {
        for j in 0..COLUMN_COUNT {
            self.centred[j] = columns[j] - self.mean[j];
        }
        for i in 0..POOLED_COUNT {
            self.projected[i] = (0..COLUMN_COUNT)
                .map(|j| self.components[i * COLUMN_COUNT + j] * self.centred[j])
                .sum();
        }
        self.projected.clone()
    }
    fn adapt(&mut self, _columns: &[f32], _delta: f32, _gradient: &[f32]) {
        let norm = 1.0 + self.centred.iter().map(|v| v * v).sum::<f32>();
        let old = self.components.clone();
        for i in 0..POOLED_COUNT {
            for j in 0..COLUMN_COUNT {
                let mut lower = 0.0;
                for k in 0..=i {
                    lower += self.projected[k] * old[k * COLUMN_COUNT + j];
                }
                self.components[i * COLUMN_COUNT + j] +=
                    OJA_RATE * self.projected[i] * (self.centred[j] - lower) / norm;
            }
        }
        for j in 0..COLUMN_COUNT {
            self.mean[j] += ADAPTATION_RATE * self.centred[j];
        }
    }
}

/// Runs one arm over one agent's records and returns the turn output per record.
fn run_arm(records: &[&Record], mut front: impl FrontEnd) -> Vec<f32> {
    let mut learner = Learner::new(front.dim());
    let mut outputs = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let x = front.input(&record.columns);
        outputs.push(learner.output(&x));
        let next_value = if record.death == 1.0 {
            0.0
        } else {
            records.get(index + 1).map_or(0.0, |next| next.value)
        };
        let gradient = learner.input_gradient();
        let delta = learner.update(&x, record, next_value);
        front.adapt(&record.columns, delta, &gradient);
    }
    outputs
}

/// Food side convention: `-sign(food_bearing)`.
fn food_side(bearing: f32) -> f64 {
    if bearing > 0.0 {
        -1.0
    } else if bearing < 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Pearson correlation of turn output and food side over the scored records of one agent.
fn score_agent(records: &[&Record], outputs: &[f32]) -> Option<f64> {
    let start = SCORE_START_FRACTION * RUN_TICKS;
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut count = 0usize;
    for (record, output) in records.iter().zip(outputs) {
        if record.food_in_view != 1.0 || record.tick < start {
            continue;
        }
        let (x, y) = (f64::from(*output), food_side(record.food_bearing));
        n += 1.0;
        count += 1;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    if count < MIN_SCORED_RECORDS {
        return None;
    }
    let (cov, var_x, var_y) = (sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n);
    let denominator = (var_x * var_y).sqrt();
    if denominator > 0.0 && denominator.is_finite() {
        Some(cov / denominator)
    } else {
        Some(0.0)
    }
}

/// Per-agent scores for arms A, B, C, D (`None` when the agent has too few scored records).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentFit {
    /// Agent index as logged.
    pub agent: f32,
    /// Scores in the order A, B, C, D.
    pub scores: [Option<f64>; ARM_COUNT],
}

/// Fits all four arms to every agent in one log (records in log order per agent).
pub fn fit_log(records: &[Record]) -> Vec<AgentFit> {
    let mut by_agent: BTreeMap<u32, Vec<&Record>> = BTreeMap::new();
    for record in records {
        by_agent
            .entry(record.agent.to_bits())
            .or_default()
            .push(record);
    }
    by_agent
        .into_values()
        .map(|agent_records| {
            let outputs = [
                run_arm(&agent_records, HemifieldFront),
                run_arm(&agent_records, ColumnFront),
                run_arm(&agent_records, CreditPoolFront::new()),
                run_arm(&agent_records, OjaFront::new()),
            ];
            let scores = std::array::from_fn(|arm| score_agent(&agent_records, &outputs[arm]));
            AgentFit {
                agent: agent_records[0].agent,
                scores,
            }
        })
        .collect()
}

/// Summary of one arm over scored agents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmSummary {
    /// Mean score over scored agents.
    pub mean: f64,
    /// Standard error of the mean.
    pub standard_error: f64,
    /// Scored agents with a score above zero.
    pub above_zero: usize,
    /// Agents scored.
    pub scored: usize,
    /// Agents excluded for too few records.
    pub excluded: usize,
}

/// Summarises per-agent scores; excluded agents are counted, not scored zero.
pub fn summarize(scores: &[Option<f64>]) -> ArmSummary {
    let (mut n, mut sum, mut above_zero, mut excluded) = (0.0f64, 0.0f64, 0, 0);
    for score in scores {
        match score {
            Some(value) => {
                n += 1.0;
                sum += value;
                above_zero += usize::from(*value > 0.0);
            }
            None => excluded += 1,
        }
    }
    let scored = scores.len() - excluded;
    if scored == 0 {
        return ArmSummary {
            mean: 0.0,
            standard_error: 0.0,
            above_zero,
            scored,
            excluded,
        };
    }
    let mean = sum / n;
    let squares: f64 = scores
        .iter()
        .flatten()
        .map(|v| (v - mean) * (v - mean))
        .sum();
    let standard_error = if scored > 1 {
        (squares / (n - 1.0) / n).sqrt()
    } else {
        0.0
    };
    ArmSummary {
        mean,
        standard_error,
        above_zero,
        scored,
        excluded,
    }
}

/// Outcome of the pooling gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// A learned pooling matches or beats the hemifield reference.
    Ported,
    /// No learned pooling reaches the hemifield reference.
    NotPorted,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ported => write!(f, "PORTED"),
            Self::NotPorted => write!(f, "NOT PORTED"),
        }
    }
}

/// Gate: ported iff `max(mean C, mean D) >= mean A` (ties port).
pub fn gate_verdict(mean_a: f64, mean_c: f64, mean_d: f64) -> Verdict {
    if mean_c.max(mean_d) >= mean_a {
        Verdict::Ported
    } else {
        Verdict::NotPorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Agents in a synthetic log.
    const TEST_AGENTS: u16 = 4;
    /// Records per synthetic agent.
    const TEST_RECORDS: u16 = 3000;
    /// First tick of synthetic records, inside the scored window.
    const TEST_START_TICK: f32 = 250_000.0;
    /// Noise amplitude on synthetic columns.
    const TEST_NOISE: f32 = 0.05;
    /// Reward granted when the innovation points toward the food side.
    const TEST_REWARD: f32 = 0.5;
    /// Tolerance around zero for a cue-free log.
    const TEST_ZERO_TOLERANCE: f64 = 0.1;
    /// Largest `u16` as a float, for unit-interval draws.
    const UNIT_DIVISOR: f32 = 65535.0;
    /// Midpoint of the unit interval, splitting draws into signs.
    const SIGN_SPLIT: f32 = 0.5;
    /// Seed of the synthetic log generator.
    const TEST_SEED: u64 = 7;
    /// Records written in the malformed-log test.
    const MALFORMED_RECORDS: usize = 3;
    /// Byte that replaces the first magic byte in the bad-magic case.
    const BAD_MAGIC_BYTE: u8 = b'Z';
    /// Gate baseline (arm A) score.
    const GATE_BASELINE: f64 = 0.3;
    /// Gate candidate score above the baseline.
    const GATE_ABOVE: f64 = 0.4;
    /// Gate candidate score below the baseline.
    const GATE_MID: f64 = 0.2;
    /// Gate candidate score well below the baseline.
    const GATE_LOW: f64 = 0.1;
    /// LCG multiplier.
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    /// LCG increment.
    const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
    /// Shift selecting the top 16 bits of the LCG state.
    const LCG_SHIFT: u32 = 48;

    fn unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        let bits = u16::try_from(*state >> LCG_SHIFT).unwrap_or(0);
        f32::from(bits) / UNIT_DIVISOR
    }

    fn signed(state: &mut u64) -> f32 {
        if unit(state) < SIGN_SPLIT {
            -1.0
        } else {
            1.0
        }
    }

    fn synthetic(cue: bool) -> Vec<Record> {
        let mut state = TEST_SEED;
        let mut records = Vec::new();
        for agent in 0..TEST_AGENTS {
            for step in 0..TEST_RECORDS {
                let bearing = signed(&mut state);
                let side = -bearing;
                let innovation = signed(&mut state);
                let mut columns = [0.0; COLUMN_COUNT];
                for value in &mut columns {
                    *value = TEST_NOISE * (unit(&mut state) - SIGN_SPLIT);
                }
                if cue {
                    for column in 0..RAY_COLUMNS {
                        let sign = if column < HEMIFIELD_COLUMNS {
                            1.0
                        } else {
                            -1.0
                        };
                        columns[column * CHANNEL_COUNT] += sign * side;
                    }
                }
                let reward = if cue {
                    TEST_REWARD * innovation * side
                } else {
                    TEST_REWARD * signed(&mut state)
                };
                records.push(Record {
                    columns,
                    reward,
                    value: 0.0,
                    turn_innovation: innovation,
                    food_bearing: bearing,
                    food_in_view: 1.0,
                    death: 0.0,
                    agent: f32::from(agent),
                    tick: TEST_START_TICK + f32::from(step),
                });
            }
        }
        records
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pooling_study_{}_{name}.bin", std::process::id()))
    }

    fn reference_scores(records: &[Record]) -> Vec<Option<f64>> {
        fit_log(records).iter().map(|fit| fit.scores[0]).collect()
    }

    #[test]
    fn planted_side_cue_is_learned_by_reference_arm() {
        let summary = summarize(&reference_scores(&synthetic(true)));
        assert_eq!(summary.scored, usize::from(TEST_AGENTS));
        assert!(summary.mean > TEST_ZERO_TOLERANCE, "mean {}", summary.mean);
    }

    #[test]
    fn cue_free_log_scores_near_zero() {
        let summary = summarize(&reference_scores(&synthetic(false)));
        assert_eq!(summary.scored, usize::from(TEST_AGENTS));
        assert!(
            summary.mean.abs() < TEST_ZERO_TOLERANCE,
            "mean {}",
            summary.mean
        );
    }

    #[test]
    fn malformed_records_are_rejected() {
        let records = synthetic(false);
        let path = temp_path("malformed");
        write_log(&path, &records[..MALFORMED_RECORDS]).unwrap();
        assert_eq!(read_log(&path).unwrap().len(), MALFORMED_RECORDS);
        let good = std::fs::read(&path).unwrap();

        let mut bad_magic = good.clone();
        bad_magic[0] = BAD_MAGIC_BYTE;
        let short = good[..HEADER_BYTES - 1].to_vec();
        let truncated = good[..good.len() - 1].to_vec();
        let mut non_finite = good.clone();
        non_finite[HEADER_BYTES..HEADER_BYTES + FLOAT_BYTES]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        for bytes in [bad_magic, short, truncated, non_finite] {
            std::fs::write(&path, bytes).unwrap();
            let err = read_log(&path).unwrap_err();
            assert!(err.to_string().contains("pooling_study_"));
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn gate_matches_fixed_rule() {
        assert_eq!(
            gate_verdict(GATE_BASELINE, GATE_MID, GATE_ABOVE),
            Verdict::Ported
        );
        assert_eq!(
            gate_verdict(GATE_BASELINE, GATE_BASELINE, GATE_LOW),
            Verdict::Ported
        );
        assert_eq!(
            gate_verdict(GATE_BASELINE, GATE_LOW, GATE_BASELINE),
            Verdict::Ported
        );
        assert_eq!(
            gate_verdict(GATE_BASELINE, GATE_MID, GATE_LOW),
            Verdict::NotPorted
        );
        assert_eq!(Verdict::NotPorted.to_string(), "NOT PORTED");
    }
}
