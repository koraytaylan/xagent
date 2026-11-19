//! Free-life per-tick sample, streaming per-agent accumulator and metrics.

use serde::{Deserialize, Serialize};
use xagent_brain::buffers::ENCODED_DIMENSION;

/// Food counts as in view at or below this distance.
pub const FOOD_IN_VIEW_DISTANCE: f64 = 5.5;
/// Food or hazard counts as in the field of view when `|bearing|` is below this.
pub const VIEW_HALF_ANGLE: f64 = std::f64::consts::FRAC_PI_4;
/// A hazard counts as ahead at or below this distance.
pub const HAZARD_AHEAD_DISTANCE: f64 = 10.0;
/// Food is centred when `|bearing|` is below this.
pub const CENTRED_BEARING: f64 = 0.15;
/// Food is off axis when `|bearing|` is above this.
pub const OFF_AXIS_BEARING: f64 = 0.45;
/// Number of equal distance bins over `[0, FOOD_IN_VIEW_DISTANCE)`.
pub const VALUE_DISTANCE_BINS: usize = 2;
/// Samples a sighting survives after food leaves view.
pub const SIGHTING_GRACE_BRAIN_TICKS: u32 = 10;
/// Fraction of the life, counted from its end, that forms the late window.
pub const LATE_WINDOW_FRACTION: f64 = 0.3;
/// Width of the brain's encoded state.
pub const ENCODING_WIDTH: usize = ENCODED_DIMENSION;
/// Number of pathway inputs feeding the credit traces (scent, vision and motor
/// features of the trunk; a literal fixed by the kernel layout).
pub const PATHWAY_INPUTS: usize = 10;
/// Variance below which a Pearson correlation is reported as zero.
const VARIANCE_EPSILON: f64 = 1e-12;

/// One agent's state at one brain tick.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FreeLifeSample {
    /// 0-based brain tick index within the free life.
    pub brain_tick: u64,
    pub alive: bool,
    pub meals_total: u64,
    pub deaths_total: u64,
    pub on_hazard: bool,
    /// The executed turn.
    pub motor_turn: f32,
    pub food_distance: f32,
    pub food_bearing: f32,
    pub hazard_distance: f32,
    pub hazard_bearing: f32,
    pub encoding: Vec<f32>,
    pub pathway_inputs: [f32; PATHWAY_INPUTS],
    pub reward: f32,
    pub value: f32,
    pub td_error: f32,
    pub forward_noise: f32,
    pub turn_noise: f32,
    pub turn_innovation: f32,
    pub encoding_turn_output: f32,
    pub scent_turn_output: f32,
    pub vision_turn_output: f32,
    pub encoding_forward_output: f32,
}

/// Accumulator settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub trace_decay: f64,
    pub total_brain_ticks: u64,
}

/// Pearson correlations of each turn output with the food side.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PathwayCorrelation {
    pub encoding: f64,
    pub scent: f64,
    pub vision: f64,
}

/// Metrics over one window of samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowMetrics {
    pub samples: u64,
    pub alive_samples: u64,
    pub meals: u64,
    pub deaths: u64,
    pub hazard_deaths: u64,
    pub sightings: u64,
    pub sightings_ending_in_meal: u64,
    pub sighting_meal_share: f64,
    pub turn_toward_food: f64,
    pub turn_away_from_hazard: f64,
    pub pathway_food_correlation: PathwayCorrelation,
    pub turn_credit: [f64; PATHWAY_INPUTS],
    pub forward_credit: [f64; PATHWAY_INPUTS],
    pub value_gap_in_view: f64,
    pub value_gap_centred: f64,
}

/// Metrics of one agent over the whole life and its late window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub whole: WindowMetrics,
    pub late: WindowMetrics,
}

/// Sign with `sign(0.0) == 0.0` (unlike `f32::signum`).
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Converts a count to `f64`.
///
/// Counts stay far below 2^53, so the conversion is exact.
#[allow(clippy::cast_precision_loss)]
pub fn count_as_f64(n: u64) -> f64 {
    n as f64
}

/// Converts a rounded length back to a count clamped to `[0, total]`.
///
/// The value is non-negative and clamped to `total` first, so neither
/// truncation nor sign loss can occur.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn clamped_count(x: f64, total: u64) -> u64 {
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    if x >= count_as_f64(total) {
        return total;
    }
    (x as u64).min(total)
}

/// First `brain_tick` of the late window for a life of `total` ticks.
pub fn late_window_start(total: u64) -> u64 {
    let late_length = clamped_count((count_as_f64(total) * LATE_WINDOW_FRACTION).round(), total);
    total - late_length
}

/// Pearson correlation of two equal-length series; 0 when either variance is 0.
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let mut sums = PearsonSums::default();
    for (x, y) in xs.iter().zip(ys) {
        sums.push(*x, *y);
    }
    sums.value()
}

#[derive(Clone, Copy, Default)]
struct PearsonSums {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl PearsonSums {
    fn push(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }

    fn value(&self) -> f64 {
        let vx = self.n * self.sxx - self.sx * self.sx;
        let vy = self.n * self.syy - self.sy * self.sy;
        if self.n < 2.0 || vx <= VARIANCE_EPSILON || vy <= VARIANCE_EPSILON {
            return 0.0;
        }
        let r = (self.n * self.sxy - self.sx * self.sy) / (vx * vy).sqrt();
        finite(r.clamp(-1.0, 1.0))
    }
}

#[derive(Clone, Copy, Default)]
struct Mean {
    sum: f64,
    n: u64,
}

impl Mean {
    fn push(&mut self, x: f64) {
        self.sum += x;
        self.n += 1;
    }

    fn value(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            finite(self.sum / count_as_f64(self.n))
        }
    }
}

fn finite(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Default)]
struct BinStats {
    centred: Mean,
    off: Mean,
}

/// Per-sample quantities derived against the previous sample.
struct Step {
    meals: u64,
    deaths: u64,
    hazard_deaths: u64,
    in_view: bool,
    food_side: f64,
}

#[derive(Default)]
struct Window {
    samples: u64,
    alive_samples: u64,
    meals: u64,
    deaths: u64,
    hazard_deaths: u64,
    sightings: u64,
    ending: u64,
    toward: Mean,
    away: Mean,
    corr: [PearsonSums; 3],
    turn_credit: [f64; PATHWAY_INPUTS],
    forward_credit: [f64; PATHWAY_INPUTS],
    in_view: Mean,
    out_of_view: Mean,
    bins: [BinStats; VALUE_DISTANCE_BINS],
}

fn distance_bin(distance: f64) -> usize {
    let width = FOOD_IN_VIEW_DISTANCE
        / count_as_f64(u64::try_from(VALUE_DISTANCE_BINS).unwrap_or(u64::MAX));
    let mut edge = width;
    for bin in 0..VALUE_DISTANCE_BINS - 1 {
        if distance < edge {
            return bin;
        }
        edge += width;
    }
    VALUE_DISTANCE_BINS - 1
}

impl Window {
    fn add(
        &mut self,
        s: &FreeLifeSample,
        step: &Step,
        turn_credit: &[f64; PATHWAY_INPUTS],
        forward_credit: &[f64; PATHWAY_INPUTS],
    ) {
        self.samples += 1;
        self.meals += step.meals;
        self.deaths += step.deaths;
        self.hazard_deaths += step.hazard_deaths;
        for k in 0..PATHWAY_INPUTS {
            self.turn_credit[k] += turn_credit[k];
            self.forward_credit[k] += forward_credit[k];
        }
        if !s.alive {
            return;
        }
        self.alive_samples += 1;
        self.add_behaviour(s, step);
    }

    fn add_behaviour(&mut self, s: &FreeLifeSample, step: &Step) {
        let turn = f64::from(s.motor_turn);
        let value = f64::from(s.value);
        let hbearing = f64::from(s.hazard_bearing);
        if f64::from(s.hazard_distance) <= HAZARD_AHEAD_DISTANCE
            && hbearing.abs() < VIEW_HALF_ANGLE
            && !s.on_hazard
        {
            self.away.push(turn * sign(hbearing));
        }
        if !step.in_view {
            self.out_of_view.push(value);
            return;
        }
        self.in_view.push(value);
        self.toward.push(turn * step.food_side);
        let outputs = [
            s.encoding_turn_output,
            s.scent_turn_output,
            s.vision_turn_output,
        ];
        for (sums, out) in self.corr.iter_mut().zip(outputs) {
            sums.push(step.food_side, f64::from(out));
        }
        let bearing = f64::from(s.food_bearing).abs();
        let bin = &mut self.bins[distance_bin(f64::from(s.food_distance))];
        if bearing < CENTRED_BEARING {
            bin.centred.push(value);
        } else if bearing > OFF_AXIS_BEARING {
            bin.off.push(value);
        }
    }

    fn value_gap_centred(&self) -> f64 {
        let mut sum = 0.0;
        let mut n = 0_u64;
        for bin in &self.bins {
            if bin.centred.n > 0 && bin.off.n > 0 {
                sum += bin.centred.value() - bin.off.value();
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            finite(sum / count_as_f64(n))
        }
    }

    fn finish(self) -> WindowMetrics {
        let per_sample = |credit: [f64; PATHWAY_INPUTS]| {
            credit.map(|c| {
                if self.samples == 0 {
                    0.0
                } else {
                    finite(c / count_as_f64(self.samples))
                }
            })
        };
        let gap = if self.in_view.n > 0 && self.out_of_view.n > 0 {
            self.in_view.value() - self.out_of_view.value()
        } else {
            0.0
        };
        let share = if self.sightings == 0 {
            0.0
        } else {
            count_as_f64(self.ending) / count_as_f64(self.sightings)
        };
        WindowMetrics {
            samples: self.samples,
            alive_samples: self.alive_samples,
            meals: self.meals,
            deaths: self.deaths,
            hazard_deaths: self.hazard_deaths,
            sightings: self.sightings,
            sightings_ending_in_meal: self.ending,
            sighting_meal_share: share,
            turn_toward_food: self.toward.value(),
            turn_away_from_hazard: self.away.value(),
            pathway_food_correlation: PathwayCorrelation {
                encoding: self.corr[0].value(),
                scent: self.corr[1].value(),
                vision: self.corr[2].value(),
            },
            turn_credit: per_sample(self.turn_credit),
            forward_credit: per_sample(self.forward_credit),
            value_gap_in_view: finite(gap),
            value_gap_centred: self.value_gap_centred(),
        }
    }
}

struct Prev {
    meals_total: u64,
    deaths_total: u64,
    on_hazard: bool,
}

struct OpenSighting {
    late: bool,
    out_of_view: u32,
}

/// Streaming metrics accumulator for one agent.
pub struct AgentAccumulator {
    config: MetricsConfig,
    late_start: u64,
    whole: Window,
    late: Window,
    prev: Option<Prev>,
    armed: bool,
    open: Option<OpenSighting>,
    turn_trace: [f64; PATHWAY_INPUTS],
    forward_trace: [f64; PATHWAY_INPUTS],
}

impl AgentAccumulator {
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            late_start: late_window_start(config.total_brain_ticks),
            whole: Window::default(),
            late: Window::default(),
            prev: None,
            armed: false,
            open: None,
            turn_trace: [0.0; PATHWAY_INPUTS],
            forward_trace: [0.0; PATHWAY_INPUTS],
        }
    }

    pub fn push(&mut self, s: &FreeLifeSample) {
        let late = s.brain_tick >= self.late_start;
        let step = self.step(s);
        if step.deaths > 0 {
            self.turn_trace = [0.0; PATHWAY_INPUTS];
            self.forward_trace = [0.0; PATHWAY_INPUTS];
        }
        self.advance_sighting(s, &step, late);
        let (turn_credit, forward_credit) = self.advance_traces(s);
        self.whole.add(s, &step, &turn_credit, &forward_credit);
        if late {
            self.late.add(s, &step, &turn_credit, &forward_credit);
        }
        self.prev = Some(Prev {
            meals_total: s.meals_total,
            deaths_total: s.deaths_total,
            on_hazard: s.on_hazard,
        });
    }

    pub fn finish(self) -> AgentMetrics {
        AgentMetrics {
            whole: self.whole.finish(),
            late: self.late.finish(),
        }
    }

    fn step(&self, s: &FreeLifeSample) -> Step {
        let (meals, deaths, hazard_deaths) = match &self.prev {
            Some(p) => {
                let deaths = s.deaths_total.saturating_sub(p.deaths_total);
                (
                    s.meals_total.saturating_sub(p.meals_total),
                    deaths,
                    if p.on_hazard { deaths } else { 0 },
                )
            }
            None => (0, 0, 0),
        };
        let bearing = f64::from(s.food_bearing);
        let in_view = s.alive
            && f64::from(s.food_distance) <= FOOD_IN_VIEW_DISTANCE
            && bearing.abs() < VIEW_HALF_ANGLE;
        Step {
            meals,
            deaths,
            hazard_deaths,
            in_view,
            food_side: -sign(bearing),
        }
    }

    fn advance_sighting(&mut self, s: &FreeLifeSample, step: &Step, late: bool) {
        if step.deaths > 0 {
            self.open = None;
            self.armed = false;
        } else if step.meals > 0 {
            if let Some(open) = self.open.take() {
                self.whole.ending += 1;
                if open.late {
                    self.late.ending += 1;
                }
            }
        }
        if !s.alive {
            return;
        }
        if !step.in_view {
            self.armed = true;
            if let Some(open) = &mut self.open {
                open.out_of_view += 1;
                if open.out_of_view > SIGHTING_GRACE_BRAIN_TICKS {
                    self.open = None;
                }
            }
        } else if let Some(open) = &mut self.open {
            open.out_of_view = 0;
        } else if self.armed {
            self.armed = false;
            self.open = Some(OpenSighting {
                late,
                out_of_view: 0,
            });
            self.whole.sightings += 1;
            if late {
                self.late.sightings += 1;
            }
        }
    }

    fn advance_traces(
        &mut self,
        s: &FreeLifeSample,
    ) -> ([f64; PATHWAY_INPUTS], [f64; PATHWAY_INPUTS]) {
        let td = f64::from(s.td_error);
        let mut turn_credit = [0.0; PATHWAY_INPUTS];
        let mut forward_credit = [0.0; PATHWAY_INPUTS];
        let x = s.pathway_inputs.map(f64::from);
        let norm = 1.0 + x.iter().map(|v| v * v).sum::<f64>();
        let decay = self.config.trace_decay;
        let turn_noise = f64::from(s.turn_innovation);
        let forward_noise = f64::from(s.forward_noise);
        for k in 0..PATHWAY_INPUTS {
            turn_credit[k] = td * self.turn_trace[k];
            forward_credit[k] = td * self.forward_trace[k];
            self.turn_trace[k] = decay * self.turn_trace[k] + turn_noise * x[k] / norm;
            self.forward_trace[k] = decay * self.forward_trace[k] + forward_noise * x[k] / norm;
        }
        (turn_credit, forward_credit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(tick: u64) -> FreeLifeSample {
        FreeLifeSample {
            brain_tick: tick,
            alive: true,
            meals_total: 0,
            deaths_total: 0,
            on_hazard: false,
            motor_turn: 0.0,
            food_distance: 100.0,
            food_bearing: 0.0,
            hazard_distance: 100.0,
            hazard_bearing: 0.0,
            encoding: vec![0.0; ENCODING_WIDTH],
            pathway_inputs: [0.0; PATHWAY_INPUTS],
            reward: 0.0,
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

    fn viewed(tick: u64, in_view: bool) -> FreeLifeSample {
        let mut s = base(tick);
        if in_view {
            s.food_distance = 2.0;
        }
        s
    }

    fn run(samples: &[FreeLifeSample], total: u64, decay: f64) -> AgentMetrics {
        let mut acc = AgentAccumulator::new(MetricsConfig {
            trace_decay: decay,
            total_brain_ticks: total,
        });
        for s in samples {
            acc.push(s);
        }
        acc.finish()
    }

    fn sighting_run(views: &[bool], meal_at: usize) -> WindowMetrics {
        let samples: Vec<_> = views
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut s = viewed(u64::try_from(i).unwrap(), *v);
                s.meals_total = u64::from(i >= meal_at);
                s
            })
            .collect();
        run(&samples, u64::try_from(samples.len()).unwrap(), 0.5).whole
    }

    #[test]
    fn sign_zero_is_zero() {
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
        assert_eq!(sign(3.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
    }

    #[test]
    fn pearson_positive_pair() {
        let r = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!((r - 1.0).abs() < 1e-12);
        let r = pearson(&[1.0, 2.0, 3.0], &[6.0, 4.0, 2.0]);
        assert!((r + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_zero_variance_is_zero() {
        assert_eq!(pearson(&[1.0, 2.0, 3.0], &[0.3, 0.3, 0.3]), 0.0);
        assert_eq!(pearson(&[], &[]), 0.0);
    }

    #[test]
    fn sighting_meal_inside_grace() {
        let mut views = vec![false, true];
        views.extend([false; 11]);
        let m = sighting_run(&views, 12);
        assert_eq!(m.sightings, 1);
        assert_eq!(m.sightings_ending_in_meal, 1);
        assert_eq!(m.sighting_meal_share, 1.0);
        assert_eq!(m.meals, 1);
    }

    #[test]
    fn sighting_expires_after_grace() {
        let mut views = vec![false, true];
        views.extend([false; 12]);
        let m = sighting_run(&views, 13);
        assert_eq!(m.sightings, 1);
        assert_eq!(m.sightings_ending_in_meal, 0);
        assert_eq!(m.sighting_meal_share, 0.0);
    }

    #[test]
    fn hazard_death_counted() {
        let mut a = base(0);
        a.on_hazard = true;
        let mut b = base(1);
        b.alive = false;
        b.deaths_total = 1;
        let m = run(&[a, b], 2, 0.5).whole;
        assert_eq!(m.deaths, 1);
        assert_eq!(m.hazard_deaths, 1);
    }

    #[test]
    fn off_hazard_death_not_counted() {
        let a = base(0);
        let mut b = base(1);
        b.deaths_total = 1;
        let m = run(&[a, b], 2, 0.5).whole;
        assert_eq!(m.deaths, 1);
        assert_eq!(m.hazard_deaths, 0);
    }

    #[test]
    fn sighting_open_at_finish() {
        let m = sighting_run(&[false, true, true], 99);
        assert_eq!(m.sightings, 1);
        assert_eq!(m.sightings_ending_in_meal, 0);
    }

    #[test]
    fn late_window_boundary() {
        assert_eq!(late_window_start(10), 7);
        assert_eq!(late_window_start(300_000), 210_000);
        let samples: Vec<_> = (0..10).map(base).collect();
        let m = run(&samples, 10, 0.5);
        assert_eq!(m.whole.samples, 10);
        assert_eq!(m.late.samples, 3);
    }

    #[test]
    fn turn_toward_food_left_right() {
        let mut right = viewed(0, true);
        right.food_bearing = -0.2;
        right.motor_turn = 1.0;
        let m = run(&[right], 1, 0.5).whole;
        assert!((m.turn_toward_food - 1.0).abs() < 1e-9);
        let mut left = viewed(0, true);
        left.food_bearing = 0.2;
        left.motor_turn = 1.0;
        let m = run(&[left], 1, 0.5).whole;
        assert!((m.turn_toward_food + 1.0).abs() < 1e-9);
    }

    #[test]
    fn turn_away_from_hazard_off_ground() {
        let mut s = base(0);
        s.hazard_distance = 5.0;
        s.hazard_bearing = 0.3;
        s.motor_turn = 1.0;
        let m = run(&[s.clone()], 1, 0.5).whole;
        assert!((m.turn_away_from_hazard - 1.0).abs() < 1e-9);
        s.on_hazard = true;
        let m = run(&[s], 1, 0.5).whole;
        assert_eq!(m.turn_away_from_hazard, 0.0);
    }

    #[test]
    fn correlation_positive_sign() {
        let samples: Vec<_> = (0..6)
            .map(|i| {
                let mut s = viewed(i, true);
                s.food_bearing = if i % 2 == 0 { -0.2 } else { 0.2 };
                let side = if i % 2 == 0 { 1.0 } else { -1.0 };
                s.scent_turn_output = 0.5 * side;
                s.vision_turn_output = -side;
                s.encoding_turn_output = 0.3;
                s
            })
            .collect();
        let c = run(&samples, 6, 0.5).whole.pathway_food_correlation;
        assert!((c.scent - 1.0).abs() < 1e-9);
        assert!((c.vision + 1.0).abs() < 1e-9);
        assert_eq!(c.encoding, 0.0);
    }

    #[test]
    fn value_gap_skips_empty_bin() {
        let mut a = viewed(0, true);
        a.food_distance = 1.0;
        a.value = 3.0;
        let mut b = a.clone();
        b.brain_tick = 1;
        b.food_bearing = 0.6;
        b.value = 1.0;
        let mut c = viewed(2, true);
        c.food_distance = 4.0;
        c.value = 100.0;
        let m = run(&[a, b, c], 3, 0.5).whole;
        assert!((m.value_gap_centred - 2.0).abs() < 1e-9);
    }

    fn credit_sample(tick: u64, td: f32) -> FreeLifeSample {
        let mut s = base(tick);
        s.td_error = td;
        s.turn_innovation = 2.0;
        s.forward_noise = 1.0;
        s.pathway_inputs[0] = 1.0;
        s
    }

    #[test]
    fn credit_two_tick_hand_computed() {
        let mut a = credit_sample(0, 2.0);
        a.turn_innovation = 4.0;
        a.forward_noise = 2.0;
        let mut b = credit_sample(1, 3.0);
        b.turn_innovation = 4.0;
        let m = run(&[a, b], 2, 0.5);
        assert!((m.whole.turn_credit[0] - 3.0).abs() < 1e-9);
        assert!((m.whole.forward_credit[0] - 1.5).abs() < 1e-9);
        assert!((m.late.turn_credit[0] - 6.0).abs() < 1e-9);
        assert_eq!(m.whole.turn_credit[1], 0.0);
    }

    #[test]
    fn late_credit_excludes_early_sample() {
        let samples = [
            credit_sample(0, 1.0),
            credit_sample(1, 1.0),
            credit_sample(2, 2.0),
        ];
        let m = run(&samples, 3, 0.5);
        assert!((m.whole.turn_credit[0] - 4.0 / 3.0).abs() < 1e-9);
        assert!((m.late.turn_credit[0] - 3.0).abs() < 1e-9);
        assert_eq!(m.late.samples, 1);
    }

    #[test]
    fn death_resets_traces() {
        let a = credit_sample(0, 0.0);
        let mut b = credit_sample(1, 5.0);
        b.deaths_total = 1;
        let c = credit_sample(2, 1.0);
        let m = run(&[a, b, c], 3, 0.5).whole;
        assert!((m.turn_credit[0] - 1.0 / 3.0).abs() < 1e-9);
    }
}
