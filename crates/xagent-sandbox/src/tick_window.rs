//! Bounded rolling window of per-tick agent and food frames, with pinned
//! ranges around notable ticks and a versioned binary codec.

use std::fmt;

use xagent_brain::buffers::{
    FOOD_STATE_STRIDE, PHYS_STRIDE, P_ALIVE, P_ENERGY, P_EXPLORATION_RATE_OUT,
    P_FATIGUE_FACTOR_OUT, P_GRADIENT_OUT, P_INTEGRITY, P_MOTOR_FWD_OUT, P_MOTOR_TURN_OUT, P_POS_X,
    P_POS_Y, P_POS_Z, P_PREDICTION_ERROR, P_RAW_GRADIENT_OUT, P_URGENCY_OUT, P_YAW,
};

use crate::replay::TickRecord;

/// Default rolling window length in ticks.
pub const DEFAULT_TICK_WINDOW_TICKS: u64 = 20_000;
/// Maximum rolling (unpinned) frames held at once.
pub const MAX_WINDOW_FRAMES: usize = 4_096;
/// A pin keeps frames within this many ticks on each side.
pub const PIN_HALF_WINDOW_TICKS: u64 = 2_000;
/// Maximum pins retained; the oldest is evicted first.
pub const MAX_PINNED_WINDOWS: usize = 32;
/// Maximum frames kept per pin.
pub const MAX_FRAMES_PER_PIN: usize = 512;
/// Maximum agents in one frame.
pub const MAX_FRAME_AGENTS: usize = 1_024;
/// Maximum food items in one frame.
pub const MAX_FRAME_FOOD: usize = 65_536;
/// Value of `TickRecord::phase` when the phase was not recorded.
pub const PHASE_NOT_RECORDED: u8 = 0;
/// Leading version of the frame codec.
pub const FRAME_CODEC_VERSION: u32 = 1;

/// Errors from window construction, snapshot conversion and the codec.
#[derive(Debug, PartialEq, Eq)]
pub enum TickWindowError {
    InvalidConfig(&'static str),
    PhysicsLength(usize),
    FoodLength(usize),
    TooManyAgents(usize),
    TooManyFood(usize),
    UnsupportedVersion(u32),
    Truncated,
    TrailingBytes,
    CountOutOfBounds(u64),
    NonFinite,
}

impl fmt::Display for TickWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(field) => write!(f, "invalid tick window config: {field}"),
            Self::PhysicsLength(n) => write!(f, "physics length {n} is not a stride multiple"),
            Self::FoodLength(n) => write!(f, "food length {n} is not a stride multiple"),
            Self::TooManyAgents(n) => write!(f, "too many agents: {n}"),
            Self::TooManyFood(n) => write!(f, "too many food items: {n}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported frame codec version {v}"),
            Self::Truncated => write!(f, "frame blob truncated"),
            Self::TrailingBytes => write!(f, "frame blob has trailing bytes"),
            Self::CountOutOfBounds(n) => write!(f, "count {n} out of bounds"),
            Self::NonFinite => write!(f, "non-finite position in frame"),
        }
    }
}

impl std::error::Error for TickWindowError {}

/// Retention bounds of a [`TickWindow`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TickWindowConfig {
    pub window_ticks: u64,
    pub max_frames: usize,
    pub pin_half_window_ticks: u64,
    pub max_pins: usize,
    pub max_frames_per_pin: usize,
}

impl Default for TickWindowConfig {
    fn default() -> Self {
        Self {
            window_ticks: DEFAULT_TICK_WINDOW_TICKS,
            max_frames: MAX_WINDOW_FRAMES,
            pin_half_window_ticks: PIN_HALF_WINDOW_TICKS,
            max_pins: MAX_PINNED_WINDOWS,
            max_frames_per_pin: MAX_FRAMES_PER_PIN,
        }
    }
}

impl TickWindowConfig {
    /// Refuses any zero bound.
    pub fn validate(&self) -> Result<(), TickWindowError> {
        if self.window_ticks == 0 {
            return Err(TickWindowError::InvalidConfig("window_ticks"));
        }
        if self.max_frames == 0 {
            return Err(TickWindowError::InvalidConfig("max_frames"));
        }
        if self.pin_half_window_ticks == 0 {
            return Err(TickWindowError::InvalidConfig("pin_half_window_ticks"));
        }
        if self.max_pins == 0 {
            return Err(TickWindowError::InvalidConfig("max_pins"));
        }
        if self.max_frames_per_pin == 0 {
            return Err(TickWindowError::InvalidConfig("max_frames_per_pin"));
        }
        Ok(())
    }
}

/// Why a range is pinned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinKind {
    Replacement { slot: usize },
    SeasonFlip,
}

/// A tick whose surrounding frames are retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pin {
    pub tick: u64,
    pub kind: PinKind,
}

impl Pin {
    fn same_as(&self, other: &Pin) -> bool {
        self.tick == other.tick
            && std::mem::discriminant(&self.kind) == std::mem::discriminant(&other.kind)
    }
}

/// One captured tick.
#[derive(Clone, Debug)]
pub struct WindowFrame {
    pub tick: u64,
    pub agents: Vec<TickRecord>,
    pub food: Vec<[f32; 4]>,
}

/// Builds a record from one physics row; fields the row lacks are not-recorded.
pub fn record_from_physics_row(row: &[f32]) -> TickRecord {
    TickRecord {
        position: [row[P_POS_X], row[P_POS_Y], row[P_POS_Z]],
        yaw: row[P_YAW],
        alive: row[P_ALIVE] > 0.5,
        energy: row[P_ENERGY],
        integrity: row[P_INTEGRITY],
        motor_forward: row[P_MOTOR_FWD_OUT],
        motor_turn: row[P_MOTOR_TURN_OUT],
        exploration_rate: row[P_EXPLORATION_RATE_OUT],
        prediction_error: row[P_PREDICTION_ERROR],
        gradient: row[P_GRADIENT_OUT],
        raw_gradient: row[P_RAW_GRADIENT_OUT],
        urgency: row[P_URGENCY_OUT],
        credit_magnitude: 0.0,
        patterns_recalled: 0,
        phase: PHASE_NOT_RECORDED,
        mean_attenuation: 0.0,
        curiosity_bonus: 0.0,
        fatigue_factor: row[P_FATIGUE_FACTOR_OUT],
        staleness: 0.0,
        vision_color: None,
    }
}

/// Builds a frame from already-copied physics and food slices.
pub fn frame_from_snapshot(
    tick: u64,
    physics: &[f32],
    food: &[f32],
) -> Result<WindowFrame, TickWindowError> {
    if physics.len() % PHYS_STRIDE != 0 {
        return Err(TickWindowError::PhysicsLength(physics.len()));
    }
    let agent_count = physics.len() / PHYS_STRIDE;
    if agent_count > MAX_FRAME_AGENTS {
        return Err(TickWindowError::TooManyAgents(agent_count));
    }
    if food.len() % FOOD_STATE_STRIDE != 0 {
        return Err(TickWindowError::FoodLength(food.len()));
    }
    let food_count = food.len() / FOOD_STATE_STRIDE;
    if food_count > MAX_FRAME_FOOD {
        return Err(TickWindowError::TooManyFood(food_count));
    }
    let agents = physics
        .as_chunks::<PHYS_STRIDE>()
        .0
        .iter()
        .map(|row| record_from_physics_row(row))
        .collect();
    let food = food
        .as_chunks::<FOOD_STATE_STRIDE>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    Ok(WindowFrame { tick, agents, food })
}

/// Rolling window of frames plus pinned ranges.
#[derive(Debug)]
pub struct TickWindow {
    config: TickWindowConfig,
    frames: Vec<WindowFrame>,
    pins: Vec<Pin>,
}

impl TickWindow {
    pub fn new(config: TickWindowConfig) -> Result<Self, TickWindowError> {
        config.validate()?;
        Ok(Self {
            config,
            frames: Vec::new(),
            pins: Vec::new(),
        })
    }

    /// Appends a frame newer than every held frame; returns false otherwise.
    pub fn capture(&mut self, frame: WindowFrame) -> bool {
        if self.newest_tick().is_some_and(|n| frame.tick <= n) {
            return false;
        }
        self.frames.push(frame);
        self.prune();
        true
    }

    /// Records a pin unless the same tick and kind variant is already held.
    pub fn pin(&mut self, pin: Pin) {
        if self.pins.iter().any(|p| p.same_as(&pin)) {
            return;
        }
        self.pins.push(pin);
        if self.pins.len() > self.config.max_pins {
            self.pins.remove(0);
        }
        self.prune();
    }

    fn prune(&mut self) {
        let mut pinned = vec![false; self.frames.len()];
        for pin in &self.pins {
            let mut hits: Vec<(u64, u64, usize)> = self
                .frames
                .iter()
                .enumerate()
                .filter_map(|(i, f)| {
                    let d = f.tick.abs_diff(pin.tick);
                    (d <= self.config.pin_half_window_ticks).then_some((d, f.tick, i))
                })
                .collect();
            hits.sort_unstable();
            for &(_, _, i) in hits.iter().take(self.config.max_frames_per_pin) {
                pinned[i] = true;
            }
        }
        let newest = self.newest_tick().unwrap_or(0);
        let window = self.config.window_ticks;
        let mut keep: Vec<bool> = self
            .frames
            .iter()
            .zip(&pinned)
            .map(|(f, &p)| p || newest.saturating_sub(f.tick) <= window)
            .collect();
        let mut unmarked = keep.iter().zip(&pinned).filter(|(&k, &p)| k && !p).count();
        for (k, &p) in keep.iter_mut().zip(&pinned) {
            if unmarked <= self.config.max_frames {
                break;
            }
            if *k && !p {
                *k = false;
                unmarked -= 1;
            }
        }
        let mut idx = 0;
        self.frames.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    /// Frames in tick order.
    pub fn frames(&self) -> impl Iterator<Item = &WindowFrame> {
        self.frames.iter()
    }

    /// Frames with `start <= tick <= end`.
    pub fn frames_in(&self, start: u64, end: u64) -> impl Iterator<Item = &WindowFrame> {
        self.frames
            .iter()
            .filter(move |f| f.tick >= start && f.tick <= end)
    }

    /// Retained pins in insertion order (oldest first).
    pub fn pins(&self) -> &[Pin] {
        &self.pins
    }

    /// Tick of the newest held frame, or `None` when the window is empty.
    pub fn newest_tick(&self) -> Option<u64> {
        self.frames.last().map(|f| f.tick)
    }

    /// Number of frames held, pinned plus rolling.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the window holds no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The validated configuration this window enforces.
    pub fn config(&self) -> &TickWindowConfig {
        &self.config
    }
}

/// Fixed bytes of one encoded agent without the vision payload.
const AGENT_MIN_BYTES: usize = 4 * 4 + 1 + 10 * 4 + 2 + 1 + 4 * 4 + 1;
/// Bytes of one encoded food entry: four `f32` values.
const FOOD_BYTES: usize = 4 * 4;
/// Fixed bytes of a frame outside its records: `u32` version, `u64` tick,
/// `u32` agent count and `u32` food count.
const FRAME_HEADER_BYTES: usize = 4 + 8 + 4 + 4;

/// Encodes a frame (little-endian, versioned).
///
/// # Panics
///
/// Panics if an agent, food or vision count exceeds `u32::MAX`, which the
/// format cannot represent; frames from `frame_from_snapshot` stay far below.
pub fn encode_frame(frame: &WindowFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        FRAME_HEADER_BYTES + frame.agents.len() * AGENT_MIN_BYTES + frame.food.len() * FOOD_BYTES,
    );
    out.extend_from_slice(&FRAME_CODEC_VERSION.to_le_bytes());
    out.extend_from_slice(&frame.tick.to_le_bytes());
    put_u32(&mut out, frame.agents.len());
    for a in &frame.agents {
        for v in [a.position[0], a.position[1], a.position[2], a.yaw] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(u8::from(a.alive));
        for v in [
            a.energy,
            a.integrity,
            a.motor_forward,
            a.motor_turn,
            a.exploration_rate,
            a.prediction_error,
            a.gradient,
            a.raw_gradient,
            a.urgency,
            a.credit_magnitude,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&a.patterns_recalled.to_le_bytes());
        out.push(a.phase);
        for v in [
            a.mean_attenuation,
            a.curiosity_bonus,
            a.fatigue_factor,
            a.staleness,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        match &a.vision_color {
            Some(v) => {
                out.push(1);
                put_u32(&mut out, v.len());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            None => out.push(0),
        }
    }
    put_u32(&mut out, frame.food.len());
    for f in &frame.food {
        for v in f {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn put_u32(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(
        &u32::try_from(n)
            .expect("frame count exceeds the codec's u32 range")
            .to_le_bytes(),
    );
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], TickWindowError> {
        if self.bytes.len() < n {
            return Err(TickWindowError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, TickWindowError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, TickWindowError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, TickWindowError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, TickWindowError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn f32(&mut self) -> Result<f32, TickWindowError> {
        Ok(f32::from_bits(self.u32()?))
    }
}

/// Decodes a frame, bounding every count before allocating.
pub fn decode_frame(bytes: &[u8]) -> Result<WindowFrame, TickWindowError> {
    let mut r = Reader { bytes };
    let version = r.u32()?;
    if version != FRAME_CODEC_VERSION {
        return Err(TickWindowError::UnsupportedVersion(version));
    }
    let tick = r.u64()?;
    let raw_agent_count = r.u32()?;
    let agent_count = usize::try_from(raw_agent_count)
        .map_err(|_| TickWindowError::CountOutOfBounds(u64::from(raw_agent_count)))?;
    if agent_count > MAX_FRAME_AGENTS {
        return Err(TickWindowError::CountOutOfBounds(u64::from(
            raw_agent_count,
        )));
    }
    if agent_count.saturating_mul(AGENT_MIN_BYTES) > r.bytes.len() {
        return Err(TickWindowError::Truncated);
    }
    let mut agents = Vec::with_capacity(agent_count);
    for _ in 0..agent_count {
        let position = [r.f32()?, r.f32()?, r.f32()?];
        let yaw = r.f32()?;
        if position.iter().any(|v| !v.is_finite()) {
            return Err(TickWindowError::NonFinite);
        }
        let alive = r.u8()? != 0;
        let energy = r.f32()?;
        let integrity = r.f32()?;
        let motor_forward = r.f32()?;
        let motor_turn = r.f32()?;
        let exploration_rate = r.f32()?;
        let prediction_error = r.f32()?;
        let gradient = r.f32()?;
        let raw_gradient = r.f32()?;
        let urgency = r.f32()?;
        let credit_magnitude = r.f32()?;
        let patterns_recalled = r.u16()?;
        let phase = r.u8()?;
        let mean_attenuation = r.f32()?;
        let curiosity_bonus = r.f32()?;
        let fatigue_factor = r.f32()?;
        let staleness = r.f32()?;
        let vision_color = if r.u8()? == 1 {
            let n = usize::try_from(r.u32()?).map_err(|_| TickWindowError::Truncated)?;
            if n.saturating_mul(4) > r.bytes.len() {
                return Err(TickWindowError::Truncated);
            }
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(r.f32()?);
            }
            Some(v)
        } else {
            None
        };
        agents.push(TickRecord {
            position,
            yaw,
            alive,
            energy,
            integrity,
            motor_forward,
            motor_turn,
            exploration_rate,
            prediction_error,
            gradient,
            raw_gradient,
            urgency,
            credit_magnitude,
            patterns_recalled,
            phase,
            mean_attenuation,
            curiosity_bonus,
            fatigue_factor,
            staleness,
            vision_color,
        });
    }
    let raw_food_count = r.u32()?;
    let food_count = usize::try_from(raw_food_count)
        .map_err(|_| TickWindowError::CountOutOfBounds(u64::from(raw_food_count)))?;
    if food_count > MAX_FRAME_FOOD {
        return Err(TickWindowError::CountOutOfBounds(u64::from(raw_food_count)));
    }
    if food_count.saturating_mul(FOOD_BYTES) > r.bytes.len() {
        return Err(TickWindowError::Truncated);
    }
    let mut food = Vec::with_capacity(food_count);
    for _ in 0..food_count {
        let f = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
        if f[..3].iter().any(|v| !v.is_finite()) {
            return Err(TickWindowError::NonFinite);
        }
        food.push(f);
    }
    if !r.bytes.is_empty() {
        return Err(TickWindowError::TrailingBytes);
    }
    Ok(WindowFrame { tick, agents, food })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(tick: u64) -> WindowFrame {
        WindowFrame {
            tick,
            agents: Vec::new(),
            food: Vec::new(),
        }
    }

    fn cfg(window: u64, frames: usize, half: u64, pins: usize, per_pin: usize) -> TickWindowConfig {
        TickWindowConfig {
            window_ticks: window,
            max_frames: frames,
            pin_half_window_ticks: half,
            max_pins: pins,
            max_frames_per_pin: per_pin,
        }
    }

    fn ticks(w: &TickWindow) -> Vec<u64> {
        w.frames().map(|f| f.tick).collect()
    }

    fn sample_frame() -> WindowFrame {
        let mut row = vec![0.0f32; PHYS_STRIDE];
        row[P_POS_X] = 1.5;
        row[P_ALIVE] = 1.0;
        row[P_ENERGY] = 7.0;
        let mut rec = record_from_physics_row(&row);
        rec.vision_color = Some(vec![0.25, 0.5]);
        rec.patterns_recalled = 9;
        WindowFrame {
            tick: 42,
            agents: vec![rec, record_from_physics_row(&row)],
            food: vec![[1.0, 2.0, 3.0, 4.0]],
        }
    }

    #[test]
    fn tick_window_config_validate_rejects_zero_fields() {
        assert!(TickWindowConfig::default().validate().is_ok());
        let mut c = TickWindowConfig::default();
        c.window_ticks = 0;
        assert_eq!(
            c.validate(),
            Err(TickWindowError::InvalidConfig("window_ticks"))
        );
        let mut c = TickWindowConfig::default();
        c.max_frames = 0;
        assert_eq!(
            c.validate(),
            Err(TickWindowError::InvalidConfig("max_frames"))
        );
        let mut c = TickWindowConfig::default();
        c.pin_half_window_ticks = 0;
        assert!(c.validate().is_err());
        let mut c = TickWindowConfig::default();
        c.max_pins = 0;
        assert!(c.validate().is_err());
        let mut c = TickWindowConfig::default();
        c.max_frames_per_pin = 0;
        assert!(c.validate().is_err());
        assert!(TickWindow::new(c).is_err());
    }

    #[test]
    fn tick_window_rolling_eviction_drops_old_frames() {
        let mut w = TickWindow::new(cfg(10, 100, 1, 1, 1)).unwrap();
        for t in [1, 5, 12, 20] {
            assert!(w.capture(frame(t)));
        }
        assert_eq!(ticks(&w), vec![12, 20]);
        assert_eq!(w.len(), 2);
        assert_eq!(w.newest_tick(), Some(20));
    }

    #[test]
    fn tick_window_rolling_cap_drops_oldest() {
        let mut w = TickWindow::new(cfg(1_000, 3, 1, 1, 1)).unwrap();
        for t in 1..=6 {
            w.capture(frame(t));
        }
        assert_eq!(ticks(&w), vec![4, 5, 6]);
    }

    #[test]
    fn tick_window_per_pin_cap_keeps_nearest() {
        let mut w = TickWindow::new(cfg(5, 1, 10, 1, 3)).unwrap();
        w.pin(Pin {
            tick: 5,
            kind: PinKind::SeasonFlip,
        });
        for t in 1..=40 {
            w.capture(frame(t));
        }
        let held = ticks(&w);
        assert!(held.contains(&4) && held.contains(&5) && held.contains(&6));
        assert!(!held.contains(&2));
        assert_eq!(held.len(), 3 + 1);
    }

    #[test]
    fn tick_window_pin_cap_evicts_oldest_pin() {
        let mut w = TickWindow::new(cfg(10, 10, 1, 2, 4)).unwrap();
        for t in [10, 20, 30] {
            w.pin(Pin {
                tick: t,
                kind: PinKind::Replacement { slot: 0 },
            });
        }
        let pins: Vec<u64> = w.pins().iter().map(|p| p.tick).collect();
        assert_eq!(pins, vec![20, 30]);
    }

    #[test]
    fn tick_window_out_of_order_frame_is_ignored() {
        let mut w = TickWindow::new(TickWindowConfig::default()).unwrap();
        assert!(w.capture(frame(10)));
        assert!(!w.capture(frame(10)));
        assert!(!w.capture(frame(5)));
        assert_eq!(ticks(&w), vec![10]);
        assert_eq!(w.frames_in(0, 10).count(), 1);
        assert_eq!(w.frames_in(11, 20).count(), 0);
    }

    #[test]
    fn tick_window_pinned_frames_survive_rolling_eviction() {
        let mut w = TickWindow::new(cfg(10, 100, 3, 4, 100)).unwrap();
        for t in 1..=10 {
            w.capture(frame(t));
        }
        w.pin(Pin {
            tick: 5,
            kind: PinKind::Replacement { slot: 2 },
        });
        for t in 11..=100 {
            w.capture(frame(t));
        }
        let held = ticks(&w);
        for t in 2..=8 {
            assert!(held.contains(&t), "tick {t} missing");
        }
        assert!(!held.contains(&1));
        assert!(!held.contains(&9));
    }

    #[test]
    fn tick_window_duplicate_pin_ignored_slot_ignored() {
        let mut w = TickWindow::new(TickWindowConfig::default()).unwrap();
        w.pin(Pin {
            tick: 5,
            kind: PinKind::Replacement { slot: 1 },
        });
        w.pin(Pin {
            tick: 5,
            kind: PinKind::Replacement { slot: 2 },
        });
        w.pin(Pin {
            tick: 5,
            kind: PinKind::SeasonFlip,
        });
        assert_eq!(w.pins().len(), 2);
    }

    #[test]
    fn tick_window_snapshot_rejects_bad_lengths() {
        assert_eq!(
            frame_from_snapshot(1, &[0.0; PHYS_STRIDE + 1], &[]).unwrap_err(),
            TickWindowError::PhysicsLength(PHYS_STRIDE + 1)
        );
        assert_eq!(
            frame_from_snapshot(1, &[], &[0.0; 3]).unwrap_err(),
            TickWindowError::FoodLength(3)
        );
        let too_many = vec![0.0f32; PHYS_STRIDE * (MAX_FRAME_AGENTS + 1)];
        assert_eq!(
            frame_from_snapshot(1, &too_many, &[]).unwrap_err(),
            TickWindowError::TooManyAgents(MAX_FRAME_AGENTS + 1)
        );
        let too_much = vec![0.0f32; FOOD_STATE_STRIDE * (MAX_FRAME_FOOD + 1)];
        assert_eq!(
            frame_from_snapshot(1, &[], &too_much).unwrap_err(),
            TickWindowError::TooManyFood(MAX_FRAME_FOOD + 1)
        );
        let ok = frame_from_snapshot(3, &[0.0; PHYS_STRIDE * 2], &[1.0; 8]).unwrap();
        assert_eq!((ok.tick, ok.agents.len(), ok.food.len()), (3, 2, 2));
    }

    #[test]
    fn tick_window_record_marks_unrecorded_fields() {
        let mut row = vec![0.0f32; PHYS_STRIDE];
        row[P_ALIVE] = 0.4;
        row[P_YAW] = 2.0;
        let r = record_from_physics_row(&row);
        assert!(!r.alive);
        assert_eq!(r.yaw, 2.0);
        assert_eq!(r.phase, PHASE_NOT_RECORDED);
        assert!(r.vision_color.is_none());
        assert_eq!(r.credit_magnitude, 0.0);
        assert_eq!(r.patterns_recalled, 0);
        row[P_ALIVE] = 0.6;
        assert!(record_from_physics_row(&row).alive);
    }

    #[test]
    fn tick_window_codec_round_trips() {
        let f = sample_frame();
        let bytes = encode_frame(&f);
        let d = decode_frame(&bytes).unwrap();
        assert_eq!(d.tick, 42);
        assert_eq!(d.food, f.food);
        assert_eq!(d.agents.len(), 2);
        assert_eq!(d.agents[0].position, f.agents[0].position);
        assert_eq!(d.agents[0].energy, 7.0);
        assert_eq!(d.agents[0].patterns_recalled, 9);
        assert_eq!(d.agents[0].vision_color, Some(vec![0.25, 0.5]));
        assert!(d.agents[1].vision_color.is_none());
        assert_eq!(encode_frame(&d), bytes);
    }

    #[test]
    fn tick_window_codec_rejects_wrong_version() {
        let mut bytes = encode_frame(&sample_frame());
        bytes[0] = 9;
        assert_eq!(
            decode_frame(&bytes).unwrap_err(),
            TickWindowError::UnsupportedVersion(9)
        );
    }

    #[test]
    fn tick_window_codec_rejects_truncation() {
        let bytes = encode_frame(&sample_frame());
        for cut in [0, 3, 12, 20, bytes.len() - 1] {
            assert_eq!(
                decode_frame(&bytes[..cut]).unwrap_err(),
                TickWindowError::Truncated,
                "cut {cut}"
            );
        }
    }

    #[test]
    fn tick_window_codec_rejects_trailing_bytes() {
        let mut bytes = encode_frame(&sample_frame());
        bytes.push(0);
        assert_eq!(
            decode_frame(&bytes).unwrap_err(),
            TickWindowError::TrailingBytes
        );
    }

    #[test]
    fn tick_window_codec_rejects_oversized_count() {
        let mut bytes = encode_frame(&frame(1));
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            decode_frame(&bytes).unwrap_err(),
            TickWindowError::CountOutOfBounds(u64::from(u32::MAX))
        );
        let mut bytes = encode_frame(&frame(1));
        let n = bytes.len();
        bytes[n - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_frame(&bytes).unwrap_err(),
            TickWindowError::CountOutOfBounds(_)
        ));
    }

    #[test]
    fn tick_window_codec_rejects_non_finite_position() {
        let mut f = sample_frame();
        f.agents[0].position[1] = f32::NAN;
        assert_eq!(
            decode_frame(&encode_frame(&f)).unwrap_err(),
            TickWindowError::NonFinite
        );
        let mut f = sample_frame();
        f.food[0][0] = f32::INFINITY;
        assert_eq!(
            decode_frame(&encode_frame(&f)).unwrap_err(),
            TickWindowError::NonFinite
        );
    }

    #[test]
    fn tick_window_source_names_no_gpu_kernel() {
        let needle = concat!("Gpu", "Kernel");
        for src in [
            include_str!("tick_window.rs"),
            include_str!("tick_window_store.rs"),
        ] {
            let hits = src.matches(needle).count();
            assert_eq!(hits, 0);
        }
    }
}
