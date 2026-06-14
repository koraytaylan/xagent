//! Replay recording and playback coordination for the live sandbox loop.
//!
//! Hosts `App::record_replay_tick` (append the current frame's per-agent state
//! to the active `GenerationRecording`) and `App::advance_replay_playback`
//! (advance the playback cursor while a finished recording is reviewed). The
//! on-disk recording format itself lives in the library `replay` module.

use xagent_sandbox::replay::{GenerationRecording, TickRecord};

use crate::app::App;

impl App {
    /// Append the latest published per-agent state to the active recording.
    ///
    /// Called once per *collected state snapshot* (bounded by
    /// `STATE_SNAPSHOT_MAX_HZ`), not once per simulated tick: like the CPU
    /// sparkline histories, the replay samples the latest published state, so a
    /// generation's recording is a cadence-sampled trace rather than a dense
    /// per-tick log.
    ///
    /// Position/yaw/alive/energy/integrity come from the latest async physics
    /// readback stored on each agent body. Motor outputs, gradient/urgency, and
    /// the exploration/prediction/attenuation/curiosity/fatigue fields come from
    /// the per-agent `cached_*` telemetry; in this GPU path only the selected
    /// agent's cache is refreshed each frame. `credit_magnitude`,
    /// `patterns_recalled`, and `vision_color` are left at defaults and `phase`
    /// is recorded as `RANDOM`, since this GPU path tracks none of them;
    /// `raw_gradient` mirrors `gradient` because the GPU readback exposes no
    /// separate raw gradient. No-op when no recording is active.
    pub(crate) fn record_replay_tick(&mut self) {
        let Some(rec) = self.recording.as_mut() else {
            return;
        };
        // Async readback snapshots arrive independently of governor tick
        // advancement, so record them at the next dense replay index.
        let tick = rec.total_ticks;
        let records: Vec<TickRecord> = self
            .agents
            .iter()
            .map(|a| TickRecord {
                position: [
                    a.body.body.position.x,
                    a.body.body.position.y,
                    a.body.body.position.z,
                ],
                yaw: a.body.yaw,
                alive: a.body.body.alive,
                energy: a.body.body.internal.energy,
                integrity: a.body.body.internal.integrity,
                motor_forward: a.cached_motor.forward,
                motor_turn: a.cached_motor.turn,
                exploration_rate: a.cached_exploration_rate,
                prediction_error: a.cached_prediction_error,
                gradient: a.cached_gradient,
                raw_gradient: a.cached_gradient,
                urgency: a.cached_urgency,
                credit_magnitude: 0.0,
                patterns_recalled: 0,
                phase: GenerationRecording::phase_to_u8("RANDOM"),
                mean_attenuation: a.cached_mean_attenuation,
                curiosity_bonus: a.cached_curiosity_bonus,
                fatigue_factor: a.cached_fatigue_factor,
                staleness: a.cached_staleness,
                vision_color: None,
            })
            .collect();
        rec.record_tick(tick, &records);
    }

    /// Advance replay playback by the configured speed while a recording plays,
    /// clamping at the final tick and stopping playback once it is reached.
    pub(crate) fn advance_replay_playback(&mut self) {
        if !(self.replay_state.active && self.replay_state.playing) {
            return;
        }
        let advance = (self.replay_state.speed as u64).max(1);
        self.replay_state.current_tick = (self.replay_state.current_tick + advance)
            .min(self.replay_state.total_ticks.saturating_sub(1));
        if self.replay_state.current_tick >= self.replay_state.total_ticks.saturating_sub(1) {
            self.replay_state.playing = false;
        }
    }
}
