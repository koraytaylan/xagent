//! Per-tick snapshot assembly helpers for the egui/sparkline UI.
//!
//! Currently hosts `record_agent_histories`, which appends the most recent
//! cached telemetry values to an agent's bounded sparkline history buffers.

use xagent_sandbox::agent::Agent;

/// Record per-tick telemetry into agent sparkline histories.
pub(crate) fn record_agent_histories(agent: &mut Agent) {
    let cap = 10_000;
    macro_rules! push_hist {
        ($h:expr, $v:expr) => {
            if $h.len() >= cap {
                $h.pop_front();
            }
            $h.push_back($v);
        };
    }
    push_hist!(
        agent.prediction_error_history,
        agent.cached_prediction_error
    );
    push_hist!(
        agent.exploration_rate_history,
        agent.cached_exploration_rate
    );
    let ef = agent.body.body.internal.energy / agent.body.body.internal.max_energy.max(0.001);
    push_hist!(agent.energy_history, ef.clamp(0.0, 1.0));
    let inf =
        agent.body.body.internal.integrity / agent.body.body.internal.max_integrity.max(0.001);
    push_hist!(agent.integrity_history, inf.clamp(0.0, 1.0));
    push_hist!(agent.fatigue_history, agent.cached_fatigue_factor);
}
