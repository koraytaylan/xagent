//! Free-life measurement: per-tick samples of agents living without evolution,
//! the per-agent behavioural and learning metrics computed from them, and the
//! binary tick log that records them.
//!
//! Everything here is pure CPU code with no GPU dependency.

pub mod metrics;
pub mod tick_log;
