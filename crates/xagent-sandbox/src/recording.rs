//! CSV metrics recording for offline analysis and experiment tracking.
//!
//! Logs one row per agent per tick with 29 columns covering brain telemetry,
//! body state, position, biome, motor commands, and behavioral indicators.
//! Files are named with UTC timestamps for easy session identification.

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::SystemTime;

use glam::Vec3;
use xagent_brain::BrainTelemetry;
use xagent_shared::MotorCommand;

use crate::world::biome::BiomeType;

/// CSV metrics logger that records per-tick simulation data.
pub struct MetricsLogger {
    writer: BufWriter<File>,
    pub file_name: String,
    tick_count: u64,
}

const HEADER: &str = "agent_id,tick,prediction_error,avg_prediction_error,memory_utilization,\
    memory_capacity,exploration_rate,homeostatic_gradient,\
    energy,max_energy,integrity,max_integrity,\
    position_x,position_y,position_z,facing_x,facing_z,\
    biome,action_forward,action_strafe,action_turn,action_discrete,alive,\
    exploitation_ratio,decision_quality,behavior_phase,death_count,life_ticks,\
    generation";

impl MetricsLogger {
    pub fn new() -> std::io::Result<Self> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let total_secs = now.as_secs();

        // Derive a human-readable timestamp without chrono.
        let file_name = format!("xagent_log_{}.csv", format_timestamp(total_secs));

        let file = File::create(&file_name)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{}", HEADER)?;
        writer.flush()?;

        Ok(Self {
            writer,
            file_name,
            tick_count: 0,
        })
    }

    /// Append one row of per-tick metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn log_tick(
        &mut self,
        agent_id: u32,
        telemetry: &BrainTelemetry,
        memory_capacity: usize,
        energy: f32,
        max_energy: f32,
        integrity: f32,
        max_integrity: f32,
        position: Vec3,
        facing: Vec3,
        biome: BiomeType,
        motor: &MotorCommand,
        alive: bool,
        death_count: u32,
        life_ticks: u64,
        generation: u32,
    ) -> std::io::Result<()> {
        self.tick_count += 1;

        let biome_str = match biome {
            BiomeType::FoodRich => "FoodRich",
            BiomeType::Barren => "Barren",
            BiomeType::Danger => "Danger",
        };

        let action_str = match &motor.action {
            Some(a) => format!("{:?}", a),
            None => "None".into(),
        };

        writeln!(
            self.writer,
            "{},{},{:.4},{:.4},{:.4},{},{:.4},{:.4},\
             {:.2},{:.2},{:.2},{:.2},\
             {:.3},{:.3},{:.3},{:.3},{:.3},\
             {},{:.3},{:.3},{:.3},{},{},\
             {:.4},{:.4},{},{},{},{}",
            agent_id,
            telemetry.tick,
            telemetry.prediction_error,
            telemetry.avg_prediction_error,
            telemetry.memory_utilization,
            memory_capacity,
            telemetry.exploration_rate,
            telemetry.homeostatic_gradient,
            energy,
            max_energy,
            integrity,
            max_integrity,
            position.x,
            position.y,
            position.z,
            facing.x,
            facing.z,
            biome_str,
            motor.forward,
            motor.strafe,
            motor.turn,
            action_str,
            alive,
            telemetry.exploitation_ratio,
            telemetry.decision_quality,
            telemetry.behavior_phase(),
            death_count,
            life_ticks,
            generation,
        )?;

        // Flush every 100 ticks for crash safety
        if self.tick_count % 100 == 0 {
            self.writer.flush()?;
        }

        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Format epoch seconds as `YYYY-MM-DD_HH-MM-SS` (UTC, no chrono dependency).
fn format_timestamp(epoch_secs: u64) -> String {
    // Days from 1970-01-01
    let secs_per_day: u64 = 86400;
    let mut days = epoch_secs / secs_per_day;
    let day_secs = epoch_secs % secs_per_day;

    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;

    // Compute year/month/day from days since epoch
    let mut year: u64 = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month: u64 = 1;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    let day = days + 1;

    let mut buf = String::with_capacity(20);
    let _ = write!(
        buf,
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        year, month, day, hours, minutes, seconds
    );
    buf
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
