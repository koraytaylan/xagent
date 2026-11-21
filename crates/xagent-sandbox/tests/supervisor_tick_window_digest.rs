//! Capture on and off produce the same supervisor results, and the captured
//! window surrounds every replacement pin.

use rusqlite::types::Value;
use rusqlite::Connection;
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::{run_headless_supervisor_with_tick_window, HeadlessOptions};
use xagent_sandbox::supervisor_checkpoint::decode;
use xagent_sandbox::tick_window::{
    TickWindowConfig, MAX_FRAMES_PER_PIN, MAX_WINDOW_FRAMES, PIN_HALF_WINDOW_TICKS,
};
use xagent_shared::{EvolutionMode, FullConfig, Inheritance, SupervisorConfig};

/// Agents in every run.
const AGENTS: usize = 10;
/// Boundary interval in ticks.
const BOUNDARY_TICKS: u64 = 1_000;
/// Minimum age before an underperformer can be replaced.
const MIN_AGE_TICKS: u64 = 2_000;
/// Score window in ticks.
const WINDOW_TICKS: u64 = 1_000;
/// Season period in ticks.
const SEASON_TICKS: u64 = 3_000;
/// Ticks per agent in a run.
const RUN_TICKS: u64 = 6_000;
/// Agent-tick budget of a run.
const BUDGET: u64 = AGENTS as u64 * RUN_TICKS;
/// Archive capacity of a run.
const ARCHIVE: usize = 8;
/// World seed of a run.
const SEED: u64 = 5;

/// Replay tables, excluded from the non-replay comparison.
const REPLAY_TABLES: [&str; 2] = ["supervisor_replay_frame", "supervisor_replay_pin"];

/// Returns whether the test should run; panics when the GPU is required but absent.
fn gpu_adapter_present_when_required() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        std::env::var("XAGENT_REQUIRE_GPU").is_err(),
        "XAGENT_REQUIRE_GPU is set but no GPU adapter is available"
    );
    println!("skipping: no GPU adapter");
    false
}

fn supervisor_config() -> FullConfig {
    let mut config = FullConfig::default();
    config.governor.evolution_mode = EvolutionMode::Supervisor;
    config.governor.population_size = AGENTS;
    config.governor.supervisor = SupervisorConfig {
        boundary_interval_ticks: BOUNDARY_TICKS,
        min_age_ticks: MIN_AGE_TICKS,
        score_window_ticks: WINDOW_TICKS,
        archive_capacity: ARCHIVE,
        season_period_ticks: SEASON_TICKS,
        inheritance: Inheritance::Darwinian,
        agent_tick_budget: BUDGET,
    };
    config.world.food_kinds_enabled = true;
    config.world.seed = SEED;
    config
}

/// Whether a column holds a wall-clock value: its name ends in `_at`, contains
/// `time`, or is exactly the measured wall-clock throughput column
/// `supervisor_progress.ticks_per_second`, the one extra exclusion the
/// amended packet permits.
fn is_timestamp_column(name: &str) -> bool {
    name.ends_with("_at") || name.contains("time") || name == "ticks_per_second"
}

/// The checkpoint payload with only the wall-clock field
/// `last_ticks_per_second_bits` and the `last_frame` progress
/// `ticks_per_second` of the `run` section masked; every other
/// section and field stays in the comparison.
fn mask_checkpoint(payload: &[u8]) -> Vec<u8> {
    let mut checkpoint = decode(payload).expect("decode checkpoint");
    let mut out = checkpoint.tick.to_le_bytes().to_vec();
    for (name, bytes) in &mut checkpoint.sections {
        if name == "run" {
            let mut json: serde_json::Value = serde_json::from_slice(bytes).expect("run json");
            json["last_ticks_per_second_bits"] = serde_json::Value::Null;
            if let Some(progress) = json["last_frame"].get_mut("progress") {
                progress["ticks_per_second"] = serde_json::Value::Null;
            }
            *bytes = serde_json::to_vec(&json).expect("run json bytes");
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Names the checkpoint sections (and `run` fields) that differ between runs.
fn checkpoint_difference(a: &Connection, b: &Connection) -> String {
    let load = |db: &Connection| {
        let payload: Vec<u8> = db
            .query_row("SELECT payload FROM supervisor_checkpoint", [], |r| {
                r.get(0)
            })
            .expect("payload");
        decode(&payload).expect("decode")
    };
    let (a, b) = (load(a), load(b));
    let mut out = String::new();
    for ((name, x), (_, y)) in a.sections.iter().zip(&b.sections) {
        if x == y {
            continue;
        }
        out.push_str(name);
        if name == "run" {
            let x: serde_json::Value = serde_json::from_slice(x).expect("json");
            let y: serde_json::Value = serde_json::from_slice(y).expect("json");
            for (key, value) in x.as_object().expect("object") {
                if *value != y[key] {
                    out.push('[');
                    out.push_str(key);
                    out.push(']');
                }
            }
        }
        out.push(' ');
    }
    out
}

/// All rows of `table` without timestamp columns; the checkpoint payload is
/// compared with only its wall-clock field masked.
fn rows(db: &Connection, table: &str) -> Vec<Vec<Value>> {
    let mut columns = db
        .prepare(&format!("SELECT * FROM {table}"))
        .expect("prepare")
        .column_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let has_id = columns.iter().any(|c| c == "id");
    columns.retain(|c| !is_timestamp_column(c));
    let order = if has_id { "id" } else { "rowid" };
    let masked = columns
        .iter()
        .position(|c| table == "supervisor_checkpoint" && c == "payload");
    let sql = format!(
        "SELECT {} FROM {table} ORDER BY {order}",
        columns.join(", ")
    );
    let mut stmt = db.prepare(&sql).expect("prepare rows");
    let count = columns.len();
    stmt.query_map([], |row| {
        (0..count)
            .map(|i| {
                let value = row.get::<_, Value>(i)?;
                Ok::<_, rusqlite::Error>(match (Some(i) == masked, value) {
                    (true, Value::Blob(bytes)) => Value::Blob(mask_checkpoint(&bytes)),
                    (_, value) => value,
                })
            })
            .collect()
    })
    .expect("query")
    .map(|r| r.expect("row"))
    .collect()
}

fn table_names(db: &Connection) -> Vec<String> {
    let mut stmt = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .expect("prepare tables");
    stmt.query_map([], |row| row.get::<_, String>(0))
        .expect("tables")
        .map(|r| r.expect("name"))
        .collect()
}

/// The supervisor digest inputs: the event, archive and slot rows.
fn digest(db: &Connection) -> (Vec<Vec<Value>>, Vec<Vec<Value>>, Vec<Vec<Value>>) {
    (
        rows(db, "supervisor_event"),
        rows(db, "supervisor_archive"),
        rows(db, "supervisor_slot"),
    )
}

struct Run {
    db: Connection,
    summary: Vec<u8>,
    _dir: tempfile::TempDir,
}

fn run(window: Option<TickWindowConfig>) -> Run {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("run.db");
    let summary_path = dir.path().join("summary.json");
    let options = HeadlessOptions {
        run_summary: Some(summary_path.clone()),
        ..HeadlessOptions::default()
    };
    run_headless_supervisor_with_tick_window(
        supervisor_config(),
        db_path.to_str().expect("utf8 path"),
        &options,
        window,
    )
    .expect("supervisor run");
    Run {
        db: Connection::open(&db_path).expect("open"),
        summary: std::fs::read(&summary_path).expect("summary written"),
        _dir: dir,
    }
}

fn pins(db: &Connection) -> Vec<(u64, String)> {
    let mut stmt = db
        .prepare("SELECT tick, kind FROM supervisor_replay_pin ORDER BY rowid")
        .expect("prepare pins");
    stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })
    .expect("pins")
    .map(|r| {
        let (tick, kind) = r.expect("pin");
        (u64::try_from(tick).expect("tick"), kind)
    })
    .collect()
}

fn frame_ticks(db: &Connection) -> Vec<u64> {
    let mut stmt = db
        .prepare("SELECT tick FROM supervisor_replay_frame ORDER BY tick")
        .expect("prepare frames");
    stmt.query_map([], |row| row.get::<_, i64>(0))
        .expect("frames")
        .map(|r| u64::try_from(r.expect("tick")).expect("tick"))
        .collect()
}

#[test]
fn capture_on_and_off_agree() {
    if !gpu_adapter_present_when_required() {
        return;
    }
    let off = run(None);
    let on = run(Some(TickWindowConfig::default()));
    let off_digest = digest(&off.db);
    assert!(!off_digest.0.is_empty(), "the run recorded events");
    assert!(off_digest == digest(&on.db), "supervisor digest differs");
    let tables = table_names(&off.db)
        .into_iter()
        .filter(|t| !REPLAY_TABLES.contains(&t.as_str()))
        .collect::<Vec<_>>();
    assert!(!tables.is_empty());
    for table in &tables {
        assert!(
            rows(&off.db, table) == rows(&on.db, table),
            "table {table} differs: {}",
            if table == "supervisor_checkpoint" {
                checkpoint_difference(&off.db, &on.db)
            } else {
                String::new()
            }
        );
    }
    assert!(off.summary == on.summary, "run summary bytes differ");
    let pins = pins(&on.db);
    assert!(
        pins.iter().any(|(_, kind)| kind == "replacement"),
        "a replacement pin was recorded: {pins:?}"
    );
    assert!(
        pins.iter().any(|(_, kind)| kind == "season"),
        "a season-flip pin was recorded: {pins:?}"
    );
}

#[test]
fn captured_frames_are_bounded_and_surround_replacement_pins() {
    if !gpu_adapter_present_when_required() {
        return;
    }
    let on = run(Some(TickWindowConfig::default()));
    let frames = frame_ticks(&on.db);
    assert!(!frames.is_empty() && frames.len() <= MAX_WINDOW_FRAMES);
    let pins = pins(&on.db);
    let replacements = pins
        .iter()
        .filter(|(_, kind)| kind == "replacement")
        .collect::<Vec<_>>();
    assert!(!replacements.is_empty(), "a replacement pin was recorded");
    for (tick, _) in replacements {
        let lo = tick.saturating_sub(PIN_HALF_WINDOW_TICKS);
        let hi = tick + PIN_HALF_WINDOW_TICKS;
        let near = frames.iter().filter(|t| (lo..=hi).contains(t)).count();
        assert!(near >= 1, "pin at {tick} has frames in [{lo}, {hi}]");
        assert!(near <= MAX_FRAMES_PER_PIN, "pin at {tick} within the cap");
    }
}

#[test]
fn capture_sample_names_no_kernel_and_follows_on_sample() {
    let source = include_str!("../src/headless.rs");
    let start = source.find("fn capture_sample(").expect("capture_sample");
    let body_end = source[start..].find("\n}\n").expect("closing brace") + start;
    assert!(
        !source[start..body_end].contains("kernel"),
        "capture_sample names no kernel"
    );
    let step = source.find("fn supervisor_step(").expect("supervisor_step");
    let step_end = source[step..].find("\n}\n").expect("step closing brace") + step;
    let body = &source[step..step_end];
    let on_sample = body.find(".on_sample(").expect("on_sample");
    let call = body
        .find("capture_sample(window")
        .expect("call site in supervisor_step");
    assert!(call > on_sample, "capture_sample is called after on_sample");
    let mut seen = 0;
    let mut rest = body;
    while let Some(at) = rest.find(".on_sample(") {
        let after = &rest[at..];
        let next = after
            .find("capture_sample(window")
            .expect("capture after each on_sample");
        assert!(
            !after[..next].contains("Ok("),
            "every on_sample is followed by capture before returning"
        );
        seen += 1;
        rest = &after[next..];
    }
    assert!(seen >= 1, "supervisor_step calls on_sample");
    assert!(
        !source[step..step + on_sample].contains("capture_sample"),
        "the step does not capture before on_sample"
    );
}
