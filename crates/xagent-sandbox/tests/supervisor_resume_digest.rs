//! Stop-and-resume digest and refusal gate: a supervisor run stopped after a
//! checkpoint and resumed has the digest of the uninterrupted run, which equals
//! the recorded golden, and damaged or mismatched checkpoints are refused with
//! the matching error while the other tables stay unchanged.

use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{params, Connection, OpenFlags};
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::{run_headless_with_options, HeadlessError, HeadlessOptions};
use xagent_sandbox::supervisor_checkpoint::CheckpointError;
use xagent_shared::{EvolutionMode, FullConfig};

/// World seed of the run.
const SEED: u64 = 5;
/// Population of the run.
const AGENTS: usize = 10;
/// Total agent-tick budget of the run.
const AGENT_TICK_BUDGET: u64 = 600_000;
/// Tick at or after which the stop run stops at a checkpoint.
const STOP_AT_TICK: u64 = 10_000;
/// Version a mismatching checkpoint claims.
const OTHER_VERSION: u32 = 2;
/// Version the reader expects.
const EXPECTED_VERSION: u32 = 1;
/// Byte range of the format version in a payload.
const VERSION_RANGE: std::ops::Range<usize> = 8..12;
/// Trailer length of a payload (the checksum).
const TRAILER_LEN: usize = 8;
/// World seed written into a mismatching stored config.
const OTHER_SEED: u64 = 6;
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Number of hex digits in a rendered digest.
const DIGEST_HEX_LEN: usize = 16;
/// Busy timeout for opening the finished database, in milliseconds.
const BUSY_TIMEOUT_MS: u64 = 5000;
/// Minimum archive rows a digested run must hold.
const MIN_ARCHIVE_ROWS: i64 = 2;
/// Value tag for SQL NULL.
const TAG_NULL: u8 = 0;
/// Value tag for SQL INTEGER.
const TAG_INTEGER: u8 = 1;
/// Value tag for SQL REAL.
const TAG_REAL: u8 = 2;
/// Value tag for SQL TEXT.
const TAG_TEXT: u8 = 3;
/// Value tag for SQL BLOB.
const TAG_BLOB: u8 = 4;
/// Recorded fixture, relative to the crate root.
const FIXTURE_PATH: &str = "tests/fixtures/supervisor_digest.json";
/// Divisor that picks the payload midpoint for truncation and corruption.
const TRUNCATE_DIVISOR: usize = 2;
/// Mask that flips every bit of the corrupted payload byte.
const CORRUPT_MASK: u8 = 0xff;

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Incremental FNV-1a 64-bit hasher.
struct Fnv1a(u64);

impl Fnv1a {
    /// Starts a hash at the offset basis.
    fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    /// Mixes `bytes` into the hash.
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    /// Mixes a length as a little-endian `u64`.
    fn write_len(&mut self, len: usize) {
        let len: u64 = len.try_into().expect("length fits u64");
        self.write(&len.to_le_bytes());
    }

    /// Mixes one tagged SQL value.
    fn write_value(&mut self, value: ValueRef<'_>) {
        match value {
            ValueRef::Null => self.write(&[TAG_NULL]),
            ValueRef::Integer(integer) => {
                self.write(&[TAG_INTEGER]);
                self.write(&integer.to_le_bytes());
            }
            ValueRef::Real(real) => {
                self.write(&[TAG_REAL]);
                self.write(&real.to_bits().to_le_bytes());
            }
            ValueRef::Text(text) => {
                self.write(&[TAG_TEXT]);
                self.write_len(text.len());
                self.write(text);
            }
            ValueRef::Blob(blob) => {
                self.write(&[TAG_BLOB]);
                self.write_len(blob.len());
                self.write(blob);
            }
        }
    }
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// FNV-1a 64-bit of `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hasher = Fnv1a::new();
    hasher.write(bytes);
    hasher.0
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Hashes a section tag followed by every row of `sql`, skipping the columns
/// named in `skip`.
fn hash_section(hasher: &mut Fnv1a, conn: &Connection, tag: &str, sql: &str, skip: &[&str]) {
    hasher.write(tag.as_bytes());
    let mut stmt = conn.prepare(sql).expect("prepare digest query");
    let keep: Vec<usize> = (0..stmt.column_count())
        .filter(|&i| !skip.contains(&stmt.column_name(i).expect("column name")))
        .collect();
    let mut rows = stmt.query([]).expect("run digest query");
    while let Some(row) = rows.next().expect("read digest row") {
        for &i in &keep {
            hasher.write_value(row.get_ref(i).expect("read digest value"));
        }
    }
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Reduces the supervisor tables in the database at `db_path` to a digest.
fn digest_database(db_path: &Path) -> String {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("open finished database");
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))
        .expect("set busy timeout");
    let archive_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM supervisor_archive", [], |r| r.get(0))
        .expect("count archive rows");
    assert!(
        archive_rows >= MIN_ARCHIVE_ROWS,
        "supervisor archive holds {archive_rows} rows, expected at least {MIN_ARCHIVE_ROWS}"
    );
    let mut hasher = Fnv1a::new();
    hash_section(
        &mut hasher,
        &conn,
        "supervisor_slot",
        "SELECT * FROM supervisor_slot ORDER BY slot",
        &["run_id"],
    );
    hash_section(
        &mut hasher,
        &conn,
        "supervisor_archive",
        "SELECT * FROM supervisor_archive ORDER BY entry_id",
        &["id", "run_id"],
    );
    hash_section(
        &mut hasher,
        &conn,
        "supervisor_event",
        "SELECT * FROM supervisor_event ORDER BY id",
        &["id", "run_id"],
    );
    format!("{:016x}", hasher.0)
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// The seeded supervisor configuration of the recorded run.
fn seeded_config() -> FullConfig {
    let mut config = FullConfig::default();
    config.world.seed = SEED;
    config.governor.evolution_mode = EvolutionMode::Supervisor;
    config.governor.population_size = AGENTS;
    config.governor.supervisor.agent_tick_budget = AGENT_TICK_BUDGET;
    config
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Whether environment variable `name` is set to `1`.
fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Describes the GPU adapter as `name (backend)`.
fn adapter_description() -> String {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("GPU adapter");
    let info = adapter.get_info();
    format!("{} ({:?})", info.name, info.backend)
}

// Copied from crates/xagent-sandbox/tests/supervisor_digest.rs.
/// Whether `text` is exactly 16 lowercase hex digits.
fn is_digest_hex(text: &str) -> bool {
    text.len() == DIGEST_HEX_LEN && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

// Copied from crates/xagent-sandbox/tests/supervisor_headless.rs.
/// Panics without an adapter when `XAGENT_REQUIRE_GPU=1`; otherwise reports
/// whether to run.
fn gpu_or_skip() -> bool {
    if GpuKernel::is_available() {
        return true;
    }
    assert!(
        !env_flag("XAGENT_REQUIRE_GPU"),
        "XAGENT_REQUIRE_GPU=1 but no GPU adapter is available"
    );
    println!("skipping: no GPU adapter");
    false
}

/// Runs the seeded supervisor uninterrupted into `db`.
fn run_uninterrupted(db: &Path) {
    run_headless_with_options(
        seeded_config(),
        db.to_str().expect("utf-8 path"),
        false,
        &HeadlessOptions::default(),
    )
    .expect("uninterrupted run");
}

/// Runs the seeded supervisor into `db`, stopping at the first checkpoint at or
/// after `STOP_AT_TICK`.
fn run_stop(db: &Path) {
    let options = HeadlessOptions {
        stop_after_checkpoint_at_or_after: Some(STOP_AT_TICK),
        ..HeadlessOptions::default()
    };
    run_headless_with_options(
        seeded_config(),
        db.to_str().expect("utf-8 path"),
        false,
        &options,
    )
    .expect("stopped run");
}

/// Runs the stop run and then resumes it to the end.
fn run_stop_then_resume(db: &Path) {
    run_stop(db);
    run_headless_with_options(
        seeded_config(),
        db.to_str().expect("utf-8 path"),
        true,
        &HeadlessOptions::default(),
    )
    .expect("resumed run");
}

/// Opens `db` read-write.
fn open_rw(db: &Path) -> Connection {
    let conn = Connection::open(db).expect("open database");
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))
        .expect("set busy timeout");
    conn
}

/// The payload of the single checkpoint row.
fn checkpoint_payload(db: &Path) -> Vec<u8> {
    open_rw(db)
        .query_row("SELECT payload FROM supervisor_checkpoint", [], |r| {
            r.get(0)
        })
        .expect("read checkpoint payload")
}

/// Replaces the payload of the single checkpoint row.
fn set_payload(db: &Path, payload: &[u8]) {
    let changed = open_rw(db)
        .execute(
            "UPDATE supervisor_checkpoint SET payload = ?1",
            params![payload],
        )
        .expect("write checkpoint payload");
    assert_eq!(changed, 1, "exactly one checkpoint row");
}

/// The `world_config` text of the single run row.
fn world_config(db: &Path) -> String {
    open_rw(db)
        .query_row("SELECT world_config FROM run", [], |r| r.get(0))
        .expect("read world_config")
}

/// Replaces the `world_config` text of the single run row.
fn set_world_config(db: &Path, text: &str) {
    let changed = open_rw(db)
        .execute("UPDATE run SET world_config = ?1", params![text])
        .expect("write world_config");
    assert_eq!(changed, 1, "exactly one run row");
}

/// Hash of every table except `supervisor_checkpoint`.
fn table_snapshot(db: &Path) -> u64 {
    let conn = open_rw(db);
    let tables: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_master WHERE type = 'table' \
                 AND name <> 'supervisor_checkpoint' ORDER BY name",
            )
            .expect("prepare table list");
        let names = stmt
            .query_map([], |r| r.get(0))
            .expect("list tables")
            .map(|r| r.expect("table name"))
            .collect();
        names
    };
    let mut hasher = Fnv1a::new();
    for t in &tables {
        hash_section(
            &mut hasher,
            &conn,
            t,
            &format!("SELECT * FROM \"{t}\" ORDER BY rowid"),
            &[],
        );
    }
    hasher.0
}

/// Resumes `db` and returns the checkpoint error it is refused with.
fn refusal(db: &Path) -> CheckpointError {
    let result = run_headless_with_options(
        seeded_config(),
        db.to_str().expect("utf-8 path"),
        true,
        &HeadlessOptions::default(),
    );
    match result {
        Err(HeadlessError::Checkpoint(error)) => error,
        Err(other) => panic!("expected a checkpoint refusal, got error: {other}"),
        Ok(()) => panic!("expected a checkpoint refusal, resume succeeded"),
    }
}

/// Stops a run, applies `mutate` to the payload, and asserts the resume is
/// refused with `expected` and that nothing else changed.
fn assert_payload_refusal(mutate: impl FnOnce(&mut Vec<u8>), expected: &CheckpointError) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("run.db");
    run_stop(&db);
    let mut payload = checkpoint_payload(&db);
    mutate(&mut payload);
    set_payload(&db, &payload);
    let before = table_snapshot(&db);
    let err = refusal(&db);
    assert_eq!(&err, expected);
    assert!(err.to_string().contains(&expected.to_string()));
    assert_eq!(table_snapshot(&db), before, "database tables changed");
    assert_eq!(checkpoint_payload(&db), payload, "payload changed");
}

#[test]
fn resume_digest_matches_uninterrupted() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let uninterrupted_db = dir.path().join("a.db");
    let resumed_db = dir.path().join("b.db");
    run_uninterrupted(&uninterrupted_db);
    run_stop_then_resume(&resumed_db);
    let digest_a = digest_database(&uninterrupted_db);
    let digest_b = digest_database(&resumed_db);
    assert_eq!(
        digest_b, digest_a,
        "resumed digest differs from uninterrupted"
    );
}

#[test]
fn resume_digest_checkpointing_does_not_change_digest() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let checkpointing_db = dir.path().join("a.db");
    run_uninterrupted(&checkpointing_db);
    let digest = digest_database(&checkpointing_db);
    assert!(is_digest_hex(&digest), "digest is not 16 hex digits");

    let fixture_file = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);
    let text = std::fs::read_to_string(&fixture_file).expect("read digest fixture");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("parse digest fixture");
    assert_eq!(fixture["seed"].as_u64(), Some(SEED));
    assert_eq!(
        fixture["agent_tick_budget"].as_u64(),
        Some(AGENT_TICK_BUDGET)
    );
    assert_eq!(fixture["inheritance"].as_str(), Some("weight_copy"));
    let recorded = fixture["digest"].as_str().expect("fixture digest string");
    assert!(
        is_digest_hex(recorded),
        "fixture digest is not 16 hex digits"
    );
    let recorded_adapter = fixture["adapter"].as_str().expect("fixture adapter string");
    let adapter = adapter_description();
    // The fixture was recorded by code that wrote no checkpoints, so equality
    // shows that checkpoint writes change no decision.
    if adapter == recorded_adapter || env_flag("XAGENT_REQUIRE_SEEDED_DIGEST") {
        assert_eq!(
            digest, recorded,
            "computed digest {digest} differs from recorded digest {recorded}"
        );
    } else {
        println!(
            "adapter differs ({adapter} vs recorded {recorded_adapter}): skipping fixture comparison"
        );
    }
}

#[test]
fn resume_refuses_truncated() {
    if !gpu_or_skip() {
        return;
    }
    assert_payload_refusal(
        |p| p.truncate(p.len() / TRUNCATE_DIVISOR),
        &CheckpointError::Truncated,
    );
}

#[test]
fn resume_refuses_corrupt_byte() {
    if !gpu_or_skip() {
        return;
    }
    assert_payload_refusal(
        |p| {
            let mid = p.len() / TRUNCATE_DIVISOR;
            p[mid] ^= CORRUPT_MASK;
        },
        &CheckpointError::ChecksumMismatch,
    );
}

#[test]
fn resume_refuses_version_mismatch() {
    if !gpu_or_skip() {
        return;
    }
    assert_payload_refusal(
        |p| {
            p[VERSION_RANGE].copy_from_slice(&OTHER_VERSION.to_le_bytes());
            let body = p.len() - TRAILER_LEN;
            let sum = fnv1a(&p[..body]).to_le_bytes();
            p[body..].copy_from_slice(&sum);
        },
        &CheckpointError::VersionMismatch {
            found: OTHER_VERSION,
            expected: EXPECTED_VERSION,
        },
    );
}

#[test]
fn resume_refuses_config_mismatch() {
    if !gpu_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("run.db");
    run_stop(&db);
    let mut value: serde_json::Value =
        serde_json::from_str(&world_config(&db)).expect("parse stored world config");
    let seed = value
        .get_mut("seed")
        .expect("stored world config has a seed key");
    assert_eq!(seed.as_u64(), Some(SEED));
    *seed = serde_json::Value::from(OTHER_SEED);
    let mutated = serde_json::to_string(&value).expect("serialise world config");
    set_world_config(&db, &mutated);
    let payload = checkpoint_payload(&db);
    let before = table_snapshot(&db);
    let err = refusal(&db);
    assert_eq!(err, CheckpointError::ConfigMismatch);
    assert!(err
        .to_string()
        .contains(&CheckpointError::ConfigMismatch.to_string()));
    assert_eq!(table_snapshot(&db), before, "database tables changed");
    assert_eq!(world_config(&db), mutated, "world_config changed");
    assert_eq!(checkpoint_payload(&db), payload, "payload changed");
}
