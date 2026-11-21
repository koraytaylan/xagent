//! Seeded headless supervisor runs are bit-reproducible: a run is reduced to an
//! FNV-1a digest of its stored supervisor tables, the digest is equal across
//! recorded and unrecorded runs, and it matches the captured fixture on the
//! adapter it was captured on.

use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use xagent_brain::GpuKernel;
use xagent_sandbox::headless::HeadlessOptions;
use xagent_shared::{EvolutionMode, FullConfig, Inheritance};

/// World seed of the recorded run.
const SEED: u64 = 5;
/// Population of the recorded run.
const POPULATION: usize = 10;
/// Total agent-tick budget of the recorded run.
const AGENT_TICK_BUDGET: u64 = 600_000;
/// Minimum archive rows a digested run must hold.
const MIN_ARCHIVE_ROWS: i64 = 2;
/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Number of hex digits in a rendered digest.
const DIGEST_HEX_LEN: usize = 16;
/// Busy timeout for opening the finished database, in milliseconds.
const BUSY_TIMEOUT_MS: u64 = 5000;
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
/// FNV-1a 64-bit of the single byte `a`.
const FNV_OF_A: u64 = 0xaf63_dc4c_8601_ec8c;

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
            ValueRef::Integer(i) => {
                self.write(&[TAG_INTEGER]);
                self.write(&i.to_le_bytes());
            }
            ValueRef::Real(r) => {
                self.write(&[TAG_REAL]);
                self.write(&r.to_bits().to_le_bytes());
            }
            ValueRef::Text(t) => {
                self.write(&[TAG_TEXT]);
                self.write_len(t.len());
                self.write(t);
            }
            ValueRef::Blob(b) => {
                self.write(&[TAG_BLOB]);
                self.write_len(b.len());
                self.write(b);
            }
        }
    }
}

/// FNV-1a 64-bit of `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.write(bytes);
    h.0
}

/// Hashes a section tag followed by every row of `sql`, skipping the columns
/// named in `skip`.
fn hash_section(h: &mut Fnv1a, conn: &Connection, tag: &str, sql: &str, skip: &[&str]) {
    h.write(tag.as_bytes());
    let mut stmt = conn.prepare(sql).expect("prepare digest query");
    let keep: Vec<usize> = (0..stmt.column_count())
        .filter(|&i| !skip.contains(&stmt.column_name(i).expect("column name")))
        .collect();
    let mut rows = stmt.query([]).expect("run digest query");
    while let Some(row) = rows.next().expect("read digest row") {
        for &i in &keep {
            h.write_value(row.get_ref(i).expect("read digest value"));
        }
    }
}

/// The seeded supervisor configuration of the recorded run.
fn seeded_config() -> FullConfig {
    let mut config = FullConfig::default();
    config.world.seed = SEED;
    config.governor.evolution_mode = EvolutionMode::Supervisor;
    config.governor.population_size = POPULATION;
    config.governor.supervisor.agent_tick_budget = AGENT_TICK_BUDGET;
    config
}

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
    let mut h = Fnv1a::new();
    hash_section(
        &mut h,
        &conn,
        "supervisor_slot",
        "SELECT * FROM supervisor_slot ORDER BY slot",
        &["run_id"],
    );
    hash_section(
        &mut h,
        &conn,
        "supervisor_archive",
        "SELECT * FROM supervisor_archive ORDER BY entry_id",
        &["id", "run_id"],
    );
    hash_section(
        &mut h,
        &conn,
        "supervisor_event",
        "SELECT * FROM supervisor_event ORDER BY id",
        &["id", "run_id"],
    );
    format!("{:016x}", h.0)
}

/// Runs the seeded supervisor into a fresh temporary database and returns its digest.
fn run_in_temp_dir(record: bool) -> String {
    let dir = tempfile::TempDir::new().expect("create temporary directory");
    let db_path = dir.path().join("run.db");
    let path = db_path.to_str().expect("utf-8 database path");
    xagent_sandbox::headless::run_supervisor_with_recording(
        seeded_config(),
        path,
        &HeadlessOptions::default(),
        record,
    )
    .expect("seeded supervisor run");
    digest_database(&db_path)
}

/// Whether environment variable `name` is set to `1`.
fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

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

/// Whether `s` is exactly 16 lowercase hex digits.
fn is_digest_hex(s: &str) -> bool {
    s.len() == DIGEST_HEX_LEN && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[test]
fn fnv1a_matches_reference_vectors() {
    assert_eq!(fnv1a(b""), FNV_OFFSET_BASIS);
    assert_eq!(fnv1a(b"a"), FNV_OF_A);
}

#[test]
fn supervisor_digest_is_reproducible_and_recording_independent() {
    assert!(matches!(
        seeded_config().governor.supervisor.inheritance,
        Inheritance::WeightCopy
    ));
    if !GpuKernel::is_available() {
        assert!(
            !env_flag("XAGENT_REQUIRE_GPU"),
            "no GPU adapter but XAGENT_REQUIRE_GPU=1"
        );
        println!("skipping supervisor digest: no GPU adapter");
        return;
    }
    let first = run_in_temp_dir(true);
    let second = run_in_temp_dir(true);
    let unrecorded = run_in_temp_dir(false);
    assert_eq!(first, second, "two recorded runs gave different digests");
    assert_eq!(
        first, unrecorded,
        "recorded and unrecorded runs gave different digests"
    );
    assert!(is_digest_hex(&first), "digest is not 16 hex digits");
    println!("digest: {first}");

    if env_flag("XAGENT_CAPTURE_SUPERVISOR_DIGEST") {
        println!("adapter: {}", adapter_description());
        return;
    }

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
    if adapter == recorded_adapter || env_flag("XAGENT_REQUIRE_SEEDED_DIGEST") {
        assert_eq!(
            first, recorded,
            "computed digest {first} differs from recorded digest {recorded}"
        );
    } else {
        println!(
            "adapter differs ({adapter} vs recorded {recorded_adapter}): skipping fixture comparison"
        );
    }
}
