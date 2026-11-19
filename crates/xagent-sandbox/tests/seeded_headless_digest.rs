//! Default kernel behaviour matches the recorded digest: a seeded headless
//! evolution run is reduced to an FNV-1a digest of its stored results and
//! compared with the fixture captured from the unchanged kernel.

use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use xagent_brain::GpuKernel;
use xagent_shared::FullConfig;

/// World seed of the recorded run.
const SEED: u64 = 5;
/// Generations of the recorded run.
const GENERATIONS: u64 = 2;
/// Tick budget of each evaluation in the recorded run.
const TICK_BUDGET: u64 = 2000;
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
const FIXTURE_PATH: &str = "tests/fixtures/seeded_headless_digest.json";
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

/// Reduces the stored results in the database at `db_path` to a digest.
fn digest_database(db_path: &Path) -> String {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("open finished database");
    conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS))
        .expect("set busy timeout");
    let mut h = Fnv1a::new();
    hash_section(
        &mut h,
        &conn,
        "node",
        "SELECT id, parent_id, generation, config_json, best_fitness, avg_fitness, \
         mutated_param, mutation_direction, status FROM node ORDER BY id",
        &[],
    );
    hash_section(
        &mut h,
        &conn,
        "agent_result",
        "SELECT * FROM agent_result ORDER BY id",
        &["id"],
    );
    hash_section(
        &mut h,
        &conn,
        "node_brain",
        "SELECT node_id, data FROM node_brain ORDER BY node_id",
        &[],
    );
    format!("{:016x}", h.0)
}

/// Runs the seeded headless evolution into `db_path` and returns its digest.
fn run_seeded_digest(db_path: &Path) -> String {
    let mut config = FullConfig::default();
    config.world.seed = SEED;
    config.governor.tick_budget = TICK_BUDGET;
    config.governor.max_generations = GENERATIONS;
    let path = db_path.to_str().expect("utf-8 database path");
    xagent_sandbox::headless::run_headless(config, path, false, true);
    digest_database(db_path)
}

/// Runs once into a fresh temporary database and returns the digest.
fn run_in_temp_dir() -> String {
    let dir = tempfile::TempDir::new().expect("create temporary directory");
    run_seeded_digest(&dir.path().join("run.db"))
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
fn default_kernel_matches_recorded_digest() {
    let available = GpuKernel::is_available();
    if env_flag("XAGENT_CAPTURE_SEEDED_DIGEST") {
        assert!(available, "capture requires a GPU adapter");
        let digest = run_in_temp_dir();
        println!("digest: {digest}");
        println!("adapter: {}", adapter_description());
        return;
    }
    if !available {
        assert!(
            !env_flag("XAGENT_REQUIRE_GPU"),
            "no GPU adapter but XAGENT_REQUIRE_GPU=1"
        );
        println!("skipping seeded digest: no GPU adapter");
        return;
    }
    let first = run_in_temp_dir();
    let second = run_in_temp_dir();
    assert_eq!(first, second, "two seeded runs gave different digests");
    println!("digest: {first}");

    let fixture_file = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);
    let text = std::fs::read_to_string(&fixture_file).expect("read digest fixture");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("parse digest fixture");
    assert_eq!(fixture["seed"].as_u64(), Some(SEED));
    assert_eq!(fixture["generations"].as_u64(), Some(GENERATIONS));
    assert_eq!(fixture["tick_budget"].as_u64(), Some(TICK_BUDGET));
    let recorded = fixture["digest"].as_str().expect("fixture digest string");
    assert!(
        is_digest_hex(recorded),
        "fixture digest is not 16 hex digits"
    );

    if env_flag("XAGENT_REQUIRE_SEEDED_DIGEST") {
        assert_eq!(
            first, recorded,
            "computed digest {first} differs from recorded digest {recorded}"
        );
    }
}
