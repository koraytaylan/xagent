//! Behavioural fingerprint of one seeded free life with both pathway routes
//! off, replayed through the built `xagent` binary.
//!
//! The life runs with the pinned config fixture, a fixed seed, agent count and
//! tick count, and a fresh birth brain. Two FNV-1a digests are compared with
//! the recorded fixture: the per-agent metrics (`agents` value of the report)
//! and the tick-log payload after its header line. A different GPU adapter
//! than the recording one is a skip, never a comparison.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pinned config with both pathway routes off.
const CONFIG_FIXTURE: &str = "tests/fixtures/pathway_routes_off_config.json";
/// Recorded fingerprint.
const FINGERPRINT_FIXTURE: &str = "tests/fixtures/pathway_routes_off_fingerprint.json";

/// World and run seed.
const SEED: u64 = 5;
/// Agents in the free life.
const AGENT_COUNT: usize = 10;
/// Simulation ticks in the free life.
const TICKS: u64 = 20_000;
/// Birth brain source: no champion database is passed, so brains are fresh.
const BRAIN_SOURCE: &str = "fresh";

/// Flag loading the full config from a JSON file.
const FLAG_CONFIG: &str = "--config";
/// Flag setting the seed.
const FLAG_SEED: &str = "--seed";
/// Switch selecting one free life.
const FLAG_FREE_LIFE: &str = "--free-life";
/// Flag setting the free-life agent count.
const FLAG_AGENTS: &str = "--free-life-agents";
/// Flag setting the free-life tick count.
const FLAG_TICKS: &str = "--free-life-ticks";
/// Flag naming the tick-log output file.
const FLAG_LOG: &str = "--free-life-log";
/// Flag naming the metrics report output file.
const FLAG_OUT: &str = "--free-life-out";

/// Top-level report key holding the per-agent metrics array.
const METRICS_KEY: &str = "agents";
/// The tick-log header is one JSON line ending at the first newline byte.
const HEADER_TERMINATOR: u8 = b'\n';
/// Log filter making the adapter info record reach the child's stderr.
const RUST_LOG_VALUE: &str = "info";
/// Prefix of the adapter info record on stderr; the name runs to the next `"`.
const ADAPTER_PREFIX: &str = "[GpuKernel] Adapter: AdapterInfo { name: \"";

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Writes one line straight to the process stderr handle, which the test
/// harness does not capture, so a passing run still shows it.
macro_rules! report {
    ($($arg:tt)*) => {
        writeln!(std::io::stderr(), $($arg)*).expect("write to stderr")
    };
}

/// Recorded fingerprint values.
#[derive(Clone, Debug)]
struct Fingerprint {
    gpu_adapter: String,
    metrics_digest: String,
    log_digest: String,
}

/// Outcome of comparing a run with the recorded fingerprint.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Match,
    Mismatch {
        digest: &'static str,
        computed: String,
        recorded: String,
    },
    OtherAdapter {
        recorded: String,
        actual: String,
    },
}

/// Compares adapter name first, then the metrics digest, then the log digest.
fn fingerprint_verdict(
    recorded: &Fingerprint,
    adapter: &str,
    metrics_digest: &str,
    log_digest: &str,
) -> Verdict {
    if recorded.gpu_adapter != adapter {
        return Verdict::OtherAdapter {
            recorded: recorded.gpu_adapter.clone(),
            actual: adapter.to_string(),
        };
    }
    if recorded.metrics_digest != metrics_digest {
        return Verdict::Mismatch {
            digest: "metrics",
            computed: metrics_digest.to_string(),
            recorded: recorded.metrics_digest.clone(),
        };
    }
    if recorded.log_digest != log_digest {
        return Verdict::Mismatch {
            digest: "log",
            computed: log_digest.to_string(),
            recorded: recorded.log_digest.clone(),
        };
    }
    Verdict::Match
}

/// 64-bit FNV-1a over `bytes`, rendered as 16 lowercase hex digits.
fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// Digest of the report's `agents` value re-serialised with `serde_json`.
fn metrics_digest(report: &[u8]) -> String {
    let value: serde_json::Value = serde_json::from_slice(report).expect("parse metrics report");
    let agents = value.get(METRICS_KEY).expect("report has an agents key");
    fnv1a_hex(&serde_json::to_vec(agents).expect("serialise agents"))
}

/// Digest of the tick-log bytes after the header line.
fn log_digest(log: &[u8]) -> String {
    let split = log
        .iter()
        .position(|b| *b == HEADER_TERMINATOR)
        .expect("tick log has a header line");
    fnv1a_hex(&log[split + 1..])
}

/// Adapter name from the first adapter info record on stderr.
fn parse_adapter(stderr: &str) -> Option<String> {
    let start = stderr.find(ADAPTER_PREFIX)? + ADAPTER_PREFIX.len();
    let rest = &stderr[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn manifest_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Loads the recorded fingerprint and checks its pinned parameters.
fn load_fingerprint() -> Fingerprint {
    let text = std::fs::read_to_string(manifest_path(FINGERPRINT_FIXTURE))
        .expect("read fingerprint fixture");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse fingerprint fixture");
    assert_eq!(v["digest_algorithm"], "fnv1a-64");
    assert_eq!(v["parameters"]["seed"], SEED);
    assert_eq!(v["parameters"]["agent_count"], AGENT_COUNT);
    assert_eq!(v["parameters"]["ticks"], TICKS);
    assert_eq!(v["parameters"]["brain_source"], BRAIN_SOURCE);
    assert_eq!(v["metrics_scope"], METRICS_KEY);
    assert_eq!(v["log_scope"], "payload_after_header_line");
    let field = |k: &str| {
        v[k].as_str()
            .unwrap_or_else(|| panic!("fixture field {k} is a string"))
            .to_string()
    };
    Fingerprint {
        gpu_adapter: field("gpu_adapter"),
        metrics_digest: field("metrics_digest"),
        log_digest: field("log_digest"),
    }
}

/// Replays the free life and compares it with the recorded fingerprint.
#[test]
fn routes_off_free_life_matches_recorded_fingerprint() {
    if !xagent_brain::GpuKernel::is_available() {
        assert!(
            std::env::var("XAGENT_REQUIRE_GPU").is_err(),
            "XAGENT_REQUIRE_GPU set but no GPU/fallback adapter available"
        );
        report!("Skipping: no GPU/fallback adapter available");
        return;
    }
    let recorded = load_fingerprint();
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("pathway_default_fingerprint");
    std::fs::create_dir_all(&scratch).expect("create scratch dir");
    let log_path = scratch.join("ticks.bin");
    let out_path = scratch.join("metrics.json");
    let _ = std::fs::remove_file(&log_path);
    let _ = std::fs::remove_file(&out_path);
    let seed = SEED.to_string();
    let agents = AGENT_COUNT.to_string();
    let ticks = TICKS.to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_xagent"))
        .arg(FLAG_CONFIG)
        .arg(manifest_path(CONFIG_FIXTURE))
        .args([FLAG_SEED, &seed, FLAG_FREE_LIFE, FLAG_AGENTS, &agents])
        .args([FLAG_TICKS, &ticks])
        .arg(FLAG_LOG)
        .arg(&log_path)
        .arg(FLAG_OUT)
        .arg(&out_path)
        .env("RUST_LOG", RUST_LOG_VALUE)
        .current_dir(&scratch)
        .output()
        .expect("run xagent binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "free life failed; stderr: {stderr}"
    );
    let adapter = parse_adapter(&stderr)
        .unwrap_or_else(|| panic!("no adapter info record on stderr; stderr: {stderr}"));
    let metrics = metrics_digest(&std::fs::read(&out_path).expect("read metrics report"));
    let log = log_digest(&std::fs::read(&log_path).expect("read tick log"));
    report!("fingerprint adapter: {adapter}");
    report!("fingerprint metrics digest: {metrics}");
    report!("fingerprint log digest: {log}");
    match fingerprint_verdict(&recorded, &adapter, &metrics, &log) {
        Verdict::Match => report!("fingerprint: match"),
        Verdict::OtherAdapter { recorded, actual } => {
            report!("Skipping: recorded on adapter {recorded:?}, running on {actual:?}");
        }
        Verdict::Mismatch {
            digest,
            computed,
            recorded,
        } => panic!("{digest} digest mismatch: computed {computed}, recorded {recorded}"),
    }
}

/// A different metrics digest is rejected as a metrics mismatch.
#[test]
fn altered_metrics_digest_is_rejected() {
    let recorded = load_fingerprint();
    let mut altered = recorded.clone();
    altered.metrics_digest = if recorded.metrics_digest == "0000000000000000" {
        "0000000000000001".to_string()
    } else {
        "0000000000000000".to_string()
    };
    let verdict = fingerprint_verdict(
        &altered,
        &recorded.gpu_adapter,
        &recorded.metrics_digest,
        &recorded.log_digest,
    );
    assert!(
        matches!(
            verdict,
            Verdict::Mismatch {
                digest: "metrics",
                ..
            }
        ),
        "{verdict:?}"
    );
}

/// A different log digest is rejected as a log mismatch.
#[test]
fn altered_log_digest_is_rejected() {
    let recorded = load_fingerprint();
    let mut altered = recorded.clone();
    altered.log_digest = if recorded.log_digest == "0000000000000000" {
        "0000000000000001".to_string()
    } else {
        "0000000000000000".to_string()
    };
    let verdict = fingerprint_verdict(
        &altered,
        &recorded.gpu_adapter,
        &recorded.metrics_digest,
        &recorded.log_digest,
    );
    assert!(
        matches!(verdict, Verdict::Mismatch { digest: "log", .. }),
        "{verdict:?}"
    );
}

/// The recorded digests and adapter match themselves.
#[test]
fn recorded_digests_match_themselves() {
    let r = load_fingerprint();
    assert_eq!(
        fingerprint_verdict(&r, &r.gpu_adapter, &r.metrics_digest, &r.log_digest),
        Verdict::Match
    );
}

/// A different adapter name is reported as another adapter, not compared.
#[test]
fn other_adapter_is_skipped() {
    let r = load_fingerprint();
    let other = format!("{} (other)", r.gpu_adapter);
    assert_eq!(
        fingerprint_verdict(&r, &other, &r.metrics_digest, &r.log_digest),
        Verdict::OtherAdapter {
            recorded: r.gpu_adapter.clone(),
            actual: other,
        }
    );
}
