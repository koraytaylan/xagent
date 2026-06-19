//! Mechanical enforcement of the `CONTRIBUTING.md` rules that no `clippy` lint
//! can express, wired into the `cargo test -p xagent-sandbox` gate so a
//! violation blocks a merge the same way a failing unit test does.
//!
//! Two rules are guarded:
//!
//! 1. **Planning-agnostic source.** `CONTRIBUTING.md` forbids naming an internal
//!    plan, task, workstream, decision doc, or spec section anywhere in source —
//!    comments, doc-comments, and runtime log/error/assert strings alike. The
//!    codebase carries pre-existing debt here, and the same document mandates
//!    clearing it *file-by-file as code is touched*, never in one big-bang pass.
//!    So this is enforced as a **ratchet**: every offending file's current count
//!    is frozen in [`PLANNING_REFERENCE_BASELINE`]; the test fails the moment a
//!    file exceeds its frozen count or a fresh file introduces a reference. The
//!    debt may only shrink, never grow.
//!
//! 2. **No pending-work markers.** `TODO`/`FIXME` comments are banned outright.
//!    The tree is currently clean, so this is a hard zero — no baseline.
//!
//! Neither rule relies on a model reading prose: the gate is deterministic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// This test file is excluded from the scan — it necessarily contains the very
/// patterns it searches for.
const GUARD_TEST_FILE_NAME: &str = "contributing_guard.rs";

/// Frozen per-file count of planning references (see module docs). An entry's
/// number is the *maximum* tolerated for that file; the ratchet allows it to
/// fall but never rise. When a file is cleaned, lower or remove its entry in the
/// same commit. Do **not** raise a number or add a row to admit a new
/// reference — that defeats the guard. The test prints a ready-to-paste
/// replacement table whenever this list drifts from reality.
const PLANNING_REFERENCE_BASELINE: &[(&str, usize)] = &[
    ("crates/xagent-brain/src/buffers.rs", 11),
    ("crates/xagent-brain/src/complex.rs", 1),
    ("crates/xagent-brain/src/dog.rs", 3),
    ("crates/xagent-brain/src/gabor.rs", 5),
    ("crates/xagent-brain/src/gpu_kernel.rs", 4),
    (
        "crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl",
        20,
    ),
    ("crates/xagent-brain/src/shaders/kernel/brain_tick.wgsl", 1),
    ("crates/xagent-brain/src/shaders/kernel/common.wgsl", 17),
    ("crates/xagent-brain/src/shaders/kernel/kernel_tick.wgsl", 3),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_brain_encode_tiled.wgsl",
        1,
    ),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_brain_encoder_credit_tiled.wgsl",
        1,
    ),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_brain_features.wgsl",
        3,
    ),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_brain_predictor_tiled.wgsl",
        1,
    ),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_brain_tail_from_scratch.wgsl",
        1,
    ),
    (
        "crates/xagent-brain/src/shaders/kernel/phase_physics.wgsl",
        2,
    ),
    ("crates/xagent-sandbox/src/agent/mod.rs", 5),
    ("crates/xagent-sandbox/src/bench.rs", 4),
    ("crates/xagent-sandbox/src/governor.rs", 1),
    ("crates/xagent-sandbox/src/main.rs", 2),
    ("crates/xagent-sandbox/src/sim_runtime.rs", 1),
    ("crates/xagent-sandbox/src/ui.rs", 3),
    ("crates/xagent-sandbox/tests/integration.rs", 20),
    ("crates/xagent-shared/src/config.rs", 17),
];

/// Plain case-insensitive substrings that always signal a planning reference.
const PLANNING_REFERENCE_SUBSTRINGS: [&str; 4] =
    ["workstream", "status.md", "tasks.md", "docs/superpowers"];

/// Uppercase markers for pending work, banned outright by `CONTRIBUTING.md`.
const PENDING_WORK_MARKERS: [&str; 2] = ["TODO", "FIXME"];

#[test]
fn planning_references_do_not_exceed_baseline() {
    let workspace_root = workspace_root();
    let actual = count_planning_references(&workspace_root);
    let baseline: BTreeMap<&str, usize> = PLANNING_REFERENCE_BASELINE.iter().copied().collect();

    let mut increases = Vec::new();
    for (path, &count) in &actual {
        let allowed = baseline.get(path.as_str()).copied().unwrap_or(0);
        if count > allowed {
            increases.push((path.clone(), allowed, count));
        }
    }
    let mut reductions = Vec::new();
    for (&path, &allowed) in &baseline {
        let current = actual.get(path).copied().unwrap_or(0);
        if current < allowed {
            reductions.push((path.to_string(), allowed, current));
        }
    }

    assert!(
        increases.is_empty(),
        "New or increased planning references in source. CONTRIBUTING.md forbids \
         naming any plan, task, workstream, decision doc, or spec section in \
         source — including runtime log/error/assert strings. State the technical \
         reason in place instead; the planning rationale belongs in the commit \
         message, PR description, and planning docs.\n\nOffenders (file: \
         allowed -> found):\n{}\n\nDo NOT raise the baseline to silence this.\n",
        format_offenders(&increases)
    );

    assert!(
        reductions.is_empty(),
        "Planning-reference debt fell below the frozen baseline — tighten the \
         ratchet so it can never silently climb back. Replace \
         PLANNING_REFERENCE_BASELINE in this file with:\n\n{}\n\nReduced (file: \
         was -> now):\n{}\n",
        render_baseline(&actual),
        format_offenders(&reductions)
    );
}

#[test]
fn source_contains_no_pending_work_markers() {
    let workspace_root = workspace_root();
    let mut hits = Vec::new();
    for path in collect_scanned_sources(&workspace_root.join("crates")) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        for (line_number, line) in text.lines().enumerate() {
            if PENDING_WORK_MARKERS
                .iter()
                .any(|marker| line.contains(marker))
            {
                let relative = relative_to(&path, &workspace_root);
                hits.push(format!("{relative}:{}: {}", line_number + 1, line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "CONTRIBUTING.md bans TODO/FIXME comments — track pending work in the \
         issue tracker, not the source:\n{}",
        hits.join("\n")
    );
}

/// Scans every `.rs`/`.wgsl` file under `crates/` and returns the per-file count
/// of lines carrying a planning reference, omitting files with zero.
fn count_planning_references(workspace_root: &Path) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for path in collect_scanned_sources(&workspace_root.join("crates")) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let count = text
            .lines()
            .filter(|line| line_has_planning_reference(line))
            .count();
        if count > 0 {
            counts.insert(relative_to(&path, workspace_root), count);
        }
    }
    counts
}

/// True when a line names a plan, task, workstream, decision doc, or spec.
fn line_has_planning_reference(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if PLANNING_REFERENCE_SUBSTRINGS
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return true;
    }
    let bytes = lower.as_bytes();
    contains_plan_number(bytes) || contains_task_slug(bytes)
}

/// Matches the pattern `plan 0[0-9]{3}` (e.g. "plan 0009") on lowercased bytes.
fn contains_plan_number(bytes: &[u8]) -> bool {
    const PREFIX: &[u8] = b"plan 0";
    /// The pattern requires three ASCII digits after the `plan 0` prefix.
    const DIGITS_REQUIRED: usize = 3;
    bytes
        .windows(PREFIX.len())
        .enumerate()
        .any(|(start, window)| {
            if window != PREFIX {
                return false;
            }
            let digits_start = start + PREFIX.len();
            let digits_end = digits_start + DIGITS_REQUIRED;
            digits_end <= bytes.len()
                && bytes[digits_start..digits_end]
                    .iter()
                    .all(u8::is_ascii_digit)
        })
}

/// Matches the on-disk slug form `000[0-9]-[a-z]` (e.g. "0009-intent") on
/// lowercased bytes.
fn contains_task_slug(bytes: &[u8]) -> bool {
    const PREFIX: &[u8] = b"000";
    /// `000` + one digit + `-` + one letter.
    const PATTERN_LENGTH: usize = 6;
    bytes.windows(PATTERN_LENGTH).any(|window| {
        &window[0..PREFIX.len()] == PREFIX
            && window[3].is_ascii_digit()
            && window[4] == b'-'
            && window[5].is_ascii_lowercase()
    })
}

/// Recursively collects scanned source files under `directory`, sorted.
fn collect_scanned_sources(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if is_scanned_source(&path) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// True for `.rs`/`.wgsl` files other than this guard test itself.
fn is_scanned_source(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) == Some(GUARD_TEST_FILE_NAME) {
        return false;
    }
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "wgsl")
    )
}

/// Workspace root: two directories above this crate's manifest.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root is two levels above the crate manifest")
        .to_path_buf()
}

/// Path relative to the workspace root, with forward slashes for stable output.
fn relative_to(path: &Path, workspace_root: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Renders the current scan as a ready-to-paste `PLANNING_REFERENCE_BASELINE`.
fn render_baseline(actual: &BTreeMap<String, usize>) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("const PLANNING_REFERENCE_BASELINE: &[(&str, usize)] = &[\n");
    for (path, count) in actual {
        let _ = writeln!(out, "    (\"{path}\", {count}),");
    }
    out.push_str("];");
    out
}

/// Formats `(path, expected, found)` rows one per line.
fn format_offenders(rows: &[(String, usize, usize)]) -> String {
    rows.iter()
        .map(|(path, expected, found)| format!("  {path}: {expected} -> {found}"))
        .collect::<Vec<_>>()
        .join("\n")
}
