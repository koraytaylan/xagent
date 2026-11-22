//! Removed generation flags exit 2 and invalid supervisor flags exit 1,
//! both before any database is opened.

use std::process::Command;

fn run(args: &[&str]) -> (std::process::Output, std::path::PathBuf, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("run.db");
    let output = Command::new(env!("CARGO_BIN_EXE_xagent"))
        .arg("--no-render")
        .arg("--db")
        .arg(&db)
        .args(args)
        .output()
        .expect("spawn xagent");
    (output, db, dir)
}

fn assert_removed(args: &[&str], flag: &str) {
    let (out, db, _dir) = run(args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains(&format!("--{flag}")), "{stderr}");
    assert!(stderr.contains("removed with generation mode"), "{stderr}");
    assert!(!db.exists());
}

#[test]
fn removed_generations_flag_exits_2() {
    assert_removed(&["--generations", "3"], "generations");
}

#[test]
fn removed_dump_tree_flag_exits_2() {
    assert_removed(&["--dump-tree"], "dump-tree");
}

#[test]
fn removed_evolution_mode_flag_exits_2() {
    assert_removed(&["--evolution-mode", "supervisor"], "evolution-mode");
}

#[test]
fn removed_validation_flags_exit_2() {
    assert_removed(
        &["--validate-innate-instincts"],
        "validate-innate-instincts",
    );
    assert_removed(&["--validate-danger-percept"], "validate-danger-percept");
    assert_removed(&["--validation-generations", "5"], "validation-generations");
    assert_removed(&["--validation-population", "5"], "validation-population");
    assert_removed(&["--validation-replicates", "5"], "validation-replicates");
    assert_removed(
        &["--validation-tick-budget", "10"],
        "validation-tick-budget",
    );
    assert_removed(&["--validation-seed-start", "1"], "validation-seed-start");
}

#[test]
fn removed_free_life_champion_flags_exit_2() {
    assert_removed(
        &["--free-life-champion-db", "c.db"],
        "free-life-champion-db",
    );
    assert_removed(
        &["--free-life-champion-node", "3"],
        "free-life-champion-node",
    );
}

#[test]
fn evolution_cli_exit_bad_budget() {
    let (out, db, _dir) = run(&["--agent-tick-budget", "7"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Error:"), "{stderr}");
    assert!(stderr.contains("agent_tick_budget"), "{stderr}");
    assert!(!db.exists());
}
