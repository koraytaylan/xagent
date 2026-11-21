//! Invalid evolution flag combinations exit 1 before any database is opened.

use std::process::Command;

fn run(args: &[&str]) -> (std::process::Output, std::path::PathBuf, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("run.db");
    let output = Command::new(env!("CARGO_BIN_EXE_xagent"))
        .args(args)
        .arg("--db")
        .arg(&db)
        .output()
        .expect("spawn xagent");
    (output, db, dir)
}

#[test]
fn evolution_cli_exit_resume_supervisor() {
    let (out, db, _dir) = run(&["--no-render", "--evolution-mode", "supervisor", "--resume"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Error:"), "{stderr}");
    assert!(stderr.contains("checkpoint"), "{stderr}");
    assert!(!db.exists());
}

#[test]
fn evolution_cli_exit_bad_budget() {
    let (out, db, _dir) = run(&[
        "--no-render",
        "--evolution-mode",
        "supervisor",
        "--agent-tick-budget",
        "7",
    ]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Error:"), "{stderr}");
    assert!(stderr.contains("agent_tick_budget"), "{stderr}");
    assert!(!db.exists());
}
