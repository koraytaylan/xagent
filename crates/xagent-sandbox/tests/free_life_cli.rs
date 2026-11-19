//! Integration test for the built `xagent` binary's free-life option errors.

use std::process::Command;

/// The built binary rejects a zero agent count with status 1 and a stderr message.
#[test]
fn free_life_zero_agents_exits_non_zero_with_message() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let output = Command::new(env!("CARGO_BIN_EXE_xagent"))
        .args(["--free-life", "--free-life-agents", "0"])
        .current_dir(dir.path())
        .output()
        .expect("run xagent binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "expected non-zero exit; stderr: {stderr}"
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit status 1; stderr: {stderr}"
    );
    let expected = format!(
        "--free-life-agents must be in 1..={}",
        xagent_sandbox::agent::MAX_AGENTS
    );
    assert!(
        stderr.contains(&expected),
        "stderr missing {expected:?}; stderr: {stderr}"
    );
}
