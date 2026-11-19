#!/bin/sh
# Baseline feasibility probe for this plan's integration checks.
#
# Runs the repository-mandated group commands (CONTRIBUTING.md "Tooling") on
# the baseline revision in a fresh host-owned checkout, proving they are green
# before any task lands, so a failure at integration is attributable to this
# plan's work. `cargo test -p xagent-brain` is the `brain-crate-tests`
# integration check. The strict seeded-digest check is not run here: its test and
# fixture are created by the first task.
set -eu

cargo --version
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xagent-brain
cargo test -p xagent-sandbox
