#!/bin/sh
# Baseline feasibility probe for the architect-role tasks of this plan (the
# baseline digest capture and the lesion matrix).
#
# Runs in a fresh host-owned checkout of the baseline revision. It proves that
# a release build of the sandbox binary succeeds, that the binary dumps its
# default config (the matrix derives its evolution config from it), that a GPU
# (or fallback) adapter runs the fused kernel at measurable throughput (the
# matrix's wall-clock budget is derived from this number), and that the
# sandbox test runner and crate-scoped lint/format commands the architect
# tasks' checks use are available and green.
#
# The host may redirect Cargo's target directory (CARGO_TARGET_DIR) outside the
# checkout, so the binary is never addressed as ./target/release/xagent; every
# invocation goes through `cargo run --release -p xagent-sandbox --`.
set -eu

cargo --version
rustc --version
cargo build --release -p xagent-sandbox

cargo run --release -p xagent-sandbox -- --dump-config > /dev/null
cargo run --release -p xagent-sandbox -- --bench --bench-ticks 10000 --bench-agents 10

cargo test -p xagent-sandbox --test contributing_guard
cargo clippy -p xagent-sandbox --all-targets -- -D warnings
cargo fmt -p xagent-sandbox -- --check
