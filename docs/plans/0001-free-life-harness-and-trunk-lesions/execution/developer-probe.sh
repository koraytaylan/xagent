#!/bin/sh
# Baseline feasibility probe for the developer-role tasks of this plan.
#
# Runs in a fresh host-owned checkout of the baseline revision. It proves that
# the declared Rust toolchain resolves the workspace's dependencies, that every
# named-test runner and crate-scoped lint/format command the developer tasks'
# checks rely on exists and is green before any task lands, and that a GPU (or
# fallback) adapter is reachable from host checks, because the GPU-backed task
# checks run with XAGENT_REQUIRE_GPU=1 and fail when no adapter exists.
set -eu

cargo --version
rustc --version

# Kernel crate: whole suite (lib layout tests, GPU tests, doctests) must be
# green at baseline. It covers the focused targets the kernel tasks check
# (`--lib`, `--lib gpu_kernel` and the policy integration tests) and the
# crate-wide `brain-crate-tests` integration check.
cargo test -p xagent-brain

# Sandbox runners named by the champion-loader, metrics, runner and CLI tasks.
cargo test -p xagent-sandbox --lib governor::tests
cargo test -p xagent-sandbox --lib headless::tests
# Only the main.rs `tests` module (speed_label tests): the bin's sim_runtime
# tests are wall-clock timing tests and are covered by the integration
# `sandbox-tests` check instead.
cargo test -p xagent-sandbox --bin xagent -- --skip sim_runtime::
cargo test -p xagent-sandbox --test contributing_guard

# Crate-scoped lint and format commands every developer task declares.
cargo clippy -p xagent-brain --all-targets -- -D warnings
cargo clippy -p xagent-sandbox --all-targets -- -D warnings
cargo fmt -p xagent-brain -- --check
cargo fmt -p xagent-sandbox -- --check

# A GPU or fallback adapter must be reachable: GpuKernel::new needs one, and
# the bench mode builds a kernel and runs it.
cargo run -p xagent-sandbox -- --bench --bench-ticks 1000 --bench-agents 10
