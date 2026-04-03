# CI/CD Pipelines Design

## Overview

GitHub Actions pipelines for build, test, and release of the `xagent` binary. Two workflows: a lightweight CI for everyday development, and a tag-triggered release pipeline that builds cross-platform binaries, generates a changelog, merges to `main`, and publishes a GitHub Release.

## Branching Strategy

- `develop` — active development branch. All commits land here.
- `main` — release-only branch. Only receives `--no-ff` merge commits from the release pipeline.
- `main`'s first-parent log shows one commit per release. Full commit history is traceable through merge commit parents.

## Workflow 1: CI (`ci.yml`)

**Trigger:** Push to `develop`, PRs targeting `develop`.

**Runner:** `ubuntu-latest`

**Steps:**
1. Checkout code
2. Install Rust stable toolchain (`dtolnay/rust-toolchain`, respects `rust-toolchain.toml`)
3. Restore cache (`Swatinem/rust-cache`)
4. `cargo check --workspace`
5. `cargo test -p xagent-sandbox`

No cross-compilation, no artifacts. Linux only.

`rusqlite` with `bundled` feature compiles SQLite from C source; `ubuntu-latest` has `gcc` pre-installed — no extra setup needed.

## Workflow 2: Release (`release.yml`)

**Trigger:** Push of tag matching `v*` (e.g., `v0.2.0`).

### Job 1: `test`

- Runner: `ubuntu-latest`
- Same check + test as CI — gate before doing anything else
- If this fails, the whole release aborts. `main` stays clean.

### Job 2: `build` (needs `test`)

Matrix strategy, 4 targets in parallel:

| Target | Runner | Notes |
|--------|--------|-------|
| `x86_64-unknown-linux-gnu` | `ubuntu-latest` | Native build |
| `x86_64-pc-windows-msvc` | `windows-latest` | Native MSVC toolchain |
| `x86_64-apple-darwin` | `macos-latest` | Cross-compiled on Apple Silicon |
| `aarch64-apple-darwin` | `macos-latest` | Apple Silicon (native on M-series runners) |

Each matrix leg:
1. Checkout code
2. Install Rust stable + target
3. `cargo build --release -p xagent-sandbox`
4. Package binary into tarball (Linux/macOS: `.tar.gz`) or zip (Windows: `.zip`), named `xagent-{version}-{target}.tar.gz` / `.zip`
5. Upload as workflow artifact

### Job 3: `release` (needs `build`)

- Runner: `ubuntu-latest`
- Steps:
  1. Checkout with full history (`fetch-depth: 0`)
  2. Download all build artifacts from Job 2
  3. Run `git-cliff` to generate changelog (commits between previous tag and current tag)
  4. Checkout `main`, run `git merge --no-ff {tag} -m "release: {version}"`
  5. Push `main`
  6. Create GitHub Release on the tag with generated changelog as body, attach all 4 binary archives

**Permissions:** `contents: write` on the default `GITHUB_TOKEN`. No PAT needed.

## Tooling

### git-cliff

Changelog generator. Configured via `cliff.toml` at repo root.

- Groups commits by conventional prefix: `feat:` → Features, `fix:` → Bug Fixes, `perf:` → Performance
- Scope: commits between previous tag and current tag
- Output goes directly into GitHub Release body — no `CHANGELOG.md` file committed to repo

### rust-toolchain.toml

Pinned at repo root (e.g., `channel = "stable"`). Ensures CI and local dev use the same Rust version. The `dtolnay/rust-toolchain` action respects this file automatically.

### Swatinem/rust-cache

Used in both workflows. Caches `target/` and cargo registry.

## Version Management

- Version is explicit in the tag (e.g., `v0.2.0`)
- You decide what the next version is — no auto-inference from commit types
- Workspace version in root `Cargo.toml` is updated manually
- No auto-bump commits after release

## Release Flow

**You do:**
1. Ensure `develop` is in the state you want to release
2. `git tag v0.2.0` on the desired commit
3. `git push origin v0.2.0`

**Pipeline does:**
1. Test on Linux — abort if fail
2. Build all 4 targets in parallel
3. Generate changelog with git-cliff
4. Merge to `main` with `--no-ff`, message: `release: v0.2.0`
5. Push `main`
6. Create GitHub Release with changelog and binary archives

**Result:**
- `main` has one new merge commit with full traceability into `develop`
- GitHub Releases page has the version with auto-generated notes and downloadable binaries
- `develop` is untouched

## Failure Modes

- Test fails → everything stops, nothing published, `main` untouched
- Build fails on one target → release job never runs, nothing published
- Merge conflict on `main` → release job fails, manual investigation (should not happen if `main` only receives these merge commits)

## Artifact Naming

```
xagent-v0.2.0-x86_64-unknown-linux-gnu.tar.gz
xagent-v0.2.0-x86_64-pc-windows-msvc.zip
xagent-v0.2.0-x86_64-apple-darwin.tar.gz
xagent-v0.2.0-aarch64-apple-darwin.tar.gz
```
