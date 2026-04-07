# Contributing to XAgent

## Tooling

All code must pass the following before merge:

```bash
cargo fmt --all -- --check   # formatting
cargo clippy --workspace --all-targets -- -D warnings   # lints (pedantic enabled)
cargo test -p xagent-sandbox # tests
```

These are enforced by CI. Run them locally before pushing.

## Code Style

We follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) and
enforce `clippy::pedantic` at the workspace level. The rules below supplement those
defaults.

### Naming

| Rule | Example |
|------|---------|
| No single-letter variables outside closures, iterators, or trivial math (`x`, `y`). | `gw` → `grid_width`, `c` → `contact` |
| No bare abbreviations. Spell out names so they read clearly. | `ppp` → `pixels_per_point`, `fc` → `feature_count` |
| Domain abbreviations that are universal in the project may be used if documented here. | `buf` (buffer), `wt`/`wts` (weight/weights) |

### Magic Numbers

Any numeric literal that isn't `0`, `1`, `-1`, `0.0`, or `1.0` should be a named
`const` with a doc comment explaining *why* that value was chosen:

```rust
/// Death penalty decay factor for composite fitness scoring.
const DEATH_PENALTY_DECAY: f32 = 0.5;
```

### Function Length

If a function exceeds roughly 50 lines, consider splitting it into smaller,
single-responsibility helpers. Very long functions (100+ lines) should be
reviewed and refactored as part of normal code review.

### Documentation

- All public types and functions should have `///` doc comments.
- Module-level `//!` docs should describe purpose and key concepts.
- We plan to enable `#![warn(missing_docs)]` incrementally per crate.

## Incremental Cleanup

We do not require a big-bang rewrite. Existing code is cleaned up file-by-file
as it is touched. Priority targets:

1. `governor.rs` — extract magic numbers into named constants; split `advance()`.
2. `buffers.rs` — replace short abbreviations (`gw`, `go`, `fc`).
3. `ui.rs` — rename `ppp`; extract inline RGB colors into a palette module.

Files that already meet the standard (e.g., `config.rs`, `body.rs`) should be
left as-is.
