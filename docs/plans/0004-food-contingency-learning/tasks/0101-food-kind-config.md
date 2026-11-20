---
id: food-kind-config
title: "Food-kind and second-odour config flags with the nourishing-kind draw"
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-shared/src/config.rs
  - crates/xagent-shared/src/lib.rs
  - crates/xagent-sandbox/src/agent/mod.rs
status: done
merged_as: "4bbc44d56d63836fe0b914b104b945e75cd3a8ab"
checks:
  - name: focused-proof
    command: "cargo test -p xagent-shared --lib food_kind"
  - name: crossover-flag
    command: "cargo test -p xagent-sandbox --lib crossover"
---

# Food-kind and second-odour config flags with the nourishing-kind draw

## Outcome

The shared config gains two opt-in flags and the deterministic nourishing-kind
draw. Both flags default to `false`, also when omitted from JSON. No GPU, shader,
physics or sense code changes. Invariants: ARCHITECTURE.md Invariants INV-01,
INV-02 and INV-05.

## Non-goals

- No change to `WorldConfig::easy()` or `hard()`; they use `..Self::default()`.
- No change to `mutate_sensory_genes`; it uses `..parent.clone()`.
- No GPU, WGSL, buffer, sense-block, metrics or CLI change. No reader of
  `food_kinds_enabled` outside this task.
- No edit to `crates/xagent-shared/README.md` or other documentation.
- Added source lines (comments included) contain no `TODO`, `plan`, `task`,
  `workstream` or `spec`; `crates/xagent-sandbox/tests/contributing_guard.rs`
  enforces this.

## Prerequisites

None inside this plan. The full-literal sweep is done: the only `BrainConfig {`
or `WorldConfig {` literals not ending in `..` are `BrainConfig::default`,
`BrainConfig::tiny`, `BrainConfig::large`, `WorldConfig::default` in `config.rs`,
and `mutate_config_with_strength_rng` and `crossover_config` in `agent/mod.rs`.
`mutate_config` and `mutate_config_seeded` call `mutate_config_with_strength_rng`
and need no edit.

## Decisions

- **Kind identity.** `FoodKind::of_item(i)` is `First` for even `i`, `Second` for
  odd `i`. The kind is an index property and is never stored.
- **Nourishing-kind draw.** `nourishing_food_kind(draw_index)`:
  1. `x = self.seed ^ draw_index.wrapping_mul(SPLITMIX_GAMMA)`,
     `SPLITMIX_GAMMA = 0x9E37_79B9_7F4A_7C15`.
  2. splitmix64 output finalizer on `x`, with no added gamma step:
     `z = x ^ (x >> SPLITMIX_SHIFT_A)`; `z = z.wrapping_mul(SPLITMIX_MUL_A)`;
     `z ^= z >> SPLITMIX_SHIFT_B`; `z = z.wrapping_mul(SPLITMIX_MUL_B)`;
     `z ^= z >> SPLITMIX_SHIFT_C`. Constants: `SHIFT_A = 30`,
     `MUL_A = 0xBF58_476D_1CE4_E5B9`, `SHIFT_B = 27`,
     `MUL_B = 0x94D0_49BB_1331_11EB`, `SHIFT_C = 31`.
  3. Low bit of `z`: 0 gives `First`, 1 gives `Second`.
  Each literal is a named `const` with a `///` reason (the golden-ratio gamma
  spreads consecutive draw indices; the finalizer constants are the standard
  splitmix64 mixers). The method doc says the kernel draws index 0 at
  construction and the next index on every agent reset.
- **Opt-in.** Both serde defaults are `false`, so existing JSON, presets and
  stored champions keep their behaviour.
- **Brain flag not heritable.** `second_odour_sense_enabled` is never mutated.
  Crossover copies it from parent `a`, as the other locked per-batch genes do.

## Steps

**Steps:**

1. In `crates/xagent-shared/src/config.rs`, add `pub enum FoodKind { First, Second }`
   with derives `Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize`,
   doc comments, `of_item(index: usize)`, `index(self) -> u32` (`First` 0,
   `Second` 1) and the named splitmix consts. Add `food_kind_item_parity`:
   `of_item(0)` is `First`, `of_item(1)` is `Second`, `of_item(10)` is `First`,
   `First.index()` is 0, `Second.index()` is 1.
2. In the same file, add `WorldConfig.food_kinds_enabled` after `seed`
   (`#[serde(default)]`, doc: opt-in two-kind food scenario; off keeps the world
   exactly as before), `false` in `WorldConfig::default()`, and
   `WorldConfig::nourishing_food_kind(&self, draw_index: u64) -> FoodKind` per
   Decisions. Add `food_kind_draw_is_deterministic` (same seed and index give equal
   kinds across two separately built configs; seed 42's first 64 draws are not all
   equal) and `food_kind_draw_is_balanced` (seeds `0..1000` at draw index 0, and
   draw indices `0..1000` for seed 42: the `First` share is within `[0.45, 0.55]`).
   Count into a `u32` and divide with `f64::from`, not `as`.
3. In the same file, add `BrainConfig.second_odour_sense_enabled` after
   `cortex_stage_limit` (`#[serde(default)]`, doc: declares the second nostril-pair
   sense; not a heritable gene; never mutated; crossover copies parent `a`), and
   `false` in `BrainConfig::default()`, `tiny()` and `large()`. Add
   `food_kind_flags_default_off`: both flags are `false` for `WorldConfig::default()`,
   `easy()`, `hard()`, `BrainConfig::default()`, `tiny()` and `large()`. It also
   serialises `BrainConfig::default()` and `WorldConfig::default()` with `serde_json`
   (a dev-dependency), removes both flag keys from each object, deserialises, and
   asserts both flags are `false`. Add `food_kind_flags_round_trip`: set both flags
   `true`, serialise, deserialise, and assert both are still `true`.
4. In `crates/xagent-shared/src/lib.rs`, add `FoodKind` to the `pub use config::{...}`
   list in alphabetical position, using the list's existing formatting.
5. In `crates/xagent-sandbox/src/agent/mod.rs`, add
   `second_odour_sense_enabled: a.second_odour_sense_enabled,` to `crossover_config`
   with a one-line comment (locked per batch, not heritable, taken from `a`), and
   `second_odour_sense_enabled: parent.second_odour_sense_enabled,` to
   `mutate_config_with_strength_rng` with a comment saying it passes through. Both
   full literals must list the field, or the crate fails to compile. Add
   `crossover_keeps_second_odour_sense_flag_from_parent_a`: `a` true with `b` false
   gives a true child; `a` false with `b` true gives a false child; run crossover at
   least 32 times per case. Add `crossover_family_mutation_keeps_second_odour_sense_flag`:
   a parent with the flag true gives `mutate_config`, `mutate_config_seeded` and
   `mutate_sensory_genes` children that are still true, and a parent with the flag
   false gives children that are still false.

- **Done when:** the focused-proof check and the crossover-flag check both pass on this task's tree. focused-proof runs `food_kind_item_parity`, `food_kind_draw_is_deterministic`, `food_kind_draw_is_balanced`, `food_kind_flags_default_off` and `food_kind_flags_round_trip`; crossover-flag runs `crossover_keeps_second_odour_sense_flag_from_parent_a` and `crossover_family_mutation_keeps_second_odour_sense_flag`. These prove INV-01 (both flags default to false, including when omitted from JSON), INV-02 (the draw is deterministic and balanced over the stated ranges, and a kind is only an index parity or a seed/draw hash) and INV-05 (the brain flag is copied from parent a in crossover and kept by every mutation path, so it is never heritable).

## Evidence and risks

- The worker has no shell. The host runs both checks. The balance bound is about
  3.2 standard deviations for 1,000 samples, so a failing balance points at the
  hash constants, not the bound.
- If a compile error names a `BrainConfig {` or `WorldConfig {` literal in a file
  outside `touches`, the worker stops and reports that file and line instead of
  editing it.
