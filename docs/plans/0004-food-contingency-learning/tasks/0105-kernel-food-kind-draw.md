---
id: kernel-food-kind-draw
title: "Kernel nourishing-kind draw and world-config packing"
workstream: "0100"
kind: task
role: developer
depends_on:
  - food-kind-layout
gated: false
touches:
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/tests/food_kind_draw.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test food_kind_draw"
    timeout_ms: 3600000
  - name: brain-lib
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --lib"
    timeout_ms: 5400000
  - name: parity
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-sandbox --test sense_block_parity -- --include-ignored"
    timeout_ms: 5400000
---

# Kernel nourishing-kind draw and world-config packing

## Outcome

`GpuKernel` holds the per-run nourishing-kind assignment. It redraws the
assignment deterministically at construction and on every agent reset that
actually runs, while food kinds are on. It exposes the current kind and packs
it into `WC_FOOD_KINDS` on every world-config upload. With food kinds off,
nothing changes: the kind is `None` and the slot is `0.0`. Building the kernel
from the second-odour flag is task `second-odour-kernel`.

## Scope

- `crates/xagent-brain/src/gpu_kernel.rs`:
  - New fields on `GpuKernel`: `food_kinds_enabled: bool` (copied from
    `world_config.food_kinds_enabled` in the constructor), `food_kind_draws:
    u64`, and `nourishing_food_kind: Option<FoodKind>`. The kernel already
    stores `world_config: WorldConfig`, which the draw reads, so no other
    field is added for the draw.
  - Draw rule, used at construction and at each reset that runs: when enabled,
    set `nourishing_food_kind` to `Some(self.world_config.nourishing_food_kind(
    self.food_kind_draws))`, then increment `food_kind_draws`. When disabled,
    the kind is `None` and the counter is not used.
  - `reset_agents_with_rng` (reached by `reset_agents_seeded` and `reset_agents`)
    draws at its start.
  - `try_reset_agents` draws only after its in-flight staging-buffer loop, so the
    early `return false` consumes no draw.
  - `pub fn nourishing_food_kind(&self) -> Option<FoodKind>` with a doc comment
    stating that the kernel's owning thread (sim worker or headless loop) is the
    only writer of the draw state.
  - In `write_world_config_scratch`, after `fill_world_config` and beside the
    `WC_PHASE_MASK` override: `self.world_config_scratch[WC_FOOD_KINDS] =
    food_kinds_slot_value(self.nourishing_food_kind);`. `WC_FOOD_KINDS` and
    `food_kinds_slot_value` come from food-kind-layout in `buffers.rs`.
- `crates/xagent-brain/tests/food_kind_draw.rs` (new): GPU-backed tests that
  start with the fail-closed helper described under Decisions. They follow
  `crates/xagent-brain/tests/food_claims.rs` for building configs and kernels.
  - `food_kind_draw_off_by_default`: a default world gives `None` after
    construction and after a reset. With kinds off,
    `food_kinds_slot_value(kernel.nourishing_food_kind())` is `0.0`.
  - `food_kind_draw_redraws_on_every_reset`: with kinds on, the sequence read
    after construction and after N mixed resets (`reset_agents_seeded` and
    `reset_agents`, N = 8 as a named const) equals `world.nourishing_food_kind(i)`
    for `i` in `0..=N`, in order.
  - `food_kind_draw_is_seed_deterministic`: two kernels with the same seed give
    the same sequence.
  - `food_kind_draw_packs_world_config_slot`: the accessor and
    `food_kinds_slot_value` agree for the drawn kind. Add no other public accessor.
  - `gpu_guard_fails_closed_when_required` (CPU only): the pure decision
    function used by the helper returns an error when no adapter is present and
    `XAGENT_REQUIRE_GPU` is set, and `Ok(false)` (skip) when it is unset.

## Non-goals

- No WGSL, `buffers.rs` or `senses.rs` change; no change to `GpuKernel::new`'s
  sense table or `vision_override_constants` (task `second-odour-kernel`).
- No change to `fill_world_config` or `build_world_config` signatures.
- No change to `WORLD_CONFIG_SIZE`, food claims, meals, vision or the harness.
- No planning vocabulary or `TODO` in source or test comments.

## Prerequisites

food-kind-layout provides `WC_FOOD_KINDS`, `food_kinds_slot_value` and
`xagent_shared::FoodKind` (through food-kind-config, which also provides
`WorldConfig::nourishing_food_kind(&self, draw_index: u64) -> FoodKind` and
`WorldConfig.food_kinds_enabled`).

## Decisions

- **Draw placement.** The draw is in the shared reset path and in `try_reset_agents`
  after its early return. A deferred reset therefore consumes no draw, which
  keeps the sequence aligned with the config draws.
- **No new tunable.** The kind comes only from the config draw function.
- **Fail-closed GPU helper.** `tests/food_kind_draw.rs` defines
  `fn gpu_guard_decision(available: bool, required: bool) -> Result<bool,
  &'static str>`: `Ok(true)` when an adapter is available, `Err(message naming
  the missing adapter)` when none is available and `required`, and `Ok(false)`
  otherwise. `fn gpu_or_skip() -> bool` calls
  `GpuKernel::is_available()` and `std::env::var_os("XAGENT_REQUIRE_GPU").is_some()`,
  panics on `Err`, prints a skip line on `Ok(false)`, and returns the result.
  Every GPU test starts with `if !gpu_or_skip() { return; }`. The checks set
  `XAGENT_REQUIRE_GPU=1`, so a host without an adapter fails rather than
  passing on skipped tests. `gpu_guard_fails_closed_when_required` proves the
  decision on any host, so INV-10 has a host-run proof even on a machine that
  has an adapter.
## Steps

**Steps:**
1. In `crates/xagent-brain/src/gpu_kernel.rs`, add the fields
   `food_kinds_enabled`, `food_kind_draws` and `nourishing_food_kind` to
   `GpuKernel`. In the constructor, copy the flag and initialise the draw state. When kinds are on, draw index 0 and set the
   counter to 1. When off, leave the kind `None` and the counter at 0.
2. In `reset_agents_with_rng`, perform the draw at the start of the function.
   In `try_reset_agents`, perform the draw after the staging-buffer loop, so the
   deferred `return false` runs first. Both use the same private helper, which
   applies the draw rule in the Scope.
3. Add `pub fn nourishing_food_kind(&self) -> Option<FoodKind>` beside
   `is_available`, with its doc comment naming the single owning thread.
4. In `write_world_config_scratch`, set `WC_FOOD_KINDS` from
   `food_kinds_slot_value` after `fill_world_config`, beside `WC_PHASE_MASK`.
5. Create `crates/xagent-brain/tests/food_kind_draw.rs` with
   `gpu_guard_decision`, `gpu_or_skip`, the `N = 8` const, and the five tests
   in Scope.

- **Done when:** the check focused-proof passes with the five food_kind_draw tests ok (food_kind_draw_off_by_default, food_kind_draw_redraws_on_every_reset, food_kind_draw_is_seed_deterministic, food_kind_draw_packs_world_config_slot, gpu_guard_fails_closed_when_required), the check brain-lib passes, and the check parity matches the unchanged golden; together they prove INV-02 (the kernel's kind sequence equals the config draw sequence at construction and after every reset, and is seed-deterministic), INV-01 (with kinds off the kind is None after construction and resets, the slot value is 0.0, and the golden is unchanged), and INV-10 (the GPU guard panics when an adapter is missing and XAGENT_REQUIRE_GPU is set).

## Evidence and risks

- The worker has no shell. The proof is the host-run checks named above, so the
  worker reads the reset, constructor and scratch code by search and edit only.
- The packed `WC_FOOD_KINDS` value is written to private scratch, and the tests
  check it through `food_kinds_slot_value` and the accessor, not by reading the
  uploaded buffer. The write line is verified by the code review of Step 4.
- `try_reset_agents`'s draw placement has no dedicated test in this task. Its
  correctness rests on the code placement in Step 2.
- If the constructor or a reset path differs from Steps 1-2, the worker reports
  the file and line rather than editing outside `touches`.
