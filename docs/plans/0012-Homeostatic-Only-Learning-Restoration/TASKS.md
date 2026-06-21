# XAgent Plan 0012 — Homeostatic-Only Learning Restoration

First wire a CPU-readable `raw_gradient` handle (`P_RAW_GRADIENT_OUT` physics slot, surfaced as `AgentTelemetry::raw_gradient`) and capture the pre-removal baseline distribution at default config (measurement-first, gating both removals); then remove the approach-PBRS term (`shaping` computed from `APPROACH_SHAPING_GAIN` and `P_PREV_POTENTIAL`, lines 791–805 of brain_passes.wgsl) and the avoidance-PBRS term (`danger_shaping`, lines 807–822) from the `raw_gradient` assembly at line 823–826, and rewrite the tests that exist to verify that shaping (`shaped_reward_rewards_approach`, `avoidance_potential_sign`) so they assert the post-removal homeostatic-only behavior; verify the fused `kernel_tick.wgsl` path is the SOLE gradient assembly (`phase_physics.wgsl` does not assemble `raw_gradient`); then do the cleanup the removal makes safe — DELETE the genuinely-orphaned `APPROACH_SHAPING_GAIN`, RENAME the still-live `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS` (it bounds the food-detect scan in `kernel_tick.wgsl`, independent of shaping), and mark the now-unwritten `P_PREV_POTENTIAL`/`P_PREV_DANGER_POTENTIAL` slots RESERVED (keep their consts — layout-parity and integration tests reference them; no `PHYS_STRIDE` shift); add a falsifiable test asserting `raw_gradient == energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT` at default config; and update the brain-crate README to reflect the restored pure-homeostatic design.

See [SCOPE.md](SCOPE.md) for boundaries and [ARCHITECTURE.md](ARCHITECTURE.md) for the deltas.

**Conventions**
- Each task has a stable kebab-case **id** (also its branch `task/{id}` and
  worktree `.makina/worktrees/{plan_slug}--{id}/`).
- **Depends on** lists *direct* prerequisites only ("—" means none).
- **Done when** is the verifiable acceptance criterion; every task must keep
  `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test -p xagent-sandbox` green (stated as "cargo fmt/clippy/test green").
- GPU tests self-skip without an adapter (`GpuKernel::is_available()`); CI runs Mesa lavapipe.
- Line numbers are hints; locate every site by the named symbol (grep).

---

## 0001 — Measurement-Baseline

### measure-baseline-pre-removal — Measure Baseline Learning Signal Pre-Removal

Before removing the PBRS terms, we need a falsifiable baseline to confirm the removal does not break unintended downstream behavior. This task is authored first and gates both removal tasks (`remove-approach-shaping`, `remove-danger-shaping` depend on it) so it executes against the BEFORE state — with approach shaping default on and danger shaping default off per Plan 0009 — capturing the distribution that post-removal tests compare against. The baseline measures `raw_gradient` over a representative run (100 ticks of a single agent in a standard world config); `raw_gradient` is the local in `coop_habituate_homeo` (`brain_passes.wgsl:823`) whose amplified value is published to the workgroup-shared `s_homeo[1u]` (`brain_passes.wgsl:846`) but is NOT written to any CPU-readable physics slot today, so step 1 wires the dedicated debug slot that `add-homeostatic-only-gate` will reuse.

**Steps:**
1. Add a CPU-readable debug slot for `raw_gradient`, following the existing `P_*_OUT` physics-slot idiom. In `crates/xagent-brain/src/buffers.rs`, immediately before `pub const PHYS_STRIDE: usize = 44;`, add a new const `pub const P_RAW_GRADIENT_OUT: usize = 44;` with the doc-comment `/// Pre-amplification homeostatic learning signal raw_gradient (energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT + shaping terms), written by coop_habituate_homeo for CPU readback. Per-agent live state, never serialized.` and bump `PHYS_STRIDE` to `45`. Mirror the const in `crates/xagent-brain/src/shaders/kernel/common.wgsl`: locate `const P_GRADIENT_OUT: u32 = 29u;`, add `const P_RAW_GRADIENT_OUT: u32 = 44u;` near it, and bump the corresponding `PHYS_STRIDE` const in `common.wgsl` to `45u`. Then update the `buffers.rs` PHYS_STRIDE layout-parity test (the one asserting `PHYS_STRIDE == max offset + 1` and enumerating the `P_*` slots, ≈line 1488) to include `P_RAW_GRADIENT_OUT`, so the parity assertion still holds at stride 45.
2. Publish `raw_gradient` to CPU readback in `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`. The local `raw_gradient` is in scope only at the homeostasis assembly site (line 823), not at the telemetry write-out block, so route it through workgroup-shared memory. (a) Change the declaration `var<workgroup> s_homeo: array<f32, 6>;` (line 31) to `var<workgroup> s_homeo: array<f32, 7>;`. (b) Directly after the existing `s_homeo[1u] = raw_gradient_amplified;` (line 846), add `s_homeo[6u] = raw_gradient;` (the un-amplified value). (c) In the `if (tid == 0u)` telemetry write-out block (the one containing `physics_state[phys_base + P_GRADIENT_OUT] = gradient;` around line 1470), add `physics_state[phys_base + P_RAW_GRADIENT_OUT] = s_homeo[6u];`.
3. Expose the slot through telemetry: in `crates/xagent-brain/src/gpu_kernel.rs`, add a `pub raw_gradient: f32,` field to `struct AgentTelemetry` (around line 330, beside `pub gradient: f32,`), and in `read_agent_telemetry_blocking` (around line 2944, beside `let gradient = phys[P_GRADIENT_OUT];`) add `let raw_gradient = phys[P_RAW_GRADIENT_OUT];` and include `raw_gradient,` in the returned struct literal. Do the same for the `try_collect_telemetry` path (around line 3153) and its struct literal.
4. Create a new test in `crates/xagent-brain/tests/` named `learning_signal_baseline.rs` that, after embedding the standard GPU self-skip guard verbatim, invokes a single-agent GPU kernel for 100 ticks with the default config (`BrainConfig::default()`, `WorldConfig::default()`), reading `read_agent_telemetry_blocking(0).raw_gradient` after each tick.
5. Record the mean, std, min, max of `raw_gradient` over the 100 ticks as a doc comment on the test function (e.g., `mean: 0.0125, std: 0.0456, min: -0.0832, max: 0.0921` — exact numbers TBD by the run). Do NOT assert specific values; this is a measurement probe, not a gate.

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- **Depends on:** —
- **Done when:** The `P_RAW_GRADIENT_OUT` debug slot is wired through the kernel and `AgentTelemetry`, and the `learning_signal_baseline.rs` test compiles and runs green, outputting the baseline `raw_gradient` statistics (mean, std, min, max) over 100 ticks recorded in the test's doc comment for use by `add-homeostatic-only-gate`; cargo fmt/clippy/test green.

---

## 0002 — Approach-Shaping-Removal

### remove-approach-shaping — Remove Approach-PBRS Term from raw_gradient

The `coop_habituate_homeo` function in `brain_passes.wgsl` (lines 791–805) computes a potential-based reward shaping term for food seeking. This contradicts the README's stated homeostasis-only design. The term reads the nearest-food distance, normalizes it, applies a gain (`APPROACH_SHAPING_GAIN = 0.05`), and stores/retrieves state from `P_PREV_POTENTIAL` to compute the shaping increment. This increment is then added to `raw_gradient` at line 825. Removing it restores pure homeostatic learning.

**Steps:**
1. Open `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, navigate to the `coop_habituate_homeo` function around line 765.
2. Within the `if (tid == 0u) { … }` block, locate lines 791–805 (the comment "Potential-based approach shaping…" through the assignment to `P_PREV_POTENTIAL`).
3. Delete the entire approach-shaping block (lines 791–805 inclusive): the `let d_norm = …`, `let potential = …`, `let prev_potential = …`, `let shaping = …`, and `physics_state[…P_PREV_POTENTIAL] = …` statements.
4. In their place, insert a single line: `let shaping: f32 = 0.0;` (with a doc comment: `// Shaping term removed: pure homeostatic learning only.`).
5. Verify the `shaping` variable is still consumed in the `raw_gradient` assembly at line 823 (now line ~813 after deletion): `let raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT + shaping + danger_shaping;` — it will now contribute zero. (Leaving the binding avoids touching the assembly line, which `remove-danger-shaping` edits next.)
6. **Update the test that exists to verify the approach shaping.** `shaped_reward_rewards_approach` (`crates/xagent-sandbox/tests/integration.rs:2267`) asserts an approaching agent's published gradient exceeds a receding agent's; its own comment notes *"Deleting the shaping fold makes the two equal — the test [fails]"*. Rewrite it to assert the post-removal truth — approaching and receding agents now receive an **equal** `raw_gradient` (no distance-closing credit) — or delete it with a one-line justification. Then audit the other shaping-dependent assertions in the same file (the distance-closing credit tests around `integration.rs:2369` and `:2703`) and update or remove them so they assert homeostatic-only behavior.
7. Leave `P_NEAREST_FOOD_DISTANCE` and the `agent_food_detect` scan intact (still written by the kernel and read by the nav test at `integration.rs:5598`); only the shaping's *read* of it (`brain_passes.wgsl:800`) is removed here. The `P_PREV_POTENTIAL` *write* is removed; its const/slot stay (handled in `remove-orphaned-consts`).
8. Run `cargo fmt --all` and verify no format changes beyond the edit.

- **Depends on:** measure-baseline-pre-removal
- **Done when:** The approach-shaping block is removed, replaced with `let shaping: f32 = 0.0;`; `shaped_reward_rewards_approach` and the distance-closing credit assertions are rewritten to assert the post-removal homeostatic-only behavior (or removed with justification). The shader compiles and `cargo test -p xagent-sandbox` is green; cargo fmt/clippy/test green.

---

## 0003 — Avoidance-Shaping-Removal

### remove-danger-shaping — Remove Avoidance-PBRS Term from raw_gradient

The `coop_habituate_homeo` function (lines 807–822) computes an avoidance-PBRS term behind the `CFG_DANGER_PERCEPT_ENABLED` flag. Despite the flag (default false), the term is an engineered reward kernel that contradicts homeostasis-only design. It reads danger distance, normalizes it, computes a potential, and stores/retrieves state from `P_PREV_DANGER_POTENTIAL` to yield a shaping increment that is added to `raw_gradient` at line 826. Removing it restores pure homeostatic learning (the danger-percept sensory features remain available for the encoder/predictor to discover naturally).

**Steps:**
1. Open `crates/xagent-brain/src/shaders/kernel/brain_passes.wgsl`, locate the `coop_habituate_homeo` function.
2. Within the `if (tid == 0u) { … }` block, find lines 807–822 (the comment "Avoidance potential shaping…" through the closing `}` of the `if (bc_f32(CFG_DANGER_PERCEPT_ENABLED) != 0.0)` block).
3. Delete the entire block (lines 812–822 inclusive): the `var danger_shaping: f32 = 0.0;` declaration and the `if (bc_f32(CFG_DANGER_PERCEPT_ENABLED) != 0.0) { … }` conditional.
4. In their place, insert: `let danger_shaping: f32 = 0.0;` (with a doc comment: `// Avoidance shaping term removed: danger percept remains available to encoder/predictor for natural discovery.`).
5. Verify the `danger_shaping` variable is still consumed in the `raw_gradient` assembly (line 826, or nearby after prior edits): it will now contribute zero.
6. **Update the test that exists to verify the avoidance shaping.** `avoidance_potential_sign` (`crates/xagent-sandbox/tests/integration.rs:6902`, spanning roughly `:6894-7058`) drives `P_PREV_DANGER_POTENTIAL` across two ticks and asserts the danger-potential telescoping sign. With the shaping removed the slot is never written (stays zero), so the "on" telescoping cases no longer hold — rewrite them to assert the shaping no longer fires (e.g. the published gradient is identical whether danger nears or recedes), or delete them with a one-line justification. The flag-off `P_PREV_DANGER_POTENTIAL == 0.0` case still holds because the slot is retained and respawn-zeroed.
7. The `P_PREV_DANGER_POTENTIAL` *write* is removed; its const/slot stay (referenced by the `buffers.rs` layout-parity test and the integration test above; kept as reserved in `remove-orphaned-consts`). The danger-percept sensory channel (distance+bearing features) is untouched.
8. Run `cargo fmt --all` and verify no format changes beyond the edit.

- **Depends on:** measure-baseline-pre-removal, remove-approach-shaping
- **Done when:** The avoidance-shaping conditional block is removed, replaced with `let danger_shaping: f32 = 0.0;`; `avoidance_potential_sign` is rewritten to assert the post-removal behavior (or removed with justification). The shader compiles and `cargo test -p xagent-sandbox` is green; cargo fmt/clippy/test green.

---

## 0004 — Dead-Code-Cleanup

### remove-orphaned-consts — Clean Up the Shaping Consts: Delete the Orphan, Rename the Survivor, Reserve the Slots

Removing the two shaping terms (`remove-approach-shaping`, `remove-danger-shaping`) leaves the supporting machinery in **three different states** — verified by grep, each handled correctly rather than blanket-deleted:
- `APPROACH_SHAPING_GAIN` (`common.wgsl:500`) is **genuinely orphaned** — after the approach block is gone it is referenced nowhere in source (no test references it; only historical `docs/plans/0004-*` mentions remain, which is fine). **Delete it.**
- `SHAPING_RADIUS` (`common.wgsl:427`) is **NOT orphaned** — `kernel_tick.wgsl` uses it at six sites in `agent_food_detect` (≈lines 361, 373, 387, 427–444, 588) to bound the nearest-food scan that writes `P_NEAREST_FOOD_DISTANCE`, independent of shaping. **Rename it to `FOOD_SENSE_RADIUS`** so the name matches its real (food-sense) role, updating every reference.
- `P_PREV_POTENTIAL` (`buffers.rs:193`, slot 33) and `P_PREV_DANGER_POTENTIAL` (`buffers.rs:217`, slot 41) are no longer *written*, but their consts are referenced by the `buffers.rs` PHYS_STRIDE layout-parity test (≈1231, 1249–1250, 1488, 1496) and `P_PREV_DANGER_POTENTIAL` by `integration.rs` (≈6903–7058). **Keep the consts and slots** (no layout shift) and mark them **reserved** in their doc-comments.

**Steps:**
1. In `crates/xagent-brain/src/shaders/kernel/common.wgsl`, delete the `APPROACH_SHAPING_GAIN` const (≈line 500) and its doc-comment. Confirm with `grep -rn APPROACH_SHAPING_GAIN crates/` that no source reference remains.
2. Rename `SHAPING_RADIUS`→`FOOD_SENSE_RADIUS`: update the definition in `common.wgsl` (≈427) and **every** reference in `kernel_tick.wgsl` (≈361, 373, 387, 427–444, 588) and the doc-comments in `buffers.rs` (≈183–184, 231). The new name's comment must be purely technical (e.g. "radius of the nearest-food sense scan") — no plan/process language (`contributing_guard.rs` enforces this).
3. In `crates/xagent-brain/src/buffers.rs`, **keep** the `P_PREV_POTENTIAL` and `P_PREV_DANGER_POTENTIAL` consts but update their doc-comments to note they are *reserved* (no longer written after the shaping removal). Do **not** delete them and do **not** shift `PHYS_STRIDE`. Leave the respawn zeroing of those slots (`phase_death.wgsl`, `kernel_tick.wgsl`) in place — it is harmless and keeps the flag-off integration assertion valid.
4. Run `grep -rn 'SHAPING_RADIUS\|APPROACH_SHAPING_GAIN' crates/` and confirm zero `APPROACH_SHAPING_GAIN` and zero stale `SHAPING_RADIUS` (all now `FOOD_SENSE_RADIUS`).
5. Run `cargo build -p xagent-brain` and `cargo build -p xagent-sandbox`; both compile. Run `cargo fmt --all`.

- **Depends on:** remove-approach-shaping, remove-danger-shaping, verify-gradient-parity
- **Done when:** `APPROACH_SHAPING_GAIN` is deleted; `SHAPING_RADIUS` is renamed to `FOOD_SENSE_RADIUS` everywhere (food-detect still compiles and runs); `P_PREV_POTENTIAL`/`P_PREV_DANGER_POTENTIAL` consts and slots are retained as reserved with updated doc-comments and **no `PHYS_STRIDE` shift**. The full workspace builds and `cargo test` passes; `cargo clippy --workspace --all-targets -- -D warnings` is clean; cargo fmt/clippy/test green.

---

## 0005 — Parity-And-Tests

### verify-gradient-parity — Verify Fused and Split Kernel Gradient Paths Compute Identically

The brain's learning signal is computed in at least one path: the fused kernel in `kernel_tick.wgsl` / `brain_passes.wgsl`. A secondary split-kernel path may exist in `phase_physics.wgsl` or other shaders that also computes gradients. After PBRS removal, both paths must compute `raw_gradient` identically (pure homeostatic deltas, no shaping). This probe verifies parity and flags any divergence.

**Steps:**
1. Search for all locations where `raw_gradient` is computed or where homeostatic/learning signals are assembled: `grep -n "raw_gradient\|energy_delta.*ENERGY_WEIGHT\|integrity_delta.*INTEGRITY_WEIGHT" /Users/koraytaylan/Workspace/xagent/.claude/worktrees/nice-liskov-fe972d/crates/xagent-brain/src/shaders/kernel/*.wgsl` to identify all compute paths.
2. For each path found, verify it computes the same expression post-removal. Expected result: all paths should compute `raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT` (or equivalent, with no `shaping` or `danger_shaping` terms).
3. Confirm there is NO second gradient assembly: `phase_physics.wgsl` only snapshots `P_PREV_ENERGY`/`P_PREV_INTEGRITY` and does not assemble `raw_gradient`, so the fused path (`kernel_tick.wgsl` → `coop_habituate_homeo` in `brain_passes.wgsl`) is the sole assembly. If grep surfaces any other assembly site, apply the same shaping-removal edits there to keep it in sync.
4. Document the findings in a comment or test doc: list all paths found, confirm they are now identical post-removal.

- **Depends on:** remove-approach-shaping, remove-danger-shaping
- **Done when:** All kernel paths that compute the learning signal are identified and verified to compute `raw_gradient = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT` with no shaping terms. If multiple paths exist, they are confirmed to be byte-identical in their gradient assembly. Documentation lists the paths and confirms parity; cargo fmt/clippy/test green.

---

### add-homeostatic-only-gate — Add Falsifiable Test: raw_gradient is Homeostatic-Only at Default Config

To ensure the removal is correct and prevent future reintroduction of shaping terms, we add a falsifiable test that asserts `raw_gradient` contains only the homeostatic terms (energy delta and integrity delta) at default config, with no reward-shaping components. The test constructs a scenario with known zero deltas and verifies the gradient is zero; it also tests non-zero deltas and confirms the result matches the weighted sum. It reads `raw_gradient` directly through the `AgentTelemetry::raw_gradient` field that `measure-baseline-pre-removal` already wired into the kernel (the `P_RAW_GRADIENT_OUT` physics slot exposed via `read_agent_telemetry_blocking`); no new readback plumbing is invented here, so there are no design decisions left to the executor.

**Steps:**
1. Create a new test function `test_raw_gradient_is_homeostatic_only()` in a new file `crates/xagent-brain/tests/raw_gradient_homeostatic_only.rs`. The test instantiates a single-agent GPU kernel at default config, ticks it, and reads the post-tick `raw_gradient` via `kernel.read_agent_telemetry_blocking(0).raw_gradient` — the field `measure-baseline-pre-removal` added to `AgentTelemetry` (sourced from the `P_RAW_GRADIENT_OUT` physics slot, never the amplified `s_homeo[1u]` value). Embed the standard GPU self-skip guard verbatim as the first lines of the test (see the fenced block below).
2. Test Case 1 — Zero Deltas: construct the physics state so current energy and integrity equal their previous values (`energy_delta = 0.0`, `integrity_delta = 0.0`). Tick the kernel and assert `read_agent_telemetry_blocking(0).raw_gradient == 0.0`.
3. Test Case 2 — Energy Only: set `energy_delta = 0.1`, `integrity_delta = 0.0`. Tick and assert `read_agent_telemetry_blocking(0).raw_gradient` is within `1e-6` of `0.1 * ENERGY_WEIGHT` (`ENERGY_WEIGHT` from `common.wgsl`; use the value verified by grep in the test's doc comment).
4. Test Case 3 — Integrity Only: set `energy_delta = 0.0`, `integrity_delta = 0.05`. Tick and assert `read_agent_telemetry_blocking(0).raw_gradient` is within `1e-6` of `0.05 * INTEGRITY_WEIGHT`.
5. Document the test with a doc comment: 'Verifies that raw_gradient contains only homeostatic deltas (energy + integrity), with both approach-PBRS and avoidance-PBRS terms removed. This is a regression guard: if a future edit reintroduces shaping, this test will fail.'
6. Embed the baseline statistics from the `measure-baseline-pre-removal` probe in the test's doc comment for reference (e.g., 'Pre-removal baseline: mean raw_gradient 0.0125, std 0.0456').

```rust
if !xagent_brain::GpuKernel::is_available() {
    eprintln!("Skipping: no GPU/fallback adapter available");
    return;
}
```

- **Depends on:** remove-approach-shaping, remove-danger-shaping, measure-baseline-pre-removal
- **Done when:** The test `test_raw_gradient_is_homeostatic_only()` is added, compiles, and passes, reading `raw_gradient` via `AgentTelemetry::raw_gradient`. It asserts zero deltas yield a zero gradient, and non-zero deltas yield the expected weighted sum with no shaping component. The test is a permanent regression guard; cargo fmt/clippy/test green.

---

## 0006 — Documentation-Update

### update-readme-vision — Document Restored Homeostasis-Only Learning in README and Crate Docs

The README (§1, §10) and the brain crate README state that the system uses homeostasis-only evaluation with no reward functions. After Plan 0012 removes the PBRS terms, this design is now true by construction. Documentation should acknowledge the alignment and clarify the restored design for future readers.

**Steps:**
1. Open `README.md` and review §1 (Project Vision) and §10 (Why Homeostasis-Only Evaluation?). Verify the wording already correctly describes the homeostasis-only design (it does, based on earlier reading). No edits to README.md are required; the current text is aspirational and now matches reality.
2. Open `crates/xagent-brain/README.md` (the deep dive into `GpuKernel` internals). Locate any section describing the learning signal, gradient computation, or the 7-stage pipeline. If it mentions PBRS or shaping terms, update it to reflect the new post-Plan-0012 design: raw_gradient is purely homeostatic (energy_delta*ENERGY_WEIGHT + integrity_delta*INTEGRITY_WEIGHT).
3. In the brain README, verify the section on 'Homeostatic Feedback' (or similar) correctly states that the gradient is the only evaluative signal and is derived solely from energy/integrity deltas.
4. (Optional) Add a one-line doc comment in the brain_passes.wgsl file, at the start of the `coop_habituate_homeo` function, to document the design: '// Pass 3: Habituate & Homeostasis — computes pure homeostatic gradients (energy + integrity deltas, no reward shaping).'
5. Run `cargo fmt --all` to ensure no formatting issues.

- **Depends on:** remove-approach-shaping, remove-danger-shaping, verify-gradient-parity
- **Done when:** README.md and `crates/xagent-brain/README.md` correctly document the homeostasis-only design. Any references to PBRS or shaping mechanisms are removed or clarified as historical context (e.g., 'Plans 0004 and 0009 explored reward shaping; Plan 0012 removed them, restoring the original homeostasis-only vision'). Documentation is consistent with the code. `cargo fmt` is clean; cargo fmt/clippy/test green.

---

**End of plan 0012 TASKS.** When every "Done when" bullet is green, the plan's end state is reached.
