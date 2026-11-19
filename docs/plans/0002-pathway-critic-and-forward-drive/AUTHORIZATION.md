# Authorization record

## Corrected requirement: `witness-config-size` baseline (2026-11-19)

Authority: the plan owner's standing authorization for automatic repair of
this plan's run.

- Superseded: preflight `witness-config-size`, the `brain-slot-layout`
  constraint check of the same name, and `preflight-brain-slot-layout` required
  `pub const CONFIG_SIZE: usize = 16;` in `crates/xagent-brain/src/buffers.rs`.
- Contradiction: plan 0001 (landed in `abda4da`) added `CFG_LESION_*` and
  `CFG_BRAIN_TICK_PROBE` at slots 16-20 and grew `CONFIG_SIZE` to 24, exactly
  as ARCHITECTURE.md and task 0103 already anticipate; the witness therefore
  failed on the real baseline.
- Correction: the three checks now require
  `pub const CONFIG_SIZE: usize = 24;` (source: `buffers.rs` line with
  `CONFIG_SIZE: usize = 24`, slots 16-20 used, 21-23 padding). Task 0103's
  expected slot assignment (21, 22, size unchanged) is unaffected; no bound,
  task, dependency or behavior changed.
- Witness: `grep -q -F 'pub const CONFIG_SIZE: usize = 24;' crates/xagent-brain/src/buffers.rs`
  exits 0; all 58 preflight and constraint checks pass on the baseline.
