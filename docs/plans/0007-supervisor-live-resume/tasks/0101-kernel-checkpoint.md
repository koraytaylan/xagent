---
id: kernel-checkpoint
title: Kernel checkpoint capture in one submit and restore
workstream: "0100"
kind: task
role: developer
depends_on: []
gated: false
touches:
  - crates/xagent-brain/src/gpu_kernel.rs
  - crates/xagent-brain/src/gpu_kernel/checkpoint.rs
  - crates/xagent-brain/tests/kernel_checkpoint.rs
status: todo
merged_as: ""
checks:
  - name: focused-proof
    command: "env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test kernel_checkpoint"
    timeout_ms: 3600000
  - name: brain-clippy
    command: "cargo clippy -p xagent-brain --all-targets -- -D warnings"
    timeout_ms: 3600000
  - name: brain-fmt
    command: "cargo fmt -p xagent-brain -- --check"
    fix: "cargo fmt -p xagent-brain"
---

# Kernel checkpoint capture in one submit and restore

## Outcome

`GpuKernel` captures its persistent state as a `KernelCheckpoint` with one
command encoder, one staging buffer and exactly one counted `queue.submit`, and
restores that checkpoint into a freshly built kernel so that further dispatches
are bit-identical to the original kernel. No product behaviour changes: no
shader edit, no compute pass, no extra readback.

## Decisions (explicit)

- New submodule `crates/xagent-brain/src/gpu_kernel/checkpoint.rs`, declared by
  `mod checkpoint;` in `gpu_kernel.rs`, which `pub use`s `KernelCheckpoint`,
  `KernelSection`, `KernelCpuState` and `KernelRestoreError`.
- No serde and no dependency. `crates/xagent-brain/Cargo.toml` and `Cargo.lock`
  are not edited.
- `KernelSection { name: &'static str, bytes: Vec<u8> }`. Section names are
  stable string constants in `checkpoint.rs`, one per captured buffer
  (`agent_phys`, `decision`, `food_state`, `food_flags`, `sensory`,
  `sensory_next`, `brain_state`, `brain_scratch`, `pattern`, `brain_config`,
  `trail_ring`, plus any persistent submodule buffer such as `global_credit`'s).
- `KernelCpuState` holds `food_kind_draws: u64`, `nourishing_food_kind`,
  `active_config_index`, `freeze_steering_weights`, `trunk_lesions` and every
  other non-cache field that alters dispatch output. Hand-written
  `to_bytes(&self) -> Vec<u8>` and `from_bytes(&[u8]) -> Result<Self,
  KernelRestoreError>` use a fixed layout: fields in declaration order, integers
  via `to_le_bytes`, `bool` as one byte (nonzero is true), `f32` as
  `to_bits()` written little-endian. A constant `CPU_STATE_LEN` gives the byte
  length; `from_bytes` refuses any other length with
  `KernelRestoreError::CpuStateLength { expected, actual }`.
- `KernelRestoreError` variants: `UnknownSection(String)`,
  `MissingSection(&'static str)`, `LengthMismatch { name: &'static str,
  expected: usize, actual: usize }`, `CpuStateLength { expected: usize, actual:
  usize }`; implements `Display` and `std::error::Error`.
- Accessors, exact names: `KernelCheckpoint::physics(&self) -> &[f32]` returns
  the `agent_phys` rows; `KernelCheckpoint::food(&self) -> Option<&[f32]>`
  returns the `food_state` rows, `None` when that section is empty.
- `GpuKernel::queue_submit_count(&self) -> u64` is backed by a counter field
  incremented by one private helper that every production `self.queue.submit`
  in `gpu_kernel.rs` calls, then calls `queue.submit`. `read_full_state_blocking`
  goes through the same helper.
- `read_checkpoint_blocking` also updates `state_cache`, `food_cache` and the
  staging sequence exactly as `read_full_state_blocking` does for `state_cache`.
- Buffers not captured are listed in a `DERIVED_BUFFERS` doc table in
  `checkpoint.rs` with a reason each. Expected set: `heightmap` and `biome`
  (rebuilt by `upload_world`), world-config uniforms, `dispatch_args`,
  `food_grid`, `agent_grid`, `collision_scratch` and the staging buffers. Step 1
  decides the final list.
- `restore_checkpoint` validates every section name and length before the first
  write, then `queue.write_buffer`s each section, restores the CPU fields, and
  invalidates the CPU-side caches (`invalidate_packed_encoder`, context,
  food-grid and physics caches).

## Non-goals

- No change to `upload_world`, `replace_agent_slot`, shader code or dispatch
  order. No supervisor, store, headless or worker code (other tasks).
- No new dependency, no dependency version change, no `Cargo.lock` change.
- No checkpoint file format or envelope; in-memory types only.

## Steps

**Steps:**

1. Audit every `wgpu::Buffer` field of `GpuKernel` in `gpu_kernel.rs` and every
   `wgpu::Buffer` owned by a submodule under `gpu_kernel/`. Classify each as
   captured (carries state from one batch to the next) or derived (rebuilt from
   the seed or rewritten before every read). Write the `DERIVED_BUFFERS` table
   in `checkpoint.rs` with a reason per derived buffer.
2. Create `gpu_kernel/checkpoint.rs` with the section name constants,
   `KernelSection`, `KernelCpuState` (with `CPU_STATE_LEN`, `to_bytes`,
   `from_bytes`), `KernelCheckpoint` with the `physics` and `food` accessors, and
   `KernelRestoreError` with `Display` and `Error`. Declare `mod checkpoint;` and
   the `pub use` lines in `gpu_kernel.rs`.
3. In `gpu_kernel.rs`, add the submit counter field, `queue_submit_count`, and
   the counting submit helper. Route every production `self.queue.submit` through
   it with behaviour unchanged. Keep `probe_kernel_batches` and the existing probe
   counters as they are.
4. Implement `read_checkpoint_blocking(&mut self) -> KernelCheckpoint`: one
   command encoder copies every captured buffer into one staging buffer at
   recorded offsets, exactly one counted submit, no compute pass,
   `device.poll(Maintain::Wait)`, then slices the staging buffer into sections,
   fills `KernelCpuState` from the CPU fields, and updates `state_cache`,
   `food_cache` and the staging sequence like `read_full_state_blocking`.
5. Implement `restore_checkpoint(&mut self, &KernelCheckpoint) ->
   Result<(), KernelRestoreError>`: validate all section names, lengths and the
   CPU state length first, returning the error with no write on failure. Then
   `write_buffer` each section, restore the CPU fields and invalidate the caches.
6. Create `crates/xagent-brain/tests/kernel_checkpoint.rs` with a fail-closed GPU
   helper modelled on `gpu_adapter_present_when_required` in
   `crates/xagent-sandbox/tests/integration.rs`, and these tests:
   `kernel_cpu_state_round_trips_bits` (no GPU; `to_bytes` then `from_bytes`
   yields bit-equal fields including a NaN payload and `-0.0`, and a wrong-length
   input yields `CpuStateLength`), `kernel_checkpoint_restore_continues_bit_identically`,
   `kernel_checkpoint_one_submit` and `kernel_checkpoint_refuses_wrong_length`.
   Build kernels the way `crates/xagent-brain/tests/supervisor_kernel.rs` does
   (`GpuKernel::new` plus `upload_world` from the same world data). The
   bit-identical test seeds food and at least four agents, runs batches,
   captures, restores into a fresh kernel, runs N more ticks on both, and
   compares physics rows, food rows and every `read_agent_state` by bit.

- **Done when:** `env XAGENT_REQUIRE_GPU=1 cargo test -p xagent-brain --test kernel_checkpoint` passes `kernel_checkpoint_restore_continues_bit_identically` (INV-5: physics, food and every agent state bit-equal after N more ticks), `kernel_checkpoint_one_submit` (INV-4: `read_checkpoint_blocking` raises `queue_submit_count` by exactly 1, as `read_full_state_blocking` does, and `probe_kernel_batches` is unchanged), `kernel_checkpoint_refuses_wrong_length` and `kernel_cpu_state_round_trips_bits`, and the `focused-proof` check is the proof.
