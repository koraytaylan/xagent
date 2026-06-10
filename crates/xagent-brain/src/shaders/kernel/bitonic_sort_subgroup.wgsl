// ── Subgroup-accelerated bitonic sort (replaces the workgroup-memory sort) ──
// This fragment is spliced between the `BEGIN_BITONIC_SORT` /
// `END_BITONIC_SORT` markers in `brain_passes.wgsl` by the Rust host before
// creating the shader module/pipeline, when the `wgpu::Features::SUBGROUP`
// feature is present AND the adapter guarantees a subgroup width of at least 32
// (see the width contract below). This is host-side shader composition, not
// WGSL `override` specialization via `PipelineCompilationOptions::constants`.
//
// The fragment is self-contained — it relies on the same shared memory
// (`s_similarities`, `shared_sort_indices`) declared in brain_passes.wgsl,
// and on local variables `tid` (thread id, 0..255) and `sgid` (subgroup
// invocation id) being in scope at the splice site.
//
// ── Subgroup-width contract ──
// This fragment REQUIRES a subgroup width of at least 32 invocations. Stages
// 0–4 pair lanes via `subgroupShuffle(my_val, sgid ^ half)` with `half` up to
// `1 << 4 == 16`, so lanes 0..=31 must all be valid members of the same
// subgroup. The mapping of `tid` onto subgroup lanes must also be contiguous in
// aligned blocks of the subgroup width — true for the 1-D workgroup used here,
// where `sgid == tid % subgroup_size` — so that `tid ^ half` and `sgid ^ half`
// address the same bitonic partner. On 16-wide subgroups (some Intel iGPUs,
// some AMD configs) `sgid ^ 16` would address a lane outside the subgroup and
// return an implementation-defined value, corrupting top-K recall. The Rust
// host therefore only splices this fragment in when the adapter reports
// `min_subgroup_size >= 32` — see `subgroup_bitonic_supported()` in
// gpu_kernel.rs.
//
// See `gpu_kernel.rs::apply_subgroup_markers()` for the full marker contract.

    // ── Subgroup-accelerated stages 0–4 (15 barrier-free passes) ──
    // Each of 128 threads holds one element in registers.
    // subgroupShuffle exchanges values within 32-wide subgroups.
    var my_val: f32 = -3.0;
    var my_index: u32 = 0u;
    if (tid < MEMORY_CAP) {
        my_val = s_similarities[tid];
        my_index = shared_sort_indices[tid];
    }

    for (var stage: u32 = 0u; stage < 5u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < MEMORY_CAP) {
                let half = 1u << (stage - step);
                let partner_tid = tid ^ half;
                let partner_val = subgroupShuffle(my_val, sgid ^ half);
                let partner_idx = subgroupShuffle(my_index, sgid ^ half);

                let i = min(tid, partner_tid);
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;
                let i_am_low = tid < partner_tid;
                let want_max = i_am_low == descending;
                if ((my_val < partner_val) == want_max) {
                    my_val = partner_val;
                    my_index = partner_idx;
                }
            }
            // No barrier needed — subgroup ops are synchronous within subgroup
        }
    }

    // Write back to shared memory for stages 5–6
    if (tid < MEMORY_CAP) {
        s_similarities[tid] = my_val;
        shared_sort_indices[tid] = my_index;
    }
    workgroupBarrier();

    // ── Shared-memory stages 5–6 (13 passes with barriers) ──
    for (var stage: u32 = 5u; stage < 7u; stage = stage + 1u) {
        for (var step: u32 = 0u; step <= stage; step = step + 1u) {
            if (tid < 64u) {
                let block_size = 1u << (stage + 1u - step);
                let half = block_size >> 1u;
                let group = tid / half;
                let local_id = tid % half;
                let i = group * block_size + local_id;
                let j = i + half;
                let descending = ((i >> (stage + 1u)) & 1u) == 0u;

                let val_i = s_similarities[i];
                let val_j = s_similarities[j];
                let idx_i = shared_sort_indices[i];
                let idx_j = shared_sort_indices[j];

                let should_swap = (descending && val_i < val_j) || (!descending && val_i > val_j);
                if (should_swap) {
                    s_similarities[i] = val_j;
                    s_similarities[j] = val_i;
                    shared_sort_indices[i] = idx_j;
                    shared_sort_indices[j] = idx_i;
                }
            }
            workgroupBarrier();
        }
    }
