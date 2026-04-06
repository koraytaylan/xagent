// Pass 5: Select top-K recalled patterns from similarity scores

@group(0) @binding(0) var<storage, read_write> similarities: array<f32>;
@group(0) @binding(1) var<storage, read_write> patterns: array<f32>;
@group(0) @binding(2) var<storage, read_write> recall_buf: array<f32>;
@group(0) @binding(3) var<storage, read> brain_state: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let s_base = agent * MEMORY_CAP;
    let p_base = agent * PATTERN_STRIDE;
    let r_base = agent * RECALL_IDX_STRIDE;
    let b_base = agent * BRAIN_STRIDE;
    let tick = brain_state[b_base + O_TICK_COUNT];

    var count: u32 = 0u;

    for (var k: u32 = 0u; k < RECALL_K; k = k + 1u) {
        var best_idx: u32 = 0u;
        var best_sim: f32 = -3.0;

        for (var j: u32 = 0u; j < MEMORY_CAP; j = j + 1u) {
            let sim = similarities[s_base + j];
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = j;
            }
        }

        if (best_sim <= -1.5) {
            break;
        }

        recall_buf[r_base + k] = f32(best_idx);
        count = count + 1u;

        // Update recalled pattern metadata
        patterns[p_base + O_PAT_META + best_idx * 3u + 1u] = tick;  // last_accessed
        patterns[p_base + O_PAT_META + best_idx * 3u + 2u] += 1.0;  // activation_count

        // Exclude from future iterations
        similarities[s_base + best_idx] = -3.0;
    }

    // Zero remaining slots
    for (var k: u32 = count; k < RECALL_K; k = k + 1u) {
        recall_buf[r_base + k] = 0.0;
    }

    // Store count at the end
    recall_buf[r_base + RECALL_K] = f32(count);
}
