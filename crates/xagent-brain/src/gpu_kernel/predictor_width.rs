//! Optional FP32 predictor reductions with configurable lanes per row.
//! All invocations retain their original per-weight update and clamp; only
//! the assignment of dot terms to lanes and the final addition tree change.

/// Supported widths divide both the 256-thread group and 128 input columns.
pub(super) const LANE_WIDTHS: [u32; 4] = [4, 8, 16, 32];

/// The production brain entry launches this many invocations per agent.
const WORKGROUP_INVOCATIONS: u32 = 256;
/// The original predictor maps contiguous groups of four lanes to each row.
const ORIGINAL_REDUCTION: &str = r"            if (lane == 0u) {
                let base = tid; // When lane==0, tid = output_in_tile*4
                let reduced = s_dense_partials[base] + s_dense_partials[base + 1u] + s_dense_partials[base + 2u] + s_dense_partials[base + 3u];
                s_prediction[dim] = reduced;
            }
";

/// Compose the inline predictor's lane layout without changing shared storage.
/// The four-lane path returns the source unchanged; wider paths change FP32
/// association and therefore can change later simulation trajectories.
pub(super) fn wider_predictor(source: &str, lanes: u32) -> String {
    assert!(LANE_WIDTHS.contains(&lanes));
    if lanes == LANE_WIDTHS[0] {
        return source.to_owned();
    }
    let start = source.find("fn coop_predict_and_act(").unwrap();
    let end = start
        + source[start..]
            .find("    // ── Recalled cosine similarities:")
            .unwrap();
    let block = &source[start..end];
    assert_eq!(block.matches(ORIGINAL_REDUCTION).count(), 1);
    let reduction = format!(
        r"            for (var stride = {lanes}u / 2u; stride > 0u; stride /= 2u) {{
                if (lane < stride) {{
                    s_dense_partials[tid] += s_dense_partials[tid + stride];
                }}
                workgroupBarrier();
            }}
            if (lane == 0u) {{
                s_prediction[dim] = s_dense_partials[tid];
            }}
"
    );
    let candidate = block
        .replace("64 rows × 4 lanes", "contiguous lanes per output row")
        .replace(";   // 0..63", ";")
        .replace(";              // 0..3", ";")
        .replace("DENSE_INNER_LANES", &format!("{lanes}u"))
        .replace(
            "DENSE_OUTPUT_TILE",
            &format!("{}u", WORKGROUP_INVOCATIONS / lanes),
        )
        .replace(ORIGINAL_REDUCTION, &reduction);
    format!("{}{}{}", &source[..start], candidate, &source[end..])
}
