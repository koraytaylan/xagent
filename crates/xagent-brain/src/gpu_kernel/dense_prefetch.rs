//! Optional scalar prefetch for ordered encoder and fused predictor loops.
//! Independent loads precede arithmetic, while each lane retains its original
//! stride-four addition sequence and each weight retains its FP32 clamp.

use std::fmt::Write;

/// Both factors expose independent loads without allocating shared arrays.
const PREFETCH_FACTORS: [u32; 2] = [4, 8];

/// Exact production encoder loop; its bias and reduction remain outside it.
const ENCODER_LOOP: &str = r"        for (var f = lane; f < FEATURE_COUNT; f += DENSE_INNER_LANES) {
            partial += s_features[f] * brain_state[brain_base + O_ENC_WEIGHTS + f * ENCODED_DIMENSION + dim];
        }
";

/// Exact loop produced by fuse_inline_predictor; source drift fails loudly.
const PREDICTOR_LOOP: &str = r"            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES) {
                let previous_input = brain_state[brain_base + O_PREV_ENCODED + j];
                let grad = clamp(transition_error * tanh_derivative * previous_input, -1.0, 1.0);
                var w = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] - predictor_learning_rate * grad;
                w = clamp(w, -3.0, 3.0);
                brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + j] = w;
                partial += s_encoded[j] * w;
            }
";

fn encoder_prefetch_loop(factor: u32) -> String {
    let mut source = format!(
        "        for (var f = lane; f < FEATURE_COUNT; f += DENSE_INNER_LANES * {factor}u) {{\n"
    );
    for item in 0..factor {
        writeln!(
            source,
            "            let feature_index_{item} = f + {item}u * DENSE_INNER_LANES;"
        )
        .unwrap();
        writeln!(
            source,
            "            var feature_{item}: f32 = 0.0;\n            var weight_{item}: f32 = 0.0;"
        )
        .unwrap();
        writeln!(source, "            if (feature_index_{item} < FEATURE_COUNT) {{\n                feature_{item} = s_features[feature_index_{item}];\n                weight_{item} = brain_state[brain_base + O_ENC_WEIGHTS + feature_index_{item} * ENCODED_DIMENSION + dim];\n            }}").unwrap();
    }
    // All valid loads precede the first addition. The guards omit absent tail
    // terms completely, preserving signed-zero and NaN arithmetic behavior.
    for item in 0..factor {
        writeln!(source, "            if (feature_index_{item} < FEATURE_COUNT) {{\n                partial += feature_{item} * weight_{item};\n            }}").unwrap();
    }
    source.push_str("        }\n");
    source
}

fn predictor_prefetch_loop(factor: u32) -> String {
    let mut source = format!(
        "            for (var j = lane; j < ENCODED_DIMENSION; j += DENSE_INNER_LANES * {factor}u) {{\n"
    );
    for item in 0..factor {
        writeln!(
            source,
            "                let input_index_{item} = j + {item}u * DENSE_INNER_LANES;"
        )
        .unwrap();
        writeln!(source, "                var previous_input_{item}: f32 = 0.0;\n                var old_weight_{item}: f32 = 0.0;\n                var encoded_{item}: f32 = 0.0;\n                var updated_weight_{item}: f32 = 0.0;").unwrap();
        writeln!(source, "                if (input_index_{item} < ENCODED_DIMENSION) {{\n                    previous_input_{item} = brain_state[brain_base + O_PREV_ENCODED + input_index_{item}];\n                    old_weight_{item} = brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + input_index_{item}];\n                    encoded_{item} = s_encoded[input_index_{item}];\n                }}").unwrap();
    }
    // Each invocation exclusively owns these columns. Updating independent
    // weights before consuming them changes no value or cross-lane dependency.
    for item in 0..factor {
        writeln!(source, "                if (input_index_{item} < ENCODED_DIMENSION) {{\n                    let grad_{item} = clamp(transition_error * tanh_derivative * previous_input_{item}, -1.0, 1.0);\n                    var weight_{item} = old_weight_{item} - predictor_learning_rate * grad_{item};\n                    weight_{item} = clamp(weight_{item}, -3.0, 3.0);\n                    brain_state[brain_base + O_PREDICTOR_WEIGHTS + dim * ENCODED_DIMENSION + input_index_{item}] = weight_{item};\n                    updated_weight_{item} = weight_{item};\n                }}").unwrap();
    }
    for item in 0..factor {
        writeln!(source, "                if (input_index_{item} < ENCODED_DIMENSION) {{\n                    partial += encoded_{item} * updated_weight_{item};\n                }}").unwrap();
    }
    source.push_str("            }\n");
    source
}

pub(super) fn prefetch_passes(baseline: &str, factor: u32) -> String {
    assert!(PREFETCH_FACTORS.contains(&factor));
    assert_eq!(baseline.matches(ENCODER_LOOP).count(), 1);
    assert_eq!(baseline.matches(PREDICTOR_LOOP).count(), 1);
    assert!(baseline.contains("const DENSE_INNER_LANES: u32 = 4u;"));
    let source = baseline
        .replacen(ENCODER_LOOP, &encoder_prefetch_loop(factor), 1)
        .replacen(PREDICTOR_LOOP, &predictor_prefetch_loop(factor), 1);
    // This transformation touches no phase boundary or shared-storage resource.
    for unchanged in ["workgroupBarrier();", "storageBarrier();", "var<workgroup>"] {
        assert_eq!(
            source.matches(unchanged).count(),
            baseline.matches(unchanged).count()
        );
    }
    source
}
