// Pass 1: Sensory input -> feature vector
// Reads packed sensory data (267 f32 per agent), produces features (265 f32 per agent).
// Feature layout: vision RGBA (192), vision depth (48), then non-visual (25).

@group(0) @binding(0) var<storage, read> sensory: array<f32>;
@group(0) @binding(1) var<storage, read_write> features: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let s_base = agent * SENSORY_STRIDE;
    let f_base = agent * FEATURES_STRIDE;

    var fi: u32 = 0u;

    // Vision color: RGBA values (direct copy)
    for (var i: u32 = 0u; i < VISION_COLOR_COUNT; i = i + 1u) {
        features[f_base + fi] = sensory[s_base + i];
        fi = fi + 1u;
    }

    // Vision depth (direct copy)
    for (var i: u32 = 0u; i < VISION_DEPTH_COUNT; i = i + 1u) {
        features[f_base + fi] = sensory[s_base + VISION_COLOR_COUNT + i];
        fi = fi + 1u;
    }

    // Proprioception: velocity magnitude (1)
    let vel_offset = VISION_COLOR_COUNT + VISION_DEPTH_COUNT;
    let vx = sensory[s_base + vel_offset];
    let vy = sensory[s_base + vel_offset + 1u];
    let vz = sensory[s_base + vel_offset + 2u];
    features[f_base + fi] = sqrt(vx * vx + vy * vy + vz * vz);
    fi = fi + 1u;

    // Facing direction (3)
    let fac_offset = vel_offset + 3u;
    features[f_base + fi] = sensory[s_base + fac_offset];
    fi = fi + 1u;
    features[f_base + fi] = sensory[s_base + fac_offset + 1u];
    fi = fi + 1u;
    features[f_base + fi] = sensory[s_base + fac_offset + 2u];
    fi = fi + 1u;

    // Angular velocity (1)
    let ang_offset = fac_offset + 3u;
    features[f_base + fi] = sensory[s_base + ang_offset];
    fi = fi + 1u;

    // Energy signal (1)
    features[f_base + fi] = sensory[s_base + ang_offset + 1u];
    fi = fi + 1u;

    // Integrity signal (1)
    features[f_base + fi] = sensory[s_base + ang_offset + 2u];
    fi = fi + 1u;

    // Energy delta (1)
    features[f_base + fi] = sensory[s_base + ang_offset + 3u];
    fi = fi + 1u;

    // Integrity delta (1)
    features[f_base + fi] = sensory[s_base + ang_offset + 4u];
    fi = fi + 1u;

    // Touch contacts: 4 slots x 4 features = 16
    let touch_offset = ang_offset + 5u;
    for (var t: u32 = 0u; t < 4u; t = t + 1u) {
        let to = touch_offset + t * 4u;
        features[f_base + fi] = sensory[s_base + to];       // dir_x
        fi = fi + 1u;
        features[f_base + fi] = sensory[s_base + to + 1u];  // dir_z
        fi = fi + 1u;
        features[f_base + fi] = sensory[s_base + to + 2u];  // intensity
        fi = fi + 1u;
        features[f_base + fi] = sensory[s_base + to + 3u];  // surface_tag/4
        fi = fi + 1u;
    }
    // fi should now be 192 + 48 + 1 + 3 + 1 + 1 + 1 + 1 + 1 + 16 = 265 = FEATURE_COUNT
}
