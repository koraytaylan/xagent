// Pass 3: Habituation + homeostasis
// Two independent subsystems grouped to reduce dispatch count.

@group(0) @binding(0) var<storage, read> encoded: array<f32>;
@group(0) @binding(1) var<storage, read> sensory: array<f32>;
@group(0) @binding(2) var<storage, read_write> brain_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> habituated: array<f32>;
@group(0) @binding(4) var<storage, read_write> homeo_out: array<f32>;
@group(0) @binding(5) var<uniform> config: array<vec4<f32>, 2>;

const HAB_EMA_ALPHA: f32 = 0.02;
const ATTEN_FLOOR: f32 = 0.1;
const ENERGY_WEIGHT: f32 = 0.6;
const INTEGRITY_WEIGHT: f32 = 0.4;
const FAST_ALPHA: f32 = 0.6;
const MED_ALPHA: f32 = 0.04;
const SLOW_ALPHA: f32 = 0.004;
const DISTRESS_SCALE: f32 = 10.0;
const MAX_DISTRESS: f32 = 10.0;
const GRAD_BLEND_FAST: f32 = 0.5;
const GRAD_BLEND_MED: f32 = 0.35;
const GRAD_BLEND_SLOW: f32 = 0.15;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent = gid.x;
    let b = agent * BRAIN_STRIDE;
    let e_base = agent * DIM;
    let h_base = agent * DIM;
    let ho_base = agent * HOMEO_OUT_STRIDE;
    let s_base = agent * SENSORY_STRIDE;

    let sensitivity = brain_state[b + O_HAB_SENSITIVITY];
    let max_curiosity = brain_state[b + O_HAB_MAX_CURIOSITY];

    // ── Habituation ──
    var atten_sum: f32 = 0.0;
    for (var d: u32 = 0u; d < DIM; d = d + 1u) {
        let enc = encoded[e_base + d];
        let prev = brain_state[b + O_PREV_ENCODED + d];
        let delta = abs(enc - prev);

        let old_ema = brain_state[b + O_HAB_EMA + d];
        let new_ema = (1.0 - HAB_EMA_ALPHA) * old_ema + HAB_EMA_ALPHA * delta;
        brain_state[b + O_HAB_EMA + d] = new_ema;

        let atten = clamp(new_ema * sensitivity, ATTEN_FLOOR, 1.0);
        brain_state[b + O_HAB_ATTEN + d] = atten;
        atten_sum += atten;

        habituated[h_base + d] = enc * atten;
        brain_state[b + O_PREV_ENCODED + d] = enc;
    }

    // Curiosity bonus (not stored separately — consumed by action selector via homeo_out or brain_state)
    // The curiosity bonus is: (1 - mean_atten) * max_curiosity
    // We don't write it to a specific location in brain_state yet;
    // predict_and_act will compute it from O_HAB_ATTEN directly.

    // ── Homeostasis ──
    // Read energy/integrity from sensory input (packed format)
    let vel_offset = 192u + 48u; // after vision_color + vision_depth
    let energy = sensory[s_base + vel_offset + 7u];      // energy_signal
    let integrity = sensory[s_base + vel_offset + 8u];    // integrity_signal

    let prev_energy = brain_state[b + O_HOMEO + 4u];
    let prev_integrity = brain_state[b + O_HOMEO + 5u];

    let energy_delta = energy - prev_energy;
    let integrity_delta = integrity - prev_integrity;
    let raw_grad = energy_delta * ENERGY_WEIGHT + integrity_delta * INTEGRITY_WEIGHT;

    // Update multi-timescale EMAs
    let grad_fast = brain_state[b + O_HOMEO + 0u] * (1.0 - FAST_ALPHA) + raw_grad * FAST_ALPHA;
    let grad_med = brain_state[b + O_HOMEO + 1u] * (1.0 - MED_ALPHA) + raw_grad * MED_ALPHA;
    let grad_slow = brain_state[b + O_HOMEO + 2u] * (1.0 - SLOW_ALPHA) + raw_grad * SLOW_ALPHA;

    brain_state[b + O_HOMEO + 0u] = grad_fast;
    brain_state[b + O_HOMEO + 1u] = grad_med;
    brain_state[b + O_HOMEO + 2u] = grad_slow;

    // Distress and urgency
    let distress_exp = config[CFG_DISTRESS_EXP / 4u][CFG_DISTRESS_EXP % 4u];
    let e_clamped = clamp(energy, 0.01, 1.0);
    let i_clamped = clamp(integrity, 0.01, 1.0);
    let e_distress = min(pow(1.0 - e_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
    let i_distress = min(pow(1.0 - i_clamped, distress_exp) * DISTRESS_SCALE, MAX_DISTRESS);
    let urgency = (e_distress + i_distress) * 0.5;

    brain_state[b + O_HOMEO + 3u] = urgency;
    brain_state[b + O_HOMEO + 4u] = energy;
    brain_state[b + O_HOMEO + 5u] = integrity;

    // Composite gradient (urgency-amplified)
    let base_grad = grad_fast * GRAD_BLEND_FAST + grad_med * GRAD_BLEND_MED + grad_slow * GRAD_BLEND_SLOW;
    let gradient = base_grad * (1.0 + urgency);
    let raw_gradient_amplified = raw_grad * (1.0 + urgency);

    // Output to homeo_out: [gradient, raw_gradient, urgency, grad_fast, grad_med, grad_slow]
    homeo_out[ho_base + 0u] = gradient;
    homeo_out[ho_base + 1u] = raw_gradient_amplified;
    homeo_out[ho_base + 2u] = urgency;
    homeo_out[ho_base + 3u] = grad_fast;
    homeo_out[ho_base + 4u] = grad_med;
    homeo_out[ho_base + 5u] = grad_slow;
}
