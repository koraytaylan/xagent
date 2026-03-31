//! Sensory encoder: compresses raw sensory frames into fixed-size internal representations.

use rand::Rng;
use rand::SeedableRng;
use xagent_shared::SensoryFrame;

/// Number of non-visual features: velocity_mag, facing(x,y,z), angular_vel,
/// energy, integrity, energy_delta, integrity_delta.
const NON_VISUAL_FEATURES: usize = 9;

/// Channels per pixel in the raw visual encoding: R, G, B, depth.
const CHANNELS_PER_PIXEL: usize = 4;

// (L2_REGULARIZATION_FACTOR and WEIGHT_CLAMP_RANGE removed:
//  the encoder is now a fixed random projection — no adaptation.)

/// Compresses raw sensory input into a fixed-size internal representation.
///
/// Visual data is encoded per-pixel with full spatial structure preserved:
/// each pixel contributes its own RGB + depth features in spatial order
/// (row-major). This lets the brain learn directional associations like
/// "green on the left → turn left" — something the old pooled-bin approach
/// could never capture because it averaged away positional information.
///
/// The encoder is a **fixed random projection** — weights are initialized
/// deterministically from `representation_dim` as a seed and never adapted.
/// This guarantees that all brains with the same config produce identical
/// encodings, making action policy weights transferable across generations
/// without needing to inherit encoder weights.
pub struct SensoryEncoder {
    /// Dimensionality of the output representation.
    representation_dim: usize,
    /// Number of input features (visual pixels × channels + non-visual).
    feature_count: usize,
    /// Encoding weights: [representation_dim × feature_count], row-major.
    /// Fixed after initialization — never adapted.
    weights: Vec<f32>,
    /// Per-feature bias terms (fixed at zero).
    biases: Vec<f32>,
    /// Reusable scratch buffer for extracted features.
    feature_scratch: Vec<f32>,
    /// Reusable scratch buffer for encoded output.
    encode_scratch: Vec<f32>,
    /// Whether the weight matrix has been initialized (deferred to first encode).
    initialized: bool,
}

/// Maximum representation dimensionality. Sized for stack allocation.
pub const MAX_REPR_DIM: usize = 128;

/// Internal representation produced by encoding.
/// Uses a fixed-size stack buffer to avoid heap allocation on every brain tick.
#[derive(Clone, Debug)]
pub struct EncodedState {
    buf: [f32; MAX_REPR_DIM],
    len: usize,
}

impl EncodedState {
    /// Create a new EncodedState from a slice. Panics if slice.len() > MAX_REPR_DIM.
    pub fn from_slice(data: &[f32]) -> Self {
        assert!(data.len() <= MAX_REPR_DIM, "representation_dim {} exceeds MAX_REPR_DIM {}", data.len(), MAX_REPR_DIM);
        let mut buf = [0.0f32; MAX_REPR_DIM];
        buf[..data.len()].copy_from_slice(data);
        Self { buf, len: data.len() }
    }

    /// Create a zeroed EncodedState of the given dimension.
    pub fn zeros(dim: usize) -> Self {
        assert!(dim <= MAX_REPR_DIM);
        Self { buf: [0.0f32; MAX_REPR_DIM], len: dim }
    }

    /// Access the active data as a slice.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.buf[..self.len]
    }

    /// Access the active data as a mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.buf[..self.len]
    }

    /// Length of the representation.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl SensoryEncoder {
    /// Create a new encoder. Weight initialization is deferred to the first
    /// `encode()` call so the encoder can size itself to the actual frame
    /// dimensions without requiring them at construction time.
    ///
    /// `representation_dim` controls the output vector size and seeds the
    /// deterministic weight initialization. All brains with the same
    /// `representation_dim` produce identical encoder weights, ensuring
    /// action policy weights are compatible across generations.
    pub fn new(representation_dim: usize, _visual_encoding_size: usize) -> Self {
        Self {
            representation_dim,
            feature_count: 0,
            weights: Vec::new(),
            biases: vec![0.0; representation_dim],
            feature_scratch: Vec::new(),
            encode_scratch: vec![0.0; representation_dim],
            initialized: false,
        }
    }

    /// Initialize weights on first encode, when we know the actual frame size.
    /// Uses `representation_dim` as the RNG seed so all brains with the same
    /// config produce identical encodings — a deterministic random projection.
    fn lazy_init(&mut self, feature_count: usize) {
        self.feature_count = feature_count;
        let weight_count = feature_count * self.representation_dim;

        // Xavier / Glorot-style initialization with DETERMINISTIC seed.
        let limit = (6.0 / (feature_count + self.representation_dim) as f32).sqrt();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.representation_dim as u64);
        self.weights = (0..weight_count)
            .map(|_| rng.random_range(-limit..=limit))
            .collect();
        self.biases = vec![0.0; self.representation_dim];
        self.feature_scratch = vec![0.0; feature_count];
        self.initialized = true;
    }

    /// Encode a sensory frame into the internal representation.
    pub fn encode(&mut self, frame: &SensoryFrame) -> EncodedState {
        self.extract_features_into(frame);

        for i in 0..self.representation_dim {
            let mut sum = self.biases[i];
            let row_base = i * self.feature_count;
            for j in 0..self.feature_count {
                sum += self.feature_scratch[j] * self.weights[row_base + j];
            }
            self.encode_scratch[i] = crate::fast_tanh(sum);
        }

        EncodedState::from_slice(&self.encode_scratch[..self.representation_dim])
    }

    /// Extract features from a sensory frame into a flat buffer (GPU upload helper).
    pub fn extract_features(&mut self, frame: &SensoryFrame) -> &[f32] {
        self.extract_features_into(frame);
        &self.feature_scratch[..self.feature_count]
    }

    /// Encoder weights, row-major \[representation_dim × feature_count\].
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Encoder bias terms \[representation_dim\].
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    /// Snapshot of encoder weights. Since the encoder is a fixed deterministic
    /// projection, this is identical to `weights()` — included for API symmetry.
    pub fn weights_snapshot(&self) -> Vec<f32> {
        self.weights.to_vec()
    }

    /// No-op for the fixed encoder: all brains with the same representation_dim
    /// already produce identical weights deterministically.
    pub fn import_weights(&mut self, _weights: &[f32], _biases: &[f32]) {}

    /// Number of input features.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Output dimensionality.
    pub fn representation_dim(&self) -> usize {
        self.representation_dim
    }

    /// No-op: the encoder is a fixed random projection. Weights are
    /// deterministic and never adapted, ensuring cross-generation
    /// compatibility of action policy weights.
    pub fn adapt(&mut self, _error: f32, _learning_rate: f32) {}

    /// Extract raw per-pixel features from sensory data into the scratch buffer.
    ///
    /// Each pixel contributes RGB + depth in spatial order (row-major),
    /// preserving the directional layout of the agent's visual field.
    /// The brain can learn spatial associations like "green-left → turn left"
    /// because pixel position maps directly to viewing angle.
    fn extract_features_into(&mut self, frame: &SensoryFrame) {
        let total_pixels = (frame.vision.width * frame.vision.height) as usize;
        let needed = total_pixels * CHANNELS_PER_PIXEL + NON_VISUAL_FEATURES;

        // Lazy-initialize on first call when we know the actual frame size
        if !self.initialized {
            self.lazy_init(needed);
        }

        let mut cursor = 0;

        // Per-pixel RGB + depth, preserving full spatial structure
        if total_pixels > 0 && !frame.vision.color.is_empty() {
            for px in 0..total_pixels {
                let base = px * 4;
                if base + 2 < frame.vision.color.len() {
                    self.feature_scratch[cursor] = frame.vision.color[base];     // R
                    self.feature_scratch[cursor + 1] = frame.vision.color[base + 1]; // G
                    self.feature_scratch[cursor + 2] = frame.vision.color[base + 2]; // B
                } else {
                    self.feature_scratch[cursor] = 0.0;
                    self.feature_scratch[cursor + 1] = 0.0;
                    self.feature_scratch[cursor + 2] = 0.0;
                }
                // Depth (normalized 0..1): how far away is the surface at this pixel
                if px < frame.vision.depth.len() {
                    self.feature_scratch[cursor + 3] = frame.vision.depth[px];
                } else {
                    self.feature_scratch[cursor + 3] = 1.0;
                }
                cursor += CHANNELS_PER_PIXEL;
            }
        } else {
            for _ in 0..(total_pixels * CHANNELS_PER_PIXEL) {
                self.feature_scratch[cursor] = 0.0;
                cursor += 1;
            }
        }

        // Proprioceptive features
        self.feature_scratch[cursor] = frame.velocity.length();
        self.feature_scratch[cursor + 1] = frame.facing.x;
        self.feature_scratch[cursor + 2] = frame.facing.y;
        self.feature_scratch[cursor + 3] = frame.facing.z;
        self.feature_scratch[cursor + 4] = frame.angular_velocity;

        // Interoceptive features
        self.feature_scratch[cursor + 5] = frame.energy_signal;
        self.feature_scratch[cursor + 6] = frame.integrity_signal;
        self.feature_scratch[cursor + 7] = frame.energy_delta;
        self.feature_scratch[cursor + 8] = frame.integrity_delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use xagent_shared::{SensoryFrame, VisualField};

    fn make_frame(color_val: f32, energy: f32) -> SensoryFrame {
        let w = 4;
        let h = 3;
        let pixels = (w * h) as usize;
        SensoryFrame {
            vision: VisualField {
                width: w,
                height: h,
                color: vec![color_val; pixels * 4],
                depth: vec![1.0; pixels],
            },
            velocity: Vec3::ZERO,
            facing: Vec3::Z,
            angular_velocity: 0.0,
            energy_signal: energy,
            integrity_signal: 1.0,
            energy_delta: 0.0,
            integrity_delta: 0.0,
            touch_contacts: vec![],
            tick: 0,
        }
    }

    /// Build a frame where the LEFT half is one color and the RIGHT half is another.
    /// This tests that the encoder preserves spatial structure.
    fn make_spatial_frame(left_r: f32, left_g: f32, right_r: f32, right_g: f32) -> SensoryFrame {
        let w = 4u32;
        let h = 3u32;
        let pixels = (w * h) as usize;
        let mut color = vec![0.0f32; pixels * 4];
        for row in 0..h {
            for col in 0..w {
                let px = (row * w + col) as usize;
                let base = px * 4;
                if col < w / 2 {
                    color[base] = left_r;
                    color[base + 1] = left_g;
                } else {
                    color[base] = right_r;
                    color[base + 1] = right_g;
                }
                color[base + 2] = 0.1; // B
                color[base + 3] = 1.0; // A
            }
        }
        SensoryFrame {
            vision: VisualField { width: w, height: h, color, depth: vec![0.5; pixels] },
            velocity: Vec3::ZERO,
            facing: Vec3::Z,
            angular_velocity: 0.0,
            energy_signal: 0.5,
            integrity_signal: 1.0,
            energy_delta: 0.0,
            integrity_delta: 0.0,
            touch_contacts: vec![],
            tick: 0,
        }
    }

    #[test]
    fn similar_inputs_produce_similar_encodings() {
        let mut enc = SensoryEncoder::new(16, 8);
        let a = enc.encode(&make_frame(0.5, 0.8));
        let b = enc.encode(&make_frame(0.51, 0.8));
        let c = enc.encode(&make_frame(0.0, 0.1));

        let sim_ab = cosine_sim(a.data(), b.data());
        let sim_ac = cosine_sim(a.data(), c.data());

        assert!(
            sim_ab > sim_ac,
            "Similar inputs should produce more similar encodings: sim_ab={sim_ab}, sim_ac={sim_ac}"
        );
    }

    #[test]
    fn encoding_dimension_matches_config() {
        let mut enc = SensoryEncoder::new(32, 16);
        let state = enc.encode(&make_frame(0.5, 0.5));
        assert_eq!(state.len(), 32);
    }

    #[test]
    fn encoder_weights_are_fixed() {
        // The encoder is a deterministic random projection — adapt() is a
        // no-op. Weights must NOT change, ensuring cross-generation
        // compatibility of inherited action policy weights.
        let mut enc = SensoryEncoder::new(8, 4);
        let _ = enc.encode(&make_frame(0.5, 0.5));
        let weights_before: Vec<f32> = enc.weights.clone();
        for _ in 0..100 {
            enc.adapt(0.5, 0.1);
        }
        assert_eq!(enc.weights, weights_before, "Encoder weights should be fixed");
    }

    #[test]
    fn same_config_produces_identical_encoder() {
        // Two brains with the same representation_dim must produce the
        // same encoder weights — this is what makes action weight
        // inheritance work across generations.
        let mut enc1 = SensoryEncoder::new(16, 8);
        let mut enc2 = SensoryEncoder::new(16, 8);
        let frame = make_frame(0.5, 0.8);
        let a = enc1.encode(&frame);
        let b = enc2.encode(&frame);
        assert_eq!(a.data(), b.data(), "Same config must produce identical encodings");
    }

    #[test]
    fn different_inputs_produce_different_encodings() {
        let mut enc = SensoryEncoder::new(16, 8);
        let a = enc.encode(&make_frame(0.0, 0.1));
        let b = enc.encode(&make_frame(1.0, 0.9));
        let sim = cosine_sim(a.data(), b.data());
        assert!(
            sim < 0.99,
            "Very different inputs should produce different encodings: sim={sim}"
        );
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut enc = SensoryEncoder::new(16, 8);
        let frame = make_frame(0.5, 0.8);
        let a = enc.encode(&frame);
        let b = enc.encode(&frame);
        assert_eq!(a.data(), b.data(), "Same input should produce same output");
    }

    #[test]
    fn spatial_structure_is_preserved() {
        // Green-left / red-right produces a DIFFERENT encoding from
        // red-left / green-right.  The old pooled encoder would have
        // averaged them to the same global color → identical output.
        let mut enc = SensoryEncoder::new(16, 8);
        let food_left  = enc.encode(&make_spatial_frame(0.1, 0.8, 0.8, 0.1));
        let food_right = enc.encode(&make_spatial_frame(0.8, 0.1, 0.1, 0.8));

        let sim = cosine_sim(food_left.data(), food_right.data());
        assert!(
            sim < 0.95,
            "Spatially mirrored scenes should produce distinct encodings: sim={sim}"
        );
    }

    #[test]
    fn feature_count_includes_depth_and_interoception() {
        let mut enc = SensoryEncoder::new(8, 4);
        let frame = make_frame(0.5, 0.5);
        let feats = enc.extract_features(&frame);
        // 4×3 pixels × 4 channels (RGBD) + 9 interoceptive = 57
        assert_eq!(feats.len(), 4 * 3 * 4 + 9);
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if ma < 1e-8 || mb < 1e-8 {
            return 0.0;
        }
        dot / (ma * mb)
    }
}
