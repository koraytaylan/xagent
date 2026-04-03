//! Sensory encoder: compresses raw sensory frames into fixed-size internal representations.

use rand::Rng;
use rand::SeedableRng;
use xagent_shared::SensoryFrame;

/// Number of non-visual features: velocity_mag, facing(x,y,z), angular_vel,
/// energy, integrity, energy_delta, integrity_delta (9 proprioceptive/interoceptive)
/// + 4 touch contacts × 4 features each (direction.x, direction.z, intensity,
/// normalized surface_tag) = 25 total.
const NON_VISUAL_FEATURES: usize = 25;

/// Maximum number of touch contacts encoded into the feature vector.
/// The strongest contacts by intensity are selected; remaining slots are zero-padded.
const MAX_TOUCH_CONTACTS: usize = 4;

/// Features per touch contact: direction.x, direction.z, intensity, surface_tag/4.0.
const TOUCH_FEATURES_PER_CONTACT: usize = 4;

/// Channels per pixel in the raw visual encoding: R, G, B, depth.
const CHANNELS_PER_PIXEL: usize = 4;

// (L2_REGULARIZATION_FACTOR and WEIGHT_CLAMP_RANGE removed:
//  the encoder uses Hebbian credit-driven adaptation via adapt_from_credit.)

/// Compresses raw sensory input into a fixed-size internal representation.
///
/// Visual data is encoded per-pixel with full spatial structure preserved:
/// each pixel contributes its own RGB + depth features in spatial order
/// (row-major). This lets the brain learn directional associations like
/// "green on the left → turn left" — something the old pooled-bin approach
/// could never capture because it averaged away positional information.
///
/// Weights are initialized deterministically from `representation_dim` as a
/// seed, ensuring all brains with the same config start from identical
/// encodings. During the agent's lifetime, `adapt_from_credit()` refines
/// the weights using the action selector's credit signal (Hebbian rule),
/// amplifying the raw features that contribute to behaviourally relevant
/// encoded dimensions. Inherited encoder weights are transferred via
/// `import_weights()` / `weights_snapshot()`.
pub struct SensoryEncoder {
    /// Dimensionality of the output representation.
    representation_dim: usize,
    /// Number of input features (visual pixels × channels + non-visual).
    feature_count: usize,
    /// Encoding weights: [representation_dim × feature_count], row-major.
    /// Initialized deterministically, then refined by `adapt_from_credit()`.
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

    /// Snapshot of encoder weights for cross-generation inheritance.
    pub fn weights_snapshot(&self) -> Vec<f32> {
        self.weights.clone()
    }

    /// Import encoder weights and biases from a previous generation.
    /// Silently skipped if dimensions don't match.
    pub fn import_weights(&mut self, weights: &[f32], biases: &[f32]) {
        if weights.len() == self.weights.len() {
            self.weights.copy_from_slice(weights);
        }
        if biases.len() == self.biases.len() {
            self.biases.copy_from_slice(biases);
        }
    }

    /// Number of input features.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Output dimensionality.
    pub fn representation_dim(&self) -> usize {
        self.representation_dim
    }

    /// Legacy no-op: kept for backward compatibility.
    /// The brain now calls `adapt_from_credit()` instead.
    pub fn adapt(&mut self, _error: f32, _learning_rate: f32) {}

    /// Adapt encoder weights based on the action selector's credit signal.
    ///
    /// When the action selector learns "dimension i in encoded state correlates
    /// with positive outcomes", the encoder amplifies the raw input features
    /// that contribute to dimension i.
    ///
    /// Simple Hebbian rule: `delta_w[i,j] += lr * credit_signal[i] * input_feature[j]`
    pub fn adapt_from_credit(&mut self, credit_signal: &[f32], learning_rate: f32) {
        if !self.initialized || credit_signal.len() != self.representation_dim {
            return;
        }
        let fc = self.feature_count;
        for i in 0..self.representation_dim {
            if credit_signal[i].abs() < 1e-6 {
                continue;
            }
            let row_base = i * fc;
            let scale = learning_rate * credit_signal[i] * 0.001; // small encoder LR
            for j in 0..fc {
                self.weights[row_base + j] += scale * self.feature_scratch[j];
                self.weights[row_base + j] = self.weights[row_base + j].clamp(-2.0, 2.0);
            }
        }
    }

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

        // Touch contacts: top MAX_TOUCH_CONTACTS by intensity, 4 features each.
        // The brain receives direction, proximity, and an opaque surface tag —
        // it must learn what each tag value correlates with through experience.
        let touch_start = cursor + 9;
        let mut contact_count = frame.touch_contacts.len().min(MAX_TOUCH_CONTACTS);

        if contact_count > 0 {
            // Find the top contacts by intensity without allocating.
            // We use a simple selection: for each slot, find the strongest
            // unused contact. With MAX_TOUCH_CONTACTS=4, this is 4×N comparisons
            // which is fine for typical contact counts (<10).
            let mut used = [false; 64]; // more than enough for any realistic contact count
            for slot in 0..contact_count {
                let mut best_idx = usize::MAX;
                let mut best_intensity = -1.0_f32;
                for (ci, contact) in frame.touch_contacts.iter().enumerate() {
                    if ci < used.len() && !used[ci] && contact.intensity > best_intensity {
                        best_intensity = contact.intensity;
                        best_idx = ci;
                    }
                }
                if best_idx == usize::MAX {
                    // Fewer usable contacts than expected
                    contact_count = slot;
                    break;
                }
                used[best_idx] = true;
                let c = &frame.touch_contacts[best_idx];
                let base = touch_start + slot * TOUCH_FEATURES_PER_CONTACT;
                self.feature_scratch[base] = c.direction.x;
                self.feature_scratch[base + 1] = c.direction.z;
                self.feature_scratch[base + 2] = c.intensity;
                self.feature_scratch[base + 3] = c.surface_tag as f32 / 4.0;
            }
        }

        // Zero-pad remaining touch slots
        for slot in contact_count..MAX_TOUCH_CONTACTS {
            let base = touch_start + slot * TOUCH_FEATURES_PER_CONTACT;
            self.feature_scratch[base] = 0.0;
            self.feature_scratch[base + 1] = 0.0;
            self.feature_scratch[base + 2] = 0.0;
            self.feature_scratch[base + 3] = 0.0;
        }
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
        // 4×3 pixels × 4 channels (RGBD) + 25 non-visual (9 proprioceptive/interoceptive + 16 touch) = 73
        assert_eq!(feats.len(), 4 * 3 * 4 + NON_VISUAL_FEATURES);
    }

    #[test]
    fn adapt_from_credit_modifies_weights() {
        let mut enc = SensoryEncoder::new(8, 4);
        // Force initialization by encoding a frame.
        let _ = enc.encode(&make_frame(0.5, 0.5));
        let weights_before: Vec<f32> = enc.weights.clone();

        // Build a credit signal with non-zero values.
        let credit: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        enc.adapt_from_credit(&credit, 0.5);

        assert_ne!(
            enc.weights, weights_before,
            "adapt_from_credit with non-zero signal should modify weights"
        );
    }

    #[test]
    fn adapt_from_credit_skips_zero_signal() {
        let mut enc = SensoryEncoder::new(8, 4);
        let _ = enc.encode(&make_frame(0.5, 0.5));
        let weights_before: Vec<f32> = enc.weights.clone();

        let credit = vec![0.0; 8];
        enc.adapt_from_credit(&credit, 0.5);

        assert_eq!(
            enc.weights, weights_before,
            "adapt_from_credit with zero signal should not modify weights"
        );
    }

    #[test]
    fn import_weights_restores_snapshot() {
        let mut enc = SensoryEncoder::new(8, 4);
        let _ = enc.encode(&make_frame(0.5, 0.5));

        // Mutate weights via credit.
        let credit: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        enc.adapt_from_credit(&credit, 0.5);
        let snapshot = enc.weights_snapshot();
        let bias_snapshot: Vec<f32> = enc.biases().to_vec();

        // Create a fresh encoder and import the snapshot.
        let mut enc2 = SensoryEncoder::new(8, 4);
        let _ = enc2.encode(&make_frame(0.5, 0.5));
        enc2.import_weights(&snapshot, &bias_snapshot);

        assert_eq!(
            enc2.weights, enc.weights,
            "import_weights should restore the snapshot"
        );
    }

    #[test]
    fn touch_contacts_appear_in_features() {
        use xagent_shared::TouchContact;

        let mut enc = SensoryEncoder::new(8, 4);
        let mut frame = make_frame(0.5, 0.5);

        // Add touch contacts
        frame.touch_contacts = vec![
            TouchContact {
                direction: Vec3::new(1.0, 0.0, 0.0),
                intensity: 0.8,
                surface_tag: 1,
            },
            TouchContact {
                direction: Vec3::new(0.0, 0.0, -1.0),
                intensity: 0.3,
                surface_tag: 2,
            },
        ];

        let feats = enc.extract_features(&frame);

        // Non-visual features start after visual pixels
        // 4×3 pixels × 4 channels = 48, then 9 proprioceptive/interoceptive, then touch
        let touch_start = 4 * 3 * 4 + 9;

        // First touch slot should be the strongest contact (intensity 0.8)
        assert!((feats[touch_start] - 1.0).abs() < 1e-6, "direction.x of strongest contact");
        assert!((feats[touch_start + 1] - 0.0).abs() < 1e-6, "direction.z of strongest contact");
        assert!((feats[touch_start + 2] - 0.8).abs() < 1e-6, "intensity of strongest contact");
        assert!((feats[touch_start + 3] - 0.25).abs() < 1e-6, "surface_tag/4.0 of strongest contact (tag 1)");

        // Second touch slot should be the weaker contact (intensity 0.3)
        let slot2 = touch_start + 4;
        assert!((feats[slot2] - 0.0).abs() < 1e-6, "direction.x of second contact");
        assert!((feats[slot2 + 1] - (-1.0)).abs() < 1e-6, "direction.z of second contact");
        assert!((feats[slot2 + 2] - 0.3).abs() < 1e-6, "intensity of second contact");
        assert!((feats[slot2 + 3] - 0.5).abs() < 1e-6, "surface_tag/4.0 of second contact (tag 2)");

        // Third and fourth slots should be zero-padded
        let slot3 = touch_start + 8;
        for i in 0..4 {
            assert!((feats[slot3 + i]).abs() < 1e-6, "slot 3 feature {} should be zero-padded", i);
        }
        let slot4 = touch_start + 12;
        for i in 0..4 {
            assert!((feats[slot4 + i]).abs() < 1e-6, "slot 4 feature {} should be zero-padded", i);
        }
    }

    #[test]
    fn touch_contacts_sorted_by_intensity() {
        use xagent_shared::TouchContact;

        let mut enc = SensoryEncoder::new(8, 4);
        let mut frame = make_frame(0.5, 0.5);

        // Add contacts in non-sorted order (weakest first)
        frame.touch_contacts = vec![
            TouchContact {
                direction: Vec3::new(0.0, 0.0, 1.0),
                intensity: 0.1,
                surface_tag: 3,
            },
            TouchContact {
                direction: Vec3::new(1.0, 0.0, 0.0),
                intensity: 0.9,
                surface_tag: 1,
            },
            TouchContact {
                direction: Vec3::new(-1.0, 0.0, 0.0),
                intensity: 0.5,
                surface_tag: 4,
            },
        ];

        let feats = enc.extract_features(&frame);
        let touch_start = 4 * 3 * 4 + 9;

        // Slot 0: strongest (0.9, tag 1)
        assert!((feats[touch_start + 2] - 0.9).abs() < 1e-6, "slot 0 should have intensity 0.9");
        assert!((feats[touch_start + 3] - 0.25).abs() < 1e-6, "slot 0 should have tag 1 (0.25)");

        // Slot 1: second strongest (0.5, tag 4)
        let slot1 = touch_start + 4;
        assert!((feats[slot1 + 2] - 0.5).abs() < 1e-6, "slot 1 should have intensity 0.5");
        assert!((feats[slot1 + 3] - 1.0).abs() < 1e-6, "slot 1 should have tag 4 (1.0)");

        // Slot 2: weakest (0.1, tag 3)
        let slot2 = touch_start + 8;
        assert!((feats[slot2 + 2] - 0.1).abs() < 1e-6, "slot 2 should have intensity 0.1");
        assert!((feats[slot2 + 3] - 0.75).abs() < 1e-6, "slot 2 should have tag 3 (0.75)");
    }

    #[test]
    fn no_touch_contacts_produces_zero_features() {
        let mut enc = SensoryEncoder::new(8, 4);
        let frame = make_frame(0.5, 0.5); // no touch contacts

        let feats = enc.extract_features(&frame);
        let touch_start = 4 * 3 * 4 + 9;

        // All 16 touch features should be zero
        for i in 0..16 {
            assert!(
                feats[touch_start + i].abs() < 1e-6,
                "touch feature {} should be zero with no contacts, got {}",
                i,
                feats[touch_start + i]
            );
        }
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
