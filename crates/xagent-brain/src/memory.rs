//! Pattern memory: fixed-capacity storage with temporal sequencing, associative recall,
//! and smart decay. The memory bank is the brain's only persistent knowledge store.

use crate::encoder::EncodedState;

// --- Constants ---
const MAX_ASSOCIATION_STRENGTH: f32 = 5.0;
const MAX_ASSOCIATIONS_PER_PATTERN: usize = 16;
const MAX_REINFORCEMENT: f32 = 10.0;
const SIMILARITY_THRESHOLD: f32 = 0.3;
const HIGH_SIMILARITY_THRESHOLD: f32 = 0.5;
const FORWARD_ASSOCIATION_DEFAULT: f32 = 0.5;
const BACKWARD_ASSOCIATION_DEFAULT: f32 = 0.3;

/// Compute L2 norm of a slice.
#[inline]
fn compute_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// An association link between patterns, tracking validity via generation.
#[derive(Clone, Copy, Debug)]
pub struct AssociationLink {
    pub target_idx: usize,
    pub target_generation: u64,
    pub strength: f32,
}

/// A stored pattern in memory — an encoded state with metadata.
#[derive(Clone, Debug)]
pub struct Pattern {
    /// The encoded representation.
    pub state: EncodedState,
    /// Cached L2 norm of the state vector (avoids recomputation in similarity).
    pub norm: f32,
    /// How strongly this pattern is reinforced (higher = more persistent).
    pub reinforcement: f32,
    /// Tick when this pattern was first stored.
    pub created_at: u64,
    /// Tick when this pattern was last accessed.
    pub last_accessed: u64,
    /// Number of times this pattern was activated.
    pub activation_count: u32,
    /// Association links to other pattern indices (inline, no heap alloc).
    pub associations: [AssociationLink; MAX_ASSOCIATIONS_PER_PATTERN],
    /// Number of valid associations in the array.
    pub association_count: usize,
    /// Temporal predecessor (what came before this pattern).
    pub predecessor: Option<usize>,
    /// Temporal successor (what came after this pattern).
    pub successor: Option<usize>,
    /// Generation counter — incremented each time this slot is written.
    pub generation: u64,
    /// Motor forward output active when this pattern was stored.
    pub motor_forward: f32,
    /// Motor turn output active when this pattern was stored.
    pub motor_turn: f32,
    /// Homeostatic gradient at storage time (positive = good outcome).
    pub outcome_valence: f32,
}

/// A pattern recalled from memory, including its motor context.
#[derive(Clone, Debug)]
pub struct RecalledPattern {
    pub state: EncodedState,
    pub similarity: f32,
    pub motor_forward: f32,
    pub motor_turn: f32,
    pub outcome_valence: f32,
}

/// Fixed-capacity memory bank for temporal patterns.
///
/// Patterns are stored, recalled by similarity, associated by co-occurrence,
/// and decay over time unless reinforced. When full, the weakest pattern
/// is overwritten. Tracks temporal sequences so patterns know what came
/// before and after them.
pub struct PatternMemory {
    /// All stored patterns.
    patterns: Vec<Option<Pattern>>,
    /// Maximum number of patterns.
    capacity: usize,
    /// Dimensionality of pattern representations.
    #[allow(dead_code)]
    dim: usize,
    /// Current tick (for last-accessed tracking).
    current_tick: u64,
    /// Index of the most recently stored pattern.
    last_stored_idx: Option<usize>,
    /// Monotonically increasing generation counter for pattern validity.
    next_generation: u64,
    /// Reusable scratch buffer for scored indices during recall.
    scored_scratch: Vec<(usize, f32)>,
    /// Incremental count of active (non-None) patterns.
    active_count: usize,
    /// Index of the pattern with the lowest reinforcement (cached from decay).
    min_reinforcement_idx: usize,
    /// Scratch buffer for retrieve_associated: visited flags.
    assoc_visited: Vec<bool>,
    /// Scratch buffer for retrieve_associated: BFS frontier.
    assoc_frontier: Vec<(usize, f32)>,
    /// Scratch buffer for gpu_pattern_data: flat pattern data.
    gpu_data_scratch: Vec<f32>,
    /// Scratch buffer for gpu_pattern_data: active mask.
    gpu_active_scratch: Vec<u32>,
}

impl PatternMemory {
    /// Create a new empty pattern memory with the given capacity and representation dimensionality.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            patterns: (0..capacity).map(|_| None).collect(),
            capacity,
            dim,
            current_tick: 0,
            last_stored_idx: None,
            next_generation: 0,
            scored_scratch: Vec::with_capacity(capacity),
            active_count: 0,
            min_reinforcement_idx: 0,
            assoc_visited: vec![false; capacity],
            assoc_frontier: Vec::with_capacity(capacity),
            gpu_data_scratch: vec![0.0f32; capacity * dim],
            gpu_active_scratch: vec![0u32; capacity],
        }
    }

    /// Advance the internal clock by one tick. Must be called once per brain
    /// tick to keep recency-based decay working correctly.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Store a new pattern. Overwrites the weakest existing pattern if full.
    pub fn store(&mut self, state: EncodedState, motor_forward: f32, motor_turn: f32, outcome_valence: f32) {
        self.next_generation += 1;

        let prev_idx = self.last_stored_idx;
        let slot = self.find_slot();

        // Track active_count: overwriting occupied slot doesn't change count
        let was_occupied = self.patterns[slot].is_some();
        if self.patterns[slot].is_some() {
            self.unlink_temporal(slot);
        }
        if !was_occupied {
            self.active_count += 1;
        }

        let norm = compute_norm(state.data());
        let empty_link = AssociationLink { target_idx: 0, target_generation: 0, strength: 0.0 };
        self.patterns[slot] = Some(Pattern {
            state,
            norm,
            reinforcement: 1.0,
            created_at: self.current_tick,
            last_accessed: self.current_tick,
            activation_count: 1,
            associations: [empty_link; MAX_ASSOCIATIONS_PER_PATTERN],
            association_count: 0,
            predecessor: prev_idx,
            successor: None,
            generation: self.next_generation,
            motor_forward,
            motor_turn,
            outcome_valence,
        });

        // Wire temporal sequence: prev → new
        if let Some(prev) = prev_idx {
            if prev != slot {
                // Forward association (pred → succ is stronger for prediction)
                self.associate(prev, slot, FORWARD_ASSOCIATION_DEFAULT);
                // Backward association (weaker)
                self.associate(slot, prev, BACKWARD_ASSOCIATION_DEFAULT);

                // Set temporal links
                if let Some(ref mut p) = self.patterns[prev] {
                    p.successor = Some(slot);
                }
            }
        }

        self.last_stored_idx = Some(slot);
    }

    /// Partial-sort scored_scratch to keep top `budget` entries by descending score.
    fn top_k_scratch(&mut self, budget: usize) {
        if self.scored_scratch.len() > budget && budget > 0 {
            self.scored_scratch.select_nth_unstable_by(budget - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            self.scored_scratch.truncate(budget);
        }
        // Sort the remaining K for deterministic ordering
        self.scored_scratch.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Recall the top-N most similar patterns, within budget.
    /// Returns RecalledPattern values with state, similarity, and motor context.
    pub fn recall(
        &mut self,
        query: &EncodedState,
        budget: usize,
    ) -> Vec<RecalledPattern> {
        let query_norm = compute_norm(query.data());
        self.scored_scratch.clear();
        for (i, p) in self.patterns.iter().enumerate() {
            if let Some(pat) = p.as_ref() {
                let sim = Self::cosine_similarity_prenorm(query.data(), query_norm, pat.state.data());
                self.scored_scratch.push((i, sim));
            }
        }

        self.top_k_scratch(budget);

        // Update access metadata for recalled patterns
        for k in 0..self.scored_scratch.len() {
            let idx = self.scored_scratch[k].0;
            if let Some(ref mut pat) = self.patterns[idx] {
                pat.last_accessed = self.current_tick;
                pat.activation_count += 1;
            }
        }

        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx].as_ref().map(|p| RecalledPattern {
                    state: p.state.clone(),
                    similarity: sim,
                    motor_forward: p.motor_forward,
                    motor_turn: p.motor_turn,
                    outcome_valence: p.outcome_valence,
                })
            })
            .collect()
    }

    /// Follow association chains from a pattern index, returning up to `depth`
    /// associated patterns ordered by association strength.
    /// Skips stale links where the target pattern's generation doesn't match.
    pub fn retrieve_associated(&mut self, from_idx: usize, depth: usize) -> Vec<EncodedState> {
        // Reuse pre-allocated scratch buffers
        for v in self.assoc_visited.iter_mut() {
            *v = false;
        }
        self.assoc_frontier.clear();
        let mut result = Vec::new();

        if let Some(pat) = self.patterns[from_idx].as_ref() {
            self.assoc_visited[from_idx] = true;
            for k in 0..pat.association_count {
                let link = &pat.associations[k];
                if link.target_idx < self.capacity {
                    if let Some(target) = self.patterns[link.target_idx].as_ref() {
                        if target.generation == link.target_generation {
                            self.assoc_frontier.push((link.target_idx, link.strength));
                        }
                    }
                }
            }
        }

        // BFS by strongest association
        self.assoc_frontier.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        while let Some((idx, _strength)) = self.assoc_frontier.pop() {
            if result.len() >= depth {
                break;
            }
            if self.assoc_visited[idx] {
                continue;
            }
            self.assoc_visited[idx] = true;

            if let Some(pat) = self.patterns[idx].as_ref() {
                result.push(pat.state.clone());
                for k in 0..pat.association_count {
                    let link = &pat.associations[k];
                    if link.target_idx < self.capacity && !self.assoc_visited[link.target_idx] {
                        if let Some(target) = self.patterns[link.target_idx].as_ref() {
                            if target.generation == link.target_generation {
                                self.assoc_frontier.push((link.target_idx, link.strength * 0.5));
                            }
                        }
                    }
                }
                self.assoc_frontier
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        result
    }

    /// Retrieve the temporal successor of a pattern (what came next).
    pub fn get_successor(&self, idx: usize) -> Option<&Pattern> {
        self.get(idx)
            .and_then(|p| p.successor)
            .and_then(|s| self.get(s))
    }

    /// Learn from prediction error: reinforce patterns similar to current state
    /// and strengthen co-occurrence associations between recently active patterns.
    pub fn learn(&mut self, current: &EncodedState, error: f32, learning_rate: f32, gradient: f32) {
        let current_norm = compute_norm(current.data());
        // Collect similarity scores first to avoid borrow conflicts
        let sims: Vec<(usize, f32)> = self
            .patterns
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.as_ref()
                    .map(|pat| (i, Self::cosine_similarity_prenorm(current.data(), current_norm, pat.state.data())))
            })
            .filter(|&(_, sim)| sim > SIMILARITY_THRESHOLD)
            .collect();

        // Reinforce patterns proportional to similarity and low prediction error
        for &(i, sim) in &sims {
            if let Some(ref mut pat) = self.patterns[i] {
                pat.reinforcement += sim * learning_rate * (1.0 - error);
                pat.reinforcement = pat.reinforcement.clamp(0.0, MAX_REINFORCEMENT);
                // Retroactive valence update: nudge toward current gradient
                let valence_lr = learning_rate * 0.3;
                pat.outcome_valence += sim * valence_lr * (gradient - pat.outcome_valence);
            }
        }

        // Strengthen co-occurrence associations between similar patterns
        // (patterns that are both similar to the current state co-occur)
        let high_sim: Vec<usize> = sims.iter().filter(|&&(_, s)| s > HIGH_SIMILARITY_THRESHOLD).map(|&(i, _)| i).take(8).collect();
        for i in 0..high_sim.len() {
            for j in (i + 1)..high_sim.len() {
                let boost = learning_rate * 0.1;
                self.associate(high_sim[i], high_sim[j], boost);
                self.associate(high_sim[j], high_sim[i], boost);
            }
        }
    }

    /// Smart decay: patterns that were accessed recently or frequently decay
    /// slower. Removes patterns that fall below threshold.
    pub fn decay(&mut self, base_rate: f32) {
        let tick = self.current_tick;
        let mut min_reinf = f32::MAX;
        let mut min_idx = 0;
        for (i, pattern) in self.patterns.iter_mut().enumerate() {
            if let Some(ref mut p) = pattern {
                let recency = (tick.saturating_sub(p.last_accessed)) as f32;
                let frequency_factor = 1.0 / (1.0 + p.activation_count as f32 * 0.2);
                let recency_factor = (recency / 100.0).min(3.0);
                let effective_rate = base_rate * frequency_factor * (0.2 + recency_factor);
                p.reinforcement -= effective_rate;
                if p.reinforcement <= 0.0 {
                    *pattern = None;
                    self.active_count = self.active_count.saturating_sub(1);
                } else if p.reinforcement < min_reinf {
                    min_reinf = p.reinforcement;
                    min_idx = i;
                }
            }
        }
        self.min_reinforcement_idx = min_idx;
    }

    /// Apply trauma: uniformly reduce all pattern reinforcements by the given
    /// fraction (e.g. 0.2 = 20% loss). Patterns whose reinforcement drops to
    /// zero are removed. Models the cognitive cost of a catastrophic sensory
    /// discontinuity (death/respawn) without being a hardcoded punishment.
    pub fn trauma(&mut self, fraction: f32) {
        let keep = 1.0 - fraction.clamp(0.0, 1.0);
        for pattern in self.patterns.iter_mut() {
            if let Some(ref mut p) = pattern {
                p.reinforcement *= keep;
                if p.reinforcement <= 0.0 {
                    *pattern = None;
                    self.active_count = self.active_count.saturating_sub(1);
                }
            }
        }
    }

    /// Get pattern at index.
    pub fn get(&self, idx: usize) -> Option<&Pattern> {
        self.patterns.get(idx).and_then(|p| p.as_ref())
    }

    /// Count of active (non-None) patterns (O(1) cached counter).
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Memory utilization as a fraction of capacity.
    pub fn utilization(&self) -> f32 {
        self.active_count() as f32 / self.capacity.max(1) as f32
    }

    /// Compute cosine similarity between two encoded states (public for use by other modules).
    pub fn cosine_similarity_states(a: &EncodedState, b: &EncodedState) -> f32 {
        Self::cosine_similarity(a.data(), b.data())
    }

    fn find_slot(&self) -> usize {
        // First: find an empty slot
        if let Some(idx) = self.patterns.iter().position(|p| p.is_none()) {
            return idx;
        }

        // Otherwise: use the cached weakest-pattern index from decay()
        self.min_reinforcement_idx
    }

    /// Remove temporal links pointing to `idx` from predecessor/successor.
    fn unlink_temporal(&mut self, idx: usize) {
        let (pred, succ) = {
            let pat = match self.patterns[idx].as_ref() {
                Some(p) => p,
                None => return,
            };
            (pat.predecessor, pat.successor)
        };
        if let Some(p) = pred {
            if let Some(ref mut pp) = self.patterns[p] {
                if pp.successor == Some(idx) {
                    pp.successor = None;
                }
            }
        }
        if let Some(s) = succ {
            if let Some(ref mut sp) = self.patterns[s] {
                if sp.predecessor == Some(idx) {
                    sp.predecessor = None;
                }
            }
        }
    }

    /// Single-pass cosine similarity with pre-computed norm for `a`.
    fn cosine_similarity_prenorm(a: &[f32], a_norm: f32, b: &[f32]) -> f32 {
        if a_norm < 1e-8 {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let mut b_norm_sq = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            b_norm_sq += y * y;
        }
        let b_norm = b_norm_sq.sqrt();
        if b_norm < 1e-8 {
            return 0.0;
        }
        (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
    }

    pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut a_norm_sq = 0.0f32;
        let mut b_norm_sq = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            a_norm_sq += x * x;
            b_norm_sq += y * y;
        }
        let a_norm = a_norm_sq.sqrt();
        let b_norm = b_norm_sq.sqrt();
        if a_norm < 1e-8 || b_norm < 1e-8 {
            return 0.0;
        }
        (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
    }

    fn associate(&mut self, from: usize, to: usize, strength: f32) {
        let target_gen = self.patterns[to].as_ref().map(|p| p.generation).unwrap_or(0);
        if let Some(ref mut pattern) = self.patterns[from] {
            // Check if link to this target already exists
            if let Some(link) = pattern.associations[..pattern.association_count]
                .iter_mut()
                .find(|l| l.target_idx == to)
            {
                link.strength = (link.strength + strength).min(MAX_ASSOCIATION_STRENGTH);
                link.target_generation = target_gen;
            } else if pattern.association_count < MAX_ASSOCIATIONS_PER_PATTERN {
                pattern.associations[pattern.association_count] = AssociationLink {
                    target_idx: to,
                    target_generation: target_gen,
                    strength,
                };
                pattern.association_count += 1;
            } else {
                // Replace the weakest link if the new one is stronger
                if let Some(weakest) = pattern.associations[..pattern.association_count]
                    .iter_mut()
                    .min_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if strength > weakest.strength {
                        weakest.target_idx = to;
                        weakest.target_generation = target_gen;
                        weakest.strength = strength;
                    }
                }
            }
        }
    }

    /// Memory capacity (max number of patterns).
    pub fn max_capacity(&self) -> usize {
        self.capacity
    }

    /// Representation dimensionality.
    pub fn representation_dim(&self) -> usize {
        self.dim
    }

    /// Flatten all pattern data into contiguous buffers for GPU upload.
    /// Returns (data\[capacity × dim\], active_mask\[capacity\]).
    /// Inactive slots have active_mask\[i\] = 0 and data is zeroed.
    pub fn gpu_pattern_data(&mut self) -> (&[f32], &[u32]) {
        for v in self.gpu_data_scratch.iter_mut() {
            *v = 0.0;
        }
        for v in self.gpu_active_scratch.iter_mut() {
            *v = 0;
        }
        for (i, p) in self.patterns.iter().enumerate() {
            if let Some(pat) = p {
                self.gpu_active_scratch[i] = 1;
                let start = i * self.dim;
                let end = start + self.dim;
                if pat.state.len() == self.dim {
                    self.gpu_data_scratch[start..end].copy_from_slice(pat.state.data());
                }
            }
        }
        (&self.gpu_data_scratch, &self.gpu_active_scratch)
    }

    /// Recall patterns using pre-computed similarity scores (from GPU).
    /// `scores` must be \[capacity\] in length. Values < -1.5 are treated as inactive.
    pub fn recall_with_gpu_similarities(
        &mut self,
        scores: &[f32],
        budget: usize,
    ) -> Vec<RecalledPattern> {
        self.scored_scratch.clear();
        for (i, &sim) in scores.iter().enumerate().take(self.capacity) {
            if sim > -1.5 {
                self.scored_scratch.push((i, sim));
            }
        }

        self.top_k_scratch(budget);

        for k in 0..self.scored_scratch.len() {
            let idx = self.scored_scratch[k].0;
            if let Some(ref mut pat) = self.patterns[idx] {
                pat.last_accessed = self.current_tick;
                pat.activation_count += 1;
            }
        }

        self.scored_scratch
            .iter()
            .filter_map(|&(idx, sim)| {
                self.patterns[idx].as_ref().map(|p| RecalledPattern {
                    state: p.state.clone(),
                    similarity: sim,
                    motor_forward: p.motor_forward,
                    motor_turn: p.motor_turn,
                    outcome_valence: p.outcome_valence,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(vals: &[f32]) -> EncodedState {
        EncodedState::from_slice(vals)
    }

    #[test]
    fn store_and_recall() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0);

        let results = mem.recall(&make_state(&[0.9, 0.1, 0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        // Should recall the pattern most similar to query
        assert!(results[0].state.data()[0] > 0.5, "Should recall the [1,0,0,0] pattern");
    }

    #[test]
    fn decay_removes_weak_patterns() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
        assert_eq!(mem.active_count(), 1);

        // Decay many times — smart decay is slower, so we need more iterations
        for _ in 0..500 {
            mem.advance_tick();
            mem.decay(0.01);
        }
        assert_eq!(mem.active_count(), 0, "Pattern should have decayed away");
    }

    #[test]
    fn frequently_accessed_patterns_decay_slower() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0);

        // Access pattern 0 many times
        for _ in 0..20 {
            mem.recall(&make_state(&[1.0, 0.0, 0.0, 0.0]), 1);
        }

        // Decay
        for _ in 0..80 {
            mem.advance_tick();
            mem.decay(0.01);
        }

        // Pattern 0 (frequently accessed) should survive; pattern 1 may not
        let p0 = mem.get(0);
        assert!(p0.is_some(), "Frequently accessed pattern should survive");
    }

    #[test]
    fn temporal_sequence_tracking() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 0
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 1
        mem.store(make_state(&[0.0, 0.0, 1.0, 0.0]), 0.0, 0.0, 0.0); // idx 2

        // Check successor chain: 0 → 1 → 2
        let p0 = mem.get(0).unwrap();
        assert_eq!(p0.successor, Some(1));

        let p1 = mem.get(1).unwrap();
        assert_eq!(p1.predecessor, Some(0));
        assert_eq!(p1.successor, Some(2));

        let p2 = mem.get(2).unwrap();
        assert_eq!(p2.predecessor, Some(1));
    }

    #[test]
    fn association_chain_retrieval() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 0
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 1
        mem.store(make_state(&[0.0, 0.0, 1.0, 0.0]), 0.0, 0.0, 0.0); // idx 2

        let assoc = mem.retrieve_associated(0, 5);
        assert!(!assoc.is_empty(), "Should find associated patterns");
    }

    #[test]
    fn capacity_limit_overwrites_weakest() {
        let mut mem = PatternMemory::new(3, 4);
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0);
        mem.store(make_state(&[0.0, 0.0, 1.0, 0.0]), 0.0, 0.0, 0.0);
        assert_eq!(mem.active_count(), 3);

        // A 4th store should overwrite the weakest
        mem.store(make_state(&[0.0, 0.0, 0.0, 1.0]), 0.0, 0.0, 0.0);
        assert_eq!(mem.active_count(), 3);
    }

    #[test]
    fn overwritten_associations_are_invalidated() {
        let mut mem = PatternMemory::new(3, 4);

        // Fill memory with 3 patterns
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 0
        mem.store(make_state(&[0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0); // idx 1
        mem.store(make_state(&[0.0, 0.0, 1.0, 0.0]), 0.0, 0.0, 0.0); // idx 2

        // Verify pattern 1 has association to pattern 0 (backward temporal link)
        let gen0_before = mem.get(0).unwrap().generation;
        let p1 = mem.get(1).unwrap();
        let has_link_to_0 = p1.associations.iter().any(|l| l.target_idx == 0);
        assert!(has_link_to_0, "Pattern 1 should have association to pattern 0");

        // Overwrite pattern 0 (all equal reinforcement, so first/weakest is chosen)
        mem.store(make_state(&[0.9, 0.9, 0.9, 0.9]), 0.0, 0.0, 0.0);

        // Generation should have changed
        let gen0_after = mem.get(0).unwrap().generation;
        assert_ne!(
            gen0_before, gen0_after,
            "Generation should change on overwrite"
        );

        // Pattern 1's old association to idx 0 should have stale generation
        let p1_after = mem.get(1).unwrap();
        let stale_link = p1_after
            .associations
            .iter()
            .find(|l| l.target_idx == 0)
            .expect("Link should still exist in data structure");
        assert_ne!(
            stale_link.target_generation, gen0_after,
            "Stale link should have old generation"
        );
    }

    #[test]
    fn co_occurrence_strengthens_associations() {
        let mut mem = PatternMemory::new(10, 4);

        // Store two similar patterns
        mem.store(make_state(&[0.9, 0.1, 0.0, 0.0]), 0.0, 0.0, 0.0);
        mem.store(make_state(&[0.1, 0.9, 0.0, 0.0]), 0.0, 0.0, 0.0);

        // Get initial association strength (from temporal linking)
        let initial_strength = mem.get(0).unwrap()
            .associations.iter()
            .find(|l| l.target_idx == 1)
            .map(|l| l.strength)
            .unwrap_or(0.0);

        // A state similar to both — learn repeatedly
        let current = make_state(&[0.5, 0.5, 0.0, 0.0]);
        for _ in 0..20 {
            mem.learn(&current, 0.1, 0.1, 0.0);
        }

        // Check that the association strengthened
        let final_strength = mem.get(0).unwrap()
            .associations.iter()
            .find(|l| l.target_idx == 1)
            .map(|l| l.strength)
            .unwrap_or(0.0);
        assert!(
            final_strength > initial_strength,
            "Co-occurrence should strengthen association: initial={initial_strength}, final={final_strength}"
        );
    }

    #[test]
    fn trauma_reduces_reinforcement() {
        let dim = 8;
        let mut mem = PatternMemory::new(10, dim);
        let s1 = EncodedState::from_slice(&vec![1.0; dim]);
        let s2 = EncodedState::from_slice(&vec![0.5; dim]);
        mem.store(s1, 0.0, 0.0, 0.0);
        mem.store(s2, 0.0, 0.0, 0.0);
        assert_eq!(mem.active_count(), 2);

        let before: f32 = mem
            .patterns
            .iter()
            .filter_map(|p| p.as_ref().map(|p| p.reinforcement))
            .sum();

        mem.trauma(0.2);

        let after: f32 = mem
            .patterns
            .iter()
            .filter_map(|p| p.as_ref().map(|p| p.reinforcement))
            .sum();

        assert!(
            after < before,
            "Trauma should reduce total reinforcement: before={before}, after={after}"
        );
        // 20% reduction
        let expected = before * 0.8;
        assert!((after - expected).abs() < 0.001);
    }

    #[test]
    fn trauma_removes_weak_patterns() {
        let dim = 4;
        let mut mem = PatternMemory::new(10, dim);
        let s = EncodedState::from_slice(&vec![1.0; dim]);
        mem.store(s, 0.0, 0.0, 0.0);
        assert_eq!(mem.active_count(), 1);

        // Apply 100% trauma — should wipe everything
        mem.trauma(1.0);
        assert_eq!(mem.active_count(), 0);
    }

    #[test]
    fn learn_updates_outcome_valence() {
        let mut mem = PatternMemory::new(10, 4);
        // Store a pattern with neutral valence
        mem.store(make_state(&[1.0, 0.0, 0.0, 0.0]), 0.5, 0.0, 0.0);

        // Learn with a positive gradient near the stored pattern
        let current = make_state(&[0.9, 0.1, 0.0, 0.0]);
        for _ in 0..20 {
            mem.learn(&current, 0.1, 0.1, 0.5);
        }

        let p = mem.get(0).unwrap();
        assert!(
            p.outcome_valence > 0.1,
            "Valence should become positive after positive gradient: got {}",
            p.outcome_valence
        );
    }

    #[test]
    fn pattern_stores_motor_context() {
        let mut mem = PatternMemory::new(10, 4);
        mem.store(
            make_state(&[1.0, 0.0, 0.0, 0.0]),
            0.8,   // motor_forward
            -0.3,  // motor_turn
            0.05,  // outcome_valence
        );
        let p = mem.get(0).unwrap();
        assert!((p.motor_forward - 0.8).abs() < 1e-6);
        assert!((p.motor_turn - (-0.3)).abs() < 1e-6);
        assert!((p.outcome_valence - 0.05).abs() < 1e-6);
    }
}
