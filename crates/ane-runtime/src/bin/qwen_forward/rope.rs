/// Precomputed cosine/sine tables for partial Rotary Position Embeddings (RoPE).
///
/// Qwen 3.5-0.8B uses partial RoPE:
/// - Applied to Q and K only (not V)
/// - Only the first `rope_dim` of `head_dim` dims per head get rotated
///   (partial_rotary_factor = 0.25 => 64 of 256 dims)
/// - theta = 10,000,000.0 (10M, not the standard 10K)
/// - Interleaved real/imaginary pairs: dims (2i, 2i+1) form a rotation pair
pub struct RopeTable {
    /// Cosine values: [max_seq * half_rope_dim], row-major
    cos: Box<[f32]>,
    /// Sine values: [max_seq * half_rope_dim], row-major
    sin: Box<[f32]>,
    /// Number of dimensions that receive rotation (e.g. 64)
    rope_dim: usize,
}

impl RopeTable {
    /// Build cosine/sine lookup tables for the given RoPE parameters.
    ///
    /// - `rope_dim`: number of dims per head that get rotated (must be even)
    /// - `max_seq`: maximum sequence length to precompute for
    /// - `theta`: base frequency (10_000_000.0 for Qwen 3.5)
    pub fn new(rope_dim: usize, max_seq: usize, theta: f64) -> Self {
        debug_assert!(rope_dim.is_multiple_of(2), "rope_dim must be even");
        let half_dim = rope_dim / 2;
        let mut cos = vec![0.0f32; max_seq * half_dim];
        let mut sin = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
                let angle = pos as f64 * freq;
                cos[pos * half_dim + i] = angle.cos() as f32;
                sin[pos * half_dim + i] = angle.sin() as f32;
            }
        }
        Self {
            cos: cos.into_boxed_slice(),
            sin: sin.into_boxed_slice(),
            rope_dim,
        }
    }

    /// Apply RoPE in-place to a tensor slice in ANE channel-first layout.
    ///
    /// The data is laid out as `[num_heads * head_dim, 1, seq_len]` (channel-first),
    /// stored contiguously as `data[channel * seq_len + seq_index]`.
    ///
    /// Only the first `rope_dim` dimensions of each head are rotated;
    /// the remaining `head_dim - rope_dim` dimensions are left untouched.
    ///
    /// - `num_heads`: number of heads in this tensor (Q heads or KV heads)
    /// - `head_dim`: dimension per head (256 for Qwen 3.5)
    /// - `seq_len`: number of tokens in the current tensor
    /// - `position_offset`: starting position index (0 for prefill, cache.position for decode)
    pub fn apply_inplace(
        &self,
        data: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        position_offset: usize,
    ) {
        let half_rope = self.rope_dim / 2;

        for head in 0..num_heads {
            for i in 0..half_rope {
                let dim_even = head * head_dim + 2 * i;
                let dim_odd = head * head_dim + 2 * i + 1;
                for seq in 0..seq_len {
                    let pos = position_offset + seq;
                    let cos_val = self.cos[pos * half_rope + i];
                    let sin_val = self.sin[pos * half_rope + i];

                    let x0 = data[dim_even * seq_len + seq];
                    let x1 = data[dim_odd * seq_len + seq];

                    data[dim_even * seq_len + seq] = x0 * cos_val - x1 * sin_val;
                    data[dim_odd * seq_len + seq] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_position_zero_is_identity() {
        // At position 0 all angles are 0, so cos=1, sin=0.
        // RoPE should leave the data unchanged.
        let rope = RopeTable::new(4, 16, 10_000_000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, head_dim=4, seq_len=1
        let original = data.clone();
        rope.apply_inplace(&mut data, 1, 4, 1, 0);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 should be identity");
        }
    }

    #[test]
    fn rope_rotates_only_first_rope_dim() {
        let rope_dim = 4;
        let head_dim = 8;
        let rope = RopeTable::new(rope_dim, 16, 10_000.0);
        // 1 head, head_dim=8, seq_len=1, position=5 (non-trivial angle)
        let mut data = vec![1.0, 0.0, 0.0, 1.0, 99.0, 88.0, 77.0, 66.0];
        let original = data.clone();
        rope.apply_inplace(&mut data, 1, head_dim, 1, 5);
        // Dims 4..8 (beyond rope_dim) must be untouched.
        assert_eq!(data[4], original[4]);
        assert_eq!(data[5], original[5]);
        assert_eq!(data[6], original[6]);
        assert_eq!(data[7], original[7]);
        // Dims 0..4 should have changed (position 5, non-zero angle).
        let changed = data[0] != original[0] || data[1] != original[1];
        assert!(changed, "rotated dims should differ at non-zero position");
    }
}
