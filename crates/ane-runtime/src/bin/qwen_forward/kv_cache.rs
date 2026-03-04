use ane::{Shape, TensorData};

/// Per-layer key/value cache backed by IOSurfaces for zero-copy ANE decode.
///
/// Each layer's K and V cache is a `TensorData` with shape `[kv_dim, 1, max_sequence_length]`.
/// The decode attention kernel reads these directly as placeholder inputs.
///
/// **GQA-aware**: Qwen 3.5 uses Grouped Query Attention with 2 KV heads (vs 8 Q heads),
/// so `kv_dim = num_kv_heads * head_dim = 2 * 256 = 512` (not the full Q dim of 2048).
///
/// **Only full attention layers** (6 of 24) need KV cache. Linear attention layers
/// use DeltaNet recurrent state instead and are handled separately.
pub struct KvCache {
    pub keys: Box<[TensorData]>,
    pub values: Box<[TensorData]>,
    pub max_sequence_length: usize,
    /// Total KV dimension: num_kv_heads * head_dim (512 for Qwen 3.5-0.8B)
    pub kv_dim: usize,
    /// Current write position (number of tokens cached so far)
    pub position: usize,
}

impl KvCache {
    /// Create a new KV cache for the full attention layers.
    ///
    /// - `num_full_attn_layers`: number of layers using full attention (6 for Qwen 3.5)
    /// - `kv_dim`: num_kv_heads * head_dim (512)
    /// - `max_sequence_length`: maximum context length to cache
    pub fn new(num_full_attn_layers: usize, kv_dim: usize, max_sequence_length: usize) -> Self {
        let cache_shape = Shape::spatial(kv_dim, 1, max_sequence_length);
        Self {
            keys: (0..num_full_attn_layers)
                .map(|_| TensorData::new(cache_shape))
                .collect(),
            values: (0..num_full_attn_layers)
                .map(|_| TensorData::new(cache_shape))
                .collect(),
            max_sequence_length,
            kv_dim,
            position: 0,
        }
    }

    /// Write a single token's K and V vectors into the cache at the given position.
    ///
    /// `key_vector` and `value_vector` are `[kv_dim]` in row-major order.
    /// `layer_index` is the index into the full-attention layer array (0..5), NOT the
    /// global layer index.
    pub fn write_kv(
        &self,
        layer_index: usize,
        key_vector: &[f32],
        value_vector: &[f32],
        position: usize,
    ) {
        {
            let mut key_surface = self.keys[layer_index].as_f32_slice_mut();
            for channel in 0..self.kv_dim {
                key_surface[channel * self.max_sequence_length + position] = key_vector[channel];
            }
        }
        {
            let mut value_surface = self.values[layer_index].as_f32_slice_mut();
            for channel in 0..self.kv_dim {
                value_surface[channel * self.max_sequence_length + position] =
                    value_vector[channel];
            }
        }
    }

    /// Write K and V vectors for a contiguous range of positions from channel-first
    /// source data with the given stride (source width may differ from `token_count`).
    ///
    /// `layer_index` is the index into the full-attention layer array (0..5).
    pub fn write_kv_sequence(
        &self,
        layer_index: usize,
        key_data: &[f32],
        value_data: &[f32],
        token_count: usize,
        source_stride: usize,
    ) {
        let mut key_surface = self.keys[layer_index].as_f32_slice_mut();
        let mut value_surface = self.values[layer_index].as_f32_slice_mut();
        for channel in 0..self.kv_dim {
            for seq_index in 0..token_count {
                key_surface[channel * self.max_sequence_length + seq_index] =
                    key_data[channel * source_stride + seq_index];
                value_surface[channel * self.max_sequence_length + seq_index] =
                    value_data[channel * source_stride + seq_index];
            }
        }
    }
}
