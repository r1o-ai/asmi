use ane::{Shape, TensorData};

/// Per-layer key/value cache backed by IOSurfaces for zero-copy ANE decode.
///
/// Each layer's K and V cache is a `TensorData` with shape `[embedding_dim, 1, max_sequence_length]`.
/// The decode attention kernel reads these directly as placeholder inputs.
pub struct KvCache {
    pub keys: Box<[TensorData]>,
    pub values: Box<[TensorData]>,
    pub max_sequence_length: usize,
    pub embedding_dim: usize,
    pub position: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, embedding_dim: usize, max_sequence_length: usize) -> Self {
        let cache_shape = Shape::spatial(embedding_dim, 1, max_sequence_length);
        Self {
            keys: (0..num_layers).map(|_| TensorData::new(cache_shape)).collect(),
            values: (0..num_layers).map(|_| TensorData::new(cache_shape)).collect(),
            max_sequence_length,
            embedding_dim,
            position: 0,
        }
    }

    /// Write a single token's K and V vectors into the cache at the given position.
    ///
    /// `key_vector` and `value_vector` are `[embedding_dim]` in row-major order.
    pub fn write_kv(
        &self,
        layer_index: usize,
        key_vector: &[f32],
        value_vector: &[f32],
        position: usize,
    ) {
        {
            let mut key_surface = self.keys[layer_index].as_f32_slice_mut();
            for channel in 0..self.embedding_dim {
                key_surface[channel * self.max_sequence_length + position] = key_vector[channel];
            }
        }
        {
            let mut value_surface = self.values[layer_index].as_f32_slice_mut();
            for channel in 0..self.embedding_dim {
                value_surface[channel * self.max_sequence_length + position] = value_vector[channel];
            }
        }
    }

    /// Write K and V vectors for a contiguous range of positions from channel-first
    /// source data with the given stride (source width may differ from `token_count`).
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
        for channel in 0..self.embedding_dim {
            for seq_index in 0..token_count {
                key_surface[channel * self.max_sequence_length + seq_index] =
                    key_data[channel * source_stride + seq_index];
                value_surface[channel * self.max_sequence_length + seq_index] =
                    value_data[channel * source_stride + seq_index];
            }
        }
    }
}
