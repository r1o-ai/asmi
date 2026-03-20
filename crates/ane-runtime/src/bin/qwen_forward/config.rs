use serde::Deserialize;

/// Qwen 3.5-0.8B model configuration deserialized from HuggingFace `config.json`.
///
/// This model uses a hybrid architecture with two attention types:
/// - **Full attention** (standard GQA with RoPE) every `full_attention_interval` layers
/// - **Linear attention** (DeltaNet-style) for all other layers
#[derive(Debug, Deserialize)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,

    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,

    // Linear attention (DeltaNet) params
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,

    #[serde(default)]
    pub attn_output_gate: bool,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

fn default_partial_rotary_factor() -> f64 {
    0.25
}

fn default_full_attention_interval() -> usize {
    4
}

fn default_linear_num_key_heads() -> usize {
    16
}

fn default_linear_key_head_dim() -> usize {
    128
}

fn default_linear_value_head_dim() -> usize {
    128
}

fn default_linear_num_value_heads() -> usize {
    16
}

fn default_linear_conv_kernel_dim() -> usize {
    4
}

fn default_max_position_embeddings() -> usize {
    131072
}

impl QwenConfig {
    /// Per-head dimension (256 for Qwen 3.5-0.8B).
    pub fn head_size(&self) -> usize {
        self.head_dim
    }

    /// Total KV dimension for full attention: num_kv_heads * head_dim (512).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Total Q dimension for full attention, including the output gate.
    ///
    /// The Q projection outputs 2x head_dim per head (query + sigmoid gate),
    /// so: num_attention_heads * head_dim * 2 = 4096 for 0.8B.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    /// Whether layer `layer_index` uses full (quadratic) attention.
    ///
    /// Full attention layers occur at indices: full_attention_interval-1,
    /// 2*full_attention_interval-1, ... i.e. every 4th layer starting at 3:
    /// layers 3, 7, 11, 15, 19, 23.
    pub fn is_full_attention(&self, layer_index: usize) -> bool {
        (layer_index + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Number of dimensions that receive RoPE embeddings.
    ///
    /// Only `partial_rotary_factor` of head_dim gets RoPE (64 of 256).
    pub fn rope_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// Total QKV dimension for linear attention conv1d.
    ///
    /// conv1d operates over concatenated Q, K, V projections:
    /// (num_key_heads * key_head_dim) + (num_key_heads * key_head_dim) + (num_value_heads * value_head_dim)
    pub fn linear_total_qkv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim  // Q
        + self.linear_num_key_heads * self.linear_key_head_dim  // K
        + self.linear_num_value_heads * self.linear_value_head_dim  // V
    }
}
