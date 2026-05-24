/// Feed-forward network weights shared by both attention types.
pub struct FfnWeights {
    /// SwiGLU gate projection: [intermediate_size, hidden_size]
    pub gate_proj_weight: Box<[f32]>,
    /// SwiGLU up projection: [intermediate_size, hidden_size]
    pub up_proj_weight: Box<[f32]>,
    /// SwiGLU down projection: [hidden_size, intermediate_size]
    pub down_proj_weight: Box<[f32]>,
}

pub struct MoeWeights {
    /// Router gate weights: [num_experts, hidden_size]
    pub gate_weight: Box<[f32]>,
    /// Shared expert FFN weights
    pub shared_expert: FfnWeights,
    /// Shared expert gate weights (optional): [1, hidden_size]
    pub shared_expert_gate_weight: Option<Box<[f32]>>,
    /// Sparse expert FFN weights
    pub experts: Vec<FfnWeights>,
}

pub enum FfnVariant {
    Dense(FfnWeights),
    Moe(MoeWeights),
}

/// Weights for a full (quadratic) attention layer with GQA and RoPE.
///
/// Occurs at layers 3, 7, 11, 15, 19, 23 (every 4th starting at 3).
pub struct FullAttentionWeights {
    /// RMSNorm before attention: [hidden_size]
    pub input_layernorm_weight: Box<[f32]>,

    // --- Full attention projections ---
    /// Q projection with output gate: [num_heads * head_dim * 2, hidden_size]
    pub q_proj_weight: Box<[f32]>,
    /// Q projection bias: [num_heads * head_dim * 2]
    pub q_proj_bias: Box<[f32]>,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub k_proj_weight: Box<[f32]>,
    /// K projection bias: [num_kv_heads * head_dim]
    pub k_proj_bias: Box<[f32]>,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub v_proj_weight: Box<[f32]>,
    /// V projection bias: [num_kv_heads * head_dim]
    pub v_proj_bias: Box<[f32]>,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub o_proj_weight: Box<[f32]>,

    /// Per-head Q RMSNorm: [head_dim]
    pub q_norm_weight: Box<[f32]>,
    /// Per-head K RMSNorm: [head_dim]
    pub k_norm_weight: Box<[f32]>,

    /// RMSNorm before FFN: [hidden_size]
    pub post_attention_layernorm_weight: Box<[f32]>,

    /// FFN weights (SwiGLU or MoE)
    pub ffn: FfnVariant,
}

/// Weights for a linear (DeltaNet) attention layer.
///
/// Occurs at all layers except the full attention ones.
pub struct LinearAttentionWeights {
    /// RMSNorm before attention: [hidden_size]
    pub input_layernorm_weight: Box<[f32]>,

    // --- Linear attention projections ---
    /// Q projection: [linear_num_key_heads * linear_key_head_dim, hidden_size]
    pub q_proj_weight: Box<[f32]>,
    /// K projection: [linear_num_key_heads * linear_key_head_dim, hidden_size]
    pub k_proj_weight: Box<[f32]>,
    /// V projection: [linear_num_value_heads * linear_value_head_dim, hidden_size]
    pub v_proj_weight: Box<[f32]>,

    /// Decay projection (alpha): [linear_num_key_heads, hidden_size]
    pub a_proj_weight: Box<[f32]>,
    /// Update gate projection (beta): [linear_num_key_heads, hidden_size]
    pub b_proj_weight: Box<[f32]>,
    /// Output gate projection: [linear_num_value_heads * linear_value_head_dim, hidden_size]
    pub z_proj_weight: Box<[f32]>,

    /// Log-space decay parameter (A_log): [linear_num_key_heads]
    pub a_log: Box<[f32]>,
    /// Time-delta bias: [linear_num_key_heads]
    pub dt_bias: Box<[f32]>,

    /// Output projection: [hidden_size, linear_num_value_heads * linear_value_head_dim]
    pub o_proj_weight: Box<[f32]>,

    /// Conv1d weight over concatenated QKV: [total_qkv_dim, 1, conv_kernel_dim]
    pub conv1d_weight: Box<[f32]>,
    /// Conv1d bias: [total_qkv_dim]
    pub conv1d_bias: Box<[f32]>,

    /// Output norm (RMSNorm): [linear_num_value_heads * linear_value_head_dim]
    pub norm_weight: Box<[f32]>,
    /// Gate norm (RMSNorm, optional): [linear_num_value_heads * linear_value_head_dim]
    pub gate_norm_weight: Option<Box<[f32]>>,

    /// RMSNorm before FFN: [hidden_size]
    pub post_attention_layernorm_weight: Box<[f32]>,

    /// FFN weights (SwiGLU or MoE)
    pub ffn: FfnVariant,
}

/// Per-layer weights, dispatched by attention type.
pub enum LayerWeights {
    FullAttention(FullAttentionWeights),
    LinearAttention(LinearAttentionWeights),
}
