use ane::{Executable, Graph, NSQualityOfService, Shape};

use crate::config::QwenConfig;
use crate::weights::{
    FfnWeights, FullAttentionWeights, LayerWeights, LinearAttentionWeights,
};

// ---------------------------------------------------------------------------
// Compiled layer types
// ---------------------------------------------------------------------------

pub enum FfnExecutable {
    Dense(Executable),
    Moe {
        shared_expert: Executable,
    },
}

/// Full attention layer executables (layers 3, 7, 11, 15, 19, 23).
///
/// Split into two ANE kernels with a CPU step between them:
///   1. `projection` — RMSNorm + Q/K/V/gate linear projections
///   2. (CPU) — per-head Q/K RMSNorm, partial RoPE, GQA KV expansion
///   3. `attention` — multi-head dot-product attention + sigmoid gate + o_proj
///   4. `feed_forward` — post-attention RMSNorm + SwiGLU FFN (Dense or Shared Expert)
pub struct FullAttnPrefillLayer {
    pub projection: Executable,
    pub attention: Executable,
    pub feed_forward: FfnExecutable,
}

/// Full attention decode layer. Same split as prefill but with cache inputs.
pub struct FullAttnDecodeLayer {
    pub projection: Executable,
    pub attention: Executable,
    pub feed_forward: FfnExecutable,
}

/// Linear attention (DeltaNet) layers only run the FFN on ANE.
///
/// The entire DeltaNet attention is handled on CPU because it involves
/// recurrent state updates, causal convolution, and element-wise gating
/// that don't map well to the ANE's fixed-function pipeline.
pub struct LinearAttnPrefillLayer {
    pub feed_forward: FfnExecutable,
}

/// Linear attention decode layer — FFN only, DeltaNet on CPU.
pub struct LinearAttnDecodeLayer {
    pub feed_forward: FfnExecutable,
}

/// Enum dispatch for a single prefill layer.
pub enum PrefillLayer {
    FullAttention(FullAttnPrefillLayer),
    LinearAttention(LinearAttnPrefillLayer),
}

/// Enum dispatch for a single decode layer.
pub enum DecodeLayer {
    FullAttention(FullAttnDecodeLayer),
    LinearAttention(LinearAttnDecodeLayer),
}

/// All compiled ANE executables for prefill and decode phases.
pub struct CompiledExecutables {
    pub prefill: Box<[PrefillLayer]>,
    pub decode: Box<[DecodeLayer]>,

    /// Generic MoE expert executable for the prefill phase.
    pub moe_expert_prefill: Option<Executable>,
    /// Generic MoE expert executable for the decode phase.
    pub moe_expert_decode: Option<Executable>,
}

/// Minimum spatial width for ANE execution (hardware constraint).
pub const DECODE_SPATIAL_WIDTH: usize = 64;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_shape() -> Shape {
    Shape { batch: 1, channels: 1, height: 1, width: 1 }
}

/// Build RMSNorm as ANE graph operations (6 ops).
///
/// Formula: `x * rsqrt(mean(x^2) + eps) * weight`
///
/// Unlike LayerNorm, there is no mean subtraction and no bias term,
/// making this significantly cheaper on the ANE.
fn rms_norm(
    graph: &mut Graph,
    input: ane::Tensor,
    weight: &[f32],
    hidden_size: usize,
    eps: f64,
) -> ane::Tensor {
    let bcast = scalar_shape();
    let inv_dim = graph.constant_with_scalar(1.0 / hidden_size as f32, bcast);
    let eps_const = graph.constant_with_scalar(eps as f32, bcast);
    let neg_half = graph.constant_with_scalar(-0.5, bcast);
    let w = graph.constant(
        weight,
        Shape { batch: 1, channels: hidden_size, height: 1, width: 1 },
    );

    // x^2
    let squared = graph.multiplication(input, input);
    // sum(x^2) over channel axis
    let sum_sq = graph.reduce_sum(squared, 1);
    // mean(x^2) = sum / dim
    let mean_sq = graph.multiplication(sum_sq, inv_dim);
    // mean(x^2) + eps
    let added = graph.addition(mean_sq, eps_const);
    // rsqrt = (mean + eps)^(-0.5)
    let rstd = graph.power(added, neg_half);
    // x * rsqrt
    let normalized = graph.multiplication(input, rstd);
    // x * rsqrt * weight
    graph.multiplication(normalized, w)
}

/// SiLU activation: `x * sigmoid(x)` (2 ops).
fn silu(graph: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let sig = graph.sigmoid(input);
    graph.multiplication(input, sig)
}

/// Causal attention mask for prefill: 0 where attending, -65504 where masked.
///
/// Returns a flat `[seq * seq]` array where `mask[row * seq + col] = 0.0`
/// if `col <= row` (causal), else `-65504.0` (fp16 -inf).
pub fn causal_mask(sequence_length: usize) -> Box<[f32]> {
    (0..sequence_length * sequence_length)
        .map(|flat_index| {
            let column = flat_index % sequence_length;
            let row = flat_index / sequence_length;
            if column <= row { 0.0 } else { -65504.0 }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Full attention: projection kernel
// ---------------------------------------------------------------------------

/// Build the projection kernel for a full attention layer.
///
/// **Input:**  `[hidden_size, 1, seq]`
///
/// **Output:** `[q_dim + kv_dim + kv_dim, 1, seq]` — concatenated Q+gate, K, V
///   - Q+gate: `[num_heads * head_dim * 2, 1, seq]` = `[4096, 1, seq]`
///   - K:      `[num_kv_heads * head_dim, 1, seq]`  = `[512, 1, seq]`
///   - V:      `[num_kv_heads * head_dim, 1, seq]`  = `[512, 1, seq]`
///   - Total:  `[5120, 1, seq]`
///
/// The caller (CPU) is responsible for:
///   - Splitting Q+gate into query `[2048]` and gate `[2048]`
///   - Per-head Q and K RMSNorm
///   - Partial RoPE on Q and K (first `rope_dim` of `head_dim`)
///   - GQA expansion of K and V (2 heads -> 8 by repeating 4x)
fn full_attn_projection_body(
    graph: &mut Graph,
    input: ane::Tensor,
    weights: &FullAttentionWeights,
    config: &QwenConfig,
) -> ane::Tensor {
    let hidden = config.hidden_size;
    let q_dim = config.q_dim(); // num_heads * head_dim * 2 = 4096
    let kv_dim = config.kv_dim(); // num_kv_heads * head_dim = 512

    // Pre-attention RMSNorm
    let normalized = rms_norm(
        graph, input,
        &weights.input_layernorm_weight, hidden, config.rms_norm_eps,
    );

    // Q projection (includes output gate, so output dim is 2x)
    let q_weight = graph.constant(&weights.q_proj_weight, Shape::spatial(q_dim, 1, 1));
    let q_proj = graph.convolution_2d_1x1(normalized, q_weight, None);
    let q_bias = graph.constant(
        &weights.q_proj_bias,
        Shape { batch: 1, channels: q_dim, height: 1, width: 1 },
    );
    let q_proj = graph.addition(q_proj, q_bias);

    // K projection
    let k_weight = graph.constant(&weights.k_proj_weight, Shape::spatial(kv_dim, 1, 1));
    let k_proj = graph.convolution_2d_1x1(normalized, k_weight, None);
    let k_bias = graph.constant(
        &weights.k_proj_bias,
        Shape { batch: 1, channels: kv_dim, height: 1, width: 1 },
    );
    let k_proj = graph.addition(k_proj, k_bias);

    // V projection
    let v_weight = graph.constant(&weights.v_proj_weight, Shape::spatial(kv_dim, 1, 1));
    let v_proj = graph.convolution_2d_1x1(normalized, v_weight, None);
    let v_bias = graph.constant(
        &weights.v_proj_bias,
        Shape { batch: 1, channels: kv_dim, height: 1, width: 1 },
    );
    let v_proj = graph.addition(v_proj, v_bias);

    // Concatenate: [q_dim + kv_dim + kv_dim, 1, seq] = [5120, 1, seq]
    graph.concat(&[q_proj, k_proj, v_proj], 1)
}

/// Prefill projection: `[hidden_size, 1, seq]` -> `[q_dim + 2*kv_dim, 1, seq]`.
pub fn build_full_attn_projection_prefill(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
    seq_len: usize,
) -> Result<Executable, ane::Error> {
    let mut graph = Graph::new();
    let input = graph.placeholder(Shape::spatial(config.hidden_size, 1, seq_len));
    let _output = full_attn_projection_body(&mut graph, input, weights, config);
    graph.compile(NSQualityOfService::Default)
}

/// Decode projection: `[hidden_size, 1, 64]` -> `[q_dim + 2*kv_dim, 1, 64]`.
pub fn build_full_attn_projection_decode(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
) -> Result<Executable, ane::Error> {
    let mut graph = Graph::new();
    let input = graph.placeholder(Shape::spatial(config.hidden_size, 1, DECODE_SPATIAL_WIDTH));
    let _output = full_attn_projection_body(&mut graph, input, weights, config);
    graph.compile(NSQualityOfService::Default)
}

// ---------------------------------------------------------------------------
// Full attention: attention body kernel
// ---------------------------------------------------------------------------

/// Build the attention body kernel for a full attention layer.
///
/// This kernel expects pre-processed inputs (CPU has already done per-head
/// Q/K RMSNorm, partial RoPE, and GQA KV expansion):
///
/// **Inputs:**
///   - `query`: `[num_heads * head_dim, 1, query_seq]` = `[2048, 1, query_seq]`
///   - `gate`:  `[num_heads * head_dim, 1, query_seq]` = `[2048, 1, query_seq]`
///   - `key`:   `[num_heads * head_dim, 1, key_seq]`   = `[2048, 1, key_seq]` (GQA-expanded)
///   - `value`: `[num_heads * head_dim, 1, key_seq]`   = `[2048, 1, key_seq]` (GQA-expanded)
///   - `mask`:  `[1, 1, query_seq, key_seq]` (prefill only, decode uses placeholder)
///
/// **Output:** `[hidden_size, 1, query_seq]` = `[1024, 1, query_seq]`
fn full_attn_body(
    graph: &mut Graph,
    query: ane::Tensor,
    gate: ane::Tensor,
    key: ane::Tensor,
    value: ane::Tensor,
    mask: ane::Tensor,
    weights: &FullAttentionWeights,
    config: &QwenConfig,
    query_seq: usize,
    key_seq: usize,
) -> ane::Tensor {
    let num_heads = config.num_attention_heads; // 8
    let head_dim = config.head_dim; // 256
    let hidden = config.hidden_size; // 1024
    let transpose_hw = [0, 1, 3, 2];

    // Reshape Q to multi-head: [1, num_heads, head_dim, query_seq]
    let q = graph.reshape(
        query,
        Shape { batch: 1, channels: num_heads, height: head_dim, width: query_seq },
    );
    // Transpose to [1, num_heads, query_seq, head_dim] for matmul
    let q = graph.transpose(q, transpose_hw);

    // Reshape K to multi-head: [1, num_heads, head_dim, key_seq]
    let k = graph.reshape(
        key,
        Shape { batch: 1, channels: num_heads, height: head_dim, width: key_seq },
    );
    // K stays as [1, num_heads, head_dim, key_seq] — we'll transpose in matmul

    // Reshape V to multi-head: [1, num_heads, head_dim, key_seq]
    let v = graph.reshape(
        value,
        Shape { batch: 1, channels: num_heads, height: head_dim, width: key_seq },
    );
    // Transpose to [1, num_heads, key_seq, head_dim]
    let v = graph.transpose(v, transpose_hw);

    // Attention scores: Q @ K^T = [1, num_heads, query_seq, key_seq]
    // Q is [1, num_heads, query_seq, head_dim], K is [1, num_heads, head_dim, key_seq]
    // matmul(Q, K, false, false) gives [1, num_heads, query_seq, key_seq]
    let scores = graph.matrix_multiplication(q, k, false, false);

    // Scale by 1/sqrt(head_dim)
    let scale = graph.constant_with_scalar(1.0 / (head_dim as f32).sqrt(), scalar_shape());
    let scores = graph.multiplication(scores, scale);

    // Apply causal mask
    let scores = graph.addition(scores, mask);

    // Softmax over key dimension (last axis)
    let probs = graph.soft_max(scores, -1);

    // Attention output: probs @ V = [1, num_heads, query_seq, head_dim]
    let attn = graph.matrix_multiplication(probs, v, false, false);

    // Transpose back to [1, num_heads, head_dim, query_seq]
    let attn = graph.transpose(attn, transpose_hw);

    // Reshape to flat: [num_heads * head_dim, 1, query_seq] = [2048, 1, query_seq]
    let attn = graph.reshape(attn, Shape::spatial(num_heads * head_dim, 1, query_seq));

    // Sigmoid output gate: attn *= sigmoid(gate)
    let gate_sig = graph.sigmoid(gate);
    let gated = graph.multiplication(attn, gate_sig);

    // Output projection: [2048, 1, query_seq] -> [1024, 1, query_seq]
    let o_weight = graph.constant(
        &weights.o_proj_weight,
        Shape::spatial(hidden, 1, 1),
    );
    graph.convolution_2d_1x1(gated, o_weight, None)
}

/// Prefill attention body.
///
/// 5 inputs: query, gate, key, value (all GQA-expanded), and causal mask.
/// Output: `[hidden_size, 1, seq]`.
pub fn build_full_attn_body_prefill(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
    seq_len: usize,
) -> Result<Executable, ane::Error> {
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim;
    let q_flat = num_heads * head_dim; // 2048

    let mut graph = Graph::new();
    let query = graph.placeholder(Shape::spatial(q_flat, 1, seq_len));
    let gate = graph.placeholder(Shape::spatial(q_flat, 1, seq_len));
    let key = graph.placeholder(Shape::spatial(q_flat, 1, seq_len));
    let value = graph.placeholder(Shape::spatial(q_flat, 1, seq_len));
    let mask = graph.placeholder(Shape {
        batch: 1, channels: 1, height: seq_len, width: seq_len,
    });

    let _output = full_attn_body(
        &mut graph, query, gate, key, value, mask,
        weights, config, seq_len, seq_len,
    );
    graph.compile(NSQualityOfService::Default)
}

/// Decode attention body.
///
/// 5 inputs: query `[q_flat, 1, 64]`, gate `[q_flat, 1, 64]`,
/// key `[q_flat, 1, max_seq]` (full cache), value `[q_flat, 1, max_seq]`,
/// mask `[1, 1, 64, max_seq]`.
/// Output: `[hidden_size, 1, 64]`.
pub fn build_full_attn_body_decode(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
    max_seq_len: usize,
) -> Result<Executable, ane::Error> {
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim;
    let q_flat = num_heads * head_dim; // 2048

    let mut graph = Graph::new();
    let query = graph.placeholder(Shape::spatial(q_flat, 1, DECODE_SPATIAL_WIDTH));
    let gate = graph.placeholder(Shape::spatial(q_flat, 1, DECODE_SPATIAL_WIDTH));
    let key = graph.placeholder(Shape::spatial(q_flat, 1, max_seq_len));
    let value = graph.placeholder(Shape::spatial(q_flat, 1, max_seq_len));
    let mask = graph.placeholder(Shape {
        batch: 1, channels: 1, height: DECODE_SPATIAL_WIDTH, width: max_seq_len,
    });

    let _output = full_attn_body(
        &mut graph, query, gate, key, value, mask,
        weights, config, DECODE_SPATIAL_WIDTH, max_seq_len,
    );
    graph.compile(NSQualityOfService::Default)
}

// ---------------------------------------------------------------------------
// SwiGLU Feed-Forward Network
// ---------------------------------------------------------------------------

/// Build the SwiGLU FFN body: RMSNorm -> gate/up projections -> SiLU gating -> down projection.
///
/// **Input:**  `[hidden_size, 1, seq]` = `[1024, 1, seq]`
/// **Output:** `[hidden_size, 1, seq]` = `[1024, 1, seq]`
///
/// SwiGLU: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
fn swiglu_ffn_body(
    graph: &mut Graph,
    input: ane::Tensor,
    norm_weight: &[f32],
    ffn: &FfnWeights,
    hidden: usize,
    intermediate: usize,
    eps: f64,
) -> (ane::Tensor, ane::Tensor) {
    // Post-attention RMSNorm
    let normalized = rms_norm(graph, input, norm_weight, hidden, eps);

    // Gate projection: [hidden_size] -> [intermediate_size]
    let gate_w = graph.constant(&ffn.gate_proj_weight, Shape::spatial(intermediate, 1, 1));
    let gate = graph.convolution_2d_1x1(normalized, gate_w, None);
    let gate = silu(graph, gate);

    // Up projection: [hidden_size] -> [intermediate_size]
    let up_w = graph.constant(&ffn.up_proj_weight, Shape::spatial(intermediate, 1, 1));
    let up = graph.convolution_2d_1x1(normalized, up_w, None);

    // Gated multiplication: SiLU(gate) * up
    let gated = graph.multiplication(gate, up);

    // Down projection: [intermediate_size] -> [hidden_size]
    let down_w = graph.constant(&ffn.down_proj_weight, Shape::spatial(hidden, 1, 1));
    (graph.convolution_2d_1x1(gated, down_w, None), normalized)
}

fn build_ffn_executable(
    norm_weight: &[f32],
    ffn_variant: &crate::weights::FfnVariant,
    config: &QwenConfig,
    seq_len: usize,
) -> Result<FfnExecutable, ane::Error> {
    match ffn_variant {
        crate::weights::FfnVariant::Dense(ffn) => {
            let mut graph = Graph::new();
            let input = graph.placeholder(Shape::spatial(config.hidden_size, 1, seq_len));
            let (_out, _norm) = swiglu_ffn_body(
                &mut graph, input, norm_weight, ffn,
                config.hidden_size, config.intermediate_size, config.rms_norm_eps
            );
            Ok(FfnExecutable::Dense(graph.compile(NSQualityOfService::Default)?))
        }
        crate::weights::FfnVariant::Moe(moe) => {
            let mut graph = Graph::new();
            let input = graph.placeholder(Shape::spatial(config.hidden_size, 1, seq_len));
            let shared_intermediate = config.shared_expert_intermediate_size.unwrap();
            let (shared_out, normalized) = swiglu_ffn_body(
                &mut graph, input, norm_weight, &moe.shared_expert,
                config.hidden_size, shared_intermediate, config.rms_norm_eps
            );

            // If there's a shared expert gate, multiply output by F.sigmoid(gate(normalized_x))
            // Qwen2Moe uses a linear layer for the shared expert gate: [1, hidden_size]
            if let Some(gate_w) = &moe.shared_expert_gate_weight {
                let gate_w_const = graph.constant(gate_w, Shape::spatial(1, 1, config.hidden_size));
                let norm_transposed = graph.transpose(normalized, [0, 2, 3, 1]); // [1, 1, seq, hidden]
                let gate_logits = graph.matrix_multiplication(norm_transposed, gate_w_const, false, true); // [1, 1, seq, 1]
                let gate_act = graph.sigmoid(gate_logits); // [1, 1, seq, 1]

                // Multiply gate_act by shared_out [1, hidden, 1, seq].
                // We reshape gate_act to [1, 1, 1, seq] to broadcast nicely.
                let gate_reshaped = graph.reshape(gate_act, Shape::spatial(1, 1, seq_len));
                let _gated_shared_out = graph.multiplication(shared_out, gate_reshaped);
            }

            Ok(FfnExecutable::Moe {
                shared_expert: graph.compile(NSQualityOfService::Default)?
            })
        }
    }
}

/// Prefill FFN for a full attention layer.
/// Input: `[hidden_size, 1, seq]` -> Output: `[hidden_size, 1, seq]`.
pub fn build_full_attn_ffn_prefill(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
    seq_len: usize,
) -> Result<FfnExecutable, ane::Error> {
    build_ffn_executable(&weights.post_attention_layernorm_weight, &weights.ffn, config, seq_len)
}

/// Decode FFN for a full attention layer.
/// Input: `[hidden_size, 1, 64]` -> Output: `[hidden_size, 1, 64]`.
pub fn build_full_attn_ffn_decode(
    weights: &FullAttentionWeights,
    config: &QwenConfig,
) -> Result<FfnExecutable, ane::Error> {
    build_ffn_executable(&weights.post_attention_layernorm_weight, &weights.ffn, config, DECODE_SPATIAL_WIDTH)
}

/// Prefill FFN for a linear attention (DeltaNet) layer.
/// Same SwiGLU structure, different weight source.
/// Input: `[hidden_size, 1, seq]` -> Output: `[hidden_size, 1, seq]`.
pub fn build_linear_attn_ffn_prefill(
    weights: &LinearAttentionWeights,
    config: &QwenConfig,
    seq_len: usize,
) -> Result<FfnExecutable, ane::Error> {
    build_ffn_executable(&weights.post_attention_layernorm_weight, &weights.ffn, config, seq_len)
}

/// Decode FFN for a linear attention (DeltaNet) layer.
/// Input: `[hidden_size, 1, 64]` -> Output: `[hidden_size, 1, 64]`.
pub fn build_linear_attn_ffn_decode(
    weights: &LinearAttentionWeights,
    config: &QwenConfig,
) -> Result<FfnExecutable, ane::Error> {
    build_ffn_executable(&weights.post_attention_layernorm_weight, &weights.ffn, config, DECODE_SPATIAL_WIDTH)
}

/// Build generic MoE expert executable.
///
/// Takes normalized input [1, hidden, 1, seq] and placeholders for the expert weights:
/// gate: [1, 1, intermediate, hidden]
/// up: [1, 1, intermediate, hidden]
/// down: [1, 1, hidden, intermediate]
fn build_generic_moe_expert(config: &QwenConfig, seq_len: usize) -> Result<Executable, ane::Error> {
    let mut graph = Graph::new();
    let hidden = config.hidden_size;
    let intermediate = config.moe_intermediate_size.unwrap_or(1);

    // [1, hidden, 1, seq]
    let input = graph.placeholder(Shape::spatial(hidden, 1, seq_len));

    let gate_w = graph.placeholder(Shape { batch: 1, channels: 1, height: intermediate, width: hidden });
    let up_w = graph.placeholder(Shape { batch: 1, channels: 1, height: intermediate, width: hidden });
    let down_w = graph.placeholder(Shape { batch: 1, channels: 1, height: hidden, width: intermediate });

    // Transpose input to [1, 1, seq, hidden]
    let x_t = graph.transpose(input, [0, 2, 3, 1]);

    // gate = matmul(x_t, gate_w^T) => [1, 1, seq, intermediate]
    let gate = graph.matrix_multiplication(x_t, gate_w, false, true);
    let gate_act = silu(&mut graph, gate);

    // up = matmul(x_t, up_w^T) => [1, 1, seq, intermediate]
    let up = graph.matrix_multiplication(x_t, up_w, false, true);

    // multiply gate and up => [1, 1, seq, intermediate]
    let gated = graph.multiplication(gate_act, up);

    // down = matmul(gated, down_w^T) => [1, 1, seq, hidden]
    let out_t = graph.matrix_multiplication(gated, down_w, false, true);

    // Transpose back to [1, hidden, 1, seq]
    let _out = graph.transpose(out_t, [0, 3, 1, 2]);

    graph.compile(NSQualityOfService::Default)
}

// ---------------------------------------------------------------------------
// Top-level compilation
// ---------------------------------------------------------------------------

/// Compile all ANE executables for every layer at the given sequence lengths.
///
/// For each of the 24 layers:
/// - **Full attention** (6 layers): projection + attention body + FFN (3 executables x2 phases)
/// - **Linear attention** (18 layers): FFN only (1 executable x2 phases), DeltaNet on CPU
pub fn compile_all(
    layer_weights: &[LayerWeights],
    config: &QwenConfig,
    padded_prompt_length: usize,
    max_sequence_length: usize,
) -> Result<CompiledExecutables, ane::Error> {
    let mut moe_expert_prefill = None;
    let mut moe_expert_decode = None;
    if config.num_experts.unwrap_or(0) > 0 {
        moe_expert_prefill = Some(build_generic_moe_expert(config, padded_prompt_length)?);
        moe_expert_decode = Some(build_generic_moe_expert(config, DECODE_SPATIAL_WIDTH)?);
    }

    let prefill: Box<[PrefillLayer]> = layer_weights
        .iter()
        .map(|lw| match lw {
            LayerWeights::FullAttention(w) => Ok(PrefillLayer::FullAttention(
                FullAttnPrefillLayer {
                    projection: build_full_attn_projection_prefill(
                        w, config, padded_prompt_length,
                    )?,
                    attention: build_full_attn_body_prefill(
                        w, config, padded_prompt_length,
                    )?,
                    feed_forward: build_full_attn_ffn_prefill(
                        w, config, padded_prompt_length,
                    )?,
                },
            )),
            LayerWeights::LinearAttention(w) => Ok(PrefillLayer::LinearAttention(
                LinearAttnPrefillLayer {
                    feed_forward: build_linear_attn_ffn_prefill(
                        w, config, padded_prompt_length,
                    )?,
                },
            )),
        })
        .collect::<Result<_, ane::Error>>()?;

    let decode: Box<[DecodeLayer]> = layer_weights
        .iter()
        .map(|lw| match lw {
            LayerWeights::FullAttention(w) => Ok(DecodeLayer::FullAttention(
                FullAttnDecodeLayer {
                    projection: build_full_attn_projection_decode(w, config)?,
                    attention: build_full_attn_body_decode(
                        w, config, max_sequence_length,
                    )?,
                    feed_forward: build_full_attn_ffn_decode(w, config)?,
                },
            )),
            LayerWeights::LinearAttention(w) => Ok(DecodeLayer::LinearAttention(
                LinearAttnDecodeLayer {
                    feed_forward: build_linear_attn_ffn_decode(w, config)?,
                },
            )),
        })
        .collect::<Result<_, ane::Error>>()?;

    Ok(CompiledExecutables {
        prefill,
        decode,
        moe_expert_prefill,
        moe_expert_decode,
    })
}
