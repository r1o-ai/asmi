use safetensors::SafeTensors;

use crate::config::QwenConfig;

use super::layer_weights::{
    FfnWeights, FullAttentionWeights, LayerWeights, LinearAttentionWeights,
};
use super::safetensors_ext::try_tensor_to_f32;

/// All model weights: token embeddings, per-layer weights, and final RMSNorm.
pub struct ModelWeights {
    /// Token embeddings: [vocab_size, hidden_size]
    pub embed_tokens: Box<[f32]>,
    /// Per-layer weights (24 layers, mixed full/linear attention)
    pub layers: Box<[LayerWeights]>,
    /// Final RMSNorm weight: [hidden_size]
    pub norm_weight: Box<[f32]>,
}

/// Helper to load a tensor from any of multiple safetensor shards.
///
/// Searches each shard in order and returns the first match.
fn tensor_from_shards<'a>(shards: &'a [SafeTensors<'a>], name: &str) -> Box<[f32]> {
    for shard in shards {
        if let Some(data) = try_tensor_to_f32(shard, name) {
            return data;
        }
    }
    panic!("tensor not found in any shard: {name}");
}

/// Helper to try loading a tensor from any shard, returning None if absent.
fn try_tensor_from_shards<'a>(shards: &'a [SafeTensors<'a>], name: &str) -> Option<Box<[f32]>> {
    for shard in shards {
        if let Some(data) = try_tensor_to_f32(shard, name) {
            return Some(data);
        }
    }
    None
}

/// Load FFN weights shared by both attention types.
fn load_ffn_weights(shards: &[SafeTensors], layer_prefix: &str) -> FfnWeights {
    FfnWeights {
        gate_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.gate_proj.weight")),
        up_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.up_proj.weight")),
        down_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.down_proj.weight")),
    }
}

/// Load FFN weights, dispatching to Dense or MoE depending on config.
fn load_ffn_variant(shards: &[SafeTensors], layer_prefix: &str, config: &QwenConfig, layer_idx: usize) -> crate::weights::layer_weights::FfnVariant {
    let is_moe = config.num_experts.unwrap_or(0) > 0 &&
                 config.decoder_sparse_step > 0 &&
                 layer_idx % config.decoder_sparse_step == 0;

    if is_moe {
        let num_experts = config.num_experts.unwrap();
        let gate_weight = tensor_from_shards(shards, &format!("{layer_prefix}.mlp.gate.weight"));

        let shared_expert = FfnWeights {
            gate_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.shared_expert.gate_proj.weight")),
            up_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.shared_expert.up_proj.weight")),
            down_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.shared_expert.down_proj.weight")),
        };

        let shared_expert_gate_weight = try_tensor_from_shards(shards, &format!("{layer_prefix}.mlp.shared_expert_gate.weight"));

        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let gate_proj_weight = try_tensor_from_shards(shards, &format!("{layer_prefix}.mlp.experts.{i}.gate_proj.weight"));
            let up_proj_weight = try_tensor_from_shards(shards, &format!("{layer_prefix}.mlp.experts.{i}.up_proj.weight"));

            if gate_proj_weight.is_none() {
                // If not found, check fused
                if let Some(_fused) = try_tensor_from_shards(shards, &format!("{layer_prefix}.mlp.experts.{i}.gate_up_proj.weight")) {
                    panic!("Fused gate_up_proj not fully implemented yet for expert {i}");
                }
            }

            experts.push(FfnWeights {
                gate_proj_weight: gate_proj_weight.unwrap_or_else(|| panic!("Missing gate_proj.weight for expert {i}")),
                up_proj_weight: up_proj_weight.unwrap_or_else(|| panic!("Missing up_proj.weight for expert {i}")),
                down_proj_weight: tensor_from_shards(shards, &format!("{layer_prefix}.mlp.experts.{i}.down_proj.weight")),
            });
        }

        crate::weights::layer_weights::FfnVariant::Moe(crate::weights::layer_weights::MoeWeights {
            gate_weight,
            shared_expert,
            shared_expert_gate_weight,
            experts,
        })
    } else {
        crate::weights::layer_weights::FfnVariant::Dense(load_ffn_weights(shards, layer_prefix))
    }
}

/// Load full (quadratic) attention layer weights.
fn load_full_attention_layer(
    shards: &[SafeTensors],
    layer_prefix: &str,
    config: &QwenConfig,
    layer_idx: usize,
) -> FullAttentionWeights {
    FullAttentionWeights {
        input_layernorm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.input_layernorm.weight"),
        ),
        q_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.q_proj.weight"),
        ),
        q_proj_bias: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.q_proj.bias"),
        ),
        k_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.k_proj.weight"),
        ),
        k_proj_bias: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.k_proj.bias"),
        ),
        v_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.v_proj.weight"),
        ),
        v_proj_bias: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.v_proj.bias"),
        ),
        o_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.o_proj.weight"),
        ),
        q_norm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.q_norm.weight"),
        ),
        k_norm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.k_norm.weight"),
        ),
        post_attention_layernorm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
        ),
        ffn: load_ffn_variant(shards, layer_prefix, config, layer_idx),
    }
}

/// Load linear (DeltaNet) attention layer weights.
fn load_linear_attention_layer(
    shards: &[SafeTensors],
    layer_prefix: &str,
    config: &QwenConfig,
    layer_idx: usize,
) -> LinearAttentionWeights {
    LinearAttentionWeights {
        input_layernorm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.input_layernorm.weight"),
        ),
        q_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.q_proj.weight"),
        ),
        k_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.k_proj.weight"),
        ),
        v_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.v_proj.weight"),
        ),
        a_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.a_proj.weight"),
        ),
        b_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.b_proj.weight"),
        ),
        z_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.z_proj.weight"),
        ),
        a_log: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.A_log"),
        ),
        dt_bias: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.dt_bias"),
        ),
        o_proj_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.o_proj.weight"),
        ),
        conv1d_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.conv1d.weight"),
        ),
        conv1d_bias: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.conv1d.bias"),
        ),
        norm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.norm.weight"),
        ),
        gate_norm_weight: try_tensor_from_shards(
            shards,
            &format!("{layer_prefix}.self_attn.gate_norm.weight"),
        ),
        post_attention_layernorm_weight: tensor_from_shards(
            shards,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
        ),
        ffn: load_ffn_variant(shards, layer_prefix, config, layer_idx),
    }
}

/// Load all Qwen 3.5 weights from one or more safetensor shards.
///
/// Weight matrices are stored in standard PyTorch `[out_features, in_features]`
/// layout and do NOT need transposing (unlike GPT-2 Conv1D weights).
pub fn load_weights(shards: &[SafeTensors], config: &QwenConfig) -> ModelWeights {
    let embed_tokens = tensor_from_shards(shards, "model.embed_tokens.weight");

    let layers: Box<[LayerWeights]> = (0..config.num_hidden_layers)
        .map(|i| {
            let layer_prefix = format!("model.layers.{i}");
            if config.is_full_attention(i) {
                LayerWeights::FullAttention(load_full_attention_layer(shards, &layer_prefix, config, i))
            } else {
                LayerWeights::LinearAttention(load_linear_attention_layer(shards, &layer_prefix, config, i))
            }
        })
        .collect();

    let norm_weight = tensor_from_shards(shards, "model.norm.weight");

    ModelWeights {
        embed_tokens,
        layers,
        norm_weight,
    }
}
