use ane::{Shape, TensorData};

use crate::compiled_model::CompiledModel;
use crate::config::QwenConfig;
use crate::deltanet::DeltaNetState;
use crate::executables::{DecodeLayer, FullAttnDecodeLayer, FullAttnPrefillLayer, PrefillLayer, DECODE_SPATIAL_WIDTH};
use crate::kv_cache::KvCache;
use crate::lm_head;
use crate::rope::RopeTable;
use crate::weights::{FullAttentionWeights, LayerWeights, LinearAttentionWeights};

/// Inference session: owns KV cache, DeltaNet state, RoPE tables, and scratch IOSurfaces.
///
/// Borrows an immutable [`CompiledModel`] and provides `prefill()` and
/// `decode_step()` methods that return raw logits.
pub struct Session<'model> {
    model: &'model CompiledModel,
    kv_cache: KvCache,
    deltanet_state: DeltaNetState,
    rope_table: RopeTable,

    // -- Scratch IOSurfaces for ANE execution --

    // Prefill scratch
    prefill_hidden: TensorData,
    /// Projection output: [q_dim + 2*kv_dim, 1, padded_seq] = [5120, 1, seq]
    prefill_proj_out: TensorData,
    /// Attention body output: [hidden_size, 1, padded_seq]
    prefill_attn_out: TensorData,
    /// Attention body inputs (Q, gate, K_expanded, V_expanded, mask) — separate TensorData
    prefill_q: TensorData,
    prefill_gate: TensorData,
    prefill_k_expanded: TensorData,
    prefill_v_expanded: TensorData,
    prefill_mask: TensorData,
    /// FFN output: [hidden_size, 1, padded_seq]
    prefill_ffn_out: TensorData,
    /// MoE shared expert norm output: [hidden_size, 1, padded_seq]
    prefill_norm_out: Option<TensorData>,

    // Decode scratch
    decode_hidden: TensorData,
    decode_proj_out: TensorData,
    decode_attn_out: TensorData,
    decode_q: TensorData,
    decode_gate: TensorData,
    decode_k_expanded: TensorData,
    decode_v_expanded: TensorData,
    decode_mask: TensorData,
    decode_ffn_out: TensorData,
    decode_norm_out: Option<TensorData>,

    /// Current position in the sequence.
    position: usize,
}

impl<'model> Session<'model> {
    pub fn new(model: &'model CompiledModel, padded_prompt_length: usize) -> Self {
        let config = &model.config;
        let hidden = config.hidden_size;
        let max_seq = model.max_sequence_length;
        let q_dim = config.q_dim(); // 4096
        let kv_dim = config.kv_dim(); // 512
        let qk_flat = config.num_attention_heads * config.head_dim; // 2048
        let proj_total = q_dim + 2 * kv_dim; // 5120

        // Count full attention layers for KV cache
        let num_full_attn = (0..config.num_hidden_layers)
            .filter(|&i| config.is_full_attention(i))
            .count();
        let num_linear_attn = config.num_hidden_layers - num_full_attn;

        let rope_table = RopeTable::new(config.rope_dim(), max_seq, config.rope_theta);

        Self {
            model,
            kv_cache: KvCache::new(num_full_attn, kv_dim, max_seq),
            deltanet_state: DeltaNetState::new(
                num_linear_attn,
                config.linear_num_key_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                config.linear_total_qkv_dim(),
                config.hidden_size,
                config.rms_norm_eps as f32,
            ),
            rope_table,

            // Prefill scratch
            prefill_hidden: TensorData::new(Shape::spatial(hidden, 1, padded_prompt_length)),
            prefill_proj_out: TensorData::new(Shape::spatial(proj_total, 1, padded_prompt_length)),
            prefill_attn_out: TensorData::new(Shape::spatial(hidden, 1, padded_prompt_length)),
            prefill_q: TensorData::new(Shape::spatial(qk_flat, 1, padded_prompt_length)),
            prefill_gate: TensorData::new(Shape::spatial(qk_flat, 1, padded_prompt_length)),
            prefill_k_expanded: TensorData::new(Shape::spatial(qk_flat, 1, padded_prompt_length)),
            prefill_v_expanded: TensorData::new(Shape::spatial(qk_flat, 1, padded_prompt_length)),
            prefill_mask: TensorData::new(Shape { batch: 1, channels: 1, height: padded_prompt_length, width: padded_prompt_length }),
            prefill_ffn_out: TensorData::new(Shape::spatial(hidden, 1, padded_prompt_length)),
            prefill_norm_out: if config.num_experts.unwrap_or(0) > 0 {
                Some(TensorData::new(Shape::spatial(hidden, 1, padded_prompt_length)))
            } else { None },

            // Decode scratch
            decode_hidden: TensorData::new(Shape::spatial(hidden, 1, DECODE_SPATIAL_WIDTH)),
            decode_proj_out: TensorData::new(Shape::spatial(proj_total, 1, DECODE_SPATIAL_WIDTH)),
            decode_attn_out: TensorData::new(Shape::spatial(hidden, 1, DECODE_SPATIAL_WIDTH)),
            decode_q: TensorData::new(Shape::spatial(qk_flat, 1, DECODE_SPATIAL_WIDTH)),
            decode_gate: TensorData::new(Shape::spatial(qk_flat, 1, DECODE_SPATIAL_WIDTH)),
            decode_k_expanded: TensorData::new(Shape::spatial(qk_flat, 1, max_seq)),
            decode_v_expanded: TensorData::new(Shape::spatial(qk_flat, 1, max_seq)),
            decode_mask: TensorData::new(Shape { batch: 1, channels: 1, height: DECODE_SPATIAL_WIDTH, width: max_seq }),
            decode_ffn_out: TensorData::new(Shape::spatial(hidden, 1, DECODE_SPATIAL_WIDTH)),
            decode_norm_out: if config.num_experts.unwrap_or(0) > 0 {
                Some(TensorData::new(Shape::spatial(hidden, 1, DECODE_SPATIAL_WIDTH)))
            } else { None },

            position: 0,
        }
    }

    /// Run prefill: process padded prompt through all layers, return logits for last real token.
    pub fn prefill(&mut self, token_ids: &[u32], real_prompt_length: usize) -> Box<[f32]> {
        let config = &self.model.config;
        let hidden = config.hidden_size;
        let seq_len = token_ids.len();
        // Embedding lookup: token embeddings (no position embeddings — Qwen uses RoPE)
        {
            let mut surface = self.prefill_hidden.as_f32_slice_mut();
            embedding_lookup_into(&mut surface, token_ids, &self.model.weights.embed_tokens, hidden);
        }

        let mut full_attn_idx = 0usize;
        let mut linear_attn_idx = 0usize;

        for (layer_index, (layer_exec, layer_weights)) in self
            .model
            .executables
            .prefill
            .iter()
            .zip(self.model.weights.layers.iter())
            .enumerate()
        {
            match (layer_exec, layer_weights) {
                (PrefillLayer::FullAttention(exec), LayerWeights::FullAttention(weights)) => {
                    self.prefill_full_attention_layer(
                        exec, weights, config, layer_index, full_attn_idx, seq_len, real_prompt_length,
                    );
                    full_attn_idx += 1;
                }
                (PrefillLayer::LinearAttention(exec), LayerWeights::LinearAttention(weights)) => {
                    self.prefill_linear_attention_layer(
                        exec, weights, config, layer_index, linear_attn_idx, seq_len, real_prompt_length,
                    );
                    linear_attn_idx += 1;
                }
                _ => panic!("layer type mismatch at index {layer_index}"),
            }
        }

        self.position = real_prompt_length;
        self.kv_cache.position = real_prompt_length;

        // Extract last real token's hidden state and compute logits
        self.compute_logits_from_prefill(seq_len, real_prompt_length)
    }

    /// Run one autoregressive decode step: process a single token, return logits.
    pub fn decode_step(&mut self, token: u32) -> Box<[f32]> {
        let config = &self.model.config;
        let hidden = config.hidden_size;

        // Embedding lookup for single token (padded to DECODE_SPATIAL_WIDTH)
        {
            let mut surface = self.decode_hidden.as_f32_slice_mut();
            surface.fill(0.0);
            let token_idx = token as usize;
            for dim in 0..hidden {
                surface[dim * DECODE_SPATIAL_WIDTH] =
                    self.model.weights.embed_tokens[token_idx * hidden + dim];
            }
        }

        let mut full_attn_idx = 0usize;
        let mut linear_attn_idx = 0usize;

        for (layer_index, (layer_exec, layer_weights)) in self
            .model
            .executables
            .decode
            .iter()
            .zip(self.model.weights.layers.iter())
            .enumerate()
        {
            match (layer_exec, layer_weights) {
                (DecodeLayer::FullAttention(exec), LayerWeights::FullAttention(weights)) => {
                    self.decode_full_attention_layer(exec, weights, config, layer_index, full_attn_idx);
                    full_attn_idx += 1;
                }
                (DecodeLayer::LinearAttention(exec), LayerWeights::LinearAttention(weights)) => {
                    self.decode_linear_attention_layer(exec, weights, config, layer_index, linear_attn_idx);
                    linear_attn_idx += 1;
                }
                _ => panic!("layer type mismatch at index {layer_index}"),
            }
        }

        self.position += 1;
        self.kv_cache.position = self.position;

        self.compute_logits_from_decode()
    }

    // -----------------------------------------------------------------------
    // FFN execution helper (MoE router + dense fallback)
    // -----------------------------------------------------------------------

    fn run_ffn_layer(
        &mut self,
        exec: &crate::executables::FfnExecutable,
        layer_index: usize,
        seq_len: usize,
        is_prefill: bool,
    ) {
        let (hidden_in, ffn_out) = if is_prefill {
            (&self.prefill_hidden, &self.prefill_ffn_out)
        } else {
            (&self.decode_hidden, &self.decode_ffn_out)
        };

        match exec {
            crate::executables::FfnExecutable::Dense(dense) => {
                dense
                    .run(&[hidden_in], &[ffn_out])
                    .unwrap_or_else(|e| panic!("L{layer_index} dense ffn: {e}"));
            }
            crate::executables::FfnExecutable::Moe { shared_expert } => {
                let norm_out = if is_prefill {
                    self.prefill_norm_out.as_ref().unwrap()
                } else {
                    self.decode_norm_out.as_ref().unwrap()
                };

                // 1. Run shared expert; output goes to ffn_out, and normalized goes to norm_out
                shared_expert
                    .run(&[hidden_in], &[ffn_out, norm_out])
                    .unwrap_or_else(|e| panic!("L{layer_index} shared expert: {e}"));

                let config = &self.model.config;
                let num_experts = config.num_experts.unwrap();
                let num_active = config.num_experts_per_tok.unwrap();

                let moe_weights = match &self.model.weights.layers[layer_index] {
                    crate::weights::LayerWeights::FullAttention(w) => match &w.ffn {
                        crate::weights::FfnVariant::Moe(m) => m,
                        _ => unreachable!(),
                    },
                    crate::weights::LayerWeights::LinearAttention(w) => match &w.ffn {
                        crate::weights::FfnVariant::Moe(m) => m,
                        _ => unreachable!(),
                    }
                };

                let gate_w = &moe_weights.gate_weight;
                let hidden = config.hidden_size;
                let expert_tensors = &self.model.moe_expert_tensors.as_ref().unwrap()[layer_index];

                let generic_expert = if is_prefill {
                    self.model.executables.moe_expert_prefill.as_ref().unwrap()
                } else {
                    self.model.executables.moe_expert_decode.as_ref().unwrap()
                };

                // Allocate a single token buffer for routing and executing experts
                let token_in = ane::TensorData::new(ane::Shape::spatial(hidden, 1, seq_len));
                let mut token_out = ane::TensorData::new(ane::Shape::spatial(hidden, 1, seq_len));

                // 2. CPU Router + dynamic expert dispatch
                let norm_slice = norm_out.as_f32_slice();
                let mut out_slice = ffn_out.as_f32_slice_mut();

                for s in 0..seq_len {
                    // Compute router logits
                    let mut logits = vec![0.0f32; num_experts];
                    for e in 0..num_experts {
                        let mut sum = 0.0;
                        for ch in 0..hidden {
                            sum += norm_slice[ch * seq_len + s] * gate_w[e * hidden + ch];
                        }
                        logits[e] = sum;
                    }

                    // Top-K routing
                    let mut indices: Vec<usize> = (0..num_experts).collect();
                    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
                    indices.truncate(num_active);

                    let max_logit = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
                    let mut routing_weights = vec![0.0f32; num_active];
                    let mut exp_sum = 0.0;
                    for (k, &e) in indices.iter().enumerate() {
                        let w = (logits[e] - max_logit).exp();
                        routing_weights[k] = w;
                        exp_sum += w;
                    }
                    for k in 0..num_active {
                        routing_weights[k] /= exp_sum;
                    }

                    // For each selected expert, run the generic ANE executable
                    for (k, &e) in indices.iter().enumerate() {
                        let w = routing_weights[k];
                        let tensors = &expert_tensors[e];

                        // Copy the single token to expert input
                        {
                            let mut t_in = token_in.as_f32_slice_mut();
                            t_in.fill(0.0);
                            for ch in 0..hidden {
                                t_in[ch * seq_len + s] = norm_slice[ch * seq_len + s];
                            }
                        }

                        // Run ANE expert! The graph expects: [input, gate_w, up_w, down_w]
                        generic_expert
                            .run(
                                &[&token_in, &tensors.gate_w, &tensors.up_w, &tensors.down_w],
                                &[&mut token_out]
                            )
                            .unwrap_or_else(|err| panic!("L{layer_index} expert {e}: {err}"));

                        // Accumulate output
                        {
                            let t_out = token_out.as_f32_slice();
                            for ch in 0..hidden {
                                out_slice[ch * seq_len + s] += w * t_out[ch * seq_len + s];
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full attention: prefill
    // -----------------------------------------------------------------------

    fn prefill_full_attention_layer(
        &mut self,
        exec: &FullAttnPrefillLayer,
        weights: &FullAttentionWeights,
        config: &QwenConfig,
        layer_index: usize,
        cache_idx: usize,
        seq_len: usize,
        real_length: usize,
    ) {
        let _hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kv_dim = config.kv_dim();
        let q_dim = config.q_dim();
        let qk_flat = n_heads * head_dim; // 2048

        // 1. ANE projection: hidden → [q_dim + 2*kv_dim, 1, seq]
        exec.projection
            .run(&[&self.prefill_hidden], &[&self.prefill_proj_out])
            .unwrap_or_else(|e| panic!("prefill L{layer_index} projection: {e}"));

        // 2. CPU: split projection output, per-head norm, RoPE, GQA expand
        {
            let proj = self.prefill_proj_out.as_f32_slice();
            let mut q_surface = self.prefill_q.as_f32_slice_mut();
            let mut gate_surface = self.prefill_gate.as_f32_slice_mut();

            // Split Q+gate and K, V from projection output
            // Layout: [q_dim + kv_dim + kv_dim, 1, seq] channel-first
            // Q+gate occupies channels 0..q_dim, K is q_dim..q_dim+kv_dim, V is rest

            // Extract Q (first half of q_dim = qk_flat channels) and gate (second half)
            for ch in 0..qk_flat {
                for s in 0..seq_len {
                    q_surface[ch * seq_len + s] = proj[ch * seq_len + s];
                }
            }
            for ch in 0..qk_flat {
                for s in 0..seq_len {
                    gate_surface[ch * seq_len + s] = proj[(qk_flat + ch) * seq_len + s];
                }
            }

            // Per-head Q RMSNorm (in-place on q_surface)
            per_head_rms_norm_inplace(&mut q_surface, &weights.q_norm_weight, n_heads, head_dim, seq_len, config.rms_norm_eps as f32);

            // Extract K and apply per-head K RMSNorm
            let k_start = q_dim; // channel offset for K in projection output
            let mut k_raw = vec![0.0f32; kv_dim * seq_len];
            for ch in 0..kv_dim {
                for s in 0..seq_len {
                    k_raw[ch * seq_len + s] = proj[(k_start + ch) * seq_len + s];
                }
            }
            per_head_rms_norm_inplace(&mut k_raw, &weights.k_norm_weight, n_kv_heads, head_dim, seq_len, config.rms_norm_eps as f32);

            // Extract V
            let v_start = q_dim + kv_dim;
            let mut v_raw = vec![0.0f32; kv_dim * seq_len];
            for ch in 0..kv_dim {
                for s in 0..seq_len {
                    v_raw[ch * seq_len + s] = proj[(v_start + ch) * seq_len + s];
                }
            }

            // Apply partial RoPE to Q and K
            self.rope_table.apply_inplace(&mut q_surface, n_heads, head_dim, seq_len, 0);
            self.rope_table.apply_inplace(&mut k_raw, n_kv_heads, head_dim, seq_len, 0);

            // Write K, V to cache (before GQA expansion, raw 2-head form)
            self.kv_cache.write_kv_sequence(cache_idx, &k_raw, &v_raw, real_length, seq_len);

            // GQA expansion: 2 KV heads → 8 by repeating each head 4x
            let gqa_ratio = n_heads / n_kv_heads; // 4
            let mut k_expanded = self.prefill_k_expanded.as_f32_slice_mut();
            let mut v_expanded = self.prefill_v_expanded.as_f32_slice_mut();
            gqa_expand_channel_first(&k_raw, &mut k_expanded, n_kv_heads, head_dim, seq_len, gqa_ratio);
            gqa_expand_channel_first(&v_raw, &mut v_expanded, n_kv_heads, head_dim, seq_len, gqa_ratio);
        }

        // 3. ANE attention body
        exec.attention
            .run(
                &[&self.prefill_q, &self.prefill_gate, &self.prefill_k_expanded, &self.prefill_v_expanded, &self.prefill_mask],
                &[&self.prefill_attn_out],
            )
            .unwrap_or_else(|e| panic!("prefill L{layer_index} attention: {e}"));

        // Add attention output to residual
        self.prefill_hidden.add_from(&self.prefill_attn_out);

        // 4. FFN (Dense or MoE)
        self.run_ffn_layer(&exec.feed_forward, layer_index, seq_len, true);

        self.prefill_hidden.add_from(&self.prefill_ffn_out);
    }

    // -----------------------------------------------------------------------
    // Linear attention (DeltaNet): prefill
    // -----------------------------------------------------------------------

    fn prefill_linear_attention_layer(
        &mut self,
        exec: &crate::executables::LinearAttnPrefillLayer,
        weights: &LinearAttentionWeights,
        config: &QwenConfig,
        layer_index: usize,
        linear_idx: usize,
        seq_len: usize,
        real_length: usize,
    ) {
        let hidden = config.hidden_size;

        // DeltaNet on CPU: process each token sequentially.
        // The new API modifies hidden in-place (adds residual internally).
        for t in 0..real_length {
            // Extract hidden state for token t from channel-first layout
            let mut hidden_vec: Vec<f32> = {
                let surface = self.prefill_hidden.as_f32_slice();
                (0..hidden).map(|dim| surface[dim * seq_len + t]).collect()
            };

            // Run DeltaNet step (modifies hidden_vec in-place: adds attention output)
            self.deltanet_state.forward_step(linear_idx, &mut hidden_vec, weights);

            // Write back to channel-first surface
            {
                let mut surface = self.prefill_hidden.as_f32_slice_mut();
                for dim in 0..hidden {
                    surface[dim * seq_len + t] = hidden_vec[dim];
                }
            }
        }

        // ANE FFN
        self.run_ffn_layer(&exec.feed_forward, layer_index, seq_len, true);

        self.prefill_hidden.add_from(&self.prefill_ffn_out);
    }

    // -----------------------------------------------------------------------
    // Full attention: decode
    // -----------------------------------------------------------------------

    fn decode_full_attention_layer(
        &mut self,
        exec: &FullAttnDecodeLayer,
        weights: &FullAttentionWeights,
        config: &QwenConfig,
        layer_index: usize,
        cache_idx: usize,
    ) {
        let hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kv_dim = config.kv_dim();
        let q_dim = config.q_dim();
        let qk_flat = n_heads * head_dim;
        let max_seq = self.model.max_sequence_length;

        // 1. ANE projection
        exec.projection
            .run(&[&self.decode_hidden], &[&self.decode_proj_out])
            .unwrap_or_else(|e| panic!("decode L{layer_index} projection: {e}"));

        // 2. CPU: split, norm, RoPE, cache update, GQA expand
        {
            let proj = self.decode_proj_out.as_f32_slice();
            let mut q_surface = self.decode_q.as_f32_slice_mut();
            let mut gate_surface = self.decode_gate.as_f32_slice_mut();

            q_surface.fill(0.0);
            gate_surface.fill(0.0);

            // Extract Q and gate (single token at position 0 of padded width)
            for ch in 0..qk_flat {
                q_surface[ch * DECODE_SPATIAL_WIDTH] = proj[ch * DECODE_SPATIAL_WIDTH];
            }
            for ch in 0..qk_flat {
                gate_surface[ch * DECODE_SPATIAL_WIDTH] = proj[(qk_flat + ch) * DECODE_SPATIAL_WIDTH];
            }

            // Per-head Q/K RMSNorm
            per_head_rms_norm_inplace(&mut q_surface, &weights.q_norm_weight, n_heads, head_dim, DECODE_SPATIAL_WIDTH, config.rms_norm_eps as f32);

            // Extract K (single token)
            let k_start = q_dim;
            let mut k_vec = vec![0.0f32; kv_dim];
            for ch in 0..kv_dim {
                k_vec[ch] = proj[(k_start + ch) * DECODE_SPATIAL_WIDTH];
            }

            // Extract V (single token)
            let v_start = q_dim + kv_dim;
            let mut v_vec = vec![0.0f32; kv_dim];
            for ch in 0..kv_dim {
                v_vec[ch] = proj[(v_start + ch) * DECODE_SPATIAL_WIDTH];
            }

            // Per-head K RMSNorm (on the vector)
            per_head_rms_norm_vec(&mut k_vec, &weights.k_norm_weight, n_kv_heads, head_dim, config.rms_norm_eps as f32);

            // RoPE on Q (in IOSurface layout) and K (in vector)
            self.rope_table.apply_inplace(&mut q_surface, n_heads, head_dim, DECODE_SPATIAL_WIDTH, self.position);

            // RoPE on K vector: need to put in channel-first layout temporarily
            let mut k_surface = vec![0.0f32; kv_dim]; // seq_len=1, so same as vec
            k_surface.copy_from_slice(&k_vec);
            self.rope_table.apply_inplace(&mut k_surface, n_kv_heads, head_dim, 1, self.position);
            k_vec.copy_from_slice(&k_surface);

            // Write new K, V to cache
            self.kv_cache.write_kv(cache_idx, &k_vec, &v_vec, self.position);

            // GQA expand the full cache into decode_k_expanded, decode_v_expanded
            let gqa_ratio = n_heads / n_kv_heads;
            {
                let k_cache = self.kv_cache.keys[cache_idx].as_f32_slice();
                let mut k_exp = self.decode_k_expanded.as_f32_slice_mut();
                gqa_expand_channel_first(&k_cache, &mut k_exp, n_kv_heads, head_dim, max_seq, gqa_ratio);
            }
            {
                let v_cache = self.kv_cache.values[cache_idx].as_f32_slice();
                let mut v_exp = self.decode_v_expanded.as_f32_slice_mut();
                gqa_expand_channel_first(&v_cache, &mut v_exp, n_kv_heads, head_dim, max_seq, gqa_ratio);
            }
        }

        // Build decode mask
        {
            let mut mask = self.decode_mask.as_f32_slice_mut();
            mask.fill(-65504.0);
            for col in 0..=self.position {
                mask[col] = 0.0;
            }
        }

        // 3. ANE attention body
        exec.attention
            .run(
                &[&self.decode_q, &self.decode_gate, &self.decode_k_expanded, &self.decode_v_expanded, &self.decode_mask],
                &[&self.decode_attn_out],
            )
            .unwrap_or_else(|e| panic!("decode L{layer_index} attention: {e}"));

        // Add to residual
        {
            let delta = self.decode_attn_out.as_f32_slice();
            let mut h = self.decode_hidden.as_f32_slice_mut();
            for dim in 0..hidden {
                h[dim * DECODE_SPATIAL_WIDTH] += delta[dim * DECODE_SPATIAL_WIDTH];
            }
        }

        // 4. ANE FFN
        self.run_ffn_layer(&exec.feed_forward, layer_index, 1, false);

        {
            let delta = self.decode_ffn_out.as_f32_slice();
            let mut h = self.decode_hidden.as_f32_slice_mut();
            for dim in 0..hidden {
                h[dim * DECODE_SPATIAL_WIDTH] += delta[dim * DECODE_SPATIAL_WIDTH];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Linear attention (DeltaNet): decode
    // -----------------------------------------------------------------------

    fn decode_linear_attention_layer(
        &mut self,
        exec: &crate::executables::LinearAttnDecodeLayer,
        weights: &LinearAttentionWeights,
        config: &QwenConfig,
        layer_index: usize,
        linear_idx: usize,
    ) {
        let hidden = config.hidden_size;

        // Extract single token hidden state
        let mut hidden_vec: Vec<f32> = {
            let surface = self.decode_hidden.as_f32_slice();
            (0..hidden).map(|dim| surface[dim * DECODE_SPATIAL_WIDTH]).collect()
        };

        // DeltaNet step on CPU (modifies hidden_vec in-place: adds attention output)
        self.deltanet_state.forward_step(linear_idx, &mut hidden_vec, weights);

        // Write back to decode surface
        {
            let mut surface = self.decode_hidden.as_f32_slice_mut();
            for dim in 0..hidden {
                surface[dim * DECODE_SPATIAL_WIDTH] = hidden_vec[dim];
            }
        }

        // ANE FFN
        self.run_ffn_layer(&exec.feed_forward, layer_index, 1, false);

        {
            let delta = self.decode_ffn_out.as_f32_slice();
            let mut h = self.decode_hidden.as_f32_slice_mut();
            for dim in 0..hidden {
                h[dim * DECODE_SPATIAL_WIDTH] += delta[dim * DECODE_SPATIAL_WIDTH];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Logits computation
    // -----------------------------------------------------------------------

    fn compute_logits_from_prefill(&self, seq_len: usize, real_length: usize) -> Box<[f32]> {
        let config = &self.model.config;
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps as f32;

        // Extract last real token's hidden state from channel-first layout
        let last_hidden: Box<[f32]> = {
            let surface = self.prefill_hidden.as_f32_slice();
            (0..hidden)
                .map(|dim| surface[dim * seq_len + (real_length - 1)])
                .collect()
        };

        let mut normalized = vec![0.0f32; hidden];
        lm_head::final_rms_norm(&mut normalized, &last_hidden, &self.model.weights.norm_weight, hidden, eps);

        let mut logits = vec![0.0f32; config.vocab_size];
        lm_head::compute_logits(
            &mut logits, &self.model.weights.embed_tokens, &normalized,
            config.vocab_size, hidden,
        );

        logits.into_boxed_slice()
    }

    fn compute_logits_from_decode(&self) -> Box<[f32]> {
        let config = &self.model.config;
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps as f32;

        let last_hidden: Box<[f32]> = {
            let surface = self.decode_hidden.as_f32_slice();
            (0..hidden)
                .map(|dim| surface[dim * DECODE_SPATIAL_WIDTH])
                .collect()
        };

        let mut normalized = vec![0.0f32; hidden];
        lm_head::final_rms_norm(&mut normalized, &last_hidden, &self.model.weights.norm_weight, hidden, eps);

        let mut logits = vec![0.0f32; config.vocab_size];
        lm_head::compute_logits(
            &mut logits, &self.model.weights.embed_tokens, &normalized,
            config.vocab_size, hidden,
        );

        logits.into_boxed_slice()
    }
}

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/// Embedding lookup into channel-first layout (no position embeddings — Qwen uses RoPE).
fn embedding_lookup_into(
    destination: &mut [f32],
    token_ids: &[u32],
    token_embeddings: &[f32],
    embedding_dim: usize,
) {
    let seq_len = token_ids.len();
    for (seq_idx, &token_id) in token_ids.iter().enumerate() {
        let token = token_id as usize;
        for dim in 0..embedding_dim {
            destination[dim * seq_len + seq_idx] =
                token_embeddings[token * embedding_dim + dim];
        }
    }
}

/// Per-head RMSNorm in-place on channel-first data [num_heads * head_dim, 1, seq_len].
///
/// Each head is normalized independently using the same weight vector [head_dim].
fn per_head_rms_norm_inplace(
    data: &mut [f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    eps: f32,
) {
    for head in 0..num_heads {
        for s in 0..seq_len {
            // Compute RMS for this head at this position
            let mut sum_sq = 0.0f32;
            for d in 0..head_dim {
                let ch = head * head_dim + d;
                let val = data[ch * seq_len + s];
                sum_sq += val * val;
            }
            let rstd = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                let ch = head * head_dim + d;
                data[ch * seq_len + s] *= rstd * weight[d];
            }
        }
    }
}

/// Per-head RMSNorm on a flat vector [num_heads * head_dim] (single token).
fn per_head_rms_norm_vec(
    data: &mut [f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for head in 0..num_heads {
        let start = head * head_dim;
        let mut sum_sq = 0.0f32;
        for d in 0..head_dim {
            sum_sq += data[start + d] * data[start + d];
        }
        let rstd = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            data[start + d] *= rstd * weight[d];
        }
    }
}

/// GQA expansion: repeat each KV head `ratio` times in channel-first layout.
///
/// src: [n_kv_heads * head_dim, 1, seq_len] → dst: [n_kv_heads * ratio * head_dim, 1, seq_len]
fn gqa_expand_channel_first(
    src: &[f32],
    dst: &mut [f32],
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    ratio: usize,
) {
    for kv_head in 0..n_kv_heads {
        for rep in 0..ratio {
            let dst_head = kv_head * ratio + rep;
            for d in 0..head_dim {
                let src_ch = kv_head * head_dim + d;
                let dst_ch = dst_head * head_dim + d;
                let src_offset = src_ch * seq_len;
                let dst_offset = dst_ch * seq_len;
                dst[dst_offset..dst_offset + seq_len]
                    .copy_from_slice(&src[src_offset..src_offset + seq_len]);
            }
        }
    }
}

