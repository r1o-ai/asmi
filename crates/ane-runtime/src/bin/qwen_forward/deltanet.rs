//! CPU-side Gated DeltaNet recurrence for the 18 linear attention layers.
//!
//! Qwen 3.5-0.8B uses a hybrid architecture where 18 of 24 layers use DeltaNet
//! (linear attention) instead of standard GQA. The entire linear attention layer
//! runs on CPU: projections, conv1d, recurrence, gated output, and o_proj.
//!
//! Per linear attention layer:
//! - num_heads: 16 (linear_num_key_heads / linear_num_value_heads)
//! - key_head_dim: 128
//! - value_head_dim: 128
//! - State per head: [key_dim, value_dim] = [128, 128] matrix
//! - Conv1d kernel size: 4
//!
//! The recurrence per head per token:
//! ```text
//! alpha = exp(-exp(A_log) * softplus(a + dt_bias))     // decay
//! beta = sigmoid(b)                                     // update gate
//! S = alpha * S                                         // decay old memory
//! predicted = S^T @ k                                   // predict value
//! delta = beta * (v - predicted)                        // error correction
//! S = S + outer(k, delta)                               // update memory
//! o = S^T @ q                                           // read output
//! ```
//!
//! Output goes through: `RMSNorm(o) * SiLU(z)`, then `o_proj`.

use crate::weights::LinearAttentionWeights;

// ---------------------------------------------------------------------------
// CPU math helpers
// ---------------------------------------------------------------------------

/// Matrix-vector product: out[i] = sum_j(mat[i * cols + j] * vec[j])
///
/// `mat` is `[rows, cols]` row-major, `vec` is `[cols]`, `out` is `[rows]`.
fn matvec(out: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let mut sum = 0.0f32;
        let row_offset = i * cols;
        for j in 0..cols {
            sum += mat[row_offset + j] * vec[j];
        }
        out[i] = sum;
    }
}

/// Transposed matrix-vector product: out[j] = sum_i(mat[i * cols + j] * vec[i])
///
/// Equivalent to mat^T @ vec, where mat is [rows, cols], vec is [rows], out is [cols].
fn matvec_transposed(out: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    out[..cols].fill(0.0);
    for i in 0..rows {
        let row_offset = i * cols;
        let vi = vec[i];
        for j in 0..cols {
            out[j] += mat[row_offset + j] * vi;
        }
    }
}

/// Rank-1 update: mat[i * cols + j] += vec_a[i] * vec_b[j]
///
/// `mat` is `[rows, cols]`, `vec_a` is `[rows]`, `vec_b` is `[cols]`.
fn outer_add(mat: &mut [f32], vec_a: &[f32], vec_b: &[f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_offset = i * cols;
        let ai = vec_a[i];
        for j in 0..cols {
            mat[row_offset + j] += ai * vec_b[j];
        }
    }
}

/// Scale all elements: mat[i] *= scalar
fn mat_scale(mat: &mut [f32], scalar: f32) {
    for x in mat.iter_mut() {
        *x *= scalar;
    }
}

/// RMSNorm: out[i] = (x[i] / rms) * weight[i]
fn rms_norm_vec(out: &mut [f32], x: &[f32], weight: &[f32], dim: usize, eps: f32) {
    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        sum_sq += x[i] * x[i];
    }
    let rstd = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * rstd * weight[i];
    }
}

/// L2-normalize a vector in-place: x[i] /= ||x||_2
fn l2_normalize(x: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for &v in x.iter() {
        norm_sq += v * v;
    }
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for v in x.iter_mut() {
            *v *= inv_norm;
        }
    }
}

/// softplus(x) = ln(1 + exp(x))
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // avoid overflow
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// sigmoid(x) = 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU(x) = x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ---------------------------------------------------------------------------
// DeltaNet state
// ---------------------------------------------------------------------------

/// Persistent recurrent state for the 18 DeltaNet (linear attention) layers.
///
/// Each layer has `num_heads` independent recurrent states, each a
/// `[key_dim, value_dim]` matrix. Additionally, each layer maintains a
/// conv1d sliding window state for the `(kernel_size - 1)` most recent
/// concatenated QKV inputs.
pub struct DeltaNetState {
    /// Recurrent states: one `[key_dim * value_dim]` matrix per head per layer.
    /// Indexed as `states[layer_idx]`, which contains `[num_heads * key_dim * value_dim]`.
    pub states: Box<[Box<[f32]>]>,

    /// Conv1d sliding window state per layer.
    /// Each stores `(kernel_size - 1)` frames per channel, laid out as
    /// `[channel * (kernel_size - 1) + frame_index]`.
    pub conv_states: Box<[Box<[f32]>]>,

    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel_size: usize,
    pub total_qkv_dim: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
}

impl DeltaNetState {
    /// Create zero-initialized DeltaNet state for all linear attention layers.
    ///
    /// - `num_linear_layers`: number of DeltaNet layers (18 for Qwen 3.5)
    /// - `num_heads`: heads per layer (16)
    /// - `key_dim`: per-head key dimension (128)
    /// - `value_dim`: per-head value dimension (128)
    /// - `conv_kernel_size`: conv1d kernel width (4)
    /// - `total_qkv_dim`: concatenated QKV dimension for conv1d
    /// - `hidden_size`: model hidden dimension (1024)
    /// - `rms_norm_eps`: epsilon for RMSNorm (1e-6)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_linear_layers: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel_size: usize,
        total_qkv_dim: usize,
        hidden_size: usize,
        rms_norm_eps: f32,
    ) -> Self {
        let state_size = num_heads * key_dim * value_dim;
        let hist_len = conv_kernel_size - 1;
        let conv_state_size = total_qkv_dim * hist_len;

        Self {
            states: (0..num_linear_layers)
                .map(|_| vec![0.0f32; state_size].into_boxed_slice())
                .collect(),
            conv_states: (0..num_linear_layers)
                .map(|_| vec![0.0f32; conv_state_size].into_boxed_slice())
                .collect(),
            num_heads,
            key_dim,
            value_dim,
            conv_kernel_size,
            total_qkv_dim,
            hidden_size,
            rms_norm_eps,
        }
    }

    /// Run a single-token DeltaNet forward step for one linear attention layer.
    ///
    /// This performs the complete linear attention layer on CPU:
    /// 1. Input RMSNorm (pre-norm)
    /// 2. Project Q, K, V, A (decay), B (beta), Z (gate)
    /// 3. Conv1d over concatenated QKV (update sliding window, apply)
    /// 4. L2-normalize Q and K per head
    /// 5. Compute decay alpha and update gate beta
    /// 6. Recurrent state update (delta rule)
    /// 7. Read output from state
    /// 8. Gated output: RMSNorm(o) * SiLU(z)
    /// 9. O projection back to hidden_size
    /// 10. Add to residual (hidden += attn_out)
    ///
    /// `layer_idx`: index into the linear-attention-only array (0..17)
    /// `hidden`: input hidden state `[hidden_size]`, modified in-place (residual added)
    /// `weights`: weight struct for this linear attention layer
    pub fn forward_step(
        &mut self,
        layer_idx: usize,
        hidden: &mut [f32],
        weights: &LinearAttentionWeights,
    ) {
        let num_heads = self.num_heads;
        let key_dim = self.key_dim;
        let value_dim = self.value_dim;
        let hidden_size = self.hidden_size;
        let total_qkv = self.total_qkv_dim;
        let q_total = num_heads * key_dim;
        let k_total = num_heads * key_dim;
        let v_total = num_heads * value_dim;

        // --- 1. Input RMSNorm ---
        let mut normed = vec![0.0f32; hidden_size];
        rms_norm_vec(
            &mut normed,
            hidden,
            &weights.input_layernorm_weight,
            hidden_size,
            self.rms_norm_eps,
        );

        // --- 2. Linear projections ---
        let mut q = vec![0.0f32; q_total];
        let mut k = vec![0.0f32; k_total];
        let mut v = vec![0.0f32; v_total];
        let mut a = vec![0.0f32; num_heads];
        let mut b = vec![0.0f32; num_heads];
        let mut z = vec![0.0f32; v_total];

        matvec(&mut q, &weights.q_proj_weight, &normed, q_total, hidden_size);
        matvec(&mut k, &weights.k_proj_weight, &normed, k_total, hidden_size);
        matvec(&mut v, &weights.v_proj_weight, &normed, v_total, hidden_size);
        matvec(&mut a, &weights.a_proj_weight, &normed, num_heads, hidden_size);
        matvec(&mut b, &weights.b_proj_weight, &normed, num_heads, hidden_size);
        matvec(&mut z, &weights.z_proj_weight, &normed, v_total, hidden_size);

        // --- 3. Conv1d over concatenated QKV ---
        let mut qkv_concat = vec![0.0f32; total_qkv];
        qkv_concat[..q_total].copy_from_slice(&q);
        qkv_concat[q_total..q_total + k_total].copy_from_slice(&k);
        qkv_concat[q_total + k_total..].copy_from_slice(&v);

        let conv_out = self.apply_conv1d_step(layer_idx, &qkv_concat, weights);

        // Split conv output back into q, k, v
        q.copy_from_slice(&conv_out[..q_total]);
        k.copy_from_slice(&conv_out[q_total..q_total + k_total]);
        v.copy_from_slice(&conv_out[q_total + k_total..]);

        // --- 4. L2-normalize Q and K per head ---
        for h in 0..num_heads {
            l2_normalize(&mut q[h * key_dim..(h + 1) * key_dim]);
            l2_normalize(&mut k[h * key_dim..(h + 1) * key_dim]);
        }

        // --- 5. Compute decay (alpha) and update gate (beta) per head ---
        let mut alpha = vec![0.0f32; num_heads];
        let mut beta_vals = vec![0.0f32; num_heads];
        for h in 0..num_heads {
            // A_log stores log(A), so A = exp(A_log)
            let big_a = weights.a_log[h].exp();
            // alpha = exp(-A * softplus(a_h + dt_bias_h))
            alpha[h] = (-big_a * softplus(a[h] + weights.dt_bias[h])).exp();
            beta_vals[h] = sigmoid(b[h]);
        }

        // --- 6. Recurrent state update (delta rule) per head ---
        let state = &mut self.states[layer_idx];
        let state_per_head = key_dim * value_dim;
        let mut output_vec = vec![0.0f32; v_total];

        for h in 0..num_heads {
            let s_offset = h * state_per_head;
            let s = &mut state[s_offset..s_offset + state_per_head];
            let k_h = &k[h * key_dim..(h + 1) * key_dim];
            let v_h = &v[h * value_dim..(h + 1) * value_dim];
            let q_h = &q[h * key_dim..(h + 1) * key_dim];

            // Decay old memory: S = alpha * S
            mat_scale(s, alpha[h]);

            // Predict: predicted_v = S^T @ k  (S is [key_dim, value_dim])
            let mut predicted_v = vec![0.0f32; value_dim];
            matvec_transposed(&mut predicted_v, s, k_h, key_dim, value_dim);

            // Delta correction: delta = beta * (v - predicted_v)
            let mut delta = vec![0.0f32; value_dim];
            for j in 0..value_dim {
                delta[j] = beta_vals[h] * (v_h[j] - predicted_v[j]);
            }

            // Update: S += outer(k, delta)
            outer_add(s, k_h, &delta, key_dim, value_dim);

            // Read output: o_h = S^T @ q
            let o_h = &mut output_vec[h * value_dim..(h + 1) * value_dim];
            matvec_transposed(o_h, s, q_h, key_dim, value_dim);
        }

        // --- 7. Gated output: RMSNorm(o) * SiLU(z) ---
        let mut normed_o = vec![0.0f32; v_total];
        rms_norm_vec(
            &mut normed_o,
            &output_vec,
            &weights.norm_weight,
            v_total,
            self.rms_norm_eps,
        );

        // Apply gate norm to z if present, otherwise use z directly
        let z_for_gate = if let Some(ref gate_norm_w) = weights.gate_norm_weight {
            let mut z_normed = vec![0.0f32; v_total];
            rms_norm_vec(&mut z_normed, &z, gate_norm_w, v_total, self.rms_norm_eps);
            z_normed
        } else {
            z
        };

        let mut gated_output = vec![0.0f32; v_total];
        for i in 0..v_total {
            gated_output[i] = normed_o[i] * silu(z_for_gate[i]);
        }

        // --- 8. O projection: [hidden_size, v_total] @ [v_total] -> [hidden_size] ---
        let mut attn_out = vec![0.0f32; hidden_size];
        matvec(
            &mut attn_out,
            &weights.o_proj_weight,
            &gated_output,
            hidden_size,
            v_total,
        );

        // --- 9. Residual connection ---
        for i in 0..hidden_size {
            hidden[i] += attn_out[i];
        }
    }

    /// Apply causal depthwise conv1d for a single step, updating the sliding window.
    ///
    /// Conv1d weight shape: `[total_qkv_dim, 1, kernel_size]` (depthwise).
    /// Conv state layout per layer: `[channel * hist_len + frame_index]`, where
    /// `hist_len = kernel_size - 1` and frames are ordered oldest-to-newest.
    ///
    /// After computing the output, the SiLU activation is applied.
    fn apply_conv1d_step(
        &mut self,
        layer_idx: usize,
        input: &[f32],
        weights: &LinearAttentionWeights,
    ) -> Vec<f32> {
        let dim = self.total_qkv_dim;
        let k_size = self.conv_kernel_size;
        let hist_len = k_size - 1;
        let conv_state = &mut self.conv_states[layer_idx];

        let mut output = vec![0.0f32; dim];

        for ch in 0..dim {
            let mut sum = weights.conv1d_bias[ch];

            // Historical frames (kernel positions 0..hist_len, oldest to newest)
            for kp in 0..hist_len {
                let w = weights.conv1d_weight[ch * k_size + kp];
                sum += w * conv_state[ch * hist_len + kp];
            }
            // Current frame (last kernel position)
            let w = weights.conv1d_weight[ch * k_size + hist_len];
            sum += w * input[ch];

            // SiLU activation after conv
            output[ch] = silu(sum);

            // Shift conv state: drop oldest, append current
            for kp in 0..hist_len - 1 {
                conv_state[ch * hist_len + kp] = conv_state[ch * hist_len + kp + 1];
            }
            conv_state[ch * hist_len + hist_len - 1] = input[ch];
        }

        output
    }

    /// Run DeltaNet forward over a sequence of tokens (prefill).
    ///
    /// For the MVP, this processes tokens sequentially (one at a time).
    /// A chunkwise parallel algorithm can be added later for speed.
    ///
    /// `layer_idx`: index into the linear-attention-only array (0..17)
    /// `hidden_seq`: `[seq_len, hidden_size]` row-major, modified in-place
    /// `weights`: weight struct for this linear attention layer
    /// `seq_len`: number of tokens to process
    pub fn forward_prefill(
        &mut self,
        layer_idx: usize,
        hidden_seq: &mut [f32],
        weights: &LinearAttentionWeights,
        seq_len: usize,
    ) {
        let hidden_size = self.hidden_size;
        for t in 0..seq_len {
            let offset = t * hidden_size;
            let token_hidden = &mut hidden_seq[offset..offset + hidden_size];
            self.forward_step(layer_idx, token_hidden, weights);
        }
    }

    /// Reset all recurrent states and conv states to zero.
    pub fn reset(&mut self) {
        for state in self.states.iter_mut() {
            state.fill(0.0);
        }
        for conv_state in self.conv_states.iter_mut() {
            conv_state.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_basic() {
        assert!((softplus(0.0) - 0.6931).abs() < 1e-3); // ln(2)
        assert!((softplus(1.0) - 1.3133).abs() < 1e-3); // ln(1+e)
        assert!(softplus(25.0) > 24.9); // saturates to x
        assert!(softplus(-25.0) < 0.01); // saturates to 0
    }

    #[test]
    fn sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn l2_normalize_basic() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn matvec_basic() {
        let mat = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let vec_in = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        matvec(&mut out, &mat, &vec_in, 2, 2);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn outer_add_basic() {
        let mut mat = [0.0f32; 4]; // 2x2
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        outer_add(&mut mat, &a, &b, 2, 2);
        assert!((mat[0] - 3.0).abs() < 1e-6); // 1*3
        assert!((mat[1] - 4.0).abs() < 1e-6); // 1*4
        assert!((mat[2] - 6.0).abs() < 1e-6); // 2*3
        assert!((mat[3] - 8.0).abs() < 1e-6); // 2*4
    }

    #[test]
    fn matvec_transposed_basic() {
        // mat = [[1, 2], [3, 4]], vec = [1, 1]
        // mat^T @ vec = [1+3, 2+4] = [4, 6]
        let mat = [1.0f32, 2.0, 3.0, 4.0];
        let vec_in = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        matvec_transposed(&mut out, &mat, &vec_in, 2, 2);
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_unit_weight() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rms_norm_vec(&mut out, &x, &w, 2, 1e-6);
        // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }
}
