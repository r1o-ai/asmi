/// Apply RMSNorm to a single token vector `[hidden_size]`.
///
/// Unlike LayerNorm, RMSNorm has no bias and no mean subtraction — it normalizes
/// by the root mean square of the input, then scales by a learned weight vector.
///
/// Qwen 3.5 uses `rms_norm_eps = 1e-6`.
pub fn final_rms_norm(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    hidden_size: usize,
    epsilon: f32,
) {
    let mut sum_sq = 0.0f32;
    for i in 0..hidden_size {
        sum_sq += input[i] * input[i];
    }
    let mean_sq = sum_sq / hidden_size as f32;
    let rstd = 1.0 / (mean_sq + epsilon).sqrt();
    for i in 0..hidden_size {
        output[i] = input[i] * rstd * weight[i];
    }
}

/// Compute logits: `output[v] = dot(weights[v, :], input[:])` for a single token.
///
/// `weights` is `[vocab_size, hidden_size]` row-major. For Qwen 3.5-0.8B with
/// `tie_word_embeddings: true`, this is the same as `embed_tokens.weight`.
///
/// With `vocab_size = 248320` and `hidden_size = 1024`, this is ~248K dot products.
/// A BLAS-backed version (cblas_sgemv) can replace this for production speed.
pub fn compute_logits(
    output: &mut [f32],
    weights: &[f32],
    input: &[f32],
    vocab_size: usize,
    hidden_size: usize,
) {
    for v in 0..vocab_size {
        let mut sum = 0.0f32;
        let row = v * hidden_size;
        for d in 0..hidden_size {
            sum += weights[row + d] * input[d];
        }
        output[v] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight_is_normalization() {
        let input = [3.0f32, 4.0];
        let weight = [1.0f32, 1.0];
        let mut output = [0.0f32; 2];
        final_rms_norm(&mut output, &input, &weight, 2, 1e-6);
        // RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        // output = input / rms = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        let rms = (12.5f32).sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-5);
        assert!((output[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn logits_simple_dot_product() {
        let weights = [1.0f32, 2.0, 3.0, 4.0]; // 2 vocab, 2 hidden
        let input = [1.0f32, 1.0];
        let mut output = [0.0f32; 2];
        compute_logits(&mut output, &weights, &input, 2, 2);
        assert!((output[0] - 3.0).abs() < 1e-6); // 1*1 + 2*1
        assert!((output[1] - 7.0).abs() < 1e-6); // 3*1 + 4*1
    }
}
