/// Apply LayerNorm to a single token vector `[embedding_dim]`.
pub fn final_layer_norm(
    output: &mut [f32],
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    embedding_dim: usize,
    epsilon: f32,
) {
    let mut mean = 0.0f32;
    for dim_index in 0..embedding_dim {
        mean += input[dim_index];
    }
    mean /= embedding_dim as f32;

    let mut variance = 0.0f32;
    for dim_index in 0..embedding_dim {
        let centered = input[dim_index] - mean;
        variance += centered * centered;
    }
    variance /= embedding_dim as f32;

    let rstd = 1.0 / (variance + epsilon).sqrt();
    for dim_index in 0..embedding_dim {
        output[dim_index] = (input[dim_index] - mean) * rstd * gamma[dim_index] + beta[dim_index];
    }
}

/// Compute logits: `output += weights @ input` for a single token.
///
/// `weights` is `[vocab_size, embedding_dim]` row-major (tied wte).
pub fn compute_logits(
    output: &mut [f32],
    weights: &[f32],
    input: &[f32],
    vocab_size: usize,
    embedding_dim: usize,
) {
    for vocab_index in 0..vocab_size {
        let mut sum = 0.0f32;
        for dim_index in 0..embedding_dim {
            sum += weights[vocab_index * embedding_dim + dim_index] * input[dim_index];
        }
        output[vocab_index] += sum;
    }
}
