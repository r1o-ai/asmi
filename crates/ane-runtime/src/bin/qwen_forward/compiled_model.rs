use safetensors::SafeTensors;

use crate::config::QwenConfig;
use crate::executables::{self, CompiledExecutables};
use crate::spinner::Spinner;
use crate::weights::{self, ModelWeights};

/// Immutable compiled model: owns config, weights, and ANE executables.
///
/// Created once from safetensors. Use [`Session`](super::session::Session) to
/// run inference with pre-allocated scratch buffers.
pub struct CompiledModel {
    pub config: QwenConfig,
    pub weights: ModelWeights,
    pub executables: CompiledExecutables,
    pub max_sequence_length: usize,
}

impl CompiledModel {
    /// Load weights from safetensor shards and compile all ANE executables.
    pub fn from_safetensors(
        config: QwenConfig,
        shards: &[SafeTensors],
        padded_prompt_length: usize,
        max_sequence_length: usize,
    ) -> Result<Self, ane::Error> {
        let mut spinner = Spinner::new("Loading weights");
        let model_weights = weights::load_weights(shards, &config);

        spinner.update("Compiling ANE executables");
        let executables = executables::compile_all(
            &model_weights.layers, &config,
            padded_prompt_length, max_sequence_length,
        )?;

        spinner.finish("Compiled ANE model");

        Ok(Self {
            config,
            weights: model_weights,
            executables,
            max_sequence_length,
        })
    }
}
