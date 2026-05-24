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

    /// Pre-allocated IOSurfaces for MoE expert weights: [layer_idx][expert_idx]
    pub moe_expert_tensors: Option<Box<[Box<[MoeExpertTensors]>]>>,
}

pub struct MoeExpertTensors {
    pub gate_w: ane::TensorData,
    pub up_w: ane::TensorData,
    pub down_w: ane::TensorData,
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

        let mut moe_tensors: Option<Vec<Box<[MoeExpertTensors]>>> = None;
        if config.num_experts.unwrap_or(0) > 0 {
            spinner.update("Allocating IOSurfaces for MoE Experts");
            let hidden = config.hidden_size;
            let intermediate = config.moe_intermediate_size.unwrap();
            let mut layers_tensors = Vec::with_capacity(config.num_hidden_layers);

            for layer in model_weights.layers.iter() {
                use crate::weights::{LayerWeights, FfnVariant};
                let ffn = match layer {
                    LayerWeights::FullAttention(w) => &w.ffn,
                    LayerWeights::LinearAttention(w) => &w.ffn,
                };
                match ffn {
                    FfnVariant::Moe(moe) => {
                        let mut experts_tensors = Vec::with_capacity(moe.experts.len());
                        for expert in &moe.experts {
                            let gate_t = ane::TensorData::with_f32(&expert.gate_proj_weight, ane::Shape { batch: 1, channels: 1, height: intermediate, width: hidden });
                            let up_t = ane::TensorData::with_f32(&expert.up_proj_weight, ane::Shape { batch: 1, channels: 1, height: intermediate, width: hidden });
                            let down_t = ane::TensorData::with_f32(&expert.down_proj_weight, ane::Shape { batch: 1, channels: 1, height: hidden, width: intermediate });
                            experts_tensors.push(MoeExpertTensors {
                                gate_w: gate_t,
                                up_w: up_t,
                                down_w: down_t,
                            });
                        }
                        layers_tensors.push(experts_tensors.into_boxed_slice());
                    }
                    FfnVariant::Dense(_) => {
                        // MoE models might have dense layers (e.g. layer 0 is dense in Qwen2MoE sometimes).
                        // In that case, we just push an empty slice to keep indices aligned.
                        layers_tensors.push(Vec::new().into_boxed_slice());
                    }
                }
            }
            moe_tensors = Some(layers_tensors);
        }

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
            moe_expert_tensors: moe_tensors.map(|v| v.into_boxed_slice()),
        })
    }
}
