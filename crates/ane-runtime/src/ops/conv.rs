use crate::ops::weights::WeightBlob;

use crate::ops::pad_mode::PadMode;

#[derive(Clone)]
pub struct ConvOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub groups: usize,
    pub pad_mode: PadMode,
    pub pad_top: usize,
    pub pad_bottom: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub weights: WeightBlob,
    pub bias: Option<WeightBlob>,
    pub fused_relu: bool,
    pub fused_tanh: bool,
}
