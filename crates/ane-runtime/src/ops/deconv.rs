use crate::ops::pad_mode::PadMode;
use crate::ops::weights::WeightBlob;

#[derive(Clone)]
pub struct DeconvOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub stride_width: usize,
    pub stride_height: usize,
    pub groups: usize,
    pub pad_mode: PadMode,
    pub pad_top: usize,
    pub pad_bottom: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub output_padding_width: usize,
    pub output_padding_height: usize,
    pub weights: WeightBlob,
    pub bias: Option<WeightBlob>,
    pub fused_relu: bool,
    pub fused_tanh: bool,
}
