use crate::PadMode;

#[derive(Clone, Debug)]
pub struct Convolution2dDescriptor {
    pub groups: usize,
    pub pad_mode: PadMode,
}

impl Default for Convolution2dDescriptor {
    fn default() -> Self {
        Self { groups: 1, pad_mode: PadMode::Valid }
    }
}

#[derive(Clone, Debug)]
pub struct ConvolutionTranspose2dDescriptor {
    pub groups: usize,
    pub stride_height: usize,
    pub stride_width: usize,
    pub pad_mode: PadMode,
}

impl Default for ConvolutionTranspose2dDescriptor {
    fn default() -> Self {
        Self { groups: 1, stride_height: 1, stride_width: 1, pad_mode: PadMode::Valid }
    }
}
