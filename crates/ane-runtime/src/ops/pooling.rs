use crate::ops::pad_mode::PadMode;

use crate::ops::pool_type::PoolType;

#[derive(Clone)]
pub struct PoolingOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub pool_type: PoolType,
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub stride_width: usize,
    pub stride_height: usize,
    pub pad_mode: PadMode,
    pub pad_top: usize,
    pub pad_bottom: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub global_pooling: bool,
}
