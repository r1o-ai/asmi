use crate::ops::pad_fill_mode::PadFillMode;

#[derive(Clone)]
pub struct PaddingOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub pad_top: usize,
    pub pad_bottom: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub pad_fill_mode: PadFillMode,
    pub pad_value: f64,
}
