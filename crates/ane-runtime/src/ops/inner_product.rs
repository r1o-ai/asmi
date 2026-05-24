use crate::ops::weights::WeightBlob;

#[derive(Clone)]
pub struct InnerProductOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub weights: WeightBlob,
    pub bias: Option<WeightBlob>,
    pub has_relu: bool,
    pub has_tanh: bool,
}
