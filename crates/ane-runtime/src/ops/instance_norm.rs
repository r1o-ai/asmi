use crate::ops::weights::WeightBlob;

#[derive(Clone)]
pub struct InstanceNormOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub channels: usize,
    pub epsilon: f64,
    pub params: WeightBlob,
}
