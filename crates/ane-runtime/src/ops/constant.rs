use crate::ops::weights::WeightBlob;

#[derive(Clone)]
pub struct ConstantOp {
    pub name: String,
    pub top: String,
    pub data: WeightBlob,
}
