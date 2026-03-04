use crate::ops::reduction_mode::ReductionMode;

#[derive(Clone)]
pub struct ReductionOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub mode: ReductionMode,
    pub axis: i64,
}
