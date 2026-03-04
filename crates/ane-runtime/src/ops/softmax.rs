#[derive(Clone)]
pub struct SoftmaxOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub axis: i64,
}
