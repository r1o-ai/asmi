#[derive(Clone)]
pub struct ReshapeOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub target_shape: [usize; 4],
}
