#[derive(Clone)]
pub struct TransposeOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub perm: [usize; 4],
}
