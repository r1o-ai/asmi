#[derive(Clone)]
pub struct SliceBySizeOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub begin: [usize; 4],
    pub size: [usize; 4],
}
