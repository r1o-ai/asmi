#[derive(Clone)]
pub struct ConcatOp {
    pub name: String,
    pub bottoms: Box<[String]>,
    pub top: String,
    pub axis: usize,
}
