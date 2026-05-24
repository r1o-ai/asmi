#[derive(Clone)]
pub struct MatmulOp {
    pub name: String,
    pub bottom_x: String,
    pub bottom_y: String,
    pub top: String,
    pub transpose_x: bool,
    pub transpose_y: bool,
}
