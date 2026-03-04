use crate::ops::activation_mode::ActivationMode;

#[derive(Clone)]
pub struct ActivationOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub mode: ActivationMode,
}
