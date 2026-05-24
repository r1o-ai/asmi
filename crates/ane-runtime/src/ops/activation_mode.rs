#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationMode {
    Relu,
    Tanh,
    LeakyRelu { negative_slope: f64 },
    Sigmoid,
    Elu { alpha: f64 },
    Linear { alpha: f64, beta: f64 },
    /// Espresso "SIGMOID_HARD": clamp(alpha * x + beta, 0, 1). Default alpha=0.2, beta=0.5.
    SigmoidHard { alpha: f64, beta: f64 },
    SoftPlus,
    SoftSign,
}
