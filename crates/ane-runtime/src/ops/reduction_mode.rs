#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionMode {
    Sum = 0,
    Mean = 1,
    Min = 2,
    Max = 3,
}
