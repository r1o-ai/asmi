use crate::Shape;

/// A symbolic handle to a tensor in a [`super::Graph`].
///
/// `Tensor` is `Copy` — pass it by value and shadow-rebind freely:
///
/// ```ignore
/// let x = g.placeholder(Shape::channels(64));
/// let x = g.relu(x);                      // shadows previous x
/// let x = g.convolution_2d_1x1(x, w, None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub(crate) id: usize,
    /// The shape of this tensor in NCHW order.
    pub shape: Shape,
}
