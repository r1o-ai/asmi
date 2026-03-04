mod compile;
mod descriptor;
mod ops;
mod tensor;

pub use compile::MIN_SPATIAL_WIDTH;
pub use descriptor::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor};
pub use tensor::Tensor;

use std::collections::HashMap;

use crate::{Op, Shape};

pub(crate) struct GraphOp {
    pub(crate) op: Op,
    pub(crate) output: Tensor,
}

/// A symbolic computation graph for the Apple Neural Engine.
///
/// Build a graph by calling the op methods (which return [`Tensor`] handles),
/// then compile it with [`Graph::compile`].
///
/// ```ignore
/// let mut g = Graph::new();
/// let x   = g.placeholder(Shape::channels(64));
/// let w   = g.constant(&weights, Shape { channels: 64, height: 1, width: 1, batch: 1 });
/// let x   = g.convolution_2d_1x1(x, w, None);
/// let out = g.relu(x);
/// let model = client.compile(&g)?;
/// ```
pub struct Graph {
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) constants: HashMap<usize, (Box<[u8]>, Shape)>,
    pub(crate) ops: Vec<GraphOp>,
    pub(crate) counter: usize,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            constants: HashMap::new(),
            ops: Vec::new(),
            counter: 0,
        }
    }

    pub(crate) fn alloc(&mut self, shape: Shape) -> Tensor {
        let id = self.counter;
        self.counter += 1;
        Tensor { id, shape }
    }

    pub(crate) fn tensor_name(tensor: Tensor) -> String {
        format!("t{}", tensor.id)
    }

    pub(crate) fn op_name(tensor: Tensor) -> String {
        format!("op{}", tensor.id)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
