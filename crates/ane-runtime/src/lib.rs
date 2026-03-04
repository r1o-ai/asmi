//! Rust bindings for Apple Neural Engine (ANE) via the private `AppleNeuralEngine.framework`.
//!
//! Provides a symbolic graph builder and the compile -> run lifecycle through
//! `_ANEInMemoryModel`, using IOSurface-backed zero-copy I/O.
//!
//! # Lifecycle
//!
//! ```ignore
//! let mut g = Graph::new();
//! let x   = g.placeholder(Shape::channels(64));
//! let w   = g.constant(&weights, Shape::spatial(64, 1, 1));
//! let out = g.convolution_2d_1x1(x, w, None);
//!
//! let executable = g.compile(NSQualityOfService::Default)?;
//!
//! let input  = TensorData::with_f32(&data, Shape::channels(64));
//! let output = TensorData::new(Shape::channels(64));
//! executable.run(&[&input], &[&output])?;
//! ```

mod ane_in_memory_model;
mod ane_in_memory_model_descriptor;
mod ane_io_surface_object;
mod ane_request;
pub(crate) mod client;
mod error;
pub mod graph;
pub mod io_surface;
mod executable;
pub mod ops;
pub(crate) mod request;
mod tensor_data;

pub use error::Error;
pub use graph::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor, Graph, Tensor, MIN_SPATIAL_WIDTH};
pub use io_surface::IOSurfaceExt;
pub use executable::Executable;
pub use objc2_foundation::NSQualityOfService;
pub use ops::{
    ActivationOp, ActivationMode, ConcatOp, ConstantOp, ConvOp, DeconvOp,
    ElementwiseOp, ElementwiseOpType, FlattenOp, InnerProductOp, InstanceNormOp,
    Op, MatmulOp, PadFillMode, PadMode, PaddingOp, PoolType, PoolingOp, ReductionOp,
    ReductionMode, ReshapeOp, ScalarOp, ScalarOpType, Shape, SliceBySizeOp,
    SoftmaxOp, TransposeOp,
};
pub use tensor_data::{LockedSlice, LockedSliceMut, TensorData};

/// Convert f32 values to IEEE 754 fp16 bytes (2 bytes per element, little-endian).
pub fn f32_to_fp16_bytes(values: &[f32]) -> Box<[u8]> {
    let mut bytes = vec![0u8; values.len() * 2];
    for (index, &value) in values.iter().enumerate() {
        let f16 = ops::weights::f32_to_f16(value);
        bytes[index * 2] = (f16 & 0xFF) as u8;
        bytes[index * 2 + 1] = (f16 >> 8) as u8;
    }
    bytes.into_boxed_slice()
}
