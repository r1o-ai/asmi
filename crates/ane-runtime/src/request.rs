use objc2::rc::Retained;
use objc2_io_surface::IOSurface;

use crate::ane_io_surface_object::ANEIOSurfaceObject;
use crate::ane_request::ANERequest;
use crate::Error;

/// A set of IOSurface buffers bound to a compiled model for a single evaluation.
///
/// Inputs and outputs must match the tensor shapes declared in the [`Network`](crate::Network)
/// that produced the [`Executable`](crate::Executable). Each surface is accessed by the ANE via DMA;
/// no CPU-side copy occurs during evaluation.
pub struct Request {
    pub(crate) inner: Retained<ANERequest>,
}

unsafe impl Send for Request {}
unsafe impl Sync for Request {}

impl Request {
    /// Wrap `inputs` and `outputs` IOSurfaces into an ANE evaluation request.
    ///
    /// Surface order must match the order of input and output blobs in the compiled network.
    pub fn new(inputs: &[&IOSurface], outputs: &[&IOSurface]) -> Result<Self, Error> {
        let input_objs = inputs
            .iter()
            .map(|surface| ANEIOSurfaceObject::with_io_surface(surface))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::SurfaceWrap)?;

        let output_objs = outputs
            .iter()
            .map(|surface| ANEIOSurfaceObject::with_io_surface(surface))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::SurfaceWrap)?;

        let input_refs: Vec<&ANEIOSurfaceObject> = input_objs.iter().map(|object| &**object).collect();
        let output_refs: Vec<&ANEIOSurfaceObject> = output_objs.iter().map(|object| &**object).collect();

        let inner = ANERequest::with_multiple_io(&input_refs, &output_refs)
            .ok_or(Error::RequestCreation)?;

        Ok(Self { inner })
    }
}
