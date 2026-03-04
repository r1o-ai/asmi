use objc2::rc::Retained;
use objc2_foundation::NSQualityOfService;

use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::request::Request;
use crate::tensor_data::TensorData;
use crate::Error;

/// A compiled, loaded ANE program ready for repeated evaluation.
///
/// Obtained from [`Graph::compile`](crate::Graph::compile).
/// Automatically unloads from ANE hardware on drop.
pub struct Executable {
    pub(crate) inner: Retained<ANEInMemoryModel>,
    pub(crate) qos: NSQualityOfService,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Executable {
    /// Run the compiled program on the ANE.
    ///
    /// `inputs` and `outputs` are positional [`TensorData`] arrays matching the
    /// order of [`placeholder`](crate::Graph::placeholder) calls and output tensors
    /// in the graph.
    pub fn run(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<(), Error> {
        let input_surfaces: Vec<&objc2_io_surface::IOSurface> =
            inputs.iter().map(|tensor_data| tensor_data.surface()).collect();
        let output_surfaces: Vec<&objc2_io_surface::IOSurface> =
            outputs.iter().map(|tensor_data| tensor_data.surface()).collect();
        let request = Request::new(&input_surfaces, &output_surfaces)?;
        self.inner
            .evaluate(self.qos, &request.inner)
            .map_err(|error| Error::Evaluate(error.localizedDescription().to_string()))
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        self.inner.unload(self.qos);
    }
}
