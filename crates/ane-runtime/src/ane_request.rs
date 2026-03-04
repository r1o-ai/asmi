use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{extern_class, extern_conformance, msg_send, ClassType, Message};
use objc2_foundation::{NSArray, NSNumber, NSObjectProtocol};

use crate::ane_io_surface_object::ANEIOSurfaceObject;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANERequest"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ANERequest;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANERequest {}
);

impl ANERequest {
    pub fn with_multiple_io(
        input_surfaces: &[&ANEIOSurfaceObject],
        output_surfaces: &[&ANEIOSurfaceObject],
    ) -> Option<Retained<ANERequest>> {
        let zero = NSNumber::new_u32(0);

        let inputs = NSArray::from_retained_slice(
            &input_surfaces
                .iter()
                .map(|s| (*s).retain())
                .collect::<Vec<_>>(),
        );
        let outputs = NSArray::from_retained_slice(
            &output_surfaces
                .iter()
                .map(|s| (*s).retain())
                .collect::<Vec<_>>(),
        );
        let in_indices = NSArray::from_retained_slice(
            &(0..input_surfaces.len() as u32)
                .map(NSNumber::new_u32)
                .collect::<Vec<_>>(),
        );
        let out_indices = NSArray::from_retained_slice(
            &(0..output_surfaces.len() as u32)
                .map(NSNumber::new_u32)
                .collect::<Vec<_>>(),
        );

        unsafe {
            msg_send![Self::class(),
                requestWithInputs: &*inputs,
                inputIndices: &*in_indices,
                outputs: &*outputs,
                outputIndices: &*out_indices,
                weightsBuffer: Option::<&AnyObject>::None,
                perfStats: Option::<&AnyObject>::None,
                procedureIndex: &*zero]
        }
    }
}
