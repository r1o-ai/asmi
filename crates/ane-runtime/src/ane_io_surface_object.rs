use objc2::encode::{Encode, Encoding};
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_conformance, msg_send, ClassType};
use objc2_foundation::NSObjectProtocol;
use objc2_io_surface::IOSurface;

#[repr(transparent)]
#[derive(Clone, Copy)]
struct IOSurfaceCFRef(*const IOSurface);

unsafe impl Encode for IOSurfaceCFRef {
    const ENCODING: Encoding = Encoding::Pointer(&Encoding::Struct("__IOSurface", &[]));
}

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEIOSurfaceObject"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ANEIOSurfaceObject;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANEIOSurfaceObject {}
);

impl ANEIOSurfaceObject {
    pub(crate) fn with_io_surface(surface: &IOSurface) -> Option<Retained<ANEIOSurfaceObject>> {
        let cf_ref = IOSurfaceCFRef(surface as *const IOSurface);
        unsafe { msg_send![Self::class(), objectWithIOSurface: cf_ref] }
    }
}
