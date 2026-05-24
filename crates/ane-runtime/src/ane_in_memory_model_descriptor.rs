use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{extern_class, extern_conformance, msg_send, ClassType};
use objc2_foundation::{NSDictionary, NSData, NSObjectProtocol, NSString};

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEInMemoryModelDescriptor"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub(crate) struct ANEInMemoryModelDescriptor;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANEInMemoryModelDescriptor {}
);

impl ANEInMemoryModelDescriptor {
    pub fn new(
        mil_text: &NSData,
        weights: Option<&NSDictionary<NSString, AnyObject>>,
    ) -> Option<Retained<ANEInMemoryModelDescriptor>> {
        unsafe {
            msg_send![
                Self::class(),
                modelWithMILText: mil_text,
                weights: weights,
                optionsPlist: Option::<&AnyObject>::None
            ]
        }
    }
}
