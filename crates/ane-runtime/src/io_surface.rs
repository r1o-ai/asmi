use std::ptr;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::AnyThread;
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use objc2_io_surface::{
    IOSurface, IOSurfaceLockOptions, IOSurfacePropertyKeyBytesPerElement,
    IOSurfacePropertyKeyBytesPerRow, IOSurfacePropertyKeyHeight, IOSurfacePropertyKeyWidth,
};

/// Extension methods for [`IOSurface`] covering the flat byte-buffer layout used by ANE tensors.
pub trait IOSurfaceExt {
    /// Allocate a flat byte-buffer IOSurface with `byte_count` bytes.
    ///
    /// The surface is laid out as `width = byte_count`, `height = 1`,
    /// `bytes_per_element = 1`. Because MIL function signatures use `tensor<fp32, ...>` for
    /// I/O, pass `element_count * 4` bytes (fp32). The ANE casts to fp16 internally.
    fn with_byte_count(byte_count: usize) -> Retained<IOSurface>;

    /// Copy `data` into the surface under a write lock.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` exceeds the surface's allocated size.
    fn write_bytes(&self, data: &[u8]);

    /// Copy bytes out of the surface into `buf` under a read-only lock.
    ///
    /// # Panics
    ///
    /// Panics if `buf.len()` exceeds the surface's allocated size.
    fn read_bytes(&self, buf: &mut [u8]);
}

impl IOSurfaceExt for IOSurface {
    fn with_byte_count(byte_count: usize) -> Retained<IOSurface> {
        let dict: Retained<NSDictionary<NSString, AnyObject>> = unsafe {
            NSDictionary::from_slices(
                &[
                    IOSurfacePropertyKeyWidth,
                    IOSurfacePropertyKeyHeight,
                    IOSurfacePropertyKeyBytesPerElement,
                    IOSurfacePropertyKeyBytesPerRow,
                ],
                &[
                    &NSNumber::new_usize(byte_count) as &AnyObject,
                    &NSNumber::new_usize(1) as &AnyObject,
                    &NSNumber::new_usize(1) as &AnyObject,
                    &NSNumber::new_usize(byte_count) as &AnyObject,
                ],
            )
        };
        IOSurface::initWithProperties(IOSurface::alloc(), &dict)
            .expect("IOSurface creation failed")
    }

    fn write_bytes(&self, data: &[u8]) {
        assert!(
            data.len() <= self.allocationSize() as usize,
            "data ({} bytes) exceeds surface allocation ({} bytes)",
            data.len(),
            self.allocationSize(),
        );
        unsafe {
            self.lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
            let dst = self.baseAddress().as_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            self.unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        }
    }

    fn read_bytes(&self, buf: &mut [u8]) {
        assert!(
            buf.len() <= self.allocationSize() as usize,
            "buf ({} bytes) exceeds surface allocation ({} bytes)",
            buf.len(),
            self.allocationSize(),
        );
        unsafe {
            self.lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
            let src = self.baseAddress().as_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), buf.len());
            self.unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        }
    }
}
