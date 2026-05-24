//! Experimental ANE compute bridge via private Apple APIs.
//!
//! # EXPERIMENTAL
//!
//! This module uses private, undocumented APIs from Apple's
//! `AppleNeuralEngine.framework`. These APIs can break on any macOS
//! update without warning. This module is only compiled when the `ane`
//! Cargo feature is enabled.
//!
//! # Safety
//!
//! All functions in this module are unsafe because they call into
//! Objective-C code that uses private Apple frameworks via dlopen/objc_msgSend.

#[cfg(feature = "ane")]
mod ffi {
    extern "C" {
        pub fn ane_bridge_init() -> i32;
        pub fn ane_bridge_available() -> i32;
        pub fn ane_bridge_compile(
            mil_text: *const u8,
            mil_len: usize,
            weight_data: *const u8,
            weight_len: usize,
            input_sizes: *const usize,
            n_inputs: i32,
            output_sizes: *const usize,
            n_outputs: i32,
        ) -> *mut std::ffi::c_void;
        pub fn ane_bridge_write_input(
            handle: *mut std::ffi::c_void,
            idx: i32,
            data: *const u8,
            bytes: usize,
        );
        pub fn ane_bridge_eval(handle: *mut std::ffi::c_void) -> i32;
        pub fn ane_bridge_read_output(
            handle: *mut std::ffi::c_void,
            idx: i32,
            data: *mut u8,
            bytes: usize,
        );
        pub fn ane_bridge_free(handle: *mut std::ffi::c_void);
    }
}

/// Check if ANE direct compute is available.
///
/// Returns `false` if compiled without the `ane` feature, or if the
/// private framework fails to load at runtime.
pub fn ane_available() -> bool {
    #[cfg(feature = "ane")]
    {
        unsafe { ffi::ane_bridge_init() == 1 }
    }
    #[cfg(not(feature = "ane"))]
    {
        false
    }
}

/// Initialize the ANE bridge. Must be called before any other ane_bridge functions.
/// Returns true if successful.
pub fn init() -> bool {
    #[cfg(feature = "ane")]
    {
        unsafe { ffi::ane_bridge_init() == 1 }
    }
    #[cfg(not(feature = "ane"))]
    {
        false
    }
}