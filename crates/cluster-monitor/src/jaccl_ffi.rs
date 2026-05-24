//! Rust FFI bindings for the JACCL C shim.
//!
//! Provides safe wrappers around the C-level JACCL functions for RDMA
//! group lifecycle, point-to-point transfers, and PD health probing.
//!
//! All functions are gated behind `#[cfg(feature = "jaccl")]`.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

// ---------------------------------------------------------------------------
// Raw extern "C" declarations matching vendor/jaccl/jaccl_shim.h
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn jaccl_is_available() -> bool;
    fn jaccl_pd_budget_probe(device_name: *const c_char) -> c_int;

    fn jaccl_init_mesh(
        rank: c_int,
        world_size: c_int,
        coordinator_ip: *const c_char,
        coordinator_port: c_int,
        devices_json_path: *const c_char,
        timeout_ms: c_int,
    ) -> *mut c_void;

    fn jaccl_group_rank(g: *mut c_void) -> c_int;
    fn jaccl_group_size(g: *mut c_void) -> c_int;
    fn jaccl_group_probe(g: *mut c_void) -> c_int;

    fn jaccl_group_send(
        g: *mut c_void,
        buf: *const c_void,
        len: usize,
        dst: c_int,
        timeout_ms: c_int,
    ) -> c_int;

    fn jaccl_group_recv(
        g: *mut c_void,
        buf: *mut c_void,
        len: usize,
        src: c_int,
        timeout_ms: c_int,
    ) -> c_int;

    fn jaccl_group_free(g: *mut c_void);
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Check whether JACCL (libibverbs via dlopen) is available on this host.
pub fn available() -> bool {
    // The C side already catches all exceptions and returns false on failure.
    unsafe { jaccl_is_available() }
}

/// Probe the PD budget for a named RDMA device.
///
/// Returns:
/// -  `1` — at least one PD can still be allocated
/// -  `0` — PD exhausted
/// - `-1` — error (device not found, libibverbs unavailable, etc.)
pub fn pd_probe(device_name: &str) -> i32 {
    let Ok(c_name) = CString::new(device_name) else {
        return -1;
    };
    unsafe { jaccl_pd_budget_probe(c_name.as_ptr()) }
}

/// Transfer error codes returned by send/recv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferError {
    /// The operation timed out (-1 from C).
    Timeout,
    /// An RDMA-level error occurred (-2 from C).
    RdmaError,
    /// An unknown negative return code.
    Unknown(i32),
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferError::Timeout => write!(f, "JACCL transfer timed out"),
            TransferError::RdmaError => write!(f, "JACCL RDMA error"),
            TransferError::Unknown(code) => write!(f, "JACCL unknown error ({})", code),
        }
    }
}

impl std::error::Error for TransferError {}

fn transfer_result(code: c_int) -> Result<(), TransferError> {
    match code {
        0 => Ok(()),
        -1 => Err(TransferError::Timeout),
        -2 => Err(TransferError::RdmaError),
        other => Err(TransferError::Unknown(other)),
    }
}

/// Safe wrapper around a JACCL MeshGroup handle.
///
/// Owns the opaque `jaccl_group_t` pointer and frees it on drop.
/// Not `Send`/`Sync` because the underlying C++ object is not thread-safe
/// (`PhantomData<*mut ()>` opts out of auto-trait inference).
pub struct JacclGroup {
    handle: *mut c_void,
    _not_send_sync: PhantomData<*mut ()>,
}

impl JacclGroup {
    /// Create a new JACCL mesh group.
    ///
    /// # Arguments
    /// * `rank` — this node's rank (0 = coordinator/source)
    /// * `world_size` — total number of peers
    /// * `coordinator_ip` — IP of rank-0 node
    /// * `coordinator_port` — TCP port for the side channel
    /// * `devices_json_path` — path to the devices JSON file
    /// * `timeout_ms` — handshake timeout in milliseconds
    ///
    /// Returns `None` if initialization fails (timeout, bad config, etc.).
    pub fn new(
        rank: i32,
        world_size: i32,
        coordinator_ip: &str,
        coordinator_port: i32,
        devices_json_path: &str,
        timeout_ms: i32,
    ) -> Option<Self> {
        let c_ip = CString::new(coordinator_ip).ok()?;
        let c_devices = CString::new(devices_json_path).ok()?;

        let handle = unsafe {
            jaccl_init_mesh(
                rank,
                world_size,
                c_ip.as_ptr(),
                coordinator_port,
                c_devices.as_ptr(),
                timeout_ms,
            )
        };

        if handle.is_null() {
            None
        } else {
            Some(JacclGroup {
                handle,
                _not_send_sync: PhantomData,
            })
        }
    }

    /// This node's rank in the group.
    pub fn rank(&self) -> i32 {
        unsafe { jaccl_group_rank(self.handle) }
    }

    /// Total number of peers in the group.
    pub fn size(&self) -> i32 {
        unsafe { jaccl_group_size(self.handle) }
    }

    /// Probe the QP liveness — sends and receives 1 byte to/from peer.
    ///
    /// Returns `true` if the connection is alive, `false` if the QP is stale
    /// (e.g. cable reseated). Caller should re-init on `false`.
    pub fn probe(&self) -> bool {
        unsafe { jaccl_group_probe(self.handle) == 0 }
    }

    /// Send a buffer to a destination rank.
    ///
    /// Returns `Ok(())` on success, or a `TransferError` on timeout/RDMA failure.
    pub fn send(&self, buf: &[u8], dst: i32, timeout_ms: i32) -> Result<(), TransferError> {
        let code = unsafe {
            jaccl_group_send(
                self.handle,
                buf.as_ptr() as *const c_void,
                buf.len(),
                dst,
                timeout_ms,
            )
        };
        transfer_result(code)
    }

    /// Receive into a buffer from a source rank.
    ///
    /// Returns `Ok(())` on success, or a `TransferError` on timeout/RDMA failure.
    pub fn recv(&self, buf: &mut [u8], src: i32, timeout_ms: i32) -> Result<(), TransferError> {
        let code = unsafe {
            jaccl_group_recv(
                self.handle,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                src,
                timeout_ms,
            )
        };
        transfer_result(code)
    }
}

impl Drop for JacclGroup {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { jaccl_group_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_does_not_crash() {
        // Just verify the FFI call doesn't panic or segfault.
        // On machines without RDMA, this returns false.
        let _ = available();
    }

    #[test]
    fn test_pd_probe_nonexistent_device() {
        // A device that doesn't exist should return -1.
        assert_eq!(pd_probe("nonexistent_rdma_device_12345"), -1);
    }
}
