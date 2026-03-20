//! IOSurface memory layout probing for RDMA compatibility research.
//!
//! Creates IOSurfaces of various sizes and inspects their backing memory
//! properties to determine RDMA memory registration compatibility.

use std::ptr;

use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};
use serde::Serialize;

use crate::io_surface::IOSurfaceExt;

/// Results from probing a single IOSurface allocation.
#[derive(Debug, Serialize)]
pub struct SurfaceProbe {
    pub label: String,
    pub requested_bytes: usize,
    pub allocated_bytes: usize,
    pub base_address: u64,
    pub page_aligned: bool,
    pub plane_count: usize,
    pub overhead_bytes: i64,
    pub rdma_likely_compatible: bool,
}

/// Probe IOSurface memory layout for a given byte count.
pub fn probe_surface(label: &str, byte_count: usize) -> SurfaceProbe {
    let surface = IOSurface::with_byte_count(byte_count);

    let alloc_size = surface.allocationSize() as usize;
    let plane_count = surface.planeCount();

    // baseAddress() requires a lock to return a valid pointer.
    surface.lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
    let base_addr = surface.baseAddress().as_ptr() as u64;
    surface.unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());

    let page_aligned = base_addr.is_multiple_of(4096);

    SurfaceProbe {
        label: label.to_string(),
        requested_bytes: byte_count,
        allocated_bytes: alloc_size,
        base_address: base_addr,
        page_aligned,
        plane_count,
        overhead_bytes: alloc_size as i64 - byte_count as i64,
        rdma_likely_compatible: page_aligned && plane_count <= 1,
    }
}

/// Run probes for typical activation transfer sizes.
pub fn probe_standard_sizes() -> Vec<SurfaceProbe> {
    let configs = [
        ("gpt2_768x128", 768 * 128 * 4),
        ("qwen08b_1024x128", 1024 * 128 * 4),
        ("qwen27b_3584x128", 3584 * 128 * 4),
        ("qwen35b_4096x128", 4096 * 128 * 4),
        ("small_64x64", 64 * 64 * 4),
        ("large_8192x256", 8192 * 256 * 4),
    ];

    configs
        .iter()
        .map(|(label, size)| probe_surface(label, *size))
        .collect()
}
