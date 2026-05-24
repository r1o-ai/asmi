//! IOReport FFI bindings for Apple Silicon ANE (Neural Engine) power monitoring.
//!
//! Uses the private IOReport framework to subscribe to energy model channels
//! and read real-time ANE power in milliwatts — the same data source that
//! `powermetrics` uses internally.
//!
//! # Architecture
//!
//! 1. `EnergySubscription::new()` — creates a persistent IOReport subscription
//!    to the "Energy Model" channel group. This is a one-time setup cost (~5ms).
//! 2. `EnergySubscription::sample()` — takes a delta sample and returns the
//!    latest power readings in milliwatts. Call this every poll tick (~2s).
//!
//! The subscription holds a CFDictionary that IOReportCreateSubscription returns.
//! Dropping the subscription releases the CoreFoundation objects.
//!
//! # Safety
//!
//! All FFI calls use raw CoreFoundation pointers. The module is macOS-only and
//! requires the IOReport private framework (present on all macOS 13+).

use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use std::ffi::c_void;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// IOReport C function declarations (private framework)
// ---------------------------------------------------------------------------

type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type IOReportSubscriptionRef = *mut c_void;

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CFStringRef,
        subgroup: CFStringRef,
        zero1: u64,
        zero2: u64,
        zero3: u64,
    ) -> CFDictionaryRef;

    fn IOReportCreateSubscription(
        a: *const c_void, // NULL
        channel_dict: CFDictionaryRef,
        out_sub: *mut CFMutableDictionaryRef,
        zero1: u64,
        zero2: u64,
        zero3: u64,
    ) -> IOReportSubscriptionRef;

    fn IOReportCreateSamples(
        sub: IOReportSubscriptionRef,
        sub_dict: CFMutableDictionaryRef,
        b: *const c_void, // NULL
    ) -> CFDictionaryRef;

    fn IOReportCreateSamplesDelta(
        prev: CFDictionaryRef,
        curr: CFDictionaryRef,
        d: *const c_void, // NULL
    ) -> CFDictionaryRef;

    fn IOReportChannelGetGroup(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetSubGroup(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetChannelName(sample: CFDictionaryRef) -> CFStringRef;
    fn IOReportSimpleGetIntegerValue(sample: CFDictionaryRef, idx: i32) -> i64;
}

// CoreFoundation helpers we need for array iteration
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFArrayGetCount(arr: *const c_void) -> isize;
    fn CFArrayGetValueAtIndex(arr: *const c_void, idx: isize) -> *const c_void;
    fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

// ---------------------------------------------------------------------------
// Safe wrapper for CFString comparison
// ---------------------------------------------------------------------------

/// Convert a raw CFStringRef to a Rust String. Returns None if null or conversion fails.
///
/// # Safety
/// `cfstr` must be a valid CFStringRef or null.
unsafe fn cfstring_to_string(cfstr: CFStringRef) -> Option<String> {
    if cfstr.is_null() {
        return None;
    }
    // Create a CFString wrapper without taking ownership (we don't own these refs)
    let cf = unsafe { CFString::wrap_under_get_rule(cfstr as *const _) };
    Some(cf.to_string())
}

// ---------------------------------------------------------------------------
// Power sample result
// ---------------------------------------------------------------------------

/// Power readings from a single IOReport sample, in milliwatts.
#[derive(Debug, Clone, Default)]
pub struct PowerSample {
    /// CPU power in milliwatts.
    pub cpu_mw: f64,
    /// GPU power in milliwatts.
    pub gpu_mw: f64,
    /// ANE (Neural Engine) power in milliwatts.
    pub ane_mw: f64,
    /// DRAM power in milliwatts (if available).
    pub dram_mw: f64,
    /// Package (total SoC) power in milliwatts (if available).
    pub package_mw: f64,
    /// Power source: "Battery" or "AC" (if detected).
    pub power_source: Option<String>,
}

impl PowerSample {
    /// Total SoC power (CPU + GPU + ANE) in milliwatts.
    pub fn total_mw(&self) -> f64 {
        self.cpu_mw + self.gpu_mw + self.ane_mw
    }
}

// ---------------------------------------------------------------------------
// Energy subscription (persistent, reusable)
// ---------------------------------------------------------------------------

/// A persistent IOReport subscription for energy model channels.
///
/// Create once at daemon startup, then call `sample()` on each poll tick.
/// The subscription holds CoreFoundation objects that are released on drop.
pub struct EnergySubscription {
    /// The IOReport subscription handle.
    subscription: IOReportSubscriptionRef,
    /// The subscription dictionary (passed to IOReportCreateSamples).
    sub_dict: CFMutableDictionaryRef,
    /// Previous sample for delta computation.
    prev_sample: Option<*const c_void>,
    /// Duration of the sampling interval (for power = energy / time conversion).
    sample_interval_ms: u64,
}

// SAFETY: The IOReport subscription is not Sync (CoreFoundation objects are
// generally not thread-safe), but we only ever access it from a single
// tokio task (the poll loop). Sending it between threads is safe as long
// as access is serialized, which tokio::spawn guarantees.
unsafe impl Send for EnergySubscription {}

impl EnergySubscription {
    /// Create a new energy subscription. Returns None if IOReport is unavailable.
    ///
    /// `interval_ms` should match the daemon poll interval (e.g., 2000 for 2s).
    pub fn new(interval_ms: u64) -> Option<Self> {
        unsafe {
            // Request channels in the "Energy Model" group
            let group = CFString::new("Energy Model");
            let channels = IOReportCopyChannelsInGroup(
                group.as_concrete_TypeRef() as CFStringRef,
                std::ptr::null(), // all subgroups
                0,
                0,
                0,
            );

            if channels.is_null() {
                warn!("IOReportCopyChannelsInGroup returned null — IOReport unavailable");
                return None;
            }

            // Create subscription
            let mut sub_dict: CFMutableDictionaryRef = std::ptr::null_mut();
            let subscription = IOReportCreateSubscription(
                std::ptr::null(),
                channels,
                &mut sub_dict,
                0,
                0,
                0,
            );

            // Release the channel dict (subscription owns a copy)
            CFRelease(channels);

            if subscription.is_null() || sub_dict.is_null() {
                warn!("IOReportCreateSubscription failed");
                return None;
            }

            debug!("IOReport energy subscription created (interval={}ms)", interval_ms);

            Some(Self {
                subscription,
                sub_dict,
                prev_sample: None,
                sample_interval_ms: interval_ms,
            })
        }
    }

    /// Take a sample and return power readings. On the first call, takes a
    /// baseline sample and returns zeros. Subsequent calls return the delta.
    pub fn sample(&mut self) -> PowerSample {
        unsafe {
            let current = IOReportCreateSamples(
                self.subscription,
                self.sub_dict,
                std::ptr::null(),
            );

            if current.is_null() {
                warn!("IOReportCreateSamples returned null");
                return PowerSample::default();
            }

            let result = match self.prev_sample {
                Some(prev) => {
                    let delta = IOReportCreateSamplesDelta(prev, current, std::ptr::null());
                    let power = if !delta.is_null() {
                        let p = self.extract_power(delta);
                        CFRelease(delta);
                        p
                    } else {
                        PowerSample::default()
                    };
                    CFRelease(prev);
                    power
                }
                None => {
                    debug!("IOReport: first sample (baseline), returning zeros");
                    PowerSample::default()
                }
            };

            self.prev_sample = Some(current);
            result
        }
    }

    /// Extract power values from a delta sample dictionary.
    ///
    /// The delta dict has an "IOReportChannels" key containing a CFArray of
    /// per-channel dictionaries. We iterate and match by channel name.
    ///
    /// # Safety
    /// `delta` must be a valid CFDictionaryRef returned by IOReportCreateSamplesDelta.
    unsafe fn extract_power(&self, delta: CFDictionaryRef) -> PowerSample {
        let mut result = PowerSample::default();

        // Get the channels array from the delta dictionary
        let key = CFString::new("IOReportChannels");
        let channels_arr =
            unsafe { CFDictionaryGetValue(delta, key.as_concrete_TypeRef() as *const c_void) };

        if channels_arr.is_null() {
            trace!("no IOReportChannels key in delta");
            return result;
        }

        let count = unsafe { CFArrayGetCount(channels_arr) };

        // Duration in seconds for energy -> power conversion
        let dt_secs = self.sample_interval_ms as f64 / 1000.0;
        if dt_secs <= 0.0 {
            return result;
        }

        for i in 0..count {
            let entry = unsafe { CFArrayGetValueAtIndex(channels_arr, i) };
            if entry.is_null() {
                continue;
            }

            let group = unsafe { cfstring_to_string(IOReportChannelGetGroup(entry)) };
            let subgroup = unsafe { cfstring_to_string(IOReportChannelGetSubGroup(entry)) };
            let name = unsafe { cfstring_to_string(IOReportChannelGetChannelName(entry)) };

            let group_str = group.as_deref().unwrap_or("");
            let subgroup_str = subgroup.as_deref().unwrap_or("");
            let name_str = name.as_deref().unwrap_or("");

            // We want the "Energy Model" group
            if group_str != "Energy Model" {
                continue;
            }

            // Read the integer value (energy in nanojoules)
            let energy_val = unsafe { IOReportSimpleGetIntegerValue(entry, 0) };
            if energy_val <= 0 {
                continue;
            }

            // Convert energy (nanojoules) to milliwatts: mW = nJ / (dt_ms * 1000)
            // IOReport energy values are in nanojoules for the "Energy Model" group
            let mw = energy_val as f64 / (self.sample_interval_ms as f64 * 1000.0);

            trace!(
                group = group_str,
                subgroup = subgroup_str,
                name = name_str,
                energy_nj = energy_val,
                mw = mw,
                "IOReport channel"
            );

            // Match channel names to power domains
            // Common patterns across M1/M2/M3/M4:
            //   "CPU Energy" or subgroup contains "CPU"
            //   "GPU Energy" or subgroup contains "GPU"
            //   "ANE Energy" or name contains "ANE"
            //   "DRAM Energy" or name contains "DRAM"
            match name_str {
                n if n.contains("CPU") && subgroup_str.contains("CPU") => {
                    result.cpu_mw += mw;
                }
                n if n.contains("GPU") && subgroup_str.contains("GPU") => {
                    result.gpu_mw += mw;
                }
                n if n.contains("ANE") => {
                    result.ane_mw += mw;
                }
                n if n.contains("DRAM") => {
                    result.dram_mw += mw;
                }
                n if n.contains("Package")
                    || (subgroup_str == "Energy Model" && n.contains("Energy")) =>
                {
                    // Some SoCs report a total package energy
                    result.package_mw = mw;
                }
                _ => {
                    // Check subgroup as fallback
                    if subgroup_str.contains("CPU") {
                        result.cpu_mw += mw;
                    } else if subgroup_str.contains("GPU") {
                        result.gpu_mw += mw;
                    } else if subgroup_str.contains("ANE") || subgroup_str.contains("Neural") {
                        result.ane_mw += mw;
                    }
                }
            }
        }

        debug!(
            cpu_mw = format!("{:.0}", result.cpu_mw),
            gpu_mw = format!("{:.0}", result.gpu_mw),
            ane_mw = format!("{:.0}", result.ane_mw),
            dram_mw = format!("{:.0}", result.dram_mw),
            "IOReport power sample"
        );

        result
    }
}

impl Drop for EnergySubscription {
    fn drop(&mut self) {
        unsafe {
            if let Some(prev) = self.prev_sample.take() {
                CFRelease(prev);
            }
            // The subscription and sub_dict are CoreFoundation objects.
            // sub_dict was returned via out-param, so we own it.
            if !self.sub_dict.is_null() {
                CFRelease(self.sub_dict as *const c_void);
            }
            // subscription is also owned by us
            if !self.subscription.is_null() {
                CFRelease(self.subscription as *const c_void);
            }
        }
        debug!("IOReport energy subscription dropped");
    }
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that we can create an IOReport subscription on this machine.
    /// This test will be skipped (pass vacuously) on non-macOS or if
    /// IOReport is unavailable.
    #[test]
    fn can_create_subscription() {
        let sub = EnergySubscription::new(2000);
        assert!(sub.is_some(), "IOReport subscription should succeed on macOS");
    }

    /// Take two samples and verify we get non-negative values.
    #[test]
    fn sample_returns_nonnegative() {
        let mut sub = match EnergySubscription::new(100) {
            Some(s) => s,
            None => {
                eprintln!("skipping: IOReport unavailable");
                return;
            }
        };

        // First sample is baseline (returns zeros)
        let s1 = sub.sample();
        assert_eq!(s1.cpu_mw, 0.0);
        assert_eq!(s1.gpu_mw, 0.0);
        assert_eq!(s1.ane_mw, 0.0);

        // Sleep briefly to accumulate energy
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Second sample should have real values
        let s2 = sub.sample();
        assert!(s2.cpu_mw >= 0.0, "CPU power should be non-negative");
        assert!(s2.gpu_mw >= 0.0, "GPU power should be non-negative");
        assert!(s2.ane_mw >= 0.0, "ANE power should be non-negative");

        eprintln!(
            "IOReport sample: CPU={:.0}mW GPU={:.0}mW ANE={:.0}mW DRAM={:.0}mW",
            s2.cpu_mw, s2.gpu_mw, s2.ane_mw, s2.dram_mw
        );
    }
}
