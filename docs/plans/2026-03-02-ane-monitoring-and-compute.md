# ANE Monitoring + Experimental Compute Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add ANE hardware monitoring (power, execution time) to asmi's metrics pipeline, and provide an experimental ANE compute endpoint behind a feature flag — making asmi the first tool to expose both ANE telemetry and direct ANE dispatch as a network service.

**Architecture:** Two-path design. **Path A** (ANE Monitoring) integrates IOReport-based sudoless ANE power sampling into the poll loop and adds `_ANEPerformanceStats` execution time probing via a thin Objective-C bridge. **Path B** (ANE Compute) wraps the `maderix/ANE` runtime in a Rust FFI bridge behind `--experimental-ane` CLI flag and Cargo feature flag `ane`, exposing `POST /ane/eval` and `GET /ane/status` endpoints. Path B compiles to a no-op when the feature is disabled.

**Tech Stack:** Rust (asmi binary + asmi-core), Objective-C (thin FFI bridge for private APIs), IOReport C API (ANE power), IOSurface (ANE I/O), Cargo feature flags, `cc` build crate.

---

## Path A: ANE Monitoring (Production)

### Background

Currently asmi gets ANE power from `powermetrics` which requires `sudo`. The `macmon` project proved you can read ANE power from the IOReport `"Energy Model"` channel group **without sudo** using the IOReport C API. Channel names are `"ANE"` (base chips), `"ANE0"` (Max), `"ANE0_0"`/`"ANE0_1"` (Ultra).

Additionally, `_ANEPerformanceStats.hwExecutionTime` (private API) can report actual ANE execution time per evaluation — asmi could expose this for inference processes using ANE via Core ML.

### Task A1: Add IOReport FFI bindings for ANE power

**Files:**
- Create: `crates/cluster-monitor/src/ioreport.rs`
- Modify: `crates/cluster-monitor/src/lib.rs` — add `pub mod ioreport;`
- Modify: `crates/cluster-monitor/Cargo.toml` — add `core-foundation` and `core-foundation-sys` deps

**Step 1: Add dependencies**

In `crates/cluster-monitor/Cargo.toml`, add:
```toml
core-foundation = "0.10"
core-foundation-sys = "0.8"
```

**Step 2: Write the IOReport FFI module**

Create `crates/cluster-monitor/src/ioreport.rs`:

```rust
//! Sudoless ANE power monitoring via IOReport.
//!
//! Uses the same IOReport "Energy Model" channel group that macmon and
//! powermetrics use, but without requiring root privileges.
//!
//! # Safety
//!
//! This module uses `unsafe` FFI calls to IOReport (a private-but-stable
//! Apple framework). IOReport has been stable across macOS 13-26.

use core_foundation::base::{CFType, CFTypeRef, TCFType};
use core_foundation::dictionary::{CFDictionary, CFMutableDictionary};
use core_foundation::string::CFString;
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// IOReport FFI declarations
// ---------------------------------------------------------------------------

type IOReportSubscriptionRef = *mut c_void;

#[link(name = "IOReport", kind = "dylib")]
extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: core_foundation_sys::string::CFStringRef,
        subgroup: core_foundation_sys::string::CFStringRef,
        a: u64,
        b: u64,
        c: u64,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportCreateSubscription(
        a: *const c_void,
        channels: core_foundation_sys::dictionary::CFMutableDictionaryRef,
        sub_channels: *mut core_foundation_sys::dictionary::CFMutableDictionaryRef,
        d: u64,
        e: CFTypeRef,
    ) -> IOReportSubscriptionRef;

    fn IOReportCreateSamples(
        sub: IOReportSubscriptionRef,
        channels: core_foundation_sys::dictionary::CFMutableDictionaryRef,
        c: CFTypeRef,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportCreateSamplesDelta(
        prev: core_foundation_sys::dictionary::CFDictionaryRef,
        curr: core_foundation_sys::dictionary::CFDictionaryRef,
        c: CFTypeRef,
    ) -> core_foundation_sys::dictionary::CFDictionaryRef;

    fn IOReportMergeChannels(
        a: core_foundation_sys::dictionary::CFDictionaryRef,
        b: core_foundation_sys::dictionary::CFDictionaryRef,
        c: CFTypeRef,
    );

    fn IOReportChannelGetGroup(item: core_foundation_sys::dictionary::CFDictionaryRef)
        -> core_foundation_sys::string::CFStringRef;

    fn IOReportChannelGetChannelName(item: core_foundation_sys::dictionary::CFDictionaryRef)
        -> core_foundation_sys::string::CFStringRef;

    fn IOReportChannelGetUnitLabel(item: core_foundation_sys::dictionary::CFDictionaryRef)
        -> core_foundation_sys::string::CFStringRef;

    fn IOReportSimpleGetIntegerValue(
        item: core_foundation_sys::dictionary::CFDictionaryRef,
        a: i32,
    ) -> i64;
}

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

/// A subscription to IOReport energy channels.
/// Create once, call `sample()` repeatedly to get delta-based power readings.
pub struct EnergySubscription {
    sub: IOReportSubscriptionRef,
    channels: CFMutableDictionary,
    prev_sample: Option<CFDictionary>,
}

// IOReportSubscriptionRef is a pointer — not Send by default.
// IOReport subscriptions are safe to move between threads (no thread-local state).
unsafe impl Send for EnergySubscription {}

/// Sampled power readings in milliwatts.
#[derive(Debug, Clone, Default)]
pub struct PowerSample {
    pub cpu_mw: f64,
    pub gpu_mw: f64,
    pub ane_mw: f64,
    pub package_mw: f64,
}

impl EnergySubscription {
    /// Create a new subscription to Energy Model channels.
    /// Returns `None` if IOReport is unavailable (shouldn't happen on macOS).
    pub fn new() -> Option<Self> {
        let energy_group = CFString::new("Energy Model");

        let channels = unsafe {
            let ch = IOReportCopyChannelsInGroup(
                energy_group.as_concrete_TypeRef(),
                std::ptr::null(),
                0, 0, 0,
            );
            if ch.is_null() {
                return None;
            }
            CFMutableDictionary::wrap_under_create_rule(
                ch as core_foundation_sys::dictionary::CFMutableDictionaryRef,
            )
        };

        let mut sub_channels = std::ptr::null_mut();
        let sub = unsafe {
            IOReportCreateSubscription(
                std::ptr::null(),
                channels.as_concrete_TypeRef(),
                &mut sub_channels,
                0,
                std::ptr::null(),
            )
        };

        if sub.is_null() {
            return None;
        }

        Some(Self {
            sub,
            channels,
            prev_sample: None,
        })
    }

    /// Take a sample and compute delta power in mW.
    /// First call returns zeros (needs two samples for a delta).
    pub fn sample(&mut self, dt_secs: f64) -> PowerSample {
        let current = unsafe {
            let raw = IOReportCreateSamples(
                self.sub,
                self.channels.as_concrete_TypeRef(),
                std::ptr::null(),
            );
            if raw.is_null() {
                return PowerSample::default();
            }
            CFDictionary::wrap_under_create_rule(raw)
        };

        let result = if let Some(ref prev) = self.prev_sample {
            let delta = unsafe {
                let raw = IOReportCreateSamplesDelta(
                    prev.as_concrete_TypeRef(),
                    current.as_concrete_TypeRef(),
                    std::ptr::null(),
                );
                if raw.is_null() {
                    return PowerSample::default();
                }
                CFDictionary::wrap_under_create_rule(raw)
            };
            Self::extract_power(&delta, dt_secs)
        } else {
            PowerSample::default()
        };

        self.prev_sample = Some(current);
        result
    }

    /// Walk the delta dictionary and extract power values by channel name.
    fn extract_power(delta: &CFDictionary, dt_secs: f64) -> PowerSample {
        // IOReport dictionaries have an "IOReportChannels" key containing a CFArray.
        // Each element is a channel dictionary with group, name, unit, and value.
        //
        // For Energy Model channels, the value is energy in nJ (nanojoules).
        // Power (mW) = energy (nJ) / dt (ns) = energy_nj / (dt_secs * 1e9) * 1e3

        let mut sample = PowerSample::default();

        let channels_key = CFString::new("IOReportChannels");
        if let Some(channels_ref) = delta.find(channels_key.as_CFType().as_CFTypeRef()) {
            let channels_array: core_foundation::array::CFArray =
                unsafe { CFType::wrap_under_get_rule(*channels_ref as CFTypeRef) };

            for i in 0..channels_array.len() {
                let item_ref = channels_array.get(i);
                if item_ref.is_none() { continue; }
                let item = item_ref.unwrap() as core_foundation_sys::dictionary::CFDictionaryRef;

                let name = unsafe {
                    let name_ref = IOReportChannelGetChannelName(item);
                    if name_ref.is_null() { continue; }
                    CFString::wrap_under_get_rule(name_ref).to_string()
                };

                let energy_nj = unsafe {
                    IOReportSimpleGetIntegerValue(item, 0)
                } as f64;

                let mw = if dt_secs > 0.0 {
                    energy_nj / (dt_secs * 1e6) // nJ / ms = mW
                } else {
                    0.0
                };

                match name.as_str() {
                    n if n.starts_with("CPU") => sample.cpu_mw += mw,
                    n if n.starts_with("GPU") => sample.gpu_mw += mw,
                    n if n.starts_with("ANE") => sample.ane_mw += mw,
                    n if n.contains("package") || n.contains("Package") => sample.package_mw += mw,
                    _ => {}
                }
            }
        }

        sample
    }
}
```

> **Note to implementer:** The IOReport FFI above is a starting point based on macmon's approach. The exact `CFDictionary` traversal for channel items may need adjustment — IOReport's internal structure isn't documented. Test empirically on a real Mac and iterate. The `extract_power` function walks the delta dict; the exact key name for the channels array (`"IOReportChannels"`) should be verified by printing the dict contents.

**Step 3: Register the module**

In `crates/cluster-monitor/src/lib.rs`, add:
```rust
pub mod ioreport;
```

**Step 4: Write a smoke test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_subscription_creates() {
        // Should succeed on macOS, fail gracefully elsewhere
        let sub = EnergySubscription::new();
        if cfg!(target_os = "macos") {
            assert!(sub.is_some(), "IOReport should be available on macOS");
        }
    }

    #[test]
    fn test_power_sample_default() {
        let s = PowerSample::default();
        assert_eq!(s.cpu_mw, 0.0);
        assert_eq!(s.gpu_mw, 0.0);
        assert_eq!(s.ane_mw, 0.0);
    }
}
```

**Step 5: Build and test**

```bash
cd /Users/ma/Projects/personal/apple-smi && cargo build -p asmi-core 2>&1
cargo test -p asmi-core ioreport 2>&1
```

**Step 6: Commit**

```bash
git add crates/cluster-monitor/src/ioreport.rs crates/cluster-monitor/src/lib.rs crates/cluster-monitor/Cargo.toml
git commit -m "feat(core): add IOReport FFI for sudoless ANE power monitoring"
```

---

### Task A2: Integrate IOReport into the daemon poll loop

**Files:**
- Modify: `src/daemon_startup.rs` — create `EnergySubscription` at startup, sample each tick
- Modify: `crates/cluster-monitor/src/types.rs` — add `ane_power_source` field to NodeSnapshot

**Step 1: Add power source field**

In `crates/cluster-monitor/src/types.rs`, after `pub ane_watts: f64`:
```rust
    /// Source of ANE power reading: "powermetrics" (sudo) or "ioreport" (sudoless).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_source: Option<String>,
```

**Step 2: Create subscription in daemon_startup.rs**

In `run_serve()`, after `let snapshot = ...`:
```rust
    // Sudoless power monitoring via IOReport (supplements powermetrics)
    let energy_sub = asmi_core::ioreport::EnergySubscription::new();
    if energy_sub.is_some() {
        tracing::info!("IOReport energy subscription active (sudoless ANE power)");
    } else {
        tracing::warn!("IOReport unavailable — ANE power requires sudo powermetrics");
    }
    let energy_sub = Arc::new(tokio::sync::Mutex::new(energy_sub));
```

**Step 3: Sample in the poll loop**

Inside the `tokio::spawn` poll loop, after `collect_node_metrics`:
```rust
                // Supplement with IOReport power if available (sudoless, more accurate)
                {
                    let mut sub_guard = energy_sub.lock().await;
                    if let Some(ref mut sub) = *sub_guard {
                        let power = sub.sample(interval as f64);
                        if power.ane_mw > 0.0 || power.cpu_mw > 0.0 {
                            snap.cpu_watts = power.cpu_mw;
                            snap.gpu_watts = power.gpu_mw;
                            snap.ane_watts = power.ane_mw;
                            snap.power_source = Some("ioreport".to_string());
                        }
                    }
                }
```

**Step 4: Pass energy_sub into the poll loop closure**

Add `let energy_sub = Arc::clone(&energy_sub);` in the clone block before `tokio::spawn`.

**Step 5: Build and verify**

```bash
cargo build --release 2>&1
```

**Step 6: Manual test**

```bash
./target/release/asmi --serve &
sleep 3
curl -s localhost:9090/metrics | jq '{ane_watts, power_source}'
kill %1
```

Expected: `power_source: "ioreport"` with a non-zero `ane_watts` if ANE is active.

**Step 7: Commit**

```bash
git add src/daemon_startup.rs crates/cluster-monitor/src/types.rs
git commit -m "feat: integrate IOReport for sudoless ANE power in daemon poll loop"
```

---

### Task A3: Add GET /ane endpoint for ANE-specific metrics

**Files:**
- Modify: `src/daemon.rs` — add `ane_handler`, wire route

**Step 1: Add handler**

Before `pub fn build_router`:
```rust
/// GET /ane — ANE-specific metrics (power, frequency, utilization).
async fn ane_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => Ok(Json(serde_json::json!({
            "hostname": s.hostname,
            "ane_watts": s.ane_watts / 1000.0,  // mW -> W
            "ane_mw": s.ane_watts,
            "power_source": s.power_source,
            "power_gated": s.ane_watts < 1.0,   // ANE has hard power gating (0mW when idle)
        }))),
        None => Err(ApiError::NotFound("no data yet".into())),
    }
}
```

**Step 2: Wire route**

In `build_router`, add:
```rust
        .route("/ane", get(ane_handler))
```

**Step 3: Update startup banner**

In `daemon_startup.rs`, add to the `eprintln!` block:
```rust
    eprintln!("  GET  /ane               ANE power + status");
```

**Step 4: Build, test, commit**

```bash
cargo build 2>&1
git add src/daemon.rs src/daemon_startup.rs
git commit -m "feat: add GET /ane endpoint for ANE-specific metrics"
```

---

## Path B: ANE Compute Server (Experimental, Feature-Flagged)

### Background

The `maderix/ANE` project (`github.com/maderix/ANE`) reverse-engineers Apple's private `AppleNeuralEngine.framework` to submit compute directly to the ANE hardware. It uses `_ANEInMemoryModel` to compile MIL (Model Intermediate Language) text into ANE programs at runtime, then dispatches evaluation via IOSurface shared memory.

**This is experimental.** The private APIs can break on any macOS update without warning. All Path B code is gated behind:
- Cargo feature: `ane` (disabled by default)
- CLI flag: `--experimental-ane`
- Module: `src/ane.rs` (compiles to empty stubs without feature)

### Task B1: Add `ane` Cargo feature flag

**Files:**
- Modify: `Cargo.toml` — add feature flag + optional dep
- Modify: `crates/cluster-monitor/Cargo.toml` — add `cc` build dep

**Step 1: Add feature in root Cargo.toml**

```toml
[features]
default = []
ane = []  # Experimental: direct ANE compute via private APIs

[build-dependencies]
cc = { version = "1.2", optional = true }
```

Under `[dependencies]`, add:
```toml
cc = { version = "1.2", optional = true }
```

**Step 2: Commit**

```bash
git add Cargo.toml
git commit -m "feat: add 'ane' feature flag for experimental ANE compute"
```

---

### Task B2: Create the Objective-C FFI bridge

**Files:**
- Create: `crates/cluster-monitor/src/ane_bridge/mod.rs` — Rust FFI declarations
- Create: `crates/cluster-monitor/src/ane_bridge/bridge.m` — Thin ObjC wrapper (~150 lines)
- Create: `crates/cluster-monitor/build.rs` — compile ObjC with `cc`

**Step 1: Create the Objective-C bridge**

Create `crates/cluster-monitor/src/ane_bridge/bridge.m`:

```objc
// ANE FFI Bridge — Thin wrapper around Apple's private ANE APIs.
//
// EXPERIMENTAL: Uses _ANEInMemoryModel and related private classes from
// AppleNeuralEngine.framework. These APIs are undocumented and can break
// on any macOS update.
//
// License: MIT (same as asmi)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/runtime.h>
#import <objc/message.h>

// ---------------------------------------------------------------------------
// Runtime class references (resolved lazily)
// ---------------------------------------------------------------------------

static Class cls_InMemoryModelDescriptor = nil;
static Class cls_InMemoryModel = nil;
static Class cls_IOSurfaceObject = nil;
static Class cls_Request = nil;
static Class cls_PerfStats = nil;

static bool ane_classes_loaded = false;

// ---------------------------------------------------------------------------
// Public C API (called from Rust via FFI)
// ---------------------------------------------------------------------------

/// Initialize: load the private framework and resolve classes.
/// Returns 1 on success, 0 if ANE framework is unavailable.
int ane_bridge_init(void) {
    if (ane_classes_loaded) return 1;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    );
    if (!handle) return 0;

    cls_InMemoryModelDescriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    cls_InMemoryModel = NSClassFromString(@"_ANEInMemoryModel");
    cls_IOSurfaceObject = NSClassFromString(@"_ANEIOSurfaceObject");
    cls_Request = NSClassFromString(@"_ANERequest");
    cls_PerfStats = NSClassFromString(@"_ANEPerformanceStats");

    if (!cls_InMemoryModel || !cls_InMemoryModelDescriptor) return 0;

    ane_classes_loaded = true;
    return 1;
}

/// Check if ANE bridge is available.
int ane_bridge_available(void) {
    return ane_classes_loaded ? 1 : 0;
}

/// Opaque handle to a compiled ANE kernel.
typedef struct {
    id model;           // _ANEInMemoryModel
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int n_inputs;
    int n_outputs;
} ANEKernelHandle;

/// Create an IOSurface of the given byte size.
static IOSurfaceRef create_surface(size_t bytes) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfacePixelFormat: @0x00000000,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

/// Compile a MIL program with baked weights into an ANE kernel.
/// Returns an opaque handle, or NULL on failure.
///
/// - mil_text: UTF-8 MIL program text
/// - mil_len: byte length of mil_text
/// - weight_data: binary weight blob (fp16 with headers)
/// - weight_len: byte length of weight_data
/// - input_sizes: array of byte sizes for each input IOSurface
/// - n_inputs: number of inputs
/// - output_sizes: array of byte sizes for each output IOSurface
/// - n_outputs: number of outputs
void *ane_bridge_compile(
    const char *mil_text, size_t mil_len,
    const uint8_t *weight_data, size_t weight_len,
    const size_t *input_sizes, int n_inputs,
    const size_t *output_sizes, int n_outputs
) {
    if (!ane_classes_loaded) return NULL;

    @autoreleasepool {
        NSString *mil = [[NSString alloc] initWithBytes:mil_text
                                                 length:mil_len
                                               encoding:NSUTF8StringEncoding];
        NSData *weights = [NSData dataWithBytes:weight_data length:weight_len];

        // Create model descriptor
        id descriptor = ((id (*)(id, SEL, id, id, id))objc_msgSend)(
            (id)cls_InMemoryModelDescriptor,
            NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
            mil, weights, nil
        );
        if (!descriptor) return NULL;

        // Create in-memory model
        id model = ((id (*)(id, SEL, id))objc_msgSend)(
            (id)cls_InMemoryModel,
            NSSelectorFromString(@"inMemoryModelWithDescriptor:"),
            descriptor
        );
        if (!model) return NULL;

        // Compile
        NSError *error = nil;
        BOOL ok = ((BOOL (*)(id, SEL, int, id, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"compileWithQoS:options:error:"),
            0, nil, &error
        );
        if (!ok) return NULL;

        // Load
        ok = ((BOOL (*)(id, SEL, int, id, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"loadWithQoS:options:error:"),
            0, nil, &error
        );
        if (!ok) return NULL;

        // Allocate handle
        ANEKernelHandle *h = calloc(1, sizeof(ANEKernelHandle));
        h->model = (__bridge_retained id)model;
        h->n_inputs = n_inputs;
        h->n_outputs = n_outputs;

        // Create IOSurfaces
        h->inputs = calloc(n_inputs, sizeof(IOSurfaceRef));
        h->outputs = calloc(n_outputs, sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++) {
            h->inputs[i] = create_surface(input_sizes[i]);
        }
        for (int i = 0; i < n_outputs; i++) {
            h->outputs[i] = create_surface(output_sizes[i]);
        }

        return h;
    }
}

/// Write data to an input IOSurface.
void ane_bridge_write_input(void *handle, int idx, const void *data, size_t bytes) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h || idx < 0 || idx >= h->n_inputs) return;

    IOSurfaceLock(h->inputs[idx], 0, NULL);
    void *base = IOSurfaceGetBaseAddress(h->inputs[idx]);
    memcpy(base, data, bytes);
    IOSurfaceUnlock(h->inputs[idx], 0, NULL);
}

/// Evaluate the compiled ANE kernel. Returns 1 on success, 0 on failure.
int ane_bridge_eval(void *handle) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h) return 0;

    @autoreleasepool {
        // Wrap IOSurfaces as _ANEIOSurfaceObject
        NSMutableArray *inputObjs = [NSMutableArray arrayWithCapacity:h->n_inputs];
        NSMutableArray *outputObjs = [NSMutableArray arrayWithCapacity:h->n_outputs];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:h->n_inputs];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:h->n_outputs];

        for (int i = 0; i < h->n_inputs; i++) {
            id obj = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                (id)cls_IOSurfaceObject,
                NSSelectorFromString(@"objectWithIOSurface:"),
                h->inputs[i]
            );
            [inputObjs addObject:obj];
            [inputIndices addObject:@(i)];
        }
        for (int i = 0; i < h->n_outputs; i++) {
            id obj = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
                (id)cls_IOSurfaceObject,
                NSSelectorFromString(@"objectWithIOSurface:"),
                h->outputs[i]
            );
            [outputObjs addObject:obj];
            [outputIndices addObject:@(i)];
        }

        // Build request
        id request = ((id (*)(id, SEL, id, id, id, id, id, id, int))objc_msgSend)(
            (id)cls_Request,
            NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            inputObjs, inputIndices, outputObjs, outputIndices, nil, nil, 0
        );
        if (!request) return 0;

        // Evaluate
        NSError *error = nil;
        BOOL ok = ((BOOL (*)(id, SEL, int, id, id, NSError **))objc_msgSend)(
            (__bridge id)h->model,
            NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
            0, nil, request, &error
        );
        return ok ? 1 : 0;
    }
}

/// Read data from an output IOSurface.
void ane_bridge_read_output(void *handle, int idx, void *data, size_t bytes) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h || idx < 0 || idx >= h->n_outputs) return;

    IOSurfaceLock(h->outputs[idx], kIOSurfaceLockReadOnly, NULL);
    void *base = IOSurfaceGetBaseAddress(h->outputs[idx]);
    memcpy(data, base, bytes);
    IOSurfaceUnlock(h->outputs[idx], kIOSurfaceLockReadOnly, NULL);
}

/// Free a compiled ANE kernel and all its IOSurfaces.
void ane_bridge_free(void *handle) {
    ANEKernelHandle *h = (ANEKernelHandle *)handle;
    if (!h) return;

    @autoreleasepool {
        // Unload model
        id model = (__bridge_transfer id)h->model;
        ((void (*)(id, SEL, int, NSError **))objc_msgSend)(
            model,
            NSSelectorFromString(@"unloadWithQoS:error:"),
            0, nil
        );

        for (int i = 0; i < h->n_inputs; i++) {
            if (h->inputs[i]) CFRelease(h->inputs[i]);
        }
        for (int i = 0; i < h->n_outputs; i++) {
            if (h->outputs[i]) CFRelease(h->outputs[i]);
        }
        free(h->inputs);
        free(h->outputs);
        free(h);
    }
}
```

**Step 2: Create the Rust FFI module**

Create `crates/cluster-monitor/src/ane_bridge/mod.rs`:

```rust
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
```

**Step 3: Create build.rs**

Create `crates/cluster-monitor/build.rs`:

```rust
fn main() {
    #[cfg(feature = "ane")]
    {
        let bridge_path = std::path::Path::new("src/ane_bridge/bridge.m");
        if bridge_path.exists() {
            cc::Build::new()
                .file(bridge_path)
                .flag("-fobjc-arc")
                .flag("-framework").flag("Foundation")
                .flag("-framework").flag("IOSurface")
                .flag("-ldl")
                .compile("ane_bridge");

            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=IOSurface");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rerun-if-changed=src/ane_bridge/bridge.m");
        }
    }
}
```

**Step 4: Register in lib.rs**

In `crates/cluster-monitor/src/lib.rs`:
```rust
pub mod ane_bridge;
```

**Step 5: Update cluster-monitor Cargo.toml**

```toml
[features]
default = []
ane = ["dep:cc"]

[build-dependencies]
cc = { version = "1.2", optional = true }
```

And in the root `Cargo.toml`, update the asmi-core dependency:
```toml
asmi-core = { path = "crates/cluster-monitor", features = [] }
```

And the root features:
```toml
[features]
default = []
ane = ["asmi-core/ane"]
```

**Step 6: Build both variants**

```bash
# Without ane feature (default — should compile clean)
cargo build 2>&1

# With ane feature (compiles bridge.m)
cargo build --features ane 2>&1
```

**Step 7: Commit**

```bash
git add crates/cluster-monitor/src/ane_bridge/ crates/cluster-monitor/build.rs \
        crates/cluster-monitor/Cargo.toml Cargo.toml
git commit -m "feat(experimental): add ANE compute bridge behind 'ane' feature flag

Uses private AppleNeuralEngine.framework APIs via Objective-C FFI.
Compile with: cargo build --features ane
EXPERIMENTAL: private APIs can break on any macOS update."
```

---

### Task B3: Add `--experimental-ane` CLI flag and ANE daemon endpoints

**Files:**
- Modify: `src/main.rs` — add `--experimental-ane` flag
- Create: `src/ane.rs` — ANE state manager + endpoints
- Modify: `src/daemon.rs` — conditionally wire ANE routes
- Modify: `src/daemon_startup.rs` — init ANE bridge on startup if flagged

**Step 1: Add CLI flag**

In `src/main.rs`, add to the `Cli` struct:
```rust
    /// Enable experimental ANE compute endpoints (requires --features ane at build time).
    #[arg(long, hide = true)]
    experimental_ane: bool,
```

Pass it to `run_serve`:
```rust
    return daemon_startup::run_serve(
        args.port, args.interval, args.cluster,
        args.models_dir, args.experimental_ane,
    ).await;
```

**Step 2: Create src/ane.rs**

```rust
//! Experimental ANE compute server endpoints.
//!
//! # EXPERIMENTAL
//!
//! Gated behind `--experimental-ane` CLI flag AND `ane` Cargo feature.
//! Uses private Apple APIs that can break without warning.

use axum::{extract::State, response::Json, routing::{get, post}};
use serde::Deserialize;

use crate::daemon::{ApiError, AppState};

/// ANE subsystem status.
#[derive(Clone)]
pub struct AneState {
    pub enabled: bool,
    pub available: bool,
    pub compile_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl AneState {
    /// Create a new ANE state. Attempts to init the bridge if enabled.
    pub fn new(enabled: bool) -> Self {
        let available = if enabled {
            asmi_core::ane_bridge::init()
        } else {
            false
        };
        Self {
            enabled,
            available,
            compile_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    pub fn noop() -> Self {
        Self {
            enabled: false,
            available: false,
            compile_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }
}

/// GET /ane/status — ANE compute subsystem status.
pub async fn ane_status_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ane = &state.ane;
    Json(serde_json::json!({
        "experimental": true,
        "enabled": ane.enabled,
        "available": ane.available,
        "compile_count": ane.compile_count.load(std::sync::atomic::Ordering::Relaxed),
        "compile_limit_warning": "ANE compiler leaks ~119 compiles per process. Daemon restart required after limit.",
        "private_api_warning": "Uses undocumented Apple APIs. Can break on any macOS update.",
    }))
}

/// POST /ane/eval — placeholder for MIL evaluation.
/// Full implementation requires MIL text + weight blob in request body.
pub async fn ane_eval_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !state.ane.available {
        return Err(ApiError::BadRequest(
            "ANE compute not available. Build with --features ane and start with --experimental-ane".into()
        ));
    }

    // TODO: Accept MIL text + weight blob, compile, eval, return output.
    // This is the scaffold — real implementation in a future PR.
    Err(ApiError::BadRequest(
        "ANE eval endpoint is scaffolded but not yet implemented. See ROADMAP.md".into()
    ))
}
```

**Step 3: Wire into AppState and router**

In `src/daemon.rs`, add to `AppState`:
```rust
    pub ane: crate::ane::AneState,
```

In `build_router`, conditionally add routes:
```rust
        // Experimental ANE compute (feature-flagged)
        .route("/ane/status", get(crate::ane::ane_status_handler))
        .route("/ane/eval", post(crate::ane::ane_eval_handler))
```

**Step 4: Init in daemon_startup.rs**

Update `run_serve` signature:
```rust
pub async fn run_serve(
    port: u16, interval: u64, cluster_hub: bool,
    cli_models_dir: Vec<String>, experimental_ane: bool,
) -> Result<()> {
```

Before building `app_state`:
```rust
    let ane_state = if experimental_ane {
        tracing::warn!("EXPERIMENTAL: ANE compute endpoints enabled (--experimental-ane)");
        crate::ane::AneState::new(true)
    } else {
        crate::ane::AneState::noop()
    };
```

Add to `app_state`:
```rust
        ane: ane_state,
```

**Step 5: Register module**

In `src/main.rs`, add:
```rust
mod ane;
```

**Step 6: Update startup banner**

In `daemon_startup.rs`, add conditionally:
```rust
    if experimental_ane {
        eprintln!("  \x1b[33m[EXPERIMENTAL]\x1b[0m ANE compute endpoints enabled");
        eprintln!("  GET  /ane/status        ANE compute subsystem status");
        eprintln!("  POST /ane/eval          Submit MIL program to ANE (not yet implemented)");
    }
```

**Step 7: Build both variants and test**

```bash
# Default build (ane disabled) — must compile clean
cargo build 2>&1

# With feature flag
cargo build --features ane 2>&1
```

**Step 8: Commit**

```bash
git add src/ane.rs src/main.rs src/daemon.rs src/daemon_startup.rs
git commit -m "feat(experimental): scaffold ANE compute endpoints behind --experimental-ane

- GET /ane/status returns subsystem availability
- POST /ane/eval scaffolded (not yet implemented)
- Requires: cargo build --features ane && asmi --serve --experimental-ane
- EXPERIMENTAL: uses private Apple APIs, can break on any macOS update"
```

---

### Task B4: Update ROADMAP.md

**Files:**
- Modify: `ROADMAP.md`

**Step 1: Add v0.6 ANE section**

After the v0.5 section, add:
```markdown
## v0.6 — ANE Integration (in progress)

- [x] **ANE power (sudoless)** — IOReport `"Energy Model"` channel for ANE power without sudo
- [x] **GET /ane** — dedicated ANE metrics endpoint (power, power-gated status)
- [ ] **ANE execution time** — `_ANEPerformanceStats.hwExecutionTime` for per-eval timing
- [ ] **ANE compute (experimental)** — `POST /ane/eval` for direct ANE dispatch via private APIs
  - Feature flag: `cargo build --features ane`
  - CLI flag: `--experimental-ane`
  - Uses `_ANEInMemoryModel` private API — can break on any macOS update
```

Update the v0.9 section to reference the ANE bridge:
```markdown
- [ ] **ANE utilization** — extend experimental ANE bridge with per-eval performance counters
```

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: update ROADMAP with v0.6 ANE integration plan"
```

---

## Summary

| Task | Type | Risk | What it does |
|---|---|---|---|
| A1 | Production | Low | IOReport FFI for sudoless ANE power |
| A2 | Production | Low | Integrate into daemon poll loop |
| A3 | Production | Low | `GET /ane` endpoint |
| B1 | Experimental | None | Feature flag scaffolding |
| B2 | Experimental | Medium | ObjC FFI bridge to private ANE APIs |
| B3 | Experimental | Medium | CLI flag + endpoint scaffolding |
| B4 | Docs | None | ROADMAP update |

**Total new endpoints:** 3 (`/ane`, `/ane/status`, `/ane/eval`)
**Total new files:** 4 (`ioreport.rs`, `ane_bridge/mod.rs`, `ane_bridge/bridge.m`, `ane.rs`)
**Feature flags:** `ane` Cargo feature + `--experimental-ane` CLI flag
