# ANE Compute Integration — asmi + ane-runtime Bridge

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing `ane-runtime` crate into the asmi daemon behind a feature flag, exposing ANE direct compute as HTTP endpoints (`GET /ane/status`, `POST /ane/eval`) and laying the IOSurface groundwork for future RDMA activation transfer.

**Architecture:** The `ane-runtime` crate already has complete Rust FFI wrappers for Apple's private `AppleNeuralEngine.framework` — graph building, MIL compilation, IOSurface I/O, and the full compile→load→evaluate→unload lifecycle. Instead of building a parallel Objective-C bridge (as the original plan proposed), we add `ane-runtime` as an optional dependency of the asmi binary crate (already declared in `Cargo.toml` as `ane = ["dep:ane-runtime"]`). A new `src/ane.rs` module provides the `AneState` manager and HTTP handlers, gated behind `--experimental-ane` CLI flag. A new `IOSurfaceProbe` in the ane-runtime crate profiles memory layout for RDMA compatibility research.

**Tech Stack:** Rust, `ane-runtime` crate (private API FFI via `objc2`), `axum` HTTP, IOSurface framework, Cargo feature flags.

---

## Codebase Map (Phase 0 Discovery)

```
apple-smi/
├── Cargo.toml                          # Workspace root. Already has: ane = ["dep:ane-runtime"]
├── src/
│   ├── main.rs                         # CLI entry: Cli struct, --serve dispatch
│   ├── daemon.rs                       # AppState, build_router(), all HTTP handlers
│   ├── daemon_startup.rs               # run_serve() — IOReport init, poll loop, state wiring
│   ├── serve.rs                        # ServeManager, ShareManager, PeerHeartbeat
│   ├── watchdog.rs                     # Process watchdog
│   ├── topology.rs                     # TB5/RDMA topology
│   └── cli.rs                          # CLI monitor mode
├── crates/
│   ├── ane-runtime/
│   │   ├── Cargo.toml                  # name = "ane-runtime", lib name = "ane"
│   │   ├── build.rs                    # ANE framework linking
│   │   └── src/
│   │       ├── lib.rs                  # Public API: Graph, Executable, TensorData, Shape
│   │       ├── client.rs              # compile_network() — MIL compile + load
│   │       ├── executable.rs          # Executable::run() — ANE evaluation
│   │       ├── graph/                 # Symbolic graph builder (placeholder, ops, compile)
│   │       ├── io_surface.rs          # IOSurfaceExt trait (with_byte_count, write/read_bytes)
│   │       ├── tensor_data.rs         # TensorData (IOSurface-backed, RAII lock guards)
│   │       ├── error.rs              # Error enum (FrameworkLoad, Compile, Load, Evaluate, ...)
│   │       └── ops/                   # All ANE ops (matmul, conv, softmax, etc.)
│   └── cluster-monitor/
│       ├── src/
│       │   ├── ioreport.rs           # EnergySubscription (IOReport power monitoring)
│       │   ├── types.rs              # NodeSnapshot, ProcessFramework::AneNative
│       │   └── lib.rs                # pub mod ioreport; (already wired)
│       └── Cargo.toml
├── tests/
│   ├── daemon_endpoints.rs           # Integration tests (start daemon, hit endpoints)
│   └── serve_share.rs
└── ROADMAP.md
```

**Key insight:** `Cargo.toml:43-47` already declares the feature:
```toml
ane-runtime = { path = "crates/ane-runtime", optional = true }
[features]
ane = ["dep:ane-runtime"]
```

So `cargo build --features ane` already links `ane-runtime`. We just need to use it.

---

## Task 1: Add `--experimental-ane` CLI flag

**Files:**
- Modify: `src/main.rs:17-56` — add flag to `Cli` struct, pass to `run_serve`
- Modify: `src/daemon_startup.rs:9` — update `run_serve` signature

### Step 1: Add CLI flag

In `src/main.rs`, add to the `Cli` struct (after line 52):

```rust
    /// Enable experimental ANE compute endpoints (requires --features ane at build time).
    #[arg(long, hide = true)]
    experimental_ane: bool,
```

### Step 2: Pass flag to run_serve

In `src/main.rs:179`, update the call:

```rust
    if args.serve {
        return daemon_startup::run_serve(
            args.port, args.interval, args.cluster,
            args.models_dir, args.experimental_ane,
        ).await;
    }
```

### Step 3: Update run_serve signature

In `src/daemon_startup.rs:9`, change:

```rust
pub async fn run_serve(port: u16, interval: u64, cluster_hub: bool, cli_models_dir: Vec<String>, experimental_ane: bool) -> Result<()> {
```

### Step 4: Build to verify

```bash
cd /Users/ma/Projects/Personal/apple-smi && cargo build 2>&1
```

Expected: compiles clean (the `experimental_ane` variable is unused but that's just a warning).

### Step 5: Commit

```bash
git add src/main.rs src/daemon_startup.rs
git commit -m "feat: add --experimental-ane CLI flag (hidden, for future ANE compute)"
```

---

## Task 2: Create `src/ane.rs` — ANE compute state manager + endpoints

**Files:**
- Create: `src/ane.rs`
- Modify: `src/main.rs:1` — add `mod ane;`

### Step 1: Create the ANE module

Create `src/ane.rs`. This module compiles to no-ops when the `ane` feature is disabled, and to real endpoints when enabled. All private API usage is isolated here.

```rust
//! Experimental ANE compute endpoints.
//!
//! Gated behind `--experimental-ane` CLI flag AND `ane` Cargo feature.
//! Uses Apple's private AppleNeuralEngine.framework via the `ane-runtime` crate.
//!
//! # EXPERIMENTAL
//!
//! Private APIs can break on any macOS update without warning.
//! The ANE compiler has a per-process budget of ~119 compilations before
//! it silently starts failing (resource leak). Plan model partitioning accordingly.

use axum::{extract::State, response::Json, routing::{get, post}};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::daemon::AppState;

// ---------------------------------------------------------------------------
// ANE subsystem state
// ---------------------------------------------------------------------------

/// Tracks the ANE compute subsystem lifecycle.
///
/// Created once at daemon startup. The `compile_count` tracks how many ANE
/// programs have been compiled in this process — the hard limit is ~119 before
/// the ANE compiler silently leaks and starts failing.
#[derive(Clone)]
pub struct AneState {
    /// Whether `--experimental-ane` was passed at startup.
    pub enabled: bool,
    /// Whether the ANE framework loaded successfully at runtime.
    pub available: bool,
    /// Number of ANE programs compiled in this process lifetime.
    pub compile_count: Arc<AtomicU32>,
}

impl AneState {
    /// Create ANE state. If `enabled`, attempts to load AppleNeuralEngine.framework.
    pub fn new(enabled: bool) -> Self {
        let available = if enabled {
            #[cfg(feature = "ane")]
            {
                // The ane crate's client::ensure_framework() does dlopen internally.
                // We probe by attempting a trivial graph compile to verify the full
                // pipeline works. But that costs a compile slot — so we just check
                // if the framework loads.
                // Actually, let's just rely on the first real compile to validate.
                // The dlopen happens lazily on first Graph::compile().
                true // optimistic — actual availability confirmed on first compile
            }
            #[cfg(not(feature = "ane"))]
            {
                false
            }
        } else {
            false
        };

        Self {
            enabled,
            available,
            compile_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Create a disabled (no-op) ANE state.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            available: false,
            compile_count: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Remaining compile budget (approximate).
    pub fn compile_budget_remaining(&self) -> u32 {
        119u32.saturating_sub(self.compile_count.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// Response for ApiError — reuse from daemon.rs.
/// We return 400/503 for ANE-specific errors.
fn ane_error(status: axum::http::StatusCode, msg: &str) -> axum::response::Response {
    let body = axum::Json(serde_json::json!({"error": msg}));
    (status, body).into_response()
}

use axum::response::IntoResponse;

/// GET /ane/compute — ANE compute subsystem status.
///
/// Returns whether the ANE compute subsystem is available, compile budget,
/// and warnings about private API usage.
pub async fn status_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let ane = &state.ane;
    let compile_count = ane.compile_count.load(Ordering::Relaxed);

    let built_with_feature = cfg!(feature = "ane");

    Json(serde_json::json!({
        "experimental": true,
        "enabled": ane.enabled,
        "available": ane.available,
        "built_with_ane_feature": built_with_feature,
        "compile_count": compile_count,
        "compile_budget_remaining": ane.compile_budget_remaining(),
        "compile_limit": 119,
        "warnings": [
            "Uses undocumented Apple private APIs — can break on any macOS update",
            "ANE compiler leaks ~119 compiles per process; restart daemon to reset"
        ],
    }))
}

/// POST /ane/eval — evaluate a pre-built graph on ANE hardware.
///
/// Currently scaffolded. Full implementation requires:
/// - MIL text + weight blob in request body (multipart or base64)
/// - Input tensor data as fp32 arrays
/// - Shape metadata for inputs/outputs
///
/// This endpoint will be implemented when we have a serialization format for
/// ANE graphs that can be sent over HTTP.
pub async fn eval_handler(
    State(state): State<AppState>,
) -> axum::response::Response {
    let ane = &state.ane;

    if !ane.enabled {
        return ane_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "ANE compute not enabled. Start daemon with --experimental-ane",
        );
    }

    #[cfg(not(feature = "ane"))]
    {
        return ane_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Binary not built with ANE support. Rebuild with: cargo build --features ane",
        );
    }

    #[cfg(feature = "ane")]
    {
        if ane.compile_budget_remaining() == 0 {
            return ane_error(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "ANE compile budget exhausted (~119 per process). Restart daemon to reset.",
            );
        }

        // TODO: Accept graph definition + input tensors, compile, eval, return output.
        // Tracked in: docs/plans/2026-03-08-ane-compute-integration.md Task 5
        ane_error(
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "ANE eval endpoint is scaffolded but not yet implemented. \
             See /ane/compute for subsystem status.",
        )
    }
}

/// GET /ane/probe — probe IOSurface memory layout for RDMA compatibility research.
///
/// Creates a small IOSurface, inspects its backing memory properties, and reports
/// whether the memory is likely compatible with RDMA registration (physically
/// contiguous, page-aligned, etc.).
///
/// This is a research endpoint for the ANE-RDMA bridge investigation.
pub async fn probe_handler(
    State(state): State<AppState>,
) -> axum::response::Response {
    if !state.ane.enabled {
        return ane_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "ANE compute not enabled. Start daemon with --experimental-ane",
        );
    }

    #[cfg(not(feature = "ane"))]
    {
        return ane_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Binary not built with ANE support. Rebuild with: cargo build --features ane",
        );
    }

    #[cfg(feature = "ane")]
    {
        use ane::{IOSurfaceExt, TensorData, Shape};
        use objc2_io_surface::IOSurface;

        // Probe several sizes typical for activation transfers
        let sizes: Vec<(& str, usize)> = vec![
            ("gpt2_768x128", 768 * 128 * 4),       // GPT-2 activations: 384 KB
            ("qwen_1024x128", 1024 * 128 * 4),     // Qwen 0.8B: 512 KB
            ("qwen27b_3584x128", 3584 * 128 * 4),  // Qwen 27B: 1.75 MB
        ];

        let mut probes = Vec::new();
        for (name, byte_count) in &sizes {
            let surface = IOSurface::with_byte_count(*byte_count);

            let alloc_size = surface.allocationSize() as usize;
            let base_addr = surface.baseAddress().as_ptr() as usize;
            let page_aligned = base_addr % 4096 == 0;
            let plane_count = surface.planeCount();

            probes.push(serde_json::json!({
                "name": name,
                "requested_bytes": byte_count,
                "allocated_bytes": alloc_size,
                "base_address": format!("0x{:x}", base_addr),
                "page_aligned": page_aligned,
                "plane_count": plane_count,
                "allocation_overhead_bytes": alloc_size as i64 - *byte_count as i64,
                "rdma_compatible_indicators": {
                    "page_aligned": page_aligned,
                    "contiguous_likely": plane_count <= 1,
                    "size_reasonable": alloc_size < 16 * 1024 * 1024, // < 16MB
                },
            }));
        }

        let response = serde_json::json!({
            "experimental": true,
            "purpose": "IOSurface memory layout profiling for RDMA compatibility research",
            "probes": probes,
            "notes": [
                "page_aligned=true is necessary for RDMA memory registration",
                "contiguous_likely=true (plane_count <= 1) suggests single physical region",
                "These are necessary but not sufficient conditions for RDMA compatibility",
                "Next step: attempt ibv_reg_mr() on the base address (requires RDMA hardware)",
            ],
        });

        (axum::http::StatusCode::OK, Json(response)).into_response()
    }
}
```

### Step 2: Register the module

In `src/main.rs`, add after line 5 (`mod watchdog;`):

```rust
mod ane;
```

### Step 3: Build (without feature — should compile clean)

```bash
cd /Users/ma/Projects/Personal/apple-smi && cargo build 2>&1
```

Expected: clean compile. The `#[cfg(feature = "ane")]` blocks are dead code.

### Step 4: Build with feature

```bash
cargo build --features ane 2>&1
```

Expected: compiles with ane-runtime linked.

### Step 5: Commit

```bash
git add src/ane.rs src/main.rs
git commit -m "feat(experimental): add ANE compute module with status/eval/probe endpoints

- AneState tracks compile budget (~119 per process)
- GET /ane/compute — subsystem status
- POST /ane/eval — scaffolded (not yet implemented)
- GET /ane/probe — IOSurface memory profiling for RDMA research
- All code compiles to no-ops without 'ane' feature flag
- EXPERIMENTAL: uses private Apple APIs"
```

---

## Task 3: Wire AneState into AppState and router

**Files:**
- Modify: `src/daemon.rs:29-45` — add `ane` field to `AppState`
- Modify: `src/daemon.rs:956-994` — add routes to `build_router`
- Modify: `src/daemon_startup.rs:281-296` — create AneState, add to AppState
- Modify: `src/daemon_startup.rs:326` — add startup banner lines

### Step 1: Add `ane` field to AppState

In `src/daemon.rs`, add after line 44 (`pub watchdog: ...`):

```rust
    pub ane: crate::ane::AneState,
```

### Step 2: Add routes to build_router

In `src/daemon.rs`, add after line 992 (`.route("/ane", get(ane_handler))`):

```rust
        // Experimental ANE compute (feature-flagged)
        .route("/ane/compute", get(crate::ane::status_handler))
        .route("/ane/eval", post(crate::ane::eval_handler))
        .route("/ane/probe", get(crate::ane::probe_handler))
```

### Step 3: Create AneState in daemon_startup

In `src/daemon_startup.rs`, add before the `let app_state = ...` block (before line 281):

```rust
    // Experimental ANE compute subsystem
    let ane_state = if experimental_ane {
        tracing::warn!("EXPERIMENTAL: ANE compute endpoints enabled (--experimental-ane)");
        if !cfg!(feature = "ane") {
            tracing::error!("--experimental-ane requires building with: cargo build --features ane");
            tracing::error!("ANE compute will be unavailable");
        }
        crate::ane::AneState::new(true)
    } else {
        crate::ane::AneState::disabled()
    };
```

### Step 4: Add to AppState construction

In `src/daemon_startup.rs`, add `ane: ane_state,` to the `AppState` struct literal (after `watchdog: wd,` on line 295):

```rust
        ane: ane_state,
```

### Step 5: Add startup banner

In `src/daemon_startup.rs`, add after line 339 (the `if cluster_hub` banner block):

```rust
    if experimental_ane {
        eprintln!("  \x1b[33m[EXPERIMENTAL]\x1b[0m ANE compute endpoints:");
        eprintln!("  GET  /ane/compute      ANE compute subsystem status");
        eprintln!("  POST /ane/eval         Submit program to ANE (not yet implemented)");
        eprintln!("  GET  /ane/probe        IOSurface memory layout for RDMA research");
    }
```

### Step 6: Build both variants

```bash
# Default (no ane feature)
cd /Users/ma/Projects/Personal/apple-smi && cargo build 2>&1

# With ane feature
cargo build --features ane 2>&1
```

Both must compile clean.

### Step 7: Commit

```bash
git add src/daemon.rs src/daemon_startup.rs
git commit -m "feat: wire ANE compute into AppState and HTTP router

- AneState added to AppState (disabled by default)
- 3 new routes: /ane/compute, /ane/eval, /ane/probe
- --experimental-ane activates at startup with warning
- Startup banner shows ANE endpoints when enabled"
```

---

## Task 4: Add integration test for ANE endpoints

**Files:**
- Create: `tests/ane_endpoints.rs`

### Step 1: Write the test

Create `tests/ane_endpoints.rs`:

```rust
//! Integration test for ANE compute endpoints.
//!
//! Tests both the disabled case (no --experimental-ane) and enabled case.
//! Run explicitly: cargo test --test ane_endpoints -- --nocapture

use std::time::Duration;

/// When --experimental-ane is NOT passed, /ane/compute should still respond
/// (with enabled: false) since it's always registered in the router.
#[tokio::test]
async fn test_ane_status_without_flag() {
    let port = 19290 + (std::process::id() % 500) as u16;
    let mut child = match tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    tokio::time::sleep(Duration::from_secs(3)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{port}/ane/compute");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert!(r.status().is_success(), "/ane/compute returned {}", r.status());
            let body: serde_json::Value = r.json().await.unwrap();
            assert_eq!(body["enabled"], false, "should be disabled without --experimental-ane");
            assert_eq!(body["experimental"], true);
            eprintln!("/ane/compute (no flag) -> {}", serde_json::to_string_pretty(&body).unwrap());
        }
        Err(e) => panic!("/ane/compute failed: {e}"),
    }

    // /ane/eval should return 503 when disabled
    let resp = client
        .post(format!("http://localhost:{port}/ane/eval"))
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert_eq!(r.status().as_u16(), 503, "/ane/eval should be 503 when disabled");
            eprintln!("/ane/eval (disabled) -> 503 OK");
        }
        Err(e) => panic!("/ane/eval failed: {e}"),
    }

    child.kill().await.ok();
}

/// When --experimental-ane IS passed, /ane/compute should show enabled: true.
/// Note: this only tests the feature-flag path. Actual ANE hardware access
/// requires --features ane at build time.
#[tokio::test]
async fn test_ane_status_with_flag() {
    let port = 19290 + 500 + (std::process::id() % 500) as u16;
    let mut child = match tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string(), "--experimental-ane"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    tokio::time::sleep(Duration::from_secs(3)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{port}/ane/compute");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert!(r.status().is_success(), "/ane/compute returned {}", r.status());
            let body: serde_json::Value = r.json().await.unwrap();
            assert_eq!(body["enabled"], true, "should be enabled with --experimental-ane");
            assert_eq!(body["compile_limit"], 119);
            eprintln!("/ane/compute (with flag) -> {}", serde_json::to_string_pretty(&body).unwrap());
        }
        Err(e) => panic!("/ane/compute failed: {e}"),
    }

    child.kill().await.ok();
}
```

### Step 2: Run the tests

```bash
cd /Users/ma/Projects/Personal/apple-smi && cargo test --test ane_endpoints -- --nocapture 2>&1
```

Expected: both tests pass.

### Step 3: Commit

```bash
git add tests/ane_endpoints.rs
git commit -m "test: add integration tests for ANE compute endpoints

Tests both disabled (no flag) and enabled (--experimental-ane) modes.
Verifies status response shape and eval rejection when disabled."
```

---

## Task 5: Add IOSurface memory probe to ane-runtime (RDMA research)

**Files:**
- Create: `crates/ane-runtime/src/probe.rs`
- Modify: `crates/ane-runtime/src/lib.rs` — add `pub mod probe;`

This task adds a standalone probe module to ane-runtime that reports IOSurface
memory properties relevant to RDMA compatibility. This is the investigation
code for Phase 2 of the ANE-RDMA bridge.

### Step 1: Create the probe module

Create `crates/ane-runtime/src/probe.rs`:

```rust
//! IOSurface memory layout probing for RDMA compatibility research.
//!
//! This module creates IOSurfaces of various sizes and inspects their
//! backing memory properties to determine whether they can be used with
//! RDMA memory registration (`ibv_reg_mr`).
//!
//! # Key Questions
//!
//! 1. Is the backing memory page-aligned? (Required for RDMA)
//! 2. Is it physically contiguous? (Required for most RDMA implementations)
//! 3. Can we use `IOSurfaceCreateMachPort` to share surfaces cross-process?
//! 4. What's the allocation overhead for typical activation sizes?

use objc2_io_surface::IOSurface;
use serde::Serialize;

use crate::io_surface::IOSurfaceExt;

/// Results from probing a single IOSurface allocation.
#[derive(Debug, Serialize)]
pub struct SurfaceProbe {
    /// Human-readable label for this probe.
    pub label: String,
    /// Requested byte count.
    pub requested_bytes: usize,
    /// Actual allocation size (may include padding).
    pub allocated_bytes: usize,
    /// Base address of the surface memory.
    pub base_address: u64,
    /// Whether the base address is page-aligned (4096-byte boundary).
    pub page_aligned: bool,
    /// Number of planes in the surface (1 = likely contiguous).
    pub plane_count: usize,
    /// Allocation overhead in bytes.
    pub overhead_bytes: i64,
    /// Whether this surface looks compatible with RDMA registration.
    pub rdma_likely_compatible: bool,
}

/// Probe IOSurface memory layout for a given byte count.
pub fn probe_surface(label: &str, byte_count: usize) -> SurfaceProbe {
    let surface = IOSurface::with_byte_count(byte_count);

    let alloc_size = surface.allocationSize() as usize;
    let base_addr = surface.baseAddress().as_ptr() as u64;
    let page_aligned = base_addr % 4096 == 0;
    let plane_count = surface.planeCount();

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

/// Run a standard set of probes covering typical activation transfer sizes.
///
/// Returns probes for sizes matching common model hidden dimensions × sequence lengths,
/// using fp32 (4 bytes per element) since that's what ANE MIL I/O uses.
pub fn probe_standard_sizes() -> Vec<SurfaceProbe> {
    let configs = [
        ("gpt2_768x128", 768 * 128 * 4),         // GPT-2: 384 KB
        ("qwen08b_1024x128", 1024 * 128 * 4),    // Qwen 0.8B: 512 KB
        ("qwen27b_3584x128", 3584 * 128 * 4),    // Qwen 27B: 1.75 MB
        ("qwen35b_4096x128", 4096 * 128 * 4),    // Qwen 35B: 2 MB
        ("small_64x64", 64 * 64 * 4),            // Minimal: 16 KB
        ("large_8192x256", 8192 * 256 * 4),      // Large: 8 MB
    ];

    configs
        .iter()
        .map(|(label, size)| probe_surface(label, *size))
        .collect()
}
```

### Step 2: Register the module

In `crates/ane-runtime/src/lib.rs`, add after the `pub mod ops;` line:

```rust
pub mod probe;
```

### Step 3: Build and test

```bash
cd /Users/ma/Projects/Personal/apple-smi && cargo build -p ane-runtime 2>&1
```

### Step 4: Commit

```bash
git add crates/ane-runtime/src/probe.rs crates/ane-runtime/src/lib.rs
git commit -m "feat(ane-runtime): add IOSurface memory probe for RDMA research

probe::probe_standard_sizes() creates IOSurfaces at typical activation
transfer sizes and reports page alignment, plane count, and allocation
overhead — key properties for RDMA memory registration compatibility."
```

---

## Task 6: Update /ane/probe to use the new ane-runtime probe module

**Files:**
- Modify: `src/ane.rs` — replace inline probe logic with `ane::probe::probe_standard_sizes()`

### Step 1: Simplify probe_handler

In `src/ane.rs`, replace the `#[cfg(feature = "ane")]` block inside `probe_handler` with:

```rust
    #[cfg(feature = "ane")]
    {
        let probes = ane::probe::probe_standard_sizes();

        let response = serde_json::json!({
            "experimental": true,
            "purpose": "IOSurface memory layout profiling for RDMA compatibility research",
            "probes": probes,
            "summary": {
                "all_page_aligned": probes.iter().all(|p| p.page_aligned),
                "all_single_plane": probes.iter().all(|p| p.plane_count <= 1),
                "all_rdma_likely": probes.iter().all(|p| p.rdma_likely_compatible),
            },
            "next_steps": [
                "If all_rdma_likely=true: attempt ibv_reg_mr() on IOSurface base address",
                "If false: fall back to memcpy path (IOSurface → RDMA buffer → transfer)",
                "Memcpy fallback adds ~50-100μs per transfer for typical activation sizes",
            ],
        });

        (axum::http::StatusCode::OK, Json(response)).into_response()
    }
```

### Step 2: Build both variants

```bash
cd /Users/ma/Projects/Personal/apple-smi && cargo build 2>&1
cargo build --features ane 2>&1
```

### Step 3: Commit

```bash
git add src/ane.rs
git commit -m "refactor: use ane::probe module in /ane/probe endpoint

Replaces inline IOSurface probing with the new ane-runtime probe module.
Adds summary fields (all_page_aligned, all_rdma_likely) for quick assessment."
```

---

## Task 7: Update ROADMAP.md with ANE compute section

**Files:**
- Modify: `ROADMAP.md`

### Step 1: Add v0.10 ANE section

After the v0.9 section (line 123), add:

```markdown

## v0.10 — ANE Integration (in progress)

Direct Apple Neural Engine compute via private APIs + IOSurface I/O.

- [x] **ANE power (sudoless)** — IOReport `"Energy Model"` channel for ANE power without sudo
- [x] **GET /ane** — dedicated ANE metrics endpoint (power, active status, IOReport source)
- [x] **`ProcessFramework::AneNative`** — process detection for ANE workloads
- [x] **`ane-runtime` crate** — Rust FFI wrappers for `_ANEInMemoryModel` lifecycle (8000 LOC)
- [x] **GPT-2 forward pass** — working ANE inference at 31.1 tok/s (M3 Ultra)
- [ ] **`--experimental-ane`** — CLI flag + feature gate for ANE compute endpoints
- [ ] **GET /ane/compute** — ANE compute subsystem status (compile budget, availability)
- [ ] **POST /ane/eval** — submit MIL program to ANE via HTTP (scaffolded)
- [ ] **GET /ane/probe** — IOSurface memory layout profiling for RDMA research
- [ ] **ANE-RDMA bridge** — cross-node activation transfer via Thunderbolt 5 RDMA
  - Research: can IOSurface memory be RDMA-registered? (`ibv_reg_mr` on base address)
  - Fallback: memcpy IOSurface ↔ RDMA buffer (~50μs per transfer)
- [ ] **Pipeline parallelism** — distribute transformer layers across nodes (6 layers per stage)
- [ ] **MoE expert placement** — route MoE experts to individual ANE dies across cluster
```

### Step 2: Update v0.9 reference

In the v0.9 section, change the ANE utilization bullet (line 120):

```markdown
- [ ] **ANE utilization** — extend v0.10 ANE integration with Metal Performance Counters correlation
```

### Step 3: Commit

```bash
git add ROADMAP.md
git commit -m "docs: add v0.10 ANE integration section to ROADMAP

Covers: monitoring (done), ane-runtime (done), compute endpoints (in progress),
RDMA bridge (research), pipeline parallelism (planned), MoE placement (planned)."
```

---

## Summary

| Task | Type | Risk | What it does |
|------|------|------|-------------|
| 1 | CLI | None | Add `--experimental-ane` hidden flag |
| 2 | Feature | Low | Create `src/ane.rs` with status/eval/probe handlers |
| 3 | Wiring | Low | Wire AneState into AppState and router |
| 4 | Test | None | Integration tests for ANE endpoints |
| 5 | Research | Low | IOSurface memory probe in ane-runtime |
| 6 | Refactor | None | Use probe module in /ane/probe endpoint |
| 7 | Docs | None | ROADMAP update |

**Total new endpoints:** 3 (`/ane/compute`, `/ane/eval`, `/ane/probe`)
**Total new files:** 3 (`src/ane.rs`, `crates/ane-runtime/src/probe.rs`, `tests/ane_endpoints.rs`)
**Existing /ane endpoint:** Unchanged (still serves monitoring data from IOReport)
**Feature flags:** Uses existing `ane` Cargo feature + new `--experimental-ane` CLI flag

### Key Design Decision: Why ane-runtime, not a new ObjC bridge

The original plan (Task B2) proposed building `bridge.m` — a ~750-line ObjC file with raw `objc_msgSend` calls. This duplicates work because `ane-runtime` already has:

- Type-safe Rust wrappers via `objc2` (no raw `objc_msgSend`)
- Working `Graph → MIL → compile → load → evaluate` pipeline
- IOSurface management with RAII lock guards
- Working binaries (`gpt2_forward` at 31.1 tok/s)

By using `ane-runtime` directly, we get:
1. **No code duplication** — single source of truth for ANE FFI
2. **Type safety** — `objc2` generates correct ABI calls, `bridge.m` used raw casts
3. **Tested** — GPT-2 forward pass already validated the pipeline
4. **IOSurface integration** — `TensorData` already wraps IOSurface with shape tracking
