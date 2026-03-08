//! Experimental ANE compute endpoints.
//!
//! Gated behind `--experimental-ane` CLI flag AND `ane` Cargo feature.
//! Uses Apple's private AppleNeuralEngine.framework via the `ane-runtime` crate.

use axum::{extract::State, response::Json, response::IntoResponse};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::daemon::AppState;

// ---------------------------------------------------------------------------
// ANE subsystem state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AneState {
    pub enabled: bool,
    pub available: bool,
    pub compile_count: Arc<AtomicU32>,
}

impl AneState {
    pub fn new(enabled: bool) -> Self {
        let available = if enabled {
            cfg!(feature = "ane")
        } else {
            false
        };
        Self {
            enabled,
            available,
            compile_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn compile_budget_remaining(&self) -> u32 {
        119u32.saturating_sub(self.compile_count.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// GET /ane/compute — ANE compute subsystem status.
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

/// POST /ane/eval — evaluate a pre-built graph on ANE hardware (scaffolded).
pub async fn eval_handler(
    State(state): State<AppState>,
) -> axum::response::Response {
    let ane = &state.ane;

    if !ane.enabled {
        let body = Json(serde_json::json!({"error": "ANE compute not enabled. Start daemon with --experimental-ane"}));
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    }

    #[cfg(not(feature = "ane"))]
    {
        let body = Json(serde_json::json!({"error": "Binary not built with ANE support. Rebuild with: cargo build --features ane"}));
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    }

    #[cfg(feature = "ane")]
    {
        if ane.compile_budget_remaining() == 0 {
            let body = Json(serde_json::json!({"error": "ANE compile budget exhausted (~119 per process). Restart daemon to reset."}));
            return (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response();
        }

        let body = Json(serde_json::json!({"error": "ANE eval endpoint is scaffolded but not yet implemented. See /ane/compute for subsystem status."}));
        (axum::http::StatusCode::NOT_IMPLEMENTED, body).into_response()
    }
}

/// GET /ane/probe — probe IOSurface memory layout for RDMA compatibility research.
pub async fn probe_handler(
    State(state): State<AppState>,
) -> axum::response::Response {
    if !state.ane.enabled {
        let body = Json(serde_json::json!({"error": "ANE compute not enabled. Start daemon with --experimental-ane"}));
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    }

    #[cfg(not(feature = "ane"))]
    {
        let body = Json(serde_json::json!({"error": "Binary not built with ANE support. Rebuild with: cargo build --features ane"}));
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response();
    }

    #[cfg(feature = "ane")]
    {
        let probes = ane::probe::probe_standard_sizes();
        let all_page_aligned = probes.iter().all(|p| p.page_aligned);
        let all_single_plane = probes.iter().all(|p| p.plane_count <= 1);
        let all_rdma_likely = probes.iter().all(|p| p.rdma_likely_compatible);

        let response = serde_json::json!({
            "experimental": true,
            "purpose": "IOSurface memory layout profiling for RDMA compatibility research",
            "probes": probes,
            "summary": {
                "all_page_aligned": all_page_aligned,
                "all_single_plane": all_single_plane,
                "all_rdma_likely": all_rdma_likely,
            },
            "next_steps": [
                "If all_rdma_likely=true: attempt ibv_reg_mr() on IOSurface base address",
                "If false: fall back to memcpy path (IOSurface -> RDMA buffer -> transfer)",
                "Memcpy fallback adds ~50-100μs per transfer for typical activation sizes",
            ],
        });

        (axum::http::StatusCode::OK, Json(response)).into_response()
    }
}
