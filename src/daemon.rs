use axum::{extract::State, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub snapshot: Arc<RwLock<Option<asmi_core::NodeSnapshot>>>,
    pub cluster_state: Option<Arc<RwLock<asmi_core::ClusterState>>>,
    pub node_map: Arc<RwLock<asmi_core::NodeMap>>,
    pub hostname: String,
    pub started_at: std::time::Instant,
    pub metrics_tx: tokio::sync::broadcast::Sender<String>,
    pub model_cache: Arc<RwLock<Option<(Vec<asmi_core::LocalModel>, std::time::Instant)>>>,
    pub runtime: Arc<RuntimeInfo>,
}

/// Cached Python/MLX/macOS version info, probed once at startup.
#[derive(Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub python_version: Option<String>,
    pub mlx_version: Option<String>,
    pub mlx_device: Option<String>,
    pub vllm_version: Option<String>,
    pub macos_version: Option<String>,
}

/// Probe the local Python environment for ML framework versions.
pub async fn probe_runtime() -> RuntimeInfo {
    use tokio::process::Command;

    let python = Command::new("python3")
        .args(["-c", "import sys; print(sys.version.split()[0])"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let mlx = Command::new("python3")
        .args(["-c", "import mlx.core as mx; print(mx.__version__); print(mx.default_device())"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let (mlx_version, mlx_device) = match mlx {
        Some(output) => {
            let mut lines = output.lines();
            (
                lines.next().map(|s| s.to_string()),
                lines.next().map(|s| s.to_string()),
            )
        }
        None => (None, None),
    };

    let vllm = Command::new("python3")
        .args(["-c", "import vllm; print(vllm.__version__)"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let macos = Command::new("sw_vers")
        .args(["-productVersion"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    RuntimeInfo {
        python_version: python,
        mlx_version,
        mlx_device,
        vllm_version: vllm,
        macos_version: macos,
    }
}

async fn metrics_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => Json(serde_json::to_value(s).unwrap_or(serde_json::json!({"error": "serialize failed"}))),
        None => Json(serde_json::json!({"error": "no data yet", "hostname": state.hostname})),
    }
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.snapshot.read().await;
    let has_data = snap.is_some();
    let process_count = snap.as_ref().map(|s| s.processes.len()).unwrap_or(0);
    Json(serde_json::json!({
        "ok": has_data,
        "hostname": state.hostname,
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
        "process_count": process_count,
    }))
}

async fn processes_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => Json(serde_json::json!({
            "hostname": s.hostname,
            "processes": s.processes,
        })),
        None => Json(serde_json::json!({"processes": []})),
    }
}

/// GET /cluster → Vec<NodeSnapshot> for all polled nodes (hub mode only)
async fn cluster_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    match &state.cluster_state {
        Some(cs) => {
            let s = cs.read().await;
            let snapshots: Vec<&asmi_core::NodeSnapshot> = s.snapshots.values().collect();
            Json(serde_json::to_value(&snapshots)
                .unwrap_or(serde_json::json!([])))
        }
        None => Json(serde_json::json!({
            "error": "not running in cluster hub mode (start with --cluster)"
        })),
    }
}

/// GET /nodes → list of known node hostnames
async fn nodes_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    match &state.cluster_state {
        Some(cs) => {
            let s = cs.read().await;
            let hostnames: Vec<&str> = s.snapshots.keys().map(|k| k.as_str()).collect();
            Json(serde_json::json!({ "nodes": hostnames, "total": hostnames.len() }))
        }
        None => Json(serde_json::json!({ "nodes": [], "total": 0 })),
    }
}

/// GET /stream → SSE push of NodeSnapshot JSON on every poll tick (~2s)
async fn stream_handler(
    State(state): State<AppState>,
) -> axum::response::sse::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>
{
    use futures::StreamExt;

    let rx = state.metrics_tx.subscribe();
    let stream = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|result| {
        futures::future::ready(match result {
            Ok(json) => Some(Ok(axum::response::sse::Event::default().data(json))),
            Err(_) => None, // lagged subscriber — skip missed messages
        })
    });

    axum::response::sse::Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// GET /jaccl/config → JACCL hostfile matrix from stored RDMA link topology.
/// Query params: ?hosts=m3u2,m3u1 (comma-separated hostnames, optional — defaults to all)
async fn jaccl_config_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let nm = state.node_map.read().await;

    if nm.rdma_links.is_empty() {
        return Json(serde_json::json!({
            "success": false,
            "error": "No RDMA links discovered. Run `asmi` TUI scan first to discover RDMA topology."
        }));
    }

    let hostfile_json = nm.hostfile_jaccl(&state.hostname);
    if hostfile_json == "[]" {
        return Json(serde_json::json!({
            "success": false,
            "error": "No RDMA-connected nodes found in topology"
        }));
    }

    // Parse the JSON string from hostfile_jaccl and optionally filter by requested hosts
    match serde_json::from_str::<serde_json::Value>(&hostfile_json) {
        Ok(hosts) => {
            let filtered = if let Some(hosts_param) = params.get("hosts") {
                let requested: Vec<&str> = hosts_param.split(',').collect();
                if let Some(arr) = hosts.as_array() {
                    let filtered: Vec<&serde_json::Value> = arr.iter().filter(|h| {
                        h.get("ssh").and_then(|s| s.as_str()).map_or(false, |ssh| {
                            requested.iter().any(|r| ssh.starts_with(r))
                        })
                    }).collect();
                    serde_json::to_value(&filtered).unwrap_or(hosts.clone())
                } else {
                    hosts.clone()
                }
            } else {
                hosts.clone()
            };

            let count = filtered.as_array().map(|a| a.len()).unwrap_or(0);
            Json(serde_json::json!({
                "success": true,
                "hosts": filtered,
                "nodeCount": count,
                "rdma_links_total": nm.rdma_links.len(),
                "local_hostname": state.hostname,
            }))
        }
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": format!("Failed to build JACCL matrix: {e}"),
            "raw": hostfile_json,
        })),
    }
}

/// POST /jaccl/config → generate and write JACCL hostfile to coordinator
async fn jaccl_generate_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let nm = state.node_map.read().await;

    if nm.rdma_links.is_empty() {
        return Json(serde_json::json!({
            "success": false,
            "error": "No RDMA links discovered. Run `asmi` TUI scan first."
        }));
    }

    let hostfile_json = nm.hostfile_jaccl(&state.hostname);
    let hosts: serde_json::Value = match serde_json::from_str(&hostfile_json) {
        Ok(v) => v,
        Err(e) => return Json(serde_json::json!({
            "success": false,
            "error": format!("Failed to build matrix: {e}"),
        })),
    };

    let count = hosts.as_array().map(|a| a.len()).unwrap_or(0);
    if count == 0 {
        return Json(serde_json::json!({
            "success": false,
            "error": "No RDMA-connected nodes found",
        }));
    }

    // Optionally write to disk if 'write' flag is set
    let should_write = body.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
    let path = format!("~/hostfile-jaccl-{}node.json", count);

    if should_write {
        let pretty = serde_json::to_string_pretty(&hosts).unwrap_or_default();
        // Write via local file (asmi runs on the coordinator)
        let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
        match std::fs::write(&expanded, &pretty) {
            Ok(()) => {
                tracing::info!(path = %expanded, nodes = count, "wrote JACCL hostfile");
            }
            Err(e) => {
                return Json(serde_json::json!({
                    "success": false,
                    "error": format!("Failed to write hostfile: {e}"),
                    "hosts": hosts,
                }));
            }
        }
    }

    Json(serde_json::json!({
        "success": true,
        "action": if should_write { "generate" } else { "discover" },
        "hosts": hosts,
        "nodeCount": count,
        "path": if should_write { Some(&path) } else { None },
    }))
}

/// GET /models → cached local model file listing
async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache = state.model_cache.read().await;
    let (models, scanned_at) = match cache.as_ref() {
        Some((m, t)) => (m.clone(), t.elapsed().as_secs()),
        None => (vec![], 0),
    };
    Json(serde_json::json!({
        "models": models,
        "scan_age_seconds": scanned_at,
    }))
}

/// GET /logs?name=mlx-server&lines=50 → tail server log files
async fn logs_handler(
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let name = params.get("name").cloned().unwrap_or_else(|| "mlx-server".to_string());
    let lines: usize = params.get("lines")
        .and_then(|l| l.parse().ok())
        .unwrap_or(50)
        .min(500);

    let log_path = match name.as_str() {
        "mlx-server" | "mlx_lm" => "/tmp/r1o-mlx_lm-server.log",
        "mlx-vlm" | "mlx_vlm" => "/tmp/r1o-mlx_vlm-server.log",
        "vllm" | "vllm_mlx" => "/tmp/r1o-vllm_mlx-server.log",
        "asmi" | "daemon" => "~/Library/Logs/asmi-daemon.log",
        _ => {
            return Json(serde_json::json!({
                "error": format!("unknown log name: {name}"),
                "known_names": ["mlx-server", "mlx-vlm", "vllm", "asmi"],
            }));
        }
    };

    let expanded = log_path.replace('~', &std::env::var("HOME").unwrap_or_default());

    match std::fs::read_to_string(&expanded) {
        Ok(content) => {
            let all_lines: Vec<&str> = content.lines().collect();
            let start = all_lines.len().saturating_sub(lines);
            let tail: Vec<&str> = all_lines[start..].to_vec();
            Json(serde_json::json!({
                "name": name,
                "path": expanded,
                "lines": tail,
                "total_lines": all_lines.len(),
            }))
        }
        Err(e) => Json(serde_json::json!({
            "name": name,
            "path": expanded,
            "error": format!("could not read log: {e}"),
        })),
    }
}

/// GET /runtime → Python/MLX/macOS version info (cached at startup)
async fn runtime_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.runtime.as_ref())
        .unwrap_or(serde_json::json!({"error": "no runtime info"})))
}

/// GET /health/setup → run setup validation checks
async fn setup_handler() -> Json<serde_json::Value> {
    let checks = asmi_core::run_setup_checks().await;
    Json(serde_json::to_value(&checks)
        .unwrap_or(serde_json::json!({"error": "check failed"})))
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/health/setup", get(setup_handler))
        .route("/processes", get(processes_handler))
        .route("/models", get(models_handler))
        .route("/logs", get(logs_handler))
        .route("/runtime", get(runtime_handler))
        .route("/cluster", get(cluster_handler))
        .route("/nodes", get(nodes_handler))
        .route("/stream", get(stream_handler))
        .route("/jaccl/config", get(jaccl_config_handler).post(jaccl_generate_handler))
        .with_state(state)
}
