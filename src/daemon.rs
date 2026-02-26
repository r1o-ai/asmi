use axum::{extract::State, response::Json, routing::get, Router};
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
}

async fn metrics_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => Json(serde_json::to_value(s).unwrap_or(serde_json::json!({"error": "serialize failed"}))),
        None => Json(serde_json::json!({"error": "no data yet", "hostname": state.hostname})),
    }
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let has_data = state.snapshot.read().await.is_some();
    Json(serde_json::json!({
        "ok": has_data,
        "hostname": state.hostname,
        "uptime_secs": state.started_at.elapsed().as_secs(),
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

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/processes", get(processes_handler))
        .route("/cluster", get(cluster_handler))
        .route("/nodes", get(nodes_handler))
        .route("/stream", get(stream_handler))
        .route("/jaccl/config", get(jaccl_config_handler).post(jaccl_generate_handler))
        .with_state(state)
}
