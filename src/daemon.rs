use axum::{extract::{Path, Query, State}, response::Json, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Typed API error — returns proper HTTP status codes instead of 200 + error JSON
// ---------------------------------------------------------------------------

enum ApiError {
    BadRequest(String),
    NotFound(String),
    Internal(String),
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (axum::http::StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (axum::http::StatusCode::NOT_FOUND, msg),
            ApiError::Internal(msg) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        let body = axum::Json(serde_json::json!({"error": message}));
        (status, body).into_response()
    }
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct AppState {
    pub snapshot: Arc<RwLock<Option<asmi_core::NodeSnapshot>>>,
    pub cluster_state: Option<Arc<RwLock<asmi_core::ClusterState>>>,
    pub node_map: Arc<RwLock<asmi_core::NodeMap>>,
    pub hostname: String,
    pub started_at: std::time::Instant,
    pub metrics_tx: tokio::sync::broadcast::Sender<String>,
    pub model_cache: Arc<RwLock<Option<(Vec<asmi_core::LocalModel>, std::time::Instant)>>>,
    pub thunderbolt_cache: Arc<RwLock<Option<(serde_json::Value, std::time::Instant)>>>,
    pub topology_cache: Arc<RwLock<Option<(crate::topology::TopologyReport, std::time::Instant)>>>,
    pub runtime: Arc<RuntimeInfo>,
    pub serve_managers: Arc<RwLock<HashMap<u16, crate::serve::ServeManager>>>,
    pub share_manager: crate::serve::ShareManager,
    pub peer_heartbeat: Arc<crate::serve::PeerHeartbeat>,
    pub watchdog: Arc<crate::watchdog::Watchdog>,
    pub ane: crate::ane::AneState,
    pub egpu_cache: Arc<RwLock<Option<(serde_json::Value, std::time::Instant)>>>,
    /// JACCL worker — dedicated OS thread for all RDMA operations (Phase 3).
    pub jaccl_worker: Arc<crate::transfer::JacclWorker>,
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

/// Re-export from asmi_core so `crate::daemon::resolve_python` still works
/// for serve.rs and other consumers in the binary crate.
pub use asmi_core::resolve_python;

/// Probe the local Python environment for ML framework versions.
pub async fn probe_runtime() -> RuntimeInfo {
    use tokio::process::Command;

    let py = resolve_python();

    let python = Command::new(py)
        .args(["-c", "import sys; print(sys.version.split()[0])"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let mlx = Command::new(py)
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

    let vllm = Command::new(py)
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

async fn metrics_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => Ok(Json(serde_json::to_value(s)
            .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?)),
        None => Err(ApiError::NotFound(format!("no data yet (hostname: {})", state.hostname))),
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
async fn cluster_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    match &state.cluster_state {
        Some(cs) => {
            let s = cs.read().await;
            let snapshots: Vec<&asmi_core::NodeSnapshot> = s.snapshots.values().collect();
            Ok(Json(serde_json::to_value(&snapshots)
                .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?))
        }
        None => Err(ApiError::BadRequest("not running in cluster hub mode (start with --cluster)".into())),
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
) -> Result<Json<serde_json::Value>, ApiError> {
    // Prefer live topology cache over stale NodeMap rdma_links
    let topo_cache = state.topology_cache.read().await;
    if let Some((report, scanned_at)) = topo_cache.as_ref() {
        if report.links.is_empty() {
            return Err(ApiError::NotFound("Topology scanned but no RDMA links found.".into()));
        }

        // Build JACCL hostfile from live topology links
        let target_nodes = if let Some(largest_subset) = report.jaccl_ready_subsets.iter().max_by_key(|s| s.len()) {
            if largest_subset.len() >= 2 {
                largest_subset.clone()
            } else {
                report.nodes.clone()
            }
        } else {
            report.nodes.clone()
        };

        let links = &report.links;

        let entries: Vec<serde_json::Value> = target_nodes.iter().map(|node| {
            let rdma_row: Vec<serde_json::Value> = target_nodes.iter().map(|other| {
                if node == other {
                    serde_json::Value::Null
                } else {
                    // Find the link between node and other
                    links.iter()
                        .find(|l| {
                            (l.node_a == *node && l.node_b == *other) ||
                            (l.node_b == *node && l.node_a == *other)
                        })
                        .map(|l| {
                            let device = if l.node_a == *node { &l.device_a } else { &l.device_b };
                            let clean_device = device.strip_prefix("rdma_").unwrap_or(device);
                            serde_json::Value::String(format!("rdma_{clean_device}"))
                        })
                        .unwrap_or(serde_json::Value::Null)
                }
            }).collect();

            serde_json::json!({
                "ssh": node,
                "ips": [],
                "rdma": rdma_row,
            })
        }).collect();

        let filtered = if let Some(hosts_param) = params.get("hosts") {
            let requested: Vec<&str> = hosts_param.split(',').collect();
            let filtered: Vec<&serde_json::Value> = entries.iter().filter(|h| {
                h.get("ssh").and_then(|s| s.as_str()).is_some_and(|ssh| {
                    requested.iter().any(|r| ssh.starts_with(r))
                })
            }).collect();
            serde_json::to_value(&filtered).unwrap_or(serde_json::json!([]))
        } else {
            serde_json::to_value(&entries).unwrap_or(serde_json::json!([]))
        };

        let count = filtered.as_array().map(|a| a.len()).unwrap_or(0);
        return Ok(Json(serde_json::json!({
            "success": true,
            "source": "live_topology",
            "hosts": filtered,
            "nodeCount": count,
            "links_total": links.len(),
            "mesh_complete": report.mesh_complete,
            "jaccl_ready": report.jaccl_ready,
            "scan_age_seconds": scanned_at.elapsed().as_secs(),
            "local_hostname": state.hostname,
        })));
    }
    drop(topo_cache);

    // Fallback: use stale NodeMap rdma_links (pre-topology scan)
    let nm = state.node_map.read().await;

    if nm.rdma_links.is_empty() {
        return Err(ApiError::NotFound(
            "No RDMA links discovered. Run `asmi` TUI scan first to discover RDMA topology.".into(),
        ));
    }

    let hostfile_json = nm.hostfile_jaccl(&state.hostname);
    if hostfile_json == "[]" {
        return Err(ApiError::NotFound("No RDMA-connected nodes found in topology".into()));
    }

    match serde_json::from_str::<serde_json::Value>(&hostfile_json) {
        Ok(hosts) => {
            let filtered = if let Some(hosts_param) = params.get("hosts") {
                let requested: Vec<&str> = hosts_param.split(',').collect();
                if let Some(arr) = hosts.as_array() {
                    let filtered: Vec<&serde_json::Value> = arr.iter().filter(|h| {
                        h.get("ssh").and_then(|s| s.as_str()).is_some_and(|ssh| {
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
            Ok(Json(serde_json::json!({
                "success": true,
                "source": "node_map_fallback",
                "hosts": filtered,
                "nodeCount": count,
                "rdma_links_total": nm.rdma_links.len(),
                "local_hostname": state.hostname,
            })))
        }
        Err(e) => Err(ApiError::Internal(format!("Failed to build JACCL matrix: {e}"))),
    }
}

/// POST /jaccl/config → generate and write JACCL hostfile to coordinator
async fn jaccl_generate_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut hostfile_json = String::new();

    // Prefer live topology cache over stale NodeMap rdma_links
    {
        let topo_cache = state.topology_cache.read().await;
        if let Some((report, _)) = topo_cache.as_ref() {
            if !report.links.is_empty() {
                // Find the largest jaccl-ready subset
                let target_nodes = if let Some(largest_subset) = report.jaccl_ready_subsets.iter().max_by_key(|s| s.len()) {
                    if largest_subset.len() >= 2 {
                        largest_subset.clone()
                    } else {
                        report.nodes.clone() // Fallback if no valid subset >= 2
                    }
                } else {
                    report.nodes.clone() // Fallback
                };

                let links = &report.links;

                let entries: Vec<serde_json::Value> = target_nodes.iter().map(|node| {
                    let rdma_row: Vec<serde_json::Value> = target_nodes.iter().map(|other| {
                        if node == other {
                            serde_json::Value::Null
                        } else {
                            links.iter()
                                .find(|l| {
                                    (l.node_a == *node && l.node_b == *other) ||
                                    (l.node_b == *node && l.node_a == *other)
                                })
                                .map(|l| {
                                    let device = if l.node_a == *node { &l.device_a } else { &l.device_b };
                                    // Strip "rdma_" prefix if it already exists, then re-add it
                                    let clean_device = device.strip_prefix("rdma_").unwrap_or(device);
                                    serde_json::Value::String(format!("rdma_{clean_device}"))
                                })
                                .unwrap_or(serde_json::Value::Null)
                        }
                    }).collect();

                    serde_json::json!({
                        "ssh": node,
                        "ips": [],
                        "rdma": rdma_row,
                    })
                }).collect();

                hostfile_json = serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string());
            }
        }
    }

    // Fallback: use stale NodeMap rdma_links
    if hostfile_json.is_empty() || hostfile_json == "[]" {
        let nm = state.node_map.read().await;
        if nm.rdma_links.is_empty() {
            return Err(ApiError::NotFound("No RDMA links discovered. Run `asmi` TUI scan first to discover RDMA topology.".into()));
        }
        hostfile_json = nm.hostfile_jaccl(&state.hostname);
    }

    let hosts: serde_json::Value = serde_json::from_str(&hostfile_json)
        .map_err(|e| ApiError::Internal(format!("Failed to build matrix: {e}")))?;

    let count = hosts.as_array().map(|a| a.len()).unwrap_or(0);
    if count == 0 {
        return Err(ApiError::NotFound("No RDMA-connected nodes found".into()));
    }

    // Optionally write to disk if 'write' flag is set
    let should_write = body.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
    let path = format!("~/hostfile-jaccl-{}node.json", count);

    if should_write {
        let pretty = serde_json::to_string_pretty(&hosts).unwrap_or_default();
        // Write via local file (asmi runs on the coordinator)
        let expanded = path.replace('~', &std::env::var("HOME").unwrap_or_default());
        std::fs::write(&expanded, &pretty)
            .map_err(|e| ApiError::Internal(format!("Failed to write hostfile: {e}")))?;
        tracing::info!(path = %expanded, nodes = count, "wrote JACCL hostfile");
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "action": if should_write { "generate" } else { "discover" },
        "hosts": hosts,
        "nodeCount": count,
        "path": if should_write { Some(&path) } else { None },
    })))
}

/// GET /models → cached local model file listing (now includes external volumes)
/// Each model is annotated with live serving info if it's currently loaded.
async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache = state.model_cache.read().await;
    let (models, scanned_at) = match cache.as_ref() {
        Some((m, t)) => (m.clone(), t.elapsed().as_secs()),
        None => (vec![], 0),
    };

    // Collect serving info: map of normalized-model-name → serving details
    let serving_map = collect_serving_map(&state).await;

    // Annotate each model with serving status
    let annotated: Vec<serde_json::Value> = models.iter().map(|m| {
        let mut v = serde_json::to_value(m).unwrap_or_default();
        if let Some(serving) = match_serving(&m.name, &m.path, &serving_map) {
            v.as_object_mut().map(|o| o.insert("serving".into(), serving));
        }
        v
    }).collect();

    Json(serde_json::json!({
        "models": annotated,
        "scan_age_seconds": scanned_at,
    }))
}

/// Serving info for a model currently loaded in memory.
#[derive(Debug, Clone, serde::Serialize)]
struct ServingInfo {
    port: u16,
    engine: String,
    pid: Option<u32>,
    managed: bool,
}

/// Collect all currently-serving models from managed servers and detected processes.
/// Returns a map of normalized name fragments → serving info.
async fn collect_serving_map(state: &AppState) -> Vec<(Vec<String>, ServingInfo)> {
    let mut entries = Vec::new();

    // 1. Managed servers
    for (&port, mgr) in state.serve_managers.read().await.iter() {
        let status = mgr.status().await;
        if let Some(ref model_id) = status.model {
            let info = ServingInfo {
                port,
                engine: format!("{:?}", status.engine).to_lowercase(),
                pid: status.pid,
                managed: true,
            };
            entries.push((normalize_model_id(model_id), info));
        }
    }

    // 2. Unmanaged processes from latest snapshot
    if let Some(snap) = state.snapshot.read().await.as_ref() {
        for proc in &snap.processes {
            let port = match proc.port {
                Some(p) => p,
                None => continue,
            };
            let engine = format!("{:?}", proc.framework).to_lowercase();

            // From server_models (probed /v1/models)
            for sm in &proc.server_models {
                let info = ServingInfo {
                    port,
                    engine: engine.clone(),
                    pid: Some(proc.pid),
                    managed: false,
                };
                entries.push((normalize_model_id(&sm.id), info));
            }
            // Fallback to ps-parsed model name
            if proc.server_models.is_empty() {
                if let Some(ref model) = proc.model {
                    let info = ServingInfo {
                        port,
                        engine: engine.clone(),
                        pid: Some(proc.pid),
                        managed: false,
                    };
                    entries.push((normalize_model_id(model), info));
                }
            }
        }
    }

    entries
}

/// Extract searchable name fragments from a model ID.
/// "mlx-community/Qwen3.5-122B-A10B-6bit" → ["qwen3.5-122b-a10b-6bit"]
/// "/Users/ma/Models/Qwen3.5-REAP-262B-A17B-4bit-mlx" → ["qwen3.5-reap-262b-a17b-4bit-mlx"]
fn normalize_model_id(id: &str) -> Vec<String> {
    let mut frags = Vec::new();
    // Last path component (works for both HF IDs and absolute paths)
    if let Some(last) = id.rsplit('/').next() {
        frags.push(last.to_lowercase());
    }
    // Full ID lowercased
    frags.push(id.to_lowercase());
    frags
}

/// Try to match a LocalModel against the serving map.
/// Returns the first match. Deduplicates by only matching once per model.
fn match_serving(
    name: &str,
    path: &std::path::Path,
    serving_map: &[(Vec<String>, ServingInfo)],
) -> Option<serde_json::Value> {
    let name_lower = name.to_lowercase();
    let path_lower = path.to_string_lossy().to_lowercase();
    let dir_name = path.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    for (frags, info) in serving_map {
        for frag in frags {
            if frag == &name_lower || frag == &dir_name || frag == &path_lower {
                return Some(serde_json::to_value(info).unwrap_or_default());
            }
        }
    }
    None
}

/// GET /cluster/models → cluster-wide model inventory.
/// Fetches /models from every node in the NodeMap in parallel, merges into
/// a unified list. Each model gets a `node` field. Includes local models too.
/// Response groups models by name across nodes for easy dedup analysis.
async fn cluster_models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_map = state.node_map.read().await;
    let local_hostname = state.hostname.clone();

    // Deduplicate nodes: resolve aliases to canonical names, then dedup
    let mut canonical_nodes: Vec<String> = node_map.nodes.iter()
        .map(|n| node_map.resolve(n).to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    canonical_nodes.sort();

    if canonical_nodes.is_empty() {
        // Fallback: just return local models with node tag
        let cache = state.model_cache.read().await;
        let models = match cache.as_ref() {
            Some((m, _)) => m.clone(),
            None => vec![],
        };
        let tagged: Vec<serde_json::Value> = models.iter().map(|m| {
            let mut v = serde_json::to_value(m).unwrap_or_default();
            v.as_object_mut().map(|o| o.insert("node".into(), serde_json::json!(local_hostname)));
            v
        }).collect();
        return Json(serde_json::json!({
            "models": tagged,
            "nodes_polled": 1,
            "nodes_responded": 1,
            "total_models": tagged.len(),
        }));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .unwrap_or_default();

    // Fetch /models from every node in parallel
    let futures: Vec<_> = canonical_nodes.iter().map(|node| {
        let client = client.clone();
        let node = node.clone();
        let local_hostname = local_hostname.clone();
        let model_cache = state.model_cache.clone();
        async move {
            // For local node, use the in-memory cache (faster, no HTTP round-trip)
            let is_local = node.eq_ignore_ascii_case(&local_hostname)
                || node == "hub" || node == "HUB" || node == "localhost";
            if is_local {
                let cache = model_cache.read().await;
                let models = match cache.as_ref() {
                    Some((m, _)) => m.clone(),
                    None => vec![],
                };
                return (local_hostname.clone(), Some(models));
            }

            // Remote node: try .local mDNS first, then Tailscale hostname
            let urls = vec![
                format!("http://{}.local:9090/models", node),
                format!("http://{}:9090/models", node),
            ];
            for url in &urls {
                if let Ok(resp) = client.get(url).send().await {
                    if resp.status().is_success() {
                        if let Ok(data) = resp.json::<serde_json::Value>().await {
                            if let Some(arr) = data.get("models").and_then(|v| v.as_array()) {
                                let models: Vec<asmi_core::LocalModel> = arr.iter()
                                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                                    .collect();
                                return (node, Some(models));
                            }
                        }
                    }
                }
            }
            (node, None)
        }
    }).collect();

    let results = futures::future::join_all(futures).await;

    // Merge all models with node attribution
    let mut all_models: Vec<serde_json::Value> = Vec::new();
    let mut nodes_responded = 0usize;
    let mut node_summaries: Vec<serde_json::Value> = Vec::new();

    for (node, models_opt) in &results {
        match models_opt {
            Some(models) => {
                nodes_responded += 1;
                // Dedup within node (old binaries may have case-insensitive dupes)
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut node_count = 0usize;
                let mut node_bytes = 0u64;
                for m in models {
                    let key = m.name.to_lowercase();
                    if seen.insert(key) {
                        let mut v = serde_json::to_value(m).unwrap_or_default();
                        v.as_object_mut().map(|o| o.insert("node".into(), serde_json::json!(node)));
                        node_bytes += m.size_bytes;
                        node_count += 1;
                        all_models.push(v);
                    }
                }
                node_summaries.push(serde_json::json!({
                    "node": node,
                    "online": true,
                    "model_count": node_count,
                    "total_bytes": node_bytes,
                }));
            }
            None => {
                node_summaries.push(serde_json::json!({
                    "node": node,
                    "online": false,
                    "model_count": 0,
                    "total_bytes": 0,
                }));
            }
        }
    }

    // Sort by name for consistent output
    all_models.sort_by(|a, b| {
        let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
        an.cmp(bn)
    });

    let total_bytes: u64 = all_models.iter()
        .filter_map(|m| m.get("size_bytes").and_then(|v| v.as_u64()))
        .sum();

    Json(serde_json::json!({
        "models": all_models,
        "nodes_polled": canonical_nodes.len(),
        "nodes_responded": nodes_responded,
        "total_models": all_models.len(),
        "total_bytes": total_bytes,
        "total_human": asmi_core::models::human_size(total_bytes),
        "node_summary": node_summaries,
    }))
}

/// GET /volumes → discover mounted external volumes with size info
async fn volumes_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let volumes = asmi_core::discover_volumes();
    Json(serde_json::json!({
        "hostname": state.hostname,
        "volumes": volumes,
    }))
}

/// GET /logs?name=mlx-server&lines=50 → tail server log files
async fn logs_handler(
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ApiError> {
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
            return Err(ApiError::BadRequest(format!(
                "unknown log name: {name} (known: mlx-server, mlx-vlm, vllm, asmi)"
            )));
        }
    };

    let expanded = log_path.replace('~', &std::env::var("HOME").unwrap_or_default());

    let content = std::fs::read_to_string(&expanded)
        .map_err(|e| ApiError::Internal(format!("could not read log {expanded}: {e}")))?;

    let all_lines: Vec<&str> = content.lines().collect();
    let start = all_lines.len().saturating_sub(lines);
    let tail: Vec<&str> = all_lines[start..].to_vec();
    Ok(Json(serde_json::json!({
        "name": name,
        "path": expanded,
        "lines": tail,
        "total_lines": all_lines.len(),
    })))
}

/// GET /runtime → Python/MLX/macOS version info (cached at startup)
async fn runtime_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    Ok(Json(serde_json::to_value(state.runtime.as_ref())
        .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?))
}

/// GET /health/setup → run setup validation checks
async fn setup_handler() -> Result<Json<serde_json::Value>, ApiError> {
    let checks = asmi_core::run_setup_checks().await;
    Ok(Json(serde_json::to_value(&checks)
        .map_err(|e| ApiError::Internal(format!("check failed: {e}")))?))
}

/// GET /arp → ARP table entries on Thunderbolt/bridge interfaces (en*).
/// Used by the web layer to correlate TB links across nodes:
/// if node A sees remote IP X on en3, and node B owns IP X on en5, that's a cable.
async fn arp_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let output = match tokio::process::Command::new("arp")
        .args(["-an"])
        .output()
        .await
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return Json(serde_json::json!({ "hostname": state.hostname, "peers": [] })),
    };

    // Parse: ? (169.254.89.126) at 0a:e0:af:... on en3 ifscope [ethernet]
    let mut peers = Vec::new();
    for line in output.lines() {
        // Skip incomplete entries and non-ethernet
        if line.contains("incomplete") || !line.contains("ifscope") {
            continue;
        }
        // Extract IP, interface
        let ip = line.split('(').nth(1).and_then(|s| s.split(')').next());
        let iface = line.split(" on ").nth(1).and_then(|s| s.split_whitespace().next());
        if let (Some(ip), Some(iface)) = (ip, iface) {
            // Only TB/bridge interfaces (en2-en31, bridge*)
            if (iface.starts_with("en") && iface.len() <= 4) || iface.starts_with("bridge") {
                peers.push(serde_json::json!({
                    "ip": ip,
                    "interface": iface,
                }));
            }
        }
    }

    Json(serde_json::json!({
        "hostname": state.hostname,
        "peers": peers,
    }))
}

/// Parse `networksetup -listallhardwareports` to build bus_index → interface name map.
/// "Thunderbolt N" hardware port → bus_index = N-1, device = next line's "Device: enX".
async fn build_tb_bus_map() -> serde_json::Map<String, serde_json::Value> {
    let output = tokio::process::Command::new("networksetup")
        .args(["-listallhardwareports"])
        .output()
        .await
        .ok();
    let mut map = serde_json::Map::new();
    if let Some(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains("Thunderbolt") && !line.contains("Bridge") {
                if let Some(n) = line.split("Thunderbolt ").nth(1).and_then(|s| s.trim().parse::<u8>().ok()) {
                    if let Some(dev_line) = lines.get(i + 1) {
                        if let Some(dev) = dev_line.strip_prefix("Device: ") {
                            map.insert((n - 1).to_string(), serde_json::json!(dev.trim()));
                        }
                    }
                }
            }
        }
    }
    map
}

/// Scan Thunderbolt device tree via system_profiler. Called by background cache loop.
pub async fn scan_thunderbolt(hostname: &str) -> serde_json::Value {
    // Get this node's hardware model identifier (e.g., "Mac15,14", "Mac16,9")
    // Used by the cable route to match TB device entries to specific hostnames.
    let hw_model = tokio::process::Command::new("sysctl")
        .args(["-n", "hw.model"])
        .output()
        .await
        .ok()
        .and_then(|o| if o.status.success() {
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        } else { None })
        .unwrap_or_default();

    let output = match tokio::process::Command::new("system_profiler")
        .args(["SPThunderboltDataType", "-json"])
        .output()
        .await
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => return serde_json::json!({
            "hostname": hostname, "hw_model": hw_model, "ports": [], "error": "system_profiler failed"
        }),
    };

    let json: serde_json::Value = match serde_json::from_slice(&output) {
        Ok(v) => v,
        Err(_) => return serde_json::json!({
            "hostname": hostname, "ports": [], "error": "json parse failed"
        }),
    };

    // Walk the SPThunderboltDataType array → each bus has _items for connected devices.
    let mut ports = Vec::new();
    if let Some(buses) = json.get("SPThunderboltDataType").and_then(|v| v.as_array()) {
        for bus in buses {
            let port_tag = bus.get("receptacle_1_tag");
            let receptacle = port_tag
                .and_then(|t| t.get("receptacle_id_key"))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<u64>().ok());
            let speed = port_tag
                .and_then(|t| t.get("current_speed_key"))
                .and_then(|v| v.as_str());
            let status = port_tag
                .and_then(|t| t.get("receptacle_status_key"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let connected = status.contains("connected");

            let domain_uuid = bus.get("domain_uuid_key")
                .and_then(|v| v.as_str()).unwrap_or("");
            let switch_uid = bus.get("switch_uid_key")
                .and_then(|v| v.as_str()).unwrap_or("");

            // Parse bus index from _name: "thunderboltusb4_bus_3" → 3
            let bus_index: Option<u8> = bus.get("_name")
                .and_then(|v| v.as_str())
                .and_then(|s| s.strip_prefix("thunderboltusb4_bus_"))
                .and_then(|n| n.parse().ok());

            // Extract peer domain_uuid from first connected _items[] entry
            let peer_domain_uuid = bus.get("_items")
                .and_then(|v| v.as_array())
                .and_then(|items| items.first())
                .and_then(|item| item.get("domain_uuid_key"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            fn find_apple_devices(items: &[serde_json::Value], results: &mut Vec<(String, String)>) {
                for item in items {
                    let vendor = item.get("vendor_name_key").and_then(|v| v.as_str()).unwrap_or("");
                    let device_name = item.get("device_name_key").and_then(|v| v.as_str()).unwrap_or("");
                    let display_name = item.get("_name").and_then(|v| v.as_str()).unwrap_or("");
                    if vendor.contains("Apple") && !device_name.is_empty() {
                        results.push((display_name.to_string(), device_name.to_string()));
                    }
                    if let Some(sub) = item.get("_items").and_then(|v| v.as_array()) {
                        find_apple_devices(sub, results);
                    }
                }
            }

            let mut devices = Vec::new();
            if let Some(items) = bus.get("_items").and_then(|v| v.as_array()) {
                find_apple_devices(items, &mut devices);
            }

            let mut port_json = serde_json::json!({
                "port": receptacle,
                "connected": connected || !devices.is_empty(),
                "speed": speed,
                "bus_index": bus_index,
                "domain_uuid": domain_uuid,
                "switch_uid": switch_uid,
                "peer_domain_uuid": if peer_domain_uuid.is_empty() { None } else { Some(peer_domain_uuid) },
            });
            if !devices.is_empty() {
                port_json["devices"] = serde_json::json!(devices.iter().map(|(name, model)| {
                    serde_json::json!({ "name": name, "model_id": model })
                }).collect::<Vec<_>>());
            }
            ports.push(port_json);
        }
    }

    // Bus-to-interface map from networksetup (e.g., "Thunderbolt 3" → "en4" → bus_index 2)
    let bus_map = build_tb_bus_map().await;

    serde_json::json!({
        "hostname": hostname,
        "hw_model": hw_model,
        "ports": ports,
        "bus_map": bus_map,
    })
}

/// GET /thunderbolt → cached Thunderbolt device tree (refreshed every 60s).
async fn thunderbolt_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache = state.thunderbolt_cache.read().await;
    match cache.as_ref() {
        Some((data, scanned_at)) => {
            let mut result = data.clone();
            if let Some(obj) = result.as_object_mut() {
                obj.insert("scan_age_seconds".to_string(),
                    serde_json::json!(scanned_at.elapsed().as_secs()));
            }
            Json(result)
        }
        None => Json(serde_json::json!({
            "hostname": state.hostname,
            "ports": [],
            "scan_age_seconds": null,
        })),
    }
}

/// GET /topology → cached topology report (JSON). Refreshed by background loop.
async fn topology_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let cache = state.topology_cache.read().await;
    match cache.as_ref() {
        Some((report, scanned_at)) => {
            let mut val = serde_json::to_value(report)
                .map_err(|e| ApiError::Internal(format!("serialize: {e}")))?;
            if let Some(obj) = val.as_object_mut() {
                obj.insert("scan_age_seconds".to_string(),
                    serde_json::json!(scanned_at.elapsed().as_secs()));
            }
            Ok(Json(val))
        }
        None => Err(ApiError::NotFound("topology not yet scanned — check back in ~60s".into())),
    }
}

/// GET /topology/dot → raw DOT graph output for visualization.
async fn topology_dot_handler(State(state): State<AppState>) -> Result<String, ApiError> {
    let cache = state.topology_cache.read().await;
    match cache.as_ref() {
        Some((report, _)) => Ok(report.raw_dot.clone()),
        None => Err(ApiError::NotFound("topology not yet scanned".into())),
    }
}

/// GET /topology/validate → mesh completeness check + JACCL readiness.
async fn topology_validate_handler(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let cache = state.topology_cache.read().await;
    match cache.as_ref() {
        Some((report, scanned_at)) => {
            Ok(Json(serde_json::json!({
                "nodes": report.nodes,
                "link_count": report.links.len(),
                "expected_links": report.nodes.len() * (report.nodes.len() - 1) / 2,
                "mesh_complete": report.mesh_complete,
                "missing_links": report.missing_links,
                "jaccl_ready": report.jaccl_ready,
                "jaccl_ready_subsets": report.jaccl_ready_subsets,
                "scan_age_seconds": scanned_at.elapsed().as_secs(),
            })))
        }
        None => Err(ApiError::NotFound("topology not yet scanned".into())),
    }
}

/// GET /health/network → validate local Thunderbolt network service names
async fn network_health_handler() -> Result<Json<serde_json::Value>, ApiError> {
    let status = asmi_core::validate_thunderbolt_services().await;
    Ok(Json(serde_json::to_value(&status)
        .map_err(|e| ApiError::Internal(format!("check failed: {e}")))?))
}

/// POST /health/network/fix → auto-repair Thunderbolt service names
async fn network_fix_handler() -> Result<Json<serde_json::Value>, ApiError> {
    let result = asmi_core::fix_thunderbolt_services().await;
    Ok(Json(serde_json::to_value(&result)
        .map_err(|e| ApiError::Internal(format!("fix failed: {e}")))?))
}

// ---------------------------------------------------------------------------
// Serve lifecycle endpoints (replaces mlx_daemon.py on port 19079)
// ---------------------------------------------------------------------------

/// Query params for serve endpoints — optional ?port= (defaults to 19080).
#[derive(Deserialize)]
struct ServeQuery {
    port: Option<u16>,
}

/// Default MLX server port (backwards compatible).
const DEFAULT_SERVE_PORT: u16 = 19080;

/// GET /serve/status → serve status.
/// No ?port= → returns all servers: {"servers": [...], "unmanaged": [...]}
/// ?port=19080 → returns single ServeStatus for that port
async fn serve_status_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if let Some(port) = q.port {
        // Single-port query
        let managers = state.serve_managers.read().await;
        let p = port;
        let mgr = managers.get(&p)
            .ok_or_else(|| ApiError::NotFound(format!("unknown port: {p}")))?;
        let status = mgr.status().await;
        Ok(Json(serde_json::to_value(&status)
            .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?))
    } else {
        // All ports → {"servers": [...], "unmanaged": [...]}
        let mut servers = Vec::new();
        {
            let managers = state.serve_managers.read().await;
            for mgr in managers.values() {
                servers.push(mgr.status().await);
            }
        }
        servers.sort_by_key(|s| s.port);

        // Collect managed PIDs for diffing against detected processes
        let mut managed_pids = std::collections::HashSet::new();
        for s in &servers {
            if let Some(pid) = s.pid {
                managed_pids.insert(pid);
            }
        }

        // Auto-adopt unmanaged processes — creates new managers for unknown ports
        let orphans = adopt_unmanaged(&state, &managed_pids).await;

        // Re-read statuses after adoption (managers may have been created)
        let mut servers = Vec::new();
        {
            let managers = state.serve_managers.read().await;
            for mgr in managers.values() {
                servers.push(mgr.status().await);
            }
        }
        servers.sort_by_key(|s| s.port);

        Ok(Json(serde_json::json!({
            "servers": servers,
            "unmanaged": orphans,
        })))
    }
}

/// Auto-adopt unmanaged model processes into serve managers.
/// If a process is on a managed port → adopt into that manager.
/// If on an unknown port → dynamically create a new manager and adopt.
/// Returns any processes that truly couldn't be adopted (no port).
async fn adopt_unmanaged(
    state: &AppState,
    managed_pids: &std::collections::HashSet<u32>,
) -> Vec<asmi_core::UnmanagedProcess> {
    let snap = state.snapshot.read().await;
    let processes = match snap.as_ref() {
        Some(s) => &s.processes,
        None => return Vec::new(),
    };

    let mut orphans = Vec::new();
    for proc in processes {
        if !is_model_server_framework(proc.framework) {
            continue;
        }
        if managed_pids.contains(&proc.pid) {
            continue;
        }

        let model: Option<String> = if !proc.server_models.is_empty() {
            Some(proc.server_models[0].id.clone())
        } else {
            proc.model.clone()
        };

        let engine = framework_to_engine(proc.framework);

        if let Some(port) = proc.port {
            // Try existing manager first
            {
                let managers = state.serve_managers.read().await;
                if let Some(mgr) = managers.get(&port) {
                    mgr.adopt_external(proc.pid, model, engine).await;
                    continue;
                }
            }

            // No manager for this port — create an empty one and adopt the external process.
            // Use new() not restore() — restore() spawns a bare server which would
            // conflict with the already-running external process on this port.
            tracing::info!(port, pid = proc.pid, "auto-adopting server on new port");
            let new_mgr = crate::serve::ServeManager::new(port, engine);
            new_mgr.adopt_external(proc.pid, model, engine).await;
            {
                let mut managers = state.serve_managers.write().await;
                managers.insert(port, new_mgr);
            }
            continue;
        }

        // No port at all — truly unmanageable
        let models: Vec<String> = if !proc.server_models.is_empty() {
            proc.server_models.iter().map(|m| m.id.clone()).collect()
        } else if let Some(ref m) = proc.model {
            vec![m.clone()]
        } else {
            Vec::new()
        };
        let launchd = crate::launchd::describe_pid(proc.pid).await;
        orphans.push(asmi_core::UnmanagedProcess {
            pid: proc.pid,
            port: proc.port,
            engine: proc.framework.to_string(),
            models,
            source: "external",
            launchd,
        });
    }
    orphans.sort_by_key(|u| u.pid);
    orphans
}

/// Map ProcessFramework to ServeEngine for adoption.
fn framework_to_engine(fw: asmi_core::ProcessFramework) -> asmi_core::ServeEngine {
    match fw {
        asmi_core::ProcessFramework::MlxLm | asmi_core::ProcessFramework::MlxLmShare
            => asmi_core::ServeEngine::MlxLm,
        asmi_core::ProcessFramework::MlxVlm => asmi_core::ServeEngine::MlxVlm,
        asmi_core::ProcessFramework::VllmMlx => asmi_core::ServeEngine::VllmMlx,
        asmi_core::ProcessFramework::DFlashProc => asmi_core::ServeEngine::DFlash,
        _ => asmi_core::ServeEngine::MlxLm,
    }
}

/// Whether a `ProcessFramework` represents a model-serving process
/// (as opposed to a watchdog, distributed launcher, or unknown binary).
fn is_model_server_framework(fw: asmi_core::ProcessFramework) -> bool {
    matches!(
        fw,
        asmi_core::ProcessFramework::MlxLm
            | asmi_core::ProcessFramework::MlxLmShare
            | asmi_core::ProcessFramework::MlxVlm
            | asmi_core::ProcessFramework::VllmMlx
            | asmi_core::ProcessFramework::MlxAudio
            | asmi_core::ProcessFramework::DFlashProc
    )
}

/// POST /serve/load → begin loading a model.
/// Infers port from ?port= or engine default (MlxVlm→19082, else→19080).
async fn serve_load_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
    Json(req): Json<asmi_core::LoadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Allow None model_path — starts an idle server (LM-Studio-style JIT).
    // Reject empty string though, that's always a mistake.
    if let Some(p) = &req.model_path {
        if p.is_empty() {
            return Err(ApiError::BadRequest("model_path must not be empty (omit field for idle start)".into()));
        }
    }

    // Infer port: explicit ?port= > engine default (from env or built-in)
    let port = q.port.unwrap_or(crate::serve::port_for_engine(req.engine));

    // Get or create a manager for this port
    {
        let managers = state.serve_managers.read().await;
        if !managers.contains_key(&port) {
            drop(managers);
            let new_mgr = crate::serve::ServeManager::restore(port, req.engine).await;
            state.serve_managers.write().await.insert(port, new_mgr);
        }
    }

    // Start peer heartbeat for JACCL distributed sessions
    let backend = crate::serve::resolve_backend(&req.backend, req.hostfile.as_deref());
    if backend == asmi_core::ServeBackend::Jaccl {
        let hf_path = req
            .hostfile
            .clone()
            .unwrap_or_else(|| crate::serve::default_hostfile().to_string_lossy().to_string());
        let peers = crate::serve::parse_hostfile_peers(&hf_path, &state.hostname);
        if !peers.is_empty() {
            state
                .peer_heartbeat
                .start(peers, 9090, state.serve_managers.clone(), state.share_manager.clone())
                .await;
        }
    } else {
        state.peer_heartbeat.stop().await;
    }

    let engine = req.engine;
    let managers = state.serve_managers.read().await;
    let mgr = managers.get(&port)
        .ok_or_else(|| ApiError::Internal(format!("manager for port {port} disappeared")))?;
    mgr.load(req).await;
    Ok(Json(serde_json::json!({"ok": true, "state": "loading", "engine": engine, "port": port})))
}

/// POST /serve/stop → stop the running server on a port.
async fn serve_stop_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let port = q.port.unwrap_or(DEFAULT_SERVE_PORT);
    let managers = state.serve_managers.read().await;
    let mgr = managers.get(&port)
        .ok_or_else(|| ApiError::NotFound(format!("unknown port: {port}")))?;
    mgr.stop().await;
    drop(managers);
    state.peer_heartbeat.stop().await;
    Ok(Json(serde_json::json!({"ok": true, "port": port})))
}

// ---------------------------------------------------------------------------
// launchd agent actions — disable/enable a managed plist by label.
// Protected labels (com.asmi.*, com.r1o.watchdog) are rejected at the guard.
// ---------------------------------------------------------------------------

/// Request body for `/launchd/disable` and `/launchd/enable`.
#[derive(Deserialize)]
struct LaunchdAction {
    label: String,
}

/// Map a `LaunchdError` to an axum `Response` with the right status code.
fn launchd_err_response(e: crate::launchd::LaunchdError) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    match e {
        crate::launchd::LaunchdError::Protected(label) => (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": format!("protected label: {label}")})),
        )
            .into_response(),
        other => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": other.to_string()})),
        )
            .into_response(),
    }
}

/// POST /launchd/disable — `launchctl disable` + `bootout` for the given label.
///
/// Body: `{"label": "com.foo.bar"}`
/// - 200 `{"ok": true}` on success
/// - 403 `{"error": "protected label: …"}` for asmi / watchdog
/// - 500 `{"error": "…"}` for any other launchctl failure
async fn launchd_disable_handler(
    State(_state): State<AppState>,
    Json(body): Json<LaunchdAction>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match crate::launchd::disable(&body.label).await {
        Ok(()) => (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({"ok": true})),
        )
            .into_response(),
        Err(e) => launchd_err_response(e),
    }
}

/// POST /launchd/enable — `launchctl enable` + bootstrap the plist.
///
/// Mirror of `launchd_disable_handler`; same response contract.
async fn launchd_enable_handler(
    State(_state): State<AppState>,
    Json(body): Json<LaunchdAction>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match crate::launchd::enable(&body.label).await {
        Ok(()) => (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({"ok": true})),
        )
            .into_response(),
        Err(e) => launchd_err_response(e),
    }
}

// ---------------------------------------------------------------------------
// Hermes status + lifecycle — proxy to hermes-api on localhost.
//
// Phase 3 of the dmg-unified plan lifts the read-only endpoint into asmi so
// iOS can probe "is this an asmi-only node or asmi+hermes node?" from a
// single origin (asmi:9090). Phase 4 (Feature E) adds the write endpoints
// (POST /hermes/restart, POST /hermes/config) on top of that.
//
// Phase 4 design constraint (executed-state finding 2026-05-28): the
// MLX-side LaunchAgent label is NOT hardcoded. It's read from
// `~/.r1o/cluster.json` key `mlx_label` (if present); else falls back to
// scanning `launchctl list | grep -E "com.r1o.(mlx-lm|dflash-retriever|mlx-vlm)"`.
// This handles the live state where the canonical label is
// `com.r1o.dflash-retriever` instead of plan-v5's assumed `com.r1o.mlx-lm`.
// ---------------------------------------------------------------------------

/// Check if a peer IP is trusted for mutating hermes endpoints.
///
/// Trusted sources:
/// - Loopback (127.0.0.0/8, ::1) — local processes (web app, CLI).
/// - Tailscale CGNAT (100.64.0.0/10, i.e. 100.64.0.0–100.127.255.255) —
///   only authenticated Tailscale peers receive IPs in this range. This is
///   how the iOS app reaches asmi over the Tailnet.
///
/// This replaces the previous loopback-only guard that blocked all remote
/// peers, including the iOS client which is a legitimate Tailnet peer.
fn is_trusted_peer(addr: &std::net::SocketAddr) -> bool {
    let ip = addr.ip();
    if ip.is_loopback() {
        return true;
    }
    match ip {
        std::net::IpAddr::V4(v4) => {
            let octets = v4.octets();
            // Tailscale CGNAT: 100.64.0.0/10 → first octet 100, second 64..127
            octets[0] == 100 && (64..=127).contains(&octets[1])
        }
        std::net::IpAddr::V6(_) => false,
    }
}

/// GET /hermes/status → `{running, pid, port, model, last_request_at}` for the
/// local hermes-api daemon. Reads `http://localhost:41104/health` (cheap, no
/// side effects). Returns 404 if hermes-api is unreachable (a normal state
/// for asmi-only nodes — iOS uses the 404 to route via /serve/status only).
async fn hermes_status_handler(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    const HERMES_PORT: u16 = 41104;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(800))
        .build()
        .map_err(|e| ApiError::Internal(format!("reqwest build: {e}")))?;

    let url = format!("http://localhost:{HERMES_PORT}/health");
    let resp = client.get(&url).send().await.map_err(|e| {
        // Connection refused / timeout — treat as "not running" with NOT_FOUND
        // so iOS can fall back to /serve/status only.
        ApiError::NotFound(format!("hermes-api unreachable at {url}: {e}"))
    })?;

    if !resp.status().is_success() {
        return Err(ApiError::NotFound(format!(
            "hermes-api returned {} from /health",
            resp.status()
        )));
    }

    // Pass through hermes-api's /health payload — keys vary across versions
    // (older: {status, service, hermes_binary}, newer adds {pid, model}).
    // We wrap it with `running: true` + `port` for iOS's expected envelope.
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ApiError::Internal(format!("hermes /health not JSON: {e}")))?;

    // Best-effort label resolution so iOS can see what MLX service is bound
    // here. Adds a small extra cost (~1ms) but avoids a second round trip.
    let mlx_label = resolve_mlx_label().await;

    let payload = serde_json::json!({
        "running": true,
        "port": HERMES_PORT,
        "pid": body.get("pid").cloned().unwrap_or(serde_json::Value::Null),
        "model": body.get("model").cloned().unwrap_or(serde_json::Value::Null),
        "last_request_at": body
            .get("last_request_at")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "mlx_label": mlx_label,
        "raw": body,
    });
    Ok(Json(payload))
}

/// Resolve the MLX-side LaunchAgent label that Phase 4 actions target.
///
/// Resolution order:
/// 1. `~/.r1o/cluster.json` key `mlx_label` (string).
/// 2. Scan `launchctl list` for the first hit of
///    `com.r1o.(dflash-retriever|mlx-lm|mlx-vlm)` (running OR registered).
/// 3. `None` — no label found; lifecycle endpoints will 404.
///
/// Phase 0 of the dmg-unified plan discovered that the live label on hub is
/// `com.r1o.dflash-retriever`, not the plan-assumed `com.r1o.mlx-lm`. This
/// resolver makes the Phase 4 endpoints work on whatever label is actually
/// installed.
async fn resolve_mlx_label() -> Option<String> {
    // Try cluster.json first.
    if let Some(home) = dirs::home_dir() {
        let path = home.join(".r1o").join("cluster.json");
        if let Ok(bytes) = tokio::fs::read(&path).await {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(label) = v.get("mlx_label").and_then(|x| x.as_str()) {
                    if !label.is_empty() {
                        return Some(label.to_string());
                    }
                }
            }
        }
    }

    // Fall back to launchctl scan.
    let output = tokio::process::Command::new("launchctl")
        .arg("list")
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        // launchctl list format: PID\tStatus\tLabel
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let label = parts[2].trim();
        if matches!(
            label,
            "com.r1o.dflash-retriever" | "com.r1o.mlx-lm" | "com.r1o.mlx-vlm"
        ) {
            return Some(label.to_string());
        }
    }
    None
}

/// POST /hermes/restart — disable + enable the `com.ace.hermes-api`
/// LaunchAgent, then poll `:41104/health` until it responds (or timeout).
///
/// Returns 200 with `{ok: true, restart_secs: <float>}` on success, 504 if
/// hermes-api does not come back within 15s, and 500 for any launchctl error.
///
/// Note: this is the HERMES side. The MLX-side service (label dynamically
/// resolved) is NOT restarted here. Use the existing
/// `POST /launchd/disable` + `POST /launchd/enable` with the resolved label
/// directly for that.
///
/// **Peer-gated:** allows loopback + Tailscale CGNAT peers (see
/// `is_trusted_peer`). The original loopback-only guard (PR #23
/// adversarial-critic verdict) blocked iOS which is a legitimate Tailnet
/// peer. Tailscale CGNAT IPs are only assigned to authenticated peers, so
/// this is equivalent to "is this caller on our Tailnet?"
async fn hermes_restart_handler(
    State(_state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    const HERMES_LABEL: &str = "com.ace.hermes-api";
    const HERMES_PORT: u16 = 41104;
    const POLL_TIMEOUT_SECS: u64 = 15;

    // Safety: only allow from loopback or Tailscale CGNAT peers.
    // This handler triggers launchctl disable+enable, which can knock
    // hermes-api offline if run in a tight loop — must not be reachable
    // from arbitrary LAN/internet hosts.
    if !is_trusted_peer(&addr) {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "hermes/restart requires a trusted peer (loopback or Tailscale)"
        }))).into_response();
    }

    // 1. disable
    if let Err(e) = crate::launchd::disable(HERMES_LABEL).await {
        return launchd_err_response(e);
    }

    // 2. enable
    if let Err(e) = crate::launchd::enable(HERMES_LABEL).await {
        return launchd_err_response(e);
    }

    // 3. poll /health every 500ms until reachable or timeout
    let started = std::time::Instant::now();
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("reqwest build: {e}")})),
            )
                .into_response();
        }
    };
    let url = format!("http://localhost:{HERMES_PORT}/health");
    let timeout = std::time::Duration::from_secs(POLL_TIMEOUT_SECS);
    loop {
        if started.elapsed() > timeout {
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({
                    "error": format!("hermes-api did not respond on :{HERMES_PORT} within {POLL_TIMEOUT_SECS}s after enable"),
                    "elapsed_secs": started.elapsed().as_secs_f32(),
                })),
            )
                .into_response();
        }
        if let Ok(r) = client.get(&url).send().await {
            if r.status().is_success() {
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "restart_secs": started.elapsed().as_secs_f32(),
                    })),
                )
                    .into_response();
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

/// POST /hermes/config — proxy `PATCH http://localhost:41104/v1/config` so
/// iOS / desktop can update Hermes config (e.g. model swap) from a single
/// origin (asmi:9090) without needing to know hermes-api's port.
///
/// Request body is forwarded as-is. Response body is forwarded as-is with
/// the same status code. Errors at the proxy layer (connect refused,
/// timeout) return 502 Bad Gateway.
///
/// **Peer-gated:** allows loopback + Tailscale CGNAT peers (see
/// `is_trusted_peer`). The original loopback-only guard blocked iOS which
/// needs to PATCH hermes config (e.g. model swap) over the Tailnet. The
/// upstream hermes-api `update_config()` is itself reachable only on
/// 127.0.0.1, so this asmi proxy is the only path a remote peer can
/// mutate hermes config.
async fn hermes_config_handler(
    State(_state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    const HERMES_PORT: u16 = 41104;

    // Safety: only allow from loopback or Tailscale CGNAT peers.
    // Mutates persistent config on disk via the hermes-api PATCH proxy —
    // must not be reachable from arbitrary LAN/internet hosts.
    if !is_trusted_peer(&addr) {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "hermes/config requires a trusted peer (loopback or Tailscale)"
        }))).into_response();
    }

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("reqwest build: {e}")})),
            )
                .into_response();
        }
    };

    let url = format!("http://localhost:{HERMES_PORT}/v1/config");
    let upstream = match client
        .patch(&url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("hermes-api unreachable: {e}")})),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let bytes = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("hermes-api read body: {e}")})),
            )
                .into_response();
        }
    };

    // Pass through the upstream status + body. The upstream sends JSON; we
    // copy the bytes verbatim so any future schema changes don't need code
    // here.
    (
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK),
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Autoresearch benchmark endpoints
// ---------------------------------------------------------------------------

/// GET /autoresearch/gate → benchmark-readiness check.
///
/// Returns a JSON object with thermal, memory, power, ghost-process, and RDMA
/// status. The `ready` field is true only when ALL conditions are met:
/// non-throttled thermals, >20 GB available RAM, AC power, zero ghost MLX
/// procs, and no snapshot errors.
async fn autoresearch_gate_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    let snap = snap.as_ref().ok_or_else(|| {
        ApiError::NotFound(format!("no snapshot yet (hostname: {})", state.hostname))
    })?;

    let cpu_temp_c = snap.cpu_temp_c;
    let gpu_temp_c = snap.gpu_temp_c;

    // macOS throttles around 95 C. NodeSnapshot doesn't carry a direct throttle
    // flag yet, so we infer from temperature.
    let cpu_throttled = cpu_temp_c.map(|t| t >= 95.0).unwrap_or(false);
    let gpu_throttled = gpu_temp_c.map(|t| t >= 95.0).unwrap_or(false);

    let mem_available_gb =
        (snap.ram_total_bytes.saturating_sub(snap.ram_app_bytes)) as f64 / 1_073_741_824.0;

    let power_state = snap.power_source.clone();
    // Desktop-class Macs (Mac Studio, Mac Pro, Mac mini) don't expose battery
    // state via IOReport, so `power_source` stays None. Treat those hosts as
    // always-AC — a desktop Mac physically cannot run on battery. Laptops
    // (MacBook Pro/Air) always get a real reading.
    let is_desktop_chassis = match snap.model_name.as_deref() {
        Some(m) => {
            let lower = m.to_ascii_lowercase();
            lower.contains("mac studio")
                || lower.contains("mac pro")
                || lower.contains("mac mini")
                || lower.contains("imac")
        }
        None => false,
    };
    let on_ac = match power_state.as_deref() {
        Some("AC") => true,
        Some(_) => false,
        None => is_desktop_chassis,
    };

    // Ghost procs: MLX-framework processes whose PID is NOT tracked by any
    // ServeManager.  These leak GPU memory and invalidate benchmarks.
    let managed_pids: std::collections::HashSet<u32> = {
        let managers = state.serve_managers.read().await;
        let mut pids = std::collections::HashSet::with_capacity(managers.len());
        for mgr in managers.values() {
            if let Some(pid) = mgr.status().await.pid {
                pids.insert(pid);
            }
        }
        pids
    };
    let ghost_procs = snap
        .processes
        .iter()
        .filter(|p| is_model_server_framework(p.framework) && !managed_pids.contains(&p.pid))
        .count() as u32;

    let rdma_peers_up = snap
        .rdma
        .as_ref()
        .map(|r| r.active_count() as u32)
        .unwrap_or(0);

    let ready =
        !cpu_throttled && !gpu_throttled && mem_available_gb > 20.0 && on_ac && ghost_procs == 0;

    Ok(Json(serde_json::json!({
        "hostname": state.hostname,
        "ready": ready,
        "cpu_temp_c": cpu_temp_c,
        "gpu_temp_c": gpu_temp_c,
        "cpu_throttled": cpu_throttled,
        "gpu_throttled": gpu_throttled,
        "mem_available_gb": mem_available_gb,
        "power_state": power_state,
        "on_ac": on_ac,
        "ghost_procs": ghost_procs,
        "rdma_peers_up": rdma_peers_up,
        "ts": chrono::Utc::now().timestamp(),
    })))
}

/// POST /autoresearch/reset → cold-start every benchmark-relevant resource.
///
/// Stops all serve managers, kills ghost MLX procs, flushes the file cache,
/// and waits up to 60 s for thermal recovery (< 70 C).
async fn autoresearch_reset_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    use tokio::process::Command;

    let start = std::time::Instant::now();
    let mut errors: Vec<String> = Vec::new();

    // 1. Stop all managed serve sessions.
    let mgr_ports: Vec<u16> = state.serve_managers.read().await.keys().copied().collect();
    let mut stopped_ports: Vec<u16> = Vec::new();
    for &port in &mgr_ports {
        let managers = state.serve_managers.read().await;
        if let Some(mgr) = managers.get(&port) {
            mgr.stop().await;
            stopped_ports.push(port);
        }
    }

    // 2. Stop peer heartbeat and share manager.
    state.peer_heartbeat.stop().await;
    state.share_manager.emergency_stop().await;

    // 3. Kill any ghost MLX processes.
    if let Err(e) = Command::new("pkill").arg("-f").arg("mlx_lm").output().await {
        errors.push(format!("pkill mlx_lm: {e}"));
    }
    if let Err(e) = Command::new("pkill").arg("-f").arg("mlx.launch").output().await {
        errors.push(format!("pkill mlx.launch: {e}"));
    }

    // 3b. Wait up to 30 s for those pkilled procs to actually vanish before
    // flushing caches. pkill returns immediately; the OS takes time to reap.
    // If we proceed too early, the next benchmark inherits a node with MLX
    // still holding Metal buffers and we get phantom OOMs on the next iter.
    let proc_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut procs_alive_at_deadline = 0u32;
    loop {
        let snap = state.snapshot.read().await;
        let alive = snap
            .as_ref()
            .map(|s| {
                s.processes
                    .iter()
                    .filter(|p| is_model_server_framework(p.framework))
                    .count() as u32
            })
            .unwrap_or(0);
        drop(snap);
        if alive == 0 {
            break;
        }
        if std::time::Instant::now() >= proc_deadline {
            procs_alive_at_deadline = alive;
            errors.push(format!(
                "mlx proc reap timeout: {alive} still alive after 30 s"
            ));
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    // 4. Flush file cache.
    if let Err(e) = Command::new("purge").output().await {
        errors.push(format!("purge: {e}"));
    }

    // 5. Wait for thermal recovery (both < 70 C), bounded to 60 s.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let snap = state.snapshot.read().await;
        if let Some(snap) = snap.as_ref() {
            let cpu_ok = snap.cpu_temp_c.map(|t| t < 70.0).unwrap_or(true);
            let gpu_ok = snap.gpu_temp_c.map(|t| t < 70.0).unwrap_or(true);
            if cpu_ok && gpu_ok {
                break;
            }
        } else {
            // No snapshot available — nothing to wait on.
            break;
        }
        drop(snap);
        if std::time::Instant::now() >= deadline {
            errors.push("thermal recovery timeout (60 s)".into());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "reset": true,
        "hostname": state.hostname,
        "stopped_ports": stopped_ports,
        "mlx_procs_alive": procs_alive_at_deadline,
        "duration_ms": duration_ms,
        "errors": errors,
        "ts": chrono::Utc::now().timestamp(),
    })))
}

/// POST /serve/reload → reload the current model on a port.
async fn serve_reload_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let port = q.port.unwrap_or(DEFAULT_SERVE_PORT);
    let managers = state.serve_managers.read().await;
    let mgr = managers.get(&port)
        .ok_or_else(|| ApiError::NotFound(format!("unknown port: {port}")))?;
    let status = mgr.status().await;
    let model = status.model
        .ok_or_else(|| ApiError::NotFound("no model loaded".into()))?;
    let req = asmi_core::LoadRequest {
        model_path: Some(model),
        backend: status.backend.to_string(),
        hostfile: None,
        engine: status.engine,
        ..Default::default()
    };
    mgr.load(req).await;
    Ok(Json(serde_json::json!({"ok": true, "state": "loading", "port": status.port})))
}

/// POST /serve/share — start a distributed share session.
async fn serve_share_handler(
    State(state): State<AppState>,
    Json(req): Json<asmi_core::ShareRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.model_path.is_empty() {
        return Err(ApiError::BadRequest("model_path required".into()));
    }
    // Start peer heartbeat for JACCL distributed sessions
    let backend = crate::serve::resolve_backend(&req.backend, req.hostfile.as_deref());
    if backend == asmi_core::ServeBackend::Jaccl {
        let hf_path = req
            .hostfile
            .clone()
            .unwrap_or_else(|| crate::serve::default_hostfile().to_string_lossy().to_string());
        let peers = crate::serve::parse_hostfile_peers(&hf_path, &state.hostname);
        if !peers.is_empty() {
            state
                .peer_heartbeat
                .start(peers, 9090, state.serve_managers.clone(), state.share_manager.clone())
                .await;
        }
    }
    state.share_manager.start(req).await;
    Ok(Json(serde_json::json!({"ok": true, "state": "loading"})))
}

/// GET /serve/share/status — share session status.
async fn serve_share_status_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let status = state.share_manager.status().await;
    Ok(Json(serde_json::to_value(&status)
        .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?))
}

/// POST /serve/share/stop — stop the running share session.
async fn serve_share_stop_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    state.share_manager.stop().await;
    state.peer_heartbeat.stop().await;
    Json(serde_json::json!({"ok": true}))
}

/// POST /serve/distributed/join — join a distributed inference session as a worker.
///
/// Called by the hub's asmi to recruit peers. Each peer starts a local
/// mlx_lm.server process with JACCL env vars. No SSH — asmi handles it.
async fn serve_distributed_join_handler(
    State(state): State<AppState>,
    Json(req): Json<DistributedJoinRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    use tokio::process::Command;

    let py = crate::daemon::resolve_python().to_string();
    let model_path = if req.model_path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            format!("{}/{}", home.display(), &req.model_path[2..])
        } else {
            req.model_path.clone()
        }
    } else {
        req.model_path.clone()
    };

    // Write IBV devices to temp file (JACCL reads from file path)
    let ibv_tmp = std::env::temp_dir().join(format!("asmi-ibv-{}.json", req.rank));
    tokio::fs::write(&ibv_tmp, &req.ibv_devices).await
        .map_err(|e| ApiError::Internal(format!("write ibv devices: {e}")))?;

    // Build command: python3 -m mlx_lm server --model <path> --port <port> --host 0.0.0.0
    let mut cmd = Command::new(&py);
    cmd.arg("-m").arg("mlx_lm").arg("server")
        .arg("--model").arg(&model_path)
        .arg("--port").arg(req.port.to_string())
        .arg("--host").arg("127.0.0.1");

    // Set distributed env vars (backend-specific)
    cmd.env("MLX_RANK", req.rank.to_string())
        .env("MLX_WORLD_SIZE", req.world_size.to_string())
        .env("MLX_METAL_FAST_SYNCH", "1");

    if req.backend == "jaccl" {
        cmd.env("MLX_DISTRIBUTED_BACKEND", "jaccl")
            .env("MLX_JACCL_COORDINATOR", &req.coordinator)
            .env("MLX_IBV_DEVICES", ibv_tmp.to_string_lossy().to_string());
    } else {
        // Ring backend: MLX_HOSTFILE must point to a temp file containing the ring JSON
        cmd.env("MLX_DISTRIBUTED_BACKEND", "ring");
        if let Some(ref hf) = req.ring_hostfile {
            let hf_tmp = std::env::temp_dir().join(format!("asmi-ring-{}.json", req.rank));
            std::fs::write(&hf_tmp, hf)
                .map_err(|e| ApiError::Internal(format!("write ring hostfile: {e}")))?;
            cmd.env("MLX_HOSTFILE", hf_tmp.to_string_lossy().to_string());
        }
    }

    // Log file
    let log_path = format!("/tmp/r1o-distributed-rank{}.log", req.rank);
    let log_file = std::fs::OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(&log_path)
        .map_err(|e| ApiError::Internal(format!("open log: {e}")))?;
    let log_stderr = log_file.try_clone()
        .map_err(|e| ApiError::Internal(format!("clone log: {e}")))?;

    cmd.stdout(log_file).stderr(log_stderr).kill_on_drop(false);

    tracing::info!(
        rank = req.rank,
        world_size = req.world_size,
        model = %model_path,
        coordinator = %req.coordinator,
        "joining distributed session"
    );

    let child = cmd.spawn()
        .map_err(|e| ApiError::Internal(format!("spawn worker: {e}")))?;
    let pid = child.id().unwrap_or(0);

    // Store in share manager so it can be stopped
    state.share_manager.adopt_child(child, &model_path, asmi_core::ServeBackend::Jaccl).await;

    Ok(Json(serde_json::json!({
        "ok": true,
        "rank": req.rank,
        "pid": pid,
        "log": log_path,
        "hostname": state.hostname,
    })))
}

#[derive(Deserialize)]
struct DistributedJoinRequest {
    model_path: String,
    rank: u32,
    world_size: u32,
    coordinator: String,
    backend: String,
    ibv_devices: String,
    port: u16,
    /// Ring backend: JSON hostfile string e.g. [["ip1:port1"], ["ip2:port2"]]
    ring_hostfile: Option<String>,
}

// ---------------------------------------------------------------------------
// Watchdog endpoints
// ---------------------------------------------------------------------------

/// GET /watchdog/peers → RDMA peer heartbeat status.
async fn peer_heartbeat_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let status = state.peer_heartbeat.status().await;
    Json(serde_json::to_value(&status).unwrap_or_default())
}

/// GET /watchdog → full WatchdogReport (processes + gpu_lock + peer_heartbeat).
async fn watchdog_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let report = state.watchdog.report().await;
    Json(serde_json::to_value(&report).unwrap_or_default())
}

/// GET /watchdog/gpu-lock → GPU Lock detection status.
async fn gpu_lock_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let status = state.watchdog.gpu_lock_status().await;
    Json(serde_json::to_value(&status).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// v0.5 endpoints: process kill, process tree, disk, network
// ---------------------------------------------------------------------------

/// Query params for POST /processes/:pid/kill
#[derive(Deserialize)]
struct KillParams {
    /// "term" (default) or "kill"
    #[serde(default = "default_signal")]
    signal: String,
    /// Optional remote node hostname (default: local)
    node: Option<String>,
}

fn default_signal() -> String {
    "term".to_string()
}

/// POST /processes/:pid/kill — kill a process by PID.
///
/// Safety: refuses PID 0, PID 1, and the asmi daemon's own PID.
/// Supports remote kill via SSH when ?node=hostname is specified.
async fn kill_process_handler(
    State(state): State<AppState>,
    Path(pid): Path<u32>,
    Query(params): Query<KillParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Safety checks
    if pid == 0 || pid == 1 {
        return Err(ApiError::BadRequest(format!("refusing to kill PID {pid} (system process)")));
    }
    let my_pid = std::process::id();
    if pid == my_pid {
        return Err(ApiError::BadRequest("refusing to kill the asmi daemon itself".into()));
    }

    let signal_name = match params.signal.to_lowercase().as_str() {
        "term" | "sigterm" | "15" => "TERM",
        "kill" | "sigkill" | "9" => "KILL",
        other => return Err(ApiError::BadRequest(format!(
            "unsupported signal: {other} (use 'term' or 'kill')"
        ))),
    };

    let node = params.node.as_deref().unwrap_or("local");
    let is_local = node == "local" || node == state.hostname;

    if is_local {
        // Local kill via nix
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        let nix_signal = match signal_name {
            "KILL" => Signal::SIGKILL,
            _ => Signal::SIGTERM,
        };
        let nix_pid = Pid::from_raw(pid as i32);
        kill(nix_pid, nix_signal).map_err(|e| {
            ApiError::BadRequest(format!("kill({pid}, SIG{signal_name}) failed: {e}"))
        })?;
    } else {
        // Remote kill via SSH
        let cmd = format!("kill -{signal_name} {pid}");
        let output = tokio::process::Command::new("ssh")
            .args(["-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", node, &cmd])
            .output()
            .await
            .map_err(|e| ApiError::Internal(format!("SSH to {node} failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ApiError::BadRequest(format!(
                "kill on {node} failed: {stderr}"
            )));
        }
    }

    tracing::warn!(pid, signal = signal_name, node, "process killed via API");

    Ok(Json(serde_json::json!({
        "ok": true,
        "pid": pid,
        "signal": format!("SIG{signal_name}"),
        "node": node,
    })))
}

/// Query params for GET /processes/tree
#[derive(Deserialize)]
struct TreeParams {
    /// Minimum CPU% to include (default: 1.0)
    min_cpu: Option<f64>,
    /// Minimum MEM% to include (default: 0.5)
    min_mem: Option<f64>,
}

/// GET /port/:port — find what process is listening on a given port.
async fn port_lookup_handler(
    Path(port): Path<u16>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let output = tokio::process::Command::new("lsof")
        .args(["-t", "-sTCP:LISTEN", "-i", &format!("TCP:{}", port)])
        .output()
        .await
        .map_err(|e| ApiError::Internal(format!("lsof failed: {e}")))?;

    let pid_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if pid_str.is_empty() {
        return Ok(Json(serde_json::json!({
            "port": port,
            "in_use": false,
        })));
    }

    let pid: u32 = pid_str.lines().next()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(0);

    let ps_out = tokio::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid,pcpu,rss,comm="])
        .output()
        .await
        .ok();

    let (cpu, rss_mb, name) = if let Some(ref o) = ps_out {
        let line = String::from_utf8_lossy(&o.stdout);
        let parts: Vec<&str> = line.trim().lines().last()
            .unwrap_or("").split_whitespace().collect();
        if parts.len() >= 4 {
            let cpu: f64 = parts[1].parse().unwrap_or(0.0);
            let rss_kb: u64 = parts[2].parse().unwrap_or(0);
            let name = parts[3..].join(" ");
            (cpu, rss_kb / 1024, name)
        } else {
            (0.0, 0, "unknown".to_string())
        }
    } else {
        (0.0, 0, "unknown".to_string())
    };

    Ok(Json(serde_json::json!({
        "port": port,
        "in_use": true,
        "pid": pid,
        "process_name": name,
        "cpu_percent": cpu,
        "rss_mb": rss_mb,
    })))
}

/// GET /processes/tree — build a process tree from ps output.
///
/// On-demand endpoint (not part of the regular poll loop).
async fn process_tree_handler(
    State(state): State<AppState>,
    Query(params): Query<TreeParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let min_cpu = params.min_cpu.unwrap_or(1.0);
    let min_mem = params.min_mem.unwrap_or(0.5);

    let output = tokio::process::Command::new("sh")
        .args(["-c", asmi_core::CMD_PS_TREE])
        .output()
        .await
        .map_err(|e| ApiError::Internal(format!("ps failed: {e}")))?;

    let text = String::from_utf8_lossy(&output.stdout);
    let tree = asmi_core::parse_process_tree(&text, min_cpu, min_mem);

    Ok(Json(serde_json::json!({
        "hostname": state.hostname,
        "min_cpu": min_cpu,
        "min_mem": min_mem,
        "processes": tree,
    })))
}

/// GET /disk — latest disk I/O stats from the snapshot.
async fn disk_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    match snap.as_ref().and_then(|s| s.disk_io.as_ref()) {
        Some(disk) => Ok(Json(serde_json::json!({
            "hostname": state.hostname,
            "disk_io": disk,
        }))),
        None => Err(ApiError::NotFound("no disk I/O data available yet".into())),
    }
}

/// GET /network — latest network throughput stats.
async fn network_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    match snap.as_ref().and_then(|s| s.network.as_ref()) {
        Some(net) => Ok(Json(serde_json::json!({
            "hostname": state.hostname,
            "network": net,
        }))),
        None => Err(ApiError::NotFound("no network data available yet".into())),
    }
}

/// GET /ane — ANE (Apple Neural Engine) power and status.
///
/// Returns the current ANE power draw from the latest snapshot, along with
/// CPU/GPU power for context. The `source` field indicates whether the ANE
/// reading came from IOReport (no sudo) or powermetrics (needs sudo).
async fn ane_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let snap = state.snapshot.read().await;
    match snap.as_ref() {
        Some(s) => {
            // ANE draws 2000-3000 mW under load; sub-1 mW is IOReport noise
            let ane_active = s.ane_watts > 1.0;
            Ok(Json(serde_json::json!({
                "hostname": state.hostname,
                "ane_watts": s.ane_watts,
                "ane_active": ane_active,
                "cpu_watts": s.cpu_watts,
                "gpu_watts": s.gpu_watts,
                "total_soc_mw": s.cpu_watts + s.gpu_watts + s.ane_watts,
                "power_source": s.power_source,
                "timestamp": s.timestamp,
            })))
        }
        None => Err(ApiError::NotFound("no data yet — daemon still starting".into())),
    }
}

// ---------------------------------------------------------------------------
// eGPU / TinyGPU detection
// ---------------------------------------------------------------------------

/// Scan for eGPU devices and TinyGPU DriverKit extension.
/// Checks three layers:
///   1. DriverKit extension status (systemextensionsctl)
///   2. Discrete/external GPUs (system_profiler SPDisplaysDataType)
///   3. tinygrad runtime availability
pub async fn scan_egpu(hostname: &str) -> serde_json::Value {
    use tokio::process::Command;

    // 1. Check TinyGPU DriverKit extension
    let (driver_installed, driver_version, driver_status) = {
        let output = Command::new("systemextensionsctl")
            .arg("list")
            .output()
            .await;
        match output {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout).to_string()
                    + &String::from_utf8_lossy(&o.stderr);
                // Lines look like: "--- com.tinygrad.tinygpu.dext (1.0.0/1) [activated enabled]"
                if let Some(line) = text.lines().find(|l| {
                    let lower = l.to_lowercase();
                    lower.contains("tinygpu") || lower.contains("tinygrad")
                }) {
                    let version = line.split('(')
                        .nth(1)
                        .and_then(|s| s.split(')').next())
                        .map(|s| s.split('/').next().unwrap_or(s).trim().to_string());
                    let status = if line.contains("activated") && line.contains("enabled") {
                        "activated"
                    } else if line.contains("activated") {
                        "activated_disabled"
                    } else {
                        "installed"
                    };
                    (true, version, status.to_string())
                } else {
                    (false, None, "not_found".to_string())
                }
            }
            Err(_) => (false, None, "check_failed".to_string()),
        }
    };

    // 2. Enumerate external/discrete GPUs via system_profiler
    let egpus = {
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .await;
        match output {
            Ok(o) if o.status.success() => {
                let json: serde_json::Value = serde_json::from_slice(&o.stdout)
                    .unwrap_or(serde_json::Value::Null);
                let mut devices = Vec::new();
                if let Some(displays) = json.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
                    for gpu in displays {
                        let name = gpu.get("sppci_model")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        // Skip Apple integrated GPUs
                        if name.starts_with("Apple") {
                            continue;
                        }
                        let vendor_str = gpu.get("sppci_vendor")
                            .or_else(|| gpu.get("spdisplays_vendor"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let vendor = if vendor_str.to_lowercase().contains("nvidia") {
                            "nvidia"
                        } else if vendor_str.to_lowercase().contains("amd")
                            || vendor_str.to_lowercase().contains("ati")
                        {
                            "amd"
                        } else if vendor_str.to_lowercase().contains("intel") {
                            "intel"
                        } else {
                            "unknown"
                        };
                        let vram = gpu.get("sppci_vram")
                            .or_else(|| gpu.get("_spdisplays_vram"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("0");
                        // Parse VRAM like "24 GB" or "24576 MB"
                        let vram_gb: f64 = {
                            let lower = vram.to_lowercase();
                            let num: f64 = lower.split_whitespace()
                                .next()
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0.0);
                            if lower.contains("mb") { num / 1024.0 } else { num }
                        };
                        let bus = gpu.get("sppci_bus")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let bus_type = if bus.to_lowercase().contains("thunderbolt") {
                            "thunderbolt"
                        } else if bus.to_lowercase().contains("usb") {
                            "usb4"
                        } else if bus.to_lowercase().contains("pci") {
                            "pcie"
                        } else {
                            bus
                        };
                        let pci_slot = gpu.get("sppci_slot_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let device_id = gpu.get("sppci_device_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        devices.push(serde_json::json!({
                            "name": name,
                            "vendor": vendor,
                            "vram_gb": vram_gb,
                            "bus": bus_type,
                            "pci_slot": pci_slot,
                            "device_id": device_id,
                        }));
                    }
                }
                devices
            }
            _ => Vec::new(),
        }
    };

    // 3. Check tinygrad availability
    let py = resolve_python();
    let tinygrad_info = {
        let output = Command::new(py)
            .args(["-c", "import tinygrad; print(tinygrad.__version__); print(tinygrad.Device.DEFAULT)"])
            .output()
            .await;
        match output {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout).trim().to_string();
                let mut lines = text.lines();
                let version = lines.next().map(|s| s.to_string());
                let device = lines.next().map(|s| s.to_string());
                (true, version, device)
            }
            _ => (false, None, None),
        }
    };

    serde_json::json!({
        "hostname": hostname,
        "tinygpu_driver": {
            "installed": driver_installed,
            "version": driver_version,
            "status": driver_status,
        },
        "egpus": egpus,
        "egpu_count": egpus.len(),
        "tinygrad": {
            "available": tinygrad_info.0,
            "version": tinygrad_info.1,
            "default_device": tinygrad_info.2,
        },
        "scanned_at": chrono::Utc::now().to_rfc3339(),
    })
}

/// GET /egpu → cached eGPU/TinyGPU status (refreshed every 30s).
/// Gated behind `--experimental-egpu` CLI flag.
async fn egpu_handler(State(state): State<AppState>) -> axum::response::Response {
    use axum::response::IntoResponse;

    let cache = state.egpu_cache.read().await;
    match cache.as_ref() {
        Some((data, scanned_at)) => {
            let mut result = data.clone();
            if let Some(obj) = result.as_object_mut() {
                obj.insert("experimental".to_string(), serde_json::Value::Bool(true));
                obj.insert("cache_age_secs".to_string(),
                    serde_json::Value::from(scanned_at.elapsed().as_secs()));
            }
            Json(result).into_response()
        }
        None => {
            // Cache is None → --experimental-egpu was not passed
            let body = Json(serde_json::json!({
                "experimental": true,
                "enabled": false,
                "hostname": state.hostname,
                "error": "eGPU detection not enabled. Start daemon with --experimental-egpu",
            }));
            (axum::http::StatusCode::SERVICE_UNAVAILABLE, body).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// RDMA check endpoints — remote monitoring via Tailscale
// ---------------------------------------------------------------------------

/// A single ping result for a TB5 peer.
#[derive(Serialize)]
struct PingResult {
    ip: String,
    interface: String,
    reachable: bool,
    latency_ms: Option<f64>,
}

/// GET /rdma/check → comprehensive local RDMA health check.
///
/// Returns bridge interfaces, 169.254 link-local IPs, RDMA device states,
/// and live ping results to all known TB5 peers. Designed for remote
/// monitoring via Tailscale — hit any node's :9090/rdma/check from anywhere.
async fn rdma_check_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    use tokio::process::Command;

    let hostname = &state.hostname;

    // 1. Parse ifconfig for bridge interfaces and 169.254 link-local addresses
    let ifconfig_out = Command::new("ifconfig")
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    let mut bridges: Vec<serde_json::Value> = Vec::new();
    let mut link_local: Vec<serde_json::Value> = Vec::new();
    let mut bridge0_members: Vec<String> = Vec::new();
    let mut current_iface = String::new();

    for line in ifconfig_out.lines() {
        // Interface header line: "bridge100: flags=..."  or  "en3: flags=..."
        if !line.starts_with('\t') && !line.starts_with(' ') {
            if let Some(name) = line.split(':').next() {
                current_iface = name.to_string();
            }
        }
        // Capture bridge0 member interfaces: "	member: en1 flags=3<...>"
        if current_iface == "bridge0" {
            let trimmed = line.trim();
            if trimmed.starts_with("member:") {
                if let Some(iface) = trimmed.split_whitespace().nth(1) {
                    bridge0_members.push(iface.to_string());
                }
            }
        }
        if line.contains("inet ") && !line.contains("inet6") {
            if let Some(ip) = line.split_whitespace().nth(1) {
                if current_iface.starts_with("bridge") {
                    bridges.push(serde_json::json!({
                        "interface": current_iface,
                        "ip": ip,
                    }));
                }
                // RDMA-reachable IPs: TB5 link-local (169.254) plus the canonical
                // assigned RDMA /30 subnet (192.168.10.x). Without the 192.168.10 prefix,
                // assigned RDMA IPs are invisible to peer discovery, so Active ports get
                // filtered out at the link_local_ifaces gate.
                if (ip.starts_with("169.254.") || ip.starts_with("192.168.10."))
                    && !current_iface.starts_with("lo")
                {
                    link_local.push(serde_json::json!({
                        "interface": current_iface,
                        "ip": ip,
                    }));
                }
            }
        }
    }

    let bridge0_blocking = !bridge0_members.is_empty();

    // 2. RDMA device states from snapshot
    let rdma_devices = {
        let snap = state.snapshot.read().await;
        snap.as_ref()
            .and_then(|s| s.rdma.as_ref())
            .map(|r| {
                r.devices
                    .iter()
                    .map(|d| {
                        serde_json::json!({
                            "name": d.name,
                            "port_state": format!("{:?}", d.port_state),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };

    let active_count = rdma_devices.iter().filter(|d| {
        d.get("port_state").and_then(|v| v.as_str()) == Some("Active")
    }).count();

    // 3. Collect known RDMA IPs from NodeMap + ARP discovery, then ping all.
    //
    // Two sources of peer IPs:
    //   a) NodeMap.rdma_ips — persisted from previous scans, survives reboots
    //   b) ARP table — ephemeral, only populated when traffic has flowed recently
    // Using both ensures we find peers even when ARP has gone stale.

    let known_rdma_ips: Vec<(String, String)> = {
        let nm = state.node_map.read().await;
        nm.rdma_ips
            .iter()
            .flat_map(|(hostname, ips)| {
                ips.iter().map(move |ip| (ip.clone(), hostname.clone()))
            })
            .collect()
    };

    let arp_out = Command::new("arp")
        .args(["-an"])
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    // Collect our own IPs to filter self-pings
    let own_ips: std::collections::HashSet<String> = link_local
        .iter()
        .filter_map(|v| v.get("ip").and_then(|ip| ip.as_str()).map(|s| s.to_string()))
        .collect();

    // Collect interfaces with Active RDMA devices for filtering stale ARP
    let active_ifaces: std::collections::HashSet<String> = rdma_devices
        .iter()
        .filter(|d| d.get("port_state").and_then(|v| v.as_str()) == Some("Active"))
        .filter_map(|d| {
            d.get("name")
                .and_then(|v| v.as_str())
                .map(|name| name.strip_prefix("rdma_").unwrap_or(name).to_string())
        })
        .collect();

    // Collect interfaces that have a link-local (169.254.x.x) address assigned.
    let link_local_ifaces: std::collections::HashSet<String> = link_local
        .iter()
        .filter_map(|v| v.get("interface").and_then(|i| i.as_str()).map(|s| s.to_string()))
        .collect();

    // Start with known RDMA IPs from NodeMap (interface unknown, will be resolved by ping)
    let mut peer_ips: Vec<(String, String)> = known_rdma_ips
        .iter()
        .filter(|(ip, _)| !own_ips.contains(ip))
        .map(|(ip, _)| (ip.clone(), "nodemap".to_string()))
        .collect();

    // Also add ARP-discovered peers (may overlap — dedup below)
    let mut seen_ips: std::collections::HashSet<String> = peer_ips.iter().map(|(ip, _)| ip.clone()).collect();

    for line in arp_out.lines() {
        // Accept ARP peers on the TB5 link-local range and the canonical RDMA
        // /30 subnet (192.168.10.x).
        if line.contains("incomplete")
            || !(line.contains("169.254.") || line.contains("192.168.10."))
        {
            continue;
        }
        let ip = line.split('(').nth(1).and_then(|s| s.split(')').next());
        let iface = line.split(" on ").nth(1).and_then(|s| s.split_whitespace().next());
        if let (Some(ip), Some(iface)) = (ip, iface) {
            // Skip: bridge/lo interfaces, our own IPs, and interfaces without Active RDMA
            if iface.starts_with("bridge") || iface.starts_with("lo") {
                continue;
            }
            if own_ips.contains(ip) {
                continue;
            }
            if !active_ifaces.contains(iface) {
                continue; // stale ARP on a Down RDMA port — ignore
            }
            if !link_local_ifaces.contains(iface) {
                continue; // Active RDMA but no link-local IP (e.g. bridge/sharing) — skip
            }
            if seen_ips.insert(ip.to_string()) {
                peer_ips.push((ip.to_string(), iface.to_string()));
            }
        }
    }

    // Ping all peers concurrently (1 packet, 1s timeout)
    let ping_futures: Vec<_> = peer_ips
        .iter()
        .map(|(ip, iface)| {
            let ip = ip.clone();
            let iface = iface.clone();
            async move {
                let output = Command::new("ping")
                    .args(["-c", "1", "-t", "1", &ip])
                    .output()
                    .await;
                let (reachable, latency_ms) = match output {
                    Ok(o) if o.status.success() => {
                        let stdout = String::from_utf8_lossy(&o.stdout);
                        let latency = stdout
                            .lines()
                            .find(|l| l.contains("time="))
                            .and_then(|l| {
                                l.split("time=").nth(1)?.split_whitespace().next()?.parse::<f64>().ok()
                            });
                        (true, latency)
                    }
                    _ => (false, None),
                };
                PingResult { ip, interface: iface, reachable, latency_ms }
            }
        })
        .collect();

    let ping_results: Vec<PingResult> = futures::future::join_all(ping_futures).await;

    let peers_reachable = ping_results.iter().filter(|p| p.reachable).count();
    let peers_total = ping_results.len();

    // 3b. Persist newly discovered reachable IPs to NodeMap so future scans
    //     don't depend on ARP cache being warm.
    {
        let reachable_ips: Vec<String> = ping_results
            .iter()
            .filter(|p| p.reachable)
            .map(|p| p.ip.clone())
            .collect();
        if !reachable_ips.is_empty() {
            let mut nm = state.node_map.write().await;
            let changed = nm.add_rdma_ips(hostname, &reachable_ips);
            if changed {
                nm.save();
                tracing::info!(
                    ips = ?reachable_ips,
                    "persisted {} new RDMA peer IPs to NodeMap",
                    reachable_ips.len()
                );
            }
        }
    }

    // 4. Get RDMA links from node_map (discovered link ↔ peer mapping)
    let rdma_links = {
        let nm = state.node_map.read().await;
        nm.rdma_links
            .iter()
            .map(|link| {
                serde_json::json!({
                    "local_interface": link.local_interface,
                    "local_ip": link.local_ip,
                    "remote_ip": link.remote_ip,
                    "remote_hostname": link.remote_hostname,
                    "rdma_device": link.rdma_device,
                    "port_state": link.port_state.as_ref().map(|ps| format!("{ps:?}")),
                })
            })
            .collect::<Vec<_>>()
    };

    // 5. Also include topology cache status if available
    let topology = {
        let cache = state.topology_cache.read().await;
        cache.as_ref().map(|(report, scanned_at)| {
            serde_json::json!({
                "mesh_complete": report.mesh_complete,
                "nodes": report.nodes,
                "link_count": report.links.len(),
                "missing_links": report.missing_links,
                "jaccl_ready": report.jaccl_ready,
                "jaccl_ready_subsets": report.jaccl_ready_subsets,
                "scan_age_secs": scanned_at.elapsed().as_secs(),
            })
        })
    };

    // Overall health: all active RDMA devices can reach their peers
    let healthy = active_count > 0 && peers_reachable > 0 && peers_reachable == peers_total;

    Json(serde_json::json!({
        "hostname": hostname,
        "healthy": healthy,
        "summary": format!(
            "{} RDMA devices active, {}/{} peers reachable, {} bridges",
            active_count, peers_reachable, peers_total, bridges.len()
        ),
        "rdma_devices": rdma_devices,
        "rdma_active_count": active_count,
        "bridges": bridges,
        "bridge0_blocking": bridge0_blocking,
        "bridge0_members": bridge0_members,
        "link_local_ips": link_local,
        "peer_pings": ping_results,
        "peers_reachable": peers_reachable,
        "peers_total": peers_total,
        "rdma_links": rdma_links,
        "topology": topology,
    }))
}

/// POST /bridge0/destroy — Remove bridge0 and create individual TB5 network services.
/// Localhost-only for safety. Returns the list of freed interfaces.
async fn bridge0_destroy_handler(
    State(_state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use tokio::process::Command;

    // Safety: only allow from localhost
    if !addr.ip().is_loopback() {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "bridge0/destroy is localhost-only"
        }))).into_response();
    }

    // 1. Detect bridge0 members from ifconfig
    let ifconfig = Command::new("ifconfig").arg("bridge0").output().await;
    let members: Vec<String> = match ifconfig {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.trim().starts_with("member:"))
                .filter_map(|l| l.split_whitespace().nth(1).map(String::from))
                .collect()
        }
        _ => {
            return Json(serde_json::json!({
                "status": "ok",
                "message": "No bridge0 found — nothing to do",
                "bridge0_gone": true,
                "freed_interfaces": [],
            })).into_response();
        }
    };

    if members.is_empty() {
        return Json(serde_json::json!({
            "status": "ok",
            "message": "bridge0 exists but has no member interfaces",
            "bridge0_gone": false,
            "freed_interfaces": [],
        })).into_response();
    }

    let mut steps_log: Vec<serde_json::Value> = Vec::new();
    let plist = "/Library/Preferences/SystemConfiguration/preferences.plist";

    // 2. Remove bridge0 from preferences.plist
    let rm = Command::new("sudo")
        .args(["/usr/libexec/PlistBuddy", "-c",
               "Delete :VirtualNetworkInterfaces:Bridge:bridge0", plist])
        .output().await;
    steps_log.push(serde_json::json!({
        "step": "remove_plist",
        "ok": rm.as_ref().map_or(false, |o| o.status.success()),
    }));

    // 3. Create network services for each freed interface
    for iface in &members {
        let svc_name = format!("r1o TB {}", iface);
        let create = Command::new("sudo")
            .args(["networksetup", "-createnetworkservice", &svc_name, iface])
            .output().await;
        steps_log.push(serde_json::json!({
            "step": format!("create_service_{}", iface),
            "ok": create.as_ref().map_or(false, |o| o.status.success()),
        }));
    }

    // 4. Restart configd to apply changes
    let _ = Command::new("sudo").args(["killall", "configd"]).output().await;
    steps_log.push(serde_json::json!({"step": "restart_configd", "ok": true}));

    // 5. Wait for IPv4LL assignment
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    let _ = Command::new("sudo").args(["ipconfig", "waitall"]).output().await;
    steps_log.push(serde_json::json!({"step": "wait_ipv4ll", "ok": true}));

    // 6. Verify bridge0 is gone
    let verify = Command::new("ifconfig").arg("bridge0").output().await;
    let bridge0_gone = verify.map_or(true, |o| !o.status.success());
    steps_log.push(serde_json::json!({"step": "verify", "ok": bridge0_gone}));

    // 7. Collect new link-local IPs on freed interfaces
    let mut new_ips: Vec<serde_json::Value> = Vec::new();
    for iface in &members {
        let out = Command::new("ifconfig").arg(iface).output().await;
        if let Ok(o) = out {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines() {
                if line.contains("inet 169.254") {
                    if let Some(ip) = line.split_whitespace().nth(1) {
                        new_ips.push(serde_json::json!({"interface": iface, "ip": ip}));
                    }
                }
            }
        }
    }

    Json(serde_json::json!({
        "status": if bridge0_gone { "ok" } else { "partial" },
        "bridge0_gone": bridge0_gone,
        "freed_interfaces": members,
        "new_ips": new_ips,
        "steps": steps_log,
    })).into_response()
}

/// POST /bridge0/restore — Recreate bridge0 from r1o TB network services.
/// Localhost-only. Reverses the effect of /bridge0/destroy.
async fn bridge0_restore_handler(
    State(_state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use tokio::process::Command;

    if !addr.ip().is_loopback() {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "bridge0/restore is localhost-only"
        }))).into_response();
    }

    // Check if bridge0 already exists
    let check = Command::new("ifconfig").arg("bridge0").output().await;
    if check.as_ref().map_or(false, |o| o.status.success()) {
        return Json(serde_json::json!({
            "status": "ok",
            "message": "bridge0 already exists — nothing to restore",
        })).into_response();
    }

    // Find r1o TB services to convert back
    let svc_out = Command::new("networksetup")
        .arg("-listallnetworkservices")
        .output().await;
    let services: Vec<String> = svc_out.map_or(vec![], |o| {
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| l.starts_with("r1o TB "))
            .map(String::from)
            .collect()
    });

    let ifaces: Vec<String> = services.iter()
        .filter_map(|s| s.strip_prefix("r1o TB ").map(String::from))
        .collect();

    if ifaces.is_empty() {
        return Json(serde_json::json!({
            "status": "error",
            "message": "No r1o TB services found to restore",
        })).into_response();
    }

    let plist = "/Library/Preferences/SystemConfiguration/preferences.plist";

    // Recreate bridge0 in preferences.plist
    let _ = Command::new("sudo").args(["/usr/libexec/PlistBuddy", "-c",
        "Add :VirtualNetworkInterfaces:Bridge:bridge0:Options:__AUTO__ string thunderbolt-bridge",
        plist]).output().await;
    let _ = Command::new("sudo").args(["/usr/libexec/PlistBuddy", "-c",
        "Add :VirtualNetworkInterfaces:Bridge:bridge0:Interfaces array",
        plist]).output().await;

    for iface in &ifaces {
        let _ = Command::new("sudo").args(["/usr/libexec/PlistBuddy", "-c",
            &format!("Add :VirtualNetworkInterfaces:Bridge:bridge0:Interfaces: string {}", iface),
            plist]).output().await;
    }

    // Remove r1o TB services
    for svc in &services {
        let _ = Command::new("sudo")
            .args(["networksetup", "-removenetworkservice", svc])
            .output().await;
    }

    // Apply
    let _ = Command::new("sudo").args(["killall", "configd"]).output().await;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let verify = Command::new("ifconfig").arg("bridge0").output().await;
    let restored = verify.as_ref().map_or(false, |o| o.status.success());

    Json(serde_json::json!({
        "status": if restored { "ok" } else { "error" },
        "bridge0_restored": restored,
        "interfaces_returned": ifaces,
    })).into_response()
}

/// POST /rdma/setup → run full RDMA autosetup (bridge0, IPs, routes, peers, hostfile).
///
/// Optional body: `{"hosts": ["host1","host2",...]}` — passed straight to
/// mlx.distributed_config. Use to restrict setup to currently-reachable nodes
/// (mlx.distributed_config aborts if any listed host fails SSH). If body is
/// omitted or `hosts` is absent, falls back to NodeMap.rdma_ips.keys().
async fn rdma_setup_handler(
    State(state): State<AppState>,
    body: Option<Json<serde_json::Value>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hosts_override: Option<Vec<String>> = body
        .and_then(|Json(v)| {
            v.get("hosts")
                .and_then(|h| h.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        })
        .filter(|v: &Vec<String>| !v.is_empty());
    let report = crate::rdma_autosetup::autosetup(&state.node_map, hosts_override).await;
    Ok(Json(
        serde_json::to_value(&report)
            .map_err(|e| ApiError::Internal(format!("serialize: {e}")))?,
    ))
}

/// GET /rdma/health → per-device PD budget and port state.
///
/// When compiled with `--features jaccl`, uses the native JACCL FFI to probe
/// PD budget directly (no shell-out). Otherwise falls back to `asmi-pd-probe`
/// binary or `ibv_devinfo` text parsing.
///
/// This is the data the web UI's networking section uses to show PD exhaustion
/// warnings.
async fn rdma_health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    // ── Native FFI path (jaccl feature) ──────────────────────────────
    #[cfg(feature = "jaccl")]
    {
        use asmi_core::jaccl_ffi;

        let jaccl_available = jaccl_ffi::available();

        if jaccl_available {
            // Discover devices via ibv_devinfo (names only) then probe each
            // natively. We still need device names from ibv_devinfo but avoid
            // the heavier asmi-pd-probe binary.
            use tokio::process::Command;

            let devinfo = Command::new("ibv_devinfo")
                .output()
                .await
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            let mut devices = Vec::new();
            for line in devinfo.lines() {
                let trimmed = line.trim();
                if let Some(name) = trimmed.strip_prefix("hca_id:") {
                    let name = name.trim().to_string();
                    let pd_status = jaccl_ffi::pd_probe(&name);
                    devices.push(serde_json::json!({
                        "name": name,
                        "pd_available": pd_status == 1,
                        "pd_probe_raw": pd_status,
                    }));
                }
            }

            return Json(serde_json::json!({
                "hostname": state.hostname,
                "probe": "jaccl-native-ffi",
                "jaccl_available": true,
                "rdma": { "devices": devices },
            }));
        }

        // JACCL compiled in but libibverbs not loadable — fall through
        // to shell-out path below.
    }

    // ── Shell-out path (no jaccl feature, or libibverbs not available) ──
    use tokio::process::Command;

    // Primary: asmi-pd-probe returns structured JSON
    if let Ok(output) = Command::new("asmi-pd-probe").output().await {
        if output.status.success() {
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                return Json(serde_json::json!({
                    "hostname": state.hostname,
                    "probe": "asmi-pd-probe",
                    "rdma": json,
                }));
            }
        }
    }

    // Fallback: parse ibv_devinfo -v for each device
    let devinfo = Command::new("ibv_devinfo")
        .arg("-v")
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    let mut devices = Vec::new();
    let mut current_name = String::new();
    let mut max_pd = 0i64;
    let mut max_qp = 0i64;
    let mut max_mr = 0i64;

    for line in devinfo.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("hca_id:") {
            if !current_name.is_empty() {
                devices.push(serde_json::json!({
                    "name": current_name,
                    "max_pd": max_pd,
                    "max_qp": max_qp,
                    "max_mr": max_mr,
                }));
            }
            current_name = trimmed.strip_prefix("hca_id:").unwrap_or("").trim().to_string();
            max_pd = 0;
            max_qp = 0;
            max_mr = 0;
        } else if trimmed.starts_with("max_pd:") {
            max_pd = trimmed.strip_prefix("max_pd:").unwrap_or("0").trim().parse().unwrap_or(0);
        } else if trimmed.starts_with("max_qp:") {
            max_qp = trimmed.strip_prefix("max_qp:").unwrap_or("0").trim().parse().unwrap_or(0);
        } else if trimmed.starts_with("max_mr:") {
            max_mr = trimmed.strip_prefix("max_mr:").unwrap_or("0").trim().parse().unwrap_or(0);
        }
    }
    if !current_name.is_empty() {
        devices.push(serde_json::json!({
            "name": current_name,
            "max_pd": max_pd,
            "max_qp": max_qp,
            "max_mr": max_mr,
        }));
    }

    Json(serde_json::json!({
        "hostname": state.hostname,
        "probe": "ibv_devinfo-fallback",
        "rdma": { "devices": devices },
    }))
}

/// GET /rdma/mesh → query all cluster nodes, aggregate mesh health.
///
/// Reads node list from NodeMap (seeded from ~/.r1o/cluster.json).
/// Queries each node over local network (hostname.local:9090), not Tailscale.
async fn rdma_mesh_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let nm = state.node_map.read().await;
    let all_nodes: Vec<String> = nm.nodes.clone();
    let local_hostname = state.hostname.clone();
    drop(nm);

    // Exclude self — we'll include our own /rdma/check inline
    let remote_nodes: Vec<&str> = all_nodes.iter()
        .filter(|n| n.as_str() != local_hostname)
        .map(|s| s.as_str())
        .collect();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    // Query remote nodes over local network (.local mDNS)
    let futures: Vec<_> = remote_nodes
        .iter()
        .map(|&name| {
            let client = client.clone();
            let name = name.to_string();
            let url = format!("http://{}.local:9090/rdma/check", name);
            async move {
                match client.get(&url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        match resp.json::<serde_json::Value>().await {
                            Ok(data) => (name, Some(data)),
                            Err(_) => (name, None),
                        }
                    }
                    _ => (name, None),
                }
            }
        })
        .collect();

    let results: Vec<(String, Option<serde_json::Value>)> =
        futures::future::join_all(futures).await;

    let mut nodes: Vec<serde_json::Value> = Vec::new();
    let mut total_active = 0usize;
    let mut total_reachable_peers = 0usize;
    let mut total_peers = 0usize;
    let mut nodes_healthy = 0usize;
    let mut nodes_online = 0usize;

    for (name, data) in &results {
        match data {
            Some(d) => {
                nodes_online += 1;
                let healthy = d.get("healthy").and_then(|v| v.as_bool()).unwrap_or(false);
                if healthy {
                    nodes_healthy += 1;
                }
                total_active += d.get("rdma_active_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                total_reachable_peers += d.get("peers_reachable").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                total_peers += d.get("peers_total").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                nodes.push(serde_json::json!({
                    "hostname": name,
                    "online": true,
                    "healthy": healthy,
                    "summary": d.get("summary"),
                    "rdma_active_count": d.get("rdma_active_count"),
                    "peers_reachable": d.get("peers_reachable"),
                    "peers_total": d.get("peers_total"),
                    "peer_pings": d.get("peer_pings"),
                    "bridges": d.get("bridges"),
                    "topology": d.get("topology"),
                }));
            }
            None => {
                nodes.push(serde_json::json!({
                    "hostname": name,
                    "online": false,
                    "healthy": false,
                    "summary": "unreachable on local network",
                }));
            }
        }
    }

    let nodes_total = all_nodes.len();
    let mesh_healthy = nodes_healthy == nodes_online && nodes_online > 0;

    // Get topology from local cache for mesh-level summary
    let mesh_topology = {
        let cache = state.topology_cache.read().await;
        cache.as_ref().map(|(report, scanned_at)| {
            serde_json::json!({
                "mesh_complete": report.mesh_complete,
                "link_count": report.links.len(),
                "expected_links": report.nodes.len() * (report.nodes.len() - 1) / 2,
                "missing_links": report.missing_links,
                "jaccl_ready": report.jaccl_ready,
                "scan_age_secs": scanned_at.elapsed().as_secs(),
            })
        })
    };

    Json(serde_json::json!({
        "mesh_healthy": mesh_healthy,
        "summary": format!(
            "{}/{} nodes healthy, {} RDMA devices active, {}/{} peer links verified",
            nodes_healthy, nodes_total, total_active, total_reachable_peers, total_peers
        ),
        "nodes_online": nodes_online,
        "nodes_total": nodes_total,
        "nodes_healthy": nodes_healthy,
        "total_rdma_active": total_active,
        "total_peers_reachable": total_reachable_peers,
        "total_peers": total_peers,
        "topology": mesh_topology,
        "nodes": nodes,
        "queried_via": "local_network",
        "queried_from": state.hostname,
    }))
}

// ---------------------------------------------------------------------------
// POST /prep — benchmark prep mode: kill apps, flush cache, toggle spotlight
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PrepBody {
    /// "on" — kill non-essential apps, flush caches, disable Spotlight
    /// "off" — re-enable Spotlight, report
    /// "status" — check if prep mode is active
    action: String,
}

const PREP_STATE_FILE: &str = "/tmp/.bench-prep-active";

/// Processes to kill during prep (background daemons that waste RAM/GPU)
const KILL_PROCS: &[&str] = &[
    "speechrecognitiond",
    "localspeechrecognition",
    "DictationIM",
    "SpeechSynthesisServerXPC",
    "Playwright",
    "playwright",
    "SpringBoard.app",
    "SimulatorTrampoline",
    "Simulator.app",
];

/// Apps to keep alive (everything else gets killed)
const KEEP_APPS: &[&str] = &["Finder", "cmux"];

/// Snapshot RAM + process state for pre/post-flight comparison.
/// Uses sysctl + vm_stat for fresh memory numbers (not the cached asmi snapshot).
async fn prep_snapshot() -> serde_json::Value {
    use tokio::process::Command;

    // RAM via sysctl (total) + vm_stat (free/inactive pages → available)
    let total_bytes: f64 = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .await
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok())
        .unwrap_or(0.0);
    let total_gb = total_bytes / 1_073_741_824.0;

    // vm_stat gives page counts; page size is 16384 on Apple Silicon
    let available_gb = Command::new("vm_stat")
        .output()
        .await
        .ok()
        .map(|o| {
            let text = String::from_utf8_lossy(&o.stdout);
            let page_size: f64 = 16384.0;
            let mut free: f64 = 0.0;
            let mut inactive: f64 = 0.0;
            let mut purgeable: f64 = 0.0;
            for line in text.lines() {
                let val = || -> Option<f64> {
                    line.split(':').nth(1)?.trim().trim_end_matches('.').parse().ok()
                };
                if line.starts_with("Pages free") {
                    free = val().unwrap_or(0.0);
                } else if line.starts_with("Pages inactive") {
                    inactive = val().unwrap_or(0.0);
                } else if line.starts_with("Pages purgeable") {
                    purgeable = val().unwrap_or(0.0);
                }
            }
            (free + inactive + purgeable) * page_size / 1_073_741_824.0
        })
        .unwrap_or(0.0);

    let used_gb = total_gb - available_gb;

    // Process count
    let process_count: u32 = Command::new("ps")
        .args(["-A", "-o", "pid="])
        .output()
        .await
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().count() as u32)
        .unwrap_or(0);

    // GUI app count via osascript
    let app_count = Command::new("osascript")
        .arg("-e")
        .arg(r#"tell application "System Events" to count of (every application process whose background only is false)"#)
        .output()
        .await
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u32>().ok())
        .unwrap_or(0);

    // Spotlight status
    let spotlight_on = Command::new("mdutil")
        .args(["-s", "/"])
        .output()
        .await
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("Indexing enabled"))
        .unwrap_or(false);

    serde_json::json!({
        "ram_total_gb": (total_gb * 10.0).round() / 10.0,
        "ram_used_gb": (used_gb * 10.0).round() / 10.0,
        "ram_available_gb": (available_gb * 10.0).round() / 10.0,
        "process_count": process_count,
        "gui_app_count": app_count,
        "spotlight_indexing": spotlight_on,
    })
}

async fn prep_handler(
    State(_state): State<AppState>,
    Json(body): Json<PrepBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    use tokio::process::Command;
    use tokio::fs;

    match body.action.as_str() {
        "status" => {
            let active = fs::metadata(PREP_STATE_FILE).await.is_ok();
            let snapshot = prep_snapshot().await;
            Ok(Json(serde_json::json!({
                "prep": active,
                "current": snapshot,
            })))
        }

        "on" => {
            let start = std::time::Instant::now();

            // ── Pre-flight snapshot ──
            let preflight = prep_snapshot().await;

            let mut killed = Vec::new();
            let mut errors = Vec::new();

            // 1. Kill non-essential macOS GUI apps via osascript
            let script = format!(
                r#"tell application "System Events"
                    set keepList to {{{}}}
                    set appNames to name of every application process whose background only is false
                    set killList to {{}}
                    repeat with a in appNames
                        if a is not in keepList then
                            set end of killList to (a as text)
                        end if
                    end repeat
                    return killList
                end tell"#,
                KEEP_APPS.iter().map(|a| format!("\"{a}\"")).collect::<Vec<_>>().join(", ")
            );

            if let Ok(out) = Command::new("osascript").arg("-e").arg(&script).output().await {
                let apps = String::from_utf8_lossy(&out.stdout);
                for app in apps.trim().split(", ").filter(|a| !a.is_empty()) {
                    let app = app.trim();
                    if Command::new("osascript")
                        .arg("-e")
                        .arg(format!(r#"tell application "{app}" to quit"#))
                        .output()
                        .await
                        .is_ok()
                    {
                        killed.push(app.to_string());
                    }
                }
            }

            // 2. Kill background processes
            for proc in KILL_PROCS {
                if Command::new("pkill").arg("-x").arg(proc).output().await.is_ok() {
                    killed.push(proc.to_string());
                }
            }

            // 3. Flush file cache
            if let Err(e) = Command::new("purge").output().await {
                errors.push(format!("purge failed: {e}"));
            }

            // 4. Disable Spotlight
            if let Err(e) = Command::new("mdutil").args(["-a", "-i", "off"]).output().await {
                errors.push(format!("mdutil failed: {e}"));
            }

            // 5. Mark prep active
            let _ = fs::write(PREP_STATE_FILE, "on").await;

            // Brief settle for OS to release memory
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // ── Post-flight snapshot ──
            let postflight = prep_snapshot().await;
            let duration_ms = start.elapsed().as_millis() as u64;

            // Compute delta
            let ram_freed_gb = postflight["ram_available_gb"].as_f64().unwrap_or(0.0)
                - preflight["ram_available_gb"].as_f64().unwrap_or(0.0);
            let procs_killed = preflight["process_count"].as_u64().unwrap_or(0) as i64
                - postflight["process_count"].as_u64().unwrap_or(0) as i64;

            Ok(Json(serde_json::json!({
                "prep": true,
                "duration_ms": duration_ms,
                "killed": killed,
                "errors": errors,
                "preflight": preflight,
                "postflight": postflight,
                "delta": {
                    "ram_freed_gb": (ram_freed_gb * 10.0).round() / 10.0,
                    "processes_removed": procs_killed.max(0),
                    "gui_apps_removed": preflight["gui_app_count"].as_u64().unwrap_or(0) as i64
                        - postflight["gui_app_count"].as_u64().unwrap_or(0) as i64,
                },
            })))
        }

        "off" => {
            let start = std::time::Instant::now();
            let preflight = prep_snapshot().await;
            let mut errors = Vec::new();

            // Re-enable Spotlight
            if let Err(e) = Command::new("mdutil").args(["-a", "-i", "on"]).output().await {
                errors.push(format!("mdutil failed: {e}"));
            }

            // Remove state file
            let _ = tokio::fs::remove_file(PREP_STATE_FILE).await;

            let postflight = prep_snapshot().await;
            let duration_ms = start.elapsed().as_millis() as u64;

            Ok(Json(serde_json::json!({
                "prep": false,
                "duration_ms": duration_ms,
                "errors": errors,
                "preflight": preflight,
                "postflight": postflight,
            })))
        }

        other => Err(ApiError::BadRequest(format!(
            "unknown prep action: {other} (use 'on', 'off', or 'status')"
        ))),
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/health/setup", get(setup_handler))
        .route("/health/network", get(network_health_handler))
        .route("/health/network/fix", post(network_fix_handler))
        .route("/processes", get(processes_handler))
        .route("/port/{port}", get(port_lookup_handler))
        .route("/models", get(models_handler))
        .route("/volumes", get(volumes_handler))
        .route("/logs", get(logs_handler))
        .route("/runtime", get(runtime_handler))
        .route("/cluster", get(cluster_handler))
        .route("/cluster/models", get(cluster_models_handler))
        .route("/nodes", get(nodes_handler))
        .route("/stream", get(stream_handler))
        .route("/jaccl/config", get(jaccl_config_handler).post(jaccl_generate_handler))
        .route("/arp", get(arp_handler))
        .route("/thunderbolt", get(thunderbolt_handler))
        .route("/topology", get(topology_handler))
        .route("/topology/dot", get(topology_dot_handler))
        .route("/topology/validate", get(topology_validate_handler))
        .route("/serve/status", get(serve_status_handler))
        .route("/serve/load", post(serve_load_handler))
        .route("/serve/stop", post(serve_stop_handler))
        .route("/serve/reload", post(serve_reload_handler))
        .route("/launchd/disable", post(launchd_disable_handler))
        .route("/launchd/enable", post(launchd_enable_handler))
        // Phase 3 — read-only Hermes probe (Phase 4 adds /hermes/restart, /hermes/config)
        .route("/hermes/status", get(hermes_status_handler))
        // Phase 4 — Hermes lifecycle (Feature E)
        .route("/hermes/restart", post(hermes_restart_handler))
        .route("/hermes/config", post(hermes_config_handler))
        .route("/serve/share", post(serve_share_handler))
        .route("/serve/share/status", get(serve_share_status_handler))
        .route("/serve/share/stop", post(serve_share_stop_handler))
        .route("/serve/distributed/join", post(serve_distributed_join_handler))
        .route("/watchdog", get(watchdog_handler))
        .route("/watchdog/peers", get(peer_heartbeat_handler))
        .route("/watchdog/gpu-lock", get(gpu_lock_handler))
        // v0.5: metrics parity + process management
        .route("/processes/{pid}/kill", post(kill_process_handler))
        .route("/processes/tree", get(process_tree_handler))
        .route("/disk", get(disk_handler))
        .route("/network", get(network_handler))
        .route("/ane", get(ane_handler))
        .route("/egpu", get(egpu_handler))
        .route("/rdma/check", get(rdma_check_handler))
        .route("/rdma/health", get(rdma_health_handler))
        .route("/rdma/mesh", get(rdma_mesh_handler))
        .route("/rdma/setup", post(rdma_setup_handler))
        .route("/bridge0/destroy", post(bridge0_destroy_handler))
        .route("/bridge0/restore", post(bridge0_restore_handler))
        .route("/prep", post(prep_handler))
        // Native RDMA file transfer (gated by --features jaccl)
        .route("/transfer", post(crate::transfer::transfer_handler))
        .route("/transfer/accept", post(crate::transfer::transfer_accept_handler))
        // Experimental ANE compute endpoints (gated by --experimental-ane + --features ane)
        .route("/ane/compute", get(crate::ane::status_handler))
        .route("/ane/eval", post(crate::ane::eval_handler))
        .route("/ane/probe", get(crate::ane::probe_handler))
        // Autoresearch benchmark validation
        .route("/autoresearch/gate", get(autoresearch_gate_handler))
        .route("/autoresearch/reset", post(autoresearch_reset_handler))
        .with_state(state)
}
