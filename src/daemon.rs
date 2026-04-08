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
    pub serve_managers: Arc<HashMap<u16, crate::serve::ServeManager>>,
    pub share_manager: crate::serve::ShareManager,
    pub peer_heartbeat: Arc<crate::serve::PeerHeartbeat>,
    pub watchdog: Arc<crate::watchdog::Watchdog>,
    pub ane: crate::ane::AneState,
    pub egpu_cache: Arc<RwLock<Option<(serde_json::Value, std::time::Instant)>>>,
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

            let port_json = if devices.is_empty() {
                serde_json::json!({
                    "port": receptacle,
                    "connected": connected,
                    "speed": speed,
                })
            } else {
                serde_json::json!({
                    "port": receptacle,
                    "connected": true,
                    "speed": speed,
                    "devices": devices.iter().map(|(name, model)| {
                        serde_json::json!({ "name": name, "model_id": model })
                    }).collect::<Vec<_>>(),
                })
            };
            ports.push(port_json);
        }
    }

    serde_json::json!({
        "hostname": hostname,
        "hw_model": hw_model,
        "ports": ports,
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

/// Look up the ServeManager for a given port. Falls back to default.
fn get_manager(state: &AppState, port: Option<u16>) -> Option<&crate::serve::ServeManager> {
    let p = port.unwrap_or(DEFAULT_SERVE_PORT);
    state.serve_managers.get(&p)
}

/// GET /serve/status → serve status.
/// No ?port= → returns all servers: {"servers": [...], "unmanaged": [...]}
/// ?port=19080 → returns single ServeStatus for that port
async fn serve_status_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if let Some(port) = q.port {
        // Single-port query
        let mgr = get_manager(&state, Some(port))
            .ok_or_else(|| ApiError::NotFound(format!("unknown port: {port}")))?;
        let status = mgr.status().await;
        Ok(Json(serde_json::to_value(&status)
            .map_err(|e| ApiError::Internal(format!("serialize failed: {e}")))?))
    } else {
        // All ports → {"servers": [...], "unmanaged": [...]}
        let mut servers = Vec::new();
        for mgr in state.serve_managers.values() {
            servers.push(mgr.status().await);
        }
        servers.sort_by_key(|s| s.port);

        // Collect managed PIDs for diffing against detected processes
        let mut managed_pids = std::collections::HashSet::new();
        for s in &servers {
            if let Some(pid) = s.pid {
                managed_pids.insert(pid);
            }
        }

        // Auto-adopt unmanaged processes into their port's manager
        let orphans = adopt_unmanaged(&state, &managed_pids).await;

        // Re-read statuses after adoption (managers may have updated)
        let mut servers = Vec::new();
        for mgr in state.serve_managers.values() {
            servers.push(mgr.status().await);
        }
        servers.sort_by_key(|s| s.port);

        Ok(Json(serde_json::json!({
            "servers": servers,
            "unmanaged": orphans,
        })))
    }
}

/// Auto-adopt unmanaged model processes into the serve managers.
/// If a process is on a managed port → adopt into that manager.
/// Returns any processes that couldn't be adopted (no matching port manager).
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

        // Try to adopt into existing manager for this port
        if let Some(port) = proc.port {
            if let Some(mgr) = state.serve_managers.get(&port) {
                mgr.adopt_external(proc.pid, model, engine).await;
                continue;
            }
        }

        // No manager for this port — report as orphan
        let models: Vec<String> = if !proc.server_models.is_empty() {
            proc.server_models.iter().map(|m| m.id.clone()).collect()
        } else if let Some(ref m) = proc.model {
            vec![m.clone()]
        } else {
            Vec::new()
        };
        orphans.push(asmi_core::UnmanagedProcess {
            pid: proc.pid,
            port: proc.port,
            engine: proc.framework.to_string(),
            models,
            source: "external",
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
        _ => asmi_core::ServeEngine::MlxLm, // fallback
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

    let mgr = get_manager(&state, Some(port))
        .ok_or_else(|| ApiError::NotFound(format!("unknown port: {port}")))?;

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
        // Non-distributed load — stop heartbeat if running
        state.peer_heartbeat.stop().await;
    }

    let engine = req.engine;
    mgr.load(req).await;
    Ok(Json(serde_json::json!({"ok": true, "state": "loading", "engine": engine, "port": port})))
}

/// POST /serve/stop → stop the running server on a port.
async fn serve_stop_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let port = q.port.unwrap_or(DEFAULT_SERVE_PORT);
    let mgr = get_manager(&state, q.port)
        .ok_or_else(|| ApiError::NotFound(format!("unknown port: {port}")))?;
    mgr.stop().await;
    state.peer_heartbeat.stop().await;
    Ok(Json(serde_json::json!({"ok": true, "port": port})))
}

/// POST /serve/reload → reload the current model on a port.
async fn serve_reload_handler(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ServeQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let port = q.port.unwrap_or(DEFAULT_SERVE_PORT);
    let mgr = get_manager(&state, q.port)
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

    // Build command: python3 -m mlx_lm.server --model <path> --port <port> --host 0.0.0.0
    let mut cmd = Command::new(&py);
    cmd.arg("-m").arg("mlx_lm.server")
        .arg("--model").arg(&model_path)
        .arg("--port").arg(req.port.to_string())
        .arg("--host").arg("0.0.0.0");

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
    let mut current_iface = String::new();

    for line in ifconfig_out.lines() {
        // Interface header line: "bridge100: flags=..."  or  "en3: flags=..."
        if !line.starts_with('\t') && !line.starts_with(' ') {
            if let Some(name) = line.split(':').next() {
                current_iface = name.to_string();
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
                if ip.starts_with("169.254.") && !current_iface.starts_with("lo") {
                    link_local.push(serde_json::json!({
                        "interface": current_iface,
                        "ip": ip,
                    }));
                }
            }
        }
    }

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

    // 3. Get ARP peers (169.254.x.x) and ping each one
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
    // An interface with an Active RDMA device but no link-local IP (e.g. en4 with
    // 192.168.x.x from Internet Sharing) cannot reach 169.254 peers.
    let link_local_ifaces: std::collections::HashSet<String> = link_local
        .iter()
        .filter_map(|v| v.get("interface").and_then(|i| i.as_str()).map(|s| s.to_string()))
        .collect();

    let mut peer_ips: Vec<(String, String)> = Vec::new(); // (ip, interface)
    for line in arp_out.lines() {
        if line.contains("incomplete") || !line.contains("169.254.") {
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
            peer_ips.push((ip.to_string(), iface.to_string()));
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
        "link_local_ips": link_local,
        "peer_pings": ping_results,
        "peers_reachable": peers_reachable,
        "peers_total": peers_total,
        "rdma_links": rdma_links,
        "topology": topology,
    }))
}

/// Known RDMA cluster nodes with their Tailscale IPs.
/// These nodes have TB5 direct connections for RDMA.
const RDMA_NODES: &[(&str, &str)] = &[
    ("m3u2", "100.125.18.51"),
    ("m3u1", "100.127.90.10"),
    ("m3u3", "100.112.127.66"),
    ("m4m1", "100.78.120.10"),
];

/// GET /rdma/mesh → query all RDMA nodes via Tailscale, aggregate mesh health.
///
/// This is the "single pane of glass" endpoint. Hit it from your phone,
/// laptop, or any Tailscale-connected device to see the full RDMA mesh status.
async fn rdma_mesh_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    // Query all RDMA nodes' /rdma/check endpoint in parallel via Tailscale
    let futures: Vec<_> = RDMA_NODES
        .iter()
        .map(|(name, ts_ip)| {
            let client = client.clone();
            let name = name.to_string();
            let url = format!("http://{}:9090/rdma/check", ts_ip);
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
                    "summary": "unreachable via Tailscale",
                }));
            }
        }
    }

    let mesh_healthy = nodes_healthy == nodes_online && nodes_online == RDMA_NODES.len();

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
            nodes_healthy, RDMA_NODES.len(), total_active, total_reachable_peers, total_peers
        ),
        "nodes_online": nodes_online,
        "nodes_total": RDMA_NODES.len(),
        "nodes_healthy": nodes_healthy,
        "total_rdma_active": total_active,
        "total_peers_reachable": total_reachable_peers,
        "total_peers": total_peers,
        "topology": mesh_topology,
        "nodes": nodes,
        "queried_via": "tailscale",
        "queried_from": state.hostname,
    }))
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/health/setup", get(setup_handler))
        .route("/health/network", get(network_health_handler))
        .route("/health/network/fix", post(network_fix_handler))
        .route("/processes", get(processes_handler))
        .route("/models", get(models_handler))
        .route("/volumes", get(volumes_handler))
        .route("/logs", get(logs_handler))
        .route("/runtime", get(runtime_handler))
        .route("/cluster", get(cluster_handler))
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
        .route("/rdma/mesh", get(rdma_mesh_handler))
        // Experimental ANE compute endpoints (gated by --experimental-ane + --features ane)
        .route("/ane/compute", get(crate::ane::status_handler))
        .route("/ane/eval", post(crate::ane::eval_handler))
        .route("/ane/probe", get(crate::ane::probe_handler))
        .with_state(state)
}
