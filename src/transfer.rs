//! Native RDMA file transfer via JACCL.
//!
//! Two endpoints:
//! - `POST /transfer`        — initiate a file transfer (source side)
//! - `POST /transfer/accept`  — peer coordination (destination side)
//!
//! Files are streamed in 64 MB chunks over RDMA. After transfer, SHA-256
//! verification confirms integrity. The JACCL group is cached in `AppState`
//! for potential reuse.
//!
//! Everything is gated behind `#[cfg(feature = "jaccl")]`. Without the
//! feature, both endpoints return 501 Not Implemented.

use axum::{
    extract::State,
    response::{Json, IntoResponse},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use crate::daemon::AppState;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Request / response types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Deserialize)]
pub struct TransferRequest {
    /// Model directory name (relative to ~/Models/)
    pub model_dir: String,
    /// Peer hostname (e.g. "m3u3")
    pub peer: String,
    /// "send" or "recv"
    pub direction: String,
}

#[derive(Deserialize, Serialize)]
pub struct TransferAcceptRequest {
    /// Model directory name (relative to ~/Models/)
    pub model_dir: String,
    /// Coordinator IP (LAN)
    pub coordinator_ip: String,
    /// Coordinator port for JACCL side-channel
    pub coordinator_port: i32,
    /// This node's rank (always 1 for the peer)
    pub rank: i32,
    /// World size (always 2 for point-to-point)
    pub size: i32,
}

#[derive(Serialize)]
struct SseEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transport: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    percent: Option<u8>,
    #[serde(rename = "bytesPerSec", skip_serializing_if = "Option::is_none")]
    bytes_per_sec: Option<u64>,
    #[serde(rename = "durationMs", skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files: Option<u64>,
    #[serde(rename = "totalBytes", skip_serializing_if = "Option::is_none")]
    total_bytes: Option<u64>,
}

impl SseEvent {
    fn stage(name: &str) -> Self {
        SseEvent {
            event_type: "stage".into(),
            stage: Some(name.into()),
            transport: Some("jaccl-rdma".into()),
            percent: None, bytes_per_sec: None, duration_ms: None,
            error: None, files: None, total_bytes: None,
        }
    }

    fn progress(pct: u8, bps: u64) -> Self {
        SseEvent {
            event_type: "progress".into(),
            stage: None, transport: None,
            percent: Some(pct), bytes_per_sec: Some(bps),
            duration_ms: None, error: None, files: None, total_bytes: None,
        }
    }

    fn done(duration_ms: u64) -> Self {
        SseEvent {
            event_type: "done".into(),
            stage: None, transport: Some("jaccl-rdma".into()),
            percent: None, bytes_per_sec: None,
            duration_ms: Some(duration_ms),
            error: None, files: None, total_bytes: None,
        }
    }

    fn error(msg: &str) -> Self {
        SseEvent {
            event_type: "error".into(),
            stage: None, transport: Some("jaccl-rdma".into()),
            percent: None, bytes_per_sec: None, duration_ms: None,
            error: Some(msg.into()), files: None, total_bytes: None,
        }
    }

    fn to_sse_line(&self) -> String {
        format!("data: {}\n\n", serde_json::to_string(self).unwrap_or_default())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Group cache wrapper (used in AppState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Wrapper to store JacclGroup in AppState.
///
/// JacclGroup is `!Send + !Sync` (C++ handle). We wrap it so the HashMap
/// itself can live in an `Arc<Mutex<>>`. The Mutex ensures exclusive access
/// and the group is only ever touched from a single dedicated thread.
#[cfg(feature = "jaccl")]
pub struct TransferGroupHandle {
    _group: asmi_core::jaccl_ffi::JacclGroup,
}

// SAFETY: JacclGroup is only accessed from one thread at a time.
// The Mutex<HashMap<..>> ensures no concurrent access.
#[cfg(feature = "jaccl")]
unsafe impl Send for TransferGroupHandle {}
#[cfg(feature = "jaccl")]
unsafe impl Sync for TransferGroupHandle {}

/// Stub when jaccl is not compiled in.
#[cfg(not(feature = "jaccl"))]
pub struct TransferGroupHandle {
    _phantom: (),
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Stub handlers (no jaccl feature)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(not(feature = "jaccl"))]
pub async fn transfer_handler(
    State(_state): State<AppState>,
    Json(_req): Json<TransferRequest>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "RDMA transfer requires --features jaccl"
        })),
    )
}

#[cfg(not(feature = "jaccl"))]
pub async fn transfer_accept_handler(
    State(_state): State<AppState>,
    Json(_req): Json<TransferAcceptRequest>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "RDMA transfer requires --features jaccl"
        })),
    )
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Native JACCL handlers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// POST /transfer — initiate a file transfer.
///
/// Streams SSE events to the caller with stage/progress/done/error updates.
#[cfg(feature = "jaccl")]
pub async fn transfer_handler(
    State(state): State<AppState>,
    Json(req): Json<TransferRequest>,
) -> impl IntoResponse {
    use axum::body::Body;
    use futures::StreamExt;
    use tokio::sync::mpsc;

    let direction = req.direction.to_lowercase();
    if direction != "send" && direction != "recv" {
        return (
            StatusCode::BAD_REQUEST,
            axum::response::Response::builder()
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"error": "direction must be 'send' or 'recv'"}).to_string(),
                ))
                .unwrap(),
        );
    }

    // SSE channel — handler streams events to the caller.
    let (tx, rx) = mpsc::channel::<String>(64);

    // Spawn the async pipeline (which internally uses a dedicated OS thread for JACCL).
    let jaccl_groups = state.jaccl_groups.clone();
    tokio::spawn(async move {
        transfer_pipeline(tx, jaccl_groups, req).await;
    });

    // Convert the mpsc receiver into an SSE body stream.
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream.map(|line| Ok::<_, std::convert::Infallible>(line)));

    (
        StatusCode::OK,
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(body)
            .unwrap(),
    )
}

/// POST /transfer/accept — peer coordination endpoint.
///
/// Called by the coordinator's asmi on the target node. Spawns a background
/// thread to init the JACCL group (rank 1) and receive files.
#[cfg(feature = "jaccl")]
pub async fn transfer_accept_handler(
    State(state): State<AppState>,
    Json(req): Json<TransferAcceptRequest>,
) -> impl IntoResponse {
    if req.model_dir.is_empty() || req.coordinator_ip.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "model_dir and coordinator_ip required"})),
        );
    }

    let models_root = dirs::home_dir()
        .map(|h| h.join("Models"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/Models"));
    let model_path = models_root.join(&req.model_dir);

    let jaccl_groups = state.jaccl_groups.clone();
    let coordinator_ip = req.coordinator_ip.clone();
    let coordinator_port = req.coordinator_port;
    let model_dir = req.model_dir.clone();

    // All JACCL work runs on a dedicated OS thread (JacclGroup is !Send).
    std::thread::spawn(move || {
        accept_worker(
            &coordinator_ip,
            coordinator_port,
            &model_path,
            &model_dir,
            jaccl_groups,
        );
    });

    // Return immediately — peer runs in background.
    (
        StatusCode::OK,
        Json(serde_json::json!({"ok": true, "state": "accepting"})),
    )
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Transfer pipeline (coordinator / source side)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(feature = "jaccl")]
async fn transfer_pipeline(
    tx: tokio::sync::mpsc::Sender<String>,
    jaccl_groups: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, TransferGroupHandle>>>,
    req: TransferRequest,
) {
    use asmi_core::jaccl_ffi;

    let start = std::time::Instant::now();

    // ── 1. Preflight: PD probe ──────────────────────────────────────────
    let _ = tx.send(SseEvent::stage("preflight").to_sse_line()).await;

    let pd_ok = tokio::task::spawn_blocking(|| jaccl_ffi::pd_probe_any_active())
        .await
        .unwrap_or(-1);

    if pd_ok == 0 {
        let _ = tx.send(SseEvent::error("PD exhausted — reboot required (shutdown -h, wait 10s)").to_sse_line()).await;
        return;
    }
    if pd_ok < 0 {
        let _ = tx.send(SseEvent::error("RDMA not available on this host (no libibverbs)").to_sse_line()).await;
        return;
    }

    // ── 2. Resolve model directory ───────────────────────────────────────
    let models_root = dirs::home_dir()
        .map(|h| h.join("Models"))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/Models"));
    let model_path = models_root.join(&req.model_dir);

    if req.direction == "send" && !model_path.is_dir() {
        let _ = tx.send(SseEvent::error(&format!("model directory not found: {}", model_path.display())).to_sse_line()).await;
        return;
    }

    // ── 3. Resolve peer IP (mDNS → Tailscale fallback) ──────────────────
    let _ = tx.send(SseEvent::stage("coordinate").to_sse_line()).await;

    // ── 4. Pick a dynamic port ──────────────────────────────────────────
    let coordinator_port = pick_dynamic_port();

    // ── 5. Resolve our own LAN IP (for coordinator) ─────────────────────
    let local_ip = match resolve_local_lan_ip() {
        Some(ip) => ip,
        None => {
            let _ = tx.send(SseEvent::error("cannot determine local LAN IP").to_sse_line()).await;
            return;
        }
    };

    // ── 6. Notify peer to accept ────────────────────────────────────────
    let peer_req = TransferAcceptRequest {
        model_dir: req.model_dir.clone(),
        coordinator_ip: local_ip.clone(),
        coordinator_port,
        rank: 1,
        size: 2,
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    // Resolve peer IP via system getent/ping (reqwest async DNS can't do mDNS)
    let peer_ip = match tokio::process::Command::new("python3")
        .args(["-c", &format!(
            "import socket; r=socket.getaddrinfo('{}.local',9090,socket.AF_INET); print(r[0][4][0])",
            req.peer
        )])
        .output().await
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => {
            let _ = tx.send(SseEvent::error(&format!("cannot resolve peer '{}'", req.peer)).to_sse_line()).await;
            return;
        }
    };
    let peer_urls = vec![
        format!("http://{}:9090/transfer/accept", peer_ip),
    ];
    let mut peer_ok = false;
    let mut last_err = String::new();
    for peer_url in &peer_urls {
        match client.post(peer_url).json(&peer_req).send().await {
            Ok(r) if r.status().is_success() => { peer_ok = true; break; }
            Ok(r) => {
                last_err = r.text().await.unwrap_or_default();
            }
            Err(e) => {
                last_err = format!("{e:#}");
            }
        }
    }
    if !peer_ok {
        let _ = tx.send(SseEvent::error(&format!("peer unreachable: {last_err}")).to_sse_line()).await;
        return;
    }

    // ── 7–9. Init + Transfer + Verify ───────────────────────────────────
    // All JACCL work runs on a dedicated OS thread because JacclGroup is !Send.
    let _ = tx.send(SseEvent::stage("init").to_sse_line()).await;

    let direction = req.direction.clone();
    let model_dir_key = req.model_dir.clone();
    let peer_key = req.peer.clone();
    let model_path_clone = model_path.clone();
    let tx_clone = tx.clone();

    let result = tokio::task::spawn_blocking(move || {
        coordinator_worker(
            &local_ip,
            coordinator_port,
            &model_path_clone,
            &direction,
            tx_clone,
        )
    })
    .await;

    match result {
        Ok(Ok((total_bytes, group_handle))) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let mut done = SseEvent::done(duration_ms);
            done.total_bytes = Some(total_bytes);
            let _ = tx.send(done.to_sse_line()).await;

            // Cache the group for reuse
            let cache_key = format!("{}:{}", peer_key, model_dir_key);
            if let Ok(mut groups) = jaccl_groups.lock() {
                groups.insert(cache_key, group_handle);
            }
        }
        Ok(Err(e)) => {
            let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
        }
        Err(e) => {
            let _ = tx.send(SseEvent::error(&format!("worker thread panicked: {e}")).to_sse_line()).await;
        }
    }
}

/// All JACCL work for the coordinator (rank 0) — runs on a dedicated OS thread.
///
/// Returns `(total_bytes_transferred, group_handle)` on success.
#[cfg(feature = "jaccl")]
fn coordinator_worker(
    local_ip: &str,
    coordinator_port: i32,
    model_path: &std::path::Path,
    direction: &str,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<(u64, TransferGroupHandle), String> {
    use asmi_core::jaccl_ffi;

    let group = jaccl_ffi::JacclGroup::new_auto(
        0,                  // rank 0 = coordinator
        2,                  // world_size
        local_ip,
        coordinator_port,
        30_000,             // 30s handshake timeout
    ).ok_or_else(|| "JACCL group init failed (handshake timeout or device error)".to_string())?;

    // Transfer
    let _ = tx.blocking_send(SseEvent::stage("transfer").to_sse_line());

    let total_bytes = if direction == "send" {
        do_send(&group, model_path, &tx)?
    } else {
        do_recv(model_path, &group, &tx)?
    };

    // Verify
    let _ = tx.blocking_send(SseEvent::stage("verify").to_sse_line());
    do_verify(&group, model_path, direction, &tx)?;
    let _ = tx.blocking_send(SseEvent::stage("verified").to_sse_line());

    Ok((total_bytes, TransferGroupHandle { _group: group }))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Accept worker (peer / rank 1 side)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// All JACCL work for the peer (rank 1) — runs on a dedicated OS thread.
#[cfg(feature = "jaccl")]
fn accept_worker(
    coordinator_ip: &str,
    coordinator_port: i32,
    model_path: &std::path::Path,
    model_dir: &str,
    jaccl_groups: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, TransferGroupHandle>>>,
) {
    use asmi_core::jaccl_ffi;

    let group = match jaccl_ffi::JacclGroup::new_auto(
        1,                  // rank 1 = peer
        2,                  // world_size
        coordinator_ip,
        coordinator_port,
        30_000,             // 30s timeout
    ) {
        Some(g) => g,
        None => {
            tracing::error!("transfer/accept: JACCL group init failed for {model_dir}");
            return;
        }
    };

    // Dummy progress channel (peer has no SSE stream)
    let (tx, _rx) = tokio::sync::mpsc::channel::<String>(8);

    // Ensure target directory exists
    if let Err(e) = std::fs::create_dir_all(model_path) {
        tracing::error!("transfer/accept: create dir failed for {model_dir}: {e}");
        return;
    }

    // Receive files
    match do_recv(model_path, &group, &tx) {
        Ok(bytes) => {
            tracing::info!("transfer/accept: received {bytes} bytes for {model_dir}");

            // Send verification checksums back to sender
            if let Err(e) = do_verify(&group, model_path, "recv", &tx) {
                tracing::error!("transfer/accept: verify failed for {model_dir}: {e}");
                return;
            }

            // Cache group
            if let Ok(mut groups) = jaccl_groups.lock() {
                groups.insert(
                    format!("accept:{model_dir}"),
                    TransferGroupHandle { _group: group },
                );
            }
        }
        Err(e) => {
            tracing::error!("transfer/accept: recv failed for {model_dir}: {e}");
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Constants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 64 MB chunk size for RDMA transfers.
#[cfg(feature = "jaccl")]
const CHUNK_SIZE: usize = 64 * 1024 * 1024;

/// Timeout per chunk (60s — large enough for 64 MB even on degraded links).
#[cfg(feature = "jaccl")]
const CHUNK_TIMEOUT_MS: i32 = 60_000;

/// Header message timeout (short — just metadata).
#[cfg(feature = "jaccl")]
const HEADER_TIMEOUT_MS: i32 = 10_000;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Wire types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// File manifest sent as the first RDMA message (JSON-encoded).
#[cfg(feature = "jaccl")]
#[derive(Serialize, Deserialize)]
struct FileManifest {
    files: Vec<FileEntry>,
}

#[cfg(feature = "jaccl")]
#[derive(Serialize, Deserialize)]
struct FileEntry {
    name: String,
    size: u64,
}

/// SHA-256 verification manifest sent after all data.
#[cfg(feature = "jaccl")]
#[derive(Serialize, Deserialize)]
struct VerifyManifest {
    checksums: Vec<FileChecksum>,
}

#[cfg(feature = "jaccl")]
#[derive(Serialize, Deserialize)]
struct FileChecksum {
    name: String,
    sha256: String,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Send / Recv / Verify (blocking, single-threaded — called from OS thread)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Send all files in `model_path` to rank 1.
#[cfg(feature = "jaccl")]
fn do_send(
    group: &asmi_core::jaccl_ffi::JacclGroup,
    model_path: &std::path::Path,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<u64, String> {
    // Collect files
    let entries: Vec<FileEntry> = std::fs::read_dir(model_path)
        .map_err(|e| format!("read model dir: {e}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let meta = entry.metadata().ok()?;
            if meta.is_file() {
                Some(FileEntry {
                    name: entry.file_name().to_string_lossy().to_string(),
                    size: meta.len(),
                })
            } else {
                None
            }
        })
        .collect();

    if entries.is_empty() {
        return Err("no files in model directory".into());
    }

    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let manifest = FileManifest { files: entries };
    let manifest_json = serde_json::to_vec(&manifest)
        .map_err(|e| format!("serialize manifest: {e}"))?;

    // Send manifest length (8 bytes) + manifest (padded to 1 MB)
    let manifest_len = manifest_json.len();
    let length_buf = (manifest_len as u64).to_le_bytes();
    group.send(&length_buf, 1, HEADER_TIMEOUT_MS)
        .map_err(|e| format!("send manifest length: {e}"))?;

    let mut header_buf = vec![0u8; 1024 * 1024];
    let copy_len = manifest_len.min(header_buf.len());
    header_buf[..copy_len].copy_from_slice(&manifest_json[..copy_len]);
    group.send(&header_buf, 1, HEADER_TIMEOUT_MS)
        .map_err(|e| format!("send manifest: {e}"))?;

    // Send each file in chunks
    let mut bytes_sent: u64 = 0;

    for file_entry in &manifest.files {
        let file_path = model_path.join(&file_entry.name);
        let file_data = std::fs::read(&file_path)
            .map_err(|e| format!("read {}: {e}", file_entry.name))?;

        let mut offset = 0usize;
        while offset < file_data.len() {
            let end = (offset + CHUNK_SIZE).min(file_data.len());
            let chunk_len = end - offset;

            // Send chunk size (8 bytes) then chunk data
            let size_buf = (chunk_len as u64).to_le_bytes();
            group.send(&size_buf, 1, CHUNK_TIMEOUT_MS)
                .map_err(|e| format!("send chunk size for {}: {e}", file_entry.name))?;
            group.send(&file_data[offset..end], 1, CHUNK_TIMEOUT_MS)
                .map_err(|e| format!("send chunk for {}: {e}", file_entry.name))?;

            offset = end;
            bytes_sent += chunk_len as u64;

            let pct = ((bytes_sent as f64 / total_bytes as f64) * 100.0) as u8;
            let _ = tx.blocking_send(SseEvent::progress(pct.min(99), 0).to_sse_line());
        }
    }

    Ok(bytes_sent)
}

/// Receive files from rank 0 into `model_path`.
#[cfg(feature = "jaccl")]
fn do_recv(
    model_path: &std::path::Path,
    group: &asmi_core::jaccl_ffi::JacclGroup,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<u64, String> {
    // Ensure target directory exists
    std::fs::create_dir_all(model_path)
        .map_err(|e| format!("create model dir: {e}"))?;

    // Receive manifest length (8 bytes)
    let mut length_buf = [0u8; 8];
    group.recv(&mut length_buf, 0, HEADER_TIMEOUT_MS)
        .map_err(|e| format!("recv manifest length: {e}"))?;
    let manifest_len = u64::from_le_bytes(length_buf) as usize;

    // Receive manifest (1 MB buffer)
    let mut header_buf = vec![0u8; 1024 * 1024];
    group.recv(&mut header_buf, 0, HEADER_TIMEOUT_MS)
        .map_err(|e| format!("recv manifest: {e}"))?;

    let manifest: FileManifest = serde_json::from_slice(&header_buf[..manifest_len])
        .map_err(|e| format!("parse manifest: {e}"))?;

    let total_bytes: u64 = manifest.files.iter().map(|e| e.size).sum();
    let mut bytes_received: u64 = 0;

    for file_entry in &manifest.files {
        let file_path = model_path.join(&file_entry.name);
        let mut file_data = Vec::with_capacity(file_entry.size as usize);

        let mut remaining = file_entry.size as usize;
        while remaining > 0 {
            // Receive chunk size (8 bytes)
            let mut size_buf = [0u8; 8];
            group.recv(&mut size_buf, 0, CHUNK_TIMEOUT_MS)
                .map_err(|e| format!("recv chunk size for {}: {e}", file_entry.name))?;
            let chunk_len = u64::from_le_bytes(size_buf) as usize;

            // Receive chunk data
            let mut chunk_buf = vec![0u8; chunk_len];
            group.recv(&mut chunk_buf, 0, CHUNK_TIMEOUT_MS)
                .map_err(|e| format!("recv chunk for {}: {e}", file_entry.name))?;

            file_data.extend_from_slice(&chunk_buf);
            remaining = remaining.saturating_sub(chunk_len);
            bytes_received += chunk_len as u64;

            let pct = ((bytes_received as f64 / total_bytes as f64) * 100.0) as u8;
            let _ = tx.blocking_send(SseEvent::progress(pct.min(99), 0).to_sse_line());
        }

        // Write file
        std::fs::write(&file_path, &file_data)
            .map_err(|e| format!("write {}: {e}", file_entry.name))?;
    }

    Ok(bytes_received)
}

/// SHA-256 verification exchange.
///
/// - direction="send": compute local checksums, receive peer's, compare.
/// - direction="recv": compute local checksums, send to peer.
#[cfg(feature = "jaccl")]
fn do_verify(
    group: &asmi_core::jaccl_ffi::JacclGroup,
    model_path: &std::path::Path,
    direction: &str,
    _tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<(), String> {
    if direction == "send" {
        let local_checksums = compute_checksums(model_path)?;

        // Receive verification manifest from peer
        let mut length_buf = [0u8; 8];
        group.recv(&mut length_buf, 1, HEADER_TIMEOUT_MS)
            .map_err(|e| format!("recv verify length: {e}"))?;
        let verify_len = u64::from_le_bytes(length_buf) as usize;

        let mut verify_buf = vec![0u8; 1024 * 1024];
        group.recv(&mut verify_buf, 1, HEADER_TIMEOUT_MS)
            .map_err(|e| format!("recv verify manifest: {e}"))?;

        let remote: VerifyManifest = serde_json::from_slice(&verify_buf[..verify_len])
            .map_err(|e| format!("parse verify manifest: {e}"))?;

        // Compare
        for local in &local_checksums.checksums {
            match remote.checksums.iter().find(|r| r.name == local.name) {
                Some(r) if r.sha256 == local.sha256 => {}
                Some(r) => {
                    return Err(format!(
                        "checksum mismatch for {}: local={} remote={}",
                        local.name, local.sha256, r.sha256
                    ));
                }
                None => {
                    return Err(format!("file {} missing on peer", local.name));
                }
            }
        }
    } else {
        // Receiver: compute checksums and send to sender
        let checksums = compute_checksums(model_path)?;
        let verify_json = serde_json::to_vec(&checksums)
            .map_err(|e| format!("serialize verify: {e}"))?;

        let verify_len = verify_json.len();
        let length_buf = (verify_len as u64).to_le_bytes();
        group.send(&length_buf, 0, HEADER_TIMEOUT_MS)
            .map_err(|e| format!("send verify length: {e}"))?;

        let mut verify_buf = vec![0u8; 1024 * 1024];
        let copy_len = verify_len.min(verify_buf.len());
        verify_buf[..copy_len].copy_from_slice(&verify_json[..copy_len]);
        group.send(&verify_buf, 0, HEADER_TIMEOUT_MS)
            .map_err(|e| format!("send verify manifest: {e}"))?;
    }

    Ok(())
}

/// Compute SHA-256 checksums for all files in a directory.
#[cfg(feature = "jaccl")]
fn compute_checksums(model_path: &std::path::Path) -> Result<VerifyManifest, String> {
    use sha2::{Sha256, Digest};

    let mut checksums = Vec::new();

    for entry in std::fs::read_dir(model_path)
        .map_err(|e| format!("read dir for verify: {e}"))?
    {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let meta = entry.metadata().map_err(|e| format!("metadata: {e}"))?;
        if meta.is_file() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let data = std::fs::read(&path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let hash = format!("{:x}", hasher.finalize());
            checksums.push(FileChecksum { name, sha256: hash });
        }
    }

    Ok(VerifyManifest { checksums })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resolve a peer hostname to an IPv4 address.
/// Prefers IPv4 over IPv6 — link-local IPv6 (fe80::) needs scope IDs
/// that reqwest doesn't handle. Tries mDNS first, then bare hostname.
#[cfg(feature = "jaccl")]
async fn resolve_peer_ip(peer: &str) -> Option<String> {
    use tokio::net::lookup_host;

    // Try mDNS first — prefer IPv4
    let mdns = format!("{}.local:9090", peer);
    if let Ok(addrs) = lookup_host(&mdns).await {
        let all: Vec<_> = addrs.collect();
        if let Some(v4) = all.iter().find(|a| a.is_ipv4()) {
            return Some(v4.ip().to_string());
        }
        if let Some(addr) = all.first() {
            return Some(addr.ip().to_string());
        }
    }

    // Fallback: bare hostname (Tailscale / /etc/hosts) — prefer IPv4
    let bare = format!("{}:9090", peer);
    if let Ok(addrs) = lookup_host(&bare).await {
        let all: Vec<_> = addrs.collect();
        if let Some(v4) = all.iter().find(|a| a.is_ipv4()) {
            return Some(v4.ip().to_string());
        }
        if let Some(addr) = all.first() {
            return Some(addr.ip().to_string());
        }
    }

    None
}

/// Resolve this machine's LAN IP (non-loopback, non-link-local).
/// Uses the UDP-connect trick: open a UDP socket to an external IP,
/// the OS picks the right source address.
#[cfg(feature = "jaccl")]
fn resolve_local_lan_ip() -> Option<String> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("10.255.255.255:1").ok()?;
    let local_addr = sock.local_addr().ok()?;
    Some(local_addr.ip().to_string())
}

/// Pick a random high port for JACCL coordinator.
/// Binds to :0, lets the OS pick, then closes — the port is briefly available.
#[cfg(feature = "jaccl")]
fn pick_dynamic_port() -> i32 {
    std::net::TcpListener::bind("0.0.0.0:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port() as i32)
        .unwrap_or(49200 + (std::process::id() as i32 % 1000))
}
