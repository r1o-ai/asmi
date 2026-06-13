//! Native RDMA file transfer via JACCL.
//!
//! Endpoints:
//! - `POST /transfer`        — fire-and-forget: starts transfer, returns {id, status} immediately
//! - `GET  /transfer/status`  — all active/completed transfers with progress
//! - `GET  /transfer/:id/log` — SSE logtail for a specific transfer (disconnect-safe)
//! - `POST /transfer/accept`  — peer coordination (destination side)
//!
//! The transfer runs to completion server-side regardless of client connection.
//! Progress is stored in-memory (ActiveTransfers). Clients can poll /status or
//! tail /log — disconnecting either has no effect on the transfer.
//!
//! Phase 3: All JACCL operations go through a dedicated worker thread via
//! an mpsc channel. The worker owns all JacclGroup handles (which are !Send),
//! processes commands sequentially, and calls the C shim's detach-based
//! send/recv (Phase 1b) to avoid head-of-line blocking.
//!
//! Everything is gated behind `#[cfg(feature = "jaccl")]`. Without the
//! feature, both endpoints return 501 Not Implemented.

use axum::{
    extract::{Path as AxumPath, State},
    response::{Json, IntoResponse},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::daemon::AppState;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Transfer state (shared across handlers — the transfer owns this, not the HTTP connection)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub type ActiveTransfers = Arc<RwLock<HashMap<String, TransferState>>>;

#[derive(Clone, Serialize)]
pub struct TransferState {
    pub id: String,
    pub peer: String,
    pub direction: String,
    pub model_dir: String,
    pub status: String,
    pub percent: u8,
    #[serde(rename = "bytesPerSec")]
    pub bytes_per_sec: u64,
    #[serde(rename = "totalBytes")]
    pub total_bytes: u64,
    #[serde(rename = "bytesSent")]
    pub bytes_sent: u64,
    pub error: Option<String>,
    #[serde(rename = "startedAt")]
    pub started_at_epoch: u64,
    #[serde(rename = "completedAt")]
    pub completed_at_epoch: Option<u64>,
    #[serde(skip)]
    pub log: Vec<String>,
}

#[cfg(feature = "jaccl")]
impl TransferState {
    fn push_log(&mut self, event: &SseEvent) {
        self.log.push(event.to_sse_line());
        if self.log.len() > 200 {
            self.log.drain(..100);
        }
    }
}

#[cfg(feature = "jaccl")]
fn generate_transfer_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("tx-{:x}", ts)
}

#[cfg(feature = "jaccl")]
fn epoch_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Request / response types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Fields are only read by the jaccl transfer pipeline; the no-jaccl stub
// deserializes the payload but ignores it to keep the HTTP API shape stable.
#[cfg_attr(not(feature = "jaccl"), allow(dead_code))]
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
    /// RDMA device the peer should use (topology-resolved by coordinator)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_device: Option<String>,
    /// RDMA device the coordinator uses (needed for peer's devices.json matrix)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinator_device: Option<String>,
}

#[cfg(feature = "jaccl")]
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

#[cfg(feature = "jaccl")]
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
// JACCL Worker Thread + Channel API (Phase 3)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Commands sent to the JACCL worker thread via mpsc channel.
#[cfg(feature = "jaccl")]
pub enum JacclCmd {
    /// Get an existing group for `peer` or init a new one.
    /// Returns the coordinator port (reused or newly picked).
    GetOrInitGroup {
        peer: String,
        rank: i32,
        coordinator_ip: String,
        coordinator_port: Option<i32>, // None = pick new port
        /// Topology-resolved RDMA device on this node (e.g. "rdma_en15").
        local_device: Option<String>,
        /// Topology-resolved RDMA device on the peer (e.g. "rdma_en10").
        peer_device: Option<String>,
        reply: tokio::sync::oneshot::Sender<Result<i32, String>>,
    },
    /// Send data to rank `dst` on the group for `peer`.
    Send {
        peer: String,
        data: Vec<u8>,
        dst: i32,
        timeout_ms: i32,
        reply: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    /// Receive `len` bytes from rank `src` on the group for `peer`.
    Recv {
        peer: String,
        len: usize,
        src: i32,
        timeout_ms: i32,
        reply: tokio::sync::oneshot::Sender<Result<Vec<u8>, String>>,
    },
    /// Probe group liveness for `peer`.
    /// Handled by the worker loop and backed by jaccl_group_probe FFI;
    /// the HTTP endpoint that sends this is not wired up yet.
    #[allow(dead_code)]
    Probe {
        peer: String,
        reply: tokio::sync::oneshot::Sender<bool>,
    },
    /// Drop the group for `peer`, freeing PDs.
    /// Worker-side handled; send-side endpoint pending (see Probe).
    #[allow(dead_code)]
    DropGroup {
        peer: String,
    },
    /// Poison and cancel all pending ops on `peer`'s group.
    /// Worker-side handled; send-side endpoint pending (see Probe).
    #[allow(dead_code)]
    CancelPending {
        peer: String,
    },
    /// Bulk send a file — tight loop on the worker thread, no per-chunk channel hops.
    BulkSendFile {
        peer: String,
        file_path: std::path::PathBuf,
        file_size: u64,
        dst: i32,
        progress: std::sync::mpsc::Sender<(u64, u64)>, // (bytes_sent, total)
        reply: tokio::sync::oneshot::Sender<Result<u64, String>>,
    },
    /// Bulk receive a file — tight loop on the worker thread.
    BulkRecvFile {
        peer: String,
        file_path: std::path::PathBuf,
        file_size: u64,
        src: i32,
        progress: std::sync::mpsc::Sender<(u64, u64)>,
        reply: tokio::sync::oneshot::Sender<Result<u64, String>>,
    },
}

/// The JACCL worker — owns all groups, runs on a dedicated OS thread.
#[cfg(feature = "jaccl")]
pub struct JacclWorker {
    tx: std::sync::mpsc::Sender<JacclCmd>,
}

#[cfg(feature = "jaccl")]
impl JacclWorker {
    /// Spawn the worker thread and return the command sender.
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<JacclCmd>();

        std::thread::Builder::new()
            .name("jaccl-worker".into())
            .spawn(move || {
                Self::run(rx);
            })
            .expect("failed to spawn jaccl-worker thread");

        JacclWorker { tx }
    }

    /// Send a command to the worker. Returns Err if the worker thread panicked.
    pub fn send(&self, cmd: JacclCmd) -> Result<(), String> {
        self.tx.send(cmd).map_err(|_| "jaccl worker thread is dead".to_string())
    }

    /// Worker thread main loop — processes commands sequentially.
    /// Owns `HashMap<String, (JacclGroup, i32)>` — group + cached port.
    fn run(rx: std::sync::mpsc::Receiver<JacclCmd>) {
        use asmi_core::jaccl_ffi;
        use std::collections::HashMap;

        // peer_key → (JacclGroup, coordinator_port)
        let mut groups: HashMap<String, (jaccl_ffi::JacclGroup, i32)> = HashMap::new();

        while let Ok(cmd) = rx.recv() {
            match cmd {
                JacclCmd::GetOrInitGroup { peer, rank, coordinator_ip, coordinator_port, local_device, peer_device, reply } => {
                    // Check if we have a cached group that's alive
                    if let Some((group, port)) = groups.get(&peer) {
                        if group.probe() {
                            let _ = reply.send(Ok(*port));
                            continue;
                        }
                        tracing::info!(%peer, "cached JACCL group is stale, re-initializing");
                    }
                    // Remove stale group (if any)
                    if let Some((old, _)) = groups.remove(&peer) {
                        old.cancel_pending();
                        drop(old);
                    }

                    let port = coordinator_port.unwrap_or_else(pick_dynamic_port);

                    let result = match (&local_device, &peer_device) {
                        (Some(local_dev), Some(peer_dev)) => {
                            match write_devices_json(rank, local_dev, peer_dev) {
                                Ok(path) => {
                                    tracing::info!(%peer, %local_dev, %peer_dev, "using topology-resolved RDMA devices");
                                    let group = jaccl_ffi::JacclGroup::new(
                                        rank, 2, &coordinator_ip, port, &path, 30_000,
                                    );
                                    let _ = std::fs::remove_file(&path);
                                    group
                                }
                                Err(e) => {
                                    tracing::warn!(%peer, error = %e, "devices.json write failed, falling back to auto");
                                    jaccl_ffi::JacclGroup::new_auto(rank, 2, &coordinator_ip, port, 30_000)
                                }
                            }
                        }
                        _ => {
                            tracing::warn!(%peer, "no topology device info, using auto-discover");
                            jaccl_ffi::JacclGroup::new_auto(rank, 2, &coordinator_ip, port, 30_000)
                        }
                    };

                    match result {
                        Some(group) => {
                            groups.insert(peer, (group, port));
                            let _ = reply.send(Ok(port));
                        }
                        None => {
                            let _ = reply.send(Err("JACCL group init failed (handshake timeout or device error)".into()));
                        }
                    }
                }

                JacclCmd::Send { peer, data, dst, timeout_ms, reply } => {
                    match groups.get(&peer) {
                        Some((group, _)) => {
                            let result = group.send(&data, dst, timeout_ms);
                            let _ = reply.send(result.map_err(|e| e.to_string()));
                        }
                        None => {
                            let _ = reply.send(Err(format!("no JACCL group for peer '{peer}'")));
                        }
                    }
                }

                JacclCmd::Recv { peer, len, src, timeout_ms, reply } => {
                    match groups.get(&peer) {
                        Some((group, _)) => {
                            let mut buf = vec![0u8; len];
                            let result = group.recv(&mut buf, src, timeout_ms);
                            let _ = reply.send(result.map(|_| buf).map_err(|e| e.to_string()));
                        }
                        None => {
                            let _ = reply.send(Err(format!("no JACCL group for peer '{peer}'")));
                        }
                    }
                }

                JacclCmd::Probe { peer, reply } => {
                    let alive = groups.get(&peer).map(|(g, _)| g.probe()).unwrap_or(false);
                    let _ = reply.send(alive);
                }

                JacclCmd::DropGroup { peer } => {
                    if let Some((group, _)) = groups.remove(&peer) {
                        group.cancel_pending();
                        drop(group);
                    }
                }

                JacclCmd::CancelPending { peer } => {
                    if let Some((group, _)) = groups.get(&peer) {
                        group.cancel_pending();
                    }
                }

                JacclCmd::BulkSendFile { peer, file_path, file_size, dst, progress, reply } => {
                    match groups.get(&peer) {
                        Some((group, _)) => {
                            let result = bulk_send_file(group, &file_path, file_size, dst, &progress);
                            let _ = reply.send(result);
                        }
                        None => {
                            let _ = reply.send(Err(format!("no JACCL group for peer '{peer}'")));
                        }
                    }
                }

                JacclCmd::BulkRecvFile { peer, file_path, file_size, src, progress, reply } => {
                    match groups.get(&peer) {
                        Some((group, _)) => {
                            let result = bulk_recv_file(group, &file_path, file_size, src, &progress);
                            let _ = reply.send(result);
                        }
                        None => {
                            let _ = reply.send(Err(format!("no JACCL group for peer '{peer}'")));
                        }
                    }
                }
            }
        }

        tracing::info!("jaccl-worker thread exiting, dropping {} groups", groups.len());
        // Groups drop here — RAII cleans up PDs
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bulk file transfer — tight loop on the worker thread, zero channel hops
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 4 MB chunk — large enough to amortize RDMA overhead, small enough for
/// JACCL's internal pipelining. The old per-channel path used 256 KB.
#[cfg(feature = "jaccl")]
const BULK_CHUNK: usize = 4 * 1024 * 1024;

/// Timeout per bulk chunk (5 min — generous for degraded links).
#[cfg(feature = "jaccl")]
const BULK_CHUNK_TIMEOUT_MS: i32 = 300_000;

/// Send an entire file over RDMA in a tight loop on the worker thread.
///
/// Protocol: send 8-byte LE file size, then stream 4 MB chunks until done.
/// Progress is reported via a `std::sync::mpsc` channel (not tokio — we're
/// on a plain OS thread).
#[cfg(feature = "jaccl")]
fn bulk_send_file(
    group: &asmi_core::jaccl_ffi::JacclGroup,
    path: &std::path::Path,
    file_size: u64,
    dst: i32,
    progress: &std::sync::mpsc::Sender<(u64, u64)>,
) -> Result<u64, String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = vec![0u8; BULK_CHUNK];
    let mut sent: u64 = 0;

    // Send file size header (8 bytes)
    group.send(&file_size.to_le_bytes(), dst, 30_000)
        .map_err(|e| format!("send file size: {e}"))?;

    while sent < file_size {
        let to_read = BULK_CHUNK.min((file_size - sent) as usize);
        file.read_exact(&mut buf[..to_read])
            .map_err(|e| format!("read {} at offset {sent}: {e}", path.display()))?;
        group.send(&buf[..to_read], dst, BULK_CHUNK_TIMEOUT_MS)
            .map_err(|e| format!("send chunk at {sent}: {e}"))?;
        sent += to_read as u64;
        let _ = progress.send((sent, file_size));
    }

    Ok(sent)
}

/// Receive an entire file over RDMA in a tight loop on the worker thread.
///
/// Protocol: recv 8-byte LE file size, then stream 4 MB chunks until done.
#[cfg(feature = "jaccl")]
fn bulk_recv_file(
    group: &asmi_core::jaccl_ffi::JacclGroup,
    path: &std::path::Path,
    file_size: u64,
    src: i32,
    progress: &std::sync::mpsc::Sender<(u64, u64)>,
) -> Result<u64, String> {
    use std::io::Write;

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
    }

    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut buf = vec![0u8; BULK_CHUNK];
    let mut received: u64 = 0;

    // Receive file size header (8 bytes)
    let mut size_buf = [0u8; 8];
    group.recv(&mut size_buf, src, 30_000)
        .map_err(|e| format!("recv file size: {e}"))?;
    let wire_size = u64::from_le_bytes(size_buf);

    if wire_size != file_size {
        return Err(format!(
            "file size mismatch: manifest says {file_size}, wire says {wire_size}"
        ));
    }

    while received < file_size {
        let to_recv = BULK_CHUNK.min((file_size - received) as usize);
        group.recv(&mut buf[..to_recv], src, BULK_CHUNK_TIMEOUT_MS)
            .map_err(|e| format!("recv chunk at {received}: {e}"))?;
        file.write_all(&buf[..to_recv])
            .map_err(|e| format!("write {} at offset {received}: {e}", path.display()))?;
        received += to_recv as u64;
        let _ = progress.send((received, file_size));
    }

    Ok(received)
}

/// Stub when jaccl is not compiled in.
#[cfg(not(feature = "jaccl"))]
pub struct JacclWorker;

#[cfg(not(feature = "jaccl"))]
impl JacclWorker {
    pub fn new() -> Self { JacclWorker }
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

/// POST /transfer — fire-and-forget: starts transfer, returns {id, status} immediately.
///
/// The transfer runs to completion server-side. Poll GET /transfer/status or
/// tail GET /transfer/:id/log for progress. Client disconnect has no effect.
#[cfg(feature = "jaccl")]
pub async fn transfer_handler(
    State(state): State<AppState>,
    Json(req): Json<TransferRequest>,
) -> impl IntoResponse {
    let direction = req.direction.to_lowercase();
    if direction != "send" && direction != "recv" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "direction must be 'send' or 'recv'"})),
        );
    }

    let id = generate_transfer_id();

    // Register the transfer in shared state before spawning.
    let transfer = TransferState {
        id: id.clone(),
        peer: req.peer.clone(),
        direction: direction.clone(),
        model_dir: req.model_dir.clone(),
        status: "started".into(),
        percent: 0,
        bytes_per_sec: 0,
        total_bytes: 0,
        bytes_sent: 0,
        error: None,
        started_at_epoch: epoch_now(),
        completed_at_epoch: None,
        log: Vec::new(),
    };
    state.active_transfers.write().await.insert(id.clone(), transfer);

    // Spawn fire-and-forget — pipeline writes to shared state via drainer, not HTTP.
    let transfers = state.active_transfers.clone();
    let jaccl_worker = state.jaccl_worker.clone();
    let topology_cache = state.topology_cache.clone();
    let hostname = state.hostname.clone();
    let transfer_id = id.clone();
    tokio::spawn(async move {
        run_transfer(transfers, transfer_id, jaccl_worker, topology_cache, hostname, req).await;
    });

    (StatusCode::OK, Json(serde_json::json!({"id": id, "status": "started"})))
}

/// GET /transfer/status — all active/completed transfers with progress.
pub async fn transfer_status_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let map = state.active_transfers.read().await;
    let transfers: Vec<&TransferState> = map.values().collect();
    Json(serde_json::json!({"transfers": transfers}))
}

/// GET /transfer/:id/log — SSE logtail for a specific transfer. Disconnect-safe.
pub async fn transfer_log_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    use axum::body::Body;
    use futures::StreamExt;

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

    let transfers = state.active_transfers.clone();
    tokio::spawn(async move {
        let mut cursor = 0usize;
        loop {
            let (log_slice, is_done) = {
                let map = transfers.read().await;
                match map.get(&id) {
                    Some(state) => {
                        let new_lines: Vec<String> = state.log[cursor..].to_vec();
                        let done = state.status == "done" || state.status == "error";
                        (new_lines, done)
                    }
                    None => {
                        let _ = tx.send("data: {\"error\":\"transfer not found\"}\n\n".into()).await;
                        return;
                    }
                }
            };
            for line in &log_slice {
                if tx.send(line.clone()).await.is_err() {
                    return; // client disconnected — just stop tailing, transfer continues
                }
            }
            cursor += log_slice.len();
            if is_done {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream.map(|line| Ok::<_, std::convert::Infallible>(line)));

    axum::response::Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap()
}

/// POST /transfer/accept — peer coordination endpoint.
///
/// Called by the coordinator's asmi on the target node. Sends init command
/// to the JACCL worker thread, then receives files.
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

    let jaccl_worker = state.jaccl_worker.clone();
    let coordinator_ip = req.coordinator_ip.clone();
    let coordinator_port = req.coordinator_port;
    let model_dir = req.model_dir.clone();
    let peer_device = req.peer_device.clone();
    let coordinator_device = req.coordinator_device.clone();

    // Spawn a task to handle accept (uses JACCL worker channel internally).
    tokio::spawn(async move {
        accept_worker(
            &coordinator_ip,
            coordinator_port,
            &model_path,
            &model_dir,
            jaccl_worker,
            peer_device,
            coordinator_device,
        ).await;
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
async fn run_transfer(
    transfers: ActiveTransfers,
    transfer_id: String,
    jaccl_worker: std::sync::Arc<JacclWorker>,
    topology_cache: TopologyCache,
    local_hostname: String,
    req: TransferRequest,
) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

    // Drainer task: reads SSE lines from the pipeline and writes to shared state.
    // Runs independently — no HTTP coupling. Parses events to update progress fields.
    let drain_transfers = transfers.clone();
    let drain_id = transfer_id.clone();
    tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            if let Some(state) = drain_transfers.write().await.get_mut(&drain_id) {
                state.push_log(&SseEvent::stage("_raw"));
                // Parse the SSE data line for progress/stage/done/error
                if let Some(json_str) = line.strip_prefix("data: ").and_then(|s| s.strip_suffix("\n\n")) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                        match v.get("type").and_then(|t| t.as_str()) {
                            Some("progress") => {
                                state.percent = v.get("percent").and_then(|p| p.as_u64()).unwrap_or(0) as u8;
                                state.bytes_per_sec = v.get("bytesPerSec").and_then(|b| b.as_u64()).unwrap_or(0);
                            }
                            Some("stage") => {
                                if let Some(s) = v.get("stage").and_then(|s| s.as_str()) {
                                    state.status = s.to_string();
                                }
                            }
                            Some("done") => {
                                state.status = "done".into();
                                state.percent = 100;
                                state.total_bytes = v.get("totalBytes").and_then(|b| b.as_u64()).unwrap_or(state.bytes_sent);
                                state.completed_at_epoch = Some(epoch_now());
                            }
                            Some("error") => {
                                state.status = "error".into();
                                state.error = v.get("error").and_then(|e| e.as_str()).map(|s| s.to_string());
                                state.completed_at_epoch = Some(epoch_now());
                            }
                            _ => {}
                        }
                    }
                }
                state.log.push(line);
                if state.log.len() > 200 { state.log.drain(..100); }
            }
        }
    });

    transfer_pipeline(tx, jaccl_worker, topology_cache, &local_hostname, req).await;
}

#[cfg(feature = "jaccl")]
async fn transfer_pipeline(
    tx: tokio::sync::mpsc::Sender<String>,
    jaccl_worker: std::sync::Arc<JacclWorker>,
    topology_cache: TopologyCache,
    local_hostname: &str,
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

    // ── 3. Resolve peer IP (native mDNS — Phase 4) ──────────────────────
    let _ = tx.send(SseEvent::stage("coordinate").to_sse_line()).await;

    // ── 4. Resolve coordinator IP ────────────────────────────────────────
    // Use /30 IP from the RDMA hardware port for the coordinator TCP side-channel.
    // LAN IP gives errno 65 (EHOSTUNREACH) from inside the JACCL worker thread,
    // despite working from curl/nc. /30 is the direct point-to-point TB5 path.
    let devices = resolve_device_for_peer(&topology_cache, local_hostname, &req.peer).await;
    let local_ip = devices.as_ref()
        .and_then(|(ld, _)| {
            let hw_port = ld.strip_prefix("rdma_")?;
            let out = std::process::Command::new("sh")
                .args(["-c", &format!("ifconfig {} | grep 'inet ' | grep -v '127\\|169.254' | awk '{{print $2}}' | head -1", hw_port)])
                .output().ok()?;
            let ip = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if ip.is_empty() { None } else {
                tracing::info!(hw_port, %ip, "coordinator IP from RDMA port");
                Some(ip)
            }
        })
        .or_else(|| resolve_local_lan_ip())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    tracing::info!(peer = %req.peer, coordinator_ip = %local_ip, "resolved coordinator IP");

    // ── 5. Resolve devices + notify peer + init coordinator ───────────
    //
    // Ordering is critical: jaccl_init_mesh blocks until all ranks connect.
    // The coordinator (rank 0) opens a TCP listener; rank 1 connects to it.
    // If we await init before notifying the peer, we deadlock.
    //
    // Flow: pick port → notify peer → start coordinator init → await both.
    let _ = tx.send(SseEvent::stage("init").to_sse_line()).await;

    // devices already resolved in step 4 for coordinator IP
    if let Some((ref ld, ref pd)) = devices {
        tracing::info!(peer = %req.peer, local_device = %ld, peer_device = %pd, "topology-resolved RDMA devices");
    }

    let coordinator_port = pick_dynamic_port();

    // 5a. Start coordinator init (non-blocking mpsc send — worker thread
    //     opens TCP listener while we notify the peer).
    let (init_reply_tx, init_reply_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = jaccl_worker.send(JacclCmd::GetOrInitGroup {
        peer: req.peer.clone(),
        rank: 0,
        coordinator_ip: local_ip.clone(),
        coordinator_port: Some(coordinator_port),
        local_device: devices.as_ref().map(|(ld, _)| ld.clone()),
        peer_device: devices.as_ref().map(|(_, pd)| pd.clone()),
        reply: init_reply_tx,
    }) {
        let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
        return;
    }

    // 5b. Yield — let the worker thread pick up the command and open
    //     the TCP listener before rank 1 tries to connect. 5s to
    //     account for worker thread scheduling + PD alloc + CQ creation.
    tokio::time::sleep(std::time::Duration::from_millis(5000)).await;

    // 5c. Notify peer to connect (coordinator listener is already up).
    let peer_req = TransferAcceptRequest {
        model_dir: req.model_dir.clone(),
        coordinator_ip: local_ip.clone(),
        coordinator_port,
        rank: 1,
        size: 2,
        peer_device: devices.as_ref().map(|(_, pd)| pd.clone()),
        coordinator_device: devices.as_ref().map(|(ld, _)| ld.clone()),
    };

    // Resolve peer for the coordination POST (/transfer/accept).
    // Priority: /etc/hosts (LAN) → mDNS (.local) → Tailscale (last resort).
    // LAN is preferred because Tailscale can be slow/flaky post-reboot and the
    // JACCL init has a 30s timeout — a slow coordination POST wastes that budget.
    let peer_host = resolve_peer_lan(&req.peer)
        .or_else(|| resolve_peer_mdns_only(&req.peer))
        .or_else(|| resolve_peer_tailscale(&req.peer))
        .unwrap_or_else(|| format!("{}.local", req.peer));
    tracing::info!(peer = %req.peer, peer_host = %peer_host, "resolved peer IP for coordination");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();
    let peer_url = format!("http://{}:9090/transfer/accept", peer_host);
    match client.post(&peer_url).json(&peer_req).send().await {
        Ok(r) if r.status().is_success() => {}
        Ok(r) => {
            let err = r.text().await.unwrap_or_default();
            let _ = tx.send(SseEvent::error(&format!("peer unreachable: {err}")).to_sse_line()).await;
            return;
        }
        Err(e) => {
            let _ = tx.send(SseEvent::error(&format!("peer unreachable: {e:#}")).to_sse_line()).await;
            return;
        }
    }

    // 5d. Await coordinator init result (both sides are handshaking).
    match init_reply_rx.await {
        Ok(Ok(_port)) => {}
        Ok(Err(e)) => {
            let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
            return;
        }
        Err(_) => {
            let _ = tx.send(SseEvent::error("jaccl worker dropped reply channel").to_sse_line()).await;
            return;
        }
    }

    // ── 7–9. Transfer + Verify via worker channel ───────────────────────
    let _ = tx.send(SseEvent::stage("transfer").to_sse_line()).await;

    let direction = req.direction.clone();
    let peer_key = req.peer.clone();
    let model_path_clone = model_path.clone();
    let tx_clone = tx.clone();
    let worker_clone = jaccl_worker.clone();

    let result = if direction == "send" {
        do_send_via_worker(&worker_clone, &peer_key, &model_path_clone, &tx_clone).await
    } else {
        do_recv_via_worker(&worker_clone, &peer_key, &model_path_clone, &tx_clone).await
    };

    match result {
        Ok(total_bytes) => {
            // Verify
            let _ = tx.send(SseEvent::stage("verify").to_sse_line()).await;
            match do_verify_via_worker(&worker_clone, &peer_key, &model_path, &direction, &tx).await {
                Ok(()) => {
                    let _ = tx.send(SseEvent::stage("verified").to_sse_line()).await;
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let mut done = SseEvent::done(duration_ms);
                    done.total_bytes = Some(total_bytes);
                    let _ = tx.send(done.to_sse_line()).await;
                }
                Err(e) => {
                    let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
                }
            }
        }
        Err(e) => {
            let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Accept worker (peer / rank 1 side)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(feature = "jaccl")]
async fn accept_worker(
    coordinator_ip: &str,
    coordinator_port: i32,
    model_path: &std::path::Path,
    model_dir: &str,
    jaccl_worker: std::sync::Arc<JacclWorker>,
    peer_device: Option<String>,
    coordinator_device: Option<String>,
) {
    // Validate coordinator-supplied device name against local ibverbs
    let validated_local = match peer_device {
        Some(ref dev) => {
            let d = dev.clone();
            let probe = tokio::task::spawn_blocking(move || {
                asmi_core::jaccl_ffi::pd_probe(&d)
            }).await.unwrap_or(-1);
            if probe > 0 {
                Some(dev.clone())
            } else {
                tracing::warn!(%dev, probe, "coordinator-supplied device not valid locally, using auto");
                None
            }
        }
        None => None,
    };

    let peer_key = format!("accept:{model_dir}");

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = jaccl_worker.send(JacclCmd::GetOrInitGroup {
        peer: peer_key.clone(),
        rank: 1,
        coordinator_ip: coordinator_ip.to_string(),
        coordinator_port: Some(coordinator_port),
        local_device: validated_local,
        peer_device: coordinator_device,
        reply: reply_tx,
    }) {
        tracing::error!("transfer/accept: worker send failed for {model_dir}: {e}");
        return;
    }

    match reply_rx.await {
        Ok(Ok(_port)) => {}
        Ok(Err(e)) => {
            tracing::error!("transfer/accept: JACCL group init failed for {model_dir}: {e}");
            return;
        }
        Err(_) => {
            tracing::error!("transfer/accept: worker dropped reply for {model_dir}");
            return;
        }
    }

    // Ensure target directory exists
    if let Err(e) = std::fs::create_dir_all(model_path) {
        tracing::error!("transfer/accept: create dir failed for {model_dir}: {e}");
        return;
    }

    // Dummy progress channel (peer has no SSE stream)
    let (tx, _rx) = tokio::sync::mpsc::channel::<String>(8);

    // Receive files via worker
    match do_recv_via_worker(&jaccl_worker, &peer_key, model_path, &tx).await {
        Ok(bytes) => {
            tracing::info!("transfer/accept: received {bytes} bytes for {model_dir}");

            // Send verification checksums back to sender
            if let Err(e) = do_verify_via_worker(&jaccl_worker, &peer_key, model_path, "recv", &tx).await {
                tracing::error!("transfer/accept: verify failed for {model_dir}: {e}");
                return;
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

/// Header / small-message timeout (manifests, checksums, length headers).
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
// Worker-based Send / Recv / Verify helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Helper to send data through the worker channel.
#[cfg(feature = "jaccl")]
async fn worker_send(
    worker: &JacclWorker,
    peer: &str,
    data: &[u8],
    dst: i32,
    timeout_ms: i32,
) -> Result<(), String> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    worker.send(JacclCmd::Send {
        peer: peer.to_string(),
        data: data.to_vec(),
        dst,
        timeout_ms,
        reply: reply_tx,
    })?;
    reply_rx.await.map_err(|_| "worker dropped reply".to_string())?
}

/// Helper to receive data through the worker channel.
#[cfg(feature = "jaccl")]
async fn worker_recv(
    worker: &JacclWorker,
    peer: &str,
    len: usize,
    src: i32,
    timeout_ms: i32,
) -> Result<Vec<u8>, String> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    worker.send(JacclCmd::Recv {
        peer: peer.to_string(),
        len,
        src,
        timeout_ms,
        reply: reply_tx,
    })?;
    reply_rx.await.map_err(|_| "worker dropped reply".to_string())?
}

/// Send all files in `model_path` to rank 1 via bulk worker commands.
///
/// Manifest exchange still uses the small `worker_send`/`worker_recv` path.
/// File data goes through `BulkSendFile` — the entire file streams in a
/// tight loop on the worker thread with zero per-chunk channel hops.
#[cfg(feature = "jaccl")]
async fn do_send_via_worker(
    worker: &JacclWorker,
    peer: &str,
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

    let start = std::time::Instant::now();
    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let manifest = FileManifest { files: entries };
    let manifest_json = serde_json::to_vec(&manifest)
        .map_err(|e| format!("serialize manifest: {e}"))?;

    // Send manifest length (8 bytes) + manifest (padded to 1 MB) — small, uses channel path
    let manifest_len = manifest_json.len();
    let length_buf = (manifest_len as u64).to_le_bytes();
    worker_send(worker, peer, &length_buf, 1, HEADER_TIMEOUT_MS).await
        .map_err(|e| format!("send manifest length: {e}"))?;

    let mut header_buf = vec![0u8; 1024 * 1024];
    let copy_len = manifest_len.min(header_buf.len());
    header_buf[..copy_len].copy_from_slice(&manifest_json[..copy_len]);
    worker_send(worker, peer, &header_buf, 1, HEADER_TIMEOUT_MS).await
        .map_err(|e| format!("send manifest: {e}"))?;

    // Send each file via BulkSendFile — tight loop on the worker thread
    let mut bytes_sent: u64 = 0;

    for file_entry in &manifest.files {
        let file_path = model_path.join(&file_entry.name);

        // Progress channel: std::sync::mpsc (worker is a plain OS thread)
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<(u64, u64)>();
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();

        worker.send(JacclCmd::BulkSendFile {
            peer: peer.to_string(),
            file_path: file_path.clone(),
            file_size: file_entry.size,
            dst: 1,
            progress: progress_tx,
            reply: reply_tx,
        })?;

        // Drain progress channel in a background task to emit SSE events
        let tx_sse = tx.clone();
        let file_name = file_entry.name.clone();
        let total = total_bytes;
        let base_sent = bytes_sent;
        let transfer_start = start;
        tokio::task::spawn_blocking(move || {
            while let Ok((file_bytes, _file_total)) = progress_rx.recv() {
                let global_sent = base_sent + file_bytes;
                let pct = ((global_sent as f64 / total as f64) * 100.0) as u8;
                let elapsed = transfer_start.elapsed().as_secs();
                let bps = if elapsed > 0 { global_sent / elapsed } else { 0 };
                // Best-effort SSE — drop if the channel is full
                let _ = tx_sse.try_send(SseEvent::progress(pct.min(99), bps).to_sse_line());
            }
            tracing::debug!("progress drain done for {file_name}");
        });

        // Await the worker's reply for this file
        let file_bytes = reply_rx.await
            .map_err(|_| format!("worker dropped reply for {}", file_entry.name))?
            .map_err(|e| format!("bulk send {}: {e}", file_entry.name))?;

        bytes_sent += file_bytes;
    }

    Ok(bytes_sent)
}

/// Receive files from rank 0 into `model_path` via bulk worker commands.
///
/// Manifest exchange still uses the small `worker_send`/`worker_recv` path.
/// File data goes through `BulkRecvFile` — tight loop on the worker thread.
#[cfg(feature = "jaccl")]
async fn do_recv_via_worker(
    worker: &JacclWorker,
    peer: &str,
    model_path: &std::path::Path,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<u64, String> {
    // Ensure target directory exists
    std::fs::create_dir_all(model_path)
        .map_err(|e| format!("create model dir: {e}"))?;

    // Receive manifest length (8 bytes) — small, uses channel path
    let length_data = worker_recv(worker, peer, 8, 0, HEADER_TIMEOUT_MS).await
        .map_err(|e| format!("recv manifest length: {e}"))?;
    let manifest_len = u64::from_le_bytes(length_data[..8].try_into().unwrap()) as usize;

    // Receive manifest (1 MB buffer)
    let header_data = worker_recv(worker, peer, 1024 * 1024, 0, HEADER_TIMEOUT_MS).await
        .map_err(|e| format!("recv manifest: {e}"))?;

    let manifest: FileManifest = serde_json::from_slice(&header_data[..manifest_len])
        .map_err(|e| format!("parse manifest: {e}"))?;

    let total_bytes: u64 = manifest.files.iter().map(|e| e.size).sum();
    let mut bytes_received: u64 = 0;

    let start = std::time::Instant::now();
    for file_entry in &manifest.files {
        let file_path = model_path.join(&file_entry.name);

        // Progress channel: std::sync::mpsc (worker is a plain OS thread)
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<(u64, u64)>();
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();

        worker.send(JacclCmd::BulkRecvFile {
            peer: peer.to_string(),
            file_path: file_path.clone(),
            file_size: file_entry.size,
            src: 0,
            progress: progress_tx,
            reply: reply_tx,
        })?;

        // Drain progress channel in a background task to emit SSE events
        let tx_sse = tx.clone();
        let file_name = file_entry.name.clone();
        let total = total_bytes;
        let base_received = bytes_received;
        let transfer_start = start;
        tokio::task::spawn_blocking(move || {
            while let Ok((file_bytes, _file_total)) = progress_rx.recv() {
                let global_received = base_received + file_bytes;
                let pct = ((global_received as f64 / total as f64) * 100.0) as u8;
                let elapsed = transfer_start.elapsed().as_secs();
                let bps = if elapsed > 0 { global_received / elapsed } else { 0 };
                let _ = tx_sse.try_send(SseEvent::progress(pct.min(99), bps).to_sse_line());
            }
            tracing::debug!("progress drain done for {file_name}");
        });

        // Await the worker's reply for this file
        let file_bytes = reply_rx.await
            .map_err(|_| format!("worker dropped reply for {}", file_entry.name))?
            .map_err(|e| format!("bulk recv {}: {e}", file_entry.name))?;

        bytes_received += file_bytes;
    }

    Ok(bytes_received)
}

/// SHA-256 verification exchange via worker channel.
#[cfg(feature = "jaccl")]
async fn do_verify_via_worker(
    worker: &JacclWorker,
    peer: &str,
    model_path: &std::path::Path,
    direction: &str,
    _tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<(), String> {
    if direction == "send" {
        let local_checksums = compute_checksums(model_path)?;

        // Receive verification manifest from peer
        let length_data = worker_recv(worker, peer, 8, 1, HEADER_TIMEOUT_MS).await
            .map_err(|e| format!("recv verify length: {e}"))?;
        let verify_len = u64::from_le_bytes(length_data[..8].try_into().unwrap()) as usize;

        let verify_data = worker_recv(worker, peer, 1024 * 1024, 1, HEADER_TIMEOUT_MS).await
            .map_err(|e| format!("recv verify manifest: {e}"))?;

        let remote: VerifyManifest = serde_json::from_slice(&verify_data[..verify_len])
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
        worker_send(worker, peer, &length_buf, 0, HEADER_TIMEOUT_MS).await
            .map_err(|e| format!("send verify length: {e}"))?;

        let mut verify_buf = vec![0u8; 1024 * 1024];
        let copy_len = verify_len.min(verify_buf.len());
        verify_buf[..copy_len].copy_from_slice(&verify_json[..copy_len]);
        worker_send(worker, peer, &verify_buf, 0, HEADER_TIMEOUT_MS).await
            .map_err(|e| format!("send verify manifest: {e}"))?;
    }

    Ok(())
}

/// Compute SHA-256 checksums for all files in a directory, hashing files in
/// parallel (one scoped thread per file) and streaming each in 8 MiB chunks to
/// bound memory. The RDMA transfer runs at ~5 GB/s; serial whole-file SHA-256
/// was the dominant end-to-end cost (e.g. ~70s of a 76s 28 GB transfer), so
/// parallel + streaming keeps integrity while cutting verify time ~N-fold.
#[cfg(feature = "jaccl")]
fn compute_checksums(model_path: &std::path::Path) -> Result<VerifyManifest, String> {
    use sha2::{Sha256, Digest};
    use std::io::Read;

    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(model_path)
        .map_err(|e| format!("read dir for verify: {e}"))?
    {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let meta = entry.metadata().map_err(|e| format!("metadata: {e}"))?;
        if meta.is_file() {
            files.push((entry.file_name().to_string_lossy().to_string(), entry.path()));
        }
    }

    // Hash files concurrently; each thread streams its file so peak memory is
    // bounded by (file_count × 8 MiB), not the model size.
    let results: Vec<Result<FileChecksum, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = files
            .iter()
            .map(|(name, path)| {
                s.spawn(move || -> Result<FileChecksum, String> {
                    let mut f = std::fs::File::open(path)
                        .map_err(|e| format!("open {}: {e}", path.display()))?;
                    let mut hasher = Sha256::new();
                    let mut buf = vec![0u8; 8 * 1024 * 1024];
                    loop {
                        let n = f
                            .read(&mut buf)
                            .map_err(|e| format!("read {}: {e}", path.display()))?;
                        if n == 0 {
                            break;
                        }
                        hasher.update(&buf[..n]);
                    }
                    Ok(FileChecksum {
                        name: name.clone(),
                        sha256: format!("{:x}", hasher.finalize()),
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("verify hash thread panicked".to_string())))
            .collect()
    });

    let mut checksums = Vec::with_capacity(results.len());
    for r in results {
        checksums.push(r?);
    }
    Ok(VerifyManifest { checksums })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Topology-aware device resolution
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(feature = "jaccl")]
type TopologyCache = std::sync::Arc<tokio::sync::RwLock<Option<(crate::topology::TopologyReport, std::time::Instant)>>>;

/// Look up the RDMA devices connecting `local_hostname` ↔ `peer` from the
/// cached topology report. Returns `(local_device, peer_device)`.
///
/// Validates that the local device actually exists in ibverbs (the topology
/// layer may report network interface names that don't match RDMA device names).
#[cfg(feature = "jaccl")]
async fn resolve_device_for_peer(
    topology_cache: &TopologyCache,
    local_hostname: &str,
    peer: &str,
) -> Option<(String, String)> {
    let cache = topology_cache.read().await;
    let (report, _) = cache.as_ref()?;

    for link in &report.links {
        let (local_dev, peer_dev) = if link.node_a == local_hostname && link.node_b == peer {
            (link.device_a.clone(), link.device_b.clone())
        } else if link.node_b == local_hostname && link.node_a == peer {
            (link.device_b.clone(), link.device_a.clone())
        } else {
            continue;
        };

        // Reject ARP-fallback interface names (iface:en15 is NOT an RDMA device)
        if local_dev.starts_with("iface:") || peer_dev.starts_with("iface:") {
            tracing::warn!(
                %local_dev, %peer_dev,
                "topology has network interface names, not RDMA devices — falling back to auto-discover"
            );
            return None;
        }

        // Validate: check that local device exists via ibverbs PD probe.
        // Returns -1 if device doesn't exist, 0 if PD exhausted, 1 if OK.
        let dev_check = local_dev.clone();
        let probe = tokio::task::spawn_blocking(move || {
            asmi_core::jaccl_ffi::pd_probe(&dev_check)
        }).await.unwrap_or(-1);

        if probe < 0 {
            tracing::warn!(
                %local_dev, %peer_dev,
                "topology device not found in ibverbs, falling back to auto-discover"
            );
            return None;
        }
        if probe == 0 {
            tracing::warn!(%local_dev, "PD exhausted on topology-resolved device");
            return None;
        }

        return Some((local_dev, peer_dev));
    }
    None
}

/// Write a 2-node JACCL devices.json to /tmp and return the path.
///
/// Format: `devices[rank][dst]` — each rank reads only its own row.
/// - `devices[0][1]` = device rank 0 uses to reach rank 1 (= local_dev for coordinator)
/// - `devices[1][0]` = device rank 1 uses to reach rank 0 (= peer_dev for peer)
#[cfg(feature = "jaccl")]
fn write_devices_json(rank: i32, local_dev: &str, peer_dev: &str) -> Result<String, String> {
    // devices[rank][dst]: the device RANK uses to reach DST.
    // The matrix is a global view — always the same regardless of who writes it.
    let (rank0_dev, rank1_dev) = if rank == 0 {
        (local_dev, peer_dev)
    } else {
        (peer_dev, local_dev)
    };
    let devices = serde_json::json!([[null, rank0_dev], [rank1_dev, null]]);

    let path = format!("/tmp/jaccl-devices-{}.json", std::process::id());
    std::fs::write(&path, devices.to_string())
        .map_err(|e| format!("write devices json: {e}"))?;
    Ok(path)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resolve peer via /etc/hosts — returns LAN IP (10.x.x.x). Fastest, most reliable.
#[cfg(feature = "jaccl")]
fn resolve_peer_lan(peer: &str) -> Option<String> {
    let hosts = std::fs::read_to_string("/etc/hosts").ok()?;
    for line in hosts.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let ip = parts[0];
            if !ip.starts_with("10.") && !ip.starts_with("192.168.") { continue; }
            // Skip 192.168.10.x (/30 TB5 IPs — not routable from all nodes)
            if ip.starts_with("192.168.10.") { continue; }
            for &name in &parts[1..] {
                if name.eq_ignore_ascii_case(peer)
                    || name.eq_ignore_ascii_case(&format!("{}.local", peer))
                {
                    tracing::info!(%peer, %ip, "resolved peer via /etc/hosts (LAN)");
                    return Some(ip.to_string());
                }
            }
        }
    }
    None
}

/// Resolve peer via mDNS (.local) only — no Tailscale.
/// std's ToSocketAddrs goes through getaddrinfo → mDNSResponder on macOS.
#[cfg(feature = "jaccl")]
fn resolve_peer_mdns_only(peer: &str) -> Option<String> {
    use std::net::ToSocketAddrs;

    let addr = format!("{}.local:0", peer)
        .to_socket_addrs()
        .ok()?
        .find(|a| a.is_ipv4())?;
    let ip = addr.ip().to_string();
    tracing::info!(%peer, %ip, "resolved peer via mDNS");
    Some(ip)
}

/// Resolve peer IP via `tailscale status --json`. Stable IPs that survive reboots.
#[cfg(feature = "jaccl")]
fn resolve_peer_tailscale(peer: &str) -> Option<String> {
    let output = std::process::Command::new("tailscale")
        .args(["status", "--json"])
        .output()
        .ok()?;
    if !output.status.success() { return None; }
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    // Search peers
    if let Some(peers) = json.get("Peer").and_then(|p| p.as_object()) {
        for (_id, info) in peers {
            let hostname = info.get("HostName").and_then(|h| h.as_str()).unwrap_or("");
            if hostname.eq_ignore_ascii_case(peer) {
                if let Some(ips) = info.get("TailscaleIPs").and_then(|i| i.as_array()) {
                    // Return first IPv4 (skip IPv6)
                    for ip in ips {
                        if let Some(s) = ip.as_str() {
                            if !s.contains(':') { return Some(s.to_string()); }
                        }
                    }
                }
            }
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
/// Retries up to 3 times as defense-in-depth against TOCTOU (Phase 5).
#[cfg(feature = "jaccl")]
fn pick_dynamic_port() -> i32 {
    for _ in 0..3 {
        if let Ok(listener) = std::net::TcpListener::bind("0.0.0.0:0") {
            if let Ok(addr) = listener.local_addr() {
                let port = addr.port() as i32;
                drop(listener);
                return port;
            }
        }
    }
    49200 + (std::process::id() as i32 % 1000)
}
