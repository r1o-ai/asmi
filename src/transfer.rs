//! Native RDMA file transfer via JACCL.
//!
//! Two endpoints:
//! - `POST /transfer`        — initiate a file transfer (source side)
//! - `POST /transfer/accept`  — peer coordination (destination side)
//!
//! Files are streamed in 64 MB chunks over RDMA. After transfer, SHA-256
//! verification confirms integrity.
//!
//! Phase 3: All JACCL operations go through a dedicated worker thread via
//! an mpsc channel. The worker owns all JacclGroup handles (which are !Send),
//! processes commands sequentially, and calls the C shim's detach-based
//! send/recv (Phase 1b) to avoid head-of-line blocking.
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
    Probe {
        peer: String,
        reply: tokio::sync::oneshot::Sender<bool>,
    },
    /// Drop the group for `peer`, freeing PDs.
    DropGroup {
        peer: String,
    },
    /// Poison and cancel all pending ops on `peer`'s group.
    CancelPending {
        peer: String,
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
                JacclCmd::GetOrInitGroup { peer, rank, coordinator_ip, coordinator_port, reply } => {
                    // Check if we have a cached group that's alive
                    if let Some((group, port)) = groups.get(&peer) {
                        if group.probe() {
                            let _ = reply.send(Ok(*port));
                            continue;
                        }
                        // Dead group — remove and re-init
                        tracing::info!(%peer, "cached JACCL group is stale, re-initializing");
                    }
                    // Remove stale group (if any)
                    if let Some((old, _)) = groups.remove(&peer) {
                        old.cancel_pending();
                        drop(old);
                    }

                    let port = coordinator_port.unwrap_or_else(pick_dynamic_port);

                    let result = jaccl_ffi::JacclGroup::new_auto(
                        rank,
                        2, // world_size always 2 for point-to-point
                        &coordinator_ip,
                        port,
                        30_000, // 30s handshake timeout
                    );

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
            }
        }

        tracing::info!("jaccl-worker thread exiting, dropping {} groups", groups.len());
        // Groups drop here — RAII cleans up PDs
    }
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

    // Spawn the async pipeline.
    let jaccl_worker = state.jaccl_worker.clone();
    tokio::spawn(async move {
        transfer_pipeline(tx, jaccl_worker, req).await;
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

    // Spawn a task to handle accept (uses JACCL worker channel internally).
    tokio::spawn(async move {
        accept_worker(
            &coordinator_ip,
            coordinator_port,
            &model_path,
            &model_dir,
            jaccl_worker,
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
async fn transfer_pipeline(
    tx: tokio::sync::mpsc::Sender<String>,
    jaccl_worker: std::sync::Arc<JacclWorker>,
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

    // ── 4. Resolve our own LAN IP (for coordinator) ─────────────────────
    let local_ip = match resolve_local_lan_ip() {
        Some(ip) => ip,
        None => {
            let _ = tx.send(SseEvent::error("cannot determine local LAN IP").to_sse_line()).await;
            return;
        }
    };

    // ── 5. Init or reuse group via worker (port pinning) ────────────────
    let _ = tx.send(SseEvent::stage("init").to_sse_line()).await;

    let (init_reply_tx, init_reply_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = jaccl_worker.send(JacclCmd::GetOrInitGroup {
        peer: req.peer.clone(),
        rank: 0, // coordinator
        coordinator_ip: local_ip.clone(),
        coordinator_port: None, // worker picks or reuses
        reply: init_reply_tx,
    }) {
        let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
        return;
    }

    let coordinator_port = match init_reply_rx.await {
        Ok(Ok(port)) => port,
        Ok(Err(e)) => {
            let _ = tx.send(SseEvent::error(&e).to_sse_line()).await;
            return;
        }
        Err(_) => {
            let _ = tx.send(SseEvent::error("jaccl worker dropped reply channel").to_sse_line()).await;
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

    // Resolve peer IP via native mDNS (Phase 4)
    let peer_ip = match resolve_peer_mdns(&req.peer) {
        Some(ip) => ip,
        None => {
            let _ = tx.send(SseEvent::error(&format!("cannot resolve peer '{}'", req.peer)).to_sse_line()).await;
            return;
        }
    };
    let peer_url = format!("http://{}:9090/transfer/accept", peer_ip);
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
) {
    // Init group via worker channel
    let peer_key = format!("accept:{model_dir}");

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = jaccl_worker.send(JacclCmd::GetOrInitGroup {
        peer: peer_key.clone(),
        rank: 1,
        coordinator_ip: coordinator_ip.to_string(),
        coordinator_port: Some(coordinator_port),
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

/// Send all files in `model_path` to rank 1 via worker channel.
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

    let total_bytes: u64 = entries.iter().map(|e| e.size).sum();
    let manifest = FileManifest { files: entries };
    let manifest_json = serde_json::to_vec(&manifest)
        .map_err(|e| format!("serialize manifest: {e}"))?;

    // Send manifest length (8 bytes) + manifest (padded to 1 MB)
    let manifest_len = manifest_json.len();
    let length_buf = (manifest_len as u64).to_le_bytes();
    worker_send(worker, peer, &length_buf, 1, HEADER_TIMEOUT_MS).await
        .map_err(|e| format!("send manifest length: {e}"))?;

    let mut header_buf = vec![0u8; 1024 * 1024];
    let copy_len = manifest_len.min(header_buf.len());
    header_buf[..copy_len].copy_from_slice(&manifest_json[..copy_len]);
    worker_send(worker, peer, &header_buf, 1, HEADER_TIMEOUT_MS).await
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
            worker_send(worker, peer, &size_buf, 1, CHUNK_TIMEOUT_MS).await
                .map_err(|e| format!("send chunk size for {}: {e}", file_entry.name))?;
            worker_send(worker, peer, &file_data[offset..end], 1, CHUNK_TIMEOUT_MS).await
                .map_err(|e| format!("send chunk for {}: {e}", file_entry.name))?;

            offset = end;
            bytes_sent += chunk_len as u64;

            let pct = ((bytes_sent as f64 / total_bytes as f64) * 100.0) as u8;
            let _ = tx.send(SseEvent::progress(pct.min(99), 0).to_sse_line()).await;
        }
    }

    Ok(bytes_sent)
}

/// Receive files from rank 0 into `model_path` via worker channel.
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

    // Receive manifest length (8 bytes)
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

    for file_entry in &manifest.files {
        let file_path = model_path.join(&file_entry.name);
        let mut file_data = Vec::with_capacity(file_entry.size as usize);

        let mut remaining = file_entry.size as usize;
        while remaining > 0 {
            // Receive chunk size (8 bytes)
            let size_data = worker_recv(worker, peer, 8, 0, CHUNK_TIMEOUT_MS).await
                .map_err(|e| format!("recv chunk size for {}: {e}", file_entry.name))?;
            let chunk_len = u64::from_le_bytes(size_data[..8].try_into().unwrap()) as usize;

            // Receive chunk data
            let chunk_data = worker_recv(worker, peer, chunk_len, 0, CHUNK_TIMEOUT_MS).await
                .map_err(|e| format!("recv chunk for {}: {e}", file_entry.name))?;

            file_data.extend_from_slice(&chunk_data);
            remaining = remaining.saturating_sub(chunk_len);
            bytes_received += chunk_len as u64;

            let pct = ((bytes_received as f64 / total_bytes as f64) * 100.0) as u8;
            let _ = tx.send(SseEvent::progress(pct.min(99), 0).to_sse_line()).await;
        }

        // Write file
        std::fs::write(&file_path, &file_data)
            .map_err(|e| format!("write {}: {e}", file_entry.name))?;
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

/// Resolve a peer hostname to an IPv4 address via native mDNS (Phase 4).
/// Uses libc::getaddrinfo — no Python subprocess.
#[cfg(feature = "jaccl")]
fn resolve_peer_mdns(peer: &str) -> Option<String> {
    use std::ffi::CString;
    use std::net::Ipv4Addr;

    let hostname = CString::new(format!("{}.local", peer)).ok()?;
    let mut res: *mut libc::addrinfo = std::ptr::null_mut();
    let hints = libc::addrinfo {
        ai_family: libc::AF_INET,
        ai_socktype: libc::SOCK_STREAM,
        ai_flags: 0,
        ai_protocol: 0,
        ai_addrlen: 0,
        ai_canonname: std::ptr::null_mut(),
        ai_addr: std::ptr::null_mut(),
        ai_next: std::ptr::null_mut(),
    };
    let rc = unsafe {
        libc::getaddrinfo(hostname.as_ptr(), std::ptr::null(), &hints, &mut res)
    };
    if rc != 0 || res.is_null() {
        return None;
    }
    let addr = unsafe {
        let sa = (*res).ai_addr as *const libc::sockaddr_in;
        Ipv4Addr::from(u32::from_be((*sa).sin_addr.s_addr))
    };
    unsafe { libc::freeaddrinfo(res); }
    Some(addr.to_string())
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
