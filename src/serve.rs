//! MLX server lifecycle manager — Rust port of mlx_daemon.py.
//!
//! Manages per-port MLX server subprocesses. Each port has its own
//! `ProcessManager<HttpHealth>` (aliased as `ServeManager`) with independent
//! state file for crash recovery.
//!
//! The share session is managed by `ProcessManager<LogMonitor>` (aliased as
//! `ShareManager`).

use asmi_core::{LoadRequest, ServeBackend, ServeEngine, ServeState, ServeStatus, ShareRequest, ShareStatus};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;

use crate::daemon::resolve_python;

// ===========================================================================
// Constants and helpers
// ===========================================================================

/// Default managed ports (overridable via ASMI_MLX_LM_PORT / ASMI_MLX_VLM_PORT).
///
/// ds4 is intentionally NOT auto-managed: it is a native engine that cannot
/// start "bare" (no model) like the MLX servers, so a managed ds4 port lands
/// in `error` on every boot (`cannot open model 'ds4flash.gguf'`). Load ds4 on
/// demand via /serve/load instead of keeping a perpetually-erroring slot.
pub fn managed_ports() -> Vec<(u16, ServeEngine)> {
    let mut ports = vec![
        (port_for_engine(ServeEngine::MlxLm), ServeEngine::MlxLm),
        (port_for_engine(ServeEngine::MlxVlm), ServeEngine::MlxVlm),
    ];
    // Media gen slots are opt-in per node (config `media_autostart` or env
    // ASMI_MEDIA_AUTOSTART="image_gen,video_gen") — same rationale as ds4:
    // auto-managing them on nodes without mflux/mlx-video breeds error slots.
    // The /health endpoint works without the toolchain, so a started slot is
    // harmless; the opt-in is about not starting stray processes fleet-wide.
    let mut autostart = asmi_core::NodeMap::load().media_autostart;
    if let Ok(env_list) = std::env::var("ASMI_MEDIA_AUTOSTART") {
        autostart.extend(env_list.split(',').map(|s| s.trim().to_string()));
    }
    for name in autostart {
        let engine = match name.as_str() {
            "image_gen" => ServeEngine::ImageGen,
            "video_gen" => ServeEngine::VideoGen,
            other => {
                tracing::warn!(engine = other, "unknown media_autostart entry — skipping");
                continue;
            }
        };
        let port = port_for_engine(engine);
        if !ports.iter().any(|(p, _)| *p == port) {
            ports.push((port, engine));
        }
    }
    ports
}

/// Resolve port for an engine: env var > default.
pub fn port_for_engine(engine: ServeEngine) -> u16 {
    match engine {
        ServeEngine::MlxLm | ServeEngine::MlxLmShare => std::env::var("ASMI_MLX_LM_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(19080),
        ServeEngine::MlxVlm => std::env::var("ASMI_MLX_VLM_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(19084),
        ServeEngine::VllmMlx => std::env::var("ASMI_VLLM_MLX_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(8000),
        ServeEngine::DFlash => std::env::var("ASMI_DFLASH_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(19080),
        ServeEngine::Ds4 => std::env::var("ASMI_DS4_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(8080),
        ServeEngine::ImageGen => std::env::var("ASMI_IMAGE_GEN_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(19095),
        ServeEngine::VideoGen => std::env::var("ASMI_VIDEO_GEN_PORT")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(19096),
    }
}

/// Share session log file.
const SHARE_LOG_PATH: &str = "/tmp/r1o-mlx-share.log";
/// Default port for the distributed inference server.
const SHARE_PORT: u16 = 19080;

/// Resolve the `mlx.launch` CLI script path.
/// Checks known locations first (launchd doesn't have Homebrew in PATH).
fn resolve_mlx_launch() -> String {
    // Check known Homebrew locations first (launchd has no PATH)
    for path in &[
        "/opt/homebrew/bin/mlx.launch",
        "/usr/local/bin/mlx.launch",
    ] {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }
    // Try which (works in interactive shells)
    if let Ok(output) = std::process::Command::new("which").arg("mlx.launch").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() && std::path::Path::new(&path).exists() {
                return path;
            }
        }
    }
    // Should not reach here — mlx.launch is installed via pip
    tracing::warn!("mlx.launch not found! Distributed inference will fail.");
    "mlx.launch".to_string()
}

/// Resolve a native (non-Python) server binary.
///
/// Search order:
///   1. `DS4_SERVER_PATH` env var (explicit override)
///   2. `~/.r1o/bin/<binary>` (managed install location)
///   3. `which <binary>` (PATH lookup)
///   4. Common fallback paths (`/usr/local/bin/`, `~/opensource/ds4/`)
fn resolve_native_binary(binary: &str, engine: &ServeEngine) -> Result<PathBuf, anyhow::Error> {
    // 1. Env var override (e.g. DS4_SERVER_PATH)
    let env_key = format!("{}_PATH", binary.to_uppercase().replace('-', "_"));
    if let Ok(p) = std::env::var(&env_key) {
        let path = PathBuf::from(&p);
        if path.exists() {
            tracing::info!(%env_key, ?path, "native binary from env");
            return Ok(path);
        }
        tracing::warn!(%env_key, path = %p, "env var set but path does not exist");
    }

    // 2. Managed install location
    let managed = r1o_dir().join("bin").join(binary);
    if managed.exists() {
        tracing::info!(?managed, "native binary from ~/.r1o/bin");
        return Ok(managed);
    }

    // 3. PATH lookup via `which`
    if let Ok(output) = std::process::Command::new("which").arg(binary).output() {
        if output.status.success() {
            let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !p.is_empty() && std::path::Path::new(&p).exists() {
                tracing::info!(path = %p, "native binary from PATH");
                return Ok(PathBuf::from(p));
            }
        }
    }

    // 4. Common fallback paths
    let fallbacks: Vec<PathBuf> = vec![
        PathBuf::from(format!("/usr/local/bin/{binary}")),
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(format!("opensource/ds4/{binary}")),
    ];
    for fb in &fallbacks {
        if fb.exists() {
            tracing::info!(?fb, "native binary from fallback path");
            return Ok(fb.clone());
        }
    }

    anyhow::bail!(
        "native binary '{}' not found for engine {:?}. Set {} or place it in ~/.r1o/bin/",
        binary, engine, env_key
    )
}

/// Resolve a media gen server script (image-gen-server.py / video-gen-server.py).
///
/// Search order:
///   1. `ASMI_IMAGE_GEN_SCRIPT` / `ASMI_VIDEO_GEN_SCRIPT` env var
///   2. `~/.r1o/bin/<script>` (managed install location, populated by deploy)
///   3. `~/<script>` (legacy hand-deployed location on hub)
fn resolve_media_script(engine: ServeEngine) -> Result<PathBuf, anyhow::Error> {
    let script = engine.config().binary; // e.g. "image-gen-server.py"
    let env_key = match engine {
        ServeEngine::ImageGen => "ASMI_IMAGE_GEN_SCRIPT",
        ServeEngine::VideoGen => "ASMI_VIDEO_GEN_SCRIPT",
        other => anyhow::bail!("resolve_media_script called for non-media engine {other}"),
    };
    if let Ok(p) = std::env::var(env_key) {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Ok(path);
        }
        tracing::warn!(%env_key, path = %p, "env var set but path does not exist");
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    for cand in [r1o_dir().join("bin").join(script), home.join(script)] {
        if cand.exists() {
            return Ok(cand);
        }
    }
    anyhow::bail!(
        "media server script '{script}' not found. Set {env_key}, or copy it from \
         the apple-smi repo's media/ dir to ~/.r1o/bin/"
    )
}

/// Warmup timeout for bare server start (no model — should be fast).
const WARMUP_TIMEOUT_BARE_SECS: u64 = 60;
/// Warmup timeout for model loading (large models can take 5+ minutes on M3 Ultra).
const WARMUP_TIMEOUT_MODEL_SECS: u64 = 300;
/// Warmup timeout for distributed share session start.
const WARMUP_TIMEOUT_SHARE_SECS: u64 = 300;

/// r1o config directory (~/.r1o/).
fn r1o_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".r1o")
}

/// Persistent state file for crash recovery (per-port).
fn state_file(port: u16) -> PathBuf {
    r1o_dir().join(format!("serve-state-{port}.json"))
}

/// Legacy state file (pre-multi-port).
fn legacy_state_file() -> PathBuf {
    r1o_dir().join("serve-state.json")
}

/// Persistent state file for share crash recovery.
fn share_state_file() -> PathBuf {
    r1o_dir().join("share-state.json")
}

/// Default JACCL hostfile location.
pub fn default_hostfile() -> PathBuf {
    r1o_dir().join("hostfiles/default.json")
}

/// Resolve a backend string to a ServeBackend. "auto" upgrades to jaccl when
/// a hostfile exists; explicit distributed backends ("jaccl", "jaccl-ring",
/// "ring") also require their hostfile to exist, else fall back to single.
pub fn resolve_backend(backend: &str, hostfile: Option<&str>) -> ServeBackend {
    if backend == "single" {
        return ServeBackend::Single;
    }
    let hf = hostfile
        .map(PathBuf::from)
        .unwrap_or_else(default_hostfile);
    if !hf.exists() {
        if backend != "auto" {
            tracing::warn!(backend, hostfile = %hf.display(), "distributed backend requested but hostfile missing — falling back to single");
        }
        return ServeBackend::Single;
    }
    match backend {
        "jaccl" | "auto" => ServeBackend::Jaccl,
        "jaccl-ring" => ServeBackend::JacclRing,
        "ring" => ServeBackend::Ring,
        other => {
            tracing::warn!(backend = other, "unknown backend string — falling back to single");
            ServeBackend::Single
        }
    }
}

/// Env-key prefixes asmi will forward into serve processes (and all
/// distributed ranks). Prefix-gated because /serve/load is tailnet-reachable:
/// PATH / DYLD_* / PYTHONPATH must never be injectable into spawned
/// interpreters. Values are rejected on control chars; `mlx.launch` shlex-
/// quotes them again on the remote side.
const ENV_FORWARD_PREFIXES: &[&str] = &["MLX_", "KV_", "HF_", "JACCL_", "NCCL_"];

fn allowlisted_env(env: Option<&std::collections::HashMap<String, String>>) -> Vec<(String, String)> {
    let Some(env) = env else { return Vec::new() };
    let mut out: Vec<(String, String)> = Vec::new();
    for (k, v) in env {
        let key_ok = ENV_FORWARD_PREFIXES.iter().any(|p| k.starts_with(p))
            && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        let val_ok = !v.chars().any(|c| c.is_control());
        if key_ok && val_ok {
            out.push((k.clone(), v.clone()));
        } else {
            tracing::warn!(key = %k, "env var dropped (not allowlisted or invalid value)");
        }
    }
    out.sort(); // deterministic spawn args
    out
}

/// Probe every host in a JACCL hostfile via its asmi `/health` endpoint,
/// concurrently (serial 2 s timeouts would add hosts x 2 s of load latency).
/// `Err` carries the unreachable host names (or a parse-level reason).
///
/// Why: a stale default hostfile must not silently turn a plain single-model
/// serve into a doomed multi-rank launch. Observed 2026-06-10: both hub's and
/// m3u3's `hostfiles/default.json` were 4-node JACCL configs left over from
/// experiments — one still listing a node that had been SOLD — so every
/// `backend: "auto"` load failed with rank exits 255/-15.
pub async fn hostfile_hosts_alive(hf: &std::path::Path) -> Result<(), Vec<String>> {
    let data = std::fs::read_to_string(hf)
        .map_err(|e| vec![format!("unreadable hostfile: {e}")])?;
    let json: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| vec![format!("invalid hostfile json: {e}")])?;
    let hosts = json
        .get("hosts")
        .and_then(|h| h.as_array())
        .cloned()
        .unwrap_or_default();
    if hosts.is_empty() {
        return Err(vec!["hostfile has no hosts".to_string()]);
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(e) => return Err(vec![format!("probe client: {e}")]),
    };
    let probes = hosts.iter().map(|h| {
        let ssh = h
            .get("ssh")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown")
            .to_string();
        let client = client.clone();
        async move {
            let url = format!("http://{ssh}:9090/health");
            let alive = matches!(client.get(&url).send().await, Ok(r) if r.status().is_success());
            (ssh, alive)
        }
    });
    let results = futures::future::join_all(probes).await;
    let dead: Vec<String> = results
        .into_iter()
        .filter_map(|(ssh, alive)| (!alive).then_some(ssh))
        .collect();
    if dead.is_empty() { Ok(()) } else { Err(dead) }
}

/// `resolve_backend`, plus a liveness gate on the implicit path: when "auto"
/// resolves to jaccl purely because a hostfile EXISTS, every host in it must
/// answer its asmi `/health` probe — otherwise fall back to single with a
/// warning. An EXPLICIT `backend: "jaccl"` request is honored unvalidated
/// (the user asked for it; the launch error will name the dead rank).
pub async fn resolve_backend_validated(backend: &str, hostfile: Option<&str>) -> ServeBackend {
    let resolved = resolve_backend(backend, hostfile);
    if resolved != ServeBackend::Jaccl || backend != "auto" {
        return resolved;
    }
    let hf = hostfile
        .map(PathBuf::from)
        .unwrap_or_else(default_hostfile);
    match hostfile_hosts_alive(&hf).await {
        Ok(()) => ServeBackend::Jaccl,
        Err(dead) => {
            tracing::warn!(
                hostfile = %hf.display(),
                dead = ?dead,
                "auto backend: hostfile hosts unreachable — falling back to single"
            );
            ServeBackend::Single
        }
    }
}

/// Read the last N lines from a log file (best-effort).
async fn read_log_tail(path: &str, lines: usize) -> String {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => {
            let tail: Vec<&str> = content.lines().rev().take(lines).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            // Find the most useful line: last Python exception or traceback line
            let useful = tail.iter().find(|l| {
                l.contains("Error:") || l.contains("Exception:") || l.contains("error:")
            });
            if let Some(line) = useful {
                line.trim().to_string()
            } else {
                tail.join("\n").trim().to_string()
            }
        }
        Err(_) => String::new(),
    }
}

/// Verify a process owns the expected port via lsof.
async fn verify_port_owner(pid: u32, port: u16) -> bool {
    let output = Command::new("/usr/sbin/lsof")
        .args([
            "-a",
            "-p",
            &pid.to_string(),
            "-iTCP",
            "-sTCP:LISTEN",
            "-P",
            "-n",
        ])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout.contains(&format!(":{port}"))
        }
        _ => false,
    }
}

// ===========================================================================
// ReadinessCheck trait + implementations
// ===========================================================================

/// Trait for polling a child process until it signals readiness.
/// Returns `Ok(true)` = ready, `Ok(false)` = timeout, `Err(msg)` = crash/error.
pub trait ReadinessCheck: Send + Sync + 'static {
    fn poll_ready(
        &self,
        child: &mut tokio::process::Child,
        timeout_secs: u64,
    ) -> impl std::future::Future<Output = Result<bool, String>> + Send;
}

/// HTTP health-check readiness (for serve managers).
#[derive(Clone)]
pub struct HttpHealth {
    port: u16,
    endpoints: Vec<&'static str>,
}

impl ReadinessCheck for HttpHealth {
    async fn poll_ready(
        &self,
        child: &mut tokio::process::Child,
        timeout_secs: u64,
    ) -> Result<bool, String> {
        let log_path = format!("/tmp/r1o-mlx-server-{}.log", self.port);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;

        let port = self.port;
        let endpoints: Vec<&str> = self.endpoints.clone();

        tokio::select! {
            exit_result = child.wait() => {
                let detail = read_log_tail(&log_path, 15).await;
                let code_str = match exit_result {
                    Ok(status) => status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                    Err(e) => format!("wait error: {e}"),
                };
                Err(format!("server exited during startup (exit {code_str}): {detail}"))
            }
            result = poll_health(&client, port, &endpoints, timeout_secs) => {
                result
            }
        }
    }
}

/// Poll health endpoints until one returns 200 or timeout.
/// Returns Ok(true) on success, Ok(false) on timeout.
async fn poll_health(
    client: &reqwest::Client,
    port: u16,
    endpoints: &[&str],
    timeout_secs: u64,
) -> Result<bool, String> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        for ep in endpoints {
            let url = format!("http://127.0.0.1:{port}{ep}");
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(true);
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

/// Log-file readiness monitor (for share manager).
#[derive(Clone)]
pub struct LogMonitor {
    log_path: String,
    ready_markers: Vec<&'static str>,
    error_markers: Vec<&'static str>,
}

impl ReadinessCheck for LogMonitor {
    async fn poll_ready(
        &self,
        child: &mut tokio::process::Child,
        timeout_secs: u64,
    ) -> Result<bool, String> {
        let log_path = self.log_path.clone();
        let ready_markers = self.ready_markers.clone();
        let error_markers = self.error_markers.clone();

        tokio::select! {
            exit_result = child.wait() => {
                let detail = read_log_tail(&log_path, 15).await;
                let code_str = match exit_result {
                    Ok(status) => status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                    Err(e) => format!("wait error: {e}"),
                };
                Err(format!("share exited during startup (exit {code_str}): {detail}"))
            }
            result = poll_log(&log_path, &ready_markers, &error_markers, timeout_secs) => {
                result
            }
        }
    }
}

/// Poll a log file for readiness/error markers + HTTP health check on share port.
/// Returns Ok(true) when ready, Ok(false) on timeout, Err on error markers.
async fn poll_log(
    log_path: &str,
    ready_markers: &[&str],
    error_markers: &[&str],
    timeout_secs: u64,
) -> Result<bool, String> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let health_url = format!("http://localhost:{SHARE_PORT}/v1/models");
    loop {
        // Check log markers
        if let Ok(content) = tokio::fs::read_to_string(log_path).await {
            if ready_markers.iter().any(|m| content.contains(m)) {
                return Ok(true);
            }
            if error_markers.iter().any(|m| content.contains(m)) {
                let detail = read_log_tail(log_path, 10).await;
                return Err(format!("share error: {detail}"));
            }
        }
        // Also try HTTP health check (server may be ready before log flushes)
        if let Ok(resp) = reqwest::Client::new()
            .get(&health_url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            if resp.status().is_success() {
                tracing::info!("share server ready via HTTP health check on port {SHARE_PORT}");
                return Ok(true);
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

// ===========================================================================
// ManagedProcess — unified inner state
// ===========================================================================

/// Internal mutable state behind the RwLock.
struct ManagedProcess {
    state: ServeState,
    model: Option<String>,
    engine: ServeEngine,
    backend: ServeBackend,
    port: Option<u16>,
    child: Option<tokio::process::Child>,
    pid: Option<u32>,
    load_started: Option<std::time::Instant>,
    error: Option<String>,
    stopped_at: Option<std::time::Instant>,
    /// Cached result of verify_port_owner — refreshed on a 15s timer (Phase C).
    port_verified_cached: bool,
    /// When port_verified_cached was last updated.
    port_verified_at: Option<std::time::Instant>,
}

/// Kill the existing child process (SIGTERM → 5s → SIGKILL).
async fn kill_child(s: &mut ManagedProcess) {
    if let Some(ref mut child) = s.child {
        // Managed child — SIGTERM then SIGKILL with bounded waits
        if let Some(pid) = s.pid {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );
        }
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                // SIGKILL + bounded wait — child.kill().await can hang on zombies
                let _ = child.start_kill();
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    child.wait(),
                ).await;
            }
        }
    } else if let Some(pid) = s.pid {
        // Adopted external process — no child handle, kill by PID
        tracing::info!(pid, "killing adopted process by PID");
        let nix_pid = nix::unistd::Pid::from_raw(pid as i32);
        let _ = nix::sys::signal::kill(nix_pid, nix::sys::signal::Signal::SIGTERM);
        // Wait up to 5s for graceful exit, then SIGKILL
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if nix::sys::signal::kill(nix_pid, None).is_err() {
                break; // process gone
            }
        }
        if nix::sys::signal::kill(nix_pid, None).is_ok() {
            tracing::warn!(pid, "SIGTERM didn't work, sending SIGKILL");
            let _ = nix::sys::signal::kill(nix_pid, nix::sys::signal::Signal::SIGKILL);
        }
    }
    s.child = None;
    s.pid = None;
}

/// Persist model/engine/backend to disk for crash recovery.
/// Uses `port` to select the file path: Some(port) → serve state, None → share state.
async fn persist_state(s: &ManagedProcess) {
    let sf = match s.port {
        Some(port) => state_file(port),
        None => share_state_file(),
    };
    if let Some(parent) = sf.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    let mut data = serde_json::json!({
        "model": s.model,
        "backend": s.backend.to_string(),
    });
    // Serve state also persists engine
    if s.port.is_some() {
        data.as_object_mut().unwrap().insert("engine".to_string(), serde_json::to_value(s.engine).unwrap());
    }
    let _ = tokio::fs::write(&sf, serde_json::to_string_pretty(&data).unwrap_or_default()).await;
}

// ===========================================================================
// ProcessManager<R> — generic manager
// ===========================================================================

/// Thread-safe process manager. Clone-friendly (wraps Arc).
/// Generic over the readiness-check strategy.
#[derive(Clone)]
pub struct ProcessManager<R: ReadinessCheck> {
    inner: Arc<RwLock<ManagedProcess>>,
    readiness: Arc<R>,
}

impl<R: ReadinessCheck> ProcessManager<R> {
    /// Stop the running process and return to idle.
    pub async fn stop(&self) {
        let mut s = self.inner.write().await;
        kill_child(&mut s).await;
        s.state = ServeState::Idle;
        s.model = None;
        s.error = None;
        s.stopped_at = Some(std::time::Instant::now());
        persist_state(&s).await;
    }

    /// Emergency stop: SIGKILL immediately, no SIGTERM grace period.
    /// Used when RDMA peer death is detected to prevent GPU Lock.
    pub async fn emergency_stop(&self) {
        let mut s = self.inner.write().await;
        let pid = s.pid;
        if let Some(ref mut child) = s.child {
            tracing::warn!(pid = pid, "EMERGENCY STOP: sending SIGKILL to prevent GPU Lock");
            let _ = child.kill().await;
        }
        s.child = None;
        s.pid = None;
        s.state = ServeState::Error;
        s.model = None;
        s.error = Some("emergency stop: RDMA peer death detected".to_string());
        persist_state(&s).await;
    }
}

// ===========================================================================
// ServeManager = ProcessManager<HttpHealth>
// ===========================================================================

/// Backward-compatible type alias.
pub type ServeManager = ProcessManager<HttpHealth>;

impl ServeManager {
    /// Create a new idle manager.
    pub fn new(port: u16, engine: ServeEngine) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ManagedProcess {
                state: ServeState::Idle,
                model: None,
                engine,
                backend: ServeBackend::default(),
                port: Some(port),
                child: None,
                pid: None,
                load_started: None,
                error: None,
                stopped_at: None,
                port_verified_cached: false,
                port_verified_at: None,
            })),
            readiness: Arc::new(HttpHealth {
                port,
                endpoints: engine.config().health_endpoints.to_vec(),
            }),
        }
    }

    /// Create a manager and restore from persisted state.
    /// If saved state has a model → reload it.
    /// Otherwise → auto-start bare (process running, no model).
    pub async fn restore(port: u16, default_engine: ServeEngine) -> Self {
        // Migrate legacy state file for port 19080
        if port == 19080 {
            let legacy = legacy_state_file();
            let new_path = state_file(port);
            if legacy.exists() && !new_path.exists() {
                tracing::info!("migrating legacy serve-state.json → serve-state-{port}.json");
                let _ = tokio::fs::rename(&legacy, &new_path).await;
            }
        }

        let mgr = Self::new(port, default_engine);
        let sf = state_file(port);
        if sf.exists() {
            if let Ok(data) = tokio::fs::read_to_string(&sf).await {
                if let Ok(saved) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(model) = saved.get("model").and_then(|v| v.as_str()) {
                        if !model.is_empty() {
                            let backend = saved
                                .get("backend")
                                .and_then(|v| v.as_str())
                                .unwrap_or("auto")
                                .to_string();
                            let engine: ServeEngine = saved
                                .get("engine")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or(default_engine);
                            let req = LoadRequest {
                                model_path: Some(model.to_string()),
                                backend,
                                hostfile: None,
                                engine,
                                ..Default::default()
                            };
                            tracing::info!(model, %engine, port, "restoring last served model");
                            mgr.load(req).await;
                            return mgr;
                        }
                    }
                }
            }
        }

        // No saved model — auto-start bare
        tracing::info!(port, %default_engine, "no saved model, starting bare server");
        mgr.start_bare(default_engine).await;
        mgr
    }

    /// Start a bare server (process running, no model loaded).
    /// Used at boot to make ports immediately available.
    pub async fn start_bare(&self, engine: ServeEngine) {
        {
            let mut s = self.inner.write().await;
            s.state = ServeState::Loading;
            s.engine = engine;
            s.error = None;
            s.load_started = Some(std::time::Instant::now());
        }
        // Update readiness endpoints for the new engine
        let readiness = Arc::new(HttpHealth {
            port: {
                let s = self.inner.read().await;
                s.port.unwrap_or(19080)
            },
            endpoints: engine.config().health_endpoints.to_vec(),
        });
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let req = LoadRequest {
                model_path: None,
                backend: "single".to_string(),
                hostfile: None,
                engine,
                ..Default::default()
            };
            do_serve_load(inner, readiness, req).await;
        });
    }

    /// Begin loading a model. Spawns a background task and returns immediately.
    pub async fn load(&self, req: LoadRequest) {
        {
            let mut s = self.inner.write().await;
            s.state = ServeState::Loading;
            s.error = None;
            s.load_started = Some(std::time::Instant::now());
        }
        // Update readiness endpoints for the requested engine
        let readiness = Arc::new(HttpHealth {
            port: {
                let s = self.inner.read().await;
                s.port.unwrap_or(19080)
            },
            endpoints: req.engine.config().health_endpoints.to_vec(),
        });
        let inner = self.inner.clone();
        tokio::spawn(async move {
            do_serve_load(inner, readiness, req).await;
        });
    }

    /// Lightweight model info — just reads model + state from the lock.
    /// No subprocess calls (unlike `status()` which runs `verify_port_owner`).
    pub async fn model_snapshot(&self) -> (ServeState, Option<String>) {
        let s = self.inner.read().await;
        (s.state, s.model.clone())
    }

    /// Cheap snapshot for SSE broadcast — no subprocess forks.
    pub async fn slot_snapshot(&self) -> asmi_core::ServeSlotSnapshot {
        let s = self.inner.read().await;
        asmi_core::ServeSlotSnapshot {
            port: s.port.unwrap_or(19080),
            state: s.state,
            model: s.model.clone(),
            engine: s.engine,
            backend: s.backend,
            error: s.error.clone(),
            pid: s.pid,
            elapsed_ms: s.load_started.map(|t| t.elapsed().as_millis() as u64).unwrap_or(0),
            port_verified: s.port_verified_cached,
        }
    }

    /// Whether port_verified cache is stale and needs a refresh.
    pub async fn needs_port_verify_refresh(&self, max_age: std::time::Duration) -> bool {
        let s = self.inner.read().await;
        if !matches!(s.state, ServeState::Ready | ServeState::Bare) { return false; }
        if s.pid.is_none() { return false; }
        match s.port_verified_at {
            Some(at) => at.elapsed() > max_age,
            None => true,
        }
    }

    /// Run verify_port_owner and cache the result. Acquires write lock.
    pub async fn refresh_port_verified(&self) {
        let (pid, port, should_run) = {
            let s = self.inner.read().await;
            let should = matches!(s.state, ServeState::Ready | ServeState::Bare) && s.pid.is_some();
            (s.pid, s.port, should)
        };
        if !should_run { return; }
        let verified = verify_port_owner(pid.unwrap(), port.unwrap_or(19080)).await;
        let mut s = self.inner.write().await;
        s.port_verified_cached = verified;
        s.port_verified_at = Some(std::time::Instant::now());
    }

    /// Get a read-only status snapshot.
    pub async fn status(&self) -> ServeStatus {
        let s = self.inner.read().await;
        let port = s.port.unwrap_or(19080);
        let elapsed = s
            .load_started
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
        let port_verified = if s.pid.is_some()
            && (s.state == ServeState::Ready || s.state == ServeState::Bare)
        {
            verify_port_owner(s.pid.unwrap(), port).await
        } else {
            false
        };
        let pid = s.pid;
        let state = s.state;
        let model = s.model.clone();
        let engine = s.engine;
        let backend = s.backend;
        let error = s.error.clone();
        // Release the read lock before the (potentially slow) launchctl probe.
        drop(s);
        let launchd = match pid {
            Some(p) => crate::launchd::describe_pid(p).await,
            None => None,
        };
        let port_squatter = if pid.is_none() && matches!(state, ServeState::Error | ServeState::Bare) {
            detect_port_squatter(port).await
        } else {
            None
        };
        ServeStatus {
            state,
            model,
            engine,
            backend,
            port,
            pid,
            port_verified,
            elapsed_ms: elapsed,
            error,
            launchd,
            port_squatter,
        }
    }

    /// Adopt an external process we don't own (detected by metrics scanner).
    /// We track PID + model but don't hold a Child handle — can't send signals.
    pub async fn adopt_external(&self, pid: u32, model: Option<String>, engine: ServeEngine) {
        let mut s = self.inner.write().await;
        // Don't overwrite a managed process that's already running
        if s.child.is_some() || s.state == ServeState::Loading {
            return;
        }
        // Don't re-adopt if we intentionally stopped within the last 10s
        if let Some(stopped) = s.stopped_at {
            if stopped.elapsed() < std::time::Duration::from_secs(10) {
                return;
            }
        }
        s.pid = Some(pid);
        s.engine = engine;
        s.model = model.clone();
        s.backend = ServeBackend::Single;
        s.state = if model.is_some() { ServeState::Ready } else { ServeState::Bare };
        s.load_started = Some(std::time::Instant::now());
        s.stopped_at = None;
        s.error = None;
        tracing::info!(pid, model = model.as_deref().unwrap_or("none"), "adopted external process");
    }

    /// Check if a port-conflict error has resolved (port is now free).
    /// Called from the daemon poll loop to auto-recover Error state managers.
    pub async fn check_port_recovery(&self) {
        let port = {
            let s = self.inner.read().await;
            if s.state != ServeState::Error {
                return;
            }
            let is_port_conflict = s.error.as_deref()
                .map_or(false, |e| e.contains("already in use"));
            if !is_port_conflict {
                return;
            }
            match s.port {
                Some(p) => p,
                None => return,
            }
        };
        // TCP probe outside the lock — don't hold RwLock across await
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_err()
        {
            let mut s = self.inner.write().await;
            // Re-check state — may have changed while we probed
            if s.state == ServeState::Error {
                tracing::info!(port, "port conflict resolved — port is now free");
                s.state = ServeState::Idle;
                s.error = None;
            }
        }
    }

    /// Detect and adopt unmanaged model servers on managed ports.
    /// Called from the poll loop. Probes any occupied port where we don't own
    /// the child process — covers DFlash, manual mlx_lm, or any external launcher.
    pub async fn check_port_adoption(&self) {
        let (port, engine) = {
            let s = self.inner.read().await;
            if s.pid.is_some() || s.child.is_some() {
                return;
            }
            match s.port {
                Some(p) => (p, s.engine),
                None => return,
            }
        };
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_err() {
            return;
        }
        if let Some((pid, model)) = probe_model_server(port).await {
            self.adopt_external(pid, model, engine).await;
        }
    }
}

static PROBE_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .expect("probe client")
});

/// Probe an occupied port for an OpenAI-compatible model server.
/// Returns (pid, model_name) if it responds to /v1/models.
async fn probe_model_server(port: u16) -> Option<(u32, Option<String>)> {
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    if let Some(resp) = PROBE_CLIENT.get(&url).send().await.ok().filter(|r| r.status().is_success()) {
        let body: serde_json::Value = resp.json().await.ok()?;
        let model = body["data"][0]["id"].as_str().map(String::from);
        let pid = get_pid_on_port(port).await?;
        return Some((pid, model));
    }
    // Media gen dialect: launchd-owned image/video servers answer /health with
    // {"engine": "mflux"|"mlx-video", ...}. Recognize and adopt them so a
    // media serve slot on an occupied port doesn't land in error.
    let url = format!("http://127.0.0.1:{}/health", port);
    let resp = PROBE_CLIENT.get(&url).send().await.ok()?;
    if !resp.status().is_success() { return None; }
    let body: serde_json::Value = resp.json().await.ok()?;
    match body["engine"].as_str() {
        Some("mflux") | Some("mlx-video") => {
            let pid = get_pid_on_port(port).await?;
            Some((pid, None))
        }
        _ => None,
    }
}

async fn get_pid_on_port(port: u16) -> Option<u32> {
    // Multiple processes can legitimately listen on the same port number on
    // different addresses — e.g. `tailscale serve` publishes :19095 on the
    // tailnet IP while the real gen server sits on 127.0.0.1. The process we
    // would manage (and later SIGTERM!) binds loopback or wildcard; prefer
    // that, and only fall back to whatever listener we found first.
    let output = tokio::process::Command::new("lsof")
        .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN"])
        .output().await.ok()?;
    let s = String::from_utf8_lossy(&output.stdout);
    let mut fallback: Option<u32> = None;
    for line in s.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 9 {
            continue;
        }
        let Ok(pid) = cols[1].parse::<u32>() else { continue };
        let addr = cols[8];
        if addr.starts_with("127.0.0.1:")
            || addr.starts_with("*:")
            || addr.starts_with("[::]:")
            || addr.starts_with("[::1]:")
        {
            return Some(pid);
        }
        fallback.get_or_insert(pid);
    }
    fallback
}

async fn detect_port_squatter(port: u16) -> Option<asmi_core::PortSquatter> {
    let pid = get_pid_on_port(port).await?;
    let output = tokio::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output().await.ok()?;
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() { return None; }
    Some(asmi_core::PortSquatter { pid, process_name: name })
}

/// Background serve load task.
async fn do_serve_load(inner: Arc<RwLock<ManagedProcess>>, readiness: Arc<HttpHealth>, req: LoadRequest) {
    let result = do_serve_load_inner(&inner, &readiness, &req).await;
    if let Err(e) = result {
        let mut s = inner.write().await;
        s.state = ServeState::Error;
        s.error = Some(e.to_string());
    }
}

/// Build the `(program, args)` a serve request would spawn — PURE and
/// side-effect-free. Shared by the real spawn path (`do_serve_load_inner`) and
/// the synchronous `dry_run` preview in the HTTP handler, so the preview can
/// never drift from what actually runs. Appends `req.extra_args` last. Expands
/// `~` in the model path internally (idempotent), so callers may pass a raw or
/// already-expanded request.
pub(crate) fn build_serve_argv(
    req: &LoadRequest,
    port: u16,
    engine: ServeEngine,
    backend: ServeBackend,
) -> Result<(String, Vec<String>), anyhow::Error> {
    // Expand ~ in model path (no shell to do it for us). Idempotent.
    let mut req = req.clone();
    if let Some(ref mut path) = req.model_path {
        if path.starts_with("~/") {
            if let Some(home) = dirs::home_dir() {
                *path = format!("{}/{}", home.display(), &path[2..]);
            }
        }
    }
    let req = &req;

    let cfg = engine.config();
    let is_bare = req.model_path.is_none();
    let is_native = matches!(engine, ServeEngine::Ds4);

    let mut cmd_args: Vec<String> = Vec::new();
    let program: String;

    if engine.is_media() {
        // Media gen servers are plain scripts (stdlib-only) run under the
        // resolved python. They take no CLI flags — port/bind go via env on spawn.
        let script = resolve_media_script(engine)?;
        program = resolve_python().to_string();
        cmd_args.push(script.to_string_lossy().to_string());
    } else if is_native {
        let binary_path = resolve_native_binary(cfg.binary, &engine)?;
        program = binary_path.to_string_lossy().to_string();

        // Model flag + path
        if let (Some(flag), Some(model_path)) = (cfg.model_flag, &req.model_path) {
            cmd_args.push(flag.into());
            cmd_args.push(model_path.clone());
        }

        // Port binding (critical: without --port, ds4 defaults to 8000, not 8080)
        cmd_args.extend(["--port".into(), port.to_string()]);
        cmd_args.extend(["--host".into(), "0.0.0.0".into()]);

        // Backend: force Metal on Apple Silicon. The production GGUF quant
        // (DeepSeek-V4-Flash Q4-imatrix) has Q4_K routed experts that CRASH on
        // ds4's CPU path; Metal handles both Q4_K and IQ2_XXS. ds4's `--mtp`
        // speculative path is also Metal-only. asmi is Apple-Silicon-exclusive.
        cmd_args.push("--metal".into());

        // Context window size. Without -c, ds4-server defaults to 8192 (silent
        // truncation trap); callers pass the model's real max via ctx_size.
        if let Some(ctx) = req.ctx_size {
            cmd_args.extend(["-c".into(), ctx.to_string()]);
        }

        // Speculative decoding via the MTP draft head. ds4's `--mtp FILE` takes
        // a draft GGUF, distinct from mlx_lm's boolean `--mtp`.
        if let Some(ref draft) = req.draft_model {
            cmd_args.extend(["--mtp".into(), draft.clone()]);
            if let Some(n) = req.num_draft_tokens {
                cmd_args.extend(["--mtp-draft".into(), n.to_string()]);
            }
        }
    } else if let Some(uvicorn_app) = cfg.uvicorn_app {
        // Always invoke via resolve_python() since launchd doesn't have Homebrew in PATH.
        let py = resolve_python().to_string();
        program = py;
        cmd_args.extend([
            "-m".into(),
            "uvicorn".into(),
            uvicorn_app.into(),
            "--host".into(),
            "0.0.0.0".into(),
            "--port".into(),
            port.to_string(),
            "--workers".into(),
            "1".into(),
            "--no-access-log".into(),
        ]);
    } else {
        // Run as python3 -m <module> (e.g. python3 -m mlx_lm.server)
        let py = resolve_python().to_string();
        program = py;
        cmd_args.push("-m".into());
        cmd_args.push(cfg.binary.to_string());
        cmd_args.extend(cfg.binary_args.iter().map(|s| s.to_string()));
        // Only pass --model flag when we have a model to load
        if let (Some(flag), Some(model_path)) = (cfg.model_flag, &req.model_path) {
            cmd_args.push(flag.into());
            cmd_args.push(model_path.clone());
        }
        cmd_args.extend(["--port".into(), port.to_string(), "--host".into(), "0.0.0.0".into()]);

        // DFlash-specific flags (dflash_mlx.serve uses --draft for the drafter model)
        if matches!(engine, ServeEngine::DFlash) {
            if let Some(ref draft) = req.draft_model {
                cmd_args.extend(["--draft".into(), draft.clone()]);
            }
        }

        // Optimization passthrough (mlx_lm only — these flags are mlx_lm.server-specific)
        if matches!(engine, ServeEngine::MlxLm | ServeEngine::MlxLmShare) {
            if let Some(ref draft) = req.draft_model {
                cmd_args.extend(["--draft-model".into(), draft.clone()]);
            }
            if let Some(n) = req.num_draft_tokens {
                cmd_args.extend(["--num-draft-tokens".into(), n.to_string()]);
            }
            if let Some(n) = req.decode_concurrency {
                cmd_args.extend(["--decode-concurrency".into(), n.to_string()]);
            }
            if let Some(n) = req.prompt_concurrency {
                cmd_args.extend(["--prompt-concurrency".into(), n.to_string()]);
            }
            if let Some(n) = req.prefill_step_size {
                cmd_args.extend(["--prefill-step-size".into(), n.to_string()]);
            }
            if let Some(n) = req.prompt_cache_size {
                cmd_args.extend(["--prompt-cache-size".into(), n.to_string()]);
            }
            if let Some(n) = req.prompt_cache_bytes {
                cmd_args.extend(["--prompt-cache-bytes".into(), n.to_string()]);
            }
            if req.pipeline {
                cmd_args.push("--pipeline".into());
            }
            if req.use_mtp {
                cmd_args.push("--mtp".into());
            }
            if let Some(ref ct) = req.cache_type {
                cmd_args.extend(["--cache-type-k".into(), ct.clone(), "--cache-type-v".into(), ct.clone()]);
            }
            if let Some(n) = req.max_tokens {
                cmd_args.extend(["--max-tokens".into(), n.to_string()]);
            }
        }
    }

    // Generic passthrough: caller-supplied extra flags, appended LAST so they
    // override earlier ones. Pre-tokenized — asmi does not re-parse (no quoting
    // drift). For distributed runs these ride after `--`, i.e. to the inner
    // program, because the wrapper below wraps `program` + `cmd_args` wholesale.
    if let Some(ref extra) = req.extra_args {
        cmd_args.extend(extra.iter().cloned());
    }

    // Distributed wrapper (only for engines with model_flag and non-bare).
    // Covers jaccl, jaccl-ring, and ring — mlx.launch accepts all three.
    let (final_program, final_args) = if !is_bare
        && backend.is_distributed()
        && cfg.model_flag.is_some()
    {
        let hf = req
            .hostfile
            .clone()
            .unwrap_or_else(|| default_hostfile().to_string_lossy().to_string());
        let launcher = resolve_mlx_launch();
        let backend_str = backend.as_str();
        let mut jaccl_args = vec![
            "--hostfile".to_string(),
            hf,
            "--backend".to_string(),
            backend_str.to_string(),
            // MLX_DISTRIBUTED_BACKEND is read by the _mlx_backend_fix.pth rank
            // hook and passed to mx.distributed.init() — without it the
            // backend-selection race can hang rank 0.
            "--env".to_string(),
            format!("MLX_DISTRIBUTED_BACKEND={backend_str}"),
            "--env".to_string(),
            "MLX_METAL_FAST_SYNCH=1".to_string(),
        ];
        for (k, v) in allowlisted_env(req.env.as_ref()) {
            jaccl_args.push("--env".to_string());
            jaccl_args.push(format!("{k}={v}"));
        }
        jaccl_args.push("--".to_string());
        jaccl_args.push(program);
        jaccl_args.extend(cmd_args);
        (launcher, jaccl_args)
    } else {
        (program, cmd_args)
    };

    Ok((final_program, final_args))
}

#[cfg(test)]
mod parity_tests {
    //! Differential golden-parity harness for the data-driven-engines refactor
    //! (plan: docs/plans/2026-07-07-asmi-data-driven-engines.md, step 1).
    //!
    //! `build_serve_argv_legacy` below is a FROZEN verbatim copy of the current
    //! hardcoded per-engine logic. The refactor rewrites the production
    //! `build_serve_argv` to be data-driven; this test asserts it stays
    //! BYTE-IDENTICAL to the legacy oracle across every engine + code path.
    //! Same-process, same resolver fns on both sides → exact equality with zero
    //! cross-machine brittleness. DELETE legacy once the refactor lands + is green.
    use super::*;

    /// FROZEN 2026-07-07 — do NOT edit. The behavioral contract the refactor must preserve.
    #[allow(dead_code)]
    fn build_serve_argv_legacy(
        req: &LoadRequest,
        port: u16,
        engine: ServeEngine,
        backend: ServeBackend,
    ) -> Result<(String, Vec<String>), anyhow::Error> {
        let mut req = req.clone();
        if let Some(ref mut path) = req.model_path {
            if path.starts_with("~/") {
                if let Some(home) = dirs::home_dir() {
                    *path = format!("{}/{}", home.display(), &path[2..]);
                }
            }
        }
        let req = &req;

        let cfg = engine.config();
        let is_bare = req.model_path.is_none();
        let is_native = matches!(engine, ServeEngine::Ds4);

        let mut cmd_args: Vec<String> = Vec::new();
        let program: String;

        if engine.is_media() {
            let script = resolve_media_script(engine)?;
            program = resolve_python().to_string();
            cmd_args.push(script.to_string_lossy().to_string());
        } else if is_native {
            let binary_path = resolve_native_binary(cfg.binary, &engine)?;
            program = binary_path.to_string_lossy().to_string();
            if let (Some(flag), Some(model_path)) = (cfg.model_flag, &req.model_path) {
                cmd_args.push(flag.into());
                cmd_args.push(model_path.clone());
            }
            cmd_args.extend(["--port".into(), port.to_string()]);
            cmd_args.extend(["--host".into(), "0.0.0.0".into()]);
            cmd_args.push("--metal".into());
            if let Some(ctx) = req.ctx_size {
                cmd_args.extend(["-c".into(), ctx.to_string()]);
            }
            if let Some(ref draft) = req.draft_model {
                cmd_args.extend(["--mtp".into(), draft.clone()]);
                if let Some(n) = req.num_draft_tokens {
                    cmd_args.extend(["--mtp-draft".into(), n.to_string()]);
                }
            }
        } else if let Some(uvicorn_app) = cfg.uvicorn_app {
            let py = resolve_python().to_string();
            program = py;
            cmd_args.extend([
                "-m".into(), "uvicorn".into(), uvicorn_app.into(),
                "--host".into(), "0.0.0.0".into(), "--port".into(), port.to_string(),
                "--workers".into(), "1".into(), "--no-access-log".into(),
            ]);
        } else {
            let py = resolve_python().to_string();
            program = py;
            cmd_args.push("-m".into());
            cmd_args.push(cfg.binary.to_string());
            cmd_args.extend(cfg.binary_args.iter().map(|s| s.to_string()));
            if let (Some(flag), Some(model_path)) = (cfg.model_flag, &req.model_path) {
                cmd_args.push(flag.into());
                cmd_args.push(model_path.clone());
            }
            cmd_args.extend(["--port".into(), port.to_string(), "--host".into(), "0.0.0.0".into()]);
            if matches!(engine, ServeEngine::DFlash) {
                if let Some(ref draft) = req.draft_model {
                    cmd_args.extend(["--draft".into(), draft.clone()]);
                }
            }
            if matches!(engine, ServeEngine::MlxLm | ServeEngine::MlxLmShare) {
                if let Some(ref draft) = req.draft_model {
                    cmd_args.extend(["--draft-model".into(), draft.clone()]);
                }
                if let Some(n) = req.num_draft_tokens {
                    cmd_args.extend(["--num-draft-tokens".into(), n.to_string()]);
                }
                if let Some(n) = req.decode_concurrency {
                    cmd_args.extend(["--decode-concurrency".into(), n.to_string()]);
                }
                if let Some(n) = req.prompt_concurrency {
                    cmd_args.extend(["--prompt-concurrency".into(), n.to_string()]);
                }
                if let Some(n) = req.prefill_step_size {
                    cmd_args.extend(["--prefill-step-size".into(), n.to_string()]);
                }
                if let Some(n) = req.prompt_cache_size {
                    cmd_args.extend(["--prompt-cache-size".into(), n.to_string()]);
                }
                if let Some(n) = req.prompt_cache_bytes {
                    cmd_args.extend(["--prompt-cache-bytes".into(), n.to_string()]);
                }
                if req.pipeline {
                    cmd_args.push("--pipeline".into());
                }
                if req.use_mtp {
                    cmd_args.push("--mtp".into());
                }
                if let Some(ref ct) = req.cache_type {
                    cmd_args.extend(["--cache-type-k".into(), ct.clone(), "--cache-type-v".into(), ct.clone()]);
                }
                if let Some(n) = req.max_tokens {
                    cmd_args.extend(["--max-tokens".into(), n.to_string()]);
                }
            }
        }

        if let Some(ref extra) = req.extra_args {
            cmd_args.extend(extra.iter().cloned());
        }

        let (final_program, final_args) = if !is_bare
            && backend.is_distributed()
            && cfg.model_flag.is_some()
        {
            let hf = req.hostfile.clone()
                .unwrap_or_else(|| default_hostfile().to_string_lossy().to_string());
            let launcher = resolve_mlx_launch();
            let backend_str = backend.as_str();
            let mut jaccl_args = vec![
                "--hostfile".to_string(), hf,
                "--backend".to_string(), backend_str.to_string(),
                "--env".to_string(), format!("MLX_DISTRIBUTED_BACKEND={backend_str}"),
                "--env".to_string(), "MLX_METAL_FAST_SYNCH=1".to_string(),
            ];
            for (k, v) in allowlisted_env(req.env.as_ref()) {
                jaccl_args.push("--env".to_string());
                jaccl_args.push(format!("{k}={v}"));
            }
            jaccl_args.push("--".to_string());
            jaccl_args.push(program);
            jaccl_args.extend(cmd_args);
            (launcher, jaccl_args)
        } else {
            (program, cmd_args)
        };

        Ok((final_program, final_args))
    }

    /// Representative (label, engine, backend, request) cases — one per engine
    /// plus every conditional code path (bare, full-flags, extra_args, native
    /// ctx/mtp, distributed wrapper, media). env kept to None on the distributed
    /// case so HashMap iteration order can't make the assertion flaky.
    fn cases() -> Vec<(&'static str, ServeEngine, ServeBackend, LoadRequest)> {
        let m = || Some("/m/model".to_string());
        vec![
            ("mlx_lm_bare", ServeEngine::MlxLm, ServeBackend::Single, LoadRequest::default()),
            ("mlx_lm_basic", ServeEngine::MlxLm, ServeBackend::Single,
                LoadRequest { model_path: m(), ..Default::default() }),
            ("mlx_lm_full", ServeEngine::MlxLm, ServeBackend::Single, LoadRequest {
                model_path: m(), draft_model: Some("/m/draft".into()), num_draft_tokens: Some(3),
                decode_concurrency: Some(1), prompt_concurrency: Some(4), prefill_step_size: Some(2048),
                prompt_cache_size: Some(8), prompt_cache_bytes: Some(34_359_738_368),
                pipeline: true, use_mtp: true, cache_type: Some("q8".into()), max_tokens: Some(4096),
                ..Default::default()
            }),
            ("mlx_lm_extra", ServeEngine::MlxLm, ServeBackend::Single, LoadRequest {
                model_path: m(), extra_args: Some(vec!["--temp".into(), "0.7".into()]), ..Default::default()
            }),
            ("mlx_lm_jaccl", ServeEngine::MlxLm, ServeBackend::Jaccl, LoadRequest {
                model_path: m(), hostfile: Some("/hf/hosts.json".into()), ..Default::default()
            }),
            ("mlx_vlm_model", ServeEngine::MlxVlm, ServeBackend::Single,
                LoadRequest { model_path: m(), ..Default::default() }),
            ("mlx_vlm_bare", ServeEngine::MlxVlm, ServeBackend::Single, LoadRequest::default()),
            ("vllm_mlx", ServeEngine::VllmMlx, ServeBackend::Single,
                LoadRequest { model_path: m(), ..Default::default() }),
            ("mlx_lm_share", ServeEngine::MlxLmShare, ServeBackend::Single,
                LoadRequest { model_path: m(), ..Default::default() }),
            ("dflash", ServeEngine::DFlash, ServeBackend::Single,
                LoadRequest { model_path: m(), draft_model: Some("/m/draft".into()), ..Default::default() }),
            ("ds4_full", ServeEngine::Ds4, ServeBackend::Single, LoadRequest {
                model_path: m(), ctx_size: Some(262_144), draft_model: Some("/m/draft".into()),
                num_draft_tokens: Some(3), ..Default::default()
            }),
            ("ds4_extra", ServeEngine::Ds4, ServeBackend::Single, LoadRequest {
                model_path: m(), extra_args: Some(vec!["--foo".into(), "bar".into()]), ..Default::default()
            }),
            ("image_gen", ServeEngine::ImageGen, ServeBackend::Single, LoadRequest::default()),
            ("video_gen", ServeEngine::VideoGen, ServeBackend::Single, LoadRequest::default()),

            // ── Adversarial cases — the per-engine QUIRKS a naive data model
            // would get wrong. These are the ones that make the parity gate real.
            // (a) ds4 mtp is NESTED: --mtp-draft only if draft_model ALSO set. So
            // num_draft_tokens WITHOUT draft_model → ds4 emits NOTHING for mtp.
            ("ds4_ndt_no_draft", ServeEngine::Ds4, ServeBackend::Single, LoadRequest {
                model_path: m(), num_draft_tokens: Some(5), ..Default::default()
            }),
            // (b) ds4 draft WITHOUT num_draft_tokens → `--mtp <file>` only, no --mtp-draft.
            ("ds4_draft_no_ndt", ServeEngine::Ds4, ServeBackend::Single, LoadRequest {
                model_path: m(), draft_model: Some("/m/draft".into()), ..Default::default()
            }),
            // (c) mlx_lm is INDEPENDENT (asymmetric vs ds4): num_draft_tokens
            // WITHOUT draft_model → `--num-draft-tokens N` IS emitted.
            ("mlx_lm_ndt_no_draft", ServeEngine::MlxLm, ServeBackend::Single, LoadRequest {
                model_path: m(), num_draft_tokens: Some(5), ..Default::default()
            }),
            // (d) cache_type: ONE field → TWO flags (--cache-type-k + --cache-type-v).
            ("mlx_lm_cache_type_only", ServeEngine::MlxLm, ServeBackend::Single, LoadRequest {
                model_path: m(), cache_type: Some("q4".into()), ..Default::default()
            }),
            // (e) extra_args + distributed: extra_args join cmd_args BEFORE the
            // jaccl wrapper → they must land AFTER `--` (to the inner program).
            ("mlx_lm_jaccl_extra", ServeEngine::MlxLm, ServeBackend::Jaccl, LoadRequest {
                model_path: m(), hostfile: Some("/hf/hosts.json".into()),
                extra_args: Some(vec!["--temp".into(), "0.5".into()]), ..Default::default()
            }),
            // (f) model_flag:None (mlx_vlm) + distributed + non-bare → must NOT
            // wrap (guard requires model_flag.is_some()).
            ("mlx_vlm_jaccl_noswrap", ServeEngine::MlxVlm, ServeBackend::Jaccl, LoadRequest {
                model_path: m(), hostfile: Some("/hf/hosts.json".into()), ..Default::default()
            }),
        ]
    }

    /// Baseline: production == frozen legacy for every case. Trivially green now
    /// (identical code); becomes the real gate once build_serve_argv goes
    /// data-driven. `{:?}` on the Result compares Ok values AND Err messages,
    /// so media-script-absent machines still assert equal.
    #[test]
    fn production_matches_legacy_oracle() {
        // Fixed port for all cases — build_serve_argv uses it verbatim on both
        // sides (fixed-port pinning lives in the caller, not here), so parity
        // holds regardless of the value.
        const PORT: u16 = 19080;
        for (label, engine, backend, req) in cases() {
            let prod = format!("{:?}", build_serve_argv(&req, PORT, engine, backend));
            let legacy = format!("{:?}", build_serve_argv_legacy(&req, PORT, engine, backend));
            assert_eq!(prod, legacy, "build_serve_argv parity mismatch for case '{label}'");
        }
    }
}

#[cfg(test)]
mod build_serve_argv_tests {
    use super::*;

    #[test]
    fn mlx_lm_has_model_port_and_appends_extra_args_last() {
        let req = LoadRequest {
            model_path: Some("/Users/ma/Models/Qwen3.6-27B-mlx-4bit".to_string()),
            engine: ServeEngine::MlxLm,
            extra_args: Some(vec!["--temp".into(), "0.7".into()]),
            ..Default::default()
        };
        let (_program, args) =
            build_serve_argv(&req, 19080, ServeEngine::MlxLm, ServeBackend::Single).unwrap();
        // asmi invokes the mlx_lm SUBCOMMAND form (`python -m mlx_lm server`),
        // NOT the module form (`-m mlx_lm.server`) the web config assumes — the
        // exact reason the preview must come from asmi, not a web reconstruction.
        assert!(
            args.windows(3).any(|w| w[0] == "-m" && w[1] == "mlx_lm" && w[2] == "server"),
            "expected `-m mlx_lm server`; got {args:?}"
        );
        assert!(
            args.windows(2)
                .any(|w| w[0] == "--model" && w[1] == "/Users/ma/Models/Qwen3.6-27B-mlx-4bit"),
            "expected --model <path>; got {args:?}"
        );
        assert!(
            args.windows(2).any(|w| w[0] == "--port" && w[1] == "19080"),
            "expected --port 19080; got {args:?}"
        );
        // Generic passthrough must be appended LAST (Single backend = no wrapper).
        assert_eq!(
            &args[args.len() - 2..],
            &["--temp".to_string(), "0.7".to_string()],
            "extra_args must be last; got {args:?}"
        );
    }

    #[test]
    fn bare_request_omits_model_flag() {
        let req = LoadRequest { engine: ServeEngine::MlxLm, ..Default::default() };
        let (_p, args) =
            build_serve_argv(&req, 19080, ServeEngine::MlxLm, ServeBackend::Single).unwrap();
        assert!(
            !args.iter().any(|a| a == "--model"),
            "bare start must omit --model; got {args:?}"
        );
    }
}

async fn do_serve_load_inner(
    inner: &Arc<RwLock<ManagedProcess>>,
    readiness: &Arc<HttpHealth>,
    req: &LoadRequest,
) -> Result<(), anyhow::Error> {
    let (port, engine) = {
        let mut s = inner.write().await;
        kill_child(&mut s).await;
        (s.port.unwrap_or(19080), req.engine)
    };

    // Media gen servers never pre-load a model into the HTTP process — weights
    // load per request in a CLI subprocess. Drop any model_path so the slot
    // lands in Bare (= healthy, request-routed) instead of a fake Ready.
    let mut req = req.clone();
    if engine.is_media() {
        req.model_path = None;
    }
    let req = &req;

    let is_bare = req.model_path.is_none();

    // Validate optimization param conflicts
    if req.draft_model.is_some() {
        if let Some(dc) = req.decode_concurrency {
            if dc > 1 {
                anyhow::bail!(
                    "draft_model (speculative decoding) is incompatible with decode_concurrency > 1 (batching). \
                     Set decode_concurrency to 1 or remove draft_model."
                );
            }
        }
    }

    // Engines with model_flag: None lazy-load models via request body (e.g. mlx_vlm).
    // The server starts bare, then a warmup request pre-loads the model.
    let cfg_check = engine.config();
    let lazy_load = cfg_check.model_flag.is_none() && req.model_path.is_some();

    // Expand ~ in model path (no shell to do it for us)
    let mut req = req.clone();
    if let Some(ref mut path) = req.model_path {
        if path.starts_with("~/") {
            if let Some(home) = dirs::home_dir() {
                *path = format!("{}/{}", home.display(), &path[2..]);
            }
        }
    }

    // Check port free — if occupied, probe for a model server to adopt
    if tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
    {
        match probe_model_server(port).await {
            Some((pid, model)) => {
                let mut s = inner.write().await;
                // Don't overwrite a manager that has its own spawned child. The
                // sibling `state == Loading` check that lived here previously was
                // self-referential — `ServeManager::load` sets state to Loading at
                // the call-site BEFORE spawning the task that runs this code, so
                // the guard always tripped and every `/serve/load` against an
                // occupied port bailed with "manager is already loading". The
                // child-handle check above already covers the only real race
                // (another load() call that actually got far enough to spawn a
                // process); state alone is not load-bearing here.
                if s.child.is_some() {
                    anyhow::bail!("port {} occupied but manager already has a spawned child", port);
                }
                if let Some(stopped) = s.stopped_at {
                    if stopped.elapsed() < std::time::Duration::from_secs(10) {
                        anyhow::bail!("port {} occupied but manager in stop cooldown", port);
                    }
                }
                tracing::info!(port, pid, model = model.as_deref(), "port occupied by model server, adopting");
                s.pid = Some(pid);
                s.engine = engine;
                s.backend = ServeBackend::Single;
                s.state = if model.is_some() { ServeState::Ready } else { ServeState::Bare };
                s.model = model;
                s.error = None;
                s.stopped_at = None;
                return Ok(());
            }
            None => {
                anyhow::bail!("port {} already in use by a non-model process", port);
            }
        }
    }

    // Resolve backend (bare always single)
    let backend = if is_bare {
        ServeBackend::Single
    } else {
        resolve_backend_validated(&req.backend, req.hostfile.as_deref()).await
    };

    // Build the exact (program, args) via the shared builder — the SAME code the
    // dry-run preview uses, so preview and reality can't drift. Also appends
    // req.extra_args. `is_native` is still needed below (ds4 shader cwd).
    let is_native = matches!(engine, ServeEngine::Ds4);
    let (final_program, final_args) = build_serve_argv(&req, port, engine, backend)?;

    // Spawn — truncate log so read_log_tail reads only this run's output
    let log_path = format!("/tmp/r1o-mlx-server-{port}.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)?;
    let log_stderr = log_file.try_clone()?;

    tracing::info!(
        program = %final_program,
        args = ?final_args,
        %log_path,
        bare = is_bare,
        "spawning MLX server"
    );

    let mut spawn_cmd = Command::new(&final_program);
    spawn_cmd
        .args(&final_args)
        .env("MLX_METAL_FAST_SYNCH", "1")
        .stdout(log_file)
        .stderr(log_stderr)
        .kill_on_drop(false); // we manage lifetime ourselves

    // Native engines (ds4) need Metal shader source files at runtime. ds4
    // looks for `metal/flash_attn.metal` relative to cwd OR from the
    // `DS4_METAL_FLASH_ATTN_SOURCE` env var. The binary may be installed at
    // ~/.r1o/bin/ds4-server (managed location) far from the source tree, so
    // cwd-relative resolution fails. We search three paths for the shaders:
    //   1. Next to the binary (in-tree dev build: ~/opensource/ds4/)
    //   2. ~/opensource/ds4/ (the canonical ds4 source tree on this cluster)
    //   3. ~/.r1o/ds4/ (managed shader install, future)
    // The first hit sets both cwd and the env var. If none found, ds4 falls
    // back to its own resolution (may still fail, but we don't mask the error).
    if is_native {
        let candidates: Vec<std::path::PathBuf> = {
            let mut v = Vec::new();
            if let Some(p) = std::path::Path::new(&final_program).parent() {
                v.push(p.to_path_buf());
            }
            if let Some(home) = dirs::home_dir() {
                v.push(home.join("opensource/ds4"));
                v.push(home.join(".r1o/ds4"));
            }
            v
        };
        for dir in &candidates {
            let shader = dir.join("metal").join("flash_attn.metal");
            if shader.exists() {
                spawn_cmd.current_dir(dir);
                spawn_cmd.env("DS4_METAL_FLASH_ATTN_SOURCE", &shader);
                break;
            }
        }
    }

    // Generic env lock-in for the LOCAL process (single-node serves, and the
    // mlx.launch launcher itself). Distributed ranks get the same vars via
    // the --env forwarding above. Applied before the typed VLM fields below
    // so the explicit fields win on key collision.
    if backend.is_distributed() {
        spawn_cmd.env("MLX_DISTRIBUTED_BACKEND", backend.as_str());
    }
    for (k, v) in allowlisted_env(req.env.as_ref()) {
        spawn_cmd.env(k, v);
    }

    // VLM KV-quant / vision-cache tuning is ENV-driven: mlx_vlm reads KV_BITS,
    // KV_QUANT_SCHEME, and MLX_VLM_VISION_CACHE_SIZE at startup (see
    // mlx_vlm/server/generation.py). mlx_lm has no equivalent env hooks — its
    // tuning is the CLI flags emitted above — so these apply to the VLM engine only.
    if matches!(engine, ServeEngine::MlxVlm) {
        if let Some(bits) = req.kv_bits {
            spawn_cmd.env("KV_BITS", format!("{bits}"));
        }
        if let Some(ref scheme) = req.kv_quant_scheme {
            if !scheme.is_empty() {
                spawn_cmd.env("KV_QUANT_SCHEME", scheme);
            }
        }
        if let Some(n) = req.vision_cache_size {
            spawn_cmd.env("MLX_VLM_VISION_CACHE_SIZE", n.to_string());
        }
    }

    // Media gen servers read port/bind from env (no --port flag). Loopback
    // bind is intentional: the asmi daemon proxies /media/* for the mesh.
    match engine {
        ServeEngine::ImageGen => {
            spawn_cmd.env("IMAGE_GEN_PORT", port.to_string());
        }
        ServeEngine::VideoGen => {
            spawn_cmd.env("VIDEO_GEN_PORT", port.to_string());
        }
        _ => {}
    }

    let mut child = spawn_cmd.spawn()?;

    let child_pid = child.id().unwrap_or(0);

    // Configurable warmup timeout: bare servers (and lazy-load servers that start bare)
    // should start fast. Only engines that pre-load via --model need the long timeout.
    let timeout_secs = if is_bare || lazy_load {
        WARMUP_TIMEOUT_BARE_SECS
    } else {
        WARMUP_TIMEOUT_MODEL_SECS
    };

    // Use the readiness check (HTTP health polling racing against child exit).
    let health_result = readiness.poll_ready(&mut child, timeout_secs).await;

    let mut s = inner.write().await;
    match health_result {
        Ok(true) if verify_port_owner(child_pid, port).await => {
            s.pid = Some(child_pid);
            s.child = Some(child);
            s.engine = engine;
            s.backend = backend;

            if is_bare {
                s.model = None;
                s.state = ServeState::Bare;
                tracing::info!(pid = child_pid, port, %engine, "bare server ready");
            } else {
                s.model = req.model_path.clone();
                s.state = ServeState::Ready;
                tracing::info!(model = ?req.model_path, pid = child_pid, port, "server ready");
            }
            persist_state(&s).await;

            // For lazy-load engines (model_flag: None with model_path), fire a warmup
            // request to pre-load the model via /chat/completions. This is fire-and-forget:
            // if it fails, the model loads on the first real user request instead.
            if lazy_load {
                if let Some(ref model_path) = req.model_path {
                    let url = format!("http://localhost:{port}/chat/completions");
                    let model_path = model_path.clone();
                    tracing::info!(%url, model = %model_path, "firing warmup request for lazy-load engine");
                    tokio::spawn(async move {
                        let body = serde_json::json!({
                            "model": model_path,
                            "messages": [{"role": "user", "content": "warmup"}],
                            "max_tokens": 1
                        });
                        match reqwest::Client::new()
                            .post(&url)
                            .json(&body)
                            .timeout(std::time::Duration::from_secs(WARMUP_TIMEOUT_MODEL_SECS))
                            .send()
                            .await
                        {
                            Ok(resp) => tracing::info!(status = %resp.status(), "warmup complete — model pre-loaded"),
                            Err(e) => tracing::warn!(error = %e, "warmup failed — model will load on first request"),
                        }
                    });
                }
            }
        }
        Ok(true) => {
            s.state = ServeState::Error;
            s.error = Some(format!(
                "server started but bound to wrong port (not {port})"
            ));
            let _ = child.kill().await;
        }
        Ok(false) => {
            tracing::error!(
                port, %engine, timeout_secs,
                "warmup timeout exceeded — killing stuck process"
            );
            s.state = ServeState::Error;
            s.error = Some(format!(
                "warmup timeout exceeded ({timeout_secs}s) — process killed"
            ));
            let _ = child.kill().await;
        }
        Err(crash_msg) => {
            s.state = ServeState::Error;
            s.error = Some(crash_msg.clone());
            tracing::error!(%crash_msg, port, "server process crashed during startup");
            // Child already exited — no need to kill
        }
    }

    Ok(())
}

// ===========================================================================
// ShareManager = ProcessManager<LogMonitor>
// ===========================================================================

/// Backward-compatible type alias.
pub type ShareManager = ProcessManager<LogMonitor>;

impl ShareManager {
    /// Create a new idle share manager.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ManagedProcess {
                state: ServeState::Idle,
                model: None,
                engine: ServeEngine::MlxLmShare,
                backend: ServeBackend::Single,
                port: None,
                child: None,
                pid: None,
                load_started: None,
                error: None,
                stopped_at: None,
                port_verified_cached: false,
                port_verified_at: None,
            })),
            readiness: Arc::new(LogMonitor {
                log_path: SHARE_LOG_PATH.to_string(),
                ready_markers: vec![
                    // mlx_lm.server (uvicorn)
                    "Uvicorn running on",
                    "Application startup complete",
                    // Legacy markers
                    "Starting endpoint",
                    "Connected to",
                    "Listening on",
                ],
                error_markers: vec![
                    "Error:",
                    "Exception:",
                    "ValueError:",
                    "RuntimeError:",
                ],
            }),
        }
    }

    /// Create a share manager and restore from persisted state.
    /// If saved state has a model → restart the share session.
    pub async fn restore() -> Self {
        let mgr = Self::new();
        let sf = share_state_file();
        if sf.exists() {
            if let Ok(data) = tokio::fs::read_to_string(&sf).await {
                if let Ok(saved) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(model) = saved.get("model").and_then(|v| v.as_str()) {
                        if !model.is_empty() {
                            let backend = saved
                                .get("backend")
                                .and_then(|v| v.as_str())
                                .unwrap_or("auto")
                                .to_string();
                            let hostfile = saved
                                .get("hostfile")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            let req = ShareRequest {
                                model_path: model.to_string(),
                                backend,
                                hostfile,
                            };
                            tracing::info!(model, "restoring last share session");
                            mgr.start(req).await;
                            return mgr;
                        }
                    }
                }
            }
        }
        mgr
    }

    /// Start a share session. Spawns a background task and returns immediately.
    pub async fn start(&self, req: ShareRequest) {
        {
            let mut s = self.inner.write().await;
            kill_child(&mut s).await;
            s.state = ServeState::Loading;
            s.error = None;
            s.load_started = Some(std::time::Instant::now());
        }
        let inner = self.inner.clone();
        let readiness = self.readiness.clone();
        tokio::spawn(async move {
            do_share_load(inner, readiness, req).await;
        });
    }

    /// Get a read-only status snapshot.
    pub async fn status(&self) -> ShareStatus {
        let s = self.inner.read().await;
        let elapsed = s
            .load_started
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);
        ShareStatus {
            state: s.state,
            model: s.model.clone(),
            backend: s.backend,
            pid: s.pid,
            elapsed_ms: elapsed,
            error: s.error.clone(),
        }
    }

    /// Adopt an externally-spawned child process (used by /serve/distributed/join).
    pub async fn adopt_child(
        &self,
        child: tokio::process::Child,
        model: &str,
        backend: ServeBackend,
    ) {
        let mut s = self.inner.write().await;
        let pid = child.id().unwrap_or(0);
        s.pid = Some(pid);
        s.child = Some(child);
        s.model = Some(model.to_string());
        s.backend = backend;
        s.state = ServeState::Loading;
        s.load_started = Some(std::time::Instant::now());
        tracing::info!(pid, model, "adopted distributed worker process");
    }

}

/// Background share load task.
async fn do_share_load(inner: Arc<RwLock<ManagedProcess>>, readiness: Arc<LogMonitor>, req: ShareRequest) {
    let result = do_share_load_inner(&inner, &readiness, &req).await;
    if let Err(e) = result {
        let mut s = inner.write().await;
        s.state = ServeState::Error;
        s.error = Some(e.to_string());
    }
}

async fn do_share_load_inner(
    inner: &Arc<RwLock<ManagedProcess>>,
    readiness: &Arc<LogMonitor>,
    req: &ShareRequest,
) -> Result<(), anyhow::Error> {
    {
        let mut s = inner.write().await;
        kill_child(&mut s).await;
    }

    // Expand ~ in model path
    let mut model_path = req.model_path.clone();
    if model_path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            model_path = format!("{}/{}", home.display(), &model_path[2..]);
        }
    }

    // Resolve backend
    let backend = resolve_backend_validated(&req.backend, req.hostfile.as_deref()).await;

    let py = resolve_python().to_string();

    // For distributed JACCL: orchestrate via asmi peer HTTP APIs
    // For single-node: run python3 -m mlx_lm.server directly
    if backend == ServeBackend::Jaccl {
        let hf_path = req
            .hostfile
            .clone()
            .unwrap_or_else(|| default_hostfile().to_string_lossy().to_string());
        return do_jaccl_orchestrate(inner, readiness, &model_path, &hf_path).await;
    }

    let model_args = vec![
        "--model".to_string(),
        model_path.clone(),
        "--port".to_string(),
        SHARE_PORT.to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
    ];
    let final_program = py;
    let mut final_args = vec!["-m".to_string(), "mlx_lm".to_string(), "server".to_string()];
    final_args.extend(model_args);

    // Truncate log for fresh output
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(SHARE_LOG_PATH)?;
    let log_stderr = log_file.try_clone()?;

    tracing::info!(
        program = %final_program,
        args = ?final_args,
        log_path = SHARE_LOG_PATH,
        "spawning distributed mlx_lm.server"
    );

    let mut child = Command::new(&final_program)
        .args(&final_args)
        .env("MLX_METAL_FAST_SYNCH", "1")
        .stdout(log_file)
        .stderr(log_stderr)
        .kill_on_drop(false)
        .spawn()?;

    let child_pid = child.id().unwrap_or(0);

    // Use the readiness check (log monitoring racing against child exit).
    let readiness_result = readiness.poll_ready(&mut child, WARMUP_TIMEOUT_SHARE_SECS).await;

    let mut s = inner.write().await;
    match readiness_result {
        Ok(true) => {
            s.pid = Some(child_pid);
            s.child = Some(child);
            s.model = Some(model_path);
            s.backend = backend;
            s.state = ServeState::Ready;
            tracing::info!(pid = child_pid, model = ?req.model_path, "share session ready");
            persist_state(&s).await;
        }
        Ok(false) => {
            tracing::error!(
                timeout_secs = WARMUP_TIMEOUT_SHARE_SECS,
                "share warmup timeout exceeded — killing stuck process"
            );
            s.state = ServeState::Error;
            let detail = read_log_tail(SHARE_LOG_PATH, 10).await;
            s.error = Some(format!(
                "warmup timeout exceeded ({WARMUP_TIMEOUT_SHARE_SECS}s) — process killed: {detail}"
            ));
            let _ = child.kill().await;
        }
        Err(crash_msg) => {
            s.state = ServeState::Error;
            s.error = Some(crash_msg.clone());
            tracing::error!(%crash_msg, "share process crashed during startup");
        }
    }

    Ok(())
}

// ===========================================================================
// JACCL orchestration via asmi peer HTTP APIs
// ===========================================================================

/// Orchestrate distributed JACCL inference by calling each peer's asmi daemon.
/// No SSH, no mlx.launch — asmi is the launcher on every node.
async fn do_jaccl_orchestrate(
    inner: &Arc<RwLock<ManagedProcess>>,
    readiness: &Arc<LogMonitor>,
    model_path: &str,
    hostfile_path: &str,
) -> Result<(), anyhow::Error> {
    use serde_json::json;

    // Parse hostfile to get hosts + RDMA matrix
    let hf_content = tokio::fs::read_to_string(hostfile_path).await?;
    let hf: serde_json::Value = serde_json::from_str(&hf_content)?;
    let hosts = hf.get("hosts")
        .and_then(|h| h.as_array())
        .ok_or_else(|| anyhow::anyhow!("hostfile missing 'hosts' array"))?;

    let world_size = hosts.len() as u32;
    if world_size < 2 {
        anyhow::bail!("need >= 2 hosts for distributed, got {world_size}");
    }

    // Coordinator is rank 0's IP
    let coordinator_ip = hosts[0]
        .get("ips").and_then(|i| i.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("rank 0 missing ips in hostfile"))?;
    let coordinator = format!("{coordinator_ip}:32323");

    // Build backend-specific payloads
    let backend_str = hf.get("backend").and_then(|b| b.as_str()).unwrap_or("ring");

    let ibv_devices: Vec<serde_json::Value> = hosts.iter()
        .map(|h| h.get("rdma").cloned().unwrap_or(json!([])))
        .collect();
    let ibv_json = serde_json::to_string(&ibv_devices)?;

    // Ring hostfile: [["ip1:port1"], ["ip2:port2"]]
    let ring_port_start = 32323u16;
    let ring_hostfile: Vec<Vec<String>> = hosts.iter().enumerate()
        .map(|(i, h)| {
            let ip = h.get("ips").and_then(|a| a.as_array())
                .and_then(|a| a.first()).and_then(|v| v.as_str())
                .unwrap_or("127.0.0.1");
            vec![format!("{}:{}", ip, ring_port_start + i as u16)]
        })
        .collect();
    let ring_hostfile_json = serde_json::to_string(&ring_hostfile)?;

    tracing::info!(
        world_size,
        coordinator = %coordinator,
        backend = backend_str,
        model = model_path,
        "orchestrating distributed session via asmi peers"
    );

    // Step 1: Call each remote peer's /serve/distributed/join
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let mut peer_results = Vec::new();
    // If hub is NOT rank 0 (orchestrator-only), recruit ALL ranks including rank 0
    // If hub IS rank 0, skip rank 0 (started locally below)
    let local_hostname = std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let rank0_ssh = hosts[0].get("ssh").and_then(|s| s.as_str()).unwrap_or("");
    let hub_is_rank0 = rank0_ssh == "127.0.0.1"
        || rank0_ssh == "localhost"
        || rank0_ssh == local_hostname;
    let start_rank = if hub_is_rank0 { 1 } else { 0 };

    for (rank, host) in hosts.iter().enumerate().skip(start_rank) {
        let ssh_name = host.get("ssh").and_then(|s| s.as_str()).unwrap_or("unknown");
        let peer_url = format!("http://{}:9090/serve/distributed/join", ssh_name);

        tracing::info!(rank, peer = ssh_name, "recruiting peer");
        let resp = client.post(&peer_url)
            .json(&json!({
                "model_path": model_path,
                "rank": rank,
                "world_size": world_size,
                "coordinator": coordinator,
                "backend": backend_str,
                "ibv_devices": ibv_json,
                "ring_hostfile": ring_hostfile_json,
                "port": SHARE_PORT,
            }))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let body: serde_json::Value = r.json().await.unwrap_or(json!({"ok": false}));
                tracing::info!(rank, peer = ssh_name, pid = ?body.get("pid"), "peer joined");
                peer_results.push((rank, ssh_name.to_string(), true));
            }
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                tracing::error!(rank, peer = ssh_name, %status, body = %body, "peer join failed");
                peer_results.push((rank, ssh_name.to_string(), false));
            }
            Err(e) => {
                tracing::error!(rank, peer = ssh_name, error = %e, "peer unreachable");
                peer_results.push((rank, ssh_name.to_string(), false));
            }
        }
    }

    // Check all peers joined
    let failed: Vec<_> = peer_results.iter().filter(|(_, _, ok)| !ok).collect();
    if !failed.is_empty() {
        let names: Vec<_> = failed.iter().map(|(r, n, _)| format!("rank{r}={n}")).collect();
        anyhow::bail!("peers failed to join: {}", names.join(", "));
    }

    // Step 2: Start rank 0 locally (only if hub IS rank 0)
    if !hub_is_rank0 {
        // Hub is orchestrator-only — all ranks run on remote peers
        // Monitor readiness via HTTP to rank 0's node
        let rank0_url = format!("http://{}:{}/v1/models", rank0_ssh, SHARE_PORT);
        tracing::info!(rank0_url = %rank0_url, "hub is orchestrator-only, polling rank 0 remotely");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(WARMUP_TIMEOUT_SHARE_SECS);
        loop {
            if let Ok(resp) = reqwest::Client::new()
                .get(&rank0_url)
                .timeout(std::time::Duration::from_secs(2))
                .send().await
            {
                if resp.status().is_success() {
                    let mut s = inner.write().await;
                    s.model = Some(model_path.to_string());
                    s.backend = ServeBackend::Jaccl;
                    s.state = ServeState::Ready;
                    tracing::info!(world_size, "distributed session ready (orchestrator-only mode)");
                    return Ok(());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                let mut s = inner.write().await;
                s.state = ServeState::Error;
                s.error = Some("timeout waiting for rank 0 to become ready".into());
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    }

    let py = resolve_python().to_string();
    let ibv_tmp = std::env::temp_dir().join("asmi-ibv-0.json");
    tokio::fs::write(&ibv_tmp, &ibv_json).await?;

    let log_file = std::fs::OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(SHARE_LOG_PATH)?;
    let log_stderr = log_file.try_clone()?;

    let mut cmd = Command::new(&py);
    cmd.arg("-m").arg("mlx_lm").arg("server")
        .arg("--model").arg(model_path)
        .arg("--port").arg(SHARE_PORT.to_string())
        .arg("--host").arg("0.0.0.0")
        .env("MLX_RANK", "0")
        .env("MLX_WORLD_SIZE", world_size.to_string())
        .env("MLX_METAL_FAST_SYNCH", "1")
        .stdout(log_file)
        .stderr(log_stderr)
        .kill_on_drop(false);

    if backend_str == "jaccl" {
        cmd.env("MLX_DISTRIBUTED_BACKEND", "jaccl")
            .env("MLX_JACCL_COORDINATOR", &coordinator)
            .env("MLX_IBV_DEVICES", ibv_tmp.to_string_lossy().to_string());
    } else {
        // Ring: write hostfile JSON to temp file, set env to file path
        let ring_tmp = std::env::temp_dir().join("asmi-ring-0.json");
        tokio::fs::write(&ring_tmp, &ring_hostfile_json).await?;
        cmd.env("MLX_DISTRIBUTED_BACKEND", "ring")
            .env("MLX_HOSTFILE", ring_tmp.to_string_lossy().to_string());
    }

    tracing::info!(
        model = model_path,
        port = SHARE_PORT,
        "starting rank 0 locally"
    );

    let mut child = cmd.spawn()?;
    let child_pid = child.id().unwrap_or(0);

    // Step 3: Wait for readiness (HTTP health check + log markers)
    let readiness_result = readiness.poll_ready(&mut child, WARMUP_TIMEOUT_SHARE_SECS).await;

    let mut s = inner.write().await;
    match readiness_result {
        Ok(true) => {
            s.pid = Some(child_pid);
            s.child = Some(child);
            s.model = Some(model_path.to_string());
            s.backend = ServeBackend::Jaccl;
            s.state = ServeState::Ready;
            tracing::info!(
                pid = child_pid,
                model = model_path,
                world_size,
                "distributed JACCL session ready"
            );
            persist_state(&s).await;
        }
        Ok(false) => {
            s.state = ServeState::Error;
            let detail = read_log_tail(SHARE_LOG_PATH, 10).await;
            s.error = Some(format!(
                "distributed warmup timeout ({WARMUP_TIMEOUT_SHARE_SECS}s) — {detail}"
            ));
            let _ = child.kill().await;
        }
        Err(crash_msg) => {
            s.state = ServeState::Error;
            s.error = Some(crash_msg);
        }
    }

    Ok(())
}

// ===========================================================================
// PeerHeartbeat — detect RDMA peer death to prevent GPU Lock
// ===========================================================================

use asmi_core::{PeerHeartbeatStatus, PeerStatus};
use tokio_util::sync::CancellationToken;

/// How often to ping each peer (seconds).
const HEARTBEAT_INTERVAL_SECS: u64 = 1;
/// How many consecutive misses before triggering emergency stop.
const HEARTBEAT_MISS_THRESHOLD: u32 = 3;

/// RDMA peer heartbeat monitor. Pings each peer's asmi `/health` endpoint
/// every second. If any peer misses 3 consecutive checks, kills all local
/// inference processes to prevent GPU Lock from hung Metal command buffers.
///
/// Thread-safe via `Arc` — all methods take `&self`.
pub struct PeerHeartbeat {
    status: Arc<RwLock<PeerHeartbeatStatus>>,
    state: tokio::sync::Mutex<HeartbeatState>,
}

struct HeartbeatState {
    cancel: Option<CancellationToken>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl PeerHeartbeat {
    /// Create a new (inactive) peer heartbeat monitor.
    pub fn new() -> Self {
        Self {
            status: Arc::new(RwLock::new(PeerHeartbeatStatus {
                active: false,
                peers: vec![],
                session_start: None,
            })),
            state: tokio::sync::Mutex::new(HeartbeatState {
                cancel: None,
                handle: None,
            }),
        }
    }

    /// Start monitoring peers. Pings each peer's asmi health endpoint at `asmi_port`.
    /// If any peer is unreachable for 3+ consecutive checks, triggers emergency stop
    /// on all serve managers and the share manager.
    pub async fn start(
        &self,
        peer_hostnames: Vec<String>,
        asmi_port: u16,
        serve_managers: Arc<tokio::sync::RwLock<std::collections::HashMap<u16, ServeManager>>>,
        share_manager: ShareManager,
    ) {
        // Stop any existing heartbeat first
        self.stop().await;

        if peer_hostnames.is_empty() {
            return;
        }

        // Initialize status with peer list
        {
            let mut s = self.status.write().await;
            s.active = true;
            s.session_start = Some(chrono::Utc::now().to_rfc3339());
            s.peers = peer_hostnames
                .iter()
                .map(|h| PeerStatus {
                    hostname: h.clone(),
                    reachable: true,
                    last_seen: None,
                    consecutive_misses: 0,
                })
                .collect();
        }

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let status = self.status.clone();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel_clone.cancelled() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(HEARTBEAT_INTERVAL_SECS)) => {
                        // Ping all peers concurrently
                        let results: Vec<bool> = futures::future::join_all(
                            peer_hostnames.iter().map(|peer| {
                                let client = client.clone();
                                let url = format!("http://{}:{}/health", peer, asmi_port);
                                async move {
                                    matches!(
                                        client.get(&url).send().await,
                                        Ok(resp) if resp.status().is_success()
                                    )
                                }
                            })
                        ).await;

                        // Update status and check for dead peers
                        let mut any_dead = false;
                        {
                            let mut s = status.write().await;
                            for (i, reachable) in results.iter().enumerate() {
                                if let Some(ps) = s.peers.get_mut(i) {
                                    if *reachable {
                                        ps.reachable = true;
                                        ps.last_seen = Some(chrono::Utc::now().to_rfc3339());
                                        ps.consecutive_misses = 0;
                                    } else {
                                        ps.reachable = false;
                                        ps.consecutive_misses += 1;
                                        if ps.consecutive_misses >= HEARTBEAT_MISS_THRESHOLD {
                                            tracing::error!(
                                                peer = %ps.hostname,
                                                misses = ps.consecutive_misses,
                                                "RDMA peer unreachable for {}s — killing local inference to prevent GPU Lock",
                                                ps.consecutive_misses
                                            );
                                            any_dead = true;
                                        }
                                    }
                                }
                            }
                        } // release status lock before emergency stop

                        if any_dead {
                            // EMERGENCY: Kill all local inference to prevent GPU Lock
                            for mgr in serve_managers.read().await.values() {
                                mgr.emergency_stop().await;
                            }
                            share_manager.emergency_stop().await;

                            // Mark heartbeat as inactive
                            let mut s = status.write().await;
                            s.active = false;
                            break;
                        }
                    }
                }
            }
        });

        let mut st = self.state.lock().await;
        st.cancel = Some(cancel);
        st.handle = Some(handle);
    }

    /// Stop the heartbeat loop.
    pub async fn stop(&self) {
        let mut st = self.state.lock().await;
        if let Some(cancel) = st.cancel.take() {
            cancel.cancel();
        }
        if let Some(handle) = st.handle.take() {
            handle.abort();
        }
        let mut s = self.status.write().await;
        s.active = false;
    }

    /// Get the current heartbeat status (lock-free read).
    pub async fn status(&self) -> PeerHeartbeatStatus {
        self.status.read().await.clone()
    }
}

/// Parse peer hostnames from a JACCL hostfile (JSON array with "ssh" fields).
/// Returns hostnames excluding `local_hostname`.
pub fn parse_hostfile_peers(hostfile_path: &str, local_hostname: &str) -> Vec<String> {
    let content = match std::fs::read_to_string(hostfile_path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let entries: Vec<serde_json::Value> = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    entries
        .iter()
        .filter_map(|e| {
            e.get("ssh")
                .and_then(|v| v.as_str())
                .and_then(|ssh| ssh.split('@').nth(1))
                .map(|h| h.to_string())
        })
        .filter(|h| h != local_hostname)
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_peer_heartbeat_detects_dead_peer() {
        // Start a mock asmi health endpoint using axum
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({"ok": true}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mock_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Create heartbeat with empty managers (emergency_stop is a no-op on empty)
        let hb = Arc::new(PeerHeartbeat::new());
        let managers = Arc::new(tokio::sync::RwLock::new(HashMap::<u16, ServeManager>::new()));
        let share = ShareManager::new();

        hb.start(vec!["127.0.0.1".to_string()], port, managers, share)
            .await;

        // Let it detect the peer as alive
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let status = hb.status().await;
        assert!(status.active, "heartbeat should be active");
        assert_eq!(status.peers.len(), 1);
        assert!(status.peers[0].reachable, "peer should be reachable");
        assert_eq!(status.peers[0].consecutive_misses, 0);

        // Kill the mock server → peer goes dark
        mock_handle.abort();

        // Wait for 3+ missed heartbeats (3s interval + buffer)
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let status = hb.status().await;
        assert!(!status.peers[0].reachable, "peer should be unreachable");
        assert!(
            status.peers[0].consecutive_misses >= HEARTBEAT_MISS_THRESHOLD,
            "should have >= {} misses, got {}",
            HEARTBEAT_MISS_THRESHOLD,
            status.peers[0].consecutive_misses
        );
        // Heartbeat should have deactivated after emergency stop
        assert!(!status.active, "heartbeat should deactivate after peer death");

        hb.stop().await;
    }

    #[tokio::test]
    async fn test_peer_heartbeat_healthy_peer_stays_reachable() {
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({"ok": true}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mock_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let hb = Arc::new(PeerHeartbeat::new());
        let managers = Arc::new(tokio::sync::RwLock::new(HashMap::<u16, ServeManager>::new()));
        let share = ShareManager::new();

        hb.start(vec!["127.0.0.1".to_string()], port, managers, share)
            .await;

        // Let several cycles pass
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;

        let status = hb.status().await;
        assert!(status.active);
        assert!(status.peers[0].reachable);
        assert_eq!(status.peers[0].consecutive_misses, 0);
        assert!(status.peers[0].last_seen.is_some());

        hb.stop().await;
        mock_handle.abort();

        let status = hb.status().await;
        assert!(!status.active, "should be inactive after stop");
    }

    #[test]
    fn test_resolve_backend_all_strings() {
        // With an existing hostfile every distributed string must survive the
        // pipeline — "jaccl-ring"/"ring" collapsing to Single was the bug that
        // made the web UI's backend picker a no-op.
        let dir = std::env::temp_dir();
        let path = dir.join("test-resolve-backend-hostfile.json");
        std::fs::write(&path, "[]").unwrap();
        let hf = path.to_str().unwrap();

        assert_eq!(resolve_backend("jaccl", Some(hf)), ServeBackend::Jaccl);
        assert_eq!(resolve_backend("auto", Some(hf)), ServeBackend::Jaccl);
        assert_eq!(resolve_backend("jaccl-ring", Some(hf)), ServeBackend::JacclRing);
        assert_eq!(resolve_backend("ring", Some(hf)), ServeBackend::Ring);
        assert_eq!(resolve_backend("single", Some(hf)), ServeBackend::Single);
        assert_eq!(resolve_backend("bogus", Some(hf)), ServeBackend::Single);

        // Missing hostfile: every distributed request degrades to single
        let missing = "/nonexistent/hostfile.json";
        assert_eq!(resolve_backend("jaccl", Some(missing)), ServeBackend::Single);
        assert_eq!(resolve_backend("jaccl-ring", Some(missing)), ServeBackend::Single);
        assert_eq!(resolve_backend("ring", Some(missing)), ServeBackend::Single);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_backend_serde_roundtrip() {
        // The wire strings are a contract with the web LaunchRequest and the
        // _mlx_backend_fix hook — pin them.
        for (b, s) in [
            (ServeBackend::Single, "\"single\""),
            (ServeBackend::Jaccl, "\"jaccl\""),
            (ServeBackend::JacclRing, "\"jaccl-ring\""),
            (ServeBackend::Ring, "\"ring\""),
        ] {
            assert_eq!(serde_json::to_string(&b).unwrap(), s);
            assert_eq!(serde_json::from_str::<ServeBackend>(s).unwrap(), b);
            assert_eq!(format!("\"{b}\""), s);
        }
    }

    #[test]
    fn test_allowlisted_env_filters() {
        let mut env = std::collections::HashMap::new();
        env.insert("MLX_FOO".to_string(), "1".to_string());
        env.insert("KV_BITS".to_string(), "3.5".to_string());
        env.insert("HF_HOME".to_string(), "/Users/ma/.cache/huggingface".to_string());
        // Must be dropped: non-allowlisted prefix (interpreter hijack vectors)
        env.insert("PATH".to_string(), "/evil".to_string());
        env.insert("DYLD_INSERT_LIBRARIES".to_string(), "/evil.dylib".to_string());
        env.insert("PYTHONPATH".to_string(), "/evil".to_string());
        // Must be dropped: bad key chars / control chars in value
        env.insert("MLX_BAD-KEY".to_string(), "x".to_string());
        env.insert("MLX_NEWLINE".to_string(), "a\nb".to_string());

        let out = allowlisted_env(Some(&env));
        let keys: Vec<&str> = out.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["HF_HOME", "KV_BITS", "MLX_FOO"]); // sorted, filtered

        assert!(allowlisted_env(None).is_empty());
    }

    #[test]
    fn test_parse_hostfile_peers() {
        let dir = std::env::temp_dir();
        let path = dir.join("test-hostfile.json");
        std::fs::write(
            &path,
            r#"[
                {"ssh": "ma@m3u2", "rdma": ["169.254.1.1"]},
                {"ssh": "ma@m3u1", "rdma": ["169.254.1.2"]},
                {"ssh": "ma@m3u3", "rdma": ["169.254.1.3"]}
            ]"#,
        )
        .unwrap();

        let peers = parse_hostfile_peers(path.to_str().unwrap(), "m3u2");
        assert_eq!(peers, vec!["m3u1".to_string(), "m3u3".to_string()]);

        let peers = parse_hostfile_peers(path.to_str().unwrap(), "m3u1");
        assert_eq!(peers, vec!["m3u2".to_string(), "m3u3".to_string()]);

        // Non-existent file returns empty
        let peers = parse_hostfile_peers("/nonexistent/file.json", "m3u2");
        assert!(peers.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn test_warmup_timeout_returns_false() {
        // Bind a port but never accept connections — simulates a stuck process
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        // Don't accept — the port is bound but nobody responds to HTTP

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .unwrap();

        // poll_health with 2-second timeout should return Ok(false)
        let start = std::time::Instant::now();
        let result = poll_health(&client, port, &["/health"], 2).await;
        let elapsed = start.elapsed();

        assert_eq!(result.unwrap(), false, "should timeout, not succeed");
        assert!(
            elapsed.as_secs() >= 2,
            "should have waited at least 2s, got {:?}",
            elapsed
        );

        drop(listener);
    }

    #[tokio::test]
    async fn test_warmup_timeout_constants_are_sane() {
        // Bare timeout should be shorter than model timeout
        assert!(WARMUP_TIMEOUT_BARE_SECS < WARMUP_TIMEOUT_MODEL_SECS);
        // Model timeout should be at least 5 minutes
        assert!(WARMUP_TIMEOUT_MODEL_SECS >= 300);
        // Share timeout should be at least 5 minutes
        assert!(WARMUP_TIMEOUT_SHARE_SECS >= 300);
    }
}

#[cfg(test)]
mod backend_validation_tests {
    use super::*;

    #[tokio::test]
    async fn auto_falls_back_to_single_when_hostfile_hosts_dead() {
        let dir = std::env::temp_dir().join("asmi-test-hostfiles");
        std::fs::create_dir_all(&dir).unwrap();
        let hf = dir.join("dead-hosts.json");
        std::fs::write(
            &hf,
            r#"{"backend":"jaccl","hosts":[{"ssh":"asmi-test-nonexistent.invalid"}]}"#,
        )
        .unwrap();
        let resolved =
            resolve_backend_validated("auto", Some(hf.to_str().unwrap())).await;
        assert_eq!(resolved, ServeBackend::Single);
    }

    #[tokio::test]
    async fn explicit_jaccl_is_honored_without_validation() {
        let dir = std::env::temp_dir().join("asmi-test-hostfiles");
        std::fs::create_dir_all(&dir).unwrap();
        let hf = dir.join("explicit.json");
        std::fs::write(
            &hf,
            r#"{"backend":"jaccl","hosts":[{"ssh":"asmi-test-nonexistent.invalid"}]}"#,
        )
        .unwrap();
        let resolved =
            resolve_backend_validated("jaccl", Some(hf.to_str().unwrap())).await;
        assert_eq!(resolved, ServeBackend::Jaccl);
    }

    #[tokio::test]
    async fn unparseable_hostfile_falls_back_to_single() {
        let dir = std::env::temp_dir().join("asmi-test-hostfiles");
        std::fs::create_dir_all(&dir).unwrap();
        let hf = dir.join("garbage.json");
        std::fs::write(&hf, "not json").unwrap();
        let resolved =
            resolve_backend_validated("auto", Some(hf.to_str().unwrap())).await;
        assert_eq!(resolved, ServeBackend::Single);
    }
}
