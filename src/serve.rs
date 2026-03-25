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

/// Managed ports and their default engines.
/// This is the single source of truth for which ports asmi auto-starts.
pub const MANAGED_PORTS: &[(u16, ServeEngine)] = &[
    (19080, ServeEngine::MlxLm),
    (19082, ServeEngine::MlxVlm),
];

/// Share session log file.
const SHARE_LOG_PATH: &str = "/tmp/r1o-mlx-share.log";
/// Default port for the distributed inference server.
const SHARE_PORT: u16 = 19080;

/// Resolve the `mlx.launch` CLI script path.
/// Falls back to running as `python3 -m mlx._distributed_utils.launch`.
fn resolve_mlx_launch() -> String {
    // Try the pip-installed CLI script first
    if let Ok(output) = std::process::Command::new("which").arg("mlx.launch").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return path;
            }
        }
    }
    // Homebrew common path
    let brew_path = "/opt/homebrew/bin/mlx.launch";
    if std::path::Path::new(brew_path).exists() {
        return brew_path.to_string();
    }
    // Fallback: invoke as python module (caller must prepend "-m mlx._distributed_utils.launch")
    resolve_python().to_string()
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

/// Resolve "auto" backend to single or jaccl based on hostfile existence.
pub fn resolve_backend(backend: &str, hostfile: Option<&str>) -> ServeBackend {
    if backend == "single" {
        return ServeBackend::Single;
    }
    let hf = hostfile
        .map(PathBuf::from)
        .unwrap_or_else(default_hostfile);
    if hf.exists() && (backend == "jaccl" || backend == "auto") {
        ServeBackend::Jaccl
    } else {
        ServeBackend::Single
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
}

/// Kill the existing child process (SIGTERM → 5s → SIGKILL).
async fn kill_child(s: &mut ManagedProcess) {
    if let Some(ref mut child) = s.child {
        if let Some(pid) = s.pid {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );
        }
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                let _ = child.kill().await;
            }
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
        ServeStatus {
            state: s.state,
            model: s.model.clone(),
            engine: s.engine,
            backend: s.backend,
            port,
            pid: s.pid,
            port_verified,
            elapsed_ms: elapsed,
            error: s.error.clone(),
        }
    }
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

    let is_bare = req.model_path.is_none();

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

    // Check port free
    if tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
    {
        anyhow::bail!("port {} already in use by another process", port);
    }

    // Resolve backend (bare always single)
    let backend = if is_bare {
        ServeBackend::Single
    } else {
        resolve_backend(&req.backend, req.hostfile.as_deref())
    };

    // Build command
    let cfg = engine.config();
    let mut cmd_args: Vec<String> = Vec::new();
    let program: String;

    // Always invoke via resolve_python() since launchd doesn't have Homebrew in PATH.
    let py = resolve_python().to_string();

    if let Some(uvicorn_app) = cfg.uvicorn_app {
        // Uvicorn-wrapped engines (avoids reload=True bugs in mlx_vlm)
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
    }

    // JACCL distributed wrapper (only for engines with model_flag and non-bare)
    let (final_program, final_args) = if !is_bare
        && backend == ServeBackend::Jaccl
        && cfg.model_flag.is_some()
    {
        let hf = req
            .hostfile
            .clone()
            .unwrap_or_else(|| default_hostfile().to_string_lossy().to_string());
        let jaccl_py = resolve_python().to_string();
        let mut jaccl_args = vec![
            "-m".to_string(),
            "mlx.launch".to_string(),
            "--hostfile".to_string(),
            hf,
            "--backend".to_string(),
            "jaccl".to_string(),
            "--".to_string(),
            program,
        ];
        jaccl_args.extend(cmd_args);
        (jaccl_py, jaccl_args)
    } else {
        (program, cmd_args)
    };

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

    let mut child = Command::new(&final_program)
        .args(&final_args)
        .env("MLX_METAL_FAST_SYNCH", "1")
        .stdout(log_file)
        .stderr(log_stderr)
        .kill_on_drop(false) // we manage lifetime ourselves
        .spawn()?;

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
    let backend = resolve_backend(&req.backend, req.hostfile.as_deref());

    let py = resolve_python().to_string();
    let share_port = SHARE_PORT.to_string();

    // Build the target command: python3 -m mlx_lm.server --model <path> --port <port> --host 0.0.0.0
    let server_args = vec![
        "-m".to_string(),
        "mlx_lm.server".to_string(),
        "--model".to_string(),
        model_path.clone(),
        "--port".to_string(),
        share_port.clone(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
    ];

    // For distributed: wrap with mlx.launch --hostfile X --backend Y -- <server_args>
    // For single-node: run server directly
    let (final_program, final_args) = if backend == ServeBackend::Jaccl {
        let hf = req
            .hostfile
            .clone()
            .unwrap_or_else(|| default_hostfile().to_string_lossy().to_string());
        let launch = resolve_mlx_launch();
        let mut args = vec![
            "--hostfile".to_string(),
            hf,
            "--env".to_string(),
            "MLX_METAL_FAST_SYNCH=1".to_string(),
            "--".to_string(),
        ];
        args.push(py.clone());
        args.extend(server_args);
        (launch, args)
    } else {
        (py, server_args)
    };

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
        serve_managers: Arc<std::collections::HashMap<u16, ServeManager>>,
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
                            for mgr in serve_managers.values() {
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
        let managers = Arc::new(HashMap::<u16, ServeManager>::new());
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
        let managers = Arc::new(HashMap::<u16, ServeManager>::new());
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
