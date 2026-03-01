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
fn default_hostfile() -> PathBuf {
    r1o_dir().join("hostfiles/default.json")
}

/// Resolve "auto" backend to single or jaccl based on hostfile existence.
fn resolve_backend(backend: &str, hostfile: Option<&str>) -> ServeBackend {
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

/// Poll a log file for readiness/error markers.
/// Returns Ok(true) when ready, Ok(false) on timeout, Err on error markers.
async fn poll_log(
    log_path: &str,
    ready_markers: &[&str],
    error_markers: &[&str],
    timeout_secs: u64,
) -> Result<bool, String> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if let Ok(content) = tokio::fs::read_to_string(log_path).await {
            if ready_markers.iter().any(|m| content.contains(m)) {
                return Ok(true);
            }
            if error_markers.iter().any(|m| content.contains(m)) {
                let detail = read_log_tail(log_path, 10).await;
                return Err(format!("share error: {detail}"));
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

    // Use the readiness check (HTTP health polling racing against child exit).
    let health_result = readiness.poll_ready(&mut child, 60).await;

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
        }
        Ok(true) => {
            s.state = ServeState::Error;
            s.error = Some(format!(
                "server started but bound to wrong port (not {port})"
            ));
            let _ = child.kill().await;
        }
        Ok(false) => {
            s.state = ServeState::Error;
            s.error = Some(format!("timeout waiting for health check ({engine})"));
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
                    "Starting endpoint",
                    "Connected to",
                    "Listening on",
                ],
                error_markers: vec!["Error:", "Exception:"],
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

    // Build command: python3 -m mlx_lm.share --model <path>
    let py = resolve_python().to_string();
    let cfg = ServeEngine::MlxLmShare.config();
    let mut cmd_args: Vec<String> = vec![
        "-m".into(),
        cfg.binary.to_string(),
    ];
    if let Some(flag) = cfg.model_flag {
        cmd_args.push(flag.into());
        cmd_args.push(model_path.clone());
    }

    // JACCL distributed wrapper
    let (final_program, final_args) = if backend == ServeBackend::Jaccl {
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
            py,
        ];
        jaccl_args.extend(cmd_args);
        (jaccl_py, jaccl_args)
    } else {
        (py, cmd_args)
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
        "spawning mlx_lm.share"
    );

    let mut child = Command::new(&final_program)
        .args(&final_args)
        .env("MLX_METAL_FAST_SYNCH", "1")
        .stdout(log_file)
        .stderr(log_stderr)
        .kill_on_drop(false)
        .spawn()?;

    let child_pid = child.id().unwrap_or(0);

    // Use the readiness check (log monitoring racing against child exit, 120s timeout).
    let readiness_result = readiness.poll_ready(&mut child, 120).await;

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
            s.state = ServeState::Error;
            let detail = read_log_tail(SHARE_LOG_PATH, 10).await;
            s.error = Some(format!("timeout waiting for share readiness (120s): {detail}"));
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
