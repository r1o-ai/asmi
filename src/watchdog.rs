//! Process watchdog — detect stuck/zombie inference processes and GPU Lock.
//!
//! Runs a periodic check loop (default 5s) that:
//! 1. Inspects each inference process (framework != Unknown) from the node snapshot
//! 2. Checks port reachability via fast TCP connect
//! 3. Tracks stuck processes over time (port unreachable for extended periods)
//! 4. Detects GPU Lock (high CPU + low GPU + port dead for >15s)
//! 5. Builds a `WatchdogReport` aggregating all signals
//!
//! Thread-safe via interior mutability (Arc<RwLock<>>), all methods take `&self`.

use asmi_core::{
    GpuLockSeverity, GpuLockStatus, NodeSnapshot, PeerHeartbeatStatus, ProcessFramework,
    ProcessInfo, WatchdogReport, WatchdogVerdict, WatchedProcess,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Watchdog configuration. Can be loaded from `~/.r1o/watchdog.toml` or use defaults.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// How often to run the check loop (seconds).
    pub check_interval_secs: u64,
    /// CPU threshold above which a process is suspicious (percent).
    pub cpu_lock_threshold: f64,
    /// GPU threshold below which the GPU is considered idle (percent).
    pub gpu_idle_threshold: f64,
    /// Seconds of sustained GPU Lock signature before attempting kill.
    pub auto_kill_delay_secs: u64,
    /// Whether to automatically kill stuck processes.
    pub auto_kill_enabled: bool,
    /// TCP connect timeout for port reachability checks (milliseconds).
    pub port_check_timeout_ms: u64,
    /// Seconds before a port-unreachable process is considered stuck.
    pub stuck_threshold_secs: u64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 5,
            cpu_lock_threshold: 95.0,
            gpu_idle_threshold: 5.0,
            auto_kill_delay_secs: 30,
            auto_kill_enabled: true,
            port_check_timeout_ms: 500,
            stuck_threshold_secs: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------

/// Process watchdog that monitors inference processes and detects GPU Lock.
pub struct Watchdog {
    config: WatchdogConfig,
    report: Arc<RwLock<WatchdogReport>>,
    /// Reference to the node snapshot (written by the metrics collection loop).
    snapshot: Arc<RwLock<Option<NodeSnapshot>>>,
    /// Reference to peer heartbeat status (from PeerHeartbeat in serve.rs).
    peer_heartbeat: Arc<crate::serve::PeerHeartbeat>,
    /// Mutable state protected by a Mutex (cancel token + task handle).
    state: tokio::sync::Mutex<WatchdogState>,
}

struct WatchdogState {
    cancel: Option<CancellationToken>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl Watchdog {
    /// Create a new watchdog (not yet running).
    pub fn new(
        config: WatchdogConfig,
        snapshot: Arc<RwLock<Option<NodeSnapshot>>>,
        peer_heartbeat: Arc<crate::serve::PeerHeartbeat>,
    ) -> Self {
        let default_report = WatchdogReport {
            processes: vec![],
            gpu_lock: GpuLockStatus {
                detected: false,
                suspect_pids: vec![],
                since: None,
                severity: GpuLockSeverity::None,
            },
            peer_heartbeat: PeerHeartbeatStatus {
                active: false,
                peers: vec![],
                session_start: None,
            },
            last_check: chrono::Utc::now().to_rfc3339(),
        };

        Self {
            config,
            report: Arc::new(RwLock::new(default_report)),
            snapshot,
            peer_heartbeat,
            state: tokio::sync::Mutex::new(WatchdogState {
                cancel: None,
                handle: None,
            }),
        }
    }

    /// Start the watchdog loop. Idempotent — stops existing loop first.
    pub async fn start(&self) {
        self.stop().await;

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let config = self.config.clone();
        let report = self.report.clone();
        let snapshot = self.snapshot.clone();
        let peer_heartbeat = self.peer_heartbeat.clone();

        let handle = tokio::spawn(async move {
            watchdog_loop(cancel_clone, config, report, snapshot, peer_heartbeat).await;
        });

        let mut st = self.state.lock().await;
        st.cancel = Some(cancel);
        st.handle = Some(handle);
    }

    /// Stop the watchdog loop.
    pub async fn stop(&self) {
        let mut st = self.state.lock().await;
        if let Some(cancel) = st.cancel.take() {
            cancel.cancel();
        }
        if let Some(handle) = st.handle.take() {
            handle.abort();
        }
    }

    /// Get the current watchdog report.
    pub async fn report(&self) -> WatchdogReport {
        self.report.read().await.clone()
    }

    /// Get just the GPU Lock status.
    pub async fn gpu_lock_status(&self) -> GpuLockStatus {
        self.report.read().await.gpu_lock.clone()
    }
}

// ---------------------------------------------------------------------------
// Main watchdog loop
// ---------------------------------------------------------------------------

async fn watchdog_loop(
    cancel: CancellationToken,
    config: WatchdogConfig,
    report: Arc<RwLock<WatchdogReport>>,
    snapshot: Arc<RwLock<Option<NodeSnapshot>>>,
    peer_heartbeat: Arc<crate::serve::PeerHeartbeat>,
) {
    // Track when each process was first seen as stuck (pid → first_seen)
    let mut stuck_since: HashMap<u32, Instant> = HashMap::new();
    // Track when GPU Lock signature was first seen (suspect_pids → first_seen)
    let mut gpu_lock_first_seen: Option<(Vec<u32>, Instant)> = None;
    // Track PIDs we've already tried to kill (to escalate to Unrecoverable)
    let mut kill_attempted: HashMap<u32, Instant> = HashMap::new();

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = tokio::time::sleep(Duration::from_secs(config.check_interval_secs)) => {
                let snap = snapshot.read().await;
                let Some(ref snap) = *snap else {
                    continue;
                };

                let node_gpu_percent = snap.gpu_percent;
                let processes = &snap.processes;

                // Evaluate each inference process
                let mut watched = Vec::new();
                let mut inference_procs: Vec<&ProcessInfo> = Vec::new();

                for proc in processes {
                    if proc.framework == ProcessFramework::Unknown {
                        continue;
                    }

                    // Track inference processes for GPU Lock detection
                    inference_procs.push(proc);

                    // Check port reachability
                    let port_ok = if let Some(port) = proc.port {
                        Some(check_port_reachable(port, config.port_check_timeout_ms).await)
                    } else {
                        None
                    };

                    // Evaluate process verdict
                    let verdict = evaluate_process(
                        proc,
                        port_ok,
                        &mut stuck_since,
                        &kill_attempted,
                        &config,
                    );

                    // Auto-kill if stuck for too long
                    if let WatchdogVerdict::Stuck { duration_secs, .. } = &verdict {
                        if config.auto_kill_enabled
                            && *duration_secs >= config.stuck_threshold_secs
                            && !kill_attempted.contains_key(&proc.pid)
                        {
                            tracing::warn!(
                                pid = proc.pid,
                                duration_secs,
                                "watchdog: killing stuck process"
                            );
                            kill_process(proc.pid).await;
                            kill_attempted.insert(proc.pid, Instant::now());
                        }
                    }

                    watched.push(WatchedProcess {
                        pid: proc.pid,
                        framework: proc.framework.to_string(),
                        verdict,
                        since: chrono::Utc::now().to_rfc3339(),
                        port_reachable: port_ok,
                        cpu_percent: proc.cpu_percent,
                    });
                }

                // Clean up stuck_since for PIDs no longer present
                let active_pids: Vec<u32> = processes.iter().map(|p| p.pid).collect();
                stuck_since.retain(|pid, _| active_pids.contains(pid));
                kill_attempted.retain(|pid, _| active_pids.contains(pid));

                // Evaluate GPU Lock
                let gpu_lock = evaluate_gpu_lock(
                    &inference_procs,
                    node_gpu_percent,
                    &config,
                    &mut gpu_lock_first_seen,
                    &mut kill_attempted,
                ).await;

                // Get peer heartbeat status
                let hb_status = peer_heartbeat.status().await;

                // Update report
                let mut r = report.write().await;
                r.processes = watched;
                r.gpu_lock = gpu_lock;
                r.peer_heartbeat = hb_status;
                r.last_check = chrono::Utc::now().to_rfc3339();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Process verdict evaluation
// ---------------------------------------------------------------------------

fn evaluate_process(
    proc: &ProcessInfo,
    port_ok: Option<bool>,
    stuck_since: &mut HashMap<u32, Instant>,
    kill_attempted: &HashMap<u32, Instant>,
    _config: &WatchdogConfig,
) -> WatchdogVerdict {
    // If we already killed this process, report Killed
    if let Some(killed_at) = kill_attempted.get(&proc.pid) {
        return WatchdogVerdict::Killed {
            reason: "watchdog killed stuck process".to_string(),
            at: format!("{}s ago", killed_at.elapsed().as_secs()),
        };
    }

    // Port unreachable → track as stuck
    if let Some(false) = port_ok {
        let first_seen = stuck_since.entry(proc.pid).or_insert_with(Instant::now);
        let duration = first_seen.elapsed();

        if duration.as_secs() > 0 {
            return WatchdogVerdict::Stuck {
                reason: format!(
                    "port {} unreachable for {}s",
                    proc.port.unwrap_or(0),
                    duration.as_secs()
                ),
                duration_secs: duration.as_secs(),
            };
        }
    } else if port_ok == Some(true) {
        // Port is reachable — clear stuck tracking
        stuck_since.remove(&proc.pid);
    }

    // No port to check → consider healthy but degraded if high CPU
    if port_ok.is_none() && proc.cpu_percent > 95.0 {
        return WatchdogVerdict::Degraded {
            reason: format!("no port to verify, CPU at {:.1}%", proc.cpu_percent),
        };
    }

    WatchdogVerdict::Healthy
}

// ---------------------------------------------------------------------------
// GPU Lock detection (Task 3)
// ---------------------------------------------------------------------------

/// Evaluate GPU Lock condition across all inference processes.
///
/// GPU Lock signature (all must be true simultaneously):
/// 1. An MLX/inference process exists (framework != Unknown)
/// 2. That process has cpu_percent > cpu_lock_threshold (default 95%)
/// 3. Node-level gpu_percent < gpu_idle_threshold (default 5%)
/// 4. The process's HTTP port is unreachable
///
/// Severity escalation:
/// - Suspected: signature detected, <15s
/// - Confirmed (>15s): attempt SIGTERM → 3s → SIGKILL on suspect PIDs
/// - Unrecoverable (>30s): SIGKILL sent but process still alive, requires_reboot
async fn evaluate_gpu_lock(
    inference_procs: &[&ProcessInfo],
    node_gpu_percent: f64,
    config: &WatchdogConfig,
    gpu_lock_first_seen: &mut Option<(Vec<u32>, Instant)>,
    kill_attempted: &mut HashMap<u32, Instant>,
) -> GpuLockStatus {
    // Find suspect PIDs: high CPU + low GPU + port dead
    let mut suspect_pids: Vec<u32> = Vec::new();

    for proc in inference_procs {
        let cpu_high = proc.cpu_percent > config.cpu_lock_threshold;
        let gpu_low = node_gpu_percent < config.gpu_idle_threshold;

        // Check port reachability (quick TCP probe)
        let port_dead = if let Some(port) = proc.port {
            !check_port_reachable(port, config.port_check_timeout_ms).await
        } else {
            true // no port = can't verify = assume dead
        };

        if cpu_high && gpu_low && port_dead {
            suspect_pids.push(proc.pid);
        }
    }

    if suspect_pids.is_empty() {
        *gpu_lock_first_seen = None;
        return GpuLockStatus {
            detected: false,
            suspect_pids: vec![],
            since: None,
            severity: GpuLockSeverity::None,
        };
    }

    // Track when we first saw this GPU Lock signature
    let first_seen = match gpu_lock_first_seen {
        Some((prev_pids, instant)) if *prev_pids == suspect_pids => *instant,
        _ => {
            let now = Instant::now();
            *gpu_lock_first_seen = Some((suspect_pids.clone(), now));
            now
        }
    };

    let duration_secs = first_seen.elapsed().as_secs();

    let severity = if duration_secs > config.auto_kill_delay_secs {
        // >30s: check if we already tried to kill and process is still alive
        GpuLockSeverity::Unrecoverable
    } else if duration_secs > 15 {
        // >15s: Confirmed GPU Lock — attempt SIGTERM → 3s → SIGKILL
        if config.auto_kill_enabled {
            for &pid in &suspect_pids {
                if !kill_attempted.contains_key(&pid) {
                    tracing::error!(
                        pid,
                        duration_secs,
                        "GPU Lock confirmed — killing suspect process"
                    );
                    kill_process_with_escalation(pid).await;
                    kill_attempted.insert(pid, Instant::now());
                }
            }
        }
        GpuLockSeverity::Confirmed
    } else {
        // <15s: suspected, might recover
        tracing::warn!(
            pids = ?suspect_pids,
            duration_secs,
            cpu_threshold = config.cpu_lock_threshold,
            gpu_percent = node_gpu_percent,
            "GPU Lock suspected — monitoring"
        );
        GpuLockSeverity::Suspected
    };

    let since_str = chrono::Utc::now()
        .checked_sub_signed(chrono::Duration::seconds(duration_secs as i64))
        .map(|dt| dt.to_rfc3339());

    GpuLockStatus {
        detected: true,
        suspect_pids,
        since: since_str,
        severity,
    }
}

// ---------------------------------------------------------------------------
// Port reachability check
// ---------------------------------------------------------------------------

/// Check if a TCP port is reachable with a short timeout.
/// Uses raw TCP connect (not HTTP) for speed.
pub async fn check_port_reachable(port: u16, timeout_ms: u64) -> bool {
    let addr = format!("127.0.0.1:{}", port);
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        tokio::net::TcpStream::connect(&addr),
    )
    .await
    .map(|r| r.is_ok())
    .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Process killing
// ---------------------------------------------------------------------------

/// Kill a process with SIGTERM.
async fn kill_process(pid: u32) {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let nix_pid = Pid::from_raw(pid as i32);
    if let Err(e) = kill(nix_pid, Signal::SIGTERM) {
        tracing::warn!(pid, error = %e, "watchdog: failed to SIGTERM process");
    }
}

/// Kill a process with SIGTERM, wait 3s, then SIGKILL if still alive.
async fn kill_process_with_escalation(pid: u32) {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let nix_pid = Pid::from_raw(pid as i32);

    // SIGTERM first
    if let Err(e) = kill(nix_pid, Signal::SIGTERM) {
        tracing::warn!(pid, error = %e, "GPU Lock: failed to SIGTERM process");
        return;
    }
    tracing::info!(pid, "GPU Lock: sent SIGTERM, waiting 3s before SIGKILL");

    tokio::time::sleep(Duration::from_secs(3)).await;

    // Check if process is still alive
    if kill(nix_pid, None).is_ok() {
        // Process still alive — SIGKILL
        tracing::error!(pid, "GPU Lock: process survived SIGTERM, sending SIGKILL");
        if let Err(e) = kill(nix_pid, Signal::SIGKILL) {
            tracing::error!(pid, error = %e, "GPU Lock: SIGKILL failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use asmi_core::{ProcessFramework, ProcessInfo};

    fn make_process(pid: u32, framework: ProcessFramework, port: Option<u16>, cpu: f64) -> ProcessInfo {
        ProcessInfo {
            pid,
            framework,
            model: Some("test-model".to_string()),
            port,
            cpu_percent: cpu,
            mem_percent: 10.0,
            footprint_mb: None,
            distributed: None,
            server_models: vec![],
        }
    }

    fn make_snapshot(
        processes: Vec<ProcessInfo>,
        gpu_percent: f64,
        cpu_percent: f64,
    ) -> NodeSnapshot {
        NodeSnapshot {
            hostname: "test-node".to_string(),
            online: true,
            timestamp: chrono::Utc::now(),
            chip_model: None,
            serial_number: None,
            model_name: None,
            cpu_watts: 0.0,
            gpu_watts: 0.0,
            ane_watts: 0.0,
            power_source: None,
            cpu_percent,
            gpu_percent,
            ram_used_bytes: 0,
            ram_total_bytes: 0,
            ram_percent: 0.0,
            ram_app_bytes: 0,
            ram_cached_bytes: 0,
            cpu_clusters: vec![],
            gpu_frequency_mhz: None,
            disk_io: None,
            network: None,
            cpu_temp_c: None,
            gpu_temp_c: None,
            processes,
            top_tasks: vec![],
            rdma: None,
            interface_ips: Default::default(),
        }
    }

    #[test]
    fn test_default_config() {
        let config = WatchdogConfig::default();
        assert_eq!(config.check_interval_secs, 5);
        assert_eq!(config.cpu_lock_threshold, 95.0);
        assert_eq!(config.gpu_idle_threshold, 5.0);
        assert_eq!(config.auto_kill_delay_secs, 30);
        assert!(config.auto_kill_enabled);
    }

    #[test]
    fn test_evaluate_process_healthy() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 10.0);
        let mut stuck_since = HashMap::new();
        let kill_attempted = HashMap::new();
        let config = WatchdogConfig::default();

        let verdict = evaluate_process(&proc, Some(true), &mut stuck_since, &kill_attempted, &config);
        assert_eq!(verdict, WatchdogVerdict::Healthy);
        assert!(stuck_since.is_empty());
    }

    #[test]
    fn test_evaluate_process_port_unreachable_becomes_stuck() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 10.0);
        let mut stuck_since = HashMap::new();
        let kill_attempted = HashMap::new();
        let config = WatchdogConfig::default();

        // First check: port unreachable, insert into stuck_since
        let _verdict = evaluate_process(&proc, Some(false), &mut stuck_since, &kill_attempted, &config);
        // Duration is 0 initially (just inserted) — returns Healthy on the boundary check
        // since duration is 0 and we check > 0
        assert!(stuck_since.contains_key(&1234));

        // Simulate time passing by backdating the stuck_since entry
        stuck_since.insert(1234, Instant::now() - Duration::from_secs(10));

        let verdict = evaluate_process(&proc, Some(false), &mut stuck_since, &kill_attempted, &config);
        match verdict {
            WatchdogVerdict::Stuck { duration_secs, .. } => {
                assert!(duration_secs >= 10);
            }
            _ => panic!("expected Stuck verdict, got {:?}", verdict),
        }
    }

    #[test]
    fn test_evaluate_process_port_recovers_clears_stuck() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 10.0);
        let mut stuck_since = HashMap::new();
        let kill_attempted = HashMap::new();
        let config = WatchdogConfig::default();

        // Insert as stuck
        stuck_since.insert(1234, Instant::now() - Duration::from_secs(10));

        // Port recovers
        let verdict = evaluate_process(&proc, Some(true), &mut stuck_since, &kill_attempted, &config);
        assert_eq!(verdict, WatchdogVerdict::Healthy);
        assert!(!stuck_since.contains_key(&1234));
    }

    #[test]
    fn test_evaluate_process_killed_verdict() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 10.0);
        let mut stuck_since = HashMap::new();
        let mut kill_attempted = HashMap::new();
        let config = WatchdogConfig::default();

        kill_attempted.insert(1234, Instant::now() - Duration::from_secs(5));

        let verdict = evaluate_process(&proc, Some(false), &mut stuck_since, &kill_attempted, &config);
        match verdict {
            WatchdogVerdict::Killed { reason, .. } => {
                assert!(reason.contains("watchdog killed"));
            }
            _ => panic!("expected Killed verdict, got {:?}", verdict),
        }
    }

    #[test]
    fn test_evaluate_process_no_port_high_cpu_degraded() {
        let proc = make_process(1234, ProcessFramework::MlxLm, None, 98.0);
        let mut stuck_since = HashMap::new();
        let kill_attempted = HashMap::new();
        let config = WatchdogConfig::default();

        let verdict = evaluate_process(&proc, None, &mut stuck_since, &kill_attempted, &config);
        match verdict {
            WatchdogVerdict::Degraded { reason } => {
                assert!(reason.contains("no port"));
            }
            _ => panic!("expected Degraded verdict, got {:?}", verdict),
        }
    }

    #[tokio::test]
    async fn test_check_port_reachable_with_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        assert!(check_port_reachable(port, 500).await);

        drop(listener);
        // Small delay to let OS release the port
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!check_port_reachable(port, 200).await);
    }

    #[tokio::test]
    async fn test_check_port_unreachable() {
        // Port 1 should not be available to normal processes
        assert!(!check_port_reachable(1, 200).await);
    }

    #[tokio::test]
    async fn test_gpu_lock_not_detected_when_healthy() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 10.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let config = WatchdogConfig::default();
        let mut first_seen = None;
        let mut kill_attempted = HashMap::new();

        let status = evaluate_gpu_lock(&procs, 80.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(!status.detected);
        assert_eq!(status.severity, GpuLockSeverity::None);
        assert!(status.suspect_pids.is_empty());
    }

    #[tokio::test]
    async fn test_gpu_lock_not_triggered_with_healthy_gpu() {
        // High CPU but GPU is active — this is normal inference, NOT GPU Lock
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19080), 98.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let config = WatchdogConfig::default();
        let mut first_seen = None;
        let mut kill_attempted = HashMap::new();

        // GPU at 90% = healthy, should NOT trigger
        let status = evaluate_gpu_lock(&procs, 90.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(!status.detected);
        assert_eq!(status.severity, GpuLockSeverity::None);
    }

    #[tokio::test]
    async fn test_gpu_lock_suspected_with_dead_port() {
        // High CPU + low GPU + port dead = GPU Lock signature
        // Use a port that's definitely not listening
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19999), 98.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let config = WatchdogConfig::default();
        let mut first_seen = None;
        let mut kill_attempted = HashMap::new();

        let status = evaluate_gpu_lock(&procs, 2.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(status.detected);
        assert_eq!(status.severity, GpuLockSeverity::Suspected);
        assert_eq!(status.suspect_pids, vec![1234]);
    }

    #[tokio::test]
    async fn test_gpu_lock_escalates_to_confirmed() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19999), 98.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let mut config = WatchdogConfig::default();
        config.auto_kill_enabled = false; // don't actually kill in tests
        let mut kill_attempted = HashMap::new();

        // Simulate >15s by backdating the first_seen
        let mut first_seen = Some((vec![1234u32], Instant::now() - Duration::from_secs(20)));

        let status = evaluate_gpu_lock(&procs, 2.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(status.detected);
        assert_eq!(status.severity, GpuLockSeverity::Confirmed);
    }

    #[tokio::test]
    async fn test_gpu_lock_escalates_to_unrecoverable() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19999), 98.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let mut config = WatchdogConfig::default();
        config.auto_kill_enabled = false;
        let mut kill_attempted = HashMap::new();

        // Simulate >30s (auto_kill_delay_secs default)
        let mut first_seen = Some((vec![1234u32], Instant::now() - Duration::from_secs(35)));

        let status = evaluate_gpu_lock(&procs, 2.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(status.detected);
        assert_eq!(status.severity, GpuLockSeverity::Unrecoverable);
    }

    #[tokio::test]
    async fn test_gpu_lock_clears_when_recovered() {
        let proc = make_process(1234, ProcessFramework::MlxLm, Some(19999), 10.0);
        let procs: Vec<&ProcessInfo> = vec![&proc];
        let config = WatchdogConfig::default();
        let mut kill_attempted = HashMap::new();

        // Was previously detected
        let mut first_seen = Some((vec![1234u32], Instant::now() - Duration::from_secs(10)));

        // CPU dropped below threshold → not suspected
        let status = evaluate_gpu_lock(&procs, 2.0, &config, &mut first_seen, &mut kill_attempted).await;
        assert!(!status.detected);
        assert_eq!(status.severity, GpuLockSeverity::None);
        assert!(first_seen.is_none());
    }

    #[tokio::test]
    async fn test_watchdog_start_stop() {
        let snapshot: Arc<RwLock<Option<NodeSnapshot>>> = Arc::new(RwLock::new(None));
        let peer_hb = Arc::new(crate::serve::PeerHeartbeat::new());
        let config = WatchdogConfig {
            check_interval_secs: 1,
            ..Default::default()
        };
        let watchdog = Watchdog::new(config, snapshot, peer_hb);

        watchdog.start().await;

        // Let it run a couple cycles
        tokio::time::sleep(Duration::from_secs(2)).await;

        let report = watchdog.report().await;
        assert!(report.processes.is_empty()); // no snapshot = no processes
        assert!(!report.gpu_lock.detected);

        watchdog.stop().await;
    }

    #[tokio::test]
    async fn test_watchdog_with_snapshot() {
        let proc = make_process(1234, ProcessFramework::MlxLm, None, 10.0);
        let snap = make_snapshot(vec![proc], 50.0, 10.0);
        let snapshot: Arc<RwLock<Option<NodeSnapshot>>> = Arc::new(RwLock::new(Some(snap)));
        let peer_hb = Arc::new(crate::serve::PeerHeartbeat::new());
        let config = WatchdogConfig {
            check_interval_secs: 1,
            ..Default::default()
        };
        let watchdog = Watchdog::new(config, snapshot, peer_hb);

        watchdog.start().await;
        tokio::time::sleep(Duration::from_secs(2)).await;

        let report = watchdog.report().await;
        // Should have found the inference process
        assert_eq!(report.processes.len(), 1);
        assert_eq!(report.processes[0].pid, 1234);

        watchdog.stop().await;
    }

    #[tokio::test]
    async fn test_watchdog_unknown_framework_ignored() {
        let proc = make_process(5678, ProcessFramework::Unknown, Some(9999), 50.0);
        let snap = make_snapshot(vec![proc], 50.0, 50.0);
        let snapshot: Arc<RwLock<Option<NodeSnapshot>>> = Arc::new(RwLock::new(Some(snap)));
        let peer_hb = Arc::new(crate::serve::PeerHeartbeat::new());
        let config = WatchdogConfig {
            check_interval_secs: 1,
            ..Default::default()
        };
        let watchdog = Watchdog::new(config, snapshot, peer_hb);

        watchdog.start().await;
        tokio::time::sleep(Duration::from_secs(2)).await;

        let report = watchdog.report().await;
        // Unknown framework processes should not be watched
        assert!(report.processes.is_empty());

        watchdog.stop().await;
    }
}
