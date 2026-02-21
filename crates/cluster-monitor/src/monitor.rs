//! Background cluster monitor — polls metrics and runs scans on intervals.
//!
//! The monitor maintains a shared [`ClusterState`] behind `Arc<RwLock<>>`.
//! Consumers (TUI, r1o-top) read from this state on their render tick or
//! subscribe to the epoch counter for change notifications.

use crate::aggregator::ClusterState;
use crate::collector::collect_node_metrics;
use crate::config::ClusterConfig;
use crate::scanner::scan_cluster;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Background cluster monitor with shared state.
///
/// # Usage
///
/// ```ignore
/// let mut monitor = ClusterMonitor::new(ClusterConfig::default());
/// let state = monitor.state(); // Arc<RwLock<ClusterState>>
/// monitor.start();
///
/// // Read state from any task
/// let s = state.read().await;
/// println!("{} nodes online", s.online_count());
///
/// // Or subscribe to updates
/// let mut rx = monitor.subscribe();
/// while rx.changed().await.is_ok() {
///     let epoch = *rx.borrow();
///     println!("state updated, epoch={epoch}");
/// }
///
/// // Shutdown
/// monitor.stop();
/// ```
pub struct ClusterMonitor {
    config: ClusterConfig,
    state: Arc<RwLock<ClusterState>>,
    epoch: Arc<tokio::sync::watch::Sender<u64>>,
    shutdown_tx: Option<tokio::sync::watch::Sender<bool>>,
}

impl ClusterMonitor {
    /// Create a new monitor (not yet started).
    pub fn new(config: ClusterConfig) -> Self {
        let state = Arc::new(RwLock::new(ClusterState::new(config.history_capacity)));
        let (epoch_tx, _) = tokio::sync::watch::channel(0u64);
        Self {
            config,
            state,
            epoch: Arc::new(epoch_tx),
            shutdown_tx: None,
        }
    }

    /// Get a handle to the shared cluster state.
    pub fn state(&self) -> Arc<RwLock<ClusterState>> {
        Arc::clone(&self.state)
    }

    /// Subscribe to state change notifications. The receiver yields the
    /// epoch counter which increments on each update (metrics poll or scan).
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.epoch.subscribe()
    }

    /// Start the background polling loops. Returns immediately — work
    /// happens in spawned tasks.
    pub fn start(&mut self) {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        self.shutdown_tx = Some(shutdown_tx);

        let local_hostname = whoami::fallible::hostname()
            .unwrap_or_else(|_| "localhost".to_string());

        // --- Scan loop (runs first to populate initial node list) ---
        {
            let config = self.config.clone();
            let state = Arc::clone(&self.state);
            let epoch = Arc::clone(&self.epoch);
            let mut rx = shutdown_rx.clone();

            tokio::spawn(async move {
                // Run initial scan immediately
                run_scan(&config, &state, &epoch).await;

                loop {
                    tokio::select! {
                        _ = rx.changed() => {
                            info!("scan loop shutting down");
                            break;
                        }
                        _ = tokio::time::sleep(config.scan_interval) => {
                            run_scan(&config, &state, &epoch).await;
                        }
                    }
                }
            });
        }

        // --- Metrics polling loop ---
        {
            let config = self.config.clone();
            let state = Arc::clone(&self.state);
            let epoch = Arc::clone(&self.epoch);
            let mut rx = shutdown_rx;
            let local = local_hostname;

            tokio::spawn(async move {
                // Brief delay to let scan populate nodes first
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                loop {
                    tokio::select! {
                        _ = rx.changed() => {
                            info!("metrics loop shutting down");
                            break;
                        }
                        _ = tokio::time::sleep(config.poll_interval) => {
                            poll_metrics(&config, &state, &epoch, &local).await;
                        }
                    }
                }
            });
        }

        info!(
            poll_interval_ms = self.config.poll_interval.as_millis() as u64,
            scan_interval_s = self.config.scan_interval.as_secs(),
            "cluster monitor started"
        );
    }

    /// Stop all background loops gracefully.
    pub fn stop(&self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(true);
            info!("cluster monitor stop signal sent");
        }
    }

    /// Whether the monitor is currently running.
    pub fn is_running(&self) -> bool {
        self.shutdown_tx.is_some()
    }
}

impl Drop for ClusterMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Collect metrics from all known nodes in parallel.
async fn poll_metrics(
    config: &ClusterConfig,
    state: &Arc<RwLock<ClusterState>>,
    epoch: &Arc<tokio::sync::watch::Sender<u64>>,
    local_hostname: &str,
) {
    // Read current known hostnames — prefer scan results, fall back to
    // existing snapshots, then seed hosts (so first poll doesn't wait for scan)
    let hostnames: Vec<String> = {
        let s = state.read().await;
        if !s.scan_results.is_empty() {
            s.scan_results
                .iter()
                .filter(|r| r.ssh_ok)
                .map(|r| r.hostname.clone())
                .collect()
        } else if !s.snapshots.is_empty() {
            s.sorted_hostnames()
        } else {
            config.seed_hosts.clone()
        }
    };

    if hostnames.is_empty() {
        debug!("no known nodes yet, skipping metrics poll");
        return;
    }

    debug!(count = hostnames.len(), "polling metrics from nodes");

    // Collect from all nodes in parallel
    let futs: Vec<_> = hostnames
        .iter()
        .map(|h| {
            let is_local = h == local_hostname;
            let config = config.clone();
            let hostname = h.clone();
            async move { collect_node_metrics(&hostname, &config, is_local).await }
        })
        .collect();

    let snapshots = futures::future::join_all(futs).await;

    // Batch-update state (single write lock, single recalculate)
    {
        let mut s = state.write().await;
        s.update_nodes(snapshots);
    }

    // Notify subscribers
    let _ = epoch.send_modify(|v| *v += 1);
}

/// Run a full cluster scan (discover + probe).
async fn run_scan(
    config: &ClusterConfig,
    state: &Arc<RwLock<ClusterState>>,
    epoch: &Arc<tokio::sync::watch::Sender<u64>>,
) {
    info!("starting cluster scan");
    let results = scan_cluster(config).await;
    info!(count = results.len(), "cluster scan complete");

    {
        let mut s = state.write().await;
        s.update_scan(results);
    }

    let _ = epoch.send_modify(|v| *v += 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DiscoveryMethod;
    use std::time::Duration;

    #[test]
    fn test_monitor_creation() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config);
        assert!(!monitor.is_running());
    }

    #[test]
    fn test_monitor_state_shared() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config);
        let s1 = monitor.state();
        let s2 = monitor.state();
        // Both point to the same allocation
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_subscribe() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config);
        let rx = monitor.subscribe();
        assert_eq!(*rx.borrow(), 0);
    }

    #[tokio::test]
    async fn test_monitor_start_stop() {
        let config = ClusterConfig {
            // Use seed hosts to avoid real discovery
            seed_hosts: vec!["localhost".to_string()],
            discovery: vec![], // no discovery methods
            poll_interval: Duration::from_millis(100),
            scan_interval: Duration::from_secs(60),
            ..ClusterConfig::default()
        };
        let mut monitor = ClusterMonitor::new(config);
        monitor.start();
        assert!(monitor.is_running());

        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(300)).await;

        // State should have been updated (scan populates localhost)
        let state = monitor.state();
        let s = state.read().await;
        // scan_cluster with empty discovery + seed "localhost" should produce something
        debug!("scan results: {:?}", s.scan_results.len());

        monitor.stop();
    }
}
