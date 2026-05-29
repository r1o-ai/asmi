//! Background cluster monitor — polls metrics and runs scans on intervals.
//!
//! The monitor maintains a shared [`ClusterState`] behind `Arc<RwLock<>>`.
//! Consumers (TUI, r1o-top) read from this state on their render tick or
//! subscribe to the epoch counter for change notifications.

use crate::aggregator::ClusterState;
use crate::collector::collect_node_metrics;
use crate::config::{ClusterConfig, NodeMap};
use crate::scanner::{scan_cluster, scan_seeds};
use crate::types::{ClusterEvent, EventSink};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Background cluster monitor with shared state.
///
/// # Usage
///
/// ```ignore
/// let node_map = NodeMap::load();
/// let mut monitor = ClusterMonitor::new(ClusterConfig::default(), node_map);
/// let state = monitor.state(); // Arc<RwLock<ClusterState>>
/// monitor.start();
///
/// // Read state from any task
/// let s = state.read().await;
/// println!("{} nodes online", s.online_count());
///
/// // Shutdown
/// monitor.stop();
/// ```
pub struct ClusterMonitor {
    config: ClusterConfig,
    state: Arc<RwLock<ClusterState>>,
    node_map: Arc<RwLock<NodeMap>>,
    epoch: Arc<tokio::sync::watch::Sender<u64>>,
    shutdown_tx: Option<tokio::sync::watch::Sender<bool>>,
    events_tx: tokio::sync::broadcast::Sender<ClusterEvent>,
}

impl ClusterMonitor {
    /// Create a new monitor (not yet started).
    pub fn new(config: ClusterConfig, node_map: NodeMap) -> Self {
        // Hand the canonical node names to ClusterState so that update_node
        // collapses mDNS rename variants (hub-2 → hub) at insert time. See
        // aggregator::canonicalize_hostname for the rules.
        let state = Arc::new(RwLock::new(ClusterState::with_canonical(
            config.history_capacity,
            node_map.nodes.clone(),
        )));
        let (epoch_tx, _) = tokio::sync::watch::channel(0u64);
        let (events_tx, _) = tokio::sync::broadcast::channel(256);
        Self {
            config,
            state,
            node_map: Arc::new(RwLock::new(node_map)),
            epoch: Arc::new(epoch_tx),
            shutdown_tx: None,
            events_tx,
        }
    }

    /// Subscribe to real-time cluster events (discovery, probing, metrics).
    pub fn events(&self) -> tokio::sync::broadcast::Receiver<ClusterEvent> {
        self.events_tx.subscribe()
    }

    /// Get an EventSink for emitting events into this monitor's channel.
    pub fn event_sink(&self) -> EventSink {
        EventSink::new(self.events_tx.clone())
    }

    /// Get a handle to the shared cluster state.
    pub fn state(&self) -> Arc<RwLock<ClusterState>> {
        Arc::clone(&self.state)
    }

    /// Get a handle to the shared node map (for TUI alias display / manual merge).
    pub fn node_map(&self) -> Arc<RwLock<NodeMap>> {
        Arc::clone(&self.node_map)
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

        // Prefer scutil LocalHostName (Bonjour identity, matches Tailscale names)
        // over whoami (which returns the Unix hostname — often a stale default).
        let local_hostname = std::process::Command::new("scutil")
            .args(["--get", "LocalHostName"])
            .output()
            .ok()
            .and_then(|o| {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_lowercase();
                if s.is_empty() { None } else { Some(s) }
            })
            .unwrap_or_else(|| {
                whoami::fallible::hostname()
                    .unwrap_or_else(|_| "localhost".to_string())
                    .split('.')
                    .next()
                    .unwrap_or("localhost")
                    .to_string()
            });

        let events = self.event_sink();

        // --- Hierarchical scan loop ---
        // Phase 1: Fast probe of seed/registry hosts (instant results)
        // Phase 2: Full discovery scan (finds additional nodes)
        // Phase 3+: Periodic re-scan on interval
        {
            let config = self.config.clone();
            let state = Arc::clone(&self.state);
            let epoch = Arc::clone(&self.epoch);
            let events = events.clone();
            let mut rx = shutdown_rx.clone();

            tokio::spawn(async move {
                // Phase 1: Probe seed hosts immediately (no discovery overhead)
                if !config.seed_hosts.is_empty() {
                    run_seed_scan(&config, &state, &epoch, &events).await;
                }

                // Phase 2: Full discovery scan (ARP, Tailscale, TB, etc.)
                run_scan(&config, &state, &epoch, &events).await;

                // Phase 3+: Periodic re-scan
                loop {
                    tokio::select! {
                        _ = rx.changed() => {
                            info!("scan loop shutting down");
                            break;
                        }
                        _ = tokio::time::sleep(config.scan_interval) => {
                            run_scan(&config, &state, &epoch, &events).await;
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
            let node_map = Arc::clone(&self.node_map);
            let events = events.clone();
            let mut rx = shutdown_rx;
            let local = local_hostname;

            tokio::spawn(async move {
                // Brief delay only if we need to wait for scan to discover
                // nodes. With seed hosts, start immediately.
                if config.seed_hosts.is_empty() {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }

                loop {
                    tokio::select! {
                        _ = rx.changed() => {
                            info!("metrics loop shutting down");
                            break;
                        }
                        _ = tokio::time::sleep(config.poll_interval) => {
                            poll_metrics(&config, &state, &epoch, &local, &events, &node_map).await;
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
/// Applies the NodeMap alias resolution to dedup the poll list.
async fn poll_metrics(
    config: &ClusterConfig,
    state: &Arc<RwLock<ClusterState>>,
    epoch: &Arc<tokio::sync::watch::Sender<u64>>,
    local_hostname: &str,
    events: &EventSink,
    node_map: &Arc<RwLock<NodeMap>>,
) {
    // Build raw hostname list from scan results or seeds
    let raw_hostnames: Vec<String> = {
        let s = state.read().await;
        if !s.scan_results.is_empty() {
            s.scan_results
                .iter()
                .filter(|r| r.ssh_ok)
                .map(|r| r.hostname.clone())
                .collect()
        } else if !config.seed_hosts.is_empty() {
            config.seed_hosts.clone()
        } else {
            s.sorted_hostnames()
        }
    };

    // Apply alias map to resolve and deduplicate
    let hostnames = {
        let nm = node_map.read().await;
        nm.resolve_dedup(&raw_hostnames)
    };

    if hostnames.is_empty() {
        debug!("no known nodes yet, skipping metrics poll");
        return;
    }

    debug!(count = hostnames.len(), "polling metrics from nodes");
    events.emit(ClusterEvent::MetricsPollStarted { count: hostnames.len() });

    let futs: Vec<_> = hostnames
        .iter()
        .map(|h| {
            let is_local = h == local_hostname;
            let config = config.clone();
            let hostname = h.clone();
            let events = events.clone();
            async move {
                let snap = collect_node_metrics(&hostname, &config, is_local).await;
                events.emit(ClusterEvent::MetricsReceived { hostname: hostname.clone() });
                snap
            }
        })
        .collect();

    let snapshots = futures::future::join_all(futs).await;

    {
        let mut s = state.write().await;
        s.update_nodes(snapshots);
    }

    epoch.send_modify(|v| *v += 1);
}

/// Fast probe of seed/registry hosts only — no discovery overhead.
/// Shows known nodes in the TUI immediately.
async fn run_seed_scan(
    config: &ClusterConfig,
    state: &Arc<RwLock<ClusterState>>,
    epoch: &Arc<tokio::sync::watch::Sender<u64>>,
    events: &EventSink,
) {
    info!("starting fast seed probe");
    let results = scan_seeds(config, events).await;
    let online = results.iter().filter(|r| r.ssh_ok).count();
    info!(online, total = results.len(), "seed probe complete");

    {
        let mut s = state.write().await;
        s.update_scan(results);
    }

    epoch.send_modify(|v| *v += 1);
}

/// Run a full cluster scan (discover + probe). Merges results into
/// existing state so seed-probed nodes aren't lost.
async fn run_scan(
    config: &ClusterConfig,
    state: &Arc<RwLock<ClusterState>>,
    epoch: &Arc<tokio::sync::watch::Sender<u64>>,
    events: &EventSink,
) {
    info!("starting cluster scan");
    let results = scan_cluster(config, events).await;
    info!(count = results.len(), "cluster scan complete");

    {
        let mut s = state.write().await;
        s.merge_scan(results);
    }

    epoch.send_modify(|v| *v += 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_monitor_creation() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config, NodeMap::default());
        assert!(!monitor.is_running());
    }

    #[test]
    fn test_monitor_state_shared() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config, NodeMap::default());
        let s1 = monitor.state();
        let s2 = monitor.state();
        // Both point to the same allocation
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_subscribe() {
        let config = ClusterConfig::default();
        let monitor = ClusterMonitor::new(config, NodeMap::default());
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
        let mut monitor = ClusterMonitor::new(config, NodeMap::default());
        monitor.start();
        assert!(monitor.is_running());

        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(300)).await;

        // State should have been updated (scan populates localhost)
        let state = monitor.state();
        let s = state.read().await;
        debug!("scan results: {:?}", s.scan_results.len());

        monitor.stop();
    }
}
