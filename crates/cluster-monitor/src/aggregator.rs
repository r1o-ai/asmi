//! Cluster state aggregation — owns per-node snapshots, history ring buffers,
//! and cluster-wide aggregates. This is the single source of truth for all
//! cluster monitoring data.

use crate::types::*;
use std::collections::{HashMap, HashSet};

/// Strip `-N` numeric suffixes from `hostname` recursively, returning the
/// first prefix that matches a known cluster node in `real` (case-insensitive).
/// If the input itself is already a known canonical node, return it unchanged.
/// If no parent in the strip chain is in `real`, return the input lowered.
///
/// Handles mDNS rename cascades (`hub` → `hub-2` → `hub-2-3` ...) without
/// collapsing legitimate sibling names (e.g. `worker` + `worker-2` as
/// distinct registered nodes).
pub fn canonicalize_hostname(hostname: &str, real: &HashSet<String>) -> String {
    let lowered = hostname.to_lowercase();
    if real.contains(&lowered) {
        return lowered;
    }
    let mut current = lowered;
    for _ in 0..6 {
        let Some(idx) = current.rfind('-') else { return current; };
        let suffix = &current[idx + 1..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            return current;
        }
        let parent = current[..idx].to_string();
        if real.contains(&parent) {
            return parent;
        }
        current = parent;
    }
    current
}

/// Full cluster state — latest snapshots, per-node histories, and cluster-wide
/// aggregates. Updated by the [`ClusterMonitor`](crate::ClusterMonitor) polling
/// loops and read by TUI / r1o-top for rendering.
#[derive(Debug, Clone)]
pub struct ClusterState {
    /// Latest snapshot per node hostname.
    pub snapshots: HashMap<String, NodeSnapshot>,
    /// Per-node metrics history for sparkline rendering.
    pub histories: HashMap<String, MetricsHistory>,
    /// Cluster-wide aggregate metrics.
    pub aggregates: ClusterAggregates,
    /// Cluster-wide metrics history (for aggregate sparklines).
    pub cluster_history: MetricsHistory,
    /// Last scan results (hardware probes).
    pub scan_results: Vec<ScanResult>,
    /// Total known nodes (including offline). Set by scan.
    pub total_nodes: usize,
    /// Ring buffer capacity for histories.
    history_capacity: usize,
}

impl ClusterState {
    /// Create a new empty cluster state with the given history capacity.
    pub fn new(history_capacity: usize) -> Self {
        Self {
            snapshots: HashMap::new(),
            histories: HashMap::new(),
            aggregates: ClusterAggregates::default(),
            cluster_history: MetricsHistory::new(history_capacity),
            scan_results: Vec::new(),
            total_nodes: 0,
            history_capacity,
        }
    }

    /// Update with a new snapshot for a node. Automatically pushes to
    /// per-node history, recalculates cluster aggregates, and pushes to
    /// cluster-wide history.
    pub fn update_node(&mut self, snapshot: NodeSnapshot) {
        let hostname = snapshot.hostname.clone();

        // Push to per-node history ring buffer
        let history = self
            .histories
            .entry(hostname.clone())
            .or_insert_with(|| MetricsHistory::new(self.history_capacity));
        history.push(
            snapshot.cpu_percent,
            snapshot.gpu_percent,
            snapshot.ram_percent,
            snapshot.total_watts(),
        );

        self.snapshots.insert(hostname, snapshot);
        self.recalculate();
    }

    /// Batch-update multiple nodes at once (avoids N recalculations).
    pub fn update_nodes(&mut self, snapshots: Vec<NodeSnapshot>) {
        for snapshot in snapshots {
            let hostname = snapshot.hostname.clone();

            let history = self
                .histories
                .entry(hostname.clone())
                .or_insert_with(|| MetricsHistory::new(self.history_capacity));
            history.push(
                snapshot.cpu_percent,
                snapshot.gpu_percent,
                snapshot.ram_percent,
                snapshot.total_watts(),
            );

            self.snapshots.insert(hostname, snapshot);
        }
        self.recalculate();
    }

    /// Update scan results and total node count.
    /// Deduplicates by hostname — keeps the first (most complete) result
    /// for each resolved hostname.
    pub fn update_scan(&mut self, results: Vec<ScanResult>) {
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::with_capacity(results.len());
        for r in results {
            let key = r.hostname.to_lowercase();
            if seen.insert(key) {
                deduped.push(r);
            }
        }
        // Only count SSH-reachable nodes — discovery candidates that can't
        // be reached aren't real cluster nodes and shouldn't inflate the total.
        self.total_nodes = deduped.iter().filter(|r| r.ssh_ok).count();
        self.scan_results = deduped;
    }

    /// Merge new scan results into existing ones without replacing.
    /// New SSH-reachable nodes are added; existing entries are updated.
    pub fn merge_scan(&mut self, results: Vec<ScanResult>) {
        let mut by_host: std::collections::HashMap<String, ScanResult> = self
            .scan_results
            .drain(..)
            .map(|r| (r.hostname.to_lowercase(), r))
            .collect();

        for r in results {
            let key = r.hostname.to_lowercase();
            // Prefer the newer result if it reached the node
            if r.ssh_ok || !by_host.contains_key(&key) {
                by_host.insert(key, r);
            }
        }

        self.scan_results = by_host.into_values().collect();
        self.total_nodes = self.scan_results.iter().filter(|r| r.ssh_ok).count();
    }

    /// Mark a node as offline (sets online=false in its snapshot).
    pub fn mark_offline(&mut self, hostname: &str) {
        if let Some(snap) = self.snapshots.get_mut(hostname) {
            snap.online = false;
        }
        self.recalculate();
    }

    /// Get node hostnames in sorted order (for stable rendering).
    pub fn sorted_hostnames(&self) -> Vec<String> {
        let mut names: Vec<String> = self.snapshots.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of currently online nodes.
    pub fn online_count(&self) -> usize {
        self.snapshots.values().filter(|s| s.online).count()
    }

    /// Recalculate cluster-wide aggregates from current snapshots and push
    /// to cluster history.
    fn recalculate(&mut self) {
        let snapshots: Vec<NodeSnapshot> = self.snapshots.values().cloned().collect();
        let total = if self.total_nodes > 0 {
            self.total_nodes
        } else {
            snapshots.len()
        };
        self.aggregates = ClusterAggregates::from_snapshots(&snapshots, total);

        let mem_pct = if self.aggregates.total_ram_total_bytes > 0 {
            (self.aggregates.total_ram_used_bytes as f64
                / self.aggregates.total_ram_total_bytes as f64)
                * 100.0
        } else {
            0.0
        };
        self.cluster_history.push(
            self.aggregates.cpu_avg_percent,
            self.aggregates.gpu_avg_percent,
            mem_pct,
            self.aggregates.total_watts,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn mock_snapshot(hostname: &str, online: bool) -> NodeSnapshot {
        NodeSnapshot {
            hostname: hostname.to_string(),
            online,
            timestamp: Utc::now(),
            chip_model: None,
            serial_number: None,
            model_name: None,
            cpu_watts: 5000.0,  // 5W
            gpu_watts: 8000.0,  // 8W
            ane_watts: 100.0,   // 0.1W
            power_source: None,
            cpu_percent: 25.0,
            gpu_percent: 60.0,
            ram_used_bytes: 128 * 1024 * 1024 * 1024, // 128 GiB
            ram_total_bytes: 512 * 1024 * 1024 * 1024, // 512 GiB
            ram_percent: 25.0,
            ram_app_bytes: 128 * 1024 * 1024 * 1024,
            ram_cached_bytes: 0,
            cpu_clusters: vec![],
            gpu_frequency_mhz: None,
            disk_io: None,
            network: None,
            cpu_temp_c: Some(42.0),
            gpu_temp_c: None,
            processes: vec![],
            top_tasks: vec![],
            rdma: None,
            interface_ips: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn test_empty_state() {
        let state = ClusterState::new(60);
        assert_eq!(state.online_count(), 0);
        assert_eq!(state.aggregates.nodes_online, 0);
        assert!(state.sorted_hostnames().is_empty());
        assert!(state.cluster_history.is_empty());
    }

    #[test]
    fn test_update_single_node() {
        let mut state = ClusterState::new(60);
        state.update_node(mock_snapshot("m3u2", true));

        assert_eq!(state.online_count(), 1);
        assert_eq!(state.aggregates.nodes_online, 1);
        assert_eq!(state.sorted_hostnames(), vec!["m3u2"]);
        assert_eq!(state.histories["m3u2"].len(), 1);
        assert_eq!(state.cluster_history.len(), 1);
    }

    #[test]
    fn test_update_multiple_nodes() {
        let mut state = ClusterState::new(60);
        state.update_node(mock_snapshot("m3u2", true));
        state.update_node(mock_snapshot("m3u1", true));

        assert_eq!(state.online_count(), 2);
        assert_eq!(state.aggregates.nodes_online, 2);
        // total_watts: each node = (5000+8000+100)/1000 = 13.1W, two = 26.2W
        assert!((state.aggregates.total_watts - 26.2).abs() < 0.1);
        assert_eq!(state.sorted_hostnames(), vec!["m3u1", "m3u2"]);
    }

    #[test]
    fn test_batch_update() {
        let mut state = ClusterState::new(60);
        state.update_nodes(vec![
            mock_snapshot("m3u2", true),
            mock_snapshot("m3u1", true),
            mock_snapshot("m3u3", true),
        ]);

        assert_eq!(state.online_count(), 3);
        // Batch only triggers one recalculate, but each node gets a history push
        assert_eq!(state.histories.len(), 3);
        // cluster_history gets one push from recalculate
        assert_eq!(state.cluster_history.len(), 1);
    }

    #[test]
    fn test_mark_offline() {
        let mut state = ClusterState::new(60);
        state.update_node(mock_snapshot("m3u2", true));
        state.update_node(mock_snapshot("m3u1", true));

        state.mark_offline("m3u1");
        assert_eq!(state.online_count(), 1);
        assert_eq!(state.aggregates.nodes_online, 1);
    }

    #[test]
    fn test_history_accumulates() {
        let mut state = ClusterState::new(60);
        for _ in 0..5 {
            state.update_node(mock_snapshot("m3u2", true));
        }

        assert_eq!(state.histories["m3u2"].len(), 5);
        // cluster_history gets one push per update_node call
        assert_eq!(state.cluster_history.len(), 5);
    }

    #[test]
    fn test_history_capacity() {
        let mut state = ClusterState::new(3); // tiny capacity
        for _ in 0..10 {
            state.update_node(mock_snapshot("m3u2", true));
        }

        // Ring buffer should cap at 3
        assert_eq!(state.histories["m3u2"].len(), 3);
        assert_eq!(state.cluster_history.len(), 3);
    }

    #[test]
    fn test_update_scan() {
        let mut state = ClusterState::new(60);
        state.update_scan(vec![
            ScanResult {
                hostname: "m3u2".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: Some(80),
                rdma: None,
                mlx_servers: vec![],
                latency_ms: Some(0.5),
                link_speed: None,
            },
            ScanResult {
                hostname: "m3u1".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: Some(80),
                rdma: None,
                mlx_servers: vec![],
                latency_ms: Some(0.4),
                link_speed: None,
            },
        ]);

        assert_eq!(state.total_nodes, 2);
        assert_eq!(state.scan_results.len(), 2);
    }

    #[test]
    fn test_total_nodes_excludes_unreachable() {
        let mut state = ClusterState::new(60);
        state.update_scan(vec![
            ScanResult {
                hostname: "m3u1".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
            ScanResult {
                hostname: "some-iot-device".to_string(),
                reachable: true,
                ssh_ok: false,
                chip: None,
                ram_gb: None,
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
            ScanResult {
                hostname: "unreachable-host".to_string(),
                reachable: false,
                ssh_ok: false,
                chip: None,
                ram_gb: None,
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
        ]);

        // Only 1 SSH-reachable node counts toward total
        assert_eq!(state.total_nodes, 1);
        // All 3 scan results are preserved for reference
        assert_eq!(state.scan_results.len(), 3);
    }

    #[test]
    fn test_merge_scan() {
        let mut state = ClusterState::new(60);
        // Phase 1: seed scan finds 2 nodes
        state.update_scan(vec![
            ScanResult {
                hostname: "m3u1".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
            ScanResult {
                hostname: "m3u2".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
        ]);
        assert_eq!(state.total_nodes, 2);

        // Phase 2: full scan finds same 2 + 1 new node + 5 unreachable
        state.merge_scan(vec![
            ScanResult {
                hostname: "m3u1".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M3 Ultra".to_string()),
                ram_gb: Some(512),
                gpu_cores: Some(80), // updated info
                rdma: None,
                mlx_servers: vec![],
                latency_ms: Some(0.3),
                link_speed: None,
            },
            ScanResult {
                hostname: "mse1".to_string(),
                reachable: true,
                ssh_ok: true,
                chip: Some("Apple M4 Max".to_string()),
                ram_gb: Some(128),
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
            ScanResult {
                hostname: "iot-lamp".to_string(),
                reachable: true,
                ssh_ok: false,
                chip: None,
                ram_gb: None,
                gpu_cores: None,
                rdma: None,
                mlx_servers: vec![],
                latency_ms: None,
                link_speed: None,
            },
        ]);

        // total_nodes = 3 (m3u1 + m3u2 + mse1), not counting iot-lamp
        assert_eq!(state.total_nodes, 3);
        // All unique entries preserved
        assert_eq!(state.scan_results.len(), 4);
    }

    #[test]
    fn test_offline_node_excluded_from_aggregates() {
        let mut state = ClusterState::new(60);
        state.update_node(mock_snapshot("m3u2", true));
        state.update_node(mock_snapshot("m3u1", false)); // offline

        assert_eq!(state.aggregates.nodes_online, 1);
        // Only online node contributes to total_watts
        assert!((state.aggregates.total_watts - 13.1).abs() < 0.1);
    }

    // ── canonicalize_hostname ─────────────────────────────────────────────

    fn real_set(nodes: &[&str]) -> HashSet<String> {
        nodes.iter().map(|n| n.to_lowercase()).collect()
    }

    #[test]
    fn canonicalize_already_canonical_returns_unchanged() {
        assert_eq!(canonicalize_hostname("hub", &real_set(&["hub", "m3u2"])), "hub");
    }

    #[test]
    fn canonicalize_strips_rename_to_canonical() {
        assert_eq!(canonicalize_hostname("hub-2", &real_set(&["hub"])), "hub");
        assert_eq!(canonicalize_hostname("m3u4-2237", &real_set(&["m3u4"])), "m3u4");
    }

    #[test]
    fn canonicalize_chained_renames() {
        assert_eq!(canonicalize_hostname("hub-2-3", &real_set(&["hub"])), "hub");
    }

    #[test]
    fn canonicalize_case_insensitive() {
        assert_eq!(canonicalize_hostname("Hub-2", &real_set(&["hub"])), "hub");
    }

    #[test]
    fn canonicalize_preserves_legit_sibling() {
        // If cluster.json registers BOTH `worker` and `worker-2` as distinct
        // canonical nodes, the input `worker-2` must NOT collapse to `worker`.
        assert_eq!(
            canonicalize_hostname("worker-2", &real_set(&["worker", "worker-2"])),
            "worker-2"
        );
    }

    #[test]
    fn canonicalize_non_numeric_suffix_preserved() {
        assert_eq!(canonicalize_hostname("hub-spare", &real_set(&["hub"])), "hub-spare");
    }

    #[test]
    fn canonicalize_no_parent_in_real_returns_stripped() {
        // No `foo` in real; strips once to `foo`, then no more hyphen → returns `foo`.
        assert_eq!(canonicalize_hostname("foo-1", &real_set(&["hub"])), "foo");
    }

    #[test]
    fn canonicalize_empty_real_set_terminates() {
        // Edge case: NodeMap empty. Strip still terminates without panic.
        assert_eq!(canonicalize_hostname("hub-2", &real_set(&[])), "hub");
    }

    #[test]
    fn canonicalize_empty_string_returns_empty() {
        assert_eq!(canonicalize_hostname("", &real_set(&["hub"])), "");
    }

    #[test]
    fn canonicalize_deep_cascade_capped_at_six() {
        // Attacker DoS test: 8-level cascade should not panic or loop forever.
        // After 6 iters, returns whatever's left — NOT "hub", because cap exits
        // before reaching the canonical parent.
        let result = canonicalize_hostname("hub-1-2-3-4-5-6-7-8", &real_set(&["hub"]));
        // Verify: terminates (no panic), returns something (not the original).
        assert_ne!(result, "hub-1-2-3-4-5-6-7-8");
        // After 6 strips from the right: hub-1-2-3-4-5-6-7-8 → hub-1-2-3-4-5-6-7
        // → hub-1-2-3-4-5-6 → hub-1-2-3-4-5 → hub-1-2-3-4 → hub-1-2-3 → hub-1-2
        // (6 iters; current="hub-1-2", loop exits). Function returns "hub-1-2".
        // This is "hub-1-2" — NOT "hub". Confirms the cap leaves uncanonical names.
        assert_eq!(result, "hub-1-2");
    }
}
