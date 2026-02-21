//! Cluster state aggregation — owns per-node snapshots, history ring buffers,
//! and cluster-wide aggregates. This is the single source of truth for all
//! cluster monitoring data.

use crate::types::*;
use std::collections::HashMap;

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
    pub fn update_scan(&mut self, results: Vec<ScanResult>) {
        self.total_nodes = results.len();
        self.scan_results = results;
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
            cpu_watts: 5000.0,  // 5W
            gpu_watts: 8000.0,  // 8W
            ane_watts: 100.0,   // 0.1W
            cpu_percent: 25.0,
            gpu_percent: 60.0,
            ram_used_bytes: 128 * 1024 * 1024 * 1024, // 128 GiB
            ram_total_bytes: 512 * 1024 * 1024 * 1024, // 512 GiB
            ram_percent: 25.0,
            cpu_temp_c: Some(42.0),
            gpu_temp_c: None,
            processes: vec![],
            top_tasks: vec![],
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
            },
        ]);

        assert_eq!(state.total_nodes, 2);
        assert_eq!(state.scan_results.len(), 2);
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
}
