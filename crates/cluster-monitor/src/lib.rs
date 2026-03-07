//! Cluster monitor for Apple Silicon nodes.
//!
//! This crate provides:
//! - Type definitions for node snapshots, processes, RDMA status, and aggregates
//! - SSH execution layer (remote + local command execution)
//! - Metrics collector (parallel SSH + footprint enrichment)
//! - Cluster configuration with dynamic node discovery (zero hardcoded nodes)
//! - Dynamic node discovery and scanning via multiple methods
//! - Aggregated cluster state with per-node history ring buffers
//! - Background polling monitor with shared state
//!
//! # Architecture
//!
//! All monitoring is done via SSH + local shell commands. No agents are installed
//! on remote nodes. Data sources include:
//!
//! - `powermetrics` — CPU/GPU/ANE power draw, GPU residency, task energy
//! - `ps aux` — process listing for MLX servers, watchdogs
//! - `sysctl hw.memsize` / `vm_stat` — memory usage
//! - `footprint` — real process memory footprint
//! - `system_profiler SPThunderboltDataType` — Thunderbolt topology
//! - `ibstat` / RDMA tools — Thunderbolt RDMA mesh status
//! - `tailscale status --json` — Tailscale peer discovery
//! - `arp -an` — link-local neighbour discovery
//!
//! # Quick Start
//!
//! ```ignore
//! use asmi_core::{ClusterConfig, ClusterMonitor};
//!
//! let config = ClusterConfig::default()
//!     .with_seeds(vec!["m3u2".into(), "m3u1".into()]);
//! let mut monitor = ClusterMonitor::new(config);
//! let state = monitor.state();
//! monitor.start();
//!
//! // Read state from render loop
//! let s = state.read().await;
//! println!("{}/{} nodes online", s.online_count(), s.total_nodes);
//! ```

pub mod aggregator;
pub mod collector;
pub mod config;
pub mod ioreport;
pub mod monitor;
pub mod scanner;
pub mod ssh;
pub mod types;
pub mod models;
pub mod health;

/// Resolve the best python3 binary. Homebrew python has the real MLX install;
/// the system /usr/bin/python3 (3.9.6) has an older version.
/// Returns the path as a string so callers can use it with Command::new().
pub fn resolve_python() -> &'static str {
    // Candidates in priority order
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ];
    for p in CANDIDATES {
        if std::path::Path::new(p).exists() {
            return p;
        }
    }
    "python3" // fallback to PATH
}

pub use aggregator::ClusterState;
pub use collector::{
    collect_node_metrics, diff_netstat_samples, local_hardware_identity, parse_cpu_clusters,
    parse_footprint, parse_iostat, parse_netstat_ib, parse_powermetrics_text, parse_process_tree,
    parse_ps_mlx, parse_vmstat_and_memsize, NetstatSample, PowerMetricsResult, CMD_NETSTAT_IB,
    CMD_PS_TREE,
};
pub use config::{ClusterConfig, DiscoveryMethod, NodeMap};
pub use monitor::ClusterMonitor;
pub use scanner::{
    DiscoveredPeer, discover_nodes, parse_ifconfig_all_ips, parse_ifconfig_bridges,
    parse_v1_models_metadata, scan_cluster, scan_node, scan_node_fast, scan_seeds,
};
pub use ssh::{SshResult, local_run, ssh_run};
pub use types::{
    ClusterAggregates, ClusterEvent, ClusterType, CoreInfo, CpuClusterInfo, DiskDeviceIo,
    DiskIoStats, DistributedBackend, EngineConfig, EventSink, GpuLockSeverity, GpuLockStatus,
    InterfaceStats, LoadRequest, MetricsHistory, MlxServerInfo, ModelServerMetadata, MonitorError,
    NetworkStats, NodeSnapshot, PeerHeartbeatStatus, PeerStatus, PortState, ProcessFramework,
    ProcessInfo, ProcessTreeNode, RdmaDevice, RdmaLink, RdmaStatus, ScanResult, ServeBackend,
    ServeEngine, ServeState, ServeStatus, ShareRequest, ShareStatus, TaskEnergy, WatchdogReport,
    WatchdogVerdict, WatchedProcess,
};
pub use models::{LocalModel, DiscoveredVolume, default_model_dirs, discover_volumes, external_model_dirs, parse_model_name, scan_models};
pub use health::{
    CheckResult, SetupChecks, ThunderboltFixResult, ThunderboltServiceStatus,
    find_thunderbolt_issues, fix_thunderbolt_services, parse_thunderbolt_services,
    run_setup_checks, validate_thunderbolt_services,
};
