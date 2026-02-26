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
pub mod monitor;
pub mod scanner;
pub mod ssh;
pub mod types;
pub mod models;
pub mod health;

pub use aggregator::ClusterState;
pub use collector::{
    collect_node_metrics, parse_footprint, parse_powermetrics_text, parse_ps_mlx,
    parse_vmstat_and_memsize, PowerMetricsResult,
};
pub use config::{ClusterConfig, DiscoveryMethod, NodeMap};
pub use monitor::ClusterMonitor;
pub use scanner::{
    DiscoveredPeer, discover_nodes, parse_ifconfig_all_ips, parse_ifconfig_bridges,
    parse_v1_models_metadata, scan_cluster, scan_node, scan_node_fast, scan_seeds,
};
pub use ssh::{SshResult, local_run, ssh_run};
pub use types::{
    ClusterAggregates, ClusterEvent, DistributedBackend, EventSink, MetricsHistory,
    MlxServerInfo, ModelServerMetadata, MonitorError, NodeSnapshot, PortState, ProcessFramework,
    ProcessInfo, RdmaDevice, RdmaLink, RdmaStatus, ScanResult, TaskEnergy,
};
pub use models::{LocalModel, default_model_dirs, parse_model_name, scan_models};
pub use health::{CheckResult, SetupChecks, run_setup_checks};
