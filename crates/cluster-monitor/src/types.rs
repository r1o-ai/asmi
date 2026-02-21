//! Core types for Apple Silicon cluster monitoring.
//!
//! All types are derived from real `powermetrics`, `ps`, `sysctl`, and RDMA
//! output captured in `testdata/`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Node-level snapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a single cluster node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub hostname: String,
    pub online: bool,
    pub timestamp: DateTime<Utc>,

    // Power (milliwatts from powermetrics)
    pub cpu_watts: f64,
    pub gpu_watts: f64,
    pub ane_watts: f64,

    // Utilisation (percent)
    pub cpu_percent: f64,
    pub gpu_percent: f64,

    // Memory
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub ram_percent: f64,

    // Thermals (celsius, optional because not all sources report them)
    pub cpu_temp_c: Option<f64>,
    pub gpu_temp_c: Option<f64>,

    // Running processes of interest (MLX servers, watchdogs, etc.)
    pub processes: Vec<ProcessInfo>,

    // Top energy-consuming tasks from powermetrics
    pub top_tasks: Vec<TaskEnergy>,
}

impl NodeSnapshot {
    /// Combined CPU + GPU + ANE power in watts.
    pub fn total_watts(&self) -> f64 {
        (self.cpu_watts + self.gpu_watts + self.ane_watts) / 1000.0
    }

    /// RAM used in GiB.
    pub fn ram_used_gib(&self) -> f64 {
        self.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// RAM total in GiB.
    pub fn ram_total_gib(&self) -> f64 {
        self.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// ---------------------------------------------------------------------------
// Process information
// ---------------------------------------------------------------------------

/// A process running on a node that we care about (MLX server, watchdog, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub framework: ProcessFramework,
    pub model: Option<String>,
    pub port: Option<u16>,
    pub cpu_percent: f64,
    pub mem_percent: f64,
    pub footprint_mb: Option<f64>,
    /// Distributed backend if this is part of a distributed run.
    pub distributed: Option<DistributedBackend>,
}

/// Recognised ML frameworks / process types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessFramework {
    MlxLm,
    MlxLmShare,
    MlxVlm,
    VllmMlx,
    MlxLaunch,
    Watchdog,
    Unknown,
}

impl fmt::Display for ProcessFramework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MlxLm => write!(f, "mlx-lm"),
            Self::MlxLmShare => write!(f, "mlx-share"),
            Self::MlxVlm => write!(f, "mlx-vlm"),
            Self::VllmMlx => write!(f, "vllm-mlx"),
            Self::MlxLaunch => write!(f, "mlx-dist"),
            Self::Watchdog => write!(f, "watchdog"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Distributed backend detected from process args or environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistributedBackend {
    Jaccl,
    Ring,
}

impl fmt::Display for DistributedBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jaccl => write!(f, "jaccl"),
            Self::Ring => write!(f, "ring"),
        }
    }
}

// ---------------------------------------------------------------------------
// Energy / task info (from powermetrics)
// ---------------------------------------------------------------------------

/// Per-task energy impact from powermetrics sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEnergy {
    pub name: String,
    pub pid: u32,
    pub energy_impact: f64,
    pub watts_share: f64,
}

// ---------------------------------------------------------------------------
// RDMA / Thunderbolt mesh
// ---------------------------------------------------------------------------

/// RDMA subsystem status for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaStatus {
    pub enabled: bool,
    pub devices: Vec<RdmaDevice>,
}

impl RdmaStatus {
    /// Count of devices in PORT_ACTIVE state.
    pub fn active_count(&self) -> usize {
        self.devices
            .iter()
            .filter(|d| d.port_state == PortState::Active)
            .count()
    }
}

/// A single RDMA device (e.g. rdma_en3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaDevice {
    pub name: String,
    pub port_state: PortState,
}

/// RDMA port link state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortState {
    Active,
    Down,
    Unknown,
}

impl fmt::Display for PortState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "ACTIVE"),
            Self::Down => write!(f, "DOWN"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

impl PortState {
    /// Parse from the ibstat output string, e.g. `"PORT_ACTIVE (4)"`.
    pub fn from_ibstat(s: &str) -> Self {
        let upper = s.to_uppercase();
        if upper.contains("ACTIVE") {
            Self::Active
        } else if upper.contains("DOWN") {
            Self::Down
        } else {
            Self::Unknown
        }
    }
}

// ---------------------------------------------------------------------------
// Scan result (full node probe)
// ---------------------------------------------------------------------------

/// Result of a full scan / probe of a node (SSH connectivity, hardware info,
/// running MLX servers, RDMA status).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub hostname: String,
    pub reachable: bool,
    pub ssh_ok: bool,
    pub chip: Option<String>,
    pub ram_gb: Option<u64>,
    pub gpu_cores: Option<u32>,
    pub rdma: Option<RdmaStatus>,
    pub mlx_servers: Vec<MlxServerInfo>,
    pub latency_ms: Option<f64>,
}

/// An MLX-LM (or compatible) server discovered on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxServerInfo {
    pub port: u16,
    pub models: Vec<String>,
    pub engine: ProcessFramework,
}

// ---------------------------------------------------------------------------
// Cluster aggregates
// ---------------------------------------------------------------------------

/// Aggregated metrics across all online nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterAggregates {
    /// Total combined power draw in watts.
    pub total_watts: f64,
    /// Total RAM used in bytes across all nodes.
    pub total_ram_used_bytes: u64,
    /// Total RAM capacity in bytes across all nodes.
    pub total_ram_total_bytes: u64,
    /// Average CPU utilisation (percent) across all online nodes.
    pub cpu_avg_percent: f64,
    /// Average GPU utilisation (percent) across all online nodes.
    pub gpu_avg_percent: f64,
    /// Number of online nodes.
    pub nodes_online: usize,
    /// Total number of known nodes (online + offline).
    pub nodes_total: usize,
    /// Distinct model names currently loaded across the cluster.
    pub models_loaded: Vec<String>,
    /// Estimated GPU memory footprint in bytes (sum of MLX process footprints).
    pub gpu_footprint_bytes: u64,
}

impl ClusterAggregates {
    /// Recompute aggregates from a set of node snapshots.
    pub fn from_snapshots(snapshots: &[NodeSnapshot], total_nodes: usize) -> Self {
        let online: Vec<&NodeSnapshot> = snapshots.iter().filter(|s| s.online).collect();
        let n = online.len() as f64;

        let total_watts: f64 = online.iter().map(|s| s.total_watts()).sum();
        let total_ram_used: u64 = online.iter().map(|s| s.ram_used_bytes).sum();
        let total_ram_total: u64 = online.iter().map(|s| s.ram_total_bytes).sum();
        let cpu_avg = if n > 0.0 {
            online.iter().map(|s| s.cpu_percent).sum::<f64>() / n
        } else {
            0.0
        };
        let gpu_avg = if n > 0.0 {
            online.iter().map(|s| s.gpu_percent).sum::<f64>() / n
        } else {
            0.0
        };

        let mut models: Vec<String> = online
            .iter()
            .flat_map(|s| {
                s.processes
                    .iter()
                    .filter_map(|p| p.model.clone())
            })
            .collect();
        models.sort();
        models.dedup();

        let gpu_footprint: u64 = online
            .iter()
            .flat_map(|s| &s.processes)
            .filter_map(|p| p.footprint_mb.map(|mb| (mb * 1024.0 * 1024.0) as u64))
            .sum();

        Self {
            total_watts,
            total_ram_used_bytes: total_ram_used,
            total_ram_total_bytes: total_ram_total,
            cpu_avg_percent: cpu_avg,
            gpu_avg_percent: gpu_avg,
            nodes_online: online.len(),
            nodes_total: total_nodes,
            models_loaded: models,
            gpu_footprint_bytes: gpu_footprint,
        }
    }

    /// Total RAM used in GiB.
    pub fn total_ram_used_gib(&self) -> f64 {
        self.total_ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Total RAM capacity in GiB.
    pub fn total_ram_total_gib(&self) -> f64 {
        self.total_ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// ---------------------------------------------------------------------------
// Metrics history (ring buffers for charts)
// ---------------------------------------------------------------------------

/// Rolling history of metrics for time-series rendering.
#[derive(Debug, Clone)]
pub struct MetricsHistory {
    capacity: usize,
    pub cpu: VecDeque<f64>,
    pub gpu: VecDeque<f64>,
    pub memory: VecDeque<f64>,
    pub power: VecDeque<f64>,
}

impl MetricsHistory {
    /// Create a new history with the given capacity (number of samples to keep).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cpu: VecDeque::with_capacity(capacity),
            gpu: VecDeque::with_capacity(capacity),
            memory: VecDeque::with_capacity(capacity),
            power: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new sample, evicting the oldest if at capacity.
    pub fn push(&mut self, cpu: f64, gpu: f64, memory: f64, power: f64) {
        if self.cpu.len() >= self.capacity {
            self.cpu.pop_front();
            self.gpu.pop_front();
            self.memory.pop_front();
            self.power.pop_front();
        }
        self.cpu.push_back(cpu);
        self.gpu.push_back(gpu);
        self.memory.push_back(memory);
        self.power.push_back(power);
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.cpu.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.cpu.is_empty()
    }

    /// Maximum number of samples this history can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from cluster monitoring operations.
#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("SSH command failed for {host}: {message}")]
    SshFailed { host: String, message: String },

    #[error("command timed out after {timeout_secs}s for {host}")]
    Timeout { host: String, timeout_secs: u64 },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("node unreachable: {0}")]
    Unreachable(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
