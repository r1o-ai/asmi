//! Core types for Apple Silicon cluster monitoring.
//!
//! All types are derived from real `powermetrics`, `ps`, `sysctl`, and RDMA
//! output captured in `testdata/`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
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

    // Hardware identity (collected once at daemon startup, never changes)
    #[serde(default)]
    pub chip_model: Option<String>,     // "Apple M3 Ultra"
    #[serde(default)]
    pub serial_number: Option<String>,  // "H7WQ2P7L6X"
    #[serde(default)]
    pub model_name: Option<String>,     // "Mac Studio", "MacBook Pro", "Mac mini"

    // Power (milliwatts from powermetrics)
    pub cpu_watts: f64,
    pub gpu_watts: f64,
    pub ane_watts: f64,

    /// Power source: "Battery", "AC", or None if unknown.
    /// Populated by IOReport when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_source: Option<String>,

    // Utilisation (percent)
    pub cpu_percent: f64,
    pub gpu_percent: f64,

    // Memory
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub ram_percent: f64,
    // Memory breakdown (macOS vm_stat categories)
    // app = active + wired + compressor (what processes actually need)
    // cached = speculative + inactive (file cache, immediately reclaimable)
    #[serde(default)]
    pub ram_app_bytes: u64,
    #[serde(default)]
    pub ram_cached_bytes: u64,

    // Thermals (celsius, optional because not all sources report them)
    pub cpu_temp_c: Option<f64>,
    pub gpu_temp_c: Option<f64>,

    // Per-cluster CPU breakdown (E0, P0, E1, P1, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_clusters: Vec<CpuClusterInfo>,

    // GPU HW active frequency in MHz
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_frequency_mhz: Option<u32>,

    // Disk I/O stats (from iostat)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_io: Option<DiskIoStats>,

    // Network throughput stats (from netstat -ib diff)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkStats>,

    // Running processes of interest (MLX servers, watchdogs, etc.)
    pub processes: Vec<ProcessInfo>,

    // Top energy-consuming tasks from powermetrics
    pub top_tasks: Vec<TaskEnergy>,

    // RDMA subsystem status (devices and port states)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rdma: Option<RdmaStatus>,

    // Interface → IPs mapping for RDMA link correlation.
    // Only includes interfaces with RDMA-relevant IPs (192.168.0.x, 169.254.x.x).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub interface_ips: BTreeMap<String, Vec<String>>,
}

impl NodeSnapshot {
    /// Combined CPU + GPU + ANE power in watts.
    pub fn total_watts(&self) -> f64 {
        (self.cpu_watts + self.gpu_watts + self.ane_watts) / 1000.0
    }

    /// RAM used in GiB (includes file cache — legacy).
    pub fn ram_used_gib(&self) -> f64 {
        self.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// RAM total in GiB.
    pub fn ram_total_gib(&self) -> f64 {
        self.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// App RAM in GiB (active + wired + compressor — what processes actually need).
    pub fn ram_app_gib(&self) -> f64 {
        self.ram_app_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Cached RAM in GiB (speculative + inactive — file cache, reclaimable).
    pub fn ram_cached_gib(&self) -> f64 {
        self.ram_cached_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
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
    /// Model metadata from probing the server's `/v1/models` endpoint.
    /// Empty if the server is not reachable or has no models loaded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_models: Vec<ModelServerMetadata>,
}

/// Recognised ML frameworks / process types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessFramework {
    #[serde(rename = "mlx-lm")]
    MlxLm,
    #[serde(rename = "mlx-share")]
    MlxLmShare,
    #[serde(rename = "mlx-vlm")]
    MlxVlm,
    #[serde(rename = "vllm-mlx")]
    VllmMlx,
    #[serde(rename = "mlx-dist")]
    MlxLaunch,
    #[serde(rename = "mlx-audio")]
    MlxAudio,
    #[serde(rename = "watchdog")]
    Watchdog,
    #[serde(rename = "unknown")]
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
            Self::MlxAudio => write!(f, "mlx-audio"),
            Self::Watchdog => write!(f, "watchdog"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Distributed backend detected from process args or environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
// CPU cluster / frequency info (from powermetrics)
// ---------------------------------------------------------------------------

/// Efficiency or Performance cluster type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClusterType {
    Efficiency,
    Performance,
}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Efficiency => write!(f, "E"),
            Self::Performance => write!(f, "P"),
        }
    }
}

/// Per-CPU-cluster breakdown from powermetrics (e.g., E0, P0, E1, P1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuClusterInfo {
    /// Cluster name from powermetrics (e.g., "E0", "P0", "E1", "P1").
    pub name: String,
    /// Efficiency or Performance.
    pub cluster_type: ClusterType,
    /// HW active frequency in MHz.
    pub frequency_mhz: u32,
    /// HW active residency (percent).
    pub active_residency: f64,
    /// Number of CPU cores in this cluster.
    pub core_count: u32,
    /// Per-core detail.
    pub cores: Vec<CoreInfo>,
}

/// Per-core info within a CPU cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreInfo {
    /// CPU number (0-31).
    pub id: u32,
    /// Per-core frequency in MHz.
    pub frequency_mhz: u32,
    /// Per-core active residency (percent).
    pub active_residency: f64,
}

// ---------------------------------------------------------------------------
// Disk I/O stats (from iostat)
// ---------------------------------------------------------------------------

/// Aggregated disk I/O statistics from `iostat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIoStats {
    /// Per-device I/O.
    pub devices: Vec<DiskDeviceIo>,
    /// Total read throughput across all devices (MB/s).
    pub total_read_mbps: f64,
    /// Total write throughput across all devices (MB/s).
    pub total_write_mbps: f64,
}

/// Per-device disk I/O from `iostat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskDeviceIo {
    /// Device name (e.g., "disk0", "disk3").
    pub name: String,
    /// KB per transfer.
    pub kb_per_transfer: f64,
    /// Transfers per second.
    pub transfers_per_sec: f64,
    /// Throughput in MB/s.
    pub mb_per_sec: f64,
}

// ---------------------------------------------------------------------------
// Network throughput stats (from netstat -ib diff)
// ---------------------------------------------------------------------------

/// Network throughput statistics across interfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Per-interface stats.
    pub interfaces: Vec<InterfaceStats>,
    /// Total receive throughput (Mbps).
    pub total_rx_mbps: f64,
    /// Total transmit throughput (Mbps).
    pub total_tx_mbps: f64,
}

/// Per-interface network throughput.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceStats {
    /// Interface name (e.g., "en3").
    pub name: String,
    /// Receive bytes per second.
    pub rx_bytes_sec: u64,
    /// Transmit bytes per second.
    pub tx_bytes_sec: u64,
    /// Receive throughput (Mbps).
    pub rx_mbps: f64,
    /// Transmit throughput (Mbps).
    pub tx_mbps: f64,
}

// ---------------------------------------------------------------------------
// Process tree (on-demand, not polled)
// ---------------------------------------------------------------------------

/// A node in the process tree, built from `ps -axo pid,ppid,pcpu,pmem,rss,comm`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessTreeNode {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    pub cpu_percent: f64,
    pub mem_percent: f64,
    pub rss_bytes: u64,
    pub children: Vec<ProcessTreeNode>,
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

/// A discovered RDMA link: maps a local interface/device to a remote peer.
/// Built by correlating RDMA device names (rdma_en3 → en3), ifconfig bridges,
/// and ARP table peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaLink {
    /// Local interface name (e.g., "en3").
    pub local_interface: String,
    /// Local IP on that interface (e.g., "169.254.19.163").
    pub local_ip: String,
    /// Remote peer IP (e.g., "169.254.204.162").
    pub remote_ip: String,
    /// Remote peer hostname (e.g., "m3u3").
    pub remote_hostname: String,
    /// RDMA device name derived from interface (e.g., "rdma_en3").
    #[serde(default)]
    pub rdma_device: Option<String>,
    /// Port state of the RDMA device (ACTIVE/DOWN/Unknown).
    #[serde(default)]
    pub port_state: Option<PortState>,
}

impl RdmaLink {
    /// Derive the expected RDMA device name from the interface (en3 → rdma_en3).
    pub fn expected_rdma_device(&self) -> String {
        format!("rdma_{}", self.local_interface)
    }
}

impl std::fmt::Display for RdmaLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self
            .port_state
            .map(|s| s.to_string())
            .unwrap_or_else(|| "?".to_string());
        write!(f, "{} -> {} [{}]", self.local_interface, self.remote_hostname, state)
    }
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
    pub link_speed: Option<String>,
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
    /// Total RAM used in bytes across all nodes (includes file cache — legacy).
    pub total_ram_used_bytes: u64,
    /// Total RAM capacity in bytes across all nodes.
    pub total_ram_total_bytes: u64,
    /// Total app RAM in bytes (active + wired + compressor).
    #[serde(default)]
    pub total_ram_app_bytes: u64,
    /// Total cached RAM in bytes (file cache, reclaimable).
    #[serde(default)]
    pub total_ram_cached_bytes: u64,
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
        let total_ram_app: u64 = online.iter().map(|s| s.ram_app_bytes).sum();
        let total_ram_cached: u64 = online.iter().map(|s| s.ram_cached_bytes).sum();
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
            .flat_map(|s| s.processes.iter())
            .flat_map(|p| {
                // Prefer server-reported model IDs, fall back to ps-parsed model name
                if !p.server_models.is_empty() {
                    p.server_models.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
                } else if let Some(ref model) = p.model {
                    vec![model.clone()]
                } else {
                    vec![]
                }
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
            total_ram_app_bytes: total_ram_app,
            total_ram_cached_bytes: total_ram_cached,
            cpu_avg_percent: cpu_avg,
            gpu_avg_percent: gpu_avg,
            nodes_online: online.len(),
            nodes_total: total_nodes,
            models_loaded: models,
            gpu_footprint_bytes: gpu_footprint,
        }
    }

    /// Total RAM used in GiB (includes file cache — legacy).
    pub fn total_ram_used_gib(&self) -> f64 {
        self.total_ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Total RAM capacity in GiB.
    pub fn total_ram_total_gib(&self) -> f64 {
        self.total_ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Total app RAM in GiB (what processes actually need).
    pub fn total_ram_app_gib(&self) -> f64 {
        self.total_ram_app_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Total cached RAM in GiB (file cache, reclaimable).
    pub fn total_ram_cached_gib(&self) -> f64 {
        self.total_ram_cached_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
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
// Cluster events (real-time activity feed for TUI)
// ---------------------------------------------------------------------------

/// Events emitted during cluster monitoring for real-time UI feedback.
#[derive(Debug, Clone)]
pub enum ClusterEvent {
    /// A discovery method is starting.
    DiscoveryStarted { method: String },
    /// A discovery method found peers.
    DiscoveryFound { method: String, count: usize },
    /// Node probing phase starting.
    ProbingStarted { count: usize },
    /// A single node probe completed.
    NodeProbed {
        hostname: String,
        online: bool,
        chip: Option<String>,
        ram_gb: Option<u64>,
    },
    /// Full cluster scan complete.
    ScanComplete { online: usize, total: usize },
    /// Metrics polling started.
    MetricsPollStarted { count: usize },
    /// Metrics received from one node.
    MetricsReceived { hostname: String },
    /// Node registry saved.
    RegistrySaved { count: usize },
    /// A hostname alias was auto-discovered (e.g., ARP "mac-360" → SSH "m3u2").
    AliasDiscovered { alias: String, canonical: String },
    /// Thunderbolt bridge IPs discovered for a node (used for RDMA/mlx-share).
    RdmaIpsDiscovered {
        canonical: String,
        ips: Vec<String>,
        interface: Option<String>,
    },
    /// An RDMA link was mapped: local interface → remote peer.
    RdmaLinkDiscovered {
        local_interface: String,
        local_ip: String,
        remote_ip: String,
        remote_hostname: String,
        rdma_device: Option<String>,
        port_state: Option<PortState>,
    },
    /// Local RDMA device correlated with interface (post-scan).
    /// Used to update existing RdmaLinks with port state.
    RdmaDeviceCorrelated {
        interface: String,
        rdma_device: String,
        port_state: PortState,
    },
    /// Thunderbolt network service naming issues detected on a node.
    /// Fires when duplicate or non-r1o-prefixed services are found.
    ThunderboltServiceIssue {
        hostname: String,
        issues: Vec<String>,
    },
}

/// Sink for emitting cluster events. Cheap to clone, silently drops if
/// no subscribers or if constructed as no-op.
#[derive(Clone)]
pub struct EventSink(Option<tokio::sync::broadcast::Sender<ClusterEvent>>);

impl EventSink {
    /// Create a sink that emits to a broadcast channel.
    pub fn new(tx: tokio::sync::broadcast::Sender<ClusterEvent>) -> Self {
        Self(Some(tx))
    }

    /// Create a no-op sink that discards all events.
    pub fn noop() -> Self {
        Self(None)
    }

    /// Emit an event. Silently drops if no subscribers.
    pub fn emit(&self, event: ClusterEvent) {
        if let Some(tx) = &self.0 {
            let _ = tx.send(event);
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Metadata returned by a `/v1/models` endpoint for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelServerMetadata {
    pub id: String,
    pub context_length: Option<u64>,
    pub max_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// MLX serve lifecycle (merged from mlx_daemon.py)
// ---------------------------------------------------------------------------

/// Lifecycle state of the managed MLX server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServeState {
    Idle,
    Bare,
    Loading,
    Ready,
    Error,
}

impl fmt::Display for ServeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Bare => write!(f, "bare"),
            Self::Loading => write!(f, "loading"),
            Self::Ready => write!(f, "ready"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// ML serving engine variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ServeEngine {
    #[default]
    #[serde(rename = "mlx_lm")]
    MlxLm,
    #[serde(rename = "mlx_vlm")]
    MlxVlm,
    #[serde(rename = "vllm_mlx")]
    VllmMlx,
    #[serde(rename = "mlx_lm_share")]
    MlxLmShare,
}


impl fmt::Display for ServeEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MlxLm => write!(f, "mlx_lm"),
            Self::MlxVlm => write!(f, "mlx_vlm"),
            Self::VllmMlx => write!(f, "vllm_mlx"),
            Self::MlxLmShare => write!(f, "mlx_lm_share"),
        }
    }
}

/// Distributed vs single-node serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServeBackend {
    #[default]
    Single,
    Jaccl,
}


impl fmt::Display for ServeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single => write!(f, "single"),
            Self::Jaccl => write!(f, "jaccl"),
        }
    }
}

/// Per-engine command configuration — replaces the Python ENGINES dict.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Binary to invoke (e.g. "mlx_lm.server", "uvicorn", "vllm").
    pub binary: &'static str,
    /// Extra args after the binary (e.g. ["serve"] for vllm).
    pub binary_args: &'static [&'static str],
    /// Uvicorn app string (e.g. "mlx_vlm.server:app"). None for non-uvicorn engines.
    pub uvicorn_app: Option<&'static str>,
    /// Flag to pass the model path (e.g. "--model"). None if models load lazily.
    pub model_flag: Option<&'static str>,
    /// HTTP paths to poll for health (tried in order).
    pub health_endpoints: &'static [&'static str],
}

impl ServeEngine {
    /// Get the command configuration for this engine.
    pub fn config(self) -> EngineConfig {
        match self {
            Self::MlxLm => EngineConfig {
                binary: "mlx_lm",
                binary_args: &["server"],
                uvicorn_app: None,
                model_flag: Some("--model"),
                health_endpoints: &["/v1/models", "/models", "/health"],
            },
            Self::MlxVlm => EngineConfig {
                binary: "mlx_vlm",
                binary_args: &["server"],
                uvicorn_app: None,
                // mlx_vlm.server does NOT support --model flag.
                // Models load lazily via the `model` field in /chat/completions requests.
                // See warmup logic in serve.rs for pre-loading after bare start.
                model_flag: None,
                health_endpoints: &["/health", "/models"],
            },
            Self::VllmMlx => EngineConfig {
                binary: "vllm-mlx",
                binary_args: &["serve"],
                uvicorn_app: None,
                model_flag: Some("--model"),
                health_endpoints: &["/v1/models", "/health"],
            },
            Self::MlxLmShare => EngineConfig {
                binary: "mlx_lm",
                binary_args: &["share"],
                uvicorn_app: None,
                model_flag: Some("--model"),
                // share has no HTTP server — readiness checked via log output
                health_endpoints: &[],
            },
        }
    }
}

/// State of a launchd-managed agent, as reported by asmi's `launchd` module.
///
/// These correspond to `launchctl print` / `launchctl print-disabled` output:
/// - `Running` — top-level `state = running`
/// - `Waiting` — registered but not currently running (e.g. event-gated)
/// - `Disabled` — user or asmi has run `launchctl disable`; process is booted out
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LaunchdState {
    Running,
    Waiting,
    Disabled,
}

/// Information about a launchd agent that manages (or managed) a PID on disk.
///
/// Populated for any model-serving process that is backed by a `com.*.plist`
/// in `~/Library/LaunchAgents`. The web UI and TUI render this to show a
/// "KeepAlive" badge and offer a one-click disable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchdInfo {
    pub label: String,
    pub keep_alive: Option<bool>,
    pub run_at_load: Option<bool>,
    pub state: LaunchdState,
    pub program: Option<String>,
}

/// Read-only snapshot of the serve manager state, returned by status endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeStatus {
    pub state: ServeState,
    pub model: Option<String>,
    pub engine: ServeEngine,
    pub backend: ServeBackend,
    pub port: u16,
    pub pid: Option<u32>,
    pub port_verified: bool,
    pub elapsed_ms: u64,
    pub error: Option<String>,
    /// launchd agent backing this PID (if any). `None` when the process is not
    /// launchd-managed or the agent couldn't be resolved. `#[serde(default)]`
    /// keeps older clients compatible.
    #[serde(default)]
    pub launchd: Option<LaunchdInfo>,
}

/// A model process detected on the node that was NOT launched by asmi.
/// Found by diffing the metrics process scanner against managed ServeManagers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmanagedProcess {
    pub pid: u32,
    pub port: Option<u16>,
    pub engine: String,
    pub models: Vec<String>,
    pub source: &'static str,
    #[serde(default)]
    pub launchd: Option<LaunchdInfo>,
}

/// Request body for POST /serve/load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default = "default_backend_str")]
    pub backend: String,
    pub hostfile: Option<String>,
    #[serde(default)]
    pub engine: ServeEngine,

    // --- Optimization parameters (mlx_lm.server passthrough) ---

    /// Speculative decoding: path/repo for the draft model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// Speculative decoding: number of draft tokens per step (default: 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_draft_tokens: Option<u32>,
    /// Max concurrent decode requests in a batch (incompatible with draft_model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_concurrency: Option<u32>,
    /// Max concurrent prompt/prefill requests in a batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_concurrency: Option<u32>,
    /// Prefill step size in tokens (default: 2048).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_step_size: Option<u32>,
    /// Use pipeline parallelism instead of tensor parallelism (JACCL only).
    #[serde(default)]
    pub pipeline: bool,
    /// Max KV caches held in prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_size: Option<u32>,
    /// Max bytes for prompt KV cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_bytes: Option<u64>,
}

fn default_backend_str() -> String {
    "auto".to_string()
}

impl Default for LoadRequest {
    fn default() -> Self {
        Self {
            model_path: None,
            backend: default_backend_str(),
            hostfile: None,
            engine: ServeEngine::default(),
            draft_model: None,
            num_draft_tokens: None,
            decode_concurrency: None,
            prompt_concurrency: None,
            prefill_step_size: None,
            pipeline: false,
            prompt_cache_size: None,
            prompt_cache_bytes: None,
        }
    }
}

/// Request body for POST /serve/share.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareRequest {
    /// Model path or HuggingFace repo ID (required — no bare share).
    pub model_path: String,
    /// Backend: "auto" | "jaccl" (defaults to auto, resolves to jaccl if hostfile exists).
    #[serde(default = "default_backend_str")]
    pub backend: String,
    /// JACCL hostfile path. Falls back to ~/.r1o/hostfiles/default.json.
    pub hostfile: Option<String>,
}

/// Read-only snapshot of the share session state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareStatus {
    pub state: ServeState,
    pub model: Option<String>,
    pub backend: ServeBackend,
    pub pid: Option<u32>,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Watchdog: process watchdog + GPU lock detection
// ---------------------------------------------------------------------------

/// Verdict for a watched inference process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WatchdogVerdict {
    Healthy,
    Degraded { reason: String },
    Stuck { reason: String, duration_secs: u64 },
    Killed { reason: String, at: String },
}

/// A process being actively watched by the watchdog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchedProcess {
    pub pid: u32,
    pub framework: String,
    pub verdict: WatchdogVerdict,
    pub since: String,
    pub port_reachable: Option<bool>,
    pub cpu_percent: f64,
}

/// Full watchdog report aggregating all monitoring signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogReport {
    pub processes: Vec<WatchedProcess>,
    pub gpu_lock: GpuLockStatus,
    pub peer_heartbeat: PeerHeartbeatStatus,
    pub last_check: String,
}

/// GPU Lock detection status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuLockStatus {
    /// Whether GPU Lock is currently detected.
    pub detected: bool,
    /// PIDs suspected of causing the lock.
    pub suspect_pids: Vec<u32>,
    /// When the lock condition was first detected (ISO 8601).
    pub since: Option<String>,
    /// Current severity level.
    pub severity: GpuLockSeverity,
}

/// Severity levels for GPU Lock detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuLockSeverity {
    /// No GPU Lock detected.
    None,
    /// High CPU + low GPU, might recover on its own.
    Suspected,
    /// Sustained >15s, port unreachable — SIGTERM/SIGKILL attempted.
    Confirmed,
    /// SIGKILL sent but process still alive — requires manual reboot.
    Unrecoverable,
}

// ---------------------------------------------------------------------------
// Watchdog: peer heartbeat
// ---------------------------------------------------------------------------

/// Status of the RDMA peer heartbeat monitor.
/// Tracks whether each peer's asmi daemon is reachable. If a peer goes dark
/// for 3+ consecutive checks (1s interval), the local inference is killed
/// to prevent GPU Lock from hung RDMA operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerHeartbeatStatus {
    /// Whether the heartbeat loop is actively running.
    pub active: bool,
    /// Status of each monitored peer.
    pub peers: Vec<PeerStatus>,
    /// When the current monitoring session started (ISO 8601).
    pub session_start: Option<String>,
}

/// Health status of a single RDMA peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStatus {
    pub hostname: String,
    pub reachable: bool,
    /// Last time this peer responded to a health check (ISO 8601).
    pub last_seen: Option<String>,
    /// Number of consecutive missed health checks.
    pub consecutive_misses: u32,
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
