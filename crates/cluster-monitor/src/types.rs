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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServeEngine {
    #[serde(rename = "mlx_lm")]
    MlxLm,
    #[serde(rename = "mlx_vlm")]
    MlxVlm,
    #[serde(rename = "vllm_mlx")]
    VllmMlx,
    #[serde(rename = "mlx_lm_share")]
    MlxLmShare,
}

impl Default for ServeEngine {
    fn default() -> Self {
        Self::MlxLm
    }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServeBackend {
    Single,
    Jaccl,
}

impl Default for ServeBackend {
    fn default() -> Self {
        Self::Single
    }
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
                binary: "mlx_lm.server",
                binary_args: &[],
                uvicorn_app: None,
                model_flag: Some("--model"),
                health_endpoints: &["/v1/models", "/models", "/health"],
            },
            Self::MlxVlm => EngineConfig {
                binary: "uvicorn",
                binary_args: &[],
                uvicorn_app: Some("mlx_vlm.server:app"),
                model_flag: None, // models load lazily via chat body
                health_endpoints: &["/models", "/health", "/v1/models"],
            },
            Self::VllmMlx => EngineConfig {
                binary: "vllm",
                binary_args: &["serve"],
                uvicorn_app: None,
                model_flag: Some("--model"),
                health_endpoints: &["/v1/models", "/health"],
            },
            Self::MlxLmShare => EngineConfig {
                binary: "mlx_lm.share",
                binary_args: &[],
                uvicorn_app: None,
                model_flag: Some("--model"),
                // share has no HTTP server — readiness checked via log output
                health_endpoints: &[],
            },
        }
    }
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
}

fn default_backend_str() -> String {
    "auto".to_string()
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
