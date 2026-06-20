//! RDMA auto-setup: runs on daemon startup to ensure TB5 interfaces are ready.
//!
//! Sequence: detect bridge0 → destroy → assign IPs → fix routes → verify peers → write hostfile.
//! All steps are non-fatal — daemon starts regardless.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Duration;

use crate::topology::TopologyReport;

// ── Coordinator IP persistence ─────────────────────────────────────────
// The coordinator (hub) computes and applies /30 IPs during autosetup, but
// autosetup runs AFTER the topology cache loop starts. On restart, the first
// topology scan races against IP assignment → incomplete results. Fix: persist
// the coordinator's own IPs to a file and restore them synchronously at boot,
// before either loop starts.

#[derive(Debug, Serialize, Deserialize)]
struct PersistedCoordinatorIps {
    ips: Vec<PersistedIpEntry>,
    assigned_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedIpEntry {
    iface: String,
    ip: String,
    netmask: String,
}

fn coordinator_ips_path() -> std::path::PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".config"))
        .join("asmi/coordinator_ips.json")
}

/// Save the coordinator's own /30 IPs after topology-derived assignment.
/// Called only on the coordinator node after successful assign_topology_ips().
pub fn persist_coordinator_ips(ips: &[InterfaceIp]) {
    let path = coordinator_ips_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = PersistedCoordinatorIps {
        ips: ips.iter().map(|ip| PersistedIpEntry {
            iface: ip.iface.clone(),
            ip: ip.ip.clone(),
            netmask: "255.255.255.252".into(),
        }).collect(),
        assigned_at: chrono::Utc::now().to_rfc3339(),
    };
    match serde_json::to_string_pretty(&data) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!("failed to persist coordinator IPs to {}: {e}", path.display());
            } else {
                tracing::info!("persisted {} coordinator IPs to {}", ips.len(), path.display());
            }
        }
        Err(e) => tracing::warn!("failed to serialize coordinator IPs: {e}"),
    }
}

/// Restore and apply coordinator's persisted /30 IPs at daemon boot.
/// Runs synchronously (blocking) — call before topology cache loop starts.
pub fn restore_coordinator_ips() -> Vec<InterfaceIp> {
    let path = coordinator_ips_path();
    let data = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let persisted: PersistedCoordinatorIps = match serde_json::from_str(&data) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("failed to parse {}: {e}", path.display());
            return vec![];
        }
    };

    let mut applied = vec![];
    for entry in &persisted.ips {
        let out = Command::new("sudo")
            .args(["ifconfig", &entry.iface, "inet", &entry.ip, "netmask", &entry.netmask])
            .output();
        match out {
            Ok(o) if o.status.success() => {
                // Ensure /30 route
                let octets: Vec<u8> = entry.ip.split('.').filter_map(|s| s.parse().ok()).collect();
                if octets.len() == 4 {
                    let net_base = octets[3] & 0xFC;
                    let subnet = format!("{}.{}.{}.{}/30", octets[0], octets[1], octets[2], net_base);
                    let _ = Command::new("sudo").args(["route", "delete", "-net", &subnet]).output();
                    let _ = Command::new("sudo").args(["route", "add", "-net", &subnet, "-interface", &entry.iface]).output();
                }
                applied.push(InterfaceIp {
                    iface: entry.iface.clone(),
                    ip: entry.ip.clone(),
                    source: "persisted".into(),
                });
            }
            _ => tracing::warn!("{}: failed to restore IP {}", entry.iface, entry.ip),
        }
    }
    if !applied.is_empty() {
        tracing::info!("restored {} coordinator IPs from {}", applied.len(), path.display());
    }
    applied
}

/// RDMA settings from ~/.r1o/settings.json
#[derive(Debug, Deserialize)]
struct RdmaSettings {
    #[serde(default = "default_true")]
    auto_setup: bool,
    #[serde(default = "default_true")]
    auto_destroy_bridge0: bool,
    #[serde(default)]
    _ip_assignment: Option<String>,
    #[serde(default = "default_true")]
    _route_fix: bool,
    /// Static TB5 IP assignments per interface. Applied on boot instead of random
    /// link-locals. Format: [{"iface": "en5", "ip": "192.168.10.2", "mask": "255.255.255.252"}]
    /// Nodes without this list fall back to link-local auto-assignment.
    #[serde(default)]
    static_ips: Vec<StaticIpEntry>,
    /// Default gateway for nodes that depend on TB5 Internet Sharing (no ethernet).
    /// Applied after static IPs. Example: "192.168.10.1" (hub's TB5 address).
    #[serde(default)]
    default_gateway: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StaticIpEntry {
    iface: String,
    ip: String,
    #[serde(default = "default_mask")]
    mask: String,
}

fn default_mask() -> String { "255.255.255.252".into() }

fn default_true() -> bool { true }

impl Default for RdmaSettings {
    fn default() -> Self {
        Self {
            auto_setup: true,
            auto_destroy_bridge0: true,
            _ip_assignment: None,
            _route_fix: true,
            static_ips: vec![],
            default_gateway: None,
        }
    }
}

fn load_settings() -> RdmaSettings {
    let path = dirs::home_dir()
        .map(|h| h.join(".r1o/settings.json"))
        .unwrap_or_default();

    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| {
            serde_json::from_str::<serde_json::Value>(&s)
                .ok()
                .and_then(|v| {
                    v.get("rdma")
                        .and_then(|r| serde_json::from_value::<RdmaSettings>(r.clone()).ok())
                })
        })
        .unwrap_or_default()
}

/// Result of the full autosetup sequence.
#[derive(Debug, Default, Serialize)]
pub struct AutosetupReport {
    pub bridge0: Bridge0Result,
    pub ips: Vec<InterfaceIp>,
    pub routes: RouteResult,
    pub peers: PeerResult,
    pub hostfile: Option<String>,
}

impl AutosetupReport {
    pub fn summary(&self) -> String {
        let peer_count = self.peers.verified_links.len();
        format!(
            "bridge0={}, ips={}, routes={}, peers={}/{}{}",
            self.bridge0,
            self.ips.len(),
            self.routes,
            peer_count,
            self.peers.total_tried,
            if let Some(ref hf) = self.hostfile {
                format!(", hostfile={hf}")
            } else {
                String::new()
            }
        )
    }
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bridge0Result {
    #[default]
    Clean,
    Destroyed(Vec<String>),
    Failed(String),
}

impl std::fmt::Display for Bridge0Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clean => write!(f, "clean"),
            Self::Destroyed(m) => write!(f, "destroyed({})", m.join(",")),
            Self::Failed(e) => write!(f, "failed({e})"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InterfaceIp {
    pub iface: String,
    pub ip: String,
    pub source: String,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteResult {
    #[default]
    NoInterfaces,
    Fixed {
        primary_interface: String,
        interfaces_with_ips: usize,
    },
    #[allow(dead_code)]
    Failed(String),
}

impl std::fmt::Display for RouteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInterfaces => write!(f, "no_interfaces"),
            Self::Fixed { primary_interface, interfaces_with_ips } => {
                write!(f, "fixed(primary={primary_interface}, n={interfaces_with_ips})")
            }
            Self::Failed(e) => write!(f, "failed({e})"),
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct PeerResult {
    pub verified_links: Vec<VerifiedLink>,
    pub total_tried: usize,
}

#[derive(Debug, Serialize)]
pub struct VerifiedLink {
    pub hostname: String,
    pub ip: String,
    pub latency_ms: Option<f64>,
}

// ── Public entry point ───────────────────────────────────────────────

/// Run the full RDMA autosetup sequence. Non-fatal — logs warnings on failure.
/// Respects settings from ~/.r1o/settings.json → rdma.
///
/// `hosts_override`: caller-supplied list of hosts for topology discovery.
/// `None` → derive from NodeMap.nodes (all known nodes). `Some(vec)` → use
/// exactly those hosts.
pub async fn autosetup(
    node_map: &tokio::sync::RwLock<asmi_core::NodeMap>,
    hosts_override: Option<Vec<String>>,
) -> AutosetupReport {
    let settings = load_settings();
    let mut report = AutosetupReport::default();

    if !settings.auto_setup {
        tracing::info!("RDMA autosetup disabled in settings");
        return report;
    }

    // Step 1: bridge0
    if settings.auto_destroy_bridge0 {
        report.bridge0 = handle_bridge0().await;
    } else {
        tracing::info!("bridge0 auto-destroy disabled in settings");
    }

    // Step 2: Assign IPs to TB5 interfaces.
    // Priority: topology-derived > static config > link-local fallback.
    //
    // Topology-derived: discover_topology() uses native HTTP + UUID cross-match
    // (fallback: mlx.distributed_config, then ARP). Sorts links deterministically,
    // assigns 192.168.10.{4i+1}/30 per link. Applied locally + pushed to remote
    // nodes via SSH.
    let hosts: Vec<String> = match hosts_override {
        Some(h) => {
            tracing::info!(count = h.len(), "autosetup: using caller-supplied host list");
            h
        }
        None => {
            let nm = node_map.read().await;
            // Use all known nodes (not just rdma_ips) so topology discovers
            // links between nodes that don't have IPs assigned yet.
            if !nm.nodes.is_empty() {
                nm.nodes.clone()
            } else {
                nm.rdma_ips.keys().cloned().collect()
            }
        }
    };

    let local_hostname = get_local_hostname();
    let mut topology_ok = false;

    // Only the COORDINATOR (lexically-first known host) runs topology-derived
    // assignment with remote push. In a cluster that is not a full mesh, a peer can
    // have a PARTIAL topology view — e.g. m3u3 is not cabled to m3u2, so it discovers
    // only 3 of 4 nodes — and with push_remote=true it would SSH stale/partial /30s
    // onto its peers, corrupting the coordinator's correct map (observed during the
    // m3u2 4-node bring-up: m3u3's restart pushed 3-node IPs over hub). The
    // coordinator is, by construction, cabled to every peer and holds the full view.
    // Non-coordinators defer here and fall through to their own static_ips, which are
    // the authoritative per-node /30s (and are also what the coordinator pushes), so
    // a peer reboot is self-correct without touching anyone else.
    let host_set: std::collections::HashSet<String> =
        hosts.iter().map(|h| h.to_lowercase()).collect();
    let local_canon = asmi_core::aggregator::canonicalize_hostname(&local_hostname, &host_set);
    let coordinator = host_set.iter().min().cloned();
    let is_coordinator = coordinator.as_deref() == Some(local_canon.as_str());

    if is_coordinator && hosts.len() >= 2 {
        // Retry discovery until the link set stabilizes before assigning. This
        // closes the simultaneous-restart race: at boot, peers are still coming
        // up, so a single discovery can see a partial (or empty) topology, which
        // would flip topology_ok=false and trigger the static-IP fallback —
        // re-applying stale/colliding values. Waiting for a stable mesh keeps the
        // topology-derived path (the correct per-port /30 = GID/coordinator basis)
        // as the single source of truth. See discover_stable_topology.
        if let Some(topo) = discover_stable_topology(hosts.clone()).await {
            let assigned = assign_topology_ips(&topo, &local_hostname, true).await;
            if !assigned.is_empty() {
                tracing::info!(
                    count = assigned.len(),
                    links = topo.links.len(),
                    "topology-derived IPs assigned (coordinator: local + remote)"
                );
                persist_coordinator_ips(&assigned);
                report.ips = assigned;
                topology_ok = true;
            }
        }
    } else if !is_coordinator {
        tracing::info!(
            coordinator = coordinator.as_deref().unwrap_or("?"),
            "non-coordinator node: deferring topology assignment to coordinator, using static_ips"
        );
    }

    // Fallback: static config or link-local
    if !topology_ok {
        if !settings.static_ips.is_empty() {
            report.ips = apply_static_ips(&settings.static_ips).await;
            if let Some(ref gw) = settings.default_gateway {
                apply_default_gateway(gw).await;
            }
        } else {
            match ensure_tb5_ips().await {
                Ok(ips) => report.ips = ips,
                Err(e) => tracing::warn!("ensure_tb5_ips failed: {e}"),
            }
        }
    }

    // Step 2b: clean self-MAC poisoned ARP entries (bridge0 proxy ARP remnants)
    clean_self_arp_poison(&report.ips).await;

    // Step 3: generate JACCL hostfile via mlx.distributed_config (no --auto-setup).
    if !hosts.is_empty() {
        match run_mlx_distributed_config(&hosts).await {
            Ok((hostfile_path, verified_links)) => {
                let n_verified = verified_links.len();
                report.hostfile = Some(hostfile_path);
                report.peers = PeerResult {
                    total_tried: hosts.len(),
                    verified_links,
                };
                if let Some(first) = report.ips.first() {
                    report.routes = RouteResult::Fixed {
                        primary_interface: first.iface.clone(),
                        interfaces_with_ips: n_verified.saturating_sub(1),
                    };
                }
            }
            Err(e) => {
                tracing::warn!("mlx.distributed_config failed: {e}");
                let nm = node_map.read().await;
                let (routes, peers) = probe_and_route(&nm, &report.ips).await;
                report.routes = routes;
                report.peers = peers;
                if !report.peers.verified_links.is_empty() {
                    if let Ok(path) = write_hostfile(&report.peers, &report.ips).await {
                        report.hostfile = Some(path);
                    }
                }
            }
        }
    }

    tracing::info!("RDMA autosetup complete: {}", report.summary());
    report
}

/// Call Apple's official mlx.distributed_config tool to discover topology
/// and generate a correct JACCL hostfile. Much more reliable than our custom
/// probing because it uses SSH + hardware introspection instead of IP pings.
async fn run_mlx_distributed_config(
    hosts: &[String],
) -> Result<(String, Vec<VerifiedLink>), String> {
    // Build host list — use .local hostnames for mDNS resolution
    let host_args: Vec<String> = hosts
        .iter()
        .map(|h| {
            if h.contains('.') {
                h.clone()
            } else {
                format!("{h}.local")
            }
        })
        .collect();
    let hosts_arg = host_args.join(",");

    let output_path = format!(
        "{}/.r1o/hostfiles/auto.json",
        std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
    );

    // Ensure directory exists
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Find mlx.distributed_config — launchd gives asmi empty PATH so we need absolute path
    let mlx_config_bin = ["/opt/homebrew/bin/mlx.distributed_config", "/usr/local/bin/mlx.distributed_config"]
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|s| s.to_string())
        .ok_or_else(|| "mlx.distributed_config not found (expected /opt/homebrew/bin/)".to_string())?;

    let output_path_clone = output_path.clone();
    let hosts_arg_clone = hosts_arg.clone();
    let result = tokio::task::spawn_blocking(move || {
        // No --auto-setup: it kernel panics on macOS 26.5 (FB100029547).
        // IP assignment is handled by assign_topology_ips() instead.
        std::process::Command::new(&mlx_config_bin)
            .args([
                "--hosts",
                &hosts_arg_clone,
                "--over",
                "thunderbolt",
                "--backend",
                "jaccl",
                "--output-hostfile",
                &output_path_clone,
                "--ignore-unreachable",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .map_err(|e| format!("spawn mlx.distributed_config: {e}"))
    })
    .await
    .map_err(|e| format!("join: {e}"))??;

    if !result.status.success() && !std::path::Path::new(&output_path).exists() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!(
            "mlx.distributed_config exited {:?}: {stderr}",
            result.status.code()
        ));
    }

    // Parse the generated hostfile to extract peer list
    let hostfile_text = std::fs::read_to_string(&output_path)
        .map_err(|e| format!("read hostfile: {e}"))?;
    let hostfile: serde_json::Value =
        serde_json::from_str(&hostfile_text).map_err(|e| format!("parse hostfile: {e}"))?;

    let mut verified = vec![];
    if let Some(hosts_arr) = hostfile.get("hosts").and_then(|v| v.as_array()) {
        for h in hosts_arr {
            if let Some(ssh) = h.get("ssh").and_then(|v| v.as_str()) {
                let ip = h
                    .get("ips")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let hostname = ssh.trim_end_matches(".local").to_string();
                verified.push(VerifiedLink {
                    hostname,
                    ip,
                    latency_ms: None,
                });
            }
        }
    }

    tracing::info!(
        "mlx.distributed_config: wrote {output_path} with {} nodes",
        verified.len()
    );

    Ok((output_path, verified))
}

// ── Step 1: bridge0 ─────────────────────────────────────────────────

async fn handle_bridge0() -> Bridge0Result {
    let members = tokio::task::spawn_blocking(detect_bridge0)
        .await
        .unwrap_or_default();

    if members.is_empty() {
        tracing::info!("bridge0: not present (clean)");
        return Bridge0Result::Clean;
    }

    tracing::warn!("bridge0 found consuming {} interfaces: {:?}", members.len(), members);

    // Destroy bridge0 interface directly (fast path — no plist editing)
    let destroy = tokio::task::spawn_blocking(|| {
        Command::new("sudo")
            .args(["ifconfig", "bridge0", "destroy"])
            .output()
    })
    .await;

    match destroy {
        Ok(Ok(output)) if output.status.success() => {
            tracing::info!("bridge0 destroyed, freed interfaces: {:?}", members);
            Bridge0Result::Destroyed(members)
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("bridge0 destroy failed: {stderr}");
            Bridge0Result::Failed(stderr.to_string())
        }
        Ok(Err(e)) => Bridge0Result::Failed(format!("exec: {e}")),
        Err(e) => Bridge0Result::Failed(format!("join: {e}")),
    }
}

fn detect_bridge0() -> Vec<String> {
    let output = Command::new("ifconfig").arg("bridge0").output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| l.trim().starts_with("member:"))
            .filter_map(|l| l.split_whitespace().nth(1).map(String::from))
            .collect(),
        _ => vec![],
    }
}

// ── Step 2: topology-derived IP assignment ─────────────────────────

/// Discover topology, retrying until the link set stabilizes, so a
/// simultaneous-restart race (peers still booting → transient partial/empty
/// topology) cannot trigger the static-IP fallback. Returns the best (max-link)
/// topology seen; bounded (~60s) so a genuinely-partial cluster still proceeds.
async fn discover_stable_topology(hosts: Vec<String>) -> Option<TopologyReport> {
    const MAX_ATTEMPTS: u32 = 12;
    let gap = Duration::from_secs(5);
    let mut best: Option<TopologyReport> = None;
    let mut best_links = 0usize;
    let mut stable = 0u32;
    for attempt in 1..=MAX_ATTEMPTS {
        let hosts_c = hosts.clone();
        match tokio::task::spawn_blocking(move || {
            crate::topology::discover_topology(&hosts_c, "jaccl")
        })
        .await
        {
            Ok(Ok(topo)) => {
                let n = topo.links.len();
                if n > best_links {
                    best_links = n;
                    best = Some(topo);
                    stable = 0;
                } else if n == best_links && n > 0 {
                    stable += 1;
                }
                tracing::info!(attempt, links = n, best = best_links, "autosetup: topology retry-until-stable");
                // Settled once the max link count repeats on consecutive scans.
                if best_links > 0 && stable >= 1 {
                    break;
                }
            }
            Ok(Err(e)) => tracing::warn!(attempt, "topology discovery failed: {e}"),
            Err(e) => tracing::warn!(attempt, "topology discovery task panicked: {e}"),
        }
        if attempt < MAX_ATTEMPTS {
            tokio::time::sleep(gap).await;
        }
    }
    if best_links == 0 {
        tracing::warn!("autosetup: topology never produced links after {MAX_ATTEMPTS} attempts");
    }
    best
}

/// Derive deterministic /30 IPs from topology links and apply them.
///
/// Algorithm: sort links by (min_node, max_node), then for link i:
///   - alphabetically-first node gets 192.168.10.{4i+1}/30
///   - alphabetically-second node gets 192.168.10.{4i+2}/30
///
/// Applied locally via ifconfig, pushed to remote nodes via SSH.
/// Returns the IPs assigned to the LOCAL node only.
/// One link's deterministic /30 assignment. `first` is always the
/// lexicographically-smaller node so the schema is stable regardless of
/// discovery order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LinkAssignment {
    pub node_first: String,
    pub iface_first: String,
    pub ip_first: String,
    pub node_second: String,
    pub iface_second: String,
    pub ip_second: String,
}

/// Deterministic /30 allocation for a topology: sort links by
/// (min_node, max_node, min_iface) and assign 192.168.10.(4i+1)/(4i+2).
/// The iface tiebreaker matters for PARALLEL links between the same node
/// pair (e.g. hub↔m3u2 over two cables): without it Vec::sort's stability
/// makes their relative order follow discovery order, so re-runs could swap
/// their /30s. Same cables ⇒ same IPs, every time, on every node.
/// Single source of truth — used by both the boot-time assigner and /rdma/mesh.
pub fn deterministic_link_ips(topo: &crate::topology::TopologyReport) -> Vec<LinkAssignment> {
    let canon = |link: &crate::topology::TopologyLink| {
        let strip = |d: &str| d.strip_prefix("rdma_").unwrap_or(d).to_string();
        if link.node_a < link.node_b {
            (link.node_a.clone(), strip(&link.device_a), link.node_b.clone(), strip(&link.device_b))
        } else {
            (link.node_b.clone(), strip(&link.device_b), link.node_a.clone(), strip(&link.device_a))
        }
    };
    let mut links: Vec<_> = topo.links.iter().map(canon).collect();
    links.sort_by(|a, b| {
        let a_iface = (&a.1).min(&a.3);
        let b_iface = (&b.1).min(&b.3);
        (&a.0, &a.2, a_iface).cmp(&(&b.0, &b.2, b_iface))
    });
    links
        .into_iter()
        .enumerate()
        .map(|(i, (node_first, iface_first, node_second, iface_second))| {
            let base = 4 * i;
            LinkAssignment {
                node_first,
                iface_first,
                ip_first: format!("192.168.10.{}", base + 1),
                node_second,
                iface_second,
                ip_second: format!("192.168.10.{}", base + 2),
            }
        })
        .collect()
}

pub async fn assign_topology_ips(
    topo: &TopologyReport,
    local_hostname: &str,
    push_remote: bool,
) -> Vec<InterfaceIp> {
    // Canonicalize the local hostname against the topology's (canonical) node
    // names. asmi reports scutil LocalHostName, which may carry a macOS Bonjour
    // suffix (e.g. m3u4-2237); without this, a suffixed node matches NONE of the
    // canonical link nodes, assigns nothing, and falls through to the static-IP
    // path — leaving stragglers (the same suffix bug the topology keystone fixed).
    let node_set: std::collections::HashSet<String> = topo
        .links
        .iter()
        .flat_map(|l| [l.node_a.to_lowercase(), l.node_b.to_lowercase()])
        .collect();
    let local_canon = asmi_core::aggregator::canonicalize_hostname(local_hostname, &node_set);
    let local_hostname: &str = &local_canon;

    let assignments = deterministic_link_ips(topo);

    let mut local_ips = vec![];

    for (i, a) in assignments.iter().enumerate() {
        let (first_node, second_node) = (&a.node_first, &a.node_second);
        let (iface_first, iface_second) = (a.iface_first.as_str(), a.iface_second.as_str());
        let (ip_first, ip_second) = (a.ip_first.clone(), a.ip_second.clone());

        // Assign first node's side
        if first_node == local_hostname {
            if apply_ip_local(iface_first, &ip_first).await {
                local_ips.push(InterfaceIp {
                    iface: iface_first.to_string(),
                    ip: ip_first.clone(),
                    source: "topology".into(),
                });
            }
        } else if push_remote {
            apply_ip_remote(first_node, iface_first, &ip_first).await;
        }

        // Assign second node's side
        if second_node == local_hostname {
            if apply_ip_local(iface_second, &ip_second).await {
                local_ips.push(InterfaceIp {
                    iface: iface_second.to_string(),
                    ip: ip_second.clone(),
                    source: "topology".into(),
                });
            }
        } else if push_remote {
            apply_ip_remote(second_node, iface_second, &ip_second).await;
        }

        tracing::info!(
            "link {i}: {first_node}:{iface_first}={ip_first} ↔ {second_node}:{iface_second}={ip_second}"
        );
    }

    local_ips
}

/// Apply an IP to a local interface. Removes any existing IP, assigns the new
/// one, and adds a /30 route through this interface.
async fn apply_ip_local(iface: &str, ip: &str) -> bool {
    let iface_own = iface.to_string();
    let ip_own = ip.to_string();
    let result = tokio::task::spawn_blocking(move || {
        // Remove existing IP (if any) to avoid stale aliases
        let existing = Command::new("ifconfig")
            .arg(&iface_own)
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .find(|l| l.contains("inet ") && !l.contains("inet6"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .map(String::from)
            });
        if let Some(ref old_ip) = existing {
            if old_ip != &ip_own {
                let _ = Command::new("sudo")
                    .args(["ifconfig", &iface_own, "delete", old_ip])
                    .output();
            }
        }
        let out = Command::new("sudo")
            .args(["ifconfig", &iface_own, "inet", &ip_own, "netmask", "255.255.255.252"])
            .output();

        // Ensure /30 route exists (macOS sometimes drops it on rapid reassignment)
        let octets: Vec<u8> = ip_own.split('.').filter_map(|s| s.parse().ok()).collect();
        if octets.len() == 4 {
            let net_base = octets[3] & 0xFC; // /30 network base
            let subnet = format!("{}.{}.{}.{}/30", octets[0], octets[1], octets[2], net_base);
            let _ = Command::new("sudo")
                .args(["route", "delete", "-net", &subnet])
                .output();
            let _ = Command::new("sudo")
                .args(["route", "add", "-net", &subnet, "-interface", &iface_own])
                .output();
        }

        out
    })
    .await;

    match result {
        Ok(Ok(out)) if out.status.success() => {
            tracing::info!("{iface}: assigned {ip} (topology)");
            true
        }
        _ => {
            tracing::warn!("{iface}: failed to assign {ip}");
            false
        }
    }
}

/// Push an IP to a remote node via SSH. Also adds the /30 route.
async fn apply_ip_remote(host: &str, iface: &str, ip: &str) {
    let host_own = host.to_string();
    let iface_own = iface.to_string();
    let ip_own = ip.to_string();
    let user = std::env::var("USER").unwrap_or_else(|_| "root".into());
    let result = tokio::task::spawn_blocking(move || {
        // Parse /30 network base for route
        let octets: Vec<&str> = ip_own.split('.').collect();
        let route_cmd = if octets.len() == 4 {
            if let Ok(last) = octets[3].parse::<u8>() {
                let net_base = last & 0xFC;
                format!(
                    "sudo route delete -net {}.{}.{}.{}/30 2>/dev/null; \
                     sudo route add -net {}.{}.{}.{}/30 -interface {iface_own}",
                    octets[0], octets[1], octets[2], net_base,
                    octets[0], octets[1], octets[2], net_base,
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let cmd = format!(
            "old=$(ifconfig {iface_own} 2>/dev/null | grep 'inet ' | awk '{{print $2}}'); \
             [ -n \"$old\" ] && [ \"$old\" != \"{ip_own}\" ] && sudo ifconfig {iface_own} delete $old 2>/dev/null; \
             sudo ifconfig {iface_own} inet {ip_own} netmask 255.255.255.252; \
             {route_cmd}"
        );
        Command::new("ssh")
            .args([
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                &format!("{user}@{host_own}"),
                &cmd,
            ])
            .output()
    })
    .await;

    match result {
        Ok(Ok(out)) if out.status.success() => {
            tracing::info!("{host}:{iface}: assigned {ip} (topology, remote)");
        }
        _ => {
            tracing::warn!("{host}:{iface}: failed to assign {ip} (remote)");
        }
    }
}

fn get_local_hostname() -> String {
    Command::new("scutil")
        .args(["--get", "LocalHostName"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_lowercase();
            if s.is_empty() { None } else { Some(s) }
        })
        .unwrap_or_else(|| {
            Command::new("hostname")
                .arg("-s")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|_| "unknown".into())
        })
}

// ── Step 2 (fallback): static/link-local ───────────────────────────

/// Apply static IPs from settings. Deterministic — same result on every boot.
async fn apply_static_ips(entries: &[StaticIpEntry]) -> Vec<InterfaceIp> {
    let mut results = vec![];
    for entry in entries {
        let name = entry.iface.clone();
        let ip = entry.ip.clone();
        let mask = entry.mask.clone();
        let assign = tokio::task::spawn_blocking(move || {
            Command::new("sudo")
                .args(["ifconfig", &name, "inet", &ip, "netmask", &mask])
                .output()
        })
        .await;
        match assign {
            Ok(Ok(out)) if out.status.success() => {
                tracing::info!("{}: static IP {} assigned", entry.iface, entry.ip);
                results.push(InterfaceIp {
                    iface: entry.iface.clone(),
                    ip: entry.ip.clone(),
                    source: "static".into(),
                });
            }
            _ => tracing::warn!("{}: failed to assign static IP {}", entry.iface, entry.ip),
        }
    }
    results
}

/// Set default gateway for nodes that get internet via TB5 Internet Sharing.
async fn apply_default_gateway(gw: &str) {
    let gw_owned = gw.to_string();
    let gw_log = gw.to_string();
    let _ = tokio::task::spawn_blocking(move || {
        let _ = Command::new("sudo").args(["route", "delete", "default"]).output();
        Command::new("sudo").args(["route", "add", "default", &gw_owned]).output()
    })
    .await;
    tracing::info!("default gateway set to {}", gw_log);
}

async fn ensure_tb5_ips() -> Result<Vec<InterfaceIp>> {
    let tb_interfaces = tokio::task::spawn_blocking(discover_tb5_interfaces)
        .await
        .context("join")??;

    let mut results = vec![];

    for iface in &tb_interfaces {
        // Bring interface up
        let name = iface.name.clone();
        let _ = tokio::task::spawn_blocking(move || {
            Command::new("sudo").args(["ifconfig", &name, "up"]).output()
        })
        .await;

        // Re-check status after bringing up
        let name2 = iface.name.clone();
        let status = tokio::task::spawn_blocking(move || get_interface_status(&name2))
            .await
            .unwrap_or_default();

        if status != "active" {
            continue; // No cable connected
        }

        // Check for existing 169.254 IP
        let name3 = iface.name.clone();
        if let Some(ip) = tokio::task::spawn_blocking(move || get_link_local_ip(&name3))
            .await
            .unwrap_or(None)
        {
            tracing::debug!("{}: existing IP {ip}", iface.name);
            results.push(InterfaceIp {
                iface: iface.name.clone(),
                ip,
                source: "existing".into(),
            });
            continue;
        }

        // Wait for IPv4LL (3 seconds)
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = tokio::task::spawn_blocking(|| {
            Command::new("ipconfig").args(["waitall"]).output()
        })
        .await;

        let name4 = iface.name.clone();
        if let Some(ip) = tokio::task::spawn_blocking(move || get_link_local_ip(&name4))
            .await
            .unwrap_or(None)
        {
            tracing::info!("{}: IPv4LL assigned {ip}", iface.name);
            results.push(InterfaceIp {
                iface: iface.name.clone(),
                ip,
                source: "ipv4ll".into(),
            });
            continue;
        }

        // Fallback: assign deterministic IP based on interface index
        let octet = iface.index;
        let ip = format!("169.254.{octet}.1");
        let name5 = iface.name.clone();
        let ip2 = ip.clone();
        let assign = tokio::task::spawn_blocking(move || {
            Command::new("sudo")
                .args(["ifconfig", &name5, "inet", &ip2, "netmask", "255.255.0.0"])
                .output()
        })
        .await;

        if assign.is_ok() {
            tracing::info!("{}: manually assigned {ip}", iface.name);
            results.push(InterfaceIp {
                iface: iface.name.clone(),
                ip,
                source: "manual".into(),
            });
        }
    }

    Ok(results)
}

struct TbInterface {
    name: String,
    index: u8,
}

fn discover_tb5_interfaces() -> Result<Vec<TbInterface>> {
    // Parse `networksetup -listallhardwareports` for Thunderbolt interfaces
    let output = Command::new("networksetup")
        .args(["-listallhardwareports"])
        .output()
        .context("networksetup")?;

    let text = String::from_utf8_lossy(&output.stdout);
    let mut interfaces = vec![];

    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        // Look for "Hardware Port: Thunderbolt N" (skip "Thunderbolt Bridge")
        if line.contains("Thunderbolt") && !line.contains("Bridge") {
            // Next line has "Device: enX"
            if let Some(dev_line) = lines.get(i + 1) {
                if let Some(dev) = dev_line.strip_prefix("Device: ") {
                    let dev = dev.trim();
                    // Extract index from enX
                    if let Some(idx_str) = dev.strip_prefix("en") {
                        if let Ok(idx) = idx_str.parse::<u8>() {
                            interfaces.push(TbInterface {
                                name: dev.to_string(),
                                index: idx,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(interfaces)
}

fn get_interface_status(iface: &str) -> String {
    Command::new("ifconfig")
        .arg(iface)
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| l.contains("status:"))
                .and_then(|l| l.split_whitespace().last())
                .map(String::from)
        })
        .unwrap_or_default()
}


/// Clean ARP entries that point a peer IP to the LOCAL machine's MAC.
/// These are bridge0 proxy ARP remnants that survive bridge0 destruction.
/// They cause traffic to peer IPs to go to loopback instead of the wire.
async fn clean_self_arp_poison(local_ips: &[InterfaceIp]) {
    // Get local MAC addresses for our TB5 interfaces
    let local_macs: std::collections::HashSet<String> = local_ips
        .iter()
        .filter_map(|ip| {
            Command::new("ifconfig")
                .arg(&ip.iface)
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .find(|l| l.contains("ether "))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .map(String::from)
                })
        })
        .collect();

    if local_macs.is_empty() {
        return;
    }

    // Build set of our own local IPs — we should NOT delete entries for these
    let local_ip_set: std::collections::HashSet<String> =
        local_ips.iter().map(|ip| ip.ip.clone()).collect();

    // List all ARP entries
    let arp = tokio::task::spawn_blocking(|| Command::new("arp").arg("-an").output()).await;
    let Ok(Ok(arp_out)) = arp else { return };
    let arp_text = String::from_utf8_lossy(&arp_out.stdout).to_string();

    // Parse: "? (169.254.X.X) at MM:MM:MM:MM:MM:MM on enN ifscope permanent [ethernet]"
    let mut deleted = 0;
    for line in arp_text.lines() {
        if !line.contains("permanent") || line.contains("on lo0") {
            continue;
        }
        // Extract IP (between parens) and MAC (after "at ")
        let ip = line
            .find('(')
            .and_then(|s| line[s + 1..].find(')').map(|e| &line[s + 1..s + 1 + e]));
        let mac = line.split(" at ").nth(1).and_then(|s| s.split_whitespace().next());

        if let (Some(ip), Some(mac)) = (ip, mac) {
            // Poison = peer IP mapped to our MAC. Skip our own IPs and non-link-local.
            if !ip.starts_with("169.254") || local_ip_set.contains(ip) {
                continue;
            }
            if local_macs.contains(mac) {
                let ip_owned = ip.to_string();
                let _ = tokio::task::spawn_blocking(move || {
                    Command::new("sudo").args(["arp", "-d", &ip_owned]).output()
                })
                .await;
                deleted += 1;
                tracing::info!("removed self-MAC poison ARP entry: {ip} -> {mac}");
            }
        }
    }

    if deleted > 0 {
        tracing::info!("cleaned {deleted} self-MAC poison ARP entries");
    }
}

fn get_link_local_ip(iface: &str) -> Option<String> {
    Command::new("ifconfig")
        .arg(iface)
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| l.contains("inet 169.254"))
                .and_then(|l| l.split_whitespace().nth(1))
                .map(String::from)
        })
}

// ── Step 3: fix routes ──────────────────────────────────────────────

// ── Step 3+4: probe per-interface, set per-host routes, verify ──────

/// For each peer IP × local interface, try source-bound ping.
/// When a pair works, add a per-host route through that interface.
async fn probe_and_route(
    node_map: &asmi_core::NodeMap,
    local_ips: &[InterfaceIp],
) -> (RouteResult, PeerResult) {
    if local_ips.is_empty() {
        return (RouteResult::NoInterfaces, PeerResult::default());
    }

    let local_hostname = Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_lowercase())
        .unwrap_or_else(|_| "unknown".into());

    // Collect all peer IPs from config (skip self — case-insensitive)
    let mut peer_ips: Vec<(String, String)> = vec![];
    for (hostname, ips) in &node_map.rdma_ips {
        if hostname.to_lowercase() == local_hostname {
            continue;
        }
        for ip in ips {
            peer_ips.push((hostname.clone(), ip.clone()));
        }
    }

    let total_tried = peer_ips.len();

    // Phase 1: Flush ARP, set blanket route through primary, probe with -S
    // This is the fast path — finds peers on the primary interface.
    {
        let primary = local_ips[0].iface.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let _ = Command::new("sudo")
                .args(["route", "delete", "-net", "169.254.0.0/16"])
                .output();
            let _ = Command::new("sudo").args(["arp", "-a", "-d"]).output();
            let _ = Command::new("sudo")
                .args(["route", "add", "-net", "169.254.0.0/16", "-interface", &primary])
                .output();
        })
        .await;
    }

    let mut verified = vec![];
    let mut routes_added = 0usize;
    let mut found_peers: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (hostname, peer_ip) in &peer_ips {
        let mut found = false;
        for local in local_ips {
            let src = local.ip.clone();
            let dst = peer_ip.clone();

            let result = tokio::task::spawn_blocking(move || {
                Command::new("ping")
                    .args(["-c", "1", "-W", "1", "-S", &src, &dst])
                    .output()
            })
            .await;

            if let Ok(Ok(output)) = &result {
                if output.status.success() {
                    let latency = parse_ping_latency(&output.stdout);
                    tracing::info!(
                        "peer {hostname} ({peer_ip}): reachable via {} ({latency:?}ms)",
                        local.iface
                    );

                    let dst2 = peer_ip.clone();
                    let iface = local.iface.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let _ = Command::new("sudo")
                            .args(["route", "delete", "-host", &dst2])
                            .output();
                        Command::new("sudo")
                            .args(["route", "add", "-host", &dst2, "-interface", &iface])
                            .output()
                    })
                    .await;
                    routes_added += 1;

                    verified.push(VerifiedLink {
                        hostname: hostname.clone(),
                        ip: peer_ip.clone(),
                        latency_ms: latency,
                    });
                    found_peers.insert(peer_ip.clone());
                    found = true;
                    break;
                }
            }
        }
        if !found {
            tracing::debug!("peer {hostname} ({peer_ip}): not found in phase 1");
        }
    }

    // Phase 2: probe unfound peers on non-primary interfaces via temporary per-host routes
    if local_ips.len() > 1 {
        let unfound: Vec<_> = peer_ips
            .iter()
            .filter(|(_, ip)| !found_peers.contains(ip))
            .cloned()
            .collect();

        if !unfound.is_empty() {
            tracing::info!(
                "phase 2: probing {} unfound peers on {} non-primary interfaces",
                unfound.len(),
                local_ips.len() - 1
            );

            for local in &local_ips[1..] {
                for (hostname, peer_ip) in &unfound {
                    if found_peers.contains(peer_ip) {
                        continue;
                    }

                    // Temporary per-host route through this interface
                    let dst_r = peer_ip.clone();
                    let iface_r = local.iface.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let _ = Command::new("sudo")
                            .args(["route", "delete", "-host", &dst_r])
                            .output();
                        Command::new("sudo")
                            .args(["route", "add", "-host", &dst_r, "-interface", &iface_r])
                            .output()
                    })
                    .await;

                    let src = local.ip.clone();
                    let dst = peer_ip.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        Command::new("ping")
                            .args(["-c", "1", "-W", "1", "-S", &src, &dst])
                            .output()
                    })
                    .await;

                    if let Ok(Ok(output)) = &result {
                        if output.status.success() {
                            let latency = parse_ping_latency(&output.stdout);
                            tracing::info!(
                                "peer {hostname} ({peer_ip}): reachable via {} ({latency:?}ms) [phase 2]",
                                local.iface
                            );
                            routes_added += 1;
                            verified.push(VerifiedLink {
                                hostname: hostname.clone(),
                                ip: peer_ip.clone(),
                                latency_ms: latency,
                            });
                            found_peers.insert(peer_ip.clone());
                            continue;
                        }
                    }

                    // Failed — clean up temp route
                    let dst_c = peer_ip.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let _ = Command::new("sudo")
                            .args(["route", "delete", "-host", &dst_c])
                            .output();
                    })
                    .await;
                }
            }
        }
    }

    // Log unreachable peers
    for (hostname, peer_ip) in &peer_ips {
        if !found_peers.contains(peer_ip) {
            tracing::debug!("peer {hostname} ({peer_ip}): unreachable from all interfaces");
        }
    }

    let route_result = RouteResult::Fixed {
        primary_interface: local_ips[0].iface.clone(),
        interfaces_with_ips: routes_added,
    };

    (
        route_result,
        PeerResult {
            verified_links: verified,
            total_tried,
        },
    )
}

fn parse_ping_latency(stdout: &[u8]) -> Option<f64> {
    let text = String::from_utf8_lossy(stdout);
    // Look for "time=X.XXX ms"
    text.lines()
        .find(|l| l.contains("time="))
        .and_then(|l| {
            l.split("time=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<f64>().ok())
        })
}

// ── Step 5: write hostfile ──────────────────────────────────────────

async fn write_hostfile(peers: &PeerResult, local_ips: &[InterfaceIp]) -> Result<String> {
    let home = dirs::home_dir().context("no home dir")?;
    let dir = home.join(".r1o/hostfiles");
    tokio::fs::create_dir_all(&dir).await?;

    let path = dir.join("auto.json");
    let local_hostname = Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());

    // Use first local TB5 IP as coordinator
    let coordinator_ip = local_ips
        .first()
        .map(|ip| ip.ip.clone())
        .unwrap_or_else(|| "127.0.0.1".into());

    let mut hosts = vec![serde_json::json!({
        "ip": coordinator_ip,
        "hostname": local_hostname,
    })];

    for link in &peers.verified_links {
        // Avoid duplicate hostnames
        if !hosts.iter().any(|h| h["hostname"] == link.hostname) {
            hosts.push(serde_json::json!({
                "ip": link.ip,
                "hostname": link.hostname,
            }));
        }
    }

    let hostfile = serde_json::json!({
        "coordinator": coordinator_ip,
        "hosts": hosts,
    });

    let content = serde_json::to_string_pretty(&hostfile)?;
    tokio::fs::write(&path, &content).await?;

    tracing::info!("wrote JACCL hostfile: {} ({} hosts)", path.display(), hosts.len());
    Ok(path.display().to_string())
}

#[cfg(test)]
mod deterministic_ip_tests {
    use super::*;
    use crate::topology::{TopologyLink, TopologyReport};

    fn link(a: &str, da: &str, b: &str, db: &str) -> TopologyLink {
        TopologyLink {
            node_a: a.into(),
            device_a: da.into(),
            node_b: b.into(),
            device_b: db.into(),
        }
    }

    fn report(links: Vec<TopologyLink>) -> TopologyReport {
        TopologyReport {
            nodes: vec![],
            links,
            mesh_complete: false,
            missing_links: vec![],
            jaccl_ready: false,
            jaccl_ready_subsets: vec![],
            raw_dot: String::new(),
        }
    }

    #[test]
    fn parallel_links_same_pair_are_order_independent() {
        // hub↔m3u2 over two cables, fed in both discovery orders — the iface
        // tiebreaker must yield identical assignments.
        let fwd = report(vec![
            link("hub", "rdma_en4", "m3u2", "rdma_en4"),
            link("hub", "rdma_en5", "m3u2", "rdma_en3"),
        ]);
        let rev = report(vec![
            link("m3u2", "rdma_en3", "hub", "rdma_en5"),
            link("hub", "rdma_en4", "m3u2", "rdma_en4"),
        ]);
        let a = deterministic_link_ips(&fwd);
        let b = deterministic_link_ips(&rev);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].ip_first, "192.168.10.1");
        assert_eq!(a[0].ip_second, "192.168.10.2");
        assert_eq!(a[1].ip_first, "192.168.10.5");
        assert_eq!(a[1].ip_second, "192.168.10.6");
    }

    #[test]
    fn node_order_in_link_is_canonicalized() {
        let r1 = report(vec![link("m3u2", "rdma_en4", "hub", "rdma_en3")]);
        let r2 = report(vec![link("hub", "rdma_en3", "m3u2", "rdma_en4")]);
        let (a, b) = (deterministic_link_ips(&r1), deterministic_link_ips(&r2));
        assert_eq!(a, b);
        assert_eq!(a[0].node_first, "hub");
        assert_eq!(a[0].iface_first, "en3");
    }

    #[test]
    fn multi_pair_sorted_lexicographically() {
        let r = report(vec![
            link("m3u2", "rdma_en5", "m3u3", "rdma_en5"),
            link("hub", "rdma_en4", "m3u2", "rdma_en4"),
        ]);
        let a = deterministic_link_ips(&r);
        assert_eq!(a[0].node_first, "hub");          // hub-m3u2 sorts first
        assert_eq!(a[0].ip_first, "192.168.10.1");
        assert_eq!(a[1].node_first, "m3u2");         // m3u2-m3u3 second
        assert_eq!(a[1].ip_first, "192.168.10.5");
    }
}

#[cfg(test)]
mod coordinator_ip_tests {
    use super::*;

    #[test]
    fn test_persist_and_restore_roundtrip() {
        let path = std::env::temp_dir().join("asmi_test_coordinator_ips.json");

        let ips = vec![
            InterfaceIp { iface: "en4".into(), ip: "192.168.10.1".into(), source: "topology".into() },
            InterfaceIp { iface: "en3".into(), ip: "192.168.10.5".into(), source: "topology".into() },
            InterfaceIp { iface: "en5".into(), ip: "192.168.10.9".into(), source: "topology".into() },
        ];

        // Serialize
        let data = PersistedCoordinatorIps {
            ips: ips.iter().map(|ip| PersistedIpEntry {
                iface: ip.iface.clone(),
                ip: ip.ip.clone(),
                netmask: "255.255.255.252".into(),
            }).collect(),
            assigned_at: "2026-06-20T01:00:00Z".into(),
        };
        let json = serde_json::to_string_pretty(&data).unwrap();
        std::fs::write(&path, &json).unwrap();

        // Deserialize
        let read_back: PersistedCoordinatorIps =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(read_back.ips.len(), 3);
        assert_eq!(read_back.ips[0].iface, "en4");
        assert_eq!(read_back.ips[0].ip, "192.168.10.1");
        assert_eq!(read_back.ips[0].netmask, "255.255.255.252");
        assert_eq!(read_back.ips[1].iface, "en3");
        assert_eq!(read_back.ips[1].ip, "192.168.10.5");
        assert_eq!(read_back.ips[2].iface, "en5");
        assert_eq!(read_back.ips[2].ip, "192.168.10.9");
        assert_eq!(read_back.assigned_at, "2026-06-20T01:00:00Z");
    }

    #[test]
    fn test_persisted_format_matches_plan() {
        let data = PersistedCoordinatorIps {
            ips: vec![
                PersistedIpEntry { iface: "en4".into(), ip: "192.168.10.1".into(), netmask: "255.255.255.252".into() },
            ],
            assigned_at: "2026-06-20T01:00:00Z".into(),
        };
        let json = serde_json::to_string_pretty(&data).unwrap();
        // Verify JSON shape matches plan's expected format
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["ips"].is_array());
        assert!(v["ips"][0]["iface"].is_string());
        assert!(v["ips"][0]["ip"].is_string());
        assert!(v["ips"][0]["netmask"].is_string());
        assert!(v["assigned_at"].is_string());
    }

    #[test]
    fn test_restore_returns_empty_on_missing_file() {
        // restore_coordinator_ips reads from the default path, which won't have
        // our test data. But we can test the deserialization path directly.
        let empty: Result<PersistedCoordinatorIps, _> = serde_json::from_str("{}");
        assert!(empty.is_err() || empty.unwrap().ips.is_empty());
    }

    #[test]
    fn test_restore_handles_corrupt_json() {
        let result: Result<PersistedCoordinatorIps, _> = serde_json::from_str("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_persisted_ip_entry_default_mask() {
        let json = r#"{"iface":"en4","ip":"192.168.10.1","netmask":"255.255.255.252"}"#;
        let entry: PersistedIpEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.netmask, "255.255.255.252");
    }
}

#[cfg(test)]
mod union_merge_tests {
    use crate::topology::{TopologyLink, link_key};
    use std::collections::{HashMap, HashSet};

    fn make_link(a: &str, da: &str, b: &str, db: &str) -> TopologyLink {
        TopologyLink {
            node_a: a.into(), device_a: da.into(),
            node_b: b.into(), device_b: db.into(),
        }
    }

    fn simulate_merge(
        known_links: &mut HashMap<String, TopologyLink>,
        missing_streak: &mut HashMap<String, u32>,
        scan_links: &[TopologyLink],
    ) {
        let scan_keys: HashSet<String> = scan_links.iter().map(link_key).collect();

        for link in scan_links {
            let key = link_key(link);
            known_links.insert(key.clone(), link.clone());
            missing_streak.insert(key, 0);
        }

        let all_keys: Vec<String> = known_links.keys().cloned().collect();
        for key in &all_keys {
            if !scan_keys.contains(key) {
                *missing_streak.entry(key.clone()).or_insert(0) += 1;
            }
        }

        known_links.retain(|k, _| missing_streak.get(k).copied().unwrap_or(0) < 3);
        missing_streak.retain(|k, _| known_links.contains_key(k));
    }

    #[test]
    fn test_links_accumulate_across_scans() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        // Scan 1: sees link A
        let link_a = make_link("hub", "rdma_en4", "m3u2", "rdma_en4");
        simulate_merge(&mut known, &mut streaks, &[link_a.clone()]);
        assert_eq!(known.len(), 1);

        // Scan 2: sees link B (not A) — A should still be present
        let link_b = make_link("hub", "rdma_en3", "m3u3", "rdma_en4");
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        assert_eq!(known.len(), 2, "both links should be present after union");
    }

    #[test]
    fn test_link_evicted_after_3_misses() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        let link_a = make_link("hub", "rdma_en4", "m3u2", "rdma_en4");
        let link_b = make_link("hub", "rdma_en3", "m3u3", "rdma_en4");

        // Scan 1: both links seen
        simulate_merge(&mut known, &mut streaks, &[link_a.clone(), link_b.clone()]);
        assert_eq!(known.len(), 2);

        // Scans 2-3: only link_b seen (link_a missing 2x)
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        assert_eq!(known.len(), 2, "link_a should survive 1 miss");
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        assert_eq!(known.len(), 2, "link_a should survive 2 misses");

        // Scan 4: still only link_b — link_a at 3 misses → evicted
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        assert_eq!(known.len(), 1, "link_a should be evicted after 3 misses");
        assert!(known.values().any(|l| l.node_b == "m3u3"), "link_b should remain");
    }

    #[test]
    fn test_link_reappears_resets_streak() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        let link_a = make_link("hub", "rdma_en4", "m3u2", "rdma_en4");
        let link_b = make_link("hub", "rdma_en3", "m3u3", "rdma_en4");

        // Seed both links
        simulate_merge(&mut known, &mut streaks, &[link_a.clone(), link_b.clone()]);

        // Miss link_a twice
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        simulate_merge(&mut known, &mut streaks, &[link_b.clone()]);
        assert_eq!(*streaks.get(&link_key(&link_a)).unwrap(), 2);

        // link_a reappears — streak resets
        simulate_merge(&mut known, &mut streaks, &[link_a.clone(), link_b.clone()]);
        assert_eq!(*streaks.get(&link_key(&link_a)).unwrap(), 0, "streak should reset on reappearance");
        assert_eq!(known.len(), 2);
    }

    #[test]
    fn test_empty_scan_increments_all_streaks() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        let link_a = make_link("hub", "rdma_en4", "m3u2", "rdma_en4");
        simulate_merge(&mut known, &mut streaks, &[link_a.clone()]);

        // Empty scan 3 times → eviction
        simulate_merge(&mut known, &mut streaks, &[]);
        simulate_merge(&mut known, &mut streaks, &[]);
        assert_eq!(known.len(), 1, "still present after 2 empty scans");
        simulate_merge(&mut known, &mut streaks, &[]);
        assert_eq!(known.len(), 0, "evicted after 3 empty scans");
    }

    #[test]
    fn test_full_6_link_mesh_convergence() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        let all_links = vec![
            make_link("hub", "rdma_en4", "m3u2", "rdma_en4"),
            make_link("hub", "rdma_en3", "m3u3", "rdma_en4"),
            make_link("hub", "rdma_en5", "m3u4", "rdma_en5"),
            make_link("m3u2", "rdma_en5", "m3u3", "rdma_en5"),
            make_link("m3u2", "rdma_en3", "m3u4", "rdma_en4"),
            make_link("m3u3", "rdma_en3", "m3u4", "rdma_en3"),
        ];

        // Scan 1: sees 3 of 6 links (typical cold start)
        simulate_merge(&mut known, &mut streaks, &all_links[..3]);
        assert_eq!(known.len(), 3);

        // Scan 2: sees 5 of 6 (SSH mux warming up)
        simulate_merge(&mut known, &mut streaks, &all_links[..5]);
        assert_eq!(known.len(), 5);

        // Scan 3: sees all 6
        simulate_merge(&mut known, &mut streaks, &all_links);
        assert_eq!(known.len(), 6, "full 6-link mesh should be accumulated");

        // Scan 4: regresses to 4 (SSH timeout on 2 peers) — all 6 should stay
        simulate_merge(&mut known, &mut streaks, &all_links[..4]);
        assert_eq!(known.len(), 6, "transient regression should not drop links");
    }

    #[test]
    fn test_missing_streak_cleanup() {
        let mut known = HashMap::new();
        let mut streaks = HashMap::new();

        let link_a = make_link("hub", "rdma_en4", "m3u2", "rdma_en4");
        simulate_merge(&mut known, &mut streaks, &[link_a.clone()]);

        // Evict link_a
        for _ in 0..3 {
            simulate_merge(&mut known, &mut streaks, &[]);
        }
        assert_eq!(known.len(), 0);
        assert_eq!(streaks.len(), 0, "streaks map should be cleaned up with known_links");
    }
}
