//! RDMA auto-setup: runs on daemon startup to ensure TB5 interfaces are ready.
//!
//! Sequence: detect bridge0 → destroy → assign IPs → fix routes → verify peers → write hostfile.
//! All steps are non-fatal — daemon starts regardless.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Duration;

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
    route_fix: bool,
}

fn default_true() -> bool { true }

impl Default for RdmaSettings {
    fn default() -> Self {
        Self {
            auto_setup: true,
            auto_destroy_bridge0: true,
            _ip_assignment: None,
            route_fix: true,
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
pub async fn autosetup(node_map: &tokio::sync::RwLock<asmi_core::NodeMap>) -> AutosetupReport {
    let settings = load_settings();
    let mut report = AutosetupReport::default();

    if !settings.auto_setup {
        tracing::info!("RDMA autosetup disabled in settings");
        return report;
    }

    // Step 1: bridge0 (still needed — mlx.distributed_config doesn't destroy it)
    if settings.auto_destroy_bridge0 {
        report.bridge0 = handle_bridge0().await;
    } else {
        tracing::info!("bridge0 auto-destroy disabled in settings");
    }

    // Step 2: ensure IPs on active TB5 interfaces (so we can report them)
    match ensure_tb5_ips().await {
        Ok(ips) => report.ips = ips,
        Err(e) => tracing::warn!("ensure_tb5_ips failed: {e}"),
    }

    // Step 2b: clean self-MAC poisoned ARP entries (bridge0 proxy ARP remnants)
    clean_self_arp_poison(&report.ips).await;

    // Step 3: delegate to mlx.distributed_config (Apple's official tool)
    // It discovers topology via SSH + system_profiler, configures 192.168.0.x /30
    // subnets per TB5 link, and generates the correct JACCL hostfile.
    // Only include nodes that have rdma_ips configured (have TB5 hardware).
    let hosts: Vec<String> = {
        let nm = node_map.read().await;
        nm.rdma_ips.keys().cloned().collect()
    };

    if !hosts.is_empty() {
        match run_mlx_distributed_config(&hosts).await {
            Ok((hostfile_path, verified_links)) => {
                let n_verified = verified_links.len();
                report.hostfile = Some(hostfile_path);
                // Preserve existing schema: populate routes + peers fields
                // consumed by r1o web/electron UI
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
                // Fallback: use our legacy probe
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
        // Pipe empty input to skip the "Enter to continue" prompts
        let mut child = std::process::Command::new(&mlx_config_bin)
            .args([
                "--hosts",
                &hosts_arg_clone,
                "--over",
                "thunderbolt",
                "--backend",
                "jaccl",
                "--auto-setup",
                "--output-hostfile",
                &output_path_clone,
                "--ignore-unreachable",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn mlx.distributed_config: {e}"))?;

        // Send newlines to pass through interactive prompts
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(b"\n\n\n\n\n\n\n\n\n\n");
        }

        child
            .wait_with_output()
            .map_err(|e| format!("wait mlx.distributed_config: {e}"))
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

// ── Step 2: ensure TB5 IPs ──────────────────────────────────────────

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

/// Ensure each Thunderbolt port (en1/en2/en3/en4/en5/en6) has its own
/// network service. macOS won't route traffic through interfaces that
/// only belong to the destroyed bridge0 — they need an explicit service.
async fn ensure_tb_network_services() {
    // Get list of active TB hardware ports and their interface names
    let ports = tokio::task::spawn_blocking(|| {
        Command::new("networksetup")
            .arg("-listallhardwareports")
            .output()
    })
    .await;

    let Ok(Ok(ports_out)) = ports else { return };
    let ports_text = String::from_utf8_lossy(&ports_out.stdout).to_string();

    // Parse: lines come in 3-line blocks: "Hardware Port: X", "Device: enN", "Ethernet Address: ..."
    let mut tb_devices: Vec<String> = vec![];
    let lines: Vec<&str> = ports_text.lines().collect();
    for i in 0..lines.len() {
        if lines[i].starts_with("Hardware Port:") && lines[i].contains("Thunderbolt")
            && !lines[i].contains("Bridge")
            && i + 1 < lines.len()
        {
            if let Some(dev) = lines[i + 1].strip_prefix("Device: ") {
                tb_devices.push(dev.trim().to_string());
            }
        }
    }

    // Get existing services
    let services = tokio::task::spawn_blocking(|| {
        Command::new("networksetup")
            .arg("-listnetworkserviceorder")
            .output()
    })
    .await;
    let Ok(Ok(services_out)) = services else { return };
    let services_text = String::from_utf8_lossy(&services_out.stdout).to_string();

    // For each TB device, check if it has a service; create if missing
    for dev in &tb_devices {
        let device_marker = format!("Device: {})", dev);
        if services_text.contains(&device_marker) {
            continue; // Already has a service
        }

        let service_name = format!("r1o TB {dev}");
        let dev_clone = dev.clone();
        let result = tokio::task::spawn_blocking(move || {
            Command::new("sudo")
                .args(["networksetup", "-createnetworkservice", &service_name, &dev_clone])
                .output()
        })
        .await;

        match &result {
            Ok(Ok(out)) if out.status.success() => {
                tracing::info!("created network service for {dev}");
            }
            Ok(Ok(out)) => {
                let err = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                tracing::warn!(
                    "failed to create service for {dev} (macOS may require System Settings > Network — manually create 'r1o TB {dev}' service): stderr={err} stdout={stdout}"
                );
            }
            _ => {
                tracing::warn!("failed to spawn networksetup for {dev}");
            }
        }
    }
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
mod tests {
    /// Live-cluster invariant: across all rdma-capable nodes, no two interfaces
    /// should have been assigned the same link-local IP by /rdma/setup.
    /// This is the bug filed at https://github.com/r1o-ai/asmi/issues/1 — the
    /// deterministic 169.254.{iface_index}.1 fallback produces collisions when
    /// two nodes have an interface at the same index (extremely common with TB5
    /// since en4 + en5 are the standard TB5 ports on M3 Ultra).
    #[tokio::test]
    async fn no_ip_collisions_across_cluster() {
        if std::env::var("ASMI_LIVE_CLUSTER_TEST").is_err() {
            eprintln!(
                "skipping no_ip_collisions_across_cluster — set ASMI_LIVE_CLUSTER_TEST=1 to run"
            );
            return;
        }

        let nodes = ["hub", "m3u1", "m3u3"];
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("reqwest client");

        let mut all_ips: Vec<(String, String)> = Vec::new();
        for node in nodes {
            let url = if node == "hub" {
                "http://localhost:9090/rdma/setup".to_string()
            } else {
                format!("http://{node}:9090/rdma/setup")
            };
            let resp: serde_json::Value = client
                .post(&url)
                .send()
                .await
                .unwrap_or_else(|e| panic!("POST {url} failed: {e}"))
                .json()
                .await
                .unwrap_or_else(|e| panic!("parse {url} body: {e}"));

            for ip_entry in resp["ips"].as_array().expect("ips array") {
                let ip = ip_entry["ip"].as_str().expect("ip string").to_string();
                all_ips.push((node.to_string(), ip));
            }
        }

        // Group by IP, fail if any IP appears with multiple distinct nodes.
        let mut by_ip: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (node, ip) in &all_ips {
            by_ip.entry(ip.clone()).or_default().push(node.clone());
        }

        let collisions: Vec<_> = by_ip
            .iter()
            .filter(|(_, nodes)| {
                let unique: std::collections::HashSet<_> = nodes.iter().collect();
                unique.len() > 1
            })
            .collect();

        assert!(
            collisions.is_empty(),
            "IP collisions detected across cluster (asmi#1): {:#?}\n\nFull IP list: {:#?}",
            collisions,
            all_ips
        );
    }
}
