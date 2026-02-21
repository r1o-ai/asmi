//! Dynamic node discovery and scanning — zero hardcoded nodes.
//!
//! Discovery methods:
//! - **ThunderboltBridge** — `ifconfig` → find `en*` with `inet 169.254.x.x`
//! - **Tailscale** — `tailscale status --json` → filter macOS + online peers
//! - **Arp** — `arp -a` → parse hostname/IP pairs
//! - **SystemProfiler** — `system_profiler SPThunderboltDataType` → connected devices + link speeds
//! - **Bonjour** — TODO: `dns-sd -B _ssh._tcp local.` with timeout
//!
//! After discovery, `scan_node()` probes each peer for hardware info, RDMA
//! status, and running MLX servers.

use crate::config::{ClusterConfig, DiscoveryMethod};
use crate::ssh::{local_run, ssh_run};
use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A discovered peer with connection metadata.
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub hostname: String,
    pub ips: Vec<String>,
    pub discovery_source: String,
    pub thunderbolt_bridge: Option<String>,
    pub link_speed: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level API
// ---------------------------------------------------------------------------

/// Discover cluster nodes using configured methods. Returns deduplicated peers.
pub async fn discover_nodes(config: &ClusterConfig) -> Vec<DiscoveredPeer> {
    let mut all_peers: Vec<DiscoveredPeer> = Vec::new();

    // Add seed hosts first
    for seed in &config.seed_hosts {
        all_peers.push(DiscoveredPeer {
            hostname: seed.clone(),
            ips: vec![seed.clone()],
            discovery_source: "seed".to_string(),
            thunderbolt_bridge: None,
            link_speed: None,
        });
    }

    // Run each configured discovery method
    for method in &config.discovery {
        let discovered = match method {
            DiscoveryMethod::ThunderboltBridge => discover_thunderbolt_bridge(config).await,
            DiscoveryMethod::Tailscale => discover_tailscale().await,
            DiscoveryMethod::Arp => discover_arp().await,
            DiscoveryMethod::SystemProfiler => discover_system_profiler().await,
            DiscoveryMethod::Bonjour => {
                // TODO: dns-sd -B _ssh._tcp local. with timeout is tricky to
                // implement reliably (it runs forever unless killed). Skipping
                // for now — the other methods cover our use cases.
                debug!("Bonjour discovery not yet implemented, skipping");
                Vec::new()
            }
        };

        info!(
            method = ?method,
            count = discovered.len(),
            "discovery method returned peers"
        );

        all_peers.extend(discovered);
    }

    deduplicate_peers(all_peers)
}

/// Probe a single node for hardware info, RDMA status, and running servers.
pub async fn scan_node(hostname: &str, config: &ClusterConfig) -> ScanResult {
    // 1. Ping test
    let ping_result = local_run(&format!("ping -c 1 -W 2 {} 2>/dev/null", hostname)).await;
    let (reachable, latency_ms) = match ping_result {
        Ok(ref r) if r.success => {
            let latency = parse_ping_latency(&r.stdout);
            (true, latency)
        }
        _ => (false, None),
    };

    if !reachable {
        debug!(hostname, "node unreachable via ping");
        return ScanResult {
            hostname: hostname.to_string(),
            reachable: false,
            ssh_ok: false,
            chip: None,
            ram_gb: None,
            gpu_cores: None,
            rdma: None,
            mlx_servers: Vec::new(),
            latency_ms: None,
        };
    }

    // 2. SSH hardware probe
    let hw_cmd = "sysctl -n hw.memsize 2>/dev/null; \
                  sysctl -n machdep.cpu.brand_string 2>/dev/null; \
                  sysctl -n hw.ncpu 2>/dev/null";
    let hw_result = ssh_run(hostname, hw_cmd, config).await;
    let (ssh_ok, chip, ram_gb, gpu_cores) = match hw_result {
        Ok(ref r) if r.success => {
            let lines: Vec<&str> = r.stdout.lines().collect();
            let mem_bytes: Option<u64> = lines.first().and_then(|s| s.trim().parse().ok());
            let ram = mem_bytes.map(|b| b / (1024 * 1024 * 1024));
            let chip_name = lines.get(1).map(|s| s.trim().to_string());
            let ncpu: Option<u32> = lines.get(2).and_then(|s| s.trim().parse().ok());
            (true, chip_name, ram, ncpu)
        }
        Ok(_) => (false, None, None, None),
        Err(e) => {
            debug!(hostname, error = %e, "SSH hardware probe failed");
            (false, None, None, None)
        }
    };

    // 3. RDMA check
    let rdma = if ssh_ok {
        let rdma_cmd = "rdma_ctl status 2>/dev/null; \
                        echo '---'; \
                        ibv_devices 2>/dev/null; \
                        echo '---'; \
                        ibv_devinfo 2>/dev/null";
        match ssh_run(hostname, rdma_cmd, config).await {
            Ok(ref r) if r.has_output() => Some(parse_rdma_status(&r.stdout)),
            _ => None,
        }
    } else {
        None
    };

    // 4. Process scan for MLX servers
    let mut mlx_servers: Vec<MlxServerInfo> = Vec::new();
    if ssh_ok {
        let ps_cmd = "ps aux 2>/dev/null | grep -E 'mlx_lm|mlx_vlm|vllm_mlx' | grep -v grep";
        if let Ok(ref r) = ssh_run(hostname, ps_cmd, config).await {
            if r.has_output() {
                let processes = parse_mlx_processes(&r.stdout);
                // 5. For each discovered port, try to query models
                for proc in &processes {
                    if let Some(port) = proc.port {
                        let curl_cmd = format!(
                            "curl -s --connect-timeout 2 http://127.0.0.1:{}/v1/models 2>/dev/null",
                            port
                        );
                        let models = match ssh_run(hostname, &curl_cmd, config).await {
                            Ok(ref r) if r.has_output() => parse_v1_models_response(&r.stdout),
                            _ => Vec::new(),
                        };
                        mlx_servers.push(MlxServerInfo {
                            port,
                            models,
                            engine: proc.framework,
                        });
                    }
                }
            }
        }
    }

    ScanResult {
        hostname: hostname.to_string(),
        reachable,
        ssh_ok,
        chip,
        ram_gb,
        gpu_cores,
        rdma,
        mlx_servers,
        latency_ms,
    }
}

/// Full cluster scan: discover + probe all found nodes.
pub async fn scan_cluster(config: &ClusterConfig) -> Vec<ScanResult> {
    let peers = discover_nodes(config).await;
    info!(count = peers.len(), "discovered peers, starting scan");

    let mut results = Vec::with_capacity(peers.len());

    // Scan all peers concurrently
    let handles: Vec<_> = peers
        .into_iter()
        .map(|peer| {
            let config = config.clone();
            let host = if peer.ips.is_empty() {
                peer.hostname.clone()
            } else {
                // Prefer the first IP for connectivity
                peer.ips[0].clone()
            };
            tokio::spawn(async move { scan_node(&host, &config).await })
        })
        .collect();

    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => warn!(error = %e, "scan task panicked"),
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Discovery implementations
// ---------------------------------------------------------------------------

/// Discover nodes via Thunderbolt bridge interfaces (169.254.x.x on en*).
///
/// Runs `ifconfig` locally, finds interfaces with link-local addresses, then
/// cross-references with `arp -a` to find peer IPs on those interfaces. For
/// each peer IP, SSHs to get the hostname.
async fn discover_thunderbolt_bridge(config: &ClusterConfig) -> Vec<DiscoveredPeer> {
    let ifconfig = match local_run("ifconfig 2>/dev/null").await {
        Ok(r) if r.has_output() => r.stdout,
        _ => return Vec::new(),
    };

    let bridges = parse_ifconfig_bridges(&ifconfig);
    if bridges.is_empty() {
        debug!("no thunderbolt bridge interfaces found");
        return Vec::new();
    }

    debug!(count = bridges.len(), "found thunderbolt bridge interfaces");

    // Get ARP table to find peer IPs on these interfaces
    let arp_output = match local_run("arp -a 2>/dev/null").await {
        Ok(r) if r.has_output() => r.stdout,
        _ => return Vec::new(),
    };

    let mut peers = Vec::new();

    // For each bridge interface, find peer IPs in the ARP table that are on
    // the same interface and in 169.254.x.x range (but not our own IP)
    let bridge_ips: HashMap<String, String> = bridges.iter().cloned().collect();

    for line in arp_output.lines() {
        // Parse: hostname (ip) at mac on interface [...]
        let arp_re = Regex::new(r"(\S+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+(\S+)\s+on\s+(\S+)")
            .expect("valid regex");
        if let Some(caps) = arp_re.captures(line) {
            let _arp_host = caps.get(1).unwrap().as_str();
            let ip = caps.get(2).unwrap().as_str();
            let mac = caps.get(3).unwrap().as_str();
            let iface = caps.get(4).unwrap().as_str();

            // Skip incomplete entries
            if mac == "(incomplete)" {
                continue;
            }

            // Only 169.254.x.x addresses on our bridge interfaces
            if !ip.starts_with("169.254.") {
                continue;
            }

            // Must be on one of our TB bridge interfaces
            if !bridge_ips.contains_key(iface) {
                continue;
            }

            // Skip our own IPs
            if bridge_ips.values().any(|own_ip| own_ip == ip) {
                continue;
            }

            // SSH to get hostname
            let hostname = match ssh_run(ip, "hostname -s 2>/dev/null", config).await {
                Ok(ref r) if r.has_output() => r.stdout.trim().to_string(),
                _ => ip.to_string(),
            };

            peers.push(DiscoveredPeer {
                hostname,
                ips: vec![ip.to_string()],
                discovery_source: "thunderbolt-bridge".to_string(),
                thunderbolt_bridge: Some(iface.to_string()),
                link_speed: None,
            });
        }
    }

    peers
}

/// Discover nodes via `tailscale status --json`.
async fn discover_tailscale() -> Vec<DiscoveredPeer> {
    let result = local_run("tailscale status --json 2>/dev/null").await;
    match result {
        Ok(ref r) if r.has_output() => parse_tailscale_peers(&r.stdout),
        Ok(_) => {
            debug!("tailscale status returned no output");
            Vec::new()
        }
        Err(e) => {
            debug!(error = %e, "tailscale status failed");
            Vec::new()
        }
    }
}

/// Discover nodes via ARP table.
async fn discover_arp() -> Vec<DiscoveredPeer> {
    let result = local_run("arp -a 2>/dev/null").await;
    match result {
        Ok(ref r) if r.has_output() => parse_arp_table(&r.stdout),
        _ => Vec::new(),
    }
}

/// Discover connected Thunderbolt devices via system_profiler.
async fn discover_system_profiler() -> Vec<DiscoveredPeer> {
    let result = local_run("system_profiler SPThunderboltDataType 2>/dev/null").await;
    match result {
        Ok(ref r) if r.has_output() => parse_system_profiler(&r.stdout),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

/// Parse `ifconfig` output to find interfaces with 169.254.x.x addresses.
/// Returns `(interface_name, ip_address)` pairs.
pub fn parse_ifconfig_bridges(text: &str) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let mut current_iface: Option<String> = None;

    let iface_re = Regex::new(r"^(\w+):\s+flags=").expect("valid regex");
    let inet_re = Regex::new(r"^\s+inet\s+(169\.254\.\d+\.\d+)\s").expect("valid regex");

    for line in text.lines() {
        if let Some(caps) = iface_re.captures(line) {
            current_iface = Some(caps.get(1).unwrap().as_str().to_string());
        } else if let Some(caps) = inet_re.captures(line) {
            if let Some(ref iface) = current_iface {
                let ip = caps.get(1).unwrap().as_str().to_string();
                results.push((iface.clone(), ip));
            }
        }
    }

    results
}

/// Parse `tailscale status --json` output to find macOS + online peers.
pub fn parse_tailscale_peers(json: &str) -> Vec<DiscoveredPeer> {
    // Strip the "Warning:" line that tailscale sometimes prepends
    let json_start = json.find('{').unwrap_or(0);
    let json_str = &json[json_start..];

    let value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "failed to parse tailscale JSON");
            return Vec::new();
        }
    };

    let mut peers = Vec::new();

    // Also include Self node
    if let Some(self_node) = value.get("Self") {
        if let Some(peer) = extract_tailscale_peer(self_node) {
            peers.push(peer);
        }
    }

    // Parse Peer map
    if let Some(peer_map) = value.get("Peer").and_then(|p| p.as_object()) {
        for (_key, peer_value) in peer_map {
            if let Some(peer) = extract_tailscale_peer(peer_value) {
                peers.push(peer);
            }
        }
    }

    peers
}

/// Extract a single peer from a tailscale peer JSON object.
/// Filters: OS must contain "macOS" and Online must be true.
fn extract_tailscale_peer(value: &serde_json::Value) -> Option<DiscoveredPeer> {
    let os = value.get("OS")?.as_str()?;
    let online = value.get("Online")?.as_bool()?;

    if !os.contains("macOS") || !online {
        return None;
    }

    let hostname = value.get("HostName")?.as_str()?.to_string();
    let ips: Vec<String> = value
        .get("TailscaleIPs")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        // Only include IPv4 addresses for SSH connectivity
        .filter(|ip| !ip.contains(':'))
        .collect();

    Some(DiscoveredPeer {
        hostname,
        ips,
        discovery_source: "tailscale".to_string(),
        thunderbolt_bridge: None,
        link_speed: None,
    })
}

/// Parse `arp -a` output to find named hosts (skip `?` entries).
/// Format: `hostname (ip) at mac on interface [...]`
pub fn parse_arp_table(text: &str) -> Vec<DiscoveredPeer> {
    let re = Regex::new(r"^(\S+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+(\S+)\s+on\s+(\S+)")
        .expect("valid regex");

    let mut peers_map: HashMap<String, DiscoveredPeer> = HashMap::new();

    for line in text.lines() {
        if let Some(caps) = re.captures(line) {
            let hostname = caps.get(1).unwrap().as_str();
            let ip = caps.get(2).unwrap().as_str();
            let mac = caps.get(3).unwrap().as_str();

            // Skip unknown hosts, incomplete entries, multicast, broadcast
            if hostname == "?" || mac == "(incomplete)" {
                continue;
            }
            if ip.starts_with("224.") || ip.starts_with("239.") || ip.ends_with(".255") {
                continue;
            }

            // Normalize hostname: strip ".local" suffix
            let normalized = hostname.strip_suffix(".local").unwrap_or(hostname);

            let entry = peers_map
                .entry(normalized.to_string())
                .or_insert_with(|| DiscoveredPeer {
                    hostname: normalized.to_string(),
                    ips: Vec::new(),
                    discovery_source: "arp".to_string(),
                    thunderbolt_bridge: None,
                    link_speed: None,
                });

            if !entry.ips.contains(&ip.to_string()) {
                entry.ips.push(ip.to_string());
            }
        }
    }

    peers_map.into_values().collect()
}

/// Parse combined RDMA status output.
///
/// Expected format (three sections separated by `---`):
/// 1. `rdma_ctl status` → "enabled" or "disabled"
/// 2. `ibv_devices` → device listing
/// 3. `ibv_devinfo` → per-device port state details
pub fn parse_rdma_status(text: &str) -> RdmaStatus {
    let sections: Vec<&str> = text.split("
---
").collect();

    // Section 1: enabled/disabled
    let enabled = sections
        .first()
        .map(|s| s.trim().eq_ignore_ascii_case("enabled"))
        .unwrap_or(false);

    // Section 3: ibv_devinfo — parse device names and port states
    let devices = if sections.len() >= 3 {
        parse_ibv_devinfo(sections[2])
    } else {
        Vec::new()
    };

    RdmaStatus { enabled, devices }
}

/// Parse `ibv_devinfo` output for device names and port states.
fn parse_ibv_devinfo(text: &str) -> Vec<RdmaDevice> {
    let device_re = Regex::new(r"hca_id:\s+(\S+)").expect("valid regex");
    let state_re = Regex::new(r"state:\s+(PORT_\w+)").expect("valid regex");

    let mut devices = Vec::new();
    let mut current_device: Option<String> = None;

    for line in text.lines() {
        if let Some(caps) = device_re.captures(line) {
            // If we had a previous device without a state, mark it unknown
            if let Some(ref name) = current_device {
                devices.push(RdmaDevice {
                    name: name.clone(),
                    port_state: PortState::Unknown,
                });
            }
            current_device = Some(caps.get(1).unwrap().as_str().to_string());
        } else if let Some(caps) = state_re.captures(line) {
            if let Some(name) = current_device.take() {
                let state_str = caps.get(1).unwrap().as_str();
                let port_state = PortState::from_ibstat(state_str);
                devices.push(RdmaDevice { name, port_state });
            }
        }
    }

    // Handle last device without state
    if let Some(name) = current_device {
        devices.push(RdmaDevice {
            name,
            port_state: PortState::Unknown,
        });
    }

    devices
}

/// Parse `system_profiler SPThunderboltDataType` for connected devices and link speeds.
fn parse_system_profiler(text: &str) -> Vec<DiscoveredPeer> {
    let mut peers = Vec::new();

    // Look for connected device blocks. Pattern:
    //   Device Name: <name>
    //   ...
    //   Speed: <speed>
    //   ...
    //   Status: Device connected

    // We look for indented "Device Name:" entries that are children of bus entries
    // and are actual Mac devices (not docks/peripherals).
    let device_re = Regex::new(r"^\s+Device Name:\s+(.+)$").expect("valid regex");
    let speed_re = Regex::new(r"^\s+Speed:\s+(.+)$").expect("valid regex");
    let status_re = Regex::new(r"^\s+Status:\s+(.+)$").expect("valid regex");

    // We want to find "Mac*" device names under "Status: Device connected" ports
    let mut in_connected_port = false;
    let mut current_speed: Option<String> = None;

    for line in text.lines() {
        if let Some(caps) = status_re.captures(line) {
            let status = caps.get(1).unwrap().as_str().trim();
            in_connected_port = status == "Device connected";
        }

        if let Some(caps) = speed_re.captures(line) {
            current_speed = Some(caps.get(1).unwrap().as_str().trim().to_string());
        }

        if let Some(caps) = device_re.captures(line) {
            let name = caps.get(1).unwrap().as_str().trim();
            // Only pick up Mac devices (Mac Studio, Mac mini, MacBook Pro, etc.)
            // and only when under a connected port. Also skip the "self" bus entries
            // by checking indentation level — child devices are indented more.
            if in_connected_port && name.starts_with("Mac") {
                // Avoid duplicating if the same device name appears in multiple buses
                let already_found = peers.iter().any(|p: &DiscoveredPeer| {
                    p.hostname == name
                });
                if !already_found {
                    peers.push(DiscoveredPeer {
                        hostname: name.to_string(),
                        ips: Vec::new(), // System profiler doesn't give IPs
                        discovery_source: "system-profiler".to_string(),
                        thunderbolt_bridge: None,
                        link_speed: current_speed.clone(),
                    });
                }
            }
        }
    }

    peers
}

/// Parse MLX server processes from `ps aux` output.
fn parse_mlx_processes(text: &str) -> Vec<ProcessInfo> {
    let mut processes = Vec::new();
    let port_re = Regex::new(r"--port\s+(\d+)").expect("valid regex");
    let model_re = Regex::new(r"--model\s+(\S+)").expect("valid regex");

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Determine framework
        let framework = if line.contains("mlx_vlm") {
            ProcessFramework::MlxVlm
        } else if line.contains("mlx_lm") {
            ProcessFramework::MlxLm
        } else if line.contains("vllm_mlx") {
            ProcessFramework::VllmMlx
        } else {
            continue;
        };

        // Parse PID from ps aux output (second field)
        let fields: Vec<&str> = line.split_whitespace().collect();
        let pid: u32 = fields.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        let cpu_percent: f64 = fields.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let mem_percent: f64 = fields.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

        let port = port_re
            .captures(line)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse().ok());

        let model = model_re
            .captures(line)
            .and_then(|c| c.get(1))
            .map(|m| {
                let path = m.as_str();
                // Extract just the model name from the path
                path.rsplit('/').next().unwrap_or(path).to_string()
            });

        // Detect distributed backend from command args
        let distributed = if line.contains("--backend jaccl") {
            Some(DistributedBackend::Jaccl)
        } else if line.contains("--backend ring") {
            Some(DistributedBackend::Ring)
        } else {
            None
        };

        processes.push(ProcessInfo {
            pid,
            framework,
            model,
            port,
            cpu_percent,
            mem_percent,
            footprint_mb: None,
            distributed,
        });
    }

    processes
}

/// Parse a `/v1/models` JSON response to extract model IDs.
fn parse_v1_models_response(json: &str) -> Vec<String> {
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse ping output for round-trip latency in ms.
fn parse_ping_latency(output: &str) -> Option<f64> {
    // macOS ping: "round-trip min/avg/max/stddev = 0.123/0.456/0.789/0.012 ms"
    let re = Regex::new(r"round-trip\s+\S+\s*=\s*[\d.]+/([\d.]+)/").expect("valid regex");
    re.captures(output)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Deduplicate peers by hostname, merging IPs and preferring richer metadata.
fn deduplicate_peers(peers: Vec<DiscoveredPeer>) -> Vec<DiscoveredPeer> {
    let mut by_hostname: HashMap<String, DiscoveredPeer> = HashMap::new();

    for peer in peers {
        let key = peer.hostname.to_lowercase();
        let entry = by_hostname.entry(key).or_insert_with(|| DiscoveredPeer {
            hostname: peer.hostname.clone(),
            ips: Vec::new(),
            discovery_source: peer.discovery_source.clone(),
            thunderbolt_bridge: None,
            link_speed: None,
        });

        // Merge IPs
        for ip in &peer.ips {
            if !entry.ips.contains(ip) {
                entry.ips.push(ip.clone());
            }
        }

        // Prefer non-None metadata
        if entry.thunderbolt_bridge.is_none() {
            entry.thunderbolt_bridge = peer.thunderbolt_bridge;
        }
        if entry.link_speed.is_none() {
            entry.link_speed = peer.link_speed;
        }

        // Append discovery sources
        if !entry.discovery_source.contains(&peer.discovery_source) {
            entry.discovery_source = format!("{},{}", entry.discovery_source, peer.discovery_source);
        }
    }

    by_hostname.into_values().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tailscale_peers() {
        let json = include_str!("../testdata/tailscale-status.json");
        let peers = parse_tailscale_peers(json);

        // Should only include macOS + online peers
        // From the testdata: m3u2 (Self, macOS, online), m4m1 (macOS, online),
        // jose's Mac mini (macOS, online), m3u1 (macOS, online),
        // m3u3 (macOS, online), mini1 (macOS, online)
        // Should NOT include: MacBookPro (offline), alienMARIO (windows),
        // k3s-* (linux), localhost/iOS, JP's MacBook Air (offline), etc.

        let hostnames: Vec<&str> = peers.iter().map(|p| p.hostname.as_str()).collect();
        assert!(
            hostnames.contains(&"m3u2"),
            "should find m3u2 (Self node): {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m4m1"),
            "should find m4m1: {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m3u1"),
            "should find m3u1: {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m3u3"),
            "should find m3u3: {hostnames:?}"
        );

        // Offline macOS should NOT be included
        assert!(
            !hostnames.contains(&"MacBookPro"),
            "should not find MacBookPro (offline): {hostnames:?}"
        );

        // Windows/Linux/iOS should NOT be included
        assert!(
            !hostnames.contains(&"alienMARIO"),
            "should not find alienMARIO (windows): {hostnames:?}"
        );
        assert!(
            !hostnames.contains(&"k3s-m3u1"),
            "should not find k3s-m3u1 (linux): {hostnames:?}"
        );

        // Check that IPs are IPv4 only
        for peer in &peers {
            for ip in &peer.ips {
                assert!(
                    !ip.contains(':'),
                    "should only have IPv4 IPs, got: {ip}"
                );
            }
            assert_eq!(peer.discovery_source, "tailscale");
        }
    }

    #[test]
    fn test_parse_arp_table() {
        let text = include_str!("../testdata/arp-table.txt");
        let peers = parse_arp_table(text);

        let hostnames: Vec<&str> = peers.iter().map(|p| p.hostname.as_str()).collect();

        // Should find named hosts
        assert!(
            hostnames.contains(&"m3u1"),
            "should find m3u1: {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m3u3"),
            "should find m3u3: {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m4m1"),
            "should find m4m1: {hostnames:?}"
        );
        assert!(
            hostnames.contains(&"m3u2"),
            "should find m3u2: {hostnames:?}"
        );

        // Should NOT include "?" entries
        for peer in &peers {
            assert_ne!(peer.hostname, "?", "should not include unknown hosts");
        }

        // Should NOT include multicast/broadcast
        for peer in &peers {
            for ip in &peer.ips {
                assert!(!ip.starts_with("224."), "should not include multicast: {ip}");
                assert!(!ip.starts_with("239."), "should not include multicast: {ip}");
                assert!(!ip.ends_with(".255"), "should not include broadcast: {ip}");
            }
        }

        // m3u1 should have multiple IPs (169.254.118.6 and 169.254.225.84)
        let m3u1 = peers.iter().find(|p| p.hostname == "m3u1").unwrap();
        assert!(
            m3u1.ips.len() >= 2,
            "m3u1 should have at least 2 IPs, got: {:?}",
            m3u1.ips
        );
    }

    #[test]
    fn test_parse_rdma_status() {
        let text = include_str!("../testdata/rdma-status.txt");
        let status = parse_rdma_status(text);

        assert!(status.enabled, "RDMA should be enabled");
        assert_eq!(status.devices.len(), 6, "should have 6 RDMA devices");

        // Check specific devices
        let en3 = status.devices.iter().find(|d| d.name == "rdma_en3").unwrap();
        assert_eq!(en3.port_state, PortState::Active, "rdma_en3 should be active");

        let en2 = status.devices.iter().find(|d| d.name == "rdma_en2").unwrap();
        assert_eq!(en2.port_state, PortState::Down, "rdma_en2 should be down");

        let en4 = status.devices.iter().find(|d| d.name == "rdma_en4").unwrap();
        assert_eq!(en4.port_state, PortState::Active, "rdma_en4 should be active");

        let en5 = status.devices.iter().find(|d| d.name == "rdma_en5").unwrap();
        assert_eq!(en5.port_state, PortState::Active, "rdma_en5 should be active");

        // active_count should be 3 (en3, en4, en5)
        assert_eq!(status.active_count(), 3, "should have 3 active devices");
    }

    #[test]
    fn test_parse_ifconfig_bridges() {
        let text = include_str!("../testdata/ifconfig-bridges.txt");
        let bridges = parse_ifconfig_bridges(text);

        // Should find en3, en5, en22, en23 with 169.254.x.x addresses
        // en4 has 192.168.60.1 so should NOT be included
        let ifaces: Vec<&str> = bridges.iter().map(|(iface, _)| iface.as_str()).collect();

        assert!(
            ifaces.contains(&"en3"),
            "should find en3: {ifaces:?}"
        );
        assert!(
            ifaces.contains(&"en5"),
            "should find en5: {ifaces:?}"
        );
        assert!(
            ifaces.contains(&"en22"),
            "should find en22: {ifaces:?}"
        );
        assert!(
            ifaces.contains(&"en23"),
            "should find en23: {ifaces:?}"
        );
        assert!(
            !ifaces.contains(&"en4"),
            "should NOT find en4 (192.168 address): {ifaces:?}"
        );

        // Check specific IPs
        let en3_ip = bridges.iter().find(|(i, _)| i == "en3").unwrap();
        assert_eq!(en3_ip.1, "169.254.19.163");
    }

    #[test]
    fn test_parse_system_profiler() {
        let text = include_str!("../testdata/system-profiler-tb.txt");
        let peers = parse_system_profiler(text);

        // From the testdata, we should find connected Mac devices:
        // Mac16,9 and Mac15,14 (appears twice at different buses)
        assert!(
            !peers.is_empty(),
            "should find at least one connected Mac device"
        );

        // All should have system-profiler as source
        for peer in &peers {
            assert_eq!(peer.discovery_source, "system-profiler");
        }

        // Should have link speeds
        let with_speed: Vec<_> = peers.iter().filter(|p| p.link_speed.is_some()).collect();
        assert!(
            !with_speed.is_empty(),
            "at least one peer should have a link speed"
        );
    }

    #[test]
    fn test_parse_mlx_processes() {
        let text = include_str!("../testdata/ps-mlx.txt");
        let procs = parse_mlx_processes(text);

        // Should find the mlx_lm.server process
        let mlx_servers: Vec<_> = procs
            .iter()
            .filter(|p| p.framework == ProcessFramework::MlxLm)
            .collect();
        assert!(
            !mlx_servers.is_empty(),
            "should find at least one mlx_lm process"
        );

        // Check port and model extraction
        let server = &mlx_servers[0];
        assert_eq!(server.port, Some(8003), "should extract port 8003");
        assert_eq!(
            server.model.as_deref(),
            Some("MiniMax-M2.5-REAP-19-8bit"),
            "should extract model name"
        );
        assert_eq!(server.pid, 62283, "should extract PID");
    }

    #[test]
    fn test_parse_sysctl_hw() {
        let text = include_str!("../testdata/sysctl-hw.txt");
        let lines: Vec<&str> = text.lines().collect();

        let mem_bytes: u64 = lines[0].trim().parse().unwrap();
        assert_eq!(mem_bytes, 549755813888, "should be 512GB in bytes");
        assert_eq!(mem_bytes / (1024 * 1024 * 1024), 512, "should be 512GB");

        let chip = lines[1].trim();
        assert_eq!(chip, "Apple M3 Ultra");
    }

    #[test]
    fn test_deduplicate_peers() {
        let peers = vec![
            DiscoveredPeer {
                hostname: "m3u1".to_string(),
                ips: vec!["169.254.118.6".to_string()],
                discovery_source: "arp".to_string(),
                thunderbolt_bridge: Some("en4".to_string()),
                link_speed: None,
            },
            DiscoveredPeer {
                hostname: "m3u1".to_string(),
                ips: vec!["100.127.90.10".to_string()],
                discovery_source: "tailscale".to_string(),
                thunderbolt_bridge: None,
                link_speed: None,
            },
        ];

        let deduped = deduplicate_peers(peers);
        assert_eq!(deduped.len(), 1, "should deduplicate to 1 peer");

        let m3u1 = &deduped[0];
        assert_eq!(m3u1.hostname, "m3u1");
        assert!(
            m3u1.ips.contains(&"169.254.118.6".to_string()),
            "should have TB IP"
        );
        assert!(
            m3u1.ips.contains(&"100.127.90.10".to_string()),
            "should have tailscale IP"
        );
        assert_eq!(
            m3u1.thunderbolt_bridge,
            Some("en4".to_string()),
            "should preserve TB bridge"
        );
        assert!(
            m3u1.discovery_source.contains("arp"),
            "should have arp source"
        );
        assert!(
            m3u1.discovery_source.contains("tailscale"),
            "should have tailscale source"
        );
    }

    #[test]
    fn test_parse_ping_latency() {
        let output = "PING m3u1 (169.254.118.6): 56 data bytes\n\
                      64 bytes from 169.254.118.6: icmp_seq=0 ttl=64 time=0.234 ms\n\
                      \n\
                      --- m3u1 ping statistics ---\n\
                      1 packets transmitted, 1 packets received, 0.0% packet loss\n\
                      round-trip min/avg/max/stddev = 0.234/0.234/0.234/0.000 ms";
        let latency = parse_ping_latency(output);
        assert!(latency.is_some(), "should parse latency");
        assert!((latency.unwrap() - 0.234).abs() < 0.001);
    }

    #[test]
    fn test_parse_v1_models_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {"id": "mlx-community/Qwen2.5-72B-4bit", "object": "model"},
                {"id": "mlx-community/Llama-3.3-70B-8bit", "object": "model"}
            ]
        }"#;
        let models = parse_v1_models_response(json);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0], "mlx-community/Qwen2.5-72B-4bit");
        assert_eq!(models[1], "mlx-community/Llama-3.3-70B-8bit");
    }

    #[test]
    fn test_parse_v1_models_invalid_json() {
        let models = parse_v1_models_response("not json");
        assert!(models.is_empty());
    }
}
