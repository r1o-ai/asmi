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

use crate::collector::fetch_from_daemon;
use crate::config::{ClusterConfig, DiscoveryMethod};
use crate::health::{find_thunderbolt_issues, parse_thunderbolt_services};
use crate::ssh::{local_run, ssh_run};
use crate::types::*;
use crate::types::ModelServerMetadata;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Static regexes (compiled once, reused across all scan calls)
// ---------------------------------------------------------------------------

// parse_ifconfig_bridges + parse_ifconfig_all_ips (shared interface regex)
static IFACE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\w+):\s+flags=").unwrap()
});

// parse_ifconfig_bridges — 169.254.x.x only
static INET_LINK_LOCAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s+inet\s+(169\.254\.\d+\.\d+)\s").unwrap()
});

// parse_ifconfig_all_ips — any IPv4
static INET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s+inet\s+(\d+\.\d+\.\d+\.\d+)\s").unwrap()
});

// parse_arp_table + discover_thunderbolt_bridge
static ARP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\S+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+(\S+)\s+on\s+(\S+)").unwrap()
});

// parse_ibv_devinfo
static DEVICE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"hca_id:\s+(\S+)").unwrap()
});
static STATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"state:\s+(PORT_\w+)").unwrap()
});

// parse_system_profiler
static SP_DEVICE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s+Device Name:\s+(.+)$").unwrap()
});
static SP_SPEED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s+Speed:\s+(.+)$").unwrap()
});
static SP_STATUS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s+Status:\s+(.+)$").unwrap()
});

// parse_mlx_processes
static MLX_PORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"--port\s+(\d+)").unwrap()
});
static MLX_MODEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"--model\s+(\S+)").unwrap()
});

// parse_ping_latency
static PING_LATENCY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"round-trip\s+\S+\s*=\s*[\d.]+/([\d.]+)/").unwrap()
});

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
    /// Local IP on the thunderbolt bridge interface (our side of the link).
    pub local_ip: Option<String>,
    pub link_speed: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level API
// ---------------------------------------------------------------------------

/// Discover cluster nodes using configured methods. Returns deduplicated peers.
///
/// All discovery methods run **concurrently** — events stream in as each
/// method completes independently.
pub async fn discover_nodes(config: &ClusterConfig, events: &EventSink) -> Vec<DiscoveredPeer> {
    let mut all_peers: Vec<DiscoveredPeer> = Vec::new();

    // Add seed hosts first
    for seed in &config.seed_hosts {
        all_peers.push(DiscoveredPeer {
            hostname: seed.clone(),
            ips: vec![seed.clone()],
            discovery_source: "seed".to_string(),
            thunderbolt_bridge: None,
            local_ip: None,
            link_speed: None,
        });
    }
    if !config.seed_hosts.is_empty() {
        events.emit(ClusterEvent::DiscoveryFound {
            method: "Seeds".to_string(),
            count: config.seed_hosts.len(),
        });
    }

    if config.discovery.is_empty() {
        return deduplicate_peers(all_peers);
    }

    // Emit all DiscoveryStarted events upfront
    for method in &config.discovery {
        events.emit(ClusterEvent::DiscoveryStarted {
            method: discovery_method_name(method),
        });
    }

    // Spawn all discovery methods concurrently
    let handles: Vec<_> = config.discovery.iter().map(|method| {
        let config = config.clone();
        let events = events.clone();
        let method = method.clone();
        tokio::spawn(async move {
            let method_name = discovery_method_name(&method);
            let discovered = match method {
                DiscoveryMethod::ThunderboltBridge => discover_thunderbolt_bridge(&config).await,
                DiscoveryMethod::Tailscale => discover_tailscale().await,
                DiscoveryMethod::Arp => discover_arp(true).await,
                DiscoveryMethod::ArpAll => discover_arp(false).await,
                DiscoveryMethod::SystemProfiler => discover_system_profiler().await,
                DiscoveryMethod::Bonjour => {
                    debug!("Bonjour discovery not yet implemented, skipping");
                    Vec::new()
                }
            };

            info!(
                method = method_name.as_str(),
                count = discovered.len(),
                "discovery method returned peers"
            );

            events.emit(ClusterEvent::DiscoveryFound {
                method: method_name,
                count: discovered.len(),
            });

            discovered
        })
    }).collect();

    for handle in handles {
        match handle.await {
            Ok(peers) => all_peers.extend(peers),
            Err(e) => warn!(error = %e, "discovery task panicked"),
        }
    }

    deduplicate_peers(all_peers)
}

fn discovery_method_name(method: &DiscoveryMethod) -> String {
    match method {
        DiscoveryMethod::ThunderboltBridge => "Thunderbolt".to_string(),
        DiscoveryMethod::Tailscale => "Tailscale".to_string(),
        DiscoveryMethod::Arp => "ARP".to_string(),
        DiscoveryMethod::ArpAll => "ARP (all)".to_string(),
        DiscoveryMethod::SystemProfiler => "System Profiler".to_string(),
        DiscoveryMethod::Bonjour => "Bonjour".to_string(),
    }
}

/// Probe a single node for hardware info, RDMA status, and running servers.
///
/// If `skip_ping` is true, assumes the node is reachable and goes straight
/// to SSH. Use for seed hosts and previously-known nodes.
pub async fn scan_node(hostname: &str, config: &ClusterConfig) -> ScanResult {
    scan_node_inner(hostname, config, false).await
}

/// Like `scan_node` but skips the ping check for known-reachable hosts.
pub async fn scan_node_fast(hostname: &str, config: &ClusterConfig) -> ScanResult {
    scan_node_inner(hostname, config, true).await
}

async fn scan_node_inner(hostname: &str, config: &ClusterConfig, skip_ping: bool) -> ScanResult {
    // 1. Try HTTP daemon first — fast online check + basic hardware info.
    //    If it responds, we know the node is reachable and get hostname/RAM
    //    from the snapshot without any SSH.
    if let Some(snap) = fetch_from_daemon(hostname, config.daemon_port).await {
        info!(hostname, "scan: daemon online, skipping SSH hardware probe");
        let ram_gb = if snap.ram_total_bytes > 0 {
            Some(snap.ram_total_bytes / (1024 * 1024 * 1024))
        } else {
            None
        };
        return ScanResult {
            hostname: snap.hostname,
            reachable: true,
            ssh_ok: true,
            chip: snap.chip_model,
            ram_gb,
            gpu_cores: None,  // Not in NodeSnapshot
            rdma: snap.rdma,
            mlx_servers: snap.processes.into_iter().filter_map(|p| {
                p.port.map(|port| MlxServerInfo {
                    port,
                    models: p.server_models.into_iter().map(|m| m.id).collect(),
                    engine: p.framework,
                })
            }).collect(),
            latency_ms: None,
            link_speed: None,
        };
    }

    // 2. Daemon unreachable — fall back to ping + SSH.
    let (reachable, latency_ms) = if skip_ping {
        (true, None)
    } else {
        let ping_result = local_run(&format!("ping -c 1 -W 2 {} 2>/dev/null", hostname)).await;
        match ping_result {
            Ok(ref r) if r.success => {
                let latency = parse_ping_latency(&r.stdout);
                (true, latency)
            }
            _ => (false, None),
        }
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
            link_speed: None,
        };
    }

    // 3. Batched SSH probe: hardware + RDMA + process scan in a SINGLE SSH call.
    //    Previously this was 3 sequential SSH sessions (~200-500ms each for TCP+auth).
    //    Now we combine all probes with section delimiters (===HW===, ===RDMA===,
    //    ===PS===) and parse the combined output.
    let batched_cmd = "\
        echo '===HW==='; \
        hostname -s 2>/dev/null; \
        echo '---'; \
        sysctl -n hw.memsize 2>/dev/null; \
        echo '---'; \
        sysctl -n machdep.cpu.brand_string 2>/dev/null; \
        echo '---'; \
        sysctl -n hw.ncpu 2>/dev/null; \
        echo '===RDMA==='; \
        rdma_ctl status 2>/dev/null; \
        echo '---'; \
        ibv_devices 2>/dev/null; \
        echo '---'; \
        ibv_devinfo 2>/dev/null; \
        echo '===PS==='; \
        ps aux 2>/dev/null | grep -E 'mlx_lm|mlx_vlm|vllm_mlx' | grep -v grep";
    let batched_result = ssh_run(hostname, batched_cmd, config).await;

    let (ssh_ok, resolved_hostname, chip, ram_gb, gpu_cores, rdma, processes) =
        match batched_result {
        Ok(ref r) if r.success => {
            let output = &r.stdout;

            // --- Parse HW section (between ===HW=== and ===RDMA===) ---
            let hw_start = output.find("===HW===").map(|i| i + 8).unwrap_or(0);
            let hw_end = output.find("===RDMA===").unwrap_or(output.len());
            let hw_text = output[hw_start..hw_end].trim();
            // HW values are separated by --- delimiters: hostname, memsize, cpu_brand, ncpu
            let hw_parts: Vec<&str> = hw_text.split("\n---\n").collect();
            let resolved = hw_parts.first()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let mem_bytes: Option<u64> = hw_parts.get(1).and_then(|s| s.trim().parse().ok());
            let ram = mem_bytes.map(|b| b / (1024 * 1024 * 1024));
            let chip_name = hw_parts.get(2).map(|s| s.trim().to_string());
            let ncpu: Option<u32> = hw_parts.get(3).and_then(|s| s.trim().parse().ok());

            // --- Parse RDMA section (between ===RDMA=== and ===PS===) ---
            // Feed the raw section text to parse_rdma_status, which expects
            // rdma_ctl / ibv_devices / ibv_devinfo separated by \n---\n.
            let rdma_start = output.find("===RDMA===").map(|i| i + 10).unwrap_or(output.len());
            let rdma_end = output.find("===PS===").unwrap_or(output.len());
            let rdma_text = output[rdma_start..rdma_end].trim();
            let rdma_status = if rdma_text.is_empty() {
                None
            } else {
                Some(parse_rdma_status(rdma_text))
            };

            // --- Parse PS section (after ===PS===) ---
            let ps_start = output.find("===PS===").map(|i| i + 8).unwrap_or(output.len());
            let ps_text = output[ps_start..].trim();
            let procs = if ps_text.is_empty() {
                Vec::new()
            } else {
                parse_mlx_processes(ps_text)
            };

            (true, resolved, chip_name, ram, ncpu, rdma_status, procs)
        }
        Ok(_) => (false, None, None, None, None, None, Vec::new()),
        Err(e) => {
            debug!(hostname, error = %e, "SSH batched probe failed");
            (false, None, None, None, None, None, Vec::new())
        }
    };

    // Use SSH-resolved hostname if we probed by IP
    let final_hostname = resolved_hostname
        .unwrap_or_else(|| hostname.to_string());

    // 4. Batch model endpoint probes into a SINGLE SSH call for all discovered ports.
    //    Each port's curl response is delimited by ===PORT_NNNN=== markers so we
    //    can split the combined output and feed each section to parse_v1_models_metadata.
    let mut mlx_servers: Vec<MlxServerInfo> = Vec::new();
    if ssh_ok {
        let ports_and_frameworks: Vec<(u16, ProcessFramework)> = processes
            .iter()
            .filter_map(|p| p.port.map(|port| (port, p.framework)))
            .collect();

        if !ports_and_frameworks.is_empty() {
            // Build a single SSH command that probes ALL ports with section delimiters
            let curl_parts: Vec<String> = ports_and_frameworks
                .iter()
                .map(|(port, _)| {
                    format!(
                        "echo '===PORT_{port}==='; \
                         curl -s --connect-timeout 2 http://127.0.0.1:{port}/v1/models 2>/dev/null"
                    )
                })
                .collect();
            let curl_cmd = curl_parts.join("; ");

            let port_output: Option<String> = match ssh_run(hostname, &curl_cmd, config).await {
                Ok(ref r) if r.has_output() => Some(r.stdout.clone()),
                _ => None,
            };

            // Parse each port's section from the combined output
            for (port, framework) in &ports_and_frameworks {
                let models = if let Some(ref output) = port_output {
                    let marker = format!("===PORT_{port}===");
                    if let Some(section_start) = output.find(&marker).map(|i| i + marker.len()) {
                        // Find end: next ===PORT_ marker or end of output
                        let section_end = output[section_start..]
                            .find("===PORT_")
                            .map(|i| section_start + i)
                            .unwrap_or(output.len());
                        let section_text = output[section_start..section_end].trim();
                        parse_v1_models_metadata(section_text)
                            .iter()
                            .map(|m| m.id.clone())
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                mlx_servers.push(MlxServerInfo {
                    port: *port,
                    models,
                    engine: *framework,
                });
            }
        }
    }

    ScanResult {
        hostname: final_hostname,
        reachable,
        ssh_ok,
        chip,
        ram_gb,
        gpu_cores,
        rdma,
        mlx_servers,
        latency_ms,
        link_speed: None,
    }
}

/// Quick probe of seed hosts only — no discovery. Used for the fast first
/// phase of hierarchical scanning to show known nodes immediately.
/// Emits `AliasDiscovered` for any seed that resolves to a different canonical name.
pub async fn scan_seeds(config: &ClusterConfig, events: &EventSink) -> Vec<ScanResult> {
    if config.seed_hosts.is_empty() {
        return Vec::new();
    }

    info!(count = config.seed_hosts.len(), "fast-probing seed hosts");
    events.emit(ClusterEvent::ProbingStarted {
        count: config.seed_hosts.len(),
    });

    let handles: Vec<_> = config
        .seed_hosts
        .iter()
        .map(|host| {
            let config = config.clone();
            let events = events.clone();
            let host = host.clone();
            tokio::spawn(async move {
                let result = scan_node_fast(&host, &config).await;
                events.emit(ClusterEvent::NodeProbed {
                    hostname: result.hostname.clone(),
                    online: result.ssh_ok,
                    chip: result.chip.clone(),
                    ram_gb: result.ram_gb,
                });
                // Emit alias if the seed resolved to a different canonical name
                if result.ssh_ok && host != result.hostname {
                    events.emit(ClusterEvent::AliasDiscovered {
                        alias: host.clone(),
                        canonical: result.hostname.clone(),
                    });
                }
                result
            })
        })
        .collect();

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => warn!(error = %e, "seed scan task panicked"),
        }
    }

    let online = results.iter().filter(|r| r.ssh_ok).count();
    events.emit(ClusterEvent::ScanComplete {
        online,
        total: results.len(),
    });
    results
}

/// Full cluster scan: discover + probe all found nodes.
/// Emits `AliasDiscovered` events for every hostname/IP that resolves to a
/// different canonical SSH name — this auto-populates the persistent NodeMap.
pub async fn scan_cluster(config: &ClusterConfig, events: &EventSink) -> Vec<ScanResult> {
    let seeds: std::collections::HashSet<String> = config.seed_hosts.iter().cloned().collect();
    let peers = discover_nodes(config, events).await;
    info!(count = peers.len(), "discovered peers, starting scan");
    events.emit(ClusterEvent::ProbingStarted { count: peers.len() });

    let mut results = Vec::with_capacity(peers.len());

    // Scan all peers concurrently, returning (peer, probed_host, result) so we
    // can extract alias mappings after scanning.
    let handles: Vec<_> = peers
        .into_iter()
        .map(|peer| {
            let config = config.clone();
            let events = events.clone();
            let is_seed = seeds.contains(&peer.hostname)
                || peer.ips.iter().any(|ip| seeds.contains(ip));
            let host = if peer.ips.is_empty() {
                peer.hostname.clone()
            } else {
                peer.ips[0].clone()
            };
            tokio::spawn(async move {
                let result = if is_seed {
                    scan_node_fast(&host, &config).await
                } else {
                    scan_node(&host, &config).await
                };
                events.emit(ClusterEvent::NodeProbed {
                    hostname: result.hostname.clone(),
                    online: result.ssh_ok,
                    chip: result.chip.clone(),
                    ram_gb: result.ram_gb,
                });
                (peer, host, result)
            })
        })
        .collect();

    for handle in handles {
        match handle.await {
            Ok((peer, probed_host, result)) => {
                // Emit alias mappings for any name that resolved differently
                if result.ssh_ok {
                    if probed_host != result.hostname {
                        events.emit(ClusterEvent::AliasDiscovered {
                            alias: probed_host,
                            canonical: result.hostname.clone(),
                        });
                    }
                    if peer.hostname != result.hostname {
                        events.emit(ClusterEvent::AliasDiscovered {
                            alias: peer.hostname,
                            canonical: result.hostname.clone(),
                        });
                    }
                    // Collect TB bridge IPs (169.254.x.x) for RDMA
                    let tb_ips: Vec<String> = peer
                        .ips
                        .iter()
                        .filter(|ip| ip.starts_with("169.254."))
                        .cloned()
                        .collect();
                    if !tb_ips.is_empty() {
                        events.emit(ClusterEvent::RdmaIpsDiscovered {
                            canonical: result.hostname.clone(),
                            ips: tb_ips.clone(),
                            interface: peer.thunderbolt_bridge.clone(),
                        });
                        // Emit RDMA link mapping if we know the local interface + IP
                        if let (Some(iface), Some(local_ip)) =
                            (&peer.thunderbolt_bridge, &peer.local_ip)
                        {
                            for remote_ip in &tb_ips {
                                // Derive RDMA device name: en3 → rdma_en3
                                let rdma_dev_name = format!("rdma_{iface}");
                                events.emit(ClusterEvent::RdmaLinkDiscovered {
                                    local_interface: iface.clone(),
                                    local_ip: local_ip.clone(),
                                    remote_ip: remote_ip.clone(),
                                    remote_hostname: result.hostname.clone(),
                                    rdma_device: Some(rdma_dev_name),
                                    port_state: None, // filled in post-scan correlation
                                });
                            }
                        }
                    }
                    for ip in peer.ips {
                        if ip != result.hostname {
                            events.emit(ClusterEvent::AliasDiscovered {
                                alias: ip,
                                canonical: result.hostname.clone(),
                            });
                        }
                    }
                }
                let mut result = result;
                result.link_speed = peer.link_speed;
                results.push(result);
            }
            Err(e) => warn!(error = %e, "scan task panicked"),
        }
    }

    // Post-scan: get local RDMA device states to correlate with links.
    // Run `rdma_ctl status; ibv_devinfo` locally to get our own RDMA info.
    let local_rdma = match local_run("rdma_ctl status 2>/dev/null; echo '---'; ibv_devices 2>/dev/null; echo '---'; ibv_devinfo 2>/dev/null").await {
        Ok(ref r) if r.has_output() => Some(parse_rdma_status(&r.stdout)),
        _ => None,
    };
    if let Some(ref rdma) = local_rdma {
        // Re-emit links with port state from local RDMA devices
        // This is done via the CorrelateRdmaLinks approach: find device by name
        for device in &rdma.devices {
            // rdma_en3 → en3
            if let Some(iface) = device.name.strip_prefix("rdma_") {
                events.emit(ClusterEvent::RdmaDeviceCorrelated {
                    interface: iface.to_string(),
                    rdma_device: device.name.clone(),
                    port_state: device.port_state,
                });
            }
        }
    }

    // Post-scan: check Thunderbolt network service names concurrently on each SSH-reachable node.
    // Uses `networksetup -listallnetworkservices` to detect duplicates / non-r1o prefixes.
    let tb_futures: Vec<_> = results
        .iter()
        .filter(|r| r.ssh_ok)
        .map(|result| {
            let hostname = result.hostname.clone();
            let config = config.clone();
            let events = events.clone();
            async move {
                let tb_cmd = "networksetup -listallnetworkservices 2>/dev/null | grep -i thunder";
                let tb_services = match ssh_run(&hostname, tb_cmd, &config).await {
                    Ok(ref r) if r.has_output() => parse_thunderbolt_services(&r.stdout),
                    _ => return,
                };
                let issues = find_thunderbolt_issues(&tb_services);
                if !issues.is_empty() {
                    events.emit(ClusterEvent::ThunderboltServiceIssue {
                        hostname,
                        issues,
                    });
                }
            }
        })
        .collect();

    futures::future::join_all(tb_futures).await;

    let online = results.iter().filter(|r| r.ssh_ok).count();
    events.emit(ClusterEvent::ScanComplete { online, total: results.len() });

    results
}

// ---------------------------------------------------------------------------
// Discovery implementations
// ---------------------------------------------------------------------------

/// Discover nodes via Thunderbolt bridge interfaces.
///
/// Two-pass approach:
/// 1. `ifconfig` → find interfaces with 169.254.x.x local IPs (obvious TB bridges)
/// 2. `arp -a` → find any `en*` interface carrying 169.254.x.x *peers* (catches
///    interfaces like en4 that have a non-link-local local IP but still carry TB traffic)
///
/// For each discovered peer IP, SSHs to get the remote hostname.
async fn discover_thunderbolt_bridge(config: &ClusterConfig) -> Vec<DiscoveredPeer> {
    let ifconfig = match local_run("ifconfig 2>/dev/null").await {
        Ok(r) if r.has_output() => r.stdout,
        _ => return Vec::new(),
    };

    // Pass 1: interfaces with 169.254 local IPs
    let bridges = parse_ifconfig_bridges(&ifconfig);
    // Also collect ALL interface IPs for skipping our own addresses
    let all_iface_ips = parse_ifconfig_all_ips(&ifconfig);

    debug!(
        bridge_count = bridges.len(),
        all_iface_count = all_iface_ips.len(),
        "parsed ifconfig"
    );

    // Get ARP table to find peer IPs
    let arp_output = match local_run("arp -a 2>/dev/null").await {
        Ok(r) if r.has_output() => r.stdout,
        _ => return Vec::new(),
    };

    let mut peers = Vec::new();
    let mut seen_ips = std::collections::HashSet::new();

    // Build set of our own IPs (any local address on any interface)
    let our_ips: std::collections::HashSet<&str> = all_iface_ips
        .values()
        .flat_map(|ips| ips.iter().map(|s| s.as_str()))
        .collect();

    // Map from interface → local 169.254 IP (for RDMA link mapping)
    let bridge_ips: HashMap<String, String> = bridges.iter().cloned().collect();

    for line in arp_output.lines() {
        if let Some(caps) = ARP_RE.captures(line) {
            let _arp_host = caps.get(1).unwrap().as_str();
            let ip = caps.get(2).unwrap().as_str();
            let mac = caps.get(3).unwrap().as_str();
            let iface = caps.get(4).unwrap().as_str();

            // Skip incomplete entries
            if mac == "(incomplete)" {
                continue;
            }

            // Only 169.254.x.x peer addresses (link-local = TB direct-connect)
            if !ip.starts_with("169.254.") {
                continue;
            }

            // Must be on an en* interface (Thunderbolt bridges are always en*)
            if !iface.starts_with("en") {
                continue;
            }

            // Skip our own IPs (detected from ifconfig)
            if our_ips.contains(ip) {
                continue;
            }

            // Skip permanent (self) ARP entries
            if line.contains("permanent") {
                continue;
            }

            // Dedup by IP
            if !seen_ips.insert(ip.to_string()) {
                continue;
            }

            // SSH to get hostname
            let hostname = match ssh_run(ip, "hostname -s 2>/dev/null", config).await {
                Ok(ref r) if r.has_output() => r.stdout.trim().to_string(),
                _ => ip.to_string(),
            };

            // Local IP: prefer 169.254 address from bridge_ips, else grab from all_iface_ips
            let local_ip = bridge_ips
                .get(iface)
                .cloned()
                .or_else(|| {
                    all_iface_ips
                        .get(iface)
                        .and_then(|ips| ips.first().cloned())
                });

            peers.push(DiscoveredPeer {
                hostname,
                ips: vec![ip.to_string()],
                discovery_source: "thunderbolt-bridge".to_string(),
                thunderbolt_bridge: Some(iface.to_string()),
                local_ip,
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
///
/// When `filter_non_ssh` is true, skips hostnames that are unlikely to be
/// SSH-able Mac/cluster nodes (IoT devices, phones, etc.) to avoid 10s
/// SSH timeouts that stall the scan.
async fn discover_arp(filter_non_ssh: bool) -> Vec<DiscoveredPeer> {
    let result = local_run("arp -a 2>/dev/null").await;
    match result {
        Ok(ref r) if r.has_output() => parse_arp_table(&r.stdout, filter_non_ssh),
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

    for line in text.lines() {
        if let Some(caps) = IFACE_RE.captures(line) {
            current_iface = Some(caps.get(1).unwrap().as_str().to_string());
        } else if let (Some(caps), Some(iface)) = (INET_LINK_LOCAL_RE.captures(line), &current_iface) {
            let ip = caps.get(1).unwrap().as_str().to_string();
            results.push((iface.clone(), ip));
        }
    }

    results
}

/// Parse `ifconfig` output to get ALL IPv4 addresses per interface.
/// Returns `interface_name → [ip1, ip2, ...]`.
/// Used to identify our own IPs for ARP filtering and to find local IPs on
/// non-link-local TB interfaces (e.g., en4 with 192.168.60.1).
pub fn parse_ifconfig_all_ips(text: &str) -> HashMap<String, Vec<String>> {
    let mut results: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_iface: Option<String> = None;

    for line in text.lines() {
        if let Some(caps) = IFACE_RE.captures(line) {
            current_iface = Some(caps.get(1).unwrap().as_str().to_string());
        } else if let (Some(caps), Some(iface)) = (INET_RE.captures(line), &current_iface) {
            let ip = caps.get(1).unwrap().as_str().to_string();
            results.entry(iface.clone()).or_default().push(ip);
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
    if let Some(peer) = value.get("Self").and_then(extract_tailscale_peer) {
        peers.push(peer);
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
        local_ip: None,
        link_speed: None,
    })
}

/// Parse `arp -a` output to find named hosts (skip `?` entries).
/// Format: `hostname (ip) at mac on interface [...]`
///
/// When `filter_non_ssh` is true, only keeps entries on 169.254.x.x
/// (Thunderbolt link-local) — these are always direct-connected Macs.
/// LAN entries are deferred to `ssh_prefilter` which tests connectivity.
pub fn parse_arp_table(text: &str, filter_non_ssh: bool) -> Vec<DiscoveredPeer> {

    let mut peers_map: HashMap<String, DiscoveredPeer> = HashMap::new();

    for line in text.lines() {
        if let Some(caps) = ARP_RE.captures(line) {
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

            // When filtering, only keep link-local (169.254 = TB direct-connect).
            // LAN entries go through ssh_prefilter separately.
            if filter_non_ssh && !ip.starts_with("169.254.") {
                continue;
            }

            let entry = peers_map
                .entry(normalized.to_string())
                .or_insert_with(|| DiscoveredPeer {
                    hostname: normalized.to_string(),
                    ips: Vec::new(),
                    discovery_source: "arp".to_string(),
                    thunderbolt_bridge: None,
                    local_ip: None,
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
    let mut devices = Vec::new();
    let mut current_device: Option<String> = None;

    for line in text.lines() {
        if let Some(caps) = DEVICE_RE.captures(line) {
            // If we had a previous device without a state, mark it unknown
            if let Some(ref name) = current_device {
                devices.push(RdmaDevice {
                    name: name.clone(),
                    port_state: PortState::Unknown,
                });
            }
            current_device = Some(caps.get(1).unwrap().as_str().to_string());
        } else if let Some(caps) = STATE_RE.captures(line) {
            let state_str = caps.get(1).unwrap().as_str();
            let port_state = PortState::from_ibstat(state_str);
            if let Some(name) = current_device.take() {
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

    // We want to find "Mac*" device names under "Status: Device connected" ports
    let mut in_connected_port = false;
    let mut current_speed: Option<String> = None;

    for line in text.lines() {
        if let Some(caps) = SP_STATUS_RE.captures(line) {
            let status = caps.get(1).unwrap().as_str().trim();
            in_connected_port = status == "Device connected";
        }

        if let Some(caps) = SP_SPEED_RE.captures(line) {
            current_speed = Some(caps.get(1).unwrap().as_str().trim().to_string());
        }

        if let Some(caps) = SP_DEVICE_RE.captures(line) {
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
                        local_ip: None,
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
        let rss_kb: u64 = fields.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
        let rss_mb = if rss_kb > 0 { Some(rss_kb as f64 / 1024.0) } else { None };

        let port = MLX_PORT_RE
            .captures(line)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse().ok());

        let model = MLX_MODEL_RE
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
            footprint_mb: rss_mb,
            distributed,
            server_models: Vec::new(),
        });
    }

    processes
}

/// Parse a `/v1/models` JSON response to extract model IDs.
#[cfg(test)]
fn parse_v1_models_response(json: &str) -> Vec<String> {
    parse_v1_models_metadata(json)
        .into_iter()
        .map(|m| m.id)
        .collect()
}

/// Parse a `/v1/models` JSON response into full model metadata.
///
/// Extracts `id`, `context_length`, and `max_tokens` from each model entry.
/// Returns an empty vec on invalid JSON or missing `data` array.
pub fn parse_v1_models_metadata(json: &str) -> Vec<ModelServerMetadata> {
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let id = item.get("id")?.as_str()?.to_string();
                    Some(ModelServerMetadata {
                        id,
                        context_length: item
                            .get("context_length")
                            .and_then(|v| v.as_u64()),
                        max_tokens: item
                            .get("max_tokens")
                            .and_then(|v| v.as_u64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse ping output for round-trip latency in ms.
fn parse_ping_latency(output: &str) -> Option<f64> {
    // macOS ping: "round-trip min/avg/max/stddev = 0.123/0.456/0.789/0.012 ms"
    PING_LATENCY_RE.captures(output)
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
            local_ip: None,
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
        if entry.local_ip.is_none() {
            entry.local_ip = peer.local_ip;
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
        // m3u1 (macOS, online), m3u3 (macOS, online)
        // Should NOT include: MacBookPro (offline), alienMARIO (windows),
        // k3s-m3u1 (linux), localhost/iOS, etc.

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
        let peers = parse_arp_table(text, false);

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
    fn test_parse_arp_table_filtered() {
        let text = include_str!("../testdata/arp-table.txt");
        let peers = parse_arp_table(text, true);

        let hostnames: Vec<&str> = peers.iter().map(|p| p.hostname.as_str()).collect();

        // Should still find Mac/cluster nodes
        assert!(hostnames.contains(&"m3u1"), "should find m3u1: {hostnames:?}");
        assert!(hostnames.contains(&"m3u2"), "should find m3u2: {hostnames:?}");
        assert!(hostnames.contains(&"m4m1"), "should find m4m1: {hostnames:?}");

        // Should NOT include non-Mac devices (samsung, lifx, amazon, iphone, etc.)
        for peer in &peers {
            let lower = peer.hostname.to_lowercase();
            assert!(
                !lower.contains("samsung") && !lower.contains("lifx")
                    && !lower.contains("amazon") && !lower.contains("iphone")
                    && !lower.contains("docsis"),
                "filtered ARP should not include non-Mac device: {}",
                peer.hostname
            );
        }
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
    fn test_parse_ifconfig_all_ips() {
        let text = include_str!("../testdata/ifconfig-bridges.txt");
        let all_ips = parse_ifconfig_all_ips(text);

        // en4 should be found with its 192.168.60.1 address (the blind spot fix)
        assert!(
            all_ips.contains_key("en4"),
            "should find en4: {all_ips:?}"
        );
        assert!(
            all_ips["en4"].contains(&"192.168.60.1".to_string()),
            "en4 should have 192.168.60.1: {:?}",
            all_ips["en4"]
        );

        // en3 should have 169.254.19.163
        assert!(all_ips["en3"].contains(&"169.254.19.163".to_string()));

        // en5 should have 169.254.124.8
        assert!(all_ips["en5"].contains(&"169.254.124.8".to_string()));
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
            Some("ExampleModel-8bit"),
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
                local_ip: Some("169.254.19.163".to_string()),
                link_speed: None,
            },
            DiscoveredPeer {
                hostname: "m3u1".to_string(),
                ips: vec!["100.127.90.10".to_string()],
                discovery_source: "tailscale".to_string(),
                thunderbolt_bridge: None,
                local_ip: None,
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

    #[test]
    fn test_parse_v1_models_metadata() {
        let json = std::fs::read_to_string("testdata/v1-models-response.json").unwrap();
        let models = parse_v1_models_metadata(&json);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mlx-community/Qwen3-32B-4bit");
        assert_eq!(models[0].context_length, Some(131072));
        assert_eq!(models[0].max_tokens, Some(16384));
    }

    #[test]
    fn test_parse_v1_models_metadata_no_context() {
        let json = r#"{"data":[{"id":"test-model","object":"model"}]}"#;
        let models = parse_v1_models_metadata(json);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test-model");
        assert_eq!(models[0].context_length, None);
        assert_eq!(models[0].max_tokens, None);
    }

    #[test]
    fn test_parse_v1_models_metadata_empty() {
        let models = parse_v1_models_metadata("{}");
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_v1_models_metadata_invalid_json() {
        let models = parse_v1_models_metadata("not json");
        assert!(models.is_empty());
    }
}
