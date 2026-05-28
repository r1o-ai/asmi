//! Cluster topology discovery for TB5/RDMA mesh.
//!
//! Primary: native HTTP + TB UUID cross-matching via each node's asmi /thunderbolt endpoint.
//! Fallback 1: `mlx.distributed_config --dot` (requires Python/MLX).
//! Fallback 2: ARP-based discovery.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A single TB5/RDMA link between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyLink {
    pub node_a: String,
    pub device_a: String,
    pub node_b: String,
    pub device_b: String,
}

/// Full cluster topology report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyReport {
    pub nodes: Vec<String>,
    pub links: Vec<TopologyLink>,
    pub mesh_complete: bool,
    pub missing_links: Vec<(String, String)>,
    pub jaccl_ready: bool,
    /// Fully-meshed subsets of 3+ nodes that can run JACCL.
    pub jaccl_ready_subsets: Vec<Vec<String>>,
    pub raw_dot: String,
}

/// Which `mlx.distributed_config` binary to use.
fn find_distributed_config() -> Result<String> {
    // Check common locations
    for path in &[
        "/opt/homebrew/bin/mlx.distributed_config",
        "/usr/local/bin/mlx.distributed_config",
    ] {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    // Try PATH
    let which = Command::new("which")
        .arg("mlx.distributed_config")
        .output();
    if let Ok(out) = which {
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(path);
            }
        }
    }
    bail!("mlx.distributed_config not found. Install with: pip install mlx")
}

/// Discover cluster topology. Tries native HTTP first (no Python/MLX dependency),
/// falls back to mlx.distributed_config, then ARP.
pub fn discover_topology(hosts: &[String], backend: &str) -> Result<TopologyReport> {
    // Primary: native HTTP + UUID cross-match (no SSH, no Python)
    let http_reachable: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts
            .iter()
            .map(|h| {
                let h = h.clone();
                s.spawn(move || {
                    if is_host_reachable_http(&h) { Some(h) } else { None }
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });

    for h in hosts {
        if !http_reachable.contains(h) {
            eprintln!("topology: {h} not reachable via HTTP");
        }
    }

    if http_reachable.len() >= 2 {
        match discover_via_native(&http_reachable) {
            Ok(report) => {
                eprintln!(
                    "topology: native discovery found {} links across {} nodes",
                    report.links.len(),
                    report.nodes.len()
                );
                return finalize_report(report, hosts);
            }
            Err(e) => eprintln!("native topology failed ({e:#}), trying mlx fallback"),
        }
    }

    // Fallback 1: mlx.distributed_config (kept permanently for nodes without asmi)
    let ssh_reachable: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts
            .iter()
            .map(|h| {
                let h = h.clone();
                s.spawn(move || {
                    if is_host_reachable_ssh(&h) { Some(h) } else { None }
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });

    if ssh_reachable.len() >= 2 {
        match discover_via_mlx(&ssh_reachable, backend) {
            Ok(report) => return finalize_report(report, hosts),
            Err(e) => eprintln!("mlx.distributed_config failed ({e:#}), trying ARP"),
        }
    }

    // Fallback 2: ARP-based
    discover_via_arp(hosts)
}

fn finalize_report(mut report: TopologyReport, all_hosts: &[String]) -> Result<TopologyReport> {
    for h in all_hosts {
        if !report.nodes.contains(h) {
            report.nodes.push(h.clone());
        }
    }
    let missing = compute_missing_links(&report.nodes, &report.links);
    report.mesh_complete = missing.is_empty();
    report.missing_links = missing;
    report.jaccl_ready_subsets = find_jaccl_subsets_from_links(&report.nodes, &report.links);
    report.jaccl_ready = compute_jaccl_ready(&report.links, &report.jaccl_ready_subsets);
    Ok(report)
}

/// Quick SSH reachability check (3s timeout, BatchMode).
fn is_host_reachable_ssh(host: &str) -> bool {
    let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
    Command::new("ssh")
        .args([
            "-o", "ConnectTimeout=3",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            &format!("{user}@{host}"),
            "true",
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compute missing links (node pairs not directly connected).
fn compute_missing_links(nodes: &[String], links: &[TopologyLink]) -> Vec<(String, String)> {
    let connected: HashSet<(String, String)> = links.iter().map(|l| {
        if l.node_a < l.node_b {
            (l.node_a.clone(), l.node_b.clone())
        } else {
            (l.node_b.clone(), l.node_a.clone())
        }
    }).collect();
    let mut missing = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let pair = (nodes[i].clone(), nodes[j].clone());
            if !connected.contains(&pair) {
                missing.push(pair);
            }
        }
    }
    missing
}

/// Build JACCL-ready subsets from link data (avoids duplicating the connected set logic).
fn find_jaccl_subsets_from_links(nodes: &[String], links: &[TopologyLink]) -> Vec<Vec<String>> {
    let connected: HashSet<(String, String)> = links.iter().map(|l| {
        if l.node_a < l.node_b {
            (l.node_a.clone(), l.node_b.clone())
        } else {
            (l.node_b.clone(), l.node_a.clone())
        }
    }).collect();
    find_jaccl_subsets(nodes, &connected)
}

/// Fallback path: `mlx.distributed_config --dot` (requires Python/MLX).
fn discover_via_mlx(hosts: &[String], backend: &str) -> Result<TopologyReport> {
    let bin = find_distributed_config()?;
    let hosts_arg = hosts.join(",");

    let output = Command::new(&bin)
        .args([
            "--hosts",
            &hosts_arg,
            "--over",
            "thunderbolt",
            "--backend",
            backend,
            "--dot",
        ])
        .output()
        .context("Failed to run mlx.distributed_config")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let dot = if stdout.contains("graph G") {
        stdout.clone()
    } else if stderr.contains("graph G") {
        stderr
            .lines()
            .skip_while(|l| !l.contains("graph G"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        String::new()
    };

    if dot.is_empty() {
        bail!(
            "mlx.distributed_config produced no topology output.\nstderr: {}",
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        );
    }

    parse_dot(&dot, hosts)
}

/// Fallback: discover mesh by SSHing to each node, collecting link-local IPs
/// and ARP tables, then cross-referencing to find which interfaces connect.
fn discover_via_arp(hosts: &[String]) -> Result<TopologyReport> {
    use std::collections::BTreeMap;

    // Collect ifconfig IPs and ARP from each node in parallel (threads, not async)
    let results: Vec<(String, BTreeMap<String, String>, Vec<(String, String)>)> =
        std::thread::scope(|s| {
            let handles: Vec<_> = hosts
                .iter()
                .map(|host| {
                    let h = host.clone();
                    s.spawn(move || {
                        let iface_ips = collect_link_local_ips(&h);
                        let arp = collect_arp_peers(&h);
                        (h, iface_ips, arp)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

    // Build IP → (hostname, interface) lookup
    let mut ip_to_node: HashMap<String, (String, String)> = HashMap::new();
    for (host, iface_ips, _) in &results {
        for (iface, ip) in iface_ips {
            ip_to_node.insert(ip.clone(), (host.clone(), iface.clone()));
        }
    }

    // Find links by matching ARP entries to known node IPs
    let mut seen_links: HashSet<(String, String)> = HashSet::new();
    let mut links: Vec<TopologyLink> = Vec::new();

    for (host, _iface_ips, arp_entries) in &results {
        for (remote_ip, local_iface) in arp_entries {
            if let Some((remote_host, remote_iface)) = ip_to_node.get(remote_ip) {
                if remote_host == host {
                    continue; // skip self
                }
                let (a, b) = if host < remote_host {
                    (host.clone(), remote_host.clone())
                } else {
                    (remote_host.clone(), host.clone())
                };
                if seen_links.insert((a.clone(), b.clone())) {
                    // Use iface: prefix — these are network interface names (en15),
                    // NOT ibverbs RDMA device names (rdma_en5). Consumers must
                    // validate against ibv_devices before using for JACCL init.
                    let (dev_a, dev_b) = if host < remote_host {
                        (format!("iface:{local_iface}"), format!("iface:{remote_iface}"))
                    } else {
                        (format!("iface:{remote_iface}"), format!("iface:{local_iface}"))
                    };
                    links.push(TopologyLink {
                        node_a: a,
                        device_a: dev_a,
                        node_b: b,
                        device_b: dev_b,
                    });
                }
            }
        }
    }

    // Build connectivity set and check mesh completeness
    let connected: HashSet<(String, String)> = seen_links;
    let mut missing: Vec<(String, String)> = Vec::new();
    let nodes = hosts.to_vec();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let pair = (nodes[i].clone(), nodes[j].clone());
            if !connected.contains(&pair) {
                missing.push(pair);
            }
        }
    }

    let mesh_complete = missing.is_empty();
    let subsets = find_jaccl_subsets(&nodes, &connected);
    let jaccl_ready = compute_jaccl_ready(&links, &subsets);
    let raw_dot = generate_dot(&nodes, &links);

    Ok(TopologyReport {
        nodes,
        links,
        mesh_complete,
        missing_links: missing,
        jaccl_ready,
        jaccl_ready_subsets: subsets,
        raw_dot,
    })
}

/// Collect link-local (169.254.x.x) IPs from a node via SSH `ifconfig`.
fn collect_link_local_ips(host: &str) -> BTreeMap<String, String> {
    use std::collections::BTreeMap;
    let output = ssh_cmd(host, "ifconfig 2>/dev/null");
    let mut result = BTreeMap::new();
    let mut current_iface = String::new();
    for line in output.lines() {
        if !line.starts_with('\t') && !line.starts_with(' ') {
            if let Some(name) = line.split(':').next() {
                current_iface = name.to_string();
            }
        }
        if line.contains("inet 169.254.") {
            if let Some(ip) = line.split_whitespace().nth(1) {
                if !current_iface.is_empty()
                    && !current_iface.starts_with("lo")
                    && !current_iface.starts_with("bridge")
                {
                    result.insert(current_iface.clone(), ip.to_string());
                }
            }
        }
    }
    result
}

/// Collect resolved ARP entries for 169.254.x.x peers, returning (remote_ip, local_interface).
fn collect_arp_peers(host: &str) -> Vec<(String, String)> {
    let output = ssh_cmd(host, "arp -an 2>/dev/null | grep '169.254.' | grep -v incomplete");
    let mut peers = Vec::new();
    for line in output.lines() {
        // Format: ? (169.254.x.x) at MAC on enN [ethernet]
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 6 && parts[3] != "(incomplete)" {
            let ip = parts[1].trim_matches(|c| c == '(' || c == ')').to_string();
            let iface = parts[5].to_string();
            // Skip bridge interfaces — they don't map to a single TB port
            if !iface.starts_with("bridge") {
                peers.push((ip, iface));
            }
        }
    }
    peers
}

/// Run an SSH command synchronously and return stdout.
fn ssh_cmd(host: &str, cmd: &str) -> String {
    let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
    Command::new("ssh")
        .args([
            "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            &format!("{user}@{host}"),
            cmd,
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

// ── Native topology: HTTP + UUID cross-match ────────────────────────

fn fetch_thunderbolt(host: &str) -> Option<serde_json::Value> {
    for suffix in ["", ".local"] {
        let url = format!("http://{}{}:9090/thunderbolt", host, suffix);
        if let Some(output) = Command::new("curl")
            .args(["-s", "--connect-timeout", "5", "--max-time", "5", &url])
            .output()
            .ok()
        {
            if output.status.success() {
                if let Ok(v) = serde_json::from_slice(&output.stdout) {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn is_host_reachable_http(host: &str) -> bool {
    for suffix in ["", ".local"] {
        let url = format!("http://{}{}:9090/health", host, suffix);
        let ok = Command::new("curl")
            .args(["-s", "--connect-timeout", "3", "--max-time", "3", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).starts_with("200"))
            .unwrap_or(false);
        if ok { return true; }
    }
    false
}

struct NativeBusInfo {
    _bus_index: u8,
    domain_uuid: String,
    peer_domain_uuid: Option<String>,
    interface: String,
}

fn parse_thunderbolt_response(data: &serde_json::Value) -> (HashMap<String, String>, Vec<NativeBusInfo>) {
    let bus_map: HashMap<String, String> = data
        .get("bus_map")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let mut buses = Vec::new();
    if let Some(ports) = data.get("ports").and_then(|v| v.as_array()) {
        for port in ports {
            let bus_index = match port.get("bus_index").and_then(|v| v.as_u64()) {
                Some(n) => n as u8,
                None => continue,
            };
            let domain_uuid = match port.get("domain_uuid").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            let peer_domain_uuid = port
                .get("peer_domain_uuid")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from);
            let interface = bus_map
                .get(&bus_index.to_string())
                .cloned()
                .unwrap_or_else(|| format!("en{}", bus_index + 2));
            buses.push(NativeBusInfo { _bus_index: bus_index, domain_uuid, peer_domain_uuid, interface });
        }
    }
    (bus_map, buses)
}

fn discover_via_native(hosts: &[String]) -> Result<TopologyReport> {
    let node_data: Vec<(String, serde_json::Value)> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts
            .iter()
            .map(|h| {
                let h = h.clone();
                s.spawn(move || fetch_thunderbolt(&h).map(|v| (h, v)))
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect()
    });

    if node_data.len() < 2 {
        bail!(
            "native topology: need ≥2 reachable nodes with asmi, got {}",
            node_data.len()
        );
    }

    let parsed: Vec<(String, Vec<NativeBusInfo>)> = node_data
        .iter()
        .map(|(hostname, data)| {
            let (_bus_map, buses) = parse_thunderbolt_response(data);
            (hostname.clone(), buses)
        })
        .collect();

    // Build domain_uuid → (hostname, interface) lookup from all buses
    let mut uuid_to_node: HashMap<String, (String, String)> = HashMap::new();
    for (hostname, buses) in &parsed {
        for bus in buses {
            uuid_to_node.insert(
                bus.domain_uuid.clone(),
                (hostname.clone(), bus.interface.clone()),
            );
        }
    }

    // UUID cross-match: find links where nodeA.peer_domain == nodeB.local_domain
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut links: Vec<TopologyLink> = Vec::new();

    for (hostname, buses) in &parsed {
        for bus in buses {
            let peer_uuid = match &bus.peer_domain_uuid {
                Some(u) => u,
                None => continue,
            };
            let (remote_host, remote_iface) = match uuid_to_node.get(peer_uuid) {
                Some(entry) if &entry.0 != hostname => entry,
                _ => continue,
            };
            let pair = if hostname < remote_host {
                (hostname.clone(), remote_host.clone())
            } else {
                (remote_host.clone(), hostname.clone())
            };
            if !seen.insert(pair) {
                continue;
            }
            let (node_a, dev_a, node_b, dev_b) = if hostname < remote_host {
                (
                    hostname.clone(),
                    format!("rdma_{}", bus.interface),
                    remote_host.clone(),
                    format!("rdma_{remote_iface}"),
                )
            } else {
                (
                    remote_host.clone(),
                    format!("rdma_{remote_iface}"),
                    hostname.clone(),
                    format!("rdma_{}", bus.interface),
                )
            };
            links.push(TopologyLink { node_a, device_a: dev_a, node_b, device_b: dev_b });
        }
    }

    let nodes: Vec<String> = hosts.to_vec();
    let raw_dot = generate_dot(&nodes, &links);
    let missing = compute_missing_links(&nodes, &links);
    let mesh_complete = missing.is_empty();
    let jaccl_ready_subsets = find_jaccl_subsets_from_links(&nodes, &links);
    let jaccl_ready = compute_jaccl_ready(&links, &jaccl_ready_subsets);

    Ok(TopologyReport {
        nodes,
        links,
        mesh_complete,
        missing_links: missing,
        jaccl_ready,
        jaccl_ready_subsets,
        raw_dot,
    })
}

// ── DOT graph generation ────────────────────────────────────────────

/// Generate a DOT graph from discovered topology.
fn generate_dot(nodes: &[String], links: &[TopologyLink]) -> String {
    let mut dot = String::from("graph G {\n  node [shape=rectangle];\n");
    let ids: Vec<String> = (b'a'..).take(nodes.len()).map(|c| String::from(c as char)).collect();
    let name_to_id: HashMap<&str, &str> = nodes
        .iter()
        .zip(ids.iter())
        .map(|(n, id)| (n.as_str(), id.as_str()))
        .collect();
    for (name, id) in nodes.iter().zip(ids.iter()) {
        dot.push_str(&format!("  {id} [label=\"{name}\"];\n"));
    }
    for link in links {
        let id_a = name_to_id.get(link.node_a.as_str()).unwrap_or(&"?");
        let id_b = name_to_id.get(link.node_b.as_str()).unwrap_or(&"?");
        let dev_a = link.device_a.strip_prefix("rdma_").unwrap_or(&link.device_a);
        let dev_b = link.device_b.strip_prefix("rdma_").unwrap_or(&link.device_b);
        dot.push_str(&format!("  {id_a} -- {id_b} [label=\"{dev_a}/{dev_b}\"]\n"));
    }
    dot.push_str("}\n");
    dot
}

/// Parse DOT graph output into a TopologyReport.
///
/// Expected format:
/// ```dot
/// graph G {
///   node [shape=rectangle];
///   a [label="m3u1"];
///   b [label="m3u2"];
///   a -- b [label="en3/en4"]
/// }
/// ```
fn parse_dot(dot: &str, requested_hosts: &[String]) -> Result<TopologyReport> {
    let mut id_to_name: HashMap<String, String> = HashMap::new();
    let mut links: Vec<TopologyLink> = Vec::new();

    for line in dot.lines() {
        let line = line.trim();

        // Parse node declarations: `a [label="m3u1"];`
        if line.contains("[label=") && !line.contains("--") && !line.contains("shape=") {
            if let Some((id, rest)) = line.split_once(' ') {
                if let Some(label) = extract_quoted(rest, "label=") {
                    id_to_name.insert(id.to_string(), label);
                }
            }
        }

        // Parse edge declarations: `a -- b [label="en3/en4"]`
        if line.contains("--") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[1] == "--" {
                let id_a = parts[0];
                let id_b = parts[2].trim_end_matches(';');

                let name_a = id_to_name
                    .get(id_a)
                    .cloned()
                    .unwrap_or_else(|| id_a.to_string());
                let name_b = id_to_name
                    .get(id_b)
                    .cloned()
                    .unwrap_or_else(|| id_b.to_string());

                // Extract RDMA device names from label: "en3/en4"
                let (dev_a, dev_b) = if let Some(label) = extract_quoted(line, "label=") {
                    if let Some((a, b)) = label.split_once('/') {
                        (format!("rdma_{a}"), format!("rdma_{b}"))
                    } else {
                        (label.clone(), label)
                    }
                } else {
                    ("unknown".to_string(), "unknown".to_string())
                };

                links.push(TopologyLink {
                    node_a: name_a,
                    device_a: dev_a,
                    node_b: name_b,
                    device_b: dev_b,
                });
            }
        }
    }

    // Use requested hosts as the canonical node list
    let nodes: Vec<String> = requested_hosts.to_vec();

    // Build set of connected pairs
    let mut connected: HashSet<(String, String)> = HashSet::new();
    for link in &links {
        let (a, b) = if link.node_a < link.node_b {
            (link.node_a.clone(), link.node_b.clone())
        } else {
            (link.node_b.clone(), link.node_a.clone())
        };
        connected.insert((a, b));
    }

    // Find missing links (full mesh = every pair connected)
    let mut missing: Vec<(String, String)> = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let pair = (nodes[i].clone(), nodes[j].clone());
            if !connected.contains(&pair) {
                missing.push(pair);
            }
        }
    }

    let mesh_complete = missing.is_empty();
    let subsets = find_jaccl_subsets(&nodes, &connected);
    let jaccl_ready = compute_jaccl_ready(&links, &subsets);

    Ok(TopologyReport {
        nodes,
        links,
        mesh_complete,
        missing_links: missing,
        jaccl_ready,
        jaccl_ready_subsets: subsets,
        raw_dot: dot.to_string(),
    })
}

/// Find all fully-meshed subsets of 3+ nodes.
fn find_jaccl_subsets(
    nodes: &[String],
    connected: &HashSet<(String, String)>,
) -> Vec<Vec<String>> {
    let mut subsets = Vec::new();
    let n = nodes.len();

    // Check all subsets of size 3 to n-1
    for size in (3..n).rev() {
        for combo in combinations(n, size) {
            let subset: Vec<String> = combo.iter().map(|&i| nodes[i].clone()).collect();
            if is_full_mesh(&subset, connected) {
                // Don't include subsets that are subsets of already-found ones
                let dominated = subsets.iter().any(|s: &Vec<String>| {
                    subset.iter().all(|n| s.contains(n))
                });
                if !dominated {
                    subsets.push(subset);
                }
            }
        }
    }

    subsets
}

/// True when all RDMA-capable nodes (those appearing in any link) form a full mesh.
fn compute_jaccl_ready(links: &[TopologyLink], _subsets: &[Vec<String>]) -> bool {
    let rdma_nodes: HashSet<&str> = links.iter()
        .flat_map(|l| [l.node_a.as_str(), l.node_b.as_str()])
        .collect();
    let rn = rdma_nodes.len();
    if rn < 2 { return false; }
    let connected: HashSet<(String, String)> = links.iter().map(|l| {
        if l.node_a < l.node_b {
            (l.node_a.clone(), l.node_b.clone())
        } else {
            (l.node_b.clone(), l.node_a.clone())
        }
    }).collect();
    let rdma_vec: Vec<String> = rdma_nodes.into_iter().map(String::from).collect();
    is_full_mesh(&rdma_vec, &connected)
}

/// Check if all pairs in the subset are connected.
fn is_full_mesh(subset: &[String], connected: &HashSet<(String, String)>) -> bool {
    for i in 0..subset.len() {
        for j in (i + 1)..subset.len() {
            let pair = if subset[i] < subset[j] {
                (subset[i].clone(), subset[j].clone())
            } else {
                (subset[j].clone(), subset[i].clone())
            };
            if !connected.contains(&pair) {
                return false;
            }
        }
    }
    true
}

/// Generate all combinations of `k` items from `0..n`.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    fn recurse(start: usize, depth: usize, n: usize, k: usize, combo: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if depth == k {
            result.push(combo.clone());
            return;
        }
        for i in start..=(n - k + depth) {
            combo[depth] = i;
            recurse(i + 1, depth + 1, n, k, combo, result);
        }
    }
    recurse(0, 0, n, k, &mut combo, &mut result);
    result
}

/// Extract a quoted value from a DOT attribute string.
/// e.g., `[label="m3u1"]` → `"m3u1"`
fn extract_quoted(s: &str, key: &str) -> Option<String> {
    let idx = s.find(key)?;
    let after = &s[idx + key.len()..];
    let start = after.find('"')? + 1;
    let end = after[start..].find('"')? + start;
    Some(after[start..end].to_string())
}

/// Format the topology as a human-readable table.
pub fn format_table(report: &TopologyReport) -> String {
    let mut out = String::new();
    out.push_str("=== CLUSTER TOPOLOGY ===\n\n");

    // Adjacency matrix header
    let nodes = &report.nodes;
    let width = nodes.iter().map(|n| n.len()).max().unwrap_or(6).max(6);

    out.push_str(&format!("{:>width$}", "", width = width));
    for n in nodes {
        out.push_str(&format!(" {:>width$}", n, width = width));
    }
    out.push('\n');

    // Build lookup: (a, b) → (device_a, device_b)
    let mut link_map: HashMap<(String, String), (String, String)> = HashMap::new();
    for link in &report.links {
        link_map.insert(
            (link.node_a.clone(), link.node_b.clone()),
            (link.device_a.clone(), link.device_b.clone()),
        );
        link_map.insert(
            (link.node_b.clone(), link.node_a.clone()),
            (link.device_b.clone(), link.device_a.clone()),
        );
    }

    for a in nodes {
        out.push_str(&format!("{:>width$}", a, width = width));
        for b in nodes {
            if a == b {
                out.push_str(&format!(" {:>width$}", "—", width = width));
            } else if let Some((dev, _)) = link_map.get(&(a.clone(), b.clone())) {
                out.push_str(&format!(" {:>width$}", dev.replace("rdma_", ""), width = width));
            } else {
                out.push_str(&format!(" {:>width$}", "MISS", width = width));
            }
        }
        out.push('\n');
    }

    out.push('\n');
    let total_expected = nodes.len() * (nodes.len() - 1) / 2;
    out.push_str(&format!(
        "Links: {}/{} | JACCL ready: {}\n",
        report.links.len(),
        total_expected,
        if report.jaccl_ready { "YES" } else { "NO" }
    ));

    if !report.missing_links.is_empty() {
        out.push_str("\nMissing links:\n");
        for (a, b) in &report.missing_links {
            out.push_str(&format!("  {} ↔ {} — add a TB5 cable\n", a, b));
        }
    }

    if !report.jaccl_ready && !report.jaccl_ready_subsets.is_empty() {
        out.push_str("\nJACCL-ready subsets:\n");
        for subset in &report.jaccl_ready_subsets {
            out.push_str(&format!("  {} ({} nodes)\n", subset.join(", "), subset.len()));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DOT: &str = r#"graph G {
  node [shape=rectangle];
  a [label="m3u1"];
  b [label="m3u2"];
  c [label="m3u3"];
  d [label="m4m1"];
  a -- b [label="en3/en4"]
  a -- d [label="en4/en4"]
  b -- c [label="en3/en3"]
  b -- d [label="en5/en3"]
  c -- d [label="en5/en5"]
}"#;

    #[test]
    fn test_parse_dot_incomplete_mesh() {
        let hosts = vec![
            "m3u1".into(),
            "m3u2".into(),
            "m3u3".into(),
            "m4m1".into(),
        ];
        let report = parse_dot(SAMPLE_DOT, &hosts).unwrap();

        assert_eq!(report.nodes.len(), 4);
        assert_eq!(report.links.len(), 5);
        assert!(!report.mesh_complete);
        assert!(!report.jaccl_ready);
        assert_eq!(report.missing_links.len(), 1);
        assert_eq!(report.missing_links[0], ("m3u1".to_string(), "m3u3".to_string()));
    }

    #[test]
    fn test_parse_dot_link_devices() {
        let hosts = vec!["m3u1".into(), "m3u2".into()];
        let dot = r#"graph G {
  node [shape=rectangle];
  a [label="m3u1"];
  b [label="m3u2"];
  a -- b [label="en3/en4"]
}"#;
        let report = parse_dot(dot, &hosts).unwrap();
        assert_eq!(report.links.len(), 1);
        assert_eq!(report.links[0].device_a, "rdma_en3");
        assert_eq!(report.links[0].device_b, "rdma_en4");
        assert!(report.mesh_complete);
        assert!(report.jaccl_ready);
    }

    #[test]
    fn test_jaccl_subsets() {
        let hosts = vec![
            "m3u1".into(),
            "m3u2".into(),
            "m3u3".into(),
            "m4m1".into(),
        ];
        let report = parse_dot(SAMPLE_DOT, &hosts).unwrap();

        // Should find 3-node subsets that are fully meshed
        assert!(!report.jaccl_ready_subsets.is_empty());
        // m3u2, m3u3, m4m1 should be one (all connected)
        let has_bcd = report.jaccl_ready_subsets.iter().any(|s| {
            s.contains(&"m3u2".to_string())
                && s.contains(&"m3u3".to_string())
                && s.contains(&"m4m1".to_string())
        });
        assert!(has_bcd, "m3u2/m3u3/m4m1 should be a valid subset");
    }

    #[test]
    fn test_format_table() {
        let hosts = vec!["m3u1".into(), "m3u2".into()];
        let dot = r#"graph G {
  node [shape=rectangle];
  a [label="m3u1"];
  b [label="m3u2"];
  a -- b [label="en3/en4"]
}"#;
        let report = parse_dot(dot, &hosts).unwrap();
        let table = format_table(&report);
        assert!(table.contains("JACCL ready: YES"));
        assert!(table.contains("en3"));
    }
}
