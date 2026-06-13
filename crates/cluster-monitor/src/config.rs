//! Cluster configuration — zero hardcoded nodes.
//!
//! All node discovery is dynamic: Thunderbolt bridge scanning, system-profiler,
//! mDNS/Bonjour, Tailscale status, or ARP table inspection.

use crate::aggregator::canonicalize_hostname;
use crate::types::RdmaLink;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;

/// How to discover cluster nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    /// Scan Thunderbolt bridge interfaces (169.254.x.x on en3-en23).
    ThunderboltBridge,
    /// Parse `system_profiler SPThunderboltDataType` for connected devices.
    SystemProfiler,
    /// mDNS / Bonjour `.local` resolution.
    Bonjour,
    /// `tailscale status --json` for peers on the tailnet.
    Tailscale,
    /// Parse ARP table — filtered to likely Mac/cluster devices.
    Arp,
    /// Parse ARP table — all named hosts, no filtering.
    ArpAll,
}

/// Configuration for the cluster monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Discovery methods to try, in order of preference.
    pub discovery: Vec<DiscoveryMethod>,

    /// Optional seed hostnames / IPs to always probe (in addition to
    /// discovered nodes). This is the only way to provide "known" hosts
    /// without hardcoding them in code.
    pub seed_hosts: Vec<String>,

    /// SSH connect timeout.
    #[serde(with = "humantime_serde")]
    pub ssh_timeout: Duration,

    /// How often to poll metrics from each node.
    #[serde(with = "humantime_serde")]
    pub poll_interval: Duration,

    /// How often to run a full scan / probe.
    #[serde(with = "humantime_serde")]
    pub scan_interval: Duration,

    /// SSH user (defaults to current user).
    pub ssh_user: Option<String>,

    /// SSH identity file (defaults to ~/.ssh/id_ed25519).
    pub ssh_identity: Option<String>,

    /// Number of metrics history samples to keep per node.
    pub history_capacity: usize,

    /// Port the asmi HTTP daemon listens on (default: 9090).
    /// Used for HTTP-first metrics fetching on remote nodes.
    pub daemon_port: u16,

    /// When true, the local collector falls back to `sudo powermetrics`
    /// for ANE/GPU power detail if the asmi-helper Unix socket is
    /// unreachable. When false (the default), the fallback is skipped
    /// and the snapshot only carries IOReport-sourced power values.
    ///
    /// Rationale: the powermetrics fallback spawns a subprocess on every
    /// poll cycle (default 2s). That keeps PerfPowerServices warm and
    /// can pin a couple of CPU cores on machines where asmi-helper
    /// isn't running. Cluster monitoring rarely needs the extra ANE
    /// precision — but when it does, flip this on via Settings → System
    /// or by editing ~/.config/asmi/config.json directly.
    #[serde(default)]
    pub ane_power_check: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            discovery: vec![
                DiscoveryMethod::ThunderboltBridge,
                DiscoveryMethod::Arp,
            ],
            seed_hosts: Vec::new(),
            ssh_timeout: Duration::from_secs(5),
            poll_interval: Duration::from_secs(2),
            scan_interval: Duration::from_secs(30),
            ssh_user: None,
            ssh_identity: None,
            history_capacity: 300,
            daemon_port: 9090,
            ane_power_check: false,
        }
    }
}

impl ClusterConfig {
    /// SSH connect timeout as whole seconds (for the `-o ConnectTimeout=N` flag).
    pub fn ssh_timeout_secs(&self) -> u64 {
        self.ssh_timeout.as_secs().max(1)
    }

    /// Build with custom seed hosts.
    pub fn with_seeds(mut self, seeds: Vec<String>) -> Self {
        self.seed_hosts = seeds;
        self
    }

    /// Build with custom discovery methods.
    pub fn with_discovery(mut self, methods: Vec<DiscoveryMethod>) -> Self {
        self.discovery = methods;
        self
    }

    /// Build with a custom SSH user.
    pub fn with_ssh_user(mut self, user: impl Into<String>) -> Self {
        self.ssh_user = Some(user.into());
        self
    }

    /// Build with a custom poll interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }
}

// ---------------------------------------------------------------------------
// NodeMap — persistent hostname alias map
// ---------------------------------------------------------------------------

/// Persistent hostname alias map and RDMA host registry.
/// Maps discovered names to canonical SSH hostnames and stores
/// Thunderbolt bridge IPs for RDMA/distributed inference.
///
/// # Example config.json
/// ```json
/// {
///   "aliases": { "device-a": "node-a", "169.254.10.1": "node-b" },
///   "nodes": ["node-a", "node-b", "node-c", "node-d"],
///   "rdma_ips": {
///     "node-b": ["169.254.10.1", "169.254.10.2"],
///     "node-a": ["169.254.10.3"]
///   }
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMap {
    /// Hostname aliases: discovered_name → canonical SSH hostname.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
    /// Known canonical node hostnames (the cluster).
    #[serde(default)]
    pub nodes: Vec<String>,
    /// When true, the local collector falls back to `sudo powermetrics`
    /// for ANE/GPU power detail when the asmi-helper Unix socket is
    /// unreachable. False by default — see ClusterConfig::ane_power_check
    /// for the full rationale. Persisted here so a Settings toggle in
    /// the r1o web UI can flip it without a daemon restart pulling args
    /// from a different source.
    #[serde(default)]
    pub ane_power_check: bool,
    /// Per-node Thunderbolt bridge IPs for RDMA, keyed by canonical hostname.
    /// Dynamically populated from Thunderbolt bridge discovery.
    #[serde(default)]
    pub rdma_ips: HashMap<String, Vec<String>>,
    /// RDMA link topology: local interface → remote peer mappings.
    /// Maps which local RDMA device (rdma_enX → enX) connects to which peer.
    #[serde(default)]
    pub rdma_links: Vec<RdmaLink>,
    /// Custom Python interpreter path for MLX operations.
    /// If set, overrides the default `resolve_python()` auto-detection.
    /// Example: "~/venvs/hermes/bin/python3"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_cmd: Option<String>,
}

impl NodeMap {
    /// Path to the persistent config file.
    /// Respects `XDG_CONFIG_HOME` if set, otherwise `~/.config/asmi/config.json`.
    pub fn config_path() -> PathBuf {
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            let p = PathBuf::from(&xdg);
            if p.is_absolute() {
                return p.join("asmi").join("config.json");
            }
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".config")
            .join("asmi")
            .join("config.json")
    }

    /// Load from disk. If `~/.config/asmi/config.json` doesn't exist, seeds
    /// the node list from `~/.r1o/cluster.json` (the shared cluster inventory)
    /// and persists it so subsequent loads are self-contained.
    pub fn load() -> Self {
        let path = Self::config_path();
        if let Some(nm) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<Self>(&s).ok())
        {
            return nm;
        }

        // Seed from ~/.r1o/cluster.json if it exists
        let mut nm = Self::default();
        if let Some(nodes) = Self::load_r1o_cluster_nodes() {
            for node in &nodes {
                nm.register_node(node);
            }
            nm.save();
            tracing::info!(
                nodes = ?nm.nodes,
                "seeded NodeMap from ~/.r1o/cluster.json ({} nodes)",
                nm.nodes.len()
            );
        }
        nm
    }

    /// Read node hostnames from `~/.r1o/cluster.json`.
    /// Format: `{ "nodes": [{ "hostname": "hub", ... }, ...] }`
    fn load_r1o_cluster_nodes() -> Option<Vec<String>> {
        let path = dirs::home_dir()?.join(".r1o").join("cluster.json");
        let content = std::fs::read_to_string(&path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        let nodes = json.get("nodes")?.as_array()?;
        let hostnames: Vec<String> = nodes
            .iter()
            .filter_map(|n| n.get("hostname")?.as_str().map(String::from))
            .collect();
        if hostnames.is_empty() { None } else { Some(hostnames) }
    }

    /// Save to disk.
    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Resolve a hostname through the alias map. Returns canonical name.
    pub fn resolve<'a>(&'a self, hostname: &'a str) -> &'a str {
        self.aliases
            .get(hostname)
            .or_else(|| self.aliases.get(&hostname.to_lowercase()))
            .map(|s| s.as_str())
            .unwrap_or(hostname)
    }

    /// Register a canonical hostname in the nodes list. Returns true if new.
    ///
    /// Strips mDNS rename-cascade suffixes (`hub-2`, `m3u4-2237`, `hub-2-3`)
    /// at the writer so phantom entries never reach `self.nodes` regardless of
    /// the discovery path that found them (Bonjour browse, ARP scan, HTTP
    /// probe, etc.). See [`crate::aggregator::canonicalize_hostname`].
    pub fn register_node(&mut self, hostname: &str) -> bool {
        if hostname.is_empty() {
            return false;
        }
        let real: HashSet<String> = self.nodes.iter().map(|n| n.to_lowercase()).collect();
        let canonical = canonicalize_hostname(hostname, &real);
        if canonical.is_empty() {
            return false;
        }
        if !self.nodes.iter().any(|n| n.eq_ignore_ascii_case(&canonical)) {
            self.nodes.push(canonical);
            self.nodes.sort();
            true
        } else {
            false
        }
    }

    /// Add an alias mapping. Returns true if the map changed.
    pub fn add_alias(&mut self, from: String, canonical: String) -> bool {
        if from == canonical || from.is_empty() || canonical.is_empty() {
            return false;
        }
        let key = from.to_lowercase();
        let changed = self.aliases.get(&key) != Some(&canonical);
        if changed {
            self.aliases.insert(key, canonical.clone());
        }
        if !self.nodes.contains(&canonical) {
            self.nodes.push(canonical);
            self.nodes.sort();
        }
        changed
    }

    /// Resolve a list of hostnames, deduplicating by canonical name.
    /// Preserves order (first occurrence wins).
    pub fn resolve_dedup(&self, hostnames: &[String]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for h in hostnames {
            let canonical = self.resolve(h).to_string();
            if seen.insert(canonical.to_lowercase()) {
                result.push(canonical);
            }
        }
        result
    }

    /// Whether the map has any entries.
    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty() && self.nodes.is_empty()
    }

    /// Add RDMA/Thunderbolt bridge IPs for a node. Returns true if changed.
    pub fn add_rdma_ips(&mut self, canonical: &str, ips: &[String]) -> bool {
        let entry = self.rdma_ips.entry(canonical.to_string()).or_default();
        let mut changed = false;
        for ip in ips {
            if !ip.is_empty() && !entry.contains(ip) {
                entry.push(ip.clone());
                changed = true;
            }
        }
        if changed {
            entry.sort();
        }
        changed
    }

    /// Add an RDMA link mapping. Returns true if the link is new.
    pub fn add_rdma_link(&mut self, link: RdmaLink) -> bool {
        let exists = self.rdma_links.iter().any(|l| {
            l.local_interface == link.local_interface && l.remote_ip == link.remote_ip
        });
        if !exists {
            self.rdma_links.push(link);
            self.rdma_links.sort_by(|a, b| a.local_interface.cmp(&b.local_interface));
            true
        } else {
            false
        }
    }

    /// Get RDMA links for a specific remote hostname.
    pub fn rdma_links_to(&self, hostname: &str) -> Vec<&RdmaLink> {
        self.rdma_links
            .iter()
            .filter(|l| l.remote_hostname == hostname)
            .collect()
    }

    /// Generate an mlx.launch-compatible hostfile JSON for RDMA/distributed inference.
    /// Uses the first Thunderbolt bridge IP for each node that has RDMA IPs.
    ///
    /// Output format:
    /// ```json
    /// [
    ///   {"hostname": "169.254.118.6", "port": 0},
    ///   {"hostname": "169.254.19.163", "port": 0}
    /// ]
    /// ```
    pub fn hostfile_json(&self, port: u16) -> String {
        let entries: Vec<serde_json::Value> = self
            .nodes
            .iter()
            .filter_map(|node| {
                self.rdma_ips
                    .get(node)
                    .and_then(|ips| ips.first())
                    .map(|ip| {
                        serde_json::json!({
                            "hostname": ip,
                            "port": port,
                        })
                    })
            })
            .collect();
        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Generate a JACCL hostfile with RDMA device matrix.
    ///
    /// Format: `[{"ssh": "m3u1", "ips": [], "rdma": [null, "rdma_en3"]}, ...]`
    /// where `rdma[i]` = the RDMA device on this host that connects to host `i`,
    /// and `null` = self.
    ///
    /// Only includes nodes that have RDMA link topology data. The `local_hostname`
    /// parameter identifies which node is running this code (its links are the
    /// ones stored in `rdma_links`).
    pub fn hostfile_jaccl(&self, local_hostname: &str) -> String {
        // Collect nodes that have RDMA connectivity
        let rdma_nodes: Vec<&str> = self
            .nodes
            .iter()
            .filter(|n| {
                // Include the local node (we have its links) and any node
                // that appears as a remote_hostname in our links
                n.as_str() == local_hostname
                    || self.rdma_links.iter().any(|l| l.remote_hostname == **n)
            })
            .map(|s| s.as_str())
            .collect();

        if rdma_nodes.is_empty() {
            return "[]".to_string();
        }

        let entries: Vec<serde_json::Value> = rdma_nodes
            .iter()
            .map(|&node| {
                // Build the NxN RDMA device matrix row for this node
                let rdma_row: Vec<serde_json::Value> = rdma_nodes
                    .iter()
                    .map(|&other| {
                        if node == other {
                            // Self → null
                            serde_json::Value::Null
                        } else if node == local_hostname {
                            // Local node: look up which device connects to `other`
                            self.rdma_links
                                .iter()
                                .find(|l| l.remote_hostname == other)
                                .and_then(|l| l.rdma_device.as_deref())
                                .map(|d| serde_json::Value::String(d.to_string()))
                                .unwrap_or(serde_json::Value::Null)
                        } else if other == local_hostname {
                            // Remote node connecting back to us: the remote node's
                            // device is the interface we see them on (symmetric).
                            // e.g., if we see m3u1 on en3, m3u1 sees us on their en*
                            // We can only know our side, so use the device name from
                            // our link to that node (the remote likely has rdma_enX too)
                            self.rdma_links
                                .iter()
                                .find(|l| l.remote_hostname == node)
                                .and_then(|l| l.rdma_device.as_deref())
                                .map(|d| serde_json::Value::String(d.to_string()))
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            // Two remote nodes connecting to each other — we don't
                            // have this info from our perspective
                            serde_json::Value::Null
                        }
                    })
                    .collect();

                let ips: Vec<&str> = self
                    .rdma_ips
                    .get(&node.to_string())
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();

                serde_json::json!({
                    "ssh": node,
                    "ips": ips,
                    "rdma": rdma_row,
                })
            })
            .collect();

        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get all RDMA-capable hosts as a flat list of (canonical_name, ip) pairs.
    pub fn rdma_hosts(&self) -> Vec<(&str, &str)> {
        let mut hosts = Vec::new();
        for node in &self.nodes {
            if let Some(ips) = self.rdma_ips.get(node) {
                for ip in ips {
                    hosts.push((node.as_str(), ip.as_str()));
                }
            }
        }
        hosts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PortState;

    #[test]
    fn test_register_node() {
        let mut nm = NodeMap::default();
        assert!(nm.register_node("m3u1"));
        assert!(!nm.register_node("m3u1")); // duplicate
        assert!(nm.register_node("m3u2"));
        assert_eq!(nm.nodes, vec!["m3u1", "m3u2"]);
    }

    #[test]
    fn test_register_node_canonicalizes_mdns_variants() {
        // Discovery may surface mDNS conflict-renamed names (`-2`, `-3`,
        // `-2237`). register_node must canonicalize them so phantoms never
        // reach self.nodes regardless of arrival order vs. the canonical name.
        let mut nm = NodeMap::default();

        // Phantom-first: hub-3 arrives before hub. Should land as "hub".
        assert!(nm.register_node("hub-3"));
        assert_eq!(nm.nodes, vec!["hub"]);

        // Canonical-then-phantom: real hub then mini2 then conflict-renames.
        assert!(!nm.register_node("hub")); // already there as canonical
        assert!(nm.register_node("mini2"));
        assert!(!nm.register_node("mini2-6")); // canonicalizes to existing mini2

        // 4-digit Bonjour instance suffix (`m3u4-2237`) is the worst real case.
        assert!(nm.register_node("m3u4-2237"));
        assert_eq!(
            nm.nodes,
            vec!["hub", "m3u4", "mini2"],
            "all phantom suffixes must collapse to canonical"
        );

        // Case-fold: Hub-2 should still hit existing hub.
        assert!(!nm.register_node("Hub-2"));
        assert_eq!(nm.nodes, vec!["hub", "m3u4", "mini2"]);
    }

    #[test]
    fn test_add_rdma_link() {
        let mut nm = NodeMap::default();
        let link = RdmaLink {
            local_interface: "en3".to_string(),
            local_ip: "169.254.19.163".to_string(),
            remote_ip: "169.254.204.162".to_string(),
            remote_hostname: "m3u3".to_string(),
            rdma_device: Some("rdma_en3".to_string()),
            port_state: Some(PortState::Active),
        };
        assert!(nm.add_rdma_link(link.clone()));
        assert!(!nm.add_rdma_link(link)); // duplicate
        assert_eq!(nm.rdma_links.len(), 1);
    }

    #[test]
    fn test_hostfile_jaccl_2node() {
        let mut nm = NodeMap::default();
        nm.nodes = vec!["m3u1".to_string(), "m3u2".to_string()];
        nm.rdma_links = vec![RdmaLink {
            local_interface: "en3".to_string(),
            local_ip: "169.254.19.163".to_string(),
            remote_ip: "169.254.118.6".to_string(),
            remote_hostname: "m3u1".to_string(),
            rdma_device: Some("rdma_en3".to_string()),
            port_state: Some(PortState::Active),
        }];

        let json = nm.hostfile_jaccl("m3u2");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();

        assert_eq!(arr.len(), 2, "should have 2 hosts");

        // m3u1 entry
        let m3u1 = &arr[0];
        assert_eq!(m3u1["ssh"], "m3u1");
        let m3u1_rdma = m3u1["rdma"].as_array().unwrap();
        assert!(m3u1_rdma[0].is_null(), "m3u1→m3u1 should be null (self)");
        // m3u1→m3u2: the device connecting back to us
        assert_eq!(m3u1_rdma[1], "rdma_en3");

        // m3u2 entry (local node)
        let m3u2 = &arr[1];
        assert_eq!(m3u2["ssh"], "m3u2");
        let m3u2_rdma = m3u2["rdma"].as_array().unwrap();
        // m3u2→m3u1: our device to m3u1
        assert_eq!(m3u2_rdma[0], "rdma_en3");
        assert!(m3u2_rdma[1].is_null(), "m3u2→m3u2 should be null (self)");
    }

    #[test]
    fn test_hostfile_jaccl_empty() {
        let nm = NodeMap::default();
        assert_eq!(nm.hostfile_jaccl("m3u2"), "[]");
    }

    #[test]
    fn test_config_path_ends_with_asmi() {
        let path = NodeMap::config_path();
        assert!(path.ends_with("asmi/config.json"),
            "config path should end with asmi/config.json, got: {}", path.display());
    }
}

/// Serde support for Duration as human-readable strings (e.g. "5s", "30s").
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
