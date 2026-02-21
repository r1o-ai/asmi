//! Cluster configuration — zero hardcoded nodes.
//!
//! All node discovery is dynamic: Thunderbolt bridge scanning, system-profiler,
//! mDNS/Bonjour, Tailscale status, or ARP table inspection.

use serde::{Deserialize, Serialize};
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
    /// Parse ARP table (`arp -an`) for link-local neighbours.
    Arp,
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
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            discovery: vec![
                DiscoveryMethod::ThunderboltBridge,
                DiscoveryMethod::Tailscale,
                DiscoveryMethod::Arp,
            ],
            seed_hosts: Vec::new(),
            ssh_timeout: Duration::from_secs(5),
            poll_interval: Duration::from_secs(2),
            scan_interval: Duration::from_secs(30),
            ssh_user: None,
            ssh_identity: None,
            history_capacity: 300,
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
