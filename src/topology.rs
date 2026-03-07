//! Cluster topology discovery by wrapping `mlx.distributed_config`.
//!
//! Shells out to Apple's `mlx.distributed_config --dot` which SSHes to all nodes
//! and discovers the full TB5/RDMA mesh. Parses the DOT output into structured
//! JSON with mesh completeness validation.

use std::collections::{HashMap, HashSet};
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

/// Run `mlx.distributed_config --dot` and capture output.
pub fn discover_topology(hosts: &[String], backend: &str) -> Result<TopologyReport> {
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

    // mlx.distributed_config outputs DOT to stdout even on incomplete mesh
    // (the error goes to stderr). Parse whatever DOT we get.
    let dot = if stdout.contains("graph G") {
        stdout.clone()
    } else if stderr.contains("graph G") {
        // Sometimes DOT goes to stderr alongside error messages
        stderr
            .lines()
            .skip_while(|l| !l.contains("graph G"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        // No DOT at all — might be a complete failure
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
    let n = nodes.len();
    let expected_links = n * (n - 1) / 2;
    let jaccl_ready = mesh_complete && links.len() >= expected_links;

    // Find fully-meshed subsets of 3+ nodes
    let subsets = find_jaccl_subsets(&nodes, &connected);

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
