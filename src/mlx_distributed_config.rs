//! Thin wrapper around Apple's `mlx.distributed_config` binary.
//!
//! This is the `--auto-setup` companion to `topology.rs` (which wraps `--dot`).
//! Apple's tool handles cross-node IP assignment, cable-pair symmetry, and
//! hostfile generation correctly — replacing the deterministic-index fallback
//! in `rdma_autosetup.rs` that collides across nodes (r1o-ai/asmi#1).

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Hostfile {
    pub backend: String,
    #[serde(default)]
    pub envs: Vec<String>,
    pub hosts: Vec<HostEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HostEntry {
    pub ssh: String,
    #[serde(default)]
    pub ips: Vec<String>,
    #[serde(default)]
    pub rdma: Vec<Option<String>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("mlx.distributed_config binary not found in PATH")]
    BinaryNotFound,
    #[error("mlx.distributed_config exited with status {0}: {1}")]
    NonZeroExit(i32, String),
    #[error("invalid JSON output: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Run `mlx.distributed_config --auto-setup` and parse the resulting hostfile.
///
/// The tool assigns link-local IPs to cable-paired Thunderbolt interfaces
/// across nodes such that no two nodes share an IP — fixing the deterministic
/// `169.254.{iface_index}.1` collision in our legacy fallback.
pub async fn auto_setup(
    hosts: &[String],
    backend: &str,
) -> Result<Hostfile, ConfigError> {
    let tmp = tempfile::NamedTempFile::new()?;
    let tmp_path = tmp.path().to_path_buf();

    let output = tokio::process::Command::new("mlx.distributed_config")
        .args([
            "--verbose",
            "--hosts",
            &hosts.join(","),
            "--over",
            "thunderbolt",
            "--backend",
            backend,
            "--auto-setup",
            "--output-hostfile",
            tmp_path.to_str().unwrap(),
        ])
        .output()
        .await
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::BinaryNotFound
            } else {
                ConfigError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(ConfigError::NonZeroExit(
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let body = tokio::fs::read_to_string(&tmp_path).await?;
    let hostfile: Hostfile = serde_json::from_str(&body)?;
    Ok(hostfile)
}

/// Run `mlx.distributed_config --dot` and return the raw GraphViz output.
pub async fn dot_topology(hosts: &[String]) -> Result<String, ConfigError> {
    let output = tokio::process::Command::new("mlx.distributed_config")
        .args([
            "--hosts",
            &hosts.join(","),
            "--over",
            "thunderbolt",
            "--dot",
        ])
        .output()
        .await
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::BinaryNotFound
            } else {
                ConfigError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(ConfigError::NonZeroExit(
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn auto_setup_against_live_cluster() {
        // Live test only — requires hub + m3u1 + m3u3 on TB5 mesh.
        if std::env::var("ASMI_LIVE_CLUSTER_TEST").is_err() {
            eprintln!("skipping — set ASMI_LIVE_CLUSTER_TEST=1 to run");
            return;
        }
        let result = auto_setup(
            &["hub".into(), "m3u1".into(), "m3u3".into()],
            "jaccl-ring",
        )
        .await
        .expect("auto_setup should succeed");
        assert_eq!(result.hosts.len(), 3, "should produce 3 hosts");
        assert!(
            result.hosts.iter().all(|h| !h.ips.is_empty()),
            "every host must have at least one IP"
        );

        // No two hosts share an IP — the bug we're fixing
        let mut all_ips: Vec<&str> = result
            .hosts
            .iter()
            .flat_map(|h| h.ips.iter().map(String::as_str))
            .collect();
        all_ips.sort();
        let before = all_ips.len();
        all_ips.dedup();
        assert_eq!(before, all_ips.len(), "duplicate IPs across hosts");
    }
}
