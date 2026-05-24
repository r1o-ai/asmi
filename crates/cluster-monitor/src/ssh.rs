//! SSH execution layer — runs commands on remote (and local) nodes.
//!
//! Mirrors the TypeScript `sshRun()` pattern from the web app:
//! - Uses `ssh` with `ConnectTimeout`, `StrictHostKeyChecking=no`
//! - Returns structured `SshResult` with stdout/stderr/success
//! - `local_run()` for commands on the local machine

use crate::config::ClusterConfig;
use crate::types::MonitorError;
use std::time::Duration;
use tokio::process::Command;
use tracing::{debug, warn};

/// Result of running a command (local or remote).
#[derive(Debug, Clone)]
pub struct SshResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

impl SshResult {
    /// Convenience: was the command successful and did it produce output?
    pub fn has_output(&self) -> bool {
        self.success && !self.stdout.is_empty()
    }
}

/// Run a command on a remote host via SSH.
///
/// Uses the config's SSH timeout, user, and identity settings. Flags match
/// the TypeScript sshRun() pattern:
/// - `ConnectTimeout=<N>`
/// - `StrictHostKeyChecking=no`
/// - `BatchMode=yes` (no interactive prompts)
/// - `LogLevel=ERROR` (suppress banners)
pub async fn ssh_run(
    host: &str,
    command: &str,
    config: &ClusterConfig,
) -> Result<SshResult, MonitorError> {
    let timeout_secs = config.ssh_timeout_secs();
    let user = config
        .ssh_user
        .clone()
        .unwrap_or_else(|| whoami::fallible::username().unwrap_or_else(|_| "root".to_string()));

    let mut cmd = Command::new("ssh");
    cmd.arg("-o")
        .arg(format!("ConnectTimeout={timeout_secs}"))
        .arg("-o")
        .arg("StrictHostKeyChecking=no")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("LogLevel=ERROR")
        .arg("-o")
        .arg("ControlMaster=auto")
        .arg("-o")
        .arg("ControlPath=/tmp/asmi-%r@%h:%p")
        .arg("-o")
        .arg("ControlPersist=5m");

    if let Some(ref identity) = config.ssh_identity {
        cmd.arg("-i").arg(identity);
    }

    cmd.arg(format!("{user}@{host}")).arg(command);

    debug!(host, command, "ssh_run");

    let result = tokio::time::timeout(
        Duration::from_secs(timeout_secs + 5), // grace period beyond SSH's own timeout
        cmd.output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let success = output.status.success();
            let exit_code = output.status.code();

            if !success {
                debug!(host, exit_code, stderr = stderr.as_str(), "ssh command failed");
            }

            Ok(SshResult {
                success,
                stdout,
                stderr,
                exit_code,
            })
        }
        Ok(Err(e)) => {
            warn!(host, error = %e, "ssh process IO error");
            Err(MonitorError::Io(e))
        }
        Err(_) => {
            warn!(host, timeout_secs, "ssh command timed out");
            Err(MonitorError::Timeout {
                host: host.to_string(),
                timeout_secs,
            })
        }
    }
}

/// Run a command on the local machine.
pub async fn local_run(command: &str) -> Result<SshResult, MonitorError> {
    debug!(command, "local_run");

    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();
    let exit_code = output.status.code();

    Ok(SshResult {
        success,
        stdout,
        stderr,
        exit_code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_run_hostname() {
        let result = local_run("hostname").await.expect("local_run should succeed");
        assert!(result.success, "hostname command should succeed");
        assert!(!result.stdout.trim().is_empty(), "hostname should produce output");
        assert_eq!(result.exit_code, Some(0));
    }

    #[tokio::test]
    async fn test_local_run_failure() {
        let result = local_run("false").await.expect("local_run should not error on non-zero exit");
        assert!(!result.success, "false command should fail");
        assert_eq!(result.exit_code, Some(1));
    }

    #[tokio::test]
    async fn test_ssh_run_localhost() {
        let config = ClusterConfig {
            ssh_timeout: Duration::from_secs(5),
            ssh_user: None,
            ssh_identity: None,
            ..ClusterConfig::default()
        };

        let result = ssh_run("localhost", "hostname", &config).await;

        // This test may fail in CI where SSH to localhost is not configured,
        // so we just verify the function does not panic and returns a valid result.
        match result {
            Ok(r) => {
                // If SSH works, we should get a hostname back
                if r.success {
                    assert!(!r.stdout.trim().is_empty(), "hostname should produce output");
                }
            }
            Err(MonitorError::Timeout { .. }) => {
                // Acceptable: SSH to localhost may not be configured
            }
            Err(MonitorError::Io(_)) => {
                // Acceptable: SSH binary might not be available
            }
            Err(e) => {
                // SshFailed is not returned by ssh_run directly, but just in case
                eprintln!("ssh_run to localhost returned error (may be expected): {e}");
            }
        }
    }
}
