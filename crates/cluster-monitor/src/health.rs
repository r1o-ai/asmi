//! Node health/setup validation checks.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub id: String,
    pub pass: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupChecks {
    pub checks: Vec<CheckResult>,
    pub all_pass: bool,
}

/// Run all setup validation checks.
pub async fn run_setup_checks() -> SetupChecks {
    let mut checks = Vec::new();

    checks.push(check_python_mlx().await);
    checks.push(check_rdma().await);
    checks.push(check_disk_space().await);
    checks.push(check_ssh_keys().await);
    checks.push(check_zshenv().await);

    let all_pass = checks.iter().all(|c| c.pass);
    SetupChecks { checks, all_pass }
}

async fn check_python_mlx() -> CheckResult {
    let output = tokio::process::Command::new("python3")
        .args(["-c", "import mlx.core as mx; print(f'MLX {mx.__version__}')"])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => CheckResult {
            id: "python-mlx".into(),
            pass: true,
            detail: Some(String::from_utf8_lossy(&o.stdout).trim().to_string()),
        },
        Ok(o) => CheckResult {
            id: "python-mlx".into(),
            pass: false,
            detail: Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        },
        Err(e) => CheckResult {
            id: "python-mlx".into(),
            pass: false,
            detail: Some(format!("python3 not found: {e}")),
        },
    }
}

async fn check_rdma() -> CheckResult {
    let output = tokio::process::Command::new("sh")
        .args(["-c", "ibv_devices 2>/dev/null | grep -c rdma"])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => {
            let count: i32 = String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse()
                .unwrap_or(0);
            CheckResult {
                id: "rdma".into(),
                pass: count > 0,
                detail: Some(format!("{count} devices")),
            }
        }
        _ => CheckResult {
            id: "rdma".into(),
            pass: false,
            detail: Some("ibv_devices not available".into()),
        },
    }
}

async fn check_disk_space() -> CheckResult {
    let output = tokio::process::Command::new("sh")
        .args(["-c", "df -g / | awk 'NR==2{print $4}'"])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => {
            let gb: u64 = String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse()
                .unwrap_or(0);
            CheckResult {
                id: "disk-space".into(),
                pass: gb >= 50,
                detail: Some(format!("{gb}GB free")),
            }
        }
        _ => CheckResult {
            id: "disk-space".into(),
            pass: false,
            detail: Some("could not check disk space".into()),
        },
    }
}

async fn check_ssh_keys() -> CheckResult {
    let home = dirs::home_dir().unwrap_or_default();
    let ed25519 = home.join(".ssh/id_ed25519");
    let rsa = home.join(".ssh/id_rsa");
    let pass = ed25519.exists() || rsa.exists();
    CheckResult {
        id: "ssh-keys".into(),
        pass,
        detail: if pass {
            Some("found".into())
        } else {
            Some("no SSH keys in ~/.ssh/".into())
        },
    }
}

async fn check_zshenv() -> CheckResult {
    let home = dirs::home_dir().unwrap_or_default();
    let zshenv = home.join(".zshenv");
    match std::fs::read_to_string(&zshenv) {
        Ok(content) => {
            let has_cargo = content.contains(".cargo/bin");
            CheckResult {
                id: "zshenv".into(),
                pass: has_cargo,
                detail: if has_cargo {
                    Some("configured".into())
                } else {
                    Some("missing .cargo/bin in PATH".into())
                },
            }
        }
        Err(_) => CheckResult {
            id: "zshenv".into(),
            pass: false,
            detail: Some("~/.zshenv not found".into()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_pass() {
        let c = CheckResult {
            id: "test".into(),
            pass: true,
            detail: Some("ok".into()),
        };
        assert!(c.pass);
    }

    #[test]
    fn test_check_result_fail() {
        let c = CheckResult {
            id: "test".into(),
            pass: false,
            detail: Some("missing".into()),
        };
        assert!(!c.pass);
    }
}
