//! Node health/setup validation checks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resolve the best python3 binary (homebrew > system).
fn resolve_python() -> &'static str {
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ];
    for p in CANDIDATES {
        if std::path::Path::new(p).exists() {
            return p;
        }
    }
    "python3"
}

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

/// Result of Thunderbolt network service validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThunderboltServiceStatus {
    /// All Thunderbolt network services found.
    pub thunderbolt_services: Vec<String>,
    /// Issues detected (duplicates, non-r1o-prefixed).
    pub issues: Vec<String>,
    /// True if no issues found.
    pub clean: bool,
}

/// Actions taken by `fix_thunderbolt_services`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThunderboltFixResult {
    pub actions: Vec<String>,
    pub success: bool,
    pub error: Option<String>,
}

/// Run all setup validation checks.
pub async fn run_setup_checks() -> SetupChecks {
    let mut checks = Vec::new();

    checks.push(check_python_mlx().await);
    checks.push(check_rdma().await);
    checks.push(check_disk_space().await);
    checks.push(check_ssh_keys().await);
    checks.push(check_zshenv().await);
    checks.push(check_thunderbolt_services().await);

    let all_pass = checks.iter().all(|c| c.pass);
    SetupChecks { checks, all_pass }
}

/// Check Thunderbolt network service names for duplicates and non-r1o prefixes.
///
/// macOS `networksetup -listallnetworkservices` returns lines like:
/// ```text
/// An asterisk (*) denotes that a network service is disabled.
/// Ethernet
/// r1o Thunderbolt 1
/// r1o Thunderbolt 2
/// Wi-Fi
/// ```
///
/// We flag:
/// - Non-r1o-prefixed Thunderbolt services (e.g. "EXO Thunderbolt 1")
/// - Duplicate ports: multiple services for the same TB port number
pub async fn check_thunderbolt_services() -> CheckResult {
    let status = validate_thunderbolt_services().await;
    CheckResult {
        id: "thunderbolt-services".into(),
        pass: status.clean,
        detail: if status.clean {
            let count = status.thunderbolt_services.len();
            Some(format!("{count} services, all clean"))
        } else {
            Some(status.issues.join("; "))
        },
    }
}

/// Parse and validate Thunderbolt network services. Used by both the health
/// check and the `/health/network` daemon endpoint.
pub async fn validate_thunderbolt_services() -> ThunderboltServiceStatus {
    let output = tokio::process::Command::new("networksetup")
        .args(["-listallnetworkservices"])
        .output()
        .await;

    let raw = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => {
            return ThunderboltServiceStatus {
                thunderbolt_services: vec![],
                issues: vec!["networksetup command failed".into()],
                clean: false,
            };
        }
    };

    let tb_services: Vec<String> = raw
        .lines()
        .map(|l| l.trim().trim_start_matches('*').trim().to_string())
        .filter(|l| l.to_lowercase().contains("thunderbolt"))
        .collect();

    let issues = analyze_thunderbolt_services(&tb_services);

    ThunderboltServiceStatus {
        clean: issues.is_empty(),
        thunderbolt_services: tb_services,
        issues,
    }
}

/// Analyze a list of Thunderbolt service names for issues.
fn analyze_thunderbolt_services(services: &[String]) -> Vec<String> {
    let mut issues = Vec::new();

    // Group services by their Thunderbolt port number (the trailing digit).
    // e.g. "r1o Thunderbolt 1" and "EXO Thunderbolt 1" both map to port "1".
    let mut port_map: HashMap<String, Vec<&String>> = HashMap::new();
    for svc in services {
        let port = extract_tb_port_number(svc);
        port_map.entry(port).or_default().push(svc);
    }

    // Check for duplicates per port
    for (port, svcs) in &port_map {
        if svcs.len() > 1 {
            let names: Vec<&str> = svcs.iter().map(|s| s.as_str()).collect();
            issues.push(format!(
                "Thunderbolt {} has duplicates: {}",
                port,
                names.join(", ")
            ));
        }
    }

    // Check for non-r1o-prefixed services
    for svc in services {
        if !svc.starts_with("r1o ") {
            issues.push(format!("{svc} (not r1o-prefixed)"));
        }
    }

    issues
}

/// Extract the port number from a service name like "r1o Thunderbolt 1" → "1".
/// Falls back to the full name if no number is found.
fn extract_tb_port_number(name: &str) -> String {
    // Look for the last word that is a digit
    name.split_whitespace()
        .rev()
        .find(|w| w.chars().all(|c| c.is_ascii_digit()))
        .unwrap_or(name)
        .to_string()
}

/// Fix Thunderbolt network service names:
/// - Rename lone non-r1o services to `r1o Thunderbolt N`
/// - Remove duplicate services where an r1o version already exists
pub async fn fix_thunderbolt_services() -> ThunderboltFixResult {
    let status = validate_thunderbolt_services().await;
    if status.clean {
        return ThunderboltFixResult {
            actions: vec!["No issues found, nothing to fix".into()],
            success: true,
            error: None,
        };
    }

    let mut actions = Vec::new();

    // Group by port number
    let mut port_map: HashMap<String, Vec<String>> = HashMap::new();
    for svc in &status.thunderbolt_services {
        let port = extract_tb_port_number(svc);
        port_map.entry(port).or_default().push(svc.clone());
    }

    for (port, svcs) in &port_map {
        let r1o_name = format!("r1o Thunderbolt {port}");
        let has_r1o = svcs.iter().any(|s| s == &r1o_name);

        if has_r1o {
            // Remove duplicates (non-r1o versions for this port)
            for svc in svcs {
                if svc != &r1o_name {
                    let result = tokio::process::Command::new("networksetup")
                        .args(["-removenetworkservice", svc])
                        .output()
                        .await;
                    match result {
                        Ok(o) if o.status.success() => {
                            actions.push(format!("Removed duplicate: {svc}"));
                        }
                        Ok(o) => {
                            let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                            actions.push(format!("Failed to remove {svc}: {err}"));
                        }
                        Err(e) => {
                            actions.push(format!("Failed to remove {svc}: {e}"));
                        }
                    }
                }
            }
        } else if svcs.len() == 1 {
            // Lone non-r1o service → rename it
            let svc = &svcs[0];
            let result = tokio::process::Command::new("networksetup")
                .args(["-renamenetworkservice", svc, &r1o_name])
                .output()
                .await;
            match result {
                Ok(o) if o.status.success() => {
                    actions.push(format!("Renamed: {svc} -> {r1o_name}"));
                }
                Ok(o) => {
                    let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    actions.push(format!("Failed to rename {svc}: {err}"));
                }
                Err(e) => {
                    actions.push(format!("Failed to rename {svc}: {e}"));
                }
            }
        } else {
            // Multiple non-r1o services — rename the first, remove the rest
            let first = &svcs[0];
            let result = tokio::process::Command::new("networksetup")
                .args(["-renamenetworkservice", first, &r1o_name])
                .output()
                .await;
            match result {
                Ok(o) if o.status.success() => {
                    actions.push(format!("Renamed: {first} -> {r1o_name}"));
                }
                Ok(o) => {
                    let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    actions.push(format!("Failed to rename {first}: {err}"));
                }
                Err(e) => {
                    actions.push(format!("Failed to rename {first}: {e}"));
                }
            }
            for svc in &svcs[1..] {
                let result = tokio::process::Command::new("networksetup")
                    .args(["-removenetworkservice", svc])
                    .output()
                    .await;
                match result {
                    Ok(o) if o.status.success() => {
                        actions.push(format!("Removed duplicate: {svc}"));
                    }
                    Ok(o) => {
                        let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                        actions.push(format!("Failed to remove {svc}: {err}"));
                    }
                    Err(e) => {
                        actions.push(format!("Failed to remove {svc}: {e}"));
                    }
                }
            }
        }
    }

    let success = actions.iter().all(|a| !a.starts_with("Failed"));
    ThunderboltFixResult {
        actions,
        success,
        error: None,
    }
}

/// Parse Thunderbolt service names from raw `networksetup -listallnetworkservices`
/// output (or SSH `grep -i thunder` filtered output). Used by scanner for remote checks.
pub fn parse_thunderbolt_services(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.trim().trim_start_matches('*').trim().to_string())
        .filter(|l| l.to_lowercase().contains("thunderbolt"))
        .filter(|l| !l.is_empty())
        .collect()
}

/// Analyze parsed service names and return issue descriptions (empty = clean).
pub fn find_thunderbolt_issues(services: &[String]) -> Vec<String> {
    analyze_thunderbolt_services(services)
}

async fn check_python_mlx() -> CheckResult {
    let output = tokio::process::Command::new(resolve_python())
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

    #[test]
    fn test_analyze_clean_services() {
        let services = vec![
            "r1o Thunderbolt 1".to_string(),
            "r1o Thunderbolt 2".to_string(),
        ];
        let issues = analyze_thunderbolt_services(&services);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_analyze_non_r1o_prefix() {
        let services = vec![
            "EXO Thunderbolt 1".to_string(),
            "r1o Thunderbolt 2".to_string(),
        ];
        let issues = analyze_thunderbolt_services(&services);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("not r1o-prefixed"));
    }

    #[test]
    fn test_analyze_duplicates() {
        let services = vec![
            "r1o Thunderbolt 1".to_string(),
            "EXO Thunderbolt 1".to_string(),
        ];
        let issues = analyze_thunderbolt_services(&services);
        // Should have both a duplicate warning and a non-r1o warning
        assert!(issues.iter().any(|i| i.contains("duplicates")));
        assert!(issues.iter().any(|i| i.contains("not r1o-prefixed")));
    }

    #[test]
    fn test_extract_tb_port_number() {
        assert_eq!(extract_tb_port_number("r1o Thunderbolt 1"), "1");
        assert_eq!(extract_tb_port_number("EXO Thunderbolt 2"), "2");
        assert_eq!(extract_tb_port_number("Thunderbolt Bridge"), "Thunderbolt Bridge");
    }

    #[test]
    fn test_parse_thunderbolt_services() {
        let output = "An asterisk (*) denotes...\nEthernet\nr1o Thunderbolt 1\nEXO Thunderbolt 2\nWi-Fi\n";
        let services = parse_thunderbolt_services(output);
        assert_eq!(services, vec!["r1o Thunderbolt 1", "EXO Thunderbolt 2"]);
    }

    #[test]
    fn test_parse_thunderbolt_services_disabled() {
        let output = "*r1o Thunderbolt 1\nr1o Thunderbolt 2\n";
        let services = parse_thunderbolt_services(output);
        assert_eq!(services.len(), 2);
        assert_eq!(services[0], "r1o Thunderbolt 1"); // asterisk stripped
    }
}
