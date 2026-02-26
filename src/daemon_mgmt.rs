use anyhow::Result;
use std::time::Duration;

use super::{bin_name, DaemonAction};

const DAEMON_PLIST: &str = "com.asmi.daemon";

/// Manage asmi daemons across cluster nodes.
pub(crate) async fn run_daemon(action: DaemonAction, port: u16) -> Result<()> {
    let local_hostname = std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let node_map = asmi_core::NodeMap::load();
    let known_nodes: Vec<String> = if node_map.nodes.is_empty() {
        eprintln!("No known nodes in NodeMap. Run `asmi` first to discover cluster nodes,");
        eprintln!("or add seed hosts: `asmi --hosts node1,node2,node3`");
        std::process::exit(1);
    } else {
        node_map.nodes.clone()
    };

    match action {
        DaemonAction::Status => {
            println!("{} daemon status (:{})", bin_name(), port);
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(1))
                .timeout(Duration::from_secs(2))
                .build()?;

            for name in &known_nodes {
                let name = name.as_str();
                // Try bare name first, fall back to <name>.local (mDNS) if it fails.
                let candidates: &[String] = &[
                    format!("http://{}:{}/health", name, port),
                    format!("http://{}.local:{}/health", name, port),
                ];
                let mut found = false;
                for url in candidates {
                    match client.get(url).send().await {
                        Ok(resp) if resp.status().is_success() => {
                            if let Ok(json) = resp.json::<serde_json::Value>().await {
                                let secs = json["uptime_secs"].as_u64().unwrap_or(0);
                                let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
                                let uptime = if h > 0 {
                                    format!("{}h{:02}m", h, m)
                                } else {
                                    format!("{}m{:02}s", m, s)
                                };
                                println!("  {:<6} \x1b[32m●\x1b[0m online  ({})", name, uptime);
                                found = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if !found {
                    println!("  {:<6} \x1b[31m●\x1b[0m offline", name);
                }
            }
        }
        DaemonAction::Start { node } => {
            let target = node.as_deref().unwrap_or("all");
            for name in &known_nodes {
                let name = name.as_str();
                if target != "all" && target != name { continue; }
                print!("  {}: ", name);
                let plist = format!("~/Library/LaunchAgents/{}.plist", DAEMON_PLIST);
                let cmd = format!("launchctl bootstrap gui/$(id -u) {}", plist);
                let ok = run_on_node(name, &local_hostname, &cmd);
                println!("{}", if ok { "started" } else { "already running (or error)" });
            }
        }
        DaemonAction::Stop { node } => {
            let target = node.as_deref().unwrap_or("all");
            for name in &known_nodes {
                let name = name.as_str();
                if target != "all" && target != name { continue; }
                print!("  {}: ", name);
                let plist = format!("~/Library/LaunchAgents/{}.plist", DAEMON_PLIST);
                let cmd = format!("launchctl bootout gui/$(id -u) {}", plist);
                let ok = run_on_node(name, &local_hostname, &cmd);
                println!("{}", if ok { "stopped" } else { "not running" });
            }
        }
        DaemonAction::Restart { node } => {
            let target = node.as_deref().unwrap_or("all");
            let plist = format!("~/Library/LaunchAgents/{}.plist", DAEMON_PLIST);
            for name in &known_nodes {
                let name = name.as_str();
                if target != "all" && target != name { continue; }
                print!("  {}: ", name);
                let stop_cmd = format!("launchctl bootout gui/$(id -u) {}", plist);
                run_on_node(name, &local_hostname, &stop_cmd);
                std::thread::sleep(std::time::Duration::from_millis(500));
                let start_cmd = format!("launchctl bootstrap gui/$(id -u) {}", plist);
                let ok = run_on_node(name, &local_hostname, &start_cmd);
                println!("{}", if ok { "restarted" } else { "failed" });
            }
        }
        DaemonAction::Deploy { node } => {
            let bin = std::env::current_exe()
                .ok()
                .and_then(|p| {
                    let dir = p.parent()?;
                    let release = dir.join("asmi");
                    if release.exists() { return Some(release); }
                    let up = dir.parent()?.join("release").join("asmi");
                    if up.exists() { return Some(up); }
                    None
                })
                .or_else(|| {
                    std::process::Command::new("which")
                        .arg("asmi")
                        .output()
                        .ok()
                        .filter(|o| o.status.success())
                        .and_then(|o| {
                            let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
                            let p = std::path::PathBuf::from(&path);
                            if p.exists() { Some(p) } else { None }
                        })
                })
                .unwrap_or_else(|| {
                    dirs::home_dir()
                        .unwrap_or_default()
                        .join(".cargo/bin/asmi")
                });
            let plist_path = dirs::home_dir()
                .unwrap_or_default()
                .join(format!("Library/LaunchAgents/{}.plist", DAEMON_PLIST));

            if !bin.exists() {
                eprintln!("Release binary not found at {}", bin.display());
                eprintln!("Build first: cargo build --release");
                std::process::exit(1);
            }

            let target = node.as_deref().unwrap_or("all");
            for name in &known_nodes {
                let name = name.as_str();
                if target != "all" && target != name { continue; }
                if name == local_hostname { continue; }
                print!("  {}: ", name);
                let ok1 = std::process::Command::new("scp")
                    .args(["-o", "ConnectTimeout=5",
                        bin.to_str().unwrap_or(""),
                        &format!("{}:~/.cargo/bin/asmi", name)])
                    .status().map(|s| s.success()).unwrap_or(false);
                let ok2 = std::process::Command::new("scp")
                    .args(["-o", "ConnectTimeout=5",
                        plist_path.to_str().unwrap_or(""),
                        &format!("{}:~/Library/LaunchAgents/", name)])
                    .status().map(|s| s.success()).unwrap_or(false);
                println!("{}", if ok1 && ok2 { "deployed" } else { "FAILED" });
            }
        }
        DaemonAction::Logs { node } => {
            let target = node.as_deref().unwrap_or(&local_hostname);
            let cmd = "tail -50 ~/Library/Application\\ Support/asmi/asmi.log";
            if target == local_hostname {
                let _ = std::process::Command::new("sh")
                    .args(["-c", cmd])
                    .status();
            } else {
                let _ = std::process::Command::new("ssh")
                    .args(["-o", "ConnectTimeout=3", target, cmd])
                    .status();
            }
        }
    }

    Ok(())
}

/// Run a shell command on a node (locally or via SSH).
fn run_on_node(node: &str, local_hostname: &str, cmd: &str) -> bool {
    if node == local_hostname {
        std::process::Command::new("sh")
            .args(["-c", cmd])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        std::process::Command::new("ssh")
            .args(["-o", "ConnectTimeout=5", node, cmd])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}
