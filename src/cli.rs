use anyhow::Result;
use asmi_core::{
    ClusterConfig, ClusterEvent, ClusterMonitor, ClusterState, NodeMap, RdmaLink,
};
use std::io::{stdout, IsTerminal};
use std::sync::Arc;
use std::time::Duration;

use crate::{bin_name, DaemonAction, Format, Scan};

// ---------------------------------------------------------------------------
// One-shot + watch modes
// ---------------------------------------------------------------------------

/// Run the CLI monitor: one-shot or streaming watch mode.
pub async fn run_monitor(
    hosts: Vec<String>,
    interval: u64,
    format: Format,
    watch: bool,
    scan: Vec<Scan>,
) -> Result<()> {
    // Init tracing to file (not stdout — would corrupt table output in watch mode)
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("asmi");
    std::fs::create_dir_all(&log_dir)?;
    let log_file = std::fs::File::create(log_dir.join("asmi.log"))?;
    tracing_subscriber::fmt()
        .with_env_filter("asmi_core=info")
        .with_writer(std::sync::Mutex::new(log_file))
        .with_ansi(false)
        .init();

    // Load persistent NodeMap (aliases + known nodes)
    let node_map = NodeMap::load();

    // Resolve seed hosts: CLI --hosts > NodeMap known nodes > discovery
    let seeds = if !hosts.is_empty() {
        hosts
    } else {
        node_map.nodes.clone()
    };

    // Start cluster monitor
    let mut config = ClusterConfig::default()
        .with_seeds(seeds)
        .with_poll_interval(Duration::from_secs(interval));

    if !scan.is_empty() {
        config = config.with_discovery(scan.iter().map(Into::into).collect());
    }

    let mut monitor = ClusterMonitor::new(config.clone(), node_map);
    let state = monitor.state();
    let node_map = monitor.node_map();

    // Subscribe to events BEFORE starting — otherwise early events are lost
    let events_rx_bg = monitor.events();

    monitor.start();

    // Background task: handle cluster events → update NodeMap → save
    spawn_node_map_updater(Arc::clone(&node_map), events_rx_bg);

    if !watch {
        // One-shot: wait for first scan + metrics, print, exit
        let mut rx = monitor.subscribe();
        for _ in 0..2 {
            let _ = tokio::time::timeout(Duration::from_secs(10), rx.changed()).await;
        }

        // Persist discovered nodes
        {
            let s = state.read().await;
            let mut nm = node_map.write().await;
            for result in &s.scan_results {
                if result.ssh_ok {
                    nm.register_node(&result.hostname);
                }
            }
            if !nm.nodes.is_empty() {
                nm.save();
            }
        }

        let s = state.read().await;
        match format {
            Format::Json => print_json(&s),
            Format::Table => print_table(&s),
        }
        monitor.stop();
        return Ok(());
    }

    // Streaming watch mode: --watch → continuous stdout
    let mut rx = monitor.subscribe();
    for _ in 0..2 {
        let _ = tokio::time::timeout(Duration::from_secs(10), rx.changed()).await;
    }

    let is_tty = stdout().is_terminal();
    loop {
        let s = state.read().await;
        if is_tty {
            print!("\x1b[2J\x1b[H");
        }
        match format {
            Format::Json => print_json(&s),
            Format::Table => print_table(&s),
        }
        drop(s);

        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(interval)) => {}
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    monitor.stop();

    Ok(())
}

/// Background task: handle cluster events and persist NodeMap changes.
fn spawn_node_map_updater(
    node_map: Arc<tokio::sync::RwLock<NodeMap>>,
    mut events_rx: tokio::sync::broadcast::Receiver<ClusterEvent>,
) {
    tokio::spawn(async move {
        loop {
            match events_rx.recv().await {
                Ok(ClusterEvent::NodeProbed { hostname, online, .. }) => {
                    if online {
                        let mut nm = node_map.write().await;
                        if nm.register_node(&hostname) {
                            nm.save();
                            tracing::info!(
                                node = hostname.as_str(),
                                nodes = nm.nodes.len(),
                                "node registered and saved"
                            );
                        }
                    }
                }
                Ok(ClusterEvent::AliasDiscovered { alias, canonical }) => {
                    let mut nm = node_map.write().await;
                    if nm.add_alias(alias, canonical) {
                        nm.save();
                        tracing::info!(
                            aliases = nm.aliases.len(),
                            nodes = nm.nodes.len(),
                            "node map updated and saved"
                        );
                    }
                }
                Ok(ClusterEvent::RdmaIpsDiscovered { canonical, ips, .. }) => {
                    let mut nm = node_map.write().await;
                    if nm.add_rdma_ips(&canonical, &ips) {
                        nm.save();
                        tracing::info!(
                            node = canonical.as_str(),
                            ips = ?ips,
                            "RDMA IPs discovered and saved"
                        );
                    }
                }
                Ok(ClusterEvent::RdmaLinkDiscovered {
                    local_interface,
                    local_ip,
                    remote_ip,
                    remote_hostname,
                    rdma_device,
                    port_state,
                }) => {
                    let mut nm = node_map.write().await;
                    let link = RdmaLink {
                        local_interface,
                        local_ip,
                        remote_ip,
                        remote_hostname: remote_hostname.clone(),
                        rdma_device,
                        port_state,
                    };
                    if nm.add_rdma_link(link) {
                        nm.save();
                        tracing::info!(
                            remote = remote_hostname.as_str(),
                            links = nm.rdma_links.len(),
                            "RDMA link discovered and saved"
                        );
                    }
                }
                Ok(ClusterEvent::RdmaDeviceCorrelated {
                    interface,
                    rdma_device,
                    port_state,
                }) => {
                    let mut nm = node_map.write().await;
                    let mut changed = false;
                    for link in &mut nm.rdma_links {
                        if link.local_interface == interface &&
                            (link.rdma_device.as_deref() != Some(&rdma_device)
                                || link.port_state != Some(port_state))
                        {
                            link.rdma_device = Some(rdma_device.clone());
                            link.port_state = Some(port_state);
                            changed = true;
                        }
                    }
                    if changed {
                        nm.save();
                        tracing::info!(
                            device = rdma_device.as_str(),
                            state = %port_state,
                            "RDMA device state correlated"
                        );
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                _ => {}
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

/// Format the distributed backend for display.
pub fn format_backend(backend: Option<&asmi_core::DistributedBackend>) -> String {
    match backend {
        Some(d) => format!("[{d}]"),
        None => "[local]".to_string(),
    }
}

/// Print a one-shot table (like nvidia-smi default output).
pub fn print_table(state: &ClusterState) {
    let agg = &state.aggregates;
    println!("+{:-<92}+", "");
    println!("| {:<90} |", format!(
        "{}   {}  nodes: {}/{}  power: {:.1}W  RAM: {:.0}/{:.0}GB",
        bin_name(),
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        agg.nodes_online,
        agg.nodes_total,
        agg.total_watts,
        agg.total_ram_used_gib(),
        agg.total_ram_total_gib(),
    ));
    println!("+{:-<92}+", "");
    println!("| {:<10} {:<10} {:<6} {:<6} {:<12} {:<7} {:<8} {:<6} {:<17} |",
        "Node", "TB", "CPU%", "GPU%", "RAM", "Cache", "Power", "RDMA", "Model");
    println!("|{:-<92}|", "");

    for name in state.sorted_hostnames() {
        if let Some(snap) = state.snapshots.get(&name) {
            let proc_desc = if snap.processes.is_empty() {
                "--".to_string()
            } else {
                snap.processes.iter().map(|p| {
                    let model_name = if let Some(m) = p.server_models.first() {
                        m.id.rsplit('/').next().unwrap_or(&m.id).to_string()
                    } else if let Some(m) = p.model.as_deref() {
                        m.rsplit('/').next().unwrap_or(m).to_string()
                    } else {
                        "no model".to_string()
                    };
                    let port_str = p.port.map(|port| format!(":{port}")).unwrap_or_default();
                    let backend_str = format_backend(p.distributed.as_ref());
                    format!("{model_name}{port_str} {backend_str}")
                }).collect::<Vec<_>>().join(", ")
            };
            let scan = state.scan_results.iter().find(|r| r.hostname == name);
            let tb_speed = scan
                .and_then(|r| r.link_speed.as_deref())
                .unwrap_or("--");
            let rdma_info = scan
                .and_then(|r| r.rdma.as_ref())
                .map(|r| {
                    let a = r.active_count();
                    let t = r.devices.len();
                    format!("{a}/{t}")
                })
                .unwrap_or_else(|| "--".to_string());
            let cache_g = format!("{:.0}G", snap.ram_cached_gib());
            println!("| {:<10} {:<10} {:>4.0}% {:>4.0}% {:>4.0}/{:<4.0}GB {:<7} {:>5.1}W  {:<6} {:<17} |",
                name,
                tb_speed,
                snap.cpu_percent,
                snap.gpu_percent,
                snap.ram_app_gib(),
                snap.ram_total_gib(),
                cache_g,
                snap.total_watts(),
                rdma_info,
                proc_desc,
            );
        }
    }
    println!("+{:-<92}+", "");
}

/// Print JSON output.
pub fn print_json(state: &ClusterState) {
    let nodes: Vec<serde_json::Value> = state
        .snapshots
        .values()
        .map(|snap| serde_json::to_value(snap).unwrap_or_default())
        .collect();

    let output = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "aggregates": serde_json::to_value(&state.aggregates).unwrap_or_default(),
        "nodes": nodes,
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

// ---------------------------------------------------------------------------
// Daemon management subcommands (merged from daemon_mgmt.rs)
// ---------------------------------------------------------------------------

const DAEMON_PLIST: &str = "com.asmi.daemon";

/// Manage asmi daemons across cluster nodes.
pub async fn run_daemon(action: DaemonAction, port: u16) -> Result<()> {
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
                // Atomic replace: scp to .new, then ssh mv -f.
                // mv on the same filesystem is a rename(2): if the daemon
                // is running, it keeps its mmap on the old inode until exit;
                // direct overwrite would corrupt the running image (SIGKILL).
                let ok1a = std::process::Command::new("scp")
                    .args(["-o", "ConnectTimeout=5",
                        bin.to_str().unwrap_or(""),
                        &format!("{}:~/.cargo/bin/asmi.new", name)])
                    .status().map(|s| s.success()).unwrap_or(false);
                let ok1b = ok1a && std::process::Command::new("ssh")
                    .args(["-o", "ConnectTimeout=5", name,
                        "mv -f ~/.cargo/bin/asmi.new ~/.cargo/bin/asmi && chmod 755 ~/.cargo/bin/asmi"])
                    .status().map(|s| s.success()).unwrap_or(false);
                let ok1 = ok1a && ok1b;
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
            let cmd = "tail -50 ~/Library/Logs/asmi-daemon.log";
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
