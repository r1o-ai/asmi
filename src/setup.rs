//! First-time node setup: destroy bridge0, seed config, install daemon.
//!
//! `asmi setup` is the single command that configures a fresh Mac for r1o.
//! It handles everything that can be automated — bridge0 destruction, network
//! service creation, NodeMap seeding, launchd plist installation, and daemon
//! startup. The only thing it can't do is enable RDMA (requires Recovery OS).

use anyhow::{Context, Result};
use std::process::Command;

const PLIST_LABEL: &str = "com.asmi.daemon";
const PREFERENCES_PLIST: &str = "/Library/Preferences/SystemConfiguration/preferences.plist";

/// Run the full setup sequence.
pub async fn run_setup(port: u16, cluster: bool, skip_bridge0: bool, dry_run: bool) -> Result<()> {
    let home = dirs::home_dir().context("cannot determine home directory")?;
    let hostname = Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    println!("asmi setup — configuring {} for r1o", hostname);
    if dry_run {
        println!("  (dry run — no changes will be made)\n");
    } else {
        println!();
    }

    // ── Step 1: Detect bridge0 ──────────────────────────────────────────
    let bridge0_members = detect_bridge0();
    if bridge0_members.is_empty() {
        step("bridge0", "No Thunderbolt Bridge detected — skipping");
    } else if skip_bridge0 {
        step("bridge0", &format!(
            "Thunderbolt Bridge found (members: {}), but --skip-bridge0 set",
            bridge0_members.join(", ")
        ));
    } else {
        step("bridge0", &format!(
            "Destroying Thunderbolt Bridge (freeing {})",
            bridge0_members.join(", ")
        ));
        if !dry_run {
            destroy_bridge0(&bridge0_members)?;
        }
    }

    // ── Step 2: Seed NodeMap from ~/.r1o/cluster.json ───────────────────
    let config_path = asmi_core::config::NodeMap::config_path();
    if config_path.exists() {
        step("config", &format!("NodeMap already exists at {}", config_path.display()));
    } else {
        step("config", "Seeding NodeMap from ~/.r1o/cluster.json");
        if !dry_run {
            // NodeMap::load() already handles the seed-from-cluster.json logic
            let nm = asmi_core::config::NodeMap::load();
            if nm.nodes.is_empty() {
                eprintln!("  ⚠ No nodes found — create ~/.r1o/cluster.json first");
            } else {
                println!("  Registered {} nodes: {}", nm.nodes.len(), nm.nodes.join(", "));
            }
        }
    }

    // ── Step 3: SSH key setup ─────────────────────────────────────────
    let ssh_key = home.join(".ssh/id_ed25519");
    if ssh_key.exists() {
        step("ssh", "SSH key already exists");
    } else {
        step("ssh", "Generating ed25519 SSH key (no passphrase)");
        if !dry_run {
            let ssh_dir = home.join(".ssh");
            std::fs::create_dir_all(&ssh_dir).ok();
            let keygen = Command::new("ssh-keygen")
                .args(["-t", "ed25519", "-N", "", "-f", ssh_key.to_str().unwrap_or("")])
                .output();
            if keygen.map_or(false, |o| o.status.success()) {
                println!("  Generated {}", ssh_key.display());
            } else {
                eprintln!("  ⚠ ssh-keygen failed");
            }
        }
    }

    // Distribute SSH key to other cluster nodes (if NodeMap has nodes)
    let nm = asmi_core::config::NodeMap::load();
    let pub_key_path = home.join(".ssh/id_ed25519.pub");
    let remote_nodes: Vec<&str> = nm.nodes.iter()
        .filter(|n| n.as_str() != hostname)
        .map(|s| s.as_str())
        .collect();

    if remote_nodes.is_empty() || !pub_key_path.exists() {
        if !remote_nodes.is_empty() {
            step("ssh", "No public key to distribute — skipping key push");
        }
    } else {
        step("ssh", &format!("Distributing key to {} nodes", remote_nodes.len()));
        if !dry_run {
            let pub_key = std::fs::read_to_string(&pub_key_path).unwrap_or_default();
            for node in &remote_nodes {
                // Check if key is already in authorized_keys via SSH
                let check = Command::new("ssh")
                    .args(["-o", "ConnectTimeout=3", "-o", "BatchMode=yes", node,
                           &format!("grep -qF '{}' ~/.ssh/authorized_keys 2>/dev/null && echo exists || echo missing",
                                    pub_key.trim())])
                    .output();

                let needs_push = check.as_ref()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "missing")
                    .unwrap_or(true);

                if needs_push {
                    // Try ssh-copy-id first (handles authorized_keys creation + permissions)
                    let copy = Command::new("ssh-copy-id")
                        .args(["-i", pub_key_path.to_str().unwrap_or(""),
                               &format!("ma@{}", node)])
                        .output();
                    if copy.map_or(false, |o| o.status.success()) {
                        println!("  {} — key installed", node);
                    } else {
                        println!("  {} — ssh-copy-id failed (may need manual setup)", node);
                    }
                } else {
                    println!("  {} — key already present", node);
                }
            }
        }
    }

    // ── Step 4: Scoped passwordless sudo ────────────────────────────────
    let sudoers_path = "/etc/sudoers.d/r1o-asmi";
    let sudoers_exists = std::path::Path::new(sudoers_path).exists();
    if sudoers_exists {
        step("sudo", "Scoped sudoers already configured");
    } else {
        let user = std::env::var("USER").unwrap_or_else(|_| "ma".to_string());
        step("sudo", &format!("Installing scoped NOPASSWD rules for {}", user));
        if !dry_run {
            // Write to temp file, validate with visudo, then install
            let rules = format!(
                "# r1o/asmi — scoped passwordless sudo for cluster operations\n\
                 {} ALL=(ALL) NOPASSWD: /usr/libexec/PlistBuddy\n\
                 {} ALL=(ALL) NOPASSWD: /usr/sbin/networksetup\n\
                 {} ALL=(ALL) NOPASSWD: /usr/bin/killall configd\n\
                 {} ALL=(ALL) NOPASSWD: /usr/sbin/ipconfig\n\
                 {} ALL=(ALL) NOPASSWD: /usr/bin/powermetrics\n\
                 {} ALL=(ALL) NOPASSWD: /usr/sbin/systemsetup\n",
                user, user, user, user, user, user
            );
            let tmp = "/tmp/.r1o-sudoers-staging";
            if std::fs::write(tmp, &rules).is_ok() {
                // Validate syntax
                let valid = Command::new("sudo")
                    .args(["visudo", "-cf", tmp])
                    .output()
                    .map_or(false, |o| o.status.success());

                if valid {
                    let install = Command::new("sudo")
                        .args(["cp", tmp, sudoers_path])
                        .output()
                        .map_or(false, |o| o.status.success());
                    let chmod = Command::new("sudo")
                        .args(["chmod", "440", sudoers_path])
                        .output()
                        .map_or(false, |o| o.status.success());

                    if install && chmod {
                        println!("  Installed {} (6 scoped rules)", sudoers_path);
                    } else {
                        eprintln!("  ⚠ Failed to install sudoers file");
                    }
                } else {
                    eprintln!("  ⚠ Sudoers syntax validation failed — skipping");
                }
                let _ = std::fs::remove_file(tmp);
            }
        }
    }

    // ── Step 5: SSH ControlMaster config ──────────────────────────────
    let ssh_config = home.join(".ssh/config");
    let ssh_config_content = std::fs::read_to_string(&ssh_config).unwrap_or_default();
    if ssh_config_content.contains("ControlMaster") {
        step("ssh-mux", "SSH ControlMaster already configured");
    } else {
        step("ssh-mux", "Adding ControlMaster to ~/.ssh/config");
        if !dry_run {
            let block = "\n\
# r1o cluster — persistent SSH multiplexing\n\
Host *\n\
    ControlMaster auto\n\
    ControlPath /tmp/asmi-%r@%h:%p\n\
    ControlPersist 10m\n\
    AddKeysToAgent yes\n\
    UseKeychain yes\n\
    IdentityFile ~/.ssh/id_ed25519\n\
    ServerAliveInterval 30\n\
    ServerAliveCountMax 3\n";
            let mut content = ssh_config_content.clone();
            content.push_str(block);
            if std::fs::write(&ssh_config, &content).is_ok() {
                println!("  ControlPath: /tmp/asmi-<user>@<host>:<port>");
                println!("  ControlPersist: 10m (connections stay open after last session)");
            } else {
                eprintln!("  ⚠ Failed to write ~/.ssh/config");
            }
        }
    }

    // ── Step 6: autossh for persistent node connections ─────────────────
    let has_autossh = Command::new("which").arg("autossh").output()
        .map_or(false, |o| o.status.success());

    if remote_nodes.is_empty() {
        step("autossh", "No remote nodes — skipping persistent connections");
    } else if !has_autossh {
        step("autossh", "Installing autossh via Homebrew");
        if !dry_run {
            let brew = Command::new("brew")
                .args(["install", "autossh"])
                .output();
            if brew.map_or(false, |o| o.status.success()) {
                println!("  autossh installed");
            } else {
                eprintln!("  ⚠ brew install autossh failed — install manually");
            }
        }
    } else {
        step("autossh", "autossh already installed");
    }

    // Write per-node autossh launchd plists
    if !remote_nodes.is_empty() {
        let autossh_bin = Command::new("which").arg("autossh").output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "/opt/homebrew/bin/autossh".to_string());
        let user = std::env::var("USER").unwrap_or_else(|_| "ma".to_string());
        let la_dir = home.join("Library/LaunchAgents");

        for node in &remote_nodes {
            let plist_name = format!("com.r1o.autossh.{}.plist", node);
            let plist_path = la_dir.join(&plist_name);

            if plist_path.exists() {
                println!("  [autossh] {} — plist exists", node);
                continue;
            }

            step("autossh", &format!("Creating persistent connection to {}", node));
            if !dry_run {
                // autossh with ControlMaster: opens an SSH mux master connection
                // that stays alive. asmi and mlx.launch reuse it via ControlPath.
                // -M 0 disables autossh's monitoring port (ServerAlive handles it)
                // -N = no command, -f = background (but launchd manages lifecycle)
                let content = format!(
r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.r1o.autossh.{node}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{autossh}</string>
        <string>-M</string>
        <string>0</string>
        <string>-N</string>
        <string>-o</string>
        <string>ServerAliveInterval=30</string>
        <string>-o</string>
        <string>ServerAliveCountMax=3</string>
        <string>-o</string>
        <string>ExitOnForwardFailure=yes</string>
        <string>-o</string>
        <string>ControlMaster=auto</string>
        <string>-o</string>
        <string>ControlPath=/tmp/asmi-{user}@{node}:22</string>
        <string>-o</string>
        <string>ControlPersist=yes</string>
        <string>{user}@{node}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>30</integer>
    <key>StandardOutPath</key>
    <string>/tmp/autossh-{node}.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/autossh-{node}.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>AUTOSSH_GATETIME</key>
        <string>0</string>
    </dict>
</dict>
</plist>"#,
                    node = node,
                    autossh = autossh_bin,
                    user = user,
                );

                if std::fs::write(&plist_path, &content).is_ok() {
                    // Load the agent
                    let uid = Command::new("id").arg("-u").output()
                        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                        .unwrap_or_else(|_| "501".to_string());
                    let _ = Command::new("launchctl")
                        .args(["bootstrap", &format!("gui/{}", uid), plist_path.to_str().unwrap_or("")])
                        .output();
                    println!("  {} — autossh agent started (mux: /tmp/asmi-{}@{}:22)", node, user, node);
                }
            }
        }
    }

    // ── Step 7: Install binary ───────────────────────────────────────
    let cargo_bin = home.join(".cargo/bin/asmi");
    let current_exe = std::env::current_exe().unwrap_or_default();
    if current_exe == cargo_bin {
        step("binary", "Already running from ~/.cargo/bin/asmi");
    } else {
        step("binary", &format!("Installing to {}", cargo_bin.display()));
        if !dry_run {
            std::fs::copy(&current_exe, &cargo_bin)
                .context("failed to copy binary to ~/.cargo/bin/")?;
            // Codesign
            let _ = Command::new("codesign")
                .args(["-f", "-s", "-", cargo_bin.to_str().unwrap_or("")])
                .output();
            println!("  Installed and codesigned");
        }
    }

    // ── Step 4: Write launchd plist ─────────────────────────────────────
    let plist_path = home.join(format!("Library/LaunchAgents/{}.plist", PLIST_LABEL));
    let bin_path = cargo_bin.to_str().unwrap_or("/Users/ma/.cargo/bin/asmi");

    let mut args_xml = format!(
        "        <string>{}</string>\n\
         \x20       <string>--serve</string>\n\
         \x20       <string>--port</string>\n\
         \x20       <string>{}</string>",
        bin_path, port
    );
    if cluster {
        args_xml.push_str("\n        <string>--cluster</string>");
    }

    let log_path = home.join("Library/Logs/asmi-daemon.log");
    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{args}
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>5</integer>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>"#,
        label = PLIST_LABEL,
        args = args_xml,
        log = log_path.display(),
    );

    step("plist", &format!("Writing {}", plist_path.display()));
    if !dry_run {
        if let Some(parent) = plist_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&plist_path, &plist_content)
            .context("failed to write launchd plist")?;
    }

    // ── Step 5: Load / restart daemon ───────────────────────────────────
    step("daemon", "Starting asmi daemon via launchctl");
    if !dry_run {
        let uid = Command::new("id").arg("-u").output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "501".to_string());
        let gui_domain = format!("gui/{}", uid);

        // Try to unload existing (ignore failure — it may not be loaded)
        let _ = Command::new("launchctl")
            .args(["bootout", &gui_domain, plist_path.to_str().unwrap_or("")])
            .output();

        let load = Command::new("launchctl")
            .args(["bootstrap", &gui_domain, plist_path.to_str().unwrap_or("")])
            .output()
            .context("failed to run launchctl bootstrap")?;

        if load.status.success() {
            println!("  Daemon started on port {}", port);
        } else {
            let stderr = String::from_utf8_lossy(&load.stderr);
            if stderr.contains("already loaded") || stderr.contains("service already loaded") {
                // Kick the existing one
                let _ = Command::new("launchctl")
                    .args(["kickstart", "-k", &format!("{}/{}", gui_domain, PLIST_LABEL)])
                    .output();
                println!("  Daemon restarted on port {}", port);
            } else {
                eprintln!("  ⚠ launchctl bootstrap failed: {}", stderr.trim());
            }
        }
    }

    // ── Step 6: Verify ──────────────────────────────────────────────────
    if !dry_run {
        step("verify", "Checking daemon health...");
        std::thread::sleep(std::time::Duration::from_secs(2));

        let health = Command::new("curl")
            .args(["-s", "--max-time", "3", &format!("http://localhost:{}/health", port)])
            .output();

        match health {
            Ok(o) if o.status.success() => {
                let body = String::from_utf8_lossy(&o.stdout);
                println!("  ✓ {}", body.trim());
            }
            _ => {
                eprintln!("  ⚠ Daemon not responding on port {} — check logs:", port);
                eprintln!("    tail -20 {}", log_path.display());
            }
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!();
    if dry_run {
        println!("Dry run complete. Re-run without --dry-run to apply.");
    } else {
        println!("Setup complete. asmi is running on port {}.", port);
        if !bridge0_members.is_empty() && !skip_bridge0 {
            println!("  Thunderbolt Bridge removed — RDMA interfaces freed.");
        }
        println!();
        println!("Remaining manual step (one-time, requires physical access):");
        println!("  Hold Power → Options → Terminal → rdma_ctl enable → reboot");
        println!();
        println!("After RDMA is enabled, distributed inference works immediately:");
        println!("  mlx.launch --hostfile /path/to/hostfile.json your_script.py");
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn step(name: &str, msg: &str) {
    println!("[{:>8}] {}", name, msg);
}

/// Detect bridge0 member interfaces from ifconfig.
fn detect_bridge0() -> Vec<String> {
    let output = Command::new("ifconfig").arg("bridge0").output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.trim().starts_with("member:"))
                .filter_map(|l| l.split_whitespace().nth(1).map(String::from))
                .collect()
        }
        _ => vec![],
    }
}

/// Destroy bridge0: remove from preferences, create r1o TB services, restart configd.
fn destroy_bridge0(members: &[String]) -> Result<()> {
    // 1. Remove from preferences.plist (requires sudo)
    let rm = Command::new("sudo")
        .args(["/usr/libexec/PlistBuddy", "-c",
               "Delete :VirtualNetworkInterfaces:Bridge:bridge0",
               PREFERENCES_PLIST])
        .output()
        .context("failed to run PlistBuddy")?;

    if !rm.status.success() {
        let stderr = String::from_utf8_lossy(&rm.stderr);
        // Not fatal if key doesn't exist
        if !stderr.contains("Does Not Exist") {
            eprintln!("  ⚠ PlistBuddy warning: {}", stderr.trim());
        }
    }

    // 2. Create network services for each freed interface
    for iface in members {
        let svc_name = format!("r1o TB {}", iface);
        let _ = Command::new("sudo")
            .args(["networksetup", "-createnetworkservice", &svc_name, iface])
            .output();
        println!("  Created service: {}", svc_name);
    }

    // 3. Restart configd to apply
    println!("  Restarting configd (network will briefly disconnect)...");
    let _ = Command::new("sudo")
        .args(["killall", "configd"])
        .output();

    // 4. Wait for IPv4LL assignment
    std::thread::sleep(std::time::Duration::from_secs(5));
    let _ = Command::new("sudo")
        .args(["ipconfig", "waitall"])
        .output();

    // 5. Report new IPs
    for iface in members {
        let out = Command::new("ifconfig").arg(iface).output();
        if let Ok(o) = out {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines() {
                if line.contains("inet 169.254") {
                    if let Some(ip) = line.split_whitespace().nth(1) {
                        println!("  {}: {}", iface, ip);
                    }
                }
            }
        }
    }

    // 6. Verify bridge0 is gone
    let verify = Command::new("ifconfig").arg("bridge0").output();
    if verify.map_or(true, |o| !o.status.success()) {
        println!("  ✓ bridge0 destroyed");
    } else {
        eprintln!("  ⚠ bridge0 still present after removal attempt");
    }

    Ok(())
}
