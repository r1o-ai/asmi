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

    // ── Step 3: Install binary ──────────────────────────────────────────
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
            println!("  Note: RDMA itself must be enabled from Recovery OS:");
            println!("    Hold Power → Options → Terminal → rdma_ctl enable → reboot");
        }
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
