//! Metrics collector — HTTP-first (daemon), SSH fallback.
//!
//! For remote nodes: fetch from `http://<host>:9090/metrics` if the asmi
//! daemon is running. This avoids SSH overhead (~200-500ms per session) and
//! keeps the TUI responsive. Falls back to 3 parallel SSH commands if the
//! daemon is unreachable.
//!
//! Local node always uses direct command execution (powermetrics needs sudo).

use crate::config::ClusterConfig;
use crate::ssh::{local_run, ssh_run};
use crate::types::*;
use chrono::Utc;
use regex::Regex;
use std::sync::{LazyLock, OnceLock};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Static regexes (compiled once, reused across all poll ticks)
// ---------------------------------------------------------------------------

// parse_powermetrics_text
static POWER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(CPU|GPU|ANE) Power:\s+([\d.]+)\s+mW").unwrap()
});
static GPU_ACTIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"GPU HW active residency:\s+([\d.]+)%").unwrap()
});
static GPU_IDLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"GPU idle residency:\s+([\d.]+)%").unwrap()
});
static CPU_ACTIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^CPU \d+ active residency:\s+([\d.]+)%").unwrap()
});

// parse_vmstat_and_memsize
static PAGE_SIZE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(page size of (\d+) bytes\)").unwrap()
});
static PAGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^Pages (\w[\w ]*\w):\s+(\d+)\.").unwrap()
});
static ANON_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^Anonymous pages:\s+(\d+)\.").unwrap()
});
static FILE_BACKED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^File-backed pages:\s+(\d+)\.").unwrap()
});

// parse_ps_mlx
static PS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^\s*(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+)$"
    ).unwrap()
});

// parse_cpu_clusters (from powermetrics processor usage section)
static CLUSTER_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(E\d+|P\d+)-Cluster").unwrap()
});
static CLUSTER_FREQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(E\d+|P\d+)-Cluster HW active frequency:\s+(\d+)\s+MHz").unwrap()
});
static CLUSTER_RES_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(E\d+|P\d+)-Cluster HW active residency:\s+([\d.]+)%").unwrap()
});
static CPU_FREQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^CPU (\d+) frequency:\s+(\d+)\s+MHz").unwrap()
});
static GPU_FREQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"GPU HW active frequency:\s+(\d+)\s+MHz").unwrap()
});

// parse_footprint
static FOOTPRINT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)").unwrap()
});

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

const CMD_POWERMETRICS: &str =
    "sudo powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power 2>/dev/null";

const CMD_VMSTAT_SYSCTL: &str =
    "hostname -s; echo '---HOSTNAME---'; vm_stat; echo '---MEMSIZE---'; sysctl -n hw.memsize";

/// Hardware identity via system_profiler (for remote nodes via SSH).
const CMD_HW_IDENTITY: &str =
    "system_profiler SPHardwareDataType 2>/dev/null | grep -E 'Chip:|Serial Number \\(system\\):|Model Name:'";

/// Parse system_profiler output into (chip, serial, model).
fn parse_hw_identity(text: &str) -> (Option<String>, Option<String>, Option<String>) {
    let mut chip = None;
    let mut serial = None;
    let mut model = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("Chip:") {
            chip = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Serial Number (system):") {
            serial = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Model Name:") {
            model = Some(v.trim().to_string());
        }
    }
    (chip, serial, model)
}

/// Also captures mlx.launch (distributed launcher), mlx_lm.share, mlx_audio.
/// JACCL detection: --backend jaccl in args, or ps -E showing MLX_JACCL env vars.
const CMD_PS_MLX: &str =
    "ps aux | grep -E 'mlx_lm\\.(server|share)|mlx_vlm\\.server|vllm_mlx|mlx\\.launch|mlx_audio' | grep -v grep";

/// RDMA status + ifconfig in one command to minimize SSH connections.
const CMD_RDMA_NET: &str =
    "rdma_ctl status 2>/dev/null || echo disabled; echo '---'; ibv_devices 2>/dev/null || echo ''; echo '---'; ibv_devinfo 2>/dev/null || echo ''; echo '===IFCONFIG==='; ifconfig 2>/dev/null";

/// Detect JACCL/distributed by checking `mlx.launch` wrapper processes.
/// Only `mlx.launch --backend jaccl` is a reliable signal — env vars
/// can persist from previous runs and cause false positives on single-node.
const CMD_JACCL_ENV: &str =
    "ps aux | grep 'mlx\\.launch.*--backend' | grep -v grep | head -5";

/// iostat: 2 samples, 1s apart. Take the second (real-time) sample.
const CMD_IOSTAT: &str =
    "iostat -d -C -K -c 2 -w 1 2>/dev/null";

/// Full process listing with PPID for tree building.
pub const CMD_PS_TREE: &str =
    "ps -axo pid,ppid,pcpu,pmem,rss,comm";

/// netstat -ib for network byte counters (used for delta-based throughput).
pub const CMD_NETSTAT_IB: &str = "netstat -ib 2>/dev/null";

/// Build the footprint command for a given PID.
fn footprint_cmd(pid: u32) -> String {
    format!("sudo footprint -p {pid} 2>/dev/null | head -2 | tail -1")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Collect a full metrics snapshot from a node.
///
/// Strategy:
/// - **Remote nodes**: try `http://<host>:<daemon_port>/metrics` first (fast,
///   no SSH overhead). Falls back to SSH if the daemon is unreachable.
/// - **Local node**: always runs commands directly to avoid self-referencing
///   the daemon's own HTTP endpoint (which would return stale cached data).
///
/// # INVARIANT: Self-Reference Prevention
///
/// When `is_local=true`, this function MUST NOT call `fetch_from_daemon()`.
/// The daemon calls this function in its own poll loop — if it fetches from
/// its own `/metrics` endpoint, it gets back its own cached (stale) snapshot,
/// creating an infinite self-referencing loop where processes are never
/// refreshed from `ps aux`.
///
/// This invariant is tested by `test_collect_node_metrics_local_bypasses_http`.
pub async fn collect_node_metrics(
    hostname: &str,
    config: &ClusterConfig,
    is_local: bool,
) -> NodeSnapshot {
    // Local node: run commands directly — never fetch from our own daemon
    // (that would return stale cached data, creating a self-referencing loop).
    if is_local {
        return collect_via_ssh(hostname, config, true).await;
    }

    // Remote nodes: try HTTP daemon first (fast, no SSH overhead).
    if let Some(snap) = fetch_from_daemon(hostname, config.daemon_port).await {
        return snap;
    }

    warn!(hostname, "daemon unreachable, falling back to SSH");
    collect_via_ssh(hostname, config, false).await
}

/// Fetch a NodeSnapshot from the asmi daemon HTTP endpoint.
/// Returns `None` if unreachable or response fails to parse.
/// Public so `scanner` can use it for HTTP-first online checks.
pub async fn fetch_from_daemon(hostname: &str, port: u16) -> Option<NodeSnapshot> {
    // Try bare hostname first, then <hostname>.local (mDNS fallback).
    let candidates = [
        format!("http://{}:{}/metrics", hostname, port),
        format!("http://{}.local:{}/metrics", hostname, port),
    ];
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    for url in &candidates {
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<NodeSnapshot>().await {
                    Ok(snap) => {
                        debug!(hostname, url, "fetched metrics from daemon");
                        return Some(snap);
                    }
                    Err(e) => {
                        warn!(hostname, url, error = %e, "daemon metrics parse error");
                    }
                }
            }
            Ok(resp) => {
                debug!(hostname, url, status = %resp.status(), "daemon returned error status");
            }
            Err(_) => {
                // Connection refused / timeout — try next candidate silently
            }
        }
    }
    None
}

/// Cached local hardware identity from `system_profiler SPHardwareDataType`.
/// Runs once per process lifetime (hardware doesn't change at runtime).
static LOCAL_HW_IDENTITY: OnceLock<(Option<String>, Option<String>, Option<String>)> =
    OnceLock::new();

pub fn local_hardware_identity() -> (Option<String>, Option<String>, Option<String>) {
    let (c, s, m) = LOCAL_HW_IDENTITY.get_or_init(|| {
        let output = std::process::Command::new("system_profiler")
            .arg("SPHardwareDataType")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default();
        parse_hw_identity(&output)
    });
    (c.clone(), s.clone(), m.clone())
}

/// Collect metrics via SSH commands (original path, used for local node
/// and as fallback for remote nodes without a running daemon).
async fn collect_via_ssh(
    hostname: &str,
    config: &ClusterConfig,
    is_local: bool,
) -> NodeSnapshot {
    info!(hostname, is_local, "collecting metrics via SSH");
    let run = |cmd: &'static str| async move {
        if is_local {
            local_run(cmd).await
        } else {
            ssh_run(hostname, cmd, config).await
        }
    };

    // 1. Run commands in parallel (+ hardware identity for remote nodes)
    let (power_res, mem_res, ps_res, jaccl_res, rdma_net_res, iostat_res) = tokio::join!(
        run(CMD_POWERMETRICS),
        run(CMD_VMSTAT_SYSCTL),
        run(CMD_PS_MLX),
        run(CMD_JACCL_ENV),
        run(CMD_RDMA_NET),
        run(CMD_IOSTAT),
    );
    // Hardware identity: local uses cached OnceLock, remote runs via SSH
    let hw_identity = if is_local {
        local_hardware_identity()
    } else {
        match run(CMD_HW_IDENTITY).await {
            Ok(r) if r.has_output() => parse_hw_identity(&r.stdout),
            _ => (None, None, None),
        }
    };

    // 2. Parse powermetrics
    let power = match &power_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "powermetrics OK");
            parse_powermetrics_text(&r.stdout)
        }
        Ok(r) => {
            debug!(hostname, stderr = r.stderr.as_str(), "powermetrics empty/failed");
            PowerMetricsResult::default()
        }
        Err(e) => {
            warn!(hostname, error = %e, "powermetrics command error");
            PowerMetricsResult::default()
        }
    };

    // 3. Parse hostname -s + vm_stat + sysctl
    // The combined command outputs: hostname\n---HOSTNAME---\nvm_stat...\n---MEMSIZE---\nmemsize
    let (resolved_hostname, mem_stats) = match &mem_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "vm_stat/sysctl OK");
            let (resolved, vmstat_text) = match r.stdout.split_once("---HOSTNAME---\n") {
                Some((h, rest)) => (Some(h.trim().to_string()), rest.to_string()),
                None => (None, r.stdout.clone()),
            };
            (resolved, parse_vmstat_and_memsize(&vmstat_text))
        }
        Ok(r) => {
            debug!(hostname, stderr = r.stderr.as_str(), "vm_stat/sysctl empty/failed");
            (None, MemoryStats::default())
        }
        Err(e) => {
            warn!(hostname, error = %e, "vm_stat/sysctl command error");
            (None, MemoryStats::default())
        }
    };

    // Use SSH-resolved hostname as canonical name to prevent duplicates
    // when the same node is discovered via different networks (LAN vs TB).
    let canonical_hostname = resolved_hostname
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| hostname.to_string());

    // 4. Parse ps aux for MLX processes
    let mut processes = match &ps_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "ps aux OK");
            parse_ps_mlx(&r.stdout)
        }
        Ok(_) => {
            debug!(hostname, "no MLX processes found");
            Vec::new()
        }
        Err(e) => {
            warn!(hostname, error = %e, "ps aux command error");
            Vec::new()
        }
    };

    // 5. Detect distributed backend from mlx.launch wrapper processes.
    // Only tag child processes (mlx_lm.server/share) when a matching
    // mlx.launch with --backend is running on the same node.
    let dist_backend = match &jaccl_res {
        Ok(r) if r.has_output() => {
            if r.stdout.contains("--backend jaccl") {
                Some(DistributedBackend::Jaccl)
            } else if r.stdout.contains("--backend ring") {
                Some(DistributedBackend::Ring)
            } else {
                None
            }
        }
        _ => None,
    };
    if let Some(backend) = dist_backend {
        for proc in &mut processes {
            if proc.distributed.is_none() {
                proc.distributed = Some(backend);
            }
        }
    }

    // 6. Verify actual listening ports via lsof (parallel, one per PID).
    //    The --port flag from ps aux may not match the real socket when a
    //    server fails to bind and falls back to a different port.
    if !processes.is_empty() {
        let port_futs: Vec<_> = processes.iter().map(|proc| {
            let pid = proc.pid;
            let hostname_owned = hostname.to_string();
            let config_clone = config.clone();
            async move {
                let cmd = format!(
                    "lsof -a -p {pid} -iTCP -sTCP:LISTEN -P -n 2>/dev/null | awk 'NR>1 {{print $9}}' | head -1"
                );
                let res = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    async {
                        if is_local {
                            local_run(&cmd).await
                        } else {
                            ssh_run(&hostname_owned, &cmd, &config_clone).await
                        }
                    },
                ).await;
                let actual_port: Option<u16> = match res {
                    Ok(Ok(r)) if r.has_output() => {
                        // lsof output looks like "*:8091" or "127.0.0.1:8091"
                        r.stdout.trim()
                            .rsplit(':')
                            .next()
                            .and_then(|p| p.parse().ok())
                    }
                    _ => None,
                };
                (pid, actual_port)
            }
        }).collect();

        let port_results = futures::future::join_all(port_futs).await;
        for (pid, actual_port) in port_results {
            if let (Some(proc), Some(real_port)) = (processes.iter_mut().find(|p| p.pid == pid), actual_port) {
                match proc.port {
                    Some(cli_port) if cli_port != real_port => {
                            // CLI --port takes priority over lsof. The lsof
                            // port may be an internal IPC socket (e.g. Python
                            // runtime or Metal framework) rather than the
                            // server's actual listening port.
                            debug!(
                                hostname,
                                pid,
                                cli_port,
                                lsof_port = real_port,
                                "port mismatch: CLI --port wins over lsof"
                            );
                            // Keep proc.port unchanged (CLI value).
                        }
                        Some(_) => {
                            // CLI and lsof agree — nothing to do.
                        }
                        None => {
                            // No --port on command line; use lsof as fallback.
                            proc.port = Some(real_port);
                        }
                    }
            }
        }
    }

    // 7. Enrich each process with footprint data (all PIDs in parallel)
    if !processes.is_empty() {
        let fp_futs: Vec<_> = processes.iter().map(|proc| {
            let cmd = footprint_cmd(proc.pid);
            let pid = proc.pid;
            let hostname = hostname.to_string();
            let config = config.clone();
            async move {
                let fp_res = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    async {
                        if is_local {
                            local_run(&cmd).await
                        } else {
                            ssh_run(&hostname, &cmd, &config).await
                        }
                    },
                ).await;
                let mb = match fp_res {
                    Ok(Ok(r)) if r.has_output() => {
                        let mb = parse_footprint(&r.stdout);
                        debug!(hostname = hostname.as_str(), pid, footprint_mb = ?mb, "footprint");
                        mb
                    }
                    _ => {
                        debug!(hostname = hostname.as_str(), pid, "footprint unavailable");
                        None
                    }
                };
                (pid, mb)
            }
        }).collect();

        let results = futures::future::join_all(fp_futs).await;
        for (pid, mb) in results {
            if let Some(proc) = processes.iter_mut().find(|p| p.pid == pid) {
                proc.footprint_mb = mb;
            }
        }
    }

    // 8. Enrich processes with /v1/models metadata (non-blocking, best-effort)
    probe_model_endpoints(hostname, config, is_local, &mut processes).await;

    // 9. Parse RDMA status + interface IPs
    let (rdma_status, interface_ips) = match &rdma_net_res {
        Ok(r) if r.has_output() => {
            let delim = "===IFCONFIG===";
            let (rdma_text, ifconfig_text) = match r.stdout.find(delim) {
                Some(idx) => (&r.stdout[..idx], &r.stdout[idx + delim.len()..]),
                None => (r.stdout.as_str(), ""),
            };
            let rdma = crate::scanner::parse_rdma_status(rdma_text);
            let all_ips = crate::scanner::parse_ifconfig_all_ips(ifconfig_text);
            // Filter to RDMA-relevant IPs only (192.168.0.x, 169.254.x.x)
            let mut iface_ips = std::collections::BTreeMap::new();
            for (iface, ips) in all_ips {
                let rdma_ips: Vec<String> = ips
                    .into_iter()
                    .filter(|ip| ip.starts_with("192.168.0.") || ip.starts_with("169.254."))
                    .collect();
                if !rdma_ips.is_empty() {
                    iface_ips.insert(iface, rdma_ips);
                }
            }
            debug!(hostname, devices = rdma.devices.len(), ifaces = iface_ips.len(), "RDMA + ifconfig parsed");
            (Some(rdma), iface_ips)
        }
        _ => {
            debug!(hostname, "RDMA/ifconfig unavailable");
            (None, std::collections::BTreeMap::new())
        }
    };

    // 10. Assemble NodeSnapshot
    // ram_percent reflects actual app usage (excludes file cache)
    let ram_percent = if mem_stats.total_bytes > 0 {
        (mem_stats.app_bytes as f64 / mem_stats.total_bytes as f64) * 100.0
    } else {
        0.0
    };

    let (chip_model, serial_number, model_name) = hw_identity;

    // A remote node with 0 total RAM means all SSH commands failed (unreachable).
    let online = is_local || mem_stats.total_bytes > 0;
    if !online {
        warn!(hostname, "marking node offline — SSH collection returned no data");
    }

    NodeSnapshot {
        hostname: canonical_hostname,
        online,
        timestamp: Utc::now(),
        chip_model,
        serial_number,
        model_name,
        cpu_watts: power.cpu_mw,
        gpu_watts: power.gpu_mw,
        ane_watts: power.ane_mw,
        power_source: None,
        cpu_percent: power.cpu_percent,
        gpu_percent: power.gpu_percent,
        ram_used_bytes: mem_stats.used_bytes,
        ram_total_bytes: mem_stats.total_bytes,
        ram_percent,
        ram_app_bytes: mem_stats.app_bytes,
        ram_cached_bytes: mem_stats.cached_bytes,
        cpu_clusters: power.cpu_clusters,
        gpu_frequency_mhz: power.gpu_frequency_mhz,
        disk_io: match &iostat_res {
            Ok(r) if r.has_output() => {
                debug!(hostname, "iostat OK");
                parse_iostat(&r.stdout)
            }
            _ => {
                debug!(hostname, "iostat unavailable");
                None
            }
        },
        network: None,   // populated by daemon poll loop via netstat diff
        cpu_temp_c: None,
        gpu_temp_c: None,
        processes,
        top_tasks: Vec::new(),
        rdma: rdma_status,
        interface_ips,
    }
}

// ---------------------------------------------------------------------------
// /v1/models endpoint probing
// ---------------------------------------------------------------------------

/// Probe `/v1/models` on each process that has a port, enriching with server metadata.
/// Failures leave `server_models` empty (not an error).
async fn probe_model_endpoints(
    hostname: &str,
    config: &ClusterConfig,
    is_local: bool,
    processes: &mut [ProcessInfo],
) {
    let client = if is_local {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(2))
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .ok()
    } else {
        None
    };

    let probes: Vec<_> = processes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.port.map(|port| (i, port)))
        .map(|(i, port)| {
            let hostname = hostname.to_string();
            let config = config.clone();
            let client = client.clone();
            async move {
                let endpoint = "/v1/models";

                let models = if is_local {
                    if let Some(client) = client {
                        let url = format!("http://127.0.0.1:{}{}", port, endpoint);
                        match client.get(&url).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                match resp.text().await {
                                    Ok(json) => {
                                        crate::scanner::parse_v1_models_metadata(&json)
                                    }
                                    Err(_) => Vec::new(),
                                }
                            }
                            _ => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    let curl_cmd = format!(
                        "curl -s --connect-timeout 2 --max-time 5 http://127.0.0.1:{}{} 2>/dev/null",
                        port, endpoint
                    );
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(8),
                        ssh_run(&hostname, &curl_cmd, &config),
                    ).await;
                    match result {
                        Ok(Ok(ref r)) if r.has_output() => {
                            crate::scanner::parse_v1_models_metadata(&r.stdout)
                        }
                        _ => Vec::new(),
                    }
                };
                (i, models)
            }
        })
        .collect();

    for (i, models) in futures::future::join_all(probes).await {
        processes[i].server_models = models;
    }
}

// ---------------------------------------------------------------------------
// Intermediate parsed result from powermetrics
// ---------------------------------------------------------------------------

/// Intermediate result from parsing powermetrics text output.
#[derive(Debug, Clone, Default)]
pub struct PowerMetricsResult {
    /// CPU power in milliwatts.
    pub cpu_mw: f64,
    /// GPU power in milliwatts.
    pub gpu_mw: f64,
    /// ANE power in milliwatts.
    pub ane_mw: f64,
    /// Average CPU active residency across all cores (percent).
    pub cpu_percent: f64,
    /// GPU HW active residency (percent).
    pub gpu_percent: f64,
    /// Per-cluster CPU breakdown (E0, P0, E1, P1, etc.).
    pub cpu_clusters: Vec<CpuClusterInfo>,
    /// GPU HW active frequency in MHz.
    pub gpu_frequency_mhz: Option<u32>,
}

// ---------------------------------------------------------------------------
// Parsers (each is a standalone, testable function)
// ---------------------------------------------------------------------------

/// Parse macOS powermetrics text output (non-JSON mode).
///
/// Extracts:
/// - CPU/GPU/ANE power in mW from the summary block (`CPU Power: 8916 mW`)
/// - GPU active residency from `GPU HW active residency: 100.00%`
/// - Average CPU active residency from all `CPU N active residency:  X.XX%` lines
///
/// Note: The testdata has two `GPU Power:` lines — one in the CPU summary block
/// and one in the GPU usage section. We take the **first** `CPU Power` and `ANE Power`
/// from the summary block, and the **last** `GPU Power` (from the GPU section) since
/// it is the more specific GPU-section measurement. However, the summary block's
/// `GPU Power` and the GPU section's `GPU Power` may differ slightly. We use the
/// values from the summary block (first occurrence) to match the `Combined Power`
/// line.
pub fn parse_powermetrics_text(text: &str) -> PowerMetricsResult {
    let mut result = PowerMetricsResult::default();

    // Power values from the summary block:
    //   CPU Power: 8916 mW
    //   GPU Power: 9462 mW
    //   ANE Power: 0 mW
    //   Combined Power (CPU + GPU + ANE): 18378 mW
    //
    // There may be a second "GPU Power:" in the GPU section. We take the first
    // occurrence of each to stay consistent with Combined Power.
    let mut seen_cpu = false;
    let mut seen_gpu = false;
    let mut seen_ane = false;
    for cap in POWER_RE.captures_iter(text) {
        let kind = &cap[1];
        let mw: f64 = cap[2].parse().unwrap_or(0.0);
        match kind {
            "CPU" if !seen_cpu => {
                result.cpu_mw = mw;
                seen_cpu = true;
            }
            "GPU" if !seen_gpu => {
                result.gpu_mw = mw;
                seen_gpu = true;
            }
            "ANE" if !seen_ane => {
                result.ane_mw = mw;
                seen_ane = true;
            }
            _ => {}
        }
    }

    // GPU HW active residency: "GPU HW active residency: 100.00% (...)"
    if let Some(cap) = GPU_ACTIVE_RE.captures(text) {
        result.gpu_percent = cap[1].parse().unwrap_or(0.0);
    } else {
        // Fallback: derive from GPU idle residency
        if let Some(cap) = GPU_IDLE_RE.captures(text) {
            let idle: f64 = cap[1].parse().unwrap_or(0.0);
            result.gpu_percent = 100.0 - idle;
        }
    }

    // CPU active residency: "CPU N active residency:  X.XX% (...)"
    // Average across all CPU cores
    let mut sum = 0.0;
    let mut count = 0u32;
    for cap in CPU_ACTIVE_RE.captures_iter(text) {
        sum += cap[1].parse::<f64>().unwrap_or(0.0);
        count += 1;
    }
    if count > 0 {
        result.cpu_percent = sum / count as f64;
    }

    // GPU HW active frequency
    if let Some(cap) = GPU_FREQ_RE.captures(text) {
        result.gpu_frequency_mhz = cap[1].parse().ok();
    }

    // Per-cluster CPU breakdown
    result.cpu_clusters = parse_cpu_clusters(text);

    result
}

/// Breakdown of macOS memory categories from vm_stat.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total physical RAM from sysctl hw.memsize.
    pub total_bytes: u64,
    /// Legacy "used" = active + inactive + speculative + wired + compressor.
    /// Kept for backward compatibility. Includes file cache.
    pub used_bytes: u64,
    /// App memory = anonymous + wired + compressor. What processes actually need.
    pub app_bytes: u64,
    /// Cached memory = file-backed + speculative. File cache, immediately reclaimable.
    pub cached_bytes: u64,
}

/// Parse combined vm_stat + sysctl output.
///
/// The input format is:
/// ```text
/// Mach Virtual Memory Statistics: (page size of 16384 bytes)
/// Pages free:                                  9563156.
/// Pages active:                               11058419.
/// ...
/// ---MEMSIZE---
/// 549755813888
/// ```
///
/// Returns a `MemoryStats` separating app memory (anonymous + wired + compressor)
/// from cached memory (file-backed + speculative). `used_bytes` is the legacy total
/// for backward compatibility.
pub fn parse_vmstat_and_memsize(text: &str) -> MemoryStats {
    // Split on the separator
    let parts: Vec<&str> = text.splitn(2, "---MEMSIZE---").collect();
    let vmstat_text = parts.first().copied().unwrap_or("");
    let memsize_text = parts.get(1).copied().unwrap_or("");

    // Parse total memory from sysctl
    let total_bytes: u64 = memsize_text
        .trim()
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .parse()
        .unwrap_or(0);

    // Parse page size from vm_stat header
    let page_size: u64 = PAGE_SIZE_RE
        .captures(vmstat_text)
        .and_then(|c| c[1].parse().ok())
        .unwrap_or(16384);

    // Parse page counts
    let mut speculative: u64 = 0;
    let mut wired: u64 = 0;
    let mut compressor: u64 = 0;

    for cap in PAGE_RE.captures_iter(vmstat_text) {
        let name = &cap[1];
        let pages: u64 = cap[2].parse().unwrap_or(0);
        match name {
            "speculative" => speculative = pages,
            "wired down" => wired = pages,
            "occupied by compressor" => compressor = pages,
            _ => {}
        }
    }

    // Use anonymous/file-backed for accurate app vs cache split.
    // "Anonymous pages" = process-allocated memory (heap, stack, mmap private).
    // "File-backed pages" = memory-mapped files, dyld shared cache, file cache.
    let anonymous: u64 = ANON_RE
        .captures(vmstat_text)
        .and_then(|c| c[1].parse().ok())
        .unwrap_or(0);
    let file_backed: u64 = FILE_BACKED_RE
        .captures(vmstat_text)
        .and_then(|c| c[1].parse().ok())
        .unwrap_or(0);

    let app_bytes = (anonymous + wired + compressor) * page_size;
    let cached_bytes = (file_backed + speculative) * page_size;
    let used_bytes = app_bytes + cached_bytes;

    MemoryStats {
        total_bytes,
        used_bytes,
        app_bytes,
        cached_bytes,
    }
}

/// Parse `ps aux` output filtered for MLX/vLLM processes.
///
/// Each line has the format (columns separated by variable whitespace):
/// ```text
/// USER  PID  %CPU  %MEM  VSZ  RSS  TT  STAT  STARTED  TIME  COMMAND...
/// ```
///
/// Extracts framework, model path, port, CPU%, and MEM%.
/// Filters out chrome-devtools-mcp watchdog, Microsoft Teams watchdog,
/// and system watchdogd processes.
pub fn parse_ps_mlx(text: &str) -> Vec<ProcessInfo> {
    let mut procs = Vec::new();

    for cap in PS_RE.captures_iter(text) {
        let command = &cap[11];

        // Filter out watchdog/noise processes
        if command.contains("chrome-devtools-mcp")
            || command.contains("Microsoft Teams")
            || command.contains("/usr/libexec/watchdogd")
            || command.contains("watchdog.sh")
        {
            continue;
        }

        // Detect framework — check mlx.launch first since its command line
        // contains the child command (e.g. "mlx.launch ... -- python -m mlx_lm share")
        let framework = if command.contains("mlx.launch") {
            ProcessFramework::MlxLaunch
        } else if command.contains("mlx_lm.server") || command.contains("mlx_lm server") {
            ProcessFramework::MlxLm
        } else if command.contains("mlx_lm.share") || command.contains("mlx_lm share") {
            ProcessFramework::MlxLmShare
        } else if command.contains("mlx_vlm.server") || command.contains("mlx_vlm server") {
            ProcessFramework::MlxVlm
        } else if command.contains("vllm_mlx") {
            ProcessFramework::VllmMlx
        } else if command.contains("mlx_audio") {
            ProcessFramework::MlxAudio
        } else {
            continue; // Not a recognised ML process
        };

        let pid: u32 = cap[2].parse().unwrap_or(0);
        let cpu_percent: f64 = cap[3].parse().unwrap_or(0.0);
        let mem_percent: f64 = cap[4].parse().unwrap_or(0.0);

        // Extract model path
        let model = extract_flag_value(command, "--model");

        // Simplify model path to just the model name (last path component)
        let model = model.map(|m| {
            m.rsplit('/')
                .next()
                .unwrap_or(&m)
                .to_string()
        });

        // Extract port
        let port: Option<u16> = extract_flag_value(command, "--port")
            .and_then(|p| p.parse().ok());

        // Detect distributed backend from --backend flag
        let distributed = extract_flag_value(command, "--backend").and_then(|b| {
            match b.as_str() {
                "jaccl" => Some(DistributedBackend::Jaccl),
                "ring" => Some(DistributedBackend::Ring),
                _ => None,
            }
        });

        procs.push(ProcessInfo {
            pid,
            framework,
            model,
            port,
            cpu_percent,
            mem_percent,
            footprint_mb: None,
            distributed,
            server_models: Vec::new(),
        });
    }

    procs
}

/// Parse footprint output to extract memory in MB.
///
/// Input line format:
/// ```text
/// Python [62283]: 64-bit    Footprint: 199.2 GB (16384 bytes per page)
/// ```
/// or:
/// ```text
/// zsh [9002]: 64-bit    Footprint: 2272 KB (16384 bytes per page)
/// ```
///
/// Returns footprint in MB.
pub fn parse_footprint(text: &str) -> Option<f64> {
    let cap = FOOTPRINT_RE.captures(text)?;
    let value: f64 = cap[1].parse().ok()?;
    let unit = &cap[2];
    let mb = match unit {
        "KB" => value / 1024.0,
        "MB" => value,
        "GB" => value * 1024.0,
        _ => return None,
    };
    Some(mb)
}

// ---------------------------------------------------------------------------
// CPU cluster / frequency parsing (from powermetrics)
// ---------------------------------------------------------------------------

/// Parse per-cluster CPU breakdown from powermetrics text.
///
/// Iterates line-by-line tracking which cluster header we're under, then
/// assigns each `CPU N` line to its owning cluster. Also extracts cluster-level
/// frequency and residency.
pub fn parse_cpu_clusters(text: &str) -> Vec<CpuClusterInfo> {
    // 1. Collect cluster-level frequency and residency
    let mut cluster_freq: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut cluster_res: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for cap in CLUSTER_FREQ_RE.captures_iter(text) {
        let name = cap[1].to_string();
        let freq: u32 = cap[2].parse().unwrap_or(0);
        cluster_freq.insert(name, freq);
    }
    for cap in CLUSTER_RES_RE.captures_iter(text) {
        let name = cap[1].to_string();
        let res: f64 = cap[2].parse().unwrap_or(0.0);
        cluster_res.insert(name, res);
    }

    // 2. Collect per-CPU frequency and residency
    let mut cpu_freq: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut cpu_res: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();

    for cap in CPU_FREQ_RE.captures_iter(text) {
        let id: u32 = cap[1].parse().unwrap_or(0);
        let freq: u32 = cap[2].parse().unwrap_or(0);
        cpu_freq.insert(id, freq);
    }
    // Use a dedicated regex for per-CPU active residency with the CPU ID
    static CPU_ACTIVE_ID_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^CPU (\d+) active residency:\s+([\d.]+)%").unwrap()
    });
    for cap in CPU_ACTIVE_ID_RE.captures_iter(text) {
        let id: u32 = cap[1].parse().unwrap_or(0);
        let res: f64 = cap[2].parse().unwrap_or(0.0);
        cpu_res.insert(id, res);
    }

    // 3. Walk line-by-line, tracking current cluster, assign CPUs to clusters
    let mut current_cluster: Option<String> = None;
    let mut cluster_cpus: std::collections::HashMap<String, Vec<u32>> = std::collections::HashMap::new();

    // Sort cluster names by order of appearance
    let mut cluster_order: Vec<String> = Vec::new();

    for line in text.lines() {
        if let Some(cap) = CLUSTER_HEADER_RE.captures(line) {
            let name = cap[1].to_string();
            if !cluster_cpus.contains_key(&name) {
                cluster_order.push(name.clone());
                cluster_cpus.insert(name.clone(), Vec::new());
            }
            current_cluster = Some(name);
        } else if let Some(cap) = CPU_FREQ_RE.captures(line) {
            let cpu_id: u32 = cap[1].parse().unwrap_or(0);
            if let Some(ref cluster) = current_cluster {
                cluster_cpus.entry(cluster.clone()).or_default().push(cpu_id);
            }
        }
    }

    // 4. Build CpuClusterInfo structs
    cluster_order
        .iter()
        .map(|name| {
            let cluster_type = if name.starts_with('E') {
                ClusterType::Efficiency
            } else {
                ClusterType::Performance
            };
            let cpu_ids = cluster_cpus.get(name).cloned().unwrap_or_default();
            let cores: Vec<CoreInfo> = cpu_ids
                .iter()
                .map(|&id| CoreInfo {
                    id,
                    frequency_mhz: cpu_freq.get(&id).copied().unwrap_or(0),
                    active_residency: cpu_res.get(&id).copied().unwrap_or(0.0),
                })
                .collect();

            CpuClusterInfo {
                name: name.clone(),
                cluster_type,
                frequency_mhz: cluster_freq.get(name).copied().unwrap_or(0),
                active_residency: cluster_res.get(name).copied().unwrap_or(0.0),
                core_count: cores.len() as u32,
                cores,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Disk I/O parsing (from iostat)
// ---------------------------------------------------------------------------

/// Parse `iostat -d -C -K -c 2 -w 1` output.
///
/// Takes the **second** sample (first is cumulative since boot).
/// macOS iostat format:
/// ```text
///               disk0               disk3
///     KB/t  tps  MB/s     KB/t  tps  MB/s
///    21.91   29  0.62    54.03   72  3.79
///     0.00    0  0.00     0.00    0  0.00
/// ```
pub fn parse_iostat(text: &str) -> Option<DiskIoStats> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 4 {
        return None;
    }

    // First line: device names (e.g., "   disk0   disk3")
    let device_names: Vec<String> = lines[0]
        .split_whitespace()
        .filter(|s| s.starts_with("disk"))
        .map(|s| s.to_string())
        .collect();

    if device_names.is_empty() {
        return None;
    }

    // Second line: headers (KB/t  tps  MB/s repeated)
    // Third line: first sample (cumulative) — skip
    // Fourth line: second sample (real-time) — use this
    let data_line = if lines.len() >= 4 { lines[3] } else { lines[2] };
    let values: Vec<f64> = data_line
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    // Each device has 3 values: KB/t, tps, MB/s
    let mut devices = Vec::new();
    let mut total_mbps = 0.0;

    for (i, name) in device_names.iter().enumerate() {
        let base = i * 3;
        if base + 2 < values.len() {
            let kb_t = values[base];
            let tps = values[base + 1];
            let mbps = values[base + 2];
            total_mbps += mbps;
            devices.push(DiskDeviceIo {
                name: name.clone(),
                kb_per_transfer: kb_t,
                transfers_per_sec: tps,
                mb_per_sec: mbps,
            });
        }
    }

    Some(DiskIoStats {
        devices,
        // iostat doesn't split read/write — total_read_mbps and total_write_mbps
        // are approximations. We report total as read (conservative) and 0 as write.
        // For precise read/write split, we'd need `iostat -d -x` which isn't on macOS.
        total_read_mbps: total_mbps,
        total_write_mbps: 0.0,
    })
}

// ---------------------------------------------------------------------------
// Network throughput parsing (from netstat -ib)
// ---------------------------------------------------------------------------

/// Raw byte counters from a single `netstat -ib` sample.
#[derive(Debug, Clone)]
pub struct NetstatSample {
    /// Interface name → (ibytes, obytes)
    pub counters: std::collections::HashMap<String, (u64, u64)>,
}

/// Parse `netstat -ib` output into byte counters per interface.
///
/// macOS format:
/// ```text
/// Name  Mtu   Network       Address            Ipkts Ierrs     Ibytes    Opkts Oerrs     Obytes  Coll
/// en3   1500  <Link#13>     0a:e0:af:d0:79:f4  13802     0    9271342     7685     0    1145788     0
/// ```
///
/// Only includes `en*` interfaces (Thunderbolt/Ethernet bridges).
pub fn parse_netstat_ib(text: &str) -> NetstatSample {
    let mut counters = std::collections::HashMap::new();

    for line in text.lines().skip(1) {
        // skip header
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 11 {
            continue;
        }
        let name = parts[0];
        // Only en* interfaces (TB bridges, ethernet)
        if !name.starts_with("en") {
            continue;
        }
        let ibytes: u64 = parts[6].parse().unwrap_or(0);
        let obytes: u64 = parts[9].parse().unwrap_or(0);
        if ibytes > 0 || obytes > 0 {
            // netstat can list an interface multiple times (link + inet).
            // Use the entry with higher byte counts (link-level row).
            let entry = counters.entry(name.to_string()).or_insert((0u64, 0u64));
            if ibytes > entry.0 {
                *entry = (ibytes, obytes);
            }
        }
    }

    NetstatSample { counters }
}

/// Compute network throughput by diffing two netstat samples taken `interval_secs` apart.
pub fn diff_netstat_samples(
    prev: &NetstatSample,
    curr: &NetstatSample,
    interval_secs: f64,
) -> NetworkStats {
    let mut interfaces = Vec::new();
    let mut total_rx: f64 = 0.0;
    let mut total_tx: f64 = 0.0;

    for (name, &(curr_rx, curr_tx)) in &curr.counters {
        if let Some(&(prev_rx, prev_tx)) = prev.counters.get(name) {
            // Handle counter wraparound (unlikely but safe)
            let delta_rx = curr_rx.saturating_sub(prev_rx);
            let delta_tx = curr_tx.saturating_sub(prev_tx);

            if interval_secs > 0.0 {
                let rx_bps = delta_rx as f64 / interval_secs;
                let tx_bps = delta_tx as f64 / interval_secs;
                let rx_mbps = (rx_bps * 8.0) / 1_000_000.0;
                let tx_mbps = (tx_bps * 8.0) / 1_000_000.0;

                // Only include interfaces with traffic
                if delta_rx > 0 || delta_tx > 0 {
                    total_rx += rx_mbps;
                    total_tx += tx_mbps;
                    interfaces.push(InterfaceStats {
                        name: name.clone(),
                        rx_bytes_sec: (rx_bps as u64),
                        tx_bytes_sec: (tx_bps as u64),
                        rx_mbps,
                        tx_mbps,
                    });
                }
            }
        }
    }

    // Sort by name for deterministic output
    interfaces.sort_by(|a, b| a.name.cmp(&b.name));

    NetworkStats {
        interfaces,
        total_rx_mbps: total_rx,
        total_tx_mbps: total_tx,
    }
}

// ---------------------------------------------------------------------------
// Process tree parsing (on-demand)
// ---------------------------------------------------------------------------

/// Parse `ps -axo pid,ppid,pcpu,pmem,rss,comm` and build a process tree.
///
/// Query params control filtering:
/// - `min_cpu`: only include processes with >N% CPU
/// - `min_mem`: only include processes with >N% memory
///
/// Returns root-level processes with children nested.
pub fn parse_process_tree(text: &str, min_cpu: f64, min_mem: f64) -> Vec<ProcessTreeNode> {
    let mut all: Vec<ProcessTreeNode> = Vec::new();

    for line in text.lines().skip(1) {
        // skip header
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }
        let pid: u32 = match parts[0].parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let ppid: u32 = parts[1].parse().unwrap_or(0);
        let cpu: f64 = parts[2].parse().unwrap_or(0.0);
        let mem: f64 = parts[3].parse().unwrap_or(0.0);
        let rss_kb: u64 = parts[4].parse().unwrap_or(0);
        let name = parts[5..].join(" ");

        all.push(ProcessTreeNode {
            pid,
            ppid,
            name,
            cpu_percent: cpu,
            mem_percent: mem,
            rss_bytes: rss_kb * 1024,
            children: Vec::new(),
        });
    }

    // Filter by thresholds (keep parent references even if below threshold)
    let significant_pids: std::collections::HashSet<u32> = all
        .iter()
        .filter(|p| p.cpu_percent >= min_cpu || p.mem_percent >= min_mem)
        .map(|p| p.pid)
        .collect();

    // Also keep parents of significant processes
    let mut keep_pids = significant_pids.clone();
    for p in &all {
        if significant_pids.contains(&p.pid) {
            keep_pids.insert(p.ppid);
        }
    }

    let filtered: Vec<ProcessTreeNode> = all
        .into_iter()
        .filter(|p| keep_pids.contains(&p.pid))
        .collect();

    // Build tree: group children under parents
    let pid_set: std::collections::HashSet<u32> = filtered.iter().map(|p| p.pid).collect();
    let mut by_parent: std::collections::HashMap<u32, Vec<ProcessTreeNode>> = std::collections::HashMap::new();
    let mut roots = Vec::new();

    for p in filtered {
        if !pid_set.contains(&p.ppid) {
            // Parent not in our filtered set → this is a root
            roots.push(p);
        } else {
            by_parent.entry(p.ppid).or_default().push(p);
        }
    }

    // Recursively attach children
    fn attach_children(
        node: &mut ProcessTreeNode,
        by_parent: &mut std::collections::HashMap<u32, Vec<ProcessTreeNode>>,
    ) {
        if let Some(mut children) = by_parent.remove(&node.pid) {
            for child in &mut children {
                attach_children(child, by_parent);
            }
            children.sort_by(|a, b| {
                b.cpu_percent
                    .partial_cmp(&a.cpu_percent)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            node.children = children;
        }
    }

    for root in &mut roots {
        attach_children(root, &mut by_parent);
    }

    // Sort roots by CPU descending
    roots.sort_by(|a, b| {
        b.cpu_percent
            .partial_cmp(&a.cpu_percent)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    roots
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the value following a `--flag` in a command string.
///
/// E.g., `extract_flag_value("--model /path/to/model --port 8003", "--model")`
/// returns `Some("/path/to/model")`.
fn extract_flag_value(command: &str, flag: &str) -> Option<String> {
    let parts: Vec<&str> = command.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == flag {
            return parts.get(i + 1).map(|s| s.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_powermetrics_text() {
        let text = include_str!("../testdata/powermetrics-text.txt");
        let result = parse_powermetrics_text(text);

        // Power values (first occurrence of each in the summary block)
        assert!((result.cpu_mw - 8916.0).abs() < 0.1, "cpu_mw: {}", result.cpu_mw);
        assert!((result.gpu_mw - 9462.0).abs() < 0.1, "gpu_mw: {}", result.gpu_mw);
        assert!((result.ane_mw - 0.0).abs() < 0.1, "ane_mw: {}", result.ane_mw);

        // GPU active residency should be 100%
        assert!(
            (result.gpu_percent - 100.0).abs() < 0.01,
            "gpu_percent: {}",
            result.gpu_percent
        );

        // CPU active residency — average of 32 cores
        // Sanity: mostly idle P-cores + active E-cores -> should be roughly 15-20%
        assert!(
            result.cpu_percent > 5.0 && result.cpu_percent < 50.0,
            "cpu_percent out of expected range: {}",
            result.cpu_percent
        );

        eprintln!(
            "Parsed: cpu_mw={}, gpu_mw={}, ane_mw={}, cpu%={:.2}, gpu%={:.2}",
            result.cpu_mw, result.gpu_mw, result.ane_mw, result.cpu_percent, result.gpu_percent
        );
    }

    #[test]
    fn test_parse_powermetrics_gpu_from_idle() {
        // If GPU HW active residency is missing, derive from idle
        let text = "GPU idle residency:  25.00%\n";
        let result = parse_powermetrics_text(text);
        assert!((result.gpu_percent - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_vmstat_and_memsize() {
        let vmstat = include_str!("../testdata/vmstat.txt");
        let sysctl = include_str!("../testdata/sysctl-hw.txt");
        let combined = format!("{vmstat}\n---MEMSIZE---\n{sysctl}");

        let stats = parse_vmstat_and_memsize(&combined);

        // Total should be 549755813888 (512 GiB)
        assert_eq!(stats.total_bytes, 549_755_813_888, "total_bytes");

        // App = anonymous + wired + compressor = 18962475 + 1434855 + 422 = 20397752 pages
        let expected_app_pages: u64 = 18_962_475 + 1_434_855 + 422;
        let expected_app = expected_app_pages * 16384;
        assert_eq!(stats.app_bytes, expected_app, "app_bytes");

        // Cached = file-backed + speculative = 3516901 + 133882 = 3650783 pages
        let expected_cached_pages: u64 = 3_516_901 + 133_882;
        let expected_cached = expected_cached_pages * 16384;
        assert_eq!(stats.cached_bytes, expected_cached, "cached_bytes");

        // Used = app + cached (backward compat)
        assert_eq!(stats.used_bytes, expected_app + expected_cached, "used_bytes = app + cached");

        // Sanity: app should be less than total
        assert!(stats.app_bytes < stats.total_bytes, "app_bytes should be < total");

        eprintln!(
            "Parsed: app={:.1} GiB, cached={:.1} GiB, used={:.1} GiB, total={:.1} GiB",
            stats.app_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.cached_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }

    #[test]
    fn test_parse_vmstat_custom_page_size() {
        let text = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
                     Pages active:                               100.\n\
                     Anonymous pages:                             80.\n\
                     File-backed pages:                           20.\n\
                     ---MEMSIZE---\n\
                     8589934592\n";
        let stats = parse_vmstat_and_memsize(text);
        assert_eq!(stats.total_bytes, 8_589_934_592);
        // anonymous=80 + wired=0 + compressor=0 = 80 pages app
        assert_eq!(stats.app_bytes, 80 * 4096);
        // file-backed=20 + speculative=0 = 20 pages cached
        assert_eq!(stats.cached_bytes, 20 * 4096);
        assert_eq!(stats.used_bytes, 100 * 4096);
    }

    #[test]
    fn test_parse_ps_mlx() {
        let text = include_str!("../testdata/ps-mlx.txt");
        let procs = parse_ps_mlx(text);

        // Should find 1 MLX server, filtering out watchdogs and Teams
        assert_eq!(procs.len(), 1, "expected 1 MLX process, got {}", procs.len());

        let p = &procs[0];
        assert_eq!(p.pid, 62283);
        assert_eq!(p.framework, ProcessFramework::MlxLm);
        assert_eq!(p.model.as_deref(), Some("ExampleModel-8bit"));
        assert_eq!(p.port, Some(8003));
        assert!((p.cpu_percent - 0.0).abs() < 0.1);
        assert!((p.mem_percent - 37.8).abs() < 0.1);
        assert_eq!(p.distributed, None);

        eprintln!("Parsed process: {p:?}");
    }

    #[test]
    fn test_parse_ps_filters_watchdogs() {
        let text = include_str!("../testdata/ps-mlx.txt");
        let procs = parse_ps_mlx(text);

        // Verify none of the filtered processes made it through
        for p in &procs {
            assert_ne!(p.framework, ProcessFramework::Watchdog);
        }
    }

    #[test]
    fn test_parse_footprint_gb() {
        let text = "Python [62283]: 64-bit    Footprint: 199.2 GB (16384 bytes per page)";
        let mb = parse_footprint(text);
        assert!(mb.is_some(), "should parse footprint");
        let mb = mb.unwrap();
        // 199.2 GB = 199.2 * 1024 = 203980.8 MB
        assert!((mb - 203_980.8).abs() < 0.1, "footprint MB: {mb}");
    }

    #[test]
    fn test_parse_footprint_kb() {
        let text = include_str!("../testdata/footprint-sample.txt");
        let mb = parse_footprint(text);
        assert!(mb.is_some(), "should parse footprint");
        let mb = mb.unwrap();
        // 2272 KB = 2272 / 1024 = 2.21875 MB
        assert!((mb - 2.21875).abs() < 0.01, "footprint MB: {mb}");
    }

    #[test]
    fn test_parse_footprint_mb() {
        let text = "node [1234]: 64-bit    Footprint: 512.5 MB (16384 bytes per page)";
        let mb = parse_footprint(text);
        assert_eq!(mb, Some(512.5));
    }

    #[test]
    fn test_parse_footprint_no_match() {
        let text = "no footprint data here";
        assert!(parse_footprint(text).is_none());
    }

    #[test]
    fn test_extract_flag_value() {
        let cmd = "--model /path/to/model --port 8003 --host 0.0.0.0";
        assert_eq!(
            extract_flag_value(cmd, "--model"),
            Some("/path/to/model".to_string())
        );
        assert_eq!(
            extract_flag_value(cmd, "--port"),
            Some("8003".to_string())
        );
        assert_eq!(
            extract_flag_value(cmd, "--host"),
            Some("0.0.0.0".to_string())
        );
        assert_eq!(extract_flag_value(cmd, "--missing"), None);
    }

    #[test]
    fn test_parse_ps_vlm() {
        let line = "ma  99999  12.3  45.6 100000 200000  ??  S  1:00AM  0:01.00 python -m mlx_vlm.server --model /models/llava-1.5 --port 8010 --host 0.0.0.0";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::MlxVlm);
        assert_eq!(procs[0].port, Some(8010));
    }

    #[test]
    fn test_parse_ps_vllm_mlx() {
        let line = "ma  88888  5.0  20.0 100000 200000  ??  S  2:00AM  0:05.00 python -m vllm_mlx --model /models/qwen2 --port 8020";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::VllmMlx);
    }

    #[test]
    fn test_parse_ps_mlx_share() {
        let line = "ma  77777  8.0  30.0 100000 200000  ??  S  3:00AM  0:10.00 python -m mlx_lm.share --model /models/Kimi-K2.5 --port 8080";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::MlxLmShare);
        assert_eq!(procs[0].model.as_deref(), Some("Kimi-K2.5"));
        assert_eq!(procs[0].port, Some(8080));
        assert_eq!(procs[0].distributed, None);
    }

    #[test]
    fn test_parse_ps_mlx_launch_jaccl() {
        let line = "ma  66666  2.0  1.0 100000 200000  ??  S  4:00AM  0:00.50 mlx.launch --backend jaccl --hostfile ~/hostfile.json -- python -m mlx_lm.share --model /models/Kimi-K2.5";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::MlxLaunch);
        assert_eq!(procs[0].distributed, Some(DistributedBackend::Jaccl));
    }

    #[test]
    fn test_parse_ps_mlx_launch_ring() {
        let line = "ma  55555  1.0  0.5 100000 200000  ??  S  5:00AM  0:00.10 mlx.launch --backend ring --hostfile ~/hostfile-ring.json -- python -m mlx_lm.server --model /models/Qwen3-8B";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::MlxLaunch);
        assert_eq!(procs[0].distributed, Some(DistributedBackend::Ring));
    }

    #[test]
    fn test_parse_ps_mlx_audio() {
        let line = "ma  44444  15.0  10.0 100000 200000  ??  S  6:00AM  0:20.00 python -m mlx_audio.server --model /models/kokoro-tts --port 8030";
        let procs = parse_ps_mlx(line);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].framework, ProcessFramework::MlxAudio);
        assert_eq!(procs[0].model.as_deref(), Some("kokoro-tts"));
        assert_eq!(procs[0].port, Some(8030));
    }

    #[test]
    fn test_parse_cpu_clusters() {
        let text = include_str!("../testdata/powermetrics-text.txt");
        let clusters = parse_cpu_clusters(text);

        // M3 Ultra has 6 clusters: E0, P0, P1, E1, P2, P3
        assert_eq!(clusters.len(), 6, "expected 6 clusters, got {}", clusters.len());

        // E0: 4 efficiency cores (CPU 0-3)
        let e0 = &clusters[0];
        assert_eq!(e0.name, "E0");
        assert_eq!(e0.cluster_type, ClusterType::Efficiency);
        assert_eq!(e0.core_count, 4);
        assert_eq!(e0.frequency_mhz, 2181);
        assert!((e0.active_residency - 90.12).abs() < 0.01);
        assert_eq!(e0.cores[0].id, 0);
        assert_eq!(e0.cores[0].frequency_mhz, 2244);
        assert!((e0.cores[0].active_residency - 67.95).abs() < 0.01);

        // P0: 6 performance cores (CPU 4-9)
        let p0 = &clusters[1];
        assert_eq!(p0.name, "P0");
        assert_eq!(p0.cluster_type, ClusterType::Performance);
        assert_eq!(p0.core_count, 6);

        // P2: 6 cores (CPU 20-25)
        let p2 = &clusters[4];
        assert_eq!(p2.name, "P2");
        assert_eq!(p2.core_count, 6);
        assert_eq!(p2.frequency_mhz, 3598);
        assert!((p2.active_residency - 36.48).abs() < 0.01);

        eprintln!(
            "Parsed {} clusters: {}",
            clusters.len(),
            clusters
                .iter()
                .map(|c| format!("{}({}c, {}MHz, {:.1}%)", c.name, c.core_count, c.frequency_mhz, c.active_residency))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[test]
    fn test_parse_powermetrics_includes_clusters_and_gpu_freq() {
        let text = include_str!("../testdata/powermetrics-text.txt");
        let result = parse_powermetrics_text(text);

        // Should have cluster data
        assert!(!result.cpu_clusters.is_empty(), "cpu_clusters should not be empty");
        assert_eq!(result.cpu_clusters.len(), 6);

        // GPU frequency should be extracted
        assert_eq!(result.gpu_frequency_mhz, Some(1380));
    }

    #[test]
    fn test_parse_iostat() {
        let text = include_str!("../testdata/iostat.txt");
        let stats = parse_iostat(text);

        assert!(stats.is_some(), "should parse iostat output");
        let stats = stats.unwrap();

        assert_eq!(stats.devices.len(), 2);
        assert_eq!(stats.devices[0].name, "disk0");
        assert!((stats.devices[0].kb_per_transfer - 8.0).abs() < 0.01);
        assert!((stats.devices[0].transfers_per_sec - 5.0).abs() < 0.01);
        assert!((stats.devices[0].mb_per_sec - 0.04).abs() < 0.01);

        assert_eq!(stats.devices[1].name, "disk3");
        assert!((stats.devices[1].mb_per_sec - 1.88).abs() < 0.01);

        // Total = 0.04 + 1.88 = 1.92
        assert!((stats.total_read_mbps - 1.92).abs() < 0.01);

        eprintln!("Parsed iostat: {:?}", stats);
    }

    #[test]
    fn test_parse_iostat_empty() {
        assert!(parse_iostat("").is_none());
        assert!(parse_iostat("no data\n").is_none());
    }

    #[test]
    fn test_parse_netstat_ib() {
        let text = include_str!("../testdata/netstat-ib.txt");
        let sample = parse_netstat_ib(text);

        // Should have en0, en3, en5 (lo0 is skipped — not en*)
        assert!(sample.counters.contains_key("en0"));
        assert!(sample.counters.contains_key("en3"));
        assert!(sample.counters.contains_key("en5"));
        assert!(!sample.counters.contains_key("lo0"));

        // en3 should use the link-level row (higher byte count: 9271342)
        let (en3_rx, en3_tx) = sample.counters["en3"];
        assert_eq!(en3_rx, 9271342);
        assert_eq!(en3_tx, 1145788);

        eprintln!("Parsed netstat: {:?}", sample.counters);
    }

    #[test]
    fn test_diff_netstat_samples() {
        let mut prev = NetstatSample { counters: std::collections::HashMap::new() };
        prev.counters.insert("en3".to_string(), (1_000_000, 500_000));
        prev.counters.insert("en5".to_string(), (2_000_000, 1_000_000));

        let mut curr = NetstatSample { counters: std::collections::HashMap::new() };
        curr.counters.insert("en3".to_string(), (2_000_000, 600_000));
        curr.counters.insert("en5".to_string(), (3_000_000, 1_500_000));

        let stats = diff_netstat_samples(&prev, &curr, 1.0);

        assert_eq!(stats.interfaces.len(), 2);

        // en3: delta_rx=1MB, delta_tx=100KB in 1s
        let en3 = stats.interfaces.iter().find(|i| i.name == "en3").unwrap();
        assert_eq!(en3.rx_bytes_sec, 1_000_000);
        assert_eq!(en3.tx_bytes_sec, 100_000);
        // 1MB/s = 8 Mbps
        assert!((en3.rx_mbps - 8.0).abs() < 0.01);

        eprintln!("Network stats: total_rx={:.1} Mbps, total_tx={:.1} Mbps", stats.total_rx_mbps, stats.total_tx_mbps);
    }

    #[test]
    fn test_parse_process_tree() {
        let text = "  PID  PPID  %CPU %MEM      RSS COMM\n\
                     1     0   0.0  0.1    32768 /sbin/launchd\n\
                     100   1   5.0  2.0   204800 /usr/sbin/some_daemon\n\
                     200   1   0.5  0.3    65536 /usr/libexec/other\n\
                     300  100  15.0  8.0  1048576 python3 -m mlx_lm.server\n\
                     400  100   2.0  1.0   131072 python3 worker\n";

        let tree = parse_process_tree(text, 1.0, 0.5);

        // Should include PID 1 (parent of significant procs), 100, 300, 400
        // PID 200 has 0.5% CPU and 0.3% mem → below both thresholds but
        // it may be included as a parent (it's not a parent of anything significant)
        assert!(!tree.is_empty());

        // PID 300 (15% CPU) should be in the tree under PID 100
        let has_mlx = tree.iter().any(|n| {
            n.pid == 300 || n.children.iter().any(|c| c.pid == 300)
                || n.children.iter().any(|c| c.children.iter().any(|gc| gc.pid == 300))
        });
        assert!(has_mlx, "should include the MLX server process");

        eprintln!("Process tree roots: {:?}", tree.iter().map(|n| (n.pid, &n.name, n.children.len())).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_collect_node_metrics_local_bypasses_http() {
        // When is_local=true, collect_node_metrics should NEVER call
        // fetch_from_daemon(). It should go directly to collect_via_ssh()
        // with local command execution.
        //
        // We can't easily mock the HTTP call, but we CAN verify that:
        // 1. The function returns a snapshot (not an HTTP error)
        // 2. The hostname in the snapshot is set (from `hostname -s`)
        // 3. ram_total_bytes > 0 (from sysctl, proving local exec ran)
        //
        // If this were hitting an HTTP endpoint that doesn't exist on the
        // test host, ram_total_bytes would be 0.
        let config = crate::config::ClusterConfig::default();
        let snap = super::collect_node_metrics("localhost", &config, true).await;

        assert!(snap.online, "local snapshot should be online");
        assert!(
            snap.ram_total_bytes > 0,
            "ram_total_bytes should be > 0 (proves local exec ran, not HTTP fetch). Got: {}",
            snap.ram_total_bytes
        );
        assert!(
            !snap.hostname.is_empty(),
            "hostname should be resolved from `hostname -s`"
        );
    }

    #[tokio::test]
    async fn test_collect_node_metrics_remote_tries_http_first() {
        // When is_local=false and the daemon is unreachable, the function
        // should fall back to SSH. With a bogus hostname, both HTTP and SSH
        // will fail, resulting in an offline-looking snapshot.
        //
        // This test verifies the function doesn't panic and handles
        // unreachable remotes gracefully.
        let config = crate::config::ClusterConfig {
            daemon_port: 19999, // deliberately wrong port
            ssh_timeout: std::time::Duration::from_secs(1),
            ..Default::default()
        };
        let snap = super::collect_node_metrics("nonexistent-host-12345", &config, false).await;

        // Should return a snapshot (not panic), but with zero data and
        // online=false because both HTTP and SSH failed
        assert!(!snap.online, "unreachable remote should be marked offline");
        assert_eq!(snap.ram_total_bytes, 0, "unreachable host should have 0 RAM");
    }
}
