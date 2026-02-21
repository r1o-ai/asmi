//! Metrics collector — runs parallel SSH commands per node, parses output,
//! and enriches with footprint data.
//!
//! This mirrors the proven TypeScript pattern from the web app:
//! 3 parallel SSH commands (powermetrics, vm_stat+sysctl, ps aux), then
//! a sequential footprint lookup for each discovered MLX process.

use crate::config::ClusterConfig;
use crate::ssh::{local_run, ssh_run};
use crate::types::*;
use chrono::Utc;
use regex::Regex;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

const CMD_POWERMETRICS: &str =
    "sudo powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power 2>/dev/null";

const CMD_VMSTAT_SYSCTL: &str = "vm_stat; echo '---MEMSIZE---'; sysctl -n hw.memsize";

/// Also captures mlx.launch (distributed launcher) and mlx_lm.share.
/// JACCL detection: --backend jaccl in args, or ps -E showing MLX_JACCL env vars.
const CMD_PS_MLX: &str =
    "ps aux | grep -E 'mlx_lm\\.(server|share)|mlx_vlm\\.server|vllm_mlx|mlx\\.launch' | grep -v grep";

/// Detect JACCL/distributed by checking `mlx.launch` wrapper processes.
/// Only `mlx.launch --backend jaccl` is a reliable signal — env vars
/// can persist from previous runs and cause false positives on single-node.
const CMD_JACCL_ENV: &str =
    "ps aux | grep 'mlx\\.launch.*--backend' | grep -v grep | head -5";

/// Build the footprint command for a given PID.
fn footprint_cmd(pid: u32) -> String {
    format!("sudo footprint -p {pid} 2>/dev/null | head -2 | tail -1")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Collect a full metrics snapshot from a node.
///
/// Runs powermetrics + vm_stat/sysctl + ps aux in parallel via `tokio::join!`,
/// then enriches each discovered MLX process with a footprint lookup.
pub async fn collect_node_metrics(
    hostname: &str,
    config: &ClusterConfig,
    is_local: bool,
) -> NodeSnapshot {
    let run = |cmd: &'static str| async move {
        if is_local {
            local_run(cmd).await
        } else {
            ssh_run(hostname, cmd, config).await
        }
    };

    // 1. Run 4 commands in parallel
    let (power_res, mem_res, ps_res, jaccl_res) = tokio::join!(
        run(CMD_POWERMETRICS),
        run(CMD_VMSTAT_SYSCTL),
        run(CMD_PS_MLX),
        run(CMD_JACCL_ENV),
    );

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

    // 3. Parse vm_stat + sysctl
    let (ram_used_bytes, ram_total_bytes) = match &mem_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "vm_stat/sysctl OK");
            parse_vmstat_and_memsize(&r.stdout)
        }
        Ok(r) => {
            debug!(hostname, stderr = r.stderr.as_str(), "vm_stat/sysctl empty/failed");
            (0, 0)
        }
        Err(e) => {
            warn!(hostname, error = %e, "vm_stat/sysctl command error");
            (0, 0)
        }
    };

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

    // 7. Enrich each process with footprint data (sequential — one per process)
    for proc in &mut processes {
        let cmd = footprint_cmd(proc.pid);
        let fp_res = if is_local {
            local_run(&cmd).await
        } else {
            ssh_run(hostname, &cmd, config).await
        };

        match fp_res {
            Ok(r) if r.has_output() => {
                proc.footprint_mb = parse_footprint(&r.stdout);
                debug!(hostname, pid = proc.pid, footprint_mb = ?proc.footprint_mb, "footprint");
            }
            _ => {
                debug!(hostname, pid = proc.pid, "footprint unavailable");
            }
        }
    }

    // 8. Assemble NodeSnapshot
    let ram_percent = if ram_total_bytes > 0 {
        (ram_used_bytes as f64 / ram_total_bytes as f64) * 100.0
    } else {
        0.0
    };

    NodeSnapshot {
        hostname: hostname.to_string(),
        online: true,
        timestamp: Utc::now(),
        cpu_watts: power.cpu_mw,
        gpu_watts: power.gpu_mw,
        ane_watts: power.ane_mw,
        cpu_percent: power.cpu_percent,
        gpu_percent: power.gpu_percent,
        ram_used_bytes,
        ram_total_bytes,
        ram_percent,
        cpu_temp_c: None,
        gpu_temp_c: None,
        processes,
        top_tasks: Vec::new(),
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
    let power_re = Regex::new(r"(?m)^(CPU|GPU|ANE) Power:\s+([\d.]+)\s+mW").unwrap();
    let mut seen_cpu = false;
    let mut seen_gpu = false;
    let mut seen_ane = false;
    for cap in power_re.captures_iter(text) {
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
    let gpu_re = Regex::new(r"GPU HW active residency:\s+([\d.]+)%").unwrap();
    if let Some(cap) = gpu_re.captures(text) {
        result.gpu_percent = cap[1].parse().unwrap_or(0.0);
    } else {
        // Fallback: derive from GPU idle residency
        let idle_re = Regex::new(r"GPU idle residency:\s+([\d.]+)%").unwrap();
        if let Some(cap) = idle_re.captures(text) {
            let idle: f64 = cap[1].parse().unwrap_or(0.0);
            result.gpu_percent = 100.0 - idle;
        }
    }

    // CPU active residency: "CPU N active residency:  X.XX% (...)"
    // Average across all CPU cores
    let cpu_re = Regex::new(r"(?m)^CPU \d+ active residency:\s+([\d.]+)%").unwrap();
    let mut sum = 0.0;
    let mut count = 0u32;
    for cap in cpu_re.captures_iter(text) {
        sum += cap[1].parse::<f64>().unwrap_or(0.0);
        count += 1;
    }
    if count > 0 {
        result.cpu_percent = sum / count as f64;
    }

    result
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
/// Returns `(used_bytes, total_bytes)`.
///
/// Used memory = (active + inactive + speculative + wired + occupied_by_compressor) * page_size
pub fn parse_vmstat_and_memsize(text: &str) -> (u64, u64) {
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
    let page_size_re = Regex::new(r"\(page size of (\d+) bytes\)").unwrap();
    let page_size: u64 = page_size_re
        .captures(vmstat_text)
        .and_then(|c| c[1].parse().ok())
        .unwrap_or(16384);

    // Parse page counts
    let page_re = Regex::new(r"(?m)^Pages (\w[\w ]*\w):\s+(\d+)\.").unwrap();
    let mut active: u64 = 0;
    let mut inactive: u64 = 0;
    let mut speculative: u64 = 0;
    let mut wired: u64 = 0;
    let mut compressor: u64 = 0;

    for cap in page_re.captures_iter(vmstat_text) {
        let name = &cap[1];
        let pages: u64 = cap[2].parse().unwrap_or(0);
        match name {
            "active" => active = pages,
            "inactive" => inactive = pages,
            "speculative" => speculative = pages,
            "wired down" => wired = pages,
            "occupied by compressor" => compressor = pages,
            _ => {}
        }
    }

    let used_bytes = (active + inactive + speculative + wired + compressor) * page_size;

    (used_bytes, total_bytes)
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
    // Regex to split ps aux lines into 11 groups:
    // user, pid, %cpu, %mem, vsz, rss, tt, stat, started, time, command
    // The first 10 fields are whitespace-delimited, the 11th (command) is the rest.
    let ps_re = Regex::new(
        r"(?m)^\s*(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+)$"
    ).unwrap();

    let mut procs = Vec::new();

    for cap in ps_re.captures_iter(text) {
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
        // contains the child command (e.g. "mlx.launch ... -- python -m mlx_lm.share")
        let framework = if command.contains("mlx.launch") {
            ProcessFramework::MlxLaunch
        } else if command.contains("mlx_lm.server") {
            ProcessFramework::MlxLm
        } else if command.contains("mlx_lm.share") {
            ProcessFramework::MlxLmShare
        } else if command.contains("mlx_vlm.server") {
            ProcessFramework::MlxVlm
        } else if command.contains("vllm_mlx") {
            ProcessFramework::VllmMlx
        } else {
            continue; // Not an MLX process we care about
        };

        let pid: u32 = cap[2].parse().unwrap_or(0);
        let cpu_percent: f64 = cap[3].parse().unwrap_or(0.0);
        let mem_percent: f64 = cap[4].parse().unwrap_or(0.0);

        // Extract model path: --model <path>
        let model = extract_flag_value(command, "--model");

        // Simplify model path to just the model name (last path component)
        let model = model.map(|m| {
            m.rsplit('/')
                .next()
                .unwrap_or(&m)
                .to_string()
        });

        // Extract port: --port <N>
        let port: Option<u16> = extract_flag_value(command, "--port")
            .and_then(|p| p.parse().ok());

        // Detect distributed backend from --backend flag or env hints
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
    let re = Regex::new(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)").unwrap();
    let cap = re.captures(text)?;
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

        let (used, total) = parse_vmstat_and_memsize(&combined);

        // Total should be 549755813888 (512 GiB)
        assert_eq!(total, 549_755_813_888, "total_bytes");

        // Used should be > 0 and < total
        assert!(used > 0, "used_bytes should be > 0, got {used}");
        assert!(used < total, "used_bytes ({used}) should be < total ({total})");

        // Sanity: active=11058419 + inactive=11287075 + speculative=133882 +
        //         wired=1434855 + compressor=422 = 23914653 pages
        // 23914653 * 16384 = 391,808,155,648 bytes
        let expected_pages: u64 = 11_058_419 + 11_287_075 + 133_882 + 1_434_855 + 422;
        let expected = expected_pages * 16384;
        assert_eq!(used, expected, "used bytes mismatch");

        eprintln!("Parsed: used={used} ({:.1} GiB), total={total} ({:.1} GiB)",
            used as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }

    #[test]
    fn test_parse_vmstat_custom_page_size() {
        let text = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
                     Pages active:                               100.\n\
                     ---MEMSIZE---\n\
                     8589934592\n";
        let (used, total) = parse_vmstat_and_memsize(text);
        assert_eq!(total, 8_589_934_592);
        assert_eq!(used, 100 * 4096);
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
        assert_eq!(p.model.as_deref(), Some("MiniMax-M2.5-REAP-19-8bit"));
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
}
