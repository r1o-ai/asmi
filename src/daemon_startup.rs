use anyhow::Result;
use asmi_core::ClusterConfig;
use std::sync::Arc;
use std::time::Duration;

use crate::{bin_name, daemon, serve};

/// Collect hardware identity from `system_profiler SPHardwareDataType`.
/// Returns (chip_model, serial_number, model_name). Runs synchronously (once at startup).
fn collect_hardware_identity() -> (Option<String>, Option<String>, Option<String>) {
    let output = std::process::Command::new("system_profiler")
        .arg("SPHardwareDataType")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let mut chip_model = None;
    let mut serial_number = None;
    let mut model_name = None;

    for line in output.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("Chip:") {
            chip_model = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Serial Number (system):") {
            serial_number = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Model Name:") {
            model_name = Some(v.trim().to_string());
        }
    }

    (chip_model, serial_number, model_name)
}

/// Run asmi as an HTTP daemon serving local node metrics.
pub async fn run_serve(port: u16, interval: u64, cluster_hub: bool, cli_models_dir: Vec<String>) -> Result<()> {
    // Init tracing to stderr
    tracing_subscriber::fmt()
        .with_env_filter("asmi_core=info,asmi=info")
        .with_ansi(true)
        .init();

    let hostname = std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    tracing::info!(
        hostname = %hostname,
        port = port,
        interval_secs = interval,
        cluster_hub = cluster_hub,
        "{} daemon starting", bin_name()
    );

    // Collect hardware identity once at startup
    let (chip_model, serial_number, model_name) = collect_hardware_identity();
    tracing::info!(
        chip = chip_model.as_deref().unwrap_or("unknown"),
        serial = serial_number.as_deref().unwrap_or("unknown"),
        model = model_name.as_deref().unwrap_or("unknown"),
        "hardware identity"
    );

    // Config for local-only collection (no SSH, no discovery)
    let config = ClusterConfig::default()
        .with_poll_interval(Duration::from_secs(interval));

    // Shared state: latest snapshot from this node
    let snapshot: Arc<tokio::sync::RwLock<Option<asmi_core::NodeSnapshot>>> =
        Arc::new(tokio::sync::RwLock::new(None));
    let started_at = std::time::Instant::now();

    // Broadcast channel for SSE streaming
    let (metrics_tx, _) = tokio::sync::broadcast::channel::<String>(16);

    let hw_chip = chip_model.clone();
    let hw_serial = serial_number.clone();
    let hw_model = model_name.clone();

    // Background polling loop — collect local metrics every N seconds
    {
        let snapshot = Arc::clone(&snapshot);
        let config = config.clone();
        let hostname = hostname.clone();
        let hw_chip = hw_chip.clone();
        let hw_serial = hw_serial.clone();
        let hw_model = hw_model.clone();
        let metrics_tx = metrics_tx.clone();
        tokio::spawn(async move {
            loop {
                let mut snap = asmi_core::collect_node_metrics(&hostname, &config, true).await;
                snap.chip_model = hw_chip.clone();
                snap.serial_number = hw_serial.clone();
                snap.model_name = hw_model.clone();
                tracing::debug!(
                    cpu = format!("{:.1}%", snap.cpu_percent),
                    gpu = format!("{:.1}%", snap.gpu_percent),
                    ram = format!("{:.1}/{:.1} GiB", snap.ram_used_gib(), snap.ram_total_gib()),
                    procs = snap.processes.len(),
                    "metrics collected"
                );
                if let Ok(json) = serde_json::to_string(&snap) {
                    let _ = metrics_tx.send(json);
                }
                *snapshot.write().await = Some(snap);
                tokio::time::sleep(Duration::from_secs(interval)).await;
            }
        });
    }

    // Optional cluster hub
    let cluster_state: Option<Arc<tokio::sync::RwLock<asmi_core::ClusterState>>> = if cluster_hub {
        let node_map = asmi_core::NodeMap::load();
        if node_map.nodes.is_empty() {
            tracing::warn!("--cluster requested but NodeMap is empty; run `asmi` first to discover nodes");
            None
        } else {
            let remote_nodes: Vec<String> = node_map.nodes.iter()
                .filter(|n| n.as_str() != hostname.as_str())
                .cloned()
                .collect();
            tracing::info!(
                all_nodes = ?node_map.nodes,
                remote_nodes = ?remote_nodes,
                "cluster hub: polling {} remote nodes (excluded self: {})",
                remote_nodes.len(), hostname
            );
            let cfg = ClusterConfig::default()
                .with_seeds(remote_nodes)
                .with_poll_interval(Duration::from_secs(interval));
            let mut monitor = asmi_core::ClusterMonitor::new(cfg, node_map);
            let state = monitor.state();
            monitor.start();
            std::mem::forget(monitor);
            Some(state)
        }
    } else {
        None
    };

    // Probe runtime versions once at startup
    let runtime = Arc::new(daemon::probe_runtime().await);
    tracing::info!(
        python = runtime.python_version.as_deref().unwrap_or("none"),
        mlx = runtime.mlx_version.as_deref().unwrap_or("none"),
        "runtime probed"
    );

    // Model cache — populated by background scan loop
    let model_cache: Arc<tokio::sync::RwLock<Option<(Vec<asmi_core::LocalModel>, std::time::Instant)>>> =
        Arc::new(tokio::sync::RwLock::new(None));

    let model_dirs: Vec<std::path::PathBuf> = if cli_models_dir.is_empty() {
        asmi_core::default_model_dirs()
    } else {
        cli_models_dir.iter().map(std::path::PathBuf::from).collect()
    };

    // Background model scan — refresh every 60s
    {
        let model_cache = Arc::clone(&model_cache);
        let dirs = model_dirs.clone();
        tokio::spawn(async move {
            loop {
                let models = asmi_core::scan_models(&dirs);
                tracing::info!(count = models.len(), "model scan complete");
                *model_cache.write().await = Some((models, std::time::Instant::now()));
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
    }

    // Thunderbolt cache — populated by background scan loop (60s)
    let thunderbolt_cache: Arc<tokio::sync::RwLock<Option<(serde_json::Value, std::time::Instant)>>> =
        Arc::new(tokio::sync::RwLock::new(None));
    {
        let tb_cache = Arc::clone(&thunderbolt_cache);
        let tb_hostname = hostname.clone();
        tokio::spawn(async move {
            loop {
                let data = daemon::scan_thunderbolt(&tb_hostname).await;
                *tb_cache.write().await = Some((data, std::time::Instant::now()));
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
    }

    // Build axum router — init all managed MLX servers in parallel
    let managers: Vec<_> = futures::future::join_all(
        serve::MANAGED_PORTS.iter().map(|&(port, engine)| {
            serve::ServeManager::restore(port, engine)
        })
    ).await;

    let mut serve_managers = std::collections::HashMap::new();
    for (i, mgr) in managers.into_iter().enumerate() {
        serve_managers.insert(serve::MANAGED_PORTS[i].0, mgr);
    }

    let share_manager = serve::ShareManager::restore().await;

    let app_state = daemon::AppState {
        snapshot,
        cluster_state,
        node_map: Arc::new(tokio::sync::RwLock::new(asmi_core::NodeMap::load())),
        hostname: hostname.clone(),
        started_at,
        metrics_tx: metrics_tx.clone(),
        model_cache,
        thunderbolt_cache,
        runtime,
        serve_managers: Arc::new(serve_managers),
        share_manager,
    };

    let app = daemon::build_router(app_state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!(%addr, "HTTP server listening");
    let base = format!("http://{hostname}:{port}");
    let name = bin_name();
    if cluster_hub {
        eprintln!("{name} cluster hub: {base}");
    } else {
        eprintln!("{name} daemon: {base}");
    }
    eprintln!("  GET  /metrics          Node snapshot (CPU, GPU, RAM, power)");
    eprintln!("  GET  /health           Daemon liveness + uptime");
    eprintln!("  GET  /health/setup     Setup validation (MLX, RDMA, SSH, disk)");
    eprintln!("  GET  /health/network   Thunderbolt service validation");
    eprintln!("  POST /health/network/fix  Auto-repair TB service names");
    eprintln!("  GET  /processes        Running MLX/VLM processes");
    eprintln!("  GET  /models           Local model inventory");
    eprintln!("  GET  /runtime          Python/MLX/macOS versions");
    eprintln!("  GET  /logs?name=asmi   Tail log files");
    eprintln!("  GET  /stream           SSE live metrics stream");
    eprintln!("  GET  /serve/status     All server states (or ?port=N)");
    eprintln!("  POST /serve/load       Load model (?port=N)");
    eprintln!("  POST /serve/stop       Stop server (?port=N)");
    eprintln!("  POST /serve/reload     Reload model (?port=N)");
    eprintln!("  POST /serve/share      Start distributed share session");
    eprintln!("  GET  /serve/share/status Share session status");
    eprintln!("  POST /serve/share/stop  Stop share session");
    let ports_str: Vec<String> = serve::MANAGED_PORTS.iter().map(|(p, e)| format!("{p}({e})")).collect();
    eprintln!("  Managed ports: {}", ports_str.join(", "));
    if cluster_hub {
        eprintln!("  GET  /cluster          All node snapshots (hub mode)");
        eprintln!("  GET  /nodes            Known node hostnames");
        eprintln!("  GET  /jaccl/config     RDMA topology for JACCL");
        eprintln!("  POST /jaccl/config     Generate JACCL hostfile");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
