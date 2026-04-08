use anyhow::Result;
use asmi_core::ClusterConfig;
use std::sync::Arc;
use std::time::Duration;

use crate::{ane::AneState, bin_name, daemon, serve, watchdog};

/// Run asmi as an HTTP daemon serving local node metrics.
pub async fn run_serve(port: u16, interval: u64, cluster_hub: bool, cli_models_dir: Vec<String>, _experimental_ane: bool, experimental_egpu: bool) -> Result<()> {
    // Init tracing to stderr
    tracing_subscriber::fmt()
        .with_env_filter("asmi_core=info,asmi=info")
        .with_ansi(true)
        .init();

    // Prefer scutil LocalHostName (Bonjour/mDNS identity, matches .local resolution
    // and Tailscale names) over `hostname -s` (Unix hostname, often the default
    // machine name like "Mac" which doesn't match cluster identity).
    let hostname = std::process::Command::new("scutil")
        .args(["--get", "LocalHostName"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_lowercase();
            if s.is_empty() { None } else { Some(s) }
        })
        .unwrap_or_else(|| {
            std::process::Command::new("hostname")
                .arg("-s")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        });

    tracing::info!(
        hostname = %hostname,
        port = port,
        interval_secs = interval,
        cluster_hub = cluster_hub,
        "{} daemon starting", bin_name()
    );

    // Collect hardware identity once (cached in OnceLock across all callers)
    let (chip_model, serial_number, model_name) = asmi_core::local_hardware_identity();
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

    // Init all managed MLX servers (before poll loop so we can enrich metrics)
    let ports = serve::managed_ports();
    let managers: Vec<_> = futures::future::join_all(
        ports.iter().map(|&(port, engine)| {
            serve::ServeManager::restore(port, engine)
        })
    ).await;

    let mut serve_managers = std::collections::HashMap::new();
    for (i, mgr) in managers.into_iter().enumerate() {
        serve_managers.insert(ports[i].0, mgr);
    }
    let serve_managers = Arc::new(serve_managers);

    // Create IOReport energy subscription for ANE power monitoring.
    // This uses the private IOReport framework (same data source as powermetrics)
    // but doesn't require sudo. Created once, sampled every poll tick.
    let energy_sub = asmi_core::ioreport::EnergySubscription::new(interval * 1000);
    if energy_sub.is_some() {
        tracing::info!("IOReport energy subscription active (no-sudo ANE power)");
    } else {
        tracing::warn!("IOReport unavailable — ANE power from powermetrics only");
    }

    // Background polling loop — collect local metrics every N seconds
    {
        let snapshot = Arc::clone(&snapshot);
        let config = config.clone();
        let hostname = hostname.clone();
        let metrics_tx = metrics_tx.clone();
        let serve_managers = Arc::clone(&serve_managers);
        // Wrap in Mutex so we can mutate inside the async move block
        let energy_sub = std::sync::Mutex::new(energy_sub);
        tokio::spawn(async move {
            loop {
                let mut snap = asmi_core::collect_node_metrics(&hostname, &config, true).await;

                // Sample IOReport for ANE power (no sudo required).
                // If powermetrics returned 0 for ANE (common without sudo),
                // use the IOReport value instead. Also enrich power_source.
                if let Ok(mut sub_guard) = energy_sub.lock() {
                    if let Some(ref mut sub) = *sub_guard {
                        let sample = sub.sample();
                        // IOReport ANE is authoritative — powermetrics needs sudo for ANE
                        if snap.ane_watts == 0.0 && sample.ane_mw > 0.0 {
                            snap.ane_watts = sample.ane_mw;
                        }
                        // Also backfill CPU/GPU if powermetrics failed (rare)
                        if snap.cpu_watts == 0.0 && sample.cpu_mw > 0.0 {
                            snap.cpu_watts = sample.cpu_mw;
                        }
                        if snap.gpu_watts == 0.0 && sample.gpu_mw > 0.0 {
                            snap.gpu_watts = sample.gpu_mw;
                        }
                        snap.power_source = sample.power_source.clone();
                    }
                }

                // Enrich processes with model names from serve managers.
                // ps aux parsing misses models for engines that don't use --model
                // (e.g. mlx_vlm loads via uvicorn, no --model flag on the CLI).
                for proc in &mut snap.processes {
                    if proc.model.is_none() {
                        if let Some(port) = proc.port {
                            if let Some(mgr) = serve_managers.get(&port) {
                                let (state, model) = mgr.model_snapshot().await;
                                if state == asmi_core::ServeState::Ready || state == asmi_core::ServeState::Loading {
                                    if let Some(ref model_path) = model {
                                        proc.model = Some(
                                            model_path.trim_end_matches('/')
                                                .rsplit('/')
                                                .next()
                                                .unwrap_or(model_path)
                                                .to_string()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                tracing::debug!(
                    cpu = format!("{:.1}%", snap.cpu_percent),
                    gpu = format!("{:.1}%", snap.gpu_percent),
                    ane_mw = format!("{:.0}", snap.ane_watts),
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

    // eGPU cache — scan for TinyGPU DriverKit + external GPUs every 30s
    // Gated behind --experimental-egpu flag
    let egpu_cache: Arc<tokio::sync::RwLock<Option<(serde_json::Value, std::time::Instant)>>> =
        Arc::new(tokio::sync::RwLock::new(None));
    if experimental_egpu {
        tracing::info!("experimental eGPU detection enabled");
        let egpu_c = Arc::clone(&egpu_cache);
        let egpu_hostname = hostname.clone();
        tokio::spawn(async move {
            loop {
                let data = daemon::scan_egpu(&egpu_hostname).await;
                let egpu_count = data.get("egpu_count").and_then(|v| v.as_u64()).unwrap_or(0);
                let driver = data.get("tinygpu_driver")
                    .and_then(|d| d.get("installed"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if driver || egpu_count > 0 {
                    tracing::info!(
                        driver = driver,
                        egpu_count = egpu_count,
                        "eGPU scan: detected"
                    );
                }
                *egpu_c.write().await = Some((data, std::time::Instant::now()));
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });
    }

    // Topology cache — runs mlx.distributed_config every 60s (only in cluster hub mode)
    let topology_cache: Arc<tokio::sync::RwLock<Option<(crate::topology::TopologyReport, std::time::Instant)>>> =
        Arc::new(tokio::sync::RwLock::new(None));
    if cluster_hub {
        let topo_cache = Arc::clone(&topology_cache);
        let topo_nodes: Vec<String> = {
            let nm = asmi_core::NodeMap::load();
            nm.nodes.clone()
        };
        if !topo_nodes.is_empty() {
            tokio::spawn(async move {
                loop {
                    let nodes = topo_nodes.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        crate::topology::discover_topology(&nodes, "jaccl")
                    }).await;
                    match result {
                        Ok(Ok(report)) => {
                            tracing::info!(
                                nodes = report.nodes.len(),
                                links = report.links.len(),
                                jaccl_ready = report.jaccl_ready,
                                "topology scan complete"
                            );
                            *topo_cache.write().await = Some((report, std::time::Instant::now()));
                        }
                        Ok(Err(e)) => {
                            tracing::warn!(error = %e, "topology scan failed");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "topology task panicked");
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
            });
        }
    }

    // Build axum router (serve_managers already created above for poll loop enrichment)
    let share_manager = serve::ShareManager::restore().await;

    let peer_heartbeat = Arc::new(serve::PeerHeartbeat::new());

    // Process watchdog — monitors inference processes and detects GPU Lock
    let watchdog_config = watchdog::WatchdogConfig::default();
    let wd = Arc::new(watchdog::Watchdog::new(
        watchdog_config,
        Arc::clone(&snapshot),
        Arc::clone(&peer_heartbeat),
    ));
    // Start watchdog loop in background
    wd.start().await;
    tracing::info!("process watchdog started (5s interval)");

    let app_state = daemon::AppState {
        snapshot,
        cluster_state,
        node_map: Arc::new(tokio::sync::RwLock::new(asmi_core::NodeMap::load())),
        hostname: hostname.clone(),
        started_at,
        metrics_tx: metrics_tx.clone(),
        model_cache,
        thunderbolt_cache,
        topology_cache,
        runtime,
        serve_managers,
        share_manager,
        peer_heartbeat,
        watchdog: wd,
        ane: AneState::new(_experimental_ane),
        egpu_cache,
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
    eprintln!("  GET  /ane              ANE power + status (IOReport)");
    eprintln!("  GET  /egpu             eGPU / TinyGPU detection");
    eprintln!("  GET  /watchdog          Full watchdog report");
    eprintln!("  GET  /watchdog/peers    RDMA peer heartbeat status");
    eprintln!("  GET  /watchdog/gpu-lock GPU Lock detection status");
    let ports_str: Vec<String> = serve::managed_ports().iter().map(|(p, e)| format!("{p}({e})")).collect();
    eprintln!("  Managed ports: {}", ports_str.join(", "));
    if cluster_hub {
        eprintln!("  GET  /cluster          All node snapshots (hub mode)");
        eprintln!("  GET  /nodes            Known node hostnames");
        eprintln!("  GET  /jaccl/config     RDMA topology for JACCL");
        eprintln!("  POST /jaccl/config     Generate JACCL hostfile");
        eprintln!("  GET  /topology         TB5/RDMA mesh topology (JSON)");
        eprintln!("  GET  /topology/dot     Topology as DOT graph");
        eprintln!("  GET  /topology/validate Mesh completeness + JACCL readiness");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
