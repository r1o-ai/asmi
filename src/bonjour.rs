//! Bonjour / mDNS publisher for the `_r1o._tcp` service type.
//!
//! Publishes `r1o-<hostname>._r1o._tcp.local` on the asmi port (default 9090) so
//! r1o clients (iOS, desktop) can autodiscover the asmi daemon on the local LAN
//! without any manual config. TXT records advertise the other ports the node is
//! known to expose:
//!
//! ```text
//! Service: r1o-<hostname>._r1o._tcp.local
//! Port:    9090   (asmi — the "front door"; iOS probes /serve/status + /hermes/status to route)
//! TXT:     asmi=9090
//!          hermes=41104?   (only present when http://localhost:41104/health responds)
//!          mlx=19080?      (only present when at least one served slot exists)
//!          host=<hostname>
//!          version=<asmi crate version>
//! ```
//!
//! The publisher re-probes Hermes + MLX on a 30s loop and re-registers the
//! service whenever the TXT records change. Re-registration is cheap with
//! mdns-sd's `register` API (idempotent under the same fullname).
//!
//! Caveat: mDNS does NOT propagate over Tailscale tunnels. Same-LAN discovery
//! works; remote (phone on cellular) must fall back to the Tailscale device
//! list. See plan §"Tailscale caveat" for the routing logic on the iOS side.

use anyhow::{Context, Result};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Service type literal advertised on the LAN. Must match iOS's
/// `NWBrowser(for: .bonjour(type: "_r1o._tcp", ...))` browser type in
/// `ServerDiscoveryService.swift:73`.
const SERVICE_TYPE: &str = "_r1o._tcp.local.";

/// How often to re-probe Hermes + MLX and re-register if TXT changes.
const REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// Default ports we probe for the optional TXT records. These match the
/// canonical r1o port table in `the r1o project docs`.
const HERMES_PORT: u16 = 41104;
const MLX_PORT: u16 = 19080;

/// Snapshot of the TXT-record state. We compare a new probe against the cached
/// snapshot before re-registering to avoid unnecessary mDNS churn.
#[derive(Default, Clone, PartialEq, Eq, Debug)]
struct TxtSnapshot {
    hermes_reachable: bool,
    mlx_reachable: bool,
}

/// Bonjour publisher handle. The daemon is held alive via the inner `Arc` so
/// the background refresh task can keep re-registering on TXT changes.
///
/// `Drop` is intentionally not implemented — when the asmi process exits, the
/// `ServiceDaemon`'s own drop will send a goodbye packet. Adding a manual
/// shutdown here would be redundant.
#[derive(Clone)]
pub struct BonjourPublisher {
    #[allow(dead_code)] // held to keep the daemon alive across the refresh task
    daemon: Arc<ServiceDaemon>,
    #[allow(dead_code)] // held to keep the snapshot lock alive
    snapshot: Arc<RwLock<TxtSnapshot>>,
    #[allow(dead_code)] // held to track instance name (used by refresh task)
    instance_name: String,
}

impl BonjourPublisher {
    /// Register `r1o-<hostname>._r1o._tcp` on `port` and spawn a background
    /// task that re-registers when TXT records change.
    ///
    /// On failure, returns the error to the caller — the daemon HTTP server
    /// continues without mDNS (logged at WARN level).
    pub async fn start(hostname: &str, port: u16) -> Result<Self> {
        let daemon = ServiceDaemon::new().context("create mdns-sd daemon")?;
        let daemon = Arc::new(daemon);

        // Instance name: "r1o-<hostname>". iOS strips `._r1o._tcp.local.` for
        // display per `ServerConnection.swift:91`, leaving "r1o-<hostname>".
        let instance_name = format!("r1o-{}", hostname);

        // Initial TXT snapshot — probe before first registration so first
        // packet reflects reality.
        let initial = probe_optional_ports().await;

        let info = build_service_info(&instance_name, hostname, port, &initial)?;
        daemon.register(info).context("register r1o._tcp service")?;
        info!(
            instance = %instance_name,
            port = port,
            hermes = initial.hermes_reachable,
            mlx = initial.mlx_reachable,
            "Bonjour: registered _r1o._tcp on LAN"
        );

        let snapshot = Arc::new(RwLock::new(initial));

        // Spawn the refresh task. It owns clones of `daemon` + `snapshot` and
        // ends when the task is dropped (i.e. process exit).
        let refresh_daemon = daemon.clone();
        let refresh_snapshot = snapshot.clone();
        let refresh_instance = instance_name.clone();
        let refresh_hostname = hostname.to_string();
        tokio::spawn(async move {
            run_refresh_loop(
                refresh_daemon,
                refresh_snapshot,
                refresh_instance,
                refresh_hostname,
                port,
            )
            .await;
        });

        Ok(Self {
            daemon,
            snapshot,
            instance_name,
        })
    }
}

/// Build a `ServiceInfo` for the current snapshot. Only sets optional TXT
/// records (`hermes=`, `mlx=`) when the corresponding service is reachable.
fn build_service_info(
    instance: &str,
    hostname: &str,
    port: u16,
    snap: &TxtSnapshot,
) -> Result<ServiceInfo> {
    let mut props: Vec<(String, String)> = vec![
        ("asmi".to_string(), port.to_string()),
        ("host".to_string(), hostname.to_string()),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
    ];
    if snap.hermes_reachable {
        props.push(("hermes".to_string(), HERMES_PORT.to_string()));
    }
    if snap.mlx_reachable {
        props.push(("mlx".to_string(), MLX_PORT.to_string()));
    }

    // Publish the A/AAAA record under the UNIQUE service-instance name
    // (`r1o-<hostname>.local.`), NOT the OS's own `<hostname>.local.`. Using the
    // bare `<hostname>.local.` collides with the system's own mDNS record, which
    // makes macOS's mDNSResponder rename the host (LocalHostName hub→hub-2→hub-3)
    // and corrupts cluster topology with phantom nodes. The `host=` TXT record
    // (set above) still carries the canonical hostname for consumers.
    let host_fqdn = format!("{}.local.", instance);
    let info = ServiceInfo::new(
        SERVICE_TYPE,
        instance,
        &host_fqdn,
        "",          // empty `ip` → mdns-sd auto-discovers all local IPs
        port,
        &props[..],
    )
    .context("build ServiceInfo")?
    .enable_addr_auto();
    Ok(info)
}

/// Probe `http://localhost:<hermes_port>/health` and the MLX port to determine
/// which optional TXT records should be set.
async fn probe_optional_ports() -> TxtSnapshot {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
        .expect("reqwest client");
    let hermes = client
        .get(format!("http://localhost:{}/health", HERMES_PORT))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    let mlx = client
        .get(format!("http://localhost:{}/v1/models", MLX_PORT))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    TxtSnapshot {
        hermes_reachable: hermes,
        mlx_reachable: mlx,
    }
}

/// Background loop: every 30s re-probe and re-register if TXT records changed.
async fn run_refresh_loop(
    daemon: Arc<ServiceDaemon>,
    snapshot: Arc<RwLock<TxtSnapshot>>,
    instance_name: String,
    hostname: String,
    port: u16,
) {
    let mut interval = tokio::time::interval(REFRESH_INTERVAL);
    interval.tick().await; // burn the immediate first tick (we just registered)
    loop {
        interval.tick().await;
        let next = probe_optional_ports().await;
        let prev = { snapshot.read().await.clone() };
        if next == prev {
            continue;
        }

        match build_service_info(&instance_name, &hostname, port, &next) {
            Ok(info) => {
                if let Err(e) = daemon.register(info) {
                    warn!(error = %e, "Bonjour: re-register failed");
                    continue;
                }
                info!(
                    hermes = next.hermes_reachable,
                    mlx = next.mlx_reachable,
                    "Bonjour: re-registered with updated TXT"
                );
                *snapshot.write().await = next;
            }
            Err(e) => {
                warn!(error = %e, "Bonjour: rebuild ServiceInfo failed");
            }
        }
    }
}
