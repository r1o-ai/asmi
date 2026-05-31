use anyhow::Result;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const SOCK_PATH: &str = "/var/run/eu.r1o.asmi.sock";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    info!("Starting asmi-helper daemon as root...");

    let latest_json = Arc::new(RwLock::new(String::from("{}")));

    // --samplers picks which power domains powermetrics emits. Adding
    // ane_power gives us the Neural Engine watts for free — same process,
    // same 1Hz cadence, same parse pass. Without it, parsed.ane_mw is 0
    // and any "ANE check" feature can only report from IOReport (which
    // gives utilization, not watts).
    let mut child = Command::new("powermetrics")
        .arg("-i")
        .arg("1000")
        .arg("--samplers")
        .arg("cpu_power,gpu_power,ane_power")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn powermetrics");

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let mut reader = BufReader::new(stdout).lines();

    let json_clone = latest_json.clone();
    tokio::spawn(async move {
        let mut buffer = String::new();
        while let Ok(Some(line)) = reader.next_line().await {
            buffer.push_str(&line);
            buffer.push('\n');
            
            if line.contains("Combined Power") {
                let parsed = asmi_core::collector::parse_powermetrics_text(&buffer);
                
                let json_val = serde_json::json!({
                    "cpu_mw": parsed.cpu_mw,
                    "gpu_mw": parsed.gpu_mw,
                    "ane_mw": parsed.ane_mw,
                    "cpu_percent": parsed.cpu_percent,
                    "gpu_percent": parsed.gpu_percent,
                    "gpu_frequency_mhz": parsed.gpu_frequency_mhz,
                });
                
                *json_clone.write().await = serde_json::to_string(&json_val).unwrap_or_default();
                buffer.clear();
            }
        }
        warn!("powermetrics process exited!");
    });

    if fs::metadata(SOCK_PATH).is_ok() {
        fs::remove_file(SOCK_PATH).unwrap_or_else(|e| {
            error!(error = %e, "Failed to remove existing socket");
        });
    }

    let listener = UnixListener::bind(SOCK_PATH)?;
    let mut perms = fs::metadata(SOCK_PATH)?.permissions();
    perms.set_mode(0o666);
    fs::set_permissions(SOCK_PATH, perms)?;

    info!(path = SOCK_PATH, "Listening on Unix socket");

    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                let json_arc = latest_json.clone();
                tokio::spawn(async move {
                    let mut json_str = json_arc.read().await.clone();
                    json_str.push('\n');
                    if let Err(e) = stream.write_all(json_str.as_bytes()).await {
                        error!(error = %e, "Failed to write to client socket");
                    }
                });
            }
            Err(e) => {
                error!(error = %e, "Failed to accept connection");
            }
        }
    }
}
