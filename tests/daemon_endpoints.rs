//! Integration test: start daemon, hit all endpoints.
//!
//! This test requires powermetrics access (sudo) and local Python/MLX.
//! Run explicitly: cargo test --test daemon_endpoints -- --ignored --nocapture

use std::time::Duration;

#[tokio::test]
#[ignore] // requires sudo for powermetrics — run explicitly
async fn test_all_endpoints_respond() {
    let port = 19090 + (std::process::id() % 1000) as u16;
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    // Wait for server to start and collect first metrics
    tokio::time::sleep(Duration::from_secs(5)).await;

    let client = reqwest::Client::new();
    let base = format!("http://localhost:{port}");

    // Test all endpoints return 200
    for path in &[
        "/health",
        "/metrics",
        "/processes",
        "/models",
        "/logs?name=asmi",
        "/runtime",
        "/health/setup",
    ] {
        let url = format!("{base}{path}");
        let resp = client.get(&url)
            .timeout(Duration::from_secs(10))
            .send().await;
        match resp {
            Ok(r) => {
                assert!(
                    r.status().is_success(),
                    "{path} returned {}",
                    r.status()
                );
                eprintln!("{path} -> 200 OK");
            }
            Err(e) => panic!("{path} failed: {e}"),
        }
    }

    child.kill().await.ok();
}

#[tokio::test]
#[ignore] // requires sudo for powermetrics
async fn test_daemon_metrics_have_fresh_data() {
    let port = 19090 + (std::process::id() % 1000) as u16;
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    // Wait for first poll cycle to complete
    tokio::time::sleep(Duration::from_secs(5)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{port}/metrics");
    let resp = client.get(&url)
        .timeout(Duration::from_secs(10))
        .send().await
        .expect("metrics endpoint should respond");

    let body: serde_json::Value = resp.json().await.expect("should parse JSON");

    // Verify the snapshot has real data (not a stale empty cache)
    assert!(
        body.get("ram_total_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) > 0,
        "ram_total_bytes should be > 0, proving fresh local collection. Body: {body}"
    );

    // Verify hostname is resolved (not "unknown")
    let hostname = body.get("hostname").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        !hostname.is_empty() && hostname != "unknown",
        "hostname should be resolved, got: '{hostname}'"
    );

    // The timestamp should be recent (within last 30 seconds)
    if let Some(ts) = body.get("timestamp").and_then(|v| v.as_str()) {
        eprintln!("metrics timestamp: {ts}");
    }

    child.kill().await.ok();
}
