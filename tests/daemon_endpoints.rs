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
